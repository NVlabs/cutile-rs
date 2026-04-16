/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA Tile op compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_cuda_tile_op.rs` -- translates annotated
//! op functions into tile-ir operations. Only type and IR-emission changes; the
//! dispatch logic, operand assembly, and attribute handling are identical.

use super::shared_types::Kind::{self, PrimitiveType, StructuredType};
use super::shared_utils::{get_const_hex, AtomicMode};
use super::tile_rust_type::TileRustType;
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::generics::{get_cga_from_type, GenericVars};
use crate::syn_utils::*;
use crate::types::*;

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, TileRustValue, TypeMeta};
use super::utils::*;

use cutile_ir::builder::{append_op, build_block, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, BlockId, Module, Region, Type as TileIrType, Value};

use quote::ToTokens;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Expr, ExprCall, ExprLit, ItemFn, Lit, Token, Type, UnOp};

// ---------------------------------------------------------------------------
// Helpers ported from old CompilerContext utilities
// ---------------------------------------------------------------------------

/// Resolves `static_params` from a `#[cuda_tile::op]` attribute against call-site arguments.
///
/// Each `static_params` entry has the format:
///   `"param_name={TypeA: attr_name=attr_val, TypeB: attr_name=attr_val}"`
///
/// For each entry, this function:
/// 1. Finds the argument index by matching `param_name` to the function signature
/// 2. Reads the concrete ZST type name from the call-site expression (e.g., `ftz::Enabled`)
/// 3. Looks up the type name in the mapping and returns `(attr_name, attr_val)` pairs
///
/// Returns a list of `"attr_name=attr_val"` strings to be emitted as tile-ir named attributes.
fn resolve_static_params(
    static_params: &[String],
    call_expr: &ExprCall,
    fn_item: &ItemFn,
) -> Result<Vec<String>, String> {
    if static_params.is_empty() {
        return Ok(vec![]);
    }
    let fn_params = get_sig_param_names(&fn_item.sig);
    let mut attrs = vec![];

    for spec in static_params {
        // Parse: "param_name={TypeA: attr=val, TypeB: attr=val}"
        let eq_pos = spec
            .find('=')
            .ok_or_else(|| format!("Invalid static_params entry (missing '='): {spec}"))?;
        let param_name = spec[..eq_pos].trim();
        let mapping_str = spec[eq_pos + 1..].trim();

        // Find argument index by param name.
        let Some(arg_idx) = fn_params.iter().position(|s| s == param_name) else {
            return Err(format!(
                "static_params: param '{param_name}' not found in function signature"
            ));
        };
        if arg_idx >= call_expr.args.len() {
            return Err(format!(
                "static_params: param '{param_name}' at index {arg_idx} but only {} args in call",
                call_expr.args.len()
            ));
        }

        // Extract the ZST type name from the call-site argument.
        // Expected forms: `ftz::Enabled`, `Enabled`, `Latency::<3>`
        let arg_expr = &call_expr.args[arg_idx];
        let type_name = match arg_expr {
            Expr::Path(path) => {
                // e.g., `ftz::Enabled` -> "Enabled", or just `Enabled` -> "Enabled"
                path.path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default()
            }
            _ => {
                return Err(format!(
                    "static_params: expected path expression for param '{param_name}', got: {}",
                    arg_expr.to_token_stream()
                ));
            }
        };

        // Look up the type name in the mapping.
        // Format: "{TypeA: attr=val, TypeB: attr=val}"
        if !mapping_str.starts_with('{') || !mapping_str.ends_with('}') {
            return Err(format!(
                "Invalid static_params mapping (expected {{...}}): {mapping_str}"
            ));
        }
        let inner = &mapping_str[1..mapping_str.len() - 1];
        // Split on ',' to get individual type mappings.
        let mut matched = false;
        for entry in inner.split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            // Format: "TypeName: attr_name=attr_val" or "TypeName:" (empty = no attribute)
            let colon_pos = entry
                .find(':')
                .ok_or_else(|| format!("Invalid static_params entry (missing ':'): {entry}"))?;
            let entry_type = entry[..colon_pos].trim();
            let entry_attr = entry[colon_pos + 1..].trim();

            if entry_type == type_name {
                if !entry_attr.is_empty() {
                    attrs.push(entry_attr.to_string());
                }
                matched = true;
                break;
            }
        }
        // Types with no mapping entry (e.g., ftz::Disabled) emit nothing -- that's valid.
        if !matched {
            // No mapping found -- this is the "omit" case (e.g., Disabled).
            // This is not an error.
        }
    }

    Ok(attrs)
}

fn get_signedness_attr(key: &str, element_type_str: &str) -> Result<(String, Attribute), JITError> {
    let signedness_str = match element_type_str {
        "bool" | "u32" | "u64" => "unsigned",
        _ => "signed",
    };
    Ok(signedness_attr(key, signedness_str))
}

fn memory_scope_value(scope: &str) -> Option<i64> {
    match scope {
        "TileBlock" => Some(0),
        "Device" => Some(1),
        "System" => Some(2),
        _ => None,
    }
}

fn extract_optional_zst_type_name(
    expr: &Expr,
    ctx: &CompilerContext,
    param_name: &str,
) -> Result<Option<String>, JITError> {
    match super::shared_utils::resolve_option_arg(expr, ctx) {
        Some(inner) => super::shared_utils::extract_zst_type_name(&inner, param_name).map(Some),
        None => Ok(None),
    }
}

fn extract_latency_cycles(expr: &Expr, generic_args: &GenericVars) -> Result<i32, JITError> {
    let Expr::Path(path_expr) = expr else {
        return JITError::generic(&format!(
            "`latency` must be a Latency<N> unit-struct path, got `{}`",
            expr.to_token_stream()
        ));
    };
    let Some(segment) = path_expr.path.segments.last() else {
        return JITError::generic("`latency` path has no segments");
    };
    if segment.ident != "Latency" {
        return JITError::generic(&format!(
            "`latency` must use `Latency<N>`, got `{}`",
            expr.to_token_stream()
        ));
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return JITError::generic("`latency` must specify a const generic, e.g. `Latency::<4>`");
    };
    let Some(syn::GenericArgument::Const(cycles_expr)) = args.args.first() else {
        return JITError::generic("`latency` must specify a const generic cycle count");
    };
    match cycles_expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(int_lit),
            ..
        }) => int_lit
            .base10_parse::<i32>()
            .map_err(|e| JITError::Generic(format!("invalid latency value: {e}"))),
        Expr::Path(path_expr) => {
            let ident = crate::syn_utils::get_ident_from_path_expr(path_expr);
            generic_args.get_i32(&ident.to_string()).ok_or_else(|| {
                JITError::Generic(format!(
                    "`latency`: const generic `{ident}` has no resolved value"
                ))
            })
        }
        _ => JITError::generic("`latency` const generic must be an integer literal or const param"),
    }
}

impl<'m> CUDATileFunctionCompiler<'m> {
    pub fn compile_cuda_tile_op_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let Expr::Path(path) = &*call_expr.func else {
            return self.jit_error_result(
                &call_expr.func.span(),
                "expected a function path in call expression",
            );
        };
        let last_seg = path.path.segments.last().unwrap();
        let rust_function_name = last_seg.ident.to_string();

        let fn_item = self
            .modules
            .get_function_by_name(rust_function_name.as_str());
        if fn_item.is_none() {
            // If the name was brought into scope by a use statement that
            // points to an unsupported source (stdlib, third-party crate,
            // or an unannotated user module), append a hint pointing back
            // to the import. Falls through to the bare error otherwise.
            let primary = format!("undefined function `{rust_function_name}`");
            let msg = match self.modules.unresolved_name_hint(&rust_function_name) {
                Some(hint) => format!("{primary}: {hint}"),
                None => primary,
            };
            return self.jit_error_result(&call_expr.func.span(), &msg);
        }
        let (_, fn_item) = fn_item.unwrap();

        let op_attrs = match self
            .modules
            .get_cuda_tile_op_attrs(rust_function_name.as_str())
        {
            Some(op_attrs) => op_attrs,
            None => {
                return self.jit_error_result(&call_expr.func.span(), "undefined operation call")
            }
        };
        let op_name = match op_attrs.parse_string("name") {
            Some(name) => name,
            None => {
                return self.jit_error_result(
                    &call_expr.func.span(),
                    &format!("missing operation name for function `{rust_function_name:?}"),
                )
            }
        };

        let cuda_tile_op_params = op_attrs.parse_string_arr("params").unwrap_or_default();
        let cuda_tile_op_attribute_params = op_attrs
            .parse_string_arr("attribute_params")
            .unwrap_or_default();
        let cuda_tile_op_hint_params = op_attrs.parse_string_arr("hint_params").unwrap_or_default();
        let cuda_tile_op_named_attributes = op_attrs
            .parse_string_arr("named_attributes")
            .unwrap_or_default();
        let cuda_tile_op_static_params = op_attrs
            .parse_string_arr("static_params")
            .unwrap_or_default();
        if call_expr.args.len() < cuda_tile_op_params.len() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "expected {} parameter(s) but got {}",
                    cuda_tile_op_params.len(),
                    call_expr.args.len()
                ),
            );
        }

        // Special-case handling for specific ops that need custom compilation.
        if let Some(result) = self.try_compile_cuda_tile_special_op(
            module,
            block_id,
            call_expr,
            fn_item,
            &op_name,
            &cuda_tile_op_hint_params,
            generic_args,
            ctx,
            return_type.clone(),
        )? {
            return Ok(Some(result));
        }

        // General op compilation path.
        self.compile_general_op(
            module,
            block_id,
            call_expr,
            fn_item,
            &op_name,
            &op_attrs,
            &cuda_tile_op_params,
            &cuda_tile_op_attribute_params,
            &cuda_tile_op_hint_params,
            &cuda_tile_op_named_attributes,
            &cuda_tile_op_static_params,
            generic_args,
            ctx,
            return_type,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_compile_cuda_tile_special_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        op_name: &str,
        cuda_tile_op_hint_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        match op_name {
            "cuda_tile.load_ptr_tko" => self.compile_load_ptr_tko(
                module,
                block_id,
                call_expr,
                generic_args,
                ctx,
                return_type,
            ),
            "cuda_tile.store_ptr_tko" => {
                self.compile_store_ptr_tko(module, block_id, call_expr, generic_args, ctx)
            }
            "cuda_tile.atomic_rmw_tko" => self.compile_atomic_rmw_tko(
                module,
                block_id,
                call_expr,
                generic_args,
                ctx,
                return_type,
            ),
            "cuda_tile.atomic_cas_tko" => self.compile_atomic_cas_tko(
                module,
                block_id,
                call_expr,
                generic_args,
                ctx,
                return_type,
            ),
            "load_view_tko" => self.compile_load_view_tko(
                module,
                block_id,
                call_expr,
                fn_item,
                cuda_tile_op_hint_params,
                generic_args,
                ctx,
                return_type,
            ),
            "store_view_tko" => self.compile_store_view_tko(
                module,
                block_id,
                call_expr,
                fn_item,
                cuda_tile_op_hint_params,
                generic_args,
                ctx,
            ),
            "cuda_tile.print_tko" => {
                self.compile_print_tko(module, block_id, call_expr, generic_args, ctx)
            }
            "cuda_tile.get_index_space_shape" => self.compile_shape_query_op(
                module,
                block_id,
                call_expr,
                generic_args,
                ctx,
                return_type,
                Opcode::GetIndexSpaceShape,
            ),
            "cuda_tile.get_tensor_shape" => self.compile_shape_query_op(
                module,
                block_id,
                call_expr,
                generic_args,
                ctx,
                return_type,
                Opcode::GetTensorShape,
            ),
            "cuda_tile.reduce" => {
                self.compile_reduce_op(module, block_id, call_expr, generic_args, ctx, return_type)
            }
            "cuda_tile.scan" => {
                self.compile_scan_op(module, block_id, call_expr, generic_args, ctx)
            }
            _ => Ok(None),
        }
    }

    fn compile_load_ptr_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if return_type.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "unable to infer call; add a return type annotation",
            );
        }
        let return_type_outer = return_type.unwrap();
        let Type::Tuple(tuple_type) = &return_type_outer.rust_ty else {
            return self.jit_error_result(
                &call_expr.span(),
                "expected a tuple return type for `load_ptr_tko",
            );
        };
        let tile_elem_ty = self
            .compile_type(&tuple_type.elems[0], generic_args, &HashMap::new())?
            .unwrap();
        let token_elem_ty = self
            .compile_type(&tuple_type.elems[1], generic_args, &HashMap::new())?
            .unwrap();
        let tile_result_ir_ty = super::_type::convert_type(&tile_elem_ty)
            .expect("failed to convert tile result type for load_ptr_tko");
        let token_result_ir_ty = TileIrType::Token;

        let source_arg = &call_expr.args[0];
        let Some(source_value) =
            self.compile_expression(module, block_id, source_arg, generic_args, ctx, None)?
        else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "unable to compile source argument for `load_ptr_tko",
            );
        };
        let Some(source_ptr) = source_value.value else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "unable to compile source pointer value",
            );
        };

        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&call_expr.args[1], "memory_ordering")?;
        let memory_ordering_value: i64 = match memory_ordering.as_str() { "Weak" => 0, "Relaxed" => 1, "Acquire" => 2, _ => return self.jit_error_result(&call_expr.span(), &format!("invalid `memory_ordering` for `load_ptr_tko: '{}'. Valid: Weak, Relaxed, Acquire", memory_ordering)) };

        let memory_scope = extract_optional_zst_type_name(&call_expr.args[2], ctx, "memory_scope")?;
        let memory_scope_value = match memory_scope.as_deref() {
            Some(scope) => Some(memory_scope_value(scope).ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    &format!("invalid `memory_scope`: '{scope}'. Valid: TileBlock, Device, System"),
                )
            })?),
            None => None,
        };

        let mut operands = vec![source_ptr];
        let mut mask_count: i64 = 0;
        let mut padding_count: i64 = 0;
        let mut token_count: i64 = 0;

        if let Some(mask_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[3], ctx) {
            if let Some(mask_value) =
                self.compile_expression(module, block_id, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }

        if let Some(padding_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[4], ctx)
        {
            if let Some(padding_value) =
                self.compile_expression(module, block_id, &padding_arg, generic_args, ctx, None)?
            {
                if let Some(padding_val) = padding_value.value {
                    let padding_is_scalar = match super::_type::convert_type(&padding_value.ty) {
                        Some(TileIrType::Tile(t)) => t.shape.is_empty(),
                        _ => false,
                    };
                    let result_is_shaped = match &tile_result_ir_ty {
                        TileIrType::Tile(t) => !t.shape.is_empty(),
                        _ => false,
                    };
                    let promoted_padding = if padding_is_scalar && result_is_shaped {
                        let padding_ir_ty = super::_type::convert_type(&padding_value.ty)
                            .expect("failed to convert padding type");
                        let ones_shape_ty = match &padding_ir_ty {
                            TileIrType::Tile(tile_ty) => {
                                TileIrType::Tile(cutile_ir::ir::TileType {
                                    shape: vec![1],
                                    element_type: tile_ty.element_type.clone(),
                                })
                            }
                            _ => padding_ir_ty.clone(),
                        };
                        let (reshape_op_id, reshape_results) =
                            OpBuilder::new(Opcode::Reshape, self.ir_location(&call_expr.span()))
                                .result(ones_shape_ty)
                                .operand(padding_val)
                                .build(module);
                        append_op(module, block_id, reshape_op_id);
                        let reshaped = reshape_results[0];
                        let (broadcast_op_id, broadcast_results) =
                            OpBuilder::new(Opcode::Broadcast, self.ir_location(&call_expr.span()))
                                .result(tile_result_ir_ty.clone())
                                .operand(reshaped)
                                .build(module);
                        append_op(module, block_id, broadcast_op_id);
                        broadcast_results[0]
                    } else {
                        padding_val
                    };
                    operands.push(promoted_padding);
                    padding_count = 1;
                }
            }
        }

        if let Some(token_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[5], ctx) {
            if let Some(token_value) =
                self.compile_expression(module, block_id, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        let mut hint_params: HashMap<String, i32> = HashMap::new();
        let latency = extract_latency_cycles(&call_expr.args[6], generic_args)?;
        if latency > 0 {
            hint_params.insert("latency".to_string(), latency);
        }

        let operand_segments: Vec<i64> = vec![1, mask_count, padding_count, token_count];
        let mut op_builder =
            OpBuilder::new(Opcode::LoadPtrTko, self.ir_location(&call_expr.span()))
                .result(tile_result_ir_ty)
                .result(token_result_ir_ty)
                .operands(operands.iter().copied())
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                );
        if memory_ordering != "Weak" {
            if let Some(memory_scope_value) = memory_scope_value {
                op_builder = op_builder.attr("memory_scope", Attribute::i32(memory_scope_value));
            }
        }
        if let Some(hints_attr) =
            super::optimization_hints::build_load_store_hints(&self.optimization_hints, hint_params)
        {
            op_builder = op_builder.attr("optimization_hints", hints_attr);
        }
        op_builder = op_builder.attr(
            "operandSegmentSizes",
            Attribute::Array(
                operand_segments
                    .iter()
                    .map(|&x| Attribute::i32(x))
                    .collect(),
            ),
        );

        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);
        let values = vec![
            TileRustValue::new_structured_type(results[0], tile_elem_ty, None),
            TileRustValue::new_primitive(results[1], token_elem_ty, None),
        ];
        Ok(Some(TileRustValue::new_compound(values, return_type_outer)))
    }

    fn compile_store_ptr_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let token_result_ir_ty = TileIrType::Token;
        let Some(dest_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[0],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "unable to compile destination argument for `store_ptr_tko",
            );
        };
        let Some(dest_ptr) = dest_value.value else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "unable to compile destination pointer value",
            );
        };
        let Some(value_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[1],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "unable to compile source argument for `store_ptr_tko",
            );
        };
        let Some(tile_value) = value_value.value else {
            return self
                .jit_error_result(&call_expr.args[1].span(), "unable to compile tile value");
        };

        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&call_expr.args[2], "memory_ordering")?;
        let memory_ordering_value: i64 = match memory_ordering.as_str() { "Weak" => 0, "Relaxed" => 1, "Release" => 3, _ => return self.jit_error_result(&call_expr.span(), &format!("invalid `memory_ordering` for `store_ptr_tko: '{}'. Valid: Weak, Relaxed, Release", memory_ordering)) };

        let memory_scope = extract_optional_zst_type_name(&call_expr.args[3], ctx, "memory_scope")?;
        let memory_scope_value = match memory_scope.as_deref() {
            Some(scope) => Some(memory_scope_value(scope).ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    &format!("invalid `memory_scope`: '{scope}'. Valid: TileBlock, Device, System"),
                )
            })?),
            None => None,
        };

        let mut operands = vec![dest_ptr, tile_value];
        let mut mask_count: i64 = 0;
        let mut token_count: i64 = 0;
        if let Some(mask_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[4], ctx) {
            if let Some(mask_value) =
                self.compile_expression(module, block_id, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }
        if let Some(token_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[5], ctx) {
            if let Some(token_value) =
                self.compile_expression(module, block_id, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }
        let mut hint_params: HashMap<String, i32> = HashMap::new();
        let latency = extract_latency_cycles(&call_expr.args[6], generic_args)?;
        if latency > 0 {
            hint_params.insert("latency".to_string(), latency);
        }

        let operand_segments: Vec<i64> = vec![1, 1, mask_count, token_count];
        let mut op_builder =
            OpBuilder::new(Opcode::StorePtrTko, self.ir_location(&call_expr.span()))
                .result(token_result_ir_ty)
                .operands(operands.iter().copied())
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                );
        if memory_ordering != "Weak" {
            if let Some(memory_scope_value) = memory_scope_value {
                op_builder = op_builder.attr("memory_scope", Attribute::i32(memory_scope_value));
            }
        }
        if let Some(hints_attr) =
            super::optimization_hints::build_load_store_hints(&self.optimization_hints, hint_params)
        {
            op_builder = op_builder.attr("optimization_hints", hints_attr);
        }
        op_builder = op_builder.attr(
            "operandSegmentSizes",
            Attribute::Array(
                operand_segments
                    .iter()
                    .map(|&x| Attribute::i32(x))
                    .collect(),
            ),
        );

        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);
        let token_type = self
            .compile_type(&syn::parse_quote!(Token), generic_args, &HashMap::new())?
            .unwrap();
        Ok(Some(TileRustValue::new_primitive(
            results[0], token_type, None,
        )))
    }

    fn compile_atomic_rmw_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if return_type.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "unable to infer call; add a return type annotation",
            );
        }
        let return_type_outer = return_type.unwrap();
        let Type::Tuple(tuple_type) = &return_type_outer.rust_ty else {
            return self.jit_error_result(
                &call_expr.span(),
                "expected a tuple return type for `atomic_rmw_tko",
            );
        };
        let tile_elem_ty = self
            .compile_type(&tuple_type.elems[0], generic_args, &HashMap::new())?
            .unwrap();
        let token_elem_ty = self
            .compile_type(&tuple_type.elems[1], generic_args, &HashMap::new())?
            .unwrap();
        let tile_result_ir_ty = super::_type::convert_type(&tile_elem_ty)
            .expect("failed to convert tile result type for atomic_rmw_tko");
        let token_result_ir_ty = TileIrType::Token;

        let Some(ptr_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[0],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "unable to compile pointer argument for `atomic_rmw_tko",
            );
        };
        let Some(ptrs) = ptr_value.value else {
            return self
                .jit_error_result(&call_expr.args[0].span(), "unable to compile pointer value");
        };
        let Some(arg_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[1],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "unable to compile argument for `atomic_rmw_tko",
            );
        };
        let Some(arg) = arg_value.value else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "unable to compile argument value",
            );
        };

        let mode = super::shared_utils::extract_zst_type_name(&call_expr.args[2], "mode")?;
        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&call_expr.args[3], "memory_ordering")?;
        let memory_scope =
            super::shared_utils::extract_zst_type_name(&call_expr.args[4], "memory_scope")?;

        let memory_ordering_value: i64 = match memory_ordering.as_str() { "Relaxed" => 1, "Acquire" => 2, "Release" => 3, "AcqRel" => 4, "Weak" => return self.jit_error_result(&call_expr.span(), "atomic_rmw_tko does not support `Weak` memory ordering. Valid: Relaxed, Acquire, Release, AcqRel"), _ => return self.jit_error_result(&call_expr.span(), &format!("invalid `memory_ordering` for `atomic_rmw_tko: '{}'. Valid: Relaxed, Acquire, Release, AcqRel", memory_ordering)) };
        let memory_scope_value: i64 = match memory_scope.as_str() {
            "TileBlock" => 0,
            "Device" => 1,
            "System" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: `'{}'. Valid: TileBlock, Device, System",
                        memory_scope
                    ),
                )
            }
        };

        let elem_ty_prefix = ptr_value
            .ty
            .get_cuda_tile_element_type_prefix(self.modules.primitives())?;
        let atomic_mode = AtomicMode::new(mode.as_str(), elem_ty_prefix)? as i64;

        let mut operands = vec![ptrs, arg];
        let mut mask_count: i64 = 0;
        let mut token_count: i64 = 0;
        if let Some(mask_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[5], ctx) {
            if let Some(mask_value) =
                self.compile_expression(module, block_id, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }
        if let Some(token_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[6], ctx) {
            if let Some(token_value) =
                self.compile_expression(module, block_id, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        let operand_segments: Vec<i64> = vec![1, 1, mask_count, token_count];
        let (op_id, results) =
            OpBuilder::new(Opcode::AtomicRMW, self.ir_location(&call_expr.span()))
                .result(tile_result_ir_ty)
                .result(token_result_ir_ty)
                .operands(operands.iter().copied())
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                )
                .attr("memory_scope", Attribute::i32(memory_scope_value))
                .attr("mode", Attribute::i32(atomic_mode))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(
                        operand_segments
                            .iter()
                            .map(|&x| Attribute::i32(x))
                            .collect(),
                    ),
                )
                .build(module);
        append_op(module, block_id, op_id);
        let values = vec![
            TileRustValue::new_structured_type(results[0], tile_elem_ty, None),
            TileRustValue::new_primitive(results[1], token_elem_ty, None),
        ];
        Ok(Some(TileRustValue::new_compound(values, return_type_outer)))
    }

    fn compile_atomic_cas_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if return_type.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "unable to infer call; add a return type annotation",
            );
        }
        let return_type_outer = return_type.unwrap();
        let Type::Tuple(tuple_type) = &return_type_outer.rust_ty else {
            return self.jit_error_result(
                &call_expr.span(),
                "expected a tuple return type for `atomic_cas_tko",
            );
        };
        let tile_elem_ty = self
            .compile_type(&tuple_type.elems[0], generic_args, &HashMap::new())?
            .unwrap();
        let token_elem_ty = self
            .compile_type(&tuple_type.elems[1], generic_args, &HashMap::new())?
            .unwrap();
        let tile_result_ir_ty = super::_type::convert_type(&tile_elem_ty)
            .expect("failed to convert tile result type for atomic_cas_tko");
        let token_result_ir_ty = TileIrType::Token;

        let Some(ptr_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[0],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "unable to compile pointer argument for `atomic_cas_tko",
            );
        };
        let Some(ptrs) = ptr_value.value else {
            return self
                .jit_error_result(&call_expr.args[0].span(), "unable to compile pointer value");
        };
        let Some(cmp_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[1],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "unable to compile comparison argument for `atomic_cas_tko",
            );
        };
        let Some(cmp) = cmp_value.value else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "unable to compile comparison value",
            );
        };
        let Some(val_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[2],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[2].span(),
                "unable to compile value argument for `atomic_cas_tko",
            );
        };
        let Some(val) = val_value.value else {
            return self.jit_error_result(&call_expr.args[2].span(), "unable to compile value");
        };

        let memory_ordering =
            super::shared_utils::extract_zst_type_name(&call_expr.args[3], "memory_ordering")?;
        let memory_scope =
            super::shared_utils::extract_zst_type_name(&call_expr.args[4], "memory_scope")?;
        let memory_ordering_value: i64 = match memory_ordering.as_str() { "Relaxed" => 1, "Acquire" => 2, "Release" => 3, "AcqRel" => 4, "Weak" => return self.jit_error_result(&call_expr.span(), "atomic_cas_tko does not support `Weak` memory ordering. Valid: Relaxed, Acquire, Release, AcqRel"), _ => return self.jit_error_result(&call_expr.span(), &format!("invalid `memory_ordering` for `atomic_cas_tko: '{}'. Valid: Relaxed, Acquire, Release, AcqRel", memory_ordering)) };
        let memory_scope_value: i64 = match memory_scope.as_str() {
            "TileBlock" => 0,
            "Device" => 1,
            "System" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: `'{}'. Valid: TileBlock, Device, System",
                        memory_scope
                    ),
                )
            }
        };

        let mut operands = vec![ptrs, cmp, val];
        let mut mask_count: i64 = 0;
        let mut token_count: i64 = 0;
        if let Some(mask_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[5], ctx) {
            if let Some(mask_value) =
                self.compile_expression(module, block_id, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }
        if let Some(token_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[6], ctx) {
            if let Some(token_value) =
                self.compile_expression(module, block_id, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        let operand_segments: Vec<i64> = vec![1, 1, 1, mask_count, token_count];
        let (op_id, results) =
            OpBuilder::new(Opcode::AtomicCAS, self.ir_location(&call_expr.span()))
                .result(tile_result_ir_ty)
                .result(token_result_ir_ty)
                .operands(operands.iter().copied())
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                )
                .attr("memory_scope", Attribute::i32(memory_scope_value))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(
                        operand_segments
                            .iter()
                            .map(|&x| Attribute::i32(x))
                            .collect(),
                    ),
                )
                .build(module);
        append_op(module, block_id, op_id);
        let values = vec![
            TileRustValue::new_structured_type(results[0], tile_elem_ty, None),
            TileRustValue::new_primitive(results[1], token_elem_ty, None),
        ];
        Ok(Some(TileRustValue::new_compound(values, return_type_outer)))
    }

    /// Reads `memory_ordering`, `memory_scope` ZST args from a view op call.
    /// Returns `(ordering_i32, scope_i32, ordering_ident)`. `is_load = true`
    /// restricts ordering to `Weak/Relaxed/Acquire`; `false` restricts to
    /// `Weak/Relaxed/Release`.
    fn read_view_ordering_scope(
        &self,
        fn_params: &[String],
        call_expr: &ExprCall,
        op_name: &str,
        is_load: bool,
    ) -> Result<(i64, i64, String), JITError> {
        let ord_idx = fn_params
            .iter()
            .position(|s| s == "memory_ordering")
            .expect("view op missing `memory_ordering` parameter");
        let scope_idx = fn_params
            .iter()
            .position(|s| s == "memory_scope")
            .expect("view op missing `memory_scope` parameter");

        let memory_ordering = super::shared_utils::extract_zst_type_name(
            &call_expr.args[ord_idx],
            "memory_ordering",
        )?;
        let memory_ordering_value: i64 = if is_load {
            match memory_ordering.as_str() {
                "Weak" => 0,
                "Relaxed" => 1,
                "Acquire" => 2,
                _ => {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "invalid `memory_ordering` for `{op_name}: '{memory_ordering}'. Valid: Weak, Relaxed, Acquire"
                        ),
                    )
                }
            }
        } else {
            match memory_ordering.as_str() {
                "Weak" => 0,
                "Relaxed" => 1,
                "Release" => 3,
                _ => {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "invalid `memory_ordering` for `{op_name}: '{memory_ordering}'. Valid: Weak, Relaxed, Release"
                        ),
                    )
                }
            }
        };

        let memory_scope =
            super::shared_utils::extract_zst_type_name(&call_expr.args[scope_idx], "memory_scope")?;
        let memory_scope_value: i64 = match memory_scope.as_str() {
            "TileBlock" => 0,
            "Device" => 1,
            "System" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: '{memory_scope}'. Valid: TileBlock, Device, System"
                    ),
                )
            }
        };
        Ok((memory_ordering_value, memory_scope_value, memory_ordering))
    }

    /// Reads `latency: Option<i32>` and `tma: T` ZST args from a view op call
    /// and converts them to optimization-hint key/value pairs.
    fn read_view_hint_params(
        &self,
        fn_params: &[String],
        call_expr: &ExprCall,
        ctx: &CompilerContext,
        generic_args: &GenericVars,
    ) -> Result<HashMap<String, i32>, JITError> {
        let mut hint_params: HashMap<String, i32> = HashMap::new();

        if let Some(i) = fn_params.iter().position(|s| s == "latency") {
            if let Some(inner) = super::shared_utils::resolve_option_arg(&call_expr.args[i], ctx) {
                let val: i32 = match &inner {
                    Expr::Lit(lit_expr) => {
                        let Lit::Int(int_lit) = &lit_expr.lit else {
                            return self.jit_error_result(
                                &lit_expr.span(),
                                "non-integer literal for `latency`",
                            );
                        };
                        int_lit.base10_parse::<i32>().unwrap()
                    }
                    Expr::Path(path_expr) => {
                        let ident = crate::syn_utils::get_ident_from_path_expr(path_expr);
                        generic_args.get_i32(&ident.to_string()).ok_or_else(|| {
                            self.jit_error(
                                &call_expr.args[i].span(),
                                &format!(
                                    "`latency`: const generic `{ident}` has no resolved value"
                                ),
                            )
                        })?
                    }
                    _ => {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            "`latency` must be a literal or const generic",
                        );
                    }
                };
                hint_params.insert("latency".to_string(), val);
            }
        }

        if let Some(i) = fn_params.iter().position(|s| s == "tma") {
            let ident = super::shared_utils::extract_zst_type_name(&call_expr.args[i], "tma")?;
            match ident.as_str() {
                "Enabled" => {} // no hint
                "Disabled" => {
                    hint_params.insert("allow_tma".to_string(), 0);
                }
                _ => {
                    return self.jit_error_result(
                        &call_expr.args[i].span(),
                        &format!("invalid `tma`: '{ident}'. Valid: Enabled, Disabled"),
                    )
                }
            }
        }

        Ok(hint_params)
    }

    fn compile_load_view_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        _cuda_tile_op_hint_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if return_type.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "unable to infer call; add a return type annotation",
            );
        }
        let return_type = return_type.unwrap();
        if return_type.tile_ir_ty.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "Expected cuda_tile_ty for load_view_tko return type",
            );
        }
        let tile_result_ir_ty = super::_type::convert_type(&return_type)
            .expect("failed to convert load_view_tko result type");
        let token_result_ir_ty = TileIrType::Token;

        let Some(view_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[0],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(&call_expr.args[0].span(), "Unable to compile view");
        };
        let Some(cuda_tile_view_value) = view_value.value else {
            return self.jit_error_result(&call_expr.args[0].span(), "Unable to compile view");
        };
        let Some(type_meta) = view_value.type_meta else {
            return self
                .jit_error_result(&call_expr.args[0].span(), "Expected some TypeMeta for view");
        };
        let Some(token_value) = type_meta.fields.get("token") else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "Expected token value in TypeMeta for view",
            );
        };
        let Some(cuda_tile_token) = token_value.value else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "Expected token value in TypeMeta for view",
            );
        };

        let index_arg = &call_expr.args[1];
        let index_arg_str = index_arg.to_token_stream().to_string();
        let index_value = self
            .compile_expression(module, block_id, index_arg, generic_args, ctx, None)?
            .unwrap();
        if index_value.values.is_none() {
            return self.jit_error_result(&call_expr.args[1].span(), "Expected values for index");
        }
        let mut index_values_vec = Vec::new();
        for value in index_value.values.as_ref().unwrap().iter() {
            let Some(v) = value.value else {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    &format!("Unexpected nested array {index_arg_str}"),
                );
            };
            index_values_vec.push(v);
        }
        let index_values = index_values_vec;

        let fn_params = get_sig_param_names(&fn_item.sig);
        let (memory_ordering_value, memory_scope_value, memory_ordering) =
            self.read_view_ordering_scope(&fn_params, call_expr, "load_view_tko", true)?;
        let hint_params = self.read_view_hint_params(&fn_params, call_expr, ctx, generic_args)?;
        let mut opt_hint_attrs: Vec<(String, Attribute)> = vec![];
        if let Some(load_store_hints_attr) =
            super::optimization_hints::build_load_store_hints(&self.optimization_hints, hint_params)
        {
            opt_hint_attrs.push(("optimization_hints".to_string(), load_store_hints_attr));
        }

        let mut all_operands = vec![cuda_tile_view_value];
        all_operands.extend_from_slice(&index_values);
        all_operands.push(cuda_tile_token);
        let operand_segments: Vec<i64> = vec![1, index_values.len() as i64, 1];

        let mut op_builder =
            OpBuilder::new(Opcode::LoadViewTko, self.ir_location(&call_expr.span()))
                .result(tile_result_ir_ty)
                .result(token_result_ir_ty)
                .operands(all_operands.iter().copied())
                .attrs(opt_hint_attrs)
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                );
        if memory_ordering != "Weak" {
            op_builder = op_builder.attr("memory_scope", Attribute::i32(memory_scope_value));
        }
        op_builder = op_builder.attr(
            "operandSegmentSizes",
            Attribute::Array(
                operand_segments
                    .iter()
                    .map(|&x| Attribute::i32(x))
                    .collect(),
            ),
        );
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);
        let _old = super::shared_utils::update_token(&call_expr.args[0], results[1], ctx);
        Ok(Some(TileRustValue::new_structured_type(
            results[0],
            return_type,
            None,
        )))
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_store_view_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        _cuda_tile_op_hint_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let token_result_ir_ty = TileIrType::Token;
        let view_arg = &call_expr.args[0];
        let Some(mut view_value) =
            self.compile_expression(module, block_id, view_arg, generic_args, ctx, None)?
        else {
            return self.jit_error_result(&call_expr.args[0].span(), "Unable to compile view");
        };
        let Some(cuda_tile_view_value) = &mut view_value.value else {
            return self.jit_error_result(&call_expr.args[0].span(), "Unable to compile view");
        };
        let Some(type_meta) = &mut view_value.type_meta else {
            return self
                .jit_error_result(&call_expr.args[0].span(), "Expected some TypeMeta for view");
        };
        let Some(token_value) = type_meta.fields.get("token") else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "Expected token value in TypeMeta for view",
            );
        };
        let Some(cuda_tile_token) = &token_value.value else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "Expected token value in TypeMeta for view",
            );
        };

        let tile_value = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[1],
                generic_args,
                ctx,
                None,
            )?
            .unwrap();
        if tile_value.value.is_none() {
            return self.jit_error_result(
                &call_expr.args[2].span(),
                "Expected value for tile in store_view_tko",
            );
        }
        let tile_value_val = tile_value.value.unwrap();

        let index_arg = &call_expr.args[2];
        let index_arg_str = index_arg.to_token_stream().to_string();
        let index_value = self
            .compile_expression(module, block_id, index_arg, generic_args, ctx, None)?
            .unwrap();
        if index_value.values.is_none() {
            return self.jit_error_result(&call_expr.args[2].span(), "Expected values for index");
        }
        let mut index_values_vec = Vec::new();
        for value in index_value.values.as_ref().unwrap().iter() {
            let Some(v) = value.value else {
                return self.jit_error_result(
                    &call_expr.args[2].span(),
                    &format!("Unexpected nested array {index_arg_str}"),
                );
            };
            index_values_vec.push(v);
        }
        let index_values = index_values_vec;

        let fn_params = get_sig_param_names(&fn_item.sig);
        let (memory_ordering_value, memory_scope_value, memory_ordering) =
            self.read_view_ordering_scope(&fn_params, call_expr, "store_view_tko", false)?;
        let hint_params = self.read_view_hint_params(&fn_params, call_expr, ctx, generic_args)?;
        let mut opt_hint_attrs: Vec<(String, Attribute)> = vec![];
        if let Some(load_store_hints_attr) =
            super::optimization_hints::build_load_store_hints(&self.optimization_hints, hint_params)
        {
            opt_hint_attrs.push(("optimization_hints".to_string(), load_store_hints_attr));
        }

        let cuda_tile_view_val = *cuda_tile_view_value;
        let cuda_tile_token_val = *cuda_tile_token;
        let mut all_operands = vec![tile_value_val, cuda_tile_view_val];
        all_operands.extend_from_slice(&index_values);
        all_operands.push(cuda_tile_token_val);
        let operand_segments: Vec<i64> = vec![1, 1, index_values.len() as i64, 1];

        let mut op_builder =
            OpBuilder::new(Opcode::StoreViewTko, self.ir_location(&call_expr.span()))
                .result(token_result_ir_ty)
                .operands(all_operands.iter().copied())
                .attrs(opt_hint_attrs)
                .attr(
                    "memory_ordering_semantics",
                    Attribute::i32(memory_ordering_value),
                );
        if memory_ordering != "Weak" {
            op_builder = op_builder.attr("memory_scope", Attribute::i32(memory_scope_value));
        }
        op_builder = op_builder.attr(
            "operandSegmentSizes",
            Attribute::Array(
                operand_segments
                    .iter()
                    .map(|&x| Attribute::i32(x))
                    .collect(),
            ),
        );
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);
        let _old = super::shared_utils::update_token(view_arg, results[0], ctx);
        let Some(var_arg_ident) = get_ident_from_expr(view_arg) else {
            return self.jit_error_result(&view_arg.span(), "Unexpected expression");
        };
        let Some(result) = ctx.vars.get(var_arg_ident.to_string().as_str()) else {
            return self.jit_error_result(
                &view_arg.span(),
                &format!("Unexpected state: Expected {var_arg_ident} in ctx"),
            );
        };
        Ok(Some(result.clone()))
    }

    fn compile_print_tko(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        if call_expr.args.len() != 3 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`print_tko` expects 3 arguments, got {}",
                    call_expr.args.len()
                ),
            );
        }
        let str_literal =
            super::shared_utils::extract_string_literal(&call_expr.args[0], "str", ctx)?;
        let Some(arg_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[1],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(&call_expr.args[1].span(), "failed to compile print arg");
        };
        let Some(arg) = arg_value.value else {
            return self.jit_error_result(
                &call_expr.args[1].span(),
                "`print_tko` argument must be a scalar or tile value",
            );
        };

        let mut operands = vec![arg];
        let mut token_count = 0;
        if let Some(token_arg) = super::shared_utils::resolve_option_arg(&call_expr.args[2], ctx) {
            if let Some(token_value) =
                self.compile_expression(module, block_id, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        let token_type = self
            .compile_type(&syn::parse_quote!(Token), generic_args, &HashMap::new())?
            .unwrap();
        let (op_id, results) = OpBuilder::new(Opcode::Print, self.ir_location(&call_expr.span()))
            .attr("str", Attribute::String(str_literal))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![Attribute::i32(1), Attribute::i32(token_count)]),
            )
            .operands(operands.iter().copied())
            .result(TileIrType::Token)
            .build(module);
        append_op(module, block_id, op_id);
        Ok(Some(TileRustValue::new_primitive(
            results[0], token_type, None,
        )))
    }

    fn compile_shape_query_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
        opcode: Opcode,
    ) -> Result<Option<TileRustValue>, JITError> {
        if call_expr.args.len() != 1 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "shape query op expects 1 argument, got {}",
                    call_expr.args.len()
                ),
            );
        }
        let Some(src_value) = self.compile_expression(
            module,
            block_id,
            &call_expr.args[0],
            generic_args,
            ctx,
            None,
        )?
        else {
            return self.jit_error_result(
                &call_expr.args[0].span(),
                "failed to compile shape query source",
            );
        };
        let Some(src) = src_value.value else {
            return self
                .jit_error_result(&call_expr.args[0].span(), "shape query source has no value");
        };
        let src_ty = module.value_type(src).clone();
        let rank = match (&opcode, &src_ty) {
            (Opcode::GetIndexSpaceShape, TileIrType::PartitionView(pv)) => pv.tile_shape.len(),
            (Opcode::GetTensorShape, TileIrType::TensorView(tv)) => tv.shape.len(),
            (Opcode::GetIndexSpaceShape, other) => {
                return self.jit_error_result(
                    &call_expr.args[0].span(),
                    &format!("get_index_space_shape expects a partition view, got {other:?}"),
                )
            }
            (Opcode::GetTensorShape, other) => {
                return self.jit_error_result(
                    &call_expr.args[0].span(),
                    &format!("get_tensor_shape expects a tensor view, got {other:?}"),
                )
            }
            _ => unreachable!(),
        };

        let scalar_result_ty = TileIrType::Tile(cutile_ir::ir::TileType {
            shape: vec![],
            element_type: cutile_ir::ir::TileElementType::Scalar(cutile_ir::ir::ScalarType::I32),
        });
        let mut op_builder =
            OpBuilder::new(opcode, self.ir_location(&call_expr.span())).operand(src);
        for _ in 0..rank {
            op_builder = op_builder.result(scalar_result_ty.clone());
        }
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);

        let return_type = return_type.ok_or_else(|| {
            self.jit_error(
                &call_expr.span(),
                "shape query return type could not be inferred",
            )
        })?;
        let elem_type = self
            .compile_type(&syn::parse_quote!(i32), generic_args, &HashMap::new())?
            .unwrap();
        let values = results
            .into_iter()
            .map(|value| TileRustValue::new_primitive(value, elem_type.clone(), None))
            .collect();
        Ok(Some(TileRustValue::new_compound(values, return_type)))
    }

    fn compile_reduce_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        _return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let operand_value = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[0],
                generic_args,
                ctx,
                None,
            )?
            .unwrap();
        let elem_ty_str = operand_value
            .ty
            .get_cuda_tile_element_type(self.modules.primitives())?
            .unwrap();
        let elem_ir_ty = super::_type::make_scalar_tile_type(&elem_ty_str)
            .expect("failed to build scalar tile type for reduce element");
        let elem_rust_ty = operand_value
            .ty
            .type_instance
            .get_rust_element_instance_ty()
            .unwrap();
        let elem_rust_ty_parsed = syn::parse2::<Type>(elem_rust_ty.parse().unwrap()).unwrap();
        let elem_compiled_ty = self
            .compile_type(&elem_rust_ty_parsed, generic_args, &HashMap::new())?
            .unwrap();
        let return_type_inner =
            match super::tile_rust_type::TileRustType::from_scalar_tile(&elem_rust_ty) {
                Some(t) => t,
                None => {
                    let ty =
                        syn::parse_str::<syn::Type>(&format!("Tile<{}, {{[]}}>", elem_rust_ty))
                            .unwrap();
                    self.compile_type(&ty, generic_args, &HashMap::new())?
                        .unwrap()
                }
            };
        let result_ir_ty = elem_ir_ty.clone();
        let operand_tile = operand_value.value.unwrap();

        let (reduce_block_id, reduce_block_args) =
            build_block(module, &[elem_ir_ty.clone(), elem_ir_ty.clone()]);
        let arg0 = reduce_block_args[0];
        let arg1 = reduce_block_args[1];
        let has_closure = call_expr.args.len() >= 4 && is_closure(&call_expr.args[3]);

        let reduction_result: Value = if has_closure {
            let Expr::Closure(closure_expr) = &call_expr.args[3] else {
                unreachable!()
            };
            let closure_info = parse_closure(closure_expr);
            if closure_info.params.len() != 2 {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    &format!(
                        "reduce closure must have 2 parameters: |acc, x| ..., got {}",
                        closure_info.params.len()
                    ),
                );
            }
            let mut closure_variables = ctx.clone();
            closure_variables.vars.insert(
                closure_info.params[0].name.clone(),
                TileRustValue::new_value_kind_like(arg0, elem_compiled_ty.clone()),
            );
            closure_variables.vars.insert(
                closure_info.params[1].name.clone(),
                TileRustValue::new_value_kind_like(arg1, elem_compiled_ty.clone()),
            );
            let result_value = self
                .compile_expression(
                    module,
                    reduce_block_id,
                    &closure_info.body,
                    generic_args,
                    &mut closure_variables,
                    Some(elem_compiled_ty.clone()),
                )
                .unwrap_or(None);
            if result_value.is_none() {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    "Closure body must return a value",
                );
            }
            let result_value = result_value.unwrap();
            if result_value.value.is_none() {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    "Closure body must return a value with an IR value",
                );
            }
            result_value.value.unwrap()
        } else {
            let is_float =
                super::_type::scalar_from_name(&elem_ty_str).is_some_and(|s| s.is_float());
            let add_opcode = if is_float { Opcode::AddF } else { Opcode::AddI };
            let mut add_op_builder =
                OpBuilder::new(add_opcode, self.ir_location(&call_expr.span()))
                    .result(elem_ir_ty.clone())
                    .operand(arg0)
                    .operand(arg1);
            if is_float {
                let (rn, rv) = rounding_mode_attr("nearest_even");
                add_op_builder = add_op_builder.attr(rn, rv);
            } else {
                add_op_builder = add_op_builder.attr("overflow", Attribute::i32(0));
            }
            let (add_op_id, add_results) = add_op_builder.build(module);
            append_op(module, reduce_block_id, add_op_id);
            add_results[0]
        };

        let (yield_op_id, _) = OpBuilder::new(Opcode::Yield, self.ir_location(&call_expr.span()))
            .operand(reduction_result)
            .build(module);
        append_op(module, reduce_block_id, yield_op_id);
        let region_id = module.alloc_region(Region {
            blocks: vec![reduce_block_id],
        });
        let elem_scalar =
            super::_type::scalar_from_name(&elem_ty_str).unwrap_or(cutile_ir::ir::ScalarType::I32);
        let identities_attr = if elem_scalar.is_float() {
            Attribute::Array(vec![Attribute::Float(
                0.0,
                cutile_ir::ir::Type::Scalar(elem_scalar),
            )])
        } else {
            Attribute::Array(vec![Attribute::Integer(
                0,
                cutile_ir::ir::Type::Scalar(elem_scalar),
            )])
        };

        let (op_id, results) = OpBuilder::new(Opcode::Reduce, self.ir_location(&call_expr.span()))
            .result(result_ir_ty)
            .operand(operand_tile)
            .region(region_id)
            .attr("dim", Attribute::i32(0))
            .attr("identities", identities_attr)
            .build(module);
        append_op(module, block_id, op_id);
        Ok(Some(TileRustValue::new_structured_type(
            results[0],
            return_type_inner,
            None,
        )))
    }

    fn compile_scan_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let operand_value = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[0],
                generic_args,
                ctx,
                None,
            )?
            .unwrap();
        let operand_tile = operand_value.value.unwrap();
        let return_type = operand_value.ty.clone();
        let result_ir_ty = super::_type::convert_type(&operand_value.ty)
            .expect("failed to convert scan result type");
        let elem_ty_str = operand_value
            .ty
            .get_cuda_tile_element_type(self.modules.primitives())?
            .unwrap();
        let elem_ir_ty = super::_type::make_scalar_tile_type(&elem_ty_str)
            .expect("failed to build scalar tile type for scan element");
        let elem_rust_ty = operand_value
            .ty
            .type_instance
            .get_rust_element_instance_ty()
            .unwrap();
        let elem_rust_ty = syn::parse2::<Type>(elem_rust_ty.parse().unwrap()).unwrap();
        let elem_compiled_ty = self
            .compile_type(&elem_rust_ty, generic_args, &HashMap::new())?
            .unwrap();

        let (scan_block_id, scan_block_args) =
            build_block(module, &[elem_ir_ty.clone(), elem_ir_ty.clone()]);
        let arg0 = scan_block_args[0];
        let arg1 = scan_block_args[1];
        let has_closure = call_expr.args.len() >= 5 && is_closure(&call_expr.args[4]);

        let scan_result: Value = if has_closure {
            let Expr::Closure(closure_expr) = &call_expr.args[4] else {
                unreachable!()
            };
            let closure_info = parse_closure(closure_expr);
            if closure_info.params.len() != 2 {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    &format!(
                        "scan closure must have 2 parameters: |acc, x| ..., got {}",
                        closure_info.params.len()
                    ),
                );
            }
            let mut closure_variables = ctx.clone();
            closure_variables.vars.insert(
                closure_info.params[0].name.clone(),
                TileRustValue::new_value_kind_like(arg0, elem_compiled_ty.clone()),
            );
            closure_variables.vars.insert(
                closure_info.params[1].name.clone(),
                TileRustValue::new_value_kind_like(arg1, elem_compiled_ty.clone()),
            );
            let result_value = self
                .compile_expression(
                    module,
                    scan_block_id,
                    &closure_info.body,
                    generic_args,
                    &mut closure_variables,
                    Some(elem_compiled_ty.clone()),
                )
                .unwrap_or(None);
            if result_value.is_none() {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    "Closure body must return a value",
                );
            }
            let result_value = result_value.unwrap();
            if result_value.value.is_none() {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    "Closure body must return a value with an IR value",
                );
            }
            result_value.value.unwrap()
        } else {
            let is_float =
                super::_type::scalar_from_name(&elem_ty_str).is_some_and(|s| s.is_float());
            let add_opcode = if is_float { Opcode::AddF } else { Opcode::AddI };
            let mut add_op_builder =
                OpBuilder::new(add_opcode, self.ir_location(&call_expr.span()))
                    .result(elem_ir_ty.clone())
                    .operand(arg0)
                    .operand(arg1);
            if is_float {
                let (rn, rv) = rounding_mode_attr("nearest_even");
                add_op_builder = add_op_builder.attr(rn, rv);
            } else {
                add_op_builder = add_op_builder.attr("overflow", Attribute::i32(0));
            }
            let (add_op_id, add_results) = add_op_builder.build(module);
            append_op(module, scan_block_id, add_op_id);
            add_results[0]
        };

        let (yield_op_id, _) = OpBuilder::new(Opcode::Yield, self.ir_location(&call_expr.span()))
            .operand(scan_result)
            .build(module);
        append_op(module, scan_block_id, yield_op_id);
        let region_id = module.alloc_region(Region {
            blocks: vec![scan_block_id],
        });
        let elem_scalar_scan =
            super::_type::scalar_from_name(&elem_ty_str).unwrap_or(cutile_ir::ir::ScalarType::I32);
        let identities_attr = if elem_scalar_scan.is_float() {
            Attribute::Array(vec![Attribute::Float(
                0.0,
                cutile_ir::ir::Type::Scalar(elem_scalar_scan),
            )])
        } else {
            Attribute::Array(vec![Attribute::Integer(
                0,
                cutile_ir::ir::Type::Scalar(elem_scalar_scan),
            )])
        };
        let reverse = super::shared_utils::extract_zst_type_name(&call_expr.args[2], "reverse")?;
        let reverse_value = match reverse.as_str() {
            "Forward" => false,
            "Reverse" => true,
            _ => {
                return self.jit_error_result(
                    &call_expr.args[2].span(),
                    &format!("invalid `reverse`: '{reverse}'. Valid: Forward, Reverse"),
                )
            }
        };

        let (op_id, results) = OpBuilder::new(Opcode::Scan, self.ir_location(&call_expr.span()))
            .result(result_ir_ty)
            .operand(operand_tile)
            .region(region_id)
            .attr("dim", Attribute::i32(0))
            .attr("reverse", Attribute::Bool(reverse_value))
            .attr("identities", identities_attr)
            .build(module);
        append_op(module, block_id, op_id);
        Ok(Some(TileRustValue::new_structured_type(
            results[0],
            return_type,
            None,
        )))
    }

    /// General-purpose op compilation for CUDA Tile dialect operations.
    #[allow(clippy::too_many_arguments)]
    fn compile_general_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        op_name: &str,
        op_attrs: &SingleMetaList,
        cuda_tile_op_params: &[String],
        cuda_tile_op_attribute_params: &[String],
        _cuda_tile_op_hint_params: &[String],
        cuda_tile_op_named_attributes: &[String],
        cuda_tile_op_static_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let rust_function_name = {
            let Expr::Path(path) = &*call_expr.func else {
                return self.jit_error_result(
                    &call_expr.func.span(),
                    "expected a function path in call expression",
                );
            };
            path.path.segments.last().unwrap().ident.to_string()
        };

        let return_type = if return_type.is_none() {
            if rust_function_name == "constant" {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!("Return type required for {}", call_expr.to_token_stream()),
                );
            }
            self.derive_type(
                module,
                block_id,
                &Expr::Call(call_expr.clone()),
                None,
                generic_args,
                ctx,
            )?
        } else {
            return_type
        };
        if return_type.is_none() {
            return self.jit_error_result(&call_expr.span(), "Unable to infer return type for op");
        }
        let return_type = return_type.unwrap();

        let mut type_meta = None;
        if let Some(output_meta_data) = op_attrs.parse_string_arr("output_type_meta") {
            let mut meta = TypeMeta {
                fields: BTreeMap::new(),
            };
            let param_names = get_sig_param_names(&fn_item.sig);
            for field_meta_expr_str in output_meta_data {
                let field_meta_expr_parts = field_meta_expr_str.split(".").collect::<Vec<&str>>();
                let field_meta_expr_param = field_meta_expr_parts[0];
                let mut succeeded = false;
                for (param_name, call_expr_arg) in param_names.iter().zip(call_expr.args.iter()) {
                    if param_name == field_meta_expr_param {
                        let call_expr_arg_str = call_expr_arg.to_token_stream().to_string();
                        let final_expr_str =
                            field_meta_expr_str.replace(field_meta_expr_param, &call_expr_arg_str);
                        let final_expr =
                            syn::parse2::<Expr>(final_expr_str.parse().unwrap()).unwrap();
                        let op_arg = self.compile_expression(
                            module,
                            block_id,
                            &final_expr,
                            generic_args,
                            ctx,
                            None,
                        )?;
                        if op_arg.is_none() {
                            return self.jit_error_result(&call_expr.span(), &format!("Failed to compile type meta {field_meta_expr_str} via expr {final_expr_str}"));
                        }
                        meta.fields
                            .insert(field_meta_expr_str.clone(), op_arg.unwrap());
                        succeeded = true;
                    }
                }
                if !succeeded {
                    return self.jit_error_result(&call_expr.span(), &format!("Unable to find param {field_meta_expr_param}, which was derived from type meta field for type meta {field_meta_expr_str}"));
                }
            }
            type_meta = Some(meta);
        };

        let opcode = op_name_to_opcode(op_name)?;
        let mut operand_lengths: Vec<String> = vec![];
        let mut op_operands: Vec<Value> = Vec::new();
        let mut compiled_args: Vec<TileRustValue> = Vec::new();
        for (op_param, call_expr_arg) in cuda_tile_op_params.iter().zip(call_expr.args.iter()) {
            let call_expr_arg_str = call_expr_arg.to_token_stream().to_string();
            let op_arg =
                self.compile_expression(module, block_id, call_expr_arg, generic_args, ctx, None)?;
            if op_arg.is_none() {
                return self.jit_error_result(&call_expr_arg.span(), "Failed to compile op arg");
            }
            let op_arg = op_arg.unwrap();
            compiled_args.push(op_arg.clone());
            let mut arg_values: Vec<Value> = vec![];
            if let Some(value) = op_arg.value {
                arg_values.push(value);
            } else if let Some(fields) = op_arg.fields {
                let op_path = op_param.split(".").collect::<Vec<&str>>();
                if op_path.len() <= 1 {
                    return self.jit_error_result(&call_expr_arg.span(), &format!("Field expression required for struct param {call_expr_arg_str}, got {op_param}"));
                }
                let field = *op_path.last().unwrap();
                match fields.get(field) {
                    Some(field_value) => {
                        if let Some(value) = field_value.value {
                            arg_values.push(value);
                        } else if let Some(values) = &field_value.values {
                            for value in values {
                                let Some(v) = value.value else {
                                    return self.jit_error_result(&call_expr_arg.span(), &format!("Unexpected nested array {op_param} for {call_expr_arg_str}"));
                                };
                                arg_values.push(v);
                            }
                        } else if field_value.fields.is_some() {
                            return self.jit_error_result(
                                &call_expr_arg.span(),
                                &format!(
                                    "Unexpected nested struct {op_param} for {call_expr_arg_str}"
                                ),
                            );
                        } else {
                            return self.jit_error_result(
                                &call_expr_arg.span(),
                                &format!("Unexpected op param {op_param} for {call_expr_arg_str}"),
                            );
                        }
                    }
                    None => {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!("Failed to access field {op_param} for {call_expr_arg_str}"),
                        )
                    }
                }
            } else if let Some(values) = op_arg.values.as_ref() {
                for value in values {
                    let Some(v) = value.value else {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!("Unexpected nested array {op_param} for {call_expr_arg_str}"),
                        );
                    };
                    arg_values.push(v);
                }
            } else {
                return self.jit_error_result(
                    &call_expr_arg.span(),
                    &format!("Unexpected op param {op_param} for {call_expr_arg_str}"),
                );
            }
            operand_lengths.push(arg_values.len().to_string());
            op_operands.extend_from_slice(&arg_values);
        }

        let mut attrs: Vec<(String, Attribute)> = vec![];
        for named_attr in cuda_tile_op_named_attributes.iter() {
            let name_attr_split = named_attr.split("=").collect::<Vec<&str>>();
            let (attr_name, attr_value) = (name_attr_split[0], name_attr_split[1]);
            if attr_name.starts_with("signedness") && attr_value == "inferred_signedness" {
                let elem_ty = compiled_args
                    .first()
                    .and_then(|arg| {
                        arg.ty
                            .get_instantiated_rust_element_type(self.modules.primitives())
                    })
                    .expect("Failed to get element type for signedness inference.");
                for arg in &compiled_args {
                    let arg_elem_ty = arg
                        .ty
                        .get_instantiated_rust_element_type(self.modules.primitives())
                        .expect("Operand types are not all equivalent.");
                    if arg_elem_ty != elem_ty {
                        return self.jit_error_result(&call_expr.span(), &format!("Element type mismatch for signedness inference: expected {elem_ty}, got {arg_elem_ty}"));
                    }
                }
                attrs.push(get_signedness_attr(attr_name, elem_ty.as_str())?);
            } else {
                attrs.push(build_named_attr(attr_name, attr_value)?);
            }
        }

        // Resolve static_params: ZST marker types -> tile-ir attributes.
        let resolved_static_attrs =
            resolve_static_params(cuda_tile_op_static_params, call_expr, fn_item)
                .map_err(JITError::Generic)?;
        for attr_str in &resolved_static_attrs {
            if let Some((name, val_str)) = attr_str.split_once('=') {
                let name = name.trim();
                let val_str = val_str.trim();
                let attr_val = if val_str == "unit" {
                    // Unit attribute = boolean flag present.
                    Attribute::i32(1)
                } else if let Ok(v) = val_str.parse::<i64>() {
                    Attribute::i32(v)
                } else if val_str.starts_with("#cuda_tile.rounding<") {
                    // Rounding mode enum: #cuda_tile.rounding<name>
                    let inner = val_str
                        .trim_start_matches("#cuda_tile.rounding<")
                        .trim_end_matches('>');
                    let rm = match inner {
                        "nearest_even" => 0,
                        "positive_inf" => 1,
                        "negative_inf" => 2,
                        "nearest_int_to_zero" => 3,
                        "zero" => 4,
                        "approx" => 5,
                        "full" => 6,
                        other => {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("unknown rounding mode '{other}'"),
                            );
                        }
                    };
                    Attribute::i32(rm)
                } else if val_str.starts_with("#cuda_tile.overflow<") {
                    let inner = val_str
                        .trim_start_matches("#cuda_tile.overflow<")
                        .trim_end_matches('>');
                    overflow_attr(inner).1
                } else if val_str.starts_with("#cuda_tile.cmp_predicate<") {
                    let inner = val_str
                        .trim_start_matches("#cuda_tile.cmp_predicate<")
                        .trim_end_matches('>');
                    cmp_pred_attr(inner).1
                } else if val_str.starts_with("#cuda_tile.comparison_ordering<") {
                    let inner = val_str
                        .trim_start_matches("#cuda_tile.comparison_ordering<")
                        .trim_end_matches('>');
                    cmp_ordering_attr(inner).1
                } else if val_str.starts_with("#cuda_tile.signedness<") {
                    let inner = val_str
                        .trim_start_matches("#cuda_tile.signedness<")
                        .trim_end_matches('>');
                    signedness_attr(name, inner).1
                } else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "static_params: unsupported attribute value '{}' in '{}'",
                            val_str, attr_str
                        ),
                    );
                };
                attrs.push((name.to_string(), attr_val));
            }
        }

        let mut cuda_tile_op_attr_params_iter = cuda_tile_op_attribute_params.iter();
        let mut maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
        let fn_params = get_sig_param_names(&fn_item.sig);
        for (fn_param, call_expr_arg) in fn_params.iter().zip(call_expr.args.iter()) {
            if maybe_next_attr_param.is_none() {
                break;
            }
            let next_attr: &String = maybe_next_attr_param.as_ref().unwrap();
            let op_attr = next_attr.split(":").collect::<Vec<_>>();
            if op_attr.len() != 2 {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!("Expected 2-element attribute, got {}", op_attr.len()),
                );
            }
            let (attr_id, attr_ty): (&str, &str) = (op_attr[0], op_attr[1]);
            match attr_ty {
                "array" => {
                    if attr_id != fn_param {
                        continue;
                    }
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let call_expr_arg_str = call_expr_arg.to_token_stream().to_string();
                    let op_arg = self.compile_expression(
                        module,
                        block_id,
                        call_expr_arg,
                        generic_args,
                        ctx,
                        None,
                    )?;
                    if op_arg.is_none() {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!("Failed to compile attribute arg for {call_expr_arg_str}"),
                        );
                    }
                    let op_arg = op_arg.unwrap();
                    match op_arg.kind {
                        Kind::Struct | Kind::StructuredType => match &op_arg.ty.rust_ty {
                            Type::Path(_ty_path) => {
                                let Some(cga) = get_cga_from_type(&op_arg.ty.rust_ty, generic_args)
                                else {
                                    return self.jit_error_result(
                                        &call_expr_arg.span(),
                                        "Failed to build attribute",
                                    );
                                };
                                attrs.push((attr_id.to_string(), Attribute::DenseI32Array(cga)));
                            }
                            _ => {
                                return self.jit_error_result(
                                    &call_expr_arg.span(),
                                    "Attribute type not implemented.",
                                )
                            }
                        },
                        _ => {
                            return self.jit_error_result(
                                &call_expr_arg.span(),
                                &format!("Unexpected call arg {call_expr_arg_str} for {next_attr}"),
                            )
                        }
                    }
                }
                "dense" => {
                    if attr_id != fn_param {
                        continue;
                    }
                    let (lit_value, _lit_ty_name) = match &call_expr_arg {
                        Expr::Lit(lit_expr) => match &lit_expr.lit {
                            Lit::Bool(b) => (b.value.to_string(), "i1".to_string()),
                            Lit::Int(i) => (i.base10_digits().to_string(), "i32".to_string()),
                            Lit::Float(f) => (f.base10_digits().to_string(), "f32".to_string()),
                            _ => {
                                return self.jit_error_result(
                                    &call_expr_arg.span(),
                                    "Constant not supported",
                                )
                            }
                        },
                        Expr::Unary(unary_expr) => {
                            let UnOp::Neg(_) = unary_expr.op else {
                                return self.jit_error_result(
                                    &call_expr_arg.span(),
                                    "Only unary negation is supported for constant values",
                                );
                            };
                            match &*unary_expr.expr {
                                Expr::Lit(lit_expr) => match &lit_expr.lit {
                                    Lit::Int(i) => {
                                        (format!("-{}", i.base10_digits()), "i32".to_string())
                                    }
                                    Lit::Float(f) => {
                                        (format!("-{}", f.base10_digits()), "f32".to_string())
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &call_expr_arg.span(),
                                            "Unsupported literal type for negation",
                                        )
                                    }
                                },
                                _ => {
                                    return self.jit_error_result(
                                        &call_expr_arg.span(),
                                        "Only literal negation is supported for constant values",
                                    )
                                }
                            }
                        }
                        Expr::Path(path_expr) => {
                            let path_expr_string = path_expr.to_token_stream().to_string();
                            let ty_val_split = path_expr_string.split(" :: ").collect::<Vec<_>>();
                            if ty_val_split.len() != 2 {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    "Unexpected dense value.",
                                );
                            }
                            let (ty_raw, const_val) =
                                (ty_val_split[0].to_string(), ty_val_split[1].to_string());
                            // Resolve generic type parameters (e.g. `T::ZERO` where
                            // `T` is a kernel generic) to their concrete
                            // monomorphized type name before dispatching to
                            // `get_const_hex`.
                            let ty = generic_args
                                .inst_types
                                .get(&ty_raw)
                                .cloned()
                                .unwrap_or(ty_raw);
                            match const_val.as_str() {
                                "ZERO" => (get_const_hex(ty.as_str(), "zero")?, ty.clone()),
                                "ONE" => (get_const_hex(ty.as_str(), "one")?, ty.clone()),
                                "NEG_INFINITY" => (get_const_hex(ty.as_str(), "min")?, ty.clone()),
                                "INFINITY" => (get_const_hex(ty.as_str(), "max")?, ty.clone()),
                                "E" => (get_const_hex(ty.as_str(), "e")?, ty.clone()),
                                _ => {
                                    return self.jit_error_result(
                                        &call_expr_arg.span(),
                                        "Constant not supported",
                                    )
                                }
                            }
                        }
                        _ => {
                            return self.jit_error_result(
                                &call_expr_arg.span(),
                                "Unsupported expression for named attribute.",
                            )
                        }
                    };
                    // Build a DenseElements attribute from the literal value.
                    let elem_ty_str = return_type
                        .get_cuda_tile_element_type(self.modules.primitives())?
                        .unwrap_or("i32".to_string());
                    let result_ir_ty = super::_type::scalar_from_name(&elem_ty_str)
                        .map(|sc| {
                            cutile_ir::ir::Type::Tile(cutile_ir::ir::TileType {
                                shape: vec![],
                                element_type: cutile_ir::ir::TileElementType::Scalar(sc),
                            })
                        })
                        .unwrap_or_else(|| {
                            cutile_ir::ir::Type::Tile(cutile_ir::ir::TileType {
                                shape: vec![],
                                element_type: cutile_ir::ir::TileElementType::Scalar(
                                    cutile_ir::ir::ScalarType::I32,
                                ),
                            })
                        });
                    let data = crate::compiler::compile_expression::encode_literal_bytes(
                        &lit_value,
                        &elem_ty_str,
                    );
                    attrs.push((
                        "value".to_string(),
                        Attribute::DenseElements(cutile_ir::ir::DenseElements {
                            element_type: result_ir_ty,
                            shape: vec![],
                            data,
                        }),
                    ));
                }
                "rounding" => {
                    if attr_id != fn_param {
                        continue;
                    }
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let rounding_mode_str = match &call_expr_arg {
                        Expr::Lit(ExprLit {
                            lit: Lit::Str(lit_str),
                            ..
                        }) => lit_str.value(),
                        _ => {
                            return self.jit_error_result(
                                &call_expr_arg.span(),
                                "Rounding mode must be a string literal.",
                            )
                        }
                    };
                    const VALID_MODES: &[&str] = &[
                        "nearest_even",
                        "positive_inf",
                        "negative_inf",
                        "nearest_int_to_zero",
                        "approx",
                    ];
                    if !VALID_MODES.contains(&rounding_mode_str.as_str()) {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!(
                                "Invalid rounding mode: \"{}\". Valid values are: {}",
                                rounding_mode_str,
                                VALID_MODES.join(", ")
                            ),
                        );
                    }
                    attrs.push(rounding_mode_attr(&rounding_mode_str));
                }
                "memory_ordering" => {
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    attrs.push(int_attr(attr_id, 1));
                }
                "memory_scope" => {
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    attrs.push(int_attr(attr_id, 1));
                }
                "integer" => {
                    if attr_id != fn_param {
                        continue;
                    }
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let op_arg = self.compile_expression(
                        module,
                        block_id,
                        call_expr_arg,
                        generic_args,
                        ctx,
                        None,
                    )?;
                    if op_arg.is_none() {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!("Failed to compile integer attribute {attr_id}"),
                        );
                    }
                    let op_arg = op_arg.unwrap();
                    if op_arg.value.is_none() {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!("Integer attribute {attr_id} must be a value"),
                        );
                    }
                    if let Some(bounds) = op_arg.bounds {
                        if bounds.is_exact() {
                            attrs.push(int_attr(attr_id, bounds.start));
                        } else {
                            return self.jit_error_result(&call_expr_arg.span(), &format!("Integer attribute {attr_id} must be a constant value, got bounds: {bounds:?}"));
                        }
                    } else {
                        return self.jit_error_result(
                            &call_expr_arg.span(),
                            &format!("Integer attribute {attr_id} must be a constant value"),
                        );
                    }
                }
                _ => {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Attribute type not implemented: {attr_ty}"),
                    )
                }
            }
        }

        if op_attrs.parse_bool("has_variadic_params").unwrap_or(false) {
            attrs.push((
                "operandSegmentSizes".to_string(),
                Attribute::Array(
                    operand_lengths
                        .iter()
                        .map(|s| Attribute::i32(s.parse::<i64>().unwrap()))
                        .collect(),
                ),
            ));
        };

        let mut op_builder = OpBuilder::new(opcode, self.ir_location(&call_expr.span()))
            .operands(op_operands.iter().copied())
            .attrs(attrs);

        if function_returns(fn_item) {
            match return_type.kind {
                Kind::PrimitiveType | Kind::StructuredType => {
                    if return_type.tile_ir_ty.is_none() { return self.jit_error_result(&call_expr.span(), "return type is missing a compiled tile type"); }
                    let result_ir_ty = super::_type::convert_type(&return_type)
                        .ok_or_else(|| self.jit_error(&call_expr.span(), &format!("failed to convert return type to tile-ir type: {:?}", return_type.cuda_tile_name)))?;
                    op_builder = op_builder.result(result_ir_ty);
                    let (op_id, results) = op_builder.build(module);
                    append_op(module, block_id, op_id);
                    match return_type.kind {
                        Kind::PrimitiveType => Ok(Some(TileRustValue::new_primitive(results[0], return_type, None))),
                        Kind::StructuredType => Ok(Some(TileRustValue::new_structured_type(results[0], return_type, type_meta))),
                        _ => unreachable!(),
                    }
                }
                Kind::Compound => {
                    if let Type::Tuple(tuple_type) = &return_type.rust_ty {
                        let mut elem_types = vec![];
                        for elem in &tuple_type.elems {
                            let elem_ty = self.compile_type(elem, generic_args, &HashMap::new())?;
                            if elem_ty.is_none() { return self.jit_error_result(&call_expr.span(), "failed to compile type"); }
                            let elem_ty = elem_ty.unwrap();
                            if elem_ty.tile_ir_ty.is_none() { return self.jit_error_result(&call_expr.span(), "failed to compile tile type"); }
                            let elem_ir_ty = super::_type::convert_type(&elem_ty)
                                .ok_or_else(|| self.jit_error(
                                    &call_expr.span(),
                                    &format!("failed to convert element type to tile-ir type: {:?}", elem_ty.cuda_tile_name),
                                ))?;
                            op_builder = op_builder.result(elem_ir_ty);
                            elem_types.push(elem_ty);
                        }
                        let (op_id, results) = op_builder.build(module);
                        append_op(module, block_id, op_id);
                        let mut values = vec![];
                        for (i, elem_ty) in elem_types.iter().enumerate() {
                            match elem_ty.kind {
                                Kind::PrimitiveType => {
                                    let op_value = if op_name == "cuda_tile.get_num_tile_blocks" || op_name == "cuda_tile.get_tile_block_id" {
                                        self.compile_value_assumption(module, block_id, results[i], "assume_bounds_lower", &[0], elem_ty.clone(), &call_expr.span())?.value
                                            .expect("Expected a value from compiled assumption.")
                                    } else {
                                        results[i]
                                    };
                                    let maybe_bounds = if let Some(const_grid) = self.const_grid {
                                        if op_name == "cuda_tile.get_num_tile_blocks" { let cb = match i { 0 => const_grid.0, 1 => const_grid.1, 2 => const_grid.2, _ => unreachable!() }; Some(Bounds::exact(cb as i64)) }
                                        else if op_name == "cuda_tile.get_tile_block_id" { let cb = match i { 0 => const_grid.0, 1 => const_grid.1, 2 => const_grid.2, _ => unreachable!() }; Some(Bounds::new(0i64, cb as i64 - 1)) }
                                        else { None }
                                    } else { None };
                                    values.push(TileRustValue::new_primitive(op_value, elem_ty.clone(), maybe_bounds));
                                }
                                Kind::StructuredType => { values.push(TileRustValue::new_structured_type(results[i], elem_ty.clone(), None)); }
                                Kind::Compound | Kind::Struct | Kind::String | Kind::Enum => return self.jit_error_result(&call_expr.span(), &format!("this operation returned an unsupported element type ({:?}); only scalar and structured types are supported", elem_ty.kind)),
                            }
                        }
                        Ok(Some(TileRustValue::new_compound(values, return_type)))
                    } else { self.jit_error_result(&call_expr.span(), &format!("operations that return multiple values must use a tuple return type, got `{}`", return_type.rust_ty.to_token_stream())) }
                }
                Kind::Struct => self.jit_error_result(&call_expr.span(), "this operation cannot return a struct; only scalar and structured (tile) types are supported as return types"),
                Kind::String => self.jit_error_result(&call_expr.span(), "this operation cannot return a string; only scalar and structured (tile) types are supported as return types"),
                Kind::Enum => self.jit_error_result(&call_expr.span(), "this operation cannot return an enum; only scalar and structured (tile) types are supported as return types"),
            }
        } else {
            let (op_id, _) = op_builder.build(module);
            append_op(module, block_id, op_id);
            Ok(None)
        }
    }

    pub fn compile_cuda_tile_macro(
        &self,
        module: &mut Module,
        block_id: BlockId,
        mac: &syn::Macro,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        _return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let Some(mac_ident) = mac.path.get_ident() else {
            return self.jit_error_result(&mac.path.span(), "unrecognized macro invocation");
        };
        match mac_ident.to_string().as_str() {
            "cuda_tile_print" => {
                let exprs = super::shared_utils::parse_list_of_expr(mac.tokens.clone())?;
                let mut str_literal = String::new();
                let mut element_type_instance = vec![];
                let mut arg_values = vec![];
                for expr in &exprs {
                    match expr {
                        Expr::Lit(ExprLit {
                            lit: Lit::Str(lit), ..
                        }) => {
                            str_literal = lit.value();
                        }
                        _ => {
                            let Some(val) = self.compile_expression(
                                module,
                                block_id,
                                expr,
                                generic_vars,
                                ctx,
                                None,
                            )?
                            else {
                                return self.jit_error_result(
                                    &expr.span(),
                                    "failed to compile print argument",
                                );
                            };
                            if val.kind != PrimitiveType && val.kind != StructuredType {
                                return self.jit_error_result(
                                    &expr.span(),
                                    "print arguments must be scalar or tile values",
                                );
                            }
                            arg_values.push(val.value.unwrap());
                            element_type_instance
                                .push(val.ty.type_instance.get_rust_element_instance_ty())
                        }
                    }
                }
                let re_repl = Regex::new(r"\{\}").unwrap();
                for (i, element_ty) in element_type_instance.into_iter().enumerate() {
                    let rust_element_type_instance = element_ty.unwrap_or_else(|| {
                        panic!("failed to determine element type for print argument {}", i)
                    });
                    if !re_repl.is_match(&str_literal) {
                        return self.jit_error_result(
                            &mac.span(),
                            "more arguments than `{}` placeholders in print format string",
                        );
                    }
                    let Some(tile_element_type_instance) =
                        get_cuda_tile_element_type_from_rust_primitive_str(
                            &rust_element_type_instance,
                            self.modules.primitives(),
                        )
                    else {
                        return self.jit_error_result(&mac.span(), &format!("unable to determine tile element type for `{rust_element_type_instance}`"));
                    };
                    let first_char = tile_element_type_instance.chars().next().unwrap();
                    str_literal = re_repl
                        .replacen(&str_literal, 1, format!("%{first_char}"))
                        .to_string();
                }
                if re_repl.is_match(&str_literal) {
                    return self.jit_error_result(
                        &mac.span(),
                        "more `{}` placeholders than arguments in print format string",
                    );
                }
                let operand_seg_sizes: Vec<i64> = vec![arg_values.len() as i64, 0];
                let (print_op_id, _) = OpBuilder::new(Opcode::Print, self.ir_location(&mac.span()))
                    .attr("str", Attribute::String(str_literal))
                    .attr(
                        "operandSegmentSizes",
                        Attribute::Array(
                            operand_seg_sizes
                                .iter()
                                .map(|&x| Attribute::i32(x))
                                .collect(),
                        ),
                    )
                    .operands(arg_values.iter().copied())
                    .result(TileIrType::Token)
                    .build(module);
                append_op(module, block_id, print_op_id);
                Ok(None)
            }
            "cuda_tile_assert" => {
                let punctuated = Punctuated::<Expr, Token![,]>::parse_terminated;
                let expressions_err = syn::parse::Parser::parse2(punctuated, mac.tokens.clone())
                    .expect("Failed to parse cuda_tile_assert expression.");
                if expressions_err.len() != 2 {
                    return self.jit_error_result(
                        &mac.span(),
                        &format!(
                            "`cuda_tile_assert!` expects 2 arguments (condition, message), got {}",
                            expressions_err.len()
                        ),
                    );
                }
                let bool_expr = &expressions_err[0];
                let message = &expressions_err[1];
                let str_lit =
                    match message {
                        Expr::Lit(ExprLit {
                            lit: Lit::Str(lit), ..
                        }) => lit.value(),
                        _ => return self.jit_error_result(
                            &expressions_err[1].span(),
                            "the second argument to `cuda_tile_assert!` must be a string literal",
                        ),
                    };
                let assert_arg_values = {
                    let Some(val) = self.compile_expression(
                        module,
                        block_id,
                        bool_expr,
                        generic_vars,
                        ctx,
                        None,
                    )?
                    else {
                        return self.jit_error_result(
                            &bool_expr.span(),
                            "failed to compile assert condition",
                        );
                    };
                    if val.kind != PrimitiveType && val.kind != StructuredType {
                        return self.jit_error_result(
                            &bool_expr.span(),
                            "assert condition must be a scalar or tile value",
                        );
                    }
                    vec![val.value.unwrap()]
                };
                let (assert_op_id, _) =
                    OpBuilder::new(Opcode::Assert, self.ir_location(&mac.span()))
                        .attr("message", Attribute::String(str_lit))
                        .operands(assert_arg_values.iter().copied())
                        .build(module);
                append_op(module, block_id, assert_op_id);
                Ok(None)
            }
            _ => self.jit_error_result(
                &mac.path.span(),
                &format!("unrecognized macro `{}`", mac_ident),
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map an op_name string from the old compiler to a tile-ir Opcode.
fn op_name_to_opcode(op_name: &str) -> Result<Opcode, JITError> {
    let normalized = op_name.strip_prefix("cuda_tile.").unwrap_or(op_name);
    match normalized {
        "load_ptr_tko" => Ok(Opcode::LoadPtrTko),
        "store_ptr_tko" => Ok(Opcode::StorePtrTko),
        "atomic_rmw_tko" => Ok(Opcode::AtomicRMW),
        "atomic_cas_tko" => Ok(Opcode::AtomicCAS),
        "load_view_tko" => Ok(Opcode::LoadViewTko),
        "store_view_tko" => Ok(Opcode::StoreViewTko),
        "reduce" => Ok(Opcode::Reduce),
        "scan" => Ok(Opcode::Scan),
        "make_partition_view" => Ok(Opcode::MakePartitionView),
        "make_tensor_view" => Ok(Opcode::MakeTensorView),
        "make_token" => Ok(Opcode::MakeToken),
        "join_tokens" => Ok(Opcode::JoinTokens),
        "get_tensor_shape" => Ok(Opcode::GetTensorShape),
        "offset" => Ok(Opcode::Offset),
        "break" => Ok(Opcode::Break),
        "continue" => Ok(Opcode::Continue),
        "yield" => Ok(Opcode::Yield),
        "constant" => Ok(Opcode::Constant),
        "broadcast" => Ok(Opcode::Broadcast),
        "reshape" => Ok(Opcode::Reshape),
        "iota" => Ok(Opcode::Iota),
        "cat" => Ok(Opcode::Cat),
        "permute" => Ok(Opcode::Permute),
        "extract" => Ok(Opcode::Extract),
        "select" => Ok(Opcode::Select),
        "addf" => Ok(Opcode::AddF),
        "addi" => Ok(Opcode::AddI),
        "subf" => Ok(Opcode::SubF),
        "subi" => Ok(Opcode::SubI),
        "mulf" => Ok(Opcode::MulF),
        "muli" => Ok(Opcode::MulI),
        "divf" => Ok(Opcode::DivF),
        "divi" => Ok(Opcode::DivI),
        "remf" => Ok(Opcode::RemF),
        "remi" => Ok(Opcode::RemI),
        "negf" => Ok(Opcode::NegF),
        "negi" => Ok(Opcode::NegI),
        "absf" => Ok(Opcode::AbsF),
        "absi" => Ok(Opcode::AbsI),
        "atan2" => Ok(Opcode::Atan2),
        "maxf" => Ok(Opcode::MaxF),
        "maxi" => Ok(Opcode::MaxI),
        "minf" => Ok(Opcode::MinF),
        "mini" => Ok(Opcode::MinI),
        "andi" => Ok(Opcode::AndI),
        "ori" => Ok(Opcode::OrI),
        "xori" => Ok(Opcode::XOrI),
        "shli" => Ok(Opcode::ShLI),
        "shri" => Ok(Opcode::ShRI),
        "mulhii" => Ok(Opcode::MulhiI),
        "cmpf" => Ok(Opcode::CmpF),
        "cmpi" => Ok(Opcode::CmpI),
        "bitcast" => Ok(Opcode::Bitcast),
        "exti" => Ok(Opcode::ExtI),
        "trunci" => Ok(Opcode::TruncI),
        "ftof" => Ok(Opcode::FToF),
        "ftoi" => Ok(Opcode::FToI),
        "itof" => Ok(Opcode::IToF),
        "int_to_ptr" => Ok(Opcode::IntToPtr),
        "ptr_to_int" => Ok(Opcode::PtrToInt),
        "ptr_to_ptr" => Ok(Opcode::PtrToPtr),
        "exp" => Ok(Opcode::Exp),
        "exp2" => Ok(Opcode::Exp2),
        "log" => Ok(Opcode::Log),
        "log2" => Ok(Opcode::Log2),
        "sqrt" => Ok(Opcode::Sqrt),
        "rsqrt" => Ok(Opcode::Rsqrt),
        "sin" => Ok(Opcode::Sin),
        "cos" => Ok(Opcode::Cos),
        "tan" => Ok(Opcode::Tan),
        "sinh" => Ok(Opcode::SinH),
        "cosh" => Ok(Opcode::CosH),
        "tanh" => Ok(Opcode::TanH),
        "ceil" => Ok(Opcode::Ceil),
        "floor" => Ok(Opcode::Floor),
        "pow" => Ok(Opcode::Pow),
        "fma" => Ok(Opcode::Fma),
        "mmaf" => Ok(Opcode::MmaF),
        "mmai" => Ok(Opcode::MmaI),
        "assert" => Ok(Opcode::Assert),
        "assume" => Ok(Opcode::Assume),
        "print" | "print_tko" => Ok(Opcode::Print),
        "get_global" => Ok(Opcode::GetGlobal),
        "global" => Ok(Opcode::Global),
        "get_index_space_shape" => Ok(Opcode::GetIndexSpaceShape),
        "get_num_tile_blocks" => Ok(Opcode::GetNumTileBlocks),
        "get_tile_block_id" => Ok(Opcode::GetTileBlockId),
        "for" => Ok(Opcode::For),
        "if" => Ok(Opcode::If),
        "loop" => Ok(Opcode::Loop),
        _ => Err(JITError::Generic(format!(
            "unknown cuda_tile op name: `{op_name}`"
        ))),
    }
}

/// Parse a general named attribute from the old compiler's format.
/// Build a named attribute using the correct builder type based on the
/// attribute name and value string from the op annotation.
///
/// Dispatches on `attr_name` to determine the expected attribute type
/// (from Ops.td), then builds the correctly-typed `Attribute` using the
/// builder API — no string fallback.
fn build_named_attr(attr_name: &str, attr_value: &str) -> Result<(String, Attribute), JITError> {
    match attr_name {
        // --- Enum attributes (stored as Attribute::Integer with i32 type) ---
        "overflow" => {
            let inner = extract_enum_inner(attr_value);
            Ok(overflow_attr(&inner))
        }
        "rounding_mode" | "rounding" => {
            let inner = extract_enum_inner(attr_value);
            Ok(rounding_mode_attr(&inner))
        }
        "comparison_predicate" => {
            let inner = extract_enum_inner(attr_value);
            Ok(cmp_pred_attr(&inner))
        }
        "comparison_ordering" => {
            let inner = extract_enum_inner(attr_value);
            Ok(cmp_ordering_attr(&inner))
        }
        "signedness" | "signedness_lhs" | "signedness_rhs" => {
            let inner = extract_enum_inner(attr_value);
            Ok(signedness_attr(attr_name, &inner))
        }
        "memory_ordering_semantics" => {
            let inner = extract_enum_inner(attr_value);
            Ok(memory_ordering_attr(&inner))
        }
        "memory_scope" => {
            let inner = extract_enum_inner(attr_value);
            Ok(memory_scope_attr(&inner))
        }

        // --- DenseI32Array attributes ---
        "permutation" => {
            let arr = try_parse_dense_i32_array(attr_value).ok_or_else(|| {
                JITError::generic_err(&format!(
                    "failed to parse DenseI32Array for '{attr_name}': {attr_value}"
                ))
            })?;
            Ok((attr_name.to_string(), Attribute::DenseI32Array(arr)))
        }

        // --- Typed array attributes ---
        "identities" => {
            let arr = try_parse_identities_array(attr_value).ok_or_else(|| {
                JITError::generic_err(&format!(
                    "failed to parse identities array for '{attr_name}': {attr_value}"
                ))
            })?;
            Ok((attr_name.to_string(), Attribute::Array(arr)))
        }

        // --- Integer attributes ---
        "dim" => {
            let val = attr_value.trim().parse::<i64>().map_err(|_| {
                JITError::generic_err(&format!(
                    "failed to parse integer for '{attr_name}': {attr_value}"
                ))
            })?;
            Ok(int_attr(attr_name, val))
        }

        // --- Bool attributes ---
        "reverse" => {
            let val = match attr_value.trim() {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => {
                    return Err(JITError::generic_err(&format!(
                        "failed to parse bool for '{attr_name}': {attr_value}"
                    )))
                }
            };
            Ok((attr_name.to_string(), Attribute::Bool(val)))
        }

        // --- String attributes ---
        "message" | "sym_name" | "name" | "str" => Ok(str_attr(attr_name, attr_value)),

        // --- Unknown: error instead of silent str_attr fallback ---
        _ => Err(JITError::generic_err(&format!(
            "unknown named attribute '{attr_name}' with value '{attr_value}' — \
             add an explicit case to build_named_attr()"
        ))),
    }
}

/// Extract the inner value from an enum-style attribute string.
/// Handles both `#cuda_tile.foo<bar>` and plain `bar` formats.
fn extract_enum_inner(attr_value: &str) -> String {
    if let Some(inner) = extract_enum_attr_value(attr_value) {
        inner
    } else {
        // Plain value without angle brackets (e.g., "signed", "nearest_even")
        attr_value.trim().to_string()
    }
}

/// Parse an identities-style array like "[0xff800000 : f32]" or "[0x00000000 : i32]"
/// into an Array of typed Float/Integer attributes.
fn try_parse_identities_array(s: &str) -> Option<Vec<Attribute>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Some(vec![]);
    }
    let mut result = Vec::new();
    for element in inner.split(',') {
        let element = element.trim();
        // Format: "0xHEXVALUE : type"
        let parts: Vec<&str> = element.split(':').collect();
        if parts.len() != 2 {
            return None;
        }
        let hex_str = parts[0].trim();
        let ty_str = parts[1].trim();
        let hex_val = hex_str
            .strip_prefix("0x")
            .or_else(|| hex_str.strip_prefix("0X"))?;
        let bits = u64::from_str_radix(hex_val, 16).ok()?;
        let scalar_ty = super::_type::scalar_from_name(ty_str)?;
        let ir_ty = cutile_ir::ir::Type::Scalar(scalar_ty);
        if scalar_ty.is_float() {
            // Float: interpret bits as the float type's bit pattern
            let float_val = match ty_str {
                "f32" => f32::from_bits(bits as u32) as f64,
                "f64" => f64::from_bits(bits),
                "f16" => half::f16::from_bits(bits as u16).to_f64(),
                "bf16" => half::bf16::from_bits(bits as u16).to_f64(),
                _ => bits as f64,
            };
            result.push(Attribute::Float(float_val, ir_ty));
        } else {
            result.push(Attribute::Integer(bits as i64, ir_ty));
        }
    }
    Some(result)
}

fn try_parse_dense_i32_array(s: &str) -> Option<Vec<i32>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }
    let inner = s[1..s.len() - 1].trim();
    if inner.is_empty() {
        return Some(vec![]);
    }
    let values: Result<Vec<i32>, _> = inner.split(',').map(|v| v.trim().parse::<i32>()).collect();
    values.ok()
}

fn extract_enum_attr_value(s: &str) -> Option<String> {
    let start = s.find('<')?;
    let end = s.rfind('>')?;
    if start < end {
        Some(s[start + 1..end].to_string())
    } else {
        None
    }
}
