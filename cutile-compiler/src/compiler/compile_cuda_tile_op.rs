/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA Tile op compilation: handles `compile_op_call` (CUDA Tile dialect operations)
//! within the CUDA Tile compiler. This module covers the translation of annotated op functions
//! into MLIR operations.

use crate::bounds::Bounds;
use crate::compiler::_function::CUDATileFunctionCompiler;
use crate::compiler::_type::Kind::{PrimitiveType, StructuredType};
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;
use crate::compiler::utils::{
    get_const_hex, get_signedness_attr, named_array_attr, named_str_attr, parse_list_of_expr,
    update_token, AtomicMode,
};
use crate::cuda_tile;
use crate::error::JITError;
use crate::generics::{get_cga_from_type, GenericVars};
use crate::syn_utils::*;
use crate::types::*;
use melior::ir::attribute::{IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::{
    self, Attribute, Block, BlockLike, Identifier, Location, Region, RegionLike, Value,
};
use quote::ToTokens;
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{Expr, ExprCall, ExprLit, ItemFn, Lit, Token, Type, UnOp};

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
/// Returns a list of `"attr_name=attr_val"` strings to be emitted as MLIR named attributes.
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
                // e.g., `ftz::Enabled` → "Enabled", or just `Enabled` → "Enabled"
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
        // Types with no mapping entry (e.g., ftz::Disabled) emit nothing — that's valid.
        if !matched {
            // No mapping found — this is the "omit" case (e.g., Disabled).
            // This is not an error.
        }
    }

    Ok(attrs)
}

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    pub fn compile_cuda_tile_op_call(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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
            return self.jit_error_result(
                &call_expr.func.span(),
                &format!("undefined function `{rust_function_name}"),
            );
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

        let cuda_tile_op_params = op_attrs
            .parse_string_arr("params")
            .unwrap_or_else(|| vec![]);
        let cuda_tile_op_attribute_params = op_attrs
            .parse_string_arr("attribute_params")
            .unwrap_or_else(|| vec![]);
        let cuda_tile_op_hint_params = op_attrs
            .parse_string_arr("hint_params")
            .unwrap_or_else(|| vec![]);
        let cuda_tile_op_named_attributes = op_attrs
            .parse_string_arr("named_attributes")
            .unwrap_or_else(|| vec![]);
        let cuda_tile_op_static_params = op_attrs
            .parse_string_arr("static_params")
            .unwrap_or_else(|| vec![]);
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
            builder,
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
        Ok(self.compile_general_op(
            builder,
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
        )?)
    }

    /// Compile special-cased ops (load_ptr_tko, store_ptr_tko, atomic_rmw_tko,
    /// atomic_cas_tko, load_view_tko, store_view_tko, cuda_tile.reduce, cuda_tile.scan).
    /// Returns `Some(Some(...))` or `Some(None)` if a special case was handled,
    /// or `None` if the caller should fall through to general compilation.
    fn try_compile_cuda_tile_special_op(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        op_name: &str,
        cuda_tile_op_hint_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        match op_name {
            "cuda_tile.load_ptr_tko" => {
                self.compile_load_ptr_tko(builder, call_expr, generic_args, ctx, return_type)
            }
            "cuda_tile.store_ptr_tko" => {
                self.compile_store_ptr_tko(builder, call_expr, generic_args, ctx)
            }
            "cuda_tile.atomic_rmw_tko" => {
                self.compile_atomic_rmw_tko(builder, call_expr, generic_args, ctx, return_type)
            }
            "cuda_tile.atomic_cas_tko" => {
                self.compile_atomic_cas_tko(builder, call_expr, generic_args, ctx, return_type)
            }
            "load_view_tko" => self.compile_load_view_tko(
                builder,
                call_expr,
                fn_item,
                cuda_tile_op_hint_params,
                generic_args,
                ctx,
                return_type,
            ),
            "store_view_tko" => self.compile_store_view_tko(
                builder,
                call_expr,
                fn_item,
                cuda_tile_op_hint_params,
                generic_args,
                ctx,
            ),
            "cuda_tile.reduce" => {
                self.compile_reduce_op(builder, call_expr, generic_args, ctx, return_type)
            }
            "cuda_tile.scan" => self.compile_scan_op(builder, call_expr, generic_args, ctx),
            _ => Ok(None),
        }
    }

    fn compile_load_ptr_tko(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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

        let tile_result_ty = tile_elem_ty.cuda_tile_ty.unwrap();
        let token_result_ty = token_elem_ty.cuda_tile_ty.unwrap();

        // arg[0]: source (required)
        let source_arg = &call_expr.args[0];
        let Some(source_value) =
            self.compile_expression(builder, source_arg, generic_args, ctx, None)?
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

        // arg[1]: memory_ordering (required string literal)
        let memory_ordering = crate::compiler::utils::extract_string_literal(
            &call_expr.args[1],
            "memory_ordering",
            ctx,
        )?;

        let memory_ordering_value = match memory_ordering.as_str() {
            "weak" => 0,
            "relaxed" => 1,
            "acquire" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                    "invalid `memory_ordering` for `load_ptr_tko: '{}'. Valid: weak, relaxed, acquire",
                    memory_ordering
                ),
                )
            }
        };

        // arg[2]: memory_scope (required string literal)
        let memory_scope = crate::compiler::utils::extract_string_literal(
            &call_expr.args[2],
            "memory_scope",
            ctx,
        )?;

        let memory_scope_value = match memory_scope.as_str() {
            "tl_blk" => 0,
            "device" => 1,
            "sys" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: `'{}'. Valid: tl_blk, device, sys",
                        memory_scope
                    ),
                )
            }
        };

        let mut operands = vec![source_ptr];
        let mut mask_count = 0;
        let mut padding_count = 0;
        let mut token_count = 0;

        // arg[3]: mask (Option<Tile<i1, S>>)
        if let Some(mask_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[3], ctx)
        {
            if let Some(mask_value) =
                self.compile_expression(builder, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }

        // arg[4]: padding_value (Option<E>)
        if let Some(padding_arg) =
            crate::compiler::utils::resolve_option_arg(&call_expr.args[4], ctx)
        {
            if let Some(padding_value) =
                self.compile_expression(builder, &padding_arg, generic_args, ctx, None)?
            {
                if let Some(padding_val) = padding_value.value {
                    let padding_ty_str = padding_value
                        .ty
                        .get_cuda_tile_type_str()
                        .unwrap_or_default();
                    let result_ty_str = format!("{}", tile_result_ty);

                    let promoted_padding = if padding_ty_str.contains("tile<")
                        && !padding_ty_str.contains("x")
                        && result_ty_str.contains("x")
                    {
                        let ones_shape_ty = ir::Type::parse(
                            &self.context,
                            &format!(
                                "!cuda_tile.tile<1x{}>",
                                padding_ty_str
                                    .split('<')
                                    .nth(1)
                                    .unwrap()
                                    .trim_end_matches('>')
                            ),
                        )
                        .unwrap();

                        let reshape_op = OperationBuilder::new(
                            "cuda_tile.reshape",
                            Location::unknown(&self.context),
                        )
                        .add_results(&[ones_shape_ty])
                        .add_operands(&[padding_val])
                        .build()
                        .unwrap();
                        let reshape_ref = builder.append_operation(reshape_op);
                        let reshaped = reshape_ref.result(0).unwrap().into();

                        let broadcast_op = OperationBuilder::new(
                            "cuda_tile.broadcast",
                            Location::unknown(&self.context),
                        )
                        .add_results(&[tile_result_ty])
                        .add_operands(&[reshaped])
                        .build()
                        .unwrap();
                        let broadcast_ref = builder.append_operation(broadcast_op);
                        broadcast_ref.result(0).unwrap().into()
                    } else {
                        padding_val
                    };

                    operands.push(promoted_padding);
                    padding_count = 1;
                }
            }
        }

        // arg[5]: token (Option<Token>)
        if let Some(token_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[5], ctx)
        {
            if let Some(token_value) =
                self.compile_expression(builder, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        // arg[6]: latency (Option<i32>) — literal or const generic
        let mut hint_params: HashMap<String, i32> = HashMap::new();
        if let Some(latency_arg) =
            crate::compiler::utils::resolve_option_arg(&call_expr.args[6], ctx)
        {
            if let Expr::Lit(ExprLit {
                lit: Lit::Int(int_lit),
                ..
            }) = &latency_arg
            {
                hint_params.insert(
                    "latency".to_string(),
                    int_lit.base10_parse::<i32>().unwrap(),
                );
            } else if let Expr::Path(path) = &latency_arg {
                let name = path.path.segments.last().unwrap().ident.to_string();
                if let Some(&v) = generic_args.inst_i32.get(&name) {
                    hint_params.insert("latency".to_string(), v);
                }
            }
        }

        let mut opt_hints = vec![];
        if let Some(load_store_hints_attr) = self
            .optimization_hints
            .get_load_store_hints(&self.context, hint_params)?
        {
            opt_hints.push(load_store_hints_attr);
        }

        let operand_segments = format!(
            "array<i32: 1, {}, {}, {}>",
            mask_count, padding_count, token_count
        );

        let mut op_builder =
            OperationBuilder::new("cuda_tile.load_ptr_tko", Location::unknown(&self.context))
                .add_results(&[tile_result_ty, token_result_ty])
                .add_operands(&operands)
                .add_attributes(&opt_hints)
                .add_attributes(&[(
                    Identifier::new(&self.context, "memory_ordering_semantics"),
                    IntegerAttribute::new(
                        ir::Type::parse(&self.context, "i32").unwrap(),
                        memory_ordering_value,
                    )
                    .into(),
                )]);

        if memory_ordering != "weak" {
            op_builder = op_builder.add_attributes(&[(
                Identifier::new(&self.context, "memory_scope"),
                IntegerAttribute::new(
                    ir::Type::parse(&self.context, "i32").unwrap(),
                    memory_scope_value,
                )
                .into(),
            )]);
        }

        let op = op_builder
            .add_attributes(&[(
                Identifier::new(&self.context, "operandSegmentSizes"),
                Attribute::parse(&self.context, &operand_segments).unwrap(),
            )])
            .build()
            .unwrap();

        let op_ref = builder.append_operation(op);
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`load_ptr_tko` MLIR verification failed: {}",
                    op_ref.to_string()
                ),
            );
        }

        let mut values = vec![];
        values.push(TileRustValue::<'c, 'c>::new_structured_type(
            op_ref.result(0).unwrap().into(),
            tile_elem_ty,
            None,
        ));
        values.push(TileRustValue::<'c, 'c>::new_primitive(
            op_ref.result(1).unwrap().into(),
            token_elem_ty,
            None,
        ));

        return Ok(Some(TileRustValue::<'c, 'c>::new_compound(
            values,
            return_type_outer,
        )));
    }

    fn compile_store_ptr_tko(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let token_result_ty = ir::Type::parse(&self.context, "!cuda_tile.token").unwrap();

        let dest_arg = &call_expr.args[0];
        let Some(dest_value) =
            self.compile_expression(builder, dest_arg, generic_args, ctx, None)?
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

        let value_arg = &call_expr.args[1];
        let Some(value_value) =
            self.compile_expression(builder, value_arg, generic_args, ctx, None)?
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

        let memory_ordering = crate::compiler::utils::extract_string_literal(
            &call_expr.args[2],
            "memory_ordering",
            ctx,
        )?;

        let memory_ordering_value = match memory_ordering.as_str() {
            "weak" => 0,
            "relaxed" => 1,
            "release" => 3,
            _ => return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "invalid `memory_ordering` for `store_ptr_tko: '{}'. Valid: weak, relaxed, release",
                    memory_ordering
                ),
            ),
        };

        let memory_scope = crate::compiler::utils::extract_string_literal(
            &call_expr.args[3],
            "memory_scope",
            ctx,
        )?;

        let memory_scope_value = match memory_scope.as_str() {
            "tl_blk" => 0,
            "device" => 1,
            "sys" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: `'{}'. Valid: tl_blk, device, sys",
                        memory_scope
                    ),
                )
            }
        };

        let mut operands = vec![dest_ptr, tile_value];
        let mut mask_count = 0;
        let mut token_count = 0;

        // arg[4]: mask
        if let Some(mask_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[4], ctx)
        {
            if let Some(mask_value) =
                self.compile_expression(builder, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }

        // arg[5]: token
        if let Some(token_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[5], ctx)
        {
            if let Some(token_value) =
                self.compile_expression(builder, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        // arg[6]: latency (Option<i32>) — literal or const generic
        let mut hint_params: HashMap<String, i32> = HashMap::new();
        if let Some(latency_arg) =
            crate::compiler::utils::resolve_option_arg(&call_expr.args[6], ctx)
        {
            if let Expr::Lit(ExprLit {
                lit: Lit::Int(int_lit),
                ..
            }) = &latency_arg
            {
                hint_params.insert(
                    "latency".to_string(),
                    int_lit.base10_parse::<i32>().unwrap(),
                );
            } else if let Expr::Path(path) = &latency_arg {
                let name = path.path.segments.last().unwrap().ident.to_string();
                if let Some(&v) = generic_args.inst_i32.get(&name) {
                    hint_params.insert("latency".to_string(), v);
                }
            }
        }

        let mut opt_hints = vec![];
        if let Some(load_store_hints_attr) = self
            .optimization_hints
            .get_load_store_hints(&self.context, hint_params)?
        {
            opt_hints.push(load_store_hints_attr);
        }

        let operand_segments = format!("array<i32: 1, 1, {}, {}>", mask_count, token_count);

        let mut op_builder =
            OperationBuilder::new("cuda_tile.store_ptr_tko", Location::unknown(&self.context))
                .add_results(&[token_result_ty])
                .add_operands(&operands)
                .add_attributes(&opt_hints)
                .add_attributes(&[(
                    Identifier::new(&self.context, "memory_ordering_semantics"),
                    IntegerAttribute::new(
                        ir::Type::parse(&self.context, "i32").unwrap(),
                        memory_ordering_value,
                    )
                    .into(),
                )]);

        if memory_ordering != "weak" {
            op_builder = op_builder.add_attributes(&[(
                Identifier::new(&self.context, "memory_scope"),
                IntegerAttribute::new(
                    ir::Type::parse(&self.context, "i32").unwrap(),
                    memory_scope_value,
                )
                .into(),
            )]);
        }

        let op = op_builder
            .add_attributes(&[(
                Identifier::new(&self.context, "operandSegmentSizes"),
                Attribute::parse(&self.context, &operand_segments).unwrap(),
            )])
            .build()
            .unwrap();

        let op_ref = builder.append_operation(op);
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`store_ptr_tko` MLIR verification failed: {}",
                    op_ref.to_string()
                ),
            );
        }
        let op_value: Value<'c, 'c> = op_ref.result(0).unwrap().into();
        let token_type = self
            .compile_type(&syn::parse_quote!(Token), generic_args, &HashMap::new())?
            .unwrap();
        return Ok(Some(TileRustValue::<'c, 'c>::new_primitive(
            op_value, token_type, None,
        )));
    }

    fn compile_atomic_rmw_tko(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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

        let tile_result_ty = tile_elem_ty.cuda_tile_ty.unwrap();
        let token_result_ty = token_elem_ty.cuda_tile_ty.unwrap();

        let ptr_arg = &call_expr.args[0];
        let Some(ptr_value) = self.compile_expression(builder, ptr_arg, generic_args, ctx, None)?
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

        let arg_arg = &call_expr.args[1];
        let Some(arg_value) = self.compile_expression(builder, arg_arg, generic_args, ctx, None)?
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

        let mode = crate::compiler::utils::extract_string_literal(&call_expr.args[2], "mode", ctx)?;
        let memory_ordering = crate::compiler::utils::extract_string_literal(
            &call_expr.args[3],
            "memory_ordering",
            ctx,
        )?;
        let memory_scope = crate::compiler::utils::extract_string_literal(
            &call_expr.args[4],
            "memory_scope",
            ctx,
        )?;

        let memory_ordering_value = match memory_ordering.as_str() {
            "relaxed" => 1,
            "acquire" => 2,
            "release" => 3,
            "acq_rel" => 4,
            "weak" => return self.jit_error_result(
                &call_expr.span(),
                "atomic_rmw_tko does not support 'weak' memory ordering. Valid: relaxed, acquire, release, acq_rel",
            ),
            _ => return self.jit_error_result(
                &call_expr.span(),
                &format!("invalid `memory_ordering` for `atomic_rmw_tko: '{}'. Valid: relaxed, acquire, release, acq_rel", memory_ordering),
            )
        };

        let memory_scope_value = match memory_scope.as_str() {
            "tl_blk" => 0,
            "device" => 1,
            "sys" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: `'{}'. Valid: tl_blk, device, sys",
                        memory_scope
                    ),
                )
            }
        };

        let elem_ty_prefix = ptr_value
            .ty
            .get_cuda_tile_element_type_prefix(&self.modules.primitives)?;
        let atomic_mode = AtomicMode::new(mode.as_str(), elem_ty_prefix)? as i64;

        let mut operands = vec![ptrs, arg];
        let mut mask_count = 0;
        let mut token_count = 0;

        if let Some(mask_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[5], ctx)
        {
            if let Some(mask_value) =
                self.compile_expression(builder, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }

        if let Some(token_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[6], ctx)
        {
            if let Some(token_value) =
                self.compile_expression(builder, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        let operand_segments = format!("array<i32: 1, 1, {}, {}>", mask_count, token_count);

        let op =
            OperationBuilder::new("cuda_tile.atomic_rmw_tko", Location::unknown(&self.context))
                .add_results(&[tile_result_ty, token_result_ty])
                .add_operands(&operands)
                .add_attributes(&[
                    (
                        Identifier::new(&self.context, "memory_ordering_semantics"),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            memory_ordering_value,
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(&self.context, "memory_scope"),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            memory_scope_value,
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(&self.context, "mode"),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            atomic_mode,
                        )
                        .into(),
                    ),
                ])
                .add_attributes(&[(
                    Identifier::new(&self.context, "operandSegmentSizes"),
                    Attribute::parse(&self.context, &operand_segments).unwrap(),
                )])
                .build()
                .unwrap();

        let op_ref = builder.append_operation(op);
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`atomic_rmw_tko` MLIR verification failed: {}",
                    op_ref.to_string()
                ),
            );
        }

        let mut values = vec![];
        values.push(TileRustValue::<'c, 'c>::new_structured_type(
            op_ref.result(0).unwrap().into(),
            tile_elem_ty,
            None,
        ));
        values.push(TileRustValue::<'c, 'c>::new_primitive(
            op_ref.result(1).unwrap().into(),
            token_elem_ty,
            None,
        ));

        return Ok(Some(TileRustValue::<'c, 'c>::new_compound(
            values,
            return_type_outer,
        )));
    }

    fn compile_atomic_cas_tko(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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

        let tile_result_ty = tile_elem_ty.cuda_tile_ty.unwrap();
        let token_result_ty = token_elem_ty.cuda_tile_ty.unwrap();

        let ptr_arg = &call_expr.args[0];
        let Some(ptr_value) = self.compile_expression(builder, ptr_arg, generic_args, ctx, None)?
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

        let cmp_arg = &call_expr.args[1];
        let Some(cmp_value) = self.compile_expression(builder, cmp_arg, generic_args, ctx, None)?
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

        let val_arg = &call_expr.args[2];
        let Some(val_value) = self.compile_expression(builder, val_arg, generic_args, ctx, None)?
        else {
            return self.jit_error_result(
                &call_expr.args[2].span(),
                "unable to compile value argument for `atomic_cas_tko",
            );
        };
        let Some(val) = val_value.value else {
            return self.jit_error_result(&call_expr.args[2].span(), "unable to compile value");
        };

        let memory_ordering = crate::compiler::utils::extract_string_literal(
            &call_expr.args[3],
            "memory_ordering",
            ctx,
        )?;
        let memory_scope = crate::compiler::utils::extract_string_literal(
            &call_expr.args[4],
            "memory_scope",
            ctx,
        )?;

        let memory_ordering_value = match memory_ordering.as_str() {
            "relaxed" => 1,
            "acquire" => 2,
            "release" => 3,
            "acq_rel" => 4,
            "weak" => return self.jit_error_result(
                &call_expr.span(),
                "atomic_cas_tko does not support 'weak' memory ordering. Valid: relaxed, acquire, release, acq_rel",
            ),
            _ => return self.jit_error_result(
                &call_expr.span(),
                &format!("invalid `memory_ordering` for `atomic_cas_tko: '{}'. Valid: relaxed, acquire, release, acq_rel", memory_ordering),
            )
        };

        let memory_scope_value = match memory_scope.as_str() {
            "tl_blk" => 0,
            "device" => 1,
            "sys" => 2,
            _ => {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!(
                        "invalid `memory_scope`: `'{}'. Valid: tl_blk, device, sys",
                        memory_scope
                    ),
                )
            }
        };

        let mut operands = vec![ptrs, cmp, val];
        let mut mask_count = 0;
        let mut token_count = 0;

        if let Some(mask_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[5], ctx)
        {
            if let Some(mask_value) =
                self.compile_expression(builder, &mask_arg, generic_args, ctx, None)?
            {
                if let Some(mask_val) = mask_value.value {
                    operands.push(mask_val);
                    mask_count = 1;
                }
            }
        }

        if let Some(token_arg) = crate::compiler::utils::resolve_option_arg(&call_expr.args[6], ctx)
        {
            if let Some(token_value) =
                self.compile_expression(builder, &token_arg, generic_args, ctx, None)?
            {
                if let Some(token_val) = token_value.value {
                    operands.push(token_val);
                    token_count = 1;
                }
            }
        }

        let operand_segments = format!("array<i32: 1, 1, 1, {}, {}>", mask_count, token_count);

        let op =
            OperationBuilder::new("cuda_tile.atomic_cas_tko", Location::unknown(&self.context))
                .add_results(&[tile_result_ty, token_result_ty])
                .add_operands(&operands)
                .add_attributes(&[
                    (
                        Identifier::new(&self.context, "memory_ordering_semantics"),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            memory_ordering_value,
                        )
                        .into(),
                    ),
                    (
                        Identifier::new(&self.context, "memory_scope"),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            memory_scope_value,
                        )
                        .into(),
                    ),
                ])
                .add_attributes(&[(
                    Identifier::new(&self.context, "operandSegmentSizes"),
                    Attribute::parse(&self.context, &operand_segments).unwrap(),
                )])
                .build()
                .unwrap();

        let op_ref = builder.append_operation(op);
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`atomic_cas_tko` MLIR verification failed: {}",
                    op_ref.to_string()
                ),
            );
        }

        let mut values = vec![];
        values.push(TileRustValue::<'c, 'c>::new_structured_type(
            op_ref.result(0).unwrap().into(),
            tile_elem_ty,
            None,
        ));
        values.push(TileRustValue::<'c, 'c>::new_primitive(
            op_ref.result(1).unwrap().into(),
            token_elem_ty,
            None,
        ));

        return Ok(Some(TileRustValue::<'c, 'c>::new_compound(
            values,
            return_type_outer,
        )));
    }

    fn compile_load_view_tko(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        cuda_tile_op_hint_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        if return_type.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "unable to infer call; add a return type annotation",
            );
        }
        let return_type = return_type.unwrap();
        if return_type.cuda_tile_ty.is_none() {
            return self.jit_error_result(
                &call_expr.span(),
                "Expected cuda_tile_ty for load_view_tko return type",
            );
        }
        let tile_result_ty = return_type.cuda_tile_ty.unwrap();
        let token_result_ty = ir::Type::parse(&self.context, "!cuda_tile.token").unwrap();
        let op_builder =
            OperationBuilder::new("cuda_tile.load_view_tko", Location::unknown(&self.context));

        let view_arg = &call_expr.args[0];
        let Some(view_value) =
            self.compile_expression(builder, view_arg, generic_args, ctx, None)?
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
        let index_value = self.compile_expression(builder, index_arg, generic_args, ctx, None)?;
        let index_value = index_value.unwrap();
        if index_value.values.is_none() {
            return self.jit_error_result(&call_expr.args[1].span(), "Expected values for index");
        }
        let index_values = index_value.values.as_ref().unwrap();
        let mut index_values_vec = Vec::new();
        for value in index_values.iter() {
            let Some(v) = value.value.clone() else {
                return self.jit_error_result(
                    &call_expr.args[1].span(),
                    &format!("Unexpected nested array {index_arg_str}"),
                );
            };
            index_values_vec.push(v);
        }
        let index_values = index_values_vec;

        let mut opt_hints = vec![];
        let mut hint_params: HashMap<String, i32> = HashMap::new();
        let fn_params = get_sig_param_names(&fn_item.sig);
        for hint_param in cuda_tile_op_hint_params {
            let Some(i) = fn_params.iter().position(|s| *s == *hint_param) else {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!("Failed to compile hint param {hint_param}"),
                );
            };
            // Handle Option<i32> hint params (e.g. latency: Option<i32>).
            // The inner value can be a literal (Some(4)) or a const generic (Some(L)).
            if let Some(inner) = crate::compiler::utils::resolve_option_arg(&call_expr.args[i], ctx)
            {
                let value: i32 = if let Expr::Lit(lit_expr) = &inner {
                    let Lit::Int(int_lit) = &lit_expr.lit else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            "Non-integer literals not supported",
                        );
                    };
                    int_lit.base10_parse::<i32>().unwrap()
                } else if let Expr::Path(path) = &inner {
                    let name = path.path.segments.last().unwrap().ident.to_string();
                    if let Some(&v) = generic_args.inst_i32.get(&name) {
                        v
                    } else {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            &format!(
                                "Failed to compile hint param {hint_param}: \
                                 '{name}' is not a literal or resolved const generic."
                            ),
                        );
                    }
                } else {
                    return self.jit_error_result(
                        &call_expr.args[i].span(),
                        &format!(
                            "Failed to compile hint param {hint_param}, \
                             expected literal or const generic."
                        ),
                    );
                };
                hint_params.insert(hint_param.to_string(), value);
            }
        }
        // Handle disallow_tma: bool parameter.
        if let Some(i) = fn_params.iter().position(|s| s == "disallow_tma") {
            if let Expr::Lit(ExprLit {
                lit: Lit::Bool(b), ..
            }) = &call_expr.args[i]
            {
                if b.value {
                    hint_params.insert("allow_tma".to_string(), 0);
                }
            }
        }
        if let Some(load_store_hints_attr) = self
            .optimization_hints
            .get_load_store_hints(&self.context, hint_params)?
        {
            opt_hints.push(load_store_hints_attr);
        }
        let op = op_builder
            .add_results(&[tile_result_ty, token_result_ty])
            .add_operands(&[cuda_tile_view_value])
            .add_operands(&index_values)
            .add_operands(&[cuda_tile_token])
            .add_attributes(&opt_hints)
            .add_attributes(&[(
                Identifier::new(&self.context, "memory_ordering_semantics"),
                IntegerAttribute::new(ir::Type::parse(&self.context, "i32").unwrap(), 0).into(),
            )])
            .add_attributes(&[(
                Identifier::new(&self.context, "operandSegmentSizes"),
                Attribute::parse(
                    &self.context,
                    format!("array<{}: 1, {}, 1>", "i32", index_values.len()).as_str(),
                )
                .unwrap(),
            )])
            .build()
            .unwrap();
        let op_ref = builder.append_operation(op);
        let new_token: Value = op_ref.result(1).unwrap().into();
        let _old = update_token(view_arg, new_token, ctx);
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`load_view_tko` MLIR verification failed: {}",
                    op_ref.to_string()
                ),
            );
        }
        let op_value: Value<'c, 'c> = op_ref.result(0).unwrap().into();
        return Ok(Some(TileRustValue::<'c, 'c>::new_structured_type(
            op_value,
            return_type,
            None,
        )));
    }

    fn compile_store_view_tko(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        fn_item: &ItemFn,
        cuda_tile_op_hint_params: &[String],
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let token_result_ty = ir::Type::parse(&self.context, "!cuda_tile.token").unwrap();
        let op_builder =
            OperationBuilder::new("cuda_tile.store_view_tko", Location::unknown(&self.context));

        let view_arg = &call_expr.args[0];
        let Some(mut view_value) =
            self.compile_expression(builder, view_arg, generic_args, ctx, None)?
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

        let tile_arg = &call_expr.args[1];
        let tile_value = self.compile_expression(builder, tile_arg, generic_args, ctx, None)?;
        let tile_value = tile_value.unwrap();
        if tile_value.value.is_none() {
            return self.jit_error_result(
                &call_expr.args[2].span(),
                "Expected value for tile in store_view_tko",
            );
        }
        let tile_value = tile_value.value.unwrap();

        let index_arg = &call_expr.args[2];
        let index_arg_str = index_arg.to_token_stream().to_string();
        let index_value = self.compile_expression(builder, index_arg, generic_args, ctx, None)?;
        let index_value = index_value.unwrap();
        if index_value.values.is_none() {
            return self.jit_error_result(&call_expr.args[2].span(), "Expected values for index");
        }
        let index_values = index_value.values.as_ref().unwrap();
        let mut index_values_vec = Vec::new();
        for value in index_values.iter() {
            let Some(v) = value.value.clone() else {
                return self.jit_error_result(
                    &call_expr.args[2].span(),
                    &format!("Unexpected nested array {index_arg_str}"),
                );
            };
            index_values_vec.push(v);
        }
        let index_values = index_values_vec;

        let mut opt_hints = vec![];
        let mut hint_params: HashMap<String, i32> = HashMap::new();
        let fn_params = get_sig_param_names(&fn_item.sig);
        for hint_param in cuda_tile_op_hint_params {
            let Some(i) = fn_params.iter().position(|s| *s == *hint_param) else {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!("Failed to compile hint param {hint_param}"),
                );
            };
            // Handle Option<i32> hint params (e.g. latency: Option<i32>).
            if let Some(inner) = crate::compiler::utils::resolve_option_arg(&call_expr.args[i], ctx)
            {
                let Expr::Lit(lit_expr) = &inner else {
                    return self.jit_error_result(
                        &call_expr.args[i].span(),
                        &format!("Failed to compile hint param {hint_param}, expected literal."),
                    );
                };
                let Lit::Int(int_lit) = &lit_expr.lit else {
                    return self.jit_error_result(
                        &lit_expr.span(),
                        "Non-integer hint param literals not supported",
                    );
                };
                hint_params.insert(
                    hint_param.to_string(),
                    int_lit.base10_parse::<i32>().unwrap(),
                );
            }
        }
        // Handle disallow_tma: bool parameter.
        if let Some(i) = fn_params.iter().position(|s| s == "disallow_tma") {
            if let Expr::Lit(ExprLit {
                lit: Lit::Bool(b), ..
            }) = &call_expr.args[i]
            {
                if b.value {
                    hint_params.insert("allow_tma".to_string(), 0);
                }
            }
        }
        if let Some(load_store_hints_attr) = self
            .optimization_hints
            .get_load_store_hints(&self.context, hint_params)?
        {
            opt_hints.push(load_store_hints_attr);
        }
        let op = op_builder
            .add_results(&[token_result_ty])
            .add_operands(&[tile_value])
            .add_operands(&[cuda_tile_view_value.clone()])
            .add_operands(&index_values)
            .add_operands(&[cuda_tile_token.clone()])
            .add_attributes(&opt_hints)
            .add_attributes(&[(
                Identifier::new(&self.context, "memory_ordering_semantics"),
                IntegerAttribute::new(ir::Type::parse(&self.context, "i32").unwrap(), 0).into(),
            )])
            .add_attributes(&[(
                Identifier::new(&self.context, "operandSegmentSizes"),
                Attribute::parse(
                    &self.context,
                    format!("array<{}: 1, 1, {}, 1>", "i32", index_values.len()).as_str(),
                )
                .unwrap(),
            )])
            .build()
            .unwrap();
        let op_ref = builder.append_operation(op);
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`store_view_tko` MLIR verification failed: {}",
                    op_ref.to_string()
                ),
            );
        }
        let new_token: Value = op_ref.result(0).unwrap().into();
        let _old = update_token(view_arg, new_token, ctx);
        let Some(var_arg_ident) = get_ident_from_expr(view_arg) else {
            return self.jit_error_result(&view_arg.span(), "Unexpected expression");
        };
        let Some(result) = ctx.vars.get(var_arg_ident.to_string().as_str()) else {
            return self.jit_error_result(
                &view_arg.span(),
                &format!("Unexpected state: Expected {var_arg_ident} in ctx"),
            );
        };
        return Ok(Some(result.clone()));
    }

    fn compile_reduce_op(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        _return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let location = Location::unknown(&self.context);

        let operand_arg = &call_expr.args[0];
        let operand_value = self
            .compile_expression(builder, operand_arg, generic_args, ctx, None)?
            .unwrap();

        let elem_ty_str = operand_value
            .ty
            .get_cuda_tile_element_type(&self.modules.primitives)?
            .unwrap();
        let elem_ty =
            ir::Type::parse(&self.context, &format!("!cuda_tile.tile<{}>", elem_ty_str)).unwrap();

        let elem_rust_ty = operand_value
            .ty
            .type_instance
            .get_rust_element_instance_ty()
            .unwrap();
        let elem_rust_ty_parsed = syn::parse2::<Type>(elem_rust_ty.parse().unwrap()).unwrap();

        let elem_compiled_ty = self
            .compile_type(&elem_rust_ty_parsed, generic_args, &HashMap::new())?
            .unwrap();

        let scalar_tile_type_str = format!("Tile<{}, {{[]}}>", elem_rust_ty);
        let scalar_tile_rust_ty =
            syn::parse2::<Type>(scalar_tile_type_str.parse().unwrap()).unwrap();
        let return_type_inner = self
            .compile_type(&scalar_tile_rust_ty, generic_args, &HashMap::new())?
            .unwrap();
        let result_ty = elem_ty;
        let operand_tile = operand_value.value.unwrap();

        let reduce_block = Block::new(&[(elem_ty, location), (elem_ty, location)]);
        let arg0: Value = reduce_block.argument(0).unwrap().into();
        let arg1: Value = reduce_block.argument(1).unwrap().into();

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
            let param0_name = &closure_info.params[0].name;
            let param1_name = &closure_info.params[1].name;

            closure_variables.vars.insert(
                param0_name.clone(),
                TileRustValue::new_value_kind_like(arg0, elem_compiled_ty.clone()),
            );
            closure_variables.vars.insert(
                param1_name.clone(),
                TileRustValue::new_value_kind_like(arg1, elem_compiled_ty.clone()),
            );

            let result_value = self
                .compile_expression(
                    &reduce_block,
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
                    "Closure body must return a value with an MLIR value",
                );
            }
            result_value.value.unwrap()
        } else {
            let add_op_name = if elem_ty_str.starts_with('f') {
                "cuda_tile.addf"
            } else {
                "cuda_tile.addi"
            };
            let mut add_op_builder = OperationBuilder::new(add_op_name, location)
                .add_results(&[elem_ty])
                .add_operands(&[arg0, arg1]);

            if elem_ty_str.starts_with('f') {
                let rounding_attr =
                    self.parse_named_attr("rounding_mode", "#cuda_tile.rounding<nearest_even>")?;
                add_op_builder = add_op_builder.add_attributes(&[rounding_attr]);
            }

            let add_op = add_op_builder.build().unwrap();
            let add_op_ref = reduce_block.append_operation(add_op);
            add_op_ref.result(0).unwrap().into()
        };

        let yield_op = OperationBuilder::new("cuda_tile.yield", location)
            .add_operands(&[reduction_result])
            .build()
            .unwrap();
        reduce_block.append_operation(yield_op);

        let region = Region::new();
        region.append_block(reduce_block);

        let identity_val_str = if elem_ty_str.starts_with('f') {
            "0.0"
        } else {
            "0"
        };
        let identity_elem_attr = Attribute::parse(
            &self.context,
            &format!("{} : {}", identity_val_str, elem_ty_str),
        )
        .unwrap();
        let identities_attr =
            ir::attribute::ArrayAttribute::new(&self.context, &[identity_elem_attr]);

        let op = cuda_tile::ReduceOperationBuilder::new(&self.context, location)
            .results(&[result_ty])
            .operands(&[operand_tile])
            .body(region)
            .dim(IntegerAttribute::new(
                ir::Type::parse(&self.context, "i32").unwrap(),
                0,
            ))
            .identities(identities_attr)
            .build();

        let op_ref = builder.append_operation(op.into());
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!("`reduce` MLIR verification failed: {}", op_ref.to_string()),
            );
        }
        let op_value: Value<'c, 'c> = op_ref.result(0).unwrap().into();
        return Ok(Some(TileRustValue::<'c, 'c>::new_structured_type(
            op_value,
            return_type_inner,
            None,
        )));
    }

    fn compile_scan_op(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let location = Location::unknown(&self.context);

        let operand_arg = &call_expr.args[0];
        let operand_value = self
            .compile_expression(builder, operand_arg, generic_args, ctx, None)?
            .unwrap();
        let operand_tile = operand_value.value.unwrap();

        let return_type = operand_value.ty.clone();
        let result_ty = operand_value.ty.cuda_tile_ty.unwrap();

        let elem_ty_str = operand_value
            .ty
            .get_cuda_tile_element_type(&self.modules.primitives)?
            .unwrap();
        let elem_ty =
            ir::Type::parse(&self.context, &format!("!cuda_tile.tile<{}>", elem_ty_str)).unwrap();

        let elem_rust_ty = operand_value
            .ty
            .type_instance
            .get_rust_element_instance_ty()
            .unwrap();
        let elem_rust_ty = syn::parse2::<Type>(elem_rust_ty.parse().unwrap()).unwrap();
        let elem_compiled_ty = self
            .compile_type(&elem_rust_ty, generic_args, &HashMap::new())?
            .unwrap();

        let scan_block = Block::new(&[(elem_ty, location), (elem_ty, location)]);
        let arg0: Value = scan_block.argument(0).unwrap().into();
        let arg1: Value = scan_block.argument(1).unwrap().into();

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
            let param0_name = &closure_info.params[0].name;
            let param1_name = &closure_info.params[1].name;

            closure_variables.vars.insert(
                param0_name.clone(),
                TileRustValue::new_value_kind_like(arg0, elem_compiled_ty.clone()),
            );
            closure_variables.vars.insert(
                param1_name.clone(),
                TileRustValue::new_value_kind_like(arg1, elem_compiled_ty.clone()),
            );

            let result_value = self
                .compile_expression(
                    &scan_block,
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
                    "Closure body must return a value with an MLIR value",
                );
            }
            result_value.value.unwrap()
        } else {
            let add_op_name = if elem_ty_str.starts_with('f') {
                "cuda_tile.addf"
            } else {
                "cuda_tile.addi"
            };
            let mut add_op_builder = OperationBuilder::new(add_op_name, location)
                .add_results(&[elem_ty])
                .add_operands(&[arg0, arg1]);

            if elem_ty_str.starts_with('f') {
                let rounding_attr =
                    self.parse_named_attr("rounding_mode", "#cuda_tile.rounding<nearest_even>")?;
                add_op_builder = add_op_builder.add_attributes(&[rounding_attr]);
            }

            let add_op = add_op_builder.build().unwrap();
            let add_op_ref = scan_block.append_operation(add_op);
            add_op_ref.result(0).unwrap().into()
        };

        let yield_op = OperationBuilder::new("cuda_tile.yield", location)
            .add_operands(&[scan_result])
            .build()
            .unwrap();
        scan_block.append_operation(yield_op);

        let region = Region::new();
        region.append_block(scan_block);

        let identity_str = if elem_ty_str.starts_with('f') {
            "0.0"
        } else {
            "0"
        };
        let identity_elem_attr = Attribute::parse(
            &self.context,
            &format!("{} : {}", identity_str, elem_ty_str),
        )
        .unwrap();
        let identities_attr =
            ir::attribute::ArrayAttribute::new(&self.context, &[identity_elem_attr]);

        let reverse_arg = &call_expr.args[2];
        let reverse_value = if let Expr::Lit(lit_expr) = reverse_arg {
            if let syn::Lit::Bool(lit_bool) = &lit_expr.lit {
                lit_bool.value
            } else {
                false
            }
        } else {
            false
        };
        let reverse_attr =
            Attribute::parse(&self.context, if reverse_value { "true" } else { "false" }).unwrap();

        let op = cuda_tile::ScanOperationBuilder::new(&self.context, location)
            .results(&[result_ty])
            .operands(&[operand_tile])
            .body(region)
            .dim(IntegerAttribute::new(
                ir::Type::parse(&self.context, "i32").unwrap(),
                0,
            ))
            .reverse(reverse_attr)
            .identities(identities_attr)
            .build();

        let op_ref = builder.append_operation(op.into());
        if !op_ref.verify() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!("`scan` MLIR verification failed: {}", op_ref.to_string()),
            );
        }
        let op_value: Value<'c, 'c> = op_ref.result(0).unwrap().into();
        return Ok(Some(TileRustValue::<'c, 'c>::new_structured_type(
            op_value,
            return_type,
            None,
        )));
    }

    /// General-purpose op compilation for CUDA Tile dialect operations that follow
    /// the standard pattern (operands from params, attributes, results from return type).
    fn compile_general_op(
        &'c self,
        builder: &'c ir::Block<'c>,
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
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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
            match rust_function_name.as_str() {
                "constant" => {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "Return type required for {}",
                            call_expr.to_token_stream().to_string()
                        ),
                    )
                }
                _ => {}
            }
            self.derive_type(
                builder,
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

        // TODO (hme): This can be easily optimized.
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
                for i in 0..param_names.len() {
                    if param_names[i] == field_meta_expr_param {
                        let call_expr_arg = &call_expr.args[i];
                        let call_expr_arg_str = call_expr_arg.to_token_stream().to_string();
                        let final_expr_str =
                            field_meta_expr_str.replace(field_meta_expr_param, &call_expr_arg_str);
                        let final_expr =
                            syn::parse2::<Expr>(final_expr_str.parse().unwrap()).unwrap();
                        let op_arg =
                            self.compile_expression(builder, &final_expr, generic_args, ctx, None)?;
                        if op_arg.is_none() {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("Failed to compile type meta {field_meta_expr_str} via expr {final_expr_str}"),
                            );
                        }
                        let op_arg = op_arg.unwrap();
                        meta.fields.insert(field_meta_expr_str.clone(), op_arg);
                        succeeded = true;
                    }
                }
                if !succeeded {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unable to find param {field_meta_expr_param}, which was derived from type meta field for type meta {field_meta_expr_str}"),
                    );
                }
            }
            type_meta = Some(meta);
        };

        // Add args (operands).
        let mut operand_lengths = vec![];
        let mut op_builder = OperationBuilder::new(op_name, self.function_location());
        let mut compiled_args: Vec<TileRustValue<'c, 'c>> = Vec::new();
        for i in 0..cuda_tile_op_params.len() {
            let call_expr_arg = &call_expr.args[i];
            let call_expr_arg_str = call_expr_arg.to_token_stream().to_string();
            let op_arg =
                self.compile_expression(builder, call_expr_arg, generic_args, ctx, None)?;
            if op_arg.is_none() {
                return self
                    .jit_error_result(&call_expr.args[i].span(), "Failed to compile op arg");
            }
            let op_arg = op_arg.unwrap();
            compiled_args.push(op_arg.clone());
            let op_param = &cuda_tile_op_params[i];
            let mut arg_values = vec![];
            if op_arg.value.is_some() {
                arg_values.push(op_arg.value.clone().unwrap());
            } else if op_arg.fields.is_some() {
                let fields = op_arg.fields.as_ref().unwrap();
                let op_path = op_param.split(".").collect::<Vec<&str>>();
                if op_path.len() <= 1 {
                    return self.jit_error_result(
                        &call_expr.args[i].span(),
                        &format!("Field expression required for struct param {call_expr_arg_str}, got {op_param}"),
                    );
                }
                let field = *op_path.last().clone().unwrap();
                match fields.get(field) {
                    Some(field_value) => {
                        if field_value.value.is_some() {
                            arg_values.push(field_value.value.clone().unwrap());
                        } else if field_value.values.is_some() {
                            for value in field_value.values.as_ref().unwrap().iter() {
                                let Some(v) = value.value.clone() else {
                                    return self.jit_error_result(
                                        &call_expr.args[i].span(),
                                        &format!("Unexpected nested array {op_param} for {call_expr_arg_str}"),
                                    );
                                };
                                arg_values.push(v);
                            }
                        } else if field_value.fields.is_some() {
                            return self.jit_error_result(
                                &call_expr.args[i].span(),
                                &format!(
                                    "Unexpected nested struct {op_param} for {call_expr_arg_str}"
                                ),
                            );
                        } else {
                            return self.jit_error_result(
                                &call_expr.args[i].span(),
                                &format!("Unexpected op param {op_param} for {call_expr_arg_str}"),
                            );
                        }
                    }
                    None => {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            &format!("Failed to access field {op_param} for {call_expr_arg_str}"),
                        )
                    }
                }
            } else if op_arg.values.is_some() {
                let values = op_arg.values.as_ref().unwrap();
                for value in values.iter() {
                    let Some(v) = value.value.clone() else {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            &format!("Unexpected nested array {op_param} for {call_expr_arg_str}"),
                        );
                    };
                    arg_values.push(v);
                }
            } else {
                return self.jit_error_result(
                    &call_expr.args[i].span(),
                    &format!("Unexpected op param {op_param} for {call_expr_arg_str}"),
                );
            }
            operand_lengths.push(arg_values.len().to_string());
            op_builder = op_builder.add_operands(&arg_values);
        }

        // Add attribute flags.
        for named_attr in cuda_tile_op_named_attributes.iter() {
            let name_attr_split = named_attr.split("=").collect::<Vec<&str>>();
            let (attr_name, attr_value) = (name_attr_split[0], name_attr_split[1]);

            if attr_name.starts_with("signedness") && attr_value == "inferred_signedness" {
                let elem_ty = compiled_args
                    .get(0)
                    .and_then(|arg| {
                        arg.ty
                            .get_instantiated_rust_element_type(&self.modules.primitives)
                    })
                    .expect("Failed to get element type for signedness inference.");
                for arg in &compiled_args {
                    let arg_elem_ty = arg
                        .ty
                        .get_instantiated_rust_element_type(&self.modules.primitives)
                        .expect("Operand (and output?) types are not all equivalent.");
                    if arg_elem_ty != elem_ty {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!("Element type mismatch for signedness inference: expected {elem_ty}, got {arg_elem_ty}"),
                        );
                    }
                }
                op_builder = op_builder.add_attributes(&[get_signedness_attr(
                    &self.context,
                    attr_name,
                    elem_ty.as_str(),
                )?]);
            } else {
                op_builder =
                    op_builder.add_attributes(&[self.parse_named_attr(attr_name, attr_value)?]);
            }
        }

        // Resolve static_params: ZST marker types → MLIR attributes.
        let resolved_static_attrs =
            resolve_static_params(cuda_tile_op_static_params, call_expr, fn_item)
                .map_err(|e| JITError::Generic(e))?;
        for attr_str in &resolved_static_attrs {
            let parts = attr_str.splitn(2, '=').collect::<Vec<&str>>();
            if parts.len() == 2 {
                op_builder = op_builder
                    .add_attributes(&[self.parse_named_attr(parts[0].trim(), parts[1].trim())?]);
            }
        }

        // Add attributes.
        let mut cuda_tile_op_attr_params_iter = cuda_tile_op_attribute_params.iter();
        let mut maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
        let fn_params = get_sig_param_names(&fn_item.sig);
        for i in 0..fn_params.len() {
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
                    if attr_id != fn_params[i] {
                        continue;
                    }
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let call_expr_arg = &call_expr.args[i];
                    let call_expr_arg_str = call_expr_arg.to_token_stream().to_string();
                    let op_arg =
                        self.compile_expression(builder, call_expr_arg, generic_args, ctx, None)?;
                    if op_arg.is_none() {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
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
                                        &call_expr.args[i].span(),
                                        "Failed to build attribute",
                                    );
                                };
                                op_builder = op_builder.add_attributes(&[named_array_attr(
                                    &self.context,
                                    attr_id,
                                    &cga,
                                )]);
                            }
                            _ => {
                                return self.jit_error_result(
                                    &call_expr.args[i].span(),
                                    "Attribute type not implemented.",
                                )
                            }
                        },
                        _ => {
                            return self.jit_error_result(
                                &call_expr.args[i].span(),
                                &format!("Unexpected call arg {call_expr_arg_str} for {next_attr}"),
                            )
                        }
                    }
                }
                "dense" => {
                    if attr_id != fn_params[i] {
                        continue;
                    }
                    let (lit_value, _mlir_lit_ty) = match &call_expr.args[i] {
                        Expr::Lit(lit_expr) => match &lit_expr.lit {
                            Lit::Bool(bool_lit) => (bool_lit.value.to_string(), "i1".to_string()),
                            Lit::Int(int_lit) => {
                                (int_lit.base10_digits().to_string(), "i32".to_string())
                            }
                            Lit::Float(float_lit) => {
                                (float_lit.base10_digits().to_string(), "f32".to_string())
                            }
                            _ => {
                                return self.jit_error_result(
                                    &call_expr.args[i].span(),
                                    "Constant not supported",
                                )
                            }
                        },
                        Expr::Unary(unary_expr) => {
                            let UnOp::Neg(_) = unary_expr.op else {
                                return self.jit_error_result(
                                    &call_expr.args[i].span(),
                                    "Only unary negation is supported for constant values",
                                );
                            };
                            match &*unary_expr.expr {
                                Expr::Lit(lit_expr) => match &lit_expr.lit {
                                    Lit::Int(int_lit) => {
                                        (format!("-{}", int_lit.base10_digits()), "i32".to_string())
                                    }
                                    Lit::Float(float_lit) => (
                                        format!("-{}", float_lit.base10_digits()),
                                        "f32".to_string(),
                                    ),
                                    _ => {
                                        return self.jit_error_result(
                                            &call_expr.args[i].span(),
                                            "Unsupported literal type for negation",
                                        )
                                    }
                                },
                                _ => {
                                    return self.jit_error_result(
                                        &call_expr.args[i].span(),
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
                            let (ty, const_val) =
                                (ty_val_split[0].to_string(), ty_val_split[1].to_string());
                            match const_val.as_str() {
                                "ZERO" => (get_const_hex(ty.as_str(), "zero")?, ty.clone()),
                                "ONE" => (get_const_hex(ty.as_str(), "one")?, ty.clone()),
                                "NEG_INFINITY" => (get_const_hex(ty.as_str(), "min")?, ty.clone()),
                                "INFINITY" => (get_const_hex(ty.as_str(), "max")?, ty.clone()),
                                "E" => (get_const_hex(ty.as_str(), "e")?, ty.clone()),
                                _ => {
                                    return self.jit_error_result(
                                        &call_expr.args[i].span(),
                                        "Constant not supported",
                                    )
                                }
                            }
                        }
                        _ => {
                            return self.jit_error_result(
                                &call_expr.args[i].span(),
                                "Unsupported expression for named attribute.",
                            )
                        }
                    };
                    if return_type.cuda_tile_ty.is_none() {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "return type is missing a compiled tile type",
                        );
                    }
                    let result = return_type.cuda_tile_ty.unwrap();
                    let cuda_tile_tile_ty = result.to_string();
                    if !cuda_tile_tile_ty.starts_with("!cuda_tile.tile") {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "Unexpected type for dense attribute",
                        );
                    }
                    let attr = Attribute::parse(
                        &self.context,
                        format!("dense<{lit_value}> : {cuda_tile_tile_ty}").as_str(),
                    );
                    if attr.is_none() {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            &format!(
                                "Attribute parse failed: dense<{lit_value}> : {cuda_tile_tile_ty}"
                            ),
                        );
                    }
                    op_builder = op_builder.add_attributes(&[(
                        Identifier::new(&self.context, "value"),
                        attr.unwrap(),
                    )]);
                }
                "rounding" => {
                    if attr_id != fn_params[i] {
                        continue;
                    }
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let rounding_mode_str = match &call_expr.args[i] {
                        Expr::Lit(ExprLit {
                            lit: Lit::Str(lit_str),
                            ..
                        }) => lit_str.value(),
                        _ => {
                            return self.jit_error_result(
                                &call_expr.args[i].span(),
                                "Rounding mode must be a string literal. Valid values: \
                             \"nearest_even\", \"positive_inf\", \"negative_inf\", \
                             \"nearest_int_to_zero\", \"approx\".",
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
                            &call_expr.args[i].span(),
                            &format!(
                                "Invalid rounding mode: \"{}\". Valid values are: {}",
                                rounding_mode_str,
                                VALID_MODES.join(", ")
                            ),
                        );
                    }
                    let rounding_mode_attr = self.parse_named_attr(
                        "rounding_mode",
                        &format!("#cuda_tile.rounding<{}>", rounding_mode_str),
                    )?;
                    op_builder = op_builder.add_attributes(&[rounding_mode_attr]);
                }
                "memory_ordering" => {
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let attr = (
                        Identifier::new(&self.context, attr_id),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            1, // relaxed
                        )
                        .into(),
                    );
                    op_builder = op_builder.add_attributes(&[attr]);
                }
                "memory_scope" => {
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let attr = (
                        Identifier::new(&self.context, attr_id),
                        IntegerAttribute::new(
                            ir::Type::parse(&self.context, "i32").unwrap(),
                            1, // device
                        )
                        .into(),
                    );
                    op_builder = op_builder.add_attributes(&[attr]);
                }
                "integer" => {
                    if attr_id != fn_params[i] {
                        continue;
                    }
                    maybe_next_attr_param = cuda_tile_op_attr_params_iter.next();
                    let call_expr_arg = &call_expr.args[i];
                    let op_arg =
                        self.compile_expression(builder, call_expr_arg, generic_args, ctx, None)?;
                    if op_arg.is_none() {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            &format!("Failed to compile integer attribute {attr_id}"),
                        );
                    }
                    let op_arg = op_arg.unwrap();
                    if op_arg.value.is_none() {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
                            &format!("Integer attribute {attr_id} must be a value"),
                        );
                    }
                    if let Some(bounds) = op_arg.bounds {
                        if bounds.is_exact() {
                            let const_val = bounds.start as i64;
                            let int_attr = IntegerAttribute::new(
                                ir::Type::parse(&self.context, "i64").unwrap(),
                                const_val,
                            );
                            op_builder = op_builder.add_attributes(&[(
                                Identifier::new(&self.context, attr_id),
                                int_attr.into(),
                            )]);
                        } else {
                            return self.jit_error_result(
                                &call_expr.args[i].span(),
                                &format!("Integer attribute {attr_id} must be a constant value, got bounds: {bounds:?}"),
                            );
                        }
                    } else {
                        return self.jit_error_result(
                            &call_expr.args[i].span(),
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
            let operand_type = "i32";
            op_builder = op_builder.add_attributes(&[(
                Identifier::new(&self.context, "operandSegmentSizes"),
                Attribute::parse(
                    &self.context,
                    format!("array<{}: {}>", operand_type, operand_lengths.join(",")).as_str(),
                )
                .unwrap(),
            )])
        };

        // Add results.
        if function_returns(fn_item) {
            match return_type.kind {
                Kind::PrimitiveType | Kind::StructuredType => {
                    if return_type.cuda_tile_ty.is_none() {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "return type is missing a compiled tile type",
                        );
                    }
                    let result = return_type.cuda_tile_ty.unwrap();
                    let op = op_builder.add_results(&[result]).build().unwrap();
                    let op_ref = builder.append_operation(op);
                    if !op_ref.verify() {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!(
                                "op_ref={},\nret_ty={}",
                                op_ref.to_string(),
                                return_type.rust_ty.to_token_stream().to_string()
                            ),
                        );
                    }
                    let op_value: Value<'c, 'c> = op_ref.result(0).unwrap().into();
                    match return_type.kind {
                        Kind::PrimitiveType => Ok(Some(TileRustValue::<'c, 'c>::new_primitive(
                            op_value,
                            return_type,
                            None,
                        ))),
                        Kind::StructuredType => {
                            Ok(Some(TileRustValue::<'c, 'c>::new_structured_type(
                                op_value,
                                return_type,
                                type_meta,
                            )))
                        }
                        _ => unreachable!(),
                    }
                }
                Kind::Compound => {
                    if let Type::Tuple(tuple_type) = &return_type.rust_ty {
                        let mut elem_types = vec![];
                        for elem in &tuple_type.elems {
                            let elem_ty =
                                self.compile_type(&elem, generic_args, &HashMap::new())?;
                            if elem_ty.is_none() {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    "failed to compile type",
                                );
                            }
                            let elem_ty = elem_ty.unwrap();
                            if elem_ty.cuda_tile_ty.is_none() {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    "failed to compile tile type",
                                );
                            }
                            let elem_ty_result = elem_ty.cuda_tile_ty.unwrap();
                            op_builder = op_builder.add_results(&[elem_ty_result]);
                            elem_types.push(elem_ty);
                        }

                        let op = op_builder.build().unwrap();
                        let op_ref = builder.append_operation(op);
                        if !op_ref.verify() {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("compound type operation MLIR verification failed: {}", op_ref.to_string()),
                            );
                        }
                        let mut values = vec![];
                        for (i, elem_ty) in elem_types.iter().enumerate() {
                            match elem_ty.kind {
                                Kind::PrimitiveType => {
                                    let op_value: Value<'c, 'c> = if op_name == "cuda_tile.get_num_tile_blocks" || op_name == "cuda_tile.get_tile_block_id" {
                                        let op_value: Value<'c, 'c> = op_ref.result(i).unwrap().into();
                                        self.compile_value_assumption(builder, op_value, "assume_bounds_lower", &[0], elem_ty.clone(), &call_expr.span())?.value
                                            .expect("Expected a value from compiled assumption.")
                                    } else {
                                        op_ref.result(i).unwrap().into()
                                    };
                                    let maybe_bounds = if let Some(const_grid) = self.const_grid {
                                        if op_name == "cuda_tile.get_num_tile_blocks" {
                                            let const_block = match i {
                                                0 => const_grid.0,
                                                1 => const_grid.1,
                                                2 => const_grid.2,
                                                _ => unreachable!("Impossible")
                                            };
                                            Some(Bounds::exact(const_block as i64))
                                        } else if op_name == "cuda_tile.get_tile_block_id" {
                                            let const_block = match i {
                                                0 => const_grid.0,
                                                1 => const_grid.1,
                                                2 => const_grid.2,
                                                _ => unreachable!("Impossible")
                                            };
                                            Some(Bounds::new(0i64, const_block as i64 - 1))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };
                                    values.push(TileRustValue::<'c, 'c>::new_primitive(op_value, elem_ty.clone(), maybe_bounds));
                                },
                                Kind::StructuredType => {
                                    let op_value: Value<'c, 'c> = op_ref.result(i).unwrap().into();
                                    values.push(TileRustValue::<'c, 'c>::new_structured_type(op_value, elem_ty.clone(), None));
                                }
                                Kind::Compound | Kind::Struct | Kind::String => return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!("this operation returned an unsupported element type ({:?}); only scalar and structured types are supported", elem_ty.kind),
                                ),
                            }
                        }
                        Ok(Some(TileRustValue::<'c, 'c>::new_compound(
                            values,
                            return_type,
                        )))
                    } else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!("operations that return multiple values must use a tuple return type, got `{}`",
                                return_type.rust_ty.to_token_stream().to_string()),
                        );
                    }
                }
                Kind::Struct => {
                    return self.jit_error_result(
                        &call_expr.span(),
                        "this operation cannot return a struct; only scalar and structured (tile) types are supported as return types",
                    )
                }
                Kind::String => {
                    return self.jit_error_result(
                        &call_expr.span(),
                        "this operation cannot return a string; only scalar and structured (tile) types are supported as return types",
                    )
                }
            }
        } else {
            let op = op_builder.build().unwrap();
            let op_ref = builder.append_operation(op);
            if !op_ref.verify() {
                return self.jit_error_result(
                    &call_expr.span(),
                    &format!("operation MLIR verification failed: {}", op_ref.to_string()),
                );
            }
            Ok(None)
        }
    }

    pub fn compile_cuda_tile_macro(
        &'c self,
        builder: &'c ir::Block<'c>,
        mac: &syn::Macro,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        _return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let Some(mac_ident) = mac.path.get_ident() else {
            return self.jit_error_result(&mac.path.span(), "unrecognized macro invocation");
        };
        match mac_ident.to_string().as_str() {
            "cuda_tile_print" => {
                // TODO (hme): Use Punctuated<Expr, Token![,]>
                let exprs = parse_list_of_expr(mac.tokens.clone())?;
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
                            let Some(val) =
                                self.compile_expression(builder, &expr, generic_vars, ctx, None)?
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
                // We need to obtain the arg types for printing.
                let re_repl = Regex::new(r"\{\}").unwrap();
                for (i, element_ty) in element_type_instance.into_iter().enumerate() {
                    let rust_element_type_instance = element_ty.expect(
                        format!("failed to determine element type for print argument {}", i)
                            .as_str(),
                    );
                    // Make sure there is still something to match.
                    if !re_repl.is_match(&str_literal) {
                        return self.jit_error_result(
                            &mac.span(),
                            "more arguments than `{}` placeholders in print format string",
                        );
                    }
                    let Some(tile_element_type_instance) =
                        get_cuda_tile_element_type_from_rust_primitive_str(
                            &rust_element_type_instance,
                            &self.modules.primitives,
                        )
                    else {
                        return self.jit_error_result(
                            &mac.span(),
                            &format!(
                                "unable to determine tile element type for `{rust_element_type_instance}`"
                            ),
                        );
                    };
                    // TODO (hme): Is this going to work in general?
                    let first_char = tile_element_type_instance.chars().next().unwrap();
                    let replace_str = format!("%{first_char}");
                    let local_str_literal =
                        re_repl.replacen(&str_literal, 1, replace_str).to_string();
                    str_literal = local_str_literal;
                }
                if re_repl.is_match(&str_literal) {
                    return self.jit_error_result(
                        &mac.span(),
                        "more `{}` placeholders than arguments in print format string",
                    );
                }
                // TODO (hme): Update when print_tko goes online.
                // let print_builder =
                //     OperationBuilder::new("cuda_tile.print_tko", Location::unknown(&self.context));
                let print_builder =
                    OperationBuilder::new("cuda_tile.print", Location::unknown(&self.context));
                // TODO (hme): Support ordering of print commands.
                let operand_seg_sizes = format!("array<i32: {}, {}>", arg_values.len(), 0); // 0 corresponds to length of token arg, which is 0.
                let print_op = print_builder
                    .add_attributes(&[
                        (
                            Identifier::new(&self.context, "str"),
                            StringAttribute::new(&self.context, str_literal.as_str()).into(),
                        ),
                        (
                            Identifier::new(&self.context, "operandSegmentSizes"),
                            Attribute::parse(&self.context, operand_seg_sizes.as_str()).unwrap(),
                        ),
                    ])
                    .add_operands(&arg_values)
                    // .add_results(&[ir::Type::parse(&self.context, "!cuda_tile.token").unwrap()])
                    .build()
                    .unwrap();

                {
                    let op_ref = builder.append_operation(print_op);
                    if !op_ref.verify() {
                        return self.jit_error_result(
                            &mac.span(),
                            "print operation failed MLIR verification",
                        );
                    }
                }
                Ok(None)
            }
            "cuda_tile_assert" => {
                // println!("cuda_tile_assert: {}", mac.tokens);
                let punctuated = Punctuated::<Expr, Token![,]>::parse_terminated;
                let expressions_err = punctuated
                    .parse2(mac.tokens.clone())
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
                let arg_values = {
                    let Some(val) =
                        self.compile_expression(builder, bool_expr, generic_vars, ctx, None)?
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
                let assert_builder =
                    OperationBuilder::new("cuda_tile.assert", Location::unknown(&self.context));
                let assert_op = assert_builder
                    .add_attributes(&[named_str_attr(&self.context, "message", str_lit.as_str())])
                    .add_operands(&arg_values)
                    .build()
                    .unwrap();
                builder.append_operation(assert_op);
                Ok(None)
            }
            _ => {
                return self.jit_error_result(
                    &mac.path.span(),
                    &format!("unrecognized macro `{}`", mac_ident),
                )
            }
        }
    }
}
