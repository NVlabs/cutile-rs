/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expression compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_expression.rs` — translates Rust `syn::Expr`
//! AST nodes into tile-ir operations. Only type and IR-emission changes; the
//! control flow, dispatch logic, and variable binding are identical.

use super::_function::CUDATileFunctionCompiler;
use super::_value::{
    BlockTerminator, CompilerContext, DimOrigin, PartitionAxisOrigin, TileRustValue,
};
use super::shared_types::Kind;
use super::shared_utils::{
    collect_mutated_variables, collect_mutated_variables_from_block,
    collect_mutated_variables_from_expr, collect_mutated_variables_loop,
    collect_mutated_variables_while, dedup, update_outer_block_type_meta, TileBinaryOp,
    STACK_GROW_SIZE, STACK_RED_ZONE,
};
use super::tile_rust_type::TileRustType;
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::generics::{
    get_cga_from_generic_argument, GenericVars, TypeInstance, TypeInstanceUserType,
};
use crate::passes::name_resolution::{DefKind, Res};
use crate::syn_utils::*;
use crate::types::*;

use cutile_ir::builder::{append_op, build_block, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, BlockId, Location, Module, Region, ScalarType, TileElementType, TileType,
    Type as TileIrType,
};

use quote::ToTokens;
use std::collections::{BTreeMap, HashMap};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{parse_quote, Expr, ExprForLoop, ExprMacro, Lit, Member, Pat, Token, UnOp};

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Construct a ZST marker type placeholder from a path expression.
    ///
    /// Used for static_params like `ftz::Enabled`, `rounding::NearestEven`.
    /// These carry no tile-ir value — they're compile-time constants
    /// consumed by `resolve_static_params` during op compilation.
    fn make_zst_marker(path_expr: &syn::ExprPath) -> TileRustValue {
        let path_ty: syn::Type = syn::Type::Path(syn::TypePath {
            qself: None,
            path: path_expr.path.clone(),
        });
        let type_instance = TypeInstance::UserType(TypeInstanceUserType {
            maybe_generic_ty: path_ty,
        });
        let ty = TileRustType::new_string(type_instance);
        TileRustValue::new_string(Expr::Path(path_expr.clone()), ty)
    }

    fn const_shape_macro_args(
        &self,
        mac_expr: &ExprMacro,
        generic_vars: &GenericVars,
        ctx: &CompilerContext,
    ) -> Result<Vec<String>, JITError> {
        let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
        let exprs = parser.parse2(mac_expr.mac.tokens.clone()).map_err(|err| {
            self.jit_error(
                &mac_expr.span(),
                &format!("failed to parse const-shape macro arguments: {err}"),
            )
        })?;
        let expr_count = exprs.len();
        let mut args = Vec::new();
        for expr in exprs {
            match &expr {
                Expr::Path(path) if path.path.segments.len() == 1 => {
                    let name = get_ident_from_path_expr(path).to_string();
                    if let Some(cga) = generic_vars.inst_array.get(&name) {
                        if expr_count != 1 {
                            return self.jit_error_result(
                                &expr.span(),
                                &format!(
                                    "`{name}` names a const generic array; use it alone or index it as `{name}[i]`"
                                ),
                            );
                        }
                        args.extend(cga.iter().map(|dim| dim.to_string()));
                        continue;
                    }
                    self.require_compile_time_shape_expr(&expr, generic_vars, ctx)?;
                    args.push(expr.to_token_stream().to_string());
                }
                Expr::Index(index) => {
                    if let Expr::Path(path) = index.expr.as_ref() {
                        let name = get_ident_from_path_expr(path).to_string();
                        if let Some(cga) = generic_vars.inst_array.get(&name) {
                            let i = parse_signed_literal_as_i32(&index.index);
                            let Some(dim) = cga.get(i as usize) else {
                                return self.jit_error_result(
                                    &index.index.span(),
                                    &format!(
                                        "index {i} out of bounds for const generic array `{name}` of length {}",
                                        cga.len()
                                    ),
                                );
                            };
                            args.push(dim.to_string());
                            continue;
                        }
                    }
                    return self.jit_error_result(
                        &expr.span(),
                        "only const generic array indexing like `S[0]` is supported in `const_shape!` and `const_array!`",
                    );
                }
                _ => {
                    self.require_compile_time_shape_expr(&expr, generic_vars, ctx)?;
                    args.push(expr.to_token_stream().to_string());
                }
            }
        }
        Ok(args)
    }

    fn require_compile_time_shape_expr(
        &self,
        expr: &Expr,
        generic_vars: &GenericVars,
        ctx: &CompilerContext,
    ) -> Result<(), JITError> {
        match expr {
            Expr::Lit(_) | Expr::Unary(_) => Ok(()),
            Expr::Path(path) if path.path.segments.len() == 1 => {
                let name = get_ident_from_path_expr(path).to_string();
                if generic_vars.get_i32(&name).is_some() {
                    return Ok(());
                }
                if ctx
                    .vars
                    .get(&name)
                    .and_then(|value| value.bounds)
                    .is_some_and(|bounds| bounds.is_exact())
                {
                    return Ok(());
                }
                let res = self
                    .modules
                    .name_resolver
                    .resolve_path(&path.path, &self.module_name);
                if let Res::Def(DefKind::Const, def_id) = res {
                    if self
                        .modules
                        .name_resolver
                        .get_const(&def_id)
                        .and_then(crate::type_aliases::const_item_scalar_expr)
                        .is_some()
                    {
                        return Ok(());
                    }
                }
                self.jit_error_result(
                    &expr.span(),
                    "all arguments to `const_shape!` must be compile-time constants",
                )
            }
            Expr::Paren(paren) => {
                self.require_compile_time_shape_expr(&paren.expr, generic_vars, ctx)
            }
            _ => self.jit_error_result(
                &expr.span(),
                "all arguments to `const_shape!` must be compile-time constants",
            ),
        }
    }

    fn make_option_type(rust_ty: syn::Type) -> TileRustType {
        let type_instance = TypeInstance::UserType(TypeInstanceUserType {
            maybe_generic_ty: rust_ty,
        });
        TileRustType::new_enum(type_instance)
    }

    fn make_option_type_from_payload(payload_ty: &TileRustType) -> TileRustType {
        let payload_rust_ty = &payload_ty.rust_ty;
        let rust_ty: syn::Type = parse_quote!(Option<#payload_rust_ty>);
        Self::make_option_type(rust_ty)
    }

    fn path_looks_like_associated_const(
        &self,
        path_expr: &syn::ExprPath,
        generic_vars: &GenericVars,
    ) -> bool {
        if path_expr.qself.is_some() {
            return true;
        }
        if path_expr.path.segments.len() != 2 {
            return false;
        }
        let qualifier = path_expr.path.segments[0].ident.to_string();
        generic_vars.var_type(&qualifier).is_some()
            || self.modules.structs().contains_key(&qualifier)
            || self
                .modules
                .primitives()
                .keys()
                .any(|(_, self_name)| self_name == &qualifier)
    }

    fn expected_array_element_type(
        &self,
        expected: &TileRustType,
        generic_vars: &GenericVars,
    ) -> Result<Option<TileRustType>, JITError> {
        let elem_ty = match &expected.rust_ty {
            syn::Type::Array(array) => Some(&*array.elem),
            syn::Type::Slice(slice) => Some(&*slice.elem),
            _ => None,
        };
        let Some(elem_ty) = elem_ty else {
            return Ok(None);
        };
        self.compile_type(elem_ty, generic_vars, &HashMap::new())
    }

    fn compile_else_branch(
        &self,
        module: &mut Module,
        block_id: BlockId,
        else_expr: &Expr,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        match else_expr {
            Expr::Block(block_expr) => {
                self.compile_block(module, block_id, &block_expr.block, generic_vars, ctx, return_type)
            }
            Expr::If(_) => {
                let synthetic_block = syn::Block {
                    brace_token: Default::default(),
                    stmts: vec![syn::Stmt::Expr(else_expr.clone(), None)],
                };
                self.compile_block(module, block_id, &synthetic_block, generic_vars, ctx, return_type)
            }
            _ => self.jit_error_result(
                &else_expr.span(),
                "only block expressions (`{ ... }`) and chained `else if` expressions are supported in else branches",
            ),
        }
    }

    fn cga_type_arg(dims: &[i32]) -> String {
        let dims = dims
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{ [{dims}] }}")
    }

    fn mapped_partition_type_shapes(
        &self,
        value: &TileRustValue,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<(Vec<i32>, Vec<i32>), JITError> {
        let (type_ident, type_generic_args) = get_ident_generic_args(&value.ty.rust_ty);
        let Some(type_ident) = type_ident else {
            return self.jit_error_result(span, "expected a mapped partition type");
        };
        if !type_ident.to_string().starts_with("MappedPartitionMut") {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` for loops require a MappedPartitionMut receiver, got `{}`",
                    value.ty.rust_ty.to_token_stream()
                ),
            );
        }
        let Some(tile_shape_arg) = type_generic_args.args.iter().nth(1) else {
            return self.jit_error_result(
                span,
                "MappedPartitionMut is missing its tile-shape generic argument",
            );
        };
        let Some(map_shape_arg) = type_generic_args.args.iter().nth(2) else {
            return self.jit_error_result(
                span,
                "MappedPartitionMut is missing its map-shape generic argument",
            );
        };
        let Some(tile_shape) = get_cga_from_generic_argument(tile_shape_arg, generic_vars) else {
            return self.jit_error_result(
                span,
                "failed to resolve MappedPartitionMut tile-shape const generic array",
            );
        };
        let Some(map_shape) = get_cga_from_generic_argument(map_shape_arg, generic_vars) else {
            return self.jit_error_result(
                span,
                "failed to resolve MappedPartitionMut map-shape const generic array",
            );
        };
        if tile_shape.len() != 2 || map_shape.len() != 2 {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` currently supports rank-2 MappedPartitionMut values, got tile rank {} and map rank {}",
                    tile_shape.len(),
                    map_shape.len()
                ),
            );
        }
        Ok((tile_shape, map_shape))
    }

    fn compile_i32_type(
        &self,
        generic_vars: &GenericVars,
        span: &proc_macro2::Span,
    ) -> Result<TileRustType, JITError> {
        self.compile_type(&parse_quote!(i32), generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(span, "failed to compile i32 type"))
    }

    fn scalar_i32_ir_type() -> TileIrType {
        TileIrType::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(ScalarType::I32),
        })
    }

    fn compile_tile_block_tuple(
        &self,
        module: &mut Module,
        block_id: BlockId,
        opcode: Opcode,
        i32_ty: &TileRustType,
        span: &proc_macro2::Span,
    ) -> Vec<TileRustValue> {
        let is_tile_block_id = matches!(opcode, Opcode::GetTileBlockId);
        let is_num_tile_blocks = matches!(opcode, Opcode::GetNumTileBlocks);
        let scalar_i32_ty = Self::scalar_i32_ir_type();
        let mut op_builder = OpBuilder::new(opcode, self.ir_location(span));
        for _ in 0..3 {
            op_builder = op_builder.result(scalar_i32_ty.clone());
        }
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);
        results
            .into_iter()
            .enumerate()
            .map(|(axis, value)| {
                let bounds = self.const_grid.and_then(|const_grid| {
                    let axis_size = match axis {
                        0 => const_grid.0,
                        1 => const_grid.1,
                        2 => const_grid.2,
                        _ => unreachable!(),
                    } as i64;
                    if is_num_tile_blocks {
                        Some(Bounds::exact(axis_size))
                    } else if is_tile_block_id && axis_size > 0 {
                        Some(Bounds::new(0, axis_size - 1))
                    } else {
                        None
                    }
                });
                TileRustValue::new_primitive(value, i32_ty.clone(), bounds)
            })
            .collect()
    }

    fn compile_index_space_shape_values(
        &self,
        module: &mut Module,
        block_id: BlockId,
        partition_value: &TileRustValue,
        i32_ty: &TileRustType,
        span: &proc_macro2::Span,
    ) -> Result<Vec<TileRustValue>, JITError> {
        let view_value = partition_value.value.ok_or_else(|| {
            self.jit_error(span, "expected a direct value for mapped partition indices")
        })?;
        let view_ty = module.value_type(view_value).clone();
        let TileIrType::PartitionView(pv) = &view_ty else {
            return self.jit_error_result(
                span,
                &format!(
                    "`iter_indices()` expects a mapped partition view, got `{:?}`",
                    view_ty
                ),
            );
        };
        let rank = pv.tile_shape.len();
        if rank != 2 {
            return self.jit_error_result(
                span,
                &format!("`iter_indices()` currently supports rank-2 partitions, got rank {rank}"),
            );
        }

        let scalar_i32_ty = Self::scalar_i32_ir_type();
        let mut op_builder =
            OpBuilder::new(Opcode::GetIndexSpaceShape, self.ir_location(span)).operand(view_value);
        for _ in 0..rank {
            op_builder = op_builder.result(scalar_i32_ty.clone());
        }
        let (op_id, results) = op_builder.build(module);
        append_op(module, block_id, op_id);

        let mut values = Vec::with_capacity(rank);
        for axis in 0..rank {
            let mut value = TileRustValue::new_primitive(results[axis], i32_ty.clone(), None);
            let parent_axis = pv.dim_map.get(axis).copied().ok_or_else(|| {
                self.jit_error(
                    span,
                    &format!(
                        "`iter_indices()` axis {axis} is missing from partition dim_map {:?}",
                        pv.dim_map
                    ),
                )
            })?;
            if parent_axis < 0 {
                return self.jit_error_result(
                    span,
                    &format!(
                        "`iter_indices()` axis {axis} maps to invalid parent axis {parent_axis}"
                    ),
                );
            }
            let parent_axis = parent_axis as usize;
            let Some(&parent_dim) = pv.tensor_view.shape.get(parent_axis) else {
                return self.jit_error_result(
                    span,
                    &format!(
                        "`iter_indices()` axis {axis} maps to parent axis {parent_axis}, but parent tensor rank is {}",
                        pv.tensor_view.shape.len()
                    ),
                );
            };
            let tile_dim = pv.tile_shape[axis] as i64;
            if tile_dim <= 0 {
                return self.jit_error_result(
                    span,
                    &format!("`iter_indices()` axis {axis} has invalid tile dimension {tile_dim}"),
                );
            }
            if parent_dim >= 0 {
                let num_tiles = (parent_dim + tile_dim - 1) / tile_dim;
                value.bounds = Some(Bounds::exact(num_tiles));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn simple_path_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path)
                if path.qself.is_none()
                    && path.path.leading_colon.is_none()
                    && path.path.segments.len() == 1 =>
            {
                Some(path.path.segments[0].ident.to_string())
            }
            Expr::Paren(paren) => Self::simple_path_name(&paren.expr),
            _ => None,
        }
    }

    fn is_dim_new_call(func: &Expr) -> bool {
        let Expr::Path(path) = func else {
            return false;
        };
        path.path.segments.len() == 2
            && path.path.segments[0].ident == "Dim"
            && path.path.segments[1].ident == "new"
    }

    fn compile_dim_new_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &syn::ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if call_expr.args.len() != 1 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "`Dim::new` expects 1 argument, got {}",
                    call_expr.args.len()
                ),
            );
        }
        let i32_type = self
            .compile_type(&parse_quote!(i32), generic_vars, &HashMap::new())?
            .ok_or_else(|| self.jit_error(&call_expr.span(), "failed to compile i32 type"))?;
        let mut value = self
            .compile_expression(
                module,
                block_id,
                &call_expr.args[0],
                generic_vars,
                ctx,
                Some(i32_type),
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.args[0].span(),
                    "failed to compile dimension size",
                )
            })?;
        let value_id = value.value.ok_or_else(|| {
            self.jit_error(
                &call_expr.args[0].span(),
                "dimension size must compile to a scalar value",
            )
        })?;
        value.dim_origin = Some(DimOrigin::Value(value_id));
        let dim_type = match return_type {
            Some(return_type) => return_type,
            None => self
                .compile_type(&parse_quote!(Dim), generic_vars, &HashMap::new())?
                .ok_or_else(|| self.jit_error(&call_expr.span(), "failed to compile Dim type"))?,
        };
        let dim_origin = value.dim_origin.clone();
        let mut fields = BTreeMap::new();
        fields.insert("size".to_string(), value);
        let mut dim = TileRustValue::new_struct(fields, dim_type);
        dim.dim_origin = dim_origin;
        Ok(Some(dim))
    }

    fn wrap_scalar_as_dim(
        &self,
        mut value: TileRustValue,
        generic_vars: &GenericVars,
        return_type: Option<TileRustType>,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue, JITError> {
        let value_id = value
            .value
            .ok_or_else(|| self.jit_error(span, "dimension size must compile to a scalar value"))?;
        if value.dim_origin.is_none() {
            value.dim_origin = Some(DimOrigin::Value(value_id));
        }
        let dim_type = match return_type {
            Some(return_type) => return_type,
            None => self
                .compile_type(&parse_quote!(Dim), generic_vars, &HashMap::new())?
                .ok_or_else(|| self.jit_error(span, "failed to compile Dim type"))?,
        };
        let dim_origin = value.dim_origin.clone();
        let bounds = value.bounds.clone();
        let mut fields = BTreeMap::new();
        fields.insert("size".to_string(), value);
        let mut dim = TileRustValue::new_struct(fields, dim_type);
        dim.dim_origin = dim_origin;
        dim.bounds = bounds;
        Ok(dim)
    }

    fn compile_into_dim_method(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &syn::ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.method != "into_dim" {
            return Ok(None);
        }
        if !method_call.args.is_empty() {
            return self.jit_error_result(
                &method_call.args.span(),
                "`IntoDim::into_dim` does not take arguments",
            );
        }
        let receiver = self
            .compile_expression(
                module,
                block_id,
                &method_call.receiver,
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.receiver.span(),
                    "failed to compile IntoDim receiver",
                )
            })?;
        if get_type_ident(&receiver.ty.rust_ty).is_some_and(|ident| ident == "Dim") {
            return Ok(Some(receiver));
        }
        Ok(Some(self.wrap_scalar_as_dim(
            receiver,
            generic_vars,
            return_type,
            &method_call.span(),
        )?))
    }

    fn compile_partition_with_bounds_method(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call: &syn::ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        if method_call.method != "with_bounds" {
            return Ok(None);
        }
        if method_call.args.len() != 1 {
            return self.jit_error_result(
                &method_call.args.span(),
                "`Partition::with_bounds` expects exactly one tuple argument",
            );
        }
        let mut partition = self
            .compile_expression(
                module,
                block_id,
                &method_call.receiver,
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.receiver.span(),
                    "failed to compile Partition::with_bounds receiver",
                )
            })?;
        let bounds = self
            .compile_expression(
                module,
                block_id,
                &method_call.args[0],
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.args[0].span(),
                    "failed to compile Partition::with_bounds tuple",
                )
            })?;
        let Some(bound_values) = bounds.values else {
            return self.jit_error_result(
                &method_call.args[0].span(),
                "`Partition::with_bounds` expects a rank-2 tuple",
            );
        };
        if bound_values.len() != 2 {
            return self.jit_error_result(
                &method_call.args[0].span(),
                &format!(
                    "`Partition::with_bounds` expects rank-2 bounds, got rank {}",
                    bound_values.len()
                ),
            );
        }
        let mut dim_origins = Vec::with_capacity(bound_values.len());
        for (axis, value) in bound_values.into_iter().enumerate() {
            let origin = Self::value_dim_origin(&value);
            let Some(origin) = origin else {
                return self.jit_error_result(
                    &method_call.args[0].span(),
                    &format!(
                        "`Partition::with_bounds` bound {axis} must come from `num_tiles`, `Dim::new`, or `IntoDim::into_dim`"
                    ),
                );
            };
            dim_origins.push(origin);
        }
        let return_type = match return_type {
            Some(return_type) => return_type,
            None => {
                let mut bounded_ty = partition.ty.rust_ty.clone();
                let syn::Type::Path(path_ty) = &mut bounded_ty else {
                    return self.jit_error_result(
                        &method_call.receiver.span(),
                        "expected a partition type for `Partition::with_bounds`",
                    );
                };
                let Some(segment) = path_ty.path.segments.last_mut() else {
                    return self.jit_error_result(
                        &method_call.receiver.span(),
                        "expected a partition type path for `Partition::with_bounds`",
                    );
                };
                segment.ident = syn::Ident::new("BoundedPartition", segment.ident.span());
                let mut return_type = partition.ty.clone();
                return_type.rust_ty = bounded_ty;
                return_type
            }
        };
        partition.ty = return_type;
        partition.bounded_axes = Some(dim_origins);
        Ok(Some(partition))
    }

    fn value_dim_origin(value: &TileRustValue) -> Option<DimOrigin> {
        value.dim_origin.clone().or_else(|| {
            value
                .fields
                .as_ref()
                .and_then(|fields| fields.get("size"))
                .and_then(|size| size.dim_origin.clone())
        })
    }

    fn dim_size_value(
        &self,
        dim_value: &TileRustValue,
        span: &proc_macro2::Span,
    ) -> Result<cutile_ir::ir::Value, JITError> {
        if let Some(value) = dim_value.value {
            return Ok(value);
        }
        let Some(fields) = dim_value.fields.as_ref() else {
            return self.jit_error_result(span, "dimension value is missing its scalar size");
        };
        let Some(size) = fields.get("size") else {
            return self.jit_error_result(span, "dimension value is missing its `size` field");
        };
        size.value
            .ok_or_else(|| self.jit_error(span, "dimension size must compile to a scalar value"))
    }

    /// Special lowering for `for idx in mapped_partition.iter_indices()`.
    ///
    /// This is intentionally separate from normal range-loop lowering. The
    /// iterator is a DSL proof boundary: the compiler lowers it to a persistent
    /// flat tile-id loop and mints `PartitionIndex` values branded with the
    /// mapped partition that produced them.
    fn try_compile_mapped_partition_indices_for_loop(
        &self,
        module: &mut Module,
        block_id: BlockId,
        for_expr: &ExprForLoop,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<bool, JITError> {
        let Expr::MethodCall(method_call) = &*for_expr.expr else {
            return Ok(false);
        };
        if method_call.method != "iter_indices" {
            return Ok(false);
        }
        if !method_call.args.is_empty() {
            return self.jit_error_result(
                &method_call.args.span(),
                "MappedPartitionMut::iter_indices does not take arguments",
            );
        }
        let Pat::Ident(iterand_ident) = &*for_expr.pat else {
            return self.jit_error_result(
                &for_expr.pat.span(),
                "MappedPartitionMut::iter_indices loops must bind a simple index variable",
            );
        };

        let partition_value = self
            .compile_expression(
                module,
                block_id,
                &method_call.receiver,
                generic_vars,
                ctx,
                None,
            )?
            .ok_or_else(|| {
                self.jit_error(
                    &method_call.receiver.span(),
                    "failed to compile mapped partition receiver",
                )
            })?;
        let partition_origin = partition_value.value.ok_or_else(|| {
            self.jit_error(
                &method_call.receiver.span(),
                "mapped partition receiver did not produce a direct value",
            )
        })?;
        let (tile_shape, map_shape) = self.mapped_partition_type_shapes(
            &partition_value,
            generic_vars,
            &method_call.receiver.span(),
        )?;
        let i32_ty = self.compile_i32_type(generic_vars, &for_expr.span())?;
        let index_space = self.compile_index_space_shape_values(
            module,
            block_id,
            &partition_value,
            &i32_ty,
            &method_call.receiver.span(),
        )?;
        let num_bid_m = index_space[0].clone();
        let num_bid_n = index_space[1].clone();
        let total_tiles = self.compile_binary_op_from_values(
            module,
            block_id,
            num_bid_m.clone(),
            num_bid_n.clone(),
            &TileBinaryOp::Mul,
            generic_vars,
            ctx,
            None,
            &for_expr.span(),
        )?;
        let total_tiles_value = total_tiles.value.ok_or_else(|| {
            self.jit_error(
                &for_expr.span(),
                "failed to compute mapped partition tile count",
            )
        })?;

        let pid = self.compile_tile_block_tuple(
            module,
            block_id,
            Opcode::GetTileBlockId,
            &i32_ty,
            &for_expr.span(),
        );
        let grid = self.compile_tile_block_tuple(
            module,
            block_id,
            Opcode::GetNumTileBlocks,
            &i32_ty,
            &for_expr.span(),
        );
        let lower_bound = pid[0]
            .value
            .ok_or_else(|| self.jit_error(&for_expr.span(), "failed to compute tile-block id"))?;
        let step = grid[0].value.ok_or_else(|| {
            self.jit_error(&for_expr.span(), "failed to compute tile-block grid size")
        })?;

        let loop_carry_vars = collect_mutated_variables(for_expr)?
            .into_iter()
            .collect::<Vec<_>>();
        let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
        let loop_carry_arg_tys = loop_carry_args
            .iter()
            .map(|val| module.value_type(*val).clone())
            .collect::<Vec<_>>();

        let for_iterand_type = Self::scalar_i32_ir_type();
        let loop_block_arg_tys = [&[for_iterand_type][..], loop_carry_arg_tys.as_slice()].concat();
        let (loop_block_id, loop_block_args) = build_block(module, &loop_block_arg_tys);

        let mut for_variables = ctx.clone();
        let block_args: Vec<cutile_ir::ir::Value> = loop_block_args[1..].to_vec();
        for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
        for_variables.carry_vars = Some(loop_carry_vars.clone());
        for_variables.default_terminator = Some(BlockTerminator::Continue);

        let tile_id_name = "__cutile_mapped_partition_tile_id";
        let num_bid_m_name = "__cutile_mapped_partition_num_bid_m";
        let num_bid_n_name = "__cutile_mapped_partition_num_bid_n";
        let tile_id = TileRustValue::new_primitive(loop_block_args[0], i32_ty.clone(), None);
        for_variables.vars.insert(tile_id_name.to_string(), tile_id);
        for_variables
            .vars
            .insert(num_bid_m_name.to_string(), num_bid_m);
        for_variables
            .vars
            .insert(num_bid_n_name.to_string(), num_bid_n);

        let tile_shape_arg = Self::cga_type_arg(&tile_shape);
        let map_shape_arg = Self::cga_type_arg(&map_shape);
        let index_ty: syn::Type = syn::parse_str(&format!("PartitionIndex<{tile_shape_arg}>"))
            .map_err(|err| {
                self.jit_error(
                    &for_expr.span(),
                    &format!("failed to build mapped partition index type: {err}"),
                )
            })?;
        let index_return_ty = self
            .compile_type(&index_ty, generic_vars, &HashMap::new())?
            .ok_or_else(|| {
                self.jit_error(&for_expr.span(), "failed to compile PartitionIndex type")
            })?;
        let swizzle_expr: Expr = syn::parse_str(&format!(
            "swizzle_partition_index_2d::<{tile_shape_arg}, {map_shape_arg}>({tile_id_name}, {num_bid_m_name}, {num_bid_n_name})"
        ))
        .map_err(|err| {
            self.jit_error(
                &for_expr.span(),
                &format!("failed to build mapped partition index expression: {err}"),
            )
        })?;
        let mut index_value = self
            .compile_expression(
                module,
                loop_block_id,
                &swizzle_expr,
                generic_vars,
                &mut for_variables,
                Some(index_return_ty),
            )?
            .ok_or_else(|| {
                self.jit_error(&for_expr.span(), "failed to compile mapped partition index")
            })?;
        index_value.partition_origin = Some(partition_origin);
        let index_tensor_origin = Self::simple_path_name(&method_call.receiver)
            .or_else(|| partition_value.tensor_origin.clone());
        if let (Some(tensor_origin), Some(fields)) =
            (index_tensor_origin, index_value.fields.as_mut())
        {
            if let Some(coords) = fields.get_mut("coords") {
                if let Some(values) = coords.values.as_mut() {
                    for (axis, value) in values.iter_mut().enumerate() {
                        let dim_origin = DimOrigin::PartitionAxis {
                            view: partition_origin,
                            axis,
                            tile_dim: tile_shape[axis],
                        };
                        value.partition_axis_origin = Some(PartitionAxisOrigin {
                            tensor: tensor_origin.clone(),
                            axis,
                            tile_dim: tile_shape[axis],
                        });
                        value.index_origin = Some(dim_origin);
                    }
                }
            }
        }
        for_variables
            .vars
            .insert(iterand_ident.ident.to_string(), index_value);

        self.compile_block(
            module,
            loop_block_id,
            &for_expr.body,
            generic_vars,
            &mut for_variables,
            return_type,
        )?;

        let region_id = module.alloc_region(Region {
            blocks: vec![loop_block_id],
        });
        let (for_op_id, result_values) =
            OpBuilder::new(Opcode::For, self.ir_location(&for_expr.span()))
                .operands([lower_bound, total_tiles_value, step].iter().copied())
                .operands(loop_carry_args.iter().copied())
                .results(loop_carry_arg_tys.iter().cloned())
                .region(region_id)
                .build(module);
        append_op(module, block_id, for_op_id);

        if result_values.len() != loop_carry_args.len() {
            return self.jit_error_result(
                &for_expr.span(),
                &format!(
                    "mapped partition indices loop produces {} results but {} mutable variables are carried across iterations",
                    result_values.len(),
                    loop_carry_args.len()
                ),
            );
        }
        ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
        Ok(true)
    }

    /// Special lowering for `for idx in dim`.
    ///
    /// A `Dim` is the source-level iterable proof object. Iterating it lowers
    /// to `for idx in 0..dim`, while the loop variable is tagged as an index
    /// produced by that dimension. Plain `i32` values, including `num_tiles`
    /// results, continue through normal range lowering.
    fn try_compile_dim_for_loop(
        &self,
        module: &mut Module,
        block_id: BlockId,
        for_expr: &ExprForLoop,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<bool, JITError> {
        let Some(dim_name) = Self::simple_path_name(&for_expr.expr) else {
            return Ok(false);
        };
        let Some(dim_value) = ctx.vars.get(&dim_name).cloned() else {
            return Ok(false);
        };
        if !get_type_ident(&dim_value.ty.rust_ty).is_some_and(|ident| ident == "Dim") {
            return Ok(false);
        }
        let Some(dim_origin) = Self::value_dim_origin(&dim_value) else {
            return Ok(false);
        };
        let upper_bound = self.dim_size_value(&dim_value, &for_expr.expr.span())?;
        let maybe_iterand_ident = match &*for_expr.pat {
            Pat::Wild(_) => None,
            Pat::Ident(ident_pat) => Some(ident_pat),
            _ => {
                return self.jit_error_result(
                    &for_expr.pat.span(),
                    "dimension loops must bind a simple index variable or `_`",
                );
            }
        };

        let zero = self.compile_constant(module, block_id, generic_vars, 0i32)?;
        let one = self.compile_constant(module, block_id, generic_vars, 1i32)?;
        let lower_bound = zero.value.ok_or_else(|| {
            self.jit_error(
                &for_expr.span(),
                "failed to compile dimension loop lower bound",
            )
        })?;
        let step = one.value.ok_or_else(|| {
            self.jit_error(&for_expr.span(), "failed to compile dimension loop step")
        })?;

        let loop_carry_vars = collect_mutated_variables(for_expr)?
            .into_iter()
            .collect::<Vec<_>>();
        let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
        let loop_carry_arg_tys = loop_carry_args
            .iter()
            .map(|val| module.value_type(*val).clone())
            .collect::<Vec<_>>();

        let for_iterand_type = module.value_type(upper_bound).clone();
        let loop_block_arg_tys = [&[for_iterand_type][..], loop_carry_arg_tys.as_slice()].concat();
        let (loop_block_id, loop_block_args) = build_block(module, &loop_block_arg_tys);

        let mut for_variables = ctx.clone();
        let block_args: Vec<cutile_ir::ir::Value> = loop_block_args[1..].to_vec();
        for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
        if let Some(iterand_ident) = maybe_iterand_ident {
            let iterand_name = iterand_ident.ident.to_string();
            let i32_type = self
                .compile_type(&parse_quote!(i32), generic_vars, &HashMap::new())?
                .ok_or_else(|| self.jit_error(&for_expr.span(), "failed to compile i32 type"))?;
            let upper_bounds = dim_value.bounds.clone().or_else(|| {
                dim_value
                    .fields
                    .as_ref()
                    .and_then(|fields| fields.get("size"))
                    .and_then(|size| size.bounds.clone())
            });
            let mut iterand_val = if let Some(bounds) = upper_bounds {
                let upper = bounds.end - 1;
                if upper >= 0 {
                    let bounds = Bounds::new(0, upper);
                    let mut value = self.compile_value_assumption(
                        module,
                        loop_block_id,
                        loop_block_args[0],
                        "assume_bounds",
                        &[bounds.start as i32, bounds.end as i32],
                        i32_type.clone(),
                        &for_expr.span(),
                    )?;
                    value.bounds = Some(bounds);
                    value
                } else {
                    TileRustValue::new_value_kind_like(loop_block_args[0], i32_type.clone())
                }
            } else {
                TileRustValue::new_value_kind_like(loop_block_args[0], i32_type.clone())
            };
            iterand_val.index_origin = Some(dim_origin);
            for_variables.vars.insert(iterand_name, iterand_val);
        }
        for_variables.carry_vars = Some(loop_carry_vars.clone());
        for_variables.default_terminator = Some(BlockTerminator::Continue);

        self.compile_block(
            module,
            loop_block_id,
            &for_expr.body,
            generic_vars,
            &mut for_variables,
            return_type,
        )?;

        let region_id = module.alloc_region(Region {
            blocks: vec![loop_block_id],
        });
        let (for_op_id, result_values) =
            OpBuilder::new(Opcode::For, self.ir_location(&for_expr.span()))
                .operands([lower_bound, upper_bound, step].iter().copied())
                .operands(loop_carry_args.iter().copied())
                .results(loop_carry_arg_tys.iter().cloned())
                .region(region_id)
                .build(module);
        append_op(module, block_id, for_op_id);

        if result_values.len() != loop_carry_args.len() {
            return self.jit_error_result(
                &for_expr.span(),
                &format!(
                    "dimension loop produces {} results but {} mutable variables are carried across iterations",
                    result_values.len(),
                    loop_carry_args.len()
                ),
            );
        }
        ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
        Ok(true)
    }

    pub fn compile_expression(
        &self,
        module: &mut Module,
        block_id: BlockId,
        expr: &syn::Expr,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _expr_debug_str = expr.to_token_stream().to_string();
            match expr {
                Expr::ForLoop(for_expr) => {
                    if self.try_compile_mapped_partition_indices_for_loop(
                        module,
                        block_id,
                        for_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(None);
                    }
                    if self.try_compile_dim_for_loop(
                        module,
                        block_id,
                        for_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(None);
                    }

                    // A for loop: for pat in expr { ... }.
                    let maybe_iterand_ident = match &*for_expr.pat {
                        Pat::Wild(_) => {
                            // Iterand is not bounded.
                            None
                        }
                        Pat::Ident(ident_pat) => Some(ident_pat),
                        _ => return self.jit_error_result(
                            &for_expr.pat.span(),
                            "this loop pattern is not supported; use a simple variable name or `_`",
                        ),
                    };
                    // Extract range and optional step from the for-loop expression.
                    // Supports: `0..n` (step=1) and `(0..n).step_by(k)`.
                    let (range_expr, maybe_step_expr): (&syn::ExprRange, Option<&Expr>) =
                        match &*for_expr.expr {
                            Expr::Range(range) => (range, None),
                            Expr::MethodCall(mc) if mc.method == "step_by" => {
                                let receiver = match &*mc.receiver {
                                    Expr::Paren(p) => &*p.expr,
                                    other => other,
                                };
                                let Expr::Range(range) = receiver else {
                                    return self.jit_error_result(
                                        &mc.receiver.span(),
                                        "expected a range expression as the receiver of step_by (e.g. `(0..n).step_by(k)`)",
                                    );
                                };
                                if mc.args.len() != 1 {
                                    return self.jit_error_result(
                                        &mc.args.span(),
                                        "step_by expects exactly one argument",
                                    );
                                }
                                (range, Some(&mc.args[0]))
                            }
                            _ => {
                                return self.jit_error_result(
                                    &for_expr.expr.span(),
                                    "only range expressions (e.g. `0..n` or `(0..n).step_by(k)`) are supported in for loops",
                                );
                            }
                        };
                    // TODO (hme): Add meaningful errors and do more than just unwrap.
                    let Some(start_expr) = &range_expr.start else {
                        return self.jit_error_result(
                            &range_expr.span(),
                            "range expression is missing a start bound (e.g. `0..n`)",
                        );
                    };
                    let Some(end_expr) = &range_expr.end else {
                        return self.jit_error_result(
                            &range_expr.span(),
                            "range expression is missing an end bound (e.g. `0..n`)",
                        );
                    };
                    let start_return_type = self
                        .typeck_expr_tile_type(start_expr, generic_vars, &HashMap::new())?
                        .or(return_type.clone());
                    let Some(start_val) = self.compile_expression(
                        module,
                        block_id,
                        start_expr,
                        generic_vars,
                        ctx,
                        start_return_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &start_expr.span(),
                            "failed to compile range start expression",
                        );
                    };
                    let end_return_type = self
                        .typeck_expr_tile_type(end_expr, generic_vars, &HashMap::new())?
                        .or(return_type.clone());
                    let Some(end_val) = self.compile_expression(
                        module,
                        block_id,
                        end_expr,
                        generic_vars,
                        ctx,
                        end_return_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &end_expr.span(),
                            "failed to compile range end expression",
                        );
                    };
                    let iterand_lower_const = start_val.bounds;
                    let iterand_upper_const = end_val.bounds;
                    let lower_bound = start_val.value.unwrap();
                    let upper_bound = end_val.value.unwrap();
                    let step_value = if let Some(step_expr) = maybe_step_expr {
                        let Some(val) = self.compile_expression(
                            module,
                            block_id,
                            step_expr,
                            generic_vars,
                            ctx,
                            Some(start_val.ty.clone()),
                        )?
                        else {
                            return self.jit_error_result(
                                &step_expr.span(),
                                "failed to compile step_by expression",
                            );
                        };
                        val
                    } else {
                        self.compile_constant(module, block_id, generic_vars, 1)?
                    };
                    let step = step_value.value.ok_or_else(|| {
                        self.jit_error(
                            &for_expr.span(),
                            "internal: failed to produce step value for for-loop",
                        )
                    })?;

                    // We skip verifying the op here and just require that each mutated mutable vars:
                    // 1. Is passed as an operand.
                    // 2. Is a block argument.
                    // 3. Is loop-carried.
                    // 4. Is returned.
                    let for_iterand_type = module.value_type(lower_bound).clone();
                    let loop_carry_vars = collect_mutated_variables(for_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| module.value_type(*val).clone())
                        .collect::<Vec<_>>();

                    // Build the loop body block.
                    // Add iterand as first argument.
                    let loop_block_arg_tys =
                        [&[for_iterand_type][..], loop_carry_arg_tys.as_slice()].concat();
                    let (loop_block_id, loop_block_args) = build_block(module, &loop_block_arg_tys);

                    let mut for_variables = ctx.clone();
                    // Update loop carry variables within the for loop
                    // to the mutable variables accessed in this operation.
                    let block_args: Vec<cutile_ir::ir::Value> = loop_block_args[1..].to_vec();
                    for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                    if let Some(iterand_ident) = maybe_iterand_ident {
                        // maybe_iterand_ident is None if it is wild.
                        // If it's an ident, then add the iterand as a var.
                        let iterand_name = iterand_ident.ident.to_string();
                        let iterand_val = loop_block_args[0];
                        // This has the same type as start/end val.
                        let iterand_ty = start_val.ty.clone();
                        // If the loop bounds are const, then we can put a bound on the iterand.
                        // Subtract upper bound by 1, since it is the open end of the interval [start, end).
                        let mut iterand_val = match (iterand_lower_const, iterand_upper_const) {
                            (Some(iterand_lower_const), Some(iterand_upper_const)) => {
                                let bounds = Bounds::new(
                                    iterand_lower_const.start,
                                    iterand_upper_const.end - 1,
                                );
                                let mut iterand_val = self.compile_value_assumption(
                                    module,
                                    loop_block_id,
                                    iterand_val,
                                    "assume_bounds",
                                    &[bounds.start as i32, bounds.end as i32],
                                    iterand_ty,
                                    &for_expr.span(),
                                )?;
                                iterand_val.bounds = Some(bounds);
                                iterand_val
                            }
                            (Some(iterand_lower_const), None) => self.compile_value_assumption(
                                module,
                                loop_block_id,
                                iterand_val,
                                "assume_bounds_lower",
                                &[iterand_lower_const.start as i32],
                                iterand_ty,
                                &for_expr.span(),
                            )?,
                            (None, Some(iterand_upper_const)) => self.compile_value_assumption(
                                module,
                                loop_block_id,
                                iterand_val,
                                "assume_bounds_upper",
                                &[iterand_upper_const.end as i32 - 1],
                                iterand_ty,
                                &for_expr.span(),
                            )?,
                            (None, None) => TileRustValue::new_value_kind_like(
                                iterand_val,
                                start_val.ty.clone(),
                            ),
                        };
                        if start_val
                            .bounds
                            .as_ref()
                            .is_some_and(|bounds| bounds.is_exact() && bounds.start == 0)
                        {
                            iterand_val.index_origin = Self::value_dim_origin(&end_val);
                        }
                        for_variables.vars.insert(iterand_name, iterand_val);
                    }
                    for_variables.carry_vars = Some(loop_carry_vars.clone());
                    for_variables.default_terminator = Some(BlockTerminator::Continue);
                    // TODO (hme): Support returns?
                    self.compile_block(
                        module,
                        loop_block_id,
                        &for_expr.body,
                        generic_vars,
                        &mut for_variables,
                        return_type,
                    )?;

                    let region_id = module.alloc_region(Region {
                        blocks: vec![loop_block_id],
                    });

                    let (for_op_id, result_values) =
                        OpBuilder::new(Opcode::For, self.ir_location(&for_expr.span()))
                            .operands([lower_bound, upper_bound, step].iter().copied())
                            .operands(loop_carry_args.iter().copied())
                            .results(loop_carry_arg_tys.iter().cloned())
                            .region(region_id)
                            .build(module);
                    append_op(module, block_id, for_op_id);

                    // TODO (hme): This fails with "operand #0 does not dominate this use"
                    //  This may be a bug.
                    //  The compiled module in its entirety still passes verification.
                    // assert!(for_op.verify());
                    if result_values.len() != loop_carry_args.len() {
                        return self.jit_error_result(
                            &for_expr.span(),
                            &format!(
                                "for loop produces {} results but {} mutable variables are carried across iterations",
                                result_values.len(),
                                loop_carry_args.len()
                            ),
                        );
                    }
                    ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
                    Ok(None)
                }
                Expr::While(while_expr) => {
                    // While loop: while condition { body }
                    // Convert to cuda_tile.loop - simpler approach: body then check
                    let loop_carry_vars = collect_mutated_variables_while(while_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| module.value_type(*val).clone())
                        .collect::<Vec<_>>();

                    // Build the loop body block.
                    let (loop_block_id, loop_block_args) = build_block(module, &loop_carry_arg_tys);

                    let mut loop_variables = ctx.clone();
                    let block_args: Vec<cutile_ir::ir::Value> = loop_block_args.clone();
                    loop_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                    loop_variables.carry_vars = Some(loop_carry_vars.clone());
                    loop_variables.default_terminator = Some(BlockTerminator::Continue);

                    // Evaluate condition
                    let Some(TileRustValue {
                        value: Some(condition_val),
                        ..
                    }) = self.compile_expression(
                        module,
                        loop_block_id,
                        &while_expr.cond,
                        generic_vars,
                        &mut loop_variables,
                        return_type.clone(),
                    )?
                    else {
                        return self.jit_error_result(
                            &while_expr.cond.span(),
                            "failed to compile while-loop condition",
                        );
                    };

                    // Check condition first - if false, break immediately
                    // Then region: continue to body (just yield, body comes next)
                    let (then_block_id, _then_block_args) = build_block(module, &[]);
                    let (yield_op_id, _) =
                        OpBuilder::new(Opcode::Yield, self.ir_location(&while_expr.span()))
                            .build(module);
                    append_op(module, then_block_id, yield_op_id);
                    let then_region_id = module.alloc_region(Region {
                        blocks: vec![then_block_id],
                    });

                    // Else region: break out
                    let (else_block_id, _else_block_args) = build_block(module, &[]);
                    let break_values = loop_variables.unpack_some_vars(&loop_carry_vars)?;
                    let (break_op_id, _) =
                        OpBuilder::new(Opcode::Break, self.ir_location(&while_expr.span()))
                            .operands(break_values.iter().copied())
                            .build(module);
                    append_op(module, else_block_id, break_op_id);
                    let else_region_id = module.alloc_region(Region {
                        blocks: vec![else_block_id],
                    });

                    let (condition_check_id, _) =
                        OpBuilder::new(Opcode::If, self.ir_location(&while_expr.cond.span()))
                            .operand(condition_val)
                            .region(then_region_id)
                            .region(else_region_id)
                            .build(module);
                    append_op(module, loop_block_id, condition_check_id);

                    // Execute body
                    self.compile_block(
                        module,
                        loop_block_id,
                        &while_expr.body,
                        generic_vars,
                        &mut loop_variables,
                        return_type.clone(),
                    )?;
                    // compile_block will inject continue at the end

                    let region_id = module.alloc_region(Region {
                        blocks: vec![loop_block_id],
                    });

                    let (loop_op_id, result_values) =
                        OpBuilder::new(Opcode::Loop, self.ir_location(&while_expr.span()))
                            .operands(loop_carry_args.iter().copied())
                            .results(loop_carry_arg_tys.iter().cloned())
                            .region(region_id)
                            .build(module);
                    append_op(module, block_id, loop_op_id);

                    if result_values.len() != loop_carry_args.len() {
                        return self.jit_error_result(
                            &while_expr.span(),
                            &format!(
                                "while loop produces {} results but {} mutable variables are carried across iterations",
                                result_values.len(),
                                loop_carry_args.len()
                            ),
                        );
                    }
                    ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
                    Ok(None)
                }
                Expr::Loop(loop_expr) => {
                    // Infinite loop: loop { body }
                    // Same as while but without condition check
                    let loop_carry_vars = collect_mutated_variables_loop(loop_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| module.value_type(*val).clone())
                        .collect::<Vec<_>>();

                    // Build the loop body block.
                    let (loop_block_id, loop_block_args) = build_block(module, &loop_carry_arg_tys);

                    let mut loop_variables = ctx.clone();
                    let block_args: Vec<cutile_ir::ir::Value> = loop_block_args.clone();
                    loop_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                    loop_variables.carry_vars = Some(loop_carry_vars.clone());
                    loop_variables.default_terminator = Some(BlockTerminator::Continue);

                    // Execute loop body (must contain break to exit)
                    // The body should handle its own terminator (break/continue)
                    self.compile_block(
                        module,
                        loop_block_id,
                        &loop_expr.body,
                        generic_vars,
                        &mut loop_variables,
                        return_type.clone(),
                    )?;

                    // Note: compile_block will inject continue if not already present
                    let region_id = module.alloc_region(Region {
                        blocks: vec![loop_block_id],
                    });

                    let (loop_op_id, result_values) =
                        OpBuilder::new(Opcode::Loop, self.ir_location(&loop_expr.span()))
                            .operands(loop_carry_args.iter().copied())
                            .results(loop_carry_arg_tys.iter().cloned())
                            .region(region_id)
                            .build(module);
                    append_op(module, block_id, loop_op_id);

                    if result_values.len() != loop_carry_args.len() {
                        return self.jit_error_result(
                            &loop_expr.span(),
                            &format!(
                                "loop produces {} results but {} mutable variables are carried across iterations",
                                result_values.len(),
                                loop_carry_args.len()
                            ),
                        );
                    }
                    ctx.repack_some_vars(&loop_carry_vars, &result_values, true)?;
                    Ok(None)
                }
                Expr::If(if_expr) => {
                    // The condition is always bool -- don't propagate the if
                    // expression's return type into the condition.
                    let Some(conditional_val) = self.compile_expression(
                        module,
                        block_id,
                        &if_expr.cond,
                        generic_vars,
                        ctx,
                        None,
                    )?
                    else {
                        return self.jit_error_result(
                            &if_expr.cond.span(),
                            "failed to compile if-condition",
                        );
                    };
                    if let Some(bounds) = conditional_val.bounds {
                        if bounds.is_exact() {
                            // Emit the corresponding conditional, if it's defined.
                            let mut block_vars = ctx.clone();
                            // This is inlined, so no need to inject a terminator.
                            block_vars.default_terminator = None;
                            let (res, carry_vars) = match (bounds.start, &if_expr.else_branch) {
                                (1, _) => {
                                    let res = self.compile_block(
                                        module,
                                        block_id,
                                        &if_expr.then_branch,
                                        generic_vars,
                                        &mut block_vars,
                                        None,
                                    )?;
                                    let carry_vars =
                                        collect_mutated_variables_from_block(&if_expr.then_branch)?
                                            .into_iter()
                                            .collect::<Vec<_>>();
                                    (res, carry_vars)
                                }
                                (0, Some((_Else, else_expr))) => {
                                    let res = self.compile_else_branch(
                                        module,
                                        block_id,
                                        else_expr,
                                        generic_vars,
                                        &mut block_vars,
                                        None,
                                    )?;
                                    let carry_vars =
                                        collect_mutated_variables_from_expr(else_expr)?
                                            .into_iter()
                                            .collect::<Vec<_>>();
                                    (res, carry_vars)
                                }
                                _ => {
                                    // Do nothing since the conditional is false and there is no else branch.
                                    return Ok(None);
                                }
                            };
                            let result_values = block_vars.unpack_some_vars(&carry_vars)?;
                            ctx.repack_some_vars(&carry_vars, &result_values, true)?;
                            return Ok(res);
                        }
                    }

                    // The if/then block must yield captured mutable variables.
                    let then_captured_vars =
                        collect_mutated_variables_from_block(&if_expr.then_branch)?
                            .into_iter()
                            .collect::<Vec<_>>();
                    let else_captured_vars = {
                        if let Some((_Else, else_expr)) = &if_expr.else_branch {
                            collect_mutated_variables_from_expr(else_expr)?
                                .into_iter()
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        }
                    };
                    let mut if_captured_var_names = if let Some(loop_carry_vars) = &ctx.carry_vars {
                        [
                            loop_carry_vars.clone(),
                            then_captured_vars.clone(),
                            else_captured_vars.clone(),
                        ]
                        .concat()
                    } else {
                        [then_captured_vars.clone(), else_captured_vars.clone()].concat()
                    };
                    dedup(&mut if_captured_var_names);

                    let Some(condition_val) = conditional_val.value else {
                        return self.jit_error_result(
                            &if_expr.cond.span(),
                            "failed to compile if-condition",
                        );
                    };
                    // Build then region.
                    let (then_region_id, then_return_type, branch_result_type) = {
                        let mut block_vars = ctx.clone();
                        block_vars.carry_vars = Some(if_captured_var_names.clone());
                        block_vars.default_terminator = Some(BlockTerminator::Yield);
                        let (then_block_id, _then_block_args) = build_block(module, &[]);
                        let result = self.compile_block(
                            module,
                            then_block_id,
                            &if_expr.then_branch,
                            generic_vars,
                            &mut block_vars,
                            return_type.clone(),
                        )?;
                        let (branch_result_type, return_type) = {
                            if let Some(result) = result {
                                let cuda_tile_value =
                                    result.value.expect("Failed to obtain CUDA tile value.");
                                let result_ty = module.value_type(cuda_tile_value).clone();
                                (vec![result_ty], Some(result.ty.clone()))
                            } else {
                                (vec![], None)
                            }
                        };
                        let region_id = module.alloc_region(Region {
                            blocks: vec![then_block_id],
                        });
                        (region_id, return_type, branch_result_type)
                    };

                    // We don't need to check return type. Both Rust and Tile IR compiler perform this check.
                    let (else_region_id, _else_return_type) = {
                        if let Some((_Else, else_expr)) = &if_expr.else_branch {
                            let mut block_vars = ctx.clone();
                            block_vars.carry_vars = Some(if_captured_var_names.clone());
                            block_vars.default_terminator = Some(BlockTerminator::Yield);
                            let (else_block_id, _else_block_args) = build_block(module, &[]);
                            let result = self.compile_else_branch(
                                module,
                                else_block_id,
                                else_expr,
                                generic_vars,
                                &mut block_vars,
                                then_return_type.clone(),
                            )?;
                            let (_cuda_tile_return_values, return_type) = {
                                if let Some(result) = result {
                                    let cuda_tile_value =
                                        result.value.expect("Failed to obtain CUDA tile value.");
                                    (vec![cuda_tile_value], Some(result.ty.clone()))
                                } else {
                                    (vec![], None)
                                }
                            };
                            let region_id = module.alloc_region(Region {
                                blocks: vec![else_block_id],
                            });
                            (region_id, return_type)
                        } else {
                            if then_return_type.is_some() {
                                return self.jit_error_result(
                                    &if_expr.span(),
                                    "if-expression without an else branch cannot produce a return type",
                                );
                            }
                            let (else_block_id, _else_block_args) = build_block(module, &[]);
                            // If there is only a then branch, there is no return value. Yield only the captured mutable vars.
                            let captured_mutable_vars =
                                ctx.unpack_some_vars(&if_captured_var_names)?;
                            let (yield_op_id, _) =
                                OpBuilder::new(Opcode::Yield, self.ir_location(&if_expr.span()))
                                    .operands(captured_mutable_vars.iter().copied())
                                    .build(module);
                            append_op(module, else_block_id, yield_op_id);
                            let region_id = module.alloc_region(Region {
                                blocks: vec![else_block_id],
                            });
                            (region_id, None)
                        }
                    };

                    let if_result_types = {
                        let if_captured_var_args = ctx.unpack_some_vars(&if_captured_var_names)?;
                        let if_captured_var_arg_tys = if_captured_var_args
                            .iter()
                            .map(|val| module.value_type(*val).clone())
                            .collect::<Vec<_>>();
                        [if_captured_var_arg_tys, branch_result_type].concat()
                    };

                    let (if_op_id, mut result_values) =
                        OpBuilder::new(Opcode::If, self.ir_location(&if_expr.cond.span()))
                            .operand(condition_val)
                            .results(if_result_types.iter().cloned())
                            .region(then_region_id)
                            .region(else_region_id)
                            .build(module);
                    append_op(module, block_id, if_op_id);

                    if let Some(ty) = then_return_type {
                        if result_values.len() != if_captured_var_names.len() + 1 {
                            return self.jit_error_result(
                                &if_expr.span(),
                                &format!(
                                    "If expression result count ({}) does not match captured var count + 1 ({})",
                                    result_values.len(), if_captured_var_names.len() + 1
                                ),
                            );
                        }
                        let return_value = result_values.pop().unwrap();
                        ctx.repack_some_vars(&if_captured_var_names, &result_values, true)?;
                        let tr_value = TileRustValue::new_value_kind_like(return_value, ty);
                        Ok(Some(tr_value))
                    } else {
                        ctx.repack_some_vars(&if_captured_var_names, &result_values, true)?;
                        Ok(None)
                    }
                }
                Expr::Block(block_expr) => {
                    let mut inner_block_vars = ctx.clone();
                    inner_block_vars.default_terminator = None;
                    let outer_block_vars = ctx;
                    let carry_vars = collect_mutated_variables_from_block(&block_expr.block)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let result = self.compile_block(
                        module,
                        block_id,
                        &block_expr.block,
                        generic_vars,
                        &mut inner_block_vars,
                        return_type,
                    )?;
                    let result_values = inner_block_vars.unpack_some_vars(&carry_vars)?;
                    outer_block_vars.repack_some_vars(&carry_vars, &result_values, true)?;
                    // TODO (hme): Is this still needed if we're packing/unpacking above?
                    update_outer_block_type_meta(
                        &mut inner_block_vars,
                        outer_block_vars,
                        "token".to_string(),
                    );
                    Ok(result)
                }
                Expr::Unsafe(block_expr) => {
                    let mut inner_block_vars = ctx.clone();
                    inner_block_vars.default_terminator = None;
                    let outer_block_vars = ctx;
                    let carry_vars = collect_mutated_variables_from_block(&block_expr.block)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let result = self.compile_block(
                        module,
                        block_id,
                        &block_expr.block,
                        generic_vars,
                        &mut inner_block_vars,
                        return_type,
                    )?;
                    let result_values = inner_block_vars.unpack_some_vars(&carry_vars)?;
                    outer_block_vars.repack_some_vars(&carry_vars, &result_values, true)?;
                    // TODO (hme): Is this still needed if we're packing/unpacking above?
                    update_outer_block_type_meta(
                        &mut inner_block_vars,
                        outer_block_vars,
                        "token".to_string(),
                    );
                    Ok(result)
                }
                Expr::Struct(struct_expr) => {
                    let return_type = match return_type {
                        Some(return_type) => return_type,
                        None => {
                            return self.jit_error_result(
                                &struct_expr.span(),
                                "struct expressions require a known return type; try adding a type annotation",
                            )
                        }
                    };
                    let mut fields: BTreeMap<String, TileRustValue> = BTreeMap::new();
                    for field in struct_expr.fields.iter() {
                        let field_name: String = match &field.member {
                            Member::Named(named) => named.to_string(),
                            Member::Unnamed(_idx) => {
                                return self.jit_error_result(
                                    &struct_expr.span(),
                                    "unnamed (tuple) struct fields are not supported",
                                )
                            }
                        };
                        let struct_name = struct_expr.path.segments[0].ident.to_string();
                        let field_type = self
                            .modules
                            .get_struct_field_type(&struct_name, &field_name);
                        let tile_rust_ty = if let Some(field_type) = field_type {
                            // `Shape` and `Array` are compiler-known structs whose field
                            // expressions often need a concrete expected type during emission.
                            if ["Shape", "Array"].contains(&struct_name.as_str()) {
                                self.compile_type(&field_type, generic_vars, &HashMap::new())?
                            } else {
                                self.typeck_expr_tile_type(
                                    &field.expr,
                                    generic_vars,
                                    &HashMap::new(),
                                )?
                            }
                        } else {
                            self.typeck_expr_tile_type(&field.expr, generic_vars, &HashMap::new())?
                        };
                        let field_value: TileRustValue = match self.compile_expression(
                            module,
                            block_id,
                            &field.expr,
                            generic_vars,
                            ctx,
                            tile_rust_ty,
                        )? {
                            Some(field_value) => field_value,
                            None => {
                                return self.jit_error_result(
                                    &field.expr.span(),
                                    &format!("failed to compile value for field `{field_name}`"),
                                )
                            }
                        };
                        fields.insert(field_name, field_value);
                    }
                    Ok(Some(TileRustValue::new_struct(fields, return_type)))
                }
                Expr::Reference(ref_expr) => {
                    // TODO (hme): Check whether all expr types can be supported.
                    let return_type = match return_type {
                        Some(ty) => {
                            if let syn::Type::Reference(ref_type) = ty.rust_ty {
                                self.compile_type(&ref_type.elem, generic_vars, &HashMap::new())?
                            } else {
                                None
                            }
                        }
                        _ => return_type,
                    };
                    match &*ref_expr.expr {
                        Expr::Array(_array_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::Path(_path_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::Repeat(_repeat_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::MethodCall(_method_call_expr) => Ok(self.compile_expression(
                            module,
                            block_id,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        _ => self.jit_error_result(
                            &ref_expr.span(),
                            "this reference expression form is not supported",
                        ),
                    }
                }
                Expr::Tuple(tuple_expr) => {
                    let expected_elem_types = match return_type.as_ref().map(|ty| &ty.rust_ty) {
                        Some(syn::Type::Tuple(tuple_ty)) => {
                            Some(tuple_ty.elems.iter().cloned().collect::<Vec<_>>())
                        }
                        _ => None,
                    };
                    if let Some(expected_elem_types) = &expected_elem_types {
                        if expected_elem_types.len() != tuple_expr.elems.len() {
                            return self.jit_error_result(
                                &tuple_expr.span(),
                                &format!(
                                    "tuple expression has {} elements but expected tuple type has {} elements",
                                    tuple_expr.elems.len(),
                                    expected_elem_types.len()
                                ),
                            );
                        }
                    }
                    let mut rust_types: Vec<syn::Type> = vec![];
                    let mut values: Vec<TileRustValue> = vec![];
                    for (idx, elem) in tuple_expr.elems.iter().enumerate() {
                        let elem_return_type = expected_elem_types
                            .as_ref()
                            .and_then(|elem_types| elem_types.get(idx))
                            .and_then(|elem_ty| {
                                self.compile_type(elem_ty, generic_vars, &HashMap::new())
                                    .ok()
                                    .flatten()
                            });
                        match self.compile_expression(
                            module,
                            block_id,
                            elem,
                            generic_vars,
                            ctx,
                            elem_return_type,
                        )? {
                            Some(value) => {
                                rust_types.push(value.ty.rust_ty.clone());
                                values.push(value);
                            }
                            None => {
                                return self.jit_error_result(
                                    &elem.span(),
                                    "failed to compile tuple element",
                                )
                            }
                        };
                    }
                    let ty_string = rust_types
                        .iter()
                        .map(|rust_ty| rust_ty.to_token_stream().to_string())
                        .collect::<Vec<String>>()
                        .join(", ");
                    let ty: syn::Type =
                        match syn::parse2::<syn::Type>(format!("({ty_string})").parse().unwrap()) {
                            Ok(ty) => ty,
                            Err(e) => {
                                return self.jit_error_result(
                                    &tuple_expr.span(),
                                    &format!(
                                        "failed to parse inferred tuple type `({ty_string})`: {e}"
                                    ),
                                )
                            }
                        };
                    let ct_ty = match self.compile_type(&ty, generic_vars, &HashMap::new())? {
                        Some(ct_ty) => ct_ty,
                        None => {
                            return self.jit_error_result(
                                &tuple_expr.span(),
                                "unable to compile inferred tuple type",
                            )
                        }
                    };
                    Ok(Some(TileRustValue::new_compound(values, ct_ty)))
                }
                Expr::Array(array_expr) => {
                    let mut values: Vec<TileRustValue> = vec![];
                    for elem in &array_expr.elems {
                        let elem_ty = match &return_type {
                            Some(return_type) => {
                                match &return_type.rust_ty {
                                    syn::Type::Array(array_type) => self.compile_type(
                                        &array_type.elem,
                                        generic_vars,
                                        &HashMap::new(),
                                    )?,
                                    syn::Type::Slice(slice) => {
                                        // TODO (hme): Confirm this is right.
                                        self.compile_type(
                                            &slice.elem,
                                            generic_vars,
                                            &HashMap::new(),
                                        )?
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &elem.span(),
                                            &format!(
                                                "unexpected element type `{}`",
                                                return_type.rust_ty.to_token_stream()
                                            ),
                                        )
                                    }
                                }
                            }
                            None => None,
                        };
                        match self.compile_expression(
                            module,
                            block_id,
                            elem,
                            generic_vars,
                            ctx,
                            elem_ty,
                        )? {
                            Some(value) => values.push(value),
                            None => {
                                return self.jit_error_result(
                                    &elem.span(),
                                    "failed to compile array element",
                                )
                            }
                        };
                    }
                    let return_type = if let Some(return_type) = return_type {
                        return_type
                    } else {
                        if values.is_empty() {
                            return self.jit_error_result(
                                &array_expr.span(),
                                "unable to infer type for empty array; add a type annotation",
                            );
                        }
                        let ty: &TileRustType = &values[0].ty;
                        let ty_string = ty.rust_ty.to_token_stream().to_string();
                        let ty: syn::Type = match syn::parse2::<syn::Type>(
                            format!("[{ty_string}]").parse().unwrap(),
                        ) {
                            Ok(ty) => ty,
                            Err(e) => {
                                return self.jit_error_result(
                                    &array_expr.span(),
                                    &format!(
                                        "failed to parse inferred array type `[{ty_string}]`: {e}"
                                    ),
                                )
                            }
                        };
                        match self.compile_type(&ty, generic_vars, &HashMap::new())? {
                            Some(ct_ty) => ct_ty,
                            None => {
                                return self.jit_error_result(
                                    &array_expr.span(),
                                    "unable to compile inferred array type",
                                )
                            }
                        }
                    };
                    Ok(Some(TileRustValue::new_compound(values, return_type)))
                }
                Expr::Repeat(repeat_expr) => {
                    let len = {
                        let len_expr = &*repeat_expr.len;
                        if let Expr::Path(len_expr) = len_expr {
                            let var_name = len_expr.path.segments.last().unwrap().ident.to_string();
                            // Expecting a const generic primitive.
                            let Some(n) = generic_vars.get_i32(var_name.as_str()) else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    &format!("expected a const generic value for repeat length, but `{var_name}` is not a known const generic"),
                                );
                            };
                            n as usize
                        } else {
                            let Expr::Lit(lit_expr) = len_expr else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    "repeat length must be a literal or const generic",
                                );
                            };
                            let Lit::Int(int_lit) = &lit_expr.lit else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    "repeat length must be an integer literal",
                                );
                            };
                            let Ok(len) = int_lit.base10_parse::<usize>() else {
                                return self.jit_error_result(
                                    &repeat_expr.len.span(),
                                    "failed to parse repeat length as a valid integer",
                                );
                            };
                            len
                        }
                    };
                    let elem_return_type = match return_type.as_ref() {
                        Some(return_type) => {
                            self.expected_array_element_type(return_type, generic_vars)?
                        }
                        None => self.typeck_expr_tile_type(
                            &repeat_expr.expr,
                            generic_vars,
                            &HashMap::new(),
                        )?,
                    };
                    let Some(value) = self.compile_expression(
                        module,
                        block_id,
                        &repeat_expr.expr,
                        generic_vars,
                        ctx,
                        elem_return_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &repeat_expr.expr.span(),
                            "failed to compile repeat expression element",
                        );
                    };
                    let values: Vec<TileRustValue> = vec![value; len];
                    let return_type = if let Some(return_type) = return_type {
                        return_type
                    } else {
                        if values.is_empty() {
                            return self.jit_error_result(
                                &repeat_expr.span(),
                                "unable to infer type for zero-length repeat expression; add a type annotation",
                            );
                        }
                        let ty: &TileRustType = &values[0].ty;
                        let ty_string = ty.rust_ty.to_token_stream().to_string();
                        let ty: syn::Type = match syn::parse2::<syn::Type>(
                            format!("[{ty_string}]").parse().unwrap(),
                        ) {
                            Ok(ty) => ty,
                            Err(e) => {
                                return self.jit_error_result(
                                    &repeat_expr.span(),
                                    &format!(
                                        "failed to parse inferred repeat type `[{ty_string}]`: {e}"
                                    ),
                                )
                            }
                        };
                        match self.compile_type(&ty, generic_vars, &HashMap::new())? {
                            Some(ct_ty) => ct_ty,
                            None => {
                                return self.jit_error_result(
                                    &repeat_expr.span(),
                                    "unable to compile inferred repeat type",
                                )
                            }
                        }
                    };
                    Ok(Some(TileRustValue::new_compound(values, return_type)))
                }
                Expr::Path(path_expr) => {
                    let var_name = path_expr.path.segments.last().unwrap().ident.to_string();

                    // Handle None specially — Rust Option::None, not a variable.
                    if path_expr.path.segments.len() == 1 && var_name == "None" {
                        if let Some(return_type) = return_type {
                            if return_type.kind == Kind::Enum {
                                return Ok(Some(TileRustValue::new_enum(
                                    "None",
                                    None,
                                    return_type,
                                )));
                            }
                        }
                        return Ok(None);
                    }

                    // 1. Local variable (single-segment paths, locals shadow module items).
                    if path_expr.path.segments.len() == 1 {
                        if let Some(value) = ctx.vars.get(&var_name) {
                            return Ok(Some(value.clone()));
                        }
                    }

                    // 2. Resolve via name resolver (module-level structs, functions, etc.).
                    let res = self
                        .modules
                        .name_resolver
                        .resolve_path(&path_expr.path, &self.module_name);
                    match res {
                        Res::Def(DefKind::Struct, _) => {
                            // Known DSL struct — return as ZST marker placeholder.
                            return Ok(Some(Self::make_zst_marker(path_expr)));
                        }
                        Res::Def(DefKind::Const, def_id) => {
                            let Some(const_item) = self.modules.name_resolver.get_const(&def_id)
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!("failed to resolve const `{var_name}`"),
                                );
                            };
                            let const_ty =
                                self.compile_type(&const_item.ty, generic_vars, &HashMap::new())?;
                            return self.compile_expression(
                                module,
                                block_id,
                                &const_item.expr,
                                generic_vars,
                                ctx,
                                const_ty,
                            );
                        }
                        Res::Def(DefKind::Static, def_id) => {
                            let Some(static_item) = self.modules.name_resolver.get_static(&def_id)
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!("failed to resolve static `{var_name}`"),
                                );
                            };
                            let Some(static_ty) =
                                self.compile_type(&static_item.ty, generic_vars, &HashMap::new())?
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!("failed to compile static `{var_name}` type"),
                                );
                            };
                            return Ok(Some(TileRustValue::new_struct(BTreeMap::new(), static_ty)));
                        }
                        _ => {}
                    }

                    // 3. Multi-segment path not in the resolver — treat as a ZST
                    //    marker type from a nested Rust module (ftz::Enabled,
                    //    rounding::NearestEven, nan::Disabled, etc.). These modules
                    //    are defined outside the #[cutile::module] block and aren't
                    //    in the DSL AST the resolver indexes. They're valid Rust
                    //    type paths consumed by resolve_static_params.
                    if path_expr.path.segments.len() > 1 {
                        if self.path_looks_like_associated_const(path_expr, generic_vars) {
                            return self.jit_error_result(
                                &path_expr.span(),
                                "associated const values are not supported in expression position; use a literal or pass supported element constants such as `T::ZERO` directly to a DSL operation that accepts them",
                            );
                        }
                        return Ok(Some(Self::make_zst_marker(path_expr)));
                    }

                    // 4. Single-segment, not a local, not in resolver — error.
                    let suggestion = self.modules.name_resolver.find_all_definitions(&var_name);
                    if suggestion.is_empty() {
                        self.jit_error_result(
                            &path_expr.span(),
                            &format!("undefined variable `{var_name}`"),
                        )
                    } else {
                        self.jit_error_result(
                            &path_expr.span(),
                            &format!(
                                "undefined variable `{var_name}` (did you mean the function defined in {}?)",
                                suggestion.join(", ")
                            ),
                        )
                    }
                }
                Expr::Call(call_expr) => {
                    let call_expr_func_str = call_expr.func.to_token_stream().to_string();
                    let _args_str = call_expr.args.to_token_stream().to_string();
                    match &*call_expr.func {
                        Expr::Path(path_expr) => {
                            if Self::is_dim_new_call(&call_expr.func) {
                                return self.compile_dim_new_call(
                                    module,
                                    block_id,
                                    call_expr,
                                    generic_vars,
                                    ctx,
                                    return_type,
                                );
                            }
                            let ident = get_ident_from_path_expr(path_expr);
                            // Handle Some(...) specially - it's a Rust Option constructor, not a function call
                            if ident.to_string() == "Some" {
                                if call_expr.args.len() != 1 {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "`Some()` expects exactly one argument, got {}",
                                            call_expr.args.len()
                                        ),
                                    );
                                }

                                if let Some(return_type) = return_type {
                                    if return_type.kind == Kind::Enum {
                                        return Ok(Some(TileRustValue::new_enum(
                                            "Some",
                                            Some(call_expr.args[0].clone()),
                                            return_type,
                                        )));
                                    }
                                }

                                let Some(payload_value) = self.compile_expression(
                                    module,
                                    block_id,
                                    &call_expr.args[0],
                                    generic_vars,
                                    ctx,
                                    None,
                                )?
                                else {
                                    return self.jit_error_result(
                                        &call_expr.args[0].span(),
                                        "failed to compile `Some` payload",
                                    );
                                };
                                let option_type =
                                    Self::make_option_type_from_payload(&payload_value.ty);
                                return Ok(Some(TileRustValue::new_enum(
                                    "Some",
                                    Some(call_expr.args[0].clone()),
                                    option_type,
                                )));
                            }
                            if self
                                .modules
                                .get_cuda_tile_op_attrs(ident.to_string().as_str())
                                .is_some()
                            {
                                Ok(self.compile_cuda_tile_op_call(
                                    module,
                                    block_id,
                                    call_expr,
                                    generic_vars,
                                    ctx,
                                    return_type,
                                )?)
                            } else if let Some((module_name, fn_item)) = self
                                .modules
                                .get_function_by_name(ident.to_string().as_str())
                            {
                                if let Some(compiler_op_attrs) =
                                    get_meta_list("cuda_tile :: compiler_op", &fn_item.attrs)
                                {
                                    Ok(self.compile_compiler_op_call(
                                        module,
                                        block_id,
                                        call_expr,
                                        path_expr,
                                        fn_item,
                                        &compiler_op_attrs,
                                        generic_vars,
                                        ctx,
                                        return_type,
                                    )?)
                                } else {
                                    Ok(self.inline_function_call(
                                        module,
                                        block_id,
                                        module_name,
                                        fn_item,
                                        call_expr,
                                        generic_vars,
                                        ctx,
                                        return_type,
                                    )?)
                                }
                            } else {
                                self.jit_error_result(
                                    &call_expr.func.span(),
                                    &format!("call to `{}` is not supported", &call_expr_func_str),
                                )
                            }
                        }
                        _ => self.jit_error_result(
                            &call_expr.func.span(),
                            &format!("Call to {} not supported.", &call_expr_func_str),
                        ),
                    }
                }
                Expr::MethodCall(method_call_expr) => {
                    if let Some(value) = self.compile_into_dim_method(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(Some(value));
                    }
                    if let Some(value) = self.compile_partition_with_bounds_method(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(Some(value));
                    }
                    if let Some(value) = self.compile_global_method_call(
                        module,
                        block_id,
                        &method_call_expr,
                        &generic_vars,
                        ctx,
                        return_type.clone(),
                    )? {
                        return Ok(Some(value));
                    }
                    Ok(self.inline_method_call(
                        module,
                        block_id,
                        method_call_expr,
                        generic_vars,
                        ctx,
                        return_type,
                    )?)
                }
                Expr::Field(field_expr) => {
                    let Some(base) = self.compile_expression(
                        module,
                        block_id,
                        &field_expr.base,
                        generic_vars,
                        ctx,
                        None,
                    )?
                    else {
                        return self.jit_error_result(
                            &field_expr.base.span(),
                            "failed to compile the receiver of this field access",
                        );
                    };
                    match &field_expr.member {
                        Member::Named(field_name) => {
                            if base.kind != Kind::Struct {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "expected a struct value for field access",
                                );
                            }
                            if base.fields.is_none() {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "struct is missing its field data (internal)",
                                );
                            }
                            let fields = &base.fields.clone().unwrap();
                            let Some(field_value) = fields.get(&field_name.to_string()) else {
                                return self.jit_error_result(
                                    &field_name.span(),
                                    &format!("{} is not a field.", field_name),
                                );
                            };
                            Ok(Some(field_value.clone()))
                        }
                        Member::Unnamed(idx) => {
                            if base.kind != Kind::Compound {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "expected a tuple or compound value for indexed field access",
                                );
                            }
                            if base.values.is_none() {
                                return self.jit_error_result(
                                    &field_expr.base.span(),
                                    "compound value is missing its element list (internal)",
                                );
                            }
                            let values = base.values.as_ref().unwrap();
                            let index = idx.index as usize;
                            let value: Option<&TileRustValue> = values.get(index);
                            if value.is_none() {
                                return self.jit_error_result(
                                    &field_expr.span(),
                                    &format!(
                                        "Index {index} access failed with {} elements.",
                                        values.len()
                                    ),
                                );
                            }
                            Ok(Some(value.unwrap().clone()))
                        }
                    }
                }
                Expr::Unary(unary_expr) => {
                    let UnOp::Neg(_) = unary_expr.op else {
                        return self.jit_error_result(
                            &unary_expr.span(),
                            "Unary expression not supported",
                        );
                    };
                    match &*unary_expr.expr {
                        Expr::Lit(lit_expr) => {
                            let return_type = if return_type.is_none() {
                                match get_lit_type(lit_expr) {
                                    Some(ty) => {
                                        self.compile_type(&ty, generic_vars, &HashMap::new())?
                                    }
                                    None => None,
                                }
                            } else {
                                return_type
                            };
                            let Some(return_type) = return_type else {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "Failed to infer type for unary op expr.",
                                );
                            };
                            let (lit_string, bounds) = match &lit_expr.lit {
                                Lit::Float(float_lit) => {
                                    (format!("-{}", float_lit.base10_digits()), None)
                                }
                                Lit::Int(int_lit) => {
                                    let str = format!("-{}", int_lit.base10_digits());
                                    let val = -int_lit
                                        .base10_parse::<i32>()
                                        .unwrap_or_else(|_| panic!("Failed to parse literal {str}"))
                                        as i64;
                                    (str, Some(Bounds::exact(val)))
                                }
                                _ => {
                                    return self.jit_error_result(
                                        &lit_expr.span(),
                                        "Lit expression not implemented",
                                    )
                                }
                            };
                            let Some(cuda_tile_ty) = return_type
                                .get_cuda_tile_element_type(self.modules.primitives())?
                            else {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "unable to determine type for numeric literal; add a type annotation",
                                );
                            };

                            // Build Constant op with proper DenseElements encoding.
                            let (op_result, _tile_ir_ty) = build_constant_op(
                                module,
                                block_id,
                                &lit_string,
                                &cuda_tile_ty,
                                self.ir_location(&lit_expr.span()),
                            );

                            let rust_ty = return_type.rust_ty;
                            let ct_type =
                                self.compile_type(&rust_ty, generic_vars, &HashMap::new())?;
                            if ct_type.is_none() {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "failed to compile the type of this literal",
                                );
                            }
                            let ct_type = ct_type.unwrap();
                            if ct_type.kind != Kind::PrimitiveType {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    &format!(
                                        "expected a scalar type for this literal, got {:?}",
                                        ct_type.kind
                                    ),
                                );
                            }
                            Ok(Some(TileRustValue::new_primitive(
                                op_result, ct_type, bounds,
                            )))
                        }
                        _ => self.jit_error_result(
                            &unary_expr.span(),
                            "Non-const unary expressions not supported.",
                        ),
                    }
                }
                Expr::Cast(cast_expr) => {
                    let src_expr = self
                        .compile_expression(
                            module,
                            block_id,
                            &cast_expr.expr,
                            generic_vars,
                            ctx,
                            None,
                        )?
                        .unwrap();
                    let src_elem_ty: String = src_expr
                        .ty
                        .get_instantiated_rust_element_type(self.modules.primitives())
                        .unwrap();
                    let dst_elem_ty: String = get_rust_element_type_primitive(&cast_expr.ty);
                    match (src_elem_ty.as_str(), dst_elem_ty.as_str()) {
                        ("i32", "u32") => {}
                        ("i64", "u64") => {}
                        ("i32", "usize") => {}
                        ("usize", "i32") => {}
                        _ => {
                            return self.jit_error_result(
                                &cast_expr.span(),
                                &format!(
                                    "unsupported cast from `{src_elem_ty}` to `{dst_elem_ty}`"
                                ),
                            )
                        }
                    }
                    Ok(Some(src_expr))
                }
                Expr::Lit(lit_expr) => {
                    let return_type = if return_type.is_none() {
                        let typeck_return_type =
                            self.typeck_expr_tile_type(expr, generic_vars, &HashMap::new())?;
                        if typeck_return_type.is_some() {
                            typeck_return_type
                        } else {
                            match get_lit_type(lit_expr) {
                                Some(ty) => {
                                    self.compile_type(&ty, generic_vars, &HashMap::new())?
                                }
                                None => None,
                            }
                        }
                    } else {
                        return_type
                    };
                    let Some(return_type) = return_type else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!(
                                "Failed to infer type for lit expr {}.",
                                lit_expr.to_token_stream()
                            ),
                        );
                    };
                    if let Lit::Str(_) = &lit_expr.lit {
                        return Ok(Some(TileRustValue::new_string(
                            Expr::Lit(lit_expr.clone()),
                            return_type,
                        )));
                    }
                    let (lit_string, bounds) = match &lit_expr.lit {
                        Lit::Float(float_lit) => (float_lit.base10_digits().to_string(), None),
                        Lit::Int(int_lit) => {
                            let str = int_lit.base10_digits().to_string();
                            let val = int_lit
                                .base10_parse::<i32>()
                                .unwrap_or_else(|_| panic!("Failed to parse literal {str}"))
                                as i64;
                            (str, Some(Bounds::exact(val)))
                        }
                        Lit::Bool(bool_lit) => (format!("{}", bool_lit.value as i32), None),
                        _ => {
                            return self.jit_error_result(
                                &lit_expr.span(),
                                "Lit expression not implemented",
                            )
                        }
                    };
                    let Some(cuda_tile_ty) =
                        return_type.get_cuda_tile_element_type(self.modules.primitives())?
                    else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            "unable to determine type for numeric literal; add a type annotation",
                        );
                    };

                    // Build Constant op with proper DenseElements encoding.
                    let (op_result, _tile_ir_ty) = build_constant_op(
                        module,
                        block_id,
                        &lit_string,
                        &cuda_tile_ty,
                        self.ir_location(&lit_expr.span()),
                    );

                    let rust_ty = return_type.rust_ty;
                    let ct_type = self.compile_type(&rust_ty, generic_vars, &HashMap::new())?;
                    if ct_type.is_none() {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            "failed to compile the type of this literal",
                        );
                    }
                    let ct_type = ct_type.unwrap();
                    if ct_type.kind != Kind::PrimitiveType {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!(
                                "expected a scalar type for this literal, got {:?}",
                                ct_type.kind
                            ),
                        );
                    }
                    Ok(Some(TileRustValue::new_primitive(
                        op_result, ct_type, bounds,
                    )))
                }
                Expr::Binary(bin_expr) => {
                    // These are type-checked by Rust, so just do whatever the expression is asking.
                    Ok(self.compile_binary_op(
                        module,
                        block_id,
                        bin_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?)
                }
                Expr::Paren(paren_expr) => Ok(self.compile_expression(
                    module,
                    block_id,
                    &paren_expr.expr,
                    generic_vars,
                    ctx,
                    return_type.clone(),
                )?),
                Expr::Macro(mac_expr) => {
                    let last_seg = mac_expr.mac.path.segments.last();
                    if last_seg.is_none() {
                        return self.jit_error_result(
                            &mac_expr.mac.path.span(),
                            "unrecognized macro invocation",
                        );
                    }
                    let last_seg = last_seg.unwrap();
                    let mac_name = last_seg.ident.to_string();
                    Ok(match mac_name.as_str() {
                        "const_shape" | "const_array" => {
                            // TODO (hme): Remove special case for const_shape here
                            //  and on the proc-macro side (rank_instantiation.rs).
                            let args = self.const_shape_macro_args(mac_expr, generic_vars, ctx)?;
                            let cga_str = format!("{{[{}]}}", args.join(", "));
                            let ty_str = if mac_name == "const_shape" {
                                "Shape"
                            } else {
                                "Array"
                            };
                            let shape_expr = syn::parse2::<Expr>(
                                format!("{ty_str}::<{cga_str}>{{dims: &[]}}")
                                    .parse()
                                    .unwrap(),
                            )
                            .unwrap();
                            let return_type = if return_type.is_none() {
                                let shape_str = format!("{ty_str}<{cga_str}>");
                                let shape_ty =
                                    syn::parse2::<syn::Type>(shape_str.parse().unwrap()).unwrap();
                                self.compile_type(&shape_ty, generic_vars, &HashMap::new())?
                            } else {
                                return_type.clone()
                            };
                            self.compile_expression(
                                module,
                                block_id,
                                &shape_expr,
                                generic_vars,
                                ctx,
                                return_type,
                            )?
                        }
                        _ => self.compile_cuda_tile_macro(
                            module,
                            block_id,
                            &mac_expr.mac,
                            generic_vars,
                            ctx,
                            return_type.clone(),
                        )?,
                    })
                }
                Expr::Closure(closure_expr) => {
                    // Closures cannot be used as standalone expressions in CUDA Tile.
                    // They are only supported as arguments to specific operations (e.g., reduce, scan)
                    // that compile them into tile-ir regions.
                    self.jit_error_result(
                        &closure_expr.span(),
                        "closures are not supported as standalone values; \
                         they can only be used as arguments to operations like `reduce()` or `scan()`",
                    )
                }
                Expr::Index(index_expr) => {
                    let Some(expr_val) = self.compile_expression(
                        module,
                        block_id,
                        &index_expr.expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?
                    else {
                        return self.jit_error_result(
                            &index_expr.expr.span(),
                            "failed to compile the indexed expression",
                        );
                    };
                    // TODO (hme): Revisit this once we have proper type inference.
                    let i32_type: syn::Type = parse_quote! { i32 };
                    let i32_type = self.compile_type(&i32_type, generic_vars, &HashMap::new())?;
                    let Some(index_val) = self.compile_expression(
                        module,
                        block_id,
                        &index_expr.index,
                        generic_vars,
                        ctx,
                        i32_type,
                    )?
                    else {
                        return self.jit_error_result(
                            &index_expr.index.span(),
                            "failed to compile index value",
                        );
                    };
                    let idx: i32 = {
                        let Some(index_bounds) = index_val.bounds else {
                            return self.jit_error_result(
                                &index_expr.index.span(),
                                "dynamic indices are not supported; the index must be a compile-time constant",
                            );
                        };
                        if !index_bounds.is_exact() {
                            return self.jit_error_result(
                                &index_expr.index.span(),
                                "index must be a compile-time constant with exact bounds",
                            );
                        }
                        index_bounds.start as i32
                    };
                    if idx < 0 {
                        return self.jit_error_result(
                            &index_expr.index.span(),
                            &format!("index must be non-negative, got {idx}"),
                        );
                    }
                    if expr_val.kind == Kind::Compound {
                        let Some(mut values) = expr_val.values else {
                            return self.jit_error_result(
                                &index_expr.expr.span(),
                                "internal: compound value is missing its element list during index access",
                            );
                        };
                        let index = idx as usize;
                        if index >= values.len() {
                            return self.jit_error_result(
                                &index_expr.index.span(),
                                &format!(
                                    "index {idx} out of bounds for compound value of length {}",
                                    values.len()
                                ),
                            );
                        }
                        return Ok(Some(values.remove(index)));
                    }
                    if let Some(fields) = expr_val.fields.as_ref() {
                        if let Some(dims) = fields.get("dims") {
                            let Some(mut values) = dims.values.clone() else {
                                return self.jit_error_result(
                                    &index_expr.expr.span(),
                                    "shape-like value has a `dims` field that is not indexable",
                                );
                            };
                            let index = idx as usize;
                            if index >= values.len() {
                                return self.jit_error_result(
                                    &index_expr.index.span(),
                                    &format!(
                                        "index {idx} out of bounds for shape of rank {}",
                                        values.len()
                                    ),
                                );
                            }
                            return Ok(Some(values.remove(index)));
                        }
                    }
                    return self.jit_error_result(
                        &index_expr.expr.span(),
                        "indexing is only supported on tuple/compound values and shape-like descriptors",
                    );
                }
                _ => self.jit_error_result(&expr.span(), "this expression form is not supported"),
            }
        }) // stacker::maybe_grow
    }
}

/// Convert a CUDA Tile element type string (e.g. "f32", "i32") to a tile-ir scalar tile Type.
fn cuda_tile_element_type_to_tile_ir(cuda_tile_ty: &str) -> cutile_ir::ir::Type {
    use cutile_ir::ir::{ScalarType, TileElementType, TileType, Type};
    let scalar = super::_type::scalar_from_name(cuda_tile_ty).unwrap_or(ScalarType::I32);
    Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(scalar),
    })
}

/// Build a Constant op with a proper DenseElements value attribute.
/// `lit_string` is the numeric literal as text (e.g. "42", "-3.14", "0x3f800000").
/// `cuda_tile_ty` is the element type name (e.g. "f32", "i32").
fn build_constant_op(
    module: &mut cutile_ir::ir::Module,
    block_id: cutile_ir::ir::BlockId,
    lit_string: &str,
    cuda_tile_ty: &str,
    location: Location,
) -> (cutile_ir::ir::Value, cutile_ir::ir::Type) {
    use cutile_ir::ir::DenseElements;

    let result_ty = cuda_tile_element_type_to_tile_ir(cuda_tile_ty);
    let data = encode_literal_bytes(lit_string, cuda_tile_ty);

    let (op_id, results) = OpBuilder::new(Opcode::Constant, location)
        .result(result_ty.clone())
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: result_ty.clone(),
                shape: vec![],
                data,
            }),
        )
        .build(module);
    cutile_ir::builder::append_op(module, block_id, op_id);
    (results[0], result_ty)
}

/// Encode a literal value string into bytes for a DenseElements attribute.
pub fn encode_literal_bytes(lit_string: &str, cuda_tile_ty: &str) -> Vec<u8> {
    use cutile_ir::ir::ScalarType;
    let scalar = super::_type::scalar_from_name(cuda_tile_ty).unwrap_or(ScalarType::I32);
    match scalar {
        ScalarType::I1 => vec![if lit_string != "0" { 0xFF } else { 0x00 }],
        ScalarType::I8 => {
            let v: i8 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::I16 => {
            let v: i16 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::I32 => {
            let v: i32 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::I64 => {
            let v: i64 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
        ScalarType::F16 => {
            let v = parse_float_or_hex(lit_string);
            half::f16::from_f64(v).to_le_bytes().to_vec()
        }
        ScalarType::BF16 => {
            let v = parse_float_or_hex(lit_string);
            half::bf16::from_f64(v).to_le_bytes().to_vec()
        }
        ScalarType::F32 => {
            let v = parse_float_or_hex(lit_string);
            (v as f32).to_le_bytes().to_vec()
        }
        ScalarType::F64 | ScalarType::TF32 => {
            let v = parse_float_or_hex(lit_string);
            v.to_le_bytes().to_vec()
        }
        _ => {
            let v: i32 = lit_string.parse().unwrap_or(0);
            v.to_le_bytes().to_vec()
        }
    }
}

/// Parse a float literal string, handling both decimal ("3.14") and hex ("0x40490fdb") forms.
fn parse_float_or_hex(s: &str) -> f64 {
    if s.starts_with("0x") || s.starts_with("-0x") {
        let negative = s.starts_with('-');
        let hex = if negative { &s[3..] } else { &s[2..] };
        let bits = u64::from_str_radix(hex, 16).unwrap_or(0);
        let v = match hex.len() {
            1..=4 => half::f16::from_bits(bits as u16).to_f64(),
            5..=8 => f32::from_bits(bits as u32) as f64,
            _ => f64::from_bits(bits),
        };
        if negative {
            -v
        } else {
            v
        }
    } else {
        s.parse::<f64>().unwrap_or(0.0)
    }
}
