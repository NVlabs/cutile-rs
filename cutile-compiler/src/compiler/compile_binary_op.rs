/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Binary operation compilation: handles `compile_binary_op` and `compile_binary_op_from_values`
//! within the CUDA Tile compiler. This module covers the translation of binary arithmetic,
//! comparison, and bitwise operations into MLIR operations.

use quote::ToTokens;
use syn::spanned::Spanned;

use crate::bounds::bounds_from_bop;
use crate::compiler::_function::CUDATileFunctionCompiler;
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;
use crate::compiler::utils::{
    cuda_tile_tile_ty_from_type_instance, get_cmp_predicate_attr, get_tile_bop_from_rust_bop,
    TileBinaryOp,
};
use crate::error::JITError;
use crate::generics::GenericVars;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::{self, BlockLike, Value};
use std::collections::HashMap;
use syn::ExprBinary;

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    pub fn compile_binary_op(
        &'c self,
        builder: &'c ir::Block<'c>,
        bin_expr: &ExprBinary,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let lhs = self.compile_expression(
            builder,
            &bin_expr.left,
            generic_vars,
            ctx,
            return_type.clone(),
        )?;
        if lhs.is_none() {
            return self.jit_error_result(
                &bin_expr.left.span(),
                "failed to compile the left-hand side of this binary operation",
            );
        }
        let lhs = lhs.unwrap();
        let rhs = self.compile_expression(
            builder,
            &bin_expr.right,
            generic_vars,
            ctx,
            return_type.clone(),
        )?;
        if rhs.is_none() {
            return self.jit_error_result(
                &bin_expr.right.span(),
                "failed to compile the right-hand side of this binary operation",
            );
        }
        let rhs = rhs.unwrap();
        Ok(Some(self.compile_binary_op_from_values(
            builder,
            lhs,
            rhs,
            &get_tile_bop_from_rust_bop(&bin_expr.op)?,
            generic_vars,
            ctx,
            return_type,
            &bin_expr.span(),
        )?))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compile_binary_op_from_values(
        &'c self,
        builder: &'c ir::Block<'c>,
        lhs: TileRustValue<'c, 'c>,
        rhs: TileRustValue<'c, 'c>,
        tile_rust_arithmetic_op: &TileBinaryOp,
        generic_vars: &GenericVars,
        _ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue<'c, 'c>, JITError> {
        if lhs.ty.rust_ty != rhs.ty.rust_ty {
            return self.jit_error_result(
                span,
                &format!(
                    "binary `{:?}` requires operands of the same type, but got `{}` and `{}`",
                    tile_rust_arithmetic_op,
                    lhs.ty.rust_ty.to_token_stream(),
                    rhs.ty.rust_ty.to_token_stream()
                ),
            );
        }
        let lhs_value = lhs.value;
        if lhs_value.is_none() {
            return self.jit_error_result(
                span,
                "left-hand side of binary operation did not produce a value",
            );
        }
        let lhs_value = lhs_value.unwrap();
        let rhs_value = rhs.value;
        if rhs_value.is_none() {
            return self.jit_error_result(
                span,
                "right-hand side of binary operation did not produce a value",
            );
        }
        let rhs_value = rhs_value.unwrap();
        let operand_type = lhs.ty.clone();
        let operand_rust_ty = &operand_type.rust_ty;
        let Some(operand_rust_element_type) =
            operand_type.get_instantiated_rust_element_type(&self.modules.primitives)
        else {
            return self.jit_error_result(
                span,
                &format!(
                    "unable to determine element type for `{:?}` on `{}`",
                    tile_rust_arithmetic_op,
                    operand_type.rust_ty.to_token_stream()
                ),
            );
        };
        let Some(operand_cuda_tile_ty) = operand_type.cuda_tile_ty else {
            return self.jit_error_result(
                span,
                &format!(
                    "type `{}` cannot be used with binary `{:?}`",
                    operand_type.rust_ty.to_token_stream(),
                    tile_rust_arithmetic_op
                ),
            );
        };
        let Some(operand_cuda_tile_element_type) =
            operand_type.get_cuda_tile_element_type(&self.modules.primitives)?
        else {
            return self.jit_error_result(
                span,
                &format!(
                    "unable to determine compiled element type for `{:?}`",
                    tile_rust_arithmetic_op
                ),
            );
        };
        let mut is_cmp = false;
        let signedness_str = match operand_rust_element_type.as_str() {
            "bool" | "u32" | "u64" => "unsigned",
            _ => "signed",
        };
        let signedness_attr = self.parse_named_attr(
            "signedness",
            format!("#cuda_tile.signedness<{signedness_str}>").as_str(),
        )?;

        let op_builder = match operand_cuda_tile_element_type.as_ref() {
            "i1" | "i4" | "i8" | "i32" | "i64" => {
                // TODO (hme): Add i4, i8, i16 support, as needed.
                if let Some(comparison_predicate) =
                    get_cmp_predicate_attr(&self.context, tile_rust_arithmetic_op)?
                {
                    is_cmp = true;
                    let cuda_tile_bool_ty = cuda_tile_tile_ty_from_type_instance(
                        &self.context,
                        &lhs.ty.type_instance,
                        &self.modules.primitives,
                        Some("i1"),
                    )?;
                    OperationBuilder::new("cuda_tile.cmpi", self.function_location())
                        .add_attributes(&[comparison_predicate, signedness_attr])
                        .add_operands(&[lhs_value, rhs_value])
                        .add_results(&[cuda_tile_bool_ty])
                } else {
                    // If both operands have bounds, we can generate bounds on the output.
                    match tile_rust_arithmetic_op {
                        TileBinaryOp::Min => {
                            OperationBuilder::new("cuda_tile.mini", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_attributes(&[signedness_attr])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Max => {
                            OperationBuilder::new("cuda_tile.maxi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_attributes(&[signedness_attr])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Add => {
                            OperationBuilder::new("cuda_tile.addi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Sub => {
                            OperationBuilder::new("cuda_tile.subi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Mul => {
                            OperationBuilder::new("cuda_tile.muli", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Rem => {
                            OperationBuilder::new("cuda_tile.remi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_attributes(&[signedness_attr])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Div => {
                            let rounding_mode_attr = self.parse_named_attr(
                                "rounding_mode",
                                "#cuda_tile.rounding<negative_inf>",
                            )?;
                            OperationBuilder::new("cuda_tile.divi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                                .add_attributes(&[signedness_attr, rounding_mode_attr])
                        }
                        TileBinaryOp::CeilDiv => {
                            let rounding_mode_attr = self.parse_named_attr(
                                "rounding_mode",
                                "#cuda_tile.rounding<positive_inf>",
                            )?;
                            OperationBuilder::new("cuda_tile.divi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                                .add_attributes(&[signedness_attr, rounding_mode_attr])
                        }
                        TileBinaryOp::BitAnd => {
                            OperationBuilder::new("cuda_tile.andi", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::BitOr => {
                            OperationBuilder::new("cuda_tile.ori", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::BitXor => {
                            OperationBuilder::new("cuda_tile.xori", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        _ => {
                            return self.jit_error_result(
                                span,
                                &format!("Unimplemented binary op {tile_rust_arithmetic_op:#?}"),
                            );
                        }
                    }
                }
            }
            "bf16" | "f16" | "f32" | "f64" => {
                if let Some(comparison_predicate) =
                    get_cmp_predicate_attr(&self.context, tile_rust_arithmetic_op)?
                {
                    let comparison_ordering = self.parse_named_attr(
                        "comparison_ordering",
                        "#cuda_tile.comparison_ordering<ordered>",
                    )?;
                    is_cmp = true;
                    let cuda_tile_bool_ty = cuda_tile_tile_ty_from_type_instance(
                        &self.context,
                        &lhs.ty.type_instance,
                        &self.modules.primitives,
                        Some("i1"),
                    )?;
                    OperationBuilder::new("cuda_tile.cmpf", self.function_location())
                        .add_attributes(&[comparison_predicate, comparison_ordering])
                        .add_operands(&[lhs_value, rhs_value])
                        .add_results(&[cuda_tile_bool_ty])
                } else {
                    let default_rounding_mode_attr = self
                        .parse_named_attr("rounding_mode", "#cuda_tile.rounding<nearest_even>")?;
                    let attrs = vec![default_rounding_mode_attr];
                    match tile_rust_arithmetic_op {
                        TileBinaryOp::Min => {
                            OperationBuilder::new("cuda_tile.minf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Max => {
                            OperationBuilder::new("cuda_tile.maxf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Add => {
                            OperationBuilder::new("cuda_tile.addf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Sub => {
                            OperationBuilder::new("cuda_tile.subf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Mul => {
                            OperationBuilder::new("cuda_tile.mulf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Rem => {
                            OperationBuilder::new("cuda_tile.remf", self.function_location())
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::Div => {
                            OperationBuilder::new("cuda_tile.divf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        TileBinaryOp::TrueDiv => {
                            let rounding_mode_attr = self
                                .parse_named_attr("rounding_mode", "#cuda_tile.rounding<approx>")?;
                            let attrs = match operand_cuda_tile_element_type.as_ref() {
                                "f32" => {
                                    let flush_to_zero = self.flag_attr("flush_to_zero");
                                    vec![rounding_mode_attr, flush_to_zero]
                                }
                                "bf16" | "f16" | "f64" => {
                                    vec![rounding_mode_attr]
                                }
                                _ => unreachable!("Impossible"),
                            };
                            OperationBuilder::new("cuda_tile.divf", self.function_location())
                                .add_attributes(&attrs)
                                .add_operands(&[lhs_value, rhs_value])
                                .add_results(&[operand_cuda_tile_ty])
                        }
                        _ => {
                            return self.jit_error_result(
                                span,
                                &format!("Unimplemented binary op {tile_rust_arithmetic_op:#?}"),
                            );
                        }
                    }
                }
            }
            _ => {
                return self.jit_error_result(
                    span,
                    &format!(
                        "Binary operation is not implemented for {}",
                        operand_rust_ty.to_token_stream()
                    ),
                );
            }
        };

        let return_type = match return_type {
            Some(rt) => rt,
            None => {
                // Try to infer from lhs/rhs.
                if is_cmp {
                    let bool_ty = syn::parse2::<syn::Type>("bool".parse().unwrap()).unwrap();
                    self.compile_type(&bool_ty, generic_vars, &HashMap::new())?
                        .unwrap()
                } else {
                    operand_type
                }
            }
        };

        let op_bounds = if let (Some(a), Some(b)) = (lhs.bounds, rhs.bounds) {
            if !(lhs.kind == Kind::PrimitiveType && rhs.kind == Kind::PrimitiveType) {
                return self.jit_error_result(
                    span,
                    &format!(
                        "Expected PrimitiveType for binary op bounds, got lhs={:#?}, rhs={:#?}",
                        lhs.kind, rhs.kind
                    ),
                );
            }
            bounds_from_bop(tile_rust_arithmetic_op, &a, &b)
        } else {
            None
        };
        if let Some(bounds) = op_bounds {
            if bounds.is_exact() {
                // The lower/upper bounds are equivalent.
                return self.compile_constant_from_exact_bounds(builder, bounds, return_type);
            }
        }

        let op = op_builder.build();
        if op.is_err() {
            return self.jit_error_result(
                span,
                &format!("Failed to compile {tile_rust_arithmetic_op:#?}"),
            );
        }
        let op_ref = builder.append_operation(op.unwrap());
        if !op_ref.verify() {
            return self.jit_error_result(
                span,
                &format!("Failed to compile {tile_rust_arithmetic_op:#?}"),
            );
        }
        let value: Value = op_ref.result(0).unwrap().into();
        let mut tr_value = TileRustValue::new_value_kind_like(value, return_type.clone());
        tr_value.bounds = op_bounds;
        Ok(tr_value)
    }
}
