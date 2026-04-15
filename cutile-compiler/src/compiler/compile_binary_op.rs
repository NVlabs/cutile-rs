/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Binary operation compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_binary_op.rs` — translates binary
//! arithmetic, comparison, and bitwise operations into tile-ir operations.
//! Only type and IR-emission changes; the dispatch logic and bounds
//! propagation are identical.

use quote::ToTokens;
use syn::spanned::Spanned;

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, TileRustValue};
use super::shared_types::Kind;
use super::shared_utils::{get_tile_bop_from_rust_bop, TileBinaryOp};
use super::tile_rust_type::TileRustType;
use super::utils::{
    cmp_ordering_attr, cmp_pred_attr, flag_attr, rounding_mode_attr, signedness_attr, NamedAttr,
};
use crate::bounds::bounds_from_bop;
use crate::error::JITError;
use crate::generics::GenericVars;

use cutile_ir::builder::{append_op, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{Attribute, BlockId, Module, ScalarType, TileElementType, TileType, Type};

use std::collections::HashMap;
use syn::ExprBinary;

/// Port of `get_cmp_predicate_attr` from `compiler/utils.rs`.
///
/// Returns a comparison-predicate named attribute for comparison binary ops,
/// or `None` for non-comparison ops.
fn get_cmp_predicate_attr_ir(expr: &TileBinaryOp) -> Result<Option<NamedAttr>, JITError> {
    match expr {
        TileBinaryOp::Eq => Ok(Some(cmp_pred_attr("equal"))),
        TileBinaryOp::Ne => Ok(Some(cmp_pred_attr("not_equal"))),
        TileBinaryOp::Lt => Ok(Some(cmp_pred_attr("less_than"))),
        TileBinaryOp::Le => Ok(Some(cmp_pred_attr("less_than_or_equal"))),
        TileBinaryOp::Gt => Ok(Some(cmp_pred_attr("greater_than"))),
        TileBinaryOp::Ge => Ok(Some(cmp_pred_attr("greater_than_or_equal"))),
        _ => Ok(None),
    }
}

/// Construct a tile-ir bool (i1) result type that mirrors the shape of `lhs_type`.
///
/// If `lhs_type` is a `Tile`, the result is a tile with the same shape but `I1`
/// element type. If it's a scalar, the result is `Scalar(I1)`.
fn make_bool_result_type(lhs_type: &Type) -> Type {
    match lhs_type {
        Type::Tile(tile_ty) => Type::Tile(TileType {
            shape: tile_ty.shape.clone(),
            element_type: TileElementType::Scalar(ScalarType::I1),
        }),
        _ => Type::Scalar(ScalarType::I1),
    }
}

impl<'m> CUDATileFunctionCompiler<'m> {
    pub fn compile_binary_op(
        &self,
        module: &mut Module,
        block_id: BlockId,
        bin_expr: &ExprBinary,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let lhs = self.compile_expression(
            module,
            block_id,
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
            module,
            block_id,
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
            module,
            block_id,
            lhs,
            rhs,
            &get_tile_bop_from_rust_bop(&bin_expr.op)?,
            generic_vars,
            ctx,
            return_type,
            &bin_expr.span(),
        )?))
    }

    pub fn compile_binary_op_from_values(
        &self,
        module: &mut Module,
        block_id: BlockId,
        lhs: TileRustValue,
        rhs: TileRustValue,
        tile_rust_arithmetic_op: &TileBinaryOp,
        generic_vars: &GenericVars,
        _ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue, JITError> {
        if lhs.ty.rust_ty != rhs.ty.rust_ty {
            return self.jit_error_result(
                span,
                &format!(
                    "binary `{:?}` requires operands of the same type, but got `{}` and `{}`",
                    tile_rust_arithmetic_op,
                    lhs.ty.rust_ty.to_token_stream().to_string(),
                    rhs.ty.rust_ty.to_token_stream().to_string()
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
            operand_type.get_instantiated_rust_element_type(&self.modules.primitives())
        else {
            return self.jit_error_result(
                span,
                &format!(
                    "unable to determine element type for `{:?}` on `{}`",
                    tile_rust_arithmetic_op,
                    operand_type.rust_ty.to_token_stream().to_string()
                ),
            );
        };
        let Some(_operand_tile_ir_ty) = &operand_type.tile_ir_ty else {
            return self.jit_error_result(
                span,
                &format!(
                    "type `{}` cannot be used with binary `{:?}`",
                    operand_type.rust_ty.to_token_stream().to_string(),
                    tile_rust_arithmetic_op
                ),
            );
        };
        // For tile-ir, the result type for same-type operations comes from the
        // lhs value's type in the module (preserves tile shape).
        let operand_result_ty = module.value_type(lhs_value).clone();

        let Some(operand_cuda_tile_element_type) =
            operand_type.get_cuda_tile_element_type(&self.modules.primitives())?
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
        let sign_attr = signedness_attr("signedness", signedness_str);
        // Build the operation (allocates in module) but do NOT append to the
        // block yet. The old compiler defers `.build()` + `append_operation`
        // until after the exact-bounds early-return check, so we replicate that:
        // build now, check bounds, append only if we actually need the op.
        let (op_id, results) = match operand_cuda_tile_element_type.as_ref() {
            "i1" | "i4" | "i8" | "i32" | "i64" => {
                // TODO (hme): Add i4, i8, i16 support, as needed.
                if let Some(comparison_predicate) =
                    get_cmp_predicate_attr_ir(tile_rust_arithmetic_op)?
                {
                    is_cmp = true;
                    let bool_result_ty = make_bool_result_type(&operand_result_ty);
                    OpBuilder::new(Opcode::CmpI, self.ir_location(span))
                        .attr(comparison_predicate.0, comparison_predicate.1)
                        .attr(sign_attr.0, sign_attr.1)
                        .operand(lhs_value)
                        .operand(rhs_value)
                        .result(bool_result_ty)
                        .build(module)
                } else {
                    // If both operands have bounds, we can generate bounds on the output.
                    match tile_rust_arithmetic_op {
                        TileBinaryOp::Min => OpBuilder::new(Opcode::MinI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Max => OpBuilder::new(Opcode::MaxI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Add => OpBuilder::new(Opcode::AddI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Sub => OpBuilder::new(Opcode::SubI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Mul => OpBuilder::new(Opcode::MulI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr("overflow", Attribute::i32(0))
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Rem => OpBuilder::new(Opcode::RemI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .attr(sign_attr.0, sign_attr.1)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Div => {
                            // DivI uses "rounding" (not "rounding_mode") in bytecode
                            OpBuilder::new(Opcode::DivI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .attr(sign_attr.0, sign_attr.1)
                                .attr("rounding", Attribute::i32(2)) // negative_inf
                                .build(module)
                        }
                        TileBinaryOp::CeilDiv => {
                            // DivI uses "rounding" (not "rounding_mode") in bytecode
                            OpBuilder::new(Opcode::DivI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .attr(sign_attr.0, sign_attr.1)
                                .attr("rounding", Attribute::i32(3)) // positive_inf
                                .build(module)
                        }
                        TileBinaryOp::BitAnd => {
                            OpBuilder::new(Opcode::AndI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .build(module)
                        }
                        TileBinaryOp::BitOr => OpBuilder::new(Opcode::OrI, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::BitXor => {
                            OpBuilder::new(Opcode::XOrI, self.ir_location(span))
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .build(module)
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
                    get_cmp_predicate_attr_ir(tile_rust_arithmetic_op)?
                {
                    let comparison_ordering = cmp_ordering_attr("ordered");
                    is_cmp = true;
                    let bool_result_ty = make_bool_result_type(&operand_result_ty);
                    OpBuilder::new(Opcode::CmpF, self.ir_location(span))
                        .attr(comparison_predicate.0, comparison_predicate.1)
                        .attr(comparison_ordering.0, comparison_ordering.1)
                        .operand(lhs_value)
                        .operand(rhs_value)
                        .result(bool_result_ty)
                        .build(module)
                } else {
                    let default_rm_attr = rounding_mode_attr("nearest_even");
                    match tile_rust_arithmetic_op {
                        TileBinaryOp::Min => OpBuilder::new(Opcode::MinF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Max => OpBuilder::new(Opcode::MaxF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Add => OpBuilder::new(Opcode::AddF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Sub => OpBuilder::new(Opcode::SubF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Mul => OpBuilder::new(Opcode::MulF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Rem => OpBuilder::new(Opcode::RemF, self.ir_location(span))
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::Div => OpBuilder::new(Opcode::DivF, self.ir_location(span))
                            .attr(default_rm_attr.0, default_rm_attr.1)
                            .operand(lhs_value)
                            .operand(rhs_value)
                            .result(operand_result_ty.clone())
                            .build(module),
                        TileBinaryOp::TrueDiv => {
                            let approx_rm_attr = rounding_mode_attr("approx");
                            let mut builder = OpBuilder::new(Opcode::DivF, self.ir_location(span))
                                .attr(approx_rm_attr.0, approx_rm_attr.1);
                            if operand_cuda_tile_element_type.as_str() == "f32" {
                                let ftz = flag_attr("flush_to_zero");
                                builder = builder.attr(ftz.0, ftz.1);
                            }
                            builder
                                .operand(lhs_value)
                                .operand(rhs_value)
                                .result(operand_result_ty.clone())
                                .build(module)
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
                        operand_rust_ty.to_token_stream().to_string()
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
                    self.compile_type(&bool_ty, &generic_vars, &HashMap::new())?
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
        if let Some(bounds) = &op_bounds {
            if bounds.is_exact() {
                // The lower/upper bounds are equivalent — emit a constant
                // instead. The op allocated above becomes dead (not appended
                // to any block).
                return Ok(self.compile_constant_from_exact_bounds(
                    module,
                    block_id,
                    bounds.clone(),
                    return_type,
                )?);
            }
        }

        // Only now append the binary op to the block (mirrors the old
        // compiler which only calls `builder.append_operation` after the
        // bounds check).
        append_op(module, block_id, op_id);
        let value = results[0];
        let mut tr_value = TileRustValue::new_value_kind_like(value, return_type.clone());
        tr_value.bounds = op_bounds;
        Ok(tr_value)
    }
}
