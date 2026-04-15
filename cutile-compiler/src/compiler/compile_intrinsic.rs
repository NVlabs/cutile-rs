/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Intrinsic compilation for compiler2.
//!
//! Handles macro execution, compiler_op calls, and check_partition_access
//! using tile-ir operations.

use syn::spanned::Spanned;

use super::_function::CUDATileFunctionCompiler;
use super::_type as types;
use super::_value::{CompilerContext, Mutability, TileRustValue};
use super::shared_types::Kind;
use super::shared_utils::{get_binary_op_from_op_str, get_const_hex, TileBinaryOp};
use super::tile_rust_type::TileRustType;
use super::utils::{int_attr, rounding_mode_attr, signedness_attr};
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::generics::{GenericVars, TypeInstance};
use crate::syn_utils::*;
use crate::types::*;

use cutile_ir::builder::{append_op, build_block, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::{
    Attribute, BlockId, Module, Region, ScalarType, TileElementType, TileType, Type, Value,
};

use quote::ToTokens;
use std::collections::HashMap;
use syn::{Expr, ExprCall, ExprPath, GenericArgument, ItemFn, Lit, PathArguments};

/// Helper: determine signedness string from a Rust element type name.
fn get_signedness_str(element_type_str: &str) -> &'static str {
    match element_type_str {
        "bool" | "u8" | "u16" | "u32" | "u64" => "unsigned",
        _ => "signed",
    }
}

/// Convert a `TileRustType` that the old compiler built into a
/// `cutile_ir::ir::Type`.  Tries `types::convert_type` first (handles
/// primitives), then falls back to building a tile type from the
/// element-type name and shape when the type instance is structured.
fn tile_ir_type_from_trt(
    trt: &TileRustType,
    primitives: &HashMap<(String, String), syn::ItemImpl>,
) -> Option<cutile_ir::ir::Type> {
    // Fast path: primitives and simple cases handled by convert_type.
    if let Some(ty) = types::convert_type(trt) {
        return Some(ty);
    }
    // Structured types: extract element name + shape from the TypeInstance.
    if let TypeInstance::StructuredType(inst) = &trt.type_instance {
        // Use the same element-type resolution path the old compiler uses.
        let elem_name = trt.get_cuda_tile_element_type(primitives).ok()??;
        let shape: Vec<i64> = inst.shape.iter().map(|&d| d as i64).collect();
        return types::make_tile_type(&elem_name, &shape);
    }
    None
}

impl<'m> CUDATileFunctionCompiler<'m> {
    /// Compiles a `compiler_op` (intrinsic) function call.
    /// The compiler implements Rust-related functionality, such as polymorphism,
    /// for these functions.
    ///
    /// This handles the large dispatch table for calls to functions annotated
    /// with `#[cuda_tile::compiler_op(...)]`. These are internal operations
    /// like mma, tile ops, shape ops, reduce, arithmetic, cast, convert,
    /// return_type_meta_field, set_type_meta_field, check, and assume.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_compiler_op_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        path_expr: &ExprPath,
        fn_item: &ItemFn,
        compiler_op_attrs: &SingleMetaList,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        let call_expr_func_str = call_expr.func.to_token_stream().to_string();
        let ident = get_ident_from_path_expr(path_expr);
        let Some(compiler_op_name) = compiler_op_attrs.parse_string("name") else {
            return self.jit_error_result(
                &call_expr.span(),
                "compiler operation is missing a required `name` attribute",
            );
        };
        match compiler_op_name.as_str() {
            "mma" => {
                let mut operands =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let lhs = operands.remove(0);
                let rhs = operands.remove(0);
                let out = operands.remove(0);
                let out_type = out.ty.clone();
                let Some(out_rust_element_type) =
                    out_type.get_instantiated_rust_element_type(&self.modules.primitives())
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to determine element type for `{}` output",
                            compiler_op_name
                        ),
                    );
                };
                let Some(_out_tile_ir_ty) = &out_type.tile_ir_ty else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to infer return type for `{}`; add a type annotation",
                            compiler_op_name
                        ),
                    );
                };
                let Some(out_cuda_tile_element_type) =
                    out_type.get_cuda_tile_element_type(&self.modules.primitives())?
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to determine compiled element type for `{}`",
                            compiler_op_name
                        ),
                    );
                };
                let out_is_float = super::_type::scalar_from_name(&out_cuda_tile_element_type)
                    .is_some_and(|s| s.is_float());
                let (opcode, attrs) = if out_is_float {
                    (Opcode::MmaF, vec![])
                } else if !out_is_float {
                    let Some(lhs_elem_ty) = lhs
                        .ty
                        .get_instantiated_rust_element_type(&self.modules.primitives())
                    else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "unable to determine left-hand operand element type for `mma`",
                        );
                    };
                    let Some(rhs_elem_ty) = lhs
                        .ty
                        .get_instantiated_rust_element_type(&self.modules.primitives())
                    else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "unable to determine right-hand operand element type for `mma`",
                        );
                    };
                    (
                        Opcode::MmaI,
                        vec![
                            signedness_attr("signedness_lhs", get_signedness_str(&lhs_elem_ty)),
                            signedness_attr("signedness_rhs", get_signedness_str(&rhs_elem_ty)),
                        ],
                    )
                } else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`mma` does not support element type `{}`; expected a float or integer type",
                            out_rust_element_type
                        ),
                    );
                };
                // Get result type from the output value's type in the module.
                let result_type = module
                    .value_type(out.value.expect("Expected output to be a value."))
                    .clone();
                let (op_id, results) = OpBuilder::new(opcode, self.ir_location(&call_expr.span()))
                    .operands([
                        lhs.value.expect("Expected LHS to be a value."),
                        rhs.value.expect("Expected RHS to be a value."),
                        out.value.expect("Expected output to be a value."),
                    ])
                    .attrs(attrs)
                    .result(result_type)
                    .build(module);
                append_op(module, block_id, op_id);
                let value: Value = results[0];
                let tr_value = TileRustValue::new_value_kind_like(value, out_type);
                Ok(Some(tr_value))
            }
            "tile" => {
                let compiler_op_function = ident.to_string();
                if !compiler_op_function.ends_with("_tile") {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "tile operation function name must end with `_tile`, got `{}`",
                            compiler_op_function
                        ),
                    );
                }
                let op = compiler_op_function.split("_").collect::<Vec<&str>>()[0];
                let tile_binary_op = get_binary_op_from_op_str(op)?;
                let mut operands =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let lhs = operands.remove(0);
                let rhs = operands.remove(0);
                let res = self.compile_binary_op_from_values(
                    module,
                    block_id,
                    lhs,
                    rhs,
                    &tile_binary_op,
                    generic_vars,
                    ctx,
                    return_type,
                    &call_expr.span(),
                )?;
                Ok(Some(res))
            }
            "shape" => {
                let compiler_op_function = ident.to_string();
                match compiler_op_function.as_str() {
                    "get_shape_dim" => {
                        let idx = self
                            .compile_expression(
                                module,
                                block_id,
                                &call_expr.args[1],
                                generic_vars,
                                ctx,
                                None,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.args[1].span(),
                                    "failed to compile dimension index expression",
                                )
                            })?;
                        let Some(idx_bounds) = idx.bounds else {
                            return self.jit_error_result(
                                &call_expr.args[1].span(),
                                "dimension index must be a compile-time constant",
                            );
                        };
                        if !idx_bounds.is_exact() {
                            return self.jit_error_result(
                                &call_expr.args[1].span(),
                                "dimension index must have exact bounds (a single known value)",
                            );
                        }
                        let dim_index = idx_bounds.start;
                        let shape = self
                            .compile_expression(
                                module,
                                block_id,
                                &call_expr.args[0],
                                generic_vars,
                                ctx,
                                None,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.args[0].span(),
                                    "failed to compile shape expression",
                                )
                            })?;
                        let Some(mut shape_fields) = shape.fields else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "shape value is missing its fields",
                            );
                        };
                        let Some(shape_dims) = shape_fields.remove("dims") else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "shape value is missing a `dims` field",
                            );
                        };
                        let Some(mut dims_values) = shape_dims.values else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "shape `dims` must be a compound (tuple) value",
                            );
                        };
                        let dim = dims_values.remove(dim_index as usize);
                        Ok(Some(dim))
                    }
                    "permute_array" => {
                        let src_slice = self
                            .compile_expression(
                                module,
                                block_id,
                                &call_expr.args[0],
                                generic_vars,
                                ctx,
                                None,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.args[0].span(),
                                    "failed to compile source array for permutation",
                                )
                            })?;
                        let mut dst_slice = src_slice;
                        let Some(val_arr) = &mut dst_slice.values else {
                            return self.jit_error_result(
                                &call_expr.args[0].span(),
                                "expected a compound (tuple/array) value for permutation source",
                            );
                        };
                        *val_arr = {
                            let dim_map = self
                                .compile_expression(
                                    module,
                                    block_id,
                                    &call_expr.args[1],
                                    generic_vars,
                                    ctx,
                                    None,
                                )?
                                .ok_or_else(|| {
                                    self.jit_error(
                                        &call_expr.args[1].span(),
                                        "failed to compile dimension map for permutation",
                                    )
                                })?;
                            let TypeInstance::UserType(type_inst) = dim_map.ty.type_instance else {
                                return self.jit_error_result(
                                    &call_expr.args[1].span(),
                                    "expected a structured type for the dimension map argument",
                                );
                            };
                            let Some(dim_map) = type_inst.try_extract_cga(generic_vars) else {
                                return self.jit_error_result(
                                    &call_expr.args[1].span(),
                                    "dimension map must be a const generic array type",
                                );
                            };
                            if dim_map.len() != val_arr.len() {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "dimension map has {} entries but the array has {} elements",
                                        dim_map.len(),
                                        val_arr.len()
                                    ),
                                );
                            }
                            let mut result = vec![];
                            for i in 0..dim_map.len() {
                                // Permute by moving item from dim_map[i] -> i.
                                result.push(val_arr[dim_map[i] as usize].clone());
                            }
                            result
                        };
                        Ok(Some(dst_slice))
                    }
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("unrecognized shape operation `{}`", compiler_op_function),
                    ),
                }
            }
            "num_tiles" => {
                // Signature: fn num_tiles(view: &Partition<E, S>, axis: i32) -> i32
                //
                // Lowers to `cuda_tile.get_index_space_shape` producing N scalar
                // i32 results (one per partition-view axis), then returns
                // result[axis]. `axis` must be a compile-time constant.
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "`num_tiles` expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let view = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        None,
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "failed to compile partition-view argument to `num_tiles`",
                        )
                    })?;
                let i32_tr_type = self
                    .compile_type(&syn::parse_quote!(i32), generic_vars, &HashMap::new())?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.span(),
                            "failed to synthesize `i32` type for `num_tiles`",
                        )
                    })?;
                let axis_val = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[1],
                        generic_vars,
                        ctx,
                        Some(i32_tr_type.clone()),
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[1].span(),
                            "failed to compile axis argument to `num_tiles`",
                        )
                    })?;

                let Some(axis_bounds) = axis_val.bounds else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "`num_tiles` axis must be a compile-time constant",
                    );
                };
                if !axis_bounds.is_exact() {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "`num_tiles` axis must have a single known value",
                    );
                }
                if axis_bounds.start < 0 {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!("`num_tiles` axis {} out of range", axis_bounds.start),
                    );
                }
                let axis = axis_bounds.start as usize;

                let view_value = view.value.ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        "expected a direct value for the partition view",
                    )
                })?;
                let view_ty = module.value_type(view_value).clone();
                let Type::PartitionView(pv) = &view_ty else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("`num_tiles` expects a partition view, got `{:?}`", view_ty),
                    );
                };
                let rank = pv.tile_shape.len();
                if axis >= rank {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        &format!("`num_tiles` axis {axis} out of range for rank-{rank} partition"),
                    );
                }

                let i32_scalar_ty = Type::Tile(TileType {
                    shape: vec![],
                    element_type: TileElementType::Scalar(ScalarType::I32),
                });
                let mut op_builder = OpBuilder::new(
                    Opcode::GetIndexSpaceShape,
                    self.ir_location(&call_expr.span()),
                )
                .operand(view_value);
                for _ in 0..rank {
                    op_builder = op_builder.result(i32_scalar_ty.clone());
                }
                let (op_id, results) = op_builder.build(module);
                append_op(module, block_id, op_id);

                let selected = results[axis];
                let return_type = match return_type {
                    Some(return_type) => return_type,
                    None => i32_tr_type,
                };
                let mut tr_value = TileRustValue::new_value_kind_like(selected, return_type);
                let parent_axis = pv.dim_map.get(axis).copied().ok_or_else(|| {
                    self.jit_error(
                        &call_expr.args[0].span(),
                        &format!(
                            "`num_tiles` axis {axis} is missing from partition dim_map {:?}",
                            pv.dim_map
                        ),
                    )
                })?;
                if parent_axis < 0 {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`num_tiles` axis {axis} maps to invalid parent axis {parent_axis}"
                        ),
                    );
                }
                let parent_axis = parent_axis as usize;
                let Some(&parent_dim) = pv.tensor_view.shape.get(parent_axis) else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`num_tiles` axis {axis} maps to parent axis {parent_axis}, but parent tensor rank is {}",
                            pv.tensor_view.shape.len()
                        ),
                    );
                };
                let tile_dim = pv.tile_shape[axis] as i64;
                if tile_dim <= 0 {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("`num_tiles` axis {axis} has invalid tile dimension {tile_dim}"),
                    );
                }
                if parent_dim >= 0 {
                    let num_tiles = (parent_dim + tile_dim - 1) / tile_dim;
                    tr_value.bounds = Some(Bounds::exact(num_tiles));
                }
                Ok(Some(tr_value))
            }
            "reduce" => {
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("`reduce` expects 2 arguments, got {}", call_expr.args.len()),
                    );
                }
                let operand = self
                    .compile_expression(
                        module,
                        block_id,
                        &call_expr.args[0],
                        generic_vars,
                        ctx,
                        None,
                    )?
                    .ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "failed to compile reduce operand",
                        )
                    })?;
                let Expr::Lit(lit_expr) = &call_expr.args[1] else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "the dimension argument must be an integer literal",
                    );
                };
                let Lit::Int(int_lit) = &lit_expr.lit else {
                    return self.jit_error_result(
                        &call_expr.args[1].span(),
                        "Dim arg must be an integer.",
                    );
                };
                let dim = int_lit.base10_parse::<i32>().map_err(|e| {
                    self.jit_error(
                        &call_expr.args[1].span(),
                        &format!("Failed to parse lit int: {e}"),
                    )
                })?;
                let TypeInstance::StructuredType(structured_type) = operand.ty.type_instance else {
                    return self
                        .jit_error_result(&call_expr.args[0].span(), "expected a struct value");
                };
                let Some(primitive_type) = structured_type.primitive_type else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "Expected primitive type to be defined.",
                    );
                };
                let Some(element_type) = primitive_type.get_rust_element_instance_ty() else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        "Failed to obtain rust element instance type.",
                    );
                };

                let reduce_op_string = ident.to_string();
                let (identity, closure_block_op): (String, syn::Block) =
                    match reduce_op_string.as_str() {
                        "reduce_min" => (
                            get_const_hex(&element_type, "max")?,
                            syn::parse_quote! { { min(curr, prev) } },
                        ),
                        "reduce_max" => (
                            get_const_hex(&element_type, "min")?,
                            syn::parse_quote! { { max(curr, prev) } },
                        ),
                        "reduce_sum" => (
                            get_const_hex(&element_type, "zero")?,
                            syn::parse_quote! { { curr + prev } },
                        ),
                        "reduce_prod" => (
                            get_const_hex(&element_type, "one")?,
                            syn::parse_quote! { { curr * prev } },
                        ),
                        _ => {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("Unsupported reduce operation: {reduce_op_string}"),
                            );
                        }
                    };

                let mut shape = structured_type.shape.clone();
                shape.remove(dim as usize);
                // Build tile-ir types directly (no format→parse chain for the IR types).
                let ir_result_type = types::make_tile_type(
                    &element_type,
                    &shape.iter().map(|&s| s as i64).collect::<Vec<_>>(),
                )
                .ok_or_else(|| {
                    self.jit_error(
                        &call_expr.span(),
                        "Failed to build tile-ir type for reduce result.",
                    )
                })?;
                let ir_operand_type =
                    types::make_scalar_tile_type(&element_type).ok_or_else(|| {
                        self.jit_error(
                            &call_expr.span(),
                            "Failed to build tile-ir type for reduce operand.",
                        )
                    })?;
                // TileRustTypes needed for closure body variables and result wrapping.
                let tile_rust_result_type = match TileRustType::from_tile(&element_type, &shape) {
                    Some(t) => t,
                    None => {
                        let ty = syn::parse_str::<syn::Type>(&format!(
                            "Tile<{element_type}, {{ {shape:#?} }}>"
                        ))
                        .unwrap();
                        self.compile_type(&ty, generic_vars, &HashMap::new())?
                            .unwrap()
                    }
                };
                let tile_rust_iter_operand_type =
                    match TileRustType::from_scalar_tile(&element_type) {
                        Some(t) => t,
                        None => {
                            let ty = syn::parse_str::<syn::Type>(&format!(
                                "Tile<{element_type}, {{ [] }}>"
                            ))
                            .unwrap();
                            self.compile_type(&ty, generic_vars, &HashMap::new())?
                                .unwrap()
                        }
                    };
                // Build the reduce body region.
                let region_id = {
                    let local_var_types = &[
                        ir_operand_type.clone(), // operand_i_current_iter
                        ir_operand_type.clone(), // operand_i_prev_iter
                    ];
                    let (local_block_id, local_block_args) = build_block(module, local_var_types);
                    let local_var_names = ["curr", "prev"];
                    let mut local_vars = CompilerContext::empty();
                    for i in 0..local_block_args.len() {
                        let value: Value = local_block_args[i];
                        let name = local_var_names[i];
                        let ty = tile_rust_iter_operand_type.clone();
                        let tile_rust_val = TileRustValue::new_value_kind_like(value, ty);
                        local_vars.vars.insert(name.to_string(), tile_rust_val);
                    }
                    // This is a binary op on the Tile type.
                    let op = self
                        .compile_block(
                            module,
                            local_block_id,
                            &closure_block_op,
                            generic_vars,
                            &mut local_vars,
                            return_type,
                        )?
                        .ok_or_else(|| {
                            self.jit_error(&call_expr.span(), "failed to compile reduce operation")
                        })?;
                    let Some(op_value) = op.value else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "Failed to obtain value from reduce compilation.",
                        );
                    };
                    let (yield_op_id, _) =
                        OpBuilder::new(Opcode::Yield, self.ir_location(&call_expr.span()))
                            .operand(op_value)
                            .build(module);
                    append_op(module, local_block_id, yield_op_id);
                    module.alloc_region(Region {
                        blocks: vec![local_block_id],
                    })
                };

                // Build the reduce op itself.
                // Build a properly typed identities attribute from the hex identity.
                let identity_attr = {
                    let scalar_ty = super::_type::scalar_from_name(&element_type)
                        .unwrap_or(cutile_ir::ir::ScalarType::I32);
                    let ir_ty = cutile_ir::ir::Type::Scalar(scalar_ty);
                    let hex_str = identity.trim_start_matches("0x").trim_start_matches("0X");
                    let bits = u64::from_str_radix(hex_str, 16).unwrap_or(0);
                    if scalar_ty.is_float() {
                        let float_val = match element_type.as_str() {
                            "f32" => f32::from_bits(bits as u32) as f64,
                            "f64" => f64::from_bits(bits),
                            "f16" => half::f16::from_bits(bits as u16).to_f64(),
                            "bf16" => half::bf16::from_bits(bits as u16).to_f64(),
                            _ => bits as f64,
                        };
                        Attribute::Float(float_val, ir_ty)
                    } else {
                        Attribute::Integer(bits as i64, ir_ty)
                    }
                };
                let (op_id, results) =
                    OpBuilder::new(Opcode::Reduce, self.ir_location(&call_expr.span()))
                        .attrs([
                            int_attr("dim", dim as i64),
                            (
                                "identities".to_string(),
                                Attribute::Array(vec![identity_attr]),
                            ),
                        ])
                        .operand(operand.value.ok_or_else(|| {
                            self.jit_error(
                                &call_expr.args[0].span(),
                                "Expect value for reduce op operand.",
                            )
                        })?)
                        .result(ir_result_type)
                        .region(region_id)
                        .build(module);
                append_op(module, block_id, op_id);
                let value: Value = results[0];
                let tr_value = TileRustValue::new_value_kind_like(value, tile_rust_result_type);
                Ok(Some(tr_value))
            }
            "arithmetic" => {
                let num_operands = call_expr.args.len();
                match num_operands {
                    2 => {
                        let binary_op = get_binary_op_from_op_str(&ident.to_string())?;
                        // Binary arithmetic operation.
                        let mut args = self.compile_call_args(
                            module,
                            block_id,
                            &call_expr.args,
                            generic_vars,
                            ctx,
                        )?;
                        let lhs = args.remove(0);
                        let rhs = args.remove(0);
                        Ok(Some(self.compile_binary_op_from_values(
                            module,
                            block_id,
                            lhs,
                            rhs,
                            &binary_op,
                            generic_vars,
                            ctx,
                            return_type,
                            &call_expr.span(),
                        )?))
                    }
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("arithmetic ops with {num_operands} operands not supported"),
                    ),
                }
            }
            "cast" => {
                let compiler_op_function = ident.to_string();
                // For casts, we require the rust types compiles to the same value.
                // We therefore only need to update the rust type.
                let args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                if args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("cast expects 1 argument, got {}", args.len()),
                    );
                }
                let mut new_value = args[0].clone();
                let old_type = new_value.ty.rust_ty;
                match compiler_op_function.as_str() {
                    "scalar_to_tile" => {
                        let element_type = get_rust_element_type_primitive(&old_type);
                        if let Some(t) = TileRustType::from_scalar_tile(&element_type) {
                            new_value.ty = t;
                        } else {
                            new_value.ty.rust_ty = syn::parse_str::<syn::Type>(&format!(
                                "Tile<{element_type}, {{[]}}>"
                            ))
                            .unwrap();
                        }
                    }
                    "tile_to_scalar" => {
                        let Some(element_type) =
                            get_element_type_structured(&old_type, &self.modules.primitives())
                        else {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!(
                                    "Failed to cast from {} to {}",
                                    old_type.to_token_stream(),
                                    get_sig_output_type(&fn_item.sig).to_token_stream()
                                ),
                            );
                        };
                        new_value.ty.rust_ty =
                            syn::parse2::<syn::Type>(element_type.parse().unwrap()).unwrap();
                    }
                    "pointer_to_tile" => {
                        let element_type = get_rust_element_type_primitive(&old_type);
                        if let Some(t) = TileRustType::from_scalar_ptr(&element_type) {
                            new_value.ty = t;
                        } else {
                            new_value.ty.rust_ty = syn::parse_str::<syn::Type>(&format!(
                                "PointerTile<* mut {element_type}, {{[]}}>"
                            ))
                            .unwrap();
                        }
                    }
                    "tile_to_pointer" => {
                        let Some(element_type) =
                            get_element_type_structured(&old_type, &self.modules.primitives())
                        else {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!(
                                    "Failed to cast from {} to {}",
                                    old_type.to_token_stream(),
                                    get_sig_output_type(&fn_item.sig).to_token_stream()
                                ),
                            );
                        };
                        new_value.ty.rust_ty = syn::parse2::<syn::Type>(
                            format!("* mut {element_type}").parse().unwrap(),
                        )
                        .unwrap();
                    }
                    _ => {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!("Unsupported cast compiler_op: {}", compiler_op_function),
                        );
                    }
                }
                Ok(Some(new_value))
            }
            "convert" => {
                let compiler_op_function = ident.to_string();
                match compiler_op_function.as_str() {
                    "convert_scalar" | "convert_tile" => {
                        let mut args = self.compile_call_args(
                            module,
                            block_id,
                            &call_expr.args,
                            generic_vars,
                            ctx,
                        )?;
                        if args.len() != 1 {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("convert expects 1 argument, got {}", args.len()),
                            );
                        }
                        let mut arg = args.pop().unwrap();
                        let new_type_compiled = if let Some(return_type) = return_type {
                            return_type
                        } else {
                            let PathArguments::AngleBracketed(generic_args) =
                                &path_expr.path.segments.last().unwrap().arguments
                            else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!(
                                        "Failed to get type parameters for {}",
                                        path_expr.to_token_stream()
                                    ),
                                );
                            };
                            if generic_args.args.len() != 1 {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!(
                                        "Expected 1 generic argument for convert, got {}",
                                        generic_args.args.len()
                                    ),
                                );
                            }
                            let GenericArgument::Type(new_type) = &generic_args.args[0] else {
                                return self.jit_error_result(
                                    &path_expr.span(),
                                    &format!(
                                        "Failed to get type parameters for {}",
                                        path_expr.to_token_stream()
                                    ),
                                );
                            };
                            let Some(new_type_compiled) =
                                self.compile_type(new_type, generic_vars, &HashMap::new())?
                            else {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "{compiler_op_function} failed to compile new type: {}",
                                        new_type.to_token_stream()
                                    ),
                                );
                            };
                            new_type_compiled
                        };
                        let old_element_type_str = arg
                            .ty
                            .type_instance
                            .get_rust_element_instance_ty()
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.span(),
                                    "Type resolution failed for old element type.",
                                )
                            })?;
                        let new_element_type_str = new_type_compiled
                            .type_instance
                            .get_rust_element_instance_ty()
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.span(),
                                    "Type resolution failed for new element type.",
                                )
                            })?;
                        if old_element_type_str == new_element_type_str {
                            // Identity conversion — update to the resolved type.
                            arg.ty = new_type_compiled;
                            return Ok(Some(arg));
                        }
                        let output_type =
                            tile_ir_type_from_trt(&new_type_compiled, &self.modules.primitives())
                                .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.span(),
                                    &format!(
                                        "Failed to obtain tile-ir type for convert {}",
                                        call_expr.to_token_stream()
                                    ),
                                )
                            })?;
                        // These aren't required for all ops.
                        let (op_id, results) = match (
                            old_element_type_str.as_str(),
                            new_element_type_str.as_str(),
                        ) {
                            // TODO (hme): There are some more like this that make sense, but no time to implement.
                            ("i64", "i32") => {
                                // cuda_tile.trunci %from %overflow
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Conversion {old_element_type_str:#?} -> {new_element_type_str:#?} not yet implemented"
                                    ),
                                );
                            }
                            ("i32", "i64") => {
                                // cuda_tile.exti %from %signedness
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Conversion {old_element_type_str:#?} -> {new_element_type_str:#?} not yet implemented"
                                    ),
                                );
                            }
                            // Integer → float: IToF with signedness from source type.
                            (from, to)
                                if super::_type::scalar_from_name(from)
                                    .map_or(false, |s| s.is_integer())
                                    && super::_type::scalar_from_name(to)
                                        .map_or(false, |s| s.is_float()) =>
                            {
                                let signedness = signedness_attr(
                                    "signedness",
                                    get_signedness_str(&old_element_type_str),
                                );
                                let rounding = rounding_mode_attr("nearest_even");
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to compile arg {}",
                                            call_expr
                                                .args
                                                .to_token_stream()
                                        ),
                                    );
                                };
                                OpBuilder::new(Opcode::IToF, self.ir_location(&call_expr.span()))
                                    .attrs([signedness, rounding])
                                    .operand(input_value)
                                    .result(output_type)
                                    .build(module)
                            }
                            // Float → integer: FToI with signedness from target type.
                            (from, to)
                                if super::_type::scalar_from_name(from)
                                    .map_or(false, |s| s.is_float())
                                    && super::_type::scalar_from_name(to)
                                        .map_or(false, |s| s.is_integer()) =>
                            {
                                let signedness = signedness_attr(
                                    "signedness",
                                    get_signedness_str(&new_element_type_str),
                                );
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to compile arg {}",
                                            call_expr
                                                .args
                                                .to_token_stream()
                                        ),
                                    );
                                };
                                let rounding =
                                    rounding_mode_attr("nearest_int_to_zero");
                                OpBuilder::new(Opcode::FToI, self.ir_location(&call_expr.span()))
                                    .attrs([signedness, rounding])
                                    .operand(input_value)
                                    .result(output_type)
                                    .build(module)
                            }
                            // Float → float: all float type pairs use FToF
                            // with NEAREST_EVEN rounding. This covers f16, bf16,
                            // f32, f64, tf32, f8e4m3fn, f8e5m2 — matching
                            // cutile-python's _get_type_conversion_encoder.
                            (from, to)
                                if super::_type::scalar_from_name(from)
                                    .map_or(false, |s| s.is_float())
                                    && super::_type::scalar_from_name(to)
                                        .map_or(false, |s| s.is_float()) =>
                            {
                                let rounding = rounding_mode_attr("nearest_even");
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "Failed to compile arg {}",
                                            call_expr
                                                .args
                                                .to_token_stream()
                                        ),
                                    );
                                };
                                OpBuilder::new(Opcode::FToF, self.ir_location(&call_expr.span()))
                                    .attr("rounding_mode", rounding.1)
                                    .operand(input_value)
                                    .result(output_type)
                                    .build(module)
                            }
                            _ => {
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Unsupported conversion {old_element_type_str:#?} -> {new_element_type_str:#?}"
                                    ),
                                )
                            }
                        };
                        append_op(module, block_id, op_id);
                        let value: Value = results[0];
                        Ok(Some(TileRustValue::new_value_kind_like(
                            value,
                            new_type_compiled,
                        )))
                    }
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unsupported convert compiler_op: {}", compiler_op_function),
                    ),
                }
            }
            "return_type_meta_field" => {
                let Some(type_meta_field) = compiler_op_attrs.parse_string("type_meta_field")
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unexpected return_type_meta_field {compiler_op_attrs:#?}"),
                    );
                };
                let args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                if args.len() != 1 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "return_type_meta_field expects 1 argument, got {}",
                            args.len()
                        ),
                    );
                }
                let value = args[0].clone();
                let Some(ref type_meta) = value.type_meta else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "Undefined type_meta for value {value:#?} \n compiler_op_attrs = {compiler_op_attrs:#?}"
                        ),
                    );
                };
                let Some(return_value) = type_meta.fields.get(&type_meta_field) else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("undefined type metadata field `{type_meta_field}` on this value"),
                    );
                };
                Ok(Some(return_value.clone()))
            }
            "set_type_meta_field" => {
                let Some(type_meta_field) = compiler_op_attrs.parse_string("type_meta_field")
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unexpected set_type_meta_field {compiler_op_attrs:#?}"),
                    );
                };
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "set_type_meta_field expects 2 arguments, got {}",
                            call_expr.args.len()
                        ),
                    );
                }
                let Expr::Path(var_arg) = &call_expr.args[0] else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "first argument to `set_type_meta_field` must be a simple variable path, got `{}`",
                            call_expr.to_token_stream()
                        ),
                    );
                };
                let var_name = get_ident_from_path_expr(var_arg)
                    .to_token_stream()
                    .to_string();
                if !ctx.vars.contains_key(var_name.as_str()) {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "first argument to `set_type_meta_field` must be a known variable, got `{}`",
                            call_expr.to_token_stream()
                        ),
                    );
                }
                let mut args =
                    self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
                let type_meta_value = args[1].clone();
                let type_value = &mut args[0];
                let Some(ref mut type_meta) = type_value.type_meta else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "Undefined type_meta for value {type_value:#?} \n compiler_op_attrs = {compiler_op_attrs:#?}"
                        ),
                    );
                };
                let old_val = type_meta
                    .fields
                    .insert(type_meta_field.clone(), type_meta_value);
                if old_val.is_none() {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "undefined type metadata field `{type_meta_field}` on this value; cannot set a field that does not exist"
                        ),
                    );
                }
                let result_value = type_value.clone();
                if result_value.mutability != Mutability::Mutable {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!(
                            "`set_type_meta_field` requires a mutable variable, but got {:?}",
                            result_value.mutability
                        ),
                    );
                }
                ctx.vars.insert(var_name.clone(), result_value);
                Ok(None)
            }
            "check" => {
                if self.entry_attrs.get_entry_arg_bool("unchecked_accesses") {
                    // Skip checks if unchecked_accesses is set.
                    return Ok(None);
                }
                let compiler_op_function = ident.to_string();
                match compiler_op_function.as_str() {
                    "check_partition_access" => Ok(self.compile_check_partition_access(
                        module,
                        block_id,
                        call_expr,
                        &call_expr_func_str,
                        generic_vars,
                        ctx,
                    )?),
                    _ => self.jit_error_result(
                        &call_expr.span(),
                        &format!("Unexpected compiler_op call {}", &call_expr_func_str),
                    ),
                }
            }
            "assume" => {
                let tr_value =
                    self.compile_assumption_call(call_expr, module, block_id, generic_vars, ctx)?;
                Ok(Some(tr_value))
            }
            _ => self.jit_error_result(
                &call_expr.span(),
                &format!("Unexpected compiler_op {compiler_op_attrs:#?}"),
            ),
        }
    }

    /// Compiles a check_partition_access compiler_op call.
    fn compile_check_partition_access(
        &self,
        module: &mut Module,
        block_id: BlockId,
        call_expr: &ExprCall,
        call_expr_func_str: &str,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
    ) -> Result<Option<TileRustValue>, JITError> {
        let mut args =
            self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
        let partition_value = args.remove(0);
        let index_value = args.remove(0);
        if partition_value.kind != Kind::StructuredType {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "expected a structured or primitive type for first argument of `{}`, got {:?}",
                    &call_expr.to_token_stream().to_string(),
                    partition_value.kind
                ),
            );
        }
        if index_value.kind != Kind::Compound {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "Unexpected kind for arg 1 in {}",
                    &call_expr.to_token_stream().to_string()
                ),
            );
        }
        if partition_value.ty.params.len() < 2 {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "Unable to obtain type parameters for arg 0 in {}",
                    call_expr_func_str
                ),
            );
        }

        // Get static tile values.
        let TypeParam::Tile(partition_tile) = &partition_value.ty.params[0] else {
            return self
                .jit_error_result(&call_expr.span(), "the type parameter must be a Tile type");
        };
        let Some(TypeInstance::StructuredType(tile_param_inst)) =
            partition_tile.type_instance.as_ref()
        else {
            return self
                .jit_error_result(&call_expr.span(), "the Tile parameter must be instantiated");
        };
        let static_tile = tile_param_inst.shape.clone(); // This is const.

        // Get static shape values.
        // Search for the TensorView param by variant (not by index), since
        // optional params like padding_value may shift the positions.
        let partition_tensor = partition_value
            .ty
            .params
            .iter()
            .find_map(|p| match p {
                TypeParam::TensorView(tv) => Some(tv),
                _ => None,
            })
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    &format!(
                        "partition type is missing a TensorView param; found: {:?}",
                        partition_value
                            .ty
                            .params
                            .iter()
                            .map(|p| p.name().unwrap_or_else(|| "?".to_string()))
                            .collect::<Vec<_>>()
                    ),
                )
            })?;
        let Some(TypeInstance::StructuredType(tensor_param_inst)) =
            partition_tensor.type_instance.as_ref()
        else {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "expected a structured type instance for tensor_view parameter, got {:?}",
                    &partition_tensor.type_instance
                ),
            );
        };
        let static_shape = tensor_param_inst.shape.clone(); // This *may* be const. Any field that is not const is -1.

        // Get optional dim_map by searching for the DimMap variant.
        let dim_map = match partition_value.ty.params.iter().find_map(|p| match p {
            TypeParam::DimMap(dm) => Some(dm),
            _ => None,
        }) {
            Some(dim_map) => {
                let Some(TypeInstance::StructuredType(dim_map_param_inst)) =
                    dim_map.type_instance.as_ref()
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "expected a structured type instance for dimension map, got `{}`",
                            dim_map.rust_ty.to_token_stream()
                        ),
                    );
                };
                dim_map_param_inst.shape.clone()
            }
            None => {
                let mut r = vec![];
                for i in 0..static_shape.len() {
                    r.push(i as i32);
                }
                r
            }
        };

        // Get dynamic shape values.
        let tensor_shape_value = partition_value
            .take_type_meta_field("tensor_view.shape()")
            .ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    "Failed to obtain type meta field tensor_view.shape().",
                )
            })?;
        let Some(tensor_shape_values) = tensor_shape_value.fields.as_ref() else {
            return self.jit_error_result(
                &call_expr.span(),
                "Expected fields for tensor shape expression.",
            );
        };
        let Some(shape_dims) = tensor_shape_values.get("dims") else {
            return self.jit_error_result(
                &call_expr.span(),
                "Expected dims field for shape expression.",
            );
        };
        let Some(dynamic_shape) = shape_dims.values.clone() else {
            return self.jit_error_result(&call_expr.span(), "expected a compound (tuple) value");
        };

        // Get index values.
        let Some(mut indexes) = index_value.values else {
            return self.jit_error_result(&call_expr.span(), "expected a compound (tuple) value");
        };
        let len = static_tile.len();
        if len != indexes.len() || len != static_shape.len() {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "Unexpected tile ({}), shape ({}), or index ({}) length mismatch.",
                    len,
                    static_shape.len(),
                    indexes.len()
                ),
            );
        }
        for i in 0..len {
            // Because the indices may be remapped via a permutation of the tile
            // dimensions, we need to remap the tensor's shape as well.
            let remapped_i = dim_map[i] as usize;
            let static_tile_dim = static_tile[i];
            let static_shape_dim = static_shape[remapped_i];
            let is_static_shape_dim = static_shape_dim != -1;
            let index_value = indexes.remove(0);
            if let (Some(bounds), true) = (index_value.bounds, is_static_shape_dim) {
                // We can do a static bounds check.
                let num_partitions =
                    (static_shape_dim as i64 + static_tile_dim as i64 - 1) / static_tile_dim as i64;
                if !(0 <= bounds.start && bounds.end < num_partitions) {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "Bounds check failed: 0 <= {} && {} < {}",
                            bounds.start, bounds.end, num_partitions
                        ),
                    );
                }
                return Ok(None);
            }
            // In the rest of the cases, we need to generate a dynamic bounds check.
            let tile_dim_value =
                self.compile_constant(module, block_id, generic_vars, static_tile_dim)?;
            let index_value = if let Some(bounds) = index_value.bounds {
                let index_upper_bound = bounds.end;
                self.compile_constant(module, block_id, generic_vars, index_upper_bound as i32)?
            } else {
                index_value
            };
            let shape_dim_value = if is_static_shape_dim {
                self.compile_constant(module, block_id, generic_vars, static_shape_dim)?
            } else {
                dynamic_shape[remapped_i].clone()
            };
            // Compute ceil_div(shape, tile) as (shape + tile - 1) / tile
            // using floor division. This avoids positive_inf rounding which
            // can be misoptimized when the dividend carries assume hints.
            let tile_minus_one =
                self.compile_constant(module, block_id, generic_vars, static_tile_dim - 1)?;
            let shape_plus_tile_minus_one = self.compile_binary_op_from_values(
                module,
                block_id,
                shape_dim_value.clone(),
                tile_minus_one,
                &TileBinaryOp::Add,
                generic_vars,
                ctx,
                None,
                &call_expr.span(),
            )?;
            let div_result_value = self.compile_binary_op_from_values(
                module,
                block_id,
                shape_plus_tile_minus_one,
                tile_dim_value,
                &TileBinaryOp::Div,
                generic_vars,
                ctx,
                None,
                &call_expr.span(),
            )?;
            let ineq_result_value = self.compile_binary_op_from_values(
                module,
                block_id,
                index_value,
                div_result_value,
                &TileBinaryOp::Lt,
                generic_vars,
                ctx,
                None,
                &call_expr.span(),
            )?;
            let result_value = ineq_result_value.value.ok_or_else(|| {
                self.jit_error(
                    &call_expr.span(),
                    "failed to compile a binary expression operand",
                )
            })?;
            let shape_desc = if is_static_shape_dim {
                format!("{}", static_shape_dim)
            } else {
                "?".to_string()
            };
            let message = format!(
                "partition access out of bounds: dim {i}, block index >= ceil({shape_desc}/{static_tile_dim})"
            );
            let (assert_op_id, _) =
                OpBuilder::new(Opcode::Assert, self.ir_location(&call_expr.span()))
                    .attr("message", cutile_ir::ir::Attribute::String(message))
                    .operand(result_value)
                    .build(module);
            append_op(module, block_id, assert_op_id);
        }
        Ok(None)
    }
}
