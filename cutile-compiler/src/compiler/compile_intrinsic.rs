/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Intrinsic compilation: handles macro execution, compiler_op calls, and check_partition_access
//! within the CUDA Tile compiler.

use syn::spanned::Spanned;

use crate::compiler::_function::CUDATileFunctionCompiler;
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;
use crate::compiler::utils::{
    get_binary_op_from_op_str, get_const_hex, get_signedness_attr, named_str_attr, reduce_op,
    TileBinaryOp,
};
use crate::error::JITError;
use crate::generics::{GenericVars, TypeInstance};
use crate::syn_utils::*;
use crate::types::*;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::{self, Block, BlockLike, Location, Region, RegionLike, Value};
use quote::ToTokens;
use std::collections::HashMap;
use syn::{Expr, ExprCall, ExprPath, GenericArgument, ItemFn, Lit, PathArguments};

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    /// Compiles a `compiler_op` (intrinsic) function call.
    /// The compiler implements Rust-related functionality, such as polymorphism, for these functions.
    ///
    /// This handles the large dispatch table for calls to functions annotated with
    /// `#[cuda_tile::compiler_op(...)]`. These are internal operations like mma, tile ops,
    /// shape ops, reduce, arithmetic, cast, convert, return_type_meta_field, set_type_meta_field,
    /// check, and assume.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_compiler_op_call(
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        path_expr: &ExprPath,
        fn_item: &ItemFn,
        compiler_op_attrs: &SingleMetaList,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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
                    self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
                let lhs = operands.remove(0);
                let rhs = operands.remove(0);
                let out = operands.remove(0);
                let out_type = out.ty.clone();
                let Some(out_rust_element_type) =
                    out_type.get_instantiated_rust_element_type(&self.modules.primitives)
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to determine element type for `{}` output",
                            compiler_op_name
                        ),
                    );
                };
                let Some(out_cuda_tile_ty) = out_type.cuda_tile_ty else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to infer return type for `{}`; add a type annotation",
                            compiler_op_name
                        ),
                    );
                };
                let Some(out_cuda_tile_element_type) =
                    out_type.get_cuda_tile_element_type(&self.modules.primitives)?
                else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!(
                            "unable to determine compiled element type for `{}`",
                            compiler_op_name
                        ),
                    );
                };
                let (op_name, attrs) = if out_cuda_tile_element_type.starts_with("f") {
                    ("cuda_tile.mmaf", vec![])
                } else if out_cuda_tile_element_type.starts_with("i") {
                    let Some(lhs_elem_ty) = lhs
                        .ty
                        .get_instantiated_rust_element_type(&self.modules.primitives)
                    else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "unable to determine left-hand operand element type for `mma`",
                        );
                    };
                    let Some(rhs_elem_ty) = lhs
                        .ty
                        .get_instantiated_rust_element_type(&self.modules.primitives)
                    else {
                        return self.jit_error_result(
                            &call_expr.span(),
                            "unable to determine right-hand operand element type for `mma`",
                        );
                    };
                    (
                        "cuda_tile.mmai",
                        vec![
                            get_signedness_attr(&self.context, "signedness_lhs", &lhs_elem_ty)?,
                            get_signedness_attr(&self.context, "signedness_rhs", &rhs_elem_ty)?,
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
                // TODO (hme): Make it easier to compile basic ops.
                let op = OperationBuilder::new(op_name, self.function_location())
                    .add_operands(&[
                        lhs.value.expect("Expected LHS to be a value."),
                        rhs.value.expect("Expected RHS to be a value."),
                        out.value.expect("Expected output to be a value."),
                    ])
                    .add_attributes(attrs.as_slice())
                    .add_results(&[out_cuda_tile_ty])
                    .build()
                    .expect("Failed to compile mma.");
                let op_ref = builder.append_operation(op);
                if !op_ref.verify() {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("`{}` failed MLIR verification", compiler_op_name),
                    );
                }
                let value: Value = op_ref.result(0).unwrap().into();
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
                    self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
                let lhs = operands.remove(0);
                let rhs = operands.remove(0);
                let res = self.compile_binary_op_from_values(
                    builder,
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
                                builder,
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
                                builder,
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
                                builder,
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
                                    builder,
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
            "reduce" => {
                if call_expr.args.len() != 2 {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("`reduce` expects 2 arguments, got {}", call_expr.args.len()),
                    );
                }
                let operand = self
                    .compile_expression(builder, &call_expr.args[0], generic_vars, ctx, None)?
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
                let rust_result_type = syn::parse2::<syn::Type>(
                    format!("Tile<{element_type}, {{ {shape:#?} }}>")
                        .parse()
                        .unwrap(),
                )
                .unwrap();
                let tile_rust_result_type = self
                    .compile_type(&rust_result_type, generic_vars, &HashMap::new())?
                    .unwrap();
                let Some(mlir_result_type) = tile_rust_result_type.cuda_tile_ty else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        "Failed to obtain CUDA tile type for reduce result.",
                    );
                };
                let rust_iter_operand_type = syn::parse2::<syn::Type>(
                    format!("Tile<{element_type}, {{ [] }}>").parse().unwrap(),
                )
                .unwrap();
                let tile_rust_iter_operand_type = self
                    .compile_type(&rust_iter_operand_type, generic_vars, &HashMap::new())?
                    .unwrap();
                let Some(mlir_operand_type) = tile_rust_iter_operand_type.cuda_tile_ty else {
                    return self.jit_error_result(
                        &call_expr.span(),
                        "Failed to obtain CUDA tile type for reduce operand.",
                    );
                };
                let location = self.function_location();
                let op = reduce_op(
                    &self.context,
                    location,
                    operand.value.ok_or_else(|| {
                        self.jit_error(
                            &call_expr.args[0].span(),
                            "Expect value for reduce op operand.",
                        )
                    })?,
                    dim,
                    &identity,
                    element_type,
                    mlir_result_type,
                    {
                        let mut local_vars = CompilerContext::empty();
                        let local_var_names = ["curr", "prev"];
                        let local_var_types = &[
                            mlir_operand_type, // operand_i_current_iter
                            mlir_operand_type, // operand_i_prev_iter
                        ];
                        let local_block = Block::new(
                            &local_var_types
                                .iter()
                                .map(|&ty| (ty, location))
                                .collect::<Vec<_>>(),
                        );
                        for (i, name) in (0..local_block.argument_count()).zip(local_var_names) {
                            let value: Value = local_block.argument(i).unwrap().into();
                            let ty = tile_rust_iter_operand_type.clone();
                            let tile_rust_val = TileRustValue::new_value_kind_like(value, ty);
                            local_vars.vars.insert(name.to_string(), tile_rust_val);
                        }
                        // This is a binary op on the Tile type.
                        let op = self
                            .compile_block(
                                &local_block,
                                &closure_block_op,
                                generic_vars,
                                &mut local_vars,
                                return_type,
                            )?
                            .ok_or_else(|| {
                                self.jit_error(
                                    &call_expr.span(),
                                    "failed to compile reduce operation",
                                )
                            })?;
                        let Some(op_value) = op.value else {
                            return self.jit_error_result(
                                &call_expr.span(),
                                "Failed to obtain value from reduce compilation.",
                            );
                        };
                        let _yield_val = local_block.append_operation(
                            OperationBuilder::new("cuda_tile.yield", location)
                                .add_operands(&[op_value])
                                .build()
                                .unwrap(),
                        );
                        let region = Region::new();
                        region.append_block(local_block);
                        region
                    },
                );
                let op_ref = builder.append_operation(op?);
                if !op_ref.verify() {
                    return self.jit_error_result(
                        &call_expr.span(),
                        &format!("Failed to compile {compiler_op_name}"),
                    );
                }
                let value: Value = op_ref.result(0).unwrap().into();
                let tr_value = TileRustValue::new_value_kind_like(value, tile_rust_result_type);
                Ok(Some(tr_value))
            }
            "arithmetic" => {
                let num_operands = call_expr.args.len();
                match num_operands {
                    2 => {
                        let binary_op = get_binary_op_from_op_str(&ident.to_string())?;
                        // Binary arithmetic operation.
                        let mut args =
                            self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
                        let lhs = args.remove(0);
                        let rhs = args.remove(0);
                        Ok(Some(self.compile_binary_op_from_values(
                            builder,
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
                let args = self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
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
                        new_value.ty.rust_ty = syn::parse2::<syn::Type>(
                            format!("Tile<{element_type}, {{[]}}>").parse().unwrap(),
                        )
                        .unwrap();
                    }
                    "tile_to_scalar" => {
                        let Some(element_type) =
                            get_element_type_structured(&old_type, &self.modules.primitives)
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
                        new_value.ty.rust_ty = syn::parse2::<syn::Type>(
                            format!("PointerTile<* mut {element_type}, {{[]}}>")
                                .parse()
                                .unwrap(),
                        )
                        .unwrap();
                    }
                    "tile_to_pointer" => {
                        let Some(element_type) =
                            get_element_type_structured(&old_type, &self.modules.primitives)
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
                        let mut args =
                            self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
                        if args.len() != 1 {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("convert expects 1 argument, got {}", args.len()),
                            );
                        }
                        let arg = args.pop().unwrap();
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
                            // Nothing to do.
                            return Ok(Some(arg));
                        }
                        // These aren't required for all ops.
                        let op_builder = match (old_element_type_str.as_str(), new_element_type_str.as_str()) {
                            // TODO (hme): There are some more like this that make sense, but no time to implement.
                            ("i64", "i32") => {
                                // cuda_tile.trunci %from %overflow
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Conversion {old_element_type_str:#?} -> {new_element_type_str:#?} not yet implemented"
                                    ),
                                );
                            },
                            ("i32", "i64") => {
                                // cuda_tile.exti %from %signedness
                                return self.jit_error_result(
                                    &call_expr.span(),
                                    &format!(
                                        "Conversion {old_element_type_str:#?} -> {new_element_type_str:#?} not yet implemented"
                                    ),
                                );
                            },
                            ("i32", "f16") | ("u32", "f16") | ("i64", "f16") | ("u64", "f16") |
                            ("i32", "f32") | ("u32", "f32") | ("i64", "f32") | ("u64", "f32") |
                            ("i32", "f64") | ("u32", "f64") | ("i64", "f64") | ("u64", "f64") => {
                                let signedness_attr = get_signedness_attr(&self.context, "signedness", &old_element_type_str)?;
                                // This is apparently all that is supported by this op.
                                let rounding_mode_attr =self.parse_named_attr("rounding_mode", "#cuda_tile.rounding<nearest_even>")?;
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(&call_expr.span(), &format!("Failed to compile arg {}", call_expr.args.to_token_stream()));
                                };
                                let Some(output_value) = new_type_compiled.cuda_tile_ty else {
                                    return self.jit_error_result(&call_expr.span(), &format!("Failed to obtain CUDA tile type for convert {}", call_expr.to_token_stream()));
                                };
                                OperationBuilder::new("cuda_tile.itof", self.function_location())
                                    .add_attributes(&[signedness_attr, rounding_mode_attr])
                                    .add_operands(&[input_value])
                                    .add_results(&[output_value])
                            },
                            ("f16", "i32") | ("f16", "u32") | ("f16", "i64") | ("f16", "u64") |
                            ("f32", "i32") | ("f32", "u32") | ("f32", "i64") | ("f32", "u64") |
                            ("f64", "i32") | ("f64", "u32") | ("f64", "i64") | ("f64", "u64") => {
                                let signedness_attr = get_signedness_attr(&self.context, "signedness", &new_element_type_str)?;
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(&call_expr.span(), &format!("Failed to compile arg {}", call_expr.args.to_token_stream()));
                                };
                                let Some(output_value) = new_type_compiled.cuda_tile_ty else {
                                    return self.jit_error_result(&call_expr.span(), &format!("Failed to obtain CUDA tile type for convert {}", call_expr.to_token_stream()));
                                };
                                let rounding_mode_attr =self.parse_named_attr("rounding_mode", "#cuda_tile.rounding<nearest_int_to_zero>")?;
                                OperationBuilder::new("cuda_tile.ftoi", self.function_location())
                                    .add_attributes(&[signedness_attr, rounding_mode_attr])
                                    .add_operands(&[input_value])
                                    .add_results(&[output_value])
                            },
                            ("f16", "f32") | ("f16", "f64") | ("f32", "f16") |
                            ("f32", "f64") | ("f64", "f16") | ("f64", "f32") |
                            ("f32", "tf32") | ("tf32", "f32")  => {
                                let rounding_mode_attr =self.parse_named_attr("rounding_mode", "#cuda_tile.rounding<nearest_even>")?;
                                let Some(input_value) = arg.value else {
                                    return self.jit_error_result(&call_expr.span(), &format!("Failed to compile arg {}", call_expr.args.to_token_stream()));
                                };
                                let Some(output_value) = new_type_compiled.cuda_tile_ty else {
                                    return self.jit_error_result(&call_expr.span(), &format!("Failed to obtain CUDA tile type for convert {}", call_expr.to_token_stream()));
                                };
                                OperationBuilder::new("cuda_tile.ftof", self.function_location())
                                    .add_attributes(&[rounding_mode_attr])
                                    .add_operands(&[input_value])
                                    .add_results(&[output_value])
                            }
                            _ => return self.jit_error_result(
                                &call_expr.span(),
                                &format!("Unsupported conversion {old_element_type_str:#?} -> {new_element_type_str:#?}"),
                            )
                        };
                        let op = op_builder.build();
                        if op.is_err() {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("Failed to compile {}", call_expr.to_token_stream()),
                            );
                        }
                        let op_ref = builder.append_operation(op?);
                        if !op_ref.verify() {
                            return self.jit_error_result(
                                &call_expr.span(),
                                &format!("Failed to verify {}", call_expr.to_token_stream()),
                            );
                        }
                        let value: Value = op_ref.result(0).unwrap().into();
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
                let args = self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
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
                        &format!("Undefined type_meta for value {value:#?} \n compiler_op_attrs = {compiler_op_attrs:#?}"),
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
                    self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
                let type_meta_value = args[1].clone();
                let type_value = &mut args[0];
                let Some(ref mut type_meta) = type_value.type_meta else {
                    return self.jit_error_result(
                        &call_expr.args[0].span(),
                        &format!("Undefined type_meta for value {type_value:#?} \n compiler_op_attrs = {compiler_op_attrs:#?}"),
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
                        builder,
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
                    self.compile_assumption_call(call_expr, builder, generic_vars, ctx)?;
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
        &'c self,
        builder: &'c ir::Block<'c>,
        call_expr: &ExprCall,
        call_expr_func_str: &str,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        let mut args = self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
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
        let TypeParam::TensorView(partition_tensor) = &partition_value.ty.params[1] else {
            return self.jit_error_result(
                &call_expr.span(),
                "Tensor type param should be a TensorView.",
            );
        };
        let Some(TypeInstance::StructuredType(tensor_param_inst)) =
            partition_tensor.type_instance.as_ref()
        else {
            return self.jit_error_result(
                &call_expr.span(),
                &format!(
                    "expected a structured type instance for tile parameter, got {:?}",
                    &partition_tensor.type_instance
                ),
            );
        };
        let static_shape = tensor_param_inst.shape.clone(); // This *may* be const. Any field that is not const is -1.

        // Get optional dim_map.
        // If there's a dim map, the number of type parameters is 3.
        let dim_map = if partition_value.ty.params.len() == 3 {
            let TypeParam::DimMap(dim_map) = &partition_value.ty.params[2] else {
                return self.jit_error_result(
                    &call_expr.span(),
                    "the type parameter must be a DimMap type",
                );
            };
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
        } else {
            let mut r = vec![];
            for i in 0..static_shape.len() {
                r.push(i as i32);
            }
            r
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
            // Because the indices may be remapped via a permutation of the tile dimensions,
            // we need to remap the tensor's shape as well.
            let remapped_i = dim_map[i] as usize;
            let static_tile_dim = static_tile[i];
            let static_shape_dim = static_shape[remapped_i];
            let is_static_shape_dim = static_shape_dim != -1;
            let index_value = indexes.remove(0);
            if index_value.bounds.is_some() && is_static_shape_dim {
                if let Some(bounds) = index_value.bounds {
                    // We can do a static bounds check.
                    if !(0 <= bounds.start
                        && bounds.end < static_shape_dim as i64 / static_tile_dim as i64)
                    {
                        return self.jit_error_result(
                            &call_expr.span(),
                            &format!(
                                "Bounds check failed: 0 <= {} && {} < {}",
                                bounds.start,
                                bounds.end,
                                static_shape_dim as i64 / static_tile_dim as i64
                            ),
                        );
                    }
                    return Ok(None);
                }
            }
            // In the rest of the cases, we need to generate a bounds check.
            let tile_dim_value = self.compile_constant(builder, generic_vars, static_tile_dim)?;
            let index_value = if let Some(bounds) = index_value.bounds {
                let index_upper_bound = bounds.end;
                self.compile_constant(builder, generic_vars, index_upper_bound as i32)?
            } else {
                index_value
            };
            let shape_dim_value = if is_static_shape_dim {
                self.compile_constant(builder, generic_vars, static_shape_dim)?
            } else {
                dynamic_shape[remapped_i].clone()
            };
            // shape_dim / tile_dim uses ceil_div:
            let div_result_value = self.compile_binary_op_from_values(
                builder,
                shape_dim_value.clone(),
                tile_dim_value,
                &TileBinaryOp::CeilDiv,
                generic_vars,
                ctx,
                None,
                &call_expr.span(),
            )?;
            let ineq_result_value = self.compile_binary_op_from_values(
                builder,
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
            let message = format!(
                "Detected out of bounds access during {}: index pos {}, tile dim={}",
                call_expr.to_token_stream(),
                i,
                static_tile_dim
            );
            let assert_builder =
                OperationBuilder::new("cuda_tile.assert", Location::unknown(&self.context));
            let assert_op = assert_builder
                .add_attributes(&[named_str_attr(&self.context, "message", &message)])
                .add_operands(&[result_value])
                .build()
                .unwrap();
            builder.append_operation(assert_op);
        }
        Ok(None)
    }
}
