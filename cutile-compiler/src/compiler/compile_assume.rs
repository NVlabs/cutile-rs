/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Assumption compilation: translates assumption function calls (e.g. `assume_div_by`,
//! `assume_bounds`, `assume_same_elements_*`) into MLIR `cuda_tile.assume` operations
//! within the CUDA Tile compiler. These provide optimization hints to the compiler
//! by asserting properties about values.

use crate::compiler::_function::CUDATileFunctionCompiler;
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;

use crate::cuda_tile::AssumeOperationBuilder;
use crate::error::JITError;
use crate::generics::GenericVars;
use crate::syn_utils::*;
use melior::ir::operation::OperationLike;
use melior::ir::{self, BlockLike, Value};
use syn::spanned::Spanned;
use syn::{Expr, ExprCall};

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    /// Compiles an assumption function call into MLIR.
    ///
    /// Assumption functions are compiler hints that provide optimization opportunities
    /// by asserting properties about values that the compiler can exploit. These
    /// functions use const-generic parameters to pass compile-time constants.
    ///
    /// ## Supported Assume Operations
    ///
    /// - `assume_div_by<DIVISOR>` - Value is divisible by DIVISOR
    /// - `assume_div_by_every_along<DIVISOR, EVERY, ALONG>` - Complex divisibility pattern
    /// - `assume_bounds_lower<LOWER>` - Value >= LOWER (inclusive)
    /// - `assume_bounds_upper<UPPER>` - Value <= UPPER (inclusive)
    /// - `assume_bounds<LOWER, UPPER>` - LOWER <= Value <= UPPER
    /// - `assume_same_elements_1d<GROUP0>` - Elements identical within groups (1D)
    /// - `assume_same_elements_2d<GROUP0, GROUP1>` - Elements identical within groups (2D)
    /// - `assume_same_elements_3d<GROUP0, GROUP1, GROUP2>` - Elements identical within groups (3D)
    /// - `assume_same_elements_4d<GROUP0, GROUP1, GROUP2, GROUP3>` - Elements identical within groups (4D)
    pub(crate) fn compile_assumption_call(
        &'c self,
        call_expr: &ExprCall,
        builder: &'c ir::Block<'c>,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
    ) -> Result<TileRustValue<'c, 'c>, JITError> {
        let Expr::Path(path_expr) = &*call_expr.func else {
            return self.jit_error_result(
                &call_expr.func.span(),
                "expected a simple function path for assume invocation",
            );
        };
        let ident = get_ident_from_path_expr(path_expr);
        let compiler_op_function = ident.to_string();
        let mut args = self.compile_call_args(builder, &call_expr.args, generic_vars, ctx)?;
        let val = args.remove(0);
        let return_type = val.ty.clone();
        let Some(generic_args) = get_call_expression_generics(call_expr) else {
            return self.jit_error_result(
                &call_expr.span(),
                "`assume` requires generic arguments (e.g. `assume_bounds::<T, 0, 128>(...)`)",
            );
        };
        let predicate_args = get_generic_arg_ints::<i32>(&generic_args, Some(generic_vars));
        let Some(val_value) = val.value else {
            return self.jit_error_result(
                &call_expr.span(),
                "the first argument to `assume` must produce a value",
            );
        };
        self.compile_value_assumption(
            builder,
            val_value,
            compiler_op_function.as_str(),
            &predicate_args,
            return_type,
            &call_expr.span(),
        )
    }

    /// Generates MLIR assume operation with appropriate predicate attribute.
    pub(crate) fn compile_value_assumption(
        &'c self,
        builder: &'c ir::Block<'c>,
        assume_val: Value<'c, 'c>,
        assume_op_rust_function: &str,
        predicate_args: &[i32],
        return_type: TileRustType<'c>,
        span: &proc_macro2::Span,
    ) -> Result<TileRustValue<'c, 'c>, JITError> {
        let assume = AssumeOperationBuilder::new(&self.context, self.function_location());
        let assume = assume.value(assume_val);
        let predicate = match assume_op_rust_function {
            "assume_div_by" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_div_by` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!("#cuda_tile.div_by<{}>", predicate_args[0]),
                )?
            }
            "assume_div_by_every_along" => {
                if predicate_args.len() != 3 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_div_by_every_along` requires 3 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!(
                        "#cuda_tile.div_by<{}, every {} along {}>",
                        predicate_args[0], predicate_args[1], predicate_args[2]
                    ),
                )?
            }
            "assume_bounds_lower" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_bounds_lower` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!("#cuda_tile.bounded<{}, ?>", predicate_args[0]),
                )?
            }
            "assume_bounds_upper" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_bounds_upper` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!("#cuda_tile.bounded<?, {}>", predicate_args[0]),
                )?
            }
            "assume_bounds" => {
                if predicate_args.len() != 2 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_bounds` requires 2 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!(
                        "#cuda_tile.bounded<{}, {}>",
                        predicate_args[0], predicate_args[1]
                    ),
                )?
            }
            "assume_same_elements_1d" => {
                if predicate_args.len() != 1 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_1d` requires 1 generic argument, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!("#cuda_tile.same_elements<[{}]>", predicate_args[0]),
                )?
            }
            "assume_same_elements_2d" => {
                if predicate_args.len() != 2 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_2d` requires 2 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!(
                        "#cuda_tile.same_elements<[{}, {}]>",
                        predicate_args[0], predicate_args[1]
                    ),
                )?
            }
            "assume_same_elements_3d" => {
                if predicate_args.len() != 3 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_3d` requires 3 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!(
                        "#cuda_tile.same_elements<[{}, {}, {}]>",
                        predicate_args[0], predicate_args[1], predicate_args[2]
                    ),
                )?
            }
            "assume_same_elements_4d" => {
                if predicate_args.len() != 4 {
                    return self.jit_error_result(
                        span,
                        &format!(
                            "`assume_same_elements_4d` requires 4 generic arguments, got {}",
                            predicate_args.len()
                        ),
                    );
                }
                self.parse_named_attr(
                    "predicate",
                    &format!(
                        "#cuda_tile.same_elements<[{}, {}, {}, {}]>",
                        predicate_args[0], predicate_args[1], predicate_args[2], predicate_args[3]
                    ),
                )?
            }
            _ => {
                return self.jit_error_result(
                    span,
                    &format!("unrecognized assume operation `{assume_op_rust_function}`"),
                );
            }
        };
        let assume = assume.predicate(predicate.1);
        let Some(cuda_tile_ty) = return_type.cuda_tile_ty else {
            return self.jit_error_result(span, "assume requires a compilable tile type");
        };
        let assume = assume.result(cuda_tile_ty);
        let op_ref = builder.append_operation(assume.build().into());
        if !op_ref.verify() {
            return self.jit_error_result(
                span,
                &format!(
                    "failed to compile `{assume_op_rust_function}` (MLIR verification failed)"
                ),
            );
        }
        let value: Value = op_ref.result(0).unwrap().into();
        Ok(TileRustValue::new_value_kind_like(value, return_type))
    }
}
