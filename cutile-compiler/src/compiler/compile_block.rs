/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block compilation: translates Rust `syn::Block` AST nodes into MLIR operations
//! within the CUDA Tile compiler. Handles statement-level constructs including
//! let bindings, assignments, control flow terminators (continue, break, return),
//! and tuple destructuring.

use crate::compiler::_function::{CUDATileFunctionCompiler, STACK_GROW_SIZE, STACK_RED_ZONE};
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;

use crate::error::JITError;
use crate::error::SpannedJITError;
use crate::generics::GenericVars;
use crate::syn_utils::*;
use crate::types::*;
use cuda_tile_rs::cuda_tile;
use melior::ir::operation::OperationBuilder;
use melior::ir::{self, BlockLike, Location};
use quote::ToTokens;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{Expr, Item, Pat, Stmt};

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    pub fn compile_block(
        &'c self,
        builder: &'c ir::Block<'c>,
        block_expr: &syn::Block,
        generic_args: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _block_debug_str = block_expr.to_token_stream().to_string();
            let mut terminator_encountered = None;
            let mut return_value: Option<TileRustValue> = None;
            let location = Location::unknown(&self.context);
            let num_statements = &block_expr.stmts.len();
            for (i, statement) in block_expr.stmts.iter().enumerate() {
                let is_last = i == num_statements - 1;
                match statement {
                    Stmt::Local(local) => {
                        let var_name: Option<String>;
                        let mut ct_ty: Option<TileRustType> = None;
                        let mut mutability: bool = false;
                        match &local.pat {
                            // If this changes, make sure changes are reflected in compileer_utils::collect_mutated_variables.
                            Pat::Type(pat_type) => {
                                let pat_mutability = get_pat_mutability(&pat_type.pat);
                                let ty_mutability = get_type_mutability(&pat_type.ty);
                                mutability = pat_mutability || ty_mutability;
                                match &*pat_type.pat {
                                    Pat::Ident(pat_ident) => {
                                        var_name = Some(pat_ident.ident.to_string());
                                    }
                                    Pat::Tuple(pat_tuple) => {
                                        // Handle typed tuple destructuring: let (a, b): (T1, T2) = expr;
                                        ct_ty = self.compile_type(
                                            &*pat_type.ty,
                                            generic_args,
                                            &HashMap::new(),
                                        )?;

                                        let Some(init) = &local.init else {
                                            return self.jit_error_result(
                                                &local.span(),
                                                "tuple destructuring requires an initializer expression",
                                            );
                                        };
                                        let Some(tuple_value) = self.compile_expression(
                                            builder,
                                            &*init.expr,
                                            generic_args,
                                            ctx,
                                            ct_ty.clone(),
                                        )?
                                        else {
                                            return self.jit_error_result(
                                                &init.expr.span(),
                                                "failed to compile tuple initializer expression",
                                            );
                                        };

                                        // Extract variable names
                                        let mut tuple_var_names = vec![];
                                        for elem in &pat_tuple.elems {
                                            match elem {
                                            Pat::Ident(ident) => {
                                                tuple_var_names.push(ident.ident.to_string());
                                            },
                                            _ => return self.jit_error_result(
                                                &elem.span(),
                                                "only simple variable names are supported in tuple destructuring patterns",
                                            ),
                                        }
                                        }

                                        // Bind each element
                                        if tuple_value.kind == Kind::Compound {
                                            let Some(elements) = &tuple_value.values else {
                                                return self.jit_error_result(
                                                    &init.expr.span(),
                                                    "internal: expected compound value for tuple destructuring",
                                                );
                                            };
                                            if elements.len() != tuple_var_names.len() {
                                                return self.jit_error_result(
                                                    &init.expr.span(),
                                                    &format!(
                                                        "tuple pattern has {} bindings but the expression produces {} values",
                                                        tuple_var_names.len(),
                                                        elements.len()
                                                    ),
                                                );
                                            }
                                            for (i, var_name) in tuple_var_names.iter().enumerate()
                                            {
                                                let mut elem_value = elements[i].clone();
                                                elem_value.mutability = if mutability {
                                                    Mutability::Mutable
                                                } else {
                                                    Mutability::Immutable
                                                };
                                                ctx.vars.insert(var_name.clone(), elem_value);
                                            }
                                        } else {
                                            return self.jit_error_result(
                                                &init.expr.span(),
                                                "right-hand side of tuple destructuring must be a tuple expression",
                                            );
                                        }
                                        continue;
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &pat_type.pat.span(),
                                            "this pattern form is not supported on the left side of a let binding",
                                        )
                                    }
                                }
                                ct_ty = self.compile_type(
                                    &*pat_type.ty,
                                    generic_args,
                                    &HashMap::new(),
                                )?;
                            }
                            Pat::Ident(pat_ident) => {
                                // Nothing to do. Try to infer.
                                var_name = Some(pat_ident.ident.to_string());
                            }
                            Pat::Tuple(pat_tuple) => {
                                // Handle tuple destructuring: let (a, b) = expr;
                                let Some(init) = &local.init else {
                                    return self.jit_error_result(
                                        &local.span(),
                                        "tuple destructuring requires an initializer expression",
                                    );
                                };

                                // Compile the RHS expression to get the tuple value
                                let Some(tuple_value) = self.compile_expression(
                                    builder,
                                    &*init.expr,
                                    generic_args,
                                    ctx,
                                    ct_ty.clone(),
                                )?
                                else {
                                    return self.jit_error_result(
                                        &init.expr.span(),
                                        "failed to compile tuple initializer expression",
                                    );
                                };

                                // Extract variable names from tuple pattern
                                let mut tuple_var_names = vec![];
                                for elem in &pat_tuple.elems {
                                    match elem {
                                        Pat::Ident(ident) => {
                                            tuple_var_names.push(ident.ident.to_string());
                                        }
                                        _ => return self.jit_error_result(
                                            &elem.span(),
                                            "only simple variable names are supported in tuple destructuring patterns",
                                        ),
                                    }
                                }

                                // Extract each element from the tuple value and bind it
                                if tuple_value.kind == Kind::Compound {
                                    let Some(elements) = &tuple_value.values else {
                                        return self.jit_error_result(
                                            &init.expr.span(),
                                            "internal: expected compound value for tuple destructuring",
                                        );
                                    };
                                    if elements.len() != tuple_var_names.len() {
                                        return self.jit_error_result(
                                            &init.expr.span(),
                                            &format!(
                                                "tuple pattern has {} bindings but the expression produces {} values",
                                                tuple_var_names.len(),
                                                elements.len()
                                            ),
                                        );
                                    }
                                    for (i, var_name) in tuple_var_names.iter().enumerate() {
                                        let mut elem_value = elements[i].clone();
                                        elem_value.mutability = if mutability {
                                            Mutability::Mutable
                                        } else {
                                            Mutability::Immutable
                                        };
                                        ctx.vars.insert(var_name.clone(), elem_value);
                                    }
                                } else {
                                    return self.jit_error_result(
                                        &init.expr.span(),
                                        "right-hand side of tuple destructuring must be a tuple expression",
                                    );
                                }

                                // Skip the normal let binding logic below
                                continue;
                            }
                            _ => {
                                return self.jit_error_result(
                                    &local.pat.span(),
                                    "this pattern form is not supported in let bindings",
                                );
                            }
                        }
                        if var_name.is_none() {
                            return self.jit_error_result(
                                &local.span(),
                                "unable to determine variable name for let binding",
                            );
                        }
                        let var_name = var_name.unwrap();
                        match &local.init {
                            Some(init) => {
                                match self.compile_expression(
                                    builder,
                                    &*init.expr,
                                    generic_args,
                                    ctx,
                                    ct_ty,
                                )? {
                                    Some(mut value) => {
                                        // Doesn't matter what this returns since we're overwriting the previous binding.
                                        value.mutability = if mutability {
                                            Mutability::Mutable
                                        } else {
                                            Mutability::Immutable
                                        };
                                        ctx.vars.insert(var_name, value);
                                    }
                                    None => {
                                        return self.jit_error_result(
                                            &init.expr.span(),
                                            &format!(
                                                "failed to compile initializer: `{}`",
                                                init.expr.to_token_stream().to_string()
                                            ),
                                        )
                                    }
                                }
                            }
                            None => {
                                return self.jit_error_result(
                                    &local.span(),
                                    "let bindings must have an initializer expression",
                                )
                            }
                        };
                    }
                    Stmt::Item(item) => {
                        let mut binding_name: Option<String> = None;
                        let mut ct_ty: Option<TileRustType> = None;
                        match item {
                            Item::Const(const_item) => {
                                // This is like a let binding.
                                binding_name = Some(const_item.ident.to_string());
                                ct_ty = self.compile_type(
                                    &*const_item.ty,
                                    generic_args,
                                    &HashMap::new(),
                                )?;
                                let Some(binding_name) = binding_name else {
                                    return self.jit_error_result(
                                        &const_item.span(),
                                        "unable to determine name for const binding",
                                    );
                                };
                                match self.compile_expression(
                                    builder,
                                    &*const_item.expr,
                                    generic_args,
                                    ctx,
                                    ct_ty,
                                )? {
                                    Some(mut value) => {
                                        // Doesn't matter what this returns since we're overwriting the previous binding.
                                        value.mutability = Mutability::Immutable;
                                        // TODO (hme): Const bindings are just vars with exact bounds.
                                        ctx.vars.insert(binding_name, value);
                                    }
                                    None => {
                                        return self.jit_error_result(
                                            &const_item.expr.span(),
                                            &format!(
                                                "failed to compile const initializer: `{}`",
                                                const_item.expr.to_token_stream().to_string()
                                            ),
                                        )
                                    }
                                }
                            }
                            _ => {
                                return self.jit_error_result(
                                    &item.span(),
                                    "only `const` item definitions are supported inside function bodies",
                                )
                            }
                        };
                    }
                    Stmt::Expr(expr, semicolon) => {
                        match expr {
                            // Loop-related terminators.
                            Expr::Continue(_continue_expr) => {
                                let Some(loop_carry_var_names) = &ctx.carry_vars else {
                                    return self.jit_error_result(
                                        &expr.span(),
                                        "`continue` cannot be used outside of a loop",
                                    );
                                };
                                terminator_encountered = Some(BlockTerminator::Continue);
                                let loop_carry_values =
                                    ctx.unpack_some_vars(loop_carry_var_names)?;
                                let op = OperationBuilder::new("cuda_tile.continue", location)
                                    .add_operands(&loop_carry_values)
                                    .build()
                                    .unwrap();
                                let _op_ref = builder.append_operation(op);
                            }
                            Expr::Break(_break_expr) => {
                                let Some(loop_carry_var_names) = &ctx.carry_vars else {
                                    return self.jit_error_result(
                                        &expr.span(),
                                        "Executing break outside of loop is not supported.",
                                    );
                                };
                                // Break is a terminator, don't add continue after it
                                // Break exits the loop with the current loop-carried values
                                terminator_encountered = Some(BlockTerminator::Break);
                                let loop_carry_values =
                                    ctx.unpack_some_vars(loop_carry_var_names)?;
                                let op = OperationBuilder::new("cuda_tile.break", location)
                                    .add_operands(&loop_carry_values)
                                    .build()
                                    .unwrap();
                                let _op_ref = builder.append_operation(op);
                                // After break, we should not continue processing statements
                                // break;
                            }
                            Expr::Assign(assign_expr) => {
                                let var_name: String = match &*assign_expr.left {
                                    Expr::Path(path_expr) => {
                                        get_ident_from_path_expr(path_expr).to_string()
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &assign_expr.left.span(),
                                            "only simple variable names are supported on the left side of an assignment",
                                        )
                                    }
                                };
                                let mut ct_value: TileRustValue = match self.compile_expression(
                                    builder,
                                    &*assign_expr.right,
                                    generic_args,
                                    ctx,
                                    None,
                                )? {
                                    Some(value) => value,
                                    None => return self.jit_error_result(
                                        &assign_expr.right.span(),
                                        "failed to compile the right-hand side of this assignment",
                                    ),
                                };
                                ct_value.mutability = Mutability::Mutable;
                                ctx.vars.insert(var_name, ct_value);
                            }
                            Expr::Return(return_expr) => {
                                match &return_expr.expr {
                                    Some(expr) => {
                                        return_value = self.compile_expression(
                                            builder,
                                            &*expr,
                                            generic_args,
                                            ctx,
                                            return_type.clone(),
                                        )?;
                                    }
                                    None => return_value = None,
                                }
                                break;
                            }
                            _ => {
                                if is_last && semicolon.is_none() {
                                    return_value = self.compile_expression(
                                        builder,
                                        &*expr,
                                        generic_args,
                                        ctx,
                                        return_type.clone(),
                                    )?;
                                } else {
                                    self.compile_expression(
                                        builder,
                                        &*expr,
                                        generic_args,
                                        ctx,
                                        None,
                                    )?;
                                }
                            }
                        }
                    }
                    Stmt::Macro(macro_stmt) => {
                        self.compile_cuda_tile_macro(
                            builder,
                            &macro_stmt.mac,
                            generic_args,
                            ctx,
                            return_type.clone(),
                        )?;
                    }
                }
            }
            if terminator_encountered.is_none() {
                // Continue, return, and yield are required terminators for loops, functions, and if statements in TileIR,
                // but not in Rust. If no such terminator is encountered, then inject the default for this block type.
                let loop_carry_var_names = ctx.carry_vars.clone().unwrap_or(vec![]);
                match ctx.default_terminator {
                    Some(BlockTerminator::Yield) => {
                        // Include any return values here.
                        let (cuda_tile_return_values, _) = {
                            if let Some(result) = &return_value {
                                let cuda_tile_value =
                                    result.value.expect("Failed to obtain CUDA tile value.");
                                (
                                    vec![cuda_tile_value],
                                    Some(result.ty.clone_fresh(&self.context)),
                                )
                            } else {
                                (vec![], None)
                            }
                        };
                        let loop_carry_values = ctx.unpack_some_vars(&loop_carry_var_names)?;
                        let op = OperationBuilder::new("cuda_tile.yield", location)
                            .add_operands(&[loop_carry_values, cuda_tile_return_values].concat())
                            .build()
                            .unwrap();
                        let _ = builder.append_operation(op);
                    }
                    Some(BlockTerminator::Continue) => {
                        let loop_carry_values = ctx.unpack_some_vars(&loop_carry_var_names)?;
                        let op = OperationBuilder::new("cuda_tile.continue", location)
                            .add_operands(&loop_carry_values)
                            .build()
                            .unwrap();
                        let _ = builder.append_operation(op);
                    }
                    Some(BlockTerminator::Return) => {
                        self.resolve_span(&block_expr.span())
                            .jit_assert(loop_carry_var_names.len() == 0, "unexpected state")?;
                        if return_value.is_some() {
                            return self.jit_error_result(
                                &block_expr.span(),
                                "returning a value from this function is not supported",
                            );
                        }
                        let ret_op_builder =
                            cuda_tile::ReturnOperationBuilder::new(&self.context, location)
                                .operands(&[])
                                .build()
                                .into();
                        let _ = builder.append_operation(ret_op_builder);
                    }
                    Some(BlockTerminator::Break) => {
                        self.resolve_span(&block_expr.span())
                            .jit_error_result("unexpected default terminator type")?;
                    }
                    None => {}
                }
            }
            Ok(return_value)
        }) // stacker::maybe_grow
    }
}
