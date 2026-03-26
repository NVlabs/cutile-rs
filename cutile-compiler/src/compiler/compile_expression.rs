/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expression compilation: translates Rust `syn::Expr` AST nodes into MLIR operations
//! and `TileRustValue` results within the CUDA Tile compiler. This is the core dispatch
//! for all expression kinds including loops, conditionals, literals, calls, binary ops,
//! struct construction, tuples, arrays, indexing, macros, closures, etc.

use crate::bounds::Bounds;
use crate::compiler::_function::{CUDATileFunctionCompiler, STACK_GROW_SIZE, STACK_RED_ZONE};
pub use crate::compiler::_type::*;
pub use crate::compiler::_value::*;
use crate::compiler::utils::{
    collect_mutated_variables, collect_mutated_variables_from_block,
    collect_mutated_variables_loop, collect_mutated_variables_while, dedup,
    update_outer_block_type_meta,
};
use crate::error::JITError;
use crate::generics::GenericVars;
use crate::syn_utils::*;
use crate::types::*;
use cuda_tile_rs::operation_parse;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::{self, Block, BlockLike, Location, Region, RegionLike, Value, ValueLike};
use proc_macro2::TokenTree;
use quote::ToTokens;
use std::collections::{BTreeMap, HashMap};
use syn::spanned::Spanned;
use syn::{parse_quote, Expr, Lit, Member, Pat, Type, UnOp};

impl<'m, 'c> CUDATileFunctionCompiler<'m> {
    pub fn compile_expression(
        &'c self,
        builder: &'c ir::Block<'c>,
        expr: &syn::Expr,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext<'c, 'c>,
        return_type: Option<TileRustType<'c>>,
    ) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _expr_debug_str = expr.to_token_stream().to_string();
            match expr {
                Expr::ForLoop(for_expr) => {
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
                    let Expr::Range(range_expr) = &*for_expr.expr else {
                        return self.jit_error_result(
                            &for_expr.expr.span(),
                            "only range expressions (e.g. `0..n`) are supported in for loops",
                        );
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
                    let Some(start_val) = self.compile_expression(
                        builder,
                        start_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?
                    else {
                        return self.jit_error_result(
                            &start_expr.span(),
                            "failed to compile range start expression",
                        );
                    };
                    let Some(end_val) = self.compile_expression(
                        builder,
                        end_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?
                    else {
                        return self.jit_error_result(
                            &end_expr.span(),
                            "failed to compile range end expression",
                        );
                    };
                    let iterand_lower_const = start_val.bounds.clone();
                    let iterand_upper_const = end_val.bounds.clone();
                    let lower_bound = start_val.value.unwrap();
                    let upper_bound = end_val.value.unwrap();
                    let step_value = self.compile_constant(builder, generic_vars, 1)?;
                    let step: Value = step_value.value.ok_or_else(|| {
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
                    let for_iterand_type = lower_bound.r#type();
                    let loop_carry_vars = collect_mutated_variables(for_expr)?
                        .into_iter()
                        .collect::<Vec<_>>();
                    let loop_carry_args = ctx.unpack_some_vars(&loop_carry_vars)?;
                    let loop_carry_arg_tys = loop_carry_args
                        .iter()
                        .map(|val| val.r#type())
                        .collect::<Vec<_>>();

                    let location = Location::unknown(&self.context);
                    let for_builder = OperationBuilder::new("cuda_tile.for", location);
                    let for_op = for_builder
                        .add_operands(&[lower_bound, upper_bound, step])
                        .add_operands(&loop_carry_args)
                        .add_results(&loop_carry_arg_tys)
                        .add_regions([{
                            // Add iterand as argument.
                            let loop_block_args =
                                &[&[for_iterand_type], loop_carry_arg_tys.as_slice()].concat();
                            let loop_block = Block::new(
                                &loop_block_args
                                    .iter()
                                    .map(|ty| (ty.clone(), location))
                                    .collect::<Vec<_>>(),
                            );
                            let mut for_variables = ctx.clone();
                            // Update loop carry variables within the for loop
                            // to the mutable variables accessed in this operation.
                            let mut block_args = vec![];
                            for i in 1..loop_block.argument_count() {
                                let val: Value = loop_block.argument(i).unwrap().into();
                                block_args.push(val);
                            }
                            for_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                            if let Some(iterand_ident) = maybe_iterand_ident {
                                // maybe_iterand_ident is None if it is wild.
                                // If it's an ident, then add the iterand as a var.
                                let iterand_name = iterand_ident.ident.to_string();
                                let iterand_mlir_val: Value =
                                    loop_block.argument(0).unwrap().into();
                                // This has the same type as start/end val.
                                let iterand_ty = start_val.ty.clone();
                                // If the loop bounds are const, then we can put a bound on the iterand.
                                // Subtract upper bound by 1, since it is the open end of the interval [start, end).
                                let iterand_val = match (iterand_lower_const, iterand_upper_const) {
                                    (Some(iterand_lower_const), Some(iterand_upper_const)) => {
                                        let bounds = Bounds::new(
                                            iterand_lower_const.start,
                                            iterand_upper_const.end - 1,
                                        );
                                        let mut iterand_val = self.compile_value_assumption(
                                            &loop_block,
                                            iterand_mlir_val.clone(),
                                            "assume_bounds",
                                            &[bounds.start as i32, bounds.end as i32],
                                            iterand_ty,
                                            &for_expr.span(),
                                        )?;
                                        iterand_val.bounds = Some(bounds);
                                        iterand_val
                                    }
                                    (Some(iterand_lower_const), None) => self
                                        .compile_value_assumption(
                                            &loop_block,
                                            iterand_mlir_val.clone(),
                                            "assume_bounds_lower",
                                            &[iterand_lower_const.start as i32],
                                            iterand_ty,
                                            &for_expr.span(),
                                        )?,
                                    (None, Some(iterand_upper_const)) => self
                                        .compile_value_assumption(
                                            &loop_block,
                                            iterand_mlir_val.clone(),
                                            "assume_bounds_upper",
                                            &[iterand_upper_const.end as i32 - 1],
                                            iterand_ty,
                                            &for_expr.span(),
                                        )?,
                                    (None, None) => TileRustValue::new_value_kind_like(
                                        iterand_mlir_val.clone(),
                                        start_val.ty.clone(),
                                    ),
                                };
                                for_variables.vars.insert(iterand_name, iterand_val);
                            }
                            for_variables.carry_vars = Some(loop_carry_vars.clone());
                            for_variables.default_terminator = Some(BlockTerminator::Continue);
                            // TODO (hme): Support returns?
                            self.compile_block(
                                &loop_block,
                                &for_expr.body,
                                &generic_vars,
                                &mut for_variables,
                                return_type,
                            )?;
                            let region = Region::new();
                            region.append_block(loop_block);
                            region
                        }])
                        .build()
                        .unwrap();
                    // TODO (hme): This fails with "operand #0 does not dominate this use"
                    //  This may be a bug.
                    //  The compiled module in its entirety still passes verification.
                    // assert!(for_op.verify());
                    let for_loop_ref = builder.append_operation(for_op);
                    let num_results = for_loop_ref.result_count();
                    let mut result_values: Vec<Value> = vec![];
                    for i in 0..num_results {
                        let val: Value = for_loop_ref.result(i).unwrap().into();
                        result_values.push(val);
                    }
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
                        .map(|val| val.r#type())
                        .collect::<Vec<_>>();

                    let location = Location::unknown(&self.context);
                    let loop_builder = OperationBuilder::new("cuda_tile.loop", location);
                    let loop_op = loop_builder
                        .add_operands(&loop_carry_args)
                        .add_results(&loop_carry_arg_tys)
                        .add_regions([{
                            let loop_block = Block::new(
                                &loop_carry_arg_tys
                                    .iter()
                                    .map(|ty| (ty.clone(), location))
                                    .collect::<Vec<_>>(),
                            );
                            let mut loop_variables = ctx.clone();
                            let mut block_args = vec![];
                            for i in 0..loop_block.argument_count() {
                                let val: Value = loop_block.argument(i).unwrap().into();
                                block_args.push(val);
                            }
                            loop_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                            loop_variables.carry_vars = Some(loop_carry_vars.clone());
                            loop_variables.default_terminator = Some(BlockTerminator::Continue);

                            // Evaluate condition
                            let Some(TileRustValue {
                                value: Some(condition_val),
                                ..
                            }) = self.compile_expression(
                                &loop_block,
                                &*while_expr.cond,
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
                            let condition_check = OperationBuilder::new("cuda_tile.if", location)
                                .add_operands(&[condition_val])
                                .add_regions([
                                    {
                                        // Then: continue to body (just yield, body comes next)
                                        let then_block = Block::new(&[]);
                                        let yield_op =
                                            OperationBuilder::new("cuda_tile.yield", location)
                                                .build()
                                                .unwrap();
                                        then_block.append_operation(yield_op);
                                        let region = Region::new();
                                        region.append_block(then_block);
                                        region
                                    },
                                    {
                                        // Else: break out
                                        let else_block = Block::new(&[]);
                                        let break_values =
                                            loop_variables.unpack_some_vars(&loop_carry_vars)?;
                                        let break_op =
                                            OperationBuilder::new("cuda_tile.break", location)
                                                .add_operands(&break_values)
                                                .build()
                                                .unwrap();
                                        else_block.append_operation(break_op);
                                        let region = Region::new();
                                        region.append_block(else_block);
                                        region
                                    },
                                ])
                                .build()
                                .unwrap();
                            loop_block.append_operation(condition_check);

                            // Execute body
                            self.compile_block(
                                &loop_block,
                                &while_expr.body,
                                generic_vars,
                                &mut loop_variables,
                                return_type.clone(),
                            )?;
                            // compile_block will inject continue at the end

                            let region = Region::new();
                            region.append_block(loop_block);
                            region
                        }])
                        .build()
                        .unwrap();

                    let loop_ref = builder.append_operation(loop_op);
                    let num_results = loop_ref.result_count();
                    let mut result_values: Vec<Value> = vec![];
                    for i in 0..num_results {
                        let val: Value = loop_ref.result(i).unwrap().into();
                        result_values.push(val);
                    }
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
                        .map(|val| val.r#type())
                        .collect::<Vec<_>>();

                    let location = Location::unknown(&self.context);
                    let loop_builder = OperationBuilder::new("cuda_tile.loop", location);
                    let loop_op = loop_builder
                        .add_operands(&loop_carry_args)
                        .add_results(&loop_carry_arg_tys)
                        .add_regions([{
                            let loop_block = Block::new(
                                &loop_carry_arg_tys
                                    .iter()
                                    .map(|ty| (ty.clone(), location))
                                    .collect::<Vec<_>>(),
                            );
                            let mut loop_variables = ctx.clone();
                            let mut block_args = vec![];
                            for i in 0..loop_block.argument_count() {
                                let val: Value = loop_block.argument(i).unwrap().into();
                                block_args.push(val);
                            }
                            loop_variables.repack_some_vars(&loop_carry_vars, &block_args, true)?;
                            loop_variables.carry_vars = Some(loop_carry_vars.clone());
                            loop_variables.default_terminator = Some(BlockTerminator::Continue);

                            // Execute loop body (must contain break to exit)
                            // The body should handle its own terminator (break/continue)
                            self.compile_block(
                                &loop_block,
                                &loop_expr.body,
                                generic_vars,
                                &mut loop_variables,
                                return_type.clone(),
                            )?;

                            // Note: compile_block will inject continue if not already present
                            let region = Region::new();
                            region.append_block(loop_block);
                            region
                        }])
                        .build()
                        .unwrap();

                    let loop_ref = builder.append_operation(loop_op);
                    let num_results = loop_ref.result_count();
                    let mut result_values: Vec<Value> = vec![];
                    for i in 0..num_results {
                        let val: Value = loop_ref.result(i).unwrap().into();
                        result_values.push(val);
                    }
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
                    let Some(conditional_val) = self.compile_expression(
                        builder,
                        &*if_expr.cond,
                        generic_vars,
                        ctx,
                        return_type.clone(),
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
                            let branch_block = match (bounds.start, &if_expr.else_branch) {
                                (1, _) => Some(&if_expr.then_branch),
                                (0, Some((_Else, else_expr))) => {
                                    let Expr::Block(block_expr) = &**else_expr else {
                                        return self.jit_error_result(
                                            &else_expr.span(),
                                            "only block expressions (`{ ... }`) are supported in else branches",
                                        );
                                    };
                                    Some(&block_expr.block)
                                }
                                _ => {
                                    // Do nothing since the conditional is false and there is no else branch.
                                    None
                                }
                            };
                            if let Some(branch_block) = branch_block {
                                let mut block_vars = ctx.clone();
                                // This is inlined, so no need to inject a terminator.
                                block_vars.default_terminator = None;
                                let res = self.compile_block(
                                    builder,
                                    branch_block,
                                    generic_vars,
                                    &mut block_vars,
                                    None,
                                )?;
                                let carry_vars =
                                    collect_mutated_variables_from_block(branch_block)?
                                        .into_iter()
                                        .collect::<Vec<_>>();
                                let result_values = block_vars.unpack_some_vars(&carry_vars)?;
                                ctx.repack_some_vars(&carry_vars, &result_values, true)?;
                                return Ok(res);
                            }
                            return Ok(None);
                        }
                    }

                    // The if/then block must yield captured mutable variables.
                    let then_captured_vars =
                        collect_mutated_variables_from_block(&if_expr.then_branch)?
                            .into_iter()
                            .collect::<Vec<_>>();
                    let else_captured_vars = {
                        if let Some((_Else, else_expr)) = &if_expr.else_branch {
                            let Expr::Block(block_expr) = &**else_expr else {
                                return self.jit_error_result(
                                    &else_expr.span(),
                                    "only block expressions (`{ ... }`) are supported in else branches",
                                );
                            };
                            collect_mutated_variables_from_block(&block_expr.block)?
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
                    let location = self.location_from_span(&if_expr.span());

                    let if_builder = OperationBuilder::new("cuda_tile.if", location);
                    let (then_region, then_return_type) = {
                        let mut block_vars = ctx.clone();
                        block_vars.carry_vars = Some(if_captured_var_names.clone());
                        block_vars.default_terminator = Some(BlockTerminator::Yield);
                        let then_block = Block::new(&[]);
                        let result = self.compile_block(
                            &then_block,
                            &if_expr.then_branch,
                            generic_vars,
                            &mut block_vars,
                            return_type.clone(),
                        )?;
                        let (_cuda_tile_return_values, return_type) = {
                            if let Some(result) = result {
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
                        let region = Region::new();
                        region.append_block(then_block);
                        (region, return_type)
                    };

                    let branch_result_type = {
                        if let Some(return_type) = &then_return_type {
                            let cuda_tile_ty = return_type.cuda_tile_ty;
                            vec![cuda_tile_ty.expect("Failed to obtain CUDA tile type.")]
                        } else {
                            vec![]
                        }
                    };

                    // We don't need to check return type. Both Rust and Tile IR compiler perform this check.
                    let (else_region, _else_return_type) = {
                        if let Some((_Else, else_expr)) = &if_expr.else_branch {
                            let Expr::Block(block_expr) = &**else_expr else {
                                return self.jit_error_result(
                                    &else_expr.span(),
                                    "only block expressions (`{ ... }`) are supported in else branches",
                                );
                            };
                            let mut block_vars = ctx.clone();
                            block_vars.carry_vars = Some(if_captured_var_names.clone());
                            block_vars.default_terminator = Some(BlockTerminator::Yield);
                            let else_block = Block::new(&[]);
                            let result = self.compile_block(
                                &else_block,
                                &block_expr.block,
                                generic_vars,
                                &mut block_vars,
                                then_return_type.clone(),
                            )?;
                            let (_cuda_tile_return_values, return_type) = {
                                if let Some(result) = result {
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
                            let region = Region::new();
                            region.append_block(else_block);
                            (region, return_type)
                        } else {
                            if then_return_type.is_some() {
                                return self.jit_error_result(
                                    &if_expr.span(),
                                    "if-expression without an else branch cannot produce a return type",
                                );
                            }
                            let else_block = Block::new(&[]);
                            // If there is only a then branch, there is no return value. Yield only the captured mutable vars.
                            let captured_mutable_vars =
                                ctx.unpack_some_vars(&if_captured_var_names)?;
                            let _yield_val = else_block.append_operation(
                                OperationBuilder::new("cuda_tile.yield", location)
                                    .add_operands(&captured_mutable_vars)
                                    .build()
                                    .unwrap(),
                            );
                            let region = Region::new();
                            region.append_block(else_block);
                            (region, None)
                        }
                    };

                    let if_result_types = {
                        let if_captured_var_args = ctx.unpack_some_vars(&if_captured_var_names)?;
                        let if_captured_var_arg_tys = if_captured_var_args
                            .iter()
                            .map(|val| val.r#type())
                            .collect::<Vec<_>>();
                        [if_captured_var_arg_tys, branch_result_type].concat()
                    };

                    let if_then_op = if_builder
                        .add_operands(&[condition_val])
                        .add_results(&if_result_types)
                        .add_regions([then_region, else_region])
                        .build()
                        .unwrap();

                    let if_op_ref = builder.append_operation(if_then_op);
                    let num_results = if_op_ref.result_count();
                    let mut result_values: Vec<Value> = vec![];
                    for i in 0..num_results {
                        let val: Value = if_op_ref.result(i).unwrap().into();
                        result_values.push(val);
                    }
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
                        let return_value: Value = result_values.pop().unwrap();
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
                        builder,
                        &block_expr.block,
                        &generic_vars,
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
                        builder,
                        &block_expr.block,
                        &generic_vars,
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
                            // TODO (hme): Unclear if this works in general for all structs.
                            if ["Shape", "Array"].contains(&struct_name.as_str()) {
                                self.compile_type(&field_type, generic_vars, &HashMap::new())?
                            } else {
                                // Returning None here is equivalent to asking the programmer to
                                // specify the field type.
                                None
                            }
                        } else {
                            None
                        };
                        let field_value: TileRustValue = match self.compile_expression(
                            builder,
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
                    return Ok(Some(TileRustValue::new_struct(fields, return_type)));
                }
                Expr::Reference(ref_expr) => {
                    // TODO (hme): Check whether all expr types can be supported.
                    let return_type = match return_type {
                        Some(ty) => {
                            if let Type::Reference(ref_type) = ty.rust_ty {
                                self.compile_type(&*ref_type.elem, generic_vars, &HashMap::new())?
                            } else {
                                None
                            }
                        }
                        _ => return_type,
                    };
                    match &*ref_expr.expr {
                        Expr::Array(_array_expr) => Ok(self.compile_expression(
                            builder,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::Path(_path_expr) => Ok(self.compile_expression(
                            builder,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::Repeat(_repeat_expr) => Ok(self.compile_expression(
                            builder,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        Expr::MethodCall(_method_call_expr) => Ok(self.compile_expression(
                            builder,
                            &ref_expr.expr,
                            generic_vars,
                            ctx,
                            return_type,
                        )?),
                        _ => {
                            return self.jit_error_result(
                                &ref_expr.span(),
                                "this reference expression form is not supported",
                            )
                        }
                    }
                }
                Expr::Tuple(tuple_expr) => {
                    let mut rust_types: Vec<syn::Type> = vec![];
                    let mut values: Vec<TileRustValue> = vec![];
                    for elem in &tuple_expr.elems {
                        match self.compile_expression(builder, &elem, generic_vars, ctx, None)? {
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
                                    Type::Array(array_type) => self.compile_type(
                                        &*array_type.elem,
                                        generic_vars,
                                        &HashMap::new(),
                                    )?,
                                    Type::Slice(slice) => {
                                        // TODO (hme): Confirm this is right.
                                        self.compile_type(
                                            &*slice.elem,
                                            generic_vars,
                                            &HashMap::new(),
                                        )?
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &elem.span(),
                                            &format!(
                                                "unexpected element type `{}`",
                                                return_type.rust_ty.to_token_stream().to_string()
                                            ),
                                        )
                                    }
                                }
                            }
                            None => None,
                        };
                        match self.compile_expression(builder, &elem, generic_vars, ctx, elem_ty)? {
                            Some(value) => values.push(value),
                            None => {
                                return self.jit_error_result(
                                    &elem.span(),
                                    "failed to compile array element",
                                )
                            }
                        };
                    }
                    let return_type = if return_type.is_none() {
                        if values.len() == 0 {
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
                    } else {
                        return_type.unwrap()
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
                    let Some(value) = self.compile_expression(
                        builder,
                        &repeat_expr.expr,
                        generic_vars,
                        ctx,
                        None,
                    )?
                    else {
                        return self.jit_error_result(
                            &repeat_expr.expr.span(),
                            "failed to compile repeat expression element",
                        );
                    };
                    let values: Vec<TileRustValue> = vec![value; len];
                    let return_type = if return_type.is_none() {
                        if values.len() == 0 {
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
                    } else {
                        return_type.unwrap()
                    };
                    Ok(Some(TileRustValue::new_compound(values, return_type)))
                }
                Expr::Path(path_expr) => {
                    if path_expr.path.segments.len() != 1 {
                        return self.jit_error_result(
                            &path_expr.span(),
                            "qualified paths (e.g. `module::name`) are not supported here; use a simple variable name",
                        );
                    }
                    let var_name = path_expr.path.segments.last().unwrap().ident.to_string();

                    // Handle None specially - it's a Rust Option::None value, not a variable
                    if var_name == "None" {
                        // None is used for optional parameters - return None to indicate absence
                        return Ok(None);
                    }

                    // TODO (hme): Assuming this is a var.
                    let value = match ctx.vars.get(&var_name) {
                        Some(ct_value) => ct_value,
                        None => {
                            return self.jit_error_result(
                                &path_expr.span(),
                                &format!("undefined variable `{}`", var_name),
                            )
                        }
                    };
                    Ok(Some(value.clone()))
                }
                Expr::Call(call_expr) => {
                    let call_expr_func_str = call_expr.func.to_token_stream().to_string();
                    let _args_str = call_expr.args.to_token_stream().to_string();
                    match &*call_expr.func {
                        Expr::Path(path_expr) => {
                            let ident = get_ident_from_path_expr(&path_expr);
                            // Handle Some(...) specially - it's a Rust Option constructor, not a function call
                            if ident.to_string() == "Some" {
                                // Some is used for optional parameters - extract the inner expression and compile it
                                if call_expr.args.len() == 1 {
                                    return Ok(self.compile_expression(
                                        builder,
                                        &call_expr.args[0],
                                        generic_vars,
                                        ctx,
                                        return_type,
                                    )?);
                                } else {
                                    return self.jit_error_result(
                                        &call_expr.span(),
                                        &format!(
                                            "`Some()` expects exactly one argument, got {}",
                                            call_expr.args.len()
                                        ),
                                    );
                                }
                            }
                            if let Some(_) = self
                                .modules
                                .get_cuda_tile_op_attrs(ident.to_string().as_str())
                            {
                                Ok(self.compile_cuda_tile_op_call(
                                    builder,
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
                                        builder,
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
                                        builder,
                                        module_name,
                                        fn_item,
                                        call_expr,
                                        &generic_vars,
                                        ctx,
                                        return_type,
                                    )?)
                                }
                            } else {
                                return self.jit_error_result(
                                    &call_expr.func.span(),
                                    &format!("call to `{}` is not supported", &call_expr_func_str),
                                );
                            }
                        }
                        _ => {
                            return self.jit_error_result(
                                &call_expr.func.span(),
                                &format!("Call to {} not supported.", &call_expr_func_str),
                            )
                        }
                    }
                }
                Expr::MethodCall(method_call_expr) => Ok(self.inline_method_call(
                    builder,
                    &method_call_expr,
                    &generic_vars,
                    ctx,
                    return_type,
                )?),
                Expr::Field(field_expr) => {
                    let Some(base) = self.compile_expression(
                        builder,
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
                                    &format!("{} is not a field.", field_name.to_string()),
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
                                        .expect(format!("Failed to parse literal {str}").as_str())
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
                            let Some(cuda_tile_ty) =
                                return_type.get_cuda_tile_element_type(&self.modules.primitives)?
                            else {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    "unable to determine type for numeric literal; add a type annotation",
                                );
                            };
                            let op_str = format!("%0 = cuda_tile.constant <{cuda_tile_ty}: {lit_string}> : !cuda_tile.tile<{cuda_tile_ty}>");
                            let op = operation_parse(&self.context, op_str.as_str(), None);
                            if op.is_none() {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    &format!("failed to compile {}", op_str),
                                );
                            }
                            let op = op.unwrap();
                            if !op.verify() {
                                return self.jit_error_result(
                                    &lit_expr.span(),
                                    &format!("failed to verify {}", op_str),
                                );
                            }
                            let op_ref = builder.append_operation(op);
                            let op_result = op_ref.result(0).unwrap();

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
                                op_result.into(),
                                ct_type,
                                bounds,
                            )))
                        }
                        _ => {
                            return self.jit_error_result(
                                &unary_expr.span(),
                                "Non-const unary expressions not supported.",
                            )
                        }
                    }
                }
                Expr::Cast(cast_expr) => {
                    let src_expr = self
                        .compile_expression(builder, &*cast_expr.expr, generic_vars, ctx, None)?
                        .unwrap();
                    let src_elem_ty: String = src_expr
                        .ty
                        .get_instantiated_rust_element_type(&self.modules.primitives)
                        .unwrap();
                    let dst_elem_ty: String = get_rust_element_type_primitive(&cast_expr.ty);
                    match (src_elem_ty.as_str(), dst_elem_ty.as_str()) {
                        ("i32", "u32") => {}
                        ("i64", "u64") => {}
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
                        match get_lit_type(lit_expr) {
                            Some(ty) => self.compile_type(&ty, generic_vars, &HashMap::new())?,
                            None => None,
                        }
                    } else {
                        return_type
                    };
                    let Some(return_type) = return_type else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!(
                                "Failed to infer type for lit expr {}.",
                                lit_expr.to_token_stream().to_string()
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
                                .expect(format!("Failed to parse literal {str}").as_str())
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
                        return_type.get_cuda_tile_element_type(&self.modules.primitives)?
                    else {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            "unable to determine type for numeric literal; add a type annotation",
                        );
                    };
                    let op_str = format!("%0 = cuda_tile.constant <{cuda_tile_ty}: {lit_string}> : !cuda_tile.tile<{cuda_tile_ty}>");
                    let op = operation_parse(&self.context, op_str.as_str(), None);
                    if op.is_none() {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!("failed to compile {}", op_str),
                        );
                    }
                    let op = op.unwrap();
                    if !op.verify() {
                        return self.jit_error_result(
                            &lit_expr.span(),
                            &format!("failed to verify {}", op_str),
                        );
                    }
                    let op_ref = builder.append_operation(op);
                    let op_result = op_ref.result(0).unwrap();

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
                        op_result.into(),
                        ct_type,
                        bounds,
                    )))
                }
                Expr::Binary(bin_expr) => {
                    // These are type-checked by Rust, so just do whatever the expression is asking.
                    Ok(self.compile_binary_op(
                        builder,
                        &bin_expr,
                        generic_vars,
                        ctx,
                        return_type.clone(),
                    )?)
                }
                Expr::Paren(paren_expr) => Ok(self.compile_expression(
                    builder,
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
                            //  and in rewrite_variadics (proc macro side).
                            let mut args = vec![];
                            let mut is_cga = false;
                            let mut is_consts = false;
                            for token in mac_expr.mac.tokens.clone() {
                                if is_cga && is_consts {
                                    return self.jit_error_result(
                                        &mac_expr.span(),
                                        &format!("inconsistent arguments to `{mac_name}!`: cannot mix CGA and literal arguments"),
                                    );
                                }
                                match token {
                                    TokenTree::Literal(lit) => {
                                        args.push(lit.to_string());
                                    }
                                    TokenTree::Ident(ident) => {
                                        let const_var = ident.to_string();
                                        if let Some(cga) = generic_vars.inst_array.get(&const_var) {
                                            is_cga = true;
                                            args = cga
                                                .iter()
                                                .map(|x| x.to_string())
                                                .collect::<Vec<String>>();
                                        } else {
                                            is_consts = true;
                                            let mut is_const =
                                                generic_vars.get_i32(&const_var).is_some();
                                            let const_var_value = ctx.vars.get(&const_var).unwrap();
                                            if let Some(bounds) = const_var_value.bounds {
                                                is_const = is_const || bounds.is_exact();
                                            }
                                            if !is_const {
                                                return self.jit_error_result(
                                                    &mac_expr.span(),
                                                    "all arguments to `const_shape!` must be compile-time constants",
                                                );
                                            }
                                            args.push(const_var);
                                        }
                                    }
                                    TokenTree::Punct(punct) => {
                                        if punct.as_char() == ',' {
                                            continue;
                                        } else {
                                            return self.jit_error_result(
                                                &mac_expr.span(),
                                                &format!("unexpected punctuation `{punct}` in macro arguments"),
                                            );
                                        }
                                    }
                                    _ => {
                                        return self.jit_error_result(
                                            &mac_expr.span(),
                                            "unexpected token in macro arguments",
                                        )
                                    }
                                }
                            }
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
                                    syn::parse2::<Type>(shape_str.parse().unwrap()).unwrap();
                                self.compile_type(&shape_ty, generic_vars, &HashMap::new())?
                            } else {
                                return_type.clone()
                            };
                            self.compile_expression(
                                builder,
                                &shape_expr,
                                generic_vars,
                                ctx,
                                return_type,
                            )?
                        }
                        _ => self.compile_cuda_tile_macro(
                            builder,
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
                    // that compile them into MLIR regions.
                    return self.jit_error_result(
                        &closure_expr.span(),
                        "closures are not supported as standalone values; \
                         they can only be used as arguments to operations like `reduce()` or `scan()`",
                    );
                }
                Expr::Index(index_expr) => {
                    let Some(expr_val) = self.compile_expression(
                        builder,
                        &*index_expr.expr,
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
                    if expr_val.kind != Kind::Compound {
                        return self.jit_error_result(
                            &index_expr.expr.span(),
                            "indexing is only supported on tuple/compound values",
                        );
                    }
                    // TODO (hme): Revisit this once we have proper type inference.
                    let i32_type: Type = parse_quote! { i32 };
                    let i32_type = self.compile_type(&i32_type, generic_vars, &HashMap::new())?;
                    let Some(index_val) = self.compile_expression(
                        builder,
                        &*index_expr.index,
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
                    let Some(mut values) = expr_val.values else {
                        return self.jit_error_result(
                            &index_expr.expr.span(),
                            "internal: compound value is missing its element list during index access",
                        );
                    };
                    return Ok(Some(values.remove(idx as usize)));
                }
                _ => {
                    return self
                        .jit_error_result(&expr.span(), "this expression form is not supported")
                }
            }
        }) // stacker::maybe_grow
    }
}
