/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline compilation for compiler2.
//!
//! Mechanical port of `compiler/compile_inline.rs` — handles inlining of
//! function calls and method calls. Only type and IR-emission changes; the
//! control flow, dispatch logic, and variable binding are identical.

use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;

use super::_function::CUDATileFunctionCompiler;
use super::_value::{CompilerContext, Mutability, TileRustValue};
use super::shared_utils::{STACK_GROW_SIZE, STACK_RED_ZONE};
use super::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::{GenericArgInference, GenericVars};
use crate::syn_utils::*;
use crate::types::*;

use cutile_ir::ir::{BlockId, Module};

use proc_macro2::Span;
use quote::ToTokens;
use std::collections::HashMap;
use syn::{Expr, ExprCall, ExprMethodCall, ItemFn, Type};

/// Port of `crate::compiler::utils::update_type_meta` for compiler2 value types.
/// Copies mutable type metadata fields from inner to outer context using a variable name mapping.
fn update_type_meta(
    inner_block_vars: &mut CompilerContext,
    outer_block_vars: &mut CompilerContext,
    outer2inner_vars: &HashMap<String, String>,
    _field_name: String,
) {
    let outer_keys: Vec<String> = outer_block_vars.var_keys();
    for outer_key in &outer_keys {
        let Some(outer_val) = outer_block_vars.vars.get(outer_key) else {
            continue;
        };
        if outer_val.mutability == Mutability::Mutable {
            if let Some(inner_key) = outer2inner_vars.get(outer_key) {
                if let Some(inner_val) = inner_block_vars.vars.get(inner_key) {
                    if inner_val.mutability == Mutability::Mutable {
                        let mut new_val = outer_val.clone();
                        new_val.type_meta = inner_val.type_meta.clone();
                        outer_block_vars.vars.insert(outer_key.clone(), new_val);
                    }
                }
            }
        }
    }
}

/// Rewrites every span in a syn AST node to a fixed target span.
///
/// When inlining library/core functions, the callee body's spans point into
/// the core module's source text.  Resolving those spans against the user
/// module's [`SpanBase`] produces nonsensical line numbers.  By rewriting all
/// spans to the call-site span we ensure errors point to the user's code.
struct CallSiteSpanSetter {
    target_span: Span,
}

impl VisitMut for CallSiteSpanSetter {
    fn visit_span_mut(&mut self, span: &mut Span) {
        *span = self.target_span;
    }
}

impl<'m> CUDATileFunctionCompiler<'m> {
    #[allow(clippy::too_many_arguments)]
    pub fn inline_function_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        module_name: &str,
        fn_item: &ItemFn,
        call_expr: &ExprCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _inline_function_call_debug_str = call_expr.to_token_stream().to_string();
            // println!("enter_function_call: {}", call_expr.to_token_stream().to_string());

            // Compile caller arguments.
            let call_arg_values =
                self.compile_call_args(module, block_id, &call_expr.args, generic_vars, ctx)?;
            // Map function generic params to caller generic args.
            let mut generic_arg_inference = GenericArgInference::new_function(fn_item.sig.clone());
            let call_arg_rust_tys = call_arg_values
                .iter()
                .map(|arg| arg.ty.rust_ty.clone())
                .collect::<Vec<_>>();
            // println!("{call_arg_rust_tys:#?}");
            generic_arg_inference.map_args_to_params(&call_arg_rust_tys, None);
            // Bind new variables.
            // The variables must:
            // - Have the names of the parameters in the callee.
            // - Have the type of parameters in the callee. This is an inductive property.
            let param_names = get_sig_param_names(&fn_item.sig);
            let (input_params, _output_param) = get_sig_types(&fn_item.sig, None);
            let mut call_variables = CompilerContext::empty();
            call_variables.module_scope.push(module_name.into());
            let mut outer2inner_map = HashMap::new();
            let sig_param_mutability = get_sig_param_mutability(&fn_item.sig);

            for i in 0..param_names.len() {
                let param_name = &param_names[i];
                let param_type = &input_params[i];
                let mut param_val = call_arg_values[i].clone();
                // TODO (hme): This may not be enough, depending on what level of inspection we require of compound / struct types.
                param_val.ty.rust_ty = param_type.clone();
                param_val.mutability = if sig_param_mutability[i] {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                call_variables.vars.insert(param_name.clone(), param_val);
                if let Some(call_arg_name) = get_ident_from_expr(&call_expr.args[i]) {
                    outer2inner_map.insert(call_arg_name.to_string(), param_name.clone());
                };
            }
            // Remap generic parameters.
            let expr_generic_args = get_call_expression_generics(call_expr);
            let call_generic_vars = if GenericVars::is_empty(&fn_item.sig.generics) {
                // If there are no generics, we're done.
                GenericVars::empty(&fn_item.sig.generics)?
            } else if expr_generic_args.is_some() {
                // If the caller specifies generics args, use them.
                generic_vars.from_expr_generic_args(&fn_item.sig.generics, &expr_generic_args)?
            } else {
                // If nothing is specified, try to infer an instance of GenericVars.
                let mut generic_arg_inference =
                    GenericArgInference::new_function(fn_item.sig.clone());
                let call_arg_rust_tys = call_arg_values
                    .iter()
                    .map(|arg| arg.ty.rust_ty.clone())
                    .collect::<Vec<_>>();
                generic_arg_inference.map_args_to_params(&call_arg_rust_tys, None);
                // println!("inline_function_call {:#?}: generic_vars={generic_vars:#?} \nexpr_generic_args={expr_generic_args:#?} \ngeneric_arg_inference={generic_arg_inference:#?}", fn_item.sig.ident.to_string());
                generic_arg_inference
                    .get_generic_vars_instance(generic_vars, self.modules.primitives())
            };
            // Add function call const generics as variables.
            for (key, value) in &call_generic_vars.inst_i32 {
                let tr_val = self.compile_constant(module, block_id, &call_generic_vars, *value)?;
                call_variables.vars.insert(key.clone(), tr_val);
            }
            // Add function call CGAs arrays as variables.
            for (key, value) in &call_generic_vars.inst_array {
                let arr_expr = syn::parse2::<Expr>(format!("{value:?}").parse().unwrap()).unwrap();
                let arr_ty =
                    syn::parse2::<Type>(format!("[i32;{}]", value.len()).parse().unwrap()).unwrap();
                let ty = self.compile_type(&arr_ty, &call_generic_vars, &HashMap::new())?;
                let tr_val = self
                    .compile_expression(
                        module,
                        block_id,
                        &arr_expr,
                        &call_generic_vars,
                        &mut call_variables,
                        ty,
                    )?
                    .expect("Failed to compile CGA as var.");
                call_variables.vars.insert(key.clone(), tr_val);
            }
            // println!("inline_function_call {:#?}: generic_args={generic_args:#?} \nexpr_generic_args={expr_generic_args:#?} \ncall_generic_args={call_generic_args:#?}", fn_item.sig.ident.to_string());
            // println!("inline_function_call {:#?}: \n variables={call_variables:#?}", fn_item.sig.ident.to_string());
            let result = self.compile_block(
                module,
                block_id,
                &fn_item.block,
                &call_generic_vars,
                &mut call_variables,
                return_type.clone(),
            )?;
            update_type_meta(
                &mut call_variables,
                ctx,
                &outer2inner_map,
                "token".to_string(),
            );
            // println!("exit_function_call: {}", call_expr.to_token_stream().to_string());
            if let Some(mut res) = result {
                if let Some(rt) = return_type {
                    // Use specified return type.
                    res.ty = rt;
                    return Ok(Some(res));
                };
                let type_params = res.ty.params;
                // println!("inline call res.ty.params: {:#?}", type_params);
                let Some(derived_ret_ty) = self.derive_type(
                    module,
                    block_id,
                    &Expr::Call(call_expr.clone()),
                    Some(type_params),
                    generic_vars,
                    ctx,
                )?
                else {
                    return self
                        .jit_error_result(&call_expr.func.span(), "Failed to derive return type");
                };
                res.ty = derived_ret_ty;
                Ok(Some(res))
            } else {
                Ok(None)
            }
        }) // stacker::maybe_grow
    }

    pub fn inline_method_call(
        &self,
        module: &mut Module,
        block_id: BlockId,
        method_call_expr: &ExprMethodCall,
        generic_vars: &GenericVars,
        ctx: &mut CompilerContext,
        return_type: Option<TileRustType>,
    ) -> Result<Option<TileRustValue>, JITError> {
        stacker::maybe_grow(STACK_RED_ZONE, STACK_GROW_SIZE, || {
            let _inline_method_call_debug_str = method_call_expr.to_token_stream().to_string();
            // println!("enter_method_call: {}", method_call_expr.to_token_stream().to_string());
            // Compile caller arguments.
            // Receiver is prepended to args, so value of receiver is present for method calls.
            // args have generics from outer scope for both receiver + method call args.
            let mut args = method_call_expr.args.clone();
            args.insert(0, *method_call_expr.receiver.clone());
            let call_arg_values =
                self.compile_call_args(module, block_id, &args, generic_vars, ctx)?;
            let receiver_rust_ty = &call_arg_values[0].ty.rust_ty;
            let impl_item_fn =
                self.modules
                    .get_impl_item_fn(receiver_rust_ty, method_call_expr, generic_vars)?;
            if impl_item_fn.is_none() {
                return self.jit_error_result(&method_call_expr.method.span(), "method not found");
            }
            let (module_name, impl_item, impl_method) = impl_item_fn.unwrap();
            // println!("Expr::MethodCall: {:#?}, generic_vars: {generic_vars:#?}", impl_item_fn.to_token_stream().to_string());

            // Remap function parameters.
            // Do this by constructing new values from the method's parameters.
            let self_ty = &*impl_item.self_ty;
            // Note that self_ty here is treated as a param type.
            // Bind new variables.
            // The variables must:
            // - Have the names of the parameters in the callee.
            // - Have the type of parameters in the callee. This is an inductive property.
            // get_sig_param_names includes value for self if the signature contains self.
            let param_names = get_sig_param_names(&impl_method.sig);
            let (input_params, _output_param) = get_sig_types(&impl_method.sig, Some(self_ty));
            let mut call_variables = CompilerContext::empty();
            call_variables.module_scope.push(module_name.clone());
            let mut outer2inner_map = HashMap::new();
            let sig_param_mutability = get_sig_param_mutability(&impl_method.sig);
            for i in 0..param_names.len() {
                let param_name = &param_names[i];
                let param_type = &input_params[i];
                let mut param_val = call_arg_values[i].clone();
                // TODO (hme): This may not be enough, depending on what level of inspection we require of compound / struct types.
                param_val.ty.rust_ty = param_type.clone();
                param_val.mutability = if sig_param_mutability[i] {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                call_variables.vars.insert(param_name.clone(), param_val);
                // Including self here.
                if let Some(call_arg_name) = get_ident_from_expr(&args[i]) {
                    outer2inner_map.insert(call_arg_name.to_string(), param_name.clone());
                };
            }
            // Remap generic parameters.
            // This is different from a function call, because passing generics to a method
            // does not capture all generics available within the method.
            let generic_arg_inference = GenericArgInference::new_method(&impl_item, &impl_method);
            let call_generic_vars = if generic_arg_inference.param2arg.is_empty() {
                // There are no generics in this method.
                GenericVars::empty(&impl_method.sig.generics)?
            } else {
                // Infer generics from call arguments, including self.
                let method_call_turbofish = &method_call_expr.turbofish;
                // println!("infer generics for {}, \nturbofish={method_call_turbofish:#?}, \ngeneric_arg_inference: {generic_arg_inference:#?}", impl_method.sig.to_token_stream().to_string());
                let mut generic_arg_inference =
                    GenericArgInference::new_method(&impl_item, &impl_method);
                let call_arg_rust_tys = call_arg_values
                    .iter()
                    .map(|arg| arg.ty.rust_ty.clone())
                    .collect::<Vec<_>>();
                generic_arg_inference.map_args_to_params(&call_arg_rust_tys, Some(self_ty));
                // println!("sig={} \nargs={} \narg_map={:#?}", impl_method.sig.to_token_stream().to_string(), args.to_token_stream().to_string(), generic_arg_inference.param2arg);
                let inferred_generics = generic_arg_inference
                    .get_generic_vars_instance(generic_vars, self.modules.primitives());

                // If there are generics passed as part of the method call, capture them.
                if method_call_turbofish.is_some() {
                    let passed_generics = generic_vars
                        .from_expr_generic_args(&impl_method.sig.generics, method_call_turbofish)?;
                    inferred_generics.merge(passed_generics)?
                } else {
                    inferred_generics
                }
            };

            // Add method call const generics as variables.
            for (key, value) in &call_generic_vars.inst_i32 {
                let tr_val = self.compile_constant(module, block_id, generic_vars, *value)?;
                call_variables.vars.insert(key.clone(), tr_val);
            }
            for (key, value) in &call_generic_vars.inst_array {
                let arr_expr = syn::parse2::<Expr>(format!("{value:?}").parse().unwrap()).unwrap();
                let arr_ty =
                    syn::parse2::<Type>(format!("[i32;{}]", value.len()).parse().unwrap()).unwrap();
                let ty = self.compile_type(&arr_ty, &call_generic_vars, &HashMap::new())?;
                let tr_val = self
                    .compile_expression(
                        module,
                        block_id,
                        &arr_expr,
                        &call_generic_vars,
                        &mut call_variables,
                        ty,
                    )?
                    .expect("Failed to compile CGA as var.");
                call_variables.vars.insert(key.clone(), tr_val);
            }
            // println!("inline_method_call {:#?}: generic_vars={generic_vars:#?} \nexpr_generic_args={expr_generic_args:#?} \ncall_generic_args={call_generic_args:#?}", impl_method.sig.ident.to_string());
            // Method calls are always core/library methods (user kernel code
            // does not define impl blocks).  Rewrite all spans to the call
            // site so that errors point to the user's method call expression
            // rather than into the library source.
            let mut compile_block = impl_method.block.clone();
            let mut setter = CallSiteSpanSetter {
                target_span: method_call_expr.span(),
            };
            setter.visit_block_mut(&mut compile_block);
            let result = self.compile_block(
                module,
                block_id,
                &compile_block,
                &call_generic_vars,
                &mut call_variables,
                return_type.clone(),
            )?;
            update_type_meta(
                &mut call_variables,
                ctx,
                &outer2inner_map,
                "token".to_string(),
            );
            // println!("exit_method_call: {}", method_call_expr.to_token_stream().to_string());
            if let Some(mut res) = result {
                if let Some(rt) = return_type {
                    // Use specified rust type.
                    // We don't want the entire provided type, because the computed TileRustValue
                    // contains cuda tile type information that can't be inferred.
                    res.ty.rust_ty = rt.rust_ty;
                    return Ok(Some(res));
                };
                // Reverse type inference for resulting rust type in res.
                let type_params = res.ty.params;
                let Some(derived_ret_ty) = self.derive_type(
                    module,
                    block_id,
                    &Expr::MethodCall(method_call_expr.clone()),
                    Some(type_params),
                    generic_vars,
                    ctx,
                )?
                else {
                    return self.jit_error_result(
                        &method_call_expr.method.span(),
                        "Failed to derive return type",
                    );
                };
                res.ty = derived_ret_ty;
                Ok(Some(res))
            } else {
                Ok(None)
            }
        }) // stacker::maybe_grow
    }
}
