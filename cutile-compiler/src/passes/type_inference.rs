/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler1-compatible type inference and DSL dispatch selection.
//!
//! This is intentionally narrower than a Rust type checker. It builds the
//! side-table shape needed by the compiler3 pass plan: expression types where
//! the DSL can infer them, selected method impls, and dispatch-wrapper calls
//! that should be erased before emission.

use crate::compiler::_function::CUDATileFunctionCompiler;
use crate::compiler::tile_rust_type::TileRustType;
use crate::error::JITError;
use crate::generics::{GenericArgInference, GenericVars, TypeInstance, TypeInstanceUserType};
use crate::passes::node_ids::{self, NodeId};
use crate::syn_utils::*;
use crate::types::TypeParam;
use std::collections::HashMap;
use syn::{
    Expr, ExprCall, ExprMethodCall, GenericArgument, ImplItemFn, ItemFn, ItemImpl, Pat,
    PathArguments, Stmt, Type,
};

#[derive(Clone)]
pub struct MethodSelection {
    pub module_name: String,
    pub impl_item: ItemImpl,
    pub impl_method: ImplItemFn,
    pub generic_vars: GenericVars,
    pub return_type: Option<TileRustType>,
}

#[derive(Clone, Default)]
pub struct TypeckResults {
    expr_types: HashMap<NodeId, TileRustType>,
    method_selections: HashMap<NodeId, MethodSelection>,
    lowered_method_calls: HashMap<NodeId, ExprMethodCall>,
}

impl TypeckResults {
    pub fn insert_expr_type(&mut self, expr: &Expr, ty: TileRustType) {
        let Some(id) = node_ids::expr_id(expr) else {
            return;
        };
        self.expr_types.insert(id, ty);
    }

    pub fn expr_type(&self, expr: &Expr) -> Option<&TileRustType> {
        self.expr_types.get(&node_ids::expr_id(expr)?)
    }

    pub fn insert_method_selection(
        &mut self,
        method_call: &ExprMethodCall,
        selection: MethodSelection,
    ) {
        self.insert_method_selection_for_expr(&Expr::MethodCall(method_call.clone()), selection);
    }

    pub fn insert_method_selection_for_expr(&mut self, expr: &Expr, selection: MethodSelection) {
        let Some(id) = node_ids::expr_id(expr) else {
            return;
        };
        self.method_selections.insert(id, selection);
    }

    pub fn method_selection(&self, method_call: &ExprMethodCall) -> Option<&MethodSelection> {
        self.method_selections
            .get(&node_ids::expr_id(&Expr::MethodCall(method_call.clone()))?)
    }

    pub fn insert_lowered_method_call(&mut self, call: &ExprCall, method_call: ExprMethodCall) {
        let Some(id) = node_ids::expr_id(&Expr::Call(call.clone())) else {
            return;
        };
        self.lowered_method_calls.insert(id, method_call);
    }

    pub fn lowered_method_call(&self, call: &ExprCall) -> Option<&ExprMethodCall> {
        self.lowered_method_calls
            .get(&node_ids::expr_id(&Expr::Call(call.clone()))?)
    }
}

pub fn infer_function(
    compiler: &CUDATileFunctionCompiler<'_>,
    fn_item: &ItemFn,
    generic_vars: &GenericVars,
    initial_types: HashMap<String, TileRustType>,
) -> Result<TypeckResults, JITError> {
    let mut cx = TypeInferenceCx {
        compiler,
        generic_vars,
        vars: initial_types,
        results: TypeckResults::default(),
    };
    cx.infer_block(&fn_item.block)?;
    Ok(cx.results)
}

struct TypeInferenceCx<'a, 'm> {
    compiler: &'a CUDATileFunctionCompiler<'m>,
    generic_vars: &'a GenericVars,
    vars: HashMap<String, TileRustType>,
    results: TypeckResults,
}

impl TypeInferenceCx<'_, '_> {
    fn infer_block(&mut self, block: &syn::Block) -> Result<Option<TileRustType>, JITError> {
        let mut last_type = None;
        for stmt in &block.stmts {
            last_type = self.infer_stmt(stmt)?;
        }
        Ok(last_type)
    }

    fn infer_stmt(&mut self, stmt: &Stmt) -> Result<Option<TileRustType>, JITError> {
        match stmt {
            Stmt::Local(local) => {
                let annotated_type = match &local.pat {
                    Pat::Type(pat_type) => self.compiler.compile_type(
                        &pat_type.ty,
                        self.generic_vars,
                        &HashMap::new(),
                    )?,
                    _ => None,
                };

                let init_type = if let Some(init) = &local.init {
                    self.infer_expr(&init.expr, annotated_type.clone())?
                } else {
                    None
                };

                let binding_type = annotated_type.or(init_type);
                if let Some(binding_type) = binding_type.clone() {
                    if let Some(name) = local_binding_name(&local.pat) {
                        self.vars.insert(name, binding_type);
                    }
                }
                Ok(binding_type)
            }
            Stmt::Expr(expr, _) => self.infer_expr(expr, None),
            Stmt::Item(_) | Stmt::Macro(_) => Ok(None),
        }
    }

    fn infer_expr(
        &mut self,
        expr: &Expr,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let inferred = match expr {
            Expr::Path(path) => {
                if path.path.segments.len() == 1 {
                    let name = get_ident_from_path_expr(path).to_string();
                    self.vars.get(&name).cloned()
                } else {
                    self.infer_zst_marker_type(expr)
                }
            }
            Expr::Reference(reference) => {
                let inner = self.infer_expr(&reference.expr, None)?;
                inner.map(|mut ty| {
                    ty.rust_ty = Type::Reference(syn::TypeReference {
                        and_token: reference.and_token,
                        lifetime: None,
                        mutability: reference.mutability,
                        elem: Box::new(ty.rust_ty.clone()),
                    });
                    ty
                })
            }
            Expr::Paren(paren) => self.infer_expr(&paren.expr, expected.clone())?,
            Expr::Block(block) => self.infer_block(&block.block)?,
            Expr::Tuple(tuple) => self.infer_tuple(tuple)?,
            Expr::Array(array) => {
                for elem in &array.elems {
                    let _ = self.infer_expr(elem, None)?;
                }
                expected
            }
            Expr::Field(field) => self.infer_field(field)?,
            Expr::Call(call) => self.infer_call(call, expected.clone())?,
            Expr::MethodCall(method_call) => {
                self.infer_method_call(method_call, expected.clone())?
            }
            Expr::Binary(binary) => {
                let lhs = self.infer_expr(&binary.left, None)?;
                let _ = self.infer_expr(&binary.right, lhs.clone())?;
                lhs.or(expected)
            }
            Expr::Unary(unary) => self.infer_expr(&unary.expr, expected.clone())?,
            Expr::Lit(_) => expected,
            Expr::If(if_expr) => {
                let _ = self.infer_expr(&if_expr.cond, None)?;
                let then_ty = self.infer_block(&if_expr.then_branch)?;
                if let Some((_, else_expr)) = &if_expr.else_branch {
                    let _ = self.infer_expr(else_expr, then_ty.clone())?;
                }
                then_ty.or(expected)
            }
            Expr::ForLoop(for_loop) => {
                let _ = self.infer_expr(&for_loop.expr, None)?;
                let mut nested = self.fork();
                let _ = nested.infer_block(&for_loop.body)?;
                self.results = nested.results;
                None
            }
            Expr::Unsafe(unsafe_expr) => self.infer_block(&unsafe_expr.block)?,
            _ => expected,
        };

        if let Some(ty) = inferred.clone() {
            self.results.insert_expr_type(expr, ty);
        }
        Ok(inferred)
    }

    fn fork(&self) -> TypeInferenceCx<'_, '_> {
        TypeInferenceCx {
            compiler: self.compiler,
            generic_vars: self.generic_vars,
            vars: self.vars.clone(),
            results: self.results.clone(),
        }
    }

    fn infer_zst_marker_type(&self, expr: &Expr) -> Option<TileRustType> {
        let Expr::Path(path_expr) = expr else {
            return None;
        };
        let path_ty = Type::Path(syn::TypePath {
            qself: None,
            path: path_expr.path.clone(),
        });
        let type_instance = TypeInstance::UserType(TypeInstanceUserType {
            maybe_generic_ty: path_ty,
        });
        Some(TileRustType::new_string(type_instance))
    }

    fn infer_tuple(&mut self, tuple: &syn::ExprTuple) -> Result<Option<TileRustType>, JITError> {
        let mut elem_types = Vec::new();
        for elem in &tuple.elems {
            if let Some(elem_ty) = self.infer_expr(elem, None)? {
                elem_types.push(elem_ty.rust_ty);
            }
        }
        if elem_types.len() != tuple.elems.len() {
            return Ok(None);
        }
        let tuple_ty = Type::Tuple(syn::TypeTuple {
            paren_token: syn::token::Paren::default(),
            elems: elem_types.into_iter().collect(),
        });
        self.compiler
            .compile_type(&tuple_ty, self.generic_vars, &HashMap::new())
    }

    fn infer_field(&mut self, field: &syn::ExprField) -> Result<Option<TileRustType>, JITError> {
        let Some(base_ty) = self.infer_expr(&field.base, None)? else {
            return Ok(None);
        };
        match (&base_ty.rust_ty, &field.member) {
            (Type::Tuple(tuple), syn::Member::Unnamed(index)) => {
                let Some(elem_ty) = tuple.elems.iter().nth(index.index as usize) else {
                    return Ok(None);
                };
                self.compiler
                    .compile_type(elem_ty, self.generic_vars, &HashMap::new())
            }
            _ => Ok(None),
        }
    }

    fn infer_call(
        &mut self,
        call: &ExprCall,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let Expr::Path(path) = &*call.func else {
            return Ok(expected);
        };
        let ident = get_ident_from_path_expr(path).to_string();
        if ident == "Some" && call.args.len() == 1 {
            return self.infer_expr(&call.args[0], expected);
        }

        let arg_types = self.infer_call_arg_types(call.args.iter())?;
        let Some((_, fn_item)) = self.compiler.modules.get_function_by_name(&ident) else {
            return Ok(expected);
        };

        if let Some(lowered_method_call) =
            crate::passes::typed_dispatch_lowering::lower_dispatch_wrapper_call(fn_item, call)
        {
            let method_arg_types = self.method_call_arg_types(&lowered_method_call)?;
            if let Some(selection) =
                self.select_method_call(&lowered_method_call, &method_arg_types)?
            {
                self.results
                    .insert_lowered_method_call(call, lowered_method_call.clone());
                self.results
                    .insert_method_selection_for_expr(&Expr::Call(call.clone()), selection.clone());
                return Ok(selection.return_type.or(expected));
            }
        }

        self.infer_function_return(fn_item, call, &arg_types, expected)
    }

    fn infer_method_call(
        &mut self,
        method_call: &ExprMethodCall,
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        let call_arg_types = self.method_call_arg_types(method_call)?;
        if let Some(selection) = self.select_method_call(method_call, &call_arg_types)? {
            self.results
                .insert_method_selection(method_call, selection.clone());
            Ok(selection.return_type.or(expected))
        } else {
            Ok(expected)
        }
    }

    fn infer_call_arg_types<'a>(
        &mut self,
        args: impl Iterator<Item = &'a Expr>,
    ) -> Result<Vec<TileRustType>, JITError> {
        let mut arg_types = Vec::new();
        for arg in args {
            let Some(arg_ty) = self.infer_expr(arg, None)? else {
                return Ok(Vec::new());
            };
            arg_types.push(arg_ty);
        }
        Ok(arg_types)
    }

    fn method_call_arg_types(
        &mut self,
        method_call: &ExprMethodCall,
    ) -> Result<Vec<TileRustType>, JITError> {
        let receiver = self.infer_expr(&method_call.receiver, None)?;
        let Some(receiver) = receiver else {
            return Ok(Vec::new());
        };
        let mut arg_types = vec![receiver];
        for arg in &method_call.args {
            let Some(arg_ty) = self.infer_expr(arg, None)? else {
                return Ok(Vec::new());
            };
            arg_types.push(arg_ty);
        }
        Ok(arg_types)
    }

    fn select_method_call(
        &self,
        method_call: &ExprMethodCall,
        call_arg_types: &[TileRustType],
    ) -> Result<Option<MethodSelection>, JITError> {
        if call_arg_types.is_empty() {
            return Ok(None);
        }
        let call_arg_rust_tys = call_arg_types
            .iter()
            .map(|arg| arg.rust_ty.clone())
            .collect::<Vec<_>>();
        let receiver_ty = &call_arg_rust_tys[0];
        let Some((module_name, impl_item, impl_method)) = self.compiler.modules.get_impl_item_fn(
            receiver_ty,
            method_call,
            self.generic_vars,
            &call_arg_rust_tys,
        )?
        else {
            return Ok(None);
        };
        let self_ty = &*impl_item.self_ty;
        let call_generic_vars = infer_method_generics(
            &impl_item,
            &impl_method,
            method_call,
            &call_arg_rust_tys,
            self_ty,
            self.generic_vars,
            self.compiler.modules.primitives(),
        )?;
        let return_type = self.infer_method_return_type(
            &impl_item,
            &impl_method,
            &call_arg_rust_tys,
            self_ty,
            &call_generic_vars,
            method_call,
        )?;
        Ok(Some(MethodSelection {
            module_name,
            impl_item,
            impl_method,
            generic_vars: call_generic_vars,
            return_type,
        }))
    }

    fn infer_method_return_type(
        &self,
        impl_item: &ItemImpl,
        impl_method: &ImplItemFn,
        call_arg_rust_tys: &[Type],
        self_ty: &Type,
        call_generic_vars: &GenericVars,
        method_call: &ExprMethodCall,
    ) -> Result<Option<TileRustType>, JITError> {
        let (arg_types, return_type) = get_sig_types(&impl_method.sig, Some(self_ty));
        if arg_types.iter().any(type_has_impl_trait) || type_has_impl_trait(&return_type) {
            return Ok(None);
        }
        let mut generic_arg_inf = GenericArgInference::new_method(impl_item, impl_method);
        generic_arg_inf.map_args_to_params(call_arg_rust_tys, Some(self_ty));
        generic_arg_inf.apply_provided_generics_method_call(method_call, self.generic_vars);
        if !generic_arg_inf.verify() {
            return Ok(None);
        }
        let call_output_type = generic_arg_inf.infer_type(&return_type, self.generic_vars);
        if !type_is_resolvable(self.compiler, &call_output_type, call_generic_vars) {
            return Ok(None);
        }
        match self
            .compiler
            .compile_type(&call_output_type, call_generic_vars, &HashMap::new())
        {
            Ok(ty) => Ok(ty),
            Err(_) => Ok(None),
        }
    }

    fn infer_function_return(
        &self,
        fn_item: &ItemFn,
        call: &ExprCall,
        arg_types: &[TileRustType],
        expected: Option<TileRustType>,
    ) -> Result<Option<TileRustType>, JITError> {
        if arg_types.is_empty() && !call.args.is_empty() {
            return Ok(expected);
        }
        let call_arg_rust_tys = arg_types
            .iter()
            .map(|arg| arg.rust_ty.clone())
            .collect::<Vec<_>>();
        let (fn_arg_types, return_type) = get_sig_types(&fn_item.sig, None);
        if fn_arg_types.iter().any(type_has_impl_trait) || type_has_impl_trait(&return_type) {
            return Ok(expected);
        }
        let mut generic_arg_inf = GenericArgInference::new_function(fn_item.sig.clone());
        generic_arg_inf.map_args_to_params(&call_arg_rust_tys, None);
        generic_arg_inf.apply_provided_generics_fn_call(call, self.generic_vars);
        if !generic_arg_inf.verify() {
            return Ok(expected);
        }
        let call_output_type = generic_arg_inf.infer_type(&return_type, self.generic_vars);
        if !type_is_resolvable(self.compiler, &call_output_type, self.generic_vars) {
            return Ok(expected);
        }
        let mut type_params = expected
            .as_ref()
            .map(type_params_by_name)
            .unwrap_or_default();

        if let Some(op_attrs) = self
            .compiler
            .modules
            .get_cuda_tile_op_attrs(fn_item.sig.ident.to_string().as_str())
        {
            if let Some(output_type_params) = op_attrs.parse_string_arr("output_type_params") {
                let param_names = get_sig_param_names(&fn_item.sig);
                let arg_types_by_name = param_names
                    .iter()
                    .zip(arg_types.iter())
                    .map(|(name, ty)| (name.clone(), ty.clone()))
                    .collect::<HashMap<_, _>>();
                for type_param_name in output_type_params {
                    if let Some(arg_type) = arg_types_by_name.get(&type_param_name) {
                        if should_skip_optional_output_type_param(
                            &type_param_name,
                            &arg_type.rust_ty,
                        ) {
                            continue;
                        }
                        let cuda_tile_type_str = arg_type.get_cuda_tile_type_str();
                        type_params.insert(
                            type_param_name.clone(),
                            TypeParam::derive_param_from_type(
                                type_param_name,
                                arg_type.rust_ty.clone(),
                                cuda_tile_type_str,
                                Some(arg_type.type_instance.clone()),
                            ),
                        );
                    }
                }
            }
        }

        match self
            .compiler
            .compile_type(&call_output_type, self.generic_vars, &type_params)
        {
            Ok(ty) => Ok(ty),
            Err(_) => Ok(expected),
        }
    }
}

fn local_binding_name(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Ident(ident) => Some(ident.ident.to_string()),
        Pat::Type(pat_type) => local_binding_name(&pat_type.pat),
        _ => None,
    }
}

fn type_params_by_name(ty: &TileRustType) -> HashMap<String, TypeParam> {
    ty.params
        .iter()
        .filter_map(|param| param.name().map(|name| (name.to_string(), param.clone())))
        .collect()
}

fn type_has_impl_trait(ty: &Type) -> bool {
    match ty {
        Type::ImplTrait(_) => true,
        Type::Reference(reference) => type_has_impl_trait(&reference.elem),
        Type::Tuple(tuple) => tuple.elems.iter().any(type_has_impl_trait),
        Type::Array(array) => type_has_impl_trait(&array.elem),
        Type::Ptr(ptr) => type_has_impl_trait(&ptr.elem),
        Type::Path(path) => path
            .path
            .segments
            .iter()
            .any(|segment| match &segment.arguments {
                PathArguments::AngleBracketed(args) => args.args.iter().any(|arg| match arg {
                    GenericArgument::Type(arg_ty) => type_has_impl_trait(arg_ty),
                    _ => false,
                }),
                _ => false,
            }),
        _ => false,
    }
}

fn type_is_resolvable(
    compiler: &CUDATileFunctionCompiler<'_>,
    ty: &Type,
    generic_vars: &GenericVars,
) -> bool {
    match ty {
        Type::Reference(reference) => type_is_resolvable(compiler, &reference.elem, generic_vars),
        Type::Tuple(tuple) => tuple
            .elems
            .iter()
            .all(|elem| type_is_resolvable(compiler, elem, generic_vars)),
        Type::Array(array) => {
            type_is_resolvable(compiler, &array.elem, generic_vars)
                && const_expr_is_resolvable(&array.len, generic_vars)
        }
        Type::Ptr(ptr) => type_is_resolvable(compiler, &ptr.elem, generic_vars),
        Type::Path(path) => {
            if path.qself.is_some() {
                return false;
            }
            let Some(segment) = path.path.segments.last() else {
                return false;
            };
            let ident = segment.ident.to_string();
            match &segment.arguments {
                PathArguments::None => {
                    generic_vars.var_type(&ident).is_some()
                        || compiler.modules.structs().contains_key(&ident)
                        || compiler
                            .modules
                            .primitives()
                            .contains_key(&("ElementType".to_string(), ident.clone()))
                        || compiler
                            .modules
                            .primitives()
                            .contains_key(&("Scalar".to_string(), ident.clone()))
                        || matches!(
                            ident.as_str(),
                            "i32" | "u32" | "i64" | "u64" | "f32" | "f64" | "bool"
                        )
                }
                PathArguments::AngleBracketed(args) => args.args.iter().all(|arg| match arg {
                    GenericArgument::Type(arg_ty) => {
                        type_is_resolvable(compiler, arg_ty, generic_vars)
                    }
                    GenericArgument::Const(expr) => const_expr_is_resolvable(expr, generic_vars),
                    GenericArgument::Lifetime(_) => true,
                    _ => false,
                }),
                _ => false,
            }
        }
        _ => true,
    }
}

fn const_expr_is_resolvable(expr: &Expr, generic_vars: &GenericVars) -> bool {
    match expr {
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path).to_string();
            generic_vars.var_type(&ident).is_some()
        }
        Expr::Block(block) => match block.block.stmts.as_slice() {
            [Stmt::Expr(inner, _)] => const_expr_is_resolvable(inner, generic_vars),
            _ => false,
        },
        Expr::Array(array) => array
            .elems
            .iter()
            .all(|elem| const_expr_is_resolvable(elem, generic_vars)),
        Expr::Repeat(repeat) => {
            const_expr_is_resolvable(&repeat.expr, generic_vars)
                && const_expr_is_resolvable(&repeat.len, generic_vars)
        }
        Expr::Lit(_) => true,
        Expr::Unary(unary) => const_expr_is_resolvable(&unary.expr, generic_vars),
        Expr::Paren(paren) => const_expr_is_resolvable(&paren.expr, generic_vars),
        _ => false,
    }
}

pub fn infer_method_generics(
    impl_item: &ItemImpl,
    impl_method: &ImplItemFn,
    method_call: &ExprMethodCall,
    call_arg_rust_tys: &[Type],
    self_ty: &Type,
    caller_generic_vars: &GenericVars,
    primitives: &HashMap<(String, String), ItemImpl>,
) -> Result<GenericVars, JITError> {
    let generic_arg_inference = GenericArgInference::new_method(impl_item, impl_method);
    if generic_arg_inference.param2arg.is_empty() {
        return GenericVars::empty(&impl_method.sig.generics);
    }

    let mut generic_arg_inference = GenericArgInference::new_method(impl_item, impl_method);
    generic_arg_inference.map_args_to_params(call_arg_rust_tys, Some(self_ty));
    let inferred = generic_arg_inference.get_generic_vars_instance(caller_generic_vars, primitives);

    if method_call.turbofish.is_some() {
        let passed = caller_generic_vars
            .from_expr_generic_args(&impl_method.sig.generics, &method_call.turbofish)?;
        inferred.merge(passed)
    } else {
        Ok(inferred)
    }
}

fn should_skip_optional_output_type_param(type_param_name: &str, rust_ty: &Type) -> bool {
    let Some(ident) = get_type_ident(rust_ty).map(|ident| ident.to_string()) else {
        return false;
    };
    matches!(
        (type_param_name, ident.as_str()),
        ("padding_value", "None") | ("dim_map", "Identity")
    )
}
