/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module registry: collects and indexes all parsed DSL modules, structs, trait impls,
//! and functions for lookup during compilation.

use crate::ast::{Module, SourceLocation, SpanBase};
use crate::error::{JITError, SpannedJITError};
use crate::generics::{GenericVars, TypeInstance};
use crate::syn_utils::*;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{
    ExprMethodCall, ImplItem, ImplItemFn, Item, ItemFn, ItemImpl, ItemMod, ItemStruct, Type,
};

/// Aggregated index of all DSL modules, types, impls, and functions available to the compiler.
pub struct CUDATileModules {
    pub(crate) modules: HashMap<String, ItemMod>,
    // Rust primitives marked as cuda tile types.
    // These are trait impls with the "cuda_tile::ty" annotation.
    pub(crate) primitives: HashMap<(String, String), ItemImpl>,
    // User-defined structs.
    // This also contains structs for cuda tile types.
    // They are structs with the "cuda_tile::ty" annotation.
    pub(crate) structs: HashMap<String, ItemStruct>,
    // User-defined struct impls.
    // This also contains impls for cuda tile types.
    pub(crate) struct_impls: HashMap<String, Vec<(String, ItemImpl)>>,
    // Internal trait impls. User-defined traits are not supported.
    pub(crate) trait_impls: HashMap<(String, String), (String, ItemImpl)>,
    // User-defined functions.
    // This also contains functions for cuda tile ops.
    // These are functions with the "cuda_tile::op" annotation.
    pub(crate) functions: HashMap<String, (String, ItemFn)>,
    // Span bases captured at proc macro expansion time.
    // Keyed by module name → SpanBase, which stores the (file, base_line,
    // base_col) anchor needed to convert any runtime syn span in that
    // module's AST into an absolute source location.
    pub(crate) span_bases: HashMap<String, SpanBase>,
}

impl CUDATileModules {
    pub fn new(modules_vec: Vec<Module>) -> Result<Self, JITError> {
        let mut modules: HashMap<String, ItemMod> = HashMap::new();
        let mut structs: HashMap<String, ItemStruct> = HashMap::new();
        let mut struct_impls: HashMap<String, Vec<(String, ItemImpl)>> = HashMap::new();
        let mut trait_impls: HashMap<(String, String), (String, ItemImpl)> = HashMap::new();
        let mut primitives: HashMap<(String, String), ItemImpl> = HashMap::new();
        let mut functions: HashMap<String, (String, ItemFn)> = HashMap::new();
        let mut span_bases: HashMap<String, SpanBase> = HashMap::new();

        for module in &modules_vec {
            let module_ast = module.ast();
            // println!("module_ast: {:#?}", module_ast);
            let module_name = module.name().to_string();
            match &module_ast.content {
                Some(content) => {
                    for item in &content.1 {
                        match item {
                            Item::Struct(struct_item) => {
                                let struct_name = struct_item.ident.to_string();
                                structs.insert(struct_name.clone(), struct_item.clone());
                            }
                            Item::Fn(function_item) => {
                                let fn_name = function_item.sig.ident.to_string();
                                if functions
                                    .insert(
                                        fn_name.clone(),
                                        (module_name.clone(), function_item.clone()),
                                    )
                                    .is_some()
                                {
                                    return Err(JITError::generic_err(
                                        format!("duplicate functions are not supported; try renaming your function: {fn_name}").as_str()
                                    ));
                                };
                            }
                            Item::Trait(_trait_item) => {
                                // TODO (hme): Do we need to collect variadic traits?
                                //  The impl contains all the information we need.
                            }
                            Item::Impl(impl_item) => {
                                let self_ident_str = get_type_str(&impl_item.self_ty);
                                let trait_ident_str = match &impl_item.trait_ {
                                    Some((_, trait_path, _)) => {
                                        let last_seg = trait_path.segments.last().unwrap();
                                        Some(last_seg.ident.to_string())
                                    }
                                    None => None,
                                };
                                // This may be an impl for types with trait bound ElementType.
                                match (self_ident_str, trait_ident_str) {
                                    (Some(self_ident_str), Some(trait_ident_str)) => {
                                        if let Some(_attribute_list) =
                                            get_meta_list("cuda_tile :: ty", &impl_item.attrs)
                                        {
                                            // println!("primitive type trait impl: {trait_ident_str} for {}", self_ident_str);
                                            // An impl with a type annotation and self ident is a Rust type tagged for compilation to cuda tile.
                                            if primitives
                                                .insert(
                                                    (
                                                        trait_ident_str.clone(),
                                                        self_ident_str.clone(),
                                                    ),
                                                    impl_item.clone(),
                                                )
                                                .is_some()
                                            {
                                                return module.resolve_span(&impl_item.span())
                                                    .jit_error_result(&format!(
                                                        "duplicate primitive type trait impl: `{trait_ident_str}` for `{self_ident_str}`"
                                                    ));
                                            }
                                        } else if let Some(_attribute_list) = get_meta_list(
                                            "cuda_tile :: variadic_trait_impl",
                                            &impl_item.attrs,
                                        ) {
                                            if trait_impls
                                                .insert(
                                                    (
                                                        trait_ident_str.clone(),
                                                        self_ident_str.clone(),
                                                    ),
                                                    (module_name.clone(), impl_item.clone()),
                                                )
                                                .is_some()
                                            {
                                                return module.resolve_span(&impl_item.span())
                                                    .jit_error_result(&format!(
                                                        "duplicate trait impl: `{trait_ident_str}` for `{self_ident_str}`"
                                                    ));
                                            }
                                        }
                                    }
                                    (Some(self_ident_str), None) => {
                                        // println!("struct impl: {self_ident_str}");
                                        if !struct_impls.contains_key(self_ident_str.as_str()) {
                                            struct_impls.insert(
                                                self_ident_str.clone(),
                                                vec![(module_name.clone(), impl_item.clone())],
                                            );
                                        } else {
                                            struct_impls
                                                .get_mut(&self_ident_str)
                                                .unwrap()
                                                .push((module_name.clone(), impl_item.clone()));
                                        }
                                    }
                                    (None, Some(trait_ident_str)) => {
                                        return module
                                            .resolve_span(&impl_item.span())
                                            .jit_error_result(&format!(
                                            "impl block for trait `{trait_ident_str}` is missing a Self type"
                                        ));
                                    }
                                    (None, None) => {
                                        return module
                                            .resolve_span(&impl_item.span())
                                            .jit_error_result(
                                            "impl block is missing both a Self type and a trait",
                                        );
                                    }
                                }
                            }
                            // Unsupported items for user-defined modules are rejected by the macro.
                            _ => continue,
                        }
                    }
                }
                None => {
                    return module
                        .resolve_span(&module_ast.span())
                        .jit_error_result(&format!(
                            "module `{module_name}` must have a body (non-empty content)"
                        ));
                }
            }
            modules.insert(module_name.clone(), module_ast.clone());
            span_bases.insert(module_name, module.span_base().clone());
        }
        Ok(CUDATileModules {
            modules,
            primitives,
            structs,
            struct_impls,
            trait_impls,
            functions,
            span_bases,
        })
    }

    /// Get the [`SpanBase`] for a module, if one was captured.
    pub fn get_span_base(&self, module_name: &str) -> Option<&SpanBase> {
        self.span_bases.get(module_name)
    }

    /// Resolve any `proc_macro2::Span` from the given module's AST to an
    /// absolute [`SourceLocation`].
    ///
    /// The span's line/column are string-relative (produced by
    /// `syn::parse_str` on the verbatim source text).  The module's
    /// [`SpanBase`] supplies the file path and base offset so that:
    ///
    /// ```text
    /// abs_line = base_line + (span_line - 1)
    /// abs_col  = if span_line == 1 { base_col + span_col } else { span_col }
    /// ```
    pub fn resolve_span(&self, module_name: &str, span: &proc_macro2::Span) -> SourceLocation {
        match self.span_bases.get(module_name) {
            Some(base) => base.resolve_span(span),
            None => SourceLocation::unknown(),
        }
    }

    /// Get the source file path for a module, if available.
    pub fn get_source_file(&self, module_name: &str) -> Option<&str> {
        self.span_bases.get(module_name).and_then(|sb| {
            if sb.file.is_empty() {
                None
            } else {
                Some(sb.file.as_str())
            }
        })
    }

    pub fn get_primitives_attrs(
        &self,
        trait_name: &str,
        rust_type_name: &str,
    ) -> Option<SingleMetaList> {
        match self
            .primitives
            .get(&(trait_name.to_string(), rust_type_name.to_string()))
        {
            Some(item_impl) => get_meta_list("cuda_tile :: ty", &item_impl.attrs),
            None => None,
        }
    }

    pub fn get_cuda_tile_type_attrs(&self, ident: &str) -> Option<SingleMetaList> {
        // TODO (hme): This is slow but flexible.
        match self.structs.get(ident) {
            Some(item_struct) => get_meta_list("cuda_tile :: ty", &item_struct.attrs),
            None => None,
        }
    }

    pub fn get_cuda_tile_op_attrs(&self, ident: &str) -> Option<SingleMetaList> {
        // TODO (hme): This is slow but flexible.
        match self.functions.get(ident) {
            Some((_, item_fn)) => get_meta_list("cuda_tile :: op", &item_fn.attrs),
            None => None,
        }
    }

    pub fn get_fn_item(
        &self,
        module_name: &str,
        function_name: &str,
    ) -> Result<&(String, ItemFn), JITError> {
        if !self.modules.contains_key(module_name) {
            return JITError::generic(&format!("undefined module: `{module_name}`"));
        }
        match self.functions.get(function_name) {
            Some(function) => Ok(function),
            None => JITError::generic(&format!("undefined function: `{function_name}`")),
        }
    }

    pub fn get_fn_entry_attrs(&self, fn_item: &ItemFn) -> Result<SingleMetaList, JITError> {
        let entry_attrs = get_meta_list_by_last_segment("entry", &fn_item.attrs);
        let Some(entry_attrs) = entry_attrs else {
            return JITError::generic("function is missing a required `#[entry(...)]` attribute");
        };
        Ok(entry_attrs)
    }

    pub fn get_entry_arg_bool_by_function_name(
        &self,
        module_name: &str,
        function_name: &str,
        name: &str,
    ) -> Result<bool, JITError> {
        let (_, fn_item) = self.get_fn_item(module_name, function_name)?;
        let entry_attrs = self.get_fn_entry_attrs(fn_item)?;
        Ok(entry_attrs.parse_bool(name).unwrap_or(false))
    }

    pub fn get_entry_arg_string_by_function_name(
        &self,
        module_name: &str,
        function_name: &str,
        name: &str,
    ) -> Result<Option<String>, JITError> {
        let (_, fn_item) = self.get_fn_item(module_name, function_name)?;
        let entry_attrs = self.get_fn_entry_attrs(fn_item)?;
        Ok(entry_attrs.parse_string(name))
    }

    pub fn get_impl_item_fn(
        &self,
        receiver_rust_ty: &syn::Type,
        method_call_expr: &ExprMethodCall,
        generic_vars: &GenericVars,
        // String is module_name.
    ) -> Result<Option<(String, ItemImpl, ImplItemFn)>, crate::error::JITError> {
        // Check if we're calling a method on a primitive type trait impl.
        let impls = match generic_vars.instantiate_type(receiver_rust_ty, &self.primitives)? {
            TypeInstance::ElementType(_elem_ty) => {
                #[allow(clippy::manual_map)]
                match self
                    .trait_impls
                    .get(&("BroadcastScalar".to_string(), "E".to_string()))
                {
                    Some(trait_impl) => Some(&vec![trait_impl.clone()]),
                    None => None,
                }
            }
            _ => {
                let ident = get_type_ident(receiver_rust_ty);
                if ident.is_none() {
                    return Ok(None);
                }
                let receiver_type_str = ident.unwrap().to_string();
                self.struct_impls.get(&receiver_type_str)
            }
        };
        let impls_vec = impls.unwrap();
        let method_name = method_call_expr.method.to_string();
        for (module_name, item_impl) in impls_vec {
            for item in &item_impl.items {
                match item {
                    ImplItem::Fn(impl_item_fn) => {
                        let impl_item_fn_name = impl_item_fn.sig.ident.to_string();
                        if method_name == impl_item_fn_name {
                            return Ok(Some((
                                module_name.clone(),
                                item_impl.clone(),
                                impl_item_fn.clone(),
                            )));
                        }
                    }
                    _ => continue,
                }
            }
        }
        Ok(None)
    }

    pub fn get_struct_field_type(&self, struct_name: &str, field_name: &str) -> Option<Type> {
        let s = self
            .structs
            .get(struct_name)
            .unwrap_or_else(|| panic!("{struct_name} doesn't exist."));
        for field in &s.fields {
            let Some(curr_field_ident) = &field.ident else {
                continue;
            };
            if field_name == curr_field_ident.to_string().as_str() {
                return Some(field.ty.clone());
            }
        }
        None
    }
}
