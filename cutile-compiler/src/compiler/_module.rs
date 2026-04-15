/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module registry: thin wrapper around [`NameResolver`] that preserves
//! the compiler's existing lookup API.
//!
//! All item storage and name resolution lives in the [`NameResolver`]
//! (Pass 1 output). This struct provides backward-compatible accessors
//! so the rest of the compiler doesn't need to change yet.

use crate::ast::{Module, SourceLocation, SpanBase};
use crate::error::{JITError, SpannedJITError};
use crate::generics::{GenericVars, TypeInstance};
use crate::kernel_naming::KernelNaming;
use crate::passes::name_resolution::NameResolver;
use crate::syn_utils::*;
use std::collections::HashMap;
use syn::spanned::Spanned;
use syn::{ExprMethodCall, ImplItem, ImplItemFn, ItemFn, ItemImpl, ItemMod, ItemStruct, Type};

/// Aggregated index of all DSL modules, types, impls, and functions.
///
/// Internally delegates to [`NameResolver`] for all item storage and
/// name resolution. The flat accessors (`primitives`, `structs`, etc.)
/// are backward-compatible shims that will be removed as the compiler
/// migrates to path-based resolution.
pub struct CUDATileModules {
    /// The name resolver (Pass 1 output). Owns all items and resolves paths.
    pub(crate) name_resolver: NameResolver,

    /// Span bases for source location mapping.
    pub(crate) span_bases: HashMap<String, SpanBase>,
}

// ---------------------------------------------------------------------------
// Backward-compatible field-like accessors
// ---------------------------------------------------------------------------

impl CUDATileModules {
    /// Flat map of all modules: name → ItemMod.
    /// Compatibility shim — prefer `name_resolver.module(name)`.
    pub fn modules(&self) -> &HashMap<String, ItemMod> {
        self.name_resolver.all_modules()
    }

    /// Flat map of all primitives: (trait, type) → ItemImpl.
    /// Compatibility shim — passed to generics/types functions.
    pub fn primitives(&self) -> &HashMap<(String, String), ItemImpl> {
        self.name_resolver.primitives()
    }

    /// Flat map of all structs: name → ItemStruct.
    /// Compatibility shim.
    pub fn structs(&self) -> &HashMap<String, ItemStruct> {
        self.name_resolver.structs()
    }

    /// Flat map of all struct impls: struct_name → [(module, ItemImpl)].
    /// Compatibility shim.
    pub fn struct_impls(&self) -> &HashMap<String, Vec<(String, ItemImpl)>> {
        self.name_resolver.struct_impls()
    }

    /// Flat map of all functions: name → (module, ItemFn).
    /// Compatibility shim.
    pub fn functions(&self) -> &HashMap<String, (String, ItemFn)> {
        self.name_resolver.functions()
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl CUDATileModules {
    pub fn new(modules_vec: Vec<Module>) -> Result<Self, JITError> {
        // Collect module ASTs and span bases.
        let mut module_asts: Vec<(String, ItemMod)> = Vec::new();
        let mut span_bases: HashMap<String, SpanBase> = HashMap::new();

        for module in &modules_vec {
            let module_name = module.name().to_string();
            let module_ast = module.ast();

            if module_ast.content.is_none() {
                return module
                    .resolve_span(&module_ast.span())
                    .jit_error_result(&format!(
                        "module `{module_name}` must have a body (non-empty content)"
                    ));
            }

            module_asts.push((module_name.clone(), module_ast.clone()));
            span_bases.insert(module_name, module.span_base().clone());
        }

        // Pass 1: Build the name resolver (indexes all items, processes imports).
        let name_resolver = NameResolver::build(&module_asts)?;

        Ok(CUDATileModules {
            name_resolver,
            span_bases,
        })
    }
}

// ---------------------------------------------------------------------------
// Span / location utilities
// ---------------------------------------------------------------------------

impl CUDATileModules {
    pub fn get_span_base(&self, module_name: &str) -> Option<&SpanBase> {
        self.span_bases.get(module_name)
    }

    pub fn resolve_span(&self, module_name: &str, span: &proc_macro2::Span) -> SourceLocation {
        match self.span_bases.get(module_name) {
            Some(base) => base.resolve_span(span),
            None => SourceLocation::unknown(),
        }
    }

    pub fn get_source_file(&self, module_name: &str) -> Option<&str> {
        self.span_bases.get(module_name).and_then(|sb| {
            if sb.file.is_empty() {
                None
            } else {
                Some(sb.file.as_str())
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Item lookup (backward-compatible methods)
// ---------------------------------------------------------------------------

impl CUDATileModules {
    pub fn get_primitives_attrs(
        &self,
        trait_name: &str,
        rust_type_name: &str,
    ) -> Option<SingleMetaList> {
        self.name_resolver
            .get_primitive_attrs(trait_name, rust_type_name)
    }

    pub fn get_cuda_tile_type_attrs(&self, ident: &str) -> Option<SingleMetaList> {
        self.name_resolver.get_type_attrs(ident)
    }

    pub fn get_function_by_name(&self, function_name: &str) -> Option<&(String, ItemFn)> {
        let canonical_name = KernelNaming::canonical_public_name(function_name);
        self.name_resolver.functions().get(canonical_name.as_str())
    }

    pub fn get_cuda_tile_op_attrs(&self, ident: &str) -> Option<SingleMetaList> {
        self.name_resolver.get_op_attrs(ident)
    }

    pub fn get_fn_item(
        &self,
        module_name: &str,
        function_name: &str,
    ) -> Result<&(String, ItemFn), JITError> {
        if !self.name_resolver.has_module(module_name) {
            return JITError::generic(&format!("undefined module: `{module_name}`"));
        }
        match self.get_function_by_name(function_name) {
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
    ) -> Result<Option<(String, ItemImpl, ImplItemFn)>, JITError> {
        // Check if we're calling a method on a primitive type trait impl.
        let impls = match generic_vars.instantiate_type(receiver_rust_ty, self.primitives())? {
            TypeInstance::ElementType(_elem_ty) => {
                match self
                    .name_resolver
                    .trait_impls()
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
                self.name_resolver.struct_impls().get(&receiver_type_str)
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
        self.name_resolver
            .get_struct_field_type(struct_name, field_name)
    }
}
