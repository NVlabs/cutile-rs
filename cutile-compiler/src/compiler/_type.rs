/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type representations for the CUDA Tile compiler.
//! Maps Rust types to their CUDA Tile MLIR equivalents.

use crate::compiler::utils::ElementTypePrefix;
use crate::error::JITError;
use crate::generics::{GenericVars, TypeInstance};
use crate::types::*;
use melior::ir;
use melior::Context;
use quote::ToTokens;
use std::collections::HashMap;
use syn::ItemImpl;

/// Classification of a Rust type within the CUDA Tile type system.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Kind {
    // A scalar type. The corresponding Rust type will be something like f32.
    /// Scalar element type (e.g. `f32`, `i32`).
    PrimitiveType,
    // A structured type. The corresponding Rust type will be something like Tile<f32, {[]}>
    /// Shaped type with element and dimensions (e.g. `Tile<f32, {[128, 64]}>`).
    StructuredType,
    // A compound type. This may be a tuple (i32, i32, ...) or array [i32; 2].
    // These don't compile to cuda tile.
    /// Tuple or array compound type; not compiled to CUDA Tile IR.
    Compound,
    // A structure type. These don't compile to cuda tile.
    /// User-defined struct; not compiled to CUDA Tile IR.
    Struct,
    // A string. Non-numeric types require special handling.
    /// String literal type; requires special handling.
    String,
}

/// A compiled type binding a Rust `syn::Type` to its CUDA Tile MLIR type and metadata.
#[derive(Debug, Clone)]
pub struct TileRustType<'c> {
    pub(crate) kind: Kind,
    pub(crate) cuda_tile_name: Option<String>,
    pub(crate) cuda_tile_ty: Option<ir::Type<'c>>,
    pub(crate) params: Vec<TypeParam>,
    // TODO (hme): Remove rust_ty and use get_source_rust_ty().
    pub(crate) rust_ty: syn::Type,
    pub(crate) type_instance: TypeInstance,
}

impl<'c, 'a> TileRustType<'c> {
    pub fn clone_fresh(&'c self, context: &'a Context) -> TileRustType<'a> {
        let cuda_tile_ty = ir::Type::parse(
            context,
            &self.cuda_tile_ty.expect("copy_fresh failed.").to_string(),
        );
        TileRustType {
            cuda_tile_ty,
            cuda_tile_name: self.cuda_tile_name.clone(),
            params: self.params.clone(),
            rust_ty: self.rust_ty.clone(),
            type_instance: self.type_instance.clone(),
            kind: self.kind.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get_instantiated_rust_ty(&self) -> &syn::Type {
        self.type_instance.get_instantiated_type()
    }
    #[allow(dead_code)]
    pub(crate) fn get_source_rust_ty(&self) -> &syn::Type {
        self.type_instance.get_source_type()
    }
    pub(crate) fn get_instantiated_rust_element_type(
        &self,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<String> {
        self.type_instance.get_rust_element_instance_ty()
    }
    pub(crate) fn get_cuda_tile_element_type(
        &self,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<Option<String>, JITError> {
        let type_instance = self.type_instance.get_instantiated_type();
        match self.kind {
            Kind::PrimitiveType => Ok(Some(get_cuda_tile_element_type_primitive(
                type_instance,
                primitives,
            ))),
            Kind::StructuredType => Ok(get_cuda_tile_element_type_structured(
                type_instance,
                primitives,
            )),
            _ => JITError::generic(&format!(
                "cannot get element type for {:?} values",
                self.kind
            )),
        }
    }
    pub(crate) fn get_cuda_tile_element_type_prefix(
        &self,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<ElementTypePrefix, JITError> {
        let cuda_elem_ty_str = self
            .get_cuda_tile_element_type(primitives)?
            .ok_or_else(|| {
                JITError::generic_err(&format!(
                    "unable to determine element type for `{}`",
                    self.rust_ty.to_token_stream()
                ))
            })?;
        ElementTypePrefix::new(&cuda_elem_ty_str)
    }
    pub(crate) fn get_cuda_tile_type_str(&self) -> Option<String> {
        self.cuda_tile_ty.map(|ty| ty.to_string())
    }
    pub(crate) fn new_primitive_type(
        context: &'c Context,
        cuda_tile_name: String,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
        mut params: Vec<TypeParam>,
        type_instance: TypeInstance,
    ) -> Result<TileRustType<'c>, JITError> {
        let rust_ty = type_instance.get_source_type().clone();
        let type_param_str = params
            .iter_mut()
            .map(|tp| tp.instantiate(generic_vars, primitives))
            .collect::<Result<Vec<_>, _>>()?
            .join(",");
        let type_str = format!("{}<{}>", cuda_tile_name, type_param_str);
        let ty = match ir::Type::parse(context, type_str.as_str()) {
            Some(ty) => ty,
            None => {
                return JITError::generic(&format!(
                    "failed to compile type `{}` (resolved to `{}`)",
                    rust_ty.to_token_stream(),
                    type_str
                ))
            }
        };
        Ok(TileRustType {
            cuda_tile_ty: Some(ty),
            cuda_tile_name: Some(cuda_tile_name),
            params,
            rust_ty,
            type_instance,
            kind: Kind::PrimitiveType,
        })
    }
    pub(crate) fn new_structured_type(
        context: &'c Context,
        cuda_tile_name: String,
        generic_args: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
        mut params: Vec<TypeParam>,
        type_instance: TypeInstance,
    ) -> Result<TileRustType<'c>, JITError> {
        let rust_ty = type_instance.get_source_type().clone();
        let type_str = if params.is_empty() {
            cuda_tile_name.to_string()
        } else {
            let type_param_str = params
                .iter_mut()
                .map(|tp| tp.instantiate(generic_args, primitives))
                .collect::<Result<Vec<_>, _>>()?
                .join(",");
            format!("{}<{}>", cuda_tile_name, type_param_str)
        };
        // println!("CUDATileType::new {:?} \n type str: {:?}", cuda_tile_name, type_str);
        let ty = match ir::Type::parse(context, type_str.as_str()) {
            Some(ty) => ty,
            None => {
                return JITError::generic(&format!(
                    "failed to compile type `{}` (resolved to `{}`)",
                    rust_ty.to_token_stream(),
                    type_str
                ))
            }
        };
        Ok(TileRustType {
            cuda_tile_ty: Some(ty),
            cuda_tile_name: Some(cuda_tile_name),
            params,
            rust_ty,
            type_instance,
            kind: Kind::StructuredType,
        })
    }
    pub(crate) fn new_structure(
        _context: &'c Context,
        cuda_tile_name: String,
        type_instance: TypeInstance,
    ) -> TileRustType<'c> {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty: None,
            cuda_tile_name: Some(cuda_tile_name),
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::Struct,
        }
    }
    pub(crate) fn new_compound(
        _context: &'c Context,
        type_instance: TypeInstance,
    ) -> TileRustType<'c> {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty: None,
            cuda_tile_name: None,
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::Compound,
        }
    }
    pub(crate) fn new_string(
        _context: &'c Context,
        type_instance: TypeInstance,
    ) -> TileRustType<'c> {
        let rust_ty = type_instance.get_source_type().clone();
        TileRustType {
            cuda_tile_ty: None,
            cuda_tile_name: None,
            params: vec![],
            rust_ty,
            type_instance,
            kind: Kind::String,
        }
    }
}
