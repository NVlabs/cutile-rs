/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiled value representation for compiler2.
//!
//! Uses `cutile_ir::ir::Value` (Copy, index-based, no lifetimes).

use super::shared_types::Kind;
use super::tile_rust_type::TileRustType;
use crate::bounds::Bounds;
use crate::error::JITError;
use crate::syn_utils::get_type_ident;
use cutile_ir::ir::Value;
use std::collections::BTreeMap;
use syn::Expr;

// Re-export shared types.
pub use super::shared_types::{BlockTerminator, Mutability};

/// Flattens all values in a `BTreeMap` of [`TileRustValue`]s into a linear list.
pub fn unpack_btree_to(
    btree: &BTreeMap<String, TileRustValue>,
    values: &mut Vec<Value>,
) -> Result<(), JITError> {
    for key in btree.keys() {
        let value = btree[key].clone();
        value.unpack_to(values)?;
    }
    Ok(())
}

/// Reconstructs a `BTreeMap` of [`TileRustValue`]s from a flat list of values.
pub fn repack_btree_from(
    old_btree: &BTreeMap<String, TileRustValue>,
    values: &Vec<Value>,
    mut pos: usize,
) -> Result<(BTreeMap<String, TileRustValue>, usize), JITError> {
    let mut new_btree = BTreeMap::new();
    for key in old_btree.keys() {
        let value = old_btree[key].clone();
        let res = value.repack_from(values, pos)?;
        new_btree.insert(key.to_string(), res.0);
        pos = res.1;
    }
    Ok((new_btree, pos))
}

/// Type-level metadata (named sub-fields) attached to structured values like views.
#[derive(Debug, Clone)]
pub struct TypeMeta {
    pub fields: BTreeMap<String, TileRustValue>,
}

impl TypeMeta {
    fn unpack_to(&self, values: &mut Vec<Value>) -> Result<(), JITError> {
        unpack_btree_to(&self.fields, values)
    }
    fn repack_from(&self, values: &Vec<Value>, pos: usize) -> Result<(TypeMeta, usize), JITError> {
        let res = repack_btree_from(&self.fields, values, pos)?;
        Ok((TypeMeta { fields: res.0 }, res.1))
    }
}

/// A compiled value: wraps a `cutile_ir::ir::Value` together with its Rust type, kind, bounds,
/// and metadata.
///
/// Port of old `TileRustValue<'c, 'a>` — all lifetime parameters are gone because
/// `cutile_ir::ir::Value` is `Copy` (a u32 arena index) and `TileRustType` is owned.
#[derive(Debug, Clone)]
pub struct TileRustValue {
    pub(crate) kind: Kind,
    pub(crate) fields: Option<BTreeMap<String, TileRustValue>>,
    pub(crate) values: Option<Vec<TileRustValue>>,
    pub(crate) value: Option<Value>,
    pub(crate) ty: TileRustType,
    pub(crate) type_meta: Option<TypeMeta>,
    pub(crate) mutability: Mutability,
    pub(crate) bounds: Option<Bounds<i64>>,
    pub(crate) string_literal: Option<syn::Expr>,
}

impl TileRustValue {
    pub fn new_struct(fields: BTreeMap<String, TileRustValue>, ty: TileRustType) -> TileRustValue {
        Self {
            fields: Some(fields),
            values: None,
            value: None,
            ty,
            kind: Kind::Struct,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
        }
    }

    pub fn new_compound(values: Vec<TileRustValue>, ty: TileRustType) -> TileRustValue {
        Self {
            fields: None,
            values: Some(values),
            value: None,
            ty,
            kind: Kind::Compound,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
        }
    }

    pub fn new_structured_type(
        value: Value,
        ty: TileRustType,
        type_meta: Option<TypeMeta>,
    ) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: Some(value),
            ty,
            kind: Kind::StructuredType,
            type_meta,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: None,
        }
    }

    pub fn new_primitive(
        value: Value,
        ty: TileRustType,
        bounds: Option<Bounds<i64>>,
    ) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: Some(value),
            ty,
            kind: Kind::PrimitiveType,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds,
            string_literal: None,
        }
    }

    pub fn new_string(string_literal: Expr, ty: TileRustType) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: None,
            ty,
            kind: Kind::String,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: Some(string_literal),
        }
    }

    pub fn new_value_kind_like(value: Value, ty: TileRustType) -> TileRustValue {
        let kind = ty.kind.clone();
        match kind {
            Kind::StructuredType => Self::new_structured_type(
                value,
                ty,
                Some(TypeMeta {
                    fields: BTreeMap::new(),
                }),
            ),
            _ => Self {
                fields: None,
                values: None,
                value: Some(value),
                ty,
                kind,
                type_meta: None,
                mutability: Mutability::Unset,
                bounds: None,
                string_literal: None,
            },
        }
    }

    pub fn new_literal(literal_expr: syn::Expr, ty: TileRustType) -> TileRustValue {
        Self {
            fields: None,
            values: None,
            value: None,
            ty,
            kind: Kind::PrimitiveType,
            type_meta: None,
            mutability: Mutability::Unset,
            bounds: None,
            string_literal: Some(literal_expr),
        }
    }

    pub fn verify(&self) -> Result<(), JITError> {
        match self.kind {
            Kind::String => {
                if !(self.string_literal.is_some()
                    && self.type_meta.is_none()
                    && self.value.is_none()
                    && self.values.is_none()
                    && self.fields.is_none())
                {
                    return JITError::generic("internal: string value has inconsistent fields set");
                }
            }
            Kind::PrimitiveType => {
                if !(self.value.is_some() && self.values.is_none() && self.fields.is_none()) {
                    return JITError::generic(
                        "internal: primitive value has inconsistent fields set",
                    );
                }
            }
            Kind::StructuredType => {
                if !(self.value.is_some() && self.values.is_none() && self.fields.is_none()) {
                    return JITError::generic(
                        "internal: structured type value has inconsistent fields set",
                    );
                }
            }
            Kind::Compound => {
                if !(self.value.is_none() && self.values.is_some() && self.fields.is_none()) {
                    return JITError::generic(
                        "internal: compound value has inconsistent fields set",
                    );
                }
            }
            Kind::Struct => {
                if !(self.value.is_none() && self.values.is_none() && self.fields.is_some()) {
                    return JITError::generic("internal: struct value has inconsistent fields set");
                }
            }
        }
        Ok(())
    }

    pub fn get_type_meta_field(&self, name: &str) -> Option<&Self> {
        let Some(type_meta) = &self.type_meta else {
            return None;
        };
        type_meta.fields.get(name)
    }

    pub fn take_type_meta_field(self, name: &str) -> Option<Self> {
        let mut type_meta = self.type_meta?;
        type_meta.fields.remove(name)
    }

    pub fn insert_type_meta_field(
        &mut self,
        name: &str,
        val: TileRustValue,
    ) -> Result<(), JITError> {
        let Some(type_meta) = &mut self.type_meta else {
            return JITError::generic(&format!(
                "type metadata not supported for {:?} values",
                self.ty.kind
            ));
        };
        type_meta.fields.insert(name.to_string(), val.clone());
        Ok(())
    }

    pub fn get_token(&self) -> Option<&Self> {
        self.get_type_meta_field("token")
    }

    pub fn is_tile(&self) -> bool {
        let Some(ident) = get_type_ident(&self.ty.rust_ty) else {
            return false;
        };
        ident.to_string().starts_with("Tile")
    }

    pub fn is_partition(&self) -> bool {
        let Some(ident) = get_type_ident(&self.ty.rust_ty) else {
            return false;
        };
        ident.to_string().starts_with("Partition")
    }

    pub fn unpack_to(&self, values: &mut Vec<Value>) -> Result<(), JITError> {
        self.verify()?;
        match self.kind {
            Kind::String => {}
            Kind::PrimitiveType => {
                values.push(self.value.unwrap());
                if let Some(old_type_meta) = &self.type_meta {
                    old_type_meta.unpack_to(values)?;
                }
            }
            Kind::StructuredType => {
                values.push(self.value.unwrap());
                if let Some(old_type_meta) = &self.type_meta {
                    old_type_meta.unpack_to(values)?;
                }
            }
            Kind::Compound => {
                let Some(self_values) = &self.values else {
                    return JITError::generic("internal: compound value missing its element list");
                };
                for value in self_values {
                    value.unpack_to(values)?;
                }
            }
            Kind::Struct => {
                let Some(fields) = &self.fields else {
                    return JITError::generic("internal: struct value missing its fields");
                };
                unpack_btree_to(fields, values)?;
            }
        }
        Ok(())
    }

    pub fn repack_from(
        &self,
        values: &Vec<Value>,
        mut pos: usize,
    ) -> Result<(Self, usize), JITError> {
        self.verify()?;
        let mut result = self.clone();
        match self.kind {
            Kind::String => {}
            Kind::PrimitiveType => {
                result.value = Some(values[pos]);
                pos += 1;
                if let Some(old_type_meta) = result.type_meta {
                    let res = old_type_meta.repack_from(values, pos)?;
                    result.type_meta = Some(res.0);
                    pos = res.1;
                }
            }
            Kind::StructuredType => {
                result.value = Some(values[pos]);
                pos += 1;
                if let Some(old_type_meta) = result.type_meta {
                    let res = old_type_meta.repack_from(values, pos)?;
                    result.type_meta = Some(res.0);
                    pos = res.1;
                }
            }
            Kind::Compound => {
                let Some(self_values) = &result.values else {
                    return JITError::generic("internal: compound value missing its element list");
                };
                let mut result_values = vec![];
                for value in self_values {
                    let res = value.repack_from(values, pos)?;
                    result_values.push(res.0);
                    pos = res.1;
                }
                result.values = Some(result_values);
            }
            Kind::Struct => {
                let Some(fields) = &result.fields else {
                    return JITError::generic("internal: struct value missing its fields");
                };
                let res = repack_btree_from(fields, values, pos)?;
                result.fields = Some(res.0);
                pos = res.1;
            }
        }
        result.verify()?;
        Ok((result, pos))
    }
}

/// Variable scope and control-flow state for a compilation block.
#[derive(Debug, Clone)]
pub struct CompilerContext {
    pub vars: BTreeMap<String, TileRustValue>,
    pub carry_vars: Option<Vec<String>>,
    pub default_terminator: Option<BlockTerminator>,
    pub module_scope: Vec<String>,
}

impl CompilerContext {
    pub fn empty() -> CompilerContext {
        Self {
            vars: BTreeMap::new(),
            carry_vars: None,
            default_terminator: None,
            module_scope: vec![],
        }
    }

    pub fn var_keys(&self) -> Vec<String> {
        self.vars.keys().cloned().collect()
    }

    pub fn unpack_vars(&self) -> Result<Vec<Value>, JITError> {
        let mut result = vec![];
        unpack_btree_to(&self.vars, &mut result)?;
        Ok(result)
    }

    pub fn repack_vars(
        &self,
        vars: &Vec<Value>,
        module_scope: Vec<String>,
        carry_vars: Option<Vec<String>>,
        default_terminator: Option<BlockTerminator>,
    ) -> Result<CompilerContext, JITError> {
        let res = repack_btree_from(&self.vars, vars, 0)?;
        Ok(CompilerContext {
            vars: res.0,
            carry_vars,
            default_terminator,
            module_scope,
        })
    }

    pub fn unpack_some_vars(&self, keys: &Vec<String>) -> Result<Vec<Value>, JITError> {
        let mut result = vec![];
        for key in keys {
            let Some(value) = self.vars.get(key) else {
                return JITError::generic(&format!("Variable not found {key}"));
            };
            value.unpack_to(&mut result)?;
        }
        Ok(result)
    }

    pub fn repack_some_vars(
        &mut self,
        keys: &Vec<String>,
        vars: &Vec<Value>,
        invalidate_bounds: bool,
    ) -> Result<(), JITError> {
        let mut pos = 0;
        for key in keys {
            let Some(value) = self.vars.get(key) else {
                return JITError::generic(&format!("Variable not found {key}"));
            };
            let (mut new_value, new_pos) = value.repack_from(vars, pos)?;
            if invalidate_bounds {
                new_value.bounds = None;
            }
            pos = new_pos;
            self.vars.insert(key.clone(), new_value);
        }
        Ok(())
    }
}
