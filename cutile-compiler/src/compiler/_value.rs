/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value representations used during compilation: typed MLIR values, structs,
//! compounds, compiler context (variable scopes), and block terminators.

use crate::bounds::Bounds;
use crate::compiler::_type::{Kind, TileRustType};
use crate::error::JITError;
use crate::syn_utils::get_type_ident;
use melior::ir::Value;
use std::collections::BTreeMap;
use syn::Expr;

/// Flattens all values in a `BTreeMap` of [`TileRustValue`]s into a linear list.
pub fn unpack_btree_to<'c, 'a>(
    btree: &BTreeMap<String, TileRustValue<'c, 'a>>,
    values: &mut Vec<Value<'c, 'a>>,
) -> Result<(), JITError> {
    for key in btree.keys() {
        let value = btree[key].clone();
        value.unpack_to(values)?;
    }
    Ok(())
}

/// Reconstructs a `BTreeMap` of [`TileRustValue`]s from a flat list of MLIR values.
pub fn repack_btree_from<'c, 'a>(
    old_btree: &BTreeMap<String, TileRustValue<'c, 'a>>,
    values: &Vec<Value<'c, 'a>>,
    mut pos: usize,
) -> Result<(BTreeMap<String, TileRustValue<'c, 'a>>, usize), JITError> {
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
pub struct TypeMeta<'c, 'a> {
    pub fields: BTreeMap<String, TileRustValue<'c, 'a>>,
}

impl<'c, 'a> TypeMeta<'c, 'a> {
    fn unpack_to(&self, values: &mut Vec<Value<'c, 'a>>) -> Result<(), JITError> {
        unpack_btree_to(&self.fields, values)
    }
    fn repack_from(
        &self,
        values: &Vec<Value<'c, 'a>>,
        pos: usize,
    ) -> Result<(TypeMeta<'c, 'a>, usize), JITError> {
        let res = repack_btree_from(&self.fields, values, pos)?;
        Ok((TypeMeta { fields: res.0 }, res.1))
    }
}

/// Mutability state of a variable binding during compilation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Mutability {
    /// Mutability has not been determined yet.
    Unset,
    /// The binding is mutable (`let mut`).
    Mutable,
    /// The binding is immutable (`let`).
    Immutable,
}

/// A compiled value: wraps an MLIR `Value` together with its Rust type, kind, bounds, and metadata.
#[derive(Debug, Clone)]
pub struct TileRustValue<'c, 'a> {
    // This may be a structured cuda tile type, a struct, or compound value.
    pub(crate) kind: Kind,
    pub(crate) fields: Option<BTreeMap<String, TileRustValue<'c, 'a>>>,
    pub(crate) values: Option<Vec<TileRustValue<'c, 'a>>>,
    pub(crate) value: Option<Value<'c, 'a>>,
    pub(crate) ty: TileRustType<'c>,
    // Type specific metadata required by some operations on some types.
    // These are complex cuda tile types which (at the time of authoring)
    // provide no way to access the corresponding associated fields.
    pub(crate) type_meta: Option<TypeMeta<'c, 'a>>,
    // Some operations require a token.
    // This is initialized to Mutability::Unset since it is meaningless outside of binding operations.
    pub(crate) mutability: Mutability,
    // Optional bounds for i32 types.
    // This is used to check bounds statically, when available.
    pub(crate) bounds: Option<Bounds<i64>>,
    // Original AST expression if this value was created from a string literal.
    pub(crate) string_literal: Option<syn::Expr>,
}

impl<'c, 'a> TileRustValue<'c, 'a> {
    /// Creates a struct value from named fields.
    pub fn new_struct(
        fields: BTreeMap<String, TileRustValue<'c, 'a>>,
        ty: TileRustType<'c>,
    ) -> TileRustValue<'c, 'a> {
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
    /// Creates a compound (tuple/array) value from ordered elements.
    pub fn new_compound(
        values: Vec<TileRustValue<'c, 'a>>,
        ty: TileRustType<'c>,
    ) -> TileRustValue<'c, 'a> {
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
    /// Creates a structured CUDA Tile type value (e.g. Tile, TensorView).
    pub fn new_structured_type(
        value: Value<'c, 'a>,
        ty: TileRustType<'c>,
        type_meta: Option<TypeMeta<'c, 'a>>,
    ) -> TileRustValue<'c, 'a> {
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
    /// Creates a scalar primitive value with optional static bounds.
    pub fn new_primitive(
        value: Value<'c, 'a>,
        ty: TileRustType<'c>,
        bounds: Option<Bounds<i64>>,
    ) -> TileRustValue<'c, 'a> {
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
    /// Creates a compile-time string literal value.
    pub fn new_string(string_literal: Expr, ty: TileRustType<'c>) -> TileRustValue<'c, 'a> {
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
    /// Creates a new value matching the kind of the given type.
    pub fn new_value_kind_like(
        value: Value<'c, 'a>,
        ty: TileRustType<'c>,
    ) -> TileRustValue<'c, 'a> {
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
    /// Creates an uncompiled literal value (stored as AST until needed).
    pub fn new_literal(literal_expr: syn::Expr, ty: TileRustType<'c>) -> TileRustValue<'c, 'a> {
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
    /// Validates internal consistency between `kind` and which fields are set.
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
                // Fields may or may not be set for cuda tile types.
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
    /// Returns a reference to a named type-metadata sub-field, if present.
    pub fn get_type_meta_field(&self, name: &str) -> Option<&Self> {
        let Some(type_meta) = &self.type_meta else {
            return None;
        };
        type_meta.fields.get(name)
    }
    /// Consumes self and removes a named type-metadata sub-field, returning it.
    pub fn take_type_meta_field(self, name: &str) -> Option<Self> {
        let Some(mut type_meta) = self.type_meta else {
            return None;
        };
        type_meta.fields.remove(name)
    }
    /// Inserts or replaces a named type-metadata sub-field.
    pub fn insert_type_meta_field(
        &mut self,
        name: &str,
        val: TileRustValue<'c, 'a>,
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
    /// Returns the ordering token sub-field, if present.
    pub fn get_token(&self) -> Option<&Self> {
        self.get_type_meta_field("token")
    }
    /// Returns `true` if the underlying Rust type is a `Tile*` type.
    pub fn is_tile(&self) -> bool {
        let Some(ident) = get_type_ident(&self.ty.rust_ty) else {
            return false;
        };
        ident.to_string().starts_with("Tile")
    }
    /// Returns `true` if the underlying Rust type is a `Partition*` type.
    pub fn is_partition(&self) -> bool {
        let Some(ident) = get_type_ident(&self.ty.rust_ty) else {
            return false;
        };
        ident.to_string().starts_with("Partition")
    }

    /// Recursively flattens this value's MLIR values into a linear list.
    pub fn unpack_to(&self, values: &mut Vec<Value<'c, 'a>>) -> Result<(), JITError> {
        self.verify()?;
        match self.kind {
            Kind::String => {
                // String types do not compile to Tile IR.
            }
            Kind::PrimitiveType => {
                // self.value is some. self.type_meta may be non-empty.
                values.push(self.value.unwrap());
                if let Some(old_type_meta) = &self.type_meta {
                    old_type_meta.unpack_to(values)?;
                }
            }
            Kind::StructuredType => {
                // self.value is some. self.type_meta may be non-empty.
                values.push(self.value.unwrap());
                if let Some(old_type_meta) = &self.type_meta {
                    old_type_meta.unpack_to(values)?;
                }
            }
            Kind::Compound => {
                // self.values is some.
                let Some(self_values) = &self.values else {
                    return JITError::generic("internal: compound value missing its element list");
                };
                for value in self_values {
                    value.unpack_to(values)?;
                }
            }
            Kind::Struct => {
                // self.fields is some.
                let Some(fields) = &self.fields else {
                    return JITError::generic("internal: struct value missing its fields");
                };
                unpack_btree_to(fields, values)?;
            }
        }
        Ok(())
    }

    /// Reconstructs this value's structure from a flat list of MLIR values starting at `pos`.
    pub fn repack_from(
        &self,
        values: &Vec<Value<'c, 'a>>,
        mut pos: usize,
    ) -> Result<(Self, usize), JITError> {
        self.verify()?;
        let mut result = self.clone();
        match self.kind {
            Kind::String => {
                // String types do not compile to Tile IR.
            }
            Kind::PrimitiveType => {
                // self.value is some. self.type_meta may be non-empty.
                result.value = Some(values[pos].clone());
                pos += 1;
                if let Some(old_type_meta) = result.type_meta {
                    let res = old_type_meta.repack_from(values, pos)?;
                    result.type_meta = Some(res.0);
                    pos = res.1;
                }
            }
            Kind::StructuredType => {
                // self.value is some. self.type_meta may be non-empty.
                result.value = Some(values[pos].clone());
                pos += 1;
                if let Some(old_type_meta) = result.type_meta {
                    let res = old_type_meta.repack_from(values, pos)?;
                    result.type_meta = Some(res.0);
                    pos = res.1;
                }
            }
            Kind::Compound => {
                // self.values is some.
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
                // self.fields is some.
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

/// How a compiled block transfers control at its end.
#[derive(Debug, Clone, Copy)]
pub enum BlockTerminator {
    /// Yield values to the enclosing region.
    Yield,
    /// Continue to the next loop iteration.
    Continue,
    /// Return from the function.
    Return,
    /// Break out of the enclosing loop.
    Break,
}

/// Variable scope and control-flow state for a compilation block.
#[derive(Debug, Clone)]
pub struct CompilerContext<'c, 'a> {
    /// Named variables in the current scope.
    pub vars: BTreeMap<String, TileRustValue<'c, 'a>>,
    /// Variables carried across loop iterations, if inside a loop.
    pub carry_vars: Option<Vec<String>>,
    /// Default terminator for the current block (e.g. yield for loops).
    pub default_terminator: Option<BlockTerminator>,
    /// Current module scope path used for symbol resolution.
    pub module_scope: Vec<String>,
}

impl<'c, 'a> CompilerContext<'c, 'a> {
    /// Creates an empty context with no variables or terminators.
    pub fn empty() -> CompilerContext<'c, 'a> {
        Self {
            vars: BTreeMap::new(),
            carry_vars: None,
            default_terminator: None,
            module_scope: vec![],
        }
    }
    /// Returns the names of all variables in scope.
    pub fn var_keys(&self) -> Vec<String> {
        self.vars.keys().cloned().collect()
    }

    /// Flattens all variable values into a linear MLIR value list.
    pub fn unpack_vars(&self) -> Result<Vec<Value<'c, 'a>>, JITError> {
        let mut result = vec![];
        unpack_btree_to(&self.vars, &mut result)?;
        Ok(result)
    }

    /// Reconstructs variables from a flat value list, replacing scope metadata.
    pub fn repack_vars(
        &self,
        vars: &Vec<Value<'c, 'a>>,
        module_scope: Vec<String>,
        carry_vars: Option<Vec<String>>,
        default_terminator: Option<BlockTerminator>,
    ) -> Result<CompilerContext<'c, 'a>, JITError> {
        let res = repack_btree_from(&self.vars, vars, 0)?;
        Ok(CompilerContext {
            vars: res.0,
            carry_vars,
            default_terminator,
            module_scope,
        })
    }

    /// Flattens only the specified variables into a linear value list.
    pub fn unpack_some_vars(&self, keys: &Vec<String>) -> Result<Vec<Value<'c, 'a>>, JITError> {
        let mut result = vec![];
        for key in keys {
            let Some(value) = self.vars.get(key) else {
                return JITError::generic(&format!("Variable not found {key}"));
            };
            value.unpack_to(&mut result)?;
        }
        Ok(result)
    }

    /// Repacks a subset of variables from a flat value list, optionally invalidating bounds.
    pub fn repack_some_vars(
        &mut self,
        keys: &Vec<String>,
        vars: &Vec<Value<'c, 'a>>,
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
