/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generic parameter resolution: maps Rust generic type and const parameters to
//! concrete values, and infers generic arguments from call-site context.

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use crate::syn_utils::{
    get_call_expression_generics, get_ident_from_path_expr, get_ident_generic_args, get_sig_types,
    get_supported_generic_params, get_type_ident, maybe_generic_args, strip_generic_args_lifetimes,
    CGAParameter, VarCGAParameter,
};
use crate::types::{
    get_ptr_type, get_ptr_type_instance, is_element_type, is_element_type_ptr,
    parse_signed_literal_as_i32, try_extract_cga,
};
use quote::ToTokens;
use std::collections::HashMap;
use std::hash::Hash;
use syn::{
    AngleBracketedGenericArguments, Expr, ExprCall, ExprMethodCall, GenericArgument, GenericParam,
    Generics, ImplItemFn, ItemImpl, Lit, PathArguments, Signature, Stmt, Type, TypePath,
};

/// Classification of a generic parameter variable.
#[derive(Debug)]
pub enum GenericVarType {
    // This is a generic type parameter. The T in <T, ...>
    /// A generic type parameter (the `T` in `<T, ...>`).
    TypeVariable,
    // This is a const generic parameter. The const X: T in <const X: T, ...>
    /// A const generic parameter (the `const X: T` in `<const X: T, ...>`).
    ConstVariable,
    // This is a length variable. The N in <const X: [i32; N], ...>
    /// An array length variable (the `N` in `<const X: [i32; N], ...>`).
    LengthVariable,
}

/// `GenericVars` = symbol table that maps generic parameter names (`E`, `BM`) to their concrete instantiated values (`f32`, `64`) for this specific compilation.
///
/// Rust generics need concrete values at compile-time.
///
/// **Example kernel**:
/// ```rust,ignore
/// fn gemm<E: ElementType, const BM: i32, const BN: i32, const K: i32>(
///     z: &mut Tensor<E, {[BM, BN]}>,
///     x: &Tensor<E, {[BM, K]}>,
///     y: &Tensor<E, {[K, BN]}>
/// )
/// ```
/// **When called with**: `gemm::<f32, 64, 64, 32>(...)`
/// **GenericVars stores the mapping**:
/// ```rust,ignore
/// GenericVars {
///     inst_types: {
///         "E" => f32
///     },
///     inst_i32: {
///         "BM" => 64,
///         "BN" => 64,
///         "K" => 32
///     }
/// }
/// ```
/// **Usage**: When compiler sees `Tensor<E, {[BM, BN]}>` in the code:
///  - Lookup `"E"` → `f32`
///  - Lookup `"BM"` → `64`, `"BN"` → `64`
///  - Substitute: `Tensor<f32, {[64, 64]}>`
#[derive(Debug)]
pub struct GenericVars {
    // Generic type param (the T in scalar: T or x: Tensor<T, ...>) to concrete impl ElementType.
    // TODO (hme): Ensure the type mapped here is _always_ concrete. Check inlining specifically.
    pub inst_types: HashMap<String, String>,
    // Generic i32 param name to i32.
    pub inst_i32: HashMap<String, i32>,
    // Generic array param name to generic array instance.
    pub inst_array: HashMap<String, Vec<i32>>,
    // A map from the length variable of a const generic array to the corresponding key in inst_array.
    pub len2array: HashMap<String, String>,
    pub ordered_param_vars: Vec<String>,
}

impl GenericVars {
    /// Returns the kind of generic variable for the given name, if it exists.
    pub fn var_type(&self, var: &str) -> Option<GenericVarType> {
        if self.inst_types.contains_key(var) {
            Some(GenericVarType::TypeVariable)
        } else if self.len2array.contains_key(var) {
            Some(GenericVarType::LengthVariable)
        } else if self.inst_i32.contains_key(var) || self.inst_array.contains_key(var) {
            Some(GenericVarType::ConstVariable)
        } else {
            return None;
        }
    }

    pub fn is_empty(generics: &Generics) -> bool {
        for generic_param in &generics.params {
            match generic_param {
                GenericParam::Type(_) => return false,
                GenericParam::Const(_) => return false,
                GenericParam::Lifetime(_) => continue,
            }
        }
        true
    }
    /// Creates an empty instance without validation. Use [`empty`](Self::empty) when possible.
    pub fn empty_unchecked() -> Self {
        let inst_types: HashMap<String, String> = HashMap::new();
        let inst_i32: HashMap<String, i32> = HashMap::new();
        let inst_array: HashMap<String, Vec<i32>> = HashMap::new();
        let len2array: HashMap<String, String> = HashMap::new();
        let ordered_param_vars: Vec<String> = Vec::new();
        GenericVars {
            inst_types,
            inst_i32,
            inst_array,
            len2array,
            ordered_param_vars,
        }
    }
    /// Creates an empty instance, returning an error if the signature has unresolved generics.
    pub fn empty(generics: &Generics) -> Result<Self, JITError> {
        if !Self::is_empty(generics) {
            return SourceLocation::unknown().jit_error_result(
                "expected no generic parameters, but found type or const parameters",
            );
        }
        Ok(Self::empty_unchecked())
    }
    pub fn from_flat(generics: &Generics, args: &[String]) -> Result<Self, JITError> {
        // This is used to initialize generics for kernel entry points.
        let mut inst_types: HashMap<String, String> = HashMap::new();
        // These are kernel entry points and require known length.
        // There are no variable length consts.
        let mut inst_i32: HashMap<String, i32> = HashMap::new();
        let mut inst_array: HashMap<String, Vec<i32>> = HashMap::new();
        let len2array: HashMap<String, String> = HashMap::new();
        let mut ordered_param_vars: Vec<String> = Vec::new();
        let mut pos: usize = 0;
        for generic_param in &generics.params {
            match generic_param {
                GenericParam::Const(const_param) => {
                    // const X: i32 or const Y: [i32; 3]
                    match &const_param.ty {
                        syn::Type::Array(_ty_arr) => {
                            let cga_param = CGAParameter::from_const_param(const_param);
                            let mut inst: Vec<i32> = vec![];
                            if pos + cga_param.length as usize > args.len() {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "not enough generic arguments to instantiate const array parameter `{}`; need {} more value(s)",
                                    cga_param.name, cga_param.length as usize - (args.len() - pos)
                                ));
                            }
                            for _ in 0..cga_param.length {
                                let arg = args[pos].parse::<i32>().unwrap();
                                inst.push(arg);
                                pos += 1;
                            }
                            ordered_param_vars.push(cga_param.name.clone());
                            inst_array.insert(cga_param.name.clone(), inst);
                        }
                        syn::Type::Path(type_path) => {
                            if pos + 1 > args.len() {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "not enough generic arguments to instantiate const parameter `{}`",
                                    const_param.ident
                                ));
                            }
                            let name = const_param.ident.to_string();
                            let ty = type_path.to_token_stream().to_string();
                            if ty != "i32" {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "const generic `{}` must be `i32`, got `{ty}`",
                                    const_param.ident
                                ));
                            }
                            let arg = args[pos].parse::<i32>().unwrap();
                            ordered_param_vars.push(name.clone());
                            inst_i32.insert(name, arg);
                            pos += 1;
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "unsupported type for const generic parameter `{}`; only `i32` and `[i32; N]` are supported",
                                const_param.ident
                            ));
                        }
                    }
                }
                GenericParam::Type(type_param) => {
                    let var_str = type_param.ident.to_string();
                    ordered_param_vars.push(var_str.clone());
                    inst_types.insert(var_str, args[pos].clone());
                    pos += 1;
                }
                GenericParam::Lifetime(_) => {}
            }
        }
        if pos != args.len() {
            return SourceLocation::unknown().jit_error_result(&format!(
                "too many generic arguments: expected {pos} but got {}",
                args.len()
            ));
        }
        Ok(GenericVars {
            inst_types,
            inst_i32,
            inst_array,
            len2array,
            ordered_param_vars,
        })
    }
    /// Constructs a `GenericVars` by matching explicit generic arguments to parameter declarations.
    pub fn from_expr_generic_args(
        &self,
        generics: &Generics,
        expr_generic_args: &Option<AngleBracketedGenericArguments>,
    ) -> Result<Self, JITError> {
        // This is used to initialize generics for inlined functions and methods.
        // self are the generics from the caller scope.
        // generics are generics from the callee scope.
        // The returned GenericVars instance are instantiated generic vars for the callee.
        // println!("inlining function with \n generics={generics:#?} \n expr_generic_args={expr_generic_args:#?}");
        let mut inst_types: HashMap<String, String> = HashMap::new();
        let mut inst_i32: HashMap<String, i32> = HashMap::new();
        let mut inst_array: HashMap<String, Vec<i32>> = HashMap::new();
        let mut len2array: HashMap<String, String> = HashMap::new();
        let mut ordered_param_vars: Vec<String> = vec![];
        if expr_generic_args.is_none() {
            return Ok(GenericVars {
                inst_types,
                inst_i32,
                inst_array,
                len2array,
                ordered_param_vars,
            });
        }
        let expr_generic_args = expr_generic_args.as_ref().unwrap();
        if generics.params.len() != expr_generic_args.args.len() {
            return SourceLocation::unknown().jit_error_result(&format!(
                "generic parameter count ({}) does not match argument count ({})",
                generics.params.len(),
                expr_generic_args.args.len()
            ));
        }

        let num_args = expr_generic_args.args.len();
        for i in 0..num_args {
            let generic_arg = &expr_generic_args.args[i];
            let generic_param = &generics.params[i];
            match (generic_arg, generic_param) {
                (GenericArgument::Const(const_arg), GenericParam::Const(const_param)) => {
                    // Instantiate a const generic param from a const arg expr.
                    match (&const_arg, &const_param.ty) {
                        (syn::Expr::Path(arg_path_expr), syn::Type::Array(_param_ty_array)) => {
                            let ident_str = get_ident_from_path_expr(arg_path_expr).to_string();
                            if VarCGAParameter::is_var_cga(const_param) {
                                let var_cga_param = VarCGAParameter::from_const_param(const_param);
                                let name = var_cga_param.name.to_string();
                                let Some(inst) = self.inst_array.get(ident_str.as_str()) else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{ident_str}` is not a known const generic array"
                                    ));
                                };
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name.clone(), inst.clone());
                                len2array.insert(var_cga_param.length_var, name);
                            } else {
                                let cga_param = CGAParameter::from_const_param(const_param);
                                let name = cga_param.name.to_string();
                                let Some(inst) = self.inst_array.get(ident_str.as_str()) else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{ident_str}` is not a known const generic array"
                                    ));
                                };
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name, inst.clone());
                                if cga_param.length as usize != inst.len() {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "const array parameter `{}` expects {} elements but got {}",
                                        cga_param.name,
                                        cga_param.length,
                                        inst.len()
                                    ));
                                }
                            }
                        }
                        (syn::Expr::Path(arg_path_expr), syn::Type::Path(_param_ty_path)) => {
                            let name = const_param.ident.to_string();
                            let ident_str = get_ident_from_path_expr(arg_path_expr).to_string();
                            let Some(inst) = self.inst_i32.get(ident_str.as_str()) else {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "variable `{ident_str}` is not a known const generic scalar"
                                ));
                            };
                            ordered_param_vars.push(name.clone());
                            inst_i32.insert(name, inst.clone());
                        }
                        (syn::Expr::Lit(_arg_lit_expr), syn::Type::Path(_param_ty_path)) => {
                            let name = const_param.ident.to_string();
                            let literal_i32 = parse_signed_literal_as_i32(const_arg);
                            ordered_param_vars.push(name.clone());
                            inst_i32.insert(name, literal_i32);
                        }
                        (syn::Expr::Block(_), syn::Type::Array(ty_arr)) => {
                            let name = const_param.ident.to_string();
                            let from_generic_args = self;
                            if let Some(res) = try_get_const_generic_from_generic_argument(
                                &generic_arg,
                                from_generic_args,
                            ) {
                                // This is something like
                                // CONST_ARG -> N
                                inst_i32.insert(name.clone(), res);
                            } else {
                                let Some(res) =
                                    get_cga_from_generic_argument(&generic_arg, from_generic_args)
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "unable to resolve generic argument `{}` for parameter `{}`",
                                        generic_arg.to_token_stream().to_string(),
                                        generic_param.to_token_stream().to_string()
                                    ));
                                };
                                // This is something like
                                // {[...]} -> CONST_ARRAY_PARAM
                                inst_array.insert(name.clone(), res);
                                if let Expr::Path(length_expr) = &ty_arr.len {
                                    let length_var = length_expr
                                        .path
                                        .get_ident()
                                        .unwrap()
                                        .to_string()
                                        .to_string();
                                    len2array.insert(length_var, name.clone());
                                }
                            }
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "unable to resolve generic argument `{}` for parameter `{}`",
                                generic_arg.to_token_stream().to_string(),
                                generic_param.to_token_stream().to_string()
                            ));
                        }
                    }
                }
                (GenericArgument::Type(ty_arg), GenericParam::Const(const_param)) => {
                    // Instantiate a const generic param from a type arg.
                    // println!("instantiating generic={const_param:#?} \n expr={generic_arg:#?}");
                    let Some(arg_ident_str) = get_type_ident(ty_arg) else {
                        return SourceLocation::unknown().jit_error_result(
                            "unable to extract type identifier from const generic argument",
                        );
                    };
                    match &const_param.ty {
                        syn::Type::Array(_param_ty_array) => {
                            if VarCGAParameter::is_var_cga(const_param) {
                                let var_cga_param = VarCGAParameter::from_const_param(const_param);
                                let name = var_cga_param.name.to_string();
                                let Some(inst) =
                                    self.inst_array.get(arg_ident_str.to_string().as_str())
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{}` is not a known const generic array",
                                        arg_ident_str
                                    ));
                                };
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name.clone(), inst.clone());
                                len2array.insert(var_cga_param.length_var, name);
                            } else {
                                let cga_param = CGAParameter::from_const_param(const_param);
                                let name = cga_param.name.to_string();
                                let Some(inst) =
                                    self.inst_array.get(arg_ident_str.to_string().as_str())
                                else {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "variable `{}` is not a known const generic array",
                                        arg_ident_str
                                    ));
                                };
                                if cga_param.length as usize != inst.len() {
                                    return SourceLocation::unknown().jit_error_result(&format!(
                                        "const array parameter `{}` expects {} elements but got {}",
                                        cga_param.name,
                                        cga_param.length,
                                        inst.len()
                                    ));
                                }
                                ordered_param_vars.push(name.clone());
                                inst_array.insert(name, inst.clone());
                            }
                        }
                        syn::Type::Path(_param_ty_path) => {
                            let name = const_param.ident.to_string();
                            let Some(inst) = self.inst_i32.get(arg_ident_str.to_string().as_str())
                            else {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "variable `{}` is not a known const generic scalar",
                                    arg_ident_str
                                ));
                            };
                            ordered_param_vars.push(name.clone());
                            inst_i32.insert(name.clone(), inst.clone());
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "unable to resolve generic argument `{}` for parameter `{}`",
                                generic_arg.to_token_stream().to_string(),
                                generic_param.to_token_stream().to_string()
                            ));
                        }
                    }
                }
                (GenericArgument::Type(ty_arg), GenericParam::Type(ty_param)) => {
                    // Instantiate a type parameter from a type arg.
                    let Some(arg_ident_str) = get_type_ident(ty_arg) else {
                        return SourceLocation::unknown().jit_error_result(
                            "unable to extract type identifier from type argument",
                        );
                    };
                    let name = ty_param.ident.to_string();
                    let Some(inst) = self.inst_types.get(arg_ident_str.to_string().as_str()) else {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "type `{}` is not a known generic type parameter",
                            arg_ident_str
                        ));
                    };
                    ordered_param_vars.push(name.clone());
                    inst_types.insert(name.clone(), inst.clone());
                }
                (_, _) => {
                    return SourceLocation::unknown().jit_error_result(&format!(
                        "Generic arg / param not supported: {generic_arg:#?} \n {generic_param:#?}"
                    ));
                }
            }
        }
        Ok(GenericVars {
            inst_types,
            inst_i32,
            inst_array,
            len2array,
            ordered_param_vars,
        })
    }
    /// Looks up a const-generic `i32` value by parameter name.
    pub fn get_i32(&self, name: &str) -> Option<i32> {
        if let Some(inst) = self.inst_i32.get(name) {
            return Some(*inst);
        }
        if let Some(arr_name) = self.len2array.get(name) {
            if let Some(inst) = self.inst_array.get(arr_name) {
                return Some(inst.len() as i32);
            } else {
                // Internal invariant: length var exists but corresponding array doesn't.
                return None;
            }
        }
        None
    }

    /// Merges another `GenericVars` into this one, erroring on conflicting entries.
    pub fn merge(mut self, other: GenericVars) -> Result<GenericVars, JITError> {
        self.inst_types = self.inst_types.merge_if_eq(other.inst_types);
        self.inst_i32 = self.inst_i32.merge_if_eq(other.inst_i32);
        self.inst_array = self.inst_array.merge_if_eq(other.inst_array);
        self.len2array = self.len2array.merge_if_eq(other.len2array);
        if !self.ordered_param_vars.is_empty() && !other.ordered_param_vars.is_empty() {
            if self.ordered_param_vars != other.ordered_param_vars {
                return SourceLocation::unknown().jit_error_result(&format!(
                    "Ordered param vars mismatch: {:?} != {:?}",
                    self.ordered_param_vars, other.ordered_param_vars
                ));
            }
        } else {
            // At least one is empty. Take the value of the other.
            self.ordered_param_vars.extend(other.ordered_param_vars);
        }
        Ok(self)
    }

    /// Resolves a possibly-generic `syn::Type` into a concrete [`TypeInstance`].
    pub fn instantiate_type(
        &self,
        ty: &syn::Type,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Result<TypeInstance, JITError> {
        // Identify all generic args, and replace them with concrete (instantiated) values.
        // The Instantiable trait returns an instance even if the given type is concrete.
        let maybe_generic_ty = ty.clone();
        // Is this an element type? This is just a T.
        if let Some(instance) =
            TypeInstanceTokenType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::TokenType(instance));
        }
        if let Some(instance) =
            TypeInstanceElementType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::ElementType(instance));
        }
        // Is this a pointer? Something like * mut T.
        if let Some(instance) =
            TypeInstancePtrType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::PtrType(instance));
        }
        // Is this a string?
        if let Some(instance) =
            TypeInstanceStringType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::StringType(instance));
        }
        // Assume it's a structured type, something like Tile/Tensor/Partition/PartitionMut <T, Shape>
        // This also handles PointerTile<* mut T, Shape>.
        if let Some(instance) =
            TypeInstanceStructuredType::instantiate(&maybe_generic_ty, self, primitives)
        {
            return Ok(TypeInstance::StructuredType(instance));
        } else {
            return SourceLocation::unknown().jit_error_result(&format!(
                "unable to resolve generic type `{}`",
                maybe_generic_ty.to_token_stream().to_string()
            ));
        }
    }
}

/// Trait for types that can be instantiated from a generic `syn::Type`.
pub trait Instantiable {
    fn instantiate(
        generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self>
    where
        Self: Sized;
}

/// A fully-instantiated type, classifying how a Rust type maps to the CUDA Tile type system.
#[derive(Debug, Clone)]
pub enum TypeInstance {
    /// A plain user-defined or unresolved type.
    UserType(TypeInstanceUserType),
    /// The `str` string type.
    StringType(TypeInstanceStringType),
    /// The `Token` ordering-token type.
    TokenType(TypeInstanceTokenType),
    /// A scalar element type (e.g. `f32`, `i32`).
    ElementType(TypeInstanceElementType),
    /// A pointer type (`*mut E` / `*const E`).
    PtrType(TypeInstancePtrType),
    /// A shaped type with element type and dimensions (e.g. `Tile<f32, {[128]}>`).
    StructuredType(TypeInstanceStructuredType),
}

impl TypeInstance {
    /// Returns the concrete Rust element type name, if applicable.
    pub fn get_rust_element_instance_ty(&self) -> Option<String> {
        match self {
            Self::UserType(_inst) => None,
            Self::StringType(_inst) => None,
            Self::TokenType(_inst) => None,
            Self::ElementType(inst) => Some(inst.rust_element_instance_ty.clone()),
            Self::PtrType(inst) => Some(inst.rust_element_instance_ty.clone()),
            Self::StructuredType(inst) => {
                if let Some(primitive_type) = &inst.primitive_type {
                    primitive_type.get_rust_element_instance_ty()
                } else {
                    None
                }
            }
        }
    }
    /// Returns the concrete (instantiated) `syn::Type`.
    pub fn get_instantiated_type(&self) -> &syn::Type {
        match self {
            Self::UserType(inst) => &inst.maybe_generic_ty,
            Self::StringType(inst) => &inst.instance_ty,
            Self::TokenType(inst) => &inst.instance_ty,
            Self::ElementType(inst) => &inst.instance_ty,
            Self::PtrType(inst) => &inst.instance_ty,
            Self::StructuredType(inst) => &inst.instance_ty,
        }
    }
    /// Returns the original (possibly generic) `syn::Type`.
    pub fn get_source_type(&self) -> &syn::Type {
        match self {
            Self::UserType(inst) => &inst.maybe_generic_ty,
            Self::StringType(inst) => &inst.generic_ty,
            Self::TokenType(inst) => &inst.generic_ty,
            Self::ElementType(inst) => &inst.generic_ty,
            Self::PtrType(inst) => &inst.generic_ty,
            Self::StructuredType(inst) => &inst.generic_ty,
        }
    }
}

#[derive(Debug, Clone)]
/// Primitive type instance: either a scalar element type or a pointer type.
pub enum TypInstancePrimitiveType {
    /// Scalar element type (e.g. `f32`).
    ElementType(TypeInstanceElementType),
    /// Pointer type (e.g. `*mut f32`).
    PtrType(TypeInstancePtrType),
}

impl TypInstancePrimitiveType {
    pub fn get_rust_element_instance_ty(&self) -> Option<String> {
        match self {
            Self::ElementType(inst) => Some(inst.rust_element_instance_ty.clone()),
            Self::PtrType(inst) => Some(inst.rust_element_instance_ty.clone()),
        }
    }
    pub fn get_instantiated_type(&self) -> &syn::Type {
        match self {
            Self::ElementType(inst) => &inst.instance_ty,
            Self::PtrType(inst) => &inst.instance_ty,
        }
    }
}

#[derive(Debug, Clone)]
/// A user-defined or unresolved type that may still contain generics.
pub struct TypeInstanceUserType {
    pub(crate) maybe_generic_ty: syn::Type,
}

impl Instantiable for TypeInstanceUserType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        _generic_vars: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        // TODO (np): Add check for unresolved generics - return None if the type contains
        // generic parameters that haven't been instantiated yet. For now, we accept all types.
        Some(Self {
            maybe_generic_ty: maybe_generic_ty.clone(),
        })
    }
}

impl TypeInstanceUserType {
    /// Attempts to extract a const generic array from this type's generic arguments.
    pub fn try_extract_cga(&self, generic_vars: &GenericVars) -> Option<Vec<i32>> {
        try_extract_cga(&self.maybe_generic_ty, generic_vars)
    }
}

#[derive(Debug, Clone)]
/// A resolved `str` string type instance.
pub struct TypeInstanceStringType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
}

impl Instantiable for TypeInstanceStringType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        _generic_vars: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        // let string_lit = syn::Type::Path(maybe_generic_ty) else {
        //     panic!()
        // };
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if maybe_generic_type_str == "str" {
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved `Token` ordering-token type instance.
pub struct TypeInstanceTokenType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
}

impl Instantiable for TypeInstanceTokenType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        _generic_vars: &GenericVars,
        _primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if maybe_generic_type_str == "Token" {
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved scalar element type (e.g. `f32`) with its concrete Rust name.
pub struct TypeInstanceElementType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
    pub(crate) rust_element_instance_ty: String,
}

impl Instantiable for TypeInstanceElementType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if is_element_type(&maybe_generic_type_str, primitives) {
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
                rust_element_instance_ty: maybe_generic_type_str,
            })
        } else if let Some(rust_element_instance_ty) = generic_vars
            .inst_types
            .get(&maybe_generic_type_str)
            .cloned()
        {
            let instance_ty =
                syn::parse2::<syn::Type>(rust_element_instance_ty.parse().unwrap()).unwrap();
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty,
                rust_element_instance_ty,
            })
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved pointer type (`*mut E` / `*const E`) with mutability and element info.
pub struct TypeInstancePtrType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
    pub(crate) is_mutable: bool,
    pub(crate) rust_element_instance_ty: String,
}

impl Instantiable for TypeInstancePtrType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        let maybe_generic_type_str = maybe_generic_ty.to_token_stream().to_string();
        if is_element_type_ptr(&maybe_generic_type_str, primitives) {
            let (is_mutable, ptr_ty) =
                get_ptr_type(&maybe_generic_type_str).expect("Unexpected pointer type.");
            Some(Self {
                generic_ty: maybe_generic_ty.clone(),
                instance_ty: maybe_generic_ty.clone(),
                is_mutable,
                rust_element_instance_ty: ptr_ty,
            })
        } else if let Some((is_mutable, ptr_ty)) = get_ptr_type(&maybe_generic_type_str) {
            let ptr_prefix = if is_mutable { "* mut " } else { "* const " };
            if let Some(concrete_ptr_ty) = generic_vars.inst_types.get(&ptr_ty) {
                let instance_ty = syn::parse2::<syn::Type>(
                    format!("{ptr_prefix} {concrete_ptr_ty}").parse().unwrap(),
                )
                .unwrap();
                Some(Self {
                    generic_ty: maybe_generic_ty.clone(),
                    instance_ty: instance_ty,
                    is_mutable,
                    rust_element_instance_ty: concrete_ptr_ty.to_string(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// A resolved shaped type (e.g. `Tile<f32, {[128, 64]}>`) with element type and shape.
pub struct TypeInstanceStructuredType {
    pub(crate) generic_ty: syn::Type,
    pub(crate) instance_ty: syn::Type,
    pub(crate) primitive_type: Option<TypInstancePrimitiveType>,
    pub(crate) shape: Vec<i32>,
}

impl Instantiable for TypeInstanceStructuredType {
    fn instantiate(
        maybe_generic_ty: &syn::Type,
        generic_vars: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> Option<Self> {
        // This is Tile/Tensor/Partition/PartitionMut <T, Shape>.
        // This also handles PointerTile <* mut T, Shape>.
        let (_type_ident, mut type_generic_args) = get_ident_generic_args(maybe_generic_ty);
        strip_generic_args_lifetimes(&mut type_generic_args);
        let generic_ty = maybe_generic_ty.clone();
        let mut instance_ty = maybe_generic_ty.clone();
        let mut primitive_type: Option<TypInstancePrimitiveType> = None;
        let mut shape: Option<Vec<i32>> = None;

        let inst_mut_ref = if let Type::Reference(inner_elem) = &mut instance_ty {
            &mut *inner_elem.elem
        } else {
            &mut instance_ty
        };
        let instance_generics = if let Type::Path(type_path) = inst_mut_ref {
            let last_seg = type_path
                .path
                .segments
                .last_mut()
                .expect(format!("Unexpected structured type {maybe_generic_ty:#?}.").as_str());
            let PathArguments::AngleBracketed(type_params) = &mut last_seg.arguments else {
                panic!(
                    "Unexpected structured type generic arguments {:#?} for {maybe_generic_ty:#?}",
                    last_seg.arguments
                );
            };
            // This is a type of the form StructuredType<...>
            type_params
        } else {
            panic!("Unexpected structured type {maybe_generic_ty:#?}.");
        };

        for generic_arg in instance_generics.args.iter_mut() {
            match generic_arg {
                GenericArgument::Lifetime(_) => continue,
                GenericArgument::Type(type_param) => {
                    // Currently, this is either shape or element_type
                    match type_param {
                        syn::Type::Path(type_path) => {
                            let last_ident =
                                type_path.path.segments.last().unwrap().ident.to_string();
                            // println!("get_variadic_type_args: Type::Path: {}", last_ident);
                            if generic_vars.inst_array.contains_key(&last_ident) {
                                // This is something like Shape<D> for const generic array D: [i32; N].
                                let array_instance =
                                    generic_vars.inst_array.get(&last_ident).unwrap();
                                if shape.is_some() {
                                    panic!("Unexpected array arg: {last_ident:#?}")
                                }
                                shape = Some(array_instance.clone());
                                let shape_str = array_instance
                                    .iter()
                                    .map(|x| x.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                *generic_arg = GenericArgument::Const(
                                    syn::parse2::<Expr>(
                                        format!("{{[{}]}}", shape_str).parse().unwrap(),
                                    )
                                    .unwrap(),
                                );
                            } else if let Some(local_rust_element_instance_ty) =
                                generic_vars.inst_types.get(&last_ident)
                            {
                                // This is something like T.
                                let instance_ty = syn::parse2::<Type>(
                                    local_rust_element_instance_ty.parse().unwrap(),
                                )
                                .unwrap();
                                primitive_type = Some(TypInstancePrimitiveType::ElementType(
                                    TypeInstanceElementType {
                                        generic_ty: type_param.clone(),
                                        instance_ty: instance_ty.clone(),
                                        rust_element_instance_ty: local_rust_element_instance_ty
                                            .clone(),
                                    },
                                ));
                                *generic_arg = GenericArgument::Type(instance_ty);
                            } else if is_element_type(&last_ident, primitives) {
                                // This is something like f32.
                                primitive_type = Some(TypInstancePrimitiveType::ElementType(
                                    TypeInstanceElementType {
                                        generic_ty: type_param.clone(),
                                        instance_ty: type_param.clone(),
                                        rust_element_instance_ty: last_ident.clone(),
                                    },
                                ));
                                *generic_arg = GenericArgument::Type(
                                    syn::parse2::<Type>(last_ident.parse().unwrap()).unwrap(),
                                );
                            } else if generic_vars.inst_i32.contains_key(&last_ident) {
                                // This is something like N for const generic N: i32.
                                panic!("Unexpected const arg {last_ident} for variadic type {maybe_generic_ty:#?}");
                            } else {
                                panic!("Failed to get cuda tile type for ty={} \n generic_arg={generic_arg:#?} \n generic_args={generic_vars:#?}", maybe_generic_ty.to_token_stream().to_string());
                            }
                        }
                        syn::Type::Ptr(_) => {
                            let Some(ptr_inst) = TypeInstancePtrType::instantiate(
                                type_param,
                                &generic_vars,
                                primitives,
                            ) else {
                                panic!("Unexpected primitives {primitives:#?}.")
                            };
                            *generic_arg = GenericArgument::Type(ptr_inst.instance_ty.clone());
                            primitive_type = Some(TypInstancePrimitiveType::PtrType(ptr_inst));
                        }
                        syn::Type::Reference(type_ref) => {
                            unimplemented!("TypeInstanceStructuredType::instantiate: Type::Reference not supported: {:#?}", type_ref);
                        }
                        _ => {}
                    }
                }
                GenericArgument::Const(const_expr) => {
                    // println!("expand GenericArgument::Const? {const_param:#?}");
                    match const_expr {
                        Expr::Block(block_expr) => {
                            // This is something like Tensor<E, {[...]}>
                            assert_eq!(block_expr.block.stmts.len(), 1);
                            let statement = &block_expr.block.stmts[0];
                            let Stmt::Expr(statement_expr, _) = statement else {
                                panic!("Unexpected block expression.")
                            };
                            match statement_expr {
                                Expr::Array(array_expr) => {
                                    // This is something like Tensor<E, {[1, 2, -1]}>
                                    let mut _shape = vec![];
                                    for elem in &array_expr.elems {
                                        match elem {
                                            Expr::Lit(lit) => {
                                                let val = match &lit.lit {
                                                    Lit::Int(int_lit) => int_lit.base10_parse::<i32>().unwrap(),
                                                    _ => unimplemented!("Unexpected array element {elem:#?} in {array_expr:#?}"),
                                                };
                                                _shape.push(val);
                                            },
                                            Expr::Unary(unary_expr ) => {
                                                let unary_expr_str = unary_expr.to_token_stream().to_string();
                                                if unary_expr_str == "- 1" {
                                                    _shape.push(-1);
                                                } else {
                                                    panic!("Unexpected unary expression {unary_expr_str:#?} in {array_expr:#?}")
                                                }
                                            },
                                            Expr::Path(path) => {
                                                let ident = get_ident_from_path_expr(path);
                                                match generic_vars.inst_i32.get(ident.to_string().as_str()) {
                                                    Some(val) => _shape.push(*val),
                                                    None => panic!("Undefined generic parameter {ident}")
                                                }
                                            },
                                            _ => unimplemented!("Unexpected array element {elem:#?} in {array_expr:#?}"),
                                        }
                                    }
                                    let shape_str = _shape
                                        .iter()
                                        .map(|x| x.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    *generic_arg = GenericArgument::Const(
                                        syn::parse2::<Expr>(
                                            format!("{{[{}]}}", shape_str).parse().unwrap(),
                                        )
                                        .unwrap(),
                                    );
                                    shape = Some(_shape);
                                }
                                Expr::Repeat(repeat_expr) => {
                                    // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                                    let repeat_expr_expr = &*repeat_expr.expr;
                                    let thing_to_repeat = match repeat_expr_expr {
                                        Expr::Lit(lit) => {
                                            match &lit.lit {
                                                Lit::Int(int_lit) => int_lit.base10_parse::<i32>().unwrap(),
                                                _ => unimplemented!("Unexpected repeat expr {repeat_expr_expr:#?} in {repeat_expr:#?}"),
                                            }
                                        },
                                        Expr::Unary(unary_expr ) => {
                                            let unary_expr_str = unary_expr.to_token_stream().to_string();
                                            if unary_expr_str == "- 1" {
                                                -1
                                            } else {
                                                unimplemented!("Unexpected unary expression {repeat_expr_expr:#?} in {repeat_expr:#?}")
                                            }
                                        },
                                        Expr::Path(path) => {
                                            let ident = get_ident_from_path_expr(path);
                                            match generic_vars.inst_i32.get(ident.to_string().as_str()) {
                                                Some(val) => *val,
                                                None => panic!("Undefined generic parameter {ident}")
                                            }
                                        },
                                        _ => unimplemented!("Unexpected unary expression {repeat_expr_expr:#?} in {repeat_expr:#?}"),
                                    };
                                    let num_rep = match &*repeat_expr.len {
                                        Expr::Path(len_path) => {
                                            // This is something like Tensor<E, {[-1; N]}>
                                            let num_rep_var =
                                                len_path.to_token_stream().to_string();
                                            if !generic_vars.get_i32(&num_rep_var).is_some() {
                                                panic!(
                                                    "Expected instance for generic argument {}",
                                                    num_rep_var
                                                );
                                            }
                                            generic_vars.get_i32(&num_rep_var).unwrap()
                                        }
                                        Expr::Lit(len_lit) => {
                                            // This is something like Tensor<E, {[-1; 3]}>
                                            len_lit
                                                .to_token_stream()
                                                .to_string()
                                                .parse::<i32>()
                                                .unwrap()
                                        }
                                        _ => unimplemented!(
                                            "Unexpected repeat expression: {repeat_expr:#?}"
                                        ),
                                    };
                                    let repeat_str = format!("{{[{thing_to_repeat}; {num_rep}]}}");
                                    *generic_arg = GenericArgument::Const(
                                        syn::parse2::<Expr>(repeat_str.parse().unwrap()).unwrap(),
                                    );
                                    shape = Some(vec![thing_to_repeat; num_rep as usize]);
                                }
                                _ => panic!("Unexpected block expression."),
                            }
                        }
                        _ => panic!("Unexpected const expression {const_expr:#?}"),
                    }
                }
                _ => panic!("Unexpected generic argument {generic_arg:#?}"),
            }
        }
        match shape {
            Some(shape) => Some(Self {
                generic_ty,
                instance_ty,
                primitive_type,
                shape,
            }),
            _ => {
                panic!("Unable to parse {maybe_generic_ty:#?} \n primitive_type = {primitive_type:#?} \n shape = {shape:#?}");
            }
        }
    }
}

trait MergeIfEqual {
    fn merge_if_eq(self, other: Self) -> Self;
}

impl<K: Hash + Eq, V: Clone + PartialEq> MergeIfEqual for HashMap<K, V> {
    fn merge_if_eq(mut self, other: Self) -> Self {
        for (key, value) in other {
            if let Some(self_value) = self.insert(key, value.clone()) {
                assert!(self_value == value);
            }
        }
        self
    }
}

#[derive(Debug)]
/// Classification of a generic argument for inference purposes.
pub enum GenericArgType {
    // Any type. The T in <T, ...> or the element type E of a pointer *mut E.
    /// A type argument (e.g. the `T` in `<T>`).
    Type,
    // Any const generic expression. The D_i in <{[..., D_i, ...]}> or an array expression {[...]} in <..., {[...]}, ...>
    /// A const generic expression (e.g. `{[BM, BN]}`).
    GenericConstExpr,
}

#[derive(Debug)]
/// Infers generic arguments for a function or method call from the call-site context.
pub struct GenericArgInference {
    // Attempt to infer generics for function or impl/method pair from context of call site.
    // This is used to:
    // 1. Infer return type generic args.
    // 2. Construct GenericVars instance for inlined function and method calls.

    // TODO (hme): This is basically two structs at this point. Separate fn from method call.
    // pub impl_params: Option<Vec<String>>,
    pub method_params: Option<Vec<String>>,
    // pub impl_sig: Signature,
    // pub method_sig: Signature,
    pub sig: Signature,
    pub params: Vec<String>,
    pub param2arg: HashMap<String, Option<(GenericArgType, String)>>,
    pub param2cga: HashMap<String, Type>,
}

// TODO (hme): Separate generic parameter inference from type inference procedure.
//  Rewrite type inference procedure to instantiate types from an instance of GenericVars.
//  The basic procedure is carried out the same way, but eliminate any assumption about
//  the structure of types when inferring generic parameters from input types.
//  We still make an assumption about the structure of types on the instantiation side.
impl GenericArgInference {
    /// Creates an inference context for a method call, merging impl and method generics.
    pub fn new_method(impl_item: &ItemImpl, impl_method: &ImplItemFn) -> Self {
        let mut merged_generics = vec![];

        let impl_generics = get_supported_generic_params(&impl_item.generics);
        let mut impl_params = vec![];
        for item in &impl_generics {
            impl_params.push(item.0.clone());
            merged_generics.push(item.clone());
        }
        let method_generics = get_supported_generic_params(&impl_method.sig.generics);
        let mut method_params = vec![];
        for item in method_generics {
            method_params.push(item.0.clone());
            merged_generics.push(item)
        }

        let mut param2cga = HashMap::new();
        let mut param2arg = HashMap::new();
        let mut params = vec![];
        for (name, maybe_ty) in merged_generics {
            params.push(name.to_string());
            if let Some(ty) = maybe_ty {
                param2cga.insert(name.clone(), ty.clone());
            }
            param2arg.insert(name, None);
        }
        // TODO (hme): Change params: method_params
        //  when refactored to separate structs for method vs. fn calls.
        Self {
            sig: impl_method.sig.clone(),
            param2cga,
            param2arg,
            params,
            method_params: Some(method_params),
        }
    }

    pub fn new_function(sig: Signature) -> Self {
        let fn_generics = get_supported_generic_params(&sig.generics);
        let mut param2cga = HashMap::new();
        let mut param2arg = HashMap::new();
        let mut params = vec![];
        for (name, maybe_ty) in fn_generics {
            params.push(name.to_string());
            if let Some(ty) = maybe_ty {
                param2cga.insert(name.clone(), ty.clone());
            }
            param2arg.insert(name, None);
        }
        Self {
            sig,
            param2cga,
            param2arg,
            params,
            method_params: None,
        }
    }

    /// Maps positional call arguments to their corresponding parameter names.
    pub fn map_args_to_params(
        &mut self,
        call_arg_rust_tys: &Vec<syn::Type>,
        self_ty: Option<&Type>,
    ) -> () {
        let (fn_arg_types, _return_type) = get_sig_types(&self.sig, self_ty);
        // Get the generic parameters in this function signature.
        for i in 0..call_arg_rust_tys.len() {
            let call_arg_rust_ty = &call_arg_rust_tys[i];
            let fn_arg_types = &fn_arg_types[i];
            self.add_generic_args(fn_arg_types, call_arg_rust_ty);
        }
    }
    /// Applies explicitly provided generic arguments from a function call expression.
    pub fn apply_provided_generics_fn_call(
        &mut self,
        call_expr: &ExprCall,
        generic_vars: &GenericVars,
    ) {
        let Some(expr_generic_args) = get_call_expression_generics(call_expr) else {
            return;
        };
        assert_eq!(expr_generic_args.args.len(), self.params.len());
        for i in 0..expr_generic_args.args.len() {
            let param = &self.params[i];
            let arg = &expr_generic_args.args[i];
            if let Some(Some(_)) = self.param2arg.get(param) {
                // The type for this has already been inferred.
                // Our compiler doesn't need to check anything. Rust already has.
                continue;
            }
            match arg {
                // Get generic args from call f::<...>()
                // and infer generic parameters in fn f<...>.
                // Since these are generic arguments as part of a function call expression,
                // they are either "type" or "const."
                GenericArgument::Type(arg_ty) => {
                    let Some(arg_type_ident) = get_type_ident(arg_ty) else {
                        panic!("apply_provided_generics_fn_call: Failed to get ident for type {arg_ty:#?}");
                    };
                    let var_string = arg_type_ident.to_string();
                    let var_str = var_string.as_str();
                    if let Some(arg_type_var_type) = generic_vars.var_type(var_str) {
                        // This is a generic type variable.
                        match arg_type_var_type {
                            GenericVarType::TypeVariable => {
                                let Some(inst_type) = generic_vars.inst_types.get(var_str) else {
                                    panic!("Undefined instance type {var_str}")
                                };
                                self.param2arg.insert(
                                    param.to_string(),
                                    Some((GenericArgType::Type, inst_type.to_string())),
                                );
                            }
                            GenericVarType::ConstVariable => {
                                if let Some(inst_i32) = generic_vars.inst_i32.get(var_str) {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            (*inst_i32).to_string(),
                                        )),
                                    );
                                } else if let Some(inst_arr) = generic_vars.inst_array.get(var_str)
                                {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::GenericConstExpr,
                                            format!("{{[{:?}]}}", inst_arr),
                                        )),
                                    );
                                } else {
                                    panic!("Undefined instance type {var_str}")
                                }
                            }
                            GenericVarType::LengthVariable => {
                                panic!(
                                    "Unexpected GenericVarType::LengthVariable {var_str}. \
                                Length variables can only be inferred from const generic arrays."
                                )
                            }
                        }
                    } else {
                        // This is a type (not a type variable).
                        self.param2arg.insert(
                            param.to_string(),
                            Some((GenericArgType::Type, var_str.to_string())),
                        );
                    };
                }
                GenericArgument::Const(c) => {
                    self.param2arg.insert(
                        param.to_string(),
                        Some((
                            GenericArgType::GenericConstExpr,
                            c.to_token_stream().to_string(),
                        )),
                    );
                }
                _ => {}
            }
        }
    }

    /// Applies explicitly provided generic arguments from a method call expression.
    pub fn apply_provided_generics_method_call(
        &mut self,
        method_call_expr: &ExprMethodCall,
        generic_vars: &GenericVars,
    ) {
        let Some(expr_generic_args) = &method_call_expr.turbofish else {
            return;
        };
        // These won't be the same length.
        let Some(method_params) = &self.method_params else {
            panic!(
                "Method params undefined for {}",
                method_call_expr.to_token_stream().to_string()
            )
        };
        assert_eq!(expr_generic_args.args.len(), method_params.len());
        for i in 0..expr_generic_args.args.len() {
            let param = &self.params[i];
            let arg = &expr_generic_args.args[i];
            if let Some(Some(_)) = self.param2arg.get(param) {
                // The type for this has already been inferred.
                // Our compiler doesn't need to check anything. Rust already has.
                continue;
            }
            match arg {
                // Get generic args from call f::<...>()
                // and infer generic parameters in fn f<...>.
                // Since these are generic arguments as part of a function call expression,
                // they are either "type" or "const."
                GenericArgument::Type(arg_ty) => {
                    let Some(arg_type_ident) = get_type_ident(arg_ty) else {
                        panic!("apply_provided_generics_fn_call: Failed to get ident for type {arg_ty:#?}");
                    };
                    let var_string = arg_type_ident.to_string();
                    let var_str = var_string.as_str();
                    if let Some(arg_type_var_type) = generic_vars.var_type(var_str) {
                        // This is a generic type variable.
                        match arg_type_var_type {
                            GenericVarType::TypeVariable => {
                                let Some(inst_type) = generic_vars.inst_types.get(var_str) else {
                                    panic!("Undefined instance type {var_str}")
                                };
                                self.param2arg.insert(
                                    param.to_string(),
                                    Some((GenericArgType::Type, inst_type.to_string())),
                                );
                            }
                            GenericVarType::ConstVariable => {
                                if let Some(inst_i32) = generic_vars.inst_i32.get(var_str) {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((GenericArgType::Type, (*inst_i32).to_string())),
                                    );
                                } else if let Some(inst_arr) = generic_vars.inst_array.get(var_str)
                                {
                                    self.param2arg.insert(
                                        param.to_string(),
                                        Some((
                                            GenericArgType::Type,
                                            format!("{{[{:?}]}}", inst_arr),
                                        )),
                                    );
                                } else {
                                    panic!("Undefined instance type {var_str}")
                                }
                            }
                            GenericVarType::LengthVariable => {
                                panic!(
                                    "Unexpected GenericVarType::LengthVariable {var_str}. \
                                Length variables can only be inferred from const generic arrays."
                                )
                            }
                        }
                    } else {
                        // This is a type (not a type variable).
                        self.param2arg.insert(
                            param.to_string(),
                            Some((GenericArgType::Type, var_str.to_string())),
                        );
                    };
                }
                GenericArgument::Const(c) => {
                    self.param2arg.insert(
                        param.to_string(),
                        Some((GenericArgType::Type, c.to_token_stream().to_string())),
                    );
                }
                _ => {}
            }
        }
    }

    /// Returns `true` if all generic parameters have been resolved.
    pub fn verify(&self) -> bool {
        // Check if computed and succeeded.
        for (_key, val) in &self.param2arg {
            if val.is_none() {
                return false;
            }
        }
        true
    }

    /// Builds a [`GenericVars`] from the inferred parameter-to-argument mapping.
    pub fn get_generic_vars_instance(
        &self,
        from_generic_args: &GenericVars,
        primitives: &HashMap<(String, String), ItemImpl>,
    ) -> GenericVars {
        // This function constructs an instance of GenericVars from a filled instance of self.param2arg.
        // If a value of self.param2arg is a key in from_generic_args, then it's a type variable.
        let mut to_generic_vars = GenericVars::empty_unchecked();

        // arg_map keys are target param names, and values are their values.
        for (name, v) in &self.param2arg {
            let Some((ast_name, ast_string)) = v else {
                panic!("Unexpected None value for key={name}");
            };
            match *ast_name {
                GenericArgType::Type => {
                    // Check if ast_string is a variable.
                    if let Some(inst_i32_val) = from_generic_args.inst_i32.get(ast_string) {
                        // This is a const generic i32
                        to_generic_vars.inst_i32.insert(name.clone(), *inst_i32_val);
                    } else if let Some(inst_arr_val) = from_generic_args.inst_array.get(ast_string)
                    {
                        // This is a const generic array
                        to_generic_vars
                            .inst_array
                            .insert(name.clone(), inst_arr_val.clone());
                        if let Some(generic_cga) = self.param2cga.get(name) {
                            let Type::Array(ty_arr) = generic_cga else {
                                panic!("Expected array type.")
                            };
                            if let Expr::Path(length_expr) = &ty_arr.len {
                                let length_var = length_expr
                                    .path
                                    .get_ident()
                                    .unwrap()
                                    .to_string()
                                    .to_string();
                                to_generic_vars.len2array.insert(length_var, name.clone());
                            }
                        }
                    // Check if it's a type parameter.
                    } else if let Some(inst_type_val) = from_generic_args.inst_types.get(ast_string)
                    {
                        if is_element_type(inst_type_val.as_str(), primitives) {
                            // This is a GenericParam::Type.
                            // TODO (hme): Need to support this in more places than here.
                            to_generic_vars
                                .inst_types
                                .insert(name.clone(), inst_type_val.to_string());
                        } else {
                            panic!("Failed to get generic arg instances for ({name}, {v:?}).");
                        }
                    // Check if it's a ptr of generic type param.
                    } else if let Some((is_mutable, element_type)) =
                        get_ptr_type_instance(ast_string, from_generic_args, primitives)
                    {
                        let instantiated_ptr = if is_mutable {
                            format!("* mut {element_type}")
                        } else {
                            format!("* const {element_type}")
                        };
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), instantiated_ptr);
                    // Check if it's an element type.
                    } else if is_element_type(ast_string.as_str(), primitives) {
                        // This is a concrete element type.
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), ast_string.to_string());
                    } else if is_element_type_ptr(ast_string.as_str(), primitives) {
                        // This is a ptr with concrete element type.
                        to_generic_vars
                            .inst_types
                            .insert(name.clone(), ast_string.to_string());
                    } else {
                        panic!("Failed to get generic arg instances for ({name}, {v:?}).");
                    }
                }
                GenericArgType::GenericConstExpr => {
                    let generic_arg =
                        syn::parse2::<GenericArgument>(ast_string.parse().unwrap()).unwrap();
                    if let Some(res) =
                        try_get_const_generic_from_generic_argument(&generic_arg, from_generic_args)
                    {
                        // These are args -> param pairs like:
                        // {[..., 128, ...]} -> {[..., CONST_PARAM, ...]}
                        // {[..., CONST_ARG, ...]} -> {[..., CONST_PARAM, ...]}
                        to_generic_vars.inst_i32.insert(name.clone(), res);
                    } else {
                        let Some(res) =
                            get_cga_from_generic_argument(&generic_arg, from_generic_args)
                        else {
                            unimplemented!("Unexpected param2arg pair ({name}, {v:?}).")
                        };
                        // These are args -> param pairs like:
                        // {[...]} -> CONST_ARRAY_PARAM
                        to_generic_vars.inst_array.insert(name.clone(), res);
                        if let Some(generic_cga) = self.param2cga.get(name) {
                            let Type::Array(ty_arr) = generic_cga else {
                                panic!("Expected array type.")
                            };
                            if let Expr::Path(length_expr) = &ty_arr.len {
                                let length_var = length_expr
                                    .path
                                    .get_ident()
                                    .unwrap()
                                    .to_string()
                                    .to_string();
                                to_generic_vars.len2array.insert(length_var, name.clone());
                            }
                        }
                    }
                }
            }
        }
        to_generic_vars
    }

    fn add_generic_args(&mut self, type_param: &syn::Type, type_arg: &syn::Type) -> () {
        // Adds generic arguments to arg_map.
        // arg_map maps generic parameters (present in arg_map upon initialization) to various GenericArgument patterns (see below).
        // Each key in arg_map specifies the set of generic parameters in a function / method signature.
        // add_generic_args is called by map_args_to_params, which enumerates over all input types in a function / method signature.
        // The inferred ast fragment for each key (generic parameter) must be identical to an already inferred generic parameter.
        // The values mapped to by arg_map are ast fragments, which may be one of "Type", "GenericArgument", or "Expr".
        // When an occurrence of a generic parameter is found in type_param, the corresponding pattern is recorded in arg_map
        // as a pair of strings "(ast_fragment, string_repr)", where ast fragment is one of "Type", "GenericArgument", or "Expr",
        // and string_repr is a string-based representation of the AST fragment corresponding to the pattern.

        // The procedure succeeds if all keys in the resulting arg_map are populated by non-None values.
        // arg_map can then  be used by the infer_type function to infer the type or const vars
        // occurring in the function signature.

        // Once generic arguments have been computed, this method can also be used to infer generic arguments to a function call.

        let (Some(mut param_generic_args), Some(mut arg_generic_args)) =
            (maybe_generic_args(type_param), maybe_generic_args(type_arg))
        else {
            // Check if type itself is a generic param.
            self.add_generic_type(type_param, type_arg);
            return;
        };
        strip_generic_args_lifetimes(&mut param_generic_args);
        strip_generic_args_lifetimes(&mut arg_generic_args);

        // println!("remap ident: {param_ident:#?}, {arg_ident:#?}");
        // println!("remap: {:?} to {:?}", arg_generic_args.to_token_stream().to_string(), param_generic_args.to_token_stream().to_string());

        // Make sure there are the same number of generic arguments.
        assert_eq!(
            arg_generic_args.args.len(),
            param_generic_args.args.len(),
            "{arg_generic_args:#?}\n!=\n{param_generic_args:#?}"
        );

        for i in 0..arg_generic_args.args.len() {
            let arg_arg = &arg_generic_args.args[i];
            let param_arg = &param_generic_args.args[i];
            // Supports:
            // E -> f32
            // *mut E -> *mut f32
            // {[..., CONST_PARAM, ...]} -> {[..., 128, ...]}
            // {[..., CONST_PARAM, ...]} -> {[..., CONST_ARG, ...]}
            // CONST_ARRAY_PARAM -> CONST_ARRAY_ARG
            // CONST_ARRAY_PARAM -> {[...]}
            // CONST_ARRAY_PARAM -> {[-1; 2]}
            // TODO (HME): Unclear if we need any of this.
            // CONST_PARAM -> CONST_ARG
            // CONST_PARAM -> 128
            match (arg_arg, param_arg) {
                (GenericArgument::Type(arg_type), GenericArgument::Type(param_type)) => {
                    self.add_generic_type(param_type, arg_type);
                    match (arg_type, param_type) {
                        (syn::Type::Path(_arg_type_path), syn::Type::Path(param_type_path)) => {
                            // Something like (Tensor<f32, ...>, Tensor<E, ...>)
                            let param_ident = &param_type_path
                                .path
                                .segments
                                .last()
                                .unwrap()
                                .ident
                                .to_string();
                            let arg_type_str = arg_type.to_token_stream().to_string();
                            if self.param2arg.contains_key(param_ident) {
                                let replaced_arg = self.param2arg.insert(
                                    param_ident.to_string(),
                                    Some((GenericArgType::Type, arg_type_str.to_string())),
                                );
                                if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                                    assert_eq!(arg, arg_type_str.to_string());
                                }
                            }
                        }
                        (syn::Type::Ptr(arg_type_ptr), syn::Type::Ptr(param_type_ptr)) => {
                            // Something like (PointerTile<*mut f32, ...>, PointerTile<*mut E, ...>)
                            let param_elem_ty = match get_type_ident(&*param_type_ptr.elem) {
                                Some(ident) => ident.to_string(),
                                None => panic!(
                                    "Unable to extract ident from pointer {param_type_ptr:#?}"
                                ),
                            };
                            let arg_type_str = arg_type_ptr.elem.to_token_stream().to_string();
                            if self.param2arg.contains_key(&param_elem_ty) {
                                let replaced_arg = self.param2arg.insert(
                                    param_elem_ty.to_string(),
                                    Some((GenericArgType::Type, arg_type_str.to_string())),
                                );
                                if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                                    assert_eq!(arg, arg_type_str.to_string());
                                }
                            }
                        }
                        (syn::Type::Reference(arg_ref), syn::Type::Reference(_param_ref)) => {
                            unimplemented!(
                                "get_generic_args: Type::Reference not supported: {:#?}",
                                arg_ref
                            );
                        }
                        _ => {}
                    }
                }
                (GenericArgument::Const(arg_const), GenericArgument::Type(param_type)) => {
                    match param_type {
                        syn::Type::Path(param_type_path) => {
                            // Something like (Tensor<E, {[...]}>, Tensor<E, CONST_ARRAY_PARAM>)
                            let param_ident = &param_type_path
                                .path
                                .segments
                                .last()
                                .unwrap()
                                .ident
                                .to_string();
                            if self.param2arg.contains_key(param_ident) {
                                let arg_const_str = &arg_const.to_token_stream().to_string();
                                let _replaced_arg = self.param2arg.insert(
                                    param_ident.to_string(),
                                    Some((
                                        GenericArgType::GenericConstExpr,
                                        arg_const_str.to_string(),
                                    )),
                                );
                                // TODO (hme): Confirm this was too strict.
                                // if let Some(Some((_arg_type, arg))) = _replaced_arg {
                                //     assert_eq!(arg, arg_const_str.to_string());
                                // }
                            }
                        }
                        _ => panic!("Unexpected generics {param_type:#?} {arg_const:#?}"),
                    }
                }
                (GenericArgument::Const(arg_const), GenericArgument::Const(param_const)) => {
                    // println!("expand GenericArgument::Const? {const_param:#?}");
                    match (arg_const, param_const) {
                        (Expr::Block(arg_expr), Expr::Block(param_expr)) => {
                            assert_eq!(arg_expr.block.stmts.len(), 1);
                            let Stmt::Expr(arg_stmt_expr, _) = &arg_expr.block.stmts[0] else {
                                panic!("Unexpected block expression.")
                            };
                            let Stmt::Expr(param_stmt_expr, _) = &param_expr.block.stmts[0] else {
                                panic!("Unexpected block expression.")
                            };
                            match (arg_stmt_expr, param_stmt_expr) {
                                (Expr::Array(arg_array_expr), Expr::Array(param_array_expr)) => {
                                    // Something like (Tensor<f32, {[...]}>, Tensor<E, {[...]}>)
                                    for i in 0..arg_array_expr.elems.iter().len() {
                                        let param_elem = &param_array_expr.elems[i];
                                        let param_var = param_elem.to_token_stream().to_string();
                                        if self.param2arg.contains_key(&param_var) {
                                            let arg_elem = &arg_array_expr.elems[i];
                                            let arg_val = match arg_elem {
                                                Expr::Lit(lit) => {
                                                    match &lit.lit {
                                                        Lit::Int(_int_lit) => arg_elem.to_token_stream().to_string(),
                                                        _ => unimplemented!("Unexpected array element {arg_elem:#?} in {arg_array_expr:#?}"),
                                                    }
                                                },
                                                Expr::Unary(_unary_expr) => {
                                                    arg_elem.to_token_stream().to_string()
                                                },
                                                Expr::Path(_path) => {
                                                    arg_elem.to_token_stream().to_string()
                                                },
                                                _ => unimplemented!("Unexpected array element {arg_elem:#?} in {arg_array_expr:#?}"),
                                            };
                                            // Skip inference from dynamic dims (-1): they
                                            // carry no information about the generic param.
                                            if arg_val == "- 1" || arg_val == "-1" {
                                                continue;
                                            }
                                            let replaced_arg = self.param2arg.insert(param_var.to_string(), Some((GenericArgType::GenericConstExpr, arg_val.to_string())));
                                            if let Some(Some((_arg_type, arg))) = replaced_arg {
                                                // Allow overriding a previous -1 (dynamic)
                                                // inference with a concrete value, but
                                                // two different concrete values conflict.
                                                if arg != "- 1" && arg != "-1" {
                                                    assert_eq!(arg, arg_val.to_string());
                                                }
                                            }
                                        }
                                    }
                                },
                                (_, Expr::Repeat(_param_expr)) => {
                                    // TODO (hme): Check that this is okay.
                                    // If param is comprised of variadic literals, then skip it.
                                }
                                _ => panic!("Unexpected block expression:\nparam=\n{param_stmt_expr:#?}\narg=\n{arg_stmt_expr:#?}")
                            }
                        }
                        _ => unimplemented!(
                            "Unsupported Const inference {param_const:#?} {arg_const:#?}"
                        ),
                    }
                }
                _ => {}
            }
        }
    }

    fn add_generic_type(&mut self, param_type: &Type, arg_type: &Type) {
        let arg_map = &mut self.param2arg;
        match (arg_type, param_type) {
            (syn::Type::Path(_arg_type_path), syn::Type::Path(param_type_path)) => {
                // Something like (Tensor<f32, ...>, Tensor<E, ...>)
                let param_ident = &param_type_path
                    .path
                    .segments
                    .last()
                    .unwrap()
                    .ident
                    .to_string();
                let arg_type_str = arg_type.to_token_stream().to_string();
                if arg_map.contains_key(param_ident) {
                    let _replaced_arg = arg_map.insert(
                        param_ident.to_string(),
                        Some((GenericArgType::Type, arg_type_str.to_string())),
                    );
                    // TODO (hme): Check that this is okay.
                    // if let Some(Some((_generic_arg_type, arg))) = _replaced_arg {
                    //     assert_eq!(arg, arg_type_str.to_string(), "param_type={param_type:#?},\narg_type={arg_type:#?},\narg_map={arg_map:#?}");
                    // }
                }
            }
            (syn::Type::Ptr(_arg_type_path), syn::Type::Path(param_type_path)) => {
                // Something like (PointerTile<*mut f32, ...>, PointerTile<P, ...>)
                let param_ident = &param_type_path
                    .path
                    .segments
                    .last()
                    .unwrap()
                    .ident
                    .to_string();
                let arg_type_str = arg_type.to_token_stream().to_string();
                if arg_map.contains_key(param_ident) {
                    let replaced_arg = arg_map.insert(
                        param_ident.to_string(),
                        Some((GenericArgType::Type, arg_type_str.to_string())),
                    );
                    if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                        assert_eq!(arg, arg_type_str.to_string());
                    }
                }
            }
            (syn::Type::Ptr(arg_type_ptr), syn::Type::Ptr(param_type_ptr)) => {
                // Something like (PointerTile<*mut f32, ...>, PointerTile<*mut E, ...>)
                let param_elem_ty = match get_type_ident(&*param_type_ptr.elem) {
                    Some(ident) => ident.to_string(),
                    None => panic!("Unable to extract ident from pointer {param_type_ptr:#?}"),
                };
                let arg_type_str = arg_type_ptr.elem.to_token_stream().to_string();
                if arg_map.contains_key(&param_elem_ty) {
                    let replaced_arg = arg_map.insert(
                        param_elem_ty.to_string(),
                        Some((GenericArgType::Type, arg_type_str.to_string())),
                    );
                    if let Some(Some((_generic_arg_type, arg))) = replaced_arg {
                        assert_eq!(arg, arg_type_str.to_string());
                    }
                }
            }
            (syn::Type::Reference(arg_ref), syn::Type::Reference(_param_ref)) => {
                unimplemented!(
                    "get_generic_args: Type::Reference not supported: {:#?}",
                    arg_ref
                );
            }
            _ => {}
        }
    }

    pub fn infer_type(&self, ty: &syn::Type, _generic_vars: &GenericVars) -> syn::Type {
        let arg_map = &self.param2arg;
        // println!("Infer generic args for {} using \n {arg_map:#?}", ty.to_token_stream().to_string());
        let Some(mut result_args) = maybe_generic_args(&ty) else {
            // Is it a generic arg itself?
            // TODO (hme): *Really* need to make this recursive and just call with the following types.
            let mut result = ty.clone();
            match &mut result {
                syn::Type::Path(param_type_path) => {
                    // This is a type var or concrete type.
                    let param_ident_str = param_type_path
                        .path
                        .segments
                        .last()
                        .unwrap()
                        .ident
                        .to_string();
                    match arg_map.get(param_ident_str.as_str()) {
                        None => {
                            // This is not a generic parameter.
                        }
                        Some(None) => {
                            panic!("Failed to infer generic parameter {param_ident_str} \n{arg_map:#?}")
                        }
                        Some(Some((GenericArgType::Type, target_ty))) => {
                            result = syn::parse2::<Type>(target_ty.parse().unwrap()).unwrap();
                        }
                        Some(Some((GenericArgType::GenericConstExpr, target_ty))) => {
                            result = syn::parse2::<Type>(target_ty.parse().unwrap()).unwrap();
                        }
                    }
                }
                syn::Type::Ptr(param_type_ptr) => {
                    // This is a pointer with type var or concrete type for element type.
                    match *param_type_ptr.elem.clone() {
                        Type::Path(type_path) => {
                            let param_ident =
                                type_path.path.segments.last().unwrap().ident.to_string();
                            match arg_map.get(param_ident.as_str()) {
                                None => {
                                    // This is not a generic parameter.
                                }
                                Some(None) => {
                                    panic!("Failed to infer generic parameter {param_ident}")
                                }
                                Some(Some((GenericArgType::Type, target_ty))) => {
                                    *param_type_ptr.elem =
                                        syn::parse2::<Type>(target_ty.parse().unwrap()).unwrap();
                                }
                                Some(Some((arg_type, _arg))) => {
                                    panic!("Unexpected arg type {arg_type:#?}")
                                }
                            };
                        }
                        _ => panic!("Unable to extract ident from pointer {param_type_ptr:#?}"),
                    }
                }
                syn::Type::Array(array_ty) => {
                    // Something like [T; N]
                    let syn::Type::Path(elem) = &mut *array_ty.elem else {
                        panic!("Unexpected element type for array {array_ty:#?}")
                    };
                    // This is a type var or concrete type.
                    let elem_ident_str = elem.path.segments.last().unwrap().ident.to_string();
                    match arg_map.get(elem_ident_str.as_str()) {
                        None => {} // This is not a generic parameter.
                        Some(None) => panic!(
                            "Failed to infer generic parameter {elem_ident_str} \n{arg_map:#?}"
                        ),
                        Some(Some((GenericArgType::Type, target_ty))) => {
                            *elem = syn::parse2::<TypePath>(target_ty.parse().unwrap()).unwrap()
                        }
                        Some(Some((GenericArgType::GenericConstExpr, _target_ty))) => {
                            panic!("Unexpected element type for array {array_ty:#?}")
                        }
                    }
                    if let Expr::Path(len_path_expr) = array_ty.len.clone() {
                        // // This is a type var or concrete type.
                        let len_ident_str = len_path_expr
                            .path
                            .segments
                            .last()
                            .unwrap()
                            .ident
                            .to_string();
                        match arg_map.get(len_ident_str.as_str()) {
                            None => {} // This is not a generic parameter.
                            Some(None) => panic!(
                                "Failed to infer generic parameter {len_ident_str} \n{arg_map:#?}"
                            ),
                            Some(Some((GenericArgType::Type, _target_ty))) => {
                                panic!("Unexpected length type for array {array_ty:#?}")
                            }
                            Some(Some((GenericArgType::GenericConstExpr, target_ty))) => {
                                array_ty.len =
                                    syn::parse2::<Expr>(target_ty.parse().unwrap()).unwrap()
                            }
                        }
                    } else {
                        // Nothing to do.
                    }
                }
                _ => {}
            }
            return result;
        };

        // for arg in param_generic_args.args.iter_mut() {
        //     println!("Infer generic args {:?}", arg);
        // }
        for arg in result_args.args.iter_mut() {
            match arg {
                GenericArgument::Type(param_type) => {
                    match param_type {
                        syn::Type::Path(param_type_path) => {
                            // This is a type var or concrete type.
                            let param_ident_str = param_type_path
                                .path
                                .segments
                                .last()
                                .unwrap()
                                .ident
                                .to_string();
                            match arg_map.get(param_ident_str.as_str()) {
                                None => {
                                    // This is not a generic parameter.
                                }
                                Some(None) => {
                                    panic!("Failed to infer generic parameter {param_ident_str} \n{arg_map:#?}")
                                }
                                Some(Some((GenericArgType::Type, target_ty))) => {
                                    *arg =
                                        syn::parse2::<GenericArgument>(target_ty.parse().unwrap())
                                            .unwrap();
                                }
                                Some(Some((GenericArgType::GenericConstExpr, target_ty))) => {
                                    *arg =
                                        syn::parse2::<GenericArgument>(target_ty.parse().unwrap())
                                            .unwrap();
                                }
                            }
                        }
                        syn::Type::Ptr(param_type_ptr) => {
                            // This is a pointer with type var or concrete type for element type.
                            match *param_type_ptr.elem.clone() {
                                Type::Path(type_path) => {
                                    let param_ident =
                                        type_path.path.segments.last().unwrap().ident.to_string();
                                    match arg_map.get(param_ident.as_str()) {
                                        None => {
                                            // This is not a generic parameter.
                                        }
                                        Some(None) => {
                                            panic!(
                                                "Failed to infer generic parameter {param_ident}"
                                            )
                                        }
                                        Some(Some((GenericArgType::Type, target_ty))) => {
                                            *param_type_ptr.elem =
                                                syn::parse2::<Type>(target_ty.parse().unwrap())
                                                    .unwrap();
                                        }
                                        Some(Some((arg_type, _arg))) => {
                                            panic!("Unexpected arg type {arg_type:#?}")
                                        }
                                    };
                                }
                                _ => panic!(
                                    "Unable to extract ident from pointer {param_type_ptr:#?}"
                                ),
                            }
                        }
                        syn::Type::Reference(_param_ref) => {}
                        _ => {}
                    }
                }
                GenericArgument::Const(param_const) => {
                    // This is a literal or array expression.
                    // println!("expand GenericArgument::Const? {const_param:#?}");
                    match param_const {
                        Expr::Block(param_expr) => {
                            assert_eq!(param_expr.block.stmts.len(), 1);
                            let Stmt::Expr(param_stmt_expr, _) = &mut param_expr.block.stmts[0]
                            else {
                                panic!("Unexpected block expression.")
                            };
                            match param_stmt_expr {
                                Expr::Array(param_array_expr) => {
                                    for i in 0..param_array_expr.elems.iter().len() {
                                        let param_elem = &mut param_array_expr.elems[i];
                                        let param_var = param_elem.to_token_stream().to_string();
                                        match arg_map.get(param_var.as_str()) {
                                            None => {
                                                // This is not a generic parameter.
                                            }
                                            Some(None) => {
                                                panic!(
                                                    "Failed to infer generic parameter {param_var}"
                                                )
                                            }
                                            Some(Some((
                                                GenericArgType::GenericConstExpr,
                                                target_expr,
                                            ))) => {
                                                *param_elem = syn::parse2::<Expr>(
                                                    target_expr.parse().unwrap(),
                                                )
                                                .unwrap();
                                            }
                                            Some(Some((arg_type, _arg))) => {
                                                panic!("Unexpected arg type {arg_type:#?}")
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        Expr::Path(param_path) => {
                            unimplemented!("Expr::Path not supported {:#?}", param_path);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        let mut return_ty = ty.clone();
        match &mut return_ty {
            Type::Path(type_path) => {
                let last_seg = type_path.path.segments.last_mut().unwrap();
                last_seg.arguments = PathArguments::AngleBracketed(result_args);
            }
            Type::Reference(ref_type) => match &mut *ref_type.elem {
                Type::Path(type_path) => {
                    let last_seg = type_path.path.segments.last_mut().unwrap();
                    last_seg.arguments = PathArguments::AngleBracketed(result_args);
                }
                _ => panic!("Unexpected ref type {:#?}", ref_type),
            },
            _ => panic!("get_ident_generic_args: Unexpected type {:#?}", return_ty),
        }
        return_ty
    }
}

pub fn get_cga_from_type(ty: &syn::Type, generic_args: &GenericVars) -> Option<Vec<i32>> {
    // We assume this is a variadic type.
    let (_type_ident, type_generic_args) = get_ident_generic_args(ty);
    let mut shape: Option<Vec<i32>> = None;
    for type_generic_arg in &type_generic_args.args {
        let res = get_cga_from_generic_argument(type_generic_arg, generic_args);
        if res.is_some() {
            shape = Some(res.unwrap());
        }
    }
    shape
}

/// Attempts to extract a const generic expression from a single generic argument.
pub fn try_get_const_generic_from_generic_argument(
    generic_arg: &GenericArgument,
    generic_args: &GenericVars,
) -> Option<i32> {
    let mut result: Option<i32> = None;
    match generic_arg {
        GenericArgument::Type(type_param) => {
            match type_param {
                syn::Type::Path(type_path) => {
                    let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
                    // println!("get_variadic_type_args: Type::Path: {}", last_ident);
                    if generic_args.inst_i32.contains_key(&last_ident) {
                        // This is something like N for const generic N: i32.
                        result = Some(generic_args.inst_i32.get(&last_ident).unwrap().clone());
                    }
                    // If it's anything else, then return None.
                }
                _ => {}
            }
        }
        GenericArgument::Const(const_param) => {
            // println!("expand GenericArgument::Const? {const_param:#?}");
            match const_param {
                Expr::Lit(lit) => {
                    let Lit::Int(int_lit) = &lit.lit else {
                        panic!("Expected int literal, got {:#?}", lit)
                    };
                    // This is something like 32 in Tile<E, {[32]}>
                    // TODO (hme): Add a test for this.
                    result = Some(int_lit.base10_parse().unwrap());
                }
                _ => {}
            }
        }
        _ => {}
    }
    result
}

/// Extracts a const generic array value from a generic argument, resolving variables.
pub fn get_cga_from_generic_argument(
    generic_arg: &GenericArgument,
    generic_args: &GenericVars,
) -> Option<Vec<i32>> {
    let mut shape: Option<Vec<i32>> = None;
    match generic_arg {
        GenericArgument::Type(type_param) => {
            match type_param {
                syn::Type::Path(type_path) => {
                    // This must be a CGA, or it will fail.
                    let last_ident = type_path.path.segments.last().unwrap().ident.to_string();
                    // println!("get_variadic_type_args: Type::Path: {}", last_ident);
                    if generic_args.inst_array.contains_key(&last_ident) {
                        // This is something like Shape<D> for const generic array D: [i32; N].
                        let array_instance = generic_args.inst_array.get(&last_ident).unwrap();
                        if shape.is_some() {
                            panic!("Unexpected array arg: {last_ident:#?}")
                        }
                        shape = Some(array_instance.clone());
                    } else if generic_args.inst_i32.contains_key(&last_ident) {
                        // This is something like N for const generic N: i32.
                        // This should have been handled by
                        // try_get_const_generic_from_generic_argument.
                        unimplemented!(
                            "Unexpected const arg {last_ident} for type {type_param:#?}"
                        );
                    } else {
                        unimplemented!("Failed to get cga for {type_param:#?}");
                    }
                }
                _ => {}
            }
        }
        GenericArgument::Const(const_param) => {
            // println!("expand GenericArgument::Const? {const_param:#?}");
            match const_param {
                Expr::Block(block_expr) => {
                    // This is something like Tensor<E, {[...]}>
                    assert_eq!(block_expr.block.stmts.len(), 1);
                    let statement = &block_expr.block.stmts[0];
                    let Stmt::Expr(statement_expr, _) = statement else {
                        panic!("Unexpected block expression.")
                    };
                    match statement_expr {
                        Expr::Array(array_expr) => {
                            // This is something like Tensor<E, {[1, 2, -1]}>
                            let mut _shape: Vec<i32> = vec![];
                            for elem in &array_expr.elems {
                                _shape.push(parse_expr_as_i32(elem, generic_args));
                            }
                            shape = Some(_shape);
                        }
                        Expr::Repeat(repeat_expr) => {
                            // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                            let thing_to_repeat =
                                parse_expr_as_i32(&repeat_expr.expr, generic_args);
                            match &*repeat_expr.len {
                                Expr::Path(len_path) => {
                                    // This is something like Tensor<E, {[-1; N]}>
                                    let num_rep_var = len_path.to_token_stream().to_string();
                                    if !generic_args.get_i32(&num_rep_var).is_some() {
                                        panic!(
                                            "Expected instance for generic argument {}",
                                            num_rep_var
                                        );
                                    }
                                    let num_rep = generic_args.get_i32(&num_rep_var).unwrap();
                                    shape = Some(vec![thing_to_repeat; num_rep as usize]);
                                }
                                Expr::Lit(len_lit) => {
                                    // This is something like Tensor<E, {[-1; 3]}>
                                    let num_rep: u32 = len_lit
                                        .to_token_stream()
                                        .to_string()
                                        .parse::<u32>()
                                        .unwrap();
                                    shape = Some(vec![thing_to_repeat; num_rep as usize]);
                                }
                                _ => {
                                    unimplemented!("Unexpected repeat expression: {repeat_expr:#?}")
                                }
                            }
                        }
                        _ => panic!("Unexpected block expression."),
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    shape
}

pub fn parse_expr_as_i32(expr: &Expr, generic_args: &GenericVars) -> i32 {
    match expr {
        Expr::Lit(_lit) => parse_signed_literal_as_i32(expr),
        Expr::Unary(_unary_expr) => parse_signed_literal_as_i32(expr),
        Expr::Path(path) => {
            let ident = get_ident_from_path_expr(path);
            match generic_args.inst_i32.get(ident.to_string().as_str()) {
                Some(val) => return *val,
                None => panic!("Undefined generic parameter {ident}"),
            }
        }
        _ => unimplemented!("Unexpected expression {expr:#?}"),
    }
}
