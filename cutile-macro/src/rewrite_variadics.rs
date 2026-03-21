/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Variadic type expansion and rewriting.
//!
//! This module implements the core transformation that enables rank-polymorphic types
//! in cuTile Rust. It rewrites code using variadic types (types with const generic arrays)
//! into concrete rank-specific versions.
//!
//! ## Overview
//!
//! Rust doesn't natively support types like `Tile<E, const D: [i32; N]>` where `N` is
//! variable. This module solves that problem by:
//!
//! 1. **Expanding definitions** - Generating concrete types for each rank (1-4)
//! 2. **Rewriting calls** - Updating function/method calls to use rank-specific versions
//! 3. **Tracking CGAs** - Managing const generic array parameters throughout expansion
//! 4. **Inferring types** - Determining the correct rank from usage context
//!
//! ## Transformation Examples
//!
//! ### Struct Expansion
//!
//! ```rust,ignore
//! // Input:
//! #[cuda_tile::variadic_struct(N=4)]
//! pub struct Tile<E, const D: [i32; N]> { }
//!
//! // Output (4 structs):
//! pub struct Tile_1<E, const D: [i32; 1]> { }
//! pub struct Tile_2<E, const D: [i32; 2]> { }
//! pub struct Tile_3<E, const D: [i32; 3]> { }
//! pub struct Tile_4<E, const D: [i32; 4]> { }
//! ```
//!
//! ### Function Expansion
//!
//! ```rust,ignore
//! // Input:
//! #[cuda_tile::variadic_op(N=4)]
//! pub fn load_tile<E, const S: [i32; N]>(tensor: &mut Tensor<E, S>) -> Tile<E, S> {
//!     // ...
//! }
//!
//! // Output (4 functions):
//! pub fn load_tile_1<E, const S: [i32; 1]>(tensor: &mut Tensor_1<E, S>) -> Tile_1<E, S> { }
//! pub fn load_tile_2<E, const S: [i32; 2]>(tensor: &mut Tensor_2<E, S>) -> Tile_2<E, S> { }
//! pub fn load_tile_3<E, const S: [i32; 3]>(tensor: &mut Tensor_3<E, S>) -> Tile_3<E, S> { }
//! pub fn load_tile_4<E, const S: [i32; 4]>(tensor: &mut Tensor_4<E, S>) -> Tile_4<E, S> { }
//! ```
//!
//! ### Call Site Rewriting
//!
//! ```rust,ignore
//! // Input:
//! let tile: Tile<f32, {[128]}> = load_tile_mut(&mut tensor);
//!
//! // Rewritten to:
//! let tile: Tile_1<f32, 128> = load_tile_1(&mut tensor);
//! ```
//!
//! ## Const Generic Array (CGA) Tracking
//!
//! The module tracks CGA parameters as code is transformed:
//!
//! - **Static arrays**: `{[128, 64]}` - all dimensions known at compile time
//! - **Mixed arrays**: `{[-1, 64]}` - some dimensions dynamic, some static
//! - **Parameter arrays**: `{[N, M]}` - dimensions from generic parameters
//!
//! ## Type Inference
//!
//! When rewriting expressions, the rank is inferred from:
//!
//! - Explicit type annotations
//! - CGA literal expressions (e.g., `{[128, 64]}` → rank 2)
//! - Function parameter types
//! - Return type constraints
//!
//! ## Implementation Strategy
//!
//! The rewriter uses a visitor pattern to traverse and transform:
//!
//! - Type references in signatures and expressions
//! - Function and method calls
//! - Generic argument lists
//! - Pattern matching and variable bindings

use crate::{
    error::{syn_err, Error},
    types::{
        concrete_name, get_variadic_function_suffix, get_variadic_method_data,
        get_variadic_op_data, get_variadic_trait_type_data, get_variadic_type_data,
        ConstGenericArrayType, DimType, VariadicOpData, VariadicTypeData,
    },
};
use cutile_compiler::syn_utils::*;
use cutile_compiler::train_map::TrainMap;
use cutile_compiler::types::parse_signed_literal_as_i32;
use proc_macro2::{Ident, Span, TokenTree};
use quote::ToTokens;
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use syn::{
    parse_quote, spanned::Spanned, AngleBracketedGenericArguments, Expr, ExprCall, ExprMethodCall,
    ExprPath, FnArg, GenericArgument, GenericParam, Generics, ImplItem, ImplItemFn, Item, ItemFn,
    ItemImpl, ItemStruct, ItemTrait, Member, Pat, Path, PathArguments, PathSegment, ReturnType,
    Signature, Stmt, TraitItem, Type,
};

/// Looks up metadata for a trait method implementation.
///
/// Returns the operation name, variadic type data, and variadic operation data
/// for trait methods like `broadcast` that are called on primitive types.
/// Looks up metadata for a trait method implementation on a possibly-primitive receiver.
pub fn get_variadic_trait_impl_meta_data(
    maybe_primitive: &str,
    method_name: &str,
) -> Result<Option<(&'static str, VariadicTypeData, VariadicOpData)>, Error> {
    Ok(
        match get_variadic_trait_type_data(maybe_primitive, method_name) {
            Some(vtd) => match get_variadic_method_data(&vtd, method_name)? {
                Some((op_name, vod)) => Some((op_name, vtd, vod)),
                None => None,
            },
            None => None,
        },
    )
}

/// Looks up metadata for a method call on a variadic type.
///
/// Returns the operation name, type metadata, and operation metadata if the
/// receiver type is variadic and the method exists.
/// Looks up metadata for a method call on a variadic type receiver.
pub fn get_variadic_method_meta_data(
    receiver_ty: &Type,
    method_name: &str,
) -> Result<Option<(&'static str, VariadicTypeData, VariadicOpData)>, Error> {
    Ok(match get_vtd(receiver_ty)? {
        Some(vtd) => match get_variadic_method_data(&vtd, method_name)? {
            Some((op_name, vod)) => Some((op_name, vtd, vod)),
            None => None,
        },
        None => None,
    })
}

/// Extracts the identifier string from a simple path expression.
///
/// Attempts to extract the function or variable name from an expression that
/// is a simple path (single segment). Returns `None` if the expression is not
/// a path or has multiple segments.
///
/// ## Parameters
///
/// - `maybe_path_expr`: Expression that might be a simple path
///
/// ## Returns
///
/// - `Some(String)`: The identifier name if expr is a single-segment path
/// - `None`: If expr is not a path or has multiple segments
///
/// ## Panics
///
/// Panics if the expression is a path with more than one segment (e.g., `module::function`).
fn try_get_path_expr_ident_str(maybe_path_expr: &Expr) -> Result<Option<String>, Error> {
    match maybe_path_expr {
        Expr::Path(path_expr) => {
            if path_expr.path.segments.len() != 1 {
                return Err(syn_err(
                    path_expr.path.span(),
                    &format!(
                        "Expected single-segment path, got: {:?}",
                        path_expr.path.segments.to_token_stream().to_string()
                    ),
                ));
            }
            let fn_name = path_expr.path.segments[0].ident.to_string();
            Ok(Some(fn_name))
        }
        _ => Ok(None),
    }
}

/// Extracts variadic operation metadata from a function call expression.
///
/// Attempts to extract the function name from a call expression and looks up
/// its variadic operation metadata. This is used during macro expansion to
/// determine if a function call needs variadic rewriting.
///
/// ## Parameters
///
/// - `expr`: A function call expression to analyze
///
/// ## Returns
///
/// - `Some(VariadicOpData)`: Metadata if the called function is a variadic operation
/// - `None`: If the function is not variadic or the name cannot be extracted
fn get_vod_from_call(expr: &mut ExprCall) -> Result<Option<VariadicOpData>, Error> {
    let name = match &*expr.func {
        Expr::Path(path_expr) => {
            if path_expr.path.segments.is_empty() {
                return Ok(None);
            } else {
                let fn_name = path_expr
                    .path
                    .segments
                    .last()
                    .ok_or_else(|| syn_err(path_expr.span(), "Expected at least one path segment"))?
                    .ident
                    .to_string();
                Some(fn_name)
            }
        }
        _ => None,
    };
    Ok(match name {
        Some(name) => get_variadic_op_data(name.as_str()),
        None => None,
    })
}

/// Extracts variadic type metadata from a Rust type.
///
/// Analyzes a type and returns its variadic type metadata if it corresponds to
/// a rank-polymorphic type like `Tile`, `Tensor`, `Shape`, etc. Handles both
/// direct types and references.
///
/// ## Parameters
///
/// - `ty`: The type to analyze
///
/// ## Returns
///
/// - `Some(VariadicTypeData)`: Metadata if the type is variadic
/// - `None`: If the type is not variadic or cannot be analyzed
fn get_vtd(ty: &Type) -> Result<Option<VariadicTypeData>, Error> {
    Ok(match ty {
        Type::Path(ty_path) => {
            let last_seg = ty_path.path.segments.last();
            match last_seg {
                Some(seg) => get_variadic_type_data(seg.ident.to_string().as_str()),
                None => None,
            }
        }
        Type::Reference(ref_type) => {
            get_vtd(&ref_type.elem)?
            // unimplemented!("get_vtd Type::Reference not implemented: {:#?}", ref_type)
        }
        _ => None,
    })
}

/// Extracts the type identifier and generic arguments from a variadic type.
///
/// Given a type and its variadic type metadata, extracts the base type identifier
/// (e.g., `Tile`, `Tensor`) and its angle-bracketed generic arguments. Handles
/// both direct types and references.
///
/// ## Parameters
///
/// - `ty`: The type to analyze
/// - `vtd`: The variadic type metadata for this type
///
/// ## Returns
///
/// A tuple of (type_identifier, generic_arguments)
///
/// ## Panics
///
/// Panics if the type identifier doesn't match the expected name from `vtd` or
/// if the type doesn't have the expected generic argument structure.
fn get_ident_generic_args(
    ty: &Type,
    vtd: &VariadicTypeData,
) -> Result<(Ident, AngleBracketedGenericArguments), Error> {
    match ty {
        Type::Path(type_path) => {
            let result_type = type_path.clone();
            let maybe_last_seg =
                result_type.path.segments.last().ok_or_else(|| {
                    syn_err(type_path.span(), "Expected at least one path segment")
                })?;
            let last_seg = maybe_last_seg.clone();
            if last_seg.ident != vtd.name {
                return Err(syn_err(
                    last_seg.ident.span(),
                    &format!(
                        "get_ident_generic_args: Expected type '{}', got '{}'",
                        vtd.name, last_seg.ident
                    ),
                ));
            }
            match last_seg.arguments {
                // The type takes a const generic array as a type param:
                // f(..., shape: Shape<D>) -> ()
                PathArguments::AngleBracketed(type_params) => {
                    // This is a type of the form T<...>
                    Ok((last_seg.ident.clone(), type_params.clone()))
                }
                _ => Err(syn_err(
                    type_path.span(),
                    "Unexpected generic arguments",
                )),
            }
        }
        Type::Reference(ref_type) => get_ident_generic_args(&ref_type.elem, vtd),
        _ => Err(syn_err(ty.span(), "Unexpected type")),
    }
}

/// Resolves a variadic operation call to its concrete rank-specialized version.
///
/// Takes a variadic operation identifier and infers the concrete function name
/// (e.g., `reshape` → `reshape_2_3`) based on the input and output types. Also
/// infers the output type if not explicitly provided.
///
/// ## Parameters
///
/// - `op_ident`: The base operation identifier (e.g., `reshape`)
/// - `input_types`: Types of the function arguments (Some if known, None if unknown)
/// - `output_type`: Expected output type, if known
/// - `const_instances`: Current const generic array instances in scope
/// - `disable_output_inference`: If true, don't attempt to infer output type
///
/// ## Returns
///
/// A tuple of (concrete_identifier, inferred_output_type)
///
/// ## Panics
///
/// Panics if type inference fails due to insufficient type information.
fn get_concrete_op_ident_from_types(
    op_ident: &Ident,
    input_types: &Vec<Option<Type>>,
    output_type: Option<Type>,
    const_instances: &ConstInstances,
    disable_output_inference: bool,
) -> Result<(Ident, Option<Type>), Error> {
    let vod = get_variadic_op_data(op_ident.to_string().as_str());
    if vod.is_none() {
        return Ok((op_ident.clone(), output_type));
    }
    let vod = vod.unwrap();
    get_concrete_op_or_method_ident_from_types(
        vod,
        op_ident,
        input_types,
        output_type,
        const_instances,
        disable_output_inference,
    )
}

/// Core type inference and desugaring logic for variadic operations and methods.
///
/// This is the main workhorse function that performs type inference and generates
/// the concrete rank-specialized identifier for variadic operations. It:
/// 1. Analyzes input types to extract const generic array dimensions
/// 2. Validates dimension consistency across inputs
/// 3. Infers output type from input types when possible
/// 4. Generates the concrete function name with rank suffix
///
/// ## Parameters
///
/// - `vod`: Variadic operation metadata describing the type signature
/// - `op_or_method_ident`: Base identifier (function or method name)
/// - `input_types`: Types of the function arguments (Some if known, None if unknown)
/// - `output_type`: Expected output type, if known
/// - `const_instances`: Current const generic array instances in scope
/// - `disable_output_inference`: If true, don't attempt to infer output type
///
/// ## Returns
///
/// A tuple of (concrete_identifier, inferred_output_type)
///
/// ## Type Inference Algorithm
///
/// The function works in three phases:
/// 1. **Input Analysis**: Extracts CGA dimensions from each input type
/// 2. **Dimension Resolution**: Ensures all CGAs with same name have same length
/// 3. **Output Inference**: Constructs output type using inferred dimensions
///
/// ## Panics
///
/// Panics if:
/// - Type information is insufficient for inference
/// - Dimension mismatches occur (e.g., `Tile<f32, {[2,3]}>` and `Tile<f32, {[2,4]}>`)
/// - Expected types don't match actual types
fn get_concrete_op_or_method_ident_from_types(
    vod: VariadicOpData,
    op_or_method_ident: &Ident,
    input_types: &Vec<Option<Type>>,
    output_type: Option<Type>,
    const_instances: &ConstInstances,
    disable_output_inference: bool,
) -> Result<(Ident, Option<Type>), Error> {
    // This function outputs desugared calls to variadic functions, and their corresponding return type.
    // vod provides information about the structure of the variadic call.
    // If the output_type is not specified,
    // we attempt to "fill in the blanks" from the given input types.
    // If that can't be done, then we ask the user to specify the output type
    // by binding the call to a variable.

    let mut vod_cga_name_to_context_cga_name = HashMap::<&str, Option<String>>::new();
    let mut const_length_values = HashMap::<&str, u32>::new();

    let mut missing_idx = vec![];
    let mut missing_types = vec![];
    for (idx, expected_type_name, vod_cga_var_names) in vod.input_map {
        // Go through the input map and attempt to map from VOD -
        let Some(ty) = &input_types[idx] else {
            missing_idx.push(idx);
            missing_types.push(expected_type_name);
            continue;
        };
        let Some(vtd) = get_vtd(ty)? else {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("Unable to infer type for argument {idx} for call to {op_or_method_ident}. Expected {expected_type_name}. Required by calls to variadic functions and methods."),
            ));
        };
        // Get the cga_instances from const_instances. These are ordered.
        // Match them in order to the const var structure of the VariadicOpData type.
        let Some(cga_instances) = get_cga_type(ty, const_instances)? else {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Unable to get cga instances for type: {}", ty.to_token_stream()),
            ));
        };
        // This is a variadic type with cga instances.
        let type_name = vtd.name;
        if expected_type_name != type_name {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Unexpected positional argument type: {:#?}", (idx, ty.to_token_stream().to_string())),
            ));
        }
        if vod_cga_var_names.len() != cga_instances.n.len() {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Expected {} cga instances for {type_name}, got {:?}.", vod_cga_var_names.len(), cga_instances.n),
            ));
        }
        for i in 0..vod_cga_var_names.len() {
            let cga_var_name = vod_cga_var_names[i];
            let cga_var_length_var = vod.cga_map.get(cga_var_name).ok_or_else(|| {
                syn_err(
                    op_or_method_ident.span(),
                    &format!("Missing cga_map entry for '{cga_var_name}'"),
                )
            })?;
            let cga_var_length = cga_instances.n[i];
            vod_cga_name_to_context_cga_name
                .insert(cga_var_name, cga_instances.cga_arg_strings[i].clone());
            if const_length_values.contains_key(cga_var_length_var) {
                // If we've already recorded this cga instance, make sure other instances are equivalent.
                let current_var_length = *const_length_values.get(cga_var_length_var).unwrap();
                if current_var_length != cga_var_length {
                    return Err(syn_err(
                        op_or_method_ident.span(),
                        &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): CGA instance var length mismatch. Expected {current_var_length} but got {cga_var_length} for cga {cga_var_name}."),
                    ));
                }
            } else {
                const_length_values.insert(cga_var_length_var, cga_var_length);
            }
        }
    }

    // The result of the above procedure is an instantiation of as many const generic length variables as possible.
    // The keys of const_length_values are the length values in "VOD space," or the const_length_vars for the corresponding variadic op data entry.

    if const_length_values.len() > vod.const_length_vars.len() {
        return Err(syn_err(
            op_or_method_ident.span(),
            &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Unexpected number of cga instances: {:#?} ", const_length_values),
        ));
    } else if const_length_values.len() < vod.const_length_vars.len() {
        // Try to get the last cga instance from the output type.
        if output_type.is_none() {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("Unable to infer call to {}. Try binding it to a statically typed variable. \nDebug info:\n const_length_values={:#?}, vod.const_length_vars={:#?}",
                    op_or_method_ident,
                    const_length_values,
                    vod.const_length_vars),
            ));
        }
        let output_type = output_type.clone().unwrap();

        let maybe_vtd = get_vtd(&output_type)?;
        if maybe_vtd.is_none() {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!(
                    "Unable to infer call to {}. Try binding it to a statically typed variable.",
                    op_or_method_ident
                ),
            ));
        }
        let vtd = maybe_vtd.unwrap();
        let cga_instances = get_cga_type(&output_type, const_instances)?;
        if cga_instances.is_none() {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Unable to get cga instances for output type: {}", output_type.to_token_stream()),
            ));
        }
        let cga_instances = cga_instances.unwrap();

        let (expected_type_name, vod_cga_var_names) = vod.output_map;
        if expected_type_name != vtd.name {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Unexpected output type: {}", output_type.to_token_stream()),
            ));
        }
        if vod_cga_var_names.len() != cga_instances.n.len() {
            return Err(syn_err(
                op_or_method_ident.span(),
                &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): Expected {} cga instances, got {}.", vod_cga_var_names.len(), cga_instances.n.len()),
            ));
        }
        for i in 0..vod_cga_var_names.len() {
            let cga_var_name = &vod_cga_var_names[i];
            let cga_var_length_var = vod.cga_map.get(cga_var_name).ok_or_else(|| {
                syn_err(
                    op_or_method_ident.span(),
                    &format!("Missing cga_map entry for '{cga_var_name}'"),
                )
            })?;
            let cga_var_length = cga_instances.n[i];
            if const_length_values.contains_key(cga_var_length_var) {
                // If we've already recorded this cga instance, make sure other instances are equivalent.
                if *const_length_values.get(cga_var_length_var).unwrap() != cga_var_length {
                    return Err(syn_err(
                        op_or_method_ident.span(),
                        &format!("get_concrete_op_ident_from_types({op_or_method_ident}, ...): CGA instance var length mismatch for output type."),
                    ));
                }
            } else {
                const_length_values.insert(cga_var_length_var, cga_var_length);
            }
        }
    }

    if const_length_values.len() != vod.const_length_vars.len() {
        return Err(syn_err(
            op_or_method_ident.span(),
            &format!("Unable to infer type for argument(s) {missing_idx:?} for call to {op_or_method_ident}. Expected {missing_types:?}. Required by calls to variadic functions and methods."),
        ));
    }

    // TODO (hme): Separate this whole thing into two functions: One for output inference, and one for desugaring.
    let rtype: Option<Type> = if disable_output_inference {
        output_type
    } else {
        let (return_type_name, return_type_generic_args) = vod.return_type;
        if return_type_generic_args.is_empty() {
            let ty = syn::parse::<Type>(return_type_name.parse().map_err(|_| {
                syn_err(
                    op_or_method_ident.span(),
                    &format!("Unable to parse {return_type_name}"),
                )
            })?)
            .map_err(|e| {
                syn_err(
                    op_or_method_ident.span(),
                    &format!("Unable to parse {return_type_name}: {e}"),
                )
            })?;
            Some(ty)
        } else {
            let mut missing_cgas = vec![];
            let mut return_type_generic_arg_strings = vec![];
            let mut num_cgas = 0;
            for arg in return_type_generic_args {
                if !vod.cga_map.contains_key(*arg) {
                    continue;
                }
                num_cgas += 1;
                match vod_cga_name_to_context_cga_name.get(arg) {
                    Some(Some(s)) => return_type_generic_arg_strings.push(s.clone()),
                    _ => missing_cgas.push(arg),
                };
            }
            if return_type_generic_arg_strings.len() != num_cgas {
                // Return the given output_type if it is not none.
                if output_type.is_none() {
                    return Err(syn_err(
                        op_or_method_ident.span(),
                        &format!("Failed to infer return type generic args {:?} \nop={} \nvod_cga_name_to_context_cga_name={vod_cga_name_to_context_cga_name:#?}", missing_cgas, op_or_method_ident),
                    ));
                }
                output_type
            } else {
                let return_type_str = format!(
                    "{}<{}>",
                    return_type_name,
                    return_type_generic_arg_strings.join(", ")
                );
                let ty = syn::parse::<Type>(return_type_str.parse().map_err(|_| {
                    syn_err(
                        op_or_method_ident.span(),
                        &format!("Unable to parse {return_type_str}"),
                    )
                })?)
                .map_err(|e| {
                    syn_err(
                        op_or_method_ident.span(),
                        &format!("Unable to parse {return_type_str}: {e}"),
                    )
                })?;
                Some(ty)
            }
        }
    };

    if vod.const_length_vars.len() != const_length_values.len() {
        return Err(syn_err(
            op_or_method_ident.span(),
            &format!("Failed to infer op name from given parameters {op_or_method_ident}"),
        ));
    }
    let mut length_vec = Vec::with_capacity(vod.const_length_vars.len());
    for const_length_var in vod.const_length_vars {
        let val = const_length_values.get(const_length_var).ok_or_else(|| {
            syn_err(
                op_or_method_ident.span(),
                &format!("Missing const_length_value for '{const_length_var}'"),
            )
        })?;
        length_vec.push(*val);
    }
    let ident = get_variadic_op_ident(op_or_method_ident, &length_vec);
    Ok((ident, rtype))
}

/// Generates the rank-specialized identifier for a variadic operation.
///
/// Appends a suffix encoding the const generic array dimensions to the base
/// identifier. For example, `reshape` with dimensions `[2, 3]` becomes `reshape__2_3`.
///
/// ## Parameters
///
/// - `ident`: The base operation identifier
/// - `const_ga_lengths`: Vector of dimension lengths for each const generic array
///
/// ## Returns
///
/// A new identifier with the rank suffix appended (e.g., `reshape__2_3`)
fn get_variadic_op_ident(ident: &Ident, const_ga_lengths: &Vec<u32>) -> Ident {
    let fn_name_suffix = get_variadic_function_suffix(const_ga_lengths);
    Ident::new(&format!("{}__{}", ident, fn_name_suffix), ident.span())
}

/// Tracks const generic array instantiations during macro expansion.
///
/// This struct maintains mappings between const generic array parameters and their
/// concrete instantiations. It's used during variadic type and operation rewriting
/// to track which arrays have been instantiated with which dimensions.
///
/// ## Fields
///
/// - `inst_u32`: Maps length variable names (e.g., `"N"`) to their concrete values (e.g., `2`)
/// - `var_arrays`: Maps array names to their variable CGA parameter definitions
/// - `inst_array`: Maps array names to their concrete instantiated parameters
///
/// ## Example
///
/// For a function `fn foo<const N: usize, const S: [i32; N]>(...)`, this tracks:
/// - `inst_u32`: `{"N" => 2}`
/// - `var_arrays`: `{"S" => VarCGAParameter { name: "S", length_var: "N" }}`
/// - `inst_array`: `{"S" => CGAParameter { name: "S", length: 2 }}`
/// Tracks const generic array instantiations during variadic macro expansion.
///
/// Maps length variable names to concrete values and array names to their
/// instantiated parameters.
#[derive(Debug, Clone)]
pub struct ConstInstances {
    inst_u32: HashMap<String, u32>,
    var_arrays: HashMap<String, VarCGAParameter>,
    inst_array: HashMap<String, CGAParameter>,
}

impl ConstInstances {
    fn new() -> Self {
        let inst_u32: HashMap<String, u32> = HashMap::new();
        let inst_array: HashMap<String, CGAParameter> = HashMap::new();
        let var_arrays: HashMap<String, VarCGAParameter> = HashMap::new();
        ConstInstances {
            inst_u32,
            inst_array,
            var_arrays,
        }
    }
    fn from_variadic(
        cga_lengths: &VariadicLengthItem,
        var_cgas: &Vec<VarCGAParameter>,
    ) -> Result<Self, Error> {
        let mut inst_u32: HashMap<String, u32> = HashMap::new();
        let mut inst_array: HashMap<String, CGAParameter> = HashMap::new();
        let mut var_arrays: HashMap<String, VarCGAParameter> = HashMap::new();
        for (length_var_name, length_instance) in &cga_lengths.variadic_length_instance {
            inst_u32.insert(length_var_name.clone(), *length_instance as u32);
        }
        for i in 0..cga_lengths.cga_length_instance.len() {
            let (length_var_name, length_instance): &(String, usize) =
                &cga_lengths.cga_length_instance[i];
            let length_instance = *length_instance as u32;
            let cga = &var_cgas[i];
            if length_var_name != &cga.length_var {
                return Err(syn_err(
                    Span::call_site(),
                    &format!(
                        "CGA length var name mismatch: expected '{}', got '{}'",
                        cga.length_var, length_var_name
                    ),
                ));
            }
            if let Some(existing_length) = inst_u32.insert(length_var_name.clone(), length_instance)
            {
                if existing_length != length_instance {
                    return Err(syn_err(
                        Span::call_site(),
                        &format!(
                            "CGA length instance mismatch for '{}': expected {}, got {}",
                            length_var_name, existing_length, length_instance
                        ),
                    ));
                }
            }
            inst_array.insert(cga.name.clone(), cga.instance(length_instance));
            var_arrays.insert(cga.name.clone(), cga.clone());
        }
        Ok(ConstInstances {
            inst_u32,
            inst_array,
            var_arrays,
        })
    }
    fn from_generics(generics: &Generics) -> Result<Self, Error> {
        let (cga_param, _u32_param) = parse_cgas(generics);
        let inst_u32: HashMap<String, u32> = HashMap::new();
        let mut inst_array: HashMap<String, CGAParameter> = HashMap::new();
        let var_arrays: HashMap<String, VarCGAParameter> = HashMap::new();
        for cga in cga_param {
            inst_array.insert(cga.name.clone(), cga.clone());
        }
        Ok(ConstInstances {
            inst_u32,
            inst_array,
            var_arrays,
        })
    }
    fn instantiate_var_cgas(&self, var_cgas: &Vec<VarCGAParameter>) -> Result<Self, Error> {
        let mut result = self.clone();
        for cga in var_cgas {
            if !result.inst_u32.contains_key(&cga.length_var) {
                return Err(syn_err(
                    Span::call_site(),
                    &format!(
                        "instantiate_var_cgas: Missing inst_u32 entry for '{}'",
                        cga.length_var
                    ),
                ));
            }
            let n = result.inst_u32.get(&cga.length_var).unwrap();
            result.inst_array.insert(cga.name.clone(), cga.instance(*n));
            result.var_arrays.insert(cga.name.clone(), cga.clone());
        }
        Ok(result)
    }
    fn instantiate_new_var_cgas(
        &self,
        n_list: &Vec<u32>,
        var_cgas: &Vec<VarCGAParameter>,
    ) -> Result<Self, Error> {
        let mut result = self.clone();
        for i in 0..n_list.len() {
            let n: u32 = n_list[i];
            let cga = &var_cgas[i];
            if result.inst_u32.contains_key(&cga.length_var) {
                return Err(syn_err(
                    Span::call_site(),
                    &format!(
                        "instantiate_new_var_cgas: inst_u32 already contains entry for '{}'",
                        cga.length_var
                    ),
                ));
            }
            result.inst_u32.insert(cga.length_var.clone(), n);
            result.inst_array.insert(cga.name.clone(), cga.instance(n));
            result.var_arrays.insert(cga.name.clone(), cga.clone());
        }
        Ok(result)
    }
}

/// Iterator for generating all combinations of const generic array lengths.
///
/// This iterator generates const instances for variadic types and operations by
/// iterating through all valid combinations of array lengths. For example, if
/// a variadic struct has `N=4`, this generates instances for N=0, 1, 2, 3, 4.
///
/// ## Fields
///
/// - `i`: Current iteration index
/// - `i_max`: Maximum number of iterations (product of all unique lengths)
/// - `cga_lengths`: List of (length_var_name, max_length) tuples
/// - `arrays`: Reference to the const generic array parameters being instantiated
///
/// ## Example
/// For `#[variadic_struct(N=4)]` with two arrays depending on N, this generates:
/// - (0, 0), (1, 1), ..., (4, 4) - 5 total combinations
#[derive(Debug)]
/// Iterates over all combinations of const generic array lengths for variadic expansion.
struct VariadicLengthIterator {
    i: usize,
    i_max: usize,
    variadic_lengths: BTreeMap<String, usize>, // Deterministic order is required for correctness.
    cga_length_vars: Vec<String>,
}

impl VariadicLengthIterator {
    fn new(attribute_list: &SingleMetaList, arrays: &Vec<VarCGAParameter>) -> Result<Self, Error> {
        let mut i_max: usize = 1;
        let mut variadic_lengths: BTreeMap<String, usize> = BTreeMap::new();
        if let Some(variadic_length_vars) = attribute_list.parse_string_arr("variadic_length_vars")
        {
            for var in variadic_length_vars {
                let len = (attribute_list.parse_int(var.as_str()).ok_or_else(|| {
                    syn_err(
                        Span::call_site(),
                        &format!("Missing attribute value for '{var}'"),
                    )
                })? + 1) as usize;
                i_max *= len;
                if variadic_lengths.insert(var.clone(), len).is_some() {
                    return Err(syn_err(
                        Span::call_site(),
                        &format!("Duplicate variadic_length_var '{var}'"),
                    ));
                }
            }
        }
        let mut cga_length_vars = vec![];
        for cga in arrays {
            let var = cga.length_var.clone();
            cga_length_vars.push(var.clone());
            // This is so we don't need to explicitly specify the variadic_length_vars attribute.
            let len = (attribute_list.parse_int(var.as_str()).ok_or_else(|| {
                syn_err(
                    Span::call_site(),
                    &format!("Missing attribute value for '{var}'"),
                )
            })? + 1) as usize;
            if variadic_lengths.contains_key(&var) {
                if *variadic_lengths.get(&var).unwrap() != len {
                    return Err(syn_err(
                        Span::call_site(),
                        &format!("Variadic length mismatch for '{var}'"),
                    ));
                }
            } else {
                i_max *= len;
                variadic_lengths.insert(var.clone(), len);
            }
        }
        Ok(VariadicLengthIterator {
            i: 0,
            i_max,
            variadic_lengths,
            cga_length_vars,
        })
    }
}

/// A single combination of length variable values produced by `VariadicLengthIterator`.
pub struct VariadicLengthItem {
    variadic_length_instance: BTreeMap<String, usize>,
    cga_length_instance: Vec<(String, usize)>,
}

impl VariadicLengthItem {
    /// Returns CGA lengths as a vector, ordered by CGA declaration order.
    pub fn vec_of_cga_lengths(&self) -> Vec<u32> {
        // Ordered by key.
        self.cga_length_instance
            .iter()
            .map(|x| x.1 as u32)
            .collect::<Vec<_>>()
    }
    /// Returns unique length variable values, ordered by variable name.
    pub fn vec_of_unique_lengths(&self) -> Vec<u32> {
        // Ordered by key.
        self.variadic_length_instance
            .values()
            .map(|x| *x as u32)
            .collect::<Vec<_>>()
    }
}

impl Iterator for VariadicLengthIterator {
    type Item = VariadicLengthItem;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.i_max {
            let mut variadic_length_instance: BTreeMap<String, usize> = BTreeMap::new();
            let mut i = self.i;
            for (len_var, len) in self.variadic_lengths.iter() {
                // BTree iter sorts by key.
                let pos = i % len;
                i /= len;
                variadic_length_instance.insert(len_var.clone(), pos);
            }
            self.i += 1;
            let mut cga_length_instance: Vec<(String, usize)> = vec![];
            for len_var in &self.cga_length_vars {
                let len = *variadic_length_instance
                    .get(len_var)
                    .unwrap_or_else(|| panic!("Unexpected length var {len_var}"));
                cga_length_instance.push((len_var.clone(), len));
            }
            Some(VariadicLengthItem {
                variadic_length_instance,
                cga_length_instance,
            })
        } else {
            None
        }
    }
}

/// Extracts variable const generic array parameters from a `Generics` clause.
pub fn parse_var_cgas(generics: &Generics) -> Vec<VarCGAParameter> {
    let mut result: Vec<VarCGAParameter> = vec![];
    for param in &generics.params {
        match param {
            GenericParam::Type(_type_param) => continue,
            GenericParam::Const(const_param) => match &const_param.ty {
                Type::Array(_ty_arr) => {
                    let arr_type_param = VarCGAParameter::from_const_param(const_param);
                    result.push(arr_type_param);
                }
                _ => continue,
            },
            _ => continue,
        }
    }
    result
}

/// Expands a variadic struct into multiple rank-specific versions.
///
/// Takes a struct with const generic array parameters and generates concrete
/// versions for each rank (typically 1-4).
///
/// ## Parameters
///
/// - `attributes`: Macro attributes (e.g., `N=4`, `constructor="new"`)
/// - `item`: The struct definition to expand
///
/// ## Returns
///
/// A vector of (struct, optional impl) pairs, one for each rank. The impl is
/// generated if a constructor name is specified in the attributes.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_struct(N=4, constructor="new")]
/// pub struct Tile<E, const D: [i32; N]> { _type: PhantomData<E> }
///
/// // Generates:
/// // - Tile_1, Tile_2, Tile_3, Tile_4 structs
/// // - Optional impl blocks with constructors if constructor="new" is specified
/// ```
pub fn variadic_struct(
    attributes: &SingleMetaList,
    item: ItemStruct,
) -> Result<Vec<(ItemStruct, Option<ItemImpl>)>, Error> {
    // println!("item: {item:#?}");
    let vtd = get_variadic_type_data(item.ident.to_string().as_str());
    if vtd.is_none() {
        return Err(syn_err(
            item.ident.span(),
            &format!(
                "Generating {} requires a corresponding entry in VARIADIC_TYPES",
                item.ident
            ),
        ));
    }
    let cgas = parse_var_cgas(&item.generics);
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;

    // There is only one variadic type that we're expanding (the name of this struct).
    let maybe_constructor_name = attributes.parse_string("constructor");
    let vtd = vtd.unwrap();
    let num_cgas = vtd.num_cgas();
    if cgas.len() as u32 != num_cgas {
        return Err(syn_err(
            item.ident.span(),
            &format!(
                "Expected {} const generic arrays, got {}",
                num_cgas,
                cgas.len()
            ),
        ));
    }
    let mut result: Vec<(ItemStruct, Option<ItemImpl>)> = vec![];
    for var_cga_iter_item in cga_iter {
        let mut concrete = item.clone();
        // This just constructs the current instantiation of the const generic arrays for this struct.
        // There is usually only one CGA for structs.
        let const_instances = ConstInstances::from_variadic(&var_cga_iter_item, &cgas)?;
        // We only need the variadic type data structure to obtain the name of this instance of the struct.
        let concrete_ident = Ident::new(
            &vtd.concrete_name(&var_cga_iter_item.vec_of_cga_lengths()),
            concrete.ident.span(),
        );
        concrete.ident = concrete_ident;
        // Reuse the generics desugaring to rewrite generics based on const generic instances.
        desugar_generics(&mut concrete.generics, &const_instances)?;
        // If this is a mixed static/dynamic struct, make type-checked constructors.
        let concrete_impl = if maybe_constructor_name.is_some() {
            let mut type_params: Vec<String> = vec![];
            let mut type_args: Vec<String> = vec![];
            let mut constructors: Vec<String> = vec![];
            for cga_idx in 0..num_cgas {
                // n_list.len() == cgas.len()
                // and we know (from above) cgas.len() == num_cgas
                let n = var_cga_iter_item.vec_of_cga_lengths()[cga_idx as usize];
                let cga_name: &str = vtd.cga_names[cga_idx as usize];
                let cga_index_type: &str = vtd.cga_index_types[cga_idx as usize];
                // Expand the cga into const generics of type cga_index_type.
                for dim_idx in 0..n {
                    type_params.push(format!("const {cga_name}{dim_idx}: {cga_index_type}"));
                    type_args.push(format!("{cga_name}{dim_idx}"));
                }
                let cga_dim_type: &DimType = &vtd.cga_dim_types[cga_idx as usize];
                match cga_dim_type {
                    DimType::Mixed => {
                        // This CGA in this struct is mixed static/dynamic.
                        // Iterate up to number of dynamic fields which are expected
                        // by this struct instance (n).
                        for num_dynamic in 0..(n + 1) {
                            let struct_name = concrete.ident.to_string();
                            let constructor_name = format!(
                                "{}_{}",
                                maybe_constructor_name.clone().unwrap(),
                                num_dynamic
                            );
                            // TODO (hme): This has become quite hacky.
                            let dim_type_str = "i32";
                            let dyn_constructor = format!(
                                r#"
                                 pub fn {constructor_name}(dims: &'a [{dim_type_str}; {num_dynamic}]) -> Self {{
                                     {struct_name} {{ dims: dims }}
                                 }}
                             "#
                            );
                            constructors.push(dyn_constructor);
                            if num_dynamic == 0 {
                                let constructor_name =
                                    maybe_constructor_name.clone().unwrap().to_string();
                                let const_constructor = format!(
                                    r#"
                                 pub fn const_{constructor_name}() -> Self {{
                                     {struct_name} {{ dims: &[] }}
                                 }}
                             "#
                                );
                                constructors.push(const_constructor);
                            }
                        }
                    }
                    DimType::Static => {}
                }
            }
            if constructors.is_empty() {
                None
            } else {
                let name = concrete.ident.to_string();
                let impl_generics = type_params.join(",");
                let impl_constructors = constructors.join("\n");
                let impl_type_args = type_args.join(",");
                let constructor_impl = format!(
                    r#"
                impl<'a, {impl_generics}> {name}<'a, {impl_type_args}> {{
                        {impl_constructors}
                    }}
                "#
                );
                let parsed_impl =
                    syn::parse::<ItemImpl>(constructor_impl.parse().map_err(|_| {
                        syn_err(item.ident.span(), "Failed to parse constructor impl")
                    })?)
                    .map_err(|e| {
                        syn_err(
                            item.ident.span(),
                            &format!("Failed to parse constructor impl: {e}"),
                        )
                    })?;
                // println!("parsed_impl: {parsed_impl:#?}");
                Some(parsed_impl)
            }
        } else {
            None
        };
        result.push((concrete, concrete_impl));
    }
    Ok(result)
}

/// Expands a variadic trait into multiple rank-specific versions.
///
/// Generates a trait definition for each rank specified in the attributes.
/// Each version has the rank appended to its name (e.g., `MyTrait_1`, `MyTrait_2`).
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_trait(N=4)]
/// pub trait BroadcastScalar<E, const D: [i32; N]> {
///     fn broadcast(self, shape: Shape<D>) -> Tile<E, D>;
/// }
///
/// // Output:
/// pub trait BroadcastScalar_1<E, const D: [i32; 1]> { ... }
/// pub trait BroadcastScalar_2<E, const D: [i32; 2]> { ... }
/// pub trait BroadcastScalar_3<E, const D: [i32; 3]> { ... }
/// pub trait BroadcastScalar_4<E, const D: [i32; 4]> { ... }
/// ```
pub fn variadic_trait(
    attributes: &SingleMetaList,
    item: ItemTrait,
) -> Result<Vec<ItemTrait>, Error> {
    // println!("item: {item:#?}");
    let cgas = parse_var_cgas(&item.generics);
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;
    let rewrite_variadics = RewriteVariadicsPass {};
    let mut result: Vec<ItemTrait> = vec![];
    for n_list in cga_iter {
        let const_instances = ConstInstances::from_variadic(&n_list, &cgas)?;
        result.push(rewrite_variadics.rewrite_trait(&item, &const_instances)?);
    }
    Ok(result)
}

/// Expands a variadic implementation into multiple rank-specific versions.
///
/// Generates an impl block for each rank, with all types and method signatures
/// updated to use the rank-specific type names.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_impl(N=4)]
/// impl<E, const D: [i32; N]> Tile<E, D> {
///     pub fn shape(&self) -> Shape<D> { ... }
/// }
///
/// // Output:
/// impl<E, const D: [i32; 1]> Tile_1<E, D> { ... }
/// impl<E, const D: [i32; 2]> Tile_2<E, D> { ... }
/// impl<E, const D: [i32; 3]> Tile_3<E, D> { ... }
/// impl<E, const D: [i32; 4]> Tile_4<E, D> { ... }
/// ```
pub fn variadic_impl(attributes: &SingleMetaList, item: ItemImpl) -> Result<Vec<ItemImpl>, Error> {
    let cgas = parse_var_cgas(&item.generics);
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;

    let rewrite_variadics = RewriteVariadicsPass {};
    let mut result: Vec<ItemImpl> = vec![];
    for n_list in cga_iter {
        let const_instances = ConstInstances::from_variadic(&n_list, &cgas)?;
        result.push(rewrite_variadics.rewrite_impl(&item, &const_instances)?);
    }
    Ok(result)
}

/// Expands a variadic impl method into rank-specific versions.
 fn variadic_impl_fn_gen(
    attributes: &SingleMetaList,
    self_ty: &Type,
    item: &ImplItemFn,
    const_instances_impl: &ConstInstances,
) -> Result<Vec<ImplItemFn>, Error> {
    let cgas = parse_var_cgas(&item.sig.generics);
    // This ensures to not duplicate cgas with the same n.
    let _fn_types = get_sig_types(&item.sig, Some(self_ty));
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;
    let mut result: Vec<ImplItemFn> = vec![];
    // Iterate over the set of const generic arrays.
    for cga_iter_item in cga_iter {
        // Generate as many items as the product of const generic array instances.
        let const_instances = const_instances_impl
            .instantiate_new_var_cgas(&cga_iter_item.vec_of_cga_lengths(), &cgas)?;
        // println!("cga_iter {i}: {n_list:?}");
        let concrete_fn = rewrite_impl_fn(self_ty, item, &const_instances)?;
        // concrete_fn.sig.ident = get_concrete_op_ident_from_types(&concrete_fn.sig.ident, &fn_types.0, Some(fn_types.1.clone()), &const_instances).0;
        // rewrite_fn_sig(&mut concrete_fn.sig, &const_instances);
        result.push(concrete_fn);
    }
    Ok(result)
}

/// Rewrites a single impl method using the given const instantiations.
 fn rewrite_impl_fn(
    self_ty: &Type,
    item: &ImplItemFn,
    const_instances: &ConstInstances,
) -> Result<ImplItemFn, Error> {
    // TODO (hme): This is weird, we instantiate multiple RewriteVariadicTypesPass instances in some cases.
    let rewrite_pass = RewriteVariadicsPass {};
    let mut result = item.clone();
    rewrite_pass.rewrite_impl_fn(self_ty, &mut result, const_instances, None)?;
    Ok(result)
}

/// Desugars const generic arrays in a function signature's generics, inputs, and output.
 fn rewrite_fn_sig(
    sig: &mut Signature,
    const_instances: &ConstInstances,
) -> Result<(), Error> {
    desugar_generics(&mut sig.generics, const_instances)?;
    let mut desugared_inputs = sig.inputs.clone();
    for input in desugared_inputs.iter_mut() {
        match input {
            FnArg::Receiver(_receiver) => {
                // Leave this.
            }
            FnArg::Typed(fn_param) => {
                let fn_param_type = desugar_ty(&fn_param.ty, const_instances)?;
                *fn_param.ty = fn_param_type;
            }
        }
    }
    sig.inputs = desugared_inputs;
    let mut desugared_outputs = sig.output.clone();
    if let ReturnType::Type(_, return_type) = &mut desugared_outputs {
        **return_type = desugar_ty(&return_type.clone(), const_instances)?;
    }
    sig.output = desugared_outputs;
    Ok(())
}

/// Expands a variadic function into multiple rank-specific versions.
///
/// Generates a function for each rank combination specified in the attributes.
/// Each version is specialized for specific array ranks.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// #[cuda_tile::variadic_op(N=4)]
/// pub fn load_tile<E, const S: [i32; N]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
///     let tile_shape = y.shape();
///     // ...
/// }
///
/// // Output:
/// pub fn load_tile_1<E, const S: [i32; 1]>(y: &mut Tensor_1<E, S>) -> Tile_1<E, S> { ... }
/// pub fn load_tile_2<E, const S: [i32; 2]>(y: &mut Tensor_2<E, S>) -> Tile_2<E, S> { ... }
/// pub fn load_tile_3<E, const S: [i32; 3]>(y: &mut Tensor_3<E, S>) -> Tile_3<E, S> { ... }
/// pub fn load_tile_4<E, const S: [i32; 4]>(y: &mut Tensor_4<E, S>) -> Tile_4<E, S> { ... }
/// ```
///
/// ## Multi-dimensional Variadics
///
/// Functions can have multiple variadic parameters with different ranks:
///
/// ```rust,ignore
/// #[cuda_tile::variadic_op(N=4, M=4)]
/// pub fn reshape<E, const S: [i32; N], const R: [i32; M]>(
///     source: Tile<E, S>,
///     shape: Shape<R>
/// ) -> Tile<E, R>
///
/// // Generates reshape_1_1, reshape_1_2, ..., reshape_4_4 (16 total)
/// ```
pub fn variadic_op(attributes: &SingleMetaList, item: ItemFn) -> Result<Vec<ItemFn>, Error> {
    let op_name = item.sig.ident.to_string();
    if get_variadic_op_data(&op_name).is_none() {
        return Err(syn_err(
            item.sig.ident.span(),
            &format!("Variadic op data not found for {op_name}. VariadicOpData entry is required for ops with cuda_tile::variadic_op annotation."),
        ));
    }
    let cgas = parse_var_cgas(&item.sig.generics);
    // This ensures to not duplicate cgas with the same n.
    // let fn_types = get_sig_types(&item.sig, None);
    let cga_iter = VariadicLengthIterator::new(attributes, &cgas)?;
    let mut result: Vec<ItemFn> = vec![];
    // Iterate over the set of const generic arrays.
    let rewrite_variadics = RewriteVariadicsPass {};
    for n_list in cga_iter {
        // Generate as many items as the product of const generic array instances.
        let const_instances = ConstInstances::from_variadic(&n_list, &cgas)?;
        result.push(rewrite_variadics.rewrite_function(&item, &const_instances)?);
    }
    Ok(result)
}

/// Expands const generic array params in a `Generics` clause into individual const params.
 fn desugar_generics(
    generics: &mut Generics,
    const_instances: &ConstInstances,
) -> Result<(), Error> {
    let mut concrete_type_params = generics.params.clone();
    concrete_type_params.clear();
    for param in generics.params.iter() {
        match param {
            GenericParam::Const(const_param) => match &const_param.ty {
                Type::Array(_ty_arr) => {
                    let const_param_name = const_param.ident.to_string();
                    let cga = const_instances
                        .inst_array
                        .get(const_param_name.as_str())
                        .ok_or_else(|| {
                            syn_err(
                                const_param.ident.span(),
                                &format!("Missing inst_array entry for '{const_param_name}'"),
                            )
                        })?;
                    if cga.element_type != "i32" {
                        return Err(syn_err(
                            const_param.ident.span(),
                            &format!("Expected element_type 'i32', got '{}'", cga.element_type),
                        ));
                    }
                    for i in 0..cga.length {
                        let const_str = format!("const {}{}: {}", cga.name, i, cga.element_type);
                        let generic_param =
                            syn::parse::<GenericParam>(const_str.parse().map_err(|_| {
                                syn_err(
                                    const_param.ident.span(),
                                    &format!("Failed to parse generic param '{const_str}'"),
                                )
                            })?)
                            .map_err(|e| {
                                syn_err(
                                    const_param.ident.span(),
                                    &format!("Failed to parse generic param '{const_str}': {e}"),
                                )
                            })?;
                        concrete_type_params.push(generic_param);
                    }
                }
                _ => concrete_type_params.push(param.clone()),
            },
            _ => concrete_type_params.push(param.clone()),
        }
    }
    generics.params = concrete_type_params;
    Ok(())
}

/// Expands a CGA path into angle-bracketed individual const generic arguments.
 fn expand_cga(
    path: &Path,
    instances: &ConstInstances,
) -> Result<AngleBracketedGenericArguments, Error> {
    let _result_path = path.clone();
    let last_seg = path.segments.last().ok_or_else(|| {
        syn_err(
            path.span(),
            "Expected at least one path segment in expand_cga",
        )
    })?;
    let param_name = last_seg.ident.to_string();
    // Is it a variadic type or is it expecting a variadic type parameter?
    if instances.inst_array.contains_key(&param_name) {
        // The type is a const generic array, e.g. the D in f(..., shape: D) -> ()
        let cga = instances.inst_array.get(&param_name).unwrap();
        let mut generic_args_result: Vec<String> = vec![];
        for j in 0..cga.length {
            generic_args_result.push(format!("{}{}", cga.name, j));
        }
        let formatted = format!("<{}>", generic_args_result.join(","));
        Ok(
            syn::parse::<AngleBracketedGenericArguments>(formatted.parse().map_err(|_| {
                syn_err(
                    path.span(),
                    &format!("Failed to parse angle bracketed args '{formatted}'"),
                )
            })?)
            .map_err(|e| {
                syn_err(
                    path.span(),
                    &format!("Failed to parse angle bracketed args '{formatted}': {e}"),
                )
            })?,
        )
    } else {
        Err(syn_err(
            path.span(),
            &format!(
                "{} is not a const generic array.",
                path.to_token_stream()
            ),
        ))
    }
}

/// Desugars variadic types in a path, replacing CGA syntax with concrete type names and args.
 fn desugar_path(path: &Path, instances: &ConstInstances) -> Result<Path, Error> {
    let mut result_path = path.clone();
    for (i, seg) in path.segments.iter().enumerate() {
        let param_name = seg.ident.to_string();
        // Is it a variadic type or is it expecting a variadic type parameter?
        if instances.inst_array.contains_key(&param_name) {
            // The type is a const generic array: f(..., shape: D) -> ()
            // The result produced by this case is not supported syntax.
            return Err(syn_err(
                seg.ident.span(),
                &format!(
                    "Unexpected use of desugar_path for {}",
                    path.to_token_stream()
                ),
            ));
            // let cga = instances.inst_array.get(&param_name).unwrap();
            // let mut generic_args_result: Vec<String> = vec![];
            // for j in 0..cga.length {
            //     generic_args_result.push(format!("{}{}", cga.name, j));
            // }
            // let last_seg_args = syn::parse::<AngleBracketedGenericArguments>(format!("<{}>", generic_args_result.join(",")).parse().unwrap()).unwrap();
            // let result_seg = PathSegment{ident: seg.ident.clone(), arguments: PathArguments::AngleBracketed(last_seg_args)};
            // result_path.segments[i] = result_seg.clone();
        } else {
            let (last_type_ident, last_seg_args) = match &seg.arguments {
                // The type takes a const generic array as a type param:
                // f(..., shape: Shape<D>) -> ()
                PathArguments::AngleBracketed(type_params) => {
                    // This is a type of the form T<...>
                    let (type_ident, last_seg_args) =
                        desugar_cga(instances, &seg.ident, type_params)?;
                    (
                        type_ident.clone(),
                        PathArguments::AngleBracketed(last_seg_args),
                    )
                }
                PathArguments::None => {
                    // It's a type without type params.
                    // f(..., arg: Type) -> ()
                    // Is it a variadic type that needs to be rewritten??
                    let variadic_type_data: Option<VariadicTypeData> =
                        get_variadic_type_data(seg.ident.to_string().as_str());
                    if variadic_type_data.is_some() {
                        return Err(syn_err(
                            seg.ident.span(),
                            "Variadic type arguments are required to desugar variadic types.",
                        ));
                    }
                    (seg.ident.clone(), PathArguments::None)
                }
                _ => return Err(syn_err(seg.ident.span(), "Unexpected Path arguments.")),
            };
            let result_seg = PathSegment {
                ident: last_type_ident,
                arguments: last_seg_args,
            };
            result_path.segments[i] = result_seg.clone();
        }
    }
    // println!("desugar_path {}: {:#?}", result_path.segments.len(), result_path.to_token_stream().to_string());
    Ok(result_path)
}

/// Desugars variadic types within angle-bracketed generic arguments.
 fn desugar_generic_arguments(
    generic_args: &mut AngleBracketedGenericArguments,
    const_instances: &ConstInstances,
) -> Result<(), Error> {
    let span = generic_args.span();
    for arg in &mut generic_args.args {
        match arg {
            GenericArgument::Type(ty) => {
                *arg = GenericArgument::Type(desugar_ty(ty, const_instances)?);
            }
            _ => {
                return Err(syn_err(
                    span,
                    &format!(
                        "Unsupported generic argument {}",
                        arg.to_token_stream()
                    ),
                ))
            }
        }
    }
    Ok(())
}

/// Recursively desugars const generic array syntax within a type.
 fn desugar_ty(ty: &Type, instances: &ConstInstances) -> Result<Type, Error> {
    // Desugar const generic arrays as they appear as const generic arguments.
    Ok(match ty {
        Type::Path(type_path) => {
            // Special case: For Option<T>, recursively desugar T but don't try to
            // expand Option itself as a variadic type
            let last_segment = type_path.path.segments.last().ok_or_else(|| {
                syn_err(
                    type_path.span(),
                    "Expected at least one path segment in desugar_ty",
                )
            })?;
            if last_segment.ident == "Option" {
                let mut result_type = type_path.clone();
                if let PathArguments::AngleBracketed(args) = &last_segment.arguments {
                    let mut new_args = args.clone();
                    // Recursively desugar the type inside Option
                    for arg in &mut new_args.args {
                        if let GenericArgument::Type(inner_ty) = arg {
                            *inner_ty = desugar_ty(inner_ty, instances)?;
                        }
                    }
                    let last_idx = result_type.path.segments.len() - 1;
                    result_type.path.segments[last_idx].arguments =
                        PathArguments::AngleBracketed(new_args);
                }
                return Ok(result_type.into());
            }

            let mut result_type = type_path.clone();
            let path = desugar_path(&result_type.path, instances)?;
            result_type.path = path;
            // println!("desugar_ty: ")
            result_type.into()
        }
        Type::Array(type_array) => {
            let mut result = type_array.clone();
            *result.elem = desugar_ty(&type_array.elem, instances)?;
            let arr_len = result.len.to_token_stream().to_string();
            if instances.inst_u32.contains_key(&arr_len) {
                let n = instances.inst_u32.get(&arr_len).unwrap();
                result.len = syn::parse::<Expr>(format!("{}", n).parse().map_err(|_| {
                    syn_err(
                        type_array.span(),
                        &format!("Failed to parse array length '{n}'"),
                    )
                })?)
                .map_err(|e| {
                    syn_err(
                        type_array.span(),
                        &format!("Failed to parse array length '{n}': {e}"),
                    )
                })?;
            }
            result.into()
        }
        Type::Reference(ref_type) => {
            let mut result = ref_type.clone();
            *result.elem = desugar_ty(&ref_type.elem, instances)?;
            result.into()
            // unimplemented!("Type::Reference not implemented: {:#?}", ref_type)
        }
        Type::Tuple(tuple_type) => {
            let mut result = tuple_type.clone();
            for elem in &mut result.elems {
                *elem = desugar_ty(elem, instances)?;
            }
            Type::Tuple(result)
        }
        _ => ty.clone(),
    })
}

// TODO (hme): A lot of repetition between this and get_cga_type.
/// Desugars CGA generic arguments on a type, producing a concrete ident and expanded args.
 fn desugar_cga(
    instances: &ConstInstances,
    type_ident: &Ident,
    generic_args: &AngleBracketedGenericArguments,
) -> Result<(Ident, AngleBracketedGenericArguments), Error> {
    // println!("expand_variadic_type_generics {:#?}", generic_args);
    let mut expanded_param_name = type_ident.to_string();
    let variadic_type_data: Option<VariadicTypeData> =
        get_variadic_type_data(expanded_param_name.as_str());
    let mut generic_args_result: Vec<String> = vec![];

    for generic_arg in &generic_args.args {
        match generic_arg {
            GenericArgument::Type(type_param) => {
                match type_param {
                    Type::Path(type_path) => {
                        let last_ident = type_path
                            .path
                            .segments
                            .last()
                            .ok_or_else(|| {
                                syn_err(type_path.span(), "Expected at least one path segment")
                            })?
                            .ident
                            .to_string();
                        if instances.inst_array.contains_key(&last_ident) {
                            // This is something like Shape<D> for const generic array D: [i32; N].
                            let cga = instances.inst_array.get(&last_ident).unwrap();
                            for j in 0..cga.length {
                                generic_args_result.push(format!("{}{}", cga.name, j));
                            }
                            expanded_param_name = match &variadic_type_data {
                                Some(vtd) => vtd.concrete_name(&vec![cga.length]),
                                None => type_ident.to_string(),
                            };
                        } else {
                            // Not a const generic array instance, just convert to string
                            // This handles regular types like Tile<i1, S>, Token, etc.
                            generic_args_result.push(generic_arg.to_token_stream().to_string());
                        }
                        // println!("{n_list:?}, expand Type::Path {:?}: {:?}", generic_arg.to_token_stream().to_string(), generic_args_result);
                    }
                    Type::Reference(type_ref) => {
                        // References in generic arguments (e.g., Option<&str>) can be kept as-is
                        generic_args_result.push(type_ref.to_token_stream().to_string());
                    }
                    _ => {
                        generic_args_result.push(generic_arg.to_token_stream().to_string());
                    }
                }
            }
            GenericArgument::Const(const_param) => {
                // println!("expand GenericArgument::Const? {const_param:#?}");
                match const_param {
                    Expr::Block(block_expr) => {
                        // TODO (hme): Would be great to get rid of this syntax.
                        // This is something like Tensor<E, {[...]}>
                        if block_expr.block.stmts.len() != 1 {
                            return Err(syn_err(
                                block_expr.span(),
                                &format!(
                                    "Expected exactly 1 statement in block expression, got {}",
                                    block_expr.block.stmts.len()
                                ),
                            ));
                        }
                        let statement = &block_expr.block.stmts[0];
                        let Stmt::Expr(statement_expr, _) = statement else {
                            return Err(syn_err(block_expr.span(), "Unexpected block expression."));
                        };
                        match statement_expr {
                            Expr::Array(array_expr) => {
                                // This is something like Tensor<E, {[1, 2, -1]}>
                                let rank = array_expr.elems.len();
                                for elem in &array_expr.elems {
                                    let val = elem.to_token_stream().to_string();
                                    generic_args_result.push(val);
                                }
                                expanded_param_name = match &variadic_type_data {
                                    Some(vtd) => vtd.concrete_name(&vec![rank as u32]),
                                    None => type_ident.to_string(),
                                };
                            }
                            Expr::Repeat(repeat_expr) => {
                                // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                                let thing_to_repeat =
                                    repeat_expr.expr.to_token_stream().to_string();
                                let num_repetitions = match &*repeat_expr.len {
                                    Expr::Path(len_path) => {
                                        // This is something like Tensor<E, {[-1; N]}>
                                        let num_rep_var = len_path.to_token_stream().to_string();
                                        if !instances.inst_u32.contains_key(&num_rep_var) {
                                            return Err(syn_err(
                                                len_path.span(),
                                                &format!(
                                                    "Expected instance for generic argument {}",
                                                    num_rep_var
                                                ),
                                            ));
                                        }
                                        let num_repetitions =
                                            *instances.inst_u32.get(&num_rep_var).unwrap();
                                        for _ in 0..num_repetitions {
                                            generic_args_result.push(thing_to_repeat.clone());
                                        }
                                        num_repetitions
                                    }
                                    Expr::Lit(len_lit) => {
                                        // This is something like Tensor<E, {[-1; 3]}>
                                        let num_repetitions: u32 = len_lit
                                            .to_token_stream()
                                            .to_string()
                                            .parse::<u32>()
                                            .map_err(|e| {
                                                syn_err(
                                                    len_lit.span(),
                                                    &format!(
                                                        "Failed to parse repeat length as u32: {e}"
                                                    ),
                                                )
                                            })?;
                                        for _ in 0..num_repetitions {
                                            generic_args_result.push(thing_to_repeat.clone());
                                        }
                                        num_repetitions
                                    }
                                    _ => {
                                        return Err(syn_err(
                                            generic_args.span(),
                                            "Unexpected repeat expression.",
                                        ))
                                    }
                                };
                                expanded_param_name = match &variadic_type_data {
                                    Some(vtd) => vtd.concrete_name(&vec![num_repetitions]),
                                    None => type_ident.to_string(),
                                };
                            }
                            _ => {
                                return Err(syn_err(
                                    block_expr.span(),
                                    "Unexpected block expression.",
                                ))
                            }
                        }
                    }
                    Expr::Lit(lit_expr) => {
                        generic_args_result.push(lit_expr.lit.to_token_stream().to_string());
                    }
                    _ => {
                        generic_args_result.push(generic_arg.to_token_stream().to_string());
                    }
                }
            }
            _ => {
                generic_args_result.push(generic_arg.to_token_stream().to_string());
            }
        }
    }
    let expanded_param_ident = Ident::new(expanded_param_name.as_str(), type_ident.span());
    let formatted = format!("<{}>", generic_args_result.join(","));
    Ok((
        expanded_param_ident,
        syn::parse::<AngleBracketedGenericArguments>(formatted.parse().map_err(|_| {
            syn_err(
                type_ident.span(),
                &format!("Failed to parse angle bracketed args '{formatted}'"),
            )
        })?)
        .map_err(|e| {
            syn_err(
                type_ident.span(),
                &format!("Failed to parse angle bracketed args '{formatted}': {e}"),
            )
        })?,
    ))
}

/// Extracts the `ConstGenericArrayType` from a type expression, if it is a variadic type.
 fn get_cga_type(
    ty: &Type,
    const_instances: &ConstInstances,
) -> Result<Option<ConstGenericArrayType>, Error> {
    let vtd = match get_vtd(ty)? {
        Some(vtd) => vtd,
        None => return Ok(None),
    };
    let (_type_ident, generic_args) = get_ident_generic_args(ty, &vtd)?;
    // println!("get_cga_type: ty={}, generic_args={:#?}", ty.to_token_stream().to_string(), generic_args);
    let mut n: Vec<u32> = vec![];
    let mut cgas: Vec<Option<String>> = vec![];
    for generic_arg in &generic_args.args {
        match generic_arg {
            GenericArgument::Type(type_param) => {
                match type_param {
                    Type::Path(type_path) => {
                        let last_ident = type_path
                            .path
                            .segments
                            .last()
                            .ok_or_else(|| {
                                syn_err(
                                    type_path.span(),
                                    "Expected at least one path segment in get_cga_type",
                                )
                            })?
                            .ident
                            .to_string();
                        if const_instances.inst_array.contains_key(&last_ident) {
                            // This is something like Shape<D> for const generic array D: [i32; N].
                            let cga = const_instances.inst_array.get(&last_ident).unwrap();
                            n.push(cga.length);
                            cgas.push(Some(generic_arg.to_token_stream().to_string()));
                        }
                    }
                    Type::Reference(type_ref) => {
                        return Err(syn_err(
                            type_ref.span(),
                            &format!(
                                "get_cga_type: Type::Reference not supported: {}",
                                type_ref.to_token_stream()
                            ),
                        ));
                    }
                    _ => {}
                }
            }
            GenericArgument::Const(const_param) => {
                // println!("expand GenericArgument::Const? {const_param:#?}");
                if let Expr::Block(block_expr) = const_param {
                    // TODO (hme): Would be great to get rid of this syntax.
                    // This is something like Tensor<E, {[...]}>
                    if block_expr.block.stmts.len() != 1 {
                        return Err(syn_err(
                            block_expr.span(),
                            &format!(
                                "Expected exactly 1 statement in block expression, got {}",
                                block_expr.block.stmts.len()
                            ),
                        ));
                    }
                    let statement = &block_expr.block.stmts[0];
                    let Stmt::Expr(statement_expr, _) = statement else {
                        return Err(syn_err(block_expr.span(), "Unexpected block expression."));
                    };
                    match statement_expr {
                        Expr::Array(array_expr) => {
                            // This is something like Tensor<E, {[1, 2, -1]}>
                            n.push(array_expr.elems.len() as u32);
                            cgas.push(Some(generic_arg.to_token_stream().to_string()));
                        }
                        Expr::Repeat(repeat_expr) => {
                            // println!("Expr::Repeat: {:?}", repeat_expr.expr);
                            let _thing_to_repeat =
                                repeat_expr.expr.to_token_stream().to_string();
                            match &*repeat_expr.len {
                                Expr::Path(len_path) => {
                                    // This is something like Tensor<E, {[-1; N]}>
                                    let num_rep_var = len_path.to_token_stream().to_string();
                                    if !const_instances.inst_u32.contains_key(&num_rep_var) {
                                        return Err(syn_err(
                                            len_path.span(),
                                            &format!(
                                                "Expected instance for generic argument {}",
                                                num_rep_var
                                            ),
                                        ));
                                    }
                                    let num_rep =
                                        const_instances.inst_u32.get(&num_rep_var).unwrap();
                                    n.push(*num_rep);
                                    cgas.push(Some(generic_arg.to_token_stream().to_string()));
                                }
                                Expr::Lit(len_lit) => {
                                    // This is something like Tensor<E, {[-1; 3]}>
                                    let num_repetitions: u32 = len_lit
                                        .to_token_stream()
                                        .to_string()
                                        .parse::<u32>()
                                        .map_err(|e| {
                                            syn_err(
                                                len_lit.span(),
                                                &format!(
                                                    "Failed to parse repeat length as u32: {e}"
                                                ),
                                            )
                                        })?;
                                    n.push(num_repetitions);
                                    cgas.push(Some(generic_arg.to_token_stream().to_string()));
                                }
                                _ => {
                                    return Err(syn_err(
                                        ty.span(),
                                        "Unexpected repeat expression.",
                                    ))
                                }
                            }
                        }
                        _ => {
                            return Err(syn_err(
                                block_expr.span(),
                                "Unexpected block expression.",
                            ))
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if n.len() != cgas.len() {
        return Err(syn_err(
            ty.span(),
            &format!(
                "get_cga_type: n.len() ({}) != cgas.len() ({})",
                n.len(),
                cgas.len()
            ),
        ));
    }
    Ok(Some(ConstGenericArrayType {
        cga_arg_strings: cgas,
        n,
    }))
}

/// Represents a variable binding with optional type information.
///
/// Used during macro expansion to track variables and their types within function
/// bodies. This enables type inference for variadic operations even when types
/// aren't explicitly annotated in let bindings.
///
/// ## Fields
///
/// - `ty`: The type of the bound variable, if known
///
/// ## Example
///
/// ```rust,ignore
/// let x = reshape(tile, shape); // Binding { ty: Some(Tile_2<f32, ...>) }
/// let y; // Binding { ty: None }
/// ```
/// A variable binding with optional type information for variadic type inference.
#[derive(Debug)]
pub struct Binding {
    ty: Option<Type>,
}

impl Binding {
    fn get_cga_type(
        &self,
        const_instances: &ConstInstances,
    ) -> Result<Option<ConstGenericArrayType>, Error> {
        match &self.ty {
            Some(ty) => get_cga_type(ty, const_instances),
            None => Ok(None),
        }
    }
    fn get_vtd(&self) -> Result<Option<VariadicTypeData>, Error> {
        match &self.ty {
            Some(ty) => get_vtd(ty),
            None => Ok(None),
        }
    }
}

/// Main rewriting pass for desugaring variadic types and operations.
///
/// This struct implements the core logic for transforming generic variadic code
/// (using const generic arrays) into concrete rank-specialized code. It performs
/// a recursive AST traversal to:
/// 1. Desugar const generic array syntax (`const S: [i32; N]` → `const S_0: i32, const S_1: i32, ...`)
/// 2. Rewrite variadic function/method calls to their rank-specialized versions
/// 3. Infer types for untyped variable bindings
/// 4. Track variable scopes and types throughout the function body
///
/// ## Usage
///
/// This pass is invoked automatically during macro expansion for functions,
/// impl blocks, and other items within `#[cutile::module]`.
/// Main AST rewriting pass that desugars variadic types and operations into rank-specific code.
pub struct RewriteVariadicsPass {}

impl RewriteVariadicsPass {
    fn rewrite_struct(
        &self,
        item: &ItemStruct,
        const_instances: &ConstInstances,
    ) -> Result<ItemStruct, Error> {
        // This is not a variadic struct, so we don't attempt to rewrite its name.
        let mut item = item.clone();
        for field in &mut item.fields {
            field.ty = desugar_ty(&field.ty, const_instances)?;
        }
        Ok(item)
    }

    fn rewrite_function(
        &self,
        item: &ItemFn,
        const_instances: &ConstInstances,
    ) -> Result<ItemFn, Error> {
        let mut item = item.clone();
        let mut variables: TrainMap<String, Binding> = self.bind_parameters(None, &item.sig)?;
        let (inputs, output) = get_sig_types(&item.sig, None);
        let inputs = inputs.into_iter().map(Some).collect::<Vec<_>>();
        item.sig.ident = get_concrete_op_ident_from_types(
            &item.sig.ident,
            &inputs,
            Some(output.clone()),
            const_instances,
            true,
        )?
        .0;
        self.rewrite_sig(&mut item.sig, const_instances)?;
        self.rewrite_statements(
            &mut item.block.stmts,
            const_instances,
            &mut variables,
            Some(output),
        )?;
        Ok(item)
    }

    fn rewrite_trait(
        &self,
        item: &ItemTrait,
        const_instances: &ConstInstances,
    ) -> Result<ItemTrait, Error> {
        let mut item = item.clone();
        if const_instances.inst_u32.is_empty() {
            return Ok(item);
        }
        if const_instances.inst_u32.len() != 1 {
            return Err(syn_err(
                item.ident.span(),
                "Only one CGA is permitted for variadic traits.",
            ));
        }
        let key = const_instances.inst_u32.keys().next().unwrap().clone();
        let n = *const_instances.inst_u32.get(&key).unwrap();
        let trait_name = item.ident.to_string();
        let concrete_name = concrete_name(&trait_name, &[n]);
        item.ident = Ident::new(&concrete_name, item.ident.span());
        desugar_generics(&mut item.generics, const_instances)?;
        // Update items.
        let mut impl_items: Vec<TraitItem> = vec![];
        for concrete_item in &mut item.items {
            match concrete_item {
                TraitItem::Fn(trait_item_fn) => {
                    let mut result = trait_item_fn.clone();
                    let cgas = parse_var_cgas(&result.sig.generics);
                    let const_instances = const_instances.instantiate_var_cgas(&cgas)?;
                    let method_name = result.sig.ident.to_string();
                    if let Some(vtd) = get_variadic_type_data(&trait_name) {
                        if let Some((_op_name, vod)) = get_variadic_method_data(&vtd, &method_name)?
                        {
                            // It's not okay to make this assumption here, but we don't actually know the type.
                            let self_type =
                                syn::parse2::<syn::Type>("Self".parse().map_err(|_| {
                                    syn_err(result.sig.ident.span(), "Failed to parse 'Self' type")
                                })?)
                                .map_err(|e| {
                                    syn_err(
                                        result.sig.ident.span(),
                                        &format!("Failed to parse 'Self' type: {e}"),
                                    )
                                })?;
                            let (inputs, output) = get_sig_types(&result.sig, Some(&self_type));
                            let inputs = inputs.into_iter().map(Some).collect::<Vec<_>>();
                            result.sig.ident = get_concrete_op_or_method_ident_from_types(
                                vod,
                                &result.sig.ident,
                                &inputs,
                                Some(output.clone()),
                                &const_instances,
                                true,
                            )?
                            .0;
                        }
                    }
                    self.rewrite_sig(&mut result.sig, &const_instances)?;
                    impl_items.push(TraitItem::Fn(result));
                }
                _ => {
                    return Err(syn_err(
                        concrete_item.span(),
                        "Unsupported impl item",
                    ))
                }
            }
        }
        item.items = impl_items;
        Ok(item)
    }

    fn rewrite_impl(
        &self,
        item: &ItemImpl,
        const_instances: &ConstInstances,
    ) -> Result<ItemImpl, Error> {
        let mut item = item.clone();
        let self_ty = *item.self_ty.clone();
        *item.self_ty = desugar_ty(&item.self_ty, const_instances)?;
        desugar_generics(&mut item.generics, const_instances)?;
        let mut variadic_trait_vtd = None;
        // Update generics in trait definition.
        if let Some(trait_) = &mut item.trait_ {
            let path_copy = trait_.1.clone();
            let path = &mut trait_.1;
            if path.segments.is_empty() {
                return Err(syn_err(
                    path.span(),
                    "Expected at least one path segment in trait path",
                ));
            }
            let last_seg = path.segments.last_mut().unwrap();
            let ident_vtd = get_variadic_type_data(last_seg.ident.to_string().as_str());
            if ident_vtd.is_some() {
                if let Some(vtd) = ident_vtd {
                    if const_instances.inst_u32.len() != 1 {
                        return Err(syn_err(
                            path.span(),
                            "Only one CGA is permitted for variadic traits.",
                        ));
                    }
                    *path = desugar_path(&path_copy, const_instances)?;
                    variadic_trait_vtd = Some(vtd);
                }
            } else if let PathArguments::AngleBracketed(path_args) = &mut last_seg.arguments {
                desugar_generic_arguments(path_args, const_instances)?
            }
        }

        // Update items.
        let mut impl_items: Vec<ImplItem> = vec![];
        for concrete_item in &mut item.items {
            match concrete_item {
                ImplItem::Type(type_impl) => {
                    let mut result = type_impl.clone();
                    result.ty = desugar_ty(&type_impl.ty, const_instances)?;
                    impl_items.push(ImplItem::Type(result));
                }
                ImplItem::Fn(fn_impl) => {
                    // We pass in the unmodified self type item.self_ty
                    let attributes = get_meta_list("cuda_tile :: variadic_impl_fn", &fn_impl.attrs);
                    match attributes {
                        Some(attributes) => {
                            if variadic_trait_vtd.is_some() {
                                return Err(syn_err(fn_impl.sig.ident.span(), "variadic_impl_fn attributes are not supported for variadic traits."));
                            }
                            clear_attributes(
                                HashSet::from(["cuda_tile :: variadic_impl_fn"]),
                                &mut fn_impl.attrs,
                            );
                            let results: Vec<ImplItemFn> = variadic_impl_fn_gen(
                                &attributes,
                                &self_ty,
                                fn_impl,
                                const_instances,
                            )?;
                            for result in results {
                                impl_items.push(ImplItem::Fn(result));
                            }
                            // println!("{:#?}", fn_impl.attrs);
                        }
                        None => {
                            let mut result = fn_impl.clone();
                            self.rewrite_impl_fn(
                                &self_ty,
                                &mut result,
                                const_instances,
                                variadic_trait_vtd.clone(),
                            )?;
                            // println!("{:#?}", &result);
                            impl_items.push(ImplItem::Fn(result));
                        }
                    }
                }
                _ => {
                    return Err(syn_err(
                        concrete_item.span(),
                        "Unsupported impl item.",
                    ))
                }
            }
        }
        item.items = impl_items;
        Ok(item)
    }

    fn rewrite_impl_fn(
        &self,
        self_ty: &Type,
        item: &mut ImplItemFn,
        const_instances: &ConstInstances,
        variadic_trait_vtd: Option<VariadicTypeData>,
    ) -> Result<(), Error> {
        let cgas = parse_var_cgas(&item.sig.generics);
        let const_instances = const_instances.instantiate_var_cgas(&cgas)?;
        let mut variables: TrainMap<String, Binding> =
            self.bind_parameters(Some(self_ty), &item.sig)?;
        let return_type: Option<Type> = match item.sig.output {
            ReturnType::Type(_, ref return_type) => Some(*return_type.clone()),
            _ => None,
        };
        // Check if it's variadic.
        let method_name = item.sig.ident.to_string();
        let vmmd = if variadic_trait_vtd.is_some() {
            let vtd = variadic_trait_vtd.unwrap();
            get_variadic_method_data(&vtd, &method_name)?.map(|(op_name, vod)| (op_name, vtd, vod))
        } else {
            get_variadic_method_meta_data(self_ty, &method_name)?
        };
        if let Some((_op_name, _vtd, vod)) = vmmd {
            // If it is, then rewrite the sig ident.
            // We do the same thing on the method call side.
            let (inputs, output) = get_sig_types(&item.sig, Some(self_ty));
            let inputs = inputs.into_iter().map(Some).collect::<Vec<_>>();
            // TODO (hme): This may result in redundant suffixes, but that should be okay.
            item.sig.ident = get_concrete_op_or_method_ident_from_types(
                vod,
                &item.sig.ident,
                &inputs,
                Some(output.clone()),
                &const_instances,
                true,
            )?
            .0;
        };
        self.rewrite_sig(&mut item.sig, &const_instances)?;
        self.rewrite_statements(
            &mut item.block.stmts,
            &const_instances,
            &mut variables,
            return_type,
        )?;
        Ok(())
    }

    fn bind_parameters(
        &self,
        self_ty: Option<&Type>,
        sig: &Signature,
    ) -> Result<TrainMap<'_, String, Binding>, Error> {
        let mut variables: TrainMap<String, Binding> = TrainMap::new();
        for input in sig.inputs.iter() {
            match input {
                FnArg::Typed(fn_param) => {
                    let name = {
                        match &*fn_param.pat {
                            Pat::Ident(ident) => ident.ident.to_string(),
                            _ => {
                                return Err(syn_err(
                                    fn_param.span(),
                                    "Unexpected function param pattern.",
                                ))
                            }
                        }
                    };
                    let ty = &*fn_param.ty;
                    variables.insert(
                        name.clone(),
                        Binding {
                            ty: Some(ty.clone()),
                        },
                    );
                }
                FnArg::Receiver(_fn_self) => {
                    if self_ty.is_none() {
                        return Err(syn_err(
                            sig.ident.span(),
                            "bind_parameters for impls requires self_ty.",
                        ));
                    }
                    let self_ty = self_ty.unwrap().clone();
                    variables.insert("self".to_string(), Binding { ty: Some(self_ty) });
                }
            }
        }
        Ok(variables)
    }

    fn rewrite_sig(
        &self,
        sig: &mut Signature,
        const_instances: &ConstInstances,
    ) -> Result<(), Error> {
        rewrite_fn_sig(sig, const_instances)
    }

    fn rewrite_statements(
        &self,
        statements: &mut [Stmt],
        const_instances: &ConstInstances,
        variables: &mut TrainMap<String, Binding>,
        mut return_type: Option<Type>,
    ) -> Result<Option<Type>, Error> {
        // Rewrite types.
        let num_statements = statements.len();
        for (i, statement) in statements.iter_mut().enumerate() {
            let is_last = i == num_statements - 1;
            match statement {
                Stmt::Local(local) => {
                    let mut binding_name: Option<String> = None;
                    let mut binding_ty: Option<Type> = None;
                    match &mut local.pat {
                        Pat::Type(pat_type) => {
                            match &*pat_type.pat {
                                Pat::Ident(pat_ident) => {
                                    binding_name = Some(pat_ident.ident.to_string());
                                }
                                Pat::Tuple(_) => {
                                    // Handle typed tuple destructuring: let (a, b): (T1, T2) = expr;
                                    // Rewrite the expression and desugar the type
                                    if let Some(init) = &mut local.init {
                                        self.rewrite_expr(
                                            &mut init.expr,
                                            const_instances,
                                            variables,
                                            None,
                                        )?;
                                    }
                                    binding_ty = Some(*pat_type.ty.clone());
                                    let new_ty = desugar_ty(&pat_type.ty, const_instances)?;
                                    *pat_type.ty = new_ty;
                                    // Skip normal single-variable logic - compiler will handle tuple binding
                                    continue;
                                }
                                _ => {
                                    return Err(syn_err(
                                        pat_type.span(),
                                        "let binding LHS not implemented.",
                                    ))
                                }
                            }
                            binding_ty = Some(*pat_type.ty.clone());
                            let new_ty = desugar_ty(&pat_type.ty, const_instances)?;
                            // println!("rewrite_statements Stmt::Local Pat::Type {:#?}", new_ty);
                            *pat_type.ty = new_ty;
                        }
                        Pat::Ident(pat_ident) => {
                            binding_name = Some(pat_ident.ident.to_string());
                            binding_ty = None;
                        } // Nothing to do. Let Rust infer.
                        Pat::Tuple(_) => {
                            // Handle tuple destructuring: let (a, b) = expr;
                            // Just rewrite the expression and pass through
                            // The cuda-tile compiler will handle variable binding
                            if let Some(init) = &mut local.init {
                                self.rewrite_expr(
                                    &mut init.expr,
                                    const_instances,
                                    variables,
                                    None,
                                )?;
                            }
                            continue; // Skip normal single-variable logic
                        }
                        _ => {
                            return Err(syn_err(
                                local.span(),
                                "Local pattern type not supported",
                            ))
                        }
                    }
                    if binding_name.is_none() {
                        return Err(syn_err(local.span(), "Unable to rewrite expr."));
                    }
                    let binding_name = binding_name.unwrap();
                    if let Some(init) = &mut local.init {
                        // Rewrite the expression but preserve explicit type annotations
                        let inferred_ty = self.rewrite_expr(
                            &mut init.expr,
                            const_instances,
                            variables,
                            binding_ty.clone(),
                        )?;
                        // Only use inferred type if we don't have an explicit type annotation
                        if binding_ty.is_none() {
                            binding_ty = inferred_ty;
                        }
                    }
                    variables.insert(
                        binding_name.clone(),
                        Binding {
                            ty: binding_ty.clone(),
                        },
                    );
                }
                Stmt::Item(item) => {
                    let mut binding_name: Option<String> = None;
                    let binding_ty: Option<Type> = match item {
                        Item::Const(const_item) => {
                            binding_name = Some(const_item.ident.to_string());
                            let return_type = Some(*const_item.ty.clone());
                            // This is like a let binding with limitations.
                            self.rewrite_expr(
                                &mut const_item.expr,
                                const_instances,
                                variables,
                                return_type,
                            )?
                        }
                        _ => {
                            return Err(syn_err(
                                item.span(),
                                &format!(
                                    "{}\nOnly const local item definitions are supported.",
                                    item.to_token_stream()
                                ),
                            ))
                        }
                    };
                    let Some(binding_name) = binding_name else {
                        return Err(syn_err(item.span(), "Unable to rewrite expr."));
                    };
                    variables.insert(
                        binding_name.clone(),
                        Binding {
                            ty: binding_ty.clone(),
                        },
                    );
                }
                Stmt::Expr(expr, semicolon) => {
                    let ty =
                        self.rewrite_expr(expr, const_instances, variables, return_type.clone())?;
                    match expr {
                        Expr::Assign(assign_expr) => {
                            let binding_name: String;
                            let mut binding_ty: Option<Type> = None;
                            match &mut *assign_expr.left {
                                Expr::Path(path_expr) => {
                                    if path_expr.path.segments.len() != 1 {
                                        return Err(syn_err(
                                            path_expr.span(),
                                            "Expected single-segment path in assignment",
                                        ));
                                    }
                                    binding_name = path_expr.path.segments[0].ident.to_string()
                                }
                                _ => {
                                    return Err(syn_err(
                                        assign_expr.span(),
                                        "Expr::Assign not supported",
                                    ))
                                }
                            }
                            // The computed binding type ty is expected to be None.
                            // Types are only needed in this pass to compute concrete variadic
                            // type and op names.
                            binding_ty = match variables.get(&binding_name) {
                                Some(old) => old.ty.clone(),
                                None => {
                                    // This is invalid, but let Rust generate a nice error.
                                    None
                                }
                            };
                            let _ = variables.insert(
                                binding_name,
                                Binding {
                                    ty: binding_ty.clone(),
                                },
                            );
                        }
                        Expr::Return(_return_expr) => {
                            return_type = ty;
                            break;
                        }
                        _ => {
                            if is_last && semicolon.is_none() {
                                return_type = ty;
                            } else {
                                // Unbinded result type.
                            }
                        }
                    }
                }
                Stmt::Macro(_mac_stmt) => continue, // There are no variadic macro expressions.
            }
        }
        Ok(return_type)
    }

    fn rewrite_expr(
        &self,
        expr: &mut Expr,
        const_instances: &ConstInstances,
        variables: &mut TrainMap<String, Binding>,
        mut return_type: Option<Type>,
    ) -> Result<Option<Type>, Error> {
        match expr {
            Expr::Index(index_expr) => {
                // println!(
                //     "rewrite_expr: Expr::Index: {}",
                //     index_expr.to_token_stream().to_string()
                // );
                // We rewrite the entire expr to desugared array.
                // Extract info we need before borrowing expr mutably.
                let index_span = index_expr.span();
                let inner_expr_span = index_expr.expr.span();
                let is_path = matches!(&*index_expr.expr, Expr::Path(_));
                if !is_path {
                    return Err(syn_err(
                        inner_expr_span,
                        &format!(
                            "Index expression not supported: {}",
                            index_expr.expr.to_token_stream()
                        ),
                    ));
                }
                let path_expr = match &*index_expr.expr {
                    Expr::Path(p) => p.clone(),
                    _ => unreachable!(),
                };
                let expr_str = path_expr.to_token_stream().to_string();
                if let Some(cga) = const_instances.inst_array.get(&expr_str) {
                    let expanded_cga = expand_cga(&path_expr.path, const_instances)?;
                    let index = parse_signed_literal_as_i32(&index_expr.index);
                    if !(0 <= index && (index as u32) < cga.length) {
                        return Err(syn_err(
                            index_span,
                            &format!(
                                "Index {index} out of bounds for CGA of length {}",
                                cga.length
                            ),
                        ));
                    }
                    if cga.element_type != "i32" {
                        return Err(syn_err(
                            index_span,
                            &format!("Expected element_type 'i32', got '{}'", cga.element_type),
                        ));
                    }
                    let desugared_idx_expression = expanded_cga.args[index as usize].clone();
                    *expr = parse_quote!(#desugared_idx_expression);
                    let return_type: Type = parse_quote! { i32 };
                    Ok(Some(return_type))
                } else {
                    let expr_span = expr.span();
                    let index_expr = match expr {
                        Expr::Index(ie) => ie,
                        _ => unreachable!(),
                    };
                    let index_expr_expr = &mut *index_expr.expr;
                    match self.rewrite_expr(
                        index_expr_expr,
                        const_instances,
                        variables,
                        return_type,
                    )? {
                        Some(Type::Array(ty)) => Ok(Some(*ty.elem.clone())),
                        Some(Type::Reference(ty)) => {
                            let Type::Slice(slice_ty) = *ty.elem.clone() else {
                                return Err(syn_err(
                                    expr_span,
                                    "Index expression not supported (reference)",
                                ));
                            };
                            Ok(Some(*slice_ty.elem.clone()))
                        }
                        None => Err(syn_err(
                            expr_span,
                            "Failed to compute type for index expression",
                        )),
                        Some(_other) => Err(syn_err(
                            expr_span,
                            "Index expression not supported",
                        )),
                    }
                }
            }
            Expr::Const(const_expr) => {
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut const_expr.block.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::Block(block_expr) => {
                // This is a new scope.
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut block_expr.block.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::Unsafe(block_expr) => {
                // This is a new scope.
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut block_expr.block.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::ForLoop(for_expr) => {
                // This is a new scope.
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut for_expr.body.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::While(while_expr) => {
                // While loop: while condition { body }
                // Rewrite condition and body
                self.rewrite_expr(&mut while_expr.cond, const_instances, variables, None)?;
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut while_expr.body.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::Loop(loop_expr) => {
                // Infinite loop: loop { body }
                // Rewrite body
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut loop_expr.body.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::If(if_expr) => {
                self.rewrite_expr(
                    &mut if_expr.cond,
                    const_instances,
                    variables,
                    return_type.clone(),
                )?;
                if let Some((_Else, else_expr)) = &mut if_expr.else_branch {
                    let mut block_vars = variables.fork();
                    self.rewrite_expr(
                        else_expr,
                        const_instances,
                        &mut block_vars,
                        return_type.clone(),
                    )?;
                }
                let mut block_vars = variables.fork();
                self.rewrite_statements(
                    &mut if_expr.then_branch.stmts,
                    const_instances,
                    &mut block_vars,
                    return_type.clone(),
                )
            }
            Expr::Continue(_continue_expr) => Ok(None),
            Expr::Break(_break_expr) => Ok(None),
            Expr::Call(call_expr) => {
                self.rewrite_call(call_expr, const_instances, variables, return_type.clone())
            }
            Expr::MethodCall(method_call_expr) => self.rewrite_method_call(
                method_call_expr,
                const_instances,
                variables,
                return_type.clone(),
            ),
            Expr::Cast(cast_expr) => {
                self.rewrite_expr(
                    &mut cast_expr.expr,
                    const_instances,
                    variables,
                    return_type.clone(),
                )?;
                *cast_expr.ty = desugar_ty(&cast_expr.ty, const_instances)?;
                Ok(return_type)
            }
            Expr::Path(path_expr) => {
                // TODO (hme): Unclear whether there's anything that needs to be rewritten at this point.
                let path_span = path_expr.span();
                let last_seg = path_expr
                    .path
                    .segments
                    .last_mut()
                    .ok_or_else(|| syn_err(path_span, "Expected at least one path segment"))?;
                let name = last_seg.ident.to_string();
                if let Some(n) = const_instances.inst_u32.get(name.as_str()) {
                    // This is a const generic primitive.
                    let new_expr = syn::parse::<Expr>(
                        format!("{n}")
                            .parse()
                            .map_err(|_| syn_err(path_span, &format!("Failed to parse '{n}'")))?,
                    )
                    .map_err(|e| syn_err(path_span, &format!("Failed to parse '{n}': {e}")))?;
                    *expr = new_expr;
                    return Ok(Some(
                        syn::parse::<Type>(
                            "i32"
                                .parse()
                                .map_err(|_| syn_err(path_span, "Failed to parse 'i32'"))?,
                        )
                        .map_err(|e| syn_err(path_span, &format!("Failed to parse 'i32': {e}")))?,
                    ));
                }
                self.rewrite_path_expr_type(
                    path_expr,
                    const_instances,
                    variables,
                    return_type.clone(),
                )
            }
            Expr::Reference(ref_expr) => self.rewrite_expr(
                &mut ref_expr.expr,
                const_instances,
                variables,
                return_type.clone(),
            ),
            Expr::Return(return_expr) => match &mut return_expr.expr {
                Some(return_expr) => self.rewrite_expr(
                    &mut *return_expr,
                    const_instances,
                    variables,
                    return_type.clone(),
                ),
                None => Ok(return_type),
            },
            Expr::Assign(assign_expr) => self.rewrite_expr(
                &mut assign_expr.right,
                const_instances,
                variables,
                return_type.clone(),
            ),
            Expr::Unary(unary_expr) => self.rewrite_expr(
                &mut unary_expr.expr,
                const_instances,
                variables,
                return_type.clone(),
            ),
            Expr::Binary(bin_expr) => {
                self.rewrite_expr(
                    &mut bin_expr.left,
                    const_instances,
                    variables,
                    return_type.clone(),
                )?;
                self.rewrite_expr(
                    &mut bin_expr.right,
                    const_instances,
                    variables,
                    return_type.clone(),
                )?;
                Ok(return_type)
            }
            Expr::Tuple(tuple_expr) => {
                for elem_expr in tuple_expr.elems.iter_mut() {
                    self.rewrite_expr(elem_expr, const_instances, variables, None)?;
                }
                Ok(return_type)
            }
            Expr::Array(arr_expr) => {
                for elem_expr in arr_expr.elems.iter_mut() {
                    self.rewrite_expr(elem_expr, const_instances, variables, None)?;
                }
                Ok(return_type)
            }
            Expr::Repeat(repeat_expr) => {
                self.rewrite_expr(&mut repeat_expr.len, const_instances, variables, None)?;
                Ok(return_type)
            }
            Expr::Field(field_expr) => {
                return_type = self.rewrite_expr(
                    &mut field_expr.base,
                    const_instances,
                    variables,
                    return_type.clone(),
                )?;
                Ok(return_type)
            }
            Expr::Struct(struct_expr) => {
                // TODO (hme): Similar code fragment in desugar_ty.
                //  Can this be refactored into a rewrite for any PathSegment?
                if struct_expr.path.segments.is_empty() {
                    return Err(syn_err(
                        struct_expr.span(),
                        "Expected at least one path segment in struct expression",
                    ));
                }
                let last_seg = struct_expr.path.segments.last_mut().unwrap();
                let name = last_seg.ident.to_string();
                let vtd = get_variadic_type_data(name.as_str());
                if let Some(_vtd) = vtd {
                    if return_type.is_none() {
                        return Err(syn_err(
                            struct_expr.span(),
                            "Variadic structs require a static type annotation. Try assigning to a statically typed let binding.",
                        ));
                    }
                    let (last_type_ident, last_seg_args) = match &last_seg.arguments {
                        PathArguments::AngleBracketed(type_params) => {
                            let (type_ident, last_seg_args) =
                                desugar_cga(const_instances, &last_seg.ident, type_params)?;
                            (
                                type_ident.clone(),
                                PathArguments::AngleBracketed(last_seg_args),
                            )
                        }
                        PathArguments::None => (last_seg.ident.clone(), PathArguments::None),
                        _ => {
                            return Err(syn_err(
                                struct_expr.span(),
                                "Unexpected Path arguments.",
                            ))
                        }
                    };
                    *last_seg = PathSegment {
                        ident: last_type_ident,
                        arguments: last_seg_args,
                    };
                }
                for field in &mut struct_expr.fields {
                    self.rewrite_expr(&mut field.expr, const_instances, variables, None)?;
                    // This check may not be necessary.
                    match &mut field.member {
                        Member::Named(_named) => {}
                        Member::Unnamed(_idx) => {
                            return Err(syn_err(struct_expr.span(), "Tuples not supported."))
                        }
                    }
                }
                Ok(return_type)
            }
            Expr::Macro(mac_expr) => {
                let last_seg = mac_expr.mac.path.segments.last();
                if last_seg.is_none() {
                    return Ok(return_type);
                }
                // TODO (hme): Revisit why we do this vs just letting the macros in _core expand.
                // TODO (hme): Implement a function to infer known variadic types.
                let last_seg = last_seg.unwrap();
                let mac_name = last_seg.ident.to_string();
                match mac_name.as_str() {
                    "const_shape" | "const_array" => {
                        // TODO (hme): Remove special case for const_shape here
                        //  and in compiler.rs (JIT side).
                        let mut args = vec![];
                        #[allow(unused_variables)]
                        let mut is_cga = false;
                        #[allow(unused_variables)]
                        let mut is_consts = false;
                        for token in mac_expr.mac.tokens.clone() {
                            match token {
                                TokenTree::Literal(lit) => {
                                    args.push(lit.to_string());
                                }
                                TokenTree::Ident(ident) => {
                                    let const_var = ident.to_string();
                                    if let Some(_cga) = const_instances.inst_array.get(&const_var) {
                                        is_cga = true;
                                        let path: Path = parse_quote! { #ident };
                                        let generic_args = expand_cga(&path, const_instances)?;
                                        args = generic_args
                                            .args
                                            .iter()
                                            .map(|x| x.to_token_stream().to_string())
                                            .collect::<Vec<String>>();
                                    } else {
                                        is_consts = true;
                                        args.push(const_var);
                                    }
                                }
                                // TODO (hme): We should support something like this.
                                //  Macro support needs to be rethought / rewritten.
                                // TokenTree::Group(group) => {
                                //     let last_token = args.remove(args.len() - 1);
                                //     args.push(format!("{}{}", last_token, group.to_string()));
                                // },
                                TokenTree::Punct(punct) => {
                                    if punct.as_char() == ',' {
                                        continue;
                                    } else {
                                        return Err(syn_err(
                                            mac_expr.span(),
                                            &format!("Unexpected punctuation {punct:}"),
                                        ));
                                    }
                                }
                                _ => {
                                    return Err(syn_err(
                                        mac_expr.span(),
                                        &format!("Unexpected token {:?}", token),
                                    ))
                                }
                            }
                        }
                        let cga_str = format!("{{[{}]}}", args.join(", "));
                        let ty_str = if mac_name == "const_shape" {
                            "Shape"
                        } else {
                            "Array"
                        };
                        let mac_span = mac_expr.span();
                        let shape_fmt = format!("{ty_str}::<{cga_str}>::const_new()");
                        let shape_expr = syn::parse2::<Expr>(shape_fmt.parse().map_err(|_| {
                            syn_err(mac_span, &format!("Failed to parse '{shape_fmt}'"))
                        })?)
                        .map_err(|e| {
                            syn_err(mac_span, &format!("Failed to parse '{shape_fmt}': {e}"))
                        })?;
                        *expr = shape_expr;
                        let shape_str = format!("{ty_str}<{cga_str}>");
                        let shape_ty = syn::parse::<Type>(shape_str.parse().map_err(|_| {
                            syn_err(mac_span, &format!("Failed to parse '{shape_str}'"))
                        })?)
                        .map_err(|e| {
                            syn_err(mac_span, &format!("Failed to parse '{shape_str}': {e}"))
                        })?;
                        // println!("Expr::Macro const_shape: {shape_expr}, {shape_ty:#?}");
                        self.rewrite_expr(&mut *expr, const_instances, variables, Some(shape_ty))
                    }
                    _ => Ok(return_type),
                }
            }
            Expr::Lit(_lit_expr) => Ok(return_type),
            Expr::Paren(paren_expr) => {
                return_type = self.rewrite_expr(
                    &mut paren_expr.expr,
                    const_instances,
                    variables,
                    return_type.clone(),
                )?;
                Ok(return_type)
            }
            Expr::Closure(_closure_expr) => {
                // Closures are passed through unchanged to the compiler
                // The compiler will handle parsing and compilation of closure bodies
                Ok(return_type)
            }
            _ => Err(syn_err(
                expr.span(),
                "Expression type not supported",
            )),
        }
    }

    fn rewrite_path_expr_type(
        &self,
        expr: &mut ExprPath,
        const_instances: &ConstInstances,
        variables: &mut TrainMap<String, Binding>,
        return_type: Option<Type>,
    ) -> Result<Option<Type>, Error> {
        let result_path = desugar_path(&expr.path, const_instances)?;
        expr.path = result_path;
        if expr.path.segments.is_empty() {
            // TODO (hme): What would this be?
            return Ok(None);
        }
        let last_seg = expr.path.segments.last_mut().unwrap();
        let name = last_seg.ident.to_string();
        // println!("rewrite_path_expr_type: name={}, is_var={}", name, variables.get(&name).is_some());
        Ok(match variables.get(&name) {
            Some(var) => var.ty.clone(),
            None => return_type, // Don't panic here. We just need to generate correct names for variadic types and ops.
        })
    }

    fn try_get_var<'a>(
        &self,
        maybe_path_expr: &mut Expr,
        variables: &'a TrainMap<String, Binding>,
    ) -> Result<Option<&'a Binding>, Error> {
        Ok(match try_get_path_expr_ident_str(maybe_path_expr)? {
            Some(name) => variables.get(&name),
            None => None,
        })
    }

    fn rewrite_method_call(
        &self,
        expr: &mut ExprMethodCall,
        const_instances: &ConstInstances,
        variables: &mut TrainMap<String, Binding>,
        return_type: Option<Type>,
    ) -> Result<Option<Type>, Error> {
        // Derive name of method call.
        let method_ident = &expr.method;
        let method_name = method_ident.to_string();
        let self_ty =
            match self.rewrite_expr(&mut expr.receiver, const_instances, variables, None)? {
                Some(ty) => ty,
                None => {
                    return Err(syn_err(
                        expr.receiver.span(),
                        "Unable to infer receiver type",
                    ))
                }
            };
        // This may be a primitive which implements a supported variadic trait.
        let maybe_primitive_type = self_ty.to_token_stream().to_string();
        let variadic_meta = get_variadic_trait_impl_meta_data(&maybe_primitive_type, &method_name)?;
        let variadic_meta = if variadic_meta.is_some() {
            variadic_meta
        } else {
            // It may also be a struct method impl corresponding to a variadic function.
            get_variadic_method_meta_data(&self_ty, &method_name)?
        };
        if let Some((_op_name, _vtd, vod)) = variadic_meta {
            let rtype = return_type.clone();
            // Derive name of method call.
            let mut maybe_input_types = vec![Some(self_ty.clone())];
            for arg in &mut expr.args {
                maybe_input_types.push(self.rewrite_expr(arg, const_instances, variables, None)?);
            }
            let (concrete_ident, inferred_rtype) = get_concrete_op_or_method_ident_from_types(
                vod,
                method_ident,
                &maybe_input_types,
                rtype.clone(),
                const_instances,
                false,
            )?;
            // if method_name.to_string() == "broadcast" {
            //     println!("\nrewrite_method_call: {}", expr.to_token_stream().to_string());
            //     println!("maybe_input_types: {:#?}", maybe_input_types);
            //     println!("concrete_ident: {:#?}", concrete_ident);
            //     println!("inferred_rtype: {:#?}", inferred_rtype);
            //     println!("\n");
            // }
            expr.method = concrete_ident;
            if inferred_rtype.is_some() {
                Ok(inferred_rtype)
            } else {
                Ok(rtype)
            }
        } else {
            Ok(return_type)
        }
    }

    fn rewrite_call(
        &self,
        expr: &mut ExprCall,
        const_instances: &ConstInstances,
        variables: &mut TrainMap<String, Binding>,
        return_type: Option<Type>,
    ) -> Result<Option<Type>, Error> {
        // println!("rewrite_call: {:#?} {:#?}", expr, return_type);
        let vod = get_vod_from_call(expr)?;
        // if expr.func.to_token_stream().to_string() == "reduce_max" {
        //     println!("\nrewrite_call: {}", expr.to_token_stream().to_string());
        //     println!("return_type: {:#?}", return_type);
        //     println!("vod: {:#?}", vod);
        //     println!("\n");
        // }
        let maybe_inferred_rtype = match vod {
            Some(_vod) => {
                let rtype = return_type.clone();
                // Actually perform function call rewrite.
                let last_seg = match &mut *expr.func {
                    Expr::Path(path_expr) => {
                        if path_expr.path.segments.is_empty() {
                            return Err(syn_err(
                                path_expr.span(),
                                "Expected at least one path segment in function call",
                            ));
                        }
                        path_expr.path.segments.last_mut().unwrap()
                    }
                    _ => {
                        return Err(syn_err(
                            expr.func.span(),
                            "Unexpected function call expression.",
                        ))
                    }
                };
                let mut maybe_input_types = vec![];
                for arg in &mut expr.args {
                    maybe_input_types.push(self.rewrite_expr(
                        arg,
                        const_instances,
                        variables,
                        None,
                    )?);
                }
                // if last_seg.ident.to_string().as_str() == "make_partition_view" {
                //     println!("rewrite_call {}: {:#?}\n{maybe_input_types:#?}", last_seg.ident.to_string().as_str(), expr.args);
                // }
                let (concrete_ident, inferred_rtype) = get_concrete_op_ident_from_types(
                    &last_seg.ident,
                    &maybe_input_types,
                    rtype.clone(),
                    const_instances,
                    false,
                )?;
                // if last_seg.ident.to_string().as_str() == "reduce_max" {
                //     println!("rewrite_call {}: {concrete_ident:#?}, \nmaybe_input_types={maybe_input_types:#?}, \ninferred_rtype={inferred_rtype:#?}, \nrtype={rtype:#?}", last_seg.ident.to_string().as_str());
                // }
                last_seg.ident = concrete_ident;
                if inferred_rtype.is_some() {
                    inferred_rtype
                } else {
                    rtype
                }
            }
            None => {
                for arg in &mut expr.args {
                    self.rewrite_expr(arg, const_instances, variables, None)?;
                }
                return_type
            }
        };
        self.rewrite_expr(&mut expr.func, const_instances, variables, None)?;
        // println!("rewrite_call {}: maybe_inferred_rtype = {maybe_inferred_rtype:#?}", expr.to_token_stream().to_string());
        Ok(maybe_inferred_rtype)
    }
}

/// Desugars const generic array syntax in a struct definition.
///
/// Transforms const generic array parameters (e.g., `const D: [i32; N]`) into
/// individual const generic parameters (e.g., `const D_0: i32, const D_1: i32, ...`).
/// This makes the types compatible with Rust's type system and MLIR code generation.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// struct MyStruct<const S: [i32; 2]> { }
///
/// // Output:
/// struct MyStruct<const S_0: i32, const S_1: i32> { }
/// ```
pub fn desugar_structure_cgas(item: &ItemStruct) -> Result<ItemStruct, Error> {
    let const_instances = ConstInstances::from_generics(&item.generics)?;
    let rewrite_pass = RewriteVariadicsPass {};
    rewrite_pass.rewrite_struct(item, &const_instances)
}

/// Desugars const generic array syntax in a function definition.
///
/// Rewrites function signatures and bodies to replace const generic arrays
/// with expanded const generic parameters.
///
/// ## Examples
///
/// ```rust,ignore
/// // Input:
/// fn my_fn<const S: [i32; 2]>(x: Tile<f32, S>) -> Shape<S> { }
///
/// // Output:
/// fn my_fn<const S_0: i32, const S_1: i32>(x: Tile_2<f32, S_0, S_1>) -> Shape_2<S_0, S_1> { }
/// ```
pub fn desugar_function_cgas(item: &ItemFn) -> Result<ItemFn, Error> {
    let rewrite_pass = RewriteVariadicsPass {};
    let const_instances = ConstInstances::from_generics(&item.sig.generics)?;
    rewrite_pass.rewrite_function(item, &const_instances)
}

/// Desugars const generic array syntax in an impl block.
///
/// Transforms impl blocks to use desugared const generic parameters.
pub fn desugar_impl_cgas(item: &ItemImpl) -> Result<ItemImpl, Error> {
    let rewrite_pass = RewriteVariadicsPass {};
    let const_instances = ConstInstances::from_generics(&item.generics)?;
    rewrite_pass.rewrite_impl(item, &const_instances)
}

/// Desugars const generic array syntax in a trait definition.
///
/// Transforms trait definitions to use desugared const generic parameters.
pub fn desugar_trait_cgas(item: &ItemTrait) -> Result<ItemTrait, Error> {
    // Nothing to do here due to current features.
    let rewrite_pass = RewriteVariadicsPass {};
    let const_instances = ConstInstances::from_generics(&item.generics)?;
    rewrite_pass.rewrite_trait(item, &const_instances)
}
