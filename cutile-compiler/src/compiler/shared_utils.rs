/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure-Rust utility functions and types used by compiler2.

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use crate::syn_utils::get_ident_from_path_expr;
use half::{bf16, f16};
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use std::collections::{BTreeSet, HashSet};
use std::fmt::{Debug, LowerHex};
use std::hash::Hash;
use syn::{Expr, Pat, Stmt};

use super::_value::{CompilerContext, TileRustValue};

// ---------------------------------------------------------------------------
// Stack management constants
// ---------------------------------------------------------------------------

/// Minimum remaining stack space before growing (1 MiB).
pub(crate) const STACK_RED_ZONE: usize = 1024 * 1024;
/// Size of each new stack segment when growth is needed (10 MiB).
pub(crate) const STACK_GROW_SIZE: usize = 10 * 1024 * 1024;

// ---------------------------------------------------------------------------
// AtomicMode
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
/// Supported atomic read-modify-write modes.
pub enum AtomicMode {
    And = 0,
    Or = 1,
    Xor = 2,
    Add = 3,
    AddF = 4,
    Max = 5,
    Min = 6,
    UMax = 7,
    UMin = 8,
    XChg = 9,
}

// ---------------------------------------------------------------------------
// ElementTypePrefix
// ---------------------------------------------------------------------------

#[derive(Debug, Eq, PartialEq)]
/// Whether an element type is floating-point or integer.
pub enum ElementTypePrefix {
    Float,
    Integer,
}

impl ElementTypePrefix {
    /// Determines the prefix from a CUDA Tile element type string (e.g. `"f32"` -> `Float`).
    pub fn new(cuda_elem_ty_str: &str) -> Result<Self, JITError> {
        match super::_type::scalar_from_name(cuda_elem_ty_str) {
            Some(s) if s.is_float() => Ok(ElementTypePrefix::Float),
            Some(_) => Ok(ElementTypePrefix::Integer),
            None => SourceLocation::unknown()
                .jit_error_result(&format!("unsupported element type `{cuda_elem_ty_str}`")),
        }
    }
}

impl AtomicMode {
    /// Parses an atomic mode from the ZST type identifier (e.g. `AddF`),
    /// validating compatibility with the element type.
    pub fn new(mode: &str, elem_ty_prefix: ElementTypePrefix) -> Result<Self, JITError> {
        let result = match mode {
            "And" => AtomicMode::And,
            "Or" => AtomicMode::Or,
            "Xor" => AtomicMode::Xor,
            "Add" => AtomicMode::Add,
            "AddF" => AtomicMode::AddF,
            "Max" => AtomicMode::Max,
            "Min" => AtomicMode::Min,
            "Umax" => AtomicMode::UMax,
            "Umin" => AtomicMode::UMin,
            "Xchg" => AtomicMode::XChg,
            _ => return SourceLocation::unknown().jit_error_result(
                &format!("invalid atomic mode `{mode}`; valid modes are: And, Or, Xor, Add, AddF, Max, Min, Umax, Umin, Xchg"),
            ),
        };
        if elem_ty_prefix == ElementTypePrefix::Float
            && ![AtomicMode::XChg, AtomicMode::AddF].contains(&result)
        {
            return SourceLocation::unknown().jit_error_result(&format!(
                "float types only support `Xchg` and `AddF` atomic modes, got `{:?}`",
                result
            ));
        }
        Ok(result)
    }
}

// Re-export from bounds.rs (canonical location shared with old compiler).
pub use crate::bounds::{get_binary_op_from_op_str, get_tile_bop_from_rust_bop, TileBinaryOp};

// ---------------------------------------------------------------------------
// Constant hex encoding
// ---------------------------------------------------------------------------

fn format_hex<T: LowerHex>(val: T) -> String {
    format!("0x{:x}", val)
}

trait Float {
    fn to_hex(&self) -> String;
    fn zero() -> Self;
    fn one() -> Self;
    fn negative_infinity() -> Self;
    fn positive_infinity() -> Self;
    fn e() -> Self;
}

impl Float for f16 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> f16 {
        f16::ZERO
    }
    fn one() -> f16 {
        f16::ONE
    }
    fn negative_infinity() -> f16 {
        f16::NEG_INFINITY
    }
    fn positive_infinity() -> f16 {
        f16::INFINITY
    }
    fn e() -> f16 {
        f16::E
    }
}

impl Float for bf16 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> bf16 {
        bf16::ZERO
    }
    fn one() -> bf16 {
        bf16::ONE
    }
    fn negative_infinity() -> bf16 {
        bf16::NEG_INFINITY
    }
    fn positive_infinity() -> bf16 {
        bf16::INFINITY
    }
    fn e() -> bf16 {
        bf16::E
    }
}

impl Float for f32 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> f32 {
        0.0f32
    }
    fn one() -> f32 {
        1.0f32
    }
    fn negative_infinity() -> f32 {
        f32::NEG_INFINITY
    }
    fn positive_infinity() -> f32 {
        f32::INFINITY
    }
    fn e() -> f32 {
        std::f32::consts::E
    }
}

impl Float for f64 {
    fn to_hex(&self) -> String {
        format_hex(self.to_bits())
    }
    fn zero() -> f64 {
        0.0f64
    }
    fn one() -> f64 {
        1.0f64
    }
    fn negative_infinity() -> f64 {
        f64::NEG_INFINITY
    }
    fn positive_infinity() -> f64 {
        f64::INFINITY
    }
    fn e() -> f64 {
        std::f64::consts::E
    }
}

trait Integer
where
    Self: LowerHex,
{
    fn to_hex(&self) -> String {
        format_hex(self)
    }
    fn zero() -> Self;
    fn one() -> Self;
    fn min() -> Self;
    fn max() -> Self;
}

impl Integer for i32 {
    fn zero() -> i32 {
        0i32
    }
    fn one() -> i32 {
        1i32
    }
    fn min() -> i32 {
        i32::MIN
    }
    fn max() -> i32 {
        i32::MAX
    }
}
impl Integer for i64 {
    fn zero() -> i64 {
        0i64
    }
    fn one() -> i64 {
        1i64
    }
    fn min() -> i64 {
        i64::MIN
    }
    fn max() -> i64 {
        i64::MAX
    }
}
impl Integer for u32 {
    fn zero() -> u32 {
        0u32
    }
    fn one() -> u32 {
        1u32
    }
    fn min() -> u32 {
        u32::MIN
    }
    fn max() -> u32 {
        u32::MAX
    }
}
impl Integer for u64 {
    fn zero() -> u64 {
        0u64
    }
    fn one() -> u64 {
        1u64
    }
    fn min() -> u64 {
        u64::MIN
    }
    fn max() -> u64 {
        u64::MAX
    }
}

fn get_float_const<T: Float>(const_str: &str) -> Result<String, JITError> {
    match const_str {
        "zero" => Ok(T::zero().to_hex()),
        "one" => Ok(T::one().to_hex()),
        "min" => Ok(T::negative_infinity().to_hex()),
        "max" => Ok(T::positive_infinity().to_hex()),
        "e" => Ok(T::e().to_hex()),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("Unsupported float constant type {}.", const_str)),
    }
}

fn get_integer_const<T: Integer>(const_str: &str) -> Result<String, JITError> {
    match const_str {
        "zero" => Ok(T::zero().to_hex()),
        "one" => Ok(T::one().to_hex()),
        "min" => Ok(T::min().to_hex()),
        "max" => Ok(T::max().to_hex()),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("Unsupported integer constant type {}.", const_str)),
    }
}

/// Returns the hex-encoded constant string for a typed constant name (e.g. `"zero"`, `"one"`).
pub fn get_const_hex(rust_element_type_str: &str, const_str: &str) -> Result<String, JITError> {
    match rust_element_type_str {
        "bf16" => get_float_const::<bf16>(const_str),
        "f16" => get_float_const::<f16>(const_str),
        "f32" => get_float_const::<f32>(const_str),
        "f64" => get_float_const::<f64>(const_str),
        "i32" => get_integer_const::<i32>(const_str),
        "i64" => get_integer_const::<i64>(const_str),
        "u32" => get_integer_const::<u32>(const_str),
        "u64" => get_integer_const::<u64>(const_str),
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "Unsupported constant type {} {}.",
            rust_element_type_str, const_str
        )),
    }
}

// ---------------------------------------------------------------------------
// String literal / option arg extraction
// ---------------------------------------------------------------------------

/// Reads the last path-segment ident from a ZST type-as-value expression.
///
/// Examples: `atomic::AddF` -> `"AddF"`, `Enabled` -> `"Enabled"`.
/// Used to resolve attribute selectors that the DSL surfaces as ZST type
/// arguments (`mode: atomic::Mode`, `memory_ordering: ordering::Mode`, etc.).
pub fn extract_zst_type_name(expr: &syn::Expr, param_name: &str) -> Result<String, JITError> {
    use syn::Expr;
    match expr {
        Expr::Path(path) => Ok(path.path.segments.last().unwrap().ident.to_string()),
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "`{param_name}` must be a unit-struct type-as-value path, got `{}`",
            expr.to_token_stream().to_string()
        )),
    }
}

pub fn padding_zst_value(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Path(path) = expr else {
        return None;
    };
    let ident = path.path.segments.last()?.ident.to_string();
    match ident.as_str() {
        "Zero" => Some("zero".to_string()),
        "NegZero" => Some("neg_zero".to_string()),
        "Nan" => Some("nan".to_string()),
        "PosInf" => Some("pos_inf".to_string()),
        "NegInf" => Some("neg_inf".to_string()),
        _ => None,
    }
}

pub fn zst_type_name(expr: &syn::Expr) -> Option<String> {
    let syn::Expr::Path(path) = expr else {
        return None;
    };
    Some(path.path.segments.last()?.ident.to_string())
}

/// Extracts a string literal value from an expression, handling both direct literals
/// and variables that were bound from string literals.
pub fn extract_string_literal(
    expr: &syn::Expr,
    param_name: &str,
    ctx: &CompilerContext,
) -> Result<String, JITError> {
    use syn::{Expr, ExprLit, Lit};

    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        Expr::Path(path_expr) => {
            let var_name = path_expr.path.segments.last().unwrap().ident.to_string();
            if let Some(val) = ctx.vars.get(&var_name) {
                if let Some(Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                })) = &val.string_literal
                {
                    return Ok(s.value());
                }
            }
            SourceLocation::unknown().jit_error_result(&format!(
                "`{param_name}` must be a string literal, but got variable `{var_name}`; \
                     ensure string literals are passed directly",
            ))
        }
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "`{param_name}` must be a string literal, got `{}`",
            expr.to_token_stream()
        )),
    }
}

/// Helper to resolve compile-time optional argument.
/// Returns the inner expression if it is Some(expr), or None if it is None.
pub fn resolve_option_arg(expr: &syn::Expr, ctx: &CompilerContext) -> Option<syn::Expr> {
    use syn::Expr;
    if let Expr::Call(call) = expr {
        if let Expr::Path(path) = &*call.func {
            if path.path.segments.last().unwrap().ident == "Some" {
                return call.args.first().cloned();
            }
        }
    } else if let Expr::Path(path) = expr {
        if path.path.segments.len() == 1 && path.path.segments.last().unwrap().ident == "None" {
            return None;
        }
        let var_name = path.path.segments.last().unwrap().ident.to_string();
        if let Some(val) = ctx.vars.get(&var_name) {
            if let Some(variant) = &val.enum_variant {
                return match variant.as_str() {
                    "Some" => val.enum_payload.as_deref().cloned(),
                    "None" => None,
                    _ => None,
                };
            }
            if let Some(ast) = &val.string_literal {
                if let Expr::Call(call) = ast {
                    if let Expr::Path(path) = &*call.func {
                        if path.path.segments.last().unwrap().ident == "Some" {
                            return call.args.first().cloned();
                        }
                    }
                } else if let Expr::Path(path) = ast {
                    if path.path.segments.len() == 1
                        && path.path.segments.last().unwrap().ident == "None"
                    {
                        return None;
                    }
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Variable mutation analysis
// ---------------------------------------------------------------------------

/// Collects the names of variables assigned (mutated) in a block that were defined outside it.
pub fn collect_mutated_variables_from_block(
    block: &syn::Block,
) -> Result<BTreeSet<String>, JITError> {
    let mut local_vars: HashSet<String> = HashSet::new();
    let mut result: BTreeSet<String> = BTreeSet::new();
    for statement in block.stmts.iter() {
        match statement {
            Stmt::Local(local) => {
                let mut var_names: Vec<String> = vec![];
                match &local.pat {
                    Pat::Type(pat_type) => match &*pat_type.pat {
                        Pat::Ident(pat_ident) => {
                            var_names.push(pat_ident.ident.to_string());
                        }
                        Pat::Tuple(pat_tuple) => {
                            for elem in &pat_tuple.elems {
                                match elem {
                                    Pat::Ident(ident) => {
                                        var_names.push(ident.ident.to_string());
                                    }
                                    _ => {
                                        return SourceLocation::unknown().jit_error_result(
                                            "Only identifier patterns supported in tuple destructuring",
                                        );
                                    }
                                }
                            }
                        }
                        _ => {
                            return SourceLocation::unknown().jit_error_result(&format!(
                                "let binding LHS not implemented {:#?}.",
                                pat_type.pat
                            ));
                        }
                    },
                    Pat::Tuple(pat_tuple) => {
                        for elem in &pat_tuple.elems {
                            match elem {
                                Pat::Ident(ident) => {
                                    var_names.push(ident.ident.to_string());
                                }
                                _ => {
                                    return SourceLocation::unknown().jit_error_result(
                                        "Only identifier patterns supported in tuple destructuring",
                                    );
                                }
                            }
                        }
                    }
                    Pat::Ident(pat_ident) => {
                        var_names.push(pat_ident.ident.to_string());
                    }
                    _ => {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "Local pattern type not supported {:#?}",
                            local.pat
                        ));
                    }
                }
                if var_names.is_empty() {
                    return SourceLocation::unknown()
                        .jit_error_result("failed to parse variable name in let expression");
                }
                local_vars.extend(var_names);
            }
            Stmt::Expr(Expr::Assign(assign_expr), _) => {
                let var_name: String = match &*assign_expr.left {
                    Expr::Path(path_expr) => get_ident_from_path_expr(path_expr).to_string(),
                    _ => {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "LHS assign expression not implemented {:#?}.",
                            assign_expr.left
                        ));
                    }
                };
                if !local_vars.contains(&var_name) {
                    result.insert(var_name);
                }
            }
            // Recurse into control-flow expressions to find nested mutations.
            Stmt::Expr(Expr::ForLoop(for_expr), _) => {
                let inner = collect_mutated_variables_from_block(&for_expr.body)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            Stmt::Expr(Expr::While(while_expr), _) => {
                let inner = collect_mutated_variables_from_block(&while_expr.body)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            Stmt::Expr(Expr::Loop(loop_expr), _) => {
                let inner = collect_mutated_variables_from_block(&loop_expr.body)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
            }
            Stmt::Expr(Expr::If(if_expr), _) => {
                let inner = collect_mutated_variables_from_block(&if_expr.then_branch)?;
                for name in inner {
                    if !local_vars.contains(&name) {
                        result.insert(name);
                    }
                }
                if let Some((_else, else_expr)) = &if_expr.else_branch {
                    if let Expr::Block(block_expr) = &**else_expr {
                        let inner = collect_mutated_variables_from_block(&block_expr.block)?;
                        for name in inner {
                            if !local_vars.contains(&name) {
                                result.insert(name);
                            }
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    Ok(result)
}

/// Collects mutated outer-scope variables from a for-loop body.
pub fn collect_mutated_variables(
    for_expr: &syn::ExprForLoop,
) -> Result<BTreeSet<String>, JITError> {
    collect_mutated_variables_from_block(&for_expr.body)
}

/// Collects mutated outer-scope variables from a while-loop body.
pub fn collect_mutated_variables_while(
    while_expr: &syn::ExprWhile,
) -> Result<BTreeSet<String>, JITError> {
    collect_mutated_variables_from_block(&while_expr.body)
}

/// Collects mutated outer-scope variables from a loop body.
pub fn collect_mutated_variables_loop(
    loop_expr: &syn::ExprLoop,
) -> Result<BTreeSet<String>, JITError> {
    collect_mutated_variables_from_block(&loop_expr.body)
}

// ---------------------------------------------------------------------------
// Misc utilities
// ---------------------------------------------------------------------------

/// Removes duplicate elements from a vector while preserving order.
pub fn dedup<T: Hash + Eq + Clone>(v: &mut Vec<T>) {
    let mut set = HashSet::new();
    v.retain(|x| set.insert(x.clone()));
}

/// Parses a comma-separated token stream into a list of `syn::Expr`.
pub fn parse_list_of_expr(tokens: TokenStream) -> Result<Vec<Expr>, JITError> {
    let mut args: Vec<Expr> = vec![];
    let mut arg_expr: Vec<TokenTree> = vec![];
    for token in tokens.clone().into_iter() {
        match &token {
            TokenTree::Literal(_lit) => {
                arg_expr.push(token.clone());
            }
            TokenTree::Ident(_ident) => {
                arg_expr.push(token.clone());
            }
            TokenTree::Punct(punct) => {
                if punct.as_char() == ',' {
                    if !arg_expr.is_empty() {
                        let expr =
                            syn::parse2::<syn::Expr>(arg_expr.into_iter().collect()).unwrap();
                        args.push(expr);
                    }
                    arg_expr = vec![];
                } else {
                    arg_expr.push(token.clone());
                }
            }
            _ => {
                return SourceLocation::unknown()
                    .jit_error_result(&format!("unexpected token `{}` in expression list", token));
            }
        }
    }
    if !arg_expr.is_empty() {
        let expr = syn::parse2::<syn::Expr>(arg_expr.into_iter().collect()).unwrap();
        args.push(expr);
    }
    Ok(args)
}

// ---------------------------------------------------------------------------
// Token / type_meta helpers
// ---------------------------------------------------------------------------

/// Updates the ordering token in a variable's type metadata.
pub fn update_token(
    var_arg: &Expr,
    new_token: cutile_ir::ir::Value,
    ctx: &mut CompilerContext,
) -> Result<Option<TileRustValue>, JITError> {
    use crate::syn_utils::get_ident_from_expr;
    let Some(var_arg_ident) = get_ident_from_expr(var_arg) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "expected a variable name, got `{}`",
            var_arg.to_token_stream()
        ));
    };
    let var_name = var_arg_ident.to_string();
    let Some(old_value) = ctx.vars.get(var_name.as_str()) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "undefined variable `{var_name}` when updating token"
        ));
    };
    let mut new_value = old_value.clone();
    let Some(new_type_meta) = &mut new_value.type_meta else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` does not have associated type metadata (expected a view type)"
        ));
    };
    let Some(new_token_value) = new_type_meta.fields.get_mut("token") else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` is missing a `token` field (expected a view with an ordering token)"
        ));
    };
    new_token_value.value = Some(new_token);
    Ok(ctx.vars.insert(var_name, new_value))
}

/// Retrieves the ordering token from a variable expression's type metadata.
pub fn get_token_from_expr(
    var_arg: &Expr,
    ctx: &mut CompilerContext,
) -> Result<TileRustValue, JITError> {
    use crate::syn_utils::get_ident_from_expr;
    let Some(var_arg_ident) = get_ident_from_expr(var_arg) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "expected a variable name, got `{}`",
            var_arg.to_token_stream()
        ));
    };
    let var_name = var_arg_ident.to_string();
    get_token(var_name.as_str(), ctx)
}

/// Retrieves the ordering token from a named variable's type metadata.
pub fn get_token(var_name: &str, ctx: &mut CompilerContext) -> Result<TileRustValue, JITError> {
    let Some(value) = ctx.vars.get(var_name) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "undefined variable `{var_name}` when reading token"
        ));
    };
    let Some(value_type_meta) = &value.type_meta else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` does not have associated type metadata (expected a view type)"
        ));
    };
    let Some(value_token_value) = value_type_meta.fields.get("token") else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "variable `{var_name}` is missing a `token` field (expected a view with an ordering token)"
        ));
    };
    Ok(value_token_value.clone())
}

/// Propagates type metadata changes from an inner block context to the outer block context.
pub fn update_outer_block_type_meta(
    inner_block_vars: &mut CompilerContext,
    outer_block_vars: &mut CompilerContext,
    field_name: String,
) {
    let mut var_map = std::collections::HashMap::new();
    for var_name in outer_block_vars.var_keys() {
        var_map.insert(var_name.clone(), var_name.clone());
    }
    update_type_meta(inner_block_vars, outer_block_vars, &var_map, field_name);
}

/// Copies mutable type metadata fields from inner to outer context using a variable name mapping.
pub fn update_type_meta(
    inner_block_vars: &mut CompilerContext,
    outer_block_vars: &mut CompilerContext,
    outer2inner_vars: &std::collections::HashMap<String, String>,
    _field_name: String,
) {
    use super::shared_types::Mutability;
    let outer_keys_ = outer_block_vars.var_keys();
    let outer_keys = outer_keys_
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
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
