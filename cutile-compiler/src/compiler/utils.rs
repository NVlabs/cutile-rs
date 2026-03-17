/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler utility functions: MLIR attribute builders, binary op mappings, token management,
//! optimization hint parsing, constant hex encoding, and variable mutation analysis.

use crate::ast::SourceLocation;
use crate::compiler::_value::{CompilerContext, Mutability, TileRustValue};
use crate::error::{JITError, SpannedJITError};
use crate::generics::TypeInstance;
use crate::syn_utils::{get_ident_from_expr, get_ident_from_path_expr};
use crate::types::{get_cuda_tile_element_type_from_rust_primitive_str, MLIRVariadicArg};
use half::f16;
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::IntegerType;
use melior::ir::*;
use melior::Context;
use mlir_sys::{
    mlirBlockGetFirstOperation, mlirOperationDump, mlirOperationGetNextInBlock,
    mlirOperationGetRegion, mlirOperationVerify, mlirRegionGetFirstBlock, MlirOperation,
};
use proc_macro2::{TokenStream, TokenTree};
use quote::ToTokens;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{Debug, Display, LowerHex};
use std::hash::Hash;
use syn::{BinOp, Expr, ItemImpl, Lit, Pat, Stmt};

/// Creates a unit (flag) MLIR named attribute.
pub fn named_flag_attr<'c>(context: &'c Context, name: &str) -> (Identifier<'c>, Attribute<'c>) {
    (Identifier::new(context, name), Attribute::unit(context))
}

/// Creates a named MLIR array attribute from a slice.
pub fn named_array_attr<'c, T: Clone + Display + Debug>(
    context: &'c Context,
    name: &str,
    arr: &[T],
) -> (Identifier<'c>, Attribute<'c>) {
    (Identifier::new(context, name), array_attr(context, arr))
}

/// Creates an MLIR array attribute from a slice.
pub fn array_attr<'c, T: Clone + Display + Debug>(
    context: &'c Context,
    arr: &[T],
) -> Attribute<'c> {
    let array_type = std::any::type_name::<T>();
    let array_str = arr
        .to_vec()
        .iter()
        .map(|x| format!("{x:#?}"))
        .collect::<Vec<String>>()
        .join(",");
    Attribute::parse(
        context,
        format!("array<{}: {}>", array_type, array_str).as_str(),
    )
    .unwrap()
}

/// Creates a named MLIR string attribute.
pub fn named_str_attr<'c>(
    context: &'c Context,
    name: &str,
    str: &str,
) -> (Identifier<'c>, Attribute<'c>) {
    (
        Identifier::new(context, name),
        StringAttribute::new(context, str).into(),
    )
}

/// Creates a named MLIR type attribute.
pub fn named_type_attr<'c>(
    context: &'c Context,
    name: &str,
    ty: Type<'c>,
) -> (Identifier<'c>, Attribute<'c>) {
    (
        Identifier::new(context, name),
        TypeAttribute::new(ty).into(),
    )
}

/// Parses an attribute string and returns it as a named attribute pair.
pub fn parse_named_attr<'c>(
    context: &'c Context,
    name: &str,
    attr_str: &str,
) -> Result<(Identifier<'c>, Attribute<'c>), JITError> {
    let Some(attr) = Attribute::parse(context, attr_str) else {
        return SourceLocation::unknown().jit_error_result(&format!(
            "failed to parse attribute `{name}` with value `{attr_str}`"
        ));
    };
    Ok((Identifier::new(context, name), attr))
}

/// Creates a named MLIR 64-bit integer attribute.
pub fn named_int_attr<'c>(
    context: &'c Context,
    name: &str,
    value: i64,
) -> (Identifier<'c>, Attribute<'c>) {
    let ty = IntegerType::new(context, 64).into();
    (
        Identifier::new(context, name),
        IntegerAttribute::new(ty, value).into(),
    )
}

/// Creates a CUDA Tile comparison predicate attribute.
pub fn cmp_pred_attr<'c>(
    context: &'c Context,
    pred: &str,
) -> Result<(Identifier<'c>, Attribute<'c>), JITError> {
    parse_named_attr(
        context,
        "comparison_predicate",
        format!("#cuda_tile.comparison_predicate<{}>", pred).as_str(),
    )
}

/// Parses a `!cuda_tile.tile<...>` MLIR type from a shape/element string.
pub fn cuda_tile_tile_ty<'c>(context: &'c Context, str: &str) -> Type<'c> {
    Type::parse(context, format!("!cuda_tile.tile<{str}>").as_str()).unwrap()
}

/// Constructs a `!cuda_tile.tile` MLIR type from a [`TypeInstance`].
pub fn cuda_tile_tile_ty_from_type_instance<'c>(
    context: &'c Context,
    type_instance: &TypeInstance,
    primitives: &HashMap<(String, String), ItemImpl>,
    force_element_type: Option<&str>,
) -> Result<Type<'c>, JITError> {
    match type_instance {
        TypeInstance::StructuredType(structured_type) => {
            if !structured_type.shape.iter().all(|x| *x > 0) {
                return SourceLocation::unknown().jit_error_result(&format!(
                    "all shape dimensions must be positive, got {:?}",
                    structured_type.shape
                ));
            }
            let mut mlir_variadic_arg =
                MLIRVariadicArg::from_structured_type_instance(structured_type, primitives);
            if let Some(force_element_type) = force_element_type {
                mlir_variadic_arg.primitive_type_str = Some(force_element_type.to_string());
            }
            let shape_element_type = mlir_variadic_arg.mlir_str("x", true);
            Ok(cuda_tile_tile_ty(context, shape_element_type.as_str()))
        }
        TypeInstance::ElementType(element_type) => {
            let generic_ty_str = element_type.generic_ty.to_token_stream().to_string();
            let mut ty_str = get_cuda_tile_element_type_from_rust_primitive_str(
                &element_type.rust_element_instance_ty,
                primitives,
            )
            .unwrap_or_else(|| {
                panic!(
                    "failed to determine tile element type for `{}`",
                    generic_ty_str
                )
            });
            if let Some(force_element_type) = force_element_type {
                ty_str = force_element_type.to_string();
            }
            Ok(cuda_tile_tile_ty(context, ty_str.as_str()))
        }
        _ => SourceLocation::unknown().jit_error_result(&format!(
            "type `{}` cannot be used as a tile type",
            type_instance.get_source_type().to_token_stream()
        )),
    }
}

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

#[derive(Debug, Eq, PartialEq)]
/// Whether an element type is floating-point or integer.
pub enum ElementTypePrefix {
    Float,
    Integer,
}

impl ElementTypePrefix {
    /// Determines the prefix from a CUDA Tile element type string (e.g. `"f32"` → `Float`).
    pub fn new(cuda_elem_ty_str: &str) -> Result<Self, JITError> {
        if cuda_elem_ty_str.starts_with("i") {
            Ok(ElementTypePrefix::Integer)
        } else if cuda_elem_ty_str.starts_with("f") {
            Ok(ElementTypePrefix::Float)
        } else {
            SourceLocation::unknown()
                .jit_error_result(&format!("unsupported element type `{cuda_elem_ty_str}`; expected an integer (`i...`) or float (`f...`) type"))
        }
    }
}

impl AtomicMode {
    /// Parses an atomic mode string, validating compatibility with the element type.
    pub fn new(mode: &str, elem_ty_prefix: ElementTypePrefix) -> Result<Self, JITError> {
        let result = match mode {
            "and" => AtomicMode::And,
            "or" => AtomicMode::Or,
            "xor" => AtomicMode::Xor,
            "add" => AtomicMode::Add,
            "addf" => AtomicMode::AddF,
            "max" => AtomicMode::Max,
            "min" => AtomicMode::Min,
            "umax" => AtomicMode::UMax,
            "umin" => AtomicMode::UMin,
            "xchg" => AtomicMode::XChg,
            _ => return SourceLocation::unknown().jit_error_result(
                &format!("invalid atomic mode `{mode}`; valid modes are: and, or, xor, add, addf, max, min, umax, umin, xchg"),
            ),
        };
        if elem_ty_prefix == ElementTypePrefix::Float
            && ![AtomicMode::XChg, AtomicMode::AddF].contains(&result)
        {
            return SourceLocation::unknown().jit_error_result(&format!(
                "float types only support `xchg` and `addf` atomic modes, got `{:?}`",
                result
            ));
        }
        Ok(result)
    }
}

#[derive(Debug, Eq, PartialEq)]
/// Enumeration of all supported binary operations in the CUDA Tile IR.
pub enum TileBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    CeilDiv,
    TrueDiv,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
}

/// Maps a string operation name (e.g. `"add"`, `"ceil_div"`) to a [`TileBinaryOp`].
pub fn get_binary_op_from_op_str(op_str: &str) -> Result<TileBinaryOp, JITError> {
    match op_str {
        "add" => Ok(TileBinaryOp::Add),
        "sub" => Ok(TileBinaryOp::Sub),
        "mul" => Ok(TileBinaryOp::Mul),
        "div" => Ok(TileBinaryOp::Div),
        "ceil_div" => Ok(TileBinaryOp::CeilDiv),
        "true_div" => Ok(TileBinaryOp::TrueDiv),
        "rem" => Ok(TileBinaryOp::Rem),
        "eq" => Ok(TileBinaryOp::Eq),
        "ne" => Ok(TileBinaryOp::Ne),
        "lt" => Ok(TileBinaryOp::Lt),
        "le" => Ok(TileBinaryOp::Le),
        "gt" => Ok(TileBinaryOp::Gt),
        "ge" => Ok(TileBinaryOp::Ge),
        "min" | "min_tile" => Ok(TileBinaryOp::Min),
        "max" | "max_tile" => Ok(TileBinaryOp::Max),
        "and" => Ok(TileBinaryOp::BitAnd),
        "or" => Ok(TileBinaryOp::BitOr),
        "xor" => Ok(TileBinaryOp::BitXor),
        _ => SourceLocation::unknown()
            .jit_error_result(&format!("unrecognized arithmetic operation `{op_str}`")),
    }
}

/// Converts a Rust `syn::BinOp` to the corresponding [`TileBinaryOp`].
pub fn get_tile_bop_from_rust_bop(rust_bin_op: &BinOp) -> Result<TileBinaryOp, JITError> {
    match rust_bin_op {
        BinOp::Add(_) => Ok(TileBinaryOp::Add),
        BinOp::Sub(_) => Ok(TileBinaryOp::Sub),
        BinOp::Mul(_) => Ok(TileBinaryOp::Mul),
        BinOp::Div(_) => Ok(TileBinaryOp::Div),
        BinOp::Rem(_) => Ok(TileBinaryOp::Rem),
        BinOp::Eq(_) => Ok(TileBinaryOp::Eq),
        BinOp::Ne(_) => Ok(TileBinaryOp::Ne),
        BinOp::Lt(_) => Ok(TileBinaryOp::Lt),
        BinOp::Le(_) => Ok(TileBinaryOp::Le),
        BinOp::Gt(_) => Ok(TileBinaryOp::Gt),
        BinOp::Ge(_) => Ok(TileBinaryOp::Ge),
        BinOp::BitAnd(_) => Ok(TileBinaryOp::BitAnd),
        BinOp::BitOr(_) => Ok(TileBinaryOp::BitOr),
        BinOp::BitXor(_) => Ok(TileBinaryOp::BitXor),
        // Booleans lower to i1, so this is the correct choice.
        BinOp::And(_) => Ok(TileBinaryOp::BitAnd),
        BinOp::Or(_) => Ok(TileBinaryOp::BitOr),
        _ => SourceLocation::unknown().jit_error_result("this binary operator is not supported"),
    }
}

/// Returns a comparison predicate MLIR attribute for comparison binary ops, or `None` for others.
pub fn get_cmp_predicate_attr<'c>(
    context: &'c Context,
    expr: &TileBinaryOp,
) -> Result<Option<(Identifier<'c>, Attribute<'c>)>, JITError> {
    // Assume ordered for all.
    match expr {
        TileBinaryOp::Eq => Ok(Some(cmp_pred_attr(context, "equal")?)),
        TileBinaryOp::Ne => Ok(Some(cmp_pred_attr(context, "not_equal")?)),
        TileBinaryOp::Lt => Ok(Some(cmp_pred_attr(context, "less_than")?)),
        TileBinaryOp::Le => Ok(Some(cmp_pred_attr(context, "less_than_or_equal")?)),
        TileBinaryOp::Gt => Ok(Some(cmp_pred_attr(context, "greater_than")?)),
        TileBinaryOp::Ge => Ok(Some(cmp_pred_attr(context, "greater_than_or_equal")?)),
        _ => Ok(None),
    }
}

/// Returns a signedness attribute based on the Rust element type name.
pub fn get_signedness_attr<'c>(
    context: &'c Context,
    key: &str,
    element_type_str: &str,
) -> Result<(Identifier<'c>, Attribute<'c>), JITError> {
    let signedness_str = match element_type_str {
        "bool" | "u32" | "u64" => "unsigned",
        _ => "signed",
    };
    parse_named_attr(
        context,
        key,
        format!("#cuda_tile.signedness<{signedness_str}>").as_str(),
    )
}

/// Updates the ordering token in a variable's type metadata.
pub fn update_token<'c>(
    var_arg: &Expr,
    new_token: Value<'c, 'c>,
    ctx: &mut CompilerContext<'c, 'c>,
) -> Result<Option<TileRustValue<'c, 'c>>, JITError> {
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
pub fn get_token_from_expr<'c>(
    var_arg: &Expr,
    ctx: &mut CompilerContext<'c, 'c>,
) -> Result<TileRustValue<'c, 'c>, JITError> {
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
pub fn get_token<'c>(
    var_name: &str,
    ctx: &mut CompilerContext<'c, 'c>,
) -> Result<TileRustValue<'c, 'c>, JITError> {
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
pub fn update_outer_block_type_meta<'c>(
    inner_block_vars: &mut CompilerContext<'c, 'c>,
    outer_block_vars: &mut CompilerContext<'c, 'c>,
    field_name: String,
) {
    // This does not work for function inlining.
    let mut var_map = HashMap::new();
    for var_name in outer_block_vars.var_keys() {
        var_map.insert(var_name.clone(), var_name.clone());
    }
    update_type_meta(inner_block_vars, outer_block_vars, &var_map, field_name);
}

/// Copies mutable type metadata fields from inner to outer context using a variable name mapping.
pub fn update_type_meta<'c>(
    inner_block_vars: &mut CompilerContext<'c, 'c>,
    outer_block_vars: &mut CompilerContext<'c, 'c>,
    outer2inner_vars: &HashMap<String, String>,
    _field_name: String,
) {
    let outer_keys_ = outer_block_vars.var_keys();
    let outer_keys = outer_keys_
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>();
    for outer_key in &outer_keys {
        let Some(outer_val) = outer_block_vars.vars.get(outer_key) else {
            // This should never fail.
            continue;
        };
        if outer_val.mutability == Mutability::Mutable {
            if let Some(inner_key) = outer2inner_vars.get(outer_key) {
                if let Some(inner_val) = inner_block_vars.vars.get(inner_key) {
                    // println!("Update vars from {inner_key} to {outer_key}");
                    if inner_val.mutability == Mutability::Mutable {
                        // Inner var must be mutable, too. This is what dictates whether
                        // the binding is written to.
                        let mut new_val = outer_val.clone();
                        new_val.type_meta = inner_val.type_meta.clone();
                        outer_block_vars.vars.insert(outer_key.clone(), new_val);
                    }
                    // Being more restrictive seemed like a good idea, but it's (currently) not.
                    // May need to be revisited.
                    // let mut new_val = outer_val.clone();
                    // let (Some(inner_meta), Some(new_meta)) = (&inner_val.type_meta, &mut new_val.type_meta) else {
                    //     panic!("Expected some TypeMeta for inner and outer context variable {:?}", inner_key);
                    // };
                    // let Some(field_value) = inner_meta.fields.get(field_name.as_str()) else {
                    //     panic!("Expected TypeMeta {field_name} for inner context variable {:?}", inner_key);
                    // };
                    // new_meta.fields.insert(field_name.clone(), field_value.clone());
                    // outer_block_vars.insert(outer_key.clone(), new_val);
                }
            }
        }
    }
}

/// Parses a comma-separated token stream into a list of `syn::Expr`.
pub fn parse_list_of_expr(tokens: TokenStream) -> Result<Vec<Expr>, JITError> {
    let mut args: Vec<Expr> = vec![];
    let mut arg_expr: Vec<TokenTree> = vec![];
    for token in tokens {
        // println!("{i} = {}", token.to_string());
        match &token {
            TokenTree::Literal(_) | TokenTree::Ident(_) => {
                arg_expr.push(token);
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
                    arg_expr.push(token);
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

/// Collects the names of variables assigned (mutated) in a block that were defined outside it.
pub fn collect_mutated_variables_from_block(
    block: &syn::Block,
) -> Result<BTreeSet<String>, JITError> {
    // In Rust, we can only mutate variables marked as such,
    // and assigning to such variables can only be done via an assignment expression,
    // e.g. x = something vs (let x = something).
    // "x = ..." is parsed as Stmt::Expr(Expr::Assign(...
    // "let x = ..." bindings are parsed as Stmt::Local(....
    // We need to check that the variable is actually being captured, or whether it was locally defined.
    let mut local_vars: HashSet<String> = HashSet::new();
    let mut result: BTreeSet<String> = BTreeSet::new();
    for statement in &block.stmts {
        match statement {
            Stmt::Local(local) => {
                let mut var_names: Vec<String> = vec![];
                // These are the local patterns we currently support in compile_block.
                // Make sure any changes there are reflected here.
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
                    // This var has not been defined in the current scope.
                    result.insert(var_name);
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

/// Walks all operations in a module's function block and verifies each one.
pub unsafe fn verify_statements_raw(cuda_tile_module: MlirOperation) -> Result<(), JITError> {
    let region = mlirOperationGetRegion(cuda_tile_module, 0);
    let block = mlirRegionGetFirstBlock(region);
    let cuda_tile_function = mlirBlockGetFirstOperation(block);
    let region = mlirOperationGetRegion(cuda_tile_function, 0);
    let block = mlirRegionGetFirstBlock(region);
    let mut stmt = mlirBlockGetFirstOperation(block);
    let mut i = -1;
    while !stmt.ptr.is_null() {
        i += 1;
        println!("verify {i}");
        mlirOperationDump(stmt);
        if !mlirOperationVerify(stmt) {
            return SourceLocation::unknown().jit_error_result(&format!(
                "MLIR operation verification failed at statement {i}"
            ));
        }
        stmt = mlirOperationGetNextInBlock(stmt);
    }
    Ok(())
}

fn get_int_hint(expr: &Expr) -> Result<i32, JITError> {
    let Expr::Lit(lit) = expr else {
        return SourceLocation::unknown()
            .jit_error_result("expected a literal value for optimization hint");
    };
    let Lit::Int(int_expr) = &lit.lit else {
        return SourceLocation::unknown()
            .jit_error_result("expected an integer literal for optimization hint");
    };
    int_expr
        .base10_parse()
        .map_err(|e| JITError::Generic(format!("Failed to parse int hint: {e}")))
}

/// Per-architecture (SM) optimization hints for kernel compilation.
pub struct SMHints {
    pub gpu_name: String,
    pub allow_tma: Option<bool>,
    pub num_cta_in_cga: Option<i32>,
    pub latency: Option<i32>,
    pub occupancy: Option<i32>,
    pub set_tensor_dim_factor: Option<i32>,
}

impl SMHints {
    /// Creates a new `SMHints` with no hints set for the given GPU architecture.
    pub fn new(gpu_name: String) -> Self {
        Self {
            gpu_name,
            num_cta_in_cga: None,
            allow_tma: None,
            latency: None,
            occupancy: None,
            set_tensor_dim_factor: None,
        }
    }

    /// Sets the TMA (Tensor Memory Access) permission hint.
    pub fn set_allow_tma(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.allow_tma.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("allow_tma hint has already been set");
        }
        let Expr::Lit(lit) = hint else {
            return SourceLocation::unknown()
                .jit_error_result("expected a literal value for allow_tma hint");
        };
        let Lit::Bool(bool_expr) = &lit.lit else {
            return SourceLocation::unknown()
                .jit_error_result("expected a boolean literal for allow_tma hint");
        };
        self.allow_tma = Some(bool_expr.value);
        Ok(())
    }

    /// Sets the number of CTAs in a CGA (Cooperative Grid Array) hint.
    pub fn set_num_cta_in_cga(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.num_cta_in_cga.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("num_cta_in_cga hint has already been set");
        }
        self.num_cta_in_cga = Some(get_int_hint(hint)?);
        Ok(())
    }

    /// Sets the target occupancy hint.
    pub fn set_occupancy(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.occupancy.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("occupancy hint has already been set");
        }
        self.occupancy = Some(get_int_hint(hint)?);
        Ok(())
    }

    /// Sets the target latency hint.
    pub fn set_latency(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.latency.is_some() {
            return SourceLocation::unknown().jit_error_result("latency hint has already been set");
        }
        self.latency = Some(get_int_hint(hint)?);
        Ok(())
    }
}

/// Collection of optimization hints for kernel compilation, keyed by SM architecture.
pub struct OptimizationHints {
    pub target_gpu_name: Option<String>,
    pub tile_as_hints: BTreeMap<String, SMHints>,
    pub tensor_dim_factor: Option<i32>,
}

impl OptimizationHints {
    /// Creates an empty set of optimization hints.
    pub fn empty() -> OptimizationHints {
        Self {
            target_gpu_name: None,
            tile_as_hints: BTreeMap::new(),
            tensor_dim_factor: None,
        }
    }

    /// Sets the tensor dimension factor hint.
    pub fn set_tensor_dim_factor(&mut self, hint: &Expr) -> Result<(), JITError> {
        if self.tensor_dim_factor.is_some() {
            return SourceLocation::unknown()
                .jit_error_result("tensor_dim_factor hint has already been set");
        }
        self.tensor_dim_factor = Some(get_int_hint(hint)?);
        Ok(())
    }

    fn parse_key_value(expr: &Expr) -> Result<(String, Expr), JITError> {
        let Expr::Assign(key_val) = expr else {
            return SourceLocation::unknown()
                .jit_error_result("expected an assignment expression in optimization hints");
        };
        let Expr::Path(key_path) = &*key_val.left else {
            return SourceLocation::unknown().jit_error_result(
                "Expected path expression on LHS of optimization hints assignment.",
            );
        };
        if key_path.path.segments.len() != 1 {
            return SourceLocation::unknown().jit_error_result(&format!(
                "Expected single-segment path in optimization hints key, got {} segments.",
                key_path.path.segments.len()
            ));
        }
        let key = key_path.path.segments.last().unwrap().ident.to_string();
        let value = *key_val.right.clone();
        Ok((key, value))
    }

    /// Parses optimization hints from a tuple expression in the entry annotation.
    pub fn parse(expr: &Expr, target_gpu_name: String) -> Result<OptimizationHints, JITError> {
        let Expr::Tuple(opt_hints) = expr else {
            return SourceLocation::unknown()
                .jit_error_result("expected a tuple expression for optimization hints");
        };
        let mut result = OptimizationHints::empty();
        result.target_gpu_name = Some(target_gpu_name);
        for sm_key_val in &opt_hints.elems {
            // println!("key: {gpu_name}, values: {hints_tuple:?}");

            let (opt_key, opt_value) = Self::parse_key_value(sm_key_val)?;
            match opt_key.as_str() {
                // Architecture agnostic optimization hints.
                "tensor_dim_factor" => {
                    result.set_tensor_dim_factor(&opt_value)?;
                }
                _ => {
                    if !opt_key.starts_with("sm_") {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "Unexpected optimization hint {}.",
                            sm_key_val.to_token_stream()
                        ));
                    }
                    // This is an architecture specific hint.
                    let Expr::Tuple(hints_tuple) = opt_value else {
                        return SourceLocation::unknown()
                            .jit_error_result("expected a tuple expression for architecture-specific optimization hints");
                    };
                    let mut sm_hints_result = SMHints::new(opt_key.clone());
                    for hint_key_val in hints_tuple.elems.iter() {
                        let (key, hints) = Self::parse_key_value(hint_key_val)?;
                        match key.as_str() {
                            "num_cta_in_cga" => {
                                sm_hints_result.set_num_cta_in_cga(&hints)?;
                            }
                            "occupancy" => {
                                sm_hints_result.set_occupancy(&hints)?;
                            }
                            "allow_tma" => {
                                sm_hints_result.set_allow_tma(&hints)?;
                            }
                            "latency" => {
                                sm_hints_result.set_latency(&hints)?;
                            }
                            _ => {
                                return SourceLocation::unknown().jit_error_result(&format!(
                                    "Unexpected optimization hint key '{key}'."
                                ));
                            }
                        }
                    }
                    if result
                        .tile_as_hints
                        .insert(opt_key.clone(), sm_hints_result)
                        .is_some()
                    {
                        return SourceLocation::unknown().jit_error_result(&format!(
                            "Duplicate optimization hint key '{opt_key}'."
                        ));
                    }
                }
            }
        }
        Ok(result)
    }

    /// Returns the SM-specific hints for the given architecture key.
    pub fn get_sm_hints(&self, key: &str) -> Option<&SMHints> {
        self.tile_as_hints.get(key)
    }

    /// Builds the MLIR `optimization_hints` attribute for the entry function.
    pub fn get_entry_opt_hints<'c>(
        &self,
        context: &'c Context,
    ) -> Result<Option<(Identifier<'c>, Attribute<'c>)>, JITError> {
        let mut results = vec![];
        for (arch, arch_hints) in self.tile_as_hints.iter() {
            let mut arch_hints_vec = vec![];
            if let Some(num_cta_in_cga) = arch_hints.num_cta_in_cga {
                arch_hints_vec.push(format!("num_cta_in_cga={num_cta_in_cga}"));
            }
            if let Some(occupancy) = arch_hints.occupancy {
                arch_hints_vec.push(format!("occupancy={occupancy}"));
            }
            if !arch_hints_vec.is_empty() {
                results.push(format!("{arch}={{ {} }}", arch_hints_vec.join(", ")));
            }
        }
        if !results.is_empty() {
            let opt_hint_str = format!("#cuda_tile.optimization_hints<{}>", results.join(", "));
            Ok(Some(parse_named_attr(
                context,
                "optimization_hints",
                &opt_hint_str,
            )?))
        } else {
            Ok(None)
        }
    }

    /// Builds the MLIR `optimization_hints` attribute for load/store operations.
    pub fn get_load_store_hints<'c>(
        &self,
        context: &'c Context,
        hint_params: HashMap<String, i32>,
    ) -> Result<Option<(Identifier<'c>, Attribute<'c>)>, JITError> {
        let mut results = vec![];
        if !hint_params.is_empty() {
            let target_arch = self
                .target_gpu_name
                .clone()
                .expect("Target gpu not yet specified. Did you compile?");
            let mut arch_hints_vec = vec![];
            for (key, val) in hint_params.iter() {
                arch_hints_vec.push(format!("{key}={val}"));
            }
            results.push(format!("{target_arch}={{ {} }}", arch_hints_vec.join(", ")));
        }
        for (arch, arch_hints) in self.tile_as_hints.iter() {
            let mut arch_hints_vec = vec![];
            if let Some(allow_tma) = arch_hints.allow_tma {
                if !hint_params.contains_key("allow_tma") {
                    // If hint params were provided, ignore corresponding optimization hints.
                    arch_hints_vec.push(format!("allow_tma={allow_tma}"));
                }
            }
            if let Some(latency) = arch_hints.latency {
                if !hint_params.contains_key("latency") {
                    arch_hints_vec.push(format!("latency={latency}"));
                }
            }
            if !arch_hints_vec.is_empty() {
                results.push(format!("{arch}={{ {} }}", arch_hints_vec.join(", ")));
            }
        }
        if !results.is_empty() {
            let opt_hint_str = format!("#cuda_tile.optimization_hints<{}>", results.join(", "));
            Ok(Some(parse_named_attr(
                context,
                "optimization_hints",
                &opt_hint_str,
            )?))
        } else {
            Ok(None)
        }
    }
}

/// Builds a `cuda_tile.reduce` MLIR operation.
#[allow(clippy::too_many_arguments)]
pub fn reduce_op<'c>(
    context: &'c Context,
    location: Location<'c>,
    operand: Value<'c, 'c>,
    dim: i32,
    identity: &str,
    element_type: String,
    result_type: Type<'c>,
    region: Region<'c>,
) -> Result<Operation<'c>, JITError> {
    Ok(OperationBuilder::new("cuda_tile.reduce", location)
        .add_attributes(&[
            parse_named_attr(context, "dim", format!("{dim}: i32").as_str())?,
            parse_named_attr(
                context,
                "identities",
                format!("[{identity} : {element_type}]").as_str(),
            )?,
        ])
        .add_operands(&[operand])
        .add_results(&[result_type])
        .add_regions([region])
        .build()
        .expect("Failed to build reduce op."))
}

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

// TODO (hme): These need to be tested.
/// Returns the hex-encoded MLIR constant string for a typed constant name (e.g. `"zero"`, `"one"`).
pub fn get_const_hex(rust_element_type_str: &str, const_str: &str) -> Result<String, JITError> {
    match rust_element_type_str {
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

/// Extracts a string literal value from an expression, handling both direct literals
/// and variables that were bound from string literals (when called from wrapper functions).
///
/// This is used by atomic operations like `atomic_rmw_tko` which require string literal
/// parameters (e.g., memory_ordering, memory_scope). When called through wrapper functions
/// like `atomic_and_tko`, these parameters are passed as variables that need to be resolved
/// back to their original string literal values.
///
/// ## Parameters
///
/// - `expr`: The expression to extract from (either a string literal or a variable)
/// - `param_name`: Name of the parameter (used in error messages)
/// - `ctx`: Variable scope to resolve variable references
///
/// ## Returns
///
/// The extracted string value
///
/// ## Panics
///
/// Panics if the expression is not a string literal and cannot be resolved to one.
pub fn extract_string_literal<'c, 'a>(
    expr: &syn::Expr,
    param_name: &str,
    ctx: &CompilerContext<'c, 'a>,
) -> Result<String, JITError> {
    use syn::{Expr, ExprLit, Lit};

    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(s), ..
        }) => Ok(s.value()),
        Expr::Path(path_expr) => {
            // This is a variable - check if it's a parameter that was bound from a string literal
            // When atomic_rmw_tko is called from a wrapper function like atomic_and_tko,
            // the arguments are variables (e.g., memory_ordering) that were bound from
            // string literals passed to the wrapper. We need to resolve these.
            let var_name = path_expr.path.segments.last().unwrap().ident.to_string();

            // Look up the variable in the variables map
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

/// Helper to resolve compile-time optional argument
/// Returns the inner expression if it is Some(expr), or None if it is None
pub fn resolve_option_arg<'c, 'a>(
    expr: &syn::Expr,
    ctx: &CompilerContext<'c, 'a>,
) -> Option<syn::Expr> {
    use syn::Expr;
    if let Expr::Call(call) = expr {
        if let Expr::Path(path) = &*call.func {
            if path.path.segments.last().unwrap().ident == "Some" {
                return Some(call.args[0].clone());
            }
        }
    } else if let Expr::Path(path) = expr {
        // Variable - check if it resolves to a compile-time constant Option
        if path.path.segments.len() == 1 && path.path.segments.last().unwrap().ident == "None" {
            return None;
        }
        let var_name = path.path.segments.last().unwrap().ident.to_string();
        if let Some(val) = ctx.vars.get(&var_name) {
            if let Some(ast) = &val.string_literal {
                // Recursively resolve the stored AST
                if let Expr::Call(call) = ast {
                    if let Expr::Path(path) = &*call.func {
                        if path.path.segments.last().unwrap().ident == "Some" {
                            return Some(call.args[0].clone());
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

/// Removes duplicate elements from a vector while preserving order.
pub fn dedup<T: Hash + Eq + Clone>(v: &mut Vec<T>) {
    let mut set = HashSet::new();
    v.retain(|x| set.insert(x.clone()));
}
