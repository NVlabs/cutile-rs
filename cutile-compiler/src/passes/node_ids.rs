/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stable node ids for dispatch-bearing expressions in compiler-owned syn bodies.
//!
//! The current compiler still emits from `syn`, so type-check side tables need
//! a way to refer back to individual call and method-call nodes without relying
//! on token strings. These ids are stable only within one cloned function body
//! and pass pipeline. Compiler3 should broaden this to all expressions and
//! statements.

use syn::visit_mut::{self, VisitMut};
use syn::{Attribute, Expr, ExprLit, ItemFn, Lit, Meta};

const NODE_ID_ATTR: &str = "__cutile_node_id";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

pub fn assign_expr_ids(fn_item: &mut ItemFn) {
    let mut assigner = NodeIdAssigner { next: 0 };
    assigner.visit_item_fn_mut(fn_item);
}

pub fn expr_id(expr: &Expr) -> Option<NodeId> {
    expr_attrs(expr)?
        .iter()
        .find_map(node_id_from_attr)
}

pub fn set_expr_id(expr: &mut Expr, id: NodeId) {
    let Some(attrs) = expr_attrs_mut(expr) else {
        return;
    };
    attrs.retain(|attr| !attr.path().is_ident(NODE_ID_ATTR));
    let raw_id = syn::LitInt::new(&id.0.to_string(), proc_macro2::Span::call_site());
    attrs.push(syn::parse_quote!(#[__cutile_node_id = #raw_id]));
}

struct NodeIdAssigner {
    next: u32,
}

impl NodeIdAssigner {
    fn fresh(&mut self) -> NodeId {
        let id = NodeId(self.next);
        self.next += 1;
        id
    }
}

impl VisitMut for NodeIdAssigner {
    fn visit_attribute_mut(&mut self, _attr: &mut Attribute) {
        // Internal expression ids must not recurse into attribute meta syntax.
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if matches!(expr, Expr::Call(_) | Expr::MethodCall(_)) {
            let id = self.fresh();
            set_expr_id(expr, id);
        }
        visit_mut::visit_expr_mut(self, expr);
    }
}

fn node_id_from_attr(attr: &Attribute) -> Option<NodeId> {
    if !attr.path().is_ident(NODE_ID_ATTR) {
        return None;
    }
    let Meta::NameValue(name_value) = &attr.meta else {
        return None;
    };
    let Expr::Lit(ExprLit {
        lit: Lit::Int(raw_id),
        ..
    }) = &name_value.value
    else {
        return None;
    };
    Some(NodeId(raw_id.base10_parse().ok()?))
}

fn expr_attrs(expr: &Expr) -> Option<&Vec<Attribute>> {
    match expr {
        Expr::Array(expr) => Some(&expr.attrs),
        Expr::Assign(expr) => Some(&expr.attrs),
        Expr::Async(expr) => Some(&expr.attrs),
        Expr::Await(expr) => Some(&expr.attrs),
        Expr::Binary(expr) => Some(&expr.attrs),
        Expr::Block(expr) => Some(&expr.attrs),
        Expr::Break(expr) => Some(&expr.attrs),
        Expr::Call(expr) => Some(&expr.attrs),
        Expr::Cast(expr) => Some(&expr.attrs),
        Expr::Closure(expr) => Some(&expr.attrs),
        Expr::Const(expr) => Some(&expr.attrs),
        Expr::Continue(expr) => Some(&expr.attrs),
        Expr::Field(expr) => Some(&expr.attrs),
        Expr::ForLoop(expr) => Some(&expr.attrs),
        Expr::Group(expr) => Some(&expr.attrs),
        Expr::If(expr) => Some(&expr.attrs),
        Expr::Index(expr) => Some(&expr.attrs),
        Expr::Infer(expr) => Some(&expr.attrs),
        Expr::Let(expr) => Some(&expr.attrs),
        Expr::Lit(expr) => Some(&expr.attrs),
        Expr::Loop(expr) => Some(&expr.attrs),
        Expr::Macro(expr) => Some(&expr.attrs),
        Expr::Match(expr) => Some(&expr.attrs),
        Expr::MethodCall(expr) => Some(&expr.attrs),
        Expr::Paren(expr) => Some(&expr.attrs),
        Expr::Path(expr) => Some(&expr.attrs),
        Expr::Range(expr) => Some(&expr.attrs),
        Expr::RawAddr(expr) => Some(&expr.attrs),
        Expr::Reference(expr) => Some(&expr.attrs),
        Expr::Repeat(expr) => Some(&expr.attrs),
        Expr::Return(expr) => Some(&expr.attrs),
        Expr::Struct(expr) => Some(&expr.attrs),
        Expr::Try(expr) => Some(&expr.attrs),
        Expr::TryBlock(expr) => Some(&expr.attrs),
        Expr::Tuple(expr) => Some(&expr.attrs),
        Expr::Unary(expr) => Some(&expr.attrs),
        Expr::Unsafe(expr) => Some(&expr.attrs),
        Expr::While(expr) => Some(&expr.attrs),
        Expr::Yield(expr) => Some(&expr.attrs),
        Expr::Verbatim(_) => None,
        _ => None,
    }
}

fn expr_attrs_mut(expr: &mut Expr) -> Option<&mut Vec<Attribute>> {
    match expr {
        Expr::Array(expr) => Some(&mut expr.attrs),
        Expr::Assign(expr) => Some(&mut expr.attrs),
        Expr::Async(expr) => Some(&mut expr.attrs),
        Expr::Await(expr) => Some(&mut expr.attrs),
        Expr::Binary(expr) => Some(&mut expr.attrs),
        Expr::Block(expr) => Some(&mut expr.attrs),
        Expr::Break(expr) => Some(&mut expr.attrs),
        Expr::Call(expr) => Some(&mut expr.attrs),
        Expr::Cast(expr) => Some(&mut expr.attrs),
        Expr::Closure(expr) => Some(&mut expr.attrs),
        Expr::Const(expr) => Some(&mut expr.attrs),
        Expr::Continue(expr) => Some(&mut expr.attrs),
        Expr::Field(expr) => Some(&mut expr.attrs),
        Expr::ForLoop(expr) => Some(&mut expr.attrs),
        Expr::Group(expr) => Some(&mut expr.attrs),
        Expr::If(expr) => Some(&mut expr.attrs),
        Expr::Index(expr) => Some(&mut expr.attrs),
        Expr::Infer(expr) => Some(&mut expr.attrs),
        Expr::Let(expr) => Some(&mut expr.attrs),
        Expr::Lit(expr) => Some(&mut expr.attrs),
        Expr::Loop(expr) => Some(&mut expr.attrs),
        Expr::Macro(expr) => Some(&mut expr.attrs),
        Expr::Match(expr) => Some(&mut expr.attrs),
        Expr::MethodCall(expr) => Some(&mut expr.attrs),
        Expr::Paren(expr) => Some(&mut expr.attrs),
        Expr::Path(expr) => Some(&mut expr.attrs),
        Expr::Range(expr) => Some(&mut expr.attrs),
        Expr::RawAddr(expr) => Some(&mut expr.attrs),
        Expr::Reference(expr) => Some(&mut expr.attrs),
        Expr::Repeat(expr) => Some(&mut expr.attrs),
        Expr::Return(expr) => Some(&mut expr.attrs),
        Expr::Struct(expr) => Some(&mut expr.attrs),
        Expr::Try(expr) => Some(&mut expr.attrs),
        Expr::TryBlock(expr) => Some(&mut expr.attrs),
        Expr::Tuple(expr) => Some(&mut expr.attrs),
        Expr::Unary(expr) => Some(&mut expr.attrs),
        Expr::Unsafe(expr) => Some(&mut expr.attrs),
        Expr::While(expr) => Some(&mut expr.attrs),
        Expr::Yield(expr) => Some(&mut expr.attrs),
        Expr::Verbatim(_) => None,
        _ => None,
    }
}
