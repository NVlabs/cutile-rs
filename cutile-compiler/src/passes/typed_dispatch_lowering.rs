/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed dispatch lowering helpers.
//!
//! This is the compiler1-compatible bridge for the pass-manager plan: when a
//! free function is only a trait-dispatch wrapper, lower the call site to the
//! method call before the current inline compiler sees it. Trait impl selection
//! still happens from the typed call arguments in method lookup.

use crate::passes::node_ids;
use crate::passes::type_inference::TypeckResults;
use crate::syn_utils::{get_ident_from_expr, get_sig_param_names};
use std::collections::HashMap;
use syn::punctuated::Punctuated;
use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprCall, ExprMethodCall, ItemFn, Stmt, Token};

/// If `fn_item` is a simple wrapper like `fn f(x, y) { x.method(y) }`, rewrite
/// `f(a, b)` to `a.method(b)`.
pub fn lower_dispatch_wrapper_call(
    fn_item: &ItemFn,
    call_expr: &ExprCall,
) -> Option<ExprMethodCall> {
    let [Stmt::Expr(Expr::MethodCall(method_call), _)] = fn_item.block.stmts.as_slice() else {
        return None;
    };

    let param_names = get_sig_param_names(&fn_item.sig);
    if param_names.len() != call_expr.args.len() {
        return None;
    }

    let arg_by_param = param_names
        .iter()
        .cloned()
        .zip(call_expr.args.iter().cloned())
        .collect::<HashMap<_, _>>();

    let receiver_ident = get_ident_from_expr(&method_call.receiver)?;
    let receiver = arg_by_param.get(&receiver_ident.to_string())?;

    let mut lowered_args = Vec::with_capacity(method_call.args.len());
    for arg in &method_call.args {
        let arg_ident = get_ident_from_expr(arg)?;
        lowered_args.push(arg_by_param.get(&arg_ident.to_string())?);
    }

    Some(ExprMethodCall {
        attrs: Vec::new(),
        receiver: Box::new(receiver.clone()),
        dot_token: Default::default(),
        method: method_call.method.clone(),
        turbofish: method_call.turbofish.clone(),
        paren_token: method_call.paren_token,
        args: lowered_args
            .into_iter()
            .cloned()
            .collect::<Punctuated<_, Token![,]>>(),
    })
}

pub fn lower_function(fn_item: &ItemFn, typeck_results: &TypeckResults) -> ItemFn {
    let mut lowered = fn_item.clone();
    let mut pass = TypedDispatchLowering { typeck_results };
    pass.visit_item_fn_mut(&mut lowered);
    lowered
}

struct TypedDispatchLowering<'a> {
    typeck_results: &'a TypeckResults,
}

impl VisitMut for TypedDispatchLowering<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);

        let Some(call_id) = node_ids::expr_id(expr) else {
            return;
        };
        let Expr::Call(call) = expr else {
            return;
        };
        let Some(method_call) = self.typeck_results.lowered_method_call(call).cloned() else {
            return;
        };
        let mut lowered = Expr::MethodCall(method_call);
        node_ids::set_expr_id(&mut lowered, call_id);
        *expr = lowered;
    }
}
