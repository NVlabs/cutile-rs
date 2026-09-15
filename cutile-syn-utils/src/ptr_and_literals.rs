/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Syn helpers shared by the compiler and macro expansion.

use syn::{Expr, Lit, UnOp};

/// Parses a possibly-negated integer literal expression as an `i32`.
pub fn parse_signed_literal_as_i32(expr: &Expr) -> i32 {
    match expr {
        Expr::Lit(lit) => {
            let val = match &lit.lit {
                Lit::Int(int_lit) => int_lit.base10_parse().unwrap(),
                _ => unimplemented!("Unexpected array element {expr:#?}"),
            };
            val
        }
        Expr::Unary(unary_expr) => match unary_expr.op {
            UnOp::Neg(_) => match &*unary_expr.expr {
                Expr::Lit(lit_expr) => {
                    let val: i32 = match &lit_expr.lit {
                        Lit::Int(int_lit) => int_lit.base10_parse().unwrap(),
                        _ => unimplemented!("Unexpected array element {expr:#?}"),
                    };
                    -val
                }
                _ => panic!("Unexpected unary expr {unary_expr:#?}"),
            },
            _ => panic!("Unexpected unary expr {unary_expr:#?}"),
        },
        _ => unimplemented!("Unexpected literal expression {expr:#?}"),
    }
}

/// Parses a pointer type string, returning `(is_mutable, pointee_type)`.
pub fn get_ptr_type(rust_ptr: &str) -> Option<(bool, String)> {
    // This also serves to check whether this is actually a pointer.
    let res = if rust_ptr.starts_with("* mut ") {
        (
            true,
            rust_ptr.split("* mut ").collect::<Vec<_>>()[1]
                .trim()
                .to_string(),
        )
    } else if rust_ptr.starts_with("* const ") {
        (
            false,
            rust_ptr.split("* const ").collect::<Vec<_>>()[1]
                .trim()
                .to_string(),
        )
    } else {
        return None;
    };
    Some(res)
}
