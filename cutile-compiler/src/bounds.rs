/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer interval arithmetic for static bounds tracking.
//! Used by the compiler to propagate and check value ranges at compile time.

use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::ast::SourceLocation;
use crate::error::{JITError, SpannedJITError};
use syn::BinOp;

// ---------------------------------------------------------------------------
// TileBinaryOp — lives here so both old and new compiler can share it
// ---------------------------------------------------------------------------

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
        BinOp::And(_) => Ok(TileBinaryOp::BitAnd),
        BinOp::Or(_) => Ok(TileBinaryOp::BitOr),
        _ => SourceLocation::unknown().jit_error_result("this binary operator is not supported"),
    }
}

// TODO (hme): Look into bounds for types other than i64.

fn div_ceil_i64(lhs: i64, rhs: i64) -> i64 {
    let quotient = lhs / rhs;
    let remainder = lhs % rhs;
    if remainder == 0 {
        quotient
    } else if (lhs > 0) == (rhs > 0) {
        quotient + 1
    } else {
        quotient
    }
}

/// An inclusive interval `[start, end]` over a copyable type.
#[derive(Debug, Copy, Clone)]
pub struct Bounds<T: Copy + PartialEq> {
    pub start: T, // Inclusive.
    pub end: T,   // Inclusive.
}

impl<T: Copy + PartialEq> Bounds<T> {
    /// Creates a new bounds interval from `start` to `end` (both inclusive).
    pub fn new(start: T, end: T) -> Bounds<T> {
        Self { start, end }
    }
    /// Creates an exact (single-value) bounds where `start == end`.
    pub fn exact(value: T) -> Bounds<T> {
        Self {
            start: value,
            end: value,
        }
    }
    /// Returns `true` if this interval represents a single known value.
    pub fn is_exact(&self) -> bool {
        self.end == self.start
    }
}

impl Add for Bounds<i64> {
    type Output = Bounds<i64>;
    fn add(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        let possible_bounds = vec![
            a.start + b.start,
            a.start + b.end,
            a.end + b.start,
            a.end + b.end,
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

impl Sub for Bounds<i64> {
    type Output = Bounds<i64>;
    fn sub(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        let possible_bounds = vec![
            a.start - b.start,
            a.start - b.end,
            a.end - b.start,
            a.end - b.end,
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

impl Mul for Bounds<i64> {
    type Output = Bounds<i64>;
    fn mul(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        let possible_bounds = vec![
            a.start * b.start,
            a.start * b.end,
            a.end * b.start,
            a.end * b.end,
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

impl Div for Bounds<i64> {
    type Output = Bounds<i64>;
    fn div(self, rhs: Bounds<i64>) -> Bounds<i64> {
        // For signed integer division:
        // - The minimum is when the numerator is smallest and divisor is largest and non-zero.
        // - The maximum is when the numerator is largest and divisor is smallest and non-zero.
        // If all values are non-zero and positive, the solution is the following
        // min = div(a.start, b.end)
        // max = div(a.start, b.start)
        // Since we permit signed values, it's easier to just take the min/max of all possible bounds.
        let a = self;
        let b = rhs;
        match (b.start, b.end) {
            (0, 0) => panic!("Division by zero"),
            (_, 0) => panic!("Division by zero"),
            (0, _) => panic!("Division by zero"),
            _ => {
                let possible_bounds = vec![
                    a.start / b.start,
                    a.start / b.end,
                    a.end / b.start,
                    a.end / b.end,
                ];
                let start = *possible_bounds
                    .iter()
                    .min()
                    .expect("Unexpected failed min op.");
                let end = *possible_bounds
                    .iter()
                    .max()
                    .expect("Unexpected failed max op.");
                Bounds::new(start, end)
            }
        }
    }
}

impl Rem for Bounds<i64> {
    type Output = Bounds<i64>;
    fn rem(self, rhs: Bounds<i64>) -> Bounds<i64> {
        let a = self;
        let b = rhs;
        // TODO (hme): Verify this one.
        let possible_bounds = vec![
            a.start % b.start,
            a.start % b.end,
            a.end % b.start,
            a.end % b.end,
        ];
        let start = *possible_bounds
            .iter()
            .min()
            .expect("Unexpected failed min op.");
        let end = *possible_bounds
            .iter()
            .max()
            .expect("Unexpected failed max op.");
        Bounds::new(start, end)
    }
}

/// Computes the output bounds of a binary operation `f` applied to two intervals.
pub fn bop_bounds<F: Fn(i64, i64) -> i64>(a: &Bounds<i64>, b: &Bounds<i64>, f: F) -> Bounds<i64> {
    // Compute bounds for various binary operations.
    // In general, the new bounds (for valid inputs) are:
    // start = min(op(a.start, b.start), op(a.start, b.end), op(a.end, b.start), op(a.end, b.end))
    // end = max(op(a.start, b.start), op(a.start, b.end), op(a.end, b.start), op(a.end, b.end))
    if a.is_exact() && b.is_exact() {
        return Bounds::exact(f(a.start, b.start));
    }
    let possible_bounds = vec![
        f(a.start, b.start),
        f(a.start, b.end),
        f(a.end, b.start),
        f(a.end, b.end),
    ];
    let start = *possible_bounds
        .iter()
        .min()
        .expect("Unexpected failed min op.");
    let end = *possible_bounds
        .iter()
        .max()
        .expect("Unexpected failed max op.");
    Bounds::new(start, end)
}

/// Returns the result bounds for a [`TileBinaryOp`], or `None` on division by zero.
pub fn bounds_from_bop(op: &TileBinaryOp, a: &Bounds<i64>, b: &Bounds<i64>) -> Option<Bounds<i64>> {
    match op {
        TileBinaryOp::CeilDiv | TileBinaryOp::Div | TileBinaryOp::TrueDiv => {
            // Deal with division separately to handle division by zero.
            match (b.start, b.end) {
                (0, 0) => None,
                (_, 0) => None,
                (0, _) => None,
                _ => Some(match op {
                    TileBinaryOp::Div | TileBinaryOp::TrueDiv => *a / *b,
                    TileBinaryOp::CeilDiv => bop_bounds(a, b, div_ceil_i64),
                    _ => unreachable!(),
                }),
            }
        }
        _ => Some(match op {
            TileBinaryOp::Add => *a + *b,
            TileBinaryOp::Sub => *a - *b,
            TileBinaryOp::Mul => *a * *b,
            TileBinaryOp::Rem => *a % *b,
            TileBinaryOp::Eq => bop_bounds(a, b, |a, b| (a == b) as i64),
            TileBinaryOp::Ne => bop_bounds(a, b, |a, b| (a != b) as i64),
            TileBinaryOp::Lt => bop_bounds(a, b, |a, b| (a < b) as i64),
            TileBinaryOp::Le => bop_bounds(a, b, |a, b| (a <= b) as i64),
            TileBinaryOp::Gt => bop_bounds(a, b, |a, b| (a > b) as i64),
            TileBinaryOp::Ge => bop_bounds(a, b, |a, b| (a >= b) as i64),
            TileBinaryOp::Min => bop_bounds(a, b, |a, b| a.min(b)),
            TileBinaryOp::Max => bop_bounds(a, b, |a, b| a.max(b)),
            TileBinaryOp::BitAnd => bop_bounds(a, b, |a, b| a & b),
            TileBinaryOp::BitOr => bop_bounds(a, b, |a, b| a | b),
            TileBinaryOp::BitXor => bop_bounds(a, b, |a, b| a ^ b),
            _ => unreachable!(),
        }),
    }
}
