/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-tensor specialization metadata for JIT compilation.
//!
//! Captures alignment and divisibility properties of tensor shape, strides,
//! and base pointer at construction time. The compiler uses these to emit
//! targeted `assume_div_by` operations, capped by `max_divisibility` when set.

/// Per-tensor metadata inferred from runtime shape, strides, and base pointer.
///
/// Computed once at tensor construction and recomputed on reshape/view.
/// Used by the JIT compiler to emit targeted `assume_div_by` operations
/// and to determine static vs dynamic strides in generated MLIR.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct SpecializationBits {
    /// Per-dimension: max power-of-2 divisor of shape[i], clamped to 16.
    pub shape_div: Vec<i32>,
    /// Per-dimension: max power-of-2 divisor of stride[i] in bytes, clamped to 16.
    pub stride_div: Vec<i32>,
    /// Per-dimension: whether stride[i] == 1.
    pub stride_one: Vec<bool>,
    /// Max power-of-2 divisor of the base device pointer, clamped to 16.
    pub base_ptr_div: i32,
    /// True if elements are non-overlapping (strides are non-aliasing).
    pub elements_disjoint: bool,
}

/// Returns the largest power-of-2 that divides `val`, clamped to 16.
/// Zero is treated as divisible by 16 (maximum).
pub fn max_pow2_divisor(val: i32) -> i32 {
    if val == 0 {
        return 16;
    }
    (val & val.wrapping_neg()).min(16)
}

/// Computes specialization bits from tensor metadata.
pub fn compute_spec(
    base_ptr: u64,
    shape: &[i32],
    strides: &[i32],
    dtype_bytes: i32,
) -> SpecializationBits {
    let ndim = shape.len();
    let mut spec = SpecializationBits {
        shape_div: Vec::with_capacity(ndim),
        stride_div: Vec::with_capacity(ndim),
        stride_one: Vec::with_capacity(ndim),
        base_ptr_div: max_pow2_divisor(base_ptr as i32),
        elements_disjoint: true,
    };
    for i in 0..ndim {
        spec.shape_div.push(max_pow2_divisor(shape[i]));
        let stride_bytes = strides[i] * dtype_bytes;
        spec.stride_div.push(max_pow2_divisor(stride_bytes));
        spec.stride_one.push(strides[i] == 1);
    }
    // Disjointness: sort by stride, check stride[i+1] >= stride[i] * shape[i].
    let mut sorted: Vec<(i32, i32)> = strides
        .iter()
        .zip(shape.iter())
        .map(|(&s, &d)| (s, d))
        .collect();
    sorted.sort();
    spec.elements_disjoint = sorted.first().map_or(true, |(s, _)| *s > 0);
    for w in sorted.windows(2) {
        if w[1].0 <= 0 || w[1].0 < w[0].0 * w[0].1 {
            spec.elements_disjoint = false;
            break;
        }
    }
    spec
}

#[cfg(test)]
#[path = "specialization_tests.rs"]
mod tests;
