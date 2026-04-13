/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-tensor specialization metadata for JIT compilation.
//!
//! Captures alignment and divisibility properties of tensor shape, strides,
//! and base pointer at construction time. The compiler uses these to emit
//! targeted `assume_div_by` operations, capped by `max_divisibility` when set.

/// A divisibility annotation: the max power-of-2 divisor of a value,
/// clamped to a configurable maximum.
///
/// Used by the JIT compiler to emit targeted `assume_div_by` operations.
/// Works for tensor dimensions, strides, pointers, and scalar integers.
///
/// `DivHint` is unit-agnostic: the unit of `divisor` depends on context.
/// See [`SpecializationBits`] for how each field's unit is defined.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct DivHint {
    /// Max power-of-2 divisor of the value, clamped to `max`.
    pub divisor: i32,
    /// Upper bound on the divisor (default 16).
    pub max: i32,
}

impl Default for DivHint {
    fn default() -> Self {
        Self {
            divisor: 1,
            max: 16,
        }
    }
}

impl DivHint {
    /// Compute divisibility from an integer value, clamped to 16.
    ///
    /// The unit of the result matches the unit of `val`: elements for
    /// shape/stride values, bytes for pointer addresses cast to i32.
    pub fn from_value(val: i32) -> Self {
        let raw: i32 = max_pow2_divisor_unclamped(val);
        Self {
            divisor: raw.min(16),
            max: 16,
        }
    }

    /// Compute divisibility from a pointer address, clamped to 16.
    ///
    /// The result is in **bytes**: a divisor of 16 means the pointer is
    /// aligned to a 16-byte boundary. This matches cutile-python's
    /// `base_addr_divisible_by` convention.
    pub fn from_ptr(ptr: u64) -> Self {
        let raw: i32 = max_pow2_divisor_unclamped(ptr as i32);
        Self {
            divisor: raw.min(16),
            max: 16,
        }
    }

    /// Override the maximum clamp. Returns a new `DivHint` with
    /// `divisor` re-clamped to the given `max`.
    pub fn with_max(self, max: i32) -> Self {
        Self {
            divisor: self.divisor.min(max),
            max,
        }
    }
}

/// Per-tensor metadata inferred from runtime shape, strides, and base pointer.
///
/// Computed once at tensor construction and recomputed on reshape/view.
/// Used by the JIT compiler to emit targeted `assume_div_by` operations
/// and to determine static vs dynamic strides in generated MLIR.
///
/// # Units
///
/// `shape_div` and `stride_div` are in **elements** (not bytes). This
/// matches NVT and cutile-python, where divisibility is dtype-independent.
/// `base_ptr_div` is in **bytes** (raw pointer alignment). This matches
/// cutile-python's `base_addr_divisible_by` convention.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct SpecializationBits {
    /// Per-dimension: divisibility of shape[i], in **elements**.
    pub shape_div: Vec<DivHint>,
    /// Per-dimension: divisibility of stride[i], in **elements** (not bytes).
    /// A stride of 1 element always has divisor 1, regardless of dtype size.
    pub stride_div: Vec<DivHint>,
    /// Per-dimension: whether stride[i] == 1.
    pub stride_one: Vec<bool>,
    /// Divisibility of the base device pointer, in **bytes**.
    /// A pointer at 0x1000 has divisor 16 (aligned to 16-byte boundary).
    pub base_ptr_div: DivHint,
    /// True if elements are non-overlapping (strides are non-aliasing).
    pub elements_disjoint: bool,
}

/// Returns the largest power-of-2 that divides `val`.
/// Zero is treated as divisible by 16 (maximum).
/// NOT clamped — callers (typically `DivHint`) apply their own clamp.
fn max_pow2_divisor_unclamped(val: i32) -> i32 {
    if val == 0 {
        return 16;
    }
    val & val.wrapping_neg()
}

/// Returns the largest power-of-2 that divides `val`, clamped to 16.
/// Zero is treated as divisible by 16 (maximum).
pub fn max_pow2_divisor(val: i32) -> i32 {
    max_pow2_divisor_unclamped(val).min(16)
}

/// DSL integer scalar types that support divisibility hints.
pub const INTEGER_SCALAR_TYPES: &[&str] = &["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"];

/// DSL float scalar types.
pub const FLOAT_SCALAR_TYPES: &[&str] = &["f16", "bf16", "f32", "f64"];

/// DSL scalar types that aren't integer or float.
pub const OTHER_SCALAR_TYPES: &[&str] = &["bool"];

/// Returns true if the type name is a DSL integer scalar type.
pub fn is_integer_scalar(ty: &str) -> bool {
    INTEGER_SCALAR_TYPES.contains(&ty)
}

/// Returns true if the type name is any DSL scalar type.
pub fn is_scalar(ty: &str) -> bool {
    INTEGER_SCALAR_TYPES.contains(&ty)
        || FLOAT_SCALAR_TYPES.contains(&ty)
        || OTHER_SCALAR_TYPES.contains(&ty)
}

/// Computes specialization bits from tensor metadata.
pub fn compute_spec(
    base_ptr: u64,
    shape: &[i32],
    strides: &[i32],
    _dtype_bytes: i32,
) -> SpecializationBits {
    let ndim = shape.len();
    let mut spec = SpecializationBits {
        shape_div: Vec::with_capacity(ndim),
        stride_div: Vec::with_capacity(ndim),
        stride_one: Vec::with_capacity(ndim),
        base_ptr_div: DivHint::from_ptr(base_ptr),
        elements_disjoint: true,
    };
    for i in 0..ndim {
        spec.shape_div.push(DivHint::from_value(shape[i]));
        // Divisibility is in elements, not bytes (matches NVT and cutile-python).
        spec.stride_div.push(DivHint::from_value(strides[i]));
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
