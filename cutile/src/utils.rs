/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Utility functions and traits for cuTile Rust.
//!
//! This module provides helper utilities for testing, debugging, and working with
//! floating-point values and tensors.
//!
//! ## Floating-Point Comparisons
//!
//! The [`Float`] trait provides approximate equality comparisons for floating-point
//! types (`f16`, `f32`, `f64`). This is essential for testing GPU computations where
//! exact floating-point equality is often inappropriate due to rounding errors.
//!
//! ## Pretty Printing
//!
//! The [`pretty_print_matrix`] function formats 2D tensors (matrices) in a readable
//! table format for debugging and inspection.

use candle_core::WithDType;
use half::f16;
use num_traits::float::FloatCore;

/// Trait for approximate floating-point comparisons.
///
/// Provides methods to compare floating-point values with tolerance for rounding errors.
/// This is crucial for testing GPU computations where exact equality is often inappropriate.
///
/// Implemented for `f16`, `f32`, and `f64`.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::utils::Float;
///
/// let a = 0.1f32 + 0.2f32;
/// let b = 0.3f32;
///
/// // Exact equality might fail due to rounding
/// assert!(a.close(b, 1e-6));
///
/// // Or use machine epsilon
/// assert!(a.epsilon_close(b));
/// ```
pub trait Float {
    /// Compares two values with a specified tolerance.
    ///
    /// Returns `true` if the absolute difference between the values is less than `tolerance`.
    fn close(&self, other: Self, tolerance: Self) -> bool;

    /// Compares two values using machine epsilon as the tolerance.
    ///
    /// Returns `true` if the values are equal within the type's epsilon (smallest
    /// representable difference).
    fn epsilon_close(&self, other: Self) -> bool;
}
impl Float for f16 {
    fn close(&self, other: Self, tolerance: f16) -> bool {
        (self - other).abs() < tolerance
    }
    fn epsilon_close(&self, other: Self) -> bool {
        self.close(other, f16::EPSILON)
    }
}
impl Float for f32 {
    fn close(&self, other: Self, tolerance: f32) -> bool {
        (self - other).abs() < tolerance
    }
    fn epsilon_close(&self, other: Self) -> bool {
        self.close(other, f32::EPSILON)
    }
}
impl Float for f64 {
    fn close(&self, other: Self, tolerance: f64) -> bool {
        (self - other).abs() < tolerance
    }
    fn epsilon_close(&self, other: Self) -> bool {
        self.close(other, f64::EPSILON)
    }
}

/// Prints a 2D tensor (matrix) in a formatted table.
///
/// Formats and prints a matrix with column indices as headers and values
/// formatted to one decimal place. Useful for debugging and visualizing
/// small matrices.
///
/// ## Parameters
///
/// - `mat`: A 2D `candle_core::Tensor` to print
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
/// use cutile::utils::pretty_print_matrix;
///
/// let matrix = api::arange::<f32>(9).reshape([3, 3]).await;
/// let cpu_matrix = api::copy_to_host(&Arc::new(matrix)).await;
/// pretty_print_matrix::<f32>(&cpu_matrix);
///
/// // Output:
/// //    0        1        2
/// // | 0.0    | 1.0    | 2.0    |
/// // | 3.0    | 4.0    | 5.0    |
/// // | 6.0    | 7.0    | 8.0    |
/// ```
///
/// ## Note
///
/// This function is intended for small matrices and debugging purposes.
/// For large matrices, consider using other visualization tools.
pub fn pretty_print_matrix<T: WithDType>(mat: &candle_core::Tensor) {
    let iter_dim = 0;
    let range = 0..mat.shape().dims()[iter_dim];
    println!(
        " {}",
        range
            .into_iter()
            .map(|x| format!("{:^9}", x))
            .collect::<String>()
    );
    for j in 0..mat.shape().dims()[iter_dim] {
        let out_vec = mat.get_on_dim(iter_dim, j).unwrap().to_vec1::<T>().unwrap();
        let out_vec = out_vec
            .iter()
            .map(|x| format!("{:^8.1}|", x))
            .collect::<String>();
        println!("|{}", out_vec);
    }
}
