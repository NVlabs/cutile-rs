/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared utilities and reference implementations for cutile examples.

/// Formats a byte count into a human-readable size string (e.g. "1.5kb", "2.3mb").
pub fn size_label(size_bytes: usize) -> String {
    if size_bytes < 10usize.pow(3) {
        // bytes
        format!("{}b", size_bytes)
    } else if size_bytes < 10usize.pow(6) {
        // kb
        format!("{:.1}kb", size_bytes as f64 / 10usize.pow(3) as f64)
    } else if size_bytes < 10usize.pow(9) {
        // mb
        format!("{:.1}mb", size_bytes as f64 / 10usize.pow(6) as f64)
    } else {
        // gb
        format!("{:.1}gb", size_bytes as f64 / 10usize.pow(9) as f64)
    }
}

#[cfg(feature = "cuda")]
mod cuda_utils;

#[cfg(feature = "cuda")]
pub use cuda_utils::*;
