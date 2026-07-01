/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for in-memory cache key correctness.
//!
//! The `cache_key_*` tests are CPU-only (no GPU required) and assert that the
//! in-memory cache key (`get_hash_string`) distinguishes every input that can
//! change the generated GPU code, so kernels are never falsely reused.

use cutile::tile_kernel::{CompileOptions, FunctionKey, TileFunctionKey};
use cutile_compiler::specialization::{DivHint, SpecializationBits};

fn default_key() -> cutile::tile_kernel::TileFunctionKeyBuilder {
    TileFunctionKey::builder("m", "f")
        .source_hash("hash")
        .gpu_name("sm_90")
        .compiler_version("0.0.1")
        .cuda_toolkit_version("12.4")
}

// TileFunctionKey hash properties
#[test]
fn cache_key_hash_deterministic() {
    let key1 = TileFunctionKey::builder("mod", "fn")
        .generics(vec!["f32".into()])
        .source_hash("abc123")
        .gpu_name("sm_90")
        .compiler_version("0.0.1-alpha")
        .cuda_toolkit_version("12.4")
        .build();
    let key2 = key1.clone();
    assert_eq!(key1.get_hash_string(), key2.get_hash_string());
}

#[test]
fn cache_key_different_source_hash() {
    let key_a = default_key().source_hash("hash_v1").build();
    let key_b = default_key().source_hash("hash_v2").build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_gpu_name() {
    let key_a = default_key().gpu_name("sm_80").build();
    let key_b = default_key().gpu_name("sm_90").build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_compiler_version() {
    let key_a = default_key().compiler_version("0.0.1").build();
    let key_b = default_key().compiler_version("0.0.2").build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_cuda_toolkit_version() {
    let key_a = default_key().cuda_toolkit_version("12.4").build();
    let key_b = default_key().cuda_toolkit_version("12.6").build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_generics() {
    let key_a = default_key().generics(vec!["f32".into()]).build();
    let key_b = default_key().generics(vec!["f16".into()]).build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

// `get_cuda_toolkit_version()` returns `"unknown"` when nvcc is unavailable.
// Verify this sentinel is distinct from any real version string.
#[test]
fn cache_key_toolkit_unknown_is_distinct() {
    let key_unknown = default_key().cuda_toolkit_version("unknown").build();
    let key_real = default_key().cuda_toolkit_version("12.4").build();
    assert_ne!(key_unknown.get_hash_string(), key_real.get_hash_string());
}

// Cache keys must distinguish data alignments to prevent incorrect kernel reuse.
#[test]
fn cache_key_different_spec_args() {
    let spec_aligned = SpecializationBits {
        shape_div: vec![DivHint::from_value(16), DivHint::from_value(16)],
        stride_div: vec![DivHint::from_value(16), DivHint::from_value(16)],
        stride_one: vec![false, true],
        base_ptr_div: DivHint::from_ptr(16),
        elements_disjoint: true,
    };
    let spec_misaligned = SpecializationBits {
        shape_div: vec![DivHint::from_value(4), DivHint::from_value(4)],
        stride_div: vec![DivHint::from_value(4), DivHint::from_value(4)],
        stride_one: vec![false, true],
        base_ptr_div: DivHint::from_ptr(4),
        elements_disjoint: true,
    };
    let key_a = default_key()
        .spec_args(vec![("x".into(), spec_aligned)])
        .build();
    let key_b = default_key()
        .spec_args(vec![("x".into(), spec_misaligned)])
        .build();
    assert_ne!(
        key_a.get_hash_string(),
        key_b.get_hash_string(),
        "different SpecializationBits must produce distinct memory keys"
    );
}

#[test]
fn cache_key_different_compile_options() {
    let key_a = default_key()
        .compile_options(CompileOptions::default().max_divisibility(8))
        .build();
    let key_b = default_key()
        .compile_options(CompileOptions::default().max_divisibility(16))
        .build();
    assert_ne!(
        key_a.get_hash_string(),
        key_b.get_hash_string(),
        "different CompileOptions must produce distinct memory keys"
    );

    let key_c = default_key()
        .compile_options(CompileOptions::default().occupancy(2))
        .build();
    let key_d = default_key()
        .compile_options(CompileOptions::default().occupancy(4))
        .build();
    assert_ne!(key_c.get_hash_string(), key_d.get_hash_string());
}

