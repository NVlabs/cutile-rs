/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for cache key correctness, JitStore integration, and warmup APIs.
//!
//! - `cache_key_*` tests are CPU-only (no GPU required).
//! - `warmup_*` tests require GPU (compile + launch).

use cutile::tile_kernel::{CompileOptions, EntryMeta, FunctionKey, TileFunctionKey, WarmupSpec};
use cutile_compiler::specialization::SpecializationBits;


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
    assert_eq!(key1.get_disk_hash_string(), key2.get_disk_hash_string());
}

#[test]
fn cache_key_different_source_hash() {
    let key_a = default_key().source_hash("hash_v1").build();
    let key_b = default_key().source_hash("hash_v2").build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
    assert_ne!(key_a.get_disk_hash_string(), key_b.get_disk_hash_string());
}

#[test]
fn cache_key_different_gpu_name() {
    let key_a = default_key().gpu_name("sm_80").build();
    let key_b = default_key().gpu_name("sm_90").build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
    assert_ne!(key_a.get_disk_hash_string(), key_b.get_disk_hash_string());
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

#[test]
fn cache_key_disk_hash_is_sha256_length() {
    let key = default_key().build();
    let disk_hash = key.get_disk_hash_string();
    // SHA-256 hex output = 64 characters.
    assert_eq!(disk_hash.len(), 64, "disk hash should be 64 hex chars");
    assert!(
        disk_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "disk hash should be lowercase hex"
    );
}

// When `nvcc` is unavailable, `get_cuda_toolkit_version()` returns `"unknown"`.
// Verify that `"unknown"` still produces a distinct key from any real version,
// so kernels compiled without a known toolkit version are never falsely reused
// when a real version becomes available (or vice versa).
#[test]
fn cache_key_toolkit_unknown_is_distinct() {
    let key_unknown = default_key().cuda_toolkit_version("unknown").build();
    let key_real = default_key().cuda_toolkit_version("12.4").build();
    assert_ne!(
        key_unknown.get_hash_string(),
        key_real.get_hash_string(),
        "unknown toolkit must produce distinct memory key"
    );
    assert_ne!(
        key_unknown.get_disk_hash_string(),
        key_real.get_disk_hash_string(),
        "unknown toolkit must produce distinct disk key"
    );
}


#[test]
fn cache_key_source_hash_change_invalidates() {
    let make_key = |source_hash: &str| -> TileFunctionKey {
        TileFunctionKey::builder("linalg", "matmul")
            .generics(vec!["f32".into(), "128".into()])
            .stride_args(vec![("a".into(), vec![1, 128])])
            .grid((4, 4, 1))
            .source_hash(source_hash)
            .gpu_name("sm_90")
            .compiler_version("0.1.0")
            .cuda_toolkit_version("12.4")
            .build()
    };
    let key_v1 = make_key("aabbccdd11223344");
    let key_v2 = make_key("eeff0011deadbeef");
    assert_ne!(key_v1.get_hash_string(), key_v2.get_hash_string());
    assert_ne!(key_v1.get_disk_hash_string(), key_v2.get_disk_hash_string());
}

// Cache keys must distinguish  data alignments to prevent incorrect kernel reuse.
#[test]
fn cache_key_different_spec_args() {
    let spec_aligned = SpecializationBits {
        shape_div: vec![16, 16],
        stride_div: vec![16, 16],
        stride_one: vec![false, true],
        base_ptr_div: 16,
        elements_disjoint: true,
    };
    let spec_misaligned = SpecializationBits {
        shape_div: vec![4, 4],
        stride_div: vec![4, 4],
        stride_one: vec![false, true],
        base_ptr_div: 4,
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
    assert_ne!(
        key_a.get_disk_hash_string(),
        key_b.get_disk_hash_string(),
        "different SpecializationBits must produce distinct disk keys"
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
    assert_ne!(
        key_a.get_disk_hash_string(),
        key_b.get_disk_hash_string(),
        "different CompileOptions must produce distinct disk keys"
    );

    let key_c = default_key()
        .compile_options(CompileOptions::default().occupancy(2))
        .build();
    let key_d = default_key()
        .compile_options(CompileOptions::default().occupancy(4))
        .build();
    assert_ne!(key_c.get_hash_string(), key_d.get_hash_string());
}

// WarmupSpec builder

#[test]
fn warmup_spec_builder() {
    let spec = WarmupSpec::new("my_kernel", vec!["f32".into(), "128".into()])
        .with_strides(vec![("x".into(), vec![1, 128])])
        .with_const_grid((4, 1, 1));
    assert_eq!(spec.function_name, "my_kernel");
    assert_eq!(spec.function_generics, vec!["f32", "128"]);
    assert_eq!(spec.stride_args.len(), 1);
    assert_eq!(spec.const_grid, Some((4, 1, 1)));
}

//  EntryMeta

#[test]
fn entry_meta_fields() {
    let meta = EntryMeta {
        module_name: "linalg",
        function_name: "vector_add",
        function_entry: "vector_add_entry",
    };
    assert_eq!(meta.module_name, "linalg");
    assert_eq!(meta.function_name, "vector_add");
    assert_eq!(meta.function_entry, "vector_add_entry");
}
