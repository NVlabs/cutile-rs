/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for cache key correctness, JitStore integration, and warmup APIs.
//!
//! - `cache_key_*` tests are CPU-only (no GPU required).
//! - `warmup_*` tests require GPU (compile + launch).

use cutile::tile_kernel::{EntryMeta, FunctionKey, TileFunctionKey, WarmupSpec};

// TileFunctionKey hash properties 

#[test]
fn cache_key_hash_deterministic() {
    let key1 = TileFunctionKey::new(
        "mod".into(),
        "fn".into(),
        vec!["f32".into()],
        vec![],
        None,
        "abc123".into(),
        "sm_90".into(),
        "0.0.1-alpha".into(),
        "12.4".into(),
    );
    let key2 = key1.clone();
    assert_eq!(key1.get_hash_string(), key2.get_hash_string());
    assert_eq!(key1.get_disk_hash_string(), key2.get_disk_hash_string());
}

#[test]
fn cache_key_different_source_hash() {
    let key_a = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash_v1".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    let key_b = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash_v2".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
    assert_ne!(key_a.get_disk_hash_string(), key_b.get_disk_hash_string());
}

#[test]
fn cache_key_different_gpu_name() {
    let key_a = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_80".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    let key_b = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
    assert_ne!(key_a.get_disk_hash_string(), key_b.get_disk_hash_string());
}

#[test]
fn cache_key_different_compiler_version() {
    let key_a = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    let key_b = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.2".into(),
        "12.4".into(),
    );
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_cuda_toolkit_version() {
    let key_a = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    let key_b = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.6".into(),
    );
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_generics() {
    let key_a = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec!["f32".into()],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    let key_b = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec!["f16".into()],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_disk_hash_is_sha256_length() {
    let key = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
    let disk_hash = key.get_disk_hash_string();
    // SHA-256 hex output = 64 characters.
    assert_eq!(disk_hash.len(), 64, "disk hash should be 64 hex chars");
    assert!(
        disk_hash.chars().all(|c| c.is_ascii_hexdigit()),
        "disk hash should be lowercase hex"
    );
}

/// When `nvcc` is unavailable, `get_cuda_toolkit_version()` returns `"unknown"`.
/// Verify that `"unknown"` still produces a distinct key from any real version,
/// so kernels compiled without a known toolkit version are never falsely reused
/// when a real version becomes available (or vice versa).
#[test]
fn cache_key_toolkit_unknown_is_distinct() {
    let key_unknown = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "unknown".into(),
    );
    let key_real = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        None,
        "hash".into(),
        "sm_90".into(),
        "0.0.1".into(),
        "12.4".into(),
    );
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

/// Two keys that differ only in source_hash must be distinct.
/// This validates that changing a dependency (which changes the module source hash
/// at compile time) invalidates the cache — the "no false hit" guarantee.
#[test]
fn cache_key_source_hash_change_invalidates() {
    let key_v1 = TileFunctionKey::new(
        "linalg".into(),
        "matmul".into(),
        vec!["f32".into(), "128".into()],
        vec![("a".into(), vec![1, 128])],
        Some((4, 4, 1)),
        "aabbccdd11223344".into(),
        "sm_90".into(),
        "0.1.0".into(),
        "12.4".into(),
    );
    let key_v2 = TileFunctionKey::new(
        "linalg".into(),
        "matmul".into(),
        vec!["f32".into(), "128".into()],
        vec![("a".into(), vec![1, 128])],
        Some((4, 4, 1)),
        "eeff0011deadbeef".into(), // only source hash changed
        "sm_90".into(),
        "0.1.0".into(),
        "12.4".into(),
    );
    assert_ne!(key_v1.get_hash_string(), key_v2.get_hash_string());
    assert_ne!(key_v1.get_disk_hash_string(), key_v2.get_disk_hash_string());
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
