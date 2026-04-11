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

/// Test helper that builds a `TileFunctionKey` with sensible defaults.
///
/// `TileFunctionKey::new` takes 11 positional arguments, so constructing keys
/// inline in every test is noisy and error-prone. This struct lets each test
/// override only the field it cares about via Rust's `..` update syntax:
///
/// ```
/// let key_a = TestKey { gpu_name: "sm_80".into(), ..TestKey::new() }.build();
/// let key_b = TestKey { gpu_name: "sm_90".into(), ..TestKey::new() }.build();
/// ```
#[derive(Clone)]
struct TestKey {
    module_name: String,
    function_name: String,
    generics: Vec<String>,
    stride_args: Vec<(String, Vec<i32>)>,
    spec_args: Vec<(String, SpecializationBits)>,
    grid: Option<(u32, u32, u32)>,
    compile_options: CompileOptions,
    source_hash: String,
    gpu_name: String,
    compiler_version: String,
    cuda_toolkit_version: String,
}

impl TestKey {
    fn new() -> Self {
        Self {
            module_name: "m".into(),
            function_name: "f".into(),
            generics: vec![],
            stride_args: vec![],
            spec_args: vec![],
            grid: None,
            compile_options: CompileOptions::default(),
            source_hash: "hash".into(),
            gpu_name: "sm_90".into(),
            compiler_version: "0.0.1".into(),
            cuda_toolkit_version: "12.4".into(),
        }
    }

    fn build(self) -> TileFunctionKey {
        TileFunctionKey::new(
            self.module_name,
            self.function_name,
            self.generics,
            self.stride_args,
            self.spec_args,
            self.grid,
            self.compile_options,
            self.source_hash,
            self.gpu_name,
            self.compiler_version,
            self.cuda_toolkit_version,
        )
    }
}

// TileFunctionKey hash properties

#[test]
fn cache_key_hash_deterministic() {
    let key1 = TestKey {
        module_name: "mod".into(),
        function_name: "fn".into(),
        generics: vec!["f32".into()],
        source_hash: "abc123".into(),
        compiler_version: "0.0.1-alpha".into(),
        ..TestKey::new()
    }
    .build();
    let key2 = key1.clone();
    assert_eq!(key1.get_hash_string(), key2.get_hash_string());
    assert_eq!(key1.get_disk_hash_string(), key2.get_disk_hash_string());
}

#[test]
fn cache_key_different_source_hash() {
    let key_a = TestKey {
        source_hash: "hash_v1".into(),
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        source_hash: "hash_v2".into(),
        ..TestKey::new()
    }
    .build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
    assert_ne!(key_a.get_disk_hash_string(), key_b.get_disk_hash_string());
}

#[test]
fn cache_key_different_gpu_name() {
    let key_a = TestKey {
        gpu_name: "sm_80".into(),
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        gpu_name: "sm_90".into(),
        ..TestKey::new()
    }
    .build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
    assert_ne!(key_a.get_disk_hash_string(), key_b.get_disk_hash_string());
}

#[test]
fn cache_key_different_compiler_version() {
    let key_a = TestKey {
        compiler_version: "0.0.1".into(),
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        compiler_version: "0.0.2".into(),
        ..TestKey::new()
    }
    .build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_cuda_toolkit_version() {
    let key_a = TestKey {
        cuda_toolkit_version: "12.4".into(),
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        cuda_toolkit_version: "12.6".into(),
        ..TestKey::new()
    }
    .build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_different_generics() {
    let key_a = TestKey {
        generics: vec!["f32".into()],
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        generics: vec!["f16".into()],
        ..TestKey::new()
    }
    .build();
    assert_ne!(key_a.get_hash_string(), key_b.get_hash_string());
}

#[test]
fn cache_key_disk_hash_is_sha256_length() {
    let key = TestKey::new().build();
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
    let key_unknown = TestKey {
        cuda_toolkit_version: "unknown".into(),
        ..TestKey::new()
    }
    .build();
    let key_real = TestKey {
        cuda_toolkit_version: "12.4".into(),
        ..TestKey::new()
    }
    .build();
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
    let base = TestKey {
        module_name: "linalg".into(),
        function_name: "matmul".into(),
        generics: vec!["f32".into(), "128".into()],
        stride_args: vec![("a".into(), vec![1, 128])],
        grid: Some((4, 4, 1)),
        compiler_version: "0.1.0".into(),
        ..TestKey::new()
    };
    let key_v1 = TestKey {
        source_hash: "aabbccdd11223344".into(),
        ..base.clone()
    }
    .build();
    let key_v2 = TestKey {
        source_hash: "eeff0011deadbeef".into(),
        ..base
    }
    .build();
    assert_ne!(key_v1.get_hash_string(), key_v2.get_hash_string());
    assert_ne!(key_v1.get_disk_hash_string(), key_v2.get_disk_hash_string());
}

/// Two tensors with the same shape/stride layout but different alignment
/// (e.g. a 16-byte-aligned base pointer vs a 4-byte-aligned one) trigger
/// different `assume_div_by` operations in the generated MLIR, and therefore
/// produce different cubins. The cache key must reflect this so a kernel
/// compiled for aligned data is never falsely reused for misaligned data.
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
    let key_a = TestKey {
        spec_args: vec![("x".into(), spec_aligned)],
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        spec_args: vec![("x".into(), spec_misaligned)],
        ..TestKey::new()
    }
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

/// `CompileOptions` (`occupancy`, `num_cta_in_cga`, `max_divisibility`) are
/// kernel-level hints that change codegen. Two launches with different hints
/// must land on different cache entries — otherwise a kernel compiled with
/// `max_divisibility=16` could be silently reused for a launch that expected
/// `max_divisibility=4`, producing incorrect assumptions about alignment.
#[test]
fn cache_key_different_compile_options() {
    let key_a = TestKey {
        compile_options: CompileOptions::default().max_divisibility(8),
        ..TestKey::new()
    }
    .build();
    let key_b = TestKey {
        compile_options: CompileOptions::default().max_divisibility(16),
        ..TestKey::new()
    }
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

    // Also check that a different field (occupancy) flips the key.
    let key_c = TestKey {
        compile_options: CompileOptions::default().occupancy(2),
        ..TestKey::new()
    }
    .build();
    let key_d = TestKey {
        compile_options: CompileOptions::default().occupancy(4),
        ..TestKey::new()
    }
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
