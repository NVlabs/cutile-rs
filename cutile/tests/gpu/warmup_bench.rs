/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Verification benchmarks for warmup and in-memory compilation caching.
//!
//! Tests assert cache behavior via [`jit_compile_count`] (process-global counter,
//! +1 per real JIT compile, +0 on cache hits) — not wall-clock timing. Durations
//! are printed for inspection but never asserted.
//!
//! All tests hold [`common::cache_test_lock`] to prevent concurrent tests from
//! moving the counter during the measured window.
//!
//! Run with `CUTILE_JIT_LOG=1 cargo test --test gpu warmup_bench -- --nocapture`
//! to see per-compile vs cache-hit logs.

use crate::common;
use cutile::api;
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::{
    contains_cuda_function, get_default_device, jit_compile_count,
    CompileOptions, TileFunctionKey, TileKernel, WarmupSpec,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_cuda_toolkit_version, get_gpu_name,
};
use cutile_compiler::specialization::SpecializationBits;
use std::time::Instant;

// Distinct module name avoids cache key collisions with warmup.rs tests.
#[cutile::module]
mod bench_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn vector_add<T: ElementType, const N: i32>(
        z: &mut Tensor<T, { [N] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

fn stride_args() -> Vec<(String, Vec<i32>)> {
    vec![
        ("z".to_string(), vec![1]),
        ("x".to_string(), vec![1]),
        ("y".to_string(), vec![1]),
    ]
}

fn vector_add_spec_args(len: usize, tile: usize) -> Vec<(String, SpecializationBits)> {
    let x = api::ones::<f32>(&[len]).sync().unwrap();
    let y = api::ones::<f32>(&[len]).sync().unwrap();
    let z = api::zeros::<f32>(&[len]).partition([tile]).sync().unwrap();
    let z_spec = z.unpartition().spec().clone();
    vec![
        ("z".to_string(), z_spec),
        ("x".to_string(), x.spec().clone()),
        ("y".to_string(), y.spec().clone()),
    ]
}

fn bench_key(
    generics: Vec<String>,
    spec_args: Vec<(String, SpecializationBits)>,
) -> TileFunctionKey {
    let device_id = get_default_device();
    TileFunctionKey::new(
        "bench_module".into(),
        "vector_add".into(),
        generics,
        stride_args(),
        spec_args,
        vec![],
        None,
        CompileOptions::default(),
        bench_module::_SOURCE_HASH.into(),
        get_gpu_name(device_id),
        get_compiler_version(),
        get_cuda_toolkit_version(),
    )
}

fn timed_kernel_call(tile_size: &str) -> std::time::Duration {
    let n: usize = tile_size.parse().unwrap();
    let t0 = Instant::now();
    let x = api::ones::<f32>(&[256]).sync().unwrap();
    let y = api::ones::<f32>(&[256]).sync().unwrap();
    let z = api::zeros::<f32>(&[256]).partition([n]).sync().unwrap();
    let _result = bench_module::vector_add(z, &x, &y)
        .generics(vec!["f32".into(), tile_size.into()])
        .sync()
        .unwrap();
    t0.elapsed()
}

// tile=32: called without warmup → first call is the JIT compile (miss).
// tile=64: pre-compiled by warmup → first real call is a cache hit.
// Fill kernel primed via spec_args_64 so only vector_add moves the counter.
#[test]
fn warmup_eliminates_first_call_jit() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        // Prime fill kernel so only vector_add moves the counter below.
        let spec_args_64 = vector_add_spec_args(256, 64);

        let c0 = jit_compile_count();
        let cold_duration = timed_kernel_call("32");
        let c_after_cold = jit_compile_count();
        assert_eq!(
            c_after_cold,
            c0 + 1,
            "un-warmed first call to tile=32 must perform exactly one JIT \
             compile (only bench_module::vector_add; full_apply was primed)"
        );

        let warmup_t0 = Instant::now();
        bench_module::_compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "64".into()])
            .with_strides(stride_args())
            .with_spec_args(spec_args_64.clone())])
        .expect("compile_warmup failed");
        let warmup_duration = warmup_t0.elapsed();
        let c_after_warmup = jit_compile_count();
        assert_eq!(
            c_after_warmup,
            c_after_cold + 1,
            "compile_warmup for tile=64 must perform exactly one JIT compile"
        );

        let warm_duration = timed_kernel_call("64");
        let c_after_warm = jit_compile_count();
        assert_eq!(
            c_after_warm, c_after_warmup,
            "warmed-up first call to tile=64 must NOT compile (cache hit): \
             counter moved from {c_after_warmup} to {c_after_warm}"
        );

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║         Warmup Verification: First-Call Latency         ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  Without warmup (tile=32): {:>10.1?}  (includes JIT)  ║",
            cold_duration
        );
        println!(
            "║  Warmup step     (tile=64): {:>10.1?}  (pre-compile)   ║",
            warmup_duration
        );
        println!(
            "║  With warmup     (tile=64): {:>10.1?}  (cache hit)     ║",
            warm_duration
        );
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  JIT compiles: cold +1, warmup +1, warmed call +0       ║"
        );
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let device_id = get_default_device();
        let key = bench_key(vec!["f32".into(), "64".into()], spec_args_64);
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in memory cache after warmup"
        );
    });
}

#[test]
fn second_call_hits_memory_cache() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        // tile=16 is unique to this test; priming fill kernel keeps counter clean.
        let spec_args_16 = vector_add_spec_args(256, 16);
        let c0 = jit_compile_count();

        let first = timed_kernel_call("16");
        let c_after_first = jit_compile_count();
        assert_eq!(
            c_after_first,
            c0 + 1,
            "first call to tile=16 must perform exactly one JIT compile \
             (only bench_module::vector_add; full_apply was primed)"
        );

        let second = timed_kernel_call("16");
        let c_after_second = jit_compile_count();
        assert_eq!(
            c_after_second, c_after_first,
            "second call to tile=16 must NOT compile (cache hit): \
             counter moved from {c_after_first} to {c_after_second}"
        );

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║       Memory Cache Verification: 1st vs 2nd Call        ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  First  call (tile=16): {:>10.1?}  (JIT: +1 compile)  ║",
            first
        );
        println!(
            "║  Second call (tile=16): {:>10.1?}  (cache: +0 compile)║",
            second
        );
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let device_id = get_default_device();
        let key = bench_key(vec!["f32".into(), "16".into()], spec_args_16);
        assert!(
            contains_cuda_function(device_id, &key),
            "tile=16 kernel should be in memory cache after first call"
        );
    });
}

// Steps 2 and 3 being cache hits proves compile_warmup's key matches the
// launch-derived key — the property that makes warmup useful.
#[test]
fn full_warmup_workflow() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        // Prime fill kernel so only vector_add moves the counter below.
        let spec_args_128 = vector_add_spec_args(256, 128);
        let c0 = jit_compile_count();

        // Step 1: compile_warmup (one miss).
        let t0 = Instant::now();
        bench_module::_compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "128".into()])
            .with_strides(stride_args())
            .with_spec_args(spec_args_128.clone())])
        .expect("compile_warmup failed");
        let compile_time = t0.elapsed();
        let c_after_compile = jit_compile_count();
        assert_eq!(
            c_after_compile,
            c0 + 1,
            "compile_warmup must perform exactly one JIT compile"
        );

        // Step 2: first real launch — same key → cache hit. 
        let t1 = Instant::now();
        let x = api::ones::<f32>(&[256]).sync().unwrap();
        let y = api::ones::<f32>(&[256]).sync().unwrap();
        let z = api::zeros::<f32>(&[256]).partition([128]).sync().unwrap();
        let _ = bench_module::vector_add(z, &x, &y)
            .generics(vec!["f32".into(), "128".into()])
            .sync()
            .unwrap();
        let first_launch_time = t1.elapsed();
        let c_after_first = jit_compile_count();
        assert_eq!(
            c_after_first, c_after_compile,
            "first launch after compile_warmup must NOT recompile (key must \
             match): counter moved from {c_after_compile} to {c_after_first}"
        );

        // Step 3: production call — cache hit.
        let production_time = timed_kernel_call("128");
        let c_after_prod = jit_compile_count();
        assert_eq!(
            c_after_prod, c_after_compile,
            "production call after warmup must NOT recompile (cache hit): \
             counter moved from {c_after_compile} to {c_after_prod}"
        );

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║           Full Warmup Workflow Verification              ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  1. compile_warmup:   {:>10.1?}  (JIT: +1 compile)    ║",
            compile_time
        );
        println!(
            "║  2. first real launch: {:>10.1?}  (cache: +0 compile)  ║",
            first_launch_time
        );
        println!(
            "║  3. production call:  {:>10.1?}  (cache: +0 compile)  ║",
            production_time
        );
        println!("║  Total JIT compiles for the whole workflow: 1           ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");

        let device_id = get_default_device();
        let key = bench_key(vec!["f32".into(), "128".into()], spec_args_128);
        assert!(
            contains_cuda_function(device_id, &key),
            "tile=128 kernel should be in memory cache after warmup workflow"
        );
    });
}
