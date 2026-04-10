/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Verification benchmarks for warmup, disk cache, and compilation dedup.
//!
//! These tests are designed to **demonstrate observable effects**, not just
//! "does it not crash". Run with:
//!
//! ```bash
//! CUTILE_JIT_LOG=1 cargo test --test gpu warmup_bench -- --nocapture 2>&1
//! ```
//!
//! You should see `[cutile::jit]` logs showing cache hits vs JIT compilations,
//! and timing comparisons printed to stdout.

use crate::common;
use cutile::api;
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::{
    compile_warmup, contains_cuda_function, execute_warmup, get_default_device, CompileOptions,
    TileFunctionKey, TileKernel, WarmupSpec,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_cuda_toolkit_version, get_gpu_name,
};
use std::time::Instant;

// Use a separate module with a distinct name to avoid cache collisions with warmup.rs tests.
#[cutile::module]
mod bench_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn vector_add<T: ElementType, const N: i32>(
        z: &mut Tensor<T, { [N] }>,
        x: &Tensor<T, { [-1] }>,
        y: &Tensor<T, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let tile_y = load_tile_like_1d(y, z);
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

// Helper: run a vector_add kernel with specific generics, return wall-clock time.
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

// Warmup eliminates first-call JIT latency 
//
// Demonstrates that compile_warmup pre-compiles kernels so the first real
// call hits the memory cache instead of triggering JIT compilation.
//
// Uses different tile sizes (32 vs 64) to ensure fresh cache entries:
// - tile_size=32: called WITHOUT warmup → first call includes JIT
// - tile_size=64: called WITH warmup → first call is a cache hit
#[test]
fn warmup_eliminates_first_call_jit() {
    common::with_test_stack(|| {
        // ── Without warmup: first call includes JIT compilation ──
        let cold_duration = timed_kernel_call("32");

        // ── With warmup: pre-compile tile_size=64, then call it ──
        let warmup_t0 = Instant::now();
        bench_module::__compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "64".into()])
            .with_strides(stride_args())])
        .expect("compile_warmup failed");
        let warmup_duration = warmup_t0.elapsed();

        // Now call the warmed-up kernel — should be near-instant (cache hit).
        let warm_duration = timed_kernel_call("64");

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
        if warm_duration < cold_duration {
            let speedup = cold_duration.as_secs_f64() / warm_duration.as_secs_f64().max(0.001);
            println!(
                "║  ✓ Warmed-up call is {:.1}x faster                       ║",
                speedup
            );
        } else {
            println!("║  (both calls similar — kernel may already be cached)    ║");
        }
        println!("╚══════════════════════════════════════════════════════════╝\n");

        // The warmed-up call should be significantly faster than the cold call.
        // We don't assert a specific ratio because CI timing varies, but the
        // JIT log output (CUTILE_JIT_LOG=1) provides definitive evidence.
        // At minimum, verify the kernel IS in cache after warmup.
        let device_id = get_default_device();
        let key = TileFunctionKey::new(
            "bench_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "64".into()],
            stride_args(),
            None,
            CompileOptions::default(),
            bench_module::__SOURCE_HASH.into(),
            get_gpu_name(device_id),
            get_compiler_version(),
            get_cuda_toolkit_version(),
        );
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in memory cache after warmup"
        );
    });
}

/// Second call always hits memory cache 
//
/// Demonstrates that the second call to the same kernel is a memory cache hit,
/// regardless of whether warmup was used.
#[test]
fn second_call_hits_memory_cache() {
    common::with_test_stack(|| {
        // First call: JIT compiles (tile=16, unique to this test).
        let first = timed_kernel_call("16");

        // Second call: memory cache hit.
        let second = timed_kernel_call("16");

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║       Memory Cache Verification: 1st vs 2nd Call        ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  First  call (tile=16): {:>10.1?}  (JIT compile)      ║",
            first
        );
        println!(
            "║  Second call (tile=16): {:>10.1?}  (memory cache)     ║",
            second
        );
        println!("╠══════════════════════════════════════════════════════════╣");
        if second < first {
            let speedup = first.as_secs_f64() / second.as_secs_f64().max(0.001);
            println!(
                "║  ✓ Cache hit is {:.0}x faster                             ║",
                speedup
            );
        }
        println!("╚══════════════════════════════════════════════════════════╝\n");

        // Memory cache hit should be dramatically faster (100x+ typical).
        assert!(
            second < first,
            "second call ({second:?}) should be faster than first ({first:?})"
        );
    });
}

// Compile_warmup + execute_warmup combined flow 
//
// Demonstrates a realistic warmup workflow:
// 1. compile_warmup pre-compiles the kernel
// 2. execute_warmup runs it with real data (also warms CUDA runtime)
// 3. Subsequent calls are fast
#[test]
fn full_warmup_workflow() {
    common::with_test_stack(|| {
        // Step 1: Pre-compile via compile_warmup.
        let t0 = Instant::now();
        bench_module::__compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "128".into()])
            .with_strides(stride_args())])
        .expect("compile_warmup failed");
        let compile_time = t0.elapsed();

        // Step 2: Execute warmup with real data (warms CUDA runtime).
        let t1 = Instant::now();
        execute_warmup(|| {
            let x = api::ones::<f32>(&[256]).sync()?;
            let y = api::ones::<f32>(&[256]).sync()?;
            let z = api::zeros::<f32>(&[256]).partition([128]).sync()?;
            let _result = bench_module::vector_add(z, &x, &y)
                .generics(vec!["f32".into(), "128".into()])
                .sync()?;
            Ok(())
        })
        .expect("execute_warmup failed");
        let execute_time = t1.elapsed();

        // Step 3: Production call — should be fast.
        let production_time = timed_kernel_call("128");

        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║           Full Warmup Workflow Verification              ║");
        println!("╠══════════════════════════════════════════════════════════╣");
        println!(
            "║  1. compile_warmup:   {:>10.1?}  (JIT to cache)       ║",
            compile_time
        );
        println!(
            "║  2. execute_warmup:   {:>10.1?}  (cache + CUDA init)  ║",
            execute_time
        );
        println!(
            "║  3. production call:  {:>10.1?}  (fully warm)         ║",
            production_time
        );
        println!("╚══════════════════════════════════════════════════════════╝\n");

        // execute_warmup should be faster than compile_warmup (no JIT).
        // production call should be fastest (everything cached + CUDA warm).
        assert!(
            execute_time < compile_time,
            "execute_warmup ({execute_time:?}) should be faster than compile_warmup ({compile_time:?})"
        );
    });
}
