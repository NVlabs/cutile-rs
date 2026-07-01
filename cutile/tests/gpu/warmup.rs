/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration tests for the kernel cache: distinct-key population under
//! concurrency and single-flight compile dedup. Warmup itself is exercised via
//! the `.compile()` terminal (see `warmup_bench::meta_compile_terminal_warms_cache`).

use crate::common;
use cutile::api;
use cutile::prelude::{DeviceOp, PartitionOp};
use cutile::tile_kernel::{
    contains_cuda_function, get_default_device, jit_compile_count, TileFunctionKey, TileKernel,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_cuda_toolkit_version, get_gpu_name,
};
use std::sync::Arc;

#[cutile::module]
mod warmup_test_module {
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

fn vector_add_stride_args() -> Vec<(String, Vec<i32>)> {
    vec![
        ("z".to_string(), vec![1]),
        ("x".to_string(), vec![1]),
        ("y".to_string(), vec![1]),
    ]
}

#[test]
fn different_keys_parallel_compile() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let b1 = Arc::clone(&barrier);
        let h1 = std::thread::Builder::new()
            .stack_size(common::TEST_STACK_SIZE)
            .spawn(move || {
                b1.wait();
                let x = api::ones::<f32>(&[256]).sync().unwrap();
                let y = api::ones::<f32>(&[256]).sync().unwrap();
                let z = api::zeros::<f32>(&[256]).partition([128]).sync().unwrap();
                warmup_test_module::vector_add(z, &x, &y)
                    .generics(vec!["f32".into(), "128".into()])
                    .sync()
                    .unwrap();
            })
            .unwrap();

        let b2 = Arc::clone(&barrier);
        let h2 = std::thread::Builder::new()
            .stack_size(common::TEST_STACK_SIZE)
            .spawn(move || {
                b2.wait();
                let x = api::ones::<f32>(&[512]).sync().unwrap();
                let y = api::ones::<f32>(&[512]).sync().unwrap();
                let z = api::zeros::<f32>(&[512]).partition([256]).sync().unwrap();
                warmup_test_module::vector_add(z, &x, &y)
                    .generics(vec!["f32".into(), "256".into()])
                    .sync()
                    .unwrap();
            })
            .unwrap();

        h1.join().expect("thread 1 panicked");
        h2.join().expect("thread 2 panicked");

        let x_probe_8 = api::ones::<f32>(&[256]).sync().unwrap();
        let y_probe_8 = api::ones::<f32>(&[256]).sync().unwrap();
        let z_probe_8 = api::zeros::<f32>(&[256]).partition([128]).sync().unwrap();
        let z_spec_8 = z_probe_8.unpartition().spec().clone();

        let x_probe_32 = api::ones::<f32>(&[512]).sync().unwrap();
        let y_probe_32 = api::ones::<f32>(&[512]).sync().unwrap();
        let z_probe_32 = api::zeros::<f32>(&[512]).partition([256]).sync().unwrap();
        let z_spec_32 = z_probe_32.unpartition().spec().clone();

        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);
        let cv = get_compiler_version();
        let tv = get_cuda_toolkit_version();
        let key_8 = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "128".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vec![
                ("z".to_string(), z_spec_8),
                ("x".to_string(), x_probe_8.spec().clone()),
                ("y".to_string(), y_probe_8.spec().clone()),
            ])
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(gpu_name.clone())
            .compiler_version(cv.clone())
            .cuda_toolkit_version(tv.clone())
            .build();
        let key_32 = TileFunctionKey::builder("warmup_test_module", "vector_add")
            .generics(vec!["f32".into(), "256".into()])
            .stride_args(vector_add_stride_args())
            .spec_args(vec![
                ("z".to_string(), z_spec_32),
                ("x".to_string(), x_probe_32.spec().clone()),
                ("y".to_string(), y_probe_32.spec().clone()),
            ])
            .source_hash(warmup_test_module::_SOURCE_HASH)
            .gpu_name(gpu_name)
            .compiler_version(cv)
            .cuda_toolkit_version(tv)
            .build();
        assert!(contains_cuda_function(device_id, &key_8));
        assert!(contains_cuda_function(device_id, &key_32));
    });
}

// 4 threads race the same fresh kernel; single-flight dedup must fire exactly once.
// Proven by jit_compile_count() — broken dedup would compile up to 4 times.
#[test]
fn multi_thread_dedup_timing_evidence() {
    common::with_test_stack(|| {
        let _guard = common::cache_test_lock();

        // Prime the fill kernel (api::ones/zeros compile full_apply once) so only
        // vector_add can move the counter during the race below.
        let _prime = api::ones::<f32>(&[256]).sync().unwrap();

        // "single" baseline: warm one specialization via the compile terminal
        // (meta inputs, no launch, no allocation).
        let t_single = std::time::Instant::now();
        let z = api::meta::<f32>(&[256]).partition([128]);
        let x = api::meta::<f32>(&[256]);
        let y = api::meta::<f32>(&[256]);
        warmup_test_module::vector_add(z, x, y)
            .generics(vec!["f32".into(), "128".into()])
            .compile()
            .unwrap();
        let single_duration = t_single.elapsed();

        // tile=8 is fresh for the race.
        let c_before_race = jit_compile_count();
        let n_threads = 4;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let t_parallel = std::time::Instant::now();
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::Builder::new()
                    .stack_size(common::TEST_STACK_SIZE)
                    .spawn(move || {
                        barrier.wait();
                        let x = api::ones::<f32>(&[256]).sync().unwrap();
                        let y = api::ones::<f32>(&[256]).sync().unwrap();
                        let z = api::zeros::<f32>(&[256]).partition([8]).sync().unwrap();
                        warmup_test_module::vector_add(z, &x, &y)
                            .generics(vec!["f32".into(), "8".into()])
                            .sync()
                            .unwrap();
                    })
                    .unwrap()
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let parallel_duration = t_parallel.elapsed();
        let compiles_during_race = jit_compile_count() - c_before_race;

        eprintln!(
            "[dedup] single={single_duration:.1?}  parallel(4)={parallel_duration:.1?}  compiles_during_race={compiles_during_race}"
        );

        assert_eq!(
            compiles_during_race, 1,
            "single-flight dedup: 4 concurrent threads on the same fresh kernel \
             must trigger exactly ONE JIT compile, got {compiles_during_race}"
        );
    });
}
