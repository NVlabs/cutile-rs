/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration tests for compile_warmup and execute_warmup.

use crate::common;
use cutile::api;
use cutile::jit_store::FileSystemJitStore;
use cutile::tile_kernel::{
    compile_warmup, contains_cuda_function, execute_warmup, get_default_device, get_kernel_cache,
    load_module_from_bytes, CompileOptions, DeviceOperation, FunctionKey, IntoDeviceOperationPartition,
    TileFunctionKey, TileKernel, WarmupSpec,
};
use cutile_compiler::cuda_tile_runtime_utils::{
    get_compiler_version, get_cuda_toolkit_version, get_gpu_name,
};
use std::sync::Arc;
use std::process::Command;

#[cutile::module]
mod warmup_test_module {
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

fn vector_add_stride_args() -> Vec<(String, Vec<i32>)> {
    vec![
        ("z".to_string(), vec![1]),
        ("x".to_string(), vec![1]),
        ("y".to_string(), vec![1]),
    ]
}

fn unique_temp_cache_dir(tag: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time went backwards")
        .as_nanos();
    std::env::temp_dir().join(format!("cutile_warmup_{tag}_{}_{}", std::process::id(), nanos))
}

fn run_warmup_worker(role: &str, cache_root: &std::path::Path) -> std::process::Output {
    let exe = std::env::current_exe().expect("failed to resolve current test binary path");
    let no_disk = if role == "no-disk-cache" { "1" } else { "0" };
    let mut cmd = Command::new(exe);
    cmd.arg("--exact")
        .arg("warmup::cross_process_warmup_worker")
        .arg("--nocapture")
        .env("CUTILE_WARMUP_WORKER_ROLE", role)
        .env("XDG_CACHE_HOME", cache_root)
        .env("CUTILE_JIT_LOG", "1")
        .env("CUTILE_NO_DISK_CACHE", no_disk);
    cmd.output().expect("failed to run warmup worker")
}

// Compile_warmup 

#[test]
fn compile_warmup_populates_cache() {
    common::with_test_stack(move || {
        // Uses the macro-generated __compile_warmup helper — callers only pass specs.
        warmup_test_module::__compile_warmup(&[WarmupSpec::new(
            "vector_add",
            vec!["f32".into(), "64".into()],
        )
        .with_strides(vector_add_stride_args())])
        .expect("__compile_warmup failed");

        // Verify the kernel is now in the global cache.
        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);
        let key = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "64".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            gpu_name,
            get_compiler_version(),
            get_cuda_toolkit_version(),
        );
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in cache after compile_warmup"
        );
    });
}

#[test]
fn compile_warmup_skips_duplicate() {
    common::with_test_stack(move || {
        let specs = &[WarmupSpec::new("vector_add", vec!["f32".into(), "128".into()])
            .with_strides(vector_add_stride_args())];
        // First call compiles.
        warmup_test_module::__compile_warmup(specs)
            .expect("first compile_warmup failed");

        // Second call should be a no-op (hits cache).
        warmup_test_module::__compile_warmup(specs)
            .expect("second compile_warmup failed");
    });
}

#[test]
fn compile_warmup_unknown_function_errors() {
    common::with_test_stack(|| {
        let result = warmup_test_module::__compile_warmup(
            &[WarmupSpec::new("nonexistent_fn", vec!["f32".into()])],
        );
        assert!(result.is_err(), "should error for unknown function");
    });
}

// Compile_warmup with JitStore disk persistence 
#[test]
fn compile_warmup_persists_to_disk() {
    common::with_test_stack(|| {
        let dir =
            std::env::temp_dir().join(format!("cutile_warmup_disk_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileSystemJitStore::new(dir.clone()).expect("failed to create store");

        // Configure JitStore (note: can only be set once per process).
        // If this fails because it's already set from another test, that's OK —
        // just skip the disk assertions.
        let store_was_set = cuda_async::jit_store::set_jit_store_if_unset(Some(Box::new(store)));

        warmup_test_module::__compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "256".into()])
            .with_strides(vector_add_stride_args())])
        .expect("compile_warmup failed");

        if store_was_set {
            // Verify a cubin file was written to disk.
            let cubin_count = std::fs::read_dir(&dir)
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .path()
                        .extension()
                        .is_some_and(|ext| ext == "cubin")
                })
                .count();
            assert!(
                cubin_count > 0,
                "at least one .cubin should be persisted to disk"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    });
}

// Execute_warmup 
#[test]
fn execute_warmup_runs_kernel() {
    common::with_test_stack(|| {
        execute_warmup(|| {
            let x = api::ones::<1, f32>([256]).sync()?;
            let y = api::ones::<1, f32>([256]).sync()?;
            let z = api::zeros::<1, f32>([256]).partition([64]).sync()?;
            let _result = warmup_test_module::vector_add(z, x.into(), y.into())
                .generics(vec!["f32".into(), "64".into()])
                .sync()?;
            Ok(())
        })
        .expect("execute_warmup failed");
    });
}

// Multi-thread compilation dedup 
// Spawns multiple threads that all compile the same kernel specialization
// concurrently.  Verifies that all threads succeed and the kernel ends up
// in cache.  With single-flight dedup, only one thread performs the actual
// JIT; the rest wait and get the cached result.
#[test]
fn multi_thread_compile_dedup() {
    common::with_test_stack(|| {
        let n_threads = 4;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::Builder::new()
                    .stack_size(common::TEST_STACK_SIZE)
                    .spawn(move || {
                        barrier.wait();
                        let x = api::ones::<1, f32>([256]).sync().unwrap();
                        let y = api::ones::<1, f32>([256]).sync().unwrap();
                        let z = api::zeros::<1, f32>([256]).partition([8]).sync().unwrap();
                        warmup_test_module::vector_add(z, x.into(), y.into())
                            .generics(vec!["f32".into(), "8".into()])
                            .sync()
                            .unwrap();
                    })
                    .unwrap()
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked during concurrent compile");
        }

        // Verify the kernel is in cache exactly once.
        let device_id = get_default_device();
        let key = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "8".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            get_gpu_name(device_id),
            get_compiler_version(),
            get_cuda_toolkit_version(),
        );
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in cache after concurrent compilation"
        );
    });
}



// Verifies the disk → memory cache path: compile a kernel (populating both
// caches), evict from memory, re-warmup → the second compilation should load
// from disk instead of re-JIT-compiling.
#[test]
fn disk_cache_hit_after_memory_eviction() {
    common::with_test_stack(|| {
        let dir = std::env::temp_dir().join(format!(
            "cutile_disk_read_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = FileSystemJitStore::new(dir.clone()).expect("failed to create store");
        let store_was_set =
            cuda_async::jit_store::set_jit_store_if_unset(Some(Box::new(store)));

        if !store_was_set {
            // Another test already configured the JitStore — we can't control the
            // disk directory, so skip.  The JIT log output still demonstrates the
            // path when run standalone.
            println!("Skipping disk_cache_hit_after_memory_eviction: JitStore already set");
            return;
        }

        let specs = &[WarmupSpec::new("vector_add", vec!["f32".into(), "2".into()])
            .with_strides(vector_add_stride_args())];

        // Step 1: compile → populates memory + disk.
        warmup_test_module::__compile_warmup(specs)
        .expect("first compile_warmup failed");

        // Verify cubin was written to disk.
        let cubin_count = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "cubin")
            })
            .count();
        assert!(cubin_count > 0, "cubin should be on disk after compile");

        // Step 2: evict from memory cache.
        let device_id = get_default_device();
        let key = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "2".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            get_gpu_name(device_id),
            get_compiler_version(),
            get_cuda_toolkit_version(),
        );
        let key_str = key.get_hash_string();
        get_kernel_cache().remove(&key_str);
        assert!(
            !contains_cuda_function(device_id, &key),
            "should be evicted from memory"
        );

        // Step 3: re-warmup → should hit disk cache (visible with CUTILE_JIT_LOG=1).
        warmup_test_module::__compile_warmup(specs)
        .expect("second compile_warmup (disk hit) failed");

        // Step 4: verify back in memory cache.
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in memory cache after disk hit"
        );

        let _ = std::fs::remove_dir_all(&dir);
    });
}

// Compilation failure does not poison cache  
// Verifies that a failed compile_warmup (unknown function) does not prevent
// a subsequent valid warmup from succeeding.
#[test]
fn failed_warmup_does_not_poison_cache() {
    common::with_test_stack(|| {
        // First call: invalid function name → should error.
        let result = warmup_test_module::__compile_warmup(
            &[WarmupSpec::new("nonexistent_fn", vec!["f32".into()])],
        );
        assert!(result.is_err(), "should error for unknown function");

        // Second call: valid params → should succeed despite prior failure.
        warmup_test_module::__compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "4".into()])
            .with_strides(vector_add_stride_args())])
        .expect("valid warmup should succeed after failed one");

        let device_id = get_default_device();
        let key = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "4".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            get_gpu_name(device_id),
            get_compiler_version(),
            get_cuda_toolkit_version(),
        );
        assert!(
            contains_cuda_function(device_id, &key),
            "kernel should be in cache after recovery from prior failure"
        );
    });
}

// Multi-spec warmup does not skip subsequent specs 
// Regression test for the IIFE fix in compile_warmup.  Pre-compile spec A,
// then warmup [A, B].  Spec A should be skipped (cache hit) and spec B must
// still be compiled.  Without the IIFE, a `return Ok(())` in the cache-hit
// path would exit the outer function, silently skipping B.
#[test]
fn multi_spec_warmup_compiles_all() {
    common::with_test_stack(|| {
        // Pre-compile spec A so it's in cache.
        warmup_test_module::__compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "32".into()])
            .with_strides(vector_add_stride_args())])
        .expect("pre-compile spec A failed");

        // Now warmup [A, B] — A should skip, B should compile.
        warmup_test_module::__compile_warmup(&[
            WarmupSpec::new("vector_add", vec!["f32".into(), "32".into()])
                .with_strides(vector_add_stride_args()),
            WarmupSpec::new("vector_add", vec!["f32".into(), "16".into()])
                .with_strides(vector_add_stride_args()),
        ])
        .expect("multi-spec warmup failed");

        // Verify BOTH are in cache — the critical check is that B was compiled.
        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);
        let compiler_version = get_compiler_version();
        let cuda_toolkit_version = get_cuda_toolkit_version();

        let key_a = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "32".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            gpu_name.clone(),
            compiler_version.clone(),
            cuda_toolkit_version.clone(),
        );
        let key_b = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "16".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            gpu_name,
            compiler_version,
            cuda_toolkit_version,
        );

        assert!(
            contains_cuda_function(device_id, &key_a),
            "spec A should be in cache"
        );
        assert!(
            contains_cuda_function(device_id, &key_b),
            "spec B must be in cache — multi-spec warmup should not skip subsequent specs"
        );
    });
}

// Load_module_from_bytes multi-thread safety 
// Verifies that load_module_from_bytes can be called concurrently from
// multiple threads without tmp file collisions (validates C2 fix).
#[test]
fn load_module_from_bytes_concurrent() {
    common::with_test_stack(|| {
        // First, compile a kernel to get valid cubin bytes.
        warmup_test_module::__compile_warmup(&[WarmupSpec::new("vector_add", vec!["f32".into(), "64".into()])
            .with_strides(vector_add_stride_args())])
        .expect("warmup failed");

        // Find the cubin file from the JitStore directory or compile output.
        // Use the compile_module path to get a cubin: re-compile and read the file.
        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);

        // Get cubin bytes by compiling the kernel and reading the output file.
        let modules =
            cutile_compiler::compiler::CUDATileModules::new(warmup_test_module::_module_asts())
                .unwrap();
        let compiler = cutile_compiler::compiler::CUDATileFunctionCompiler::new(
            &modules,
            "warmup_test_module",
            "vector_add",
            &["f32".to_string(), "64".to_string()],
            &[
                ("z", &[1i32][..]),
                ("x", &[1i32][..]),
                ("y", &[1i32][..]),
            ],
            None,
            gpu_name.clone(),
            &CompileOptions::default(),
        )
        .unwrap();
        let module_op = compiler.compile().unwrap();
        let cubin_filename =
            cutile_compiler::cuda_tile_runtime_utils::compile_module(&module_op, &gpu_name);
        let cubin_bytes = std::fs::read(&cubin_filename).expect("failed to read cubin");

        // Spawn multiple threads, each loading the same cubin bytes.
        let n_threads = 4;
        let barrier = Arc::new(std::sync::Barrier::new(n_threads));
        let cubin = Arc::new(cubin_bytes);

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let cubin = Arc::clone(&cubin);
                std::thread::Builder::new()
                    .stack_size(common::TEST_STACK_SIZE)
                    .spawn(move || {
                        barrier.wait();
                        let module = load_module_from_bytes(&cubin, device_id)
                            .expect("load_module_from_bytes failed");
                        // Verify the module can load the function.
                        let _func = module
                            .load_function("vector_add_entry")
                            .expect("failed to load function from module");
                    })
                    .unwrap()
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked in load_module_from_bytes");
        }
    });
}

// Empty specs returns Ok(())
#[test]
fn compile_warmup_empty_specs() {
    common::with_test_stack(|| {
        let result = warmup_test_module::__compile_warmup(&[]);
        assert!(
            result.is_ok(),
            "compile_warmup with empty specs should return Ok(())"
        );
    });
}

// Corrupted cubin bytes must return Err, not panic
#[test]
fn corrupted_cubin_returns_error_not_panic() {
    common::with_test_stack(|| {
        let device_id = get_default_device();
        let result = load_module_from_bytes(b"this is not a valid cubin", device_id);
        assert!(
            result.is_err(),
            "corrupted cubin should return Err, not panic"
        );
    });
}

// Different kernel specializations can compile concurrently without interference
#[test]
fn different_keys_parallel_compile() {
    common::with_test_stack(|| {
        let barrier = Arc::new(std::sync::Barrier::new(2));

        let b1 = Arc::clone(&barrier);
        let h1 = std::thread::Builder::new()
            .stack_size(common::TEST_STACK_SIZE)
            .spawn(move || {
                b1.wait();
                let x = api::ones::<1, f32>([256]).sync().unwrap();
                let y = api::ones::<1, f32>([256]).sync().unwrap();
                let z = api::zeros::<1, f32>([256]).partition([8]).sync().unwrap();
                warmup_test_module::vector_add(z, x.into(), y.into())
                    .generics(vec!["f32".into(), "8".into()])
                    .sync()
                    .unwrap();
            })
            .unwrap();

        let b2 = Arc::clone(&barrier);
        let h2 = std::thread::Builder::new()
            .stack_size(common::TEST_STACK_SIZE)
            .spawn(move || {
                b2.wait();
                let x = api::ones::<1, f32>([512]).sync().unwrap();
                let y = api::ones::<1, f32>([512]).sync().unwrap();
                let z = api::zeros::<1, f32>([512]).partition([32]).sync().unwrap();
                warmup_test_module::vector_add(z, x.into(), y.into())
                    .generics(vec!["f32".into(), "32".into()])
                    .sync()
                    .unwrap();
            })
            .unwrap();

        h1.join().expect("thread 1 panicked");
        h2.join().expect("thread 2 panicked");

        // Verify both distinct keys are in cache.
        let device_id = get_default_device();
        let gpu_name = get_gpu_name(device_id);
        let cv = get_compiler_version();
        let tv = get_cuda_toolkit_version();
        let key_8 = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "8".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            gpu_name.clone(),
            cv.clone(),
            tv.clone(),
        );
        let key_32 = TileFunctionKey::new(
            "warmup_test_module".into(),
            "vector_add".into(),
            vec!["f32".into(), "32".into()],
            vector_add_stride_args(),
            None,
            CompileOptions::default(),
            warmup_test_module::__SOURCE_HASH.into(),
            gpu_name,
            cv,
            tv,
        );
        assert!(contains_cuda_function(device_id, &key_8));
        assert!(contains_cuda_function(device_id, &key_32));
    });
}

// Multi-thread dedup: verify OnceCell single-initialization via timing.
// If dedup works, wall time ≈ 1x single compile.
// If broken (each thread compiles independently), wall time ≈ N x single compile.
#[test]
fn multi_thread_dedup_timing_evidence() {
    common::with_test_stack(|| {
        // First, compile a different spec to estimate single-compile time.
        let t_single = std::time::Instant::now();
        warmup_test_module::__compile_warmup(&[WarmupSpec::new(
            "vector_add",
            vec!["f32".into(), "128".into()],
        )
        .with_strides(vector_add_stride_args())])
        .unwrap();
        let single_duration = t_single.elapsed();

        // Now race 4 threads on a FRESH spec (tile_size=4, not previously compiled).
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
                        let x = api::ones::<1, f32>([256]).sync().unwrap();
                        let y = api::ones::<1, f32>([256]).sync().unwrap();
                        let z = api::zeros::<1, f32>([256]).partition([4]).sync().unwrap();
                        warmup_test_module::vector_add(z, x.into(), y.into())
                            .generics(vec!["f32".into(), "4".into()])
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

        // If dedup works: parallel ≈ single. If broken: parallel ≈ 4 * single.
        // Use 2.5x as threshold — generous enough to avoid flakiness.
        let ratio = parallel_duration.as_secs_f64() / single_duration.as_secs_f64();
        eprintln!(
            "[dedup timing] single={:.1?}  parallel(4)={:.1?}  ratio={:.2}",
            single_duration, parallel_duration, ratio
        );
        assert!(
            ratio < 2.5,
            "parallel compile of 4 threads took {ratio:.2}x single — dedup may be broken \
             (single={single_duration:.1?}, parallel={parallel_duration:.1?})"
        );
    });
}

// CUTILE_NO_DISK_CACHE=1 disables disk persistence
#[test]
fn no_disk_cache_env_disables_persistence() {
    common::with_test_stack(|| {
        let cache_root = unique_temp_cache_dir("no_disk_cache");
        let _ = std::fs::remove_dir_all(&cache_root);

        let output = run_warmup_worker("no-disk-cache", &cache_root);
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success(),
            "no-disk-cache worker failed\n{}",
            combined
        );

        // The cache directory should either not exist or contain no .cubin files.
        let cache_dir = cache_root.join("cutile");
        let cubin_count = if cache_dir.exists() {
            std::fs::read_dir(&cache_dir)
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .ok()
                        .and_then(|x| x.path().extension().map(|ext| ext == "cubin"))
                        .unwrap_or(false)
                })
                .count()
        } else {
            0
        };
        assert_eq!(
            cubin_count, 0,
            "no cubin should be on disk when CUTILE_NO_DISK_CACHE=1\n{}",
            combined
        );

        let _ = std::fs::remove_dir_all(&cache_root);
    });
}

// Cross-process disk-hit integration
#[test]
fn cross_process_disk_hit_integration() {
    common::with_test_stack(|| {
        let cache_root = unique_temp_cache_dir("cross_process");
        let _ = std::fs::remove_dir_all(&cache_root);

        // Process A: produce persisted cubin(s).
        let producer = run_warmup_worker("producer", &cache_root);
        assert!(
            producer.status.success(),
            "producer failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&producer.stdout),
            String::from_utf8_lossy(&producer.stderr)
        );

        let cache_dir = cache_root.join("cutile");
        let cubin_count = std::fs::read_dir(&cache_dir)
            .expect("cache dir should exist after producer")
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|x| x.path().extension().map(|ext| ext == "cubin"))
                    .unwrap_or(false)
            })
            .count();
        assert!(cubin_count > 0, "producer should persist at least one cubin");

        // Process B: first request should be disk-hit (new process memory cache is empty).
        let consumer = run_warmup_worker("consumer", &cache_root);
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&consumer.stdout),
            String::from_utf8_lossy(&consumer.stderr)
        );
        assert!(
            consumer.status.success(),
            "consumer failed\n{}",
            combined
        );
        assert!(
            combined.contains("disk cache hit"),
            "consumer should report disk cache hit\n{}",
            combined
        );

        let _ = std::fs::remove_dir_all(&cache_root);
    });
}

// Multi-spec A=disk-hit, B=cold-compile 
#[test]
fn multi_spec_disk_hit_then_cold_compile_integration() {
    common::with_test_stack(|| {
        let cache_root = unique_temp_cache_dir("multi_spec");
        let _ = std::fs::remove_dir_all(&cache_root);

        let output = run_warmup_worker("multi-spec", &cache_root);
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "worker failed\n{}", combined);
        assert!(
            combined.contains("disk cache hit"),
            "expected spec A disk-hit\n{}",
            combined
        );
        assert!(
            combined.contains("JIT compiled"),
            "expected spec B cold-compile\n{}",
            combined
        );

        let _ = std::fs::remove_dir_all(&cache_root);
    });
}

// Worker test used by cross-process integration tests above.
// Runs as no-op unless CUTILE_WARMUP_WORKER_ROLE is set.
#[test]
fn cross_process_warmup_worker() {
    let Ok(role) = std::env::var("CUTILE_WARMUP_WORKER_ROLE") else {
        return;
    };

    common::with_test_stack(move || {
        let spec_a = WarmupSpec::new("vector_add", vec!["f32".into(), "64".into()])
            .with_strides(vector_add_stride_args());
        let spec_b = WarmupSpec::new("vector_add", vec!["f32".into(), "32".into()])
            .with_strides(vector_add_stride_args());

        match role.as_str() {
            "producer" => {
                warmup_test_module::__compile_warmup(std::slice::from_ref(&spec_a))
                    .expect("producer warmup failed");
            }
            "consumer" => {
                warmup_test_module::__compile_warmup(std::slice::from_ref(&spec_a))
                    .expect("consumer warmup failed");
            }
            "multi-spec" => {
                // Step 1: compile A and ensure it's persisted.
                warmup_test_module::__compile_warmup(std::slice::from_ref(&spec_a))
                    .expect("initial warmup for spec A failed");

                // Step 2: evict A from memory so A path is forced to disk-hit.
                let device_id = get_default_device();
                let key_a = TileFunctionKey::new(
                    "warmup_test_module".into(),
                    "vector_add".into(),
                    vec!["f32".into(), "64".into()],
                    vector_add_stride_args(),
                    None,
                    CompileOptions::default(),
                    warmup_test_module::__SOURCE_HASH.into(),
                    get_gpu_name(device_id),
                    get_compiler_version(),
                    get_cuda_toolkit_version(),
                );
                get_kernel_cache().remove(&key_a.get_hash_string());

                // Step 3: A should disk-hit, B should cold-compile.
                warmup_test_module::__compile_warmup(&[spec_a.clone(), spec_b.clone()])
                    .expect("multi-spec warmup failed");

                // Verify both A and B are present in memory cache.
                let gpu_name = get_gpu_name(device_id);
                let compiler_version = get_compiler_version();
                let cuda_toolkit_version = get_cuda_toolkit_version();
                let key_b = TileFunctionKey::new(
                    "warmup_test_module".into(),
                    "vector_add".into(),
                    vec!["f32".into(), "32".into()],
                    vector_add_stride_args(),
                    None,
                    CompileOptions::default(),
                    warmup_test_module::__SOURCE_HASH.into(),
                    gpu_name.clone(),
                    compiler_version.clone(),
                    cuda_toolkit_version.clone(),
                );
                let key_a_after = TileFunctionKey::new(
                    "warmup_test_module".into(),
                    "vector_add".into(),
                    vec!["f32".into(), "64".into()],
                    vector_add_stride_args(),
                    None,
                    CompileOptions::default(),
                    warmup_test_module::__SOURCE_HASH.into(),
                    gpu_name,
                    compiler_version,
                    cuda_toolkit_version,
                );
                assert!(contains_cuda_function(device_id, &key_a_after));
                assert!(contains_cuda_function(device_id, &key_b));
            }
            "no-disk-cache" => {
                // Compile a kernel with CUTILE_NO_DISK_CACHE=1 (set by the parent process).
                // The parent asserts that no .cubin files appear on disk afterward.
                warmup_test_module::__compile_warmup(std::slice::from_ref(&spec_a))
                    .expect("no-disk-cache warmup failed");
            }
            other => panic!("unknown worker role: {other}"),
        }
    });
}
