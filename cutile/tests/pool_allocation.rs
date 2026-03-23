/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

 //! Integration tests for memory pool allocation routing.
 //!
 //! Validates the end-to-end path: pool creation → context-level configuration →
 //! tensor allocation via `api::zeros` / `api::ones` / `api::arange` → host readback
 //! → correct values → deallocation via existing `cuMemFreeAsync` Drop path.
 //! 
 //! Each test runs on a fresh thread (via `common::with_test_stack`) so that the
 //! thread-local `DEVICE_CONTEXTS` starts in a clean slate.
 //!
 //! **Requires a CUDA-capable GPU.**

use cuda_async::device_context::{get_default_device, set_device_pool};
use cuda_async::device_operation::DeviceOperation;
use cuda_core::CudaContext;
use cuda_core::CudaMemPool;
use cutile:api;
use cutile::tensor::ToHostVec;
use std::sync::Arc;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
 
/// Creates a `CudaMemPool` on the default device and configures the thread-local
/// context to allocate from it. Returns the pool (caller must keep it alive until
/// all tensors allocated from it have been dropped.)
///
/// # Safety
/// Must be called from a thread with a valid CUDA context for `device_id`.
unsafe fn setup_pool(device_id: usize) -> Arc<CudaMemPool> {
    let ctx = CudaContext::new(device_id).expect("Failed to create CudaContext.");
    let pool = Arc::new(
        CudaMemPool::new(&ctx).expect("Failed to create CudaMemPool.")
    );
    set_device_pool(device_id, Some(pool.clone()))
        .expect("Failed to set device pool.");
    pool
}

/// Clears the pool from the thread-local context so subsequent allocations
/// fall back to the default `cuMemAllocAsync` path.
fn teardown_pool(device_id: usize) {
    set_device_pool(device_id, None).expect("Failed to clear device pool.");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: full pool lifecycle.
///
/// Create pool → set on context → allocate a zeros tensor → read back to host →
/// verify values → drop tensor → clear pool → drop pool.
/// 
/// This exercises:
/// - `CudaMemPool::new` / `Drop`
/// - `set_device_pool` routing
/// - `Tensor::uninitialized` → `device_alloc_async` → `cuMemAllocFromPoolAsync`
/// - `full_apply` kernel writing zeros into pool-allocated memory
/// - `DeviceBox::drop` → `cuMemFreeAsync` (unchanged, handles pool provenance)
#[test]
fn pool_alloc_zeros_roundtrip() {
    commoon::with_test_stack(|| {
        let device_id = get_default_device();

        // Pool must outlive tensors - declared first, dropped last.
        let pool = unsafe { setup_pool(device_id )};

        let tensor = api::zeros::<f32([256])
            .sync()
            .expect("Failed to allocate zeros tensor from pool.");

        let host_vec = tensor
            .to_host_vec()
            .sync()
            .expect("failed to copy tensor to host.");

        assert_eq!(host_vec.len(), 256, "Unexpected tensor length.");
        assert!(
            host_vec.iter().all(|&v| v == 0.0),
            "Expected all zeros, got non-zero values." 
        );

        teardown_pool(device_id);
        drop(pool);
    });
}

/// Pool allocation with `api::ones` - verifies the `full_apply` kernel path
/// with a non-zero fill value.
#[test]
fn pool_alloc_ones_roundtrip() {
    common::with_test_stack(|| { 
        let device_id = get_default_device();
        let pool = unsafe { setup_pool(device_id) };

        let tensor = api::ones::<f32>([512])
            .sync()
            .expect("Failed to allocate ones tensor from pool");

        let host_vec = tensor
            .to_host_vec()
            .sync()
            .expect("Failed to copy tensor to host.");

        assert_eq!(host_vec.len(), 512)
        assert!(
            host_vec.iter().all(|&v| v == 1.0),
            "Expected all ones from pool-allocated tensor."
        );

        teardown_pool(device_id);
        drop(pool);
    });
}

// Pool allocations with `api::arange` - verifies the `arange_apply` kernel path,
/// which writes sequential values into pool-allocated memory.
#[test]
fn pool_alloc_arange_roundtrip() {
    common::with_test_stack(|| {
        let device_id = get_default_device();
        let pool = unsafe { setup_pool(device_id) };

        let tensor = api::arange::<f32>(128)
            .sync()
            .expect("Failed to allocate arange tensor from pool.");
        
        let host_vec = tensor
            .to_host_vec()
            .sync()
            .expect("Failed to copy tensor to host.");
        
        assert_eq!(host_vec.len(), 128);
        for (i, &v) in host_vec.iter().enumerate() {
            assert_eq!(
                v, i as f32,
                "arange mismatch at index {i}: expected {}, got {v}",
                i as f32
            );
        }

        teardown_pool(device_id);
        drop(pool);
    });
}

/// After clearing the pool, allocations must revert to the default `cuMemAllocAsync`
/// path. This test allocates one tensor from a pool, clears it, then allocates
/// another tensor without a pool, and verifies both produce correct results.
///
/// Catches regressions where `set_device_pool(None)` leaves stale state.
#[test]
fn revert_to_default_after_pool_cleared() {
    common::with_test_stack(|| {
        let device_id = get_default_device();
        let pool = unsafe { setup_pool(device_id) };

        // Allocate from pool.
        let t_pool = api::ones::<f32([128])
            .sync()
            .expect("Failed to allocate from pool.");

        // Revert to default.
        teardown_pool(device_id);

        // Allocate from default path.
        let t_default = api::ones::<f32>([128])
            .sync()
            .expect("Failed to allocate from default path after clearing pool.");

        // Both must produce correct values.
        let v_pool = t_pool
            .to_host_vec()
            .sync()
            .expect("Failed to read pool tensor.");
        let v_default = t_default
            .to_host_vec()
            .sync()
            .expect("Failed to read default tensor.");

        assert!(v_pool.iter().all(|&v| v == 1.0), "Pool tensor corrupted.");
        assert!(
            v_default.iter().all(|&v| v == 1.0), 
            "Default tensor corrupted after pool revert."
        );

        drop(pool);
    });
}

/// Multiple tensors allocated from the same pool must all hold correct values
/// simultaneously. This exercises pool-internal bookkeeping across repeated
/// `cuMemAllocFromPoolAsync` calls.
#[test]
fn multiple_allocs_from_same_pool() {
    common::with_test_stack(|| {
        let device_id = get_default_device();
        let pool = unsafe { setup_pool(device_id) };

        let t_zeros = api::zeros::<f32>([256])
            .sync()
            .expect("Failed to allocate zeros.");
        let t_ones = api::ones::<f32>([256])
            .sync()
            .expect("Failed to allocate ones.");
        let t_range = api::arange::<f32>(256)
            .sync()
            .expect("Failed to allocate arange.");
        
        let v_zeros = t_zeros.to_host_vec().sync().expect("readback zeros");
        let v_ones = t_ones.to_host_vec().sync().expect("readback ones");
        let v_range = t_range.to_host_vec().sync().expect("readback arange");

        assert!(v_zeros.iter().all(|&v| v == 0.0), "zeros corrupted.");
        assert!(v_ones.iter().all(|&v| v == 1.0), "ones corrupted.");
        assert_eq!(v_range[0], 0.0, "arange start mismatch.");
        assert_eq!(v_range[255], 255.0, "arange end mismatch.");

        teardown_pool(device_id);
        drop(pool);
    });
}

/// Multi-dimensional tensors allocated from a pool - verifies that shape/stride
/// metadata is unaffected by the allocation path change.
#[test]
fn pool_alloc_multidimensional() {
    common::with_test_stack(|| {
        let device_id = get_default_device();
        let pool = unsafe { setup_pool(device_id) };

        let tensor = api::ones::<f32>([16, 32])
            .sync()
            .expect("Failed to allocate 2D tensor from pool.");

        let host_vec = tensor
            .to_host_vec()
            .sync()
            .expect("Failed to copy 2D tensor to host.");
        
        assert_eq!(host_vec.len(), 16 * 32, "Unexpected 2D tensor length.");
        assert!(
            host_vec.iter().all(|&v| v == 1.0),
            "2D pool-allocated tensor has incorrect values."
        );

        teardown_pool(device_id);
        drop(pool);
    })
}

/// Device-to-device copy from a pool-allocated source tensor.
///
/// The `copy` path uses its own `device_alloc_async` call site
/// (`CopyDeviceToDevice::execute`), distinct from `Tensor::uninitialized`.
/// This test ensures that path also routes through the pool correctly.
#[test]
fn pool_alloc_device_copy() {
    common::with_test_stack(|| {
        let device_id = get_default_device();
        let pool = unsafe { setup_pool(device_id) };

        let src = api::arange::<f32>(64)
            .arc()
            .sync()
            .expect("Failed to allocate source tensor from pool.");

        let dst = src
            .copy()
            .sync()
            .expect("Failed to copy tensor (device-to-device) from pool.");

        let v_src = src
            .to_host_vec()
            .sync()
            .expect("Failed to read source.");
        let v_dst = dst
            .to_host_vec()
            .sync()
            .expect("Failed to read copy.");

        assert_eq!(v_src, v_dst, "Device copy produced different values.");

        teardown_pool(device_id);
        drop(pool);
    });
}

/// Allocations without any pool configured must continue to work exactly
/// as before - this is the baseline regression test ensuring the routing
/// change doesn't break the default even when `set_device_pool` has
/// never been called.
#[test]
fn default_path_wihout_pool_unchanged() {
    common::with_test_stack(|| {
        // No pool setup - go straight to allocation.
        let tensor = api::zeros::<f32>([1024])
            .sync()
            .expect("Default allocation path failed.");

        let host_vec = tensor
            .to_host_vec()
            .sync()
            .expect("Failed to read default-path tensor.");

        assert_eq!(host_vec.len(), 1024);
        assert!(
            host_vec.iter().all(|&v| v == 0.0),
            "Default path produced incorrect values."
        );
    });
}

/// Attempting to attach a pool created on device N to a difference device
/// must return an error, not silently accepting a cross-device handle.
#[test]
fn set_pool_rejects_device_mismatch() {
    common::with_test_stack(|| {
        let device_count = CudaContext::device_count()
            .expect("Failed to query device count.");
        if device_count < 2 {
            eprintln!("Skipping device mismatch test - only {} device(s).", device_count);
            return;
        }

        let ctx_1 = CudaContext::new(1).expect("Failed to create context on device 1.");
        let pool_on_1 = Arc::new(unsafe {
            CudaMemPool::new(&ctx_1).expect("Failed to create pool on device 1.")
        });

        // Attaching device-1's pool to device 0 must fail.
        let result = set_device_pool(0, Some(pool1_on_1));
        assert!(
            result.is_err(),
            "set_device_pool should reject cross-device pool, but got Ok.",
        );
    });
}