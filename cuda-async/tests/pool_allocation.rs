/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for custom memory pool allocation.
//!
//! Tests that exercise the `CudaMemPool` lifecycle, device-level pool
//! configuration, and pool-aware allocation through `ExecutionContext`.
//!
//! Each test runs on a fresh thread so that thread-local `DEVICE_CONTEXTS`
//! starts clean.

use cuda_async::device_context::{
    clear_device_pool, get_device_pool, global_policy, init_device_contexts, set_device_pool,
    with_cuda_context,
};
use cuda_async::device_operation::{value, DeviceOp};
use cuda_async::prelude::*;

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

// ---------------------------------------------------------------------------
// CudaMemPool RAII lifecycle
// ---------------------------------------------------------------------------

#[test]
fn create_and_drop_mem_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        assert!(!pool.cu_pool().is_null());
    });
}

#[test]
fn default_mem_pool_is_not_owned() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_cuda_context(0, |ctx| ctx.default_mem_pool())
            .expect("get context failed")
            .expect("default pool failed");

        assert!(!pool.cu_pool().is_null());
    });
}

#[test]
fn set_release_threshold() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");

        pool.set_release_threshold(1024 * 1024)
            .expect("set finite threshold failed");
    });
}

// ---------------------------------------------------------------------------
// Device-level pool configuration
// ---------------------------------------------------------------------------

#[test]
fn set_and_get_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool_opt = get_device_pool(0).expect("get pool failed");
        assert!(pool_opt.is_none());

        let pool = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        let pool_ptr = pool.cu_pool();
        set_device_pool(0, pool).expect("set pool failed");

        let retrieved = get_device_pool(0)
            .expect("get pool failed")
            .expect("pool should be set");
        assert_eq!(retrieved.cu_pool(), pool_ptr);
    });
}

#[test]
fn clear_device_pool_reverts_to_none() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        set_device_pool(0, pool).expect("set pool failed");

        assert!(get_device_pool(0).expect("get failed").is_some());

        clear_device_pool(0).expect("clear pool failed");
        assert!(get_device_pool(0).expect("get failed").is_none());
    });
}

// ---------------------------------------------------------------------------
// Pool-aware allocation through DeviceOp
// ---------------------------------------------------------------------------

#[test]
fn alloc_with_custom_pool_via_device_op() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        set_device_pool(0, pool).expect("set pool failed");

        let op = with_context(|ctx| {
            let num_bytes = 1024;
            let dptr = unsafe { ctx.alloc_async(num_bytes) };
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        });
        let dptr = op.sync().expect("device op failed");
        assert!(dptr != 0);
    });
}

#[test]
fn alloc_without_pool_uses_default() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let op = with_context(|ctx| {
            assert!(ctx.get_pool().is_none());
            let num_bytes = 1024;
            let dptr = unsafe { ctx.alloc_async(num_bytes) };
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        });
        let dptr = op.sync().expect("device op failed");
        assert!(dptr != 0);
    });
}

#[test]
fn pool_is_frozen_at_scheduling_time() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool_a = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool A creation failed");
        let pool_b = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool B creation failed");

        let pool_a_ptr = pool_a.cu_pool() as usize;

        set_device_pool(0, pool_a).expect("set pool_a failed");

        // Schedule while pool_a is active — freezes pool_a into ExecutionContext.
        let policy = global_policy(0).expect("get policy failed");
        let future = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present");
            assert_eq!(p.cu_pool() as usize, pool_a_ptr, "should use frozen pool_a, not pool_b");
            value(())
        })
        .schedule(&policy)
        .expect("schedule failed");

        // Change global pool AFTER scheduling — must not affect the already-frozen ExecutionContext.
        set_device_pool(0, pool_b).expect("set pool_b failed");

        // Execute: DeviceFuture carries pool_a in its ExecutionContext, pool_b is ignored.
        futures::executor::block_on(future).expect("future failed");
    });
}

// ---------------------------------------------------------------------------
// Explicit .schedule() path
// ---------------------------------------------------------------------------

#[test]
fn schedule_applies_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let policy = global_policy(0).expect("get policy failed");
        let future = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present via schedule");
            assert_eq!(p.cu_pool() as usize, pool_ptr, "schedule must pick up device pool");
            let dptr = unsafe { ctx.alloc_async(512) };
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        })
        .schedule(&policy)
        .expect("schedule failed");

        let dptr = futures::executor::block_on(future).expect("future failed");
        assert!(dptr != 0);
    });
}

// ---------------------------------------------------------------------------
// Multiple pools
// ---------------------------------------------------------------------------

#[test]
fn switch_between_pools() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool_a = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool A creation failed");
        let pool_b = with_cuda_context(0, |ctx| ctx.new_mem_pool())
            .expect("get context failed")
            .expect("pool B creation failed");

        let pool_a_ptr = pool_a.cu_pool();
        let pool_b_ptr = pool_b.cu_pool();
        assert_ne!(pool_a_ptr, pool_b_ptr, "pools should be distinct");

        set_device_pool(0, pool_a).expect("set A failed");
        let op_a = with_context(|ctx| {
            let dptr = unsafe { ctx.alloc_async(512) };
            assert!(dptr != 0);
            value(())
        });
        op_a.sync().expect("op A failed");

        set_device_pool(0, pool_b).expect("set B failed");
        let op_b = with_context(|ctx| {
            let dptr = unsafe { ctx.alloc_async(512) };
            assert!(dptr != 0);
            value(())
        });
        op_b.sync().expect("op B failed");

        clear_device_pool(0).expect("clear failed");
        let op_default = with_context(|ctx| {
            assert!(ctx.get_pool().is_none());
            let dptr = unsafe { ctx.alloc_async(512) };
            assert!(dptr != 0);
            value(())
        });
        op_default.sync().expect("default op failed");
    });
}
