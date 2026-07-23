/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for custom memory pool allocation.
//!
//! Tests that exercise the `MemPool` lifecycle, device-level pool
//! configuration, and pool-aware allocation through `ExecutionContext`.
//!
//! Each test runs on a fresh thread so that thread-local `DEVICE_CONTEXTS`
//! starts clean.

use cuda_async::device_buffer::DeviceBuffer;
use cuda_async::device_context::{
    clear_device_poison, clear_device_pool, get_device_pool, global_policy, init_device_contexts,
    is_device_poisoned, pool_for_stream, reset_device, set_device_pool, with_device,
    with_global_device_context, with_global_device_context_mut,
};
use cuda_async::device_operation::{value, DeviceOp};
use cuda_async::prelude::*;
use std::future::IntoFuture;

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

// ---------------------------------------------------------------------------
// MemPool RAII lifecycle
// ---------------------------------------------------------------------------

#[test]
fn create_and_drop_mem_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        assert!(!pool.cu_pool().is_null());
    });
}

#[test]
fn default_mem_pool_is_not_owned() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.default_mem_pool())
            .expect("get context failed")
            .expect("default pool failed");

        assert!(!pool.cu_pool().is_null());
    });
}

#[test]
fn set_release_threshold() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
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

        let pool = with_device(0, |device| device.new_mem_pool())
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

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        set_device_pool(0, pool).expect("set pool failed");

        assert!(get_device_pool(0).expect("get failed").is_some());

        clear_device_pool(0).expect("clear pool failed");
        assert!(get_device_pool(0).expect("get failed").is_none());
    });
}

#[test]
fn set_device_pool_rejects_cross_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        let err = set_device_pool(99, pool).expect_err("expected cross-device pool to be rejected");
        match err {
            cuda_async::error::DeviceError::Context { device_id, message } => {
                assert_eq!(device_id, 99, "error should point to target device");
                assert!(
                    message.contains("pool belongs to device 0")
                        && message.contains("expected device 99"),
                    "message should name both devices, got: {message}"
                );
            }
            other => panic!("expected DeviceError::Context, got {other:?}"),
        }

        assert!(get_device_pool(0).expect("get pool failed").is_none());
    });
}

// ---------------------------------------------------------------------------
// Pool-aware allocation through DeviceOp
// ---------------------------------------------------------------------------

#[test]
fn alloc_with_custom_pool_via_device_op() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let op = with_context(move |ctx| {
            let p = ctx
                .get_pool()
                .expect("pool must be present in ExecutionContext via sync()");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "sync() must route allocation through the registered pool"
            );
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
fn alloc_with_custom_pool_via_into_future() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let op = with_context(move |ctx| {
            let p = ctx
                .get_pool()
                .expect("pool must be present in ExecutionContext via into_future()");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "into_future() must route allocation through the registered pool"
            );
            value(())
        });
        futures::executor::block_on(op.into_future()).expect("future failed");
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

        let pool_a = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool A creation failed");
        let pool_b = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool B creation failed");

        let pool_a_ptr = pool_a.cu_pool() as usize;

        set_device_pool(0, pool_a).expect("set pool_a failed");

        // Schedule while pool_a is active — freezes pool_a into ExecutionContext.
        let policy = global_policy(0).expect("get policy failed");
        let future = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present");
            assert_eq!(
                p.cu_pool() as usize,
                pool_a_ptr,
                "should use frozen pool_a, not pool_b"
            );
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

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let policy = global_policy(0).expect("get policy failed");
        let future = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present via schedule");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "schedule must pick up device pool"
            );
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

#[test]
fn sync_on_applies_device_pool() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");
        let pool_ptr = pool.cu_pool() as usize;
        set_device_pool(0, pool).expect("set pool failed");

        let stream = global_policy(0)
            .expect("get policy failed")
            .next_stream()
            .expect("get stream failed");

        let dptr = with_context(move |ctx| {
            let p = ctx.get_pool().expect("pool should be present via sync_on");
            assert_eq!(
                p.cu_pool() as usize,
                pool_ptr,
                "sync_on must pick up device pool"
            );
            let dptr = unsafe { ctx.alloc_async(512) };
            assert!(dptr != 0, "allocation returned null pointer");
            value(dptr)
        })
        .sync_on(&stream)
        .expect("sync_on failed");
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

        let pool_a = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool A creation failed");
        let pool_b = with_device(0, |device| device.new_mem_pool())
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

// ---------------------------------------------------------------------------
// Re-entry detection (explicit-enum safety net)
// ---------------------------------------------------------------------------

#[test]
fn re_entry_to_with_global_device_context_panics() {
    let result = std::thread::spawn(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let _ = with_global_device_context(0, |_outer| {
            // Re-entry from inside the callback should panic.
            let _ = with_global_device_context(0, |_inner| {});
        });
    })
    .join();

    let payload = result.expect_err("expected re-entrant access to panic");
    let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .expect("panic payload should be a string");
    assert!(
        msg.contains("re-entrant access"),
        "panic message should name re-entrancy, got: {msg}"
    );
}

#[test]
fn mut_callback_panic_returns_err_on_subsequent_access() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context_mut(0, |_ctx| {
                panic!("simulated mutation failure for poisoning test");
            });
        }));
        assert!(panic_result.is_err(), "test setup: expected inner panic");

        assert!(
            is_device_poisoned(0).expect("query failed"),
            "device should be poisoned after a panicking _mut callback"
        );

        let read_err = with_global_device_context(0, |_| ())
            .expect_err("read-only access to poisoned device should be Err");
        match read_err {
            cuda_async::error::DeviceError::Context { device_id, message } => {
                assert_eq!(device_id, 0);
                assert!(
                    message.contains("poisoned"),
                    "error message should name poisoning, got: {message}"
                );
            }
            other => panic!("expected DeviceError::Context, got {other:?}"),
        }

        let mut_err = with_global_device_context_mut(0, |_| ())
            .expect_err("_mut access to poisoned device should be Err");
        assert!(matches!(
            mut_err,
            cuda_async::error::DeviceError::Context { device_id: 0, .. }
        ));
    });
}

// The no-nesting contract must also hold for transitive re-borrowers:
// pool_for_stream → get_device_pool → with_global_device_context.
#[test]
fn nested_pool_lookup_panics_loudly() {
    let result = std::thread::spawn(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");
        let stream = global_policy(0)
            .expect("setup: global_policy")
            .next_stream()
            .expect("setup: next_stream");
        let _ = with_global_device_context(0, |_outer| {
            let _ = pool_for_stream(&stream); // transitive re-entry
        });
    })
    .join();

    let payload = result.expect_err("expected nested pool_for_stream to panic");
    let msg = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .expect("panic payload should be a string");
    assert!(
        !msg.contains("setup:"),
        "test environment problem (not a contract failure): {msg}"
    );
    assert!(
        msg.contains("re-entrant access"),
        "re-entry contract broken — expected re-entrant panic, got: {msg}"
    );
}

// A _mut callback panic must not leave the cell stuck in Borrowed: the
// guard's Drop runs during unwinding and restores Available, so the thread
// stays usable (poison Err, not a re-entry panic) and recovers fully after
// clear_device_poison.
#[test]
fn panic_recovery_restores_context_state() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context_mut(0, |_ctx| panic!("poison"));
        }));
        assert!(panicked.is_err(), "setup: expected inner panic");

        // Stuck `Borrowed` would panic here instead of returning poison Err.
        let err = with_global_device_context(0, |_| ())
            .expect_err("poisoned device should return Err, not panic on Borrowed");
        assert!(matches!(
            err,
            cuda_async::error::DeviceError::Context { device_id: 0, .. }
        ));

        clear_device_poison(0).expect("clear failed");
        with_global_device_context(0, |_| ())
            .expect("context must be usable after panic recovery + clear");
    });
}

// The kernel cache is process-global and never reads the per-thread device
// context, so it has no poison to propagate; only pool lookups do.
#[test]
fn pool_lookups_propagate_poison_error() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");
        let stream = global_policy(0)
            .expect("setup: global_policy")
            .next_stream()
            .expect("setup: next_stream");

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context_mut(0, |_ctx| panic!("poison"));
        }));
        assert!(panicked.is_err(), "setup: expected inner panic");
        assert!(is_device_poisoned(0).expect("query failed"));

        let pool_res = pool_for_stream(&stream);
        assert!(
            matches!(
                pool_res,
                Err(cuda_async::error::DeviceError::Context { device_id: 0, .. })
            ),
            "pool_for_stream must propagate the poison error, got: {pool_res:?}"
        );
    });
}

// Freeing memory must keep working on a poisoned device: buffers are
// dropped during the very unwind that sets the poison, so a poison Err
// (and the panic it used to cause inside `Drop`) would abort the process.
#[test]
fn buffer_drop_on_poisoned_device_frees_without_panicking() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");
        let stream = global_policy(0)
            .expect("setup: global_policy")
            .next_stream()
            .expect("setup: next_stream");
        let dptr = unsafe { cuda_core::malloc_async(256, &stream) };
        let buffer = unsafe { DeviceBuffer::from_raw_parts(dptr, 256, 0) };

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context_mut(0, |_ctx| panic!("poison"));
        }));
        assert!(panicked.is_err(), "setup: expected inner panic");
        assert!(is_device_poisoned(0).expect("query failed"));

        // Frees via the deallocator stream; poison stays set for regular
        // access until cleared.
        drop(buffer);
        assert!(is_device_poisoned(0).expect("query failed"));
    });
}

#[test]
fn buffer_dropped_during_poisoning_unwind_does_not_abort() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");
        let stream = global_policy(0)
            .expect("setup: global_policy")
            .next_stream()
            .expect("setup: next_stream");
        let dptr = unsafe { cuda_core::malloc_async(256, &stream) };
        let buffer = unsafe { DeviceBuffer::from_raw_parts(dptr, 256, 0) };

        // `buffer` lives inside the panicking closure: its Drop runs while the
        // callback panic is unwinding, after the ContextGuard has already
        // poisoned device 0. A panic from that Drop would be a double panic —
        // SIGABRT, uncatchable, killing the whole test binary.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _held = buffer;
            let _ = with_global_device_context_mut(0, |_ctx| panic!("poison"));
        }));
        assert!(
            panicked.is_err(),
            "expected the callback panic to propagate"
        );
        assert!(is_device_poisoned(0).expect("query failed"));
    });
}

// Dropping a buffer inside a `with_*` callback re-borrows the thread-local
// map from `Drop`. That must neither panic (the re-entry rule would fire
// inside a destructor) nor leak: the free is deferred and flushed when the
// callback's borrow ends.
#[test]
fn buffer_dropped_inside_callback_defers_the_free() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");
        let stream = global_policy(0)
            .expect("setup: global_policy")
            .next_stream()
            .expect("setup: next_stream");
        let dptr = unsafe { cuda_core::malloc_async(256, &stream) };
        let buffer = unsafe { DeviceBuffer::from_raw_parts(dptr, 256, 0) };

        with_global_device_context(0, move |_ctx| {
            drop(buffer); // re-entrant Drop: deferred, not a panic
        })
        .expect("callback failed");

        // The deferred free was flushed on exit; the map is intact.
        with_global_device_context(0, |_ctx| ())
            .expect("context must be usable after a deferred free");
    });
}

// Combined case: the buffer drops inside a `_mut` callback that then
// panics. The deferred free must be flushed by the same guard Drop that
// poisons the device, without a second panic.
#[test]
fn buffer_dropped_inside_panicking_mut_callback_is_freed_not_aborted() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("setup: init_device_contexts (requires GPU)");
        let stream = global_policy(0)
            .expect("setup: global_policy")
            .next_stream()
            .expect("setup: next_stream");
        let dptr = unsafe { cuda_core::malloc_async(256, &stream) };
        let buffer = unsafe { DeviceBuffer::from_raw_parts(dptr, 256, 0) };

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ = with_global_device_context_mut(0, move |_ctx| {
                let _held = buffer; // dropped during the callback's unwind
                panic!("poison");
            });
        }));
        assert!(
            panicked.is_err(),
            "expected the callback panic to propagate"
        );
        assert!(is_device_poisoned(0).expect("query failed"));
    });
}

#[test]
fn clear_device_poison_recovers_existing_context() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        let pool_ptr = pool.cu_pool();
        set_device_pool(0, pool).expect("set pool failed");

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context_mut(0, |_ctx| panic!("oops"));
        }));
        assert!(is_device_poisoned(0).expect("query failed"));

        clear_device_poison(0).expect("clear failed");
        assert!(!is_device_poisoned(0).expect("query failed"));

        let pool = get_device_pool(0)
            .expect("get pool failed")
            .expect("pool should survive clear_device_poison");
        assert_eq!(
            pool.cu_pool(),
            pool_ptr,
            "clear_device_poison must not discard the existing context's state"
        );
    });
}

#[test]
fn reset_device_drops_and_rebuilds_on_next_access() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        set_device_pool(0, pool).expect("set pool failed");
        assert!(get_device_pool(0).expect("get failed").is_some());

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context_mut(0, |_ctx| panic!("oops"));
        }));
        assert!(is_device_poisoned(0).expect("query failed"));

        reset_device(0).expect("reset failed");

        assert!(!is_device_poisoned(0).expect("query failed"));
        with_global_device_context(0, |_ctx| ())
            .expect("read-only access after reset_device should succeed");
        assert!(
            get_device_pool(0).expect("get failed").is_none(),
            "reset_device must drop the previously registered pool"
        );
    });
}

#[test]
fn successful_mut_callback_does_not_poison() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        with_global_device_context_mut(0, |_ctx| ()).expect("mut access failed");
        assert!(
            !is_device_poisoned(0).expect("query failed"),
            "successful _mut callback must leave the device unpoisoned"
        );
    });
}

// The poison query/recovery fns are documented as pure no-ops on an
// uninitialized thread: they must not lazily initialize the context map
// (this test intentionally skips init_device_contexts and needs no GPU).
#[test]
fn poison_queries_on_uninitialized_thread_are_pure_no_ops() {
    on_fresh_thread(|| {
        assert!(!is_device_poisoned(0).expect("query must succeed uninitialized"));
        clear_device_poison(0).expect("clear must be a no-op uninitialized");
        reset_device(0).expect("reset must be a no-op uninitialized");
        assert!(!is_device_poisoned(0).expect("query must still succeed"));
    });
}

#[test]
fn is_device_poisoned_returns_false_for_uninitialized_entry() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        // Device 99 was never initialized: not poisoned, by definition.
        assert!(!is_device_poisoned(99).expect("query failed"));
    });
}

#[test]
fn read_only_callback_panic_does_not_poison_context() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_global_device_context(0, |_ctx| {
                panic!("simulated panic in read-only callback");
            });
        }));
        assert!(panic_result.is_err(), "test setup: expected inner panic");

        // Subsequent read-only access should still succeed — no poisoning.
        with_global_device_context(0, |_ctx| ())
            .expect("read-only access after read-only panic should succeed");
    });
}

// ---------------------------------------------------------------------------
// Memory accounting
// ---------------------------------------------------------------------------

#[test]
fn mem_stats_initially_zero() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");

        let stats = pool.mem_stats().expect("read stats failed");
        assert_eq!(stats.used_current, 0, "no allocations yet");
        assert_eq!(stats.used_high, 0, "no allocations yet");
    });
}

#[test]
fn mem_stats_track_alloc_and_free() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");

        let stream = global_policy(0)
            .expect("get policy failed")
            .next_stream()
            .expect("get stream failed");

        const N: usize = 1 << 20;

        let dptr = unsafe { cuda_core::malloc_from_pool_async(N, &pool, &stream) };
        assert!(dptr != 0);
        unsafe { stream.synchronize() }.expect("stream sync failed");

        let stats = pool.mem_stats().expect("read stats failed");
        assert!(
            stats.used_current >= N as u64,
            "used_current should reflect alloc, got {}",
            stats.used_current
        );
        assert!(
            stats.used_high >= N as u64,
            "used_high should track peak, got {}",
            stats.used_high
        );

        unsafe { cuda_core::free_async(dptr, &stream) };
        unsafe { stream.synchronize() }.expect("stream sync failed");

        let stats = pool.mem_stats().expect("read stats failed");
        assert_eq!(
            stats.used_current, 0,
            "used_current should return to 0 after free + sync"
        );
        assert!(
            stats.used_high >= N as u64,
            "used_high is a watermark and must not decrease, got {}",
            stats.used_high
        );
    });
}

#[test]
fn reset_used_high_collapses_watermark_to_current() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let pool = with_device(0, |device| device.new_mem_pool())
            .expect("get context failed")
            .expect("pool creation failed");
        pool.set_release_threshold(u64::MAX)
            .expect("set threshold failed");

        let stream = global_policy(0)
            .expect("get policy failed")
            .next_stream()
            .expect("get stream failed");

        const N: usize = 1 << 20;

        let dptr = unsafe { cuda_core::malloc_from_pool_async(N, &pool, &stream) };
        unsafe { cuda_core::free_async(dptr, &stream) };
        unsafe { stream.synchronize() }.expect("stream sync failed");

        let before = pool.mem_stats().expect("read stats failed");
        assert_eq!(before.used_current, 0);
        assert!(
            before.used_high >= N as u64,
            "precondition: high watermark should be > 0"
        );

        pool.reset_used_high().expect("reset used_high failed");

        let after = pool.mem_stats().expect("read stats failed");
        assert_eq!(
            after.used_high, after.used_current,
            "after reset, used_high should collapse to used_current"
        );
    });
}
