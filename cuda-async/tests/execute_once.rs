/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests verifying that `unzip`, `zip`, and `shared` execute ancestor
//! operations exactly once, regardless of how many downstream branches
//! consume the results.

use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::{
    value, BoxedDeviceOp, DeviceOp, SharedDeviceOp, Unzippable2, Unzippable3, Zippable,
};
use cuda_async::zip;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Run `f` on a fresh thread so thread-local `DEVICE_CONTEXTS` starts clean.
fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

/// Helper: create a `BoxedDeviceOp` that increments `counter` on each execution.
fn counted_op<T: Send + 'static>(counter: &Arc<AtomicUsize>, val: T) -> BoxedDeviceOp<T> {
    let c = counter.clone();
    value(())
        .then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value(val)
        })
        .boxed()
}

// ---------------------------------------------------------------------------
// .shared() — cloneable, execute-once operations
// ---------------------------------------------------------------------------

#[test]
fn shared_executes_ancestor_exactly_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let op = counted_op(&counter, 42u64);

        let shared = op.shared();
        let a = shared.clone().sync().expect("first failed");
        let b = shared.sync().expect("second failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(*a, 42);
        assert_eq!(*b, 42);
        assert!(Arc::ptr_eq(&a, &b));
    });
}

#[test]
fn shared_n_way_clone() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let shared = counted_op(&counter, 99u64).shared();

        let results: Vec<Arc<u64>> = (0..5)
            .map(|_| shared.clone().sync().expect("clone failed"))
            .collect();

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        for r in &results {
            assert_eq!(**r, 99);
            assert!(Arc::ptr_eq(r, &results[0]));
        }
    });
}

#[test]
fn shared_into_zip() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let shared = counted_op(&counter, 42u64).shared();

        let (a, b) = zip!(shared.clone(), shared).sync().expect("sync failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(*a, 42);
        assert_eq!(*b, 42);
        assert!(Arc::ptr_eq(&a, &b));
    });
}

#[test]
fn shared_pre_computed() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let val = Arc::new(7u64);
        let shared: SharedDeviceOp<u64> = cuda_async::device_operation::shared(val.clone());

        let a = shared.clone().sync().expect("first failed");
        let b = shared.sync().expect("second failed");

        assert_eq!(*a, 7);
        assert!(Arc::ptr_eq(&a, &b));
        assert!(Arc::ptr_eq(&a, &val));
    });
}

// ---------------------------------------------------------------------------
// unzip (2-tuple)
// ---------------------------------------------------------------------------

#[test]
fn unzip2_executes_ancestor_exactly_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64))
        });

        let (left, right) = op.unzip();
        let a = left.sync().expect("left failed");
        let b = right.sync().expect("right failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    });
}

#[test]
fn unzip2_right_before_left() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64))
        });

        let (left, right) = op.unzip();
        let b = right.sync().expect("right failed");
        let a = left.sync().expect("left failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    });
}

// ---------------------------------------------------------------------------
// unzip (3-tuple — exercises nested _unzip chain)
// ---------------------------------------------------------------------------

#[test]
fn unzip3_executes_ancestor_exactly_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64, 3u64))
        });

        let (a, b, c_op) = op.unzip();
        let a = a.sync().expect("a failed");
        let b = b.sync().expect("b failed");
        let c_val = c_op.sync().expect("c failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c_val, 3);
    });
}

#[test]
fn unzip3_reversed_execution_order() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64, 3u64))
        });

        let (a, b, c_op) = op.unzip();
        let c_val = c_op.sync().expect("c failed");
        let b = b.sync().expect("b failed");
        let a = a.sync().expect("a failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c_val, 3);
    });
}

// ---------------------------------------------------------------------------
// zip then unzip (round-trip)
// ---------------------------------------------------------------------------

#[test]
fn zip2_then_unzip2_executes_each_input_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));

        let op_a = counted_op(&counter_a, 10u64);
        let op_b = counted_op(&counter_b, 20u64);

        let (a, b) = zip!(op_a, op_b).unzip();
        let a = a.sync().expect("a failed");
        let b = b.sync().expect("b failed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
        assert_eq!(a, 10);
        assert_eq!(b, 20);
    });
}

#[test]
fn zip3_then_unzip3_executes_each_input_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));
        let counter_c = Arc::new(AtomicUsize::new(0));

        let op_a = counted_op(&counter_a, 1u64);
        let op_b = counted_op(&counter_b, 2u64);
        let op_c = counted_op(&counter_c, 3u64);

        let (a, b, c_op) = zip!(op_a, op_b, op_c).unzip();
        let a = a.sync().expect("a failed");
        let b = b.sync().expect("b failed");
        let c_val = c_op.sync().expect("c failed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
        assert_eq!(counter_c.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c_val, 3);
    });
}

// ---------------------------------------------------------------------------
// Fan-out then fan-in (unzip → re-zip)
// ---------------------------------------------------------------------------

#[test]
fn unzip_then_rezip_executes_ancestor_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let op = value(()).then(move |()| {
            c.fetch_add(1, Ordering::SeqCst);
            value((1u64, 2u64))
        });

        let (left, right) = op.unzip();
        let rezipped = zip!(left, right);
        let (a, b) = rezipped.sync().expect("sync failed");

        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert_eq!(a, 1);
        assert_eq!(b, 2);
    });
}

// ---------------------------------------------------------------------------
// Diamond: zip two inputs, unzip, transform each branch, re-zip
// ---------------------------------------------------------------------------

#[test]
fn diamond_graph_executes_each_leaf_once() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");

        let counter_a = Arc::new(AtomicUsize::new(0));
        let counter_b = Arc::new(AtomicUsize::new(0));

        let op_a = counted_op(&counter_a, 10u64);
        let op_b = counted_op(&counter_b, 20u64);

        // zip → unzip (fan-out) → transform each branch → re-zip (fan-in)
        let (a, b) = zip!(op_a, op_b).unzip();
        let a = a.then(|v| value(v + 1));
        let b = b.then(|v| value(v + 2));
        let result = zip!(a, b).sync().expect("sync failed");

        assert_eq!(counter_a.load(Ordering::SeqCst), 1);
        assert_eq!(counter_b.load(Ordering::SeqCst), 1);
        assert_eq!(result, (11, 22));
    });
}
