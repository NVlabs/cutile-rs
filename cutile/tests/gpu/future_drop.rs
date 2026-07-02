/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests: dropping a pending `DeviceFuture` must block until the
//! in-flight stream work completes. Previously `DeviceFuture` had no `Drop`
//! impl, so cancelling a polled-once future freed its pending result (e.g. a
//! `Tensor` whose device buffer is then handed back to the pool) while the
//! stream was still executing the work that produces it.
//!
//! The main test pins the hazard deterministically: it blocks the stream with
//! a gating host function, polls the future once (queuing its work behind the
//! gate), drops the future on another thread, and asserts the drop does not
//! return while the stream is still busy.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Waker};
use std::time::Duration;

use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cutile::prelude::*;
use cutile::tile_kernel::global_policy;

use crate::common;

const N: usize = 1 << 22;

/// A `DeviceOp` that performs no device work itself. The test enqueues its
/// own gating work on the stream, so this isolates pure `DeviceFuture`
/// semantics from any op-specific behavior (JIT, allocation, fills).
struct Noop;

impl DeviceOp for Noop {
    type Output = u32;
    unsafe fn execute(self, _ctx: &ExecutionContext) -> Result<u32, DeviceError> {
        Ok(42)
    }
}

impl std::future::IntoFuture for Noop {
    type Output = Result<u32, DeviceError>;
    type IntoFuture = DeviceFuture<u32, Noop>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Dropping a pending future must block until the stream drains.
///
/// Without the blocking `Drop`, the drop returns immediately and the result
/// tensor's buffer is freed while the fill that writes it is still queued —
/// the pool can recycle the memory out from under the in-flight work.
#[test]
fn drop_pending_future_blocks_until_stream_drains() {
    common::with_test_stack(|| {
        let _policy = global_policy(0);
        let stream = match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => stream,
            Ok(Err(e)) | Err(e) => panic!("failed to get stream: {e:?}"),
        };

        // Gate the stream: nothing enqueued after this host function runs
        // until the gate is released.
        let (gate_tx, gate_rx) = mpsc::channel::<()>();
        // SAFETY: the stream is kept alive by our Arc for the callback's
        // lifetime, and the closure performs no CUDA calls.
        unsafe {
            stream
                .launch_host_function(move || {
                    let _ = gate_rx.recv();
                })
                .expect("launch host function");
        }

        // Poll once: the future transitions to Executing and its completion
        // callback is queued behind the gate, so it is genuinely pending
        // with stream work outstanding.
        let mut fut = DeviceFuture::scheduled(Noop, ExecutionContext::new(stream.clone()));
        let mut cx = Context::from_waker(Waker::noop());
        let poll = Pin::new(&mut fut).poll(&mut cx);
        assert!(
            matches!(poll, Poll::Pending),
            "work is gated, must be pending"
        );

        // Drop the pending future on another thread; it must not return
        // while the stream is still busy.
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_clone = dropped.clone();
        let dropper = std::thread::spawn(move || {
            drop(fut);
            dropped_clone.store(true, Ordering::SeqCst);
        });

        std::thread::sleep(Duration::from_millis(300));
        let returned_early = dropped.load(Ordering::SeqCst);

        // Release the gate before asserting so a failure doesn't wedge the
        // stream (and the dropper thread) forever.
        gate_tx.send(()).expect("release gate");
        dropper.join().expect("dropper thread");

        assert!(
            !returned_early,
            "Drop returned while stream work was still in flight: \
             the pending result was freed while the device could still use it"
        );

        // Stream and pool sanity after the blocked drop resolved.
        let v = api::ones::<f32>(&[N]).to_host_vec().sync().expect("sync");
        assert!(v.iter().all(|&x| x == 1.0));
    });
}

/// Stress variant: repeatedly drop pending tensor futures (no gate, real
/// races) and verify the allocator/pool still produces correct results.
#[test]
fn drop_pending_tensor_future_stress() {
    common::with_test_stack(|| {
        let _policy = global_policy(0);
        let mut cx = Context::from_waker(Waker::noop());
        for _ in 0..24 {
            let op = api::zeros::<f32>(&[N]);
            let mut fut = match with_default_device_policy(|policy| {
                let stream = policy.next_stream()?;
                Ok(DeviceFuture::scheduled(op, ExecutionContext::new(stream)))
            }) {
                Ok(Ok(fut)) => fut,
                Ok(Err(e)) | Err(e) => panic!("failed to schedule future: {e:?}"),
            };
            if let Poll::Pending = Pin::new(&mut fut).poll(&mut cx) {
                drop(fut);
            }
        }
        let v = api::ones::<f32>(&[N]).to_host_vec().sync().expect("sync");
        assert!(v.iter().all(|&x| x == 1.0));
    });
}
