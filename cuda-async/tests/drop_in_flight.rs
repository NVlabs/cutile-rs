/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Dropping a `DeviceFuture` while its GPU work is in flight must not release
//! the operation's output before the device has finished with it.
//!
//! The op enqueues a long chain of memsets and then records a CUDA event on
//! the same stream, handing the event back inside its output. The output's
//! `Drop` queries that event: if the output were released early, the query
//! would report the work still in flight. Requires a GPU.

use cuda_async::device_context::{global_policy, init_device_contexts};
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::Event;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

fn noop_waker() -> Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

fn alloc_device(bytes: usize) -> u64 {
    cuda_async::device_context::with_device(0, |device| device.bind_to_thread())
        .expect("device context")
        .expect("bind_to_thread failed");
    let mut dptr = std::mem::MaybeUninit::uninit();
    let code = unsafe { cuda_bindings::cuMemAlloc_v2(dptr.as_mut_ptr(), bytes) };
    assert_eq!(code, 0, "cuMemAlloc failed: {code}");
    unsafe { dptr.assume_init() }
}

/// What the output's `Drop` observed: whether the device had passed the
/// event recorded after the op's work when the output was released.
type DropLog = Arc<Mutex<Vec<bool>>>;

/// The op's output: owns the completion event and reports, on drop, whether
/// the work had completed by then.
struct Tracked {
    event: Event,
    log: DropLog,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        let done = self.event.query().unwrap_or(false);
        self.log.lock().unwrap().push(done);
    }
}

/// `passes` memsets of a `bytes` buffer, then an event recorded after them.
struct SlowOp {
    dptr: u64,
    bytes: usize,
    passes: usize,
    log: DropLog,
}

impl DeviceOp for SlowOp {
    type Output = Tracked;
    unsafe fn execute(self, context: &ExecutionContext) -> Result<Tracked, DeviceError> {
        let stream = context.get_cuda_stream();
        for _ in 0..self.passes {
            let code =
                cuda_bindings::cuMemsetD8Async(self.dptr, 0x5A, self.bytes, stream.cu_stream());
            if code != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
                return Err(DeviceError::Internal(format!(
                    "cuMemsetD8Async failed: {code}"
                )));
            }
        }
        let event = stream.device().new_event()?;
        event.record(stream)?;
        Ok(Tracked {
            event,
            log: self.log,
        })
    }
}

impl IntoFuture for SlowOp {
    type Output = Result<Tracked, DeviceError>;
    type IntoFuture = cuda_async::device_future::DeviceFuture<Tracked, SlowOp>;
    fn into_future(self) -> Self::IntoFuture {
        let policy = global_policy(0).expect("global policy");
        match self.schedule(&policy) {
            Ok(future) => future,
            Err(error) => cuda_async::device_future::DeviceFuture::failed(error),
        }
    }
}

const BUF: usize = 64 << 20;
/// ~64 MiB x 32 of memset: well past the 20 us inline-spin budget, so the
/// first poll leaves the work in flight.
const PASSES: usize = 32;

fn slow_op(dptr: u64, log: &DropLog) -> SlowOp {
    SlowOp {
        dptr,
        bytes: BUF,
        passes: PASSES,
        log: Arc::clone(log),
    }
}

fn block_on_with_deadline<F: Future + Unpin>(mut future: F, deadline: Duration) -> F::Output {
    let start = Instant::now();
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    loop {
        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => {
                assert!(
                    start.elapsed() < deadline,
                    "future did not complete within {deadline:?}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// The core regression: poll once (work in flight), drop the future, and
/// check that the output was released only after the device passed the
/// event recorded behind the work.
#[test]
fn dropping_in_flight_future_releases_output_after_the_device_finished() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        for _ in 0..4 {
            let mut future = slow_op(dptr, &log).into_future();
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            assert!(
                Pin::new(&mut future).poll(&mut cx).is_pending(),
                "the op must still be in flight after the first poll"
            );
            drop(future);
        }

        let log = log.lock().unwrap();
        assert_eq!(log.len(), 4, "every dropped future must release its output");
        assert!(
            log.iter().all(|&done| done),
            "an output was released while its GPU work was still in flight: {log:?}"
        );
    });
}

/// A future dropped before its first poll never executed: nothing was
/// submitted, nothing is waited on, and no output exists.
#[test]
fn dropping_unpolled_future_submits_nothing() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        let started = Instant::now();
        drop(slow_op(dptr, &log).into_future());
        assert!(started.elapsed() < Duration::from_millis(50));
        assert!(log.lock().unwrap().is_empty());
    });
}

/// After the result has been delivered, dropping the future is a no-op; the
/// delivered output is the caller's and reports completion when dropped.
#[test]
fn delivered_result_is_owned_by_the_caller() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        let tracked =
            block_on_with_deadline(slow_op(dptr, &log).into_future(), Duration::from_secs(30))
                .expect("op failed");
        assert!(log.lock().unwrap().is_empty(), "nothing dropped yet");
        drop(tracked);
        assert_eq!(log.lock().unwrap().as_slice(), [true]);
    });
}

/// Cancelling a future does not disturb later work on the same streams.
#[test]
fn later_pipelines_complete_after_cancellations() {
    on_fresh_thread(|| {
        init_device_contexts(0, 1).expect("init failed (requires GPU)");
        let dptr = alloc_device(BUF);
        let log: DropLog = Arc::new(Mutex::new(Vec::new()));

        for _ in 0..8 {
            let mut future = slow_op(dptr, &log).into_future();
            let waker = noop_waker();
            let mut cx = Context::from_waker(&waker);
            let _ = Pin::new(&mut future).poll(&mut cx);
            drop(future);
        }
        for _ in 0..4 {
            block_on_with_deadline(slow_op(dptr, &log).into_future(), Duration::from_secs(30))
                .expect("op after cancellations failed");
        }
        assert!(log.lock().unwrap().iter().all(|&done| done));
    });
}
