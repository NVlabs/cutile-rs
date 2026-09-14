/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::cuda_async::device_buffer::DeviceAllocation;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::{value, ExecutionContext, Value};
use cutile::cuda_async::error::DeviceError;
use cutile::cuda_async::futures::{executor::block_on, task::noop_waker};
use cutile::prelude::*;
use std::future::{Future, IntoFuture};
use std::pin::Pin;
use std::sync::{mpsc, Arc};
use std::task::Context;
use std::time::Duration;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry]
    fn copy<const S: [i32; 1]>(out: &mut Tensor<f32, S>, input: &Tensor<f32, { [-1] }>) {
        out.store(load_tile_like(input, out));
    }
}

struct Allocation(Tensor<f32>);

// SAFETY: the owned tensor has a stable, live allocation. The test's rescue
// handle is used only to observe refcounts, never to access its GPU bytes.
unsafe impl DeviceAllocation for Allocation {
    fn device_ptr(&self) -> u64 {
        self.0.device_pointer().cu_deviceptr()
    }
    fn len_bytes(&self) -> usize {
        self.0.num_bytes()
    }
    fn device_id(&self) -> usize {
        0
    }
}

fn tracked(stream: &Arc<cuda_core::Stream>) -> (Tensor<f32>, Arc<Allocation>) {
    let allocation = Arc::new(Allocation(api::ones(&[32]).sync_on(stream).unwrap()));
    // SAFETY: no accesses are issued through `allocation`; only this foreign
    // tensor accesses the bytes. The rescue owner also makes a failing lifetime
    // assertion harmless, rather than causing an actual GPU use-after-free.
    let tensor = unsafe { Tensor::from_foreign(allocation.clone(), vec![32], vec![1]) };
    (tensor, allocation)
}

struct Gate(mpsc::Sender<()>);

impl Gate {
    fn arm(stream: &Arc<cuda_core::Stream>) -> Self {
        let (sender, receiver) = mpsc::channel();
        unsafe {
            stream
                .launch_host_function(move || {
                    // Bounded even if an unexpected synchronous path is introduced.
                    let _ = receiver.recv_timeout(Duration::from_secs(10));
                })
                .unwrap();
        }
        Self(sender)
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn forget_pending<O: DeviceOp>(op: O, stream: &Arc<cuda_core::Stream>) {
    let mut future = DeviceFuture::scheduled(op, ExecutionContext::new(stream.clone()));
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert!(Pin::new(&mut future).poll(&mut cx).is_pending());
    std::mem::forget(future);
}

fn on_gpu(f: impl FnOnce(Arc<cuda_core::Stream>, Arc<cuda_core::Stream>) + Send + 'static) {
    crate::common::with_test_stack(move || {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let other = device.new_stream().unwrap();
        let src = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        kernels::copy((&mut dst).partition([4]), &src)
            .sync_on(&stream)
            .unwrap();
        // Initialize the completion backend before placing any work behind a gate.
        unsafe {
            stream
                .launch_host_function(|| std::thread::sleep(Duration::from_millis(10)))
                .unwrap()
        };
        block_on(DeviceFuture::scheduled(
            value(()),
            ExecutionContext::new(stream.clone()),
        ))
        .unwrap();
        f(stream, other);
    });
}

#[test]
fn forgotten_borrowed_kernel_retains_inputs_outputs_and_excludes_conflicts() {
    on_gpu(|stream, other| {
        let (mut src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let mut scratch = api::zeros::<f32>(&[32]).sync_on(&other).unwrap();
        // A pre-recorded graph must perform the same access checks on replay.
        let graph = api::memcpy(&mut scratch, &dst)
            .graph_on(other.clone())
            .unwrap();
        let gate = Gate::arm(&stream);
        forget_pending(kernels::copy((&mut dst).partition([4]), &src), &stream);
        assert!(api::memcpy(&mut scratch, &dst).sync_on(&other).is_err());
        assert!(api::memcpy(&mut src, &scratch).sync_on(&other).is_err());
        assert!(graph.launch().sync_on(&other).is_err());
        drop(graph);
        drop((src, dst));
        assert!(
            Arc::strong_count(&src_owner) > 1,
            "borrowed input was released after recover"
        );
        assert!(
            Arc::strong_count(&dst_owner) > 1,
            "borrowed output was released after recover"
        );
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

#[test]
fn forgotten_view_and_mapped_output_keep_their_storage() {
    on_gpu(|stream, _| {
        let (src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let view = src.view(&[32]).unwrap();
        let gate = Gate::arm(&stream);
        forget_pending(
            kernels::copy((&mut dst).partition([4]).map([1], 8), &view).then(|_| value(())),
            &stream,
        );
        drop(view);
        drop((src, dst));
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

#[test]
fn forgotten_memcpy_keeps_both_allocations() {
    on_gpu(|stream, _| {
        let (src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let gate = Gate::arm(&stream);
        forget_pending(api::memcpy(&mut dst, &src), &stream);
        drop((src, dst));
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

#[test]
fn projected_owned_inputs_live_until_ready_and_then_release() {
    on_gpu(|stream, other| {
        let (src, src_owner) = tracked(&stream);
        let (dst, dst_owner) = tracked(&stream);
        let mut future = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src).then(|_| value(())),
            ExecutionContext::new(stream.clone()),
        );
        let gate = Gate::arm(&stream);
        let waker = noop_waker();
        assert!(Pin::new(&mut future)
            .poll(&mut Context::from_waker(&waker))
            .is_pending());
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        block_on(&mut future).unwrap();
        assert_eq!(Arc::strong_count(&src_owner), 1);
        assert_eq!(Arc::strong_count(&dst_owner), 1);

        let src = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        kernels::copy((&mut dst).partition([4]), &src)
            .sync_on(&stream)
            .unwrap();
        api::memcpy(&mut dst, &src).sync_on(&other).unwrap();
        let host = dst.to_host_vec().sync_on(&other).unwrap();
        assert_eq!(host, vec![1.0; 32]);
    });
}

#[test]
fn projected_dup_keeps_source_until_submission_completes() {
    on_gpu(|stream, _| {
        let (src, owner) = tracked(&stream);
        let op = api::dup(&src).then(|_| value(()));
        drop(src);
        let gate = Gate::arm(&stream);
        forget_pending(op, &stream);
        assert!(Arc::strong_count(&owner) > 1);
        drop(gate);
        unsafe { stream.synchronize().unwrap() };
    });
}

struct Fail;

impl DeviceOp for Fail {
    type Output = ();
    unsafe fn execute(self, _: &ExecutionContext) -> Result<(), DeviceError> {
        Err(DeviceError::Internal("injected submission error".into()))
    }
}

impl IntoFuture for Fail {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        DeviceFuture::failed(DeviceError::Internal("injected submission error".into()))
    }
}

#[test]
fn partial_submission_error_retains_discarded_resources() {
    on_gpu(|stream, _| {
        let (src, owner) = tracked(&stream);
        let dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut future = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src).then(|_| Fail),
            ExecutionContext::new(stream.clone()),
        );
        let gate = Gate::arm(&stream);
        let waker = noop_waker();
        let result = Pin::new(&mut future).poll(&mut Context::from_waker(&waker));
        assert!(matches!(result, std::task::Poll::Ready(Err(_))));
        assert!(Arc::strong_count(&owner) > 1);
        drop(gate);
        drop(future);
        assert_eq!(Arc::strong_count(&owner), 1);
    });
}

#[test]
fn partial_submission_panic_retains_discarded_resources() {
    on_gpu(|stream, _| {
        let (src, owner) = tracked(&stream);
        let dst = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();
        let mut future = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src)
                .then(|_| -> Value<()> { panic!("injected panic") }),
            ExecutionContext::new(stream.clone()),
        );
        let gate = Gate::arm(&stream);
        let waker = noop_waker();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Pin::new(&mut future).poll(&mut Context::from_waker(&waker));
        }));
        assert!(panic.is_err());
        assert!(Arc::strong_count(&owner) > 1);
        drop(gate);
        drop(future);
        assert_eq!(Arc::strong_count(&owner), 1);
    });
}

#[test]
fn graph_replay_retains_projected_storage_and_releases_launch_accesses() {
    on_gpu(|stream, other| {
        let (src, src_owner) = tracked(&stream);
        let (mut dst, dst_owner) = tracked(&stream);
        let graph = kernels::copy((&mut dst).partition([4]), &src)
            .then(|_| value(()))
            .graph_on(stream.clone())
            .unwrap();
        drop(src);
        assert!(Arc::strong_count(&src_owner) > 1);
        graph.launch().sync_on(&other).unwrap();
        let replacement = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
        graph.update(api::memcpy(&mut dst, &replacement)).unwrap();
        graph.launch().sync_on(&stream).unwrap();
        // A completed replay releases its leases even while the graph lives.
        let host = dst.dup().to_host_vec().sync_on(&other).unwrap();
        assert_eq!(host, vec![1.0; 32]);
        drop(dst);
        let launch = graph.launch();
        drop(graph);
        let gate = Gate::arm(&other);
        forget_pending(launch, &other);
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(gate);
        unsafe { other.synchronize().unwrap() };
    });
}

#[test]
fn cloned_execution_contexts_have_independent_submission_owners() {
    on_gpu(|stream, _| {
        let (src, src_owner) = tracked(&stream);
        let (dst, dst_owner) = tracked(&stream);
        let context = ExecutionContext::new(stream.clone());
        let mut first = DeviceFuture::scheduled(value(()), context.clone());
        let mut second = DeviceFuture::scheduled(
            kernels::copy(dst.partition([4]), src).then(|_| value(())),
            context,
        );
        let first_gate = Gate::arm(&stream);
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(Pin::new(&mut first).poll(&mut cx).is_pending());
        let second_gate = Gate::arm(&stream);
        assert!(Pin::new(&mut second).poll(&mut cx).is_pending());
        drop(first_gate);
        block_on(&mut first).unwrap();
        assert!(Arc::strong_count(&src_owner) > 1);
        assert!(Arc::strong_count(&dst_owner) > 1);
        drop(second_gate);
        block_on(&mut second).unwrap();
        assert_eq!(Arc::strong_count(&src_owner), 1);
        assert_eq!(Arc::strong_count(&dst_owner), 1);
    });
}
