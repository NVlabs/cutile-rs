/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stream scheduling policies that control how operations are assigned to CUDA streams.

use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOperation, ExecutionContext};
use crate::error::{device_error, DeviceError};
use cuda_core::{CudaContext, CudaStream};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

/// The active scheduling policy for a device context.
///
/// Decides which CUDA stream each [`DeviceOperation`] runs on, which controls whether
/// consecutive operations can overlap on the GPU.
///
/// | Policy          | Behavior                 | When to use                                  |
/// |-----------------|--------------------------|----------------------------------------------|
/// | `RoundRobin(N)` | Cycles through N streams | Default; enables overlap for independent ops |
/// | `SingleStream`  | All ops on one stream    | Strict ordering without manual sync          |
pub enum GlobalSchedulingPolicy {
    /// Round-robin scheduling across a pool of CUDA streams.
    RoundRobin(StreamPoolRoundRobin),
}

impl GlobalSchedulingPolicy {
    pub fn as_scheduling_policy(&self) -> Result<&impl SchedulingPolicy, DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => Ok(roundrobin),
        }
    }
}

impl WithDeviceId for GlobalSchedulingPolicy {
    fn get_device_id(&self) -> usize {
        match self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.get_device_id(),
        }
    }
}

impl SchedulingPolicy for GlobalSchedulingPolicy {
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.init(ctx),
        }
    }
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.schedule(op),
        }
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        match self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.sync(op),
        }
    }
}

impl SchedulingPolicy for Arc<GlobalSchedulingPolicy> {
    fn init(&mut self, _ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        Err(DeviceError::Scheduling(
            "Cannot initialize scheduling policy inside an Arc.".to_string(),
        ))
    }
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        match &**self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.schedule(op),
        }
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        match &**self {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.sync(op),
        }
    }
}

/// Trait for types that are bound to a specific GPU device.
pub trait WithDeviceId {
    fn get_device_id(&self) -> usize;
}

/// Assigns [`DeviceOperation`]s to CUDA streams.
///
/// Same-stream operations execute in submission order. Different-stream operations may overlap.
///
/// ```text
/// RoundRobin(4):  op1 → Stream 0,  op2 → Stream 1 (overlap),  op3 → Stream 2 (overlap)
/// SingleStream:   op1 → Stream 0,  op2 → Stream 0 (waits),    op3 → Stream 0 (waits)
/// ```
///
// TODO (hme): Isaac's feedback:
//  - Schedule op takes multiple deviceOps + meta data per policy*.
//  - Metadata type per policy impl.
pub trait SchedulingPolicy: Sync {
    /// Initialize the underlying CUDA streams. Called once during device setup.
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError>;

    /// Assign `op` to a stream and return a [`DeviceFuture`]. Execution starts on first poll.
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError>;

    /// Execute `op` synchronously: submit to a stream, then block until the GPU finishes.
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError>;
}

/// Default scheduling policy: distributes operations across a fixed pool of CUDA streams.
///
/// Each call picks the next stream in round-robin order, so consecutive independent
/// operations land on different streams and may run concurrently.
/// Default pool size: [`DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE`](crate::device_context::DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE).
#[derive(Debug)]
pub struct StreamPoolRoundRobin {
    device_id: usize,
    next_stream_idx: AtomicUsize,
    pub(crate) num_streams: usize,
    pub(crate) stream_pool: Option<Vec<Arc<CudaStream>>>,
}

impl StreamPoolRoundRobin {
    // This has to be unsafe, because we cannot otherwise guarantee correct ordering of operations.
    pub unsafe fn new(device_id: usize, num_streams: usize) -> Self {
        Self {
            device_id,
            num_streams,
            stream_pool: None,
            next_stream_idx: AtomicUsize::new(0),
        }
    }
}

impl SchedulingPolicy for StreamPoolRoundRobin {
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        let mut stream_pool = vec![];
        for _ in 0..self.num_streams {
            let stream = ctx.new_stream()?;
            stream_pool.push(stream);
        }
        self.stream_pool = Some(stream_pool);
        Ok(())
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        let non_wrapping_idx = self
            .next_stream_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stream_idx = non_wrapping_idx % self.num_streams;
        let stream_pool = self
            .stream_pool
            .as_ref()
            .ok_or(device_error(self.device_id, "Stream pool not initialized."))?;
        let stream = stream_pool[stream_idx].clone();
        op.sync_on(&stream)
    }
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        let non_wrapping_idx = self
            .next_stream_idx
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let stream_idx = non_wrapping_idx % self.num_streams;
        let stream_pool = self
            .stream_pool
            .as_ref()
            .ok_or(device_error(self.device_id, "Stream pool not initialized."))?;
        let stream = stream_pool[stream_idx].clone();
        let mut future = DeviceFuture::new();
        future.device_operation = Some(op);
        future.execution_context = Some(ExecutionContext::new(stream));
        Ok(future)
    }
}

impl WithDeviceId for StreamPoolRoundRobin {
    fn get_device_id(&self) -> usize {
        self.device_id
    }
}

/// Routes all operations to a single CUDA stream: strict sequential execution, no overlap.
///
/// Useful for debugging or when every operation depends on the previous one.
/// For most workloads, [`StreamPoolRoundRobin`] is preferred.
#[derive(Debug)]
pub struct SingleStream {
    #[expect(dead_code, reason = "unsure what this is for")]
    device_id: usize,
    pub stream: Option<Arc<CudaStream>>,
}

impl SingleStream {
    // This has to be unsafe, because we cannot otherwise guarantee correct ordering of operations.
    pub unsafe fn new(device_id: usize) -> Self {
        Self {
            device_id,
            stream: None,
        }
    }
    pub fn schedule_single<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> DeviceFuture<T, O> {
        let mut future = DeviceFuture::new();
        future.device_operation = Some(op);
        let stream = self.stream.as_ref().unwrap().clone();
        future.execution_context = Some(ExecutionContext::new(stream));
        future
    }
}

impl SchedulingPolicy for SingleStream {
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError> {
        self.stream = Some(
            ctx.new_stream()
                .expect("Failed to create dedicated stream."),
        );
        Ok(())
    }
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError> {
        Ok(self.schedule_single(op))
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        let stream = self.stream.as_ref().unwrap().clone();
        op.sync_on(&stream)
    }
}
