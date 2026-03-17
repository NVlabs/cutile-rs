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
/// A scheduling policy decides **which CUDA stream** each [`DeviceOperation`] runs on.
/// This single decision controls whether consecutive operations can overlap on the GPU:
///
/// - **Same stream** → operations execute in order (serialized).
/// - **Different streams** → operations *may* execute concurrently.
///
/// The default policy is [`RoundRobin`](GlobalSchedulingPolicy::RoundRobin), which
/// distributes operations across a pool of streams.
///
/// # Choosing a Policy
///
/// | Policy          | Behavior                 | When to use                                  |
/// |-----------------|--------------------------|----------------------------------------------|
/// | `RoundRobin(N)` | Cycles through N streams | Default; enables overlap for independent ops |
/// | `SingleStream`  | All ops on one stream    | Strict ordering without manual sync          |
///
/// See [`StreamPoolRoundRobin`] and [`SingleStream`] for details.
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
        match self.as_ref() {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.schedule(op),
        }
    }
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError> {
        match self.as_ref() {
            GlobalSchedulingPolicy::RoundRobin(roundrobin) => roundrobin.sync(op),
        }
    }
}

/// Trait for types that are bound to a specific GPU device.
pub trait WithDeviceId {
    fn get_device_id(&self) -> usize;
}

/// A strategy for assigning [`DeviceOperation`]s to CUDA streams.
///
/// Every operation submitted through `.await`, `.sync()`, or `.schedule()` passes through
/// a `SchedulingPolicy`. The policy picks a stream and returns a [`DeviceFuture`] bound to
/// that stream.
///
/// # Stream Ordering Guarantees
///
/// CUDA guarantees that work items on the **same stream** execute in submission order.
/// Work on **different streams** has no ordering guarantee — the GPU hardware scheduler
/// is free to interleave or overlap them.
///
/// This means the policy directly affects concurrency behavior:
///
/// ```text
/// ┌──────────────────────────────────────────────────-┐
/// │  RoundRobin(4)        op1 ─► Stream 0             │
/// │                       op2 ─► Stream 1  (overlap!) │
/// │                       op3 ─► Stream 2  (overlap!) │
/// │                       op4 ─► Stream 3  (overlap!) │
/// │                       op5 ─► Stream 0  (waits for │
/// │                                         op1)      │
/// ├──────────────────────────────────────────────────-┤
/// │  SingleStream         op1 ─► Stream 0             │
/// │                       op2 ─► Stream 0  (waits)    │
/// │                       op3 ─► Stream 0  (waits)    │
/// └──────────────────────────────────────────────────-┘
/// ```
///
/// # Safety
///
/// Implementations must be `Sync` because the policy is shared across async tasks.
///
// TODO (hme): Isaac's feedback:
//  - Schedule op takes multiple deviceOps + meta data per policy*.
//  - Metadata type per policy impl.
pub trait SchedulingPolicy: Sync {
    /// Create the underlying CUDA streams. Called once during device initialization.
    fn init(&mut self, ctx: &Arc<CudaContext>) -> Result<(), DeviceError>;

    /// Assign `op` to a stream and return a [`DeviceFuture`] that will execute it.
    ///
    /// The operation is **not** executed yet — execution happens when the returned
    /// future is first polled.
    fn schedule<T: Send, O: DeviceOperation<Output = T>>(
        &self,
        op: O,
    ) -> Result<DeviceFuture<T, O>, DeviceError>;

    /// Execute `op` synchronously: submit to a stream, then block until the GPU finishes.
    fn sync<T: Send, O: DeviceOperation<Output = T>>(&self, op: O) -> Result<T, DeviceError>;
}

/// Distributes operations across a fixed-size pool of CUDA streams using round-robin assignment.
///
/// This is the **default scheduling policy**. Each call to [`schedule`](SchedulingPolicy::schedule)
/// or [`sync`](SchedulingPolicy::sync) picks the next stream in the pool (wrapping around),
/// so consecutive operations typically land on **different streams** and may run concurrently
/// on the GPU.
///
/// # Default Configuration
///
/// The default pool size is **4 streams**
/// ([`DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE`](crate::device_context::DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE)).
///
/// # Overlap and Dependencies
///
/// Because consecutive operations usually go to different streams, they can overlap:
///
/// ```text
/// .await (op A) ─► Stream 0 ──▶ ████████          (GPU work A)
/// .await (op B) ─► Stream 1 ──▶    ████████       (GPU work B, overlaps A)
/// .await (op C) ─► Stream 2 ──▶       ████████    (GPU work C, overlaps B)
/// ```
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

/// Routes every operation to a single CUDA stream, guaranteeing strict sequential execution.
///
/// All operations submitted through this policy execute in exactly the order they are
/// scheduled — no overlap, no reordering. This is the simplest mental model but leaves
/// GPU concurrency on the table.
///
/// # When to Use
///
/// - Debugging: eliminates concurrency as a source of bugs.
/// - Strict pipelines: when every operation depends on the previous one and you want
///   to avoid explicit synchronization.
///
/// For most workloads, [`StreamPoolRoundRobin`] is preferred because it allows the GPU
/// to overlap independent operations.
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
