/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device_operation::{DeviceOp, ExecutionContext, GraphNode};
use crate::error::DeviceError;
use cuda_core::{stream, sys, CudaStream, IntoResult};
use std::mem::MaybeUninit;
use std::sync::Arc;

const CU_STREAM_CAPTURE_MODE_RELAXED: sys::CUstreamCaptureMode = 2;

/// A captured and instantiated CUDA graph, ready for replay.
///
/// Created via [`CudaGraph::capture`], which executes a [`DeviceOp`] once
/// on a capture stream, recording all GPU work into a graph. The graph can then
/// be replayed any number of times via [`launch`](CudaGraph::launch).
///
/// All device pointers used by the operation are baked into the graph at capture
/// time. To vary inputs between replays, pre-allocate an input buffer, pass it
/// into the operation, and memcpy new data into that buffer before each launch.
///
/// # Examples
///
/// ```rust,ignore
/// use cuda_async::prelude::*;
///
/// // Build a lazy operation (no GPU work yet).
/// let forward_op = build_forward_pass(&model, &bufs);
///
/// // Capture: executes once, records into graph, synchronizes.
/// let mut graph = CudaGraph::capture(stream.clone(), forward_op)?;
/// let bufs = graph.take_output().unwrap();
///
/// // Replay loop.
/// for _ in 0..n_tokens {
///     // Optionally: copy new input into a pre-allocated buffer here.
///     graph.launch()?;
/// }
/// ```
pub struct CudaGraph<T> {
    stream: Arc<CudaStream>,
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    output: Option<T>,
}

impl<T: Send> CudaGraph<T> {
    /// Capture a [`DeviceOp`] into a replayable CUDA graph.
    ///
    /// Executes `op` once on `stream` in capture mode. All GPU work (kernel
    /// launches, memcpys, etc.) issued by the operation is recorded into a
    /// graph. The graph is then instantiated, uploaded, and the stream is
    /// synchronized so the output `T` is safe to read immediately.
    ///
    /// Retrieve the output via [`take_output`](CudaGraph::take_output).
    pub fn capture(
        stream: Arc<CudaStream>,
        op: impl DeviceOp<Output = T>,
    ) -> Result<Self, DeviceError> {
        let ctx = stream.context().clone();
        ctx.bind_to_thread()?;

        // Begin capture.
        unsafe {
            stream::begin_capture(stream.cu_stream(), CU_STREAM_CAPTURE_MODE_RELAXED)?;
        }

        // Execute the operation on the capture stream.
        let exec_ctx = ExecutionContext::new(stream.clone());
        let op_result = unsafe { op.execute(&exec_ctx) };

        // End capture — must happen regardless of op success.
        let end_result = unsafe { stream::end_capture(stream.cu_stream()) };

        // Handle the (op_result, end_result) matrix, cleaning up on failure.
        let (output, cu_graph) = match (op_result, end_result) {
            (Err(op_err), Ok(cu_graph)) => {
                if !cu_graph.is_null() {
                    unsafe {
                        let _ = sys::cuGraphDestroy(cu_graph).result();
                    }
                }
                return Err(op_err);
            }
            (Err(op_err), Err(_)) => {
                return Err(op_err);
            }
            (Ok(_), Err(capture_err)) => {
                return Err(DeviceError::Driver(capture_err));
            }
            (Ok(output), Ok(cu_graph)) => {
                if cu_graph.is_null() {
                    return Err(DeviceError::Internal(
                        "cuStreamEndCapture returned null graph".into(),
                    ));
                }
                (output, cu_graph)
            }
        };

        // Instantiate.
        let cu_graph_exec = unsafe {
            let mut cu_graph_exec = MaybeUninit::<sys::CUgraphExec>::uninit();
            match sys::cuGraphInstantiateWithFlags(cu_graph_exec.as_mut_ptr(), cu_graph, 0).result()
            {
                Ok(()) => cu_graph_exec.assume_init(),
                Err(e) => {
                    let _ = sys::cuGraphDestroy(cu_graph).result();
                    return Err(DeviceError::Driver(e));
                }
            }
        };

        // Upload (pre-stages graph resources on the device).
        if let Err(e) = unsafe { sys::cuGraphUpload(cu_graph_exec, stream.cu_stream()).result() } {
            unsafe {
                let _ = sys::cuGraphExecDestroy(cu_graph_exec).result();
                let _ = sys::cuGraphDestroy(cu_graph).result();
            }
            return Err(DeviceError::Driver(e));
        }

        // Synchronize so the output is safe to read.
        stream.synchronize()?;

        Ok(Self {
            stream,
            cu_graph,
            cu_graph_exec,
            output: Some(output),
        })
    }

    /// Take the output produced during the capture execution.
    ///
    /// Returns `Some(T)` on the first call, `None` thereafter. Use this to
    /// recover intermediate buffers or inspect the initial result.
    pub fn take_output(&mut self) -> Option<T> {
        self.output.take()
    }

    /// Execute a [`DeviceOp`] on the graph's stream without synchronizing.
    ///
    /// Use this to update graph inputs before [`launch`](CudaGraph::launch).
    /// The operation is issued on the same stream the graph will run on, so
    /// stream ordering guarantees it completes before the graph's kernels
    /// begin. Synchronization happens when `launch()` returns.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Copy a new embedding into the graph's pre-allocated input buffer.
    /// graph.update(api::memcpy(&mut h_input, &new_embedding))?;
    /// graph.launch()?;
    /// ```
    pub fn update<O: Send>(&self, op: impl DeviceOp<Output = O>) -> Result<O, DeviceError> {
        let ctx = ExecutionContext::new(self.stream.clone());
        unsafe { op.execute(&ctx) }
    }

    /// Replay the captured graph and synchronize.
    ///
    /// Launches the graph on the capture stream and blocks until the GPU
    /// finishes. Any operations issued via [`update`](CudaGraph::update)
    /// on the same stream are guaranteed to complete before the graph runs.
    // TODO: Add `launch_on(&stream)` to support launching on a different
    // stream than the capture stream (requested by Isaac Gelado).
    pub fn launch(&self) -> Result<(), DeviceError> {
        unsafe {
            sys::cuGraphLaunch(self.cu_graph_exec, self.stream.cu_stream()).result()?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Returns a reference to the stream this graph was captured on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> Drop for CudaGraph<T> {
    fn drop(&mut self) {
        let ctx = self.stream.context();
        ctx.record_err(ctx.bind_to_thread());

        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            ctx.record_err(unsafe { sys::cuGraphExecDestroy(cu_graph_exec).result() });
        }

        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            ctx.record_err(unsafe { sys::cuGraphDestroy(cu_graph).result() });
        }
    }
}

/// A scope for recording GPU operations into a CUDA graph.
///
/// Created by [`CudaGraph::scope`]. Each call to [`record`](Scope::record)
/// records a [`GraphNode`] as a graph node. The op is consumed immediately,
/// releasing any borrows it holds. This means a buffer written by one
/// kernel can be read by the next — `record` releases the `&mut` borrow,
/// allowing a subsequent `record` to take `&` on the same buffer.
///
/// ```rust,ignore
/// let graph = CudaGraph::scope(&stream, |s| {
///     s.record(rms_norm((&mut bufs.norm).partition([1, d]), &input, &w))?;
///     // bufs.norm borrow released — can now read it:
///     s.record(matvec((&mut bufs.q).partition([bn]), &bufs.norm, &wq))?;
///     Ok(())
/// })?;
///
/// graph.launch()?;
/// ```
///
/// # Safety proof: why `record` is safe
///
/// A CUDA data race occurs when two accesses to the same device memory
/// are unordered and at least one is a write. This is UB per both CUDA
/// and Rust.
///
/// `record` is safe because of two complementary mechanisms:
///
/// ## Capture mode prevents concurrent GPU execution
///
/// The scope's stream is in **capture mode** during the closure (via
/// `cuStreamBeginCapture`). In capture mode:
///
/// 1. **No GPU work executes.** `record` records operations as graph
///    nodes — kernels are not launched, memcpys are not issued. There
///    is no in-flight GPU work that could race with anything.
///
/// 2. **Same-stream ordering is preserved.** All `record` calls go to
///    the same capture stream. When the graph is later launched via
///    [`CudaGraph::launch`], the nodes execute in recorded order on a
///    single stream. Sequential same-stream execution is ordered — no
///    data races.
///
/// 3. **Cross-stream operations during capture are harmless.** If the
///    user calls `op.sync_on(&other_stream)` inside the closure, that
///    work executes eagerly on `other_stream` — but no captured work is
///    executing concurrently (the capture stream is recording, not
///    running). At graph launch time, the eagerly-executed work is long
///    complete. No overlap, no race.
///
/// 4. **`sync_on` on the capture stream fails at runtime.** CUDA returns
///    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` if you try to synchronize
///    a stream that is capturing.
///
/// 5. **Borrow checker enforces `&mut` exclusivity.** `record` consumes
///    the op, releasing `&mut`. The next `record` can then borrow the
///    same buffer as `&` for reading.
///
/// ## `GraphNode` prevents allocation during capture
///
/// `record` accepts [`GraphNode`] (not [`DeviceOp`]). `GraphNode` is only
/// implemented by operations that do not allocate or free device memory
/// (kernel launches, `memcpy`, `value`). This prevents:
///
/// - **Address instability:** `cuMemAllocAsync` during capture allocates
///   memory, but on replay the allocation node may return a different
///   address. Subsequent nodes bake in the capture-time pointer — UB.
///
/// - **Uninitialized reads:** An allocation during capture gives the user
///   a tensor handle. The initialization (e.g., memset from `zeros`) was
///   recorded, not executed. Passing the tensor to `sync_on(&other_stream)`
///   reads uninitialized memory.
///
/// - **Invalid frees:** If a tensor allocated inside the scope is dropped,
///   `cuMemFreeAsync` is recorded. On replay, it frees the capture-time
///   address, which may no longer be valid.
///
/// Since no tensors can be allocated inside the scope, all buffers are
/// pre-allocated and passed in via borrows. No tensor created inside
/// the scope means no tensor dropped inside the scope.
///
/// # What happens if you call other operations inside the closure
///
/// While `s.record(op)` is the intended API, other operations inside
/// the closure have well-defined behavior:
///
/// | Operation | What happens |
/// |---|---|
/// | `op.sync_on(&capture_stream)` | Runtime error from CUDA driver |
/// | `op.sync_on(&other_stream)` | Executes eagerly outside the graph — no race (see point 3) |
/// | `op.sync()` | May pick the capture stream (error) or another stream (executes eagerly) |
///
/// These are all defined behavior but serve no purpose inside a graph
/// capture scope — use `s.record(op)` instead.
///
/// # Thread safety
///
/// `Scope` is `!Send` — it cannot escape to another thread.
///
/// See `.internal/cuda-graph-redesign/SAFETY_PROOF_CUDA_GRAPH.md` for the full
/// formal proof.
pub struct Scope {
    ctx: ExecutionContext,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Scope {
    /// Record a [`GraphNode`] into the graph being captured.
    ///
    /// The op is consumed, recording its GPU work (kernel launch, memcpy)
    /// as a graph node. Any borrows held by the op are released when this
    /// call returns. The return value contains valid metadata (tensor
    /// shapes, device pointers) but GPU data is not yet computed — the
    /// actual computation happens when the graph is replayed via
    /// [`CudaGraph::launch`].
    ///
    /// Only operations that implement [`GraphNode`] can be recorded.
    /// This excludes allocation ops (`zeros`, `ones`, `dup`, etc.)
    /// whose addresses may change on replay.
    pub fn record<T: Send>(
        &self,
        op: impl GraphNode + DeviceOp<Output = T>,
    ) -> Result<T, DeviceError> {
        // SAFETY: The scope's stream is in capture mode. No GPU work
        // executes — ops are recorded as graph nodes. The GraphNode bound
        // ensures no alloc/free ops are recorded. See Scope docs for
        // the full safety proof.
        unsafe { op.execute(&self.ctx) }
    }
}

impl CudaGraph<()> {
    /// Capture a CUDA graph using a scoped closure.
    ///
    /// The closure receives a [`Scope`] for recording operations. Each
    /// `s.record(op)` records a graph node and consumes the op, releasing
    /// borrows. A buffer written by one `record` call can be read by the
    /// next.
    ///
    /// Pre-allocate all buffers before calling this method — the graph
    /// replays into the same device pointers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut output = api::zeros::<f32>(&[d]).sync_on(&stream)?;
    /// let weights = api::ones::<f32>(&[d]).sync_on(&stream)?;
    ///
    /// let graph = CudaGraph::scope(&stream, |s| {
    ///     s.record(kernel1((&mut output).partition([128]), &weights))?;
    ///     s.record(kernel2((&mut output).partition([64]), &weights))?;
    ///     Ok(())
    /// })?;
    ///
    /// graph.launch()?;
    /// ```
    ///
    /// See [`Scope`] for the safety proof and edge-case behavior.
    pub fn scope<F>(stream: &Arc<CudaStream>, f: F) -> Result<Self, DeviceError>
    where
        F: FnOnce(&Scope) -> Result<(), DeviceError>,
    {
        crate::device_operation::acquire_execution_lock()?;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::scope_inner(stream, f)
        }));

        crate::device_operation::release_execution_lock();

        match result {
            Ok(inner) => inner,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    fn scope_inner<F>(stream: &Arc<CudaStream>, f: F) -> Result<Self, DeviceError>
    where
        F: FnOnce(&Scope) -> Result<(), DeviceError>,
    {
        let ctx = stream.context().clone();
        ctx.bind_to_thread()?;

        // Begin capture.
        unsafe {
            stream::begin_capture(stream.cu_stream(), CU_STREAM_CAPTURE_MODE_RELAXED)?;
        }

        let scope = Scope {
            ctx: ExecutionContext::new(stream.clone()),
            _not_send: std::marker::PhantomData,
        };

        // Run the closure. Catch panics so cuStreamEndCapture always runs.
        let scope_result =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&scope))) {
                Ok(result) => result,
                Err(panic_payload) => {
                    let _ = unsafe { stream::end_capture(stream.cu_stream()) };
                    std::panic::resume_unwind(panic_payload);
                }
            };

        // End capture.
        let end_result = unsafe { stream::end_capture(stream.cu_stream()) };

        let cu_graph = match (scope_result, end_result) {
            (Err(scope_err), Ok(cu_graph)) => {
                if !cu_graph.is_null() {
                    unsafe {
                        let _ = sys::cuGraphDestroy(cu_graph).result();
                    }
                }
                return Err(scope_err);
            }
            (Err(scope_err), Err(_)) => {
                return Err(scope_err);
            }
            (Ok(_), Err(capture_err)) => {
                return Err(DeviceError::Driver(capture_err));
            }
            (Ok(()), Ok(cu_graph)) => {
                if cu_graph.is_null() {
                    return Err(DeviceError::Internal(
                        "cuStreamEndCapture returned null graph".into(),
                    ));
                }
                cu_graph
            }
        };

        // Instantiate.
        let cu_graph_exec = unsafe {
            let mut cu_graph_exec = MaybeUninit::<sys::CUgraphExec>::uninit();
            match sys::cuGraphInstantiateWithFlags(cu_graph_exec.as_mut_ptr(), cu_graph, 0).result()
            {
                Ok(()) => cu_graph_exec.assume_init(),
                Err(e) => {
                    let _ = sys::cuGraphDestroy(cu_graph).result();
                    return Err(DeviceError::Driver(e));
                }
            }
        };

        // Upload.
        if let Err(e) = unsafe { sys::cuGraphUpload(cu_graph_exec, stream.cu_stream()).result() } {
            unsafe {
                let _ = sys::cuGraphExecDestroy(cu_graph_exec).result();
                let _ = sys::cuGraphDestroy(cu_graph).result();
            }
            return Err(DeviceError::Driver(e));
        }

        // Synchronize.
        stream.synchronize()?;

        Ok(CudaGraph {
            stream: stream.clone(),
            cu_graph,
            cu_graph_exec,
            output: Some(()),
        })
    }
}

/// A graph-backed inference module.
///
/// Implementations own a [`CudaGraph`] captured at construction time.
/// Each call to [`forward`](Module::forward) updates the input buffer and
/// replays the graph, returning the result synchronously.
///
/// # Construction
///
/// Graph capture is model-specific and happens in the implementation's
/// constructor — not in the trait. A typical pattern:
///
/// ```rust,ignore
/// use cuda_async::prelude::*;
///
/// struct MyModel {
///     graph: CudaGraph<Arc<Tensor<f32>>>,
///     h_input: Tensor<f32>,
///     output: Arc<Tensor<f32>>,
/// }
///
/// impl MyModel {
///     fn new(stream: Arc<CudaStream>) -> Result<Self, DeviceError> {
///         let h_input = api::zeros(&[d]).sync_on(&stream)?;
///         let forward_op = build_forward(h_input.clone().into());
///         let mut graph = forward_op.graph_on(stream)?;
///         let output = graph.take_output().unwrap();
///         Ok(Self { graph, h_input, output })
///     }
/// }
///
/// impl Module for MyModel {
///     type Input = Arc<Tensor<f32>>;
///     type Output = Arc<Tensor<f32>>;
///
///     fn forward(&mut self, input: Self::Input)
///         -> Result<Self::Output, DeviceError>
///     {
///         self.graph.update(
///             api::memcpy(&mut self.h_input, &input)
///         )?;
///         self.graph.launch()?;
///         Ok(self.output.clone())
///     }
/// }
/// ```
///
/// # Future extensions
///
/// This trait covers the forward pass. Planned companion traits:
/// - `Backward` — gradient computation for autodiff
/// - `Parameterized` — access to learnable parameters for optimizers
pub trait Module {
    /// The input to the module (e.g., an embedding tensor).
    type Input: Send;
    /// The output of the module (e.g., logits or a hidden state).
    type Output: Send;

    /// Run the forward pass: update the input, launch the graph, return
    /// the result.
    fn forward(&mut self, input: Self::Input) -> Result<Self::Output, DeviceError>;
}
