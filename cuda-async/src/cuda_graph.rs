/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOp, ExecutionContext, GraphNode};
use crate::error::DeviceError;
use cuda_core::{sys, Device, IntoResult, Stream};
use std::future::IntoFuture;
use std::mem::MaybeUninit;
use std::sync::Arc;

const CU_STREAM_CAPTURE_MODE_RELAXED: sys::CUstreamCaptureMode = 2;

/// A captured and instantiated CUDA graph, ready for replay.
///
/// Created via [`CudaGraph::capture`] or [`CudaGraph::scope`]. The graph can
/// be replayed any number of times via [`launch`](CudaGraph::launch).
///
/// # Lifetime `'a`: captured buffers stay borrowed
///
/// The instantiated graph bakes in the **device pointers** of every buffer
/// the captured work touches; a replay reads and writes those exact
/// addresses. `'a` ties the graph to those buffers: everything the capture
/// closure (or op) borrowed stays borrowed for as long as the graph value is
/// used, so safe code cannot drop, move, or mutably alias a captured buffer
/// while a replay could still happen, and cannot touch any of them while a
/// [`GraphLaunch`] (which borrows the graph mutably) is in flight.
///
/// Outputs are only reachable after a completed replay: through the
/// reference [`GraphLaunch::sync`]/[`sync_on`](GraphLaunch::sync_on) return,
/// through [`outputs`](CudaGraph::outputs), or by consuming the graph with
/// [`into_inner`](CudaGraph::into_inner) — each synchronizes before handing
/// anything back.
///
/// # Examples
///
/// ```rust,ignore
/// use cuda_async::prelude::*;
///
/// // Build a lazy operation (no GPU work yet).
/// let forward_op = build_forward_pass(&model, &bufs);
///
/// // Capture: records the op's GPU work into a graph. Nothing has run yet.
/// let mut graph = CudaGraph::capture(stream.clone(), forward_op)?;
///
/// // Replay loop; the completed replay hands back the outputs.
/// for _ in 0..n_tokens {
///     // Optionally: graph.update(memcpy into a pre-allocated input) here.
///     let bufs = graph.launch().sync_on(&stream)?;
/// }
/// ```
pub struct CudaGraph<'a, T> {
    stream: Arc<Stream>,
    exec: Arc<GraphExecHandle>,
    output: T,
    /// Keeps every capture-time borrow alive while the graph is in use; see
    /// the type docs.
    _captured: std::marker::PhantomData<&'a mut ()>,
}

/// Owns an instantiated CUDA graph: the `CUgraph` it was instantiated from
/// and the `CUgraphExec` that replays it.
///
/// Shared (via `Arc`) between the [`CudaGraph`] and every [`GraphLaunch`] it
/// hands out, so a launch can never replay an exec that has already been
/// destroyed: the handles live until the last owner — graph or pending
/// launch — is dropped. `GraphLaunch` used to copy the raw `CUgraphExec`, so
/// `let l = graph.launch(); drop(graph); l.sync()` launched a destroyed exec.
struct GraphExecHandle {
    /// Bound before destruction: the driver needs the owning context current.
    device: Arc<Device>,
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
}

// SAFETY: both fields are opaque driver handles that the CUDA driver
// synchronizes internally; `cuGraphLaunch` may be issued from any thread. The
// only mutation is the destroy in `Drop`, which runs exactly once, after the
// last `Arc` owner is gone.
unsafe impl Send for GraphExecHandle {}
unsafe impl Sync for GraphExecHandle {}

impl GraphExecHandle {
    /// Instantiates `cu_graph` and pre-uploads the exec on `stream`. Takes
    /// ownership of `cu_graph` on every path: it is destroyed if
    /// instantiation fails, and owned by the returned handle otherwise
    /// (whose `Drop` also covers an upload failure).
    fn instantiate(
        device: Arc<Device>,
        cu_graph: sys::CUgraph,
        stream: &Stream,
    ) -> Result<Self, DeviceError> {
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
        let handle = Self {
            device,
            cu_graph,
            cu_graph_exec,
        };
        // Upload (pre-stages graph resources on the device). On failure
        // `handle` drops here and destroys both objects.
        unsafe { sys::cuGraphUpload(handle.cu_graph_exec, stream.cu_stream()).result()? };
        Ok(handle)
    }
}

impl Drop for GraphExecHandle {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        if !self.cu_graph_exec.is_null() {
            let _ = unsafe { sys::cuGraphExecDestroy(self.cu_graph_exec).result() };
        }
        if !self.cu_graph.is_null() {
            let _ = unsafe { sys::cuGraphDestroy(self.cu_graph).result() };
        }
    }
}

/// Runs `record` with `stream` in (relaxed) capture mode and turns the
/// recorded work into an instantiated, uploaded graph.
///
/// `cuStreamEndCapture` runs on every path — success, a `record` error, and
/// (before the panic resumes) a panic inside `record` — so a failure can
/// never leave the stream stuck in capture mode, and a graph handle produced
/// on a failing path is destroyed rather than leaked.
///
/// The caller must hold the execution lock: recording executes a `DeviceOp`.
fn capture_on<R>(
    stream: &Arc<Stream>,
    record: impl FnOnce() -> Result<R, DeviceError>,
) -> Result<(R, Arc<GraphExecHandle>), DeviceError> {
    let device = stream.device().clone();
    device.bind_to_thread()?;

    unsafe {
        stream.begin_capture(CU_STREAM_CAPTURE_MODE_RELAXED)?;
    }

    let recorded = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(record)) {
        Ok(result) => result,
        Err(payload) => {
            if let Ok(cu_graph) = unsafe { stream.end_capture() } {
                destroy_graph(cu_graph);
            }
            std::panic::resume_unwind(payload);
        }
    };

    // End capture — must happen regardless of the record result.
    let end_result = unsafe { stream.end_capture() };

    // Handle the (recorded, end_result) matrix, cleaning up on failure.
    let (output, cu_graph) = match (recorded, end_result) {
        (Err(err), Ok(cu_graph)) => {
            destroy_graph(cu_graph);
            return Err(err);
        }
        (Err(err), Err(_)) => return Err(err),
        (Ok(_), Err(capture_err)) => return Err(DeviceError::Driver(capture_err)),
        (Ok(_), Ok(cu_graph)) if cu_graph.is_null() => {
            return Err(DeviceError::Internal(
                "cuStreamEndCapture returned null graph".into(),
            ));
        }
        (Ok(output), Ok(cu_graph)) => (output, cu_graph),
    };

    let exec = GraphExecHandle::instantiate(device, cu_graph, stream)?;

    // Drain the upload. The recorded work itself has *not* run.
    unsafe { stream.synchronize() }?;

    Ok((output, Arc::new(exec)))
}

fn destroy_graph(cu_graph: sys::CUgraph) {
    if !cu_graph.is_null() {
        let _ = unsafe { sys::cuGraphDestroy(cu_graph).result() };
    }
}

impl<'a, T: Send> CudaGraph<'a, T> {
    /// Capture a [`DeviceOp`] into a replayable CUDA graph.
    ///
    /// Runs `op` once on `stream` in capture mode. All GPU work (kernel
    /// launches, memcpys, etc.) issued by the operation is *recorded* into a
    /// graph, not executed. The graph is then instantiated and uploaded.
    ///
    /// The `op: 'a` bound keeps every buffer the op borrows alive and
    /// exclusively held for as long as the graph is used (see the type
    /// docs). The output `T` — typically the op's recovered buffers — is
    /// held by the graph and only handed out after a completed replay
    /// ([`GraphLaunch::sync`], [`outputs`](CudaGraph::outputs)) or when the
    /// graph is consumed ([`into_inner`](CudaGraph::into_inner)).
    ///
    /// Capture holds the thread-local execution lock for the duration of
    /// `op`, so nested `.sync()` / `.sync_on()` / `.await` inside it return
    /// the non-reentrant error (see [`Scope`]). If `op` fails or panics, the
    /// capture is ended and the stream left usable before the error or panic
    /// propagates.
    pub fn capture(
        stream: Arc<Stream>,
        op: impl DeviceOp<Output = T> + 'a,
    ) -> Result<Self, DeviceError> {
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let exec_ctx = ExecutionContext::new(stream.clone());
        let (output, exec) = capture_on(&stream, || unsafe { op.execute(&exec_ctx) })?;
        Ok(Self {
            stream,
            exec,
            output,
            _captured: std::marker::PhantomData,
        })
    }

    /// Enqueue a [`GraphNode`] on the graph's stream without synchronizing.
    ///
    /// Use this to refresh graph inputs — typically a memcpy into a
    /// pre-allocated buffer the graph reads — before [`launch`](CudaGraph::launch).
    ///
    /// # Ordering
    ///
    /// The operation is issued on [`stream`](CudaGraph::stream), so it
    /// completes before any *later work on that same stream*. That includes
    /// a launch executed on the graph's stream:
    /// `graph.launch().sync_on(graph.stream())`. A launch scheduled anywhere
    /// else — `.sync()`, `.await`, or `.sync_on(&other_stream)` — is **not**
    /// ordered after the update.
    ///
    /// # Why the bounds
    ///
    /// `update` runs a `DeviceOp` without waiting for it, which is only
    /// sound when nothing can observe the operation early: the op must be a
    /// [`GraphNode`] (it neither allocates nor frees device memory) and must
    /// produce no output. Ops with an output, or ops that are not
    /// `GraphNode`, are rejected at compile time:
    ///
    /// ```compile_fail,E0271
    /// use cuda_async::cuda_graph::CudaGraph;
    /// use cuda_async::device_operation::value;
    ///
    /// fn reject_output(graph: &CudaGraph<()>) {
    ///     let _ = graph.update(value(5));
    /// }
    /// ```
    ///
    /// ```compile_fail,E0277
    /// use cuda_async::cuda_graph::CudaGraph;
    /// use cuda_async::device_operation::{value, DeviceOp};
    ///
    /// fn reject_non_graph_node(graph: &CudaGraph<()>) {
    ///     let _ = graph.update(value(()).boxed());
    /// }
    /// ```
    ///
    /// Like `sync_on`, this takes the thread-local execution lock.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Copy a new embedding into the graph's pre-allocated input buffer.
    /// graph.update(api::memcpy(&mut h_input, &new_embedding))?;
    /// graph.launch().sync_on(graph.stream())?;
    /// ```
    pub fn update<N>(&self, op: N) -> Result<(), DeviceError>
    where
        N: GraphNode + DeviceOp<Output = ()>,
    {
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let ctx = ExecutionContext::new(self.stream.clone());
        // SAFETY: the op is a `GraphNode` (no alloc/free) with no output, so
        // nothing it produces can be read before the stream is synchronized.
        unsafe { op.execute(&ctx) }
    }

    /// Return a [`DeviceOp`] that replays the captured graph.
    ///
    /// The graph launches on whichever stream the returned op is executed
    /// on. Use the standard [`DeviceOp`] methods to control execution:
    ///
    /// ```rust,ignore
    /// graph.launch().sync_on(&stream)?;          // explicit stream, blocking
    /// graph.launch().sync()?;                    // default policy, blocking
    /// graph.launch().then(next_op).sync()?;      // compose with other ops
    /// ```
    ///
    /// The launch shares ownership of the instantiated graph, so it stays
    /// valid even if the `CudaGraph` is dropped first.
    ///
    /// Operations issued via [`update`](CudaGraph::update) are guaranteed to
    /// complete before the graph runs **only when the launch is executed on
    /// [`stream`](CudaGraph::stream)** (same-stream ordering); on any other
    /// stream there is no ordering between them.
    pub fn launch(&mut self) -> GraphLaunch<'_, 'a, T> {
        GraphLaunch { graph: self }
    }

    /// Like [`update`](CudaGraph::update), for an input buffer the graph
    /// *owns* (part of its output handle `T`): `build` receives `&mut T`
    /// and returns the refresh op — typically a memcpy into a pre-allocated
    /// input — which is enqueued on the graph's stream without
    /// synchronizing. Taking `&mut self` guarantees no replay is in flight
    /// while the host-side handle is being used to build the op; the
    /// enqueued work itself is ordered before any later replay on
    /// [`stream`](CudaGraph::stream), exactly like `update`.
    pub fn update_with<'s, N, F>(&'s mut self, build: F) -> Result<(), DeviceError>
    where
        F: FnOnce(&'s mut T) -> N,
        N: GraphNode + DeviceOp<Output = ()> + 's,
    {
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let ctx = ExecutionContext::new(self.stream.clone());
        let op = build(&mut self.output);
        // SAFETY: as in `update` — a `GraphNode` with no output, enqueued on
        // the graph's stream; nothing observes it before stream order does.
        unsafe { op.execute(&ctx) }
    }

    /// Replay once on the graph's own stream, block until complete, and
    /// return the outputs. Shorthand for
    /// `graph.launch().sync_on(graph.stream())`, which the borrow checker
    /// cannot express directly (the launch borrows the graph mutably).
    pub fn replay(&mut self) -> Result<&mut T, DeviceError> {
        let stream = self.stream.clone();
        self.launch().sync_on(&stream)
    }

    /// The captured outputs, after synchronizing the graph's stream.
    ///
    /// The synchronize guarantees no replay (or [`update`](CudaGraph::update))
    /// issued on [`stream`](CudaGraph::stream) is still writing the buffers
    /// behind `T` when the reference is handed out. A replay executed on a
    /// *different* stream is not ordered by this call — complete it through
    /// its own [`GraphLaunch::sync`]/[`sync_on`](GraphLaunch::sync_on)
    /// instead (the launch's `&mut` borrow of the graph already prevents
    /// calling this while one is pending).
    pub fn outputs(&mut self) -> Result<&mut T, DeviceError> {
        unsafe { self.stream.synchronize()? };
        Ok(&mut self.output)
    }

    /// Consume the graph: synchronize its stream, destroy nothing that a
    /// pending launch still shares, and return the captured outputs. This
    /// ends the graph's borrow of every captured buffer.
    pub fn into_inner(self) -> Result<T, DeviceError> {
        unsafe { self.stream.synchronize()? };
        Ok(self.output)
    }

    /// Returns a reference to the stream this graph was captured on.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

/// One replay of a captured CUDA graph.
///
/// Created by [`CudaGraph::launch`]. Borrows the graph **mutably** for the
/// replay's duration, so while a replay is pending nothing else can touch
/// the graph, its outputs, or (through the graph's `'a`) any captured
/// buffer.
///
/// Completing the replay hands the outputs back:
/// [`sync`](GraphLaunch::sync) / [`sync_on`](GraphLaunch::sync_on) return
/// `&mut T` tied to the graph borrow, valid only after the synchronize.
/// The launch also composes as a [`DeviceOp`] with `Output = ()` —
/// `graph.launch().then(op)`, `.await` — which never exposes `T`; read the
/// outputs afterwards via [`CudaGraph::outputs`], which synchronizes first.
pub struct GraphLaunch<'g, 'a, T> {
    graph: &'g mut CudaGraph<'a, T>,
}

impl<'g, 'a, T: Send> GraphLaunch<'g, 'a, T> {
    /// Replay on `stream`, block until complete, and return the outputs.
    ///
    /// Mirrors [`DeviceOp::sync_on`], with one addition: the completed
    /// replay yields `&mut T`, the graph's captured outputs, which cannot be
    /// reached before this synchronize returns.
    pub fn sync_on(self, stream: &Arc<Stream>) -> Result<&'g mut T, DeviceError> {
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        unsafe {
            sys::cuGraphLaunch(self.graph.exec.cu_graph_exec, stream.cu_stream()).result()?;
            stream.synchronize()?;
        }
        Ok(&mut self.graph.output)
    }

    /// Replay on a policy-chosen stream, block until complete, and return
    /// the outputs. See [`sync_on`](GraphLaunch::sync_on).
    pub fn sync(self) -> Result<&'g mut T, DeviceError> {
        let stream = with_default_device_policy(|policy| policy.next_stream())??;
        self.sync_on(&stream)
    }
}

impl<'g, 'a, T: Send> DeviceOp for GraphLaunch<'g, 'a, T> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> Result<(), DeviceError> {
        sys::cuGraphLaunch(
            self.graph.exec.cu_graph_exec,
            context.get_cuda_stream().cu_stream(),
        )
        .result()?;
        Ok(())
    }
}

impl<'g, 'a, T: Send> IntoFuture for GraphLaunch<'g, 'a, T> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), GraphLaunch<'g, 'a, T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            let mut f = DeviceFuture::new();
            f.device_operation = Some(self);
            f.execution_context = Some(ExecutionContext::new(stream));
            Ok(f)
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
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
/// graph.launch().sync_on(&stream)?;
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
/// 3. **Other executions inside the closure are rejected.** The scope
///    holds the thread-local execution lock for the duration of the
///    closure, so `op.sync_on(&other_stream)`, `op.sync()`, and
///    `op.sync_on(&capture_stream)` all return the non-reentrant
///    `DeviceError` before reaching the driver. Nothing executes eagerly
///    alongside the recording. (Independently, CUDA itself rejects
///    synchronizing or querying a capturing stream with
///    `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.) The one way around the
///    lock is a [`then_unchecked`](DeviceOp::then_unchecked) chain recorded
///    via `record`, whose closure may execute eagerly — that is exactly the
///    caller's `unsafe` assertion.
///
/// 4. **Borrow checker enforces `&mut` exclusivity.** `record` consumes
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
/// ## Lifetimes make replay safe, not just capture
///
/// Capture-time ordering alone is not enough: the instantiated graph bakes
/// in the *device pointers* of every recorded buffer, and a replay
/// dereferences them long after `record`'s borrows have ended. The
/// remaining obligations are carried by [`CudaGraph<'a, T>`]'s lifetime:
///
/// 1. **No use-after-free.** `scope`'s closure is bound `F: 'a`, so every
///    buffer it captures outlives `'a`; the returned graph keeps `'a` alive
///    at each of its uses. Dropping or moving a captured buffer while the
///    graph can still replay is a borrow-check error.
/// 2. **No aliased writes between replays.** The same borrow covers host or
///    other-stream writes: taking `&mut` to a captured buffer while the
///    graph lives is rejected.
/// 3. **Nothing touches the buffers during a replay.** A replay borrows the
///    graph mutably ([`CudaGraph::launch`] takes `&mut self`), so no other
///    graph operation — including output reads — can overlap it.
/// 4. **Outputs only after completion.** The closure's return value is held
///    by the graph and handed out only through paths that synchronize
///    first: [`GraphLaunch::sync`]/[`sync_on`](GraphLaunch::sync_on),
///    [`CudaGraph::outputs`], [`CudaGraph::into_inner`].
///
/// Point 1 covers buffers the closure captures by reference. A buffer the
/// closure *owns* (a pre-allocated tensor moved into a `move` closure) must
/// be returned as part of `T`: the graph then owns it and the same
/// guarantees apply. An owned buffer that is recorded against and then
/// dropped when the closure returns is freed while the graph still holds
/// its device address — the borrow checker cannot see that relationship,
/// and a later replay is a use-after-free.
///
/// One boundary remains the caller's: a replay awaited through the
/// `DeviceOp` composition path and *abandoned before completion* (the
/// future dropped mid-flight) leaves GPU work running that the borrow
/// system can no longer see once the graph itself is dropped. The shared
/// exec handle keeps the graph objects alive for a pending launch, but the
/// captured buffers' liveness then rests on the general stream-ordering
/// contract of [`DeviceOp`], as with every other operation in this crate.
///
/// # What happens if you call other operations inside the closure
///
/// While `s.record(op)` is the intended API, other operations inside
/// the closure have well-defined behavior:
///
/// | Operation | What happens |
/// |---|---|
/// | `op.sync_on(&capture_stream)` | Non-reentrant execution-lock error; nothing executes |
/// | `op.sync_on(&other_stream)` | Non-reentrant execution-lock error; nothing executes |
/// | `op.sync()` / `op.await` | Non-reentrant execution-lock error; nothing executes |
///
/// These are all defined behavior but serve no purpose inside a graph
/// capture scope — use `s.record(op)` instead.
///
/// # Thread safety
///
/// `Scope` is `!Send` — it cannot escape to another thread.
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
    pub fn record<T, N>(&self, op: N) -> Result<T, DeviceError>
    where
        T: Send,
        N: GraphNode + DeviceOp<Output = T>,
    {
        // SAFETY: The scope's stream is in capture mode. No GPU work
        // executes — ops are recorded as graph nodes. The GraphNode bound
        // ensures no alloc/free ops are recorded. See Scope docs for
        // the full safety proof.
        unsafe { op.execute(&self.ctx) }
    }
}

impl<'a, T: Send> CudaGraph<'a, T> {
    /// Capture a CUDA graph using a scoped closure.
    ///
    /// The closure receives a [`Scope`] for recording operations. Each
    /// `s.record(op)` records a graph node and consumes the op, releasing
    /// borrows *within the scope*. A buffer written by one `record` call can
    /// be read by the next.
    ///
    /// The closure's return value `T` becomes the graph's output handle —
    /// typically the recovered output buffers of the last `record` (or `()`
    /// when the caller keeps its own handles). It is only reachable after a
    /// completed replay; see [`CudaGraph`].
    ///
    /// The `F: 'a` bound is the lifetime half of the safety argument: every
    /// reference the closure captures must outlive `'a`, and the returned
    /// `CudaGraph<'a, T>` keeps `'a` alive for as long as the graph is used
    /// — so the buffers whose device pointers the graph baked in can be
    /// neither dropped, moved, nor mutably reborrowed while a replay can
    /// still happen. Pre-allocate all buffers before calling this method.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut output = api::zeros::<f32>(&[d]).sync_on(&stream)?;
    /// let weights = api::ones::<f32>(&[d]).sync_on(&stream)?;
    ///
    /// let mut graph = CudaGraph::scope(&stream, |s| {
    ///     s.record(kernel1((&mut output).partition([128]), &weights))?;
    ///     s.record(kernel2((&mut output).partition([64]), &weights))?;
    ///     Ok(())
    /// })?;
    ///
    /// graph.launch().sync_on(&stream)?;
    /// // `output` is borrowed until `graph`'s last use.
    /// ```
    ///
    /// See [`Scope`] for the capture-time safety proof and edge-case
    /// behavior.
    pub fn scope<F>(stream: &Arc<Stream>, f: F) -> Result<Self, DeviceError>
    where
        F: FnOnce(&Scope) -> Result<T, DeviceError> + 'a,
    {
        // The guard releases the lock on return and on unwind; `capture_on`
        // ends the capture itself before a panic propagates.
        let _execution_lock = crate::device_operation::acquire_execution_lock()?;
        let scope = Scope {
            ctx: ExecutionContext::new(stream.clone()),
            _not_send: std::marker::PhantomData,
        };
        let (output, exec) = capture_on(stream, || f(&scope))?;
        Ok(CudaGraph {
            stream: stream.clone(),
            exec,
            output,
            _captured: std::marker::PhantomData,
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
///     // The captured op owns `Arc` clones of its buffers: `'static`.
///     graph: CudaGraph<'static, Arc<Tensor<f32>>>,
///     h_input: Tensor<f32>,
/// }
///
/// impl MyModel {
///     fn new(stream: Arc<Stream>) -> Result<Self, DeviceError> {
///         let h_input = api::zeros(&[d]).sync_on(&stream)?;
///         let forward_op = build_forward(h_input.clone().into());
///         let graph = forward_op.graph_on(stream)?;
///         Ok(Self { graph, h_input })
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
///         let output = self.graph.replay()?;
///         Ok(output.clone())
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
