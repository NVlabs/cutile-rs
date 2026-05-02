/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lazy, composable GPU operations and combinator types.

use crate::device_context::{pool_for_stream, with_default_device_policy};
use crate::device_future::DeviceFuture;
use crate::error::{device_error, DeviceError};
use crate::scheduling_policies::SchedulingPolicy;
use cuda_core::{Device, MemPool, Stream};
use std::cell::{Cell, UnsafeCell};
use std::fmt::Debug;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ── Thread-local execution guard ───────────────────────────────────────────
//
// Invariant: On any given thread, only one DeviceOp may be executing at a time.
//
// This prevents CUDA data races from nested execution (e.g., calling
// sync_on(&other_stream) inside a `then` closure with in-flight tensors).

thread_local! {
    static DEVICE_OP_EXECUTING: Cell<bool> = const { Cell::new(false) };
}

/// Acquire the thread-local execution lock. Returns an error if another
/// DeviceOp is already executing on this thread.
pub(crate) fn acquire_execution_lock() -> Result<(), DeviceError> {
    DEVICE_OP_EXECUTING.with(|flag| {
        if flag.get() {
            Err(DeviceError::Internal(
                "DeviceOp execution is non-reentrant: another DeviceOp is already \
                 executing on this thread. If this is intentional and you have \
                 verified there are no cross-stream data races, use \
                 `then_unchecked`."
                    .into(),
            ))
        } else {
            flag.set(true);
            Ok(())
        }
    })
}

/// Release the thread-local execution lock.
pub(crate) fn release_execution_lock() {
    DEVICE_OP_EXECUTING.with(|flag| {
        flag.set(false);
    });
}

pub type DeviceOrdinal = usize;

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    ordinal: DeviceOrdinal,
    cuda_stream: Arc<Stream>,
    device: Arc<Device>,
    pool: Option<Arc<MemPool>>,
}

impl ExecutionContext {
    pub fn new(cuda_stream: Arc<Stream>) -> Self {
        let device = cuda_stream.device().clone();
        let ordinal = device.ordinal();
        let pool = pool_for_stream(&cuda_stream);
        Self {
            cuda_stream,
            device,
            ordinal,
            pool,
        }
    }
    pub fn get_cuda_stream(&self) -> &Arc<Stream> {
        &self.cuda_stream
    }
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    pub fn get_device_id(&self) -> DeviceOrdinal {
        self.ordinal
    }
    pub fn get_pool(&self) -> Option<&Arc<MemPool>> {
        self.pool.as_ref()
    }
    /// Allocates device memory on this context's stream, using the custom pool if set.
    ///
    /// # Safety
    /// The stream must be valid and not destroyed.
    pub unsafe fn alloc_async(&self, num_bytes: usize) -> cuda_core::sys::CUdeviceptr {
        match &self.pool {
            Some(pool) => cuda_core::malloc_from_pool_async(num_bytes, pool, &self.cuda_stream),
            None => cuda_core::malloc_async(num_bytes, &self.cuda_stream),
        }
    }
    #[expect(
        dead_code,
        reason = "kept for direct synchronous execution in tests and future blocking APIs"
    )]
    fn execute<T: Send>(&self, op: impl DeviceOp<Output = T>) -> Result<T, DeviceError> {
        unsafe {
            // Safety: ExecutionContext is only available within a DeviceOp closure.
            // DeviceOp closures can only be converted into DeviceFuture
            // which synchronizes device operations with the host thread via a host callback.
            op.execute(self)
        }
    }
}

/// A lazy, composable GPU operation that may be executed synchronously or asynchronously on a CUDA device.
///
/// `DeviceOp` represents a resource-agnostic computation that will be scheduled and executed.
/// The actual execution resource (stream, device, host machine, cluster, etc.) is determined when the
/// operation is either executed or converted into a future.
/// Device operations are lazy - they don't execute until synchronously executed, or a corresponding
/// future is awaited upon. Multiple operations can be composed together before execution,
/// enabling efficient streaming of GPU work.
///
/// # Scheduling and Stream Assignment
///
/// How an operation reaches the GPU depends on which method you use:
///
/// | Method              | Stream chosen by                      | Blocks thread?      |
/// |---------------------|---------------------------------------|---------------------|
/// | `.await`            | Default device's [`SchedulingPolicy`] | No (suspends task)  |
/// | `.sync()`           | Default device's [`SchedulingPolicy`] | Yes                 |
/// | `.sync_on(&stream)` | The explicit `stream` you provide     | Yes                 |
/// | `.into_future()`    | Default device's [`SchedulingPolicy`] | No (returns future) |
/// | `.schedule(policy)` | The `policy` you provide              | No (returns future) |
///
/// With the default [`StreamPoolRoundRobin`](crate::scheduling_policies::StreamPoolRoundRobin) policy (4 streams), consecutive `.await` or
/// `.sync()` calls rotate through streams, so independent operations can overlap on the GPU.
/// Operations chained with [`.then()`](DeviceOp::then) share a single stream
/// and always execute in order.
///
/// See [`SchedulingPolicy`] for a full explanation of ordering guarantees.
///
/// # Safety
///
/// The `execute` method is unsafe because it's asynchronous - the GPU may still be writing to
/// memory allocated by the output after `execute` returns. Converting a `DeviceOp` into
/// a `DeviceFuture` ensures memory operations complete before the output can be accessed.
///
/// ## Examples
///
/// ```rust,ignore
/// use cuda_async::device_operation::{DeviceOp, value};
///
/// // Create a simple value operation
/// let op1 = value(42);
///
/// // Chain operations together
/// let op2 = op1.then(|x| value(x * 2));
///
/// // Execute synchronously (blocks until GPU completes)
/// let result = op2.sync().expect("Device operation failed."); // returns 84
/// ```
///
/// ```rust,ignore
/// use cuda_async::device_operation::{DeviceOp, zip};
/// use cutile::api;
///
/// // Compose multiple tensor operations
/// let x = api::zeros(&[64, 64]);
/// let y = api::ones(&[64, 64]);
/// let combined = zip!(x, y).then(|(x, y)| {
///     // Both tensors are ready here
///     value((x, y))
/// });
/// ```
///
/// ## Async Usage
///
/// Operations automatically implement `IntoFuture`, enabling use with `.await`:
///
/// ```rust,ignore
/// let x: Arc<Tensor<f32>> = api::randn(0.0, 1.0, &[100, 100]).await?.into();
/// let y = some_kernel(x.clone()).await?;
/// ```
pub trait DeviceOp:
    Send + Sized + IntoFuture<Output = Result<<Self as DeviceOp>::Output, DeviceError>>
{
    type Output: Send;

    // Consumes DeviceOp and executes the implementing operation.
    // This is unsafe because it is asynchronous: A device may be writing to memory allocated
    // by the output.
    // Converting DeviceOp into a DeviceFuture ensures any memory operations are complete
    // before the output can be accessed by the async runtime.
    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError>;
    /// Schedule this operation on a specific policy and return a [`DeviceFuture`].
    fn schedule(
        self,
        policy: &Arc<dyn SchedulingPolicy>,
    ) -> Result<DeviceFuture<<Self as DeviceOp>::Output, Self>, DeviceError> {
        let stream = policy.next_stream()?;
        let mut future = DeviceFuture::new();
        future.device_operation = Some(self);
        future.execution_context = Some(ExecutionContext::new(stream));
        Ok(future)
    }
    /// Chain a follow-up operation that runs **on the same stream** as `self`.
    ///
    /// Because both operations share a stream, `f` is guaranteed to see `self`'s output
    /// fully written. This is the recommended way to express data dependencies without
    /// manual synchronization.
    ///
    /// The closure must not execute other DeviceOps (e.g., via `sync_on` or `sync`).
    /// This is enforced at runtime by the thread-local execution lock — attempting
    /// nested execution will return a `DeviceError`.
    fn then<O: Send, DO, F>(self, f: F) -> AndThen<<Self as DeviceOp>::Output, Self, O, DO, F>
    where
        DO: DeviceOp<Output = O>,
        F: FnOnce(<Self as DeviceOp>::Output) -> DO,
    {
        AndThen {
            op: self,
            closure: f,
        }
    }
    /// Like [`then`](DeviceOp::then), but without the thread-local execution lock.
    ///
    /// # Safety
    ///
    /// The closure must not submit GPU work to any stream other than the
    /// chain's stream using tensors from the output. Violating this causes
    /// CUDA data races (undefined behavior).
    unsafe fn then_unchecked<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThen<<Self as DeviceOp>::Output, Self, O, DO, F>
    where
        DO: DeviceOp<Output = O>,
        F: FnOnce(<Self as DeviceOp>::Output) -> DO,
    {
        AndThen {
            op: self,
            closure: f,
        }
    }
    /// Transform the output of this operation without issuing new GPU work.
    fn map<O: Send, F>(
        self,
        f: F,
    ) -> AndThen<
        <Self as DeviceOp>::Output,
        Self,
        O,
        Value<O>,
        impl FnOnce(<Self as DeviceOp>::Output) -> Value<O> + Send,
    >
    where
        F: FnOnce(<Self as DeviceOp>::Output) -> O + Send,
    {
        self.then(move |x| value(f(x)))
    }
    /// Peek at the output for debugging without consuming or transforming it.
    fn inspect<F>(
        self,
        f: F,
    ) -> AndThen<
        <Self as DeviceOp>::Output,
        Self,
        <Self as DeviceOp>::Output,
        Value<<Self as DeviceOp>::Output>,
        impl FnOnce(<Self as DeviceOp>::Output) -> Value<<Self as DeviceOp>::Output> + Send,
    >
    where
        F: FnOnce(&<Self as DeviceOp>::Output) + Send,
    {
        self.map(move |x| {
            f(&x);
            x
        })
    }
    fn and_then_with_context<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThenWithContext<<Self as DeviceOp>::Output, Self, O, DO, F>
    where
        DO: DeviceOp<Output = O>,
        F: FnOnce(&ExecutionContext, <Self as DeviceOp>::Output) -> DO,
    {
        AndThenWithContext {
            op: self,
            closure: f,
        }
    }
    /// Type-erase this operation into a [`BoxedDeviceOp`].
    ///
    /// This allows heterogeneous collections of operations that share the same
    /// output type but differ in their concrete type (e.g. mixing `Value`,
    /// `SelectLeft`, etc. in a single `Vec`).
    fn boxed(self) -> BoxedDeviceOp<<Self as DeviceOp>::Output>
    where
        Self: 'static,
    {
        BoxedDeviceOp {
            inner: Box::new(move |ctx| unsafe { self.execute(ctx) }),
        }
    }
    /// Convert into a cloneable, execute-once operation.
    ///
    /// The underlying op executes at most once; every clone gets `Arc::clone()`
    /// of the cached result. Follows the `FutureExt::shared()` convention.
    fn shared(self) -> SharedDeviceOp<<Self as DeviceOp>::Output>
    where
        Self: 'static,
        <Self as DeviceOp>::Output: Sync,
    {
        SharedDeviceOp {
            inner: Arc::new(SharedExec {
                computed: AtomicBool::new(false),
                op: UnsafeCell::new(Some(Box::new(move |ctx: &ExecutionContext| unsafe {
                    self.execute(ctx)
                }))),
                result: UnsafeCell::new(None),
            }),
        }
    }
    /// Capture this operation into a replayable [`CudaGraph`](crate::cuda_graph::CudaGraph)
    /// using the default device's scheduling policy to pick a stream.
    fn graph(
        self,
    ) -> Result<crate::cuda_graph::CudaGraph<<Self as DeviceOp>::Output>, DeviceError> {
        with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            self.graph_on(stream)
        })?
    }
    /// Capture this operation into a replayable [`CudaGraph`](crate::cuda_graph::CudaGraph)
    /// on an **explicit stream**.
    ///
    /// Executes the operation once on `stream` in capture mode, recording
    /// all GPU work. Returns a `CudaGraph<Self::Output>` containing the
    /// replayable graph and the initial output.
    fn graph_on(
        self,
        stream: Arc<Stream>,
    ) -> Result<crate::cuda_graph::CudaGraph<<Self as DeviceOp>::Output>, DeviceError> {
        crate::cuda_graph::CudaGraph::capture(stream, self)
    }
    /// Execute synchronously using the default device's scheduling policy.
    ///
    /// The policy picks a stream (round-robin by default), submits the work, and blocks
    /// until the GPU finishes. Equivalent to `.await` but blocking.
    fn sync(self) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            self.sync_on(&stream)
        })?
    }
    /// Execute on a stream without synchronizing. The GPU may still be
    /// writing to the output when this returns.
    ///
    /// # Safety
    ///
    /// The caller must ensure the stream is synchronized before accessing
    /// GPU data from the output.
    unsafe fn async_on(
        self,
        stream: &Arc<Stream>,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let ctx = ExecutionContext::new(stream.clone());
        unsafe { self.execute(&ctx) }
    }
    /// Execute on an **explicit stream** and block until the GPU finishes.
    ///
    /// This bypasses the scheduling policy entirely. All operations `sync_on` the same
    /// stream are guaranteed to execute in call order. Use this when you need deterministic
    /// ordering or are debugging concurrency issues.
    fn sync_on(self, stream: &Arc<Stream>) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        acquire_execution_lock()?;
        let ctx = ExecutionContext::new(stream.clone());
        let res = unsafe { self.execute(&ctx) };
        let sync_res = unsafe { stream.synchronize() };
        release_execution_lock();
        sync_res?;
        res
    }
}

// ── GraphNode ────────────────────────────────────────────────────────────────

/// Marker trait for [`DeviceOp`]s that are safe to record in a CUDA graph.
///
/// Only operations that do **not** allocate or free device memory should
/// implement this trait. During CUDA graph capture, allocation nodes may
/// return different addresses on replay, breaking baked-in pointers.
///
/// Implementors:
/// - Macro-generated kernel launchers (kernel launch only)
/// - Memcpy operations between pre-allocated buffers
/// - [`Value<T>`] (no GPU work)
///
/// Non-implementors (allocate device memory):
/// - `api::zeros`, `api::ones`, `api::full`, `api::arange`
/// - `api::randn`, `api::rand`
/// - `dup`, `copy_host_vec_to_device`
///
/// See [`Scope`](crate::cuda_graph::Scope) for the full safety proof.
pub trait GraphNode {}

// Arc

// Boxed (type-erased) DeviceOp

/// A type-erased [`DeviceOp`] that boxes the execution closure.
///
/// Created via [`DeviceOp::boxed()`].
/// Useful when you need to store operations with different concrete types but
/// the same `Output` in a homogeneous collection (e.g.
/// `Vec<BoxedDeviceOp<'_, T>>`).
pub struct BoxedDeviceOp<T: Send> {
    inner: Box<dyn FnOnce(&ExecutionContext) -> Result<T, DeviceError> + Send>,
}

impl<T: Send> DeviceOp for BoxedDeviceOp<T> {
    type Output = T;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<T, DeviceError> {
        (self.inner)(context)
    }
}

impl<T: Send> IntoFuture for BoxedDeviceOp<T> {
    type Output = Result<T, DeviceError>;
    type IntoFuture = DeviceFuture<T, Self>;
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

// Shared (cloneable, execute-once) DeviceOp

struct SharedExec<T: Send + Sync> {
    computed: AtomicBool,
    op: UnsafeCell<Option<Box<dyn FnOnce(&ExecutionContext) -> Result<T, DeviceError> + Send>>>,
    result: UnsafeCell<Option<Arc<T>>>,
}

// TODO (hme): document safety
unsafe impl<T: Send + Sync> Send for SharedExec<T> {}
unsafe impl<T: Send + Sync> Sync for SharedExec<T> {}

/// A cloneable, execute-once [`DeviceOp`].
///
/// Created via [`DeviceOp::shared()`]. The underlying operation executes at most
/// once; every clone gets `Arc::clone()` of the cached result. Follows the
/// `FutureExt::shared()` convention from the `futures` crate.
///
/// Output is always `Arc<T>` — the result is wrapped on first execution and
/// shared via refcount thereafter.
pub struct SharedDeviceOp<T: Send + Sync> {
    inner: Arc<SharedExec<T>>,
}

impl<T: Send + Sync> Clone for SharedDeviceOp<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Send + Sync> DeviceOp for SharedDeviceOp<T> {
    type Output = Arc<T>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Arc<T>, DeviceError> {
        if !self.inner.computed.load(Ordering::Acquire) {
            let op = unsafe { (&mut *self.inner.op.get()).take() }.ok_or(DeviceError::Internal(
                "SharedDeviceOp: operation already taken".to_string(),
            ))?;
            let result = op(context)?;
            unsafe {
                *self.inner.result.get() = Some(Arc::new(result));
            }
            self.inner.computed.store(true, Ordering::Release);
        }
        Ok(unsafe { (&*self.inner.result.get()).as_ref().unwrap().clone() })
    }
}

impl<T: Send + Sync> IntoFuture for SharedDeviceOp<T> {
    type Output = Result<Arc<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Arc<T>, SharedDeviceOp<T>>;
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

/// Create a pre-computed [`SharedDeviceOp`] from an existing `Arc<T>`.
///
/// The returned op is already "executed" — cloning it just bumps the refcount.
pub fn shared<T: Send + Sync>(val: Arc<T>) -> SharedDeviceOp<T> {
    SharedDeviceOp {
        inner: Arc::new(SharedExec {
            computed: AtomicBool::new(true),
            op: UnsafeCell::new(None),
            result: UnsafeCell::new(Some(val)),
        }),
    }
}

// IntoDeviceOp — convert plain values or existing DeviceOps into DeviceOp

/// Conversion trait that accepts both plain values and existing [`DeviceOp`]s.
///
/// The blanket impl covers all `DeviceOp` types (pass-through). Specific impls
/// cover plain data types (`f32`, `Arc<Tensor<T>>`, etc.) that wrap via [`Value`].
pub trait IntoDeviceOp<T: Send> {
    type Op: DeviceOp<Output = T>;
    fn into_op(self) -> Self::Op;
}

impl<T: Send, DO: DeviceOp<Output = T>> IntoDeviceOp<T> for DO {
    type Op = DO;
    fn into_op(self) -> DO {
        self
    }
}

// IntoDeviceOp impls for Arc<T> and &Arc<T> — wraps in Value.
impl<T: Send + Sync + 'static> IntoDeviceOp<Arc<T>> for Arc<T> {
    type Op = Value<Arc<T>>;
    fn into_op(self) -> Value<Arc<T>> {
        value(self)
    }
}

impl<T: Send + Sync + 'static> IntoDeviceOp<Arc<T>> for &Arc<T> {
    type Op = Value<Arc<T>>;
    fn into_op(self) -> Value<Arc<T>> {
        value(self.clone())
    }
}

// Scalar IntoDeviceOp impls — wraps the value in Value<T>.
macro_rules! impl_into_device_op_scalar {
    ($($ty:ty),*) => {
        $(
            impl IntoDeviceOp<$ty> for $ty {
                type Op = Value<$ty>;
                fn into_op(self) -> Value<$ty> { value(self) }
            }
        )*
    };
}
impl_into_device_op_scalar!(
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    bool,
    half::f16,
    half::bf16
);

// DevicePointer impl — for unsafe kernel pointer arguments.
impl<T: cuda_core::DType + Send> IntoDeviceOp<crate::device_buffer::DevicePointer<T>>
    for crate::device_buffer::DevicePointer<T>
{
    type Op = Value<crate::device_buffer::DevicePointer<T>>;
    fn into_op(self) -> Value<crate::device_buffer::DevicePointer<T>> {
        value(self)
    }
}

// Unwrap Arc
/// Extension trait: `.unwrap_arc()` on `DeviceOp<Output = Arc<T>>`.
///
/// Unwraps the Arc at execution time. Fails if the Arc has multiple owners.
pub trait DeviceOpUnwrapArc<T: Send + Sync>: DeviceOp<Output = Arc<T>> + Sized {
    fn unwrap_arc(
        self,
    ) -> AndThen<Arc<T>, Self, T, Value<T>, impl FnOnce(Arc<T>) -> Value<T> + Send> {
        self.then(|arc| {
            value(
                Arc::try_unwrap(arc)
                    .unwrap_or_else(|_| panic!("unwrap_arc: Arc has multiple owners")),
            )
        })
    }
}

impl<T: Send + Sync, DI: DeviceOp<Output = Arc<T>>> DeviceOpUnwrapArc<T> for DI {}

// AndThen

pub struct AndThen<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO,
{
    op: DI,
    closure: F,
}

unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
}

impl<I: Send, DI, O: Send, DO, F> DeviceOp for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = (self.closure)(input);
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThen<I, DI, O, DO, F>>;
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

// Value

pub struct Value<T>(T);
unsafe impl<T> Send for Value<T> {}

impl<T> Value<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: Send> DeviceOp for Value<T> {
    type Output = T;

    unsafe fn execute(
        self,
        _context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        Ok(self.0)
    }
}

impl<T: Send> GraphNode for Value<T> {}

impl<T: Send> IntoFuture for Value<T> {
    type Output = Result<T, DeviceError>;
    type IntoFuture = DeviceFuture<T, Value<T>>;
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

pub fn value<T: Send>(x: T) -> Value<T> {
    Value::new(x)
}

impl From<f32> for Value<f32> {
    fn from(val: f32) -> Self {
        Value::new(val)
    }
}

// Empty (closure)

pub struct Empty<O: Send, DO: DeviceOp<Output = O>, F: FnOnce() -> DO> {
    closure: F,
}

pub fn empty<O: Send, DO: DeviceOp<Output = O>, F: FnOnce() -> DO>(closure: F) -> Empty<O, DO, F> {
    Empty { closure }
}

unsafe impl<O: Send, DO, F> Send for Empty<O, DO, F>
where
    DO: DeviceOp<Output = O>,
    F: FnOnce() -> DO,
{
}

impl<O: Send, DO, F> DeviceOp for Empty<O, DO, F>
where
    DO: DeviceOp<Output = O>,
    F: FnOnce() -> DO,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let out_device_op = (self.closure)();
        out_device_op.execute(context)
    }
}

impl<O: Send, DO: DeviceOp<Output = O>, F: FnOnce() -> DO> IntoFuture for Empty<O, DO, F> {
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, Empty<O, DO, F>>;
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

// Zip

pub struct Zip<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> {
    phantom: PhantomData<(T1, T2)>,
    a: A,
    b: B,
}

unsafe impl<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> Send
    for Zip<T1, T2, A, B>
{
}

fn _zip<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>>(
    a: A,
    b: B,
) -> Zip<T1, T2, A, B> {
    Zip {
        phantom: PhantomData,
        a,
        b,
    }
}

impl<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> DeviceOp
    for Zip<T1, T2, A, B>
{
    type Output = (T1, T2);

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let a: T1 = self.a.execute(context)?;
        let b: T2 = self.b.execute(context)?;
        Ok((a, b))
    }
}

impl<T1: Send, T2: Send, A: DeviceOp<Output = T1>, B: DeviceOp<Output = T2>> IntoFuture
    for Zip<T1, T2, A, B>
{
    type Output = Result<(T1, T2), DeviceError>;
    type IntoFuture = DeviceFuture<(T1, T2), Zip<T1, T2, A, B>>;
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

pub trait Zippable<I, O: Send> {
    fn zip(self) -> impl DeviceOp<Output = O>;
}

impl<T0: Send, T1: Send, DI0: DeviceOp<Output = T0>, DI1: DeviceOp<Output = T1>>
    Zippable<(DI0, DI1), (T0, T1)> for (DI0, DI1)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1)> {
        _zip(self.0, self.1)
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
    > Zippable<(DI0, DI1, DI2), (T0, T1, T2)> for (DI0, DI1, DI2)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2)> {
        let cons = _zip(self.1, self.2);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, arg2))| value((arg0, arg1, arg2)))
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
        DI3: DeviceOp<Output = T3>,
    > Zippable<(DI0, DI1, DI2, DI3), (T0, T1, T2, T3)> for (DI0, DI1, DI2, DI3)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2, T3)> {
        let cons = _zip(self.2, self.3);
        let cons = _zip(self.1, cons);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, (arg2, arg3)))| value((arg0, arg1, arg2, arg3)))
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
        DI3: DeviceOp<Output = T3>,
        DI4: DeviceOp<Output = T4>,
    > Zippable<(DI0, DI1, DI2, DI3, DI4), (T0, T1, T2, T3, T4)> for (DI0, DI1, DI2, DI3, DI4)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2, T3, T4)> {
        let cons = _zip(self.3, self.4);
        let cons = _zip(self.2, cons);
        let cons = _zip(self.1, cons);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, (arg2, (arg3, arg4))))| value((arg0, arg1, arg2, arg3, arg4)))
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        T5: Send,
        DI0: DeviceOp<Output = T0>,
        DI1: DeviceOp<Output = T1>,
        DI2: DeviceOp<Output = T2>,
        DI3: DeviceOp<Output = T3>,
        DI4: DeviceOp<Output = T4>,
        DI5: DeviceOp<Output = T5>,
    > Zippable<(DI0, DI1, DI2, DI3, DI4, DI5), (T0, T1, T2, T3, T4, T5)>
    for (DI0, DI1, DI2, DI3, DI4, DI5)
{
    fn zip(self) -> impl DeviceOp<Output = (T0, T1, T2, T3, T4, T5)> {
        let cons = _zip(self.4, self.5);
        let cons = _zip(self.3, cons);
        let cons = _zip(self.2, cons);
        let cons = _zip(self.1, cons);
        let cons = _zip(self.0, cons);
        cons.then(|(arg0, (arg1, (arg2, (arg3, (arg4, arg5)))))| {
            value((arg0, arg1, arg2, arg3, arg4, arg5))
        })
    }
}

#[macro_export]
macro_rules! zip {
    ($arg0:expr) => {
        $arg0
    };
    ($arg0:expr, $arg1:expr) => {
        ($arg0, $arg1).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr) => {
        ($arg0, $arg1, $arg2).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr, $arg3:expr) => {
        ($arg0, $arg1, $arg2, $arg3).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr) => {
        ($arg0, $arg1, $arg2, $arg3, $arg4).zip()
    };
    ($arg0:expr, $arg1:expr, $arg2:expr, $arg3:expr, $arg4:expr, $arg5:expr) => {
        ($arg0, $arg1, $arg2, $arg3, $arg4, $arg5).zip()
    };
}
pub use zip;

// Unzip

fn _unzip<T1: Send, T2: Send, DI>(input: DI) -> (SelectLeft<T1, T2, DI>, SelectRight<T1, T2, DI>)
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    let select = Select {
        computed: AtomicBool::new(false),
        input: UnsafeCell::new(Some(input)),
        left: UnsafeCell::new(None),
        right: UnsafeCell::new(None),
    };
    let select_arc = Arc::new(select);
    let out1 = SelectLeft {
        select: select_arc.clone(),
    };
    let out2 = SelectRight { select: select_arc };
    (out1, out2)
}

// Select: Execute a device operation at most once.

pub struct Select<T1: Send, T2: Send, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    computed: AtomicBool,
    input: UnsafeCell<Option<DI>>,
    left: UnsafeCell<Option<T1>>,
    right: UnsafeCell<Option<T2>>,
}

impl<T1: Send, T2: Send, DI> Select<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    unsafe fn execute(self: &Arc<Self>, context: &ExecutionContext) -> Result<(), DeviceError> {
        if !self.computed.load(Ordering::Acquire) {
            // Safety: This block is guaranteed to execute at most once.
            // Put the input in a box so the pointer is dropped when this block exits.
            let input = self.input.get();
            let input = unsafe { input.as_mut() };
            let input = input
                .unwrap()
                .take()
                .ok_or_else(|| device_error(context.get_device_id(), "Select operation failed."))?;
            let (left, right) = input.execute(context)?;
            // Update internal state.
            unsafe {
                *self.left.get() = Some(left);
                *self.right.get() = Some(right);
            }
            self.computed.store(true, Ordering::Release);
        }
        Ok(())
    }
    unsafe fn left(&self) -> T1 {
        let cell = self.left.get();
        let cell = unsafe { cell.as_mut() };
        cell.unwrap().take().unwrap()
    }
    unsafe fn right(&self) -> T2 {
        let cell = self.right.get();
        let cell = unsafe { cell.as_mut() };
        cell.unwrap().take().unwrap()
    }
}

// Select Left: Execute Select and take the left result.

pub struct SelectLeft<T1: Send, T2: Send, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    select: Arc<Select<T1, T2, DI>>,
}

unsafe impl<T1: Send, T2: Send, DI: DeviceOp<Output = (T1, T2)>> Send for SelectLeft<T1, T2, DI> {}

impl<T1: Send, T2: Send, DI> IntoFuture for SelectLeft<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = Result<T1, DeviceError>;
    type IntoFuture = DeviceFuture<T1, SelectLeft<T1, T2, DI>>;
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

impl<T1: Send, T2: Send, DI> DeviceOp for SelectLeft<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = T1;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        self.select.execute(context)?;
        Ok(self.select.left())
    }
}

// Select Right: Execute Select and take the right result.

pub struct SelectRight<T1: Send, T2: Send, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    select: Arc<Select<T1, T2, DI>>,
}

unsafe impl<T1: Send, T2: Send, DI: DeviceOp<Output = (T1, T2)>> Send for SelectRight<T1, T2, DI> {}

impl<T1: Send, T2: Send, DI> IntoFuture for SelectRight<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = Result<T2, DeviceError>;
    type IntoFuture = DeviceFuture<T2, SelectRight<T1, T2, DI>>;
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

impl<T1: Send, T2: Send, DI> DeviceOp for SelectRight<T1, T2, DI>
where
    DI: DeviceOp<Output = (T1, T2)>,
{
    type Output = T2;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        self.select.execute(context)?;
        Ok(self.select.right())
    }
}

pub trait Unzippable1<T0: Send>
where
    Self: DeviceOp<Output = (T0,)>,
{
    fn unzip(self) -> (impl DeviceOp<Output = T0>,) {
        (self.then(|(r,)| value(r)),)
    }
}
impl<T0: Send, DI: DeviceOp<Output = (T0,)>> Unzippable1<T0> for DI {}

pub trait Unzippable2<T0: Send, T1: Send>
where
    Self: DeviceOp<Output = (T0, T1)>,
{
    fn unzip(self) -> (impl DeviceOp<Output = T0>, impl DeviceOp<Output = T1>) {
        _unzip(self)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T1>
    where
        Self: Sized,
    {
        self.then(|(_, last)| value(last))
    }
}
impl<T0: Send, T1: Send, DI: DeviceOp<Output = (T0, T1)>> Unzippable2<T0, T1> for DI {}

pub trait Unzippable3<T0: Send, T1: Send, T2: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
    ) {
        let cons = self.then(|(arg0, arg1, arg2)| value((arg0, (arg1, arg2))));
        let (car, cdr) = _unzip(cons);
        let (cdr_car, cdr_cdr) = _unzip(cdr);
        (car, cdr_car, cdr_cdr)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T2>
    where
        Self: Sized,
    {
        self.then(|(_, _, last)| value(last))
    }
}
impl<T0: Send, T1: Send, T2: Send, DI: DeviceOp<Output = (T0, T1, T2)>> Unzippable3<T0, T1, T2>
    for DI
{
}

pub trait Unzippable4<T0: Send, T1: Send, T2: Send, T3: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2, T3)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
        impl DeviceOp<Output = T3>,
    ) {
        let cons = self.then(|(a0, a1, a2, a3)| value((a0, (a1, (a2, a3)))));
        let (car, cdr) = _unzip(cons);
        let (cdr0, cdr1) = _unzip(cdr);
        let (cdr1_0, cdr1_1) = _unzip(cdr1);
        (car, cdr0, cdr1_0, cdr1_1)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T3>
    where
        Self: Sized,
    {
        self.then(|(_, _, _, last)| value(last))
    }
}
impl<T0: Send, T1: Send, T2: Send, T3: Send, DI: DeviceOp<Output = (T0, T1, T2, T3)>>
    Unzippable4<T0, T1, T2, T3> for DI
{
}

pub trait Unzippable5<T0: Send, T1: Send, T2: Send, T3: Send, T4: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2, T3, T4)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
        impl DeviceOp<Output = T3>,
        impl DeviceOp<Output = T4>,
    ) {
        let cons = self.then(|(a0, a1, a2, a3, a4)| value((a0, (a1, (a2, (a3, a4))))));
        let (car, cdr) = _unzip(cons);
        let (cdr0, cdr1) = _unzip(cdr);
        let (cdr1_0, cdr1_1) = _unzip(cdr1);
        let (cdr2_0, cdr2_1) = _unzip(cdr1_1);
        (car, cdr0, cdr1_0, cdr2_0, cdr2_1)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T4>
    where
        Self: Sized,
    {
        self.then(|(_, _, _, _, last)| value(last))
    }
}
impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        DI: DeviceOp<Output = (T0, T1, T2, T3, T4)>,
    > Unzippable5<T0, T1, T2, T3, T4> for DI
{
}

pub trait Unzippable6<T0: Send, T1: Send, T2: Send, T3: Send, T4: Send, T5: Send>
where
    Self: DeviceOp<Output = (T0, T1, T2, T3, T4, T5)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOp<Output = T0>,
        impl DeviceOp<Output = T1>,
        impl DeviceOp<Output = T2>,
        impl DeviceOp<Output = T3>,
        impl DeviceOp<Output = T4>,
        impl DeviceOp<Output = T5>,
    ) {
        let cons = self.then(|(a0, a1, a2, a3, a4, a5)| value((a0, (a1, (a2, (a3, (a4, a5)))))));
        let (car, cdr) = _unzip(cons);
        let (cdr0, cdr1) = _unzip(cdr);
        let (cdr1_0, cdr1_1) = _unzip(cdr1);
        let (cdr2_0, cdr2_1) = _unzip(cdr1_1);
        let (cdr3_0, cdr3_1) = _unzip(cdr2_1);
        (car, cdr0, cdr1_0, cdr2_0, cdr3_0, cdr3_1)
    }
    fn first(self) -> impl DeviceOp<Output = T0>
    where
        Self: Sized,
    {
        self.then(|(first, _, _, _, _, _)| value(first))
    }
    fn last(self) -> impl DeviceOp<Output = T5>
    where
        Self: Sized,
    {
        self.then(|(_, _, _, _, _, last)| value(last))
    }
}
impl<
        T0: Send,
        T1: Send,
        T2: Send,
        T3: Send,
        T4: Send,
        T5: Send,
        DI: DeviceOp<Output = (T0, T1, T2, T3, T4, T5)>,
    > Unzippable6<T0, T1, T2, T3, T4, T5> for DI
{
}

#[macro_export]
macro_rules! unzip {
    ($arg0:expr) => {
        $arg0.unzip()
    };
}
pub use unzip;

// StreamOperation

pub struct StreamOperation<
    O: Send,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
> {
    f: F,
}

impl<O: Send, DO: DeviceOp<Output = O>, F: FnOnce(&ExecutionContext) -> DO + Send> DeviceOp
    for StreamOperation<O, DO, F>
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let dop_out: DO = (self.f)(context);
        dop_out.execute(context)
    }
}

pub fn with_context<
    O: Send,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
>(
    f: F,
) -> impl DeviceOp<Output = O> {
    StreamOperation { f }
}

impl<O: Send, DO: DeviceOp<Output = O>, F: FnOnce(&ExecutionContext) -> DO + Send> IntoFuture
    for StreamOperation<O, DO, F>
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, StreamOperation<O, DO, F>>;
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

// AndThenWithContext

pub struct AndThenWithContext<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO,
{
    op: DI,
    closure: F,
}

unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
}

impl<I: Send, DI, O: Send, DO, F> DeviceOp for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = (self.closure)(context, input);
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOp<Output = I>,
    DO: DeviceOp<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThenWithContext<I, DI, O, DO, F>>;
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

// DeviceOpVec — execute a Vec of boxed operations, sync once.

/// A [`DeviceOp`] that executes a vector of [`BoxedDeviceOp`]s
/// sequentially on the same stream and collects their outputs into a `Vec<T>`.
///
/// This avoids per-element stream synchronization: the caller can issue a
/// single `.sync_on(&stream)` after all operations have been submitted.
pub struct DeviceOpVec<T: Send> {
    ops: Vec<BoxedDeviceOp<T>>,
}

impl<T: Send + 'static> DeviceOpVec<T> {
    pub fn empty() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ops: Vec::with_capacity(capacity),
        }
    }

    pub fn new(ops: Vec<BoxedDeviceOp<T>>) -> Self {
        Self { ops }
    }

    pub fn push<DO: DeviceOp<Output = T> + 'static>(&mut self, op: DO) {
        self.ops.push(op.boxed());
    }

    pub fn remove(&mut self, index: usize) -> BoxedDeviceOp<T> {
        self.ops.remove(index)
    }

    pub fn last(&self) -> Option<&BoxedDeviceOp<T>> {
        self.ops.last()
    }
}

impl<T: Send> DeviceOp for DeviceOpVec<T> {
    type Output = Vec<T>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Vec<T>, DeviceError> {
        let mut results = Vec::with_capacity(self.ops.len());
        for op in self.ops {
            results.push(op.execute(context)?);
        }
        Ok(results)
    }
}

impl<T: Send> IntoFuture for DeviceOpVec<T> {
    type Output = Result<Vec<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Vec<T>, Self>;
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

impl<T: Send + 'static> From<Vec<BoxedDeviceOp<T>>> for DeviceOpVec<T> {
    fn from(ops: Vec<BoxedDeviceOp<T>>) -> Self {
        Self::new(ops)
    }
}

// New names — old names kept as re-exports for backwards compatibility.
