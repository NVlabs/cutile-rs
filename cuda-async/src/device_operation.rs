/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lazy, composable GPU operations and combinator types.

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::error::{device_error, DeviceError};
use crate::scheduling_policies::SchedulingPolicy;
use cuda_core::{CudaContext, CudaStream};
use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub type Device = usize;

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    device: Device,
    cuda_stream: Arc<CudaStream>,
    cuda_context: Arc<CudaContext>,
}

impl ExecutionContext {
    pub fn new(cuda_stream: Arc<CudaStream>) -> Self {
        let cuda_context = Arc::clone(cuda_stream.context());
        let device = cuda_context.ordinal();
        Self {
            cuda_stream,
            cuda_context,
            device,
        }
    }
    pub fn get_cuda_stream(&self) -> &Arc<CudaStream> {
        &self.cuda_stream
    }
    pub fn get_cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }
    pub fn get_device_id(&self) -> Device {
        self.device
    }
    #[expect(
        dead_code,
        reason = "kept for direct synchronous execution in tests and future blocking APIs"
    )]
    fn execute<T: Send>(&self, op: impl DeviceOperation<Output = T>) -> Result<T, DeviceError> {
        unsafe {
            // Safety: ExecutionContext is only available within a DeviceOperation closure.
            // DeviceOperation closures can only be converted into DeviceFuture
            // which synchronizes device operations with the host thread via a host callback.
            op.execute(self)
        }
    }
}

/// A lazy, composable GPU operation that may be executed synchronously or asynchronously on a CUDA device.
///
/// `DeviceOperation` represents a resource-agnostic computation that will be scheduled and executed.
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
/// With the default [`StreamPoolRoundRobin`] policy (4 streams), consecutive `.await` or
/// `.sync()` calls rotate through streams, so independent operations can overlap on the GPU.
/// Operations chained with [`.and_then()`](DeviceOperation::and_then) share a single stream
/// and always execute in order.
///
/// See [`SchedulingPolicy`] for a full explanation of ordering guarantees.
///
/// # Safety
///
/// The `execute` method is unsafe because it's asynchronous - the GPU may still be writing to
/// memory allocated by the output after `execute` returns. Converting a `DeviceOperation` into
/// a `DeviceFuture` ensures memory operations complete before the output can be accessed.
///
/// ## Examples
///
/// ```rust,ignore
/// use cuda_async::device_operation::{DeviceOperation, value};
///
/// // Create a simple value operation
/// let op1 = value(42);
///
/// // Chain operations together
/// let op2 = op1.and_then(|x| value(x * 2));
///
/// // Execute synchronously (blocks until GPU completes)
/// let result = op2.sync().expect("Device operation failed."); // returns 84
/// ```
///
/// ```rust,ignore
/// use cuda_async::device_operation::{DeviceOperation, zip};
/// use cutile::api;
///
/// // Compose multiple tensor operations
/// let x = api::zeros([64, 64]);
/// let y = api::ones([64, 64]);
/// let combined = zip!(x, y).and_then(|(x, y)| {
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
/// let x = api::randn(0.0, 1.0, [100, 100]).arc().await?;
/// let y = some_kernel(Arc::clone(&x)).await;
/// ```
pub trait DeviceOperation:
    Send + Sized + IntoFuture<Output = Result<<Self as DeviceOperation>::Output, DeviceError>>
{
    type Output: Send;

    // Consumes DeviceOperation and executes the implementing operation.
    // This is unsafe because it is asynchronous: A device may be writing to memory allocated
    // by the output.
    // Converting DeviceOperation into a DeviceFuture ensures any memory operations are complete
    // before the output can be accessed by the async runtime.
    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError>;
    /// Schedule this operation on a specific policy and return a [`DeviceFuture`].
    fn schedule<P: SchedulingPolicy>(
        self,
        policy: &P,
    ) -> Result<DeviceFuture<<Self as DeviceOperation>::Output, Self>, DeviceError> {
        policy.schedule(self)
    }
    /// Alias for [`DeviceOperation::and_then`].
    ///
    /// `apply` sequences on this operation's resolved output, rather than
    /// transforming the operation object itself.
    fn apply<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThen<<Self as DeviceOperation>::Output, Self, O, DO, F>
    where
        DO: DeviceOperation<Output = O>,
        F: FnOnce(<Self as DeviceOperation>::Output) -> DO,
    {
        self.and_then(f)
    }
    /// Chain a follow-up operation that runs **on the same stream** as `self`.
    ///
    /// Because both operations share a stream, `f` is guaranteed to see `self`'s output
    /// fully written. This is the recommended way to express data dependencies without
    /// manual synchronization.
    fn and_then<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThen<<Self as DeviceOperation>::Output, Self, O, DO, F>
    where
        DO: DeviceOperation<Output = O>,
        F: FnOnce(<Self as DeviceOperation>::Output) -> DO,
    {
        AndThen {
            op: self,
            closure: f,
        }
    }
    fn and_then_with_context<O: Send, DO, F>(
        self,
        f: F,
    ) -> AndThenWithContext<<Self as DeviceOperation>::Output, Self, O, DO, F>
    where
        DO: DeviceOperation<Output = O>,
        F: FnOnce(&ExecutionContext, <Self as DeviceOperation>::Output) -> DO,
    {
        AndThenWithContext {
            op: self,
            closure: f,
        }
    }
    fn arc(self) -> DeviceOperationArc<<Self as DeviceOperation>::Output, Self>
    where
        <Self as DeviceOperation>::Output: Sync,
    {
        DeviceOperationArc { op: self }
    }
    /// Execute synchronously using the default device's scheduling policy.
    ///
    /// The policy picks a stream (round-robin by default), submits the work, and blocks
    /// until the GPU finishes. Equivalent to `.await` but blocking.
    fn sync(self) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        with_default_device_policy(|policy| policy.sync(self))?
    }
    unsafe fn async_on(
        self,
        stream: &Arc<CudaStream>,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let ctx = ExecutionContext::new(Arc::clone(stream));
        // This is okay since we synchronize immediately.

        unsafe { self.execute(&ctx) }
    }
    /// Execute on an **explicit stream** and block until the GPU finishes.
    ///
    /// This bypasses the scheduling policy entirely. All operations `sync_on` the same
    /// stream are guaranteed to execute in call order. Use this when you need deterministic
    /// ordering or are debugging concurrency issues.
    fn sync_on(
        self,
        stream: &Arc<CudaStream>,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let ctx = ExecutionContext::new(Arc::clone(stream));
        // This is okay since we synchronize immediately.
        let res = unsafe { self.execute(&ctx) };
        stream.synchronize().expect("Synchronize failed.");
        res
    }
}

// Arc

// I has to be sync since we are wrapping it in Arc.
pub struct DeviceOperationArc<I: Send + Sync, DI>
where
    DI: DeviceOperation<Output = I>,
{
    op: DI,
}

unsafe impl<I: Send + Sync, DI> Send for DeviceOperationArc<I, DI> where
    DI: DeviceOperation<Output = I>
{
}

impl<I: Send + Sync, DI> DeviceOperation for DeviceOperationArc<I, DI>
where
    DI: DeviceOperation<Output = I>,
{
    type Output = Arc<I>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let val: I = self.op.execute(context)?;
        Ok(Arc::new(val))
    }
}

impl<I: Send + Sync, DI> IntoFuture for DeviceOperationArc<I, DI>
where
    DI: DeviceOperation<Output = I>,
{
    type Output = Result<Arc<I>, DeviceError>;
    type IntoFuture = DeviceFuture<Arc<I>, DeviceOperationArc<I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

// Unwrap Arc
pub struct UnwrapArc<I: Send + Sync, DI>
where
    DI: DeviceOperation<Output = Arc<I>>,
{
    op: DI,
}

unsafe impl<I: Send + Sync, DI> Send for UnwrapArc<I, DI> where DI: DeviceOperation<Output = Arc<I>> {}

impl<I: Send + Sync + Debug, DI> DeviceOperation for UnwrapArc<I, DI>
where
    DI: DeviceOperation<Output = Arc<I>>,
{
    type Output = I;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let val = self.op.execute(context)?;
        match Arc::try_unwrap(val) {
            Ok(inner) => Ok(inner),
            Err(_) => Err(DeviceError::Internal("Arc unwrap failed.".to_string())),
        }
    }
}

impl<I: Send + Sync + Debug, DI> IntoFuture for UnwrapArc<I, DI>
where
    DI: DeviceOperation<Output = Arc<I>>,
{
    type Output = Result<I, DeviceError>;
    type IntoFuture = DeviceFuture<I, UnwrapArc<I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

// AndThen

pub struct AndThen<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO,
{
    op: DI,
    closure: F,
}

unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
}

impl<I: Send, DI, O: Send, DO, F> DeviceOperation for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = (self.closure)(input);
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThen<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThen<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
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

impl<T: Send> DeviceOperation for Value<T> {
    type Output = T;

    unsafe fn execute(
        self,
        _context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        Ok(self.0)
    }
}

impl<T: Send> IntoFuture for Value<T> {
    type Output = Result<T, DeviceError>;
    type IntoFuture = DeviceFuture<T, Value<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

pub fn value<T: Send>(x: T) -> Value<T> {
    Value::new(x)
}

// Use value to impl into for any T.
pub trait IntoDeviceOperation<T: Send> {
    fn device_operation(self) -> Value<T>;
}
impl<T: Send> IntoDeviceOperation<T> for T {
    fn device_operation(self) -> Value<T> {
        value(self)
    }
}
impl From<f32> for Value<f32> {
    fn from(val: f32) -> Self {
        Value::new(val)
    }
}

// Empty (closure)

pub struct Empty<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO> {
    closure: F,
}

pub fn empty<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO>(
    closure: F,
) -> Empty<O, DO, F> {
    Empty { closure }
}

unsafe impl<O: Send, DO, F> Send for Empty<O, DO, F>
where
    DO: DeviceOperation<Output = O>,
    F: FnOnce() -> DO,
{
}

impl<O: Send, DO, F> DeviceOperation for Empty<O, DO, F>
where
    DO: DeviceOperation<Output = O>,
    F: FnOnce() -> DO,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let out_device_op = (self.closure)();
        out_device_op.execute(context)
    }
}

impl<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce() -> DO> IntoFuture for Empty<O, DO, F> {
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, Empty<O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

// Zip

pub struct Zip<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>
{
    phantom: PhantomData<(T1, T2)>,
    a: A,
    b: B,
}

unsafe impl<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>
    Send for Zip<T1, T2, A, B>
{
}

fn _zip<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>(
    a: A,
    b: B,
) -> Zip<T1, T2, A, B> {
    Zip {
        phantom: PhantomData,
        a,
        b,
    }
}

impl<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>
    DeviceOperation for Zip<T1, T2, A, B>
{
    type Output = (T1, T2);

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let a: T1 = self.a.execute(context)?;
        let b: T2 = self.b.execute(context)?;
        Ok((a, b))
    }
}

impl<T1: Send, T2: Send, A: DeviceOperation<Output = T1>, B: DeviceOperation<Output = T2>>
    IntoFuture for Zip<T1, T2, A, B>
{
    type Output = Result<(T1, T2), DeviceError>;
    type IntoFuture = DeviceFuture<(T1, T2), Zip<T1, T2, A, B>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

pub trait Zippable<I, O: Send> {
    fn zip(self) -> impl DeviceOperation<Output = O>;
}

impl<T0: Send, T1: Send, DI0: DeviceOperation<Output = T0>, DI1: DeviceOperation<Output = T1>>
    Zippable<(DI0, DI1), (T0, T1)> for (DI0, DI1)
{
    fn zip(self) -> impl DeviceOperation<Output = (T0, T1)> {
        _zip(self.0, self.1)
    }
}

impl<
        T0: Send,
        T1: Send,
        T2: Send,
        DI0: DeviceOperation<Output = T0>,
        DI1: DeviceOperation<Output = T1>,
        DI2: DeviceOperation<Output = T2>,
    > Zippable<(DI0, DI1, DI2), (T0, T1, T2)> for (DI0, DI1, DI2)
{
    fn zip(self) -> impl DeviceOperation<Output = (T0, T1, T2)> {
        let cons = _zip(self.1, self.2);
        let cons = _zip(self.0, cons);
        cons.and_then(|(arg0, (arg1, arg2))| value((arg0, arg1, arg2)))
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
}
pub use zip;

// Unzip

fn _unzip<T1: Send, T2: Send, DI>(input: DI) -> (SelectLeft<T1, T2, DI>, SelectRight<T1, T2, DI>)
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    let select = Select {
        computed: AtomicBool::new(false),
        input: UnsafeCell::new(Some(input)),
        left: UnsafeCell::new(None),
        right: UnsafeCell::new(None),
    };
    let select = Arc::new(select);
    let out1 = SelectLeft {
        select: Arc::clone(&select),
    };
    let out2 = SelectRight { select };
    (out1, out2)
}

// Select: Execute a device operation at most once.

pub struct Select<T1: Send, T2: Send, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    computed: AtomicBool,
    input: UnsafeCell<Option<DI>>,
    left: UnsafeCell<Option<T1>>,
    right: UnsafeCell<Option<T2>>,
}

impl<T1: Send, T2: Send, DI> Select<T1, T2, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
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
    DI: DeviceOperation<Output = (T1, T2)>,
{
    select: Arc<Select<T1, T2, DI>>,
}

unsafe impl<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> Send
    for SelectLeft<T1, T2, DI>
{
}

impl<T1: Send, T2: Send, DI> IntoFuture for SelectLeft<T1, T2, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    type Output = Result<T1, DeviceError>;
    type IntoFuture = DeviceFuture<T1, SelectLeft<T1, T2, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl<T1: Send, T2: Send, DI> DeviceOperation for SelectLeft<T1, T2, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    type Output = T1;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        self.select.execute(context)?;
        Ok(self.select.left())
    }
}

// Select Right: Execute Select and take the right result.

pub struct SelectRight<T1: Send, T2: Send, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    select: Arc<Select<T1, T2, DI>>,
}

unsafe impl<T1: Send, T2: Send, DI: DeviceOperation<Output = (T1, T2)>> Send
    for SelectRight<T1, T2, DI>
{
}

impl<T1: Send, T2: Send, DI> IntoFuture for SelectRight<T1, T2, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    type Output = Result<T2, DeviceError>;
    type IntoFuture = DeviceFuture<T2, SelectRight<T1, T2, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl<T1: Send, T2: Send, DI> DeviceOperation for SelectRight<T1, T2, DI>
where
    DI: DeviceOperation<Output = (T1, T2)>,
{
    type Output = T2;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        self.select.execute(context)?;
        Ok(self.select.right())
    }
}

pub trait Unzippable1<T0: Send>
where
    Self: DeviceOperation<Output = (T0,)>,
{
    fn unzip(self) -> (impl DeviceOperation<Output = T0>,) {
        (self.and_then(|(r,)| value(r)),)
    }
}
impl<T0: Send, DI: DeviceOperation<Output = (T0,)>> Unzippable1<T0> for DI {}

pub trait Unzippable2<T0: Send, T1: Send>
where
    Self: DeviceOperation<Output = (T0, T1)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOperation<Output = T0>,
        impl DeviceOperation<Output = T1>,
    ) {
        _unzip(self)
    }
}
impl<T0: Send, T1: Send, DI: DeviceOperation<Output = (T0, T1)>> Unzippable2<T0, T1> for DI {}

pub trait Unzippable3<T0: Send, T1: Send, T2: Send>
where
    Self: DeviceOperation<Output = (T0, T1, T2)>,
{
    fn unzip(
        self,
    ) -> (
        impl DeviceOperation<Output = T0>,
        impl DeviceOperation<Output = T1>,
        impl DeviceOperation<Output = T2>,
    ) {
        let cons = self.and_then(|(arg0, arg1, arg2)| value((arg0, (arg1, arg2))));
        let (car, cdr) = _unzip(cons);
        let (cdr_car, cdr_cdr) = _unzip(cdr);
        (car, cdr_car, cdr_cdr)
    }
}
impl<T0: Send, T1: Send, T2: Send, DI: DeviceOperation<Output = (T0, T1, T2)>>
    Unzippable3<T0, T1, T2> for DI
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
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
> {
    f: F,
}

impl<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce(&ExecutionContext) -> DO + Send>
    DeviceOperation for StreamOperation<O, DO, F>
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let dop_out: DO = (self.f)(context);
        dop_out.execute(context)
    }
}

pub fn with_context<
    O: Send,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext) -> DO + Send,
>(
    f: F,
) -> impl DeviceOperation<Output = O> {
    StreamOperation { f }
}

impl<O: Send, DO: DeviceOperation<Output = O>, F: FnOnce(&ExecutionContext) -> DO + Send> IntoFuture
    for StreamOperation<O, DO, F>
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, StreamOperation<O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}

// AndThenWithContext

pub struct AndThenWithContext<I: Send, DI, O: Send, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO,
{
    op: DI,
    closure: F,
}

unsafe impl<I: Send, DI, O: Send, DO, F> Send for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
}

impl<I: Send, DI, O: Send, DO, F> DeviceOperation for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = O;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let input: I = self.op.execute(context)?;
        let output_device_op: DO = (self.closure)(context, input);
        output_device_op.execute(context)
    }
}

impl<I: Send, DI, O: Send, DO, F> IntoFuture for AndThenWithContext<I, DI, O, DO, F>
where
    DI: DeviceOperation<Output = I>,
    DO: DeviceOperation<Output = O>,
    F: FnOnce(&ExecutionContext, I) -> DO + Send,
{
    type Output = Result<O, DeviceError>;
    type IntoFuture = DeviceFuture<O, AndThenWithContext<I, DI, O, DO, F>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}
