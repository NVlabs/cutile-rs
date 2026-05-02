/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! High-level API for tensor creation and manipulation.
//!
//! This module provides NumPy-like functions for creating and manipulating GPU tensors.
//! All operations are asynchronous and return [`DeviceOp`]s that can be `.await`ed.
//!
//! ## Overview
//!
//! The API module is designed to feel familiar to NumPy or PyTorch users while leveraging
//! Rust's type system and async capabilities. Every function returns a lazy operation that
//! only executes when awaited.
//!
//! ## Tensor Creation
//!
//! ### Constant Tensors
//!
//! - [`zeros`] - Create tensor filled with zeros
//! - [`ones`] - Create tensor filled with ones
//! - [`full`] - Create tensor filled with a specific value
//!
//! ### Sequential Data
//!
//! - [`arange`] - Create tensor with evenly spaced values (like `0, 1, 2, ...`)
//!
//! ### Random Tensors
//!
//! - [`randn`] - Create tensor with values from normal distribution
//!
//! ### Memory Operations
//!
//! - [`dup`] - Copy a tensor to new GPU memory
//! - [`copy_device_to_host_vec`] - Copy GPU tensor to CPU Vec
//!
//! ## Examples
//!
//! ### Basic Tensor Creation
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create different types of tensors
//!     let zeros = api::zeros::<f32>(&[1024]).await;
//!     let ones = api::ones::<f32>(&[512, 512]).await;
//!     let range = api::arange::<i32>(100).await;
//!     let random = api::randn(0.0, 1.0, [256, 256], None).await;
//! }
//! ```
//!
//! ### Memory Management
//!
//! ```rust,ignore
//! use cutile::api;
//! use std::sync::Arc;
//!
//! // Create a tensor
//! let x: Tensor<f32> = api::zeros(&[1024]).await;
//!
//! // Duplicate to new memory
//! let y = api::dup(&x).await;
//!
//! // Copy to CPU for inspection
//! let cpu_data: Vec<f32> = x.to_host_vec().await;
//! ```
//!
//! ### Composing Operations
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // Operations compose naturally with async/await
//! let x: Tensor<f32> = api::randn(0.0, 1.0, [1024], None).await;
//! let y = api::dup(&x).await;
//! let z = y.partition([128]); // Prepare for kernel
//! ```
//!
//! ## Design Philosophy
//!
//! ### Lazy Execution
//!
//! All functions return [`DeviceOp`]s that don't execute immediately:
//!
//! ```rust,ignore
//! let x = api::zeros(&[1024]);  // No GPU work yet!
//! let y = api::ones(&[1024]);   // Still no GPU work!
//!
//! let x = x.await;  // NOW x allocates and fills
//! let y = y.await;  // NOW y allocates and fills
//! ```
//!
//! This enables:
//! - Building computation graphs before execution
//! - Optimizing execution order
//! - Parallelizing independent operations
//!
//! ### Type Safety
//!
//! Tensor shapes are tracked at compile time where possible:
//!
//! ```rust,ignore
//! let x = api::zeros(&[256]);           // Shape known at compile time
//! let partitioned = x.partition([64]); // Compiler checks 256 % 64 == 0
//! ```
//!
//! ### Async Integration
//!
//! All operations integrate seamlessly with Tokio or other async runtimes:
//!
//! ```rust,ignore
//! #[tokio::main]
//! async fn main() {
//!     let x = api::randn(0.0, 1.0, [1024, 1024]).await;
//!     // Use x in kernels or copy back to host
//! }
//! ```
//!
//! ## Performance Notes
//!
//! - **Allocation**: GPU memory allocation is relatively expensive (~microseconds)
//! - **Initialization**: Filling tensors requires a kernel launch
//! - **Copying**: Host ↔ Device copies are bandwidth-limited (~GB/s)
//! - **Async overhead**: Negligible compared to GPU operation time
//!
//! ## See Also
//!
//! - [`tile_kernel`](crate::tile_kernel) - Lower-level async execution primitives
//! - [`tensor`](crate::tensor) - Tensor type and partitioning
//! - [`kernels`](crate::kernels) - Pre-built GPU kernels

use crate::kernels::conversion::convert_apply;
use crate::kernels::creation::{arange_apply, eye_apply, full_apply, linspace as linspace_kernel};
use crate::tensor::{IntoPartition, Reshape, Tensor, Unpartition};
use cuda_async::device_buffer::DeviceBuffer;
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{value, DeviceOp, ExecutionContext, Unzippable1, Unzippable2};
use cuda_async::error::DeviceError;
use cuda_core::curand::{RandNormal, RandUniform, RNG};
use cuda_core::sys::CUdeviceptr;
use cuda_core::DType;
use cuda_core::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async};
use half::f16;
use std::alloc::{alloc, Layout};
use std::future::IntoFuture;
use std::sync::Arc;

/// Device operation for copying a tensor within GPU memory.
///
/// This internal type implements the async copy operation that allocates new
/// GPU memory and copies tensor data device-to-device.
pub struct CopyDeviceToDevice<T: DType> {
    _storage: Arc<DeviceBuffer>, // keeps source GPU memory alive
    src_ptr: CUdeviceptr,
    shape: Vec<i32>,
    strides: Vec<i32>,
    num_elements: usize,
    _dtype: std::marker::PhantomData<T>,
}

impl<T: DType> DeviceOp for CopyDeviceToDevice<T> {
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let num_bytes = self.num_elements * std::mem::size_of::<T>();
        let dst = ctx.alloc_async(num_bytes);
        memcpy_dtod_async::<T>(dst, self.src_ptr, self.num_elements, ctx.get_cuda_stream());
        Ok(Tensor::from_raw_parts(
            dst,
            num_bytes,
            ctx.get_device_id(),
            self.shape,
            self.strides,
        ))
    }
}

impl<T: DType> IntoFuture for CopyDeviceToDevice<T> {
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, CopyDeviceToDevice<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Creates a copy of a GPU tensor.
///
/// Allocates new GPU memory and copies the tensor data asynchronously. This is useful
/// when you need an independent copy of tensor data.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
/// use std::sync::Arc;
///
/// let x: Tensor<f32> = api::zeros(&[1024]).await;
/// let y = api::dup(&x).await;
/// // y is now an independent copy of x
/// ```
pub fn dup<T: DType>(tensor: &Tensor<T>) -> impl DeviceOp<Output = Tensor<T>> {
    CopyDeviceToDevice {
        _storage: tensor.storage.clone(),
        src_ptr: tensor.cu_deviceptr(),
        shape: tensor.shape.clone(),
        strides: tensor.strides.clone(),
        num_elements: tensor.size(),
        _dtype: std::marker::PhantomData,
    }
}

/// Copy data from `src` into `dst` without transferring ownership of either.
///
/// Device-to-device copy into an existing buffer.
///
/// Copies the contents of `src` into `dst`. Both must have the same number
/// of elements. No new GPU memory is allocated.
///
/// Accepts any types that deref to `Tensor<T>`: `&Tensor<T>`, `&Arc<Tensor<T>>`,
/// `Arc<Tensor<T>>`, etc.
///
/// # Safety
///
/// This function writes to `dst` through its device pointer. The caller
/// must ensure:
/// - No other operation reads from `dst` concurrently on a different stream.
/// - The copy completes (via stream ordering or synchronization) before
///   `dst` is read.
///
/// This is safe when used with [`CudaGraph::update`](cuda_async::cuda_graph::CudaGraph::update) (stream ordering
/// ensures the copy completes before graph launch) and inside
/// [`CudaGraph::scope`](cuda_async::cuda_graph::CudaGraph::scope)
/// (capture mode records the copy as a graph node).
///
/// ## Panics
///
/// Panics if `src` and `dst` have different element counts.
///
/// ## Examples
///
/// ```rust,ignore
/// // CUDA graph update pattern:
/// graph.update(api::memcpy(&mut self.input, &embedding))?;
///
/// // Scope capture pattern:
/// s.record(api::memcpy(&mut input, &bufs.residual))?;
/// ```
pub fn memcpy<T: DType>(dst: &mut Tensor<T>, src: &Tensor<T>) -> Memcpy {
    assert_eq!(
        src.size(),
        dst.size(),
        "memcpy: src length ({}) != dst length ({})",
        src.size(),
        dst.size(),
    );
    Memcpy {
        src_ptr: src.cu_deviceptr(),
        dst_ptr: dst.cu_deviceptr(),
        len: dst.num_bytes(),
    }
}

/// Unsafe variant of [`memcpy`] that accepts `&Tensor` for the destination.
///
/// Use this in CUDA graph capture scopes and graph `update()` calls where
/// the destination is borrowed immutably but written to through the device
/// pointer during graph replay.
///
pub struct Memcpy {
    src_ptr: cuda_core::sys::CUdeviceptr,
    dst_ptr: cuda_core::sys::CUdeviceptr,
    len: usize,
}

impl DeviceOp for Memcpy {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        memcpy_dtod_async::<u8>(self.dst_ptr, self.src_ptr, self.len, ctx.get_cuda_stream());
        Ok(())
    }
}

impl cuda_async::device_operation::GraphNode for Memcpy {}

impl IntoFuture for Memcpy {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Memcpy>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Device operation for copying a tensor from GPU to CPU as a Vec.
///
/// This internal type implements the async copy operation that transfers
/// data from GPU memory directly to a CPU `Vec<T>`.
struct CopyDeviceToHostVec<T: DType> {
    tensor: Arc<Tensor<T>>,
}

/// Implements the device-to-host-vec copy operation.
///
/// Allocates CPU memory and uses `memcpy_dtoh_async` to transfer data,
/// returning the result as a `Vec<T>` for direct access.
impl<T: DType> DeviceOp for CopyDeviceToHostVec<T> {
    type Output = Vec<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let cu_deviceptr = self.tensor.cu_deviceptr();
        let size = self.tensor.size();
        let layout = Layout::array::<T>(size).expect("overflow cannot happen");
        let async_ptr = unsafe { alloc(layout).cast::<T>() };
        memcpy_dtoh_async(async_ptr, cu_deviceptr, size, ctx.get_cuda_stream());
        Ok(unsafe { Vec::from_raw_parts(async_ptr, size, size) })
    }
}

impl<T: DType> IntoFuture for CopyDeviceToHostVec<T> {
    type Output = Result<Vec<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Vec<T>, CopyDeviceToHostVec<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Copies a GPU tensor to CPU memory as a `Vec<T>`.
///
/// This is an internal function used by the `ToHostVec` trait. Most users should use
/// the `.to_host_vec()` method on tensors instead.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let gpu_tensor = Arc::new(api::arange::<f32>(100).await);
/// let cpu_vec: Vec<f32> = api::copy_device_to_host_vec(&gpu_tensor).await;
/// ```
pub fn copy_device_to_host_vec<T: DType>(
    tensor: &Arc<Tensor<T>>,
) -> impl DeviceOp<Output = Vec<T>> {
    CopyDeviceToHostVec {
        tensor: tensor.clone(),
    }
}

struct CopyHostVecToDevice<T: DType> {
    vec: Arc<Vec<T>>,
}

impl<T: DType> DeviceOp for CopyHostVecToDevice<T> {
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let vec = self.vec;
        let element_size = std::mem::size_of::<T>();
        let num_elements = vec.len();
        let shape = vec![num_elements as i32];
        let strides = vec![1];
        let dptr = ctx.alloc_async(element_size * num_elements);
        memcpy_htod_async(dptr, vec.as_ptr(), num_elements, ctx.get_cuda_stream());
        Ok(Tensor::from_raw_parts(
            dptr,
            element_size * num_elements,
            ctx.get_device_id(),
            shape.clone(),
            strides.clone(),
        ))
    }
}

impl<T: DType> IntoFuture for CopyHostVecToDevice<T> {
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, CopyHostVecToDevice<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

pub fn copy_host_vec_to_device<T: DType>(vec: &Arc<Vec<T>>) -> impl DeviceOp<Output = Tensor<T>> {
    CopyHostVecToDevice { vec: vec.clone() }
}

/// Creates a tensor filled with zeros.
///
/// Allocates GPU memory and fills it with the zero value for type `T`. Supports
/// tensors up to rank 4.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// // 1D tensor
/// let x = api::zeros::<f32>(&[1024]).await;
///
/// // 2D tensor
/// let matrix = api::zeros::<f32>(&[512, 512]).await;
///
/// // 3D tensor
/// let volume = api::zeros::<i32>(&[64, 64, 64]).await;
/// ```
pub fn zeros<T: DType>(shape: &[usize]) -> impl DeviceOp<Output = Tensor<T>> {
    full(T::zero(), shape)
}

/// Creates a tensor filled with ones.
///
/// Allocates GPU memory and fills it with the one value for type `T`. Supports
/// tensors up to rank 4.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let x = api::ones::<f32>(&[1024]).await;
/// let matrix = api::ones::<f16>(&[256, 256]).await;
/// ```
pub fn ones<T: DType>(shape: &[usize]) -> impl DeviceOp<Output = Tensor<T>> {
    full(T::one(), shape)
}

/// Creates a tensor filled with a constant value.
///
/// Allocates GPU memory and fills it with the specified value. This uses a GPU kernel
/// to initialize the memory efficiently.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// // Fill with a specific value
/// let x = api::full(3.14f32, &[1024]).await;
/// let matrix = api::full(-1, &[128, 128]).await;
/// ```
pub fn full<T: DType>(val: T, shape: &[usize]) -> impl DeviceOp<Output = Tensor<T>> {
    let shape = shape.to_vec();
    let len = shape.iter().product::<usize>();
    Tensor::<T>::uninitialized(len).then(move |t| {
        // TODO (hme): It's awkward to assume_init this before actually initializing it.
        let partition_size = 128;
        let result = unsafe { t.assume_init() }.partition([partition_size]);
        let (_, res) = value((val, result)).then(full_apply).unzip();
        res.unpartition().reshape(&shape)
    })
}

pub fn fill<T: DType>(tensor: Tensor<T>, val: T) -> impl DeviceOp<Output = Tensor<T>> {
    value(tensor).then(move |t| {
        let partition_size = 128;
        let result = t.partition([partition_size]);
        let (_, res) = value((val, result)).then(full_apply).unzip();
        res.unpartition()
    })
}

/// Creates a 1D tensor with evenly spaced values from 0 to len-1.
///
/// Similar to NumPy's `arange`, this creates a tensor containing the sequence [0, 1, 2, ..., len-1].
/// The values are generated on the GPU using a kernel.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let indices = api::arange::<i32>(100).await; // [0, 1, 2, ..., 99]
/// let floats = api::arange::<f32>(1000).await;
/// ```
pub fn arange<T: DType>(len: usize) -> impl DeviceOp<Output = Tensor<T>> {
    Tensor::<T>::uninitialized(len).then(move |t| {
        let partition_size = 128;
        let result = unsafe { t.assume_init() }.partition([partition_size]);
        let res = value((result,)).then(arange_apply).unzip();
        res.0.unpartition()
    })
}

/// Creates a 1D tensor with evenly spaced values between `start` and `stop`.
///
/// Similar to NumPy's `linspace`. Generates `n` values such that the first
/// is `start` and the last is `stop` (inclusive on both ends).
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let x = api::linspace(0.0, 1.0, 100).await; // [0.0, 0.0101..., ..., 1.0]
/// let angles = api::linspace(0.0, 6.283, 360).await;
/// ```
/// Creates a 1D tensor with evenly spaced values between `start` and `stop`.
///
/// Similar to NumPy's `linspace`. Generates `n` values such that the first
/// is `start` and the last is `stop` (inclusive on both ends).
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let x = api::linspace(0.0, 1.0, 100).await; // [0.0, 0.0101..., ..., 1.0]
/// ```
pub fn linspace(start: f32, stop: f32, n: usize) -> impl DeviceOp<Output = Tensor<f32>> {
    let step = if n > 1 {
        (stop - start) / (n - 1) as f32
    } else {
        0.0
    };
    Tensor::<f32>::uninitialized(n).then(move |t| {
        let partition_size = 128;
        let result = unsafe { t.assume_init() }.partition([partition_size]);
        linspace_kernel(result, start, step)
            .then(|(tensor, _, _)| value(tensor))
            .unpartition()
    })
}

/// Creates a 2D identity matrix of shape `[n, n]`.
///
/// Elements on the diagonal are 1.0, all others are 0.0.
/// For non-square identity-like matrices, use `eye_rect`.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let I = api::eye(4).await; // 4x4 identity matrix
/// ```
pub fn eye(n: usize) -> impl DeviceOp<Output = Tensor<f32>> {
    eye_rect(n, n)
}

/// Creates a 2D identity-like matrix of shape `[rows, cols]`.
///
/// Elements where row index == column index are 1.0, all others are 0.0.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let rect = api::eye_rect(3, 5).await; // 3x5, ones on main diagonal
/// ```
pub fn eye_rect(rows: usize, cols: usize) -> impl DeviceOp<Output = Tensor<f32>> {
    let len = rows * cols;
    let br = 16;
    let bc = 16;
    Tensor::<f32>::uninitialized(len).then(move |t| {
        let t2d = unsafe { t.assume_init() }
            .reshape(&[rows, cols])
            .expect("eye: reshape failed");
        let result = t2d.partition([br, bc]);
        let res = value((result,)).then(eye_apply).unzip();
        res.0.unpartition()
    })
}

/// Converts a tensor from one element type to another (internal API).
///
/// This is an internal convenience function that creates an uninitialized destination
/// tensor and applies the conversion kernel. Most users should use the conversion kernel
/// directly with explicit partitioning.
///
/// ## Examples
///
/// ```rust,ignore
/// let src_f32 = Arc::new(api::arange::<f32>(1024).await);
/// let dst_f16: Tensor<f16> = convert(src_f32).await;
/// ```
pub fn convert<FromType: DType, ToType: DType>(
    src: Arc<Tensor<FromType>>,
) -> impl DeviceOp<Output = Tensor<ToType>> {
    let len = src.shape.clone().iter().product::<i32>() as usize;
    Tensor::<ToType>::uninitialized(len).then(move |t| {
        let partition_size = 128;
        let dst = unsafe { t.assume_init() }.partition([partition_size]);
        let res = value((src.clone(), dst)).then(convert_apply).unzip();
        res.1
            .unpartition()
            .reshape(&src.shape.iter().map(|x| *x as usize).collect::<Vec<_>>())
    })
}

/// Generates a tensor with values from a normal distribution (f16 version).
///
/// Creates an f16 tensor by first generating f32 values and then converting.
/// This is necessary because cuRAND doesn't directly support f16 generation.
///
/// ## Parameters
///
/// - `mean`: Mean of the normal distribution
/// - `std`: Standard deviation
/// - `shape`: Tensor shape
/// - `seed`: Optional random seed for reproducibility
///
/// Generates a tensor with values from a normal distribution.
///
/// Supports `f32` and `f64` natively via cuRAND. For `f16`, generates `f32`
/// and converts — use `randn_f16` for that case.
pub fn randn<T: DType + RandNormal, const RANK: usize>(
    mean: T,
    std: T,
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOp<Output = Tensor<T>> {
    let len = shape.iter().product::<usize>();
    Tensor::<T>::uninitialized(len).then(move |t| unsafe {
        let t = t.assume_init();
        let rng = RNG::new(seed);
        T::generate_normal(&rng, t.cu_deviceptr(), len, mean, std);
        value(t.reshape_unchecked(&shape))
    })
}

/// Generates a tensor with normally distributed f16 values.
///
/// cuRAND doesn't support f16 natively, so this generates f32 and converts.
pub fn randn_f16<const RANK: usize>(
    mean: f16,
    std: f16,
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOp<Output = Tensor<f16>> {
    let len = shape.clone().iter().product::<usize>();
    randn(mean.to_f32(), std.to_f32(), [len], seed).then(move |src_tensor| {
        let dst = Tensor::<f16>::uninitialized(len);
        dst.then(move |dst_tensor| {
            let partition_size = 128;
            let dst = unsafe { dst_tensor.assume_init() }.partition([partition_size]);
            let res = value((Arc::new(src_tensor), dst))
                .then(convert_apply)
                .unzip();
            res.1.unpartition().reshape(shape.as_ref())
        })
    })
}

/// Generates a tensor with uniformly distributed random values in [0, 1).
///
/// Supports `f32` and `f64` via cuRAND.
pub fn rand<T: DType + RandUniform, const RANK: usize>(
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOp<Output = Tensor<T>> {
    let len = shape.iter().product::<usize>();
    Tensor::<T>::uninitialized(len).then(move |t| unsafe {
        let t = t.assume_init();
        let rng = RNG::new(seed);
        T::generate_uniform(&rng, t.cu_deviceptr(), len);
        value(t.reshape_unchecked(&shape))
    })
}

// Reshape operations

/// Device operation that reshapes a tensor to a new static shape.
/// Device operation that reshapes a tensor output. Works for both owned
/// `Tensor<T>` (reshapes in place) and `Arc<Tensor<T>>` (zero-copy view).
pub struct ReshapeOp<O: Send, DI: DeviceOp<Output = O>> {
    shape: Vec<usize>,
    input: DI,
}

impl<T: DType, DI: DeviceOp<Output = Tensor<T>>> DeviceOp for ReshapeOp<Tensor<T>, DI> {
    type Output = Tensor<T>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Tensor<T>, DeviceError> {
        let tensor = self.input.execute(context)?;
        Ok(tensor.reshape_unchecked(&self.shape))
    }
}

impl<T: DType, DI: DeviceOp<Output = Tensor<T>>> IntoFuture for ReshapeOp<Tensor<T>, DI> {
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

impl<T: DType + Send, DI: DeviceOp<Output = Arc<Tensor<T>>>> DeviceOp
    for ReshapeOp<Arc<Tensor<T>>, DI>
{
    type Output = Arc<Tensor<T>>;

    unsafe fn execute(self, context: &ExecutionContext) -> Result<Arc<Tensor<T>>, DeviceError> {
        let arc_tensor = self.input.execute(context)?;
        arc_tensor
            .reshape_shared(&self.shape)
            .map_err(|e| DeviceError::Internal(e.to_string()))
    }
}

impl<T: DType + Send, DI: DeviceOp<Output = Arc<Tensor<T>>>> IntoFuture
    for ReshapeOp<Arc<Tensor<T>>, DI>
{
    type Output = Result<Arc<Tensor<T>>, DeviceError>;
    type IntoFuture = DeviceFuture<Arc<Tensor<T>>, Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Extension trait: `.reshape(&[usize])` on any `DeviceOp` producing `Tensor<T>`.
pub trait DeviceOpReshape<T: DType>: DeviceOp<Output = Tensor<T>> + Sized {
    fn reshape(self, shape: &[usize]) -> ReshapeOp<Tensor<T>, Self> {
        ReshapeOp {
            shape: shape.to_vec(),
            input: self,
        }
    }
}

impl<T: DType, DI: DeviceOp<Output = Tensor<T>>> DeviceOpReshape<T> for DI {}

/// Extension trait: `.reshape(&[usize])` on any `DeviceOp` producing `Arc<Tensor<T>>`.
pub trait DeviceOpReshapeShared<T: DType + Send>:
    DeviceOp<Output = Arc<Tensor<T>>> + Sized
{
    fn reshape(self, shape: &[usize]) -> ReshapeOp<Arc<Tensor<T>>, Self> {
        ReshapeOp {
            shape: shape.to_vec(),
            input: self,
        }
    }
}

impl<T: DType + Send, DI: DeviceOp<Output = Arc<Tensor<T>>>> DeviceOpReshapeShared<T> for DI {}
