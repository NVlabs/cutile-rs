/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! High-level API for tensor creation and manipulation.
//!
//! This module provides NumPy-like functions for creating and manipulating GPU tensors.
//! All operations are asynchronous and return [`DeviceOperation`]s that can be `.await`ed.
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
//! - [`empty`] - Create uninitialized tensor (unsafe, but fast)
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
//! - [`copy`] - Copy a tensor to new GPU memory
//! - [`copy_to_device`] - Copy CPU tensor to GPU
//! - [`copy_to_host`] - Copy GPU tensor to CPU
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
//!     let zeros = api::zeros::<f32>([1024]).await;
//!     let ones = api::ones::<f32>([512, 512]).await;
//!     let range = api::arange::<i32>(100).await;
//!     let random = api::randn(0.0, 1.0, [256, 256]).await;
//! }
//! ```
//!
//! ### Memory Management
//!
//! ```rust,ignore
//! use cutile::api;
//! use std::sync::Arc;
//!
//! // Create tensor and wrap in Arc for shared ownership
//! let x = Arc::new(api::zeros::<f32>([1024]).await);
//!
//! // Copy to new memory
//! let y = api::copy(&x).await;
//!
//! // Copy to CPU for inspection
//! let cpu_data = api::copy_to_host(&x).await;
//! ```
//!
//! ### Composing Operations
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // Operations compose naturally with async/await
//! let x = api::randn(0.0, 1.0, [1024]).await;
//! let x_arc = Arc::new(x);
//! let y = api::copy(&x_arc).await;
//! let z = y.partition([128]); // Prepare for kernel
//! ```
//!
//! ## Design Philosophy
//!
//! ### Lazy Execution
//!
//! All functions return [`DeviceOperation`]s that don't execute immediately:
//!
//! ```rust,ignore
//! let x = api::zeros([1024]);  // No GPU work yet!
//! let y = api::ones([1024]);   // Still no GPU work!
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
//! let x = api::zeros([256]);           // Shape known at compile time
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
//! - [`tile_async`](crate::tile_async) - Lower-level async execution primitives
//! - [`tensor`](crate::tensor) - Tensor type and partitioning
//! - [`kernels`](crate::kernels) - Pre-built GPU kernels

use crate::kernels::conversion::convert_apply;
use crate::kernels::creation::{arange_apply, full_apply};
use crate::tensor::{IntoPartition, Tensor, Unpartition};
use candle_core::{FloatDType, WithDType};
use cuda_async::device_box::DeviceBox;
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{
    value, DeviceOperation, ExecutionContext, Unzippable1, Unzippable2,
};
use cuda_async::error::{device_error, DeviceError};
use cuda_async::scheduling_policies::SchedulingPolicy;
use cuda_core::curand::RNG;
use cuda_async::device_context::device_alloc_async;
use cuda_core::{memcpy_dtod_async, memcpy_dtoh_async, memcpy_htod_async};
use half::f16;
use std::alloc::{alloc, Layout};
use std::cmp::min;
use std::future::IntoFuture;
use std::marker::PhantomData;
use std::sync::Arc;

/// Device operation for copying a tensor within GPU memory.
///
/// This internal type implements the async copy operation that allocates new
/// GPU memory and copies tensor data device-to-device.
pub struct CopyDeviceToDevice<T: WithDType + Send> {
    tensor: Arc<Tensor<T>>,
}

/// Implements the device-to-device copy operation.
///
/// Allocates new GPU memory asynchronously and uses `memcpy_dtod_async` for
/// efficient GPU-to-GPU data transfer.
impl<T: WithDType + Send> DeviceOperation for CopyDeviceToDevice<T> {
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let tensor = self.tensor;
        let shape = tensor.shape.clone();
        let strides = tensor.strides.clone();
        let element_size = std::mem::size_of::<T>();
        let num_elements = tensor.size();
        let num_bytes = element_size * num_elements;
        let src = tensor.cu_deviceptr();
        let dst = malloc_async(num_bytes, ctx.get_cuda_stream(), ctx.get_device_id());
        memcpy_dtod_async::<T>(dst, src, num_elements, ctx.get_cuda_stream());
        let device_box = DeviceBox::<[T]>::from_raw_parts(dst, num_elements, ctx.get_device_id());
        Ok(Tensor {
            device_box,
            shape: shape.clone(),
            strides: strides.clone(),
        })
    }
}

impl<T: WithDType + Send> IntoFuture for CopyDeviceToDevice<T> {
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, CopyDeviceToDevice<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
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
/// let x = Arc::new(api::zeros::<f32>([1024]).await);
/// let y = api::copy(&x).await;
/// // y is now an independent copy of x
/// ```
pub fn copy<T: WithDType + Send>(
    tensor: &Arc<Tensor<T>>,
) -> impl DeviceOperation<Output = Tensor<T>> {
    CopyDeviceToDevice {
        tensor: tensor.clone(),
    }
}

/// Device operation for copying a tensor from CPU to GPU.
///
/// This internal type implements the async copy operation that transfers
/// data from a Candle CPU tensor to GPU memory.
pub struct CopyHostToDevice<T: WithDType + Send> {
    dtype: PhantomData<T>,
    tensor: Arc<candle_core::Tensor>,
}

/// Implements the host-to-device copy operation.
///
/// Extracts data from the Candle tensor, allocates GPU memory, and uses
/// `memcpy_htod_async` for efficient CPU-to-GPU data transfer.
impl<T: WithDType + Send> DeviceOperation for CopyHostToDevice<T> {
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let tensor = self.tensor;
        let params = candle_tensor_to_vec::<T>(&tensor);
        let (vec, shape, strides) = params;
        let element_size = std::mem::size_of::<T>();
        let num_elements = vec.len();
        let dptr = device_alloc_async(element_size * num_elements, ctx.get_cuda_stream(), ctx.get_device_id());
        memcpy_htod_async(dptr, vec.as_ptr(), num_elements, ctx.get_cuda_stream());
        let device_box = DeviceBox::<[T]>::from_raw_parts(dptr, num_elements, ctx.get_device_id());
        Ok(Tensor {
            device_box,
            shape: shape.clone(),
            strides: strides.clone(),
        })
    }
}

impl<T: WithDType + Send> IntoFuture for CopyHostToDevice<T> {
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, CopyHostToDevice<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Copies a CPU tensor (Candle) to GPU memory.
///
/// Transfers data from a `candle_core::Tensor` on the CPU to a GPU `Tensor<T>`.
/// This is the primary way to move data from host to device.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
/// use std::sync::Arc;
///
/// let cpu_tensor = candle_core::Tensor::zeros((1024,), DType::F32, &Device::Cpu)?;
/// let gpu_tensor: Tensor<f32> = api::copy_to_device(&Arc::new(cpu_tensor)).await;
/// ```
pub fn copy_to_device<T: WithDType + Send>(
    tensor: &Arc<candle_core::Tensor>,
) -> CopyHostToDevice<T> {
    CopyHostToDevice {
        tensor: tensor.clone(),
        dtype: PhantomData,
    }
}

/// Device operation for copying a tensor from GPU to CPU.
///
/// This internal type implements the async copy operation that transfers
/// data from GPU memory to a Candle CPU tensor.
pub struct CopyDeviceToHost<T: WithDType + Send> {
    tensor: Arc<Tensor<T>>,
}

/// Implements the device-to-host copy operation.
///
/// Allocates CPU memory, uses `memcpy_dtoh_async` for GPU-to-CPU transfer,
/// and wraps the result in a Candle tensor.
impl<T: WithDType + Send> DeviceOperation for CopyDeviceToHost<T> {
    type Output = candle_core::Tensor;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let src = self.tensor.device_box.cu_deviceptr();
        let num_elements = self.tensor.size();
        let shape: Vec<usize> = self.tensor.shape.iter().map(|x| *x as usize).collect();
        let layout = Layout::array::<T>(num_elements).expect("overflow cannot happen");
        let dst = alloc(layout).cast::<T>();
        memcpy_dtoh_async(dst, src, num_elements, context.get_cuda_stream());
        let data = Vec::from_raw_parts(dst, num_elements, num_elements);
        let shape = candle_core::Shape::from(shape);
        match candle_core::Tensor::from_vec(data, shape, &candle_core::Device::Cpu) {
            Ok(tensor) => Ok(tensor),
            Err(err) => Err(device_error(
                context.get_device_id(),
                err.to_string().as_str(),
            )),
        }
    }
}

impl<T: WithDType + Send> IntoFuture for CopyDeviceToHost<T> {
    type Output = Result<candle_core::Tensor, DeviceError>;
    type IntoFuture = DeviceFuture<candle_core::Tensor, CopyDeviceToHost<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Copies a GPU tensor to CPU memory as a Candle tensor.
///
/// Transfers data from GPU to a CPU-based `candle_core::Tensor`. Useful when you need
/// to interoperate with other libraries that use Candle tensors.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// let gpu_tensor = Arc::new(api::arange::<f32>(100).await);
/// let cpu_tensor: candle_core::Tensor = api::copy_to_host(&gpu_tensor).await;
/// ```
pub fn copy_to_host<T: WithDType>(tensor: &Arc<Tensor<T>>) -> CopyDeviceToHost<T> {
    CopyDeviceToHost {
        tensor: tensor.clone(),
    }
}

/// Device operation for copying a tensor from GPU to CPU as a Vec.
///
/// This internal type implements the async copy operation that transfers
/// data from GPU memory directly to a CPU `Vec<T>`.
struct CopyDeviceToHostVec<T: WithDType + Send> {
    tensor: Arc<Tensor<T>>,
}

/// Implements the device-to-host-vec copy operation.
///
/// Allocates CPU memory and uses `memcpy_dtoh_async` to transfer data,
/// returning the result as a `Vec<T>` for direct access.
impl<T: WithDType + Send> DeviceOperation for CopyDeviceToHostVec<T> {
    type Output = Vec<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let cu_deviceptr = self.tensor.device_box.cu_deviceptr();
        let size = self.tensor.size();
        let layout = Layout::array::<T>(size).expect("overflow cannot happen");
        let async_ptr = unsafe { alloc(layout).cast::<T>() };
        memcpy_dtoh_async(async_ptr, cu_deviceptr, size, ctx.get_cuda_stream());
        Ok(unsafe { Vec::from_raw_parts(async_ptr, size, size) })
    }
}

impl<T: WithDType + Send> IntoFuture for CopyDeviceToHostVec<T> {
    type Output = Result<Vec<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Vec<T>, CopyDeviceToHostVec<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
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
pub fn copy_device_to_host_vec<T: WithDType>(
    tensor: &Arc<Tensor<T>>,
) -> impl DeviceOperation<Output = Vec<T>> {
    CopyDeviceToHostVec {
        tensor: tensor.clone(),
    }
}

struct CopyHostVecToDevice<T: WithDType + Send> {
    vec: Arc<Vec<T>>,
}

impl<T: WithDType + Send> DeviceOperation for CopyHostVecToDevice<T> {
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let vec = self.vec;
        let element_size = std::mem::size_of::<T>();
        let num_elements = vec.len();
        let shape = vec![num_elements as i32];
        let strides = vec![1];
        let dptr = device_alloc_async(element_size * num_elements, ctx.get_cuda_stream(), ctx.get_device_id());
        memcpy_htod_async(dptr, vec.as_ptr(), num_elements, ctx.get_cuda_stream());
        let device_box = DeviceBox::<[T]>::from_raw_parts(dptr, num_elements, ctx.get_device_id());
        Ok(Tensor {
            device_box,
            shape: shape.clone(),
            strides: strides.clone(),
        })
    }
}

impl<T: WithDType + Send> IntoFuture for CopyHostVecToDevice<T> {
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, CopyHostVecToDevice<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

pub fn copy_host_vec_to_device<T: WithDType>(
    vec: &Arc<Vec<T>>,
) -> impl DeviceOperation<Output = Tensor<T>> {
    CopyHostVecToDevice { vec: vec.clone() }
}

/// Internal helper to extract data from a Candle tensor.
///
/// Converts a Candle tensor to a flat vector along with its shape and stride information.
/// This is used internally for host-to-device data transfers.
///
/// ## Returns
///
/// A tuple of:
/// - `Vec<T>` - Flattened tensor data
/// - `Vec<i32>` - Tensor shape dimensions
/// - `Vec<i32>` - Tensor stride information
pub(crate) fn candle_tensor_to_vec<T: WithDType>(
    tensor: &Arc<candle_core::Tensor>,
) -> (Vec<T>, Vec<i32>, Vec<i32>) {
    let shape: Vec<i32> = tensor.shape().dims().iter().map(|x| *x as i32).collect();
    let strides: Vec<i32> = tensor.stride().iter().map(|x| *x as i32).collect();
    let size: usize = tensor.shape().dims().iter().product();
    let vec = tensor.reshape((size,)).unwrap().to_vec1().unwrap();
    (vec, shape, strides)
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
/// let x = api::zeros::<f32>([1024]).await;
///
/// // 2D tensor
/// let matrix = api::zeros::<f32>([512, 512]).await;
///
/// // 3D tensor
/// let volume = api::zeros::<i32>([64, 64, 64]).await;
/// ```
pub fn zeros<const RANK: usize, T: WithDType>(
    shape: [usize; RANK],
) -> impl DeviceOperation<Output = Tensor<T>> {
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
/// let x = api::ones::<f32>([1024]).await;
/// let matrix = api::ones::<f16>([256, 256]).await;
/// ```
pub fn ones<const RANK: usize, T: WithDType>(
    shape: [usize; RANK],
) -> impl DeviceOperation<Output = Tensor<T>> {
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
/// let x = api::full(3.14f32, [1024]).await;
/// let matrix = api::full(-1, [128, 128]).await;
/// ```
pub fn full<const RANK: usize, T: WithDType>(
    val: T,
    shape: [usize; RANK],
) -> impl DeviceOperation<Output = Tensor<T>> {
    let len = shape.iter().product::<usize>();
    Tensor::<T>::uninitialized(len).and_then(move |t| {
        // TODO (hme): It's awkward to assume_init this before actually initializing it.
        let partition_size = min(len, 128);
        let result = unsafe { t.assume_init() }.partition([partition_size as i32]);
        let (_, res) = value((val, result)).apply(full_apply).unzip();
        res.unpartition().reshape::<RANK>(shape)
    })
}

pub fn fill<T: WithDType>(tensor: Tensor<T>, val: T) -> impl DeviceOperation<Output = Tensor<T>> {
    value(tensor).and_then(move |t| {
        let len = t.shape.iter().product::<i32>() as usize;
        let partition_size = min(len, 128);
        let result = t.partition([partition_size as i32]);
        let (_, res) = value((val, result)).apply(full_apply).unzip();
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
pub fn arange<T: WithDType>(len: usize) -> impl DeviceOperation<Output = Tensor<T>> {
    Tensor::<T>::uninitialized(len).and_then(move |t| {
        let partition_size = min(len, 128);
        let result = unsafe { t.assume_init() }.partition([partition_size as i32]);
        let res = value((result,)).apply(arange_apply).unzip();
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
pub fn convert<FromType: WithDType, ToType: WithDType>(
    src: Arc<Tensor<FromType>>,
) -> impl DeviceOperation<Output = Tensor<ToType>> {
    let len = src.shape.clone().iter().product::<i32>() as usize;
    Tensor::<ToType>::uninitialized(len).and_then(move |t| {
        let partition_size = min(len, 128);
        let dst = unsafe { t.assume_init() }.partition([partition_size as i32]);
        let res = value((src.clone(), dst)).apply(convert_apply).unzip();
        res.1
            .unpartition()
            .reshape_dyn(src.shape.iter().map(|x| *x as usize).collect::<Vec<_>>())
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
pub fn randn_f16<const RANK: usize>(
    mean: f16,
    std: f16,
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOperation<Output = Tensor<f16>> {
    let len = shape.clone().iter().product::<usize>();
    randn_f32(mean.to_f32(), std.to_f32(), [len], seed).and_then(move |src_tensor| {
        let dst = Tensor::<f16>::uninitialized(len);
        dst.and_then(move |dst_tensor| {
            let partition_size = min(len, 128);
            let dst = unsafe { dst_tensor.assume_init() }.partition([partition_size as i32]);
            let res = value((Arc::new(src_tensor), dst))
                .apply(convert_apply)
                .unzip();
            res.1.unpartition().reshape_dyn(shape.to_vec())
        })
    })
}

/// Generates a tensor with values from a normal distribution (f32 version).
///
/// Uses cuRAND to generate random values directly on the GPU with the specified
/// mean and standard deviation.
///
/// ## Parameters
///
/// - `mean`: Mean of the normal distribution
/// - `std`: Standard deviation
/// - `shape`: Tensor shape
/// - `seed`: Optional random seed for reproducibility
pub fn randn_f32<const RANK: usize>(
    mean: f32,
    std: f32,
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOperation<Output = Tensor<f32>> {
    let len = shape.iter().product::<usize>();
    Tensor::<f32>::uninitialized(len).and_then(move |t| unsafe {
        let t = t.assume_init();
        let rng = RNG::new(seed);
        rng.generate_normal_f32(t.cu_deviceptr(), len, mean, std);
        value(t.reshape::<RANK>(shape))
    })
}

/// Generates a tensor with values from a normal distribution (f64 version).
///
/// Uses cuRAND to generate random double-precision values on the GPU.
///
/// ## Parameters
///
/// - `mean`: Mean of the normal distribution
/// - `std`: Standard deviation
/// - `shape`: Tensor shape
/// - `seed`: Optional random seed for reproducibility
pub fn randn_f64<const RANK: usize>(
    mean: f64,
    std: f64,
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOperation<Output = Tensor<f64>> {
    let len = shape.iter().product::<usize>();
    Tensor::<f64>::uninitialized(len).and_then(move |t| unsafe {
        let t = t.assume_init();
        let rng = RNG::new(seed);
        rng.generate_normal_f64(t.cu_deviceptr(), len, mean, std);
        value(t.reshape::<RANK>(shape))
    })
}

/// Generates a tensor with uniformly distributed random values in [0, 1) (f32).
///
/// Uses cuRAND to generate uniform random values on the GPU.
///
/// ## Parameters
///
/// - `shape`: Tensor shape
/// - `seed`: Optional random seed for reproducibility
pub fn rand_f32<const RANK: usize>(
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOperation<Output = Tensor<f32>> {
    let len = shape.iter().product::<usize>();
    Tensor::<f32>::uninitialized(len).and_then(move |t| unsafe {
        let t = t.assume_init();
        let rng = RNG::new(seed);
        rng.generate_uniform_f32(t.cu_deviceptr(), len);
        value(t.reshape::<RANK>(shape))
    })
}

/// Generates a tensor with uniformly distributed random values in [0, 1) (f64).
///
/// Uses cuRAND to generate uniform random double-precision values on the GPU.
///
/// ## Parameters
///
/// - `shape`: Tensor shape
/// - `seed`: Optional random seed for reproducibility
pub fn rand_f64<const RANK: usize>(
    shape: [usize; RANK],
    seed: Option<u64>,
) -> impl DeviceOperation<Output = Tensor<f64>> {
    let len = shape.iter().product::<usize>();
    Tensor::<f64>::uninitialized(len).and_then(move |t| unsafe {
        let t = t.assume_init();
        let rng = RNG::new(seed);
        rng.generate_uniform_f64(t.cu_deviceptr(), len);
        value(t.reshape::<RANK>(shape))
    })
}

pub fn randn<const RANK: usize, T: FloatDType>(
    mean: T,
    std: T,
    shape: [usize; RANK],
) -> impl DeviceOperation<Output = Tensor<T>> {
    // TODO (hme): No random number generator for TileIR?
    let t = candle_core::Tensor::randn(mean, std, &shape, &candle_core::Device::Cpu)
        .expect("randn failed.");
    copy_to_device(&Arc::new(t))
}

// Reshape operations

/// Device operation that reshapes a tensor to a new static shape.
///
/// This wraps another device operation and reshapes its tensor output.
/// The reshape is performed after the input operation executes.
pub struct Reshape<const RANK: usize, T: WithDType + Send, DI: DeviceOperation<Output = Tensor<T>>>
{
    shape: [usize; RANK],
    input: DI,
}

/// Implements reshape as a device operation.
///
/// Executes the input operation and then reshapes the resulting tensor.
impl<const RANK: usize, T: WithDType + Send, DI: DeviceOperation<Output = Tensor<T>>>
    DeviceOperation for Reshape<RANK, T, DI>
{
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let tensor = self.input.execute(context)?;
        Ok(tensor.reshape(self.shape))
    }
}

impl<const RANK: usize, T: WithDType + Send, DI: DeviceOperation<Output = Tensor<T>>> IntoFuture
    for Reshape<RANK, T, DI>
{
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, Reshape<RANK, T, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Extension trait for reshaping tensor operations.
///
/// This trait allows device operations that produce tensors to be reshaped without
/// materializing the intermediate tensor. The reshape happens as part of the operation chain.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// // Reshape after creation
/// let x = api::arange::<f32>(1024).reshape([32, 32]).await;
///
/// // Chain with other operations
/// let y = api::zeros([256])
///     .reshape([16, 16])
///     .await;
/// ```
pub trait DeviceOperationReshape<T, DI>
where
    T: Send + WithDType,
    DI: DeviceOperation<Output = Tensor<T>>,
{
    /// Reshapes the output tensor of this operation to the specified shape.
    ///
    /// The new shape must have the same total number of elements as the original.
    fn reshape<const RANK: usize>(self, shape: [usize; RANK]) -> Reshape<RANK, T, DI>;
}

impl<T, DI> DeviceOperationReshape<T, DI> for DI
where
    T: Send + WithDType,
    DI: DeviceOperation<Output = Tensor<T>>,
{
    fn reshape<const RANK: usize>(self, shape: [usize; RANK]) -> Reshape<RANK, T, DI>
    where
        Self: Sized,
    {
        Reshape::<RANK, T, DI> { shape, input: self }
    }
}

// DynamicReshape.

/// Device operation that reshapes a tensor to a new dynamic shape.
///
/// Similar to [`Reshape`] but uses a runtime-determined shape (`Vec<usize>`)
/// instead of a compile-time constant array.
pub struct DynamicReshape<T: WithDType + Send, DI: DeviceOperation<Output = Tensor<T>>> {
    shape: Vec<usize>,
    input: DI,
}

/// Implements dynamic reshape as a device operation.
///
/// Executes the input operation and then reshapes the resulting tensor
/// using the runtime-specified shape.
impl<T: WithDType + Send, DI: DeviceOperation<Output = Tensor<T>>> DeviceOperation
    for DynamicReshape<T, DI>
{
    type Output = Tensor<T>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOperation>::Output, DeviceError> {
        let tensor = self.input.execute(context)?;
        Ok(tensor.reshape_dyn(&self.shape))
    }
}

impl<T: WithDType + Send, DI: DeviceOperation<Output = Tensor<T>>> IntoFuture
    for DynamicReshape<T, DI>
{
    type Output = Result<Tensor<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Tensor<T>, DynamicReshape<T, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Extension trait for dynamically reshaping device operation outputs.
///
/// This trait enables chaining a reshape operation with a runtime-determined shape
/// onto any device operation that produces a tensor.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api::{self, DeviceOperationDynamicReshape};
///
/// let shape_vec = vec![32, 32];  // Runtime-determined shape
/// let tensor = api::arange::<f32>(1024)
///     .reshape_dyn(shape_vec)
///     .await;
/// ```
pub trait DeviceOperationDynamicReshape<T, DI>
where
    T: Send + WithDType,
    DI: DeviceOperation<Output = Tensor<T>>,
{
    /// Reshapes the output tensor using a runtime-specified shape.
    ///
    /// The new shape must have the same total number of elements as the original.
    fn reshape_dyn(self, shape: Vec<usize>) -> DynamicReshape<T, DI>;
}

impl<T, DI> DeviceOperationDynamicReshape<T, DI> for DI
where
    T: Send + WithDType,
    DI: DeviceOperation<Output = Tensor<T>>,
{
    fn reshape_dyn(self, shape: Vec<usize>) -> DynamicReshape<T, DI>
    where
        Self: Sized,
    {
        DynamicReshape::<T, DI> { shape, input: self }
    }
}
