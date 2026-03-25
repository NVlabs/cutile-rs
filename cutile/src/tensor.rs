/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! GPU tensor types and partitioning primitives.
//!
//! This module provides the core [`Tensor`] type for GPU memory management and the [`Partition`]
//! type for dividing tensors into tiles that map to CUDA thread blocks.
//!
//! ## Overview
//!
//! This module is the foundation for GPU memory management in cuTile Rust. It provides:
//!
//! - **[`Tensor`]** - Smart pointer to GPU memory with shape and stride information
//! - **[`Partition`]** - View of a tensor divided into tiles for parallel processing
//! - **Traits** - For converting between tensors, partitions, and device operations
//!
//! ## Core Types
//!
//! ### Tensor
//!
//! A [`Tensor<T>`] represents a multi-dimensional array stored in GPU memory. Key features:
//!
//! - **Automatic memory management**: Uses RAII via [`DeviceBox`]
//! - **Shape tracking**: Maintains shape and stride information
//! - **Zero-copy operations**: Reshape and view operations don't copy data
//! - **Safe concurrency**: `Send + Sync` for safe sharing across async tasks
//!
//! ### Partition
//!
//! A [`Partition<Tensor<T>>`] divides a tensor into tiles (blocks) for GPU kernels. Each tile
//! maps to one CUDA thread block, enabling efficient parallel processing.
//!
//! Key features:
//! - **Grid inference**: Automatically calculates launch grid from partition shape
//! - **Shape validation**: Ensures tensor shape is evenly divisible by partition shape
//! - **Zero-cost abstraction**: No runtime overhead, just metadata
//!
//! ## Traits
//!
//! ### IntoPartition
//!
//! The [`IntoPartition`] trait enables partitioning tensors:
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::IntoPartition;
//!
//! let tensor = api::zeros([256]).await;
//! let partitioned = tensor.partition([64]);  // 4 tiles
//! assert_eq!(partitioned.grid(), (4, 1, 1));
//! ```
//!
//! ### Unpartition
//!
//! The [`Unpartition`] trait removes partition structure, returning the underlying tensor:
//!
//! ```rust,ignore
//! let tensor = partitioned.unpartition();
//! ```
//!
//! ### ToHostVec
//!
//! The [`ToHostVec`] trait provides convenient GPU → CPU data transfer:
//!
//! ```rust,ignore
//! use cutile::tensor::ToHostVec;
//!
//! let tensor = api::ones([1024]).await;
//! let host_vec: Vec<f32> = tensor.to_host_vec().await;
//! ```
//!
//! ## Memory Layout
//!
//! Tensors use row-major (C-style) memory layout by default:
//!
//! ```text
//! 2D Tensor [3, 4]:
//! Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
//! Shape:  +-------------+
//!         | 0  1  2  3  |  Row 0
//!         | 4  5  6  7  |  Row 1
//!         | 8  9 10 11  |  Row 2
//!         +-------------+
//! Strides: [4, 1]  (4 elements between rows, 1 between columns)
//! ```
//!
//! ## Partitioning Example
//!
//! ```text
//! Tensor [256] partitioned into [64]:
//!
//! +---------+---------+---------+---------+
//! | Tile 0  | Tile 1  | Tile 2  | Tile 3  |
//! | [0:64)  | [64:128)| [128:192| [192:256|
//! +---------+---------+---------+---------+
//!
//! Launch grid: (4, 1, 1)
//! Each CUDA block processes one tile (64 elements)
//! ```
//!
//! ```text
//! Tensor [128, 128] partitioned into [32, 32]:
//!
//! +------+------+------+------+
//! | 0,0  | 0,1  | 0,2  | 0,3  |  4x4 grid of tiles
//! +------+------+------+------+  Each tile: 32x32 elements
//! | 1,0  | 1,1  | 1,2  | 1,3  |  Grid: (4, 4, 1)
//! +------+------+------+------+  Total: 16 thread blocks
//! | 2,0  | 2,1  | 2,2  | 2,3  |
//! +------+------+------+------+
//! | 3,0  | 3,1  | 3,2  | 3,3  |
//! +------+------+------+------+
//! ```
//!
//! ## Examples
//!
//! ### Basic Tensor Operations
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // Create tensor
//! let tensor = api::zeros::<f32>([1024]).await;
//!
//! // Access properties
//! println!("Shape: {:?}", tensor.shape);
//! println!("Size: {}", tensor.size());
//! println!("Bytes: {}", tensor.num_bytes());
//! ```
//!
//! ### Partitioning for Kernels
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::IntoPartition;
//!
//! let tensor = api::zeros([256]).await;
//! let partitioned = tensor.partition([64]);
//!
//! // Use in kernel launch
//! // Each of 4 thread blocks processes 64 elements
//! ```
//!
//! ### Copying to Host
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::ToHostVec;
//!
//! let gpu_tensor = api::ones([1024]).await;
//! let cpu_vec: Vec<f32> = gpu_tensor.to_host_vec().await;
//! assert_eq!(cpu_vec.len(), 1024);
//! ```
//!
//! ### Working with Arc
//!
//! ```rust,ignore
//! use cutile::api;
//! use cutile::tensor::IntoPartitionArc;
//! use std::sync::Arc;
//!
//! let tensor = Arc::new(api::zeros([256]).await);
//!
//! // Can partition Arc<Tensor> directly
//! let partitioned = tensor.partition_arc([64]);
//! ```
//!
//! ## Safety and Concurrency
//!
//! ### Thread Safety
//!
//! - `Tensor<T>` is `Send + Sync` - safe to share across threads
//! - `Partition<Tensor<T>>` is `Send + Sync` but not `Clone`
//! - GPU memory is freed automatically when the last reference is dropped
//!
//! ### Memory Safety
//!
//! Partitioning ensures that each thread block accesses disjoint memory regions:
//!
//! ```rust,ignore
//! // Safe: Each block writes to non-overlapping tiles
//! let z = api::zeros([256]).partition([64]);
//! // Block 0: writes to [0:64)
//! // Block 1: writes to [64:128)
//! // Block 2: writes to [128:192)
//! // Block 3: writes to [192:256)
//! ```
//!
//! ## Performance Considerations
//!
//! - **Partitioning**: Zero-cost abstraction (just metadata)
//! - **Reshaping**: Zero-cost (updates strides, no data copy)
//! - **Copying**: Expensive (requires GPU memory bandwidth)
//! - **Host transfers**: Very expensive (PCIe bandwidth-limited)
//!
//! ## See Also
//!
//! - [`api`](crate::api) - High-level tensor creation functions
//! - [`tile_async`](crate::tile_async) - Async execution infrastructure
//! - [`core`](crate::core) - GPU kernel DSL types

use crate::api::{
    copy, copy_device_to_host_vec, copy_host_vec_to_device, copy_to_device, copy_to_host,
};
use crate::error::{tensor_error_result, Error};
use crate::tile_kernel::UnwrapPartition;
use anyhow::Result;
use candle_core::{DType, WithDType};
use cuda_async::device_box::{DeviceBox, DevicePointer};
use cuda_async::device_context::{device_alloc_async, with_default_device_policy};
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation;
use cuda_async::device_operation::{value, DeviceOperation, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::sys::CUdeviceptr;
use cuda_core::CudaStream;
use std::marker::PhantomData;
use std::future::IntoFuture;
use std::mem::MaybeUninit;
use std::fmt::Debug;
use std::ops::Index;
use std::sync::Arc;

/// A partitioned view of a tensor that divides it into tiles for GPU kernel processing.
///
/// `Partition` wraps a tensor and adds partition shape and stride information, enabling
/// tile-based GPU kernels to process the data in blocks that map to CUDA thread blocks.
/// Each thread block processes one partition (tile) of the tensor.
///
/// ## Memory Safety
///
/// This type is `Send + Sync` but not `Clone` or `Copy`. It provides tile kernels with
/// mutable access to disjoint regions of memory, making parallel access safe. When wrapped
/// in an `Arc`, the Arc prevents mutable access, maintaining safety.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::api;
///
/// // Create a tensor and partition it into 64-element tiles
/// let tensor = api::ones([256]).await;
/// let partitioned = tensor.partition([64]);
///
/// // The partition has 4 tiles: (256 / 64 = 4)
/// // Grid will be (4, 1, 1)
/// assert_eq!(partitioned.grid(), (4, 1, 1));
/// ```
///
/// ## Grid Inference
///
/// Partitions automatically calculate the launch grid for kernels:
///
/// ```rust,ignore
/// let x = api::zeros([128, 128]).partition([32, 32]);
/// assert_eq!(x.grid(), (4, 4, 1)); // 128/32 = 4 in each dimension
/// ```
pub struct Partition<T> {
    pub(crate) object: T,
    pub partition_shape: Vec<i32>,
    pub partition_strides: Vec<i32>,
}

impl<T> Partition<T> {
    /// Unwraps the partition to retrieve the underlying object.
    ///
    /// This consumes the partition and returns the original tensor or value.
    pub fn unpartition(self) -> T {
        self.object
    }
}

impl<T: WithDType> Partition<Tensor<T>> {
    /// Returns the total size of the tensor in bytes.
    pub fn num_bytes(&self) -> usize {
        self.object.size() * size_of::<T>()
    }

    /// Returns the size of the tensor in megabytes (base 10).
    pub fn num_mb(&self) -> usize {
        self.num_bytes() / 10usize.pow(6)
    }

    /// Returns the size of the tensor in gigabytes (base 10).
    pub fn num_gb(&self) -> usize {
        self.num_bytes() / 10usize.pow(9)
    }

    /// Returns the data type of the tensor elements.
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    /// Calculates the CUDA launch grid dimensions based on the partition.
    ///
    /// The grid is computed as `tensor_shape / partition_shape` for each dimension.
    /// Supports 1D, 2D, and 3D tensors.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x = api::zeros([256]).partition([64]);
    /// assert_eq!(x.grid(), (4, 1, 1));
    ///
    /// let y = api::zeros([128, 256]).partition([32, 64]);
    /// assert_eq!(y.grid(), (4, 4, 1));
    /// ```
    ///
    /// ## Panics
    ///
    /// Panics if the tensor rank is greater than 3.
    pub fn grid(&self) -> Result<(u32, u32, u32), Error> {
        let check_i32 = |x: &i32| *x > 0;
        if !self.object.shape.iter().all(check_i32) {
            // TODO (hme): This check may be relaxed or unnecessary if we let shapes be u32.
            //  Doing so can't break future features around dynamic shape dims in tile kernels.
            return tensor_error_result("Shape dimensions must be positive.");
        }
        let to_u32 = |x: &i32| *x as u32;
        let shape = self.object.shape.iter().map(to_u32).collect::<Vec<u32>>();
        let partition_shape = self
            .partition_shape
            .iter()
            .map(to_u32)
            .collect::<Vec<u32>>();
        let rank = shape.len();
        match rank {
            1 => Ok((u32::div_ceil(shape[0], partition_shape[0]), 1, 1)),
            2 => Ok((
                u32::div_ceil(shape[0], partition_shape[0]),
                u32::div_ceil(shape[1], partition_shape[1]),
                1,
            )),
            3 => Ok((
                u32::div_ceil(shape[0], partition_shape[0]),
                u32::div_ceil(shape[1], partition_shape[1]),
                u32::div_ceil(shape[2], partition_shape[2]),
            )),
            _ => tensor_error_result("Mutable tensor must be at most rank 3."),
        }
    }
}

impl<T> From<Partition<T>> for Arc<T> {
    fn from(val: Partition<T>) -> Self {
        Arc::new(val.unpartition())
    }
}

/// Enables partitioning a value into tiles.
///
/// This trait allows values to be divided into partitions for tile-based processing.
/// The partition shape determines how the value is divided across thread blocks.
pub trait IntoPartition {
    /// Partitions this value with the specified partition shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor = api::zeros([1024]).await;
    /// let partitioned = tensor.partition([128]); // 8 partitions
    /// ```
    fn partition<const RANK: usize>(self, partition_shape: [i32; RANK]) -> Partition<Self>
    where
        Self: Sized;
}

/// Enables partitioning an `Arc`-wrapped value into tiles.
///
/// This trait is similar to [`IntoPartition`] but works with `Arc`-wrapped values,
/// consuming the `Arc` to create a partition. This is commonly used with async operations.
pub trait IntoPartitionArc {
    /// Partitions this Arc-wrapped value with the specified partition shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor = Arc::new(api::zeros([1024]).await);
    /// let partitioned = tensor.partition([128]);
    /// ```
    fn partition<const RANK: usize>(
        self: Arc<Self>,
        partition_shape: [i32; RANK],
    ) -> Partition<Self>
    where
        Self: Sized;
}

/// Device operation for allocating uninitialized tensor memory.
struct AllocUninitializedTensor<T: WithDType + Send> {
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: WithDType + Send> Send for AllocUninitializedTensor<T> {}

impl<T: WithDType + Send> DeviceOperation for AllocUninitializedTensor<T> {
    type Output = MaybeUninit<Tensor<T>>;

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<MaybeUninit<Tensor<T>>, DeviceError> {
        let num_bytes = self.len * size_of::<T>();
        let dptr = device_alloc_async(num_bytes, ctx.get_cuda_stream(), ctx.get_device_id())?;
        Ok(MaybeUninit::new(Tensor {
            device_box: DeviceBox::from_raw_parts(dptr, self.len, ctx.get_device_id()),
            shape: vec![self.len as i32],
            strides: vec![1],
        }))
    }
}

impl<T: WithDType + Send> IntoFuture for AllocUninitializedTensor<T> {
    type Output = Result<MaybeUninit<Tensor<T>>, DeviceError>;
    type IntoFuture = DeviceFuture<MaybeUninit<Tensor<T>>, AllocUninitializedTensor<T>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// A multi-dimensional array stored in GPU memory.
///
/// `Tensor` is the primary type for working with GPU data in cuTile Rust. It wraps a
/// [`DeviceBox`] with shape and stride information, providing a typed, multi-dimensional
/// view of GPU memory.
///
/// ## Memory Management
///
/// Tensors own their GPU memory through a `DeviceBox`. Memory is automatically freed
/// when the tensor is dropped. For shared ownership, use `Arc<Tensor<T>>`.
///
/// ## Examples
///
/// ### Creating tensors
///
/// ```rust,ignore
/// use cutile::api;
///
/// // Create tensors using the API
/// let x = api::zeros::<f32>([1024]).await;
/// let y = api::ones::<f32>([512, 512]).await;
/// let z = api::arange::<i32>(256).await;
/// ```
///
/// ### Copying and reshaping
///
/// ```rust,ignore
/// let x = api::zeros([1024]).await;
/// let x_arc = Arc::new(x);
///
/// // Copy to create a new tensor
/// let y = x_arc.copy().await;
///
/// // Reshape (must preserve total size)
/// let reshaped = y.reshape([32, 32]); // 1024 = 32 * 32
/// ```
///
/// ### Transferring to host
///
/// ```rust,ignore
/// use cutile::tensor::ToHostVec;
///
/// let gpu_tensor = api::arange::<f32>(100).await;
/// let cpu_vec: Vec<f32> = gpu_tensor.to_host_vec().await;
/// ```
#[derive(Debug)]
pub struct Tensor<T: WithDType> {
    pub device_box: DeviceBox<[T]>,
    pub shape: Vec<i32>,
    pub strides: Vec<i32>,
}

impl<T: WithDType> Tensor<T> {
    /// Allocates uninitialized GPU memory for a 1D tensor.
    ///
    /// ## Safety
    ///
    /// The returned `DeviceOperation` yields a `MaybeUninit<Tensor<T>>` that must
    /// be initialized by a kernel before calling `assume_init()`.
    ///
    /// # Errors
    ///
    /// Returns `DeviceError` if memory allocation fails (e.g., OOM).
    pub fn uninitialized(len: usize) -> impl DeviceOperation<Output = MaybeUninit<Self>> {
        assert!(len > 0, "Non-zero length required.");
        AllocUninitializedTensor::<T> {
            len,
            _marker: PhantomData,
        }
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.device_box.cu_deviceptr()
    }

    /// Returns a typed device pointer.
    pub fn device_pointer(&self) -> DevicePointer<T> {
        self.device_box.device_pointer()
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        self.device_box.len()
    }

    /// Creates a copy of this tensor on the GPU.
    ///
    /// Returns a device operation that, when executed, will allocate new GPU memory
    /// and copy the tensor's data.
    pub fn copy(self: &Arc<Self>) -> impl DeviceOperation<Output = Self> {
        copy(self)
    }

    /// Synchronously copies this tensor on the GPU using the specified stream.
    pub fn copy_sync(self: &Arc<Self>, stream: &Arc<CudaStream>) -> Result<Self, DeviceError> {
        copy(self).sync_on(stream)
    }

    /// Returns the total size of the tensor in bytes.
    pub fn num_bytes(self: &Arc<Self>) -> usize {
        self.size() * size_of::<T>()
    }

    /// Returns the size of the tensor in megabytes (base 10).
    pub fn num_mb(self: &Arc<Self>) -> usize {
        self.num_bytes() / 10usize.pow(6)
    }

    /// Returns the size of the tensor in gigabytes (base 10).
    pub fn num_gb(self: &Arc<Self>) -> usize {
        self.num_bytes() / 10usize.pow(9)
    }

    /// Reshapes the tensor to a new shape without copying data.
    ///
    /// The new shape must have the same total number of elements as the original.
    /// This operation updates the shape and stride information but does not move data.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x = api::arange::<f32>(1024).await;
    /// let reshaped = x.reshape([32, 32]); // 1024 = 32 * 32
    /// ```
    ///
    /// ## Panics
    ///
    /// Panics if:
    /// - The new shape has a different total number of elements
    /// - The rank is greater than 4
    pub fn reshape<const RANK: usize>(mut self, shape: [usize; RANK]) -> Self {
        // Make sure it's a valid shape for this tensor.
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
        assert_eq!(
            shape.iter().product::<i32>(),
            self.shape.iter().product::<i32>()
        );
        self.shape = shape.to_vec();
        match RANK {
            1 => self.strides = vec![1],
            2 => self.strides = vec![shape[1], 1],
            3 => self.strides = vec![shape[1] * shape[2], shape[2], 1],
            4 => {
                self.strides = vec![
                    shape[1] * shape[2] * shape[3],
                    shape[2] * shape[3],
                    shape[3],
                    1,
                ]
            }
            _ => unimplemented!("Static reshape of rank {}", RANK),
        }
        self
    }
    pub fn reshape_dyn(mut self, shape: &[usize]) -> Self {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
        assert_eq!(
            shape.iter().product::<i32>(),
            self.shape.iter().product::<i32>()
        );
        self.shape = shape.to_vec();
        let mut stride = 1;
        let mut strides = Vec::with_capacity(shape.len());
        for i in (0..shape.len()).rev() {
            strides.insert(0, stride);
            stride *= shape[i]
        }
        self.strides = strides;
        self
    }
}

/// Converts a GPU tensor to a host-side vector.
///
/// This trait provides a method to asynchronously copy tensor data from GPU to CPU memory
/// as a `Vec<T>`. Implemented for both owned tensors and `Arc<Tensor<T>>`.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tensor::ToHostVec;
///
/// let gpu_tensor = api::arange::<f32>(100).await;
/// let cpu_data: Vec<f32> = gpu_tensor.to_host_vec().await;
/// assert_eq!(cpu_data.len(), 100);
/// ```
pub trait ToHostVec<T: Send> {
    /// Copies the tensor data from GPU to host memory, returning a `Vec<T>`.
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>>;
}

impl<T: WithDType> ToHostVec<T> for Tensor<T> {
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>> {
        let arc_self = Arc::new(self);
        copy_device_to_host_vec(&arc_self)
    }
}

impl<T: WithDType> ToHostVec<T> for Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>> {
        copy_device_to_host_vec(&self)
    }
}

impl<T: WithDType> ToHostVec<T> for &Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>> {
        copy_device_to_host_vec(self)
    }
}

impl<T: WithDType + Debug> IntoPartitionArc for Tensor<T> {
    fn partition<const RANK: usize>(
        self: Arc<Tensor<T>>,
        partition_shape: [i32; RANK],
    ) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides = self.strides.clone();
        let tensor = Arc::try_unwrap(self).expect("Failed to convert Arc to Partition.");
        Partition::<Tensor<T>> {
            object: tensor,
            partition_shape,
            partition_strides,
        }
    }
}

impl<T: WithDType> IntoPartition for Tensor<T> {
    fn partition<const RANK: usize>(self, partition_shape: [i32; RANK]) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides = self.strides.clone();
        Partition::<Tensor<T>> {
            object: self,
            partition_shape,
            partition_strides,
        }
    }
}

/// Converts a Candle tensor (CPU) to a cuTile Rust tensor (GPU).
///
/// This trait provides a method to copy data from a CPU-based `candle_core::Tensor`
/// to a GPU-based `Tensor<T>`.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tensor::CopyToDevice;
/// use std::sync::Arc;
///
/// let cpu_tensor = candle_core::Tensor::zeros((1024,), DType::F32, &Device::Cpu)?;
/// let cpu_arc = Arc::new(cpu_tensor);
/// let gpu_tensor: Arc<Tensor<f32>> = cpu_arc.copy_to_device().await;
/// ```
pub trait CopyToDevice {
    /// Copies this CPU tensor to GPU memory.
    fn copy_to_device<T: WithDType>(
        self: &Arc<Self>,
    ) -> impl DeviceOperation<Output = Arc<Tensor<T>>>;
}

pub trait CopyToHost {
    fn copy_to_host(self) -> impl DeviceOperation<Output = candle_core::Tensor>;
}

impl CopyToDevice for candle_core::Tensor {
    fn copy_to_device<T: WithDType>(
        self: &Arc<Self>,
    ) -> impl DeviceOperation<Output = Arc<Tensor<T>>> {
        copy_to_device(self).arc()
    }
}

pub trait CopyToDeviceTensor<T: WithDType> {
    fn copy_to_device_tensor(self: &Arc<Self>) -> impl DeviceOperation<Output = Tensor<T>>;
}

impl<T: WithDType> CopyToDeviceTensor<T> for Vec<T> {
    fn copy_to_device_tensor(self: &Arc<Self>) -> impl DeviceOperation<Output = Tensor<T>> {
        copy_host_vec_to_device(self)
    }
}

impl<T: WithDType> CopyToHost for &Arc<Tensor<T>> {
    fn copy_to_host(self) -> impl DeviceOperation<Output = candle_core::Tensor> {
        copy_to_host(self)
    }
}

impl<T: WithDType> CopyToHost for Tensor<T> {
    fn copy_to_host(self) -> impl DeviceOperation<Output = candle_core::Tensor> {
        copy_to_host(&Arc::new(self))
    }
}

pub trait Unpartition<T: WithDType> {
    /// Unwraps the partition to produce the underlying value.
    fn unpartition(self) -> impl DeviceOperation<Output = Tensor<T>>;
}

impl<T: WithDType, DI: DeviceOperation<Output = Partition<Tensor<T>>>> Unpartition<T> for DI {
    fn unpartition(self) -> impl DeviceOperation<Output = Tensor<T>> {
        UnwrapPartition { op: self }
    }
}

// Preliminary support for vectors of tensors is done by providing an unsafe interior mutability pattern.
#[derive(Clone, Debug)]
pub struct DeviceVec<T> {
    _ty: PhantomData<T>,
    host_vec: Vec<Arc<T>>,
    device_vec: Arc<Tensor<i64>>,
}

impl<T: WithDType> DeviceVec<Tensor<T>> {
    pub fn from(v: Vec<Tensor<T>>) -> DeviceVec<Tensor<T>> {
        let i64vec: Arc<Vec<i64>> = v
            .iter()
            .map(|x| x.cu_deviceptr() as i64)
            .collect::<Vec<_>>()
            .into();
        let device_vec: Arc<Tensor<i64>> = i64vec
            .copy_to_device_tensor()
            .sync()
            .expect("Failed to execute device operation.")
            .reshape([v.len()])
            .into();
        let host_vec: Vec<Arc<Tensor<T>>> = v.into_iter().map(Arc::new).collect::<Vec<_>>();
        DeviceVec {
            _ty: PhantomData,
            host_vec,
            device_vec,
        }
    }
    pub fn len(&self) -> usize {
        self.host_vec.len()
    }
    pub unsafe fn inner(&self) -> &Arc<Tensor<i64>> {
        &self.device_vec
    }
}

impl<T: WithDType> From<Vec<Tensor<T>>> for DeviceVec<Tensor<T>> {
    fn from(v: Vec<Tensor<T>>) -> Self {
        DeviceVec::from(v)
    }
}

impl<T: WithDType> Index<usize> for DeviceVec<Tensor<T>> {
    type Output = Arc<Tensor<T>>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.host_vec[index]
    }
}

pub struct DeviceVecIntoIter<Item> {
    items: DeviceVec<Item>,
}

impl<T: WithDType + Debug> Iterator for DeviceVecIntoIter<Tensor<T>> {
    type Item = Tensor<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.items.len() > 0 {
            let x = self.items.host_vec.remove(0);
            let x = Arc::try_unwrap(x).expect("Unable to perform into_iter from non-unique Arc.");
            Some(x)
        } else {
            None
        }
    }
}

impl<T: WithDType + Debug> IntoIterator for DeviceVec<Tensor<T>> {
    type Item = Tensor<T>;
    type IntoIter = DeviceVecIntoIter<Tensor<T>>;
    fn into_iter(self) -> Self::IntoIter {
        DeviceVecIntoIter { items: self }
    }
}
