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
//! - **Automatic memory management**: Uses RAII via [`DeviceBuffer`]
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

use crate::api::{copy, copy_device_to_host_vec, copy_host_vec_to_device};
use crate::error::{tensor_error_result, Error};
use crate::tile_kernel::UnwrapPartition;
use anyhow::Result;
use cuda_async::device_buffer::{DeviceBuffer, DevicePointer};
use cuda_async::device_operation;
use cuda_async::device_operation::{value, DeviceOperation};
use cuda_async::error::DeviceError;
use cuda_core::sys::CUdeviceptr;
use cuda_core::{malloc_async, CudaStream};
use cuda_core::{DType, DTypeId};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, MaybeUninit};
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

impl<T: DType> Partition<Tensor<T>> {
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
    pub fn dtype(&self) -> DTypeId {
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

/// A multi-dimensional array stored in GPU memory.
///
/// `Tensor` is the primary type for working with GPU data in cuTile Rust. It wraps a
/// [`DeviceBuffer`] with shape and stride information, providing a typed, multi-dimensional
/// view of GPU memory.
///
/// ## Memory Management
///
/// Tensors share GPU memory ownership through `Arc<DeviceBuffer>`. Memory is automatically
/// freed when the last reference is dropped. For shared tensor ownership, use
/// `Arc<Tensor<T>>`, which enables zero-copy views over the same storage.
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
pub struct Tensor<T: DType> {
    pub(crate) storage: Arc<DeviceBuffer>,
    pub shape: Vec<i32>,
    pub strides: Vec<i32>,
    _dtype: PhantomData<T>,
}

// Computes row-major contiguous strides for a given shape.
fn contiguous_strides(shape: &[i32]) -> Vec<i32> {
    let mut stride = 1;
    let mut strides = Vec::with_capacity(shape.len());
    for dim in shape.iter().rev() {
        strides.push(stride);
        stride *= *dim;
    }
    strides.reverse();
    strides
}

// Multiplies shape dimensions with overflow checks to recover the logical element count.
fn checked_num_elements(shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| crate::error::tensor_error("Tensor shape overflowed usize."))
    })
}

// Computes the logical byte size for a typed shape while guarding against overflow.
fn checked_num_bytes<T>(shape: &[usize]) -> Result<usize, Error> {
    checked_num_elements(shape)?
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::error::tensor_error("Tensor byte size overflowed usize."))
}

// Variant of checked_num_elements for i32-backed metadata, rejecting negative dimensions.
fn checked_num_elements_i32(shape: &[i32]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, dim| {
        let dim = usize::try_from(*dim)
            .map_err(|_| crate::error::tensor_error("Tensor shape contains negative dimension."))?;
        acc.checked_mul(dim)
            .ok_or_else(|| crate::error::tensor_error("Tensor shape overflowed usize."))
    })
}

// Computes the logical byte size for i32-backed tensor metadata.
fn checked_num_bytes_i32<T>(shape: &[i32]) -> Result<usize, Error> {
    checked_num_elements_i32(shape)?
        .checked_mul(size_of::<T>())
        .ok_or_else(|| crate::error::tensor_error("Tensor byte size overflowed usize."))
}

impl<T: DType> Tensor<T> {
    // Enforces the core tensor invariant: shape/stride ranks must agree and the logical
    // typed byte size must exactly match the backing storage byte length.
    fn assert_valid_metadata(shape: &[i32], strides: &[i32], storage_num_bytes: usize) {
        assert_eq!(
            shape.len(),
            strides.len(),
            "Tensor shape/stride rank mismatch."
        );

        let num_bytes = checked_num_bytes_i32::<T>(shape)
            .expect("Tensor shape contains invalid dimensions or overflows.");
        assert_eq!(
            num_bytes, storage_num_bytes,
            "Tensor logical byte size must match storage byte size."
        );
    }

    /// Wraps an owned byte allocation as a tensor after validating that the supplied
    /// shape/stride metadata is consistent with the allocation size.
    pub(crate) fn from_device_buffer(
        device_buffer: DeviceBuffer,
        shape: Vec<i32>,
        strides: Vec<i32>,
    ) -> Self {
        Self::assert_valid_metadata(&shape, &strides, device_buffer.len_bytes());
        Self {
            storage: Arc::new(device_buffer),
            shape,
            strides,
            _dtype: PhantomData,
        }
    }

    /// Rebuilds a tensor from raw device allocation parts and validates the metadata
    /// against the provided byte length before taking ownership of the pointer.
    pub(crate) unsafe fn from_raw_parts(
        dptr: CUdeviceptr,
        len_bytes: usize,
        device_id: usize,
        shape: Vec<i32>,
        strides: Vec<i32>,
    ) -> Self {
        Self::assert_valid_metadata(&shape, &strides, len_bytes);
        Self::from_device_buffer(
            DeviceBuffer::from_raw_parts(dptr, len_bytes, device_id),
            shape,
            strides,
        )
    }

    // Returns the physical byte length of the shared backing allocation.
    fn storage_num_bytes(&self) -> usize {
        self.storage.len_bytes()
    }

    // Returns the logical element count described by the tensor's shape metadata.
    fn num_elements(&self) -> usize {
        checked_num_elements_i32(&self.shape)
            .expect("Tensor shape contains invalid dimensions or overflows.")
    }

    // Returns the byte size implied by shape metadata and dtype T.
    fn typed_num_bytes(&self) -> usize {
        checked_num_bytes_i32::<T>(&self.shape)
            .expect("Tensor shape contains invalid dimensions or overflows.")
    }

    // Validates that a zero-copy view keeps the same logical byte size and starts from
    // a layout that this implementation can safely reinterpret as contiguous.
    fn validate_view_shape(&self, shape: &[usize]) -> Result<(), Error> {
        if !self.is_contiguous() {
            return tensor_error_result("Zero-copy tensor views require contiguous storage.");
        }
        let target_num_bytes = checked_num_bytes::<T>(shape)?;
        if target_num_bytes != self.typed_num_bytes() {
            return tensor_error_result("View shape must preserve tensor size.");
        }
        Ok(())
    }

    // Validates zero-copy reinterpret by checking total byte size and target-type
    // alignment on top of the same contiguous-layout requirement as views.
    fn validate_reinterpret_shape<U: DType>(&self, shape: &[usize]) -> Result<(), Error> {
        if !self.is_contiguous() {
            return tensor_error_result("Zero-copy reinterpret requires contiguous storage.");
        }
        let target_num_bytes = checked_num_bytes::<U>(shape)?;
        if target_num_bytes != self.typed_num_bytes() {
            return tensor_error_result("Reinterpret shape must preserve total byte size.");
        }
        let alignment = align_of::<U>() as u64;
        if alignment > 1 && self.cu_deviceptr() % alignment != 0 {
            return tensor_error_result(
                "Tensor storage alignment is incompatible with reinterpret target type.",
            );
        }
        Ok(())
    }

    // Mutable partitioning is only sound when no other tensor/view aliases the backing storage.
    fn assert_unique_storage(&self) {
        assert!(
            Arc::strong_count(&self.storage) == 1,
            "Cannot create mutable partition from shared tensor storage."
        );
    }

    /// Allocates uninitialized GPU memory for a 1D tensor.
    ///
    /// This is a low-level function that allocates memory asynchronously but does not
    /// initialize it. The returned value must be initialized before use with `assume_init()`.
    ///
    /// ## Safety
    ///
    /// The returned tensor is wrapped in `MaybeUninit`. It must be initialized by a kernel
    /// or other operation before calling `assume_init()` on it.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// use cutile::tensor::Tensor;
    ///
    /// let uninit = Tensor::<f32>::uninitialized(1024).await;
    /// // Must initialize before use
    /// let tensor = unsafe { uninit.assume_init() };
    /// ```
    pub fn uninitialized(len: usize) -> impl DeviceOperation<Output = MaybeUninit<Self>> {
        assert!(len > 0, "Non-zero length required.");
        device_operation::with_context(move |ctx| {
            let num_bytes = len * size_of::<T>();
            value(MaybeUninit::new(unsafe {
                Self::from_raw_parts(
                    malloc_async(num_bytes, ctx.get_cuda_stream()),
                    num_bytes,
                    ctx.get_device_id(),
                    vec![len as i32],
                    vec![1],
                )
            }))
        })
    }

    pub fn dtype(&self) -> DTypeId {
        T::DTYPE
    }

    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.storage.cu_deviceptr()
    }

    /// Returns a typed device pointer.
    pub fn device_pointer(&self) -> DevicePointer<T> {
        unsafe { DevicePointer::from_cu_deviceptr(self.cu_deviceptr()) }
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        debug_assert_eq!(self.typed_num_bytes(), self.storage_num_bytes());
        self.num_elements()
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
        let n = self.typed_num_bytes();
        debug_assert_eq!(n, self.storage_num_bytes());
        n
    }

    /// Returns the size of the tensor in megabytes (base 10).
    pub fn num_mb(self: &Arc<Self>) -> usize {
        self.num_bytes() / 10usize.pow(6)
    }

    /// Returns the size of the tensor in gigabytes (base 10).
    pub fn num_gb(self: &Arc<Self>) -> usize {
        self.num_bytes() / 10usize.pow(9)
    }

    /// Returns `true` if the tensor metadata describes a contiguous row-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
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
    pub fn reshape<const RANK: usize>(mut self, shape: [usize; RANK]) -> Self {
        // Make sure it's a valid shape for this tensor.
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
        assert_eq!(
            shape.iter().product::<i32>(),
            self.shape.iter().product::<i32>()
        );
        self.shape = shape.to_vec();
        self.strides = contiguous_strides(&shape);
        self
    }

    pub fn reshape_dyn(mut self, shape: &[usize]) -> Self {
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
        assert_eq!(
            shape.iter().product::<i32>(),
            self.shape.iter().product::<i32>()
        );
        self.shape = shape.to_vec();
        self.strides = contiguous_strides(&shape);
        self
    }

    /// Flattens an owned tensor into a rank-1 tensor without copying data.
    pub fn flatten(self) -> Self {
        let size = self.size();
        self.reshape([size])
    }

    /// Creates a zero-copy Arc-backed view with a new static shape.
    pub fn try_view<const RANK: usize>(
        self: &Arc<Self>,
        shape: [usize; RANK],
    ) -> Result<Arc<Self>, Error> {
        self.try_view_dyn(&shape)
    }

    /// Creates a zero-copy Arc-backed view with a new runtime shape.
    pub fn try_view_dyn(self: &Arc<Self>, shape: &[usize]) -> Result<Arc<Self>, Error> {
        self.validate_view_shape(shape)?;
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
        Ok(Arc::new(Self {
            storage: self.storage.clone(),
            strides: contiguous_strides(&shape),
            shape,
            _dtype: PhantomData,
        }))
    }

    /// Flattens an Arc-backed tensor into a rank-1 view without copying data.
    pub fn try_flatten_view(self: &Arc<Self>) -> Result<Arc<Self>, Error> {
        self.try_view([self.size()])
    }

    /// Creates a zero-copy Arc-backed view with a new static shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    ///
    /// let x = Arc::new(api::arange::<f32>(8).await);
    /// let y = x.view([2, 4]);
    /// ```
    ///
    /// ## Errors
    ///
    /// This convenience API panics on failure instead of returning an error.
    /// Use [`Tensor::try_view`] to handle failures as `Result`.
    ///
    /// ## Panics
    ///
    /// Panics if the tensor is not contiguous or if the target shape does not preserve
    /// the logical element count.
    pub fn view<const RANK: usize>(self: &Arc<Self>, shape: [usize; RANK]) -> Arc<Self> {
        self.try_view(shape)
            .expect("Failed to create zero-copy tensor view.")
    }

    /// Creates a zero-copy Arc-backed view with a new runtime shape.
    pub fn view_dyn(self: &Arc<Self>, shape: &[usize]) -> Arc<Self> {
        self.try_view_dyn(shape)
            .expect("Failed to create zero-copy tensor view.")
    }

    /// Flattens an Arc-backed tensor into a rank-1 view without copying data.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x = Arc::new(api::arange::<f32>(8).await).view([2, 4]);
    /// let y = x.flatten_view();
    /// ```
    ///
    /// ## Errors
    ///
    /// This convenience API panics on failure instead of returning an error.
    /// Use [`Tensor::try_flatten_view`] to handle failures as `Result`.
    ///
    /// ## Panics
    ///
    /// Panics if the tensor is not contiguous.
    pub fn flatten_view(self: &Arc<Self>) -> Arc<Self> {
        self.try_flatten_view()
            .expect("Failed to create zero-copy tensor view.")
    }

    /// Creates a zero-copy reinterpret view with a new static shape.
    pub fn try_reinterpret<U: DType, const RANK: usize>(
        self: &Arc<Self>,
        shape: [usize; RANK],
    ) -> Result<Arc<Tensor<U>>, Error> {
        self.try_reinterpret_dyn(&shape)
    }

    /// Creates a zero-copy reinterpret view with a new runtime shape.
    pub fn try_reinterpret_dyn<U: DType>(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<Arc<Tensor<U>>, Error> {
        self.validate_reinterpret_shape::<U>(shape)?;
        let shape = shape.iter().map(|x| *x as i32).collect::<Vec<_>>();
        Ok(Arc::new(Tensor::<U> {
            storage: self.storage.clone(),
            strides: contiguous_strides(&shape),
            shape,
            _dtype: PhantomData,
        }))
    }

    /// Creates a zero-copy reinterpret view with a new static shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let bits: Arc<Vec<u32>> = Arc::new(vec![0x3f800000, 0x40000000]);
    /// let base = Arc::new(bits.copy_to_device_tensor().await);
    /// let floats = base.reinterpret::<f32, 1>([2]);
    /// ```
    ///
    /// ## Errors
    ///
    /// This convenience API panics on failure instead of returning an error.
    /// Use [`Tensor::try_reinterpret`] to handle failures as `Result`.
    ///
    /// ## Panics
    ///
    /// Panics if the tensor is not contiguous, if the target shape does not preserve
    /// total byte size, or if pointer alignment is incompatible with the target type.
    pub fn reinterpret<U: DType, const RANK: usize>(
        self: &Arc<Self>,
        shape: [usize; RANK],
    ) -> Arc<Tensor<U>> {
        self.try_reinterpret(shape)
            .expect("Failed to reinterpret tensor storage.")
    }

    /// Creates a zero-copy reinterpret view with a new runtime shape.
    pub fn reinterpret_dyn<U: DType>(self: &Arc<Self>, shape: &[usize]) -> Arc<Tensor<U>> {
        self.try_reinterpret_dyn(shape)
            .expect("Failed to reinterpret tensor storage.")
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

impl<T: DType> ToHostVec<T> for Tensor<T> {
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>> {
        let arc_self = Arc::new(self);
        copy_device_to_host_vec(&arc_self)
    }
}

impl<T: DType> ToHostVec<T> for Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>> {
        copy_device_to_host_vec(&self)
    }
}

impl<T: DType> ToHostVec<T> for &Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOperation<Output = Vec<T>> {
        copy_device_to_host_vec(self)
    }
}

impl<T: DType> IntoPartitionArc for Tensor<T> {
    fn partition<const RANK: usize>(
        self: Arc<Tensor<T>>,
        partition_shape: [i32; RANK],
    ) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides = self.strides.clone();
        let tensor = Arc::try_unwrap(self).expect("Failed to convert Arc to Partition.");
        tensor.assert_unique_storage();
        Partition::<Tensor<T>> {
            object: tensor,
            partition_shape,
            partition_strides,
        }
    }
}

impl<T: DType> IntoPartition for Tensor<T> {
    fn partition<const RANK: usize>(self, partition_shape: [i32; RANK]) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides = self.strides.clone();
        self.assert_unique_storage();
        Partition::<Tensor<T>> {
            object: self,
            partition_shape,
            partition_strides,
        }
    }
}

pub trait Unpartition<T: DType> {
    /// Unwraps the partition to produce the underlying value.
    fn unpartition(self) -> impl DeviceOperation<Output = Tensor<T>>;
}

impl<T: DType, DI: DeviceOperation<Output = Partition<Tensor<T>>>> Unpartition<T> for DI {
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

impl<T: DType> DeviceVec<Tensor<T>> {
    pub fn from(v: Vec<Tensor<T>>) -> DeviceVec<Tensor<T>> {
        let i64vec: Arc<Vec<i64>> = v
            .iter()
            .map(|x| x.cu_deviceptr() as i64)
            .collect::<Vec<_>>()
            .into();
        let device_vec: Arc<Tensor<i64>> = copy_host_vec_to_device(&i64vec)
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

impl<T: DType> From<Vec<Tensor<T>>> for DeviceVec<Tensor<T>> {
    fn from(v: Vec<Tensor<T>>) -> Self {
        DeviceVec::from(v)
    }
}

impl<T: DType> Index<usize> for DeviceVec<Tensor<T>> {
    type Output = Arc<Tensor<T>>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.host_vec[index]
    }
}

pub struct DeviceVecIntoIter<Item> {
    items: DeviceVec<Item>,
}

impl<T: DType> Iterator for DeviceVecIntoIter<Tensor<T>> {
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

impl<T: DType> IntoIterator for DeviceVec<Tensor<T>> {
    type Item = Tensor<T>;
    type IntoIter = DeviceVecIntoIter<Tensor<T>>;
    fn into_iter(self) -> Self::IntoIter {
        DeviceVecIntoIter { items: self }
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;
    use crate::api;
    use cuda_async::device_operation::DeviceOperation;
    use std::mem::forget;
    use std::sync::Arc;

    #[test]
    fn reinterpret_rejects_misaligned_storage() {
        let base = Arc::new(api::zeros::<1, u8>([8]).sync().expect("Failed."));
        // Test: reinterpret must reject storage whose base pointer is not aligned
        // for the target dtype, even when the byte count itself would otherwise fit.
        // Shift the pointer by one byte so the storage is no longer aligned for u32.
        let misaligned = Arc::new(unsafe {
            Tensor::<u8>::from_raw_parts(
                base.cu_deviceptr() + 1,
                4,
                base.storage.device_id(),
                vec![4],
                vec![1],
            )
        });

        assert!(misaligned.try_reinterpret::<u32, 1>([1]).is_err());

        // The misaligned tensor is a borrowed view onto `base`'s allocation and must not free it.
        forget(misaligned);
    }

    #[test]
    #[should_panic(expected = "Tensor logical byte size must match storage byte size.")]
    fn from_raw_parts_rejects_shape_storage_mismatch() {
        let base = Arc::new(api::zeros::<1, u8>([4]).sync().expect("Failed."));

        // Test: raw tensor construction must preserve the invariant that logical
        // tensor bytes derived from shape/dtype exactly match the backing storage size.
        // Four bytes of storage cannot describe a Tensor<u32> with shape [2], which would
        // logically require eight bytes.
        let _ = unsafe {
            Tensor::<u32>::from_raw_parts(
                base.cu_deviceptr(),
                4,
                base.storage.device_id(),
                vec![2],
                vec![1],
            )
        };
    }
}
