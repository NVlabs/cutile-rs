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
//! let tensor = api::zeros(&[256]).await;
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
//! let tensor = api::ones(&[1024]).await;
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
//! let tensor = api::zeros::<f32>(&[1024]).await;
//!
//! // Access properties
//! println!("Shape: {:?}", tensor.shape());
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
//! let tensor = api::zeros(&[256]).await;
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
//! let gpu_tensor = api::ones(&[1024]).await;
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
//! let tensor = Arc::new(api::zeros(&[256]).await);
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
//! let z = api::zeros(&[256]).partition([64]);
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

use crate::api::{copy_device_to_host_vec, copy_host_vec_to_device};
use crate::error::{tensor_error_result, Error};
use crate::tile_kernel::UnwrapPartition;
use anyhow::Result;
use cuda_async::device_buffer::{DeviceBuffer, DevicePointer};
use cuda_async::device_operation;
use cuda_async::device_operation::{value, DeviceOp, IntoDeviceOp, Value};
use cuda_core::malloc_async;
use cuda_core::sys::CUdeviceptr;
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
/// let tensor = api::ones(&[256]).await;
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
/// let x = api::zeros(&[128, 128]).partition([32, 32]);
/// assert_eq!(x.grid(), (4, 4, 1)); // 128/32 = 4 in each dimension
/// ```
pub struct Partition<T> {
    pub(crate) object: T,
    pub partition_shape: Vec<usize>,
    pub partition_strides: Vec<usize>,
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

    /// Returns the data type name as a string.
    pub fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }

    /// Calculates the CUDA launch grid dimensions based on the partition.
    ///
    /// The grid is computed as `tensor_shape / partition_shape` for each dimension.
    /// Supports 1D, 2D, and 3D tensors.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x = api::zeros(&[256]).partition([64]);
    /// assert_eq!(x.grid(), (4, 1, 1));
    ///
    /// let y = api::zeros(&[128, 256]).partition([32, 64]);
    /// assert_eq!(y.grid(), (4, 4, 1));
    /// ```
    ///
    /// ## Panics
    ///
    /// Panics if the tensor rank is greater than 3.
    pub fn grid(&self) -> Result<(u32, u32, u32), Error> {
        if !self.object.shape.iter().all(|&x| x > 0) {
            return tensor_error_result("Shape dimensions must be positive.");
        }
        let shape: Vec<u32> = self.object.shape.iter().map(|&x| x as u32).collect();
        let partition_shape: Vec<u32> = self.partition_shape.iter().map(|&x| x as u32).collect();
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
    /// let tensor = api::zeros(&[1024]).await;
    /// let partitioned = tensor.partition([128]); // 8 partitions
    /// ```
    fn partition<const RANK: usize>(self, partition_shape: [usize; RANK]) -> Partition<Self>
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
    /// let tensor = Arc::new(api::zeros(&[1024]).await);
    /// let partitioned = tensor.partition([128]);
    /// ```
    fn partition<const RANK: usize>(
        self: Arc<Self>,
        partition_shape: [usize; RANK],
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
/// let x = api::zeros::<f32>(&[1024]).await;
/// let y = api::ones::<f32>(&[512, 512]).await;
/// let z = api::arange::<i32>(256).await;
/// ```
///
/// ### Copying and reshaping
///
/// ```rust,ignore
/// let x: Tensor<f32> = api::zeros(&[1024]).await;
///
/// // Duplicate to create a new tensor with the same data
/// let y: Tensor<f32> = x.dup().await;
///
/// // Reshape (must preserve total size)
/// let reshaped = y.reshape(&[32, 32]); // 1024 = 32 * 32
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
pub use cutile_compiler::specialization::{compute_spec, SpecializationBits};

#[derive(Debug)]
pub struct Tensor<T: DType> {
    pub(crate) storage: Arc<DeviceBuffer>,
    pub(crate) shape: Vec<i32>,
    pub(crate) strides: Vec<i32>,
    pub(crate) spec: SpecializationBits,
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
        let storage = Arc::new(device_buffer);
        let spec = compute_spec(
            storage.cu_deviceptr(),
            &shape,
            &strides,
            size_of::<T>() as i32,
        );
        Self {
            storage,
            shape,
            strides,
            spec,
            _dtype: PhantomData,
        }
    }

    /// Rebuilds a tensor from raw device allocation parts and validates the metadata
    /// against the provided byte length before taking ownership of the pointer.
    pub unsafe fn from_raw_parts(
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
    pub fn uninitialized(len: usize) -> impl DeviceOp<Output = MaybeUninit<Self>> {
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

    pub(crate) fn cu_deviceptr(&self) -> CUdeviceptr {
        self.storage.cu_deviceptr()
    }

    pub fn device_id(&self) -> usize {
        self.storage.device_id()
    }

    /// Returns a typed device pointer.
    pub fn device_pointer(&self) -> DevicePointer<T> {
        unsafe { DevicePointer::from_cu_deviceptr(self.cu_deviceptr()) }
    }

    /// Returns the tensor's shape.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Returns the tensor's strides.
    pub fn strides(&self) -> &[i32] {
        &self.strides
    }

    /// Returns the tensor's specialization bits.
    pub fn spec(&self) -> &SpecializationBits {
        &self.spec
    }

    /// Returns the total number of elements in the tensor.
    pub fn size(&self) -> usize {
        debug_assert_eq!(self.typed_num_bytes(), self.storage_num_bytes());
        self.num_elements()
    }

    /// Creates an independent copy of this tensor's GPU data.
    ///
    /// Returns a device operation that, when executed, will allocate new GPU memory
    /// and copy the tensor's data.
    pub fn dup(&self) -> impl DeviceOp<Output = Self> {
        crate::api::dup(self)
    }

    /// Returns the total size of the tensor in bytes.
    pub fn num_bytes(&self) -> usize {
        self.typed_num_bytes()
    }

    /// Returns `true` if the tensor metadata describes a contiguous row-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Create an `Arc<Tensor<T>>` that shares this tensor's device memory.
    ///
    /// # Safety
    ///
    /// Two tensors sharing storage can cause mutable aliasing if both are
    /// passed to kernels. The caller must ensure only one is written at a
    /// time. Prefer `tensor.view()` (returns `TensorView`) for safe
    /// borrow-based sharing.
    pub unsafe fn into_shared_alias(&self) -> Arc<Self> {
        Arc::new(Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            spec: self.spec.clone(),
            _dtype: PhantomData,
        })
    }

    // Internal: reshape without validation. Caller must ensure element count matches.
    pub(crate) fn reshape_unchecked(mut self, shape: &[usize]) -> Self {
        let shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        self.strides = contiguous_strides(&shape);
        self.spec = compute_spec(
            self.storage.cu_deviceptr(),
            &shape,
            &self.strides,
            size_of::<T>() as i32,
        );
        self.shape = shape;
        self
    }

    // Internal: create a new Arc sharing storage with different shape.
    // Used by ReshapeOp and the Reshape trait impl for &Arc<Tensor<T>>.
    pub(crate) fn reshape_shared(self: &Arc<Self>, shape: &[usize]) -> Result<Arc<Self>, Error> {
        self.validate_view_shape(shape)?;
        let new_shape: Vec<i32> = shape.iter().map(|x| *x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.storage.cu_deviceptr(),
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(Arc::new(Self {
            storage: self.storage.clone(),
            strides: new_strides,
            shape: new_shape,
            spec,
            _dtype: PhantomData,
        }))
    }

    /// Reinterprets the tensor's bytes as a different type with a new shape.
    ///
    /// Zero-copy. Returns `Err` if the tensor is not contiguous, the total
    /// byte size doesn't match, or the pointer alignment is incompatible.
    pub fn reinterpret<U: DType>(
        self: &Arc<Self>,
        shape: &[usize],
    ) -> Result<Arc<Tensor<U>>, Error> {
        self.validate_reinterpret_shape::<U>(shape)?;
        let new_shape: Vec<i32> = shape.iter().map(|x| *x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.storage.cu_deviceptr(),
            &new_shape,
            &new_strides,
            size_of::<U>() as i32,
        );
        Ok(Arc::new(Tensor::<U> {
            storage: self.storage.clone(),
            strides: new_strides,
            shape: new_shape,
            spec,
            _dtype: PhantomData,
        }))
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
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>>;
}

impl<T: DType> ToHostVec<T> for Tensor<T> {
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>> {
        let arc_self = Arc::new(self);
        copy_device_to_host_vec(&arc_self)
    }
}

impl<T: DType> ToHostVec<T> for Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>> {
        copy_device_to_host_vec(&self)
    }
}

impl<T: DType> ToHostVec<T> for &Arc<Tensor<T>> {
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>> {
        copy_device_to_host_vec(self)
    }
}

// ── Reshape trait ────────────────────────────────────────────────────────────

/// Reshape a tensor or Arc<Tensor> to a new shape.
///
/// - On `Tensor<T>`: consumes and returns a reshaped `Tensor<T>`.
/// - On `&Arc<Tensor<T>>`: creates a new `Arc` sharing device memory.
pub trait Reshape {
    type Output;
    fn reshape(self, shape: &[usize]) -> Result<Self::Output, Error>;
}

impl<T: DType> Reshape for Tensor<T> {
    type Output = Tensor<T>;
    fn reshape(self, shape: &[usize]) -> Result<Tensor<T>, Error> {
        let current_elems: i32 = self.shape.iter().product();
        let new_elems: i32 = shape.iter().map(|&x| x as i32).product();
        if new_elems != current_elems {
            return tensor_error_result("reshape: new shape must preserve element count.");
        }
        Ok(self.reshape_unchecked(shape))
    }
}

impl<'a, T: DType> Reshape for &'a Arc<Tensor<T>> {
    type Output = Arc<Tensor<T>>;
    fn reshape(self, shape: &[usize]) -> Result<Arc<Tensor<T>>, Error> {
        self.reshape_shared(shape)
    }
}

// ── TensorView ──────────────────────────────────────────────────────────────

/// A borrowed, reshaped view of a tensor.
///
/// Created by [`Tensor::view`]. The view borrows the base tensor's device
/// memory with different shape/strides metadata. The borrow checker ensures
/// the base tensor can't be mutated while the view exists.
///
/// Kernel `&Tensor` params accept `&TensorView<T>` via `KernelInput`.
///
/// ```rust,ignore
/// let tensor = api::ones::<f32>(&[1024]).sync()?;
/// let view = tensor.view(&[32, 32])?;    // borrows tensor
/// kernel(out, &view).sync()?;             // view accepted as &Tensor param
/// // view dropped — tensor can be mutated again
/// ```
pub struct TensorView<'a, T: DType> {
    base: &'a Tensor<T>,
    offset_bytes: usize,
    shape: Vec<i32>,
    strides: Vec<i32>,
    spec: SpecializationBits,
}

impl<'a, T: DType> TensorView<'a, T> {
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub fn strides(&self) -> &[i32] {
        &self.strides
    }
    pub fn spec(&self) -> &SpecializationBits {
        &self.spec
    }
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&x| x as usize).product()
    }
    /// Re-view with a different shape.
    pub fn view(&self, shape: &[usize]) -> Result<TensorView<'_, T>, Error> {
        if self.strides != contiguous_strides(&self.shape) {
            return tensor_error_result("view: cannot reshape a non-contiguous view.");
        }
        let current_elems: i32 = self.shape.iter().product();
        let new_elems: i32 = shape.iter().map(|&x| x as i32).product();
        if new_elems != current_elems {
            return tensor_error_result("view: new shape must preserve element count.");
        }
        let new_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.base.storage.cu_deviceptr(),
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self.base,
            offset_bytes: self.offset_bytes,
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }
    /// Slice this view along one or more axes, numpy-style.
    ///
    /// Each range corresponds to an axis. Fewer ranges than axes is
    /// allowed: trailing axes are left unsliced. The strides are
    /// preserved (slicing never changes strides).
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<TensorView<'_, T>, Error> {
        if ranges.len() > self.shape.len() {
            return tensor_error_result("slice: more ranges than axes.");
        }
        let mut offset_elems: usize = 0;
        let mut new_shape = self.shape.clone();
        for (axis, range) in ranges.iter().enumerate() {
            let dim = self.shape[axis] as usize;
            if range.start > range.end || range.end > dim {
                return tensor_error_result("slice: range out of bounds.");
            }
            offset_elems += range.start * self.strides[axis] as usize;
            new_shape[axis] = (range.end - range.start) as i32;
        }
        let new_strides = self.strides.clone();
        let spec = compute_spec(
            self.base.storage.cu_deviceptr()
                + (self.offset_bytes + offset_elems * size_of::<T>()) as u64,
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self.base,
            offset_bytes: self.offset_bytes + offset_elems * size_of::<T>(),
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }
}

impl<T: DType> Tensor<T> {
    /// Create a borrowed view with a different shape.
    ///
    /// The view borrows `self` — the tensor can't be mutated while the
    /// view exists. No allocation or copy.
    pub fn view(&self, shape: &[usize]) -> Result<TensorView<'_, T>, Error> {
        let current_elems: i32 = self.shape.iter().product();
        let new_elems: i32 = shape.iter().map(|&x| x as i32).product();
        if new_elems != current_elems {
            return tensor_error_result("view: new shape must preserve element count.");
        }
        let new_shape: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        let new_strides = contiguous_strides(&new_shape);
        let spec = compute_spec(
            self.storage.cu_deviceptr(),
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self,
            offset_bytes: 0,
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }

    /// Slice this tensor along one or more axes, numpy-style.
    ///
    /// Returns a `TensorView` with adjusted offset and shape. Strides
    /// are preserved from the original tensor. No copy.
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<TensorView<'_, T>, Error> {
        if ranges.len() > self.shape.len() {
            return tensor_error_result("slice: more ranges than axes.");
        }
        let mut offset_elems: usize = 0;
        let mut new_shape = self.shape.clone();
        for (axis, range) in ranges.iter().enumerate() {
            let dim = self.shape[axis] as usize;
            if range.start > range.end || range.end > dim {
                return tensor_error_result("slice: range out of bounds.");
            }
            offset_elems += range.start * self.strides[axis] as usize;
            new_shape[axis] = (range.end - range.start) as i32;
        }
        let new_strides = self.strides.clone();
        let spec = compute_spec(
            self.storage.cu_deviceptr() + (offset_elems * size_of::<T>()) as u64,
            &new_shape,
            &new_strides,
            size_of::<T>() as i32,
        );
        Ok(TensorView {
            base: self,
            offset_bytes: offset_elems * size_of::<T>(),
            shape: new_shape,
            strides: new_strides,
            spec,
        })
    }
}

impl<T: DType> IntoPartitionArc for Tensor<T> {
    fn partition<const RANK: usize>(
        self: Arc<Tensor<T>>,
        partition_shape: [usize; RANK],
    ) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides: Vec<usize> = self.strides.iter().map(|&s| s as usize).collect();
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
    fn partition<const RANK: usize>(self, partition_shape: [usize; RANK]) -> Partition<Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides: Vec<usize> = self.strides.iter().map(|&s| s as usize).collect();
        self.assert_unique_storage();
        Partition::<Tensor<T>> {
            object: self,
            partition_shape,
            partition_strides,
        }
    }
}

// ── Partition<&'a mut Tensor<T>> ─────────────────────────────────────────────

/// Partition a mutably borrowed tensor. The partition borrows the tensor,
/// so no `unpartition()` is needed — the tensor already has the kernel's output.
pub trait PartitionMut<'a, T: DType> {
    fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Partition<&'a mut Tensor<T>>;
}

impl<'a, T: DType> PartitionMut<'a, T> for &'a mut Tensor<T> {
    fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Partition<&'a mut Tensor<T>> {
        let partition_shape = partition_shape.to_vec();
        let partition_strides: Vec<usize> = self.strides.iter().map(|&s| s as usize).collect();
        Partition {
            object: self,
            partition_shape,
            partition_strides,
        }
    }
}

impl<'a, T: DType> Partition<&'a mut Tensor<T>> {
    pub fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }

    pub fn grid(&self) -> Result<(u32, u32, u32), Error> {
        if !self.object.shape.iter().all(|&x| x > 0) {
            return tensor_error_result("Shape dimensions must be positive.");
        }
        let shape: Vec<u32> = self.object.shape.iter().map(|&x| x as u32).collect();
        let partition_shape: Vec<u32> = self.partition_shape.iter().map(|&x| x as u32).collect();
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

impl<'a, T: DType + Sync> IntoDeviceOp<Partition<&'a mut Tensor<T>>>
    for Partition<&'a mut Tensor<T>>
{
    type Op = Value<Partition<&'a mut Tensor<T>>>;
    fn into_op(self) -> Value<Partition<&'a mut Tensor<T>>> {
        value(self)
    }
}

/// Extension trait for partitioning an `Arc<Tensor<T>>` by consuming sole ownership.
pub trait TryPartition<T: DType> {
    /// Consumes the Arc and partitions the tensor.
    ///
    /// Returns `Err` if the Arc has other owners (refcount > 1) or the
    /// underlying storage is shared with views.
    fn try_partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Result<Partition<Tensor<T>>, Error>;
}

impl<T: DType> TryPartition<T> for Arc<Tensor<T>> {
    fn try_partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> Result<Partition<Tensor<T>>, Error> {
        let tensor = Arc::try_unwrap(self).map_err(|_| {
            crate::error::tensor_error("try_partition: Arc<Tensor> has multiple owners")
        })?;
        Ok(tensor.partition(partition_shape))
    }
}

pub trait Unpartition<T: DType> {
    /// Unwraps the partition to produce the underlying value.
    fn unpartition(self) -> impl DeviceOp<Output = Tensor<T>>;
}

impl<T: DType, DI: DeviceOp<Output = Partition<Tensor<T>>>> Unpartition<T> for DI {
    fn unpartition(self) -> impl DeviceOp<Output = Tensor<T>> {
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
            .reshape_unchecked(&[v.len()])
            .into();
        let host_vec: Vec<Arc<Tensor<T>>> = v.into_iter().map(Arc::new).collect::<Vec<_>>();
        DeviceVec {
            _ty: PhantomData,
            host_vec,
            device_vec,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.host_vec.len() == 0
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
        if !self.items.is_empty() {
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

// IntoDeviceOp impls for Tensor types

impl<T: DType> IntoDeviceOp<Partition<Tensor<T>>> for Partition<Tensor<T>> {
    type Op = Value<Partition<Tensor<T>>>;
    fn into_op(self) -> Value<Partition<Tensor<T>>> {
        value(self)
    }
}

impl<T: DType> IntoDeviceOp<Tensor<T>> for Tensor<T> {
    type Op = Value<Tensor<T>>;
    fn into_op(self) -> Value<Tensor<T>> {
        value(self)
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<&'a Tensor<T>> for &'a Tensor<T> {
    type Op = Value<&'a Tensor<T>>;
    fn into_op(self) -> Value<&'a Tensor<T>> {
        value(self)
    }
}

// KernelInput impls — how &Tensor kernel params are held and recovered.

use cuda_async::launch::AsyncKernelLaunch;

// ── KernelOutput trait ──────────────────────────────────────────────────────
//
// Abstracts over Partition<Tensor<T>> and Partition<&mut Tensor<T>> so the
// macro-generated launcher accepts both for &mut Tensor params.

/// How a `&mut Tensor` kernel param is stored during execution and recovered.
///
/// | Input | Stored | Returned |
/// |---|---|---|
/// | `Partition<Tensor<T>>` | `Partition<Tensor<T>>` | `Partition<Tensor<T>>` |
/// | `Partition<&'a mut Tensor<T>>` | `Partition<&'a mut Tensor<T>>` | `Partition<&'a mut Tensor<T>>` |
pub trait KernelOutputStored<T: DType>: Send {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch);
    fn grid(&self) -> Result<(u32, u32, u32), Error>;
    fn dtype_str(&self) -> &'static str;
    fn partition_shape_as_i32(&self) -> Vec<i32>;
    fn strides_hint(&self) -> Vec<i32>;
    fn spec(&self) -> &SpecializationBits;
    fn shape_as_i32(&self) -> Vec<i32>;
}

pub trait KernelOutput<T: DType>: Send + Sized {
    type Stored: KernelOutputStored<T>;
    type Returned: Send;
    fn prepare(self) -> Self::Stored;
    fn recover(stored: Self::Stored) -> Self::Returned;
}

impl<T: DType> KernelOutputStored<T> for Partition<Tensor<T>> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.object.cu_deviceptr());
        }
        for dim in self.object.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(*stride);
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(*dim as i32);
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(*stride as i32);
        }
    }
    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        let shape: Vec<u32> = self.shape_as_i32().iter().map(|&x| x as u32).collect();
        let pshape: Vec<u32> = self
            .partition_shape_as_i32()
            .iter()
            .map(|&x| x as u32)
            .collect();
        match shape.len() {
            1 => Ok((u32::div_ceil(shape[0], pshape[0]), 1, 1)),
            2 => Ok((
                u32::div_ceil(shape[0], pshape[0]),
                u32::div_ceil(shape[1], pshape[1]),
                1,
            )),
            3 => Ok((
                u32::div_ceil(shape[0], pshape[0]),
                u32::div_ceil(shape[1], pshape[1]),
                u32::div_ceil(shape[2], pshape[2]),
            )),
            _ => tensor_error_result("Mutable tensor must be at most rank 3."),
        }
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition_shape.iter().map(|&x| x as i32).collect()
    }
    fn strides_hint(&self) -> Vec<i32> {
        self.object
            .spec
            .stride_one
            .iter()
            .map(|&is_one| if is_one { 1 } else { -1 })
            .collect()
    }
    fn spec(&self) -> &SpecializationBits {
        &self.object.spec
    }
    fn shape_as_i32(&self) -> Vec<i32> {
        self.object.shape.clone()
    }
}

impl<'a, T: DType> KernelOutputStored<T> for Partition<&'a mut Tensor<T>> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.object.cu_deviceptr());
        }
        for dim in self.object.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(*stride);
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(*dim as i32);
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(*stride as i32);
        }
    }
    fn grid(&self) -> Result<(u32, u32, u32), Error> {
        let shape: Vec<u32> = self.shape_as_i32().iter().map(|&x| x as u32).collect();
        let pshape: Vec<u32> = self
            .partition_shape_as_i32()
            .iter()
            .map(|&x| x as u32)
            .collect();
        match shape.len() {
            1 => Ok((u32::div_ceil(shape[0], pshape[0]), 1, 1)),
            2 => Ok((
                u32::div_ceil(shape[0], pshape[0]),
                u32::div_ceil(shape[1], pshape[1]),
                1,
            )),
            3 => Ok((
                u32::div_ceil(shape[0], pshape[0]),
                u32::div_ceil(shape[1], pshape[1]),
                u32::div_ceil(shape[2], pshape[2]),
            )),
            _ => tensor_error_result("Mutable tensor must be at most rank 3."),
        }
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
    fn partition_shape_as_i32(&self) -> Vec<i32> {
        self.partition_shape.iter().map(|&x| x as i32).collect()
    }
    fn strides_hint(&self) -> Vec<i32> {
        self.object
            .spec
            .stride_one
            .iter()
            .map(|&is_one| if is_one { 1 } else { -1 })
            .collect()
    }
    fn spec(&self) -> &SpecializationBits {
        &self.object.spec
    }
    fn shape_as_i32(&self) -> Vec<i32> {
        self.object.shape.clone()
    }
}

impl<T: DType> KernelOutput<T> for Partition<Tensor<T>> {
    type Stored = Partition<Tensor<T>>;
    type Returned = Partition<Tensor<T>>;
    fn prepare(self) -> Self::Stored {
        self
    }
    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<'a, T: DType> KernelOutput<T> for Partition<&'a mut Tensor<T>> {
    type Stored = Partition<&'a mut Tensor<T>>;
    type Returned = Partition<&'a mut Tensor<T>>;
    fn prepare(self) -> Self::Stored {
        self
    }
    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

// ── KernelInput traits ──────────────────────────────────────────────────────
//
// Defined here (not in cuda-async) so that impls on Arc<Tensor<T>> satisfy
// the orphan rule — both trait and Tensor<T> are in the same crate.

/// How a stored kernel input pushes its arguments to the launcher.
///
/// Implemented for `Arc<Tensor<T>>` and `&Tensor<T>`. Both push the same
/// data: device pointer, shape, and strides.
pub trait KernelInputStored: Send {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch);
    fn shape(&self) -> &[i32];
    fn strides(&self) -> &[i32];
    fn spec(&self) -> &SpecializationBits;
    fn dtype_str(&self) -> &'static str;
}

/// Converts a user-provided kernel input into a stored form for execution,
/// and recovers the caller's original type afterward.
///
/// | Input | Stored | Returned | `'static`? |
/// |---|---|---|---|
/// | `Tensor<T>` | `Arc<Tensor<T>>` | `Tensor<T>` | Yes |
/// | `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | Yes |
/// | `&'a Tensor<T>` | `&'a Tensor<T>` | `&'a Tensor<T>` | No |
pub trait KernelInput<T: DType>: Send + Sized {
    type Stored: KernelInputStored;
    type Returned: Send;
    fn prepare(self) -> Self::Stored;
    fn recover(stored: Self::Stored) -> Self::Returned;
}

// ── KernelInputStored impls ─────────────────────────────────────────────────

impl<T: DType> KernelInputStored for Arc<Tensor<T>> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.cu_deviceptr());
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
    fn shape(&self) -> &[i32] {
        Tensor::shape(self)
    }
    fn strides(&self) -> &[i32] {
        Tensor::strides(self)
    }
    fn spec(&self) -> &SpecializationBits {
        Tensor::spec(self)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

impl<'a, T: DType + Sync> KernelInputStored for &'a Tensor<T> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.cu_deviceptr());
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
    fn shape(&self) -> &[i32] {
        Tensor::shape(self)
    }
    fn strides(&self) -> &[i32] {
        Tensor::strides(self)
    }
    fn spec(&self) -> &SpecializationBits {
        Tensor::spec(self)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

// ── KernelInput impls ───────────────────────────────────────────────────────

impl<T: DType> KernelInput<T> for Tensor<T> {
    type Stored = Arc<Tensor<T>>;
    type Returned = Tensor<T>;
    fn prepare(self) -> Arc<Tensor<T>> {
        Arc::new(self)
    }
    fn recover(stored: Arc<Tensor<T>>) -> Tensor<T> {
        Arc::try_unwrap(stored).expect("KernelInput::recover: Arc has multiple owners")
    }
}

impl<T: DType> KernelInput<T> for Arc<Tensor<T>> {
    type Stored = Arc<Tensor<T>>;
    type Returned = Arc<Tensor<T>>;
    fn prepare(self) -> Arc<Tensor<T>> {
        self
    }
    fn recover(stored: Arc<Tensor<T>>) -> Arc<Tensor<T>> {
        stored
    }
}

impl<'a, T: DType + Sync> KernelInput<T> for &'a Tensor<T> {
    type Stored = &'a Tensor<T>;
    type Returned = &'a Tensor<T>;
    fn prepare(self) -> &'a Tensor<T> {
        self
    }
    fn recover(stored: &'a Tensor<T>) -> &'a Tensor<T> {
        stored
    }
}

// ── TensorView KernelInput impls ────────────────────────────────────────────

impl<'a, T: DType + Sync> KernelInputStored for &'a TensorView<'a, T> {
    fn push_kernel_args(&self, launcher: &mut AsyncKernelLaunch) {
        // Push the already-offset device pointer. The offset is applied
        // host-side so the kernel sees the correct base address directly.
        unsafe {
            launcher.push_device_ptr(self.base.cu_deviceptr() + self.offset_bytes as u64);
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
    fn shape(&self) -> &[i32] {
        &self.shape
    }
    fn strides(&self) -> &[i32] {
        &self.strides
    }
    fn spec(&self) -> &SpecializationBits {
        TensorView::spec(self)
    }
    fn dtype_str(&self) -> &'static str {
        T::DTYPE.as_str()
    }
}

impl<'a, T: DType + Sync> KernelInput<T> for &'a TensorView<'a, T> {
    type Stored = &'a TensorView<'a, T>;
    type Returned = &'a TensorView<'a, T>;
    fn prepare(self) -> Self::Stored {
        self
    }
    fn recover(stored: Self::Stored) -> Self::Returned {
        stored
    }
}

impl<'a, T: DType + Sync> IntoDeviceOp<&'a TensorView<'a, T>> for &'a TensorView<'a, T> {
    type Op = Value<&'a TensorView<'a, T>>;
    fn into_op(self) -> Value<&'a TensorView<'a, T>> {
        value(self)
    }
}
