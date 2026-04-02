/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owned and borrowed wrappers around CUDA device pointers.

use crate::device_context::with_deallocator_stream;
use cuda_core::free_async;
use cuda_core::sys::CUdeviceptr;
use std::marker::PhantomData;

/// A non-owning, copyable handle to a typed device memory address.
#[derive(Debug, Copy, Clone)]
pub struct DevicePointer<T> {
    dtype: PhantomData<T>,
    pub dptr: CUdeviceptr,
}

unsafe impl<T> Send for DevicePointer<T> {}

impl<T> DevicePointer<T> {
    /// Returns the raw CUDA device pointer.
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.dptr
    }

    /// Constructs a typed device pointer from a raw CUDA device pointer.
    ///
    /// # Safety
    /// The caller must ensure `dptr` is valid for values of type `T`.
    pub unsafe fn from_cu_deviceptr(dptr: CUdeviceptr) -> Self {
        Self {
            dtype: PhantomData,
            dptr,
        }
    }
}

/// An owning, type-erased handle to a CUDA device memory allocation, freed asynchronously on drop.
///
/// `DeviceBox` manages a raw byte buffer on the GPU. It stores only the device pointer,
/// byte length, and device id — all type information lives in higher-level wrappers
/// such as `Tensor<T>`.
#[derive(Debug)]
pub struct DeviceBox {
    device_id: usize,
    cudptr: CUdeviceptr,
    len: usize,
}

unsafe impl Send for DeviceBox {}
unsafe impl Sync for DeviceBox {}

impl Drop for DeviceBox {
    fn drop(&mut self) {
        unsafe {
            // Safety: The CUDA driver is guaranteed to complete any queued async operations.
            with_deallocator_stream(self.device_id, |stream| {
                free_async(self.cudptr, stream);
            })
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to free device pointer on device_id={}",
                    self.device_id
                )
            })
        }
    }
}

impl DeviceBox {
    /// Constructs a `DeviceBox` from a raw device pointer, byte length, and device id.
    ///
    /// # Safety
    /// The caller must ensure `dptr` points to a valid device allocation of at least
    /// `len_bytes` bytes.
    pub unsafe fn from_raw_parts(dptr: CUdeviceptr, len_bytes: usize, device_id: usize) -> Self {
        Self {
            cudptr: dptr,
            len: len_bytes,
            device_id,
        }
    }
    /// Returns whether the allocation is empty (zero bytes).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Returns the byte length of the allocation.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Returns the raw CUDA device pointer.
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.cudptr
    }
    /// Returns the device id this allocation belongs to.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}
