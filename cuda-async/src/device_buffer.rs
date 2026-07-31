/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owned and borrowed wrappers around CUDA device pointers.

use crate::device_context::free_on_deallocator_stream;
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
/// `DeviceBuffer` manages a raw byte buffer on the GPU. It stores only the device pointer,
/// byte length, and device id — all type information lives in higher-level wrappers
/// such as `Tensor<T>`.
#[derive(Debug)]
pub struct DeviceBuffer {
    device_id: usize,
    cudptr: CUdeviceptr,
    len: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            // Safety: The CUDA driver is guaranteed to complete any queued async operations.
            //
            // Never panic here: buffers are dropped during unwinding (e.g. while
            // a panicking `_mut` callback is poisoning its device context), and
            // a panic inside `Drop` during unwinding aborts the process. On an
            // unrecoverable context error, leaking the allocation is the lesser
            // evil. Drops that happen inside a `with_*` callback are deferred
            // by `free_on_deallocator_stream` instead of panicking on re-entry.
            if let Err(e) = free_on_deallocator_stream(self.device_id, self.cudptr) {
                eprintln!(
                    "cuda-async: leaking device pointer on device_id={}: {e:?}",
                    self.device_id
                );
            }
        }
    }
}

impl DeviceBuffer {
    /// Constructs a `DeviceBuffer` from a raw device pointer, byte length, and device id.
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
        self.len_bytes() == 0
    }
    /// Returns the byte length of the allocation.
    pub fn len_bytes(&self) -> usize {
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
