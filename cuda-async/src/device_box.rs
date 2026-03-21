/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owned and borrowed wrappers around CUDA device pointers.

use crate::device_context::with_deallocator_stream;
use crate::launch::{AsyncKernelLaunch, KernelArgument};
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
}

impl<T: Send + Sized> KernelArgument for DevicePointer<T> {
    /// Pushes this device pointer as a kernel launch argument.
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        launcher.push_arg(Box::new(self.cu_deviceptr()));
    }
}

/// An owning handle to a CUDA device memory allocation, freed asynchronously on drop.
#[derive(Debug)]
pub struct DeviceBox<T: Send + ?Sized> {
    _device_id: usize,
    dtype: PhantomData<T>,
    cudptr: CUdeviceptr,
    len: usize,
}

unsafe impl<T: Send + ?Sized> Send for DeviceBox<T> {}
unsafe impl<T: Send + ?Sized> Sync for DeviceBox<T> {}

impl<T: Send + ?Sized> Drop for DeviceBox<T> {
    fn drop(&mut self) {
        unsafe {
            // Safety: The CUDA driver is guaranteed to complete any queued async operations.
            with_deallocator_stream(self._device_id, |stream| {
                free_async(self.cudptr, stream);
            })
            .unwrap_or_else(|_| {
                panic!(
                    "Failed to free device pointer on device_id={}",
                    self._device_id
                )
            })
        }
    }
}

impl<DType: Send + Sized> DeviceBox<[DType]> {
    /// Constructs a `DeviceBox<[DType]>` from a raw device pointer, length, and device id.
    ///
    /// # Safety
    /// The caller must ensure `dptr` points to a valid device allocation of at least `len` elements.
    pub unsafe fn from_raw_parts(dptr: CUdeviceptr, len: usize, device_id: usize) -> Self {
        Self {
            dtype: PhantomData,
            cudptr: dptr,
            len,
            _device_id: device_id,
        }
    }
    /// Returns the number of elements in the device slice.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Returns the raw CUDA device pointer.
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.cudptr
    }
    /// Returns the device id this allocation belongs to.
    pub fn device_id(&self) -> usize {
        self._device_id
    }
    /// Returns a non-owning `DevicePointer` to the underlying allocation.
    pub fn device_pointer(&self) -> DevicePointer<DType> {
        DevicePointer::<DType> {
            dtype: PhantomData,
            dptr: self.cudptr,
        }
    }
}
