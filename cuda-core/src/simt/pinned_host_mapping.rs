/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Zero-copy host memory registration for CUDA transfers and Tile kernels.
//!
//! [`PinnedHostMapping`] registers pre-allocated host memory using `cuMemHostRegister`.
//! The memory is mapped directly into the CUDA device address space with zero copy,
//! and automatically unregistered on drop.

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use crate::error::DriverError;
use crate::simt::context::CudaContext;

/// If set, the host memory is mapped into the address space of all CUDA contexts,
/// not just the one that performed the registration.
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 0x01;

/// If set, the host memory is mapped directly into the CUDA device address space
/// and [`device_ptr`](PinnedHostMapping::device_ptr) can be used by kernels.
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;

/// If set, the host memory is treated as I/O memory (e.g. PCIe MMIO or BARs).
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 0x04;

/// If set, the host memory is mapped read-only from the GPU's perspective.
pub const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 0x08;

/// Owned registration token for page-locked host memory.
///
/// Backing host memory is registered via `cuMemHostRegister` and unregistered
/// via `cuMemHostUnregister` on drop.
pub struct PinnedHostMapping {
    host_ptr: NonNull<u8>,
    dev_ptr: cuda_bindings::CUdeviceptr,
    len_bytes: usize,
    device_id: usize,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for PinnedHostMapping {}
unsafe impl Sync for PinnedHostMapping {}

impl PinnedHostMapping {
    /// Registers an existing caller-owned host memory buffer for zero-copy device access
    /// using default flags (`CU_MEMHOSTREGISTER_DEVICEMAP`).
    ///
    /// # Safety
    ///
    /// - `host_ptr` must point to at least `len_bytes` of valid, allocated host memory.
    /// - Backing host memory must remain valid and must not be freed while this mapping is alive.
    pub unsafe fn new(
        ctx: &Arc<CudaContext>,
        host_ptr: NonNull<u8>,
        len_bytes: usize,
    ) -> Result<Self, DriverError> {
        unsafe { Self::register(ctx, host_ptr, len_bytes, CU_MEMHOSTREGISTER_DEVICEMAP) }
    }

    /// Registers an existing caller-owned host memory buffer for zero-copy device access
    /// with explicit registration flags.
    ///
    /// Note that `CU_MEMHOSTREGISTER_DEVICEMAP` is always set so that a device pointer
    /// can be queried for kernels.
    ///
    /// # Safety
    ///
    /// - `host_ptr` must point to at least `len_bytes` of valid, allocated host memory.
    /// - Backing host memory must remain valid and must not be freed while this mapping is alive.
    pub unsafe fn register(
        ctx: &Arc<CudaContext>,
        host_ptr: NonNull<u8>,
        len_bytes: usize,
        flags: u32,
    ) -> Result<Self, DriverError> {
        if len_bytes == 0 {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }

        let effective_flags = flags | CU_MEMHOSTREGISTER_DEVICEMAP;

        ctx.bind_to_thread()?;
        unsafe {
            crate::simt::memory::host_register(
                host_ptr.as_ptr().cast(),
                len_bytes,
                effective_flags,
            )?;
        }

        let dev_ptr = unsafe {
            match crate::simt::memory::host_get_device_pointer(host_ptr.as_ptr().cast(), 0) {
                Ok(ptr) => ptr,
                Err(e) => {
                    let _ = crate::simt::memory::host_unregister(host_ptr.as_ptr().cast());
                    return Err(e);
                }
            }
        };

        let device_id = ctx.device().ordinal() as usize;

        Ok(Self {
            host_ptr,
            dev_ptr,
            len_bytes,
            device_id,
            ctx: ctx.clone(),
        })
    }

    /// Returns the mapped device pointer.
    #[inline]
    pub fn device_ptr(&self) -> cuda_bindings::CUdeviceptr {
        self.dev_ptr
    }

    /// Number of bytes registered.
    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// Number of bytes registered (equivalent to [`len_bytes`](Self::len_bytes)).
    #[inline]
    pub fn len(&self) -> usize {
        self.len_bytes
    }

    /// Returns true if the mapping has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_bytes == 0
    }

    /// Returns the device ordinal.
    #[inline]
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Host pointer.
    #[inline]
    pub fn host_ptr(&self) -> NonNull<u8> {
        self.host_ptr
    }

    /// Returns the host pointer as a raw const pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.host_ptr.as_ptr()
    }

    /// Returns the host pointer as a raw mut pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.host_ptr.as_ptr()
    }

    /// Returns the registered memory as a host byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.host_ptr.as_ptr(), self.len_bytes) }
    }

    /// Returns the registered memory as a mutable host byte slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.host_ptr.as_ptr(), self.len_bytes) }
    }

    /// Associated CUDA context.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

impl Drop for PinnedHostMapping {
    fn drop(&mut self) {
        if self.len_bytes != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx.record_err(unsafe {
                crate::simt::memory::host_unregister(self.host_ptr.as_ptr().cast())
            });
        }
    }
}

impl fmt::Debug for PinnedHostMapping {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PinnedHostMapping")
            .field("host_ptr", &self.host_ptr)
            .field("dev_ptr", &self.dev_ptr)
            .field("len_bytes", &self.len_bytes)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl AsRef<[u8]> for PinnedHostMapping {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PinnedHostMapping {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Deref for PinnedHostMapping {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for PinnedHostMapping {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_rejects_zero_length() {
        let dummy = NonNull::dangling();
        // Zero-length must fail early without touching the driver.
        let ctx = Arc::new(unsafe { std::mem::zeroed::<CudaContext>() });
        let result = unsafe { PinnedHostMapping::register(&ctx, dummy, 0, 0) };
        assert!(result.is_err());
        std::mem::forget(ctx);
    }
}
