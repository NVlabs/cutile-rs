/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA kernel launch builder with argument marshalling.

use crate::device_context::with_default_device_policy;
use crate::device_future::DeviceFuture;
use crate::device_operation::{DeviceOp, ExecutionContext};
use crate::error::DeviceError;
use anyhow::{Context, Result};
use cuda_core::sys::CUdeviceptr;
use cuda_core::{launch_kernel, DType, Function, LaunchConfig, Stream};
use std::ffi::c_void;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::sync::Arc;
use std::vec::Vec;

/// A builder for asynchronously launching a CUDA kernel on a stream.
///
/// Arguments are heap-allocated by [`push_arg`](Self::push_arg) /
/// [`push_device_ptr`](Self::push_device_ptr) and handed to the driver as a
/// `*mut c_void` array at launch; the driver copies the parameter values out
/// before `cuLaunchKernel` returns, so the storage is freed when the launch is
/// dropped (after submission, or unsubmitted).
#[derive(Debug)]
pub struct AsyncKernelLaunch {
    pub func: Arc<Function>,
    args: KernelArgStorage,
    cfg: Option<LaunchConfig>,
}

// SAFETY: `func` is an `Arc<Function>` (`Send + Sync`). Every pointer in `args`
// was produced by `Box::<T>::into_raw` for a `T: Send` (scalars are `DType:
// Send + Sync + Copy + 'static`; device pointers are `CUdeviceptr`), the boxed
// values are never exposed to safe code, and `KernelArgStorage` frees each one
// on the dropping thread with its original type. No thread-affine value can
// enter the storage, so moving the launch to another thread is sound.
unsafe impl Send for AsyncKernelLaunch {}

/// Type-erased kernel argument values, stored inline in 16-byte slots.
///
/// `cuLaunchKernel` takes an array of `*mut c_void` pointing at the parameter
/// VALUES. Every value pushed here is a plain `Copy` scalar or device pointer
/// (the only two `push` callers), so the storage is a bump arena with no
/// per-argument heap allocation and no destructor bookkeeping. The pointer
/// array is materialized only at launch, after every push, so arena growth
/// can never invalidate a recorded pointer.
#[derive(Default)]
struct KernelArgStorage {
    /// 16-byte-aligned value slots.
    values: Vec<u128>,
    /// Slot index of each argument, in push order.
    offsets: Vec<usize>,
    /// Parameter pointers in push order; rebuilt from `offsets` at launch.
    ptrs: Vec<*mut c_void>,
}

impl Debug for KernelArgStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelArgStorage")
            .field("len", &self.offsets.len())
            .field("offsets", &self.offsets)
            .finish()
    }
}

impl KernelArgStorage {
    /// Copies `arg` into the arena. `Copy` makes the missing destructor
    /// bookkeeping sound: dropping the arena drops nothing.
    fn push<T: Copy + Send>(&mut self, arg: T) {
        const SLOT: usize = std::mem::size_of::<u128>();
        const {
            assert!(std::mem::align_of::<T>() <= SLOT);
        }
        let slots = std::mem::size_of::<T>().div_ceil(SLOT).max(1);
        let offset = self.values.len();
        self.values.resize(offset + slots, 0);
        // SAFETY: the reserved span is at least `size_of::<T>()` bytes and
        // 16-byte aligned, which the const assertion above bounds `T`'s
        // alignment by; `T: Copy` so overwriting the zeroed slots drops
        // nothing.
        unsafe { std::ptr::write(self.values.as_mut_ptr().add(offset) as *mut T, arg) };
        self.offsets.push(offset);
    }

    /// The parameter pointer array in push order, as `cuLaunchKernel` wants it.
    fn as_mut_slice(&mut self) -> &mut [*mut c_void] {
        let base = self.values.as_mut_ptr();
        self.ptrs.clear();
        self.ptrs.extend(
            self.offsets
                .iter()
                // SAFETY: every offset was a valid slot index at push time and
                // the arena only grows.
                .map(|&offset| unsafe { base.add(offset) } as *mut c_void),
        );
        &mut self.ptrs
    }
}

impl AsyncKernelLaunch {
    /// Creates a new kernel launch builder for the given CUDA function.
    pub fn new(func: Arc<Function>) -> AsyncKernelLaunch {
        AsyncKernelLaunch {
            func,
            args: KernelArgStorage::default(),
            cfg: None,
        }
    }

    /// Pushes a kernel argument by value.
    #[inline(always)]
    pub fn push_arg<T: KernelArgument>(&mut self, arg: T) -> &mut Self {
        arg.push_arg(self);
        self
    }

    /// Pushes a kernel argument from an `Arc` reference.
    #[inline(always)]
    pub fn push_arg_arc<T: ArcKernelArgument>(&mut self, arg: &Arc<T>) -> &mut Self {
        arg.push_arg_arc(self);
        self
    }

    /// Pushes a device pointer as a kernel argument.
    ///
    /// # Safety
    /// `ptr` must stay a valid device allocation, on the device that owns the
    /// stream this launch is executed on, until the kernel has completed — not
    /// merely until the launch is submitted. The kernel signature must expect a
    /// pointer at this position.
    pub unsafe fn push_device_ptr(&mut self, ptr: CUdeviceptr) -> &mut Self {
        self.push_arg_raw(ptr)
    }

    /// Pushes a raw argument to the kernel parameter list.
    ///
    /// # Safety
    /// `T` must match the size and alignment of the kernel's formal parameter
    /// at this position; the driver copies `size_of::<T>()` bytes from the
    /// stored value.
    unsafe fn push_arg_raw<T: Copy + Send>(&mut self, arg: T) -> &mut Self {
        self.args.push(arg);
        self
    }

    /// Sets the grid/block dimensions and shared memory configuration for the launch.
    pub fn set_launch_config(&mut self, cfg: LaunchConfig) -> &mut Self {
        self.cfg = Some(cfg);
        self
    }

    /// Launches the kernel on the given CUDA stream.
    ///
    /// # Safety
    /// The caller must ensure the kernel arguments and launch config are valid.
    unsafe fn launch(mut self, stream: &Arc<Stream>) -> Result<(), DeviceError> {
        let cfg = self.cfg.ok_or_else(|| {
            DeviceError::Launch("Await called before launching the kernel.".to_string())
        })?;
        launch_kernel(
            self.func.cu_function(),
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            self.args.as_mut_slice(),
        )
        .with_context(|| {
            format!(
                r#"
                Failed to launch kernel.
                args: {:#?}
                cfg: {:#?}"#,
                self.args, cfg
            )
        })?;
        Ok(())
    }
}

/// A kernel argument that can be pushed from an `Arc` reference.
pub trait ArcKernelArgument {
    // #[inline(always)] Dont think this is necessary. This will be deprecated for required trait methods
    fn push_arg_arc(self: &Arc<Self>, launcher: &mut AsyncKernelLaunch);
}

/// A kernel argument that can be pushed by value into an `AsyncKernelLaunch`.
pub trait KernelArgument {
    // #[inline(always)] Dont think this is necessary. This will be deprecated for required trait methods
    fn push_arg(self, launcher: &mut AsyncKernelLaunch);
}

/// Safe implementation for scalar types. Values implementing `DType` are copied
/// into the kernel's parameter space during launch — the kernel reads the value,
/// not a device pointer, so no `unsafe` is required.
impl<T: DType> KernelArgument for T {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        // SAFETY: a `DType` scalar is a plain `Copy` value with the layout the
        // compiled kernel declares for that scalar type; the launcher's
        // signature validation (in cutile) checks the position, and the value
        // is copied out by the driver at launch, so nothing outlives the box.
        unsafe {
            launcher.push_arg_raw(self);
        }
    }
}

impl DeviceOp for AsyncKernelLaunch {
    type Output = ();

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        self.launch(ctx.get_cuda_stream())
    }
}

impl IntoFuture for AsyncKernelLaunch {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), AsyncKernelLaunch>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            let mut f = DeviceFuture::new();
            f.device_operation = Some(self);
            f.execution_context = Some(ExecutionContext::new(stream));
            Ok(f)
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

#[cfg(test)]
mod arg_storage_tests {
    //! Host-only: the storage never touches the driver.

    use super::*;

    /// Values of every accepted width roundtrip through the arena, and the
    /// pointer array — materialized after all pushes — points at the values
    /// even when the arena reallocated while growing.
    #[test]
    fn values_roundtrip_and_survive_arena_growth() {
        let mut storage = KernelArgStorage::default();
        storage.push(7u8);
        storage.push(0x1122_3344_5566_7788u64);
        storage.push(-5i32);
        for i in 0..64u64 {
            storage.push(i);
        }
        let ptrs = storage.as_mut_slice();
        assert_eq!(ptrs.len(), 3 + 64);
        assert_eq!(unsafe { *(ptrs[0] as *const u8) }, 7);
        assert_eq!(unsafe { *(ptrs[1] as *const u64) }, 0x1122_3344_5566_7788);
        assert_eq!(unsafe { *(ptrs[2] as *const i32) }, -5);
        for i in 0..64usize {
            assert_eq!(unsafe { *(ptrs[3 + i] as *const u64) }, i as u64);
        }
    }

    /// Every parameter pointer is aligned for its slot (16 bytes), which
    /// bounds all accepted argument types; the `const` assertion in `push`
    /// rejects wider alignments at compile time.
    #[test]
    fn slots_are_sixteen_byte_aligned() {
        let mut storage = KernelArgStorage::default();
        storage.push(1u8);
        storage.push(2u128);
        let ptrs = storage.as_mut_slice();
        assert!(ptrs.iter().all(|&p| (p as usize).is_multiple_of(16)));
        assert_eq!(unsafe { *(ptrs[1] as *const u128) }, 2);
    }
}
