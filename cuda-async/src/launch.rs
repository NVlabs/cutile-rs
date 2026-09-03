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

/// Heap-allocated, type-erased kernel arguments, each paired with the
/// destructor for its concrete type.
///
/// `cuLaunchKernel` takes an array of `*mut c_void` pointing at the parameter
/// values, so the boxes must be erased for the driver. They must *not* be
/// erased for deallocation: the allocator contract requires freeing with the
/// `Layout` the value was allocated with, and `Box::<c_void>::from_raw` on a
/// pointer that came from `Box::<T>::into_raw` deallocates with `c_void`'s
/// layout (size 1, align 1) instead of `T`'s — undefined behavior for every
/// real argument type. Each entry therefore records `drop_box::<T>` for the
/// `T` it was pushed as.
#[derive(Default)]
struct KernelArgStorage {
    ptrs: Vec<*mut c_void>,
    drops: Vec<unsafe fn(*mut c_void)>,
}

impl Debug for KernelArgStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelArgStorage")
            .field("len", &self.ptrs.len())
            .field("ptrs", &self.ptrs)
            .finish()
    }
}

impl Drop for KernelArgStorage {
    fn drop(&mut self) {
        for (arg, drop_arg) in self.ptrs.drain(..).zip(self.drops.drain(..)) {
            // SAFETY: `arg` came from `Box::<T>::into_raw` in `push`, and
            // `drop_arg` is `drop_box::<T>` for that same `T`, so the box is
            // reconstituted and freed with the layout it was allocated with.
            unsafe { drop_arg(arg) };
        }
    }
}

impl KernelArgStorage {
    /// Erases `arg` for the driver and records its typed destructor.
    fn push<T: Send>(&mut self, arg: Box<T>) {
        unsafe fn drop_box<T>(arg: *mut c_void) {
            // SAFETY: the caller (`Drop for KernelArgStorage`) passes the
            // pointer this entry was created from, typed as `T`.
            drop(unsafe { Box::from_raw(arg as *mut T) });
        }
        self.ptrs.push(Box::into_raw(arg) as *mut c_void);
        self.drops.push(drop_box::<T>);
    }

    /// The parameter pointer array in push order, as `cuLaunchKernel` wants it.
    fn as_mut_slice(&mut self) -> &mut [*mut c_void] {
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
        self.push_arg_raw(Box::new(ptr))
    }

    /// Pushes a raw argument to the kernel parameter list.
    ///
    /// # Safety
    /// `T` must match the size and alignment of the kernel's formal parameter
    /// at this position; the driver copies `size_of::<T>()` bytes from the box.
    unsafe fn push_arg_raw<T: Send>(&mut self, arg: Box<T>) -> &mut Self {
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
        let cfg = self.cfg.ok_or(DeviceError::Launch(
            "Await called before launching the kernel.".to_string(),
        ))?;
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
            launcher.push_arg_raw(Box::new(self));
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Each argument's destructor runs exactly once, when the storage drops,
    /// through the `drop_box::<T>` recorded for its own `T`.
    #[test]
    fn arguments_drop_once_with_their_original_type() {
        struct DropCounter(Arc<AtomicUsize>);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        let mut storage = KernelArgStorage::default();
        storage.push(Box::new(DropCounter(Arc::clone(&drops))));
        storage.push(Box::new(DropCounter(Arc::clone(&drops))));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(storage);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    /// Wide and over-aligned arguments are allocated and freed with their own
    /// layout, and the parameter array points at the values themselves. The
    /// previous `Box::<c_void>::from_raw` teardown deallocated these with a
    /// 1-byte/1-align layout, which Miri and ASan report as a layout mismatch.
    #[test]
    fn over_aligned_argument_roundtrips() {
        #[repr(C, align(64))]
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct Wide([u64; 8]);

        let value = Wide([1, 2, 3, 4, 5, 6, 7, 8]);
        let mut storage = KernelArgStorage::default();
        storage.push(Box::new(value));
        storage.push(Box::new(7u8));

        let ptrs = storage.as_mut_slice();
        assert_eq!(ptrs.len(), 2);
        assert_eq!(ptrs[0] as usize % 64, 0, "box honours the type's alignment");
        assert_eq!(unsafe { *(ptrs[0] as *const Wide) }, value);
        assert_eq!(unsafe { *(ptrs[1] as *const u8) }, 7);
    }
}
