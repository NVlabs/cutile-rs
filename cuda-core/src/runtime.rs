/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA runtime types: Device, Stream, Module, Function, LaunchConfig.
//!
//! These are the public API of `cuda-core`. They wrap raw CUDA driver
//! handles with RAII lifetimes and provide `borrow_raw` constructors
//! for interop with external frameworks (cudarc, etc.).

use std::ffi::{c_int, c_void, CString};
use std::sync::Arc;

use crate::cudarc_shim::{ctx, device, module, primary_ctx, stream};
use crate::error::*;
use crate::init;

/// Kernel launch configuration specifying grid, block, and shared memory sizes.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// Grid dimensions `(x, y, z)` in thread blocks.
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions `(x, y, z)` in threads.
    pub block_dim: (u32, u32, u32),
    /// Bytes of dynamic shared memory per block.
    pub shared_mem_bytes: u32,
}

/// A GPU device handle wrapping a CUDA primary context.
///
/// Can be either **owned** (created via [`Device::new`], releases the
/// primary context on drop) or **borrowed** (created via
/// [`Device::borrow_raw`], does NOT release on drop).
#[derive(Debug)]
pub struct Device {
    pub(crate) cu_device: cuda_bindings::CUdevice,
    pub(crate) cu_ctx: cuda_bindings::CUcontext,
    pub(crate) ordinal: usize,
    owned: bool,
}

unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Drop for Device {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _ = self.bind_to_thread();
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            let _ = unsafe { primary_ctx::release(self.cu_device) };
        }
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.cu_device == other.cu_device
            && self.cu_ctx == other.cu_ctx
            && self.ordinal == other.ordinal
    }
}
impl Eq for Device {}

impl Device {
    /// Creates a new owned device on the specified ordinal.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        unsafe { init(0)? };
        let cu_device = device::get(ordinal as c_int)?;
        let cu_ctx = unsafe { primary_ctx::retain(cu_device) }?;
        let device = Arc::new(Device {
            cu_device,
            cu_ctx,
            ordinal,
            owned: true,
        });
        device.bind_to_thread()?;
        Ok(device)
    }

    /// Wraps externally-owned CUDA handles without taking ownership.
    ///
    /// Inputs are the raw C primitives (`CUcontext` is an opaque pointer,
    /// `CUdevice` is `int` in the driver API). Accepting primitives rather
    /// than `cuda_bindings::CU*` typedefs keeps this API agnostic to which
    /// binding crate the caller uses — a cudarc `CUcontext`, a fresh
    /// `bindgen` wrapper, or a hand-rolled FFI type all cast in the same way.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_ctx` points to a valid retained `CUcontext` for `cu_device`
    /// - The handles outlive the returned `Device`
    /// - No concurrent destruction of the handles
    pub unsafe fn borrow_raw(cu_ctx: *mut c_void, cu_device: c_int, ordinal: usize) -> Arc<Self> {
        Arc::new(Device {
            cu_device: cu_device as cuda_bindings::CUdevice,
            cu_ctx: cu_ctx as cuda_bindings::CUcontext,
            ordinal,
            owned: false,
        })
    }

    /// Returns the number of CUDA-capable devices available.
    pub fn device_count() -> Result<i32, DriverError> {
        unsafe { init(0)? };
        device::get_count()
    }

    /// Returns the raw `CUdevice` handle for a given ordinal without
    /// creating a full `Device` (no context retained).
    pub fn raw_device(ordinal: usize) -> Result<cuda_bindings::CUdevice, DriverError> {
        unsafe { init(0)? };
        device::get(ordinal as c_int)
    }

    /// Get the `ordinal` index of the device this is on.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the name of this device.
    pub fn name(&self) -> Result<String, DriverError> {
        device::get_name(self.cu_device)
    }

    /// Returns the raw `CUdevice` handle.
    pub fn cu_device(&self) -> cuda_bindings::CUdevice {
        self.cu_device
    }

    /// Returns the raw `CUcontext` handle.
    pub fn cu_ctx(&self) -> cuda_bindings::CUcontext {
        self.cu_ctx
    }

    /// Binds this context to the calling thread if not already current.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        if match ctx::get_current()? {
            Some(curr_ctx) => curr_ctx != self.cu_ctx,
            None => true,
        } {
            unsafe { ctx::set_current(self.cu_ctx) }?;
        }
        Ok(())
    }

    /// Blocks until all work on this device's context is complete.
    ///
    /// # Safety
    /// The caller must ensure this device's context is current on the
    /// calling thread (via [`bind_to_thread`](Device::bind_to_thread)).
    pub unsafe fn synchronize(&self) -> Result<(), DriverError> {
        ctx::synchronize()
    }

    /// Creates a new non-blocking CUDA stream on this device.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<Stream>, DriverError> {
        self.bind_to_thread()?;
        let cu_stream = stream::create(stream::StreamKind::NonBlocking)?;
        Ok(Arc::new(Stream {
            cu_stream,
            device: self.clone(),
            owned: true,
        }))
    }

    /// Loads a CUDA module from a PTX source string.
    pub fn load_module_from_ptx_src(
        self: &Arc<Self>,
        ptx_src: &str,
    ) -> Result<Arc<Module>, DriverError> {
        self.bind_to_thread()?;
        let cu_module = {
            let c_src = CString::new(ptx_src).unwrap();
            unsafe { module::load_data(c_src.as_ptr() as *const _) }
        }?;
        Ok(Arc::new(Module {
            cu_module,
            device: self.clone(),
            owned: true,
        }))
    }

    /// Loads a CUDA module from a file path (PTX or cubin).
    pub fn load_module_from_file(
        self: &Arc<Self>,
        filename: &str,
    ) -> Result<Arc<Module>, DriverError> {
        self.bind_to_thread()?;
        let cu_module = { module::load(filename) }?;
        Ok(Arc::new(Module {
            cu_module,
            device: self.clone(),
            owned: true,
        }))
    }
}

/// A CUDA stream handle.
///
/// Can be either **owned** (created via [`Device::new_stream`], destroyed
/// on drop) or **borrowed** (created via [`Stream::borrow_raw`], does
/// NOT destroy on drop).
#[derive(Debug, PartialEq, Eq)]
pub struct Stream {
    pub(crate) cu_stream: cuda_bindings::CUstream,
    pub(crate) device: Arc<Device>,
    owned: bool,
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _ = self.device.bind_to_thread();
        if !self.cu_stream.is_null() {
            let _ = unsafe { stream::destroy(self.cu_stream) };
        }
    }
}

impl Stream {
    /// Wraps an externally-owned CUDA stream without taking ownership.
    ///
    /// `cu_stream` is the raw `CUstream` opaque pointer. See
    /// [`Device::borrow_raw`] for why this is a `*mut c_void` rather than a
    /// `cuda_bindings::CUstream` typedef.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_stream` points to a valid CUDA stream on `device`
    /// - The stream outlives the returned `Stream`
    /// - No concurrent destruction of the stream
    pub unsafe fn borrow_raw(cu_stream: *mut c_void, device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Stream {
            cu_stream: cu_stream as cuda_bindings::CUstream,
            device: device.clone(),
            owned: false,
        })
    }

    /// Returns the raw `CUstream` handle.
    pub fn cu_stream(&self) -> cuda_bindings::CUstream {
        self.cu_stream
    }

    /// Returns a reference to the parent device.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Blocks until all work on this stream is complete.
    ///
    /// # Safety
    /// The caller must ensure the parent device's context is current on
    /// the calling thread.
    pub unsafe fn synchronize(&self) -> Result<(), DriverError> {
        stream::synchronize(self.cu_stream)
    }

    /// Enqueues a host-side callback to execute after all prior stream work completes.
    ///
    /// # Safety
    /// The caller must ensure the parent device's context is current on
    /// the calling thread.
    pub unsafe fn launch_host_function<F: FnOnce() + Send>(
        &self,
        host_func: F,
    ) -> Result<(), DriverError> {
        let boxed_host_func = Box::new(host_func);
        stream::launch_host_function(
            self.cu_stream,
            Self::callback_wrapper::<F>,
            Box::into_raw(boxed_host_func) as *mut c_void,
        )
    }

    unsafe extern "C" fn callback_wrapper<F: FnOnce() + Send>(callback: *mut c_void) {
        let _ = std::panic::catch_unwind(|| {
            let callback: Box<F> = Box::from_raw(callback as *mut F);
            callback();
        });
    }

    /// Begins stream capture for CUDA graph construction.
    ///
    /// # Safety
    /// The caller must ensure the context is current and the stream is not
    /// already being captured.
    pub unsafe fn begin_capture(
        &self,
        mode: cuda_bindings::CUstreamCaptureMode,
    ) -> Result<(), DriverError> {
        stream::begin_capture(self.cu_stream, mode)
    }

    /// Ends stream capture and returns the captured CUDA graph.
    ///
    /// # Safety
    /// The caller must ensure `begin_capture` was previously called on this stream.
    pub unsafe fn end_capture(&self) -> Result<cuda_bindings::CUgraph, DriverError> {
        stream::end_capture(self.cu_stream)
    }
}

/// A loaded CUDA module (PTX/cubin).
///
/// Can be either **owned** (created via [`Device::load_module_from_ptx_src`]
/// / [`Device::load_module_from_file`], unloads on drop) or **borrowed**
/// (created via [`Module::borrow_raw`], does NOT unload on drop).
#[derive(Debug)]
pub struct Module {
    pub(crate) cu_module: cuda_bindings::CUmodule,
    pub(crate) device: Arc<Device>,
    owned: bool,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        let _ = self.device.bind_to_thread();
        let _ = unsafe { module::unload(self.cu_module) };
    }
}

impl Module {
    /// Wraps an externally-owned CUDA module without taking ownership.
    ///
    /// `cu_module` is the raw `CUmodule` opaque pointer. See
    /// [`Device::borrow_raw`] for why this is a `*mut c_void`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_module` points to a valid module loaded on `device`
    /// - The module outlives the returned `Module`
    /// - No concurrent unload of the module
    pub unsafe fn borrow_raw(cu_module: *mut c_void, device: &Arc<Device>) -> Arc<Self> {
        Arc::new(Module {
            cu_module: cu_module as cuda_bindings::CUmodule,
            device: device.clone(),
            owned: false,
        })
    }

    /// Returns the raw `CUmodule` handle.
    pub fn cu_module(&self) -> cuda_bindings::CUmodule {
        self.cu_module
    }

    /// Looks up a device function by name within this module.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<Function, DriverError> {
        let cu_function = unsafe { module::get_function(self.cu_module, fn_name) }?;
        Ok(Function {
            cu_function,
            module: self.clone(),
        })
    }
}

/// Handle to a device function loaded from a [`Module`].
#[derive(Debug, Clone)]
pub struct Function {
    pub(crate) cu_function: cuda_bindings::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<Module>,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Wraps an externally-owned CUDA function without taking ownership.
    ///
    /// `cu_function` is the raw `CUfunction` opaque pointer. The returned
    /// `Function` holds a clone of `module` to keep the parent alive;
    /// there is no `owned` flag because `Function` has no `Drop` (functions
    /// are not freed independently — they live as long as their module).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `cu_function` points to a valid function within `module`
    /// - `module` is the module that `cu_function` was obtained from
    pub unsafe fn borrow_raw(cu_function: *mut c_void, module: &Arc<Module>) -> Function {
        Function {
            cu_function: cu_function as cuda_bindings::CUfunction,
            module: module.clone(),
        }
    }

    /// Returns the raw `CUfunction` handle.
    ///
    /// # Safety
    /// The caller must not use the handle after the parent module is dropped.
    pub unsafe fn cu_function(&self) -> cuda_bindings::CUfunction {
        self.cu_function
    }
}
