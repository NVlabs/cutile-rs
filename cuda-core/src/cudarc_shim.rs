/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Portions of this file are copyright per https://github.com/chelsea0x3b/cudarc
 */

//! CUDA context, stream, event, module, and function wrappers.
//!
//! Provides RAII types that manage CUDA driver object lifetimes and expose
//! safe(r) methods for common operations.

use std::ffi::{c_int, c_uint, c_void, CString};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
    Arc,
};

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

/// Owns a CUDA primary context and tracks stream/event/error state.
#[derive(Debug)]
pub struct CudaContext {
    pub(crate) cu_device: cuda_bindings::CUdevice,
    pub(crate) cu_ctx: cuda_bindings::CUcontext,
    pub(crate) ordinal: usize,
    #[expect(
        dead_code,
        reason = "cached device capability for future async memory pool decisions"
    )]
    pub(crate) has_async_alloc: bool,
    pub(crate) num_streams: AtomicUsize,
    pub(crate) event_tracking: AtomicBool,
    pub(crate) error_state: AtomicU32,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl Drop for CudaContext {
    fn drop(&mut self) {
        self.record_err(self.bind_to_thread());
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            self.record_err(unsafe { primary_ctx::release(self.cu_device) });
        }
    }
}

impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        self.cu_device == other.cu_device
            && self.cu_ctx == other.cu_ctx
            && self.ordinal == other.ordinal
    }
}
impl Eq for CudaContext {}

impl CudaContext {
    /// Creates a new context on the specified device ordinal.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        unsafe { init(0)? };
        let cu_device = device::get(ordinal as c_int)?;
        let cu_ctx = unsafe { primary_ctx::retain(cu_device) }?;
        let has_async_alloc = unsafe {
            let memory_pools_supported = device::get_attribute(
                cu_device,
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
            )?;
            memory_pools_supported > 0
        };
        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            has_async_alloc,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Returns the number of CUDA-capable devices available.
    pub fn device_count() -> Result<i32, DriverError> {
        unsafe { init(0)? };
        device::get_count()
    }

    /// Get the `ordinal` index of the device this is on.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the name of this device.
    pub fn name(&self) -> Result<String, DriverError> {
        self.check_err()?;
        device::get_name(self.cu_device)
    }

    /// Get the UUID of this device.
    pub fn uuid(&self) -> Result<cuda_bindings::CUuuid, DriverError> {
        self.check_err()?;
        device::get_uuid(self.cu_device)
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
        self.check_err()?;
        if match ctx::get_current()? {
            Some(curr_ctx) => curr_ctx != self.cu_ctx,
            None => true,
        } {
            unsafe { ctx::set_current(self.cu_ctx) }?;
        }
        Ok(())
    }

    /// Queries a device attribute for the underlying device.
    pub fn attribute(&self, attrib: cuda_bindings::CUdevice_attribute) -> Result<i32, DriverError> {
        self.check_err()?;
        unsafe { device::get_attribute(self.cu_device, attrib) }
    }

    /// Blocks until all work on this context's device is complete.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        ctx::synchronize()
    }

    /// Configures the context to use blocking synchronization.
    pub fn set_blocking_synchronize(&self) -> Result<(), DriverError> {
        self.set_flags(cuda_bindings::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC)
    }

    /// Sets context flags (e.g. scheduling policy).
    pub fn set_flags(&self, flags: cuda_bindings::CUctx_flags) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        ctx::set_flags(flags)
    }

    /// Returns `true` if more than one stream has been created on this context.
    pub fn is_in_multi_stream_mode(&self) -> bool {
        self.num_streams.load(Ordering::Relaxed) > 0
    }

    /// Returns `true` if event tracking is enabled.
    pub fn is_event_tracking(&self) -> bool {
        self.event_tracking.load(Ordering::Relaxed)
    }

    /// Enables event tracking for stream synchronization on multi-stream transitions.
    ///
    /// # Safety
    /// Caller must ensure no concurrent stream creation races with this call.
    pub unsafe fn enable_event_tracking(&self) {
        self.event_tracking.store(true, Ordering::Relaxed);
    }

    /// Disables event tracking.
    ///
    /// # Safety
    /// Caller must ensure disabling tracking does not introduce data races.
    pub unsafe fn disable_event_tracking(&self) {
        self.event_tracking.store(false, Ordering::Relaxed);
    }

    /// Checks and clears the context's recorded error state.
    pub fn check_err(&self) -> Result<(), DriverError> {
        let error_state = self.error_state.swap(0, Ordering::Relaxed);
        if error_state == 0 {
            Ok(())
        } else {
            Err(DriverError(unsafe {
                std::mem::transmute::<u32, cuda_bindings::cudaError_enum>(error_state)
            }))
        }
    }

    /// Records an error into the context's error state if the result is `Err`.
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(err) = result {
            self.error_state.store(err.0 as u32, Ordering::Relaxed)
        }
    }
}

/// Owns a CUDA stream and its parent context reference.
#[derive(Debug, PartialEq, Eq)]
pub struct CudaStream {
    pub(crate) cu_stream: cuda_bindings::CUstream,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        if !self.cu_stream.is_null() {
            self.ctx.num_streams.fetch_sub(1, Ordering::Relaxed);
            self.ctx
                .record_err(unsafe { stream::destroy(self.cu_stream) });
        }
    }
}

impl CudaContext {
    /// Returns the default (null) CUDA stream for this context.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            cu_stream: std::ptr::null_mut(),
            ctx: self.clone(),
        })
    }
    /// Creates a new non-blocking CUDA stream on this context.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        self.bind_to_thread()?;
        let prev_num_streams = self.num_streams.fetch_add(1, Ordering::Relaxed);
        if prev_num_streams == 0 && self.is_event_tracking() {
            self.synchronize()?;
        }
        let cu_stream = stream::create(stream::StreamKind::NonBlocking)?;
        Ok(Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
        }))
    }
}

impl CudaStream {
    /// Creates a new stream that waits on this stream's current work before proceeding.
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        self.ctx.bind_to_thread()?;
        self.ctx.num_streams.fetch_add(1, Ordering::Relaxed);
        let cu_stream = stream::create(stream::StreamKind::NonBlocking)?;
        let stream = Arc::new(CudaStream {
            cu_stream,
            ctx: self.ctx.clone(),
        });
        stream.join(self)?;
        Ok(stream)
    }

    /// Returns the raw `CUstream` handle.
    pub fn cu_stream(&self) -> cuda_bindings::CUstream {
        self.cu_stream
    }

    /// Returns a reference to the parent context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Blocks until all work on this stream is complete.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { stream::synchronize(self.cu_stream) }
    }

    /// Records a new event on this stream and returns it.
    pub fn record_event(
        &self,
        flags: Option<cuda_bindings::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let event = self.ctx.new_event(flags)?;
        event.record(self)?;
        Ok(event)
    }

    /// Makes this stream wait until the given event has been recorded.
    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        if self.ctx != event.ctx {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.ctx.bind_to_thread()?;
        unsafe {
            stream::wait_event(
                self.cu_stream,
                event.cu_event,
                cuda_bindings::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
            )
        }
    }
    /// Makes this stream wait until all prior work on `other` is complete.
    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        self.wait(&other.record_event(None)?)
    }
    /// Enqueues a host-side callback to execute after all prior stream work completes.
    pub fn launch_host_function<F: FnOnce() + Send>(
        &self,
        host_func: F,
    ) -> Result<(), DriverError> {
        let boxed_host_func = Box::new(host_func);
        unsafe {
            stream::launch_host_function(
                self.cu_stream,
                CudaStream::callback_wrapper::<F>,
                // Memory allocated for the callback is wrapped in a ManuallyDrop.
                Box::into_raw(boxed_host_func) as *mut c_void,
            )
        }
    }
    unsafe extern "C" fn callback_wrapper<F: FnOnce() + Send>(callback: *mut c_void) {
        // Stop panics from unwinding across the FFI.
        let _ = std::panic::catch_unwind(|| {
            // Any memory allocated for the callback is freed when this Box goes out of scope.
            let callback: Box<F> = Box::from_raw(callback as *mut F);
            callback();
        });
    }
}

/// Owns a CUDA event and its parent context reference.
#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) cu_event: cuda_bindings::CUevent,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { event::destroy(self.cu_event) });
    }
}

impl CudaContext {
    /// Creates a new CUDA event with the given flags (defaults to disable-timing).
    pub fn new_event(
        self: &Arc<Self>,
        flags: Option<cuda_bindings::CUevent_flags>,
    ) -> Result<CudaEvent, DriverError> {
        let flags = flags.unwrap_or(cuda_bindings::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING);
        self.bind_to_thread()?;
        let cu_event = event::create(flags)?;
        Ok(CudaEvent {
            cu_event,
            ctx: self.clone(),
        })
    }
}

impl CudaEvent {
    /// Returns the raw `CUevent` handle.
    pub fn cu_event(&self) -> cuda_bindings::CUevent {
        self.cu_event
    }

    /// Returns a reference to the parent context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Records this event on the given stream.
    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx != stream.ctx {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.ctx.bind_to_thread()?;
        unsafe { event::record(self.cu_event, stream.cu_stream) }
    }

    /// Blocks until this event has been recorded.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { event::synchronize(self.cu_event) }
    }

    /// Returns elapsed time in milliseconds between `self` (start) and `end`.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, DriverError> {
        if self.ctx != end.ctx {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.ctx.bind_to_thread()?;
        self.synchronize()?;
        end.synchronize()?;
        unsafe { event::elapsed(self.cu_event, end.cu_event) }
    }

    /// Returns `true` if all work preceding this event has completed.
    pub fn is_complete(&self) -> bool {
        unsafe { event::query(self.cu_event) }.is_ok()
    }
}

/// Owns a loaded CUDA module (PTX/cubin) and its parent context reference.
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: cuda_bindings::CUmodule,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { module::unload(self.cu_module) });
    }
}

impl CudaContext {
    /// Loads a CUDA module from a PTX source string.
    pub fn load_module_from_ptx_src(
        self: &Arc<Self>,
        ptx_src: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let cu_module = {
            let c_src = CString::new(ptx_src).unwrap();
            unsafe { module::load_data(c_src.as_ptr() as *const _) }
        }?;
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
    /// Loads a CUDA module from a file path (PTX or cubin).
    pub fn load_module_from_file(
        self: &Arc<Self>,
        filename: &str,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let cu_module = { module::load(filename) }?;
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
        }))
    }
}

/// Handle to a device function loaded from a [`CudaModule`].
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: cuda_bindings::CUfunction,
    #[allow(unused)]
    pub(crate) module: Arc<CudaModule>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaModule {
    /// Looks up a device function by name within this module.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        let cu_function = unsafe { module::get_function(self.cu_module, fn_name) }?;
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }
}

impl CudaFunction {
    /// Returns the raw `CUfunction` handle.
    ///
    /// # Safety
    /// The caller must not use the handle after the parent module is dropped.
    pub unsafe fn cu_function(&self) -> cuda_bindings::CUfunction {
        self.cu_function
    }

    /// Returns the available dynamic shared memory per block for the given configuration.
    pub fn occupancy_available_dynamic_smem_per_block(
        &self,
        num_blocks: u32,
        block_size: u32,
    ) -> Result<usize, DriverError> {
        let mut dynamic_smem_size: usize = 0;

        unsafe {
            cuda_bindings::cuOccupancyAvailableDynamicSMemPerBlock(
                &mut dynamic_smem_size,
                self.cu_function,
                num_blocks as c_int,
                block_size as c_int,
            )
            .result()?
        };

        Ok(dynamic_smem_size)
    }

    /// Returns the maximum number of active blocks per SM for the given block size.
    pub fn occupancy_max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_smem_size: usize,
        flags: Option<cuda_bindings::CUoccupancy_flags_enum>,
    ) -> Result<u32, DriverError> {
        let mut num_blocks: c_int = 0;
        let flags = flags.unwrap_or(cuda_bindings::CUoccupancy_flags_enum_CU_OCCUPANCY_DEFAULT);

        unsafe {
            cuda_bindings::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &mut num_blocks,
                self.cu_function,
                block_size as c_int,
                dynamic_smem_size,
                flags as c_uint,
            )
            .result()?
        };

        Ok(num_blocks as u32)
    }

    /// Returns the maximum number of active clusters for the given launch configuration.
    pub fn occupancy_max_active_clusters(
        &self,
        config: LaunchConfig,
        stream: &CudaStream,
    ) -> Result<u32, DriverError> {
        let mut num_clusters: c_int = 0;

        let cfg = cuda_bindings::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: stream.cu_stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            cuda_bindings::cuOccupancyMaxActiveClusters(&mut num_clusters, self.cu_function, &cfg)
                .result()?
        };

        Ok(num_clusters as u32)
    }

    /// Returns `(min_grid_size, block_size)` that achieves maximum occupancy.
    pub fn occupancy_max_potential_block_size(
        &self,
        block_size_to_dynamic_smem_size: extern "C" fn(block_size: c_int) -> usize,
        dynamic_smem_size: usize,
        block_size_limit: u32,
        flags: Option<cuda_bindings::CUoccupancy_flags_enum>,
    ) -> Result<(u32, u32), DriverError> {
        let mut min_grid_size: c_int = 0;
        let mut block_size: c_int = 0;
        let flags = flags.unwrap_or(cuda_bindings::CUoccupancy_flags_enum_CU_OCCUPANCY_DEFAULT);

        unsafe {
            cuda_bindings::cuOccupancyMaxPotentialBlockSizeWithFlags(
                &mut min_grid_size,
                &mut block_size,
                self.cu_function,
                Some(block_size_to_dynamic_smem_size),
                dynamic_smem_size,
                block_size_limit as c_int,
                flags as c_uint,
            )
            .result()?
        };

        Ok((min_grid_size as u32, block_size as u32))
    }

    /// Returns the maximum potential cluster size for the given launch configuration.
    pub fn occupancy_max_potential_cluster_size(
        &self,
        config: LaunchConfig,
        stream: &CudaStream,
    ) -> Result<u32, DriverError> {
        let mut cluster_size: c_int = 0;

        let cfg = cuda_bindings::CUlaunchConfig {
            gridDimX: config.grid_dim.0,
            gridDimY: config.grid_dim.1,
            gridDimZ: config.grid_dim.2,
            blockDimX: config.block_dim.0,
            blockDimY: config.block_dim.1,
            blockDimZ: config.block_dim.2,
            sharedMemBytes: config.shared_mem_bytes,
            hStream: stream.cu_stream,
            attrs: std::ptr::null_mut(),
            numAttrs: 0,
        };

        unsafe {
            cuda_bindings::cuOccupancyMaxPotentialClusterSize(
                &mut cluster_size,
                self.cu_function,
                &cfg,
            )
            .result()?
        };

        Ok(cluster_size as u32)
    }

    /// Sets a function attribute (e.g. max dynamic shared memory).
    pub fn set_attribute(
        &self,
        attribute: cuda_bindings::CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), DriverError> {
        unsafe { function::set_function_attribute(self.cu_function, attribute, value) }
    }

    /// Sets the preferred cache configuration for this function.
    pub fn set_function_cache_config(
        &self,
        attribute: cuda_bindings::CUfunc_cache_enum,
    ) -> Result<(), DriverError> {
        unsafe { function::set_function_cache_config(self.cu_function, attribute) }
    }
}

/// Low-level primary context retain/release operations.
pub mod primary_ctx {

    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Retains the primary context for the given device, incrementing its reference count.
    ///
    /// # Safety
    /// `dev` must be a valid CUDA device handle.
    pub unsafe fn retain(
        dev: cuda_bindings::CUdevice,
    ) -> Result<cuda_bindings::CUcontext, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        cuda_bindings::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev).result()?;
        Ok(ctx.assume_init())
    }

    /// Releases the primary context for the given device.
    ///
    /// # Safety
    /// Must be paired with a prior `retain` call.
    pub unsafe fn release(dev: cuda_bindings::CUdevice) -> Result<(), DriverError> {
        cuda_bindings::cuDevicePrimaryCtxRelease_v2(dev).result()
    }
}

/// Low-level device query operations.
pub mod device {

    use super::{DriverError, IntoResult};
    use std::{
        ffi::{c_int, CStr},
        mem::MaybeUninit,
        string::String,
    };

    /// Returns the device handle for the given ordinal.
    pub fn get(ordinal: c_int) -> Result<cuda_bindings::CUdevice, DriverError> {
        let mut dev = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGet(dev.as_mut_ptr(), ordinal).result()?;
            Ok(dev.assume_init())
        }
    }

    /// Returns the number of CUDA-capable devices.
    pub fn get_count() -> Result<c_int, DriverError> {
        let mut count = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGetCount(count.as_mut_ptr()).result()?;
            Ok(count.assume_init())
        }
    }

    /// Returns the total memory in bytes on the device.
    ///
    /// # Safety
    /// `dev` must be a valid device handle.
    pub unsafe fn total_mem(dev: cuda_bindings::CUdevice) -> Result<usize, DriverError> {
        let mut bytes = MaybeUninit::uninit();
        cuda_bindings::cuDeviceTotalMem_v2(bytes.as_mut_ptr(), dev).result()?;
        Ok(bytes.assume_init())
    }

    /// Queries a device attribute value.
    ///
    /// # Safety
    /// `dev` must be a valid device handle.
    pub unsafe fn get_attribute(
        dev: cuda_bindings::CUdevice,
        attrib: cuda_bindings::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut value = MaybeUninit::uninit();
        cuda_bindings::cuDeviceGetAttribute(value.as_mut_ptr(), attrib, dev).result()?;
        Ok(value.assume_init())
    }

    /// Returns the device name as a string.
    pub fn get_name(dev: cuda_bindings::CUdevice) -> Result<String, DriverError> {
        const BUF_SIZE: usize = 128;
        let mut buf = [0u8; BUF_SIZE];
        unsafe {
            cuda_bindings::cuDeviceGetName(buf.as_mut_ptr() as _, BUF_SIZE as _, dev).result()?;
        }
        let name = CStr::from_bytes_until_nul(&buf).expect("No null byte was present");
        Ok(String::from_utf8_lossy(name.to_bytes()).into())
    }

    /// Returns the UUID of the device.
    pub fn get_uuid(dev: cuda_bindings::CUdevice) -> Result<cuda_bindings::CUuuid, DriverError> {
        let id: cuda_bindings::CUuuid;
        unsafe {
            let mut uuid = MaybeUninit::uninit();
            cuda_bindings::cuDeviceGetUuid_v2(uuid.as_mut_ptr(), dev).result()?;
            id = uuid.assume_init();
        }
        Ok(id)
    }
}

/// Low-level function attribute operations.
pub mod function {

    use super::{DriverError, IntoResult};

    /// Sets a function attribute value.
    ///
    /// # Safety
    /// `f` must be a valid function handle.
    pub unsafe fn set_function_attribute(
        f: cuda_bindings::CUfunction,
        attribute: cuda_bindings::CUfunction_attribute_enum,
        value: i32,
    ) -> Result<(), DriverError> {
        unsafe {
            cuda_bindings::cuFuncSetAttribute(f, attribute, value).result()?;
        }
        Ok(())
    }

    /// Sets the preferred cache configuration for a function.
    ///
    /// # Safety
    /// `f` must be a valid function handle.
    pub unsafe fn set_function_cache_config(
        f: cuda_bindings::CUfunction,
        attribute: cuda_bindings::CUfunc_cache_enum,
    ) -> Result<(), DriverError> {
        unsafe {
            cuda_bindings::cuFuncSetCacheConfig(f, attribute).result()?;
        }
        Ok(())
    }
}

/// Low-level CUDA context management operations.
pub mod ctx {
    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Sets the current CUDA context for the calling thread.
    ///
    /// # Safety
    /// `ctx` must be a valid context handle.
    pub unsafe fn set_current(ctx: cuda_bindings::CUcontext) -> Result<(), DriverError> {
        cuda_bindings::cuCtxSetCurrent(ctx).result()
    }

    /// Returns the CUDA context bound to the calling thread, or `None`.
    pub fn get_current() -> Result<Option<cuda_bindings::CUcontext>, DriverError> {
        let mut ctx = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuCtxGetCurrent(ctx.as_mut_ptr()).result()?;
            let ctx: cuda_bindings::CUcontext = ctx.assume_init();
            if ctx.is_null() {
                Ok(None)
            } else {
                Ok(Some(ctx))
            }
        }
    }

    /// Sets flags on the current context.
    pub fn set_flags(flags: cuda_bindings::CUctx_flags) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuCtxSetFlags(flags as u32).result() }
    }

    /// Blocks until all work in the current context is complete.
    pub fn synchronize() -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuCtxSynchronize() }.result()
    }
}

/// Low-level CUDA stream operations.
pub mod stream {
    use super::{DriverError, IntoResult};
    use std::ffi::c_void;
    use std::mem::MaybeUninit;

    /// The kind of CUDA stream to create.
    pub enum StreamKind {
        /// > Default stream creation flag.
        Default,

        /// > Specifies that work running in the created stream
        /// > may run concurrently with work in stream 0 (the NULL stream),
        /// > and that the created stream should perform no implicit
        /// > synchronization with stream 0.
        NonBlocking,
    }

    impl StreamKind {
        fn flags(self) -> cuda_bindings::CUstream_flags {
            match self {
                Self::Default => cuda_bindings::CUstream_flags_enum_CU_STREAM_DEFAULT,
                Self::NonBlocking => cuda_bindings::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
            }
        }
    }

    /// Returns the null (default) stream handle.
    pub fn null() -> cuda_bindings::CUstream {
        std::ptr::null_mut()
    }

    /// Creates a new CUDA stream of the given kind.
    pub fn create(kind: StreamKind) -> Result<cuda_bindings::CUstream, DriverError> {
        let mut stream = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuStreamCreate(stream.as_mut_ptr(), kind.flags() as u32).result()?;
            Ok(stream.assume_init())
        }
    }

    /// Blocks until all work on the stream is complete.
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn synchronize(stream: cuda_bindings::CUstream) -> Result<(), DriverError> {
        cuda_bindings::cuStreamSynchronize(stream).result()
    }

    /// Destroys a CUDA stream.
    ///
    /// # Safety
    /// `stream` must be valid and not in use.
    pub unsafe fn destroy(stream: cuda_bindings::CUstream) -> Result<(), DriverError> {
        cuda_bindings::cuStreamDestroy_v2(stream).result()
    }

    /// Makes a stream wait on an event.
    ///
    /// # Safety
    /// Both handles must be valid.
    pub unsafe fn wait_event(
        stream: cuda_bindings::CUstream,
        event: cuda_bindings::CUevent,
        flags: cuda_bindings::CUevent_wait_flags,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuStreamWaitEvent(stream, event, flags as u32).result()
    }

    /// Attaches memory to a stream for managed memory visibility.
    ///
    /// # Safety
    /// `dptr` must be a valid managed memory pointer.
    pub unsafe fn attach_mem_async(
        stream: cuda_bindings::CUstream,
        dptr: cuda_bindings::CUdeviceptr,
        num_bytes: usize,
        flags: cuda_bindings::CUmemAttach_flags,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuStreamAttachMemAsync(stream, dptr, num_bytes, flags as u32).result()
    }

    /// Enqueues a host function callback on the stream.
    ///
    /// # Safety
    /// `func` and `arg` must remain valid until the callback executes.
    pub unsafe fn launch_host_function(
        stream: cuda_bindings::CUstream,
        func: unsafe extern "C" fn(*mut ::core::ffi::c_void),
        arg: *mut c_void,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuLaunchHostFunc(stream, Some(func), arg).result()
    }

    /// Begins stream capture for graph construction.
    ///
    /// # Safety
    /// `stream` must be valid and not already capturing.
    pub unsafe fn begin_capture(
        stream: cuda_bindings::CUstream,
        mode: cuda_bindings::CUstreamCaptureMode,
    ) -> Result<(), DriverError> {
        cuda_bindings::cuStreamBeginCapture_v2(stream, mode).result()
    }

    /// Ends stream capture and returns the captured graph.
    ///
    /// # Safety
    /// `stream` must be in a capturing state.
    pub unsafe fn end_capture(
        stream: cuda_bindings::CUstream,
    ) -> Result<cuda_bindings::CUgraph, DriverError> {
        let mut graph = MaybeUninit::uninit();
        cuda_bindings::cuStreamEndCapture(stream, graph.as_mut_ptr()).result()?;
        Ok(graph.assume_init())
    }

    /// Queries whether the stream is currently capturing.
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn is_capturing(
        stream: cuda_bindings::CUstream,
    ) -> Result<cuda_bindings::CUstreamCaptureStatus, DriverError> {
        let mut status = MaybeUninit::uninit();
        cuda_bindings::cuStreamIsCapturing(stream, status.as_mut_ptr()).result()?;
        Ok(status.assume_init())
    }
}

/// Low-level CUDA module load/unload and function lookup operations.
pub mod module {
    use super::{DriverError, IntoResult};
    use core::ffi::c_void;
    use std::ffi::CString;
    use std::mem::MaybeUninit;

    /// Loads a CUDA module from a file path.
    pub fn load(filename: &str) -> Result<cuda_bindings::CUmodule, DriverError> {
        let c_str = CString::new(filename).unwrap();
        let fname_ptr = c_str.as_c_str().as_ptr();
        let mut module = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuModuleLoad(module.as_mut_ptr(), fname_ptr).result()?;
            Ok(module.assume_init())
        }
    }

    /// Loads a CUDA module from a PTX source string.
    ///
    /// # Safety
    /// The PTX source must be valid.
    pub unsafe fn load_ptx_str(src_str: &str) -> Result<cuda_bindings::CUmodule, DriverError> {
        let mut module = MaybeUninit::uninit();
        let c_str = CString::new(src_str).unwrap();
        let module_res =
            cuda_bindings::cuModuleLoadData(module.as_mut_ptr(), c_str.as_ptr() as *const _);
        (module_res, module).result()
    }

    /// Loads a CUDA module from a raw data image pointer.
    ///
    /// # Safety
    /// `image` must point to valid module data (PTX or cubin).
    pub unsafe fn load_data(image: *const c_void) -> Result<cuda_bindings::CUmodule, DriverError> {
        let mut module = MaybeUninit::uninit();
        cuda_bindings::cuModuleLoadData(module.as_mut_ptr(), image).result()?;
        Ok(module.assume_init())
    }

    /// Looks up a device function by name within a module.
    ///
    /// # Safety
    /// `module` must be a valid, loaded module handle.
    pub unsafe fn get_function(
        module: cuda_bindings::CUmodule,
        name: &str,
    ) -> Result<cuda_bindings::CUfunction, DriverError> {
        let name = CString::new(name).unwrap();
        let name_ptr = name.as_c_str().as_ptr();
        let mut func = MaybeUninit::uninit();
        let res = cuda_bindings::cuModuleGetFunction(func.as_mut_ptr(), module, name_ptr);
        (res, func).result()
    }

    /// Unloads a CUDA module.
    ///
    /// # Safety
    /// `module` must be valid and all functions from it must no longer be in use.
    pub unsafe fn unload(module: cuda_bindings::CUmodule) -> Result<(), DriverError> {
        cuda_bindings::cuModuleUnload(module).result()
    }
}

/// Low-level CUDA event operations.
pub mod event {
    use super::{DriverError, IntoResult};
    use std::mem::MaybeUninit;

    /// Creates a new CUDA event with the given flags.
    pub fn create(
        flags: cuda_bindings::CUevent_flags,
    ) -> Result<cuda_bindings::CUevent, DriverError> {
        let mut event = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuEventCreate(event.as_mut_ptr(), flags as u32).result()?;
            Ok(event.assume_init())
        }
    }

    /// Records an event on a stream.
    ///
    /// # Safety
    /// Both `event` and `stream` must be valid handles.
    pub unsafe fn record(
        event: cuda_bindings::CUevent,
        stream: cuda_bindings::CUstream,
    ) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuEventRecord(event, stream).result() }
    }

    /// Returns elapsed time in milliseconds between two recorded events.
    ///
    /// # Safety
    /// Both events must have been recorded and completed.
    pub unsafe fn elapsed(
        start: cuda_bindings::CUevent,
        end: cuda_bindings::CUevent,
    ) -> Result<f32, DriverError> {
        let mut ms: f32 = 0.0;
        unsafe {
            cuda_bindings::cuEventElapsedTime_v2((&mut ms) as *mut _, start, end).result()?;
        }
        Ok(ms)
    }

    /// Queries whether an event has completed. Returns `Ok` if complete.
    ///
    /// # Safety
    /// `event` must be a valid event handle.
    pub unsafe fn query(event: cuda_bindings::CUevent) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuEventQuery(event).result() }
    }

    /// Blocks until the event has been recorded.
    ///
    /// # Safety
    /// `event` must be a valid event handle.
    pub unsafe fn synchronize(event: cuda_bindings::CUevent) -> Result<(), DriverError> {
        unsafe { cuda_bindings::cuEventSynchronize(event).result() }
    }

    /// Destroys a CUDA event.
    ///
    /// # Safety
    /// `event` must be valid and not in use by any stream.
    pub unsafe fn destroy(event: cuda_bindings::CUevent) -> Result<(), DriverError> {
        cuda_bindings::cuEventDestroy_v2(event).result()
    }
}

/// Low-level CUDA memory allocation, transfer, and management operations.
pub mod memory {

    use crate::sys::{self};
    use std::ffi::{c_uchar, c_uint, c_void};
    use std::mem::MaybeUninit;

    use crate::error::*;

    /// Allocates device memory asynchronously on the given stream.
    ///
    /// # Safety
    /// `stream` must be a valid stream handle.
    pub unsafe fn malloc_async(
        stream: sys::CUstream,
        num_bytes: usize,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAllocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Allocates device memory synchronously.
    ///
    /// # Safety
    /// A valid CUDA context must be current.
    pub unsafe fn malloc_sync(num_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), num_bytes).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Allocates managed (unified) memory accessible from both host and device.
    ///
    /// # Safety
    /// A valid CUDA context must be current.
    pub unsafe fn malloc_managed(
        num_bytes: usize,
        flags: sys::CUmemAttach_flags,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        let mut dev_ptr = MaybeUninit::uninit();
        sys::cuMemAllocManaged(dev_ptr.as_mut_ptr(), num_bytes, flags as u32).result()?;
        Ok(dev_ptr.assume_init())
    }

    /// Allocates page-locked host memory.
    ///
    /// # Safety
    /// A valid CUDA context must be current.
    pub unsafe fn malloc_host(num_bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
        let mut host_ptr = MaybeUninit::uninit();
        sys::cuMemHostAlloc(host_ptr.as_mut_ptr(), num_bytes, flags).result()?;
        Ok(host_ptr.assume_init())
    }

    /// Frees page-locked host memory allocated by `malloc_host`.
    ///
    /// # Safety
    /// `host_ptr` must have been allocated with `malloc_host`.
    pub unsafe fn free_host(host_ptr: *mut c_void) -> Result<(), DriverError> {
        sys::cuMemFreeHost(host_ptr).result()
    }

    /// Advises the CUDA runtime about the expected access pattern for managed memory.
    ///
    /// # Safety
    /// `dptr` must be a valid managed memory pointer.
    pub unsafe fn mem_advise(
        dptr: sys::CUdeviceptr,
        num_bytes: usize,
        advice: sys::CUmem_advise,
        location: sys::CUmemLocation,
    ) -> Result<(), DriverError> {
        sys::cuMemAdvise_v2(dptr, num_bytes, advice, location).result()
    }

    /// Asynchronously prefetches managed memory to the specified location.
    ///
    /// # Safety
    /// `dptr` must be valid managed memory; `stream` must be valid.
    pub unsafe fn mem_prefetch_async(
        dptr: sys::CUdeviceptr,
        num_bytes: usize,
        location: sys::CUmemLocation,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemPrefetchAsync_v2(dptr, num_bytes, location, 0, stream).result()
    }

    /// Frees device memory asynchronously on the given stream.
    ///
    /// # Safety
    /// `dptr` must have been allocated with `malloc_async` and must not be used after this call.
    pub unsafe fn free_async(
        dptr: sys::CUdeviceptr,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemFreeAsync(dptr, stream).result()
    }

    /// Frees device memory synchronously.
    ///
    /// # Safety
    /// `dptr` must be a valid device pointer not in use.
    pub unsafe fn free_sync(dptr: sys::CUdeviceptr) -> Result<(), DriverError> {
        sys::cuMemFree_v2(dptr).result()
    }

    /// Frees device memory synchronously (alias for `free_sync`).
    ///
    /// # Safety
    /// `device_ptr` must be a valid device pointer not in use.
    pub unsafe fn memory_free(device_ptr: sys::CUdeviceptr) -> Result<(), DriverError> {
        sys::cuMemFree_v2(device_ptr).result()
    }

    /// Asynchronously sets device memory to a byte value.
    ///
    /// # Safety
    /// `dptr` must be valid device memory with at least `num_bytes` capacity.
    pub unsafe fn memset_d8_async(
        dptr: sys::CUdeviceptr,
        uc: c_uchar,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemsetD8Async(dptr, uc, num_bytes, stream).result()
    }

    /// Synchronously sets device memory to a byte value.
    ///
    /// # Safety
    /// `dptr` must be valid device memory with at least `num_bytes` capacity.
    pub unsafe fn memset_d8_sync(
        dptr: sys::CUdeviceptr,
        uc: c_uchar,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        sys::cuMemsetD8_v2(dptr, uc, num_bytes).result()
    }

    /// Asynchronously copies bytes from host to device memory.
    ///
    /// # Safety
    /// `src` and `dst` must be valid with sufficient capacity; `stream` must be valid.
    pub unsafe fn memcpy_htod_async<T>(
        dst: sys::CUdeviceptr,
        src: *const T,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyHtoDAsync_v2(dst, src as *const _, num_bytes, stream).result()
    }

    /// Synchronously copies a host slice to device memory.
    ///
    /// # Safety
    /// `dst` must have capacity for the full slice.
    pub unsafe fn memcpy_htod_sync<T>(dst: sys::CUdeviceptr, src: &[T]) -> Result<(), DriverError> {
        sys::cuMemcpyHtoD_v2(dst, src.as_ptr() as *const _, std::mem::size_of_val(src)).result()
    }

    /// Asynchronously copies bytes from device to host memory.
    ///
    /// # Safety
    /// `dst` and `src` must be valid with sufficient capacity; `stream` must be valid.
    pub unsafe fn memcpy_dtoh_async<T>(
        dst: *mut T,
        src: sys::CUdeviceptr,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoHAsync_v2(dst as *mut _, src, num_bytes, stream).result()
    }

    /// Synchronously copies device memory into a host slice.
    ///
    /// # Safety
    /// `src` must have at least as many bytes as `dst`.
    pub unsafe fn memcpy_dtoh_sync<T>(
        dst: &mut [T],
        src: sys::CUdeviceptr,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoH_v2(dst.as_mut_ptr() as *mut _, src, std::mem::size_of_val(dst)).result()
    }

    /// Asynchronously copies bytes between device memory regions.
    ///
    /// # Safety
    /// Both pointers must be valid with sufficient capacity; `stream` must be valid.
    pub unsafe fn memcpy_dtod_async(
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        num_bytes: usize,
        stream: sys::CUstream,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoDAsync_v2(dst, src, num_bytes, stream).result()
    }

    /// Synchronously copies bytes between device memory regions.
    ///
    /// # Safety
    /// Both pointers must be valid with sufficient capacity.
    pub unsafe fn memcpy_dtod_sync(
        dst: sys::CUdeviceptr,
        src: sys::CUdeviceptr,
        num_bytes: usize,
    ) -> Result<(), DriverError> {
        sys::cuMemcpyDtoD_v2(dst, src, num_bytes).result()
    }

    /// Returns `(free, total)` bytes of device memory for the current context.
    pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
        let mut free = 0;
        let mut total = 0;
        unsafe { sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) }.result()?;
        Ok((free, total))
    }
}
