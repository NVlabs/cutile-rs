// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

//! Dynamic loading of CUDA driver and cuRAND libraries at runtime.
//!
//! Instead of linking `libcuda` and `libcurand` at build time, this module
//! loads them on first use via [`libloading`]. This allows the crate to
//! compile on machines without a GPU or CUDA driver installed — only the
//! CUDA toolkit **headers** are needed at build time (for type definitions).
//!
//! Use [`is_cuda_driver_available`] to check at runtime whether the CUDA
//! driver library can be loaded.

use std::ffi::{c_char, c_int, c_uchar, c_uint, c_ulonglong, c_void};
use std::sync::OnceLock;

#[allow(unused_imports)]
use crate::*;

// ---------------------------------------------------------------------------
// Dynamic-load error type
// ---------------------------------------------------------------------------

/// Error returned when a CUDA shared library or symbol cannot be loaded.
#[derive(Debug)]
pub enum DynLoadError {
    /// The shared library itself could not be opened.
    LibraryNotFound {
        name: &'static str,
        source: libloading::Error,
    },
    /// The library was opened, but a required symbol was missing.
    SymbolNotFound {
        symbol: &'static str,
        lib: &'static str,
        source: libloading::Error,
    },
}

impl std::fmt::Display for DynLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynLoadError::LibraryNotFound { name, source } => {
                write!(f, "failed to load {name}: {source}")
            }
            DynLoadError::SymbolNotFound {
                symbol,
                lib,
                source,
            } => {
                write!(f, "symbol `{symbol}` not found in {lib}: {source}")
            }
        }
    }
}

impl std::error::Error for DynLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DynLoadError::LibraryNotFound { source, .. }
            | DynLoadError::SymbolNotFound { source, .. } => Some(source),
        }
    }
}

// ---------------------------------------------------------------------------
// Platform-specific library names
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
const CUDA_LIB_NAME: &str = "libcuda.so.1";
#[cfg(target_os = "macos")]
const CUDA_LIB_NAME: &str = "libcuda.dylib";
#[cfg(target_os = "windows")]
const CUDA_LIB_NAME: &str = "nvcuda.dll";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CUDA_LIB_NAME: &str = "libcuda.so";

#[cfg(target_os = "linux")]
const CURAND_LIB_NAME: &str = "libcurand.so";
#[cfg(target_os = "macos")]
const CURAND_LIB_NAME: &str = "libcurand.dylib";
#[cfg(target_os = "windows")]
const CURAND_LIB_NAME: &str = "curand64_10.dll";
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CURAND_LIB_NAME: &str = "libcurand.so";

// ---------------------------------------------------------------------------
// Macro: define a dynamically-loaded API table + thin wrapper functions
// ---------------------------------------------------------------------------

macro_rules! dynamic_api {
    (
        static $static:ident : $Struct:ident = $lib_name:expr;
        init_fn: $init_fn:ident;
        available_fn: $is_available_fn:ident;
        load_error_fn: $load_error_fn:ident;
        load_error_value: $load_error_value:expr;
        $(
            fn $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty ;
        )*
    ) => {
        struct $Struct {
            _lib: libloading::Library,
            $( $name: unsafe extern "C" fn( $($ty),* ) -> $ret, )*
        }

        // SAFETY: The function pointers are obtained from a shared library
        // that is held in a `OnceLock` and never unloaded. The function
        // pointers therefore remain valid for the lifetime of the process
        // and carry no thread-affinity.
        unsafe impl Send for $Struct {}
        unsafe impl Sync for $Struct {}

        impl $Struct {
            fn load() -> Result<Self, DynLoadError> {
                let lib_name = $lib_name;
                let lib = unsafe { libloading::Library::new(lib_name) }
                    .map_err(|e| DynLoadError::LibraryNotFound { name: lib_name, source: e })?;
                Ok(Self {
                    $(
                        $name: unsafe {
                            *lib.get(stringify!($name).as_bytes())
                                .map_err(|e| DynLoadError::SymbolNotFound {
                                    symbol: stringify!($name),
                                    lib: lib_name,
                                    source: e,
                                })?
                        },
                    )*
                    _lib: lib,
                })
            }
        }

        static $static: OnceLock<Result<$Struct, DynLoadError>> = OnceLock::new();

        fn $init_fn() -> Result<&'static $Struct, &'static DynLoadError> {
            $static.get_or_init(|| $Struct::load()).as_ref()
        }

        /// Returns `true` if the library can be loaded on this system.
        pub fn $is_available_fn() -> bool {
            $init_fn().is_ok()
        }

        /// Returns the load error, if the library failed to load.
        pub fn $load_error_fn() -> Option<&'static DynLoadError> {
            $init_fn().err()
        }

        $(
            #[inline]
            pub unsafe fn $name( $($arg : $ty),* ) -> $ret {
                // Compile-time check: load_error_value must be compatible
                // with this function's return type.
                const _: fn() -> $ret = || $load_error_value;
                match $init_fn() {
                    Ok(api) => (api.$name)( $($arg),* ),
                    Err(_) => $load_error_value,
                }
            }
        )*
    };
}

// ---------------------------------------------------------------------------
// CUDA Driver API
// ---------------------------------------------------------------------------

dynamic_api! {
    static CUDA_DRIVER: CudaDriverApi = CUDA_LIB_NAME;
    init_fn: cuda_driver;
    available_fn: is_cuda_driver_available;
    load_error_fn: cuda_driver_load_error;
    load_error_value: cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;

    // ── Initialization ─────────────────────────────────────────────────
    fn cuInit(flags: c_uint) -> CUresult;

    // ── Error handling ─────────────────────────────────────────────────
    fn cuGetErrorName(error: CUresult, p_str: *mut *const c_char) -> CUresult;
    fn cuGetErrorString(error: CUresult, p_str: *mut *const c_char) -> CUresult;

    // ── Device management ──────────────────────────────────────────────
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    fn cuDeviceGetAttribute(
        pi: *mut c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult;
    fn cuDeviceGetName(
        name: *mut c_char,
        len: c_int,
        dev: CUdevice,
    ) -> CUresult;
    fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;

    // ── Primary context ────────────────────────────────────────────────
    fn cuDevicePrimaryCtxRetain(
        pctx: *mut CUcontext,
        dev: CUdevice,
    ) -> CUresult;
    fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;

    // ── Context management ─────────────────────────────────────────────
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    fn cuCtxSetFlags(flags: c_uint) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult;

    // ── Stream management ──────────────────────────────────────────────
    fn cuStreamCreate(ph_stream: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamDestroy_v2(h_stream: CUstream) -> CUresult;
    fn cuStreamSynchronize(h_stream: CUstream) -> CUresult;
    fn cuStreamWaitEvent(
        h_stream: CUstream,
        h_event: CUevent,
        flags: c_uint,
    ) -> CUresult;
    fn cuStreamAttachMemAsync(
        h_stream: CUstream,
        dptr: CUdeviceptr,
        length: usize,
        flags: c_uint,
    ) -> CUresult;
    fn cuStreamBeginCapture_v2(
        h_stream: CUstream,
        mode: CUstreamCaptureMode,
    ) -> CUresult;
    fn cuStreamEndCapture(
        h_stream: CUstream,
        ph_graph: *mut CUgraph,
    ) -> CUresult;
    fn cuStreamIsCapturing(
        h_stream: CUstream,
        capture_status: *mut CUstreamCaptureStatus,
    ) -> CUresult;
    fn cuLaunchHostFunc(
        h_stream: CUstream,
        fn_: CUhostFn,
        user_data: *mut c_void,
    ) -> CUresult;

    // ── Event management ───────────────────────────────────────────────
    fn cuEventCreate(ph_event: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventDestroy_v2(h_event: CUevent) -> CUresult;
    fn cuEventRecord(h_event: CUevent, h_stream: CUstream) -> CUresult;
    fn cuEventSynchronize(h_event: CUevent) -> CUresult;
    fn cuEventQuery(h_event: CUevent) -> CUresult;
    fn cuEventElapsedTime_v2(
        p_milliseconds: *mut f32,
        h_start: CUevent,
        h_end: CUevent,
    ) -> CUresult;

    // ── Module management ──────────────────────────────────────────────
    fn cuModuleLoad(
        module: *mut CUmodule,
        fname: *const c_char,
    ) -> CUresult;
    fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const c_void,
    ) -> CUresult;
    fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;

    // ── Kernel launch ──────────────────────────────────────────────────
    fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        h_stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    // ── Function attributes ────────────────────────────────────────────
    fn cuFuncSetAttribute(
        hfunc: CUfunction,
        attrib: CUfunction_attribute,
        value: c_int,
    ) -> CUresult;
    fn cuFuncSetCacheConfig(
        hfunc: CUfunction,
        config: CUfunc_cache,
    ) -> CUresult;

    // ── Memory management ──────────────────────────────────────────────
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemAllocAsync(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        h_stream: CUstream,
    ) -> CUresult;
    fn cuMemAllocManaged(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        flags: c_uint,
    ) -> CUresult;
    fn cuMemHostAlloc(
        pp: *mut *mut c_void,
        bytesize: usize,
        flags: c_uint,
    ) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemFreeAsync(dptr: CUdeviceptr, h_stream: CUstream) -> CUresult;
    fn cuMemFreeHost(p: *mut c_void) -> CUresult;

    // ── Memory transfer ────────────────────────────────────────────────
    fn cuMemcpyHtoD_v2(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        byte_count: usize,
    ) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(
        dst_device: CUdeviceptr,
        src_host: *const c_void,
        byte_count: usize,
        h_stream: CUstream,
    ) -> CUresult;
    fn cuMemcpyDtoH_v2(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult;
    fn cuMemcpyDtoHAsync_v2(
        dst_host: *mut c_void,
        src_device: CUdeviceptr,
        byte_count: usize,
        h_stream: CUstream,
    ) -> CUresult;
    fn cuMemcpyDtoD_v2(
        dst_device: CUdeviceptr,
        src_device: CUdeviceptr,
        byte_count: usize,
    ) -> CUresult;
    fn cuMemcpyDtoDAsync_v2(
        dst_device: CUdeviceptr,
        src_device: CUdeviceptr,
        byte_count: usize,
        h_stream: CUstream,
    ) -> CUresult;
    fn cuMemsetD8_v2(
        dst_device: CUdeviceptr,
        uc: c_uchar,
        n: usize,
    ) -> CUresult;
    fn cuMemsetD8Async(
        dst_device: CUdeviceptr,
        uc: c_uchar,
        n: usize,
        h_stream: CUstream,
    ) -> CUresult;
    fn cuMemAdvise_v2(
        dev_ptr: CUdeviceptr,
        count: usize,
        advice: CUmem_advise,
        location: CUmemLocation,
    ) -> CUresult;
    fn cuMemPrefetchAsync_v2(
        dev_ptr: CUdeviceptr,
        count: usize,
        dst_location: CUmemLocation,
        flags: c_uint,
        h_stream: CUstream,
    ) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

    // ── Occupancy ──────────────────────────────────────────────────────
    fn cuOccupancyAvailableDynamicSMemPerBlock(
        dynamic_smem_size: *mut usize,
        func: CUfunction,
        num_blocks: c_int,
        block_size: c_int,
    ) -> CUresult;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        num_blocks: *mut c_int,
        func: CUfunction,
        block_size: c_int,
        dynamic_smem_size: usize,
        flags: c_uint,
    ) -> CUresult;
    fn cuOccupancyMaxPotentialBlockSizeWithFlags(
        min_grid_size: *mut c_int,
        block_size: *mut c_int,
        func: CUfunction,
        block_size_to_dynamic_smem_size: CUoccupancyB2DSize,
        dynamic_smem_size: usize,
        block_size_limit: c_int,
        flags: c_uint,
    ) -> CUresult;
    fn cuOccupancyMaxActiveClusters(
        num_clusters: *mut c_int,
        func: CUfunction,
        config: *const CUlaunchConfig,
    ) -> CUresult;
    fn cuOccupancyMaxPotentialClusterSize(
        cluster_size: *mut c_int,
        func: CUfunction,
        config: *const CUlaunchConfig,
    ) -> CUresult;

    // ── CUDA Graphs ────────────────────────────────────────────────────
    fn cuGraphInstantiateWithFlags(
        ph_graph_exec: *mut CUgraphExec,
        h_graph: CUgraph,
        flags: c_ulonglong,
    ) -> CUresult;
    fn cuGraphUpload(h_graph_exec: CUgraphExec, h_stream: CUstream) -> CUresult;
    fn cuGraphLaunch(h_graph_exec: CUgraphExec, h_stream: CUstream) -> CUresult;
    fn cuGraphDestroy(h_graph: CUgraph) -> CUresult;
    fn cuGraphExecDestroy(h_graph_exec: CUgraphExec) -> CUresult;
}

// ---------------------------------------------------------------------------
// cuRAND API
// ---------------------------------------------------------------------------

dynamic_api! {
    static CURAND: CurandApi = CURAND_LIB_NAME;
    init_fn: curand_api;
    available_fn: is_curand_available;
    load_error_fn: curand_load_error;
    load_error_value: curandStatus_CURAND_STATUS_NOT_INITIALIZED;

    fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;
    fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;
    fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: c_ulonglong,
    ) -> curandStatus_t;
    fn curandGenerateNormal(
        generator: curandGenerator_t,
        output_ptr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;
    fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        output_ptr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> curandStatus_t;
    fn curandGenerateUniform(
        generator: curandGenerator_t,
        output_ptr: *mut f32,
        num: usize,
    ) -> curandStatus_t;
    fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        output_ptr: *mut f64,
        num: usize,
    ) -> curandStatus_t;
}
