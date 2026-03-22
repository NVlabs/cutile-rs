/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! High-level wrappers around CUDA driver API functions.
//!
//! Provides safe(r) Rust interfaces for initialization, kernel launch, memory
//! operations, device queries, and random number generation.

pub use cuda_bindings as sys;
use cuda_bindings::{
    cuDeviceGetAttribute, CUdevice, CUdevice_attribute,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
};
use std::ffi::{c_int, c_uint, c_void};
use std::mem::{self, MaybeUninit};
use std::sync::Arc;

use crate::error::*;
use crate::CudaStream;

/// Initializes the CUDA driver API. Must be called before any other driver call.
///
/// # Safety
/// Caller must ensure CUDA is available and `flags` is valid (typically `0`).
pub unsafe fn init(flags: c_uint) -> Result<(), DriverError> {
    cuda_bindings::cuInit(flags).result()
}

/// Returns the API version associated with the given CUDA context.
///
/// # Safety
/// `ctx` must be a valid CUDA context handle.
pub unsafe fn api_version(ctx: cuda_bindings::CUcontext) -> c_uint {
    let mut api_version = 0 as c_uint;
    unsafe { cuda_bindings::cuCtxGetApiVersion(ctx, &mut api_version) };
    api_version
}

/// Launches a CUDA kernel with the given grid/block dimensions and parameters.
///
/// # Safety
/// `f`, `stream`, and all pointers in `kernel_params` must be valid.
#[inline]
pub unsafe fn launch_kernel(
    f: cuda_bindings::CUfunction,
    grid_dim: (c_uint, c_uint, c_uint),
    block_dim: (c_uint, c_uint, c_uint),
    shared_mem_bytes: c_uint,
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut c_void],
) -> Result<(), DriverError> {
    cuda_bindings::cuLaunchKernel(
        f,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        kernel_params.as_mut_ptr(),
        std::ptr::null_mut(),
    )
    .result()
}

/// Asynchronously allocates `num_bytes` of device memory on the given stream.
///
/// # Safety
/// `stream` must be a valid, non-destroyed CUDA stream.
pub unsafe fn malloc_async(num_bytes: usize, stream: &Arc<CudaStream>) -> sys::CUdeviceptr {
    crate::memory::malloc_async(stream.cu_stream(), num_bytes).expect("Malloc async failed.")
}

/// Asynchronously allocates `num_bytes` of device memory from a specific pool.
///
/// # Safety
/// `stream` must be a valid cuda stream. `pool` must be a valid memory pool handle.
pub unsafe fn malloc_from_pool_async(
    num_bytes: usize,
    stream: &Arc<CudaStream>,
    pool: sys::CUmemoryPool,
) -> sys::CUdeviceptr {
    crate::memory::malloc_from_pool_async(stream.cu_stream(), num_bytes, pool)
        .expect("Malloc from pool async failed.")
}

/// Asynchronously frees device memory on the given stream.
///
/// # Safety
/// `dptr` must have been allocated with `malloc_async` and must not be used after this call.
pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: &Arc<CudaStream>) {
    crate::memory::free_async(dptr, stream.cu_stream()).expect("Free async failed.")
}

/// Asynchronously copies `num_elements` of type `T` from host to device memory.
///
/// # Safety
/// `src` must point to at least `num_elements` valid elements; `dst` must have sufficient capacity.
pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: *const T,
    num_elements: usize,
    stream: &Arc<CudaStream>,
) {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe { crate::memory::memcpy_htod_async(dst, src, num_bytes, stream.cu_stream()) }
        .expect("memcpy_htod_async failed.")
}

/// Asynchronously copies `num_elements` of type `T` from device to host memory.
///
/// # Safety
/// `dst` must point to at least `num_elements` writable elements; `src` must be valid device memory.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: *mut T,
    src: sys::CUdeviceptr,
    num_elements: usize,
    stream: &Arc<CudaStream>,
) {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe { crate::memory::memcpy_dtoh_async(dst, src, num_bytes, stream.cu_stream()) }
        .expect("memcpy_dtoh_async failed.")
}

/// Asynchronously copies `num_elements` of type `T` between device memory regions.
///
/// # Safety
/// Both `dst` and `src` must be valid device pointers with sufficient capacity.
pub unsafe fn memcpy_dtod_async<T>(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_elements: usize,
    stream: &Arc<CudaStream>,
) {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe { crate::memory::memcpy_dtod_async(dst, src, num_bytes, stream.cu_stream()) }
        .expect("memcpy_dtod_async failed.")
}

/// Wrappers around the cuRAND random number generation library.
pub mod curand {
    // TODO (hme): Probably move this into its own file at some point.

    use cuda_bindings::{
        curandCreateGenerator, curandDestroyGenerator, curandGenerateNormal,
        curandGenerateNormalDouble, curandGenerateUniform, curandGenerateUniformDouble,
        curandGenerator_t, curandRngType_CURAND_RNG_PSEUDO_DEFAULT,
        curandSetPseudoRandomGeneratorSeed, CUdeviceptr,
    };
    use std::ffi::c_ulonglong;
    use std::mem::MaybeUninit;

    /// Creates a new pseudo-random number generator with default RNG type.
    ///
    /// # Safety
    /// cuRAND library must be available.
    pub unsafe fn get_rng() -> curandGenerator_t {
        let mut curand_gen_uninited: MaybeUninit<curandGenerator_t> = MaybeUninit::uninit();
        let curand_rng_type = curandRngType_CURAND_RNG_PSEUDO_DEFAULT;
        assert!(curandCreateGenerator(curand_gen_uninited.as_mut_ptr(), curand_rng_type) == 0);
        curand_gen_uninited.assume_init()
    }

    /// Sets the seed for a pseudo-random number generator.
    ///
    /// # Safety
    /// `gen` must be a valid cuRAND generator handle.
    pub unsafe fn set_seed(gen: curandGenerator_t, seed: u64) {
        assert!(curandSetPseudoRandomGeneratorSeed(gen, c_ulonglong::from(seed)) == 0);
    }

    /// Generates normally distributed `f32` values into device memory.
    ///
    /// # Safety
    /// `dptr` must be valid device memory with capacity for `num_elements` floats.
    pub unsafe fn generate_normal_f32(
        curand_gen: curandGenerator_t,
        dptr: CUdeviceptr,
        num_elements: usize,
        mean: f32,
        std: f32,
    ) {
        assert!(curandGenerateNormal(curand_gen, dptr as *mut f32, num_elements, mean, std) == 0);
    }

    /// RAII wrapper around a cuRAND pseudo-random number generator.
    pub struct RNG {
        curand_gen: curandGenerator_t,
    }

    impl RNG {
        /// Creates a new RNG, optionally seeded.
        ///
        /// # Safety
        /// cuRAND library must be available.
        pub unsafe fn new(seed: Option<u64>) -> Self {
            let curand_gen = get_rng();
            if let Some(seed) = seed {
                set_seed(curand_gen, seed);
            }
            Self { curand_gen }
        }

        /// Generates normally distributed `f32` values into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` floats.
        pub unsafe fn generate_normal_f32(
            &self,
            dptr: CUdeviceptr,
            num_elements: usize,
            mean: f32,
            std: f32,
        ) {
            assert!(
                curandGenerateNormal(self.curand_gen, dptr as *mut f32, num_elements, mean, std)
                    == 0
            );
        }

        /// Generates normally distributed `f64` values into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` doubles.
        pub unsafe fn generate_normal_f64(
            &self,
            dptr: CUdeviceptr,
            num_elements: usize,
            mean: f64,
            std: f64,
        ) {
            assert!(
                curandGenerateNormalDouble(
                    self.curand_gen,
                    dptr as *mut f64,
                    num_elements,
                    mean,
                    std
                ) == 0
            );
        }

        /// Generates uniformly distributed `f32` values in `[0, 1)` into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` floats.
        pub unsafe fn generate_uniform_f32(&self, dptr: CUdeviceptr, num_elements: usize) {
            assert!(curandGenerateUniform(self.curand_gen, dptr as *mut f32, num_elements) == 0);
        }

        /// Generates uniformly distributed `f64` values in `[0, 1)` into device memory.
        ///
        /// # Safety
        /// `dptr` must be valid device memory with capacity for `num_elements` doubles.
        pub unsafe fn generate_uniform_f64(&self, dptr: CUdeviceptr, num_elements: usize) {
            assert!(
                curandGenerateUniformDouble(self.curand_gen, dptr as *mut f64, num_elements) == 0
            );
        }
    }

    impl Drop for RNG {
        fn drop(&mut self) {
            unsafe { assert!(curandDestroyGenerator(self.curand_gen) == 0) };
        }
    }
}

/// Queries a device attribute value for the given device.
///
/// # Safety
/// `device` must be a valid CUDA device handle.
pub unsafe fn get_device_attribute(
    device: CUdevice,
    device_attr: CUdevice_attribute,
) -> Result<i32, DriverError> {
    let mut result: MaybeUninit<c_int> = MaybeUninit::uninit();
    assert!(cuDeviceGetAttribute(result.as_mut_ptr(), device_attr, device) == 0);
    Ok(result.assume_init())
}

/// Returns the device clock rate in MHz.
///
/// # Safety
/// `device` must be a valid CUDA device handle.
pub unsafe fn get_device_clock_rate_mhz(device: CUdevice) -> Result<f64, DriverError> {
    let result = get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    )?;
    Ok(f64::from(result) * 1e-3)
}

/// Returns the device clock rate in kHz.
///
/// # Safety
/// `device` must be a valid CUDA device handle.
pub unsafe fn get_device_clock_rate(device: CUdevice) -> Result<i32, DriverError> {
    get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    )
}

/// Returns the number of multiprocessors on the device.
///
/// # Safety
/// `device` must be a valid CUDA device handle.
pub unsafe fn get_device_multiprocessor_count(device: CUdevice) -> Result<i32, DriverError> {
    get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
}
