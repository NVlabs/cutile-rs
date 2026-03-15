/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub use cuda_bindings as sys;
use cuda_bindings::{
    cuDeviceGetAttribute, CUdevice, CUdevice_attribute,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
};
use std::ffi::{c_int, c_uint, c_void};
use std::mem::{self, MaybeUninit};
use std::sync::Arc;

use crate::error::*;
use crate::CudaStream;

pub unsafe fn init(flags: c_uint) -> Result<(), DriverError> {
    cuda_bindings::cuInit(flags).result()
}

pub unsafe fn api_version(ctx: cuda_bindings::CUcontext) -> c_uint {
    let mut api_version = 0 as c_uint;
    unsafe { cuda_bindings::cuCtxGetApiVersion(ctx, &mut api_version) };
    api_version
}

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

pub unsafe fn malloc_async(num_bytes: usize, stream: &Arc<CudaStream>) -> sys::CUdeviceptr {
    crate::memory::malloc_async(stream.cu_stream(), num_bytes).expect("Malloc async failed.")
}

pub unsafe fn free_async(dptr: sys::CUdeviceptr, stream: &Arc<CudaStream>) -> () {
    crate::memory::free_async(dptr, stream.cu_stream()).expect("Free async failed.")
}

pub unsafe fn memcpy_htod_async<T>(
    dst: sys::CUdeviceptr,
    src: *const T,
    num_elements: usize,
    stream: &Arc<CudaStream>,
) -> () {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe { crate::memory::memcpy_htod_async(dst, src, num_bytes, stream.cu_stream()) }
        .expect("memcpy_htod_async failed.")
}

pub unsafe fn memcpy_dtoh_async<T>(
    dst: *mut T,
    src: sys::CUdeviceptr,
    num_elements: usize,
    stream: &Arc<CudaStream>,
) -> () {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe { crate::memory::memcpy_dtoh_async(dst, src, num_bytes, stream.cu_stream()) }
        .expect("memcpy_dtoh_async failed.")
}

pub unsafe fn memcpy_dtod_async<T>(
    dst: sys::CUdeviceptr,
    src: sys::CUdeviceptr,
    num_elements: usize,
    stream: &Arc<CudaStream>,
) -> () {
    let num_bytes = num_elements * mem::size_of::<T>();
    unsafe { crate::memory::memcpy_dtod_async(dst, src, num_bytes, stream.cu_stream()) }
        .expect("memcpy_dtod_async failed.")
}

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

    pub unsafe fn get_rng() -> curandGenerator_t {
        let mut curand_gen_uninited: MaybeUninit<curandGenerator_t> = MaybeUninit::uninit();
        let curand_rng_type = curandRngType_CURAND_RNG_PSEUDO_DEFAULT;
        assert!(curandCreateGenerator(curand_gen_uninited.as_mut_ptr(), curand_rng_type) == 0);
        curand_gen_uninited.assume_init()
    }

    pub unsafe fn set_seed(gen: curandGenerator_t, seed: u64) {
        assert!(curandSetPseudoRandomGeneratorSeed(gen, c_ulonglong::from(seed)) == 0);
    }

    pub unsafe fn generate_normal_f32(
        curand_gen: curandGenerator_t,
        dptr: CUdeviceptr,
        num_elements: usize,
        mean: f32,
        std: f32,
    ) {
        assert!(curandGenerateNormal(curand_gen, dptr as *mut f32, num_elements, mean, std) == 0);
    }

    pub struct RNG {
        curand_gen: curandGenerator_t,
    }

    impl RNG {
        pub unsafe fn new(seed: Option<u64>) -> Self {
            let curand_gen = get_rng();
            if let Some(seed) = seed {
                set_seed(curand_gen, seed);
            }
            Self { curand_gen }
        }

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

        pub unsafe fn generate_uniform_f32(&self, dptr: CUdeviceptr, num_elements: usize) {
            assert!(curandGenerateUniform(self.curand_gen, dptr as *mut f32, num_elements) == 0);
        }

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

pub unsafe fn get_device_attribute(
    device: CUdevice,
    device_attr: CUdevice_attribute,
) -> Result<i32, DriverError> {
    let mut result: MaybeUninit<c_int> = MaybeUninit::uninit();
    assert!(cuDeviceGetAttribute(result.as_mut_ptr(), device_attr, device) == 0);
    Ok(result.assume_init())
}

pub unsafe fn get_device_clock_rate_mhz(device: CUdevice) -> Result<f64, DriverError> {
    let result = get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    )?;
    Ok(f64::from(result) * 1e-3)
}

pub unsafe fn get_device_clock_rate(device: CUdevice) -> Result<i32, DriverError> {
    get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
    )
}

pub unsafe fn get_device_multiprocessor_count(device: CUdevice) -> Result<i32, DriverError> {
    get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
}

pub unsafe fn get_device_sm_name(device: CUdevice) -> Result<String, DriverError> {
    let major = get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let minor = get_device_attribute(
        device,
        CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;
    Ok(format!("sm_{major}{minor}"))
}
