// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

//! Raw FFI bindings to the CUDA toolkit libraries (CUDA driver API, cuRAND, etc.).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::env;

pub fn cuda_toolkit_dir() -> String {
    env::var("CUDA_TOOLKIT_PATH").expect("CUDA_TOOLKIT_PATH is required but not set")
}

#[cfg(test)]
mod cuda_tests {
    use super::*;
    use std::ffi::{c_int, c_ulonglong};
    use std::mem::MaybeUninit;

    fn init() -> (CUdevice, CUcontext) {
        unsafe {
            let init_res = crate::cuInit(0);
            assert_eq!(init_res, 0, "init failed");

            let mut dev: MaybeUninit<crate::CUdevice> = MaybeUninit::uninit();
            let dev_result = crate::cuDeviceGet(dev.as_mut_ptr(), 0 as c_int);
            assert_eq!(dev_result, 0, "get device failed");
            let dev = dev.assume_init();

            let mut ctx = MaybeUninit::uninit();
            let ctx_res = crate::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev);
            assert_eq!(ctx_res, 0, "retain context failed");
            let ctx = ctx.assume_init();
            assert_eq!(
                crate::cuCtxSetCurrent(ctx),
                0,
                "failed to set current context"
            );
            (dev, ctx)
        }
    }

    unsafe fn get_dptr(bytesize: usize) -> CUdeviceptr {
        let mut dptr: MaybeUninit<CUdeviceptr> = MaybeUninit::uninit();
        assert!(cuMemAlloc_v2(dptr.as_mut_ptr(), bytesize) == 0);
        dptr.assume_init()
    }

    unsafe fn get_rng() -> curandGenerator_t {
        let mut curand_gen_uninited: MaybeUninit<curandGenerator_t> = MaybeUninit::uninit();
        let curand_rng_type = curandRngType_CURAND_RNG_PSEUDO_DEFAULT;
        assert!(curandCreateGenerator(curand_gen_uninited.as_mut_ptr(), curand_rng_type) == 0);
        curand_gen_uninited.assume_init()
    }

    unsafe fn set_seed(gen: curandGenerator_t, seed: u64) {
        assert!(curandSetPseudoRandomGeneratorSeed(gen, c_ulonglong::from(seed)) == 0);
    }

    #[test]
    fn test_curand() {
        unsafe {
            let (_dev, _ctx) = init();
            let curand_gen = get_rng();
            set_seed(curand_gen, 123);
            let num_elements = 32;
            let bytesize = num_elements * size_of::<f32>();
            let dptr = get_dptr(bytesize);
            assert!(
                curandGenerateNormal(curand_gen, dptr as *mut f32, num_elements, 0.0, 1.0) == 0
            );
            assert!(curandDestroyGenerator(curand_gen) == 0);
            assert!(cuMemFree_v2(dptr) == 0);
        }
    }
}
