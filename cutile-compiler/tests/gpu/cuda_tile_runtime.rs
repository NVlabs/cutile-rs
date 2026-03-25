/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::sys::cuDriverGetVersion;
use cuda_core::{
    api_version, ctx, device, init, launch_kernel, module, primary_ctx, stream, DriverError,
};
use cuda_tile_rs::cuda_tile::ModuleOperation;
use cutile_compiler::context_all;
use cutile_compiler::cuda_tile_runtime_utils::{compile_module, get_gpu_name, parse_tile_entry};
use melior::Context;
use std::ffi::{c_int, CString};
use std::fs;
use std::mem::MaybeUninit;

fn get_test_module<'c>(context: &'c Context) -> ModuleOperation<'c> {
    const HELLO_TILE_BLOCK_MLIR: &str = r#"
        cuda_tile.entry @hello_world_kernel() {
            cuda_tile.print "Hello World From MLIR String!\n"
        }
    "#;
    parse_tile_entry(&context, "my_kernels", HELLO_TILE_BLOCK_MLIR)
}

#[test]
fn test_mlir_to_cubin() {
    let context = context_all();
    let module = get_test_module(&context);
    let cubin_filename = compile_module(&module, &get_gpu_name(0));
    println!("cubin_filename: {}", cubin_filename);
}

#[test]
fn test_load_cubin_file() {
    let context = context_all();
    let module = get_test_module(&context);
    let cubin_filename = compile_module(&module, &get_gpu_name(0));
    let mut driver_version = 0 as c_int;
    unsafe { cuDriverGetVersion(&mut driver_version) };
    println!("Driver version: {driver_version}");
    unsafe {
        let init_res = cuda_core::sys::cuInit(0);
        assert_eq!(init_res, 0, "init failed");

        let mut dev: MaybeUninit<cuda_core::sys::CUdevice> = MaybeUninit::uninit();
        let dev_result = cuda_core::sys::cuDeviceGet(dev.as_mut_ptr(), 0 as c_int);
        assert_eq!(dev_result, 0, "get device failed");
        let dev = dev.assume_init();

        let mut ctx = MaybeUninit::uninit();
        let ctx_res = cuda_core::sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev);
        assert_eq!(ctx_res, 0, "retain context failed");
        let ctx = ctx.assume_init();
        assert_eq!(
            cuda_core::sys::cuCtxSetCurrent(ctx),
            0,
            "failed to set current context"
        );

        let mut module = MaybeUninit::uninit();
        let fname_c_str = CString::new(cubin_filename).unwrap();
        let fname_ptr = fname_c_str.as_c_str().as_ptr();
        let module_res = cuda_core::sys::cuModuleLoad(module.as_mut_ptr(), fname_ptr);
        assert_eq!(module_res, 0, "module load failed");
        let _module = module.assume_init();
    }
}

#[test]
fn test_load_cubin_data() {
    let context = context_all();
    let module = get_test_module(&context);
    let cubin_filename = compile_module(&module, &get_gpu_name(0));
    let mut driver_version = 0 as c_int;
    unsafe { cuDriverGetVersion(&mut driver_version) };
    println!("Driver version: {driver_version}");
    unsafe {
        let init_res = cuda_core::sys::cuInit(0);
        assert_eq!(init_res, 0, "init failed");

        let mut dev: MaybeUninit<cuda_core::sys::CUdevice> = MaybeUninit::uninit();
        let dev_result = cuda_core::sys::cuDeviceGet(dev.as_mut_ptr(), 0 as c_int);
        assert_eq!(dev_result, 0, "get device failed");
        let dev = dev.assume_init();

        let mut ctx = MaybeUninit::uninit();
        let ctx_res = cuda_core::sys::cuDevicePrimaryCtxRetain(ctx.as_mut_ptr(), dev);
        assert_eq!(ctx_res, 0, "retain context failed");
        let ctx = ctx.assume_init();
        assert_eq!(
            cuda_core::sys::cuCtxSetCurrent(ctx),
            0,
            "failed to set current context"
        );

        let mut module = MaybeUninit::uninit();
        let byte_content = fs::read(cubin_filename).unwrap();
        let module_res = cuda_core::sys::cuModuleLoadData(
            module.as_mut_ptr(),
            byte_content.as_ptr() as *const _,
        );
        assert_eq!(module_res, 0, "module load failed");
        let _module = module.assume_init();
    }
}

#[test]
fn test_compile_mlir_str() -> Result<(), DriverError> {
    let context = context_all();
    let module = get_test_module(&context);
    let cubin_filename = compile_module(&module, &get_gpu_name(0));
    unsafe {
        init(0)?;
        let dev = device::get(0)?;
        let ctx = primary_ctx::retain(dev)?;
        ctx::set_current(ctx)?;
        println!("API version: {}", api_version(ctx));
        let module = module::load(&cubin_filename)?;
        let func = module::get_function(module, "hello_world_kernel")?;
        let s = stream::create(stream::StreamKind::NonBlocking)?;
        let _ = launch_kernel(func, (1, 1, 1), (1, 1, 1), 48000, s, &mut []);
        stream::synchronize(s)?;
        Ok(())
    }
}
