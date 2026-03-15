/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{device, get_device_sm_name, init};
use cuda_tile_rs::cuda_tile::ModuleOperation;
use cuda_tile_rs::{cuda_tile, cuda_tile_write_bytecode, operation_parse};
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{Block, BlockLike, Location, Region, RegionLike};
use melior::Context;
use std::env;
use std::process::Command;
use uuid::Uuid;

pub fn get_gpu_name(device_id: usize) -> String {
    unsafe { init(0) }.expect("failed to initialize CUDA driver");
    let dev = device::get(device_id as i32).expect("failed to get CUDA device");
    unsafe { get_device_sm_name(dev) }.expect("failed to get SM name")
}

pub fn parse_tile_entry<'c>(
    context: &'c Context,
    module_name: &str,
    entry: &str,
) -> ModuleOperation<'c> {
    let location = Location::unknown(&context);
    let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location)
        .body({
            let entry_op = operation_parse(&context, entry, None).unwrap();
            let module_block = Block::new(&[]);
            module_block.append_operation(entry_op);

            let region = Region::new();
            region.append_block(module_block);
            region
        })
        .sym_name(StringAttribute::new(&context, module_name))
        .build();
    assert!(module_op.as_operation().verify());
    return module_op;
}

pub fn compile_module(module_op: &ModuleOperation, gpu_name: &str) -> String {
    let tmp_dir = env::temp_dir();
    let base_filename = tmp_dir.join(Uuid::new_v4().to_string());
    let bc_filename = format!("{}.bc", base_filename.to_str().unwrap());
    let cubin_filename = format!("{}.cubin", base_filename.to_str().unwrap());

    let res = cuda_tile_write_bytecode(&module_op, bc_filename.as_str());
    assert!(res.is_ok());
    let _ = Command::new("tileiras")
        .arg("--gpu-name")
        .arg(gpu_name)
        .arg("--opt-level")
        .arg("3")
        .arg("-o")
        .arg(&cubin_filename)
        .arg(&bc_filename)
        .output()
        .expect(format!("Failed to compile bytecode {bc_filename}").as_str());
    cubin_filename
}

#[cfg(test)]
mod test {
    use cuda_tile_rs::cuda_tile::ModuleOperation;
    use melior::Context;
    use cuda_core::sys::cuDriverGetVersion;
    use cuda_core::{
        api_version, ctx, device, init, launch_kernel, module, primary_ctx, stream, DriverError,
    };
    use std::ffi::{c_int, CString};
    use std::fs;
    use std::mem::MaybeUninit;

    use crate::context_all;
    use crate::cuda_tile_runtime_utils::{compile_module, get_gpu_name, parse_tile_entry};

    fn get_test_module<'c>(context: &'c Context) -> ModuleOperation<'c> {
        const HELLO_TILE_BLOCK_MLIR: &'static str = r#"
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
    fn test_load_cubin_file() -> () {
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
    fn test_load_cubin_data() -> () {
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
}
