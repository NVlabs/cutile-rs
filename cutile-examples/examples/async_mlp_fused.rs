/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_context;
use cuda_async::device_context::global_policy;
use cuda_async::device_operation::*;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_async::scheduling_policies::WithDeviceId;
use cuda_core::LaunchConfig;
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::IntoDeviceOperationPartition;
use cutile::{api, error::Error};
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile::ModuleOperation;
use cutile_compiler::cuda_tile_runtime_utils::{compile_module, get_gpu_name};
use std::sync::Arc;

#[cutile::module]
pub mod my_kernels {

    use cutile::core::*;

    fn relu<const D: i32>(input: Tile<f32, { [D] }>) -> Tile<f32, { [D] }> {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, const_shape![D]);
        max_tile(zero_tile, input)
    }

    #[cutile::entry()]
    fn fused_mlp<const BM: i32, const BN: i32, const BK: i32, const N: i32, const K: i32>(
        out: &mut Tensor<f32, { [BM] }>,
        data: &Tensor<f32, { [-1, -1] }>,
        w0: &Tensor<f32, { [-1, K] }>,
        w1: &Tensor<f32, { [K] }>,
    ) {
        let part_data = data.partition(const_shape![BM, BN]);
        let part_w0 = w0.partition(const_shape![BN, BK]);
        let part_w1 = w1.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let m = pid.0;
        let mut tile_out = out.load().reshape(const_shape![BM, 1]);
        for k in 0i32..(K / BK) {
            // TODO (hme): Infer type from const.
            let mut tile_data_x_w0: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
            for n in 0i32..(N / BN) {
                let tile_data = part_data.load([m, n]);
                let tile_w0 = part_w0.load([n, k]);
                tile_data_x_w0 = mma(tile_data, tile_w0, tile_data_x_w0);
            }
            let tile_w1 = part_w1.load([k]).reshape(const_shape![BK, 1]);
            tile_out = mma(tile_data_x_w0, tile_w1, tile_out);
        }
        out.store(relu(tile_out.reshape(const_shape![BM])));
    }
}

// Simulate loading input data.
fn load_data<const RANK: usize>(
    batch_size: [usize; RANK],
) -> impl DeviceOperation<Output = Tensor<f32>> {
    api::randn(0.0, 1.0, batch_size)
}

use my_kernels::_module_asts;

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    // Data
    let (m, n, k) = (16, 16, 16);
    let (bm, bn, bk) = (4, 4, 4);
    let data = load_data([m, n]).arc().await?;
    let w0 = api::randn(0.0f32, 1.0, [n, k]).arc().await?; // impl DeviceOperation
    let w1 = api::randn(0.0f32, 1.0, [k]).arc().await?; // impl DeviceOperation
    let out = api::zeros::<1, f32>([m]).partition([bm]).await?;

    // Compilation
    let module_name = "my_kernels";
    let function_name = "fused_mlp";
    let function_entry = "fused_mlp_entry";

    let modules = CUDATileModules::new(_module_asts())?;
    let generics = [
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        n.to_string(),
        k.to_string(),
    ];
    let stride_args = [
        ("out", vec![1]),
        ("data", vec![n as i32, 1]),
        ("w0", vec![k as i32, 1]),
        ("w1", vec![1]),
    ];
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        module_name,
        function_name,
        &generics,
        &stride_args
            .iter()
            .map(|(x, y)| (*x, y.as_slice()))
            .collect::<Vec<_>>(),
        None,
        get_gpu_name(0),
    )?;
    let module_op: ModuleOperation = compiler.compile()?;
    println!("{}", module_op.as_operation());
    let device = global_policy(0)?;
    let module_filename = compile_module(&module_op, &get_gpu_name(device.get_device_id()));
    let module = device_context::load_module_from_file(&module_filename, device.get_device_id())?;
    let function = Arc::new(
        module
            .load_function(function_entry)
            .expect("Failed to compile function."),
    );

    let launch_grid = (4, 1, 1);
    let mut kernel_launch = AsyncKernelLaunch::new(function.clone());
    kernel_launch
        .push_arg(&out)
        .push_arg_arc(&data)
        .push_arg_arc(&w0)
        .push_arg_arc(&w1)
        .set_launch_config(LaunchConfig {
            grid_dim: launch_grid,
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        });
    kernel_launch.await?;
    let host_vec = out.unpartition().to_host_vec().await?;
    println!("{:?}", host_vec);
    Ok(())
}
