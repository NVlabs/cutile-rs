/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Manually use the async API to compile and launch the saxpy kernel using the cutile API.
 */

use cuda_async::device_operation::{value, with_context, DeviceOp};
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;
use cutile;
use cutile::api::arange;
use cutile::api::DeviceOpReshape;
use cutile::error::Error;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::global_policy;
use cutile::tile_kernel::{compile_from_context, CompileOptions};
use my_module::_module_asts;
use std::sync::Arc;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(y: &mut Tensor<f32, S>, a: f32, x: &Tensor<f32, { [-1, -1] }>) {
        let tile_a = broadcast_scalar(a, y.shape());
        let tile_x = load_tile_like_2d(x, y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    let policy = global_policy(0)?;
    let num_elements: usize = 2usize.pow(5);
    let strides = &[8, 1];
    let y_partition_shape = [2, 4];
    let y_partition_strides = strides;
    let function_generics: Vec<String> = y_partition_shape.iter().map(|x| x.to_string()).collect();
    let stride_args: Vec<(String, Vec<i32>)> = vec![
        ("y".to_string(), y_partition_strides.to_vec()),
        ("x".to_string(), strides.to_vec()),
    ];

    // We can start compiling the function while we fetch input.
    let compilation_task = tokio::spawn(async move {
        with_context(|ctx| {
            let func = compile_from_context(
                ctx,
                _module_asts,
                "my_module",
                "saxpy",
                "saxpy_entry",
                function_generics,
                stride_args,
                vec![],
                None,
                CompileOptions::default(),
                my_module::__SOURCE_HASH,
            );
            value(func)
        })
        .schedule(&global_policy(0)?)?
        .await
    });
    // Spawn allocation of tensor 1 and 2.
    let a: f32 = 2.0;
    let x = arange::<f32>(num_elements)
        .reshape(&[4, 8])
        .schedule(&policy)?
        .await?;
    let y = arange::<f32>(num_elements)
        .reshape(&[4, 8])
        .schedule(&policy)?
        .await?;
    // We need the function to build the launcher, so we wait on compilation.
    let (function, _) = compilation_task
        .await
        .expect("Failed to compile module.")??;

    // Launch the kernel.
    let cuda_async_op = async move {
        let y_part = y.partition(y_partition_shape);
        let mut launcher = AsyncKernelLaunch::new(function.clone());
        let x_arc: Arc<_> = x.into();
        launcher
            .push_arg(&y_part)
            .push_arg(a)
            .push_arg_arc(&x_arc)
            .set_launch_config(LaunchConfig {
                grid_dim: y_part.grid().expect("Invalid grid."),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            });
        launcher.await.expect("Kernel launch failed.");
        y_part.unpartition()
    };
    let res = tokio::spawn(cuda_async_op).await;
    assert!(res.is_ok());
    let y = res.unwrap();

    // Check output.
    let y_host = y.to_host_vec().await?;
    let input_host: Vec<f32> = arange(num_elements).await?.to_host_vec().await?;
    println!("{:?}", y_host);
    for i in 0..num_elements {
        let x_i = input_host[i];
        let y_i = input_host[i];
        let answer = a * x_i + y_i;
        println!("{} * {} + {} = {}", a, x_i, y_i, y_host[i]);
        assert_eq!(answer, y_host[i], "{} != {} ?", answer, y_host[i]);
    }
    Ok(())
}
