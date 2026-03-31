/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::{DeviceOperation, IntoDeviceOperation, Zippable};
use cutile;
use cutile::api::{arange, DeviceOperationReshape};
use cutile::error::Error;
use cutile::tensor::ToHostVec;
use cutile::tile_kernel::global_policy;
use cutile::tile_kernel::IntoDeviceOperationPartition;
use cutile::DType;
use std::fmt::Display;
use std::ops::{Add, Mul};

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2], T: ElementType>(
        a: T,
        x: &Tensor<T, { [-1, -1] }>,
        y: &mut Tensor<T, S>,
    ) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = load_tile_like_2d(x, y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

use my_module::*;

async fn execute<T: DType + Display + PartialEq + Mul<Output = T> + Add<Output = T>>(
    size: usize,
) -> Result<(), Error> {
    let a = (<T as DType>::one() + <T as DType>::one()).device_operation();
    let x = arange(size).reshape([4, 8]);
    let y = arange(size).reshape([4, 8]);
    let saxpy_op = (a, x.arc(), y.partition([2, 4])).zip().apply(saxpy_apply);
    // Convert the saxpy DeviceOperation into a DeviceFuture.
    let saxpy_future = saxpy_op.schedule(global_policy(0)?.as_scheduling_policy()?)?;
    // Spawn a tokio task to execute the saxpy future.
    let (a, _x, y) = saxpy_future.await?;
    let y_host: Vec<T> = y.unpartition().to_host_vec().await?;
    let input_host: Vec<T> = arange(size).await?.to_host_vec().await?;
    for i in 0..input_host.len() {
        let x_i: T = input_host[i];
        let y_i: T = input_host[i];
        let answer = a * x_i + y_i;
        println!("{} * {} + {} = {}", a, x_i, y_i, y_host[i]);
        assert_eq!(answer, y_host[i], "{} != {} ?", answer, y_host[i]);
    }
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    // execute::<i64>(2usize.pow(5)).await;
    execute::<f32>(2usize.pow(5)).await?;
    execute::<f64>(2usize.pow(5)).await?;
    Ok(())
}
