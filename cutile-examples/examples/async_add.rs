/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::{DeviceOperation, IntoDeviceOperation};
use cutile::api::{arange, ones, zeros};
use cutile::tensor::ToHostVec;
use cutile::tile_kernel::IntoDeviceOperationPartition;
use my_module::{add_apply, add_async};
use std::future::IntoFuture;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry(print_ir = true)]
    fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tile_x = load_tile_like_1d(x, z);
        let tile_y = load_tile_like_1d(y, z);
        z.store(tile_x + tile_y);
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), cuda_async::error::DeviceError> {
    // Operations in api::default::{*} will use the default global device, which is set to 0.
    // Create tensors.
    let len = 2usize.pow(5);
    let z = zeros([len]);
    let x = arange(len);
    let y = ones([len]);
    // Use function calling convention:
    let add_op_1 = add_async(z.partition([2]), x.arc(), y.arc());
    // or chain the invocation of the kernel via the DeviceOperation apply method:
    let add_op_2 = add_op_1.apply(add_apply);
    // Schedule the operation using the default scheduler.
    // This converts the DeviceOperation into a DeviceFuture.
    let fut = add_op_2.into_future();
    // We can spawn:
    let (z, x, y) = tokio::spawn(fut)
        .await
        .expect("Failed to execute tokio task.")?;
    // We can convert this back into a device operation.
    let args = (z, x, y).device_operation();
    // We can directly await:
    let (z, x, y) = args.apply(add_apply).await?;
    // Check output.
    let x_host = x.to_host_vec().await?;
    let y_host = y.to_host_vec().await?;
    let z_host = z.unpartition().to_host_vec().await?;
    for i in 0..z_host.len() {
        let x_i: f32 = x_host[i];
        let y_i: f32 = y_host[i];
        let answer = x_i + y_i;
        println!("{} + {} = {}", x_i, y_i, z_host[i]);
        assert_eq!(answer, z_host[i], "{} != {} ?", answer, z_host[i]);
    }
    Ok(())
}
