/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_async::device_context::global_policy;
use cuda_async::device_operation::*;
use cutile::api::copy;
use cutile::tensor::{Partition, Tensor, ToHostVec, Unpartition};
use cutile::tile_kernel::{IntoDeviceOperationPartition, TileKernel};
use cutile::{api, error::Error};
use tokio::task::JoinHandle;

#[cutile::module]
pub mod my_kernels {

    use cutile::core::*;

    #[cutile::entry()]
    pub fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load();
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            // TODO (hme): Inject continue.
            continue;
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load().reshape(const_shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i]).reshape(const_shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z.reshape(const_shape![BM]));
    }

    #[cutile::entry()]
    fn relu<const D: i32>(input_output: &mut Tensor<f32, { [D] }>) {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, const_shape![D]);
        let input = input_output.load();
        input_output.store(max_tile(zero_tile, input));
    }
}

use my_kernels::*;

// Simulate loading input data.
fn load_data<const RANK: usize>(
    batch_size: [usize; RANK],
) -> impl DeviceOperation<Output = Tensor<f32>> {
    api::randn_f32(0.0, 1.0, batch_size, None)
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), Error> {
    // Get device scheduling policies.
    let num_devices = 4;
    let devices = {
        let mut r = vec![];
        for _ in 0..num_devices {
            // Pretend we have multiple devices...
            r.push(global_policy(0)?);
        }
        r
    };

    let dim = 16;
    let block_dim = 4;
    let fully_connected_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let output_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let w0 = api::randn_f32(0.0f32, 1.0, [dim, dim], None); // impl DeviceOperation
    let w1 = api::randn_f32(0.0f32, 1.0, [dim], None); // impl DeviceOperation
    let w = zip!(w0.arc(), w1.arc()).schedule(&devices[0])?.await?;
    let mut joins = vec![];
    for i in 1..num_devices {
        let w_copy = tokio::spawn(zip!(copy(&w.0).arc(), copy(&w.1).arc()).schedule(&devices[i])?);
        joins.push(w_copy);
    }
    let mut model_weights = vec![w];
    for join in joins {
        model_weights.push(join.await.unwrap()?);
    }

    // Asynchronously compute forward pass for each batch of data on each device.
    let mut futures: Vec<
        JoinHandle<Result<Partition<Tensor<f32>>, cuda_async::error::DeviceError>>,
    > = vec![];
    for i in 0..num_devices {
        let w = &model_weights[i];
        let (w0, w1) = (w.0.clone(), w.1.clone());
        let data = load_data([dim, dim]).arc();
        let out0 = api::zeros::<2, f32>([dim, dim]).partition([block_dim, block_dim]);
        let fully_connected_layer = fully_connected_layer.to_vec();
        let (out0, _, _) = gemm_op(out0, data, value(w0))
            .generics(fully_connected_layer)
            .unzip();
        let out1 = api::zeros::<1, f32>([dim]).partition([block_dim]);
        let output_layer = output_layer.to_vec();
        let (out1, _, _) = matvec_op(out1, out0.unpartition().arc(), value(w1))
            .generics(output_layer)
            .unzip();
        let (out1,) = relu_op(out1).unzip();
        futures.push(tokio::spawn(out1.schedule(&devices[i])?));
    }

    // Wait on results.
    let mut outputs: Vec<Tensor<f32>> = vec![];
    for future in futures.into_iter() {
        let tensor = future.await.unwrap()?.unpartition();
        outputs.push(tensor);
    }
    for output in outputs {
        println!("{:?}", output.to_host_vec().await?);
    }
    Ok(())
}
