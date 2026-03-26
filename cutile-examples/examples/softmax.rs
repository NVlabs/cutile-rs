/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::DeviceOperation;
use cuda_core::CudaContext;
use cutile;
use cutile::api::arange;
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};
use std::sync::Arc;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn softmax<const BM: i32, const BN: i32>(
        x: &Tensor<f32, { [-1, -1] }>,
        y: &mut Tensor<f32, { [BM, BN] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like_2d(x, y);
        let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
        let tile_x_max: Tile<f32, { [BM, BN] }> =
            tile_x_max.reshape(const_shape![BM, 1]).broadcast(y.shape());
        let num: Tile<f32, { [BM, BN] }> = exp(tile_x - tile_x_max);
        let denom: Tile<f32, { [BM] }> = reduce_sum(num, 1);
        let denom = denom.reshape(const_shape![BM, 1]).broadcast(y.shape());
        y.store(num / denom);
    }
}

use cutile::utils::Float;
use my_module::softmax;

fn main() -> Result<(), Error> {
    // Create a context. Device 0 is associated with the context.
    let ctx = CudaContext::new(0)?;
    // Create a new stream on which we run CUDA operations.
    let stream = ctx.new_stream()?;
    let (m, n) = (4, 8);
    let (bm, bn) = (2, n as i32);
    let input: Arc<Tensor<f32>> = arange(m * n).sync_on(&stream)?.into();
    let x = input.copy_sync(&stream)?.reshape([m, n]).into();
    let y = input
        .copy_sync(&stream)?
        .reshape([m, n])
        .partition([bm, bn]);
    let (_x, y) = softmax(x, y).sync_on(&stream)?;
    let y_host: Vec<f32> = y.unpartition().to_host_vec().sync_on(&stream)?;
    for i in (0..y_host.len()).step_by(8) {
        let x = y_host[i..i + 8].to_vec();
        let sum: f32 = x.iter().sum();
        println!("softmax(x).sum(axis=1) = {}", sum);
        assert!(sum.epsilon_close(1.0));
    }
    Ok(())
}
