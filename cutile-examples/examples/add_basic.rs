/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_async::device_operation::{DeviceOperation, Zippable};
use cutile::{
    self, api,
    tensor::Unpartition,
    tile_kernel::{
        zip, IntoDeviceOperationPartition, TensorDeviceOpToHostVec, TileKernel, Unzippable3,
    },
};
use my_module::add;

#[cutile::module]
mod my_module {
    use cutile::core::*;
    #[cutile::entry(print_ir = true)]
    fn add<const S: [i32; 1]>(
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
        c: &mut Tensor<f32, S>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(const_shape!(S), [pid]);
        let tile_b = b.load_tile(const_shape!(S), [pid]);
        c.store(tile_a + tile_b);
    }
}

fn main() -> () {
    let a = api::ones([32]).arc();
    let b = api::ones([32]).arc();
    let c = api::zeros([32]).partition([4]);
    let c_host_vec = zip!(a, b, c)
        .apply(|(a, b, c)| add(a, b, c).grid((8, 1, 1)))
        .unzip()
        .2
        .unpartition()
        .to_host_vec()
        .sync();
    println!("{:#?}", c_host_vec);
}
