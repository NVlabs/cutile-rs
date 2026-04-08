/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile::prelude::*;
use my_module::add;

#[cutile::module]
mod my_module {
    use cutile::core::*;
    #[cutile::entry(print_ir = true)]
    fn add<const S: [i32; 1]>(
        c: &mut Tensor<f32, S>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(const_shape!(S), [pid]);
        let tile_b = b.load_tile(const_shape!(S), [pid]);
        c.store(tile_a + tile_b);
    }
}

fn main() {
    let c_host_vec = add(
        api::zeros(&[32]).partition([4]),
        api::ones(&[32]),
        api::ones(&[32]),
    )
    .grid((8, 1, 1))
    .first()
    .unpartition()
    .to_host_vec()
    .sync();
    println!("{:#?}", c_host_vec);
}
