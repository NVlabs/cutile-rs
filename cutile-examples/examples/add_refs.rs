/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Demonstrates the borrow-based kernel API: pass `&Tensor` for inputs and
//! `Partition<&mut Tensor>` for outputs. No Arc, no return capture, no
//! unpartition — kernels write in place through the borrows.

use cutile::prelude::*;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
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

use my_module::add;

fn main() {
    let device = cuda_core::Device::new(0).unwrap();
    let stream = device.new_stream().unwrap();

    let x = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
    let y = api::ones::<f32>(&[32]).sync_on(&stream).unwrap();
    let mut z = api::zeros::<f32>(&[32]).sync_on(&stream).unwrap();

    // Borrow-based launch: &x, &y for inputs, &mut z for output.
    // No return value needed — z is written in place.
    let _ = add((&mut z).partition([4]), &x, &y)
        .sync_on(&stream)
        .unwrap();

    // z already has the result. Use dup() to read without consuming.
    let z_host: Vec<f32> = z.dup().to_host_vec().sync_on(&stream).unwrap();
    assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    println!("add_refs: z = {:?}", &z_host[..8]);

    // Run again — reuse the same buffers, no allocation.
    let _ = add((&mut z).partition([4]), &x, &y)
        .sync_on(&stream)
        .unwrap();

    // Final read — can consume z since we're done.
    let z_host: Vec<f32> = z.to_host_vec().sync_on(&stream).unwrap();
    assert!(z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    println!("add_refs: second run, same result — buffers reused");
}
