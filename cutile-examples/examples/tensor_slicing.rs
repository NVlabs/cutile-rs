/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Example: Tensor views and slicing.
//!
//! Demonstrates zero-copy tensor slicing — borrowing subregions of a
//! tensor and passing them to GPU kernels without copying data.
//!
//! Pattern: Compute on a slice of a larger tensor. This is common when
//! processing batches, applying windowed operations, or working with
//! padded data where only a subregion is valid.

use cutile::prelude::*;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    /// Adds two tensors element-wise.
    #[cutile::entry()]
    fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(const_shape![B], [pid]);
        let tile_b = b.load_tile(const_shape![B], [pid]);
        out.store(tile_a + tile_b);
    }

    /// Scales a tensor by a scalar.
    #[cutile::entry()]
    fn scale<const B: i32>(out: &mut Tensor<f32, { [B] }>, a: &Tensor<f32, { [-1] }>, scalar: f32) {
        let pid = get_tile_block_id().0;
        let tile_a = a.load_tile(const_shape![B], [pid]);
        let s: Tile<f32, { [B] }> = scalar.broadcast(out.shape());
        out.store(tile_a * s);
    }
}

use my_module::{add, scale};

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let block = 128;

    // Create a 1024-element tensor: [0, 1, 2, ..., 1023]
    let data = api::arange::<f32>(1024).sync_on(&stream)?;

    // -- Example 1: Slice a subregion --
    // Borrow elements [256..512] without copying. The slice is a
    // TensorView that points into the original tensor's memory.
    let slice_a = data.slice(&[256..512])?;
    println!("slice_a shape: {:?} (elements 256-511)", slice_a.shape());

    // Compute on the slice: add it to a ones tensor of the same size.
    let ones = api::ones::<f32>(&[256]).sync_on(&stream)?;
    let mut out = api::zeros::<f32>(&[256]).sync_on(&stream)?;
    add((&mut out).partition([block]), &slice_a, &ones).sync_on(&stream)?;

    let host: Vec<f32> = out.dup().to_host_vec().sync_on(&stream)?;
    println!("out[0] = {} (expected {})", host[0], 256.0 + 1.0);
    println!("out[255] = {} (expected {})", host[255], 511.0 + 1.0);
    assert!((host[0] - 257.0).abs() < 1e-3);
    assert!((host[255] - 512.0).abs() < 1e-3);

    // -- Example 2: Chained slices --
    // Slicing a slice composes offsets. data[128..896][128..384]
    // is the same as data[256..512].
    let outer = data.slice(&[128..896])?;
    let inner = outer.slice(&[128..384])?;
    println!(
        "\nchained slice shape: {:?} (elements 256-511)",
        inner.shape()
    );

    let mut out2 = api::zeros::<f32>(&[256]).sync_on(&stream)?;
    scale((&mut out2).partition([block]), &inner, 2.0).sync_on(&stream)?;

    let host2: Vec<f32> = out2.dup().to_host_vec().sync_on(&stream)?;
    println!("out2[0] = {} (expected {})", host2[0], 256.0 * 2.0);
    println!("out2[255] = {} (expected {})", host2[255], 511.0 * 2.0);
    assert!((host2[0] - 512.0).abs() < 1e-3);
    assert!((host2[255] - 1022.0).abs() < 1e-3);

    // -- Example 3: View + Slice (2D) --
    // Reshape data as 32x32 matrix, then slice rows 8..16.
    let matrix = data.view(&[32, 32])?;
    let row_slice = matrix.slice(&[8..16])?;
    println!(
        "\n2D slice shape: {:?} (rows 8-15 of 32x32)",
        row_slice.shape()
    );
    println!("2D slice strides: {:?}", row_slice.strides());

    println!("\nAll examples passed.");
    Ok(())
}
