/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Example: Inter-module device function calls.
//!
//! Demonstrates how device functions defined in one module can be called
//! from entry points in another module. The compiler resolves cross-module
//! calls at JIT compile time — all device functions are inlined into the
//! entry point.
//!
//! Note: both modules must use `cutile::core::*` for DSL access. The
//! second module uses `use super::activations::*` to import device
//! functions from the first.

use cutile::prelude::*;

/// Module A: reusable activation device functions.
/// No entry points — these are building blocks only.
#[cutile::module]
mod activations {
    use cutile::core::*;

    /// ReLU: max(x, 0)
    pub fn relu<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        max_tile(x, zero)
    }

    /// Square: x * x
    pub fn square<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        x * x
    }
}

/// Module B: kernels that use device functions from Module A.
/// Import device functions from the activations module so both
/// Rust's type checker and the JIT compiler can resolve them.
#[cutile::module]
mod my_kernels {
    use crate::activations::{relu, square};
    use cutile::core::*;

    /// Composes relu and square from Module A.
    #[cutile::entry()]
    fn apply_relu_square<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, S> = input.load_tile(const_shape!(S), [pid.0]);
        let activated: Tile<f32, S> = relu(tile);
        output.store(square(activated));
    }

    /// Uses just relu from Module A.
    #[cutile::entry()]
    fn apply_relu<const S: [i32; 1]>(output: &mut Tensor<f32, S>, input: &Tensor<f32, { [-1] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, S> = input.load_tile(const_shape!(S), [pid.0]);
        output.store(relu(tile));
    }
}

use my_kernels::{apply_relu, apply_relu_square};

fn main() -> Result<(), Error> {
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let block: usize = 64;
    let n: usize = 128;

    // Input: [0, 1, 2, ..., 127]
    let input: Tensor<f32> = api::arange::<f32>(n).sync_on(&stream)?;

    // -- relu_square: calls relu() and square() from Module A --
    let mut rs_out: Tensor<f32> = api::zeros::<f32>(&[n]).sync_on(&stream)?;
    apply_relu_square((&mut rs_out).partition([block]), &input).sync_on(&stream)?;

    let rs_host: Vec<f32> = rs_out.dup().to_host_vec().sync_on(&stream)?;
    println!("relu_square(arange(128)) — calls activations::relu + activations::square:");
    println!("  [0]   = {} (expected 0)", rs_host[0]);
    println!("  [10]  = {} (expected 100)", rs_host[10]);
    println!("  [127] = {} (expected {})", rs_host[127], 127.0f32 * 127.0);

    for (i, &v) in rs_host.iter().enumerate() {
        let expected: f32 = (i as f32) * (i as f32);
        assert!(
            (v - expected).abs() < 1.0,
            "relu_square[{i}]: got {v}, expected {expected}"
        );
    }

    // -- relu only: calls relu() from Module A --
    let mut relu_out: Tensor<f32> = api::zeros::<f32>(&[n]).sync_on(&stream)?;
    apply_relu((&mut relu_out).partition([block]), &input).sync_on(&stream)?;

    let relu_host: Vec<f32> = relu_out.dup().to_host_vec().sync_on(&stream)?;
    println!("\nrelu(arange(128)) — calls activations::relu:");
    println!("  [0]   = {} (expected 0)", relu_host[0]);
    println!("  [64]  = {} (expected 64)", relu_host[64]);

    for (i, &v) in relu_host.iter().enumerate() {
        let expected: f32 = i as f32;
        assert!(
            (v - expected).abs() < 1e-3,
            "relu[{i}]: got {v}, expected {expected}"
        );
    }

    println!("\nAll inter-module examples passed.");
    Ok(())
}
