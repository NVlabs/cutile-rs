/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: Compile kernels to Tile IR and bytecode without requiring a GPU.
 *
 * This demonstrates the compile-only API: the binary compiles on any machine
 * with CUDA **headers** (for type definitions). The CUDA driver library is
 * only loaded at runtime when GPU operations are needed — this example never
 * touches the GPU.
 *
 * Run with: cargo run -p cutile-examples --example compile_only
 * Or with a specific target: cargo run -p cutile-examples --example compile_only -- sm_80
 */

use cutile::compile_api::KernelCompiler;
use std::env;

#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    /// Simple kernel that does tile math without dynamic tensor inputs.
    #[cutile::entry()]
    fn tile_math<const S: [i32; 1]>(output: &mut Tensor<f32, S>, scalar: f32) {
        let scalar_tile: Tile<f32, S> = broadcast_scalar(scalar, output.shape());
        let ones: Tile<f32, S> = broadcast_scalar(1.0f32, output.shape());
        let result = scalar_tile + ones;
        output.store(result);
    }
}

fn main() {
    println!(
        "CUDA driver: {}",
        if cuda_bindings::is_cuda_driver_available() {
            "available"
        } else {
            "not available (compile-only mode)"
        }
    );

    let gpu_name = env::args().nth(1).unwrap_or_else(|| "sm_80".to_string());
    println!("Target GPU: {gpu_name}");
    println!("Compiling my_kernels::tile_math\n");

    let artifacts = KernelCompiler::new(my_kernels::_module_asts, "my_kernels", "tile_math")
        .generics(vec!["32".into()])
        .strides(&[("output", &[1])])
        .target(&gpu_name)
        .compile()
        .expect("compilation failed");

    // Print human-readable Tile IR
    println!("Generated Tile IR:\n");
    println!("{}", artifacts.ir_text());

    // Serialize to bytecode
    let bytecode = artifacts.bytecode().expect("bytecode serialization failed");
    println!("Compiled bytecode: {} bytes", bytecode.len());
    println!(
        "First 32 bytes (hex): {:02x?}",
        &bytecode[..bytecode.len().min(32)]
    );

    // Write artifacts to files
    std::fs::write("output.mlir", artifacts.ir_text()).unwrap();
    std::fs::write("output.bc", &bytecode).unwrap();
    println!("\nWrote output.mlir and output.bc");
}
