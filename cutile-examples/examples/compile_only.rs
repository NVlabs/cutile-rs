/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: Compile to without requiring a GPU to run on
 *
 * Run with: cargo run -p cutile-examples --example compile_only --no-default-features
 */

use cutile::cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile::cutile_compiler::cuda_tile_write_bytecode_to_buffer;
use std::env;
use std::slice;

// Same syntax as full CUDA version - features control behavior
#[cutile::module(compile_only = true)]
mod my_kernels {
    use cutile::core::*;

    /// Simple kernel that does tile math without dynamic tensor inputs
    #[cutile::entry()]
    fn tile_math<const S: [i32; 1]>(output: &mut Tensor<f32, S>, scalar: f32) {
        // Get block ID and create tiles
        let _pid = get_tile_block_id().0;
        let scalar_tile: Tile<f32, S> = broadcast_scalar(scalar, output.shape());
        let ones: Tile<f32, S> = broadcast_scalar(1.0f32, output.shape());

        // Simple computation
        let result = scalar_tile + ones;
        output.store(result);
    }
}

fn main() {
    // Default to sm_90 (Hopper) if not specified
    let gpu_name = env::args().nth(1).unwrap_or_else(|| "sm_90".to_string());
    println!("Target GPU: {}", gpu_name);

    // Get the module ASTs from the generated code
    let module_asts = my_kernels::_module_asts();

    // Create the modules container
    let modules = match CUDATileModules::new(module_asts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to create modules: {:?}", e);
            return;
        }
    };

    // Compile with specific generic args (tile size = 32)
    let module_name = "my_kernels";
    let function_name = "tile_math";
    let function_generics = vec!["32".to_string()];
    // Stride args for the output tensor (1D tensor with stride 1)
    let output_strides: [i32; 1] = [1];
    let stride_args: Vec<(&str, &[i32])> = vec![("output", &output_strides)];
    let const_grid: Option<(u32, u32, u32)> = None;

    println!("Compiling {module_name}::{function_name}");

    let compiler = match CUDATileFunctionCompiler::new(
        &modules,
        module_name,
        function_name,
        &function_generics,
        &stride_args,
        const_grid,
        gpu_name.clone(),
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create compiler: {:?}", e);
            return;
        }
    };

    let module_op = match compiler.compile() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Compilation failed: {:?}", e);
            return;
        }
    };

    // Print human readable MLIR IR
    let mlir_string = module_op.as_operation().to_string();
    println!("Generated MLIR IR:\n");
    println!("{}", mlir_string);

    // Get compiled bytecode
    let bytecode = cuda_tile_write_bytecode_to_buffer(&module_op);
    let raw = bytecode.to_raw();
    let bytes: &[u8] = unsafe { slice::from_raw_parts(raw.data as *const u8, raw.length) };

    println!("\nCompiled bytecode: {} bytes", bytes.len());
    println!(
        "First 32 bytes (hex): {:02x?}",
        &bytes[..bytes.len().min(32)]
    );

    // Write MLIR and bytecode to files
    std::fs::write("output.mlir", mlir_string).unwrap();
    std::fs::write("output.bc", bytes).unwrap();
}
