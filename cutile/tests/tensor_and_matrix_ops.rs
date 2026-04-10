/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod tensor_and_matrix_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn cat_kernel(output: &mut Tensor<f32, { [8] }>) {
        // Test cat operation - concatenate two tiles
        let source: Tile<f32, { [8] }> = load_tile_mut(output);

        // Split into two halves
        let idx0: Tile<i32, { [] }> = scalar_to_tile(0i32);
        let tile_a: Tile<f32, { [4] }> = extract(source, [idx0]);

        let idx1: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let tile_b: Tile<f32, { [4] }> = extract(source, [idx1]);

        // Concatenate them back together
        let result: Tile<f32, { [8] }> = cat(tile_a, tile_b, 0i32);

        output.store(result);
    }

    #[cutile::entry()]
    fn extract_kernel(output: &mut Tensor<f32, { [8] }>) {
        // Test extract operation - extract 4-element slices from an 8-element tile
        let source: Tile<f32, { [8] }> = load_tile_mut(output);

        // Extract first half [0:4]
        let idx0: Tile<i32, { [] }> = scalar_to_tile(0i32);
        // This extract is independent from the second extract below. Each extract slices the source tile independently.
        // The number of slices is determined by the number of indices provided.
        let _slice0: Tile<f32, { [4] }> = extract(source, [idx0]);

        // Extract second half [4:8]
        let idx1: Tile<i32, { [] }> = scalar_to_tile(1i32);
        let _slice1: Tile<f32, { [4] }> = extract(source, [idx1]);

        // Store original (extract operations will appear in MLIR)
        output.store(source);
    }

    #[cutile::entry()]
    fn mmai_kernel(output: &mut Tensor<i64, { [16, 16] }>) {
        // Test mmai operation - integer matrix multiply-accumulate
        // NOTE: Using i64 tensor because mma output is i32, extended to i64 for storage

        let lhs_shape: Shape<{ [16, 32] }> = Shape::<{ [16, 32] }> {
            dims: &[16i32, 32i32],
        };
        let rhs_shape: Shape<{ [32, 16] }> = Shape::<{ [32, 16] }> {
            dims: &[32i32, 16i32],
        };
        let acc_shape: Shape<{ [16, 16] }> = Shape::<{ [16, 16] }> {
            dims: &[16i32, 16i32],
        };

        let lhs: Tile<i8, { [16, 32] }> = constant(1i8, lhs_shape);
        let rhs: Tile<i8, { [32, 16] }> = constant(1i8, rhs_shape);
        let acc: Tile<i32, { [16, 16] }> = constant(0i32, acc_shape);

        // Perform integer matrix multiply-accumulate
        let result_i32: Tile<i32, { [16, 16] }> = mma(lhs, rhs, acc);

        // Convert to i64 for storage
        let result_i64: Tile<i64, { [16, 16] }> = exti(result_i32);

        output.store(result_i64);
    }
}

use tensor_and_matrix_ops_module::_module_asts;

#[test]
fn compile_cat() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "tensor_and_matrix_ops_module",
            "cat_kernel",
            &[],
            &[("output", &[8])],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== CAT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= cat"),
            "Expected cat operation in MLIR output"
        );
        assert!(
            module_op_str.contains("dim = 0"),
            "Expected dim=0 attribute in cat operation"
        );

        println!("\n✓ cat operation verified in MLIR output");
    });
}

#[test]
fn compile_extract() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "tensor_and_matrix_ops_module",
            "extract_kernel",
            &[],
            &[("output", &[1])],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== EXTRACT MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= extract"),
            "Expected extract operation in MLIR output"
        );

        println!("\n✓ extract operation verified in MLIR output");
    });
}

#[test]
fn compile_mmai() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "tensor_and_matrix_ops_module",
            "mmai_kernel",
            &[],
            &[("output", &[16, 16])],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== MMAI MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mmai"),
            "Expected mmai operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected signedness attributes in mmai operation"
        );
        assert!(
            module_op_str.contains("= exti"),
            "Expected exti for i32->i64 conversion"
        );

        println!("\n✓ mmai operation verified in MLIR output (using i64 tensor workaround)");
    });
}
