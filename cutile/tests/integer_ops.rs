/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod integer_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn maxi_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer maximum operation
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);
        let result: Tile<i64, S> = maxi(x, y);
        output.store(result);
    }

    #[cutile::entry()]
    fn mulhii_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test multiply high operation
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);
        let result: Tile<i64, S> = mulhii(x, y);
        output.store(result);
    }

    #[cutile::entry()]
    fn maxi_unsigned_kernel<const S: [i32; 1]>(output: &mut Tensor<u32, S>) {
        // Test integer maximum with unsigned types
        let x: Tile<u32, S> = load_tile_mut(output);
        let y: Tile<u32, S> = load_tile_mut(output);
        let result: Tile<u32, S> = maxi(x, y);
        output.store(result);
    }
}

use integer_ops_module::_module_asts;

#[test]
fn compile_maxi() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "maxi_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== MAXI MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= maxi"),
            "Expected maxi operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected signedness attribute in maxi operation"
        );

        println!("\n✓ maxi operation verified in MLIR output");
    });
}

#[test]
fn compile_mulhii() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "mulhii_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== MULHII MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mulhii"),
            "Expected mulhii operation in MLIR output"
        );

        println!("\n✓ mulhii operation verified in MLIR output");
    });
}

#[test]
fn compile_maxi_unsigned() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "maxi_unsigned_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== MAXI UNSIGNED MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= maxi"),
            "Expected maxi operation in MLIR output"
        );
        assert!(
            module_op_str.contains("unsigned"),
            "Expected unsigned signedness attribute in maxi operation"
        );

        println!("\n✓ maxi with unsigned types verified in MLIR output");
    });
}
