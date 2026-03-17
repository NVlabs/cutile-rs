/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod type_conversion_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn conversion_ops_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer conversion operations
        let x: Tile<i64, S> = load_tile_mut(output);
        // Truncate to i32, then extend back to i64
        let truncated: Tile<i32, S> = trunci(x);
        let extended: Tile<i64, S> = exti(truncated);
        output.store(extended);
    }

    #[cutile::entry()]
    fn ptr_conversion_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test pointer conversion operations
        let x: Tile<i64, S> = load_tile_mut(output);
        // Convert to pointer, cast pointer type, convert back to int
        let ptrs: PointerTile<*mut i64, S> = int_to_ptr(x);
        let ptrs_f32: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs);
        let ptrs_back: PointerTile<*mut i64, S> = ptr_to_ptr(ptrs_f32);
        let ints: Tile<i64, S> = ptr_to_int(ptrs_back);
        output.store(ints);
    }

    #[cutile::entry()]
    fn exti_unsigned_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer extension with unsigned types (zero extension)
        // Note: Using i64 tensor but operating on u32 tiles due to WithDType limitations
        let x: Tile<i64, S> = load_tile_mut(output);
        // Truncate to u32, then extend back to i64 with unsigned (zero extension)
        let truncated: Tile<u32, S> = trunci(x);
        let extended: Tile<i64, S> = exti(truncated);
        output.store(extended);
    }

    
    #[cutile::entry()]
    fn bf16_conversion_kernel<const S: [i32; 1]>(output: &mut Tensor<bf16, S>) {
        // Exercises bf16 <-> f32 tile conversion lowering
        let x: Tile<bf16, S> = load_tile_mut(output);
        let upcast: Tile<f32, S> = convert_tile(x);
        let downcast: Tile<bf16, S> = convert_tile(upcast);
        output.store(downcast);
    }
}

use type_conversion_ops_module::_module_asts;

#[test]
fn compile_conversion_ops() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "type_conversion_ops_module",
            "conversion_ops_kernel",
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
        println!("\n=== CONVERSION OPS MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= trunci"),
            "Expected trunci operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= exti"),
            "Expected exti operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected signedness attribute in exti operation"
        );

        println!("\n✓ trunci and exti operations verified in MLIR output");
    });
}

#[test]
fn compile_ptr_conversion() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "type_conversion_ops_module",
            "ptr_conversion_kernel",
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
        println!("\n=== PTR CONVERSION MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= int_to_ptr"),
            "Expected int_to_ptr operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= ptr_to_ptr"),
            "Expected ptr_to_ptr operation in MLIR output"
        );
        assert!(
            module_op_str.contains("= ptr_to_int"),
            "Expected ptr_to_int operation in MLIR output"
        );

        println!("\n✓ Pointer conversion operations verified in MLIR output");
    });
}

#[test]
fn compile_exti_unsigned() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "type_conversion_ops_module",
            "exti_unsigned_kernel",
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
        println!("\n=== EXTI UNSIGNED MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= exti"),
            "Expected exti operation in MLIR output"
        );
        assert!(
            module_op_str.contains("unsigned"),
            "Expected unsigned signedness attribute (zero extension)"
        );

        println!("\n✓ exti with unsigned types (zero extension) verified in MLIR output");
    });
}

#[test]
fn compile_bf16_conversion() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "type_conversion_ops_module",
            "bf16_conversion_kernel",
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
        println!("\n=== BF16 CONVERSION MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= ftof"),
            "Expected floating-point conversion operation in MLIR output"
        );
        assert!(
            module_op_str.contains("bf16"),
            "Expected bf16 type in MLIR output"
        );
    });
}
