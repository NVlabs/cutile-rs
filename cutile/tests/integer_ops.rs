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
    fn mini_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test integer minimum operation
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);
        let result: Tile<i64, S> = mini(x, y);
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

    #[cutile::entry()]
    fn named_integer_arithmetic_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let x: Tile<i64, S> = load_tile_mut(output);
        let one: Tile<i64, S> = constant(1i64, output.shape());

        let sum: Tile<i64, S> = addi(x, one, overflow::NoSignedWrap);
        let diff: Tile<i64, S> = subi(sum, one, overflow::NoSignedWrap);
        let product: Tile<i64, S> = muli(diff, one, overflow::NoSignedWrap);
        let quotient: Tile<i64, S> = divi(product, one, rounding::Zero);
        let result: Tile<i64, S> = remi(quotient, one);
        output.store(result);
    }

    #[cutile::entry()]
    fn cmpi_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = constant(1i64, output.shape());
        let mask: Tile<bool, S> = cmpi(x, y, predicate::GreaterThanOrEqual);
        let result: Tile<i64, S> = select(mask, x, y);
        output.store(result);
    }
}

use integer_ops_module::__module_ast_self;

#[test]
fn compile_maxi() -> () {
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "maxi_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
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
fn compile_mini() -> () {
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "mini_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
        println!("\n=== MINI MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= mini"),
            "Expected mini operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected signedness attribute in mini operation"
        );
    });
}

#[test]
fn compile_mulhii() -> () {
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "mulhii_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
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
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "maxi_unsigned_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
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

#[test]
fn compile_named_integer_arithmetic() -> () {
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "named_integer_arithmetic_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
        println!("\n=== NAMED INTEGER ARITHMETIC MLIR ===\n{}", module_op_str);

        for op in ["addi", "subi", "muli", "divi", "remi"] {
            assert!(
                module_op_str.contains(format!("= {op}").as_str()),
                "Expected {op} operation in MLIR output"
            );
        }
        assert!(
            module_op_str.contains("overflow"),
            "Expected overflow attributes in named integer arithmetic"
        );
    });
}

#[test]
fn compile_cmpi() -> () {
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "integer_ops_module",
            "cmpi_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
        println!("\n=== CMPI MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("= cmpi"),
            "Expected cmpi operation in MLIR output"
        );
        assert!(
            module_op_str.contains("signed"),
            "Expected inferred signedness on cmpi"
        );
        assert!(
            module_op_str.contains("select"),
            "Expected select consuming the cmpi mask"
        );
    });
}
