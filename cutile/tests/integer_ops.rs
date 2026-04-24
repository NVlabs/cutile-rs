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

    // ----- Kernels that exercise the bytecode writer's DefaultValuedAttr
    // path. Before the fix, write_bytecode panicked with
    // `missing attribute 'overflow' on op <...>` because the DSL macro
    // emits these ops without an explicit overflow attr; the compiler
    // passes that through; and op_writer.rs used write_inline_attr instead
    // of write_inline_attr_or_default.

    #[cutile::entry()]
    fn trunci_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let x: Tile<i64, S> = load_tile_mut(output);
        let t: Tile<i32, S> = trunci(x);
        let e: Tile<i64, S> = exti(t);
        output.store(e);
    }

    #[cutile::entry()]
    fn shli_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);
        let z: Tile<i64, S> = shli(x, y);
        output.store(z);
    }

    #[cutile::entry()]
    fn addi_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);
        let z: Tile<i64, S> = x + y; // lowers to AddI
        output.store(z);
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
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
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

// =========================================================================
// DefaultValuedAttr bytecode-write regression tests.
//
// These tests drive the full DSL → compile → `write_bytecode` path for
// ops whose TableGen schema declares attrs as `DefaultValuedAttr<...>`
// (AddI / MulI / SubI / ShLI / TruncI overflow, DivI rounding, FToF
// rounding_mode). Before the fix, these kernels compiled to MLIR
// correctly but panicked during bytecode serialization with
// `BytecodeWrite("missing attribute 'overflow' on op <Opcode>")`
// because op_writer.rs used `write_inline_attr` instead of
// `write_inline_attr_or_default`.
// =========================================================================

fn compile_and_write_bytecode(kernel_name: &str) -> Vec<u8> {
    let modules = CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "integer_ops_module",
        kernel_name,
        &[128.to_string()],
        &[("output", &[1])],
        &[],
        &[],
        None,
        gpu_name,
        &CompileOptions::default(),
    )
    .expect("compiler");
    let module = compiler.compile().expect("compile");
    cutile_ir::write_bytecode(&module).expect("write_bytecode failed")
}

#[test]
fn trunci_kernel_bytecode_writes_without_panic() {
    common::with_test_stack(|| {
        let bc = compile_and_write_bytecode("trunci_kernel");
        assert!(bc.len() > 12, "bytecode too short");
    });
}

#[test]
fn shli_kernel_bytecode_writes_without_panic() {
    common::with_test_stack(|| {
        let bc = compile_and_write_bytecode("shli_kernel");
        assert!(bc.len() > 12, "bytecode too short");
    });
}

#[test]
fn addi_kernel_bytecode_writes_without_panic() {
    common::with_test_stack(|| {
        let bc = compile_and_write_bytecode("addi_kernel");
        assert!(bc.len() > 12, "bytecode too short");
    });
}
