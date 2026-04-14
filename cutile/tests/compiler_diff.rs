/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![cfg(feature = "mlir")]

//! Comprehensive MLIR diff tests: compile the same kernel through both old
//! (melior/MLIR) and new (compiler2/tile-ir) compilers and print the outputs
//! side by side for manual inspection.

use cutile;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::compiler2::Compiler2;
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

// =========================================================================
// Re-use kernel modules from existing test files
// =========================================================================

#[cutile::module]
mod diff_unary_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn sqrt_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let result: Tile<f32, S> = sqrt(x, "negative_inf");
        output.store(result);
    }
}

#[cutile::module]
mod diff_binary_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn minmax_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let x: Tile<f32, S> = load_tile_mut(output);
        let y: Tile<f32, S> = load_tile_mut(output);
        let max_result: Tile<f32, S> = maxf(x, y);
        let min_result: Tile<f32, S> = minf(max_result, y);
        output.store(min_result);
    }
}

#[cutile::module]
mod diff_integer_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn maxi_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let x: Tile<i64, S> = load_tile_mut(output);
        let y: Tile<i64, S> = load_tile_mut(output);
        let result: Tile<i64, S> = maxi(x, y);
        output.store(result);
    }
}

#[cutile::module]
mod diff_control_flow_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn control_flow_test_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        dynamic_value: i32,
    ) {
        let mut sum: Tile<f32, S> = load_tile_mut(output);

        for _i in 0i32..10i32 {
            sum = sum + sum;
        }

        if dynamic_value < 5i32 {
            sum = sum + sum;
        } else {
            sum = sum - sum;
        }

        output.store(sum);
    }
}

// =========================================================================
// Helpers
// =========================================================================

fn compile_old(
    asts: Vec<cutile_compiler::ast::Module>,
    module_name: &str,
    function_name: &str,
    shape: &[String],
    strides: &[(&str, &[i32])],
) -> String {
    let modules = CUDATileModules::new(asts).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        module_name,
        function_name,
        shape,
        strides,
        None,
        gpu_name,
    )
    .expect("Old compiler creation failed");
    let module_op = compiler.compile().expect("Old compiler failed");
    let result = module_op.as_operation().to_string();
    result
}

fn compile_new(
    asts: Vec<cutile_compiler::ast::Module>,
    module_name: &str,
    function_name: &str,
    shape: &[String],
    strides: &[(&str, &[i32])],
) -> String {
    let modules = CUDATileModules::new(asts).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler2 = Compiler2::new(
        &modules,
        module_name,
        function_name,
        shape,
        strides,
        None,
        gpu_name,
    )
    .expect("New compiler creation failed");
    let tile_module = compiler2.compile().expect("New compiler failed");
    tile_module.to_mlir_text()
}

fn print_diff(test_name: &str, old_mlir: &str, new_mlir: &str) {
    let sep = "=".repeat(80);
    println!("\n{}", sep);
    println!("KERNEL: {}", test_name);
    println!("{}", sep);

    println!("\n--- OLD COMPILER (melior/MLIR) ---");
    for (i, line) in old_mlir.lines().enumerate() {
        println!("OLD {:3}: {}", i + 1, line);
    }

    println!("\n--- NEW COMPILER (compiler2/tile-ir) ---");
    for (i, line) in new_mlir.lines().enumerate() {
        println!("NEW {:3}: {}", i + 1, line);
    }

    // Line-by-line comparison
    println!("\n--- LINE-BY-LINE DIFF ---");
    let old_lines: Vec<&str> = old_mlir.lines().collect();
    let new_lines: Vec<&str> = new_mlir.lines().collect();
    let max_lines = old_lines.len().max(new_lines.len());

    let mut diff_count = 0;
    for i in 0..max_lines {
        let old_line = old_lines.get(i).unwrap_or(&"<missing>");
        let new_line = new_lines.get(i).unwrap_or(&"<missing>");
        if old_line.trim() != new_line.trim() {
            diff_count += 1;
            println!("DIFF at line {}:", i + 1);
            println!("  OLD: {}", old_line);
            println!("  NEW: {}", new_line);
        }
    }

    if diff_count == 0 {
        println!("  (no differences found)");
    } else {
        println!("\nTotal differences: {}", diff_count);
    }

    println!(
        "\nOld line count: {}, New line count: {}",
        old_lines.len(),
        new_lines.len()
    );
}

// =========================================================================
// Tests
// =========================================================================

#[test]
fn diff_sqrt_kernel() {
    common::with_test_stack(|| {
        let shape = vec![128.to_string()];
        let strides = vec![("output", &[1][..])];
        let old = compile_old(
            diff_unary_module::_module_asts(),
            "diff_unary_module",
            "sqrt_kernel",
            &shape,
            &strides,
        );
        let new = compile_new(
            diff_unary_module::_module_asts(),
            "diff_unary_module",
            "sqrt_kernel",
            &shape,
            &strides,
        );
        print_diff("sqrt_kernel", &old, &new);
    });
}

#[test]
fn diff_minmax_kernel() {
    common::with_test_stack(|| {
        let shape = vec![128.to_string()];
        let strides = vec![("output", &[1][..])];
        let old = compile_old(
            diff_binary_module::_module_asts(),
            "diff_binary_module",
            "minmax_kernel",
            &shape,
            &strides,
        );
        let new = compile_new(
            diff_binary_module::_module_asts(),
            "diff_binary_module",
            "minmax_kernel",
            &shape,
            &strides,
        );
        print_diff("minmax_kernel", &old, &new);
    });
}

#[test]
fn diff_maxi_kernel() {
    common::with_test_stack(|| {
        let shape = vec![128.to_string()];
        let strides = vec![("output", &[1][..])];
        let old = compile_old(
            diff_integer_module::_module_asts(),
            "diff_integer_module",
            "maxi_kernel",
            &shape,
            &strides,
        );
        let new = compile_new(
            diff_integer_module::_module_asts(),
            "diff_integer_module",
            "maxi_kernel",
            &shape,
            &strides,
        );
        print_diff("maxi_kernel", &old, &new);
    });
}

#[test]
fn diff_control_flow_kernel() {
    common::with_test_stack(|| {
        let shape = vec![128.to_string()];
        let strides = vec![("output", &[1][..])];
        let old = compile_old(
            diff_control_flow_module::_module_asts(),
            "diff_control_flow_module",
            "control_flow_test_kernel",
            &shape,
            &strides,
        );
        let new = compile_new(
            diff_control_flow_module::_module_asts(),
            "diff_control_flow_module",
            "control_flow_test_kernel",
            &shape,
            &strides,
        );
        print_diff("control_flow_test_kernel", &old, &new);
    });
}
