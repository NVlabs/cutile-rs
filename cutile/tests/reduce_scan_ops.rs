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
mod reduce_scan_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn scan_sum_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test scan_sum operation
        let tile: Tile<f32, S> = load_tile_mut(output);

        // Scan along dimension 0 (cumulative sum) - result has same shape
        let prefix_sums: Tile<f32, S> = scan_sum(tile, 0i32, false, 0.0f32);

        // Store the prefix sums
        output.store(prefix_sums);
    }

    #[cutile::entry()]
    fn reduce_closure_test_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        // Test reduce with custom closure - sum
        let tile: Tile<f32, S> = load_tile_mut(input);

        // Use closure for sum reduction
        let sum_scalar = reduce(tile, 0i32, 0.0f32, |acc, x| acc + x);

        // Reshape and store
        let sum_as_array: Tile<f32, { [1] }> = sum_scalar.reshape(const_shape![1]);
        result.store(sum_as_array);
    }

    #[cutile::entry()]
    fn reduce_product_closure_test_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        // Test reduce with custom closure - product
        let tile: Tile<f32, S> = load_tile_mut(input);

        // Use closure for product reduction
        let product_scalar = reduce(tile, 0i32, 1.0f32, |acc, x| acc * x);

        // Reshape and store
        let product_as_array: Tile<f32, { [1] }> = product_scalar.reshape(const_shape![1]);
        result.store(product_as_array);
    }

    #[cutile::entry()]
    fn reduce_max_closure_test_kernel<const S: [i32; 1]>(
        input: &mut Tensor<f32, S>,
        result: &mut Tensor<f32, { [1] }>,
    ) {
        // Test reduce with custom closure - max
        let tile: Tile<f32, S> = load_tile_mut(input);

        // Use closure for max reduction with NEG_INFINITY as identity
        // f32::NEG_INFINITY works as identity because max(NEG_INFINITY, x) = x for any x
        //
        // TODO (np): Using maxf(acc, x, nan::Disabled, ftz::Disabled) here gives "cannot find function" error even though
        // it's imported via `use cutile::core::{*}`. This appears to be a scoping bug
        // where some functions aren't visible inside reduce/scan closures. Using max()
        // (scalar version) works correctly and is the right function for this context anyway.
        let max_scalar = reduce(tile, 0i32, f32::NEG_INFINITY, |acc, x| max(acc, x));

        // Reshape and store
        let max_as_array: Tile<f32, { [1] }> = max_scalar.reshape(const_shape![1]);
        result.store(max_as_array);
    }

    #[cutile::entry()]
    fn scan_closure_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test scan with custom closure - prefix product
        let tile: Tile<f32, S> = load_tile_mut(output);

        // Use closure for prefix product
        let prefix_products: Tile<f32, S> = scan(tile, 0i32, false, 1.0f32, |acc, x| acc * x);

        // Store the result
        output.store(prefix_products);
    }

    /// Regression: pre-fix, `reverse = true` was silently defaulted to
    /// `false` if the arg came through as anything other than a bool
    /// literal. With the `extract_bool_arg` helper the literal path is
    /// preserved; this kernel pins the positive `true` case so a future
    /// refactor can't silently swap it back.
    #[cutile::entry()]
    fn scan_reverse_true_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = load_tile_mut(output);
        let suffix_sums: Tile<f32, S> = scan_sum(tile, 0i32, true, 0.0f32);
        output.store(suffix_sums);
    }
}

use reduce_scan_ops_module::_module_asts;

#[test]
fn compile_scan_sum_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "reduce_scan_ops_module",
            "scan_sum_test_kernel",
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
        println!("\n=== SCAN_SUM MLIR ===\n{}", module_op_str);

        // Verify scan operation appears
        assert!(
            module_op_str.contains("scan"),
            "Expected scan operation in MLIR output"
        );

        // Verify it has a region with yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in scan region"
        );

        // Verify reverse attribute
        assert!(
            module_op_str.contains("reverse=false") || module_op_str.contains("reverse = false"),
            "Expected reverse=false in scan operation"
        );

        println!("\n✓ scan_sum operation verified (with prefix sum scan region)");
    });
}

#[test]
fn compile_reduce_closure_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "reduce_scan_ops_module",
            "reduce_closure_test_kernel",
            &[128.to_string()],
            &[("input", &[1]), ("result", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
        println!(
            "\n=== REDUCE WITH CLOSURE (SUM) MLIR ===\n{}",
            module_op_str
        );

        // Verify reduce operation appears
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );

        // Verify it has a region with add operation
        assert!(
            module_op_str.contains("addf") || module_op_str.contains("addi"),
            "Expected add operation in reduce region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in reduce region"
        );

        println!("\n✓ reduce with closure (sum) operation verified");
    });
}

#[test]
fn compile_reduce_product_closure_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "reduce_scan_ops_module",
            "reduce_product_closure_test_kernel",
            &[128.to_string()],
            &[("input", &[1]), ("result", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
        println!(
            "\n=== REDUCE WITH CLOSURE (PRODUCT) MLIR ===\n{}",
            module_op_str
        );

        // Verify reduce operation appears
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );

        // Verify it has a region with multiply operation
        assert!(
            module_op_str.contains("mulf") || module_op_str.contains("muli"),
            "Expected multiply operation in reduce region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in reduce region"
        );

        println!("\n✓ reduce with closure (product) operation verified");
    });
}

#[test]
fn compile_reduce_max_closure_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "reduce_scan_ops_module",
            "reduce_max_closure_test_kernel",
            &[128.to_string()],
            &[("input", &[1]), ("result", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let module_op_str = compiler.compile().expect("Failed.").to_string();
        println!(
            "\n=== REDUCE WITH CLOSURE (MAX) MLIR ===\n{}",
            module_op_str
        );

        // Verify reduce operation appears
        assert!(
            module_op_str.contains("reduce"),
            "Expected reduce operation in MLIR output"
        );

        // Verify it has a region with maxf operation
        assert!(
            module_op_str.contains("maxf"),
            "Expected maxf operation in reduce region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in reduce region"
        );

        println!("\n✓ reduce with closure (max) operation verified");
    });
}

#[test]
fn compile_scan_closure_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "reduce_scan_ops_module",
            "scan_closure_test_kernel",
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
        println!(
            "\n=== SCAN WITH CLOSURE (PREFIX PRODUCT) MLIR ===\n{}",
            module_op_str
        );

        // Verify scan operation appears
        assert!(
            module_op_str.contains("scan"),
            "Expected scan operation in MLIR output"
        );

        // Verify it has a region with multiply operation
        assert!(
            module_op_str.contains("mulf") || module_op_str.contains("muli"),
            "Expected multiply operation in scan region"
        );

        // Verify it has a yield
        assert!(
            module_op_str.contains("yield"),
            "Expected yield in scan region"
        );

        // Verify reverse attribute
        assert!(
            module_op_str.contains("reverse=false") || module_op_str.contains("reverse = false"),
            "Expected reverse=false in scan operation"
        );

        println!("\n✓ scan with closure (prefix product) operation verified");
    });
}

/// Regression: ensure `scan(..., reverse=true, ...)` actually lowers to
/// `reverse=true` in the IR. Before the fix, the special-op handler
/// would silently default to `false` if the arg was anything but a bool
/// literal, so we specifically pin the literal-`true` path here.
#[test]
fn compile_scan_reverse_true() {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "reduce_scan_ops_module",
            "scan_reverse_true_kernel",
            &[128.to_string()],
            &[("output", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed.");
        let mlir = compiler.compile().expect("Failed.").to_string();
        println!("\n=== SCAN REVERSE=TRUE MLIR ===\n{mlir}");

        assert!(
            mlir.contains("reverse=true") || mlir.contains("reverse = true"),
            "Expected reverse=true in scan op (was silently dropping non-literal bools \
             before fix; ensure literal `true` still threads through).\nMLIR:\n{mlir}"
        );
    });
}
