/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::{self, api::*, tensor::*, tile_kernel::*};
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod control_flow_ops_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn control_flow_test_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        dynamic_value: i32,
    ) {
        // Test what control flow operations are generated
        let mut sum: Tile<f32, S> = load_tile_mut(output);

        // Test for loop (should generate cuda_tile.for)
        for _i in 0i32..10i32 {
            sum = sum + sum;
        }

        // Test if/else (should generate cuda_tile.if with yield)
        if dynamic_value < 5i32 {
            sum = sum + sum;
        } else {
            sum = sum - sum;
        }

        output.store(sum);
    }

    #[cutile::entry()]
    fn break_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test break statement in infinite loop
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        let mut i: i32 = 0;
        loop {
            sum = sum + sum;
            i = i + 1i32;
            if i >= 2i32 {
                break;
            }
        }
        output.store(sum);
    }

    #[cutile::entry()]
    fn if_return_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>, conditional: bool) {
        let mut val: Tile<i64, S> = output.load();
        let result: Tile<i64, S> = if conditional {
            val = val + val;
            constant(2, val.shape())
        } else {
            val = val + val + val;
            constant(3, val.shape())
        };
        val = val + result;
        output.store(val);
    }

    #[cutile::entry()]
    fn while_loop_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test while loop
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        let mut counter: i32 = 0i32;

        while counter < 10i32 {
            sum = sum + sum;
            counter = counter + 1i32;
        }

        output.store(sum);
    }

    #[cutile::entry()]
    fn infinite_loop_test_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        // Test infinite loop
        let mut sum: Tile<f32, S> = load_tile_mut(output);
        let mut counter: i32 = 0i32;

        // Simulate loop with while instead
        while counter < 10i32 {
            sum = sum + sum;
            counter = counter + 1i32;
        }

        output.store(sum);
    }

    #[cutile::entry()]
    fn assume_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        // Test assume operation - provides optimization hints to compiler
        // Note: Using i64 tensor because bounded predicate only works with integers
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Tell compiler to assume tile values are non-negative (bounded<0, ?>)
        // This can enable additional optimizations
        // Using Melih's const-generic assume_bounds_lower with lower bound of 0
        let assumed_tile: Tile<i64, S> = unsafe { assume_bounds_lower::<_, 0>(tile) };

        // Use the assumed tile
        let result: Tile<i64, S> = assumed_tile + constant(1i64, output.shape());

        output.store(result);
    }

    #[cutile::entry()]
    fn assume_non_negative_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Assume values are non-negative (>= 0) using Melih's const-generic version
        let non_neg_tile: Tile<i64, S> = unsafe { assume_bounds_lower::<_, 0>(tile) };

        let result: Tile<i64, S> = non_neg_tile + constant(1i64, output.shape());
        output.store(result);
    }

    #[cutile::entry()]
    fn assume_div_by_test_kernel<const S: [i32; 1]>(output: &mut Tensor<i64, S>) {
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Assume all values are divisible by 16 (using Melih's const generic version)
        let aligned_tile: Tile<i64, S> = unsafe { assume_div_by::<_, 16>(tile) };

        let result: Tile<i64, S> = aligned_tile + constant(1i64, output.shape());
        output.store(result);
    }

    #[cutile::entry()]
    fn assume_same_elements_test_kernel<const S: [i32; 2]>(output: &mut Tensor<i64, S>) {
        let tile: Tile<i64, S> = load_tile_mut(output);

        // Assume groups of size 2 along dim 0, size 4 along dim 1 have same elements
        // Using Melih's const-generic pattern instead of runtime parameters
        let same_tile: Tile<i64, S> = unsafe { assume_same_elements_2d::<_, 2, 4>(tile) };

        let result: Tile<i64, S> = same_tile + constant(1i64, output.shape());
        output.store(result);
    }
}

use control_flow_ops_module::{_module_asts, break_test_kernel_sync, if_return_test_kernel_sync};

#[test]
fn compile_control_flow_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "control_flow_test_kernel",
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
        println!("\n=== CONTROL FLOW TEST MLIR ===\n{}", module_op_str);

        let has_for = module_op_str.contains(" for ");
        let has_if = module_op_str.contains("if %");
        let has_continue = module_op_str.contains("continue ");
        let has_return = module_op_str.contains("return");

        println!("\n=== Control Flow Operations Found ===");
        println!("for:      {}", if has_for { "✓" } else { "✗" });
        println!("if:       {}", if has_if { "✓" } else { "✗" });
        println!("continue: {}", if has_continue { "✓" } else { "✗" });
        println!("return:   {}", if has_return { "✓" } else { "✗" });

        assert!(has_for, "Expected 'for' loop in MLIR");
        assert!(has_if, "Expected 'if' in MLIR");
        assert!(has_continue, "Expected 'continue' as loop terminator");
        assert!(has_return, "Expected 'return' at function end");

        println!("\n✓ All control flow operations generated (for, if, continue, return)!");
    });
}

#[test]
fn compile_if_result_test() -> () {
    common::with_test_stack(|| {
        let arg: Tensor<i64> = ones([16]).sync().expect("Failed.");
        // If true, double and add 2.
        let (result, _) = if_return_test_kernel_sync(arg.partition([4]), true)
            .sync()
            .expect("Failed.");
        let result: Vec<i64> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert!(result.iter().all(|x| *x == 4));

        // If false, triple and add 3.
        let arg: Tensor<i64> = ones([16]).sync().expect("Failed.");
        let (result, _) = if_return_test_kernel_sync(arg.partition([4]), false)
            .sync()
            .expect("Failed.");
        let result: Vec<i64> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert!(result.iter().all(|x| *x == 6));
    });
}

#[test]
fn execute_break_test() -> () {
    common::with_test_stack(|| {
        // break_test_kernel loads output, doubles it twice (loop runs 2 iterations then breaks),
        // and stores the result. Starting from 1.0, we expect 1.0 * 2 * 2 = 4.0.
        let arg: Tensor<f32> = ones([16]).sync().expect("Failed.");
        let (result,) = break_test_kernel_sync(arg.partition([4]))
            .sync()
            .expect("Failed.");
        let result: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");
        assert!(
            result.iter().all(|x| *x == 4.0),
            "Expected all elements to be 4.0, got: {:?}",
            result
        );
    });
}

#[test]
fn compile_break_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "break_test_kernel",
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
        println!("\n=== BREAK TEST MLIR ===\n{}", module_op_str);

        let has_loop = module_op_str.contains(" loop ");
        let has_break = module_op_str.contains("break ");

        assert!(has_loop, "Expected loop operation in MLIR");
        assert!(has_break, "Expected break operation in MLIR");

        println!("\n✓ break statement compiled successfully");
    });
}

#[test]
fn compile_while_loop_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "while_loop_test_kernel",
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
        println!("\n=== WHILE LOOP TEST MLIR ===\n{}", module_op_str);

        let has_loop = module_op_str.contains("cuda_tile.loop") || module_op_str.contains(" loop ");
        let has_break = module_op_str.contains("break ");

        assert!(has_loop, "Expected cuda_tile.loop operation in MLIR");
        assert!(has_break, "Expected break operation for while loop exit");

        println!("\n✓ while loop compiled to cuda_tile.loop with break!");
    });
}

#[test]
fn compile_loop_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "infinite_loop_test_kernel",
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
        println!("\n=== LOOP TEST MLIR ===\n{}", module_op_str);

        let has_loop = module_op_str.contains("cuda_tile.loop") || module_op_str.contains(" loop ");
        let has_break = module_op_str.contains("break ");

        assert!(has_loop, "Expected cuda_tile.loop operation in MLIR");
        assert!(has_break, "Expected break operation for loop exit");

        println!("\n✓ loop expression compiled to cuda_tile.loop with break!");
    });
}

#[test]
fn compile_assume_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "assume_test_kernel",
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
        println!("\n=== ASSUME MLIR ===\n{}", module_op_str);

        // Verify assume operation appears
        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );

        // Verify it has the bounded predicate attribute (bounded<0, ?> = non-negative)
        assert!(
            module_op_str.contains("bounded"),
            "Expected bounded predicate on assume operation"
        );

        println!(
            "\n✓ assume operation verified (compiler optimization hint with bounded predicate)"
        );
    });
}

#[test]
fn compile_assume_non_negative_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "assume_non_negative_test_kernel",
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
        println!("\n=== ASSUME_NON_NEGATIVE MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );
        assert!(
            module_op_str.contains("bounded<0, ?>"),
            "Expected bounded<0, ?> predicate on assume operation"
        );

        println!("\n✓ assume_non_negative operation verified with bounded<0, ?>");
    });
}

#[test]
fn compile_assume_div_by_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "assume_div_by_test_kernel",
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
        println!("\n=== ASSUME_DIV_BY MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );
        assert!(
            module_op_str.contains("div_by<16>"),
            "Expected div_by<16> predicate on assume operation"
        );

        println!("\n✓ assume_div_by operation verified with div_by<16>");
    });
}

#[test]
fn compile_assume_same_elements_test() -> () {
    common::with_test_stack(|| {
        let modules =
            CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "control_flow_ops_module",
            "assume_same_elements_test_kernel",
            &[4.to_string(), 8.to_string()],
            &[("output", &[2, 2])],
            None,
            gpu_name,
        )
        .expect("Failed.");
        let module_op_str = compiler
            .compile()
            .expect("Failed.")
            .as_operation()
            .to_string();
        println!("\n=== ASSUME_SAME_ELEMENTS MLIR ===\n{}", module_op_str);

        assert!(
            module_op_str.contains("assume"),
            "Expected assume operation in MLIR output"
        );
        assert!(
            module_op_str.contains("same_elements<[2, 4]>"),
            "Expected same_elements<[2, 4]> predicate on assume operation"
        );

        println!("\n✓ assume_same_elements operation verified with same_elements<[2, 4]>");
    });
}
