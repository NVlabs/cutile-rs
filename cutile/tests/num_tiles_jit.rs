/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT-compile tests for `num_tiles(view, axis)`.
//!
//! Verifies that the intrinsic handler in `compile_intrinsic.rs` emits
//! `cuda_tile.get_index_space_shape` against the partition view and extracts
//! the axis-th result. No GPU execution — these tests only cover the
//! DSL → IR lowering path.

use cutile;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod num_tiles_kernels {
    use cutile::core::*;

    /// Calls `num_tiles` along both axes of a rank-2 partition.
    #[cutile::entry()]
    fn num_tiles_2d<const BM: i32, const BN: i32>(
        input: &Tensor<f32, { [-1, -1] }>,
        out: &mut Tensor<i32, { [1] }>,
    ) {
        let part = input.partition(const_shape![BM, BN]);
        let nm: i32 = num_tiles(&part, 0);
        let nn: i32 = num_tiles(&part, 1);
        // Fold the pair of axis counts into a single scalar so the kernel
        // references both values (prevents DCE from dropping either call).
        let combined: Tile<i32, { [1] }> = broadcast_scalar(nm * 100i32 + nn, const_shape![1]);
        out.store(combined);
    }

    /// Calls `num_tiles` along axis 2 of a rank-3 partition.
    #[cutile::entry()]
    fn num_tiles_3d_axis2<const BM: i32, const BN: i32, const BK: i32>(
        input: &Tensor<f32, { [-1, -1, -1] }>,
        out: &mut Tensor<i32, { [1] }>,
    ) {
        let part = input.partition(const_shape![BM, BN, BK]);
        let nk: i32 = num_tiles(&part, 2);
        let tile: Tile<i32, { [1] }> = broadcast_scalar(nk, const_shape![1]);
        out.store(tile);
    }

    /// Directly exercises the raw Tile IR view-shape query surfaces.
    #[cutile::entry()]
    fn raw_view_shape_queries<const BM: i32, const BN: i32>(
        input: &Tensor<f32, { [-1, -1] }>,
        out: &mut Tensor<i32, { [1] }>,
    ) {
        let tensor_shape: [i32; 2] = get_tensor_shape(input);
        let part = input.partition(const_shape![BM, BN]);
        let index_shape: [i32; 2] = get_index_space_shape(&part);
        let combined: i32 = tensor_shape[0] + tensor_shape[1] + index_shape[0] + index_shape[1];
        let tile: Tile<i32, { [1] }> = broadcast_scalar(combined, const_shape![1]);
        out.store(tile);
    }

    /// Uses a dynamic `num_tiles` loop bound over a statically-shaped tensor.
    ///
    /// The returned value must remain the SSA result from
    /// `get_index_space_shape`, but it should carry exact static bounds so the
    /// partition load inside the loop is proven safe at compile time.
    #[cutile::entry()]
    fn static_num_tiles_loop_bounds<const BK: i32, const K: i32>(
        input: &Tensor<f32, { [K] }>,
        out: &mut Tensor<f32, { [BK] }>,
    ) {
        let part = input.partition(const_shape![BK]);
        let nk: i32 = num_tiles(&part, 0);
        let mut tile: Tile<f32, { [BK] }> = broadcast_scalar(0.0f32, const_shape![BK]);
        for i in 0i32..nk {
            tile = tile + part.load([i]);
        }
        out.store(tile);
    }
}

use num_tiles_kernels::__module_ast_self;

fn compile(kernel: &str, gen_args: &[String], strides: &[(&str, &[i32])]) -> String {
    let modules = CUDATileModules::from_kernel(__module_ast_self())
        .expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "num_tiles_kernels",
        kernel,
        gen_args,
        strides,
        &[],
        &[],
        None,
        gpu_name,
        &CompileOptions::default(),
    )
    .expect("Failed to create compiler");
    let mlir = compiler.compile().expect("Failed to compile").to_string();
    println!("=== MLIR for {kernel} ===\n{mlir}");
    mlir
}

#[test]
fn rank2_emits_get_index_space_shape() {
    common::with_test_stack(|| {
        let mlir = compile(
            "num_tiles_2d",
            &[64.to_string(), 64.to_string()],
            &[("input", &[-1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `get_index_space_shape` op in emitted MLIR"
        );
        assert!(
            mlir.contains("partition_view"),
            "expected partition_view operand in the op's input"
        );
    });
}

#[test]
fn rank3_emits_get_index_space_shape() {
    common::with_test_stack(|| {
        let mlir = compile(
            "num_tiles_3d_axis2",
            &[32.to_string(), 32.to_string(), 32.to_string()],
            &[("input", &[-1, -1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `get_index_space_shape` op in emitted MLIR"
        );
    });
}

#[test]
fn raw_view_shape_queries_emit_view_ops() {
    common::with_test_stack(|| {
        let mlir = compile(
            "raw_view_shape_queries",
            &[64.to_string(), 32.to_string()],
            &[("input", &[-1, -1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_tensor_shape"),
            "expected `get_tensor_shape` op in emitted MLIR"
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `get_index_space_shape` op in emitted MLIR"
        );
    });
}

#[test]
fn static_num_tiles_loop_keeps_dynamic_upper_bound_but_proves_access() {
    common::with_test_stack(|| {
        let mlir = compile(
            "static_num_tiles_loop_bounds",
            &[8.to_string(), 32.to_string()],
            &[("input", &[1]), ("out", &[1])],
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected `num_tiles` to lower to a dynamic `get_index_space_shape` result"
        );
        let num_tiles_value = mlir
            .lines()
            .find(|line| line.contains("get_index_space_shape"))
            .and_then(|line| {
                line.split_once('=')
                    .map(|(value, _)| value.trim().to_string())
            })
            .expect("expected to find the value produced by `get_index_space_shape`");
        let for_line = mlir
            .lines()
            .find(|line| line.contains(" = for "))
            .expect("expected a Tile IR loop");
        assert!(
            for_line.contains(&format!(" to {num_tiles_value},")),
            "expected the loop upper bound to use the dynamic `num_tiles` result; loop was `{for_line}`"
        );
        assert!(
            !mlir.contains("partition access out of bounds"),
            "expected static bounds on `num_tiles` to discharge the partition access check"
        );
    });
}
