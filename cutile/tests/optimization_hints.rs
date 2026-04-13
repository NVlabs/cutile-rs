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
mod opt_hints_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn load_ptr_latency_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);
        let (loaded, _tok): (Tile<f32, S>, Token) =
            load_ptr_tko(ptrs, "weak", "tl_blk", None, None, None, Some(4));
        output.store(loaded);
    }

    #[cutile::entry()]
    fn store_ptr_latency_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);
        let vals: Tile<f32, S> = constant(1.0f32, output.shape());
        let _tok: Token = store_ptr_tko(ptrs, vals, "weak", "tl_blk", None, None, Some(2));
        output.store(vals);
    }

    #[cutile::entry()]
    fn load_view_latency_kernel<const S: [i32; 1]>(input: &Tensor<f32, S>) {
        let token: Token = new_token_unordered();
        let shape = input.shape();
        let partition: Partition<f32, S> = make_partition_view(input, shape, token);
        let idx: [i32; 1] = [0i32];
        let _tile: Tile<f32, S> = load_from_view(&partition, idx, Some(8), false);
    }

    #[cutile::entry()]
    fn store_view_disallow_tma_kernel<const S: [i32; 1]>(y: &mut Tensor<f32, S>) {
        let shape = y.shape();
        let token: Token = get_tensor_token(y);
        let mut partition: PartitionMut<f32, S> =
            unsafe { make_partition_view_mut(y, shape, token) };
        let tile: Tile<f32, S> = constant(1.0f32, shape);
        let idx: [i32; 1] = [0i32];
        unsafe {
            store_to_view_mut(&mut partition, tile, idx, None, true);
        }
    }

    #[cutile::entry(optimization_hints = (
        sm_120 = (occupancy = 4, num_cta_in_cga = 2),
    ))]
    fn entry_hints_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }
}

use opt_hints_module::_module_asts;

fn compile_kernel(name: &str, strides: &[(&str, &[i32])], options: &CompileOptions) -> String {
    let modules = CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "opt_hints_module",
        name,
        &[128.to_string()],
        strides,
        &[],
        &[],
        None,
        gpu_name,
        options,
    )
    .expect("Failed to create compiler");
    let module_op = compiler.compile().expect("Failed to compile");
    let result = module_op.as_operation().to_string();
    drop(module_op);
    drop(compiler);
    result
}

#[test]
fn load_ptr_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "load_ptr_latency_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 4"),
            "Expected latency=4 in load_ptr_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn store_ptr_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "store_ptr_latency_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 2"),
            "Expected latency=2 in store_ptr_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn load_view_latency_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "load_view_latency_kernel",
            &[("input", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 8"),
            "Expected latency=8 in load_view_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn store_view_disallow_tma_hint_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "store_view_disallow_tma_kernel",
            &[("y", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("allow_tma = false"),
            "Expected allow_tma=false in store_view_tko optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn entry_level_occupancy_hints_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("occupancy = 4"),
            "Expected occupancy=4 in entry optimization_hints.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("num_cta_in_cga = 2"),
            "Expected num_cta_in_cga=2 in entry optimization_hints.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn compile_options_override_entry_hints() {
    common::with_test_stack(|| {
        let options = CompileOptions::default().occupancy(8).num_cta_in_cga(4);
        let mlir = compile_kernel("entry_hints_kernel", &[("output", &[1])], &options);
        println!("{mlir}");
        assert!(
            mlir.contains("occupancy = 8"),
            "Expected runtime occupancy=8 to override entry-level occupancy=4.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("num_cta_in_cga = 4"),
            "Expected runtime num_cta_in_cga=4 to override entry-level num_cta_in_cga=2.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn different_compile_options_produce_different_mlir() {
    common::with_test_stack(|| {
        let mlir_a = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default().occupancy(2),
        );
        let mlir_b = compile_kernel(
            "entry_hints_kernel",
            &[("output", &[1])],
            &CompileOptions::default().occupancy(16),
        );
        assert!(
            mlir_a.contains("occupancy = 2"),
            "First compilation should have occupancy=2.\nMLIR:\n{mlir_a}"
        );
        assert!(
            mlir_b.contains("occupancy = 16"),
            "Second compilation should have occupancy=16.\nMLIR:\n{mlir_b}"
        );
        assert_ne!(
            mlir_a, mlir_b,
            "Different CompileOptions should produce different MLIR"
        );
    });
}
