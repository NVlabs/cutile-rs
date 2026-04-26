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
        let (loaded, _tok): (Tile<f32, S>, Token) = load_ptr_tko(
            ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<4>,
        );
        output.store(loaded);
    }

    #[cutile::entry()]
    fn store_ptr_latency_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);
        let vals: Tile<f32, S> = constant(1.0f32, output.shape());
        let _tok: Token = store_ptr_tko(
            ptrs,
            vals,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<2>,
        );
        output.store(vals);
    }

    #[cutile::entry()]
    fn option_bindings_for_optional_operands_kernel<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
    ) {
        let ptr_seed: Tile<i64, S> = constant(0i64, output.shape());
        let ptrs_i64: PointerTile<*mut i64, S> = int_to_ptr(ptr_seed);
        let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs_i64);

        let scope_opt: Option<scope::Device> = Some(scope::Device);
        let mask_opt: Option<Tile<bool, S>> = None;
        let padding_opt: Option<f32> = None;
        let token_none: Option<Token> = None;
        let (loaded, _tok): (Tile<f32, S>, Token) = load_ptr_tko(
            ptrs,
            ordering::Relaxed,
            scope_opt,
            mask_opt,
            padding_opt,
            token_none,
            Latency::<0>,
        );

        let token: Token = new_token_unordered();
        let token_some: Option<Token> = Some(token);
        let scope_none: Option<scope::TileBlock> = None;
        let _store_tok: Token = store_ptr_tko(
            ptrs,
            loaded,
            ordering::Weak,
            scope_none,
            None,
            token_some,
            Latency::<0>,
        );

        output.store(loaded);
    }

    #[cutile::entry()]
    fn load_view_latency_kernel<const S: [i32; 1]>(input: &Tensor<f32, S>) {
        let token: Token = new_token_unordered();
        let shape = input.shape();
        let partition: Partition<f32, S> =
            make_partition_view(input, shape, padding::None, dim_map::Identity, token);
        let idx: [i32; 1] = [0i32];
        let _tile: Tile<f32, S> = load_view_tko(
            &partition,
            idx,
            ordering::Weak,
            scope::TileBlock,
            Some(8),
            tma::Enabled,
        );
    }

    #[cutile::entry()]
    fn store_view_disallow_tma_kernel<const S: [i32; 1]>(y: &mut Tensor<f32, S>) {
        let shape = y.shape();
        let token: Token = get_tensor_token(y);
        let mut partition: PartitionMut<f32, S> =
            unsafe { make_partition_view_mut(y, shape, padding::None, token) };
        let tile: Tile<f32, S> = constant(1.0f32, shape);
        let idx: [i32; 1] = [0i32];
        unsafe {
            store_view_tko_mut(
                &mut partition,
                tile,
                idx,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Disabled,
            );
        }
    }

    #[cutile::entry(optimization_hints = (
        sm_120 = (occupancy = 4, num_cta_in_cga = 2),
    ))]
    fn entry_hints_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    /// Latency as a const generic — specialized at launch time.
    #[cutile::entry()]
    fn load_view_const_latency_kernel<const S: [i32; 1], const L: i32>(input: &Tensor<f32, S>) {
        let token: Token = new_token_unordered();
        let shape = input.shape();
        let partition: Partition<f32, S> =
            make_partition_view(input, shape, padding::None, dim_map::Identity, token);
        let idx: [i32; 1] = [0i32];
        let _tile: Tile<f32, S> = load_view_tko(
            &partition,
            idx,
            ordering::Weak,
            scope::TileBlock,
            Some(L),
            tma::Enabled,
        );
    }
}

use opt_hints_module::__module_ast_self;

fn compile_kernel(name: &str, strides: &[(&str, &[i32])], options: &CompileOptions) -> String {
    let modules = CUDATileModules::from_kernel(__module_ast_self())
        .expect("Failed to create CUDATileModules");
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
    let result = module_op.to_string();
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
fn option_bindings_for_optional_operands_in_mlir() {
    common::with_test_stack(|| {
        let mlir = compile_kernel(
            "option_bindings_for_optional_operands_kernel",
            &[("output", &[1])],
            &CompileOptions::default(),
        );
        println!("{mlir}");
        assert!(
            mlir.contains("load_ptr_tko relaxed device"),
            "Expected Option-bound device scope on load_ptr_tko.\nMLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("store_ptr_tko weak") && mlir.contains("token=%"),
            "Expected Option-bound token operand on store_ptr_tko.\nMLIR:\n{mlir}"
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

#[test]
fn load_view_const_latency_in_mlir() {
    // Latency as a const generic: L=5 should appear as `latency = 5` in MLIR.
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(__module_ast_self())
            .expect("Failed to create CUDATileModules");
        let gpu_name = get_gpu_name(0);
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "opt_hints_module",
            "load_view_const_latency_kernel",
            &[128.to_string(), 5.to_string()], // S=128, L=5
            &[("input", &[1])],
            &[],
            &[],
            None,
            gpu_name,
            &CompileOptions::default(),
        )
        .expect("Failed to create compiler");
        let module_op = compiler.compile().expect("Failed to compile");
        let mlir = module_op.to_string();
        drop(module_op);
        drop(compiler);
        println!("{mlir}");
        assert!(
            mlir.contains("latency = 5"),
            "Expected latency=5 from const generic L=5.\nMLIR:\n{mlir}"
        );
    });
}
