/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT-compile tests for `T::ZERO` usage through a generic type parameter.
//!
//! Verifies the cutile-compiler resolves `T::ZERO` against the monomorphized
//! type in the `constant` op's dense attribute. This was broken until the
//! `generic_args.inst_types` lookup was added to the Expr::Path branch in
//! `compile_cuda_tile_op.rs`.

use cutile;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;

mod common;

#[cutile::module]
mod zero_kernel_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn zero_fill<T: ElementType, const BM: i32, const BN: i32>(out: &mut Tensor<T, { [BM, BN] }>) {
        let z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        out.store(z);
    }
}

use zero_kernel_module::_module_asts;

fn compile_zero_fill(ty: &str) -> String {
    let modules = CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "zero_kernel_module",
        "zero_fill",
        &[ty.to_string(), 64.to_string(), 64.to_string()],
        &[("out", &[64, 1])],
        &[],
        &[],
        None,
        gpu_name,
        &CompileOptions::default(),
    )
    .expect("Failed to create compiler");
    let module_op_str = compiler.compile().expect("Failed to compile").to_string();
    println!("=== MLIR for zero_fill<{ty}> ===\n{module_op_str}");
    module_op_str
}

#[test]
fn zero_fill_f32_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("f32");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("f32"), "expected f32 element type in MLIR");
    });
}

#[test]
fn zero_fill_f16_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("f16");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("f16"), "expected f16 element type in MLIR");
    });
}

#[test]
fn zero_fill_bf16_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("bf16");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("bf16"), "expected bf16 element type in MLIR");
    });
}

#[test]
fn zero_fill_i32_resolves_t_zero() {
    common::with_test_stack(|| {
        let mlir = compile_zero_fill("i32");
        assert!(mlir.contains("constant"), "expected constant op");
        assert!(mlir.contains("i32"), "expected i32 element type in MLIR");
    });
}
