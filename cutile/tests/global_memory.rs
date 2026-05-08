/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};

mod common;

#[cutile::module]
mod global_memory_module {
    use cutile::core::*;

    const INITIAL_COUNTER: i32 = 7;

    static COUNTER: Global<i32, { [] }> = Global::new(INITIAL_COUNTER);
    static FLOAT_ACCUM: Global<f32, { [] }> = Global::new(0.0f32);

    #[cutile::entry()]
    fn load_global_kernel(out: &mut Tensor<i32, { [1] }>) {
        let (value, _token) = COUNTER.load(ordering::Acquire, scope::Device);
        out.store(value.reshape(const_shape![1]));
    }

    #[cutile::entry()]
    fn store_global_kernel(out: &mut Tensor<i32, { [1] }>) {
        let value = constant(3i32, const_shape![]);
        let _token = COUNTER.store(value, ordering::Release, scope::Device);
        out.store(value.reshape(const_shape![1]));
    }

    #[cutile::entry()]
    fn atomic_add_global_kernel(out: &mut Tensor<f32, { [1] }>) {
        let increment = constant(1.0f32, const_shape![]);
        let (old, _token) = FLOAT_ACCUM.atomic_add(increment, ordering::AcqRel, scope::Device);
        out.store(old.reshape(const_shape![1]));
    }
}

#[cutile::module]
mod bad_static_module {
    use cutile::core::*;

    static BAD_STATIC: i32 = 0;

    #[cutile::entry()]
    fn kernel(out: &mut Tensor<i32, { [1] }>) {
        let value = constant(1i32, const_shape![1]);
        out.store(value);
    }
}

use bad_static_module::__module_ast_self as bad_static_module_ast;
use global_memory_module::__module_ast_self as global_memory_module_ast;

fn compile_global_kernel(name: &str) -> String {
    let modules = CUDATileModules::from_kernel(global_memory_module_ast())
        .expect("Failed to create CUDATileModules");
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "global_memory_module",
        name,
        &[],
        &[("out", &[1])],
        &[],
        &[],
        None,
        "sm_120".to_string(),
        &CompileOptions::default(),
    )
    .expect("Failed to create compiler.");
    compiler.compile().expect("Failed to compile.").to_string()
}

#[test]
fn global_load_lowers_to_get_global_and_load_ptr() {
    common::with_test_stack(|| {
        let mlir = compile_global_kernel("load_global_kernel");
        assert!(mlir.contains("cuda_tile.global @global_memory_module_COUNTER"));
        assert!(mlir.contains("tile<1xi32>"));
        assert!(mlir.contains("get_global @global_memory_module_COUNTER"));
        assert!(mlir.contains("load_ptr_tko"));
        assert!(mlir.contains("load_ptr_tko acquire device"));
    });
}

#[test]
fn global_store_lowers_to_get_global_and_store_ptr() {
    common::with_test_stack(|| {
        let mlir = compile_global_kernel("store_global_kernel");
        assert!(mlir.contains("cuda_tile.global @global_memory_module_COUNTER"));
        assert!(mlir.contains("tile<1xi32>"));
        assert!(mlir.contains("get_global @global_memory_module_COUNTER"));
        assert!(mlir.contains("store_ptr_tko"));
        assert!(mlir.contains("store_ptr_tko release device"));
    });
}

#[test]
fn global_atomic_add_lowers_to_atomic_rmw() {
    common::with_test_stack(|| {
        let mlir = compile_global_kernel("atomic_add_global_kernel");
        assert!(mlir.contains("cuda_tile.global @global_memory_module_FLOAT_ACCUM"));
        assert!(mlir.contains("tile<1xf32>"));
        assert!(mlir.contains("get_global @global_memory_module_FLOAT_ACCUM"));
        assert!(mlir.contains("atomic_rmw_tko"));
        assert!(mlir.contains("atomic_rmw_tko acq_rel device"));
    });
}

#[test]
fn non_global_static_is_rejected() {
    common::with_test_stack(|| {
        let modules = CUDATileModules::from_kernel(bad_static_module_ast())
            .expect("Failed to create CUDATileModules");
        let compiler = CUDATileFunctionCompiler::new(
            &modules,
            "bad_static_module",
            "kernel",
            &[],
            &[("out", &[1])],
            &[],
            &[],
            None,
            "sm_120".to_string(),
            &CompileOptions::default(),
        )
        .expect("Failed to create compiler.");
        let err = compiler.compile().expect_err("expected static rejection");
        let msg = err.to_string();
        assert!(msg.contains("only `static NAME: Global<E, { [] }>` items are supported"));
    });
}
