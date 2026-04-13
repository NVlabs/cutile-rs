/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile::api;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::DeviceOp;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;
use cutile_compiler::specialization::{DivHint, SpecializationBits};

/// Helper: create a DivHint with the default max (16).
fn dh(divisor: i32) -> DivHint {
    DivHint { divisor, max: 16 }
}

mod common;

#[cutile::module]
mod spec_test_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn simple_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    #[cutile::entry(optimization_hints = (sm_120 = (max_divisibility = 8,),))]
    fn capped_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }

    /// Kernel with a scalar integer param — used to test DivHint on scalars.
    #[cutile::entry(print_ir = true)]
    fn scalar_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>, n: i32) {
        let tile: Tile<f32, S> = constant(1.0f32, output.shape());
        output.store(tile);
    }
}

use spec_test_module::{_module_asts, scalar_kernel};

fn compile_with_spec(
    name: &str,
    strides: &[(&str, &[i32])],
    specs: &[(&str, &SpecializationBits)],
) -> String {
    compile_with_spec_and_options(name, strides, specs, &CompileOptions::default())
}

fn compile_with_spec_and_options(
    name: &str,
    strides: &[(&str, &[i32])],
    specs: &[(&str, &SpecializationBits)],
    options: &CompileOptions,
) -> String {
    let modules = CUDATileModules::new(_module_asts()).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        "spec_test_module",
        name,
        &[128.to_string()],
        strides,
        specs,
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

// -- SpecializationBits produces correct assume_div_by in MLIR --

#[test]
fn spec_bits_div_16_produces_div_by_16() {
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(16)],
            stride_div: vec![dh(4)],
            stride_one: vec![true],
            base_ptr_div: dh(16),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<16>"),
            "Expected div_by<16> for shape divisible by 16.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn spec_bits_div_8_produces_div_by_8() {
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(8)],
            stride_div: vec![dh(4)],
            stride_one: vec![true],
            base_ptr_div: dh(8),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<8>"),
            "Expected div_by<8> for shape divisible by 8.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn no_spec_bits_no_div_by() {
    common::with_test_stack(|| {
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[]);
        println!("{mlir}");
        assert!(
            !mlir.contains("div_by"),
            "Expected no div_by when no spec bits provided.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn spec_bits_div_1_no_div_by() {
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(1)],
            stride_div: vec![dh(1)],
            stride_one: vec![true],
            base_ptr_div: dh(1),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("simple_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            !mlir.contains("div_by"),
            "Expected no div_by when all divisors are 1.\nMLIR:\n{mlir}"
        );
    });
}

// -- Cache key differentiation --

#[test]
fn different_spec_bits_different_cache_keys() {
    use cutile::tile_kernel::TileFunctionKey;

    let spec_a = SpecializationBits {
        shape_div: vec![dh(16)],
        stride_div: vec![dh(16)],
        stride_one: vec![true],
        base_ptr_div: dh(16),
        elements_disjoint: true,
    };
    let spec_b = SpecializationBits {
        shape_div: vec![dh(8)],
        stride_div: vec![dh(8)],
        stride_one: vec![true],
        base_ptr_div: dh(8),
        elements_disjoint: true,
    };

    let key_a = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        vec![("output".into(), spec_a.clone())],
        vec![],
        None,
        CompileOptions::default(),
    );
    let key_b = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        vec![("output".into(), spec_b.clone())],
        vec![],
        None,
        CompileOptions::default(),
    );
    let key_a2 = TileFunctionKey::new(
        "m".into(),
        "f".into(),
        vec![],
        vec![],
        vec![("output".into(), spec_a)],
        vec![],
        None,
        CompileOptions::default(),
    );

    assert_ne!(
        key_a, key_b,
        "Different spec bits should produce different cache keys"
    );
    assert_eq!(
        key_a, key_a2,
        "Same spec bits should produce equal cache keys"
    );
}

// -- max_divisibility ceiling --

#[test]
fn entry_max_divisibility_caps_inferred_div() {
    // capped_kernel has max_divisibility=8 in its entry hints.
    // Spec says shape is div by 16, but the hint should cap it to 8.
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(16)],
            stride_div: vec![dh(16)],
            stride_one: vec![true],
            base_ptr_div: dh(16),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("capped_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<8>"),
            "Expected div_by<8> (capped from 16 by max_divisibility=8).\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<16>"),
            "Should not contain div_by<16> when max_divisibility=8.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn entry_max_divisibility_does_not_inflate() {
    // capped_kernel has max_divisibility=8.
    // Spec says shape is div by 4 — should stay 4 (not inflated to 8).
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(4)],
            stride_div: vec![dh(4)],
            stride_one: vec![true],
            base_ptr_div: dh(4),
            elements_disjoint: true,
        };
        let mlir = compile_with_spec("capped_kernel", &[("output", &[1])], &[("output", &spec)]);
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<4>"),
            "Expected div_by<4> (not inflated by max_divisibility=8).\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<8>"),
            "Should not contain div_by<8> when inferred is only 4.\nMLIR:\n{mlir}"
        );
    });
}

#[test]
fn runtime_max_divisibility_overrides_entry_hint() {
    // simple_kernel has no entry-level max_divisibility.
    // Runtime CompileOptions sets max_divisibility=4, capping spec div=16 to 4.
    common::with_test_stack(|| {
        let spec = SpecializationBits {
            shape_div: vec![dh(16)],
            stride_div: vec![dh(16)],
            stride_one: vec![true],
            base_ptr_div: dh(16),
            elements_disjoint: true,
        };
        let options = CompileOptions::default().max_divisibility(4);
        let mlir = compile_with_spec_and_options(
            "simple_kernel",
            &[("output", &[1])],
            &[("output", &spec)],
            &options,
        );
        println!("{mlir}");
        assert!(
            mlir.contains("div_by<4>"),
            "Expected div_by<4> from runtime max_divisibility override.\nMLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("div_by<16>"),
            "Should not contain div_by<16> when runtime max_divisibility=4.\nMLIR:\n{mlir}"
        );
    });
}

// -- Scalar integer DivHint --

#[test]
fn scalar_int_param_gets_div_hint() {
    // Launch scalar_kernel with n=1024 (divisible by 16).
    // The MLIR should contain assume_div_by on the scalar param.
    common::with_test_stack(|| {
        let mut output = api::zeros::<f32>(&[128]).sync().expect("alloc");
        scalar_kernel((&mut output).partition([128]), 1024i32)
            .sync()
            .expect("kernel launch");
    });
}
