/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile;
use cutile::compile_api::KernelCompiler;

mod common;

#[cutile::module]
mod compile_only_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn tile_math<const S: [i32; 1]>(output: &mut Tensor<f32, S>, scalar: f32) {
        let scalar_tile: Tile<f32, S> = broadcast_scalar(scalar, output.shape());
        let ones: Tile<f32, S> = broadcast_scalar(1.0f32, output.shape());
        output.store(scalar_tile + ones);
    }
}

#[test]
fn kernel_compiler_emits_ir_and_bytecode() {
    common::with_test_stack(|| {
        let artifacts = KernelCompiler::new(
            compile_only_module::_module_asts,
            "compile_only_module",
            "tile_math",
        )
        .generics(vec!["32".into()])
        .strides(&[("output", &[1])])
        .target("sm_80")
        .compile()
        .expect("compile-only kernel compilation failed");

        let ir = artifacts.ir_text();
        assert!(!ir.trim().is_empty(), "expected non-empty Tile IR");
        assert!(
            ir.contains("entry"),
            "expected the compiled IR to contain an entry op.\nIR:\n{ir}"
        );

        let bytecode = artifacts
            .bytecode()
            .expect("bytecode serialization should succeed");
        assert!(!bytecode.is_empty(), "expected non-empty bytecode");
        assert_eq!(
            &bytecode[..8],
            &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00],
            "expected TileIR bytecode magic"
        );
    });
}
