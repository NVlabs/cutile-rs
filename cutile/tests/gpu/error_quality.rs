/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU-dependent error-quality tests.

use cutile;
use cutile_compiler::ast::Module;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;
use cutile_compiler::error::JITError;

use crate::common;

const FORBIDDEN_INTERNALS: &[&str] = &[
    "TileRustValue",
    "TileRustType",
    "TypeMeta",
    "Kind::Compound",
    "Kind::Struct",
    "Kind::PrimitiveType",
    "Kind::StructuredType",
    "Kind::String",
    "get_concrete_op_ident_from_types",
];

fn assert_no_internal_leaks(text: &str, context: &str) {
    for &forbidden in FORBIDDEN_INTERNALS {
        assert!(
            !text.contains(forbidden),
            "{context}: error message must not expose internal name `{forbidden}`.\n  \
             Full message: {text}"
        );
    }
}

fn assert_single_error_prefix(text: &str, context: &str) {
    assert!(
        text.starts_with("error: "),
        "{context}: missing outer error prefix"
    );
    assert!(
        !text.starts_with("error: error: "),
        "{context}: 'error: ' prefix is doubled.\n  Full message: {text}"
    );
}

fn assert_jit_error_has_no_prefix(err: &JITError, context: &str) {
    let output = format!("{err}");
    assert!(
        !output.starts_with("error: "),
        "{context}: JITError must NOT start with 'error: '.\n  Got: {output}"
    );
}

fn assert_display_eq_debug_jit(err: &JITError, context: &str) {
    let display = format!("{err}");
    let debug = format!("{err:?}");
    assert_eq!(
        display, debug,
        "{context}: Display and Debug must be identical.\n  Display: {display}\n  Debug:   {debug}"
    );
}

#[cutile::module]
mod error_quality_untyped_literal {
    use cutile::core::*;

    #[cutile::entry()]
    fn untyped_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) {
        let _x = 42;
        let tile = load_tile_mut(output);
        output.store(tile);
    }
}

fn compile_and_get_error(kernel: Module, module_name: &str, function_name: &str) -> JITError {
    let modules = CUDATileModules::from_kernel(kernel).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        module_name,
        function_name,
        &[128.to_string()],
        &[("output", &[1])],
        &[],
        &[],
        None,
        gpu_name,
        &CompileOptions::default(),
    )
    .expect("Compiler construction should succeed");

    let result = compiler.compile();
    let err = match result {
        Err(e) => Some(e),
        Ok(_) => None,
    };
    err.unwrap_or_else(|| {
        panic!("Expected compilation of {module_name}::{function_name} to fail, but it succeeded.")
    })
}

#[test]
fn untyped_literal_error_message_quality() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_untyped_literal::__module_ast_self(),
            "error_quality_untyped_literal",
            "untyped_kernel",
        );

        let display = format!("{err}");
        let debug = format!("{err:?}");

        assert_no_internal_leaks(&display, "untyped literal (Display)");
        assert_no_internal_leaks(&debug, "untyped literal (Debug)");
        assert_display_eq_debug_jit(&err, "untyped literal");
        assert_jit_error_has_no_prefix(&err, "untyped literal");
        assert!(
            display.contains("42")
                || display.contains("type")
                || display.contains("annotation")
                || display.contains("literal")
        );

        match &err {
            JITError::Located(msg, loc) => {
                assert!(loc.is_known());
                assert!(loc.file.ends_with("gpu/error_quality.rs"));
                assert!(display.contains("-->"));
                assert_no_internal_leaks(msg, "untyped literal (Located msg)");
            }
            JITError::Generic(msg) => {
                assert_no_internal_leaks(msg, "untyped literal (Generic msg)");
            }
            _ => {
                assert_no_internal_leaks(&display, "untyped literal (other variant)");
            }
        }

        let outer: cutile::error::Error = err.into();
        let outer_display = format!("{outer}");
        assert_single_error_prefix(&outer_display, "untyped literal (outer)");
    });
}

#[test]
fn untyped_literal_error_location_points_to_this_file() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_untyped_literal::__module_ast_self(),
            "error_quality_untyped_literal",
            "untyped_kernel",
        );

        match &err {
            JITError::Located(_msg, loc) => {
                assert!(loc.is_known(), "Error should have a known source location");
                assert!(
                    loc.file.ends_with("gpu/error_quality.rs"),
                    "Error location file should end with 'gpu/error_quality.rs', got: '{}'",
                    loc.file
                );

                let source = include_str!("error_quality.rs");
                let target_line = source
                    .lines()
                    .enumerate()
                    .find(|(_, line)| {
                        let trimmed = line.trim_start();
                        trimmed.starts_with("let _x = 42;")
                    })
                    .map(|(idx, _)| idx + 1);

                if let Some(expected_line) = target_line {
                    assert_eq!(loc.line, expected_line);
                }

                assert!(loc.column > 0);
            }
            _ => {
                panic!("Expected a Located JIT error, got: {err}");
            }
        }
    });
}
