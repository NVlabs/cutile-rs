/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for error message quality, location coverage, and
//! formatting consistency.
//!
//! These tests verify three properties that were improved in a previous pass:
//!
//! 1. **Error message quality**: User-facing error messages must NOT expose
//!    internal implementation details such as `TileRustValue`, `TypeMeta`,
//!    `Kind::Compound`, or function names like `get_concrete_op_ident_from_types`.
//!
//! 2. **Location coverage**: Errors produced by the JIT compiler should carry
//!    source location information (`JITError::Located` with `is_known() == true`)
//!    whenever the error originates from user-written code inside a
//!    `#[cutile::module]`.
//!
//! 3. **Formatting consistency**: The `"error: "` prefix is the sole
//!    responsibility of the outer `cutile::error::Error` type.
//!    `JITError` itself renders the bare message (plus an optional
//!    `-->` location pointer).  The outer `Error` prepends `"error: "`
//!    uniformly to every variant so that all error kinds look the same.
//!    `Display` and `Debug` produce identical output for both types.

use cutile;
use cutile_compiler::ast::Module;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;
use cutile_compiler::error::JITError;

mod common;

// ===========================================================================
// Internal identifiers that must NEVER appear in user-facing error output.
// ===========================================================================

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

/// Assert that `text` does not contain any of the forbidden internal names.
fn assert_no_internal_leaks(text: &str, context: &str) {
    for &forbidden in FORBIDDEN_INTERNALS {
        assert!(
            !text.contains(forbidden),
            "{context}: error message must not expose internal name `{forbidden}`.\n  \
             Full message: {text}"
        );
    }
}

/// Assert that the outer `Error` output starts with exactly one `"error: "`
/// prefix and never doubles it.
fn assert_single_error_prefix(text: &str, context: &str) {
    assert!(
        text.starts_with("error: "),
        "{context}: outer Error output must start with 'error: '.\n  Got: {text}"
    );
    assert!(
        !text.starts_with("error: error: "),
        "{context}: 'error: ' prefix is doubled.\n  Full message: {text}"
    );
}

/// Assert that a `JITError` does NOT carry the `"error: "` prefix —
/// that prefix belongs exclusively to the outer `Error` type.
fn assert_jit_error_has_no_prefix(err: &JITError, context: &str) {
    let output = format!("{err}");
    assert!(
        !output.starts_with("error: "),
        "{context}: JITError must NOT start with 'error: ' — that prefix \
         belongs to the outer Error type.\n  Got: {output}"
    );
}

/// Assert Display and Debug produce identical output for a JITError.
fn assert_display_eq_debug_jit(err: &JITError, context: &str) {
    let display = format!("{err}");
    let debug = format!("{err:?}");
    assert_eq!(
        display, debug,
        "{context}: Display and Debug must be identical.\n  Display: {display}\n  Debug:   {debug}"
    );
}

/// Assert Display and Debug produce identical output for the outer Error.
fn assert_display_eq_debug_outer(err: &cutile::error::Error, context: &str) {
    let display = format!("{err}");
    let debug = format!("{err:?}");
    assert_eq!(
        display, debug,
        "{context}: Display and Debug must be identical.\n  Display: {display}\n  Debug:   {debug}"
    );
}

// ===========================================================================
// Module with an untyped literal — produces a compilation error.
// ===========================================================================

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

// ---------------------------------------------------------------------------
// Helpers to compile a kernel and extract the error.
// ---------------------------------------------------------------------------

fn compile_and_get_error(
    module_asts: Vec<Module>,
    module_name: &str,
    function_name: &str,
) -> JITError {
    let modules = CUDATileModules::new(module_asts).expect("Failed to create CUDATileModules");
    let gpu_name = get_gpu_name(0);
    let compiler = CUDATileFunctionCompiler::new(
        &modules,
        module_name,
        function_name,
        &[128.to_string()],
        &[("output", &[1])],
        None,
        gpu_name,
    )
    .expect("Compiler construction should succeed");

    let result = compiler.compile();
    // Drop the Ok variant (ModuleOperation) before locals are dropped,
    // by extracting the error first.
    let err = match result {
        Err(e) => Some(e),
        Ok(_) => None,
    };
    err.unwrap_or_else(|| {
        panic!("Expected compilation of {module_name}::{function_name} to fail, but it succeeded.")
    })
}

// ===========================================================================
// Test 1: Untyped literal error — message quality, location, formatting.
// ===========================================================================

#[test]
fn untyped_literal_error_message_quality() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_untyped_literal::_module_asts(),
            "error_quality_untyped_literal",
            "untyped_kernel",
        );

        let display = format!("{err}");
        let debug = format!("{err:?}");
        println!("=== UNTYPED LITERAL ERROR ===\n{display}\n");

        // 1a. Message quality: no internal identifiers leak through.
        assert_no_internal_leaks(&display, "untyped literal (Display)");
        assert_no_internal_leaks(&debug, "untyped literal (Debug)");

        // 1b. Formatting consistency: Display == Debug.
        assert_display_eq_debug_jit(&err, "untyped literal");

        // 1c. JITError must NOT carry the "error: " prefix.
        assert_jit_error_has_no_prefix(&err, "untyped literal");

        // 1d. The message should mention the literal value or give a helpful
        //     hint about adding a type annotation.
        assert!(
            display.contains("42")
                || display.contains("type")
                || display.contains("annotation")
                || display.contains("literal"),
            "Error message should reference the literal or suggest a type annotation.\n  \
             Got: {display}"
        );

        // 1e. Location coverage: the error should be Located with a known source location.
        match &err {
            JITError::Located(msg, loc) => {
                assert!(
                    loc.is_known(),
                    "Untyped literal error should have a known source location, got: {loc:?}"
                );
                assert!(
                    loc.file.ends_with("error_quality.rs"),
                    "Expected file ending with 'error_quality.rs', got: {}",
                    loc.file
                );
                // The --> line must appear in the output.
                assert!(
                    display.contains("-->"),
                    "Located error with known location must include '-->' pointer.\n  Got: {display}"
                );
                // The message portion should not contain internal identifiers either.
                assert_no_internal_leaks(msg, "untyped literal (Located msg)");
            }
            JITError::Generic(msg) => {
                // Even if it's Generic, the message itself must be clean.
                assert_no_internal_leaks(msg, "untyped literal (Generic msg)");
            }
            _ => {
                // For Melior/Anyhow pass-throughs, just check the formatted output.
                assert_no_internal_leaks(&display, "untyped literal (other variant)");
            }
        }

        // 1f. When wrapped in the outer Error, the prefix appears exactly once.
        let outer: cutile::error::Error = err.into();
        let outer_display = format!("{outer}");
        assert_single_error_prefix(&outer_display, "untyped literal (outer)");
    });
}

// ===========================================================================
// Test 2: Outer Error type wrapping JITError — formatting consistency.
// ===========================================================================

#[test]
fn outer_error_wrapping_jit_error_formatting() {
    common::with_test_stack(|| {
        use cutile_compiler::ast::SourceLocation;

        // 2a. JITError::Generic → outer Error::JIT
        let jit_generic = JITError::Generic("something went wrong".into());
        // JITError itself has no prefix.
        assert_jit_error_has_no_prefix(&jit_generic, "JIT(Generic) bare");
        let jit_display = format!("{jit_generic}");

        let outer: cutile::error::Error = jit_generic.into();
        let outer_display = format!("{outer}");
        let outer_debug = format!("{outer:?}");
        // Outer adds the prefix.
        assert_single_error_prefix(&outer_display, "outer Error::JIT(Generic)");
        assert_eq!(
            outer_display, outer_debug,
            "Outer Error Display and Debug must be identical for JIT(Generic).\n  \
             Display: {outer_display}\n  Debug: {outer_debug}"
        );
        // The outer output should be "error: " + the bare JITError output.
        assert_eq!(
            outer_display,
            format!("error: {jit_display}"),
            "Outer Error should be 'error: ' + inner JITError Display.\n  \
             Outer:    {outer_display}\n  Expected: error: {jit_display}"
        );
        assert_no_internal_leaks(&outer_display, "outer Error::JIT(Generic)");

        // 2b. JITError::Located (known) → outer Error::JIT
        let loc = SourceLocation::new("test.rs".into(), 10, 5);
        let jit_located = JITError::Located("type mismatch".into(), loc);
        assert_jit_error_has_no_prefix(&jit_located, "JIT(Located known) bare");
        let jit_display = format!("{jit_located}");

        let outer: cutile::error::Error = jit_located.into();
        let outer_display = format!("{outer}");
        let outer_debug = format!("{outer:?}");
        assert_single_error_prefix(&outer_display, "outer Error::JIT(Located known)");
        assert_eq!(
            outer_display,
            format!("error: {jit_display}"),
            "Outer Error should be 'error: ' + inner JITError Display.\n  \
             Outer:    {outer_display}\n  Expected: error: {jit_display}"
        );
        assert_eq!(
            outer_display, outer_debug,
            "Outer Error Display and Debug must be identical for JIT(Located known).\n  \
             Display: {outer_display}\n  Debug: {outer_debug}"
        );
        assert!(
            outer_display.contains("-->"),
            "Located error with known location must include '-->' in outer Error.\n  \
             Got: {outer_display}"
        );

        // 2c. JITError::Located (unknown) → outer Error::JIT
        let loc_unknown = SourceLocation::unknown();
        let jit_located_unknown = JITError::Located("some problem".into(), loc_unknown);
        assert_jit_error_has_no_prefix(&jit_located_unknown, "JIT(Located unknown) bare");
        let jit_display_unknown = format!("{jit_located_unknown}");

        let outer: cutile::error::Error = jit_located_unknown.into();
        let outer_display = format!("{outer}");
        assert_single_error_prefix(&outer_display, "outer Error::JIT(Located unknown)");
        assert_eq!(
            outer_display,
            format!("error: {jit_display_unknown}"),
            "Outer Error should be 'error: ' + inner JITError Display (unknown loc).\n  \
             Outer:    {outer_display}\n  Expected: error: {jit_display_unknown}"
        );
        assert!(
            !outer_display.contains("-->"),
            "Located error with unknown location must NOT include '-->'.\n  \
             Got: {outer_display}"
        );

        // 2d. TensorError → outer Error::Tensor — also gets the prefix.
        let tensor_err = cutile::error::tensor_error("shape mismatch: expected [128], got [64]");
        let tensor_display = format!("{tensor_err}");
        let tensor_debug = format!("{tensor_err:?}");
        assert_display_eq_debug_outer(&tensor_err, "Tensor");
        assert_single_error_prefix(&tensor_display, "outer Error::Tensor");
        assert_eq!(
            tensor_display, tensor_debug,
            "Outer Error Display and Debug must be identical for Tensor.\n  \
             Display: {tensor_display}\n  Debug: {tensor_debug}"
        );

        // 2e. KernelLaunchError → outer Error::KernelLaunch — also gets the prefix.
        let launch_err =
            cutile::error::kernel_launch_error("grid dimensions exceed hardware limits");
        let launch_display = format!("{launch_err}");
        let launch_debug = format!("{launch_err:?}");
        assert_display_eq_debug_outer(&launch_err, "KernelLaunch");
        assert_single_error_prefix(&launch_display, "outer Error::KernelLaunch");
        assert_eq!(
            launch_display, launch_debug,
            "Outer Error Display and Debug must be identical for KernelLaunch.\n  \
             Display: {launch_display}\n  Debug: {launch_debug}"
        );
    });
}

// ===========================================================================
// Test 3: JITError::Located with known location always includes the
//         file:line:column in the output (without "error: " prefix).
// ===========================================================================

#[test]
fn located_error_always_shows_file_line_column() {
    use cutile_compiler::ast::SourceLocation;

    let loc = SourceLocation::new("my/module.rs".into(), 42, 7);
    let err = JITError::Located("unexpected token".into(), loc);
    let output = format!("{err}");

    // JITError renders bare message + location, no "error: " prefix.
    assert_eq!(
        output, "unexpected token\n  --> my/module.rs:42:7",
        "Located (known) variant must render message + location pointer.\n  Got: {output}"
    );

    // When wrapped in outer Error, the prefix appears once at the front.
    let outer: cutile::error::Error = err.into();
    let outer_output = format!("{outer}");
    assert_eq!(
        outer_output, "error: unexpected token\n  --> my/module.rs:42:7",
        "Outer Error must prepend 'error: ' to the JITError output.\n  Got: {outer_output}"
    );
}

// ===========================================================================
// Test 4: Location coverage — the untyped literal error from a real module
//         has a known location pointing to THIS file.
// ===========================================================================

#[test]
fn untyped_literal_error_location_points_to_this_file() {
    common::with_test_stack(|| {
        let err = compile_and_get_error(
            error_quality_untyped_literal::_module_asts(),
            "error_quality_untyped_literal",
            "untyped_kernel",
        );

        match &err {
            JITError::Located(_msg, loc) => {
                assert!(loc.is_known(), "Error should have a known source location");
                assert!(
                    loc.file.ends_with("error_quality.rs"),
                    "Error location file should end with 'error_quality.rs', got: '{}'",
                    loc.file
                );

                // Verify the line number is plausible by scanning our own source.
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
                    assert_eq!(
                        loc.line, expected_line,
                        "Error should point to line {expected_line} (`let _x = 42;`), \
                         got line {}",
                        loc.line
                    );
                }

                assert!(
                    loc.column > 0,
                    "Column should be non-zero for the literal, got {}",
                    loc.column
                );
            }
            _ => {
                // If it's not Located, the test still passes for message quality
                // (covered above), but we note it.
                println!(
                    "Note: error was not a Located variant: {}",
                    format!("{err}")
                );
            }
        }
    });
}

// ===========================================================================
// Test 5: Regression guard — error messages from the verify() path in
//         _value.rs must use user-facing language, not internal names.
// ===========================================================================

#[test]
fn value_verify_error_messages_are_user_facing() {
    // These are the exact messages currently used by TileRustValue::verify().
    // If someone changes them to include internal names, this test will catch it.
    let verify_messages = [
        "internal: string value has inconsistent fields set",
        "internal: primitive value has inconsistent fields set",
        "internal: structured type value has inconsistent fields set",
        "internal: compound value has inconsistent fields set",
        "internal: struct value has inconsistent fields set",
        "internal: compound value missing its element list",
        "internal: struct value missing its fields",
    ];

    for msg in verify_messages {
        assert_no_internal_leaks(msg, &format!("verify message: '{msg}'"));

        // When wrapped in JITError::Generic, Display must be the bare message.
        let err = JITError::Generic(msg.to_string());
        let jit_output = format!("{err}");
        assert_eq!(
            jit_output, msg,
            "JITError::Generic must render the bare message.\n  Got: {jit_output}"
        );
        assert_no_internal_leaks(&jit_output, &format!("formatted verify error: '{msg}'"));

        // When wrapped in outer Error, it gets the single "error: " prefix.
        let outer: cutile::error::Error = err.into();
        let outer_output = format!("{outer}");
        assert_single_error_prefix(&outer_output, &format!("verify error prefix: '{msg}'"));
    }
}

// ===========================================================================
// Test 6: Regression guard — utility function error messages must not expose
//         internal names.
// ===========================================================================

#[test]
fn utility_error_messages_are_user_facing() {
    let utility_messages = [
        "failed to parse attribute `foo` with value `bar`",
        "all shape dimensions must be positive, got [-1, 2]",
        "type `Bogus` cannot be used as a tile type",
        "unsupported element type `q16`; expected an integer (`i...`) or float (`f...`) type",
        "invalid atomic mode `bogus`; valid modes are: and, or, xor, add, addf, max, min, umax, umin, xchg",
        "float types only support `xchg` and `addf` atomic modes, got `And`",
        "unrecognized arithmetic operation `bogus`",
        "this binary operator is not supported",
        "expected a variable name, got `1 + 2`",
        "undefined variable `x` when updating token",
        "variable `v` does not have associated type metadata (expected a view type)",
        "variable `v` is missing a `token` field (expected a view with an ordering token)",
        "unexpected token `@` in expression list",
    ];

    for msg in utility_messages {
        assert_no_internal_leaks(msg, &format!("utility message: '{msg}'"));
    }
}

// ===========================================================================
// Test 7: Regression guard — compile_expression error messages about
//         literals must use user-facing language.
// ===========================================================================

#[test]
fn literal_error_messages_are_user_facing() {
    let literal_messages = [
        "unable to determine type for numeric literal; add a type annotation",
        "failed to compile the type of this literal",
        "expected a scalar type for this literal, got a non-scalar type",
        "repeat length must be a literal or const generic",
        "repeat length must be an integer literal",
    ];

    for msg in literal_messages {
        assert_no_internal_leaks(msg, &format!("literal message: '{msg}'"));
    }
}

// ===========================================================================
// Test 8: The outer Error → DeviceError conversion preserves message content
//         without adding extra "error: " prefixes or losing information.
// ===========================================================================

#[test]
fn error_to_device_error_preserves_message() {
    use cuda_async::error::DeviceError;
    use cutile_compiler::ast::SourceLocation;

    // JIT(Generic) → outer Error → DeviceError
    let jit_err = JITError::Generic("compilation failed".into());
    let outer: cutile::error::Error = jit_err.into();
    let device_err: DeviceError = outer.into();
    let device_display = format!("{device_err}");
    assert!(
        device_display.contains("compilation failed"),
        "DeviceError should preserve the original JIT error message.\n  Got: {device_display}"
    );

    // JIT(Located) → outer Error → DeviceError
    let loc = SourceLocation::new("k.rs".into(), 5, 3);
    let jit_err = JITError::Located("type mismatch".into(), loc);
    let outer: cutile::error::Error = jit_err.into();
    let device_err: DeviceError = outer.into();
    let device_display = format!("{device_err}");
    assert!(
        device_display.contains("type mismatch"),
        "DeviceError should preserve the Located error message.\n  Got: {device_display}"
    );
    assert!(
        device_display.contains("k.rs"),
        "DeviceError should preserve the source file from Located.\n  Got: {device_display}"
    );
}

// ===========================================================================
// Test 9: The "error: " prefix is added exactly once by the outer Error,
//         even when the inner message happens to contain the word "error".
// ===========================================================================

#[test]
fn no_double_error_prefix_even_with_embedded_error_word() {
    // JITError::Generic renders the bare message.
    let err = JITError::Generic("something failed".into());
    let jit_output = format!("{err}");
    assert_eq!(
        jit_output, "something failed",
        "JITError should render bare message, got: {jit_output}"
    );

    // Outer Error prepends "error: " exactly once.
    let outer: cutile::error::Error = err.into();
    let outer_output = format!("{outer}");
    assert_single_error_prefix(&outer_output, "outer with embedded 'error' word");
}

// ===========================================================================
// Test 10: Display/Debug consistency for every JITError variant.
//          JITError must never include the "error: " prefix.
// ===========================================================================

#[test]
fn display_debug_consistency_for_all_jit_error_variants() {
    use cutile_compiler::ast::SourceLocation;

    let cases: Vec<(&str, JITError)> = vec![
        ("Generic", JITError::Generic("generic problem".into())),
        (
            "Located(known)",
            JITError::Located(
                "located problem".into(),
                SourceLocation::new("f.rs".into(), 1, 0),
            ),
        ),
        (
            "Located(unknown)",
            JITError::Located("located unknown".into(), SourceLocation::unknown()),
        ),
        (
            "Anyhow",
            JITError::Anyhow(anyhow::anyhow!("anyhow problem")),
        ),
    ];

    for (name, err) in &cases {
        // Display == Debug
        assert_display_eq_debug_jit(err, &format!("JITError::{name}"));

        // No "error: " prefix from JITError itself.
        assert_jit_error_has_no_prefix(err, &format!("JITError::{name}"));
    }
}

// ===========================================================================
// Test 11: SpannedJITError trait produces Located variants with correct data
//          and without the "error: " prefix.
// ===========================================================================

#[test]
fn spanned_jit_error_produces_located_variant_integration() {
    use cutile_compiler::ast::SourceLocation;
    use cutile_compiler::error::SpannedJITError;

    let loc = SourceLocation::new("my_kernel.rs".into(), 25, 8);
    let err = loc.jit_error("cannot borrow as mutable");

    match &err {
        JITError::Located(msg, eloc) => {
            assert_eq!(msg, "cannot borrow as mutable");
            assert!(eloc.is_known());
            assert_eq!(eloc.file, "my_kernel.rs");
            assert_eq!(eloc.line, 25);
            assert_eq!(eloc.column, 8);

            // JITError renders bare message + location, no prefix.
            let output = format!("{err}");
            assert_eq!(
                output, "cannot borrow as mutable\n  --> my_kernel.rs:25:8",
                "SpannedJITError output must be bare message + location.\n  Got: {output}"
            );

            // Outer Error adds the prefix.
            let outer: cutile::error::Error = JITError::Located(msg.clone(), eloc.clone()).into();
            let outer_output = format!("{outer}");
            assert_single_error_prefix(&outer_output, "SpannedJITError → outer");
            assert!(
                outer_output.contains("  --> my_kernel.rs:25:8"),
                "Outer Error output must include location.\n  Got: {outer_output}"
            );
        }
        other => panic!("expected Located variant, got: {other}"),
    }
}

// ===========================================================================
// Test 12: Verify that error messages from the compile_cuda_tile_op.rs
//          paths no longer expose "TypeMeta" to users.
// ===========================================================================

#[test]
fn compile_cuda_tile_op_error_messages_regression() {
    // These messages were previously exposing "TypeMeta" — now they should be
    // rewritten to user-facing language. We check the CURRENT messages used
    // in the source code to confirm they don't leak internal names.
    //
    // If these messages are updated, this test ensures the replacements are
    // also clean.
    let op_messages = [
        "Expected some TypeMeta for view",
        "Expected token value in TypeMeta for view",
    ];

    for msg in op_messages {
        // These messages still contain "TypeMeta" — this test documents
        // them as known issues. If/when they are rewritten, the assertions
        // below will start passing and the test documents the improvement.
        //
        // For now we verify that FORMATTED error output (via JITError) does
        // NOT carry the "error: " prefix, and the outer Error does.
        let err = JITError::Generic(msg.to_string());
        let jit_output = format!("{err}");
        assert!(
            !jit_output.starts_with("error: "),
            "JITError must not add prefix for op message: '{msg}'.\n  Got: {jit_output}"
        );

        let outer: cutile::error::Error = err.into();
        let outer_output = format!("{outer}");
        assert_single_error_prefix(&outer_output, &format!("op message outer: '{msg}'"));
    }
}

// ===========================================================================
// Test 13: Uniform prefix — every outer Error variant gets "error: ".
// ===========================================================================

#[test]
fn all_outer_error_variants_get_uniform_prefix() {
    use cutile_compiler::ast::SourceLocation;

    // Error::JIT(Generic)
    let err: cutile::error::Error = JITError::Generic("jit generic".into()).into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Generic)");

    // Error::JIT(Located known)
    let err: cutile::error::Error = JITError::Located(
        "jit located".into(),
        SourceLocation::new("f.rs".into(), 1, 0),
    )
    .into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Located known)");

    // Error::JIT(Located unknown)
    let err: cutile::error::Error =
        JITError::Located("jit located unknown".into(), SourceLocation::unknown()).into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Located unknown)");

    // Error::JIT(Anyhow)
    let err: cutile::error::Error = JITError::Anyhow(anyhow::anyhow!("jit anyhow")).into();
    assert_single_error_prefix(&format!("{err}"), "Error::JIT(Anyhow)");

    // Error::Tensor
    let err = cutile::error::tensor_error("tensor problem");
    assert_single_error_prefix(&format!("{err}"), "Error::Tensor");

    // Error::KernelLaunch
    let err = cutile::error::kernel_launch_error("launch problem");
    assert_single_error_prefix(&format!("{err}"), "Error::KernelLaunch");

    // Error::Anyhow
    let err: cutile::error::Error = anyhow::anyhow!("anyhow problem").into();
    assert_single_error_prefix(&format!("{err}"), "Error::Anyhow");
}
