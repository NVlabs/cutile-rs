/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conformance against real reference bytecode, read directly from the
//! `cuda-tile` submodule (no copies, no C++ build). The fixtures are NVIDIA's
//! version-pinned inputs whose op bodies predate the decoder's v13.3 target.
//! We check idempotency rather than byte-identity (the writer re-emits at the
//! current version, as the reference's own round-trip does). Tests skip if the
//! submodule isn't checked out.

use cutile_ir::{decode_module, write_bytecode};

/// Path to a fixture under `cuda-tile/test/Bytecode/versioning/Inputs/`.
fn fixture_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../cuda-tile-rs/cuda-tile/test/Bytecode/versioning/Inputs")
        .join(rel)
}

/// Decode → re-encode → decode → re-encode reaches a stable fixed point.
/// Skips if the submodule fixture is absent.
fn assert_idempotent(rel: &str) {
    let path = fixture_path(rel);
    let Ok(data) = std::fs::read(&path) else {
        eprintln!(
            "skipping {rel}: submodule fixture not found at {}",
            path.display()
        );
        return;
    };
    let m1 = decode_module(&data).unwrap_or_else(|e| panic!("{rel}: decode failed: {e}"));
    let b1 = write_bytecode(&m1).expect("re-encode");
    let m2 = decode_module(&b1).unwrap_or_else(|e| panic!("{rel}: re-decode failed: {e}"));
    let b2 = write_bytecode(&m2).expect("re-encode 2");
    assert_eq!(b1, b2, "{rel}: decode/re-encode is not idempotent");
}

#[test]
fn decode_negi_13_1() {
    // NegI's overflow attr was added v13.2; v13.1 omits it (used to fail with
    // "operand value index out of range").
    assert_idempotent("13.1/negi-op-13.1.tileirbc");
}

#[test]
fn decode_print_13_1() {
    // Print's flags field was added in v13.2; a v13.1 body omits it.
    assert_idempotent("13.1/print-op-13.1.tileirbc");
}
