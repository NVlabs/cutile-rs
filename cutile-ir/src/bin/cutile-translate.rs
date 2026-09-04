/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Decode-only translation CLI mirroring `cuda-tile-translate`, so the upstream
//! lit tests can run against our decoder:
//!
//! ```text
//! cutile-translate --bc-to-mlir <input.tileirbc>
//! ```
//!
//! Prints textual IR to stdout, or a stderr diagnostic and non-zero exit on
//! failure (so negative tests work too).

use std::process::ExitCode;

fn usage() -> String {
    "usage: cutile-translate --bc-to-mlir <input.tileirbc>".to_string()
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mode = match args.next() {
        Some(m) => m,
        None => {
            eprintln!("{}", usage());
            return ExitCode::FAILURE;
        }
    };

    match mode.as_str() {
        "--bc-to-mlir" | "-cudatilebc-to-mlir" => {}
        other => {
            eprintln!("error: unknown mode {other:?}\n{}", usage());
            return ExitCode::FAILURE;
        }
    }

    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("error: missing input path\n{}", usage());
            return ExitCode::FAILURE;
        }
    };

    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    match cutile_ir::decode_module(&data) {
        Ok(module) => {
            print!("{module}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: failed to decode {path}: {e}");
            ExitCode::FAILURE
        }
    }
}
