// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

use std::{env, error::Error, path::Path, process::exit};

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

/// Generates CUDA type bindings via bindgen.
///
/// Only CUDA headers are required at build time (for type definitions).
/// The CUDA driver and cuRAND libraries are loaded dynamically at runtime
/// via `libloading`, so they do **not** need to be present at build time.
fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");

    let cuda_toolkit =
        env::var("CUDA_TOOLKIT_PATH").expect("CUDA_TOOLKIT_PATH is required but not set");

    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{cuda_toolkit}/include"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Skip function declarations; they are provided by the dynamic loading
        // layer in src/dyn_load.rs. Types, enums, and constants are still generated.
        .blocklist_function(".*")
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}
