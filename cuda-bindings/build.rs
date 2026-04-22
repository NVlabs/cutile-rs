// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

use std::{env, error::Error, path::Path, process::exit, str};

use std::path::PathBuf;

/// Returns the CUDA toolkit directory from the `CUDA_TOOLKIT_PATH` environment variable.
pub fn cuda_toolkit_dir() -> String {
    env::var("CUDA_TOOLKIT_PATH").expect("CUDA_TOOLKIT_PATH is required but not set")
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

/// Generates CUDA bindings via bindgen and configures native library link paths.
fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");

    // CUDA Toolkit
    let toolkit_paths = collect_paths(&cuda_toolkit_dir());
    for path in toolkit_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=curand");

    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", cuda_toolkit_dir()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .unwrap()
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

/// Collects valid library search paths from the given CUDA toolkit root directory.
pub fn collect_paths(cuda_toolkit: &str) -> Vec<PathBuf> {
    let candidates = vec![PathBuf::from(cuda_toolkit)];
    let mut valid_paths = vec![];
    for base in &candidates {
        let lib = PathBuf::from(base).join("lib64");
        if lib.is_dir() {
            valid_paths.push(lib.clone());
            valid_paths.push(lib.join("stubs"));
        }
        let base = base.join("targets/x86_64-linux");
        let header = base.join("include/cuda.h");
        if header.is_file() {
            valid_paths.push(base.join("lib"));
            valid_paths.push(base.join("lib/stubs"));
            continue;
        }
    }
    valid_paths
}
