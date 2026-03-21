/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cmake::Config;
use std::{env, fs, path::PathBuf};

static MLIR_SYS_ENV_VAR: &str = "MLIR_SYS_210_PREFIX";
static SUBMODULE_PATH: &str = "cuda-tile";

static CUDA_TILE_USE_LLVM_SOURCE_DIR_VAR: &str = "CUDA_TILE_USE_LLVM_SOURCE_DIR";
static CUDA_TILE_USE_LLVM_INSTALL_DIR_VAR: &str = "CUDA_TILE_USE_LLVM_INSTALL_DIR";

static LLVM_INCLUDE_PATH_VAR: &str = "LLVM_INCLUDE_DIRS";
static LLVM_LIB_PATH_VAR: &str = "LLVM_LIBRARY_DIR";
static LLVM_BUILD_PATH_VAR: &str = "LLVM_BUILD_PATH";
static CUDA_TILE_MLIR_INCLUDE_DIR: &str = "CUDA_TILE_MLIR_INCLUDE_DIR";

#[derive(Debug)]
enum BuildMode {
    Download,
    Source(PathBuf),
    Prebuilt(PathBuf),
}

// TODO (hme): Need a way to get melior's llvm-config dependency.
fn main() {
    // Generate env var to cuda tile MLIR dialect include dir.
    let manifest_dir_str = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let manifest_dir =
        fs::canonicalize(PathBuf::from(manifest_dir_str)).expect("Failed to canonicalize path");
    eprintln!("manifest_dir={}", manifest_dir.to_str().expect(""));
    let cuda_tile_include_dir = manifest_dir.join("cuda-tile/include");
    println!(
        "cargo:rustc-env={CUDA_TILE_MLIR_INCLUDE_DIR}={}",
        cuda_tile_include_dir.to_str().expect("Failed to compile.")
    );
    println!("cargo:rerun-if-env-changed={CUDA_TILE_MLIR_INCLUDE_DIR}");

    // Ensure that the build.rs file is re-run if the configuration changes.
    println!("cargo::rerun-if-changed=build.rs");
    println!(
        "cargo::rerun-if-env-changed={}",
        CUDA_TILE_USE_LLVM_SOURCE_DIR_VAR
    );
    println!(
        "cargo::rerun-if-env-changed={}",
        CUDA_TILE_USE_LLVM_INSTALL_DIR_VAR
    );

    println!("cargo::rerun-if-env-changed={}", LLVM_INCLUDE_PATH_VAR);
    println!("cargo::rerun-if-env-changed={}", LLVM_LIB_PATH_VAR);
    println!("cargo::rerun-if-env-changed={}", LLVM_BUILD_PATH_VAR);

    let cuda_tile_source_dir =
        std::env::var("CUDA_TILE_SOURCE_DIR").unwrap_or_else(|_| SUBMODULE_PATH.to_string());

    println!("cuda_tile_source_dir: {}", cuda_tile_source_dir);

    let build_mode = match std::env::var(CUDA_TILE_USE_LLVM_INSTALL_DIR_VAR) {
        Ok(prebuilt_llvm_path) => BuildMode::Prebuilt(PathBuf::from(prebuilt_llvm_path)),
        Err(_) => match std::env::var(CUDA_TILE_USE_LLVM_SOURCE_DIR_VAR) {
            Ok(llvm_syspath) => BuildMode::Source(PathBuf::from(llvm_syspath)),
            Err(_) => BuildMode::Download,
        },
    };

    println!("build_mode: {:?}", build_mode);

    println!("cuda_tile_source_dir: {}", cuda_tile_source_dir);

    let mut config = Config::new(&cuda_tile_source_dir);

    config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LLVM_ENABLE_ASSERTIONS", "OFF")
        .define("CUDA_TILE_ENABLE_TESTING", "OFF")
        .cxxflag("-Wno-deprecated-declarations")
        .cxxflag("-Wno-unused-result");

    let llvm_path;
    match build_mode {
        BuildMode::Prebuilt(prebuilt_llvm_path) => {
            llvm_path = prebuilt_llvm_path;
            println!("prebuilt_llvm_path is {}", llvm_path.display());
            println!(
                "cargo:rustc-env={}={}",
                MLIR_SYS_ENV_VAR,
                llvm_path.display()
            );
            config.define(
                CUDA_TILE_USE_LLVM_INSTALL_DIR_VAR,
                llvm_path.display().to_string(),
            );
        }
        BuildMode::Source(_llvm_syspath) => {
            unimplemented!("BuildMode::Source not implemented.");
            // TODO (hme): How do we support this mode?
            // llvm_path = llvm_syspath;
            // println!("llvm_syspath is {}", llvm_path.display());
            // config.define(
            //     CUDA_TILE_USE_LLVM_SOURCE_DIR_VAR,
            //     llvm_path.display().to_string(),
            // );
        }
        BuildMode::Download => {
            llvm_path = PathBuf::from(std::env::var("OUT_DIR").unwrap())
                .join("build")
                .join("llvm-project-subbuild");
            println!("llvm_path: {}", llvm_path.display());
            println!(
                "cargo:rustc-env={}={}",
                LLVM_BUILD_PATH_VAR,
                llvm_path.display()
            );
            println!(
                "cargo:rustc-env={}={}",
                MLIR_SYS_ENV_VAR,
                llvm_path.display()
            );
            println!(
                "cargo:rustc-env=TABLEGEN_210_PREFIX={}",
                llvm_path.display()
            );

            let llvm_include_path = llvm_path.join("include");
            println!(
                "cargo:rustc-env={}={}",
                LLVM_INCLUDE_PATH_VAR,
                llvm_include_path.display()
            );

            let llvm_lib_path = llvm_path.join("lib");
            println!(
                "cargo:rustc-env={}={}",
                LLVM_LIB_PATH_VAR,
                llvm_lib_path.display()
            );
            println!("cargo:warning=Defaultling to download mode.");
        }
    }

    let cuda_tile_path = config.build();

    println!("output: {}", cuda_tile_path.display());

    // TODO(@jroesch): we need to make this work in all modes, probably want to support dynamic + static linking.
    //
    // Set up LLVM link paths.
    println!("cargo:rustc-link-search={}", llvm_path.to_str().unwrap());
    println!("cargo:rustc-link-lib=LLVM-21");
    println!("cargo:rustc-link-lib=MLIR");
    println!("cargo:rustc-link-lib=LLVMSupport");

    // Now we need to bindgen the generated libraries.
    let lib_path = cuda_tile_path.join("lib");
    let include_path = cuda_tile_path.join("include").join("include");

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", lib_path.display());

    // Tell cargo to tell rustc to link the static libraries built by the CMake build.
    println!("cargo:rustc-link-lib=static=CudaTileBytecodeCommon");
    println!("cargo:rustc-link-lib=static=CudaTileBytecodeWriter");
    println!("cargo:rustc-link-lib=static=CudaTileDialect");
    println!("cargo:rustc-link-lib=static=CudaTileBytecodeReader");
    println!("cargo:rustc-link-lib=static=CudaTileCAPIDialects");
    println!("cargo:rustc-link-lib=static=CudaTileTransforms");
    println!("cargo:rustc-link-lib=static=CudaTileBytecodeTranslation");
    println!("cargo:rustc-link-lib=static=CudaTileCAPIRegistration");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(
            include_path
                .join("cuda_tile-c")
                .join("Registration.h")
                .to_str()
                .unwrap(),
        )
        .header(
            include_path
                .join("cuda_tile-c")
                .join("Dialect")
                .join("CudaTileDialect.h")
                .to_str()
                .unwrap(),
        )
        .clang_arg(format!("-I{}", llvm_path.join("include").to_str().unwrap()))
        .allowlist_file(".*/cuda_tile-c/.*")
        .blocklist_item("MlirDialectRegistry")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let _ = bindings.write_to_file(out_path.join("bindings.rs"));
}
