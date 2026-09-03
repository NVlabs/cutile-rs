/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Advisory Tile-floor warning at build time.
//!
//! The authoritative check runs at tool discovery in
//! `cuda_tile_runtime_utils`, on the machine that executes the JIT. This
//! warning covers the common case where the build box and the run box are
//! the same, so a too-old toolkit is reported at compile time instead of
//! first launch. It never fails the build: emitting bytecode and
//! cross-building are legitimate on machines without a 13.2+ toolkit.

use std::env;
use std::fs;
use std::path::Path;

const TOOLKIT_ENV_VARS: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME"];
const DEFAULT_TOOLKIT_DIR: &str = "/usr/local/cuda";
const MIN_TILE_CUDA_VERSION: u32 = 13020;

fn main() {
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let toolkit = TOOLKIT_ENV_VARS
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| DEFAULT_TOOLKIT_DIR.to_string());
    let cuda_h = Path::new(&toolkit).join("include").join("cuda.h");
    let Ok(header) = fs::read_to_string(&cuda_h) else {
        return;
    };
    let version = header.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        match (parts.next(), parts.next(), parts.next()) {
            (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse::<u32>().ok(),
            _ => None,
        }
    });
    if let Some(version) = version {
        if version < MIN_TILE_CUDA_VERSION {
            println!(
                "cargo:warning=cutile-compiler: the toolkit at {} is CUDA {}.{}; \
                 cuTile requires CUDA 13.2+ at run time (tileiras ships with it). \
                 The shared CUDA host-side crates support 13.0+.",
                toolkit,
                version / 1000,
                (version % 1000) / 10
            );
        }
    }
}
