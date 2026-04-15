/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling Tile IR modules to GPU cubins.
//! Provides GPU detection and bytecode compilation helpers.

use cuda_core::{device, get_device_sm_name, init};
use std::env;
use std::process::Command;
use uuid::Uuid;

/// Queries the CUDA driver to determine the SM architecture name (e.g. `"sm_90"`) for a device.
pub fn get_gpu_name(device_id: usize) -> String {
    unsafe { init(0) }.expect("failed to initialize CUDA driver");
    let dev = device::get(device_id as i32).expect("failed to get CUDA device");
    unsafe { get_device_sm_name(dev) }.expect("failed to get SM name")
}

/// Compiles a `cutile_ir::Module` to a `.cubin` file via bytecode serialization and `tileiras`.
pub fn compile_tile_ir_module(module: &cutile_ir::Module, gpu_name: &str) -> String {
    let tmp_dir = env::temp_dir();
    let base_filename = tmp_dir.join(Uuid::new_v4().to_string());
    let bc_filename = format!("{}.bc", base_filename.to_str().unwrap());
    let cubin_filename = format!("{}.cubin", base_filename.to_str().unwrap());

    module
        .verify_dominance()
        .expect("tile-ir dominance verification failed");

    module
        .verify_bytecode_indices()
        .expect("tile-ir bytecode value-index verification failed");

    // Dump IR via unified CUTILE_DUMP mechanism (also honors legacy TILE_IR_DUMP).
    crate::dump::dump_module(
        crate::dump::DumpStage::Ir,
        &module.name,
        &module.to_mlir_text(),
    );

    cutile_ir::write_bytecode_to_file(module, bc_filename.as_str())
        .unwrap_or_else(|_| panic!("Failed to write bytecode for {bc_filename}"));
    let output = Command::new("tileiras")
        .arg("--gpu-name")
        .arg(gpu_name)
        .arg("--opt-level")
        .arg("3")
        .arg("-o")
        .arg(&cubin_filename)
        .arg(&bc_filename)
        .output()
        .unwrap_or_else(|_| panic!("Failed to launch tileiras for {bc_filename}"));
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "tileiras failed (exit {}) for gpu {gpu_name}:\nstderr: {stderr}\nstdout: {stdout}",
            output.status
        );
    }
    cubin_filename
}
