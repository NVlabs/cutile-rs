/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling Tile IR modules to GPU cubins.
//! Provides GPU detection and bytecode compilation helpers.

use cuda_core::{get_device_sm_name, Device};
use std::env;
use std::process::Command;
use uuid::Uuid;

/// Queries the CUDA driver to determine the SM architecture name (e.g. `"sm_90"`) for a device.
pub fn get_gpu_name(device_id: usize) -> String {
    let dev = Device::raw_device(device_id).expect("failed to get CUDA device");
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

    let bytes = cutile_ir::write_bytecode(module)
        .unwrap_or_else(|e| panic!("Failed to serialize bytecode for {bc_filename}: {e}"));

    if crate::dump::should_dump(crate::dump::DumpStage::Bytecode) {
        let decoded = cutile_ir::decode_bytecode(&bytes)
            .unwrap_or_else(|e| format!("<bytecode decode failed: {e}>"));
        crate::dump::dump_module(crate::dump::DumpStage::Bytecode, &module.name, &decoded);
    }

    std::fs::write(&bc_filename, &bytes)
        .unwrap_or_else(|e| panic!("Failed to write bytecode for {bc_filename}: {e}"));
    let output = Command::new("tileiras")
        .arg("--gpu-name")
        .arg(gpu_name)
        .arg("--opt-level")
        .arg("3")
        .arg("-o")
        .arg(&cubin_filename)
        .arg(&bc_filename)
        .output()
        .expect(format!("Failed to launch tileiras for {bc_filename}").as_str());
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
