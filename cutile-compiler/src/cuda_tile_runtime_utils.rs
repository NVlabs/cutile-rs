/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling CUDA Tile MLIR modules to GPU cubins.
//! Provides GPU detection, MLIR parsing, and bytecode compilation helpers.

use cuda_core::{device, get_device_sm_name, init};
use cuda_tile_rs::cuda_tile::ModuleOperation;
use cuda_tile_rs::{cuda_tile, cuda_tile_write_bytecode, operation_parse};
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{Block, BlockLike, Location, Region, RegionLike};
use melior::Context;
use std::env;
use std::process::Command;
use uuid::Uuid;

/// Returns the cutile compiler version (from the workspace Cargo.toml).
pub fn get_compiler_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Returns the CUDA toolkit version by parsing `nvcc --version` output.
///
/// Falls back to `"unknown"` if `nvcc` is not available.
pub fn get_cuda_toolkit_version() -> String {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if !output.status.success() {
                return None;
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse lines like "Cuda compilation tools, release 12.4, V12.4.131"
            for line in stdout.lines() {
                if let Some(pos) = line.find("release ") {
                    let rest = &line[pos + "release ".len()..];
                    if let Some(comma) = rest.find(',') {
                        return Some(rest[..comma].to_string());
                    }
                    return Some(rest.trim().to_string());
                }
            }
            None
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Queries `nvidia-smi` to determine the SM architecture name (e.g. `"sm_90"`) for a device.
pub fn get_gpu_name(device_id: usize) -> String {
    unsafe { init(0) }.expect("failed to initialize CUDA driver");
    let dev = device::get(device_id as i32).expect("failed to get CUDA device");
    unsafe { get_device_sm_name(dev) }.expect("failed to get SM name")
}

/// Parses a CUDA Tile MLIR entry string into a verified module operation.
pub fn parse_tile_entry<'c>(
    context: &'c Context,
    module_name: &str,
    entry: &str,
) -> ModuleOperation<'c> {
    let location = Location::unknown(&context);
    let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location)
        .body({
            let entry_op = operation_parse(&context, entry, None).unwrap();
            let module_block = Block::new(&[]);
            module_block.append_operation(entry_op);

            let region = Region::new();
            region.append_block(module_block);
            region
        })
        .sym_name(StringAttribute::new(&context, module_name))
        .build();
    assert!(module_op.as_operation().verify());
    return module_op;
}

/// Compiles a CUDA Tile module operation to a `.cubin` file via `tileiras`, returning the path.
pub fn compile_module(module_op: &ModuleOperation, gpu_name: &str) -> String {
    let tmp_dir = env::temp_dir();
    let base_filename = tmp_dir.join(Uuid::new_v4().to_string());
    let bc_filename = format!("{}.bc", base_filename.to_str().unwrap());
    let cubin_filename = format!("{}.cubin", base_filename.to_str().unwrap());

    let res = cuda_tile_write_bytecode(&module_op, bc_filename.as_str());
    assert!(res.is_ok());
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
