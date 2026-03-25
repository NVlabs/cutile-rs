/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling CUDA Tile MLIR modules to GPU cubins.
//! Provides GPU detection, MLIR parsing, and bytecode compilation helpers.

use cuda_tile_rs::cuda_tile::ModuleOperation;
use cuda_tile_rs::{cuda_tile, cuda_tile_write_bytecode, operation_parse};
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{Block, BlockLike, Location, Region, RegionLike};
use melior::Context;
use regex::Regex;
use std::env;
use std::process::Command;
use uuid::Uuid;

/// Queries `nvidia-smi` to determine the SM architecture name (e.g. `"sm_90"`) for a device.
pub fn get_gpu_name(device_id: usize) -> String {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .arg(format!("--id={device_id}"))
        .output()
        .expect(format!("Failed to determine compute capability for device {device_id}").as_str());
    if !output.status.success() {
        let error_output = String::from_utf8_lossy(&output.stderr).to_string();
        panic!("{}", error_output)
    }
    // This has decimals.
    let compute_cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let re_ver = Regex::new(r"\.").unwrap();
    // TODO (hme): Confirm this solution cannot fail.
    format!("sm_{}", re_ver.replace(&compute_cap, "").to_string())
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
    let _ = Command::new("tileiras")
        .arg("--gpu-name")
        .arg(gpu_name)
        .arg("--opt-level")
        .arg("3")
        .arg("-o")
        .arg(&cubin_filename)
        .arg(&bc_filename)
        .output()
        .expect(format!("Failed to compile bytecode {bc_filename}").as_str());
    cubin_filename
}
