/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling Tile IR modules to GPU cubins.
//! Provides GPU detection and bytecode compilation helpers.

use cuda_core::{get_device_sm_name, Device};
use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;
use uuid::Uuid;

/// Environment variable used to override the `tileiras` executable.
///
/// Set this to an absolute path such as `/opt/cuda-tile/bin/tileiras` to use
/// that binary instead of the `tileiras` found on `PATH`.
pub const TILEIRAS_PATH_ENV: &str = "CUTILE_TILEIRAS_PATH";
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

/// Queries the CUDA driver to determine the SM architecture name (e.g. `"sm_90"`) for a device.
pub fn get_gpu_name(device_id: usize) -> String {
    let dev = Device::raw_device(device_id).expect("failed to get CUDA device");
    unsafe { get_device_sm_name(dev) }.expect("failed to get SM name")
}

fn resolve_tileiras_binary(value: Option<OsString>) -> PathBuf {
    value
        .filter(|value| !value.as_os_str().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tileiras"))
}

/// Returns the `tileiras` executable path used by the JIT.
///
/// Defaults to `tileiras`, which uses normal `PATH` lookup. Set
/// [`TILEIRAS_PATH_ENV`] to override this with a specific binary.
pub fn tileiras_binary() -> PathBuf {
    resolve_tileiras_binary(env::var_os(TILEIRAS_PATH_ENV))
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
    let tileiras = tileiras_binary();
    let output = Command::new(&tileiras)
        .arg("--gpu-name")
        .arg(gpu_name)
        .arg("--opt-level")
        .arg("3")
        .arg("-o")
        .arg(&cubin_filename)
        .arg(&bc_filename)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to launch {} for {bc_filename}: {e}",
                tileiras.display()
            )
        });
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "{} failed (exit {}) for gpu {gpu_name}:\nstderr: {stderr}\nstdout: {stdout}",
            tileiras.display(),
            output.status
        );
    }
    cubin_filename
}

#[cfg(test)]
mod tests {
    use super::*;
    use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
    use cutile_ir::bytecode::Opcode;
    use cutile_ir::ir::{Attribute, FuncType, Location, Module, Type};
    use std::fs;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn tileiras_binary_defaults_to_path_lookup() {
        assert_eq!(resolve_tileiras_binary(None), PathBuf::from("tileiras"));
    }

    #[test]
    fn tileiras_binary_uses_override_path() {
        assert_eq!(
            resolve_tileiras_binary(Some(OsString::from("/opt/cuda/bin/tileiras"))),
            PathBuf::from("/opt/cuda/bin/tileiras")
        );
    }

    #[test]
    fn tileiras_binary_treats_empty_override_as_default() {
        assert_eq!(
            resolve_tileiras_binary(Some(OsString::new())),
            PathBuf::from("tileiras")
        );
    }

    #[test]
    #[cfg(unix)]
    fn compile_tile_ir_module_uses_tileiras_path_override() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join(format!("cutile_tileiras_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let fake_tileiras = temp_dir.join("tileiras");
        write_fake_tileiras(&fake_tileiras);

        let _tileiras_env = EnvVarGuard::set(TILEIRAS_PATH_ENV, &fake_tileiras);

        let module = empty_kernel_module();
        let cubin_path = compile_tile_ir_module(&module, "sm_120");

        let args_path = fake_tileiras.with_extension("args");
        let args = fs::read_to_string(&args_path).unwrap();
        assert!(
            args.lines()
                .next()
                .is_some_and(|line| line == fake_tileiras.to_string_lossy()),
            "expected fake tileiras to record its own path, got:\n{args}"
        );
        assert!(args.contains("--gpu-name\nsm_120"), "args:\n{args}");
        assert!(args.contains("-o\n"), "args:\n{args}");
        assert!(PathBuf::from(&cubin_path).exists());

        let bc_path = args.lines().last().unwrap_or_default();
        let _ = fs::remove_file(bc_path);
        let _ = fs::remove_file(&cubin_path);
        let _ = fs::remove_file(args_path);
        let _ = fs::remove_file(fake_tileiras);
        let _ = fs::remove_dir(temp_dir);
    }

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let previous = env::var_os(key);
            env::set_var(key, value);
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => env::set_var(self.key, previous),
                None => env::remove_var(self.key),
            }
        }
    }

    fn empty_kernel_module() -> Module {
        let mut module = Module::new("tileiras_override_test");
        let func_type = Type::Func(FuncType {
            inputs: vec![],
            results: vec![],
        });

        let (region_id, block_id, _) = build_single_block_region(&mut module, &[]);
        let (ret_id, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
        append_op(&mut module, block_id, ret_id);

        let (entry_id, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
            .attr("sym_name", Attribute::String("empty_kernel".into()))
            .attr("function_type", Attribute::Type(func_type))
            .region(region_id)
            .build(&mut module);
        module.functions.push(entry_id);
        module
    }

    #[cfg(unix)]
    fn write_fake_tileiras(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;

        fs::write(
            path,
            r#"#!/bin/sh
set -eu
args_file="$0.args"
printf '%s\n' "$0" "$@" > "$args_file"
out=""
while [ "$#" -gt 0 ]; do
    if [ "$1" = "-o" ]; then
        shift
        out="$1"
    fi
    shift || break
done
if [ -z "$out" ]; then
    echo "missing -o output" >&2
    exit 2
fi
printf 'fake cubin\n' > "$out"
"#,
        )
        .unwrap();

        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions).unwrap();
    }
}
