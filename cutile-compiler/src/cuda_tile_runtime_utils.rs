/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling Tile IR modules to GPU cubins.
//! Provides GPU detection and bytecode compilation helpers.

use cuda_core::{get_device_sm_name, Device};
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use uuid::Uuid;

/// Environment variable used to override the `tileiras` executable.
///
/// Set this to an absolute path such as `/opt/cuda-tile/bin/tileiras` to use
/// that binary instead of the `tileiras` found on `PATH`.
pub const TILEIRAS_PATH_ENV: &str = "CUTILE_TILEIRAS_PATH";
pub const SETUP_DIAGNOSTICS_ENV: &str = "CUTILE_SETUP_DIAGNOSTICS";

const CUDA_TOOLKIT_PATH_ENV: &str = "CUDA_TOOLKIT_PATH";
const MIN_CUDA_VERSION: u32 = 13020;

/// Queries the CUDA driver to determine the SM architecture name (e.g. `"sm_90"`) for a device.
pub fn get_gpu_name(device_id: usize) -> String {
    let dev = Device::raw_device(device_id).unwrap_or_else(|e| {
        panic!(
            "failed to get CUDA device {device_id}: {e}\n\
             Ensure an NVIDIA GPU is visible to the process and the CUDA driver is installed."
        )
    });
    unsafe { get_device_sm_name(dev) }.unwrap_or_else(|e| {
        panic!(
            "failed to query CUDA SM name for device {device_id}: {e}\n\
             Ensure the installed CUDA driver supports this GPU."
        )
    })
}

fn tileiras_executable_name() -> &'static str {
    if cfg!(windows) {
        "tileiras.exe"
    } else {
        "tileiras"
    }
}

fn cuda_toolkit_tileiras(cuda_toolkit_path: Option<OsString>) -> Option<PathBuf> {
    let tileiras = cuda_toolkit_path
        .filter(|value| !value.as_os_str().is_empty())
        .map(PathBuf::from)
        .map(|path| path.join("bin").join(tileiras_executable_name()));
    match tileiras {
        Some(path) if path.is_file() => {
            emit_setup_diagnostic(format_args!(
                "using {CUDA_TOOLKIT_PATH_ENV} tileiras at {}",
                path.display()
            ));
            Some(path)
        }
        Some(path) => {
            emit_setup_diagnostic(format_args!(
                "{CUDA_TOOLKIT_PATH_ENV} did not contain tileiras at {}",
                path.display()
            ));
            None
        }
        None => None,
    }
}

fn resolve_tileiras_binary(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<OsString>,
) -> PathBuf {
    resolve_tileiras_binary_with_candidates(
        tileiras_override,
        cuda_toolkit_path,
        default_cuda_toolkit_candidates(),
    )
}

fn resolve_tileiras_binary_with_candidates(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<OsString>,
    default_cuda_toolkit_candidates: &[PathBuf],
) -> PathBuf {
    if let Some(path) = tileiras_override.filter(|value| !value.as_os_str().is_empty()) {
        let path = PathBuf::from(path);
        emit_setup_diagnostic(format_args!("using {TILEIRAS_PATH_ENV}={}", path.display()));
        return path;
    }

    if let Some(path) = cuda_toolkit_tileiras(cuda_toolkit_path) {
        if path.is_file() {
            return path;
        }
    }

    if let Some(path) = default_cuda_toolkit_tileiras(default_cuda_toolkit_candidates) {
        return path;
    }

    emit_setup_diagnostic(format_args!(
        "falling back to {} through PATH lookup",
        tileiras_executable_name()
    ));
    PathBuf::from(tileiras_executable_name())
}

/// Returns the `tileiras` executable path used by the JIT.
///
/// Resolution order:
///
/// 1. [`TILEIRAS_PATH_ENV`] when set.
/// 2. `$CUDA_TOOLKIT_PATH/bin/tileiras` when `CUDA_TOOLKIT_PATH` is set and
///    the binary exists there.
/// 3. `$CUDA_TOOLKIT_PATH`-style default CUDA installs with CUDA 13.2+ and
///    `bin/tileiras`.
/// 4. `tileiras` through normal `PATH` lookup.
pub fn tileiras_binary() -> PathBuf {
    resolve_tileiras_binary(
        env::var_os(TILEIRAS_PATH_ENV),
        env::var_os(CUDA_TOOLKIT_PATH_ENV),
    )
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
    let args = [
        "--gpu-name",
        gpu_name,
        "--opt-level",
        "3",
        "-o",
        &cubin_filename,
        &bc_filename,
    ];
    let output = Command::new(&tileiras)
        .args(args)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "{}",
                tileiras_launch_error(&tileiras, &args, &bc_filename, e)
            )
        });
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "{} failed while compiling Tile IR bytecode.\n\
             status: {}\n\
             command: {}\n\
             target gpu: {gpu_name}\n\
             bytecode: {bc_filename}\n\
             output cubin: {cubin_filename}\n\
             stdout:\n{stdout}\n\
             stderr:\n{stderr}\n\
             hint: run with CUTILE_DUMP=ir,bytecode to include the generated Tile IR and decoded bytecode in stderr.",
            tileiras.display(),
            output.status,
            display_command(&tileiras, &args),
        );
    }
    cubin_filename
}

fn tileiras_launch_error(
    tileiras: &Path,
    args: &[&str],
    bc_filename: &str,
    error: std::io::Error,
) -> String {
    let mut message = format!(
        "failed to launch tileiras.\n\
         error: {error}\n\
         command: {}\n\
         bytecode: {bc_filename}\n\
         CUTILE_TILEIRAS_PATH: {}\n\
         CUDA_TOOLKIT_PATH: {}\n",
        display_command(tileiras, args),
        env::var(TILEIRAS_PATH_ENV).unwrap_or_else(|_| "<unset>".to_string()),
        env::var(CUDA_TOOLKIT_PATH_ENV).unwrap_or_else(|_| "<unset>".to_string()),
    );

    if env::var_os(TILEIRAS_PATH_ENV).is_none() {
        message.push_str(
            "hint: install CUDA 13.2+ with tileiras, set CUDA_TOOLKIT_PATH to that toolkit, \
             set CUTILE_TILEIRAS_PATH to the absolute tileiras path, or rerun with \
             CUTILE_SETUP_DIAGNOSTICS=1 to trace toolkit discovery.",
        );
    } else {
        message
            .push_str("hint: verify CUTILE_TILEIRAS_PATH points to an executable tileiras binary.");
    }

    message
}

fn default_cuda_toolkit_candidates() -> &'static [PathBuf] {
    static CANDIDATES: std::sync::OnceLock<Vec<PathBuf>> = std::sync::OnceLock::new();
    CANDIDATES.get_or_init(|| {
        #[cfg(windows)]
        let candidates = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        ];
        #[cfg(not(windows))]
        let candidates = [
            "/usr/local/cuda-13.3",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13",
            "/usr/local/cuda",
        ];

        candidates.into_iter().map(PathBuf::from).collect()
    })
}

fn default_cuda_toolkit_tileiras(candidates: &[PathBuf]) -> Option<PathBuf> {
    for candidate in candidates {
        match supported_cuda_toolkit_tileiras(candidate) {
            Ok(tileiras) => {
                emit_setup_diagnostic(format_args!(
                    "{CUDA_TOOLKIT_PATH_ENV} is unset; using discovered tileiras at {}",
                    tileiras.display()
                ));
                return Some(tileiras);
            }
            Err(error) => {
                emit_setup_diagnostic(format_args!(
                    "{CUDA_TOOLKIT_PATH_ENV} is unset; skipping {}: {error}",
                    candidate.display()
                ));
            }
        }
    }

    None
}

fn supported_cuda_toolkit_tileiras(cuda_toolkit: &Path) -> Result<PathBuf, String> {
    if !cuda_toolkit.is_dir() {
        return Err("not a directory".to_string());
    }

    let cuda_h = cuda_toolkit.join("include").join("cuda.h");
    let version = cuda_version_from_header(&cuda_h)?;
    if version < MIN_CUDA_VERSION {
        return Err(format!(
            "CUDA toolkit {} is too old",
            format_cuda_version(version)
        ));
    }

    let tileiras = cuda_toolkit.join("bin").join(tileiras_executable_name());
    if !tileiras.is_file() {
        return Err(format!("missing {}", tileiras.display()));
    }

    Ok(tileiras)
}

fn cuda_version_from_header(cuda_h: &Path) -> Result<u32, String> {
    let source = std::fs::read_to_string(cuda_h)
        .map_err(|error| format!("could not read {}: {error}", cuda_h.display()))?;
    source
        .lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next(), parts.next()) {
                (Some("#define"), Some("CUDA_VERSION"), Some(version)) => version.parse().ok(),
                _ => None,
            }
        })
        .ok_or_else(|| format!("could not find CUDA_VERSION in {}", cuda_h.display()))
}

fn format_cuda_version(version: u32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

fn setup_diagnostics_enabled() -> bool {
    env::var(SETUP_DIAGNOSTICS_ENV)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn emit_setup_diagnostic(args: std::fmt::Arguments<'_>) {
    if setup_diagnostics_enabled() {
        eprintln!("cutile setup: {args}");
    }
}

fn display_command(program: &Path, args: &[&str]) -> String {
    std::iter::once(shell_display(program.as_os_str()))
        .chain(args.iter().map(|arg| shell_display(arg.as_ref())))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_display(value: &std::ffi::OsStr) -> String {
    let value = value.to_string_lossy();
    if value.is_empty() {
        "''".to_string()
    } else if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '='))
    {
        value.into_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
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
        assert_eq!(
            resolve_tileiras_binary_with_candidates(None, None, &[]),
            PathBuf::from("tileiras")
        );
    }

    #[test]
    fn tileiras_binary_uses_override_path() {
        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                Some(OsString::from("/opt/cuda/bin/tileiras")),
                None,
                &[]
            ),
            PathBuf::from("/opt/cuda/bin/tileiras")
        );
    }

    #[test]
    fn tileiras_binary_treats_empty_override_as_default() {
        assert_eq!(
            resolve_tileiras_binary_with_candidates(Some(OsString::new()), None, &[]),
            PathBuf::from("tileiras")
        );
    }

    #[test]
    #[cfg(unix)]
    fn tileiras_binary_uses_cuda_toolkit_path_when_present() {
        let temp_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let bin_dir = temp_dir.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let tileiras = bin_dir.join(tileiras_executable_name());
        fs::write(&tileiras, "").unwrap();

        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                None,
                Some(temp_dir.clone().into_os_string()),
                &[]
            ),
            tileiras
        );

        let _ = fs::remove_file(bin_dir.join(tileiras_executable_name()));
        let _ = fs::remove_dir(bin_dir);
        let _ = fs::remove_dir(temp_dir);
    }

    #[test]
    fn tileiras_binary_ignores_cuda_toolkit_path_without_tileiras() {
        let temp_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        assert_eq!(
            resolve_tileiras_binary_with_candidates(None, Some(temp_dir.into_os_string()), &[]),
            PathBuf::from(tileiras_executable_name())
        );
    }

    #[test]
    fn tileiras_binary_uses_default_cuda_toolkit_when_supported() {
        let temp_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let tileiras = create_fake_cuda_toolkit(&temp_dir, 13020, true);

        assert_eq!(
            resolve_tileiras_binary_with_candidates(None, None, &[temp_dir.clone()]),
            tileiras
        );

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn tileiras_binary_skips_old_default_cuda_toolkit() {
        let old_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let new_dir = env::temp_dir().join(format!("cutile_cuda_toolkit_{}", Uuid::new_v4()));
        let _old_tileiras = create_fake_cuda_toolkit(&old_dir, 13010, true);
        let new_tileiras = create_fake_cuda_toolkit(&new_dir, 13020, true);

        assert_eq!(
            resolve_tileiras_binary_with_candidates(
                None,
                None,
                &[old_dir.clone(), new_dir.clone()]
            ),
            new_tileiras
        );

        let _ = fs::remove_dir_all(old_dir);
        let _ = fs::remove_dir_all(new_dir);
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

    fn create_fake_cuda_toolkit(path: &Path, cuda_version: u32, include_tileiras: bool) -> PathBuf {
        let include_dir = path.join("include");
        let bin_dir = path.join("bin");
        fs::create_dir_all(&include_dir).unwrap();
        fs::create_dir_all(&bin_dir).unwrap();
        fs::write(
            include_dir.join("cuda.h"),
            format!("#define CUDA_VERSION {cuda_version}\n"),
        )
        .unwrap();

        let tileiras = bin_dir.join(tileiras_executable_name());
        if include_tileiras {
            fs::write(&tileiras, "").unwrap();
        }
        tileiras
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
