/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime utilities for compiling Tile IR modules to GPU cubins.
//! Provides GPU detection and bytecode compilation helpers.

use crate::error::JITError;
use cuda_core::{get_device_sm_name, Device};
use cutile_ir::bytecode::{write_bytecode_version, BytecodeVersion};
use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

/// Environment variable used to override the `tileiras` executable.
///
/// Set this to an absolute path such as `/opt/cuda-tile/bin/tileiras` to use
/// that binary instead of the `tileiras` found on `PATH`.
pub const TILEIRAS_PATH_ENV: &str = "CUTILE_TILEIRAS_PATH";
pub const SETUP_DIAGNOSTICS_ENV: &str = "CUTILE_SETUP_DIAGNOSTICS";

const CUDA_TOOLKIT_PATH_ENV: &str = "CUDA_TOOLKIT_PATH";
const MIN_CUDA_VERSION: u32 = 13020;

/// Environment variable to force the emitted Tile IR bytecode version
/// (e.g. `13.2`). Overrides toolkit detection and probing.
pub const BYTECODE_VERSION_ENV: &str = "CUTILE_BYTECODE_VERSION";

/// Returns the cutile compiler version (from the workspace Cargo.toml).
pub fn get_compiler_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Queries the CUDA driver to determine the SM architecture name (e.g. `"sm_90"`) for a device.
///
/// Cached per device: the driver is queried once per device and cache hits are
/// lock-free (`OnceLock::get` is an atomic load). CUDA device ordinals are small
/// and contiguous, so a fixed array of `OnceLock` suffices; an ordinal beyond it
/// (never in practice) skips the cache and queries the driver each time.
pub fn get_gpu_name(device_id: usize) -> String {
    const MAX_CACHED_DEVICES: usize = 64;
    static NAMES: [OnceLock<String>; MAX_CACHED_DEVICES] =
        [const { OnceLock::new() }; MAX_CACHED_DEVICES];

    let query = || -> String {
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
    };

    match NAMES.get(device_id) {
        Some(slot) => slot.get_or_init(query).clone(),
        None => query(),
    }
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
) -> (PathBuf, Option<PathBuf>) {
    resolve_tileiras_with_toolkit_candidates(
        tileiras_override,
        cuda_toolkit_path,
        default_cuda_toolkit_candidates(),
    )
}

/// Resolves the `tileiras` binary and, when it was found via a CUDA toolkit
/// (not a `CUTILE_TILEIRAS_PATH` override or bare `PATH`), the toolkit root used
/// to locate `cuda.h` for bytecode-version selection.
fn resolve_tileiras_with_toolkit_candidates(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<OsString>,
    default_cuda_toolkit_candidates: &[PathBuf],
) -> (PathBuf, Option<PathBuf>) {
    if let Some(path) = tileiras_override.filter(|value| !value.as_os_str().is_empty()) {
        let path = PathBuf::from(path);
        emit_setup_diagnostic(format_args!("using {TILEIRAS_PATH_ENV}={}", path.display()));
        // An overridden binary may be newer than the installed CTK, so its
        // version is decided by probing rather than the toolkit's cuda.h.
        return (path, None);
    }

    if let Some(path) = cuda_toolkit_tileiras(cuda_toolkit_path) {
        if path.is_file() {
            let toolkit = toolkit_root_of(&path);
            return (path, toolkit);
        }
    }

    if let Some(path) = default_cuda_toolkit_tileiras(default_cuda_toolkit_candidates) {
        let toolkit = toolkit_root_of(&path);
        return (path, toolkit);
    }

    emit_setup_diagnostic(format_args!(
        "falling back to {} through PATH lookup",
        tileiras_executable_name()
    ));
    (PathBuf::from(tileiras_executable_name()), None)
}

/// CUDA toolkit root for a `<root>/bin/tileiras` path (strips `bin/tileiras`).
fn toolkit_root_of(tileiras: &Path) -> Option<PathBuf> {
    tileiras.parent()?.parent().map(PathBuf::from)
}

/// Test-only helper that returns just the resolved `tileiras` path.
#[cfg(test)]
fn resolve_tileiras_binary_with_candidates(
    tileiras_override: Option<OsString>,
    cuda_toolkit_path: Option<OsString>,
    default_cuda_toolkit_candidates: &[PathBuf],
) -> PathBuf {
    resolve_tileiras_with_toolkit_candidates(
        tileiras_override,
        cuda_toolkit_path,
        default_cuda_toolkit_candidates,
    )
    .0
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
    tileiras_and_toolkit().0
}

/// Identifies which `tileiras` binary compiled a cubin.
///
/// This belongs in every cache key that names a cubin: without it, upgrading the
/// toolkit leaves the key unchanged and a cubin built by the previous `tileiras`
/// is served as a hit.
///
/// The fingerprint is the `--version` stdout. It carries the build number
/// (`Build local.local.37905922_`), and unlike `(size, mtime)` it survives a
/// reinstall of the same toolkit, so the cache stays warm. Measured cost on
/// CUDA 13.3: under 5 ms, `maxrss` 21.4 MB. Cached per process, and the key path
/// runs on cache hits too, so this must stay cheap.
///
/// Note it does not distinguish two binaries that report the same version, such
/// as a locally patched one.
///
/// Falls back to `(canonical path, size, mtime)` when `--version` fails, which
/// covers a future `tileiras` that drops the flag. An empty fingerprint is never
/// returned: that would drop the compiler out of the key.
pub fn tileiras_fingerprint() -> &'static str {
    fingerprint_of(&tileiras_binary())
}

/// Fingerprint of a specific resolved `tileiras`, cached **per path** rather than
/// once per process. A process that switches `CUTILE_TILEIRAS_PATH` mid-run then
/// keys entries by the binary actually in effect, not the one seen at the first
/// call — otherwise cubins built by the new binary are stored under the old
/// binary's fingerprint and served to a process that genuinely uses the old one.
/// Mirrors [`cached_bytecode_version`]. The `--version` spawn happens once per
/// distinct binary; the interned string lives for the process (bounded: one per
/// tileiras path, normally one).
fn fingerprint_of(tileiras: &Path) -> &'static str {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, &'static str>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&fp) = cache.lock().unwrap().get(tileiras) {
        return fp;
    }
    let fp: &'static str = Box::leak(compute_tileiras_fingerprint(tileiras).into_boxed_str());
    cache.lock().unwrap().insert(tileiras.to_path_buf(), fp);
    fp
}

fn compute_tileiras_fingerprint(tileiras: &Path) -> String {
    if let Ok(output) = Command::new(tileiras).arg("--version").output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !version.is_empty() {
                return version;
            }
        }
    }
    emit_setup_diagnostic(format_args!(
        "{} --version failed; fingerprinting it by path, size and mtime instead",
        tileiras.display()
    ));
    stat_fingerprint(tileiras)
}

/// `(canonical path, size, mtime)`, the fallback when `--version` is unavailable.
///
/// Weaker than the version string in one direction: reinstalling the same
/// toolkit changes `mtime`, so every key changes and the disk cache misses
/// across the board. That costs one recompile per kernel, not correctness.
fn stat_fingerprint(tileiras: &Path) -> String {
    let path = std::fs::canonicalize(tileiras).unwrap_or_else(|_| tileiras.to_path_buf());
    let (len, mtime_ns) = std::fs::metadata(&path)
        .map(|meta| {
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            (meta.len(), mtime_ns)
        })
        .unwrap_or((0, 0));
    format!("stat\0{}\0{len}\0{mtime_ns}", path.display())
}

/// Resolves `tileiras` together with the CUDA toolkit root (when applicable),
/// using the active `CUTILE_TILEIRAS_PATH` / `CUDA_TOOLKIT_PATH` environment.
fn tileiras_and_toolkit() -> (PathBuf, Option<PathBuf>) {
    resolve_tileiras_binary(
        env::var_os(TILEIRAS_PATH_ENV),
        env::var_os(CUDA_TOOLKIT_PATH_ENV),
    )
}

// =========================================================================
// Bytecode version selection
//
// The writer and decoder are already version-aware; this decides which
// version to emit so a newer toolchain default (13.3) is not handed to an
// older `tileiras`.
// =========================================================================

/// Selects the Tile IR bytecode version to emit for the active toolchain,
/// caching the result per resolved (tileiras, toolkit) pair. Resolution order:
///
/// 1. `CUTILE_BYTECODE_VERSION` — explicit override (e.g. `13.2`).
/// 2. The toolkit's `cuda.h` `CUDA_VERSION` — the coherent-install case.
/// 3. Probing the resolved `tileiras` — the override / bare `PATH` case, where
///    no trusted toolkit `cuda.h` is available.
///
/// The result is clamped to `[MIN_SUPPORTED, CURRENT]`. Feature
/// incompatibilities (e.g. an FP4 kernel against a 13.2 toolchain) are left for
/// `tileiras` to diagnose rather than pre-checked here.
fn selected_bytecode_version() -> BytecodeVersion {
    let (tileiras, toolkit) = tileiras_and_toolkit();
    cached_bytecode_version(&tileiras, toolkit.as_deref())
}

fn cached_bytecode_version(tileiras: &Path, toolkit_dir: Option<&Path>) -> BytecodeVersion {
    static CACHE: OnceLock<Mutex<HashMap<(PathBuf, Option<PathBuf>), BytecodeVersion>>> =
        OnceLock::new();
    let key = (tileiras.to_path_buf(), toolkit_dir.map(PathBuf::from));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&version) = cache.lock().unwrap().get(&key) {
        return version;
    }
    let version = compute_bytecode_version(tileiras, toolkit_dir);
    cache.lock().unwrap().insert(key, version);
    version
}

fn compute_bytecode_version(tileiras: &Path, toolkit_dir: Option<&Path>) -> BytecodeVersion {
    if let Some(value) = env::var_os(BYTECODE_VERSION_ENV).filter(|v| !v.is_empty()) {
        let text = value.to_string_lossy();
        match parse_bytecode_version(&text) {
            Some(version) => {
                emit_setup_diagnostic(format_args!("{BYTECODE_VERSION_ENV}={version} (override)"));
                return version;
            }
            None => emit_setup_diagnostic(format_args!(
                "ignoring invalid {BYTECODE_VERSION_ENV}={text}"
            )),
        }
    }

    if let Some(dir) = toolkit_dir {
        let cuda_h = dir.join("include").join("cuda.h");
        if let Ok(cuda_version) = cuda_version_from_header(&cuda_h) {
            let version = bytecode_version_from_cuda_version(cuda_version);
            emit_setup_diagnostic(format_args!(
                "bytecode version {version} from {}",
                cuda_h.display()
            ));
            return version;
        }
    }

    let version = probe_max_supported_bytecode_version(tileiras);
    emit_setup_diagnostic(format_args!(
        "bytecode version {version} from probing {}",
        tileiras.display()
    ));
    version
}

/// Maps a CUDA `CUDA_VERSION` integer (e.g. `13030`) to a clamped bytecode version.
fn bytecode_version_from_cuda_version(cuda_version: u32) -> BytecodeVersion {
    let candidate = BytecodeVersion {
        major: (cuda_version / 1000) as u8,
        minor: ((cuda_version % 1000) / 10) as u8,
        tag: 0,
    };
    clamp_bytecode_version(candidate)
}

/// Parses a `major.minor[.tag]` string (e.g. `13.2`) to a clamped bytecode version.
fn parse_bytecode_version(text: &str) -> Option<BytecodeVersion> {
    let mut parts = text.trim().split('.');
    let major: u8 = parts.next()?.trim().parse().ok()?;
    let minor: u8 = parts.next()?.trim().parse().ok()?;
    let tag: u16 = match parts.next() {
        Some(part) => part.trim().parse().ok()?,
        None => 0,
    };
    if parts.next().is_some() {
        return None;
    }
    Some(clamp_bytecode_version(BytecodeVersion {
        major,
        minor,
        tag,
    }))
}

/// Clamps a version to the range this writer can emit.
fn clamp_bytecode_version(version: BytecodeVersion) -> BytecodeVersion {
    version
        .max(BytecodeVersion::MIN_SUPPORTED)
        .min(BytecodeVersion::CURRENT)
}

/// Probes `tileiras` for the newest bytecode version it accepts by compiling a
/// tiny empty module at each candidate version, newest first.
fn probe_max_supported_bytecode_version(tileiras: &Path) -> BytecodeVersion {
    let tmp_dir = env::temp_dir();
    for &version in BytecodeVersion::SUPPORTED.iter().rev() {
        let module = cutile_ir::Module::new("__cutile_probe");
        let Ok(bytes) = write_bytecode_version(&module, version) else {
            continue;
        };
        let base = tmp_dir.join(Uuid::new_v4().to_string());
        let bc_filename = format!("{}.bc", base.display());
        let cubin_filename = format!("{}.cubin", base.display());
        if std::fs::write(&bc_filename, &bytes).is_err() {
            continue;
        }
        let accepted = Command::new(tileiras)
            .args(["--gpu-name", "sm_120", "-o", &cubin_filename, &bc_filename])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        let _ = std::fs::remove_file(&bc_filename);
        let _ = std::fs::remove_file(&cubin_filename);
        if accepted {
            return version;
        }
    }
    emit_setup_diagnostic(format_args!(
        "could not probe a supported bytecode version from {}; using {}",
        tileiras.display(),
        BytecodeVersion::MIN_SUPPORTED
    ));
    BytecodeVersion::MIN_SUPPORTED
}

/// `--opt-level` passed to `tileiras`. Not configurable yet.
/// Numeric, not a string: the disk-cache key and entry header store it as one
/// byte.
pub const DEFAULT_OPT_LEVEL: u8 = 3;

/// A path removed when dropped, so the error paths clean up too.
struct ScopedTempFile(Option<PathBuf>);

impl ScopedTempFile {
    fn new(path: PathBuf) -> Self {
        Self(Some(path))
    }

    fn path(&self) -> &Path {
        self.0.as_deref().expect("path is taken only by `keep`")
    }

    /// Leaves the file on disk. Used for the `.bc` a failing `tileiras` run was
    /// given, which the error message points at.
    fn keep(mut self) {
        self.0 = None;
    }
}

impl Drop for ScopedTempFile {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Serializes a `cutile_ir::Module` to Tile IR bytecode (the `.bc` image).
///
/// Runs the module verifiers first, so bytecode returned here is what `tileiras`
/// is expected to accept. Together with the target and opt level, these bytes are
/// the complete input to [`run_tileiras`].
///
/// Also returns the [`BytecodeVersion`] actually written into the image, so the
/// disk-cache key names that exact version instead of re-resolving it (which
/// could drift from the bytes if the toolchain env changed in between).
pub fn serialize_tile_ir_bytecode(
    module: &cutile_ir::Module,
) -> Result<(Vec<u8>, BytecodeVersion), JITError> {
    module
        .verify_dominance()
        .map_err(|e| JITError::Generic(format!("tile-ir dominance verification failed: {e}")))?;

    module.verify_bytecode_indices().map_err(|e| {
        JITError::Generic(format!(
            "tile-ir bytecode value-index verification failed: {e}"
        ))
    })?;

    // Dump IR via unified CUTILE_DUMP mechanism (also honors legacy TILE_IR_DUMP).
    // `to_mlir_text` renders the whole module, so it stays behind `should_dump`.
    if crate::dump::should_dump(crate::dump::DumpStage::Ir) {
        crate::dump::dump_module(
            crate::dump::DumpStage::Ir,
            &module.name,
            &module.to_mlir_text(),
        );
    }

    let bytecode_version = selected_bytecode_version();
    let bytes = write_bytecode_version(module, bytecode_version).map_err(|e| {
        JITError::Generic(format!(
            "Failed to serialize bytecode for module {}: {e}",
            module.name
        ))
    })?;

    if crate::dump::should_dump(crate::dump::DumpStage::Bytecode) {
        let decoded = cutile_ir::decode_bytecode(&bytes)
            .unwrap_or_else(|e| format!("<bytecode decode failed: {e}>"));
        crate::dump::dump_module(crate::dump::DumpStage::Bytecode, &module.name, &decoded);
    }

    Ok((bytes, bytecode_version))
}

/// Compiles Tile IR bytecode to a cubin image by spawning `tileiras`.
///
/// `bytecode`, `gpu_name` and `opt_level`, plus the `tileiras` binary itself, are
/// the complete input to this stage.
///
/// The temporary `.bc` and `.cubin` are removed before returning. The one
/// exception is a failing `tileiras` run, which leaves the `.bc` on disk because
/// the error message names it.
pub fn run_tileiras(bytecode: &[u8], gpu_name: &str, opt_level: u8) -> Result<Vec<u8>, JITError> {
    let base_filename = env::temp_dir().join(Uuid::new_v4().to_string());
    let bc_file = ScopedTempFile::new(base_filename.with_extension("bc"));
    let cubin_file = ScopedTempFile::new(base_filename.with_extension("cubin"));
    let bc_filename = bc_file.path().to_string_lossy().into_owned();
    let cubin_filename = cubin_file.path().to_string_lossy().into_owned();

    std::fs::write(bc_file.path(), bytecode).map_err(|e| {
        JITError::Generic(format!("Failed to write bytecode for {bc_filename}: {e}"))
    })?;

    let tileiras = tileiras_binary();
    let opt_level_arg = opt_level.to_string();
    let args = [
        "--gpu-name",
        gpu_name,
        "--opt-level",
        &opt_level_arg,
        "-o",
        &cubin_filename,
        &bc_filename,
    ];
    let output = Command::new(&tileiras)
        .args(args)
        .output()
        .map_err(|e| JITError::Generic(tileiras_launch_error(&tileiras, &args, &bc_filename, e)))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        // The message points at the bytecode, so it has to outlive this call.
        bc_file.keep();
        return Err(JITError::Generic(format!(
            "{} failed while compiling Tile IR bytecode.\n\
             status: {}\n\
             command: {}\n\
             target gpu: {gpu_name}\n\
             bytecode: {bc_filename} (kept for inspection)\n\
             output cubin: {cubin_filename}\n\
             stdout:\n{stdout}\n\
             stderr:\n{stderr}\n\
             hint: run with CUTILE_DUMP=ir,bytecode to include the generated Tile IR and decoded bytecode in stderr.",
            tileiras.display(),
            output.status,
            display_command(&tileiras, &args),
        )));
    }

    let cubin = std::fs::read(cubin_file.path()).map_err(|e| {
        JITError::Generic(format!(
            "{} reported success but its output cubin at {cubin_filename} could not be read: {e}",
            tileiras.display(),
        ))
    })?;
    if cubin.is_empty() {
        return Err(JITError::Generic(format!(
            "{} reported success but wrote an empty cubin at {cubin_filename}",
            tileiras.display(),
        )));
    }
    crate::jit_store::record_backend_compile();
    Ok(cubin)
}

/// Where a stage-2 cubin came from.
///
/// `DiskCache` carries the store it was served from and the key it validated
/// against, so a `cuModuleLoadData` rejection can evict *that exact entry* from
/// *that exact store* and recompile — see [`recompile_after_disk_rejection`] —
/// without re-deriving the key (which would drift if `opt_level` ever became
/// configurable) or re-reading a possibly-swapped global store slot.
#[derive(Clone)]
pub enum Stage2Source {
    Tileiras,
    DiskCache {
        store: Arc<dyn crate::jit_store::JitStore>,
        key: String,
    },
}

/// Compiles Tile IR bytecode to a cubin, consulting the disk cache when one is
/// installed (see [`crate::jit_store::enable`]).
///
/// The lookup sits exactly between bytecode serialization and the `tileiras`
/// spawn: `bytecode` plus `gpu_name`, `opt_level` and the resolved `tileiras`
/// are the subprocess's complete input, so the content-addressed key derived
/// from them (see [`crate::jit_store::l2_key`]) is correct by construction.
///
/// Store I/O failures are soft: counted in `stats().io_errors`, logged, and
/// the compile proceeds as if no cache were installed.
pub fn compile_bytecode_cached(
    bytecode: &[u8],
    bc_version: BytecodeVersion,
    gpu_name: &str,
    opt_level: u8,
) -> Result<(Vec<u8>, Stage2Source), JITError> {
    use crate::jit_store::{self, EntryParams};
    use sha2::{Digest, Sha256};
    use std::sync::atomic::Ordering;

    let Some(store) = jit_store::installed_store() else {
        return run_tileiras(bytecode, gpu_name, opt_level).map(|c| (c, Stage2Source::Tileiras));
    };

    // `bc_version` is the version the caller actually serialized into `bytecode`,
    // not a fresh re-resolution — so the key's version field can never disagree
    // with the bytes it sits next to (see #7).
    let tileiras_fp = tileiras_fingerprint();
    let key = jit_store::l2_key(bytecode, bc_version, gpu_name, opt_level, tileiras_fp);
    let params = EntryParams {
        bc_sha256: Sha256::digest(bytecode).into(),
        gpu_name,
        opt_level,
        tileiras_fp,
    };

    match store.get(&key) {
        Ok(Some(entry)) => {
            if let Some(cubin) = jit_store::decode_entry(&entry, &params) {
                jit_store::STATS.hits.fetch_add(1, Ordering::Relaxed);
                // Hand the store and key back so a driver rejection recovers
                // against this exact entry (see `recompile_after_disk_rejection`).
                return Ok((cubin, Stage2Source::DiskCache { store, key }));
            }
            // Key matched but the entry does not validate against this request:
            // a torn write, tampering, or a hash collision. Drop it and recompile
            // rather than ever serving it.
            crate::jit_store::cache_log(format_args!(
                "disk cache entry {key} failed validation; deleting and recompiling"
            ));
            if let Err(e) = store.delete(&key) {
                jit_store::STATS.io_errors.fetch_add(1, Ordering::Relaxed);
                crate::jit_store::cache_log(format_args!(
                    "failed to delete invalid entry {key}: {e}"
                ));
            }
        }
        Ok(None) => {}
        Err(e) => {
            jit_store::STATS.io_errors.fetch_add(1, Ordering::Relaxed);
            crate::jit_store::cache_log(format_args!("disk cache read for {key} failed: {e}"));
        }
    }

    jit_store::STATS.misses.fetch_add(1, Ordering::Relaxed);
    let cubin = run_tileiras(bytecode, gpu_name, opt_level)?;

    match jit_store::encode_entry(&params, &cubin) {
        Some(entry) => match store.put(&key, &entry) {
            Ok(()) => {
                jit_store::STATS.puts.fetch_add(1, Ordering::Relaxed);
                jit_store::STATS
                    .bytes_written
                    .fetch_add(entry.len() as u64, Ordering::Relaxed);
            }
            Err(e) => {
                jit_store::STATS.io_errors.fetch_add(1, Ordering::Relaxed);
                crate::jit_store::cache_log(format_args!("disk cache write for {key} failed: {e}"));
            }
        },
        None => crate::jit_store::cache_log(format_args!(
            "not caching {key}: gpu name or tileiras fingerprint exceeds the entry format's u16 length field"
        )),
    }

    Ok((cubin, Stage2Source::Tileiras))
}

/// Recovery for a disk-served cubin the driver rejected (a torn write the
/// checksum missed, driver/toolkit skew, …).
///
/// Deletes the offending entry from the store it came from (best-effort) and
/// compiles with `tileiras` **directly, without consulting the cache**. The
/// bypass is the point: if the delete fails — a read-only or shared cache
/// directory, an entry owned by another user — reading the store again would
/// just re-serve the very cubin the driver already rejected, and the launch
/// would fail permanently. `store` and `key` come from the [`Stage2Source::DiskCache`]
/// that produced the bad cubin, so this evicts exactly that entry.
pub fn recompile_after_disk_rejection(
    store: &dyn crate::jit_store::JitStore,
    key: &str,
    bytecode: &[u8],
    gpu_name: &str,
    opt_level: u8,
) -> Result<Vec<u8>, JITError> {
    use std::sync::atomic::Ordering;

    if let Err(e) = store.delete(key) {
        crate::jit_store::STATS
            .io_errors
            .fetch_add(1, Ordering::Relaxed);
        crate::jit_store::cache_log(format_args!("failed to evict entry {key}: {e}"));
    }
    run_tileiras(bytecode, gpu_name, opt_level)
}

/// Compiles a `cutile_ir::Module` to a cubin image via bytecode serialization and
/// `tileiras`, consulting the disk cache when one is installed.
///
/// Returns `Err` (not panic) on any failure so callers can propagate it and run
/// their cache-cleanup paths; a panic would unwind past that and across FFI frames.
pub fn compile_tile_ir_module(
    module: &cutile_ir::Module,
    gpu_name: &str,
) -> Result<Vec<u8>, JITError> {
    let (bytecode, bc_version) = serialize_tile_ir_bytecode(module)?;
    compile_bytecode_cached(&bytecode, bc_version, gpu_name, DEFAULT_OPT_LEVEL).map(|(cubin, _)| cubin)
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

/// Returns whether the environment variable `var` is set to a truthy value
/// (`1` / `true` / `yes` / `on`, case-insensitive, surrounding whitespace ignored).
///
/// Shared by the crate's on/off diagnostic env vars so they all parse the same way.
pub fn env_flag_enabled(var: &str) -> bool {
    env::var(var).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn setup_diagnostics_enabled() -> bool {
    env_flag_enabled(SETUP_DIAGNOSTICS_ENV)
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
    fn maps_cuda_version_to_bytecode_version() {
        assert_eq!(
            bytecode_version_from_cuda_version(13030),
            BytecodeVersion::V13_3
        );
        assert_eq!(
            bytecode_version_from_cuda_version(13020),
            BytecodeVersion::V13_2
        );
        assert_eq!(
            bytecode_version_from_cuda_version(13010),
            BytecodeVersion::V13_1
        );
        // Out-of-range values clamp into [MIN_SUPPORTED, CURRENT].
        assert_eq!(
            bytecode_version_from_cuda_version(13000),
            BytecodeVersion::MIN_SUPPORTED
        );
        assert_eq!(
            bytecode_version_from_cuda_version(13040),
            BytecodeVersion::CURRENT
        );
    }

    #[test]
    fn parses_bytecode_version_override() {
        assert_eq!(parse_bytecode_version("13.2"), Some(BytecodeVersion::V13_2));
        assert_eq!(
            parse_bytecode_version(" 13.3 "),
            Some(BytecodeVersion::V13_3)
        );
        assert_eq!(
            parse_bytecode_version("13.3.0"),
            Some(BytecodeVersion::V13_3)
        );
        // Out-of-range clamps to CURRENT; malformed input is rejected.
        assert_eq!(
            parse_bytecode_version("13.9"),
            Some(BytecodeVersion::CURRENT)
        );
        assert_eq!(parse_bytecode_version("13"), None);
        assert_eq!(parse_bytecode_version("nonsense"), None);
        assert_eq!(parse_bytecode_version("13.2.3.4"), None);
    }

    #[test]
    fn selects_bytecode_version_from_toolkit_cuda_h() {
        let temp_dir = env::temp_dir().join(format!("cutile_bc_ver_{}", Uuid::new_v4()));
        let tileiras = create_fake_cuda_toolkit(&temp_dir, 13020, true);
        let toolkit = toolkit_root_of(&tileiras);
        assert_eq!(toolkit.as_deref(), Some(temp_dir.as_path()));
        // cuda.h reports CUDA 13.2, so we emit bytecode 13.2 without probing.
        assert_eq!(
            compute_bytecode_version(&tileiras, toolkit.as_deref()),
            BytecodeVersion::V13_2
        );
        let _ = fs::remove_dir_all(temp_dir);
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
        let cubin = compile_tile_ir_module(&module, "sm_120")
            .expect("compiling an empty kernel with the fake tileiras should succeed");

        let args_path = fake_tileiras.with_extension("args");
        let args = fs::read_to_string(&args_path).unwrap();
        assert!(
            args.lines()
                .next()
                .is_some_and(|line| line == fake_tileiras.to_string_lossy()),
            "expected fake tileiras to record its own path, got:\n{args}"
        );
        assert!(args.contains("--gpu-name\nsm_120"), "args:\n{args}");
        assert!(args.contains("--opt-level\n3"), "args:\n{args}");
        assert!(args.contains("-o\n"), "args:\n{args}");

        // `write_fake_tileiras` writes exactly this to the `-o` path.
        assert_eq!(cubin, b"fake cubin\n".to_vec());

        // Both temp files are removed before `run_tileiras` returns. The fake
        // tileiras recorded their paths, so check them directly.
        let cubin_path = {
            let mut lines = args.lines();
            lines.find(|line| *line == "-o");
            lines
                .next()
                .expect("fake tileiras should have recorded an -o path")
        };
        let bc_path = args.lines().last().unwrap_or_default();
        assert!(
            !PathBuf::from(cubin_path).exists(),
            "run_tileiras leaked its output cubin at {cubin_path}"
        );
        assert!(
            !PathBuf::from(bc_path).exists(),
            "run_tileiras leaked its input bytecode at {bc_path}"
        );

        let _ = fs::remove_file(args_path);
        let _ = fs::remove_file(fake_tileiras);
        let _ = fs::remove_dir(temp_dir);
    }

    /// End-to-end cache path with a fake `tileiras`: the first compile spawns
    /// the subprocess and writes the store entry, the second is served from
    /// disk without spawning. `enable`/`disable` happen under `ENV_LOCK`, the
    /// same lock the other tileiras-spawning test takes, so the global store
    /// never leaks into it.
    #[test]
    #[cfg(unix)]
    fn disk_cache_serves_second_compile_without_spawning() {
        let _env_guard = ENV_LOCK.lock().unwrap();
        let temp_dir = env::temp_dir().join(format!("cutile_jit_cache_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let fake_tileiras = temp_dir.join("tileiras");
        write_fake_tileiras(&fake_tileiras);
        let _tileiras_env = EnvVarGuard::set(TILEIRAS_PATH_ENV, &fake_tileiras);

        let store_dir = temp_dir.join("store");
        crate::jit_store::enable(std::sync::Arc::new(
            crate::jit_store::FileSystemJitStore::new(&store_dir).unwrap(),
        ));

        let module = empty_kernel_module();
        let backend_before = crate::jit_store::jit_backend_compile_count();
        let hits_before = crate::jit_store::jit_disk_hit_count();

        let first =
            compile_tile_ir_module(&module, "sm_120").expect("first compile (miss) should succeed");
        let second =
            compile_tile_ir_module(&module, "sm_120").expect("second compile (hit) should succeed");

        // A different target is a different key: this one must miss.
        let other_arch = compile_tile_ir_module(&module, "sm_100")
            .expect("different-arch compile should succeed");

        crate::jit_store::disable();

        assert_eq!(first, second, "hit must return the exact bytes stored");
        assert_eq!(first, other_arch, "fake tileiras writes constant bytes");
        assert_eq!(
            crate::jit_store::jit_backend_compile_count() - backend_before,
            2,
            "exactly the two misses spawn tileiras"
        );
        assert_eq!(
            crate::jit_store::jit_disk_hit_count() - hits_before,
            1,
            "exactly the repeat compile hits the disk"
        );

        let _ = fs::remove_dir_all(&temp_dir);
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
