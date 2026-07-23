/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile kernel compilation, caching, launching, and partitioning for CUDA device operations.

use anyhow::{Context, Result};
use cuda_async::error::DeviceError;
use cuda_core::DType;
use cuda_core::{memcpy_dtoh_async, Function};
use cutile_compiler::ast::Module;
use cutile_compiler::compiler::{CUDATileFunctionCompiler, CUDATileModules};
use cutile_compiler::cuda_tile_runtime_utils::{
    compile_bytecode_cached, env_flag_enabled, get_compiler_version, get_gpu_name,
    recompile_after_disk_rejection, serialize_tile_ir_bytecode, tileiras_fingerprint, Stage2Source,
    DEFAULT_OPT_LEVEL,
};
use cutile_compiler::specialization::{DivHint, SpecializationBits};
use dashmap::DashMap;
use once_cell::sync::OnceCell;
use std::alloc::{alloc, Layout};
use std::fs;
use std::future::IntoFuture;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

// JIT diagnostic logging (set CUTILE_JIT_LOG=1, true, yes, or on to enable)

fn jit_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag_enabled("CUTILE_JIT_LOG"))
}

macro_rules! jit_log {
    ($($arg:tt)*) => {
        if jit_log_enabled() {
            eprintln!("[cutile::jit] {}", format!($($arg)*));
        }
    };
}

static JIT_COMPILE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Process-global JIT compile counter: +1 per successful compile, +0 on cache
/// hits and on failed compiles. Equals the number of distinct kernels cached.
/// Snapshot before a call and check the delta to get exact miss counts.
///
/// A disk-cache hit still counts: the counter tracks in-memory misses, which
/// run the compiler frontend either way. Absent failures,
/// `jit_compile_count == jit_backend_compile_count + jit_disk_hit_count`
/// (both in [`crate::jit_cache`]).
pub fn jit_compile_count() -> u64 {
    JIT_COMPILE_COUNT.load(Ordering::Relaxed)
}

#[inline]
fn record_jit_compile() {
    JIT_COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);
}

use crate::error::*;
use crate::tensor::{IntoPartition, IntoPartitionArc, Partition, Tensor};

pub use cuda_async::{
    device_buffer::*, device_context::*, device_future::*, device_operation::*, launch::*,
    scheduling_policies::*,
};

pub use cutile_compiler::compiler::utils::CompileOptions;

/// Cache key for a compiled tile kernel.
///
/// Two kernel invocations that share the same `TileFunctionKey` can reuse the same compiled
/// CUDA module and function, avoiding recompilation. The key captures everything that can
/// change the generated GPU code: module name, function name, generic type/const parameters,
/// tensor stride layouts, (optionally) the launch grid, compile options, source hash,
/// GPU architecture, compiler version, and the `tileiras` binary that assembles the cubin.
///
/// Tensor extents are deliberately absent: `stride_args` records only which
/// dimensions have stride 1, and `spec_args` only power-of-two divisibility. A
/// `[1024, 1024]` and a `[4096, 4096]` matmul share one key, because extents are
/// runtime kernel arguments and do not reach the generated code.
///
/// `source_hash` covers the kernel's own module, not the dependency modules the
/// use-graph links in. Editing a helper module that the kernel
/// calls changes the cubin without changing this field. Within a process this is
/// harmless, since a rebuild restarts it; it is why the on-disk cache keys on the
/// serialized bytecode rather than on this struct.
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct TileFunctionKey {
    module_name: String,
    function_name: String,
    pub function_generics: Vec<String>,
    pub stride_args: Vec<(String, Vec<i32>)>,
    pub spec_args: Vec<(String, SpecializationBits)>,
    pub scalar_hints: Vec<(String, DivHint)>,
    pub grid: Option<(u32, u32, u32)>,
    pub compile_options: CompileOptions,
    source_hash: String,
    device_id: usize,
    gpu_name: String,
    compiler_version: String,
    /// Output of `tileiras --version`, not `nvcc --version`: the JIT resolves
    /// `tileiras` on its own, so `CUTILE_TILEIRAS_PATH` can point at a binary the
    /// toolkit version knows nothing about.
    tileiras_fingerprint: String,
}

/// Builder for [`TileFunctionKey`].
///
/// With 11 positional arguments it is easy to silently transpose two `String`
/// fields and produce a wrong-but-valid key. The builder makes each field
/// self-documenting and keeps future additions backward-compatible.
///
/// # Example
///
/// ```rust,ignore
/// let key = TileFunctionKey::builder("linalg", "matmul")
///     .generics(vec!["f32".into(), "128".into()])
///     .source_hash(linalg::_SOURCE_HASH)
///     .device_id(device_id)
///     .gpu_name(get_gpu_name(device_id))
///     .compiler_version(get_compiler_version())
///     .tileiras_fingerprint(tileiras_fingerprint())
///     .build();
/// ```
pub struct TileFunctionKeyBuilder {
    module_name: String,
    function_name: String,
    function_generics: Vec<String>,
    stride_args: Vec<(String, Vec<i32>)>,
    spec_args: Vec<(String, SpecializationBits)>,
    scalar_hints: Vec<(String, DivHint)>,
    grid: Option<(u32, u32, u32)>,
    compile_options: CompileOptions,
    source_hash: String,
    device_id: usize,
    gpu_name: String,
    compiler_version: String,
    tileiras_fingerprint: String,
}

impl TileFunctionKeyBuilder {
    pub fn generics(mut self, generics: Vec<String>) -> Self {
        self.function_generics = generics;
        self
    }
    pub fn stride_args(mut self, stride_args: Vec<(String, Vec<i32>)>) -> Self {
        self.stride_args = stride_args;
        self
    }
    pub fn spec_args(mut self, spec_args: Vec<(String, SpecializationBits)>) -> Self {
        self.spec_args = spec_args;
        self
    }
    pub fn scalar_hints(mut self, scalar_hints: Vec<(String, DivHint)>) -> Self {
        self.scalar_hints = scalar_hints;
        self
    }
    pub fn grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.grid = Some(grid);
        self
    }
    pub fn compile_options(mut self, options: CompileOptions) -> Self {
        self.compile_options = options;
        self
    }
    pub fn source_hash(mut self, hash: impl Into<String>) -> Self {
        self.source_hash = hash.into();
        self
    }
    pub fn device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }
    pub fn gpu_name(mut self, name: impl Into<String>) -> Self {
        self.gpu_name = name.into();
        self
    }
    pub fn compiler_version(mut self, version: impl Into<String>) -> Self {
        self.compiler_version = version.into();
        self
    }
    /// Output of `tileiras --version`; see [`tileiras_fingerprint`].
    pub fn tileiras_fingerprint(mut self, fingerprint: impl Into<String>) -> Self {
        self.tileiras_fingerprint = fingerprint.into();
        self
    }
    pub fn build(self) -> TileFunctionKey {
        TileFunctionKey {
            module_name: self.module_name,
            function_name: self.function_name,
            function_generics: self.function_generics,
            stride_args: self.stride_args,
            spec_args: self.spec_args,
            scalar_hints: self.scalar_hints,
            grid: self.grid,
            compile_options: self.compile_options,
            source_hash: self.source_hash,
            device_id: self.device_id,
            gpu_name: self.gpu_name,
            compiler_version: self.compiler_version,
            tileiras_fingerprint: self.tileiras_fingerprint,
        }
    }
}

impl TileFunctionKey {
    /// Start building a key with required `module_name` and `function_name`.
    /// All other fields default to empty / `None` / `default()`.
    pub fn builder(
        module_name: impl Into<String>,
        function_name: impl Into<String>,
    ) -> TileFunctionKeyBuilder {
        TileFunctionKeyBuilder {
            module_name: module_name.into(),
            function_name: function_name.into(),
            function_generics: vec![],
            stride_args: vec![],
            spec_args: vec![],
            scalar_hints: vec![],
            grid: None,
            compile_options: CompileOptions::default(),
            source_hash: String::new(),
            device_id: 0,
            gpu_name: String::new(),
            compiler_version: String::new(),
            tileiras_fingerprint: String::new(),
        }
    }
}

impl FunctionKey for TileFunctionKey {}

// ── Global kernel cache (process-wide, cross-thread) ────────────────────────

/// Global kernel cache. `DashMap` for cross-thread sharing; inner `OnceCell` for
/// single-flight compilation dedup (if multiple threads need the same kernel,
/// only one compiles while the rest wait). `once_cell::sync::OnceCell` gives
/// fallible initialization (`get_or_try_init`).
///
/// Keyed on the whole [`TileFunctionKey`], not on a digest of it: 64 bits collide
/// often enough to matter once a process caches many kernels, and a collision
/// here hands back a cubin compiled for a different kernel.
///
/// Intentionally unbounded: no cap or LRU. Capacity management lives in the L2
/// disk cache, not here — the same shape as cutile-python (unbounded in-memory
/// kernel cache, 2 GiB LRU on disk). Bounding L1 is a harder problem than L2:
/// evicting a `CompiledKernel` unloads its `Module`, which may still be
/// executing on the GPU, whereas deleting an L2 file is always safe.
static KERNEL_CACHE: OnceLock<DashMap<TileFunctionKey, Arc<OnceCell<CompiledKernel>>>> =
    OnceLock::new();

pub fn get_kernel_cache() -> &'static DashMap<TileFunctionKey, Arc<OnceCell<CompiledKernel>>> {
    KERNEL_CACHE.get_or_init(DashMap::new)
}

/// Get (or create) the single-flight compilation slot for `key`.
///
/// The returned `OnceCell` lets the caller `get_or_try_init` the compile
/// exactly once across threads. The DashMap shard lock is released before
/// this returns, so the slow compile never holds it.
///
/// Hits take the read path (shard read lock, no allocation); only a miss falls
/// back to `entry()` (write lock + owned key).
pub fn kernel_cache_slot(key: &TileFunctionKey) -> Arc<OnceCell<CompiledKernel>> {
    let cache = get_kernel_cache();
    if let Some(existing) = cache.get(key) {
        return Arc::clone(existing.value());
    }
    // `get` returned None holding no lock, so the write path is deadlock-free;
    // `or_insert_with` still resolves a concurrent insert into one slot per key.
    Arc::clone(
        cache
            .entry(key.clone())
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .value(),
    )
}

/// Check whether a kernel with the given key has already been compiled and cached.
pub fn contains_cuda_function(key: &TileFunctionKey) -> bool {
    get_kernel_cache()
        .get(key)
        .is_some_and(|slot| slot.value().get().is_some())
}

/// Reads Tile IR text from a file.
///
/// This helper function reads intermediate representation files from disk, typically
/// for debugging purposes when using `use_debug_mlir` or similar options.
///
/// ## Parameters
///
/// - `path`: Path to the IR file to read
///
/// ## Returns
///
/// The file contents as a UTF-8 string, or an I/O error if reading fails.
#[expect(unused)]
fn read_ir(path: String) -> Result<String, std::io::Error> {
    let s = String::from_utf8(fs::read(path)?).expect("Unable to convert from utf8 to string.");
    Ok(s)
}

/// Writes Tile IR text to a file for debugging.
///
/// This helper function writes intermediate representation to disk when kernel functions
/// are marked with `dump_mlir_dir` entry attributes. The filename
/// includes the module name, function name, and cache hash for uniqueness.
///
/// ## Parameters
///
/// - `module_name`: Name of the module containing the kernel
/// - `function_name`: Name of the kernel function
/// - `cache_hash_str`: Unique hash identifying this compilation
/// - `extension`: File extension (usually "mlir" for the MLIR-like Tile IR text)
/// - `dir`: Directory to write the file to
/// - `contents`: IR contents to write
///
/// ## Panics
///
/// Panics if the file cannot be written.
fn write_ir(
    module_name: &str,
    function_name: &str,
    cache_hash_str: &str,
    extension: &str,
    dir: &str,
    contents: &str,
) {
    let filename = format!("{module_name}_{function_name}_{cache_hash_str}.{extension}");
    let path = PathBuf::from(dir).join(filename);
    fs::write(path.clone(), contents).unwrap_or_else(|_| panic!("Failed to write {path:?}")); // Writes the string as bytes
    println!("IR written to {path:?}");
}

// ── Single-flight compilation dedup is handled by once_cell::sync::OnceCell ──

/// Compiles one tile-function specialization to a CUBIN and loads it into a
/// [`CompiledKernel`].
///
/// This is the single compile-and-load core behind [`compile_from_context`],
/// which serves both real `.sync()` / `.await` launches and the `.compile()`
/// warmup terminal. It runs the compiler, honors the `print_ir` /
/// `dump_mlir_dir` entry attributes,
/// lowers to a CUBIN, loads the module, and resolves `function_entry`, emitting
/// per-stage `CUTILE_JIT_TIMING` along the way.
///
/// Callers own the cache concerns: they build the [`TileFunctionKey`], dedup via
/// the cache slot, and call [`record_jit_compile`]. This function assumes it
/// runs exactly once per cache miss and does no caching itself.
#[allow(clippy::too_many_arguments)]
fn compile_and_load_kernel(
    modules: &CUDATileModules,
    module_name: &str,
    function_name: &str,
    function_entry: &str,
    generics: &[String],
    stride_args: &[(String, Vec<i32>)],
    spec_args: &[(String, SpecializationBits)],
    scalar_hints: &[(String, DivHint)],
    const_grid: Option<(u32, u32, u32)>,
    gpu_name: &str,
    compile_options: &CompileOptions,
    device_id: usize,
    key_str: &str,
) -> Result<CompiledKernel, Error> {
    let t0 = std::time::Instant::now();

    let stride_args_refs: Vec<(&str, &[i32])> = stride_args
        .iter()
        .map(|x| (x.0.as_str(), x.1.as_slice()))
        .collect();
    let spec_args_refs: Vec<(&str, &SpecializationBits)> =
        spec_args.iter().map(|x| (x.0.as_str(), &x.1)).collect();
    let scalar_hints_refs: Vec<(&str, &DivHint)> =
        scalar_hints.iter().map(|x| (x.0.as_str(), &x.1)).collect();

    let stage1_start = std::time::Instant::now();
    let (tile_module, validator) = {
        let compiler = CUDATileFunctionCompiler::new(
            modules,
            module_name,
            function_name,
            generics,
            &stride_args_refs,
            &spec_args_refs,
            &scalar_hints_refs,
            const_grid,
            gpu_name.to_string(),
            compile_options,
        )?;
        let validator = Arc::new(compiler.get_validator());
        let tile_module = compiler.compile()?;
        (tile_module, validator)
    };
    let stage1_ms = stage1_start.elapsed().as_secs_f64() * 1000.0;

    let stage2_start = std::time::Instant::now();
    {
        let print_ir =
            modules.get_entry_arg_bool_by_function_name(module_name, function_name, "print_ir")?;
        let dump_mlir_dir = modules.get_entry_arg_string_by_function_name(
            module_name,
            function_name,
            "dump_mlir_dir",
        )?;
        // `to_mlir_text` renders the whole module; only pay for it when asked.
        if print_ir || dump_mlir_dir.is_some() {
            let ir_text = tile_module.to_mlir_text();
            if print_ir {
                println!("COMPILED IR: {module_name}::{function_name}\n{ir_text}");
            }
            if let Some(path) = dump_mlir_dir {
                write_ir(
                    module_name,
                    function_name,
                    key_str,
                    "mlir",
                    path.as_str(),
                    ir_text.as_str(),
                );
            }
        }
    }
    let (bytecode, bc_version) = serialize_tile_ir_bytecode(&tile_module)?;
    let (cubin, mut stage2_source) =
        compile_bytecode_cached(&bytecode, bc_version, gpu_name, DEFAULT_OPT_LEVEL)?;
    let mut stage2_ms = stage2_start.elapsed().as_secs_f64() * 1000.0;

    // A retry recompile (below) runs inside the stage-3 window but is really
    // stage-2 work; track it so the timing line attributes it to stage2 (which
    // then reports source=tileiras) instead of inflating stage3.
    let mut recompile_ms = 0.0;
    let stage3_start = std::time::Instant::now();
    let module = match load_module_from_bytes(&cubin, device_id) {
        Ok(module) => module,
        // A disk-served cubin the driver rejects (partial write the checksum
        // missed, driver/toolkit skew, …) must not fail the launch: evict that
        // exact entry and recompile with tileiras, bypassing the cache read so a
        // still-present bad entry can't be re-served. `mem::replace` moves the
        // store/key out and leaves `Tileiras`, which is now the true source of
        // the loaded cubin. Only a second failure is a real error.
        Err(e) => match std::mem::replace(&mut stage2_source, Stage2Source::Tileiras) {
            Stage2Source::DiskCache { store, key } => {
                jit_log!(
                    "{module_name}::{function_name} → cached cubin rejected by the driver ({e}); \
                     evicting and recompiling"
                );
                let recompile_start = std::time::Instant::now();
                let cubin = recompile_after_disk_rejection(
                    store.as_ref(),
                    &key,
                    &bytecode,
                    gpu_name,
                    DEFAULT_OPT_LEVEL,
                )?;
                recompile_ms = recompile_start.elapsed().as_secs_f64() * 1000.0;
                stage2_ms += recompile_ms;
                load_module_from_bytes(&cubin, device_id)?
            }
            Stage2Source::Tileiras => return Err(e.into()),
        },
    };
    let function = Arc::new(module.load_function(function_entry).map_err(|e| {
        Error::KernelLaunch(KernelLaunchError(format!(
            "failed to load '{function_entry}' from compiled cubin: {e}"
        )))
    })?);
    // Exclude the retry recompile: it was moved into stage2_ms above, so the
    // stage-3 figure stays "module load only". `max(0.0)` guards float noise.
    let stage3_ms = (stage3_start.elapsed().as_secs_f64() * 1000.0 - recompile_ms).max(0.0);

    jit_log!(
        "{module_name}::{function_name} → JIT compiled in {:.1?}",
        t0.elapsed()
    );
    if std::env::var_os("CUTILE_JIT_TIMING").is_some() {
        let stage2_source = match stage2_source {
            Stage2Source::Tileiras => "tileiras",
            Stage2Source::DiskCache { .. } => "disk",
        };
        eprintln!(
            "CUTILE_JIT_TIMING module={module_name} function={function_name} key={key_str} stage1_ms={stage1_ms:.3} stage2_ms={stage2_ms:.3} stage2_source={stage2_source} stage3_ms={stage3_ms:.3} generics={}",
            generics.join(","),
        );
    }

    Ok(CompiledKernel {
        module,
        function,
        validator,
    })
}

/// Compiles a tile function to CUDA and caches it for reuse.
///
/// Handles the complete compilation pipeline from Rust to CUDA:
/// 1. Checks the global kernel cache (process-wide, cross-thread)
/// 2. If not cached, compiles the module AST to Tile IR bytecode, then to a cubin
/// 3. Stores the result in the global kernel cache
///
/// **Compilation dedup**: When multiple threads need the same kernel, `OnceCell::get_or_try_init`
/// ensures only one thread performs compilation while others block. Once initialization completes,
/// all threads see the same cached result.
///
/// The caching key is based on the module name, function name, type generics, stride arguments,
/// and compile-time grid dimensions, ensuring correct reuse across different specializations.
///
/// ## Arguments
///
/// * `ctx` - Execution context containing device information
/// * `module_asts` - Closure that produces the AST modules to compile
/// * `module_name` - Name of the module containing the function
/// * `function_name` - Name of the function to compile
/// * `function_entry` - Entry point name in the compiled CUDA code
/// * `function_generics` - Type and const generic arguments (e.g., `["f32", "256"]`)
/// * `stride_args` - Stride information for tensor arguments
/// * `const_grid` - Optional compile-time constant grid dimensions
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_kernel::compile_from_context;
///
/// let ctx = get_execution_context();
/// let function = compile_from_context(
///     &ctx,
///     || vec![my_module_ast()],
///     "my_module",
///     "my_function",
///     "my_function_kernel",
///     vec!["f32".to_string(), "128".to_string()],
///     vec![],
///     None
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn compile_from_context<F: Fn() -> Module>(
    ctx: &ExecutionContext,
    kernel_ast: F,
    module_name: &str,
    function_name: &str,
    function_entry: &str,
    function_generics: Vec<String>,
    stride_args: Vec<(String, Vec<i32>)>,
    spec_args: Vec<(String, SpecializationBits)>,
    scalar_hints: Vec<(String, DivHint)>,
    const_grid: Option<(u32, u32, u32)>,
    compile_options: CompileOptions,
    source_hash: &str,
) -> Result<(Arc<Function>, Arc<Validator>), Error> {
    let device_id: usize = ctx.get_device_id();
    let gpu_name = get_gpu_name(device_id);
    let compiler_version = get_compiler_version();
    let tileiras_fp = tileiras_fingerprint();
    let mut key_builder = TileFunctionKey::builder(module_name, function_name)
        .generics(function_generics)
        .stride_args(stride_args)
        .spec_args(spec_args)
        .scalar_hints(scalar_hints)
        .compile_options(compile_options)
        .source_hash(source_hash)
        .device_id(device_id)
        .gpu_name(gpu_name.clone())
        .compiler_version(compiler_version)
        .tileiras_fingerprint(tileiras_fp);
    if let Some(grid) = const_grid {
        key_builder = key_builder.grid(grid);
    }
    let key = key_builder.build();
    let slot = kernel_cache_slot(&key);

    // Use OnceCell::get_or_try_init for single-flight compilation dedup.
    // Only one thread executes the closure; others block and see the result.
    let compiled = match slot.get_or_try_init(|| -> Result<CompiledKernel, Error> {
        jit_log!("{module_name}::{function_name} → JIT compiling...");
        // Build the module ASTs lazily — only on a real cache miss.
        let modules = CUDATileModules::from_kernel(kernel_ast())?;
        let kernel = compile_and_load_kernel(
            &modules,
            module_name,
            function_name,
            function_entry,
            &key.function_generics,
            &key.stride_args,
            &key.spec_args,
            &key.scalar_hints,
            const_grid,
            &gpu_name,
            &key.compile_options,
            device_id,
            &key.display_hash(),
        )?;
        // Count only a successful compile: a failed attempt leaves the slot empty
        // and retries, so counting at the top would double-count on retry and
        // break the "+1 per cached kernel" contract.
        record_jit_compile();
        Ok(kernel)
    }) {
        Ok(compiled) => compiled,
        Err(e) => {
            // A failed compile leaves an empty slot; evict it so repeated failing
            // specializations don't grow the cache unbounded.
            //
            // On failure, once_cell gives the cell to a blocked waiter to retry.
            // To avoid removing the slot while that waiter is still compiling
            // (which would orphan its success and break single-flight), drop our
            // own `slot` first, then (under the shard write lock) remove only
            // when the cell is still empty and `strong_count == 1`.
            drop(slot);
            get_kernel_cache().remove_if(&key, |_, cell| {
                cell.get().is_none() && Arc::strong_count(cell) == 1
            });
            return Err(e);
        }
    };

    Ok((
        Arc::clone(&compiled.function),
        Arc::clone(&compiled.validator),
    ))
}

/// Validates that all partition grids match the expected launch grid.
pub fn validate_grids(
    grid: (u32, u32, u32),
    partition_grids: &[(u32, u32, u32)],
) -> Result<(), Error> {
    // Make sure we're not trying to map mutable references to incorrect launch grid.
    if let Some(partition_grid) = partition_grids.iter().find(|&&i| i != grid) {
        Err(Error::KernelLaunch(KernelLaunchError(format!(
            "{:?} != {:?}",
            grid, partition_grid
        ))))
    } else {
        Ok(())
    }
}

/// Infers the launch grid for a kernel from partitioned tensor inputs.
///
/// If a grid is explicitly specified (non-zero), it is used directly. Otherwise, the grid
/// is inferred from partitioned tensor inputs. All inferred grids must match, or the
/// function will return an error.
///
/// ## Errors
///
/// Returns an error if no grid is specified and no inferred grids are available, or if inferred
/// grids from different inputs don't match.
pub fn infer_launch_grid(
    grid: (u32, u32, u32),
    inferred_grids: &[(u32, u32, u32)],
) -> Result<(u32, u32, u32), Error> {
    if grid != (0, 0, 0) {
        // A launch grid was specified.
        if !inferred_grids.is_empty() {
            validate_grids(grid, inferred_grids).with_context(|| {
                "Specified launch grid does not match inferred tensor partition grid"
            })?;
        }
        return Ok(grid);
    }
    // Try to infer launch grid.
    if inferred_grids.is_empty() {
        return kernel_launch_error_result("Launch grid required.");
    }
    let grid = inferred_grids[0];
    validate_grids(grid, inferred_grids)
        .with_context(|| "Inferred tensor partition grids do not match")?;
    Ok(grid)
}

/// A compiled CUDA kernel generated from Rust code that can be launched on the GPU.
///
/// `TileKernel` extends [`DeviceOp`] with kernel-specific functionality. Kernels are
/// automatically generated from Rust functions marked with `#[cutile::entry]` and compiled
/// to Tile IR bytecode, then to a CUDA cubin at runtime.
///
/// The trait provides methods for configuring kernel launch parameters such as grid dimensions,
/// type generics, and shared memory. Grid dimensions can be set explicitly or inferred from
/// partitioned tensor inputs.
///
/// ## Examples
///
/// ### Basic kernel launch
///
/// ```rust,ignore
/// #[cutile::module]
/// mod my_module {
///     use cutile::core::*;
///
///     #[cutile::entry]
///     fn hello_world() {
///         let pid = get_tile_block_id();
///         cuda_tile_print!("Hello from block {}\n", pid.0);
///     }
/// }
///
/// // Launch with explicit grid
/// my_module::hello_world()
///     .grid((4, 1, 1))
///     .sync_on(&stream)?;
/// ```
///
/// ### Kernel with arguments and grid inference
///
/// ```rust,ignore
/// // Output-first convention: &mut param is the first argument.
/// // Grid is inferred from partitioned tensors.
/// // The unified launcher accepts both plain values and DeviceOps.
/// let result = add(
///     api::zeros(&[256]).partition([64]),
///     api::ones(&[256]),
///     api::ones(&[256]),
/// )
/// .first()        // extract the &mut output
/// .unpartition()  // recover Tensor from Partition
/// .to_host_vec()
/// .sync()?;
/// ```
///
/// ### Using with async composition
///
/// ```rust,ignore
/// async fn pipeline() -> impl DeviceOp<Output=Tensor<f32>> {
///     let x = api::randn(0.0, 1.0, [128, 128]).await;
///
///     // Chain kernel operations
///     let y = my_kernel_1(x.clone())
///         .grid((8, 8, 1))
///         .await;
///
///     let z = my_kernel_2(y)
///         .grid((4, 4, 1))
///         .await;
///
///     z
/// }
/// ```
pub trait TileKernel<ARGS: Send, DI, STORED: Send = ARGS>: DeviceOp<Output = ARGS>
where
    DI: DeviceOp<Output = STORED>,
{
    /// Compiles the kernel from its module AST, returning the CUDA function
    /// and validator.
    ///
    /// This is the internal compile-and-cache entry point used by the generated
    /// launcher (both the `.sync()`/`.await` launch path and the `.compile()`
    /// warmup terminal). The user-facing `.compile()` terminal is a separate,
    /// no-argument method generated per kernel; this one keeps the descriptive
    /// name `jit_compile` so it does not collide with it.
    ///
    /// `kernel_ast` is invoked once on cache miss to obtain the kernel's own
    /// [`Module`] (typically the macro-generated `__module_ast_self` fn).
    /// Dep modules are discovered by walking the kernel's `use` statements
    /// against the linker registry.
    #[allow(clippy::too_many_arguments)]
    fn jit_compile<F: Fn() -> Module>(
        &mut self,
        ctx: &ExecutionContext,
        kernel_ast: F,
        module_name: &str,
        function_name: &str,
        function_entry: &str,
        function_generics: Vec<String>,
        stride_args: Vec<(String, Vec<i32>)>,
        spec_args: Vec<(String, SpecializationBits)>,
        scalar_hints: Vec<(String, DivHint)>,
        grid: Option<(u32, u32, u32)>,
        compile_options: CompileOptions,
        source_hash: &str,
    ) -> Result<(Arc<Function>, Arc<Validator>), Error> {
        compile_from_context(
            ctx,
            kernel_ast,
            module_name,
            function_name,
            function_entry,
            function_generics,
            stride_args,
            spec_args,
            scalar_hints,
            grid,
            compile_options,
            source_hash,
        )
    }
    /// Sets the type and const generic arguments for this kernel.
    fn generics(self, generics: Vec<String>) -> Self;
    /// Sets a compile-time constant grid, enabling grid-dependent optimizations.
    fn const_grid(self, grid: (u32, u32, u32)) -> Self;
    /// Sets the runtime launch grid dimensions.
    fn grid(self, grid: (u32, u32, u32)) -> Self;
    /// Sets the runtime compile options (occupancy, num_cta_in_cga).
    fn compile_options(self, options: CompileOptions) -> Self;
    /// Infers the launch grid from partitioned tensor inputs, or uses the explicit grid.
    fn infer_launch_grid(
        &self,
        inferred_grids: &[(u32, u32, u32)],
    ) -> Result<(u32, u32, u32), Error> {
        let grid = self.get_launch_grid();
        infer_launch_grid(grid, inferred_grids)
    }
    /// Returns the currently configured launch grid dimensions.
    fn get_launch_grid(&self) -> (u32, u32, u32);
    /// Returns the dynamic shared memory size in bytes. Defaults to 0.
    fn get_launch_smem(&self) -> u32 {
        0
    }
    /// Returns the thread block dimensions. Defaults to `(1, 1, 1)`.
    fn get_launch_block(&self) -> (u32, u32, u32) {
        (1, 1, 1)
    }
    // fn validate(validator: &Validator) -> Result<(), Error> {

    // }
    // fn validate_arc<T: DType>(
    //     &self,
    //     func_name: String,
    //     var_name: String,
    //     arc: &Arc<Tensor<T>>,
    //     shape: &[i32],
    // ) -> Result<(), KernelLauncherError> {
    //     let input_shape = &arc.shape;
    //     if input_shape != shape {
    //         return Err(KernelLauncherError::InvalidTensorShape(format!(
    //             "Unexpected shape {:?} for argument {} for function {}.",
    //             input_shape, var_name, func_name
    //         )));
    //     }
    //     Ok(())

    //     // if input_shape.len() != shape.len() {
    //     //     return Err(KernelLauncherError::InvalidTensorShape(format!("Unexpected rank {} for argument {} for function {}.",
    //     //         input_shape.len(),
    //     //         var_name,
    //     //         func_name
    //     //     )));
    //     // }
    //     // for i in 0..input_shape.len() {
    //     //     let input_dim = input_shape[i];
    //     //     let param_dim = shape[i];
    //     //     if param_dim == -1 {
    //     //         continue;
    //     //     }
    //     //     if input_dim != param_dim {
    //     //         return Err(KernelLauncherError::InvalidTensorShape(format!("Unexpected rank {} for argument {} for function {}.",
    //     //             input_shape.len(),
    //     //             var_name,
    //     //             func_name
    //     //         )));
    //     //     }
    //     // }
    // }
}

/// Implements kernel argument passing for `Tensor` when wrapped in `Arc`.
///
/// Pushes the device pointer, shape, and stride information to the kernel launcher
/// in the order expected by compiled tile functions.
impl<T: DType> ArcKernelArgument for Tensor<T> {
    fn push_arg_arc(self: &Arc<Self>, launcher: &mut AsyncKernelLaunch) {
        // TODO (hme): document safety
        unsafe {
            launcher.push_device_ptr(self.cu_deviceptr());
        }
        for dim in self.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.strides.iter() {
            launcher.push_arg(*stride);
        }
    }
}

/// Implements kernel argument passing for partitioned tensors.
///
/// Pushes the device pointer, tensor shape and strides, followed by partition shape
/// and strides. This allows kernels to access both the full tensor and the partition
/// information for block-level indexing.
impl<T: DType> KernelArgument for &Partition<Tensor<T>> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        // TODO (hme): document safety
        unsafe {
            launcher.push_device_ptr(self.object.cu_deviceptr());
        }
        for dim in self.object.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(*stride);
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(*dim as i32);
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(*stride as i32);
        }
    }
}

/// Same as above but for borrowed mutable tensor partitions.
impl<'a, T: DType> KernelArgument for &Partition<&'a mut Tensor<T>> {
    fn push_arg(self, launcher: &mut AsyncKernelLaunch) {
        unsafe {
            launcher.push_device_ptr(self.object.cu_deviceptr());
        }
        for dim in self.object.shape.iter() {
            launcher.push_arg(*dim);
        }
        for stride in self.object.strides.iter() {
            launcher.push_arg(*stride);
        }
        for dim in self.partition_shape.iter() {
            launcher.push_arg(*dim as i32);
        }
        for stride in self.partition_strides.iter() {
            launcher.push_arg(*stride as i32);
        }
    }
}

// Partition

/// Extension trait that enables partitioning device operations into tiles.
///
/// This trait allows async operations that produce tensors to be partitioned before
/// execution, enabling automatic grid inference for tile kernels. The partition divides
/// the tensor into blocks that map to CUDA thread blocks.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_kernel::PartitionOp;
///
/// // Partition a tensor operation before it executes
/// let x = api::ones(&[1024]).partition([128]);  // Creates 8 partitions
///
/// // Use partitioned tensors with kernels for automatic grid inference
/// let y = api::randn(0.0, 1.0, [256, 256]).partition([64, 64]);  // 4x4 grid
/// let result = my_kernel(y).await;  // Grid (4, 4, 1) inferred automatically
/// ```
pub trait PartitionOp<I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOp<Output = I>,
{
    /// Partitions the output of this device operation into tiles of the given shape.
    ///
    /// The partition shape determines how the tensor is divided across CUDA thread blocks.
    fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> DeviceOperationPartition<RANK, I, DI>;
}

impl<I, DI> PartitionOp<I, DI> for DI
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOp<Output = I>,
{
    fn partition<const RANK: usize>(
        self,
        partition_shape: [usize; RANK],
    ) -> DeviceOperationPartition<RANK, I, DI>
    where
        Self: Sized,
    {
        DeviceOperationPartition::<RANK, I, DI> {
            partition_shape,
            op: self,
        }
    }
}

/// A device operation that partitions its output into tiles.
///
/// This wrapper executes the underlying device operation and then partitions its result
/// according to the specified partition shape. The resulting partitioned tensor can be
/// used with tile kernels to automatically infer launch grid dimensions.
///
/// Created by calling `.partition()` on any device operation that produces a partitionable output.
///
/// ## Examples
///
/// ```rust,ignore
/// // Create a partitioned tensor operation
/// let z = api::zeros(&[1024]).partition([64]);
///
/// // Pass directly to kernel — grid inferred from partition
/// let result = my_kernel(z, x, y).first().unpartition().sync()?;
/// ```
pub struct DeviceOperationPartition<const RANK: usize, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOp<Output = I>,
{
    partition_shape: [usize; RANK],
    op: DI,
}

unsafe impl<const RANK: usize, I, DI> Send for DeviceOperationPartition<RANK, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOp<Output = I>,
{
}

impl<const RANK: usize, I, DI> DeviceOp for DeviceOperationPartition<RANK, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOp<Output = I>,
{
    type Output = Partition<I>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let val = self.op.execute(context)?;
        Ok(val.partition(self.partition_shape))
    }
}

impl<const RANK: usize, I, DI> IntoFuture for DeviceOperationPartition<RANK, I, DI>
where
    I: Send + IntoPartition + IntoPartitionArc,
    DI: DeviceOp<Output = I>,
{
    type Output = Result<Partition<I>, DeviceError>;
    type IntoFuture = DeviceFuture<Partition<I>, DeviceOperationPartition<RANK, I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

// Unwrap Partition

/// A device operation that unwraps a partitioned tensor back to a regular tensor.
///
/// This operation removes the partition structure from a tensor, converting a
/// `Partition<Tensor<T>>` back to `Tensor<T>`. This is useful after kernel operations
/// that work on partitioned inputs but need to return regular tensors for further
/// processing.
///
/// Created by calling `unwrap_partition()` on a device operation that produces a partition.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_kernel::unwrap_partition;
///
/// // After a kernel operation on partitioned tensors
/// let x = api::ones(&[256]).partition([64]);
/// let y = my_kernel(x).await;  // Returns Partition<Tensor<f32>>
///
/// // Unwrap back to a regular tensor
/// let z = unwrap_partition(y).await;  // Now Tensor<f32>
/// ```
pub struct UnwrapPartition<I: Send, DI>
where
    DI: DeviceOp<Output = Partition<I>>,
{
    pub(crate) op: DI,
}

unsafe impl<I: Send, DI> Send for UnwrapPartition<I, DI> where DI: DeviceOp<Output = Partition<I>> {}

impl<I: Send, DI> DeviceOp for UnwrapPartition<I, DI>
where
    DI: DeviceOp<Output = Partition<I>>,
{
    type Output = I;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let val = self.op.execute(context)?;
        Ok(val.unpartition())
    }
}

impl<I: Send, DI> IntoFuture for UnwrapPartition<I, DI>
where
    DI: DeviceOp<Output = Partition<I>>,
{
    type Output = Result<I, DeviceError>;
    type IntoFuture = DeviceFuture<I, UnwrapPartition<I, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Unwraps a partitioned device operation back to a regular tensor operation.
///
/// Converts a device operation that produces a `Partition<T>` into one
/// that produces `T` directly. Useful for converting partitioned kernel outputs
/// back to regular tensors for further processing.
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_kernel::unwrap_partition;
///
/// async fn process_data() -> Tensor<f32> {
///     let x = api::randn(0.0, 1.0, [1024]).partition([128]);
///     let processed = my_tiled_kernel(x);  // Returns Partition<Tensor<f32>>
///
///     // Unwrap to get a regular tensor
///     unwrap_partition(processed).await
/// }
/// ```
pub fn unwrap_partition<I: Send, DI>(op: DI) -> UnwrapPartition<I, DI>
where
    DI: DeviceOp<Output = Partition<I>>,
{
    UnwrapPartition { op }
}

// ToHostVec

/// A device operation that copies a tensor from device memory to a host `Vec<T>`.
pub struct TensorToHostVec<T: DType, DI>
where
    DI: DeviceOp<Output = Tensor<T>>,
{
    pub(crate) op: DI,
}

unsafe impl<T: DType, DI> Send for TensorToHostVec<T, DI> where DI: DeviceOp<Output = Tensor<T>> {}

impl<T: DType, DI> DeviceOp for TensorToHostVec<T, DI>
where
    DI: DeviceOp<Output = Tensor<T>>,
{
    type Output = Vec<T>;

    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let tensor = self.op.execute(context)?;
        let cu_deviceptr = tensor.cu_deviceptr();
        let size = tensor.size();
        let layout = Layout::array::<T>(size).expect("overflow cannot happen");
        let async_ptr = unsafe { alloc(layout).cast::<T>() };
        memcpy_dtoh_async(async_ptr, cu_deviceptr, size, context.get_cuda_stream());
        Ok(unsafe { Vec::from_raw_parts(async_ptr, size, size) })
    }
}

impl<T: DType, DI> IntoFuture for TensorToHostVec<T, DI>
where
    DI: DeviceOp<Output = Tensor<T>>,
{
    type Output = Result<Vec<T>, DeviceError>;
    type IntoFuture = DeviceFuture<Vec<T>, TensorToHostVec<T, DI>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}

/// Extension trait for converting a tensor device operation into a host `Vec<T>` operation.
pub trait ToHostVecOp<T: DType> {
    /// Wraps this operation to copy the resulting tensor to a host `Vec<T>`.
    fn to_host_vec(self) -> impl DeviceOp<Output = Vec<T>>
    where
        Self: DeviceOp<Output = Tensor<T>>,
    {
        TensorToHostVec { op: self }
    }
}

impl<T: DType, DI> ToHostVecOp<T> for DI where DI: DeviceOp<Output = Tensor<T>> {}
