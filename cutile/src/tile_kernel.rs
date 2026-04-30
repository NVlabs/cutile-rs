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
use cutile_compiler::cuda_tile_runtime_utils::{compile_tile_ir_module, get_gpu_name, get_cuda_toolkit_version, get_compiler_version};
use cutile_compiler::specialization::{DivHint, SpecializationBits};
use once_cell::sync::OnceCell;
use sha2::{Digest, Sha256};
use std::alloc::{alloc, Layout};
use std::fs;
use std::future::IntoFuture;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

// JIT diagnostic logging (set CUTILE_JIT_LOG=1 to enable) 

fn jit_log_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CUTILE_JIT_LOG").is_ok_and(|v| v == "1"))
}

macro_rules! jit_log {
    ($($arg:tt)*) => {
        if jit_log_enabled() {
            eprintln!("[cutile::jit] {}", format!($($arg)*));
        }
    };
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
/// GPU architecture, compiler version, and CUDA toolkit version.
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
    gpu_name: String,
    compiler_version: String,
    cuda_toolkit_version: String,
}

impl TileFunctionKey {
    pub fn new(
        module_name: String,
        function_name: String,
        function_generics: Vec<String>,
        stride_args: Vec<(String, Vec<i32>)>,
        spec_args: Vec<(String, SpecializationBits)>,
        scalar_hints: Vec<(String, DivHint)>,
        grid: Option<(u32, u32, u32)>,
        compile_options: CompileOptions,
        source_hash: String,
        gpu_name: String,
        compiler_version: String,
        cuda_toolkit_version: String,
    ) -> Self {
        Self {
            module_name,
            function_name,
            function_generics,
            stride_args,
            spec_args,
            scalar_hints,
            grid,
            compile_options,
            source_hash,
            gpu_name,
            compiler_version,
            cuda_toolkit_version,
        }
    }
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
///     .gpu_name(get_gpu_name(device_id))
///     .compiler_version(get_compiler_version())
///     .cuda_toolkit_version(get_cuda_toolkit_version())
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
    gpu_name: String,
    compiler_version: String,
    cuda_toolkit_version: String,
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
    pub fn gpu_name(mut self, name: impl Into<String>) -> Self {
        self.gpu_name = name.into();
        self
    }
    pub fn compiler_version(mut self, version: impl Into<String>) -> Self {
        self.compiler_version = version.into();
        self
    }
    pub fn cuda_toolkit_version(mut self, version: impl Into<String>) -> Self {
        self.cuda_toolkit_version = version.into();
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
            gpu_name: self.gpu_name,
            compiler_version: self.compiler_version,
            cuda_toolkit_version: self.cuda_toolkit_version,
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
            gpu_name: String::new(),
            compiler_version: String::new(),
            cuda_toolkit_version: String::new(),
        }
    }
}

impl FunctionKey for TileFunctionKey {
    fn get_disk_hash_string(&self) -> String {
        let canonical = format!(
            "{}:{}:{}:{}:{}:{}:{:?}:{:?}:{}:{}:{}:{}",
            self.module_name,
            self.function_name,
            self.function_generics.join(","),
            self.stride_args
                .iter()
                .map(|(k, v)| format!("{}={:?}", k, v))
                .collect::<Vec<_>>()
                .join(";"),
            self.spec_args
                .iter()
                .map(|(k, v)| format!("{}={:?}", k, v))
                .collect::<Vec<_>>()
                .join(";"),
            self.scalar_hints
                .iter()
                .map(|(k, v)| format!("{}={:?}", k, v))
                .collect::<Vec<_>>()
                .join(";"),
            self.grid,
            self.compile_options,
            self.source_hash,
            self.gpu_name,
            self.compiler_version,
            self.cuda_toolkit_version,
        );
        let hash = Sha256::digest(canonical.as_bytes());
        format!("{:x}", hash)
    }
}

/// Reads IR (MLIR or PTX) from a file.
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

/// Writes IR (MLIR or PTX) to a file for debugging.
///
/// This helper function writes intermediate representation to disk when kernel functions
/// are marked with `dump_mlir_dir` or `dump_ptx_dir` entry attributes. The filename
/// includes the module name, function name, and cache hash for uniqueness.
///
/// ## Parameters
///
/// - `module_name`: Name of the module containing the kernel
/// - `function_name`: Name of the kernel function
/// - `cache_hash_str`: Unique hash identifying this compilation
/// - `extension`: File extension (e.g., "mlir", "ptx")
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

/// Attempt to load a cubin from the global JitStore.
fn try_load_from_jit_store(disk_key: &str) -> Option<Vec<u8>> {
    let store = cuda_async::jit_store::get_jit_store()?;
    store.get(disk_key).ok().flatten()
}

// ── Shared compilation pipeline (used by compile_from_context + compile_warmup) ──

/// Get (or insert) the cache slot for a key.
///
/// The DashMap shard lock is released before the slot is returned, so the
/// expensive `OnceCell` initialization happens without holding it.
fn get_or_create_slot(key_str: &str) -> Arc<OnceCell<CompiledKernel>> {
    let cache = get_kernel_cache();
    let entry = cache
        .entry(key_str.to_string())
        .or_insert_with(|| Arc::new(OnceCell::new()));
    Arc::clone(entry.value())
    // entry dropped here — releases DashMap shard lock.
}

/// Build a `CUDATileFunctionCompiler` from a `TileFunctionKey`.
///
/// Borrows from `modules`; the returned compiler must not outlive it.
fn build_function_compiler<'m>(
    modules: &'m CUDATileModules,
    key: &TileFunctionKey,
) -> Result<CUDATileFunctionCompiler<'m>, Error> {
    let stride_args: Vec<(&str, &[i32])> = key
        .stride_args
        .iter()
        .map(|x| (x.0.as_str(), x.1.as_slice()))
        .collect();
    let spec_args: Vec<(&str, &SpecializationBits)> =
        key.spec_args.iter().map(|x| (x.0.as_str(), &x.1)).collect();
    #[allow(unused_variables)]
    let scalar_hints: Vec<(&str, &DivHint)> = key
        .scalar_hints
        .iter()
        .map(|x| (x.0.as_str(), &x.1))
        .collect();
    let compiler = CUDATileFunctionCompiler::new(
        modules,
        &key.module_name,
        &key.function_name,
        &key.function_generics,
        &stride_args,
        &spec_args,
        &scalar_hints,
        key.grid,
        key.gpu_name.clone(),
        &key.compile_options,
    )?;
    Ok(compiler)
}

/// Compile a kernel from scratch: try the disk cache first, then full JIT.
///
/// On disk hit, loads the persisted cubin and rebuilds the validator. On a
/// corrupted/incompatible cubin, deletes the bad entry and falls through to
/// JIT recompile. After a successful JIT, persists the cubin to disk if a
/// `JitStore` is configured.
fn compile_kernel_uncached(
    key: &TileFunctionKey,
    modules: &CUDATileModules,
    function_entry: &str,
    device_id: usize,
    disk_key: &str,
    key_str: &str,
) -> Result<CompiledKernel, Error> {
    let module_name = key.module_name.as_str();
    let function_name = key.function_name.as_str();
    let gpu_name = key.gpu_name.as_str();

    // Try disk cache first.
    if let Some(cubin_bytes) = try_load_from_jit_store(disk_key) {
        jit_log!(
            "{module_name}::{function_name} → disk cache hit ({} bytes)",
            cubin_bytes.len()
        );
        let compiler = build_function_compiler(modules, key)?;
        let validator = Arc::new(compiler.get_validator());
        match load_module_from_bytes(&cubin_bytes, device_id) {
            Ok(module) => {
                let function = Arc::new(module.load_function(function_entry).map_err(|e| {
                    Error::KernelLaunch(KernelLaunchError(format!(
                        "failed to load '{function_entry}' from cached cubin: {e}"
                    )))
                })?);
                return Ok(CompiledKernel {
                    module,
                    function,
                    validator,
                });
            }
            Err(e) => {
                // Corrupted/incompatible cubin (e.g. disk error, partial write,
                // architecture mismatch). Delete the bad entry and fall through
                // to full JIT recompilation so the caller is not permanently
                // blocked.
                jit_log!(
                    "{module_name}::{function_name} → corrupted disk cubin, \
                     deleting and recompiling (error: {e})"
                );
                if let Some(store) = cuda_async::jit_store::get_jit_store() {
                    let _ = store.delete(disk_key);
                }
            }
        }
    }

    // Full JIT compilation.
    jit_log!("{module_name}::{function_name} → JIT compiling...");
    let t0 = std::time::Instant::now();
    let _debug_mlir_path = modules.get_entry_arg_string_by_function_name(
        module_name,
        function_name,
        "use_debug_mlir",
    )?;
    // TODO (hme): Re-enable some debug support for internal.
    // let mlir = if let Some(debug_mlir_path) = &debug_mlir_path {
    //     println!("USING DEBUG MLIR: {debug_mlir_path}");
    //     let mlir = read_ir(debug_mlir_path.to_string()).expect("Failed to read debug MLIR.");
    //     mlir
    // } else {
    //     let module_op: ModuleOperation = compiler.compile(
    //         module_name,
    //         function_name,
    //         &key.function_generics,
    //         &key.stride_args
    //             .iter()
    //             .map(|x| (x.0.as_str(), x.1.as_slice()))
    //             .collect::<Vec<_>>(),
    //         const_grid,
    //         gpu_name.clone(),
    //     );
    //     let mlir = module_op.as_operation().to_string();
    //     mlir
    // };
    let compiler = build_function_compiler(modules, key)?;
    let validator = Arc::new(compiler.get_validator());
    let tile_module = compiler.compile()?;

    // Optional debug output (controlled by `#[entry(...)]` attributes).
    if modules.get_entry_arg_bool_by_function_name(module_name, function_name, "print_ir")? {
        println!(
            "COMPILED IR: {module_name}::{function_name}\n{}",
            tile_module.to_mlir_text()
        );
    }
    if let Some(path) =
        modules.get_entry_arg_string_by_function_name(module_name, function_name, "dump_mlir_dir")?
    {
        write_ir(
            module_name,
            function_name,
            key_str,
            "mlir",
            path.as_str(),
            tile_module.to_mlir_text().as_str(),
        );
    }

    let cubin_filename = compile_tile_ir_module(&tile_module, gpu_name);
    // if let Some(path) = compiler.get_entry_arg_string_by_function_name(
    //     module_name,
    //     function_name,
    //     "dump_ptx_dir",
    // ) {
    //     write_ir(
    //         module_name,
    //         function_name,
    //         cache_hash_str.as_str(),
    //         "ptx",
    //         path.as_str(),
    //         ptx.as_str(),
    //     );
    // }
    // if compiler.get_entry_arg_bool_by_function_name(module_name, function_name,"print_ir") {
    //     println!("COMPILED PTX: {module_name}::{function_name}");
    //     println!("{ptx}");
    //     println!();
    // }
    let jit_elapsed = t0.elapsed();

    // Persist to disk cache if a JitStore is configured.
    if let Some(store) = cuda_async::jit_store::get_jit_store() {
        if let Ok(cubin_bytes) = std::fs::read(&cubin_filename) {
            if store.put(disk_key, &cubin_bytes).is_ok() {
                jit_log!(
                    "{module_name}::{function_name} → saved to disk cache ({} bytes)",
                    cubin_bytes.len()
                );
            }
        }
    }

    let module = load_module_from_file(&cubin_filename, device_id)?;
    let function = Arc::new(module.load_function(function_entry).map_err(|e| {
        Error::KernelLaunch(KernelLaunchError(format!(
            "failed to load '{function_entry}' from compiled cubin: {e}"
        )))
    })?);
    jit_log!(
        "{module_name}::{function_name} → JIT compiled in {:.1?}",
        jit_elapsed
    );
    Ok(CompiledKernel {
        module,
        function,
        validator,
    })
}

/// Cache-aware kernel compilation with single-flight dedup.
///
/// Returns the cache slot, guaranteed initialized on success. Concurrent
/// callers requesting the same key block on a single `OnceCell` so only
/// one performs JIT while the rest see the same cached result.
///
/// `cached_modules` lets callers (e.g. [`compile_warmup`]) pre-build a
/// [`CUDATileModules`] once and reuse it across many specs. Pass `None`
/// for lazy per-call construction (only built on cache miss).
fn get_or_compile_kernel(
    key: &TileFunctionKey,
    cached_modules: Option<&CUDATileModules>,
    module_builder: &dyn Fn() -> Result<CUDATileModules, Error>,
    function_entry: &str,
    device_id: usize,
) -> Result<Arc<OnceCell<CompiledKernel>>, Error> {
    let key_str = key.get_hash_string();
    let disk_key = key.get_disk_hash_string();
    let slot = get_or_create_slot(&key_str);

    slot.get_or_try_init(|| -> Result<CompiledKernel, Error> {
        let owned;
        let modules: &CUDATileModules = match cached_modules {
            Some(m) => m,
            None => {
                owned = module_builder()?;
                &owned
            }
        };
        compile_kernel_uncached(key, modules, function_entry, device_id, &disk_key, &key_str)
    })?;

    Ok(slot)
}

/// Compiles a tile function to CUDA and caches it for reuse.
///
/// The runtime kernel-launch entry point. Builds a `TileFunctionKey` from
/// the call site and dispatches to [`get_or_compile_kernel`] which handles
/// memory cache lookup, single-flight dedup, disk cache load, and JIT.
///
/// Module ASTs are built lazily — only on cache miss — via the closure.
///
/// ## Arguments
///
/// * `ctx` - Execution context containing device information
/// * `kernel_ast` - Closure that builds the kernel module AST
/// * `module_name` - Name of the module containing the function
/// * `function_name` - Name of the function to compile
/// * `function_entry` - Entry point name in the compiled CUDA code
/// * `function_generics` - Type and const generic arguments (e.g., `["f32", "256"]`)
/// * `stride_args` - Stride information for tensor arguments
/// * `spec_args` - Per-tensor specialization bits
/// * `scalar_hints` - Per-scalar divisibility hints
/// * `const_grid` - Optional compile-time constant grid dimensions
/// * `compile_options` - Per-call compile options (occupancy, cga, divisibility)
/// * `source_hash` - SHA-256 of the module source (from the macro-generated `_SOURCE_HASH`)
///
/// ## Examples
///
/// ```rust,ignore
/// use cutile::tile_kernel::compile_from_context;
///
/// let ctx = get_execution_context();
/// let function = compile_from_context(
///     &ctx,
///     || my_module::__module_ast_self(),
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
    cuda_async::jit_store::ensure_default_jit_store();

    let device_id: usize = ctx.get_device_id();
    let key = TileFunctionKey::new(
        module_name.to_string(),
        function_name.to_string(),
        function_generics,
        stride_args,
        spec_args,
        scalar_hints,
        const_grid,
        compile_options,
        source_hash.to_string(),
        get_gpu_name(device_id),
        get_compiler_version(),
        get_cuda_toolkit_version(),
    );

    let module_builder = || {
        Ok(CUDATileModules::from_kernel(kernel_ast())?)
    };

    let slot = get_or_compile_kernel(&key, None, &module_builder, function_entry, device_id)?;
    let compiled = slot.get().expect("OnceCell initialized by get_or_compile_kernel");
    Ok((
        Arc::clone(&compiled.function),
        Arc::clone(&compiled.validator),
    ))
}

// ── Warmup types and functions ───────────────────────────────────────────────

/// Metadata for a single kernel entry point, generated by the `#[cutile::module]` macro.
#[derive(Debug, Clone)]
pub struct EntryMeta {
    pub module_name: &'static str,
    pub function_name: &'static str,
    pub function_entry: &'static str,
}

/// User-provided specialization for warmup compilation.
///
/// Each `WarmupSpec` describes one kernel specialization to pre-compile.
#[derive(Debug, Clone)]
pub struct WarmupSpec {
    pub function_name: String,
    pub function_generics: Vec<String>,
    pub stride_args: Vec<(String, Vec<i32>)>,
    pub spec_args: Vec<(String, SpecializationBits)>,
    pub scalar_hints: Vec<(String, DivHint)>,
    pub const_grid: Option<(u32, u32, u32)>,
}

impl WarmupSpec {
    /// Create a warmup spec with just generics (no strides, no const grid).
    pub fn new(function_name: &str, generics: Vec<String>) -> Self {
        Self {
            function_name: function_name.to_string(),
            function_generics: generics,
            stride_args: vec![],
            spec_args: vec![],
            scalar_hints: vec![],
            const_grid: None,
        }
    }

    /// Set stride arguments for this spec.
    pub fn with_strides(mut self, stride_args: Vec<(String, Vec<i32>)>) -> Self {
        self.stride_args = stride_args;
        self
    }

    /// Set specialization arguments for this spec.
    pub fn with_spec_args(mut self, spec_args: Vec<(String, SpecializationBits)>) -> Self {
        self.spec_args = spec_args;
        self
    }

    /// Set scalar divisibility hints for this spec.
    ///
    /// Cache key includes scalar hints, so a kernel warmed up with hints
    /// will only be reused at launch time when the launcher passes the
    /// same hints. Omit this to warm up the no-hints variant.
    pub fn with_scalar_hints(mut self, scalar_hints: Vec<(String, DivHint)>) -> Self {
        self.scalar_hints = scalar_hints;
        self
    }

    /// Set a const grid for this spec.
    pub fn with_const_grid(mut self, grid: (u32, u32, u32)) -> Self {
        self.const_grid = Some(grid);
        self
    }
}

/// Pre-compile a set of kernel specializations without launching.
///
/// Builds the module ASTs once, then compiles each requested specialization.
/// Results are placed in the global kernel cache and optionally persisted
/// to the JitStore.
///
/// # Example
///
/// ```rust,ignore
/// compile_warmup(
///     || linalg::__module_ast_self(),
///     &linalg::_entries(),
///     "linalg",
///     linalg::_SOURCE_HASH,
///     &[
///         WarmupSpec::new("vector_add", vec!["f32".into(), "128".into()]),
///         WarmupSpec::new("vector_add", vec!["f16".into(), "256".into()]),
///         WarmupSpec::new("relu",       vec!["f32".into(), "128".into()]),
///     ],
/// )?;
/// ```
pub fn compile_warmup<F: Fn() -> Module>(
    kernel_ast: F,
    entries: &[EntryMeta],
    module_name: &str,
    source_hash: &str,
    specs: &[WarmupSpec],
) -> Result<(), Error> {
    cuda_async::jit_store::ensure_default_jit_store();

    let device_id = get_default_device();
    let gpu_name = get_gpu_name(device_id);
    let compiler_version = get_compiler_version();
    let cuda_toolkit_version = get_cuda_toolkit_version();

    // Build module ASTs once, shared across all specs in this warmup call.
    let module_builder = || {
        Ok(CUDATileModules::from_kernel(kernel_ast())?)
    };
    let modules = module_builder()?;

    for spec in specs {
        let entry = entries
            .iter()
            .find(|e| e.function_name == spec.function_name)
            .ok_or_else(|| {
                Error::KernelLaunch(KernelLaunchError(format!(
                    "compile_warmup: unknown function '{}' in module '{}'",
                    spec.function_name, module_name
                )))
            })?;

        let key = TileFunctionKey::new(
            module_name.to_string(),
            spec.function_name.clone(),
            spec.function_generics.clone(),
            spec.stride_args.clone(),
            spec.spec_args.clone(),
            spec.scalar_hints.clone(),
            spec.const_grid,
            CompileOptions::default(),
            source_hash.to_string(),
            gpu_name.clone(),
            compiler_version.clone(),
            cuda_toolkit_version.clone(),
        );

        get_or_compile_kernel(
            &key,
            Some(&modules),
            &module_builder,
            entry.function_entry,
            device_id,
        )?;
    }

    Ok(())
}

/// Execute a warmup routine with realistic kernel launches.
///
/// The provided closure should launch kernels with production-representative
/// shapes and data. This warms up both the JIT compilation cache and the
/// CUDA runtime (driver initialization, shared memory allocation, occupancy
/// calculation, etc.).
///
/// # Example
///
/// ```rust,ignore
/// execute_warmup(|| {
///     let x = api::zeros::<f32>([4096, 4096]).sync()?;
///     let y = api::zeros::<f32>([4096, 4096]).sync()?;
///     let z = api::zeros::<f32>([4096, 4096]).sync()?;
///     linalg::matmul(z, x, y)
///         .generics(vec!["f32".into(), "128".into()])
///         .grid((32, 32, 1))
///         .sync()?;
///     Ok(())
/// })?;
/// ```
pub fn execute_warmup<F>(f: F) -> Result<(), Error>
where
    F: FnOnce() -> Result<(), Error>,
{
    // Ensure device context is initialized.
    let device_id = get_default_device();
    let _ = with_global_device_context(device_id, |_| {})?;

    // Run user-provided warmup routine.
    // Kernels inside will auto-JIT via existing compile_from_context path.
    f()
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
/// function will panic.
///
/// ## Panics
///
/// Panics if no grid is specified and no inferred grids are available, or if inferred
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
/// to MLIR, then to CUDA PTX at runtime.
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
    /// `kernel_ast` is invoked once on cache miss to obtain the kernel's own
    /// [`Module`] (typically the macro-generated `__module_ast_self` fn).
    /// Dep modules are discovered by walking the kernel's `use` statements
    /// against the linker registry.
    #[allow(clippy::too_many_arguments)]
    fn compile<F: Fn() -> Module>(
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
