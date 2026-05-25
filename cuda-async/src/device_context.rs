/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU device state, global kernel cache, and scheduling policy management.
//!
//! ## Architecture
//!
//! - **Global (process-wide)**: [`Device`] per device and compiled kernel cache are shared
//!   across all threads via [`OnceLock`] and [`DashMap`]. This allows compilation results from
//!   one thread (e.g. warmup) to be visible to all worker threads.
//!
//! - **Per-thread**: Scheduling policy and deallocator stream remain thread-local, since
//!   different threads may want different stream assignments.
//!
//! - **Compilation dedup**: When multiple threads need the same kernel, only one compiles it
//!   while the rest wait, via `DashMap<Key, Arc<OnceLock<CompiledKernel>>>`.

use crate::error::{device_assert, device_error, DeviceError};
use crate::scheduling_policies::{SchedulingPolicy, StreamPoolRoundRobin};
use cuda_core::{Device, Function, MemPool, Module, Stream};
use dashmap::DashMap;
use once_cell::sync::OnceCell;
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

/// The GPU device used when no explicit device is specified. Device 0 is the first GPU.
pub const DEFAULT_DEVICE_ID: usize = 0;

/// The number of GPU devices initialized by default.
pub const DEFAULT_NUM_DEVICES: usize = 1;

/// The number of CUDA streams in the default round-robin pool.
///
/// With a pool of 4 streams, consecutive operations cycle through streams 0 → 1 → 2 → 3 → 0 → …,
/// allowing up to 4 independent operations to overlap on the GPU. Increasing this value adds more
/// potential concurrency at the cost of additional stream resources; decreasing it (down to 1)
/// makes behavior equivalent to [`SingleStream`](crate::scheduling_policies::SingleStream).
pub const DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE: usize = 4;

pub trait FunctionKey: Hash {
    /// Fast hash for in-memory cache lookup (uses `DefaultHasher`).
    fn get_hash_string(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash_value: u64 = hasher.finish();
        format!("{:x}", hash_value)
    }
}

#[derive(Debug, Clone)]
pub enum ValidParamType {
    Scalar(ScalarParamType),
    Pointer(PointerParamType),
    Tensor(TensorParamType),
}

#[derive(Debug, Clone)]
pub struct ScalarParamType {
    pub element_type: String,
}

#[derive(Debug, Clone)]
pub struct PointerParamType {
    pub mutable: bool,
    pub element_type: String,
}

// TODO (hme): This is note entirely tile-agnostic with this param type.
#[derive(Debug, Clone)]
pub struct TensorParamType {
    pub element_type: String,
    pub shape: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct Validator {
    pub params: Vec<ValidParamType>,
}

// ── Global Device (process-wide, per-device singleton) ─────────────────────

/// Global per-device handles. Shared across all threads so that
/// `Module`/`Function` loaded against a device can be used from any thread.
static DEVICES: OnceLock<Mutex<HashMap<usize, Arc<Device>>>> = OnceLock::new();

fn devices() -> &'static Mutex<HashMap<usize, Arc<Device>>> {
    DEVICES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get or create the global [`Device`] for a device ordinal.
///
/// The first call for a given `device_id` creates the device handle; subsequent
/// calls return the same `Arc<Device>`.
fn get_or_init_device(device_id: usize) -> Result<Arc<Device>, DeviceError> {
    let mut devices = devices()
        .lock()
        .map_err(|_| device_error(device_id, "device map lock poisoned"))?;
    if let Some(device) = devices.get(&device_id) {
        return Ok(Arc::clone(device));
    }
    let device = Device::new(device_id)?;
    devices.insert(device_id, Arc::clone(&device));
    Ok(device)
}

// ── Global kernel cache (process-wide, cross-thread) ────────────────────────

/// A compiled kernel: module, function handle, and parameter validator.
#[derive(Debug)]
pub struct CompiledKernel {
    pub module: Arc<Module>,
    pub function: Arc<Function>,
    pub validator: Arc<Validator>,
}

/// Global kernel cache. `DashMap` for cross-thread sharing; inner `OnceLock` for
/// single-flight compilation dedup (if multiple threads need the same kernel,
/// only one compiles while the rest wait). Uses `once_cell::sync::OnceCell`
/// for stable fallible initialization (`get_or_try_init`).
static KERNEL_CACHE: OnceLock<DashMap<String, Arc<OnceCell<CompiledKernel>>>> = OnceLock::new();

pub fn get_kernel_cache() -> &'static DashMap<String, Arc<OnceCell<CompiledKernel>>> {
    KERNEL_CACHE.get_or_init(DashMap::new)
}

/// Remove all compiled kernels from the in-memory cache.
pub fn clear_kernel_cache() {
    get_kernel_cache().clear();
}

/// Evict a single compiled kernel from the in-memory cache by its hash string.
///
/// Returns `true` if an entry was removed, `false` if the key was not present.
pub fn evict_kernel(key_str: &str) -> bool {
    get_kernel_cache().remove(key_str).is_some()
}

/// Get (or create) the single-flight compilation slot for `key_str`.
///
/// The returned `OnceCell` lets the caller `get_or_try_init` the compile
/// exactly once across threads. The DashMap shard lock is released before
/// this returns, so the slow compile never holds it.
pub fn kernel_cache_slot(key_str: &str) -> Arc<OnceCell<CompiledKernel>> {
    let entry = get_kernel_cache()
        .entry(key_str.to_string())
        .or_insert_with(|| Arc::new(OnceCell::new()));
    Arc::clone(entry.value())
}

// ── Per-thread device state ──────────────────────────────────────────────────

/// Per-thread, per-device state: scheduling policy, deallocator stream, and
/// optional memory pool.
///
/// The CUDA context and kernel cache are global (see above). This struct only
/// holds thread-local state.
pub struct AsyncDeviceContext {
    #[expect(dead_code, reason = "will be used when multi-device is implemented")]
    device_id: usize,
    deallocator_stream: Arc<Stream>,
    policy: Arc<dyn SchedulingPolicy>,
    pool: Option<Arc<MemPool>>,
}

pub struct AsyncDeviceContexts {
    default_device: Cell<usize>,
    devices: Cell<Option<HashMap<usize, AsyncDeviceContext>>>,
}

// Manage a statically accessible device context, and their associated streams.
thread_local!(static DEVICE_CONTEXTS: AsyncDeviceContexts = const {
    AsyncDeviceContexts {
        default_device: Cell::new(DEFAULT_DEVICE_ID),
        devices: Cell::new(None),
    }
});

/// Returns the current thread's default GPU device ID.
///
/// This is the device used by `.sync()`, `.await`, and other operations that do not
/// specify a device explicitly. Defaults to [`DEFAULT_DEVICE_ID`] (0).
pub fn get_default_device() -> usize {
    DEVICE_CONTEXTS.with(|ctx| ctx.default_device.get())
}

/// Initialize the device context map for the current thread.
///
/// Call this **before** any GPU work if you need to change the default device or
/// pre-allocate contexts for multiple devices. Individual device contexts are still
/// lazily created on first access (with the default round-robin policy) if not
/// explicitly added via [`init_device`].
///
/// # Panics
///
/// Panics if contexts have already been initialized on this thread.
pub fn init_device_contexts(
    default_device_id: usize,
    num_devices: usize,
) -> Result<(), DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        device_assert(
            default_device_id,
            ctx.devices.replace(None).is_none(),
            "Context already initialized.",
        )
    })?;
    let devices = HashMap::with_capacity(num_devices);
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
        ctx.devices.set(Some(devices));
    });
    Ok(())
}

pub fn init_device_contexts_default() -> Result<(), DeviceError> {
    let default_device = get_default_device();
    // TODO (hme): Detect number of devices.
    init_device_contexts(default_device, DEFAULT_NUM_DEVICES)
}

/// Create a new [`AsyncDeviceContext`] with a custom scheduling policy.
///
/// This is the low-level constructor. Most users should use [`init_device`] or let the
/// runtime auto-initialize with the default policy.
pub fn new_device_context(
    device_id: usize,
    policy: Arc<dyn SchedulingPolicy>,
) -> Result<AsyncDeviceContext, DeviceError> {
    let device = get_or_init_device(device_id)?;
    let deallocator_stream = device.new_stream()?;
    Ok(AsyncDeviceContext {
        device_id,
        deallocator_stream,
        policy,
        pool: None,
    })
}

/// Add a device with a specific scheduling policy to the context map.
///
/// # Example: Using 8 streams instead of the default 4
///
/// ```rust,ignore
/// use cuda_async::device_context::*;
/// use cuda_async::scheduling_policies::*;
///
/// // Before any GPU work:
/// init_device_contexts(0, 1).unwrap();
/// // Then add device 0 with a custom stream pool size:
/// let policy = unsafe { StreamPoolRoundRobin::new(0, 8) };
/// // (use with_global_device_context_mut or init_device internally)
/// ```
pub fn init_device(
    hashmap: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
    policy: Arc<dyn SchedulingPolicy>,
) -> Result<(), DeviceError> {
    let device_context = new_device_context(device_id, policy)?;
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

pub fn init_with_default_policy(
    hashmap: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
) -> Result<(), DeviceError> {
    let device = get_or_init_device(device_id)?;
    let policy = StreamPoolRoundRobin::new(&device, DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE)?;
    let deallocator_stream = device.new_stream()?;
    let device_context = AsyncDeviceContext {
        device_id,
        deallocator_stream,
        policy: Arc::new(policy),
        pool: None,
    };
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

pub fn with_global_device_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&AsyncDeviceContext) -> R,
{
    DEVICE_CONTEXTS.with(|ctx| {
        let mut hashmap = match ctx.devices.take() {
            Some(hashmap) => hashmap,
            None => {
                init_device_contexts_default()?;
                ctx.devices
                    .take()
                    .ok_or(device_error(device_id, "Failed to initialize context"))?
            }
        };
        if !hashmap.contains_key(&device_id) {
            init_with_default_policy(&mut hashmap, device_id)?;
        }
        let device_context = hashmap
            .get(&device_id)
            .ok_or(device_error(device_id, "Failed to get context"))?;
        let r = f(device_context);
        ctx.devices.replace(Some(hashmap));
        Ok(r)
    })
}

pub fn with_global_device_context_mut<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&mut AsyncDeviceContext) -> R,
{
    DEVICE_CONTEXTS.with(|ctx| {
        let mut hashmap = match ctx.devices.take() {
            Some(hashmap) => hashmap,
            None => {
                init_device_contexts_default()?;
                ctx.devices
                    .take()
                    .ok_or(device_error(device_id, "Failed to initialize context"))?
            }
        };
        if !hashmap.contains_key(&device_id) {
            init_with_default_policy(&mut hashmap, device_id)?;
        }
        let device_context = hashmap
            .get_mut(&device_id)
            .ok_or(device_error(device_id, "Failed to get context"))?;
        let r = f(device_context);
        ctx.devices.replace(Some(hashmap));
        Ok(r)
    })
}

/// Run a closure with a reference to the scheduling policy for `device_id`.
pub fn with_device_policy<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<dyn SchedulingPolicy>) -> R,
{
    with_global_device_context(device_id, |device_context| f(&device_context.policy))
}

/// Get a cloned `Arc` of the scheduling policy for `device_id`.
///
/// Useful when you need to schedule operations on a specific device outside the
/// default `.await` / `.sync()` path.
pub fn global_policy(device_id: usize) -> Result<Arc<dyn SchedulingPolicy>, DeviceError> {
    with_global_device_context(device_id, |device_context| device_context.policy.clone())
}

pub unsafe fn with_deallocator_stream<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<Stream>) -> R,
{
    with_global_device_context(device_id, |device_context| {
        f(&device_context.deallocator_stream)
    })
}

/// Run a closure with a reference to the [`Device`] for `device_id`.
pub fn with_device<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<Device>) -> R,
{
    let device = get_or_init_device(device_id)?;
    Ok(f(&device))
}

// Default device policy.

/// Change the default GPU device for the current thread.
///
/// All subsequent `.sync()`, `.await`, and `with_default_device_policy` calls on this
/// thread will target `default_device_id`. The context for that device is lazily created
/// with the default round-robin policy if it doesn't already exist.
///
/// # Multi-GPU Example
///
/// ```rust,ignore
/// // Thread dedicated to device 1:
/// set_default_device(1);
/// let tensor = api::zeros(&[1024, 1024]).await; // runs on GPU 1
/// ```
pub fn set_default_device(default_device_id: usize) {
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
    })
}

/// Set a custom memory pool for the given device **on the current thread**.
///
/// Subsequent allocations on this device will use the given pool instead of the
/// default pool. The pool is resolved at scheduling time (`.sync()`, `.await`,
/// `.schedule()`, `.sync_on()`, `.async_on()`) and carried on
/// [`ExecutionContext`](crate::device_operation::ExecutionContext), so it also
/// applies to futures that are later polled on other threads.
///
/// # Thread-locality
///
/// `AsyncDeviceContext` — and therefore the pool registration — lives in a
/// `thread_local!`. Calling `set_device_pool(0, pool)` on thread A does **not**
/// affect allocations scheduled by thread B on device 0.
///
/// If you build a `DeviceFuture` on one thread and move it to another, the pool
/// travels with the future via its `ExecutionContext` snapshot — the destination
/// thread does not need its own `set_device_pool` call. But if thread B
/// independently creates ops via `.sync()`/`.await`, those ops see thread B's
/// pool (typically `None` unless B also called `set_device_pool`).
///
/// Multi-threaded workers that want a shared pool should each call
/// `set_device_pool` during their initialization.
///
/// # Errors
///
/// Returns [`DeviceError::Context`](crate::error::DeviceError::Context) if
/// `pool` was created on a different device than `device_id`.
pub fn set_device_pool(device_id: usize, pool: Arc<MemPool>) -> Result<(), DeviceError> {
    let pool_device = pool.device().ordinal();
    device_assert(
        device_id,
        pool_device == device_id,
        &format!("pool belongs to device {pool_device}, expected device {device_id}"),
    )?;
    with_global_device_context_mut(device_id, |device_context| {
        device_context.pool = Some(pool);
    })
}

/// Clear the custom memory pool for the given device **on the current thread**,
/// reverting to the default pool.
///
/// Only affects the calling thread's pool registration; see
/// [`set_device_pool`] for the full thread-locality contract. In-flight
/// `DeviceFuture`s that already captured the pool are unaffected (the pool is
/// kept alive via `Arc` until those futures complete).
pub fn clear_device_pool(device_id: usize) -> Result<(), DeviceError> {
    with_global_device_context_mut(device_id, |device_context| {
        device_context.pool = None;
    })
}

/// Returns the custom memory pool registered for the given device **on the
/// current thread**, if any.
///
/// Returns `Ok(None)` when the calling thread has not registered a pool, even
/// if another thread has done so. See [`set_device_pool`] for thread-locality.
pub fn get_device_pool(device_id: usize) -> Result<Option<Arc<MemPool>>, DeviceError> {
    with_global_device_context(device_id, |device_context| device_context.pool.clone())
}

/// Resolve the custom memory pool associated with the device that owns `stream`.
///
/// Errors from the device-context lookup are downgraded to `None`; this is the
/// single choke-point for that decision so callers don't each re-derive it.
pub fn pool_for_stream(stream: &Arc<Stream>) -> Option<Arc<MemPool>> {
    get_device_pool(stream.device().ordinal()).ok().flatten()
}

/// Run a closure with the scheduling policy of the current thread's default device.
///
/// This is the function called internally by
/// [`DeviceOp::sync()`](crate::device_operation::DeviceOp::sync) and by the
/// [`IntoFuture`](std::future::IntoFuture) implementation to schedule operations
/// when no explicit device is given.
pub fn with_default_device_policy<F, R>(f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<dyn SchedulingPolicy>) -> R,
{
    let default_device = get_default_device();
    with_global_device_context(default_device, |device_context| f(&device_context.policy))
}

// Kernel operations — compile, cache, and retrieve GPU kernels.

/// Load a compiled CUDA module from a `.cubin` file.
pub fn load_module_from_file(filename: &str, device_id: usize) -> Result<Arc<Module>, DeviceError> {
    with_device(device_id, |device| {
        let module = device.load_module_from_file(filename)?;
        Ok(module)
    })?
}

/// JIT-compile a PTX string into a CUDA module for the given device.
pub fn load_module_from_ptx(ptx_src: &str, device_id: usize) -> Result<Arc<Module>, DeviceError> {
    with_device(device_id, |device| {
        let module = device.load_module_from_ptx_src(ptx_src)?;
        Ok(module)
    })?
}

/// Check whether a kernel with the given key has already been compiled and cached.
pub fn contains_cuda_function(_device_id: usize, func_key: &impl FunctionKey) -> bool {
    let key = func_key.get_hash_string();
    let cache = get_kernel_cache();
    if let Some(slot) = cache.get(&key) {
        let lock: &OnceCell<CompiledKernel> = slot.value().as_ref();
        lock.get().is_some()
    } else {
        false
    }
}

