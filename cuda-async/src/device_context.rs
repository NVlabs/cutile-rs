/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-local GPU device state, kernel cache, and scheduling policy management.

use crate::error::{device_assert, device_error, DeviceError};
use crate::scheduling_policies::{SchedulingPolicy, StreamPoolRoundRobin};
use cuda_core::{Device, Function, MemPool, Module, Stream};
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

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

type DeviceFunctions = HashMap<String, (Arc<Module>, Arc<Function>)>;
type DeviceFunctionValidators = HashMap<String, Arc<Validator>>;

/// Per-device state: GPU device, scheduling policy, and compiled kernel cache.
///
/// Each GPU device has one `AsyncDeviceContext` stored in a thread-local map. It holds:
///
/// - A [`Device`] for driver API calls.
/// - A [`SchedulingPolicy`] that decides which stream each operation runs on.
/// - A cache of already-compiled kernel functions (keyed by [`FunctionKey::get_hash_string()`]).
///
/// The context is lazily initialized on first use with the default round-robin policy
/// ([`DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE`] = 4 streams). To customize, call
/// [`init_device_contexts`] before any GPU work.
// TODO (hme): None of this needs to be compiled per thread.
pub struct AsyncDeviceContext {
    #[expect(dead_code, reason = "will be used when multi-device is implemented")]
    device_id: usize,
    /// Set to `true` when a `_mut` callback on this device panicked. Cleared
    /// via [`clear_device_poison`] or [`reset_device`]. Per-device because a
    /// callback's `&mut AsyncDeviceContext` can only damage this one entry.
    poisoned: bool,
    // TODO: (hme): This will hurt perf due to contention. This should at least be static (OnceLock?).
    device: Arc<Device>,
    deallocator_stream: Arc<Stream>,
    policy: Arc<dyn SchedulingPolicy>,
    pool: Option<Arc<MemPool>>,
    functions: DeviceFunctions,
    validators: DeviceFunctionValidators,
}

/// Lifecycle state of the per-thread device context map. `Borrowed` makes
/// re-entry observable so it can panic instead of silently rebuilding the
/// map (the original cause of #133). Poison lives on individual
/// [`AsyncDeviceContext`] entries, not here.
#[derive(Default)]
enum ContextState {
    #[default]
    Uninitialized,
    Available(HashMap<usize, AsyncDeviceContext>),
    Borrowed,
}

pub struct AsyncDeviceContexts {
    default_device: Cell<usize>,
    devices: Cell<ContextState>,
}

// Manage a statically accessible device context, and their associated streams.
thread_local!(static DEVICE_CONTEXTS: AsyncDeviceContexts = const {
    AsyncDeviceContexts {
        default_device: Cell::new(DEFAULT_DEVICE_ID),
        devices: Cell::new(ContextState::Uninitialized),
    }
});

/// RAII handle on the borrowed map. Drop always restores it to `Available`;
/// if `poison_device_on_panic = Some(id)` and the thread is unwinding, that
/// one device's `poisoned` flag is set on the way out.
struct ContextGuard<'a> {
    cell: &'a Cell<ContextState>,
    map: HashMap<usize, AsyncDeviceContext>,
    poison_device_on_panic: Option<usize>,
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let mut map = std::mem::take(&mut self.map);
        if std::thread::panicking() {
            if let Some(device_id) = self.poison_device_on_panic {
                if let Some(ctx) = map.get_mut(&device_id) {
                    ctx.poisoned = true;
                }
            }
        }
        self.cell.set(ContextState::Available(map));
    }
}

/// Take the device map out of the cell (`Available → Borrowed`), lazy-init
/// if needed. Panics on re-entry — `_mut` callers must arm
/// `poison_device_on_panic` themselves after picking a device.
fn borrow_devices(ctx: &AsyncDeviceContexts) -> Result<ContextGuard<'_>, DeviceError> {
    let map = match ctx.devices.take() {
        ContextState::Available(map) => map,
        ContextState::Uninitialized => {
            init_device_contexts_default()?;
            match ctx.devices.take() {
                ContextState::Available(map) => map,
                _ => {
                    return Err(device_error(
                        get_default_device(),
                        "Failed to initialize context",
                    ));
                }
            }
        }
        ContextState::Borrowed => {
            // Restore Borrowed so any unwinding observer sees the right state.
            ctx.devices.set(ContextState::Borrowed);
            panic!(
                "re-entrant access to device context: every with_*, \
                 is_device_poisoned, clear_device_poison and reset_device \
                 borrows the same thread-local map",
            );
        }
    };
    ctx.devices.set(ContextState::Borrowed);
    Ok(ContextGuard {
        cell: &ctx.devices,
        map,
        poison_device_on_panic: None,
    })
}

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
    DEVICE_CONTEXTS.with(|ctx| match ctx.devices.take() {
        ContextState::Uninitialized => {
            ctx.default_device.set(default_device_id);
            ctx.devices.set(ContextState::Available(HashMap::with_capacity(num_devices)));
            Ok(())
        }
        ContextState::Available(map) => {
            ctx.devices.set(ContextState::Available(map));
            device_assert(default_device_id, false, "Context already initialized.")
        }
        ContextState::Borrowed => {
            ctx.devices.set(ContextState::Borrowed);
            panic!(
                "init_device_contexts called while the device context is \
                 currently borrowed by a callback",
            );
        }
    })
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
    let device = Device::new(device_id)?;
    let deallocator_stream = device.new_stream()?;
    Ok(AsyncDeviceContext {
        device_id,
        poisoned: false,
        device,
        deallocator_stream,
        policy,
        pool: None,
        functions: HashMap::new(),
        validators: HashMap::new(),
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
    let device = Device::new(device_id)?;
    let policy = StreamPoolRoundRobin::new(&device, DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE)?;
    let deallocator_stream = device.new_stream()?;
    let device_context = AsyncDeviceContext {
        device_id,
        poisoned: false,
        device,
        deallocator_stream,
        policy: Arc::new(policy),
        pool: None,
        functions: HashMap::new(),
        validators: HashMap::new(),
    };
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

/// Returns whether `device_id` is poisoned. `Ok(false)` if the device isn't
/// initialized yet.
///
/// # Panics
///
/// Panics if called from inside a `with_global_device_context*` callback —
/// it borrows the same thread-local map. (Inside a callback the device
/// cannot be poisoned anyway: the outer call would have returned `Err` first.)
pub fn is_device_poisoned(device_id: usize) -> Result<bool, DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        let guard = borrow_devices(ctx)?;
        Ok(guard
            .map
            .get(&device_id)
            .map(|c| c.poisoned)
            .unwrap_or(false))
    })
}

/// Clear the poison flag, keeping the existing context (pool, kernel cache,
/// etc.). For a clean rebuild instead, use [`reset_device`]. No-op if the
/// device wasn't poisoned or isn't present.
///
/// # Panics
///
/// Panics if called from inside a `with_global_device_context*` callback —
/// it borrows the same thread-local map. Call it after the outer call returns.
pub fn clear_device_poison(device_id: usize) -> Result<(), DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        let mut guard = borrow_devices(ctx)?;
        if let Some(c) = guard.map.get_mut(&device_id) {
            c.poisoned = false;
        }
        Ok(())
    })
}

/// Drop the entire context entry; the next access lazily rebuilds it with
/// the default policy. Discards pool, kernel cache, and validator state.
/// No-op if the device isn't present.
///
/// # Panics
///
/// Panics if called from inside a `with_global_device_context*` callback —
/// it borrows the same thread-local map. Call it after the outer call returns.
pub fn reset_device(device_id: usize) -> Result<(), DeviceError> {
    DEVICE_CONTEXTS.with(|ctx| {
        let mut guard = borrow_devices(ctx)?;
        guard.map.remove(&device_id);
        Ok(())
    })
}

/// Run `f` with shared access to `device_id`'s context.
///
/// Nested access is unsupported: re-borrowing the per-thread map from
/// within `f` — directly or transitively (`pool_for_stream`,
/// `contains_cuda_function`, `ExecutionContext::new`, the recovery fns) —
/// panics on `Borrowed`. Intentional and guaranteed; see [`borrow_devices`].
///
/// Returns `Err(DeviceError::Context)` if `device_id` is poisoned; recover
/// via [`clear_device_poison`] / [`reset_device`].
pub fn with_global_device_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&AsyncDeviceContext) -> R,
{
    DEVICE_CONTEXTS.with(|ctx| {
        let mut guard = borrow_devices(ctx)?;
        if !guard.map.contains_key(&device_id) {
            init_with_default_policy(&mut guard.map, device_id)?;
        }
        let device_context = guard
            .map
            .get(&device_id)
            .ok_or(device_error(device_id, "Failed to get context"))?;
        if device_context.poisoned {
            return Err(poisoned_error(device_id));
        }
        Ok(f(device_context))
    })
}

/// Run `f` with exclusive access to `device_id`'s context. Same no-nesting
/// contract as [`with_global_device_context`].
///
/// If `f` panics, only `device_id` is poisoned (the guard restores the map
/// to `Available`, so other devices stay usable); access returns
/// `Err(DeviceError::Context)` until [`clear_device_poison`] /
/// [`reset_device`].
pub fn with_global_device_context_mut<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&mut AsyncDeviceContext) -> R,
{
    DEVICE_CONTEXTS.with(|ctx| {
        let mut guard = borrow_devices(ctx)?;
        if !guard.map.contains_key(&device_id) {
            init_with_default_policy(&mut guard.map, device_id)?;
        }
        if guard
            .map
            .get(&device_id)
            .ok_or(device_error(device_id, "Failed to get context"))?
            .poisoned
        {
            return Err(poisoned_error(device_id));
        }
        // Arm before handing out &mut; disarm on clean return so the guard's
        // Drop only poisons when the callback panicked.
        guard.poison_device_on_panic = Some(device_id);
        let device_context = guard
            .map
            .get_mut(&device_id)
            .expect("device entry checked above");
        let result = f(device_context);
        guard.poison_device_on_panic = None;
        Ok(result)
    })
}

fn poisoned_error(device_id: usize) -> DeviceError {
    device_error(
        device_id,
        "device context is poisoned: a previous mutable callback panicked. \
         Call clear_device_poison(id) to resume using the existing context, \
         or reset_device(id) to drop and rebuild it.",
    )
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
    with_global_device_context(device_id, |device_context| f(&device_context.device))
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

/// Store a compiled kernel in the per-device cache so that future calls with the same
/// [`FunctionKey`] can skip compilation.
pub fn insert_cuda_function(
    device_id: usize,
    func_key: &impl FunctionKey,
    value: (Arc<Module>, Arc<Function>),
) -> Result<(), DeviceError> {
    with_global_device_context_mut(device_id, |device_context| {
        let key = func_key.get_hash_string();
        let res = device_context.functions.insert(key.clone(), value);
        device_assert(device_id, res.is_none(), "Unexpected cache key collision.")
    })?
}

/// Check whether a kernel with the given key has already been compiled and cached.
pub fn contains_cuda_function(device_id: usize, func_key: &impl FunctionKey) -> bool {
    with_global_device_context(device_id, |device_context| {
        let key = func_key.get_hash_string();
        device_context.functions.contains_key(&key)
    })
    .is_ok_and(|pred| pred)
}

/// Retrieve a previously compiled kernel from the cache.
///
/// # Panics
///
/// Panics if no function with the given key exists. Use [`contains_cuda_function`] to
/// check first, or rely on the compilation pipeline which always inserts before retrieving.
pub fn get_cuda_function(
    device_id: usize,
    func_key: &impl FunctionKey,
) -> Result<Arc<Function>, DeviceError> {
    with_global_device_context(device_id, |device_context| {
        let key = func_key.get_hash_string();
        let entry = device_context
            .functions
            .get(&key)
            .ok_or(device_error(device_id, "Failed to get cuda function."))?;
        Ok(entry.1.clone())
    })?
}

pub fn insert_function_validator(
    device_id: usize,
    func_key: &impl FunctionKey,
    value: Arc<Validator>,
) -> Result<(), DeviceError> {
    with_global_device_context_mut(device_id, |device_context| {
        let key = func_key.get_hash_string();
        let res = device_context.validators.insert(key.clone(), value);
        device_assert(device_id, res.is_none(), "Unexpected cache key collision.")
    })?
}

pub fn get_function_validator(
    device_id: usize,
    func_key: &impl FunctionKey,
) -> Result<Arc<Validator>, DeviceError> {
    with_global_device_context(device_id, |device_context| {
        let key = func_key.get_hash_string();
        let entry = device_context
            .validators
            .get(&key)
            .ok_or(device_error(device_id, "Failed to get function validator."))?;
        Ok(entry.clone())
    })?
}
