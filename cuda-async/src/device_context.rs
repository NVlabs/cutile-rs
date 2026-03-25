/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-local GPU device state, kernel cache, and scheduling policy management.

use crate::error::{device_assert, device_error, DeviceError};
use crate::scheduling_policies::{GlobalSchedulingPolicy, SchedulingPolicy, StreamPoolRoundRobin};
use cuda_core::{CudaContext, CudaFunction, CudaMemPool, CudaModule, CudaStream};
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

type DeviceFunctions = HashMap<String, (Arc<CudaModule>, Arc<CudaFunction>)>;
type DeviceFunctionValidators = HashMap<String, Arc<Validator>>;

/// Per-device state: CUDA context, scheduling policy, and compiled kernel cache.
///
/// Each GPU device has one `AsyncDeviceContext` stored in a thread-local map. It holds:
///
/// - A [`CudaContext`] for driver API calls.
/// - A [`GlobalSchedulingPolicy`] that decides which stream each operation runs on.
/// - A cache of already-compiled kernel functions (keyed by [`FunctionKey::get_hash_string()`]).
///
/// The context is lazily initialized on first use with the default round-robin policy
/// ([`DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE`] = 4 streams). To customize, call
/// [`init_device_contexts`] before any GPU work.
// TODO (hme): None of this needs to be compiled per thread.
pub struct AsyncDeviceContext {
    #[expect(dead_code, reason = "will be used when multi-device is implemented")]
    device_id: usize,
    // TODO: (hme): This will hurt perf due to contention. This should at least be static (OnceLock?).
    context: Arc<CudaContext>,
    deallocator_stream: Arc<CudaStream>,
    policy: Arc<GlobalSchedulingPolicy>,
    pool: Option<Arc<CudaMemPool>>,
    functions: DeviceFunctions,
    validators: DeviceFunctionValidators,
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

/// Create a new [`AsyncDeviceContext`] with a custom scheduling policy and
/// an optional memory pool.
///
/// When `pool` is `Some`, all allocations via [`device_alloc_async`] on this
/// context will route through `cuMemAllocFromPoolAsync`. The pool's device
/// ordinal must match `device_id`.
///
/// Most users should use [`init_device`] or let the runtime auto-initialize.
pub fn new_device_context(
    device_id: usize,
    mut policy: GlobalSchedulingPolicy,
    pool: Option<Arc<CudaMemPool>>
) -> Result<AsyncDeviceContext, DeviceError> {
    if let Some(ref p) = pool {
        if p.device_ordinal() != device_id {
            return Err(device_error(
                device_id,
                &format!(
                    "Pool belongs to device {}, but target device is {}.",
                    p.device_ordinal(),
                    device_id,
                ),
            ));
        }
    }
    let context = CudaContext::new(device_id)?;
    policy.init(&context)?;
    let deallocator_stream = context.new_stream()?;
    Ok(AsyncDeviceContext {
        device_id,
        context,
        deallocator_stream,
        policy: Arc::new(policy),
        pool,
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
    policy: GlobalSchedulingPolicy,
    pool: Option<Arc<CudaMemPool>>,
) -> Result<(), DeviceError> {
    let device_context = new_device_context(device_id, policy, pool)?;
    let pred = hashmap.insert(device_id, device_context).is_none();
    device_assert(device_id, pred, "Device is already initialized.")
}

pub fn init_with_default_policy(
    hashmap: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
) -> Result<(), DeviceError> {
    let policy =
        unsafe { StreamPoolRoundRobin::new(device_id, DEFAULT_ROUND_ROBIN_STREAM_POOL_SIZE) };
    init_device(
        hashmap,
        device_id,
        GlobalSchedulingPolicy::RoundRobin(policy),
        None,
    )
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
    F: FnOnce(&Arc<GlobalSchedulingPolicy>) -> R,
{
    with_global_device_context(device_id, |device_context| f(&device_context.policy))
}

/// Get a cloned `Arc` of the scheduling policy for `device_id`.
///
/// Useful when you need to schedule operations on a specific device outside the
/// default `.await` / `.sync()` path.
pub fn global_policy(device_id: usize) -> Result<Arc<GlobalSchedulingPolicy>, DeviceError> {
    with_global_device_context(device_id, |device_context| device_context.policy.clone())
}

pub unsafe fn with_deallocator_stream<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<CudaStream>) -> R,
{
    with_global_device_context(device_id, |device_context| {
        f(&device_context.deallocator_stream)
    })
}

/// Set the memory pool for all subsequent allocations on `device_id`
/// **for the calling thread only**.
///
/// `AsyncDeviceContext` is thread-local. This configuration is not visible
/// to other threads. Each thread that needs pool-routed allocation must call
/// this independently.
///
/// Pass `None` to revert to the default `cuMemAllocAsync` path.
///
/// The pool is held via `Arc`, so it cannot be desroyed while any device
/// context still references it. 
/// 
/// # Errors
///
/// Returns `DeviceError:Context` if the pool's device ordinal does not
/// match `device_id`, preventing cross-device mismatches.
pub fn set_device_pool(
    device_id: usize,
    pool: Option<Arc<CudaMemPool>>,
) -> Result<(), DeviceError> {
    if let Some(ref p) = pool {
        if p.device_ordinal() != device_id {
            return Err(device_error(
                device_id,
                &format!(
                    "Pool belongs to device {}, but target device is {}.",
                    p.device_ordinal(),
                    device_id,
                ),
            ));
        }
    }
    with_global_device_context_mut(device_id, |dc| {
        dc.pool = pool;
    })
}

/// Allocate device memory, routing through the configured pool if one is set.
/// 
/// This is the pool-aware replacement for calling `cuda_core::malloc_async` directly.
/// If no pool is configured for `device_id`, falls back to cuMemAllocAsync`.
///
/// # Safety
/// `stream` must be a valid CUDA stream.
///
/// # Errors
/// Returns `DeviceError` if the context lookup or allocation failes (e.g., OOM)
pub unsafe fn device_alloc_async(
    num_bytes: usize,
    stream: &Arc<cuda_core::CudaStream>,
    device_id: usize,
) -> Result<cuda_core::sys::CUdeviceptr, DeviceError> {
    let pool_handle = with_global_device_context(device_id, |dc| {
        dc.pool.as_ref().map(|p| p.handle())
    })?;
    let dptr = match pool_handle {
        Some(handle) => cuda_core::malloc_from_pool_async(num_bytes, stream, handle)?,
        None => cuda_core::malloc_async(num_bytes, stream),
    };
    Ok(dptr)
}

pub fn with_cuda_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<CudaContext>) -> R,
{
    with_global_device_context(device_id, |device_context| f(&device_context.context))
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
/// let tensor = api::zeros([1024, 1024]).await; // runs on GPU 1
/// ```
pub fn set_default_device(default_device_id: usize) {
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
    })
}

/// Run a closure with the scheduling policy of the current thread's default device.
///
/// This is the function called internally by [`DeviceOperation::sync()`] and by the
/// [`IntoFuture`] implementation to schedule operations when no explicit device is given.
pub fn with_default_device_policy<F, R>(f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<GlobalSchedulingPolicy>) -> R,
{
    let default_device = get_default_device();
    with_global_device_context(default_device, |device_context| f(&device_context.policy))
}

// Kernel operations — compile, cache, and retrieve GPU kernels.

/// Load a compiled CUDA module from a `.cubin` file.
pub fn load_module_from_file(
    filename: &str,
    device_id: usize,
) -> Result<Arc<CudaModule>, DeviceError> {
    with_cuda_context(device_id, |cuda_ctx| {
        let module = cuda_ctx.load_module_from_file(filename)?;
        Ok(module)
    })?
}

/// JIT-compile a PTX string into a CUDA module for the given device.
pub fn load_module_from_ptx(
    ptx_src: &str,
    device_id: usize,
) -> Result<Arc<CudaModule>, DeviceError> {
    with_cuda_context(device_id, |cuda_ctx| {
        let module = cuda_ctx.load_module_from_ptx_src(ptx_src)?;
        Ok(module)
    })?
}

/// Store a compiled kernel in the per-device cache so that future calls with the same
/// [`FunctionKey`] can skip compilation.
pub fn insert_cuda_function(
    device_id: usize,
    func_key: &impl FunctionKey,
    value: (Arc<CudaModule>, Arc<CudaFunction>),
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
) -> Result<Arc<CudaFunction>, DeviceError> {
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
