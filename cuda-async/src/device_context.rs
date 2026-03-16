/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-local GPU device state, kernel cache, and scheduling policy management.

use crate::error::{device_assert, device_error, DeviceError};
use crate::scheduling_policies::{GlobalSchedulingPolicy, SchedulingPolicy, StreamPoolRoundRobin};
use cuda_core::{CudaContext, CudaFunction, CudaModule, CudaStream};
use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

/// The GPU device used when no explicit device is specified. Device 0 is the first GPU.
pub const DEFAULT_DEVICE_ID: usize = 0;

/// The number of GPU devices initialized by default.
pub const DEFAULT_NUM_DEVICES: usize = 1;

/// Number of CUDA streams in the default round-robin pool.
///
/// Consecutive operations cycle through streams 0…N-1, allowing up to N operations to
/// overlap. Set to 1 for behavior equivalent to [`SingleStream`](crate::scheduling_policies::SingleStream).
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
/// One instance per GPU device, stored in a thread-local map. Lazily initialized on first
/// use with the default round-robin policy. Call [`init_device_contexts`] before any GPU
/// work to customize.
// TODO (hme): None of this needs to be compiled per thread.
pub struct AsyncDeviceContext {
    #[expect(dead_code, reason = "will be used when multi-device is implemented")]
    device_id: usize,
    // TODO: (hme): This will hurt perf due to contention. This should at least be static (OnceLock?).
    context: Arc<CudaContext>,
    deallocator_stream: Arc<CudaStream>,
    policy: Arc<GlobalSchedulingPolicy>,
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

/// Returns the current thread's default GPU device ID (defaults to [`DEFAULT_DEVICE_ID`]).
pub fn get_default_device() -> usize {
    DEVICE_CONTEXTS.with(|ctx| ctx.default_device.get())
}

/// Initialize the thread-local device context map.
///
/// Call before any GPU work to change the default device or pre-allocate contexts.
/// Contexts not explicitly added are still lazily created on first access.
///
/// # Panics
///
/// Panics if already initialized on this thread.
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

/// Low-level constructor for a device context with a custom scheduling policy.
/// Prefer [`init_device`] or auto-initialization for typical use.
pub fn new_device_context(
    device_id: usize,
    mut policy: GlobalSchedulingPolicy,
) -> Result<AsyncDeviceContext, DeviceError> {
    // device_id is a usize, device_id >= 0 is always true.
    let context = CudaContext::new(device_id)?;
    policy.init(&context)?;
    let deallocator_stream = context.new_stream()?;
    Ok(AsyncDeviceContext {
        device_id,
        context,
        deallocator_stream,
        policy: Arc::new(policy),
        functions: HashMap::new(),
        validators: HashMap::new(),
    })
}

/// Add a device with a specific scheduling policy to the context map.
///
/// ```rust,ignore
/// use cuda_async::device_context::*;
/// use cuda_async::scheduling_policies::*;
///
/// init_device_contexts(0, 1).unwrap();
/// let policy = unsafe { StreamPoolRoundRobin::new(0, 8) };
/// init_device(&mut hashmap, 0, GlobalSchedulingPolicy::RoundRobin(policy)).unwrap();
/// ```
pub fn init_device(
    hashmap: &mut HashMap<usize, AsyncDeviceContext>,
    device_id: usize,
    policy: GlobalSchedulingPolicy,
) -> Result<(), DeviceError> {
    let device_context = new_device_context(device_id, policy)?;
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

/// Returns a cloned `Arc` of the scheduling policy for `device_id`.
pub fn global_policy(device_id: usize) -> Result<Arc<GlobalSchedulingPolicy>, DeviceError> {
    with_global_device_context(device_id, |device_context| {
        let policy = device_context.policy.clone();
        policy
    })
}

pub unsafe fn with_deallocator_stream<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<CudaStream>) -> R,
{
    with_global_device_context(device_id, |device_context| {
        f(&device_context.deallocator_stream)
    })
}

pub fn with_cuda_context<F, R>(device_id: usize, f: F) -> Result<R, DeviceError>
where
    F: FnOnce(&Arc<CudaContext>) -> R,
{
    with_global_device_context(device_id, |device_context| f(&device_context.context))
}

// Default device policy.

/// Set the default GPU device for the current thread.
///
/// All subsequent `.sync()` and `.await` calls target this device. The context is
/// lazily created with the default round-robin policy if it doesn't exist.
pub fn set_default_device(default_device_id: usize) -> () {
    DEVICE_CONTEXTS.with(|ctx| {
        ctx.default_device.set(default_device_id);
    })
}

/// Run a closure with the default device's scheduling policy.
/// Called internally by [`DeviceOperation::sync()`] and [`IntoFuture`].
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

/// Cache a compiled kernel so future calls with the same [`FunctionKey`] skip compilation.
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

/// Retrieve a cached kernel by key.
///
/// # Panics
///
/// Panics if no function with the given key exists. Check with [`contains_cuda_function`] first.
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
