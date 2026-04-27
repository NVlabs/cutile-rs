// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

//! Runtime loading policy for CUDA driver and cuRAND.
//!
//! Bindgen-generated wrappers own the ABI surface and function signatures.
//! This module only chooses library names, caches the generated wrappers, and
//! exposes the stable flat free-function API used throughout the workspace.

use std::sync::OnceLock;

#[allow(unused_imports)]
use crate::*;

type GeneratedCudaDriverApi = crate::generated_cuda::CudaDriverApi;
type GeneratedCurandApi = crate::generated_curand::CurandApi;

#[derive(Debug)]
pub enum DynLoadError {
    LoadFailed {
        names: &'static [&'static str],
        source: libloading::Error,
    },
}

impl std::fmt::Display for DynLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DynLoadError::LoadFailed { names, source } => {
                write!(f, "failed to load any of {names:?}: {source}")
            }
        }
    }
}

impl std::error::Error for DynLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DynLoadError::LoadFailed { source, .. } => Some(source),
        }
    }
}

#[cfg(target_os = "linux")]
const CUDA_LIB_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];
#[cfg(target_os = "macos")]
const CUDA_LIB_NAMES: &[&str] = &["libcuda.dylib"];
#[cfg(target_os = "windows")]
const CUDA_LIB_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CUDA_LIB_NAMES: &[&str] = &["libcuda.so"];

#[cfg(target_os = "linux")]
const CURAND_LIB_NAMES: &[&str] = &["libcurand.so", "libcurand.so.10"];
#[cfg(target_os = "macos")]
const CURAND_LIB_NAMES: &[&str] = &["libcurand.dylib"];
#[cfg(target_os = "windows")]
const CURAND_LIB_NAMES: &[&str] = &["curand64_10.dll", "curand64_12.dll"];
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CURAND_LIB_NAMES: &[&str] = &["libcurand.so"];

trait GeneratedApi: Sized {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error>;
}

impl GeneratedApi for GeneratedCudaDriverApi {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error> {
        unsafe { Self::new(path) }
    }
}

impl GeneratedApi for GeneratedCurandApi {
    unsafe fn open(path: &str) -> Result<Self, libloading::Error> {
        unsafe { Self::new(path) }
    }
}

fn load_api<T: GeneratedApi>(names: &'static [&'static str]) -> Result<T, DynLoadError> {
    let mut last_error = None;
    for &name in names {
        match unsafe { T::open(name) } {
            Ok(api) => return Ok(api),
            Err(error) => last_error = Some(error),
        }
    }

    Err(DynLoadError::LoadFailed {
        names,
        source: last_error.expect("library candidate lists must be non-empty"),
    })
}

fn cached_api<T: GeneratedApi>(
    slot: &'static OnceLock<Result<T, DynLoadError>>,
    names: &'static [&'static str],
) -> Result<&'static T, &'static DynLoadError> {
    slot.get_or_init(|| load_api::<T>(names)).as_ref()
}

static CUDA_DRIVER: OnceLock<Result<GeneratedCudaDriverApi, DynLoadError>> = OnceLock::new();
static CURAND: OnceLock<Result<GeneratedCurandApi, DynLoadError>> = OnceLock::new();

fn cuda_driver() -> Result<&'static GeneratedCudaDriverApi, &'static DynLoadError> {
    cached_api(&CUDA_DRIVER, CUDA_LIB_NAMES)
}

fn curand_api() -> Result<&'static GeneratedCurandApi, &'static DynLoadError> {
    cached_api(&CURAND, CURAND_LIB_NAMES)
}

pub fn is_cuda_driver_available() -> bool {
    cuda_driver().is_ok()
}

pub fn cuda_driver_load_error() -> Option<&'static DynLoadError> {
    cuda_driver().err()
}

pub fn is_curand_available() -> bool {
    curand_api().is_ok()
}

pub fn curand_load_error() -> Option<&'static DynLoadError> {
    curand_api().err()
}

include!(concat!(env!("OUT_DIR"), "/cuda_driver_shims.rs"));
include!(concat!(env!("OUT_DIR"), "/curand_shims.rs"));
