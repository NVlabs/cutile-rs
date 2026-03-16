/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA driver error types and result conversion utilities.

use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// Wrapper around a CUDA driver API error code.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub cuda_bindings::CUresult);

impl DriverError {
    fn _fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        match self.error_string() {
            Ok(err_str) => formatter
                .debug_tuple("DriverError")
                .field(&self.0)
                .field(&err_str)
                .finish(),
            Err(_) => formatter
                .debug_tuple("DriverError")
                .field(&self.0)
                .field(&"<Failure when calling cuGetErrorString()>")
                .finish(),
        }
    }
}

impl Display for DriverError {
    fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        self._fmt(formatter)
    }
}

impl std::fmt::Debug for DriverError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> fmt::Result {
        self._fmt(formatter)
    }
}

impl error::Error for DriverError {}

/// Converts a CUDA driver call return value into a `Result`.
pub trait IntoResult<T> {
    /// Returns `Ok` on `CUDA_SUCCESS`, or `Err(DriverError)` otherwise.
    fn result(self) -> Result<T, DriverError>
    where
        Self: Sized;
}

impl IntoResult<()> for cuda_bindings::CUresult {
    fn result(self) -> Result<(), DriverError> {
        match self {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(()),
            _ => Err(DriverError(self)),
        }
    }
}

impl<T> IntoResult<T> for (cuda_bindings::CUresult, T) {
    fn result(self) -> Result<T, DriverError> {
        match self.0 {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(self.1),
            _ => Err(DriverError(self.0)),
        }
    }
}

impl<T> IntoResult<T> for (cuda_bindings::CUresult, MaybeUninit<T>) {
    fn result(self) -> Result<T, DriverError> {
        match self.0 {
            cuda_bindings::cudaError_enum_CUDA_SUCCESS => Ok(unsafe { self.1.assume_init() }),
            _ => Err(DriverError(self.0)),
        }
    }
}

impl DriverError {
    /// Returns the short error name string for this CUDA error code.
    pub fn error_name(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGetErrorName(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }

    /// Returns the human-readable description string for this CUDA error code.
    pub fn error_string(&self) -> Result<&CStr, DriverError> {
        let mut err_str = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGetErrorString(self.0, err_str.as_mut_ptr()).result()?;
            Ok(CStr::from_ptr(err_str.assume_init()))
        }
    }
}
