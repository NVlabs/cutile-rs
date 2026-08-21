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
    /// Returns true when the driver cannot JIT the PTX version in a module.
    ///
    /// This usually means the selected CUDA toolkit is newer than the
    /// installed driver. PTX requires direct driver support even when other
    /// parts of the toolkit can use CUDA minor-version compatibility.
    pub fn is_unsupported_ptx_version(&self) -> bool {
        self.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION
    }

    fn _fmt(&self, formatter: &mut Formatter) -> fmt::Result {
        if self.0 == cuda_bindings::cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED {
            if let Some(load_error) = cuda_bindings::cuda_driver_load_error() {
                return formatter
                    .debug_tuple("DriverError")
                    .field(&self.0)
                    .field(&format!("CUDA driver library unavailable: {load_error}"))
                    .finish();
            }
        }

        let help = "the CUDA driver cannot JIT PTX from the selected toolkit; upgrade the driver \
                    or select a compatible toolkit with CUDA_TOOLKIT_PATH or CUDA_HOME";

        let mut output = formatter.debug_tuple("DriverError");
        output.field(&self.0);
        match self.error_string() {
            Ok(err_str) => {
                output.field(&err_str);
            }
            Err(_) => {
                output.field(&"<Failure when calling cuGetErrorString()>");
            }
        }
        if self.is_unsupported_ptx_version() {
            output.field(&help);
        }
        output.finish()
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

#[cfg(test)]
mod tests {
    use super::DriverError;

    #[test]
    fn identifies_unsupported_ptx_version() {
        let unsupported =
            DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
        let unrelated = DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE);

        assert!(unsupported.is_unsupported_ptx_version());
        assert!(!unrelated.is_unsupported_ptx_version());
    }

    #[test]
    fn shared_object_init_failed_includes_loader_error_when_driver_is_missing() {
        if cuda_bindings::cuda_driver_load_error().is_none() {
            return;
        }

        let formatted =
            DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_SHARED_OBJECT_INIT_FAILED)
                .to_string();

        assert!(
            formatted.contains("CUDA driver library unavailable"),
            "expected a human-readable loader hint, got: {formatted}"
        );
        assert!(
            formatted.contains("failed to load any of"),
            "expected the cached loader failure context to be surfaced, got: {formatted}"
        );
        assert!(
            formatted.contains("libcuda"),
            "expected the driver library name to be surfaced, got: {formatted}"
        );
    }
}
