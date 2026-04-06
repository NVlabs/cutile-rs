/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-level CUDA driver API bindings and safe wrappers.

// Always available (no CUDA required)
mod dtype;
mod validator;

pub use dtype::*;
pub use validator::*;

// CUDA-dependent modules (requires cuda feature)
#[cfg(feature = "cuda")]
mod api;
#[cfg(feature = "cuda")]
mod cudarc_shim;
#[cfg(feature = "cuda")]
mod error;

#[cfg(feature = "cuda")]
pub use api::*;
#[cfg(feature = "cuda")]
pub use cuda_bindings as sys;
#[cfg(feature = "cuda")]
pub use cudarc_shim::*;
#[cfg(feature = "cuda")]
pub use error::*;
