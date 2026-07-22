/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async runtime for CUDA device operations, providing futures-based kernel launching
//! and device memory management.

pub mod cuda_graph;
pub mod device_buffer;
pub mod device_context;
pub mod device_future;
pub mod device_operation;
pub mod error;
pub mod launch;
#[cfg(any(feature = "reactor", test))]
mod loom_compat;
pub mod prelude;
#[cfg(feature = "reactor")]
mod reactor;
pub mod scheduling_policies;
#[cfg(any(feature = "reactor", test))]
mod slot_table;

pub use futures;
