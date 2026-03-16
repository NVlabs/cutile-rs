/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Async runtime for CUDA device operations, providing futures-based kernel launching
//! and device memory management.

#![feature(slice_ptr_get)]
#![feature(type_alias_impl_trait)]
#![feature(associated_type_defaults)]
#![feature(unsafe_cell_access)]

pub mod device_box;
pub mod device_context;
pub mod device_future;
pub mod device_operation;
pub mod error;
pub mod launch;
pub mod scheduling_policies;

pub use futures;
