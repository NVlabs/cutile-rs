/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Entry point for GPU-dependent integration tests.

#[path = "common/mod.rs"]
mod common;

#[path = "gpu/tensor.rs"]
mod tensor;

#[path = "gpu/num_tiles.rs"]
mod num_tiles;

#[path = "gpu/partial_coverage.rs"]
mod partial_coverage;

#[path = "gpu/launch_preconditions.rs"]
mod launch_preconditions;

// ---------------------------------------------------------------------------
// Migrated from cutile-examples (smoke tests for kernel patterns that were
// previously exposed as runnable examples but don't teach a pattern worth
// keeping in the examples drawer).
// ---------------------------------------------------------------------------

#[path = "gpu/add_basic.rs"]
mod add_basic;

#[path = "gpu/add_ptr.rs"]
mod add_ptr;

#[path = "gpu/const_pointers.rs"]
mod const_pointers;

#[path = "gpu/program_id.rs"]
mod program_id;

#[path = "gpu/add_refs.rs"]
mod add_refs;

#[path = "gpu/global_counter.rs"]
mod global_counter;

#[path = "gpu/inter_module.rs"]
mod inter_module;

#[path = "gpu/tensor_slicing.rs"]
mod tensor_slicing;

#[path = "gpu/async_saxpy_unsafe.rs"]
mod async_saxpy_unsafe;

#[path = "gpu/async_device_op.rs"]
mod async_device_op;

#[path = "gpu/book_snippets.rs"]
mod book_snippets;

#[path = "gpu/tensor_permute.rs"]
mod tensor_permute;

#[path = "gpu/mapped_partition_values.rs"]
mod mapped_partition_values;

#[path = "gpu/mapped_partition_schedule_matrix.rs"]
mod mapped_partition_schedule_matrix;

#[path = "gpu/atomic_red_view.rs"]
mod atomic_red_view;

#[path = "gpu/borrow_raw_parts.rs"]
mod borrow_raw_parts;

#[path = "gpu/device_debug.rs"]
mod device_debug;
