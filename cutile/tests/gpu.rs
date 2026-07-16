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

// ---------------------------------------------------------------------------
// Migrated from cutile-examples (smoke tests for kernel patterns that were
// previously exposed as runnable examples but don't teach a pattern worth
// keeping in the examples drawer).
// ---------------------------------------------------------------------------

#[path = "gpu/add_basic.rs"]
mod add_basic;

#[path = "gpu/add_ptr.rs"]
mod add_ptr;

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

#[path = "gpu/warmup.rs"]
mod warmup;

#[path = "gpu/warmup_bench.rs"]
mod warmup_bench;

#[path = "gpu/future_drop.rs"]
mod future_drop;
