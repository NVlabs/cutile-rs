/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel-cache warmup tests, isolated in their own test binary.
//!
//! These tests assert on process-global state (the kernel cache and the JIT
//! compile counter). Inside the shared gpu binary they raced the other GPU
//! tests — cache_test_lock only serializes warmup tests against each other —
//! causing parallel-run segfaults and flaky counter assertions. A separate
//! integration-test target gives them their own process.

mod common;

#[path = "gpu/warmup.rs"]
mod warmup;

#[path = "gpu/warmup_bench.rs"]
mod warmup_bench;
