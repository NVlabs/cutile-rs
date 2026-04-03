/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Entry point for GPU-dependent integration tests.

#[path = "common/mod.rs"]
mod common;

#[path = "gpu/error_quality.rs"]
mod error_quality;

#[path = "gpu/tensor.rs"]
mod tensor;
