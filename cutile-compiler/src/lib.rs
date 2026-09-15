/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT compiler that translates Rust DSL modules into Tile IR and compiles them to GPU cubins.

#![allow(non_snake_case)]

pub mod compile_api;
pub mod cuda_tile_runtime_utils;
pub mod jit_cache;

/// Frontend modules. Re-exported here for backwards compatibility.
pub use cutile_frontend;
pub use cutile_frontend::{
    ast, check_optimizations, compiler, dump, error, generics, hints, kernel_naming,
    ptr_and_literals, registry, specialization, syn_utils, train_map, type_aliases, types,
    use_classifier, utils,
};
