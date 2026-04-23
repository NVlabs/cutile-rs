/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT compiler that translates Rust DSL modules into Tile IR and compiles them to GPU cubins.

#![allow(non_snake_case)]
extern crate core;

pub mod ast;
mod bounds;
pub mod compile_api;
pub mod cuda_tile_runtime_utils;
pub mod error;
pub mod generics;
pub mod hints;
mod kernel_entry_generator;
pub mod kernel_naming;
pub mod syn_utils;
pub mod train_map;
pub mod types;

pub mod compiler;
pub mod dump;
pub mod passes;
pub mod specialization;
pub use compiler::utils;
