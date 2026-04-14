/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler backend v2: translates Rust syn ASTs into Tile IR using the
//! `tile-ir` crate. Emits bytecode directly — no LLVM/MLIR dependency.
//!
//! Produces `tile_ir::Module`, which is serialized to bytecode for GPU
//! compilation via the `tileiras` tool.

mod compile_assume;
mod compile_binary_op;
mod compile_block;
mod compile_cuda_tile_op;
mod compile_expression;
mod compile_inline;
mod compile_intrinsic;
mod compile_type;
mod function;
pub mod modules;
pub mod optimization_hints;
pub mod shared_types;
pub mod shared_utils;
pub mod tile_rust_type;
pub mod types;
pub mod utils;
pub(crate) mod value;

pub use function::Compiler2;
