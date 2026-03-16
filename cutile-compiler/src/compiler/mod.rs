/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler backend: translates Rust syn ASTs into CUDA Tile MLIR operations.
//! Submodules handle functions, modules, types, values, and individual compilation passes.

pub mod _function;
pub mod _module;
pub mod _type;
pub mod _value;
pub mod utils;

pub mod compile_assume;
pub mod compile_binary_op;
pub mod compile_block;
pub mod compile_cuda_tile_op;
pub mod compile_expression;
pub mod compile_inline;
pub mod compile_intrinsic;
pub mod compile_type;

pub use _function::CUDATileFunctionCompiler;
pub use _module::CUDATileModules;
