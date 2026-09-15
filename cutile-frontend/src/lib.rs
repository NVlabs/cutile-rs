/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler frontend. Translates Rust DSL modules into Tile IR.
//!
//! The DSL is a subset of Rust, so the grammar is mostly identidcal.
//! `syn` produces the AST with [`ast`] and adds source locations for it. Then
//! [`passes`] resolves and annotates the AST, and finally [`compiler`] lowers
//! it to a [`cutile_ir::Module`].
//!
//! Lowering a [`cutile_ir::Module`]s to GPU cubins is performed by
//! `cutile-compiler`.

#![allow(non_snake_case)]

pub mod ast;
mod bounds;
pub mod check_optimizations;
pub mod error;
pub mod generics;
pub mod hints;
pub mod kernel_naming;
pub mod registry;
pub mod train_map;
pub mod types;
pub mod use_classifier;
mod value_facts;

mod kernel_entry_generator;

pub mod compiler;
pub mod dump;
pub mod passes;
pub mod specialization;

pub use compiler::utils;
pub use cutile_syn_utils::{ptr_and_literals, syn_utils, type_aliases};
