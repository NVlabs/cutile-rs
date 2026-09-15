/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rank-polymorphism expansion for cuTile Rust.
//!
//! `Tile<E, const D: [i32; N]>` is rank-polymorphic which is currently not supported in Rust.
//! [`rank_instantiation`] rewrites each such item into one concrete copy per rank, and
//! [`shadow_dispatch`] crates trait scaffolding that lets rustc resolve call sites to the right
//! one. [`validate_dsl_syntax`] verifies kernel signatures.
//!
//! All of these are pure transformations. Syn in, syn out. Having this in a separate crate from
//! the cuTile dependant cutile-macro crate means this expansion can be used outside cutile-rs.

pub mod error;
pub mod rank_instantiation;
pub mod shadow_dispatch;
pub mod validate_dsl_syntax;
