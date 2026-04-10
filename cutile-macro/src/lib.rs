/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Procedural macros for the cuTile Rust GPU kernel framework.
//!
//! This crate provides the `#[module]` procedural macro that transforms Rust code into
//! GPU kernels. It handles the compilation pipeline from high-level Rust syntax to
//! CUDA-compatible code.
//!
//! ## Overview
//!
//! The `cutile-macro` crate is the compiler frontend for cuTile Rust. It performs several
//! critical transformations:
//!
//! 1. **Syntax Validation** - Ensures kernel code follows DSL restrictions
//! 2. **Variadic Expansion** - Generates specialized versions for different ranks (1D, 2D, 3D, 4D)
//! 3. **Type System Integration** - Manages compile-time shape information and type metadata
//! 4. **Launcher Generation** - Creates host-side kernel launcher functions
//! 5. **AST Construction** - Builds intermediate representation for MLIR compilation
//!
//! ## Architecture
//!
//! ### Module Processing Pipeline
//!
//! ```text
//! Rust Source Code
//!       ↓
//! [validate_dsl_syntax] ← Verify DSL restrictions
//!       ↓
//! [rewrite_variadics] ← Expand rank-polymorphic code
//!       ↓
//! [types] ← Type system and metadata
//!       ↓
//! [_module] ← Main orchestration
//!       ↓
//! [kernel_launcher_generator] ← Generate kernel launchers
//!       ↓
//! Expanded Rust + AST builders
//! ```
//!
//! ### Key Components
//!
//! - **[`_module`]** - Main entry point that orchestrates the entire transformation
//! - **[`validate_dsl_syntax`]** - Validates that kernel code follows DSL restrictions
//! - **[`rewrite_variadics`]** - Handles variadic types and generates rank-specific versions
//! - **[`types`]** - Type system including shape inference and metadata management
//! - **[`kernel_launcher_generator`]** - Generates kernel launcher functions
//!
//! ## The `#[module]` Attribute
//!
//! The primary export of this crate is the `module` procedural macro attribute:
//!
//! ```rust,ignore
//! #[cutile::module]
//! mod my_kernels {
//!     use cutile::core::*;
//!     
//!     #[cutile::entry]
//!     fn vector_add<T: ElementType, const N: i32>(
//!         z: &mut Tensor<T, {[N]}>,
//!         x: &Tensor<T, {[-1]}>,
//!         y: &Tensor<T, {[-1]}>,
//!     ) {
//!         let tile_x = load_tile_like_1d(x, z);
//!         let tile_y = load_tile_like_1d(y, z);
//!         z.store(tile_x + tile_y);
//!     }
//! }
//! ```
//!
//! The macro transforms this into:
//! - An AST builder function for MLIR compilation
//! - A direct launcher function (`vector_add`)
//! - A unified launcher that accepts both values and DeviceOps
//! - Type metadata for shape inference
//! - Proper handling of generic parameters
//!
//! ## Variadic Type System
//!
//! One of the key features is support for rank-polymorphic code through variadics.
//! A single function can be expanded to work with 1D, 2D, 3D, and 4D tensors:
//!
//! ```rust,ignore
//! #[cuda_tile::variadic_op(N=4)]
//! pub fn load_tile<E: ElementType, const S: [i32; N]>(y: &mut Tensor<E, S>) -> Tile<E, S>
//! ```
//!
//! This generates four specialized versions:
//! - `load_tile` for 1D: `const S: [i32; 1]`
//! - `load_tile` for 2D: `const S: [i32; 2]`
//! - `load_tile` for 3D: `const S: [i32; 3]`
//! - `load_tile` for 4D: `const S: [i32; 4]`
//!
//! ## Compile-Time Shape Tracking
//!
//! The macro system tracks tensor shapes at compile time, enabling:
//! - Static verification of shape compatibility
//! - Automatic inference of result shapes
//! - Optimization opportunities for the backend
//!
//! ## Safety
//!
//! The macro system enforces several safety properties:
//! - No arbitrary unsafe blocks in kernel code
//! - Restricted control flow (no early returns in some contexts)
//! - Validated memory access patterns
//! - Type-safe tensor operations
//!
//! ## Implementation Notes
//!
//! This crate makes extensive use of:
//! - `syn` for parsing Rust syntax
//! - `quote` for code generation
//! - `proc_macro2` for token manipulation
//! - Custom AST types for MLIR generation
//!
//! ## See Also
//!
//! - `cutile` crate - The runtime library and core types
//! - `cuda-tile` crate - The MLIR compiler backend

#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(non_snake_case)]

use proc_macro::TokenStream;

// Note: These modules are private because proc-macro crates can only export proc-macro functions.
// Use `cargo doc --document-private-items` to generate documentation for these modules.
mod _module;
mod error;
mod kernel_launcher_generator;
mod rewrite_variadics;
mod types;
mod validate_dsl_syntax;

/// Transforms a Rust module into GPU kernel code with kernel launchers.
///
/// This procedural macro is the main entry point for writing GPU kernels in cuTile Rust.
/// It processes a module containing kernel functions marked with `#[entry]` and generates:
///
/// - MLIR AST builder functions for compilation to CUDA
/// - Direct launcher functions for host-side execution
/// - `Unified launchers accepting `IntoDeviceOp` for each parameter
/// - Type metadata for shape inference and validation
///
/// ## Basic Usage
///
/// ```rust,ignore
/// #[cutile::module]
/// mod kernels {
///     use cutile::core::*;
///     
///     #[cutile::entry]
///     fn my_kernel<const N: i32>(data: &mut Tensor<f32, {[N]}>) {
///         let tile = data.load();
///         data.store(tile * 2.0);
///     }
/// }
///
/// // Generated: kernels::my_kernel() unified launcher (accepts IntoDeviceOp args)
/// ```
///
/// ## Attributes
///
/// - `core=true` - Marks this as a core DSL module (for `cutile::core`)
/// - `tile_rust_crate=true` - Indicates this is within the cutile crate
///
/// ## Generated Code
///
/// For each `#[entry]` function, the macro generates:
///
/// 1. **AST Builder** - `<function>_ast()` - Builds MLIR representation
/// 2. **Direct Launcher** - `<function>()` - Wraps materialized values as device operations
/// 4. **Metadata** - Type information for shape inference
///
/// ## See Also
///
/// - Main crate documentation for usage examples
/// - [`_module::module`] for implementation details
#[proc_macro_attribute]
pub fn module(attr: TokenStream, input: TokenStream) -> TokenStream {
    _module::module(attr, input)
}
