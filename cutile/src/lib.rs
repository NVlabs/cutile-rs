/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! # cuTile Rust
//!
//! A high-performance GPU computing library that enables you to write Rust code that compiles
//! directly to CUDA kernels. cuTile Rust combines the safety and ergonomics of Rust with the
//! performance of hand-tuned GPU code.
//!
//! ## Overview
//!
//! cuTile Rust provides a complete stack for GPU programming in Rust:
//!
//! - **Core types**: GPU tensors, tiles, and partitions for structured data access
//! - **Async execution**: Modern async/await syntax for GPU operations with automatic scheduling
//! - **Kernel compilation**: Rust → MLIR → PTX/CUBIN compilation pipeline with caching
//! - **High-level API**: Familiar NumPy-like operations for tensor creation and manipulation
//! - **Built-in kernels**: Optimized implementations of common operations (GEMM, matrix-vector, etc.)
//!
//! ## Quick Start
//!
//! ### Creating and manipulating tensors
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! // Create tensors on GPU
//! let x = api::ones::<f32>([1024]).await;
//! let y = api::zeros::<f32>([1024]).await;
//! let z = api::arange::<f32>(256).await;
//!
//! // All operations are async and execute on GPU
//! ```
//!
//! ### Writing custom GPU kernels
//!
//! ```rust,ignore
//! use cutile::prelude::*;
//!
//! #[cutile::module]
//! mod kernels {
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
//!
//! // Use the kernel with partitioned tensors
//! let x = api::ones([256]).partition([64]);
//! let y = api::ones([256]).partition([64]);
//! let z = api::zeros([256]).partition([64]);
//!
//! let result = zip!(z, x, y)
//!     .apply(kernels::vector_add_apply)
//!     .generics(vec!["f32".to_string(), "64".to_string()])
//!     .await; // Automatically launches with grid (4, 1, 1)
//! ```
//!
//! ### Async GPU pipelines
//!
//! ```rust,ignore
//! use cutile::api;
//!
//! async fn compute_pipeline() -> Tensor<f32> {
//!     // All operations execute asynchronously on GPU
//!     let x = api::randn(0.0, 1.0, [1024, 1024]).await;
//!     let y = api::randn(0.0, 1.0, [1024, 1024]).await;
//!     
//!     // Use built-in kernels
//!     use cutile::kernels::linalg::gemm_apply;
//!     
//!     let x_part = x.partition([128, 128]);
//!     let y_part = y.partition([128, 128]);
//!     let z = api::zeros([1024, 1024]).partition([128, 128]);
//!     
//!     zip!(z, x_part, y_part)
//!         .apply(gemm_apply)
//!         .generics(vec!["128".to_string(), "128".to_string(),
//!                        "32".to_string(), "1024".to_string()])
//!         .unpartition()
//!         .await
//! }
//! ```
//!
//! ## Module Organization
//!
//! - [`core`] - Core GPU types: `Tile`, `Tensor`, `Partition`, and tile operations
//! - [`api`] - High-level tensor creation and manipulation functions
//! - [`tensor`] - GPU tensor type, partitioning, and memory management
//! - [`tile_async`] - Async execution primitives and kernel compilation infrastructure
//! - [`kernels`] - Pre-built optimized kernels (GEMM, creation ops, etc.)
//!
//! ## Key Concepts
//!
//! ### Tensors and Partitions
//!
//! A [`Tensor`](tensor::Tensor) represents data on the GPU. It can be partitioned into tiles
//! that map to CUDA thread blocks:
//!
//! ```rust,ignore
//! // Create a tensor and partition it into 64-element tiles
//! let x = api::ones([256]).partition([64]); // Creates 4 partitions
//! ```
//!
//! ### Tiles
//!
//! Within kernels, you work with [`Tile`](core::Tile) values that represent data in registers
//! or shared memory. Tiles support efficient element-wise operations and broadcasting:
//!
//! ```rust,ignore
//! let tile_a = load_tile_mut(tensor_a);
//! let tile_b = load_tile_mut(tensor_b);
//! let result = tile_a * tile_b + tile_a; // All operations on GPU registers
//! ```
//!
//! ### Async Execution
//!
//! All GPU operations return futures that can be awaited. The runtime automatically manages
//! dependencies, kernel launches, and memory transfers:
//!
//! ```rust,ignore
//! // Operations compose naturally with async/await
//! let x = api::zeros([1024]).await;
//! let y = my_kernel(x).await;
//! let z = another_kernel(y).await;
//! ```
//!
//! ## Feature Flags
//!
//! - `cuda` (default): Enables runtime modules and CUDA-backed execution.
//! - Without `cuda`: Core DSL types and MLIR compilation remain available, but runtime modules are disabled.
//!
//! ## Safety
//!
//! cuTile Rust uses unsafe code internally for FFI with CUDA and for performance-critical operations.
//! The public API is designed to be safe, with compile-time guarantees about memory access patterns
//! through the type system (e.g., partition shapes must divide tensor shapes evenly).
//!
//! ## Examples
//!
//! See the `cutile-examples` crate for more comprehensive examples including:
//! - Matrix multiplication (GEMM)
//! - Async execution patterns
//! - Custom kernel development
//!
//! ## Performance
//!
//! cuTile Rust kernels can achieve performance competitive with hand-written CUDA:
//! - Zero-cost abstractions: Rust compiles to MLIR then optimized PTX
//! - Compile-time specialization: Tile shapes and types are compile-time constants
//! - Kernel caching: Compiled kernels are cached per-device for reuse
//!
//! ## Learning Resources
//!
//! For comprehensive guides and tutorials, see the [cuTile Rust Book](https://nihalpasham.github.io/cutile-book/):
//! - **User Guide** - Complete walkthrough from hello world to advanced kernels
//! - **Async Execution Deep Dive** - Understanding the async execution model
//! - **Architecture & Design** - How cutile works under the hood

pub mod _core;
#[cfg(feature = "cuda")]
pub mod error;
pub use _core::core;
#[cfg(feature = "cuda")]
pub mod api;
#[cfg(feature = "cuda")]
pub mod kernels;
#[cfg(feature = "cuda")]
pub mod tensor;
#[cfg(feature = "cuda")]
pub mod tile_kernel;
#[cfg(feature = "cuda")]
pub mod utils;

#[cfg(feature = "cuda")]
pub use cuda_async;
#[cfg(feature = "cuda")]
pub use cuda_core;
#[cfg(feature = "cuda")]
pub use cuda_core::{DType, DTypeId};
pub use cutile_compiler;
pub use cutile_macro::module;
pub use half;
pub use num_traits;
// TODO (hme): Coordinate with Candle about our dependence on this.
#[cfg(feature = "cuda")]
pub use candle_core;
#[cfg(feature = "cuda")]
pub use candle_core::{FloatDType, WithDType};
