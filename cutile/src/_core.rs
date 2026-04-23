/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 Notes:
 - Any variadic macro requires a corresponding meta data entry in ::types.
   e.g. a variadic struct requires a VARIADIC_TYPES entry.
   This is required to enable desugared type inference at macro expansion time.
 - Parameter names should not be changed without careful refactoring.
   They are sometimes used to generate corresponding Rust functions for cuda_tile operations.
*/

#![allow(nonstandard_style)]
#![allow(unused_variables)]

//! Core GPU kernel programming types and operations.
//!
//! This module provides the foundational types and operations for writing GPU kernels
//! in Rust. It defines a domain-specific language (DSL) that compiles to MLIR and then
//! to optimized CUDA code.
//!
//! ## Overview
//!
//! The core module provides:
//!
//! - **Element types**: Scalar types that can be stored in tensors (`f32`, `f16`, `i32`, etc.)
//! - **Tiles**: Register/shared memory arrays that represent data being processed
//! - **Tensors**: GPU memory views with shape and stride information
//! - **Partitions**: Views that divide tensors into tiles for block-level processing
//! - **Operations**: Element-wise arithmetic, broadcasting, reshaping, matrix operations
//!
//! ## Writing GPU Kernels
//!
//! GPU kernels are written as regular Rust functions with special attributes:
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
//!         // Load tiles from input tensors based on block position
//!         let tile_x = load_tile_like_1d(x, z);
//!         let tile_y = load_tile_like_1d(y, z);
//!
//!         // Perform computation on tiles (in registers/shared memory)
//!         let result = tile_x + tile_y;
//!
//!         // Store result back to global memory
//!         z.store(result);
//!     }
//! }
//! ```
//!
//! ## Type System
//!
//! ### Element Types
//!
//! [`ElementType`] is the trait for scalar types that can be stored in tensors:
//! - Floating point: `f16`, `f32`, `f64`, `tf32`
//! - Integer: `i32`, `i64`, `u32`, `u64`
//! - Boolean: `bool` (maps to `i1`)
//!
//! ### Tiles vs Tensors
//!
//! - **[`Tile`]**: Data in registers or shared memory. Fast to access, supports element-wise operations.
//! - **[`Tensor`]**: View of data in global GPU memory. Must be loaded to tiles for processing.
//!
//! ### Shapes
//!
//! Shapes are compile-time constants represented as `const [i32; N]`:
//! - `{[128]}` - 1D with 128 elements
//! - `{[64, 64]}` - 2D with 64×64 elements
//! - `{[-1]}` - Dynamic size (inferred at runtime)
//!
//! ## Common Operations
//!
//! ### Loading and Storing
//!
//! ```rust,ignore
//! // Load entire tensor to a tile
//! let tile = load_tile_mut(&mut tensor);
//!
//! // Load from a specific position
//! let tile = load_tile_like_1d(&input_tensor, &output_tensor);
//!
//! // Store tile back to tensor
//! tensor.store(tile);
//! ```
//!
//! ### Arithmetic
//!
//! Tiles support standard arithmetic operators that work element-wise:
//!
//! ```rust,ignore
//! let a: Tile<f32, {[128]}> = ...;
//! let b: Tile<f32, {[128]}> = ...;
//!
//! let sum = a + b;
//! let product = a * b;
//! let diff = a - b;
//! let quot = a / b;
//! ```
//!
//! ### Broadcasting
//!
//! ```rust,ignore
//! // Broadcast a scalar to a tile
//! let scalar = 3.14f32;
//! let tile = scalar.broadcast(const_shape![128]);
//!
//! // Reshape a tile
//! let reshaped = tile.reshape(const_shape![8, 16]);
//! ```
//!
//! ### Matrix Operations
//!
//! ```rust,ignore
//! // Matrix multiplication using hardware accelerated MMA
//! let c = mma(a, b, acc); // c = a * b + acc
//! ```
//!
//! ## Partitions
//!
//! [`Partition`] views divide tensors into tiles for parallel processing:
//!
//! ```rust,ignore
//! let tensor_shape = tensor.shape();
//! let partition = tensor.partition(const_shape![64, 64]);
//!
//! // Each thread block loads its partition based on block ID
//! let pid = get_tile_block_id();
//! let tile = partition.load([pid.0, pid.1]);
//! ```
//!
//! ## Block Operations
//!
//! Get information about the current thread block:
//!
//! ```rust,ignore
//! let (x, y, z) = get_tile_block_id();  // Current block position
//! let (gx, gy, gz) = get_num_tile_blocks();  // Total number of blocks
//! ```
//!
//! ## Debugging
//!
//! Print from GPU kernels (for debugging):
//!
//! ```rust,ignore
//! cuda_tile_print!("Block ID: {}, Value: {}\n", pid.0, value);
//! cuda_tile_assert!(value > 0, "Value must be positive");
//! ```
//!
//! ## Advanced: Type Conversions
//!
//! ```rust,ignore
//! // Convert between element types
//! let f32_tile: Tile<f32, {[128]}> = convert_tile(i32_tile);
//!
//! // Convert scalar to tile and back
//! let tile = scalar_to_tile(3.14f32);
//! let scalar: f32 = tile_to_scalar(tile);
//! ```
//!
//! ## Safety and Correctness
//!
//! - Bounds checking can be enabled with `check_partition_access()`
//! - Type system ensures shape compatibility at compile time
//! - Undefined behavior if tensor shapes don't match partition shapes at runtime
//!
//! ## See Also
//!
//! - [`tile_async`](crate::tile_async) - Async execution and kernel compilation
//! - [`kernels`](crate::kernels) - Pre-built kernel examples

// ---------------------------------------------------------------
// Static operation parameter modules
//
// Defined outside the proc-macro-processed `core` module because the
// module macro doesn't support nested `mod` items. Re-exported from
// `core` via `pub use`.
//
// Each module defines a trait (`Mode`) and zero-sized marker structs.
// Binary switches use Enabled/Disabled. Multi-valued parameters use
// descriptive names.
// ---------------------------------------------------------------

/// Flush-to-zero modifier. Flushes denormal inputs and results to
/// sign-preserving zero. Only supported for f32.
pub mod ftz {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Rounding mode for floating-point operations.
pub mod rounding {
    pub trait Mode {}
    pub struct NearestEven;
    pub struct PositiveInf;
    pub struct NegativeInf;
    pub struct Zero;
    pub struct Approx;
    impl Mode for NearestEven {}
    impl Mode for PositiveInf {}
    impl Mode for NegativeInf {}
    impl Mode for Zero {}
    impl Mode for Approx {}
}

/// NaN propagation for maxf/minf operations.
pub mod nan {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

#[cutile_macro::module(core = true, tile_rust_crate = true)]
pub mod core {

    pub use super::ftz;
    pub use super::nan;
    pub use super::rounding;
    pub use half::{bf16, f16};
    use std::marker::PhantomData;
    use std::ops;

    /// Marker trait for types that can be stored as tensor elements.
    ///
    /// This trait is implemented for scalar types that have a corresponding
    /// GPU representation. Use these types as the element type `T` in [`Tile<T, S>`]
    /// and [`Tensor<T, S>`].
    ///
    /// ## Supported Types
    ///
    /// - **Floating point**: `bf16`, `f16`, `f32`, `f64`, `tf32`
    /// - **Signed integers**: `i32`, `i64`
    /// - **Unsigned integers**: `u32` (mapped to `i32`), `u64` (mapped to `i64`)
    /// - **Boolean**: `bool` (mapped to `i1`)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// fn my_kernel<T: ElementType>(x: &Tensor<T, {[128]}>) {
    ///     // T can be any ElementType
    /// }
    /// ```
    pub trait ElementType: Copy + Clone {
        const ZERO: Self;
    }
    #[cuda_tile::ty(name = "bf16")]
    impl ElementType for bf16 {
        const ZERO: Self = bf16::ZERO;
    }
    #[cuda_tile::ty(name = "f16")]
    impl ElementType for f16 {
        const ZERO: Self = f16::ZERO;
    }
    #[cuda_tile::ty(name = "f32")]
    impl ElementType for f32 {
        const ZERO: Self = 0.0;
    }
    #[cuda_tile::ty(name = "i8")]
    impl ElementType for i8 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i8")]
    impl ElementType for u8 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i32")]
    impl ElementType for i32 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i32")]
    impl ElementType for u32 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i64")]
    impl ElementType for i64 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i64")]
    impl ElementType for u64 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "f64")]
    impl ElementType for f64 {
        const ZERO: Self = 0.0;
    }
    #[cuda_tile::ty(name = "i16")]
    impl ElementType for i16 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i16")]
    impl ElementType for u16 {
        const ZERO: Self = 0;
    }
    #[cuda_tile::ty(name = "i1")]
    impl ElementType for bool {
        const ZERO: Self = false;
    }

    // GPU-specific types: re-exported from cuda-core.
    pub use cuda_core::f8e4m3fn;
    pub use cuda_core::f8e5m2;
    pub use cuda_core::tf32;

    #[cuda_tile::ty(name = "tf32")]
    impl ElementType for tf32 {
        const ZERO: Self = tf32(0);
    }
    #[cuda_tile::ty(name = "f8e4m3fn")]
    impl ElementType for f8e4m3fn {
        const ZERO: Self = f8e4m3fn(0);
    }
    #[cuda_tile::ty(name = "f8e5m2")]
    impl ElementType for f8e5m2 {
        const ZERO: Self = f8e5m2(0);
    }

    /// Marker trait for scalar values that can be broadcast to tiles.
    ///
    /// This trait is automatically implemented for all [`ElementType`]s,
    /// allowing them to be used with broadcasting operations.
    pub trait Scalar {}
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["E"])]
    impl<E: ElementType> Scalar for E {}

    /// Converts a scalar value to a 0-dimensional tile.
    ///
    /// This is used internally for type conversions between scalar and tile representations.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn scalar_to_tile<E: ElementType>(scalar: impl Scalar) -> Tile<E, { [] }> {
        unreachable!()
    }

    /// Converts a 0-dimensional tile back to a scalar value.
    ///
    /// This is the inverse of `scalar_to_tile`.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn tile_to_scalar<E: ElementType, S: Scalar>(tile: Tile<E, { [] }>) -> S {
        unreachable!()
    }

    /// Converts a scalar from one type to another.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let i: i32 = 42;
    /// let f: f32 = convert_scalar(i); // 42.0
    /// ```
    #[cuda_tile::compiler_op(name = "convert")]
    pub fn convert_scalar<S: Scalar>(x: impl Scalar) -> S {
        unreachable!()
    }

    /// Converts all elements of a tile from one type to another.
    ///
    /// Performs element-wise type conversion, preserving the shape of the tile.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let int_tile: Tile<i32, {[128]}> = ...;
    /// let float_tile: Tile<f32, {[128]}> = convert_tile(int_tile);
    /// ```
    #[cuda_tile::compiler_op(name = "convert")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn convert_tile<TO: ElementType, FROM: ElementType, const S: [i32; N]>(
        x: Tile<FROM, S>,
    ) -> Tile<TO, S> {
        unreachable!()
    }

    /// Checks that a partition access is within bounds.
    ///
    /// This function can be used to add runtime bounds checking to partition loads.
    /// If the compiler can prove the access is safe, no code is emitted. Otherwise,
    /// an assertion is generated.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let partition = tensor.partition(const_shape![64, 64]);
    /// let index = [pid.0, pid.1];
    /// check_partition_access(&partition, index);
    /// let tile = partition.load(index);
    /// ```
    #[cuda_tile::compiler_op(name = "check")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn check_partition_access<E: ElementType, const S: [i32; N]>(
        part: &Partition<E, S>,
        index: [i32; N],
    ) {
        // This is either instantiated, in which case an actual bounds check takes place,
        // or the check is performed statically and nothing is emitted.
        // The bounds check is implemented as an assertion.
        // The check is emitted with as many static values as possible, providing the TileIR
        // compiler an opportunity to perform loop invariant code motion.
        unreachable!()
    }

    /// Trait for broadcasting scalar element types to tiles.
    ///
    /// This trait provides the `broadcast` method that converts a scalar value
    /// into a tile filled with that value. Automatically implemented for all
    /// [`ElementType`]s.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let scalar = 3.14f32;
    /// let tile = scalar.broadcast(const_shape![128]); // Tile of 128 copies of 3.14
    /// ```
    // This generates the trait impl N times, varying the trait and method name.
    #[cuda_tile::variadic_trait(N = 6)]
    pub trait BroadcastScalar<E: ElementType, const D: [i32; N]>
    where
        Self: ElementType,
    {
        fn broadcast(self, shape: Shape<D>) -> Tile<E, D>;
    }

    #[cuda_tile::variadic_trait_impl()]
    // This implements the trait impl N times, varying the trait and method name.
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> BroadcastScalar<E, D> for E {
        fn broadcast(self, shape: Shape<D>) -> Tile<E, D> {
            broadcast_scalar(self, shape)
        }
    }

    /// Marker trait for GPU pointer types.
    ///
    /// Implemented for mutable raw pointers to element types. Used for low-level
    /// pointer operations within kernels.
    pub trait Pointer {}
    #[cuda_tile::ty(name="!cuda_tile.tile", pointer_type="!cuda_tile.ptr", type_params=["!cuda_tile.ptr<E>"])]
    impl<E: ElementType> Pointer for *mut E {}
    // impl<E: ElementType> Pointer for *const E {}

    /// Converts a raw pointer to a pointer tile.
    ///
    /// This is used internally for pointer arithmetic and indexing operations.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn pointer_to_tile<P: Pointer>(ptr: P) -> PointerTile<P, { [] }> {
        unreachable!()
    }

    /// Converts a pointer tile back to a raw pointer.
    ///
    /// This is the inverse of `pointer_to_tile`.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn tile_to_pointer<P: Pointer>(tile: PointerTile<P, { [] }>) -> P {
        unreachable!()
    }

    /// Prints formatted output from within a GPU kernel.
    ///
    /// This macro provides printf-style debugging from GPU code. It's similar to Rust's
    /// `print!` macro but works inside CUDA kernels.
    ///
    /// ## Format String
    ///
    /// The format string uses C-style format specifiers:
    /// - `{}` is automatically converted to `%` by the compiler
    /// - Use `\n` for newlines
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// #[cutile::entry]
    /// fn debug_kernel() {
    ///     let pid = get_tile_block_id();
    ///     cuda_tile_print!("Block ID: {}, {}, {}\n", pid.0, pid.1, pid.2);
    ///
    ///     let value = 42;
    ///     cuda_tile_print!("Value: {}\n", value);
    /// }
    /// ```
    ///
    /// ## Note
    ///
    /// Printing from GPU kernels can significantly impact performance and should be used
    /// for debugging only.
    // {} is converted to % by the compiler.
    #[macro_export]
    macro_rules! cuda_tile_print {
        ($s:literal $(,$args:expr)*) => {
            unreachable!();
        };
    }
    pub use cuda_tile_print;

    /// Asserts a condition is true within a GPU kernel.
    ///
    /// If the assertion fails, the kernel will terminate with the specified error message.
    /// This is useful for runtime validation of assumptions in kernel code.
    ///
    /// ## Parameters
    ///
    /// - First argument: The condition to check (must evaluate to `bool`)
    /// - Second argument: Error message string literal
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// #[cutile::entry]
    /// fn safe_kernel(x: &Tensor<f32, {[128]}>) {
    ///     let value = compute_something();
    ///     cuda_tile_assert!(value > 0, "Value must be positive");
    ///
    ///     let idx = get_index();
    ///     cuda_tile_assert!(idx < 128, "Index out of bounds");
    /// }
    /// ```
    ///
    /// ## Note
    ///
    /// Like CPU assertions, these have a runtime cost and should be used judiciously
    /// in performance-critical code.
    #[macro_export]
    macro_rules! cuda_tile_assert {
        ($args:expr, $s:literal) => {
            unreachable!();
        };
    }
    pub use cuda_tile_assert;

    /// Returns the total number of thread blocks in the grid.
    ///
    /// This returns the grid dimensions `(gridDim.x, gridDim.y, gridDim.z)` in CUDA terms.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let (gx, gy, gz) = get_num_tile_blocks();
    /// cuda_tile_print!("Grid size: {} x {} x {}\n", gx, gy, gz);
    /// ```
    #[cuda_tile::op(name="cuda_tile.get_num_tile_blocks", params=[])]
    pub fn get_num_tile_blocks() -> (i32, i32, i32) {
        unreachable!()
    }

    /// Returns the current thread block's position in the grid.
    ///
    /// This returns `(blockIdx.x, blockIdx.y, blockIdx.z)` in CUDA terms. Each block
    /// processes a different partition of the data.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let (bx, by, bz) = get_tile_block_id();
    /// // Use block ID to determine which partition to process
    /// let tile = partition.load([bx, by]);
    /// ```
    #[cuda_tile::op(name="cuda_tile.get_tile_block_id", params=[])]
    pub fn get_tile_block_id() -> (i32, i32, i32) {
        unreachable!()
    }

    /* Shape */

    /// A compile-time shape descriptor for tensors and tiles.
    ///
    /// `Shape` represents the dimensions of a multi-dimensional array. The shape can
    /// contain compile-time constants or runtime values.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Create constant shapes using the const_shape! macro
    /// let shape = const_shape![128, 64];
    ///
    /// // Get shape from a tensor
    /// let tensor_shape = tensor.shape();
    /// ```
    #[cuda_tile::variadic_struct(N = 6, constructor = "new")]
    #[derive(Copy, Clone)]
    pub struct Shape<'a, const D: [i32; N]> {
        pub dims: &'a [i32],
    }

    /// Creates a compile-time constant shape.
    ///
    /// This macro constructs a [`Shape`] with compile-time known dimensions. The shape
    /// can be 0D to 4D.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Scalar (0D)
    /// let scalar_shape = const_shape![];
    ///
    /// // 1D shape
    /// let vec_shape = const_shape![128];
    ///
    /// // 2D shape
    /// let matrix_shape = const_shape![64, 128];
    ///
    /// // 3D shape
    /// let volume_shape = const_shape![32, 64, 128];
    ///
    /// // Use in kernel code
    /// let partition = tensor.partition(const_shape![64, 64]);
    /// ```
    #[macro_export]
    macro_rules! const_shape {
        () => {
            Shape_0::const_new()
        };
        ($x1:literal) => {
            Shape_1::<$x1>::const_new()
        };
        ($x1:literal, $x2:literal) => {
            Shape_2::<$x1, $x2>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal) => {
            Shape_3::<$x1, $x2, $x3>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal, $x4:literal) => {
            Shape_4::<$x1, $x2, $x3, $x4>::const_new()
        };
    }
    pub use const_shape;

    /* Array */

    /// A compile-time array descriptor for indexing and other metadata.
    ///
    /// Similar to [`Shape`] but used for permutation indices and other array-valued metadata.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Create constant arrays using the const_array! macro
    /// let arr = const_array![0, 2, 1]; // For permutation [0, 2, 1]
    /// ```
    #[cuda_tile::variadic_struct(N = 6, constructor = "new")]
    #[derive(Copy, Clone)]
    pub struct Array<'a, const D: [i32; N]> {
        pub dims: &'a [i32],
    }

    /// Creates a compile-time constant array.
    ///
    /// This macro constructs an [`Array`] with compile-time known values. Arrays are
    /// typically used for permutation indices and other metadata.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Empty array (0D)
    /// let empty = const_array![];
    ///
    /// // 1D array
    /// let idx = const_array![0];
    ///
    /// // 2D permutation: transpose (swap dimensions 0 and 1)
    /// let transpose = const_array![1, 0];
    ///
    /// // 3D permutation: move last dimension to front
    /// let perm = const_array![2, 0, 1];
    ///
    /// // Use with permute operation
    /// let transposed = permute(tile, const_array![1, 0]);
    /// ```
    #[macro_export]
    macro_rules! const_array {
        () => {
            Array_0::const_new()
        };
        ($x1:literal) => {
            Array_1::<$x1>::const_new()
        };
        ($x1:literal, $x2:literal) => {
            Array_2::<$x1, $x2>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal) => {
            Array_3::<$x1, $x2, $x3>::const_new()
        };
        ($x1:literal, $x2:literal, $x3:literal, $x4:literal) => {
            Array_4::<$x1, $x2, $x3, $x4>::const_new()
        };
    }
    pub use const_array;

    /* PointerTile */

    /// A tile of pointers for advanced memory operations.
    ///
    /// `PointerTile` represents a multi-dimensional array of pointers, enabling
    /// gather/scatter operations and indirect memory access patterns. This is a
    /// low-level primitive rarely used directly in typical kernels.
    ///
    /// ## Type Parameters
    ///
    /// - `P`: Pointer type (must implement [`Pointer`])
    /// - `D`: Shape of the pointer tile
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xP"])]
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct PointerTile<P: Pointer, const D: [i32; N]> {
        _type: PhantomData<P>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<P: Pointer, const D: [i32; N]> PointerTile<P, D> {
        /// Offsets each pointer in the tile by the corresponding offset value.
        ///
        /// Performs element-wise pointer arithmetic using a tile of offsets.
        pub fn offset_tile<I: ElementType>(self, offset: Tile<I, D>) -> PointerTile<P, D> {
            addptr_tile(self, offset)
        }

        /// Offsets all pointers in the tile by a scalar offset.
        pub fn offset(self, offset: i32) -> PointerTile<P, D> {
            addptr(self, offset)
        }

        /// Broadcasts this pointer tile to a new shape.
        ///
        /// Dimensions of size 1 in the source can be broadcast to any size in the result.
        pub fn broadcast<const R: [i32; N]>(self, shape: Shape<R>) -> PointerTile<P, R> {
            broadcast_ptr(self, shape)
        }

        /// Reshapes this pointer tile to a new shape without moving data.
        ///
        /// The total number of elements must remain the same.
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn reshape<const R: [i32; M]>(self, shape: Shape<R>) -> PointerTile<P, R> {
            reshape_ptr(self, shape)
        }
    }

    /// Adds a scalar offset to all pointers in a pointer tile.
    ///
    /// This is pointer arithmetic: `result[i] = ptr[i] + offset`.
    #[cuda_tile::op(name="cuda_tile.offset", params=["ptr", "offset"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addptr<P: Pointer, const D: [i32; N]>(
        ptr: PointerTile<P, D>,
        offset: i32,
    ) -> PointerTile<P, D> {
        unreachable!()
    }

    /// Adds element-wise offsets to a pointer tile.
    ///
    /// Each pointer is offset by the corresponding value in the offset tile:
    /// `result[i] = ptr[i] + offset[i]`.
    #[cuda_tile::op(name="cuda_tile.offset", params=["ptr", "offset"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addptr_tile<I: ElementType, P: Pointer, const D: [i32; N]>(
        ptr: PointerTile<P, D>,
        offset: Tile<I, D>,
    ) -> PointerTile<P, D> {
        unreachable!()
    }

    /// Broadcasts a pointer tile to a new shape.
    ///
    /// Dimensions of size 1 in the source can be broadcast to any size in the result.
    #[cuda_tile::op(name="cuda_tile.broadcast", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn broadcast_ptr<P: Pointer, const S: [i32; N], const R: [i32; N]>(
        source: PointerTile<P, S>,
        shape: Shape<R>,
    ) -> PointerTile<P, R> {
        unreachable!()
    }

    /// Reshapes a pointer tile to a new shape without moving data.
    ///
    /// The total number of elements must remain the same.
    #[cuda_tile::op(name="cuda_tile.reshape", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reshape_ptr<P: Pointer, const S: [i32; N], const R: [i32; M]>(
        source: PointerTile<P, S>,
        shape: Shape<R>,
    ) -> PointerTile<P, R> {
        unreachable!()
    }

    /* Token */

    /// A token representing memory operation ordering constraints.
    ///
    /// Tokens are used internally to track dependencies between memory operations
    /// and ensure correct ordering of loads and stores. The tile DSL automatically
    /// manages tokens to maintain memory consistency.
    ///
    /// ## Note
    ///
    /// Users typically don't need to manipulate tokens directly; they're handled
    /// automatically by the load/store operations and partition views.
    #[cuda_tile::ty(name="!cuda_tile.token", params=[])]
    #[derive(Copy, Clone)]
    pub struct Token {}

    /// Creates a new unordered token.
    ///
    /// This is used internally to initialize token state. Most users won't need
    /// to call this directly.
    #[cuda_tile::op(name="cuda_tile.make_token", params=[])]
    pub fn new_token_unordered() -> Token {
        unreachable!()
    }

    /// Joins multiple synchronization tokens into a single token.
    ///
    /// This operation combines multiple tokens that represent independent async operations
    /// into a single token that represents the completion of all of them. This is useful
    /// when you have parallel loads or operations that need to complete before proceeding.
    ///
    /// ## Parameters
    ///
    /// - `tokens`: Array/slice of tokens to join
    ///
    /// ## Returns
    ///
    /// A new token that depends on all input tokens
    ///
    /// ## Examples
    ///
    /// ### Parallel loads with join
    ///
    /// ```rust,ignore
    /// let token = new_token_unordered();
    ///
    /// // Load two tiles in parallel
    /// let (tile_a, token_a) = load_from_view_mut(&partition_a, [0], token);
    /// let (tile_b, token_b) = load_from_view_mut(&partition_b, [0], token);
    ///
    /// // Wait for both loads to complete
    /// let combined = join_tokens(&[token_a, token_b]);
    ///
    /// // Now safe to proceed with both tiles
    /// let result = tile_a + tile_b;
    /// store_to_view_mut(&partition_out, result, [0], combined);
    /// ```
    #[cuda_tile::op(name="cuda_tile.join_tokens", params=["tokens"])]
    pub fn join_tokens(tokens: &[Token]) -> Token {
        unreachable!()
    }

    // ============================================================================
    // ATOMIC OPERATIONS
    // ============================================================================
    // The following functions provide atomic read-modify-write (RMW) and compare-and-swap
    // (CAS) operations on global memory with token ordering semantics.
    //
    // DESIGN:
    // - Generic atomic_rmw_tko() handles all RMW modes (hidden from users)
    // - Wrapper functions (atomic_add_tko, atomic_and_tko, etc.) provide type-safe APIs
    // - Mode, memory_ordering, and memory_scope passed as string literals
    // - Compiler converts string literals to integer MLIR attributes (not symbol refs)
    // - Optional mask and token parameters use Option<T> (not blank strings)
    //
    // MEMORY ORDERING:
    // - "relaxed": No synchronization, concurrent access allowed
    // - "acquire": Synchronizes with release, establishes happens-before for reads
    // - "release": Synchronizes with acquire, establishes happens-before for writes
    // - "acq_rel": Combined acquire+release semantics
    //
    // MEMORY SCOPE:
    // - "tl_blk": Tile block scope (same CTA)
    // - "device": GPU device scope (all CTAs on same GPU)
    // - "sys": System scope (all devices including CPU)
    //
    // SUPPORTED TYPES:
    // - Integer operations (and, or, xor, add, max, min, umax, umin): i32, i64
    // - Float operations (addf): f16, f32, f64
    // - Exchange (xchg): i32, i64, f32, f64
    // - Compare-and-swap (CAS): i32, i64, f32, f64
    // ============================================================================

    /// Internal generic implementation for atomic read-modify-write operations with token ordering.
    ///
    /// Performs element-wise atomic read-modify-write operations on global memory.
    /// This function is not directly exposed - use the mode-specific wrapper functions instead
    /// (e.g., `atomic_and_tko`, `atomic_add_tko`, `atomic_max_tko`, etc.).
    ///
    /// ## Parameters
    ///
    /// **Operands (values passed to MLIR operation):**
    /// - `pointers`: Tile of memory addresses to operate on (required)
    /// - `arg`: Tile of values to use in the atomic operation (required)
    /// - `mask`: Optional mask tile (`Tile<bool, S>`) to selectively perform operations
    /// - `token`: Optional input token for ordering
    ///
    /// **Attributes (metadata/configuration):**
    /// - `mode`: Atomic operation mode (string literal)
    ///   - `"and"`: Bitwise AND (integer types: i32, i64)
    ///   - `"or"`: Bitwise OR (integer types: i32, i64)
    ///   - `"xor"`: Bitwise XOR (integer types: i32, i64)
    ///   - `"add"`: Integer addition (integer types: i32, i64)
    ///   - `"addf"`: Floating-point addition (float types: f16, f32, f64)
    ///   - `"max"`: Signed maximum (integer types: i32, i64)
    ///   - `"min"`: Signed minimum (integer types: i32, i64)
    ///   - `"umax"`: Unsigned maximum (integer types: i32, i64)
    ///   - `"umin"`: Unsigned minimum (integer types: i32, i64)
    ///   - `"xchg"`: Exchange/swap (integer types: i32, i64; float types: f32, f64)
    /// - `memory_ordering`: Memory ordering semantics (string literal)
    ///   - `"relaxed"`: Concurrent access allowed, no synchronization
    ///   - `"acquire"`: Read establishes happens-before if observing a release
    ///   - `"release"`: Write establishes happens-before for observers with acquire
    ///   - `"acq_rel"`: Combined acquire and release semantics
    /// - `memory_scope`: Memory scope for synchronization (string literal)
    ///   - `"tl_blk"`: Tile block scope (same CTA)
    ///   - `"device"`: GPU device scope (all CTAs on same GPU)
    ///   - `"sys"`: System scope (all devices including CPU)
    ///
    /// **Note:** `mode`, `memory_ordering`, and `memory_scope` are passed as function parameters in the
    /// Rust API, but are compiled to MLIR attributes (not operands) by the compiler.
    ///
    /// **Design Note:** This function can be called from wrapper functions (e.g., `atomic_and_tko`)
    /// which pass variables instead of string literals. The compiler automatically resolves these
    /// variables back to their original string literal AST expressions using a parameter-to-AST mapping.
    ///
    /// ## Mode and Element Type Compatibility
    ///
    /// The `mode` parameter must be compatible with the element type `E`:
    /// - Integer modes (`and`, `or`, `xor`, `add`, `max`, `min`, `umax`, `umin`) require integer types
    /// - Floating-point mode (`addf`) requires floating-point types
    /// - Exchange mode (`xchg`) works with both integer and floating-point types
    ///
    /// The compiler validates this at compile-time and will panic with a descriptive error if
    /// an incompatible mode is used with a given element type.
    ///
    /// ## Memory Semantics
    ///
    /// The `memory_ordering` parameter controls visibility and synchronization:
    /// - `relaxed`: Allows concurrent access but provides no ordering guarantees
    /// - `acquire`: Synchronizes with release operations, establishing happens-before for reads
    /// - `release`: Synchronizes with acquire operations, establishing happens-before for writes
    /// - `acq_rel`: Combined acquire and release semantics (for read-modify-write operations)
    ///
    /// The `memory_scope` parameter defines the range of threads that may observe the operation.
    ///
    /// ## Examples
    ///
    /// This function is typically called indirectly through wrapper functions:
    /// ```rust,ignore
    /// // Use atomic_and_tko instead of calling atomic_rmw_tko directly
    /// let (old_values, token) = atomic_and_tko(
    ///     ptrs,
    ///     values,
    ///     "relaxed",
    ///     "device",
    ///     None,
    ///     None
    /// );
    /// ```
    ///
    /// ## Returns
    ///
    /// Returns a tuple of (old_values, result_token) where:
    /// - `old_values`: Tile containing the values that were in memory before the atomic operation
    /// - `result_token`: Token representing completion of the atomic operation
    #[doc(hidden)]
    #[cuda_tile::op(name="cuda_tile.atomic_rmw_tko", params=["pointers", "arg"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn atomic_rmw_tko<E: ElementType, const S: [i32; N]>(
        pointers: PointerTile<*mut E, S>,
        arg: Tile<E, S>,
        mode: &str,
        memory_ordering: &str,
        memory_scope: &str,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Atomic compare-and-swap operation with token ordering.
    ///
    /// Atomically compares the value at each memory location with a comparison value.
    /// If they match, replaces the memory value with a new value. This is the fundamental
    /// building block for lock-free algorithms.
    ///
    /// ## Parameters
    ///
    /// **Operands:**
    /// - `pointers`: Tile of memory addresses to operate on (required)
    /// - `cmp`: Tile of comparison values (expected old values) (required)
    /// - `val`: Tile of new values to write if comparison succeeds (required)
    /// - `mask`: Optional mask tile (`Tile<bool, S>`) to selectively perform operations
    /// - `token`: Optional input token for ordering
    ///
    /// **Attributes:**
    /// - `memory_ordering`: Memory ordering semantics (`"relaxed"`, `"acquire"`, `"release"`, `"acq_rel"`)
    /// - `memory_scope`: Memory scope (`"tl_blk"`, `"device"`, `"sys"`)
    ///
    /// ## Semantics
    ///
    /// For each element i:
    /// ```text
    /// old = *pointers[i]
    /// if (old == cmp[i]) {
    ///     *pointers[i] = val[i]
    /// }
    /// return old
    /// ```
    ///
    /// Masked-out elements return `cmp[i]` value instead of performing the atomic operation.
    ///
    /// ## Memory Semantics
    ///
    /// The `memory_ordering` parameter controls visibility and synchronization:
    /// - `relaxed`: Allows concurrent access but provides no ordering guarantees
    /// - `acquire`: Synchronizes with release operations, establishing happens-before for reads
    /// - `release`: Synchronizes with acquire operations, establishing happens-before for writes
    /// - `acq_rel`: Combined acquire and release semantics (for read-modify-write operations)
    ///
    /// The `memory_scope` parameter defines the range of threads that may observe the operation.
    ///
    /// ## Supported Types
    ///
    /// - Integer types: i32, i64
    /// - Floating-point types: f32, f64
    ///
    /// For floating-point types, the comparison uses bitwise equality rather than IEEE-754 semantics.
    /// This means different NaN bit patterns are treated as distinct values, and +0.0 and -0.0
    /// are considered different if their bit patterns differ.
    ///
    /// ## Examples
    ///
    /// ### Lock-free update
    ///
    /// ```rust,ignore
    /// // Try to update value from expected to new
    /// let (old_values, new_token) = atomic_cas_tko(
    ///     ptr_tile,
    ///     expected_values,
    ///     new_values,
    ///     "relaxed",
    ///     "device",
    ///     None,
    ///     None
    /// );
    /// // Check old_values to see if CAS succeeded
    /// ```
    ///
    /// ## Returns
    ///
    /// Returns a tuple of (old_values, result_token) where old_values contains the
    /// values that were in memory before the operation (regardless of success/failure).
    #[cuda_tile::op(name="cuda_tile.atomic_cas_tko", params=["pointers", "cmp", "val"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn atomic_cas_tko<E: ElementType, const S: [i32; N]>(
        pointers: PointerTile<*mut E, S>,
        cmp: Tile<E, S>,
        val: Tile<E, S>,
        memory_ordering: &str,
        memory_scope: &str,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Load from pointer tile with token ordering and full memory semantics control.
    ///
    /// Performs a gather operation by loading a tile of data from global memory into
    /// a result tile based on a tile of pointers.
    ///
    /// ## Parameters
    ///
    /// **Operands (values passed to MLIR operation):**
    /// - `source`: Tile of pointers to load from (required)
    /// - `mask`: Optional mask tile (`Tile<bool, S>`) to selectively load elements
    /// - `padding_value`: Optional scalar value (`E`) to use for masked-out elements
    /// - `token`: Optional input token for ordering
    ///
    /// **Attributes (metadata/configuration):**
    /// - `memory_ordering`: Memory ordering semantics (string literal)
    ///   - `"weak"`: No concurrent accesses (compiler can assume exclusive access)
    ///   - `"relaxed"`: Concurrent access allowed, no synchronization
    ///   - `"acquire"`: Load establishes happens-before if observing a release
    /// - `memory_scope`: Memory scope for synchronization (string literal)
    ///   - `"tl_blk"`: Tile block scope (same CTA)
    ///   - `"device"`: GPU device scope (all CTAs on same GPU)
    ///   - `"sys"`: System scope (all devices including CPU)
    /// - `latency`: Optional latency hint for the compiler
    ///
    /// **Note:** When `memory_ordering` is `"weak"`, `memory_scope` is not included as an attribute
    /// (per TileIR spec).
    ///
    /// **Design Note:** `memory_ordering` and `memory_scope` are passed as function parameters in the
    /// Rust API, but are compiled to MLIR attributes (not operands) by the compiler.
    ///
    /// ## Scalar-to-Tile Promotion
    ///
    /// The `padding_value` parameter accepts a scalar type (`E`, e.g., `0.0f32`). The compiler
    /// automatically promotes it to a tile matching the result shape:
    /// 1. Scalar value → Reshape to `tile<1xT>`
    /// 2. `tile<1xT>` → Broadcast to `tile<NxT>` (matching result shape)
    ///
    /// This allows using simple scalar values like `0.0f32` instead of creating a full tile.
    ///
    /// ## Memory Semantics
    ///
    /// The `memory_ordering` parameter controls visibility and synchronization:
    /// - `weak`: Assumes no concurrent access (fastest, but unsafe if concurrent access exists)
    /// - `relaxed`: Allows concurrent access but provides no ordering guarantees
    /// - `acquire`: Synchronizes with release operations, establishing happens-before
    ///
    /// The `memory_scope` parameter defines the range of threads that may observe the operation.
    ///
    /// ## Examples
    ///
    /// Basic load with default semantics:
    /// ```rust,ignore
    /// let ptrs: PointerTile<*mut f32, {[128]}> = ...;
    /// let (values, token) = load_ptr_tko(ptrs, "relaxed", "device", None, None, None, None);
    /// ```
    ///
    /// Load with acquire semantics and input token:
    /// ```rust,ignore
    /// let (values, token) = load_ptr_tko(ptrs, "acquire", "device", None, None, Some(input_token), None);
    /// ```
    ///
    /// Load with mask and padding value (scalar automatically promoted to tile):
    /// ```rust,ignore
    /// let mask: Tile<bool, {[128]}> = ...;
    /// let padding = 0.0f32;  // Scalar - compiler promotes to tile<128xf32>
    /// let (values, token) = load_ptr_tko(ptrs, "relaxed", "device", Some(mask), Some(padding), None, None);
    /// ```
    ///
    /// ## Returns
    ///
    /// Returns a tuple of (loaded_values, result_token) where:
    /// - `loaded_values`: Tile containing the loaded data
    /// - `result_token`: Token representing completion of the load operation
    #[cuda_tile::op(name="cuda_tile.load_ptr_tko", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_ptr_tko<E: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut E, S>,
        memory_ordering: &str,
        memory_scope: &str,
        mask: Option<Tile<bool, S>>,
        padding_value: Option<E>,
        token: Option<Token>,
        latency: Option<i32>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Store to pointer tile with token ordering and full memory semantics control.
    ///
    /// Performs a scatter operation by storing a tile of data to global memory addresses
    /// specified by a tile of pointers.
    ///
    /// ## Parameters
    ///
    /// **Operands (values passed to MLIR operation):**
    /// - `destination`: Tile of pointers to store to (required)
    /// - `value`: Tile of values to store (required)
    /// - `mask`: Optional mask tile (`Tile<bool, S>`) to selectively store elements
    /// - `token`: Optional input token for ordering
    ///
    /// **Attributes (metadata/configuration):**
    /// - `memory_ordering`: Memory ordering semantics (string literal)
    ///   - `"weak"`: No concurrent accesses (compiler can assume exclusive access)
    ///   - `"relaxed"`: Concurrent access allowed, no synchronization
    ///   - `"release"`: Store establishes happens-before for observers with acquire
    /// - `memory_scope`: Memory scope for synchronization (string literal)
    ///   - `"tl_blk"`: Tile block scope (same CTA)
    ///   - `"device"`: GPU device scope (all CTAs on same GPU)
    ///   - `"sys"`: System scope (all devices including CPU)
    /// - `latency`: Optional latency hint for the compiler
    ///
    /// **Note:** When `memory_ordering` is `"weak"`, `memory_scope` is not included as an attribute
    /// (per TileIR spec).
    ///
    /// **Design Note:** `memory_ordering` and `memory_scope` are passed as function parameters in the
    /// Rust API, but are compiled to MLIR attributes (not operands) by the compiler.
    ///
    /// ## Memory Semantics
    ///
    /// The `memory_ordering` parameter controls visibility and synchronization:
    /// - `weak`: Assumes no concurrent access (fastest, but unsafe if concurrent access exists)
    /// - `relaxed`: Allows concurrent access but provides no ordering guarantees
    /// - `release`: Synchronizes with acquire operations, establishing happens-before
    ///
    /// The `memory_scope` parameter defines the range of threads that may observe the operation.
    ///
    /// ## Examples
    ///
    /// Basic store with default semantics:
    /// ```rust,ignore
    /// let ptrs: PointerTile<*mut f32, {[128]}> = ...;
    /// let values: Tile<f32, {[128]}> = ...;
    /// let token = store_ptr_tko(ptrs, values, "relaxed", "device", None, None, None);
    /// ```
    ///
    /// Store with release semantics and input token:
    /// ```rust,ignore
    /// let token = store_ptr_tko(ptrs, values, "release", "device", None, Some(input_token), None);
    /// ```
    ///
    /// Store with mask to selectively store elements:
    /// ```rust,ignore
    /// let mask: Tile<bool, {[128]}> = ...;
    /// let token = store_ptr_tko(ptrs, values, "relaxed", "device", Some(mask), None, None);
    /// ```
    ///
    /// ## Returns
    ///
    /// Returns the result token representing completion of the store operation.
    #[cuda_tile::op(name="cuda_tile.store_ptr_tko", params=["destination", "value"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn store_ptr_tko<E: ElementType, const S: [i32; N]>(
        destination: PointerTile<*mut E, S>,
        value: Tile<E, S>,
        memory_ordering: &str,
        memory_scope: &str,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
        latency: Option<i32>,
    ) -> Token {
        unreachable!()
    }

    /// Gets a reference to a global variable.
    ///
    /// Returns a pointer to a global variable declared with the `global` operation.
    /// Global variables are shared across all thread blocks and persist for the
    /// lifetime of the module.
    ///
    /// ## Parameters
    ///
    /// The global variable is identified by name through a compiler attribute.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Assuming a global variable "shared_counter" was declared
    /// let counter_ptr: PointerTile<*mut i32, {[]}> = get_global();
    /// ```
    ///
    /// Note: This operation requires the global to be declared first using `global`.
    /// Currently this is an advanced operation with limited Rust API support.
    #[cuda_tile::op(name="cuda_tile.get_global", params=[], named_attributes=["name:symbol_ref"])]
    pub fn get_global<E: ElementType>() -> PointerTile<*mut E, { [] }> {
        unreachable!()
    }

    /// Generic reduce operation with custom closure.
    ///
    /// Reduces a tile along a dimension using a user-provided binary operation.
    /// This is the generic version that accepts any reduction operation via a closure.
    ///
    /// ## Parameters
    ///
    /// - `operand`: Tile to reduce
    /// - `dim`: Dimension to reduce along (becomes scalar)
    /// - `identity`: Identity value for the reduction
    /// - `f`: Binary operation closure `|acc, x| -> result`
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Sum reduction
    /// let sum = reduce(tile, 0, 0.0f32, |acc, x| acc + x);
    ///
    /// // Product reduction
    /// let product = reduce(tile, 0, 1.0f32, |acc, x| acc * x);
    ///
    /// // Min reduction
    /// let min = reduce(tile, 0, f32::MAX, |acc, x| minf(acc, x));
    ///
    /// // Max reduction
    /// let max = reduce(tile, 0, f32::MIN, |acc, x| maxf(acc, x));
    ///
    /// // Custom reduction
    /// let result = reduce(tile, 0, 0.0f32, |acc, x| maxf(acc * 0.9, x));
    /// ```
    ///
    /// Note: The closure body is compiled into an MLIR region at compile-time.
    #[cuda_tile::op(name="cuda_tile.reduce", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn reduce<E: ElementType, const S: [i32; N], F>(
        operand: Tile<E, S>,
        dim: i32,
        identity: E,
        f: F,
    ) -> Tile<E, S>
    where
        F: Fn(E, E) -> E,
    {
        unreachable!()
    }

    /// Scan sum operation - parallel prefix sum along a dimension.
    ///
    /// Computes cumulative sums along the specified dimension, preserving all
    /// intermediate results.
    ///
    /// ## Parameters
    ///
    /// - `operand`: Tile to scan
    /// - `dim`: Dimension to scan along
    /// - `reverse`: Whether to scan in reverse direction
    /// - `identity`: Identity value for sum (typically 0.0 or 0)
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let data: Tile<f32, {[128]}> = ...; // [1, 2, 3, 4, ...]
    /// let prefix_sums: Tile<f32, {[128]}> = scan_sum(data, 0, false, 0.0);
    /// // Result: [1, 3, 6, 10, ...] (cumulative sums)
    /// ```
    ///
    /// Note: Compiler builds MLIR region with addf/addi operation automatically.
    #[cuda_tile::op(name="cuda_tile.scan", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn scan_sum<E: ElementType, const S: [i32; N]>(
        operand: Tile<E, S>,
        dim: i32,
        reverse: bool,
        identity: E,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Generic scan operation with custom closure.
    ///
    /// Computes a prefix scan (cumulative operation) along a dimension using a
    /// user-provided binary operation. Preserves all intermediate results.
    ///
    /// ## Parameters
    ///
    /// - `operand`: Tile to scan
    /// - `dim`: Dimension to scan along
    /// - `reverse`: Whether to scan in reverse direction
    /// - `identity`: Identity value for the scan
    /// - `f`: Binary operation closure `|acc, x| -> result`
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Prefix sum (cumulative sum)
    /// let prefix_sum = scan(tile, 0, false, 0.0f32, |acc, x| acc + x);
    /// // Input:  [1, 2, 3, 4]
    /// // Output: [1, 3, 6, 10]
    ///
    /// // Prefix product
    /// let prefix_prod = scan(tile, 0, false, 1.0f32, |acc, x| acc * x);
    /// // Input:  [2, 3, 4, 5]
    /// // Output: [2, 6, 24, 120]
    ///
    /// // Running maximum
    /// let running_max = scan(tile, 0, false, f32::MIN, |acc, x| maxf(acc, x));
    /// // Input:  [3, 1, 4, 2]
    /// // Output: [3, 3, 4, 4]
    ///
    /// // Reverse scan
    /// let suffix_sum = scan(tile, 0, true, 0.0f32, |acc, x| acc + x);
    /// // Input:  [1, 2, 3, 4]
    /// // Output: [10, 9, 7, 4] (scans from right to left)
    /// ```
    ///
    /// Note: The closure body is compiled into an MLIR region at compile-time.
    #[cuda_tile::op(name="cuda_tile.scan", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn scan<E: ElementType, const S: [i32; N], F>(
        operand: Tile<E, S>,
        dim: i32,
        reverse: bool,
        identity: E,
        f: F,
    ) -> Tile<E, S>
    where
        F: Fn(E, E) -> E,
    {
        unreachable!()
    }

    /* Tensor */

    /// A view into GPU global memory with shape and stride information.
    ///
    /// `Tensor` represents a multi-dimensional array stored in GPU memory. Unlike the
    /// host-side [`crate::tensor::Tensor`], this is used within GPU kernels to reference
    /// input and output data.
    ///
    /// ## Type Parameters
    ///
    /// - `E`: Element type (must implement [`ElementType`])
    /// - `D`: Shape as a compile-time constant array. Use `-1` for dynamic dimensions.
    ///
    /// ## Shape Specifications
    ///
    /// - `{[128]}` - 1D tensor with exactly 128 elements
    /// - `{[64, 64]}` - 2D tensor with shape 64×64
    /// - `{[-1]}` - 1D tensor with dynamic size
    /// - `{[-1, 64]}` - 2D tensor with dynamic first dimension, fixed second dimension
    ///
    /// ## Examples
    ///
    /// ### Reading from a tensor
    ///
    /// ```rust,ignore
    /// #[cutile::entry]
    /// fn my_kernel<const N: i32>(
    ///     output: &mut Tensor<f32, {[N]}>,
    ///     input: &Tensor<f32, {[-1]}>,
    /// ) {
    ///     // Load the input tile corresponding to this block
    ///     let tile = load_tile_like_1d(input, output);
    ///
    ///     // Process and store
    ///     output.store(tile * 2.0);
    /// }
    /// ```
    ///
    /// ### Using partitions for large tensors
    ///
    /// ```rust,ignore
    /// #[cutile::entry]
    /// fn process_matrix<const BM: i32, const BN: i32>(
    ///     output: &mut Tensor<f32, {[BM, BN]}>,
    ///     input: &Tensor<f32, {[-1, -1]}>,
    /// ) {
    ///     let pid = get_tile_block_id();
    ///     let partition = input.partition(const_shape![BM, BN]);
    ///     let tile = partition.load([pid.0, pid.1]);
    ///     output.store(tile);
    /// }
    /// ```
    #[cuda_tile::ty(name="!cuda_tile.tensor_view",
                    type_params=["{D}xE", "strides"],
                    type_meta=["base", "shape", "strides", "token"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct Tensor<E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const S: [i32; N]> Tensor<E, S> {
        /// Creates a read-only partition view of this tensor.
        ///
        /// Partitions divide the tensor into tiles for parallel processing. Each tile
        /// can be loaded independently by different thread blocks.
        ///
        /// ## Examples
        ///
        /// ```rust,ignore
        /// let input: &Tensor<f32, {[-1, -1]}> = ...;
        /// let partition = input.partition(const_shape![64, 64]);
        /// let pid = get_tile_block_id();
        /// let tile = partition.load([pid.0, pid.1]);
        /// ```
        // TODO (hme): Need to add support for params/output_type_params like make_partition_view
        pub fn partition<'a, const R: [i32; N]>(&'a self, tile: Shape<R>) -> Partition<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            let p: Partition<E, R> = make_partition_view_padded(self, tile, "zero", tensor_token);
            p
        }
        pub fn partition_permuted<'a, const R: [i32; N], const I: [i32; N]>(
            &'a self,
            tile: Shape<R>,
            dim_map: Array<I>,
        ) -> Partition<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            let p: Partition<E, R> =
                make_partition_view_permuted(self, tile, dim_map, tensor_token);
            p
        }
        pub unsafe fn partition_mut<'a, const R: [i32; N]>(
            &'a mut self,
            tile: Shape<R>,
        ) -> PartitionMut<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            unsafe { make_partition_view_mut(self, tile, tensor_token) }
        }

        /// Returns the shape of this tensor.
        pub fn shape<'b>(&self) -> Shape<'b, S> {
            get_tensor_shape(self)
        }
        /// Loads the entire tensor contents into a tile.
        ///
        /// This is typically used when the tensor shape matches the tile size
        /// (i.e., the entire tensor fits in a single thread block).
        ///
        /// ## Examples
        ///
        /// ```rust,ignore
        /// fn kernel(x: &mut Tensor<f32, {[128]}>) {
        ///     let tile = x.load();
        ///     // Process tile...
        /// }
        /// ```
        pub fn load(&mut self) -> Tile<E, S> {
            load_tile_mut(self)
        }

        pub fn load_tile<const R: [i32; N]>(&self, shape: Shape<R>, idx: [i32; N]) -> Tile<E, R> {
            load_tile(self, shape, idx)
        }

        /// Stores a tile's contents into this tensor.
        ///
        /// This writes the tile data back to global memory at the location
        /// corresponding to this tensor view.
        ///
        /// ## Examples
        ///
        /// ```rust,ignore
        /// fn kernel(output: &mut Tensor<f32, {[128]}>) {
        ///     let tile = compute_result();
        ///     output.store(tile);
        /// }
        /// ```
        pub fn store(&mut self, result: Tile<E, S>) {
            store_tile(self, result);
        }
    }

    /// Extracts the shape metadata from a tensor.
    ///
    /// This is a low-level function that retrieves the shape information stored in
    /// the tensor's type metadata. Typically called internally; users should use
    /// the `.shape()` method on tensors instead.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor: &Tensor<f32, {[128, 64]}> = ...;
    /// let shape = get_tensor_shape(tensor);
    /// // Equivalent to: let shape = tensor.shape();
    /// ```
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "shape")]
    pub fn get_shape_dim<const S: [i32; N]>(shape: Shape<S>, dim_idx: i32) -> i32 {
        unreachable!()
    }

    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "shape")]
    pub fn get_tensor_shape<'s, E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
    ) -> Shape<'s, S> {
        unreachable!()
    }

    /// Extracts the memory ordering token from a tensor.
    ///
    /// Tokens track memory operation ordering to ensure correct synchronization.
    /// This is used internally by the partition and load/store operations to maintain
    /// memory consistency.
    ///
    /// ## Note
    ///
    /// This is an internal function used by the tile DSL implementation. Direct use
    /// is rarely needed in user code.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_tensor_token<E: ElementType, const S: [i32; N]>(tensor: &Tensor<E, S>) -> Token {
        unreachable!()
    }

    /// Updates the memory ordering token for a tensor.
    ///
    /// After performing memory operations on a tensor, this function updates its
    /// token to reflect the new memory state. This ensures subsequent operations
    /// observe the correct ordering.
    ///
    /// ## Note
    ///
    /// This is an internal function used to maintain memory consistency. The token
    /// mechanism is handled automatically by load/store operations.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "set_type_meta_field", type_meta_field = "token")]
    pub fn set_tensor_token<E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
        token: Token,
    ) {
        unreachable!()
    }

    /// Creates a tensor view from raw components.
    ///
    /// Constructs a [`Tensor`] from a base pointer, shape, strides, and ordering token.
    /// This is a low-level unsafe operation used internally by the tile DSL.
    ///
    /// ## Parameters
    ///
    /// - `base`: Pointer to the tensor's data in global memory
    /// - `shape`: Dimensions of the tensor
    /// - `strides`: Stride values for each dimension
    /// - `token`: Memory ordering token
    ///
    /// ## Safety
    ///
    /// This function is unsafe because:
    /// - The pointer must be valid and point to allocated GPU memory
    /// - The shape and strides must correctly describe the memory layout
    /// - The token must represent valid ordering constraints
    ///
    /// Typically, users should not call this directly; tensors are created through
    /// the API functions or passed as kernel parameters.
    #[cuda_tile::op(name="cuda_tile.make_tensor_view",
                    params=["base", "shape.dims", "strides.dims"],
                    has_variadic_params=true,
                    output_type_params=["strides"],
                    output_type_meta=["base", "shape", "strides", "token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_tensor_view<E: ElementType, const D: [i32; N], const C: [i32; N]>(
        base: PointerTile<*mut E, { [] }>,
        shape: Shape<D>,
        strides: Array<C>,
        token: Token,
    ) -> Tensor<E, D> {
        unreachable!()
    }

    /* Partition */

    /// A read-only view that divides a tensor into tiles for parallel processing.
    ///
    /// `Partition` provides indexed access to tiles within a tensor. Each thread block
    /// typically processes one or more tiles from a partition.
    ///
    /// ## Type Parameters
    ///
    /// - `E`: Element type
    /// - `D`: Tile shape (shape of each partition)
    ///
    /// ## Examples
    ///
    /// ### Basic partition loading
    ///
    /// ```rust,ignore
    /// fn kernel(input: &Tensor<f32, {[-1, -1]}>) {
    ///     let partition = input.partition(const_shape![64, 64]);
    ///     let pid = get_tile_block_id();
    ///
    ///     // Load the tile for this block
    ///     let tile = partition.load([pid.0, pid.1]);
    ///     // Process tile...
    /// }
    /// ```
    ///
    /// ### Loop over multiple tiles
    ///
    /// ```rust,ignore
    /// let partition = input.partition(const_shape![64, 64]);
    /// for i in 0..num_tiles {
    ///     let tile = partition.load([pid.0, i]);
    ///     // Accumulate results...
    /// }
    /// ```
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view", "dim_map"],
                    type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct Partition<'a, E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
        _tensor: PhantomData<&'a ()>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, E: ElementType, const D: [i32; N]> Partition<'a, E, D> {
        /// Loads a tile from this partition at the specified index.
        ///
        /// The index specifies which tile to load. For a 2D partition, `[i, j]` loads
        /// the tile at position (i, j).
        ///
        /// ## Examples
        ///
        /// ```rust,ignore
        /// let partition = tensor.partition(const_shape![64]);
        /// let tile = partition.load([5]); // Load 6th tile (64 elements starting at 5*64)
        /// ```
        pub fn load(&self, index: [i32; N]) -> Tile<E, D> {
            check_partition_access(self, index);
            let result: Tile<E, D> = load_from_view(self, index, None, false);
            result
        }
    }

    /// Creates a read-only partition view from a tensor.
    ///
    /// This is the internal function used by `Tensor::partition()`. It creates a
    /// partition structure that divides the tensor into tiles of the specified shape.
    ///
    /// ## Parameters
    ///
    /// - `tensor_view`: The tensor to partition
    /// - `tile`: Shape of each partition tile
    /// - `token`: Memory ordering token
    ///
    /// ## Note
    ///
    /// Users should call `.partition()` on tensors rather than using this function directly.
    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view"],
                    output_type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn make_partition_view<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        tile: Shape<TILE_SHAPE>,
        token: Token,
    ) -> Partition<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    /// Returns the number of tiles along `axis` in the partition's tile space.
    ///
    /// For a partition of a tensor of shape `[M, N]` with tile `[BM, BN]`:
    /// - `num_tiles(&partition, 0)` returns `cdiv(M, BM)`.
    /// - `num_tiles(&partition, 1)` returns `cdiv(N, BN)`.
    ///
    /// `axis` must be a compile-time constant in `0..N` where `N` is the
    /// partition's rank.
    ///
    /// Lowers to the Tile IR `cuda_tile.get_index_space_shape` op with the
    /// axis-th result extracted at JIT time.
    ///
    /// # Safety
    ///
    /// Thin wrapper over the underlying Tile IR op. The compiler checks that
    /// `axis` is a compile-time constant in range but performs no further
    /// validation of the partition operand.
    #[cuda_tile::compiler_op(name = "num_tiles")]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn num_tiles<E: ElementType, const S: [i32; N]>(
        view: &Partition<E, S>,
        axis: i32,
    ) -> i32 {
        unreachable!()
    }

    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "dim_map"],
                    output_type_meta=["token", "tensor_view.shape()", "dim_map"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn make_partition_view_permuted<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        const I: [i32; N],
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        tile: Shape<TILE_SHAPE>,
        dim_map: Array<I>,
        token: Token,
    ) -> Partition<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn make_partition_view_padded<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        tile: Shape<TILE_SHAPE>,
        padding_value: &str,
        token: Token,
    ) -> Partition<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    // TODO (hme): Mark loads from shared refs as unsafe and add suffix _unchecked.
    #[cuda_tile::op(name="load_view_tko", params=["view", "index"], hint_params=["latency"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_from_view<E: ElementType, const D: [i32; N]>(
        view: &Partition<E, D>,
        index: [i32; N],
        latency: Option<i32>,
        disallow_tma: bool,
    ) -> Tile<E, D> {
        unreachable!()
    }

    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_partition_token<E: ElementType, const D: [i32; N]>(view: &Partition<E, D>) -> Token {
        unreachable!()
    }

    /* PartitionMut */

    /// A mutable partition view that allows unordered loads and stores.
    ///
    /// `PartitionMut` is similar to [`Partition`] but allows both loading and storing tiles.
    /// Operations on `PartitionMut` use unordered memory operations for performance,
    /// which is why they're marked unsafe.
    ///
    /// ## Safety
    ///
    /// The load and store methods are unsafe because they don't enforce memory ordering.
    /// Prefer using the ordered `load()` and `store()` methods on [`Tensor`] directly
    /// when possible.
    ///
    /// ## Type Parameters
    ///
    /// - `E`: Element type
    /// - `D`: Tile shape
    // TODO (hme): Look into consolidating into a single type.
    #[cuda_tile::ty(name="!cuda_tile.partition_view",
                    type_params=["tile"],
                    type_params_optional=["padding_value", "tensor_view"],
                    type_meta=["token"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct PartitionMut<'a, E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
        _tensor: PhantomData<&'a mut ()>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, E: ElementType, const D: [i32; N]> PartitionMut<'a, E, D> {
        /// Loads a tile from this mutable partition at the specified index.
        ///
        /// ## Safety
        ///
        /// This is unsafe because it uses unordered memory operations.
        // These are unsafe because they don't make use of ordered memory operations.
        pub unsafe fn load(&self, index: [i32; N]) -> Tile<E, D> {
            unsafe { load_from_view_mut(self, index) }
        }

        /// Stores a tile to this mutable partition at the specified index.
        ///
        /// Returns a token representing the completion of the store operation.
        ///
        /// ## Safety
        ///
        /// This is unsafe because it uses unordered memory operations.
        pub unsafe fn store(&mut self, tile: Tile<E, D>, index: [i32; N]) -> Token {
            let token: Token = unsafe { store_to_view_mut(self, tile, index, None, false) };
            token
        }
    }

    /// Creates a mutable partition view from a tensor.
    ///
    /// This is the internal function used by `Tensor::partition_mut()`. It creates a
    /// mutable partition structure for unordered memory operations.
    ///
    /// ## Safety
    ///
    /// This is unsafe because:
    /// - The lifetime `'a` is not tied to the tensor lifetime
    /// - The resulting partition uses unordered memory operations
    ///
    /// ## Note
    ///
    /// Users should prefer the ordered `load()` and `store()` methods on [`Tensor`].
    // This is unsafe because the lifetime 'a is not tied to tensor.
    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view"],
                    output_type_meta=["token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_partition_view_mut<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        token: Token,
    ) -> PartitionMut<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_partition_view_mut_padded<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        padding_value: &str,
        token: Token,
    ) -> PartitionMut<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    /// Loads a tile from a mutable partition view (unordered).
    ///
    /// This is the internal implementation used by `PartitionMut::load()`.
    ///
    /// ## Safety
    ///
    /// This is unsafe because it doesn't use ordered memory operations.
    // This is unsafe because it doesn't make use of ordered memory operations.
    #[cuda_tile::op(name="load_view_tko", params=["view", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_from_view_mut<E: ElementType, const D: [i32; N]>(
        view: &PartitionMut<E, D>,
        index: [i32; N],
    ) -> Tile<E, D> {
        unreachable!()
    }

    /// Stores a tile to a mutable partition view (unordered).
    ///
    /// This is the internal implementation used by `PartitionMut::store()`.
    /// Returns a token representing the completion of the store.
    ///
    /// ## Safety
    ///
    /// This is unsafe because it doesn't use ordered memory operations.
    // This is unsafe because it doesn't make use of ordered memory operations.
    #[cuda_tile::op(name="store_view_tko", params=["view", "tile", "index"], hint_params=["latency"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_to_view_mut<E: ElementType, const D: [i32; N]>(
        view: &mut PartitionMut<E, D>,
        tile: Tile<E, D>,
        index: [i32; N],
        latency: Option<i32>,
        disallow_tma: bool,
    ) -> Token {
        unreachable!()
    }

    /// Extracts the memory ordering token from a mutable partition view.
    ///
    /// Used internally to maintain memory consistency.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_partition_token_mut<E: ElementType, const D: [i32; N]>(
        view: &PartitionMut<E, D>,
    ) -> Token {
        unreachable!()
    }

    /* Tile */

    /// A multi-dimensional array stored in registers or shared memory.
    ///
    /// `Tile` is the fundamental data structure for computation within GPU kernels.
    /// Unlike [`Tensor`] which resides in slow global memory, tiles are stored in fast
    /// registers or shared memory, making operations on them extremely efficient.
    ///
    /// ## Type Parameters
    ///
    /// - `E`: Element type (must implement [`ElementType`])
    /// - `D`: Shape as a compile-time constant array
    ///
    /// ## Usage Pattern
    ///
    /// 1. Load data from global memory ([`Tensor`]) into a tile
    /// 2. Perform computations on the tile using operators (+, *, etc.)
    /// 3. Store the result back to global memory
    ///
    /// ## Examples
    ///
    /// ### Element-wise operations
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = partition_a.load([pid.0]);
    /// let b: Tile<f32, {[128]}> = partition_b.load([pid.0]);
    ///
    /// // All operations happen in registers/shared memory
    /// let sum = a + b;
    /// let product = a * b;
    /// let result = sum * 2.0 + product;
    /// ```
    ///
    /// ### Broadcasting and reshaping
    ///
    /// ```rust,ignore
    /// // Broadcast a scalar
    /// let scalar_tile = 3.14f32.broadcast(const_shape![128]);
    ///
    /// // Reshape a tile
    /// let matrix: Tile<f32, {[8, 16]}> = vector.reshape(const_shape![8, 16]);
    /// ```
    ///
    /// ### Matrix multiplication
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[64, 32]}> = ...;
    /// let b: Tile<f32, {[32, 64]}> = ...;
    /// let c: Tile<f32, {[64, 64]}> = ...;
    ///
    /// // Hardware-accelerated matrix multiply-accumulate
    /// let result = mma(a, b, c); // c + a * b
    /// ```
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xE"])]
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct Tile<E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> Tile<E, D> {
        /// Returns the shape of this tile.
        pub fn shape(&self) -> Shape<D> {
            unreachable!()
        }

        /// Broadcasts this tile to a new shape.
        ///
        /// The tile's dimensions must be compatible with the target shape
        /// (dimensions must be 1 or match).
        ///
        /// ## Examples
        ///
        /// ```rust,ignore
        /// let vec: Tile<f32, {[64, 1]}> = ...;
        /// let matrix = vec.broadcast(const_shape![64, 128]); // Broadcast column across rows
        /// ```
        pub fn broadcast<const R: [i32; N]>(self, shape: Shape<R>) -> Tile<E, R> {
            broadcast(self, shape)
        }

        /// Reshapes this tile to a new shape without moving data.
        ///
        /// The total number of elements must remain the same.
        ///
        /// ## Examples
        ///
        /// ```rust,ignore
        /// let vec: Tile<f32, {[128]}> = ...;
        /// let matrix = vec.reshape(const_shape![8, 16]); // 128 = 8 * 16
        /// ```
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn reshape<const R: [i32; M]>(self, shape: Shape<R>) -> Tile<E, R> {
            reshape(self, shape)
        }
    }

    /// Element-wise addition of tiles.
    ///
    /// Enables the `+` operator for tiles. Performs element-wise addition where
    /// `result[i] = self[i] + rhs[i]`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let sum = a + b; // Element-wise addition
    /// ```
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Add<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn add(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    /// Element-wise subtraction of tiles.
    ///
    /// Enables the `-` operator for tiles. Performs element-wise subtraction where
    /// `result[i] = self[i] - rhs[i]`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let diff = a - b; // Element-wise subtraction
    /// ```
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Sub<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn sub(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    /// Element-wise multiplication of tiles.
    ///
    /// Enables the `*` operator for tiles. Performs element-wise (Hadamard) multiplication
    /// where `result[i] = self[i] * rhs[i]`.
    ///
    /// ## Note
    ///
    /// This is **not** matrix multiplication. For matrix multiplication, use [`mma()`].
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let product = a * b; // Element-wise multiplication
    /// ```
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Mul<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn mul(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    /// Element-wise division of tiles.
    ///
    /// Enables the `/` operator for tiles. Performs element-wise division where
    /// `result[i] = self[i] / rhs[i]`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let quotient = a / b; // Element-wise division
    /// ```
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Div<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn div(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    /// Element-wise remainder (modulo) of tiles.
    ///
    /// Enables the `%` operator for tiles. Performs element-wise remainder where
    /// `result[i] = self[i] % rhs[i]`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i32, {[128]}> = ...;
    /// let b: Tile<i32, {[128]}> = ...;
    /// let remainder = a % b; // Element-wise modulo
    /// ```
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Rem<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn rem(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::BitAnd<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn bitand(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::BitOr<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn bitor(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::BitXor<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn bitxor(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    // These aren't going to be possible in Rust.
    // #[cuda_tile::variadic_impl(N=4)]
    // impl<E: ElementType, const D: [i32; N]> cmp::PartialEq for Tile<E, D> {
    //     fn eq(&self, other: &Self) -> bool {
    //         unreachable!()
    //     }
    // }
    //
    // #[cuda_tile::variadic_impl(N=4)]
    // impl<E: ElementType, const D: [i32; N]> cmp::Eq for Tile<E, D> {}
    //
    // #[cuda_tile::variadic_impl(N=4)]
    // impl<E: ElementType, const D: [i32; N]> cmp::PartialOrd for Tile<E, D> {
    //     fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
    //         unreachable!()
    //     }
    // }
    //
    // #[cuda_tile::variadic_impl(N=4)]
    // impl<E: ElementType, const D: [i32; N]> cmp::Ord for Tile<E, D> {
    //     fn cmp(&self, other: &Self) -> cmp::Ordering {
    //         unreachable!()
    //     }
    // }

    // The compiler expects these ops to end with _tile.
    /// Element-wise equality comparison.
    ///
    /// Compares two tiles element-wise, returning a boolean tile where each element
    /// is `true` if the corresponding elements are equal, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let equal = eq_tile(a, b);  // Tile<bool, {[128]}>
    /// // Use with select for conditional operations
    /// let result = select(equal, a, b);  // Choose a where equal, else b
    /// ```
    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn eq_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise inequality comparison.
    ///
    /// Compares two tiles element-wise, returning a boolean tile where each element
    /// is `true` if the corresponding elements are not equal, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i32, {[64, 64]}> = ...;
    /// let b: Tile<i32, {[64, 64]}> = ...;
    /// let not_equal = ne_tile(a, b);  // Tile<bool, {[64, 64]}>
    /// ```
    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ne_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise greater-than comparison.
    ///
    /// Compares two tiles element-wise, returning a boolean tile where each element
    /// is `true` if `lhs[i] > rhs[i]`, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let values: Tile<f32, {[128]}> = ...;
    /// let threshold: Tile<f32, {[128]}> = ...;
    /// let above = gt_tile(values, threshold);  // values > threshold
    /// ```
    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn gt_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise greater-than-or-equal comparison.
    ///
    /// Compares two tiles element-wise, returning a boolean tile where each element
    /// is `true` if `lhs[i] >= rhs[i]`, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let values: Tile<f32, {[128]}> = ...;
    /// let min_value = 0.0f32.broadcast(const_shape![128]);
    /// let valid = ge_tile(values, min_value);  // values >= 0
    /// ```
    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ge_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise less-than comparison.
    ///
    /// Compares two tiles element-wise, returning a boolean tile where each element
    /// is `true` if `lhs[i] < rhs[i]`, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let predictions: Tile<f32, {[64]}> = ...;
    /// let targets: Tile<f32, {[64]}> = ...;
    /// let underestimated = lt_tile(predictions, targets);
    /// ```
    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn lt_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise less-than-or-equal comparison.
    ///
    /// Compares two tiles element-wise, returning a boolean tile where each element
    /// is `true` if `lhs[i] <= rhs[i]`, `false` otherwise.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let values: Tile<i32, {[256]}> = ...;
    /// let max_val = 255i32.broadcast(const_shape![256]);
    /// let in_range = le_tile(values, max_val);  // values <= 255
    /// ```
    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn le_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Returns the minimum of two scalar values.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let result = min(5, 3); // 3
    /// ```
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn min<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Returns the maximum of two values.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let result = max(5, 3); // 5
    /// let result = max(2.7, 8.1); // 8.1
    /// ```
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn max<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Element-wise minimum of two tiles.
    ///
    /// Returns a tile where each element is the minimum of the corresponding elements
    /// from the two input tiles.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let minimums = min_tile(a, b);  // min(a[i], b[i]) for each i
    ///
    /// // Clamp values to maximum
    /// let max_limit = 100.0f32.broadcast(const_shape![128]);
    /// let clamped = min_tile(values, max_limit);  // values clamped to [0, 100]
    /// ```
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn min_tile<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise maximum of two tiles.
    ///
    /// Returns a tile where each element is the maximum of the corresponding elements
    /// from the two input tiles.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let maximums = max_tile(a, b);  // max(a[i], b[i]) for each i
    ///
    /// // Clamp values to minimum (ReLU-like)
    /// let zero = 0.0f32.broadcast(const_shape![128]);
    /// let activated = max_tile(values, zero);  // ReLU: max(values, 0)
    /// ```
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn max_tile<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Ceiling division: `⌈a / b⌉` for scalars.
    ///
    /// Returns the smallest integer greater than or equal to `a / b`.
    /// Equivalent to `(a + b - 1) / b` for positive integers.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let result = ceil_div(7, 3);  // 3 (since 7/3 = 2.33...)
    /// let result = ceil_div(6, 3);  // 2 (since 6/3 = 2.0)
    /// let result = ceil_div(10, 4); // 3 (since 10/4 = 2.5)
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Computing number of blocks needed for tile computation
    /// - Rounding up to next multiple
    /// - Allocating resources in discrete units
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn ceil_div<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Element-wise true division (floating-point division).
    ///
    /// Performs element-wise division ensuring floating-point result even for integer inputs.
    /// This differs from the `/` operator which may perform integer division.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...;
    /// let b: Tile<f32, {[128]}> = ...;
    /// let quotients = true_div(a, b);  // Always floating-point division
    /// ```
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn true_div<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    // Common Unary Operations
    /// Computes element-wise ceiling (round up) of floating-point tiles.
    ///
    /// Returns a tile where each element is rounded up to the nearest integer.
    ///
    /// ## Parameters
    ///
    /// - `x`: Input tile
    /// - `rounding_mode`: Rounding mode string (e.g., "nearest_even", "positive_inf", "negative_inf", "nearest_int_to_zero", "approx")
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [1.2, 2.7, -1.5, ...]
    /// let result = ceil(x, "nearest_even"); // [2.0, 3.0, -1.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.ceil", params=["x"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ceil<E: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<E, S>,
        rounding: R,
    ) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise cosine of floating-point tiles.
    ///
    /// Returns a tile where each element is the cosine of the corresponding
    /// input element (in radians).
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, π/2, π, ...]
    /// let result = cos(x); // [1.0, 0.0, -1.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.cos", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cos<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise exponential of floating-point tiles.
    ///
    /// Returns a tile where each element is e raised to the power of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, 1.0, 2.0, ...]
    /// let result = exp(x); // [1.0, 2.718..., 7.389..., ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.exp", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exp<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise natural logarithm of floating-point tiles.
    ///
    /// Returns a tile where each element is the natural logarithm of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [1.0, 2.718..., 7.389..., ...]
    /// let result = log(x); // [0.0, 1.0, 2.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.log", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn log<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise reciprocal square root of floating-point tiles.
    ///
    /// Returns a tile where each element is the reciprocal square root of the
    /// corresponding input element. This is often faster than computing 1.0/sqrt(x).
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [4.0, 9.0, 16.0, ...]
    /// let result = rsqrt(x); // [0.5, 0.333..., 0.25, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.rsqrt", params=["x"], static_params=["ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn rsqrt<E: ElementType, const S: [i32; N], F: ftz::Mode>(
        x: Tile<E, S>,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise sine of floating-point tiles.
    ///
    /// Returns a tile where each element is the sine of the corresponding
    /// input element (in radians).
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, π/2, π, ...]
    /// let result = sin(x); // [0.0, 1.0, 0.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.sin", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sin<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise square root of floating-point tiles.
    ///
    /// Returns a tile where each element is the square root of the corresponding
    /// input element.
    ///
    /// ## Parameters
    ///
    /// - `x`: Input tile
    /// - `rounding_mode`: Rounding mode string (e.g., "nearest_even", "positive_inf", "negative_inf", "nearest_int_to_zero", "approx")
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [4.0, 9.0, 16.0, ...]
    /// let result = sqrt(x, "negative_inf"); // [2.0, 3.0, 4.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.sqrt", params=["x"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sqrt<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        x: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }
    /// Computes element-wise tangent of floating-point tiles.
    ///
    /// Returns a tile where each element is the tangent of the corresponding
    /// input element (in radians).
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, π/4, π, ...]
    /// let result = tan(x); // [0.0, 1.0, 0.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.tan", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn tan<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    // Reduce operations
    /// Reduces a tile along a dimension by computing the minimum.
    ///
    /// Computes the minimum value along the specified dimension, collapsing that
    /// dimension to size 1. This is a specialized version of [`reduce`] for minimum.
    ///
    /// ## Parameters
    ///
    /// - `x`: The input tile to reduce
    /// - `dim`: Which dimension to reduce (0-indexed)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Find minimum along dimension 0 (rows)
    /// let matrix: Tile<f32, {[64, 128]}> = ...;
    /// let col_mins = reduce_min(matrix, 0);  // Shape: {[1, 128]}
    ///
    /// // Find minimum along dimension 1 (columns)
    /// let row_mins = reduce_min(matrix, 1);  // Shape: {[64, 1]}
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Finding minimum values per feature
    /// - Clipping/normalization operations
    /// - Statistical analysis
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_min<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Reduces a tile along a dimension by computing the maximum.
    ///
    /// Computes the maximum value along the specified dimension, collapsing that
    /// dimension to size 1. This is a specialized version of [`reduce`] for maximum.
    ///
    /// ## Parameters
    ///
    /// - `x`: The input tile to reduce
    /// - `dim`: Which dimension to reduce (0-indexed)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Find maximum along dimension 0
    /// let matrix: Tile<f32, {[64, 128]}> = ...;
    /// let col_maxs = reduce_max(matrix, 0);  // Shape: {[1, 128]}
    ///
    /// // Find global maximum (reduce all dimensions)
    /// let row_max = reduce_max(matrix, 1);  // Shape: {[64, 1]}
    /// let global_max = reduce_max(row_max, 0);  // Shape: {[1, 1]}
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Finding peak values
    /// - Computing log-sum-exp for numerical stability
    /// - Max pooling in neural networks
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_max<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Reduces a tile along a dimension by computing the sum.
    ///
    /// Computes the sum of values along the specified dimension, collapsing that
    /// dimension to size 1. This is a specialized version of [`reduce`] for summation.
    ///
    /// ## Parameters
    ///
    /// - `x`: The input tile to reduce
    /// - `dim`: Which dimension to reduce (0-indexed)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Sum along dimension 0 (column sums)
    /// let matrix: Tile<f32, {[64, 128]}> = ...;
    /// let col_sums = reduce_sum(matrix, 0);  // Shape: {[1, 128]}
    ///
    /// // Sum along dimension 1 (row sums)
    /// let row_sums = reduce_sum(matrix, 1);  // Shape: {[64, 1]}
    ///
    /// // Compute mean by dividing result
    /// let mean = reduce_sum(matrix, 0) / 64.0;
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Computing totals and aggregates
    /// - Mean/variance calculations
    /// - Matrix-vector products (with broadcasting)
    /// - Softmax normalization
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_sum<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Reduces a tile along a dimension by computing the product.
    ///
    /// Computes the product of values along the specified dimension, collapsing that
    /// dimension to size 1. This is a specialized version of [`reduce`] for multiplication.
    ///
    /// ## Parameters
    ///
    /// - `x`: The input tile to reduce
    /// - `dim`: Which dimension to reduce (0-indexed)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Product along dimension 0
    /// let matrix: Tile<f32, {[64, 128]}> = ...;
    /// let col_products = reduce_prod(matrix, 0);  // Shape: {[1, 128]}
    ///
    /// // Useful for computing factorials or combinations
    /// let numbers: Tile<f32, {[10]}> = ...; // [1, 2, 3, ..., 10]
    /// let factorial = reduce_prod(numbers, 0);  // 10! = 3628800
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Computing geometric means (with log/exp)
    /// - Probability calculations (product of independent probabilities)
    /// - Determinant computations
    ///
    /// ## Note
    ///
    /// Be careful of numeric overflow/underflow with products. Consider using
    /// sum of logarithms for better numerical stability.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_prod<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Reshapes a tile to a new shape without moving data.
    ///
    /// Reinterprets the tile's data with a different shape. The total number of elements
    /// must remain the same (product of dimensions). This is a view operation that doesn't
    /// copy or rearrange data in memory.
    ///
    /// ## Parameters
    ///
    /// - `source`: The input tile to reshape
    /// - `shape`: The desired output shape
    ///
    /// ## Constraints
    ///
    /// The product of the source dimensions must equal the product of the result dimensions:
    /// `S[0] * S[1] * ... * S[N-1] == R[0] * R[1] * ... * R[M-1]`
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Reshape 1D to 2D
    /// let vec: Tile<f32, {[128]}> = ...;
    /// let matrix = reshape(vec, const_shape![8, 16]);  // 128 = 8 * 16
    ///
    /// // Reshape 2D to 1D (flatten)
    /// let matrix: Tile<f32, {[4, 32]}> = ...;
    /// let flat = reshape(matrix, const_shape![128]);  // 4 * 32 = 128
    ///
    /// // Change dimensionality
    /// let tile_2d: Tile<i32, {[64, 64]}> = ...;
    /// let tile_3d = reshape(tile_2d, const_shape![16, 16, 16]);  // 4096 elements
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Flattening multi-dimensional data for linear operations
    /// - Preparing data for matrix multiplication with specific shapes
    /// - Converting between different tensor representations
    /// - Batching/unbatching operations
    ///
    /// ## Note
    ///
    /// Unlike `permute`, this doesn't change the order of elements in memory,
    /// only how they're indexed. For reordering dimensions, use `permute`.
    #[cuda_tile::op(name="cuda_tile.reshape", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reshape<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        source: Tile<E, S>,
        shape: Shape<R>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Broadcasts a tile to a new shape.
    ///
    /// Dimensions of size 1 in the source can be broadcast to any size in the result.
    /// Non-unit dimensions must match between source and result.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Broadcast a column vector across columns
    /// let col: Tile<f32, {[64, 1]}> = ...;
    /// let matrix = broadcast(col, const_shape![64, 128]);
    /// // Each column of matrix is a copy of col
    /// ```
    #[cuda_tile::op(name="cuda_tile.broadcast", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn broadcast<E: ElementType, const S: [i32; N], const R: [i32; N]>(
        source: Tile<E, S>,
        shape: Shape<R>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Permutes (reorders) the dimensions of a tile.
    ///
    /// Transposes or rearranges dimensions according to the specified permutation.
    /// This is a generalization of matrix transpose to arbitrary dimensions.
    ///
    /// ## Parameters
    ///
    /// - `source`: The input tile to permute
    /// - `permutation`: Array specifying the new dimension order
    ///
    /// ## Semantics
    ///
    /// The permutation array maps output dimensions to input dimensions. For example,
    /// `[1, 0]` swaps dimensions 0 and 1 (transpose), while `[2, 0, 1]` moves the
    /// last dimension to the front.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Transpose a 2D matrix
    /// let matrix: Tile<f32, {[64, 128]}> = ...;
    /// let transposed = permute(matrix, const_array![1, 0]);
    /// // Result shape: {[128, 64]}
    ///
    /// // Rotate 3D tensor dimensions
    /// let tensor: Tile<f32, {[32, 64, 16]}> = ...;
    /// let rotated = permute(tensor, const_array![2, 0, 1]);
    /// // Result shape: {[16, 32, 64]} - last dim moved to front
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Matrix transposition for matrix multiplication
    /// - Changing memory layout for optimal access patterns
    /// - Implementing operations like `einsum` with dimension reordering
    #[cuda_tile::op(name="cuda_tile.permute", params=["source"], attribute_params=["permutation:array"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn permute<E: ElementType, const A: [i32; N], const I: [i32; N], const R: [i32; N]>(
        source: Tile<E, A>,
        permutation: Array<I>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Creates a tile filled with a constant value.
    ///
    /// Generates a tile of the specified shape where all elements have the same value.
    /// This is more efficient than broadcasting for compile-time constants as the value
    /// can be embedded directly in the generated code.
    ///
    /// ## Parameters
    ///
    /// - `value`: The constant value to fill the tile with
    /// - `shape`: The shape of the resulting tile
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Create a tile of zeros
    /// let zeros: Tile<f32, {[128, 64]}> = constant(0.0f32, const_shape![128, 64]);
    ///
    /// // Create a tile of ones
    /// let ones: Tile<i32, {[256]}> = constant(1i32, const_shape![256]);
    ///
    /// // Create a tile with a specific constant
    /// let tile: Tile<f32, {[64, 64]}> = constant(3.14f32, const_shape![64, 64]);
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Initializing accumulators (zeros)
    /// - Creating masks (ones/zeros)
    /// - Generating constant tensors for operations
    ///
    /// ## Note
    ///
    /// For scalar broadcast followed by operations, consider using `broadcast_scalar`
    /// or the `.broadcast()` method on scalars instead.
    #[cuda_tile::op(name="cuda_tile.constant", params=[], attribute_params=["value:dense"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn constant<E: ElementType, const S: [i32; N]>(value: E, shape: Shape<S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Matrix multiply-accumulate using hardware-accelerated Tensor Cores.
    ///
    /// Performs `result = lhs × rhs + acc` using GPU Tensor Core instructions for
    /// maximum performance. This is the primary operation for deep learning and
    /// dense linear algebra on modern GPUs.
    ///
    /// ## Type Parameters
    ///
    /// - `E1`: Element type for input matrices (typically `f16`, `f32`, or `tf32`)
    /// - `E2`: Element type for accumulator and result (typically `f32`)
    /// - `M`, `N`, `K`: Matrix dimensions (M×K) × (K×N) + (M×N) → (M×N)
    ///
    /// ## Matrix Shapes
    ///
    /// - `lhs`: M×K matrix (left operand)
    /// - `rhs`: K×N matrix (right operand)
    /// - `acc`: M×N matrix (accumulator, added to product)
    /// - Returns: M×N matrix (result)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Basic matrix multiplication with accumulation
    /// let a: Tile<f16, {[64, 32]}> = ...;  // 64×32
    /// let b: Tile<f16, {[32, 64]}> = ...;  // 32×64
    /// let c: Tile<f32, {[64, 64]}> = ...;  // 64×64 accumulator
    /// let result = mma(a, b, c);  // result = a × b + c, shape {[64, 64]}
    ///
    /// // Initialize accumulator to zero for pure multiplication
    /// let zeros: Tile<f32, {[64, 64]}> = constant(0.0f32, const_shape![64, 64]);
    /// let product = mma(a, b, zeros);  // product = a × b
    /// ```
    ///
    /// ## Performance Notes
    ///
    /// - **Tensor Cores**: On NVIDIA Ampere and later GPUs, this uses specialized
    ///   Tensor Core hardware for massive throughput (up to 312 TFLOPS on A100)
    /// - **Mixed Precision**: Using `f16` inputs with `f32` accumulation balances
    ///   performance and numerical accuracy
    /// - **Tile Sizes**: Optimal tile sizes are typically multiples of 16 or 32
    ///   depending on GPU architecture
    ///
    /// ## Supported Type Combinations
    ///
    /// - `f16 × f16 + f32 → f32` (most common for deep learning)
    /// - `f32 × f32 + f32 → f32` (standard precision)
    /// - `tf32 × tf32 + f32 → f32` (TensorFloat-32 on Ampere+)
    ///
    /// ## Use Cases
    ///
    /// - Neural network layers (fully connected, attention)
    /// - GEMM operations in deep learning
    /// - Dense linear algebra (BLAS-3 operations)
    /// - Matrix factorizations
    #[cuda_tile::compiler_op(name = "mma")]
    pub fn mma<E1: ElementType, E2: ElementType, const M: i32, const N: i32, const K: i32>(
        lhs: Tile<E1, { [M, K] }>,
        rhs: Tile<E1, { [K, N] }>,
        acc: Tile<E2, { [M, N] }>,
    ) -> Tile<E2, { [M, N] }> {
        unreachable!()
    }

    /// Generates a 1D tile with sequential integer values.
    ///
    /// Creates a tile containing the sequence `[0, 1, 2, ..., N-1]` where N is
    /// the specified size. This is useful for generating indices, ranges, and
    /// arithmetic sequences.
    ///
    /// ## Parameters
    ///
    /// - `shape`: The shape of the resulting 1D tile
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Generate indices 0 to 127
    /// let indices: Tile<i32, {[128]}> = iota(const_shape![128]);
    /// // indices = [0, 1, 2, 3, ..., 127]
    ///
    /// // Use for indexing operations
    /// let offsets: Tile<i32, {[64]}> = iota(const_shape![64]);
    /// let scaled = offsets * 4;  // [0, 4, 8, 12, ..., 252]
    ///
    /// // Generate a range for lookup table
    /// let range: Tile<f32, {[256]}> = convert_tile(iota(const_shape![256]));
    /// let normalized = range / 255.0;  // [0.0, 1/255, 2/255, ..., 1.0]
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Generating array indices for gather/scatter operations
    /// - Creating coordinate grids for image processing
    /// - Building lookup tables and sequences
    /// - Index calculations for strided memory access
    ///
    /// ## Note
    ///
    /// Currently only supports 1D tiles. For multi-dimensional index grids,
    /// use `iota` with `reshape` and `broadcast`.
    #[cuda_tile::op(name = "cuda_tile.iota")]
    pub fn iota<E: ElementType, const S: [i32; 1]>(shape: Shape<S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise absolute value of integer tiles.
    ///
    /// Returns a tile where each element is the absolute value of the corresponding
    /// input element (for integer types).
    ///
    /// **Note:** Only `i64` is supported for signed integers in cutile, not `i32`.
    /// TileIR itself supports both types.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<i64, {[128]}> = ...; // [-1, 2, -3, ...]
    /// let result = absi(x); // [1, 2, 3, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.absi", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn absi<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise absolute value of floating-point tiles.
    ///
    /// Returns a tile where each element is the absolute value of the corresponding
    /// input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [-1.5, 2.0, -3.5, ...]
    /// let result = absf(x); // [1.5, 2.0, 3.5, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.absf", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn absf<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise negation of integer tiles.
    ///
    /// **Note:** Only `i64` is supported for signed integers in cutile, not `i32`.
    /// TileIR itself supports both types.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<i64, {[128]}> = ...; // [1, -2, 3, ...]
    /// let result = negi(x); // [-1, 2, -3, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.negi", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn negi<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise negation of floating-point tiles.
    ///
    /// Returns a tile where each element is the negation of the corresponding
    /// input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [1.5, -2.0, 3.5, ...]
    /// let result = negf(x); // [-1.5, 2.0, -3.5, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.negf", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn negf<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise floor of floating-point tiles.
    ///
    /// Returns a tile where each element is rounded down to the nearest integer.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [1.2, 2.7, -1.5, ...]
    /// let result = floor(x); // [1.0, 2.0, -2.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.floor", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn floor<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise hyperbolic sine of floating-point tiles.
    ///
    /// Returns a tile where each element is the hyperbolic sine of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, 1.0, 2.0, ...]
    /// let result = sinh(x); // [0.0, 1.175..., 3.626..., ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.sinh", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sinh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise hyperbolic cosine of floating-point tiles.
    ///
    /// Returns a tile where each element is the hyperbolic cosine of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, 1.0, 2.0, ...]
    /// let result = cosh(x); // [1.0, 1.543..., 3.762..., ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.cosh", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cosh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise hyperbolic tangent of floating-point tiles.
    ///
    /// Returns a tile where each element is the hyperbolic tangent of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, 1.0, 2.0, ...]
    /// let result = tanh(x); // [0.0, 0.761..., 0.964..., ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.tanh", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn tanh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise base-2 exponential (2^x) of floating-point tiles.
    ///
    /// Returns a tile where each element is 2 raised to the power of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [0.0, 1.0, 2.0, ...]
    /// let result = exp2(x, ftz::Disabled); // [1.0, 2.0, 4.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.exp2", params=["x"], static_params=["ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exp2<E: ElementType, const S: [i32; N], F: ftz::Mode>(
        x: Tile<E, S>,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Computes element-wise base-2 logarithm of floating-point tiles.
    ///
    /// Returns a tile where each element is the base-2 logarithm of the
    /// corresponding input element.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [1.0, 2.0, 4.0, ...]
    /// let result = log2(x); // [0.0, 1.0, 2.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.log2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn log2<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Fused multiply-add operation: `result = lhs * rhs + acc`.
    ///
    /// Computes the multiply-add as a single operation with no intermediate rounding,
    /// which is more accurate and faster than separate multiply and add operations.
    ///
    /// ## Parameters
    ///
    /// - `lhs`: Left-hand side tile
    /// - `rhs`: Right-hand side tile
    /// - `acc`: Accumulator tile
    /// - `rounding_mode`: Rounding mode string (e.g., "nearest_even", "positive_inf", "negative_inf", "nearest_int_to_zero", "approx")
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...; // [1.0, 2.0, 3.0, ...]
    /// let b: Tile<f32, {[128]}> = ...; // [2.0, 3.0, 4.0, ...]
    /// let c: Tile<f32, {[128]}> = ...; // [1.0, 1.0, 1.0, ...]
    /// let result = fma_op(a, b, c, "nearest_even"); // [3.0, 7.0, 13.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.fma", params=["lhs", "rhs", "acc"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn fma<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        acc: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Power operation: `result = source ^ exponent`.
    ///
    /// Computes the element-wise power of the source raised to the exponent.
    /// Both operands must have the same shape.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let x: Tile<f32, {[128]}> = ...; // [2.0, 3.0, 4.0, ...]
    /// let y: Tile<f32, {[128]}> = ...; // [2.0, 3.0, 2.0, ...]
    /// let result = pow(x, y); // [4.0, 27.0, 16.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.pow", params=["source", "exponent"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn pow<E: ElementType, const S: [i32; N]>(
        source: Tile<E, S>,
        exponent: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floating-point maximum.
    ///
    /// Computes the maximum of two floating-point tiles element-wise.
    /// Returns the larger of the two values. `-0.0` is considered less than `+0.0`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...; // [1.0, 5.0, 3.0, ...]
    /// let b: Tile<f32, {[128]}> = ...; // [2.0, 4.0, 6.0, ...]
    /// let result = maxf(a, b); // [2.0, 5.0, 6.0, ...]
    /// ```
    ///
    #[cuda_tile::op(name="cuda_tile.maxf", params=["lhs", "rhs"], static_params=["nan={Enabled: propagate_nan=unit}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn maxf<E: ElementType, const S: [i32; N], P: nan::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        nan: P,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floating-point minimum.
    ///
    /// Computes the minimum of two floating-point tiles element-wise.
    /// Returns the smaller of the two values. `-0.0` is considered less than `+0.0`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<f32, {[128]}> = ...; // [1.0, 5.0, 3.0, ...]
    /// let b: Tile<f32, {[128]}> = ...; // [2.0, 4.0, 6.0, ...]
    /// let result = minf(a, b, nan::Disabled, ftz::Disabled); // [1.0, 4.0, 3.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.minf", params=["lhs", "rhs"], static_params=["nan={Enabled: propagate_nan=unit}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn minf<E: ElementType, const S: [i32; N], P: nan::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        nan: P,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floating-point addition with explicit rounding and FTZ control.
    #[cuda_tile::op(name="cuda_tile.addf", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floating-point subtraction with explicit rounding and FTZ control.
    #[cuda_tile::op(name="cuda_tile.subf", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn subf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floating-point multiplication with explicit rounding and FTZ control.
    #[cuda_tile::op(name="cuda_tile.mulf", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mulf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floating-point division with explicit rounding and FTZ control.
    #[cuda_tile::op(name="cuda_tile.divf", params=["lhs", "rhs"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn divf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Conditional selection operation.
    ///
    /// Returns `val_if_true` where `cond` is true, otherwise returns `val_if_false`.
    /// This is equivalent to the ternary operator: `cond ? val_if_true : val_if_false`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let cond: Tile<bool, {[128]}> = ...; // [true, false, true, ...]
    /// let a: Tile<f32, {[128]}> = ...; // [1.0, 2.0, 3.0, ...]
    /// let b: Tile<f32, {[128]}> = ...; // [9.0, 8.0, 7.0, ...]
    /// let result = select(cond, a, b); // [1.0, 8.0, 3.0, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.select", params=["cond", "val_if_true", "val_if_false"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn select<E: ElementType, const S: [i32; N]>(
        cond: Tile<bool, S>,
        val_if_true: Tile<E, S>,
        val_if_false: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Bitwise AND operation on integer tiles.
    ///
    /// Computes the element-wise bitwise AND of two integer tiles.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i64, {[128]}> = ...; // [0b1010, 0b1100, ...]
    /// let b: Tile<i64, {[128]}> = ...; // [0b1100, 0b1010, ...]
    /// let result = andi(a, b); // [0b1000, 0b1000, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.andi", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn andi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Bitwise OR operation on integer tiles.
    ///
    /// Computes the element-wise bitwise OR of two integer tiles.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i64, {[128]}> = ...; // [0b1010, 0b1100, ...]
    /// let b: Tile<i64, {[128]}> = ...; // [0b0101, 0b0011, ...]
    /// let result = ori(a, b); // [0b1111, 0b1111, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.ori", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ori<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Bitwise XOR operation on integer tiles.
    ///
    /// Computes the element-wise bitwise XOR of two integer tiles.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i64, {[128]}> = ...; // [0b1010, 0b1100, ...]
    /// let b: Tile<i64, {[128]}> = ...; // [0b1100, 0b1010, ...]
    /// let result = xori(a, b); // [0b0110, 0b0110, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.xori", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn xori<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Left bit shift operation on integer tiles.
    ///
    /// Shifts the bits of `lhs` left by `rhs` positions, filling with zeros.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i64, {[128]}> = ...; // [1, 2, 4, ...]
    /// let b: Tile<i64, {[128]}> = ...; // [2, 3, 1, ...]
    /// let result = shli(a, b); // [4, 16, 8, ...]
    /// ```
    #[cuda_tile::op(name="cuda_tile.shli", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn shli<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Right bit shift operation on integer tiles.
    ///
    /// Shifts the bits of `lhs` right by `rhs` positions.
    /// For signed integers, performs arithmetic shift (sign extension).
    /// For unsigned integers, performs logical shift (zero fill).
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i64, {[128]}> = ...; // [8, 16, 32, ...]
    /// let b: Tile<i64, {[128]}> = ...; // [2, 3, 1, ...]
    /// let result = shri(a, b); // [2, 2, 16, ...]
    /// ```
    ///
    /// Note: Signedness is automatically inferred from the Rust element type:
    /// - u32/u64 → logical shift (zero fill)
    /// - i32/i64 and others → arithmetic shift (sign extension)
    #[cuda_tile::op(name="cuda_tile.shri", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn shri<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Bitcast operation - reinterprets bits as a different type.
    ///
    /// Reinterprets the bit representation of the source as the target type
    /// without changing the underlying bits. The source and target types must
    /// have the same total size in bits.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<u32, {[128]}> = ...; // bit pattern 0x40400000
    /// let result: Tile<f32, {[128]}> = bitcast(a); // interprets as 3.0
    /// ```
    #[cuda_tile::op(name="cuda_tile.bitcast", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn bitcast<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        source: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Element-wise integer maximum.
    ///
    /// Returns the element-wise maximum of two integer tiles. The signedness
    /// attribute determines whether the comparison is signed or unsigned.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let a: Tile<i64, {[128]}> = ...; // [1, -5, 3, ...]
    /// let b: Tile<i64, {[128]}> = ...; // [2, -3, 1, ...]
    /// let result = maxi(a, b); // [2, -3, 3, ...]
    /// ```
    ///
    /// ## Notes
    ///
    /// - For signed comparison: treats values as signed integers (e.g., -1 < 0)
    /// - For unsigned comparison: treats values as unsigned integers (e.g., 0xFF > 0x01)
    ///
    /// Note: Signedness is automatically inferred from the Rust element type (u32/u64 → unsigned, others → signed)
    #[cuda_tile::op(name="cuda_tile.maxi", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn maxi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Multiply high - returns upper bits of integer multiplication.
    ///
    /// Computes the element-wise product of two integer tiles and returns
    /// the high bits of the result. This is useful for computing the upper
    /// part of a double-width multiplication.
    ///
    /// ## Semantics
    ///
    /// For `N`-bit integers:
    /// - Performs `2N`-bit multiplication: `x[i] * y[i]`
    /// - Returns the upper `N` bits of the result
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // For 32-bit integers:
    /// let a: Tile<i32, {[128]}> = ...; // [0x10000, 0x20000, ...]
    /// let b: Tile<i32, {[128]}> = ...; // [0x10000, 0x10000, ...]
    /// let result = mulhii(a, b);
    /// // a[0] * b[0] = 0x100000000 (64-bit), upper 32 bits = 0x1
    /// // a[1] * b[1] = 0x200000000 (64-bit), upper 32 bits = 0x2
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Fixed-point arithmetic
    /// - Computing 128-bit products from 64-bit inputs
    /// - Efficient division approximations
    #[cuda_tile::op(name="cuda_tile.mulhii", params=["x", "y"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mulhii<E: ElementType, const S: [i32; N]>(x: Tile<E, S>, y: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Integer extension - widen integer type with sign/zero extension.
    ///
    /// Converts integers from a narrower to a wider type. The signedness
    /// attribute determines whether to perform sign extension or zero extension.
    ///
    /// ## Semantics
    ///
    /// - **Signed extension**: Preserves the sign by replicating the sign bit
    /// - **Zero extension**: Fills upper bits with zeros
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Sign extension: i32 -> i64
    /// let a: Tile<i32, {[128]}> = ...; // [-1, 127, ...]
    /// let result: Tile<i64, {[128]}> = exti(a);
    /// // [-1i32 extends to -1i64 (0xFFFFFFFFFFFFFFFF)]
    /// // [127i32 extends to 127i64 (0x000000000000007F)]
    /// ```
    ///
    /// ## Notes
    ///
    /// - The target type must be wider than the source type
    /// - Signedness is automatically inferred from the source Rust element type
    /// - Shape remains the same, only element type changes
    #[cuda_tile::op(name="cuda_tile.exti", params=["from"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exti<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        from: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer truncation - narrow integer type by discarding upper bits.
    ///
    /// Converts integers from a wider to a narrower type by discarding the
    /// upper bits. This is the inverse operation of `exti`.
    ///
    /// ## Semantics
    ///
    /// - Discards upper bits that don't fit in the target type
    /// - Preserves lower bits (modulo arithmetic behavior)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // i64 -> i32 truncation
    /// let a: Tile<i64, {[128]}> = ...; // [0x100000001, 0xFFFFFFFF, ...]
    /// let result: Tile<i32, {[128]}> = trunci(a);
    /// // [0x100000001 truncates to 0x00000001]
    /// // [0xFFFFFFFF truncates to 0xFFFFFFFF]
    /// ```
    ///
    /// ## Notes
    ///
    /// - The target type must be narrower than the source type
    /// - Value may be lost if the source value doesn't fit in target type
    /// - Shape remains the same, only element type changes
    #[cuda_tile::op(name="cuda_tile.trunci", params=["from"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn trunci<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        from: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Convert integer to pointer.
    ///
    /// Converts a tile of integer values to a tile of pointer values.
    /// This is used for pointer arithmetic and address manipulation.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let addresses: Tile<u64, {[128]}> = ...; // [0x1000, 0x1004, ...]
    /// let pointers: PointerTile<*mut f32, {[128]}> = int_to_ptr(addresses);
    /// ```
    ///
    /// ## Notes
    ///
    /// - Typically used with `u64` or `i64` integer types
    /// - Result is a `PointerTile` that can be used with memory operations
    /// - This operation is useful for computing dynamic memory addresses
    #[cuda_tile::op(name="cuda_tile.int_to_ptr", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn int_to_ptr<SRC_T: ElementType, PTR_T: ElementType, const S: [i32; N]>(
        source: Tile<SRC_T, S>,
    ) -> PointerTile<*mut PTR_T, S> {
        unreachable!()
    }

    /// Convert pointer to integer.
    ///
    /// Converts a tile of pointer values to a tile of integer values.
    /// This is used for pointer arithmetic and address comparison.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let pointers: PointerTile<*mut f32, {[128]}> = ...;
    /// let addresses: Tile<u64, {[128]}> = ptr_to_int(pointers);
    /// // Now addresses can be used for arithmetic operations
    /// ```
    ///
    /// ## Notes
    ///
    /// - Result is typically `u64` or `i64` integer type
    /// - Useful for computing pointer offsets or alignments
    /// - This operation preserves pointer provenance information for the compiler
    #[cuda_tile::op(name="cuda_tile.ptr_to_int", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ptr_to_int<E: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Cast pointer type - reinterpret pointers as pointing to different type.
    ///
    /// Converts a tile of pointers from one element type to another without
    /// changing the address values. This is similar to a C-style pointer cast.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let float_ptrs: PointerTile<*mut f32, {[128]}> = ...;
    /// let int_ptrs: PointerTile<*mut i32, {[128]}> = ptr_to_ptr(float_ptrs);
    /// // Same addresses, but now interpreted as pointing to i32
    /// ```
    ///
    /// ## Notes
    ///
    /// - Only changes the pointed-to type, not the address
    /// - Distinct from `int_to_ptr` and `ptr_to_int` for provenance tracking
    /// - Use with caution: accessing memory through wrong pointer type is unsafe
    #[cuda_tile::op(name="cuda_tile.ptr_to_ptr", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ptr_to_ptr<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut EIn, S>,
    ) -> PointerTile<*mut EOut, S> {
        unreachable!()
    }

    /// Extract a subtile from a tile.
    ///
    /// Extracts a subtile from a source tile at specified indices. The result
    /// shape must evenly divide the source shape.
    ///
    /// ## Semantics
    ///
    /// The indices specify which slice to extract, not the offsets. Only full
    /// slices can be extracted - the result shape must evenly divide the source.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Extract 4-element slices from an 8-element tile
    /// let source: Tile<f32, {[8]}> = ...; // [1, 2, 3, 4, 5, 6, 7, 8]
    /// let idx0: Tile<i32, {[]}> = scalar_to_tile(0i32);
    /// let slice0: Tile<f32, {[4]}> = extract(source, [idx0]);
    /// // slice0 = [1, 2, 3, 4]
    ///
    /// let idx1: Tile<i32, {[]}> = scalar_to_tile(1i32);
    /// let slice1: Tile<f32, {[4]}> = extract(source, [idx1]);
    /// // slice1 = [5, 6, 7, 8]
    /// ```
    ///
    /// ## Notes
    ///
    /// - Result shape must evenly divide source shape in each dimension
    /// - Indices specify slice number, not element offset
    /// - Out-of-bounds indices result in undefined behavior
    /// - Extracted slices are non-overlapping for unique indices
    #[cuda_tile::op(name="cuda_tile.extract", params=["source", "...indices"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn extract<E: ElementType, const SIn: [i32; N], const SOut: [i32; N]>(
        source: Tile<E, SIn>,
        indices: [Tile<i32, { [] }>; N],
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /// Concatenate two tiles along a specified dimension.
    ///
    /// Joins two tiles together along a specified dimension. The tiles must have
    /// the same shape in all dimensions except the concatenation dimension.
    ///
    /// ## Semantics
    ///
    /// The `dim` parameter specifies which dimension to concatenate along (0-indexed).
    /// The result dimension size equals the sum of the two input dimension sizes.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // 1D concatenation
    /// let a: Tile<f32, {[4]}> = ...; // [1, 2, 3, 4]
    /// let b: Tile<f32, {[4]}> = ...; // [5, 6, 7, 8]
    /// let result: Tile<f32, {[8]}> = cat(a, b, 0);
    /// // result = [1, 2, 3, 4, 5, 6, 7, 8]
    ///
    /// // 2D concatenation along rows (dim=0)
    /// let a: Tile<f32, {[2, 3]}> = ...; // [[1, 2, 3],
    ///                                    //  [4, 5, 6]]
    /// let b: Tile<f32, {[2, 3]}> = ...; // [[7, 8, 9],
    ///                                    //  [10, 11, 12]]
    /// let result: Tile<f32, {[4, 3]}> = cat(a, b, 0);
    /// // result = [[1, 2, 3],
    /// //           [4, 5, 6],
    /// //           [7, 8, 9],
    /// //           [10, 11, 12]]
    /// ```
    ///
    /// ## Notes
    ///
    /// - All dimensions except `dim` must match between inputs
    /// - `dim` must be a valid dimension index (0 ≤ dim < rank)
    /// - Result shape is same as inputs except along concatenation dimension
    #[cuda_tile::op(name="cuda_tile.cat", params=["lhs", "rhs"], attribute_params=["dim:integer"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cat<E: ElementType, const SLhs: [i32; N], const SRhs: [i32; N], const SOut: [i32; N]>(
        lhs: Tile<E, SLhs>,
        rhs: Tile<E, SRhs>,
        dim: i32,
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /* High-level Functions */

    /// Broadcasts a scalar value to a tile of the specified shape.
    ///
    /// This is the underlying implementation for the [`BroadcastScalar`] trait.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tile = broadcast_scalar(3.14f32, const_shape![128]);
    /// // tile contains 128 copies of 3.14
    /// ```
    #[cuda_tile::variadic_op(N = 6)]
    pub fn broadcast_scalar<E: ElementType, const S: [i32; N]>(
        x: E,
        shape: Shape<S>,
    ) -> Tile<E, S> {
        let ones_shape: Shape<{ [1; N] }> = Shape::<{ [1; N] }> { dims: &[1i32; N] };
        let tile_x: Tile<E, { [] }> = scalar_to_tile(x);
        tile_x.reshape(ones_shape).broadcast(shape)
    }

    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_tile<E: ElementType, const S: [i32; N], const R: [i32; N]>(
        x: &Tensor<E, S>,
        tile_shape: Shape<R>,
        idx: [i32; N],
    ) -> Tile<E, R> {
        let tensor_token: Token = get_tensor_token(x);
        let x_partition: Partition<E, R> =
            make_partition_view_padded(x, tile_shape, "zero", tensor_token);
        let tile_x: Tile<E, R> = load_from_view(&x_partition, idx, None, false);
        tile_x
    }

    /// Loads the entire tensor into a tile.
    ///
    /// This is used when the tensor shape matches the processing tile size.
    /// The tensor must fit entirely within a single thread block's processing capacity.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor: &mut Tensor<f32, {[128]}> = ...;
    /// let tile = load_tile(tensor);
    /// ```
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_tile_mut<E: ElementType, const S: [i32; N]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let tile_shape: Shape<S> = y.shape();
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut_padded(y, tile_shape, "zero", tensor_token) };
        let tile_y: Tile<E, S> = unsafe { load_from_view_mut(&y_partition, [0i32; N]) };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
        tile_y
    }

    /// Stores a tile back to a tensor.
    ///
    /// Writes the tile data to global memory at the location of the tensor.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// let tensor: &mut Tensor<f32, {[128]}> = ...;
    /// let tile: Tile<f32, {[128]}> = compute_result();
    /// store_tile(tensor, tile);
    /// ```
    #[cuda_tile::variadic_op(N = 6)]
    pub fn store_tile<E: ElementType, const S: [i32; N]>(y: &mut Tensor<E, S>, result: Tile<E, S>) {
        let tile_shape: Shape<S> = y.shape();
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut_padded(y, tile_shape, "zero", tensor_token) };
        unsafe { store_to_view_mut(&mut y_partition, result, [0i32; N], None, false) };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
    }

    /// Loads a tile from a dynamically-sized 2D input tensor using the output tensor's shape.
    ///
    /// This helper function loads a tile from `x` that corresponds to the current thread
    /// block's position, using `y`'s static shape to determine tile size. Common pattern
    /// for kernels where the output shape determines processing granularity.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// fn kernel(
    ///     output: &mut Tensor<f32, {[64, 64]}>,
    ///     input: &Tensor<f32, {[-1, -1]}>,
    /// ) {
    ///     let tile = load_tile_like_2d(input, output);
    ///     // Process tile corresponding to this block
    /// }
    /// ```
    pub fn load_tile_like_2d<E: ElementType, const S: [i32; 2]>(
        x: &Tensor<E, { [-1, -1] }>,
        y: &mut Tensor<E, S>,
    ) -> Tile<E, S> {
        // Load a tile of x from a statically shaped mutable tensor like y.
        // Since y is a mutable ref, it is partitioned and mapped to pid.
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = y.shape();
        let tensor_token: Token = get_tensor_token(x);
        let x_partition: Partition<E, S> = make_partition_view(x, tile_shape, tensor_token);
        let tile_x: Tile<E, S> = load_from_view(&x_partition, [pid.0, pid.1], None, false);
        tile_x
    }

    /// Loads a tile from a dynamically-sized 1D input tensor using the output tensor's shape.
    ///
    /// This is the 1D version of [`load_tile_like_2d`]. It loads a tile from `x` that
    /// corresponds to the current thread block's position, using `y`'s static shape to
    /// determine tile size.
    ///
    /// ## Type Parameters
    ///
    /// - `E1`: Element type of input tensor (can differ from output)
    /// - `E2`: Element type of output tensor
    ///
    /// ## Parameters
    ///
    /// - `x`: Input tensor with dynamic shape `{[-1]}`
    /// - `y`: Output tensor with static shape (used to determine tile size)
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// fn vector_process(
    ///     output: &mut Tensor<f32, {[128]}>,
    ///     input: &Tensor<f32, {[-1]}>,
    /// ) {
    ///     let tile = load_tile_like_1d(input, output);
    ///     // Process 128-element tile corresponding to this block
    ///     let result = tile * 2.0;
    ///     output.store(result);
    /// }
    /// ```
    pub fn load_tile_like_1d<E1: ElementType, E2: ElementType, const S: [i32; 1]>(
        x: &Tensor<E1, { [-1] }>,
        y: &mut Tensor<E2, S>,
    ) -> Tile<E1, S> {
        // Load a tile of x from a statically shaped mutable tensor like y.
        // Since y is a mutable ref, it is partitioned and mapped to pid.
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = y.shape();
        let tensor_token: Token = get_tensor_token(x);
        let x_partition: Partition<E1, S> = make_partition_view(x, tile_shape, tensor_token);
        let tile_x: Tile<E1, S> = load_from_view(&x_partition, [pid.0], None, false);
        tile_x
    }

    // TODO (hme): Need to add a way to specify individual instances of N for pid-dependent operations.
    // pub fn load_tile_like<const S: [i32; 2]>(x: &Tensor<f32, {[-1; 2]}>, y: &mut Tensor<f32, S>) -> Tile<f32, S> {
    //     // Load a tile of x from a statically shaped mutable tensor like y.
    //     // Since y is a mutable ref, it is partitioned and mapped to pid.
    //     let pid: (i32, i32, i32) = get_tile_block_id();
    //     // TODO (hme): Need to add support for N in function block.
    //     //  pid should be provided up to N somehow.
    //     let pids: [i32; 2] = [pid.0, pid.1];
    //     let tile_shape: Shape<S> = y.shape();
    //     let x_partition: Partition<f32, S> = make_partition_view(&x, tile_shape);
    //     let tile_x: Tile<f32, S> = load_from_view(&x_partition, pids);
    //     tile_x
    // }

    /// Assume that a value is divisible by a constant.
    ///
    /// Tells the compiler that the input value (or all elements if a tile) is divisible
    /// by `DIVISOR`. This enables optimizations like strength reduction (replacing division
    /// with cheaper operations) and loop unrolling.
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Can be a scalar integer (`i32`) or a tile of integers (`Tile<i32, S>`)
    /// - `DIVISOR`: The divisor constant (compile-time)
    ///
    /// ## Safety
    ///
    /// Undefined behavior if the assumption is violated. The compiler may generate code
    /// that produces incorrect results or crashes if any element is not divisible by `DIVISOR`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Tell compiler that dimension is divisible by tile size
    /// let dim = get_dimension();
    /// let aligned_dim = unsafe { assume_div_by::<_, 64>(dim) };
    /// // Compiler can now optimize loops with this knowledge
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Asserting alignment properties (e.g., pointer offsets divisible by 8)
    /// - Loop trip counts divisible by unroll factor
    /// - Array dimensions divisible by tile size
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_div_by<T, const DIVISOR: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assume complex divisibility pattern along specific dimensions.
    ///
    /// Tells the compiler that every `every` elements along dimension `along` are
    /// divisible by `divisor`. This is useful for asserting strided memory access patterns.
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Typically a tile or shaped value
    /// - `divisor`: The divisor constant
    /// - `every`: Spacing between elements that satisfy the divisibility property
    /// - `along`: Which dimension to apply the pattern to
    ///
    /// ## Safety
    ///
    /// Undefined behavior if the assumption is violated at runtime.
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let tile: Tile<i32, {[128, 64]}> = ...;
    /// // Every 8 elements along dimension 1 are divisible by 16
    /// let tile = unsafe { assume_div_by_every_along::<_, 16, 8, 1>(tile) };
    /// ```
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_div_by_every_along<
        T,
        const divisor: i32,
        const every: i32,
        const along: i32,
    >(
        x: T,
    ) -> T {
        unreachable!()
    }

    /// Assume that a value has a lower bound (inclusive).
    ///
    /// Tells the compiler that the input value (or all elements if a tile) is greater than
    /// or equal to `LOWER`. This enables optimizations like:
    /// - Eliminating unnecessary bounds checks
    /// - Simplifying comparison operations
    /// - Enabling unsigned optimizations when LOWER >= 0
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Can be a scalar integer or a tile of integers
    /// - `LOWER`: The minimum value (inclusive, compile-time constant)
    ///
    /// ## Safety
    ///
    /// Undefined behavior if any element is less than `LOWER`. The compiler may generate
    /// optimized code that produces incorrect results for out-of-range values.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Assert that array index is non-negative
    /// let idx = compute_index();
    /// let safe_idx = unsafe { assume_bounds_lower::<_, 0>(idx) };
    /// // Compiler knows idx >= 0, can optimize accordingly
    ///
    /// // Assert all tile elements are positive
    /// let tile: Tile<i32, {[128]}> = ...;
    /// let positive = unsafe { assume_bounds_lower::<_, 1>(tile) };
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Asserting non-negativity (LOWER = 0) after validation
    /// - Expressing value ranges after clamping
    /// - Enabling vectorization of loops with known ranges
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds_lower<T, const LOWER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assume that a value has an upper bound (inclusive).
    ///
    /// Tells the compiler that the input value (or all elements if a tile) is less than
    /// or equal to `UPPER`. This enables optimizations like:
    /// - Eliminating unnecessary bounds checks
    /// - Using smaller integer types internally
    /// - Better branch prediction
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Can be a scalar integer or a tile of integers
    /// - `UPPER`: The maximum value (inclusive, compile-time constant)
    ///
    /// ## Safety
    ///
    /// Undefined behavior if any element is greater than `UPPER`.
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let idx = compute_bounded_index();
    /// let bounded_idx = unsafe { assume_bounds_upper::<_, 255>(idx) };
    /// // Compiler knows idx <= 255, can use 8-bit operations
    /// ```
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds_upper<T, const UPPER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assume that a value is within a specific range (inclusive bounds).
    ///
    /// Combines lower and upper bound assumptions. Tells the compiler that the input
    /// value (or all elements if a tile) satisfies `LOWER <= value <= UPPER`.
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Can be a scalar integer or a tile of integers
    /// - `LOWER`: The minimum value (inclusive, compile-time constant)
    /// - `UPPER`: The maximum value (inclusive, compile-time constant)
    ///
    /// ## Safety
    ///
    /// Undefined behavior if any element is outside the range `[LOWER, UPPER]`.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // Assert value is in valid byte range
    /// let byte_val = compute_value();
    /// let clamped = unsafe { assume_bounds::<_, 0, 255>(byte_val) };
    ///
    /// // Assert tile elements are in valid range
    /// let tile: Tile<i32, {[128]}> = ...;
    /// let bounded = unsafe { assume_bounds::<_, -100, 100>(tile) };
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Asserting values after clamping/saturation
    /// - Expressing valid ranges for lookup table indices
    /// - Enabling optimizations based on value ranges
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds<T, const LOWER: i32, const UPPER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assume that elements are identical within groups along dimension 0 (1D).
    ///
    /// Tells the compiler that within every consecutive group of `GROUP0` elements,
    /// all elements have the same value. This enables optimizations like:
    /// - Loading only one element per group (broadcast the rest)
    /// - Eliminating redundant computations on identical values
    /// - Better vectorization opportunities
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Typically a 1D tile (`Tile<E, {[N]}>`)
    /// - `GROUP0`: Size of groups with identical elements
    ///
    /// ## Semantics
    ///
    /// For a tile with elements `[a₀, a₁, a₂, ..., a_{N-1}]`, this asserts:
    /// - Elements `[0..GROUP0)` are all equal
    /// - Elements `[GROUP0..2*GROUP0)` are all equal
    /// - And so on for each group
    ///
    /// ## Safety
    ///
    /// Undefined behavior if the assumption is violated. The compiler may generate code
    /// that loads only one element per group, producing incorrect results if elements differ.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // After broadcasting a value
    /// let tile: Tile<i32, {[128]}> = scalar.broadcast(const_shape![128]);
    /// // Tell compiler all elements are identical
    /// let optimized = unsafe { assume_same_elements_1d::<_, 128>(tile) };
    ///
    /// // After grouping by blocks of 4
    /// let data: Tile<f32, {[64]}> = ...;
    /// // Groups of 4 consecutive elements are identical
    /// let grouped = unsafe { assume_same_elements_1d::<_, 4>(data) };
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - After broadcasting operations where all elements are the same
    /// - When processing block-replicated data
    /// - After computing values that are constant within groups
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_1d<T, const GROUP0: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assume that elements are identical within groups along each dimension (2D).
    ///
    /// Tells the compiler that within rectangular groups of size `GROUP0 × GROUP1`,
    /// all elements have the same value. This is the 2D extension of `assume_same_elements_1d`.
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Typically a 2D tile (`Tile<E, {[M, N]}>`)
    /// - `GROUP0`: Group size along dimension 0 (rows)
    /// - `GROUP1`: Group size along dimension 1 (columns)
    ///
    /// ## Semantics
    ///
    /// For a 2D tile, this asserts that within each `GROUP0 × GROUP1` rectangular block,
    /// all elements are identical. The tile is partitioned into non-overlapping blocks.
    ///
    /// ## Safety
    ///
    /// Undefined behavior if any element within a group differs from others in that group.
    ///
    /// ## Examples
    ///
    /// ```rust,ignore
    /// // After 2D broadcast
    /// let tile: Tile<f32, {[8, 16]}> = ...;
    /// // Groups of 2×4 have identical elements
    /// let grouped = unsafe { assume_same_elements_2d::<_, 2, 4>(tile) };
    ///
    /// // Matrix with block-constant structure
    /// let matrix: Tile<i32, {[64, 64]}> = ...;
    /// // Each 8×8 block is constant
    /// let block_const = unsafe { assume_same_elements_2d::<_, 8, 8>(matrix) };
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - Block-diagonal matrices where blocks are constant
    /// - After 2D broadcasting/replication
    /// - Tiled convolution with constant kernels per region
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_2d<T, const GROUP0: i32, const GROUP1: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assume that elements are identical within groups along each dimension (3D).
    ///
    /// Tells the compiler that within cubic groups of size `GROUP0 × GROUP1 × GROUP2`,
    /// all elements have the same value. This is the 3D extension of `assume_same_elements_2d`.
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Typically a 3D tile (`Tile<E, {[D0, D1, D2]}>`)
    /// - `GROUP0`: Group size along dimension 0
    /// - `GROUP1`: Group size along dimension 1
    /// - `GROUP2`: Group size along dimension 2
    ///
    /// ## Safety
    ///
    /// Undefined behavior if any element within a group differs from others in that group.
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let volume: Tile<f32, {[32, 32, 32]}> = ...;
    /// // Each 4×4×4 block has identical elements
    /// let blocked = unsafe { assume_same_elements_3d::<_, 4, 4, 4>(volume) };
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - 3D volume processing with block-constant regions
    /// - Batched 2D operations where each batch is constant
    /// - Spatiotemporal data with temporal replication
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_3d<
        T,
        const GROUP0: i32,
        const GROUP1: i32,
        const GROUP2: i32,
    >(
        x: T,
    ) -> T {
        unreachable!()
    }

    /// Assume that elements are identical within groups along each dimension (4D).
    ///
    /// Tells the compiler that within 4D hypercubic groups of size
    /// `GROUP0 × GROUP1 × GROUP2 × GROUP3`, all elements have the same value.
    /// This is the 4D extension of `assume_same_elements_3d`.
    ///
    /// ## Type Parameters
    ///
    /// - `T`: Typically a 4D tile (`Tile<E, {[D0, D1, D2, D3]}>`)
    /// - `GROUP0`: Group size along dimension 0
    /// - `GROUP1`: Group size along dimension 1
    /// - `GROUP2`: Group size along dimension 2
    /// - `GROUP3`: Group size along dimension 3
    ///
    /// ## Safety
    ///
    /// Undefined behavior if any element within a group differs from others in that group.
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let tensor: Tile<f32, {[16, 16, 16, 16]}> = ...;
    /// // Each 2×2×2×2 hyperblock has identical elements
    /// let blocked = unsafe { assume_same_elements_4d::<_, 2, 2, 2, 2>(tensor) };
    /// ```
    ///
    /// ## Use Cases
    ///
    /// - High-dimensional tensor operations with block structure
    /// - Batched 3D operations where batches are constant
    /// - Neural network layers with structured sparsity
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_4d<
        T,
        const GROUP0: i32,
        const GROUP1: i32,
        const GROUP2: i32,
        const GROUP3: i32,
    >(
        x: T,
    ) -> T {
        unreachable!()
    }

    /* TensorArray */
    // TODO (hme): Add support for this sort of type.
    // #[cuda_tile::variadic_struct(N = 6)]
    // struct TensorArray<E: ElementType, const D: [i32; N]> {
    //     _type: PhantomData<E>,
    // }

    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub unsafe fn load_tensor<T: ElementType, const S: [i32; N], const R: [i32; M]>(
        dst: &Tensor<i64, S>,
        idx: [i32; N],
        shape: Shape<R>,
        strides: Array<{ [-1; M] }>,
    ) -> Tensor<T, R> {
        let dims: &[i32] = &[];
        let ones_shape: Shape<{ [1; N] }> = Shape::<{ [1; N] }> { dims: dims };
        let dst_part: Partition<i64, { [1; N] }> = dst.partition(ones_shape);
        let dst_ptr_int: Tile<i64, { [1; N] }> = dst_part.load(idx);
        let dst_ptr_int: Tile<i64, { [] }> = dst_ptr_int.reshape(const_shape![]);
        let dst_ptr: PointerTile<*mut T, { [] }> = int_to_ptr(dst_ptr_int);
        let dst_tensor: Tensor<T, R> =
            unsafe { make_tensor_view(dst_ptr, shape, strides, new_token_unordered()) };
        dst_tensor
    }

    #[cuda_tile::compiler_op(name = "shape")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn permute_array<const I: [i32; N]>(source: [i32; N], permutation: Array<I>) -> [i32; N] {
        unreachable!()
    }

    // #[cuda_tile::op(name="cuda_tile.permute", params=["source"], attribute_params=["permutation:array"])]
    // #[cuda_tile::variadic_op(N = 6)]
    // pub fn permute<E: ElementType, const A: [i32; N], const I: [i32; N], const R: [i32; N]>(
    //     source: Tile<E, A>,
    //     permutation: Array<I>,
    // ) -> Tile<E, R> {
    //     unreachable!()
    // }
}
