/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 Notes:
 - Parameter names should not be changed without careful refactoring.
   They are sometimes used to generate corresponding Rust functions for cuda_tile operations.
*/

#![allow(nonstandard_style)]
#![allow(unused_variables)]

//! Core GPU kernel programming types and operations.
//!
//! Sections mirror the Tile IR documentation
//! (<https://docs.nvidia.com/cuda/tile-ir/latest/sections/>).
//! Doc comments here are intentionally terse — see Tile IR for op semantics.

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

/// Rounding mode for floating-point operations. Mirrors Tile IR `RoundingMode`.
pub mod rounding {
    pub trait Mode {}
    pub struct NearestEven;
    pub struct PositiveInf;
    pub struct NegativeInf;
    pub struct Zero;
    pub struct Approx;
    pub struct Full;
    impl Mode for NearestEven {}
    impl Mode for PositiveInf {}
    impl Mode for NegativeInf {}
    impl Mode for Zero {}
    impl Mode for Approx {}
    impl Mode for Full {}
}

/// NaN propagation for maxf/minf operations.
pub mod nan {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Atomic read-modify-write op selector. Mirrors Tile IR `AtomicRMWMode`.
pub mod atomic {
    pub trait Mode {}
    pub struct Add;
    pub struct AddF;
    pub struct And;
    pub struct Or;
    pub struct Xor;
    pub struct Max;
    pub struct Min;
    pub struct Umax;
    pub struct Umin;
    pub struct Xchg;
    impl Mode for Add {}
    impl Mode for AddF {}
    impl Mode for And {}
    impl Mode for Or {}
    impl Mode for Xor {}
    impl Mode for Max {}
    impl Mode for Min {}
    impl Mode for Umax {}
    impl Mode for Umin {}
    impl Mode for Xchg {}
}

/// Memory ordering, with per-op-family sub-traits restricting which variants
/// each op accepts (mirrors Tile IR `OnlyVariants` constraints).
pub mod ordering {
    pub trait Mode {}
    /// `load_*_tko` ops: `Weak`, `Relaxed`, `Acquire`.
    pub trait LoadMode: Mode {}
    /// `store_*_tko` ops: `Weak`, `Relaxed`, `Release`.
    pub trait StoreMode: Mode {}
    /// `atomic_*_tko` ops: `Relaxed`, `Acquire`, `Release`, `AcqRel`.
    pub trait AtomicMode: Mode {}

    pub struct Weak;
    pub struct Relaxed;
    pub struct Acquire;
    pub struct Release;
    pub struct AcqRel;

    impl Mode for Weak {}
    impl LoadMode for Weak {}
    impl StoreMode for Weak {}

    impl Mode for Relaxed {}
    impl LoadMode for Relaxed {}
    impl StoreMode for Relaxed {}
    impl AtomicMode for Relaxed {}

    impl Mode for Acquire {}
    impl LoadMode for Acquire {}
    impl AtomicMode for Acquire {}

    impl Mode for Release {}
    impl StoreMode for Release {}
    impl AtomicMode for Release {}

    impl Mode for AcqRel {}
    impl AtomicMode for AcqRel {}
}

/// Memory scope for atomics and load/store ordering. Mirrors Tile IR
/// `MemoryScope`.
pub mod scope {
    pub trait Mode {}
    pub struct TileBlock;
    pub struct Device;
    pub struct System;
    impl Mode for TileBlock {}
    impl Mode for Device {}
    impl Mode for System {}
}

/// Whether an op may use TMA (Tensor Memory Accelerator) when the hardware
/// supports it. Encodes Tile IR's `optimization_hints.allow_tma` hint.
pub mod tma {
    pub trait Mode {}
    pub struct Enabled;
    pub struct Disabled;
    impl Mode for Enabled {}
    impl Mode for Disabled {}
}

/// Latency hint (Tile IR `optimization_hints.latency`). Single value applied
/// across SM archs; per-arch dictionary form is deferred.
pub struct Latency<const CYCLES: u32>;

/// Integer-overflow behavior. Mirrors Tile IR `IntegerOverflow`.
pub mod overflow {
    pub trait Mode {}
    pub struct None;
    pub struct NoSignedWrap;
    pub struct NoUnsignedWrap;
    pub struct NoWrap;
    impl Mode for None {}
    impl Mode for NoSignedWrap {}
    impl Mode for NoUnsignedWrap {}
    impl Mode for NoWrap {}
}

/// Comparison predicate for `cmpi` / `cmpf`. Mirrors Tile IR
/// `ComparisonPredicate`.
pub mod predicate {
    pub trait Mode {}
    pub struct Equal;
    pub struct NotEqual;
    pub struct LessThan;
    pub struct LessThanOrEqual;
    pub struct GreaterThan;
    pub struct GreaterThanOrEqual;
    impl Mode for Equal {}
    impl Mode for NotEqual {}
    impl Mode for LessThan {}
    impl Mode for LessThanOrEqual {}
    impl Mode for GreaterThan {}
    impl Mode for GreaterThanOrEqual {}
}

/// Floating-point comparison ordering for `cmpf` (`Ordered` requires both
/// operands to be non-NaN; `Unordered` succeeds if either is NaN). Mirrors
/// Tile IR `ComparisonOrdering`. Distinct from memory `ordering`.
pub mod cmp_ordering {
    pub trait Mode {}
    pub struct Unordered;
    pub struct Ordered;
    impl Mode for Unordered {}
    impl Mode for Ordered {}
}

/// Padding marker for partition-view construction.
///
/// `None` omits the Tile IR `padding_value` type parameter. The other markers
/// map to Tile IR `PaddingValue` values for out-of-bounds reads through views.
pub mod padding {
    pub trait Mode {}
    /// Omit the partition-view `padding_value` type parameter.
    pub struct None;
    /// Pad out-of-bounds lanes with zero.
    pub struct Zero;
    /// Pad out-of-bounds lanes with negative zero.
    pub struct NegZero;
    /// Pad out-of-bounds lanes with NaN.
    pub struct Nan;
    /// Pad out-of-bounds lanes with positive infinity.
    pub struct PosInf;
    /// Pad out-of-bounds lanes with negative infinity.
    pub struct NegInf;
    impl Mode for None {}
    impl Mode for Zero {}
    impl Mode for NegZero {}
    impl Mode for Nan {}
    impl Mode for PosInf {}
    impl Mode for NegInf {}
}

/// Optional dimension map for partition views.
pub mod dim_map {
    pub trait Mode {}
    /// Omit the partition-view `dim_map` type parameter.
    pub struct Identity;
    impl Mode for Identity {}
}

/// Direction for `scan`. Encodes Tile IR's `reverse: BoolAttr`.
pub mod reverse {
    pub trait Mode {}
    pub struct Forward;
    pub struct Reverse;
    impl Mode for Forward {}
    impl Mode for Reverse {}
}

/// Signedness for ops where it's user-facing (`mmai`, `ftoi`, `itof`).
/// For ops like `divi`/`remi`/`maxi` the compiler infers signedness from
/// the operand element type.
pub mod signedness {
    pub trait Mode {}
    pub struct Signed;
    pub struct Unsigned;
    impl Mode for Signed {}
    impl Mode for Unsigned {}
}

/// cuTile core DSL surface.
///
/// Sections mirror the Tile IR documentation
/// (<https://docs.nvidia.com/cuda/tile-ir/latest/sections/>) and are
/// marked with banner comments below. See the linked Tile IR pages for
/// op semantics; doc comments here are intentionally terse.
#[cutile_macro::module(tile_rust_crate = true)]
pub mod core {

    pub use super::atomic;
    pub use super::cmp_ordering;
    pub use super::dim_map;
    pub use super::ftz;
    pub use super::nan;
    pub use super::ordering;
    pub use super::overflow;
    pub use super::padding;
    pub use super::predicate;
    pub use super::reverse;
    pub use super::rounding;
    pub use super::scope;
    pub use super::signedness;
    pub use super::tma;
    pub use super::Latency;
    pub use half::{bf16, f16};
    use std::marker::PhantomData;
    use std::ops;

    // ========================================================================
    // TYPES — Tile IR §5
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html
    // ========================================================================

    // ---- §5.1 ELEMENT TYPES ------------------------------------------------

    /// Marker trait for valid `Tile<E, …>` / `Tensor<E, …>` element types.
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
    /// Auto-implemented for every `ElementType`.
    pub trait Scalar {}
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["E"])]
    impl<E: ElementType> Scalar for E {}

    /// Method-form `scalar.broadcast(shape)` — auto-impl'd for every `ElementType`.
    #[cuda_tile::variadic_trait(N = 6)]
    pub trait BroadcastScalar<E: ElementType, const D: [i32; N]>
    where
        Self: ElementType,
    {
        fn broadcast(self, shape: Shape<D>) -> Tile<E, D>;
    }

    #[cuda_tile::variadic_trait_impl()]
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> BroadcastScalar<E, D> for E {
        fn broadcast(self, shape: Shape<D>) -> Tile<E, D> {
            broadcast_scalar(self, shape)
        }
    }

    // ---- §5.2 POINTERS -----------------------------------------------------

    /// Marker for GPU pointer types. Impl'd for `*mut E` where `E: ElementType`.
    pub trait Pointer {}
    #[cuda_tile::ty(name="!cuda_tile.tile", pointer_type="!cuda_tile.ptr", type_params=["!cuda_tile.ptr<E>"])]
    impl<E: ElementType> Pointer for *mut E {}
    // impl<E: ElementType> Pointer for *const E {}

    /// Tile of pointers — enables gather/scatter and indirect access.
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xP"])]
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct PointerTile<P: Pointer, const D: [i32; N]> {
        _type: PhantomData<P>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<P: Pointer, const D: [i32; N]> PointerTile<P, D> {
        pub fn offset_tile<I: ElementType>(self, offset: Tile<I, D>) -> PointerTile<P, D> {
            addptr_tile(self, offset)
        }
        pub fn offset(self, offset: i32) -> PointerTile<P, D> {
            addptr(self, offset)
        }
        pub fn broadcast<const R: [i32; N]>(self, shape: Shape<R>) -> PointerTile<P, R> {
            broadcast_ptr(self, shape)
        }
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn reshape<const R: [i32; M]>(self, shape: Shape<R>) -> PointerTile<P, R> {
            reshape_ptr(self, shape)
        }
    }

    // ---- §5.3 TENSOR TYPES (Tile, Tensor, Partition, PartitionMut) ---------

    /// Multi-dimensional array stored in registers / shared memory. The unit
    /// of compute inside a kernel.
    #[cuda_tile::ty(name="!cuda_tile.tile", type_params=["{D}xE"])]
    #[cuda_tile::variadic_struct(N = 6)]
    #[derive(Copy, Clone)]
    pub struct Tile<E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> Tile<E, D> {
        pub fn shape(&self) -> Shape<D> {
            unreachable!()
        }
        pub fn broadcast<const R: [i32; N]>(self, shape: Shape<R>) -> Tile<E, R> {
            broadcast(self, shape)
        }
        #[cuda_tile::variadic_impl_fn(M = 6)]
        pub fn reshape<const R: [i32; M]>(self, shape: Shape<R>) -> Tile<E, R> {
            reshape(self, shape)
        }
    }

    // Operator overloads on `Tile`. Element-wise; `*` is the Hadamard product
    // — for matrix multiply use `mma`.
    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Add<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn add(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Sub<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn sub(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Mul<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn mul(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const D: [i32; N]> ops::Div<Tile<E, D>> for Tile<E, D> {
        type Output = Tile<E, D>;
        fn div(self, _rhs: Tile<E, D>) -> Tile<E, D> {
            unreachable!()
        }
    }

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

    /// Kernel-side view into GPU global memory. `-1` in `D` marks a dynamic dim.
    #[cuda_tile::ty(name="!cuda_tile.tensor_view",
                    type_params=["{D}xE", "strides"],
                    type_meta=["base", "shape", "strides", "token"])]
    #[cuda_tile::variadic_struct(N = 6)]
    pub struct Tensor<E: ElementType, const D: [i32; N]> {
        _type: PhantomData<E>,
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<E: ElementType, const S: [i32; N]> Tensor<E, S> {
        // TODO (hme): forward params/output_type_params like make_partition_view.
        pub fn partition<'a, const R: [i32; N]>(&'a self, tile: Shape<R>) -> Partition<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            let p: Partition<E, R> =
                make_partition_view(self, tile, padding::Zero, dim_map::Identity, tensor_token);
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
                make_partition_view(self, tile, padding::None, dim_map, tensor_token);
            p
        }
        pub unsafe fn partition_mut<'a, const R: [i32; N]>(
            &'a mut self,
            tile: Shape<R>,
        ) -> PartitionMut<'a, E, R> {
            // TODO (hme): Bounds checks.
            let tensor_token: Token = get_tensor_token(self);
            let outer_tile: Shape<S> = Shape::<S> { dims: &[] };
            let mut p: PartitionMut<E, R> =
                unsafe { make_nested_partition_view_mut(self, tile, padding::None, tensor_token) };
            set_nested_mutable_partition_access_offset(&mut p, outer_tile);
            p
        }

        /// Returns the shape of this tensor.
        pub fn shape<'b>(&self) -> Shape<'b, S> {
            get_tensor_shape_meta(self)
        }
        pub fn load_tile<const R: [i32; N]>(&self, shape: Shape<R>, idx: [i32; N]) -> Tile<E, R> {
            load_tile(self, shape, idx)
        }
    }

    /// Read-only tiled view of a `Tensor`. Index a tile via `partition.load([i, j, …])`.
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
        pub fn load(&self, index: [i32; N]) -> Tile<E, D> {
            check_partition_access(self, index);
            let result: Tile<E, D> = load_view_tko(
                self,
                index,
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            result
        }
    }

    /// Mutable partition view. Loads/stores are unordered (hence `unsafe`);
    /// prefer `Tensor::load`/`store` for ordered access.
    // TODO (hme): consolidate Partition + PartitionMut into a single type.
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
        // Unordered load — caller must provide ordering via tokens or other sync.
        pub unsafe fn load(&self, index: [i32; N]) -> Tile<E, D> {
            let result: Tile<E, D> = unsafe {
                load_view_tko_mut(
                    self,
                    index,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            result
        }

        /// Stores a tile to this mutable partition at the specified index.
        ///
        /// Returns a token representing the completion of the store operation.
        ///
        /// ## Safety
        ///
        /// This is unsafe because it uses unordered memory operations.
        pub unsafe fn store(&mut self, tile: Tile<E, D>, index: [i32; N]) -> Token {
            let token: Token = unsafe {
                store_view_tko_mut(
                    self,
                    tile,
                    index,
                    ordering::Weak,
                    scope::TileBlock,
                    None,
                    tma::Enabled,
                )
            };
            token
        }
    }

    /// Memory-ordering token. Threaded through async memory ops to express
    /// dependencies; managed automatically by the load/store/partition APIs.
    #[cuda_tile::ty(name="!cuda_tile.token", params=[])]
    #[derive(Copy, Clone)]
    pub struct Token {}

    // ========================================================================
    // UTILITIES — compile-time descriptors, kernel intrinsics
    // ========================================================================

    /// Compile-time shape descriptor for tensors and tiles. Construct via
    /// [`const_shape!`].
    #[cuda_tile::variadic_struct(N = 6, constructor = "new")]
    #[derive(Copy, Clone)]
    pub struct Shape<'a, const D: [i32; N]> {
        pub dims: &'a [i32],
    }

    /// Construct a compile-time `Shape` from literal dims (0–4D supported).
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

    /// Compile-time index/metadata array — used for permutations and dim maps.
    #[cuda_tile::variadic_struct(N = 6, constructor = "new")]
    #[derive(Copy, Clone)]
    pub struct Array<'a, const D: [i32; N]> {
        pub dims: &'a [i32],
    }

    #[cuda_tile::variadic_impl(N = 6)]
    impl<'a, const D: [i32; N]> dim_map::Mode for Array<'a, D> {}

    /// Construct a compile-time `Array` from literal values (0–4D supported).
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

    /// Printf-style debug output from inside a kernel. `{}` becomes `%` at lowering.
    #[macro_export]
    macro_rules! cuda_tile_print {
        ($s:literal $(,$args:expr)*) => {
            unreachable!();
        };
    }
    pub use cuda_tile_print;

    /// Runtime assertion inside a kernel.
    #[macro_export]
    macro_rules! cuda_tile_assert {
        ($args:expr, $s:literal) => {
            unreachable!();
        };
    }
    pub use cuda_tile_assert;

    /// Grid dimensions `(gridDim.x, gridDim.y, gridDim.z)`.
    #[cuda_tile::op(name="cuda_tile.get_num_tile_blocks", params=[])]
    pub fn get_num_tile_blocks() -> (i32, i32, i32) {
        unreachable!()
    }

    /// Current block id `(blockIdx.x, blockIdx.y, blockIdx.z)`.
    #[cuda_tile::op(name="cuda_tile.get_tile_block_id", params=[])]
    pub fn get_tile_block_id() -> (i32, i32, i32) {
        unreachable!()
    }

    /// Wrap a scalar in a 0-dim tile.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn scalar_to_tile<E: ElementType>(scalar: impl Scalar) -> Tile<E, { [] }> {
        unreachable!()
    }

    /// Unwrap a 0-dim tile back into a scalar.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn tile_to_scalar<E: ElementType, S: Scalar>(tile: Tile<E, { [] }>) -> S {
        unreachable!()
    }

    /// Convert a scalar between element types.
    #[cuda_tile::compiler_op(name = "convert")]
    pub fn convert_scalar<S: Scalar>(x: impl Scalar) -> S {
        unreachable!()
    }

    /// Convert every element of a tile between element types.
    #[cuda_tile::compiler_op(name = "convert")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn convert_tile<TO: ElementType, FROM: ElementType, const S: [i32; N]>(
        x: Tile<FROM, S>,
    ) -> Tile<TO, S> {
        unreachable!()
    }

    /// Wrap a raw pointer as a 0-dim `PointerTile`.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn pointer_to_tile<P: Pointer>(ptr: P) -> PointerTile<P, { [] }> {
        unreachable!()
    }

    /// Unwrap a 0-dim `PointerTile` back into a raw pointer.
    #[cuda_tile::compiler_op(name = "cast")]
    pub fn tile_to_pointer<P: Pointer>(tile: PointerTile<P, { [] }>) -> P {
        unreachable!()
    }

    /// Bounds-check a partition access. Optimizer drops it when provably safe;
    /// otherwise emits an assertion at runtime.
    #[cuda_tile::compiler_op(name = "check")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn check_partition_access<E: ElementType, const S: [i32; N]>(
        part: &Partition<E, S>,
        index: [i32; N],
    ) {
        unreachable!()
    }

    // ========================================================================
    // OPS § CORE — Tile IR §8.3
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations/core.html
    // ========================================================================

    /// Broadcast a tile to a new shape (size-1 dims expand).
    #[cuda_tile::op(name="cuda_tile.broadcast", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn broadcast<E: ElementType, const S: [i32; N], const R: [i32; N]>(
        source: Tile<E, S>,
        shape: Shape<R>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Concatenate `lhs` and `rhs` along `dim`. All other dims must match.
    #[cuda_tile::op(name="cuda_tile.cat", params=["lhs", "rhs"], attribute_params=["dim:integer"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cat<E: ElementType, const SLhs: [i32; N], const SRhs: [i32; N], const SOut: [i32; N]>(
        lhs: Tile<E, SLhs>,
        rhs: Tile<E, SRhs>,
        dim: i32,
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /// Tile filled with a compile-time `value`.
    #[cuda_tile::op(name="cuda_tile.constant", params=[], attribute_params=["value:dense"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn constant<E: ElementType, const S: [i32; N]>(value: E, shape: Shape<S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Extract a subtile. Result shape must evenly divide source shape.
    /// `indices` are slice indices (not byte offsets).
    #[cuda_tile::op(name="cuda_tile.extract", params=["source", "...indices"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn extract<E: ElementType, const SIn: [i32; N], const SOut: [i32; N]>(
        source: Tile<E, SIn>,
        indices: [Tile<i32, { [] }>; N],
    ) -> Tile<E, SOut> {
        unreachable!()
    }

    /// 1D sequence `[0, 1, …, N-1]`.
    #[cuda_tile::op(name = "cuda_tile.iota")]
    pub fn iota<E: ElementType, const S: [i32; 1]>(shape: Shape<S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Tensor-Core matrix multiply-accumulate: `result = lhs × rhs + acc`.
    #[cuda_tile::compiler_op(name = "mma")]
    pub fn mma<E1: ElementType, E2: ElementType, const M: i32, const N: i32, const K: i32>(
        lhs: Tile<E1, { [M, K] }>,
        rhs: Tile<E1, { [K, N] }>,
        acc: Tile<E2, { [M, N] }>,
    ) -> Tile<E2, { [M, N] }> {
        unreachable!()
    }

    /// Permute dimensions per the index array (e.g. `[1, 0]` = transpose).
    #[cuda_tile::op(name="cuda_tile.permute", params=["source"], attribute_params=["permutation:array"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn permute<E: ElementType, const A: [i32; N], const I: [i32; N], const R: [i32; N]>(
        source: Tile<E, A>,
        permutation: Array<I>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Reshape a tile (element count must match).
    #[cuda_tile::op(name="cuda_tile.reshape", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reshape<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        source: Tile<E, S>,
        shape: Shape<R>,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Generic reduce along `dim` with closure `f` and `identity`.
    /// Closure body lowers to an MLIR region at compile time.
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

    /// Min along `dim` (collapses that axis).
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_min<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Max along `dim`.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_max<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Sum along `dim`.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_sum<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Product along `dim`. Watch for overflow with large tiles.
    #[cuda_tile::compiler_op(name = "reduce")]
    #[cuda_tile::variadic_op(N = 6, M = 6)]
    pub fn reduce_prod<E: ElementType, const S: [i32; N], const R: [i32; M]>(
        x: Tile<E, S>,
        dim: i32,
    ) -> Tile<E, R> {
        unreachable!()
    }

    /// Prefix sum along `dim`. The compiler emits the addf/addi region.
    #[cuda_tile::op(name="cuda_tile.scan", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn scan_sum<E: ElementType, const S: [i32; N], R: reverse::Mode>(
        operand: Tile<E, S>,
        dim: i32,
        reverse: R,
        identity: E,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Generic prefix scan along `dim`. Closure body lowers to an MLIR region.
    #[cuda_tile::op(name="cuda_tile.scan", params=["operand"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn scan<E: ElementType, const S: [i32; N], R: reverse::Mode, F>(
        operand: Tile<E, S>,
        dim: i32,
        reverse: R,
        identity: E,
        f: F,
    ) -> Tile<E, S>
    where
        F: Fn(E, E) -> E,
    {
        unreachable!()
    }

    /// `cond ? val_if_true : val_if_false` (element-wise).
    #[cuda_tile::op(name="cuda_tile.select", params=["cond", "val_if_true", "val_if_false"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn select<E: ElementType, const S: [i32; N]>(
        cond: Tile<bool, S>,
        val_if_true: Tile<E, S>,
        val_if_false: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Creates a new unordered token.
    #[cuda_tile::op(name="cuda_tile.make_token", params=[])]
    pub fn new_token_unordered() -> Token {
        unreachable!()
    }

    /// Combine independent tokens into one that depends on all of them.
    #[cuda_tile::op(name="cuda_tile.join_tokens", params=["tokens"])]
    pub fn join_tokens(tokens: &[Token]) -> Token {
        unreachable!()
    }

    /// Extract a dimension size from a `Shape`.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "shape")]
    pub fn get_shape_dim<const S: [i32; N]>(shape: Shape<S>, dim_idx: i32) -> i32 {
        unreachable!()
    }

    /// Extract a `Tensor`'s shape from its type metadata.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "shape")]
    pub fn get_tensor_shape_meta<'s, E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
    ) -> Shape<'s, S> {
        unreachable!()
    }

    /// Extract a `Tensor`'s ordering token.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_tensor_token<E: ElementType, const S: [i32; N]>(tensor: &Tensor<E, S>) -> Token {
        unreachable!()
    }

    /// Update a `Tensor`'s ordering token after a memory op.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "set_type_meta_field", type_meta_field = "token")]
    pub fn set_tensor_token<E: ElementType, const S: [i32; N]>(
        tensor: &Tensor<E, S>,
        token: Token,
    ) {
        unreachable!()
    }

    /// Attach CTA-local index offset metadata to a nested mutable partition.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "set_nested_mutable_partition_access_offset")]
    pub fn set_nested_mutable_partition_access_offset<
        'a,
        E: ElementType,
        const OUTER_TILE: [i32; N],
        const NESTED_TILE: [i32; N],
    >(
        partition: &mut PartitionMut<'a, E, NESTED_TILE>,
        outer_tile: Shape<OUTER_TILE>,
    ) {
        unreachable!()
    }

    /// Extract a `Partition`'s ordering token.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_partition_token<E: ElementType, const D: [i32; N]>(view: &Partition<E, D>) -> Token {
        unreachable!()
    }

    /// Extract a `PartitionMut`'s ordering token.
    #[cuda_tile::variadic_op(N = 6)]
    #[cuda_tile::compiler_op(name = "return_type_meta_field", type_meta_field = "token")]
    pub fn get_partition_token_mut<E: ElementType, const D: [i32; N]>(
        view: &PartitionMut<E, D>,
    ) -> Token {
        unreachable!()
    }

    /// Number of tiles along `axis` (`cdiv(tensor_dim, tile_dim)`).
    /// `axis` must be a compile-time constant in `0..N`. Lowers to
    /// `cuda_tile.get_index_space_shape` with axis-th result extracted.
    #[cuda_tile::compiler_op(name = "num_tiles")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn num_tiles<E: ElementType, const S: [i32; N]>(view: &Partition<E, S>, axis: i32) -> i32 {
        unreachable!()
    }

    // ========================================================================
    // OPS § CONVERSIONS — Tile IR §8.4
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations/conversions.html
    // ========================================================================

    /// Float-to-float conversion (e.g. `f32` → `f16`).
    #[cuda_tile::op(name = "cuda_tile.ftof", params = ["x"], static_params = ["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ftof<EIn: ElementType, EOut: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<EIn, S>,
        rounding: R,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Float-to-integer conversion. Destination signedness is inferred from `EOut`.
    #[cuda_tile::op(name = "cuda_tile.ftoi", params = ["x"], static_params = ["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ftoi<EIn: ElementType, EOut: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<EIn, S>,
        rounding: R,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer-to-float conversion. Source signedness is inferred from `EIn`.
    #[cuda_tile::op(name = "cuda_tile.itof", params = ["x"], static_params = ["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>, Full: rounding_mode=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn itof<EIn: ElementType, EOut: ElementType, const S: [i32; N], R: rounding::Mode>(
        x: Tile<EIn, S>,
        rounding: R,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer extension. Signedness is inferred from `EIn` by the JIT.
    #[cuda_tile::op(name="cuda_tile.exti", params=["from"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exti<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        from: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Integer truncation.
    #[cuda_tile::op(name = "cuda_tile.trunci", params = ["from"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn trunci<EIn: ElementType, EOut: ElementType, const S: [i32; N], O: overflow::Mode>(
        from: Tile<EIn, S>,
        overflow: O,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Bit-reinterpretation between same-size types.
    #[cuda_tile::op(name="cuda_tile.bitcast", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn bitcast<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        source: Tile<EIn, S>,
    ) -> Tile<EOut, S> {
        unreachable!()
    }

    /// Convert integer to pointer.
    #[cuda_tile::op(name="cuda_tile.int_to_ptr", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn int_to_ptr<SRC_T: ElementType, PTR_T: ElementType, const S: [i32; N]>(
        source: Tile<SRC_T, S>,
    ) -> PointerTile<*mut PTR_T, S> {
        unreachable!()
    }

    /// Convert pointer to integer.
    #[cuda_tile::op(name="cuda_tile.ptr_to_int", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ptr_to_int<E: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Cast pointer type — reinterpret pointers as pointing to a different type.
    #[cuda_tile::op(name="cuda_tile.ptr_to_ptr", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ptr_to_ptr<EIn: ElementType, EOut: ElementType, const S: [i32; N]>(
        source: PointerTile<*mut EIn, S>,
    ) -> PointerTile<*mut EOut, S> {
        unreachable!()
    }

    // ========================================================================
    // OPS § NUMERIC / BITWISE — Tile IR §§8.7-8.9
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#floating-point
    // ========================================================================

    // ---- Integer arithmetic ------------------------------------------------

    /// Element-wise integer add. `overflow` lets the compiler assume no
    /// signed/unsigned/both wrap.
    #[cuda_tile::op(name = "cuda_tile.addi", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addi<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer subtract.
    #[cuda_tile::op(name = "cuda_tile.subi", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn subi<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer multiply.
    #[cuda_tile::op(name = "cuda_tile.muli", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn muli<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer divide. `Zero` rounding = truncation;
    /// `PositiveInf` = ceiling div; `NegativeInf` = floor div (signed only).
    #[cuda_tile::op(name = "cuda_tile.divi", params = ["lhs", "rhs"], static_params = ["rounding={Zero: , NearestEven: rounding=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding=#cuda_tile.rounding<negative_inf>, Approx: rounding=#cuda_tile.rounding<approx>, Full: rounding=#cuda_tile.rounding<full>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn divi<E: ElementType, const S: [i32; N], R: rounding::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        rounding: R,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer remainder. Result sign matches dividend.
    #[cuda_tile::op(name = "cuda_tile.remi", params = ["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn remi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer negation.
    #[cuda_tile::op(name = "cuda_tile.negi", params = ["x"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn negi<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        x: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer absolute value. Note: cuTile maps signed ints to `i64` (not `i32`).
    #[cuda_tile::op(name="cuda_tile.absi", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn absi<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Multiply high — upper N bits of `x * y` for N-bit integer `E`.
    #[cuda_tile::op(name="cuda_tile.mulhii", params=["x", "y"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mulhii<E: ElementType, const S: [i32; N]>(x: Tile<E, S>, y: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer maximum. Signedness inferred from `E`.
    #[cuda_tile::op(name="cuda_tile.maxi", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn maxi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise integer minimum. Signedness inferred from `E`.
    #[cuda_tile::op(name="cuda_tile.mini", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn mini<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Integer Tensor-Core matrix multiply-accumulate.
    #[cuda_tile::op(
        name = "cuda_tile.mmai",
        params = ["lhs", "rhs", "acc"],
        static_params = [
            "signedness_lhs={Signed: signedness_lhs=#cuda_tile.signedness<signed>, Unsigned: signedness_lhs=#cuda_tile.signedness<unsigned>}",
            "signedness_rhs={Signed: signedness_rhs=#cuda_tile.signedness<signed>, Unsigned: signedness_rhs=#cuda_tile.signedness<unsigned>}"
        ]
    )]
    #[cuda_tile::variadic_op(N = 3)]
    pub fn mmai<
        EIn: ElementType,
        const LHS: [i32; N],
        const RHS: [i32; N],
        const ACC: [i32; N],
        SL: signedness::Mode,
        SR: signedness::Mode,
    >(
        lhs: Tile<EIn, LHS>,
        rhs: Tile<EIn, RHS>,
        acc: Tile<i32, ACC>,
        signedness_lhs: SL,
        signedness_rhs: SR,
    ) -> Tile<i32, ACC> {
        unreachable!()
    }

    // ---- Float arithmetic --------------------------------------------------

    /// Element-wise float add.
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

    /// Element-wise float subtract.
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

    /// Element-wise float multiply.
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

    /// Element-wise float divide.
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

    /// Element-wise float remainder.
    #[cuda_tile::op(name = "cuda_tile.remf", params = ["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn remf<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float negation.
    #[cuda_tile::op(name="cuda_tile.negf", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn negf<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float absolute value.
    #[cuda_tile::op(name="cuda_tile.absf", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn absf<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `atan2(x, y)`.
    #[cuda_tile::op(name = "cuda_tile.atan2", params = ["x", "y"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn atan2<E: ElementType, const S: [i32; N]>(x: Tile<E, S>, y: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Fused multiply-add: `lhs * rhs + acc` with one rounding step.
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

    /// Floating-point Tensor-Core matrix multiply-accumulate.
    #[cuda_tile::op(name = "cuda_tile.mmaf", params = ["lhs", "rhs", "acc"])]
    #[cuda_tile::variadic_op(N = 3)]
    pub fn mmaf<
        EIn: ElementType,
        EOut: ElementType,
        const LHS: [i32; N],
        const RHS: [i32; N],
        const ACC: [i32; N],
    >(
        lhs: Tile<EIn, LHS>,
        rhs: Tile<EIn, RHS>,
        acc: Tile<EOut, ACC>,
    ) -> Tile<EOut, ACC> {
        unreachable!()
    }

    /// Element-wise `source ^ exponent`.
    #[cuda_tile::op(name="cuda_tile.pow", params=["source", "exponent"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn pow<E: ElementType, const S: [i32; N]>(
        source: Tile<E, S>,
        exponent: Tile<E, S>,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise float maximum. `-0.0 < +0.0`.
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

    /// Element-wise float minimum.
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

    // ---- Min/Max + arithmetic helpers --------------------------------------

    /// Scalar minimum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn min<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Scalar maximum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn max<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Element-wise tile minimum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn min_tile<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise tile maximum.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn max_tile<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Ceiling division `⌈a / b⌉` (scalar).
    #[cuda_tile::compiler_op(name = "arithmetic")]
    pub fn ceil_div<E: ElementType>(a: E, b: E) -> E {
        unreachable!()
    }

    /// Element-wise true (floating-point) division.
    #[cuda_tile::compiler_op(name = "arithmetic")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn true_div<E: ElementType, const S: [i32; N]>(a: Tile<E, S>, b: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    // ---- Math --------------------------------------------------------------

    /// Element-wise ceiling.
    #[cuda_tile::op(name="cuda_tile.ceil", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ceil<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise floor.
    #[cuda_tile::op(name="cuda_tile.floor", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn floor<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise sine (radians).
    #[cuda_tile::op(name="cuda_tile.sin", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sin<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise cosine (radians).
    #[cuda_tile::op(name="cuda_tile.cos", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cos<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise tangent (radians).
    #[cuda_tile::op(name="cuda_tile.tan", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn tan<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise hyperbolic sine.
    #[cuda_tile::op(name="cuda_tile.sinh", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sinh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise hyperbolic cosine.
    #[cuda_tile::op(name="cuda_tile.cosh", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cosh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise hyperbolic tangent.
    #[cuda_tile::op(name="cuda_tile.tanh", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn tanh<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `e^x`.
    #[cuda_tile::op(name="cuda_tile.exp", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exp<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `2^x`.
    #[cuda_tile::op(name="cuda_tile.exp2", params=["x"], static_params=["ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn exp2<E: ElementType, const S: [i32; N], F: ftz::Mode>(
        x: Tile<E, S>,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise natural log.
    #[cuda_tile::op(name="cuda_tile.log", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn log<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise base-2 logarithm.
    #[cuda_tile::op(name="cuda_tile.log2", params=["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn log2<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise square root.
    #[cuda_tile::op(name="cuda_tile.sqrt", params=["x"], static_params=["rounding={NearestEven: rounding_mode=#cuda_tile.rounding<nearest_even>, PositiveInf: rounding_mode=#cuda_tile.rounding<positive_inf>, NegativeInf: rounding_mode=#cuda_tile.rounding<negative_inf>, Zero: rounding_mode=#cuda_tile.rounding<zero>, Approx: rounding_mode=#cuda_tile.rounding<approx>}", "ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn sqrt<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
        x: Tile<E, S>,
        rounding: R,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise `1 / sqrt(x)` (faster than `1.0 / sqrt(x)`).
    #[cuda_tile::op(name="cuda_tile.rsqrt", params=["x"], static_params=["ftz={Enabled: flush_to_zero=unit}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn rsqrt<E: ElementType, const S: [i32; N], F: ftz::Mode>(
        x: Tile<E, S>,
        ftz: F,
    ) -> Tile<E, S> {
        unreachable!()
    }

    // ---- Comparison --------------------------------------------------------

    /// Element-wise integer compare with explicit predicate.
    #[cuda_tile::op(
        name = "cuda_tile.cmpi",
        params = ["lhs", "rhs"],
        static_params = ["predicate={Equal: comparison_predicate=#cuda_tile.cmp_predicate<equal>, NotEqual: comparison_predicate=#cuda_tile.cmp_predicate<not_equal>, LessThan: comparison_predicate=#cuda_tile.cmp_predicate<less_than>, LessThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<less_than_or_equal>, GreaterThan: comparison_predicate=#cuda_tile.cmp_predicate<greater_than>, GreaterThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<greater_than_or_equal>}"],
        named_attributes = ["signedness=inferred_signedness"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cmpi<E: ElementType, const S: [i32; N], P: predicate::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        predicate: P,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    /// Element-wise float compare. `Ordered` requires both operands non-NaN;
    /// `Unordered` returns `true` if either is NaN.
    #[cuda_tile::op(
        name = "cuda_tile.cmpf",
        params = ["lhs", "rhs"],
        static_params = [
            "predicate={Equal: comparison_predicate=#cuda_tile.cmp_predicate<equal>, NotEqual: comparison_predicate=#cuda_tile.cmp_predicate<not_equal>, LessThan: comparison_predicate=#cuda_tile.cmp_predicate<less_than>, LessThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<less_than_or_equal>, GreaterThan: comparison_predicate=#cuda_tile.cmp_predicate<greater_than>, GreaterThanOrEqual: comparison_predicate=#cuda_tile.cmp_predicate<greater_than_or_equal>}",
            "ordering={Unordered: comparison_ordering=#cuda_tile.comparison_ordering<unordered>, Ordered: comparison_ordering=#cuda_tile.comparison_ordering<ordered>}"
        ]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn cmpf<E: ElementType, const S: [i32; N], P: predicate::Mode, Ord: cmp_ordering::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        predicate: P,
        ordering: Ord,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    // `_tile` suffix is required by the compiler — bare names collide with
    // Rust's PartialEq/PartialOrd which can't be variadic-impl'd over CGAs.

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn eq_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ne_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn gt_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ge_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn lt_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    #[cuda_tile::compiler_op(name = "tile")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn le_tile<E: ElementType, const S: [i32; N]>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
    ) -> Tile<bool, S> {
        unreachable!()
    }

    // ---- Bitwise -----------------------------------------------------------

    /// Element-wise bitwise AND.
    #[cuda_tile::op(name="cuda_tile.andi", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn andi<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise bitwise OR.
    #[cuda_tile::op(name="cuda_tile.ori", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn ori<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise bitwise XOR.
    #[cuda_tile::op(name="cuda_tile.xori", params=["lhs", "rhs"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn xori<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise bitwise NOT.
    #[cuda_tile::op(name = "cuda_tile.noti", params = ["x"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn noti<E: ElementType, const S: [i32; N]>(x: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise left shift.
    #[cuda_tile::op(name = "cuda_tile.shli", params = ["lhs", "rhs"], static_params = ["overflow={None: , NoSignedWrap: overflow=#cuda_tile.overflow<no_signed_wrap>, NoUnsignedWrap: overflow=#cuda_tile.overflow<no_unsigned_wrap>, NoWrap: overflow=#cuda_tile.overflow<no_wrap>}"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn shli<E: ElementType, const S: [i32; N], O: overflow::Mode>(
        lhs: Tile<E, S>,
        rhs: Tile<E, S>,
        overflow: O,
    ) -> Tile<E, S> {
        unreachable!()
    }

    /// Element-wise right shift. Arithmetic for signed `E`, logical for unsigned.
    #[cuda_tile::op(name="cuda_tile.shri", params=["lhs", "rhs"], named_attributes=["signedness=inferred_signedness"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn shri<E: ElementType, const S: [i32; N]>(lhs: Tile<E, S>, rhs: Tile<E, S>) -> Tile<E, S> {
        unreachable!()
    }

    // ========================================================================
    // OPS § ATOMICS — Tile IR §8.10
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#atomics
    // ========================================================================

    /// Atomic read-modify-write. `mode` selects the op (`atomic::*`),
    /// `memory_ordering` excludes `Weak`. Returns `(old_values, token)`.
    #[doc(hidden)]
    #[cuda_tile::op(name="cuda_tile.atomic_rmw_tko", params=["pointers", "arg"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn atomic_rmw_tko<
        E: ElementType,
        const S: [i32; N],
        M: atomic::Mode,
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        pointers: PointerTile<*mut E, S>,
        arg: Tile<E, S>,
        mode: M,
        memory_ordering: O,
        memory_scope: Sc,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Atomic compare-and-swap. Bitwise comparison: NaN ≠ NaN, ±0.0 distinct
    /// if their bit patterns differ. Returns `(old_values, token)`.
    #[cuda_tile::op(name="cuda_tile.atomic_cas_tko", params=["pointers", "cmp", "val"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn atomic_cas_tko<
        E: ElementType,
        const S: [i32; N],
        O: ordering::AtomicMode,
        Sc: scope::Mode,
    >(
        pointers: PointerTile<*mut E, S>,
        cmp: Tile<E, S>,
        val: Tile<E, S>,
        memory_ordering: O,
        memory_scope: Sc,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    // ========================================================================
    // OPS § MEMORY — Tile IR §8.6
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#memory
    // ========================================================================

    /// Gather from a pointer tile. Returns `(loaded_values, token)`.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Acquire}; `Weak` drops scope.
    #[cuda_tile::op(name="cuda_tile.load_ptr_tko", params=["source"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_ptr_tko<
        E: ElementType,
        const S: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        const CYCLES: u32,
    >(
        source: PointerTile<*mut E, S>,
        memory_ordering: O,
        memory_scope: Option<Sc>,
        mask: Option<Tile<bool, S>>,
        padding_value: Option<E>,
        token: Option<Token>,
        latency: Latency<CYCLES>,
    ) -> (Tile<E, S>, Token) {
        unreachable!()
    }

    /// Scatter to a pointer tile. Returns the completion token.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Release}; `Weak` drops scope.
    #[cuda_tile::op(name="cuda_tile.store_ptr_tko", params=["destination", "value"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn store_ptr_tko<
        E: ElementType,
        const S: [i32; N],
        O: ordering::StoreMode,
        Sc: scope::Mode,
        const CYCLES: u32,
    >(
        destination: PointerTile<*mut E, S>,
        value: Tile<E, S>,
        memory_ordering: O,
        memory_scope: Option<Sc>,
        mask: Option<Tile<bool, S>>,
        token: Option<Token>,
        latency: Latency<CYCLES>,
    ) -> Token {
        unreachable!()
    }

    // ========================================================================
    // OPS § VIEWS — Tile IR §8.11
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#views
    // ========================================================================

    /// Query a partition view's index-space shape as scalar tile values.
    #[cuda_tile::op(name = "cuda_tile.get_index_space_shape", params = ["src"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn get_index_space_shape<E: ElementType, const S: [i32; N]>(
        src: &Partition<E, S>,
    ) -> [i32; N] {
        unreachable!()
    }

    /// Query a tensor view's dynamic shape as scalar tile values.
    #[cuda_tile::op(name = "cuda_tile.get_tensor_shape", params = ["src"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn get_tensor_shape<E: ElementType, const S: [i32; N]>(src: &Tensor<E, S>) -> [i32; N] {
        unreachable!()
    }

    /// Load a tile from a partition view at `index`. `tma::Disabled` suppresses
    /// TMA lowering. `memory_ordering` ⊆ {Weak, Relaxed, Acquire}.
    // TODO (hme): Mark loads from shared refs as unsafe and add `_unchecked` suffix.
    #[cuda_tile::op(name = "load_view_tko", params = ["view", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_view_tko<
        E: ElementType,
        const D: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &Partition<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Tile<E, D> {
        unreachable!()
    }

    /// `load_view_tko` for `PartitionMut`. Caller must ensure no aliasing.
    // TODO (hme): document safety
    #[cuda_tile::op(name = "load_view_tko", params = ["view", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn load_view_tko_mut<
        E: ElementType,
        const D: [i32; N],
        O: ordering::LoadMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &PartitionMut<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Tile<E, D> {
        unreachable!()
    }

    /// Store a tile into a `PartitionMut` at `index`. Returns the completion token.
    /// `memory_ordering` ⊆ {Weak, Relaxed, Release}.
    // TODO (hme): document safety
    #[cuda_tile::op(name = "store_view_tko", params = ["view", "tile", "index"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn store_view_tko_mut<
        E: ElementType,
        const D: [i32; N],
        O: ordering::StoreMode,
        Sc: scope::Mode,
        T: tma::Mode,
    >(
        view: &mut PartitionMut<E, D>,
        tile: Tile<E, D>,
        index: [i32; N],
        memory_ordering: O,
        memory_scope: Sc,
        latency: Option<i32>,
        tma: T,
    ) -> Token {
        unreachable!()
    }

    /// Build a `Tensor` view from a base pointer, shape, strides, and a token.
    /// Caller must guarantee the layout is valid.
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

    /// Build a read-only partition view. Prefer `tensor.partition(tile)`.
    ///
    /// Pass `padding::None` to omit the Tile IR `padding_value` type
    /// parameter, or a real padding marker such as `padding::Zero` to include
    /// it. Pass `dim_map::Identity` to omit `dim_map`, or an `Array` dim map
    /// to include it.
    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value", "dim_map"],
                    output_type_meta=["token", "tensor_view.shape()"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn make_partition_view<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        P: padding::Mode,
        M: dim_map::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        tile: Shape<TILE_SHAPE>,
        padding_value: P,
        dim_map: M,
        token: Token,
    ) -> Partition<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    /// Build a mutable partition view. Prefer `tensor.partition_mut(tile)`.
    ///
    /// Pass `padding::None` to omit the Tile IR `padding_value` type parameter,
    /// or a real padding marker such as `padding::Zero` to include it.
    ///
    /// # Safety
    ///
    /// The returned partition lifetime is not tied to the tensor argument, and
    /// callers must preserve the aliasing guarantees required by mutable view
    /// access.
    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_partition_view_mut<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        P: padding::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        padding_value: P,
        token: Token,
    ) -> PartitionMut<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    /// Build a nested mutable partition view from an already partitioned
    /// mutable tensor. This still partitions the full tensor view; the compiler
    /// attaches metadata so tile accesses are offset by the enclosing CTA tile.
    #[cuda_tile::op(name="cuda_tile.make_partition_view",
                    params=["tensor_view"],
                    output_type_params=["tensor_view", "padding_value"],
                    output_type_meta=["token"]
    )]
    #[cuda_tile::variadic_op(N = 6)]
    pub unsafe fn make_nested_partition_view_mut<
        'a,
        E: ElementType,
        const TENSOR_SHAPE: [i32; N],
        const TILE_SHAPE: [i32; N],
        P: padding::Mode,
    >(
        tensor_view: &Tensor<E, TENSOR_SHAPE>,
        shape: Shape<TILE_SHAPE>,
        padding_value: P,
        token: Token,
    ) -> PartitionMut<'a, E, TILE_SHAPE> {
        unreachable!()
    }

    // ---- Core pointer helpers ---------------------------------------------

    /// Pointer to a kernel-module global declared via the `global` op.
    #[cuda_tile::op(name="cuda_tile.get_global", params=[], named_attributes=["name:symbol_ref"])]
    pub fn get_global<E: ElementType>() -> PointerTile<*mut E, { [] }> {
        unreachable!()
    }

    /// `result[i] = ptr[i] + offset`.
    #[cuda_tile::op(name="cuda_tile.offset", params=["ptr", "offset"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addptr<P: Pointer, const D: [i32; N]>(
        ptr: PointerTile<P, D>,
        offset: i32,
    ) -> PointerTile<P, D> {
        unreachable!()
    }

    /// `result[i] = ptr[i] + offset[i]`.
    #[cuda_tile::op(name="cuda_tile.offset", params=["ptr", "offset"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn addptr_tile<I: ElementType, P: Pointer, const D: [i32; N]>(
        ptr: PointerTile<P, D>,
        offset: Tile<I, D>,
    ) -> PointerTile<P, D> {
        unreachable!()
    }

    /// Broadcast a `PointerTile` to a new shape (size-1 dims expand).
    #[cuda_tile::op(name="cuda_tile.broadcast", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, method = "broadcast")]
    pub fn broadcast_ptr<P: Pointer, const S: [i32; N], const R: [i32; N]>(
        source: PointerTile<P, S>,
        shape: Shape<R>,
    ) -> PointerTile<P, R> {
        unreachable!()
    }

    /// Reshape a `PointerTile` (element count must match).
    #[cuda_tile::op(name="cuda_tile.reshape", params=["source"])]
    #[cuda_tile::variadic_op(N = 6, M = 6, method = "reshape")]
    pub fn reshape_ptr<P: Pointer, const S: [i32; N], const R: [i32; M]>(
        source: PointerTile<P, S>,
        shape: Shape<R>,
    ) -> PointerTile<P, R> {
        unreachable!()
    }

    // ========================================================================
    // OPS § MISC — Tile IR §8.12
    // https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations/misc.html
    // ========================================================================

    /// Token-ordered debug print for a single tile argument.
    #[cuda_tile::op(name = "cuda_tile.print_tko", params = ["arg"])]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn print_tko<E: ElementType, const S: [i32; N]>(
        str: &str,
        arg: Tile<E, S>,
        token: Option<Token>,
    ) -> Token {
        unreachable!()
    }

    // ========================================================================
    // OPTIMIZATION ASSUMPTIONS
    // ========================================================================
    // Compile-time hints the optimizer treats as facts. UB if the property
    // doesn't hold at runtime — hence `unsafe`.

    /// Assert `x` is divisible by `DIVISOR`. UB if it isn't.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_div_by<T, const DIVISOR: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert every `every`-th element along dimension `along` is divisible by `divisor`.
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

    /// Assert `x >= LOWER`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds_lower<T, const LOWER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert `x <= UPPER`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds_upper<T, const UPPER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert `LOWER <= x <= UPPER`.
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_bounds<T, const LOWER: i32, const UPPER: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert elements within each consecutive group of `GROUP0` are equal (1D).
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_1d<T, const GROUP0: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert elements within each `GROUP0 × GROUP1` block are equal (2D).
    #[cuda_tile::compiler_op(name = "assume")]
    pub unsafe fn assume_same_elements_2d<T, const GROUP0: i32, const GROUP1: i32>(x: T) -> T {
        unreachable!()
    }

    /// Assert elements within each `GROUP0 × GROUP1 × GROUP2` block are equal (3D).
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

    /// Assert elements within each `GROUP0 × … × GROUP3` block are equal (4D).
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

    // ========================================================================
    // HIGH-LEVEL HELPERS
    // ========================================================================
    // Convenience wrappers built on the primitives above. Not Tile IR ops.

    /// Broadcast a scalar to a tile of the given shape.
    // `trait_name = "BroadcastScalarFn"` avoids colliding with the user
    // `BroadcastScalar` trait (PascalCase(broadcast_scalar) = BroadcastScalar).
    #[cuda_tile::variadic_op(N = 6, trait_name = "BroadcastScalarFn")]
    pub fn broadcast_scalar<E: ElementType, const S: [i32; N]>(
        x: E,
        shape: Shape<S>,
    ) -> Tile<E, S> {
        let ones_shape: Shape<{ [1; N] }> = Shape::<{ [1; N] }> { dims: &[1i32; N] };
        let tile_x: Tile<E, { [] }> = scalar_to_tile(x);
        tile_x.reshape(ones_shape).broadcast(shape)
    }

    /// Load a tile from `x` at `idx` with zero padding.
    #[cuda_tile::variadic_op(N = 6)]
    pub fn load_tile<E: ElementType, const S: [i32; N], const R: [i32; N]>(
        x: &Tensor<E, S>,
        tile_shape: Shape<R>,
        idx: [i32; N],
    ) -> Tile<E, R> {
        let tensor_token: Token = get_tensor_token(x);
        let x_partition: Partition<E, R> = make_partition_view(
            x,
            tile_shape,
            padding::Zero,
            dim_map::Identity,
            tensor_token,
        );
        let tile_x: Tile<E, R> = load_view_tko(
            &x_partition,
            idx,
            ordering::Weak,
            scope::TileBlock,
            None,
            tma::Enabled,
        );
        tile_x
    }

    fn load_tile_mut_1d<E: ElementType, const S: [i32; 1]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        let tile_y: Tile<E, S> = unsafe {
            load_view_tko_mut(
                &y_partition,
                [pid.0],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
        tile_y
    }

    fn load_tile_mut_2d<E: ElementType, const S: [i32; 2]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        let tile_y: Tile<E, S> = unsafe {
            load_view_tko_mut(
                &y_partition,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
        tile_y
    }

    fn load_tile_mut_3d<E: ElementType, const S: [i32; 3]>(y: &mut Tensor<E, S>) -> Tile<E, S> {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        let tile_y: Tile<E, S> = unsafe {
            load_view_tko_mut(
                &y_partition,
                [pid.0, pid.1, pid.2],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
        tile_y
    }

    fn store_tile_1d<E: ElementType, const S: [i32; 1]>(y: &mut Tensor<E, S>, result: Tile<E, S>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
    }

    fn store_tile_2d<E: ElementType, const S: [i32; 2]>(y: &mut Tensor<E, S>, result: Tile<E, S>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
    }

    fn store_tile_3d<E: ElementType, const S: [i32; 3]>(y: &mut Tensor<E, S>, result: Tile<E, S>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape: Shape<S> = Shape::<S> { dims: &[] };
        let tensor_token: Token = get_tensor_token(y);
        let mut y_partition: PartitionMut<E, S> =
            unsafe { make_partition_view_mut(y, tile_shape, padding::None, tensor_token) };
        unsafe {
            store_view_tko_mut(
                &mut y_partition,
                result,
                [pid.0, pid.1, pid.2],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            )
        };
        let new_token: Token = get_partition_token_mut(&y_partition);
        set_tensor_token(y, new_token);
    }

    /// Type-level dispatch for `load_tile_mut`.
    pub trait LoadTileMutAtCurrentBlock {
        type Out;

        fn load(&mut self) -> Self::Out;
    }

    impl<E: ElementType, const S: [i32; 1]> LoadTileMutAtCurrentBlock for Tensor<E, S> {
        type Out = Tile<E, S>;

        fn load(&mut self) -> Tile<E, S> {
            load_tile_mut_1d(self)
        }
    }

    impl<E: ElementType, const S: [i32; 2]> LoadTileMutAtCurrentBlock for Tensor<E, S> {
        type Out = Tile<E, S>;

        fn load(&mut self) -> Tile<E, S> {
            load_tile_mut_2d(self)
        }
    }

    impl<E: ElementType, const S: [i32; 3]> LoadTileMutAtCurrentBlock for Tensor<E, S> {
        type Out = Tile<E, S>;

        fn load(&mut self) -> Tile<E, S> {
            load_tile_mut_3d(self)
        }
    }

    /// Load a mutable tensor tile at the current tile-block id.
    pub fn load_tile_mut<Y>(y: &mut Y) -> <Y as LoadTileMutAtCurrentBlock>::Out
    where
        Y: LoadTileMutAtCurrentBlock,
    {
        y.load()
    }

    /// Type-level dispatch for `store_tile`.
    pub trait StoreTileAtCurrentBlock<Result> {
        fn store(&mut self, result: Result);
    }

    impl<E: ElementType, const S: [i32; 1]> StoreTileAtCurrentBlock<Tile<E, S>> for Tensor<E, S> {
        fn store(&mut self, result: Tile<E, S>) {
            store_tile_1d(self, result);
        }
    }

    impl<E: ElementType, const S: [i32; 2]> StoreTileAtCurrentBlock<Tile<E, S>> for Tensor<E, S> {
        fn store(&mut self, result: Tile<E, S>) {
            store_tile_2d(self, result);
        }
    }

    impl<E: ElementType, const S: [i32; 3]> StoreTileAtCurrentBlock<Tile<E, S>> for Tensor<E, S> {
        fn store(&mut self, result: Tile<E, S>) {
            store_tile_3d(self, result);
        }
    }

    /// Store `result` into a mutable tensor tile at the current tile-block id.
    pub fn store_tile<Y, Result>(y: &mut Y, result: Result)
    where
        Y: StoreTileAtCurrentBlock<Result>,
    {
        StoreTileAtCurrentBlock::store(y, result);
    }

    /// Type-level dispatch for `load_tile_like`.
    pub trait LoadTileLike<Y> {
        type Out;

        fn load_tile_like(&self, y: &Y) -> Self::Out;
    }

    impl<E1: ElementType, E2: ElementType, const S: [i32; 1]> LoadTileLike<Tensor<E2, S>>
        for Tensor<E1, { [-1] }>
    {
        type Out = Tile<E1, S>;

        fn load_tile_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            let x = self;
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = y.shape();
            let tensor_token: Token = get_tensor_token(x);
            let x_partition: Partition<E1, S> = make_partition_view(
                x,
                tile_shape,
                padding::None,
                dim_map::Identity,
                tensor_token,
            );
            let tile_x: Tile<E1, S> = load_view_tko(
                &x_partition,
                [pid.0],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            tile_x
        }
    }

    impl<E1: ElementType, E2: ElementType, const S: [i32; 2]> LoadTileLike<Tensor<E2, S>>
        for Tensor<E1, { [-1, -1] }>
    {
        type Out = Tile<E1, S>;

        fn load_tile_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            let x = self;
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = y.shape();
            let tensor_token: Token = get_tensor_token(x);
            let x_partition: Partition<E1, S> = make_partition_view(
                x,
                tile_shape,
                padding::None,
                dim_map::Identity,
                tensor_token,
            );
            let tile_x: Tile<E1, S> = load_view_tko(
                &x_partition,
                [pid.0, pid.1],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            tile_x
        }
    }

    impl<E1: ElementType, E2: ElementType, const S: [i32; 3]> LoadTileLike<Tensor<E2, S>>
        for Tensor<E1, { [-1, -1, -1] }>
    {
        type Out = Tile<E1, S>;

        fn load_tile_like(&self, y: &Tensor<E2, S>) -> Tile<E1, S> {
            let x = self;
            let pid: (i32, i32, i32) = get_tile_block_id();
            let tile_shape: Shape<S> = y.shape();
            let tensor_token: Token = get_tensor_token(x);
            let x_partition: Partition<E1, S> = make_partition_view(
                x,
                tile_shape,
                padding::None,
                dim_map::Identity,
                tensor_token,
            );
            let tile_x: Tile<E1, S> = load_view_tko(
                &x_partition,
                [pid.0, pid.1, pid.2],
                ordering::Weak,
                scope::TileBlock,
                None,
                tma::Enabled,
            );
            tile_x
        }
    }

    /// Load a tile of `x` matching `y`'s shape, indexed by the current tile-block id.
    pub fn load_tile_like<X, Y>(x: &X, y: &Y) -> <X as LoadTileLike<Y>>::Out
    where
        X: LoadTileLike<Y>,
    {
        x.load_tile_like(y)
    }

    /* TensorArray */
    // TODO (hme): Add a TensorArray type.
    //   #[cuda_tile::variadic_struct(N = 6)]
    //   struct TensorArray<E: ElementType, const D: [i32; N]> { _type: PhantomData<E> }

    /// Build a Tensor view from a strided pointer stored in `dst[idx]`.
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

    /// Permute a `[i32; N]` index array per `permutation`.
    #[cuda_tile::compiler_op(name = "shape")]
    #[cuda_tile::variadic_op(N = 6)]
    pub fn permute_array<const I: [i32; N]>(source: [i32; N], permutation: Array<I>) -> [i32; N] {
        unreachable!()
    }
}
