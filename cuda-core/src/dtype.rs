/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Primitive data types for GPU tensors and kernel arguments.
//!
//! `DType` marks Rust types that can be used as tensor element types (`Tensor<T: DType>`)
//! and as scalar kernel arguments. `DTypeId` is the corresponding runtime identifier.

use half::{bf16, f16};
use std::fmt::{Debug, Display};

// ---------------------------------------------------------------------------
// GPU-specific type wrappers (no host arithmetic, storage only)
// ---------------------------------------------------------------------------

/// TensorFloat-32 format (TF32). 19-bit format with FP32 range and FP16 precision.
/// Used by Ampere+ GPUs for accelerated matrix multiplication.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct tf32(pub u32);

/// FP8 E4M3FN format (4-bit exponent, 3-bit mantissa, no infinity).
/// Used by Hopper+ GPUs for efficient matrix multiplication and quantized inference.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct f8e4m3fn(pub u8);

/// FP8 E5M2 format (5-bit exponent, 2-bit mantissa).
/// Same exponent range as FP16 with reduced precision.
/// Used by Hopper+ GPUs.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct f8e5m2(pub u8);

/// Runtime identifier for a `DType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeId {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    TF32,
    F64,
    F8E4M3FN,
    F8E5M2,
}

impl DTypeId {
    /// Returns the string representation of this data type.
    pub fn as_str(&self) -> &'static str {
        match self {
            DTypeId::Bool => "bool",
            DTypeId::U8 => "u8",
            DTypeId::U16 => "u16",
            DTypeId::U32 => "u32",
            DTypeId::U64 => "u64",
            DTypeId::I8 => "i8",
            DTypeId::I16 => "i16",
            DTypeId::I32 => "i32",
            DTypeId::I64 => "i64",
            DTypeId::F16 => "f16",
            DTypeId::BF16 => "bf16",
            DTypeId::F32 => "f32",
            DTypeId::TF32 => "tf32",
            DTypeId::F64 => "f64",
            DTypeId::F8E4M3FN => "f8e4m3fn",
            DTypeId::F8E5M2 => "f8e5m2",
        }
    }

    /// Returns the size in bytes of this data type.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DTypeId::Bool | DTypeId::U8 | DTypeId::I8 => 1,
            DTypeId::U16 | DTypeId::I16 | DTypeId::F16 | DTypeId::BF16 => 2,
            DTypeId::U32 | DTypeId::I32 | DTypeId::F32 | DTypeId::TF32 => 4,
            DTypeId::U64 | DTypeId::I64 | DTypeId::F64 => 8,
            DTypeId::F8E4M3FN | DTypeId::F8E5M2 => 1,
        }
    }
}

impl Display for DTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A primitive type that can be stored in a `Tensor` or passed as a scalar kernel argument.
pub trait DType: Send + Sync + Copy + Debug + 'static {
    /// The runtime data type identifier for this scalar type.
    const DTYPE: DTypeId;
    /// Returns the zero value for this scalar type.
    fn zero() -> Self;
    /// Returns the one value for this scalar type.
    fn one() -> Self;
}

macro_rules! impl_dtype {
    ($($ty:ty => $variant:ident, $zero:expr, $one:expr),* $(,)?) => {
        $(
            impl DType for $ty {
                const DTYPE: DTypeId = DTypeId::$variant;
                fn zero() -> Self { $zero }
                fn one() -> Self { $one }
            }
        )*
    }
}

impl_dtype!(
    bool => Bool, false, true,
    u8 => U8, 0, 1,
    u16 => U16, 0, 1,
    u32 => U32, 0, 1,
    u64 => U64, 0, 1,
    i8 => I8, 0, 1,
    i16 => I16, 0, 1,
    i32 => I32, 0, 1,
    i64 => I64, 0, 1,
    f16 => F16, f16::ZERO, f16::ONE,
    bf16 => BF16, bf16::ZERO, bf16::ONE,
    f32 => F32, 0.0, 1.0,
    tf32 => TF32, tf32(0), tf32(0x3F800000),  // IEEE 754 1.0 in f32 bits
    f64 => F64, 0.0, 1.0,
    f8e4m3fn => F8E4M3FN, f8e4m3fn(0), f8e4m3fn(0x38),  // 1.0 in E4M3FN
    f8e5m2 => F8E5M2, f8e5m2(0), f8e5m2(0x3C),          // 1.0 in E5M2
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_enum_as_str() {
        assert_eq!(DTypeId::Bool.as_str(), "bool");
        assert_eq!(DTypeId::U8.as_str(), "u8");
        assert_eq!(DTypeId::U16.as_str(), "u16");
        assert_eq!(DTypeId::U32.as_str(), "u32");
        assert_eq!(DTypeId::U64.as_str(), "u64");
        assert_eq!(DTypeId::I8.as_str(), "i8");
        assert_eq!(DTypeId::I16.as_str(), "i16");
        assert_eq!(DTypeId::I32.as_str(), "i32");
        assert_eq!(DTypeId::I64.as_str(), "i64");
        assert_eq!(DTypeId::F16.as_str(), "f16");
        assert_eq!(DTypeId::BF16.as_str(), "bf16");
        assert_eq!(DTypeId::F32.as_str(), "f32");
        assert_eq!(DTypeId::TF32.as_str(), "tf32");
        assert_eq!(DTypeId::F64.as_str(), "f64");
        assert_eq!(DTypeId::F8E4M3FN.as_str(), "f8e4m3fn");
        assert_eq!(DTypeId::F8E5M2.as_str(), "f8e5m2");
    }

    #[test]
    fn dtype_enum_size_in_bytes() {
        assert_eq!(DTypeId::Bool.size_in_bytes(), 1);
        assert_eq!(DTypeId::U8.size_in_bytes(), 1);
        assert_eq!(DTypeId::I8.size_in_bytes(), 1);
        assert_eq!(DTypeId::U16.size_in_bytes(), 2);
        assert_eq!(DTypeId::I16.size_in_bytes(), 2);
        assert_eq!(DTypeId::F16.size_in_bytes(), 2);
        assert_eq!(DTypeId::BF16.size_in_bytes(), 2);
        assert_eq!(DTypeId::U32.size_in_bytes(), 4);
        assert_eq!(DTypeId::I32.size_in_bytes(), 4);
        assert_eq!(DTypeId::F32.size_in_bytes(), 4);
        assert_eq!(DTypeId::TF32.size_in_bytes(), 4);
        assert_eq!(DTypeId::U64.size_in_bytes(), 8);
        assert_eq!(DTypeId::I64.size_in_bytes(), 8);
        assert_eq!(DTypeId::F64.size_in_bytes(), 8);
    }

    #[test]
    fn dtype_enum_display() {
        assert_eq!(format!("{}", DTypeId::F32), "f32");
        assert_eq!(format!("{}", DTypeId::I64), "i64");
    }

    #[test]
    fn dtype_enum_equality() {
        assert_eq!(DTypeId::F32, DTypeId::F32);
        assert_ne!(DTypeId::F32, DTypeId::F64);
    }

    #[test]
    fn dtype_trait_associated_constants() {
        assert_eq!(<bool as DType>::DTYPE, DTypeId::Bool);
        assert_eq!(<u8 as DType>::DTYPE, DTypeId::U8);
        assert_eq!(<u16 as DType>::DTYPE, DTypeId::U16);
        assert_eq!(<u32 as DType>::DTYPE, DTypeId::U32);
        assert_eq!(<u64 as DType>::DTYPE, DTypeId::U64);
        assert_eq!(<i8 as DType>::DTYPE, DTypeId::I8);
        assert_eq!(<i16 as DType>::DTYPE, DTypeId::I16);
        assert_eq!(<i32 as DType>::DTYPE, DTypeId::I32);
        assert_eq!(<i64 as DType>::DTYPE, DTypeId::I64);
        assert_eq!(<f16 as DType>::DTYPE, DTypeId::F16);
        assert_eq!(<bf16 as DType>::DTYPE, DTypeId::BF16);
        assert_eq!(<f32 as DType>::DTYPE, DTypeId::F32);
        assert_eq!(<f64 as DType>::DTYPE, DTypeId::F64);
    }

    #[test]
    fn dtype_as_str_matches_type_name() {
        // Verify as_str() matches Rust's built-in type names, which is
        // important for kernel compilation generic parameter passing.
        assert_eq!(<f32 as DType>::DTYPE.as_str(), "f32");
        assert_eq!(<i32 as DType>::DTYPE.as_str(), "i32");
        assert_eq!(<u64 as DType>::DTYPE.as_str(), "u64");
    }

    #[test]
    fn dtype_size_matches_std_mem_size() {
        assert_eq!(
            <bool as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<bool>()
        );
        assert_eq!(
            <u8 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<u8>()
        );
        assert_eq!(
            <u16 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<u16>()
        );
        assert_eq!(
            <u32 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<u32>()
        );
        assert_eq!(
            <u64 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<u64>()
        );
        assert_eq!(
            <i8 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<i8>()
        );
        assert_eq!(
            <i16 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<i16>()
        );
        assert_eq!(
            <i32 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<i32>()
        );
        assert_eq!(
            <i64 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<i64>()
        );
        assert_eq!(
            <f16 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<f16>()
        );
        assert_eq!(
            <bf16 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<bf16>()
        );
        assert_eq!(
            <f32 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<f32>()
        );
        assert_eq!(
            <f64 as DType>::DTYPE.size_in_bytes(),
            std::mem::size_of::<f64>()
        );
    }

    #[test]
    fn dtype_zero_one() {
        assert_eq!(bool::zero(), false);
        assert_eq!(bool::one(), true);
        assert_eq!(u8::zero(), 0u8);
        assert_eq!(u8::one(), 1u8);
        assert_eq!(i32::zero(), 0i32);
        assert_eq!(i32::one(), 1i32);
        assert_eq!(f32::zero(), 0.0f32);
        assert_eq!(f32::one(), 1.0f32);
        assert_eq!(f64::zero(), 0.0f64);
        assert_eq!(f64::one(), 1.0f64);
        assert_eq!(f16::zero(), f16::ZERO);
        assert_eq!(f16::one(), f16::ONE);
        assert_eq!(bf16::zero(), bf16::ZERO);
        assert_eq!(bf16::one(), bf16::ONE);
    }

    // Compile-time verification that DType types satisfy the required bounds.
    fn _assert_dtype_bounds<T: DType>() {}
    #[test]
    fn dtype_satisfies_bounds() {
        _assert_dtype_bounds::<bool>();
        _assert_dtype_bounds::<u8>();
        _assert_dtype_bounds::<u16>();
        _assert_dtype_bounds::<u32>();
        _assert_dtype_bounds::<u64>();
        _assert_dtype_bounds::<i8>();
        _assert_dtype_bounds::<i16>();
        _assert_dtype_bounds::<i32>();
        _assert_dtype_bounds::<i64>();
        _assert_dtype_bounds::<f16>();
        _assert_dtype_bounds::<bf16>();
        _assert_dtype_bounds::<f32>();
        _assert_dtype_bounds::<f64>();
    }
}
