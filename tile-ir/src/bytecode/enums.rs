/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode format enums: section IDs, type tags, attribute tags,
//! function flags, debug tags, and version constants.
//!
//! Ported from `BytecodeEnums.h` and `Version.h` in the `cuda-tile` submodule.

/// Arbitrary byte used to fill alignment padding in the bytecode stream.
pub const ALIGNMENT_BYTE: u8 = 0xCB;

/// Magic number at the start of every Tile IR bytecode file.
pub const MAGIC: [u8; 8] = [0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00];

// ---------------------------------------------------------------------------
// Section IDs
// ---------------------------------------------------------------------------

/// Bytecode section identifiers.
///
/// The lower 7 bits of the on-disk section byte carry the ID; the high
/// bit indicates whether an alignment field follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Section {
    EndOfBytecode = 0x00,
    String = 0x01,
    Func = 0x02,
    Debug = 0x03,
    Constant = 0x04,
    Type = 0x05,
    Global = 0x06,
}

/// Total number of section kinds (excluding `EndOfBytecode` sentinel).
pub const NUM_SECTIONS: u8 = 0x07;

// ---------------------------------------------------------------------------
// Type tags
// ---------------------------------------------------------------------------

/// Tags written into the Type section to identify each type variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TypeTag {
    I1 = 0,
    I8 = 1,
    I16 = 2,
    I32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F32 = 7,
    TF32 = 8,
    F64 = 9,
    F8E4M3FN = 10,
    F8E5M2 = 11,
    Pointer = 12,
    Tile = 13,
    TensorView = 14,
    PartitionView = 15,
    Func = 16,
    Token = 17,
    Unknown = 18,
}

// ---------------------------------------------------------------------------
// Debug tags
// ---------------------------------------------------------------------------

/// Tags written into the Debug section to identify debug-info entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DebugTag {
    Unknown = 0,
    DICompileUnit = 1,
    DIFile = 2,
    DILexicalBlock = 3,
    DILoc = 4,
    DISubprogram = 5,
    CallSite = 6,
}

/// Reserved debug-location indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DebugReserved {
    /// Represents an unknown source location.
    UnknownLoc = 0,
    /// Number of reserved entries (first usable index).
    Size = 1,
}

// ---------------------------------------------------------------------------
// Function flags
// ---------------------------------------------------------------------------

/// Bit-flags stored per-function in the Func section header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FunctionFlag {
    /// Bit 0: 0 = public, 1 = private.
    VisibilityPrivate = 0x01,
    /// Bit 1: 0 = device function, 1 = kernel entry point.
    KindKernel = 0x02,
    /// Bit 2: 0 = no optimization hints, 1 = has optimization hints.
    HasOptimizationHints = 0x04,
}

// ---------------------------------------------------------------------------
// Attribute tags
// ---------------------------------------------------------------------------

/// Tags written before attribute values in the bytecode stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AttributeTag {
    Integer = 1,
    Float = 2,
    Bool = 3,
    Type = 4,
    String = 5,
    Array = 6,
    DenseElements = 7,
    DivBy = 8,
    SameElements = 9,
    Dictionary = 10,
    OptimizationHints = 11,
    Bounded = 12,
}

// ---------------------------------------------------------------------------
// Bytecode version
// ---------------------------------------------------------------------------

/// A Tile IR bytecode version triple: `major.minor.tag`.
///
/// Serialized as three little-endian fields in the file header
/// (u8, u8, u16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BytecodeVersion {
    pub major: u8,
    pub minor: u8,
    pub tag: u16,
}

impl BytecodeVersion {
    pub const CURRENT: Self = Self {
        major: 13,
        minor: 2,
        tag: 0,
    };

    pub const MIN_SUPPORTED: Self = Self {
        major: 13,
        minor: 1,
        tag: 0,
    };
}

impl std::fmt::Display for BytecodeVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.tag != 0 {
            write!(f, "{}.{}.{}", self.major, self.minor, self.tag)
        } else {
            write!(f, "{}.{}", self.major, self.minor)
        }
    }
}

impl PartialOrd for BytecodeVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BytecodeVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.tag.cmp(&other.tag))
    }
}
