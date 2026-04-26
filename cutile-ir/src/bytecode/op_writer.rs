/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Per-operation bytecode serialization dispatch.
//!
//! Each operation has a specific serialization format determined by its
//! definition in `Ops.td`. The field order is:
//!
//! 1. Result types (type indices)
//! 2. Variadic result count (only if op has variadic results)
//! 3. Flags field (bitfield for optional attributes/operands)
//! 4. Required attributes (in declaration order, inline — no tag prefix)
//! 5. Optional attributes (only if flagged present)
//! 6. Operands (value indices, with or without size encoding)
//! 7. Regions (recursive serialization)
//!
//! Ported from generated `Bytecode.inc` in the `cuda-tile` submodule.

use super::encoding::EncodingWriter;
use super::opcode::Opcode;
use super::writer::WriterCtx;
use crate::ir::{Attribute, Operation};
use crate::{Error, Result};

// =========================================================================
// Per-op dispatch entry point
// =========================================================================

/// Serialize the body of an operation (everything after the opcode varint).
///
/// The opcode itself is written by the caller before calling this function.
/// After this returns, the caller registers the op's result values.
pub(super) fn write_op_body(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    use Opcode::*;
    match op.opcode {
        // ----- Simple: result types + operands (no size) -----
        AbsF | AbsI | AndI | Atan2 | Bitcast | Broadcast | Ceil | Cos | CosH | Exp | Floor
        | IntToPtr | Log | Log2 | MmaF | MulhiI | NegF | Offset | OrI | Pow | PtrToInt
        | PtrToPtr | RemF | Reshape | Select | Sin | SinH | Tan | XOrI => {
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- v13.2: NegI gains overflow attr -----
        NegI => {
            write_result_types(op, w, ctx)?;
            // overflow defaults to NONE (0) if not set
            write_inline_attr_or_default(op, "overflow", 0, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        // ----- v13.2: TanH gains rounding_mode attr -----
        TanH => {
            write_result_types(op, w, ctx)?;
            // rounding_mode defaults to FULL (5) if not set
            write_inline_attr_or_default(op, "rounding_mode", 5, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- No operands, just result types -----
        Iota => {
            write_result_types(op, w, ctx)?;
        }

        // ----- Required attributes only -----
        AddI | MulI | SubI | ShLI | TruncI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "overflow", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        ShRI | ExtI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Cat => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "dim", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Permute => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "permutation", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Assert => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "message", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Assume => {
            write_result_types(op, w, ctx)?;
            write_interface_attr(op, "predicate", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        CmpF => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "comparison_predicate", w, ctx)?;
            write_inline_attr(op, "comparison_ordering", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        CmpI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "comparison_predicate", w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Constant => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "value", w, ctx)?;
        }
        DivI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_inline_attr(op, "rounding", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        FToF => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        FToI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        IToF => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        MaxI | MinI | RemI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        MmaI => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "signedness_lhs", w, ctx)?;
            write_inline_attr(op, "signedness_rhs", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- No operands -----
        GetNumTileBlocks | GetTileBlockId | MakeToken => {
            write_result_types(op, w, ctx)?;
        }

        // ----- Simple: result types + operands (view construction) -----
        MakePartitionView => {
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- JoinTokens: variadic result count + operands with size -----
        JoinTokens => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
        }

        // ----- MakeTensorView: result count + AttrSizedOperandSegments -----
        // Args: base (ptr), dynamicShape (variadic), dynamicStrides (variadic)
        MakeTensorView => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            // ODS 0: base (1 operand, no size)
            write_operand_group(op, w, ctx, 0, 1)?;
            // ODS 1: dynamicShape (variadic, with size)
            write_variadic_operand_group(op, w, ctx, 1)?;
            // ODS 2: dynamicStrides (variadic, with size)
            write_variadic_operand_group(op, w, ctx, 2)?;
        }
        GetGlobal => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "name", w, ctx)?;
        }

        // ----- Extract: result count + variadic operands -----
        Extract => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
        }

        // ----- Flags + optional attributes -----
        AddF | DivF | MulF | SubF => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Fma => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Sqrt => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_inline_attr(op, "rounding_mode", w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }
        Exp2 | Rsqrt => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "flush_to_zero", 0);
            w.write_varint(flags);
            write_operands(op, w, ctx, false)?;
        }
        MaxF | MinF => {
            write_result_types(op, w, ctx)?;
            let flags =
                flag_if_present(op, "propagate_nan", 0) | flag_if_present(op, "flush_to_zero", 1);
            w.write_varint(flags);
            write_operands(op, w, ctx, false)?;
        }

        // ----- Variadic results, no operands -----
        GetIndexSpaceShape | GetTensorShape => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
        }

        // ----- Global: result types + attrs (handled in global section, but also as op) -----
        Global => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "sym_name", w, ctx)?;
            write_inline_attr(op, "value", w, ctx)?;
            write_inline_attr(op, "alignment", w, ctx)?;
        }

        // ----- Module: result types + attr + regions -----
        Module => {
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "sym_name", w, ctx)?;
            write_regions(op, w, ctx)?;
        }

        // ----- Terminators: result count + variadic operands -----
        Break | Continue | Return | Yield => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
        }

        // ----- Print: result count + v13.2 flags + attributes + variadic operands -----
        Print => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            // v13.2: flags field (bit 0 = has token operand)
            // The token is the last operand if present; we check via
            // the "has_token" attr or by result count (token result).
            let flags = 0u64; // TODO: set bit 0 if token present
            w.write_varint(flags);
            write_inline_attr(op, "str", w, ctx)?;
            write_operands(op, w, ctx, true)?;
            // v13.2: optional token operand (not written if flags bit 0 is 0)
        }

        // ----- Variadic results + regions -----
        For => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            // v13.2: flags field for unsignedCmp (bit 0)
            let flags = flag_if_present(op, "unsigned_cmp", 0);
            w.write_varint(flags);
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }
        Loop => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }
        If => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_operands(op, w, ctx, false)?;
            write_regions(op, w, ctx)?;
        }

        // ----- Variadic results + required attributes + regions -----
        Reduce => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "dim", w, ctx)?;
            write_inline_attr(op, "identities", w, ctx)?;
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }
        Scan => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            write_inline_attr(op, "dim", w, ctx)?;
            write_inline_attr(op, "reverse", w, ctx)?;
            write_inline_attr(op, "identities", w, ctx)?;
            write_operands(op, w, ctx, true)?;
            write_regions(op, w, ctx)?;
        }

        // ----- AttrSizedOperandSegments + flags -----
        AtomicCAS => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_operand_group(op, 3, 0) | flag_if_operand_group(op, 4, 1);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            write_inline_attr(op, "memory_scope", w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            write_operand_group(op, w, ctx, 2, 1)?;
            if flags & (1 << 0) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
            if flags & (1 << 1) != 0 {
                write_operand_group(op, w, ctx, 4, 1)?;
            }
        }
        AtomicRMW => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_operand_group(op, 2, 0) | flag_if_operand_group(op, 3, 1);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            write_inline_attr(op, "memory_scope", w, ctx)?;
            write_inline_attr(op, "mode", w, ctx)?;
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            if flags & (1 << 0) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
            if flags & (1 << 1) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        LoadPtrTko => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 1, 2)
                | flag_if_operand_group(op, 2, 3)
                | flag_if_operand_group(op, 3, 4);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_operand_group(op, w, ctx, 0, 1)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 1, 1)?;
            }
            if flags & (1 << 3) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
            if flags & (1 << 4) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        LoadViewTko => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 2, 2);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_operand_group(op, w, ctx, 0, 1)?;
            write_variadic_operand_group(op, w, ctx, 1)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
        }
        StorePtrTko => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 2, 2)
                | flag_if_operand_group(op, 3, 3);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 2, 1)?;
            }
            if flags & (1 << 3) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }
        StoreViewTko => {
            w.write_varint(op.result_types.len() as u64);
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "memory_scope", 0)
                | flag_if_present(op, "optimization_hints", 1)
                | flag_if_operand_group(op, 3, 2);
            w.write_varint(flags);
            write_inline_attr(op, "memory_ordering_semantics", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "memory_scope", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_operand_group(op, w, ctx, 0, 1)?;
            write_operand_group(op, w, ctx, 1, 1)?;
            write_variadic_operand_group(op, w, ctx, 2)?;
            if flags & (1 << 2) != 0 {
                write_operand_group(op, w, ctx, 3, 1)?;
            }
        }

        // ----- Entry (function-like, with flags + optional attrs + regions) -----
        Entry => {
            write_result_types(op, w, ctx)?;
            let flags = flag_if_present(op, "arg_attrs", 0)
                | flag_if_present(op, "res_attrs", 1)
                | flag_if_present(op, "optimization_hints", 2);
            w.write_varint(flags);
            write_inline_attr(op, "sym_name", w, ctx)?;
            write_inline_attr(op, "function_type", w, ctx)?;
            if flags & (1 << 0) != 0 {
                write_inline_attr(op, "arg_attrs", w, ctx)?;
            }
            if flags & (1 << 1) != 0 {
                write_inline_attr(op, "res_attrs", w, ctx)?;
            }
            if flags & (1 << 2) != 0 {
                write_inline_attr(op, "optimization_hints", w, ctx)?;
            }
            write_regions(op, w, ctx)?;
        }

        // All opcodes covered. This arm catches future additions.
        #[allow(unreachable_patterns)]
        _ => {
            return Err(Error::BytecodeWrite(format!(
                "unsupported operation {:?} in bytecode writer",
                op.opcode
            )));
        }
    }
    Ok(())
}

// =========================================================================
// Helpers
// =========================================================================

fn write_result_types(op: &Operation, w: &mut EncodingWriter, ctx: &mut WriterCtx) -> Result<()> {
    for ty in &op.result_types {
        let idx = ctx.types.get_or_insert(ty);
        w.write_varint(idx);
    }
    Ok(())
}

fn write_operands(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
    encode_size: bool,
) -> Result<()> {
    if encode_size {
        w.write_varint(op.operands.len() as u64);
    }
    for &operand in &op.operands {
        let idx = ctx.value_map.get(&operand).copied().ok_or_else(|| {
            Error::BytecodeWrite(format!(
                "operand {:?} not found in value map for op {:?}",
                operand, op.opcode
            ))
        })?;
        w.write_varint(idx);
    }
    Ok(())
}

fn write_operand_group(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
    group_idx: usize,
    expected_size: usize,
) -> Result<()> {
    let offset = operand_group_offset(op, group_idx);
    for i in 0..expected_size {
        let operand = op.operands[offset + i];
        let idx = ctx.value_map.get(&operand).copied().ok_or_else(|| {
            Error::BytecodeWrite(format!(
                "operand group {group_idx}[{i}] not found in value map"
            ))
        })?;
        w.write_varint(idx);
    }
    Ok(())
}

fn write_variadic_operand_group(
    op: &Operation,
    w: &mut EncodingWriter,
    ctx: &WriterCtx,
    group_idx: usize,
) -> Result<()> {
    let sizes = operand_segment_sizes(op);
    let size = sizes.get(group_idx).copied().unwrap_or(0) as usize;
    let offset = operand_group_offset(op, group_idx);
    w.write_varint(size as u64);
    for i in 0..size {
        let operand = op.operands[offset + i];
        let idx = ctx.value_map.get(&operand).copied().ok_or_else(|| {
            Error::BytecodeWrite(format!(
                "variadic operand group {group_idx}[{i}] not found in value map"
            ))
        })?;
        w.write_varint(idx);
    }
    Ok(())
}

fn operand_group_offset(op: &Operation, group_idx: usize) -> usize {
    let sizes = operand_segment_sizes(op);
    sizes.iter().take(group_idx).map(|&s| s as usize).sum()
}

fn operand_segment_sizes(op: &Operation) -> Vec<i32> {
    for (name, attr) in &op.attributes {
        if name == "operandSegmentSizes" {
            if let Attribute::Array(arr) = attr {
                return arr
                    .iter()
                    .map(|a| match a {
                        Attribute::Integer(v, _) => *v as i32,
                        _ => 0,
                    })
                    .collect();
            }
        }
    }
    vec![1i32; op.operands.len()]
}

fn write_regions(op: &Operation, w: &mut EncodingWriter, ctx: &mut WriterCtx) -> Result<()> {
    let region_ids: Vec<_> = op.regions.clone();
    w.write_varint(region_ids.len() as u64);
    for region_id in region_ids {
        ctx.write_region(region_id, w)?;
    }
    Ok(())
}

fn write_inline_attr(
    op: &Operation,
    name: &str,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    let attr = find_op_attr(op, name).ok_or_else(|| {
        Error::BytecodeWrite(format!("missing attribute '{name}' on op {:?}", op.opcode))
    })?;
    write_attr_value_inline(attr, w, ctx)
}

fn write_inline_attr_or_default(
    op: &Operation,
    name: &str,
    default: u64,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    match find_op_attr(op, name) {
        Some(attr) => write_attr_value_inline(attr, w, ctx),
        None => {
            w.write_varint(default);
            Ok(())
        }
    }
}

fn write_interface_attr(
    op: &Operation,
    name: &str,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    let attr = find_op_attr(op, name).ok_or_else(|| {
        Error::BytecodeWrite(format!(
            "missing interface attribute '{name}' on op {:?}",
            op.opcode
        ))
    })?;
    super::writer::write_self_contained_attribute(
        attr,
        w,
        &mut ctx.strings,
        &mut ctx.types,
        &mut ctx.constants,
    )
}

fn write_attr_value_inline(
    attr: &Attribute,
    w: &mut EncodingWriter,
    ctx: &mut WriterCtx,
) -> Result<()> {
    match attr {
        Attribute::Integer(v, _) => {
            w.write_varint(*v as u64);
        }
        Attribute::Float(v, ty) => {
            w.write_ap_float(*v, ty);
        }
        Attribute::Bool(v) => {
            w.write_byte(if *v { 0x01 } else { 0x00 });
        }
        Attribute::String(s) => {
            let idx = ctx.strings.get_or_insert(s);
            w.write_varint(idx);
        }
        Attribute::Type(ty) => {
            let idx = ctx.types.get_or_insert(ty);
            w.write_varint(idx);
        }
        Attribute::DenseI32Array(arr) => {
            w.write_le_var_size_i32(arr);
        }
        Attribute::DenseElements(de) => {
            let mut cdata = EncodingWriter::new();
            cdata.write_varint(de.data.len() as u64);
            cdata.write_bytes(&de.data);
            let idx = ctx.constants.add(cdata.into_bytes());
            w.write_varint(idx);
        }
        Attribute::Array(elems) => {
            // Inline Array: no tag prefix (unlike self-contained).
            // C++ writeSingleAttribute with isSelfContained=false skips the tag.
            w.write_varint(elems.len() as u64);
            for elem in elems {
                super::writer::write_self_contained_attribute(
                    elem,
                    w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }
        Attribute::Dictionary(entries) => {
            // Inline Dictionary: no tag prefix.
            w.write_varint(entries.len() as u64);
            for (key, val) in entries {
                let key_idx = ctx.strings.get_or_insert(key);
                w.write_varint(key_idx);
                super::writer::write_self_contained_attribute(
                    val,
                    w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }
        Attribute::OptimizationHints(oh) => {
            // Inline OptimizationHints: no OptimizationHints tag.
            // C++ writes: inner DictionaryAttr inline (no Dictionary tag either).
            // Format: size + entries where each entry value is self-contained.
            w.write_varint(oh.entries.len() as u64);
            for (arch, hints) in &oh.entries {
                let arch_idx = ctx.strings.get_or_insert(arch);
                w.write_varint(arch_idx);
                // Each arch value is a Dictionary, written self-contained
                // (with Dictionary tag) because it's an entry value.
                super::writer::write_self_contained_attribute(
                    &Attribute::Dictionary(
                        hints.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
                    ),
                    w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }
        _ => {
            super::writer::write_self_contained_attribute(
                attr,
                w,
                &mut ctx.strings,
                &mut ctx.types,
                &mut ctx.constants,
            )?;
        }
    }
    Ok(())
}

fn find_op_attr<'a>(op: &'a Operation, name: &str) -> Option<&'a Attribute> {
    op.attributes
        .iter()
        .find_map(|(k, v)| if k == name { Some(v) } else { None })
}

fn flag_if_present(op: &Operation, attr_name: &str, bit: u32) -> u64 {
    if find_op_attr(op, attr_name).is_some() {
        1u64 << bit
    } else {
        0
    }
}

fn flag_if_operand_group(op: &Operation, group_idx: usize, bit: u32) -> u64 {
    let sizes = operand_segment_sizes(op);
    let size = sizes.get(group_idx).copied().unwrap_or(0);
    if size > 0 {
        1u64 << bit
    } else {
        0
    }
}
