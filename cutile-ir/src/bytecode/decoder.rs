/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode decoder — reads Tile IR bytecode and produces a human-readable
//! text dump for debugging.
//!
//! This is not a full IR reconstructor (no round-trip to `Module`). It reads
//! the raw bytecode sections and prints their contents in a structured format.
//!
//! Ported from `BytecodeReader.cpp` in the `cuda-tile` submodule.

use super::enums::{
    BytecodeVersion, FunctionFlag, Section, TypeTag, ALIGNMENT_BYTE, MAGIC, NUM_SECTIONS,
};
use crate::ir::DYNAMIC;
use crate::{Error, Result};
use std::fmt::Write;

// =========================================================================
// Public API
// =========================================================================

/// Decode a bytecode buffer into a human-readable string.
pub fn decode_bytecode(data: &[u8]) -> Result<String> {
    let mut r = Reader::new(data);
    let mut out = String::new();

    // Header
    let version = r.read_header()?;
    writeln!(out, "TileIR bytecode v{version}").unwrap();
    writeln!(out).unwrap();

    // Collect raw sections first, then parse in dependency order.
    let mut sections = RawSections::default();
    loop {
        let sh = r.read_section_header()?;
        if sh.id == Section::EndOfBytecode as u8 {
            break;
        }
        let payload = r.read_bytes(sh.data_len)?;
        sections.insert(sh.id, payload);
    }

    // Parse string table first (other sections reference strings).
    let strings = parse_string_section(sections.get(Section::String as u8))?;
    if !strings.is_empty() {
        writeln!(out, "=== Strings ({}) ===", strings.len()).unwrap();
        for (i, s) in strings.iter().enumerate() {
            writeln!(out, "  [{i}] {s:?}").unwrap();
        }
        writeln!(out).unwrap();
    }

    // Parse type table.
    let types = parse_type_section(sections.get(Section::Type as u8), version)?;
    if !types.is_empty() {
        writeln!(out, "=== Types ({}) ===", types.len()).unwrap();
        for (i, t) in types.iter().enumerate() {
            writeln!(out, "  [{i}] {t}").unwrap();
        }
        writeln!(out).unwrap();
    }

    // Constant section.
    let constants = parse_constant_section(sections.get(Section::Constant as u8))?;
    if !constants.is_empty() {
        writeln!(out, "=== Constants ({}) ===", constants.len()).unwrap();
        for (i, c) in constants.iter().enumerate() {
            writeln!(out, "  [{i}] {} bytes", c.len()).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Global section.
    if let Some(payload) = sections.get(Section::Global as u8) {
        let globals = parse_global_section(payload, &strings, &types, version)?;
        if !globals.is_empty() {
            writeln!(out, "=== Globals ({}) ===", globals.len()).unwrap();
            for g in &globals {
                writeln!(out, "  {g}").unwrap();
            }
            writeln!(out).unwrap();
        }
    }

    // Function section.
    if let Some(payload) = sections.get(Section::Func as u8) {
        let funcs = parse_func_section(payload, &strings, &types)?;
        writeln!(out, "=== Functions ({}) ===", funcs.len()).unwrap();
        for f in &funcs {
            writeln!(out, "{f}").unwrap();
        }
    }

    // Debug section (just report size for now).
    if let Some(payload) = sections.get(Section::Debug as u8) {
        writeln!(out, "=== Debug ({} bytes) ===", payload.len()).unwrap();
    }

    Ok(out)
}

/// Convenience: decode bytecode from a file.
pub fn decode_bytecode_file(path: &str) -> Result<String> {
    let data = std::fs::read(path)
        .map_err(|e| Error::BytecodeWrite(format!("failed to read {path}: {e}")))?;
    decode_bytecode(&data)
}

// =========================================================================
// Raw section collector
// =========================================================================

#[derive(Default)]
struct RawSections<'a> {
    data: [Option<&'a [u8]>; NUM_SECTIONS as usize],
}

impl<'a> RawSections<'a> {
    fn insert(&mut self, id: u8, payload: &'a [u8]) {
        if (id as usize) < self.data.len() {
            self.data[id as usize] = Some(payload);
        }
    }
    fn get(&self, id: u8) -> Option<&'a [u8]> {
        self.data.get(id as usize).copied().flatten()
    }
}

// =========================================================================
// Section header
// =========================================================================

struct SectionHeader {
    id: u8,
    data_len: usize,
}

// =========================================================================
// Low-level reader
// =========================================================================

/// Low-level bytecode reader (mirror of EncodingWriter).
#[allow(dead_code)]
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

#[allow(dead_code)]
impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_byte(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(err("unexpected end of data"));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(err("unexpected end of data"));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            let b = self.read_byte()?;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(err("varint overflow"));
            }
        }
        Ok(result)
    }

    fn read_signed_varint(&mut self) -> Result<i64> {
        let v = self.read_varint()?;
        // zigzag decode
        Ok(((v >> 1) as i64) ^ (-((v & 1) as i64)))
    }

    fn read_le_u8(&mut self) -> Result<u8> {
        self.read_byte()
    }

    fn read_le_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_le_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_le_i64(&mut self) -> Result<i64> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_le_i32(&mut self) -> Result<i32> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Decode an arbitrary-precision float for the given scalar type — the
    /// inverse of [`EncodingWriter::write_ap_float`](super::encoding::EncodingWriter::write_ap_float).
    /// F16/BF16/F32/F64/TF32 are stored as their bit pattern in a signed varint;
    /// F8 variants as a single byte.
    fn read_ap_float(&mut self, scalar: crate::ir::ScalarType) -> Result<f64> {
        use crate::ir::ScalarType::*;
        Ok(match scalar {
            F16 => half::f16::from_bits(self.read_signed_varint()? as u16).to_f64(),
            BF16 => half::bf16::from_bits(self.read_signed_varint()? as u16).to_f64(),
            F32 | TF32 => f32::from_bits(self.read_signed_varint()? as u32) as f64,
            F64 => f64::from_bits(self.read_signed_varint()? as u64),
            F8E4M3FN | F8E5M2 | F8E8M0FNU | F4E2M1FN => {
                // Sub-byte floats are stored as a raw byte; full numeric decode
                // would mirror f64_to_f8*; not needed for the common (f32/f64) attrs.
                let _ = self.read_byte()?;
                return Err(err("sub-byte float attribute decode not yet implemented"));
            }
            other => return Err(err(&format!("{other:?} is not a float scalar"))),
        })
    }

    fn skip_padding(&mut self, alignment: u64) -> Result<()> {
        if alignment < 2 {
            return Ok(());
        }
        let padding = (alignment - (self.pos as u64 % alignment)) % alignment;
        for _ in 0..padding {
            let b = self.read_byte()?;
            if b != ALIGNMENT_BYTE {
                return Err(err(&format!(
                    "expected padding byte 0x{ALIGNMENT_BYTE:02X}, got 0x{b:02X}"
                )));
            }
        }
        Ok(())
    }

    fn read_header(&mut self) -> Result<BytecodeVersion> {
        let magic = self.read_bytes(8)?;
        if magic != MAGIC {
            return Err(err("invalid magic number"));
        }
        let major = self.read_le_u8()?;
        let minor = self.read_le_u8()?;
        let tag = self.read_le_u16()?;
        Ok(BytecodeVersion { major, minor, tag })
    }

    fn read_section_header(&mut self) -> Result<SectionHeader> {
        let id_and_align = self.read_byte()?;
        let id = id_and_align & 0x7F;
        let has_alignment = id_and_align & 0x80 != 0;

        if id == Section::EndOfBytecode as u8 {
            return Ok(SectionHeader { id, data_len: 0 });
        }

        let length = self.read_varint()? as usize;
        if has_alignment {
            let alignment = self.read_varint()?;
            self.skip_padding(alignment)?;
        }

        Ok(SectionHeader {
            id,
            data_len: length,
        })
    }
}

// =========================================================================
// String section parser
// =========================================================================

fn parse_string_section(payload: Option<&[u8]>) -> Result<Vec<String>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(4)?;

    // Read offset table.
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(r.read_le_u32()? as usize);
    }

    // Remaining bytes are concatenated string data.
    let string_data = r.read_bytes(r.remaining())?;

    let mut strings = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            string_data.len()
        };
        // reject out-of-range/backwards offsets (untrusted) rather than panic.
        if start > end || end > string_data.len() {
            return Err(err("string offset out of range"));
        }
        let s = std::str::from_utf8(&string_data[start..end])
            .unwrap_or("<invalid utf8>")
            .to_owned();
        strings.push(s);
    }
    Ok(strings)
}

// =========================================================================
// Type section parser
// =========================================================================

fn parse_type_section(payload: Option<&[u8]>, version: BytecodeVersion) -> Result<Vec<String>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(4)?;

    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        offsets.push(r.read_le_u32()? as usize);
    }

    let type_data = r.read_bytes(r.remaining())?;
    let mut types = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            type_data.len()
        };
        let desc = decode_type_entry(&type_data[start..end], &types, version);
        types.push(desc);
    }
    Ok(types)
}

fn decode_type_entry(data: &[u8], prev_types: &[String], version: BytecodeVersion) -> String {
    let mut r = Reader::new(data);
    let Ok(tag_val) = r.read_varint() else {
        return "<read error>".into();
    };
    let tag = tag_val as u8;
    match tag {
        t if t == TypeTag::I1 as u8 => "i1".into(),
        t if t == TypeTag::I4 as u8 => "i4".into(),
        t if t == TypeTag::I8 as u8 => "i8".into(),
        t if t == TypeTag::I16 as u8 => "i16".into(),
        t if t == TypeTag::I32 as u8 => "i32".into(),
        t if t == TypeTag::I64 as u8 => "i64".into(),
        t if t == TypeTag::F16 as u8 => "f16".into(),
        t if t == TypeTag::BF16 as u8 => "bf16".into(),
        t if t == TypeTag::F32 as u8 => "f32".into(),
        t if t == TypeTag::TF32 as u8 => "tf32".into(),
        t if t == TypeTag::F64 as u8 => "f64".into(),
        t if t == TypeTag::F8E4M3FN as u8 => "f8e4m3fn".into(),
        t if t == TypeTag::F8E5M2 as u8 => "f8e5m2".into(),
        t if t == TypeTag::F8E8M0FNU as u8 => "f8e8m0fnu".into(),
        t if t == TypeTag::F4E2M1FN as u8 => "f4e2m1fn".into(),
        t if t == TypeTag::Token as u8 => "token".into(),
        t if t == TypeTag::Pointer as u8 => {
            let elem = r.read_varint().unwrap_or(0) as usize;
            let elem_name = prev_types.get(elem).cloned().unwrap_or("?".into());
            format!("ptr<{elem_name}>")
        }
        t if t == TypeTag::Tile as u8 => {
            let elem = r.read_varint().unwrap_or(0) as usize;
            let elem_name = prev_types.get(elem).cloned().unwrap_or("?".into());
            let rank = r.read_varint().unwrap_or(0) as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(r.read_le_i64().unwrap_or(0));
            }
            let shape_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            if shape_str.is_empty() {
                format!("tile<{elem_name}>")
            } else {
                format!("tile<{shape_str}x{elem_name}>")
            }
        }
        t if t == TypeTag::TensorView as u8 => {
            let elem = r.read_varint().unwrap_or(0) as usize;
            let elem_name = prev_types.get(elem).cloned().unwrap_or("?".into());
            let rank = r.read_varint().unwrap_or(0) as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(r.read_le_i64().unwrap_or(0));
            }
            let stride_count = r.read_varint().unwrap_or(0) as usize;
            let mut strides = Vec::with_capacity(stride_count);
            for _ in 0..stride_count {
                strides.push(r.read_le_i64().unwrap_or(0));
            }
            let shape_str = shape
                .iter()
                .map(|d| {
                    if *d == DYNAMIC {
                        "?".into()
                    } else {
                        d.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("x");
            let stride_str = strides
                .iter()
                .map(|d| {
                    if *d == DYNAMIC {
                        "?".into()
                    } else {
                        d.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("tensor_view<{shape_str}x{elem_name}, strides=[{stride_str}]>")
        }
        t if t == TypeTag::PartitionView as u8 => {
            let mut padding_flag_from_bitfield = None;
            if version >= BytecodeVersion::V13_3 {
                let flags = r.read_varint().unwrap_or(0);
                padding_flag_from_bitfield = Some(flags & 1 != 0);
            }
            let tile_rank = r.read_varint().unwrap_or(0) as usize;
            let mut tile_shape = Vec::with_capacity(tile_rank);
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let dim_map_size = r.read_varint().unwrap_or(0) as usize;
            let mut dim_map = Vec::with_capacity(dim_map_size);
            for _ in 0..dim_map_size {
                dim_map.push(r.read_le_i32().unwrap_or(0));
            }
            let has_padding =
                padding_flag_from_bitfield.unwrap_or_else(|| r.read_byte().unwrap_or(0) != 0);
            let padding = if has_padding {
                let p = if version >= BytecodeVersion::V13_3 {
                    r.read_byte().unwrap_or(0) as u64
                } else {
                    r.read_varint().unwrap_or(0)
                };
                format!(", padding={p}")
            } else {
                String::new()
            };
            let ts = tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            format!("partition_view<tile=({ts}), tv={tv_name}{padding}>")
        }
        t if t == TypeTag::GatherScatterView as u8 => {
            let flags = r.read_varint().unwrap_or(0);
            let tile_rank = r.read_varint().unwrap_or(0) as usize;
            let mut tile_shape = Vec::with_capacity(tile_rank);
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let sparse_dim = r.read_varint().unwrap_or(0);
            let padding = if flags & 1 != 0 {
                let p = r.read_byte().unwrap_or(0);
                format!(", padding={p}")
            } else {
                String::new()
            };
            let ts = tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            format!(
                "gather_scatter_view<tile=({ts}), tv={tv_name}, sparse_dim={sparse_dim}{padding}>"
            )
        }
        t if t == TypeTag::StridedView as u8 => {
            let flags = r.read_varint().unwrap_or(0);
            let tile_rank = r.read_varint().unwrap_or(0) as usize;
            let mut tile_shape = Vec::with_capacity(tile_rank);
            for _ in 0..tile_rank {
                tile_shape.push(r.read_le_i32().unwrap_or(0));
            }
            let stride_rank = r.read_varint().unwrap_or(0) as usize;
            let mut traversal_strides = Vec::with_capacity(stride_rank);
            for _ in 0..stride_rank {
                traversal_strides.push(r.read_le_i32().unwrap_or(0));
            }
            let tv_idx = r.read_varint().unwrap_or(0) as usize;
            let tv_name = prev_types.get(tv_idx).cloned().unwrap_or("?".into());
            let dim_map_size = r.read_varint().unwrap_or(0) as usize;
            let mut dim_map = Vec::with_capacity(dim_map_size);
            for _ in 0..dim_map_size {
                dim_map.push(r.read_le_i32().unwrap_or(0));
            }
            let padding = if flags & 1 != 0 {
                let p = r.read_byte().unwrap_or(0);
                format!(", padding={p}")
            } else {
                String::new()
            };
            let ts = tile_shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let strides = traversal_strides
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let dim_map_str = dim_map
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "strided_view<tile=({ts}), traversal_strides=[{strides}], tv={tv_name}, dim_map=[{dim_map_str}]{padding}>"
            )
        }
        t if t == TypeTag::Func as u8 => {
            let num_inputs = r.read_varint().unwrap_or(0) as usize;
            let mut inputs = Vec::with_capacity(num_inputs);
            for _ in 0..num_inputs {
                let idx = r.read_varint().unwrap_or(0) as usize;
                inputs.push(prev_types.get(idx).cloned().unwrap_or("?".into()));
            }
            let num_results = r.read_varint().unwrap_or(0) as usize;
            let mut results = Vec::with_capacity(num_results);
            for _ in 0..num_results {
                let idx = r.read_varint().unwrap_or(0) as usize;
                results.push(prev_types.get(idx).cloned().unwrap_or("?".into()));
            }
            format!("({}) -> ({})", inputs.join(", "), results.join(", "))
        }
        _ => format!("<unknown type tag {tag}>"),
    }
}

// =========================================================================
// Constant section parser
// =========================================================================

fn parse_constant_section(payload: Option<&[u8]>) -> Result<Vec<Vec<u8>>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(8)?;

    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        // Offsets are u64 LE (raw array), not varints.
        let bytes = r.read_bytes(8)?;
        let v = u64::from_le_bytes(bytes.try_into().unwrap());
        offsets.push(v as usize);
    }

    let const_data = r.read_bytes(r.remaining())?;
    let mut constants = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count {
            offsets[i + 1]
        } else {
            const_data.len()
        };
        constants.push(const_data[start..end].to_vec());
    }
    Ok(constants)
}

// =========================================================================
// Global section parser
// =========================================================================

fn parse_global_section(
    data: &[u8],
    strings: &[String],
    types: &[String],
    version: BytecodeVersion,
) -> Result<Vec<String>> {
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    let mut globals = Vec::with_capacity(count);
    for _ in 0..count {
        let name_idx = r.read_varint()? as usize;
        let type_idx = r.read_varint()? as usize;
        let const_idx = r.read_varint()? as usize;
        let alignment = r.read_varint()?;
        let mut visibility = None;
        let mut constant = None;
        if version >= BytecodeVersion::V13_3 {
            visibility = Some(match r.read_byte()? {
                0 => "public",
                1 => "private",
                _ => "unknown",
            });
            constant = Some(r.read_varint()? != 0);
        }
        let name = strings.get(name_idx).cloned().unwrap_or("?".into());
        let ty = types.get(type_idx).cloned().unwrap_or("?".into());
        let suffix = match (visibility, constant) {
            (Some(vis), Some(is_constant)) => format!(", {vis}, constant={is_constant}"),
            _ => String::new(),
        };
        globals.push(format!(
            "@{name} : {ty} = const[{const_idx}], align {alignment}{suffix}"
        ));
    }
    Ok(globals)
}

// =========================================================================
// Function section parser
// =========================================================================

fn parse_func_section(data: &[u8], strings: &[String], types: &[String]) -> Result<Vec<String>> {
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    let mut funcs = Vec::with_capacity(count);

    for _ in 0..count {
        let name_idx = r.read_varint()? as usize;
        let sig_idx = r.read_varint()? as usize;
        let flags_byte = r.read_byte()?;
        let _loc_idx = r.read_varint()?;

        let name = strings.get(name_idx).cloned().unwrap_or("?".into());
        let sig = types.get(sig_idx).cloned().unwrap_or("?".into());

        let is_kernel = flags_byte & FunctionFlag::KindKernel as u8 != 0;
        let has_hints = flags_byte & FunctionFlag::HasOptimizationHints as u8 != 0;
        let kind = if is_kernel { "entry" } else { "func" };

        // Skip optimization hints if present (self-contained attribute).
        if has_hints {
            // Skip the self-contained attribute — we'd need full attribute
            // decoding to display it. For now just note it.
            // TODO: decode optimization hints attribute.
        }

        let body_len = r.read_varint()? as usize;
        let body_data = r.read_bytes(body_len)?;
        let op_count = count_ops_in_body(body_data);

        let mut out = String::new();
        writeln!(out, "  {kind} @{name} : {sig}").unwrap();
        writeln!(out, "    body: {body_len} bytes, ~{op_count} ops").unwrap();
        if has_hints {
            writeln!(out, "    [has optimization_hints]").unwrap();
        }
        funcs.push(out);
    }
    Ok(funcs)
}

/// Quick heuristic: count opcodes in a function body by scanning varints.
fn count_ops_in_body(data: &[u8]) -> usize {
    // This is an approximation — a proper count requires full per-op parsing.
    // For now just report the body byte size.
    // TODO: implement full per-op decoding for function bodies.
    data.len() // placeholder: return byte count, not op count
}

// =========================================================================
// Helpers
// =========================================================================

fn err(msg: &str) -> Error {
    Error::BytecodeWrite(format!("decode: {msg}"))
}

// =========================================================================
// Type section -> Vec<Type>  (mirror of writer::serialize_type)
// =========================================================================

use crate::bytecode::enums::AttributeTag;
use crate::ir::{
    Attribute, Bounded, DenseElements, DivBy, FuncType, GatherScatterViewType, OptimizationHints,
    PaddingValue, PartitionViewType, PointerType, SameElements, ScalarType, StridedViewType,
    TensorViewType, TileElementType, TileType, Type,
};

/// Decode a self-contained (tag-prefixed) attribute — the inverse of
/// `writer::write_self_contained_attribute`.
fn read_self_contained_attribute(
    r: &mut Reader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Attribute> {
    let ty_at = |i: usize| {
        types
            .get(i)
            .cloned()
            .ok_or_else(|| err("attr type index out of range"))
    };
    let str_at = |i: usize| {
        strings
            .get(i)
            .cloned()
            .ok_or_else(|| err("attr string index out of range"))
    };
    let tag = r.read_varint()?;
    Ok(match tag {
        t if t == AttributeTag::Integer as u64 => {
            let ty = ty_at(r.read_varint()? as usize)?;
            Attribute::Integer(r.read_varint()? as i64, ty)
        }
        t if t == AttributeTag::Float as u64 => {
            let ty = ty_at(r.read_varint()? as usize)?;
            let v = match &ty {
                Type::Scalar(s) => r.read_ap_float(*s)?,
                _ => f64::from_bits(r.read_signed_varint()? as u64),
            };
            Attribute::Float(v, ty)
        }
        t if t == AttributeTag::Bool as u64 => Attribute::Bool(r.read_byte()? != 0),
        t if t == AttributeTag::Type as u64 => Attribute::Type(ty_at(r.read_varint()? as usize)?),
        t if t == AttributeTag::String as u64 => {
            Attribute::String(str_at(r.read_varint()? as usize)?)
        }
        t if t == AttributeTag::Array as u64 => {
            let n = r.read_varint()? as usize;
            let mut elems = Vec::with_capacity(n);
            for _ in 0..n {
                elems.push(read_self_contained_attribute(r, strings, types, constants)?);
            }
            Attribute::Array(elems)
        }
        t if t == AttributeTag::DenseElements as u64 => {
            let element_type = ty_at(r.read_varint()? as usize)?;
            let const_idx = r.read_varint()? as usize;
            let raw = constants
                .get(const_idx)
                .ok_or_else(|| err("dense const index out of range"))?;
            let mut cr = Reader::new(raw);
            let len = cr.read_varint()? as usize;
            let data = cr.read_bytes(len)?.to_vec();
            // The writer does not serialize `shape`; it is not recoverable here.
            Attribute::DenseElements(DenseElements {
                element_type,
                shape: Vec::new(),
                data,
            })
        }
        t if t == AttributeTag::DivBy as u64 => {
            let divisor = r.read_varint()?;
            let flags = r.read_byte()?;
            let every = if flags & 0x01 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            let along = if flags & 0x02 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            Attribute::DivBy(DivBy {
                divisor,
                every,
                along,
            })
        }
        t if t == AttributeTag::SameElements as u64 => Attribute::SameElements(SameElements {
            values: r.read_var_size_i64()?,
        }),
        t if t == AttributeTag::Dictionary as u64 => {
            Attribute::Dictionary(read_dictionary_entries(r, strings, types, constants)?)
        }
        t if t == AttributeTag::OptimizationHints as u64 => {
            let n = r.read_varint()? as usize;
            let mut entries = Vec::with_capacity(n);
            for _ in 0..n {
                let arch = str_at(r.read_varint()? as usize)?;
                // Inner value is a Dictionary written with its tag.
                let dtag = r.read_varint()?;
                if dtag != AttributeTag::Dictionary as u64 {
                    return Err(err("optimization_hints entry is not a dictionary"));
                }
                entries.push((arch, read_dictionary_entries(r, strings, types, constants)?));
            }
            Attribute::OptimizationHints(OptimizationHints { entries })
        }
        t if t == AttributeTag::Bounded as u64 => {
            let flags = r.read_byte()?;
            let lb = if flags & 0x01 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            let ub = if flags & 0x02 != 0 {
                Some(r.read_signed_varint()?)
            } else {
                None
            };
            Attribute::Bounded(Bounded { lb, ub })
        }
        other => return Err(err(&format!("unknown attribute tag {other}"))),
    })
}

/// Inline `Array` attribute (no tag): count + `count` self-contained elements.
/// Mirrors the `Attribute::Array` arm of `write_attr_value_inline`.
fn read_inline_array(
    r: &mut Reader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Vec<Attribute>> {
    let n = r.read_varint()? as usize;
    (0..n)
        .map(|_| read_self_contained_attribute(r, strings, types, constants))
        .collect()
}

/// Inline `OptimizationHints` (no tag): count + `count` (arch_idx, self-contained
/// Dictionary) entries. Mirrors the `OptimizationHints` arm of `write_attr_value_inline`.
fn read_inline_opt_hints(
    r: &mut Reader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<OptimizationHints> {
    let n = r.read_varint()? as usize;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let arch = strings
            .get(r.read_varint()? as usize)
            .cloned()
            .ok_or_else(|| err("opt-hints arch index out of range"))?;
        match read_self_contained_attribute(r, strings, types, constants)? {
            Attribute::Dictionary(d) => entries.push((arch, d)),
            _ => return Err(err("opt-hints entry value is not a dictionary")),
        }
    }
    Ok(OptimizationHints { entries })
}

/// `count` (string_idx, self-contained value) pairs.
fn read_dictionary_entries(
    r: &mut Reader,
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
) -> Result<Vec<(String, Attribute)>> {
    let n = r.read_varint()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let key = strings
            .get(r.read_varint()? as usize)
            .cloned()
            .ok_or_else(|| err("dict key index out of range"))?;
        out.push((
            key,
            read_self_contained_attribute(r, strings, types, constants)?,
        ));
    }
    Ok(out)
}

impl<'a> Reader<'a> {
    /// varint count followed by `count` little-endian i64s.
    fn read_var_size_i64(&mut self) -> Result<Vec<i64>> {
        let n = self.read_varint()? as usize;
        (0..n).map(|_| self.read_le_i64()).collect()
    }
    /// varint count followed by `count` little-endian i32s.
    fn read_var_size_i32(&mut self) -> Result<Vec<i32>> {
        let n = self.read_varint()? as usize;
        (0..n).map(|_| self.read_le_i32()).collect()
    }
}

fn scalar_from_tag(tag: u64) -> Option<ScalarType> {
    use ScalarType::*;
    Some(match tag {
        x if x == TypeTag::I1 as u64 => I1,
        x if x == TypeTag::I4 as u64 => I4,
        x if x == TypeTag::I8 as u64 => I8,
        x if x == TypeTag::I16 as u64 => I16,
        x if x == TypeTag::I32 as u64 => I32,
        x if x == TypeTag::I64 as u64 => I64,
        x if x == TypeTag::F16 as u64 => F16,
        x if x == TypeTag::BF16 as u64 => BF16,
        x if x == TypeTag::F32 as u64 => F32,
        x if x == TypeTag::TF32 as u64 => TF32,
        x if x == TypeTag::F64 as u64 => F64,
        x if x == TypeTag::F8E4M3FN as u64 => F8E4M3FN,
        x if x == TypeTag::F8E5M2 as u64 => F8E5M2,
        x if x == TypeTag::F8E8M0FNU as u64 => F8E8M0FNU,
        x if x == TypeTag::F4E2M1FN as u64 => F4E2M1FN,
        _ => return None,
    })
}

fn padding_from_u8(b: u8) -> Result<PaddingValue> {
    use PaddingValue::*;
    Ok(match b {
        0 => Zero,
        1 => NegZero,
        2 => Nan,
        3 => PosInf,
        4 => NegInf,
        _ => return Err(err(&format!("invalid padding value {b}"))),
    })
}

/// Fetch an already-decoded `TensorViewType` referenced by index.
fn prior_tensor_view(prior: &[Type], idx: usize) -> Result<TensorViewType> {
    match prior.get(idx) {
        Some(Type::TensorView(tv)) => Ok(tv.clone()),
        _ => Err(err("expected a tensor_view at referenced type index")),
    }
}

/// Decode one type entry; sub-type references point into `prior`
/// (types are stored dependency-first, so all referents are already decoded).
fn decode_one_type(r: &mut Reader, prior: &[Type], version: BytecodeVersion) -> Result<Type> {
    let tag = r.read_varint()?;
    if let Some(scalar) = scalar_from_tag(tag) {
        return Ok(Type::Scalar(scalar));
    }
    Ok(match tag {
        t if t == TypeTag::Pointer as u64 => {
            let idx = r.read_varint()? as usize;
            match prior.get(idx) {
                Some(Type::Scalar(s)) => Type::Pointer(PointerType { pointee: *s }),
                _ => return Err(err("pointer pointee is not a scalar")),
            }
        }
        t if t == TypeTag::Tile as u64 => {
            let elem_idx = r.read_varint()? as usize;
            let shape = r.read_var_size_i64()?;
            let element_type = match prior.get(elem_idx) {
                Some(Type::Scalar(s)) => TileElementType::Scalar(*s),
                Some(Type::Pointer(p)) => TileElementType::Pointer(Box::new(p.clone())),
                _ => return Err(err("tile element is not scalar/pointer")),
            };
            Type::Tile(TileType {
                shape,
                element_type,
            })
        }
        t if t == TypeTag::TensorView as u64 => {
            let elem_idx = r.read_varint()? as usize;
            let shape = r.read_var_size_i64()?;
            let strides = r.read_var_size_i64()?;
            let element_type = match prior.get(elem_idx) {
                Some(Type::Scalar(s)) => *s,
                _ => return Err(err("tensor_view element is not a scalar")),
            };
            Type::TensorView(TensorViewType {
                element_type,
                shape,
                strides,
            })
        }
        t if t == TypeTag::PartitionView as u64 => {
            let mut has_padding = false;
            if version >= BytecodeVersion::V13_3 {
                has_padding = r.read_varint()? & 1 != 0;
            }
            let tile_shape = r.read_var_size_i32()?;
            let tv = prior_tensor_view(prior, r.read_varint()? as usize)?;
            let dim_map = r.read_var_size_i32()?;
            let padding_value = if version >= BytecodeVersion::V13_3 {
                if has_padding {
                    Some(padding_from_u8(r.read_byte()?)?)
                } else {
                    None
                }
            } else if r.read_byte()? != 0 {
                Some(padding_from_u8(r.read_varint()? as u8)?)
            } else {
                None
            };
            Type::PartitionView(PartitionViewType {
                tile_shape,
                tensor_view: tv,
                dim_map,
                padding_value,
            })
        }
        t if t == TypeTag::GatherScatterView as u64 => {
            let has_padding = r.read_varint()? != 0;
            let tile_shape = r.read_var_size_i32()?;
            let tv = prior_tensor_view(prior, r.read_varint()? as usize)?;
            let sparse_dim = r.read_varint()? as i32;
            let padding_value = if has_padding {
                Some(padding_from_u8(r.read_byte()?)?)
            } else {
                None
            };
            Type::GatherScatterView(GatherScatterViewType {
                tile_shape,
                tensor_view: tv,
                sparse_dim,
                padding_value,
            })
        }
        t if t == TypeTag::StridedView as u64 => {
            let has_padding = r.read_varint()? != 0;
            let tile_shape = r.read_var_size_i32()?;
            let traversal_strides = r.read_var_size_i32()?;
            let tv = prior_tensor_view(prior, r.read_varint()? as usize)?;
            let dim_map = r.read_var_size_i32()?;
            let padding_value = if has_padding {
                Some(padding_from_u8(r.read_byte()?)?)
            } else {
                None
            };
            Type::StridedView(StridedViewType {
                tile_shape,
                traversal_strides,
                tensor_view: tv,
                dim_map,
                padding_value,
            })
        }
        t if t == TypeTag::Func as u64 => {
            let n_in = r.read_varint()? as usize;
            let inputs = (0..n_in)
                .map(|_| {
                    prior
                        .get(r.read_varint()? as usize)
                        .cloned()
                        .ok_or_else(|| err("bad func input type idx"))
                })
                .collect::<Result<Vec<_>>>()?;
            let n_res = r.read_varint()? as usize;
            let results = (0..n_res)
                .map(|_| {
                    prior
                        .get(r.read_varint()? as usize)
                        .cloned()
                        .ok_or_else(|| err("bad func result type idx"))
                })
                .collect::<Result<Vec<_>>>()?;
            Type::Func(FuncType { inputs, results })
        }
        t if t == TypeTag::Token as u64 => Type::Token,
        other => return Err(err(&format!("unknown type tag {other}"))),
    })
}

// =========================================================================
// Operation body reader  (mirror of op_writer::write_op_body)
// =========================================================================

use crate::builder::{append_op, build_single_block_region, OpBuilder};
use crate::bytecode::Opcode;
use crate::ir::{BlockId, Location, Module, Value};

/// Per-function-body reader state: the decoded section tables plus the running
/// SSA value table (index -> Value), mirroring the writer's `value_map`.
struct OpCtx<'t> {
    types: &'t [Type],
    strings: &'t [String],
    constants: &'t [Vec<u8>],
    values: Vec<Value>,
    /// Source version; gates op-body fields added after an op's baseline.
    version: BytecodeVersion,
}

impl<'t> OpCtx<'t> {
    fn ty(&self, i: usize) -> Result<Type> {
        self.types
            .get(i)
            .cloned()
            .ok_or_else(|| err("result type index out of range"))
    }
    fn val(&self, i: usize) -> Result<Value> {
        self.values
            .get(i)
            .copied()
            .ok_or_else(|| err("operand value index out of range"))
    }
    fn string(&self, i: usize) -> Result<String> {
        self.strings
            .get(i)
            .cloned()
            .ok_or_else(|| err("string index out of range"))
    }
    fn read_operands(&self, r: &mut Reader, n: usize) -> Result<Vec<Value>> {
        (0..n)
            .map(|_| self.val(r.read_varint()? as usize))
            .collect()
    }
    /// varint size + that many operand indices.
    fn read_var_operands(&self, r: &mut Reader) -> Result<(usize, Vec<Value>)> {
        let n = r.read_varint()? as usize;
        Ok((n, self.read_operands(r, n)?))
    }
}

/// `i32`-typed integer attribute (the type isn't on the wire for inline ints).
fn int_attr(v: u64) -> Attribute {
    Attribute::Integer(v as i64, Type::Scalar(ScalarType::I32))
}

fn read_result_types(opcode: Opcode, r: &mut Reader, ctx: &OpCtx) -> Result<Vec<Type>> {
    let n = match opcode.fixed_result_count() {
        Some(n) => n,
        None => r.read_varint()? as usize,
    };
    (0..n).map(|_| ctx.ty(r.read_varint()? as usize)).collect()
}

/// Decode one operation from a function body, append it to `block`, and register
/// its result values. Returns `Ok(false)` for ops not yet supported by the reader.
fn read_op(
    opcode: Opcode,
    r: &mut Reader,
    ctx: &mut OpCtx,
    module: &mut Module,
    block: BlockId,
) -> Result<()> {
    use Opcode::*;
    let result_types = read_result_types(opcode, r, ctx)?;
    let mut attrs: Vec<(String, Attribute)> = Vec::new();
    let mut operands: Vec<Value> = Vec::new();
    let mut seg: Vec<i32> = Vec::new(); // operandSegmentSizes for grouped ops
    let mut regions: Vec<crate::ir::RegionId> = Vec::new();

    // helper to read N fixed operands into `operands`
    macro_rules! fixed_ops {
        ($n:expr) => {{
            operands = ctx.read_operands(r, $n)?;
        }};
    }

    match opcode {
        // -- no operands --
        MakeToken | GetTileBlockId | GetNumTileBlocks | Iota => {}

        // -- pure unary/binary/ternary (result types + fixed operands, no attrs) --
        AbsF
        | AbsI
        | AndI
        | Atan2
        | Bitcast
        | Broadcast
        | Ceil
        | Cos
        | CosH
        | Floor
        | IntToPtr
        | Log
        | Log2
        | MulhiI
        | NegF
        | Offset
        | OrI
        | Pow
        | PtrToInt
        | PtrToPtr
        | RemF
        | Reshape
        | Select
        | Sin
        | SinH
        | Tan
        | XOrI
        | MakePartitionView
        | MmaFScaled
        | MakeStridedView
        | MakeGatherScatterView
        | Pack
        | Unpack => {
            fixed_ops!(opcode.fixed_operand_count().unwrap());
        }
        Alloca => {
            // result + flags(global bit0) + num_elem + alignment (no operands)
            let global = r.read_varint()? & 1 != 0;
            attrs.push(("num_elem".into(), int_attr(r.read_varint()?)));
            attrs.push(("alignment".into(), int_attr(r.read_varint()?)));
            if global {
                attrs.push(("global".into(), Attribute::Bool(true)));
            }
        }

        // -- leading inline integer attr(s), then operands --
        AddI | MulI | SubI | ShLI | TruncI => {
            attrs.push(("overflow".into(), int_attr(r.read_varint()?)));
            fixed_ops!(opcode.fixed_operand_count().unwrap());
        }
        NegI => {
            // overflow added v13.2; elide default 0 (NONE) so older bodies (which
            // omit it) and re-encoded ones (or_default 0) decode the same.
            if ctx.version >= BytecodeVersion::V13_2 {
                let overflow = r.read_varint()?;
                if overflow != 0 {
                    attrs.push(("overflow".into(), int_attr(overflow)));
                }
            }
            fixed_ops!(1);
        }
        ShRI | ExtI | MaxI | MinI | RemI => {
            attrs.push(("signedness".into(), int_attr(r.read_varint()?)));
            fixed_ops!(opcode.fixed_operand_count().unwrap());
        }
        Cat => {
            attrs.push(("dim".into(), int_attr(r.read_varint()?)));
            fixed_ops!(2);
        }
        Permute => {
            attrs.push((
                "permutation".into(),
                Attribute::DenseI32Array(r.read_var_size_i32()?),
            ));
            fixed_ops!(1);
        }
        CmpF => {
            attrs.push(("comparison_predicate".into(), int_attr(r.read_varint()?)));
            attrs.push(("comparison_ordering".into(), int_attr(r.read_varint()?)));
            fixed_ops!(2);
        }
        CmpI => {
            attrs.push(("comparison_predicate".into(), int_attr(r.read_varint()?)));
            attrs.push(("signedness".into(), int_attr(r.read_varint()?)));
            fixed_ops!(2);
        }
        DivI => {
            attrs.push(("signedness".into(), int_attr(r.read_varint()?)));
            attrs.push(("rounding".into(), int_attr(r.read_varint()?)));
            fixed_ops!(2);
        }
        FToI | IToF => {
            attrs.push(("signedness".into(), int_attr(r.read_varint()?)));
            attrs.push(("rounding_mode".into(), int_attr(r.read_varint()?)));
            fixed_ops!(1);
        }
        FToF => {
            attrs.push(("rounding_mode".into(), int_attr(r.read_varint()?)));
            fixed_ops!(1);
        }
        MmaI => {
            attrs.push(("signedness_lhs".into(), int_attr(r.read_varint()?)));
            attrs.push(("signedness_rhs".into(), int_attr(r.read_varint()?)));
            fixed_ops!(3);
        }

        // -- flags + rounding_mode + operands --
        AddF | DivF | MulF | SubF | Fma | Sqrt => {
            // Stream: flags, then rounding_mode. Emit attrs in the canonical
            // order (rounding_mode before the flag-derived flush_to_zero) so the
            // string prescan interns names in the same order as the writer.
            let ftz = r.read_varint()? & 1 != 0;
            attrs.push(("rounding_mode".into(), int_attr(r.read_varint()?)));
            if ftz {
                attrs.push(("flush_to_zero".into(), Attribute::Bool(true)));
            }
            fixed_ops!(opcode.fixed_operand_count().unwrap());
        }
        Exp2 | Rsqrt => {
            let flags = r.read_varint()?;
            if flags & 1 != 0 {
                attrs.push(("flush_to_zero".into(), Attribute::Bool(true)));
            }
            fixed_ops!(1);
        }
        // rounding_mode (default 5) elided when default so the prescan matches.
        // Added v13.2 (TanH) / v13.3 (Exp); older bodies omit it.
        Exp | TanH => {
            let rm_since = if opcode == Exp {
                BytecodeVersion::V13_3
            } else {
                BytecodeVersion::V13_2
            };
            if ctx.version >= rm_since {
                let rm = r.read_varint()?;
                if rm != 5 {
                    attrs.push(("rounding_mode".into(), int_attr(rm)));
                }
            }
            fixed_ops!(1);
        }
        MmaF => {
            // fast_acc flags added v13.3; older bodies omit the varint.
            if ctx.version >= BytecodeVersion::V13_3 {
                let flags = r.read_varint()?;
                if flags & 1 != 0 {
                    attrs.push(("fast_acc".into(), Attribute::Bool(true)));
                }
            }
            fixed_ops!(3);
        }
        MaxF | MinF => {
            let flags = r.read_varint()?;
            if flags & 1 != 0 {
                attrs.push(("propagate_nan".into(), Attribute::Bool(true)));
            }
            if flags & 2 != 0 {
                attrs.push(("flush_to_zero".into(), Attribute::Bool(true)));
            }
            fixed_ops!(2);
        }

        // -- Constant: value is a DenseElements stored as a constant-pool index;
        //    element_type is inferred from the result type (not on the wire). --
        Constant => {
            let const_idx = r.read_varint()? as usize;
            let raw = ctx
                .constants
                .get(const_idx)
                .ok_or_else(|| err("constant index out of range"))?;
            let mut cr = Reader::new(raw);
            let len = cr.read_varint()? as usize;
            let data = cr.read_bytes(len)?.to_vec();
            let element_type = match result_types.first() {
                Some(Type::Tile(t)) => match &t.element_type {
                    TileElementType::Scalar(s) => Type::Scalar(*s),
                    TileElementType::Pointer(p) => Type::Pointer((**p).clone()),
                },
                Some(t) => t.clone(),
                None => return Err(err("constant has no result type")),
            };
            attrs.push((
                "value".into(),
                Attribute::DenseElements(DenseElements {
                    element_type,
                    shape: Vec::new(),
                    data,
                }),
            ));
        }
        Assume => {
            let pred = read_self_contained_attribute(r, ctx.strings, ctx.types, ctx.constants)?;
            attrs.push(("predicate".into(), pred));
            fixed_ops!(1);
        }
        Assert => {
            attrs.push((
                "message".into(),
                Attribute::String(ctx.string(r.read_varint()? as usize)?),
            ));
            fixed_ops!(1);
        }
        GetGlobal => {
            attrs.push((
                "name".into(),
                Attribute::String(ctx.string(r.read_varint()? as usize)?),
            ));
        }
        GetIndexSpaceShape | GetTensorShape => fixed_ops!(1),

        // -- variadic operands --
        JoinTokens | Extract => {
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
        }
        Return | Yield | Break | Continue => {
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
        }

        // -- MakeTensorView: base + variadic shape + variadic strides --
        MakeTensorView => {
            let base = ctx.read_operands(r, 1)?;
            let (ns, shape) = ctx.read_var_operands(r)?;
            let (nt, strides) = ctx.read_var_operands(r)?;
            seg = vec![1, ns as i32, nt as i32];
            operands = base;
            operands.extend(shape);
            operands.extend(strides);
        }

        // -- view load/store: flags + ordering + optional scope/hints + grouped operands --
        LoadViewTko => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            if flags & 1 != 0 {
                attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            }
            if flags & 2 != 0 {
                attrs.push((
                    "optimization_hints".into(),
                    Attribute::OptimizationHints(read_inline_opt_hints(
                        r,
                        ctx.strings,
                        ctx.types,
                        ctx.constants,
                    )?),
                ));
            }
            let view = ctx.read_operands(r, 1)?;
            let (ni, idx) = ctx.read_var_operands(r)?;
            let tok = if flags & 4 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, ni as i32, tok.len() as i32];
            operands = view;
            operands.extend(idx);
            operands.extend(tok);
        }
        StoreViewTko => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            if flags & 1 != 0 {
                attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            }
            if flags & 2 != 0 {
                attrs.push((
                    "optimization_hints".into(),
                    Attribute::OptimizationHints(read_inline_opt_hints(
                        r,
                        ctx.strings,
                        ctx.types,
                        ctx.constants,
                    )?),
                ));
            }
            let tile = ctx.read_operands(r, 1)?;
            let view = ctx.read_operands(r, 1)?;
            let (ni, idx) = ctx.read_var_operands(r)?;
            let tok = if flags & 4 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, 1, ni as i32, tok.len() as i32];
            operands = tile;
            operands.extend(view);
            operands.extend(idx);
            operands.extend(tok);
        }

        // -- pointer load/store with flags + operand groups --
        LoadPtrTko => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            if flags & 1 != 0 {
                attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            }
            if flags & 2 != 0 {
                attrs.push((
                    "optimization_hints".into(),
                    Attribute::OptimizationHints(read_inline_opt_hints(
                        r,
                        ctx.strings,
                        ctx.types,
                        ctx.constants,
                    )?),
                ));
            }
            operands = ctx.read_operands(r, 1)?; // ptr
            let g1 = if flags & 4 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            let g2 = if flags & 8 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            let g3 = if flags & 16 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, g1.len() as i32, g2.len() as i32, g3.len() as i32];
            operands.extend(g1);
            operands.extend(g2);
            operands.extend(g3);
        }
        StorePtrTko => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            if flags & 1 != 0 {
                attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            }
            if flags & 2 != 0 {
                attrs.push((
                    "optimization_hints".into(),
                    Attribute::OptimizationHints(read_inline_opt_hints(
                        r,
                        ctx.strings,
                        ctx.types,
                        ctx.constants,
                    )?),
                ));
            }
            operands = ctx.read_operands(r, 1)?; // ptr
            operands.extend(ctx.read_operands(r, 1)?); // value
            let g2 = if flags & 4 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            let g3 = if flags & 8 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, 1, g2.len() as i32, g3.len() as i32];
            operands.extend(g2);
            operands.extend(g3);
        }
        AtomicRMW => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            attrs.push(("mode".into(), int_attr(r.read_varint()?)));
            operands = ctx.read_operands(r, 1)?; // ptr
            operands.extend(ctx.read_operands(r, 1)?); // value
            let g2 = if flags & 1 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            let g3 = if flags & 2 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, 1, g2.len() as i32, g3.len() as i32];
            operands.extend(g2);
            operands.extend(g3);
        }
        AtomicCAS => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            operands = ctx.read_operands(r, 1)?; // ptr
            operands.extend(ctx.read_operands(r, 1)?); // cmp
            operands.extend(ctx.read_operands(r, 1)?); // val
            let g3 = if flags & 1 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            let g4 = if flags & 2 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, 1, 1, g3.len() as i32, g4.len() as i32];
            operands.extend(g3);
            operands.extend(g4);
        }
        AtomicRedViewTko => {
            let flags = r.read_varint()?;
            attrs.push((
                "memory_ordering_semantics".into(),
                int_attr(r.read_varint()?),
            ));
            attrs.push(("memory_scope".into(), int_attr(r.read_varint()?)));
            attrs.push(("mode".into(), int_attr(r.read_varint()?)));
            operands = ctx.read_operands(r, 1)?; // view/tile
            let (ni, idx) = ctx.read_var_operands(r)?; // variadic indices
            operands.extend(idx);
            operands.extend(ctx.read_operands(r, 1)?); // value
            let g3 = if flags & 1 != 0 {
                ctx.read_operands(r, 1)?
            } else {
                vec![]
            };
            seg = vec![1, ni as i32, 1, g3.len() as i32];
            operands.extend(g3);
        }

        // -- region ops --
        For => {
            // unsigned_cmp flags added v13.2; older bodies omit the varint.
            let flags = if ctx.version >= BytecodeVersion::V13_2 {
                r.read_varint()?
            } else {
                0
            };
            if flags & 1 != 0 {
                attrs.push(("unsigned_cmp".into(), Attribute::Bool(true)));
            }
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
            regions = read_regions(r, ctx, module)?;
        }
        Loop => {
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
            regions = read_regions(r, ctx, module)?;
        }
        If => {
            operands = ctx.read_operands(r, 1)?; // condition
            regions = read_regions(r, ctx, module)?;
        }
        Reduce => {
            attrs.push(("dim".into(), int_attr(r.read_varint()?)));
            attrs.push((
                "identities".into(),
                Attribute::Array(read_inline_array(r, ctx.strings, ctx.types, ctx.constants)?),
            ));
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
            regions = read_regions(r, ctx, module)?;
        }
        Scan => {
            attrs.push(("dim".into(), int_attr(r.read_varint()?)));
            attrs.push(("reverse".into(), int_attr(r.read_varint()?)));
            attrs.push((
                "identities".into(),
                Attribute::Array(read_inline_array(r, ctx.strings, ctx.types, ctx.constants)?),
            ));
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
            regions = read_regions(r, ctx, module)?;
        }
        Print => {
            // flags added v13.2 (token operand not yet modeled); older bodies omit it.
            if ctx.version >= BytecodeVersion::V13_2 {
                let _flags = r.read_varint()?;
            }
            attrs.push((
                "str".into(),
                Attribute::String(ctx.string(r.read_varint()? as usize)?),
            ));
            let (_, ops) = ctx.read_var_operands(r)?;
            operands = ops;
        }

        other => {
            return Err(err(&format!(
                "op {other:?} not yet handled by the bytecode reader"
            )));
        }
    }

    if !seg.is_empty() {
        // The writer reads `operandSegmentSizes` as an Array of integers
        // (see op_writer::operand_segment_sizes), so emit it in that shape.
        let arr = seg.into_iter().map(|v| int_attr(v as u64)).collect();
        attrs.push(("operandSegmentSizes".into(), Attribute::Array(arr)));
    }

    let (op_id, results) = OpBuilder::new(opcode, Location::Unknown)
        .results(result_types)
        .operands(operands)
        .attrs(attrs)
        .regions(regions)
        .build(module);
    append_op(module, block, op_id);
    ctx.values.extend(results);
    Ok(())
}

/// Read an op's region list (mirror of `op_writer::write_regions`): a count
/// followed by that many regions.
fn read_regions(
    r: &mut Reader,
    ctx: &mut OpCtx,
    module: &mut Module,
) -> Result<Vec<crate::ir::RegionId>> {
    let n = r.read_varint()? as usize;
    (0..n).map(|_| read_region(r, ctx, module)).collect()
}

/// Read one region (mirror of `WriterCtx::write_region`): block count + blocks.
fn read_region(
    r: &mut Reader,
    ctx: &mut OpCtx,
    module: &mut Module,
) -> Result<crate::ir::RegionId> {
    let n_blocks = r.read_varint()? as usize;
    let mut blocks = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        blocks.push(read_block(r, ctx, module)?);
    }
    Ok(module.alloc_region(crate::ir::Region { blocks }))
}

/// Read one block (mirror of `WriterCtx::write_block`): arg count + arg types,
/// then op count + ops, with block-scoped value numbering rolled back at the end.
fn read_block(r: &mut Reader, ctx: &mut OpCtx, module: &mut Module) -> Result<BlockId> {
    let saved = ctx.values.len();
    let n_args = r.read_varint()? as usize;
    let arg_types = (0..n_args)
        .map(|_| ctx.ty(r.read_varint()? as usize))
        .collect::<Result<Vec<_>>>()?;
    let (block_id, args) = crate::builder::build_block(module, &arg_types);
    ctx.values.extend(args);
    let n_ops = r.read_varint()? as usize;
    for _ in 0..n_ops {
        let opcode = Opcode::from_u16(r.read_varint()? as u16)
            .ok_or_else(|| err("unknown opcode in block"))?;
        read_op(opcode, r, ctx, module, block_id)?;
    }
    ctx.values.truncate(saved); // roll back block-scoped values
    Ok(block_id)
}

/// Decode `count` consecutive type entries (the type section body after its
/// count + offset table). Returns the types in on-wire order.
pub(super) fn decode_types_seq(
    bytes: &[u8],
    count: usize,
    version: BytecodeVersion,
) -> Result<Vec<Type>> {
    let mut r = Reader::new(bytes);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let ty = decode_one_type(&mut r, &out, version)?;
        out.push(ty);
    }
    Ok(out)
}

// =========================================================================
// decode_module: bytecode -> ir::Module  (the inverse of write_bytecode)
// =========================================================================

/// Type section -> `Vec<Type>` (mirror of `parse_type_section`, but typed).
fn parse_types(payload: Option<&[u8]>, version: BytecodeVersion) -> Result<Vec<Type>> {
    let Some(data) = payload else {
        return Ok(Vec::new());
    };
    let mut r = Reader::new(data);
    let count = r.read_varint()? as usize;
    if count == 0 {
        return Ok(Vec::new());
    }
    r.skip_padding(4)?;
    for _ in 0..count {
        let _ = r.read_le_u32()?; // offset table (entries are contiguous)
    }
    let entries = r.read_bytes(r.remaining())?;
    decode_types_seq(entries, count, version)
}

/// Decode all functions, populating `module`.
fn decode_funcs(
    payload: &[u8],
    strings: &[String],
    types: &[Type],
    constants: &[Vec<u8>],
    version: BytecodeVersion,
    module: &mut Module,
) -> Result<()> {
    use crate::bytecode::enums::FunctionFlag;
    let mut r = Reader::new(payload);
    let count = r.read_varint()? as usize;
    for _ in 0..count {
        let name = strings
            .get(r.read_varint()? as usize)
            .cloned()
            .ok_or_else(|| err("func name index out of range"))?;
        let sig_idx = r.read_varint()? as usize;
        let func_type = types
            .get(sig_idx)
            .cloned()
            .ok_or_else(|| err("func sig index out of range"))?;
        let flags = r.read_byte()?;
        let _loc = r.read_varint()?;
        let is_entry = flags & FunctionFlag::KindKernel as u8 != 0;
        let opt_hints = if flags & FunctionFlag::HasOptimizationHints as u8 != 0 {
            Some(read_self_contained_attribute(
                &mut r, strings, types, constants,
            )?)
        } else {
            None
        };
        let body_len = r.read_varint()? as usize;
        let body = r.read_bytes(body_len)?;

        // Entry block args come from the function signature inputs.
        let arg_types = match &func_type {
            Type::Func(f) => f.inputs.clone(),
            _ => return Err(err("function signature is not a func type")),
        };
        let (region_id, block_id, arg_values) = build_single_block_region(module, &arg_types);

        let mut ctx = OpCtx {
            types,
            strings,
            constants,
            values: arg_values,
            version,
        };
        let mut br = Reader::new(body);
        while br.remaining() > 0 {
            let opcode = Opcode::from_u16(br.read_varint()? as u16)
                .ok_or_else(|| err("unknown opcode in function body"))?;
            read_op(opcode, &mut br, &mut ctx, module, block_id)?;
        }

        let _ = is_entry; // all functions use the Entry opcode in this dialect
        let mut b = OpBuilder::new(Opcode::Entry, Location::Unknown)
            .attr("sym_name", Attribute::String(name))
            .attr("function_type", Attribute::Type(func_type))
            .region(region_id);
        if let Some(oh) = opt_hints {
            b = b.attr("optimization_hints", oh);
        }
        let (entry, _) = b.build(module);
        module.functions.push(entry);
    }
    Ok(())
}

/// Decode a full bytecode buffer into an [`ir::Module`](crate::ir::Module).
///
/// The inverse of [`write_bytecode`](crate::write_bytecode) for the op set the
/// reader supports; unsupported ops return an error naming the op.
pub fn decode_module(data: &[u8]) -> Result<Module> {
    let mut r = Reader::new(data);
    let version = r.read_header()?;

    let mut sections = RawSections::default();
    loop {
        let sh = r.read_section_header()?;
        if sh.id == Section::EndOfBytecode as u8 {
            break;
        }
        sections.insert(sh.id, r.read_bytes(sh.data_len)?);
    }

    let strings = parse_string_section(sections.get(Section::String as u8))?;
    let types = parse_types(sections.get(Section::Type as u8), version)?;
    let constants = parse_constant_section(sections.get(Section::Constant as u8))?;

    let mut module = Module::new("module");
    if let Some(payload) = sections.get(Section::Func as u8) {
        decode_funcs(payload, &strings, &types, &constants, version, &mut module)?;
    }
    Ok(module)
}

/// Read the bytecode version declared in a buffer's header.
///
/// [`decode_module`] is version-agnostic and re-encodes at the current format;
/// pair this with `write_bytecode_version` to reproduce an older buffer exactly:
///
/// ```no_run
/// # use cutile_ir::{bytecode_version, decode_module};
/// # use cutile_ir::bytecode::write_bytecode_version;
/// # let data: &[u8] = &[];
/// let module = decode_module(data)?;
/// let exact = write_bytecode_version(&module, bytecode_version(data)?)?;
/// assert_eq!(exact, data);
/// # Ok::<(), cutile_ir::Error>(())
/// ```
pub fn bytecode_version(data: &[u8]) -> Result<BytecodeVersion> {
    Reader::new(data).read_header()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::encoding::EncodingWriter;
    use crate::bytecode::enums::MAGIC;

    /// `read_ap_float` is the exact inverse of `write_ap_float` (at each type's
    /// precision) for the common float scalars.
    #[test]
    fn ap_float_roundtrips() {
        use crate::ir::{ScalarType, Type};
        let cases = [
            (ScalarType::F32, 3.14159_f64),
            (ScalarType::F64, 2.718281828459045_f64),
            (ScalarType::F16, 1.5_f64),
            (ScalarType::BF16, -3.5_f64),
            (ScalarType::TF32, 0.5_f64),
        ];
        for (scalar, value) in cases {
            let mut w = EncodingWriter::new();
            w.write_ap_float(value, &Type::Scalar(scalar));
            let bytes = w.into_bytes();
            let got = Reader::new(&bytes).read_ap_float(scalar).unwrap();
            let expected = match scalar {
                ScalarType::F16 => half::f16::from_f64(value).to_f64(),
                ScalarType::BF16 => half::bf16::from_f64(value).to_f64(),
                ScalarType::F32 | ScalarType::TF32 => (value as f32) as f64,
                _ => value,
            };
            assert_eq!(got, expected, "{scalar:?} ap_float roundtrip");
        }
    }

    /// Decoding the type section reproduces the exact `Vec<Type>` the writer
    /// serialized (covers scalars, pointer, tile, tensor_view, partition_view, func).
    #[test]
    fn type_section_roundtrips() {
        use crate::bytecode::writer::{serialize_type, TypeManager};
        use crate::ir::*;

        let tv = TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![DYNAMIC],
        };
        let want = [
            Type::Token,
            Type::Scalar(ScalarType::I32),
            Type::Pointer(PointerType {
                pointee: ScalarType::F32,
            }),
            Type::Tile(TileType {
                shape: vec![4, 4],
                element_type: TileElementType::Scalar(ScalarType::F32),
            }),
            Type::TensorView(tv.clone()),
            Type::PartitionView(PartitionViewType {
                tile_shape: vec![4],
                tensor_view: tv.clone(),
                dim_map: vec![0],
                padding_value: None,
            }),
            Type::Func(FuncType {
                inputs: vec![Type::Pointer(PointerType {
                    pointee: ScalarType::F32,
                })],
                results: vec![],
            }),
        ];

        let mut tm = TypeManager::new();
        for t in &want {
            tm.get_or_insert(t);
        }
        let ordered: Vec<Type> = tm.entries().to_vec();

        let mut w = EncodingWriter::new();
        for t in &ordered {
            serialize_type(t, &mut tm, &mut w, BytecodeVersion::V13_3).unwrap();
        }
        let bytes = w.into_bytes();

        let decoded = decode_types_seq(&bytes, ordered.len(), BytecodeVersion::V13_3).unwrap();
        assert_eq!(decoded, ordered);
    }

    /// Every self-contained attribute kind decodes back to the value the writer
    /// serialized.
    #[test]
    fn self_contained_attr_roundtrips() {
        use crate::bytecode::writer::{
            write_self_contained_attribute, ConstantManager, StringManager, TypeManager,
        };
        use crate::ir::*;

        let attrs = vec![
            Attribute::int(42, ScalarType::I32),
            Attribute::float(0.5, ScalarType::F32), // f32-exact so it round-trips
            Attribute::Bool(true),
            Attribute::Type(Type::Scalar(ScalarType::F64)),
            Attribute::String("hello".into()),
            Attribute::Array(vec![Attribute::i32(1), Attribute::i32(2)]),
            Attribute::DivBy(DivBy {
                divisor: 8,
                every: Some(2),
                along: None,
            }),
            Attribute::SameElements(SameElements {
                values: vec![1, 2, 3],
            }),
            Attribute::Dictionary(vec![("k".into(), Attribute::i32(7))]),
            Attribute::Bounded(Bounded {
                lb: Some(0),
                ub: Some(100),
            }),
            Attribute::OptimizationHints(OptimizationHints {
                entries: vec![("sm_90".into(), vec![("num_cta".into(), Attribute::i32(2))])],
            }),
        ];

        let (mut strings, mut types, mut constants) = (
            StringManager::new(),
            TypeManager::new(),
            ConstantManager::new(),
        );
        let mut w = EncodingWriter::new();
        for a in &attrs {
            write_self_contained_attribute(a, &mut w, &mut strings, &mut types, &mut constants)
                .unwrap();
        }
        let bytes = w.into_bytes();
        let (s, t, c) = (
            strings.entries(),
            types.entries().to_vec(),
            constants.entries().to_vec(),
        );

        let mut r = Reader::new(&bytes);
        for a in &attrs {
            assert_eq!(
                &read_self_contained_attribute(&mut r, &s, &t, &c).unwrap(),
                a
            );
        }
    }

    /// A straight-line kernel decodes from bytecode and re-encodes to the exact
    /// same bytes (full container + op-stream round-trip).
    #[test]
    fn decode_module_roundtrips_straightline() {
        use crate::builder::{append_op, build_single_block_region, OpBuilder};
        use crate::ir::*;

        let tf = Type::Tile(TileType {
            shape: vec![4],
            element_type: TileElementType::Scalar(ScalarType::F32),
        });
        let mut m = Module::new("test");
        let (region, block, args) = build_single_block_region(&mut m, &[tf.clone(), tf.clone()]);
        // %c = addf %a, %b ; %d = mulf %c, %b ; return
        let (addf, c) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .result(tf.clone())
            .operands([args[0], args[1]])
            .attr("rounding_mode", Attribute::i32(5))
            .build(&mut m);
        append_op(&mut m, block, addf);
        let (mulf, _) = OpBuilder::new(Opcode::MulF, Location::Unknown)
            .result(tf.clone())
            .operands([c[0], args[1]])
            .attr("rounding_mode", Attribute::i32(5))
            .build(&mut m);
        append_op(&mut m, block, mulf);
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut m);
        append_op(&mut m, block, ret);
        let func_type = Type::Func(FuncType {
            inputs: vec![tf.clone(), tf],
            results: vec![],
        });
        let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
            .attr("sym_name", Attribute::String("k".into()))
            .attr("function_type", Attribute::Type(func_type))
            .region(region)
            .build(&mut m);
        m.functions.push(entry);

        let bytes = crate::write_bytecode(&m).unwrap();
        let decoded = decode_module(&bytes).unwrap();
        let reencoded = crate::write_bytecode(&decoded).unwrap();
        if bytes != reencoded {
            let diff = bytes.iter().zip(&reencoded).position(|(a, b)| a != b);
            panic!(
                "len {} vs {}, first diff at {:?}\n orig: {:02x?}\n  new: {:02x?}",
                bytes.len(),
                reencoded.len(),
                diff,
                &bytes
                    [diff.unwrap_or(0).saturating_sub(2)..(diff.unwrap_or(0) + 6).min(bytes.len())],
                &reencoded[diff.unwrap_or(0).saturating_sub(2)
                    ..(diff.unwrap_or(0) + 6).min(reencoded.len())],
            );
        }
    }

    /// Build a minimal valid bytecode (header + end marker, no sections).
    fn minimal_bytecode() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(13); // major
        buf.push(1); // minor
        buf.extend_from_slice(&0u16.to_le_bytes()); // tag
        buf.push(Section::EndOfBytecode as u8);
        buf
    }

    #[test]
    fn decode_minimal() {
        let data = minimal_bytecode();
        let out = decode_bytecode(&data).unwrap();
        assert!(out.contains("TileIR bytecode v13.1"));
    }

    #[test]
    fn decode_bad_magic() {
        let mut data = minimal_bytecode();
        data[1] = b'X'; // corrupt magic
        assert!(decode_bytecode(&data).is_err());
    }

    #[test]
    fn roundtrip_string_section() {
        // Build a bytecode with just a string section.
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.push(13);
        buf.push(1);
        buf.extend_from_slice(&0u16.to_le_bytes());

        // String section: 2 strings "hello" and "world"
        let mut section = EncodingWriter::new();
        section.write_varint(2); // count
        section.align_to(4);
        let offsets_pos = section.tell();
        section.write_le_u32(0);
        section.write_le_u32(0);
        let s1 = b"hello";
        let s2 = b"world";
        // Patch offsets
        let buf_ref = section.buf_mut();
        let o1: u32 = 0;
        let o2: u32 = s1.len() as u32;
        buf_ref[offsets_pos..offsets_pos + 4].copy_from_slice(&o1.to_le_bytes());
        buf_ref[offsets_pos + 4..offsets_pos + 8].copy_from_slice(&o2.to_le_bytes());
        section.write_bytes(s1);
        section.write_bytes(s2);

        let section_bytes = section.into_bytes();
        // Write section header: String section, no alignment needed externally.
        let mut header = EncodingWriter::new();
        header.write_byte((Section::String as u8) | 0x80); // has alignment
        header.write_varint(section_bytes.len() as u64);
        header.write_varint(4); // alignment
        header.align_to(4);
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&section_bytes);

        buf.push(Section::EndOfBytecode as u8);

        let out = decode_bytecode(&buf).unwrap();
        assert!(out.contains("\"hello\""));
        assert!(out.contains("\"world\""));
    }
}
