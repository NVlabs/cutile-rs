/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Top-level bytecode writer — orchestrates header, section, and
//! operation serialization into a byte buffer or file.
//!
//! Ported from the top-level `writeBytecode` function and its helper
//! managers in `BytecodeWriter.cpp`.

use std::collections::HashMap;
use std::io::Write;

use super::encoding::{patch_u32, patch_u64, EncodingWriter};
use super::enums::{AttributeTag, BytecodeVersion, FunctionFlag, Section, TypeTag, MAGIC};
use super::opcode::Opcode;
use crate::ir::{
    Attribute, BlockId, Module, OpId, RegionId, TileElementType, Type, Value, ValueProducer,
};
use crate::{Error, Result};

// =========================================================================
// Public entry point
// =========================================================================

/// Serialize a [`Module`] into Tile IR bytecode.
///
/// Returns the bytecode as a `Vec<u8>`.
pub fn write_bytecode(module: &Module) -> Result<Vec<u8>> {
    write_bytecode_version(module, BytecodeVersion::CURRENT)
}

/// Serialize a [`Module`] with a specific bytecode version.
pub fn write_bytecode_version(module: &Module, version: BytecodeVersion) -> Result<Vec<u8>> {
    if version < BytecodeVersion::MIN_SUPPORTED {
        return Err(Error::BytecodeWrite(format!(
            "unsupported version {version}, minimum is {}",
            BytecodeVersion::MIN_SUPPORTED
        )));
    }

    let mut out = Vec::new();

    // Initialize writer context with all managers.
    let mut ctx = WriterCtx {
        module,
        value_map: HashMap::new(),
        next_idx: 0,
        strings: StringManager::new(),
        types: TypeManager::new(),
        constants: ConstantManager::new(),
        debug: DebugInfoCollector::new(),
    };

    // Pre-scan: register all types, strings, constants used by globals and functions.
    for global in &module.globals {
        ctx.strings.get_or_insert(&global.sym_name);
        ctx.types.get_or_insert(&global.value.element_type);
    }
    for &func_op in &module.functions {
        prescan_function(
            module,
            func_op,
            &mut ctx.strings,
            &mut ctx.types,
            &mut ctx.constants,
        )?;
    }

    // 1. Header
    write_header(&mut out, version);

    // 2. Global section
    write_global_section(&mut out, &mut ctx)?;

    // 3. Function table section
    write_function_section(&mut out, &mut ctx)?;

    // 4. Constant section
    write_constant_section(&mut out, &ctx.constants)?;

    // 5. Debug section (omitted when no non-unknown locations).
    write_debug_section(&mut out, &ctx)?;

    // 6. Type section
    write_type_section(&mut out, &ctx.types)?;

    // 7. String section
    write_string_section(&mut out, &ctx.strings)?;

    // 8. End marker
    out.push(Section::EndOfBytecode as u8);

    Ok(out)
}

/// Convenience: write bytecode directly to a file.
pub fn write_bytecode_to_file(module: &Module, path: &str) -> Result<()> {
    let bytes = write_bytecode(module)?;
    let mut f = std::fs::File::create(path)
        .map_err(|e| Error::BytecodeWrite(format!("failed to create {path}: {e}")))?;
    f.write_all(&bytes)
        .map_err(|e| Error::BytecodeWrite(format!("failed to write {path}: {e}")))?;
    Ok(())
}

// =========================================================================
// Header
// =========================================================================

fn write_header(out: &mut Vec<u8>, version: BytecodeVersion) {
    out.extend_from_slice(&MAGIC);
    out.push(version.major);
    out.push(version.minor);
    out.extend_from_slice(&version.tag.to_le_bytes());
}

// =========================================================================
// Section header
// =========================================================================

fn write_section_header(out: &mut Vec<u8>, section_id: Section, length: usize, alignment: u64) {
    // Write directly onto `out` so that alignment padding is relative to the
    // overall stream position (matching the C++ writer and the reader's
    // expectation).
    let mut id_byte = section_id as u8 & 0x7F;
    if alignment > 1 {
        id_byte |= 0x80;
    }
    out.push(id_byte);
    // Write length as varint inline.
    {
        let mut v = length as u64;
        loop {
            let mut byte = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
    }
    if alignment > 1 {
        // Write alignment varint.
        {
            let mut v = alignment;
            loop {
                let mut byte = (v & 0x7F) as u8;
                v >>= 7;
                if v != 0 {
                    byte |= 0x80;
                }
                out.push(byte);
                if v == 0 {
                    break;
                }
            }
        }
        // Pad to alignment relative to overall stream position.
        let pos = out.len() as u64;
        let padding = (alignment - (pos % alignment)) % alignment;
        for _ in 0..padding {
            out.push(super::enums::ALIGNMENT_BYTE);
        }
    }
}

// =========================================================================
// String manager
// =========================================================================

pub(super) struct StringManager {
    map: indexmap::IndexMap<String, u64>,
}

impl StringManager {
    pub(super) fn new() -> Self {
        Self {
            map: indexmap::IndexMap::new(),
        }
    }

    pub(super) fn get_or_insert(&mut self, s: &str) -> u64 {
        if let Some(&idx) = self.map.get(s) {
            return idx;
        }
        let idx = self.map.len() as u64;
        self.map.insert(s.to_owned(), idx);
        idx
    }
}

fn write_string_section(out: &mut Vec<u8>, strings: &StringManager) -> Result<()> {
    if strings.map.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(strings.map.len() as u64);
    w.align_to(4);

    // Reserve offset table.
    let offsets_pos = w.tell();
    for _ in 0..strings.map.len() {
        w.write_le_u32(0);
    }

    // Write strings and track offsets.
    let mut running: u32 = 0;
    let mut offsets = Vec::with_capacity(strings.map.len());
    for (s, _) in &strings.map {
        offsets.push(running);
        w.write_bytes(s.as_bytes());
        running += s.len() as u32;
    }

    // Patch offsets.
    for (i, offset) in offsets.iter().enumerate() {
        patch_u32(w.buf_mut(), offsets_pos + i * 4, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::String, buf.len(), 4);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Type manager
// =========================================================================

pub(super) struct TypeManager {
    list: Vec<Type>,
    map: HashMap<Type, u64>,
}

impl TypeManager {
    pub(super) fn new() -> Self {
        Self {
            list: Vec::new(),
            map: HashMap::new(),
        }
    }

    pub(super) fn get_or_insert(&mut self, ty: &Type) -> u64 {
        if let Some(&idx) = self.map.get(ty) {
            return idx;
        }
        // Register dependent types first.
        self.register_deps(ty);
        let idx = self.list.len() as u64;
        self.map.insert(ty.clone(), idx);
        self.list.push(ty.clone());
        idx
    }

    fn register_deps(&mut self, ty: &Type) {
        match ty {
            Type::Pointer(p) => {
                self.get_or_insert(&Type::Scalar(p.pointee));
            }
            Type::Tile(t) => match &t.element_type {
                TileElementType::Scalar(s) => {
                    self.get_or_insert(&Type::Scalar(*s));
                }
                TileElementType::Pointer(p) => {
                    self.get_or_insert(&Type::Pointer((**p).clone()));
                }
            },
            Type::TensorView(tv) => {
                self.get_or_insert(&Type::Scalar(tv.element_type));
            }
            Type::PartitionView(pv) => {
                self.get_or_insert(&Type::TensorView(pv.tensor_view.clone()));
            }
            Type::Func(f) => {
                for inp in &f.inputs {
                    self.get_or_insert(inp);
                }
                for res in &f.results {
                    self.get_or_insert(res);
                }
            }
            Type::Scalar(_) | Type::Token => {}
        }
    }
}

fn serialize_type(ty: &Type, types: &mut TypeManager, w: &mut EncodingWriter) -> Result<()> {
    match ty {
        Type::Scalar(s) => {
            w.write_varint(s.type_tag() as u64);
        }
        Type::Pointer(p) => {
            w.write_varint(TypeTag::Pointer as u64);
            let idx = types.get_or_insert(&Type::Scalar(p.pointee));
            w.write_varint(idx);
        }
        Type::Tile(t) => {
            w.write_varint(TypeTag::Tile as u64);
            let elem_ty = match &t.element_type {
                TileElementType::Scalar(s) => Type::Scalar(*s),
                TileElementType::Pointer(p) => Type::Pointer((**p).clone()),
            };
            let idx = types.get_or_insert(&elem_ty);
            w.write_varint(idx);
            w.write_le_var_size_i64(&t.shape);
        }
        Type::TensorView(tv) => {
            w.write_varint(TypeTag::TensorView as u64);
            let idx = types.get_or_insert(&Type::Scalar(tv.element_type));
            w.write_varint(idx);
            w.write_le_var_size_i64(&tv.shape);
            w.write_le_var_size_i64(&tv.strides);
        }
        Type::PartitionView(pv) => {
            w.write_varint(TypeTag::PartitionView as u64);
            w.write_le_var_size_i32(&pv.tile_shape);
            let idx = types.get_or_insert(&Type::TensorView(pv.tensor_view.clone()));
            w.write_varint(idx);
            w.write_le_var_size_i32(&pv.dim_map);
            let has_padding = pv.padding_value.is_some();
            w.write_byte(has_padding as u8);
            if let Some(pv_val) = pv.padding_value {
                w.write_varint(pv_val as u64);
            }
        }
        Type::Func(f) => {
            w.write_varint(TypeTag::Func as u64);
            w.write_varint(f.inputs.len() as u64);
            for inp in &f.inputs {
                let idx = types.get_or_insert(inp);
                w.write_varint(idx);
            }
            w.write_varint(f.results.len() as u64);
            for res in &f.results {
                let idx = types.get_or_insert(res);
                w.write_varint(idx);
            }
        }
        Type::Token => {
            w.write_varint(TypeTag::Token as u64);
        }
    }
    Ok(())
}

fn write_type_section(out: &mut Vec<u8>, types: &TypeManager) -> Result<()> {
    if types.list.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(types.list.len() as u64);
    w.align_to(4);

    // Reserve offset table.
    let offsets_pos = w.tell();
    for _ in 0..types.list.len() {
        w.write_le_u32(0);
    }

    // Serialize each type.
    let mut running: u32 = 0;
    let mut offsets = Vec::with_capacity(types.list.len());
    // Clone the type list to avoid borrow issues — the list is small.
    let type_list: Vec<Type> = types.list.clone();
    let mut types_mut = TypeManager {
        list: types.list.clone(),
        map: types.map.clone(),
    };
    for ty in &type_list {
        offsets.push(running);
        let before = w.tell();
        serialize_type(ty, &mut types_mut, &mut w)?;
        running += (w.tell() - before) as u32;
    }

    // Patch offsets.
    for (i, offset) in offsets.iter().enumerate() {
        patch_u32(w.buf_mut(), offsets_pos + i * 4, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Type, buf.len(), 4);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Constant manager
// =========================================================================

pub(super) struct ConstantManager {
    entries: Vec<Vec<u8>>,
    // We don't deduplicate by value for now — ops reference constants by index.
}

impl ConstantManager {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub(super) fn add(&mut self, data: Vec<u8>) -> u64 {
        let idx = self.entries.len() as u64;
        self.entries.push(data);
        idx
    }
}

fn write_constant_section(out: &mut Vec<u8>, constants: &ConstantManager) -> Result<()> {
    if constants.entries.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(constants.entries.len() as u64);
    w.align_to(8);

    // Reserve offset table (u64 per constant).
    let offsets_pos = w.tell();
    for _ in 0..constants.entries.len() {
        w.write_le_u64(0);
    }

    let mut running: u64 = 0;
    let mut offsets = Vec::with_capacity(constants.entries.len());
    for entry in &constants.entries {
        offsets.push(running);
        w.write_bytes(entry);
        running += entry.len() as u64;
    }

    for (i, offset) in offsets.iter().enumerate() {
        patch_u64(w.buf_mut(), offsets_pos + i * 8, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Constant, buf.len(), 8);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Global section
// =========================================================================

fn write_global_section(out: &mut Vec<u8>, ctx: &mut WriterCtx) -> Result<()> {
    if ctx.module.globals.is_empty() {
        return Ok(());
    }

    let mut w = EncodingWriter::new();
    w.write_varint(ctx.module.globals.len() as u64);

    for global in &ctx.module.globals {
        // 1. Symbol name index.
        let name_idx = ctx.strings.get_or_insert(&global.sym_name);
        w.write_varint(name_idx);

        // 2. Type index of the global's value.
        let ty_idx = ctx.types.get_or_insert(&global.value.element_type);
        w.write_varint(ty_idx);

        // 3. Constant pool index for the raw data.
        let mut cdata = EncodingWriter::new();
        cdata.write_varint(global.value.data.len() as u64);
        cdata.write_bytes(&global.value.data);
        let const_idx = ctx.constants.add(cdata.into_bytes());
        w.write_varint(const_idx);

        // 4. Alignment.
        w.write_varint(global.alignment);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Global, buf.len(), 1);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Debug info section
// =========================================================================

use super::enums::DebugTag;

/// Serialize a single debug info entry.
fn write_debug_entry(loc: &crate::ir::Location, w: &mut EncodingWriter, strings: &StringManager) {
    use crate::ir::Location;
    match loc {
        Location::Unknown => {
            w.write_varint(DebugTag::Unknown as u64);
        }
        Location::FileLineCol { .. } => {
            // FileLineCol without scope → serialize as Unknown (reserved index)
            w.write_varint(DebugTag::Unknown as u64);
        }
        Location::DebugInfo(di) => {
            // DILoc: tag, scope_index, filename_index, line, column
            w.write_varint(DebugTag::DILoc as u64);
            // Scope index — would need interning; use 0 (unknown) placeholder.
            w.write_varint(0);
            // Filename string index — look up or 0.
            if let Some(&idx) = strings.map.get(&di.filename) {
                w.write_varint(idx);
            } else {
                w.write_varint(0);
            }
            w.write_varint(di.line as u64);
            w.write_varint(di.column as u64);
        }
        Location::CallSite { .. } => {
            w.write_varint(DebugTag::CallSite as u64);
            // Callee and caller indices — would need interning; use 0 placeholder.
            w.write_varint(0);
            w.write_varint(0);
        }
    }
}

fn write_debug_section(out: &mut Vec<u8>, ctx: &WriterCtx) -> Result<()> {
    if !ctx.debug.has_debug_info() {
        return Ok(());
    }

    // For now, write an empty debug section structure.
    // Full debug info interning (scope chains, per-op indices) will be
    // implemented when the compiler starts propagating source locations.
    //
    // The section format is:
    //   diOpsNum[varint]
    //   padding[align 4]
    //   diIndexOffsets[u32*]
    //   diIndicesNum[varint]
    //   padding[align 8]
    //   diIndices[u64*]
    //   diAttrNum[varint]
    //   padding[align 4]
    //   diOffsets[u32*]
    //   diData[bytes]

    let mut w = EncodingWriter::new();

    // Number of operations with debug info.
    let num_ops = ctx.debug.op_locations.len();
    w.write_varint(num_ops as u64);
    w.align_to(4);

    // Per-op offset into debug info indices (each op has one index).
    for i in 0..num_ops {
        w.write_le_u32(i as u32);
    }

    // Total number of debug info indices = num_ops (one per op).
    w.write_varint(num_ops as u64);
    w.align_to(8);

    // Each op maps to a single debug info entry (indexed sequentially).
    // Offset by DebugReserved::Size (1) to skip the reserved UnknownLoc slot.
    for i in 0..num_ops {
        let (_, ref loc) = ctx.debug.op_locations[i];
        if matches!(loc, crate::ir::Location::Unknown) {
            w.write_le_u64(0); // Reserved UnknownLoc index.
        } else {
            w.write_le_u64((i as u64) + 1); // +1 for reserved slot.
        }
    }

    // Collect non-unknown locations as debug info entries.
    let entries: Vec<_> = ctx
        .debug
        .op_locations
        .iter()
        .filter(|(_, loc)| !matches!(loc, crate::ir::Location::Unknown))
        .map(|(_, loc)| loc)
        .collect();

    w.write_varint(entries.len() as u64);
    w.align_to(4);

    // Reserve offset table.
    let offsets_pos = w.tell();
    for _ in 0..entries.len() {
        w.write_le_u32(0);
    }

    // Serialize each debug info entry.
    let mut running: u32 = 0;
    let mut offsets = Vec::with_capacity(entries.len());
    for loc in &entries {
        offsets.push(running);
        let before = w.tell();
        write_debug_entry(loc, &mut w, &ctx.strings);
        running += (w.tell() - before) as u32;
    }

    for (i, offset) in offsets.iter().enumerate() {
        patch_u32(w.buf_mut(), offsets_pos + i * 4, *offset);
    }

    let buf = w.into_bytes();
    write_section_header(out, Section::Debug, buf.len(), 8);
    out.extend_from_slice(&buf);
    Ok(())
}

// =========================================================================
// Pre-scan: collect all types, strings, constants from functions
// =========================================================================

fn prescan_function(
    module: &Module,
    func_op: OpId,
    strings: &mut StringManager,
    types: &mut TypeManager,
    _constants: &mut ConstantManager,
) -> Result<()> {
    let op = module.op(func_op);

    // Register function name and type from attributes.
    for (name, attr) in &op.attributes {
        prescan_attribute(name, attr, strings, types);
    }

    // Recurse into regions.
    for &region_id in &op.regions {
        prescan_region(module, region_id, strings, types)?;
    }

    Ok(())
}

fn prescan_region(
    module: &Module,
    region_id: RegionId,
    strings: &mut StringManager,
    types: &mut TypeManager,
) -> Result<()> {
    let region = module.region(region_id);
    for &block_id in &region.blocks {
        let block = module.block(block_id);
        // Block argument types.
        for (_, ty) in &block.args {
            types.get_or_insert(ty);
        }
        for &op_id in &block.ops {
            let op = module.op(op_id);
            // Result types.
            for ty in &op.result_types {
                types.get_or_insert(ty);
            }
            // Attributes.
            for (name, attr) in &op.attributes {
                prescan_attribute(name, attr, strings, types);
            }
            // Nested regions.
            for &rid in &op.regions {
                prescan_region(module, rid, strings, types)?;
            }
        }
    }
    Ok(())
}

fn prescan_attribute(
    name: &str,
    attr: &Attribute,
    strings: &mut StringManager,
    types: &mut TypeManager,
) {
    strings.get_or_insert(name);
    match attr {
        Attribute::String(s) => {
            strings.get_or_insert(s);
        }
        Attribute::Type(ty) => {
            types.get_or_insert(ty);
        }
        Attribute::DenseElements(de) => {
            types.get_or_insert(&de.element_type);
        }
        Attribute::Dictionary(entries) => {
            for (k, v) in entries {
                prescan_attribute(k, v, strings, types);
            }
        }
        Attribute::OptimizationHints(oh) => {
            for (arch, hints) in &oh.entries {
                strings.get_or_insert(arch);
                for (k, v) in hints {
                    prescan_attribute(k, v, strings, types);
                }
            }
        }
        Attribute::Array(elems) => {
            for elem in elems {
                prescan_attribute("", elem, strings, types);
            }
        }
        _ => {}
    }
}

// =========================================================================
// Function section
// =========================================================================

fn write_function_section(out: &mut Vec<u8>, ctx: &mut WriterCtx) -> Result<()> {
    if ctx.module.functions.is_empty() {
        return Ok(());
    }

    let func_ids: Vec<OpId> = ctx.module.functions.clone();
    let mut w = EncodingWriter::new();
    w.write_varint(func_ids.len() as u64);

    for func_op_id in &func_ids {
        let op = ctx.module.op(*func_op_id);

        // Extract name and function type from attributes.
        let name = find_string_attr(&op.attributes, "sym_name")
            .ok_or_else(|| Error::BytecodeWrite("function missing sym_name attribute".into()))?;
        let func_type = find_type_attr(&op.attributes, "function_type").ok_or_else(|| {
            Error::BytecodeWrite("function missing function_type attribute".into())
        })?;

        let name_idx = ctx.strings.get_or_insert(&name);
        let sig_idx = ctx.types.get_or_insert(&func_type);

        w.write_varint(name_idx);
        w.write_varint(sig_idx);

        // Entry flag.
        let is_entry = op.opcode == Opcode::Entry;
        let has_hints = find_attr(&op.attributes, "optimization_hints").is_some();
        let mut flags: u8 = 0;
        if is_entry {
            flags |= FunctionFlag::KindKernel as u8;
        }
        if is_entry && has_hints {
            flags |= FunctionFlag::HasOptimizationHints as u8;
        }
        w.write_byte(flags);

        // Function location index (placeholder 0 for now — debug section TODO).
        w.write_varint(0);

        // Write optimization hints if present.
        if is_entry && has_hints {
            if let Some(hints_attr) = find_attr(&op.attributes, "optimization_hints").cloned() {
                write_self_contained_attribute(
                    &hints_attr,
                    &mut w,
                    &mut ctx.strings,
                    &mut ctx.types,
                    &mut ctx.constants,
                )?;
            }
        }

        // Write function body.
        // Reset per-function state.
        ctx.value_map.clear();
        ctx.next_idx = 0;
        let body = write_function_body(ctx, *func_op_id)?;
        w.write_varint(body.len() as u64);
        w.write_bytes(&body);
    }

    w.align_to(8);
    let buf = w.into_bytes();
    write_section_header(out, Section::Func, buf.len(), 8);
    out.extend_from_slice(&buf);
    Ok(())
}

fn write_function_body(ctx: &mut WriterCtx, func_op: OpId) -> Result<Vec<u8>> {
    let mut w = EncodingWriter::new();

    let op = ctx.module.op(func_op);
    // Register function argument values (from the entry block of the first region).
    if let Some(&region_id) = op.regions.first() {
        let region = ctx.module.region(region_id);
        if let Some(&entry_block) = region.blocks.first() {
            let block = ctx.module.block(entry_block);
            let args: Vec<_> = block.args.clone();
            let ops: Vec<_> = block.ops.clone();
            for (val, _) in &args {
                ctx.value_map.insert(*val, ctx.next_idx);
                ctx.next_idx += 1;
            }
            for op_id in ops {
                ctx.write_operation(op_id, &mut w)?;
            }
        }
    }

    Ok(w.into_bytes())
}

// =========================================================================
// Writer context — bundles mutable state to avoid borrow conflicts
// =========================================================================

/// Bundles all mutable serialization state so it can be passed as a single
/// `&mut` to per-op writers and recursive region/block writers.
pub(super) struct WriterCtx<'a> {
    pub module: &'a Module,
    pub value_map: HashMap<Value, u64>,
    pub next_idx: u64,
    pub strings: StringManager,
    pub types: TypeManager,
    pub constants: ConstantManager,
    pub debug: DebugInfoCollector,
}

/// Collects debug info (source locations) during operation serialization.
///
/// Operations are tracked by a sequential index. Each function gets a
/// function-level location index, and each operation within that function
/// contributes its location to the debug info attribute pool.
#[allow(dead_code)]
pub(super) struct DebugInfoCollector {
    /// Per-function location index (maps function OpId to debug index).
    func_loc_indices: HashMap<OpId, u64>,
    /// Per-operation debug info: (function_loc_index, op_location).
    /// Collected during function body serialization.
    op_locations: Vec<(u64, crate::ir::Location)>,
}

#[allow(dead_code)]
impl DebugInfoCollector {
    fn new() -> Self {
        Self {
            func_loc_indices: HashMap::new(),
            op_locations: Vec::new(),
        }
    }

    /// Register a function's location and return its debug index.
    pub fn register_function(&mut self, func_id: OpId) -> u64 {
        let idx = self.func_loc_indices.len() as u64 + 1; // 0 is reserved for UnknownLoc
        self.func_loc_indices.insert(func_id, idx);
        idx
    }

    /// Record an operation's location during serialization.
    pub fn record_op_location(&mut self, func_loc_idx: u64, loc: crate::ir::Location) {
        self.op_locations.push((func_loc_idx, loc));
    }

    /// Returns true if any non-unknown locations were collected.
    pub fn has_debug_info(&self) -> bool {
        self.op_locations
            .iter()
            .any(|(_, loc)| !matches!(loc, crate::ir::Location::Unknown))
    }
}

impl<'a> WriterCtx<'a> {
    pub fn write_operation(&mut self, op_id: OpId, w: &mut EncodingWriter) -> Result<()> {
        let op = self.module.op(op_id);

        // Write opcode.
        w.write_varint(op.opcode.as_u16() as u64);

        // Per-op body serialization.
        super::op_writer::write_op_body(op, w, self)?;

        // Register result values.
        let result_types_len = self.module.op(op_id).result_types.len();
        for i in 0..result_types_len {
            let val = find_result_value(self.module, op_id, i as u32);
            if let Some(v) = val {
                self.value_map.insert(v, self.next_idx);
                self.next_idx += 1;
            }
        }

        Ok(())
    }

    pub fn write_region(&mut self, region_id: RegionId, w: &mut EncodingWriter) -> Result<()> {
        let region = self.module.region(region_id);
        let block_ids: Vec<_> = region.blocks.clone();
        w.write_varint(block_ids.len() as u64);
        for block_id in block_ids {
            self.write_block(block_id, w)?;
        }
        Ok(())
    }

    pub fn write_block(&mut self, block_id: BlockId, w: &mut EncodingWriter) -> Result<()> {
        let saved_next_idx = self.next_idx;
        let block = self.module.block(block_id);
        let args: Vec<_> = block.args.clone();
        let ops: Vec<_> = block.ops.clone();

        // Block arguments.
        w.write_varint(args.len() as u64);
        for (val, ty) in &args {
            let idx = self.types.get_or_insert(ty);
            w.write_varint(idx);
            self.value_map.insert(*val, self.next_idx);
            self.next_idx += 1;
        }

        // Operations.
        w.write_varint(ops.len() as u64);
        for op_id in ops {
            self.write_operation(op_id, w)?;
        }

        // Roll back block-scoped values.
        self.value_map.retain(|_, v| *v < saved_next_idx);
        self.next_idx = saved_next_idx;

        Ok(())
    }
}

/// Find the Value in the module that was produced as result `result_index` of `op_id`.
fn find_result_value(module: &Module, op_id: OpId, result_index: u32) -> Option<Value> {
    for (i, vd) in module.values.iter().enumerate() {
        if let ValueProducer::OpResult {
            op,
            result_index: ri,
        } = vd.producer
        {
            if op == op_id && ri == result_index {
                return Some(Value(i as u32));
            }
        }
    }
    None
}

// =========================================================================
// Attribute serialization
// =========================================================================

pub(super) fn write_self_contained_attribute(
    attr: &Attribute,
    w: &mut EncodingWriter,
    strings: &mut StringManager,
    types: &mut TypeManager,
    constants: &mut ConstantManager,
) -> Result<()> {
    match attr {
        Attribute::Integer(v, ty) => {
            w.write_varint(AttributeTag::Integer as u64);
            let ty_idx = types.get_or_insert(ty);
            w.write_varint(ty_idx);
            w.write_varint(*v as u64);
        }
        Attribute::Float(v, ty) => {
            w.write_varint(AttributeTag::Float as u64);
            let ty_idx = types.get_or_insert(ty);
            w.write_varint(ty_idx);
            w.write_ap_float(*v, ty);
        }
        Attribute::Bool(v) => {
            w.write_varint(AttributeTag::Bool as u64);
            w.write_byte(if *v { 0x01 } else { 0x00 });
        }
        Attribute::Type(ty) => {
            w.write_varint(AttributeTag::Type as u64);
            let idx = types.get_or_insert(ty);
            w.write_varint(idx);
        }
        Attribute::String(s) => {
            w.write_varint(AttributeTag::String as u64);
            let idx = strings.get_or_insert(s);
            w.write_varint(idx);
        }
        Attribute::Array(elems) => {
            w.write_varint(AttributeTag::Array as u64);
            w.write_varint(elems.len() as u64);
            for elem in elems {
                write_self_contained_attribute(elem, w, strings, types, constants)?;
            }
        }
        Attribute::DenseElements(de) => {
            w.write_varint(AttributeTag::DenseElements as u64);
            let ty_idx = types.get_or_insert(&de.element_type);
            w.write_varint(ty_idx);
            // Serialize raw data into constant pool.
            let mut cdata = EncodingWriter::new();
            cdata.write_varint(de.data.len() as u64);
            cdata.write_bytes(&de.data);
            let const_idx = constants.add(cdata.into_bytes());
            w.write_varint(const_idx);
        }
        Attribute::DenseI32Array(arr) => {
            // Self-contained DenseI32Array: no tag needed (not a tagged attribute
            // in the bytecode format — it's always inline).
            w.write_le_var_size_i32(arr);
        }
        Attribute::DivBy(db) => {
            w.write_varint(AttributeTag::DivBy as u64);
            w.write_varint(db.divisor);
            let mut flags: u8 = 0;
            if db.every.is_some() {
                flags |= 0x01;
            }
            if db.along.is_some() {
                flags |= 0x02;
            }
            w.write_byte(flags);
            if let Some(every) = db.every {
                w.write_signed_varint(every);
            }
            if let Some(along) = db.along {
                w.write_signed_varint(along);
            }
        }
        Attribute::SameElements(se) => {
            w.write_varint(AttributeTag::SameElements as u64);
            w.write_le_var_size_i64(&se.values);
        }
        Attribute::Dictionary(entries) => {
            w.write_varint(AttributeTag::Dictionary as u64);
            w.write_varint(entries.len() as u64);
            for (key, val) in entries {
                let key_idx = strings.get_or_insert(key);
                w.write_varint(key_idx);
                write_self_contained_attribute(val, w, strings, types, constants)?;
            }
        }
        Attribute::OptimizationHints(oh) => {
            w.write_varint(AttributeTag::OptimizationHints as u64);
            // Serialize as a dictionary.
            w.write_varint(oh.entries.len() as u64);
            for (arch, hints) in &oh.entries {
                let arch_idx = strings.get_or_insert(arch);
                w.write_varint(arch_idx);
                // Each arch maps to a dictionary of hints.
                w.write_varint(AttributeTag::Dictionary as u64);
                w.write_varint(hints.len() as u64);
                for (k, v) in hints {
                    let k_idx = strings.get_or_insert(k);
                    w.write_varint(k_idx);
                    write_self_contained_attribute(v, w, strings, types, constants)?;
                }
            }
        }
        Attribute::Bounded(b) => {
            w.write_varint(AttributeTag::Bounded as u64);
            let mut flags: u8 = 0;
            if b.lb.is_some() {
                flags |= 0x01;
            }
            if b.ub.is_some() {
                flags |= 0x02;
            }
            w.write_byte(flags);
            if let Some(lb) = b.lb {
                w.write_signed_varint(lb);
            }
            if let Some(ub) = b.ub {
                w.write_signed_varint(ub);
            }
        }
    }
    Ok(())
}

// =========================================================================
// Attribute helpers
// =========================================================================

fn find_string_attr(attrs: &[(String, Attribute)], name: &str) -> Option<String> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::String(s) = v {
                return Some(s.clone());
            }
        }
        None
    })
}

fn find_type_attr(attrs: &[(String, Attribute)], name: &str) -> Option<Type> {
    attrs.iter().find_map(|(k, v)| {
        if k == name {
            if let Attribute::Type(ty) = v {
                return Some(ty.clone());
            }
        }
        None
    })
}

fn find_attr<'a>(attrs: &'a [(String, Attribute)], name: &str) -> Option<&'a Attribute> {
    attrs
        .iter()
        .find_map(|(k, v)| if k == name { Some(v) } else { None })
}

/// Attributes that are encoded in the function header, not per-operation.
#[allow(dead_code)]
fn is_function_level_attr(name: &str) -> bool {
    matches!(name, "sym_name" | "function_type" | "optimization_hints")
}
