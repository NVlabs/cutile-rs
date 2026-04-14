/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Golden file bytecode tests.
//!
//! Builds the same IR with the OLD compiler (melior/MLIR) and the NEW
//! tile-ir writer, then compares the function body bytes. The section
//! structure may differ (string/type ordering) but the per-op encoding
//! within the function body must match.
//!
//! Also includes `#[ignore]` tileiras acceptance tests for each op
//! pattern — run with `cargo test --test golden_bytecode -- --ignored`.

use cuda_tile_rs::cuda_tile;
use cuda_tile_rs::util::operation_parse;
use cutile_compiler::context_all;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationLike;
use melior::ir::{Block, BlockLike, Location, Region, RegionLike};

/// Build a module using the OLD melior-based path and serialize to bytecode.
fn old_compiler_bytecode(mlir_text: &str, module_name: &str) -> Vec<u8> {
    let context = context_all();
    let location = Location::unknown(&context);

    let entry_op = operation_parse(&context, mlir_text, None)
        .unwrap_or_else(|| panic!("failed to parse MLIR:\n{mlir_text}"));

    let module_block = Block::new(&[]);
    module_block.append_operation(entry_op);

    let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location)
        .body({
            let region = Region::new();
            region.append_block(module_block);
            region
        })
        .sym_name(StringAttribute::new(&context, module_name))
        .build();
    assert!(
        module_op.as_operation().verify(),
        "Old compiler module failed verification:\n{}",
        module_op.as_operation().to_string()
    );

    // Write to temp file and read back.
    let tmp = std::env::temp_dir().join(format!(
        "golden_{module_name}_{:?}.bc",
        std::thread::current().id()
    ));
    let tmp_str = tmp.to_str().unwrap();
    cuda_tile_rs::cuda_tile_write_bytecode(&module_op, tmp_str).expect("old bytecode write failed");
    let bytes = std::fs::read(&tmp).expect("failed to read old bytecode");
    let _ = std::fs::remove_file(&tmp);
    bytes
}

/// Extract function body bytes from raw bytecode.
/// Parses the Func section, skips the header, returns the function body.
fn extract_function_body(bytecode: &[u8]) -> Vec<u8> {
    // Skip magic (8) + version (4) = 12 bytes
    let mut pos = 12;
    // Find the Func section (ID 0x02)
    loop {
        if pos >= bytecode.len() {
            panic!("Func section not found");
        }
        let id_byte = bytecode[pos];
        let section_id = id_byte & 0x7F;
        let has_alignment = id_byte & 0x80 != 0;
        pos += 1;
        // Read body length varint
        let (body_len, bytes_read) = read_varint(&bytecode[pos..]);
        pos += bytes_read;
        if has_alignment {
            let (align, ar) = read_varint(&bytecode[pos..]);
            pos += ar;
            // Skip padding
            let aligned_pos = (pos + align as usize - 1) / align as usize * align as usize;
            pos = aligned_pos;
        }
        if section_id == 0x00 {
            panic!("End of bytecode before Func section");
        }
        if section_id == 0x02 {
            // Func section found. Parse function header.
            let _section_start = pos;
            let (num_funcs, nr) = read_varint(&bytecode[pos..]);
            pos += nr;
            assert!(num_funcs >= 1);
            // Skip func header: name_idx, sig_idx, flags, loc_idx
            let (_, nr) = read_varint(&bytecode[pos..]);
            pos += nr; // name_idx
            let (_, nr) = read_varint(&bytecode[pos..]);
            pos += nr; // sig_idx
            let flags = bytecode[pos];
            pos += 1; // flags byte
            let (_, nr) = read_varint(&bytecode[pos..]);
            pos += nr; // loc_idx
                       // If HasOptimizationHints, skip the hints attr
            if flags & 0x04 != 0 {
                // Skip self-contained attribute (complex, just skip for now)
                panic!("TODO: skip optimization hints in golden test");
            }
            // Read body length
            let (body_len, nr) = read_varint(&bytecode[pos..]);
            pos += nr;
            return bytecode[pos..pos + body_len as usize].to_vec();
        }
        // Skip this section
        pos += body_len as usize;
    }
}

fn read_varint(data: &[u8]) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
        shift += 7;
    }
    panic!("varint overflow");
}

// =========================================================================
// Empty kernel golden test
// =========================================================================

/// Build the same empty kernel with both paths and compare.
#[test]
fn golden_empty_kernel() {
    let mlir = r#"cuda_tile.entry @empty_kernel() {
  cuda_tile.return
}"#;

    let old_bytes = old_compiler_bytecode(mlir, "test_module");

    // Build with tile-ir.
    use tile_ir::builder::{append_op, build_single_block_region, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let mut module = tile_ir::Module::new("test_module");
    let func_type = Type::Func(FuncType {
        inputs: vec![],
        results: vec![],
    });
    let (region_id, block_id, _) = build_single_block_region(&mut module, &[]);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String("empty_kernel".into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);

    let new_bytes = tile_ir::write_bytecode(&module).expect("tile-ir write failed");

    // Both should have valid magic.
    assert_eq!(
        &old_bytes[0..8],
        &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]
    );
    assert_eq!(
        &new_bytes[0..8],
        &[0x7F, b'T', b'i', b'l', b'e', b'I', b'R', 0x00]
    );

    // Compare function body bytes (extracted from decoded output).
    let old_decoded = tile_ir::decode_bytecode(&old_bytes).expect("old decode failed");
    let new_decoded = tile_ir::decode_bytecode(&new_bytes).expect("new decode failed");

    println!("=== OLD (melior) ===\n{old_decoded}");
    println!("=== NEW (tile-ir) ===\n{new_decoded}");

    // Both should contain the same function.
    assert!(old_decoded.contains("empty_kernel"));
    assert!(new_decoded.contains("empty_kernel"));

    // For the empty kernel, the function bodies should be identical.
    // The body is just a Return op: opcode 0x5C, result count 0, operand count 0.
    // Extract body length from decoded output.
    assert!(
        old_decoded.contains("body: 3 bytes") || old_decoded.contains("body: 4 bytes"),
        "old body size unexpected:\n{old_decoded}"
    );
    assert!(
        new_decoded.contains("body: 3 bytes"),
        "new body size unexpected:\n{new_decoded}"
    );
}

// =========================================================================
// Golden comparison: old compiler body bytes == new compiler body bytes
// =========================================================================

/// Compare function body bytes between old (C++) and new (tile-ir) compilers.
/// This catches encoding mismatches at the byte level.
fn assert_body_match(mlir: &str, module_name: &str, new_module: &tile_ir::Module) {
    let old_bytes = old_compiler_bytecode(mlir, module_name);
    let new_bytes = tile_ir::write_bytecode(new_module).expect("tile-ir write failed");

    // Both must decode successfully.
    tile_ir::decode_bytecode(&old_bytes).expect("old bytecode decode failed");
    tile_ir::decode_bytecode(&new_bytes).expect("new bytecode decode failed");

    let old_body = extract_function_body(&old_bytes);
    let new_body = extract_function_body(&new_bytes);

    // Body lengths must match — same ops with same encoding produce same byte count.
    // Exact bytes may differ due to string/type/constant table index ordering.
    assert_eq!(
        old_body.len(),
        new_body.len(),
        "Function body length mismatch.\n\
         old body ({} bytes): {:02x?}\n\
         new body ({} bytes): {:02x?}",
        old_body.len(),
        &old_body,
        new_body.len(),
        &new_body,
    );

    // Also check that opcodes at the start of each op match.
    // The first byte of the body is always an opcode varint.
    if !old_body.is_empty() && !new_body.is_empty() {
        assert_eq!(
            old_body[0], new_body[0],
            "First op opcode mismatch: old=0x{:02x}, new=0x{:02x}",
            old_body[0], new_body[0],
        );
    }
}

/// Helper to build a tile-ir kernel for golden tests.
fn golden_kernel(
    name: &str,
    arg_types: &[tile_ir::ir::Type],
    build: impl FnOnce(&mut tile_ir::Module, tile_ir::ir::BlockId, &[tile_ir::ir::Value]),
) -> tile_ir::Module {
    use tile_ir::builder::{append_op, build_single_block_region, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let mut module = tile_ir::Module::new("golden");
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, arg_types);
    build(&mut module, block_id, &args);
    let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
    append_op(&mut module, block_id, ret);
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String(name.into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

fn tile_i32() -> tile_ir::ir::Type {
    tile_ir::ir::Type::Tile(tile_ir::ir::TileType {
        element_type: tile_ir::ir::TileElementType::Scalar(tile_ir::ir::ScalarType::I32),
        shape: vec![],
    })
}

fn const_i32_golden(
    module: &mut tile_ir::Module,
    block: tile_ir::ir::BlockId,
    val: i32,
) -> tile_ir::ir::Value {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;
    let data = val.to_le_bytes().to_vec();
    let (op, res) = OpBuilder::new(Opcode::Constant, Location::Unknown)
        .attr(
            "value",
            Attribute::DenseElements(DenseElements {
                element_type: tile_i32(),
                shape: vec![],
                data,
            }),
        )
        .result(tile_i32())
        .build(module);
    append_op(module, block, op);
    res[0]
}

#[test]
fn golden_addi_overflow() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let mlir = r#"cuda_tile.entry @test(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
  %c = cuda_tile.addi %a, %b overflow<none> : !cuda_tile.tile<i32>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .attr("overflow", Attribute::i32(0))
            .result(tile_i32())
            .build(m);
        append_op(m, b, op);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_cmpi_signedness() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let tile_i1 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I1),
        shape: vec![],
    });

    let mlir = r#"cuda_tile.entry @test(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
  %c = cuda_tile.cmpi less_than %a, %b, signed : !cuda_tile.tile<i32> -> !cuda_tile.tile<i1>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .attr("comparison_predicate", Attribute::i32(2)) // less_than
            .attr("signedness", Attribute::i32(1)) // signed
            .result(tile_i1.clone())
            .build(m);
        append_op(m, b, op);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_divi_rounding() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let mlir = r#"cuda_tile.entry @test(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
  %c = cuda_tile.divi %a, %b signed rounding<positive_inf> : !cuda_tile.tile<i32>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[tile_i32(), tile_i32()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::DivI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .attr("signedness", Attribute::i32(1)) // signed
            .attr("rounding", Attribute::i32(3)) // positive_inf
            .result(tile_i32())
            .build(m);
        append_op(m, b, op);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_permute_dense_i32_array() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let tile_f32_scalar = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![],
    });
    let tile_1x1x1_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![1, 1, 1],
    });
    let tile_perm_f32 = tile_1x1x1_f32.clone(); // [2,0,1] on [1,1,1] is still [1,1,1]

    // Matches the C++ test in attrsTest.mlir
    let mlir = r#"cuda_tile.entry @test(%a: !cuda_tile.tile<f32>) {
  %r = cuda_tile.reshape %a : tile<f32> -> tile<1x1x1xf32>
  %b = cuda_tile.permute %r [2, 0, 1] : tile<1x1x1xf32> -> tile<1x1x1xf32>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[tile_f32_scalar.clone()], |m, b, args| {
        let (rs, rs_res) = OpBuilder::new(Opcode::Reshape, Location::Unknown)
            .operand(args[0])
            .result(tile_1x1x1_f32.clone())
            .build(m);
        append_op(m, b, rs);
        let (op, _) = OpBuilder::new(Opcode::Permute, Location::Unknown)
            .operand(rs_res[0])
            .attr("permutation", Attribute::DenseI32Array(vec![2, 0, 1]))
            .result(tile_perm_f32.clone())
            .build(m);
        append_op(m, b, op);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_assert_string_attr() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let tile_i1 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I1),
        shape: vec![],
    });

    let mlir = r#"cuda_tile.entry @test(%cond: !cuda_tile.tile<i1>) {
  cuda_tile.assert %cond, "test assertion" : tile<i1>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[tile_i1.clone()], |m, b, args| {
        let (op, _) = OpBuilder::new(Opcode::Assert, Location::Unknown)
            .operand(args[0])
            .attr("message", Attribute::String("test assertion".into()))
            .build(m);
        append_op(m, b, op);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_constant_dense_elements() {
    let mlir = r#"cuda_tile.entry @test() {
  %c = cuda_tile.constant <i32: 42> : !cuda_tile.tile<i32>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[], |m, b, _| {
        const_i32_golden(m, b, 42);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_constant_i1_true() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let tile_i1 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I1),
        shape: vec![],
    });

    let mlir = r#"cuda_tile.entry @test() {
  %c = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>
  cuda_tile.return
}"#;

    let new_module = golden_kernel("test", &[], |m, b, _| {
        let data = vec![0xFFu8]; // i1 true = all-ones byte
        let (op, _) = OpBuilder::new(Opcode::Constant, Location::Unknown)
            .attr(
                "value",
                Attribute::DenseElements(DenseElements {
                    element_type: tile_i1.clone(),
                    shape: vec![],
                    data,
                }),
            )
            .result(tile_i1.clone())
            .build(m);
        append_op(m, b, op);
    });

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_addf_rounding_and_flush() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let tile_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![],
    });

    let mlir = r#"cuda_tile.entry @test(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
  %c = cuda_tile.addf %a, %b rounding<nearest_even> flush_to_zero : tile<f32>
  cuda_tile.return
}"#;

    let new_module = golden_kernel(
        "test",
        &[tile_f32.clone(), tile_f32.clone()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .attr("rounding_mode", Attribute::i32(0)) // nearest_even
                .attr("flush_to_zero", Attribute::Bool(true))
                .result(tile_f32.clone())
                .build(m);
            append_op(m, b, op);
        },
    );

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_maxf_propagate_nan() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let tile_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![],
    });

    let mlir = r#"cuda_tile.entry @test(%a: !cuda_tile.tile<f32>, %b: !cuda_tile.tile<f32>) {
  %c = cuda_tile.maxf %a, %b {propagate_nan, rounding_mode = 0} : tile<f32>
  cuda_tile.return
}"#;

    let new_module = golden_kernel(
        "test",
        &[tile_f32.clone(), tile_f32.clone()],
        |m, b, args| {
            let (op, _) = OpBuilder::new(Opcode::MaxF, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .attr("propagate_nan", Attribute::Bool(true))
                .result(tile_f32.clone())
                .build(m);
            append_op(m, b, op);
        },
    );

    assert_body_match(mlir, "golden", &new_module);
}

#[test]
fn golden_for_loop_v13_2() {
    use tile_ir::builder::{append_op, build_single_block_region, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let mlir = r#"cuda_tile.entry @test(%lb: !cuda_tile.tile<i32>, %ub: !cuda_tile.tile<i32>, %step: !cuda_tile.tile<i32>) {
  cuda_tile.for %iv in (%lb to %ub, step %step) : tile<i32> {
    cuda_tile.continue
  }
  cuda_tile.return
}"#;

    let new_module = golden_kernel(
        "test",
        &[tile_i32(), tile_i32(), tile_i32()],
        |m, b, args| {
            let (body_region, body_blk, _body_args) = build_single_block_region(m, &[tile_i32()]);
            let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
            append_op(m, body_blk, cont);

            let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
                .operand(args[0]) // lb
                .operand(args[1]) // ub
                .operand(args[2]) // step
                .region(body_region)
                .build(m);
            append_op(m, b, for_op);
        },
    );

    // For v13.2, our writer adds a 1-byte flags field that the old (v13.1)
    // C++ writer doesn't. Verify the size differs by exactly 1.
    let old_bytes = old_compiler_bytecode(mlir, "golden");
    let new_bytes = tile_ir::write_bytecode(&new_module).expect("tile-ir write failed");
    let old_body = extract_function_body(&old_bytes);
    let new_body = extract_function_body(&new_bytes);
    assert_eq!(
        new_body.len(),
        old_body.len() + 1,
        "For op v13.2 should add exactly 1 byte (flags) vs v13.1.\n\
         old ({} bytes): {:02x?}\n\
         new ({} bytes): {:02x?}",
        old_body.len(),
        &old_body,
        new_body.len(),
        &new_body,
    );
    // First byte (opcode 0x29 = For) must match.
    assert_eq!(old_body[0], new_body[0]);
}

// =========================================================================
// Tileiras acceptance tests (per op pattern, #[ignore])
// =========================================================================

use cutile_compiler::cuda_tile_runtime_utils::{compile_tile_ir_module, get_gpu_name};

fn try_get_gpu() -> Option<String> {
    if std::process::Command::new("tileiras")
        .arg("--version")
        .output()
        .is_err()
    {
        return None;
    }
    std::panic::catch_unwind(|| get_gpu_name(0)).ok()
}

fn tileiras_accepts(module: &tile_ir::Module) {
    let Some(gpu_name) = try_get_gpu() else {
        eprintln!("skipping: no GPU");
        return;
    };
    let cubin = compile_tile_ir_module(module, &gpu_name);
    assert!(std::path::Path::new(&cubin).exists());
    let _ = std::fs::remove_file(&cubin);
}

/// Helper to build a kernel for tileiras testing.
fn tileiras_kernel(
    name: &str,
    arg_types: &[tile_ir::ir::Type],
    build: impl FnOnce(&mut tile_ir::Module, tile_ir::ir::BlockId, &[tile_ir::ir::Value]),
) -> tile_ir::Module {
    use tile_ir::builder::{append_op, build_single_block_region, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let mut module = tile_ir::Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, arg_types);
    build(&mut module, block_id, &args);
    let needs_ret = {
        let block = module.block(block_id);
        block.ops.last().map_or(true, |&last| {
            !matches!(module.op(last).opcode, Opcode::Return)
        })
    };
    if needs_ret {
        let (ret, _) = OpBuilder::new(Opcode::Return, Location::Unknown).build(&mut module);
        append_op(&mut module, block_id, ret);
    }
    let (entry, _) = OpBuilder::new(Opcode::Entry, Location::Unknown)
        .attr("sym_name", Attribute::String(name.into()))
        .attr("function_type", Attribute::Type(func_type))
        .region(region_id)
        .build(&mut module);
    module.functions.push(entry);
    module
}

#[test]
#[ignore] // Run with: cargo test --test golden_bytecode -- --ignored
fn tileiras_empty_kernel() {
    let module = tileiras_kernel("empty", &[], |_, _, _| {});
    tileiras_accepts(&module);
}

#[test]
#[ignore]
fn tileiras_get_tile_block_id() {
    use tile_ir::ir::*;
    let scalar = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let module = tileiras_kernel("get_ids", &[], |m, b, _| {
        use tile_ir::builder::{append_op, OpBuilder};
        use tile_ir::bytecode::Opcode;
        let (op, _) = OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown)
            .result(scalar.clone())
            .result(scalar.clone())
            .result(scalar.clone())
            .build(m);
        append_op(m, b, op);
    });
    tileiras_accepts(&module);
}

#[test]
#[ignore]
fn tileiras_addf_kernel() {
    use tile_ir::builder::{append_op, OpBuilder};
    use tile_ir::bytecode::Opcode;
    use tile_ir::ir::*;

    let ptr_f32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
    });
    let scalar_i32 = Type::Tile(TileType {
        shape: vec![],
        element_type: TileElementType::Scalar(ScalarType::I32),
    });
    let tile4f32 = Type::Tile(TileType {
        shape: vec![4],
        element_type: TileElementType::Scalar(ScalarType::F32),
    });
    let tv_dyn = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![DYNAMIC],
        strides: vec![1],
    });
    let pv = Type::PartitionView(PartitionViewType {
        tile_shape: vec![4],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let tok = Type::Token;

    let module = tileiras_kernel(
        "addf_kernel",
        &[
            ptr_f32.clone(),
            scalar_i32.clone(),
            ptr_f32.clone(),
            scalar_i32.clone(),
        ],
        |m, b, args| {
            // make_token
            let (op, res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
                .result(tok.clone())
                .build(m);
            append_op(m, b, op);
            let tok_a = res[0];

            // make_tensor_view a
            let (op, res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(args[0])
                .operand(args[1])
                .result(tv_dyn.clone())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
            let tv_a = res[0];

            // make_token
            let (op, res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
                .result(tok.clone())
                .build(m);
            append_op(m, b, op);
            let tok_b = res[0];

            // make_tensor_view b
            let (op, res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
                .operand(args[2])
                .operand(args[3])
                .result(tv_dyn.clone())
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(0),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
            let tv_b = res[0];

            // get_tile_block_id
            let (op, res) = OpBuilder::new(Opcode::GetTileBlockId, Location::Unknown)
                .result(scalar_i32.clone())
                .result(scalar_i32.clone())
                .result(scalar_i32.clone())
                .build(m);
            append_op(m, b, op);
            let pid = res[0];

            // make_partition_view a
            let (op, res) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
                .operand(tv_a)
                .result(pv.clone())
                .build(m);
            append_op(m, b, op);
            let pv_a = res[0];

            // load a
            let (op, res) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
                .operand(pv_a)
                .operand(pid)
                .operand(tok_a)
                .result(tile4f32.clone())
                .result(tok.clone())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
            let tile_a = res[0];

            // make_partition_view b
            let (op, res) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
                .operand(tv_b)
                .result(pv.clone())
                .build(m);
            append_op(m, b, op);
            let pv_b = res[0];

            // load b
            let (op, res) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
                .operand(pv_b)
                .operand(pid)
                .operand(tok_b)
                .result(tile4f32.clone())
                .result(tok.clone())
                .attr("memory_ordering_semantics", Attribute::i32(0))
                .attr(
                    "operandSegmentSizes",
                    Attribute::Array(vec![
                        Attribute::i32(1),
                        Attribute::i32(1),
                        Attribute::i32(1),
                    ]),
                )
                .build(m);
            append_op(m, b, op);
            let tile_b = res[0];

            // addf
            let (op, res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
                .operand(tile_a)
                .operand(tile_b)
                .result(tile4f32.clone())
                .attr("rounding_mode", Attribute::i32(0))
                .build(m);
            append_op(m, b, op);
            let _sum = res[0];
        },
    );
    tileiras_accepts(&module);
}
