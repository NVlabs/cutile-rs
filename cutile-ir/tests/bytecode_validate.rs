/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bytecode structural validation tests.
//!
//! These tests build IR with various patterns, write bytecode, and validate
//! the output using:
//! 1. Module-level `verify_bytecode_indices()` (value numbering consistency)
//! 2. tileiras parsing (byte-level format correctness)
//!
//! Run with: cargo test -p tile-ir --test bytecode_validate -- --nocapture

use cutile_ir::builder::{append_op, build_single_block_region, OpBuilder};
use cutile_ir::bytecode::Opcode;
use cutile_ir::ir::*;

// =========================================================================
// Helpers
// =========================================================================

fn i32_ty() -> Type {
    Type::Scalar(ScalarType::I32)
}

fn tile_i32() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I32),
        shape: vec![],
    })
}

fn tile_f32() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![],
    })
}

fn tile_i1() -> Type {
    Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::I1),
        shape: vec![],
    })
}

fn token_ty() -> Type {
    Type::Token
}

fn const_i32(module: &mut Module, block: BlockId, val: i64) -> Value {
    let data = (val as i32).to_le_bytes().to_vec();
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

fn build_kernel(
    name: &str,
    arg_types: &[Type],
    build_body: impl FnOnce(&mut Module, BlockId, &[Value]),
) -> Module {
    let mut module = Module::new("test");
    let func_type = Type::Func(FuncType {
        inputs: arg_types.to_vec(),
        results: vec![],
    });
    let (region_id, block_id, args) = build_single_block_region(&mut module, arg_types);
    build_body(&mut module, block_id, &args);
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

/// Validate a module: dominance, bytecode indices, write bytecode, and
/// optionally run tileiras to check byte-level format.
fn validate_module(module: &Module) {
    module.verify_dominance().expect("dominance check failed");
    module
        .verify_bytecode_indices()
        .expect("bytecode index check failed");

    let bc = cutile_ir::write_bytecode(module).expect("write_bytecode failed");

    // Verify our own decoder can parse it.
    cutile_ir::decode_bytecode(&bc).expect("our decoder rejected the bytecode");

    // Try tileiras if available (it may not be installed in CI).
    run_tileiras(&bc, &module.name);
}

fn run_tileiras(bc: &[u8], name: &str) {
    let tmp = std::env::temp_dir().join(format!(
        "tile_ir_test_{}_{:?}.bc",
        name,
        std::thread::current().id()
    ));
    std::fs::write(&tmp, bc).expect("write bc file");
    match std::process::Command::new("tileiras")
        .arg("--gpu-name")
        .arg("sm_120")
        .arg("-o")
        .arg("/dev/null")
        .arg(tmp.to_str().unwrap())
        .output()
    {
        Ok(out) => {
            std::fs::remove_file(&tmp).ok();
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                panic!("tileiras rejected bytecode for module '{name}':\n{stderr}");
            }
        }
        Err(_) => {
            // tileiras not available — skip byte-level check
            std::fs::remove_file(&tmp).ok();
        }
    }
}

// =========================================================================
// Test cases
// =========================================================================

#[test]
fn simple_arithmetic() {
    let module = build_kernel("simple_arith", &[tile_i32(), tile_i32()], |m, blk, args| {
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[1])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, blk, add);
    });
    validate_module(&module);
}

#[test]
fn for_loop_with_parent_scope_refs() {
    // A for-loop whose body uses values from the parent scope.
    let module = build_kernel("for_parent_ref", &[tile_i32()], |m, blk, args| {
        let parent_val = args[0];
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // for %iv = lb to ub step step { addi parent_val, %iv }
        let (body_region, body_blk, body_args) = build_single_block_region(m, &[tile_i32()]);

        // body: use parent_val + iv
        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(parent_val)
            .operand(body_args[0])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, body_blk, add);

        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);
    });
    validate_module(&module);
}

#[test]
fn for_loop_with_iter_args() {
    // for-loop with carried values (iter args).
    let module = build_kernel("for_iter_args", &[tile_i32()], |m, blk, args| {
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);
        let init = args[0];

        // for %iv, %acc = lb to ub step step iter(%init) { continue %acc + %iv }
        let (body_region, body_blk, body_args) =
            build_single_block_region(m, &[tile_i32(), tile_i32()]); // iv, acc

        let (add, add_res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(body_args[1]) // acc
            .operand(body_args[0]) // iv
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, body_blk, add);

        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .operand(init) // init values
            .result(tile_i32()) // carried result
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);
    });
    validate_module(&module);
}

#[test]
fn nested_for_loops() {
    // Outer for containing an inner for.
    let module = build_kernel("nested_for", &[tile_i32()], |m, blk, args| {
        let parent_val = args[0];
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // Outer for
        let (outer_region, outer_blk, outer_args) = build_single_block_region(m, &[tile_i32()]);

        // Inner for (uses outer iv and parent_val)
        let (inner_region, inner_blk, inner_args) = build_single_block_region(m, &[tile_i32()]);

        let (add, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(parent_val) // from grandparent
            .operand(outer_args[0]) // from outer loop
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, inner_blk, add);

        let (add2, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(inner_args[0]) // inner iv
            .operand(outer_args[0]) // outer iv
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, inner_blk, add2);

        let (cont_inner, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, inner_blk, cont_inner);

        let (inner_for, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(inner_region)
            .build(m);
        append_op(m, outer_blk, inner_for);

        let (cont_outer, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, outer_blk, cont_outer);

        let (outer_for, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(outer_region)
            .build(m);
        append_op(m, blk, outer_for);
    });
    validate_module(&module);
}

#[test]
fn if_else_with_yields() {
    // if-else that yields values.
    let module = build_kernel("if_yields", &[tile_i32()], |m, blk, args| {
        // condition
        let one = const_i32(m, blk, 1);
        let (cmp, cmp_res) = OpBuilder::new(Opcode::CmpI, Location::Unknown)
            .operand(args[0])
            .operand(one)
            .attr("comparison_predicate", Attribute::Integer(0, i32_ty()))
            .attr("signedness", Attribute::Integer(0, i32_ty()))
            .result(tile_i1())
            .build(m);
        append_op(m, blk, cmp);

        // then
        let (then_region, then_blk, _) = build_single_block_region(m, &[]);
        let (add, add_res) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(args[0])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, then_blk, add);
        let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, then_blk, yld);

        // else
        let (else_region, else_blk, _) = build_single_block_region(m, &[]);
        let (yld2, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(args[0])
            .build(m);
        append_op(m, else_blk, yld2);

        // if
        let (if_op, _) = OpBuilder::new(Opcode::If, Location::Unknown)
            .operand(cmp_res[0])
            .result(tile_i32())
            .region(then_region)
            .region(else_region)
            .build(m);
        append_op(m, blk, if_op);
    });
    validate_module(&module);
}

#[test]
fn reduce_with_combiner() {
    // Reduce op with a combiner region.
    // Entry args must be scalar tiles, so we reshape + broadcast.
    let tile_1_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![1],
    });
    let tile_8_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![8],
    });

    let module = build_kernel("reduce_test", &[tile_f32()], |m, blk, args| {
        // Reshape scalar to tile<1xf32>, then broadcast to tile<8xf32>
        let (rs_op, rs_res) = OpBuilder::new(Opcode::Reshape, Location::Unknown)
            .operand(args[0])
            .result(tile_1_f32.clone())
            .build(m);
        append_op(m, blk, rs_op);
        let (bc_op, bc_res) = OpBuilder::new(Opcode::Broadcast, Location::Unknown)
            .operand(rs_res[0])
            .result(tile_8_f32.clone())
            .build(m);
        append_op(m, blk, bc_op);
        let args = &[bc_res[0]];
        // Combiner region: two scalar args -> yield their sum
        let (combiner_region, combiner_blk, combiner_args) =
            build_single_block_region(m, &[tile_f32(), tile_f32()]);
        let (add, add_res) = OpBuilder::new(Opcode::AddF, Location::Unknown)
            .operand(combiner_args[0])
            .operand(combiner_args[1])
            .attr("rounding_mode", Attribute::Integer(0, i32_ty()))
            .result(tile_f32())
            .build(m);
        append_op(m, combiner_blk, add);
        let (yld, _) = OpBuilder::new(Opcode::Yield, Location::Unknown)
            .operand(add_res[0])
            .build(m);
        append_op(m, combiner_blk, yld);

        // reduce
        let (red, _) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
            .operand(args[0])
            .attr("dim", Attribute::Integer(0, i32_ty()))
            .attr(
                "identities",
                Attribute::Array(vec![Attribute::Float(0.0, Type::Scalar(ScalarType::F32))]),
            )
            .result(tile_f32())
            .region(combiner_region)
            .build(m);
        append_op(m, blk, red);
    });
    validate_module(&module);
}

#[test]
fn many_ops_after_nested_region() {
    // After a for-loop, subsequent ops must use correct value indices.
    // This pattern catches rollback bugs.
    let module = build_kernel("after_nested", &[tile_i32()], |m, blk, args| {
        let lb = const_i32(m, blk, 0);
        let ub = const_i32(m, blk, 10);
        let step = const_i32(m, blk, 1);

        // for-loop
        let (body_region, body_blk, body_args) = build_single_block_region(m, &[tile_i32()]);
        let (add_inner, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(body_args[0])
            .operand(args[0])
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, body_blk, add_inner);
        let (cont, _) = OpBuilder::new(Opcode::Continue, Location::Unknown).build(m);
        append_op(m, body_blk, cont);

        let (for_op, _) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operand(lb)
            .operand(ub)
            .operand(step)
            .region(body_region)
            .build(m);
        append_op(m, blk, for_op);

        // After the for-loop: operations that use pre-loop values.
        // If rollback is wrong, these operand indices will be off.
        let c1 = const_i32(m, blk, 42);
        let (add_after, _) = OpBuilder::new(Opcode::AddI, Location::Unknown)
            .operand(args[0])
            .operand(c1)
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, blk, add_after);

        let (mul_after, _) = OpBuilder::new(Opcode::MulI, Location::Unknown)
            .operand(args[0])
            .operand(c1)
            .attr("overflow", Attribute::Integer(0, i32_ty()))
            .result(tile_i32())
            .build(m);
        append_op(m, blk, mul_after);
    });
    validate_module(&module);
}

#[test]
fn load_store_with_tokens() {
    // LoadViewTko and StoreViewTko with token threading.
    let tile_ptr_f32 = Type::Tile(TileType {
        element_type: TileElementType::Pointer(Box::new(PointerType {
            pointee: ScalarType::F32,
        })),
        shape: vec![],
    });
    let tv_ty = Type::TensorView(TensorViewType {
        element_type: ScalarType::F32,
        shape: vec![128],
        strides: vec![1],
    });
    let pv_ty = Type::PartitionView(PartitionViewType {
        tile_shape: vec![128],
        tensor_view: TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![128],
            strides: vec![1],
        },
        dim_map: vec![0],
        padding_value: None,
    });
    let tile_128_f32 = Type::Tile(TileType {
        element_type: TileElementType::Scalar(ScalarType::F32),
        shape: vec![128],
    });

    let module = build_kernel("load_store", &[tile_ptr_f32.clone()], |m, blk, args| {
        // make_token
        let (tok_op, tok_res) = OpBuilder::new(Opcode::MakeToken, Location::Unknown)
            .result(token_ty())
            .build(m);
        append_op(m, blk, tok_op);

        // make_tensor_view
        let (mtv, mtv_res) = OpBuilder::new(Opcode::MakeTensorView, Location::Unknown)
            .operand(args[0])
            .result(tv_ty.clone())
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::Integer(1, i32_ty()),
                    Attribute::Integer(0, i32_ty()),
                    Attribute::Integer(0, i32_ty()),
                ]),
            )
            .build(m);
        append_op(m, blk, mtv);

        // make_partition_view
        let (mpv, mpv_res) = OpBuilder::new(Opcode::MakePartitionView, Location::Unknown)
            .operand(mtv_res[0])
            .result(pv_ty.clone())
            .build(m);
        append_op(m, blk, mpv);

        // load_view_tko
        let idx = const_i32(m, blk, 0);
        let (load, load_res) = OpBuilder::new(Opcode::LoadViewTko, Location::Unknown)
            .operand(mpv_res[0]) // view
            .operand(idx) // index
            .operand(tok_res[0]) // token
            .attr("memory_ordering_semantics", Attribute::Integer(0, i32_ty()))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::Integer(1, i32_ty()),
                    Attribute::Integer(1, i32_ty()),
                    Attribute::Integer(1, i32_ty()),
                ]),
            )
            .result(tile_128_f32.clone())
            .result(token_ty())
            .build(m);
        append_op(m, blk, load);

        // store_view_tko
        let (store, _) = OpBuilder::new(Opcode::StoreViewTko, Location::Unknown)
            .operand(load_res[0]) // tile
            .operand(mpv_res[0]) // view
            .operand(idx) // index
            .operand(load_res[1]) // token
            .attr("memory_ordering_semantics", Attribute::Integer(0, i32_ty()))
            .attr(
                "operandSegmentSizes",
                Attribute::Array(vec![
                    Attribute::Integer(1, i32_ty()),
                    Attribute::Integer(1, i32_ty()),
                    Attribute::Integer(1, i32_ty()),
                    Attribute::Integer(1, i32_ty()),
                ]),
            )
            .result(token_ty())
            .build(m);
        append_op(m, blk, store);
    });
    validate_module(&module);
}
