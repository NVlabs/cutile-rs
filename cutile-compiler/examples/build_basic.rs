/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_tile_rs::util::{operation_parse, parse_named_attr};
use melior::dialect::DialectRegistry;
use melior::ir::attribute::{StringAttribute, TypeAttribute};
use melior::ir::r#type::FunctionType;
use melior::ir::{
    Attribute, Block, BlockLike, Identifier, Location, Region, RegionLike, Type, Value, ValueLike,
};
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::Context;
use std::error::Error;

use cuda_tile_rs::{cuda_tile, register_cuda_tile_dialects};
use melior::ir::operation::{Operation, OperationBuilder, OperationLike};

pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    register_cuda_tile_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

pub fn context_all() -> Context {
    let context = Context::new();
    load_all_dialects(&context);
    register_all_llvm_translations(&context);
    context.attach_diagnostic_handler(|diagnostic| {
        eprintln!("{}", diagnostic);
        true
    });
    context
}

fn main() -> Result<(), Box<dyn Error>> {
    let context = context_all();
    let location = Location::unknown(&context);
    let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location).body(
        {
            let module_block = Block::new(&[]);

            // func start
            let argument_types = vec![
                Type::parse(&context, "!cuda_tile.tile<!cuda_tile.ptr<f32>>").unwrap(),
                Type::parse(&context, "!cuda_tile.tile<i32>").unwrap(),
                Type::parse(&context, "!cuda_tile.tile<i32>").unwrap(),
                Type::parse(&context, "!cuda_tile.tile<i32>").unwrap(),
                Type::parse(&context, "!cuda_tile.tile<i32>").unwrap(),
                Type::parse(&context, "!cuda_tile.tile<f32>").unwrap(),
                Type::parse(&context, "!cuda_tile.tile<f32>").unwrap(),
            ];
            let result_types = vec![];
            let function_type = FunctionType::new(&context, &argument_types, &result_types);
            let name_attr_v = StringAttribute::new(&context, "some_func");
            let type_attr = TypeAttribute::new(function_type.into());
            let f = cuda_tile::EntryOperationBuilder::new(&context, location)
                .sym_name(name_attr_v)
                .function_type(type_attr)
                .body({
                    let func_block = Block::new(&[]);
                    func_block.add_argument(argument_types[0], location);
                    func_block.add_argument(argument_types[1], location);
                    func_block.add_argument(argument_types[2], location);
                    func_block.add_argument(argument_types[3], location);
                    func_block.add_argument(argument_types[4], location);
                    func_block.add_argument(argument_types[5], location);
                    func_block.add_argument(argument_types[6], location);

                    let base: Value = func_block.argument(0)?.into();
                    let dim1: Value = func_block.argument(1)?.into();
                    let dim2: Value = func_block.argument(2)?.into();
                    let stride1: Value = func_block.argument(3)?.into();
                    let stride2: Value = func_block.argument(4)?.into();
                    let float1: Value = func_block.argument(5)?.into();
                    let _float2: Value = func_block.argument(6)?.into();
                    let tensor_view_type = Type::parse(&context, "!cuda_tile.tensor_view<?x?xf32, strides=[?,?]>").unwrap();
                    let make_tensor_view_op_builder = OperationBuilder::new("cuda_tile.make_tensor_view", Location::unknown(&context));
                    let make_tensor_view_op = make_tensor_view_op_builder
                        .add_operands(&[base])
                        .add_operands(&[dim1, dim2])
                        .add_operands(&[stride1, stride2])
                        .add_attributes(&[(
                            Identifier::new(&context, "operandSegmentSizes"),
                            Attribute::parse(&context, "array<i32: 1, 2, 2>").unwrap()
                        )])
                        .add_results(&[tensor_view_type])
                        .build()?;

                    let make_tensor_view_op_ref = func_block.append_operation(make_tensor_view_op.into());
                    let tensor_view_value: Value = make_tensor_view_op_ref.result(0)?.into();

                    let make_partition_view_builder = OperationBuilder::new("cuda_tile.make_partition_view", Location::unknown(&context));
                    let partition_view_type = Type::parse(&context, "!cuda_tile.partition_view<tile=(128x256), !cuda_tile.tensor_view<?x?xf32, strides=[?,?]>>").unwrap();
                    let make_partition_view_op = make_partition_view_builder
                        .add_operands(&[tensor_view_value])
                        .add_results(&[partition_view_type])
                        .build()?;

                    let make_partition_view_op_ref = func_block.append_operation(make_partition_view_op.into());
                    let _partition_view_value: Value = make_partition_view_op_ref.result(0)?.into();

                    let add_op_builder = OperationBuilder::new("cuda_tile.addi", Location::unknown(&context));
                    let add_op = add_op_builder
                        .add_operands(&[dim1, dim2])
                        .add_results(&[dim1.r#type()])
                        .build()?;
                    let add_op_ref = func_block.append_operation(add_op.into());
                    let add_op_result: Value = add_op_ref.result(0)?.into();
                    let add_ptr_op_builder = OperationBuilder::new("cuda_tile.offset", Location::unknown(&context));
                    let add_ptr_op = add_ptr_op_builder
                        .add_operands(&[base, add_op_result])
                        .add_results(&[base.r#type()])
                        .build()?;
                    let add_ptr_op_ref = func_block.append_operation(add_ptr_op.into());
                    let _add_ptr_op_result: Value = add_ptr_op_ref.result(0)?.into();

                    let add_float_op_builder = OperationBuilder::new("cuda_tile.addf", Location::unknown(&context));
                    let rounding_attr = (Identifier::new(&context, "rounding_mode"),
                                         Attribute::parse(&context, "#cuda_tile.rounding<nearest_even>").unwrap());
                    let add_float_op = add_float_op_builder
                        .add_attributes(&[rounding_attr])
                        .add_operands(&[float1, float1])
                        .add_results(&[float1.r#type()])
                        .build()?;
                    let add_float_op_ref = func_block.append_operation(add_float_op.into());
                    let _add_float_op_result: Value = add_float_op_ref.result(0)?.into();

                    let permute_op_builder = OperationBuilder::new("cuda_tile.permute", Location::unknown(&context));
                    let permutation = (
                        Identifier::new(&context, "permutation"),
                        Attribute::parse(&context, "array<i32: 1, 0>").unwrap()
                    );

                    let matrix_op = operation_parse(&context, "%0 = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<128x256xf32>", None).unwrap();
                    let matrix_op_ref = func_block.append_operation(matrix_op.into());
                    let matrix_zeros_value: Value = matrix_op_ref.result(0)?.into();

                    let permute_op = permute_op_builder
                        .add_attributes(&[permutation])
                        .add_operands(&[matrix_zeros_value])
                        .add_results(&[Type::parse(&context, "!cuda_tile.tile<256x128xf32>").unwrap()])
                        .build()?;
                    let permute_op_ref = func_block.append_operation(permute_op.into());
                    let _permute_op_result: Value = permute_op_ref.result(0)?.into();

                    let lower_bound = operation_parse(&context, "%lowerBound = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>", None).unwrap();
                    let upper_bound = operation_parse(&context, "%upperBound = cuda_tile.constant <i32: 10> : !cuda_tile.tile<i32>", None).unwrap();
                    let step = operation_parse(&context, "%step = cuda_tile.constant <i32: 1> : !cuda_tile.tile<i32>", None).unwrap();
                    let lower_bound: Value = func_block.append_operation(lower_bound).result(0).unwrap().into();
                    let upper_bound: Value = func_block.append_operation(upper_bound).result(0).unwrap().into();
                    let step: Value = func_block.append_operation(step).result(0).unwrap().into();

                    // for
                    let init_val: Value = func_block.append_operation(
                        operation_parse(&context, "%initVal0 = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>", None).unwrap()
                    ).result(0).unwrap().into();
                    let random_val_1: Value = func_block.append_operation(
                        operation_parse(&context, "%random_val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>", None).unwrap()
                    ).result(0).unwrap().into();
                    let random_val_2: Value = func_block.append_operation(
                        operation_parse(&context, "%random_val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>", None).unwrap()
                    ).result(0).unwrap().into();
                    // Everything used in a for loop:
                    // 1. Needs to be passed as an operand.
                    // 2. Needs to be a block argument.
                    // 3. Needs to be loop-carried.
                    // 4. Needs to be returned.
                    let for_iterand_type = lower_bound.r#type();
                    let for_args = &[init_val, random_val_1, random_val_2];
                    let for_arg_tys = for_args.iter().map(|val| val.r#type()).collect::<Vec<_>>();

                    let for_builder = OperationBuilder::new("cuda_tile.for", Location::unknown(&context));
                    let for_op = for_builder
                        .add_operands(&[lower_bound, upper_bound, step])
                        .add_operands(for_args)
                        .add_results(&for_arg_tys)
                        .add_regions([{
                            // Can we have block arguments that aren't specified here?
                            let loop_block_args = &[&[for_iterand_type], for_arg_tys.as_slice()].concat();
                            let loop_block = Block::new(
                                &loop_block_args.iter().map(|ty| (ty.clone(), location)).collect::<Vec<_>>()
                            );
                            let mut vars = vec![];
                            for i in 0..loop_block.argument_count() {
                                let val: Value = loop_block.argument(i).unwrap().into();
                                vars.push(val);
                            }
                            let some_val_1: Value = vars[1];
                            let some_val_2: Value = loop_block.append_operation(operation_parse(&context,
                                                                                               "%val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>",
                                                                                               None).unwrap()).result(0).unwrap().into();

                            let add_float_op_builder = OperationBuilder::new("cuda_tile.addf", Location::unknown(&context));
                            let rounding_attr = (Identifier::new(&context, "rounding_mode"),
                                                 Attribute::parse(&context, "#cuda_tile.rounding<nearest_even>").unwrap());
                            let add_float_op = add_float_op_builder
                                .add_attributes(&[rounding_attr])
                                .add_operands(&[some_val_1, some_val_2])
                                .add_results(&[some_val_1.r#type()])
                                .build()?;
                            let add_float_op_ref = loop_block.append_operation(add_float_op.into());
                            let some_val: Value = add_float_op_ref.result(0)?.into();
                            vars[1] = some_val;

                            let op = OperationBuilder::new("cuda_tile.continue", Location::unknown(&context))
                                .add_operands(&vars[1..])
                                .build().unwrap();
                            let _op_ref = loop_block.append_operation(op);
                            let region = Region::new();
                            region.append_block(loop_block);
                            region
                        }]).build().unwrap();
                    let for_loop_ref = func_block.append_operation(for_op);
                    assert!(for_loop_ref.verify());
                    let _result: Value = for_loop_ref.result(0).unwrap().into();

                    let init_val: Value = func_block.append_operation(
                        operation_parse(&context, "%initVal0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>", None).unwrap()
                    ).result(0).unwrap().into();
                    let loop_init_types = &[init_val.r#type()];
                    let loop_op = OperationBuilder::new("cuda_tile.loop", Location::unknown(&context))
                        .add_operands(&[init_val])
                        .add_results(&[Type::parse(&context, "!cuda_tile.tile<f32>").unwrap()])
                        .add_regions([{
                            let loop_block = Block::new(
                                &loop_init_types.iter().map(|ty| (ty.clone(), location)).collect::<Vec<_>>()
                            );
                            let some_val: Value = loop_block.append_operation(operation_parse(&context,
                                                                                               "%finalReturnValue = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<f32>",
                                                                                               None).unwrap()).result(0).unwrap().into();

                            let op = OperationBuilder::new("cuda_tile.break", Location::unknown(&context))
                                .add_operands(&[some_val])
                                .build().unwrap();
                            let _op_ref = loop_block.append_operation(op);
                            let region = Region::new();
                            region.append_block(loop_block);
                            region
                        }]).build().unwrap();
                    let _loop_ref = func_block.append_operation(loop_op);

                    // if
                    let ifval: Value = func_block.append_operation(
                        operation_parse(&context, "%ifval0 = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>", None).unwrap()
                    ).result(0).unwrap().into();
                    let condition = operation_parse(&context, "%condition = cuda_tile.constant <i1: 1> : !cuda_tile.tile<i1>", None)
                        .unwrap();
                    let condition_val: Value = func_block.append_operation(condition).result(0).unwrap().into();
                    let if_builder = OperationBuilder::new("cuda_tile.if", Location::unknown(&context));
                    let if_op = if_builder
                        .add_operands(&[condition_val])
                        .add_results(&[Type::parse(&context, "!cuda_tile.tile<f32>").unwrap(),
                            Type::parse(&context, "!cuda_tile.tile<i32>").unwrap()])
                        .add_regions([{
                            // then
                            let block = Block::new(&[]);
                            let region = Region::new();

                            // External values work fine in if statement.
                            let some_val_1: Value = random_val_1;
                            let some_val_2: Value = block.append_operation(operation_parse(&context,
                                                                                                 "%val = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<f32>",
                                                                                                 None).unwrap()).result(0).unwrap().into();

                            let add_float_op_builder = OperationBuilder::new("cuda_tile.addf", Location::unknown(&context));
                            let add_float_op = add_float_op_builder
                                .add_attributes(&[(Identifier::new(&context, "rounding_mode"),
                                                   Attribute::parse(&context, "#cuda_tile.rounding<nearest_even>").unwrap())])
                                .add_operands(&[some_val_1, some_val_2])
                                .add_results(&[some_val_1.r#type()])
                                .build()?;
                            let add_float_op_ref = block.append_operation(add_float_op.into());
                            let x_val: Value = add_float_op_ref.result(0)?.into();

                            let _yield_val = block.append_operation(
                                OperationBuilder::new("cuda_tile.yield", location)
                                    .add_operands(&[x_val, ifval])
                                    .build().unwrap()
                            );
                            region.append_block(block);
                            region
                        }, {
                            // else
                            let block = Block::new(&[]);
                            let region = Region::new();
                            let x_val: Value = block.append_operation(
                                operation_parse(&context, "%0 = cuda_tile.constant <f32: 2.0> : !cuda_tile.tile<f32>", None).unwrap()
                            ).result(0).unwrap().into();
                            let y_val: Value = block.append_operation(
                                operation_parse(&context, "%0 = cuda_tile.constant <i32: 84> : !cuda_tile.tile<i32>", None).unwrap()
                            ).result(0).unwrap().into();
                            let _yield_val = block.append_operation(
                                OperationBuilder::new("cuda_tile.yield", location)
                                    .add_operands(&[x_val, y_val])
                                    .build().unwrap()
                            );
                            region.append_block(block);
                            region
                        }]).build().unwrap();
                    let _if_op_ref = func_block.append_operation(if_op);

                    let print_builder = OperationBuilder::new("cuda_tile.print", Location::unknown(&context));
                    let print_op = print_builder
                        .add_attributes(&[
                            (
                                Identifier::new(&context, "str"),
                                StringAttribute::new(&context, "Hello %i.\n").into()
                            ),
                            (
                                Identifier::new(&context, "operandSegmentSizes"),
                                Attribute::parse(&context, "array<i32: 1, 0>").unwrap()
                            )
                        ])
                        .add_operands(&[ifval])
                        .build().unwrap();
                    let _print_op_ref = func_block.append_operation(print_op);

                    // Sum (Reduce Example)
                    let matrix_op = operation_parse(&context, "%0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<128x256xf32>", None).unwrap();
                    let matrix_op_ref = func_block.append_operation(matrix_op.into());
                    let matrix_ones_value: Value = matrix_op_ref.result(0)?.into();

                    let result_type = Type::parse(&context, "!cuda_tile.tile<128xf32>").unwrap();
                    let iter_operand_types = Type::parse(&context, "!cuda_tile.tile<f32>").unwrap();
                    let reduce_op = OperationBuilder::new("cuda_tile.reduce", location);
                    let reduce_op = reduce_op
                        .add_attributes(&[
                            parse_named_attr(&context, "dim", "1: i32"),
                            parse_named_attr(&context, "identities", "[0.0 : f32]"),
                        ])
                        .add_operands(&[matrix_ones_value])
                        .add_results(&[result_type])
                        .add_regions({
                            let block_arg_types = &[
                                iter_operand_types, // operand_i_current_iter
                                iter_operand_types // operand_i_prev_iter
                            ];
                            let block = Block::new(
                                &block_arg_types.iter().map(|ty| (ty.clone(), location)).collect::<Vec<_>>()
                            );
                            let mut block_vars = vec![];
                            for i in 0..block.argument_count() {
                                let val: Value = block.argument(i).unwrap().into();
                                block_vars.push(val);
                            }
                            let _yield_val = block.append_operation(
                                OperationBuilder::new("cuda_tile.yield", location)
                                    .add_operands(&[block_vars[0]])
                                    .build().unwrap()
                            );
                            let region = Region::new();
                            region.append_block(block);
                            [region]
                        });
                    let reduce_op_ref = func_block.append_operation(reduce_op.build()?);
                    assert!(reduce_op_ref.verify());

                    // Scan
                    let result_type = matrix_ones_value.r#type();
                    let iter_operand_types = Type::parse(&context, "!cuda_tile.tile<f32>").unwrap();
                    let scan_op = OperationBuilder::new("cuda_tile.scan", location);
                    let scan_op = scan_op
                        .add_attributes(&[
                            parse_named_attr(&context, "dim", "1: i32"),
                            parse_named_attr(&context, "reverse", "false"),
                            parse_named_attr(&context, "identities", "[1.0 : f32]"),
                        ])
                        .add_operands(&[matrix_ones_value])
                        .add_results(&[result_type])
                        .add_regions({
                            let block_arg_types = &[
                                iter_operand_types, // acc
                                iter_operand_types // elem
                            ];
                            let block = Block::new(
                                &block_arg_types.iter().map(|ty| (ty.clone(), location)).collect::<Vec<_>>()
                            );
                            let mut block_vars = vec![];
                            for i in 0..block.argument_count() {
                                let val: Value = block.argument(i).unwrap().into();
                                block_vars.push(val);
                            }
                            let _yield_val = block.append_operation(
                                OperationBuilder::new("cuda_tile.yield", location)
                                    .add_operands(&[block_vars[0]])
                                    .build().unwrap()
                            );
                            let region = Region::new();
                            region.append_block(block);
                            [region]
                        });

                    let scan_op_ref = func_block.append_operation(scan_op.build()?);
                    assert!(scan_op_ref.verify());

                    let matrix_op = operation_parse(&context, "%0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<128x256xf32>", None).unwrap();
                    let matrix_op_ref = func_block.append_operation(matrix_op.into());
                    let matrix_left: Value = matrix_op_ref.result(0)?.into();

                    let matrix_op = operation_parse(&context, "%0 = cuda_tile.constant <f32: 1.0> : !cuda_tile.tile<128x256xf32>", None).unwrap();
                    let matrix_op_ref = func_block.append_operation(matrix_op.into());
                    let matrix_right: Value = matrix_op_ref.result(0)?.into();

                    // Select
                    let conditional_op = operation_parse(&context, "%condition = cuda_tile.constant <i1: 1> : !cuda_tile.tile<128x256xi1>", None)
                        .unwrap();
                    let conditional: Value = func_block.append_operation(conditional_op).result(0).unwrap().into();

                    let select_op = OperationBuilder::new("cuda_tile.select", location);
                    let select_op = select_op.add_operands(&[
                        conditional, matrix_left, matrix_right
                    ])
                        .add_results(&[matrix_left.r#type()])
                        .build().unwrap();
                    let select_op_ref = func_block.append_operation(select_op);
                    assert!(select_op_ref.verify());

                    let ret_op_builder = cuda_tile::ReturnOperationBuilder::new(&context, location)
                        .operands(&[])
                        .build()
                        .into();
                    func_block.append_operation(ret_op_builder);

                    let region = Region::new();
                    region.append_block(func_block);
                    region
                }).build().into();

            module_block.append_operation(f);

            let region = Region::new();
            region.append_block(module_block);
            region
        }).sym_name(StringAttribute::new(&context, "my_kernels")).build();

    let op: Operation = module_op.into();
    assert!(op.verify());
    println!("{:?}", op);

    Ok(())
}
