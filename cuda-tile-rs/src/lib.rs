/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod util;

pub use util::*;

// include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
// use mlirCudaTileWriteBytecodeToBuffer;

use std::io::Write;
use std::{fs, io, slice};

use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::{self, StringRef};
use mlir_sys::MlirOperation;

melior::dialect! {
    name: "cuda_tile",
    files: [
        "cuda_tile/Dialect/CudaTile/IR/Ops.td",
        "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td",
        "cuda_tile/Dialect/CudaTile/IR/Types.td",
    ],
    include_directory_env_vars: ["CUDA_TILE_MLIR_INCLUDE_DIR"],
}

mod cuda_tile_c_bindings;
use cuda_tile_c_bindings as cuda_tile_c;

use crate::cuda_tile::ModuleOperation;
use crate::cuda_tile_c::mlirCudaTileWriteBytecodeToBuffer;

pub fn register_cuda_tile_dialects(registry: &DialectRegistry) {
    unsafe { cuda_tile_c::mlirCudaTileRegisterAllDialects(registry.to_raw()) };
}

pub fn register_cuda_tile_passes() {
    unsafe { cuda_tile_c::mlirCudaTileRegisterAllPasses() };
}

pub fn cuda_tile_write_bytecode_to_buffer<'c>(module_operation: &ModuleOperation) -> StringRef<'c> {
    let mlir_op: MlirOperation = module_operation.as_operation().to_raw();
    let string_ref = unsafe { mlirCudaTileWriteBytecodeToBuffer(mlir_op) };
    unsafe { StringRef::from_raw(string_ref) }
}

pub fn cuda_tile_write_bytecode(
    module_operation: &ModuleOperation,
    filename: &str,
) -> Result<(), io::Error> {
    let mlir_op: MlirOperation = module_operation.as_operation().to_raw();
    let string_ref = unsafe { mlirCudaTileWriteBytecodeToBuffer(mlir_op) };
    let bytes = unsafe { slice::from_raw_parts(string_ref.data as *mut u8, string_ref.length) };
    let mut file = fs::File::create(filename)?;
    file.write_all(bytes)?;
    file.flush()

    // This doesn't work.
    // let file = fs::File::create(path).unwrap();
    // let fd = file.into_raw_fd();
    // let result = unsafe { mlirCudaTileWriteBytecode(mlir_op, fd) };
    // if result {
    //     Ok(())
    // } else {
    //     Err(Error::new(ErrorKind::Unsupported, "Failed to write tile ir bytecode."))
    // }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, MutexGuard, Once};

    use crate::cuda_tile::{self};
    use crate::util::{attribute_parse, operation_parse, type_parse};
    use melior::Context;
    use melior::dialect::DialectRegistry;
    use melior::ir::RegionLike;
    use melior::ir::attribute::StringAttribute;
    use melior::ir::operation::{OperationBuilder, OperationLike};
    use melior::ir::{Attribute, Block, Identifier, Location, Module, Region};
    use melior::utility::{register_all_dialects, register_all_llvm_translations};

    static TEST_MUTEX: Mutex<()> = Mutex::new(());
    static REGISTER_GLOBALS: Once = Once::new();

    fn test_guard() -> MutexGuard<'static, ()> {
        // MLIR pass/translation registration mutates process-global state and has
        // proven flaky under libtest's default parallel execution.
        TEST_MUTEX.lock().expect("cuda-tile-rs test mutex poisoned")
    }

    pub fn load_all_dialects(context: &Context) {
        let registry = DialectRegistry::new();
        register_all_dialects(&registry);
        crate::register_cuda_tile_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
    }

    pub fn context_all() -> Context {
        let context = Context::new();
        load_all_dialects(&context);
        REGISTER_GLOBALS.call_once(|| {
            register_all_llvm_translations(&context);
            crate::register_cuda_tile_passes();
        });
        context.attach_diagnostic_handler(|diagnostic| {
            eprintln!("{}", diagnostic);
            true
        });
        context
    }

    #[test]
    fn build_cuda_tile_module() {
        let _guard = test_guard();
        println!("Building CUDA Tile module");
        let context = context_all();
        let location = Location::unknown(&context);
        println!("Location: {:?}", location);
        let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location)
            .body({
                let module_block = Block::new(&[]);
                let region = Region::new();
                region.append_block(module_block);
                region
            })
            .sym_name(StringAttribute::new(&context, "testing"))
            .build();
        println!("module_op: {:?}", module_op.as_operation().to_string());
        assert!(module_op.as_operation().verify());
    }

    #[test]
    fn parse_cuda_tile_module() {
        let _guard = test_guard();
        const module_str: &'static str = r#"
            cuda_tile.module @hello_world_module {
            }
        "#;
        let context = context_all();
        let module: Module = Module::parse(&context, module_str).expect("Module parse failed.");
        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_parse_helpers() {
        let _guard = test_guard();
        let context = context_all();
        let op = crate::util::operation_parse(&context, "%x = gpu.thread_id x", None)
            .unwrap_or_else(|| panic!("failed."));
        assert!(op.verify());
        println!("{}", op.to_string());

        let cuda_tile_type = type_parse(
            &context,
            "!cuda_tile.tensor_view<64x32xi32, strides=[32,1]>",
        )
        .unwrap();
        println!("{:#?}", cuda_tile_type);

        let mlir_attr = attribute_parse(&context, "#cuda_tile.signedness<signed>").unwrap();
        println!("{:#?}", mlir_attr);

        let mlir_attr = attribute_parse(&context, "#cuda_tile.signedness<signed>").unwrap();
        println!("{:#?}", mlir_attr);

        let mlir_attr =
            attribute_parse(&context, "#cuda_tile.comparison_predicate<equal>").unwrap();
        println!("{:#?}", mlir_attr);

        let mlir_attr =
            attribute_parse(&context, "#cuda_tile.comparison_predicate<equal>").unwrap();
        println!("{:#?}", mlir_attr);

        let matrix_op = operation_parse(
            &context,
            "%0 = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<128x256xf32>",
            None,
        )
        .unwrap();
        assert!(matrix_op.verify());

        // Unclear why this fails to parse.
        // let op = operation_parse(
        //     &context,
        //     r#"%token = cuda_tile.print_tko "Hello world\n" -> token"#,
        //     None,
        // )

        // .unwrap_or_else(|| panic!("failed."));
        // assert!(op.verify());
        // println!("cuda_tile.print_tko: {}", op.to_string());
        // for (id, attr) in op.attributes() {
        //     println!("{}={}", id.as_string_ref().as_str().unwrap(), attr.to_string());
        // }

        // let mlir_attr = attribute_parse(
        //     &context,
        //     "#cuda_tile.optimization_hints<sm_120 = {num_cta_in_cga = 2, allow_tma=true, latency=1}>",
        // )
        // .unwrap();
        // println!("{:#?}", mlir_attr);
        // let mlir_attr = attribute_parse(&context, "#cuda_tile.optimization_hints<>").unwrap();
        // println!("{:#?}", mlir_attr);

        let mlir_attr = attribute_parse(
            &context,
            format!("array<{}: {}>", "f32", "0.0,0.0").as_str(),
        )
        .unwrap();
        println!("{:#?}", mlir_attr);

        let mlir_attr = attribute_parse(&context, "unit").unwrap();
        println!("{:#?}", mlir_attr);
    }

    #[test]
    fn build_print() {
        let _guard = test_guard();
        let context = context_all();
        // print_tko is not available in 13.1.
        let print_builder = OperationBuilder::new("cuda_tile.print", Location::unknown(&context));
        let print_op = print_builder
            .add_attributes(&[
                (
                    Identifier::new(&context, "str"),
                    StringAttribute::new(&context, "Hello.\n").into(),
                ),
                (
                    Identifier::new(&context, "operandSegmentSizes"),
                    Attribute::parse(&context, "array<i32: 0, 0>").unwrap(),
                ),
            ])
            .add_operands(&[])
            .build()
            .unwrap();
        assert!(print_op.verify());
    }

    // TODO (hme): Unclear why this is failing.
    // #[test]
    // fn build_cuda_tile_entry() {
    //     let context = context_all();
    //     let location = Location::unknown(&context);
    //     let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location)
    //         .body({
    //             let module_block = Block::new(&[]);

    //             let argument_types = vec![];
    //             let result_types = vec![];
    //             let function_type = FunctionType::new(&context, &argument_types, &result_types);
    //             let name_attr_v = StringAttribute::new(&context, "some_func");
    //             let type_attr = TypeAttribute::new(function_type.into());
    //             let f = cuda_tile::EntryOperationBuilder::new(&context, location)
    //                 .sym_name(name_attr_v)
    //                 .function_type(type_attr)
    //                 .body({
    //                     let func_block = Block::new(&[]);

    //                     let const_val: ir::Value = func_block
    //                         .append_operation(
    //                             operation_parse(
    //                                 &context,
    //                                 "%val = cuda_tile.constant <i32: 0> : !cuda_tile.tile<i32>",
    //                                 None,
    //                             )
    //                             .unwrap(),
    //                         )
    //                         .result(0)
    //                         .unwrap()
    //                         .into();

    //                     let print_builder =
    //                         OperationBuilder::new("cuda_tile.print", Location::unknown(&context));
    //                     let print_op = print_builder
    //                         .add_attributes(&[
    //                             (
    //                                 Identifier::new(&context, "str"),
    //                                 StringAttribute::new(&context, "Hello %i.\n").into(),
    //                             ),
    //                             (
    //                                 Identifier::new(&context, "operandSegmentSizes"),
    //                                 Attribute::parse(&context, "array<i32: 1, 0>").unwrap(),
    //                             ),
    //                         ])
    //                         .add_operands(&[const_val])
    //                         .build()
    //                         .unwrap();
    //                     let _print_op_ref = func_block.append_operation(print_op);

    //                     let ret_op_builder =
    //                         cuda_tile::ReturnOperationBuilder::new(&context, location)
    //                             .operands(&[])
    //                             .build()
    //                             .into();
    //                     func_block.append_operation(ret_op_builder);

    //                     let region = Region::new();
    //                     region.append_block(func_block);
    //                     region
    //                 })
    //                 .build()
    //                 .into();

    //             module_block.append_operation(f);

    //             let region = Region::new();
    //             region.append_block(module_block);
    //             region
    //         })
    //         .sym_name(StringAttribute::new(&context, "my_kernels"))
    //         .build();

    //     println!("{:?}", module_op.as_operation());
    //     assert!(module_op.as_operation().verify());
    // }

    // TODO (hme): Unclear why this is failing.
    // #[test]
    // fn parse_cuda_tile_hello_world() {
    //     const HELLO_TILE_BLOCK_MLIR: &'static str = r#"cuda_tile.module @hello_world_module {
    //             entry @hello_world_kernel() {
    //                 %0 = cuda_tile.constant <f32: 0.0> : !cuda_tile.tile<128x256xf32>
    //                 cuda_tile.return
    //             }
    //         }"#;
    //     let context = context_all();
    //     let module: Module =
    //         Module::parse(&context, HELLO_TILE_BLOCK_MLIR).expect("Module parse failed.");
    //     assert!(module.as_operation().verify());
    // }
}
