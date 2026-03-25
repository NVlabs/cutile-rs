/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT compiler that translates Rust DSL modules into CUDA Tile MLIR and compiles them to GPU cubins.

#![allow(non_snake_case)]
extern crate core;

pub use cuda_tile_rs::cuda_tile;
pub use cuda_tile_rs::cuda_tile_write_bytecode_to_buffer;
use cuda_tile_rs::register_cuda_tile_dialects;
use melior::{
    dialect::DialectRegistry,
    utility::{register_all_dialects, register_all_llvm_translations},
    Context,
};

pub mod ast;
mod bounds;
#[cfg(feature = "cuda")]
pub mod cuda_tile_runtime_utils;
pub mod error;
pub mod generics;
mod kernel_entry_generator;
pub mod syn_utils;
pub mod train_map;
pub mod types;
pub mod validator;

pub mod compiler;
pub use compiler::utils;

/// Registers all standard and CUDA Tile MLIR dialects into the given context.
pub fn load_all_dialects(context: &Context) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    register_cuda_tile_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
}

/// Creates a fully configured MLIR context with all dialects, LLVM translations, and diagnostics.
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

#[cfg(test)]
mod tests {
    use crate::{context_all, cuda_tile};
    use melior::ir::attribute::StringAttribute;
    use melior::ir::operation::OperationLike;
    use melior::ir::{Block, Location, Region, RegionLike, Type};

    #[test]
    fn build_cuda_tile_module() {
        let context = context_all();
        let location = Location::unknown(&context);
        let module_op = cuda_tile::ModuleOperationBuilder::new(&context, location)
            .body({
                let module_block = Block::new(&[]);
                let region = Region::new();
                region.append_block(module_block);
                region
            })
            .sym_name(StringAttribute::new(&context, "testing"))
            .build();
        assert!(module_op.as_operation().verify());
    }

    #[test]
    fn parse_tensor_type() {
        let context = context_all();
        let _location = Location::unknown(&context);
        let cuda_tile_type = Type::parse(&context, "!cuda_tile.tile<!cuda_tile.ptr<f32>>");
        assert!(cuda_tile_type.is_some());
    }

    #[test]
    fn parse_partition_view_type() {
        let context = context_all();
        let _location = Location::unknown(&context);
        let cuda_tile_type = Type::parse(&context, "!cuda_tile.partition_view<tile=(1024x1x32), !cuda_tile.tensor_view<?x?x?xf32, strides=[?,?,?]>>");

        assert!(cuda_tile_type.is_some());
    }
}
