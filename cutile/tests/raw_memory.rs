/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile;
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod raw_memory_module {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn raw_offset_load_store_kernel(source: *mut u8, destination: *mut f32) {
        let offsets: Tile<i32, { [128] }> = iota(const_shape![128]);
        let byte_offsets: Tile<i32, { [128] }> = offsets * constant(4i32, const_shape![128]);
        let upper_bound: Tile<i32, { [128] }> = constant(127i32, const_shape![128]);
        let mask: Tile<bool, { [128] }> = lt_tile(offsets, upper_bound);

        let input_token = new_token_unordered();
        let values: Tile<f32, { [128] }> = load_offset_as(
            source,
            byte_offsets,
            ordering::Acquire,
            Some(scope::Device),
            Some(mask),
            Some(0.0f32),
            Some(input_token),
            Latency::<4>,
        );

        let _store_token: Token = store_offset(
            destination,
            offsets,
            values,
            ordering::Release,
            Some(scope::System),
            Some(mask),
            Some(input_token),
            Latency::<2>,
        );
    }

    #[cutile::entry()]
    unsafe fn raw_offset_weak_load_kernel(source: *mut u8) {
        let offsets: Tile<i32, { [64] }> = iota(const_shape![64]);
        let _values: Tile<u8, { [64] }> = load_offset(
            source,
            offsets,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<8>,
        );
    }
}

use raw_memory_module::__module_ast_self;

#[test]
fn raw_offset_memory_lowers_to_pointer_tile_ops() {
    common::with_test_stack(|| {
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "raw_memory_module",
            "raw_offset_load_store_kernel",
            &[],
            &[],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("failed to compile raw-offset memory kernel");

        assert!(mlir.matches(" = offset ").count() >= 2, "{mlir}");
        assert!(mlir.contains("ptr_to_ptr"), "{mlir}");
        assert!(mlir.contains("load_ptr_tko acquire device"), "{mlir}");
        assert!(mlir.contains("store_ptr_tko release sys"), "{mlir}");
        assert!(mlir.contains("latency = 4"), "{mlir}");
        assert!(mlir.contains("latency = 2"), "{mlir}");
        assert!(
            mlir.contains("tile<128xptr<f32>>, tile<128xi1>, tile<128xf32>"),
            "{mlir}"
        );
        assert!(mlir.contains("token="), "{mlir}");
        assert!(!mlir.contains("make_gather_scatter_view"), "{mlir}");

        let weak_mlir = common::compile_to_ir(
            __module_ast_self,
            "raw_memory_module",
            "raw_offset_weak_load_kernel",
            &[],
            &[],
            &[],
            &[],
            None,
            &CompileOptions::default(),
        )
        .expect("failed to compile weak raw-offset load kernel");
        assert!(weak_mlir.contains("load_ptr_tko weak"), "{weak_mlir}");
        assert!(weak_mlir.contains("latency = 8"), "{weak_mlir}");
    });
}
