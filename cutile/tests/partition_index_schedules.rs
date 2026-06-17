/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cutile;
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod partition_index_schedules_module {
    use cutile::core::*;

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn static_persistent_gemm_scheduled<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [M, K] }>,
        y: &Tensor<T, { [K, N] }>,
    ) {
        let k_tiles: i32 = K / BK;

        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);

        for index in z.iter_indices() {
            let (bid_m, bid_n) = index.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in 0i32..k_tiles {
                let tile_x = part_x.load([bid_m, k_tile]);
                let tile_y = part_y.load([k_tile, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, index);
        }
    }

    #[cutile::entry(
        unchecked_accesses = false,
        preconditions = (
            dim(z, 0) == dim(x, 0),
            dim(z, 1) == dim(y, 1),
        )
    )]
    unsafe fn static_persistent_gemm_scheduled_with_preconditions<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const M: i32,
        const N: i32,
        const K: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [M, K] }>,
        y: &Tensor<T, { [K, N] }>,
    ) {
        let k_tiles: i32 = K / BK;

        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);

        for index in z.iter_indices() {
            let (bid_m, bid_n) = index.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in 0i32..k_tiles {
                let tile_x = part_x.load([bid_m, k_tile]);
                let tile_y = part_y.load([k_tile, bid_n]);
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn dynamic_persistent_gemm_bounded<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
        y: &Tensor<T, { [-1, -1] }>,
    ) {
        let m = num_tiles(&z, 0);
        let n = num_tiles(&z, 1);
        let k = Dim::new(x.shape()[1] / BK);

        let part_x = x.partition(const_shape![BM, BK]).with_bounds((m, k));
        let part_y = y.partition(const_shape![BK, BN]).with_bounds((k, n));

        for index in z.iter_indices() {
            let (bid_m, bid_n) = index.components();
            let mut tile_z: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            for k_tile in k {
                let tile_x = part_x.load(coord((bid_m, k_tile)));
                let tile_y = part_y.load(coord((k_tile, bid_n)));
                tile_z = mma(tile_x, tile_y, tile_z);
            }
            z.store(tile_z, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn dynamic_bounded_rejects_swapped_axes<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, -1] }>,
    ) {
        let m = num_tiles(&z, 0);
        let n = num_tiles(&z, 1);
        let part_x = x.partition(const_shape![BM, BK]).with_bounds((m, n));
        for index in z.iter_indices() {
            let (bid_m, bid_n) = index.components();
            let _tile_x = part_x.load(coord((bid_n, bid_m)));
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rejects_unbranded_partition_index<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
    ) {
        let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
        let index: PartitionIndex<{ [BM, BN] }> =
            swizzle_partition_index_2d::<{ [BM, BN] }, MAP_SHAPE>(0i32, 1i32, 1i32);
        z.store(tile, index);
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rejects_foreign_partition_index<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        a: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        mut b: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
    ) {
        for index in a.iter_indices() {
            let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            b.store(tile, index);
        }
    }
}

#[allow(dead_code)]
async fn mapped_partition_launcher_accepts_host_map() {
    use cutile::api;
    use cutile::prelude::*;

    let z = api::zeros::<f32>(&[256, 256]).await.unwrap();
    let x = Arc::new(api::zeros::<f32>(&[256, 128]).await.unwrap());
    let y = Arc::new(api::zeros::<f32>(&[128, 256]).await.unwrap());
    let _op = unsafe {
        partition_index_schedules_module::static_persistent_gemm_scheduled(
            z.partition([64, 64]).map([4, 1], 4),
            x,
            y,
        )
    };
}

use partition_index_schedules_module::__module_ast_self;

fn compile_static_persistent_gemm_kernel(function_name: &str) -> Result<String, String> {
    let (generics, stride_args): (Vec<String>, Vec<(&str, &[i32])>) = match function_name {
        "static_persistent_gemm_scheduled" => (
            vec![
                "f32".to_string(),
                "64".to_string(),
                "64".to_string(),
                "32".to_string(),
                "256".to_string(),
                "256".to_string(),
                "128".to_string(),
                "4".to_string(),
                "1".to_string(),
            ],
            vec![("z", &[256, 1]), ("x", &[128, 1]), ("y", &[256, 1])],
        ),
        "static_persistent_gemm_scheduled_with_preconditions" => (
            vec![
                "f32".to_string(),
                "64".to_string(),
                "64".to_string(),
                "32".to_string(),
                "256".to_string(),
                "256".to_string(),
                "128".to_string(),
                "4".to_string(),
                "1".to_string(),
            ],
            vec![("z", &[256, 1]), ("x", &[128, 1]), ("y", &[256, 1])],
        ),
        "dynamic_persistent_gemm_bounded" => (
            vec![
                "f32".to_string(),
                "64".to_string(),
                "64".to_string(),
                "32".to_string(),
                "4".to_string(),
                "1".to_string(),
            ],
            vec![("z", &[256, 1]), ("x", &[128, 1]), ("y", &[256, 1])],
        ),
        "dynamic_bounded_rejects_swapped_axes" => (
            vec![
                "f32".to_string(),
                "64".to_string(),
                "64".to_string(),
                "32".to_string(),
                "4".to_string(),
                "1".to_string(),
            ],
            vec![("z", &[256, 1]), ("x", &[128, 1])],
        ),
        "rejects_unbranded_partition_index" => (
            vec![
                "f32".to_string(),
                "64".to_string(),
                "64".to_string(),
                "4".to_string(),
                "1".to_string(),
            ],
            vec![("z", &[256, 1])],
        ),
        "rejects_foreign_partition_index" => (
            vec![
                "f32".to_string(),
                "64".to_string(),
                "64".to_string(),
                "4".to_string(),
                "1".to_string(),
            ],
            vec![("a", &[256, 1]), ("b", &[256, 1])],
        ),
        other => panic!("unexpected test kernel `{other}`"),
    };
    common::compile_to_ir(
        __module_ast_self,
        "partition_index_schedules_module",
        function_name,
        &generics,
        &stride_args,
        &[],
        &[],
        Some((4, 1, 1)),
        &CompileOptions::default(),
    )
    .map_err(|err| err.to_string())
}

#[test]
fn mapped_partition_indices_lower_to_persistent_loop() {
    common::with_test_stack(|| {
        let mlir = compile_static_persistent_gemm_kernel("static_persistent_gemm_scheduled")
            .expect("Failed to compile");
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected mapped partition index-space query in MLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("get_tile_block_id"),
            "expected mapped partition persistent loop to use tile-block id:\n{mlir}"
        );
        assert!(
            mlir.contains("get_num_tile_blocks"),
            "expected mapped partition persistent loop to use tile-block grid size:\n{mlir}"
        );
        assert!(
            !mlir.contains("swizzle partition map requires"),
            "swizzle_partition_index_2d is unsafe/unchecked and should not emit runtime schedule asserts:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "expected persistent GEMM output store in MLIR:\n{mlir}"
        );
    });
}

fn compile_dynamic_persistent_gemm_bounded_with_map(map_shape: [i32; 2]) -> Result<String, String> {
    let generics = vec![
        "f32".to_string(),
        "64".to_string(),
        "64".to_string(),
        "32".to_string(),
        map_shape[0].to_string(),
        map_shape[1].to_string(),
    ];
    let stride_args = vec![
        ("z", &[256, 1][..]),
        ("x", &[128, 1][..]),
        ("y", &[256, 1][..]),
    ];
    common::compile_to_ir(
        __module_ast_self,
        "partition_index_schedules_module",
        "dynamic_persistent_gemm_bounded",
        &generics,
        &stride_args,
        &[],
        &[],
        Some((4, 1, 1)),
        &CompileOptions::default(),
    )
    .map_err(|err| err.to_string())
}

#[test]
fn mapped_linear_map_lowers_to_linear_partition_index_math() {
    common::with_test_stack(|| {
        let mlir = compile_dynamic_persistent_gemm_bounded_with_map([1, 1])
            .expect("Failed to compile linear mapped persistent GEMM");
        assert!(
            !mlir.contains("swizzle partition map requires"),
            "linear map should not emit swizzle schedule asserts:\n{mlir}"
        );
        assert!(
            !mlir.contains("mini "),
            "linear map should not use the generic grouped/swizzled min path:\n{mlir}"
        );
    });
}

#[test]
fn mapped_persistent_mlir_is_stable_across_repeated_compiles() {
    common::with_test_stack(|| {
        let first = compile_static_persistent_gemm_kernel("dynamic_persistent_gemm_bounded")
            .expect("Failed to compile first MLIR dump");
        for _ in 0..8 {
            let next = compile_static_persistent_gemm_kernel("dynamic_persistent_gemm_bounded")
                .expect("Failed to compile repeated MLIR dump");
            assert_eq!(
                first, next,
                "repeated compiles of the same mapped persistent kernel should emit byte-identical MLIR"
            );
        }
    });
}

#[test]
fn mapped_partition_indices_keep_foreign_partition_load_checks_without_preconditions() {
    common::with_test_stack(|| {
        let mlir = compile_static_persistent_gemm_kernel("static_persistent_gemm_scheduled")
            .expect("Failed to compile");
        assert!(
            mlir.contains("partition access out of bounds: dim 0, block index >= ceil(256/64)"),
            "expected x-load bid_m check without dim preconditions:\n{mlir}"
        );
        assert!(
            mlir.contains("partition access out of bounds: dim 1, block index >= ceil(256/64)"),
            "expected y-load bid_n check without dim preconditions:\n{mlir}"
        );
    });
}

#[test]
fn mapped_partition_preconditions_discharge_foreign_partition_load_checks() {
    common::with_test_stack(|| {
        let mlir = compile_static_persistent_gemm_kernel(
            "static_persistent_gemm_scheduled_with_preconditions",
        )
        .expect("Failed to compile");
        assert!(
            !mlir.contains("partition access out of bounds"),
            "explicit dim preconditions should discharge immutable partition access checks:\n{mlir}"
        );
    });
}

#[test]
fn bounded_partition_dims_discharge_dynamic_partition_load_checks() {
    common::with_test_stack(|| {
        let mlir = compile_static_persistent_gemm_kernel("dynamic_persistent_gemm_bounded")
            .expect("Failed to compile");
        assert!(
            !mlir.contains("partition access out of bounds"),
            "bounded partition coordinates should discharge immutable partition access checks:\n{mlir}"
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "dynamic bounded GEMM should keep dynamic partition-grid queries:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "dynamic bounded GEMM should still store through mapped output partition:\n{mlir}"
        );
    });
}

#[test]
fn bounded_partition_rejects_swapped_coordinate_axes() {
    common::with_test_stack(|| {
        let err = compile_static_persistent_gemm_kernel("dynamic_bounded_rejects_swapped_axes")
            .expect_err("swapped bounded partition coordinate axes should fail");
        assert!(
            err.contains(
                "bounded partition coordinate axis 0 was produced by a different dimension"
            ),
            "unexpected error for swapped bounded coordinate axes:\n{err}"
        );
    });
}

#[test]
fn mapped_partition_store_rejects_unbranded_index() {
    common::with_test_stack(|| {
        let err = compile_static_persistent_gemm_kernel("rejects_unbranded_partition_index")
            .expect_err("unbranded partition index store should fail");
        assert!(
            err.contains("requires an index produced by this partition's iter_indices() iterator"),
            "unexpected error for unbranded partition index:\n{err}"
        );
    });
}

#[test]
fn mapped_partition_store_rejects_foreign_index() {
    common::with_test_stack(|| {
        let err = compile_static_persistent_gemm_kernel("rejects_foreign_partition_index")
            .expect_err("foreign partition index store should fail");
        assert!(
            err.contains("index was produced by a different mapped partition"),
            "unexpected error for foreign partition index:\n{err}"
        );
    });
}
