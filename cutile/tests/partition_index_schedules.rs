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

    /// Rank-3 bounded loads in the splitk-merge shape: axes 0/2 brand-match
    /// `num_tiles` bounds of the mapped output, axis 1 is a constant checked
    /// against the axis's static tile grid.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rank3_bounded_load<
        T: ElementType,
        const BS: i32,
        const BD: i32,
        const S1: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        mut z: MappedPartitionMut<T, { [1, BS, BD] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, S1, -1] }>,
    ) {
        let d0 = num_tiles(&z, 0);
        let d2 = num_tiles(&z, 2);
        let part_x = x
            .partition(const_shape![1, BS, BD])
            .with_bounds((d0, Dim::new(1), d2));
        for index in z.iter_indices() {
            let [i0, _i1, i2] = index.coords();
            let tile_x = part_x.load_pipelined::<4>(coord((i0, 0i32, i2)));
            z.store(tile_x, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rank3_bounded_rejects_swapped_axes<
        T: ElementType,
        const BS: i32,
        const BD: i32,
        const S1: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        z: MappedPartitionMut<T, { [1, BS, BD] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, S1, -1] }>,
    ) {
        let d0 = num_tiles(&z, 0);
        let d2 = num_tiles(&z, 2);
        let part_x = x
            .partition(const_shape![1, BS, BD])
            .with_bounds((d0, Dim::new(1), d2));
        for index in z.iter_indices() {
            let [i0, _i1, i2] = index.coords();
            let _tile_x = part_x.load(coord((i2, 0i32, i0)));
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rank3_bounded_rejects_out_of_grid_constant<
        T: ElementType,
        const BS: i32,
        const BD: i32,
        const S1: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        z: MappedPartitionMut<T, { [1, BS, BD] }, MAP_SHAPE>,
        x: &Tensor<T, { [-1, S1, -1] }>,
    ) {
        let d0 = num_tiles(&z, 0);
        let d2 = num_tiles(&z, 2);
        let part_x = x
            .partition(const_shape![1, BS, BD])
            .with_bounds((d0, Dim::new(1), d2));
        for index in z.iter_indices() {
            let [i0, _i1, i2] = index.coords();
            let _tile_x = part_x.load(coord((i0, 9i32, i2)));
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

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rank3_scheduled_store<
        T: ElementType,
        const BS: i32,
        const BD: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        mut z: MappedPartitionMut<T, { [1, BS, BD] }, MAP_SHAPE>,
    ) {
        for index in z.iter_indices() {
            let (_bid_b, _bid_s, _bid_d) = index.components();
            let tile: Tile<T, { [1, BS, BD] }> = constant(T::ZERO, const_shape![1, BS, BD]);
            z.store(tile, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn rank1_scheduled_store<T: ElementType, const BN: i32, const MAP_SHAPE: [i32; 1]>(
        mut z: MappedPartitionMut<T, { [BN] }, MAP_SHAPE>,
    ) {
        for index in z.iter_indices() {
            let tile: Tile<T, { [BN] }> = constant(T::ZERO, const_shape![BN]);
            z.store(tile, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn shared_map_dual_store<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        mut residual_out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
    ) {
        for index in out.iter_indices_with(&residual_out) {
            let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            out.store(tile, index);
            residual_out.store(tile, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn shared_map_rejects_third_partition<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        a: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        b: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        mut c: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
    ) {
        for index in a.iter_indices_with(&b) {
            let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            c.store(tile, index);
        }
    }

    /// Attention-shaped check-hoisting: a KV-style inner loop whose loads
    /// index [loop-invariant scalar, induction var, literal]. The bounds
    /// checks must move to the loop preheader, keeping the hot body free of
    /// asserts (grout fmha regression, 2026-07-03).
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn hoisted_checks_kv_loop<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const D: i32,
        const MAP_SHAPE: [i32; 3],
    >(
        mut out: MappedPartitionMut<T, { [BM, 1, D] }, MAP_SHAPE>,
        k: &Tensor<T, { [-1, -1, D] }>,
        kv_tiles: i32,
    ) {
        let part_k = k.partition(const_shape![1, BN, D]);
        for index in out.iter_indices() {
            let (_q_idx, head_idx, _d0) = index.components();
            let mut acc: Tile<T, { [BM, 1, D] }> = constant(T::ZERO, const_shape![BM, 1, D]);
            for j in 0i32..kv_tiles {
                let k_tile = part_k.load_pipelined::<2>([head_idx, j, 0i32]);
                let k_tile: Tile<T, { [BM, 1, D] }> = k_tile.reshape(const_shape![BM, 1, D]);
                acc = acc + k_tile;
            }
            out.store(acc, index);
        }
    }

    /// Multi-level hoisting: the load's checks are invariant across both
    /// loops; the inner loop is statically non-empty (0..2), so the checks
    /// cross it and land in the function body, before the outer `for`.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn nested_invariant_checks<T: ElementType, const BN: i32, const D: i32>(
        z: &mut Tensor<T, { [BN, D] }>,
        x: &Tensor<T, { [-1, -1] }>,
        head: i32,
        n: i32,
    ) {
        let part = x.partition(const_shape![BN, D]);
        let mut acc: Tile<T, { [BN, D] }> = constant(T::ZERO, const_shape![BN, D]);
        for _a in 0i32..n {
            for b in 0i32..2i32 {
                let t = part.load([head, b]);
                acc = acc + t;
            }
        }
        z.store(acc);
    }

    /// Affine hoisting: the row index is `2*j + 1`, monotone in the
    /// unit-step induction variable, so the check substitutes the strongest
    /// instance `2*(n-1) + 1` in the preheader.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn affine_index_checks<T: ElementType, const BN: i32, const D: i32>(
        z: &mut Tensor<T, { [BN, D] }>,
        x: &Tensor<T, { [-1, -1] }>,
        n: i32,
    ) {
        let part = x.partition(const_shape![BN, D]);
        let mut acc: Tile<T, { [BN, D] }> = constant(T::ZERO, const_shape![BN, D]);
        for j in 0i32..n {
            let t = part.load([2i32 * j + 1i32, 0i32]);
            acc = acc + t;
        }
        z.store(acc);
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn subrange_store<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        start_tile: i32,
        n_tiles: i32,
    ) {
        for index in z.iter_indices_within([(start_tile, n_tiles), (0i32, -1i32)]) {
            let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            z.store(tile, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn subrange_rejects_negative_start<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut z: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
    ) {
        for index in z.iter_indices_within([(-1i32, 2i32), (0i32, -1i32)]) {
            let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            z.store(tile, index);
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn subrange_shared_dual_store<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        mut residual_out: MappedPartitionMut<T, { [BM, BN] }, MAP_SHAPE>,
        start_tile: i32,
        n_tiles: i32,
    ) {
        for index in
            out.iter_indices_within_with([(start_tile, n_tiles), (0i32, -1i32)], &residual_out)
        {
            let tile: Tile<T, { [BM, BN] }> = constant(T::ZERO, const_shape![BM, BN]);
            out.store(tile, index);
            residual_out.store(tile, index);
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

#[allow(dead_code)]
async fn mapped_partition_launcher_accepts_rank3_and_shared_host_maps() {
    use cutile::api;
    use cutile::prelude::*;

    let z = api::zeros::<f32>(&[8, 256, 256]).await.unwrap();
    let _op = unsafe {
        partition_index_schedules_module::rank3_scheduled_store(
            z.partition([1, 64, 64]).map([1, 1, 1], 4),
        )
    };

    let out = api::zeros::<f32>(&[256, 256]).await.unwrap();
    let residual_out = api::zeros::<f32>(&[256, 256]).await.unwrap();
    let _op = unsafe {
        partition_index_schedules_module::shared_map_dual_store(
            out.partition([64, 64]).map([1, 1], 4),
            residual_out.partition([64, 64]).map([1, 1], 4),
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

fn compile_rank3_bounded_kernel(
    function_name: &str,
) -> Result<(String, cutile::compile_api::CheckPlacementCounts), String> {
    let generics = vec![
        "f32".to_string(),
        "64".to_string(),
        "64".to_string(),
        "64".to_string(),
        "1".to_string(),
        "1".to_string(),
        "1".to_string(),
    ];
    let stride_args = vec![("z", &[4096, 64, 1][..]), ("x", &[4096, 64, 1][..])];
    common::compile_to_ir_with_counts(
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
fn rank3_bounded_load_discharges_all_checks() {
    common::with_test_stack(|| {
        let (mlir, counts) =
            compile_rank3_bounded_kernel("rank3_bounded_load").expect("Failed to compile");
        assert!(
            !mlir.contains("partition access out of bounds"),
            "rank-3 bounded coordinates should discharge every load check:\n{mlir}"
        );
        assert_eq!(
            counts.in_place, 0,
            "rank-3 bounded loads must not leave in-place checks:\n{mlir}"
        );
        assert_eq!(
            counts.hoisted, 0,
            "rank-3 bounded loads must not need hoisted checks:\n{mlir}"
        );
        assert_eq!(
            counts.discharged, 3,
            "each coordinate axis should discharge at compile time:\n{mlir}"
        );
        assert!(
            mlir.contains("latency = 4"),
            "bounded pipelined load should keep its latency hint:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "rank-3 bounded kernel should still store through the mapped partition:\n{mlir}"
        );
    });
}

#[test]
fn rank3_bounded_rejects_swapped_coordinate_axes() {
    common::with_test_stack(|| {
        let err = compile_rank3_bounded_kernel("rank3_bounded_rejects_swapped_axes")
            .map(|(mlir, _)| mlir)
            .expect_err("swapped rank-3 bounded coordinate axes should fail");
        assert!(
            err.contains(
                "bounded partition coordinate axis 0 was produced by a different dimension"
            ),
            "unexpected error for swapped rank-3 bounded coordinate axes:\n{err}"
        );
    });
}

#[test]
fn rank3_bounded_rejects_out_of_grid_constant() {
    common::with_test_stack(|| {
        let err = compile_rank3_bounded_kernel("rank3_bounded_rejects_out_of_grid_constant")
            .map(|(mlir, _)| mlir)
            .expect_err("out-of-grid constant coordinate should fail at compile time");
        assert!(
            err.contains("is not within the 1-tile grid"),
            "unexpected error for out-of-grid constant coordinate:\n{err}"
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

fn compile_rank3_scheduled_store(map_shape: [i32; 3]) -> Result<String, String> {
    let generics = vec![
        "f32".to_string(),
        "64".to_string(),
        "64".to_string(),
        map_shape[0].to_string(),
        map_shape[1].to_string(),
        map_shape[2].to_string(),
    ];
    let stride_args = vec![("z", &[4096, 64, 1][..])];
    common::compile_to_ir(
        __module_ast_self,
        "partition_index_schedules_module",
        "rank3_scheduled_store",
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
fn rank3_mapped_partition_lowers_to_persistent_loop() {
    common::with_test_stack(|| {
        let mlir =
            compile_rank3_scheduled_store([1, 1, 1]).expect("Failed to compile rank-3 kernel");
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected rank-3 index-space query in MLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("get_tile_block_id"),
            "expected rank-3 persistent loop to use tile-block id:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "expected rank-3 mapped store in MLIR:\n{mlir}"
        );
        assert!(
            !mlir.contains("mini "),
            "linear rank-3 map should not use the grouped/swizzled min path:\n{mlir}"
        );
    });
}

#[test]
fn rank3_mapped_partition_supports_grouped_inner_axes() {
    common::with_test_stack(|| {
        let mlir = compile_rank3_scheduled_store([1, 4, 1])
            .expect("Failed to compile grouped rank-3 kernel");
        assert!(
            mlir.contains("mini "),
            "grouped rank-3 map should lower through the grouped swizzle path:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "expected grouped rank-3 mapped store in MLIR:\n{mlir}"
        );
    });
}

#[test]
fn rank3_mapped_partition_rejects_grouped_leading_axis() {
    common::with_test_stack(|| {
        let err = compile_rank3_scheduled_store([2, 1, 1])
            .expect_err("grouped leading map axis should fail");
        assert!(
            err.contains("leading map axis 0 must be 1"),
            "unexpected error for grouped leading map axis:\n{err}"
        );
    });
}

#[test]
fn rank1_mapped_partition_lowers_to_persistent_loop() {
    common::with_test_stack(|| {
        let generics = vec!["f32".to_string(), "128".to_string(), "1".to_string()];
        let stride_args = vec![("z", &[1][..])];
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "partition_index_schedules_module",
            "rank1_scheduled_store",
            &generics,
            &stride_args,
            &[],
            &[],
            Some((4, 1, 1)),
            &CompileOptions::default(),
        )
        .map_err(|err| err.to_string())
        .expect("Failed to compile rank-1 kernel");
        assert!(
            mlir.contains("get_index_space_shape"),
            "expected rank-1 index-space query in MLIR:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "expected rank-1 mapped store in MLIR:\n{mlir}"
        );
    });
}

fn compile_shared_map_kernel(function_name: &str) -> Result<String, String> {
    let generics = vec![
        "f32".to_string(),
        "64".to_string(),
        "64".to_string(),
        "1".to_string(),
        "1".to_string(),
    ];
    let stride_args: Vec<(&str, &[i32])> = match function_name {
        "shared_map_dual_store" => vec![("out", &[256, 1]), ("residual_out", &[256, 1])],
        "shared_map_rejects_third_partition" => {
            vec![("a", &[256, 1]), ("b", &[256, 1]), ("c", &[256, 1])]
        }
        other => panic!("unexpected shared-map test kernel `{other}`"),
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
fn shared_map_indices_store_into_both_partitions() {
    common::with_test_stack(|| {
        let mlir =
            compile_shared_map_kernel("shared_map_dual_store").expect("Failed to compile zip");
        let store_count = mlir.matches("store_view_tko").count();
        assert!(
            store_count >= 2,
            "expected two mapped stores through one shared index stream, found {store_count}:\n{mlir}"
        );
        assert!(
            mlir.contains("shared mapped partitions require equal logical partition grids"),
            "dynamic shared maps should carry per-axis grid-equality asserts:\n{mlir}"
        );
    });
}

#[test]
fn shared_map_still_rejects_unrelated_partition_store() {
    common::with_test_stack(|| {
        let err = compile_shared_map_kernel("shared_map_rejects_third_partition")
            .expect_err("store into a partition outside the shared map should fail");
        assert!(
            err.contains("index was produced by a different mapped partition"),
            "unexpected error for unrelated partition store:\n{err}"
        );
    });
}

fn compile_subrange_kernel(function_name: &str) -> Result<String, String> {
    let generics = vec![
        "f32".to_string(),
        "64".to_string(),
        "64".to_string(),
        "1".to_string(),
        "1".to_string(),
    ];
    let stride_args: Vec<(&str, &[i32])> = match function_name {
        "subrange_store" | "subrange_rejects_negative_start" => vec![("z", &[256, 1])],
        "subrange_shared_dual_store" => vec![("out", &[256, 1]), ("residual_out", &[256, 1])],
        other => panic!("unexpected sub-range test kernel `{other}`"),
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
fn subrange_iteration_checks_dynamic_ranges_at_runtime() {
    common::with_test_stack(|| {
        let mlir =
            compile_subrange_kernel("subrange_store").expect("Failed to compile sub-range kernel");
        assert!(
            mlir.contains("mapped partition sub-range out of bounds"),
            "dynamic sub-ranges should carry runtime bounds asserts:\n{mlir}"
        );
        assert!(
            mlir.contains("store_view_tko"),
            "expected mapped store in sub-range kernel:\n{mlir}"
        );
        assert!(
            mlir.contains("get_index_space_shape"),
            "sub-range iteration still queries the logical grid:\n{mlir}"
        );
    });
}

#[test]
fn subrange_rejects_provably_negative_start() {
    common::with_test_stack(|| {
        let err = compile_subrange_kernel("subrange_rejects_negative_start")
            .expect_err("negative sub-range start should fail at compile time");
        assert!(
            err.contains("start -1 is negative"),
            "unexpected error for negative sub-range start:\n{err}"
        );
    });
}

#[test]
fn subrange_composes_with_shared_maps() {
    common::with_test_stack(|| {
        let mlir = compile_subrange_kernel("subrange_shared_dual_store")
            .expect("Failed to compile shared sub-range kernel");
        assert!(
            mlir.contains("mapped partition sub-range out of bounds"),
            "shared sub-range should keep range asserts:\n{mlir}"
        );
        assert!(
            mlir.contains("shared mapped partitions require equal logical partition grids"),
            "shared sub-range should keep grid-equality asserts:\n{mlir}"
        );
        let store_count = mlir.matches("store_view_tko").count();
        assert!(
            store_count >= 2,
            "expected two mapped stores through the shared sub-range stream, found {store_count}:\n{mlir}"
        );
    });
}

#[test]
fn kv_loop_bounds_checks_hoist_to_the_preheader() {
    common::with_test_stack(|| {
        let generics = vec![
            "f32".to_string(),
            "16".to_string(),
            "16".to_string(),
            "64".to_string(),
            "1".to_string(),
            "1".to_string(),
            "1".to_string(),
        ];
        let stride_args: Vec<(&str, &[i32])> = vec![("out", &[64, 64, 1]), ("k", &[1024, 64, 1])];
        let mlir = common::compile_to_ir(
            __module_ast_self,
            "partition_index_schedules_module",
            "hoisted_checks_kv_loop",
            &generics,
            &stride_args,
            &[],
            &[],
            Some((4, 1, 1)),
            &CompileOptions::default(),
        )
        .map_err(|err| err.to_string())
        .expect("Failed to compile hoisted-checks kernel");
        assert!(
            mlir.contains("partition access out of bounds"),
            "checked loads must keep their bounds asserts somewhere:\n{mlir}"
        );
        // The inner KV loop is the `for` op carrying loop values
        // (`iter_values`, the accumulator). Nothing between its header and
        // its closing brace may assert: the checks live in the preheader.
        let kv_loop_start = mlir
            .find("iter_values")
            .expect("expected the KV loop with a loop-carried accumulator");
        let mut depth = 0usize;
        let mut kv_loop_end = mlir.len();
        let bytes = mlir.as_bytes();
        let mut began = false;
        for (offset, &byte) in bytes.iter().enumerate().skip(kv_loop_start) {
            match byte {
                b'{' => {
                    depth += 1;
                    began = true;
                }
                b'}' => {
                    depth = depth.saturating_sub(1);
                    if began && depth == 0 {
                        kv_loop_end = offset;
                        break;
                    }
                }
                _ => {}
            }
        }
        let kv_loop_body = &mlir[kv_loop_start..kv_loop_end];
        assert!(
            !kv_loop_body.contains("assert"),
            "bounds checks must hoist out of the KV loop body:\n{kv_loop_body}"
        );
    });
}

fn compile_hoist_kernel(function_name: &str, generics: &[&str]) -> String {
    let generics: Vec<String> = generics.iter().map(|g| g.to_string()).collect();
    let stride_args: Vec<(&str, &[i32])> = vec![("z", &[64, 1]), ("x", &[512, 1])];
    common::compile_to_ir(
        __module_ast_self,
        "partition_index_schedules_module",
        function_name,
        &generics,
        &stride_args,
        &[],
        &[],
        None,
        &CompileOptions::default(),
    )
    .map_err(|err| err.to_string())
    .expect("Failed to compile hoist kernel")
}

/// Every bounds assert must appear before the first loop op.
fn assert_all_checks_precede_loops(mlir: &str, label: &str) {
    let first_loop = mlir
        .find("for %")
        .unwrap_or_else(|| panic!("{label}: expected a for loop in MLIR:\n{mlir}"));
    assert!(
        mlir.contains("partition access out of bounds"),
        "{label}: checked loads must keep their asserts:\n{mlir}"
    );
    let last_assert = mlir
        .rfind("partition access out of bounds")
        .expect("asserted above");
    assert!(
        last_assert < first_loop,
        "{label}: all bounds checks must hoist above every loop:\n{mlir}"
    );
}

#[test]
fn invariant_checks_hoist_across_nested_loops() {
    common::with_test_stack(|| {
        let mlir = compile_hoist_kernel("nested_invariant_checks", &["f32", "16", "64"]);
        assert_all_checks_precede_loops(&mlir, "nested invariant");
    });
}

#[test]
fn affine_induction_checks_hoist_with_strongest_instance() {
    common::with_test_stack(|| {
        let mlir = compile_hoist_kernel("affine_index_checks", &["f32", "16", "64"]);
        assert_all_checks_precede_loops(&mlir, "affine");
        let first_loop = mlir.find("for %").unwrap();
        let preheader = &mlir[..first_loop];
        assert!(
            preheader.contains("muli"),
            "affine substitution must scale the loop bound in the preheader:\n{preheader}"
        );
    });
}
