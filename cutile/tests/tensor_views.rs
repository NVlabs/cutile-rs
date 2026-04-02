/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use cutile;
use cutile::api;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::DeviceOperation;

mod common;

#[cutile::module]
mod tensor_views_module {
    use cutile::core::*;

    // These kernels let the same backing storage be passed through launchers with
    // different expected tensor ranks.
    #[cutile::entry()]
    fn passthrough_1d(output: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let tile: Tile<f32, { [4] }> = load_tile_like_1d(input, output);
        output.store(tile);
    }

    #[cutile::entry()]
    fn passthrough_2d(output: &mut Tensor<f32, { [2, 2] }>, input: &Tensor<f32, { [-1, -1] }>) {
        let tile: Tile<f32, { [2, 2] }> = load_tile_like_2d(input, output);
        output.store(tile);
    }
}

#[test]
fn arc_views_are_zero_copy() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
        let view = base.view([2, 4]);
        let dyn_view = base.try_view_dyn(&[4, 2]).expect("Failed.");
        let flat = view.flatten_view();

        // All views must share the same device allocation and only differ in metadata.
        assert_eq!(base.cu_deviceptr(), view.cu_deviceptr());
        assert_eq!(base.cu_deviceptr(), dyn_view.cu_deviceptr());
        assert_eq!(base.cu_deviceptr(), flat.cu_deviceptr());
        assert_eq!(view.shape, vec![2, 4]);
        assert_eq!(view.strides, vec![4, 1]);
        assert_eq!(dyn_view.shape, vec![4, 2]);
        assert_eq!(dyn_view.strides, vec![2, 1]);
        assert_eq!(flat.shape, vec![8]);
        assert_eq!(flat.strides, vec![1]);

        let base_host: Vec<f32> = (&base).to_host_vec().sync().expect("Failed.");
        let flat_host: Vec<f32> = flat.to_host_vec().sync().expect("Failed.");
        assert_eq!(base_host, flat_host);

        // A real copy allocates fresh storage, so its pointer must differ from the views above.
        let copied = base.copy().sync().expect("Failed.");
        assert_ne!(base.cu_deviceptr(), copied.cu_deviceptr());
    });
}

#[test]
fn invalid_view_shape_is_rejected() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
        assert!(base.try_view([5]).is_err());
        assert!(base.try_view_dyn(&[2, 2]).is_err());
    });
}

#[test]
fn shared_storage_blocks_mutable_partition() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(8).sync().expect("Failed."));
        let _view = base.view([2, 4]);
        // Unwrapping the outer Arc gives back a Tensor value, but the backing storage is still
        // shared with `_view`, so mutable partitioning must be rejected.
        let owned = Arc::try_unwrap(base).expect("Expected unique outer Arc.");
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = owned.partition([8]);
        }));
        assert!(result.is_err(), "Expected mutable partition to be rejected");
    });
}

#[test]
fn arc_views_work_with_different_rank_kernels() {
    common::with_test_stack(|| {
        let base = Arc::new(api::arange::<f32>(4).sync().expect("Failed."));
        // These are two metadata views over the same storage, shaped to match different kernels.
        let input_1d = base.view([4]);
        let input_2d = base.view([2, 2]);

        let output_1d = api::zeros::<1, f32>([4]).sync().expect("Failed.");
        let (result_1d, _input_1d) =
            tensor_views_module::passthrough_1d(output_1d.partition([4]), input_1d)
                .sync()
                .expect("Failed.");
        let host_1d: Vec<f32> = result_1d
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("Failed.");

        let output_2d = api::zeros::<2, f32>([2, 2]).sync().expect("Failed.");
        let (result_2d, _input_2d) =
            tensor_views_module::passthrough_2d(output_2d.partition([2, 2]), input_2d)
                .sync()
                .expect("Failed.");
        let host_2d: Vec<f32> = result_2d
            .unpartition()
            .to_host_vec()
            .sync()
            .expect("Failed.");

        assert_eq!(host_1d, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(host_2d, vec![0.0, 1.0, 2.0, 3.0]);
    });
}
