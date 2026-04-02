/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use std::sync::Arc;

use cutile;
use cutile::api;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::DeviceOperation;
use half::f16;

mod common;

#[cutile::module]
mod tensor_reinterpret_module {
    use cutile::core::*;

    // This kernel consumes a reinterpreted tensor through the normal immutable launcher path.
    #[cutile::entry()]
    fn passthrough_f32(output: &mut Tensor<f32, { [4] }>, input: &Tensor<f32, { [-1] }>) {
        let tile: Tile<f32, { [4] }> = load_tile_like_1d(input, output);
        output.store(tile);
    }
}

#[test]
fn reinterpret_is_zero_copy() {
    common::with_test_stack(|| {
        // These u32 values are IEEE-754 bit patterns for 1.0, 2.0, 3.0, and 4.0.
        let bits: Arc<Vec<u32>> = Arc::new(vec![0x3f800000, 0x40000000, 0x40400000, 0x40800000]);
        let base = Arc::new(api::copy_host_vec_to_device(&bits).sync().expect("Failed."));
        // Reinterpret the same device bytes as a rank-2 f32 tensor without copying them.
        let floats_2d = base.reinterpret::<f32, 2>([2, 2]);

        // Reinterpret keeps the same backing storage and only changes dtype/shape metadata.
        assert_eq!(base.cu_deviceptr(), floats_2d.cu_deviceptr());
        assert_eq!(floats_2d.shape, vec![2, 2]);
        assert_eq!(floats_2d.strides, vec![2, 1]);
        assert_eq!(floats_2d.size(), 4);

        // This is a bit reinterpret, not a numeric conversion.
        let host: Vec<f32> = floats_2d.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![1.0, 2.0, 3.0, 4.0]);
    });
}

#[test]
fn reinterpret_rejects_invalid_byte_size() {
    common::with_test_stack(|| {
        let bytes: Arc<Vec<u8>> = Arc::new(vec![1, 2, 3]);
        let base = Arc::new(
            api::copy_host_vec_to_device(&bytes)
                .sync()
                .expect("Failed."),
        );

        // Reinterpret must preserve the total byte size exactly.
        assert!(base.try_reinterpret::<u32, 1>([1]).is_err());
        assert!(base.try_reinterpret_dyn::<u32>(&[1]).is_err());
    });
}

#[test]
fn reinterpret_u8_to_i16_succeeds() {
    common::with_test_stack(|| {
        let expected = vec![0x1234i16, 0x2BCDi16];
        // Expand each i16 into its native-endian bytes so the source tensor is byte-typed.
        let bytes: Arc<Vec<u8>> = Arc::new(
            expected
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        let base = Arc::new(
            api::copy_host_vec_to_device(&bytes)
                .sync()
                .expect("Failed."),
        );
        // Reinterpret the same bytes as i16 values; no device allocation or copy should occur.
        let words = base.reinterpret::<i16, 1>([expected.len()]);

        // Signedness is part of the new view type; the underlying bytes stay unchanged.
        assert_eq!(base.cu_deviceptr(), words.cu_deviceptr());
        // Reading back through the i16 view should reconstruct the original signed values.
        let host: Vec<i16> = words.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, expected);
    });
}

#[test]
fn reinterpret_u8_boundaries_are_enforced() {
    common::with_test_stack(|| {
        let valid_bits = vec![0x3f800000u32, 0x40000000u32];
        let valid_bytes: Arc<Vec<u8>> = Arc::new(
            valid_bits
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        let valid = Arc::new(
            api::copy_host_vec_to_device(&valid_bytes)
                .sync()
                .expect("Failed."),
        );
        let floats = valid.reinterpret::<f32, 1>([2]);
        let host: Vec<f32> = floats.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, vec![1.0, 2.0]);

        // Six bytes cannot back two f32 values, so this must fail.
        let odd_bytes: Arc<Vec<u8>> = Arc::new(vec![0, 0, 0, 0, 0, 0]);
        let odd = Arc::new(
            api::copy_host_vec_to_device(&odd_bytes)
                .sync()
                .expect("Failed."),
        );
        assert!(odd.try_reinterpret::<f32, 1>([2]).is_err());

        // Three bytes cannot back one i16 plus preserve the original tensor size invariant.
        let odd_words: Arc<Vec<u8>> = Arc::new(vec![1, 2, 3]);
        let odd_words = Arc::new(
            api::copy_host_vec_to_device(&odd_words)
                .sync()
                .expect("Failed."),
        );
        assert!(odd_words.try_reinterpret::<i16, 1>([1]).is_err());
    });
}

#[test]
fn reinterpret_i16_to_f16_succeeds() {
    common::with_test_stack(|| {
        let expected = vec![f16::from_f32(1.0), f16::from_f32(-2.5), f16::from_f32(0.5)];
        let bits: Arc<Vec<i16>> = Arc::new(
            expected
                .iter()
                .map(|value| i16::from_ne_bytes(value.to_bits().to_ne_bytes()))
                .collect(),
        );
        let base = Arc::new(api::copy_host_vec_to_device(&bits).sync().expect("Failed."));
        let halfs = base.reinterpret::<f16, 1>([expected.len()]);

        // The host-side source uses f16 bit patterns stored in i16 slots.
        assert_eq!(base.cu_deviceptr(), halfs.cu_deviceptr());
        let host: Vec<f16> = halfs.to_host_vec().sync().expect("Failed.");
        assert_eq!(host, expected);
    });
}

#[test]
fn reinterpreted_tensors_work_with_kernels() {
    common::with_test_stack(|| {
        //These u32 values are IEEE-754 bit patterns for 1.0, 2.0, 3.0, and 4.0.
        let bits: Arc<Vec<u32>> = Arc::new(vec![0x3f800000, 0x40000000, 0x40400000, 0x40800000]);
        let base = Arc::new(api::copy_host_vec_to_device(&bits).sync().expect("Failed."));
        let floats = base.reinterpret::<f32, 1>([4]);

        // If launcher validation and argument marshalling accept the reinterpreted view,
        // kernels can consume it exactly like an ordinary Arc<Tensor<f32>>.
        let output = api::zeros::<1, f32>([4]).sync().expect("Failed.");
        let (result, _input) =
            tensor_reinterpret_module::passthrough_f32(output.partition([4]), floats)
                .sync()
                .expect("Failed.");
        let host: Vec<f32> = result.unpartition().to_host_vec().sync().expect("Failed.");

        assert_eq!(host, vec![1.0, 2.0, 3.0, 4.0]);
    });
}
