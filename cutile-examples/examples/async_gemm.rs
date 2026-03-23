/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation::*;
use cutile::api::{self, copy_to_host};
use cutile::candle_core::WithDType;
use cutile::half::f16;
use cutile::num_traits::identities::*;
use cutile::tensor::{Tensor, ToHostVec, Unpartition};
use cutile::tile_kernel::{IntoDeviceOperationPartition, TileKernel};
use my_module::gemm_op;
use std::fmt::Debug;
use std::sync::Arc;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<
        E1: ElementType,
        E2: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const K: i32,
    >(
        z: &mut Tensor<E1, { [BM, BN] }>,
        x: &Tensor<E2, { [-1, K] }>,
        y: &Tensor<E2, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid = get_tile_block_id();
        let mut tile_z = z.load();
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z);
    }
}

fn gemm<T1: WithDType + Debug, T2: WithDType + Debug>(
    x: Arc<Tensor<T2>>,
    y: Arc<Tensor<T2>>,
) -> impl DeviceOperation<Output = Tensor<T1>> {
    let (m, n, k) = (
        x.shape[0] as usize,
        y.shape[1] as usize,
        x.shape[1] as usize,
    );
    let (bm, bn, bk) = (16, 16, 8);
    let generics = [
        T1::DTYPE.as_str().to_string(),
        T2::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];
    let z = api::zeros([m, n]).partition([bm, bn]); // impl DeviceOperation
    let (z, _x, _y) = gemm_op(z, x.device_operation(), y.device_operation())
        .generics(generics.to_vec())
        .unzip();
    z.unpartition()
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), cuda_async::error::DeviceError> {
    type In = f16;
    type Out = f32;
    let (m, n, k) = (64, 64, 16);
    let x = api::randn(In::zero(), In::one(), [m, k]).arc().await?; // impl DeviceOperation
    let y = api::randn(In::zero(), In::one(), [k, n]).arc().await?; // impl DeviceOperation
    let z = gemm::<Out, In>(x.clone(), y.clone()).await?;
    let z_host: Vec<Out> = z.to_host_vec().await?;
    let x_host = copy_to_host(&x).await?;
    let y_host = copy_to_host(&y).await?;
    let answer_host: Vec<f16> = x_host
        .matmul(&y_host)
        .unwrap()
        .reshape(((),))
        .unwrap()
        .to_vec1()
        .unwrap();
    for i in 0..(m * n) {
        println!(
            "z_host[{i}] == answer_host[{i}]? {} == {}",
            z_host[i], answer_host[i]
        );
    }
    Ok(())
}
