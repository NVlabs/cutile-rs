/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cuda_async::device_operation::DeviceOperation;
use cuda_core::CudaContext;
use cutile::api::{randn_f16, zeros};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, Partition, Tensor};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(print_ir=false, unchecked_accesses=true,
        optimization_hints = (
            tensor_dim_factor = 8,
            sm_120 = (allow_tma = false, latency = 1, occupancy = 1),
    ))]
    unsafe fn softmax<const BM: i32, const BN: i32>(
        x: &Tensor<f16, { [-1, -1] }>,
        y: &mut Tensor<f16, { [BM, BN] }>,
    ) {
        let tile_x: Tile<f16, { [BM, BN] }> = load_tile_like_2d(x, y);
        let tile_x_max: Tile<f16, { [BM] }> = reduce_max(tile_x, 1i32);
        let tile_x_max: Tile<f16, { [BM, BN] }> =
            tile_x_max.reshape(const_shape![BM, 1]).broadcast(y.shape());
        let num: Tile<f16, { [BM, BN] }> = exp(tile_x - tile_x_max);
        let denom: Tile<f16, { [BM] }> = reduce_sum(num, 1);
        let denom = denom.reshape(const_shape![BM, 1]).broadcast(y.shape());
        y.store(num / denom);
    }
}

fn softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    group
        .warm_up_time(Duration::from_millis(1000))
        .sample_size(10usize.pow(2))
        .measurement_time(Duration::from_millis(5000));

    let ctx = CudaContext::new(0).expect("Failed to get context.");
    let stream = ctx.new_stream().expect("Failed to get stream.");

    let tile_sizes = vec![
        (1, 1024),
        (1, 2048),
        (1, 2048),
        (1, 4096),
        (1, 4096),
        (1, 4096),
    ];
    let mut params = vec![];
    for i in 0..6 {
        let n = 2usize.pow(10 + i);
        params.push((4096, n, tile_sizes[i as usize]));
    }
    for i in 0..6 {
        let (m, n, (bm, bn)) = params[i];
        let generics = vec![bm.to_string(), bn.to_string()];
        let x: Arc<Tensor<f16>> = randn_f16(f16::ZERO, f16::ONE, [m, n], None)
            .sync_on(&stream)
            .expect("Failed.")
            .into();
        let total_bytes = 2 * x.num_bytes();
        let label = ("N", n.to_string());
        // This should report bytes/sec.
        group.throughput(Throughput::BytesDecimal(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(label.0, &label.1),
            &label.1,
            |b, _size_mb| {
                b.iter_custom(|iters| {
                    let mut out: Partition<Tensor<f16>> = zeros([m, n])
                        .sync_on(&stream)
                        .expect("Failed.")
                        .partition([bm, bn]);
                    stream.synchronize().expect("Failed to synchronize.");
                    let start = Instant::now();
                    for _i in 0..iters {
                        unsafe {
                            let (_, local_out) = kernels::softmax(x.clone(), out)
                                .generics(generics.clone())
                                .async_on(&stream)
                                .expect("Failed.");
                            out = local_out;
                        }
                    }
                    stream.synchronize().expect("Failed to synchronize.");
                    let res = start.elapsed();
                    res
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, softmax);
criterion_main!(benches);
