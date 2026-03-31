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
use kernels::rms_norm;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(print_ir=false, unchecked_accesses=true,
                       optimization_hints = (tensor_dim_factor = 8,))]
    unsafe fn rms_norm<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f16,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        // The launch grid is (M, 1, 1).
        // row is a pid in [0, M).
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        // TODO (hme): Parse 0.0f32 syntax properly.
        let mut rms: Tile<f16, { [1, BLOCK_SIZE] }> = constant(f16::ZERO, tile_shape);
        for j in 0i32..num_tiles {
            let tx: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            rms = rms + tx * tx;
        }
        // TODO (hme): Try to make this something like:
        //  let rms = (1.0 / (rms.sum(/*axis=*/1, /*keepdims=*/true) / N + eps).sqrt()).broadcast(tile_shape);
        let rms: Tile<f16, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f16, { [] }> = rms.reshape(const_shape![]);
        let rms: f16 = tile_to_scalar(rms);
        let n: f16 = convert_scalar(N);
        let one: f16 = convert_scalar(1.0f32);
        let rms: f16 = one / (rms / n + eps);
        let rms: Tile<f16, { [] }> = sqrt(scalar_to_tile(rms), "negative_inf");
        let rms: f16 = tile_to_scalar(rms);
        let rms: Tile<f16, { [1, BLOCK_SIZE] }> = broadcast_scalar(rms, tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        // TODO (hme): This is a safety leak. If this partition goes out of scope, we can partition out again,
        //  and any memory ops will not succeed tokens corresponding to write operations (since those will also be dropped).
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tout: Tile<f16, { [1, BLOCK_SIZE] }> = tx * rms * tw;
            unsafe { out_part.store(tout, [0i32, j]) };
        }
    }
}

fn ocean_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");
    group
        .warm_up_time(Duration::from_millis(1000))
        .sample_size(10usize.pow(2))
        .measurement_time(Duration::from_millis(5000));

    let ctx = CudaContext::new(0).expect("Failed to get context.");
    let stream = ctx.new_stream().expect("Failed to get stream.");

    let base_tile_size = 512;
    let tile_sizes = vec![
        base_tile_size,
        base_tile_size,
        base_tile_size,
        base_tile_size,
        base_tile_size,
        base_tile_size,
    ];
    let mut params = vec![];
    for i in 0..6 {
        let n = 2usize.pow(10 + i);
        params.push((4096, n, tile_sizes[i as usize]));
    }
    for i in 0..6 {
        let (m, n, tile_size) = params[i];
        let eps = f16::from_f32(1e-5);
        let generics = vec![n.to_string(), tile_size.to_string()];
        let x: Arc<Tensor<f16>> = randn_f16(f16::ZERO, f16::ONE, [m, n], None)
            .sync_on(&stream)
            .expect("Failed.")
            .into();
        let w: Arc<Tensor<f16>> = randn_f16(f16::ZERO, f16::ONE, [n], None)
            .sync_on(&stream)
            .expect("Failed.")
            .into();
        let label = n.to_string();
        // This should report bytes/sec.
        let total_bytes = x.num_bytes() + w.num_bytes() + x.num_bytes();
        group.throughput(Throughput::BytesDecimal(total_bytes as u64));
        group.bench_with_input(BenchmarkId::new("N", &label), &label, |b, _size_mb| {
            b.iter_custom(|iters| {
                let mut out: Partition<Tensor<f16>> = zeros([m, n])
                    .sync_on(&stream)
                    .expect("Failed.")
                    .partition([1, n as i32]);
                stream.synchronize().expect("Failed to synchronize.");
                let start = Instant::now();
                for _i in 0..iters {
                    unsafe {
                        let (_x, _w, local_out, _eps) = rms_norm(x.clone(), w.clone(), out, eps)
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
        });
    }
    group.finish();
}

criterion_group!(benches, ocean_rmsnorm);
criterion_main!(benches);
