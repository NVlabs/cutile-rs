/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
extern crate core;

use cuda_async::device_operation::DeviceOperation;
use cuda_core::CudaContext;
use cutile;
use cutile::api::{arange, zeros, DeviceOperationReshape};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Partition, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use cutile_examples::{pretty_print_matrix, to_candle_tensor};
use std::sync::Arc;

use my_module::tensor_permute;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry(
        print_ir=true,
        unchecked_accesses=false,
        optimization_hints = (tensor_dim_factor = 16,)
    )]
    unsafe fn tensor_permute<
        T: ElementType,
        const BBH: i32,
        const BB: i32,
        const BH: i32,
        const BM: i32,
        const BD: i32,
        const DIM_MAP: [i32; 4],
    >(
        src: &Tensor<T, { [-1, -1, -1, -1] }>,
        dst: &mut Tensor<T, { [BBH, BD, BM] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id(); // (b/BB*h/BH, m/BM, d/BD)

        // Tile dimensions BB and BH are collapsed into a single dimension.
        // Partition indices corresponding to those dimensions are recovered as follows.
        let h = get_shape_dim(src.shape(), 1i32);
        let b_idx = pid.0 / (h / BH); // \in [0, b/BB)
        let h_idx = pid.0 % (h / BH); // \in [0, h/BH)
        let d_idx = pid.1; // \in [0, d/BD)
        let m_idx = pid.2; // \in [0, m/BM)

        // Uncomment for debugging, but choose smaller shapes (smaller launch grid).
        // cuda_tile_print!("b_idx={}, h_idx={}, m_idx={}, d_idx={}\n", b_idx, h_idx, m_idx, d_idx);
        // cuda_tile_print!("BB={}, BH={}, BM={}, BD={}\n", BB, BH, BM, BD);
        // cuda_tile_print!("b={}, h={}, m={}, d={}\n", b, h, m, d);
        // dim_map specifies a permutation of a tensor's shape.
        let dim_map = const_array!(DIM_MAP);
        // Specify the *permuted* dimensions as the tile argument to partition.
        let src_part: Partition<T, { [BB, BH, BD, BM] }> =
            src.partition_permuted(const_shape![BB, BH, BD, BM], dim_map);
        // We load as-if the partition is laid out according to dim_map.
        // In this example, we swapped the last two dimensions.
        let src_tile: Tile<T, { [BB, BH, BD, BM] }> = src_part.load([b_idx, h_idx, d_idx, m_idx]);
        // The loaded tile is permuted according to dim_map.
        let src_tile = src_tile.reshape(const_shape![BBH, BD, BM]);
        // Here we probably fuse various operations.
        // We write the result to dst to check the answer.
        dst.store(src_tile);
    }
}

const BATCH: usize = 4;
const N_HEADS: usize = 32;
const N_CTX: usize = 1024;
const HEAD_DIM: usize = 64;

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;

    let b = BATCH; // = batch size.
    let h = N_HEADS; // = number of heads (query).
    let m = N_CTX; // = sequence length.
    let d = HEAD_DIM; // = hidden size.
    let partition = [1, 16, 128, 32];
    let dim_map = [0, 1, 3, 2];

    let bbh = partition[dim_map[0]] * partition[dim_map[1]];
    let partition_shape_rank3 = [
        bbh as i32,
        partition[dim_map[2]] as i32,
        partition[dim_map[3]] as i32,
    ];
    let src: Arc<Tensor<f32>> = arange(b * h * m * d)
        .reshape([b, h, m, d])
        .sync_on(&stream)?
        .into();
    let dst: Partition<Tensor<f32>> = zeros([b * h, d, m])
        .sync_on(&stream)?
        .partition(partition_shape_rank3);

    let mut generics: Vec<String> = [[bbh].as_slice(), partition.as_slice(), dim_map.as_slice()]
        .concat()
        .iter()
        .map(|x| x.to_string())
        .collect();
    generics.insert(0, "f32".to_string());
    let grid = dst.grid();
    println!("in shape = {:?}", src.shape);
    println!("in tile = {:?}", partition);
    println!("out shape = {:?}", [b * h, d, m]);
    println!("out tile = {:?}", partition_shape_rank3);
    println!("grid = {:?}", grid);
    println!("generics: {:?}", generics);
    let (src, dst) = unsafe { tensor_permute(src.clone(), dst) }
        .generics(generics.clone())
        .sync_on(&stream)?;

    let out_vec: Vec<f32> = dst.unpartition().to_host_vec().sync_on(&stream)?;
    let src_vec: Vec<f32> = src.to_host_vec().sync_on(&stream)?;
    let out_host = to_candle_tensor(&out_vec, &[b * h, d, m]);
    let answer_host = to_candle_tensor(&src_vec, &[b, h, m, d]);
    let answer_host = answer_host
        .permute((
            dim_map[0] as usize,
            dim_map[1] as usize,
            dim_map[2] as usize,
            dim_map[3] as usize,
        ))
        .unwrap();
    let answer_host = answer_host.reshape((b * h, d, m)).unwrap();
    for i in 0..(b * h) {
        let answer_mat = answer_host
            .get_on_dim(0, i)
            .expect("Failed to get {i} on dim 0.");
        let out_mat = out_host
            .get_on_dim(0, i)
            .expect("Failed to get {i} on dim 0.");
        let near_zero = (&answer_mat - &out_mat)
            .unwrap()
            .abs()
            .unwrap()
            .reshape((m * d,))
            .unwrap();
        let vec = near_zero.to_vec1::<f32>().unwrap();
        let check = vec.iter().all(|x| x.abs() <= 1e-4);
        if !check {
            println!("Output:");
            pretty_print_matrix::<f32>(&out_mat);
            println!("Answer:");
            pretty_print_matrix::<f32>(&answer_mat);
            assert!(check, "output check failed.");
        }
    }
    Ok(())
}
