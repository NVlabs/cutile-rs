/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![allow(unused_variables)]

//! Book Examples - Tests all code examples from the cuTile Rust Book
//!
//! This example verifies that all tutorial code from the book compiles and runs correctly.
//! Run with: cargo run --example book_examples

use cuda_async::device_operation::*;
use cuda_core::{CudaContext, CudaStream};
use cutile;
use cutile::api::{arange, ones, randn, zeros};
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Reshape, Tensor, ToHostVec, Unpartition};
use cutile::tile_kernel::{PartitionOp, TileKernel, ToHostVecOp};
use std::sync::Arc;

// ============================================================================
// Tutorial 1: Hello World
// ============================================================================
#[cutile::module]
mod hello_world_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn hello_world_kernel() {
        let pids: (i32, i32, i32) = get_tile_block_id();
        let npids: (i32, i32, i32) = get_num_tile_blocks();
        cuda_tile_print!(
            "Hello from tile <{}, {}, {}> in a grid of <{}, {}, {}> tiles!\n",
            pids.0,
            pids.1,
            pids.2,
            npids.0,
            npids.1,
            npids.2
        );
    }
}

fn test_hello_world(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("\n=== Tutorial 1: Hello World ===");
    use hello_world_module::hello_world_kernel;

    hello_world_kernel().grid((2, 2, 1)).sync_on(stream)?;
    println!("Hello World passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 2: Vector Addition
// ============================================================================
#[cutile::module]
mod vector_add_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = load_tile_like_2d(x, z);
        let tile_y = load_tile_like_2d(y, z);
        z.store(tile_x + tile_y);
    }
}

fn test_vector_addition(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 2: Vector Addition ===");
    use vector_add_module::add;

    let x: Arc<Tensor<f32>> = ones(&[32, 32]).sync_on(stream)?.into();
    let y: Arc<Tensor<f32>> = ones(&[32, 32]).sync_on(stream)?.into();
    let z = zeros(&[32, 32]).sync_on(stream)?.partition([4, 4]);

    let z_host = add(z, x, y)
        .unzip()
        .0
        .unpartition()
        .to_host_vec()
        .sync_on(stream)?;

    let all_correct = z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5);
    assert!(all_correct, "Vector addition failed: expected all 2.0s");
    println!("z[0] = {} (expected 2.0)", z_host[0]);
    println!("Vector Addition passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 3: SAXPY (y = a*x + y)
// ============================================================================
#[cutile::module]
mod saxpy_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn saxpy<const S: [i32; 2]>(a: f32, x: &Tensor<f32, { [-1, -1] }>, y: &mut Tensor<f32, S>) {
        let tile_a = a.broadcast(y.shape());
        let tile_x = load_tile_like_2d(x, y);
        let tile_y = y.load();
        y.store(tile_a * tile_x + tile_y);
    }
}

fn test_saxpy(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 3: SAXPY ===");
    use saxpy_module::saxpy;

    let a = 2.0f32;
    let input: Arc<Tensor<f32>> = arange(32usize).sync_on(stream)?.into();
    let x: Arc<Tensor<f32>> = input
        .dup()
        .sync_on(stream)?
        .reshape(&[4, 8])
        .unwrap()
        .into();
    let y = input
        .dup()
        .sync_on(stream)?
        .reshape(&[4, 8])
        .unwrap()
        .partition([2, 2]);

    let (a_out, _x, y) = saxpy(a, x, y).sync_on(stream)?;
    let y_host: Vec<f32> = y.unpartition().to_host_vec().sync_on(stream)?;
    let input_host: Vec<f32> = input.to_host_vec().sync_on(stream)?;

    for i in 0..5 {
        let x_i = input_host[i];
        let y_original = input_host[i];
        let expected = a_out * x_i + y_original;
        println!(
            "{} * {} + {} = {} (got {})",
            a_out, x_i, y_original, expected, y_host[i]
        );
        assert!(
            (expected - y_host[i]).abs() < 1e-5,
            "SAXPY mismatch at index {}",
            i
        );
    }
    println!("SAXPY passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 4: Matrix Multiplication (GEMM)
// ============================================================================
#[cutile::module]
mod gemm_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }
}

fn test_gemm(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 4: Matrix Multiplication (GEMM) ===");
    use cutile::api;
    use cutile::DType;
    use gemm_module::gemm;

    let (bm, bn, bk) = (16, 16, 8);
    let (m, n, k) = (64usize, 64usize, 64usize);

    let generics = vec![
        f32::DTYPE.as_str().to_string(),
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        k.to_string(),
    ];

    let z = api::zeros(&[m, n]).partition([bm, bn]).sync_on(stream)?;
    let x: Arc<Tensor<f32>> = api::ones(&[m, k]).sync_on(stream)?.into();
    let y: Arc<Tensor<f32>> = api::ones(&[k, n]).sync_on(stream)?.into();

    let (z, _x, _y) = gemm(z, x, y).generics(generics).sync_on(stream)?;
    let z_host: Vec<f32> = z.unpartition().to_host_vec().sync_on(stream)?;

    println!("z[0] = {} (expected {})", z_host[0], k);
    assert!((z_host[0] - k as f32).abs() < 1e-3, "GEMM result incorrect");
    println!("GEMM passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 5: Fused Softmax
// ============================================================================
#[cutile::module]
mod softmax_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn softmax<const BM: i32, const BN: i32>(
        x: &Tensor<f32, { [-1, -1] }>,
        y: &mut Tensor<f32, { [BM, BN] }>,
    ) {
        let tile_x: Tile<f32, { [BM, BN] }> = load_tile_like_2d(x, y);
        let tile_x_max: Tile<f32, { [BM] }> = reduce_max(tile_x, 1i32);
        let tile_x_max: Tile<f32, { [BM, BN] }> =
            tile_x_max.reshape(const_shape![BM, 1]).broadcast(y.shape());
        let num: Tile<f32, { [BM, BN] }> = exp(tile_x - tile_x_max);
        let denom: Tile<f32, { [BM] }> = reduce_sum(num, 1);
        let denom = denom.reshape(const_shape![BM, 1]).broadcast(y.shape());
        y.store(num / denom);
    }
}

fn test_softmax(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 5: Fused Softmax ===");
    use softmax_module::softmax;

    let (m, n) = (4usize, 8usize);
    let (bm, bn) = (2, n);

    let input: Arc<Tensor<f32>> = arange(m * n).sync_on(stream)?.into();
    let x: Arc<Tensor<f32>> = input
        .dup()
        .sync_on(stream)?
        .reshape(&[m, n])
        .unwrap()
        .into();
    let y = input
        .dup()
        .sync_on(stream)?
        .reshape(&[m, n])
        .unwrap()
        .partition([bm, bn]);

    let (_x, y) = softmax(x, y).sync_on(stream)?;
    let y_host: Vec<f32> = y.unpartition().to_host_vec().sync_on(stream)?;

    for i in (0..y_host.len()).step_by(n) {
        let row_sum: f32 = y_host[i..i + n].iter().sum();
        println!("softmax row sum = {} (expected 1.0)", row_sum);
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "Softmax row {} doesn't sum to 1.0",
            i / n
        );
    }
    println!("Softmax passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 6: Fused Multihead Attention
// ============================================================================
#[cutile::module]
mod fmha_module {
    use cutile::core::*;

    #[cutile::entry(print_ir = false)]
    fn fmha<
        const BM: i32, // Q tile size (rows of output we compute)
        const BN: i32, // K,V tile size (how many K,V we process at once)
        const D: i32,  // Head dimension
    >(
        q: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, M, D)
        k: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, N, D)
        v: &Tensor<f32, { [-1, -1, -1, -1] }>, // (B, H, N, D)
        out: &mut Tensor<f32, { [1, BM, D] }>,
        qk_scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h = get_shape_dim(q.shape(), 1i32);
        let batch_idx = pid.0 / h;
        let head_idx = pid.0 % h;
        let q_m_idx = pid.1;

        // Convert to exp2-friendly scale (exp2 is faster than exp on GPU)
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        let log2: f32 = tile_to_scalar(log(two));
        let qk_scale: f32 = qk_scale / log2;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        // Online softmax state
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(f32::NEG_INFINITY, const_shape![BM, 1]);
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        // Load Q tile once and reuse for all K,V tiles
        let q_part: Partition<f32, { [1, 1, BM, D] }> = q.partition(const_shape![1, 1, BM, D]);
        let tq: Tile<f32, { [1, 1, BM, D] }> = q_part.load([batch_idx, head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = tq.reshape(const_shape![BM, D]);

        let n: i32 = get_shape_dim(k.shape(), 2i32);
        let num_tiles: i32 = ceil_div(n, BN);

        let k_part = k.partition(const_shape![1, 1, BN, D]);
        let v_part = v.partition(const_shape![1, 1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        // Stream through K,V tiles
        for j in 0i32..num_tiles {
            let k_tile: Tile<f32, { [BN, D] }> = k_part
                .load([batch_idx, head_idx, j, 0i32])
                .reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f32, { [D, BN] }> = permute(k_tile, transpose);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp2(qk, ftz::Disabled);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp2(m_i - m_ij, ftz::Disabled);

            l_i = l_i * alpha + l_ij;
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f32, { [1, 1, BN, D] }> = v_part.load([batch_idx, head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = v_tile.reshape(const_shape![BN, D]);
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc = acc.reshape(const_shape![1, BM, D]);
        out.store(acc);
    }
}

fn test_flash_attention(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 6: Fused Multihead Attention ===");
    use fmha_module::fmha;

    let (batch, heads, seq_len, head_dim) = (1, 2, 32, 16);
    let (bm, bn) = (16, 16);

    let seed = 42u64;
    let q: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed))
        .sync_on(stream)?
        .into();
    let k: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed + 1))
        .sync_on(stream)?
        .into();
    let v: Arc<Tensor<f32>> = randn(0., 1., [batch, heads, seq_len, head_dim], Some(seed + 2))
        .sync_on(stream)?
        .into();

    let out = zeros(&[batch * heads, seq_len, head_dim])
        .sync_on(stream)?
        .partition([1, bm, head_dim]);

    let qk_scale = 1.0 / f32::sqrt(head_dim as f32);
    let generics = vec![bm.to_string(), bn.to_string(), head_dim.to_string()];

    let (_, _, _, out, _) = fmha(q, k, v, out, qk_scale)
        .generics(generics)
        .sync_on(stream)?;

    let out_host: Vec<f32> = out.unpartition().to_host_vec().sync_on(stream)?;

    // Verify output has correct number of elements.
    let expected_len = batch * heads * seq_len * head_dim;
    assert_eq!(
        out_host.len(),
        expected_len,
        "FMHA output length mismatch: got {}, expected {}",
        out_host.len(),
        expected_len
    );
    // Verify output contains finite values (softmax + matmul should produce finite results).
    assert!(
        out_host.iter().all(|v| v.is_finite()),
        "FMHA output contains non-finite values"
    );
    println!(
        "Output: {} elements, first={:.4}, last={:.4}",
        out_host.len(),
        out_host[0],
        out_host[out_host.len() - 1]
    );
    println!("Flash Attention passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 7: Intro to Async (sync equivalent — DeviceOp composition)
// ============================================================================
fn test_async_composition(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 7: Intro to Async (DeviceOp Composition) ===");
    use vector_add_module::add;

    // Demonstrate the DeviceOp composition pattern from tutorial 7:
    // Build a lazy computation graph, then execute it in one shot.
    let x: Arc<Tensor<f32>> = ones(&[32, 32]).sync_on(stream)?.into();
    let y: Arc<Tensor<f32>> = ones(&[32, 32]).sync_on(stream)?.into();

    // Compose: partition output, feed inputs, chain kernel, extract result.
    let z_host: Vec<f32> = add(zeros(&[32, 32]).partition([4, 4]), x, y)
        .first()
        .unpartition()
        .to_host_vec()
        .sync_on(stream)?;

    let all_correct = z_host.iter().all(|&v| (v - 2.0).abs() < 1e-5);
    assert!(all_correct, "Async composition failed: expected all 2.0s");
    println!("z[0] = {} (expected 2.0)", z_host[0]);

    // Demonstrate lazy graph: all inputs are DeviceOps, executed in one shot.
    let z_host2: Vec<f32> = add(
        zeros(&[32, 32]).partition([4, 4]),
        ones(&[32, 32]).map(|t: Tensor<f32>| -> Arc<Tensor<f32>> { Arc::new(t) }),
        ones(&[32, 32]).map(|t: Tensor<f32>| -> Arc<Tensor<f32>> { Arc::new(t) }),
    )
    .first()
    .unpartition()
    .to_host_vec()
    .sync_on(stream)?;

    let all_correct2 = z_host2.iter().all(|&v| (v - 2.0).abs() < 1e-5);
    assert!(
        all_correct2,
        "Lazy graph composition failed: expected all 2.0s"
    );
    println!("Lazy graph: z[0] = {} (expected 2.0)", z_host2[0]);
    println!("Async Composition passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 8: Data Parallel MLP
// ============================================================================
#[cutile::module]
mod mlp_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn mlp_gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<E, { [BM, BN] }>,
        x: &Tensor<E, { [-1, K] }>,
        y: &Tensor<E, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    #[cutile::entry()]
    pub fn mlp_matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load().reshape(const_shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i]).reshape(const_shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z.reshape(const_shape![BM]));
    }

    #[cutile::entry()]
    fn mlp_relu<const D: i32>(input_output: &mut Tensor<f32, { [D] }>) {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, const_shape![D]);
        let input = input_output.load();
        input_output.store(max_tile(zero_tile, input));
    }
}

fn test_data_parallel_mlp(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 8: Data Parallel MLP ===");
    use cutile::api;
    use cutile::DType;
    use mlp_module::{mlp_gemm, mlp_matvec, mlp_relu};

    let dim = 16usize;
    let block_dim = 4usize;
    let gemm_generics = vec![
        f32::DTYPE.as_str().to_string(),
        block_dim.to_string(),
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let mv_generics = vec![
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];

    // Weights: W0 (dim × dim), W1 (dim,)
    let w0: Arc<Tensor<f32>> = api::ones(&[dim, dim]).sync_on(stream)?.into();
    let w1: Arc<Tensor<f32>> = api::ones(&[dim]).sync_on(stream)?.into();

    // Input: ones(dim × dim) — each row is all 1s.
    let data: Arc<Tensor<f32>> = api::ones(&[dim, dim]).sync_on(stream)?.into();

    // Layer 1: GEMM — data @ W0. ones(16,16) @ ones(16,16) = full(16, 16×16).
    let out0 = api::zeros(&[dim, dim])
        .partition([block_dim, block_dim])
        .sync_on(stream)?;
    let (out0, _, _) = mlp_gemm(out0, data, w0)
        .generics(gemm_generics)
        .sync_on(stream)?;
    let out0_arc: Arc<Tensor<f32>> = out0.unpartition().into();

    // Layer 2: MatVec — out0[0] @ W1. Sum of row = dim*dim = 256, times 1 = 256.
    let out1 = api::zeros(&[dim]).partition([block_dim]).sync_on(stream)?;
    let (out1, _, _) = mlp_matvec(out1, out0_arc, w1)
        .generics(mv_generics)
        .sync_on(stream)?;

    // ReLU — all positive, so no change.
    let (out1,) = mlp_relu(out1).sync_on(stream)?;

    let out_host: Vec<f32> = out1.unpartition().to_host_vec().sync_on(stream)?;
    let expected = (dim * dim) as f32; // ones @ ones = dim per element, then matvec sums dim elements
    println!("MLP output[0] = {} (expected {})", out_host[0], expected);
    assert!(
        (out_host[0] - expected).abs() < 1e-1,
        "MLP output incorrect: got {}, expected {}",
        out_host[0],
        expected
    );

    // Verify ReLU: all values should be non-negative.
    assert!(
        out_host.iter().all(|&v| v >= 0.0),
        "ReLU output contains negative values"
    );
    println!("Data Parallel MLP passed\n");
    Ok(())
}

// ============================================================================
// Tutorial 9: Pointer Addition
// ============================================================================
#[cutile::module]
mod pointer_add_module {
    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        let tensor = make_tensor_view(ptr_tile, shape, strides, new_token_unordered());
        tensor
    }

    #[cutile::entry()]
    unsafe fn add_ptr<T: ElementType>(z_ptr: *mut T, x_ptr: *mut T, y_ptr: *mut T, len: i32) {
        let mut z_tensor: Tensor<T, { [-1] }> = get_tensor(z_ptr, len);
        let x_tensor: Tensor<T, { [-1] }> = get_tensor(x_ptr, len);
        let y_tensor: Tensor<T, { [-1] }> = get_tensor(y_ptr, len);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_shape = const_shape![4i32];
        let tile_x = x_tensor.partition(tile_shape).load([pid.0]);
        let tile_y = y_tensor.partition(tile_shape).load([pid.0]);
        z_tensor
            .partition_mut(tile_shape)
            .store(tile_x + tile_y, [pid.0]);
    }
}

fn test_pointer_addition(stream: &Arc<CudaStream>) -> Result<(), Error> {
    println!("=== Tutorial 9: Pointer Addition ===");
    use pointer_add_module::add_ptr;

    let len = 32usize;
    let tile_size = 4usize;

    let z: Tensor<f32> = zeros(&[len]).sync_on(stream)?;
    let x: Tensor<f32> = ones(&[len]).sync_on(stream)?;
    let y: Tensor<f32> = ones(&[len]).sync_on(stream)?;

    let z_ptr = z.device_pointer();
    let x_ptr = x.device_pointer();
    let y_ptr = y.device_pointer();

    let _ = unsafe { add_ptr(z_ptr, x_ptr, y_ptr, len as i32) }
        .grid(((len / tile_size) as u32, 1, 1))
        .sync_on(stream)?;

    let z_host: Vec<f32> = z.to_host_vec().sync_on(stream)?;

    for i in 0..5 {
        println!("1 + 1 = {} (expected 2.0)", z_host[i]);
        assert!(
            (z_host[i] - 2.0).abs() < 1e-5,
            "Pointer addition failed at index {}",
            i
        );
    }
    println!("Pointer Addition passed\n");
    Ok(())
}

// ============================================================================
// Main
// ============================================================================
fn main() -> Result<(), Error> {
    println!("cuTile Rust Book Examples - Verification Suite\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;

    test_hello_world(&stream)?;
    test_vector_addition(&stream)?;
    test_saxpy(&stream)?;
    test_gemm(&stream)?;
    test_softmax(&stream)?;
    test_flash_attention(&stream)?;
    test_async_composition(&stream)?;
    test_data_parallel_mlp(&stream)?;
    test_pointer_addition(&stream)?;

    println!("\nALL TESTS PASSED");
    Ok(())
}
