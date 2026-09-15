/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Graph-capture input discipline, passing side: launchers over pre-allocated
//! inputs — `&Tensor`, `Arc<Tensor>`, `&TensorView`, and `&mut` partitions —
//! are `GraphNode`s and record into a `CudaGraph::scope`, and the replayed
//! graph computes into the pre-allocated buffers. The rejected side (an
//! allocating input op) is the compile-fail case in
//! `tests/ui/graph_scope_rejects_allocating_input.rs`.

use cutile::api;
use cutile::cuda_async::cuda_graph::CudaGraph;
use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod graph_scope_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        out: &mut Tensor<f32, { [B] }>,
        a: &Tensor<f32, { [-1] }>,
        b: &Tensor<f32, { [-1] }>,
    ) {
        let ta: Tile<f32, { [B] }> = load_tile_like(a, out);
        let tb: Tile<f32, { [B] }> = load_tile_like(b, out);
        out.store(ta + tb);
    }
}

use graph_scope_module::add;

#[test]
fn pre_allocated_inputs_record_and_replay() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        let a = api::arange::<f32>(8).sync_on(&stream).expect("a");
        let ones = Arc::new(api::ones::<f32>(&[8]).sync_on(&stream).expect("ones"));
        let mut out = api::zeros::<f32>(&[8]).sync_on(&stream).expect("out");
        let mut out2 = api::zeros::<f32>(&[8]).sync_on(&stream).expect("out2");

        let mut graph = CudaGraph::scope(&stream, |s| {
            // `&mut` partition output, borrowed `&Tensor` and `Arc<Tensor>` inputs.
            s.record(add((&mut out).partition([8]), &a, ones.clone()))?;
            // A `&TensorView` input over a buffer written by the previous node.
            let view = out.view(&[8])?;
            s.record(add((&mut out2).partition([8]), &view, &a))?;
            Ok(())
        })
        .expect("scope capture");

        // Capture records; replay computes.
        graph.launch().sync_on(&stream).expect("graph launch");
        // The graph borrows the captured buffers for as long as it lives;
        // drop it to read them from the host.
        drop(graph);

        let host: Vec<f32> = out.to_host_vec().sync_on(&stream).expect("out host");
        let expected: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        assert_eq!(host, expected, "out = arange + 1");

        let host2: Vec<f32> = out2.to_host_vec().sync_on(&stream).expect("out2 host");
        let expected2: Vec<f32> = (0..8).map(|i| 2.0 * i as f32 + 1.0).collect();
        assert_eq!(host2, expected2, "out2 = out + arange");
    });
}

/// The closure-owns-the-buffers pattern: the scope moves the buffers in and
/// returns them as the graph's output `T`, giving a `CudaGraph<'static, _>`.
/// Replays accumulate (out += a each launch, since out is also an input via
/// a view), outputs are read through `replay`/`outputs`, and `into_inner`
/// synchronizes and hands the buffers back.
#[test]
fn owned_outputs_replay_twice_and_into_inner() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        let a = api::arange::<f32>(8).sync_on(&stream).expect("a");
        let out = api::zeros::<f32>(&[8]).sync_on(&stream).expect("out");
        let tmp = api::zeros::<f32>(&[8]).sync_on(&stream).expect("tmp");

        let mut graph = CudaGraph::scope(&stream, move |s| {
            let mut out = out;
            let mut tmp = tmp;
            // tmp = out + a; out = tmp — so each replay accumulates
            // out += a.
            s.record(add((&mut tmp).partition([8]), &out, &a))?;
            s.record(api::memcpy(&mut out, &tmp))?;
            // Every buffer the graph touches must be returned as part of
            // `T` (or borrowed from outside): a closure-owned buffer that
            // is dropped here would be freed while the graph still holds
            // its device address.
            Ok((out, a, tmp))
        })
        .expect("scope capture");

        // Two sequential replays; `replay` synchronizes and hands the
        // outputs back.
        graph.replay().expect("first replay");
        let bufs = graph.replay().expect("second replay");
        let host: Vec<f32> = bufs
            .0
            .dup()
            .to_host_vec()
            .sync_on(&stream)
            .expect("host after two replays");
        let expected: Vec<f32> = (0..8).map(|i| 2.0 * i as f32).collect();
        assert_eq!(host, expected, "two replays accumulate out = 2*a");

        // An async replay completes before `outputs` hands the buffers out.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async {
            graph.launch().await.expect("async replay");
        });
        let host: Vec<f32> = graph
            .outputs()
            .expect("outputs after async replay")
            .0
            .dup()
            .to_host_vec()
            .sync_on(&stream)
            .expect("host after async replay");
        let expected: Vec<f32> = (0..8).map(|i| 3.0 * i as f32).collect();
        assert_eq!(host, expected, "third (async) replay accumulates");

        // `into_inner` synchronizes and returns ownership of the buffers.
        let (out, a, tmp) = graph.into_inner().expect("into_inner");
        drop((a, tmp));
        let host: Vec<f32> = out.to_host_vec().sync_on(&stream).expect("final host");
        assert_eq!(host, expected, "into_inner returns the buffers intact");
    });
}
