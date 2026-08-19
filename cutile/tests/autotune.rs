/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU integration test: a real (tiny) tuning run end to end.

use cuda_core::Device;
use cutile::bench::BenchOptions;
use cutile::prelude::*;
use cutile::tune::{Autotuner, Config, Outcome, ParamValue};
use std::time::Duration;

fn quick_bench() -> BenchOptions {
    BenchOptions {
        warmup: Duration::from_millis(5),
        rep: Duration::from_millis(15),
        min_reps: 3,
        max_reps: 20,
        clear_l2: true,
    }
}

/// Tunes a fill "kernel" over candidate sizes; one candidate is invalid by
/// construction. Exercises: gate rejection recorded, all candidates visited,
/// a winner produced, JSONL log written and resumable.
#[test]
fn end_to_end_grid_tune_on_gpu() {
    let device = Device::new(0).expect("device 0");
    let stream = device.new_stream().expect("stream");

    let configs: Vec<Config> = [1usize << 16, 1 << 20, 0]
        .into_iter()
        .map(|n| Config::new([("N", ParamValue::Int(n as i64))]))
        .collect();

    let log = std::env::temp_dir().join(format!("cutile_autotune_test_{}", std::process::id()));
    let _ = std::fs::remove_file(&log);

    let run = |log_path: &std::path::Path| {
        Autotuner::new("fill_tune")
            .configs(configs.clone())
            .bench(quick_bench())
            .log(log_path)
            .run(&stream, |_, config| {
                let n = config.int("N").unwrap() as usize;
                if n == 0 {
                    return Err(cutile::error::Error::Tensor(cutile::error::TensorError(
                        "empty shape is not tunable".into(),
                    )));
                }
                // Correctness gate: run once, verify on host.
                let t = api::full(3.0f32, &[n])
                    .sync_on(&stream)
                    .map_err(cutile::error::Error::from)?;
                let host = t
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(cutile::error::Error::from)?;
                if host[0] != 3.0 {
                    return Err(cutile::error::Error::Tensor(cutile::error::TensorError(
                        "gate: wrong fill value".into(),
                    )));
                }
                Ok(move |s: &std::sync::Arc<cuda_core::Stream>| {
                    api::full(3.0f32, &[n])
                        .sync_on(s)
                        .map(|_| ())
                        .map_err(Into::into)
                })
            })
            .expect("tuning run")
    };

    let outcome = run(&log);
    assert_eq!(outcome.trials.len(), 3, "all candidates visited");
    let invalid: Vec<_> = outcome
        .trials
        .iter()
        .filter(|t| matches!(t.outcome, Outcome::Invalid { .. }))
        .collect();
    assert_eq!(invalid.len(), 1, "the N=0 candidate is invalid");
    assert!(invalid[0].config_id.contains("N=0"));
    let best = outcome.best.expect("a winner");
    assert_eq!(
        best.int("N"),
        Some(1 << 16),
        "smaller fill should be faster"
    );

    // Resume: a second run over the same log re-measures nothing.
    let resumed = run(&log);
    assert_eq!(
        resumed.trials.len(),
        3,
        "resumed run still reports all trials"
    );
    let _ = std::fs::remove_file(&log);
}
