/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tests for `CudaGraph::scope` — scoped CUDA graph capture.

use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::{value, DeviceOp};
use cuda_async::error::DeviceError;

fn has_gpu() -> bool {
    cuda_core::Device::device_count()
        .map(|n| n > 0)
        .unwrap_or(false)
}

fn on_fresh_thread<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::spawn(f).join().expect("test thread panicked");
}

#[test]
fn scope_empty_closure() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let graph = CudaGraph::scope(&stream, |_s| Ok(())).unwrap();
        graph.launch().sync_on(&stream).unwrap();
    });
}

#[test]
fn scope_records_value_ops() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let mut recorded = Vec::new();
        let graph = CudaGraph::scope(&stream, |s| {
            let a = s.record(value(42))?;
            let b = s.record(value("hello"))?;
            recorded.push(a);
            recorded.push(b.len() as i32);
            Ok(())
        })
        .unwrap();

        assert_eq!(recorded, vec![42, 5]);
        graph.launch().sync_on(&stream).unwrap();
    });
}

#[test]
fn scope_error_propagation() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let result = CudaGraph::scope(&stream, |_s| {
            Err(DeviceError::Internal("test error".into()))
        });

        assert!(result.is_err());
        match result {
            Err(DeviceError::Internal(msg)) => {
                assert!(
                    msg.contains("test error"),
                    "Expected test error, got: {msg}"
                );
            }
            Err(e) => panic!("Expected Internal error, got: {e}"),
            Ok(_) => panic!("Expected error, got Ok"),
        }
    });
}

#[test]
fn scope_panic_safety() {
    if !has_gpu() {
        return;
    }
    let result = std::thread::spawn(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CudaGraph::scope(&stream, |_s| {
                panic!("intentional panic in scope");
            })
        }));

        // Stream should still be usable after the panic.
        unsafe { stream.synchronize() }.unwrap();
    })
    .join();

    assert!(
        result.is_ok(),
        "Thread should not panic after scope cleanup"
    );
}

#[test]
fn scope_multiple_launches() {
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();

        let graph = CudaGraph::scope(&stream, |_s| Ok(())).unwrap();

        for _ in 0..10 {
            graph.launch().sync_on(&stream).unwrap();
        }
    });
}

#[test]
fn scope_nested_execution_rejected() {
    // Any attempt to execute a DeviceOp inside the scope closure
    // (via sync_on, sync, etc.) is rejected by the thread-local
    // execution lock — enforcing the invariant that only one
    // DeviceOp may be executing at a time per thread.
    if !has_gpu() {
        return;
    }
    on_fresh_thread(|| {
        let device = cuda_core::Device::new(0).unwrap();
        let stream = device.new_stream().unwrap();
        let other_stream = device.new_stream().unwrap();

        // sync_on capture stream — rejected by execution lock.
        let result = CudaGraph::scope(&stream, |_s| {
            let _ = value(42).sync_on(&stream)?;
            Ok(())
        });
        assert!(result.is_err(), "nested sync_on should fail");

        // sync_on other stream — also rejected by execution lock.
        let result = CudaGraph::scope(&stream, |_s| {
            let _ = value(42).sync_on(&other_stream)?;
            Ok(())
        });
        assert!(result.is_err(), "nested sync_on (other stream) should fail");

        // sync — also rejected.
        let result = CudaGraph::scope(&stream, |_s| {
            value(42).sync()?;
            Ok(())
        });
        assert!(result.is_err(), "nested sync should fail");
    });
}
