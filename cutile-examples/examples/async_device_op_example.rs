/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_async::device_operation;
use cuda_async::device_operation::*;
use cuda_async::error::DeviceError;
use cuda_core::*;
use cutile::tile_kernel::global_policy;

#[tokio::main]
async fn main() -> Result<(), DeviceError> {
    // Create a scheduler, which schedules an instance of DeviceOperation on a device stream.
    // The schedule operation produces an implementation of Future.
    let _policy = global_policy(0);

    let num_elements: usize = 2usize.pow(12);
    let num_bytes = 4 * num_elements;
    let x_host_1 = vec![1u32; num_elements];
    let y_host_1 = vec![1u32; num_elements];
    let x_op = device_operation::with_context(|ctx: &ExecutionContext| unsafe {
        let dptr = malloc_async(num_bytes, ctx.get_cuda_stream());
        memcpy_htod_async(dptr, x_host_1.as_ptr(), num_elements, ctx.get_cuda_stream());
        Value::new(dptr)
    });
    let y_op = device_operation::with_context(|ctx| unsafe {
        let dptr = malloc_async(num_bytes, ctx.get_cuda_stream());
        memcpy_htod_async(dptr, y_host_1.as_ptr(), num_elements, ctx.get_cuda_stream());
        Value::new(dptr)
    });

    let op = zip!(x_op, y_op).and_then_with_context(|ctx, (x_dptr, y_dptr)| {
        // This is safe: The operation can only be safely executed via a DeviceFuture,
        // which guarantees all asynchronous operations are complete when it is ready.
        let mut x_host = vec![0u32; num_elements];
        let mut y_host = vec![0u32; num_elements];
        // This part is of course not safe.
        unsafe {
            memcpy_dtoh_async(
                x_host.as_mut_ptr(),
                x_dptr,
                num_elements,
                ctx.get_cuda_stream(),
            );
            memcpy_dtoh_async(
                y_host.as_mut_ptr(),
                y_dptr,
                num_elements,
                ctx.get_cuda_stream(),
            );
        }
        Value::new(((x_host, y_host), (x_dptr, y_dptr)))
    });

    let ((x_host_2, y_host_2), dptrs) = op.await?;

    assert!(x_host_1.iter().all(|&v| v == 1));
    println!("x_host_2 of len {:?} ok", x_host_2.len());
    assert!(y_host_2.iter().all(|&v| v == 1));
    println!("y_host_2 of len {:?} ok", y_host_2.len());

    // Drop.
    Value::new(dptrs)
        .and_then_with_context(|ctx, (x_dptr, y_dptr)| {
            unsafe {
                free_async(x_dptr, ctx.get_cuda_stream());
                free_async(y_dptr, ctx.get_cuda_stream());
            }
            Value::new(())
        })
        .await?;
    Ok(())
}
