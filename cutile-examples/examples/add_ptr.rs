/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cutile::api::{arange, ones, zeros};
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use std::future::IntoFuture;

#[cutile::module]
mod my_module {

    use cutile::core::*;

    unsafe fn get_tensor<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
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

use my_module::add_ptr_sync;

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() -> Result<(), cutile::error::Error> {
    let len = 2usize.pow(5);
    let tile_size = 4usize;

    // Initialize tensors.
    let z = zeros::<1, f32>([len]).await?;
    let x: Tensor<f32> = arange(len).await?;
    let y: Tensor<f32> = ones([len]).await?;

    // Extract device pointers.
    let z_ptr = z.device_pointer();
    let x_ptr = x.device_pointer();
    let y_ptr = y.device_pointer();

    // Prepare kernel launch. Note that, since we're passing in pointers, unsafe is required.
    let op = unsafe { add_ptr_sync(z_ptr, x_ptr, y_ptr, len as i32) }.grid((
        (len / tile_size) as u32,
        1,
        1,
    ));

    // Spawn an asynchronous task to compute this operation.
    let op_handle = tokio::spawn(op.into_future());

    // Note that, while the device operates on these tensors, the programmer is not protected from UB!
    // We do not need the results. Device pointers are Copy (they are copied to device memory).
    let _ = op_handle.await.expect("Failed to execute tokio task.")?;

    let x_host = x.to_host_vec().await?;
    let y_host = y.to_host_vec().await?;
    let z_host = z.to_host_vec().await?;
    for i in 0..z_host.len() {
        let x_i: f32 = x_host[i];
        let y_i: f32 = y_host[i];
        let answer = x_i + y_i;
        println!("{} + {} = {}", x_i, y_i, z_host[i]);
        assert_eq!(answer, z_host[i], "{} != {} ?", answer, z_host[i]);
    }
    Ok(())
}
