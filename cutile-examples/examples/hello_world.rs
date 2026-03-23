/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![allow(unused_variables)]

use cuda_async::device_operation::DeviceOperation;
use cuda_core::CudaContext;
use cutile::error::Error;
use cutile::tile_kernel::TileKernel;

#[cutile::module]
mod hello_world_module {

    use cutile::core::*;

    #[cutile::entry(print_ir = true)]
    fn hello_world_kernel() {
        let pids: (i32, i32, i32) = get_tile_block_id();
        let npids: (i32, i32, i32) = get_num_tile_blocks();
        cuda_tile_print!(
            "Hello, I am tile <{}, {}, {}> in a kernel with <{}, {}, {}> tiles.\n",
            pids.0,
            pids.1,
            pids.2,
            npids.0,
            npids.1,
            npids.2
        );
    }
}

use hello_world_module::hello_world_kernel;

fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let launcher = hello_world_kernel();
    launcher.grid((1, 1, 1)).sync_on(&stream)?;
    Ok(())
}
