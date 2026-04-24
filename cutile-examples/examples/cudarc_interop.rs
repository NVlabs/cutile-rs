/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Borrows external CUDA handles into cutile via `Device::borrow_raw` and
//! `Stream::borrow_raw`.
//!
//! The example is structured around a simulated third-party library
//! (`foreign_cuda` below) that exposes its own `CUcontext` / `CUdevice` /
//! `CUstream` typedefs. These are nominally distinct from
//! `cuda_bindings`' typedefs — even though the underlying C ABI is
//! identical — which is exactly the situation a real cudarc, bindgen, or
//! hand-rolled-FFI caller finds themselves in.
//!
//! `borrow_raw` sidesteps the mismatch by taking `*mut c_void` + `c_int`,
//! so the caller casts foreign pointer typedefs to primitives once at the
//! boundary and keeps cutile binding-agnostic.
//!
//! The example demonstrates three properties:
//!   1. A `ForeignHandles` bundle typed against foreign typedefs can be
//!      turned into a cutile `Device`/`Stream` with only `as` casts.
//!   2. A borrowed `Stream` drives a cutile kernel via `sync_on`.
//!   3. Dropping the borrowed wrappers does NOT destroy the underlying
//!      handles — the `source_*` handles still work afterward.

use core::ffi::{c_int, c_void};
use cuda_core::{Device, Stream};
use cutile::error::Error;
use cutile::prelude::*;

/// Stand-in for a third-party CUDA bindings crate (cudarc, a newer
/// bindgen run of `cuda.h`, etc.). The opaque structs and typedefs are
/// intentionally distinct from `cuda_bindings::*` to prove that
/// `borrow_raw` doesn't require our binding-crate's typedefs at the
/// call site.
#[allow(non_camel_case_types)]
mod foreign_cuda {
    use core::ffi::c_int;

    pub enum CUctx_st {}
    pub enum CUstream_st {}

    pub type CUcontext = *mut CUctx_st;
    pub type CUdevice = c_int;
    pub type CUstream = *mut CUstream_st;

    /// What the third-party framework hands to its embedders.
    pub struct ForeignHandles {
        pub cu_ctx: CUcontext,
        pub cu_device: CUdevice,
        pub cu_stream: CUstream,
        pub ordinal: usize,
    }
}

#[cutile::module]
mod tile_add {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn add<const S: [i32; 1]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like_1d(x, z);
        let ty = load_tile_like_1d(y, z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Error> {
    const N: usize = 1024;
    const TILE: usize = 128;

    let source_device = Device::new(0)?;
    let source_stream = source_device.new_stream()?;

    // Pretend these came from a third-party framework. Pointer-to-pointer
    // and int-to-int casts are zero-cost; the types are nominally
    // different but ABI-identical.
    let foreign = foreign_cuda::ForeignHandles {
        cu_ctx: source_device.cu_ctx() as foreign_cuda::CUcontext,
        cu_device: source_device.cu_device() as foreign_cuda::CUdevice,
        cu_stream: source_stream.cu_stream() as foreign_cuda::CUstream,
        ordinal: source_device.ordinal(),
    };

    // TODO (hme): document safety — the foreign handles are derived from
    // `source_*` above, which outlive the borrowed wrappers.
    let borrowed_device = unsafe {
        Device::borrow_raw(
            foreign.cu_ctx as *mut c_void,
            foreign.cu_device as c_int,
            foreign.ordinal,
        )
    };
    let borrowed_stream =
        unsafe { Stream::borrow_raw(foreign.cu_stream as *mut c_void, &borrowed_device) };

    let x = api::ones::<f32>(&[N]).sync_on(&borrowed_stream)?;
    let y = api::ones::<f32>(&[N]).sync_on(&borrowed_stream)?;
    let z = api::zeros::<f32>(&[N]).sync_on(&borrowed_stream)?;

    let (z, _x, _y) = tile_add::add(z.partition([TILE]), x, y).sync_on(&borrowed_stream)?;
    let z = z.unpartition();
    let host = z.to_host_vec().sync_on(&borrowed_stream)?;
    assert!(host.iter().all(|v| *v == 2.0));

    drop(borrowed_stream);
    drop(borrowed_device);

    let probe = api::zeros::<f32>(&[4]).sync_on(&source_stream)?;
    let probe_host = probe.to_host_vec().sync_on(&source_stream)?;
    assert_eq!(probe_host, vec![0.0; 4]);

    println!("cudarc_interop: borrowed stream ran kernel; source handles still alive.");
    Ok(())
}
