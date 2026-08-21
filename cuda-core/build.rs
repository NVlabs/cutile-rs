/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The `simt::vmm` multicast surface is gated on `cuda_has_multicast`, which
//! cuda-oxide's build script probes the toolkit headers for. This crate's
//! minimum supported toolkit (13.2) always carries the multicast driver API
//! (introduced in 12.1), so the cfg is emitted unconditionally.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_has_multicast)");
    println!("cargo:rustc-cfg=cuda_has_multicast");
}
