/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `branded_*` families measure the deprecated annotation path against
// the plain one; that is the point of the corpus.
#![allow(deprecated)]

//! The check-placement ratchet: a fixed corpus of kernels, one per access
//! family, with EXACT placement counts pinned per kernel.
//!
//! This is the merge gate for the check-framework unification (phases A-D of
//! the plan approved 2026-08-04): across any refactor of the checkers, a
//! kernel's `(discharged, hoisted, in_place)` triple and its launch-check
//! count may only improve — `in_place` may move to `hoisted`/launch,
//! `hoisted` may move to `discharged`/launch, and nothing may regress toward
//! the device. A count change in the good direction is re-pinned consciously
//! in the same commit that causes it; a change in the bad direction fails
//! this suite.
//!
//! Keep this corpus FROZEN: it is a measurement instrument, not a feature
//! test. New behavior gets new tests elsewhere.

use cutile;
use cutile_compiler::compiler::utils::CompileOptions;

mod common;

#[cutile::module]
mod ratchet_module {
    use cutile::core::*;

    /// Family: branded loop over a `with_bounds` binding, static extents.
    /// Everything discharges at JIT.
    #[cutile::entry]
    fn branded_static<const N: i32, const B: i32>(out: &mut Tensor<f32, { [1, N] }>) {
        let cols = Dim::new(N / B);
        let mut p = out
            .partition_mut(const_shape![1, B])
            .with_bounds((Dim::new(1), cols));
        for j in cols {
            let t: Tile<f32, { [1, B] }> = constant(0.0, const_shape![1, B]);
            p.store(t, coord((0i32, j)));
        }
    }

    /// Family: branded loop over dynamic extents; bindings reduce to
    /// divisibility launch checks, accesses discharge by brand.
    #[cutile::entry]
    fn branded_dynamic<const BM: i32, const BN: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = Dim::new(x.shape()[0] / BM);
        let n = Dim::new(x.shape()[1] / BN);
        let p = x.partition(const_shape![BM, BN]).with_bounds((m, n));
        for i in m {
            for j in n {
                let t = p.load(coord((i, j)));
                z.store(t);
            }
        }
    }

    /// Family: plain partition, affine loop index, static extents. The
    /// constant/static rung discharges everything.
    #[cutile::entry]
    fn plain_static_loop<const B: i32>(z: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [256] }>) {
        let p = x.partition(const_shape![B]);
        for j in 0i32..4i32 {
            let t = p.load([j]);
            z.store(t);
        }
    }

    /// Family: plain partition, affine loop index, dynamic extent. The check
    /// hoists to the loop preheader (upper), lower discharges statically.
    #[cutile::entry]
    fn plain_dynamic_loop<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        n: i32,
    ) {
        let p = x.partition(const_shape![B]);
        for j in 0i32..n {
            let t = p.load([j]);
            z.store(t);
        }
    }

    /// Family: plain partition, raw runtime scalar, dynamic extent. Fully
    /// in-place check, both directions.
    #[cutile::entry]
    fn plain_runtime_scalar<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        idx: i32,
    ) {
        let p = x.partition(const_shape![B]);
        let t = p.load([idx]);
        z.store(t);
    }

    /// Family: mapped partition streaming a GEMM, foreign mapped components
    /// into plain partitions, discharged by declared equalities.
    #[cutile::entry(
        preconditions = (
            dim(z, 0) == dim(x, 0),
            dim(z, 1) == dim(y, 1),
        )
    )]
    fn mapped_gemm<const BM: i32, const BN: i32, const BK: i32, const MAP_SHAPE: [i32; 2]>(
        mut z: MappedPartitionMut<f32, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let py = y.partition(const_shape![BK, BN]);
        for index in z.iter_indices() {
            let (bid_m, bid_n) = index.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0, const_shape![BM, BN]);
            for k in 0i32..4i32 {
                let tx = px.load([bid_m, k]);
                let ty = py.load([k, bid_n]);
                acc = mma(tx, ty, acc);
            }
            z.store(acc, index);
        }
    }
}

use access_counts::pin;
use ratchet_module::__module_ast_self;

mod access_counts {
    use super::*;
    use cutile_compiler::compile_api::CheckPlacementCounts;

    pub struct Pinned {
        pub name: &'static str,
        pub generics: &'static [&'static str],
        pub strides: &'static [(&'static str, &'static [i32])],
        pub grid: Option<(u32, u32, u32)>,
        /// (discharged, hoisted, in_place, launch_checks)
        pub counts: (u32, u32, u32, usize),
    }

    pub fn pin(p: &Pinned) {
        let generics: Vec<String> = p.generics.iter().map(|s| s.to_string()).collect();
        let strides: Vec<(&str, &[i32])> = p.strides.to_vec();
        let mut compiler = cutile_compiler::compile_api::KernelCompiler::new(
            __module_ast_self,
            "ratchet_module",
            p.name,
        )
        .target("sm_120")
        .generics(generics)
        .strides(&strides)
        .options(CompileOptions::default());
        if let Some(grid) = p.grid {
            compiler = compiler.grid(grid);
        }
        let artifacts = compiler
            .compile()
            .unwrap_or_else(|e| panic!("compile {}: {e}", p.name));
        let CheckPlacementCounts {
            discharged,
            hoisted,
            in_place,
        } = artifacts.check_counts();
        let got = (
            discharged,
            hoisted,
            in_place,
            artifacts.launch_checks().len(),
        );
        assert_eq!(
            got, p.counts,
            "{}: placement counts moved (discharged, hoisted, in_place, launch). \
             If they moved AWAY from the device, re-pin consciously in the same \
             commit; if they moved TOWARD it, this is a regression.",
            p.name
        );
    }
}

#[test]
fn ratchet_branded_static() {
    common::with_test_stack(|| {
        pin(&access_counts::Pinned {
            name: "branded_static",
            generics: &["256", "64"],
            strides: &[("out", &[256, 1])],
            grid: None,
            counts: (2, 0, 0, 0),
        });
    });
}

#[test]
fn ratchet_branded_dynamic() {
    common::with_test_stack(|| {
        pin(&access_counts::Pinned {
            name: "branded_dynamic",
            generics: &["64", "64"],
            strides: &[("z", &[64, 1]), ("x", &[64, 1])],
            grid: None,
            counts: (2, 0, 0, 2),
        });
    });
}

#[test]
fn ratchet_plain_static_loop() {
    common::with_test_stack(|| {
        pin(&access_counts::Pinned {
            name: "plain_static_loop",
            generics: &["64"],
            strides: &[("z", &[1]), ("x", &[1])],
            grid: None,
            counts: (1, 0, 0, 0),
        });
    });
}

#[test]
fn ratchet_plain_dynamic_loop() {
    common::with_test_stack(|| {
        pin(&access_counts::Pinned {
            name: "plain_dynamic_loop",
            generics: &["64"],
            strides: &[("z", &[1]), ("x", &[1])],
            grid: None,
            counts: (0, 1, 0, 0),
        });
    });
}

#[test]
fn ratchet_plain_runtime_scalar() {
    common::with_test_stack(|| {
        pin(&access_counts::Pinned {
            name: "plain_runtime_scalar",
            generics: &["64"],
            strides: &[("z", &[1]), ("x", &[1])],
            grid: None,
            counts: (0, 0, 1, 0),
        });
    });
}

#[test]
fn ratchet_mapped_gemm() {
    common::with_test_stack(|| {
        pin(&access_counts::Pinned {
            name: "mapped_gemm",
            generics: &["32", "32", "32", "4", "4"],
            strides: &[("z", &[256, 1]), ("x", &[256, 1]), ("y", &[256, 1])],
            grid: Some((16, 1, 1)),
            counts: (2, 2, 0, 0),
        });
    });
}
