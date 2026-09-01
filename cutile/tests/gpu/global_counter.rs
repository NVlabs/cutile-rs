/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test: module-level `Global<i32>` with ordered load/store.
//!
//! The sequence is ordered (Acquire/Release) but not atomic — verifies the
//! load/store path, not concurrent-counter semantics.

use cutile::prelude::*;

use crate::common;

#[cutile::module]
mod global_kernels {
    use cutile::core::*;

    static COUNTER: Global<i32, { [] }> = Global::new(0i32);

    #[cutile::entry()]
    fn update_counter_ordered(out: &mut Tensor<i32, { [1] }>) {
        let (old_value, _load_token) = COUNTER.load(ordering::Acquire, scope::Device);
        let next_value = old_value + constant(1i32, const_shape![]);
        let _store_token = COUNTER.store(next_value, ordering::Release, scope::Device);
        out.store(old_value.reshape(const_shape![1]));
    }
}

use global_kernels::update_counter_ordered;

#[test]
fn smoke_global_counter_ordered() {
    common::with_test_stack(|| {
        let device = cuda_core::Device::new(0).expect("device");
        let stream = device.new_stream().expect("stream");

        let mut first = api::zeros::<i32>(&[1]).sync_on(&stream).expect("zeros");
        update_counter_ordered((&mut first).partition([1]))
            .grid((1, 1, 1))
            .sync_on(&stream)
            .expect("first launch");
        let first_host: Vec<i32> = first
            .dup()
            .to_host_vec()
            .sync_on(&stream)
            .expect("first to_host");
        assert_eq!(first_host, vec![0]);

        let mut second = api::zeros::<i32>(&[1]).sync_on(&stream).expect("zeros");
        update_counter_ordered((&mut second).partition([1]))
            .grid((1, 1, 1))
            .sync_on(&stream)
            .expect("second launch");
        let second_host: Vec<i32> = second
            .dup()
            .to_host_vec()
            .sync_on(&stream)
            .expect("second to_host");
        assert_eq!(second_host, vec![1]);
    });
}
