/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Example: module-level device global memory.
 *
 * The load/store sequence below demonstrates ordered memory operations. It is
 * not an atomic counter increment under concurrent writers.
 *
 * Run with: cargo run -p cutile-examples --example global_memory
 */

use cutile::prelude::*;

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

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let mut first = api::zeros::<i32>(&[1]).sync_on(&stream)?;
    update_counter_ordered((&mut first).partition([1]))
        .grid((1, 1, 1))
        .sync_on(&stream)?;
    let first_host: Vec<i32> = first.dup().to_host_vec().sync_on(&stream)?;
    assert_eq!(first_host, vec![0]);

    let mut second = api::zeros::<i32>(&[1]).sync_on(&stream)?;
    update_counter_ordered((&mut second).partition([1]))
        .grid((1, 1, 1))
        .sync_on(&stream)?;
    let second_host: Vec<i32> = second.dup().to_host_vec().sync_on(&stream)?;
    assert_eq!(second_host, vec![1]);

    println!("Global counter old values from two launches: {first_host:?}, {second_host:?}");
    Ok(())
}
