/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{Device, DeviceAttribute};

fn has_gpu() -> bool {
    Device::device_count().map(|n| n > 0).unwrap_or(false)
}

#[test]
fn queries_device_attributes() {
    if !has_gpu() {
        return;
    }

    let device = Device::new(0).unwrap();
    let attributes = [
        DeviceAttribute::ComputeCapabilityMajor,
        DeviceAttribute::ComputeCapabilityMinor,
        DeviceAttribute::MultiprocessorCount,
        DeviceAttribute::MemoryClockRate,
        DeviceAttribute::GlobalMemoryBusWidth,
        DeviceAttribute::L2CacheSize,
        DeviceAttribute::MaxPersistingL2CacheSize,
        DeviceAttribute::MaxAccessPolicyWindowSize,
        DeviceAttribute::MaxRegistersPerMultiprocessor,
        DeviceAttribute::MaxThreadsPerMultiprocessor,
        DeviceAttribute::MaxBlocksPerMultiprocessor,
        DeviceAttribute::MaxSharedMemoryPerMultiprocessor,
        DeviceAttribute::ClockRate,
        DeviceAttribute::WarpSize,
    ];

    for attribute in attributes {
        assert!(
            device.attribute(attribute).unwrap() >= 0,
            "{attribute:?} must not be negative"
        );
    }

    assert!(
        device
            .attribute(DeviceAttribute::ComputeCapabilityMajor)
            .unwrap()
            > 0
    );
    assert!(
        device
            .attribute(DeviceAttribute::MultiprocessorCount)
            .unwrap()
            > 0
    );
    assert!(device.attribute(DeviceAttribute::WarpSize).unwrap() > 0);
    assert_eq!(
        device.l2_cache_size_bytes().unwrap(),
        device.attribute(DeviceAttribute::L2CacheSize).unwrap() as usize
    );
}
