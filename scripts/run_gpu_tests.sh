#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running GPU tests"

for test_target in \
    arange \
    autotune \
    do_bench \
    dtype_float_ops \
    gpu_execution_ops \
    nested_partition_mut \
    slice_non_divisible \
    tensor_reinterpret \
    tensor_views
do
    run_step \
        "cutile GPU integration test ${test_target}" \
        cargo test -p cutile --features experimental-tune --test "$test_target"
done

run_step \
    "cutile GPU integration test control_flow_ops runtime cases" \
    cargo test -p cutile --test control_flow_ops -- --skip compile_

run_step \
    "cutile GPU integration test tensor_and_matrix_ops runtime cases" \
    cargo test -p cutile --test tensor_and_matrix_ops execute_

run_step \
    "cutile GPU integration test type_conversion_ops runtime cases" \
    cargo test -p cutile --test type_conversion_ops execute_

run_step \
    "cutile GPU integration test specialization_bits runtime cases" \
    cargo test -p cutile --test specialization_bits raw_pointer_launch

run_step \
    "cutile GPU warmup and disk-cache tests" \
    cargo test -p cutile --test warmup_suite

run_step \
    "cutile GPU aggregate tests" \
    cargo test -p cutile --test gpu

run_step \
    "cuda-core GPU integration test vmm" \
    cargo test -p cuda-core --test vmm

run_step \
    "cuda-core GPU integration test vmm_multicast" \
    cargo test -p cuda-core --test vmm_multicast

# The SIMT integration tests carried over from cuda-oxide. simt_vmm_p2p
# exercises the real two-GPU P2P path (it self-skips on single-device
# machines or when peer access is unavailable); simt_vmm_multicast
# self-skips without NVLink switch multicast support.
for test_target in \
    borrow_raw \
    simt_context_limits \
    simt_context_sync_policy \
    simt_device_buffer \
    simt_device_buffer_leaks \
    simt_pinned_host_buffer \
    simt_stream_priority \
    simt_stream_query \
    simt_vmm_multicast \
    simt_vmm_p2p
do
    run_step \
        "cuda-core GPU integration test ${test_target}" \
        cargo test -p cuda-core --test "$test_target"
done

run_step \
    "cuda-async unit tests (with live driver)" \
    cargo test -p cuda-async --lib

# Unfiltered: includes the GPU-requiring `cuda_tests` module that the CPU
# path skips.
run_step \
    "cuda-bindings tests (with live driver)" \
    cargo test -p cuda-bindings

for test_target in \
    concurrent_capture \
    cuda_graph \
    execute_once \
    pool_allocation \
    reactor_correctness
do
    run_step \
        "cuda-async GPU integration test ${test_target}" \
        cargo test -p cuda-async --test "$test_target"
done

print_summary_and_exit \
    "All GPU tests passed!" \
    "Some GPU tests failed. See output above for details."
