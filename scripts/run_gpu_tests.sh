#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running GPU tests"

run_step \
    "cutile error-quality tests" \
    cargo test -p cutile --test error_quality

run_step \
    "cutile-compiler GPU runtime tests" \
    cargo test -p cutile-compiler --test gpu

run_step \
    "cutile doc tests" \
    cargo test -p cutile --doc

for test_target in \
    basics_and_inlining \
    binary_math_ops \
    bitwise_and_bitcast_ops \
    control_flow_ops \
    integer_ops \
    memory_and_atomic_ops \
    reduce_scan_ops \
    span_source_location \
    tensor_and_matrix_ops \
    tensor_reinterpret \
    tensor_views \
    type_conversion_ops \
    unary_math_ops
do
    run_step \
        "cutile GPU integration test ${test_target}" \
        cargo test -p cutile --test "$test_target"
done

run_step \
    "cutile GPU error-quality tests" \
    cargo test -p cutile --test gpu

print_summary_and_exit \
    "All GPU tests passed!" \
    "Some GPU tests failed. See output above for details."
