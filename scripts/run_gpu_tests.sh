#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running GPU tests, examples, and benchmarks"

run_step \
    "cutile-compiler GPU runtime tests" \
    cargo test -p cutile-compiler --test gpu --quiet

run_step \
    "cutile doc tests" \
    cargo test -p cutile --doc --quiet

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
    type_conversion_ops \
    unary_math_ops
do
    run_step \
        "cutile GPU integration test ${test_target}" \
        cargo test -p cutile --test "$test_target" --quiet
done

run_step \
    "cutile GPU error-quality tests" \
    cargo test -p cutile --test gpu --quiet

echo -e "${YELLOW}>>> Running cutile-examples (GPU)${NC}"
cd "$REPO_ROOT/cutile-examples" || exit 1
run_examples "$REPO_ROOT/cutile-examples/examples"

echo -e "${YELLOW}>>> Running cutile-benchmarks (GPU)${NC}"
cd "$REPO_ROOT/cutile-benchmarks" || exit 1
run_benches "$REPO_ROOT/cutile-benchmarks/benches"

print_summary_and_exit \
    "All GPU tests, examples, and benchmarks passed!" \
    "Some GPU checks failed. See output above for details."
