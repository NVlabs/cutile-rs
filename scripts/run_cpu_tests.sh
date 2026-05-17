#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running CPU tests"

run_step \
    "cutile-ir tests" \
    cargo test -p cutile-ir

run_step \
    "cutile-compiler CPU unit tests" \
    cargo test -p cutile-compiler --lib

run_step \
    "cutile-compiler doc tests" \
    cargo test -p cutile-compiler --doc

run_step \
    "cutile-compiler Tile IR to cubin lowering tests" \
    cargo test -p cutile-compiler --test compiler2_e2e

run_step \
    "cutile library tests" \
    cargo test -p cutile --lib

run_step \
    "cutile doc tests" \
    cargo test -p cutile --doc

for test_target in \
    basics_and_inlining \
    binary_math_ops \
    bitwise_and_bitcast_ops \
    compile_error_quality \
    compile_only \
    element_type_zero \
    element_type_zero_jit \
    error_quality \
    flash_attention_compile \
    global_memory \
    integer_ops \
    kernel_compiler \
    load_tile_like_examples \
    macro_submodule \
    memory_and_atomic_ops \
    num_tiles_jit \
    optimization_hints \
    partition_index_schedules \
    reduce_scan_ops \
    registry_phase_a \
    span_source_location \
    trait_dispatch_probe \
    two_cga_trait_impl \
    type_inference_sanity \
    unary_math_ops \
    use_classifier
do
    run_step \
        "cutile CPU integration test ${test_target}" \
        cargo test -p cutile --test "$test_target"
done

run_step \
    "cutile compile-only control-flow regressions" \
    cargo test -p cutile --test control_flow_ops compile_

run_step \
    "cutile compile-only tensor and matrix regressions" \
    cargo test -p cutile --test tensor_and_matrix_ops compile_

run_step \
    "cutile compile-only type conversion regressions" \
    cargo test -p cutile --test type_conversion_ops compile_

run_step \
    "cutile compile-only specialization regressions" \
    cargo test -p cutile --test specialization_bits -- --skip raw_pointer_launch

run_step \
    "cutile warmup/cache-key CPU tests" \
    cargo test -p cutile --test warmup

print_summary_and_exit \
    "All CPU tests passed!" \
    "Some CPU checks failed. See output above for details."
