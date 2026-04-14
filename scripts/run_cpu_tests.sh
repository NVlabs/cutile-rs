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
    "cutile library tests" \
    cargo test -p cutile --lib

run_step \
    cargo test -p cutile --test warmup

run_step \
    cargo test -p cuda-async --test jit_store

print_summary_and_exit \
    "All CPU tests passed!" \
    "Some CPU checks failed. See output above for details."
