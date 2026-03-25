#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test_runner_common.sh"

print_header "Running CPU tests"

run_step \
    "cuda-tile-rs tests" \
    cargo test -p cuda-tile-rs --quiet

run_step \
    "cutile-compiler CPU unit tests" \
    cargo test -p cutile-compiler --lib --quiet

run_step \
    "cutile-compiler doc tests" \
    cargo test -p cutile-compiler --doc --quiet

run_step \
    "cutile library tests" \
    cargo test -p cutile --lib --quiet

print_summary_and_exit \
    "All CPU tests passed!" \
    "Some CPU checks failed. See output above for details."
