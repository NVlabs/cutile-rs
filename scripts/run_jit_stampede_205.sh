#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One command to run the whole issue-205 cross-process JIT stampede measurement.
#
#   source ~/pr-e2e-20260918/env-cutile.sh   # CUDA_HOME + CUTILE_TILEIRAS_PATH + compat libcuda
#   scripts/run_jit_stampede_205.sh          # full matrix
#   scripts/run_jit_stampede_205.sh --trials 2 --p 1,2 --kernels add --modes shared
#
# Arguments are passed straight through to scripts/bench_jit_stampede.py; see
# its --help for the knobs (--trials, --p, --kernels, --modes, --devices, ...).
#
# The build happens here, outside every timed region: the coordinator only ever
# starts the already-built binary.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ -z "${CUTILE_TILEIRAS_PATH:-}" || -z "${CUDA_HOME:-}" ]]; then
  echo "error: source the task environment first (env-cutile.sh):" >&2
  echo "  source \"\$HOME/pr-e2e-20260918/env-cutile.sh\"" >&2
  exit 2
fi

# Never let a wrapper left over from an earlier run leak into the toolchain path.
if [[ "${CUTILE_TILEIRAS_PATH}" == *tileiras_wrap.sh ]]; then
  echo "error: CUTILE_TILEIRAS_PATH points at the wrapper; source env-cutile.sh again" >&2
  exit 2
fi

: "${CARGO_TARGET_DIR:=$HOME/pr-e2e-20260918/target-cutile}"
export CARGO_TARGET_DIR

echo "== building the worker (outside the timed region) =="
cargo build -p cutile-examples --example jit_stampede_worker

exec python3 "$REPO/scripts/bench_jit_stampede.py" "$@"
