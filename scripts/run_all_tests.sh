# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/run_cpu_tests.sh"
CPU_STATUS=$?

"$SCRIPT_DIR/run_gpu_tests.sh"
GPU_STATUS=$?

if [[ $CPU_STATUS -eq 0 && $GPU_STATUS -eq 0 ]]; then
    exit 0
fi

exit 1
