#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $(basename "$0") <command> [args...]" >&2
  exit 2
fi

GPU_INDEX="${GPU_INDEX:-0}"
GPU_CLOCK_MHZ="${GPU_CLOCK_MHZ:-2400}"

echo "=== Lock GPU ${GPU_INDEX} clocks @ ${GPU_CLOCK_MHZ} MHz ==="
sudo nvidia-smi -i "$GPU_INDEX" -lgc "$GPU_CLOCK_MHZ,$GPU_CLOCK_MHZ"

cleanup() {
  echo "=== Release GPU ${GPU_INDEX} clocks ==="
  sudo nvidia-smi -i "$GPU_INDEX" -rgc || true
}
trap cleanup EXIT

status=0
"$@" || status=$?
exit "$status"
