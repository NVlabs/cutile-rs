#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pass-through `tileiras` wrapper for the issue-205 JIT-stampede harness.
#
# Put this script in `CUTILE_TILEIRAS_PATH`, point `CUTILE_TILEIRAS_REAL` at the
# real compiler, and set `CUTILE_TILEIRAS_WRAP_LOG` to a per-worker file. It
# records one `B` line per invocation (before) and one `E` line per invocation
# (after, with the child's exit status and the produced cubin size), then exits
# with the child's status.
#
# This is deliberately *the same wrapper in every arm*: it is the only way the
# harness gets a real spawn-attempt count, because the runtime's own
# `jit_backend_compile_count` is incremented only after a non-empty cubin
# exists and therefore is not a spawn counter.
#
# `--version` is passed through to the real binary (nothing is added to stdout),
# so `tileiras_fingerprint()` -- which hashes `--version` output and feeds the L2
# key -- is unaffected and identical to the unwrapped fingerprint. Version
# probes are tagged `version` in the log and are never counted as compiles.
#
# Log line format (one short line, single O_APPEND write, so concurrent writers
# from the ranks sharing a cache never interleave):
#
#   B <epoch_real_seconds> <pid> <version|probe|compile> <argv...>
#   E <epoch_real_seconds> <pid> rc=<status> bytes=<cubin bytes or -1>
set -u

log="${CUTILE_TILEIRAS_WRAP_LOG:-}"
real="${CUTILE_TILEIRAS_REAL:-}"

if [ -z "$real" ]; then
  echo "tileiras_wrap.sh: CUTILE_TILEIRAS_REAL is not set" >&2
  exit 127
fi

# Three spawn classes are visible in this log and they must not be conflated:
#   version - the `--version` fingerprint probe the cache key needs (never a compile)
#   probe   - the runtime's bytecode-version capability probe: it also runs
#             tileiras on a tiny synthetic module, and it is recognisable because
#             it carries no `--opt-level` (only run_tileiras passes that)
#   compile - a real stage-2 compile of a kernel's Tile IR
kind=compile
out=""
prev=""
seen_opt_level=0
for a in "$@"; do
  case "$a" in
    --version) kind=version ;;
    --opt-level) seen_opt_level=1 ;;
  esac
  if [ "$prev" = "-o" ]; then out="$a"; fi
  prev="$a"
done
if [ "$kind" = compile ] && [ "$seen_opt_level" = 0 ]; then kind=probe; fi

if [ -n "$log" ]; then
  printf 'B %s %s %s %s\n' "${EPOCHREALTIME:-0}" "$$" "$kind" "$*" >> "$log"
fi

# Run as a child rather than `exec` so the exit status and the cubin size can be
# recorded. POSIX folds a waited-for child's own reaped children into its
# parent's RUSAGE_CHILDREN, so the harness still sees the compiler's full CPU
# time through the Rust parent.
"$real" "$@"
rc=$?

if [ -n "$log" ]; then
  bytes=-1
  if [ -n "$out" ] && [ -f "$out" ]; then
    bytes=$(stat -c %s "$out" 2>/dev/null || echo -1)
  fi
  printf 'E %s %s rc=%s bytes=%s\n' "${EPOCHREALTIME:-0}" "$$" "$rc" "$bytes" >> "$log"
fi

exit $rc
