#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ad-hoc analysis helpers over the raw stampede records (no invented numbers).

    python3 scripts/jit_stampede_probe.py <raw-dir> [--phase measure] [--kernel add] [--p 8]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = [
    "barrier_to_launch_ms",
    "relaunch_min_ms",
    "jit_attributable_ms",
    "wrapper_compile_ms",
    "cpu_user_ms",
    "cpu_system_ms",
    "disk_hits_delta",
    "disk_misses_delta",
    "disk_puts_delta",
    "disk_io_errors_delta",
    "backend_attempts",
    "setup_ms",
    "idle_before_release_ms",
]


def load(raw: Path, tag: str = ""):
    suffix = f".{tag}" if tag else ""
    workers, rounds = [], []
    with (raw / f"workers{suffix}.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                workers.append(json.loads(line))
    with (raw / f"rounds{suffix}.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                rounds.append(json.loads(line))
    return workers, rounds


def main() -> int:
    raw = Path(sys.argv[1])
    args = sys.argv[2:]
    phase = "measure"
    kernel = None
    p = None
    if "--phase" in args:
        phase = args[args.index("--phase") + 1]
    if "--kernel" in args:
        kernel = args[args.index("--kernel") + 1]
    if "--p" in args:
        p = int(args[args.index("--p") + 1])
    tag = ""
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]

    workers, rounds = load(raw, tag)
    # A worker in the shared-warm arm installs a *shared* store, so its own
    # `mode` field says "shared". The arm is the coordinator's notion: join the
    # worker record to its round by trial_id rather than trusting the worker's
    # own mode string.
    arm_of = {r["trial_id"]: r["mode"] for r in rounds}
    sel = defaultdict(list)
    for w in workers:
        if w.get("phase") != phase:
            continue
        if kernel and w.get("kernel") != kernel:
            continue
        if p and w.get("worker_count") != p:
            continue
        arm = arm_of.get(w.get("trial_id"), w.get("mode"))
        sel[(arm, w.get("kernel"), w.get("worker_count"))].append(w)

    for key in sorted(sel, key=lambda k: (k[1], k[0], k[2])):
        ws = sel[key]
        print(f"\n=== mode={key[0]} kernel={key[1]} p={key[2]} n={len(ws)} ===")
        for f in FIELDS:
            vals = [w.get(f) for w in ws if isinstance(w.get(f), (int, float))]
            if not vals:
                continue
            vals.sort()
            print(
                f"  {f:26s} median={statistics.median(vals):9.1f} "
                f"min={vals[0]:9.1f} max={vals[-1]:9.1f} n={len(vals)}"
            )
        # how many ranks saw any store I/O error, and what the store ended with
        errs = sum(1 for w in ws if (w.get("disk_io_errors_delta") or 0) > 0)
        keys = {w.get("store_keys_on_disk") for w in ws}
        print(f"  ranks_with_io_errors={errs} distinct_store_key_sets={len(keys)}")

    if "--phases" in args:
        print("\n=== first-launch phase split (per rank, from the wrapper log timestamps) ===")
        for key in sorted(sel, key=lambda k: (k[1], k[2], k[0])):
            rows = []
            dropped = 0
            for w in sel[key]:
                log = w.get("spawn_log")
                if not log or not Path(log).is_file() or not w.get("release_wall_ns"):
                    continue
                rel = w["release_wall_ns"] / 1e9
                spawn = None
                with open(log) as fh:
                    for line in fh:
                        f = line.split()
                        if f and f[0] == "B" and len(f) > 3 and f[3] == "compile":
                            t = float(f[1])
                            if t >= rel:
                                spawn = t
                                break
                if spawn is None:
                    continue
                pre = (spawn - rel) * 1e3
                # A spawn log must belong to this run of this round. Anything
                # wildly out of range means the file was rewritten by a later
                # run that reused the round directory; drop it rather than let
                # it poison the medians.
                if not (0.0 <= pre <= 120_000.0):
                    dropped += 1
                    continue
                tileiras = w.get("wrapper_compile_ms") or 0.0
                total = w.get("barrier_to_launch_ms") or 0.0
                rows.append((pre, tileiras, total - pre - tileiras, total))
            if not rows:
                continue
            med = lambda i: statistics.median([r[i] for r in rows])  # noqa: E731
            print(
                f"  mode={key[0]:12s} kernel={key[1]:5s} p={key[2]} n={len(rows)} "
                f"dropped={dropped}  "
                f"pre_spawn={med(0):7.0f}  tileiras={med(1):7.0f}  "
                f"post_compile={med(2):7.0f}  total={med(3):7.0f} ms"
            )

    # round-level: last-place rank identity and per-round maxima
    print("\n=== round-level ceilings (max rank), barrier_to_result_ms ===")
    groups = defaultdict(list)
    for r in rounds:
        if r.get("phase") != phase:
            continue
        if kernel and r.get("kernel") != kernel:
            continue
        if p and r.get("worker_count") != p:
            continue
        groups[(r["mode"], r["kernel"], r["worker_count"])].append(r)
    for key in sorted(groups, key=lambda k: (k[1], k[0], k[2])):
        rs = groups[key]
        mx = sorted(r["max_barrier_to_result_ms"] for r in rs)
        md = sorted(
            r["median_barrier_to_result_ms"] for r in rs if r["median_barrier_to_result_ms"]
        )
        print(
            f"  mode={key[0]:12s} kernel={key[1]:5s} p={key[2]} "
            f"max_rank median={statistics.median(mx):8.1f} [{mx[0]:.0f}-{mx[-1]:.0f}]  "
            f"median_rank median={statistics.median(md) if md else float('nan'):8.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
