#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn `artifacts/205/summary.json` into the markdown tables used in its README.

Reads only measured data. Every number printed here comes from a round record
written by `scripts/bench_jit_stampede.py`; nothing is filled in or estimated.
Cells whose configuration produced no valid trial print `not measured`.

    python3 scripts/jit_stampede_tables.py artifacts/205/summary.run1.json artifacts/205/summary.run2.json

With several summaries, each configuration is taken from the run that has the
most valid trials for it, so a short screening run never overwrites a long one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODE_LABEL = {
    "nocache": "disk cache disabled",
    "private": "per-process cold private dir",
    "shared": "shared cold dir",
    "shared-warm": "single-process warm, then shared warm dir",
}


def fmt(spread, key="median", digits=1, scale=1.0):
    if not spread:
        return "not measured"
    v = spread.get(key)
    if v is None:
        return "not measured"
    return f"{v * scale:.{digits}f}"


def rng(spread, digits=1, scale=1.0):
    if not spread:
        return "not measured"
    lo, hi = spread.get("min"), spread.get("max")
    if lo is None or hi is None:
        return "not measured"
    return f"{lo * scale:.{digits}f}\u2013{hi * scale:.{digits}f}"


def main() -> int:
    paths = [Path(p) for p in (sys.argv[1:] or ["artifacts/205/summary.json"])]
    summary = []
    by = {}
    for path in paths:
        doc = json.loads(path.read_text())
        for s in doc["summary"]:
            key = (s["mode"], s["kernel"], s["worker_count"])
            prev = by.get(key)
            if prev is None or s["trials_valid"] > prev["trials_valid"]:
                by[key] = s
    summary = list(by.values())
    srcs = []
    for p in paths:
        n = json.loads(p.read_text())["rounds"]
        srcs.append("%s (%d rounds)" % (p.name, n))
    print("sources:", ", ".join(srcs))

    def cell(mode, kernel, p, key, digits=1, scale=1.0, which="median"):
        s = by.get((mode, kernel, p))
        return fmt(s.get(key) if s else None, which, digits, scale)

    for kernel in sorted({k for (_, k, _) in by}):
        ps = sorted({p for (_, k, p) in by if k == kernel})
        modes = [m for m in MODE_LABEL if any((m, kernel, p) in by for p in ps)]
        print(f"\n### kernel `{kernel}` — duplicate backend compiles per round (sum over ranks)\n")
        cols = " | ".join(MODE_LABEL[m] for m in modes)
        print(f"| processes | {cols} |")
        print("|---" * (len(modes) + 1) + "|")
        for p in ps:
            cells = []
            for m in modes:
                s = by.get((m, kernel, p))
                cells.append(fmt(s.get("sum_backend_attempts") if s else None))
            print(f"| {p} | " + " | ".join(cells) + " |")
        print(f"\nActual per-round distributions (backend spawn attempts):\n")
        for p in ps:
            row = []
            for m in modes:
                s = by.get((m, kernel, p))
                row.append(
                    f"{MODE_LABEL[m]}: {json.dumps(s.get('backend_attempts_histogram', {})) if s else 'not measured'}"
                )
            print(f"- p={p}: " + "; ".join(row))

        print(f"\n### kernel `{kernel}` — barrier release \u2192 all ranks correct (ms)\n")
        print(f"| processes | {cols} |")
        print("|---" * (len(modes) + 1) + "|")
        for p in ps:
            cells = [cell(m, kernel, p, "barrier_to_all_correct_ms") for m in modes]
            print(f"| {p} | " + " | ".join(cells) + " |")
        print(f"\nspread (min\u2013max over trials):\n")
        print(f"| processes | {cols} |")
        print("|---" * (len(modes) + 1) + "|")
        for p in ps:
            cells = []
            for m in modes:
                s = by.get((m, kernel, p))
                cells.append(rng(s.get("barrier_to_all_correct_ms") if s else None))
            print(f"| {p} | " + " | ".join(cells) + " |")

        print(f"\n### kernel `{kernel}` — tileiras child wall time, sum over ranks (ms)\n")
        print(f"| processes | {cols} |")
        print("|---" * (len(modes) + 1) + "|")
        for p in ps:
            cells = [cell(m, kernel, p, "sum_wrapper_compile_ms") for m in modes]
            print(f"| {p} | " + " | ".join(cells) + " |")

    print("\n### validity\n")
    print("| mode | kernel | processes | trials total | valid | invalid | keys uniform |")
    print("|---|---|---|---|---|---|---|")
    for s in sorted(summary, key=lambda x: (x["kernel"], x["mode"], x["worker_count"])):
        print(
            f"| {s['mode']} | {s['kernel']} | {s['worker_count']} | {s['trials_total']} | "
            f"{s['trials_valid']} | {s['trials_invalid']} | {s['keys_uniform_all']} |"
        )

    print("\n### warmup accounting (shared-warm mode, ms)\n")
    print("| kernel | processes | warmup (solo cold compile) | post-warm startup (all ranks) | total |")
    print("|---|---|---|---|---|")
    for s in summary:
        if s["mode"] != "shared-warm":
            continue
        print(
            f"| {s['kernel']} | {s['worker_count']} | "
            f"{fmt(s['warmup_ms'])} | {fmt(s['barrier_to_all_correct_ms'])} | "
            f"{fmt(s['warmup_plus_startup_ms'])} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
