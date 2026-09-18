#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinator for the NVlabs/cutile-rs issue-205 cross-process JIT stampede harness.

The runtime compiles each kernel once per process (an in-process cache) and can
optionally persist cubins in a shared on-disk store. Two processes that
cold-start at the same instant on one store can both miss the same L2 key, both
spawn `tileiras`, and both compile the same kernel; the store converges on one
entry because writes are atomic. This script measures what that costs. It does
not add locking or coordination of any kind to the runtime.

What it does
------------
For each configuration (mode x kernel x process count) it runs independent
trials. Each trial:

1. prepares a cache directory (fresh per trial for the cold modes),
2. starts N already-built worker binaries -- never `cargo run`, so no build or
   dependency work can land inside a timed region,
3. waits for every rank's `STAMPEDE_READY` line (each rank has bound its device,
   installed its store, derived its L2 key from the compiler frontend, and built
   its inputs),
4. writes one byte to every rank's stdin at once: the barrier release,
5. collects each rank's JSON record and its stderr, enforces a deadline, and
   kills the whole process group if a rank overruns.

Modes
-----
nocache       no JIT store installed (this is also the library default; the disk
              cache has no environment switch). Raw concurrent-compile control.
private       every rank gets its own fresh store directory: unshared cold
              compile, one store I/O stream per rank.
shared        one fresh store directory for all N ranks: the same-key stampede.
shared-warm   one store directory per trial, warmed by a single process first,
              then the N-rank round on the already-populated directory. Reports
              warmup cost, post-warm startup cost, and the total.

Every arm runs through the same `tileiras` wrapper (`scripts/tileiras_wrap.sh`),
so spawn attempts, spawn kind (version probe / capability probe / real compile),
per-spawn compile duration and cubin size are measured identically everywhere.

Outputs land under `artifacts/205/` in the repository (raw JSONL + CSV plus a
summary JSON) and under `$SCRATCH/` (per-worker logs and JSON records).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import selectors
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.path.expanduser("~")) / "pr-e2e-20260918"
DEFAULT_TARGET_DIR = SCRATCH / "target-cutile"
DEVICES = [3, 4, 5]
MODES = ["nocache", "private", "shared", "shared-warm"]
KERNELS = ["add", "gemm"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── environment / toolchain ─────────────────────────────────────────────────


def nvidia_smi_map(indices) -> dict:
    """index -> {uuid, pci_bus_id, name, compute_mode} for the GPUs we may use."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,compute_mode",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    want = {str(i) for i in indices}
    gpus = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0] in want:
            gpus[int(parts[0])] = {
                "uuid": parts[1],
                "pci_bus_id": parts[2],
                "name": parts[3],
                "compute_mode": parts[4],
            }
    return gpus


def gpu_processes() -> list:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu_uuid": parts[0], "pid": parts[1], "name": parts[2], "mem_mib": parts[3]})
    return rows


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_env(args) -> dict:
    """Resolve everything the workers need and refuse to guess."""
    need = ["CUDA_HOME", "CUTILE_TILEIRAS_PATH"]
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"error: {', '.join(missing)} not set. Source the task environment first "
            f"(env-cutile.sh) -- the harness will not invent toolchain paths."
        )
    real_tileiras = Path(os.environ["CUTILE_TILEIRAS_PATH"])
    if not real_tileiras.is_file():
        sys.exit(f"error: CUTILE_TILEIRAS_PATH={real_tileiras} is not a file")
    wrapper = REPO / "scripts" / "tileiras_wrap.sh"
    if not wrapper.is_file():
        sys.exit(f"error: wrapper {wrapper} is missing")
    if not os.access(wrapper, os.X_OK):
        sys.exit(f"error: wrapper {wrapper} is not executable")
    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", DEFAULT_TARGET_DIR))
    binary = Path(args.binary) if args.binary else target_dir / "debug" / "examples" / "jit_stampede_worker"
    if not binary.is_file():
        sys.exit(f"error: worker binary {binary} not found; build it first (--build-hint in README)")
    version = subprocess.run([str(real_tileiras), "--version"], capture_output=True, text=True).stdout.strip()
    return {
        "real_tileiras": str(real_tileiras),
        "real_tileiras_sha256": sha256_file(real_tileiras),
        "real_tileiras_size": real_tileiras.stat().st_size,
        "tileiras_version_stdout": version,
        "wrapper": str(wrapper),
        "worker_binary": str(binary),
        "worker_binary_sha256": sha256_file(binary),
        "worker_binary_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(binary.stat().st_mtime)),
        "cuda_home": os.environ["CUDA_HOME"],
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
    }


# ── process plumbing ────────────────────────────────────────────────────────


class Worker:
    """One child process: pipes, deadlines, and the record it produced."""

    def __init__(self, rank, argv, env, stdout_log, stderr_log, spawn_wall_ns):
        self.rank = rank
        self.spawn_wall_ns = spawn_wall_ns
        self.stdout_log = stdout_log
        self.stderr_log = stderr_log
        self.ready = None
        self.result_line = None
        self.stdout_buf = b""
        self.stdout_extra = []
        self.rc = None
        self.timed_out = False
        self.killed = False
        self.out_path = None
        for i, a in enumerate(argv):
            if a == "--out" and i + 1 < len(argv):
                self.out_path = argv[i + 1]
        self._err_fh = open(stderr_log, "wb")
        self._out_fh = open(stdout_log, "wb")
        self.proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err_fh,
            start_new_session=True,
            cwd=str(REPO),
        )

    @property
    def fd(self):
        return self.proc.stdout.fileno()

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._out_fh.write(data)
        self._out_fh.flush()
        self.stdout_buf += data
        while b"\n" in self.stdout_buf:
            line, self.stdout_buf = self.stdout_buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").strip()
            if text.startswith("STAMPEDE_READY "):
                self.ready = json.loads(text[len("STAMPEDE_READY ") :])
            elif text.startswith("STAMPEDE_RESULT "):
                self.result_line = json.loads(text[len("STAMPEDE_RESULT ") :])
            elif text:
                self.stdout_extra.append(text)

    def release(self) -> None:
        if self.proc.poll() is None and self.proc.stdin:
            try:
                self.proc.stdin.write(b"\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass

    def kill(self) -> None:
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.killed = True
        except (ProcessLookupError, PermissionError):
            pass

    def drain(self, sel, timeout: float) -> None:
        """Read whatever any registered worker has ready, without blocking past timeout."""
        if timeout <= 0:
            return
        for key, _ in sel.select(timeout=timeout):
            try:
                data = os.read(key.fd, 65536)
            except (BlockingIOError, OSError):
                continue
            if not data:
                try:
                    sel.unregister(key.fd)
                except (KeyError, ValueError):
                    pass
                continue
            key.data.feed(data)

    def wait(self, deadline: float, sel) -> None:
        """Follow this worker to exit (or kill it at the deadline), keeping its stdout."""
        while self.proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.timed_out = True
                self.kill()
                break
            self.drain(sel, min(0.5, remaining))
        # Bounded final drain so a grandchild holding the pipe open can never hang us.
        end = time.monotonic() + 2.0
        while time.monotonic() < end and sel.get_map():
            self.drain(sel, 0.05)
        try:
            self.rc = self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.kill()
            self.rc = self.proc.wait(timeout=15)
        self._err_fh.close()
        self._out_fh.close()

    def record(self) -> dict:
        """The worker's own JSON record, from the file it wrote (authoritative)."""
        if self.out_path and Path(self.out_path).is_file():
            with open(self.out_path) as fh:
                return json.load(fh)
        if self.result_line:
            return self.result_line
        rec = {"exit_status": "no_record", "correctness": "no_record", "rank": self.rank}
        if self.timed_out:
            rec["exit_status"] = "timeout"
            rec["correctness"] = "timeout"
        if self.rc is not None and self.rc != 0:
            rec["exit_status"] = rec.get("exit_status", "nonzero_exit")
            rec["error"] = rec.get("error", f"worker exited rc={self.rc} without a record")
        return rec


def group_metrics(workers, records, group_spawn_wall_ns, mode, kernel, p, trial_id,
                  phase, warmup_record=None) -> dict:
    ok = [r for r in records if r.get("correctness") == "correct"]
    keys = sorted({r.get("l2_key") for r in records if r.get("l2_key")})
    first_correct = [r["first_correct_wall_ns"] for r in ok if r.get("first_correct_wall_ns")]
    launch = [r.get("barrier_to_launch_ms") for r in ok if r.get("barrier_to_launch_ms") is not None]
    relaunch = [r.get("relaunch_min_ms") for r in ok if r.get("relaunch_min_ms") is not None]
    result = [r.get("barrier_to_result_ms") for r in ok if r.get("barrier_to_result_ms") is not None]
    proc = [r.get("process_to_result_ms") for r in ok if r.get("process_to_result_ms") is not None]
    exec_over = [
        (r["main_wall_ns"] - w.spawn_wall_ns) / 1e6
        for r, w in zip(records, workers)
        if r.get("main_wall_ns")
    ]

    def s(field):
        return sum(r.get(field, 0) or 0 for r in records)

    def fsum(field):
        return sum(float(r.get(field, 0.0) or 0.0) for r in records)

    m = {
        "trial_id": trial_id,
        "trial_index": int(trial_id.split("-t")[1][:3]),
        "mode": mode,
        "kernel": kernel,
        "worker_count": p,
        "phase": phase,
        "group_spawn_wall_ns": group_spawn_wall_ns,
        "n_records": len(records),
        "n_correct": len(ok),
        "n_distinct_keys": len(keys),
        "keys_uniform": len(keys) == 1,
        "l2_key": keys[0] if len(keys) == 1 else ",".join(keys),
        "all_correct": len(ok) == len(records) and len(records) == p,
        "group_to_first_correct_ms": (
            (min(first_correct) - group_spawn_wall_ns) / 1e6 if first_correct else None
        ),
        "group_to_all_correct_ms": (
            (max(first_correct) - group_spawn_wall_ns) / 1e6 if first_correct else None
        ),
        "barrier_to_all_correct_ms": max(result) if result else None,
        "barrier_to_first_correct_ms": min(result) if result else None,
        "median_barrier_to_launch_ms": statistics.median(launch) if launch else None,
        "median_barrier_to_result_ms": statistics.median(result) if result else None,
        "max_barrier_to_result_ms": max(result) if result else None,
        "median_process_to_result_ms": statistics.median(proc) if proc else None,
        "median_relaunch_min_ms": statistics.median(relaunch) if relaunch else None,
        "max_relaunch_min_ms": max(relaunch) if relaunch else None,
        "sum_jit_attributable_ms": fsum("jit_attributable_ms"),
        "max_jit_attributable_ms": max(
            [r.get("jit_attributable_ms", 0.0) or 0.0 for r in records] or [0.0]
        ),
        "all_relaunch_correct": all(bool(r.get("relaunch_correct")) for r in records) if records else None,
        "max_exec_overhead_ms": max(exec_over) if exec_over else None,
        "sum_backend_attempts": s("backend_attempts"),
        "sum_backend_success_delta": s("backend_success_delta"),
        "sum_wrapper_ok_compiles": s("wrapper_ok_compiles"),
        "sum_wrapper_failed_compiles": s("wrapper_failed_compiles"),
        "sum_wrapper_compile_ms": fsum("wrapper_compile_ms"),
        "max_wrapper_compile_ms": max([r.get("wrapper_compile_ms", 0.0) or 0.0 for r in records] or [0.0]),
        "sum_disk_hits_delta": s("disk_hits_delta"),
        "sum_disk_misses_delta": s("disk_misses_delta"),
        "sum_disk_puts_delta": s("disk_puts_delta"),
        "sum_disk_io_errors_delta": s("disk_io_errors_delta"),
        "n_ranks_with_disk_hit": sum(1 for r in records if (r.get("disk_hits_delta") or 0) > 0),
        "sum_cpu_user_ms": fsum("cpu_user_ms"),
        "sum_cpu_system_ms": fsum("cpu_system_ms"),
        "sum_cpu_user_ms_proc": fsum("cpu_user_ms_proc"),
        "sum_cpu_system_ms_proc": fsum("cpu_system_ms_proc"),
        "keys_on_disk": ",".join(
            sorted({k for r in records for k in (r.get("store_keys_on_disk") or "").split(",") if k})
        ),
        "store_contains_declared_key": all(
            bool(r.get("store_contains_l2_key")) for r in records if r.get("cache_enabled")
        )
        if any(r.get("cache_enabled") for r in records)
        else None,
        "nvidia_nodes": ",".join(sorted({r.get("nvidia_nodes", "") for r in records})),
        "device_uuids": ",".join(sorted({r.get("device_uuid", "") for r in records})),
        "cuda_visible_devices": ",".join(sorted({str(r.get("cuda_visible_devices", "")) for r in records})),
        "loadavg": Path("/proc/loadavg").read_text().split()[0] if Path("/proc/loadavg").is_file() else "",
        "warmup_ms": warmup_record.get("barrier_to_result_ms") if warmup_record else None,
        "warmup_process_to_result_ms": warmup_record.get("process_to_result_ms") if warmup_record else None,
        "warmup_backend_attempts": warmup_record.get("backend_attempts") if warmup_record else None,
        "warmup_compile_ms": warmup_record.get("wrapper_compile_ms") if warmup_record else None,
    }
    if warmup_record and m["warmup_ms"] is not None and m["barrier_to_all_correct_ms"] is not None:
        m["warmup_plus_startup_ms"] = m["warmup_ms"] + m["barrier_to_all_correct_ms"]
        m["warmup_plus_group_ms"] = m["warmup_ms"] + m["group_to_all_correct_ms"]
    else:
        m["warmup_plus_startup_ms"] = None
        m["warmup_plus_group_ms"] = None

    invalid = []
    if warmup_record is not None and warmup_record.get("correctness") != "correct":
        invalid.append("warmup_failed")
    if not m["all_correct"]:
        invalid.append("not_every_rank_produced_a_correct_result")
    if not m["keys_uniform"]:
        invalid.append("workers_disagreed_on_l2_key")
    if any(r.get("exit_status") == "timeout" for r in records):
        invalid.append("timeout")
    if any(r.get("key_precheck_touched_store") for r in records):
        invalid.append("key_precheck_touched_store")
    if any(not r.get("nvidia_node_matches_expected", True) for r in records):
        invalid.append("device_binding_unexpected")
    m["valid"] = not invalid
    m["invalid_reason"] = ";".join(invalid)
    return m


# ── round runner ────────────────────────────────────────────────────────────


def cache_dir_for(round_dir: Path, mode: str, rank: int) -> Path:
    if mode == "private":
        return round_dir / "private" / f"rank-{rank}"
    return round_dir / "shared"


def run_tag(args) -> str:
    return f"{args.tag}-" if args.tag else ""


def run_round(
    args,
    env_info,
    gpus,
    mode: str,
    kernel: str,
    p: int,
    trial_index: int,
    phase: str,
    round_root: Path,
) -> tuple[dict, list]:
    trial_id = f"{run_tag(args)}{mode}-{kernel}-p{p}-t{trial_index:03d}-{phase}"
    round_dir = round_root / mode / kernel / f"p{p}" / f"t{trial_index:03d}-{phase}"
    if round_dir.exists():
        shutil.rmtree(round_dir)  # only ever under our own scratch root
    logs_dir = round_dir / "logs"
    outs_dir = round_dir / "out"
    logs_dir.mkdir(parents=True, exist_ok=True)
    outs_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(round_dir, 0o700)

    warmup_record = None
    if mode == "shared-warm":
        # Phase A: one measured process populates the shared directory. It is a
        # full worker (same binary, same wrapper, same key check) started alone,
        # so its cost is the honest "warm the cache with one process first" price.
        # The warmup phase gets its own log/output directories: a shared spawn
        # log would mix two processes' wrapper lines into one file.
        w_logs = logs_dir / "warmup"
        w_outs = outs_dir / "warmup"
        w_logs.mkdir(parents=True, exist_ok=True)
        w_outs.mkdir(parents=True, exist_ok=True)
        w_workers, _ = spawn_group(
            args, env_info, gpus, mode, kernel, 1, trial_index, "warmup", round_dir, w_logs, w_outs
        )
        w_records = [w.record() for w in w_workers]
        with open(round_root / "warmup_workers.jsonl", "a") as fh:
            for rec in w_records:
                fh.write(json.dumps(rec) + "\n")
        warmup_record = w_records[0] if w_records else None

    # Phase B: the N-rank round, on a store that phase A has already populated
    # for shared-warm, or on a fresh directory for the cold modes.
    workers, group_spawn_wall_ns = spawn_group(
        args, env_info, gpus, mode, kernel, p, trial_index, phase, round_dir, logs_dir, outs_dir
    )
    records = [w.record() for w in workers]
    m = group_metrics(workers, records, group_spawn_wall_ns, mode, kernel, p, trial_id, phase, warmup_record)
    m["run_tag"] = run_tag(args).rstrip("-")
    m["round_dir"] = str(round_dir)
    m["ranks"] = [
        {
            "rank": r.get("rank"),
            "device_index_real": r.get("device_index_real"),
            "device_uuid": r.get("device_uuid"),
            "rc": w.rc,
            "timed_out": w.timed_out,
            "stderr_log": str(w.stderr_log),
            "stdout_log": str(w.stdout_log),
            "extra_stdout": w.stdout_extra[:5],
        }
        for r, w in zip(records, workers)
    ]
    if warmup_record:
        m["warmup_trial_id"] = warmup_record.get("trial_id")
        m["warmup_correctness"] = warmup_record.get("correctness")
    return m, records


def spawn_group(args, env_info, gpus, mode, kernel, p, trial_index, phase, round_dir, logs_dir,
                outs_dir) -> tuple[list, int]:
    """Start p workers, run the ready barrier, release the gate, collect records."""
    trial_id = f"{run_tag(args)}{mode}-{kernel}-p{p}-t{trial_index:03d}-{phase}"
    workers = []
    group_spawn_wall_ns = time.time_ns()
    for rank in range(p):
        real_index = args.device_list[rank % len(args.device_list)]
        # A missing entry here would send an empty --expected-uuid, which makes
        # the worker's UUID cross-check vacuously true. Fail instead: the device
        # set was taken from nvidia-smi, so an absent index is a real fault.
        gpu = gpus[real_index]
        cache_root = cache_dir_for(round_dir, mode, rank)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(real_index)
        env["CUTILE_TILEIRAS_REAL"] = env_info["real_tileiras"]
        env["CUTILE_TILEIRAS_PATH"] = env_info["wrapper"]
        env["CUTILE_TILEIRAS_WRAP_LOG"] = str(logs_dir / f"rank-{rank}.spawn.log")
        tmpdir = round_dir / "tmp" / f"rank-{rank}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        os.chmod(tmpdir, 0o700)
        env["TMPDIR"] = str(tmpdir)
        env["CUTILE_JIT_LOG"] = "1"
        out_path = outs_dir / f"rank-{rank}.json"
        argv = [
            env_info["worker_binary"],
            "--trial-id", trial_id,
            "--phase", phase,
            "--label", mode,
            "--mode", "shared" if mode == "shared-warm" else mode,
            "--kernel", kernel,
            "--rank", str(rank),
            "--worker-count", str(p),
            "--device", "0",  # CUDA_VISIBLE_DEVICES isolates one GPU, so it is logical 0
            "--device-index-real", str(real_index),
            "--expected-uuid", gpu["uuid"],
            "--out", str(out_path),
            "--timeout-ms", str(args.worker_timeout_ms),
        ]
        if mode != "nocache":
            argv += ["--cache-root", str(cache_root)]
        workers.append(
            Worker(
                rank,
                argv,
                env,
                logs_dir / f"rank-{rank}.stdout",
                logs_dir / f"rank-{rank}.stderr",
                time.time_ns(),
            )
        )

    # ── ready barrier ───────────────────────────────────────────────────────
    sel = selectors.DefaultSelector()
    for w in workers:
        os.set_blocking(w.fd, False)
        sel.register(w.proc.stdout, selectors.EVENT_READ, w)
    ready_deadline = time.monotonic() + args.ready_timeout_s
    while True:
        pending = [w for w in workers if w.ready is None and w.proc.poll() is None]
        if not pending:
            break
        remaining = ready_deadline - time.monotonic()
        if remaining <= 0:
            log(f"  ready timeout with {len(pending)} rank(s) not ready; releasing anyway")
            break
        for key, _ in sel.select(timeout=min(0.5, remaining)):
            try:
                data = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not data:
                continue
            key.data.feed(data)

    not_ready = [w for w in workers if w.ready is None]
    if not_ready:
        log(f"  {len(not_ready)} rank(s) never reported ready: {[w.rank for w in not_ready]}")

    # ── release ─────────────────────────────────────────────────────────────
    release_wall_ns = time.time_ns()
    for w in workers:
        w.release()

    result_deadline = time.monotonic() + args.round_timeout_s
    for w in workers:
        w.wait(result_deadline, sel)
        try:
            sel.unregister(w.proc.stdout)
        except (KeyError, ValueError):
            pass
    sel.close()
    return workers, group_spawn_wall_ns


# ── aggregation ─────────────────────────────────────────────────────────────


def spread(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    out = {
        "n": len(vals),
        "min": vals[0],
        "median": statistics.median(vals),
        "max": vals[-1],
    }
    if len(vals) >= 4:
        q = statistics.quantiles(vals, n=4, method="inclusive")
        out["p25"] = q[0]
        out["p75"] = q[2]
    return out


SUMMARY_FIELDS = [
    "group_to_all_correct_ms",
    "group_to_first_correct_ms",
    "barrier_to_all_correct_ms",
    "median_barrier_to_launch_ms",
    "median_barrier_to_result_ms",
    "max_barrier_to_result_ms",
    "median_process_to_result_ms",
    "max_exec_overhead_ms",
    "median_relaunch_min_ms",
    "max_relaunch_min_ms",
    "sum_jit_attributable_ms",
    "max_jit_attributable_ms",
    "sum_backend_attempts",
    "sum_backend_success_delta",
    "sum_wrapper_ok_compiles",
    "sum_wrapper_failed_compiles",
    "sum_wrapper_compile_ms",
    "max_wrapper_compile_ms",
    "sum_disk_hits_delta",
    "sum_disk_misses_delta",
    "sum_disk_puts_delta",
    "sum_disk_io_errors_delta",
    "n_ranks_with_disk_hit",
    "sum_cpu_user_ms",
    "sum_cpu_system_ms",
    "sum_cpu_user_ms_proc",
    "sum_cpu_system_ms_proc",
    "warmup_ms",
    "warmup_process_to_result_ms",
    "warmup_compile_ms",
    "warmup_plus_startup_ms",
    "warmup_plus_group_ms",
]


def hist(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def summarize(rounds):
    """Per-configuration spread over independent trials (measure phase only)."""
    configs = {}
    for r in rounds:
        if r["phase"] != "measure":
            continue
        key = (r["mode"], r["kernel"], r["worker_count"])
        configs.setdefault(key, []).append(r)
    out = []
    for (mode, kernel, p), rs in sorted(configs.items()):
        valid = [r for r in rs if r["valid"]]
        entry = {
            "mode": mode,
            "kernel": kernel,
            "worker_count": p,
            "trials_total": len(rs),
            "trials_valid": len(valid),
            "trials_invalid": len(rs) - len(valid),
            "invalid_reasons": ";".join(sorted({r["invalid_reason"] for r in rs if not r["valid"] and r["invalid_reason"]})),
            "keys_uniform_all": all(r["keys_uniform"] for r in valid) if valid else None,
            "l2_key": valid[0]["l2_key"] if valid else None,
            "loadavg_median": spread([float(r["loadavg"]) for r in valid if r["loadavg"]]) if valid else None,
        }
        for field in SUMMARY_FIELDS:
            entry[field] = spread([r.get(field) for r in valid])
        # The issue's own caveat: a racing early finisher can populate the
        # store before every other rank looks, so the observed number of
        # duplicate compiles per round is a distribution, not a constant P.
        entry["backend_attempts_histogram"] = hist([r["sum_backend_attempts"] for r in valid])
        entry["backend_success_histogram"] = hist([r["sum_backend_success_delta"] for r in valid])
        entry["disk_hits_histogram"] = hist([r["sum_disk_hits_delta"] for r in valid])
        entry["ranks_with_hit_histogram"] = hist([r["n_ranks_with_disk_hit"] for r in valid])
        out.append(entry)
    return out


def flatten_csv_row(prefix, spread_obj):
    row = {}
    if isinstance(spread_obj, dict):
        for k, v in spread_obj.items():
            row[f"{prefix}_{k}"] = v
    else:
        row[prefix] = spread_obj
    return row


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10, help="independent cold trials per configuration")
    ap.add_argument("--p", default="1,2,4", help="comma-separated process counts")
    ap.add_argument("--kernels", default="add,gemm")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--correctness-rounds", type=int, default=3,
                    help="rounds per configuration run first and excluded from statistics")
    ap.add_argument("--round-timeout-s", type=float, default=300.0)
    ap.add_argument("--ready-timeout-s", type=float, default=240.0)
    ap.add_argument("--worker-timeout-ms", type=int, default=600_000)
    ap.add_argument("--artifacts", default=str(REPO / "artifacts" / "205"))
    ap.add_argument("--binary", default=None, help="worker binary (default: $CARGO_TARGET_DIR/debug/examples/jit_stampede_worker)")
    ap.add_argument("--tag", default="", help="suffix for the raw output files")
    ap.add_argument("--devices", default="3",
                    help="comma-separated physical GPU indices this run may use; ranks are assigned "
                         "round-robin (default 3: 4 and 5 are Exclusive_Process and held by another "
                         "user's vLLM workers, 0-2 have foreign processes)")
    ap.add_argument("--plan", action="store_true", help="print the matrix and exit")
    ap.add_argument("--summarize-only", default=None, metavar="ROUNDS_JSONL",
                    help="re-derive summary.json/summary.csv from an existing rounds JSONL "
                         "without running anything")
    ap.add_argument("--trials-per-block", type=int, default=1,
                    help="trials to run back-to-back per configuration before rotating (interleaving)")
    args = ap.parse_args()

    ps = [int(x) for x in args.p.split(",") if x.strip()]
    kernels = [x for x in args.kernels.split(",") if x.strip()]
    modes = [x for x in args.modes.split(",") if x.strip()]
    for m in modes:
        if m not in MODES:
            sys.exit(f"unknown mode {m}; known: {MODES}")
    for k in kernels:
        if k not in KERNELS:
            sys.exit(f"unknown kernel {k}; known: {KERNELS}")

    if args.plan:
        for mode in modes:
            for kernel in kernels:
                for p in ps:
                    print(f"{mode:12s} {kernel:5s} p={p}")
        return 0

    if args.summarize_only:
        rounds_path = Path(args.summarize_only)
        rounds = []
        with rounds_path.open() as fh:
            for line in fh:
                if line.strip():
                    rounds.append(json.loads(line))
        summary = summarize(rounds)
        header_path = rounds_path.parent / rounds_path.name.replace("rounds", "run_header").replace(
            ".jsonl", ".json"
        )
        header = json.loads(header_path.read_text()) if header_path.is_file() else {}
        out_json = rounds_path.parent / (rounds_path.stem.replace("rounds", "summary") + ".json")
        out_json.write_text(
            json.dumps({"header": header, "summary": summary, "rounds": len(rounds)}, indent=2) + "\n"
        )
        rows = []
        for e in summary:
            row = {k: e[k] for k in
                   ("mode", "kernel", "worker_count", "trials_total", "trials_valid",
                    "trials_invalid", "keys_uniform_all", "l2_key")}
            for field in SUMMARY_FIELDS:
                row.update(flatten_csv_row(field, e.get(field)))
            for h in ("backend_attempts_histogram", "backend_success_histogram",
                      "disk_hits_histogram", "ranks_with_hit_histogram"):
                row[h] = json.dumps(e.get(h, {}))
            rows.append(row)
        write_csv(rounds_path.parent / (rounds_path.stem.replace("rounds", "summary") + ".csv"), rows)
        log(f"wrote {out_json}")
        return 0

    env_info = check_env(args)
    args.device_list = [int(x) for x in args.devices.split(",") if x.strip()]
    gpus = nvidia_smi_map(args.device_list)
    artifacts = Path(args.artifacts)
    (artifacts / "raw").mkdir(parents=True, exist_ok=True)
    # Per-run round root. A shared root would let a later run delete and rewrite
    # an earlier run's per-worker stderr/stdout/spawn logs (the round records
    # survive in the JSONL, but the evidence files they point at would not).
    raw_root = SCRATCH / "out" / "205" / (f"rounds-{args.tag}" if args.tag else "rounds")
    raw_root.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)

    tag = f".{args.tag}" if args.tag else ""
    wpath = artifacts / "raw" / f"workers{tag}.jsonl"
    rpath = artifacts / "raw" / f"rounds{tag}.jsonl"

    log(f"worker binary : {env_info['worker_binary']}")
    log(f"tileiras      : {env_info['real_tileiras']} sha256={env_info['real_tileiras_sha256'][:16]}…")
    log(f"devices       : {json.dumps(gpus)}")
    log(f"matrix        : modes={modes} kernels={kernels} p={ps} trials={args.trials} "
        f"correctness_rounds={args.correctness_rounds}")

    header = {
        "kind": "run_header",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": env_info,
        "gpus": gpus,
        "gpu_processes": gpu_processes(),
        "modes": modes,
        "kernels": kernels,
        "process_counts": ps,
        "trials": args.trials,
        "correctness_rounds": args.correctness_rounds,
        "args": vars(args),
        "repo_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
        "cuda_visible_devices_policy": "one real GPU index per rank, isolated as logical 0",
    }
    (artifacts / "raw" / f"run_header{tag}.json").write_text(json.dumps(header, indent=2) + "\n")

    all_rounds = []
    wfh = open(wpath, "a")
    rfh = open(rpath, "a")

    def emit_round(m, records):
        all_rounds.append(m)
        rfh.write(json.dumps(m) + "\n")
        rfh.flush()
        for rec in records:
            wfh.write(json.dumps(rec) + "\n")
        wfh.flush()
        log(
            f"  {m['mode']:11s} {m['kernel']:4s} p={m['worker_count']} t={m['trial_index']:03d} "
            f"valid={m['valid']} all_correct={m['all_correct']} "
            f"attempts={m['sum_backend_attempts']} ok={m['sum_backend_success_delta']} "
            f"hits={m['sum_disk_hits_delta']} "
            f"compile_ms={m['sum_wrapper_compile_ms']:.0f} "
            f"grp_ms={m['group_to_all_correct_ms'] if m['group_to_all_correct_ms'] is None else round(m['group_to_all_correct_ms'], 1)} "
            f"warm={m['warmup_ms'] if m['warmup_ms'] is None else round(m['warmup_ms'], 1)}"
            + (f" INVALID({m['invalid_reason']})" if not m["valid"] else "")
        )

    # Phase 1: correctness rounds per configuration (excluded from statistics).
    n_corr = max(0, args.correctness_rounds)
    for mode in modes:
        for kernel in kernels:
            for p in ps:
                for t in range(n_corr):
                    m, recs = run_round(args, env_info, gpus, mode, kernel, p, t, "correctness", raw_root)
                    emit_round(m, recs)
                    if not m["all_correct"] and m["n_correct"] == 0 and t == 0:
                        log(f"  !! no rank produced a correct result for {mode}/{kernel}/p{p}; "
                            f"see {m['round_dir']}/logs")

    # Phase 2: measurement rounds, interleaved so machine drift is shared.
    configs = [(mode, kernel, p) for mode in modes for kernel in kernels for p in ps]
    trial = 0
    while trial < args.trials:
        block = min(args.trials_per_block, args.trials - trial)
        for (mode, kernel, p) in configs:
            for k in range(block):
                m, recs = run_round(args, env_info, gpus, mode, kernel, p, trial + k, "measure", raw_root)
                emit_round(m, recs)
        trial += block

    wfh.close()
    rfh.close()

    summary = summarize(all_rounds)
    payload = json.dumps({"header": header, "summary": summary, "rounds": len(all_rounds)}, indent=2) + "\n"
    (artifacts / "summary.json").write_text(payload)
    if tag:
        (artifacts / "raw" / f"summary{tag}.json").write_text(payload)
    rows = []
    for e in summary:
        row = {
            "mode": e["mode"],
            "kernel": e["kernel"],
            "worker_count": e["worker_count"],
            "trials_total": e["trials_total"],
            "trials_valid": e["trials_valid"],
            "trials_invalid": e["trials_invalid"],
            "keys_uniform_all": e["keys_uniform_all"],
            "l2_key": e["l2_key"],
        }
        for field in SUMMARY_FIELDS:
            row.update(flatten_csv_row(field, e.get(field)))
        for h in ("backend_attempts_histogram", "backend_success_histogram",
                  "disk_hits_histogram", "ranks_with_hit_histogram"):
            row[h] = json.dumps(e.get(h, {}))
        rows.append(row)
    write_csv(artifacts / "raw" / f"summary{tag}.csv", rows)
    write_csv(
        artifacts / "raw" / f"rounds{tag}.csv",
        [{k: v for k, v in r.items() if k != "ranks"} for r in all_rounds],
    )

    log(f"wrote {wpath}, {rpath}, {artifacts / 'summary.json'}")
    n_invalid = sum(1 for r in all_rounds if not r["valid"])
    log(f"rounds: {len(all_rounds)} total, {n_invalid} invalid")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
