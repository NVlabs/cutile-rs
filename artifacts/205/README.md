# Issue 205 — cross-process JIT stampede: Phase 1 measurement

**Phase 1 only: this directory contains a harness, raw data, and conclusions. No
runtime coordination, locking, or cache-semantics change was implemented or is
proposed here.** The upstream decision (no cross-process lock at this stage)
stands; this work measures what the lock would have bought and what it would
have cost.

Upstream issue: `NVlabs/cutile-rs#205` — "the in-process JIT cache compiles each
kernel once per process; the persistent disk cache is shared across processes,
but compilation is not deduplicated across process boundaries."

## 1. What was measured, in one paragraph

`cutile`'s JIT caches compiled cubins in two layers: an in-process L1 map and an
opt-in, content-addressed, on-disk L2 store (`cutile-compiler/src/jit_cache.rs`).
The disk store is off by default and has **no environment variable that enables
it** — a program must construct a `FileSystemJitStore` over its own directory and
call `jit_cache::enable`. When N processes that share one store directory
cold-start at the same instant, each of them independently misses the same L2
key and spawns `tileiras` for the same kernel. Writes are atomic
(temp file + `rename`), so the store converges on exactly one entry and the
result is correct — the question is only what the redundant work costs.

This harness starts N already-built worker processes behind a cross-process
ready/start barrier, timed from `main` and from barrier release, and records for
every rank: the actual L2 key (derived independently up front *and* verified
against the file the runtime actually wrote), real `tileiras` spawn attempts
from a pass-through wrapper (the runtime's own counter only counts *successful*
compiles), disk hits/misses/puts, CPU time including the compiler subprocess,
GPU identity, and the correctness of the kernel result.

## 2. Environment

From `raw/run_header.run1.json`, written before the first round of the primary run:

| | |
|---|---|
| host | `jk01`: 8× NVIDIA L20 (sm_89), driver 570.86.10, shared with other users |
| device used by every rank | GPU 3, `GPU-dfe309b3-df0d-f7f1-642b-40c03e15924f`, PCI `00000000:37:00.0`, compute mode `Default` |
| foreign GPU processes at start | 12, on the other seven devices (`gpu_processes` in the run header) |
| load average while measuring | per-configuration medians 517–615, overall median 551 |
| `tileiras` | `…/cuda-13.3/bin/tileiras`, CUDA 13.3 V13.3.36, 94,855,128 bytes, sha256 `88737a8b…62bf0` |
| worker binary | `target/debug/examples/jit_stampede_worker`, sha256 `6c0072c3…e92964`, built 2026-09-18T21:57:44 |
| toolkit / userspace driver | CUDA 13.3 unpacked under the scratch volume; `LD_LIBRARY_PATH` is the CUDA 13.3 forward-compat directory, because the kernel module is older than the toolkit |

## 3. Harness layout

| file | role |
|---|---|
| `cutile-examples/examples/jit_stampede_worker.rs` | one rank: bind device, install the store for its mode, derive the L2 key from meta tensors, upload real inputs with host→device copies (no kernel compile), print `STAMPEDE_READY`, block on one stdin byte, then JIT+launch+verify and write its JSON record |
| `scripts/bench_jit_stampede.py` | coordinator: builds nothing inside a timed region, starts the binary directly, runs the ready/start barrier, enforces deadlines, kills process groups, writes raw JSONL/CSV + `summary.json` |
| `scripts/tileiras_wrap.sh` | pass-through `tileiras` wrapper used in **every** arm: logs spawn attempts, spawn class (`version` probe / `probe` capability check / `compile`), child exit status, cubin size and the child's wall time; `--version` is passed through untouched so the key fingerprint is unchanged |
| `scripts/run_jit_stampede_205.sh` | the one command that builds and runs the whole matrix |
| `scripts/jit_stampede_tables.py` | turns `summary.json` into the tables below |
| `scripts/jit_stampede_probe.py` | ad-hoc per-phase split and round-ceiling analysis over the raw worker/round records; every number it prints is read from a record, none is re-derived or estimated |

### 3.1 Barrier protocol

The coordinator starts every rank with a pipe on stdin and waits until each rank
prints `STAMPEDE_READY`. A rank reaches READY only after it has bound its
device, created a CUDA context, installed (or deliberately not installed) its
JIT store, derived its L2 key, and uploaded its inputs. The coordinator then
writes one byte to every rank's stdin, in one tight loop: that write is the
**barrier release**, and it is observed by each rank with a microsecond-scale
blocking read (no polling, no file-watching latency).

### 3.2 What the numbers mean

| field | definition |
|---|---|
| `process_to_result_ms` | first line of the rank's `main` → verified-correct result on the host. Includes device init, the frontend-only key derivation, upload, compile, launch, and the host-side check. |
| `barrier_to_result_ms` | barrier release → verified-correct result. Everything before the barrier is identical in every arm, so this is the contention-sensitive number. |
| `barrier_to_launch_ms` | barrier release → kernel result materialized on the device (`sync`), i.e. JIT + module load + launch. |
| `relaunch_min_ms` | the same launch again, after the measurement, when the in-process L1 cache is already populated: the pure launch+verify floor with no JIT at all. |
| `jit_attributable_ms` | `barrier_to_launch_ms − relaunch_min_ms`: the part of the first launch that the JIT (frontend + backend) actually added, separated from GPU/module-load noise. |
| `backend_attempts` | **real `tileiras` stage-2 spawn attempts** in the timed region, from the wrapper log. This is the spawn counter the runtime does not have. |
| `backend_success_delta` | `jit_backend_compile_count()` delta: `tileiras` runs that produced a non-empty cubin. **Not** a spawn count. |
| `wrapper_compile_ms` | wall time inside the wrapper around the `tileiras` child: the pure backend compile cost, independent of the GPU. |
| `disk_hits_delta` / `disk_misses_delta` / `disk_puts_delta` | `jit_cache::stats()` deltas across the timed region. |
| `cpu_user_ms` / `cpu_system_ms` | `getrusage(RUSAGE_SELF) + getrusage(RUSAGE_CHILDREN)` deltas across the timed region, so the `tileiras` subprocess is included, not just the Rust parent. |
| `l2_key` | 64-char SHA-256 derived up front from the compiler frontend over **meta** tensors (no store access, no backend, no GPU). |
| `store_keys_on_disk` | the `<key>.cubin` file names actually present in the rank's store after its run — an independent check that the key used by the runtime is the key declared. |

Group-level metrics (`group_to_all_correct_ms`) use `CLOCK_REALTIME` timestamps
taken in the workers, compared against the coordinator's timestamp captured just
before it spawned the first rank. All per-rank durations use a per-process
`Instant`, so no cross-process clock is used for them.

## 4. Modes

| mode | store | purpose |
|---|---|---|
| `nocache` | none installed (`jit_cache::disable()`, also the library default) | raw concurrent-compile control; must be distinguished from default behaviour, and it is: the default is *also* no disk cache, but with no way to share one |
| `private` | fresh directory **per rank per round** | unshared cold compile cost: no cross-process sharing at all, but the store code path is live |
| `shared` | one fresh directory for all ranks in the round | reproduces the same-key duplicate backend compiles |
| `shared-warm` | one directory per round, populated by a **single measured process first** | upper bound of the documented workaround; warmup cost and post-warm startup cost are reported separately, plus the total |

## 5. Results

Primary run: `raw/summary.run1.json`, 416 rounds — 4 modes × 2 kernels × `P ∈ {1,2,4,8}`, each configuration with 3 correctness rounds and 10 measured trials, started 2026-09-18T22:03+0800 on GPU 3. Every number below comes from the round records of that run, through

```bash
python3 scripts/jit_stampede_tables.py artifacts/205/raw/summary.run1.json
```

`sources: summary.run1.json (416 rounds)`

### kernel `add` — duplicate backend compiles per round (sum over ranks)

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 1.0 | 1.0 | 1.0 | 0.0 |
| 2 | 2.0 | 2.0 | 2.0 | 0.0 |
| 4 | 4.0 | 4.0 | 4.0 | 0.0 |
| 8 | 8.0 | 8.0 | 8.0 | 0.0 |

Actual per-round distributions (backend spawn attempts):

- p=1: disk cache disabled: `{"1": 10}`; per-process cold private dir: `{"1": 10}`; shared cold dir: `{"1": 10}`; single-process warm, then shared warm dir: `{"0": 10}`
- p=2: disk cache disabled: `{"2": 10}`; per-process cold private dir: `{"2": 10}`; shared cold dir: `{"2": 10}`; single-process warm, then shared warm dir: `{"0": 10}`
- p=4: disk cache disabled: `{"4": 10}`; per-process cold private dir: `{"4": 10}`; shared cold dir: `{"4": 10}`; single-process warm, then shared warm dir: `{"0": 10}`
- p=8: disk cache disabled: `{"8": 10}`; per-process cold private dir: `{"8": 10}`; shared cold dir: `{"8": 10}`; single-process warm, then shared warm dir: `{"0": 10}`

### kernel `add` — barrier release → all ranks correct (ms)

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 1285.2 | 1298.8 | 990.5 | 425.1 |
| 2 | 1464.6 | 1292.7 | 1694.3 | 290.2 |
| 4 | 1315.6 | 1882.7 | 3134.7 | 317.0 |
| 8 | 1190.4 | 1220.0 | 4634.6 | 458.7 |

spread (min–max over trials):

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 497.9–2120.5 | 456.3–2702.1 | 437.4–3979.2 | 244.2–1453.8 |
| 2 | 961.2–4718.6 | 384.8–2726.5 | 484.6–5220.3 | 243.3–1636.1 |
| 4 | 514.0–2861.2 | 527.5–4953.5 | 1332.1–5087.8 | 262.9–1277.7 |
| 8 | 681.0–3817.3 | 720.2–4121.0 | 2698.4–7893.1 | 263.2–2011.7 |

### kernel `add` — tileiras child wall time, sum over ranks (ms)

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 419.1 | 866.4 | 693.0 | 0.0 |
| 2 | 1950.2 | 1326.7 | 919.0 | 0.0 |
| 4 | 2302.7 | 2686.0 | 2391.4 | 0.0 |
| 8 | 3537.6 | 4652.5 | 6397.3 | 0.0 |

### kernel `gemm` — duplicate backend compiles per round (sum over ranks)

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 1.0 | 1.0 | 1.0 | 0.0 |
| 2 | 2.0 | 2.0 | 2.0 | 0.0 |
| 4 | 4.0 | 4.0 | 4.0 | 0.0 |
| 8 | 8.0 | 8.0 | 8.0 | 0.0 |

Actual per-round distributions (backend spawn attempts):

- p=1: disk cache disabled: `{"1": 10}`; per-process cold private dir: `{"1": 10}`; shared cold dir: `{"1": 10}`; single-process warm, then shared warm dir: `{"0": 10}`
- p=2: disk cache disabled: `{"2": 10}`; per-process cold private dir: `{"2": 10}`; shared cold dir: `{"2": 10}`; single-process warm, then shared warm dir: `{"0": 10}`
- p=4: disk cache disabled: `{"4": 10}`; per-process cold private dir: `{"4": 10}`; shared cold dir: `{"4": 10}`; single-process warm, then shared warm dir: `{"0": 10}`
- p=8: disk cache disabled: `{"8": 10}`; per-process cold private dir: `{"8": 10}`; shared cold dir: `{"8": 10}`; single-process warm, then shared warm dir: `{"0": 10}`

### kernel `gemm` — barrier release → all ranks correct (ms)

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 1500.2 | 1983.4 | 1638.9 | 348.9 |
| 2 | 1663.5 | 2142.6 | 3304.8 | 384.5 |
| 4 | 1521.9 | 1333.3 | 2817.7 | 404.7 |
| 8 | 1456.6 | 2203.8 | 4571.7 | 471.3 |

spread (min–max over trials):

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 826.3–4663.4 | 1019.6–3728.7 | 812.6–3787.6 | 320.3–924.9 |
| 2 | 816.4–4145.6 | 882.2–3841.5 | 1034.0–8555.8 | 337.3–1553.1 |
| 4 | 894.7–5361.7 | 880.0–4530.3 | 1265.6–3845.6 | 348.0–950.9 |
| 8 | 941.9–2997.2 | 1222.6–3710.7 | 1718.3–11086.0 | 359.1–1078.2 |

### kernel `gemm` — tileiras child wall time, sum over ranks (ms)

| processes | disk cache disabled | per-process cold private dir | shared cold dir | single-process warm, then shared warm dir |
|---|---|---|---|---|
| 1 | 911.1 | 1387.5 | 1032.4 | 0.0 |
| 2 | 1586.8 | 3219.1 | 2783.8 | 0.0 |
| 4 | 4308.5 | 2902.3 | 4546.7 | 0.0 |
| 8 | 6611.9 | 11640.0 | 10740.9 | 0.0 |

### validity

| mode | kernel | processes | trials total | valid | invalid | keys uniform |
|---|---|---|---|---|---|---|
| nocache | add | 1–8 | 10 | 10 | 0 | True |
| private | add | 1–8 | 10 | 10 | 0 | True |
| shared | add | 1–8 | 10 | 10 | 0 | True |
| shared-warm | add | 1–8 | 10 | 10 | 0 | True |
| nocache | gemm | 1–8 | 10 | 10 | 0 | True |
| private | gemm | 1–8 | 10 | 10 | 0 | True |
| shared | gemm | 1–8 | 10 | 10 | 0 | True |
| shared-warm | gemm | 1–8 | 10 | 10 | 0 | True |

All 32 configurations, 10/10 valid, and `keys_uniform_all` is true in every one
of them: the 64-character key each rank derived up front through the frontend
equalled the key the runtime actually used for the store.

### warmup accounting (shared-warm mode, ms)

| kernel | processes | warmup (solo cold compile) | post-warm startup (all ranks) | total |
|---|---|---|---|---|
| add | 1 | 1750.3 | 425.1 | 2248.5 |
| add | 2 | 1485.1 | 290.2 | 1759.2 |
| add | 4 | 758.0 | 317.0 | 1056.2 |
| add | 8 | 1566.9 | 458.7 | 2329.8 |
| gemm | 1 | 2602.6 | 348.9 | 2949.5 |
| gemm | 2 | 1411.1 | 384.5 | 2185.5 |
| gemm | 4 | 2062.6 | 404.7 | 2520.2 |
| gemm | 8 | 2970.9 | 471.3 | 3426.2 |

### 5.1 What the tables show

**The stampede is one compile per cold process, and sharing a directory adds
nothing to that.** The spawn-attempt histogram is `{"P": 10}` for all ten trials
of both kernels in all three cold arms, and `{"0": 10}` in the warm arm. The
identical counts in `nocache` and `private` show the redundancy is not caused by
sharing a directory — the store's atomic write + rename keeps the ranks from
blocking each other, so there is no convoy, only duplicated work. This matters
for the issue's second phase: what has to be removed is the duplication, not
mutual interference.

**The shared-cold penalty appears only at P ≥ 4, and it is not the compiler.**
`add` needs 990.5 / 1694.3 / 3134.7 / 4634.6 ms at P = 1/2/4/8 in the shared arm,
against 1298.8 / 1292.7 / 1882.7 / 1220.0 ms in the private arm — the private arm
stays flat while the shared arm grows super-linearly. At P=8 the compiler's own
wall time grows only 1.4× between those arms (4652.5 → 6397.3 ms summed over
ranks), while the group latency grows 3.8× (1220.0 → 4634.6 ms). The compiler is
not where the time goes.

**The rest is blocking on the shared directory.** Phase split of `add`, P=8,
n=80 ranks per arm, 0 dropped, from the supplementary `phases` run:

```bash
python3 scripts/jit_stampede_probe.py artifacts/205/raw --kernel add --p 8 --tag phases --phases
```

| arm | pre-spawn (frontend + L2 lookup) | `tileiras` child | post-compile (read cubin + encode + put + module load + launch) | total |
|---|---|---|---|---|
| disk cache disabled | 259 | 239 | **19** | 517 |
| per-process cold private dir | 259 | 216 | **22** | 532 |
| shared cold dir | 263 | 287 | **247** | 1035 |

Per-rank CPU (`cpu_user_ms + cpu_system_ms`, median over the same 80 ranks) is
378 / 382 / 382 ms: the ranks do the same amount of work in all three arms, and
the extra ~225 ms per rank in the shared arm is spent waiting rather than
computing. `strace` of the same configuration puts the wait in the single shared
shard directory: the worst single `rename` into it took **1294 ms** (against
0.1 ms in the private arm) and the worst `openat` 848 ms (against 397 ms). Part
of the wall-clock penalty is therefore this host's same-directory metadata
serialization, which a compile lock would not remove — and which is why the
multi-second figures must not be read as hardware-general.

**The documented warmup workaround already covers the need.** At P=8 the whole
warmup flow — one process compiles alone, then the ranks start on the populated
directory — costs 2329.8 ms (`add`) and 3426.2 ms (`gemm`), *below* the shared
cold start alone (4634.6 / 4571.7 ms). The post-warm startup (458.7 / 471.3 ms)
is below even the cache-disabled baseline (1190.4 / 1456.6 ms), so the workaround
in the issue is not a trade-off to weigh: it is already the faster option.

**Across two physical devices the redundancy holds but the magnitude does not.**
`raw/multigpu-summary.md` repeats the experiment with ranks round-robined over
GPUs 1 and 3: still exactly one compile per cold process, but at `add`/P=4 the
warm-cache advantage falls from 9.9× (3134.7 → 317.0 ms) to 2.5× (762.1 →
305.5 ms). That is the expected consequence of the phase split above: a large
part of the single-device penalty is contention on one device and its directory,
not the redundant compilation.

## 6. Raw data layout

`scripts/bench_jit_stampede.py` writes four kinds of file under `--artifacts`
(default `artifacts/205/`); `--tag` keeps runs from overwriting each other.

| file | content |
|---|---|
| `raw/workers[.tag].jsonl` | one record per rank per round: the measured durations, the key it derived and the keys actually on disk, `jit_cache` deltas, CPU time, device identity and UUID, and the path of that rank's wrapper log |
| `raw/rounds[.tag].jsonl` / `.csv` | one record per round: validity, per-rank spread, group timings, spawn-attempt histogram, load average |
| `raw/summary[.tag].json` / `.csv` | the per-configuration statistics rendered in §5 |
| `raw/run_header[.tag].json` | the environment fingerprint (§2) and GPU identity map, written before the first round |

Per-rank stdout/stderr, the JSON record each rank wrote, and the `tileiras`
wrapper log named in every worker record live under
`$HOME/pr-e2e-20260918/out/205/rounds-<tag>/<mode>/<kernel>/p<P>/<trial>/`.

Only this file and `raw/multigpu-summary.md` are committed. The primary run's
round records alone are 1.4 MB, and the per-rank logs are far larger, so the raw
files stay in the scratch directory that produced them; §5 is generated from
them by `scripts/jit_stampede_tables.py`.

## 7. Reproducing

```bash
# 1. toolchain: CUDA_HOME + CUTILE_TILEIRAS_PATH (+ the forward-compat libcuda dir here)
source ~/pr-e2e-20260918/env-cutile.sh

# 2. builds the worker outside every timed region, then runs the whole matrix
scripts/run_jit_stampede_205.sh --trials 10 --correctness-rounds 3 \
  --p 1,2,4,8 --modes nocache,private,shared,shared-warm \
  --kernels add,gemm --devices 3 --tag run1

# 3. the tables in §5, from that run's own summary
python3 scripts/jit_stampede_tables.py artifacts/205/raw/summary.run1.json

# 4. the per-phase split, which needs that run's wrapper logs
python3 scripts/jit_stampede_probe.py artifacts/205/raw --kernel add --p 8 --tag run1 --phases
```

`run_jit_stampede_205.sh` refuses to start without `CUDA_HOME` and
`CUTILE_TILEIRAS_PATH`, and refuses a `CUTILE_TILEIRAS_PATH` that already points
at the wrapper — either would silently change what is being measured. The
coordinator builds nothing inside a timed region: it only starts the binary the
build step produced. A short screening run is
`scripts/run_jit_stampede_205.sh --trials 2 --p 1,2 --kernels add --modes shared`.

## 8. Limitations

* **Every wall-clock number here is conditioned on one heavily loaded shared
  host.** The load average was 517–615 during the primary run and 130–1370 during
  the supplementary runs; the shared-cold median for one configuration drifted
  between 1519 ms and 4635 ms across runs, and repeat-to-repeat spread *within a
  single arm* reaches 5×. The multi-second figures are evidence about this
  filesystem under this load, not about L20s, and not about shared caches in
  general.
* **One process per GPU is not what the primary run measured.** GPUs 4 and 5 are
  held in `Exclusive_Process` compute mode by another user's vLLM workers
  (42.9 GB each; context creation returns `CUDA_ERROR_DEVICE_UNAVAILABLE 46`), so
  every rank shared GPU 3 and confirmed `/dev/nvidia3` through `/proc/self/fd`
  against the UUID recorded in each rank's own record. A supplement on two
  distinct devices is in `raw/multigpu-summary.md`: the duplicate compilation is
  unchanged, but the warm-cache advantage at `add`/P=4 falls from 9.9× to 2.5×.
  Read the single-device magnitudes together with that correction.
* **Not measured:** an idle machine or a low-latency filesystem, NFS or
  multi-host sharing, any lock-based or otherwise coordinated arm — so there is
  no measurement here of what a per-key lock would save — how much of the
  directory-metadata cost such a lock would remove, kernels larger than
  `add`/`gemm`, and 30 trials per configuration (a replication run at load
  average > 1300 reached 13–14 shared-cold trials before it was stopped; in 3 of
  its 52 shared-cold rounds a racing early finisher deduplicated one compile, and
  those rounds are reported at their observed counts rather than smoothed).
* **Phase 2 is not implemented here.** This directory measures; it adds no
  cross-process coordination, changes no cache semantics, and makes no claim that
  the stampede is fixed.

## 9. Appendix: the strict-parsing edit in the coordinator is measurement-neutral

The last commit also removed the coordinator's defensive handling of its own
inputs: a malformed `STAMPEDE_READY` / `STAMPEDE_RESULT` line is no longer
swallowed, `record()` no longer falls back to the stdout copy or to a placeholder
record when the rank's own JSON file cannot be read, `args.tag` replaced
`getattr(args, "tag", "")`, and the per-rank GPU entry is indexed rather than
defaulted — `gpus[real_index]` and `gpu["uuid"]`, because a defaulted empty
expected-UUID would make the rank's UUID cross-check vacuously true. The primary
run predates that edit, so it was re-measured rather than assumed:

* **Nothing measured here is produced by the changed code.** Every latency in §5
  is timed inside the rank, or derived from the wrapper log that bash writes.
  The only changed statement anywhere near a timed region is the GPU dict lookup
  in `spawn_group`, and `group_spawn_wall_ns` is captured *before* that loop, so
  it can move `group_to_all_correct_ms` by nanoseconds on a metric of seconds.
  `scripts/tileiras_wrap.sh` — the only script that runs inside the timed region —
  is byte-identical before and after the edit (sha256 `d92e8182…`), and its mtime
  precedes every run reported here.
* **A/B, identical worker binary and device, arms alternated inside each
  repetition** (3 repetitions × 5 trials × P=4, GPU 3, 2026-09-19 01:51–02:04).
  Both arms ran one build of the committed worker source (sha256 `0e8c383f…`);
  the §2 binary differs from that source only by the `#[allow(dead_code)]`
  attribute added to it afterwards, which cannot change code generation.
  `barrier_to_result_ms`, per-run median of 5 trials:

  | configuration | pre-edit runs | post-edit runs |
  |---|---|---|
  | shared / add / P=4 | 571 / 1546 / 987 (median 987) | 981 / 756 / 536 (median 756) |
  | shared / gemm / P=4 | 1143 / 5662 / 1281 (median 1281) | 2335 / 1275 / 1575 (median 1575) |
  | shared-warm / add / P=4 | 256 / 260 / 281 (median 260) | 281 / 270 / 301 (median 281) |
  | shared-warm / gemm / P=4 | 342 / 380 / 381 (median 380) | 378 / 352 / 397 (median 378) |

  The warm arm is the internal control: it compiles nothing (0 spawn attempts,
  0 ms of compiler wall time, 4 disk hits in every round of both arms), so its
  spread is pure host noise — and it is ±10%, which is the same size as the
  largest between-arm difference. The cold arm's spread *within* one arm reaches
  5× over the same period, and the sign of the difference is not consistent
  across configurations, so nothing here is attributable to the edit. Spawn
  counts are exactly 4 per cold round and 0 per warm round in both arms, and all
  20 rounds of all six runs were valid.

The edit changes failure behaviour only: a run that cannot read its own evidence
now stops instead of substituting a placeholder. It does not move any number in
§5, so there is nothing to revert.
