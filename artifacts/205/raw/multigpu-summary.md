# Supplementary run: one process per physical GPU

The primary run placed every rank on GPU 3, because GPUs 4 and 5 were held in
`Exclusive_Process` compute mode by another user's vLLM workers. Multi-GPU was
therefore the largest gap in that run. This supplement closes it.

Command (13 rounds per configuration, 0 invalid):

```bash
scripts/run_jit_stampede_205.sh --trials 10 --correctness-rounds 3 \
  --p 1,2,4 --modes nocache,shared,shared-warm --kernels add,gemm \
  --devices 1,3 --tag multigpu
```

Ranks are assigned round-robin, so with `--devices 1,3` a P=4 group uses two
distinct physical devices. Each rank independently confirmed its device and
recorded the UUID:

```
rank 0 -> physical GPU 1   234 rounds   uuid GPU-739e6fcb-7d63-24bb-bd98-23892ff84948
rank 1 -> physical GPU 3   156 rounds   uuid GPU-dfe309b3-df0d-f7f1-642b-40c03e15924f
rank 2 -> physical GPU 1    78 rounds
rank 3 -> physical GPU 3    78 rounds
```

## Duplicate compiles still scale with process count

Sum of `tileiras` spawn attempts over all ranks, per round (13 rounds/config):

| mode | kernel | P | min | median | max |
|---|---|---|---|---|---|
| nocache | add / gemm | 1, 2, 4 | P | P | P |
| shared | gemm | 1, 2, 4 | P | P | P |
| shared | add | 4 | 3 | 4 | 4 |
| shared-warm | add / gemm | 1, 2, 4 | 0 | 0 | 0 |

So sharing one cold cache directory **across two physical devices** still
produces one compile per cold process — the redundancy is per cold process, not
an artefact of a single device. One of 13 `shared`/`add`/P=4 rounds saw an early
finisher dedup (3 attempts instead of 4).

## The magnitude is much smaller here

`barrier_to_result_ms`, median:

| kernel | P | single-GPU shared | **multi-GPU shared** | single-GPU warm | **multi-GPU warm** |
|---|---|---|---|---|---|
| add | 2 | 1694.3 | **728.5** | 290.2 | **269.3** |
| add | 4 | 3134.7 | **762.1** | 317.0 | **305.5** |
| gemm | 2 | 3304.8 | **1416.9** | 384.5 | **366.3** |
| gemm | 4 | 2817.7 | **1340.5** | 404.7 | **393.1** |

The warm-cache advantage is 2-3x here, against roughly 10x in the primary run.
That is the expected result given the phase analysis in the README: a large part
of the single-GPU penalty was contention on one saturated device and its disk,
not the redundant compilation itself. With the device load spread, the saving is
mostly the compiler work and CPU time, and the absolute wall-clock latency is
already only 0.7-1.4 s.

**This confirms the caveat in the main report, and the direction is that the
general-case wall-clock benefit is smaller than the primary run suggests.**
