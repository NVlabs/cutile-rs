# Tuning for Performance

GPU performance optimization balances three concerns: memory bandwidth (moving data efficiently), compute utilization (keeping ALUs busy), and occupancy (maximizing parallel execution). Good kernels are well-balanced across all three; poor kernels are bottlenecked on one.

```{figure} ../_static/images/performance-triangle.svg
:width: 100%
:alt: The GPU performance triangle showing memory bandwidth, compute utilization, and occupancy
```

For algorithms where peak performance requires warp-level control or integration with hand-tuned CUDA C++ kernels, see [Interoperability](interoperability.md).

---

## Compiler Hints and Specialization

cuTile Rust provides `optimization_hints` at two levels: **entry-level** (kernel-wide) and **per-op** (on individual load/store operations).

Entry-level hints go on the entry annotation. They can also be overridden at runtime via `CompileOptions` for autotuning — different values trigger separate JIT compilations and are part of the kernel cache key:

```rust
#[cutile::entry(
    optimization_hints = (
        sm_120 = (                       // Blackwell-specific hints
            num_cta_in_cga = 2,
            occupancy = 2,
            max_divisibility = 16,
        ),
        sm_90 = (                        // Hopper-specific hints
            num_cta_in_cga = 1,
        ),
    )
)]
fn optimized_kernel<const S: [i32; 2]>(...) { ... }

// Runtime override for autotuning:
use cutile::tile_kernel::CompileOptions;

let result = my_kernel(input)
    .compile_options(CompileOptions::default().occupancy(4).num_cta_in_cga(2))
    .grid(grid)
    .await;
```

Per-op hints (`latency`, `disallow_tma`) apply to individual load/store operations:

```rust
let tile: Tile<f32, S> = load_from_view(&partition, idx, Some(4), false);
unsafe { store_to_view_mut(&mut partition, tile, idx, None, true); }
let (values, token) = load_ptr_tko(ptrs, "weak", "tl_blk", None, None, None, Some(4));
```

| Level | Hint | Description | Default |
|---|---|---|---|
| Entry | `max_divisibility` | Cap on auto-inferred alignment divisor | 16 |
| Entry | `num_cta_in_cga` | CTAs in Cooperative Group Array | 1 |
| Entry | `occupancy` | Target occupancy level | Auto |
| Per-op | `latency` | Latency optimization hint (`Option<i32>`) | `None` (compiler decides) |
| Per-op | `disallow_tma` | Disable Tensor Memory Accelerator for this op | `false` (TMA allowed) |

**Tile size** significantly impacts performance. Larger tiles mean fewer memory transactions but more registers per block, reducing occupancy. General guidelines:

| GPU Architecture | Recommended Tile Sizes |
|------------------|------------------------|
| Ampere (A100) | `[128, 128]`, `[64, 64]`, `[256, 64]` |
| Hopper (H100) | `[128, 128]`, `[64, 128]`, `[128, 256]` |
| Ada (RTX 4090) | `[64, 64]`, `[128, 64]` |

| Tile Size     | Registers (approx) | Max Occupancy |
|---------------|--------------------|---------------|
| `[32, 32]`    | ~32                | High          |
| `[64, 64]`    | ~64-128            | Medium-High   |
| `[128, 128]`  | ~256+              | Medium        |

Two strategies remove bounds checks on tile loads and stores, depending on how stable your problem sizes are. **`unchecked_accesses = true`** (with `unsafe`) removes all runtime bounds checks from tile loads and stores. Compile-time shape and MMA-dimension checks still apply. Use when problem sizes vary widely and compilation overhead matters more than the safety net:

```rust
#[cutile::entry(unchecked_accesses = true)]
unsafe fn fast_kernel<const S: [i32; 2]>(...) {
    // No bounds checking — programmer must ensure correctness
}
```

**`.const_grid()` with fully static tensor shapes** keeps the safety net. When every dimension is a compile-time const and the grid is passed via `.const_grid()`, the JIT compiler can prove all partition accesses are in bounds and optimize the checks away — no `unsafe` needed:

```rust
#[cutile::entry()]
fn gemm<
    E: ElementType,
    const BM: i32, const BN: i32, const BK: i32,
    const M: i32, const N: i32, const K: i32,
>(
    z: &mut Tensor<E, { [BM, BN] }>,
    x: &Tensor<E, { [M, K] }>,    // Fully static
    y: &Tensor<E, { [K, N] }>,    // Fully static
) {
    let part_x = x.partition(const_shape![BM, BK]);
    let part_y = y.partition(const_shape![BK, BN]);
    let pid: (i32, i32, i32) = get_tile_block_id();

    let mut acc = load_tile_mut(z);
    for i in 0i32..(K / BK) {
        let tile_x = part_x.load([pid.0, i]);
        let tile_y = part_y.load([i, pid.1]);
        acc = mma(tile_x, tile_y, acc);
    }
    z.store(acc);
}

// On the host side:
let grid = z.grid()?;
let (z, _x, _y) = gemm(z, x, y)
    .const_grid(grid)
    .generics(generics)
    .sync_on(&stream)?;
```

The tradeoff: every new combination of const values triggers a JIT recompilation. Use this when problem sizes come from a small, known set — the JIT cache makes repeated sizes free.

---

## Memory Optimization

**Coalesced access** — adjacent threads reading adjacent memory locations — is how the GPU memory system is designed to be used. cuTile Rust's tile load operations automatically generate coalesced access patterns, so you get this for free from `load_tile_like_2d`, `Partition::load`, and the standard loading APIs.

**Keep data in registers.** Load once from global memory, compute many times in registers:

| Memory Level | Latency | Strategy |
|--------------|---------|----------|
| Registers | ~0 cycles | Keep data in tiles |
| Shared Memory | ~20 cycles | Reuse across iterations |
| L2 Cache | ~200 cycles | Temporal locality |
| Global Memory | ~400 cycles | Minimize accesses |

```rust
#[cutile::entry()]
fn fused_ops<const S: [i32; 2]>(
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, {[-1, -1]}>
) {
    // Single load from global memory
    let tile = load_tile_like_2d(input, output);

    // Multiple operations in registers (free!)
    let normalized = tile - reduce_max(tile, 1i32);
    let exp_vals = exp(normalized);
    let softmax = true_div(exp_vals, reduce_sum(exp_vals, 1));

    // Single store to global memory
    output.store(softmax);
}
```

**Kernel fusion** is the register strategy scaled up — combining multiple logical operations into a single kernel. A pipeline of 3 kernels might read and write intermediate results to global memory 6 times; fusing into one kernel eliminates most of those round-trips:

```rust
// UNFUSED: 3 kernels, 6 loads + 3 stores total.

// FUSED: 1 kernel, 3 loads + 1 store (3× memory reduction).
#[cutile::entry()]
fn fused<const S: [i32; 2]>(
    w: &mut Tensor<f32, S>,
    a: &Tensor<f32, {[-1, -1]}>,
    b: &Tensor<f32, {[-1, -1]}>,
    c: &Tensor<f32, {[-1, -1]}>
) {
    let tile_a = load_tile_like_2d(a, w);
    let tile_b = load_tile_like_2d(b, w);
    let tile_c = load_tile_like_2d(c, w);

    // All in registers — no intermediate memory traffic
    let y = tile_a + tile_b;
    let z = y * tile_c;
    let result = exp(z);

    w.store(result);
}
```

For the full memory hierarchy model and arithmetic intensity analysis, see [Where Data Lives](memory-hierarchy.md).

---

## Compute Optimization

**Tensor Cores** deliver massive throughput for matrix operations when shapes align. cuTile Rust automatically uses them when matrix dimensions are divisible by 16 (for `f16`/`bf16`) or 8 (for `tf32`), and element types are `f16`, `bf16`, or `tf32`:

```rust
#[cutile::entry()]
fn tensor_core_matmul<const M: i32, const N: i32>(
    c: &mut Tensor<f16, {[M, N]}>,  // f16 enables Tensor Cores
    a: &Tensor<f16, {[-1, -1]}>,
    b: &Tensor<f16, {[-1, -1]}>
) {
    let tile_a = load_tile_like_2d(a, c);
    let tile_b = load_tile_like_2d(b, c);

    // MMA automatically uses Tensor Cores
    let acc = constant(0.0f32, c.shape());
    let result = mma(tile_a, tile_b, acc);
    c.store(result);
}
```

**Arithmetic intensity** is FLOPs per byte transferred. Higher is better: high-intensity kernels are compute-bound rather than memory-bound.

| Operation | Arithmetic Intensity | Bound |
|-----------|----------------------|-------|
| Vector Add | ~0.1 | Memory |
| Matrix-Vector | 1-2 | Memory |
| Matrix-Matrix | O(N) | Compute |
| Fused Softmax | ~10+ | Compute |

See [Where Data Lives: Arithmetic Intensity](memory-hierarchy.md#arithmetic-intensity) for the full treatment.

**Instruction-level parallelism** (ILP) lets the compiler overlap independent operations. Write independent branches explicitly so the compiler can schedule them in parallel:

```rust
// Independent operations — compiler can overlap them
let sum1 = reduce_sum(tile1, 1i32);
let sum2 = reduce_sum(tile2, 1i32);  // Can execute concurrently

// Dependent operations — serialize
let step1 = tile * 2.0;
let step2 = step1 + 1.0;  // Must wait for step1
```

---

## Profiling and Pitfalls

Focus on four metrics when profiling with Nsight Compute:

| Metric | Target |
|---|---|
| Memory Throughput | >80% of peak for memory-bound kernels |
| Compute Throughput | >70% for compute-bound kernels |
| Occupancy | >50% |
| Register Spills | 0 |

Identify the bottleneck from the profile:
- **High memory throughput, low compute** → memory-bound; increase arithmetic intensity, fuse kernels.
- **Low memory throughput, high compute** → compute-bound; already near-optimal for this algorithm.
- **Low on both, high stall cycles** → latency-bound; increase parallelism, overlap independent operations.

Common pitfalls:
- **Wrong tile size.** `[8, 8]` is usually too small (overhead dominates); `[512, 512]` is usually too large (register spills, low occupancy). Start with `[64, 64]` or `[128, 128]`.
- **Wrong dtype.** Using `f32` when `f16`/`bf16` would suffice leaves 2× Tensor Core throughput on the table.
- **Excessive synchronization.** Let the compiler handle thread synchronization; avoid introducing extra sync points.
- **Algorithmic stride.** Tile operations coalesce automatically, but strided access patterns in your algorithm logic defeat this.

Pre-ship checklist: tile size appropriate for workload and architecture; memory access coalesced; kernel fusion applied where possible; data types optimized (`f16`/`bf16` for Tensor Cores); arithmetic intensity maximized; occupancy balanced against tile size; profiled with Nsight Compute.

---

Continue to [Interoperability](interoperability.md) for the escape hatch when tile programming isn't enough, or [Debugging and Profiling](debugging.md) for deeper troubleshooting.
