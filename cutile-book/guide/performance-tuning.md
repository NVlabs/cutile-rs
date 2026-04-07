# Performance Tuning

This guide covers techniques for optimizing cuTile Rust kernel performance. For algorithms where peak performance requires warp-level control or integration with hand-tuned CUDA C++ kernels, see [Interoperability](interoperability.md).

## The Performance Mindset

GPU performance optimization focuses on three key areas:

1. **Memory bandwidth** — Moving data efficiently
2. **Compute utilization** — Keeping ALUs busy
3. **Occupancy** — Maximizing parallel execution

```{figure} ../_static/images/performance-triangle.svg
:width: 100%
:alt: The GPU performance triangle showing memory bandwidth, compute utilization, and occupancy
```

## Architecture-Specific Configuration

### Optimization Hints

cuTile Rust provides `optimization_hints` at two levels: **entry-level** (kernel-wide) and **per-op** (on individual load/store operations).

#### Entry-Level Hints

Set `occupancy` and `num_cta_in_cga` in the entry annotation. These can also be overridden at runtime via `CompileOptions`:

```rust
#[cutile::entry(
    optimization_hints = (
        tensor_dim_factor = 16,           // Memory alignment hint
        sm_120 = (                         // Hopper-specific hints
            num_cta_in_cga = 2,           // CTAs per Cooperative Group
            occupancy = 2,                 // Target occupancy
        ),
        sm_90 = (                          // Ampere-specific hints
            num_cta_in_cga = 1,
        ),
    )
)]
fn optimized_kernel<const S: [i32; 2]>(...) { ... }
```

To override entry-level hints at runtime (e.g. for autotuning):

```rust
use cutile::tile_kernel::CompileOptions;

let result = my_kernel(input)
    .compile_options(CompileOptions::default().occupancy(4).num_cta_in_cga(2))
    .grid(grid)
    .await;
```

Different `CompileOptions` values trigger separate JIT compilations and are part of the kernel cache key.

#### Per-Op Hints

`latency` and `disallow_tma` are set directly on individual load/store operations:

```rust
// Per-op latency hint on a view load
let tile: Tile<f32, S> = load_from_view(&partition, idx, Some(4), false);

// Disable TMA on a specific store
unsafe { store_to_view_mut(&mut partition, tile, idx, None, true); }

// Per-op latency on a pointer load
let (values, token) = load_ptr_tko(ptrs, "weak", "tl_blk", None, None, None, Some(4));
```

### Hint Reference

#### Entry-Level Hints

| Hint | Description | Default |
|------|-------------|---------|
| `tensor_dim_factor` | Memory alignment factor | Auto |
| `num_cta_in_cga` | CTAs in Cooperative Group Array | 1 |
| `occupancy` | Target occupancy level | Auto |

#### Per-Op Hints (on load/store operations)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `latency` | Latency optimization hint (`Option<i32>`) | `None` (compiler decides) |
| `disallow_tma` | Disable Tensor Memory Accelerator for this op (`bool`) | `false` (TMA allowed) |

### Tile Size Selection

Tile sizes significantly impact performance. General guidelines:

| GPU Architecture | Recommended Tile Sizes |
|------------------|----------------------|
| Ampere (A100) | `[128, 128]`, `[64, 64]`, `[256, 64]` |
| Hopper (H100) | `[128, 128]`, `[64, 128]`, `[128, 256]` |
| Ada (RTX 4090) | `[64, 64]`, `[128, 64]` |

```rust
// Choose tile size based on workload characteristics
#[cutile::entry(
    optimization_hints = (tensor_dim_factor = 16,)
)]
fn matmul<const TILE_M: i32, const TILE_N: i32, const TILE_K: i32>(
    c: &mut Tensor<f32, {[TILE_M, TILE_N]}>,
    a: &Tensor<f32, {[-1, -1]}>,
    b: &Tensor<f32, {[-1, -1]}>
) {
    // TILE_M=128, TILE_N=128, TILE_K=32 is often a good starting point
}
```

### Register Pressure

Larger tiles use more registers, potentially reducing occupancy:

| Tile Size | Registers (approx) | Max Occupancy |
|-----------|-------------------|---------------|
| `[32, 32]` | ~32 | High |
| `[64, 64]` | ~64-128 | Medium-High |
| `[128, 128]` | ~256+ | Medium |

**Trade-off:** Larger tiles = fewer memory transactions, but lower occupancy.

### Unchecked Accesses

For maximum performance, disable bounds checking (use with caution):

```rust
#[cutile::entry(unchecked_accesses = true)]
unsafe fn fast_kernel<const S: [i32; 2]>(...) {
    // No bounds checking — programmer must ensure correctness
}
```

Setting `unchecked_accesses = true` removes all runtime bounds checks on `load` and `store` operations. The entry point must be marked `unsafe`, and the call site must use an `unsafe` block. Even with bounds checks disabled, cuTile Rust's compile-time checks still apply: tile shapes, `mma` dimensions, and element types are still validated.

### Eliminating Bounds Checks with Static Shapes

An alternative to `unchecked_accesses` is to make all tensor dimensions static const generics and provide the launch grid via `.const_grid()`. When the JIT compiler knows every dimension at compile time, it can prove that all partition accesses are in bounds and optimize the bounds checks away entirely — no `unsafe` required.

```rust
#[cutile::entry()]
fn gemm<
    E: ElementType,
    const BM: i32, const BN: i32, const BK: i32,
    const M: i32, const N: i32, const K: i32,
>(
    z: &mut Tensor<E, { [BM, BN] }>,
    x: &Tensor<E, { [M, K] }>,    // Fully static — no dynamic dimensions
    y: &Tensor<E, { [K, N] }>,    // Fully static — no dynamic dimensions
) {
    let part_x = x.partition(const_shape![BM, BK]);
    let part_y = y.partition(const_shape![BK, BN]);
    let pid: (i32, i32, i32) = get_tile_block_id();

    let mut acc = load_tile_mut(z);
    for i in 0i32..(K / BK) {
        // The compiler knows K, BK, M, N, BM, BN at JIT time.
        // It can prove pid.0 < M/BM, i < K/BK, pid.1 < N/BN,
        // so these loads are guaranteed in bounds — checks eliminated.
        let tile_x = part_x.load([pid.0, i]);
        let tile_y = part_y.load([i, pid.1]);
        acc = mma(tile_x, tile_y, acc);
    }
    z.store(acc);
}
```

On the host side, pass the grid as a compile-time constant via `.const_grid()`:

```rust
let grid = z.grid()?;
let (z, _x, _y) = gemm(z, x, y)
    .const_grid(grid)
    .generics(generics)
    .sync_on(&stream)?;
```

`.const_grid()` passes the grid dimensions as compile-time constants to the JIT compiler, which enables it to reason about the range of `get_tile_block_id()` values and prove that all partition accesses derived from them are within bounds.

**Tradeoff:** Every new combination of `M`, `N`, `K`, `BM`, `BN`, `BK` triggers a JIT recompilation. Use this approach when problem sizes come from a small, known set (the JIT cache makes repeated sizes free). Use `unchecked_accesses` when problem sizes vary widely and compilation overhead matters more than safety.

## Memory Optimization

### Coalesced Memory Access

The GPU memory system is optimized for **coalesced access** — adjacent threads accessing adjacent memory:

```rust
// GOOD: Coalesced access pattern
// Threads in a warp access consecutive elements
let tile = load_tile_like_2d(input, output);  // Automatic coalescing

// The load operation automatically generates coalesced memory transactions
```

### Load/Store Hints

Use load hints to optimize memory access patterns:

```rust
// Standard load
let tile = load_tile_like_2d(input, output);

// Load with cache hints (when available)
// The compiler may generate different PTX based on access patterns
```

### Memory Hierarchy Utilization

Maximize use of faster memory levels:

| Memory Level | Relative Speed | Latency | Strategy |
|--------------|----------------|---------|----------|
| Registers | Fastest | ~0 cycles | Keep data in tiles |
| Shared Memory | Very fast | ~20 cycles | Reuse across iterations |
| L2 Cache | Fast | ~200 cycles | Temporal locality |
| Global Memory | Slowest | ~400 cycles | Minimize accesses |

```rust
// Pattern: Load once, compute many
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

## Compute Optimization

### Tensor Core Utilization

cuTile Rust automatically uses Tensor Cores for matrix operations when shapes align:

```rust
// Tensor Core requirements:
// - Matrix dimensions divisible by 16 (f16) or 8 (tf32)
// - Appropriate data types (f16, bf16, tf32)

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

### Arithmetic Intensity

**Arithmetic intensity** = FLOPs / Bytes transferred

Higher arithmetic intensity = better GPU utilization.

| Operation | Arithmetic Intensity | Bound |
|-----------|---------------------|-------|
| Vector Add | 0.125 | Memory |
| Matrix-Vector | 1-2 | Memory |
| Matrix-Matrix | O(N) | Compute |
| Fused Softmax | ~10+ | Compute |

```rust
// Low arithmetic intensity: simple elementwise
let c = a + b;  // 1 FLOP per 12 bytes (3 f32s)

// High arithmetic intensity: matrix multiply
let c = mma(a, b, acc);  // O(N) FLOPs per element

// Very high: fused operations
let result = softmax(matmul(a, b));  // Many FLOPs, few memory ops
```

### Instruction-Level Parallelism

The compiler automatically exploits ILP, but you can help:

```rust
// Independent operations can execute in parallel
let sum1 = reduce_sum(tile1, 1i32);
let sum2 = reduce_sum(tile2, 1i32);  // Can overlap with sum1

// Dependent operations serialize
let step1 = tile * 2.0;
let step2 = step1 + 1.0;  // Must wait for step1
```

## Kernel Fusion

Fusing multiple operations into one kernel reduces memory traffic:

```rust
// UNFUSED: Multiple kernels, multiple memory round-trips
// kernel1: y = a + b  (load a, b; store y)
// kernel2: z = y * c  (load y, c; store z)
// kernel3: w = exp(z) (load z; store w)
// Total: 6 loads + 3 stores

// FUSED: Single kernel, single memory round-trip
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
    
    // All in registers - no intermediate memory traffic
    let y = tile_a + tile_b;
    let z = y * tile_c;
    let result = exp(z);
    
    w.store(result);
}
// Total: 3 loads + 1 store (3x memory reduction!)
```

## Profiling

### Key Metrics

When profiling cuTile Rust kernels, focus on:

| Metric | Target | Tool |
|--------|--------|------|
| Memory Throughput | >80% of peak | Nsight Compute |
| Compute Throughput | >70% for compute-bound | Nsight Compute |
| Occupancy | >50% | Nsight Compute |
| Register Spills | 0 | Nsight Compute |

### Identifying Bottlenecks

```
Memory Bound:
- Low compute throughput
- High memory throughput (near peak)
- Solution: Increase arithmetic intensity, fuse kernels

Compute Bound:
- High compute throughput
- Low memory throughput
- Solution: Already optimal for this algorithm

Latency Bound:
- Low both compute and memory
- High stall cycles
- Solution: Increase parallelism, overlap operations
```

## Common Pitfalls

### 1. Uncoalesced Memory Access

```rust
// AVOID: Strided access patterns when possible
// The compiler handles this, but algorithm design matters
```

### 2. Excessive Synchronization

```rust
// Tile operations are designed to minimize sync points
// Trust the compiler to handle thread synchronization
```

### 3. Wrong Tile Size

```rust
// Too small: High overhead, poor utilization
const TILE_SIZE: [i32; 2] = [8, 8];  // Usually too small

// Too large: Register spills, low occupancy
const TILE_SIZE: [i32; 2] = [512, 512];  // Probably too large

// Just right: Balance of efficiency and occupancy
const TILE_SIZE: [i32; 2] = [64, 64];  // Good starting point
```

### 4. Ignoring Data Types

```rust
// f16/bf16 can double throughput on Tensor Cores
// Use when precision allows

#[cutile::entry()]
fn fast_matmul<const S: [i32; 2]>(
    c: &mut Tensor<f16, S>,  // Half precision = 2x throughput
    a: &Tensor<f16, {[-1, -1]}>,
    b: &Tensor<f16, {[-1, -1]}>
) { ... }
```

## Performance Checklist

- [ ] **Tile size** appropriate for workload and GPU architecture
- [ ] **Memory access** patterns are coalesced
- [ ] **Kernel fusion** applied where possible
- [ ] **Data types** optimized (f16/bf16 for Tensor Cores)
- [ ] **Arithmetic intensity** maximized
- [ ] **Occupancy** balanced with tile size
- [ ] **Profiled** with Nsight Compute

---

## Next Steps

- See [Memory Hierarchy](memory-hierarchy.md) for detailed memory optimization
- Learn about [Async Execution](async-execution.md) for overlapping operations
- Read [Interoperability](interoperability.md) for integrating custom CUDA kernels when tile programming isn't enough
- Check [Debugging](debugging.md) for troubleshooting performance issues
