---
orphan: true
---

# The Tile Programming Model

## Thinking in Tiles

The fundamental unit of computation in cuTile Rust is the **tile** — an immutable multi-dimensional array fragment that lives in GPU registers during kernel execution. You load data from tensors into tiles, compute on tiles, and store the results back. The compiler maps tiles onto the hardware memory hierarchy — including shared memory, caches, and registers — so you never manage these resources yourself.

![Thread-centric vs Tile-centric programming mental model](../_static/images/mental-model-shift.svg)

## Tile-Based vs Thread-Based Programming

Traditional CUDA programming asks you to think in terms of individual threads and explicit thread indices:

```cpp
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

cuTile Rust shifts the model to tiles of data:

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    c: &mut Tensor<f32, S>,
    a: &Tensor<f32, {[-1, -1]}>,
    b: &Tensor<f32, {[-1, -1]}>
) {
    let tile_a = load_tile_like(a, c);
    let tile_b = load_tile_like(b, c);
    c.store(tile_a + tile_b);
}
```

Instead of managing thread indices directly, you describe what should happen to one tile-shaped region of the data. The compiler and runtime handle how that work is mapped onto the underlying GPU execution model.

## Tile Blocks and Tile Threads

A **tile block** is a logical thread and the basic unit of concurrent execution on the GPU. Each tile block runs the kernel function once, operating on one partition of the data. A tile block is identified by its coordinates, obtained via `get_tile_block_id()`:

```rust
let pid: (i32, i32, i32) = get_tile_block_id();    // This block's (x, y, z) coordinates
let npids: (i32, i32, i32) = get_num_tile_blocks(); // Total grid dimensions
```

The cuTile Rust compiler maps each tile block to one or more underlying CUDA execution units (thread blocks, clusters, or warps) depending on the target architecture — but from the programmer's perspective, a tile block is simply a single-threaded context that processes one tile of data.

The terms **tile block** and **tile thread** are interchangeable. The API uses `get_tile_block_id()` and `get_num_tile_blocks()`, while the guides often say "tile thread" to emphasize the single-threaded programming model.

## Partitioning

To process a large tensor, you **partition** it — dividing the tensor into a grid of equally sized sub-regions, each of which is processed by one tile block. Partitioning works differently for mutable outputs and read-only inputs.

### Host-Side Partitioning (Mutable Tensors)

Mutable tensors must be partitioned on the host side before kernel launch:

```rust
let tensor = zeros(&[1024, 1024]).sync_on(&stream)?;
let partitioned = tensor.partition([64, 64]);  // 16×16 = 256 sub-tensors
```

![Partitioning divides data among tiles for parallel processing](../_static/images/vector-addition-partitioning.svg)

Calling `.partition([M, N])` on a `Tensor<T>` produces a `Partition<Tensor<T>>` — a host-side wrapper that records the `partition_shape` alongside the original tensor. The `partition_shape` determines the static shape `S` that the kernel sees: passing a `Partition` with `partition_shape = [64, 64]` means the kernel receives a `&mut Tensor<T, {[64, 64]}>`.

Only mutable tensors must be partitioned on the host side. This is because each `&mut Tensor` sub-region is written to by exactly one tile block, satisfying Rust's exclusive access requirement for mutable memory: at most one writer may access a given region at a time. By partitioning before launch, the system guarantees that no two tile blocks write to overlapping memory.

The generated launcher code accepts `Partition<Tensor<T>>` for every `&mut Tensor` parameter and `Arc<Tensor<T>>` for every `&Tensor` parameter.

### Device-Side Partitioning (Read-Only Tensors)

Read-only inputs are passed as `Arc<Tensor<T>>` on the host side — no host-side partitioning required. Multiple tile blocks may safely read from the same tensor or overlapping regions simultaneously, so there is no exclusive-access constraint to enforce.

Instead, read-only tensors are partitioned **inside the kernel** using `.partition(const_shape![M, N])`:

```rust
let part_x = x.partition(const_shape![BM, BK]);
let tile = part_x.load([pid.0, i]);
```

Because the partitioning happens on the device side, the same `&Tensor` can be partitioned in different ways within the same kernel. For example, in GEMM the input matrices `x` and `y` are each partitioned with a different shape inside the kernel body (`const_shape![BM, BK]` and `const_shape![BK, BN]` respectively), even though both were passed as plain `Arc<Tensor<T>>` from the host.

## The Grid

The **grid** determines how many tile blocks run. It can be specified explicitly or inferred from the partitioned tensors passed to the kernel.

### Grid Inference

A host-side partition's grid is computed by dividing the tensor's shape by the partition shape, rounding up:

```text
grid[i] = ceil(tensor_shape[i] / partition_shape[i])
```

The result is mapped to a 3D tuple `(x, y, z)`, with trailing dimensions set to 1 for tensors of rank less than 3. The mapping is direct and order-preserving: tensor dimension 0 maps to grid `x`, dimension 1 to `y`, and dimension 2 to `z`.

For example, a `[128, 256]` tensor partitioned with `[32, 64]` produces a grid of `(4, 4, 1)`:

```text
Tensor shape:    [128, 256]
Partition shape: [ 32,  64]
Grid:            (ceil(128/32), ceil(256/64), 1) = (4, 4, 1)
```

### From Grid Coordinates to Sub-Tensors

Inside the kernel, `get_tile_block_id()` returns the tile block's `(x, y, z)` coordinates in the grid. These coordinates correspond directly to the sub-tensor indices in the partition. For the example above, tile block `(2, 1, 0)` processes the sub-tensor at rows `2×32..3×32` and columns `1×64..2×64` — that is, the region `[64:96, 64:128]` of the original tensor.

For a `&mut Tensor`, this mapping is implicit — the kernel receives the sub-tensor directly, and loads and stores operate within the sub-tensor's bounds. For a `&Tensor` partitioned on the device side, you use the tile block ID to index into the partition explicitly:

```rust
let pid: (i32, i32, i32) = get_tile_block_id();

// For a &mut Tensor, the sub-tensor is passed directly:
let tile_z = z.load();       // Loads this block's sub-tensor

// For a &Tensor, use pid to index into a device-side partition:
let part_x = x.partition(const_shape![BM, BK]);
let tile_x = part_x.load([pid.0, i]);  // Loads tile at row pid.0, column i
```

### Launch Grid Inference

At kernel launch time, the launcher calls `.grid()` on each `&mut Tensor` parameter's host-side `Partition` and collects the resulting grids. If no explicit grid is specified via `.grid()` or `.const_grid()`, the launch grid is **inferred** from these partition grids:

```rust
// Grid is inferred from z's partition: (16, 16, 1)
let z = zeros(&[1024, 1024]).sync_on(&stream)?.partition([64, 64]);
let (z, _x, _y) = add(z, x, y).sync_on(&stream)?;
```

When multiple `&mut Tensor` parameters are present, all of their inferred grids must match or the launch will fail with an error.

### Explicit Grid

You can also set the grid manually, which overrides inference:

```rust
launcher.grid((16, 16, 1)).sync_on(&stream)?;
```

Each tile block receives unique 3-dimensional coordinates within the grid via `get_tile_block_id()`.

## Concurrent and Parallel Execution

When a kernel launches, the GPU's hardware scheduler assigns tile blocks to Streaming Multiprocessors (SMs) as resources become available. Tile blocks that fit on available SMs run **in parallel** — simultaneously on separate hardware units. The full set of tile blocks runs **concurrently** — their relative order of execution is unspecified and they are independent of one another.

This matches Rust's distinction between concurrency and parallelism: parallelism is work happening at the exact same time on different hardware, while concurrency is independently executing tasks making progress over time.

## Tile Types

| Type | Description | Lives In |
|------|-------------|----------|
| `Tensor<E, S>` | Multi-dimensional array in global memory; passed as kernel arguments | GPU DRAM (HBM) |
| `Partition<E, S>` | Logical division of a tensor into sub-regions | Metadata only |
| `Tile<E, S>` | Immutable data fragment for computation; compile-time static shapes | GPU registers |

The flow is always: **Tensor → Partition → Tile → Compute → Store**

You only load from and store to HBM (global memory). The underlying [Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/) runtime handles mapping your tiles onto the hardware memory hierarchy — including shared memory, caches, and registers — so you never need to manage these resources yourself.

---

Continue to [Data Model & Types](data-model.md) to understand cuTile Rust's type system.
