# Execution Model

This page describes how cuTile Rust programs execute on NVIDIA GPUs.

## Abstract Machine

cuTile Rust is built on [Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/) — a tile virtual machine and instruction set that models the GPU as a tile-based processor. Unlike the traditional SIMT (Single Instruction Multiple Thread) model, Tile IR enables programming in terms of tiles (multi-dimensional array fragments) rather than individual threads.

cuTile Rust targets an **abstract machine** that maps to CUDA:

| Abstract Concept | What It Represents | CUDA Mapping |
|------------------|--------------------|------------------|
| **Tile Block** | A single logical thread of execution that processes one tile of data | One or more thread blocks, clusters, or warps (compiler-determined) |
| **Tile** | An immutable multi-dimensional array fragment with compile-time static shape | Data in registers |
| **Tensor** | A multi-dimensional array in global memory, accessed via structured views | Data in HBM, accessed through typed pointers with shape and stride metadata |

The mapping of both the grid and individual tile blocks to the underlying hardware threads is abstracted away and handled entirely by the compiler. This includes:
- Thread block and cluster configuration.
- Register allocation.
- Shared memory staging.
- Memory coalescing.
- Tensor Core utilization.

## Execution Spaces

cuTile Rust operates across two execution spaces, more commonly referred to as the host-side and device-side:

### Host Side (CPU)

- Allocates GPU memory.
- Launches kernels.
- Manages data transfers.
- Coordinates async operations.

```rust
// Host code: sets up data and launches the kernel
fn main() -> Result<(), Error> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;

    let x: Arc<Tensor<f32>> = ones([1024, 1024]).arc().sync_on(&stream)?;
    let y: Arc<Tensor<f32>> = ones([1024, 1024]).arc().sync_on(&stream)?;
    let z = zeros([1024, 1024]).sync_on(&stream)?.partition([64, 64]);

    let (z, _x, _y) = add(z, x, y).sync_on(&stream)?;
    Ok(())
}
```

### Device Side (GPU)

- Concurrently executes kernel code on tile threads.
- Operates on tiles in registers.
- Accesses global memory through tensors.

```rust
// Device code: runs on GPU
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    c: &mut Tensor<f32, S>,
    a: &Tensor<f32, {[-1, -1]}>,
    b: &Tensor<f32, {[-1, -1]}>
) {
    // The compiler performs automatic parallelization, 
    // executing the following code on hundreds of GPU threads in parallel
    let tile_a = load_tile_like_2d(a, c);
    let tile_b = load_tile_like_2d(b, c);
    c.store(tile_a + tile_b);
}
```

## Kernel Entry Points

The `#[cutile::entry()]` attribute marks a function as a GPU kernel entry point:

```rust
#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn my_kernel<const TILE_SIZE: [i32; 2]>(
        output: &mut Tensor<f32, TILE_SIZE>,
        input: &Tensor<f32, {[-1, -1]}>
    ) {
        // Kernel body
    }
}
```

### Entry Point Rules

1. **Must be in a module** — Entry points must be inside a `#[cutile::module]` block.
2. **Const generics for tile size** — An output tensor's shape must be static. It determines the output tensor's tile size.
3. **Tensor parameters** — All data passes through `Tensor` references.
4. **No return values** — Results are written to output tensors.

## Tile Concurrency

When a kernel launches, the runtime creates a **grid of tile blocks**:

```{figure} ../_static/images/tile-parallelism.svg
:width: 100%
:alt: Tile concurrency showing how tensors are partitioned into sub-tensors
```

Each tile thread:
- Processes one sub-tensor of data.
- Runs independently of other tile threads.
- Executes the same kernel function.

### Tile Block Identification

Within a kernel, each tile block can query its coordinates in the grid and the total grid dimensions:

```rust
#[cutile::entry()]
fn kernel<const S: [i32; 2]>(
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, {[-1, -1]}>
) {
    let pid: (i32, i32, i32) = get_tile_block_id();    // This block's (x, y, z)
    let grid: (i32, i32, i32) = get_num_tile_blocks();  // Grid dimensions

    // For element-wise ops, load_tile_like_2d uses the output's
    // partition to determine which region this block processes:
    let tile = load_tile_like_2d(input, output);
    output.store(tile);
}
```

## Constantness

cuTile Rust enforces **compile-time constantness** for key values:

### Compile-Time Constants

These must be known at compile time. Some compile-time constants include:
- **Tile dimensions** — Shape of tiles in registers.
- **Dtype** — Element data types.
- **Reduction axes** — Which dimensions to reduce.

```rust
// Tile shape is a compile-time constant, specified via const generics
#[cutile::entry()]
fn kernel<const TILE_SIZE: [i32; 2]>(
    output: &mut Tensor<f32, TILE_SIZE>,  // [64, 64] known at compile time
    input: &Tensor<f32, {[-1, -1]}>       // Dynamic size
) {
    let tile = load_tile_like_2d(input, output);
    
    // Reduction axis is a compile-time constant
    let max_vals = reduce_max(tile, 1i32);  // Reduce along axis 1

    output.store(tile);
}
```

### Runtime Values

These can vary at runtime:
- **Tensor dimensions** — Size of input tensors.
- **Tensor data** — Actual element values.
- **Grid size** — Number of tile blocks to launch.

## Python Subset (Comparison)

If you're familiar with [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/), here's how concepts map:

| cuTile Python | cuTile Rust |
|---------------|-----------|
| `@ct.kernel` | `#[cutile::entry()]` |
| `ct.load()` | `load_tile_like_2d()` |
| `ct.store()` | `tensor.store()` |
| `ct.bid(0)` | Implicit via partition |
| `ct.launch()` | Async operation + `.await` |

Both use the same underlying compilation pipeline and generate equivalent GPU code.

---

## Next Steps

- Learn about the [Data Model](data-model.md) for details on types and shapes
- Explore [Memory Hierarchy](memory-hierarchy.md) for performance optimization
- See [Async Execution](async-execution.md) for concurrent CPU/GPU work
- See [Interoperability](interoperability.md) for integrating custom CUDA kernels into the `DeviceOperation` model
