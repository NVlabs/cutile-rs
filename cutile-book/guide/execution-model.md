# The Execution Model

This page describes how cuTile Rust programs execute on NVIDIA GPUs — the abstract machine the compiler targets, how entry points are declared, how tile blocks run concurrently, and which values are compile-time constants vs. runtime inputs.

---

## The Abstract Machine

cuTile Rust is built on [Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/): a tile virtual machine and instruction set that models the GPU as a tile-based processor. Unlike the traditional SIMT (Single Instruction Multiple Thread) model, Tile IR lets you program in terms of tiles (multi-dimensional array fragments) rather than individual threads.

The abstract machine maps to CUDA like this:

| Abstract Concept | What It Represents | CUDA Mapping |
|------------------|--------------------|------------------|
| **Tile Block** | A single logical thread of execution that processes one tile of data | One or more thread blocks, clusters, or warps (compiler-determined) |
| **Tile** | An immutable multi-dimensional array fragment with compile-time static shape | Data in registers |
| **Tensor** | A multi-dimensional array in global memory, accessed via structured views | Data in HBM, accessed through typed pointers with shape and stride metadata |

The mapping of the grid and individual tile blocks to underlying hardware threads is abstracted away and handled by the compiler: thread block and cluster configuration, register allocation, shared memory staging, memory coalescing, and Tensor Core utilization.

Execution happens across two spaces: the **host side** (CPU) allocates GPU memory, launches kernels, manages data transfers, and coordinates async operations; the **device side** (GPU) concurrently runs kernel code on tile blocks, operating on tiles in registers and accessing global memory through tensors. See [Device Operations](device-operations.md) for the host-side execution story.

---

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

Entry points have four rules:

1. **Must be in a module** — Entry points must be inside a `#[cutile::module]` block.
2. **Const generics for tile size** — An output tensor's shape must be static. It determines the output tensor's tile size.
3. **Tensor parameters** — All data passes through `Tensor` references.
4. **No return values** — Results are written to output tensors.

---

## Tile Concurrency

When a kernel launches, the runtime creates a **grid of tile blocks**. Each tile block runs the same kernel function independently, processing one sub-tensor of the data:

```{figure} ../_static/images/tile-parallelism.svg
:width: 100%
:alt: Tile concurrency showing how tensors are partitioned into sub-tensors
```

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

Tile blocks that fit on available SMs run in parallel; the full set of tile blocks runs concurrently, with unspecified relative order. See [Thinking in Tiles](thinking-in-tiles.md) for the partitioning rules that drive the grid geometry.

---

## Compile-Time vs Runtime Values

cuTile Rust enforces compile-time constantness for values the compiler needs to reason about. **Compile-time constants** include tile dimensions (shape of tiles in registers), element dtypes, and reduction axes. **Runtime values** include tensor dimensions (size of input tensors), tensor data, and grid size (number of tile blocks to launch).

```rust
#[cutile::entry()]
fn kernel<const TILE_SIZE: [i32; 2]>(
    output: &mut Tensor<f32, TILE_SIZE>,  // Tile shape: compile-time constant
    input: &Tensor<f32, {[-1, -1]}>       // Tensor shape: runtime
) {
    let tile = load_tile_like_2d(input, output);

    let max_vals = reduce_max(tile, 1i32);  // Reduction axis: compile-time constant

    output.store(tile);
}
```

Compile-time constants drive specialization: each unique combination of const generic values triggers its own compilation. Keeping the set of distinct specializations small matters for compile time and binary size.

---

Continue to [Device Operations](device-operations.md) for the host-side execution story. For tuning tile sizes and architecture-specific hints, see [Tuning for Performance](performance-tuning.md).
