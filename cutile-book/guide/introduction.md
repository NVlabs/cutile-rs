# Introduction to cuTile Rust

## What is cuTile Rust?

**cuTile Rust** is a safe tile-based parallel programming model for Rust. It automatically leverages advanced hardware capabilities, such as Tensor Cores and Tensor Memory Accelerators, while providing portability across different NVIDIA GPU architectures. cuTile Rust enables the latest hardware features without requiring code changes. On the host side, it provides a safe API for allocating device tensors, partitioning mutable tensors for safe parallel access, wrapping shared immutable tensors in `Arc`, constructing kernel launchers, and JIT-compiling and asynchronously executing tile kernels on the GPU.

## Your First cuTile Rust Kernel

cuTile Rust kernels are GPU programs that execute concurrently across a logical grid of tile blocks.
The `#[cutile::entry()]` attribute marks a Rust function as an *entry point*: a function you can call from your Rust program that executes on the GPU.

```rust
use cuda_async::device_operation::DeviceOperation;
use cuda_async::error::DeviceError;
use cutile::{self, api, tile_kernel::IntoDeviceOperationPartition};
use my_module::add;

#[cutile::module]
mod my_module {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 2]>(
        z: &mut Tensor<f32, S>,
        x: &Tensor<f32, { [-1, -1] }>,
        y: &Tensor<f32, { [-1, -1] }>,
    ) {
        let tile_x = load_tile_like_2d(x, z);
        let tile_y = load_tile_like_2d(y, z);
        z.store(tile_x + tile_y);
    }
}

fn main() -> Result<(), DeviceError> {
    let z = api::zeros([32, 32]).partition([4, 4]).sync()?;
    let x = api::ones([32, 32]).arc().sync()?;
    let y = api::ones([32, 32]).arc().sync()?;
    let (_z, _x, _y) = add(z, x, y).sync()?;
    Ok(())
}
```

Here, `main` is host Rust code: it runs on the CPU, allocates tensors, and launches work. The `add` function is device Rust code because it is marked with `#[cutile::entry()]`; when `main` first calls `add(z, x, y)`, cuTile Rust JIT-compiles that function into optimized GPU code. The `#[cutile::module]` macro makes `my_module` expose the generated host-side APIs for launching `add`.

---

## The Compilation Pipeline

![The cuTile Rust compilation pipeline from Rust to GPU execution](../_static/images/compilation-pipeline.svg)

---

## How Kernel Arguments Map

On the host side, immutable tensor arguments are typically passed as `Arc<Tensor<_>>`, which the generated kernel API maps to `&Tensor<...>` in device Rust code. Mutable tensor arguments are typically passed as `Partition<Tensor<_>>`, which the generated kernel API maps to `&mut Tensor<...>` for one tile-shaped region of the output.

Partitioning splits a tensor into disjoint regions with a fixed tile shape, such as `partition([4, 4])` for a 2D tensor. Each tile block receives one partition element, which is how cuTile Rust gives the kernel mutable access to one region at a time while keeping writes non-overlapping.

---

## Launching Kernels

Use `add(...)` when your arguments are already resolved:

```rust
let z = api::zeros([32, 32]).partition([4, 4]).sync()?;
let x = api::ones([32, 32]).arc().sync()?;
let y = api::ones([32, 32]).arc().sync()?;

let (_z, _x, _y) = add(z, x, y)
    .sync()
    .expect("Failed to launch add kernel.");
```

Use `add_op(...)` when each argument is still its own `DeviceOperation`:

```rust
use my_module::add_op;

let z = api::zeros([32, 32]).partition([4, 4]);
let x = api::ones([32, 32]).arc();
let y = api::ones([32, 32]).arc();

let (_z, _x, _y) = add_op(z, x, y)
    .sync()
    .expect("Failed to launch add kernel.");
```

If your arguments are already grouped into one `DeviceOperation`, use `.apply(...)` to launch `add(...)` inside that device operation context:

```rust
use cuda_async::device_operation::*;

let z = api::zeros([32, 32]).partition([4, 4]);
let x = api::ones([32, 32]).arc();
let y = api::ones([32, 32]).arc();

let args = zip!(z, x, y);
let (_z, _x, _y) = args
    .apply(|(z, x, y)| add(z, x, y))
    .sync()
    .expect("Failed to launch add kernel.");
```

Conceptually, `apply` builds a new `DeviceOperation` from the output of `args`. The closure does not execute immediately on the host. Instead, it describes how to construct the next deferred operation once `args` produces `(z, x, y)`, and the whole composed operation runs only when you call `.sync()` or `.await`.

These are the main generated kernel-launch APIs you should work with.

---

## Tensors and Tiles

Kernels move data between **Tensors** and **Tiles** using operations like `load_tile` and `store`. Both are tensor-like data structures: each has a specific **shape** (the number of elements along each axis) and a **dtype** (the data type of elements). However, there are important differences:

### Tensors (Global Memory)

**Tensors** are multi-dimensional arrays stored in **global memory (HBM)**. They are:

- **Kernel arguments** — Passed as `&mut Tensor<E, S>` for writable outputs or `&Tensor<E, S>` for read-only inputs
- **Physical storage** — Have strided memory layouts in GPU global memory
- **Limited operations** — Within kernel code, they mainly support loading and storing data as tiles; direct arithmetic is not supported
- **External data** — Candle tensors and other GPU buffers can be passed as tensors from host code to kernels via kernel arguments

```rust
// Tensor parameters in kernel signature
fn kernel(
    output: &mut Tensor<f32, S>,      // Mutable tensor (can store to)
    input: &Tensor<f32, {[-1, -1]}>   // Immutable tensor (read-only)
) { ... }
```

### Tiles

**Tiles** are **immutable** multi-dimensional array fragments that live in GPU **registers** during kernel execution. They are:

- **Immutable** — Operations create new tiles rather than modifying existing ones
- **Compiler-managed storage** — Tile data lives in registers; the compiler handles shared memory staging and other memory hierarchy details automatically
- **Compile-time static shapes** — Tile dimensions must be compile-time constants (often powers of two for optimal performance)
- **Rich operations** — Support elementwise arithmetic, matrix multiplication, reduction, shape manipulation, and more

```rust
// Tiles are created and transformed, never mutated
let tile_a = load_tile_like_2d(a, output);    // Load creates a tile
let tile_b = load_tile_like_2d(b, output);    // Another tile
let result_tile = tile_a + tile_b;            // New tile from operation
output.store(result_tile);                     // Store tile to tensor
```

### The Load → Compute → Store Pattern

Every cuTile Rust kernel follows this fundamental pattern:

![Data flow: Load from Tensor to Tile, Compute in registers, Store back to Tensor](../_static/images/data-flow.svg)

1. **Load**: Move data from global memory (Tensor) into tiles
2. **Compute**: Perform operations on tiles
3. **Store**: Write results from tiles back to global memory (Tensor)

This pattern is key to performance: global memory is slow compared to on-chip resources. By loading once, computing many operations, and storing once, we maximize the compute-to-memory ratio.

---

## When to Use cuTile Rust

**Use cuTile Rust when:**
- You need custom GPU kernels not available in libraries
- You want to fuse multiple operations for performance  
- You need Rust's safety guarantees on GPU code
- You're building performance-critical ML infrastructure

**Don't use cuTile Rust when:**
- Standard library operations (cuBLAS, cuDNN) suffice
- You need maximum portability across GPU *vendors*
- Your team is deeply invested in the CUDA C++ ecosystem

> **Note**: For algorithms requiring warp-level primitives or custom CUDA C++ kernels, cuTile Rust provides an [Interoperability](interoperability.md) that lets you integrate pre-compiled kernels into the same async execution model.

---

## Next Steps

Continue to learn about the [Tile Programming Model](tile-programming-model.md) or jump straight to the [Tutorials](../tutorials/01-hello-world.md).
