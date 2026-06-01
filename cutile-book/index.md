# cuTile Rust

**cuTile Rust** is a high-performance GPU programming library that lets you write Rust code that compiles directly to CUDA kernels.

---

## Project Status

We are excited to release this research project as a demonstration of how GPU programming can be made available in the Rust ecosystem. The software is in an early stage and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That being said, we hope you'll be interested to try it in your work and help shape its direction by providing feedback on your experience.

---

## Get Started

```rust
use cutile::prelude::*;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const B: i32>(
        z: &mut Tensor<f32, { [B] }>,
        x: &Tensor<f32, { [-1] }>,
        y: &Tensor<f32, { [-1] }>,
    ) {
        let tx = load_tile_like(x, z);
        let ty = load_tile_like(y, z);
        z.store(tx + ty);
    }
}

fn main() -> Result<(), Error> {
    let x = api::ones::<f32>(&[1024]);
    let y = api::ones::<f32>(&[1024]);
    let z = api::zeros::<f32>(&[1024]).partition([128]);

    let (_z, _x, _y) = kernel::add(z, x, y).sync()?;
    Ok(())
}
```

The example separates host-side tensor setup from the device-side tile program. The host constructs tensors, partitions the mutable output into 128-element chunks, and launches the generated operation with `.sync()`.

The kernel signature carries the access discipline into device code: `z` is the exclusive mutable output, while `x` and `y` are shared read-only inputs. The body loads input tiles matching the output partition, adds them, and stores the result.

---

## What it gives you

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Ownership at the launch boundary
Mutable tensors are partitioned into disjoint pieces before launch; immutable tensors are shared by `Arc`. The borrow checker covers GPU kernel arguments, not just host-side code.
:::

:::{grid-item-card} Tile programs, not threads
Tile kernels are written as single-threaded programs over tiles of data. The compiler maps tiles onto warps, blocks, and Tensor Cores; you don't manage shared memory or thread indices directly.
:::

:::{grid-item-card} Tile IR for performance
Tile kernels lower through CUDA Tile IR, NVIDIA's tile-level compiler IR, to GPU cubins. On B200, the safe API reaches 2.07 PFlop/s on persistent GEMM (96.4% of cuBLAS); the safe mapped kernel matches the raw-pointer Rust baseline within measurement noise.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials

tutorials/01-hello-world
tutorials/02-vector-addition
tutorials/03-saxpy
tutorials/04-matrix-multiplication
tutorials/05-fused-softmax
tutorials/06-flash-attention
tutorials/07-intro-to-async
tutorials/08-data-parallel-mlp
tutorials/09-pointer-addition
tutorials/10-cuda-graphs
tutorials/11-nvfp4-inference
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User Guide

guide/introduction
guide/host-vs-device
guide/tensors-and-tiles
guide/jit-compilation
guide/device-operations
guide/performance
guide/interoperability
guide/debugging-and-profiling
guide/useful-mental-models
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/dsl-api
reference/host-api
reference/glossary
```
