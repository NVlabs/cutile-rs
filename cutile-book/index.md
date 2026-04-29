# cuTile Rust

**cuTile Rust** is a high-performance GPU programming library that lets you write Rust code that compiles directly to CUDA kernels.

---

## Project Status
We are excited to release this experimental, research-driven prototype to explore how GPU programming can be made available in the Rust ecosystem. The software is in a pre-alpha state and under active development. It is **not production-ready** and should be used for evaluation purposes only: you should expect instability, including bugs, incomplete functionality, and breaking API changes as we work to improve it.

We encourage early experimentation and welcome feedback to help validate design decisions and guide future development. This project is intended to inform direction rather than represent a finalized or supported solution.

---

## 🚀 Get Started in 5 Minutes

```rust
use cutile::prelude::*;
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
        let tile_x = load_tile_like(x, z);
        let tile_y = load_tile_like(y, z);
        z.store(tile_x + tile_y);
    }
}

fn main() -> Result<(), cuda_async::error::DeviceError> {
    let x = api::ones::<f32>(&[1024, 1024]).sync()?;
    let y = api::ones::<f32>(&[1024, 1024]).sync()?;
    let mut z = api::zeros::<f32>(&[1024, 1024]).sync()?;

    add((&mut z).partition([64, 64]), &x, &y).sync()?;
    Ok(())
}
```

This compiles to optimized GPU code that processes thousands of elements in parallel.

---

## Why cuTile Rust?

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🦀 Rust Safety
Write GPU kernels with Rust's memory and type safety.
No more segfaults or race conditions.
:::

:::{grid-item-card} 🎯 Tile Abstraction
Kernels are single-threaded, allowing you to think in terms of data tiles rather than individual threads.
The compiler handles the complexity.
:::

:::{grid-item-card} ⚡ High Performance
Compiles to optimized bytecode.
Zero-cost abstractions achieve peak performance on popular tile-based kernels.
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
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: User Guide

guide/introduction
guide/thinking-in-tiles
guide/memory-hierarchy
guide/working-with-data
guide/writing-computations
guide/execution-model
guide/device-operations
guide/performance-tuning
guide/interoperability
guide/debugging
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

reference/dsl-api
reference/host-api
reference/glossary
```
