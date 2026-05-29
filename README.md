<!-- # cuTile Rust -->

<div align="center">

<img src="assets/logo.svg" alt="cuTile Rust" width="380">

[![Crates.io](https://img.shields.io/crates/v/cutile.svg)](https://crates.io/crates/cutile)
[![Build](https://img.shields.io/github/actions/workflow/status/NVlabs/cutile-rs/pr.yml?event=push&label=build)](https://github.com/NVlabs/cutile-rs/actions/workflows/pr.yml)
[![Docs](https://img.shields.io/badge/docs-book-blue.svg)](https://nvlabs.github.io/cutile-rs/)

</div>

cuTile Rust (`cutile-rs`) lets you write tile-based GPU kernels in Rust. Rust's ownership discipline is preserved across the GPU launch boundary: mutable tensors are partitioned into disjoint pieces before launch, immutable tensors are shared, and the generated launcher returns ownership when GPU work completes. Tile kernels lower to PTX through Tile IR.

## Project Status
We are excited to release this research project as a demonstration of how GPU programming can be made available in the Rust ecosystem. The software is in an early stage and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That being said, we hope you'll be interested to try it in your work and help shape its direction by providing feedback on your experience.

Please check out [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing.

## Quick Start

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

The `#[cutile::module]` macro transforms `add` into a GPU kernel and generates a host-side launcher. The host code constructs lazy tensor operations, partitions the mutable output into 128-element chunks, and calls `.sync()` to JIT-compile and execute the kernel.

The kernel signature carries the access discipline into device code: `z` is the exclusive mutable output, while `x` and `y` are shared read-only inputs. The body loads input tiles matching the output partition, adds them, and stores the result. The launch grid `(8, 1, 1)` is inferred from the partition: 1024÷128 = 8 tiles.

- Run a similar example via `cargo run -p cutile-examples --example add_basic`.
- More kernels and usage examples of the host-side API can be found [here](cutile-examples/examples).

## Related Projects

- [Grout](https://github.com/huggingface/grout): Qwen 3 inference engine in Rust by Hugging Face, built with cuTile Rust.
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide): NVlabs experimental Rust-to-CUDA compiler for writing SIMT-style GPU kernels in Rust.
- [Rust NVPTX backend](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html): rustc's target support for generating PTX for NVIDIA GPUs.

cuTile Rust targets tile-based kernels that lower through CUDA Tile IR, with APIs built around tensor partitions and tensor-core-oriented operations.

## Setup

### Requirements

- **NVIDIA GPU** with compute capability `sm_80` or higher (minimum supported architecture: `sm_80`).
  - `sm_100+` is supported by CUDA 13.1+.
  - `sm_8x` support was added in CUDA 13.2.
  - CUDA 13.3 adds `sm_90` support, so CUDA 13.3 users now have `sm_80+` coverage.
- **CUDA** 13.3 recommended (`sm_80+` support and CUDA Tile IR 13.3 features such as FP4 packing and block-scaled MMA).
- **Rust** 1.89+
- **Linux** (tested on Ubuntu 24.04)

### Install

#### Rust

To install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

#### CUDA

Install CUDA 13.3 for your OS by following the official instructions:
https://developer.nvidia.com/cuda-downloads

### Configure Environment

Set `CUDA_TOOLKIT_PATH` to your CUDA 13.3 install directory.

Example `.cargo/config.toml`:
```toml
[env]
CUDA_TOOLKIT_PATH = { value = "/usr/local/cuda-13", relative = false }
```

### Verifying Installation

Run the hello world example:

```bash
cargo run -p cutile-examples --example hello_world
```

If everything works, you should see: `Hello, I am tile <0, 0, 0> in a kernel with <1, 1, 1> tiles.`

## Via Nix

We provide a Nix flake for easy setup and development. Flakes must be enabled in your Nix configuration, if not already, add to `~/.config/nix/nix.conf`:
```
experimental-features = nix-command flakes
```

Run a command directly:
```bash
nix develop -c cargo run -p cutile-examples --example add_basic
```

Or open an interactive shell:
```bash
nix develop
# cutile-rs dev shell
#  ✓ CUDA  /nix/store/...-cuda-toolkit-13.3
#  ✓ Rust  1.90.0-nightly
```

The flake automatically locates host NVIDIA driver libraries on both NixOS and non-NixOS systems.

## Tests
- cuTile IR: `cargo test --package cutile-ir`
- cuTile Rust Compiler: `cargo test --package cutile-compiler`
- cuTile Rust Library: `cargo test --package cutile`
- Examples: run an individual example, for example `cargo run -p cutile-examples --example async_gemm`
- Benchmarks: `cargo bench`
- Everything: `./scripts/run_all.sh` (or pipe to a log file: `./scripts/run_all.sh 2>&1 | tee test_run.log`)

### Workspace Crates

```
cutile                 User-facing crate for authoring and executing tile kernels
├── cutile-macro
├── cutile-compiler
├── cuda-async
└── cuda-core

cutile-macro           cuTile Rust proc-macro
└── cutile-compiler

cutile-compiler        Compiles cuTile Rust kernels to executables
├── cutile-ir
├── cuda-async
└── cuda-core

cutile-ir              Pure Rust Tile IR builder and bytecode writer

cuda-async             Async CUDA execution via async Rust
└── cuda-core

cuda-core              Idiomatic safe CUDA API
└── cuda-bindings

cuda-bindings          NVIDIA CUDA bindings
```

## License
The `cuda-bindings` crate is licensed under NVIDIA Software License: [LICENSE-NVIDIA](LICENSE-NVIDIA).
All other crates are licensed under the Apache License, Version 2.0 https://www.apache.org/licenses/LICENSE-2.0
