# cuTile Rust
cuTile Rust (`cutile-rs`) is a research project for writing tile-based GPU kernels in Rust.
The workspace combines:

- a safe user-facing DSL for authoring kernels
- a safe host-side API for asynchronously executing kernel functions
- a pure Rust compiler pipeline backed by the CUDA Tile compiler

# Project Status
We are excited to release this research project as a demonstration of how GPU programming can be made available in the Rust ecosystem. The software is in an early stage (`-alpha`) and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That being said, we hope you'll be interested to try it in your work and help shape its direction by providing feedback on your experience.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing to the project.

# Quick Start

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
        let tile_x = load_tile_like_2d(x, z);
        let tile_y = load_tile_like_2d(y, z);
        z.store(tile_x + tile_y);
    }
}

fn main() -> Result<(), cuda_async::error::DeviceError> {
    let x = api::ones::<f32>(&[32, 32]).sync()?;
    let y = api::ones::<f32>(&[32, 32]).sync()?;
    let mut z = api::zeros::<f32>(&[32, 32]).sync()?;

    add((&mut z).partition([4, 4]), &x, &y).sync()?;
    Ok(())
}
```

The `#[cutile::module]` macro transforms the `add` function into a GPU kernel. On the host side, `add(...)` constructs a lazy kernel launcher that accepts borrowed tensors: `(&mut z).partition([4, 4])` borrows the output and partitions it into 4×4 sub-tensors, while `&x` and `&y` borrow the inputs.

`.sync()` JIT-compiles the kernel (cached after first use) and executes it. The launch grid `(8, 8, 1)` is inferred from the partition: 32÷4 = 8 tiles per dimension.

- Run the above example via `cargo run -p cutile-examples --example add_refs`.
- More kernels and usage examples of the host-side API can be found [here](cutile-examples/examples).

# Built with cuTile Rust

- [Grout](https://github.com/huggingface/grout) — Qwen 3 inference engine in Rust by Hugging Face

# Setup

## Requirements

- **NVIDIA GPU** with compute capability `sm_80` or higher (minimum supported architecture: `sm_80`).
  - `sm_100+` is supported by CUDA 13.1+.
  - `sm_8x` support was added in CUDA 13.2.
  - `sm_90` is not yet supported; it is expected in CUDA 13.3 (release date TBD).
- **CUDA** 13.2 recommended (`sm_8x` support and `sm_100+` performance improvements over 13.1).
- **Rust** 1.89+
- **Linux** (tested on Ubuntu 24.04)

## Install

### Rust

To install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

### CUDA

Install CUDA 13.2 for your OS by following the official instructions:
https://developer.nvidia.com/cuda-downloads

## Configure Environment

Set `CUDA_TOOLKIT_PATH` to your CUDA 13.2 install directory.

Example `.cargo/config.toml`:
```toml
[env]
CUDA_TOOLKIT_PATH = { value = "/usr/local/cuda-13.2", relative = false }
```

## Verifying Installation

Run the hello world example:

```bash
cargo run -p cutile-examples --example hello_world
```

If everything works, you should see: `Hello, I am tile <0, 0, 0> in a kernel with <1, 1, 1> tiles.`

# Via Nix

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
#  ✓ CUDA  /nix/store/...-cuda-toolkit-13.2
#  ✓ Rust  1.90.0-nightly
```

The flake automatically locates host NVIDIA driver libraries on both NixOS and non-NixOS systems.

# Tests
- cuTile IR: `cargo test --package cutile-ir`
- cuTile Rust Compiler: `cargo test --package cutile-compiler`
- cuTile Rust Library: `cargo test --package cutile`
- Examples: run an individual example, for example `cargo run -p cutile-examples --example async_gemm`
- Benchmarks: `cargo bench`
- Everything: `./scripts/run_all.sh` (or pipe to a log file: `./scripts/run_all.sh 2>&1 | tee test_run.log`)

## Workspace Crates

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

# License
The `cuda-bindings` crate is licensed under NVIDIA Software License: [LICENSE-NVIDIA](LICENSE-NVIDIA).
All other crates are licensed under the Apache License, Version 2.0 https://www.apache.org/licenses/LICENSE-2.0
