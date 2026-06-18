<!-- # cuTile Rust -->

<div align="center">

<img src="assets/logo.svg" alt="cuTile Rust" width="380">

[![Crates.io](https://badgen.net/crates/v/cutile)](https://crates.io/crates/cutile)
[![Build](https://img.shields.io/github/actions/workflow/status/NVlabs/cutile-rs/pr.yml?event=push&label=build)](https://github.com/NVlabs/cutile-rs/actions/workflows/pr.yml)
[![Docs](https://img.shields.io/badge/docs-book-blue.svg)](https://nvlabs.github.io/cutile-rs/)

</div>

cuTile Rust (`cutile-rs`) is a tile-based system for writing memory-safe, data-race-free GPU kernels in idiomatic Rust. It extends Rust's ownership discipline across the GPU launch boundary: mutable tensors are partitioned into disjoint pieces before launch, immutable tensors are shared, and generated launchers preserve ownership while GPU work is in flight. The same model supports synchronous launches, asynchronous pipelines, and CUDA graph replay. The `#[cutile::module]` macro embeds a captured Rust AST for each kernel in the host binary; when a kernel is needed, cuTile Rust JIT-compiles that AST through CUDA Tile IR into a GPU cubin. Local opt-outs remain available when lower-level control is needed.

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

- Run a similar example via `cargo run -p cutile-examples --example saxpy`.
- More kernels and usage examples of the host-side API can be found [here](cutile-examples/examples).

## Paper

The cuTile Rust paper, *Fearless Concurrency on the GPU*, is available [here](https://arxiv.org/abs/2606.15991). On NVIDIA B200, cuTile Rust reaches 7 TB/s for element-wise operations and 2 PFlop/s for GEMM, about 91% of peak memory bandwidth and 92% of dense `f16` peak, respectively. The GEMM result is competitive with cuBLAS, and the B200 safety-overhead microbenchmarks show that cuTile Rust adds safety without measurable runtime overhead: safe Rust persistent GEMM reaches 2.07 PFlop/s at `M=N=K=8192` (92% of the B200 dense `f16` peak), within 0.3% of the corresponding low-level Tile IR variant.

The paper also evaluates Grout, a Qwen3 inference engine built with cuTile Rust in collaboration with Hugging Face. In batch-1 Qwen3 decode, Grout reaches 171 tokens/s for Qwen3-4B on NVIDIA GeForce RTX 5090 and 82 tokens/s for Qwen3-32B on B200, showing competitive state-of-the-art performance on memory-bound inference tasks as measured by our HBM roofline analysis.

Reproducibility artifacts for the paper evaluation are available [here](cutile-benchmarks/paper/). The paper-facing measurements were run against cuTile Rust 0.2.0, and the version of Grout used for the paper is available [here](https://github.com/huggingface/grout).

## Citing

If you use cuTile Rust in research, please cite the paper:

```bibtex
@misc{elibol2026fearlessconcurrencygpu,
  title = {Fearless Concurrency on the GPU},
  author = {Elibol, Melih and Roesch, Jared and Gelado, Isaac and Buehler, Eric and Garland, Michael},
  year = {2026},
  eprint = {2606.15991},
  archivePrefix = {arXiv},
  primaryClass = {cs.PL},
  url = {https://arxiv.org/abs/2606.15991}
}
```

## Related Projects and References

- [Grout](https://github.com/huggingface/grout): Qwen 3 inference engine in Rust by Hugging Face, built with cuTile Rust and useful as a reference for production kernel call sites.
- [cuTile Python](https://github.com/nvidia/cutile-python): Python kernel programming with CUDA Tile.
- [TileGym](https://github.com/NVIDIA/TileGym): CUDA Tile kernel examples and tuning patterns.
- [cuda-oxide](https://github.com/NVlabs/cuda-oxide): NVlabs experimental Rust-to-CUDA compiler for writing SIMT-style GPU kernels in Rust.
- [CUDA Tile IR documentation](https://docs.nvidia.com/cuda/tile-ir/latest/index.html): CUDA Tile IR reference documentation.
- [CUDA documentation](https://docs.nvidia.com/cuda/): CUDA toolkit documentation.
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

Set `CUDA_TOOLKIT_PATH` to your CUDA 13.3 install directory for a reproducible
setup. If it is unset, cuTile searches standard CUDA 13.3/13.2 install
locations such as `/usr/local/cuda-13.3`, `/usr/local/cuda-13.2`,
`/usr/local/cuda-13`, and `/usr/local/cuda`.

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
nix develop -c cargo run -p cutile-examples --example saxpy
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

cutile-kernels         Reusable cuTile Rust kernels
└── cutile

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
