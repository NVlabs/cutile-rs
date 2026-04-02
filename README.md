# cuTile Rust
cuTile Rust (`cutile-rs`) is a research project for writing tile-based GPU kernels in Rust.
The workspace combines:

- a safe user-facing DSL for authoring kernels
- a safe host-side API for asynchronously executing kernel functions
- an MLIR-based compiler pipeline backed by the CUDA Tile compiler

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
├── cuda-tile-rs
├── cuda-async
└── cuda-core

cuda-async             Async CUDA execution via async Rust
└── cuda-core

cuda-core              Idiomatic safe CUDA API
└── cuda-bindings

cuda-tile-rs           CUDA Tile MLIR dialect builder API

cuda-bindings          NVIDIA CUDA bindings
```

# Project Status
We are excited to release this research project as a demonstration of how GPU programming can be made available in the Rust ecosystem. The software is in an early stage (`-alpha`) and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That being said, we hope you'll be interested to try it in your work and help shape its direction by providing feedback on your experience.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing to the project.

# Setup

## Requirements

- **NVIDIA GPU** with compute capability `sm_80` or higher (minimum supported architecture: `sm_80`).
  - `sm_100+` is supported by CUDA 13.1+.
  - `sm_8x` support was added in CUDA 13.2.
  - `sm_90` is not yet supported; it is expected in CUDA 13.3 (release date TBD).
- **CUDA** 13.2 recommended (`sm_8x` support and `sm_100+` performance improvements over 13.1).
- **LLVM** 21 with MLIR.
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

### LLVM

To install LLVM-21 with MLIR (see [https://apt.llvm.org/](https://apt.llvm.org/) for details):
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21
sudo apt-get install libmlir-21-dev mlir-21-tools
```

## Configure Environment

- Set `CUDA_TOOLKIT_PATH` to your CUDA 13.2 install directory.
- Ensure `llvm-config` points to LLVM 21. This is required by `melior`.
- Set `CUDA_TILE_USE_LLVM_INSTALL_DIR` to your LLVM 21 install directory (for example `/usr/lib/llvm-21`). This is required by `cuda-tile-rs`.
- Initialize the CUDA Tile submodule before building:

```bash
git submodule update --init --recursive
```

The environment needs access to `llvm-config` in order to resolve llvm (and mlir)-related dependencies.
You can configure multiple llvm builds using `update-alternatives`:
```bash
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/lib/llvm-21/bin/llvm-config 1
sudo update-alternatives --config llvm-config
```

Example `.cargo/config.toml`:
```toml
[env]
CUDA_TOOLKIT_PATH = { value = "/usr/local/cuda-13.2", relative = false }
CUDA_TILE_USE_LLVM_INSTALL_DIR = { value = "/usr/lib/llvm-21", relative = false }

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "link-arg=-Wl,-rpath,/usr/lib/llvm-21/lib"]
```

The `rustflags` entry embeds an rpath into compiled binaries so that MLIR shared libraries (e.g. `libMLIR.so`) can be found at runtime without needing to set `LD_LIBRARY_PATH`.

### Building cuda-tile-rs

This workspace depends on the `cuda-tile` submodule and the `cuda-tile-rs` crate. See [cuda-tile-rs/README.md](cuda-tile-rs/README.md) for the crate-specific setup and testing steps.

## Verifying Installation

Run the hello world example:

```bash
cargo run -p cutile-examples --example hello_world
```

If everything works, you should see: `Hello, I am tile <0, 0, 0> in a kernel with <1, 1, 1> tiles.`

# Quick Start

```rust
use cuda_async::device_operation::DeviceOperation;
use cutile::{self, api, tile_kernel::IntoDeviceOperationPartition};
use my_module::add_op;

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

fn main() -> Result<(), cutile::tile_kernel::DeviceError> {
    let x = api::ones([32, 32]).arc();
    let y = api::ones([32, 32]).arc();
    let z = api::zeros([32, 32]).partition([4, 4]);
    let (_z, _x, _y) = add_op(z, x, y).sync()?;
    Ok(())
}
```

The above example defines a _device-side_ module named `my_module`, which contains the _tile kernel_ `add`. 
The `add` kernel is marked as an _entry point_, allowing it to be executed from the _host-side_ (e.g. the `main` function).
Our kernel is defined such that `x` and `y` are input tensors, and `z` is an output tensor.

On the host-side, we allocate our device-side tensors `x`, `y` and `z`.
The kernel indicates that `z` must be mutable. Since the same tile kernel executes in parallel by many tile threads, we will need a way to provide each tile thread exclusive access to `z`. It is enough to wrap `x` and `y` in an Arc (see [cuda-async](cuda-async) for details), however, the tensor `z` is partitioned into a grid of `4x4` sub-tensors. In cuTile Rust,
any `&mut Tensor<...>` requires the host to pass a `Partition<Tensor<T>>` as the argument. Any `&Tensor<...>` requires the
host to pass an `Arc<Tensor<...>>` as an argument.

The expression `add_op(z, x, y)` constructs a representation of a _kernel launcher_: A structure which encodes how the GPU applies the kernel to the given arguments. By default, because we have partitioned `z` into a grid of `4x4` subtensors, the kernel launcher will pick a _launch grid_ of `(8, 8, 1)`. Each `(x, y, z)` coordinate in the launch grid corresponds to a _tile thread_.

The `sync` method picks the default device on the system and synchronously JIT-compiles the kernel to the default device's architecture and immediately executes the kernel with the provided arguments.
Before executing the user-defined kernel on the device-side, each tile thread is initialized by selecting 
a distinct sub-tensor from the partitioning of `z` as the `&mut Tensor<...>` kernel parameter.
Each tile thread has exclusive access to a distinct sub-tensor within the partition of `z`,
allowing for safe parallel mutable access.

- Run the above example via `cargo run -p cutile-examples --example add_basic`.
- More kernels and usage examples of the host-side API can be found [here](cutile-examples/examples).

# Tests
- CUDA Tile dialect bindings: `cargo test --package cuda-tile-rs`
- cuTile Rust Compiler: `cargo test --package cutile-compiler`
- cuTile Rust Library: `cargo test --package cutile`
- Examples: run an individual example, for example `cargo run -p cutile-examples --example async_gemm`
- Benchmarks: `cargo bench`
- Everything: `./scripts/run_all.sh` (or pipe to a log file: `./scripts/run_all.sh 2>&1 | tee test_run.log`)

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
#  ✓ LLVM  21.1.8
#  ✓ Rust  1.90.0-nightly
```

The flake automatically locates host NVIDIA driver libraries on both NixOS and non-NixOS systems.

# Built with cuTile Rust

- [Grout](https://github.com/huggingface/grout) — Qwen 3 inference engine in Rust by Hugging Face

# License
The `cuda-bindings` crate is licensed under NVIDIA Software License: [LICENSE-NVIDIA](LICENSE-NVIDIA).
All other crates are licensed under the Apache License, Version 2.0 https://www.apache.org/licenses/LICENSE-2.0
