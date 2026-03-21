# cuda-tile-rs

Rust bindings for the CUDA Tile MLIR dialect.

The bundled `cuda-tile` submodule is pinned to `v13.1.3`, which is the last known version in this repository to work with LLVM 21.

# Install

1. Install LLVM 21 with MLIR (see [https://apt.llvm.org/](https://apt.llvm.org/) for details):
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21
sudo apt-get install libmlir-21-dev mlir-21-tools
```

2. Initialize the bundled `cuda-tile` submodule:
```bash
git submodule update --init --recursive
```

3. Point `llvm-config` at LLVM 21:
```bash
sudo update-alternatives --config llvm-config
```

4. Set `CUDA_TILE_USE_LLVM_INSTALL_DIR` to that install root (for example `/usr/lib/llvm-21`).

5. Run the crate tests:
```bash
cargo test -p cuda-tile-rs
```

6. Build the example that translates a basic kernel to CUDA Tile bytecode:
```bash
cargo run -p cuda-tile-rs --example build_translate_basic
```
