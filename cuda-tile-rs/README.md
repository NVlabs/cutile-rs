# cuda-tile-rs

The `cuda-tile` submodule version is set to `v13.1.3`.
This is the last known version to work with LLVM-21.

Rust bindings for the CUDA Tile MLIR dialect.

# Install

1. Get LLVM-21 with MLIR (see [https://apt.llvm.org/](https://apt.llvm.org/) for details.):
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21
sudo apt-get install libmlir-21-dev mlir-21-tools
```

2. Get cuda-tile: `git submodule update --init --recursive`
3. Point to LLVM-21: `sudo update-alternatives --config llvm-config`
4. Run tests: `cargo test -p cuda-tile-rs`
5. Build a basic kernel and translate to tile ir bytecode: `cargo run -p cuda-tile-rs --example build_translate_basic`
