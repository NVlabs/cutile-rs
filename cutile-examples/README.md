# cuTile Rust Examples

This crate contains runnable examples for the user-facing API and kernel DSL.

# Running Examples

- Start with `cargo run -p cutile-examples --example hello_world` to verify the toolchain.
- Run `cargo run -p cutile-examples --example saxpy` for a small kernel launch example.
- Run `cargo run -p cutile-examples --example nvfp4` for a CUDA 13.3 NVFP4 linear-tile example with output checking on native NVFP4 targets.
- Run `cargo run -p cutile-examples --example async_gemm` for a larger async example.
