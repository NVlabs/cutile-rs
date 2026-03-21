# CUDA

`cuda-core` is the thin safe wrapper layer over `cuda-bindings`.
It exposes the lower-level CUDA concepts used by the rest of the workspace without requiring most crates to touch raw FFI directly.

# Testing

Run the crate tests with:

```bash
cargo test -p cuda-core
```
