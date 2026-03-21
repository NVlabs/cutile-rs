# cuda-bindings

Generated raw Rust FFI bindings to the CUDA toolkit libraries used by this workspace.

This crate is intentionally low level. Most code should depend on `cuda-core` instead of calling these bindings directly.

# Notes

- The bindings are generated at build time.
- `CUDA_TOOLKIT_PATH` must point at the local CUDA toolkit installation.
- The crate is licensed under [LICENSE-NVIDIA](../LICENSE-NVIDIA).
