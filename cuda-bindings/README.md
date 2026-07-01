# cuda-bindings

Generated raw Rust FFI bindings to the CUDA toolkit libraries used by this workspace.

This crate is intentionally low level. Most code should depend on `cuda-core` instead of calling these bindings directly.

# Notes

- The bindings are generated at build time.
- `CUDA_TOOLKIT_PATH` can point at the local CUDA toolkit installation. If it
  is unset, the build searches standard CUDA 13.3/13.2 install locations.
- Set `CUTILE_SETUP_DIAGNOSTICS=1` to print CUDA toolkit discovery decisions.
