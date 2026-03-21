# cuTile Rust Compiler

This crate contains the compiler support for cuTile Rust.
It lowers the Rust DSL into CUDA Tile MLIR and provides runtime helpers for turning that IR into executable GPU binaries.

# Typical Usage

Most users interact with this crate indirectly through `cutile` and `cutile-macro`.
If you are working on the compiler itself, the most useful test command is:

```bash
cargo test -p cutile-compiler
```
