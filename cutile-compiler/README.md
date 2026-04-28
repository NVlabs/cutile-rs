# cuTile Rust Compiler

This crate compiles Rust DSL kernels into Tile IR bytecode for GPU execution
via `tileiras`. Most users interact with it indirectly through `cutile` and
`cutile-macro`.

By default, the runtime invokes `tileiras` through normal `PATH` lookup. Set
`CUTILE_TILEIRAS_PATH` to use a specific binary:

```bash
CUTILE_TILEIRAS_PATH=/opt/cuda-tile/bin/tileiras \
    cargo test -p cutile-compiler
```

The override should point to a `tileiras` binary that is compatible with the
CUDA Toolkit and driver/runtime used to load the generated cubin. Mixing a newer
standalone assembler with an older runtime is not guaranteed to work. For
example, kernels assembled with a CUDA 13.3 `tileiras` binary can fail when run
with a CUDA 13.2 driver/runtime stack, either during assembly with:

```text
tileiras failed ... error: failed to compile Tile IR program
```

or later when loading/running the generated cubin. If this happens, use the
`tileiras` from the same CUDA installation as the runtime, or remove the
override.

Cargo config files can also set this environment variable:

```toml
[env]
CUTILE_TILEIRAS_PATH = { value = "/opt/cuda-tile/bin/tileiras", relative = false }
```

When set this way, Cargo injects the variable into `cargo test` and `cargo run`
even if it is not present in the shell environment.

## Testing

```bash
cargo test -p cutile-compiler
```

## Debugging

Set `CUTILE_DUMP` to inspect the compiler's internal state after each pass.
Output goes to stderr.

```bash
# Dump the Tile IR for all kernels:
CUTILE_DUMP=ir cargo test -p cutile --test my_test -- --nocapture

# Dump multiple stages:
CUTILE_DUMP=resolved,typed,ir cargo test ...

# Dump everything:
CUTILE_DUMP=all cargo test ...
```

### Stages

| Stage | Description |
|-------|------------|
| `ast` | Raw syn AST before any passes |
| `resolved` | After name resolution (paths resolved) |
| `typed` | After type inference (types annotated) |
| `instantiated` | After monomorphization (no generics remain) |
| `ir` | cutile-ir Module, pretty-printed |
| `bytecode` / `bc` | Encoded bytecode, decoded to human-readable text |

### Filtering

Use `CUTILE_DUMP_FILTER` to limit output to specific kernels:

```bash
# By function name (matches in any module):
CUTILE_DUMP=ir CUTILE_DUMP_FILTER=my_kernel cargo test ...

# By qualified path (module::function):
CUTILE_DUMP=ir CUTILE_DUMP_FILTER=my_module::my_kernel cargo test ...

# Multiple filters (comma-separated):
CUTILE_DUMP=ir CUTILE_DUMP_FILTER=add,gemm cargo test ...
```

### Legacy

`TILE_IR_DUMP=1` is still supported as an alias for `CUTILE_DUMP=ir`.
