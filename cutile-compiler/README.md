# cuTile Rust Compiler

This crate compiles Rust DSL kernels into Tile IR bytecode for GPU execution
via `tileiras`. Most users interact with it indirectly through `cutile` and
`cutile-macro`.

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
