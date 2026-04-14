# Plan: tile-ir as the sole compiler

## Goal

Replace the melior/MLIR-based compiler with the tile-ir compiler as the only
compilation path. Remove the old compiler, melior, and all MLIR C++ dependencies
from the build. This eliminates the LLVM/MLIR install requirement and enables
offline compilation as a structural property, not a feature flag.

## Current state (2026-04-08)

The tile-ir branch has two parallel compilers:

- **compiler/** (old) — Rust AST → melior MLIR ops → C++ bytecode writer → `.bc`
- **compiler2/** (new) — Rust AST → tile-ir ops → pure-Rust bytecode writer → `.bc`

compiler2 wraps the old compiler's `CUDATileFunctionCompiler` for type
compilation, generic resolution, and AST parsing. It then emits tile-ir ops
instead of melior ops. The old compiler is still linked because compiler2
depends on its type system.

Both paths produce bytecode consumed by `tileiras` (bytecode → cubin).

### What works

All examples pass with `--features tile-ir`: GEMM, flash_attention,
flash_attention_causal, hello_world, add_ptr, scale_ptr, and 16 others.
145 tests pass (golden comparison, bytecode compat, dominance, per-op roundtrip).

### What remains coupled

compiler2 calls `self.old.compile_type()` for every type it encounters. This
is the sole reason melior is still linked when `--features tile-ir` is active.
The type system itself mostly manipulates `syn::Type` → `TileRustType` →
MLIR type strings. It does not construct melior operations.

## Architecture target

```
                    pure Rust, no GPU, no LLVM
                   ┌─────────────────────────────┐
  Rust AST ──────▶│  compiler2 (type system +    │──▶ .bc bytecode
  (syn parse)     │  tile-ir op emission)        │
                   └─────────────────────────────┘
                                │
                                ▼
                   ┌─────────────────────────────┐
                   │  tileiras (external binary)  │──▶ .cubin
                   └─────────────────────────────┘
                                │
                                ▼
                   ┌─────────────────────────────┐
                   │  CUDA driver (load + launch) │
                   └─────────────────────────────┘
```

Three independent layers:

1. **Compile** (pure Rust): Rust AST → `.bc` bytecode. No MLIR, no LLVM, no GPU.
2. **Assemble** (tileiras): `.bc` → `.cubin`. Needs tileiras binary, no GPU.
3. **Execute** (CUDA): `.cubin` → kernel launch. Needs GPU.

## Migration phases

### Phase 1: Feature parity

All examples and benchmarks pass. The old compiler remains as the reference
implementation for validating bytecode output.

**Status: complete.**

### Phase 2: Extract the type system + make compiler2 self-sufficient

compiler2 now has its own MLIR-free type system and compilation pipeline:
- `tile_rust_type.rs`: `TileRustType` with `cuda_tile_ty_str: Option<String>`
  instead of `melior::ir::Type<'c>` — no lifetime parameter
- `compile_type.rs`: ported type compilation (stores type strings, no melior parse)
- `modules.rs`: `CUDATileModules` moved to compiler2 (canonical location)
- `shared_types.rs`, `shared_utils.rs`: all pure-Rust utilities ported
- `optimization_hints.rs`: tile-ir attribute builders for load/store hints
- `Compiler2` holds data directly, no longer wraps `CUDATileFunctionCompiler`
- Shared types (`TileBinaryOp`, `OptimizationHints`) live at crate level
  (`bounds.rs`, `hints.rs`), both compilers re-export

Zero `use melior` or `use crate::compiler::` imports remain in compiler2/.

**Status: complete.**

### Phase 3: Make tile-ir the default

- `tile-ir` feature flag removed entirely — tile-ir is always a dependency
- `compiler2` module is always compiled (no `#[cfg]` gate)
- `--features mlir` opts into old compiler path for validation
- `#[cfg(not(feature = "mlir"))]` for compiler2, `#[cfg(feature = "mlir")]` for old

**Status: complete.**

### Phase 4: Remove the old compiler

- Gate `compiler/` behind `#[cfg(feature = "mlir")]` to save compile time
- Make melior, mlir-sys, cuda-tile-rs optional dependencies behind `mlir` feature
- Eventually delete `compiler/` entirely
- compiler2 becomes just `compiler`

**Status: not started.**

### Phase 5: Remove melior entirely

- Remove melior, mlir-sys from Cargo.toml
- Build no longer requires LLVM/MLIR installed
- Blocked on Phase 4

**Status: not started.**

## Related efforts

### Issue #72 — Offline compilation

The user request that motivated this plan. Once Phase 5 is complete, offline
compilation is the default behavior: no LLVM, no MLIR, no GPU needed to produce
bytecode. tileiras is the only external tool required (ships with CUDA toolkit).

### PR #30 — Compile without GPU

Feature-flags GPU dependencies (cuda-bindings, cuda-core). This is orthogonal
to the MLIR removal and remains useful: even after Phase 5, you still need to
separate "compile to bytecode" from "assemble to cubin" from "execute on GPU".

PR #30 should ensure its feature gating works with `--features tile-ir` and
does not break the tile-ir compilation path.

### PR #81 — JIT warmup + persistent cache

Adds `JitStore` trait for disk-cached cubins, `compile_warmup` and
`execute_warmup` APIs. This is complementary:

- Cache key should include the bytecode version (v13.2 for tile-ir, v13.1 for
  old compiler) to avoid cross-path cache collisions during the transition.
- The `JitStore` trait works regardless of which compilation path produced the
  cubin — it caches the Layer 2 output.
- After Phase 4, the cache key simplifies (only one compiler version).

## Key reference implementations

When debugging bytecode encoding issues, cross-check against:

- **Python**: `cutile-python/src/cuda/tile/_bytecode/encodings.py` (per-op format)
- **cutile-bytecode**: `../cutile-bytecode/` (standalone Rust encoder/decoder,
  use as format spec reference only — separately licensed)
- **C++ generated**: `target/release/build/cuda-tile-rs-*/out/build/lib/Bytecode/Writer/Bytecode.inc`
