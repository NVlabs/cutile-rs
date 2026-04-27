# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.0.2] - 2026-04-26

This release is a broad API and compiler update focused on making kernel
launching composable, removing the JIT's dependency on external MLIR tooling,
and aligning the Rust DSL with the Tile IR operation model.

### Added

- `DeviceOp` combinators, shared/boxed operations, heterogeneous operation
  collections, and a unified launcher API for kernels.
- CUDA graph capture APIs, including scoped graph capture and graph launches
  that compose as `DeviceOp`s.
- Safe tensor views and slicing, plus host helpers such as `linspace`, `eye`,
  and generic random tensor creation.
- `cutile-ir`, a pure Rust Tile IR representation, formatter, bytecode writer,
  decoder, validation tests, and round-trip coverage.
- JIT compiler infrastructure for name resolution, stable node IDs, typed
  dispatch lowering, type inference groundwork, specialization hints, and
  linker-based module discovery.
- Type-safe Tile IR op modifiers for rounding, overflow, memory ordering,
  scope, padding, TMA, FTZ, NaN propagation, comparison predicates, and related
  static attributes.
- `cuda-tile-rs` as an opt-in wrapper around the bundled cuda-tile C++ library
  and `cuda-tile-translate`.
- New examples, benchmarks, and book/reference material for DeviceOps, CUDA
  graphs, interop, tensor slicing, and the updated DSL.

### Changed

- Renamed `DeviceOperation` to `DeviceOp` and simplified scheduling around a
  smaller `SchedulingPolicy` API.
- Renamed CUDA wrapper types from `CudaContext`/`CudaStream`/`CudaModule`/
  `CudaFunction` to `Device`/`Stream`/`Module`/`Function`, with borrowed raw
  handle constructors for interop.
- Consolidated tensor copy, reshape, view, random, and creation APIs around
  dynamic shapes and clearer ownership/borrowing behavior.
- Updated kernel parameter handling so tensors, borrowed tensors, mutable
  outputs, partitions, scalars, and `DeviceOp` inputs can be mixed more
  naturally.
- Reworked rank-polymorphic macro expansion through shadow dispatch and rank
  instantiation instead of the old variadic registry machinery.
- Aligned `_core.rs` with the Tile IR operation groups and expanded named DSL
  coverage for numeric, conversion, comparison, memory, atomic, view, token,
  shape, matrix, and misc operations.
- Collapsed `load_tile_like_*` helpers into a single `load_tile_like`, and
  reduced partition view construction to `make_partition_view` and
  `make_partition_view_mut`.

### Fixed

- Corrected `arange` behavior across multiple tile blocks.
- Propagated stream synchronization errors instead of panicking.
- Fixed concurrent CUDA graph capture failures caused by unnecessary context
  synchronization.
- Fixed bytecode defaults and silent-drop cases in the JIT/compiler path.
- Restored nested marker type path resolution for static op modifiers.
- Improved compiler, macro, and DSL error messages and source locations.

### Removed

- The external LLVM/MLIR dependency from the default JIT compiler path.
- The generated `_op()` launcher variant; the unified launcher is now the
  public entry point.
- Legacy `DeviceOperation*` aliases, old copy/reshape/view helper traits, and
  unused cudarc event-tracking infrastructure.

## [0.0.1] - 2026-04-07

Initial tagged release. Pre-DeviceOp redesign baseline.

### Features
- Tile-based GPU programming model with `#[cutile::entry()]` kernels.
- `DeviceOperation` trait with `.apply()`, `.and_then()`, `zip!`, `.unzip()`.
- JIT compilation pipeline: Rust AST → MLIR → CUDA PTX.
- Async execution via tokio with `DeviceFuture`.
- `Arc<Tensor<T>>` for shared inputs, `Partition<Tensor<T>>` for mutable outputs.
- Flash attention, GEMM, RMSNorm, softmax examples.
- cuTile Rust Book with tutorials 1-9.
