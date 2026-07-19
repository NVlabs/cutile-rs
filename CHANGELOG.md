# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-06-16

This release collects the changes since `0.1.0` and focuses on low-precision
inference support while also publishing the reproducibility artifacts for the
cuTile Rust paper, *Fearless Concurrency on the GPU*.

### Highlights

- Added CUDA 13.3-oriented low-precision inference support, including NVFP4
  pack/unpack support, block-scaled matrix multiply support, and runnable NVFP4
  and MXFP8 linear-tile examples.
- Added `cutile-kernels`, a reusable kernel crate organized by function for
  inference workloads. It includes attention, normalization, positional
  encoding, KV-cache update, embedding, argmax, and pointwise kernels, with
  experimental low-level and benchmark-oriented kernels for fused transformer
  paths, KVBM layout conversion, and grouped GEMM/MoE work.
- Added compile-only coverage and smoke tests for reusable kernels, and moved
  test-only examples into the test suite so `cutile-examples` stays focused on
  user-facing examples.
- Added paper reproducibility artifacts under `cutile-benchmarks/paper/`,
  including benchmark harnesses, committed result files, machine notes, and
  plotting scripts.
- Updated the root README with the paper link, citation information, paper
  artifacts, related projects, and the current cuTile Rust execution/lowering
  model.

### Paper Results

- On NVIDIA B200, cuTile Rust reaches 7 TB/s for element-wise operations and
  2 PFlop/s for GEMM, about 91% of peak memory bandwidth and 92% of dense `f16`
  peak, respectively.
- Safe Rust persistent GEMM reaches 2.07 PFlop/s at `M=N=K=8192`, within 0.3%
  of the corresponding low-level Tile IR variant, showing safety without
  measurable runtime overhead.
- Grout, a Qwen3 inference engine built with cuTile Rust in collaboration with
  Hugging Face, reaches 171 tokens/s for Qwen3-4B on NVIDIA GeForce RTX 5090
  and 82 tokens/s for Qwen3-32B on B200 in batch-1 decode, showing competitive
  state-of-the-art performance on memory-bound inference tasks as measured by
  the HBM roofline analysis.

### Changed

- Split CPU and GPU test entry points so compile-only tests do not require a
  GPU, while GPU tests actually execute GPU work.
- Updated compile-only testing to use `KernelCompiler` and default to `sm_120`
  where a local GPU architecture is not required.
- Reorganized and refreshed the book and examples around the current
  host/device API, CUDA graphs, JIT compilation, performance guidance, and
  low-precision inference, with support for versioned book builds.

## [0.1.1] - 2026-06-01

This patch release added the first CUDA 13.3 low-precision Tile IR support and
refreshed the book publishing flow.

### Added

- Added NVFP4 support in CUDA dtype handling, Tile IR formatting, bytecode
  encoding/decoding, compiler intrinsic lowering, and the public device DSL.
- Added CUDA 13.3 bytecode, per-op round-trip, and tensor/matrix operation
  tests covering the new low-precision Tile IR surface.
- Added runnable NVFP4 and MXFP8 examples, plus a new book tutorial for NVFP4
  inference.
- Added scripts and documentation for building and publishing versioned book
  releases.

### Changed

- Updated compiler lowering, specialization handling, and Tile IR type support
  for CUDA 13.3 low-precision operations.
- Updated examples, book references, and architecture notes to describe the
  current lowering path and low-precision inference APIs.
- Bumped the workspace package versions to `0.1.1`.

## [0.1.0] - 2026-05-16

This is the first cuTile Rust beta release with stable host-side and
device-side APIs. We do not plan further breaking changes to the kernel
authoring model, tensor launch API, `DeviceOp` execution model, or core device
operation surface; future work is expected to extend these APIs compatibly
unless a correctness issue requires otherwise.

### Highlights

- Stabilized the public host API around lazy `DeviceOp`s, borrowed/shared
  tensors, mutable partitions, async execution, CUDA graph capture, and CUDA
  interop.
- Stabilized the public device DSL around Tile IR-aligned operations,
  rank-polymorphic helpers such as `load_tile_like`, tensor views, partition
  views, memory ordering, atomics, tokens, shape operations, and tile math.
- Added type-check-driven JIT lowering with stable node IDs, richer expression
  type inference, static dispatch lowering, type aliases, global constants,
  `Global`, `else if`, and source-location preserving diagnostics.
- Added mapped partition support for safe persistent scheduling, including
  proof-carrying partition indexes and examples for persistent GEMM-style
  output traversal.
- Improved dynamic-shape performance by propagating `num_tiles` bounds, fixing
  nested partition overhangs, supporting static-shaped `load_tile_like`, and
  using zero-padded read-only tile-like loads where they generate better code.
- Added CUDA runtime ergonomics: dynamically loaded CUDA bindings, configurable
  `tileiras` binary override, custom memory pools, memory accounting, and JIT
  timing support.
- Updated the book, README, and examples to describe the stable host/device
  APIs and current interop story.

### Fixed

- Restored scalar divisibility hint lowering for kernel arguments.
- Fixed compile-only kernel compiler hooks and several JIT/type inference
  failures exposed by examples and downstream kernels.
- Fixed CUDA 13.0-13.2 `CUmemLocation` layout compatibility.
- Fixed custom memory pool resolution outside the default device policy
  closure.

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
