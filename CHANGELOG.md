# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **DeviceOp combinators**: `.then()`, `.map()`, `.inspect()`, `.first()`, `.last()`, `.shared()`, `.boxed()`, `zip!`/`.unzip()`, following `futures` crate conventions.
- **Unified kernel launcher**: Single function per kernel via `IntoDeviceOp`. Accepts `Tensor<T>`, `Arc<Tensor<T>>`, `&Tensor<T>`, `DeviceOp`s, and scalars interchangeably.
- **KernelInput trait**: `&Tensor` params accept `Tensor<T>`, `Arc<Tensor<T>>`, or `&Tensor<T>`. Same type in, same type out. `&Tensor<T>` inputs prevent `tokio::spawn` at compile time via Rust's lifetime system.
- **KernelOutput trait**: `&mut Tensor` params accept `Partition<Tensor<T>>` or `Partition<&mut Tensor<T>>`. Borrowed partitions write in place: No `unpartition()` needed.
- **CUDA graph capture**: `.graph()` / `.graph_on(stream)` captures any `DeviceOp` into a replayable `CudaGraph<T>`. `graph.update()` + `graph.launch()` for efficient replay.
- **`CudaGraph::scope`**: Scoped graph capture with `&mut` borrows. `s.record(op)` records `GraphNode` ops as graph nodes, releasing borrows between calls.
- **`GraphNode` trait**: Marker trait for operations safe to record in a CUDA graph (kernel launches, `memcpy`). Allocation ops are excluded at compile time.
- **Thread-local execution lock**: Enforces "only one DeviceOp may be executing at a time per thread." Prevents cross-stream data races from nested execution (e.g., `sync_on` inside a `then` closure). `unsafe then_unchecked` opts out.
- **TensorView**: Safe borrowed reshaped view via `tensor.view(&[usize])`. Replaces `unsafe fn view()`.
- **SharedDeviceOp**: Cloneable, execute-once operations via `.shared()`.
- **DeviceOpVec**: Heterogeneous collections of boxed operations.
- **Scheduling**: Simplified `SchedulingPolicy` trait with single `fn next_stream()`. `StreamPoolRoundRobin` (default, 4 streams) and `SingleStream`.
- **Prelude**: `use cutile::prelude::*` for common imports.
- **New examples**: `cuda_graphs.rs` (scope-based), `cuda_graphs_deviceop.rs` (combinator-based), `add_refs.rs` (borrow pattern).
- **cutile-book**: DeviceOp API reference, CUDA Graphs tutorial, tutorials 6-9 in book_examples.rs.

### Changed
- **`DeviceOperation` renamed to `DeviceOp`**: Shorter, consistent with `DeviceOp` combinators.
- **`zeros`/`ones`/`full` take `&[usize]`**: Dynamic shape, no const-generic rank parameter.
- **`randn`/`rand` are generic**: `randn::<T>(mean, std, shape, seed)` via `RandNormal`/`RandUniform` traits. Per-type functions (`randn_f32`, etc.) removed.
- **`copy` renamed to `dup`**: Takes `&Tensor<T>` not `&Arc<Tensor<T>>`.
- **Copy API consolidated**: `copy_from`/`copy_into`/`copy_data_into` replaced by `memcpy(&mut dst, &src)` + `dup(&tensor)`.
- **Reshape unification**: `reshape(&[usize])` returns `Result`. Old variants (`reshape_dyn`, `try_view`, `view_dyn`, `flatten_view`, etc.) removed.
- **Tensor fields**: `shape`/`strides` now `pub(crate)` with `shape()`/`strides()` accessors.
- **Partition uses `[usize]`**: Partition shapes are `[usize; N]`, not `[i32; N]`.
- **Output-first convention**: `&mut Tensor` is always the first kernel parameter.
- **Extension trait renames**: `IntoDeviceOperationPartition` → `PartitionOp`, `TensorDeviceOpToHostVec` → `ToHostVecOp`.
- **`.graph()` / `.graph_on(stream)`**: Follows `sync` / `sync_on` convention.

### Fixed
- **`sync_on` error propagation**: `stream.synchronize()` errors are now returned instead of panicking.
- **Concurrent stream capture**: Removed `cuCtxSynchronize` call in `CudaContext::new_stream` that caused `DriverError(900)` when creating streams while another thread was capturing a CUDA graph.

### Removed
- `DeviceOperationArc`, `UnwrapArc` struct, `GlobalSchedulingPolicy` enum, `WithDeviceId` trait.
- `_op()` generated launcher variant (unified launcher replaces it).
- `num_mb()`, `num_gb()`, `copy_sync()` on Tensor.
- `DeviceOperationReshape`, `DeviceOperationDynamicReshape`, `DeviceOperationCopyFrom` traits.
- cudarc event-tracking infrastructure (`num_streams`, `event_tracking` on `CudaContext`). Unused; caused interference with concurrent captures.

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
