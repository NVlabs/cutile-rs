# DeviceOp API Reference

Quick-reference for the `DeviceOp` trait and its combinators. For a
tutorial-style introduction, see [Async Execution](async-execution.md).

---

## The Futures Analogy

`DeviceOp` is to GPU work what `Future` is to async I/O. Both are lazy
descriptions of work that don't execute until driven:

| Concept | `std::future::Future` | `DeviceOp` |
|---|---|---|
| What it represents | Async computation | GPU computation |
| When it runs | On `.await` or `poll()` | On `.sync()`, `.sync_on()`, or `.await` |
| Chaining | `.then()`, `.map()` via `FutureExt` | `.then()`, `.map()` on `DeviceOp` |
| Fan-in | `join!` | `zip!` |
| Fan-out | N/A (single consumer) | `.unzip()` |
| Shared access | `FutureExt::shared()` | `.shared()` |
| Type erasure | `BoxFuture` | `.boxed()` → `BoxedDeviceOp` |
| Output wrapper | `Poll<T>` | `Result<T, DeviceError>` |

The key difference: a `Future` is pulled by an async runtime via `poll()`,
while a `DeviceOp` is pushed to the GPU via `execute()`. When you convert
a `DeviceOp` to a `Future` (via `.await` or `.into_future()`), cuTile bridges
the two models — the runtime polls a `DeviceFuture` that checks whether the
GPU has finished.

---

## Combinator Reference

All combinators follow established Rust conventions. The "Precedent" column
shows which standard library or `futures` crate method inspired the design.

### Composition

| Combinator | Signature | Precedent | What it does |
|---|---|---|---|
| `zip!(a, b, …)` | `(impl DeviceOp, …) → impl DeviceOp<Output=(A, B, …)>` | `Iterator::zip` | Combine N operations into a single tuple-producing operation |
| `.unzip()` | `impl DeviceOp<Output=(A, B, …)> → (impl DeviceOp<Output=A>, …)` | `Iterator::unzip` | Split a tuple operation into independent per-element operations |
| `.then(f)` | `self → f(Self::Output) → impl DeviceOp<Output=O>` | `FutureExt::then` | Chain follow-up GPU work **on the same stream** |
| `.map(f)` | `self → f(Self::Output) → O` (no GPU work) | `FutureExt::map` | Transform output without issuing GPU work |
| `.inspect(f)` | `self → f(&Self::Output)` (passthrough) | `FutureExt::inspect` | Peek at output for debugging; returns it unchanged |

### Selection

| Combinator | Signature | Precedent | What it does |
|---|---|---|---|
| `.first()` | `impl DeviceOp<Output=(A, B, …)> → impl DeviceOp<Output=A>` | `slice::first` | Extract the first element of a tuple output |
| `.last()` | `impl DeviceOp<Output=(A, B, …)> → impl DeviceOp<Output=Z>` | `slice::last` | Extract the last element of a tuple output |

### Sharing and Erasure

| Combinator | Signature | Precedent | What it does |
|---|---|---|---|
| `.shared()` | `self → SharedDeviceOp<Self::Output>` | `FutureExt::shared` | Cloneable, execute-once; output is `Arc<T>` |
| `shared(arc)` | `Arc<T> → SharedDeviceOp<T>` | — | Wrap an existing `Arc` as a pre-computed `SharedDeviceOp` |
| `.boxed()` | `self → BoxedDeviceOp<Self::Output>` | `FutureExt::boxed` | Type-erase for heterogeneous collections |

### Execution

| Method | Stream chosen by | Blocks? | Use case |
|---|---|---|---|
| `.sync()` | Default policy (round-robin) | Yes | Quick scripts |
| `.sync_on(&stream)` | The explicit stream | Yes | Deterministic ordering, debugging |
| `.await` | Default policy (round-robin) | No (suspends task) | Async production code |
| `.into_future()` | Default policy | No (returns `DeviceFuture`) | Manual future handling |
| `.schedule(policy)` | The policy you provide | No (returns `DeviceFuture`) | Multi-device dispatch |
| `.graph()` | Default policy (round-robin) | Yes (captures + syncs) | CUDA graph capture |
| `.graph_on(stream)` | The explicit stream | Yes (captures + syncs) | CUDA graph capture on specific stream |

:::{note}
If any kernel input is `&Tensor<T>` (borrowed), the operation is not
`'static` and cannot be used with `tokio::spawn`. Use `.sync_on()` or
`.await` in the same scope, or switch to `Arc<Tensor<T>>` for spawned tasks.
:::

---

## Supported Kernel Parameter Types

| Kernel param | Host type | Return type |
|---|---|---|
| `&Tensor<T, S>` | `Tensor<T>`, `Arc<Tensor<T>>`, or `&Tensor<T>` | Same as input |
| `&mut Tensor<T, S>` | `Partition<Tensor<T>>` or `Partition<&mut Tensor<T>>` | Same as input |
| Scalar (`f32`, `i32`, etc.) | Same scalar | Same scalar |
| `*mut T` (unsafe only) | `DevicePointer<T>` | `DevicePointer<T>` |

The borrowed partition form (`Partition<&mut Tensor<T>>`) writes in place — no
`unpartition()` needed. Create it with `(&mut tensor).partition(shape)`.

---

## Ownership Model

The core invariant: **you get back what you put in**.

### Read-only inputs (`&Tensor` params)

| Input | Returned | `tokio::spawn`? |
|---|---|---|
| `Tensor<T>` | `Tensor<T>` | Yes |
| `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | Yes |
| `&'a Tensor<T>` | `&'a Tensor<T>` | No (not `'static`) |

### Mutable outputs (`&mut Tensor` params)

| Input | Returned | `unpartition()` needed? |
|---|---|---|
| `Partition<Tensor<T>>` (owned) | `Partition<Tensor<T>>` | Yes |
| `Partition<&'a mut Tensor<T>>` (borrowed) | `Partition<&'a mut Tensor<T>>` | No — tensor is written in place |

The borrowed form is created with `(&mut tensor).partition(shape)`:

### Owned: `Tensor<T>`

Pass a tensor directly — the launcher wraps it in `Arc` internally for the
kernel, then unwraps it back afterward (safe because refcount is 1):

```rust
let output = my_kernel(
    api::zeros(&[1024]).partition([128]),
    api::ones::<f32>(&[1024]),  // DeviceOp<Output=Tensor<f32>>
)
.first()
.unpartition()
.sync_on(&stream)?;
```

Use this for single-use tensors where you don't need shared access.

### Shared: `Arc<Tensor<T>>`

Wrap in `Arc` when the same tensor is passed to multiple kernels:

```rust
let x: Arc<Tensor<f32>> = api::ones(&[1024]).sync_on(&stream)?.into();

let a = kernel_a(out_a, x.clone()).sync_on(&stream)?;
let b = kernel_b(out_b, x.clone()).sync_on(&stream)?;
```

This is the most common pattern in existing code.

### Borrowed: `&Tensor<T>`

Pass a reference when you want to retain ownership and avoid `Arc` overhead.
The borrow checker ensures the tensor outlives the kernel:

```rust
let weights: Tensor<f32> = api::ones(&[1024]).sync_on(&stream)?;

// Borrow — no Arc allocation, no refcount.
let result = my_kernel(out_partition, &weights).sync_on(&stream)?;

// weights is still available here.
```

**Key safety property**: because `&Tensor<T>` is not `'static`,
`tokio::spawn` rejects operations that borrow tensors:

```rust
let op = my_kernel(out, &weights);  // borrows weights
tokio::spawn(op);                    // ← compile error: not 'static
```

This is enforced at compile time by Rust's lifetime system — no runtime
checks needed.

### `.shared()`: Clone + Execute-Once

`.shared()` converts a `DeviceOp` into a `SharedDeviceOp<T>` that is
`Clone`. The underlying operation runs **at most once**; every clone
receives `Arc::clone()` of the cached result:

```rust
let x = api::ones::<f32>(&[32, 32]).shared();

let a = kernel_a(x.clone()).sync()?;  // x executes here (first clone to run)
let b = kernel_b(x.clone()).sync()?;  // uses cached Arc<Tensor<f32>>
```

Output type changes: `DeviceOp<Output=T>` becomes
`SharedDeviceOp` with `Output=Arc<T>`.

For pre-computed values (e.g., weight tensors), use the
`shared()` free function to wrap an `Arc<T>` directly:

```rust
use cuda_async::device_operation::shared;

let w: Arc<Tensor<f32>> = /* loaded weights */;
let w_op: SharedDeviceOp<Tensor<f32>> = shared(w);
```

### `.unwrap_arc()`

`.shared()` and `unzip` produce `Arc<T>` outputs. When you need owned `T`
back (e.g., to partition a tensor), use `.unwrap_arc()`:

```rust
let x: Arc<Tensor<f32>> = api::ones(&[1024]).shared().sync()?;

let owned: Tensor<f32> = value(x).unwrap_arc().sync()?;
let partitioned = owned.partition([128]);
```

Panics if the Arc has multiple owners.

### IntoDeviceOp: Automatic Wrapping

The `IntoDeviceOp` trait lets kernel launchers accept both `DeviceOp`s and
plain values:

| Type | Wraps as |
|---|---|
| Any `impl DeviceOp<Output=T>` | Pass-through |
| `Tensor<T>` | `Value<Tensor<T>>` |
| `Arc<T>` | `Value<Arc<T>>` |
| `&'a Tensor<T>` | `Value<&'a Tensor<T>>` |
| `&Arc<T>` | `Value<Arc<T>>` (clones the Arc) |
| `f32`, `f64`, `i32`, `i64`, `u32`, `u64`, `usize` | `Value<T>` |
| `Partition<Tensor<T>>` | `Value<Partition<Tensor<T>>>` |

```rust
// All of these work as inputs to a &Tensor kernel param:
my_kernel(out, tensor);              // Tensor<T>
my_kernel(out, arc_tensor);          // Arc<Tensor<T>>
my_kernel(out, &tensor);             // &Tensor<T>
my_kernel(out, api::ones(&[1024]));  // DeviceOp<Output=Tensor<T>>
```

---

## Scheduling Model

### How Streams Are Chosen

When you call `.sync()` or `.await`, the operation asks the **default
device's scheduling policy** for a stream. The default policy is
`StreamPoolRoundRobin` with 4 streams:

```text
op_a.sync()  →  Stream 0
op_b.sync()  →  Stream 1
op_c.sync()  →  Stream 2
op_d.sync()  →  Stream 3
op_e.sync()  →  Stream 0  (wraps around)
```

Consecutive independent operations land on different streams, enabling GPU
overlap. Operations chained with `.then()` share the parent's stream,
preserving data-dependency ordering.

### Explicit Stream: `.sync_on()`

Bypasses the policy entirely. All operations given the same stream execute
in call order:

```rust
let stream = ctx.new_stream()?;
let a = op_a.sync_on(&stream)?;  // Stream X
let b = op_b.sync_on(&stream)?;  // Stream X — guaranteed after op_a
```

### Available Policies

| Policy | Behavior |
|---|---|
| `StreamPoolRoundRobin` (default) | Rotates through N streams (default 4) |
| `SingleStream` | All operations on one stream — strict ordering |
| Custom `impl SchedulingPolicy` | Implement `fn next_stream()` for your own strategy |

### `.then()` Guarantees

`.then()` is the recommended way to express data dependencies. Both
operations share a single stream, so the second is guaranteed to see the
first's output fully written — no manual synchronization needed:

```rust
let result = allocate_buffer()
    .then(|buf| fill_kernel(buf))      // same stream
    .then(|buf| process_kernel(buf))   // same stream
    .sync()?;
```

**Non-reentrancy:** On any given thread, only one DeviceOp may be
executing at a time. Calling `sync_on`, `sync`, or `.await` inside a
`then` closure will return a runtime error. This prevents CUDA data
races from cross-stream access to in-flight tensors. If you need
nested execution and have verified there are no cross-stream data
races, use `unsafe then_unchecked`.

---

## Error Propagation

All execution methods return `Result<T, DeviceError>`. Errors propagate
through combinators: if any operation in a `.then()` chain fails, the
error short-circuits to the caller.

### DeviceError Variants

| Variant | When it occurs |
|---|---|
| `Driver(DriverError)` | CUDA driver call failed (OOM, invalid argument, etc.) |
| `Context { device_id, message }` | Device context assertion failed |
| `KernelCache(String)` | Kernel compilation or cache lookup failed |
| `Scheduling(String)` | No stream available or policy misconfigured |
| `Launch(String)` | Kernel launch precondition violated |
| `Internal(String)` | Bug in cuda-async internals |
| `Anyhow(String)` | Converted from `anyhow::Error` |

### Error Handling Patterns

```rust
// Pattern 1: Propagate with ?
let x = api::zeros(&[1024]).sync_on(&stream)?;

// Pattern 2: Match specific errors
match my_kernel(args).sync_on(&stream) {
    Ok(result) => { /* use result */ }
    Err(DeviceError::Launch(msg)) => {
        eprintln!("kernel launch failed: {msg}");
    }
    Err(e) => return Err(e.into()),
}
```

### cutile::error::Error vs DeviceError

`cutile::error::Error` is the top-level error type that wraps
`DeviceError` alongside other error categories (I/O, shape mismatches,
etc.). Functions that only do GPU work return `DeviceError`; functions
that mix host and device work (like the examples) return
`cutile::error::Error`.

---

## CUDA Graph Integration

### Combinator approach: `.graph_on(stream)`

Any `DeviceOp` can be captured into a replayable CUDA graph:

```rust
let forward_op = build_forward(&cfg, &weights, input, buffers);
let mut graph = forward_op.graph_on(stream.clone())?;
let output = graph.take_output().unwrap();

// Replay loop — no graph rebuilding, no kernel re-compilation.
for token in tokens {
    graph.update(api::memcpy(&mut input_buf, &token))?;
    graph.launch().sync_on(&stream)?;
}
```

This requires `Arc<Tensor<T>>` + `try_partition` for shared buffers.

### Scope approach: `CudaGraph::scope`

`CudaGraph::scope` provides an imperative alternative using `&mut` borrows
instead of `Arc`. Each `s.record(op)` records a graph node and releases
borrows immediately. A buffer written by one `record` call can be read
by the next:

```rust
let mut output = api::zeros::<f32>(&[d]).sync_on(&stream)?;
let weights = api::ones::<f32>(&[d]).sync_on(&stream)?;

let graph = CudaGraph::scope(&stream, |s| {
    s.record(kernel1((&mut output).partition([128]), &weights))?;
    s.record(kernel2((&mut output).partition([64]), &weights))?;
    Ok(())
})?;

graph.launch().sync_on(&stream)?;
```

`record` only accepts operations that implement `GraphNode` — kernel
launches and `memcpy`. Allocation ops (`zeros`, `ones`, `dup`) are
rejected at compile time because their addresses may change on replay.

### `GraphNode` trait

`GraphNode` is a marker trait for operations safe to record in a CUDA
graph. Only operations that do not allocate or free device memory
implement it:

| Implements `GraphNode` | Why safe |
|---|---|
| Macro-generated kernel launchers | Kernel launch only — no alloc/free |
| `Memcpy` (`api::memcpy`) | Copy between pre-allocated buffers |
| `Value<T>` (`value(x)`) | No GPU work |

### CudaGraph methods

| Method | What it does |
|---|---|
| `.graph()` / `.graph_on(stream)` | Capture a `DeviceOp` into a `CudaGraph<T>` |
| `CudaGraph::scope(&stream, \|s\| { … })` | Scoped capture with `&mut` borrows |
| `s.record(op: impl GraphNode)` | Record a graph node inside a scope |
| `graph.take_output()` | Retrieve the output from the capture execution |
| `graph.update(op)` | Run a `DeviceOp` on the graph's stream (e.g., copy new input) |
| `graph.launch()` | Returns a `DeviceOp` that replays the captured graph |

All device pointers are baked in at capture time. To vary inputs, pre-allocate
a buffer, pass it into the operation, and `memcpy` new data before each
launch. See [Tutorial 10: CUDA Graphs](../tutorials/10-cuda-graphs.md) for a
complete walkthrough.

---

## See Also

- [Async Execution](async-execution.md) — tutorial-style guide to streams, scheduling, and composition patterns
- [Tutorial 10: CUDA Graphs](../tutorials/10-cuda-graphs.md) — end-to-end CUDA graph example
- [Interoperability](interoperability.md) — integrating custom CUDA C++ kernels into the DeviceOp model
