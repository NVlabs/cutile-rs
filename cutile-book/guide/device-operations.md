# Orchestrating Device Operations

A `DeviceOp` is the host-side handle for GPU work: it describes a computation lazily, composes with other `DeviceOp`s on the host, and then runs when you choose how. Every host-side API that returns a future (`api::zeros`, kernel launchers, `.then()` / `.shared()` / `unzip`) produces a `DeviceOp`.

This chapter covers three things in sequence:

1. **Constructing** operations: what a `DeviceOp` is, how to compose them into lazy computation graphs before any GPU work starts.
2. **Executing** them: `.sync()` blocks the thread, `.await` runs under an async runtime, and `.graph_on(&stream)` records the operation into a CUDA graph for replay.
3. **Scheduling** them: how the default round-robin stream pool distributes work across streams, and how `.sync_on(&stream)` / `.then()` let you enforce ordering when data dependencies require it.

## Two Worlds: Host and Device

GPU programming involves two processors working together:

- **Host (CPU)** — Constructs and schedules operations, launches kernels, manages memory
- **Device (GPU)** — Executes kernels in massively parallel fashion

![Host-device execution: how kernel calls flow from the host to the GPU](../_static/images/async-host-device.svg)

This separation is fundamental: your Rust code runs on the CPU and schedules work on the GPU.

---

## Streams: Queues for GPU Work

A **stream** is a sequence of operations that execute in order on the GPU:

```rust
let ctx = CudaContext::new(0)?;      // Connect to GPU device 0
let stream = ctx.new_stream()?;       // Create a work queue
```

Key properties of streams:
- Operations on the **same stream** execute in order
- Operations on **different streams** may execute concurrently
- Synchronization points wait for stream completion

---

## DeviceOps: Lazy Computation Graphs

The core abstraction is `DeviceOp` — a lazy operation that describes GPU work without executing it.

### What's a DeviceOp?

Think of it as a recipe that hasn't been cooked yet:

```rust
let z = api::zeros(&[64, 64]);  // DeviceOp<Output=Tensor<f32>>
// Nothing happened yet! Just built a description of what to do.

let result = z.await;  // NOW it executes: allocates GPU memory, fills with zeros
```

### The Key Trait

```rust
pub trait DeviceOp: Send + Sized + IntoFuture
where Self::Output: Send {
    // ...
}
```

Every `DeviceOp` implements `IntoFuture`, which means every operation is awaitable.

---

## The Execution Flow

When you `.await` a DeviceOp, here's what happens:

```{raw} html
<style>
.seq-box {
  background: #161b22;
  border-radius: 8px;
  padding: 24px 28px;
  margin: 1em 0;
  cursor: zoom-in;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  overflow-x: auto;
}
.seq-box:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
.seq-box.zoomed {
  position: fixed; top: 50%; left: 50%;
  transform: translate(-50%, -50%) scale(1.5);
  z-index: 9999; cursor: zoom-out;
  box-shadow: 0 0 0 9999px rgba(0,0,0,0.9);
}
.seq-box pre {
  margin: 0;
  font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Roboto Mono', monospace;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.8;
  color: #8b949e;
}
.seq-box .r { color: #f97583; }
.seq-box .b { color: #79c0ff; }
.seq-box .p { color: #d2a8ff; }
.seq-box .g { color: #56d364; }
.seq-box .w { color: #c9d1d9; }
.seq-box .h { font-weight: 700; }
</style>
<div class="seq-box" onclick="this.classList.toggle('zoomed')">
<pre>
<span class="r h">Your Code</span>             <span class="b h">Tokio Runtime</span>            <span class="p h">cuTile Rust</span>            <span class="g h">GPU</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
<span class="r h">.await</span>  ---------------> <span class="b h">into_future()</span>            <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                   <span class="w">(immediate)</span>               <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span> -------------------> <span class="p h">schedule()</span>          <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                    <span class="p">DevicePolicy</span>         <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span> <------------------- <span class="p h">DeviceFuture</span>        <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                 <span class="b h">first poll()</span> ---------------> <span class="p h">execute()</span>           <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span> ----------------> <span class="g h">GPU WORK!</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>              <span class="b h">subsequent polls</span> <-- - - - - -<span class="p">|</span>- - - - - - --><span class="g">|</span> <span class="g">checking...</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
    <span class="r">|</span>                       <span class="b">|</span>                       <span class="p">|</span>                   <span class="g">|</span>
<span class="r h">Returns</span> <span class="g"><--------------</span> <span class="b h">Ready!</span> <span class="g"><------------------</span><span class="p">|</span><span class="g">------------------+</span>
</pre>
<div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid #30363d; display: flex; align-items: center; gap: 14px;">
  <span style="background: #76B900; color: white; padding: 6px 14px; border-radius: 4px; font-weight: 700; font-size: 13px;">KEY INSIGHT</span>
  <span style="color: #e6edf3; font-size: 14px; font-weight: 600;">GPU work starts at <code style="color: #d2a8ff; font-weight: 700;">execute()</code>, not at <code style="color: #f97583; font-weight: 700;">.await</code>!</span>
</div>
<div style="margin-top: 10px; color: #6e7681; font-size: 11px;">Click to zoom</div>
</div>
```

**Step-by-step:**

1. **`.await`** converts to `IntoFuture::into_future()`
2. **`into_future()`** immediately calls `DevicePolicy::schedule()` and returns a `DeviceFuture`
3. **Tokio's first poll** calls `DeviceFuture::poll()` → this triggers `execute()`
4. **`execute()`** submits work to the GPU (kernel launch, memory copy, etc.)
5. **Subsequent polls** check if GPU work is complete
6. When done, returns `Poll::Ready(result)`

---

## When Does GPU Work Actually Happen?

Consider the following snippet:

```rust
let x: Tensor<f32> = api::randn(0.0f32, 1.0f32, &[m, k]).await?;
```

This is what each method does:

| Step | Code | GPU Work? |
|------|------|-----------|
| 1 | `api::randn(...)` | ❌ Creates lazy DeviceOp |
| 2 | `.await` | ❌ Creates DeviceFuture |
| 3 | First poll | ✅ **NOW** allocates GPU memory, generates random values |
| 4 | Completion | Returns tensor |

GPU work happens during the **first poll**, not when you call `.await`!

---

## Starting with `.sync()`

The simplest way to run a kernel is `.sync()`:

```rust
let x = api::ones::<f32>(&[1024]).sync()?;
let y = api::ones::<f32>(&[1024]).sync()?;
let mut z = api::zeros::<f32>(&[1024]).sync()?;

add((&mut z).partition([128]), &x, &y).sync()?;
```

This is the right choice for scripts, debugging, and learning. Each `.sync()` call launches work on the GPU and blocks the CPU until it finishes.

### Why sync-per-op is expensive

In a multi-layer model, calling `.sync()` after every kernel creates a gap where *both* the CPU and GPU are idle — the CPU is waiting for the GPU, and the GPU has nothing queued:

```text
CPU:  [launch] [wait......] [launch] [wait......] [launch] [wait......]
GPU:           [kernel████]          [kernel████]          [kernel████]
                          ↑                      ↑
                     idle gap                idle gap
```

For inference, this overhead dominates — kernels are fast (microseconds), but sync round-trips are not. A 22-layer transformer with 6 kernels per layer means 132 sync gaps per token.

### The fix: compose first, sync once

Build the entire operation graph lazily, then execute in one shot:

```rust
// No GPU work yet — just building the graph.
let result = rms_norm(out1, hidden.clone(), weight.clone(), eps)
    .first()
    .unpartition()
    .shared();

let q = matvec(out2, result.clone(), wq.clone())
    .first()
    .unpartition()
    .shared();

// NOW execute everything on one stream, no gaps.
let output = q.sync_on(&stream)?;
```

```text
CPU:  [build graph...] [launch all]  [wait]
GPU:                    [norm████][mv████][add████]
                         no gaps — work is pipelined
```

The rest of this guide explains the tools for building these graphs:
`.then()`, `zip!`, `.shared()`, `.await`, and stream scheduling.

### Synchronous: `.sync()` / `.sync_on()`

```rust
kernel(args...).sync()?;          // default device, default stream policy
kernel(args...).sync_on(&stream)?; // explicit stream
```

### Asynchronous: `.await`

```rust
let result = kernel(args...).await?;  // non-blocking in async context
```

Or compose lazily and await the final result:

```rust
let result = step1(args)
    .then(|out| step2(out))
    .then(|out| step3(out))
    .await?;
```

### CUDA Graphs: `.graph_on(&stream)`

For repetitive workloads — the same `DeviceOp` chain run many times — record it as a CUDA graph once and replay with minimal launch overhead. This is how you get the lowest possible per-invocation cost for hot paths (e.g., inference loops where the same pipeline runs per token).

```rust
let stream = ctx.new_stream()?;
let graph = pipeline(input).graph_on(&stream)?;

for _ in 0..iterations {
    graph.launch(&stream)?;  // replay — no per-op dispatch
}
```

Graph recording captures exact kernel launches and memory operations at build time; subsequent launches skip CPU-side dispatch. See [CUDA Graph Integration in the DeviceOp API Reference](../reference/deviceop-reference.md#cuda-graph-integration) for the combinator (`.graph_on`) and scope-based (`CudaGraph::scope`) approaches, and [Tutorial 10](../tutorials/10-cuda-graphs.md) for a worked example.

---

## Building Computation Graphs

DeviceOps compose into computation graphs:

```rust
// Build lazy computation graph — no GPU work yet
let z = api::zeros(&[m, n]).partition([bm, bn]);
let x = api::randn(0.0, 1.0, &[m, k]);
let y = api::randn(0.0, 1.0, &[k, n]);

// Chain kernel invocations — output-first convention, direct calls
let result = matmul(z, x, y)
    .then(|(z, _x, _y)| activation(z))
    .then(|(z,)| normalize(z));

// Execute entire graph in one shot
let output = result.await?;
```

![Lazy computation graph showing how DeviceOps compose](../_static/images/computation-graph.svg)

**Benefits:**
- Operations can be fused
- Memory can be reused
- Scheduling can be optimized

---

## Splitting and Sharing Operations

`zip!` combines multiple `DeviceOp`s into one. But what about the reverse — taking a single operation's output and feeding it into multiple downstream branches? That's what `unzip` and `.shared()` are for.

### unzip: Fan-Out from a Tuple

`unzip` takes an operation that produces a tuple and splits it into independent operations, one per element:

```rust
// A kernel returns (output, weight, bias) as a 3-tuple.
let (output, weight, bias) = kernel(args).unzip();

// Each is now an independent DeviceOp.
let result = output.unpartition().await?;
let w = weight.await?;
```

`unzip` is the inverse of `zip!`:

```text
  zip! (fan-in)                     unzip (fan-out)

    op_a ─┐                           ┌── branch_a
           ├─ zip! ─── (a, b)    (a, b) ── unzip ──┤
    op_b ─┘                           └── branch_b
```

### The Execute-Once Guarantee

When you `unzip`, the ancestor operation that produces the tuple is **executed at most once**, regardless of how many branches consume it. Internally, `unzip` uses a shared gate (`Select`) that runs the ancestor on the first branch to execute and caches the results for the remaining branches:

```text
                                      ┌── SelectLeft ── .sync()  ─── runs ancestor,
  ancestor_op ────── Select (shared) ─┤                               caches both results
                                      └── SelectRight ── .sync() ─── finds cached result,
                                                                      no re-execution
```

This means fan-out patterns like "compute once, use in two places" are safe and efficient:

```rust
let (z, x, y) = zip!(z_op, x_op, y_op)
    .then(my_kernel)
    .unzip();

// The kernel runs once. z, x, and y each take their portion of the result.
let output = z.unpartition().to_host_vec().sync()?;
```

### .shared(): Cloneable, Execute-Once Operations

`.shared()` converts any `DeviceOp` into a `SharedDeviceOp<T>` that implements `Clone`. The underlying operation executes at most once — every clone gets `Arc::clone()` of the cached result. This follows the `FutureExt::shared()` convention from the `futures` crate.

```rust
// Create a shared operation — cloneable, execute-once.
let x = api::ones(&[32, 32]).shared();

// Pass to multiple consumers without consuming the original.
let a = kernel_a(x.clone()).sync()?;  // x executes here (once)
let b = kernel_b(x.clone()).sync()?;  // Uses the cached Arc — no re-execution
let c = kernel_c(x).sync()?;          // Also uses the cached result
```

Unlike `unzip` (which splits a fixed tuple), `.shared()` supports unlimited consumers. The result is always `Arc<T>`, so shared reads are cheap.

For pre-computed values (e.g., weight tensors already in `Arc`), use the `shared()` constructor:

```rust
let w: SharedDeviceOp<Tensor<f32>> = cuda_async::device_operation::shared(weight_arc);
```

### Common Patterns

```text
Diamond (fan-out then fan-in):

  op_a ─┐              ┌─ transform_a ─┐
         ├── zip! ── unzip              ├── zip! ── result
  op_b ─┘              └─ transform_b ─┘

Broadcast (.shared() into parallel kernels):

                     ┌── kernel_a ── result_a
  x.shared() ──────┤
                     ├── kernel_b ── result_b
                     └── kernel_c ── result_c
```

### Limitations

The execute-once mechanism relies on **sequential execution** — the normal mode for `cuda-async`, where operations are `.sync()`'d or `.await`'d one at a time from a single thread. Under this model, the shared gate is guaranteed to see only one caller at a time.

If two branches of an `unzip` were somehow executed **concurrently on different OS threads** (e.g., via `tokio::spawn` on a multi-threaded runtime), the gate is not safe — it uses a non-atomic check-then-act pattern internally. In practice, this is not triggerable because device contexts are thread-local, so scheduling an operation from a thread that hasn't initialized its device context will fail before reaching the gate. However, avoid designs that would poll both sides of an `unzip` from different threads.

---

## Sync Points and Memory Management

### When to Sync

You need synchronization when:
1. Reading results back to CPU
2. Before modifying data that's still being read
3. At computation boundaries

```rust
// Bad: No sync before reading
let z = kernel(x, y).sync_on(&stream);
let data = z.to_host_vec();  // ❌ May read incomplete data!

// Good: Sync before reading
let z = kernel(x, y).sync_on(&stream);
let data = z.to_host_vec().sync_on(&stream);  // ✅ Waits for completion
```

### Passing Tensors to Kernels

Kernel `&Tensor` params accept three input forms, and `&mut Tensor` params
accept two partition forms. You get back the same type you put in.

**Inputs (`&Tensor`):**

```rust
// Owned — single use, no Arc overhead.
let x: Tensor<f32> = ones(&[32, 32]).sync_on(&stream)?;
let (_, x) = kernel(out, x).sync_on(&stream)?;  // x is Tensor<f32>

// Shared — use the same tensor in multiple kernels.
let x: Arc<Tensor<f32>> = ones(&[32, 32]).sync_on(&stream)?.into();
let z1 = kernel1(out1, x.clone()).sync_on(&stream)?;
let z2 = kernel2(out2, x.clone()).sync_on(&stream)?;

// Borrowed — no allocation, borrow checker enforces lifetime.
let x: Tensor<f32> = ones(&[32, 32]).sync_on(&stream)?;
let _ = kernel(out, &x).sync_on(&stream)?;  // x still available
```

**Outputs (`&mut Tensor`):**

```rust
// Owned partition — must unpartition() to get the tensor back.
let z = zeros(&[32, 32]).sync_on(&stream)?.partition([4, 4]);
let (z, ..) = kernel(z, &x).sync_on(&stream)?;
let tensor = z.unpartition();

// Borrowed partition — writes in place, no unpartition() needed.
let mut z = zeros(&[32, 32]).sync_on(&stream)?;
let _ = kernel((&mut z).partition([4, 4]), &x).sync_on(&stream)?;
// z already has the result.
```

Borrowed inputs (`&Tensor<T>`) and borrowed partitions (`Partition<&mut Tensor<T>>`)
are not `'static`, so `tokio::spawn` rejects them at compile time — use `Arc`
and owned partitions for spawned tasks.

See the [DeviceOp API Reference](../reference/deviceop-reference.md#ownership-model)
for the full ownership model.

---

## Streams and Scheduling

This section explains **when GPU operations run in order** and **when they can overlap**. Understanding this is critical for both correctness and performance.

### The One Rule of CUDA Streams

A CUDA **stream** is an ordered queue of GPU work. The rule is simple:

> Operations on the **same stream** always execute in submission order.
> Operations on **different streams** may execute concurrently — the GPU is free to overlap them.

This means the stream an operation lands on determines its ordering guarantees with respect to other operations.

### Default Behavior: Round-Robin Stream Pool

When you call `.await` or `.sync()`, cutile does **not** put every operation on a single stream. Instead, it uses a **round-robin scheduling policy** that rotates through a pool of streams:

```text
                         ┌─────────────────────────────────────────┐
  Your Code              │          GPU (4-stream pool)            │
 ─────────────           │                                         │
                         │  Stream 0: ████████                     │
  op_a.await  ──────────►│  Stream 1:    ████████                  │
  op_b.await  ──────────►│  Stream 2:       ████████               │
  op_c.await  ──────────►│  Stream 3:          ████████            │
  op_d.await  ──────────►│  Stream 0:             ████████         │
  op_e.await  ──────────►│                                         │
                         └─────────────────────────────────────────┘
```

The default pool has **4 streams**. Each new operation goes to the next stream in rotation (0 → 1 → 2 → 3 → 0 → …). Because they land on different streams, **independent operations can overlap** — the GPU can work on multiple kernels or memory transfers simultaneously.

### When Operations Serialize

Even with the round-robin pool, operations **will** run in order in these cases:

**1. Same stream (wrap-around)**

Every 4th operation lands on the same stream. If `op_a` and `op_e` are both on Stream 0, `op_e` waits for `op_a` to finish:

```text
Stream 0: ████████ (op_a)         ████████ (op_e waits for op_a)
Stream 1:    ████████ (op_b)
Stream 2:       ████████ (op_c)
Stream 3:          ████████ (op_d)
```

**2. Chained with `.then()`**

Operations composed with `.then()` share a single stream, so the second operation always sees the first one's output:

```rust
let result = allocate_tensor()
    .then(|tensor| fill_with_ones(tensor))  // same stream → ordered
    .then(|tensor| run_kernel(tensor))       // same stream → ordered
    .await;
```

**3. Explicit stream with `.sync_on()`**

When you pass the same stream to multiple `.sync_on()` calls, all operations serialize on that stream:

```rust
let stream = ctx.new_stream()?;

let a = op_a.sync_on(&stream);  // Stream X: runs first
let b = op_b.sync_on(&stream);  // Stream X: waits for op_a
let c = op_c.sync_on(&stream);  // Stream X: waits for op_b
```

**4. Awaiting sequentially**

Each `.await` blocks the host until its GPU work completes (the `DeviceFuture` polls until the stream callback fires). So even though `op_a` and `op_b` may be on different streams, awaiting them one-by-one means `op_b` is not submitted until `op_a`'s result is ready on the host:

```rust
let a = op_a.await;  // Host waits for GPU to finish op_a
let b = op_b.await;  // op_b submitted after op_a is confirmed done
// These effectively serialize, even on different streams.
```

### When Operations Can Overlap

Overlap requires two things: (1) operations land on different streams, and (2) they are submitted to the GPU before waiting for each other.

**Building a lazy graph — direct kernel call:**

```rust
// The unified launcher accepts both DeviceOps and plain values.
// No need for zip! or value() wrapping.
let result = my_kernel(
    zeros(&[1024, 1024]).partition([64, 64]),
    x,
    y,
)
.first()
.unpartition()
.await?;
```

**Using `tokio::join!` for independent work:**

```rust
// Both futures are polled concurrently by the async runtime.
// They will likely land on different streams and overlap on the GPU.
let (result_a, result_b) = tokio::join!(
    kernel_a(x.clone()),
    kernel_b(y.clone()),
);
```

### Data Dependencies: Your Responsibility

The round-robin policy does **not** track data dependencies. If operation B reads the output of operation A, you must ensure A finishes before B starts. Otherwise B may read stale or partially-written data.

**Safe patterns for dependent operations:**

```rust
// Pattern 1: Chain with .then() — same stream, automatic ordering
let result = create_tensor()
    .then(|t| process(t))
    .await;

// Pattern 2: Await sequentially — host ensures ordering
let tensor = create_tensor().await;
let result = process(tensor).await;

// Pattern 3: Pin to the same stream — CUDA guarantees ordering
let stream = ctx.new_stream()?;
let tensor = create_tensor().sync_on(&stream);
let result = process(tensor).sync_on(&stream);
```

**Unsafe pattern to avoid:**

```rust
// ⚠️ DANGER: op_b may start before op_a finishes if they land on different streams!
let future_a = op_a.into_future();  // Submitted to Stream 0
let future_b = op_b_reads_a_output.into_future();  // Submitted to Stream 1
let (a, b) = tokio::join!(future_a, future_b);
// op_b might read incomplete data from op_a.
```

### Choosing the Right Execution Method

| Method               | Stream assignment           | Ordering guarantee          | Best for                           |
|----------------------|-----------------------------|-----------------------------|------------------------------------|
| `.then()`        | Shares parent's stream      | **Strict** — same stream    | Dependent operations               |
| `.sync_on(&stream)`  | Your explicit stream        | **Strict** — if same stream | Debugging, deterministic pipelines |
| `.sync()`            | Policy picks (round-robin)  | **None** between calls      | Quick scripts                      |
| `.await`             | Policy picks (round-robin)  | **None** between awaits     | Async code (see note below)        |
| `zip!` + `.then()`  | Single stream for the graph | **Strict** within the graph | Kernel launch patterns             |

:::{tip}
Sequential `.await` calls *appear* ordered from the host's perspective (each waits before the next starts), but the GPU work for each `.await` runs on whichever stream the policy assigns. For truly independent operations you want to overlap, use `zip!` or `tokio::join!`.
:::

---

## Performance Tips

### 1. Batch Operations

```rust
// Bad: Many small syncs
for i in 0..1000 {
    let result = kernel(data[i]).sync_on(&stream);
}

// Good: Build graph, sync once
let ops: Vec<_> = (0..1000).map(|i| kernel(data[i])).collect();
let results = join_all(ops).await;
```

### 2. Overlap Computation and Memory Transfers

The default round-robin policy already enables this — consecutive operations land on different streams, so a kernel on Stream 0 can overlap with a memory transfer on Stream 1:

```rust
// These naturally overlap with the default 4-stream pool:
let compute_op = heavy_kernel(input.clone());
let transfer_op = api::zeros(&[next_batch_size, dim]);

// Submit both before waiting for either:
let (result, next_buffer) = tokio::join!(compute_op, transfer_op);
```

For explicit control, create dedicated streams:

```rust
let compute_stream = ctx.new_stream()?;
let transfer_stream = ctx.new_stream()?;

let result = heavy_kernel(input).sync_on(&compute_stream);
let next_batch = load_data().sync_on(&transfer_stream); // overlaps!
```

### 3. Use Appropriate Grid Sizes

```rust
// Match grid to your data size
let num_tiles = data.len() / tile_size;
launcher.grid((num_tiles as u32, 1, 1)).sync_on(&stream);
```

---

## Summary

| Concept                   | What it is                                                  |
|---------------------------|-------------------------------------------------------------|
| **DeviceOp**       | Lazy computation description                                |
| **Stream**                | Ordered queue of GPU work                                   |
| **SchedulingPolicy**      | Decides which stream each operation uses                    |
| **Round-Robin (default)** | Rotates across 4 streams — enables overlap                  |
| **SingleStream**          | All ops on one stream — strict ordering                     |
| **sync_on()**             | Execute on an explicit stream and wait                      |
| **await**                 | Execute via the default device's scheduling policy (async)  |
| **.then()**           | Chain operations on the same stream                         |
| **zip!**                  | Combine multiple operations into one (fan-in)               |
| **unzip**                 | Split a tuple operation into independent branches (fan-out) |
| **.shared()**             | Cloneable, execute-once operation — share data across N branches |
| **.map(f)**               | Transform output without new GPU work                       |
| **.first()** / **.last()**| Extract first/last element from tuple output                |
| **.boxed()**              | Type-erase an operation for heterogeneous collections       |

**Key takeaways:**

1. The default policy distributes work across **4 streams** — consecutive operations can overlap.
2. Operations on the **same stream** are always ordered; operations on **different streams** are not.
3. Use `.then()`, sequential `.await`, or `.sync_on()` with a shared stream to enforce ordering between dependent operations.
4. Use `zip!`, `.then()`, or `tokio::join!` to enable overlap for independent operations.

---

Continue to [Tuning for Performance](performance-tuning.md) for optimization techniques. For the full `DeviceOp` API, see the [DeviceOp API Reference](../reference/deviceop-reference.md).
