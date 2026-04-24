# CUDA Async

CUDA Async lets programmers asynchronously compose DAGs of CUDA operations
and execute them on multiple devices using any async Rust runtime (such as tokio).

The design consists of three key pieces:
- **Device operations** — composed using the `DeviceOp` trait and combinators.
- **Scheduling** — an implementation of `SchedulingPolicy` maps `DeviceOp`s to streams.
- **Execution** — `.sync_on(&stream)`, `.sync()`, or `.await`.

## Device Operations

`DeviceOp<Output=T>` is a lazy, composable GPU operation. Nothing executes
until `.sync()`, `.sync_on()`, or `.await` is called.

```rust
use cutile::prelude::*;

fn main() -> Result<(), DeviceError> {
    let device = cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;

    let mut z = api::zeros::<f32>(&[16, 16]).sync_on(&stream)?;
    let x = api::ones::<f32>(&[16, 16]).sync_on(&stream)?;
    let y = api::ones::<f32>(&[16, 16]).sync_on(&stream)?;

    // Borrow-based: &mut z for output, &x and &y for inputs.
    let _ = saxpy((&mut z).partition([4, 4]), 2.0, &x).sync_on(&stream)?;
    // z already has the result.
    Ok(())
}
```

### Kernel Input Modes

Kernel `&Tensor` params accept three input forms. You get back the same
type you put in:

| Input | Returned | `tokio::spawn`? |
|---|---|---|
| `Tensor<T>` | `Tensor<T>` | Yes |
| `Arc<Tensor<T>>` | `Arc<Tensor<T>>` | Yes |
| `&Tensor<T>` | `&Tensor<T>` | No (not `'static`) |

Kernel `&mut Tensor` params accept two partition forms:

| Input | Returned |
|---|---|
| `Partition<Tensor<T>>` (owned) | `Partition<Tensor<T>>` |
| `Partition<&mut Tensor<T>>` (borrowed) | `Partition<&mut Tensor<T>>` |

The borrowed form writes in place — no `unpartition()` needed.

### Combinators

Operations compose via combinators that follow `futures` crate conventions:

```rust
// Chain dependent work on the same stream.
let result = allocate_buffer()
    .then(|buf| fill_kernel(buf))
    .then(|buf| process_kernel(buf))
    .sync()?;

// Combine independent operations.
let (a, b) = zip!(op_a, op_b).sync()?;

// Transform output without GPU work.
let doubled = op.map(|x| x * 2);

// Cloneable, execute-once.
let shared = op.shared();
```

## Scheduling

The `SchedulingPolicy` trait decides which CUDA stream each operation
runs on. The default `StreamPoolRoundRobin` rotates through 4 streams,
enabling overlap of independent operations.

```rust
// Implicit: .sync() and .await use the default round-robin policy.
let result = my_kernel(out, input).sync()?;

// Explicit: pin to a specific stream.
let result = my_kernel(out, input).sync_on(&stream)?;

// Multi-device: schedule on a specific device's policy.
let future = my_kernel(out, input).schedule(&policy)?;
```

Operations chained with `.then()` share a single stream and always
execute in order. Operations on different streams may overlap.

## CUDA Graphs

`CudaGraph<T>` captures a `DeviceOp` into a replayable CUDA graph using
[stream capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#creating-a-graph-using-stream-capture):

```rust
// Capture: executes once, records all GPU work into a graph.
let forward_op = build_forward(&cfg, &weights, input.clone(), buffers);
let mut graph = forward_op.graph_on(stream.clone())?;
let buffers = graph.take_output().unwrap();

// Replay loop — single driver call per iteration.
for token in tokens {
    graph.update(api::memcpy(&mut input_buf, &token))?;
    graph.launch().sync_on(&stream)?;
}
```

All device pointers are baked in at capture time. To vary inputs, copy
new data into pre-allocated buffers via `graph.update(op)` before each
`graph.launch()`. `launch()` returns a [`DeviceOp`] — use `.sync_on()`,
`.sync()`, or `.await` to control when and where the graph executes.

## API Argument Conventions

| Layer | Arguments | Return |
|---|---|---|
| **API functions** (`zeros`, `dup`, etc.) | Concrete values | `impl DeviceOp` |
| **Extension traits** (`.reshape()`, `.to_host_vec()`, etc.) | Concrete values | `impl DeviceOp` |
| **Kernel functions** (`rms_norm`, etc.) | `IntoDeviceOp` / `KernelInput` / `KernelOutput` args | `impl DeviceOp` |

Kernel launchers accept `Tensor<T>`, `Arc<Tensor<T>>`, `&Tensor<T>`,
`Partition<Tensor<T>>`, `Partition<&mut Tensor<T>>`, scalars, and lazy
`DeviceOp`s interchangeably via trait-based dispatch.

# Testing

Run the crate tests with:

```bash
cargo test -p cuda-async
```
