# CUDA Async
CUDA Async lets programmers asynchronously compose DAGs of CUDA operations,
and execute them on multiple devices using any async Rust runtime (such as tokio).

The design consists of three key pieces:
- Device operations, which are composed using the `DeviceOperation` API.
- Scheduling, which is done via an implementation of the `SchedulingPolicy` trait. The `schedule` operation maps instances of `DeviceOperation` to `DeviceFuture`.
- Future submission/execution, which is carried out by awaiting on the `DeviceFuture` type within an async context.

## Device Operations

The `DeviceOperation<Output=T>` trait exposes an API for composing device operations.
A given implementation of `DeviceOperation` can be converted into `DeviceFuture`, which implements `Future<Output=T>`.
The `DeviceFuture` type can either be spawned or awaited upon by any async runtime in Rust.
All functions in the `api` module construct implementations of `DeviceOperation<Output=T>`.

If you do this:
```rust
async fn main() {
    let a = 2.0;
    let x = api::ones::<f32>([16, 16]).arc(); // impl DeviceOperation
    let y = api::ones::<f32>([16, 16]).partition([4, 4]); // impl DeviceOperation
    let op = api::zip((a, x, y)).apply(my_kernels::saxpy); // impl DeviceOperation
    let (a, x, y) = op.await; // Implicitly converts the impl DeviceOperation into a DeviceFuture, and submits it for execution.
}
```
The `apply` operation applies the user-defined kernel operation `saxpy` on the arguments `(a, x, y)`.
The `arc` operation converts `x` into an `Arc<Tensor<f32>>`, and the `partition` operation converts
`y` into an `Partition<Tensor<f32>>`.
The kernel launcher passes `Arc<Tensor<f32>>` and `Partition<Tensor<f32>>` to tile kernels as `&Tensor<f32>` and partitioned sub tensors `&mut Tensor<f32>`, respectively
(see [Kernel Launch](#kernel-launch) for details on how user-defined kernel arguments are safely prepared).

`impl DeviceOperation` objects on which `await` is called are implicitly converted into futures.
This implicit conversion schedules the operation to execute on the default global device,
and `await` submits the resulting future for execution, blocking the current thread until execution is complete.

The `unzip` operation is the inverse of `zip`:
```rust
async fn main() {
    let a = 2.0;
    let x = api::ones::<f32>([16, 16]).arc();
    let y = api::ones::<f32>([16, 16]).partition([4, 4]);
    let op = zip!(a, x, y).apply(my_kernels::saxpy);
    let (a, x, y) = op.unzip(); // Unzip args after applying saxpy.
    let y = y.unpartition().arc();
    let y: Arc<Tensor<f32>> = y.await; // await just on y.
}
```

The above example discards `a` and `x` after executing `saxpy` by awaiting on just `y`.
`unpartition().arc()` can be used to convert a `Partition<Tensor<f32>>` into `Arc<Tensor<f32>>`,
subsequently allowing `y` to be used as an argument to multiple device operations in parallel.

An `Arc<Tensor<f32>>` can be converted into a `Tensor<f32>` in the usual way, which
requires the reference count of the `Arc<Tensor<f32>>` to be 1.

## Scheduling

The `DeviceFuture` struct represents a _scheduled_ device operation. A scheduled operation has resources assigned to it.
You can use the `DeviceOperation::schedule` method to schedule a device operation on a particular device:
```rust
async fn my_op<T>(z: impl DeviceOperation<Output=T>) {
    let zf: DeviceFuture<Output=T> = z.schedule(global_policy(1));
    let z: Tensor<f32> = zf.await;
}
```

The above invocation of `schedule` uses the global policy defined for device `1` to assign resources to `z`. 
Actual execution is deferred until `await` is invoked.

## Efficient Execution

Consider the following program:
```rust
async fn main() {
    let x: Tensor<f32> = api::ones::<f32>([16, 16]).await;
    let y: Tensor<f32> = api::ones::<f32>([16, 16]).await;
    let z: Tensor<f32> = api::add(x.into(), y.into()).await;
}
```

The above implementation is correct but inefficient: 
Whenever we invoke `await` on a `DeviceOperation`, we require synchronization with the async runtime.
We can instead submit a single future for execution, letting the scheduling policy order device operations
and synchronize with the async runtime once:
```rust
async fn main() {
    let xf = api::ones::<f32>([16, 16]);
    let yf = api::ones::<f32>([16, 16]);
    let z: Tensor<f32> = api::add(x, y).await;
}
```

The default scheduling policy executes the above composition of operations on the same stream, 
which executes operations in order without synchronization overhead.
The runtime is notified when the computations are complete via a CUDA host callback, 
which is placed on the stream after all operations are submitted to the stream.

A similar procedure takes place when `join` is called on a tuple of operations prior to `await`.

## Kernel Launch

Consider the following saxpy kernel:
```rust
#[cutile::module]
mod my_module {
  use cutile::core::{*};
  #[cutile::entry()]
  fn saxpy<const S: [i32; 2]>(a: f32, x: &Tensor<f32, {[-1, -1]}>, y: &mut Tensor<f32, S>) {
    let tile_a: Tile<f32, S> = broadcast_scalar(a, y.shape());
    let tile_x: Tile<f32, S> = load_tile_like_2d(x, y);
    let tile_y: Tile<f32, S> = load_tile_mut(y);
    store_tile(tile_a * tile_x + tile_y, y);
  }
}
```

The kernel expects a reference to `x`, and a mutable reference to `y`.
We provide a reference to `x` by wrapping it in an `Arc` on the host-side.
The `arc` method can be called directly on a host-side tensor to obtain `Arc<Tensor<T>>`.

To provide safe mutable access to `y`, it must be partitioned into sub-tensors on the host-side.
The `partition` method can be called on a host-side tensor to obtain an `impl Partition<Tensor<T>>`.
When an `impl Partition<Tensor<T>>` is passed to a kernel function, it provides mutable access to a disjoint sub-tensor
in the tensor partition.

Rules:
- `arc` and `partition` can be called on an `impl DeviceOperation<Output=Tensor<T>>` or directly on a `Tensor<T>`.
- If you have a `Partition<Tensor<T>>`, you can `unwrap` it into `Tensor<T>` or call `arc` on it to obtain an `Arc<Tensor<T>>`.
- If you have an `Arc<Tensor<T>>`, you can only `unwrap` it into `Tensor<T>` and partition it if there is exactly
one reference to it.

For example, to partition a `16x16` matrix into `4x4` sub-matrices and pass it to the given saxpy function, we do:
```rust
async fn main() {
    let x = api::ones::<f32>([16, 16]);
    let y = api::ones::<f32>([16, 16]);
    let (a, x, y) = saxpy(2.0, x.arc(), y.partition([4, 4])).await;
}
```

For an in-depth example, check out the data-parallel MLP example [here](../cutile-examples/examples/async_mlp.rs).

# Testing

Run the crate tests with:

```bash
cargo test -p cuda-async
```
