# Tutorial 8: Data Parallel MLP

> Note: While async concepts are taught using the `tokio` runtime, any async runtime can be used.

In this tutorial we show how to build a single-layer MLP, copy it to multiple GPUs, and execute distinct batches of data on each instance:

```text
Input → Linear → ReLU → Output

Where:
  Linear: hidden = input @ weights
  ReLU:   output = max(0, hidden)
```

---

## The Code

```rust
#[cutile::module]
mod data_parallel_module {

    use cutile::core::*;

    // mat-mat.
    #[cutile::entry()]
    fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM, BN] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = load_tile_mut(z);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i, pid.1]);
            tile_z = mma(tile_x, tile_y, tile_z);
        }
        z.store(tile_z);
    }

    // mat-vec.
    #[cutile::entry()]
    pub fn matvec<const BM: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f32, { [BM] }>,
        x: &Tensor<f32, { [-1, K] }>,
        y: &Tensor<f32, { [K] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z = z.load().reshape(const_shape![BM, 1]);
        for i in 0i32..(K / BK) {
            let tile_x = part_x.load([pid.0, i]);
            let tile_y = part_y.load([i]).reshape(const_shape![BK, 1]);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        z.store(tile_z.reshape(const_shape![BM]));
    }

    // ReLU.
    #[cutile::entry()]
    fn relu<const D: i32>(input_output: &mut Tensor<f32, { [D] }>) {
        let zero_tile: Tile<f32, { [D] }> = constant(0.0f32, const_shape![D]);
        let input = input_output.load();
        input_output.store(max_tile(zero_tile, input));
    }
}

use data_parallel_module::{gemm_apply, relu_apply, matvec_apply};

#[tokio::main]
async fn main() {

    use cuda_async::device_operation::*;
    use data_parallel_module::{gemm_apply, relu_apply, matvec_apply};
    use cutile::api;
    use cutile::tensor::{Unpartition, Partition, Tensor, ToHostVec};
    use cutile::tile_kernel::{IntoDeviceOperationPartition, TileKernel};
    use cuda_async::device_context::global_policy;
    use cutile::api::copy;
    use tokio::task::JoinHandle;

    // Get device scheduling policies.
    let num_devices = 2;
    let devices = {
        let mut r = vec![];
        for _ in 0..num_devices {
            // Pretend we have multiple devices...
            // If you actually do have multiple devices, use i in place of 0.
            r.push(global_policy(0)?);
        }
        r
    };

    let dim = 16;
    let block_dim = 4;
    let fully_connected_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let output_layer = [
        block_dim.to_string(),
        block_dim.to_string(),
        dim.to_string(),
    ];
    let w0 = api::randn(0.0f32, 1.0, [dim, dim]); // impl DeviceOperation
    let w1 = api::randn(0.0f32, 1.0, [dim]); // impl DeviceOperation
    let w = zip!(w0.arc(), w1.arc()).schedule(&devices[0])?.await?;
    let mut joins = vec![];
    for i in 1..num_devices {
        let w_copy = tokio::spawn(zip!(copy(&w.0).arc(), copy(&w.1).arc()).schedule(&devices[i])?);
        joins.push(w_copy);
    }
    let mut model_weights = vec![w];
    for join in joins {
        model_weights.push(join.await.unwrap()?);
    }

    // Asynchronously compute forward pass for each batch of data on each device.
    let mut futures: Vec<JoinHandle<Result<Partition<Tensor<f32>>, cuda_async::error::DeviceError>>> = vec![];
    for i in 0..num_devices {
        let w = &model_weights[i];
        let (w0, w1) = (w.0.clone(), w.1.clone());
        let data = api::randn(0.0, 1.0, [dim, dim]).arc();
        let out0 = api::zeros::<2, f32>([dim, dim]).partition([block_dim, block_dim]);
        let (out0, _, _) = zip!(out0, data, value(w0))
            .apply(|args| gemm_apply(args).generics(fully_connected_layer.to_vec()))
            .unzip();
        let out1 = api::zeros::<1, f32>([dim]).partition([block_dim]);
        let (out1, _, _) = zip!(out1, out0.unpartition().arc(), value(w1))
            .apply(|args| matvec_apply(args).generics(output_layer.to_vec()))
            .unzip();
        let (out1,) = out1.and_then(|out1| value((out1,))).apply(relu_apply).unzip();
        futures.push(tokio::spawn(out1.schedule(&devices[i])?));
    }

    // Wait on results.
    let mut outputs: Vec<Tensor<f32>> = vec![];
    for future in futures.into_iter() {
        let tensor = future.await.unwrap()?.unpartition();
        outputs.push(tensor);
    }
    for output in outputs {
        println!("{:?}", output.to_host_vec().await?);
    }

}
```

---

## Key Pattern: Compose Device Operations, Then Spawn

Every device operation in the loop below is non-blocking. The loop itself is non-blocking:

```rust
let mut futures: Vec<JoinHandle<Result<Partition<Tensor<f32>>, cuda_async::error::DeviceError>>> = vec![];
for i in 0..num_devices {
    // Obtain a reference to the model weights on device i.
    let w = &model_weights[i];
    let (w0, w1) = (w.0.clone(), w.1.clone());
    // Sample random data. Although the sampling procedure is a simulation,
    // this can be replaced with a procedure that actually samples a batch of data.
    let data = api::randn(0.0, 1.0, [dim, dim]).arc();
    // Construct the intermediate output buffer and partition, since we'll be writing to it.
    let out0 = api::zeros::<2, f32>([dim, dim]).partition([block_dim, block_dim]);
    // Execute GEMM.
    let (out0, _, _) = zip!(out0, data, value(w0))
        .apply(|args| gemm_apply(args).generics(fully_connected_layer.to_vec()))
        .unzip();
    // Construct the final output buffer and partition.
    let out1 = api::zeros::<1, f32>([dim]).partition([block_dim]);
    // Execute MatVec.
    let (out1, _, _) = zip!(out1, out0.unpartition().arc(), value(w1))
        .apply(|args| matvec_apply(args).generics(output_layer.to_vec()))
        .unzip();
    // Apply ReLU and unzip. We need to unzip here since arguments to kernels
    // are always packed into a tuple.
    let (out1,) = out1.and_then(|out1| value((out1,))).apply(relu_apply).unzip();
    // out1 now contains the work we would like to schedule on device i.
    // By invoking schedule on device i, we generate a device future which is
    // ready to execute on device i. By spawning a task for the device future,
    // we submit the work for execution to the async runtime (tokio). We then
    // collect the task handle into the futures vec.
    futures.push(tokio::spawn(out1.schedule(&devices[i])?));
}
```

After spawning tasks for each forward pass on each device, we wait on the results before proceeding:

```rust
let mut outputs: Vec<Tensor<f32>> = vec![];
for future in futures.into_iter() {
    let tensor = future.await.unwrap()?.unpartition();
    outputs.push(tensor);
}
```

---

## Key Takeaways

| Concept | What It Means |
|---------|---------------|
| **Device operations** | Chainable, resource-agnostic DAGs |
| **tokio::spawn** | Run batches concurrently |
| **schedule(device)** | Target a specific GPU |
| **Lazy execution** | Pipeline is built first, then executed on `.await` or `spawn()` |

---

### Exercise 1: Fuse the Kernel

How might we fuse the above kernels into a single kernel? Would this reduce the memory footprint of our computation?

### Exercise 2: Overlapping Data Movement with Computation

What would we need to change to construct a pipeline that overlaps data movement with computation?
