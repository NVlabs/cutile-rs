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

use data_parallel_module::{gemm, relu, matvec};

#[tokio::main]
async fn main() -> Result<(), DeviceError> {

    use cuda_async::device_operation::*;
    use data_parallel_module::{gemm, relu, matvec};
    use cutile::api;
    use cutile::tensor::{Unpartition, Partition, Tensor, ToHostVec};
    use cutile::tile_kernel::{PartitionOp, TileKernel};
    use cuda_async::device_context::global_policy;
    use cutile::api::dup;
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
    let w0 = api::randn(0.0f32, 1.0, [dim, dim]); // impl DeviceOp
    let w1 = api::randn(0.0f32, 1.0, [dim]); // impl DeviceOp
    let w = zip!(w0.map(Into::into), w1.map(Into::into)).schedule(&devices[0])?.await?;
    let mut joins = vec![];
    for i in 1..num_devices {
        let w_copy = tokio::spawn(zip!(dup(&w.0).map(Into::into), dup(&w.1).map(Into::into)).schedule(&devices[i])?);
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
        let data = api::randn(0.0, 1.0, [dim, dim]).map(Into::into);
        // Unified launcher: pass output partition and inputs directly.
        let out0 = api::zeros(&[dim, dim]).partition([block_dim, block_dim]);
        let out0 = gemm(out0, data, w0)
            .generics(fully_connected_layer.to_vec())
            .first()
            .unpartition();
        let out1 = api::zeros(&[dim]).partition([block_dim]);
        let out1 = matvec(out1, out0.map(Into::into), w1)
            .generics(output_layer.to_vec())
            .first()
            .unpartition();
        let (out1,) = relu(out1.partition([block_dim])).unzip();
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

    Ok(())
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
    let data = api::randn(0.0, 1.0, [dim, dim]).map(Into::into);
    // Unified launcher: pass output partition and inputs directly.
    let out0 = api::zeros(&[dim, dim]).partition([block_dim, block_dim]);
    let out0 = gemm(out0, data, w0)
        .generics(fully_connected_layer.to_vec())
        .first()
        .unpartition();
    // Final output: matvec + relu.
    let out1 = api::zeros(&[dim]).partition([block_dim]);
    let out1 = matvec(out1, out0.map(Into::into), w1)
        .generics(output_layer.to_vec())
        .first()
        .unpartition();
    // Apply ReLU. Partition for tile-level dispatch, then unzip the result.
    let (out1,) = relu(out1.partition([block_dim])).unzip();
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
