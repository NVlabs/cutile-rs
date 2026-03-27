# Interoperability with Custom CUDA Kernels

cuTile Rust's tile programming model excels at dense tensor algebra that decomposes naturally into tile primitives — matrix multiplications, convolutions, reductions, and element-wise operations. The compiler handles thread management, memory staging, and synchronization, producing highly optimized code for these patterns.

However, some GPU algorithms require fine-grained control over individual threads within a warp. This chapter explains when you might need to step outside the tile model and how to integrate custom CUDA kernels into a cuTile Rust program.

---

## When Tile Programming Is the Right Choice

The tile model is a good fit when your kernel follows the **load → compute → store** pattern on rectangular data:

- **Dense linear algebra**: GEMM, GEMV, batched operations
- **Element-wise operations**: activation functions, scaling, broadcasting
- **Reductions**: sum, softmax, layer normalization
- **Stencil computations**: convolutions, pooling

For these patterns, the tile compiler often produces code competitive with hand-tuned CUDA C++, with far less effort.

## When You Need Something Else

Certain algorithms depend on **warp-level primitives** — operations like `__shfl_sync`, `__ballot_sync`, or `__reduce_sync` that coordinate threads within a 32-thread warp. These include:

- **Warp-specialized kernels**: algorithms where different warps play different roles (e.g., producer/consumer patterns in some FlashAttention variants)
- **Custom scan/prefix-sum**: tree-based reductions across warp lanes
- **Fine-grained synchronization**: lock-free data structures, cooperative groups
- **Irregular data access**: graph algorithms, sparse matrix operations

The tile abstraction intentionally hides thread-level detail. If your algorithm's performance depends on explicit control of warp lanes, write that kernel in CUDA C++ and integrate it using the approach below.

---

## Integrating Custom CUDA Kernels

The key insight is that you don't need to leave the `DeviceOperation` execution model. A custom CUDA kernel can participate in the same computation graph as your tile kernels — sharing streams, chaining with `.and_then()`, and avoiding unnecessary synchronization.

### Step 1: Compile Your CUDA Kernel

Compile your CUDA C++ kernel to a `.cubin` or `.ptx` file:

```bash
nvcc -cubin -arch=sm_80 my_kernel.cu -o my_kernel.cubin
```

### Step 2: Load the Module and Function

Use `cuda-async`'s module loading functions to load the compiled kernel:

```rust
use cuda_async::device_context::load_module_from_file;

let module = load_module_from_file("my_kernel.cubin", device_id)?;
let function = Arc::new(module.load_function("my_kernel_entry")?);
```

For PTX (JIT-compiled at runtime):

```rust
use cuda_async::device_context::load_module_from_ptx;

let ptx_src = include_str!("my_kernel.ptx");
let module = load_module_from_ptx(ptx_src, device_id)?;
let function = Arc::new(module.load_function("my_kernel_entry")?);
```

### Step 3: Launch via AsyncKernelLaunch

`AsyncKernelLaunch` is a `DeviceOperation` that wraps the CUDA driver's kernel launch API:

```rust
use cuda_async::launch::{AsyncKernelLaunch, KernelArgument};
use cuda_core::LaunchConfig;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher
    .push_arg(Box::new(num_elements as u32))
    .push_arg(Box::new(x.device_pointer()))
    .push_arg(Box::new(y.device_pointer()))
    .push_arg(Box::new(z.device_pointer()))
    .set_launch_config(LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    });

// Execute as a DeviceOperation — integrates with the async model.
launcher.await?;
```

### Step 4: Compose with Tile Kernels

Because `AsyncKernelLaunch` implements `DeviceOperation`, it chains naturally with tile kernels:

```rust
use cuda_async::device_operation::DeviceOperation;

// Allocate input with cuTile Rust.
let x = api::randn(0.0f32, 1.0, [m, n]).arc().await?;

// Run a cuTile Rust tile kernel for the first stage.
let (y,) = stage1_op(y_part, x.device_operation()).unzip();
let y: Tensor<f32> = y.unpartition().await?;

// Chain a custom CUDA kernel for the second stage.
let mut launcher = AsyncKernelLaunch::new(custom_function.clone());
launcher
    .push_arg(Box::new(y.device_pointer()))
    .push_arg(Box::new(z.device_pointer()))
    .push_arg(Box::new(num_elements as u32))
    .set_launch_config(cfg);

// .and_then() ensures the custom kernel runs on the same stream,
// seeing all writes from the tile kernel.
let result = value(z)
    .and_then(move |z| {
        launcher.and_then(move |_| value(z))
    })
    .await?;
```

---

## Using `with_context` for Low-Level Control

For more direct control, use `with_context` to access the CUDA stream and issue driver API calls directly:

```rust
use cuda_async::device_operation::{with_context, value, DeviceOperation};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{malloc_async, memcpy_htod_async, free_async};

let op = with_context(|ctx: &ExecutionContext| {
    let stream = ctx.get_cuda_stream();

    unsafe {
        // Allocate device memory on this stream.
        let dptr = malloc_async(num_bytes, stream);
        memcpy_htod_async(dptr, host_data.as_ptr(), num_elements, stream);

        // Launch a custom kernel on the same stream.
        cuda_core::launch_kernel(
            function.cu_function(),
            (grid_x, grid_y, grid_z),
            (block_x, block_y, block_z),
            shared_mem_bytes,
            stream.cu_stream(),
            &mut args,
        ).expect("Kernel launch failed");
    }

    value(())
});

// This is a DeviceOperation — await it, chain it, or sync it.
op.await?;
```

This pattern gives you full access to the CUDA driver API while still participating in the `DeviceOperation` execution model.

---

## Coming from Triton: Warp Specialization in the Tile Model

[Triton](https://triton-lang.org/) and cuTile Rust share a similar programming model — both let you write kernels in terms of tile-level operations rather than individual threads. If you are coming from Triton and rely on warp specialization (e.g., `tl.async_task` for producer/consumer patterns), this section explains what the cuTile Rust compiler handles automatically and where the models differ.

### What the Tile Compiler Already Does

Many patterns that require explicit warp specialization in Triton are handled implicitly by the cuTile Rust compiler:

| Triton (manual) | cuTile Rust (automatic) |
|-----------------|------------------------|
| Assign producer warps to prefetch tiles from global → shared memory | Compiler generates shared memory staging for `load_tile` operations |
| Assign consumer warps to compute on shared memory tiles | Compiler maps tile arithmetic to Tensor Cores and registers |
| Software pipeline with `tl.async_task` to overlap loads and compute | `allow_tma = true` enables hardware-assisted pipelining via TMA on Hopper+ |
| Manual `tl.dot` placement across warps | `mma()` maps directly to Tensor Core instructions; thread/warp assignment is compiler-managed |
| Tune `num_warps` and `num_stages` for occupancy | `occupancy` and `num_cta_in_cga` optimization hints guide the compiler |

In short: the standard producer/consumer pipelining pattern — where some warps prefetch the next tile while others compute on the current tile — is what the tile compiler's code generation already targets. You express the algorithm; the compiler decides how to schedule it across warps and pipeline stages.

### Example: Pipelined GEMM

In Triton, a warp-specialized GEMM might look like:

```python
# Triton: explicit producer/consumer warp specialization
@triton.jit
def gemm_kernel(...):
    with tl.async_task([0]):          # Producer warp: prefetch A, B tiles
        a = tl.load(A_ptr + offsets)
        b = tl.load(B_ptr + offsets)
    with tl.async_task([1, 2]):       # Consumer warps: compute
        c += tl.dot(a, b)
```

The equivalent cuTile Rust kernel expresses the same algorithm without explicit warp roles:

```rust
#[cutile::entry(
    optimization_hints = (
        tensor_dim_factor = 16,
        sm_120 = (allow_tma = true, occupancy = 2,),
    )
)]
fn gemm<const S: [i32; 2]>(
    c: &mut Tensor<f32, S>,
    a: &Tensor<f32, { [-1, -1] }>,
    b: &Tensor<f32, { [-1, -1] }>,
) {
    // The compiler stages these loads through shared memory
    // and pipelines them with the compute below.
    let tile_a = load_tile_like_2d(a, c);
    let tile_b = load_tile_like_2d(b, c);
    c.store(mma(tile_a, tile_b));
}
```

The `allow_tma = true` hint enables Tensor Memory Accelerator-based bulk copies on Hopper+, which is the hardware mechanism underlying efficient producer/consumer pipelining. The compiler decides how to distribute work across warps and pipeline stages.

### When You Still Need Custom Kernels

The tile model covers the common case — pipelined dense tensor algebra — but some Triton patterns have no direct equivalent:

- **Heterogeneous warp roles** beyond producer/consumer (e.g., three distinct task types within a block)
- **Warp-level reductions** using `tl.reduce` with custom combiners that don't map to tile primitives
- **Dynamic control flow per warp** where different warps take different code paths based on runtime conditions
- **Cross-block communication** via atomic operations or cooperative groups

For these, compile the kernel with Triton (or write it in CUDA C++) and integrate it via `AsyncKernelLaunch` as described in the sections above. Since Triton outputs PTX, you can load it directly:

```rust
let module = load_module_from_ptx(triton_generated_ptx, device_id)?;
let function = Arc::new(module.load_function("gemm_kernel")?);
```

---

## Recommendations

1. **Try tile programming first.** If your kernel can be expressed as tile operations, the compiler will handle thread management and memory staging. The resulting code is safer and often performs well.

2. **Use CUDA C++ for warp-specialized kernels.** When you need explicit warp-level control, write the kernel in CUDA C++ and integrate it via `AsyncKernelLaunch`.

3. **Avoid unnecessary synchronization.** By implementing your custom kernel as a `DeviceOperation`, you can chain it with `.and_then()` alongside tile kernels. All operations on the same stream execute in order — there's no need to sync between stages.

4. **Keep data on the device.** Use `device_pointer()` to pass tensor data to custom kernels without copying back to the host. The data stays in GPU memory throughout the pipeline.

---

Continue to [Performance Tuning](performance-tuning.md) for optimization techniques, or see the [Async Execution](async-execution.md) guide for more on `DeviceOperation` composition.
