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

Compile your CUDA C++ kernel to PTX (portable) or a `.cubin` (architecture-specific):

```bash
# PTX — portable across GPU architectures, JIT-compiled at load time.
nvcc -ptx -arch=compute_80 my_kernel.cu -o my_kernel.ptx

# cubin — pre-compiled for a single architecture, no JIT overhead.
nvcc -cubin -arch=sm_80 my_kernel.cu -o my_kernel.cubin
```

> **Architecture portability:** A `.cubin` file only runs on the exact SM architecture it was compiled for. Code compiled with `-arch=sm_80` will not load on an `sm_100` GPU. PTX avoids this problem — the CUDA driver JIT-compiles it for the target GPU at load time, at the cost of a one-time compilation delay. Prefer PTX unless you need to eliminate JIT overhead. If you must ship `.cubin` files, compile for each target architecture:
>
> ```bash
> nvcc -cubin -gencode arch=compute_80,code=sm_80 \
>              -gencode arch=compute_100,code=sm_100 \
>              my_kernel.cu -o my_kernel.cubin
> ```

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
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher.push_arg(num_elements as u32);
// SAFETY: x, y, z are valid device allocations with at least num_elements
// f32 elements. z is exclusively written; x and y are read-only.
// All three remain allocated until this operation completes.
unsafe {
    launcher
        .push_arg_raw(Box::new(x.device_pointer()))
        .push_arg_raw(Box::new(y.device_pointer()))
        .push_arg_raw(Box::new(z.device_pointer()));
}
launcher.set_launch_config(LaunchConfig {
    grid_dim: (grid_size, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
});

// Execute as a DeviceOperation — integrates with the async model.
launcher.await?;
```

Scalar arguments (types implementing `DType`) can be pushed safely with `push_arg`. Device pointers must use `unsafe { push_arg_raw() }` — see [Safety: Device Pointer Arguments](#safety-device-pointer-arguments) below.

### Safety: Device Pointer Arguments

When you call `push_arg_raw` to pass a device pointer, you are telling the CUDA driver "here is an address the kernel should access." The Rust compiler has no visibility into GPU kernel code and cannot verify that:

- The pointer refers to a valid device memory allocation on the correct GPU.
- The allocation is large enough for the kernel's access pattern.
- No other operation is concurrently reading or writing the same memory.

By contrast, scalar arguments (like `num_elements as u32`) are copied into the kernel's parameter space during launch setup — the kernel reads the value, not an address. No device memory is dereferenced, so there is no validity or aliasing concern. Any type implementing `DType` can be pushed safely with `push_arg`.

To prevent data races on device memory, use the `DeviceOperation` model's stream ordering guarantees: operations chained with `.and_then()` on the same stream execute in order and see each other's writes. If two operations access the same memory on different streams, explicit synchronization is required.

> **Why generated cuTile Rust kernels don't require `unsafe`:** When you write a tile kernel with `#[cutile::entry]`, the generated launcher uses the `KernelArgument` and `ArcKernelArgument` implementations for `Tensor<T>` and `Partition<Tensor<T>>`. These implementations call `push_arg_raw` internally, but can do so safely because the framework controls both sides: device pointers come from framework-managed allocations (guaranteed valid), and the ownership model — `Partition` for exclusive access, `Arc<Tensor>` for shared reads — prevents aliasing at the type level. Custom kernels bypass this: you are pushing pointers that the framework didn't allocate and can't track, so the safety burden falls on you.

You can apply the same pattern to your own custom kernels by wrapping the launch in a struct that implements `DeviceOperation`. The struct's typed fields enforce the correct argument signature, and the `unsafe` is confined to the `execute` implementation:

```rust
use cuda_async::device_operation::{DeviceOperation, ExecutionContext};
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;

pub struct MyCustomKernel {
    function: Arc<CudaFunction>,
    n: u32,
    x: Arc<Tensor<f32>>,
    y: Partition<Tensor<f32>>,
}

impl DeviceOperation for MyCustomKernel {
    type Output = (Arc<Tensor<f32>>, Partition<Tensor<f32>>);

    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<Self::Output, DeviceError> {
        let mut launcher = AsyncKernelLaunch::new(self.function.clone());
        launcher.push_arg(self.n);
        // SAFETY: x and y are framework-managed Tensor allocations.
        // x is shared (Arc, read-only); y is exclusive (Partition, written).
        unsafe {
            launcher
                .push_arg_raw(Box::new(self.x.cu_deviceptr()))
                .push_arg_raw(Box::new(self.y.object.cu_deviceptr()));
        }
        launcher.set_launch_config(LaunchConfig {
            grid_dim: self.y.grid()?,
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        });
        launcher.execute(ctx)?;
        Ok((self.x, self.y))
    }
}
```

Callers construct `MyCustomKernel` with typed arguments and launch it like any other `DeviceOperation` — no `unsafe` at the call site:

```rust
let result = MyCustomKernel {
    function: function.clone(),
    n: num_elements,
    x: x.clone(),
    y: y_part,
}.await?;
```

This is the same pattern the `#[cutile::entry]` macro uses to generate safe launchers for tile kernels. The struct owns its arguments, the `DeviceOperation` trait provides stream ordering, and `unsafe` is scoped to the one place that marshals device pointers.

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
// SAFETY: y is read-only; z is exclusively written. Both are valid
// allocations that outlive this operation.
unsafe {
    launcher
        .push_arg_raw(Box::new(y.device_pointer()))
        .push_arg_raw(Box::new(z.device_pointer()));
}
launcher
    .push_arg(num_elements as u32)
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
| Software pipeline with `tl.async_task` to overlap loads and compute | `allow_tma = true` enables hardware-assisted pipelining via TMA on supported architectures (sm_100+) |
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

The `allow_tma = true` hint enables Tensor Memory Accelerator-based bulk copies on architectures that support TMA, which is the hardware mechanism underlying efficient producer/consumer pipelining. The compiler decides how to distribute work across warps and pipeline stages.

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

## Error Handling

Custom kernel integration can fail at several points. Here are the common errors and how to address them:

**Module loading** — `load_module_from_file` and `load_module_from_ptx` return `DeviceError` if the file is missing, the binary is corrupt, or the architecture doesn't match the current GPU. A `.cubin` compiled for `sm_80` will fail to load on an `sm_100` device. PTX avoids this class of error entirely.

**Missing launch configuration** — Calling `.await` or `.sync()` on an `AsyncKernelLaunch` without first calling `set_launch_config` produces a `DeviceError::Launch` with the message "Await called before launching the kernel."

**Invalid launch parameters** — The CUDA driver rejects launches with grid or block dimensions of zero, block dimensions exceeding device limits (typically 1024 threads per block), or shared memory requests exceeding the device's per-block limit. These surface as `DeviceError::Launch` with context from the driver.

**Argument mismatches** — If the number or types of pushed arguments don't match the kernel's signature, behavior is undefined. The CUDA driver does not validate argument layouts at launch time. Double-check that `push_arg` and `push_arg_raw` calls match the kernel's parameter list in order, count, and size.

For debugging launch failures, see [Debugging](debugging.md). Setting the environment variable `CUDA_LAUNCH_BLOCKING=1` forces synchronous kernel execution, which makes error messages report the exact failing kernel rather than a later operation.

---

## Recommendations

1. **Try tile programming first.** If your kernel can be expressed as tile operations, the compiler will handle thread management and memory staging. The resulting code is safer and often performs well.

2. **Use CUDA C++ for warp-specialized kernels.** When you need explicit warp-level control, write the kernel in CUDA C++ and integrate it via `AsyncKernelLaunch`.

3. **Prefer PTX for portability.** PTX kernels are JIT-compiled for the target GPU at load time, avoiding architecture-specific `.cubin` builds. Use `.cubin` only when JIT overhead is unacceptable.

4. **Keep `unsafe` scoped to device pointer arguments.** Push scalar arguments with `push_arg` and device pointers with `push_arg_raw` inside a focused `unsafe` block. Document the safety invariants — pointer validity, allocation size, and aliasing — in a `// SAFETY:` comment.

5. **Avoid unnecessary synchronization.** By implementing your custom kernel as a `DeviceOperation`, you can chain it with `.and_then()` alongside tile kernels. All operations on the same stream execute in order — there's no need to sync between stages.

6. **Keep data on the device.** Use `device_pointer()` to pass tensor data to custom kernels without copying back to the host. The data stays in GPU memory throughout the pipeline.

---

Continue to [Debugging](debugging.md) for troubleshooting, or see [Performance Tuning](performance-tuning.md) for optimization techniques. This chapter builds on the `DeviceOperation` model introduced in [Async Execution](async-execution.md).
