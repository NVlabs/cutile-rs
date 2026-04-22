# Integrating with CUDA C++

The tile model handles dense tensor algebra well — GEMM, element-wise operations, reductions, convolutions — but some algorithms depend on **warp-level primitives** (`__shfl_sync`, `__ballot_sync`, `__reduce_sync`) for things like custom scan/prefix-sum, cooperative groups, or irregular data access patterns. For these, write the kernel in CUDA C++ and integrate it using the approach below. A custom CUDA kernel can participate in the same `DeviceOp` execution model as your tile kernels — sharing streams, chaining with `.then()`, and avoiding unnecessary synchronization.

---

## Integrating a CUDA C++ Kernel

Compile your CUDA C++ kernel to PTX (portable) or a `.cubin` (architecture-specific):

```bash
# PTX — portable across GPU architectures, JIT-compiled at load time.
nvcc -ptx -arch=compute_80 my_kernel.cu -o my_kernel.ptx

# cubin — pre-compiled for a single architecture, no JIT overhead.
nvcc -cubin -arch=sm_80 my_kernel.cu -o my_kernel.cubin
```

> **Architecture portability:** A `.cubin` file only runs on the exact SM architecture it was compiled for. Code compiled with `-arch=sm_80` will not load on an `sm_100` GPU. PTX avoids this problem — the CUDA driver JIT-compiles it for the target GPU at load time, at the cost of a one-time compilation delay. Prefer PTX unless you need to eliminate JIT overhead.

Load the compiled module and get a handle to the entry function:

```rust
use cuda_async::device_context::{load_module_from_file, load_module_from_ptx};

// From cubin:
let module = load_module_from_file("my_kernel.cubin", device_id)?;

// Or from PTX (JIT-compiled at load time):
let ptx_src = include_str!("my_kernel.ptx");
let module = load_module_from_ptx(ptx_src, device_id)?;

let function = Arc::new(module.load_function("my_kernel_entry")?);
```

Launch the kernel via `AsyncKernelLaunch`, which is a `DeviceOp` wrapping the CUDA driver's kernel launch API:

```rust
use cuda_async::launch::AsyncKernelLaunch;
use cuda_core::LaunchConfig;

let mut launcher = AsyncKernelLaunch::new(function.clone());
launcher.push_arg(num_elements as u32);
launcher.push_arg(scale);
// SAFETY: input and output are valid device allocations with at least
// num_elements f32 elements. output is exclusively written; input is
// read-only. Both remain allocated until this operation completes.
unsafe {
    launcher
        .push_device_ptr(input.cu_deviceptr())
        .push_device_ptr(output.cu_deviceptr());
}
launcher.set_launch_config(LaunchConfig {
    grid_dim: ((num_elements as u32 + 255) / 256, 1, 1),
    block_dim: (256, 1, 1),
    shared_mem_bytes: 0,
});

// Execute as a DeviceOp — integrates with the async model.
launcher.await?;
```

Scalar arguments (types implementing `DType`) push safely with `push_arg`. Device pointers require `unsafe { push_device_ptr() }`: the Rust compiler has no visibility into GPU kernel code and cannot verify that the pointer refers to a valid allocation on the correct GPU, that the allocation is large enough for the kernel's access pattern, that no other operation is concurrently touching the same memory, or that the argument order and types match the kernel's signature. Neither the Rust compiler nor the CUDA driver validates these invariants — mistakes result in silent undefined behavior or hard-to-diagnose GPU faults, so you must verify them manually.

To prevent data races, use stream ordering: operations chained with `.then()` on the same stream execute in order and see each other's writes. Operations on different streams require explicit synchronization.

> **Why generated cuTile kernels don't require `unsafe`:** the `#[cutile::entry]` macro generates launchers that call `push_device_ptr` internally, but they do so safely because the framework controls both sides — device pointers come from framework-managed allocations, and the ownership model (`Partition` for exclusive access, `Arc<Tensor>` for shared reads) prevents aliasing at the type level. Custom kernels bypass this: you are pushing pointers the framework didn't allocate and can't track, so the safety burden falls on you.

---

## Wrapping as a Safe DeviceOp

Wrap a custom kernel launch in a struct that implements `DeviceOp`. The struct's typed fields enforce the correct argument signature; `unsafe` is confined to `execute`:

```rust
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_async::launch::AsyncKernelLaunch;
use cuda_async::scheduling_policies::SchedulingPolicy;
use cuda_core::{CudaFunction, LaunchConfig};
use std::future::IntoFuture;

pub struct ScaleKernel {
    function: Arc<CudaFunction>,
    n: u32,
    scale: f32,
    input: Arc<Tensor<f32>>,
    output: Tensor<f32>,
}

impl DeviceOp for ScaleKernel {
    type Output = (Arc<Tensor<f32>>, Tensor<f32>);

    // execute is unsafe because it enqueues async GPU work without
    // synchronizing — the returned tensors may still be in-flight.
    unsafe fn execute(
        self,
        ctx: &ExecutionContext,
    ) -> Result<<Self as DeviceOp>::Output, DeviceError> {
        let mut launcher = AsyncKernelLaunch::new(self.function);
        launcher.push_arg(self.n);
        launcher.push_arg(self.scale);
        // SAFETY: input and output are framework-managed Tensor allocations.
        // input is shared (Arc, read-only); output is exclusively written.
        unsafe {
            launcher
                .push_device_ptr(self.input.cu_deviceptr())
                .push_device_ptr(self.output.cu_deviceptr());
        }
        launcher.set_launch_config(LaunchConfig {
            grid_dim: ((self.n + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        });
        unsafe { launcher.execute(ctx)? };
        Ok((self.input, self.output))
    }
}

// IntoFuture is a supertrait of DeviceOp. Every custom DeviceOp
// needs this boilerplate to enable `.await` and `.sync()`.
impl IntoFuture for ScaleKernel {
    type Output = Result<(Arc<Tensor<f32>>, Tensor<f32>), DeviceError>;
    type IntoFuture = DeviceFuture<(Arc<Tensor<f32>>, Tensor<f32>), ScaleKernel>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.schedule(self)) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) | Err(e) => DeviceFuture::failed(e),
        }
    }
}
```

This is the same pattern the `#[cutile::entry]` macro uses to generate safe launchers for tile kernels — no `unsafe` at the call site. Once wrapped, the custom kernel composes with tile kernels through the usual `DeviceOp` combinators. This pipeline runs a tile add (`z = x + y`), then the wrapped scale kernel (`w = scale * z`):

```rust
let (z_part, _x, _y) =
    tile_add::add(z.partition([tile_size]), x.clone(), y.clone()).await?;
let z: Tensor<f32> = z_part.unpartition();

let w: Tensor<f32> = zeros(&[num_elements]).await?;
let (_z, w) = ScaleKernel {
    function: scale_function,
    n: num_elements as u32,
    scale,
    input: Arc::new(z),
    output: w,
}
.await?;
```

See [`interop.rs`](https://github.com/NVlabs/cutile-rs/blob/main/cutile-examples/examples/interop.rs) for a complete, runnable version.

---

## Low-Level Driver Access

For more direct control, use `with_context` to access the CUDA stream and issue driver API calls directly:

```rust
use cuda_async::device_operation::{with_context, value, DeviceOp};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{malloc_async, memcpy_htod_async, free_async};

let host_data: Vec<f32> = vec![1.0; num_elements];
let num_bytes = num_elements * std::mem::size_of::<f32>();

// host_data is captured by reference — it must outlive the await so that
// the async memcpy can read from it until the stream synchronizes.
let op = with_context(|ctx: &ExecutionContext| {
    let stream = ctx.get_cuda_stream();

    let dptr = unsafe {
        let dptr = malloc_async(num_bytes, stream);
        memcpy_htod_async(dptr, host_data.as_ptr(), num_elements, stream);
        dptr
    };

    value(dptr)
});

let dptr = op.await?;

// Clean up: free the device memory on a stream.
with_context(move |ctx: &ExecutionContext| {
    unsafe { free_async(dptr, ctx.get_cuda_stream()) };
    value(())
})
.await?;
```

This gives you full access to the CUDA driver API while participating in the `DeviceOp` model. Everything inside the `unsafe` block is your responsibility to get right.

---

## Migrating from Other DSLs

**From Triton.** [Triton](https://triton-lang.org/) and cuTile Rust both let you write kernels in terms of tile-level operations. Many patterns that require explicit warp specialization in Triton are handled implicitly by the cuTile Rust compiler:

| Triton (manual) | cuTile Rust (automatic) |
|-----------------|------------------------|
| Assign producer warps to prefetch tiles from global → shared memory | Compiler generates shared memory staging for `load_tile` operations |
| Assign consumer warps to compute on shared memory tiles | Compiler maps tile arithmetic to Tensor Cores and registers |
| Software pipeline with `warp_specialize` in `tl.range` | Compiler uses TMA for hardware-assisted pipelining on supported architectures |
| Manual `tl.dot` placement across warps | `mma()` maps directly to Tensor Core instructions; thread/warp assignment is compiler-managed |
| Tune `num_warps` and `num_stages` for occupancy | `occupancy` and `num_cta_in_cga` optimization hints guide the compiler |

For patterns that don't map to the tile model, compile the kernel with Triton and integrate it via `AsyncKernelLaunch` as described above. Since Triton outputs PTX, load it directly:

```rust
let module = load_module_from_ptx(triton_generated_ptx, device_id)?;
let function = Arc::new(module.load_function("gemm_kernel")?);
```

**From cuTile Python.** If you're familiar with [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/), the kernel-side concepts map directly:

| cuTile Python | cuTile Rust |
|---------------|-------------|
| `@ct.kernel` | `#[cutile::entry()]` |
| `ct.load()` | `load_tile_like_2d()` |
| `ct.store()` | `tensor.store()` |
| `ct.bid(0)` | Implicit via partition |
| `ct.launch()` | Async operation + `.await` |

Both front-ends use the same underlying Tile IR compilation pipeline and generate equivalent GPU code; the difference is the host language and its type system.

---

Continue to [Debugging and Profiling](debugging.md) for troubleshooting. This chapter builds on the `DeviceOp` model introduced in [Orchestrating Device Operations](device-operations.md).
