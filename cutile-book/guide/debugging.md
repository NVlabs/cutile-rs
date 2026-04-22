# Debugging and Profiling

This guide covers techniques for debugging cuTile Rust programs, organized by the typical debugging workflow: inspect the error, inspect values, verify correctness, then profile performance.

## Inspecting Values

### Device-Side: `cuda_tile_print!`

Print from inside a GPU kernel using printf-style formatting:

```rust
#[cutile::entry()]
fn debug_kernel<const S: [i32; 2]>(
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, {[-1, -1]}>
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let tile = load_tile_like_2d(input, output);

    cuda_tile_print!("Block ({}, {}): loaded tile\n", pid.0, pid.1);

    output.store(tile);
}
```

> **Note:** GPU printing is slow and serializes tile block execution. Use only for debugging small grids and remove before production.

### Device-Side: `cuda_tile_assert!`

Assert conditions inside a GPU kernel:

```rust
let tile = load_tile_like_2d(input, output);
cuda_tile_assert!(tile[0] > 0.0, "Value must be positive");
```

### Host-Side: Read Back and Inspect

Transfer results to the CPU to inspect them after kernel execution:

```rust
let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;

let x: Arc<Tensor<f32>> = ones(&[32, 32]).map(Into::into).sync_on(&stream)?;
let z = zeros(&[32, 32]).sync_on(&stream)?.partition([4, 4]);

let (z, _x) = my_kernel(z, x).sync_on(&stream)?;

// Read output back to host
let z_host: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream)?;

// Check for NaN/Inf
assert!(!z_host.iter().any(|x| x.is_nan()), "Output contains NaN!");
assert!(!z_host.iter().any(|x| x.is_infinite()), "Output contains Inf!");

println!("First 10 values: {:?}", &z_host[..10]);
```

---

## Inspecting Generated Code

### Print IR

Use `print_ir = true` to see the generated MLIR during JIT compilation:

```rust
#[cutile::entry(print_ir = true)]
fn debug_ir_kernel<const S: [i32; 2]>(
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, {[-1, -1]}>
) {
    let tile = load_tile_like_2d(input, output);
    output.store(tile * 2.0);
}
```

This is useful for understanding how your code is compiled, verifying that optimizations are applied, and diagnosing unexpected behavior.

### Dump IR to Files

Save the generated MLIR to a directory for detailed offline analysis:

```rust
#[cutile::entry(
    print_ir = true,
    dump_mlir_dir = "/tmp/cutile-ir"
)]
fn kernel_with_ir_dump<const S: [i32; 2]>(...) { ... }
```

### Load Custom MLIR (Advanced)

For advanced debugging, load hand-modified MLIR instead of the compiler-generated version:

```rust
#[cutile::entry(use_debug_mlir = "/path/to/custom.mlir")]
fn kernel_with_custom_mlir<const S: [i32; 2]>(...) { ... }
```

---

## Common Errors and Fixes

### Compile-Time Errors

**Shape mismatch** — Tile shapes must be compatible for operations:

```rust
let a: Tile<f32, {[64, 64]}> = ...;
let b: Tile<f32, {[32, 32]}> = ...;
let c = a + b;  // Compile error: incompatible shapes
```

Fix: ensure shapes match, or use `reshape` and `broadcast` to align them.

**Type mismatch** — Element types must match for arithmetic:

```rust
let float_tile: Tile<f32, S> = ...;
let int_tile: Tile<i32, S> = ...;
let result = float_tile + int_tile;  // Error: cannot add f32 and i32
```

Fix: use `convert_tile()` for explicit type conversion.

**Invalid reduction axis** — The axis must be in range `0..rank`:

```rust
let tile: Tile<f32, {[64, 64]}> = ...;  // 2D: axes 0 and 1
let reduced = reduce_sum(tile, 2i32);    // Error: axis 2 doesn't exist
```

| Error | Cause | Fix |
|-------|-------|-----|
| Shape mismatch | Incompatible tile shapes | Fix shapes or add broadcasting |
| Type mismatch | Wrong element types | Add explicit `convert_tile()` |
| Invalid axis | Reduction axis out of bounds | Use axis in `0..rank` |
| Not a power of 2 | Tile dimension not 2^n | Use power-of-2 dimensions |
| Missing entry | No `#[cutile::entry()]` | Add entry attribute |

### Runtime Errors

**Out-of-bounds access** — If a tensor is smaller than the tile size, loads may read out of bounds:

```rust
fn kernel<const S: [i32; 2]>(
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, {[-1, -1]}>
) {
    // If input is smaller than S in any dimension, this may crash
    let tile = load_tile_like_2d(input, output);
}
```

Fix: ensure tensor dimensions are at least as large as the tile dimensions, or that partitioning produces only in-bounds tiles.

**Numeric instability** — Operations like `exp` can overflow; always subtract the maximum before exponentiation:

```rust
// Unstable:
let result = exp(large_values);  // May overflow to Inf

// Stable:
let max_val = reduce_max(tile, 1i32);
let shifted = tile - max_val.reshape(const_shape![BM, 1]).broadcast(tile.shape());
let result = exp(shifted);  // Safe: shifted values are <= 0
```

| Error | Cause | Fix |
|-------|-------|-----|
| CUDA error: no kernel image | Wrong GPU architecture | Clear cache, rebuild |
| Failed to load kernel | CUDA toolkit issue | Check CUDA installation |
| Out of memory | Tensor too large | Reduce sizes or use streaming |
| Shape mismatch at runtime | Tensor not divisible by tile | Ensure divisibility |

---

## CPU Segfaults

A segfault (signal 11 / SIGSEGV) in the host process typically means something went wrong outside the GPU kernel itself — in the CUDA driver, the JIT compilation pipeline, or host-side memory management. GPU kernels that access invalid memory usually surface as CUDA errors, not host segfaults.

### Getting a Backtrace

The first step is always to capture a backtrace:

```bash
RUST_BACKTRACE=1 cargo run
```

For a full backtrace with all frames (including inlined functions):

```bash
RUST_BACKTRACE=full cargo run
```

If the segfault occurs inside a native library (CUDA driver, MLIR compiler), the Rust backtrace may be truncated. Use `gdb` to get the full native stack:

```bash
gdb --args ./target/debug/my_program
(gdb) run
# ... segfault happens ...
(gdb) bt
```

### Common Causes

**CUDA toolkit mismatch** — The JIT compilation pipeline calls into CUDA libraries via FFI. If the CUDA toolkit version is incompatible with the installed driver, or if `CUDA_HOME` points to a missing or broken installation, these FFI calls can segfault.

```bash
# Verify your CUDA installation
nvidia-smi              # Check driver version
nvcc --version          # Check toolkit version
echo $CUDA_HOME         # Check toolkit path
```

Fix: ensure the CUDA toolkit version is compatible with the installed driver, and that `CUDA_HOME` is set correctly.

**Use-after-free with raw pointers** — If you extract a raw device pointer via `device_pointer()` and then drop the owning tensor before the kernel completes, the kernel operates on freed memory. This can corrupt host-side state and cause a segfault when the results are read back.

```rust
// WRONG: tensor dropped while kernel is still running
let z: Tensor<f32> = zeros(&[len]).await?;
let z_ptr = z.device_pointer();
drop(z);  // Frees GPU memory!
unsafe { add_ptr(z_ptr, ...) }.sync_on(&stream)?;  // Segfault or corruption
```

Fix: ensure all tensors outlive any kernel that uses their pointers. The `await` or `.sync_on()` call must complete before the tensor is dropped.

**Async lifetime issues** — With `tokio::spawn`, the kernel runs concurrently. If tensors are moved or dropped before the spawned task completes, the kernel may access freed memory.

```rust
// WRONG: tensor may be dropped before kernel finishes
let handle = tokio::spawn(kernel_op.into_future());
// ... tensor goes out of scope here ...

// Fix: await the handle before tensors are dropped
let result = handle.await?;
```

**Out-of-memory during JIT compilation** — The MLIR compiler allocates host memory during compilation. On systems with limited RAM, this can fail and manifest as a segfault rather than a clean error.

Fix: monitor host memory usage during the first kernel launch (when JIT compilation occurs). If memory is the issue, reduce the number of concurrent compilations or increase system RAM.

### Diagnostic Checklist

- [ ] Does `nvidia-smi` report a healthy driver?
- [ ] Does `CUDA_HOME` point to a valid toolkit installation?
- [ ] Are all tensors alive for the duration of any kernel that uses their pointers?
- [ ] If using `tokio::spawn`, are all handles awaited before tensors are dropped?
- [ ] Does the backtrace point into CUDA/MLIR libraries (toolkit issue) or your code (lifetime issue)?

---

## Verifying Correctness

### Small Test Cases

Start with minimal, manually verifiable inputs:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_small_add() {
        // Use small sizes where you can verify by hand
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let expected = vec![11.0, 22.0, 33.0, 44.0];

        let result = run_add_kernel(&a, &b);
        assert_eq!(result, expected);
    }
}
```

### Reference Implementation

Compare GPU results against a known-correct CPU implementation:

```rust
fn cpu_softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = input.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|x| x / sum).collect()
}

fn test_softmax_correctness() {
    let input = random_input(1024);
    let cpu_result = cpu_softmax(&input);
    let gpu_result = run_softmax_kernel(&input);

    for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
        assert!((cpu - gpu).abs() < 1e-5, "Mismatch: CPU={}, GPU={}", cpu, gpu);
    }
}
```

### Decompose Complex Kernels

If a fused kernel produces wrong results, split it into separate kernels to isolate the bug:

```rust
// Instead of one complex kernel, run each step separately:
fn debug_step1<const S: [i32; 2]>(out: &mut Tensor<f32, S>, input: &Tensor<f32, {[-1, -1]}>) {
    out.store(step1(input));
}
fn debug_step2<const S: [i32; 2]>(out: &mut Tensor<f32, S>, input: &Tensor<f32, {[-1, -1]}>) {
    out.store(step2(input));
}

// Run each step and inspect intermediate results on the host
```

---

## Profiling

### Nsight Compute

Profile individual kernel performance:

```bash
ncu --target-processes all ./my_cutile_program
ncu --set full -o profile_report ./my_cutile_program
```

Key metrics:
- **Memory Throughput** — Should be close to theoretical peak for memory-bound kernels.
- **Compute Throughput** — Percentage of peak ALU/Tensor Core utilization.
- **Occupancy** — Percentage of maximum warps active on each SM.
- **Stall Reasons** — Why warps are waiting (memory, execution, synchronization).

### Nsight Systems

Profile system-wide behavior across CPU and GPU:

```bash
nsys profile ./my_cutile_program
nsys-ui report.nsys-rep
```

Look for:
- **Kernel launch overhead** — Time between consecutive launches.
- **Memory transfer overlap** — Whether computation hides data transfers.
- **CPU/GPU sync points** — Unnecessary blocking waits.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CUTILE_DEBUG` | Enable debug output | `0` |
| `CUDA_VISIBLE_DEVICES` | Select GPU device | All GPUs |
| `CUDA_HOME` | Path to CUDA toolkit | `/usr/local/cuda` |

```bash
CUTILE_DEBUG=1 cargo run              # Debug kernel compilation
CUDA_VISIBLE_DEVICES=0 cargo run      # Select specific GPU
```

The JIT kernel cache is in-memory per process. Restart the process to force recompilation.

---

## Debugging Checklist

- [ ] **Shapes compatible?** — Tile shapes match for operations; tensors divisible by tile size.
- [ ] **Types match?** — Element types agree or are explicitly converted.
- [ ] **Algorithm correct?** — CPU reference implementation produces expected results.
- [ ] **Numerically stable?** — No NaN/Inf in outputs; max subtracted before `exp`.
- [ ] **Small case passes?** — Manually verifiable test case produces correct output.
- [ ] **IR looks right?** — `print_ir = true` shows expected operations.

---

## Next Steps

- Review [Tuning for Performance](performance-tuning.md) for optimization techniques
- Review [Integrating with CUDA C++](interoperability.md) for integrating custom CUDA kernels
- Look up APIs in the [DSL API Reference](../reference/dsl-api.md) and [DeviceOp API Reference](../reference/deviceop-reference.md)
- Try the [Tutorials](../tutorials/01-hello-world.md) for working examples
