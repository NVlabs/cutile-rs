# Debugging and Profiling

This guide covers techniques for debugging cuTile Rust programs, organized by the typical workflow: inspect what the code is doing, understand errors, verify correctness, then profile performance.

---

## Inspecting Code and Values

**`cuda_tile_print!`** prints from inside a GPU kernel using printf-style formatting:

```rust
#[cutile::entry()]
fn debug_kernel<const S: [i32; 2]>(
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, {[-1, -1]}>
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let tile = load_tile_like(input, output);

    cuda_tile_print!("Block ({}, {}): loaded tile\n", pid.0, pid.1);

    output.store(tile);
}
```

GPU printing is slow and serializes tile block execution — use it only for small-grid debugging and remove before production.

**`cuda_tile_assert!`** asserts conditions inside a kernel:

```rust
let tile = load_tile_like(input, output);
cuda_tile_assert!(tile[0] > 0.0, "Value must be positive");
```

**Read back to the host** to inspect results after kernel execution:

```rust
let device = Device::new(0)?;
let stream = device.new_stream()?;

let x: Arc<Tensor<f32>> = ones(&[32, 32]).map(Into::into).sync_on(&stream)?;
let z = zeros(&[32, 32]).sync_on(&stream)?.partition([4, 4]);
let (z, _x) = my_kernel(z, x).sync_on(&stream)?;

let z_host: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream)?;
assert!(!z_host.iter().any(|x| x.is_nan()), "Output contains NaN!");
assert!(!z_host.iter().any(|x| x.is_infinite()), "Output contains Inf!");
println!("First 10 values: {:?}", &z_host[..10]);
```

**Inspect the generated MLIR** to see how your code is compiled, verify that optimizations are applied, or diagnose unexpected behavior. `print_ir = true` writes the IR to stdout during JIT compilation; `dump_mlir_dir` saves it to files for offline analysis; `use_debug_mlir` loads hand-modified MLIR instead of the compiler's output:

```rust
#[cutile::entry(
    print_ir = true,
    dump_mlir_dir = "/tmp/cutile-ir"
)]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }

#[cutile::entry(use_debug_mlir = "/path/to/custom.mlir")]
fn kernel_with_custom_mlir<const S: [i32; 2]>(...) { ... }
```

---

## Errors and Crashes

Most cuTile Rust errors surface at compile time. **Shape mismatches**, **type mismatches**, and **invalid reduction axes** are caught by the compiler before any kernel runs:

```rust
// Shape mismatch
let a: Tile<f32, {[64, 64]}> = ...;
let b: Tile<f32, {[32, 32]}> = ...;
let c = a + b;                       // Error: incompatible shapes

// Type mismatch
let float_tile: Tile<f32, S> = ...;
let int_tile: Tile<i32, S> = ...;
let result = float_tile + int_tile;  // Error: cannot add f32 and i32
                                      // Fix: convert_tile()

// Invalid reduction axis (tile is 2D, axes are 0 and 1 only)
let reduced = reduce_sum(tile, 2i32);  // Error
```

| Error | Cause | Fix |
|-------|-------|-----|
| Shape mismatch | Incompatible tile shapes | Align shapes or `reshape`/`broadcast` |
| Type mismatch | Wrong element types | Add explicit `convert_tile()` |
| Invalid axis | Reduction axis out of bounds | Use axis in `0..rank` |
| Not a power of 2 | Tile dimension isn't 2^n | Use power-of-2 dimensions |
| Missing entry | No `#[cutile::entry()]` | Add entry attribute |

Runtime errors typically come from **out-of-bounds accesses** (tensor smaller than expected tile size) or **numeric instability** (`exp` overflow in softmax-style kernels — always subtract the max before exponentiation). The common set:

| Error | Cause | Fix |
|-------|-------|-----|
| CUDA error: no kernel image | Wrong GPU architecture | Clear cache, rebuild |
| Failed to load kernel | CUDA toolkit issue | Check CUDA installation |
| Out of memory | Tensor too large | Reduce sizes or stream |
| Shape mismatch at runtime | Tensor not divisible by tile | Ensure divisibility |

**CPU segfaults** (SIGSEGV in the host process) are a different class — they typically mean something went wrong outside the GPU kernel itself, in the CUDA driver, JIT compilation, or host memory management. GPU kernels that access invalid memory usually surface as CUDA errors, not host segfaults.

Get a backtrace first:

```bash
RUST_BACKTRACE=1 cargo run
RUST_BACKTRACE=full cargo run   # with all frames, including inlined

# If the crash is inside a native library (CUDA driver, MLIR compiler):
gdb --args ./target/debug/my_program
(gdb) run
(gdb) bt
```

Common causes:

- **CUDA toolkit mismatch.** The JIT pipeline calls into CUDA libraries via FFI. An incompatible toolkit/driver pair, or a broken `CUDA_TOOLKIT_PATH`, can segfault in those FFI calls. Verify with `nvidia-smi`, `nvcc --version`, and `echo $CUDA_TOOLKIT_PATH`.
- **Use-after-free with raw pointers.** If you pass a `DevicePointer<T>` from `device_pointer()` into an unsafe raw-pointer kernel and drop the owning tensor before the kernel completes, the kernel operates on freed memory. Ensure all tensors outlive any kernel that uses their pointers.
- **Async lifetime issues.** With `tokio::spawn`, the kernel runs concurrently; if tensors are dropped before the spawned task completes, the kernel accesses freed memory. Await the spawn handle before tensors go out of scope.
- **OOM during JIT compilation.** The MLIR compiler allocates host memory during compilation. On RAM-constrained systems this can fail as a segfault rather than a clean error. Monitor host memory during the first kernel launch.

Diagnostic checklist for segfaults: Is `nvidia-smi` reporting a healthy driver? Does `CUDA_TOOLKIT_PATH` point to a valid toolkit? Are all tensors alive for the duration of any kernel that uses their pointers? If using `tokio::spawn`, are all handles awaited before tensors are dropped? Does the backtrace point into CUDA/MLIR libraries (toolkit issue) or your own code (lifetime issue)?

---

## Verifying Correctness

Start with minimal, manually verifiable inputs:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_small_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let expected = vec![11.0, 22.0, 33.0, 44.0];

        let result = run_add_kernel(&a, &b);
        assert_eq!(result, expected);
    }
}
```

Then compare GPU results against a known-correct CPU implementation:

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

If a fused kernel produces wrong results, split it into separate kernels and inspect intermediate results on the host. Each stage becomes its own testable unit.

---

## Profiling

**Nsight Compute** profiles individual kernel performance:

```bash
ncu --target-processes all ./my_cutile_program
ncu --set full -o profile_report ./my_cutile_program
```

Focus on memory throughput (close to peak for memory-bound kernels), compute throughput (percentage of peak ALU/Tensor Core utilization), occupancy (percentage of maximum warps active per SM), and stall reasons (why warps are waiting — memory, execution, synchronization).

**Nsight Systems** profiles system-wide behavior across CPU and GPU:

```bash
nsys profile ./my_cutile_program
nsys-ui report.nsys-rep
```

Look for kernel launch overhead (time between consecutive launches), memory transfer overlap (whether computation hides data transfers), and unnecessary CPU/GPU sync points.

A few environment variables help during debugging:

| Variable | Description | Default |
|----------|-------------|---------|
| `CUTILE_DUMP` | Dump compiler stages (`ast`, `resolved`, `typed`, `instantiated`, `ir`, `bytecode`, or `all`) | unset |
| `CUTILE_DUMP_FILTER` | Restrict dumps to matching function names or `module::function` paths | unset |
| `CUDA_VISIBLE_DEVICES` | Select GPU device | All GPUs |
| `CUDA_TOOLKIT_PATH` | Path to CUDA toolkit | Required by CUDA binding crates |
| `CUTILE_TILEIRAS_PATH` | Override the `tileiras` binary used by the JIT | `tileiras` from `PATH` |

The JIT kernel cache is in-memory per process — restart the process to force recompilation.

Pre-ship debugging checklist: shapes compatible (tile shapes match for operations; tensors divisible by tile size); types match (element types agree or are explicitly converted); algorithm correct (CPU reference produces expected results); numerically stable (no NaN/Inf in outputs; max subtracted before `exp`); small case passes (manually verifiable input produces correct output); IR looks right (`print_ir = true` shows expected operations).

---

Review [Tuning for Performance](performance-tuning.md) for optimization techniques, [Interoperability](interoperability.md) for custom CUDA kernels, the [DSL API](../reference/dsl-api.md) and [Host API](../reference/host-api.md) for API lookups, or the [Tutorials](../tutorials/01-hello-world.md) for worked examples.
