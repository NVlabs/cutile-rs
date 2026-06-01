# Debugging and Profiling

Start debugging with small, deterministic inputs. Read results back to the host, compare against a CPU reference, then inspect generated Tile IR or profile the GPU when correctness is established.

## Printing and Assertions

`cuda_tile_print!` prints from inside a GPU kernel:

```rust
#[cutile::entry()]
fn debug_kernel<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,
    x: &Tensor<f32, { [-1, -1] }>,
) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let tile = load_tile_like(x, z);

    cuda_tile_print!("Block ({}, {}): loaded tile\n", pid.0, pid.1);
    z.store(tile);
}
```

GPU printing is slow and serializes tile block execution. Use it for small grids and remove it before measuring performance.

`cuda_tile_assert!` checks conditions inside a kernel:

```rust
let tile = load_tile_like(x, z);
cuda_tile_assert!(tile[0] > 0.0, "expected positive input");
```

## Host Readback

Host readback is a `DeviceOp`; execute it before reading the host vector:

```rust
let z_host: Vec<f32> = z
    .unpartition()
    .to_host_vec()
    .sync_on(&stream)?;

assert!(!z_host.iter().any(|x| x.is_nan()));
assert!(!z_host.iter().any(|x| x.is_infinite()));
```

If a fused kernel is wrong, split it into stages and read back each intermediate. Each stage should match a simple CPU implementation on a small input.

## Correctness Tests

Use minimal inputs first:

```rust
#[test]
fn small_add_matches_cpu() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let expected = vec![11.0, 22.0, 33.0, 44.0];

    let result = run_add_kernel(&a, &b);
    assert_eq!(result, expected);
}
```

Then compare larger random inputs against a CPU reference with an appropriate tolerance:

```rust
for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
    assert!((cpu - gpu).abs() < 1e-5, "CPU={cpu}, GPU={gpu}");
}
```

For numerically sensitive kernels, test edge cases: zeros, large positive values, large negative values, non-divisible shapes if supported, and known overflow-prone inputs.

## Inspecting Tile IR

`print_ir = true` prints the generated wrapper, source kernel, and Tile IR text during JIT compilation:

```rust
#[cutile::entry(print_ir = true)]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

`dump_mlir_dir` writes the compiled Tile IR text to files:

```rust
#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

`use_debug_mlir` loads hand-modified Tile IR text:

```rust
#[cutile::entry(use_debug_mlir = "/path/to/custom.mlir")]
fn kernel_with_custom_ir<const S: [i32; 2]>(...) { ... }
```

The same compiler-stage dumps are also available with environment variables:

| Variable | Description | Default |
|---|---|---|
| `CUTILE_DUMP` | Dump compiler stages (`ast`, `resolved`, `typed`, `instantiated`, `ir`, `bytecode`, or `all`) | unset |
| `CUTILE_DUMP_FILTER` | Restrict dumps to matching function names or `module::function` paths | unset |

## Errors and Crashes

Most cuTile Rust errors are caught before a kernel runs:

| Error | Cause | Fix |
|---|---|---|
| Shape mismatch | Incompatible tile shapes | Align shapes or use `reshape` / `broadcast` |
| Element type mismatch | Different element types in one operation | Add explicit `convert_tile()` |
| Invalid reduction axis | Axis outside the tile rank | Use an axis in `0..rank` |
| Unsupported MMA shape or dtype | No lowering for that combination | Use a supported shape and element type |
| Missing entry | Function is not marked with `#[cutile::entry()]` | Add the entry attribute |

Runtime errors usually come from out-of-bounds accesses, toolkit issues, or invalid raw-pointer usage:

| Error | Cause | Fix |
|---|---|---|
| CUDA error: no kernel image | Wrong GPU architecture or stale cubin | Clear cache, rebuild, verify target SM |
| Failed to load kernel | CUDA toolkit or driver issue | Check `nvidia-smi` and toolkit version |
| Out of memory | Tensor allocation or JIT memory pressure | Reduce allocation size or specialization count |
| Shape mismatch at runtime | Tensor size incompatible with partition | Ensure expected divisibility or bounds handling |

CPU segfaults usually mean the failure happened in host-side FFI, JIT compilation, or raw-pointer lifetime management rather than inside ordinary safe tile code. Get a backtrace first:

```bash
RUST_BACKTRACE=1 cargo run
RUST_BACKTRACE=full cargo run

gdb --args ./target/debug/my_program
(gdb) run
(gdb) bt
```

Check the CUDA driver, CUDA Toolkit path, raw pointer lifetimes, spawned task lifetimes, and host memory use during first-launch compilation.

## Profiling

Use Nsight Compute for individual kernels:

```bash
ncu --target-processes all ./my_cutile_program
ncu --set full -o profile_report ./my_cutile_program
```

Watch memory throughput, compute throughput, occupancy, register spills, and stall reasons.

Use Nsight Systems for CPU/GPU scheduling:

```bash
nsys profile ./my_cutile_program
nsys-ui report.nsys-rep
```

Look for launch gaps, unnecessary synchronization, memory transfer overlap, and whether independent kernels actually overlap on separate streams.

## Debugging Checklist

- Shapes match the operation and launch partition.
- Tensor sizes are compatible with the partition shape.
- Element types match or are explicitly converted.
- Small inputs match a CPU reference.
- Numerically sensitive code handles overflow and underflow.
- Raw pointers outlive all GPU work that uses them.
- `print_ir` shows the expected Tile IR operations.
- Profiles are captured after correctness checks pass.

---

Review [Performance](performance.md) for optimization strategies or [Interoperability](interoperability.md) for custom CUDA kernels.
