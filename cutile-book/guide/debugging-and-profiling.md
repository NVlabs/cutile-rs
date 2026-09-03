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
    let pid0 = program_id(0);
    let pid1 = program_id(1);
    let tile = x.load_like(z);

    cuda_tile_print!("Program ({}, {}): loaded tile\n", pid0, pid1);
    z.store(tile);
}
```

GPU printing is slow and serializes tile block execution. Use it for small grids and remove it before measuring performance.

`cuda_tile_assert!` checks conditions inside a kernel:

```rust
let n: i32 = x.shape()[0];
cuda_tile_assert!(n > 0, "expected a non-empty input");
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

`print_ir = true` prints the generated entry point wrapper and the Tile IR text during JIT compilation:

```rust
#[cutile::entry(print_ir = true)]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

`dump_mlir_dir` writes the compiled Tile IR text to files:

```rust
#[cutile::entry(dump_mlir_dir = "/tmp/cutile-ir")]
fn debug_ir_kernel<const S: [i32; 2]>(...) { ... }
```

Module-level dumps are also available with environment variables. They are
written to stderr once per compiled module:

| Variable | Description | Default |
|---|---|---|
| `CUTILE_DUMP` | Comma-separated stages to dump: `ir` (the Tile IR module text) and `bytecode` (alias `bc`; the encoded bytecode decoded back to text), or `all`. The `ast`, `resolved`, `typed`, and `instantiated` names are accepted but no code path emits them today | unset |
| `CUTILE_DUMP_FILTER` | Comma-separated `module::function` paths; the dumps are per module, so only the `module` part is matched. Bare function names do not exclude any module | unset |

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

## Debug Builds and Sanitizers

`CompileOptions` selects debugging and instrumentation modes per launch. Each option is part of the JIT cache key, so a debug build and a release build never share a compiled kernel:

```rust
use cutile::tile_kernel::CompileOptions;

// cuda-gdb: debug information, no optimization.
my_kernel(args).compile_options(CompileOptions::new().device_debug(true)).sync()?;

// Profiler correlation: line-number information only, full optimization.
my_kernel(args).compile_options(CompileOptions::new().lineinfo(true)).sync()?;

// Compute Sanitizer: memory-access instrumentation.
my_kernel(args).compile_options(CompileOptions::new().sanitize_memcheck(true)).sync()?;
```

What each option does:

- `device_debug(true)` passes `--device-debug` to the device compiler and implies optimization level 0 (set `opt_level` explicitly to override). The frontend also stops hoisting bounds checks out of loops, so every check that runs on the device sits at the source line that wrote it. Checks the compiler proved impossible, and checks it moved to launch time on the host, are unaffected — they never reach device code in any mode.
- `lineinfo(true)` passes `--lineinfo`: source-line correlation for Nsight Compute and Nsight Systems without changing code generation. This is the option for profiling optimized kernels.
- `sanitize_memcheck(true)` passes `--sanitize=memcheck` for `compute-sanitizer --tool memcheck`.
- `opt_level(n)` selects `--opt-level` directly; the default is 3.

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
