# DSL API

> **Status**: This API is under active development. Expect changes.

The cuTile Rust DSL lets you write GPU kernel code in Rust syntax that compiles to [CUDA Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/). Inside a `#[cutile::module]` block, you write Rust-like code using the types and operations documented here. The compiler translates this code into MLIR, which is then compiled to PTX/SASS for execution on NVIDIA GPUs.

All DSL types and functions are available via `use cutile::core::*;` inside a module block.

---

## Functions

### Modules and Entry Points

GPU kernels are defined inside `#[cutile::module]` blocks. Each function marked with `#[cutile::entry()]` becomes a launchable kernel.

```rust
#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn add<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,       // mutable: partitioned output
        a: &Tensor<f32, { [-1] }>,         // immutable: read-only input
        b: &Tensor<f32, { [-1] }>,         // immutable: read-only input
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile_a: Tile<f32, S> = a.load_tile(const_shape!(S), [pid.0]);
        let tile_b: Tile<f32, S> = b.load_tile(const_shape!(S), [pid.0]);
        output.store(tile_a + tile_b);
    }
}
```

**Entry point parameters:**

| Parameter type | Description | Host-side type |
|---|---|---|
| `&mut Tensor<E, S>` | Mutable output (must be first) | `Partition<Tensor<E>>` or `Partition<&mut Tensor<E>>` |
| `&Tensor<E, S>` | Immutable input | `&Tensor<E>`, `Arc<Tensor<E>>`, `Tensor<E>`, or `&TensorView<E>` |
| `f32`, `i32`, etc. | Scalar value | Same type, passed by value |
| `*mut T`, `*const T` | Raw device pointer | `DevicePointer<T>` |

**Convention:** The mutable output tensor is always the first parameter.

**Entry attributes:**

| Attribute | Type | Description |
|---|---|---|
| `print_ir = true` | bool | Print three stages at JIT compile time: (1) the generated entry point wrapper (Rust), (2) the original kernel function, (3) the compiled Tile IR MLIR |
| `unchecked_accesses` | bool | Disable partition bounds checks. Requires `unsafe fn`. Removes runtime assertions on partition index ranges |
| `optimization_hints = expr` | expr | Pass an `OptimizationHints` expression to the compiler for target-specific optimization |
| `dump_mlir_dir = "path"` | string | Write the compiled MLIR to a file in the specified directory |

```rust
#[cutile::entry(print_ir = true)]
fn debug_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }

#[cutile::entry(unchecked_accesses)]
unsafe fn fast_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }

#[cutile::entry(dump_mlir_dir = "/tmp/mlir")]
fn traced_kernel<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }
```

**Module attributes:**

| Attribute | Type | Description |
|---|---|---|
| `core` | bool | Internal: marks the core module (`_core.rs`) that provides built-in DSL functions |
| `tile_rust_crate = true` | bool | Internal: used when defining kernels inside the cutile crate itself (changes import paths from `cutile::` to `crate::`) |

```rust
#[cutile::module]                          // standard user module
mod my_kernels { ... }

#[crate::module(tile_rust_crate = true)]   // internal to cutile crate
pub mod creation { ... }
```

### Entry Points vs Device Functions

Functions marked with `#[cutile::entry()]` are compiled as GPU kernel entry points — they can be launched from the host with a grid configuration.

Unmarked functions inside a module are **device functions** — they are inlined at the call site during compilation. They cannot be launched directly but can be called from entry points or other device functions.

```rust
#[cutile::module]
mod my_kernels {
    use cutile::core::*;

    // Device function: inlined into callers
    fn relu<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        select(gt_tile(x, zero), x, zero)
    }

    #[cutile::entry()]
    fn apply_relu<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, S> = input.load_tile(const_shape!(S), [pid.0]);
        output.store(relu(tile));  // device function call, inlined
    }
}
```

### Inter-Module Device Function Calls

Device functions defined in one `#[cutile::module]` can be called from
entry points in another module. When Module B does `use crate::activations::*`,
two things happen:

1. **Rust's type checker** resolves the function signatures at compile time.
2. **The macro** records Module A as a dependency of Module B, including
   Module A's AST in the collected module list for JIT compilation.

At JIT time, the compiler sees all dependent module ASTs and inlines
device functions from any of them into the entry point.

Each module is identified by its fully-qualified path (via `module_path!()`),
so shared dependencies like `cutile::core` are automatically deduplicated
even when imported by multiple modules.

```rust
/// Module A: reusable activation device functions (no entry points).
#[cutile::module]
mod activations {
    use cutile::core::*;

    pub fn relu<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        let zero: Tile<f32, S> = constant(0.0f32, x.shape());
        max_tile(x, zero)
    }

    pub fn square<const S: [i32; 1]>(x: Tile<f32, S>) -> Tile<f32, S> {
        x * x
    }
}

/// Module B: kernels that call device functions from Module A.
#[cutile::module]
mod my_kernels {
    use cutile::core::*;
    use crate::activations::{relu, square};  // import from Module A

    #[cutile::entry()]
    fn apply_relu_square<const S: [i32; 1]>(
        output: &mut Tensor<f32, S>,
        input: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: Tile<f32, S> = input.load_tile(const_shape!(S), [pid.0]);
        let activated: Tile<f32, S> = relu(tile);   // inlined from Module A
        output.store(square(activated));             // inlined from Module A
    }
}
```

See `cutile-examples/examples/inter_module.rs` for a runnable version.

### `cutile::core` internals

`cutile::core` (defined in `_core.rs`) is itself a `#[cutile::module]`
containing the built-in DSL operations. When you write `use cutile::core::*;`,
the macro records it as a dependency and includes its AST. The JIT compiler
then resolves calls to `constant`, `iota`, `mma`, `reduce_sum`, etc. from
core's AST and inlines them into your entry point — exactly the same
mechanism as user-defined inter-module calls.

---

## Types

### Tile

`Tile<E: ElementType, const S: [i32; N]>` — A multi-dimensional array of GPU values. This is the fundamental compute type. Tiles are register-resident and processed in parallel by all threads in a warp/block.

```rust
// Shapes are const generics
let a: Tile<f32, { [128] }>;          // 1D: 128 f32 elements
let b: Tile<f32, { [16, 16] }>;       // 2D: 16x16 matrix
let c: Tile<i32, { [4, 8, 2] }>;      // 3D

// Scalar tile (rank 0)
let s: Tile<f32, { [] }>;
```

**Methods:**
- `tile.shape()` — Returns the tile's `Shape<S>`
- `tile.broadcast(shape)` — Broadcast to a larger shape
- `tile.reshape(shape)` — Reshape (must preserve element count)

**Arithmetic operators:** `+`, `-`, `*`, `/` are overloaded for element-wise operations between tiles of the same shape and type.

```rust
let z: Tile<f32, { [128] }> = a + b;     // element-wise add
let w: Tile<f32, { [128] }> = a * b - c; // chained arithmetic
```

### Tensor

`Tensor<E: ElementType, const S: [i32; N]>` — A device-side tensor view. Represents memory that lives in GPU global memory. Unlike `Tile`, tensors must be loaded/stored explicitly.

Static dimensions are known at compile time; dynamic dimensions are marked with `-1`:

```rust
fn kernel(
    output: &mut Tensor<f32, { [128, 128] }>,  // static: 128x128
    input: &Tensor<f32, { [-1, -1] }>,          // dynamic: shape provided at runtime
)
```

**Methods:**
- `tensor.shape()` — Returns the tensor's `Shape<S>`
- `tensor.load()` — Load the entire tile (only for `&mut Tensor`, loads output tile)
- `tensor.store(tile)` — Store a tile to the tensor
- `tensor.load_tile(shape, [indices])` — Load a tile at a specific partition index
- `tensor.partition(shape)` — Create a `Partition` view for block-indexed loading
- `tensor.partition_permuted(shape, dim_map)` — Create a permuted `Partition` view
- `unsafe tensor.partition_mut(shape)` — Create a mutable `PartitionMut` view

### Partition / PartitionMut

`Partition<'a, E, const D: [i32; N]>` — A read-only partitioned view of a tensor, dividing it into tiles indexed by block ID.

`PartitionMut<'a, E, const D: [i32; N]>` — A mutable partitioned view.

```rust
let part: Partition<f32, { [128, 128] }> = input.partition(const_shape![128, 128]);
let tile: Tile<f32, { [128, 128] }> = part.load([pid.0, pid.1]);
```

### Shape / Array

`Shape<const D: [i32; N]>` — A compile-time shape descriptor. Created with `const_shape!`:

```rust
let shape: Shape<{ [128, 64] }> = const_shape![128, 64];
let shape: Shape<S> = tensor.shape();
```

`Array<const D: [i32; N]>` — A compile-time array (used for strides and indices).

#### Const Generic Array (CGA) Syntax

Shapes in the DSL use Rust's const generic arrays (`const S: [i32; N]`). There are two ways to specify them:

**1. Explicit values** — when dimensions are known literals:

```rust
fn kernel(
    output: &mut Tensor<f32, { [128, 64] }>,   // fixed 128x64
    input: &Tensor<f32, { [-1, -1] }>,          // dynamic (runtime) shape
)
```

**2. Const generic parameters** — when dimensions are specified at launch time:

```rust
fn gemm<const BM: i32, const BN: i32, const BK: i32>(
    output: &mut Tensor<f32, { [BM, BN] }>,    // shape from generics
    a: &Tensor<f32, { [-1, -1] }>,
)
```

**3. Const generic array (CGA)** — when the entire shape is a single generic parameter:

```rust
fn add<const S: [i32; 1]>(                      // S is the whole shape
    output: &mut Tensor<f32, S>,
    input: &Tensor<f32, { [-1] }>,
)
```

The CGA form (`const S: [i32; N]`) is concise but has a limitation: **the array length `N` must be fixed at definition time**. You cannot write `const S: [i32; N]` where `N` is itself generic — the rank must be a literal. This means you cannot write a single kernel that works for both 1D and 2D tensors:

```rust
// NOT supported: generic rank
fn add<const N: usize, const S: [i32; N]>(output: &mut Tensor<f32, S>) { ... }

// Instead, write separate kernels per rank:
fn add_1d<const S: [i32; 1]>(output: &mut Tensor<f32, S>) { ... }
fn add_2d<const S: [i32; 2]>(output: &mut Tensor<f32, S>) { ... }
```

**When to use which:**

| Pattern | Use when |
|---|---|
| `{ [128, 64] }` | Dimensions are fixed literals |
| `{ [BM, BN] }` | Each dimension is an independent generic (e.g., GEMM tile sizes) |
| `const S: [i32; 1]` | The whole shape is generic but rank is known |
| `{ [-1] }` | Dimension is dynamic (provided at runtime by the host) |

Dynamic dimensions (`-1`) are resolved at runtime from the tensor's actual shape. Static dimensions are baked into the compiled kernel. Mixing is allowed: `{ [-1, 128] }` means "dynamic first axis, static second axis."

### PointerTile

`PointerTile<P: Pointer, const D: [i32; N]>` — A tile of device pointers. Used for scatter/gather and atomic operations.

```rust
let ptr: PointerTile<*mut f32, { [] }> = pointer_to_tile(raw_ptr);
let offset_ptrs: PointerTile<*mut f32, { [128] }> = ptr.broadcast(const_shape![128]).offset_tile(offsets);
```

### Token

`Token` — An ordering token for memory operations. Tokens enforce ordering between loads and stores without requiring full barriers.

```rust
let token: Token = new_token_unordered();
let joined: Token = join_tokens(&[token_a, token_b]);
```

### ElementType

Trait implemented by all scalar types usable in tiles:

| Type | Description |
|------|-------------|
| `f16` | IEEE 754 half-precision |
| `bf16` | Brain floating-point |
| `f32` | Single-precision |
| `f64` | Double-precision |
| `i8`, `i16`, `i32`, `i64` | Signed integers |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `bool` | Boolean |
| `tf32` | TensorFloat-32 (NVIDIA) |
| `f8e4m3fn`, `f8e5m2` | FP8 (storage only; convert to `f16`/`f32` for compute) |


---

## Operations

### Tile IR Operation Mapping

The functions in this reference are the Rust DSL surface for
[CUDA Tile IR operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html).
Most low-level operations in `cutile::core` map directly to a `cuda_tile.*`
operation. The Rust name is usually either the Tile IR operation name without
the `cuda_tile.` prefix, or a small wrapper with a Rust-oriented name and type
signature.

Examples:

| Rust DSL surface | Tile IR operation family |
|---|---|
| `constant`, `iota`, `broadcast`, `reshape`, `permute`, `cat`, `extract`, `select` | Core tile ops such as `cuda_tile.constant`, `cuda_tile.iota`, `cuda_tile.broadcast`, `cuda_tile.reshape`, `cuda_tile.permute`, `cuda_tile.cat`, `cuda_tile.extract`, `cuda_tile.select` |
| `addf`, `mulf`, `fma`, `exp`, `sqrt`, `maxf`, `cmpf` and operator overloads | Floating-point ops such as `cuda_tile.addf`, `cuda_tile.mulf`, `cuda_tile.fma`, `cuda_tile.exp`, `cuda_tile.sqrt`, `cuda_tile.maxf`, `cuda_tile.cmpf` |
| `addi`, `muli`, `shli`, `shri`, `andi`, `ori`, `xori`, `cmpi` and operator overloads | Integer and bitwise ops such as `cuda_tile.addi`, `cuda_tile.muli`, `cuda_tile.shli`, `cuda_tile.andi`, `cuda_tile.cmpi` |
| `convert_tile`, `bitcast`, `exti`, `trunci`, `int_to_ptr`, `ptr_to_int`, `ptr_to_ptr` | Conversion ops such as `cuda_tile.bitcast`, `cuda_tile.exti`, `cuda_tile.trunci`, `cuda_tile.int_to_ptr`, `cuda_tile.ptr_to_int`, `cuda_tile.ptr_to_ptr` |
| `load_ptr_tko`, `store_ptr_tko`, `atomic_rmw_tko`, `atomic_cas_tko`, `new_token_unordered`, `join_tokens` | Memory, atomic, and token ops such as `cuda_tile.load_ptr_tko`, `cuda_tile.store_ptr_tko`, `cuda_tile.atomic_rmw_tko`, `cuda_tile.atomic_cas_tko`, `cuda_tile.make_token`, `cuda_tile.join_tokens` |
| `load_tile_like`, `tensor.load_tile`, `tensor.store`, partition and tensor view helpers | View ops such as `cuda_tile.load_view_tko`, `cuda_tile.store_view_tko`, `cuda_tile.make_tensor_view`, `cuda_tile.make_partition_view`, plus compiler-generated shape/stride plumbing |
| `assume_div_by`, `assume_bounds_*`, `cuda_tile_print!`, `cuda_tile_assert!` | Compiler and debugging ops such as `cuda_tile.assume`, `cuda_tile.print_tko`, and `cuda_tile.assert` |

Some Tile IR operations are intentionally compiler-owned rather than public DSL
functions. `cuda_tile.module`, `cuda_tile.entry`, `cuda_tile.return`, and
control-flow operations are generated from Rust modules, entry attributes,
`return`, `if`, `for`, `loop`, `break`, and `continue` syntax. This keeps user
code Rust-shaped while still lowering to the corresponding Tile IR operations.

Tile IR attributes such as memory ordering, memory scope, comparison predicate,
rounding mode, overflow behavior, and flush-to-zero mode are represented as
Rust marker types and traits where possible. This lets the Rust type checker
reject unsupported attribute combinations before the JIT compiler lowers the
operation.

### Memory: Load and Store

| Function | Signature | Description |
|---|---|---|
| `tensor.load()` | `&mut Tensor<E, S> -> Tile<E, S>` | Load the output tile |
| `tensor.store(tile)` | `(&mut Tensor<E, S>, Tile<E, S>)` | Store a tile to the tensor |
| `tensor.load_tile(shape, idx)` | `(&Tensor<E, S>, Shape<R>, [i32; N]) -> Tile<E, R>` | Load at a partition index |
| `load_tile_mut(tensor)` | `(&mut Tensor<E, S>) -> Tile<E, S>` | Load output tile (convenience) |
| `load_tile_like(src, dst)` | `(&Tensor, &Tensor) -> Tile` | Load from src at dst's tile-block position (rank 1-3) |

```rust
// Pattern 1: Direct load/store on mutable tensor
let tile: Tile<f32, { [128] }> = load_tile_mut(output);
output.store(tile * scale_tile);

// Pattern 2: Load at block position
let pid: (i32, i32, i32) = get_tile_block_id();
let tile: Tile<f32, { [128] }> = input.load_tile(const_shape![128], [pid.0]);

// Pattern 3: Load-like (positional)
let tile_x: Tile<f32, { [16, 16] }> = load_tile_like(x, output);
```

### Grid and Block

| Function | Signature | Description |
|---|---|---|
| `get_tile_block_id()` | `() -> (i32, i32, i32)` | Current thread block's (x, y, z) index in the grid |
| `get_num_tile_blocks()` | `() -> (i32, i32, i32)` | Total (x, y, z) dimensions of the grid |

```rust
let pid: (i32, i32, i32) = get_tile_block_id();
let grid: (i32, i32, i32) = get_num_tile_blocks();
// Grid-stride loop
for i in (pid.0..total).step_by(grid.0 as usize) { ... }
```

### Arithmetic (Element-wise)

In addition to operator overloading (`+`, `-`, `*`, `/`), these explicit functions are available:

| Function | Signature | Description |
|---|---|---|
| `absi(x)` | `Tile<E, S> -> Tile<E, S>` | Absolute value (integer) |
| `absf(x)` | `Tile<E, S> -> Tile<E, S>` | Absolute value (float) |
| `negi(x)` | `Tile<E, S> -> Tile<E, S>` | Negation (integer) |
| `negf(x)` | `Tile<E, S> -> Tile<E, S>` | Negation (float) |
| `fma(a, b, c)` | `(Tile<E, S>, Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Fused multiply-add: `a * b + c` |
| `fma_ftz(a, b, c)` | `(Tile<E, S>, Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Fused multiply-add (flush-to-zero) |
| `pow(base, exp)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Power |
| `ceil_div(a, b)` | `(E, E) -> E` | Ceiling division (scalar) |
| `true_div(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | True (floating-point) division |
| `mulhii(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | High bits of integer multiply |

```rust
// Absolute value
let abs_x: Tile<f32, S> = absf(x);
let abs_i: Tile<i32, S> = absi(int_tile);

// Fused multiply-add: a * b + c (single instruction, no intermediate rounding)
let result: Tile<f32, S> = fma(a, b, c);

// Power
let squared: Tile<f32, S> = pow(x, broadcast_scalar(2.0f32, x.shape()));

// Negation
let neg_x: Tile<f32, S> = negf(x);
let neg_i: Tile<i32, S> = negi(int_tile);
```

### Math (Floating-Point)

| Function | Signature | Description |
|---|---|---|
| `exp(x)` | `Tile<E, S> -> Tile<E, S>` | e^x |
| `exp2(x, ftz::Disabled)` | `Tile<E, S> -> Tile<E, S>` | 2^x |
| `exp2_ftz(x)` | `Tile<E, S> -> Tile<E, S>` | 2^x with flush-to-zero |
| `log(x)` | `Tile<E, S> -> Tile<E, S>` | Natural logarithm |
| `log2(x)` | `Tile<E, S> -> Tile<E, S>` | Base-2 logarithm |
| `sqrt(x)` | `Tile<E, S> -> Tile<E, S>` | Square root |
| `sqrt_ftz(x)` | `Tile<E, S> -> Tile<E, S>` | Square root with flush-to-zero |
| `rsqrt(x)` | `Tile<E, S> -> Tile<E, S>` | Reciprocal square root (1/sqrt(x)) |
| `rsqrt_ftz(x)` | `Tile<E, S> -> Tile<E, S>` | Reciprocal square root with flush-to-zero |
| `sin(x)` | `Tile<E, S> -> Tile<E, S>` | Sine |
| `cos(x)` | `Tile<E, S> -> Tile<E, S>` | Cosine |
| `tan(x)` | `Tile<E, S> -> Tile<E, S>` | Tangent |
| `sinh(x)` | `Tile<E, S> -> Tile<E, S>` | Hyperbolic sine |
| `cosh(x)` | `Tile<E, S> -> Tile<E, S>` | Hyperbolic cosine |
| `tanh(x)` | `Tile<E, S> -> Tile<E, S>` | Hyperbolic tangent |
| `ceil(x)` | `Tile<E, S> -> Tile<E, S>` | Ceiling |
| `floor(x)` | `Tile<E, S> -> Tile<E, S>` | Floor |
| `maxf(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float max |
| `minf(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float min |
| `maxf_ftz(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float max (flush-to-zero) |
| `minf_ftz(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float min (flush-to-zero) |
| `addf_ftz(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float add (flush-to-zero) |
| `subf_ftz(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float sub (flush-to-zero) |
| `mulf_ftz(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float mul (flush-to-zero) |
| `divf_ftz(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Float div (flush-to-zero) |

```rust
// Softmax numerics: subtract max, exponentiate
let max_val: Tile<f32, { [BM] }> = reduce_max(x, 1i32);
let shifted: Tile<f32, S> = x - max_val.reshape(const_shape![BM, 1]).broadcast(x.shape());
let softmax_exp: Tile<f32, S> = exp(shifted);

// RMS normalization
let sq: Tile<f32, S> = x * x;
let mean_sq: Tile<f32, { [BM] }> = reduce_sum(sq, 1i32);
let rms: Tile<f32, { [BM] }> = rsqrt(mean_sq + broadcast_scalar(1e-6f32, mean_sq.shape()));

// Activation functions
let gelu_approx: Tile<f32, S> = x * (constant(1.0f32, x.shape()) + tanh(x));
let swish: Tile<f32, S> = x / (constant(1.0f32, x.shape()) + exp(negf(x)));

// exp2 is faster than exp on GPU — convert: exp(x) = exp2(x * log2(e, ftz::Disabled))
let log2_e: f32 = 1.4426950408889634f32;
let fast_exp: Tile<f32, S> = exp2(x * broadcast_scalar(log2_e, x.shape()));

// Flush-to-zero variants: treat denormals as zero (faster on some hardware, f32 only)
let clamped: Tile<f32, S> = maxf_ftz(x, broadcast_scalar(0.0f32, x.shape()));
let sum: Tile<f32, S> = addf_ftz(a, b);
let product: Tile<f32, S> = mulf_ftz(a, b);
let fma_result: Tile<f32, S> = fma_ftz(a, b, c);
```

### Comparison

| Function | Signature | Description |
|---|---|---|
| `eq_tile(a, b)` | `-> Tile<bool, S>` | Equal |
| `ne_tile(a, b)` | `-> Tile<bool, S>` | Not equal |
| `gt_tile(a, b)` | `-> Tile<bool, S>` | Greater than |
| `ge_tile(a, b)` | `-> Tile<bool, S>` | Greater or equal |
| `lt_tile(a, b)` | `-> Tile<bool, S>` | Less than |
| `le_tile(a, b)` | `-> Tile<bool, S>` | Less or equal |
| `select(cond, a, b)` | `(Tile<bool, S>, Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Conditional select |
| `min(a, b)` / `max(a, b)` | Scalar min/max |
| `min_tile(a, b)` / `max_tile(a, b)` | Element-wise min/max |

```rust
let mask: Tile<bool, { [128] }> = lt_tile(indices, len_tile);
let result: Tile<f32, { [128] }> = select(mask, values, zeros);
```

### Creation

| Function | Signature | Description |
|---|---|---|
| `constant(value, shape)` | `(E, Shape<S>) -> Tile<E, S>` | Fill a tile with a constant |
| `iota(shape)` | `(Shape<S>) -> Tile<E, S>` | Sequential indices `[0, 1, 2, ...]` (1D only) |
| `broadcast_scalar(val, shape)` | `(E, Shape<S>) -> Tile<E, S>` | Broadcast a scalar to a tile shape |

```rust
let zeros: Tile<f32, { [128] }> = constant(0.0f32, const_shape![128]);
let indices: Tile<i32, { [64] }> = iota(const_shape![64]);  // [0, 1, 2, ..., 63]
let scale: Tile<f32, { [16, 16] }> = broadcast_scalar(2.0f32, const_shape![16, 16]);
```

### Shape Manipulation

| Function | Signature | Description |
|---|---|---|
| `tile.reshape(shape)` | `Tile<E, S> -> Tile<E, R>` | Reshape (preserves element count) |
| `tile.broadcast(shape)` | `Tile<E, S> -> Tile<E, R>` | Broadcast to a larger shape |
| `reshape(tile, shape)` | `(Tile<E, S>, Shape<R>) -> Tile<E, R>` | Free function reshape |
| `broadcast(tile, shape)` | `(Tile<E, S>, Shape<R>) -> Tile<E, R>` | Free function broadcast |
| `permute(tile, indices, shape)` | `(Tile<E, A>, Array<I>, Shape<R>) -> Tile<E, R>` | Transpose / permute dimensions |
| `cat(a, b, dim)` | `(Tile<E, SLhs>, Tile<E, SRhs>, i32) -> Tile<E, SOut>` | Concatenate along a dimension |
| `extract(tile, offsets, shape)` | `(Tile<E, SIn>, [i32; N], Shape<SOut>) -> Tile<E, SOut>` | Extract a sub-tile |
| `get_shape_dim(shape, dim_idx)` | `(Shape<S>, i32) -> i32` | Read one runtime dimension from a shape |

```rust
let row: Tile<f32, { [128] }> = iota(const_shape![128]);
let col: Tile<f32, { [128, 1] }> = row.reshape(const_shape![128, 1]);
let matrix: Tile<f32, { [128, 64] }> = col.broadcast(const_shape![128, 64]);
let n_cols: i32 = get_shape_dim(matrix.shape(), 1i32);
```

### Reduction and Scan

| Function | Signature | Description |
|---|---|---|
| `reduce_sum(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | Sum reduction along one dimension |
| `reduce_max(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | Max reduction |
| `reduce_min(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | Min reduction |
| `reduce_prod(tile, dim)` | `(Tile<E, S>, i32) -> Tile<E, R>` | Product reduction |
| `reduce(tile, dim, identity, f)` | `(Tile<E, S>, i32, E, Fn(E, E) -> E) -> Tile<E, R>` | Custom reduction |
| `scan_sum(tile, dim, reverse, identity)` | `(Tile<E, S>, i32, reverse::Mode, E) -> Tile<E, S>` | Prefix sum |
| `scan(tile, dim, reverse, identity, f)` | `(Tile<E, S>, i32, reverse::Mode, E, Fn(E, E) -> E) -> Tile<E, S>` | Custom prefix scan |

```rust
// Sum each row of a [128, 64] tile to [128] (reduce along axis 1)
let row_sums: Tile<f32, { [128] }> = reduce_sum(matrix, 1i32);

// Prefix sum along axis 0
let prefix: Tile<f32, { [128] }> = scan_sum(row, 0i32, reverse::Forward, 0.0f32);
```

### Matrix Multiply

| Function | Signature | Description |
|---|---|---|
| `mma(a, b, c)` | `(Tile<E, {[M,K]}>, Tile<E, {[K,N]}>, Tile<E, {[M,N]}>) -> Tile<E, {[M,N]}>` | Matrix multiply-accumulate |

Maps to hardware tensor cores when available.

```rust
let mut acc: Tile<f32, { [16, 16] }> = constant(0.0f32, const_shape![16, 16]);
for k in 0i32..(K/BK) {
    let a_tile: Tile<f32, { [16, 8] }> = a_part.load([pid.0, k]);
    let b_tile: Tile<f32, { [8, 16] }> = b_part.load([k, pid.1]);
    acc = mma(a_tile, b_tile, acc);
}
```

### Low-Level Memory Ops

These APIs are close to the Tile IR memory/view operations. Prefer the
high-level methods above (`tensor.load_tile`, `partition.load`,
`partition_mut.store`, `load_tile_like`, `tensor.store`) unless you are building
custom views, raw-pointer kernels, or compiler-facing helpers.

#### View construction and queries

View constructors create typed tensor or partition views from lower-level
metadata. Mutable view construction and raw tensor construction are `unsafe`
because the caller must preserve aliasing, layout, and lifetime invariants.

| Function | Signature | Description |
|---|---|---|
| `unsafe make_tensor_view(base, shape, strides, token)` | `(PointerTile<*mut E, {[]}>, Shape<D>, Array<C>, Token) -> Tensor<E, D>` | Build a tensor view from a base pointer |
| `make_partition_view(tensor, tile, padding, dim_map, token)` | `(&Tensor<E, S>, Shape<R>, padding::Mode, dim_map::Mode, Token) -> Partition<E, R>` | Build a read-only partition view |
| `unsafe make_partition_view_mut(tensor, tile, padding, token)` | `(&Tensor<E, S>, Shape<R>, padding::Mode, Token) -> PartitionMut<E, R>` | Build a mutable partition view |
| `get_tensor_shape(tensor)` | `&Tensor<E, S> -> [i32; N]` | Query a tensor view's runtime shape |
| `get_index_space_shape(partition)` | `&Partition<E, S> -> [i32; N]` | Query a partition's tile-grid shape |
| `get_tensor_token(tensor)` | `&Tensor<E, S> -> Token` | Read a tensor view's memory token |
| `set_tensor_token(tensor, token)` | `(&Tensor<E, S>, Token)` | Update a tensor view's memory token |
| `get_partition_token(partition)` | `&Partition<E, S> -> Token` | Read a read-only partition token |
| `get_partition_token_mut(partition)` | `&PartitionMut<E, S> -> Token` | Read a mutable partition token |
| `num_tiles(partition, axis)` | `(&Partition<E, S>, i32) -> i32` | Number of tiles along one partition axis |
| `unsafe load_tensor(ptrs, idx, shape, strides)` | `(&Tensor<i64, S>, [i32; N], Shape<R>, Array<C>) -> Tensor<T, R>` | Load a strided tensor view from an integer pointer tensor |

```rust
let shape = input.shape();
let token = get_tensor_token(input);
let part = make_partition_view(input, shape, padding::None, dim_map::Identity, token);
let tiles_m: i32 = num_tiles(&part, 0);
```

#### View loads and stores

These are the direct memory operations on partition views. They expose Tile IR
ordering, scope, latency, and TMA controls explicitly.

| Function | Signature | Description |
|---|---|---|
| `load_view_tko(view, index, ordering, scope, latency, tma)` | `(&Partition<E, S>, [i32; N], O, Sc, Option<i32>, T) -> Tile<E, S>` | Load a tile from a read-only partition |
| `unsafe load_view_tko_mut(view, index, ordering, scope, latency, tma)` | `(&PartitionMut<E, S>, [i32; N], O, Sc, Option<i32>, T) -> Tile<E, S>` | Load from a mutable partition; unsafe aliasing contract |
| `unsafe store_view_tko_mut(view, tile, index, ordering, scope, latency, tma)` | `(&mut PartitionMut<E, S>, Tile<E, S>, [i32; N], O, Sc, Option<i32>, T) -> Token` | Store a tile into a mutable partition |

```rust
let pid = get_tile_block_id();
let tile: Tile<f32, S> =
    load_view_tko(&part, [pid.0], ordering::Weak, scope::TileBlock, None, tma::Enabled);
```

#### Pointer-based loads and stores

| Function | Signature | Description |
|---|---|---|
| `load_ptr_tko(ptrs, ordering, Option<scope>, mask, fill, token, Latency<N>)` | `-> (Tile<E, S>, Token)` | Scatter-gather load via pointers |
| `store_ptr_tko(ptrs, values, ordering, Option<scope>, mask, token, Latency<N>)` | `-> Token` | Scatter-gather store via pointers |
| `pointer_to_tile(ptr)` | `P -> PointerTile<P, {[]}>` | Convert raw pointer to scalar pointer tile |
| `tile_to_pointer(ptile)` | `PointerTile<P, {[]}> -> P` | Convert back |
| `addptr(ptile, offset)` | Offset a pointer tile by a scalar |
| `addptr_tile(ptile, offsets)` | Offset a pointer tile by an index tile |
| `broadcast_ptr(ptile, shape)` | Broadcast a pointer tile to a larger shape |
| `reshape_ptr(ptile, shape)` | Reshape a pointer tile |

```rust
let base: PointerTile<*mut f32, { [] }> = pointer_to_tile(ptr);
let ptrs: PointerTile<*mut f32, { [128] }> = base.broadcast(const_shape![128]).offset_tile(offsets);
let (values, token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);
```

#### Atomics

| Function | Signature | Description |
|---|---|---|
| `atomic_rmw_tko(ptrs, vals, mode, ordering, scope, mask, hint)` | `-> (Tile<E, S>, Token)` | Atomic read-modify-write |
| `atomic_cas_tko(ptrs, cmp, new, ordering, scope, mask, hint)` | `-> (Tile<E, S>, Token)` | Atomic compare-and-swap |

**RMW modes:** `atomic::{Add, AddF, And, Or, Xor, Max, Min, Umax, Umin, Xchg}`

**Memory orderings:** `ordering::{Relaxed, Acquire, Release, AcqRel}` (atomics; load/store also accept `Weak`)

**Scopes:** `scope::{TileBlock, Device, System}`

```rust
atomic_rmw_tko(ptrs, increments, atomic::Add, ordering::Relaxed, scope::Device, None, None);
atomic_cas_tko(ptrs, expected, desired, ordering::AcqRel, scope::System, None, None);
```

#### Tokens

Tokens track ordering dependencies between memory operations. A token returned from a load guarantees that the load has completed before any operation that consumes that token. `join_tokens` merges multiple tokens into one, ensuring all joined operations complete before the result token is used.

This enables fine-grained ordering without full barriers: independent loads can execute in parallel, and a store only waits for the specific loads it depends on.

| Function | Signature | Description |
|---|---|---|
| `new_token_unordered()` | `() -> Token` | Create a fresh ordering token (no ordering guarantee) |
| `join_tokens(tokens)` | `(&[Token]) -> Token` | Join multiple tokens: result waits for all inputs |

```rust
// Thread tokens through a load → compute → store sequence:
let token: Token = new_token_unordered();

// Load returns a new token guaranteeing the load completed
let (data, load_token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(src_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);

// Compute on the loaded data
let result: Tile<f32, { [128] }> = data * data;

// Store uses the load token: waits for the load before writing
let store_token: Token =
    store_ptr_tko(dst_ptrs, result, ordering::Weak, None::<scope::TileBlock>, None, Some(load_token), Latency::<0>);
```

```rust
// Join tokens from two independent loads before a dependent store:
let (a_data, a_token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(a_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);
let (b_data, b_token): (Tile<f32, { [128] }>, Token) =
    load_ptr_tko(b_ptrs, ordering::Weak, None::<scope::TileBlock>, None, None, None, Latency::<0>);

// Both loads must complete before the store
let combined: Token = join_tokens(&[a_token, b_token]);
let result: Tile<f32, { [128] }> = a_data + b_data;
let _: Token =
    store_ptr_tko(out_ptrs, result, ordering::Weak, None::<scope::TileBlock>, None, Some(combined), Latency::<0>);
```

### Bitwise

| Function | Signature | Description |
|---|---|---|
| `andi(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Bitwise AND |
| `ori(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Bitwise OR |
| `xori(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Bitwise XOR |
| `shli(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Shift left |
| `shri(a, b)` | `(Tile<E, S>, Tile<E, S>) -> Tile<E, S>` | Shift right |

```rust
// Mask lower 8 bits
let mask: Tile<i32, S> = constant(0xFF, x.shape());
let low_byte: Tile<i32, S> = andi(x, mask);

// Shift left by 2 (multiply by 4)
let shift: Tile<i32, S> = constant(2, x.shape());
let shifted: Tile<i32, S> = shli(x, shift);

// Toggle bits with XOR
let toggled: Tile<i32, S> = xori(x, mask);
```

### Type Conversion

| Function | Signature | Description |
|---|---|---|
| `convert_tile(tile)` | `Tile<FROM, S> -> Tile<TO, S>` | Convert element type |
| `convert_scalar(x)` | `impl Scalar -> S` | Convert scalar type |
| `scalar_to_tile(x)` | `impl Scalar -> Tile<E, {[]}>` | Wrap a scalar in a rank-0 tile |
| `tile_to_scalar(tile)` | `Tile<E, {[]}> -> S` | Convert a rank-0 tile back to a scalar |
| `ftof(tile, rounding)` | `Tile<EIn, S> -> Tile<EOut, S>` | Float-to-float conversion with rounding mode |
| `ftoi(tile, rounding)` | `Tile<EIn, S> -> Tile<EOut, S>` | Float-to-integer conversion with rounding mode |
| `itof(tile, rounding)` | `Tile<EIn, S> -> Tile<EOut, S>` | Integer-to-float conversion with rounding mode |
| `bitcast(tile)` | `Tile<EIn, S> -> Tile<EOut, S>` | Reinterpret bits (no conversion) |
| `exti(tile)` | `Tile<EIn, S> -> Tile<EOut, S>` | Extend integer (sign/zero) |
| `trunci(tile, overflow)` | `Tile<EIn, S> -> Tile<EOut, S>` | Truncate integer with overflow mode |
| `int_to_ptr(tile)` | `Tile<SRC_T, S> -> PointerTile<*mut PTR_T, S>` | Integer to pointer |
| `ptr_to_int(ptrs)` | `PointerTile<*mut E, S> -> Tile<E, S>` | Pointer to integer tile |
| `ptr_to_ptr(ptrs)` | `PointerTile<*mut EIn, S> -> PointerTile<*mut EOut, S>` | Pointer cast |

```rust
// Float to int conversion
let indices: Tile<i32, { [128] }> = iota(const_shape![128]);
let float_indices: Tile<f32, { [128] }> = convert_tile(indices);

// Bitcast: reinterpret f32 bits as u32 (no value conversion)
let float_tile: Tile<f32, { [128] }> = constant(1.0f32, const_shape![128]);
let bits: Tile<u32, { [128] }> = bitcast(float_tile);  // 0x3F800000

// Integer extension and truncation
let small: Tile<i16, { [64] }> = constant(42i16, const_shape![64]);
let wide: Tile<i32, { [64] }> = exti(small);     // sign-extend i16 -> i32
let narrow: Tile<i16, { [64] }> = trunci(wide, overflow::None);
```

### Compiler Hints

| Function | Signature | Description |
|---|---|---|
| `assume_div_by::<_, N>(val)` | `T -> T` | Assert value is divisible by N |
| `assume_bounds_lower::<_, L>(val)` | `T -> T` | Assert value >= L |
| `assume_bounds_upper::<_, U>(val)` | `T -> T` | Assert value <= U |
| `assume_bounds::<_, L, U>(val)` | `T -> T` | Assert L <= value <= U |

These are `unsafe` — incorrect assumptions produce undefined behavior. They enable compiler optimizations like vectorized loads and simplified index arithmetic.

```rust
// Tell the compiler a dimension is a multiple of 16 (enables wider vector loads)
let dim: i32 = unsafe { assume_div_by::<_, 16>(dim) };

// Bound an index (enables range-based optimizations)
let idx: i32 = unsafe { assume_bounds::<_, 0, 1024>(idx) };

// Combine: non-negative and aligned
let stride: i32 = unsafe { assume_bounds_lower::<_, 0>(stride) };
let stride: i32 = unsafe { assume_div_by::<_, 4>(stride) };
```
### Debugging

| Macro | Description |
|---|---|
| `cuda_tile_print!(fmt, args...)` | Printf-style GPU print |
| `cuda_tile_assert!(cond, msg)` | GPU assertion |

```rust
let pid: (i32, i32, i32) = get_tile_block_id();
cuda_tile_print!("Block ({}, {}, {})\n", pid.0, pid.1, pid.2);

// Assert a condition — aborts the kernel if false
cuda_tile_assert!(len > 0, "Length must be positive");

// Print scalars for debugging (runs on every block, so output may interleave)
cuda_tile_print!("offset = {}\n", pid.0 * 128);
```
