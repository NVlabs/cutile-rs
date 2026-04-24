# Writing Computations

cuTile Rust provides a rich set of operations that work on tiles and leverage GPU parallelism. The [DSL API reference](../reference/dsl-api.md) has the complete catalog with full signatures; this page focuses on how these building blocks compose into complete algorithms.

---

## Operations at a Glance

Everything you express inside a kernel boils down to a small set of operation categories:

| Category | Representative operations | Reference |
|---|---|---|
| Load and store | `load_tile_mut`, `load_tile_like_2d`, `Partition::load`, `Tensor::store` | [Memory: Load and Store](../reference/dsl-api.md#memory-load-and-store) |
| Arithmetic | `+`, `-`, `*`, `/`, `fma`, `true_div` | [Arithmetic (Element-wise)](../reference/dsl-api.md#arithmetic-element-wise) |
| Math | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` | [Math (Floating-Point)](../reference/dsl-api.md#math-floating-point) |
| Reduction / scan | `reduce_max`, `reduce_sum`, `reduce_min`, `reduce_prod`, `scan` | [Reduction and Scan](../reference/dsl-api.md#reduction-and-scan) |
| Matrix multiply | `mma`, `permute` | [Matrix Multiply](../reference/dsl-api.md#matrix-multiply) |
| Shape manipulation | `reshape`, `broadcast`, `const_shape!` | [Shape Manipulation](../reference/dsl-api.md#shape-manipulation) |
| Comparison | `gt_tile`, `ge_tile`, `lt_tile`, `le_tile`, `eq_tile`, `select` | [Comparison](../reference/dsl-api.md#comparison) |
| Creation | `constant`, `iota`, `convert_tile` | [Creation](../reference/dsl-api.md#creation) |

Arithmetic operators apply element-wise to tiles of compatible shapes (broadcasting fills mismatched dimensions where one side is 1). Scalars can be combined with tiles directly or promoted to a tile with `broadcast`. Math functions, reductions, and shape operations compose freely with arithmetic to express most numerical algorithms.

---

## Composition Patterns

The real value of the tile model shows up when you combine operations into full algorithms. Three representative patterns.

### Scale and shift

Multiply a tile by a scale and add a bias — elementwise combined with broadcast:

```rust
fn scale_and_shift<const S: [i32; 2]>(
    x: Tile<f32, S>, scale: f32, shift: f32
) -> Tile<f32, S> {
    let s = scale.broadcast(x.shape());
    let b = shift.broadcast(x.shape());
    x * s + b
}
```

### Numerically stable softmax

Reductions combine with broadcasting to produce normalization. Subtracting the per-row max before `exp` prevents overflow on large inputs:

```rust
fn softmax<const BM: i32, const BN: i32>(
    x: Tile<f32, { [BM, BN] }>
) -> Tile<f32, { [BM, BN] }> {
    let max: Tile<f32, { [BM, BN] }> = reduce_max(x, 1i32)
        .reshape(const_shape![BM, 1])
        .broadcast(const_shape![BM, BN]);
    let stable = x - max;

    let exp_x = exp(stable);
    let sum: Tile<f32, { [BM, BN] }> = reduce_sum(exp_x, 1)
        .reshape(const_shape![BM, 1])
        .broadcast(const_shape![BM, BN]);

    true_div(exp_x, sum)
}
```

The `reshape(const_shape![BM, 1])` converts the reduced `[BM]` back to a column vector, then `broadcast` expands it across the column dimension to match the original `[BM, BN]` shape.

![Reduction operations along axis 0 (columns) and axis 1 (rows)](../_static/images/reduction-axes.svg)

### Tiled matrix multiply

Repeated `mma` calls accumulate into a destination tile across a K loop. Each iteration loads a `[BM, BK]` tile from `x` and a `[BK, BN]` tile from `y`, multiplies them, and accumulates into `acc`. The accumulator lives in registers throughout the loop:

```rust
fn tiled_gemm<E: ElementType, const BM: i32, const BN: i32, const BK: i32, const K: i32>(
    z: &mut Tensor<E, { [BM, BN] }>,
    x: &Tensor<E, { [-1, K] }>,
    y: &Tensor<E, { [K, -1] }>,
) {
    let part_x = x.partition(const_shape![BM, BK]);
    let part_y = y.partition(const_shape![BK, BN]);
    let pid: (i32, i32, i32) = get_tile_block_id();

    // Accumulator is `f32` regardless of input element type `E` — see
    // "Numerical Choices" below for why this matters when `E` is `f16`/`bf16`.
    let mut acc = constant(0.0f32, const_shape![BM, BN]);
    for i in 0i32..(K / BK) {
        let tile_x = part_x.load([pid.0, i]);
        let tile_y = part_y.load([i, pid.1]);
        acc = mma(tile_x, tile_y, acc);
    }

    z.store(acc);
}
```

---

## Numerical Choices

A handful of small but important habits for getting correct results with
floating-point tiles. None of these are cuTile-specific — they apply to
GPU numerics broadly — but they show up often enough that the patterns
are worth calling out.

### Accumulate in wider precision than the inputs

For `f16`/`bf16` inputs, accumulate into an `f32` tile. The tiled matmul
above is the canonical example: `E` can be `f16`/`bf16`, but `acc` is
always `f32`. `mma` is designed for this — its hardware path reads
16-bit operands and accumulates into a 32-bit register — so this
configuration is also the fastest on recent GPUs.

The same rule applies to manual reductions. Summing a large `f16` tile
directly loses bits; convert first:

```rust
let x_f16: Tile<f16, S> = load_tile_mut(input);
let x_f32: Tile<f32, S> = convert_tile(x_f16);
let sum: f32 = reduce_sum(x_f32, 0);
```

`reduce_sum` over `f16` compiles, but repeated `f16 + f16` across a
long reduction accumulates rounding error linearly in the tile size.
Converting to `f32` first bounds the error to the precision of the
converted inputs.

### `f16` vs `bf16`

Both are 16-bit, but the bit budgets differ:

| Type | Exponent bits | Mantissa bits | Range |
|---|---|---|---|
| `f16` | 5 | 10 | ±65504 (overflows at moderately large magnitudes) |
| `bf16` | 8 | 7 | Same dynamic range as `f32` |

- Prefer `bf16` for values that can span many orders of magnitude —
  gradients, intermediate activations before normalization, anything
  that might exceed ~65k.
- Prefer `f16` for values already within a bounded range — normalized
  activations, probabilities, quantized weights — where the extra
  mantissa bit helps.

### Subtract the max before `exp` (softmax)

Large positive inputs overflow `exp` to `+inf` even in `f32`
(`exp(89) ≈ 4.5e38`, the f32 max). Subtracting the per-row max first
keeps every `exp` argument `≤ 0` and every result in `[0, 1]`. The
[numerically stable softmax](#numerically-stable-softmax) pattern above
shows this.

### Prefer `fma(a, b, c)` to `a * b + c`

`a * b + c` rounds twice — once for the product, once for the sum.
`fma` produces a single correctly-rounded result in one step, giving
you roughly an extra bit of accuracy per call. Use it in inner
accumulation loops and anywhere small residuals matter.

### Rounding modes

IEEE float operations (`addf`, `subf`, `mulf`, `divf`, `sqrt`, `fma`,
…) take an explicit rounding-mode type parameter. `rounding::NearestEven`
is the right choice for essentially every numerical algorithm and is
what IEEE 754 specifies as the default; the directed modes
(`rounding::Zero`, `rounding::PositiveInf`, `rounding::NegativeInf`)
exist for numerical-methods work that needs to maintain interval
bounds.

---

## Debugging Computations

`cuda_tile_print!` prints from within a kernel:

```rust
cuda_tile_print!("Value at tile ({}, {}): {}\n",
    pid.0, pid.1, some_value);
```

:::{warning}
GPU printing is slow and serializes execution. Use it only for debugging small grids.
:::

For more techniques — type conversion sanity checks, assertions, IR dumping, and profiler integration — see [Debugging and Profiling](debugging.md).

---

Continue to [The Execution Model](execution-model.md) to see how these operations map onto the hardware. For the full catalog of operations with signatures, see the [DSL API](../reference/dsl-api.md).
