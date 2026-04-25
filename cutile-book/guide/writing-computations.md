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
