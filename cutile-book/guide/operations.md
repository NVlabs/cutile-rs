---
orphan: true
---

# Operations and Intrinsics

cuTile Rust provides a rich set of operations for GPU computation. All operations work on tiles and leverage GPU parallelism.

## Loading and Storing

### Basic Load/Store

```rust
// Load entire output tile
let tile = load_tile_mut(tensor);

// Store result
tensor.store(tile);
```

### Load Like (Positional Loading)

Load from a dynamic tensor at the position matching another tile:

```rust
// Load from x at the same position as output tile z
let tile_x = load_tile_like_2d(x, z);
let tile_y = load_tile_like_2d(y, z);
```

This is the most common pattern for element-wise operations.

### Partitioned Loading

For explicit control over which tile to load:

```rust
let part = tensor.partition(const_shape![16, 16]);
let tile = part.load([row_idx, col_idx]);
```

---

## Elementwise Operations

Standard math operations work element-by-element on tiles:

### Arithmetic

```rust
let c = a + b;    // Addition
let c = a - b;    // Subtraction
let c = a * b;    // Multiplication
let c = a / b;    // Division
```

### With Scalars

```rust
let scale = 2.0f32;
let scaled = tile * scale;           // Multiply by scalar
let shifted = tile + 1.0f32;         // Add scalar
```

### Compound Operations

```rust
// SAXPY: y = a*x + y
let result = a * x + y;

// Fused multiply-add (more accurate)
let result = fma(a, x, y);

// Fused multiply-add with rounding mode
let result = fma(a, x, y, rounding_mode);
```

---

## Mathematical Functions

### Exponential and Logarithmic

```rust
let y = exp(x);              // e^x
let y = exp2(x, ftz::Disabled);             // 2^x (faster on GPU)
let y = log(x);              // Natural log (ln)
let y = log2(x);             // Log base 2
let y = sqrt(x, "rn");       // Square root (requires rounding mode)
let y = rsqrt(x);            // 1/sqrt(x) (fast reciprocal sqrt)
```

### Trigonometric

```rust
let y = sin(x);      // Sine
let y = cos(x);      // Cosine
let y = tanh(x);     // Hyperbolic tangent
```

### Other

```rust
let y = absf(x);             // Absolute value (float)
let y = absi(x);             // Absolute value (integer)
let y = negf(x);             // Negation (float)
let y = negi(x);             // Negation (integer)
let y = ceil(x, "rn");       // Ceiling (requires rounding mode)
let y = floor(x);            // Floor
```

---

## Reduction Operations

Reduce along an axis to produce a smaller tile:

### Max and Sum

```rust
// Input: Tile<f32, {[BM, BN]}>

// Reduce across columns (axis=1) → Tile<f32, {[BM]}>
let row_max = reduce_max(tile, 1i32);
let row_sum = reduce_sum(tile, 1);

// Reduce across rows (axis=0) → Tile<f32, {[BN]}>
let col_max = reduce_max(tile, 0i32);
let col_sum = reduce_sum(tile, 0);
```

![Reduction operations along axis 0 (columns) and axis 1 (rows)](../_static/images/reduction-axes.svg)

### Min

```rust
let row_min = reduce_min(tile, 1);
let col_min = reduce_min(tile, 0);
```

### Prod

```rust
let row_prod = reduce_prod(tile, 1);
let col_prod = reduce_prod(tile, 0);
```

---

## Matrix Operations

### Matrix Multiply-Accumulate (MMA)

The workhorse of GPU computing:

```rust
// C = A @ B + C
let c = mma(a, b, c);

// For accumulation loop:
let mut acc = constant(0.0f32, const_shape![BM, BN]);
for i in 0..K {
    let a_tile = load_a(i);
    let b_tile = load_b(i);
    acc = mma(a_tile, b_tile, acc);
}
```

**Shape requirements:**
- A: `[M, K]`
- B: `[K, N]`
- C: `[M, N]`
- Result: `[M, N]`

### Transpose / Permute

```rust
// Define permutation
let transpose: Array<{[1, 0]}> = Array::<{[1, 0]}> {
    dims: &[1i32, 0i32],
};

// Apply transpose
let transposed = permute(tile, transpose);
// [M, N] → [N, M]
```

---

## Broadcasting

Expand a smaller tile to match a larger shape:

### Scalar Broadcasting

```rust
// Broadcast scalar to tile
let scalar = 2.0f32;
let tile = scalar.broadcast(const_shape![64, 64]);
// Creates 64×64 tile filled with 2.0
```

### Dimension Broadcasting

```rust
// Broadcast [BM] to [BM, BN]
let row_values: Tile<f32, {[BM]}> = ...;
let expanded = row_values
    .reshape(const_shape![BM, 1])
    .broadcast(const_shape![BM, BN]);
```

### Common Pattern: Softmax Normalization

```rust
// Get max per row: [BM, BN] → [BM]
let row_max = reduce_max(tile, 1);

// Broadcast back: [BM] → [BM, BN]
let max_broadcast = row_max
    .reshape(const_shape![BM, 1])
    .broadcast(tile.shape());

// Subtract max from each element
let normalized = tile - max_broadcast;
```

---

## Shape Operations

### Reshape

Change shape without changing data (total elements must match):

```rust
// Flatten 2D to 1D
let flat = tile.reshape(const_shape![BM * BN]);

// Reshape for broadcasting
let col_vector = row.reshape(const_shape![BM, 1]);
```

### Get Shape

```rust
let shape = tensor.shape();
let dim_0 = get_shape_dim(tensor.shape(), 0i32);
```

---

## Comparison Operations

```rust
// Element-wise comparisons return bool tiles
let mask = gt_tile(a, b);    // a > b
let mask = ge_tile(a, b);    // a >= b
let mask = lt_tile(a, b);    // a < b
let mask = le_tile(a, b);    // a <= b
let mask = eq_tile(a, b);    // a == b
```

### Select (Conditional)

```rust
// Select elements based on mask
let result = select(mask, if_true, if_false);
```

---

## Control Flow Operations

### Tile-Level Max/Min

```rust
// Element-wise max/min of two tiles
let result = max_tile(a, b);
let result = min_tile(a, b);
```

---

## Constants

```rust
// Create constant tile
let zeros = constant(0.0f32, const_shape![64, 64]);
let ones = constant(1.0f32, const_shape![64, 64]);
let neg_inf = constant(f32::NEG_INFINITY, const_shape![BM, 1]);
```

### Iota (Index Generation)

```rust
// Create [0, 1, 2, 3, ...] tile
let indices: Tile<i32, {[64]}> = iota(const_shape![64]);
```

---

## Utility Operations

### Print (Debugging)

```rust
cuda_tile_print!("Value at tile ({}, {}): {}\n", 
    pid.0, pid.1, some_value);
```

:::{warning}
GPU printing is slow and should only be used for debugging small grids.
:::

### Type Conversion

```rust
let float_tile: Tile<f32, S> = convert_tile(int_tile);
let half_tile: Tile<f16, S> = convert_tile(float_tile);
```

---

## Common Operation Patterns

### Element-wise with Broadcast

```rust
fn scale_and_shift<const S: [i32; 2]>(
    x: Tile<f32, S>, scale: f32, shift: f32
) -> Tile<f32, S> {
    let s = scale.broadcast(x.shape());
    let b = shift.broadcast(x.shape());
    x * s + b
}
```

### Numerically Stable Softmax

```rust
fn softmax<const BM: i32, const BN: i32>(
    x: Tile<f32, { [BM, BN] }>
) -> Tile<f32, { [BM, BN] }> {
    // Subtract max for numerical stability
    let max: Tile<f32, { [BM, BN] }> = reduce_max(x, 1i32)
        .reshape(const_shape![BM, 1])
        .broadcast(const_shape![BM, BN]);
    let stable = x - max;

    // Compute softmax
    let exp_x = exp(stable);
    let sum: Tile<f32, { [BM, BN] }> = reduce_sum(exp_x, 1)
        .reshape(const_shape![BM, 1])
        .broadcast(const_shape![BM, BN]);

    true_div(exp_x, sum)
}
```

### Tiled Matrix Multiply

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

## Summary

| Category | Key Operations |
|----------|---------------|
| **Load/Store** | `load_tile_mut`, `load_tile_like_2d`, `partition().load()`, `store` |
| **Arithmetic** | `+`, `-`, `*`, `/`, `fma`, `true_div` |
| **Math** | `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` |
| **Reduction** | `reduce_max`, `reduce_sum`, `reduce_min`, `reduce_prod` |
| **Matrix** | `mma`, `permute` |
| **Shape** | `reshape`, `broadcast`, `const_shape!` |
| **Compare** | `gt_tile`, `ge_tile`, `lt_tile`, `le_tile`, `eq_tile`, `select` |
| **Element-wise** | `max_tile`, `min_tile`, `absf`, `negf`, `floor`, `ceil` |
| **Constants** | `constant`, `iota`, `convert_tile` |

---

Continue to [Async Execution](async-execution.md) to learn about concurrent CPU/GPU work.
