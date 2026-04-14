# Syntax Snippets

> Note: This is a WIP -- there may be technical inaccuracies.

This comprehensive guide covers all `cutile-rs` syntax from basics to advanced patterns. Use it as a reference when writing kernels.

---

## Module Structure

Every cutile kernel lives inside a module marked with `#[cutile::module]`:

```rust
#[cutile::module]
mod my_module {
    use cutile::core::*;  // Import all tile operations

    #[cutile::entry()]
    fn my_kernel(/* parameters */) {
        // Kernel code
    }
}
```

### Entry Point Attributes

The `#[cutile::entry()]` macro supports several options:

```rust
#[cutile::entry(
    print_ir = false,           // Print MLIR IR during compilation
    unchecked_accesses = false, // Skip bounds checking
    optimization_hints = (
        sm_120 = (num_cta_in_cga = 1, max_divisibility = 16,),
    )
)]
fn my_kernel() { /* ... */ }
```

---

## Core Types

### Element Types

```rust
// Floating point
f32    // 32-bit float (most common)
f64    // 64-bit float (double precision)
f16    // 16-bit float (half precision)
tf32   // TensorFloat-32

// Integer types
i8     // 8-bit signed (for integer GEMM)
i32    // 32-bit signed integer
i64    // 64-bit signed integer
u32    // 32-bit unsigned integer
u64    // 64-bit unsigned integer
bool   // Boolean
```

### Tensor Types

| Type | Description | Memory Location |
|------|-------------|-----------------|
| `Tensor<E, S>` | Data in global memory | GPU HBM |
| `Tile<E, S>` | Data in registers | GPU registers |
| `Partition<E, S>` | Read-only view as tiles | Metadata |
| `PartitionMut<E, S>` | Mutable view as tiles | Metadata |
| `PointerTile<P, S>` | Tile of pointers | Registers |

### Shape Syntax

```rust
// Static shapes (compile-time known)
Tensor<f32, {[64, 64]}>       // 2D: 64×64
Tensor<f32, {[128, 256, 512]}>// 3D: 128×256×512
Tensor<f32, {[]}>             // Scalar

// Dynamic shapes (runtime determined)
Tensor<f32, {[-1, -1]}>       // 2D dynamic
Tensor<f32, {[-1, 128]}>      // Mixed: dynamic first, static second

// Const generics for flexible kernels
fn kernel<const S: [i32; 2]>(x: &mut Tensor<f32, S>) { }
fn kernel<const BM: i32, const BN: i32>(x: &mut Tensor<f32, {[BM, BN]}>) { }
```

---

## Kernel Parameters

### Input/Output Tensors

```rust
#[cutile::entry()]
fn kernel(
    // Mutable output (will be written to)
    output: &mut Tensor<f32, {[BM, BN]}>,
    
    // Immutable inputs
    x: &Tensor<f32, {[-1, -1]}>,
    y: &Tensor<f32, {[-1, -1]}>,
    
    // Scalar parameters
    scale: f32,
    num_iters: i32,
    flag: bool,
) { }
```

### Generic Kernels

```rust
#[cutile::entry()]
fn generic_kernel<
    E: ElementType,           // Any element type
    const BM: i32,            // Tile rows
    const BN: i32,            // Tile cols
    const K: i32,             // Full dimension
>(
    z: &mut Tensor<E, {[BM, BN]}>,
    x: &Tensor<E, {[-1, K]}>,
) { }
```

---

## Loading and Storing Data

### Basic Load/Store

```rust
// Load entire output tile into registers
let tile: Tile<f32, S> = load_tile_mut(output);

// Store tile back to tensor
output.store(tile);

// Equivalent methods on Tensor
let tile = output.load();
output.store(tile);
```

### Positional Loading (load_tile_like)

Load from a dynamic tensor at the position matching another tile:

```rust
// Load from x at the same grid position as output tile z
let tile_x = load_tile_like_2d(x, z);  // 2D tensors
let tile_y = load_tile_like_2d(y, z);
```

### Partitioned Loading

For explicit control over which tile to load:

```rust
// Create partition view
let part = tensor.partition(const_shape![16, 16]);

// Load specific tile by index
let pid: (i32, i32, i32) = get_tile_block_id();
let tile = part.load([pid.0, pid.1]);

// Load with explicit indices
let tile = part.load([row_idx, col_idx]);
```

---

## Shape Operations

### Creating Shapes

```rust
// Using const_shape! macro
let shape = const_shape![64, 64];
let shape_3d = const_shape![8, 16, 32];

// Using Shape struct
let shape: Shape<{[128, 256]}> = Shape::<{[128, 256]}> {
    dims: &[128i32, 256i32],
};
```

### Getting Shape Information

```rust
// Get shape from tensor
let shape = tensor.shape();

// Get specific dimension
let dim0 = get_shape_dim(tensor.shape(), 0i32);
let dim1 = get_shape_dim(tensor.shape(), 1i32);
```

### Reshape

Change shape without changing data (total elements must match):

```rust
// Flatten 2D to 1D
let flat = tile.reshape(const_shape![BM * BN]);

// Add dimension for broadcasting
let col_vector = row.reshape(const_shape![BM, 1]);

// Remove dimensions
let reduced = tile.reshape(const_shape![BM]);
```

### Broadcast

Expand a smaller tile to a larger shape:

```rust
// Scalar to tile
let scalar_tile = 2.0f32.broadcast(const_shape![64, 64]);

// Column vector to matrix
let col: Tile<f32, {[BM, 1]}> = ...;
let expanded = col.broadcast(const_shape![BM, BN]);

// Full pattern: reshape then broadcast
let row_values: Tile<f32, {[BM]}> = reduce_max(tile, 1i32);
let broadcast = row_values
    .reshape(const_shape![BM, 1])
    .broadcast(const_shape![BM, BN]);
```

### Permute (Transpose)

```rust
// Define permutation
let transpose: Array<{[1, 0]}> = Array::<{[1, 0]}> {
    dims: &[1i32, 0i32],
};

// Apply permutation: [M, N] → [N, M]
let transposed = permute(tile, transpose);
```

---

## Arithmetic Operations

### Basic Arithmetic

```rust
let c = a + b;    // Addition
let c = a - b;    // Subtraction
let c = a * b;    // Multiplication
let c = a / b;    // Division
let c = true_div(a, b);  // True division (for floats)
```

### Scalar Operations

```rust
let scaled = tile * 2.0f32;
let shifted = tile + 1.0f32;
```

### Fused Multiply-Add

```rust
// Simpler form (defaults to nearest_even rounding)
let result = fma(x, y, z);

// x * y + z with explicit rounding mode
let result = fma(x, y, z, "nearest_even");
```

---

## Mathematical Functions

### Exponential and Logarithmic

```rust
let y = exp(x);       // e^x
let y = exp2(x);      // 2^x (faster on GPU)
let y = log(x);       // Natural log (ln)
let y = log2(x);      // Log base 2
let y = sqrt(x, "negative_inf");   // Square root
let y = rsqrt(x);     // 1/sqrt(x) - fast reciprocal sqrt
let y = pow(x, y);    // x^y
```

### Trigonometric

```rust
let y = sin(x);       // Sine
let y = cos(x);       // Cosine
let y = tan(x);       // Tangent
let y = sinh(x);      // Hyperbolic sine
let y = cosh(x);      // Hyperbolic cosine
let y = tanh(x);      // Hyperbolic tangent
```

### Rounding and Absolute Value

```rust
let y = ceil(x, "nearest_even");   // Ceiling
let y = floor(x);                   // Floor
let y = absf(x);                    // Absolute value (float)
let y = absi(x);                    // Absolute value (int)
let y = negf(x);                    // Negation (float)
let y = negi(x);                    // Negation (int)
```

---

## Reduction Operations

Reduce along an axis to produce a smaller tile:

```rust
// Input: Tile<f32, {[BM, BN]}>

// Reduce across columns (axis=1) → Tile<f32, {[BM]}>
let row_max = reduce_max(tile, 1i32);
let row_sum = reduce_sum(tile, 1);
let row_min = reduce_min(tile, 1);

// Reduce across rows (axis=0) → Tile<f32, {[BN]}>
let col_max = reduce_max(tile, 0i32);

// Product along axis
let product = reduce_prod(tile, 0i32);
```

### Custom Reductions with Closures

```rust
// Sum reduction with closure
let sum = reduce(tile, 0i32, 0.0f32, |acc, x| acc + x);

// Product reduction
let product = reduce(tile, 0i32, 1.0f32, |acc, x| acc * x);

// Max reduction with identity element
let max_val = reduce(tile, 0i32, f32::NEG_INFINITY, |acc, x| max(acc, x));
```

---

## Scan (Prefix) Operations

Cumulative operations along an axis:

```rust
// Prefix sum (cumulative sum)
let prefix_sums: Tile<f32, S> = scan_sum(tile, 0i32, false, 0.0f32);
//                                              axis   reverse  init

// Custom scan with closure (prefix product)
let prefix_products = scan(tile, 0i32, false, 1.0f32, |acc, x| acc * x);
```

---

## Matrix Operations

### Matrix Multiply-Accumulate (MMA)

```rust
// C = A @ B + C
let c = mma(a, b, c);

// Shape requirements:
// A: [M, K]
// B: [K, N]
// C: [M, N]
// Result: [M, N]

// Typical accumulation loop
let mut acc = constant(0.0f32, const_shape![BM, BN]);
for i in 0i32..(K / BK) {
    let tile_x = part_x.load([pid.0, i]);
    let tile_y = part_y.load([i, pid.1]);
    acc = mma(tile_x, tile_y, acc);
}
```

### Integer MMA

```rust
// For i8 inputs with i32 accumulator
let lhs: Tile<i8, {[16, 32]}> = constant(1i8, lhs_shape);
let rhs: Tile<i8, {[32, 16]}> = constant(1i8, rhs_shape);
let acc: Tile<i32, {[16, 16]}> = constant(0i32, acc_shape);
let result = mma(lhs, rhs, acc);
```

---

## Comparison and Selection

### Element-wise Comparisons

```rust
let mask = gt_tile(a, b);    // a > b
let mask = ge_tile(a, b);    // a >= b
let mask = lt_tile(a, b);    // a < b
let mask = le_tile(a, b);    // a <= b
let mask = eq_tile(a, b);    // a == b
```

### Min/Max

```rust
// Element-wise max/min of two tiles
let result = max_tile(a, b);
let result = min_tile(a, b);
let result = maxf(a, b);     // Float max
let result = minf(a, b);     // Float min
```

### Select (Conditional)

```rust
// Select elements based on mask
let result = select(mask, if_true, if_false);
```

---

## Constants and Special Values

### Creating Constant Tiles

```rust
let zeros = constant(0.0f32, const_shape![64, 64]);
let ones = constant(1.0f32, const_shape![64, 64]);
let neg_inf = constant(f32::NEG_INFINITY, const_shape![BM, 1]);
let custom = constant(42i64, output.shape());
```

### Index Generation (Iota)

```rust
// Create [0, 1, 2, 3, ...] tile
let indices: Tile<i32, {[64]}> = iota(const_shape![64]);
```

---

## Type Conversions

### Scalar Conversions

```rust
let x: f32 = 0.0;
let x: i32 = convert_scalar::<i32>(x);
let x: f32 = convert_scalar::<f32>(x);
let x: f16 = convert_scalar::<f16>(x);
```

### Tile Type Casting

```rust
let float_tile: Tile<f32, S> = convert_tile(int_tile);
let half_tile: Tile<f16, S> = convert_tile(float_tile);
```

### Integer Extension/Truncation

```rust
// Truncate to smaller type
let truncated: Tile<i32, S> = trunci(tile_i64);

// Extend to larger type (sign/zero extend based on source type)
let extended: Tile<i64, S> = exti(tile_i32);
```

### Pointer Conversions

```rust
let ptrs: PointerTile<*mut i64, S> = int_to_ptr(int_tile);
let ptrs_f32: PointerTile<*mut f32, S> = ptr_to_ptr(ptrs);
let ints: Tile<i64, S> = ptr_to_int(ptrs);
```

---

## Control Flow

### For Loops

```rust
for i in 0i32..10i32 {
    // Loop body
    acc = acc + tile;
}

// With step
for i in (0i32..100i32).step_by(10) {
    // ...
}
```

### While Loops

```rust
let mut counter = 0i32;
while counter < 10i32 {
    acc = acc + acc;
    counter = counter + 1i32;
}
```

### Infinite Loop with Break

```rust
loop {
    acc = acc + acc;
    if condition {
        break;
    }
}
```

### If/Else

```rust
if dynamic_value < 5i32 {
    sum = sum + sum;
} else {
    sum = sum - sum;
}

// If as expression (returns value)
let result: Tile<i64, S> = if conditional {
    constant(2, shape)
} else {
    constant(3, shape)
};
```

---

## Grid and Block Information

### Getting Tile Position

```rust
// Get current tile's position in the grid
let pid: (i32, i32, i32) = get_tile_block_id();
// pid.0 = x position, pid.1 = y position, pid.2 = z position

// Get total number of tiles in grid
let npid: (i32, i32, i32) = get_num_tile_blocks();
```

### Common Indexing Patterns

```rust
// Batch and head indexing (for attention)
let h = get_shape_dim(q.shape(), 1i32);
let batch_idx = pid.0 / h;
let head_idx = pid.0 % h;
let seq_idx = pid.1;

// Group query attention
let kv_head_idx = head_idx / query_group_size;
```

---

## Utility Functions

### Ceiling Division

```rust
let num_tiles: i32 = ceil_div(n, BN);  // ceil(n / BN)
```

### Debug Printing

```rust
cuda_tile_print!("Value at tile ({}, {}): {}\n", pid.0, pid.1, value);
cuda_tile_print!("Shape: {} x {}\n", dim0, dim1);
```

:::{warning}
GPU printing is slow and should only be used for debugging with small grids.
:::

### Assertions

```rust
cuda_tile_assert!(condition, "Error message");
cuda_tile_assert!(shape_dim_1 != shape_dim_2, "Dimensions must differ");
```

---

## Scalar/Tile Conversions

```rust
// Scalar to 0-D tile
let tile_scalar: Tile<f32, {[]}> = scalar_to_tile(scalar);

// 0-D tile to scalar
let scalar: f32 = tile_to_scalar(tile_scalar);

// Pointer to tile
let ptr_tile: PointerTile<*mut f32, {[]}> = pointer_to_tile(ptr);
let ptr: *mut f32 = tile_to_pointer(ptr_tile);
```

---

## Advanced: Tensor Slicing

### Extract

Extract slices from a tile:

```rust
let source: Tile<f32, {[8]}> = load_tile_mut(tensor);

// Extract first half
let idx0: Tile<i32, {[]}> = scalar_to_tile(0i32);
let slice0: Tile<f32, {[4]}> = extract(source, [idx0]);

// Extract second half
let idx1: Tile<i32, {[]}> = scalar_to_tile(1i32);
let slice1: Tile<f32, {[4]}> = extract(source, [idx1]);
```

### Concatenate

```rust
// Concatenate two tiles along an axis
let result: Tile<f32, {[8]}> = cat(tile_a, tile_b, 0i32);
```

---

## Advanced: Memory Operations

### Low-Level Partition Views

```rust
unsafe {
    let token: Token = new_token_unordered();
    let partition: PartitionMut<f32, {[128, 256]}> =
        make_partition_view_mut(&tensor, shape, token);
    
    let idx: [i32; 2] = [0i32, 0i32];
    let tile = load_from_view_mut(&partition, idx);
    store_to_view_mut(&mut partition, tile, idx, None, false);
}
```

### Pointer-Based Load/Store

```rust
// Build pointer tile
let ptr_seed: Tile<i64, S> = constant(0i64, shape);
let ptrs: PointerTile<*mut f32, S> = ptr_to_ptr(int_to_ptr(ptr_seed));

// Load with memory ordering
let (values, token): (Tile<f32, S>, Token) =
    load_ptr_tko(ptrs, "relaxed", "device", None, None, None, None);

// Store with memory ordering
let token: Token =
    store_ptr_tko(ptrs, values, "relaxed", "device", None, None, None);

// Per-op latency hint
let (values, token): (Tile<f32, S>, Token) =
    load_ptr_tko(ptrs, "weak", "tl_blk", None, None, None, Some(4));
```

Memory orderings: `"relaxed"`, `"weak"`, `"acquire"`, `"release"`, `"acq_rel"`
Scopes: `"device"`, `"sys"`, `"tl_blk"`

---

## Advanced: Atomic Operations

### Atomic Read-Modify-Write

```rust
// Atomic add
let (old_values, token): (Tile<f32, S>, Token) =
    atomic_rmw_tko(ptrs, increments, "addf", "relaxed", "device", None, None);

// Operations: "add", "addf", "and", "or", "xor", "max", "min", "xchg"
```

### Atomic Compare-and-Swap

```rust
let (old_values, token): (Tile<f32, S>, Token) = atomic_cas_tko(
    ptrs,
    cmp_values,      // Expected value
    new_values,      // New value if match
    "relaxed",       // Memory ordering
    "device",        // Scope
    None,            // Optional mask
    None,            // Optional input token
);
```

---

## Advanced: Compiler Hints

### Assume Operations

Provide optimization hints to the compiler:

```rust
// Assume values are non-negative
let assumed: Tile<i64, S> = unsafe { assume_bounds_lower::<_, 0>(tile) };

// Assume values are divisible by 16
let aligned: Tile<i64, S> = unsafe { assume_div_by::<_, 16>(tile) };

// Assume groups have same elements (for 2D tiles)
let same: Tile<i64, S> = unsafe { assume_same_elements_2d::<_, 2, 4>(tile) };
```

---

## Common Patterns

### Numerically Stable Softmax

```rust
fn softmax<const BM: i32, const BN: i32>(
    x: &Tensor<f32, {[-1, -1]}>,
    y: &mut Tensor<f32, {[BM, BN]}>,
) {
    let tile_x: Tile<f32, {[BM, BN]}> = load_tile_like_2d(x, y);

    // Subtract max for stability
    let tile_max: Tile<f32, {[BM]}> = reduce_max(tile_x, 1i32);
    let tile_max = tile_max.reshape(const_shape![BM, 1]).broadcast(y.shape());

    let num = exp(tile_x - tile_max);
    let denom = reduce_sum(num, 1).reshape(const_shape![BM, 1]).broadcast(y.shape());

    y.store(num / denom);
}
```

### Tiled GEMM

```rust
fn gemm<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
    z: &mut Tensor<f32, {[BM, BN]}>,
    x: &Tensor<f32, {[-1, K]}>,
    y: &Tensor<f32, {[K, -1]}>,
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

### Fused Multihead Attention Pattern

```rust
// Running softmax state
let mut m_i = constant(f32::NEG_INFINITY, const_shape![BM, 1]);
let mut l_i = constant(0.0f32, const_shape![BM, 1]);
let mut acc = constant(0.0f32, const_shape![BM, D]);

for j in 0i32..num_tiles {
    // Q @ K^T
    let qk = mma(tq, k_tile_trans, zeros);
    let qk = qk * qk_scale;

    // Online softmax update
    let qk_max = reduce_max(qk, 1).reshape(const_shape![BM, 1]);
    let m_ij = max_tile(m_i, qk_max);
    let qk = qk - m_ij.broadcast(const_shape![BM, BN]);
    let p = exp2(qk);
    let l_ij = reduce_sum(p, 1).reshape(const_shape![BM, 1]);
    let alpha = exp2(m_i - m_ij);
    l_i = l_i * alpha + l_ij;
    acc = acc * alpha.broadcast(const_shape![BM, D]);
    
    // P @ V
    acc = mma(p, v_tile, acc);
    m_i = m_ij;
}

acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
```

---

## Host-Side API

### Kernel Parameter Types

| Kernel param | Host input | Return type |
|---|---|---|
| `&Tensor<T, S>` | `Tensor<T>`, `Arc<Tensor<T>>`, or `&Tensor<T>` | Same as input |
| `&mut Tensor<T, S>` | `Partition<Tensor<T>>` or `Partition<&mut Tensor<T>>` | Same as input |
| Scalar (`f32`, `i32`, etc.) | Same scalar | Same scalar |
| `*mut T` (unsafe only) | `DevicePointer<T>` | `DevicePointer<T>` |

Borrowed forms (`&Tensor<T>`, `Partition<&mut Tensor<T>>`) introduce a
lifetime that prevents `tokio::spawn` at compile time — use `Arc` and
owned partitions for spawned tasks.

### Launching Kernels (Sync)

```rust
use my_module::kernel;

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;

// Borrow-based: no Arc, no unpartition, no return capture.
let x = ones::<f32>(&[32, 32]).sync_on(&stream)?;
let mut z = zeros::<f32>(&[32, 32]).sync_on(&stream)?;

let _ = kernel((&mut z).partition([4, 4]), &x)
    .generics(vec!["f32".to_string(), "16".to_string()])
    .sync_on(&stream)?;
// z already has the result.

// Owned variant (useful when building lazy graphs):
let x: Arc<Tensor<f32>> = ones(&[32, 32]).sync_on(&stream)?.into();
let z = zeros(&[32, 32]).partition([4, 4]).sync_on(&stream)?;

let (z, _x) = kernel(z, x)
    .generics(vec!["f32".to_string(), "16".to_string()])
    .sync_on(&stream)?;
let z_host: Vec<f32> = z.unpartition().to_host_vec().sync_on(&stream)?;
```

### Launching Kernels (Async)

```rust
use my_module::kernel;

let x = ones(&[32, 32]).map(Into::into);
let z = zeros(&[32, 32]).partition([4, 4]);

let (z, _x) = kernel(z, x).unzip();

let z_host: Vec<f32> = z.unpartition().await?.to_host_vec().await?;
```

---

## Quick Reference Tables

### Load Functions

| Function | Use Case |
|----------|----------|
| `load_tile_mut(tensor)` | Load entire output tile |
| `tensor.load()` | Same as above (method form) |
| `load_tile_like_2d(src, ref)` | Load from src at ref's position |
| `part.load([i, j])` | Load specific partition tile |

### Store Functions

| Function | Use Case |
|----------|----------|
| `tensor.store(tile)` | Store tile to tensor |

### Reduction Functions

| Function | Result Shape | Description |
|----------|--------------|-------------|
| `reduce_max(tile, axis)` | Removes axis | Maximum along axis |
| `reduce_sum(tile, axis)` | Removes axis | Sum along axis |
| `reduce_min(tile, axis)` | Removes axis | Minimum along axis |
| `reduce_prod(tile, axis)` | Removes axis | Product along axis |
| `reduce(tile, axis, init, fn)` | Removes axis | Custom reduction |

### Math Functions

| Float | Integer | Description |
|-------|---------|-------------|
| `exp(x)` | — | e^x |
| `exp2(x)` | — | 2^x |
| `log(x)` | — | ln(x) |
| `log2(x)` | — | log₂(x) |
| `sqrt(x, mode)` | — | √x |
| `rsqrt(x)` | — | 1/√x |
| `absf(x)` | `absi(x)` | |x| |
| `negf(x)` | `negi(x)` | -x |
| `maxf(a, b)` | — | max(a, b) |
| `minf(a, b)` | — | min(a, b) |

---

## Next Steps

- Try the [Tutorials](../tutorials/01-hello-world.md) for hands-on examples
- Learn about the [Memory Hierarchy](../guide/memory-hierarchy.md) for optimization
- Understand [Async Execution](../guide/async-execution.md) for production code
