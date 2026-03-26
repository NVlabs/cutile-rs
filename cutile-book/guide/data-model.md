# Data Model & Types

cuTile Rust leverages Rust's type system to catch errors at compile time. Shape mismatches, type errors, and many common GPU programming bugs are caught before your code even runs.

## Tensors vs Tiles

cuTile Rust has two fundamental data abstractions that represent data at different levels of the memory hierarchy:

| Property | Tensor | Tile |
|----------|--------|------|
| **Location** | Global Memory (HBM) | GPU registers |
| **Mutability** | Mutable (`&mut`) or read-only (`&`) | Immutable |
| **Shape** | Mixed static / dynamic | Compile-time (static) |
| **Operations** | Load, Store | Arithmetic, Reduction, etc. |
| **Lifetime** | Persists across kernels | Exists only during kernel |
| **Addressable** | Yes (pointers) | No (compiler-managed) |

```rust
// Tensors: live in global memory, passed as kernel arguments
fn kernel(
    output: &mut Tensor<f32, S>,      // Mutable tensor (can store to)
    input: &Tensor<f32, {[-1, -1]}>   // Immutable tensor (read-only)
) {
    // Tiles: live in registers, created by loading
    let tile = load_tile_like_2d(input, output);  // Load creates a tile
    let result = tile * 2.0;                      // Operations create new tiles
    output.store(result);                         // Store tile back to tensor
}
```

---

## Element Types

cuTile Rust supports various numeric types for GPU computation:

### Floating Point Types

| Type | Size | Description | Use Case |
|------|------|-------------|----------|
| `f16` | 16-bit | Half precision | Training, inference (2× Tensor Core throughput) |
| `f32` | 32-bit | Single precision | General purpose, debugging |
| `f64` | 64-bit | Double precision | Scientific computing |
| `tf32` | 19-bit | TensorFloat-32 | Tensor Core operations |

### Integer Types

| Type | Size | Description |
|------|------|-------------|
| `i8` / `u8` | 8-bit | Signed/unsigned byte |
| `i32` / `u32` | 32-bit | Signed/unsigned int |
| `i64` / `u64` | 64-bit | Signed/unsigned long |

### Boolean Type

| Type | Description |
|------|-------------|
| `bool` | Boolean (true/false), maps to `i1` |

### Choosing Element Types

| Type | Performance | Precision | Recommendation |
|------|-------------|-----------|----------------|
| `f32` | Baseline | High | Development, debugging |
| `f16` | 2× on Tensor Cores | Medium | Inference |
| `bf16` | 2× on Tensor Cores | Medium (better range) | Training |
| `i32` | Native integer ops | Exact | Indexing, control flow |

---

## Shapes

Shapes define the dimensions of tensors and tiles.

### Static Shapes (Compile-Time)

When you know the shape at compile time, use const generics:

```rust
fn kernel<const BM: i32, const BN: i32>(
    output: &mut Tensor<f32, { [BM, BN] }>,  // Static shape
) {
    // BM and BN are known at compile time
    // Compiler can optimize layout and access patterns
}
```

**Benefits:**
- Compiler can optimize layout and access patterns
- Shape errors caught at compile time
- Zero runtime overhead for shape checks

**Drawbacks:**
- Kernels are re-compiled whenever their type or const generics change. 
- Too many consts which change across kernel launches will trigger excessive re-compilation, 
  which may not be desirable/optimal for all applications.

### Dynamic Shapes (Runtime)

When the shape is only known at runtime:

```rust
fn kernel(
    input: &Tensor<f32, { [-1, -1] }>,  // Dynamic shape
) {
    // -1 means "determined at runtime"
    let shape = input.shape();  // Query actual dimensions
}
```

Dynamic shape dimensions which vary across kernel launches do not trigger re-compilation.

### Common Tile Sizes

For optimal performance, tile dimensions are typically powers of two:

| Shape | Total Elements | Use Case |
|-------|----------------|----------|
| `[64, 64]` | 4,096 | General matrix ops |
| `[128, 128]` | 16,384 | Large matrix ops |
| `[256, 64]` | 16,384 | Tall tiles |
| `[64, 256]` | 16,384 | Wide tiles |
| `[1024]` | 1,024 | 1D vectors |

### The Common Pattern: Static Output, Dynamic Input

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,           // Static: tile knows its size
    x: &Tensor<f32, {[-1, -1]}>,      // Dynamic: full tensor
    y: &Tensor<f32, {[-1, -1]}>,      // Dynamic: full tensor
) {
    let tile_x = load_tile_like_2d(x, z);  // Load matching z's shape
    let tile_y = load_tile_like_2d(y, z);
    z.store(tile_x + tile_y);
}
```

---

## Shape Broadcasting

When operating on tiles of different shapes, cuTile Rust uses **broadcasting** rules similar to NumPy:

### Broadcasting Rules

1. **Align dimensions from the right**
2. **Dimensions are compatible if they're equal or one is 1**
3. **The result shape is the maximum along each dimension**

```rust
// Example: [64, 64] + [1, 64] -> [64, 64]
let tile_a: Tile<f32, [64, 64]> = ...;
let tile_b: Tile<f32, [1, 64]> = ...;
let result = tile_a + tile_b.broadcast(const_shape![64, 64]);  // Result is [64, 64], B broadcast along dim 0
```

<!--
### Broadcasting Examples

```
Shape A      Shape B      Result
[64, 64]  +  [64, 64]  -> [64, 64]   (exact match)
[64, 64]  +  [1, 64]   -> [64, 64]   (broadcast B along dim 0)
[64, 64]  +  [64, 1]   -> [64, 64]   (broadcast B along dim 1)
[64, 64]  +  [1, 1]    -> [64, 64]   (broadcast B along both dims)
[64, 1]   +  [1, 64]   -> [64, 64]   (broadcast both)
```
-->

<!--
### Arithmetic Promotion

When operating on tiles with different dtypes, automatic promotion occurs:

```rust
// f16 + f32 -> f32
let a: Tile<f16, S> = ...;
let b: Tile<f32, S> = ...;
let c = a + b;  // c is Tile<f32, S>
```

**Promotion Hierarchy:**
```
f16 < bf16 < f32 < f64
i8 < i16 < i32 < i64
```
-->

---

## Core Types

The `Tensor` and `Partition` types exist on both the host side (CPU) and the device side (GPU kernel), but they are different Rust types with similar semantics. Host-side types are parameterized by element type only; device-side types carry shape information in the type system for compile-time optimization.

### Host-Side Types

On the host, you allocate tensors, partition them, and pass them to kernel launchers:

```rust
// Host-side Tensor<T> — parameterized by element type only
let tensor: Tensor<f32> = zeros([1024, 1024]).sync_on(&stream)?;

// Host-side Partition<Tensor<T>> — wraps a tensor with a partition_shape
let partitioned: Partition<Tensor<f32>> = tensor.partition([16, 16]);
// 64×64 = 4096 sub-tensors, each 16×16

// Shared reference for read-only inputs
let shared: Arc<Tensor<f32>> = ones([1024, 1024]).arc().sync_on(&stream)?;
```

The generated launcher accepts `Partition<Tensor<T>>` for every `&mut Tensor` parameter and `Arc<Tensor<T>>` for every `&Tensor` parameter.

### Device-Side Types

Inside a kernel, tensors and tiles carry their shape as a type parameter. This enables compile-time shape checking and optimization:

```rust
// Device-side Tensor<E, S> — element type + shape
fn kernel(
    output: &mut Tensor<f32, { [BM, BN] }>,  // Static shape from partition
    input: &Tensor<f32, { [-1, -1] }>,       // Dynamic shape
) {
    // Device-side Partition<E, S> — view of a tensor as tiles
    let part = input.partition(const_shape![BM, BK]);
    let tile = part.load([pid.0, i]);

    // Tile<E, S> — immutable data fragment in registers
    let tile_a: Tile<f32, { [BM, BN] }> = load_tile_like_2d(input, output);
    let result = tile_a * 2.0;       // Operations create new tiles
    output.store(result);
}
```

| Type | Side | Parameterized By | Description |
|------|------|-----------------|-------------|
| `Tensor<T>` | Host | Element type | Tensor in global memory; allocated and managed on the CPU |
| `Partition<Tensor<T>>` | Host | Element type | Host-side wrapper recording a tensor and its partition shape |
| `Arc<Tensor<T>>` | Host | Element type | Shared reference for read-only kernel inputs |
| `Tensor<E, S>` | Device | Element type + shape | Kernel parameter; `S` is static or dynamic (`-1`) |
| `Partition<E, S>` | Device | Element type + shape | Read-only view of a `&Tensor` divided into tiles inside a kernel |
| `Tile<E, S>` | Device | Element type + shape (always static) | Immutable data fragment in GPU registers |

---

## Type Safety

### Compile-Time Shape Checking

The compiler catches shape mismatches:

```rust
// ❌ Won't compile: shapes don't match
let a: Tile<f32, {[4, 4]}> = ...;
let b: Tile<f32, {[8, 8]}> = ...;
let c = a + b;  // Error: cannot add [4,4] and [8,8]

// ✅ Correct: same shapes
let a: Tile<f32, {[4, 4]}> = ...;
let b: Tile<f32, {[4, 4]}> = ...;
let c = a + b;  // OK: both [4,4]
```

### Element Type Checking

```rust
// ❌ Won't compile: type mismatch without conversion
let x: Tile<f32, {[4, 4]}> = ...;
let y: Tile<i32, {[4, 4]}> = ...;
let z = x + y;  // Error: cannot add f32 and i32

// ✅ Correct: explicit conversion
let y_float: Tile<f32, {[4, 4]}> = convert_tile(y);
let z = x + y_float;  // OK
```

### Matrix Multiplication Shape Rules

For `C = A @ B`:
- A shape: `[M, K]`
- B shape: `[K, N]`
- C shape: `[M, N]`

The inner dimension K must match:

```rust
// ❌ Won't compile: inner dimensions don't match
let a: Tile<f32, {[16, 8]}>;   // [M=16, K=8]
let b: Tile<f32, {[16, 32]}>;  // [K=16, N=32]  K mismatch!
let c = mma(a, b, zeros);      // Error!

// ✅ Correct: K dimensions match
let a: Tile<f32, {[16, 8]}>;   // [M=16, K=8]
let b: Tile<f32, {[8, 32]}>;   // [K=8, N=32]   K matches!
let c = mma(a, b, zeros);      // OK: result is [16, 32]
```

---

## Type Conversions

### Explicit Casting

Convert between types explicitly:

```rust
let float_tile: Tile<f32, S> = ...;

// Float to integer
let int_tile: Tile<i32, S> = convert_tile(float_tile);

// Integer extension
let i8_tile: Tile<i8, S> = ...;
let i32_tile: Tile<i32, S> = convert_tile(i8_tile);
```

<!--
### Conversion Operations

| Operation | Description |
|-----------|-------------|
| `trunci::<T>()` | Truncate to smaller integer type |
| `exti::<T>()` | Sign-extend to larger integer type |
| `cast::<T>()` | General type conversion |
-->

---

## Generic Kernels

Use generics to specify flexible, reusable kernels:

```rust
#[cutile::entry()]
fn flexible_gemm<
    E: ElementType,              // Any element type
    const BM: i32,               // Tile rows
    const BN: i32,               // Tile cols
    const BK: i32,               // Inner tile dim
    const K: i32,                // Full inner dim
>(
    z: &mut Tensor<E, {[BM, BN]}>,
    x: &Tensor<E, {[-1, K]}>,
    y: &Tensor<E, {[K, -1]}>,
) {
    // Works for any element type and tile sizes!
}
```

Launch with specific types:

```rust
let generics = vec![
    "f32".to_string(),  // E
    "16".to_string(),   // BM
    "16".to_string(),   // BN
    "8".to_string(),    // BK
    "128".to_string(),  // K
];
gemm(z, x, y).generics(generics).sync_on(&stream);
```

### The ElementType Trait

Custom element types must implement `ElementType`:

```rust
pub trait ElementType: Copy + Clone {}

// Built-in implementations:
impl ElementType for f32 { ... }
impl ElementType for f16 { ... }
impl ElementType for i32 { ... }
// etc.
```

---

## Memory Layout

### Tensor Memory Layout

Tensors in global memory use row-major (C-style) layout:

```{figure} ../_static/images/tensor-memory-layout.svg
:width: 100%
:alt: Tensor memory layout showing row-major ordering
```

**Key insight:** Consecutive elements in a row are adjacent in memory, enabling coalesced memory access when threads read along rows.

### Tile Register Layout

Tiles exist in registers without a specific addressable layout. The compiler optimizes register usage automatically.

---

## Shape Utilities

### const_shape! Macro

Create compile-time shapes:

```rust
use cutile::core::const_shape;

let shape = const_shape![64, 64];       // [64, 64]
let shape_3d = const_shape![8, 16, 32]; // [8, 16, 32]
```

### Shape Operations

```rust
// Get shape at runtime
let dims = tensor.shape();  // Returns shape info

// Reshape (total elements must match)
let reshaped = tile.reshape(const_shape![8, 8]);

// Broadcast (expand dimensions)
let scalar: Tile<f32, {[]}> = constant(2.0f32, const_shape![]);
let expanded = scalar.broadcast(const_shape![64, 64]);
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| **Static shapes** `{[M, N]}` | Compile-time known, fully optimized |
| **Dynamic shapes** `{[-1, -1]}` | Runtime determined |
| **Tensor\<T\>** (host) | Tensor in global memory, allocated and managed on the CPU |
| **Tensor\<E, S\>** (device) | Kernel parameter with element type and shape |
| **Partition\<Tensor\<T\>\>** (host) | Wrapper recording a tensor and its partition shape |
| **Partition\<E, S\>** (device) | Read-only view of a tensor divided into tiles inside a kernel |
| **Tile\<E, S\>** (device only) | Immutable data fragment in GPU registers |
| **Const generics** | Flexible, type-safe kernels |
| **Broadcasting** | Automatic shape expansion |

**Key benefits:**
- Catch shape mismatches at compile time
- Zero runtime overhead for static shapes
- Generic kernels work with any valid configuration

---

## Next Steps

- See [Operations](operations.md) for available tile operations
- Learn about [Memory Hierarchy](memory-hierarchy.md) for performance
- Explore the [Syntax Reference](../appendix/syntax-reference.md) for complete API
