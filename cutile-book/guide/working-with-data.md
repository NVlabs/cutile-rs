# Working with Data

cuTile Rust leverages Rust's type system to catch errors at compile time. Shape mismatches, type errors, and many common GPU programming bugs are caught before your code even runs.

---

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
    let tile = load_tile_like(input, output);  // Load creates a tile
    let result = tile * 2.0;                      // Operations create new tiles
    output.store(result);                         // Store tile back to tensor
}
```

---

## Host-side and Device-side Types

The `Tensor` and `Partition` types exist on both the host side (CPU) and the device side (GPU kernel), but they are different Rust types with similar semantics. Host-side types are parameterized by element type only; device-side types additionally carry shape information in the type system for compile-time optimization.

On the host, you allocate tensors, partition them, and pass them to kernel launchers:

```rust
// Host-side Tensor<T> — parameterized by element type only
let mut tensor: Tensor<f32> = zeros(&[1024, 1024]).sync_on(&stream)?;

// Owned partition — moves the tensor into the partition
let partitioned: Partition<Tensor<f32>> = tensor.partition([16, 16]);

// Borrowed partition — borrows mutably, tensor written in place
let partitioned_ref = (&mut tensor).partition([16, 16]);

// Read-only inputs: borrow, Arc, or owned
let input: &Tensor<f32> = &tensor;
let shared: Arc<Tensor<f32>> = Arc::new(tensor);
```

The generated launcher accepts multiple forms for each parameter type. `&Tensor` params accept `&Tensor<T>`, `Arc<Tensor<T>>`, or `Tensor<T>`. `&mut Tensor` params accept `Partition<Tensor<T>>` or `Partition<&mut Tensor<T>>`.

The `api::*` module constructs tensors on the device. Each constructor returns a `DeviceOp`, so allocation and initialization are lazy until `.sync()` or `.await`:

```rust
use cutile::api;

// 3D tensor of random values from a standard normal distribution.
let weights: Tensor<f32> = api::randn(0.0f32, 1.0, [32, 64, 128], None).sync_on(&stream)?;

// Other common constructors: zeros, ones, full, arange, linspace, eye, rand, randn.
// Note: zeros/ones/full take &[usize] slices; rand/randn take [usize; RANK] arrays + Option<u64> seed.
```

**`TensorView`** provides zero-copy views and slices of an existing tensor, which matters for performance: when you want to process a subregion, views avoid the allocation and copy you'd otherwise need. The offset is applied host-side, so passing a view to a kernel hands it a pointer to the correct starting address with no data movement.

```rust
let tensor = api::arange::<f32>(1024).sync_on(&stream)?;

let matrix = tensor.view(&[32, 32])?;           // Reshape without copying
let row_slice = matrix.slice(&[1..3])?;         // Rows 1-2, all columns
let block = matrix.slice(&[1..3, 2..6])?;       // Rows 1-2, cols 2-5
```

Views and slices pass to kernels as `&Tensor` parameters. Use them for attention over a sub-sequence, GEMM over a sub-matrix, and similar sub-region patterns — no allocation, no copy. See the [Host API](../reference/host-api.md#tensor-creation-and-views) for the full list of constructors and `TensorView` methods.

Inside a kernel, tensors and tiles carry their shape as a type parameter, enabling compile-time shape checking and optimization:

```rust
fn kernel(
    output: &mut Tensor<f32, { [BM, BN] }>,  // Static shape from partition
    input: &Tensor<f32, { [-1, -1] }>,       // Dynamic shape
) {
    // Device-side Partition<E, S> — view of a tensor as tiles
    let part = input.partition(const_shape![BM, BK]);
    let tile = part.load([pid.0, i]);

    // Tile<E, S> — immutable data fragment in registers
    let tile_a: Tile<f32, { [BM, BN] }> = load_tile_like(input, output);
    let result = tile_a * 2.0;
    output.store(result);
}
```

| Type | Side | Parameterized By | Description |
|------|------|------------------|-------------|
| `Tensor<T>` | Host | Element type | Tensor in global memory; allocated and managed on the CPU |
| `Partition<Tensor<T>>` | Host | Element type | Host-side wrapper recording a tensor and its partition shape |
| `Arc<Tensor<T>>` | Host | Element type | Shared reference for read-only kernel inputs |
| `Tensor<E, S>` | Device | Element type + shape | Kernel parameter; `S` is static or dynamic (`-1`) |
| `Partition<E, S>` | Device | Element type + shape | Read-only view of a `&Tensor` divided into tiles inside a kernel |
| `Tile<E, S>` | Device | Element type + shape (always static) | Immutable data fragment in GPU registers |

For the full list of supported element types (`f16`, `bf16`, `f32`, `f64`, `tf32`, `f8e4m3fn`, `f8e5m2`, integer types, `bool`), see the [DSL API: ElementType](../reference/dsl-api.md#elementtype). For the `api::*` module and `TensorView`, see [Host API: Tensor Creation and Views](../reference/host-api.md#tensor-creation-and-views).

---

## Shapes and Broadcasting

**Static shapes** are compile-time constants. **Dynamic shapes** (written as `-1`) are determined at runtime.

```rust
// Static: known at compile time, fully optimized
fn kernel_static<const BM: i32, const BN: i32>(
    output: &mut Tensor<f32, { [BM, BN] }>,
) { /* BM and BN known; compiler can optimize layout and access patterns */ }

// Dynamic: -1 means "determined at runtime"
fn kernel_dynamic(
    input: &Tensor<f32, { [-1, -1] }>,
) {
    let shape = input.shape();  // Query actual dimensions at runtime
}
```

Static shapes let the compiler optimize layout and access patterns and catch shape errors at compile time, at the cost of re-compilation whenever a type or const generic changes. Dynamic shape dimensions that vary across launches do not trigger re-compilation.

The common pattern is **static output, dynamic input**: the tile size is a const generic, while the full tensor dimensions are runtime values:

```rust
#[cutile::entry()]
fn add<const S: [i32; 2]>(
    z: &mut Tensor<f32, S>,           // Static: tile knows its size
    x: &Tensor<f32, {[-1, -1]}>,      // Dynamic: full tensor
    y: &Tensor<f32, {[-1, -1]}>,      // Dynamic: full tensor
) {
    let tile_x = load_tile_like(x, z);  // Load matching z's shape
    let tile_y = load_tile_like(y, z);
    z.store(tile_x + tile_y);
}
```

**Broadcasting** expands a tile of one shape to operate against a tile of another, following NumPy rules: align dimensions from the right; two dimensions are compatible if they are equal or one of them is 1; the result shape is the maximum along each dimension.

```rust
// [64, 64] + [1, 64] -> [64, 64]  (broadcast B along dim 0)
let tile_a: Tile<f32, {[64, 64]}> = ...;
let tile_b: Tile<f32, {[1, 64]}> = ...;
let result = tile_a + tile_b.broadcast(const_shape![64, 64]);
```

---

## Type Safety and Generics

The compiler catches shape mismatches, element-type mismatches, and matrix-multiply dimension errors before code runs:

```rust
// ❌ Shape mismatch
let a: Tile<f32, {[4, 4]}> = ...;
let b: Tile<f32, {[8, 8]}> = ...;
let c = a + b;  // Error: cannot add [4,4] and [8,8]

// ❌ Element-type mismatch without conversion
let x: Tile<f32, {[4, 4]}> = ...;
let y: Tile<i32, {[4, 4]}> = ...;
let z = x + y;  // Error: cannot add f32 and i32

// ✅ Explicit conversion
let y_float: Tile<f32, {[4, 4]}> = convert_tile(y);
let z = x + y_float;

// ❌ MMA inner dimensions don't match: [M=16, K=8] × [K=16, N=32]
let a: Tile<f32, {[16, 8]}>;
let b: Tile<f32, {[16, 32]}>;
let c = mma(a, b, zeros);  // Error!

// ✅ MMA inner dims match: [M=16, K=8] × [K=8, N=32] -> [16, 32]
let a: Tile<f32, {[16, 8]}>;
let b: Tile<f32, {[8, 32]}>;
let c = mma(a, b, zeros);
```

**Generic kernels** let a single function handle multiple element types and shapes:

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

let generics = vec![
    "f32".to_string(),  // E
    "16".to_string(),   // BM
    "16".to_string(),   // BN
    "8".to_string(),    // BK
    "128".to_string(),  // K
];
gemm(z, x, y).generics(generics).sync_on(&stream);
```

Custom element types implement the [`ElementType`](../reference/dsl-api.md#elementtype) trait. The built-in numeric types all implement it.

---

Continue to [Writing Computations](writing-computations.md) for the operations you can apply on tiles. For type signatures, operator catalogs, and `const_shape!` / shape utilities, see the [DSL API reference](../reference/dsl-api.md).
