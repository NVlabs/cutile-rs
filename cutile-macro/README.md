# cuTile Rust Macros

This crate defines the procedural macros behind `cutile::module`.
It expands cuTile Rust modules into:

- type-checkable Rust items
- JIT-compilable AST builders
- generated host-side kernel launch functions

# Debugging

Generate backtraces:
```bash
export RUSTFLAGS="-Zproc-macro-backtrace -Zmacro-backtrace"; export RUST_BACKTRACE=1;
```

Dump the generated kernel launch code to a given directory:
```bash
export DUMP_KERNEL_LAUNCHER_DIR="temp"
```
# Description and Limitations

This crate does the following:
- Generates concrete items for structs and functions which take const generic arrays (CGAs).
- Rewrites struct types and calls to functions which take CGAs.

The procedural macro has only syntax available to it. Functions which take CGAs require the length of the CGA
to be known during macro expansion.

For example:
```rust
fn f<const S: [i32; N]>(param1: Type1<S>) -> Type2<S> {...}

fn kernel<const S: [i32; 2]>(shape: Shape<S>) {
    let x: Type1<S> =  type_1_constructor(shape);
    let y: Type2<S> = f(x);
}
```

`f` takes a type parameterized by a variable length const generic array. During macro expansion, 
const generic arrays of each length up to `N=x`, where `x` is a known literal during macro expansion,
are generated as follows:

```rust
// CGA instance generation.
fn f__0<const S: [i32; 0]>(param1: Type1__0<S>) -> Type2__0<S> {...}
fn f__1<const S: [i32; 1]>(param1: Type1__1<S>) -> Type2__1<S> {...}
fn f__2<const S: [i32; 2]>(param1: Type1__2<S>) -> Type2__2<S> {...}
```

Const generic arrays are then desugared and emitted as follows:
```rust
// CGA desugaring.
fn f__0(param1: Type1__0) -> Type2__0 {...}
fn f__1<const S1: i32>(param1: Type1__1<S1>) -> Type2__1<S1> {...}
fn f__2<const S1: i32, const S2: i32>(param1: Type1__2<S1, S2>) -> Type2__2<S1, S2> {...}
```

Defining and expanding variable length CGAs is only available in the core module.
Programmers are otherwise free to use const generic arrays with known lengths in their own cuTile Rust modules.

For example: 

```rust
fn user_function<const S: [i32; 2]>(x: f32, shape: Shape<S>) -> Tile<f32, S> {
    let ones_shape: Shape<{[1, 1]}> = Shape::<{[1, 1]}>{dims: &[]};
    let tile_x: Tile<f32, {[]}> = scalar_to_tile(x);
    tile_x.reshape(ones_shape).broadcast(shape)
}
```

The cuTile Rust macro will desugar this into the following function:
```rust
fn user_function<const S1: i32, const S2: i32>(x: f32, shape: Shape__2<S1, S2>) -> Tile__2<f32, S1, S2> {
    let ones_shape: Shape__2<1, 1> = Shape__2::<1, 1>{dims: &[]};
    let tile_x: Tile__0<f32> = scalar_to_tile(x);
    tile_x.reshape__0__2(ones_shape).broadcast(shape)
}
```

The reshape method requires distinct implementations due to its use of multiple CGAs of different rank.

A current limitation of this solution is that the types of expressions which return types which are generic over 
const arrays must be known during macro expansion. 
In some cases, this is only possible if the type information for the expression
is named before providing the expression to a function which supports variable length const generic arrays.

For example:
```rust
fn kernel<const S: [i32; 2]>(a: f32, y: Type<S>) {
    let ones_shape: Shape<{[1, 1]}> = Shape::<{[1, 1]}>{dims: &[]};
    let tile_a: Tile<f32, {[1, 1]}> = reshape(scalar_to_tile(a), ones_shape);
}
```

The above function needs to be desugared to Rust:
```rust
fn kernel<const S1: i32, const S2: i32>(a: f32, y: Type__2<S1, S2>) {
    let ones_shape: Shape__2<1, 1> = Shape__2::<1, 1>{dims: &[]};
    let tile_a: Tile__2<f32, {[1, 1]}> = reshape__0__2(scalar_to_tile(a), ones_shape);
}
```

We are able to infer the rank of the shape (2), but we are unable to infer the rank of the input tile (0).
The macro only has the syntax `scalar_to_tile(a)`. For many built-in functions, we provide hints to the macro to
infer the types of these kinds of expressions, but inference of return types more generally
is not supported.

For the above kernel function, the compiler provides the source of the problem:
```
> Unable to infer type for argument 0 for call to reshape. Expected Tile. Required by calls to variadic functions and methods.
```

These error messages can be improved to provide explicit solutions to the problem. The solution in this case is to
bind the result of `scalar_to_tile(a)` to a variable with static type:
```rust
fn kernel<const S: [i32; 2]>(a: f32, y: Type<S>) {
    let ones_shape: Shape<{[1, 1]}> = Shape::<{[1, 1]}>{dims: &[1i32; 2]};
    let tile_a: Tile<f32, {[]}> = scalar_to_tile(a);
    let tile_a: Tile<f32, {[1, 1]}> = reshape(a, ones_shape);
}
```

Since variable length const generic arrays are only available to the core module,
this is not a major limitation to programmers who write their own functions. 

For example:
```rust
fn user_function<const S: [i32; 2]>(x: f32, shape: Shape<S>) -> Tile<f32, S> {
    let ones_shape: Shape<{[1, 1]}> = Shape::<{[1, 1]}>{dims: &[1i32, 1i32]};
    let tile_x: Tile<f32, {[]}> = scalar_to_tile(x);
    tile_x.reshape(ones_shape).broadcast(shape)
}

fn kernel<const S: [i32; 2]>(a: f32, y: Type<S>) {
    let tile_a: Tile<f32, S> = user_function(a, y.shape());
}
```

Since `user_function` does not take a variable length CGA, the cuTile Rust macro does not need to rewrite it.
In other words, the expression `user_function(a, y.shape())` compiles because there is only one `user_function`.

# Testing

Run the macro crate tests with:

```bash
cargo test -p cutile-macro
```
