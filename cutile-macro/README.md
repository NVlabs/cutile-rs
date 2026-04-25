# cuTile Rust Macros

Procedural macros behind `cutile::module`. They expand cuTile Rust modules into:

- Type-checkable Rust items that drive rustc's validation of kernel code.
- An AST capture (`_module_asts()`) that hands the original generic source to the JIT compiler.
- Host-side kernel launch functions for `#[entry]` points.

## Two-track design: macro output is for rustc, not the JIT

The macro runs at Rust compile time and emits three kinds of output:

1. **Per-rank specializations and rank-polymorphic trait dispatch.** Items that use a const-generic-array (CGA) generic — `const X: [i32; N]` — get materialized as concrete Rust that rustc can type-check. See *Per-rank expansion* and *Trait dispatch* below.
2. **AST capture.** `_module_asts()` stores the pre-expansion source text via `Span::source_text()` and re-parses it at runtime for the JIT compiler. The JIT sees the *original generic* `fn foo<const S: [i32; N]>(...) { … real body … }`, not macro-emitted expansions.
3. **Kernel launchers** for `#[entry]` functions.

The separation matters: items emitted in (1) exist only to make rustc type-check kernel bodies and give good error messages. The JIT does its own per-rank instantiation from the original source, independent of anything the macro emits. Consequently, the macro never has to grow to support new user-code language features — adding support for (say) user-defined structs or methods is the JIT's job. The macro only has to handle whatever Rust patterns appear in op signatures inside `_core.rs`.

## Per-rank expansion (`rank_expansion.rs`)

For items annotated with `#[cuda_tile::variadic_struct]`, `#[cuda_tile::variadic_impl]`, or `#[cuda_tile::variadic_trait_impl]`, the macro emits one concrete copy per rank, replacing each `const X: [i32; N]` with scalar consts `X_0, X_1, …, X_{R-1}` and suffixing CGA-bearing path segments accordingly.

```rust
// Source
#[cuda_tile::variadic_struct(N = 4)]
pub struct Tile<E: ElementType, const D: [i32; N]> { _type: PhantomData<E> }

// Macro output (abridged):
pub struct Tile_1<E: ElementType, const D_0: i32> { /* … */ }
pub struct Tile_2<E: ElementType, const D_0: i32, const D_1: i32> { /* … */ }
// … through Tile_4
```

Inherent impls expand the same way; their method bodies get walked by a `syn::visit_mut::VisitMut` pass (`RankExpander`) that substitutes the active CGA bindings into types, expression paths, struct literals, and `const_shape!` / `const_array!` macros.

## Trait dispatch (`trait_dispatch.rs`)

Any function annotated with `#[cuda_tile::variadic_op(...)]`, or any trait declared with `#[cuda_tile::variadic_trait(...)]`, is treated as rank-polymorphic. Instead of emitting per-rank-mangled callables, the macro produces a single CGA-erased rank-polymorphic trait, per-rank `impl`s of it, and (for free fns) a free-fn wrapper that delegates through the trait. User code calls the unsuffixed name and rustc resolves it via normal trait lookup.

Example — `addf` (same-shape case):

```rust
// Source
#[cuda_tile::variadic_op(N = 6)]
pub fn addf<E: ElementType, const S: [i32; N], R: rounding::Mode, F: ftz::Mode>(
    lhs: Tile<E, S>, rhs: Tile<E, S>, rounding: R, ftz: F,
) -> Tile<E, S>;

// Macro output (abridged — trait + wrapper shown; 7 per-rank impls elided)
trait Addf<R, F> {
    fn addf(self, rhs: Self, rounding: R, ftz: F) -> Self;
}
impl<E: ElementType, R: rounding::Mode, F: ftz::Mode, const S_0: i32>
    Addf<R, F> for Tile_1<E, S_0>
{ fn addf(self, rhs: Self, rounding: R, ftz: F) -> Self { unreachable!() } }
// ... ranks 0..=6

pub fn addf<__T, R, F>(lhs: __T, rhs: __T, rounding: R, ftz: F) -> __T
where __T: Addf<R, F> { lhs.addf(rhs, rounding, ftz) }
```

User code writes the call naturally; rustc resolves it through the trait:

```rust
fn kernel<const S: [i32; 2]>(x: Tile<f32, S>, y: Tile<f32, S>) -> Tile<f32, S> {
    addf(x, y, rounding::NearestEven, ftz::Disabled)
}
```

### Signature patterns the emitter handles

The emitter recognizes three shapes for the return type and emits the right trait form for each:

- **Same-shape (case 3a).** Return matches the first shape-bearing arg. Method returns `Self`. Examples: `addf`, `subf`, unary math, `fma`, comparisons that preserve shape.
- **Bound return (case 3b).** Return differs from `Self` but every generic in the return is also referenced by some argument. Method returns `Self::Out` with an associated type. Examples: `reshape`, `broadcast`, `constant`, comparisons with `bool` return.
- **Free return (case 3c).** Return contains a generic not present in any argument (an associated type would break coherence). `Out` becomes a trait generic that the caller ascribes at the return site. Examples: `permute`, `reduce_sum`, `bitcast`, `exti`.

The emitter also handles: nested-type recursion (`Option<Tile<bool, S>>`), reborrow on reference receivers (`&Tensor`, `&mut Tensor`), return-only lifetimes (lifted to trait generics), rank-dependent argument types (`idx: [i32; N]` promoted to a trait generic), and literal-CGA patterns (`Tile<E, {[]}>` → `Tile_0<E>`).

## Debugging

Generate backtraces on macro panics:

```bash
export RUSTFLAGS="-Zproc-macro-backtrace -Zmacro-backtrace"
export RUST_BACKTRACE=1
```

Dump generated kernel launcher code to a directory:

```bash
export DUMP_KERNEL_LAUNCHER_DIR="temp"
```

Inspect macro-emitted code for a specific module:

```bash
cargo expand -p cutile --lib
```

## Testing

```bash
cargo test -p cutile-macro
```
