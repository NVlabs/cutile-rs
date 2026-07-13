# Bounds-Check Placement

Checked accesses (`unchecked_accesses = false`) are not free or expensive by
themselves — their cost is determined by *where the compiler places the
check*. The count of checks in a tile kernel is tiny; what matters is
whether a check sits inside a hot loop, between loads and the compute that
consumes them, where it can defeat software pipelining. The compiler
therefore places each check at the best position it can prove correct, and
this chapter states the placement rules as a contract you can write code
against, plus the tools to inspect the decision instead of guessing.

Every checked access dimension gets one of three outcomes. **Discharged**:
the compiler proves the access in bounds at compile time and emits nothing —
stores through `iter_indices()` indices, loads indexed by a partition's own
minted coordinates, constant indices against static shapes, and facts
established by `preconditions` or `with_bounds` all land here. **Hoisted**:
the check runs once, before the loop (or the outermost provable loop), off
the hot path. **In place**: the check runs at the access, every iteration.

## When a Check Hoists

A check on an access inside a loop is hoisted to before the loop when all of
the following hold, per index coordinate:

| Index coordinate form | Placement |
|---|---|
| Value computed before the loop | Hoisted |
| Compile-time constant, or value with known constant bounds | Hoisted (or discharged against static shapes) |
| The loop variable `j` of a `for j in lo..hi` loop | Hoisted, checked at `hi - 1` |
| `a * j + b` with constant `a`, `b` | Hoisted, checked at the extreme iteration |
| Loop variable of a `.step_by(...)` loop | In place |
| Value computed *inside* the loop body (other than the forms above) | In place |
| Any access written inside an `if`/`else` in the loop | In place |

Additionally, a hoisted check keeps climbing outward through directly
nested loops whose trip counts are statically non-zero, stopping at the
first loop whose bound it depends on. Hoisted checks are guarded so that a
loop which executes zero times can never trap — semantics are unchanged
except that a genuine violation traps before the loop instead of at its
first offending iteration.

The practical rules of thumb that fall out of the table:

- **Compute index arithmetic above the innermost loop.** `let kv_head =
  q_head / group;` written before the K/V loop hoists every check that uses
  it; the same expression written inside the loop body does not (the
  compiler does not currently chase invariant arithmetic through the loop
  body — it proves invariance by position).
- **Index hot-loop accesses with the loop variable directly**, or an
  affine expression of it, and write the loop as `for j in lo..hi` without
  `step_by`.
- **Keep hot-loop accesses unconditional.** A load under an `if` may
  execute on no iteration, so its check cannot move; lift the condition out
  of the loop or accept the in-place check.

```rust
for index in out.iter_indices() {
    let (q_tile, q_head, _) = index.components();
    let kv_head = q_head / GROUP;          // above the loop: hoists
    for j in 0i32..kv_tiles {
        let k = k_part.load_pipelined::<L>([kv_head, j, 0i32]);
        //                                  ^ hoisted  ^ hoisted (checked at kv_tiles - 1)
        // ...
    }
}
```

## Reading the Compiler's Decision

Never tune by guessing — the placement of every check is observable.
`CUTILE_JIT_TIMING=1` reports per-kernel totals on each compile line:

```text
CUTILE_JIT_TIMING module=kernels function=fmha_prefill ... \
    checks_discharged=3 checks_hoisted=4 checks_in_place=2
```

`CUTILE_JIT_LOG=1` explains each check that stays in a loop body:

```text
[cutile::jit] bounds check for dim 1 stays in the loop body: index is
computed inside the loop body
```

And `CUTILE_DISABLE_CHECK_HOISTING=1` pins every check at its access site,
so you can measure what hoisting is worth on your kernel with two runs of
the same binary — no rebuild.

## When to Reach for `unsafe`

Measure before dropping a kernel to `unchecked_accesses = true`: after
hoisting, the answer is usually "it buys almost nothing." On the
flash-attention prefill kernel that motivated this machinery (RTX 5090,
`checks_in_place=2`), the fully checked kernel runs at 55.0 µs/call against
a 53.6 µs floor with all checks disabled — about 2.5%, all of it from the
two in-place checks on schedule-derived coordinates that execute once per
persistent index, not per inner-loop iteration. The unsafe twin of the same
kernel runs at 56.7 µs; the checked version is faster.

The cases where `unsafe` still pays are the in-place rows of the table
above when they land in a genuinely hot loop: stepped-loop indices,
data-dependent indices (values loaded from memory), and conditional
accesses. If `CUTILE_JIT_LOG` shows in-place checks in your inner loop and
restructuring per the rules above can't move them, that — and only that —
is the measured case for `unchecked_accesses = true`.
