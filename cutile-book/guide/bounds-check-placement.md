# Bounds-Check Placement

Accesses through the safe partition API are checked by default — no
attribute needed — and checks are not free or expensive by themselves:
their cost is determined by *where the compiler places them*. The count of
checks in a tile kernel is tiny; what matters is whether a check sits
inside a hot loop, between loads and the compute that consumes them, where
it can defeat software pipelining. The compiler therefore places each check
at the best position it can prove correct, and this chapter states the
placement rules as a contract you can write code against, plus the tools to
inspect the decision instead of guessing.

Every checked access dimension gets one of three outcomes. **Discharged**:
the compiler proves the access in bounds at compile time and emits nothing
in the kernel — stores through `iter_indices()` indices, loads indexed by a
partition's own minted coordinates, constant indices against static shapes,
and facts established by declared `preconditions` all land here, as do
cross-tensor facts the compiler derives and verifies at launch. **Hoisted**:
the check runs once, before the loop (or the outermost provable loop), off
the hot path. **In place**: the check runs at the access, every iteration.

Setting `deny_in_kernel_checks = true` on the entry turns the third outcome
into a compile error: "every check left the kernel" becomes a contract the
build enforces rather than a property you audit. The diagnostic names the
access and the restructuring that would fix it.

## When a Check Hoists

A check on an access inside a loop is hoisted to before the loop when all of
the following hold, per index coordinate:

| Index coordinate form | Placement |
|---|---|
| Value computed before the loop | Hoisted |
| Compile-time constant, or value with known constant bounds | Hoisted (or discharged against static shapes) |
| The loop variable `j` of a `for j in lo..hi` loop | Hoisted, checked at `hi - 1` |
| `a * j + b` with constant `a`, `b` | Hoisted at the extreme iteration, when that extreme provably fits `i32`; otherwise in place |
| Loop variable of a `(lo..hi).step_by(k)` loop | Hoisted, checked at the *last attained* index `lo + k * floor((hi - 1 - lo) / k)`, when `lo`, `hi`, and `k` are compile-time constants; otherwise in place |
| Value computed *inside* the loop body (other than the forms above) | In place |
| Any access written inside an `if`/`else` in the loop | In place |
| Any access in a loop body that contains a `continue` | In place |

Additionally, a hoisted check keeps climbing outward through directly
nested loops whose trip counts are statically non-zero and whose bodies
contain no `continue`, stopping at the first loop whose bound it depends
on. Hoisted checks are guarded so that a loop which executes zero times can
never trap. Hoisting does change *when* a violation is reported: a hoisted
check traps before the loop starts, so the iterations that precede the
offending one — and any stores they would have made — do not run. It never
changes *whether* a kernel traps: a hoisted check tests only index values
the loop actually attains, which is why a body with `continue` (some
iterations skip the access) or a stepped loop with a runtime step (the
attained set is unknown) keeps its check in place.

The practical rules of thumb that fall out of the table:

- **Compute index arithmetic above the innermost loop.** `let kv_head =
  q_head / group;` written before the K/V loop hoists every check that uses
  it; the same expression written inside the loop body does not (the
  compiler does not currently chase invariant arithmetic through the loop
  body — it proves invariance by position).
- **Index hot-loop accesses with the loop variable directly**, or an
  affine expression of it, and write the loop as `for j in lo..hi`; a
  `step_by(k)` loop hoists only when `lo`, `hi`, and `k` are all
  compile-time constants.
- **Keep hot-loop accesses unconditional.** A load under an `if` may
  execute on no iteration, so its check cannot move; lift the condition out
  of the loop or accept the in-place check. The same holds for a body with
  a `continue`: every access after it is conditional.
- **Keep index arithmetic wrap-free.** A range fact survives an operation
  only when the operation provably cannot overflow `i32`; an expression
  that can wrap forfeits its facts (even if later `max`/`%` steps pull the
  mathematical range back in bounds), and the access pays an in-place check
  over the actual runtime value.

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

## Checks That Leave the Kernel Entirely

Hoisting to the loop preheader is not the last stop. A check whose operands
are all launch-known — tensor extents, never loop variables or loaded values —
leaves the kernel altogether and becomes one host comparison in the generated
launcher, evaluated against the real shapes before the kernel exists. It costs
zero registers and cannot be reordered or pipelined against: it either passes
once or the launch is refused with an error naming the check.

The main client is a fact the compiler derives on its own. An index that
walks one tensor's axis — the variable of a `for j in 0..num_tiles(&p, a)`
loop, or a component of a mapped partition's `iter_indices()` — carries
that provenance, and using it to index a *different* tensor turns the
cross-tensor tie into a launch check over tile counts, for example
`ceil(dim(x, 1)/BK) <= ceil(dim(y, 0)/BK)`. No annotation and no signature
change: a persistent GEMM whose mapped components index `x` and `y` and
whose k loop iterates `num_tiles` compiles with every check discharged from
the kernel and its shape ties enforced at launch
(`cutile-examples/examples/persistent_gemm.rs` is the canonical form, and
it builds under `deny_in_kernel_checks = true`).

Declared `preconditions` go one step further: the launcher already verifies
each declared fact against the real shapes, so the compiler assumes it and
discharges the matching checks at compile time. A kernel that declares
`dim(x, 1) % 64 == 0` emits nothing anywhere for the matching binding, and
`dim(a, i) == dim(b, j)` relates two tensors' axes exactly as the derived
form does. Declare a precondition when you want the contract visible in the
signature or need a fact the walk cannot derive; note the declared equality
is stricter than the derived comparison (`==` on extents versus `<=` on
tile counts), so it rejects some launches the derived form accepts.

The `with_bounds`/`Dim` annotation family is deprecated: everything it
proved is subsumed by the derived facts and declared preconditions above,
with the checks landing at launch instead of possibly in the kernel.

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

Counts summarize; the emitted Tile IR is the ground truth, and comparing it
against the unsafe twin catches a scheduling regression before you run
anything. Make the twin in three edits — `unchecked_accesses = true` on the
entry, `unsafe` on the kernel fn, and `unsafe { ... }` around the launch
call, since the generated launcher inherits the kernel's unsafety — then
dump both variants (the dump prints to stderr) and diff:

```bash
dump() { CUTILE_DUMP=ir CUTILE_DUMP_FILTER=kernels::fmha_prefill ./my_app 2>&1 \
         | sed -n '/=== CUTILE DUMP: ir/,/^}/p'; }
dump > safe.ir
# flip the kernel to its unsafe twin, then:
dump > raw.ir
diff safe.ir raw.ir
```

For a fully placed kernel the diff is **empty**: the persistent GEMM
example (`cutile-examples/examples/persistent_gemm.rs`) dumps byte-identical
IR in its safe and unsafe variants — same 103 ops, same loop bodies — with
the entire safety contract living in the launcher. When the diff is not
empty, two things are regressions worth stopping for. Any `assert` op means
a check stayed in the kernel (`grep -c assert safe.ir` should be zero when
you expect full placement — and `deny_in_kernel_checks = true` makes the
build enforce that). Any extra op *inside a loop body* — a compare, select,
or branch sitting between the loads and the `mma`-class ops that consume
them — is the pipelining hazard this chapter exists to prevent, even when
the totals look small. A hoisted check appearing before a loop is the
accepted middle ground, not a regression. If the IR matches but performance
still differs, compare register counts next; the checked and unchecked
builds of a fully placed kernel should match exactly.

Two ablation knobs let you measure placement with the same binary, no
rebuild: `CUTILE_DISABLE_CHECK_HOISTING=1` pins every residual check at its
access site, and `CUTILE_FORCE_DEVICE_CHECKS=1` additionally suppresses
every proof, checking each access two-sided over its actual values — the
reference semantics the test suite diffs placement against.

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
data-dependent indices (values loaded from memory), conditional accesses,
and index arithmetic the compiler cannot prove wrap-free. If
`CUTILE_JIT_LOG` shows in-place checks in your inner loop and restructuring
per the rules above can't move them, that — and only that — is the measured
case for `unchecked_accesses = true`.
