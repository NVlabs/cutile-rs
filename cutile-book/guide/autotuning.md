# Autotuning

`cutile::tune` searches a declared space of kernel configurations, measures
each candidate with `cutile::bench`'s device-event timing, and persists both
the trials and the winner. It is gated behind the `experimental-tune`
feature, and the API may change between releases:

```toml
cutile = { version = "...", features = ["experimental-tune"] }
```

Two workflows share the machinery: a closure front-end for tuning one
kernel in isolation, and a caller-implemented `Objective` for engine-scale
objectives where "fast" means the whole request, not one launch. The
`autotune` example runs the closure workflow end to end:
`cargo run -p cutile-examples --example autotune --features experimental-tune`.

## Declaring a space and running the search

A configuration is a set of named integer or string parameters; the space
is the list of configurations. Only `configs` and the terminal `run` are
required. The rest of the builder is optional: `prune` filters candidates,
`budget` bounds wall-clock, `log` makes the run resumable, and `require`
declares incumbents the search must cover.

```rust
let configs: Vec<Config> = [16i64, 32, 64, 128]
    .into_iter()
    .map(|bn| Config::new([("BN", ParamValue::Int(bn))]))
    .collect();

let output = Autotuner::new("fmha_decode")
    .configs(configs)
    .prune(|c| c.int("BN").unwrap() <= pp)
    .require(vec![shipping_config])
    .budget(Duration::from_secs(300))
    .arch("sm_100") // record the arch; a resume from another arch is refused
    .log("fmha_decode.trials.jsonl")
    .run(&stream, |stream, config| {
        // The setup closure is yours: read the config, build the kernel's
        // composed launcher for it, run it once as a correctness gate, and
        // return the closure to be timed. Err(..) records the candidate as
        // invalid and the search continues.
        let bn = config.int("BN").unwrap();
        let mut launch = move |stream: &Arc<Stream>| {
            fmha_decode(out.partition([1, bn as i32]), &q, &k, &v)
                .generics(generics_for(bn))
                .sync_on(stream)
        };
        launch(&stream)?;
        Ok(launch)
    })?;
let best: Config = output.best.expect("a winner");
```

Three behaviors are worth knowing rather than discovering. The winner is
decided by a paired A/B runoff between the two best candidates, because
sequential medians drift with clocks and temperature; a finalist that fails
its runoff setup forfeits. A `require`d configuration must be a member of
the space (a missing incumbent is an error, not a silent omission) and is
measured before the searcher runs, so no budget cutoff can skip it — the
winner always beat, or is, every incumbent. And the trial log is headed by
tuner name and a hash of the space: a log written by a different tuner or
space is refused, never silently adopted.

## Engine-scale measurement implements Objective

When the objective is end-to-end (tokens per second, request latency), the
library cannot own the launch. Implement `Objective` instead: `configs()`
exposes the space, `measure(index)` applies one candidate at engine scope
and returns a `Trial` built with `Trial::measured` or `Trial::invalid`, and
`budget_remaining()` reports what is left. `Autotuner::run_objective` then
provides the same trial logging, resume, and `require` coverage as the
closure path:

```rust
let mut objective = EngineObjective::new(engine, configs);
let output = Autotuner::new("engine_prefill")
    .arch(current_arch) // e.g. "sm_100"; resume refuses a foreign-arch log
    .log("prefill.trials.jsonl")
    .require(vec![current_default])
    .run_objective(&mut objective)?;
```

There is no paired runoff on this path — re-timing finalists requires the
library's bench closures, which an engine objective does not expose —
so `Output::best` is the best sequential median and contemporaneous
re-measurement of finalists, if wanted, stays with the caller. The
serialized `Trial` form is a stable contract: resumable logs survive cutile
upgrades, and an incompatible change bumps the log header's schema, which
`TrialLog::open` refuses rather than silently discarding resume state.


## Committing winners

A `tune::Record` persists winners so production loads them instead of
re-searching: one JSON file per kernel, written by `save` and committed
next to the code it tunes. The file holds a provenance header (kernel
name, source hash, cutile version, `tileiras` fingerprint, architecture,
search-space hash) and one entry per shape-class bucket.

```rust
use cutile::tune::{L2Key, Record, RecordEntry, Workspace};

// After tuning: `winner` is the winning Trial, `best` its Config
// (`output.best`), and `launcher` the kernel's composed builder applied
// with `best`'s values — the same call expression production dispatches.
let TrialState::Measured { median_ms, reps, .. } = winner.state else {
    unreachable!("the winner was measured");
};
let mut record = Record::new(&workspace);
record.insert(RecordEntry {
    bucket: "tg<=512".into(),
    config: best,
    median_ms,
    samples: reps,
    l2_key: Some(L2Key::current(launcher.l2_cache_key()?)),
});
record.save(&path)?;

// In production: load, verified against the running workspace.
let (record, warnings) = Record::load_verified(&path, &workspace, |entry| {
    Ok(Some(specialize_for(&entry.config)?.l2_cache_key()?))
})?;
let entry = record.get("tg<=512");
```

A bucket is a plain string label, matched exactly by `Record::get`; the
predicate that decides which bucket a runtime shape falls into lives in
the consumer's dispatch code, and the record only stores the label.
`median_ms` and `samples` are the winner's measured median latency and
the number of timed reps behind it — provenance for a human reading the
committed file, not inputs to verification. The `l2_key` is the winner's
persistent-cache key, computed from the composed builder without
compiling or launching (`.l2_cache_key()`).

`load_verified` refuses a record whose kernel, architecture, source hash,
or search space does not match the running workspace, and refuses any
entry whose stored cache key no longer matches the recomputed one — that
last check covers the kernel's dependencies and the toolchain, so a stale
winner fails loudly instead of applying silently. Drift that only shifts
timings comes back as warnings. A record either verifies or it does not
load; there is no best-effort application.

## Warming and managing the kernel cache during sweeps

Warm exactly what you dispatch: the compile-only `.compile()` terminal sits
on the same composed builder as `.execute` and `.sync_on`, so warmup uses
the dispatch call expression itself and cannot drift from it. `api::meta`
placeholder tensors carry shape and dtype without allocating, so warming
performs no launches and no device allocation:

```rust
my_module::my_kernel(api::meta::<f32>(&[64, 64]).sync()?.partition([16, 16]), ...)
    .generics(generics)
    .compile_options(opts)
    .compile()?;
```

Tuning sweeps churn specializations by design, and each cached kernel holds
device memory, so the in-memory kernel cache — intentionally unbounded for
steady-state engines — needs managing during a sweep. The `unsafe`
functions `clear_kernel_cache()`, `evict_kernel(&key)`, and
`retain_kernels(pred)` in `cutile::tile_kernel` (gated behind
`experimental-tune`) remove entries, releasing each module's device memory
when its last holder drops. They are `unsafe` because of the one obligation
they cannot check: quiesce first. A launched kernel executes after the launch
call returns, so synchronize any stream that may still be running cached
kernels — between tuning trials, exactly where an objective already
synchronizes.
