# Benchmarks

Criterion benchmarks for selected cuTile Rust kernels and workloads.

## Quick start

```bash
# Smoke test: verify all benchmarks execute (fast, discards results)
./scripts/test_benchmarks.sh

# Full benchmark: accurate measurements saved as baseline
cargo bench -p cutile-benchmarks

# Run a single benchmark
cargo bench -p cutile-benchmarks -- softmax
```

## Smoke test

The `smoke-test` feature runs each benchmark with minimal iterations
(`sample_size=10`, `measurement_time=1ms`) and discards results so they
don't pollute criterion baselines. Use it in CI or during development to
catch compilation and runtime errors without waiting for full measurements.

```bash
./scripts/test_benchmarks.sh
# or equivalently:
cargo bench -p cutile-benchmarks --features smoke-test
```

## GPU clock locking

For reproducible results, lock GPU clocks before benchmarking:

```bash
./scripts/setclock.sh     # lock clocks
cargo bench -p cutile-benchmarks
./scripts/resetclock.sh   # restore defaults
```

## Reference

- Criterion: https://bheisler.github.io/criterion.rs/book/criterion_rs.html
