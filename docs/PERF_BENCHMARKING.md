# Performance Counter Benchmarking

This document explores options for integrating hardware performance counters
(cache misses, branch mispredictions, etc.) into the heap benchmarking suite.

## Motivation

Wall-clock time benchmarks (via Criterion) tell us *how fast* each heap is, but
not *why*. Hardware counters can reveal:

- **Cache misses**: L1/L2/L3 data cache misses indicate poor memory locality
- **Branch mispredictions**: High misprediction rates suggest unpredictable
  control flow
- **Instructions per cycle (IPC)**: Low IPC often indicates memory stalls
- **TLB misses**: Page table overhead from scattered memory access

For heap data structures, cache behavior is often the dominant factor. Heaps
like Fibonacci and Hollow use pointer-chasing which can cause poor cache
locality, while array-based heaps (binary heap) have better spatial locality.

## Options Evaluated

### 1. iai-callgrind (Recommended for CI)

**Pros:**

- Deterministic instruction counts (no noise)
- Works in virtualized CI environments
- Measures cache simulation via Callgrind
- Generates flamegraphs for detailed analysis
- Stable, well-maintained fork of iai

**Cons:**

- Uses Valgrind simulation, not real hardware counters
- Slower than native execution (10-50x)
- Linux-only (no Windows/macOS)
- Cache simulation may not match real CPU behavior exactly

**Best for:** CI regression detection, consistent cross-system comparisons

### 2. perf-event / perf-event2 (Recommended for Local Analysis)

**Pros:**

- Real hardware counters via Linux perf_event_open
- Measures actual cache misses, branch mispredictions
- Low overhead when counters are running
- Can measure any perf event the CPU supports

**Cons:**

- Linux-only
- Requires permissions (CAP_PERFMON or perf_event_paranoid <= 1)
- Results vary between runs due to system noise
- Not integrated with Criterion directly

**Best for:** Deep local analysis, understanding *why* something is slow

### 3. perfcnt Crate

Similar to perf-event but older API. perf-event2 is more actively maintained.

### 4. Criterion with External Profiling

Run Criterion benchmarks under `perf stat` or `perf record`:

```bash
perf stat -e cache-misses,cache-references,branches,branch-misses \
    cargo bench -- --bench dimacs_benchmark
```

**Pros:** No code changes needed
**Cons:** Measures entire benchmark including setup/teardown

## Recommended Approach

### Phase 1: Add perf-event Feature (Local Analysis)

Add an optional feature that enables hardware counter collection during
benchmarks. This won't replace Criterion but will provide additional metrics.

```toml
[features]
perf-counters = ["perf-event2"]

[target.'cfg(target_os = "linux")'.dependencies]
perf-event2 = { version = "0.7", optional = true }
```

### Phase 2: Add iai-callgrind Benchmarks (CI)

Create a separate benchmark file using iai-callgrind for instruction-level
metrics that can run in CI:

```toml
[dev-dependencies]
iai-callgrind = "0.14"

[[bench]]
name = "heap_iai"
harness = false
```

### Phase 3: Custom Perf Harness

Create a custom benchmark harness that wraps operations with perf counters:

```rust
#[cfg(feature = "perf-counters")]
pub fn measure_with_perf<F, R>(f: F) -> (R, PerfMetrics)
where
    F: FnOnce() -> R,
{
    use perf_event::{Builder, Group};
    use perf_event::events::{Hardware, Cache, CacheOp, CacheResult};

    let mut group = Group::new().unwrap();
    let cycles = Builder::new().group(&mut group).kind(Hardware::CPU_CYCLES).build().unwrap();
    let instructions = Builder::new().group(&mut group).kind(Hardware::INSTRUCTIONS).build().unwrap();
    let cache_refs = Builder::new().group(&mut group).kind(Hardware::CACHE_REFERENCES).build().unwrap();
    let cache_misses = Builder::new().group(&mut group).kind(Hardware::CACHE_MISSES).build().unwrap();
    let branches = Builder::new().group(&mut group).kind(Hardware::BRANCH_INSTRUCTIONS).build().unwrap();
    let branch_misses = Builder::new().group(&mut group).kind(Hardware::BRANCH_MISSES).build().unwrap();

    group.enable().unwrap();
    let result = f();
    group.disable().unwrap();

    let counts = group.read().unwrap();
    let metrics = PerfMetrics {
        cycles: counts[&cycles],
        instructions: counts[&instructions],
        cache_references: counts[&cache_refs],
        cache_misses: counts[&cache_misses],
        branches: counts[&branches],
        branch_misses: counts[&branch_misses],
    };

    (result, metrics)
}
```

## Metrics to Collect

| Metric | Event | Interpretation |
|--------|-------|----------------|
| Instructions | INSTRUCTIONS | Work done |
| Cycles | CPU_CYCLES | Time spent |
| IPC | instructions/cycles | Efficiency (higher = better) |
| L1D Cache Misses | L1D_CACHE/MISS | Local data locality |
| LLC Misses | CACHE_MISSES | Memory bandwidth pressure |
| Branch Misses | BRANCH_MISSES | Control flow predictability |
| TLB Misses | DTLB_MISSES | Memory access pattern |

## Expected Insights

Based on heap structure, we expect:

| Heap | Expected Cache Behavior |
|------|------------------------|
| SimpleBinaryHeap | Good - array-based, sequential access |
| FibonacciHeap | Poor - pointer-chasing, scattered nodes |
| HollowHeap | Poor - similar to Fibonacci |
| PairingHeap | Moderate - tree structure but simpler |
| RadixHeap | Good - array-based buckets |
| SkipListHeap | Moderate - some locality in levels |

## Implementation Plan

1. [ ] Add `perf-counters` feature flag
2. [ ] Create `src/bench_utils.rs` with perf measurement helpers
3. [ ] Add `benches/heap_perf.rs` for perf-based benchmarks
4. [ ] Add iai-callgrind benchmarks in `benches/heap_iai.rs`
5. [ ] Create comparison script to correlate metrics with wall-clock time
6. [ ] Document how to run perf benchmarks (permissions, kernel settings)

## Running Perf Benchmarks

### Prerequisites (Linux)

```bash
# Option 1: Set perf_event_paranoid (temporary)
sudo sysctl kernel.perf_event_paranoid=1

# Option 2: Add CAP_PERFMON capability (permanent for binary)
sudo setcap cap_perfmon+ep ./target/release/deps/heap_perf-*

# Option 3: Run as root (not recommended)
sudo cargo bench --features perf-counters
```

### Running

```bash
# Standard Criterion benchmarks (wall-clock time)
cargo bench

# Perf counter benchmarks (cache misses, etc.)
cargo bench --features perf-counters --bench heap_perf

# iai-callgrind benchmarks (instruction counts)
cargo bench --bench heap_iai
```

## References

- [perf-event2 crate](https://crates.io/crates/perf-event2)
- [iai-callgrind](https://github.com/iai-callgrind/iai-callgrind)
- [Linux perf_event_open man page](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [Brendan Gregg's perf examples](https://www.brendangregg.com/perf.html)
