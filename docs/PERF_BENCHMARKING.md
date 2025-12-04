# Performance Counter Benchmarking

This document describes the hardware performance counter benchmarking
infrastructure and presents results from Dijkstra shortest-path benchmarks.

## Overview

The benchmarks measure heap performance in a realistic pathfinding workload
using Dijkstra's algorithm on synthetic sparse graphs. We capture:

- **Wall-clock time**: Total execution time
- **IPC (Instructions Per Cycle)**: CPU efficiency (higher = better)
- **LLC Miss Rate**: Last-Level Cache (L3) miss percentage

## Running Benchmarks

### Prerequisites (Linux)

```bash
# Enable perf counters for non-root users
sudo sysctl kernel.perf_event_paranoid=1
```

### Quick Start

```bash
# Run the multi-metric table benchmark
cargo bench --features perf-counters --bench perf_table

# Run with CPU pinning for stable results
BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench perf_table

# Run specific rank sizes in parallel
BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench perf_table -- 8 &
BENCH_PIN_CPU=1 cargo bench --features perf-counters --bench perf_table -- 12 &
BENCH_PIN_CPU=2 cargo bench --features perf-counters --bench perf_table -- 16 &
BENCH_PIN_CPU=3 cargo bench --features perf-counters --bench perf_table -- 20 &
wait
```

## Benchmark Results

Benchmarks run on a 2M node synthetic sparse graph (avg degree 6). The "rank"
is the Dijkstra rank - how many nodes are settled before finding the target.

### 2^8 (256 nodes settled) - Fits in L1/L2 Cache

| Algorithm | Time | IPC | LLC Miss% |
| --- | ---: | ---: | ---: |
| simple_binary | 3.2ms | 1.36 | 12.6 |
| strict_fib_opt | 6.6ms | 1.71 | 8.5 |
| twothree_opt | 6.8ms | 1.75 | 9.8 |
| fibonacci_lazy | 7.2ms | 1.71 | 10.5 |
| pairing_lazy | 8.2ms | 1.56 | 13.6 |
| skew_binom_opt | 8.2ms | 1.50 | 12.2 |
| pairing_opt | 8.7ms | 1.48 | 13.5 |
| strict_fib_lazy | 8.9ms | 1.28 | 12.2 |
| fibonacci_opt | 9.2ms | 1.34 | 16.3 |
| twothree_lazy | 9.4ms | 1.28 | 17.6 |
| hollow_opt | 9.7ms | 1.95 | 12.7 |
| skew_binom_lazy | 10.6ms | 1.18 | 15.9 |
| binomial_opt | 11.0ms | 1.47 | 12.9 |
| rank_pair_lazy | 11.3ms | 1.67 | 11.2 |
| hollow_lazy | 11.9ms | 1.37 | 13.5 |
| binomial_lazy | 12.9ms | 1.26 | 13.8 |
| skiplist_lazy | 20.1ms | 1.48 | 14.0 |

### 2^12 (4096 nodes settled) - Fits in L3 Cache

| Algorithm | Time | IPC | LLC Miss% |
| --- | ---: | ---: | ---: |
| simple_binary | 75.6ms | 0.96 | 22.6 |
| pairing_lazy | 132.8ms | 1.58 | 28.1 |
| strict_fib_lazy | 145.6ms | 1.31 | 32.7 |
| strict_fib_opt | 146.7ms | 1.31 | 34.9 |
| pairing_opt | 150.7ms | 1.43 | 28.5 |
| twothree_opt | 151.2ms | 1.33 | 30.9 |
| twothree_lazy | 151.2ms | 1.35 | 31.1 |
| fibonacci_lazy | 155.6ms | 1.34 | 34.7 |
| fibonacci_opt | 166.0ms | 1.25 | 36.1 |
| skew_binom_lazy | 167.9ms | 1.24 | 30.0 |
| skew_binom_opt | 176.1ms | 1.21 | 30.8 |
| hollow_lazy | 187.9ms | 1.50 | 29.5 |
| rank_pair_lazy | 204.5ms | 1.59 | 30.9 |
| binomial_lazy | 205.0ms | 1.56 | 29.1 |
| binomial_opt | 207.8ms | 1.55 | 30.5 |
| hollow_opt | 237.5ms | 1.37 | 37.6 |
| skiplist_lazy | 334.7ms | 1.48 | 26.6 |

### 2^16 (65536 nodes settled) - Exceeds L3 Cache

| Algorithm | Time | IPC | LLC Miss% |
| --- | ---: | ---: | ---: |
| simple_binary | 2.15s | 0.60 | 37.0 |
| pairing_opt | 3.97s | 0.87 | 40.1 |
| pairing_lazy | 3.97s | 0.89 | 40.0 |
| twothree_lazy | 4.37s | 0.77 | 42.9 |
| hollow_lazy | 4.54s | 1.04 | 37.2 |
| fibonacci_lazy | 4.65s | 0.73 | 47.4 |
| strict_fib_opt | 4.65s | 0.69 | 44.9 |
| skew_binom_lazy | 4.69s | 0.77 | 43.4 |
| strict_fib_lazy | 4.70s | 0.71 | 44.5 |
| twothree_opt | 4.75s | 0.69 | 44.4 |
| skew_binom_opt | 4.99s | 0.72 | 44.5 |
| fibonacci_opt | 5.02s | 0.67 | 49.4 |
| rank_pair_lazy | 5.27s | 1.05 | 41.2 |
| binomial_lazy | 5.28s | 1.15 | 40.5 |
| binomial_opt | 5.59s | 1.08 | 43.2 |
| hollow_opt | 6.56s | 0.85 | 49.9 |
| skiplist_lazy | 7.93s | 0.98 | 31.8 |

## Key Findings

### 1. Simple Binary Heap Dominates

Despite O(log n) `decrease_key` vs O(1) amortized for Fibonacci heap, simple
binary heap is 2-3x faster across all sizes. The cache locality of a
contiguous array beats the theoretical advantage of pointer-based structures.

### 2. IPC Drops Dramatically at Scale

| Size | simple_binary IPC | Advanced heaps IPC |
| --- | ---: | ---: |
| 2^8 | 1.36 | 1.3-1.9 |
| 2^12 | 0.96 | 1.2-1.6 |
| 2^16 | 0.60 | 0.67-1.15 |

As working set exceeds cache, the CPU spends more cycles waiting for memory.

### 3. LLC Miss Rates Correlate with Performance

| Size | LLC Miss Rate |
| --- | ---: |
| 2^8 | 8-17% |
| 2^12 | 22-37% |
| 2^16 | 37-50% |

The transition from 2^12 to 2^16 crosses the L3 cache boundary, causing
dramatic slowdowns.

### 4. Lazy vs Decrease-Key is a Wash

Contrary to theory, `pairing_lazy` (re-insertion) ties or beats `pairing_opt`
(decrease_key) at all sizes. Possible explanations:

- Re-insertion has better cache behavior (no pointer manipulation)
- The constant factors in decrease_key are higher than expected
- Graph structure favors certain access patterns

### 5. Skiplist is Consistently Slowest

SkipList heap is ~2x slower than other heaps at all sizes, likely due to its
probabilistic structure and multi-level pointer chasing.

## Metrics Collected

| Metric | Event | What it Measures |
| --- | --- | --- |
| Instructions | INSTRUCTIONS | CPU instructions retired |
| Cycles | CPU_CYCLES | CPU clock cycles |
| IPC | instructions/cycles | CPU efficiency |
| LLC Refs | CACHE_REFERENCES | Last-level cache accesses |
| LLC Misses | CACHE_MISSES | LLC misses (went to DRAM) |

## Architecture

### perf-measurement Crate

Located in `crates/perf-measurement/`, provides:

- `PerfMultiMeasurement`: Criterion `Measurement` implementation
- Captures time + hardware counters atomically via `perf_event::Group`
- Thread-local accumulation for periodic stats output

### perf_table Benchmark

Located in `benches/perf_table.rs`, provides:

- Direct multi-metric output (bypasses Criterion's single-value limitation)
- Command-line filtering for parallel execution by rank
- CSV output for easy parsing

### CPU Pinning

Set `BENCH_PIN_CPU=N` to pin the benchmark to CPU N. This reduces noise from
CPU migration and improves result stability.

## Questions for Further Investigation

1. **Crossover point**: At what size (if any) do advanced heaps beat binary
   heap?
2. **Lazy competitiveness**: Why does re-insertion compete with O(1)
   decrease_key?
3. **Graph sensitivity**: Would different topologies (grid, scale-free, road
   networks) change results?
4. **Node size**: How does heap node size affect cache behavior?

## References

- [perf-event crate](https://crates.io/crates/perf-event)
- [Linux perf_event_open man page](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
- [Brendan Gregg's perf examples](https://www.brendangregg.com/perf.html)
