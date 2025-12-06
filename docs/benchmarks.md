# Benchmarks

This crate includes benchmarks comparing different heap implementations on
shortest path problems using DIMACS format road network graphs.

## Benchmark Strategy

### Why Shortest Path?

Dijkstra's algorithm is the canonical use case for heaps with `decrease_key`:

1. **Real-world relevance**: Shortest path is used in navigation, network
   routing, and game AI
2. **Heavy `decrease_key` usage**: Each edge relaxation potentially calls
   `decrease_key`, making it the dominant operation
3. **Scalable workloads**: Graph size directly controls problem complexity

### What We Measure

We compare heap implementations using two Dijkstra variants:

**Optimized Dijkstra** (`_opt` suffix): Uses `decrease_key` for efficient
priority updates. Requires `DecreaseKeyHeap` trait.

**Lazy Dijkstra** (`_lazy` suffix): Uses re-insertion instead of `decrease_key`.
Works with any `Heap` trait implementation.

| Heap | `decrease_key` | Variants | Why Include |
| --- | --- | --- | --- |
| **Fibonacci** | O(1) amortized | opt, lazy | Optimal theoretical bounds |
| **Pairing** | o(log n) amortized | opt, lazy | Simpler, often faster in practice |
| **Rank-Pairing** | O(1) amortized | opt, lazy | Optimal bounds, simpler than Fibonacci |
| **Simple Binary** | N/A | lazy only | Baseline comparison |

This allows comparing:

1. **Same heap, different algorithms**: e.g., `fibonacci_opt` vs `fibonacci_lazy`
2. **Different heaps, same algorithm**: e.g., `pairing_opt` vs `rank_pairing_opt`
3. **Simple vs advanced heaps**: `simple_binary_lazy` vs others

### Methodology

Our benchmarks follow the methodology from the
[9th DIMACS Implementation Challenge](http://www.diag.uniroma1.it/challenge9/)
and related literature (Sanders/Schultes Highway Hierarchies):

- **Random queries**: Uses seeded PRNG for reproducible random (source, target)
  pairs instead of fixed endpoints
- **Dijkstra rank grouping**: Queries grouped by difficulty (2^10, 2^12, 2^14,
  etc.) to show performance across local vs long-distance queries
- **Multiple queries**: Each benchmark runs batches of queries and reports
  averages, following the DIMACS standard of 1000+ queries

### Synthetic vs Real Data

We use both synthetic and real graphs:

- **Synthetic graphs** run without any setup, useful for quick iteration
- **Real DIMACS data** provides realistic road network topology and weights

Synthetic graphs are deterministic (seeded LCG PRNG) for reproducible results.

## Data File Strategy

### Directory Structure

```text
rust-advanced-heaps/
├── data/                    # Git-ignored, user downloads here
│   ├── USA-road-d.NY.gr     # New York road network
│   ├── USA-road-d.BAY.gr    # San Francisco Bay
│   └── ...
├── benches/
│   └── dimacs_benchmark.rs  # Benchmark code
└── docs/
    └── benchmarks.md        # This file
```

### Git Ignored Data

The `data/` directory is listed in `.gitignore`. This means:

- **No large files in repo**: DIMACS datasets are 10-100+ MB each
- **User downloads on demand**: Only fetch datasets you need
- **Benchmarks degrade gracefully**: Missing files are silently skipped

### Benchmark Behavior

```rust
// From benches/dimacs_benchmark.rs
for path in dimacs_paths {
    if !Path::new(path).exists() {
        continue;  // Skip missing files, run what we have
    }
    // ... run benchmark
}
```

This design lets you:

- Run `cargo bench` immediately with synthetic-only benchmarks
- Add real data incrementally as needed
- Share benchmark code without sharing large data files

## Quick Start

```bash
# Run synthetic benchmarks (no data download needed)
cargo bench

# Or use the alias for synthetic-only
cargo bench-synthetic

# Download DIMACS data and run full benchmarks
./scripts/download-dimacs.sh          # Downloads NY dataset (~12MB)
./scripts/download-dimacs.sh all      # Downloads all datasets (~260MB)
./scripts/download-dimacs.sh NY BAY   # Download specific datasets

# Run benchmarks with real data
cargo bench                           # All benchmarks
cargo bench-dimacs                    # Only real DIMACS benchmarks
cargo bench-quick                     # Quick random queries only

# List available datasets
./scripts/download-dimacs.sh --list
```

### Cargo Aliases

The project includes convenient cargo aliases in `.cargo/config.toml`:

| Alias | Description |
| --- | --- |
| `cargo bench-all` | Run all benchmarks |
| `cargo bench-synthetic` | Run only synthetic benchmarks (no download needed) |
| `cargo bench-dimacs` | Run only real DIMACS benchmarks |
| `cargo bench-quick` | Quick benchmark with random queries only |

## Benchmark Groups

### `random_queries`

Compares heaps on 100 random (source, target) queries on a 10K node synthetic
graph. This is the DIMACS-standard way to measure average query performance.

### `dijkstra_rank`

Groups queries by Dijkstra rank (number of nodes settled before reaching
target), following Sanders/Schultes methodology:

| Rank | Meaning |
| --- | --- |
| 2^10 | ~1K nodes settled (local queries) |
| 2^12 | ~4K nodes settled |
| 2^14 | ~16K nodes settled (long-distance) |

This shows how heap performance varies with query difficulty. Local queries
(low rank) may favor simpler heaps, while long-distance queries (high rank)
benefit from efficient `decrease_key`.

### `graph_scale`

Tests performance across different graph sizes: 5K, 10K, 20K, 50K nodes.
Shows scaling behavior and when theoretical complexity advantages become
practical.

### `real_dimacs`

Benchmarks on real road network data (if present in `data/`):

- `USA-road-d.NY.gr` - New York (~264K nodes, ~730K edges)
- `USA-road-d.BAY.gr` - San Francisco Bay (~321K nodes, ~800K edges)
- `USA-road-d.COL.gr` - Colorado (~436K nodes, ~1M edges)

Uses 100 random queries following DIMACS methodology.

### `real_dimacs_by_rank`

Dijkstra rank analysis on real NY road network data (2^12, 2^14, 2^16, 2^18).

### `correctness`

Quick sanity check that pathfinding produces correct results on a small grid.

## Full USA Road Network Benchmark

The `usa_road_benchmark` provides detailed per-query benchmarking on the full
USA road network (23.9M nodes, 58.3M edges) with hardware performance counters.

### Features

- **Parallel execution**: Runs all 11 heap implementations simultaneously on
  separate pinned CPUs
- **Hardware counters**: Captures instructions, cycles, cache references, cache
  misses, IPC, and cache miss rate (Linux only, requires `perf-counters` feature)
- **Per-query results**: Individual timing for each query (not batched averages)
- **CSV output**: Results saved to `data/usa_bench_<heap>.csv` for easy plotting

### Setup

```bash
# 1. Enable perf counters (Linux only, requires root once)
sudo sysctl kernel.perf_event_paranoid=1

# 2. Download the full USA road network (~335MB compressed, ~1.5GB uncompressed)
./scripts/download-dimacs.sh USA
```

### Running the Benchmark

The benchmark runs in two phases:

```bash
# Step 1: Generate queries across Dijkstra rank buckets
# Default: 2^10 to 2^16, or specify max rank (e.g., 20 for 2^20, max 24)
cargo bench --features perf-counters --bench usa_road_benchmark -- generate      # default: up to 2^16
cargo bench --features perf-counters --bench usa_road_benchmark -- generate 20   # up to 2^20
cargo bench --features perf-counters --bench usa_road_benchmark -- generate 24   # up to 2^24 (maximum)

# Step 2: Run all heaps in parallel on separate pinned CPUs
cargo bench --features perf-counters --bench usa_road_benchmark -- run-parallel
```

You can also run a single heap manually:

```bash
# Run a specific heap on a specific CPU
BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench usa_road_benchmark -- run pairing_opt
```

### Available Commands

| Command | Description |
| --- | --- |
| `generate [MAX_RANK]` | Generate queries up to 2^MAX_RANK (default: 16, max: 24) |
| `run-parallel` | Run all heaps in parallel on separate CPUs |
| `run <heap> [heap2 ...]` | Run one or more heaps (use with `BENCH_PIN_CPU=N`) |
| `list` | List available heap implementations |
| `help` | Show usage information |

### Available Heaps

The benchmark includes 11 heap implementations (13 with `arena-storage` feature):

| Heap | Algorithm | Description |
| --- | --- | --- |
| `simple_binary` | lazy | Standard binary heap (baseline) |
| `pairing_lazy` | lazy | Pairing heap without decrease_key |
| `pairing_opt` | optimized | Pairing heap with decrease_key |
| `fibonacci_lazy` | lazy | Fibonacci heap without decrease_key |
| `fibonacci_opt` | optimized | Fibonacci heap with decrease_key |
| `rank_pairing_opt` | optimized | Rank-pairing heap |
| `hollow_lazy` | lazy | Hollow heap |
| `twothree_opt` | optimized | 2-3 heap |
| `strict_fib_opt` | optimized | Strict Fibonacci heap |
| `binomial_opt` | optimized | Binomial heap |
| `skew_binomial_opt` | optimized | Skew binomial heap |
| `binomial_arena` | optimized | Binomial heap with arena storage (requires `arena-storage` feature) |
| `skew_binomial_arena` | optimized | Skew binomial heap with arena storage (requires `arena-storage` feature) |

### Output Format

Results are saved to `data/usa_bench_<heap>.csv` with the following columns:

| Column | Description |
| --- | --- |
| `query_id` | Query identifier (0-indexed) |
| `source` | Source node ID |
| `target` | Target node ID |
| `dijkstra_rank` | Number of nodes settled to reach target |
| `log2_rank` | Log2 of target rank bucket (10, 12, 14, 16, 18) |
| `time_ns` | Wall-clock time in nanoseconds |
| `instructions` | Hardware instruction count |
| `cycles` | CPU cycles |
| `cache_refs` | LLC cache references |
| `cache_misses` | LLC cache misses |
| `ipc` | Instructions per cycle |
| `cache_miss_rate` | Cache miss percentage |

### Example Output

```csv
query_id,source,target,dijkstra_rank,log2_rank,time_ns,instructions,cycles,cache_refs,cache_misses,ipc,cache_miss_rate
0,2912745,4729006,1312,10,921516,2677602,1765800,33955,4209,1.5164,12.3958
1,12917471,12939516,7923,12,7672419,21556355,13774298,274985,61944,1.5650,22.5263
...
```

### CPU Pinning

The benchmark uses the `BENCH_PIN_CPU` environment variable to pin each process
to a specific CPU core. When using `run-parallel`, heaps are automatically
assigned to CPUs 0 through 10 (for 11 heaps).

For manual runs:

```bash
# Pin to CPU 0
BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench usa_road_benchmark -- run pairing_opt

# Pin to CPU 5
BENCH_PIN_CPU=5 cargo bench --features perf-counters --bench usa_road_benchmark -- run fibonacci_opt
```

### Comparing Different Build Configurations

A key design goal is enabling apples-to-apples comparisons across different
build configurations, feature flags, or code changes. Since queries are saved
to `data/usa_queries.json`, you can:

1. Generate queries once
2. Run benchmarks with different configurations
3. Compare results knowing all runs used identical queries

#### Comparing with and without arena storage

```bash
# Step 1: Generate queries (only needed once)
cargo bench --features perf-counters --bench usa_road_benchmark -- generate

# Step 2: Run with default configuration
cargo bench --features perf-counters --bench usa_road_benchmark -- run-parallel

# Save results
mkdir -p results/default
mv data/usa_bench_*.csv results/default/

# Step 3: Run with arena-storage feature enabled
cargo bench --features perf-counters,arena-storage --bench usa_road_benchmark -- run-parallel

# Save results
mkdir -p results/arena
mv data/usa_bench_*.csv results/arena/

# Step 4: Compare results
# Both runs used the exact same queries from data/usa_queries.json
diff results/default/usa_bench_pairing_opt.csv results/arena/usa_bench_pairing_opt.csv
```

#### Comparing code changes

```bash
# Generate queries on main branch
git checkout main
cargo bench --features perf-counters --bench usa_road_benchmark -- generate

# Run benchmark
cargo bench --features perf-counters --bench usa_road_benchmark -- run-parallel
mkdir -p results/before
mv data/usa_bench_*.csv results/before/

# Switch to feature branch (queries are preserved in data/)
git checkout my-optimization-branch
cargo bench --features perf-counters --bench usa_road_benchmark -- run-parallel
mkdir -p results/after
mv data/usa_bench_*.csv results/after/

# Compare - same queries, different code
paste results/before/usa_bench_pairing_opt.csv results/after/usa_bench_pairing_opt.csv | \
  awk -F',' '{print $1, $6, $19, ($6-$19)/$6*100 "%"}'
```

#### Running specific heaps for A/B comparison

```bash
# Run just two heaps to compare on the same CPU
BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench usa_road_benchmark -- run pairing_opt
mv data/usa_bench_pairing_opt.csv results/pairing_opt_baseline.csv

# Make a code change, rebuild, re-run
BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench usa_road_benchmark -- run pairing_opt
mv data/usa_bench_pairing_opt.csv results/pairing_opt_optimized.csv
```

The query file (`data/usa_queries.json`) contains:

- Graph metadata (node/edge counts for verification)
- All query parameters (source, target, expected Dijkstra rank)
- Reproducible across runs as long as the file is preserved

## DIMACS Data

Download road network datasets from the
[9th DIMACS Implementation Challenge](http://www.diag.uniroma1.it/challenge9/download.shtml).

The `data/` directory is git-ignored. Available datasets:

| Dataset | Nodes | Edges | Description |
| --- | --- | --- | --- |
| USA-road-d.NY | 264K | 730K | New York |
| USA-road-d.BAY | 321K | 800K | San Francisco Bay |
| USA-road-d.COL | 436K | 1M | Colorado |
| USA-road-d.FLA | 1.1M | 2.7M | Florida |
| USA-road-d.NE | 1.5M | 3.9M | Northeast USA |
| USA-road-d.CAL | 1.9M | 4.7M | California/Nevada |
| USA-road-d.USA | 23.9M | 58.3M | Full USA (largest) |

## DIMACS Format

The `.gr` files use a simple text format:

```text
c This is a comment
p sp <num_nodes> <num_edges>
a <from> <to> <weight>
a <from> <to> <weight>
...
```

- Lines starting with `c` are comments
- `p sp n m` declares a shortest path problem with n nodes and m edges
- `a u v w` defines a directed edge from node u to node v with weight w
- Node IDs are 1-indexed

## How Comparisons Work

### Same Problem, Different Heaps

Each benchmark runs the exact same queries with each heap type:

```rust
// Same queries for all heaps
let queries = generate_random_queries(&graph, 100, seed);

// Benchmark each heap on identical input
run_queries_fibonacci(&graph, &queries)
run_queries_pairing(&graph, &queries)
run_queries_rank_pairing(&graph, &queries)
```

This isolates the heap implementation as the only variable.

### Query Generation

- **Random pairs**: Seeded PRNG generates reproducible (source, target) pairs
- **Dijkstra rank queries**: Samples queries until finding ones with desired
  rank range (e.g., 2^12 to 2^13 nodes settled)
- **Graph shared via `Arc`**: No allocation overhead during benchmark

### Criterion Framework

We use [Criterion](https://github.com/bheisler/criterion.rs) for:

- **Statistical rigor**: Multiple samples, outlier detection, confidence
  intervals
- **Regression detection**: Compares against previous runs
- **HTML reports**: Generates `target/criterion/report/index.html`

Sample sizes:

- Synthetic graphs: 20 iterations
- Real DIMACS graphs: 10 iterations (large graphs take longer)

## Interpreting Results

Criterion reports:

- **Time**: Average time per iteration (batch of queries)
- **Throughput**: Can be configured per benchmark

### What to Look For

1. **Local vs Long-Distance**: Compare performance across Dijkstra rank
   buckets. Advanced heaps should show more benefit on high-rank (long) queries.

2. **Scale**: Performance differences become more apparent on larger graphs
   where the O(1) vs O(log n) `decrease_key` matters.

3. **Constant factors**: Fibonacci heaps have optimal theoretical bounds but
   higher constant factors. Pairing heaps often win in practice.

### Expected Results

Based on theoretical complexity and typical benchmarks:

| Query Type | Expected Winner | Why |
| --- | --- | --- |
| Local (low rank) | Pairing | Lower constant factors dominate |
| Long-distance | Rank-Pairing/Fibonacci | O(1) `decrease_key` pays off |
| Small graphs | Pairing | Overhead of advanced heaps not amortized |
| Large graphs | Rank-Pairing | Best balance of theory and practice |

### Viewing Reports

After running `cargo bench`, open the HTML report:

```bash
open target/criterion/report/index.html  # macOS
xdg-open target/criterion/report/index.html  # Linux
```

The report includes:

- Time distribution plots
- Comparison against baseline
- Regression detection

## Generating DIMACS-Style Results

The DIMACS Implementation Challenge uses a specific reporting format for
comparing shortest path algorithms. Here's how to generate consistent results:

### DIMACS Standard Format

DIMACS results are typically reported as:

- **Time per query** (microseconds or milliseconds)
- **Grouped by Dijkstra rank** (2^10, 2^12, ..., 2^24)
- **Averaged over 1000+ random queries**

### Generating Comparable Results

```bash
# 1. Download the standard NY dataset (used in most DIMACS papers)
./scripts/download-dimacs.sh NY

# 2. Run the Dijkstra rank benchmark
cargo bench -- real_dimacs_by_rank

# 3. Extract timing data from Criterion output
# Results are in target/criterion/real_dimacs_by_rank/*/new/estimates.json
```

### Export to CSV for Plotting

Use the provided export script to extract Criterion results:

```bash
# Export all results to CSV
./scripts/export-results.sh > results.csv

# Export specific benchmark group
./scripts/export-results.sh real_dimacs_by_rank > rank_results.csv

# List available benchmark groups
./scripts/export-results.sh --list
```

Output format: `benchmark,variant,mean_ns,std_dev_ns`

Example output:

```csv
benchmark,variant,mean_ns,std_dev_ns
random_queries,fibonacci_opt,45230000,1234000
random_queries,fibonacci_lazy,48900000,1567000
random_queries,pairing_opt,42100000,987000
```

### Plotting with gnuplot (DIMACS Standard)

DIMACS papers typically use log-scale plots with Dijkstra rank on x-axis:

```gnuplot
set terminal png size 800,600
set output 'dijkstra_rank.png'
set xlabel 'Dijkstra Rank (log2)'
set ylabel 'Time per Query (μs)'
set logscale y
set key top left
plot 'results.csv' using 1:2 with linespoints title 'Fibonacci', \
     '' using 1:3 with linespoints title 'Pairing', \
     '' using 1:4 with linespoints title 'Rank-Pairing'
```

### Comparison with Published Results

When comparing with DIMACS papers, note:

1. **Hardware matters**: DIMACS results from 2006-2009 used different CPUs
2. **Query count**: Papers use 1000+ queries; our benchmarks use 30-100 for speed
3. **Graph variants**: `-d` suffix means distance weights, `-t` means travel time
4. **Implementation language**: Most DIMACS entries were C/C++

To increase query count for publication-quality results, modify the benchmark:

```rust
// In benches/dimacs_benchmark.rs, change:
let queries = generate_queries_for_rank(&graph, log_rank, 1000, seed);
//                                                        ^^^^
```

## References

- [9th DIMACS Implementation Challenge](http://www.diag.uniroma1.it/challenge9/)
- Sanders, P., & Schultes, D. (2005). Highway Hierarchies Hasten Exact Shortest
  Path Queries. ESA 2005.
- [TRANSIT: Ultrafast Shortest-Path Queries](https://stubber.math-inf.uni-greifswald.de/informatik/PEOPLE/Papers/DIMACS06/DIMACS06.pdf)
