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

We compare three heap implementations:

| Heap | `decrease_key` | Why Include |
|------|----------------|-------------|
| **Fibonacci** | O(1) amortized | Optimal theoretical bounds |
| **Pairing** | o(log n) amortized | Simpler, often faster in practice |
| **Rank-Pairing** | O(1) amortized | Optimal bounds, simpler than Fibonacci |

BinomialHeap is excluded because its ownership model conflicts with storing
handles during pathfinding.

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

# Run with real DIMACS data (requires download)
mkdir -p data
cd data
wget http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz
gunzip USA-road-d.NY.gr.gz
cd ..
cargo bench
```

## Benchmark Groups

### `random_queries`

Compares heaps on 100 random (source, target) queries on a 10K node synthetic
graph. This is the DIMACS-standard way to measure average query performance.

### `dijkstra_rank`

Groups queries by Dijkstra rank (number of nodes settled before reaching
target), following Sanders/Schultes methodology:

| Rank | Meaning |
|------|---------|
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

## DIMACS Data

Download road network datasets from the
[9th DIMACS Implementation Challenge](http://www.diag.uniroma1.it/challenge9/download.shtml).

The `data/` directory is git-ignored. Available datasets:

| Dataset | Nodes | Edges | Description |
|---------|-------|-------|-------------|
| USA-road-d.NY | 264K | 730K | New York |
| USA-road-d.BAY | 321K | 800K | San Francisco Bay |
| USA-road-d.COL | 436K | 1M | Colorado |
| USA-road-d.FLA | 1.1M | 2.7M | Florida |
| USA-road-d.NE | 1.5M | 3.9M | Northeast USA |
| USA-road-d.CAL | 1.9M | 4.7M | California/Nevada |

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
|------------|-----------------|-----|
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

## References

- [9th DIMACS Implementation Challenge](http://www.diag.uniroma1.it/challenge9/)
- Sanders, P., & Schultes, D. (2005). Highway Hierarchies Hasten Exact Shortest
  Path Queries. ESA 2005.
- [TRANSIT: Ultrafast Shortest-Path Queries](https://stubber.math-inf.uni-greifswald.de/informatik/PEOPLE/Papers/DIMACS06/DIMACS06.pdf)
