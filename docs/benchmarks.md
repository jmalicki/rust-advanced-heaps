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

### Synthetic vs Real Data

We use both synthetic and real graphs:

- **Synthetic graphs** run without any setup, useful for quick iteration
- **Real DIMACS data** provides realistic road network topology and weights

Synthetic graphs are deterministic (seeded PRNG) for reproducible results.

## Data File Strategy

### Directory Structure

```
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

### `dijkstra_heaps`

Compares Fibonacci, Pairing, and Rank-Pairing heaps on synthetic graphs:

- **Grid graphs**: 100x100 and 200x200 grids with unit edge weights
- **Sparse graphs**: 10K and 50K nodes with average degree 4
- **Dense graphs**: 500 nodes with ~50% edge density

### `density_impact`

Tests how graph density affects heap performance:

- Sparse (degree 3)
- Medium (degree 10)
- Dense (degree 30)

Higher density means more `decrease_key` operations, which is where advanced
heaps excel.

### `real_dimacs`

Benchmarks on real road network data (if present in `data/`):

- `USA-road-d.NY.gr` - New York (~264K nodes, ~730K edges)
- `USA-road-d.BAY.gr` - San Francisco Bay (~321K nodes, ~800K edges)
- `USA-road-d.COL.gr` - Colorado (~436K nodes, ~1M edges)

### `correctness`

Quick sanity check that pathfinding produces correct results.

## DIMACS Data

Download road network datasets from the [9th DIMACS Implementation Challenge](http://www.diag.uniroma1.it/challenge9/download.shtml).

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

```
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

Each benchmark runs the exact same shortest path query with each heap type:

```rust
// Same graph, same start/goal for all heaps
let start = DimacsNode::new(1, num_nodes, Arc::clone(&graph));

// Benchmark each heap on identical input
dijkstra::<_, FibonacciHeap<_, _>>(&start)
dijkstra::<_, PairingHeap<_, _>>(&start)
dijkstra::<_, RankPairingHeap<_, _>>(&start)
```

This isolates the heap implementation as the only variable.

### Query Design

- **Start node**: Always node 1 (first node in graph)
- **Goal node**: Always the last node (maximizes path length)
- **Graph shared via `Arc`**: No allocation overhead during benchmark

### Criterion Framework

We use [Criterion](https://github.com/bheisler/criterion.rs) for:

- **Statistical rigor**: Multiple samples, outlier detection, confidence intervals
- **Regression detection**: Compares against previous runs
- **HTML reports**: Generates `target/criterion/report/index.html`

Sample sizes:
- Synthetic graphs: 100 iterations (default)
- Real DIMACS graphs: 10 iterations (large graphs take longer)

## Interpreting Results

Criterion reports:

- **Time**: Average time per iteration
- **Throughput**: Edges processed per second

### What to Look For

1. **Sparse vs Dense**: Advanced heaps (Fibonacci, Pairing) should show more
   benefit on denser graphs where `decrease_key` is called more frequently.

2. **Scale**: Performance differences become more apparent on larger graphs
   where the O(1) vs O(log n) `decrease_key` matters.

3. **Constant factors**: Fibonacci heaps have optimal theoretical bounds but
   higher constant factors. Pairing heaps often win in practice.

### Expected Results

Based on theoretical complexity and typical benchmarks:

| Graph Type | Expected Winner | Why |
|------------|-----------------|-----|
| Small sparse | Pairing | Lower constant factors dominate |
| Large sparse | Pairing/Rank-Pairing | Good balance of theory and practice |
| Dense | Fibonacci/Rank-Pairing | O(1) `decrease_key` pays off |
| Road networks | Pairing | Sparse, many short paths |

### Viewing Reports

After running `cargo bench`, open the HTML report:

```bash
open target/criterion/report/index.html  # macOS
xdg-open target/criterion/report/index.html  # Linux
```

The report includes:
- Time distribution plots
- Comparison against baseline
- Throughput graphs
