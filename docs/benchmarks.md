# Benchmarks

This crate includes benchmarks comparing different heap implementations on
shortest path problems using DIMACS format road network graphs.

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

## Interpreting Results

Criterion reports:

- **Time**: Average time per iteration
- **Throughput**: Edges processed per second

Key things to look for:

1. **Sparse vs Dense**: Advanced heaps (Fibonacci, Pairing) should show more
   benefit on denser graphs where `decrease_key` is called more frequently.

2. **Scale**: Performance differences become more apparent on larger graphs
   where the O(1) vs O(log n) `decrease_key` matters.

3. **Constant factors**: Fibonacci heaps have optimal theoretical bounds but
   higher constant factors. Pairing heaps often win in practice.
