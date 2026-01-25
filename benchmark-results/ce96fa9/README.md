# Benchmark Results: ce96fa9

**Date**: 2026-01-25
**Commit**: ce96fa9
**Branch**: fibonacci-cache-optimization

## Test Configuration

- **Benchmark**: `perf_table` (Dijkstra's algorithm on synthetic graph)
- **Graph**: 2M nodes, synthetic
- **Dijkstra Ranks**: 2^8, 2^12, 2^16, 2^20
- **Queries per rank**: 10
- **Metrics**: Time (ms), IPC (instructions per cycle), Cache miss %

## Results at 2^20 Dijkstra Rank (Sorted by Time)

| Heap | Time | IPC | Cache Miss % | Notes |
|------|------|-----|--------------|-------|
| **simple_binary** | **31.5s** | 0.43 | 44.6% | Fastest overall |
| pairing_lazy | 49.6s | 0.56 | 43.8% | Best advanced heap |
| pairing_opt | 56.4s | 0.45 | 46.5% | |
| skew_binom_lazy | 57.3s | 0.56 | 44.9% | |
| fibonacci_lazy | 57.5s | 0.58 | 48.4% | |
| fibonacci_opt | 58.6s | 0.50 | 48.7% | |
| twothree_lazy | 59.9s | 0.56 | 45.5% | |
| skew_binom_opt | 60.4s | 0.50 | 46.3% | |
| twothree_opt | 62.5s | 0.50 | 47.5% | |
| strict_fib_lazy | 66.2s | 0.63 | 43.6% | |
| strict_fib_opt | 66.7s | 0.55 | 42.8% | Lowest cache miss of Fib variants |
| hollow_lazy | 68.4s | 0.96 | 39.8% | Best IPC of advanced heaps |
| binomial_lazy | 73.2s | 1.18 | 41.4% | Best IPC overall |
| binomial_opt | 73.9s | 0.98 | 44.8% | |
| rank_pair_lazy | 76.1s | 0.86 | 43.3% | |
| hollow_opt | 78.9s | 0.76 | 46.5% | |
| skiplist_lazy | 88.0s | 0.70 | 33.1% | Lowest cache miss %, slowest |

## Results at All Scales

### 2^8 (256 nodes settled)

| Heap | Time (ms) | IPC | Cache Miss % |
|------|-----------|-----|--------------|
| simple_binary | 3.1 | 1.41 | 8.3% |
| pairing_lazy | 5.5 | 1.93 | 7.1% |
| pairing_opt | 5.6 | 1.92 | 9.0% |
| strict_fib_lazy | 5.8 | 1.94 | 6.8% |
| strict_fib_opt | 5.9 | 1.92 | 6.5% |
| twothree_lazy | 5.9 | 2.01 | 8.0% |
| twothree_opt | 5.9 | 2.02 | 7.9% |
| fibonacci_lazy | 6.4 | 1.87 | 6.4% |
| fibonacci_opt | 6.4 | 1.85 | 6.0% |
| skew_binom_lazy | 7.2 | 1.56 | 8.0% |
| skew_binom_opt | 7.4 | 1.53 | 8.3% |
| hollow_lazy | 7.7 | 2.10 | 7.7% |
| binomial_lazy | 8.2 | 1.94 | 6.6% |
| binomial_opt | 8.4 | 1.92 | 7.5% |
| rank_pair_lazy | 8.5 | 2.09 | 6.4% |
| hollow_opt | 8.8 | 2.13 | 12.9% |
| skiplist_lazy | 13.9 | 2.12 | 6.5% |

### 2^12 (4096 nodes settled)

| Heap | Time (ms) | IPC | Cache Miss % |
|------|-----------|-----|--------------|
| simple_binary | 63.6 | 1.13 | 19.4% |
| pairing_lazy | 105.7 | 1.66 | 24.5% |
| pairing_opt | 113.1 | 1.59 | 24.9% |
| strict_fib_lazy | 119.9 | 1.61 | 30.2% |
| twothree_lazy | 121.1 | 1.68 | 26.1% |
| twothree_opt | 121.4 | 1.66 | 28.1% |
| strict_fib_opt | 125.8 | 1.53 | 30.9% |
| fibonacci_lazy | 130.6 | 1.56 | 28.7% |
| skew_binom_lazy | 145.9 | 1.30 | 26.3% |
| skew_binom_opt | 148.3 | 1.31 | 26.9% |
| hollow_lazy | 156.3 | 1.85 | 25.2% |
| fibonacci_opt | 160.1 | 1.27 | 30.3% |
| binomial_lazy | 176.5 | 1.83 | 24.5% |
| binomial_opt | 177.4 | 1.82 | 25.5% |
| rank_pair_lazy | 177.6 | 1.88 | 24.7% |
| hollow_opt | 184.2 | 1.77 | 36.1% |
| skiplist_lazy | 286.0 | 1.73 | 25.9% |

### 2^16 (65536 nodes settled)

| Heap | Time (s) | IPC | Cache Miss % |
|------|----------|-----|--------------|
| simple_binary | 1.59 | 0.69 | 32.6% |
| pairing_lazy | 2.94 | 0.93 | 34.9% |
| pairing_opt | 3.27 | 0.81 | 36.4% |
| twothree_lazy | 3.38 | 0.93 | 36.1% |
| skew_binom_lazy | 3.49 | 0.88 | 36.4% |
| fibonacci_lazy | 3.63 | 0.85 | 40.7% |
| twothree_opt | 3.66 | 0.84 | 38.6% |
| skew_binom_opt | 3.75 | 0.81 | 38.3% |
| strict_fib_lazy | 3.76 | 0.85 | 40.0% |
| strict_fib_opt | 3.88 | 0.81 | 40.0% |
| hollow_lazy | 3.89 | 1.22 | 33.7% |
| rank_pair_lazy | 4.26 | 1.25 | 36.4% |
| binomial_lazy | 4.30 | 1.35 | 35.2% |
| fibonacci_opt | 4.44 | 0.74 | 44.4% |
| binomial_opt | 4.68 | 1.23 | 37.3% |
| hollow_opt | 5.32 | 1.01 | 41.5% |
| skiplist_lazy | 7.45 | 1.04 | 30.0% |

## Key Observations

1. **Simple binary heap dominates** at all scales - 1.6x faster than the best
   advanced heap at 2^20

2. **Lazy vs Eager**: Lazy decrease_key variants consistently outperform eager
   - pairing_lazy 13% faster than pairing_opt
   - hollow_lazy 13% faster than hollow_opt

3. **Cache efficiency matters**: IPC ranges from 0.43 to 1.18 at 2^20 scale.
   Binomial heaps have highest IPC (1.18) but slower due to more operations

4. **Skiplist has lowest cache miss rate** (33.1%) but is 2.8x slower than
   simple_binary - cache efficiency isn't the primary bottleneck

5. **O(1) decrease_key doesn't pay off** in this benchmark - constant factors
   overwhelm the asymptotic advantage even at 2^20 scale

## Files

- `perf_table.csv` - Raw benchmark data in CSV format
