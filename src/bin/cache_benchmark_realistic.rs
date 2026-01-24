//! Realistic cache benchmark simulating expensive path expansion
//!
//! This benchmark simulates a more realistic Dijkstra implementation where
//! graph traversal between heap operations causes cache pollution, evicting
//! heap node data from cache.
//!
//! Uses the same RNG seed as cache_benchmark_all.rs to ensure identical
//! heap operations, isolating just the effect of cache pollution.
//!
//! Run with:
//!   cargo build --release --bin cache_benchmark_realistic
//!   valgrind --tool=cachegrind ./target/release/cache_benchmark_realistic <heap_type> <n>
//!
//! heap_type: fibonacci, skew_binomial, pairing, rank_pairing, binomial, twothree

use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

/// Simple seeded PRNG (xorshift64) for reproducible edge weights.
/// Using the same seed in both benchmarks ensures identical heap operations.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Generate edge weight in range [1, 100]
    fn edge_weight(&mut self) -> i64 {
        (self.next() % 100 + 1) as i64
    }
}

/// Same seed as cache_benchmark_all.rs for identical heap operations
const RNG_SEED: u64 = 0xDEADBEEF_CAFEBABE;

/// Simulates graph adjacency data that must be traversed during path expansion.
/// This is designed to pollute the cache between heap operations.
struct GraphSimulator {
    /// Large array simulating adjacency lists / edge data
    /// Sized to exceed L1/L2 cache (e.g., 2MB)
    edge_data: Vec<u64>,
    /// Number of cache lines to touch per "expansion"
    lines_per_expansion: usize,
}

impl GraphSimulator {
    fn new(size_bytes: usize, lines_per_expansion: usize) -> Self {
        let num_elements = size_bytes / std::mem::size_of::<u64>();
        let mut edge_data = vec![0u64; num_elements];
        // Initialize with pseudo-random data to prevent optimization
        for (i, val) in edge_data.iter_mut().enumerate() {
            *val = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
        }
        Self {
            edge_data,
            lines_per_expansion,
        }
    }

    /// Simulate expensive path expansion by touching many cache lines.
    /// Returns a checksum to prevent dead code elimination.
    #[inline(never)]
    fn expand_path(&self, node_id: usize) -> u64 {
        let mut checksum = 0u64;
        let stride = 64 / std::mem::size_of::<u64>(); // One cache line = 64 bytes = 8 u64s

        // Touch `lines_per_expansion` cache lines, spread across the array
        let start = (node_id * stride * 7) % self.edge_data.len();
        for i in 0..self.lines_per_expansion {
            let idx = (start + i * stride * 17) % self.edge_data.len();
            checksum = checksum.wrapping_add(self.edge_data[idx]);
        }

        checksum
    }
}

fn dijkstra_realistic<H>(mut heap: H, n: usize, graph: &GraphSimulator) -> (usize, usize, u64)
where
    H: Heap<usize, i64> + DecreaseKeyHeap<usize, i64>,
    <H as DecreaseKeyHeap<usize, i64>>::Handle: Clone,
{
    let mut handles: Vec<Option<H::Handle>> = Vec::with_capacity(n);
    let mut checksum = 0u64;
    let mut rng = Rng::new(RNG_SEED);

    // Phase 1: Insert all elements with random initial priorities (same as ideal benchmark)
    for i in 0..n {
        let priority = (rng.next() % (n as u64 * 2)) as i64;
        let handle = heap.push_with_handle(priority, i);
        handles.push(Some(handle));
    }

    // Phase 2: Dijkstra-like workload with cache pollution
    let mut popped = 0;
    let mut decreased = 0;

    while let Some((priority, item)) = heap.pop() {
        popped += 1;
        handles[item] = None;

        // === CACHE POLLUTION: Simulate expensive path expansion ===
        // This touches many cache lines, evicting heap node data
        checksum = checksum.wrapping_add(graph.expand_path(item));

        // Simulate edge relaxation with random edge weights (same as ideal benchmark)
        for offset in [1, 7, 13, 31, 97] {
            let neighbor = (item + offset) % n;
            if let Some(ref handle) = handles[neighbor] {
                let edge_weight = rng.edge_weight();
                let new_priority = priority + edge_weight;

                // Now access heap data - likely a cache miss after expansion
                if heap.decrease_key(handle, new_priority).is_ok() {
                    decreased += 1;
                }
            }
        }

        if popped >= n / 2 {
            break;
        }
    }

    // Phase 3: Drain
    while heap.pop().is_some() {
        popped += 1;
    }

    (popped, decreased, checksum)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let heap_type = args.get(1).map(|s| s.as_str()).unwrap_or("skew_binomial");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    // Graph size: 2MB to exceed typical L2 cache
    // Lines per expansion: 256 cache lines = 16KB of data touched per pop
    let graph_size = 2 * 1024 * 1024; // 2MB
    let lines_per_expansion = 256;

    println!(
        "Running {} benchmark with {} elements (realistic workload)",
        heap_type, n
    );
    println!(
        "Graph simulator: {}KB, {} cache lines touched per expansion",
        graph_size / 1024,
        lines_per_expansion
    );

    let graph = GraphSimulator::new(graph_size, lines_per_expansion);

    let (popped, decreased, checksum) = match heap_type {
        "skew_binomial" => dijkstra_realistic(SkewBinomialHeap::<usize, i64>::new(), n, &graph),
        "pairing" => dijkstra_realistic(PairingHeap::<usize, i64>::new(), n, &graph),
        "rank_pairing" => dijkstra_realistic(RankPairingHeap::<usize, i64>::new(), n, &graph),
        "binomial" => dijkstra_realistic(BinomialHeap::<usize, i64>::new(), n, &graph),
        "twothree" => dijkstra_realistic(TwoThreeHeap::<usize, i64>::new(), n, &graph),
        _ => {
            eprintln!(
                "Unknown heap type: {}. Use: skew_binomial, pairing, rank_pairing, binomial, twothree",
                heap_type
            );
            std::process::exit(1);
        }
    };

    println!("Completed: {} pops, {} decrease_keys", popped, decreased);
    println!("Checksum: {} (prevents dead code elimination)", checksum);
}
