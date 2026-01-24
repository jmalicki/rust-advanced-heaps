//! Access pattern analyzer for heap implementations
//!
//! This binary runs various workload scenarios and collects access pattern statistics
//! for each heap implementation. The statistics help analyze cache efficiency by tracking
//! field access distributions and pointer chase depths.
//!
//! # Usage
//!
//! ```bash
//! # Build with access tracking enabled
//! cargo build --release --bin access_analyzer --features access_tracking
//!
//! # Run analysis for a specific heap and workload
//! ./target/release/access_analyzer dijkstra_sparse 10000 fibonacci
//!
//! # Run analysis for all heaps
//! ./target/release/access_analyzer dijkstra_sparse 10000 all
//!
//! # Available workloads:
//! #   dijkstra_sparse - 10K nodes, avg degree 4 (typical decrease_key heavy)
//! #   dijkstra_dense  - 10K nodes, avg degree 100 (more pops than decrease_key)
//! #   sort_only       - Push N, pop N (no decrease_key)
//! #   decrease_heavy  - Push N, decrease_key 10N, pop N
//! #   merge_heavy     - Many small heaps merged
//! ```
//!
//! # Output
//!
//! The tool outputs:
//! - Operation statistics (calls, nodes touched per call)
//! - Field access distributions (priority, item, pointers, etc.)
//! - Pointer chase depth (proxy for cache misses)

use rust_advanced_heaps::access_tracking::{self, reset_stats};
use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::radix::RadixHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::skiplist::SkipListHeap;
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap};
use std::env;

/// Workload configuration
#[derive(Debug, Clone, Copy)]
enum Workload {
    /// Sparse graph Dijkstra simulation (avg degree 4)
    DijkstraSparse,
    /// Dense graph Dijkstra simulation (avg degree 100)
    DijkstraDense,
    /// Sort only: push N, pop N (no decrease_key)
    SortOnly,
    /// Decrease-key heavy: push N, decrease_key 10N, pop N
    DecreaseHeavy,
    /// Merge heavy: many small heaps merged
    MergeHeavy,
}

impl Workload {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "dijkstra_sparse" => Some(Workload::DijkstraSparse),
            "dijkstra_dense" => Some(Workload::DijkstraDense),
            "sort_only" => Some(Workload::SortOnly),
            "decrease_heavy" => Some(Workload::DecreaseHeavy),
            "merge_heavy" => Some(Workload::MergeHeavy),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Workload::DijkstraSparse => "dijkstra_sparse",
            Workload::DijkstraDense => "dijkstra_dense",
            Workload::SortOnly => "sort_only",
            Workload::DecreaseHeavy => "decrease_heavy",
            Workload::MergeHeavy => "merge_heavy",
        }
    }
}

/// Simple LCG random number generator for reproducibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

/// Simulate Dijkstra's algorithm workload
fn run_dijkstra_workload<H>(n: usize, avg_degree: usize, heap_name: &str, workload_name: &str)
where
    H: DecreaseKeyHeap<usize, i64>,
{
    reset_stats();

    let mut heap = H::new();
    let mut handles = Vec::with_capacity(n);
    let mut rng = SimpleRng::new(42);

    // Initialize: push all nodes with "infinite" distance except source
    for i in 0..n {
        let priority = if i == 0 { 0 } else { i64::MAX / 2 };
        handles.push(heap.push_with_handle(priority, i));
    }

    // Track which nodes have been "finalized"
    let mut finalized = vec![false; n];
    let mut distances: Vec<i64> = (0..n)
        .map(|i| if i == 0 { 0 } else { i64::MAX / 2 })
        .collect();

    // Simulate edge relaxations
    let total_edges = n * avg_degree;
    let mut edges_processed = 0;

    while let Some((dist, node)) = heap.pop() {
        if finalized[node] {
            continue;
        }
        finalized[node] = true;

        // Simulate relaxing edges from this node
        let num_edges = avg_degree.min(total_edges - edges_processed);
        for _ in 0..num_edges {
            let neighbor = rng.next_usize(n);
            if finalized[neighbor] {
                continue;
            }

            let edge_weight = (rng.next_u64() % 1000) as i64 + 1;
            let new_dist = dist.saturating_add(edge_weight);

            if new_dist < distances[neighbor] {
                distances[neighbor] = new_dist;
                let _ = heap.decrease_key(&handles[neighbor], new_dist);
            }
        }
        edges_processed += num_edges;
    }

    access_tracking::print_report(heap_name, workload_name, n);
}

/// Sort-only workload: push N, pop N
fn run_sort_workload<H>(n: usize, heap_name: &str)
where
    H: Heap<usize, i64>,
{
    reset_stats();

    let mut heap = H::new();
    let mut rng = SimpleRng::new(42);

    // Push N elements with random priorities
    for i in 0..n {
        let priority = (rng.next_u64() % (n as u64 * 10)) as i64;
        heap.push(priority, i);
    }

    // Pop all elements
    while heap.pop().is_some() {}

    access_tracking::print_report(heap_name, "sort_only", n);
}

/// Decrease-key heavy workload
fn run_decrease_heavy_workload<H>(n: usize, heap_name: &str)
where
    H: DecreaseKeyHeap<usize, i64>,
{
    reset_stats();

    let mut heap = H::new();
    let mut handles = Vec::with_capacity(n);
    let mut priorities: Vec<i64> = Vec::with_capacity(n);
    let mut rng = SimpleRng::new(42);

    // Push N elements
    for i in 0..n {
        let priority = (n * 100 + rng.next_usize(n * 100)) as i64;
        handles.push(heap.push_with_handle(priority, i));
        priorities.push(priority);
    }

    // Decrease keys 10N times
    for _ in 0..(n * 10) {
        let idx = rng.next_usize(n);
        let decrease_amount = (rng.next_usize(100) + 1) as i64;
        let new_priority = priorities[idx] - decrease_amount;
        if heap.decrease_key(&handles[idx], new_priority).is_ok() {
            priorities[idx] = new_priority;
        }
    }

    // Pop all elements
    while heap.pop().is_some() {}

    access_tracking::print_report(heap_name, "decrease_heavy", n);
}

/// Merge-heavy workload
fn run_merge_heavy_workload<H>(n: usize, heap_name: &str)
where
    H: Heap<usize, i64>,
{
    reset_stats();

    let batch_size = 100.max(n / 100);
    let num_batches = n / batch_size;
    let mut rng = SimpleRng::new(42);

    let mut main_heap = H::new();

    // Create and merge many small heaps
    for batch in 0..num_batches {
        let mut small_heap = H::new();
        for i in 0..batch_size {
            let priority = (rng.next_u64() % (n as u64 * 10)) as i64;
            small_heap.push(priority, batch * batch_size + i);
        }
        main_heap.merge(small_heap);
    }

    // Pop all elements
    while main_heap.pop().is_some() {}

    access_tracking::print_report(heap_name, "merge_heavy", n);
}

/// Run workload for radix heap (requires unsigned keys)
fn run_radix_workload(n: usize, workload: Workload) {
    reset_stats();

    let heap_name = "RadixHeap";

    match workload {
        Workload::SortOnly | Workload::MergeHeavy => {
            // RadixHeap has different constraints
            let mut heap = RadixHeap::<usize, u64>::new();
            let mut rng = SimpleRng::new(42);

            for i in 0..n {
                let priority = (rng.next_u64() % (n as u64 * 10)) + 1;
                heap.push(priority, i);
            }

            while heap.pop().is_some() {}

            access_tracking::print_report(heap_name, workload.name(), n);
        }
        Workload::DijkstraSparse | Workload::DijkstraDense => {
            let avg_degree = match workload {
                Workload::DijkstraSparse => 4,
                Workload::DijkstraDense => 100,
                _ => unreachable!(),
            };

            let mut heap = RadixHeap::<usize, u64>::new();
            let mut handles = Vec::with_capacity(n);
            let mut rng = SimpleRng::new(42);

            // RadixHeap requires monotone access, so we simulate differently
            for i in 0..n {
                let priority = if i == 0 { 0u64 } else { u64::MAX / 2 };
                handles.push(heap.push_with_handle(priority, i));
            }

            let mut finalized = vec![false; n];
            let mut distances: Vec<u64> = (0..n)
                .map(|i| if i == 0 { 0 } else { u64::MAX / 2 })
                .collect();
            let total_edges = n * avg_degree;
            let mut edges_processed = 0;

            while let Some((dist, node)) = heap.pop() {
                if finalized[node] {
                    continue;
                }
                finalized[node] = true;

                let num_edges = avg_degree.min(total_edges - edges_processed);
                for _ in 0..num_edges {
                    let neighbor = rng.next_usize(n);
                    if finalized[neighbor] {
                        continue;
                    }

                    let edge_weight = (rng.next_u64() % 1000) + 1;
                    let new_dist = dist.saturating_add(edge_weight);

                    if new_dist < distances[neighbor] {
                        distances[neighbor] = new_dist;
                        let _ = heap.decrease_key(&handles[neighbor], new_dist);
                    }
                }
                edges_processed += num_edges;
            }

            access_tracking::print_report(heap_name, workload.name(), n);
        }
        Workload::DecreaseHeavy => {
            let mut heap = RadixHeap::<usize, u64>::new();
            let mut handles = Vec::with_capacity(n);
            let mut priorities: Vec<u64> = Vec::with_capacity(n);
            let mut rng = SimpleRng::new(42);

            // Push with high initial priorities
            for i in 0..n {
                let priority = (n as u64 * 100) + (rng.next_u64() % (n as u64 * 100));
                handles.push(heap.push_with_handle(priority, i));
                priorities.push(priority);
            }

            // Decrease keys
            for _ in 0..(n * 10) {
                let idx = rng.next_usize(n);
                let decrease_amount = (rng.next_u64() % 100) + 1;
                if priorities[idx] > decrease_amount {
                    let new_priority = priorities[idx] - decrease_amount;
                    if heap.decrease_key(&handles[idx], new_priority).is_ok() {
                        priorities[idx] = new_priority;
                    }
                }
            }

            while heap.pop().is_some() {}

            access_tracking::print_report(heap_name, workload.name(), n);
        }
    }
}

/// Run workload for skiplist heap
fn run_skiplist_workload(n: usize, workload: Workload) {
    let heap_name = "SkipListHeap";

    match workload {
        Workload::DijkstraSparse => {
            run_dijkstra_workload::<SkipListHeap<usize, i64>>(n, 4, heap_name, "dijkstra_sparse");
        }
        Workload::DijkstraDense => {
            run_dijkstra_workload::<SkipListHeap<usize, i64>>(n, 100, heap_name, "dijkstra_dense");
        }
        Workload::SortOnly => {
            run_sort_workload::<SkipListHeap<usize, i64>>(n, heap_name);
        }
        Workload::DecreaseHeavy => {
            run_decrease_heavy_workload::<SkipListHeap<usize, i64>>(n, heap_name);
        }
        Workload::MergeHeavy => {
            run_merge_heavy_workload::<SkipListHeap<usize, i64>>(n, heap_name);
        }
    }
}

/// Run analysis for a specific heap
fn run_heap_analysis(heap_type: &str, workload: Workload, n: usize) {
    match heap_type {
        "fibonacci" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<FibonacciHeap<usize, i64>>(
                    n,
                    4,
                    "FibonacciHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<FibonacciHeap<usize, i64>>(
                    n,
                    100,
                    "FibonacciHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<FibonacciHeap<usize, i64>>(n, "FibonacciHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<FibonacciHeap<usize, i64>>(n, "FibonacciHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<FibonacciHeap<usize, i64>>(n, "FibonacciHeap");
            }
        },
        "pairing" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<PairingHeap<usize, i64>>(
                    n,
                    4,
                    "PairingHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<PairingHeap<usize, i64>>(
                    n,
                    100,
                    "PairingHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<PairingHeap<usize, i64>>(n, "PairingHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<PairingHeap<usize, i64>>(n, "PairingHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<PairingHeap<usize, i64>>(n, "PairingHeap");
            }
        },
        "rank_pairing" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<RankPairingHeap<usize, i64>>(
                    n,
                    4,
                    "RankPairingHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<RankPairingHeap<usize, i64>>(
                    n,
                    100,
                    "RankPairingHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<RankPairingHeap<usize, i64>>(n, "RankPairingHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<RankPairingHeap<usize, i64>>(n, "RankPairingHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<RankPairingHeap<usize, i64>>(n, "RankPairingHeap");
            }
        },
        "binomial" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<BinomialHeap<usize, i64>>(
                    n,
                    4,
                    "BinomialHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<BinomialHeap<usize, i64>>(
                    n,
                    100,
                    "BinomialHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<BinomialHeap<usize, i64>>(n, "BinomialHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<BinomialHeap<usize, i64>>(n, "BinomialHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<BinomialHeap<usize, i64>>(n, "BinomialHeap");
            }
        },
        "strict_fibonacci" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<StrictFibonacciHeap<usize, i64>>(
                    n,
                    4,
                    "StrictFibonacciHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<StrictFibonacciHeap<usize, i64>>(
                    n,
                    100,
                    "StrictFibonacciHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<StrictFibonacciHeap<usize, i64>>(n, "StrictFibonacciHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<StrictFibonacciHeap<usize, i64>>(
                    n,
                    "StrictFibonacciHeap",
                );
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<StrictFibonacciHeap<usize, i64>>(
                    n,
                    "StrictFibonacciHeap",
                );
            }
        },
        "twothree" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<TwoThreeHeap<usize, i64>>(
                    n,
                    4,
                    "TwoThreeHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<TwoThreeHeap<usize, i64>>(
                    n,
                    100,
                    "TwoThreeHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<TwoThreeHeap<usize, i64>>(n, "TwoThreeHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<TwoThreeHeap<usize, i64>>(n, "TwoThreeHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<TwoThreeHeap<usize, i64>>(n, "TwoThreeHeap");
            }
        },
        "skew_binomial" => match workload {
            Workload::DijkstraSparse => {
                run_dijkstra_workload::<SkewBinomialHeap<usize, i64>>(
                    n,
                    4,
                    "SkewBinomialHeap",
                    "dijkstra_sparse",
                );
            }
            Workload::DijkstraDense => {
                run_dijkstra_workload::<SkewBinomialHeap<usize, i64>>(
                    n,
                    100,
                    "SkewBinomialHeap",
                    "dijkstra_dense",
                );
            }
            Workload::SortOnly => {
                run_sort_workload::<SkewBinomialHeap<usize, i64>>(n, "SkewBinomialHeap");
            }
            Workload::DecreaseHeavy => {
                run_decrease_heavy_workload::<SkewBinomialHeap<usize, i64>>(n, "SkewBinomialHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<SkewBinomialHeap<usize, i64>>(n, "SkewBinomialHeap");
            }
        },
        "radix" => {
            run_radix_workload(n, workload);
        }
        "simple_binary" => match workload {
            Workload::SortOnly => {
                run_sort_workload::<SimpleBinaryHeap<usize, i64>>(n, "SimpleBinaryHeap");
            }
            Workload::MergeHeavy => {
                run_merge_heavy_workload::<SimpleBinaryHeap<usize, i64>>(n, "SimpleBinaryHeap");
            }
            _ => {
                println!(
                    "SimpleBinaryHeap does not support decrease_key, skipping {}",
                    workload.name()
                );
            }
        },
        "skiplist" => {
            run_skiplist_workload(n, workload);
        }
        _ => {
            eprintln!("Unknown heap type: {}", heap_type);
        }
    }
}

fn print_usage() {
    eprintln!("Access Pattern Analyzer for Heap Implementations");
    eprintln!();
    eprintln!("Usage: access_analyzer <workload> <n> <heap_type>");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  workload   Workload scenario to run:");
    eprintln!("             dijkstra_sparse - Sparse graph (avg degree 4)");
    eprintln!("             dijkstra_dense  - Dense graph (avg degree 100)");
    eprintln!("             sort_only       - Push N, pop N (no decrease_key)");
    eprintln!("             decrease_heavy  - Push N, decrease_key 10N, pop N");
    eprintln!("             merge_heavy     - Many small heaps merged");
    eprintln!();
    eprintln!("  n          Number of elements (e.g., 10000)");
    eprintln!();
    eprintln!("  heap_type  One of:");
    eprintln!("             fibonacci, pairing, rank_pairing, binomial,");
    eprintln!("             strict_fibonacci, twothree, skew_binomial,");
    eprintln!("             radix, simple_binary, skiplist, all");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  cargo run --release --bin access_analyzer --features access_tracking -- dijkstra_sparse 10000 fibonacci");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        print_usage();
        std::process::exit(1);
    }

    let workload = match Workload::from_str(&args[1]) {
        Some(w) => w,
        None => {
            eprintln!("Error: Unknown workload: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    };

    let n: usize = match args[2].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: Invalid number of elements: {}", args[2]);
            print_usage();
            std::process::exit(1);
        }
    };

    let heap_type = &args[3];

    println!("Access Pattern Analysis");
    println!("=======================");
    println!("Workload: {}", workload.name());
    println!("Elements: {}", n);
    println!();

    if heap_type == "all" {
        let heaps = [
            "fibonacci",
            "pairing",
            "rank_pairing",
            "binomial",
            "strict_fibonacci",
            "twothree",
            "skew_binomial",
            "radix",
            "simple_binary",
            "skiplist",
        ];
        for heap in heaps {
            println!("\n{}", "=".repeat(60));
            run_heap_analysis(heap, workload, n);
        }
    } else {
        run_heap_analysis(heap_type, workload, n);
    }
}
