//! Hardware Performance Counter Benchmarks
//!
//! Runs the same workloads as dimacs_benchmark but measures hardware counters
//! (instructions, cycles, cache misses) in addition to wall-clock time.
//!
//! ## Running
//!
//! ```bash
//! # Enable perf access (requires Linux)
//! sudo sysctl kernel.perf_event_paranoid=1
//!
//! # Run all benchmarks
//! cargo bench --features perf-counters --bench perf_benchmark
//! ```
//!
//! ## Filtering Benchmarks
//!
//! Use Criterion's filter to run subsets of benchmarks:
//!
//! ```bash
//! # Run only optimized (decrease_key) heaps
//! cargo bench --features perf-counters --bench perf_benchmark -- '_opt/'
//!
//! # Run only lazy (re-insertion) heaps
//! cargo bench --features perf-counters --bench perf_benchmark -- '_lazy/'
//!
//! # Run only a specific rank level
//! cargo bench --features perf-counters --bench perf_benchmark -- '2\^10'
//!
//! # Combine filters: optimized heaps at rank 2^12
//! cargo bench --features perf-counters --bench perf_benchmark -- '_opt/2\^12'
//! ```
//!
//! ## CPU Pinning
//!
//! For more stable results, pin the benchmark to a specific CPU:
//!
//! ```bash
//! # Pin to CPU 0 (first physical core)
//! BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench perf_benchmark
//!
//! # Auto-select first available CPU
//! BENCH_PIN_CPU=auto cargo bench --features perf-counters --bench perf_benchmark
//! ```
//!
//! **Note on CPU numbering:** On systems with hyperthreading, CPUs 0 to N-1 are
//! typically the first thread on each physical core, and CPUs N to 2N-1 are the
//! second threads (siblings). For best isolation, use CPUs 0 to N-1. You can
//! check your topology with: `cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | sort -u`
//!
//! ## Running Multiple Benchmarks in Parallel
//!
//! Run 8 benchmark processes on 8 physical cores simultaneously:
//!
//! ```bash
//! BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench perf_benchmark -- '_opt/2\^8' &
//! BENCH_PIN_CPU=1 cargo bench --features perf-counters --bench perf_benchmark -- '_opt/2\^12' &
//! BENCH_PIN_CPU=2 cargo bench --features perf-counters --bench perf_benchmark -- '_opt/2\^16' &
//! BENCH_PIN_CPU=3 cargo bench --features perf-counters --bench perf_benchmark -- '_opt/2\^20' &
//! BENCH_PIN_CPU=4 cargo bench --features perf-counters --bench perf_benchmark -- '_lazy/2\^8' &
//! BENCH_PIN_CPU=5 cargo bench --features perf-counters --bench perf_benchmark -- '_lazy/2\^12' &
//! BENCH_PIN_CPU=6 cargo bench --features perf-counters --bench perf_benchmark -- '_lazy/2\^16' &
//! BENCH_PIN_CPU=7 cargo bench --features perf-counters --bench perf_benchmark -- '_lazy/2\^20' &
//! wait
//! ```
//!
//! ## Metrics
//!
//! The benchmark captures ALL metrics simultaneously:
//! - Wall clock time (used by Criterion for statistics and displayed results)
//! - Instructions: Total instructions retired
//! - Cycles: CPU cycles
//! - Branches: Branch instructions
//! - Branch misses: Mispredicted branches
//! - Cache refs: Cache references
//! - Cache misses: Last-level cache misses

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod perf_benches {
    use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
    use perf_measurement::PerfMultiMeasurement;
    use rust_advanced_heaps::binomial::BinomialHeap;
    use rust_advanced_heaps::fibonacci::FibonacciHeap;
    use rust_advanced_heaps::hollow::HollowHeap;
    use rust_advanced_heaps::pairing::PairingHeap;
    use rust_advanced_heaps::pathfinding::{shortest_path, shortest_path_lazy, SearchNode};
    use rust_advanced_heaps::rank_pairing::RankPairingHeap;
    use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
    use rust_advanced_heaps::skiplist::SkipListHeap;
    use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
    use rust_advanced_heaps::twothree::TwoThreeHeap;
    use std::collections::HashMap;
    use std::hint::black_box;
    use std::sync::Arc;

    // ========================================================================
    // Graph and query infrastructure (shared with dimacs_benchmark)
    // ========================================================================

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }

        fn next_range(&mut self, min: u32, max: u32) -> u32 {
            let range = (max - min) as u64;
            if range == 0 {
                return min;
            }
            min + (self.next() % range) as u32
        }
    }

    #[derive(Clone)]
    pub struct DimacsGraph {
        pub num_nodes: usize,
        pub adjacency: Vec<Vec<(u32, u32)>>,
    }

    impl DimacsGraph {
        pub fn synthetic_sparse(num_nodes: usize, avg_degree: usize, seed: u64) -> Self {
            use std::collections::HashSet;

            let mut adjacency = vec![Vec::new(); num_nodes + 1];
            let mut rng = Lcg::new(seed);
            let mut edge_set: HashSet<(usize, usize)> = HashSet::new();

            #[allow(clippy::needless_range_loop)]
            for node in 1..=num_nodes {
                let degree = avg_degree + (rng.next() % 3) as usize;

                for _ in 0..degree {
                    let target = (rng.next() as usize % num_nodes) + 1;
                    if target != node && !edge_set.contains(&(node, target)) {
                        let weight = (rng.next() % 100 + 1) as u32;
                        adjacency[node].push((target as u32, weight));
                        edge_set.insert((node, target));
                    }
                }
            }

            DimacsGraph {
                num_nodes,
                adjacency,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct Query {
        pub source: u32,
        pub target: u32,
    }

    pub fn generate_queries_for_rank(
        graph: &DimacsGraph,
        target_log_rank: u32,
        num_queries: usize,
        seed: u64,
    ) -> Vec<Query> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut rng = Lcg::new(seed);
        let mut queries = Vec::new();
        let target_rank = 1u32 << target_log_rank;

        for _ in 0..num_queries * 2 {
            if queries.len() >= num_queries {
                break;
            }

            let source = rng.next_range(1, graph.num_nodes as u32 + 1);

            let mut dist: HashMap<u32, u32> = HashMap::new();
            let mut heap = BinaryHeap::new();
            let mut settled_count = 0u32;

            dist.insert(source, 0);
            heap.push(Reverse((0u32, source)));

            while let Some(Reverse((d, node))) = heap.pop() {
                if let Some(&best) = dist.get(&node) {
                    if d > best {
                        continue;
                    }
                }

                settled_count += 1;

                if settled_count == target_rank {
                    queries.push(Query {
                        source,
                        target: node,
                    });
                    break;
                }

                if let Some(neighbors) = graph.adjacency.get(node as usize) {
                    for &(neighbor, weight) in neighbors {
                        let new_dist = d + weight;
                        let should_update = dist.get(&neighbor).is_none_or(|&old| new_dist < old);

                        if should_update {
                            dist.insert(neighbor, new_dist);
                            heap.push(Reverse((new_dist, neighbor)));
                        }
                    }
                }
            }
        }

        queries
    }

    #[derive(Clone)]
    pub struct DimacsNode {
        pub id: u32,
        pub goal: u32,
        graph: Arc<DimacsGraph>,
    }

    impl DimacsNode {
        pub fn new(id: u32, goal: u32, graph: Arc<DimacsGraph>) -> Self {
            DimacsNode { id, goal, graph }
        }
    }

    impl PartialEq for DimacsNode {
        fn eq(&self, other: &Self) -> bool {
            self.id == other.id && self.goal == other.goal
        }
    }

    impl Eq for DimacsNode {}

    impl std::hash::Hash for DimacsNode {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.id.hash(state);
            self.goal.hash(state);
        }
    }

    impl std::fmt::Debug for DimacsNode {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("DimacsNode")
                .field("id", &self.id)
                .field("goal", &self.goal)
                .finish()
        }
    }

    impl SearchNode for DimacsNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, Self::Cost)> {
            if self.id as usize >= self.graph.adjacency.len() {
                return vec![];
            }

            self.graph.adjacency[self.id as usize]
                .iter()
                .map(|&(neighbor, weight)| {
                    (
                        DimacsNode {
                            id: neighbor,
                            goal: self.goal,
                            graph: Arc::clone(&self.graph),
                        },
                        weight,
                    )
                })
                .collect()
        }

        fn is_goal(&self) -> bool {
            self.id == self.goal
        }
    }

    // ========================================================================
    // Query runners
    // ========================================================================

    macro_rules! run_queries_optimized {
        ($name:ident, $heap:ty) => {
            fn $name(graph: &Arc<DimacsGraph>, queries: &[Query]) -> usize {
                let mut found = 0;
                for query in queries {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    if shortest_path::<_, $heap>(&start).is_some() {
                        found += 1;
                    }
                }
                found
            }
        };
    }

    macro_rules! run_queries_lazy {
        ($name:ident, $heap:ty) => {
            fn $name(graph: &Arc<DimacsGraph>, queries: &[Query]) -> usize {
                let mut found = 0;
                for query in queries {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    if shortest_path_lazy::<_, $heap>(&start).is_some() {
                        found += 1;
                    }
                }
                found
            }
        };
    }

    // Optimized runners (use decrease_key)
    run_queries_optimized!(run_queries_fibonacci_opt, FibonacciHeap<usize, _>);
    run_queries_optimized!(run_queries_pairing_opt, PairingHeap<usize, _>);
    // DISABLED: RankPairingHeap has severe performance regression
    // run_queries_optimized!(run_queries_rank_pairing_opt, RankPairingHeap<usize, _>);
    run_queries_optimized!(run_queries_binomial_opt, BinomialHeap<usize, _>);
    run_queries_optimized!(run_queries_strict_fibonacci_opt, StrictFibonacciHeap<usize, _>);
    run_queries_optimized!(run_queries_twothree_opt, TwoThreeHeap<usize, _>);
    run_queries_optimized!(run_queries_skew_binomial_opt, SkewBinomialHeap<usize, _>);
    run_queries_optimized!(run_queries_hollow_opt, HollowHeap<usize, _>);

    // Lazy runners (use re-insertion)
    run_queries_lazy!(run_queries_fibonacci_lazy, FibonacciHeap<usize, _>);
    run_queries_lazy!(run_queries_pairing_lazy, PairingHeap<usize, _>);
    run_queries_lazy!(run_queries_rank_pairing_lazy, RankPairingHeap<usize, _>);
    run_queries_lazy!(run_queries_simple_binary_lazy, SimpleBinaryHeap<usize, _>);
    run_queries_lazy!(run_queries_binomial_lazy, BinomialHeap<usize, _>);
    run_queries_lazy!(run_queries_strict_fibonacci_lazy, StrictFibonacciHeap<usize, _>);
    run_queries_lazy!(run_queries_twothree_lazy, TwoThreeHeap<usize, _>);
    run_queries_lazy!(run_queries_skew_binomial_lazy, SkewBinomialHeap<usize, _>);
    run_queries_lazy!(run_queries_skiplist_lazy, SkipListHeap<usize, _>);
    run_queries_lazy!(run_queries_hollow_lazy, HollowHeap<usize, _>);

    // ========================================================================
    // Benchmarks with perf measurement
    // ========================================================================

    /// Benchmark ALL heap implementations measuring multiple perf counters.
    ///
    /// This is the primary perf counter benchmark for comparing heaps.
    /// Groups queries by Dijkstra rank (number of nodes settled) to observe
    /// how each heap's performance metrics scale with problem size.
    ///
    /// Captures: wall clock time, instructions, cycles, branches, branch misses,
    /// cache references, and cache misses.
    fn benchmark_multi(c: &mut Criterion<PerfMultiMeasurement>) {
        let mut group = c.benchmark_group("perf_instructions");
        group.sample_size(10);

        // Use a 10M node graph - for 2^20 (1M) rank queries, we want the explored
        // region to be a small fraction of the graph to avoid boundary effects
        let graph = Arc::new(DimacsGraph::synthetic_sparse(10_000_000, 6, 12345));

        // Test ranks: 2^8 (256), 2^12 (4K), 2^16 (64K), 2^20 (1M)
        let rank_levels = [8, 12, 16, 20];

        for &log_rank in &rank_levels {
            let queries = generate_queries_for_rank(&graph, log_rank, 20, 77777 + log_rank as u64);

            if queries.len() < 5 {
                continue;
            }

            let rank_label = format!("2^{}", log_rank);

            // All optimized (decrease_key) implementations
            group.bench_with_input(
                BenchmarkId::new("fibonacci_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_fibonacci_opt(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("pairing_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_pairing_opt(&graph, qs))),
            );
            // DISABLED: RankPairingHeap has severe performance regression
            // group.bench_with_input(
            //     BenchmarkId::new("rank_pairing_opt", &rank_label),
            //     &queries,
            //     |b, qs| b.iter(|| black_box(run_queries_rank_pairing_opt(&graph, qs))),
            // );
            group.bench_with_input(
                BenchmarkId::new("binomial_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_binomial_opt(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("strict_fibonacci_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_strict_fibonacci_opt(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("twothree_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_twothree_opt(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("skew_binomial_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_skew_binomial_opt(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("hollow_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_hollow_opt(&graph, qs))),
            );

            // All lazy (re-insertion) implementations
            group.bench_with_input(
                BenchmarkId::new("simple_binary_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_simple_binary_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("fibonacci_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_fibonacci_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("pairing_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_pairing_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("rank_pairing_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_rank_pairing_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("binomial_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_binomial_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("twothree_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_twothree_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("strict_fibonacci_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_strict_fibonacci_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("skew_binomial_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_skew_binomial_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("skiplist_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_skiplist_lazy(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("hollow_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_hollow_lazy(&graph, qs))),
            );
        }

        group.finish();
    }

    criterion_group!(
        name = perf_benches;
        config = Criterion::default().with_measurement(PerfMultiMeasurement::new());
        targets = benchmark_multi
    );

    criterion_main!(perf_benches);
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
    // Optional CPU pinning for more stable results
    // Set BENCH_PIN_CPU=N to pin to CPU N (0-indexed)
    // Set BENCH_PIN_CPU=auto to pin to first available CPU
    if let Ok(pin_spec) = std::env::var("BENCH_PIN_CPU") {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        if core_ids.is_empty() {
            eprintln!("Warning: No CPU cores available for pinning");
        } else {
            let core_id = if pin_spec == "auto" {
                core_ids[0]
            } else if let Ok(n) = pin_spec.parse::<usize>() {
                if n < core_ids.len() {
                    core_ids[n]
                } else {
                    eprintln!(
                        "Warning: CPU {} not available (max: {}), using CPU 0",
                        n,
                        core_ids.len() - 1
                    );
                    core_ids[0]
                }
            } else {
                eprintln!(
                    "Warning: Invalid BENCH_PIN_CPU value '{}', using auto",
                    pin_spec
                );
                core_ids[0]
            };

            if core_affinity::set_for_current(core_id) {
                eprintln!("Pinned benchmark to CPU {:?}", core_id);
            } else {
                eprintln!("Warning: Failed to pin to CPU {:?}", core_id);
            }
        }
    }

    perf_benches::perf_benches();
}

#[cfg(not(all(feature = "perf-counters", target_os = "linux")))]
fn main() {
    eprintln!("Perf benchmarks require:");
    eprintln!("  1. Linux operating system");
    eprintln!("  2. --features perf-counters flag");
    eprintln!();
    eprintln!("Run with: cargo bench --features perf-counters --bench perf_benchmark");
    std::process::exit(1);
}
