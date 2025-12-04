//! Hardware Performance Counter Benchmarks
//!
//! Runs the same workloads as dimacs_benchmark but measures hardware counters
//! (instructions, cycles, cache misses) instead of wall-clock time.
//!
//! ## Running
//!
//! ```bash
//! # Enable perf access (requires Linux)
//! sudo sysctl kernel.perf_event_paranoid=1
//!
//! # Run with instructions measurement (default)
//! cargo bench --features perf-counters --bench perf_benchmark
//!
//! # The benchmark will output instruction counts instead of time
//! ```
//!
//! ## Metrics
//!
//! Each run measures ONE metric. Run multiple times with different configurations
//! to compare metrics. Available modes in criterion-linux-perf:
//! - Instructions: Total instructions retired
//! - Cycles: CPU cycles (affected by frequency scaling)
//! - CacheMisses: Last-level cache misses
//! - CacheRefs: Cache references
//! - Branches: Branch instructions
//! - BranchMisses: Mispredicted branches

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod perf_benches {
    use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
    use criterion_linux_perf::{PerfMeasurement, PerfMode};
    use rust_advanced_heaps::fibonacci::FibonacciHeap;
    use rust_advanced_heaps::hollow::HollowHeap;
    use rust_advanced_heaps::pairing::PairingHeap;
    use rust_advanced_heaps::pathfinding::{shortest_path, shortest_path_lazy, SearchNode};
    use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
    use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
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
            if self.id as usize > self.graph.adjacency.len() {
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

    run_queries_optimized!(run_queries_fibonacci_opt, FibonacciHeap<usize, _>);
    run_queries_optimized!(run_queries_pairing_opt, PairingHeap<usize, _>);
    run_queries_optimized!(run_queries_strict_fibonacci_opt, StrictFibonacciHeap<usize, _>);
    run_queries_optimized!(run_queries_hollow_opt, HollowHeap<usize, _>);

    run_queries_lazy!(run_queries_simple_binary_lazy, SimpleBinaryHeap<usize, _>);
    run_queries_lazy!(run_queries_hollow_lazy, HollowHeap<usize, _>);

    // ========================================================================
    // Benchmarks with perf measurement
    // ========================================================================

    /// Benchmark measuring instruction count
    fn benchmark_instructions(c: &mut Criterion<PerfMeasurement>) {
        let mut group = c.benchmark_group("perf_instructions");
        group.sample_size(10);

        let graph = Arc::new(DimacsGraph::synthetic_sparse(20_000, 6, 12345));
        let rank_levels = [10, 12, 14];

        for &log_rank in &rank_levels {
            let queries = generate_queries_for_rank(&graph, log_rank, 20, 77777 + log_rank as u64);

            if queries.len() < 5 {
                continue;
            }

            let rank_label = format!("2^{}", log_rank);

            // Optimized implementations
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
            group.bench_with_input(
                BenchmarkId::new("hollow_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_hollow_opt(&graph, qs))),
            );
            group.bench_with_input(
                BenchmarkId::new("strict_fibonacci_opt", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_strict_fibonacci_opt(&graph, qs))),
            );

            // Lazy implementations
            group.bench_with_input(
                BenchmarkId::new("simple_binary_lazy", &rank_label),
                &queries,
                |b, qs| b.iter(|| black_box(run_queries_simple_binary_lazy(&graph, qs))),
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
        config = Criterion::default().with_measurement(PerfMeasurement::new(PerfMode::Instructions));
        targets = benchmark_instructions
    );

    criterion_main!(perf_benches);
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
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
