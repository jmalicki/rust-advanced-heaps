//! Multi-metric performance table benchmark
//!
//! Runs benchmarks and displays results in a table format with:
//! - Rows: algorithms
//! - Columns: sizes (2^8, 2^12, 2^16, 2^20)
//! - Sub-columns: time, IPC, cache miss rate
//!
//! ## Running
//!
//! ```bash
//! # Enable perf access (requires Linux)
//! sudo sysctl kernel.perf_event_paranoid=1
//!
//! # Run the table benchmark
//! cargo bench --features perf-counters --bench perf_table
//!
//! # Run only specific sizes (in parallel on different CPUs)
//! cargo bench --features perf-counters --bench perf_table -- 8    # 2^8 only
//! cargo bench --features perf-counters --bench perf_table -- 12   # 2^12 only
//! ```
//!
//! ## Parallel execution across sizes
//!
//! Run 4 sizes on 4 CPUs simultaneously:
//! ```bash
//! BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench perf_table -- 8 &
//! BENCH_PIN_CPU=1 cargo bench --features perf-counters --bench perf_table -- 12 &
//! BENCH_PIN_CPU=2 cargo bench --features perf-counters --bench perf_table -- 16 &
//! BENCH_PIN_CPU=3 cargo bench --features perf-counters --bench perf_table -- 20 &
//! wait
//! ```

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod perf_table {
    use perf_event::{events::Hardware, Builder, Group};
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
    use std::time::Instant;

    // ========================================================================
    // Graph and query infrastructure
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
    // Metrics collection
    // ========================================================================

    #[derive(Clone, Debug, Default)]
    pub struct Metrics {
        pub time_ns: f64,
        pub instructions: u64,
        pub cycles: u64,
        pub cache_refs: u64,
        pub cache_misses: u64,
    }

    impl Metrics {
        pub fn ipc(&self) -> f64 {
            if self.cycles == 0 {
                0.0
            } else {
                self.instructions as f64 / self.cycles as f64
            }
        }

        pub fn cache_miss_rate(&self) -> f64 {
            if self.cache_refs == 0 {
                0.0
            } else {
                self.cache_misses as f64 / self.cache_refs as f64 * 100.0
            }
        }
    }

    fn measure<F: FnMut()>(mut f: F, iterations: usize) -> Metrics {
        // Create perf group
        let mut group = Group::new().expect("Failed to create perf group");

        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()
            .expect("Failed to add instructions counter");
        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()
            .expect("Failed to add cycles counter");
        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()
            .expect("Failed to add cache_refs counter");
        let cache_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()
            .expect("Failed to add cache_misses counter");

        // Warmup
        for _ in 0..2 {
            f();
        }

        // Measure
        group.enable().expect("Failed to enable perf group");
        let start = Instant::now();

        for _ in 0..iterations {
            f();
        }

        let elapsed = start.elapsed();
        group.disable().expect("Failed to disable perf group");

        let counts = group.read().expect("Failed to read perf group");

        Metrics {
            time_ns: elapsed.as_nanos() as f64 / iterations as f64,
            instructions: counts[&instructions] / iterations as u64,
            cycles: counts[&cycles] / iterations as u64,
            cache_refs: counts[&cache_refs] / iterations as u64,
            cache_misses: counts[&cache_misses] / iterations as u64,
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
    // Table display
    // ========================================================================

    fn format_time(ns: f64) -> String {
        if ns < 1_000.0 {
            format!("{:.1}ns", ns)
        } else if ns < 1_000_000.0 {
            format!("{:.1}us", ns / 1_000.0)
        } else if ns < 1_000_000_000.0 {
            format!("{:.1}ms", ns / 1_000_000.0)
        } else {
            format!("{:.2}s", ns / 1_000_000_000.0)
        }
    }

    pub fn run_benchmarks(filter_rank: Option<u32>) {
        // Smaller graph for faster benchmarks: 2M nodes
        eprintln!("Generating synthetic graph (2M nodes)...");
        let graph = Arc::new(DimacsGraph::synthetic_sparse(2_000_000, 6, 12345));
        eprintln!("Graph generated.");

        let all_rank_levels = [8u32, 12, 16, 20];
        let rank_levels: Vec<u32> = if let Some(r) = filter_rank {
            vec![r]
        } else {
            all_rank_levels.to_vec()
        };

        let iterations = 5; // Number of iterations per measurement

        // Define all algorithms
        type RunFn = fn(&Arc<DimacsGraph>, &[Query]) -> usize;
        let algorithms: Vec<(&str, RunFn)> = vec![
            // Optimized (decrease_key)
            ("fibonacci_opt", run_queries_fibonacci_opt as RunFn),
            ("pairing_opt", run_queries_pairing_opt),
            ("binomial_opt", run_queries_binomial_opt),
            ("strict_fib_opt", run_queries_strict_fibonacci_opt),
            ("twothree_opt", run_queries_twothree_opt),
            ("skew_binom_opt", run_queries_skew_binomial_opt),
            ("hollow_opt", run_queries_hollow_opt),
            // Lazy (re-insertion)
            ("simple_binary", run_queries_simple_binary_lazy),
            ("fibonacci_lazy", run_queries_fibonacci_lazy),
            ("pairing_lazy", run_queries_pairing_lazy),
            ("rank_pair_lazy", run_queries_rank_pairing_lazy),
            ("binomial_lazy", run_queries_binomial_lazy),
            ("twothree_lazy", run_queries_twothree_lazy),
            ("strict_fib_lazy", run_queries_strict_fibonacci_lazy),
            ("skew_binom_lazy", run_queries_skew_binomial_lazy),
            ("skiplist_lazy", run_queries_skiplist_lazy),
            ("hollow_lazy", run_queries_hollow_lazy),
        ];

        // Generate queries for each rank level
        eprintln!("Generating queries for each rank level...");
        let mut queries_by_rank: HashMap<u32, Vec<Query>> = HashMap::new();
        for &log_rank in &rank_levels {
            let queries = generate_queries_for_rank(&graph, log_rank, 10, 77777 + log_rank as u64);
            eprintln!("  2^{}: {} queries", log_rank, queries.len());
            queries_by_rank.insert(log_rank, queries);
        }

        // Run benchmarks and print results immediately
        // Format: algorithm,rank,time_ms,ipc,cache_miss_pct
        println!();
        println!("algorithm,rank,time_ms,ipc,cache_miss_pct");

        for (algo_name, run_fn) in &algorithms {
            for &log_rank in &rank_levels {
                let queries = &queries_by_rank[&log_rank];
                if queries.is_empty() {
                    continue;
                }

                let metrics = measure(
                    || {
                        black_box(run_fn(&graph, queries));
                    },
                    iterations,
                );

                // Print CSV format for easy parsing
                println!(
                    "{},{},{:.2},{:.2},{:.1}",
                    algo_name,
                    log_rank,
                    metrics.time_ns / 1_000_000.0, // Convert to ms
                    metrics.ipc(),
                    metrics.cache_miss_rate()
                );

                // Also print human-readable to stderr
                eprintln!(
                    "{:<18} 2^{:<2}: {:>10}  IPC={:.2}  miss={:.1}%",
                    algo_name,
                    log_rank,
                    format_time(metrics.time_ns),
                    metrics.ipc(),
                    metrics.cache_miss_rate()
                );
            }
        }
    }
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
    // Parse command line for rank filter
    let args: Vec<String> = std::env::args().collect();
    let filter_rank: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    // Optional CPU pinning
    if let Ok(pin_spec) = std::env::var("BENCH_PIN_CPU") {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        if !core_ids.is_empty() {
            let core_id = if pin_spec == "auto" {
                core_ids[0]
            } else if let Ok(n) = pin_spec.parse::<usize>() {
                if n < core_ids.len() {
                    core_ids[n]
                } else {
                    core_ids[0]
                }
            } else {
                core_ids[0]
            };

            if core_affinity::set_for_current(core_id) {
                eprintln!("Pinned to CPU {:?}", core_id);
            }
        }
    }

    perf_table::run_benchmarks(filter_rank);
}

#[cfg(not(all(feature = "perf-counters", target_os = "linux")))]
fn main() {
    eprintln!("Perf table benchmark requires:");
    eprintln!("  1. Linux operating system");
    eprintln!("  2. --features perf-counters flag");
    eprintln!();
    eprintln!("Run with: cargo bench --features perf-counters --bench perf_table");
    std::process::exit(1);
}
