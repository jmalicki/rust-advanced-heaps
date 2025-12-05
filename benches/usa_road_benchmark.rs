//! Full USA Road Network Benchmark
//!
//! This benchmark demonstrates the practical performance of heap data structures
//! on the complete US road network (23.9M nodes, 58.3M edges).
//!
//! ## Design
//!
//! - Generates queries across many Dijkstra ranks (2^8 through 2^24) for scaling graphs
//! - Runs heap implementations in parallel on separate pinned CPUs
//! - Outputs CSV for easy plotting
//!
//! ## Setup
//!
//! ```bash
//! # Download the full USA road network (~335MB compressed, ~1.5GB uncompressed)
//! ./scripts/download-dimacs.sh USA
//! ```
//!
//! ## Running
//!
//! ```bash
//! # Enable perf counters (Linux only)
//! sudo sysctl kernel.perf_event_paranoid=1
//!
//! # Step 1: Generate queries (saves to data/usa_queries.json)
//! cargo bench --features perf-counters --bench usa_road_benchmark -- generate
//!
//! # Step 2: Run all heaps in parallel (16 CPUs)
//! cargo bench --features perf-counters --bench usa_road_benchmark -- run-parallel
//!
//! # Or run a specific heap on a specific CPU:
//! BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench usa_road_benchmark -- run simple_binary
//! BENCH_PIN_CPU=1 cargo bench --features perf-counters --bench usa_road_benchmark -- run pairing_opt
//! ```
//!
//! ## Output
//!
//! Results are saved to `data/usa_bench_<heap>.csv` with columns:
//! - query_id, source, target, dijkstra_rank, log2_rank
//! - time_ns, instructions, cycles, cache_refs, cache_misses
//! - ipc, cache_miss_rate

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod usa_benchmark {
    use perf_event::{events::Hardware, Builder, Group};
    use rust_advanced_heaps::binomial::BinomialHeap;
    #[cfg(feature = "arena-storage")]
    use rust_advanced_heaps::binomial::BinomialHeapArena;
    use rust_advanced_heaps::fibonacci::FibonacciHeap;
    use rust_advanced_heaps::hollow::HollowHeap;
    use rust_advanced_heaps::pairing::PairingHeap;
    use rust_advanced_heaps::pathfinding::{shortest_path, shortest_path_lazy, SearchNode};
    use rust_advanced_heaps::rank_pairing::RankPairingHeap;
    use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
    #[cfg(feature = "arena-storage")]
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeapArena;
    use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
    use rust_advanced_heaps::twothree::TwoThreeHeap;
    use std::collections::HashMap;
    use std::fs::File;
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::time::Instant;

    // ========================================================================
    // Constants
    // ========================================================================

    const GRAPH_PATH: &str = "data/USA-road-d.USA.gr";
    const QUERIES_PATH: &str = "data/usa_queries.json";
    const RESULTS_DIR: &str = "data";

    /// Dijkstra rank buckets: 2^10, 2^12, 2^14, 2^16, 2^18
    /// (skip 2^8 as local queries are rare on large graphs)
    /// (skip 2^20+ as finding queries with exact rank is too slow)
    const RANK_BUCKETS: &[u32] = &[10, 12, 14, 16, 18];

    /// Number of queries per rank bucket
    const QUERIES_PER_BUCKET: usize = 2;

    // ========================================================================
    // Graph parsing
    // ========================================================================

    #[derive(Clone)]
    pub struct DimacsGraph {
        pub num_nodes: usize,
        pub num_edges: usize,
        pub adjacency: Vec<Vec<(u32, u32)>>,
    }

    impl DimacsGraph {
        pub fn from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
            let file = File::open(path)?;
            let reader = BufReader::new(file);

            let mut num_nodes = 0;
            let mut num_edges = 0;
            let mut adjacency: Vec<Vec<(u32, u32)>> = Vec::new();

            for line in reader.lines() {
                let line = line?;
                let line = line.trim();

                if line.is_empty() || line.starts_with('c') {
                    continue;
                }

                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.is_empty() {
                    continue;
                }

                match parts[0] {
                    "p" => {
                        if parts.len() >= 4 && parts[1] == "sp" {
                            num_nodes = parts[2].parse().unwrap_or(0);
                            num_edges = parts[3].parse().unwrap_or(0);
                            adjacency = vec![Vec::new(); num_nodes + 1];
                        }
                    }
                    "a" => {
                        if parts.len() >= 4 {
                            let from: usize = parts[1].parse().unwrap_or(0);
                            let to: u32 = parts[2].parse().unwrap_or(0);
                            let weight: u32 = parts[3].parse().unwrap_or(1);

                            if from > 0 && from <= num_nodes && to > 0 && (to as usize) <= num_nodes
                            {
                                adjacency[from].push((to, weight));
                            }
                        }
                    }
                    _ => {}
                }
            }

            Ok(DimacsGraph {
                num_nodes,
                num_edges,
                adjacency,
            })
        }
    }

    // ========================================================================
    // Query generation and storage
    // ========================================================================

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct Query {
        pub id: usize,
        pub source: u32,
        pub target: u32,
        pub dijkstra_rank: u32,
        pub rank_bucket: u32, // log2 of target rank
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    pub struct QuerySet {
        pub graph_nodes: usize,
        pub graph_edges: usize,
        pub queries: Vec<Query>,
    }

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

    /// Generate queries for each rank bucket
    pub fn generate_queries(graph: &DimacsGraph, seed: u64) -> QuerySet {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut rng = Lcg::new(seed);
        let mut queries = Vec::new();
        let mut query_id = 0;

        for &target_log_rank in RANK_BUCKETS {
            let target_rank_min = 1u32 << target_log_rank;
            let target_rank_max = 1u32 << (target_log_rank + 1);

            eprintln!(
                "Generating {} queries for rank bucket 2^{} ({}-{})...",
                QUERIES_PER_BUCKET, target_log_rank, target_rank_min, target_rank_max
            );

            let mut bucket_queries = 0;
            let mut attempts = 0;
            const MAX_ATTEMPTS: usize = 10000;

            while bucket_queries < QUERIES_PER_BUCKET && attempts < MAX_ATTEMPTS {
                attempts += 1;

                let source = rng.next_range(1, graph.num_nodes as u32 + 1);
                let target = rng.next_range(1, graph.num_nodes as u32 + 1);

                if source == target {
                    continue;
                }

                // Compute Dijkstra rank
                let mut dist: HashMap<u32, u32> = HashMap::new();
                let mut heap = BinaryHeap::new();
                let mut settled_count = 0u32;

                dist.insert(source, 0);
                heap.push(Reverse((0u32, source)));

                let mut found_rank = None;
                while let Some(Reverse((d, node))) = heap.pop() {
                    if let Some(&best) = dist.get(&node) {
                        if d > best {
                            continue;
                        }
                    }

                    settled_count += 1;

                    // Early termination if we've exceeded the target bucket
                    if settled_count > target_rank_max {
                        break;
                    }

                    if node == target {
                        found_rank = Some(settled_count);
                        break;
                    }

                    if let Some(neighbors) = graph.adjacency.get(node as usize) {
                        for &(neighbor, weight) in neighbors {
                            let new_dist = d + weight;
                            let should_update =
                                dist.get(&neighbor).is_none_or(|&old| new_dist < old);

                            if should_update {
                                dist.insert(neighbor, new_dist);
                                heap.push(Reverse((new_dist, neighbor)));
                            }
                        }
                    }
                }

                if let Some(rank) = found_rank {
                    if rank >= target_rank_min && rank < target_rank_max {
                        queries.push(Query {
                            id: query_id,
                            source,
                            target,
                            dijkstra_rank: rank,
                            rank_bucket: target_log_rank,
                        });
                        query_id += 1;
                        bucket_queries += 1;
                        eprintln!(
                            "  Found query {}: {}->{}, rank={} (2^{:.1})",
                            query_id,
                            source,
                            target,
                            rank,
                            (rank as f64).log2()
                        );
                    }
                }
            }

            if bucket_queries < QUERIES_PER_BUCKET {
                eprintln!(
                    "  Warning: Only found {} queries for bucket 2^{}",
                    bucket_queries, target_log_rank
                );
            }
        }

        QuerySet {
            graph_nodes: graph.num_nodes,
            graph_edges: graph.num_edges,
            queries,
        }
    }

    pub fn save_queries(query_set: &QuerySet, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, query_set).map_err(std::io::Error::other)
    }

    pub fn load_queries(path: &str) -> std::io::Result<QuerySet> {
        let file = File::open(path)?;
        serde_json::from_reader(file).map_err(std::io::Error::other)
    }

    // ========================================================================
    // Pathfinding node
    // ========================================================================

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
    // Benchmark measurement
    // ========================================================================

    #[derive(Clone, Debug, Default)]
    pub struct Metrics {
        pub time_ns: u64,
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

    fn measure_query<F: FnMut() -> bool>(mut f: F) -> Metrics {
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

        group.enable().expect("Failed to enable perf group");
        let start = Instant::now();

        let _result = black_box(f());

        let elapsed = start.elapsed();
        group.disable().expect("Failed to disable perf group");

        let counts = group.read().expect("Failed to read perf group");

        Metrics {
            time_ns: elapsed.as_nanos() as u64,
            instructions: counts[&instructions],
            cycles: counts[&cycles],
            cache_refs: counts[&cache_refs],
            cache_misses: counts[&cache_misses],
        }
    }

    // ========================================================================
    // Heap runners
    // ========================================================================

    pub fn run_heap(
        heap_name: &str,
        graph: &Arc<DimacsGraph>,
        queries: &[Query],
    ) -> Vec<(Query, Metrics)> {
        let mut results = Vec::with_capacity(queries.len());

        for query in queries {
            let metrics = match heap_name {
                "simple_binary" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, SimpleBinaryHeap<usize, _>>(&start).is_some()
                }),
                "pairing_lazy" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, PairingHeap<usize, _>>(&start).is_some()
                }),
                "pairing_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, PairingHeap<usize, _>>(&start).is_some()
                }),
                "fibonacci_lazy" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, FibonacciHeap<usize, _>>(&start).is_some()
                }),
                "fibonacci_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, FibonacciHeap<usize, _>>(&start).is_some()
                }),
                "rank_pairing_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, RankPairingHeap<usize, _>>(&start).is_some()
                }),
                "hollow_lazy" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, HollowHeap<usize, _>>(&start).is_some()
                }),
                "twothree_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, TwoThreeHeap<usize, _>>(&start).is_some()
                }),
                "strict_fib_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, StrictFibonacciHeap<usize, _>>(&start).is_some()
                }),
                "binomial_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, BinomialHeap<usize, _>>(&start).is_some()
                }),
                "skew_binomial_opt" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, SkewBinomialHeap<usize, _>>(&start).is_some()
                }),
                #[cfg(feature = "arena-storage")]
                "binomial_arena" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, BinomialHeapArena<usize, _>>(&start).is_some()
                }),
                #[cfg(feature = "arena-storage")]
                "skew_binomial_arena" => measure_query(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, SkewBinomialHeapArena<usize, _>>(&start).is_some()
                }),
                _ => {
                    eprintln!("Unknown heap: {}", heap_name);
                    continue;
                }
            };

            eprintln!(
                "  Query {}: rank=2^{}, time={:.2}ms",
                query.id,
                query.rank_bucket,
                metrics.time_ns as f64 / 1_000_000.0
            );

            results.push((query.clone(), metrics));
        }

        results
    }

    pub fn save_results(heap_name: &str, results: &[(Query, Metrics)]) -> std::io::Result<()> {
        let path = format!("{}/usa_bench_{}.csv", RESULTS_DIR, heap_name);
        let mut file = File::create(&path)?;

        // CSV header
        writeln!(
            file,
            "query_id,source,target,dijkstra_rank,log2_rank,time_ns,instructions,cycles,cache_refs,cache_misses,ipc,cache_miss_rate"
        )?;

        for (query, metrics) in results {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{:.4},{:.4}",
                query.id,
                query.source,
                query.target,
                query.dijkstra_rank,
                query.rank_bucket,
                metrics.time_ns,
                metrics.instructions,
                metrics.cycles,
                metrics.cache_refs,
                metrics.cache_misses,
                metrics.ipc(),
                metrics.cache_miss_rate()
            )?;
        }

        eprintln!("Saved results to {}", path);
        Ok(())
    }

    // ========================================================================
    // Parallel execution
    // ========================================================================

    const BASE_HEAPS: &[&str] = &[
        "simple_binary",
        "pairing_lazy",
        "pairing_opt",
        "fibonacci_lazy",
        "fibonacci_opt",
        "rank_pairing_opt",
        "hollow_lazy",
        "twothree_opt",
        "strict_fib_opt",
        "binomial_opt",
        "skew_binomial_opt",
    ];

    #[cfg(feature = "arena-storage")]
    const ARENA_HEAPS: &[&str] = &["binomial_arena", "skew_binomial_arena"];

    fn all_heaps() -> Vec<&'static str> {
        let mut heaps: Vec<&str> = BASE_HEAPS.to_vec();
        #[cfg(feature = "arena-storage")]
        heaps.extend_from_slice(ARENA_HEAPS);
        heaps
    }

    pub fn run_parallel() {
        let heaps = all_heaps();

        // Get the path to the current executable
        let exe = std::env::current_exe().expect("Failed to get current executable");

        eprintln!("Launching {} heap benchmarks in parallel...", heaps.len());
        eprintln!("Executable: {:?}", exe);

        let mut children = Vec::new();

        for (cpu, heap_name) in heaps.iter().enumerate() {
            eprintln!("  Starting {} on CPU {}", heap_name, cpu);

            // The benchmark binary is already compiled - we just need to pass the right args
            // The binary expects: run <heap_name> (no --bench needed, that's for cargo)
            let child = Command::new(&exe)
                .arg("run")
                .arg(heap_name)
                .env("BENCH_PIN_CPU", cpu.to_string())
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("Failed to spawn child process");

            children.push((heap_name, child));
        }

        // Wait for all children
        for (heap_name, mut child) in children {
            let status = child.wait().expect("Failed to wait for child");
            if status.success() {
                eprintln!("  {} completed successfully", heap_name);
            } else {
                eprintln!("  {} failed with status: {}", heap_name, status);
            }
        }

        eprintln!("\nAll benchmarks complete!");
        eprintln!("Results saved to {}/usa_bench_*.csv", RESULTS_DIR);
    }

    // ========================================================================
    // CPU pinning
    // ========================================================================

    pub fn pin_to_cpu() {
        if let Ok(pin_spec) = std::env::var("BENCH_PIN_CPU") {
            let core_ids = core_affinity::get_core_ids().unwrap_or_default();
            if !core_ids.is_empty() {
                let core_id = if let Ok(n) = pin_spec.parse::<usize>() {
                    if n < core_ids.len() {
                        core_ids[n]
                    } else {
                        eprintln!("Warning: CPU {} not available, using CPU 0", n);
                        core_ids[0]
                    }
                } else {
                    core_ids[0]
                };

                if core_affinity::set_for_current(core_id) {
                    eprintln!("Pinned to CPU {:?}", core_id);
                } else {
                    eprintln!("Warning: Failed to pin to CPU {:?}", core_id);
                }
            }
        }
    }

    // ========================================================================
    // Main entry points
    // ========================================================================

    pub fn cmd_generate() {
        if !Path::new(GRAPH_PATH).exists() {
            eprintln!("ERROR: USA road network data not found at {}", GRAPH_PATH);
            eprintln!("Download with: ./scripts/download-dimacs.sh USA");
            std::process::exit(1);
        }

        eprintln!("Loading graph...");
        let start = Instant::now();
        let graph = DimacsGraph::from_file(GRAPH_PATH).expect("Failed to load graph");
        eprintln!(
            "Loaded {} nodes, {} edges in {:.1}s",
            graph.num_nodes,
            graph.num_edges,
            start.elapsed().as_secs_f64()
        );

        eprintln!(
            "\nGenerating queries across {} rank buckets...",
            RANK_BUCKETS.len()
        );
        let query_set = generate_queries(&graph, 42);

        eprintln!(
            "\nSaving {} queries to {}...",
            query_set.queries.len(),
            QUERIES_PATH
        );
        save_queries(&query_set, QUERIES_PATH).expect("Failed to save queries");

        eprintln!("Done! Run benchmarks with:");
        eprintln!(
            "  cargo bench --features perf-counters --bench usa_road_benchmark -- run-parallel"
        );
    }

    pub fn cmd_run(heap_name: &str) {
        pin_to_cpu();

        if !Path::new(QUERIES_PATH).exists() {
            eprintln!("ERROR: Queries not found at {}", QUERIES_PATH);
            eprintln!("Generate with: cargo bench --features perf-counters --bench usa_road_benchmark -- generate");
            std::process::exit(1);
        }

        eprintln!("Loading queries from {}...", QUERIES_PATH);
        let query_set = load_queries(QUERIES_PATH).expect("Failed to load queries");

        eprintln!("Loading graph...");
        let start = Instant::now();
        let graph = Arc::new(DimacsGraph::from_file(GRAPH_PATH).expect("Failed to load graph"));
        eprintln!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

        eprintln!(
            "\nRunning {} on {} queries...",
            heap_name,
            query_set.queries.len()
        );
        let results = run_heap(heap_name, &graph, &query_set.queries);

        save_results(heap_name, &results).expect("Failed to save results");
    }

    pub fn cmd_run_parallel() {
        if !Path::new(QUERIES_PATH).exists() {
            eprintln!("ERROR: Queries not found at {}", QUERIES_PATH);
            eprintln!("Generate with: cargo bench --features perf-counters --bench usa_road_benchmark -- generate");
            std::process::exit(1);
        }

        run_parallel();
    }

    pub fn cmd_help() {
        eprintln!("USA Road Network Benchmark");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  cargo bench --features perf-counters --bench usa_road_benchmark -- <command>");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  generate      Generate queries across Dijkstra rank buckets");
        eprintln!("  run <heap>    Run benchmark for a specific heap");
        eprintln!("  run-parallel  Run all heaps in parallel on separate CPUs");
        eprintln!("  list          List available heap implementations");
        eprintln!("  help          Show this help");
        eprintln!();
        eprintln!("Environment:");
        eprintln!("  BENCH_PIN_CPU=N   Pin to CPU N (used by run command)");
        eprintln!();
        eprintln!("Available heaps:");
        for heap in all_heaps() {
            eprintln!("  {}", heap);
        }
    }

    pub fn cmd_list() {
        eprintln!("Available heap implementations:");
        for (i, heap) in all_heaps().iter().enumerate() {
            eprintln!("  {:2}. {}", i, heap);
        }
    }
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Skip benchmark harness args.
    // cargo bench passes additional arguments before our "--" separator,
    // so we skip until we find our known commands or the separator.
    let args: Vec<&str> = args
        .iter()
        .map(|s| s.as_str())
        .skip_while(|&s| {
            s != "--"
                && !s.starts_with("generate")
                && !s.starts_with("run")
                && !s.starts_with("list")
                && !s.starts_with("help")
        })
        .filter(|&s| s != "--")
        .collect();

    match args.first().copied() {
        Some("generate") => usa_benchmark::cmd_generate(),
        Some("run-parallel") => usa_benchmark::cmd_run_parallel(),
        Some("run") => {
            if let Some(heap_name) = args.get(1) {
                usa_benchmark::cmd_run(heap_name);
            } else {
                eprintln!("ERROR: Missing heap name");
                eprintln!("Usage: ... -- run <heap_name>");
                usa_benchmark::cmd_list();
                std::process::exit(1);
            }
        }
        Some("list") => usa_benchmark::cmd_list(),
        Some("help") | Some("--help") | Some("-h") => usa_benchmark::cmd_help(),
        _ => usa_benchmark::cmd_help(),
    }
}

#[cfg(not(all(feature = "perf-counters", target_os = "linux")))]
fn main() {
    eprintln!("USA road benchmark requires:");
    eprintln!("  1. Linux operating system");
    eprintln!("  2. --features perf-counters flag");
    eprintln!();
    eprintln!("Run with: cargo bench --features perf-counters --bench usa_road_benchmark -- help");
    std::process::exit(1);
}
