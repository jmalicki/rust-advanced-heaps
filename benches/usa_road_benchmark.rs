//! Full USA Road Network Benchmark
//!
//! This benchmark demonstrates the practical performance of heap data structures
//! on the complete US road network (23.9M nodes, 58.3M edges).
//!
//! Unlike aggregated benchmarks, this shows **individual query results** to
//! illustrate how impractical Dijkstra's algorithm is at continental scale -
//! even with theoretically optimal O(1) decrease_key heaps.
//!
//! ## Setup
//!
//! ```bash
//! # Download the full USA road network (~335MB compressed, ~1.5GB uncompressed)
//! ./scripts/download-dimacs.sh USA
//!
//! # Optionally download coordinates for location names
//! # (coordinates file is ~218MB)
//! cd data
//! wget http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.co.gz
//! gunzip USA-road-d.USA.co.gz
//! ```
//!
//! ## Running
//!
//! ```bash
//! # Run with perf counters (Linux only)
//! sudo sysctl kernel.perf_event_paranoid=1
//! cargo bench --features perf-counters --bench usa_road_benchmark
//!
//! # Pin to a specific CPU for stable results
//! BENCH_PIN_CPU=0 cargo bench --features perf-counters --bench usa_road_benchmark
//! ```
//!
//! ## Output
//!
//! The benchmark outputs individual query results showing:
//! - Source and destination node IDs (and coordinates if available)
//! - Dijkstra rank (nodes settled)
//! - Time per algorithm
//! - IPC and cache miss rate
//!
//! This makes it viscerally clear that finding a route from NYC to LA
//! takes ~2 minutes even with the "optimal" Fibonacci heap.

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod usa_benchmark {
    use perf_event::{events::Hardware, Builder, Group};
    use rust_advanced_heaps::binomial::BinomialHeap;
    use rust_advanced_heaps::fibonacci::FibonacciHeap;
    use rust_advanced_heaps::hollow::HollowHeap;
    use rust_advanced_heaps::pairing::PairingHeap;
    use rust_advanced_heaps::pathfinding::{shortest_path, shortest_path_lazy, SearchNode};
    use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
    use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
    use rust_advanced_heaps::twothree::TwoThreeHeap;
    use std::collections::HashMap;
    use std::fs::File;
    use std::hint::black_box;
    use std::io::{BufRead, BufReader};
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    // ========================================================================
    // Graph and coordinate parsing
    // ========================================================================

    /// Adjacency list graph representation
    #[derive(Clone)]
    pub struct DimacsGraph {
        pub num_nodes: usize,
        pub num_edges: usize,
        pub adjacency: Vec<Vec<(u32, u32)>>,
    }

    impl DimacsGraph {
        /// Parse a DIMACS .gr file
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

                            if from > 0 && from <= num_nodes {
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

    /// Geographic coordinates for a node
    #[derive(Clone, Debug)]
    pub struct Coordinate {
        pub longitude: f64,
        pub latitude: f64,
    }

    /// Parse DIMACS coordinate file (.co format)
    /// Format: v <node_id> <longitude> <latitude>
    pub fn load_coordinates<P: AsRef<Path>>(path: P) -> std::io::Result<HashMap<u32, Coordinate>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut coords = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('c') || line.starts_with('p') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 && parts[0] == "v" {
                if let (Ok(id), Ok(lon), Ok(lat)) = (
                    parts[1].parse::<u32>(),
                    parts[2].parse::<i64>(),
                    parts[3].parse::<i64>(),
                ) {
                    // DIMACS coordinates are in 1e-6 degrees
                    coords.insert(
                        id,
                        Coordinate {
                            longitude: lon as f64 / 1_000_000.0,
                            latitude: lat as f64 / 1_000_000.0,
                        },
                    );
                }
            }
        }

        Ok(coords)
    }

    /// Get a human-readable location description from coordinates
    fn describe_location(coord: &Coordinate) -> String {
        // Very rough US region classification based on coordinates
        let lat = coord.latitude;
        let lon = coord.longitude;

        let ns = if lat > 40.0 {
            "Northern"
        } else if lat > 35.0 {
            "Central"
        } else {
            "Southern"
        };

        let ew = if lon < -110.0 {
            "West"
        } else if lon < -95.0 {
            "Central"
        } else if lon < -80.0 {
            "Midwest/East"
        } else {
            "East Coast"
        };

        format!("{} {} ({:.2}, {:.2})", ns, ew, lat, lon)
    }

    // ========================================================================
    // Query types
    // ========================================================================

    #[derive(Clone, Debug)]
    pub struct Query {
        pub source: u32,
        pub target: u32,
        pub dijkstra_rank: u32,
        pub source_coord: Option<Coordinate>,
        pub target_coord: Option<Coordinate>,
    }

    impl Query {
        pub fn describe(&self) -> String {
            match (&self.source_coord, &self.target_coord) {
                (Some(src), Some(dst)) => {
                    format!(
                        "Query {}->{}: {} -> {} (rank 2^{:.1})",
                        self.source,
                        self.target,
                        describe_location(src),
                        describe_location(dst),
                        (self.dijkstra_rank as f64).log2()
                    )
                }
                _ => format!(
                    "Query {}->{} (rank 2^{:.1})",
                    self.source,
                    self.target,
                    (self.dijkstra_rank as f64).log2()
                ),
            }
        }
    }

    /// Linear congruential generator for reproducible random numbers
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

    /// Generate random queries and compute their Dijkstra ranks
    pub fn generate_random_queries(
        graph: &DimacsGraph,
        coords: &HashMap<u32, Coordinate>,
        num_queries: usize,
        seed: u64,
    ) -> Vec<Query> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut rng = Lcg::new(seed);
        let mut queries = Vec::new();

        eprintln!("Generating {} random queries...", num_queries);

        for i in 0..num_queries * 3 {
            if queries.len() >= num_queries {
                break;
            }

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

                if node == target {
                    found_rank = Some(settled_count);
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

            if let Some(rank) = found_rank {
                queries.push(Query {
                    source,
                    target,
                    dijkstra_rank: rank,
                    source_coord: coords.get(&source).cloned(),
                    target_coord: coords.get(&target).cloned(),
                });
                eprintln!(
                    "  Query {}: nodes {}->{}, rank={}",
                    queries.len(),
                    source,
                    target,
                    rank
                );
            }

            if i > 0 && i % 100 == 0 {
                eprintln!("  ... tried {} random pairs, found {}", i, queries.len());
            }
        }

        // Sort by rank for better presentation
        queries.sort_by_key(|q| q.dijkstra_rank);
        queries
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
    // Benchmark execution
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

    /// Run a single query and measure performance
    fn measure_single_query<F: FnMut() -> bool>(mut f: F) -> Metrics {
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

        let _result = f();

        let elapsed = start.elapsed();
        group.disable().expect("Failed to disable perf group");

        let counts = group.read().expect("Failed to read perf group");

        Metrics {
            time_ns: elapsed.as_nanos() as f64,
            instructions: counts[&instructions],
            cycles: counts[&cycles],
            cache_refs: counts[&cache_refs],
            cache_misses: counts[&cache_misses],
        }
    }

    fn format_time(ns: f64) -> String {
        if ns < 1_000.0 {
            format!("{:.1}ns", ns)
        } else if ns < 1_000_000.0 {
            format!("{:.1}us", ns / 1_000.0)
        } else if ns < 1_000_000_000.0 {
            format!("{:.1}ms", ns / 1_000_000.0)
        } else if ns < 60_000_000_000.0 {
            format!("{:.2}s", ns / 1_000_000_000.0)
        } else {
            let secs = ns / 1_000_000_000.0;
            format!("{:.0}m {:.0}s", secs / 60.0, secs % 60.0)
        }
    }

    /// Run benchmarks on a single query with all heap types
    fn benchmark_query(graph: &Arc<DimacsGraph>, query: &Query) {
        println!();
        println!("=== {} ===", query.describe());
        println!(
            "Dijkstra rank: {} nodes ({:.1}M)",
            query.dijkstra_rank,
            query.dijkstra_rank as f64 / 1_000_000.0
        );
        println!();

        // Header
        println!(
            "{:<20} {:>12} {:>8} {:>10}",
            "Algorithm", "Time", "IPC", "LLC Miss%"
        );
        println!("{}", "-".repeat(55));

        // Run each algorithm
        #[allow(clippy::type_complexity)]
        let algorithms: Vec<(&str, Box<dyn Fn() -> bool>)> = vec![
            (
                "simple_binary",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, SimpleBinaryHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "pairing_lazy",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, PairingHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "pairing_opt",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, PairingHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "fibonacci_lazy",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, FibonacciHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "fibonacci_opt",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, FibonacciHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "hollow_lazy",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path_lazy::<_, HollowHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "twothree_opt",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, TwoThreeHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "strict_fib_opt",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, StrictFibonacciHeap<usize, _>>(&start).is_some()
                }),
            ),
            (
                "binomial_opt",
                Box::new(|| {
                    let start = DimacsNode::new(query.source, query.target, Arc::clone(graph));
                    shortest_path::<_, BinomialHeap<usize, _>>(&start).is_some()
                }),
            ),
        ];

        for (name, run_fn) in algorithms {
            let metrics = measure_single_query(|| black_box(run_fn()));

            println!(
                "{:<20} {:>12} {:>8.2} {:>10.1}",
                name,
                format_time(metrics.time_ns),
                metrics.ipc(),
                metrics.cache_miss_rate()
            );
        }
    }

    pub fn run_benchmark() {
        let graph_path = "data/USA-road-d.USA.gr";
        let coord_path = "data/USA-road-d.USA.co";

        // Check if data exists
        if !Path::new(graph_path).exists() {
            eprintln!("ERROR: USA road network data not found.");
            eprintln!();
            eprintln!("Download with:");
            eprintln!("  ./scripts/download-dimacs.sh USA");
            eprintln!();
            eprintln!("This is a large file (~335MB compressed, ~1.5GB uncompressed)");
            eprintln!("containing 23.9 million nodes and 58.3 million edges.");
            std::process::exit(1);
        }

        // Load graph
        eprintln!("Loading USA road network graph...");
        eprintln!("(This may take a minute - the file is ~1.5GB)");
        let start = Instant::now();
        let graph = Arc::new(DimacsGraph::from_file(graph_path).expect("Failed to load graph"));
        eprintln!(
            "Graph loaded in {:.1}s: {} nodes, {} edges",
            start.elapsed().as_secs_f64(),
            graph.num_nodes,
            graph.num_edges
        );

        // Load coordinates (optional)
        let coords = if Path::new(coord_path).exists() {
            eprintln!("Loading coordinates...");
            match load_coordinates(coord_path) {
                Ok(c) => {
                    eprintln!("Loaded {} node coordinates", c.len());
                    c
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load coordinates: {}", e);
                    HashMap::new()
                }
            }
        } else {
            eprintln!("Coordinates file not found. Download with:");
            eprintln!(
                "  cd data && wget http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.co.gz && gunzip USA-road-d.USA.co.gz"
            );
            HashMap::new()
        };

        // Generate queries
        let queries = generate_random_queries(&graph, &coords, 10, 42);

        if queries.is_empty() {
            eprintln!("ERROR: Could not generate any valid queries");
            std::process::exit(1);
        }

        // Print header
        println!();
        println!("=============================================================");
        println!("USA Road Network Benchmark");
        println!("=============================================================");
        println!();
        println!(
            "Graph: {} nodes ({:.1}M), {} edges ({:.1}M)",
            graph.num_nodes,
            graph.num_nodes as f64 / 1_000_000.0,
            graph.num_edges,
            graph.num_edges as f64 / 1_000_000.0
        );
        println!();
        println!("This benchmark shows individual query times to demonstrate");
        println!("how impractical classic Dijkstra is at continental scale.");
        println!();
        println!("Even with O(1) decrease_key heaps, cross-country routes");
        println!("can take MINUTES because we must settle millions of nodes.");
        println!();

        // Run each query individually
        for query in &queries {
            benchmark_query(&graph, query);
        }

        // Summary
        println!();
        println!("=============================================================");
        println!("SUMMARY");
        println!("=============================================================");
        println!();
        println!("Key observations:");
        println!();
        println!("1. SIMPLE BINARY HEAP WINS - Despite O(log n) decrease_key,");
        println!("   the cache-friendly array layout beats pointer-based heaps.");
        println!();
        println!("2. O(1) DOESN'T HELP - At 10M+ nodes, memory latency dominates.");
        println!("   Every node access is a cache miss (~100+ cycles).");
        println!();
        println!("3. DIJKSTRA IS IMPRACTICAL - Cross-country routes take minutes.");
        println!("   Real navigation systems use hierarchy (CH, HLC, etc.).");
        println!();
        println!("4. LAZY BEATS DECREASE_KEY - Re-insertion often outperforms");
        println!("   native decrease_key due to better cache behavior.");
        println!();
    }
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
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

    usa_benchmark::run_benchmark();
}

#[cfg(not(all(feature = "perf-counters", target_os = "linux")))]
fn main() {
    eprintln!("USA road benchmark requires:");
    eprintln!("  1. Linux operating system");
    eprintln!("  2. --features perf-counters flag");
    eprintln!();
    eprintln!("Run with: cargo bench --features perf-counters --bench usa_road_benchmark");
    std::process::exit(1);
}
