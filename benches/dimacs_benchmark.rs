//! DIMACS Shortest Path Benchmarks
//!
//! Benchmarks the pathfinding implementation using DIMACS format graphs,
//! following methodology from the 9th DIMACS Implementation Challenge and
//! related literature (Sanders/Schultes Highway Hierarchies).
//!
//! ## Methodology
//!
//! - **Random queries**: Uses seeded PRNG for reproducible random (source, target) pairs
//! - **Dijkstra rank grouping**: Queries grouped by difficulty (2^8, 2^12, 2^16, 2^20, 2^24)
//! - **Multiple queries**: Each benchmark runs multiple queries and reports averages
//!
//! ## Setup
//!
//! Download real datasets from: <http://www.diag.uniroma1.it/challenge9/download.shtml>
//!
//! For New York dataset:
//! ```sh
//! mkdir -p data
//! cd data
//! wget http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.NY.gr.gz
//! gunzip USA-road-d.NY.gr.gz
//! ```
//!
//! The `data/` directory is git-ignored.
//!
//! ## DIMACS .gr format
//!
//! - Lines starting with 'c' are comments
//! - Line 'p sp n m' defines problem: n nodes, m edges
//! - Lines 'a u v w' define edge from node u to node v with weight w

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

// ============================================================================
// Simple PRNG for reproducible benchmarks
// ============================================================================

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

// ============================================================================
// Graph representation
// ============================================================================

/// Adjacency list graph representation
#[derive(Clone, Default)]
pub struct DimacsGraph {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Adjacency list: node -> [(neighbor, weight), ...]
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

    /// Create a synthetic grid graph
    pub fn synthetic_grid(width: usize, height: usize) -> Self {
        let num_nodes = width * height;
        let mut adjacency = vec![Vec::new(); num_nodes + 1];

        for y in 0..height {
            for x in 0..width {
                let node = y * width + x + 1;

                if x + 1 < width {
                    let right = node + 1;
                    adjacency[node].push((right as u32, 1));
                    adjacency[right].push((node as u32, 1));
                }

                if y + 1 < height {
                    let down = node + width;
                    adjacency[node].push((down as u32, 1));
                    adjacency[down].push((node as u32, 1));
                }
            }
        }

        let num_edges = adjacency.iter().map(|v| v.len()).sum();

        DimacsGraph {
            num_nodes,
            num_edges,
            adjacency,
        }
    }

    /// Create a synthetic sparse random graph
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

        let num_edges = adjacency.iter().map(|v| v.len()).sum();

        DimacsGraph {
            num_nodes,
            num_edges,
            adjacency,
        }
    }
}

// ============================================================================
// Query generation following DIMACS methodology
// ============================================================================

/// A source-target query pair with precomputed Dijkstra rank
#[derive(Clone, Debug)]
pub struct Query {
    pub source: u32,
    pub target: u32,
    pub dijkstra_rank: u32, // Number of nodes settled before target
}

/// Generate random query pairs and compute their Dijkstra ranks
pub fn generate_queries_with_ranks(
    graph: &DimacsGraph,
    num_queries: usize,
    seed: u64,
) -> Vec<Query> {
    let mut rng = Lcg::new(seed);
    let mut queries = Vec::with_capacity(num_queries);

    for _ in 0..num_queries {
        let source = rng.next_range(1, graph.num_nodes as u32 + 1);
        let target = rng.next_range(1, graph.num_nodes as u32 + 1);

        if source != target {
            // Compute Dijkstra rank by running Dijkstra and counting settled nodes
            let rank = compute_dijkstra_rank(graph, source, target);
            queries.push(Query {
                source,
                target,
                dijkstra_rank: rank,
            });
        }
    }

    queries
}

/// Compute Dijkstra rank: number of nodes settled before reaching target
fn compute_dijkstra_rank(graph: &DimacsGraph, source: u32, target: u32) -> u32 {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

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

        if node == target {
            return settled_count;
        }

        if let Some(neighbors) = graph.adjacency.get(node as usize) {
            for &(neighbor, weight) in neighbors {
                let new_dist = d + weight;
                let should_update = dist
                    .get(&neighbor)
                    .map(|&old| new_dist < old)
                    .unwrap_or(true);

                if should_update {
                    dist.insert(neighbor, new_dist);
                    heap.push(Reverse((new_dist, neighbor)));
                }
            }
        }
    }

    // Target not reachable
    u32::MAX
}

/// Group queries by Dijkstra rank into buckets (powers of 2)
pub fn group_queries_by_rank(queries: Vec<Query>) -> HashMap<u32, Vec<Query>> {
    let mut groups: HashMap<u32, Vec<Query>> = HashMap::new();

    // Rank buckets: 2^8, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20
    let buckets = [8, 10, 12, 14, 16, 18, 20];

    for query in queries {
        if query.dijkstra_rank == u32::MAX {
            continue; // Skip unreachable queries
        }

        // Find appropriate bucket
        let log_rank = (query.dijkstra_rank as f64).log2().floor() as u32;

        for &bucket in &buckets {
            if log_rank <= bucket {
                groups.entry(bucket).or_default().push(query);
                break;
            }
        }
    }

    groups
}

/// Generate queries with exact Dijkstra ranks using Sanders/Schultes methodology.
///
/// For each query:
/// 1. Pick a random source node
/// 2. Run Dijkstra from source, recording settlement order
/// 3. Pick the target as the node settled at exactly rank 2^target_log_rank
///
/// This guarantees exact ranks and is more efficient than trial-and-error.
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
        // Try more sources than needed
        if queries.len() >= num_queries {
            break;
        }

        let source = rng.next_range(1, graph.num_nodes as u32 + 1);

        // Run Dijkstra from source, find the node settled at target_rank
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

            // Found our target: the node settled at exactly target_rank
            if settled_count == target_rank {
                queries.push(Query {
                    source,
                    target: node,
                    dijkstra_rank: target_rank,
                });
                break;
            }

            if let Some(neighbors) = graph.adjacency.get(node as usize) {
                for &(neighbor, weight) in neighbors {
                    let new_dist = d + weight;
                    let should_update = dist
                        .get(&neighbor)
                        .map(|&old| new_dist < old)
                        .unwrap_or(true);

                    if should_update {
                        dist.insert(neighbor, new_dist);
                        heap.push(Reverse((new_dist, neighbor)));
                    }
                }
            }
        }
        // If we didn't reach target_rank nodes, this source doesn't work - try next
    }

    queries
}

// ============================================================================
// Pathfinding node
// ============================================================================

/// Node for pathfinding on a DIMACS graph
#[derive(Clone, Default)]
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

// ============================================================================
// Benchmark runners for multiple queries
// ============================================================================

/// Macro to generate optimized (decrease_key) query runners
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

/// Macro to generate lazy (re-insertion) query runners
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
run_queries_optimized!(run_queries_rank_pairing_opt, RankPairingHeap<usize, _>);
run_queries_optimized!(run_queries_binomial_opt, BinomialHeap<usize, _>);
run_queries_optimized!(run_queries_strict_fibonacci_opt, StrictFibonacciHeap<usize, _>);
run_queries_optimized!(run_queries_twothree_opt, TwoThreeHeap<usize, _>);
run_queries_optimized!(run_queries_skew_binomial_opt, SkewBinomialHeap<usize, _>);
run_queries_optimized!(run_queries_hollow_opt, HollowHeap<usize, _>);
// DISABLED: skiplist_opt is 8x slower than skiplist_lazy
// run_queries_optimized!(run_queries_skiplist_opt, SkipListHeap<usize, _>);

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

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark comparing heaps on random queries (DIMACS methodology)
fn benchmark_random_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_queries");
    group.sample_size(20);

    // Use a moderate-sized synthetic graph for quick benchmarks
    let graph = Arc::new(DimacsGraph::synthetic_sparse(10_000, 6, 12345));

    // Generate 100 random queries
    let queries: Vec<Query> = {
        let mut rng = Lcg::new(54321);
        (0..100)
            .filter_map(|_| {
                let source = rng.next_range(1, graph.num_nodes as u32 + 1);
                let target = rng.next_range(1, graph.num_nodes as u32 + 1);
                if source != target {
                    Some(Query {
                        source,
                        target,
                        dijkstra_rank: 0, // Not needed for this benchmark
                    })
                } else {
                    None
                }
            })
            .collect()
    };

    // Optimized (decrease_key) implementations
    group.bench_function("fibonacci_opt", |b| {
        b.iter(|| black_box(run_queries_fibonacci_opt(&graph, &queries)));
    });

    group.bench_function("pairing_opt", |b| {
        b.iter(|| black_box(run_queries_pairing_opt(&graph, &queries)));
    });

    group.bench_function("rank_pairing_opt", |b| {
        b.iter(|| black_box(run_queries_rank_pairing_opt(&graph, &queries)));
    });

    // Lazy (re-insertion) implementations
    group.bench_function("fibonacci_lazy", |b| {
        b.iter(|| black_box(run_queries_fibonacci_lazy(&graph, &queries)));
    });

    group.bench_function("pairing_lazy", |b| {
        b.iter(|| black_box(run_queries_pairing_lazy(&graph, &queries)));
    });

    group.bench_function("rank_pairing_lazy", |b| {
        b.iter(|| black_box(run_queries_rank_pairing_lazy(&graph, &queries)));
    });

    group.bench_function("simple_binary_lazy", |b| {
        b.iter(|| black_box(run_queries_simple_binary_lazy(&graph, &queries)));
    });

    group.finish();
}

/// Benchmark by Dijkstra rank (Sanders/Schultes methodology)
///
/// Groups queries by difficulty: short (local) to long (cross-graph)
fn benchmark_by_dijkstra_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("dijkstra_rank");
    group.sample_size(10);

    // Use a larger graph to have meaningful rank distribution
    let graph = Arc::new(DimacsGraph::synthetic_sparse(20_000, 6, 12345));

    // Test different Dijkstra rank levels
    // 2^10 = 1K nodes, 2^12 = 4K nodes, 2^14 = 16K nodes (most of graph)
    let rank_levels = [10, 12, 14];

    for &log_rank in &rank_levels {
        let queries = generate_queries_for_rank(&graph, log_rank, 50, 99999 + log_rank as u64);

        if queries.len() < 10 {
            continue; // Skip if not enough queries found
        }

        let rank_label = format!("2^{}", log_rank);

        // Optimized implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_opt", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_fibonacci_opt(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing_opt", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_pairing_opt(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rank_pairing_opt", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_rank_pairing_opt(&graph, qs)));
            },
        );

        // Lazy implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_lazy", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_fibonacci_lazy(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing_lazy", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_pairing_lazy(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simple_binary_lazy", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_simple_binary_lazy(&graph, qs)));
            },
        );
    }

    group.finish();
}

/// Benchmark across different graph scales
fn benchmark_graph_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_scale");
    group.sample_size(10);

    let scales = [
        ("5k", 5_000),
        ("10k", 10_000),
        ("20k", 20_000),
        ("50k", 50_000),
    ];

    for (name, num_nodes) in scales {
        let graph = Arc::new(DimacsGraph::synthetic_sparse(num_nodes, 6, 12345));

        // Generate 50 random queries per scale
        let queries: Vec<Query> = {
            let mut rng = Lcg::new(54321);
            (0..50)
                .filter_map(|_| {
                    let source = rng.next_range(1, graph.num_nodes as u32 + 1);
                    let target = rng.next_range(1, graph.num_nodes as u32 + 1);
                    if source != target {
                        Some(Query {
                            source,
                            target,
                            dijkstra_rank: 0,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Optimized implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_opt", name),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_fibonacci_opt(&graph, qs)));
            },
        );

        group.bench_with_input(BenchmarkId::new("pairing_opt", name), &queries, |b, qs| {
            b.iter(|| black_box(run_queries_pairing_opt(&graph, qs)));
        });

        group.bench_with_input(
            BenchmarkId::new("rank_pairing_opt", name),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_rank_pairing_opt(&graph, qs)));
            },
        );

        // Lazy implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_lazy", name),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_fibonacci_lazy(&graph, qs)));
            },
        );

        group.bench_with_input(BenchmarkId::new("pairing_lazy", name), &queries, |b, qs| {
            b.iter(|| black_box(run_queries_pairing_lazy(&graph, qs)));
        });

        group.bench_with_input(
            BenchmarkId::new("simple_binary_lazy", name),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_simple_binary_lazy(&graph, qs)));
            },
        );
    }

    group.finish();
}

/// Benchmark on real DIMACS road networks (if available)
fn benchmark_real_dimacs(c: &mut Criterion) {
    let dimacs_paths = vec![
        "data/USA-road-d.NY.gr",
        "data/USA-road-d.BAY.gr",
        "data/USA-road-d.COL.gr",
    ];

    let mut group = c.benchmark_group("real_dimacs");
    group.sample_size(10);

    for path in dimacs_paths {
        if !Path::new(path).exists() {
            continue;
        }

        let graph = match DimacsGraph::from_file(path) {
            Ok(g) => Arc::new(g),
            Err(_) => continue,
        };

        let name = Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        // Generate 100 random queries following DIMACS methodology
        let queries: Vec<Query> = {
            let mut rng = Lcg::new(42);
            (0..100)
                .filter_map(|_| {
                    let source = rng.next_range(1, graph.num_nodes as u32 + 1);
                    let target = rng.next_range(1, graph.num_nodes as u32 + 1);
                    if source != target {
                        Some(Query {
                            source,
                            target,
                            dijkstra_rank: 0,
                        })
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Optimized implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_opt", name),
            &(Arc::clone(&graph), queries.clone()),
            |b, (g, qs)| {
                b.iter(|| black_box(run_queries_fibonacci_opt(g, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing_opt", name),
            &(Arc::clone(&graph), queries.clone()),
            |b, (g, qs)| {
                b.iter(|| black_box(run_queries_pairing_opt(g, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rank_pairing_opt", name),
            &(Arc::clone(&graph), queries.clone()),
            |b, (g, qs)| {
                b.iter(|| black_box(run_queries_rank_pairing_opt(g, qs)));
            },
        );

        // Lazy implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_lazy", name),
            &(Arc::clone(&graph), queries.clone()),
            |b, (g, qs)| {
                b.iter(|| black_box(run_queries_fibonacci_lazy(g, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing_lazy", name),
            &(Arc::clone(&graph), queries.clone()),
            |b, (g, qs)| {
                b.iter(|| black_box(run_queries_pairing_lazy(g, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simple_binary_lazy", name),
            &(Arc::clone(&graph), queries),
            |b, (g, qs)| {
                b.iter(|| black_box(run_queries_simple_binary_lazy(g, qs)));
            },
        );
    }

    group.finish();
}

/// Benchmark by Dijkstra rank on real DIMACS data
fn benchmark_real_dimacs_by_rank(c: &mut Criterion) {
    let path = "data/USA-road-d.NY.gr";

    if !Path::new(path).exists() {
        return;
    }

    let graph = match DimacsGraph::from_file(path) {
        Ok(g) => Arc::new(g),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("real_dimacs_by_rank");
    group.sample_size(10);

    // Test across different Dijkstra rank levels
    // Road networks typically have ~260K nodes for NY, so test up to 2^18
    let rank_levels = [12, 14, 16, 18];

    for &log_rank in &rank_levels {
        let queries = generate_queries_for_rank(&graph, log_rank, 30, 88888 + log_rank as u64);

        if queries.len() < 5 {
            continue;
        }

        let rank_label = format!("2^{}", log_rank);

        // Optimized implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_opt", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_fibonacci_opt(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing_opt", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_pairing_opt(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rank_pairing_opt", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_rank_pairing_opt(&graph, qs)));
            },
        );

        // Lazy implementations
        group.bench_with_input(
            BenchmarkId::new("fibonacci_lazy", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_fibonacci_lazy(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing_lazy", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_pairing_lazy(&graph, qs)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simple_binary_lazy", &rank_label),
            &queries,
            |b, qs| {
                b.iter(|| black_box(run_queries_simple_binary_lazy(&graph, qs)));
            },
        );
    }

    group.finish();
}

/// Quick correctness sanity check
fn benchmark_correctness_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("correctness");

    let grid = Arc::new(DimacsGraph::synthetic_grid(10, 10));

    group.bench_function("grid_10x10_corner_to_corner", |b| {
        b.iter(|| {
            let start = DimacsNode::new(1, 100, Arc::clone(&grid));
            let result = shortest_path::<_, FibonacciHeap<_, _>>(&start);
            assert!(result.is_some());
            let (_, cost) = result.unwrap();
            assert_eq!(cost, 18);
            black_box(cost)
        });
    });

    group.finish();
}

/// Benchmark ALL heap implementations by Dijkstra rank.
///
/// This is the primary benchmark for identifying aberrant big-O patterns.
/// By grouping queries by rank (number of nodes settled), we can observe
/// how each heap's performance scales with problem size.
fn benchmark_all_heaps_by_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_heaps_by_rank");
    group.sample_size(10);

    // Use a 20K node graph - large enough for meaningful ranks
    let graph = Arc::new(DimacsGraph::synthetic_sparse(20_000, 6, 12345));

    // Test ranks: 2^10 (1K), 2^12 (4K), 2^14 (16K)
    let rank_levels = [10, 12, 14];

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
        // DISABLED: RankPairingHeap has severe performance regression (8x slower than expected)
        // See: https://github.com/jmalicki/rust-advanced-heaps/issues/54
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
        // DISABLED: SkipListHeap optimized is 8x slower than lazy variant
        // The lazy (re-insertion) approach is far more efficient for this heap
        // group.bench_with_input(
        //     BenchmarkId::new("skiplist_opt", &rank_label),
        //     &queries,
        //     |b, qs| b.iter(|| black_box(run_queries_skiplist_opt(&graph, qs))),
        // );

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

/// Large-scale benchmark on California road network (1.9M nodes).
///
/// Tests high Dijkstra ranks (2^16, 2^17, 2^18, 2^19, 2^20) to stress-test heaps
/// and determine if O(1) amortized decrease_key provides practical benefit at scale.
/// This is the "galactic algorithm" test - do Fibonacci/Pairing heaps ever catch up?
fn benchmark_california_high_rank(c: &mut Criterion) {
    let path = "data/USA-road-d.CAL.gr";

    if !Path::new(path).exists() {
        eprintln!("California dataset not found. Download with: ./scripts/download-dimacs.sh CAL");
        return;
    }

    eprintln!("Loading California graph (1.9M nodes, 4.7M edges)...");
    let graph = match DimacsGraph::from_file(path) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("Failed to load California graph: {}", e);
            return;
        }
    };
    eprintln!(
        "Graph loaded: {} nodes, {} edges",
        graph.num_nodes, graph.num_edges
    );

    let mut group = c.benchmark_group("california_high_rank");
    group.sample_size(10);

    // High Dijkstra ranks: 2^16 (65K), 2^17 (131K), 2^18 (262K), 2^19 (524K), 2^20 (1M)
    // With 1.9M nodes, we can test up to 2^20 comfortably
    let rank_levels = [16, 17, 18, 19, 20];

    for &log_rank in &rank_levels {
        eprintln!("Generating queries for rank 2^{}...", log_rank);
        let queries = generate_queries_for_rank(&graph, log_rank, 10, 11111 + log_rank as u64);

        if queries.len() < 3 {
            eprintln!("Not enough queries for rank 2^{}, skipping", log_rank);
            continue;
        }
        eprintln!(
            "Generated {} queries for rank 2^{}",
            queries.len(),
            log_rank
        );

        let rank_label = format!("2^{}", log_rank);

        // Optimized (decrease_key) implementations - the ones that should shine at scale
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
            BenchmarkId::new("hollow_opt", &rank_label),
            &queries,
            |b, qs| b.iter(|| black_box(run_queries_hollow_opt(&graph, qs))),
        );

        // Lazy implementations - simpler, but O(log n) per relaxation
        group.bench_with_input(
            BenchmarkId::new("hollow_lazy", &rank_label),
            &queries,
            |b, qs| b.iter(|| black_box(run_queries_hollow_lazy(&graph, qs))),
        );
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
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_random_queries,
    benchmark_by_dijkstra_rank,
    benchmark_graph_scale,
    benchmark_real_dimacs,
    benchmark_real_dimacs_by_rank,
    benchmark_correctness_check,
    benchmark_all_heaps_by_rank,
    benchmark_california_high_rank,
);

criterion_main!(benches);
