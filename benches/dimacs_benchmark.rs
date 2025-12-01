//! DIMACS Shortest Path Benchmarks
//!
//! Benchmarks the pathfinding implementation using DIMACS format graphs.
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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::pathfinding::{dijkstra, SearchNode};
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

/// Adjacency list graph representation
#[derive(Clone)]
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
                    // Problem line: p sp <nodes> <edges>
                    if parts.len() >= 4 && parts[1] == "sp" {
                        num_nodes = parts[2].parse().unwrap_or(0);
                        num_edges = parts[3].parse().unwrap_or(0);
                        adjacency = vec![Vec::new(); num_nodes + 1]; // 1-indexed
                    }
                }
                "a" => {
                    // Arc/edge line: a <from> <to> <weight>
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

    /// Create a synthetic graph for testing (when no DIMACS file available)
    pub fn synthetic_grid(width: usize, height: usize) -> Self {
        let num_nodes = width * height;
        let mut adjacency = vec![Vec::new(); num_nodes + 1];

        // Create a grid graph (1-indexed)
        for y in 0..height {
            for x in 0..width {
                let node = y * width + x + 1; // 1-indexed

                // Right neighbor
                if x + 1 < width {
                    let right = node + 1;
                    adjacency[node].push((right as u32, 1));
                    adjacency[right].push((node as u32, 1));
                }

                // Down neighbor
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
    pub fn synthetic_sparse(num_nodes: usize, avg_degree: usize) -> Self {
        use std::collections::HashSet;

        let mut adjacency = vec![Vec::new(); num_nodes + 1];
        let mut rng_state: u64 = 12345; // Simple PRNG

        let next_rand = |state: &mut u64| -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        };

        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();

        for node in 1..=num_nodes {
            let degree = avg_degree + (next_rand(&mut rng_state) % 3) as usize;

            for _ in 0..degree {
                let target = (next_rand(&mut rng_state) as usize % num_nodes) + 1;
                if target != node && !edge_set.contains(&(node, target)) {
                    let weight = (next_rand(&mut rng_state) % 100 + 1) as u32;
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

    /// Create a dense graph (for comparison)
    pub fn synthetic_dense(num_nodes: usize) -> Self {
        let mut adjacency = vec![Vec::new(); num_nodes + 1];
        let mut rng_state: u64 = 54321;

        let next_rand = |state: &mut u64| -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        };

        // Connect ~50% of possible edges
        for from in 1..=num_nodes {
            for to in 1..=num_nodes {
                if from != to && next_rand(&mut rng_state) % 2 == 0 {
                    let weight = (next_rand(&mut rng_state) % 100 + 1) as u32;
                    adjacency[from].push((to as u32, weight));
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

/// Node for pathfinding on a DIMACS graph
#[derive(Clone)]
pub struct DimacsNode {
    /// Current node ID (1-indexed as per DIMACS)
    pub id: u32,
    /// Goal node ID
    pub goal: u32,
    /// Reference to the graph (shared)
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

/// Benchmark dijkstra on a graph with different heap implementations
fn benchmark_dijkstra_heaps(c: &mut Criterion) {
    // Create synthetic graphs of different sizes and densities
    let test_cases = vec![
        ("grid_100x100", DimacsGraph::synthetic_grid(100, 100)),
        ("grid_200x200", DimacsGraph::synthetic_grid(200, 200)),
        ("sparse_10k_deg4", DimacsGraph::synthetic_sparse(10_000, 4)),
        ("sparse_50k_deg4", DimacsGraph::synthetic_sparse(50_000, 4)),
        ("dense_500", DimacsGraph::synthetic_dense(500)),
    ];

    let mut group = c.benchmark_group("dijkstra_heaps");

    for (name, graph) in test_cases {
        let graph = Arc::new(graph);
        let num_nodes = graph.num_nodes;

        // Pick start and goal that are far apart
        let start_id = 1u32;
        let goal_id = num_nodes as u32;

        group.throughput(Throughput::Elements(graph.num_edges as u64));

        // Benchmark with FibonacciHeap
        group.bench_with_input(
            BenchmarkId::new("fibonacci", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, FibonacciHeap<_, _>>(&start))
                });
            },
        );

        // Benchmark with PairingHeap
        group.bench_with_input(
            BenchmarkId::new("pairing", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, PairingHeap<_, _>>(&start))
                });
            },
        );

        // Benchmark with RankPairingHeap
        group.bench_with_input(
            BenchmarkId::new("rank_pairing", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, RankPairingHeap<_, _>>(&start))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark graph density impact on decrease_key frequency
fn benchmark_density_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_impact");

    // Same number of nodes, different densities
    let sparse = Arc::new(DimacsGraph::synthetic_sparse(5000, 3)); // ~3 edges/node
    let medium = Arc::new(DimacsGraph::synthetic_sparse(5000, 10)); // ~10 edges/node
    let dense = Arc::new(DimacsGraph::synthetic_sparse(5000, 30)); // ~30 edges/node

    let graphs = vec![("sparse_deg3", sparse), ("medium_deg10", medium), ("dense_deg30", dense)];

    for (name, graph) in graphs {
        let start_id = 1u32;
        let goal_id = graph.num_nodes as u32;

        group.bench_with_input(
            BenchmarkId::new("fibonacci", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, FibonacciHeap<_, _>>(&start))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, PairingHeap<_, _>>(&start))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark loading and running on a real DIMACS file (if present)
fn benchmark_real_dimacs(c: &mut Criterion) {
    // Download DIMACS data from: http://www.diag.uniroma1.it/challenge9/download.shtml
    // Extract .gr files to data/ directory (git-ignored)
    let dimacs_paths = vec![
        "data/USA-road-d.NY.gr",   // New York (~260K nodes)
        "data/USA-road-d.BAY.gr",  // San Francisco Bay (~320K nodes)
        "data/USA-road-d.COL.gr",  // Colorado (~430K nodes)
    ];

    let mut group = c.benchmark_group("real_dimacs");
    group.sample_size(10); // Fewer samples for large graphs

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

        let start_id = 1u32;
        let goal_id = graph.num_nodes as u32;

        group.throughput(Throughput::Elements(graph.num_edges as u64));

        group.bench_with_input(
            BenchmarkId::new("fibonacci", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, FibonacciHeap<_, _>>(&start))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pairing", name),
            &(Arc::clone(&graph), start_id, goal_id),
            |b, (g, s, goal)| {
                b.iter(|| {
                    let start = DimacsNode::new(*s, *goal, Arc::clone(g));
                    black_box(dijkstra::<_, PairingHeap<_, _>>(&start))
                });
            },
        );
    }

    group.finish();
}

/// Quick sanity check that pathfinding works correctly
fn benchmark_correctness_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("correctness");

    // Small grid where we know the answer
    let grid = Arc::new(DimacsGraph::synthetic_grid(10, 10));

    group.bench_function("grid_10x10_corner_to_corner", |b| {
        b.iter(|| {
            let start = DimacsNode::new(1, 100, Arc::clone(&grid));
            let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
            assert!(result.is_some());
            let (_, cost) = result.unwrap();
            // Manhattan distance from (0,0) to (9,9) on a grid is 18
            assert_eq!(cost, 18);
            black_box(cost)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_dijkstra_heaps,
    benchmark_density_comparison,
    benchmark_real_dimacs,
    benchmark_correctness_check,
);

criterion_main!(benches);
