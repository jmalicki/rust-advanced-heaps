//! Dijkstra's and A* pathfinding algorithms using heap data structures
//!
//! This module provides generic graph search implementations that work with both:
//! - Simple heaps (base [`Heap`] trait) - uses lazy Dijkstra (re-insertion)
//! - Advanced heaps ([`DecreaseKeyHeap`] trait) - uses efficient `decrease_key`
//!
//! # Design
//!
//! Dijkstra and A* are the same algorithm - A* just adds a heuristic to guide
//! the search. This module unifies them: implement [`SearchNode`] and optionally
//! override `heuristic()` (defaults to zero, giving Dijkstra behavior).
//!
//! The node type carries its own goal context and implements `is_goal()` to
//! determine when the search should terminate.
//!
//! # Requirements
//!
//! **All edge costs and heuristics must be non-negative.** The algorithms in this
//! module assume non-negative values. Negative costs will produce incorrect results.
//! For A* to find optimal paths, the heuristic must also be *admissible* (never
//! overestimate the actual cost to the goal).
//!
//! # Example: Dijkstra (no heuristic)
//!
//! ```rust
//! use rust_advanced_heaps::pathfinding::{SearchNode, shortest_path};
//! use rust_advanced_heaps::pairing::PairingHeap;
//!
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct Node { value: i32, goal: i32 }
//!
//! impl SearchNode for Node {
//!     type Cost = u32;
//!     fn successors(&self) -> Vec<(Self, u32)> {
//!         vec![(Node { value: self.value + 1, goal: self.goal }, 1)]
//!     }
//!     fn is_goal(&self) -> bool { self.value == self.goal }
//!     // heuristic() defaults to 0, so this is Dijkstra's algorithm
//! }
//!
//! let start = Node { value: 0, goal: 5 };
//! let (path, cost) = shortest_path::<_, PairingHeap<_, _>>(&start).unwrap();
//! assert_eq!(cost, 5);
//! ```
//!
//! # Example: A* (with heuristic)
//!
//! ```rust
//! use rust_advanced_heaps::pathfinding::{SearchNode, shortest_path};
//! use rust_advanced_heaps::pairing::PairingHeap;
//!
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct GridPos { x: i32, y: i32, goal_x: i32, goal_y: i32 }
//!
//! impl SearchNode for GridPos {
//!     type Cost = u32;
//!
//!     fn successors(&self) -> Vec<(Self, Self::Cost)> {
//!         vec![
//!             (GridPos { x: self.x + 1, y: self.y, goal_x: self.goal_x, goal_y: self.goal_y }, 1),
//!             (GridPos { x: self.x, y: self.y + 1, goal_x: self.goal_x, goal_y: self.goal_y }, 1),
//!         ]
//!     }
//!
//!     fn is_goal(&self) -> bool {
//!         self.x == self.goal_x && self.y == self.goal_y
//!     }
//!
//!     fn heuristic(&self) -> u32 {
//!         // Manhattan distance - makes this A* search
//!         ((self.goal_x - self.x).abs() + (self.goal_y - self.y).abs()) as u32
//!     }
//! }
//!
//! let start = GridPos { x: 0, y: 0, goal_x: 3, goal_y: 3 };
//! let (path, cost) = shortest_path::<_, PairingHeap<_, _>>(&start).unwrap();
//! assert_eq!(cost, 6);
//! ```
//!
//! # Using Simple Heaps vs DecreaseKeyHeap
//!
//! ```rust
//! use rust_advanced_heaps::pathfinding::{SearchNode, shortest_path, shortest_path_lazy};
//! use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
//! use rust_advanced_heaps::pairing::PairingHeap;
//!
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct Node { value: i32, goal: i32 }
//!
//! impl SearchNode for Node {
//!     type Cost = u32;
//!     fn successors(&self) -> Vec<(Self, u32)> {
//!         if self.value < 100 {
//!             vec![(Node { value: self.value + 1, goal: self.goal }, 1)]
//!         } else {
//!             vec![]
//!         }
//!     }
//!     fn is_goal(&self) -> bool { self.value == self.goal }
//! }
//!
//! let start = Node { value: 0, goal: 5 };
//!
//! // With simple heap (no decrease_key) - uses lazy Dijkstra
//! let result = shortest_path_lazy::<_, SimpleBinaryHeap<_, _>>(&start);
//! assert_eq!(result.unwrap().1, 5);
//!
//! // With DecreaseKeyHeap - uses efficient decrease_key
//! let result = shortest_path::<_, PairingHeap<_, _>>(&start);
//! assert_eq!(result.unwrap().1, 5);
//! ```

use crate::traits::{DecreaseKeyHeap, Handle, Heap};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Add;

/// Trait for types that can be used as costs in pathfinding algorithms.
///
/// This requires the type to be orderable, copyable, and support addition.
/// It also requires a zero value for initialization.
///
/// **Important:** All cost values must be non-negative. The algorithms assume
/// non-negative costs and will produce incorrect results with negative values.
pub trait Cost: Ord + Copy + Add<Output = Self> + Default {}

impl<T> Cost for T where T: Ord + Copy + Add<Output = Self> + Default {}

/// Trait for nodes in a search graph.
///
/// Implement this trait for your node type to use Dijkstra's or A* algorithms.
/// The node type must be hashable and cloneable for efficient storage.
///
/// The node carries all context needed to:
/// - Generate successors
/// - Check if it's a goal
/// - (Optionally) compute heuristics for A*
pub trait SearchNode: Clone + Eq + Hash {
    /// The cost type for edge weights (e.g., u32, u64, f64 wrapped in OrderedFloat)
    type Cost: Cost;

    /// Returns all successor nodes along with the cost to reach them.
    ///
    /// This is where you define your graph structure. Each call should return
    /// all neighbors reachable from this node along with their edge costs.
    ///
    /// **Important:** All edge costs must be non-negative. Negative costs will
    /// cause the algorithm to produce incorrect results.
    fn successors(&self) -> Vec<(Self, Self::Cost)>;

    /// Returns true if this node is a goal state.
    ///
    /// The node should carry enough context to determine this (e.g., a reference
    /// to the goal position, or the problem instance).
    fn is_goal(&self) -> bool;

    /// Returns a heuristic estimate of the cost from this node to any goal.
    ///
    /// **Requirements:**
    /// - Must be non-negative (≥ 0)
    /// - Must be admissible: never overestimate the actual cost to the goal
    ///
    /// Common heuristics include:
    /// - Manhattan distance for grid-based movement
    /// - Euclidean distance for arbitrary movement
    /// - Zero (reduces to Dijkstra's algorithm)
    ///
    /// The default implementation returns zero, which makes the search equivalent
    /// to Dijkstra's algorithm. Override this method to enable A* search.
    fn heuristic(&self) -> Self::Cost {
        Self::Cost::default()
    }
}

/// A wrapper for costs in the heap that orders by f-score.
///
/// Lower costs have higher priority (min-heap behavior).
#[derive(Debug, Clone, Copy)]
pub struct PriorityCost<C> {
    /// The f-score: g + h (where h=0 for Dijkstra)
    pub f_score: C,
    /// The actual cost from start (g-score)
    pub g_score: C,
}

impl<C: Ord> PartialEq for PriorityCost<C> {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}

impl<C: Ord> Eq for PriorityCost<C> {}

impl<C: Ord> PartialOrd for PriorityCost<C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<C: Ord> Ord for PriorityCost<C> {
    fn cmp(&self, other: &Self) -> Ordering {
        // For min-heap: lower f_score = higher priority
        self.f_score.cmp(&other.f_score)
    }
}

/// Internal index type for the hash map.
type NodeIndex = usize;

/// Configuration for search limits.
#[derive(Default, Clone, Copy)]
struct SearchLimits<C: Cost> {
    max_cost: Option<C>,
    max_nodes: Option<usize>,
}

// ============================================================================
// Lazy Dijkstra (for simple Heap without decrease_key)
// ============================================================================

/// Metadata stored for each visited node during lazy search.
struct LazyNodeEntry<N: SearchNode> {
    /// The actual node state
    node: N,
    /// Best known cost from start to this node (g-score)
    g_score: N::Cost,
    /// Previous node in the path (for reconstruction)
    came_from: Option<NodeIndex>,
    /// Whether this node has been fully processed
    closed: bool,
}

/// State for lazy Dijkstra (no decrease_key support).
struct LazyPathFinder<N: SearchNode> {
    /// Maps node index to node data
    nodes: FxHashMap<NodeIndex, LazyNodeEntry<N>>,
    /// Maps node state to its index (for fast lookups)
    state_to_index: FxHashMap<N, NodeIndex>,
    /// Next available index
    next_index: NodeIndex,
}

impl<N: SearchNode> LazyPathFinder<N> {
    fn new() -> Self {
        LazyPathFinder {
            nodes: FxHashMap::default(),
            state_to_index: FxHashMap::default(),
            next_index: 0,
        }
    }

    /// Gets or creates an index for a node state.
    fn get_or_create_index(&mut self, node: N, g_score: N::Cost) -> (NodeIndex, bool) {
        if let Some(&index) = self.state_to_index.get(&node) {
            (index, false)
        } else {
            let index = self.next_index;
            self.next_index += 1;
            self.state_to_index.insert(node.clone(), index);
            self.nodes.insert(
                index,
                LazyNodeEntry {
                    node,
                    g_score,
                    came_from: None,
                    closed: false,
                },
            );
            (index, true)
        }
    }

    /// Reconstructs the path from start to the given node index.
    fn reconstruct_path(&self, mut current: NodeIndex) -> Vec<N> {
        let mut path = Vec::new();
        loop {
            let entry = self.nodes.get(&current).unwrap();
            path.push(entry.node.clone());
            if let Some(prev) = entry.came_from {
                current = prev;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }
}

/// Finds the shortest path using lazy Dijkstra (re-insertion instead of decrease_key).
///
/// This variant works with any heap implementing the base [`Heap`] trait.
/// It re-inserts nodes with better costs rather than using `decrease_key`.
/// This may result in duplicate heap entries but is correct and works
/// with simple heaps like `SimpleBinaryHeap`.
///
/// For better performance with heaps that support `decrease_key`, use
/// [`shortest_path`] instead.
pub fn shortest_path_lazy<N, H>(start: &N) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    search_impl_lazy::<N, H>(start, SearchLimits::default())
}

/// Internal lazy search implementation.
fn search_impl_lazy<N, H>(start: &N, limits: SearchLimits<N::Cost>) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    let mut heap = H::new();
    let mut finder = LazyPathFinder::<N>::new();
    let mut nodes_explored = 0usize;

    // Initialize with start node
    let initial_h = start.heuristic();
    let (start_index, _) = finder.get_or_create_index(start.clone(), N::Cost::default());
    let priority = PriorityCost {
        f_score: initial_h,
        g_score: N::Cost::default(),
    };
    heap.push(priority, start_index);

    while let Some((priority, current_index)) = heap.pop() {
        // Check node limit
        if let Some(max) = limits.max_nodes {
            if nodes_explored >= max {
                return None;
            }
        }
        nodes_explored += 1;

        let current_entry = finder.nodes.get_mut(&current_index).unwrap();

        // Skip if already processed (lazy Dijkstra may have duplicates)
        if current_entry.closed {
            continue;
        }

        // Skip if this is a stale entry (we found a better path)
        if priority.g_score > current_entry.g_score {
            continue;
        }

        current_entry.closed = true;

        let current_node = current_entry.node.clone();
        let current_g = priority.g_score;

        // Check cost limit
        if let Some(max) = limits.max_cost {
            if current_g > max {
                continue;
            }
        }

        if current_node.is_goal() {
            let path = finder.reconstruct_path(current_index);
            return Some((path, current_g));
        }

        for (neighbor, edge_cost) in current_node.successors() {
            let tentative_g = current_g + edge_cost;

            // Skip if exceeds max cost
            if let Some(max) = limits.max_cost {
                if tentative_g > max {
                    continue;
                }
            }

            let (neighbor_index, is_new) =
                finder.get_or_create_index(neighbor.clone(), tentative_g);

            let neighbor_entry = finder.nodes.get_mut(&neighbor_index).unwrap();

            if neighbor_entry.closed {
                continue;
            }

            // Only process if we found a better path
            if is_new || tentative_g < neighbor_entry.g_score {
                neighbor_entry.g_score = tentative_g;
                neighbor_entry.came_from = Some(current_index);

                let h = neighbor.heuristic();
                let f = tentative_g + h;
                let new_priority = PriorityCost {
                    f_score: f,
                    g_score: tentative_g,
                };

                // Lazy Dijkstra: just push a new entry instead of decrease_key
                heap.push(new_priority, neighbor_index);
            }
        }
    }

    None
}

// ============================================================================
// Optimized Dijkstra (for DecreaseKeyHeap with decrease_key)
// ============================================================================

/// Metadata stored for each visited node during optimized search.
struct OptNodeEntry<N: SearchNode, H: Handle> {
    /// The actual node state
    node: N,
    /// Cost from start to this node (g-score)
    g_score: N::Cost,
    /// Handle into the heap (if still in open set)
    handle: Option<H>,
    /// Previous node in the path (for reconstruction)
    came_from: Option<NodeIndex>,
    /// Whether this node has been fully processed
    closed: bool,
}

/// State for optimized Dijkstra (with decrease_key support).
struct OptPathFinder<N: SearchNode, H: Handle> {
    /// Maps node index to node data
    nodes: FxHashMap<NodeIndex, OptNodeEntry<N, H>>,
    /// Maps node state to its index (for fast lookups)
    state_to_index: FxHashMap<N, NodeIndex>,
    /// Next available index
    next_index: NodeIndex,
}

impl<N: SearchNode, H: Handle> OptPathFinder<N, H> {
    fn new() -> Self {
        OptPathFinder {
            nodes: FxHashMap::default(),
            state_to_index: FxHashMap::default(),
            next_index: 0,
        }
    }

    /// Gets or creates an index for a node state.
    fn get_or_create_index(&mut self, node: N, g_score: N::Cost) -> (NodeIndex, bool) {
        if let Some(&index) = self.state_to_index.get(&node) {
            (index, false)
        } else {
            let index = self.next_index;
            self.next_index += 1;
            self.state_to_index.insert(node.clone(), index);
            self.nodes.insert(
                index,
                OptNodeEntry {
                    node,
                    g_score,
                    handle: None,
                    came_from: None,
                    closed: false,
                },
            );
            (index, true)
        }
    }

    /// Reconstructs the path from start to the given node index.
    fn reconstruct_path(&self, mut current: NodeIndex) -> Vec<N> {
        let mut path = Vec::new();
        loop {
            let entry = self.nodes.get(&current).unwrap();
            path.push(entry.node.clone());
            if let Some(prev) = entry.came_from {
                current = prev;
            } else {
                break;
            }
        }
        path.reverse();
        path
    }
}

/// Finds the shortest path using optimized Dijkstra with `decrease_key`.
///
/// This variant uses the [`DecreaseKeyHeap`] trait for efficient priority
/// updates. It's more efficient than [`shortest_path_lazy`] for heaps that
/// support `decrease_key` (like `FibonacciHeap`, `PairingHeap`, etc.).
///
/// The node's `heuristic()` method is used to guide the search. If not
/// overridden, it defaults to zero (Dijkstra's algorithm). If overridden
/// with an admissible heuristic, you get A* behavior.
pub fn shortest_path<N, H>(start: &N) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: DecreaseKeyHeap<NodeIndex, PriorityCost<N::Cost>>,
{
    search_impl_opt::<N, H>(start, SearchLimits::default())
}

/// Internal optimized search implementation.
fn search_impl_opt<N, H>(start: &N, limits: SearchLimits<N::Cost>) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: DecreaseKeyHeap<NodeIndex, PriorityCost<N::Cost>>,
{
    let mut heap = H::new();
    let mut finder = OptPathFinder::<N, H::Handle>::new();
    let mut nodes_explored = 0usize;

    // Initialize with start node
    let initial_h = start.heuristic();
    let (start_index, _) = finder.get_or_create_index(start.clone(), N::Cost::default());
    let priority = PriorityCost {
        f_score: initial_h,
        g_score: N::Cost::default(),
    };
    let handle = heap.push_with_handle(priority, start_index);
    finder.nodes.get_mut(&start_index).unwrap().handle = Some(handle);

    while let Some((priority, current_index)) = heap.pop() {
        // Check node limit
        if let Some(max) = limits.max_nodes {
            if nodes_explored >= max {
                return None;
            }
        }
        nodes_explored += 1;

        let current_entry = finder.nodes.get_mut(&current_index).unwrap();

        if current_entry.closed {
            continue;
        }
        current_entry.closed = true;
        current_entry.handle = None;

        let current_node = current_entry.node.clone();
        let current_g = priority.g_score;

        // Check cost limit
        if let Some(max) = limits.max_cost {
            if current_g > max {
                continue;
            }
        }

        if current_node.is_goal() {
            let path = finder.reconstruct_path(current_index);
            return Some((path, current_g));
        }

        for (neighbor, edge_cost) in current_node.successors() {
            let tentative_g = current_g + edge_cost;

            // Skip if exceeds max cost
            if let Some(max) = limits.max_cost {
                if tentative_g > max {
                    continue;
                }
            }

            let h = neighbor.heuristic();
            let f = tentative_g + h;

            let (neighbor_index, is_new) =
                finder.get_or_create_index(neighbor.clone(), tentative_g);

            let neighbor_entry = finder.nodes.get_mut(&neighbor_index).unwrap();

            if neighbor_entry.closed {
                continue;
            }

            if is_new {
                neighbor_entry.g_score = tentative_g;
                neighbor_entry.came_from = Some(current_index);
                let new_priority = PriorityCost {
                    f_score: f,
                    g_score: tentative_g,
                };
                let handle = heap.push_with_handle(new_priority, neighbor_index);
                neighbor_entry.handle = Some(handle);
            } else if tentative_g < neighbor_entry.g_score {
                neighbor_entry.g_score = tentative_g;
                neighbor_entry.came_from = Some(current_index);
                let new_priority = PriorityCost {
                    f_score: f,
                    g_score: tentative_g,
                };

                if let Some(ref handle) = neighbor_entry.handle {
                    let _ = heap.decrease_key(handle, new_priority);
                }
            }
        }
    }

    None
}

// ============================================================================
// Builder API
// ============================================================================

/// Builder for pathfinding queries with more configuration options.
///
/// Provides a fluent API for configuring and running pathfinding searches.
/// The node type's `is_goal()` method determines when to stop.
pub struct PathFinderBuilder<N: SearchNode> {
    start: N,
    max_cost: Option<N::Cost>,
    max_nodes: Option<usize>,
}

impl<N: SearchNode> PathFinderBuilder<N> {
    /// Creates a new builder starting from the given node.
    pub fn new(start: N) -> Self {
        PathFinderBuilder {
            start,
            max_cost: None,
            max_nodes: None,
        }
    }

    /// Sets the maximum cost to explore.
    pub fn max_cost(mut self, cost: N::Cost) -> Self {
        self.max_cost = Some(cost);
        self
    }

    /// Sets the maximum number of nodes to explore.
    pub fn max_nodes(mut self, count: usize) -> Self {
        self.max_nodes = Some(count);
        self
    }

    /// Finds the shortest path with the configured settings.
    ///
    /// Uses the optimized algorithm with `decrease_key` for heaps that support it.
    pub fn shortest_path<H>(self) -> Option<(Vec<N>, N::Cost)>
    where
        H: DecreaseKeyHeap<NodeIndex, PriorityCost<N::Cost>>,
    {
        let limits = SearchLimits {
            max_cost: self.max_cost,
            max_nodes: self.max_nodes,
        };
        search_impl_opt::<N, H>(&self.start, limits)
    }

    /// Finds the shortest path using lazy Dijkstra (re-insertion).
    ///
    /// Works with any heap implementing the base `Heap` trait.
    pub fn shortest_path_lazy<H>(self) -> Option<(Vec<N>, N::Cost)>
    where
        H: Heap<NodeIndex, PriorityCost<N::Cost>>,
    {
        let limits = SearchLimits {
            max_cost: self.max_cost,
            max_nodes: self.max_nodes,
        };
        search_impl_lazy::<N, H>(&self.start, limits)
    }
}

// ============================================================================
// Reachable Within
// ============================================================================

/// Returns all nodes reachable from the start within a given cost budget.
///
/// This is useful for "what's nearby" queries. Uses optimized search with
/// `decrease_key`.
pub fn reachable_within<N, H>(start: &N, max_cost: N::Cost) -> Vec<(N, N::Cost)>
where
    N: SearchNode,
    H: DecreaseKeyHeap<NodeIndex, PriorityCost<N::Cost>>,
{
    let mut heap = H::new();
    let mut finder = OptPathFinder::<N, H::Handle>::new();
    let mut result = Vec::new();

    let (start_index, _) = finder.get_or_create_index(start.clone(), N::Cost::default());
    let priority = PriorityCost {
        f_score: N::Cost::default(),
        g_score: N::Cost::default(),
    };
    let handle = heap.push_with_handle(priority, start_index);
    finder.nodes.get_mut(&start_index).unwrap().handle = Some(handle);

    while let Some((priority, current_index)) = heap.pop() {
        let current_entry = finder.nodes.get_mut(&current_index).unwrap();

        if current_entry.closed {
            continue;
        }
        current_entry.closed = true;
        current_entry.handle = None;

        let current_node = current_entry.node.clone();
        let current_g = priority.g_score;

        if current_g > max_cost {
            continue;
        }

        result.push((current_node.clone(), current_g));

        for (neighbor, edge_cost) in current_node.successors() {
            let tentative_g = current_g + edge_cost;

            if tentative_g > max_cost {
                continue;
            }

            let (neighbor_index, is_new) =
                finder.get_or_create_index(neighbor.clone(), tentative_g);

            let neighbor_entry = finder.nodes.get_mut(&neighbor_index).unwrap();

            if neighbor_entry.closed {
                continue;
            }

            if is_new {
                neighbor_entry.g_score = tentative_g;
                neighbor_entry.came_from = Some(current_index);
                let new_priority = PriorityCost {
                    f_score: tentative_g,
                    g_score: tentative_g,
                };
                let handle = heap.push_with_handle(new_priority, neighbor_index);
                neighbor_entry.handle = Some(handle);
            } else if tentative_g < neighbor_entry.g_score {
                neighbor_entry.g_score = tentative_g;
                neighbor_entry.came_from = Some(current_index);
                let new_priority = PriorityCost {
                    f_score: tentative_g,
                    g_score: tentative_g,
                };

                if let Some(ref handle) = neighbor_entry.handle {
                    let _ = heap.decrease_key(handle, new_priority);
                }
            }
        }
    }

    result
}

/// Returns all nodes reachable from the start within a given cost budget.
///
/// Uses lazy Dijkstra, works with any heap implementing the base `Heap` trait.
pub fn reachable_within_lazy<N, H>(start: &N, max_cost: N::Cost) -> Vec<(N, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    let mut heap = H::new();
    let mut finder = LazyPathFinder::<N>::new();
    let mut result = Vec::new();

    let (start_index, _) = finder.get_or_create_index(start.clone(), N::Cost::default());
    let priority = PriorityCost {
        f_score: N::Cost::default(),
        g_score: N::Cost::default(),
    };
    heap.push(priority, start_index);

    while let Some((priority, current_index)) = heap.pop() {
        let current_entry = finder.nodes.get_mut(&current_index).unwrap();

        if current_entry.closed {
            continue;
        }

        // Skip stale entries
        if priority.g_score > current_entry.g_score {
            continue;
        }

        current_entry.closed = true;

        let current_node = current_entry.node.clone();
        let current_g = priority.g_score;

        if current_g > max_cost {
            continue;
        }

        result.push((current_node.clone(), current_g));

        for (neighbor, edge_cost) in current_node.successors() {
            let tentative_g = current_g + edge_cost;

            if tentative_g > max_cost {
                continue;
            }

            let (neighbor_index, is_new) =
                finder.get_or_create_index(neighbor.clone(), tentative_g);

            let neighbor_entry = finder.nodes.get_mut(&neighbor_index).unwrap();

            if neighbor_entry.closed {
                continue;
            }

            if is_new || tentative_g < neighbor_entry.g_score {
                neighbor_entry.g_score = tentative_g;
                neighbor_entry.came_from = Some(current_index);
                let new_priority = PriorityCost {
                    f_score: tentative_g,
                    g_score: tentative_g,
                };
                heap.push(new_priority, neighbor_index);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibonacci::FibonacciHeap;
    use crate::pairing::PairingHeap;
    use crate::simple_binary::SimpleBinaryHeap;

    // Simple linear graph node that carries its goal
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct LinearNode {
        value: i32,
        goal: i32,
    }

    impl LinearNode {
        fn new(value: i32, goal: i32) -> Self {
            LinearNode { value, goal }
        }
    }

    impl SearchNode for LinearNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            if self.value < 100 {
                vec![(LinearNode::new(self.value + 1, self.goal), 1)]
            } else {
                vec![]
            }
        }

        fn is_goal(&self) -> bool {
            self.value == self.goal
        }
    }

    // Grid-based graph for A* tests - carries goal coordinates
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct GridPos {
        x: i32,
        y: i32,
        goal_x: i32,
        goal_y: i32,
    }

    impl GridPos {
        fn new(x: i32, y: i32, goal_x: i32, goal_y: i32) -> Self {
            GridPos {
                x,
                y,
                goal_x,
                goal_y,
            }
        }
    }

    impl SearchNode for GridPos {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            vec![
                (
                    GridPos::new(self.x + 1, self.y, self.goal_x, self.goal_y),
                    1,
                ),
                (
                    GridPos::new(self.x - 1, self.y, self.goal_x, self.goal_y),
                    1,
                ),
                (
                    GridPos::new(self.x, self.y + 1, self.goal_x, self.goal_y),
                    1,
                ),
                (
                    GridPos::new(self.x, self.y - 1, self.goal_x, self.goal_y),
                    1,
                ),
            ]
        }

        fn is_goal(&self) -> bool {
            self.x == self.goal_x && self.y == self.goal_y
        }

        fn heuristic(&self) -> u32 {
            ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as u32
        }
    }

    // Node for reachable_within tests
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct ReachableNode(i32);

    impl SearchNode for ReachableNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            if self.0 < 100 {
                vec![(ReachableNode(self.0 + 1), 1)]
            } else {
                vec![]
            }
        }

        fn is_goal(&self) -> bool {
            false
        }
    }

    // ==================== Optimized (DecreaseKeyHeap) Tests ====================

    #[test]
    fn test_shortest_path_fibonacci() {
        let start = LinearNode::new(0, 5);
        let result = shortest_path::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_shortest_path_pairing() {
        let start = LinearNode::new(0, 5);
        let result = shortest_path::<_, PairingHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
    }

    // ==================== Lazy (simple Heap) Tests ====================

    #[test]
    fn test_shortest_path_lazy_simple_binary() {
        let start = LinearNode::new(0, 5);
        let result = shortest_path_lazy::<_, SimpleBinaryHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_shortest_path_lazy_fibonacci() {
        // Lazy also works with DecreaseKeyHeap (just doesn't use decrease_key)
        let start = LinearNode::new(0, 5);
        let result = shortest_path_lazy::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
    }

    // ==================== Both variants should produce same results ====================

    #[test]
    fn test_optimized_and_lazy_same_result() {
        let start = GridPos::new(0, 0, 5, 5);

        let opt_result = shortest_path::<_, FibonacciHeap<_, _>>(&start);
        let lazy_result = shortest_path_lazy::<_, SimpleBinaryHeap<_, _>>(&start);

        assert!(opt_result.is_some());
        assert!(lazy_result.is_some());

        let (_, opt_cost) = opt_result.unwrap();
        let (_, lazy_cost) = lazy_result.unwrap();

        assert_eq!(opt_cost, lazy_cost);
        assert_eq!(opt_cost, 10);
    }

    // ==================== Decrease Key correctness test ====================

    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct DecreaseKeyNode {
        id: u32,
        goal: u32,
    }

    impl DecreaseKeyNode {
        fn new(id: u32, goal: u32) -> Self {
            DecreaseKeyNode { id, goal }
        }
    }

    impl SearchNode for DecreaseKeyNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            // Graph designed to test decrease_key:
            //   0 --10-> 1 --1-> 3
            //   |        ^
            //   1        |
            //   v        5
            //   2 -------+
            match self.id {
                0 => vec![
                    (DecreaseKeyNode::new(1, self.goal), 10),
                    (DecreaseKeyNode::new(2, self.goal), 1),
                ],
                1 => vec![(DecreaseKeyNode::new(3, self.goal), 1)],
                2 => vec![(DecreaseKeyNode::new(1, self.goal), 5)],
                _ => vec![],
            }
        }

        fn is_goal(&self) -> bool {
            self.id == self.goal
        }
    }

    #[test]
    fn test_decrease_key_finds_optimal() {
        let start = DecreaseKeyNode::new(0, 3);

        // Optimized version
        let result = shortest_path::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 7); // 0->2->1->3
        assert_eq!(path.len(), 4);

        // Lazy version should also find optimal
        let lazy_result = shortest_path_lazy::<_, SimpleBinaryHeap<_, _>>(&start);
        assert!(lazy_result.is_some());
        let (_, lazy_cost) = lazy_result.unwrap();
        assert_eq!(lazy_cost, 7);
    }

    // ==================== Builder Tests ====================

    #[test]
    fn test_builder_max_cost() {
        let start = LinearNode::new(0, 10);
        let result = PathFinderBuilder::new(start)
            .max_cost(3)
            .shortest_path::<FibonacciHeap<_, _>>();
        assert!(result.is_none());
    }

    #[test]
    fn test_builder_lazy() {
        let start = LinearNode::new(0, 5);
        let result = PathFinderBuilder::new(start)
            .max_cost(10)
            .shortest_path_lazy::<SimpleBinaryHeap<_, _>>();
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 5);
    }

    // ==================== Reachable Within Tests ====================

    #[test]
    fn test_reachable_within() {
        let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&ReachableNode(0), 5);
        assert_eq!(reachable.len(), 6);
    }

    #[test]
    fn test_reachable_within_lazy() {
        let reachable = reachable_within_lazy::<_, SimpleBinaryHeap<_, _>>(&ReachableNode(0), 5);
        assert_eq!(reachable.len(), 6);
    }
}
