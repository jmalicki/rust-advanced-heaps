//! Dijkstra's and A* pathfinding algorithms using advanced heap data structures
//!
//! This module provides generic implementations of Dijkstra's shortest path algorithm
//! and A* search that leverage the efficient `decrease_key` operation of the heaps
//! in this crate.
//!
//! # Design
//!
//! For performance, only lightweight indices are stored in the heap rather than
//! full node data. A fast hash map (using FxHash) maps node states to their
//! metadata including costs and handles.
//!
//! Note: Dijkstra and A* are the same algorithm - A* just adds a heuristic to
//! guide the search. Dijkstra is A* with h(n) = 0.
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::pathfinding::{SearchNode, dijkstra};
//! use rust_advanced_heaps::pairing::PairingHeap;
//!
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct GridPos { x: i32, y: i32 }
//!
//! impl SearchNode for GridPos {
//!     type Cost = u32;
//!
//!     fn successors(&self) -> Vec<(Self, Self::Cost)> {
//!         vec![
//!             (GridPos { x: self.x + 1, y: self.y }, 1),
//!             (GridPos { x: self.x - 1, y: self.y }, 1),
//!             (GridPos { x: self.x, y: self.y + 1 }, 1),
//!             (GridPos { x: self.x, y: self.y - 1 }, 1),
//!         ]
//!     }
//! }
//!
//! let start = GridPos { x: 0, y: 0 };
//! let goal = GridPos { x: 2, y: 2 };
//!
//! let result = dijkstra::<_, PairingHeap<_, _>>(&start, |pos| *pos == goal);
//! assert!(result.is_some());
//! let (path, cost) = result.unwrap();
//! assert_eq!(cost, 4); // Manhattan distance
//! ```

use crate::traits::{Handle, Heap};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Add;

/// Trait for types that can be used as costs in pathfinding algorithms.
///
/// This requires the type to be orderable, copyable, and support addition.
/// It also requires a zero value for initialization.
pub trait Cost: Ord + Copy + Add<Output = Self> + Default {}

impl<T> Cost for T where T: Ord + Copy + Add<Output = Self> + Default {}

/// Trait for nodes in a search graph.
///
/// Implement this trait for your node type to use Dijkstra's or A* algorithms.
/// The node type must be hashable and cloneable for efficient storage.
pub trait SearchNode: Clone + Eq + Hash {
    /// The cost type for edge weights (e.g., u32, u64, f64 wrapped in OrderedFloat)
    type Cost: Cost;

    /// Returns all successor nodes along with the cost to reach them.
    ///
    /// This is where you define your graph structure. Each call should return
    /// all neighbors reachable from this node along with their edge costs.
    fn successors(&self) -> Vec<(Self, Self::Cost)>;
}

/// Trait for nodes that can provide a heuristic estimate for A* search.
///
/// The heuristic must be admissible (never overestimate the true cost)
/// for A* to find optimal paths.
pub trait AStarNode: SearchNode {
    /// Returns a heuristic estimate of the cost from this node to the goal.
    ///
    /// For A* to find optimal paths, this must never overestimate the actual cost.
    /// Common heuristics include:
    /// - Manhattan distance for grid-based movement
    /// - Euclidean distance for arbitrary movement
    /// - Zero (reduces to Dijkstra's algorithm)
    fn heuristic(&self, goal: &Self) -> Self::Cost;
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
/// We store these lightweight indices in the heap instead of full node data.
type NodeIndex = usize;

/// Metadata stored for each visited node during search.
struct NodeEntry<N: SearchNode, H: Handle> {
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

/// A pathfinder that uses a heap with efficient decrease_key for graph search.
///
/// This struct manages the open set (heap) and closed set (hash map) for
/// Dijkstra's algorithm and A* search.
pub struct PathFinder<N, H, C>
where
    N: SearchNode,
    H: Handle,
    C: Cost,
{
    /// Maps node index to node data
    nodes: FxHashMap<NodeIndex, NodeEntry<N, H>>,
    /// Maps node state to its index (for fast lookups)
    state_to_index: FxHashMap<N, NodeIndex>,
    /// Next available index
    next_index: NodeIndex,
    /// Phantom data for the cost type
    _cost: std::marker::PhantomData<C>,
}

impl<N, H, C> PathFinder<N, H, C>
where
    N: SearchNode<Cost = C>,
    H: Handle,
    C: Cost,
{
    /// Creates a new empty pathfinder.
    pub fn new() -> Self {
        PathFinder {
            nodes: FxHashMap::default(),
            state_to_index: FxHashMap::default(),
            next_index: 0,
            _cost: std::marker::PhantomData,
        }
    }

    /// Gets or creates an index for a node state.
    fn get_or_create_index(&mut self, node: N, g_score: C) -> (NodeIndex, bool) {
        if let Some(&index) = self.state_to_index.get(&node) {
            (index, false)
        } else {
            let index = self.next_index;
            self.next_index += 1;
            self.state_to_index.insert(node.clone(), index);
            self.nodes.insert(
                index,
                NodeEntry {
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

impl<N, H, C> Default for PathFinder<N, H, C>
where
    N: SearchNode<Cost = C>,
    H: Handle,
    C: Cost,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a successful pathfinding search.
#[derive(Debug, Clone)]
pub struct PathResult<N: SearchNode> {
    /// The path from start to goal (inclusive)
    pub path: Vec<N>,
    /// Total cost of the path
    pub cost: N::Cost,
}

/// Core search algorithm used by both Dijkstra and A*.
///
/// This is the unified implementation - Dijkstra is just this with h(n) = 0.
///
/// # Arguments
/// - `start`: The starting node
/// - `is_goal`: A function that returns true when a goal node is reached
/// - `heuristic`: A function returning a heuristic estimate (use |_| 0 for Dijkstra)
///
/// # Returns
/// - `Some((path, cost))` if a path is found
/// - `None` if no path exists
pub fn search<N, H, G, F>(
    start: &N,
    is_goal: G,
    heuristic: F,
) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
    G: Fn(&N) -> bool,
    F: Fn(&N) -> N::Cost,
{
    let mut heap = H::new();
    let mut finder: PathFinder<N, H::Handle, N::Cost> = PathFinder::new();

    // Initialize with start node
    let initial_h = heuristic(start);
    let (start_index, _) = finder.get_or_create_index(start.clone(), N::Cost::default());
    let priority = PriorityCost {
        f_score: initial_h,
        g_score: N::Cost::default(),
    };
    let handle = heap.push(priority, start_index);
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

        if is_goal(&current_node) {
            let path = finder.reconstruct_path(current_index);
            return Some((path, current_g));
        }

        for (neighbor, edge_cost) in current_node.successors() {
            let tentative_g = current_g + edge_cost;
            let h = heuristic(&neighbor);
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
                let handle = heap.push(new_priority, neighbor_index);
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

/// Runs Dijkstra's algorithm to find the shortest path from start to any goal.
///
/// This is equivalent to A* with a zero heuristic.
///
/// # Type Parameters
/// - `N`: The node type implementing [`SearchNode`]
/// - `H`: The heap type implementing [`Heap`]
///
/// # Arguments
/// - `start`: The starting node
/// - `is_goal`: A function that returns true when a goal node is reached
///
/// # Returns
/// - `Some((path, cost))` if a path is found
/// - `None` if no path exists
///
/// # Example
/// ```rust
/// use rust_advanced_heaps::pathfinding::{SearchNode, dijkstra};
/// use rust_advanced_heaps::fibonacci::FibonacciHeap;
///
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// struct Node(i32);
///
/// impl SearchNode for Node {
///     type Cost = u32;
///     fn successors(&self) -> Vec<(Self, u32)> {
///         if self.0 < 10 {
///             vec![(Node(self.0 + 1), 1)]
///         } else {
///             vec![]
///         }
///     }
/// }
///
/// let result = dijkstra::<_, FibonacciHeap<_, _>>(&Node(0), |n| n.0 == 5);
/// assert!(result.is_some());
/// let (path, cost) = result.unwrap();
/// assert_eq!(cost, 5);
/// assert_eq!(path.len(), 6); // 0, 1, 2, 3, 4, 5
/// ```
#[inline]
pub fn dijkstra<N, H>(start: &N, is_goal: impl Fn(&N) -> bool) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    search::<N, H, _, _>(start, is_goal, |_| N::Cost::default())
}

/// Runs A* search to find the shortest path from start to goal.
///
/// A* uses a heuristic to guide the search, which can significantly reduce
/// the number of nodes explored compared to Dijkstra's algorithm.
///
/// # Type Parameters
/// - `N`: The node type implementing [`AStarNode`]
/// - `H`: The heap type implementing [`Heap`]
///
/// # Arguments
/// - `start`: The starting node
/// - `goal`: The goal node
///
/// # Returns
/// - `Some((path, cost))` if a path is found
/// - `None` if no path exists
///
/// # Example
/// ```rust
/// use rust_advanced_heaps::pathfinding::{SearchNode, AStarNode, astar};
/// use rust_advanced_heaps::pairing::PairingHeap;
///
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// struct GridPos { x: i32, y: i32 }
///
/// impl SearchNode for GridPos {
///     type Cost = u32;
///     fn successors(&self) -> Vec<(Self, u32)> {
///         vec![
///             (GridPos { x: self.x + 1, y: self.y }, 1),
///             (GridPos { x: self.x - 1, y: self.y }, 1),
///             (GridPos { x: self.x, y: self.y + 1 }, 1),
///             (GridPos { x: self.x, y: self.y - 1 }, 1),
///         ]
///     }
/// }
///
/// impl AStarNode for GridPos {
///     fn heuristic(&self, goal: &Self) -> u32 {
///         ((self.x - goal.x).abs() + (self.y - goal.y).abs()) as u32
///     }
/// }
///
/// let start = GridPos { x: 0, y: 0 };
/// let goal = GridPos { x: 3, y: 3 };
/// let result = astar::<_, PairingHeap<_, _>>(&start, &goal);
/// assert!(result.is_some());
/// let (path, cost) = result.unwrap();
/// assert_eq!(cost, 6);
/// ```
#[inline]
pub fn astar<N, H>(start: &N, goal: &N) -> Option<(Vec<N>, N::Cost)>
where
    N: AStarNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    search::<N, H, _, _>(start, |n| n == goal, |n| n.heuristic(goal))
}

/// Runs A* search with a custom goal predicate and heuristic.
///
/// This variant allows finding a path to any node matching a predicate,
/// not just a specific goal node. The heuristic is provided as a closure.
///
/// # Type Parameters
/// - `N`: The node type implementing [`SearchNode`]
/// - `H`: The heap type implementing [`Heap`]
///
/// # Arguments
/// - `start`: The starting node
/// - `is_goal`: A function that returns true when a goal node is reached
/// - `heuristic`: A function returning a heuristic estimate for a node
///
/// # Returns
/// - `Some((path, cost))` if a path is found
/// - `None` if no path exists
#[inline]
pub fn astar_with<N, H, G, F>(
    start: &N,
    is_goal: G,
    heuristic: F,
) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
    G: Fn(&N) -> bool,
    F: Fn(&N) -> N::Cost,
{
    search::<N, H, _, _>(start, is_goal, heuristic)
}

/// Builder for pathfinding queries with more configuration options.
///
/// Provides a fluent API for configuring and running pathfinding searches.
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

    /// Runs search with the configured settings.
    pub fn search<H, G, F>(self, is_goal: G, heuristic: F) -> Option<(Vec<N>, N::Cost)>
    where
        H: Heap<NodeIndex, PriorityCost<N::Cost>>,
        G: Fn(&N) -> bool,
        F: Fn(&N) -> N::Cost,
    {
        let max_cost = self.max_cost;
        let max_nodes = self.max_nodes;
        let mut nodes_explored = 0usize;

        let mut heap = H::new();
        let mut finder: PathFinder<N, H::Handle, N::Cost> = PathFinder::new();

        let initial_h = heuristic(&self.start);
        let (start_index, _) = finder.get_or_create_index(self.start.clone(), N::Cost::default());
        let priority = PriorityCost {
            f_score: initial_h,
            g_score: N::Cost::default(),
        };
        let handle = heap.push(priority, start_index);
        finder.nodes.get_mut(&start_index).unwrap().handle = Some(handle);

        while let Some((priority, current_index)) = heap.pop() {
            // Check node limit
            if let Some(max) = max_nodes {
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
            if let Some(max) = max_cost {
                if current_g > max {
                    continue;
                }
            }

            if is_goal(&current_node) {
                let path = finder.reconstruct_path(current_index);
                return Some((path, current_g));
            }

            for (neighbor, edge_cost) in current_node.successors() {
                let tentative_g = current_g + edge_cost;

                // Skip if exceeds max cost
                if let Some(max) = max_cost {
                    if tentative_g > max {
                        continue;
                    }
                }

                let h = heuristic(&neighbor);
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
                    let handle = heap.push(new_priority, neighbor_index);
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

    /// Runs Dijkstra's algorithm with the configured settings.
    #[inline]
    pub fn dijkstra<H>(self, is_goal: impl Fn(&N) -> bool) -> Option<(Vec<N>, N::Cost)>
    where
        H: Heap<NodeIndex, PriorityCost<N::Cost>>,
    {
        self.search::<H, _, _>(is_goal, |_| N::Cost::default())
    }
}

/// Returns all nodes reachable from the start within a given cost budget.
///
/// This is useful for "what's nearby" queries.
pub fn reachable_within<N, H>(start: &N, max_cost: N::Cost) -> Vec<(N, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    let mut heap = H::new();
    let mut finder: PathFinder<N, H::Handle, N::Cost> = PathFinder::new();
    let mut result = Vec::new();

    let (start_index, _) = finder.get_or_create_index(start.clone(), N::Cost::default());
    let priority = PriorityCost {
        f_score: N::Cost::default(),
        g_score: N::Cost::default(),
    };
    let handle = heap.push(priority, start_index);
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
                let handle = heap.push(new_priority, neighbor_index);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fibonacci::FibonacciHeap;
    use crate::pairing::PairingHeap;

    // Simple linear graph for basic tests
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct LinearNode(i32);

    impl SearchNode for LinearNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            if self.0 < 100 {
                vec![(LinearNode(self.0 + 1), 1)]
            } else {
                vec![]
            }
        }
    }

    // Grid-based graph for A* tests
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct GridPos {
        x: i32,
        y: i32,
    }

    impl GridPos {
        fn new(x: i32, y: i32) -> Self {
            GridPos { x, y }
        }
    }

    impl SearchNode for GridPos {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            vec![
                (GridPos::new(self.x + 1, self.y), 1),
                (GridPos::new(self.x - 1, self.y), 1),
                (GridPos::new(self.x, self.y + 1), 1),
                (GridPos::new(self.x, self.y - 1), 1),
            ]
        }
    }

    impl AStarNode for GridPos {
        fn heuristic(&self, goal: &Self) -> u32 {
            ((self.x - goal.x).abs() + (self.y - goal.y).abs()) as u32
        }
    }

    // Weighted graph for testing decrease_key
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct WeightedNode(char);

    impl SearchNode for WeightedNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            // Graph:
            //     A --1-- B --1-- D
            //     |       |
            //     5       1
            //     |       |
            //     C --1-- E
            //
            // Path A->D via B costs 2
            // Path A->E via C costs 6
            // Path A->E via B costs 2
            match self.0 {
                'A' => vec![(WeightedNode('B'), 1), (WeightedNode('C'), 5)],
                'B' => vec![(WeightedNode('D'), 1), (WeightedNode('E'), 1)],
                'C' => vec![(WeightedNode('E'), 1)],
                'D' => vec![],
                'E' => vec![],
                _ => vec![],
            }
        }
    }

    // ==================== Dijkstra Tests ====================

    #[test]
    fn test_dijkstra_simple_path_fibonacci() {
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&LinearNode(0), |n| n.0 == 5);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
        assert_eq!(path[0], LinearNode(0));
        assert_eq!(path[5], LinearNode(5));
    }

    #[test]
    fn test_dijkstra_simple_path_pairing() {
        let result = dijkstra::<_, PairingHeap<_, _>>(&LinearNode(0), |n| n.0 == 5);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_dijkstra_no_path() {
        // LinearNode stops at 100, so 200 is unreachable
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&LinearNode(0), |n| n.0 == 200);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_start_is_goal() {
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&LinearNode(5), |n| n.0 == 5);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], LinearNode(5));
    }

    #[test]
    fn test_dijkstra_weighted_graph() {
        // Test that Dijkstra finds the shortest path in a weighted graph
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&WeightedNode('A'), |n| n.0 == 'E');
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        // Should go A -> B -> E (cost 2) not A -> C -> E (cost 6)
        assert_eq!(cost, 2);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], WeightedNode('A'));
        assert_eq!(path[1], WeightedNode('B'));
        assert_eq!(path[2], WeightedNode('E'));
    }

    #[test]
    fn test_dijkstra_grid() {
        let start = GridPos::new(0, 0);
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start, |n| n.x == 3 && n.y == 3);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        // Manhattan distance from (0,0) to (3,3) is 6
        assert_eq!(cost, 6);
        assert_eq!(path[0], start);
    }

    // ==================== A* Tests ====================

    #[test]
    fn test_astar_grid_fibonacci() {
        let start = GridPos::new(0, 0);
        let goal = GridPos::new(5, 5);
        let result = astar::<_, FibonacciHeap<_, _>>(&start, &goal);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 10);
        assert_eq!(path[0], start);
        assert_eq!(*path.last().unwrap(), goal);
    }

    #[test]
    fn test_astar_grid_pairing() {
        let start = GridPos::new(0, 0);
        let goal = GridPos::new(5, 5);
        let result = astar::<_, PairingHeap<_, _>>(&start, &goal);
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 10);
    }

    #[test]
    fn test_astar_same_start_goal() {
        let pos = GridPos::new(3, 3);
        let result = astar::<_, FibonacciHeap<_, _>>(&pos, &pos);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], pos);
    }

    #[test]
    fn test_astar_with_custom_heuristic() {
        let start = GridPos::new(0, 0);
        let goal = GridPos::new(3, 3);

        // Use zero heuristic (reduces to Dijkstra)
        let result = astar_with::<_, FibonacciHeap<_, _>, _, _>(
            &start,
            |n| *n == goal,
            |_| 0, // Zero heuristic
        );
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 6);
    }

    // ==================== Builder Tests ====================

    #[test]
    fn test_builder_max_cost() {
        let result = PathFinderBuilder::new(LinearNode(0))
            .max_cost(3)
            .dijkstra::<FibonacciHeap<_, _>>(|n| n.0 == 10);
        // Should fail because goal is at cost 10, but max is 3
        assert!(result.is_none());
    }

    #[test]
    fn test_builder_max_nodes() {
        let result = PathFinderBuilder::new(LinearNode(0))
            .max_nodes(5)
            .dijkstra::<FibonacciHeap<_, _>>(|n| n.0 == 10);
        // Should fail because we can only explore 5 nodes
        assert!(result.is_none());
    }

    #[test]
    fn test_builder_success_within_limits() {
        let result = PathFinderBuilder::new(LinearNode(0))
            .max_cost(10)
            .max_nodes(20)
            .dijkstra::<FibonacciHeap<_, _>>(|n| n.0 == 5);
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 5);
    }

    // ==================== Reachable Within Tests ====================

    #[test]
    fn test_reachable_within() {
        let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&LinearNode(0), 5);
        assert_eq!(reachable.len(), 6); // Nodes 0-5
        for (node, cost) in &reachable {
            assert!(node.0 >= 0 && node.0 <= 5);
            assert_eq!(*cost, node.0 as u32);
        }
    }

    #[test]
    fn test_reachable_within_zero() {
        let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&LinearNode(0), 0);
        assert_eq!(reachable.len(), 1);
        assert_eq!(reachable[0].0, LinearNode(0));
        assert_eq!(reachable[0].1, 0);
    }

    // ==================== Decrease Key Tests ====================

    // Graph where decrease_key is necessary for optimal path
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct DecreaseKeyNode(u32);

    impl SearchNode for DecreaseKeyNode {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            // Graph designed to test decrease_key:
            //
            //   0 --10-> 1 --1-> 3
            //   |        ^
            //   1        |
            //   v        5
            //   2 -------+
            //
            // Without decrease_key: might find 0->1->3 (cost 11)
            // With decrease_key: finds 0->2->1->3 (cost 7)
            match self.0 {
                0 => vec![(DecreaseKeyNode(1), 10), (DecreaseKeyNode(2), 1)],
                1 => vec![(DecreaseKeyNode(3), 1)],
                2 => vec![(DecreaseKeyNode(1), 5)],
                _ => vec![],
            }
        }
    }

    #[test]
    fn test_decrease_key_finds_optimal() {
        let result =
            dijkstra::<_, FibonacciHeap<_, _>>(&DecreaseKeyNode(0), |n| n.0 == 3);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        // Optimal path is 0 -> 2 -> 1 -> 3 with cost 7
        assert_eq!(cost, 7);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], DecreaseKeyNode(0));
        assert_eq!(path[1], DecreaseKeyNode(2));
        assert_eq!(path[2], DecreaseKeyNode(1));
        assert_eq!(path[3], DecreaseKeyNode(3));
    }

    #[test]
    fn test_decrease_key_with_pairing_heap() {
        let result =
            dijkstra::<_, PairingHeap<_, _>>(&DecreaseKeyNode(0), |n| n.0 == 3);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 7);
        assert_eq!(path.len(), 4);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_disconnected_graph() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct DisconnectedNode(u32);

        impl SearchNode for DisconnectedNode {
            type Cost = u32;
            fn successors(&self) -> Vec<(Self, u32)> {
                match self.0 {
                    0 => vec![(DisconnectedNode(1), 1)],
                    1 => vec![(DisconnectedNode(0), 1)],
                    // 2 is disconnected
                    2 => vec![(DisconnectedNode(3), 1)],
                    3 => vec![(DisconnectedNode(2), 1)],
                    _ => vec![],
                }
            }
        }

        let result =
            dijkstra::<_, FibonacciHeap<_, _>>(&DisconnectedNode(0), |n| n.0 == 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_cycle_in_graph() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct CyclicNode(u32);

        impl SearchNode for CyclicNode {
            type Cost = u32;
            fn successors(&self) -> Vec<(Self, u32)> {
                match self.0 {
                    0 => vec![(CyclicNode(1), 1)],
                    1 => vec![(CyclicNode(2), 1)],
                    2 => vec![(CyclicNode(0), 1), (CyclicNode(3), 1)],
                    3 => vec![],
                    _ => vec![],
                }
            }
        }

        let result = dijkstra::<_, FibonacciHeap<_, _>>(&CyclicNode(0), |n| n.0 == 3);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 3);
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_multiple_paths_same_cost() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct MultiPathNode(u32);

        impl SearchNode for MultiPathNode {
            type Cost = u32;
            fn successors(&self) -> Vec<(Self, u32)> {
                match self.0 {
                    0 => vec![(MultiPathNode(1), 1), (MultiPathNode(2), 1)],
                    1 => vec![(MultiPathNode(3), 1)],
                    2 => vec![(MultiPathNode(3), 1)],
                    _ => vec![],
                }
            }
        }

        let result =
            dijkstra::<_, FibonacciHeap<_, _>>(&MultiPathNode(0), |n| n.0 == 3);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 2);
        assert_eq!(path.len(), 3);
    }

    // ==================== Large Graph Tests ====================

    #[test]
    fn test_large_linear_graph() {
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&LinearNode(0), |n| n.0 == 99);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 99);
        assert_eq!(path.len(), 100);
    }

    #[test]
    fn test_large_grid() {
        let start = GridPos::new(0, 0);
        let goal = GridPos::new(20, 20);
        let result = astar::<_, FibonacciHeap<_, _>>(&start, &goal);
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 40);
    }

    // ==================== Priority Cost Tests ====================

    #[test]
    fn test_priority_cost_ordering() {
        let a = PriorityCost {
            f_score: 5u32,
            g_score: 3u32,
        };
        let b = PriorityCost {
            f_score: 10u32,
            g_score: 8u32,
        };
        assert!(a < b); // Lower f_score = higher priority in min-heap
    }
}
