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
//! The node type carries its own goal context and implements `is_goal()` to
//! determine when the search should terminate.
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::pathfinding::{SearchNode, dijkstra};
//! use rust_advanced_heaps::pairing::PairingHeap;
//!
//! // Node carries its goal coordinates
//! #[derive(Clone, PartialEq, Eq, Hash)]
//! struct GridPos { x: i32, y: i32, goal_x: i32, goal_y: i32 }
//!
//! impl SearchNode for GridPos {
//!     type Cost = u32;
//!
//!     fn successors(&self) -> Vec<(Self, Self::Cost)> {
//!         vec![
//!             (GridPos { x: self.x + 1, y: self.y, goal_x: self.goal_x, goal_y: self.goal_y }, 1),
//!             (GridPos { x: self.x - 1, y: self.y, goal_x: self.goal_x, goal_y: self.goal_y }, 1),
//!             (GridPos { x: self.x, y: self.y + 1, goal_x: self.goal_x, goal_y: self.goal_y }, 1),
//!             (GridPos { x: self.x, y: self.y - 1, goal_x: self.goal_x, goal_y: self.goal_y }, 1),
//!         ]
//!     }
//!
//!     fn is_goal(&self) -> bool {
//!         self.x == self.goal_x && self.y == self.goal_y
//!     }
//! }
//!
//! let start = GridPos { x: 0, y: 0, goal_x: 2, goal_y: 2 };
//!
//! let result = dijkstra::<_, PairingHeap<_, _>>(&start);
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
    fn successors(&self) -> Vec<(Self, Self::Cost)>;

    /// Returns true if this node is a goal state.
    ///
    /// The node should carry enough context to determine this (e.g., a reference
    /// to the goal position, or the problem instance).
    fn is_goal(&self) -> bool;
}

/// Trait for nodes that can provide a heuristic estimate for A* search.
///
/// The heuristic must be admissible (never overestimate the true cost)
/// for A* to find optimal paths.
pub trait AStarNode: SearchNode {
    /// Returns a heuristic estimate of the cost from this node to any goal.
    ///
    /// For A* to find optimal paths, this must never overestimate the actual cost.
    /// Common heuristics include:
    /// - Manhattan distance for grid-based movement
    /// - Euclidean distance for arbitrary movement
    /// - Zero (reduces to Dijkstra's algorithm)
    ///
    /// The node should carry enough context to compute this (e.g., a reference
    /// to the goal position).
    fn heuristic(&self) -> Self::Cost;
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

/// Runs Dijkstra's algorithm from the start node until `is_goal()` returns true.
///
/// # Type Parameters
/// - `N`: The node type implementing [`SearchNode`]
/// - `H`: The heap type implementing [`Heap`]
///
/// # Arguments
/// - `start`: The starting node (must implement `SearchNode` with `is_goal()`)
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
/// // Node that carries its own goal
/// #[derive(Clone, PartialEq, Eq, Hash, Debug)]
/// struct Node {
///     value: i32,
///     goal: i32,
/// }
///
/// impl SearchNode for Node {
///     type Cost = u32;
///
///     fn successors(&self) -> Vec<(Self, u32)> {
///         if self.value < 100 {
///             vec![(Node { value: self.value + 1, goal: self.goal }, 1)]
///         } else {
///             vec![]
///         }
///     }
///
///     fn is_goal(&self) -> bool {
///         self.value == self.goal
///     }
/// }
///
/// let start = Node { value: 0, goal: 5 };
/// let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
/// assert!(result.is_some());
/// let (path, cost) = result.unwrap();
/// assert_eq!(cost, 5);
/// ```
pub fn dijkstra<N, H>(start: &N) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    search_impl::<N, H>(start, |_| N::Cost::default())
}

/// Runs A* search from the start node until `is_goal()` returns true.
///
/// Uses the node's `heuristic()` method to guide the search.
///
/// # Type Parameters
/// - `N`: The node type implementing [`AStarNode`]
/// - `H`: The heap type implementing [`Heap`]
///
/// # Arguments
/// - `start`: The starting node
///
/// # Returns
/// - `Some((path, cost))` if a path is found
/// - `None` if no path exists
pub fn astar<N, H>(start: &N) -> Option<(Vec<N>, N::Cost)>
where
    N: AStarNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
{
    search_impl::<N, H>(start, |n| n.heuristic())
}

/// Internal search implementation.
fn search_impl<N, H>(
    start: &N,
    heuristic: impl Fn(&N) -> N::Cost,
) -> Option<(Vec<N>, N::Cost)>
where
    N: SearchNode,
    H: Heap<NodeIndex, PriorityCost<N::Cost>>,
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

        if current_node.is_goal() {
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

    /// Runs Dijkstra's algorithm with the configured settings.
    ///
    /// Uses the node's `is_goal()` method to determine when to stop.
    pub fn dijkstra<H>(self) -> Option<(Vec<N>, N::Cost)>
    where
        H: Heap<NodeIndex, PriorityCost<N::Cost>>,
    {
        self.search_with_heuristic::<H>(|_| N::Cost::default())
    }

    /// Runs A* search with the configured settings.
    ///
    /// Uses the node's `is_goal()` and `heuristic()` methods.
    pub fn astar<H>(self) -> Option<(Vec<N>, N::Cost)>
    where
        N: AStarNode,
        H: Heap<NodeIndex, PriorityCost<N::Cost>>,
    {
        self.search_with_heuristic::<H>(|n| n.heuristic())
    }

    /// Internal search implementation with configurable heuristic.
    fn search_with_heuristic<H>(
        self,
        heuristic: impl Fn(&N) -> N::Cost,
    ) -> Option<(Vec<N>, N::Cost)>
    where
        H: Heap<NodeIndex, PriorityCost<N::Cost>>,
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

            if current_node.is_goal() {
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
            GridPos { x, y, goal_x, goal_y }
        }
    }

    impl SearchNode for GridPos {
        type Cost = u32;

        fn successors(&self) -> Vec<(Self, u32)> {
            vec![
                (GridPos::new(self.x + 1, self.y, self.goal_x, self.goal_y), 1),
                (GridPos::new(self.x - 1, self.y, self.goal_x, self.goal_y), 1),
                (GridPos::new(self.x, self.y + 1, self.goal_x, self.goal_y), 1),
                (GridPos::new(self.x, self.y - 1, self.goal_x, self.goal_y), 1),
            ]
        }

        fn is_goal(&self) -> bool {
            self.x == self.goal_x && self.y == self.goal_y
        }
    }

    impl AStarNode for GridPos {
        fn heuristic(&self) -> u32 {
            ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as u32
        }
    }

    // Weighted graph node that carries its goal
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct WeightedNode {
        id: char,
        goal: char,
    }

    impl WeightedNode {
        fn new(id: char, goal: char) -> Self {
            WeightedNode { id, goal }
        }
    }

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
            match self.id {
                'A' => vec![
                    (WeightedNode::new('B', self.goal), 1),
                    (WeightedNode::new('C', self.goal), 5),
                ],
                'B' => vec![
                    (WeightedNode::new('D', self.goal), 1),
                    (WeightedNode::new('E', self.goal), 1),
                ],
                'C' => vec![(WeightedNode::new('E', self.goal), 1)],
                'D' => vec![],
                'E' => vec![],
                _ => vec![],
            }
        }

        fn is_goal(&self) -> bool {
            self.id == self.goal
        }
    }

    // Node for reachable_within tests - no goal needed, always returns false
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
            false // reachable_within doesn't use is_goal
        }
    }

    // ==================== Dijkstra Tests ====================

    #[test]
    fn test_dijkstra_simple_path_fibonacci() {
        let start = LinearNode::new(0, 5);
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
        assert_eq!(path[0].value, 0);
        assert_eq!(path[5].value, 5);
    }

    #[test]
    fn test_dijkstra_simple_path_pairing() {
        let start = LinearNode::new(0, 5);
        let result = dijkstra::<_, PairingHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 5);
        assert_eq!(path.len(), 6);
    }

    #[test]
    fn test_dijkstra_no_path() {
        // LinearNode stops at 100, so 200 is unreachable
        let start = LinearNode::new(0, 200);
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_start_is_goal() {
        let start = LinearNode::new(5, 5);
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].value, 5);
    }

    #[test]
    fn test_dijkstra_weighted_graph() {
        // Test that Dijkstra finds the shortest path in a weighted graph
        let start = WeightedNode::new('A', 'E');
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        // Should go A -> B -> E (cost 2) not A -> C -> E (cost 6)
        assert_eq!(cost, 2);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].id, 'A');
        assert_eq!(path[1].id, 'B');
        assert_eq!(path[2].id, 'E');
    }

    #[test]
    fn test_dijkstra_grid() {
        let start = GridPos::new(0, 0, 3, 3);
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        // Manhattan distance from (0,0) to (3,3) is 6
        assert_eq!(cost, 6);
        assert_eq!(path[0].x, 0);
        assert_eq!(path[0].y, 0);
    }

    // ==================== A* Tests ====================

    #[test]
    fn test_astar_grid_fibonacci() {
        let start = GridPos::new(0, 0, 5, 5);
        let result = astar::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 10);
        assert_eq!(path[0].x, 0);
        assert_eq!(path[0].y, 0);
        let last = path.last().unwrap();
        assert_eq!(last.x, 5);
        assert_eq!(last.y, 5);
    }

    #[test]
    fn test_astar_grid_pairing() {
        let start = GridPos::new(0, 0, 5, 5);
        let result = astar::<_, PairingHeap<_, _>>(&start);
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 10);
    }

    #[test]
    fn test_astar_same_start_goal() {
        let start = GridPos::new(3, 3, 3, 3);
        let result = astar::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].x, 3);
        assert_eq!(path[0].y, 3);
    }

    // ==================== Builder Tests ====================

    #[test]
    fn test_builder_max_cost() {
        let start = LinearNode::new(0, 10);
        let result = PathFinderBuilder::new(start)
            .max_cost(3)
            .dijkstra::<FibonacciHeap<_, _>>();
        // Should fail because goal is at cost 10, but max is 3
        assert!(result.is_none());
    }

    #[test]
    fn test_builder_max_nodes() {
        let start = LinearNode::new(0, 10);
        let result = PathFinderBuilder::new(start)
            .max_nodes(5)
            .dijkstra::<FibonacciHeap<_, _>>();
        // Should fail because we can only explore 5 nodes
        assert!(result.is_none());
    }

    #[test]
    fn test_builder_success_within_limits() {
        let start = LinearNode::new(0, 5);
        let result = PathFinderBuilder::new(start)
            .max_cost(10)
            .max_nodes(20)
            .dijkstra::<FibonacciHeap<_, _>>();
        assert!(result.is_some());
        let (_, cost) = result.unwrap();
        assert_eq!(cost, 5);
    }

    // ==================== Reachable Within Tests ====================

    #[test]
    fn test_reachable_within() {
        let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&ReachableNode(0), 5);
        assert_eq!(reachable.len(), 6); // Nodes 0-5
        for (node, cost) in &reachable {
            assert!(node.0 >= 0 && node.0 <= 5);
            assert_eq!(*cost, node.0 as u32);
        }
    }

    #[test]
    fn test_reachable_within_zero() {
        let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&ReachableNode(0), 0);
        assert_eq!(reachable.len(), 1);
        assert_eq!(reachable[0].0, ReachableNode(0));
        assert_eq!(reachable[0].1, 0);
    }

    // ==================== Decrease Key Tests ====================

    // Graph where decrease_key is necessary for optimal path
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
            //
            //   0 --10-> 1 --1-> 3
            //   |        ^
            //   1        |
            //   v        5
            //   2 -------+
            //
            // Without decrease_key: might find 0->1->3 (cost 11)
            // With decrease_key: finds 0->2->1->3 (cost 7)
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
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        // Optimal path is 0 -> 2 -> 1 -> 3 with cost 7
        assert_eq!(cost, 7);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0].id, 0);
        assert_eq!(path[1].id, 2);
        assert_eq!(path[2].id, 1);
        assert_eq!(path[3].id, 3);
    }

    #[test]
    fn test_decrease_key_with_pairing_heap() {
        let start = DecreaseKeyNode::new(0, 3);
        let result = dijkstra::<_, PairingHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 7);
        assert_eq!(path.len(), 4);
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_disconnected_graph() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct DisconnectedNode {
            id: u32,
            goal: u32,
        }

        impl SearchNode for DisconnectedNode {
            type Cost = u32;
            fn successors(&self) -> Vec<(Self, u32)> {
                match self.id {
                    0 => vec![(DisconnectedNode { id: 1, goal: self.goal }, 1)],
                    1 => vec![(DisconnectedNode { id: 0, goal: self.goal }, 1)],
                    // 2 is disconnected
                    2 => vec![(DisconnectedNode { id: 3, goal: self.goal }, 1)],
                    3 => vec![(DisconnectedNode { id: 2, goal: self.goal }, 1)],
                    _ => vec![],
                }
            }
            fn is_goal(&self) -> bool {
                self.id == self.goal
            }
        }

        let start = DisconnectedNode { id: 0, goal: 3 };
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_none());
    }

    #[test]
    fn test_cycle_in_graph() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct CyclicNode {
            id: u32,
            goal: u32,
        }

        impl SearchNode for CyclicNode {
            type Cost = u32;
            fn successors(&self) -> Vec<(Self, u32)> {
                match self.id {
                    0 => vec![(CyclicNode { id: 1, goal: self.goal }, 1)],
                    1 => vec![(CyclicNode { id: 2, goal: self.goal }, 1)],
                    2 => vec![
                        (CyclicNode { id: 0, goal: self.goal }, 1),
                        (CyclicNode { id: 3, goal: self.goal }, 1),
                    ],
                    3 => vec![],
                    _ => vec![],
                }
            }
            fn is_goal(&self) -> bool {
                self.id == self.goal
            }
        }

        let start = CyclicNode { id: 0, goal: 3 };
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 3);
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_multiple_paths_same_cost() {
        #[derive(Clone, PartialEq, Eq, Hash, Debug)]
        struct MultiPathNode {
            id: u32,
            goal: u32,
        }

        impl SearchNode for MultiPathNode {
            type Cost = u32;
            fn successors(&self) -> Vec<(Self, u32)> {
                match self.id {
                    0 => vec![
                        (MultiPathNode { id: 1, goal: self.goal }, 1),
                        (MultiPathNode { id: 2, goal: self.goal }, 1),
                    ],
                    1 => vec![(MultiPathNode { id: 3, goal: self.goal }, 1)],
                    2 => vec![(MultiPathNode { id: 3, goal: self.goal }, 1)],
                    _ => vec![],
                }
            }
            fn is_goal(&self) -> bool {
                self.id == self.goal
            }
        }

        let start = MultiPathNode { id: 0, goal: 3 };
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 2);
        assert_eq!(path.len(), 3);
    }

    // ==================== Large Graph Tests ====================

    #[test]
    fn test_large_linear_graph() {
        let start = LinearNode::new(0, 99);
        let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 99);
        assert_eq!(path.len(), 100);
    }

    #[test]
    fn test_large_grid() {
        let start = GridPos::new(0, 0, 20, 20);
        let result = astar::<_, FibonacciHeap<_, _>>(&start);
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
