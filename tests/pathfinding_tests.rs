//! Comprehensive tests for pathfinding algorithms
//!
//! Tests cover:
//! - Basic functionality with different heap implementations
//! - Edge cases (empty paths, unreachable goals, cycles)
//! - Correctness (optimal paths with decrease_key scenarios)
//! - Property-based testing
//! - Performance characteristics

use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::pathfinding::{
    astar, dijkstra, reachable_within, AStarNode, PathFinderBuilder, SearchNode,
};
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
// Note: BinomialHeap is not used in these tests because it has ownership limitations
// that conflict with storing handles during pathfinding (it requires exclusive ownership
// when popping nodes, which fails when handles are kept for decrease_key operations).

// ============================================================================
// Test Node Types
// ============================================================================

/// Simple numbered node for basic tests - carries its goal
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct SimpleNode {
    value: u32,
    goal: u32,
}

impl SimpleNode {
    fn new(value: u32, goal: u32) -> Self {
        SimpleNode { value, goal }
    }
}

impl SearchNode for SimpleNode {
    type Cost = u32;

    fn successors(&self) -> Vec<(Self, u32)> {
        if self.value < 1000 {
            vec![(SimpleNode::new(self.value + 1, self.goal), 1)]
        } else {
            vec![]
        }
    }

    fn is_goal(&self) -> bool {
        self.value == self.goal
    }
}

/// Node for reachable_within tests - no goal needed
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ReachableNode(u32);

impl SearchNode for ReachableNode {
    type Cost = u32;

    fn successors(&self) -> Vec<(Self, u32)> {
        if self.0 < 1000 {
            vec![(ReachableNode(self.0 + 1), 1)]
        } else {
            vec![]
        }
    }

    fn is_goal(&self) -> bool {
        false // reachable_within doesn't use is_goal
    }
}

/// Grid position for 2D pathfinding - carries goal coordinates
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Grid2D {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    goal_x: i32,
    goal_y: i32,
}

impl Grid2D {
    fn new(x: i32, y: i32, width: i32, height: i32, goal_x: i32, goal_y: i32) -> Self {
        Grid2D {
            x,
            y,
            width,
            height,
            goal_x,
            goal_y,
        }
    }

    fn bounded(x: i32, y: i32, size: i32, goal_x: i32, goal_y: i32) -> Self {
        Grid2D::new(x, y, size, size, goal_x, goal_y)
    }
}

impl SearchNode for Grid2D {
    type Cost = u32;

    fn successors(&self) -> Vec<(Self, u32)> {
        let mut neighbors = Vec::new();
        let directions = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        for (dx, dy) in directions {
            let nx = self.x + dx;
            let ny = self.y + dy;

            if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                neighbors.push((
                    Grid2D::new(nx, ny, self.width, self.height, self.goal_x, self.goal_y),
                    1,
                ));
            }
        }

        neighbors
    }

    fn is_goal(&self) -> bool {
        self.x == self.goal_x && self.y == self.goal_y
    }
}

impl AStarNode for Grid2D {
    fn heuristic(&self) -> u32 {
        ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as u32
    }
}

/// Graph structure for complex test scenarios
struct TestGraph {
    nodes: std::collections::HashMap<char, Vec<(char, u32)>>,
}

impl TestGraph {
    fn new() -> Self {
        TestGraph {
            nodes: std::collections::HashMap::new(),
        }
    }

    fn add_edge(&mut self, from: char, to: char, weight: u32) {
        self.nodes.entry(from).or_default().push((to, weight));
    }

    fn add_undirected_edge(&mut self, a: char, b: char, weight: u32) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }

    fn node(&self, id: char, goal: char) -> TestGraphNode<'_> {
        TestGraphNode {
            id,
            goal,
            graph: self,
        }
    }
}

#[derive(Clone)]
struct TestGraphNode<'a> {
    id: char,
    goal: char,
    graph: &'a TestGraph,
}

impl<'a> PartialEq for TestGraphNode<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<'a> Eq for TestGraphNode<'a> {}

impl<'a> std::hash::Hash for TestGraphNode<'a> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<'a> SearchNode for TestGraphNode<'a> {
    type Cost = u32;

    fn successors(&self) -> Vec<(Self, u32)> {
        self.graph
            .nodes
            .get(&self.id)
            .map(|edges| {
                edges
                    .iter()
                    .map(|&(to, weight)| {
                        (
                            TestGraphNode {
                                id: to,
                                goal: self.goal,
                                graph: self.graph,
                            },
                            weight,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn is_goal(&self) -> bool {
        self.id == self.goal
    }
}

// ============================================================================
// Dijkstra Tests - All Heap Types
// ============================================================================

macro_rules! test_dijkstra_with_heap {
    ($heap_type:ty, $test_name:ident) => {
        mod $test_name {
            use super::*;

            #[test]
            fn basic_path() {
                let start = SimpleNode::new(0, 10);
                let result = dijkstra::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (path, cost) = result.unwrap();
                assert_eq!(cost, 10);
                assert_eq!(path.len(), 11);
                assert_eq!(path[0].value, 0);
                assert_eq!(path[10].value, 10);
            }

            #[test]
            fn start_equals_goal() {
                let start = SimpleNode::new(5, 5);
                let result = dijkstra::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (path, cost) = result.unwrap();
                assert_eq!(cost, 0);
                assert_eq!(path.len(), 1);
            }

            #[test]
            fn unreachable_goal() {
                let start = SimpleNode::new(0, 5000);
                let result = dijkstra::<_, $heap_type>(&start);
                assert!(result.is_none());
            }

            #[test]
            fn bounded_grid() {
                let start = Grid2D::bounded(0, 0, 10, 9, 9);
                let result = dijkstra::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (path, cost) = result.unwrap();
                assert_eq!(cost, 18); // Manhattan distance
                assert_eq!(path[0].x, 0);
                assert_eq!(path[0].y, 0);
                let last = path.last().unwrap();
                assert_eq!(last.x, 9);
                assert_eq!(last.y, 9);
            }

            #[test]
            fn grid_corner_to_corner() {
                let start = Grid2D::bounded(0, 0, 5, 4, 4);
                let result = dijkstra::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (_, cost) = result.unwrap();
                assert_eq!(cost, 8);
            }
        }
    };
}

test_dijkstra_with_heap!(FibonacciHeap<usize, _>, dijkstra_fibonacci);
test_dijkstra_with_heap!(PairingHeap<usize, _>, dijkstra_pairing);
// Note: BinomialHeap has a limitation where it expects exclusive ownership on pop,
// which conflicts with keeping handles for decrease_key. Use for simple linear paths only.
test_dijkstra_with_heap!(RankPairingHeap<usize, _>, dijkstra_rank_pairing);

// ============================================================================
// A* Tests - All Heap Types
// ============================================================================

macro_rules! test_astar_with_heap {
    ($heap_type:ty, $test_name:ident) => {
        mod $test_name {
            use super::*;

            #[test]
            fn basic_grid_path() {
                let start = Grid2D::bounded(0, 0, 20, 10, 10);
                let result = astar::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (path, cost) = result.unwrap();
                assert_eq!(cost, 20);
                assert_eq!(path[0].x, 0);
                assert_eq!(path[0].y, 0);
                let last = path.last().unwrap();
                assert_eq!(last.x, 10);
                assert_eq!(last.y, 10);
            }

            #[test]
            fn same_start_and_goal() {
                let start = Grid2D::bounded(5, 5, 10, 5, 5);
                let result = astar::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (path, cost) = result.unwrap();
                assert_eq!(cost, 0);
                assert_eq!(path.len(), 1);
            }

            #[test]
            fn adjacent_nodes() {
                let start = Grid2D::bounded(5, 5, 10, 6, 5);
                let result = astar::<_, $heap_type>(&start);
                assert!(result.is_some());
                let (path, cost) = result.unwrap();
                assert_eq!(cost, 1);
                assert_eq!(path.len(), 2);
            }
        }
    };
}

test_astar_with_heap!(FibonacciHeap<usize, _>, astar_fibonacci);
test_astar_with_heap!(PairingHeap<usize, _>, astar_pairing);
// Note: BinomialHeap excluded due to ownership limitations (see dijkstra tests)
test_astar_with_heap!(RankPairingHeap<usize, _>, astar_rank_pairing);

// ============================================================================
// Decrease Key Correctness Tests
// ============================================================================

/// Node designed to force decrease_key usage
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct DecreaseKeyTestNode {
    id: u32,
    goal: u32,
}

impl DecreaseKeyTestNode {
    fn new(id: u32, goal: u32) -> Self {
        DecreaseKeyTestNode { id, goal }
    }
}

impl SearchNode for DecreaseKeyTestNode {
    type Cost = u32;

    fn successors(&self) -> Vec<(Self, u32)> {
        // Graph structure:
        //
        //     0 --100--> 2
        //     |          ^
        //     1          |
        //     v          10
        //     1 ---------+
        //     |
        //     1
        //     v
        //     3 ---1---> 4
        //
        // Without decrease_key: 0->2 costs 100
        // With decrease_key: 0->1->2 costs 11, 0->1->3->4 costs 3
        match self.id {
            0 => vec![
                (DecreaseKeyTestNode::new(1, self.goal), 1),
                (DecreaseKeyTestNode::new(2, self.goal), 100),
            ],
            1 => vec![
                (DecreaseKeyTestNode::new(2, self.goal), 10),
                (DecreaseKeyTestNode::new(3, self.goal), 1),
            ],
            3 => vec![(DecreaseKeyTestNode::new(4, self.goal), 1)],
            _ => vec![],
        }
    }

    fn is_goal(&self) -> bool {
        self.id == self.goal
    }
}

#[test]
fn test_decrease_key_optimal_path_fibonacci() {
    let start = DecreaseKeyTestNode::new(0, 2);
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    // Optimal is 0->1->2 = 11, not 0->2 = 100
    assert_eq!(cost, 11);
    assert_eq!(path.len(), 3);
    assert_eq!(path[1].id, 1);
}

#[test]
fn test_decrease_key_optimal_path_pairing() {
    let start = DecreaseKeyTestNode::new(0, 2);
    let result = dijkstra::<_, PairingHeap<_, _>>(&start);
    assert!(result.is_some());
    let (_, cost) = result.unwrap();
    assert_eq!(cost, 11);
}

#[test]
fn test_decrease_key_longer_optimal_path() {
    let start = DecreaseKeyTestNode::new(0, 4);
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    // Optimal path: 0->1->3->4 = 3
    assert_eq!(cost, 3);
    assert_eq!(path.len(), 4);
}

// ============================================================================
// Complex Graph Tests
// ============================================================================

#[test]
fn test_weighted_graph_with_multiple_paths() {
    let mut graph = TestGraph::new();
    // Create a diamond-shaped graph:
    //       B
    //      /|\
    //     1 | 2
    //    /  |  \
    //   A   3   D
    //    \  |  /
    //     4 | 1
    //      \|/
    //       C
    graph.add_edge('A', 'B', 1);
    graph.add_edge('A', 'C', 4);
    graph.add_edge('B', 'C', 3);
    graph.add_edge('B', 'D', 2);
    graph.add_edge('C', 'D', 1);

    let start = graph.node('A', 'D');
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    // Optimal: A->B->D = 3
    assert_eq!(cost, 3);
    assert_eq!(path.len(), 3);
}

#[test]
fn test_graph_with_cycles() {
    let mut graph = TestGraph::new();
    // Create a graph with a cycle:
    // A -> B -> C -> A (cycle)
    //      |
    //      v
    //      D
    graph.add_edge('A', 'B', 1);
    graph.add_edge('B', 'C', 1);
    graph.add_edge('C', 'A', 1);
    graph.add_edge('B', 'D', 5);

    let start = graph.node('A', 'D');
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    assert_eq!(cost, 6); // A->B->D
    assert_eq!(path.len(), 3);
}

#[test]
fn test_undirected_graph() {
    let mut graph = TestGraph::new();
    graph.add_undirected_edge('A', 'B', 1);
    graph.add_undirected_edge('B', 'C', 2);
    graph.add_undirected_edge('A', 'C', 10);

    let start = graph.node('A', 'C');
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (_, cost) = result.unwrap();
    // Should go A->B->C (cost 3) not A->C (cost 10)
    assert_eq!(cost, 3);
}

// ============================================================================
// PathFinderBuilder Tests
// ============================================================================

#[test]
fn test_builder_max_cost_prevents_exploration() {
    let start = SimpleNode::new(0, 10);
    let result = PathFinderBuilder::new(start)
        .max_cost(5)
        .dijkstra::<FibonacciHeap<_, _>>();
    assert!(result.is_none());
}

#[test]
fn test_builder_max_cost_allows_valid_path() {
    let start = SimpleNode::new(0, 5);
    let result = PathFinderBuilder::new(start)
        .max_cost(10)
        .dijkstra::<FibonacciHeap<_, _>>();
    assert!(result.is_some());
    let (_, cost) = result.unwrap();
    assert_eq!(cost, 5);
}

#[test]
fn test_builder_max_nodes_prevents_exploration() {
    let start = SimpleNode::new(0, 10);
    let result = PathFinderBuilder::new(start)
        .max_nodes(3)
        .dijkstra::<FibonacciHeap<_, _>>();
    assert!(result.is_none());
}

#[test]
fn test_builder_combined_limits() {
    let start = SimpleNode::new(0, 20);
    let result = PathFinderBuilder::new(start)
        .max_cost(100)
        .max_nodes(50)
        .dijkstra::<FibonacciHeap<_, _>>();
    assert!(result.is_some());
}

// ============================================================================
// Reachable Within Tests
// ============================================================================

#[test]
fn test_reachable_within_linear() {
    let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&ReachableNode(0), 10);
    assert_eq!(reachable.len(), 11); // 0 through 10

    // Verify all costs are correct
    for (node, cost) in &reachable {
        assert_eq!(*cost, node.0);
    }
}

#[test]
fn test_reachable_within_zero_cost() {
    let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&ReachableNode(50), 0);
    assert_eq!(reachable.len(), 1);
    assert_eq!(reachable[0].0, ReachableNode(50));
    assert_eq!(reachable[0].1, 0);
}

#[test]
fn test_reachable_within_grid() {
    // For reachable_within we need a grid that doesn't care about goal
    let start = Grid2D::bounded(5, 5, 11, 0, 0); // goal doesn't matter here
    let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&start, 2);

    // At distance 0: 1 node (center)
    // At distance 1: 4 nodes (cardinal directions)
    // At distance 2: 8 nodes (including diagonals via 2 steps)
    // Total: 1 + 4 + 8 = 13 nodes
    assert_eq!(reachable.len(), 13);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_node_graph_is_goal() {
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct IsolatedGoalNode;

    impl SearchNode for IsolatedGoalNode {
        type Cost = u32;
        fn successors(&self) -> Vec<(Self, u32)> {
            vec![]
        }
        fn is_goal(&self) -> bool {
            true
        }
    }

    // Goal is start
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&IsolatedGoalNode);
    assert!(result.is_some());
}

#[test]
fn test_single_node_graph_not_goal() {
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct IsolatedNonGoalNode;

    impl SearchNode for IsolatedNonGoalNode {
        type Cost = u32;
        fn successors(&self) -> Vec<(Self, u32)> {
            vec![]
        }
        fn is_goal(&self) -> bool {
            false
        }
    }

    // Goal is not start (unreachable)
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&IsolatedNonGoalNode);
    assert!(result.is_none());
}

#[test]
fn test_self_loop() {
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct SelfLoopNode {
        id: u32,
        goal: u32,
    }

    impl SearchNode for SelfLoopNode {
        type Cost = u32;
        fn successors(&self) -> Vec<(Self, u32)> {
            vec![
                (
                    SelfLoopNode {
                        id: self.id,
                        goal: self.goal,
                    },
                    1,
                ), // Self loop
                (
                    SelfLoopNode {
                        id: self.id + 1,
                        goal: self.goal,
                    },
                    2,
                ), // Forward edge
            ]
        }
        fn is_goal(&self) -> bool {
            self.id == self.goal
        }
    }

    let start = SelfLoopNode { id: 0, goal: 3 };
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    assert_eq!(cost, 6); // 0->1->2->3, each costs 2
    assert_eq!(path.len(), 4);
}

#[test]
fn test_high_branching_factor() {
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct HighBranchNode {
        id: u32,
        goal: u32,
    }

    impl SearchNode for HighBranchNode {
        type Cost = u32;
        fn successors(&self) -> Vec<(Self, u32)> {
            if self.id >= 100 {
                return vec![];
            }
            // Each node connects to 10 successors
            (0..10)
                .map(|i| {
                    (
                        HighBranchNode {
                            id: self.id * 10 + i + 1,
                            goal: self.goal,
                        },
                        1,
                    )
                })
                .collect()
        }
        fn is_goal(&self) -> bool {
            self.id == self.goal
        }
    }

    let start = HighBranchNode { id: 0, goal: 15 };
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    assert_eq!(cost, 2); // Direct path 0 -> 1-10 (pick 1) -> 11-20 (15 is in this range)
    assert_eq!(path.len(), 3);
}

// ============================================================================
// Consistency Tests (Dijkstra vs A*)
// ============================================================================

#[test]
fn test_dijkstra_astar_same_result() {
    let start = Grid2D::bounded(0, 0, 15, 10, 10);

    let dijkstra_result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    let astar_result = astar::<_, FibonacciHeap<_, _>>(&start);

    assert!(dijkstra_result.is_some());
    assert!(astar_result.is_some());

    let (_, dijkstra_cost) = dijkstra_result.unwrap();
    let (_, astar_cost) = astar_result.unwrap();

    // Both should find optimal path
    assert_eq!(dijkstra_cost, astar_cost);
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

#[test]
fn test_long_path() {
    let start = SimpleNode::new(0, 500);
    let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (path, cost) = result.unwrap();
    assert_eq!(cost, 500);
    assert_eq!(path.len(), 501);
}

#[test]
fn test_large_grid_astar() {
    let start = Grid2D::bounded(0, 0, 50, 49, 49);
    let result = astar::<_, FibonacciHeap<_, _>>(&start);
    assert!(result.is_some());
    let (_, cost) = result.unwrap();
    assert_eq!(cost, 98); // Manhattan distance in bounded grid
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn dijkstra_path_cost_matches_sum(goal in 1u32..50) {
            let start = SimpleNode::new(0, goal);
            let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
            prop_assert!(result.is_some());
            let (path, cost) = result.unwrap();

            // Cost should equal goal (since each step costs 1)
            prop_assert_eq!(cost, goal);

            // Path length should be goal + 1
            prop_assert_eq!(path.len() as u32, goal + 1);

            // Path should be contiguous
            for (i, node) in path.iter().enumerate() {
                prop_assert_eq!(node.value, i as u32);
            }
        }

        #[test]
        fn grid_path_cost_equals_manhattan(x in 1i32..20, y in 1i32..20) {
            let start = Grid2D::bounded(0, 0, 25, x, y);
            let result = astar::<_, FibonacciHeap<_, _>>(&start);
            prop_assert!(result.is_some());
            let (_, cost) = result.unwrap();

            let expected = (x + y) as u32;
            prop_assert_eq!(cost, expected);
        }

        #[test]
        fn reachable_count_within_budget(budget in 0u32..20) {
            let reachable = reachable_within::<_, FibonacciHeap<_, _>>(&ReachableNode(0), budget);

            // For linear graph, should have budget + 1 reachable nodes
            prop_assert_eq!(reachable.len() as u32, budget + 1);

            // All costs should be within budget
            for (_, cost) in &reachable {
                prop_assert!(*cost <= budget);
            }
        }

        #[test]
        fn dijkstra_finds_path_when_exists(start_val in 0u32..50, goal in 0u32..100) {
            let start = SimpleNode::new(start_val, goal);
            let result = dijkstra::<_, FibonacciHeap<_, _>>(&start);

            if goal >= start_val && goal <= 1000 {
                // Path should exist
                prop_assert!(result.is_some());
                let (path, cost) = result.unwrap();
                prop_assert_eq!(cost, goal - start_val);
                prop_assert_eq!(path[0].value, start_val);
                prop_assert_eq!(path.last().unwrap().value, goal);
            } else if goal < start_val {
                // Path doesn't exist (can only go forward)
                prop_assert!(result.is_none());
            }
        }
    }
}

// ============================================================================
// Different Heap Type Comparison
// ============================================================================

#[test]
fn test_all_heaps_produce_same_cost() {
    let start = Grid2D::bounded(0, 0, 15, 10, 10);

    let fib_result = dijkstra::<_, FibonacciHeap<_, _>>(&start);
    let pair_result = dijkstra::<_, PairingHeap<_, _>>(&start);
    // BinomialHeap excluded due to ownership limitations with handle storage
    let rank_result = dijkstra::<_, RankPairingHeap<_, _>>(&start);

    let (_, fib_cost) = fib_result.unwrap();
    let (_, pair_cost) = pair_result.unwrap();
    let (_, rank_cost) = rank_result.unwrap();

    assert_eq!(fib_cost, pair_cost);
    assert_eq!(pair_cost, rank_cost);
}
