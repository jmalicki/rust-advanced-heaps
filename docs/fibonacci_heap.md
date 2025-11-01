# Fibonacci Heap

## Overview

The **Fibonacci Heap** is one of the most theoretically optimal heap data structures, achieving O(1) amortized time for `insert`, `decrease_key`, and `merge` operations. It represents a fundamental achievement in amortized analysis and is often used as a building block for advanced algorithms.

## Historical Context and Papers

### Original Paper
- **Fredman, Michael L.; Tarjan, Robert E.** (1987). "Fibonacci heaps and their uses in improved network optimization algorithms". *Journal of the ACM*. 34 (3): 596–615. doi:10.1145/28869.28874.
   - This paper introduced Fibonacci heaps and showed how they improve algorithms like Dijkstra's shortest path and Prim's minimum spanning tree

### Key Follow-up Work

1. **Fredman, Michael L.** (1997). "A priority queue transformation". *Theory of Computing Systems*. 30 (2): 155–167. doi:10.1007/BF02679444.
   - Further analysis of Fibonacci heap properties

2. **Driscoll, James R.; Gabow, Harold N.; Shrairman, Ruth; Tarjan, Robert E.** (1988). "Relaxed heaps: An alternative to Fibonacci heaps with applications to parallel computation". *Communications of the ACM*. 31 (11): 1343–1354. doi:10.1145/50087.50090.
   - Introduced relaxed heaps as an alternative with better worst-case bounds

3. **Brodal, Gerth Stølting** (1996). "Worst-case efficient priority queues". *Proceedings of the 7th Annual ACM-SIAM Symposium on Discrete Algorithms*. pp. 52–58.
   - Brodal heaps achieve the same bounds as Fibonacci heaps but in worst-case rather than amortized time

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `insert` | **O(1)** amortized | Constant time insertion |
| `find_min` | **O(1)** | Direct pointer to minimum |
| `delete_min` | **O(log n)** amortized | Consolidation step |
| `decrease_key` | **O(1)** amortized | Cascading cuts maintain structure |
| `merge` | **O(1)** amortized | Simple list concatenation |

**Note**: These are the best possible amortized bounds for a comparison-based heap supporting these operations.

## How It Works

### Data Structure

A Fibonacci heap is a collection of **heap-ordered trees** where:
- Each tree satisfies the min-heap property
- Trees are organized in a **circular doubly-linked list** of roots
- The minimum element is tracked with a pointer
- Nodes maintain:
  - `degree`: number of children
  - `marked`: flag used in cascading cuts
  - Circular doubly-linked lists for children

**Key Invariant**: After consolidation, there is at most one tree of each degree.

### Key Operations

#### Insert (O(1) amortized)

1. Create a new single-node tree (degree 0)
2. Add it to the root list (O(1) - just link into circular list)
3. Update minimum pointer if necessary (O(1))

This is truly O(1) - no restructuring needed.

#### Consolidate (Used in delete-min)

The consolidation operation is critical:
1. Create a degree table (array indexed by degree)
2. For each root tree:
   - If another tree with the same degree exists, link them
   - Continue linking until no duplicate degrees
3. Link smaller tree as child of larger tree (heap order)
4. Update root list and minimum pointer

**Key Insight**: After consolidation, there are at most O(log n) root trees, since degrees are bounded by log n (similar to binomial heaps).

#### Delete-min (O(log n) amortized)

1. Remove minimum root from root list
2. Add its children to root list (each becomes a root)
3. **Consolidate** the root list (ensures degree invariant)
4. Scan roots to find new minimum

The consolidation step ensures O(log n) amortized cost because:
- At most O(log n) roots after consolidation
- Each link operation is O(1)
- The potential function ensures amortized bound

#### Decrease-key (O(1) amortized)

This is the operation that makes Fibonacci heaps special:

1. Decrease the priority value
2. If heap property is violated:
   - **Cut** the node from its parent
   - Add it to root list
   - If parent was marked, recursively cut parent (cascading cut)
   - Mark parent (if it's not root and not already marked)

**Why Cascading Cuts?**

Cascading cuts ensure that no node loses too many children. The invariant is:
- A node can lose **at most one child** before being cut itself
- This maintains the structural property: a tree of degree k has at least F_{k+2} nodes (where F_k is the k-th Fibonacci number)

This Fibonacci number relationship gives the data structure its name!

#### Merge (O(1) amortized)

Simply concatenate the two root lists:
1. Link the circular lists together
2. Update minimum pointer

This is clearly O(1).

### The Fibonacci Number Connection

After k children are removed from a node, the node is cut. This means:
- A tree with root of degree k must have at least F_{k+2} nodes
- Where F_k is the k-th Fibonacci number (F_1 = 1, F_2 = 1, F_3 = 2, F_4 = 3, ...)
- This exponential growth bounds the maximum degree by log_φ(n) ≈ 1.44·log₂(n)
- Where φ = (1+√5)/2 is the golden ratio

This is why consolidation runs in O(log n) time - there are at most O(log n) distinct degrees!

## Amortized Analysis

The amortized analysis uses a **potential function** that charges credits based on:
1. Number of trees in root list
2. Number of marked nodes

**Key insights:**
- Each operation pays for its actual cost plus changes in potential
- Cheap operations (insert, decrease-key) increase potential slightly
- Expensive operations (delete-min) consume large potential
- The potential balances out over sequences of operations

## Practical Considerations

**Pros:**
- Theoretically optimal bounds
- Excellent for algorithms with many decrease-key operations (e.g., Dijkstra's algorithm)
- Merge operation is very fast

**Cons:**
- High constant overhead (pointer manipulation, circular lists)
- Complex implementation
- Often slower than binary heaps for small inputs
- Marked nodes add memory overhead

**When to use:**
- Large graphs in shortest path algorithms
- Algorithms with many decrease-key operations
- When theoretical bounds matter more than constant factors

## Applications

Fibonacci heaps are particularly useful for:

1. **Dijkstra's Algorithm**: O(m + n log n) instead of O(m log n) with binary heap
2. **Prim's Algorithm**: Same improvement for MST construction
3. **Various graph algorithms**: Any algorithm with many decrease-key operations

## Implementation Details

The Rust implementation:
- Uses unsafe pointers for efficiency
- Maintains circular doubly-linked lists for O(1) insertion/deletion
- Implements full cascading cuts
- Tracks degrees carefully for consolidation
- Uses type-erased handles for API cleanliness

## References

1. Fredman, M. L., & Tarjan, R. E. (1987). Fibonacci heaps and their uses in improved network optimization algorithms. *Journal of the ACM*, 34(3), 596-615.

2. Fredman, M. L. (1997). A priority queue transformation. *Theory of Computing Systems*, 30(2), 155-167.

3. Driscoll, J. R., Gabow, H. N., Shrairman, R., & Tarjan, R. E. (1988). Relaxed heaps: An alternative to Fibonacci heaps with applications to parallel computation. *Communications of the ACM*, 31(11), 1343-1354.

4. Brodal, G. S. (1996). Worst-case efficient priority queues. *Proceedings of SODA* 1996, 52-58.

