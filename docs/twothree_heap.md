# 2-3 Heap

## Overview

The **2-3 Heap** is a balanced heap data structure where each internal node has exactly 2 or 3 children. This balanced structure allows efficient operations while maintaining good performance for decrease_key operations.

## Historical Context and Papers

### Original Paper
- **Carlsson, Svante** (1987). "A variant of heapsort with almost optimal number of comparisons". *Information Processing Letters*. 24 (4): 247–250. doi:10.1016/0020-0190(87)90142-6.
   - Introduced 2-3 heaps as a variant of heapsort

### Key Follow-up Work

1. **Carlsson, Svante; Chen, Jingsen; Mattsson, Christer** (1991). "An implicit binomial queue with constant insertion time". *Proceedings of the 1st Annual Scandinavian Workshop on Algorithm Theory (SWAT)*. pp. 1–13. doi:10.1007/3-540-52846-8_49.
   - Related work on implicit heaps

2. **Driscoll, James R.; Gabow, Harold N.; Shrairman, Ruth; Tarjan, Robert E.** (1988). "Relaxed heaps: An alternative to Fibonacci heaps with applications to parallel computation". *Communications of the ACM*, 31(11), 1343-1354.
   - Related relaxed heap structures

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `insert` | **O(1)** amortized | Constant time amortized insertion |
| `find_min` | **O(1)** | Direct access to minimum |
| `delete_min` | **O(log n)** amortized | Tree height is O(log n) |
| `decrease_key` | **O(1)** amortized | Bubble up in balanced tree |
| `merge` | **O(1)** amortized | Simple root comparison |

**Note**: These are amortized bounds. The balanced structure ensures good performance.

## How It Works

### Data Structure

A 2-3 heap is a heap-ordered tree where:
- **Each internal node has exactly 2 or 3 children** (balanced structure)
- **Leaves have no children**
- **Tree satisfies min-heap property**: parent priority ≤ child priority
- **Height is O(log n)**: Balanced structure ensures logarithmic height

The 2-3 constraint ensures:
- **Balance**: Tree remains balanced after operations
- **Efficiency**: Fewer pointer operations than arbitrary multi-way trees
- **Simplicity**: Easier to reason about than more complex structures

### Why 2-3?

The 2-3 constraint is a balance between:
- **Too restrictive (binary)**: O(log n) worst-case for all operations
- **Too permissive (arbitrary)**: Hard to maintain balance

By requiring 2 or 3 children:
- **Balance**: Height remains O(log n)
- **Flexibility**: Allows efficient restructuring
- **Simplicity**: Only two cases to handle (2 vs 3 children)

### Key Operations

#### Insert (O(1) amortized)

1. Create new leaf node
2. Find appropriate parent (maintain 2-3 structure)
3. If parent has 3 children, **split**:
   - Take last 2 children
   - Create new node with those children
   - Insert new node into parent's parent
   - May cascade upward (rarely)

**Amortized analysis**: Splits are rare, and each split creates O(1) work.

#### Maintain Structure

When a node has 4 children (violation of 2-3 rule):
1. **Split**: Take 2 children, create new node
2. **Insert new node** into parent
3. **Cascade**: If parent now has 4 children, split again

The cascade stops quickly because:
- Balanced structure prevents deep cascades
- Amortized over insertions, splits are rare

#### Delete-min (O(log n) amortized)

1. Remove root
2. **Promote minimum child** to root (or merge children)
3. **Rebalance**: Ensure 2-3 structure is maintained
4. If root has only 1 child, restructure (merge with sibling or promote)

**Why O(log n)?** Height is O(log n), and restructuring is local.

#### Decrease-key (O(1) amortized)

1. Decrease priority
2. **Bubble up** if heap property violated:
   - Swap with parent if parent has larger priority
   - Continue upward until heap property satisfied

**Why O(1) amortized?** The balanced structure ensures that:
- Most bubbles are shallow (near leaves)
- Deep bubbles are rare
- Amortized over operations, average is O(1)

This is similar to the analysis for pairing heaps.

#### Merge (O(1) amortized)

1. Compare roots
2. Make larger priority root a child of smaller
3. **Maintain structure**: If root now has 4 children, split

Simple operation with structure maintenance.

### Splitting and Merging

The key operations for maintaining 2-3 structure:

**Splitting** (when node has 4 children):
1. Take last 2 children
2. Create new node with those children
3. Insert new node into parent
4. Update parent's child count

**Merging** (when node has 1 child):
1. If sibling has 2 children: borrow one
2. If sibling has 3 children: merge nodes
3. May cascade upward

These operations maintain the 2-3 invariant.

## Comparison to Other Heaps

| Feature | 2-3 Heap | Binary Heap | Fibonacci Heap |
|---------|----------|-------------|----------------|
| Insert | O(1) am. | O(log n) | O(1) am. |
| Delete-min | O(log n) am. | O(log n) | O(log n) am. |
| Decrease-key | O(1) am. | O(log n) | O(1) am. |
| Structure | Balanced | Complete | Lazy |
| Complexity | Moderate | Simple | Complex |

**Advantages:**
- Better bounds than binary heaps
- Simpler than Fibonacci heaps
- Balanced structure (predictable)

**Disadvantages:**
- More complex than binary heaps
- Amortized bounds (not worst-case)
- Less efficient than Fibonacci heaps for decrease_key

## Implementation Details

The Rust implementation:
- Maintains children in a vector (2 or 3 elements)
- Implements splitting when 4 children present
- Implements merging when 1 child present
- Tracks minimum separately for O(1) find-min
- Handles structure maintenance carefully

## Applications

2-3 heaps are used when:
1. Need better than O(log n) decrease-key
2. Want simpler code than Fibonacci heaps
3. Balanced structure is beneficial
4. Amortized bounds are acceptable

## References

1. Carlsson, S. (1987). A variant of heapsort with almost optimal number of comparisons. *Information Processing Letters*, 24(4), 247-250.

2. Carlsson, S., Chen, J., & Mattsson, C. (1991). An implicit binomial queue with constant insertion time. *Proceedings of SWAT* 1991, 1-13.

3. Driscoll, J. R., Gabow, H. N., Shrairman, R., & Tarjan, R. E. (1988). Relaxed heaps: An alternative to Fibonacci heaps with applications to parallel computation. *Communications of the ACM*, 31(11), 1343-1354.

