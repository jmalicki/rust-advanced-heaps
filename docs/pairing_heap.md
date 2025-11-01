# Pairing Heap

## Overview

The **Pairing Heap** is a type of heap-ordered tree that provides excellent amortized performance for priority queue operations, particularly `decrease_key`. It was designed as a simpler alternative to Fibonacci heaps while still achieving sub-logarithmic amortized bounds for `decrease_key`.

## Historical Context and Papers

### Original Paper
- **Fredman, Michael L.; Sedgewick, Robert; Sleator, Daniel D.; Tarjan, Robert E.** (1986). "The pairing heap: A new form of self-adjusting heap". *Algorithmica*. 1 (1): 111–129. doi:10.1007/BF01840439.

### Key Follow-up Work

1. **Fredman, Michael L.** (1999). "On the Efficiency of Pairing Heaps and Related Data Structures". *Journal of the ACM*. 46 (4): 473–501. doi:10.1145/320211.320214.
   - Established that pairing heaps achieve o(log n) amortized `decrease_key`, which is better than O(log n)
   - Showed that the two-pass pairing strategy achieves the best amortized bounds

2. **Iacono, John; Özkan, Özgür** (2014). "A Tight Lower Bound for Decreasing the Key of an Element in a Pairing Heap". *Proceedings of the 25th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*. pp. 1108–1116.
   - Established tight bounds for pairing heap operations

3. **Elmasry, Amr; Jensen, Claus; Katajainen, Jyrki** (2009). "Two-tier relaxed heaps". *Acta Informatica*. 46 (7): 489–504. doi:10.1007/s00236-009-0094-8.
   - Introduced variants and improvements to pairing heaps

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `insert` | **O(1)** amortized | Constant time amortized insertion |
| `find_min` | **O(1)** | Direct access to minimum element |
| `delete_min` | **O(log n)** amortized | Two-pass pairing operation |
| `decrease_key` | **o(log n)** amortized | Sub-logarithmic! Better than O(log n) |
| `merge` | **O(1)** amortized | Constant time heap merging |

### Key Insight: o(log n) vs O(log n)

The notation **o(log n)** means "strictly better than O(log n)" - there exists no constant c such that the operation takes at least c·log n time in the amortized sense. This is an important distinction from binary heaps, which achieve O(log n) worst-case for all operations.

## How It Works

### Data Structure

A pairing heap is a heap-ordered tree where:
- Each node contains an item and a priority
- The tree satisfies the **heap property**: parent priority ≤ child priority
- Unlike binary heaps, nodes can have **any number of children**
- The tree is stored using a multi-way tree structure with:
  - `child`: pointer to first child
  - `sibling`: pointer to next sibling in child list
  - `prev`: pointer to previous sibling or parent (for efficient traversal)

### Key Operations

#### Insert (O(1) amortized)

1. Create a new single-node tree
2. Compare priority with current root
3. If new node has smaller priority, it becomes the root and old root becomes its child
4. Otherwise, new node becomes a child of the root

This is O(1) because we only compare with the root and potentially add one link.

#### Delete-min (O(log n) amortized)

The delete-min operation uses a clever **two-pass pairing** strategy:

**First Pass (Pairing):**
1. When the root is deleted, collect all its children
2. Pair adjacent children: first with second, third with fourth, etc.
3. Merge each pair: the node with smaller priority becomes parent
4. Result: approximately n/2 merged trees

**Second Pass (Right-to-left merge):**
1. Start with the last merged tree
2. Repeatedly merge it with the next tree from right to left
3. Each merge makes the smaller-priority tree the root

**Why Two-Pass?**

The two-pass strategy ensures that the amortized cost is O(log n). The first pass reduces the number of trees quickly, and the second pass ensures they're merged in a way that maintains good balance. This is critical for achieving the amortized bounds.

#### Decrease-key (o(log n) amortized)

1. Decrease the priority value
2. If heap property is violated (new priority < parent priority):
   - **Cut** the node from its parent (remove from child list)
   - Add it as a child of the root (or make it the new root if smaller)
3. The cutting operation is O(1), but may cause cascading cuts

The sub-logarithmic bound comes from the amortized analysis showing that, while individual operations might take O(log n), over a sequence of operations, the average cost is strictly less than O(log n).

#### Merge (O(1) amortized)

1. Compare roots of two heaps
2. Make the larger-priority root a child of the smaller-priority root
3. Update the root pointer

This is trivially O(1) - just one comparison and pointer update.

## Amortized Analysis Intuition

The amortized analysis of pairing heaps relies on a potential function argument. The key insight is:

1. **Tree structure**: Pairing heaps can become unbalanced, but delete-min operations tend to rebalance them
2. **Decrease-key**: Most decrease-key operations are cheap (cutting near the root), while expensive ones (deep cuts) are rare
3. **Pairing strategy**: The two-pass pairing ensures that the "work debt" is distributed across operations

The potential function typically charges:
- Credits for each node's position in the tree (deeper = more credit)
- When expensive operations occur, they consume credits from many cheap operations

## Comparison to Other Heaps

| Feature | Pairing Heap | Fibonacci Heap | Binary Heap |
|---------|-------------|----------------|-------------|
| Insert | O(1) am. | O(1) am. | O(log n) worst |
| Delete-min | O(log n) am. | O(log n) am. | O(log n) worst |
| Decrease-key | **o(log n) am.** | **O(1) am.** | O(log n) worst |
| Complexity | Simple | Complex | Very simple |
| Practical | Often faster | High overhead | Fast for small n |

**When to use Pairing Heaps:**
- Need better than O(log n) decrease-key but want simpler code than Fibonacci heaps
- Many decrease-key operations relative to other operations
- Want simplicity without sacrificing too much performance

## Implementation Details

The Rust implementation uses:
- Unsafe pointers (`NonNull`) for efficient tree manipulation
- Type-erased handles to allow handles to be passed without generic type parameters
- Recursive tree structure with parent/child/sibling pointers
- Careful memory management to prevent leaks

## References

1. Fredman, M. L., Sedgewick, R., Sleator, D. D., & Tarjan, R. E. (1986). The pairing heap: A new form of self-adjusting heap. *Algorithmica*, 1(1), 111-129.

2. Fredman, M. L. (1999). On the efficiency of pairing heaps and related data structures. *Journal of the ACM*, 46(4), 473-501.

3. Iacono, J., & Özkan, Ö. (2014). A tight lower bound for decreasing the key of an element in a pairing heap. *Proceedings of SODA* 2014, 1108-1116.

4. Elmasry, A., Jensen, C., & Katajainen, J. (2009). Two-tier relaxed heaps. *Acta Informatica*, 46(7), 489-504.

