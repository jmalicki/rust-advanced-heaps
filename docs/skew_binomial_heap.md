# Skew Binomial Heap

## Overview

The **Skew Binomial Heap** is an extension of binomial heaps that allows
**skew trees**, enabling O(1) insert and merge operations instead of the
O(log n) operations in standard binomial heaps. It maintains O(log n)
delete_min and decrease_key operations.

## Historical Context and Papers

### Original Paper

- **Brodal, Gerth Stølting; Okasaki, Chris** (1996). "Optimal purely
  functional priority queues". *Journal of Functional Programming*. 6 (6):
  839–857. doi:10.1017/S095679680000201X.
  - Introduced skew binomial heaps in the context of functional programming

### Key Follow-up Work

1. **Okasaki, Chris** (1999). *Purely Functional Data Structures*.
   Cambridge University Press. Chapter 9: "Bootstrapping".
   - Comprehensive treatment of functional heap structures including skew
     binomial heaps

2. **Hinze, Ralf** (2000). "A simple implementation technique for priority
   search queues". *Proceedings of the 5th ACM SIGPLAN International
   Conference on Functional Programming*. pp. 110–121.
   doi:10.1145/351240.351253.
   - Applications and variations

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
| --------- | -------------- | ----- |
| `insert` | **O(1)** worst-case | Constant time insertion |
| `find_min` | **O(1)** | Direct access to minimum |
| `delete_min` | **O(log n)** worst-case | Tree traversal and merging |
| `decrease_key` | **O(log n)** worst-case | Bubble up in tree |
| `merge` | **O(1)** worst-case | Constant time merging |

**Key Improvement**: Insert and merge are **O(1)** instead of O(log n) in
standard binomial heaps!

## How It Works

### Skew Binomial Trees

A **skew binomial tree** is similar to a binomial tree but with additional flexibility:

- **Standard binomial tree Bₖ**: Defined recursively, Bₖ has 2ᵏ nodes
- **Skew binomial tree**: Allows "skew" operations that enable O(1) insertion

**Skew property**: A tree can be "skewed" by making a special type of link
that allows efficient insertion without full restructuring.

### Data Structure

A skew binomial heap is a collection of skew binomial trees where:

- Each tree satisfies the min-heap property
- Trees are stored in a list (similar to binomial heaps)
- **Skew flag**: Each tree has a flag indicating if it's "skew"
- Skew trees can be linked efficiently

### Key Operations

#### Insert (O(1) worst-case)

This is the key improvement over standard binomial heaps:

1. Create new single-node tree (rank 0, skew = true)
2. **Skew link** with existing rank-0 tree if present:
   - Two rank-0 trees can be linked to form rank-1 tree
   - This link is **special** - it's O(1) and preserves structure
3. If rank-1 tree exists, cascade (but bounded)

**Why O(1)?** The skew link operation is specially designed to be constant
time, unlike standard binomial tree linking which may require O(log n) work.

#### Skew Link

A skew link is a special type of tree linking:

- Two rank-k skew trees can be linked in O(1) time
- Result is a rank-(k+1) tree (may or may not be skew)
- Preserves heap property and tree structure

This is the innovation: special link operations that are always O(1).

#### Merge (O(1) worst-case)

1. Compare roots
2. **Skew link** smaller with larger
3. If resulting rank exists, cascade

**Why O(1)?** Skew links are O(1), and cascading is bounded (similar to
binary addition with carry).

The key insight is that skew links prevent the O(log n) cost of standard
binomial tree merging.

#### Delete-min (O(log n) worst-case)

1. Find minimum root (O(log n) - scan list)
2. Remove root
3. Add children to heap
4. Merge children back into heap

**Why O(log n)?**

- At most O(log n) children (rank bound)
- Merging children uses skew links (O(1) each)
- Total: O(log n)

#### Decrease-key (O(log n) worst-case)

1. Decrease priority
2. **Bubble up** in tree (swap with parent)
3. Continue until heap property satisfied

**Why O(log n)?** Tree height is O(log n), and we may need to traverse from
leaf to root.

This is the same as binomial heaps - no improvement here.

## Comparison to Binomial Heaps

| Feature | Skew Binomial | Standard Binomial |
| ------- | ------------ | ----------------- |
| Insert | **O(1)** worst | **O(log n)** worst |
| Merge | **O(1)** worst | **O(log n)** worst |
| Delete-min | O(log n) worst | O(log n) worst |
| Decrease-key | O(log n) worst | O(log n) worst |
| Complexity | More complex | Simpler |

**Key Advantage**: O(1) insert and merge operations!

## Why Skew Links Work

The skew link operation is designed to:

1. **Preserve structure**: Result is still a valid skew binomial tree
2. **Be constant time**: No recursive restructuring needed
3. **Allow cascading**: Multiple skew links can cascade efficiently

The mathematical properties of skew binomial trees ensure that:

- Skew links can be applied in O(1) time
- Cascading is bounded (like binary addition)
- Structure is maintained

## Functional Programming Context

Skew binomial heaps were originally developed for **purely functional programming**:

- **Persistent**: Operations don't mutate existing structure
- **Purely functional**: No side effects
- **Efficient**: Still achieve good bounds

The Rust implementation uses mutable pointers (not purely functional), but
the algorithms are adapted from the functional versions.

## Implementation Details

The Rust implementation:

- Tracks skew flag on each node
- Implements skew link operation (O(1))
- Uses special handling for rank-0 trees
- Maintains tree structure during operations
- Handles cascading during insert/merge

## Applications

Skew binomial heaps are useful when:

1. **Many insertions**: O(1) insert is better than O(log n)
2. **Frequent merging**: O(1) merge is beneficial
3. **Functional programming**: Purely functional implementations
4. **Worst-case guarantees**: O(1) operations guaranteed, not just amortized

## References

1. Brodal, G. S., & Okasaki, C. (1996). Optimal purely functional priority
   queues. *Journal of Functional Programming*, 6(6), 839-857.

2. Okasaki, C. (1999). *Purely Functional Data Structures*. Cambridge
   University Press.

3. Hinze, R. (2000). A simple implementation technique for priority search
   queues. *Proceedings of ICFP* 2000, 110-121.
