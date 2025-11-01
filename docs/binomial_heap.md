# Binomial Heap

## Overview

The **Binomial Heap** is a heap data structure that extends the concept of binomial trees to a collection of trees. While it has worse amortized bounds than Fibonacci heaps for `decrease_key`, it is significantly simpler to implement and understand, making it a good pedagogical tool and practical choice when decrease_key operations are infrequent.

## Historical Context and Papers

### Original Paper
- **Vuillemin, Jean** (1978). "A data structure for manipulating priority queues". *Communications of the ACM*. 21 (4): 309–315. doi:10.1145/359460.359478.
   - Introduced binomial heaps as a data structure for priority queues

### Key Follow-up Work

1. **Brown, Mark R.** (1978). "Implementation and analysis of binomial queue algorithms". *SIAM Journal on Computing*. 7 (3): 298–319. doi:10.1137/0207026.
   - Detailed analysis of binomial heap operations

2. **Cormen, Thomas H.; Leiserson, Charles E.; Rivest, Ronald L.; Stein, Clifford** (2009). "Introduction to Algorithms" (3rd ed.). MIT Press. Chapter 19: "Binomial Heaps".
   - Standard textbook presentation

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `insert` | **O(log n)** worst-case | Single-element merge |
| `find_min` | **O(log n)** worst-case | Must scan all roots |
| `delete_min` | **O(log n)** worst-case | Root removal + merge |
| `decrease_key` | **O(log n)** worst-case | Bubble up in tree |
| `merge` | **O(log n)** worst-case | Merge trees by degree |

**Note**: Unlike Fibonacci heaps, these are **worst-case** bounds, not amortized. However, merge is O(1) **amortized** if merging many heaps sequentially.

## How It Works

### Binomial Trees

A **binomial tree Bₖ** of order k is defined recursively:
- B₀ is a single node
- Bₖ is formed by linking two B_{k-1} trees, making one root a child of the other

**Properties**:
- Bₖ has exactly **2ᵏ nodes**
- Bₖ has height **k**
- Bₖ has **k children** at the root
- Root has degrees 0, 1, ..., k-1

These properties are crucial for the efficiency of binomial heaps.

### Data Structure

A binomial heap is a **collection of binomial trees** where:
- Each tree satisfies the **min-heap property**
- There is **at most one tree of each degree** (0, 1, 2, ...)
- Roots are stored in a linked list (sorted by degree)
- The minimum root is tracked

This is like a binary representation: if n = 2ᵏ₁ + 2ᵏ₂ + ... (binary), then the heap contains trees Bₖ₁, Bₖ₂, ...

### Key Operations

#### Insert (O(log n) worst-case)

1. Create a new B₀ tree (single node)
2. **Merge** it into the heap

The merge operation links trees of the same degree, similar to binary addition with carry propagation.

#### Merge (O(log n) worst-case, O(1) amortized)

This is the core operation. Merging two binomial heaps is analogous to binary addition:

1. Combine the root lists (like adding two binary numbers)
2. For each degree from 0 to max:
   - If 0 trees at this degree: store tree if present
   - If 1 tree at this degree: store it
   - If 2 trees at this degree: **link** them, produce 1 tree of degree+1 (carry)
   - If 3 trees: store 1, link 2 to produce carry (like 1+1+1 = 3, write 1, carry 1)

**Example**: Merging heaps with trees of degrees [0, 2] and [0, 1]:
- Degree 0: two trees → link to degree 1 (carry)
- Degree 1: one tree + carry → link to degree 2 (carry)
- Degree 2: one tree + carry → link to degree 3
- Result: tree of degree 3

**Why O(1) amortized?** Over a sequence of insertions, the "carry propagation" cost amortizes out.

#### Link Trees

Linking two binomial trees of the same degree:
1. Compare root priorities
2. Make the larger-priority root a child of the smaller-priority root
3. The resulting tree has degree one higher

This maintains the heap property and binomial tree structure.

#### Delete-min (O(log n) worst-case)

1. Find minimum root (O(log n) - scan root list)
2. Remove it from root list
3. Add its children to the heap (each child is a binomial tree)
4. Merge the children back into the heap

**Why O(log n)?** The minimum root has at most O(log n) children (degree ≤ log n).

#### Decrease-key (O(log n) worst-case)

1. Decrease the priority value
2. **Bubble up** the node (swap with parent if heap property violated)
3. Continue bubbling up until heap property satisfied

**Why O(log n)?** The tree height is O(log n), and we may need to traverse from leaf to root.

This is the key difference from Fibonacci heaps: no cutting, just bubble up.

## Comparison to Fibonacci Heaps

| Feature | Binomial Heap | Fibonacci Heap |
|---------|--------------|----------------|
| Insert | O(log n) worst | O(1) amortized |
| Delete-min | O(log n) worst | O(log n) amortized |
| Decrease-key | **O(log n) worst** | **O(1) amortized** |
| Merge | O(log n) worst | O(1) amortized |
| Complexity | Simple | Complex |
| Bounds | Worst-case | Amortized |

**When to use Binomial Heaps:**
- Need worst-case guarantees (not just amortized)
- Few decrease-key operations
- Simpler code is preferred
- Educational purposes

## Implementation Details

The Rust implementation:
- Stores trees in a vector indexed by degree
- Uses sibling pointers to link roots
- Tracks minimum separately for O(1) find-min (after update)
- Implements careful tree linking to maintain binomial structure

## Applications

Binomial heaps are used when:
1. Worst-case guarantees are needed
2. Merge operations are frequent
3. Simplicity is valued over theoretical optimality

## References

1. Vuillemin, J. (1978). A data structure for manipulating priority queues. *Communications of the ACM*, 21(4), 309-315.

2. Brown, M. R. (1978). Implementation and analysis of binomial queue algorithms. *SIAM Journal on Computing*, 7(3), 298-319.

3. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

