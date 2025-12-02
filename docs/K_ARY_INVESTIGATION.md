# Investigation: Parametrizing Heaps for K-Ary Trees

## Overview

This document investigates whether the heap implementations in this
codebase can be parametrized to use k-ary trees instead of binary trees
(where applicable).

## Current Tree Structures

### 1. Binomial Heap (`src/binomial.rs`)

- **Tree Structure**: Binomial trees (binary trees)
- **Node Structure**:
  - `child: Option<NonNull<Node<T, P>>>` - single child pointer
  - `sibling: Option<NonNull<Node<T, P>>>` - linked list of siblings
  - `degree: usize` - number of children
- **Key Operations**:
  - `link_trees()`: Links two trees of the same degree (assumes binary)
  - Merging uses degree-based consolidation
- **K-Ary Feasibility**: ⚠️ **MODERATE**
  - Binomial trees are inherently binary (2^k nodes at level k)
  - Could generalize to "k-ary binomial trees" where each node has up to k children
  - Would need to change `child` to `Vec<Option<NonNull<Node<T, P>>>>` or array
  - Linking algorithm would need to accept up to k children instead of exactly 2

### 2. Skew Binomial Heap (`src/skew_binomial.rs`)

- **Tree Structure**: Skew binomial trees (binary with skew flags)
- **Node Structure**: Similar to binomial (child/sibling)
- **K-Ary Feasibility**: ⚠️ **MODERATE**
  - Similar considerations as binomial heap
  - Skew operations might be more complex with k-ary structure

### 3. Fibonacci Heap (`src/fibonacci.rs`)

- **Tree Structure**: Arbitrary trees with circular doubly-linked child lists
- **Node Structure**:
  - `child: Option<NonNull<Node<T, P>>>` - pointer to first child
  - `left/right: NonNull<Node<T, P>>` - circular doubly linked list
  - `degree: usize` - number of children (unbounded)
- **Key Operations**:
  - `link()`: Links y as child of x (no limit on children)
  - Consolidation links trees of the same degree
- **K-Ary Feasibility**: ❌ **NOT RECOMMENDED**
  - Already supports unbounded multi-way trees
  - Adding arbitrary k limit would degrade performance
  - Unbounded degree is key to amortized O(1) operations

### 4. Pairing Heap (`src/pairing.rs`)

- **Tree Structure**: General tree with child/sibling pointers
- **Node Structure**:
  - `child: Option<NonNull<Node<T, P>>>` - first child
  - `sibling: Option<NonNull<Node<T, P>>>` - next sibling
  - Unbounded children
- **K-Ary Feasibility**: ❌ **NOT RECOMMENDED**
  - Already supports unbounded multi-way trees
  - Unbounded children enable O(1) amortized insert/merge
  - Adding k limit would defeat the purpose

### 5. Rank Pairing Heap (`src/rank_pairing.rs`)

- **Tree Structure**: General tree with rank constraints
- **Node Structure**: child/sibling like pairing heap
- **K-Ary Feasibility**: ❌ **NOT RECOMMENDED**
  - Already supports unbounded multi-way trees
  - Rank constraints work naturally with unbounded degree
  - No benefit to arbitrary k limit

### 6. Strict Fibonacci Heap (`src/strict_fibonacci.rs`)

- **Tree Structure**: Similar to Fibonacci heap
- **K-Ary Feasibility**: ❌ **NOT RECOMMENDED**
  - Similar considerations as Fibonacci heap
  - Unbounded degree is critical for worst-case bounds

### 7. Two-Three Heap (`src/twothree.rs`)

- **Tree Structure**: **Already k-ary!** (2-3 children per node)
- **Node Structure**:
  - `children: Vec<Option<NonNull<Node<T, P>>>>` - array of children
- **K-Ary Feasibility**: ✅ **ALREADY IMPLEMENTED**
  - This is the model implementation
  - Uses vector for children (currently 2-3)
  - `maintain_structure()` splits nodes with >3 children

## Design Patterns for K-Ary Parametrization

### Pattern 1: Array-Based Children (Like Two-Three Heap)

```rust
struct Node<T, P, const K: usize> {
    item: T,
    priority: P,
    parent: Option<NonNull<Node<T, P, K>>>,
    children: [Option<NonNull<Node<T, P, K>>>; K],
    // ... other fields
}
```

**Pros**: Type-safe at compile time, fixed size
**Cons**: Requires const generics (Rust 1.51+), less flexible

### Pattern 2: Vector-Based Children

```rust
struct Node<T, P> {
    // ...
    children: Vec<Option<NonNull<Node<T, P>>>>,
    max_children: usize, // k value
}
```

**Pros**: Flexible, can change k at runtime
**Cons**: Less type-safe, dynamic allocation

### Pattern 3: Hybrid (Keep Sibling List, Track Count)

```rust
struct Node<T, P> {
    // ...
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
    child_count: usize, // Track actual number
    max_children: usize, // k value
}
```

**Pros**: Minimal changes to existing code
**Cons**: Need to maintain counts, potential for inconsistency

## Implementation Strategy

### Phase 1: Choose Common Pattern

- Decide on array vs vector vs hybrid approach
- Consider const generics for compile-time k vs runtime k

### Phase 2: Start with Simplest Heap

- **Recommendation**: Start with Pairing Heap
  - Already has simple tree structure
  - Fewer invariants to maintain
  - Good testbed for k-ary operations

### Phase 3: Modify Operations

For each heap, key operations that need modification:

1. **Insert/Link Operations**:
   - Check if parent already has k children
   - If yes, need to split or promote node

2. **Merge Operations**:
   - Ensure result tree respects k-ary constraint

3. **Delete Operations**:
   - When removing min, children may need reorganization
   - Ensure all nodes have ≤ k children

4. **Decrease Key**:
   - Cuts should respect k-ary constraints
   - Parent nodes may need reorganization

### Phase 4: Mathematical Considerations

#### Binomial Heaps

- Binomial trees have specific structure: tree of degree d has 2^d nodes
- For k-ary: tree of degree d could have k^d nodes
- Would need "k-ary binomial tree" definition

#### Fibonacci Heaps

- Current structure already flexible
- K-ary constraint: `degree <= k`
- Consolidation: need to handle when degree would exceed k

#### Rank-Pairing Heaps

- Rank constraints: `r(v) <= r(w1) + 1, r(v) <= r(w2) + 1`
- K-ary generalization: `r(v) <= min(r(w_i)) + 1` for k children

## Challenges

### 1. Splitting Nodes

When a node would have k+1 children, need splitting strategy:

- Create new sibling node?
- Promote to parent?
- Rebalance subtree?

### 2. Complexity Analysis

- K-ary trees typically have height O(log_k n) instead of O(log_2 n)
- Operations might become faster (fewer levels)
- But more work per level (more children to check)

### 3. Memory Overhead

- Array-based: Allocated space even for unused children
- Vector-based: Dynamic allocation overhead

### 4. Existing Algorithms

- Many heap algorithms assume binary trees
- Need to verify k-ary generalizations preserve invariants

## Recommendations

### Immediate Actions

1. **Create generic parameter**: Add `K: usize` const parameter or runtime `k` field
2. **Start with Pairing Heap**: Simplest to modify
3. **Use Two-Three Heap as reference**: Already has k-ary structure
4. **Add splitting logic**: When children exceed k, split or promote

### Performance Optimization Opportunity

- **SmallVec consideration**: TwoThreeHeap uses
  `Vec<Option<NonNull<...>>>` for children (2-3 typical)
- Nodes usually have 2-3 children, rarely more
- `SmallVec<[Option<NonNull<Node>>; 3]>` could eliminate
  allocations for typical nodes
- Trade-off: larger node size vs. allocation overhead
- Recommendation: Benchmark both approaches before optimizing

### Testing Strategy

1. Test with k=2 (should behave like binary)
2. Test with k=3 (compare with Two-Three heap)
3. Test with k=4, 5, etc.
4. Verify complexity bounds still hold
5. Property-based tests for all k values

### Documentation Needs

1. Define "k-ary binomial tree", "k-ary Fibonacci heap", etc.
2. Document complexity changes
3. Explain splitting strategies

## Code Examples

### Example 1: Binomial Heap with K-Ary Trees (Const Generic)

**Current Binary Structure:**

```rust
struct Node<T, P> {
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
    degree: usize,
    // ...
}

unsafe fn link_trees(&self, a: NonNull<Node<T, P>>, b: NonNull<Node<T, P>>) {
    let a_child = (*a.as_ptr()).child;
    (*b.as_ptr()).parent = Some(a);
    (*b.as_ptr()).sibling = a_child;
    (*a.as_ptr()).child = Some(b);
    (*a.as_ptr()).degree += 1;
}
```

**Proposed K-Ary Structure (Array-Based):**

```rust
struct Node<T, P, const K: usize> {
    parent: Option<NonNull<Node<T, P, K>>>,
    children: [Option<NonNull<Node<T, P, K>>>; K],
    num_children: usize, // Actual number of children (0..=K)
    degree: usize,      // Tree degree for binomial property
    // ...
}

unsafe fn link_trees(
    &self,
    a: NonNull<Node<T, P, K>>,
    b: NonNull<Node<T, P, K>>
) -> Result<NonNull<Node<T, P, K>>, NonNull<Node<T, P, K>>> {
    // Check if a has space for another child
    if (*a.as_ptr()).num_children >= K {
        return Err(b); // Would need splitting
    }

    // Find first empty slot
    for i in 0..K {
        if (*a.as_ptr()).children[i].is_none() {
            (*a.as_ptr()).children[i] = Some(b);
            (*b.as_ptr()).parent = Some(a);
            (*a.as_ptr()).num_children += 1;
            (*a.as_ptr()).degree += 1;
            return Ok(a);
        }
    }
    unreachable!()
}
```

**K-Ary Structure (Vector-Based):**

```rust
struct Node<T, P> {
    parent: Option<NonNull<Node<T, P>>>,
    children: Vec<Option<NonNull<Node<T, P>>>>,
    max_children: usize, // K value
    // ...
}

unsafe fn link_trees(
    &self,
    a: NonNull<Node<T, P>>,
    b: NonNull<Node<T, P>>
) -> Result<NonNull<Node<T, P>>, NonNull<Node<T, P>>> {
    let a_ptr = a.as_ptr();
    let max = (*a_ptr).max_children;

    if (*a_ptr).children.len() >= max {
        // Need splitting strategy
        return Err(b);
    }

    (*a_ptr).children.push(Some(b));
    (*b.as_ptr()).parent = Some(a);
    (*a_ptr).degree += 1;
    Ok(a)
}
```

### Example 2: Fibonacci Heap with K-Ary Constraint

**Current Structure (Unbounded):**

```rust
unsafe fn link(&mut self, y: NonNull<Node<T, P>>, x: NonNull<Node<T, P>>) {
    // Add y to x's child list (circular doubly linked)
    (*y.as_ptr()).parent = Some(x);

    if let Some(x_child) = (*x.as_ptr()).child {
        // Add to circular list
        let x_child_left = (*x_child.as_ptr()).left;
        (*y.as_ptr()).right = x_child;
        (*y.as_ptr()).left = x_child_left;
        (*x_child_left.as_ptr()).right = y;
        (*x_child.as_ptr()).left = y;
    } else {
        (*x.as_ptr()).child = Some(y);
        (*y.as_ptr()).left = y;
        (*y.as_ptr()).right = y;
    }

    (*x.as_ptr()).degree += 1; // Unbounded
}
```

**K-Ary Modification:**

```rust
unsafe fn link(&mut self, y: NonNull<Node<T, P>>, x: NonNull<Node<T, P>>) {
    let x_ptr = x.as_ptr();
    let max_children = self.max_children; // K value stored in heap

    if (*x_ptr).degree >= max_children {
        // Need to split or promote
        // Option 1: Split x, creating new root
        // Option 2: Promote y to root list
        self.split_node(x);
    }

    // Add y to x's child list
    (*y.as_ptr()).parent = Some(x);

    if let Some(x_child) = (*x_ptr).child {
        let x_child_left = (*x_child.as_ptr()).left;
        (*y.as_ptr()).right = x_child;
        (*y.as_ptr()).left = x_child_left;
        (*x_child_left.as_ptr()).right = y;
        (*x_child.as_ptr()).left = y;
    } else {
        (*x_ptr).child = Some(y);
        (*y.as_ptr()).left = y;
        (*y.as_ptr()).right = y;
    }

    (*x_ptr).degree += 1;
}
```

### Example 3: Pairing Heap with K-Ary Constraint

**Current Structure:**

```rust
struct Node<T, P> {
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
    // ...
}

unsafe fn merge_nodes(&self, a: NonNull<Node<T, P>>, b: NonNull<Node<T, P>>) -> NonNull<Node<T, P>> {
    if (*a.as_ptr()).priority < (*b.as_ptr()).priority {
        let a_child = (*a.as_ptr()).child;
        (*b.as_ptr()).sibling = a_child;
        (*b.as_ptr()).prev = Some(a);
        (*a.as_ptr()).child = Some(b);
        a
    } else {
        // ... similar for b
    }
}
```

**K-Ary Modification (Hybrid Approach):**

```rust
struct Node<T, P> {
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
    child_count: usize,
    max_children: usize, // K value
    // ...
}

unsafe fn merge_nodes(
    &self,
    a: NonNull<Node<T, P>>,
    b: NonNull<Node<T, P>>
) -> Result<NonNull<Node<T, P>>, SplitAction> {
    let a_ptr = a.as_ptr();
    let max = (*a_ptr).max_children;

    if (*a_ptr).child_count >= max {
        // Would exceed k-ary constraint
        // Need splitting strategy
        return Err(SplitAction::Promote(b));
    }

    if (*a_ptr).priority < (*b.as_ptr()).priority {
        let a_child = (*a_ptr).child;
        (*b.as_ptr()).sibling = a_child;
        (*b.as_ptr()).prev = Some(a);
        (*a_ptr).child = Some(b);
        (*a_ptr).child_count += 1;
        Ok(a)
    } else {
        // ... similar for b
    }
}
```

### Example 4: Using Two-Three Heap as Template

The `TwoThreeHeap` already implements k-ary (specifically 2-3 ary) structure:

```rust
struct Node<T, P> {
    children: Vec<Option<NonNull<Node<T, P>>>>, // Already vector-based!
    // ...
}

unsafe fn maintain_structure(&mut self, node: NonNull<Node<T, P>>) {
    let num_children = (*node.as_ptr()).children.iter()
        .filter(|c| c.is_some()).count();

    if num_children > 3 {
        // Split: keep first 2, move last 2 to new node
        // This pattern could be generalized to K
    }
}
```

**Generalization to K-Ary:**

```rust
struct Node<T, P> {
    children: Vec<Option<NonNull<Node<T, P>>>>,
    max_children: usize, // K value
    // ...
}

unsafe fn maintain_structure(&mut self, node: NonNull<Node<T, P>>) {
    let node_ptr = node.as_ptr();
    let max = (*node_ptr).max_children;
    let num_children = (*node_ptr).children.iter()
        .filter(|c| c.is_some()).count();

    if num_children > max {
        // Split strategy: keep first k/2, move rest to new node
        // Could be k-1 and 1, or balanced split
        let split_point = max / 2 + 1;
        // ... split logic
    }
}
```

## Findings from Investigation

### Key Discovery: Most Heaps are Already Multi-Way Trees

After analyzing all heap implementations, we found:

- **PairingHeap, FibonacciHeap, RankPairingHeap**: Already use
  unbounded multi-way trees with child/sibling pointers
- **BinomialHeap, SkewBinomialHeap**: Use child/sibling linked
  lists (effectively multi-way, not binary)
- **TwoThreeHeap**: The ONLY heap with fixed arity (2-3 children
  per node)

### K-Ary Parametrization Assessment

**Reality Check**: The investigation document's assumption that
heaps are "binary" is **incorrect**. Most heaps are already
multi-way trees. The relevant question is whether to constrain
them with a maximum degree K.

#### Option 1: Const Generics for Two-Three Heap

- **Feasibility**: ✅ Highly feasible
- **Why**: Two-Three heap already uses
  `Vec<Option<NonNull<Node>>>` for children
- **Implementation**: Add `const K: usize` parameter, constrain
  `children` vector length
- **Impact on Original Binary**: N/A (it's not binary)
- **Recommendation**: **Best candidate for k-ary generalization**

#### Option 2: Add Runtime K Constraint to Multi-Way Heaps

- **Feasibility**: ⚠️ Complex, questionable benefit
- **Why**: Would require adding splitting logic to heaps that
  deliberately allow unbounded children
- **Trade-off**: Randomly constraining multi-way heaps goes
  against their design philosophy
- **Impact**: Would degrade original implementations
- **Recommendation**: **Not recommended**

### Conclusion

The investigation reveals that **most heaps are already "k-ary"
with k=∞**. The real opportunity is:

1. **Generalize TwoThreeHeap** to support arbitrary k (not just
   2-3)
2. **Document** that other heaps are multi-way, not binary
3. **Compare performance** of different k values for the
   generalized Two-Three structure
4. **Consider SmallVec** for fixed-arity heaps to avoid
   allocations for typical nodes

## Additional Optimization Consideration

### SmallVec for Fixed-Arity Heaps

For heaps with fixed arity like TwoThreeHeap:

- Currently uses `Vec<Option<NonNull<Node>>>` which allocates on
  heap for every node
- Typical nodes have 2-3 children (very predictable)
- `SmallVec<[Option<NonNull<Node>>; 3]>` could eliminate
  allocations in 99%+ of cases
- Trade-offs:
  - **Pros**: No allocations for typical nodes, better cache
    locality for small heaps
  - **Cons**: Larger node size (24 bytes overhead vs ~24 bytes for
    Vec capacity), dependency on `smallvec` crate
- **Recommendation**: Only worth pursuing if benchmarks show
  allocation overhead is significant

## Next Steps

1. ✅ Create investigation branch and worktree
2. ✅ Analyze all heap structures and identify k-ary opportunities
3. 🔄 **IMPORTANT**: Choose approach - generalize TwoThreeHeap vs new implementation
4. ⏳ Implement generalized TwoThreeHeap with configurable k
5. ⏳ Compare performance with different k values (2, 3, 4, 5, etc.)
6. ⏳ Consider renaming TwoThreeHeap to KAryHeap or similar
7. ⏳ Benchmark SmallVec vs Vec for TwoThreeHeap children storage

**Revised Recommendation**: Start with **TwoThreeHeap
generalization** using const generics

**UPDATE**: SmallVec has been integrated into multiple heaps:

- **TwoThreeHeap**: `SmallVec<[Option<NonNull<Node>>; 4]>` for children storage
- **BinomialHeap**: `SmallVec<[Option<NonNull<Node>>; 32]>` for trees array
- **SkewBinomialHeap**: `SmallVec<[Option<NonNull<Node>>; 32]>`
  for trees array

All implementations eliminate heap allocations for typical small
heaps while maintaining the same API and functionality. All tests
pass.

### Additional SmallVec Candidates

Beyond TwoThreeHeap, other heaps have small predictable Vec sizes:

1. **BinomialHeap & SkewBinomialHeap**: ✅ **IMPLEMENTED**
   - Size: log₂(n) entries typically (0..30 entries for n < 1 billion)
   - Now use `SmallVec<[Option<NonNull<Node>>; 32]>` to cover heaps up to 2³² elements
   - **Benefits**: Eliminates allocations for heaps up to 4 billion elements
   - All tests pass

2. **General Recommendation**: ✅ All promising candidates now
   implemented. The smallvec dependency is already added for all heaps.
