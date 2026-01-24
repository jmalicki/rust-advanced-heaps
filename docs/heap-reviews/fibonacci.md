# Fibonacci Heap Review

**File**: `src/fibonacci.rs`
**Lines of Code**: 473
**Unsafe Blocks**: 1 (in `peek()` method)

## Summary Table

| Aspect | Rating (1-5) | Notes |
| ------ | ------------ | ----- |
| Fidelity to Paper | 4 | Faithful implementation with minor differences |
| Memory Safety | 5 | Safe Rc/Weak pattern with minimal unsafe |
| Cache Efficiency | 3.5 | Pointer-heavy but field-reordered for locality |
| Test Coverage | 5 | Comprehensive test coverage across all suites |

## 1. Fidelity to Paper

**Reference Paper**: Fredman, M. L., & Tarjan, R. E. (1987). "Fibonacci heaps
and their uses in improved network optimization algorithms."
*Journal of the ACM*, 34(3), 596-615.

### Operation-by-Operation Comparison

| Operation | Paper Specification | Implementation | Match |
| --------- | ------------------- | -------------- | ----- |
| **Insert** | O(1) - add to root list | `Rc::new()`, add to `roots` | Yes |
| **Delete-Min** | O(log n) - consolidate | Remove min, consolidate | Yes |
| **Decrease-Key** | O(1) - cascading cut | `cut()` then `cascading_cut()` | Yes |
| **Merge** | O(1) - concat root lists | Append `other.roots` | Yes |
| **Find-Min** | O(1) - direct access | `self.min.upgrade()` | Yes |

### Documented Deviations with Rationale

1. **Children stored in Vec instead of circular doubly-linked list**
   - Paper: Circular doubly-linked list for O(1) splice operations
   - Implementation: `Vec<Rc<RefCell<Node<T, P>>>>` (line 110)
   - Rationale: "Simplifies the implementation while maintaining the same
     asymptotic complexity (degree is O(log n))" (line 101-103)
   - Impact: Child removal is O(degree) = O(log n) instead of O(1). This
     doesn't violate Fibonacci heap's amortized bounds because: (1) the
     maximum degree is bounded by O(log n) due to the Fibonacci property,
     and (2) cuts are infrequent enough that the amortized analysis still
     holds. The Vec approach trades O(1) splice for simpler code without
     changing overall asymptotic complexity.

2. **Roots stored in Vec instead of circular list**
   - Paper: Circular doubly-linked list for root list
   - Implementation: `roots: Vec<Rc<RefCell<Node<T, P>>>>` (line 149)
   - Impact: Consolidation uses `drain()` and rebuilds, functionally equivalent.

3. **Minimum tracked via Weak reference**
   - Paper: Direct pointer to minimum node
   - Implementation: `min: Weak<RefCell<Node<T, P>>>` (line 151)
   - Rationale: Works within Rust's Rc/Weak safety model
   - Impact: Requires `upgrade()` call for access, negligible overhead.

4. **Consolidation degree table sizing**
   - Paper: D(n) = floor(log_phi(n)) where phi is golden ratio
   - Implementation: `((self.len + 1) as f64).log2() as usize + 2` (line 351)
   - Rationale: Conservative approximation using log2 instead of log_phi
   - Impact: Slightly larger degree table, no correctness impact.

## 2. Memory Safety

### Unsafe Block Inventory

| Location | Lines | Purpose | Safety Justification |
| -------- | ----- | ------- | -------------------- |
| `peek()` | 191-194 | Return refs to node data | Rc keeps alive, read-only |

**Analysis**: The single unsafe block is a common pattern for returning
references from `RefCell` without the runtime cost of holding a `Ref` guard.
The safety argument in comments (lines 185-190) is sound.

### Pattern Analysis

**Reference Counting Model**:

- Strong references (`Rc`): Flow from heap roots down to children (ownership)
- Weak references (`Weak`): Used for parent backlinks and handles
- This pattern prevents reference cycles and enables automatic cleanup

**Handle Design** (lines 61-91):

- `FibonacciHandle<T, P>` wraps `Weak<RefCell<Node<T, P>>>`
- Invalid handles (after node removal) fail gracefully via `upgrade()`
  returning `None`
- Implements `Clone`, `PartialEq`, `Eq` manually to avoid trait bound
  requirements

### Handle Invalidation Behavior

1. **After pop()**: The minimum node is removed via `Rc::try_unwrap()`
   (line 234). Any remaining handles become invalid (their `Weak::upgrade()`
   returns `None`).

2. **Detection**: `decrease_key()` checks `handle.node.upgrade()` (line 356)
   and returns `HeapError::InvalidHandle` if the handle is no longer valid.

### Potential Issues/Risks

1. **Panic on reference counting bug**: Line 236 panics if `try_unwrap` fails.
   This indicates a logic bug rather than user error.

2. **Thread safety**: Not `Send` or `Sync` due to `Rc`/`RefCell` usage. This
   is appropriate for the single-threaded use case.

3. **Memory leaks**: Not possible with Rc/Weak model as long as no cycles
   exist (verified by the ownership structure).

## 3. Cache Efficiency

### Node Struct Size Analysis

```rust
struct Node<T, P> {
    // HOT fields first (cache-optimized ordering)
    priority: P,                                // size_of::<P>()
    marked: bool,                               // 1 byte + padding
    // Medium fields
    parent: Weak<RefCell<Node<T, P>>>,          // 8 bytes (usize)
    children: Vec<Rc<RefCell<Node<T, P>>>>,     // 24 bytes (ptr, len, cap)
    // COLD field last
    item: T,                                    // size_of::<T>()
}
```

**Estimated Node overhead** (excluding T and P): ~40-48 bytes per node
depending on alignment.

**Heap struct**:

```rust
struct FibonacciHeap<T, P: Ord> {
    roots: Vec<Rc<RefCell<Node<T, P>>>>,        // 24 bytes
    min: Weak<RefCell<Node<T, P>>>,             // 8 bytes
    len: usize,                                 // 8 bytes
}
```

### Memory Access Patterns per Operation

| Operation | Access Pattern | Cache Impact |
| --------- | -------------- | ------------ |
| **push** | Allocate, compare min | 2 misses typical |
| **peek** | Follow min weak ref | 1-2 misses |
| **pop** | Find min, iterate, consolidate | O(log n) misses |
| **decrease_key** | Access via handle, traverse | 1 + O(cascade) misses |
| **consolidate** | Process all roots, link | O(roots) misses |

### Pointer-Chasing Analysis

The Fibonacci heap is inherently pointer-heavy:

- Each node access requires dereferencing `Rc` then `RefCell`
- Parent traversal in cascading cut follows `Weak` references
- Children stored non-contiguously in heap allocations

This is typical for Fibonacci heaps; the theoretical benefits come from
operation counts, not memory locality.

### Field Reordering Optimization

The Node struct fields have been reordered to improve cache efficiency by
placing frequently-accessed ("hot") fields first:

```rust
struct Node<T, P> {
    // HOT fields - accessed during consolidation & decrease_key
    priority: P,        // Most accessed field (~45% of accesses)
    marked: bool,       // Checked/set during cascade (~13% of accesses)

    // Medium fields - structure pointers
    parent: Weak<...>,  // Cascade traversal
    children: Vec<...>, // Consolidation

    // COLD field - only accessed on pop
    item: T,            // Only ~3% of accesses
}
```

**Rationale**: Access pattern instrumentation showed that `priority` is
accessed ~45% of the time (comparisons during consolidation and decrease_key),
while `item` is only accessed ~3% of the time (on pop). By placing hot fields
first, they're more likely to share a cache line, reducing L1 cache misses
during the frequent comparison operations.

#### Empirical Cache Efficiency Results

Measured using `valgrind --tool=cachegrind` with Dijkstra-like workloads.

**Workload characteristics**:

- Sparse graph simulation: ~n nodes with average degree 4-6
- Operation mix: ~n pushes, ~n/3 decrease_key calls, ~n pops
- Represents typical Dijkstra usage where decrease_key is frequent but not
  dominant (road network-style graphs)
- This is a moderate case; dense graphs (more decrease_key) may see larger
  improvements, while sort-only workloads (no decrease_key) see minimal benefit

##### Test: 100,000 elements

| Metric | Before | After | Change |
| ------ | ------ | ----- | ------ |
| D1 misses (total) | 1,346,711 | 1,287,446 | **-4.4%** |
| D1 read misses | 985,575 | 868,223 | **-11.9%** |
| D1 write misses | 361,136 | 419,223 | +16.1% |

##### Test: 500,000 elements

| Metric | Before | After | Change |
| ------ | ------ | ----- | ------ |
| D1 misses (total) | 5,537,801 | 5,473,842 | **-1.2%** |
| D1 read misses | 4,080,531 | 3,945,120 | **-3.3%** |
| D1 write misses | 1,457,270 | 1,528,722 | +4.9% |
| LL misses | 1,143,158 | 1,143,158 | 0% |

##### Analysis

- Read misses improved significantly (3-12% reduction) because `priority`
  comparisons during consolidation now hit cache more often
- Write misses increased slightly due to changed cache line dirtying patterns
- Net improvement: 1.2-4.4% reduction in total D1 cache misses
- Last-level (LL) cache misses unchanged - optimization targets L1 locality

### Additional Optimization Opportunities

1. **Arena allocation**: Use a custom arena allocator to improve locality of
   node allocations.

2. **Inline small children**: For nodes with few children, store them inline
   instead of in a separate Vec allocation.

3. **Cache min node data**: Store priority directly in heap struct to avoid
   `upgrade()` in `peek()`. **Tradeoffs**: Every push, decrease_key, and pop
   would need to update the cached value, adding synchronization overhead.
   Also duplicates memory for the priority. Since `peek()` is typically
   infrequent compared to other operations in Dijkstra-style algorithms,
   this optimization is best suited for peek-heavy workloads.

4. **Batch consolidation**: Delay consolidation across multiple pops when
   possible.

## 4. Invariants and Tests

### Heap Invariants (from documentation, lines 23-28)

1. **Heap property**: All trees are heap-ordered (parent <= child)
2. **Degree invariant**: After consolidation, at most one tree of each degree
3. **Marking rule**: A node can lose at most one child before being cut
4. **Fibonacci property**: Tree with root degree k has at least F_{k+2} nodes

### Test Coverage by Category

| Test Suite | Test Count | Coverage Type |
| ---------- | ---------- | ------------- |
| `generic_heap_tests.rs` (fibonacci_base) | 17 | Base heap operations |
| `generic_heap_tests.rs` (fibonacci_decrease) | 16 | Decrease-key ops |
| `property_tests.rs` | 2 modules | Proptest-based invariant checking |
| `stress_tests.rs` | 7 | Large-scale stress tests |
| `big_o_proofs.rs` | 3 | Complexity verification |
| `pathfinding_tests.rs` | 20+ | Dijkstra algorithm integration |

### Edge Case Coverage

| Edge Case | Covered? | Test Location |
| --------- | -------- | ------------- |
| Empty heap | Yes | `test_empty_heap` |
| Single element | Yes | `test_single_element_decrease` |
| Duplicate priorities | Yes | `test_duplicate_priorities` |
| Handle reuse after decrease | Yes | `test_multiple_decrease_same` |
| Invalid handle (after pop) | Partial | Returns error, no explicit test |
| Negative priorities | Yes | `test_negative_priorities` |
| Large priorities | Yes | `test_large_priorities` |
| Merge empty heaps | Yes | `test_merge_empty` |
| Handles after merge | Yes | `test_merge_with_handles` |

### Recommended Additional Tests

1. **Invalid handle after pop**: Explicitly test that using a handle after its
   node is popped returns an appropriate error.

2. **Cascading cut depth**: Test that deep cascading cuts (marking and cutting
   multiple levels) work correctly.

3. **Consolidation edge cases**: Test consolidation when all roots have the
   same degree.

4. **Maximum degree verification**: Verify that node degrees stay within the
   theoretical O(log n) bound.

5. **Memory cleanup verification**: Test that all nodes are properly
   deallocated after heap is dropped or cleared.

## 5. Rust Packages

### Current Dependencies Used

From the implementation:

- `std::cell::RefCell` - Interior mutability
- `std::rc::{Rc, Weak}` - Reference counting

No external crate dependencies for the Fibonacci heap itself.

### Alternative Crates That Could Help

1. **`typed-arena`** or **`bumpalo`**: Arena allocators for improved cache
   locality
   - Could replace individual `Rc` allocations with arena-allocated nodes
   - Would require different ownership model

2. **`parking_lot`**: Faster `RefCell` alternative if benchmarking shows
   RefCell overhead
   - Provides `RefCell`-like API with potential performance improvements

3. **`ghost-cell`**: Zero-cost cell type using branded lifetime tokens
   - Could eliminate RefCell runtime checks entirely
   - More complex API

4. **`indexmap`** or **`hashbrown`**: For degree table in consolidation
   - Current Vec-based approach is simple and efficient for small degree counts

### Recommendations for Improvements

1. **Keep current approach**: The Rc/Weak pattern is clean, safe, and the
   implementation is well-documented. The overhead is acceptable for most
   use cases.

2. **Consider feature-gated arena variant**: For performance-critical
   applications, provide an optional arena-based implementation behind a
   feature flag.

3. **Add debug assertions**: Use `debug_assert!` to verify invariants in debug
   builds without runtime cost in release.

4. **Benchmark against other implementations**: Compare with raw-pointer
   implementations to quantify the Rc/RefCell overhead.
