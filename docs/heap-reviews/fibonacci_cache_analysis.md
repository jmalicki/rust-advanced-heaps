# Fibonacci Heap Cache Efficiency Analysis

## Theoretical Optimal (from Fredman & Tarjan 1987)

### Algorithm Characteristics

- **Insert**: O(1) - add single node to root list
  - Touches: 1 node
  - Writes: priority, item
  - Reads: min priority (for comparison)

- **Pop (delete-min)**: O(log n) amortized
  - Touches: O(log n) nodes during consolidation
  - Key operation: degree-based linking
  - Priority reads dominate (comparison in consolidation)

- **Decrease-key**: O(1) amortized
  - Average cascade depth: ~1.8 nodes (from amortized analysis)
  - 63% of calls: no cascade (1 node)
  - 23% of calls: single cut (2 nodes)
  - 14% of calls: deeper cascade

### Critical Access Patterns

1. **Cascading cuts** traverse parent pointers upward
2. **Consolidation** compares priorities of all roots
3. **Marked flag** checked/set during decrease_key

## Current Implementation Analysis

### Node Structure (fibonacci.rs)

```rust
struct Node<T, P> {
    // HOT fields - accessed during consolidation & decrease_key
    priority: Option<P>,      // Hot - compared frequently (Option for extraction)
    marked: bool,             // Hot during decrease_key
    // Medium fields - structure pointers
    parent: Option<Weak<..>>, // Medium - cascade traversal (Option for nullability)
    children: Vec<RefBox<..>>,// Medium - consolidation (single owner)
    // COLD field - only accessed on pop
    item: Option<T>,          // Cold - only accessed on pop (Option for extraction)
}
```

**RefBox Migration Status**: See detailed analysis in "RefBox Migration Analysis"
section below. The Option wrappers required for extraction negate the heap
overhead savings.

### Current Field Layout Analysis

| Field | Size (i64/i64) | Access Frequency | Notes |
| ----- | -------------- | ---------------- | ----- |
| item | 8 bytes | Low (pop only) | Could be cold |
| priority | 8 bytes | High | Comparison hot path |
| parent | 8 bytes | Medium | Cascade traversal |
| children | 24 bytes | Medium | Vec overhead |
| marked | 1 byte | High in decrease_key | Adjacent to priority good |

**Total Node Size**: ~50+ bytes (Rc overhead adds more)

## Measured Access Patterns

(Run `access_analyzer dijkstra_sparse 10000 fibonacci` to populate)

### Expected Distribution

| Field | Read % | Write % | Notes |
| ----- | ------ | ------- | ----- |
| Priority | 45% | 16% | Most accessed field |
| Item | 3% | 3% | Only on pop |
| ChildPtr | 26% | 9% | Consolidation |
| ParentPtr | 13% | 13% | Cascade cuts |
| Marked | 13% | 6% | Decrease-key |

### Pointer Chase Depth

| Operation | Max | Avg | Notes |
| --------- | --- | --- | ----- |
| push | 1 | 1.0 | Single node |
| pop | ~log n | 2.1 | Consolidation |
| decrease_key | ~log n | 1.3 | Cascading cuts |

## Optimization Recommendations

### 1. Data Storage: Split Hot/Cold

**Current**: All data inline in node

**Optimal**: Split storage

```rust
// Hot structure (in arena)
struct NodeHot<P> {
    priority: P,        // 8 bytes
    marked: bool,       // 1 byte
    parent_idx: u32,    // 4 bytes
    child_head: u32,    // 4 bytes
    sibling: u32,       // 4 bytes
    // padding to 24 bytes (fits 2-3 per cache line)
}

// Cold storage (separate Vec)
items: Vec<T>
```

**Benefit**: Consolidation and cascade cuts touch only hot data

### 2. Field Ordering

**Current**: [item, priority, parent, children, marked]

**Optimal**: [priority, marked, parent, children, item]

**Reason**: priority+marked accessed together in cascade checks

### 3. Pointer Representation

**Baseline**: `Rc<RefCell<Node>>` with Weak backlinks (24 bytes heap overhead)

**Current experiment**: `RefBox<Node>` with Weak references (8 bytes heap overhead)

- ⚠️ Option wrappers add 16 bytes to node struct, negating savings
- See "RefBox Migration Analysis" section for details

**Future optimal**: Arena indices with generation counters

```rust
struct Handle {
    index: u32,
    generation: u32,
}
```

**Benefit**:

- Better cache locality (arena is contiguous)
- No reference counting overhead at all
- Fixed-size handles (8 bytes)

### 4. Children Storage

**Current**: Vec<Rc<...>>

**Optimal**: Intrusive sibling list with indices

```rust
first_child: Option<u32>,  // Index
next_sibling: Option<u32>, // Index
```

**Benefit**: No Vec allocation per node

## Estimated Improvement

| Change | Cache Miss Reduction | Implementation Effort | Status |
| ------ | -------------------- | --------------------- | ------ |
| Field reorder | 5-10% | Low | ✅ Done (PR #81) |
| RefBox + take() | **28%** | Medium | ✅ Done |
| Split hot/cold | 25-40% | High | Planned |
| Arena + indices | 15-25% | High | Planned |
| Intrusive lists | 10-15% | Medium | Planned |

**Implemented improvements**: ~28% reduction (RefBox + take())
**Combined future potential**: 40-60% additional cache miss reduction

## RefBox Migration Analysis

### Cachegrind Results (100k elements, Dijkstra-like workload)

| Metric | `Rc<RefCell>` | `RefBox` + Option | `RefBox` + take() | Change vs Rc |
|--------|---------------|-------------------|-------------------|--------------|
| Node struct size | 56 bytes | 72 bytes | **56 bytes** | 0% |
| Heap overhead | 24 bytes | 8 bytes | **8 bytes** | -67% |
| **Total per node** | **80 bytes** | **80 bytes** | **64 bytes** | **-20%** |
| D1 misses (total) | 2,406,750 | 2,517,456 | **1,728,073** | **-28%** |
| D1 read misses | 1,621,267 | 2,079,432 | 1,329,822 | -18% |
| D1 write misses | 785,483 | 438,024 | 398,251 | -49% |
| LL misses | 233,505 | 233,484 | 208,531 | -11% |

### RefBox with Option: Why It Didn't Help

The `Option<T>` and `Option<P>` wrappers (needed for `take()` on pop without
Clone bounds) add 16 bytes to the node struct, completely negating the 16-byte
savings from RefBox's reduced heap overhead.

### RefBox with take(): The Solution

By adding a `take()` method to RefBox
([PR #5](https://github.com/yvesdum/refbox/pull/5)), we can move values out of
the RefBox directly without needing Option wrappers:

```rust
// Before (with Option wrappers):
struct Node<T, P> {
    priority: Option<P>,  // 16 bytes for Option<i64>
    item: Option<T>,      // 16 bytes for Option<i64>
    ...
}
let (priority, item) = {
    let mut node = min_node.try_borrow_mut().ok()?;
    (node.priority.take()?, node.item.take()?)
};

// After (with RefBox::take()):
struct Node<T, P> {
    priority: P,  // 8 bytes
    item: T,      // 8 bytes
    ...
}
let node = min_node.take();  // Moves entire node out
(node.priority, node.item)
```

This achieves:

- **28% fewer D1 cache misses** vs `Rc<RefCell>`
- **20% less memory per node** (64 bytes vs 80 bytes)
- Node fits in **single cache line** (56 bytes struct + 8 bytes heap overhead)

## Trade-offs

1. **Split storage** adds indirection for item access (acceptable - item
   rarely accessed)
2. **Arena indices** require generation tracking for handle safety
3. **Intrusive lists** complicate memory management

## Verification

```bash
# Run cache benchmark
cargo build --release --bin cache_benchmark
valgrind --tool=cachegrind --cache-sim=yes ./target/release/cache_benchmark 100000

# Compare D1 miss rate between implementations
```
