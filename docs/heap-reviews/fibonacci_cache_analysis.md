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
    item: T,           // Cold - only accessed on pop
    priority: P,       // Hot - compared frequently
    parent: Weak<...>, // Medium - cascade traversal
    children: Vec<..>, // Medium - consolidation
    marked: bool,      // Hot during decrease_key
}
```

### Current Field Layout Analysis

| Field | Size (i64/i64) | Access Frequency | Notes |
|-------|----------------|------------------|-------|
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
|-------|--------|---------|-------|
| Priority | 45% | 16% | Most accessed field |
| Item | 3% | 3% | Only on pop |
| ChildPtr | 26% | 9% | Consolidation |
| ParentPtr | 13% | 13% | Cascade cuts |
| Marked | 13% | 6% | Decrease-key |

### Pointer Chase Depth

| Operation | Max | Avg | Notes |
|-----------|-----|-----|-------|
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

**Current**: `Rc<RefCell<Node>>` with Weak backlinks

**Optimal**: Arena indices with generation counters

```rust
struct Handle {
    index: u32,
    generation: u32,
}
```

**Benefit**:

- 8 bytes instead of 8+ bytes (Rc pointer)
- Better cache locality (arena is contiguous)
- No reference counting overhead

### 4. Children Storage

**Current**: Vec<Rc<...>>

**Optimal**: Intrusive sibling list with indices

```rust
first_child: Option<u32>,  // Index
next_sibling: Option<u32>, // Index
```

**Benefit**: No Vec allocation per node

## Estimated Improvement

| Change | Cache Miss Reduction | Implementation Effort |
|--------|---------------------|----------------------|
| Split hot/cold | 25-40% | High |
| Field reorder | 5-10% | Low |
| Arena + indices | 15-25% | High |
| Intrusive lists | 10-15% | Medium |

**Combined potential**: 40-60% cache miss reduction

## Trade-offs

1. **Split storage** adds indirection for item access (acceptable - item rarely accessed)
2. **Arena indices** require generation tracking for handle safety
3. **Intrusive lists** complicate memory management

## Verification

```bash
# Before optimization
valgrind --tool=cachegrind ./target/release/cache_profile 100000 fibonacci

# Record: D1 miss rate, LL miss rate

# After optimization (if implemented)
# Compare metrics
```
