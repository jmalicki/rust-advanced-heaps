# Strict Fibonacci Heap

## Overview

The **Strict Fibonacci Heap** is a refinement of Fibonacci heaps that
achieves **worst-case** bounds instead of amortized ones. Like Brodal heaps,
it achieves O(1) worst-case insert, decrease_key, and merge; O(log n)
worst-case delete_min.

## Historical Context and Papers

### Original Paper

- **Brodal, Gerth Stølting; Lagogiannis, George; Tarjan, Robert E.** (2012).
  "Strict Fibonacci heaps". *Proceedings of the 44th Annual ACM Symposium on
  Theory of Computing (STOC)*. pp. 1177–1184. doi:10.1145/2213977.2214080.

### Key Follow-up Work

1. **Brodal, Gerth Stølting** (1996). "Worst-case efficient priority
   queues". *Proceedings of SODA* 1996, 52-58.
   - Brodal heaps also achieve worst-case bounds

2. **Fredman, Michael L.; Tarjan, Robert E.** (1987). "Fibonacci heaps and
   their uses in improved network optimization algorithms". *Journal of the
   ACM*, 34(3), 596-615.
   - Original Fibonacci heap paper

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `insert` | **O(1)** worst-case | Constant time guaranteed |
| `find_min` | **O(1)** worst-case | Direct pointer to minimum |
| `delete_min` | **O(log n)** worst-case | Worst-case logarithmic |
| `decrease_key` | **O(1)** worst-case | Constant time guaranteed |
| `merge` | **O(1)** worst-case | Constant time guaranteed |

**Key Achievement**: These are **worst-case** bounds, matching the
**amortized** bounds of standard Fibonacci heaps!

## How It Works

### Differences from Fibonacci Heaps

Standard Fibonacci heaps achieve **amortized** O(1) bounds using:

- **Lazy consolidation**: Defer work until delete_min
- **Cascading cuts**: Cut parents only when already marked
- **Potential function**: Charge cheap operations to pay for expensive ones

Strict Fibonacci heaps achieve **worst-case** bounds using:

- **Active consolidation**: Fix structure immediately after operations
- **Stricter invariants**: Maintain structure constraints at all times
- **Violation prevention**: Prevent violations rather than deferring repair

### Data Structure

A Strict Fibonacci heap maintains:

- **Active root list**: Roots that are "active" (recently modified)
- **Passive root list**: Roots that are "stable" (haven't been modified recently)
- **Consolidation tracking**: Know when consolidation is needed
- **Stricter degree constraints**: Maintain degree invariants more carefully

The active/passive distinction allows us to:

- Defer consolidation of stable structures
- Immediately fix recently modified structures
- Balance work across operations

### Key Operations

#### Insert (O(1) worst-case)

1. Create new single-node tree
2. Add to **active root list** (not passive - it's newly created)
3. Update minimum pointer
4. **Consolidate if needed**: Only if active roots violate degree constraints

The key is that consolidation is **conditional** - we only consolidate when
needed, and it's worst-case O(1) per violation.

#### Consolidate (Conditional, O(1) per violation)

1. Check if degree constraints are violated
2. If yes, link trees of the same degree
3. Move stable trees to passive list
4. Keep active trees in active list

The insight is that we only consolidate **when necessary**, not on every operation.

#### Delete-min (O(log n) worst-case)

1. Remove minimum root
2. Add its children to active root list
3. **Consolidate**: This is where accumulated work happens
4. Find new minimum (scan roots)
5. Move roots to passive list (they're now stable)

**Why O(log n)?**

- At most O(log n) roots after consolidation
- Consolidation links O(log n) trees
- Each link is O(1)
- Total: O(log n)

#### Decrease-key (O(1) worst-case)

1. Decrease priority
2. If heap property violated:
   - **Cut** node from parent (no cascading!)
   - Add to active root list
   - Update parent's degree
3. **Consolidate if needed**: Only if constraints violated

**Key difference**: No cascading cuts! We cut only the immediate parent, and
the structure constraints prevent deep cascades.

#### Merge (O(1) worst-case)

1. Merge root lists (both active and passive)
2. Update minimum pointer
3. **Consolidate if needed**: Only if constraints violated

Simple structure merge, with conditional consolidation.

### Why No Cascading Cuts?

In standard Fibonacci heaps, cascading cuts ensure amortized bounds. In
Strict Fibonacci heaps:

- **Structure constraints prevent deep violations**: The stricter invariants
  mean violations don't cascade deep
- **Immediate consolidation**: We fix violations when they occur, not later
- **Worst-case bounds**: Immediate repair ensures worst-case O(1)

This is the key insight: by maintaining stricter structure, we prevent cascades.

## Comparison to Standard Fibonacci Heaps

| Feature | Strict Fibonacci | Standard Fibonacci |
|---------|------------------|-------------------|
| Insert | O(1) **worst** | O(1) **amortized** |
| Delete-min | O(log n) **worst** | O(log n) **amortized** |
| Decrease-key | O(1) **worst** | O(1) **amortized** |
| Merge | O(1) **worst** | O(1) **amortized** |
| Complexity | More complex | Complex |
| Guarantees | **Worst-case** | **Amortized** |

## Practical Considerations

**Pros:**

- Worst-case guarantees (real-time systems)
- Same asymptotic bounds as Fibonacci heaps
- Prevents pathological worst-case behavior

**Cons:**

- More complex than standard Fibonacci heaps
- Higher constant overhead
- Active/passive tracking adds complexity

**When to use:**

- Real-time systems requiring worst-case guarantees
- When amortized bounds aren't acceptable
- When worst-case behavior must be bounded

## Implementation Details

The Rust implementation:

- Maintains active and passive root lists
- Uses conditional consolidation
- Tracks structure constraints carefully
- Implements cut without cascading
- Handles root list management carefully

## References

1. Brodal, G. S., Lagogiannis, G., & Tarjan, R. E. (2012). Strict Fibonacci
   heaps. *Proceedings of STOC* 2012, 1177-1184.

2. Brodal, G. S. (1996). Worst-case efficient priority queues. *Proceedings
   of SODA* 1996, 52-58.

3. Fredman, M. L., & Tarjan, R. E. (1987). Fibonacci heaps and their uses
   in improved network optimization algorithms. *Journal of the ACM*,
   34(3), 596-615.
