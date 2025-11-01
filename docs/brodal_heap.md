# Brodal Heap

## Overview

The **Brodal Heap** achieves the same asymptotic bounds as Fibonacci heaps, but with **worst-case** guarantees instead of amortized ones. This is a remarkable achievement: O(1) worst-case insert, decrease_key, and merge; O(log n) worst-case delete_min.

## Historical Context and Papers

### Original Paper
- **Brodal, Gerth Stølting** (1996). "Worst-case efficient priority queues". *Proceedings of the 7th Annual ACM-SIAM Symposium on Discrete Algorithms*. pp. 52–58.
   - Introduced the Brodal heap and showed that worst-case bounds matching Fibonacci heaps are possible

### Key Follow-up Work

1. **Brodal, Gerth Stølting** (1996). "Priority queues on a fixed number of heaps". *Proceedings of the 3rd Annual European Symposium on Algorithms (ESA)*. pp. 209–223. doi:10.1007/3-540-61581-5_66.
   - Further analysis and variants

2. **Elmasry, Amr** (2004). "Pairing heaps with O(log log n) decrease cost". *Proceedings of the 15th Annual European Symposium on Algorithms (ESA)*. pp. 183–194. doi:10.1007/978-3-540-39658-1_19.
   - Related work on pairing heaps

3. **Brodal, Gerth Stølting; Lagogiannis, George; Tarjan, Robert E.** (2012). "Strict Fibonacci heaps". *Proceedings of the 44th Annual ACM Symposium on Theory of Computing (STOC)*. pp. 1177–1184. doi:10.1145/2213977.2214080.
   - Strict Fibonacci heaps also achieve worst-case bounds

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `insert` | **O(1)** worst-case | Constant time guaranteed |
| `find_min` | **O(1)** worst-case | Direct pointer to minimum |
| `delete_min` | **O(log n)** worst-case | Worst-case logarithmic |
| `decrease_key` | **O(1)** worst-case | Constant time guaranteed |
| `merge` | **O(1)** worst-case | Constant time guaranteed |

**Key Achievement**: These are **worst-case** bounds, matching the **amortized** bounds of Fibonacci heaps!

## How It Works

### The Challenge

Fibonacci heaps achieve O(1) amortized bounds using:
- **Lazy evaluation**: Defer work until necessary
- **Potential functions**: Charge cheap operations to pay for expensive ones

To achieve **worst-case** bounds, we need to:
- Do work immediately instead of deferring
- Maintain stricter structural invariants
- Use violation tracking and repair

### Data Structure

A Brodal heap maintains:
- A heap-ordered tree structure
- **Rank constraints** (similar to rank-pairing heaps)
- **Violation queues**: Per-rank lists of nodes violating constraints
- **Active repair**: Fix violations as they occur (worst-case O(1) per violation)

### Rank Constraints

Each node has a **rank** that must satisfy:
- For node v with children w₁, w₂ (two smallest ranks):
  - rank(v) ≤ rank(w₁) + 1
  - rank(v) ≤ rank(w₂) + 1

This bounds the tree height while allowing efficient updates.

### Violation System

The key innovation is the **violation tracking system**:

1. **Per-rank violation queues**: For each rank r, maintain a queue of nodes with rank r that violate constraints
2. **Immediate repair**: After each operation, repair at most O(1) violations
3. **Repair during delete_min**: Process all violations during delete_min (amortized over the operation)

**Why this works:**
- Each operation creates at most O(1) new violations
- Repairing one violation is O(1) (local restructuring)
- Over a sequence of operations, violations are repaired as they accumulate
- During delete_min, we can afford O(log n) work

### Key Operations

#### Insert (O(1) worst-case)

1. Create new node with rank 0
2. Merge with root (make it child or new root)
3. Update rank of parent
4. **Check for rank violations**: If violation created, add to violation queue
5. **Repair one violation**: Process one violation from current rank (worst-case O(1))

The repair is O(1) because it only affects nodes locally.

#### Decrease-key (O(1) worst-case)

1. Decrease priority
2. If heap property violated:
   - Cut node from parent
   - Merge with root
3. Update parent's rank
4. **Check for violations**: If parent's rank constraint violated, add to queue
5. **Repair one violation**: Process one violation (worst-case O(1))

The key is that we only repair one violation per operation, ensuring O(1) worst-case.

#### Delete-min (O(log n) worst-case)

1. Remove root
2. Collect all children
3. **Process all violations**: This is where accumulated violations are fixed
4. Rebuild heap from children, maintaining rank constraints
5. Find new minimum

**Why O(log n)?**
- At most O(log n) children (rank bound)
- Rebuilding maintains structure
- Violation processing: we've been repairing violations along the way, so only O(log n) remain

#### Merge (O(1) worst-case)

1. Compare roots
2. Make larger priority a child of smaller
3. Update rank
4. **Check for violations**: Merge may create violations
5. **Repair one violation**: Fix immediately (worst-case O(1))

### Violation Repair

When a violation is detected:

1. **Identify violation**: Node's rank exceeds children's rank constraints
2. **Restructure locally**: 
   - Disconnect children
   - Group children by rank
   - Link children to reduce rank
   - Reattach to parent
3. **Update ranks**: Recompute ranks after restructuring
4. **Check for new violations**: Repair may create new violations (but bounded)

The repair is **local** - only affects the violating node and its children.

## Comparison to Fibonacci Heaps

| Feature | Brodal Heap | Fibonacci Heap |
|---------|------------|----------------|
| Insert | O(1) **worst** | O(1) **amortized** |
| Delete-min | O(log n) **worst** | O(log n) **amortized** |
| Decrease-key | O(1) **worst** | O(1) **amortized** |
| Merge | O(1) **worst** | O(1) **amortized** |
| Complexity | Very complex | Complex |
| Guarantees | **Worst-case** | **Amortized** |

**Trade-offs:**
- **Pro**: Worst-case guarantees (real-time systems, hard deadlines)
- **Con**: Much more complex implementation
- **Con**: Higher constant factors due to violation tracking

## Practical Considerations

**When to use:**
- Real-time systems requiring worst-case guarantees
- When amortized bounds aren't sufficient
- When constant factors matter less than guarantees

**When not to use:**
- Simple applications (binary heap often faster)
- When amortized bounds are acceptable
- When implementation complexity is a concern

## Implementation Details

The Rust implementation includes:
- Per-rank violation queues
- Rank constraint checking after operations
- Violation repair with local restructuring
- Careful rank maintenance
- Complex but correct violation handling

## References

1. Brodal, G. S. (1996). Worst-case efficient priority queues. *Proceedings of SODA* 1996, 52-58.

2. Brodal, G. S. (1996). Priority queues on a fixed number of heaps. *Proceedings of ESA* 1996, 209-223.

3. Elmasry, A. (2004). Pairing heaps with O(log log n) decrease cost. *Proceedings of ESA* 2004, 183-194.

4. Brodal, G. S., Lagogiannis, G., & Tarjan, R. E. (2012). Strict Fibonacci heaps. *Proceedings of STOC* 2012, 1177-1184.

