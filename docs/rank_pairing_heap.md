# Rank-Pairing Heap

## Overview

The **Rank-Pairing Heap** is a heap data structure designed to achieve the
same optimal amortized bounds as Fibonacci heaps (O(1) insert, decrease_key,
merge; O(log n) delete_min) while being simpler to implement. It uses
rank-based restructuring instead of the cascading cuts used in Fibonacci
heaps.

## Historical Context and Papers

### Original Paper

- **Haeupler, Bernhard; Sen, Siddhartha; Tarjan, Robert E.** (2011).
  "Rank-pairing heaps". *SIAM Journal on Computing*. 40 (6): 1463–1485.
  doi:10.1137/100789351.

### Key Follow-up Work

1. **Haeupler, Bernhard; Sen, Siddhartha; Tarjan, Robert E.** (2009).
   "Rank-pairing heaps". *Proceedings of the 17th Annual European Symposium
   on Algorithms (ESA)*. pp. 659–670. doi:10.1007/978-3-642-04128-0_59.
   - Earlier conference version

2. **Cho, Sungjin; Sahni, Sartaj** (2000). "Mergeable heaps with decrease
   key". *Operations Research Letters*. 26 (3): 169–174.
   doi:10.1016/S0167-6377(00)00030-1.
   - Related work on mergeable heaps

## Asymptotic Complexity

| Operation | Time Complexity | Notes |
| --------- | -------------- | ----- |
| `insert` | **O(1)** amortized | Constant time insertion |
| `find_min` | **O(1)** | Direct access to minimum |
| `delete_min` | **O(log n)** amortized | Rank-based merging |
| `decrease_key` | **O(1)** amortized | Rank-based restructuring |
| `merge` | **O(1)** amortized | Simple root comparison |

**Note**: These match Fibonacci heap bounds exactly, but with a simpler implementation.

## How It Works

### Data Structure

A rank-pairing heap is similar to a pairing heap but maintains **rank** information:

- Each node stores a **rank** (non-negative integer)
- Ranks satisfy constraints based on children's ranks
- Rank constraints maintain structure without cascading cuts
- Tree structure is heap-ordered (min-heap property)

### Rank Constraints (Type-A)

The implementation uses **type-A rank-pairing heaps**, which maintain:

1. **Rank constraint**: For any node v with children w₁, w₂ (two smallest ranks):
   - rank(v) ≤ rank(w₁) + 1
   - rank(v) ≤ rank(w₂) + 1

2. **Marking rule**: A node can lose at most **one child** before being cut
   - Unmarked: no child lost
   - Marked: one child lost (after another loss, node is cut)

This is simpler than Fibonacci heaps because:

- Rank constraints are explicit
- No complex cascading cuts
- Rank updates are localized

### Key Operations

#### Insert (O(1) amortized)

1. Create new node with rank 0
2. Compare with root
3. Make smaller priority the root
4. Link as parent-child relationship
5. Update rank of parent

The rank update ensures constraints are maintained.

#### Update Rank

The rank of a node is computed from its children:

- Find two children with **smallest ranks** (r₁, r₂)
- New rank = min(r₁, r₂) + 1
- This ensures the rank constraint is satisfied

#### Delete-min (O(log n) amortized)

1. Remove root
2. Collect all children
3. **Merge children using rank-based pairing**:
   - Group children by rank
   - Link pairs of same rank
   - Continue until single root

The rank-based pairing ensures O(log n) roots remain after merging.

#### Decrease-key (O(1) amortized)

1. Decrease priority
2. If heap property violated:
   - **Cut** node from parent
   - If parent was marked, cut parent too (cascading cut)
   - Mark parent (if not already marked)
   - Merge with root
3. Update ranks as needed

The marking rule ensures that at most one cut per parent is needed (amortized).

#### Merge (O(1) amortized)

Simple root comparison:

1. Compare roots
2. Make larger priority a child of smaller
3. Update rank

This is clearly O(1).

## Advantages over Fibonacci Heaps

1. **Simplicity**: No complex circular lists or cascading cut logic
2. **Same bounds**: Achieves identical amortized bounds
3. **Explicit ranks**: Rank constraints are explicit and easier to verify
4. **Better practice**: Often performs better in practice due to lower overhead

## Rank-Based Analysis

The rank-based analysis shows:

- Maximum rank is O(log n)
- Each operation affects O(1) nodes amortized
- Rank updates maintain structure efficiently

The key insight is that rank constraints bound the tree height while
allowing efficient updates.

## Type-A vs Type-B

There are two variants of rank-pairing heaps:

**Type-A** (implemented here):

- Node can lose at most one child before being cut
- Simpler implementation
- Same bounds

**Type-B**:

- More permissive marking rules
- Slightly more complex but may have better constants
- Same asymptotic bounds

## Implementation Details

The Rust implementation:

- Uses explicit rank fields
- Updates ranks locally after operations
- Maintains mark flags for cut tracking
- Uses sibling lists (simpler than circular lists)

## References

1. Haeupler, B., Sen, S., & Tarjan, R. E. (2011). Rank-pairing heaps.
   *SIAM Journal on Computing*, 40(6), 1463-1485.

2. Haeupler, B., Sen, S., & Tarjan, R. E. (2009). Rank-pairing heaps.
   *Proceedings of ESA* 2009, 659-670.

3. Cho, S., & Sahni, S. (2000). Mergeable heaps with decrease key.
   *Operations Research Letters*, 26(3), 169-174.
