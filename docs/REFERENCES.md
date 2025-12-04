# Academic References for Heap Data Structures

This document provides citations, paper summaries, and links for heap data
structures implemented in this crate, ordered by publication date.

## Summary

| Heap | Year | decrease-key | Notes |
| --- | --- | --- | --- |
| Simple Binary | 1964 | - | No decrease_key support |
| Binomial | 1978 | O(log n) | Foundational, simple |
| Pairing | 1986 | o(log n) am. | Simple, fast in practice |
| Fibonacci | 1987 | O(1) am. | Optimal amortized bounds |
| Radix | 1990 | O(k)† | Monotone, integer keys |
| Skip List | 1990 | O(log n + m)* | Simple wrapper, good cache |
| Skew Binomial | 1996 | O(log n) | O(1) insert |
| 2-3 Heap | 1999 | O(1) am. | Simpler than Fibonacci |
| Rank-Pairing | 2011 | O(1) am. | Simple + optimal bounds |
| Strict Fibonacci | 2012 | O(1) worst | Optimal worst-case |
| Hollow | 2015 | O(1) am. | Simple, lazy deletion |

*m = duplicate (priority, id) pairs, typically 1

†k = bucket size; O(1) expected in typical Dijkstra usage, O(n) worst case

---

## Skip List (1990)

**Wikipedia:** <https://en.wikipedia.org/wiki/Skip_list>

**Pugh, W. (1990).** Skip lists: A probabilistic alternative to balanced trees.
*Communications of the ACM*, 33(6), 668-676.

- **ACM Digital Library:** <https://dl.acm.org/doi/10.1145/78973.78977>
- **PDF (original):** <https://www.cs.umd.edu/~pugh/SkipLists.pdf>

Skip lists are a probabilistic data structure invented by William Pugh in 1990
as a simpler alternative to balanced trees. They use multiple levels of linked
lists with probabilistically-chosen heights, providing O(log n) expected time
for search, insertion, and deletion.

The key insight is that by randomly assigning "express lane" pointers at
different heights, skip lists achieve the same average-case complexity as
balanced trees without the complexity of rebalancing. Pugh describes them as
"a data structure that has the same expected-time properties as a binary search
tree built from random data."

**As Priority Queues:**

Skip lists can be used as priority queues by maintaining elements in sorted
order. The minimum element is always at the front, providing O(1) peek
operations. Skip lists were first proposed as a concurrent priority queue
structure in:

**Lotan, I. & Shavit, N. (2000).** Skiplist-Based Concurrent Priority Queues.
*International Parallel and Distributed Processing Symposium (IPDPS)*, 263-268.

- **IEEE Xplore:** <https://ieeexplore.ieee.org/document/845994>

Lotan and Shavit were the first to propose using skip lists as the basis for
priority queues (building on Pugh's concurrent skip list). Their key finding
was that **skip lists scale significantly better than heaps** for concurrent
access. While heap-based concurrent priority queues hit scalability limits
around 10-20 processors, their skiplist-based "SkipQueue" continued scaling
to hundreds of processors.

The paper identifies the fundamental advantage: skip lists are **highly
distributed with no hot spots or bottlenecks**. In heaps, the root is a
contention point since every delete-min must access it. In skip lists,
operations on different elements touch mostly disjoint sets of nodes.
Their benchmarks showed deletes 3x faster and inserts 10x faster at 256
processors compared to heap-based approaches.

While the concurrent version requires complex locking or lock-free techniques,
the sequential version is straightforward: insert maintains sorted order via
skip list search, and delete-min removes the first element. The distributed
structure that benefits concurrency also provides good cache behavior in
sequential code, since level arrays are contiguous.

**Trade-offs vs Fibonacci/Pairing heaps:**

- Simpler implementation (wraps existing skiplist crate)
- Better cache locality (contiguous level arrays)
- O(log n) `decrease_key` instead of O(1) amortized
- O(n log n) merge instead of O(1)

**Trade-offs vs Binary heap:**

- Supports `decrease_key` operation
- Similar complexity for basic operations
- Slightly higher memory overhead

| Operation | Expected Time |
| --- | --- |
| insert | O(log n) |
| find-min | O(1) |
| delete-min | O(log n)* |
| decrease-key | O(log n + m)** |
| merge | O(n log n) |

*Technically O(1) to remove the first element, but we count update time
**Where m is the number of duplicate (priority, id) pairs (typically 1)

---

## Simple Binary Heap (1964)

**Wikipedia:** <https://en.wikipedia.org/wiki/Binary_heap>

**Williams, J. W. J. (1964).** Algorithm 232: Heapsort. *Communications of the
ACM*, 7(6), 347-348.

**Floyd, R. W. (1964).** Algorithm 245: Treesort 3. *Communications of the ACM*,
7(12), 701.

The binary heap is one of the most fundamental data structures in computer
science. Williams introduced it alongside the heapsort algorithm in 1964. Floyd
published an improvement the same year showing how to build a heap in O(n) time.
The structure uses a complete binary tree stored implicitly in an array, where
parent-child relationships are computed from array indices.

- **ACM Digital Library (Williams):** <https://dl.acm.org/doi/10.1145/512274.512284>
- **ACM Digital Library (Floyd):** <https://dl.acm.org/doi/10.1145/355588.365103>

The original papers are short algorithm descriptions behind the ACM paywall.
For accessible explanations, see:

- **Wikipedia:** <https://en.wikipedia.org/wiki/Binary_heap>
- **CLRS textbook:** Cormen et al., *Introduction to Algorithms*, Chapter 6

Note: Rust's standard library already provides `std::collections::BinaryHeap`.
This crate includes `SimpleBinaryHeap` for completeness and to provide a
consistent API across all heap types via the `Heap` trait. For algorithms
requiring priority updates, use one of the advanced heap implementations.

| Operation | Worst-case Time |
| --- | --- |
| insert | O(log n) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | - |
| merge | O(n log n) |

---

## Binomial Heap (1978)

**Wikipedia:** <https://en.wikipedia.org/wiki/Binomial_heap>

**Vuillemin, J. (1978).** A data structure for manipulating priority queues.
*Communications of the ACM*, 21(4), 309-315.

- **ACM Digital Library:** <https://dl.acm.org/doi/10.1145/359460.359478>

This foundational paper introduces binomial queues (now commonly called
binomial heaps), which represent a priority queue as a forest of binomial
trees. Each binomial tree B_k has 2^k nodes, and a heap with n elements
contains trees corresponding to the binary representation of n.

The elegant insight is that merging two heaps mirrors binary addition: when two
trees of the same order meet, they combine into a tree of the next higher
order, just like carrying in addition.

Vuillemin's design emphasizes efficient merging, achieving O(log n) merge
time - a significant improvement over binary heaps which require O(n) to merge.

| Operation | Worst-case Time |
| --- | --- |
| insert | O(log n) |
| find-min | O(1)* |
| delete-min | O(log n) |
| decrease-key | O(log n) |
| merge | O(log n) |

*O(1) if minimum pointer is maintained

---

## Pairing Heap (1986)

**Wikipedia:** <https://en.wikipedia.org/wiki/Pairing_heap>

**Fredman, M. L., Sedgewick, R., Sleator, D. D., & Tarjan, R. E. (1986).** The
pairing heap: A new form of self-adjusting heap. *Algorithmica*, 1(1), 111-129.

- **Springer:** <https://link.springer.com/article/10.1007/BF01840439>
- **PDF (CMU):** <https://www.cs.cmu.edu/~sleator/papers/pairing-heaps.pdf>

Pairing heaps were designed as a simpler alternative to Fibonacci heaps that
would be "competitive with the Fibonacci heap in theory and easy to implement
and fast in practice." The structure uses a simple tree representation where
each node stores pointers to its leftmost child and right sibling.

The key operation is the two-pass pairing during delete-min: children are
paired left-to-right, then the resulting trees are merged right-to-left. This
pairing strategy gives the heap its name.

Interestingly, the paper notes that "complete analysis remains an open
problem" - the exact complexity of decrease-key was unresolved for decades. The
current best bound is o(log n) amortized, proven by Iacono and Özkan in 2014.

| Operation | Amortized Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | o(log n) |
| merge | O(1) |

---

## Fibonacci Heap (1987)

**Wikipedia:** <https://en.wikipedia.org/wiki/Fibonacci_heap>

**Fredman, M. L., & Tarjan, R. E. (1987).** Fibonacci heaps and their uses in
improved network optimization algorithms. *Journal of the ACM*, 34(3), 596-615.

- **ACM Digital Library:** <https://dl.acm.org/doi/10.1145/28869.28874>
- **PDF (UMich):**
  <https://web.eecs.umich.edu/~pettie/matching/Fredman-Tarjan-Fibonacci-Heaps.pdf>

This seminal paper introduces Fibonacci heaps (F-heaps), extending binomial
queues proposed by Vuillemin. The key innovation is achieving O(1) amortized
time for insert, find-min, decrease-key, and merge operations, while delete-min
remains O(log n) amortized.

The authors demonstrate the practical importance by improving Dijkstra's
shortest path algorithm from O(E log V) to O(E + V log V), and providing
similar improvements for minimum spanning tree algorithms. The name "Fibonacci"
comes from the Fibonacci numbers appearing in the analysis of the maximum
degree of nodes.

| Operation | Amortized Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(1) |
| merge | O(1) |

---

## Radix Heap (1990)

**Wikipedia:** <https://en.wikipedia.org/wiki/Radix_heap>

**Ahuja, R. K., Mehlhorn, K., Orlin, J. B., & Tarjan, R. E. (1990).** Faster
algorithms for the shortest path problem. *Journal of the ACM*, 37(2), 213-223.

- **ACM Digital Library:** <https://dl.acm.org/doi/10.1145/77600.77615>

Radix heaps are a specialized priority queue for Dijkstra's algorithm with
integer edge weights. They exploit the **monotone property**: in Dijkstra,
extracted distances are non-decreasing. This allows bucketing elements by the
highest differing bit from the last extracted minimum.

The key insight is that when we extract the minimum, all remaining elements
have keys ≥ that minimum. Elements are stored in buckets based on the position
of the highest bit that differs from the current minimum. When bucket 0 (exact
matches) is empty, we find the minimum in the smallest non-empty bucket,
update our reference point, and redistribute elements into finer buckets.

**Why not wrap the `radix-heap` crate?**

The existing `radix-heap` crate (v0.4.2) is incompatible with our `Heap` trait:

1. **No decrease-key**: The crate provides no mechanism to update priorities
2. **Max-heap orientation**: The crate is a max-heap; our heaps are min-heaps
3. **No merge support**: The crate lacks a `merge` operation
4. **Different key bounds**: Requires `Radix + Ord + Copy` vs our `P: Ord`

Our implementation provides a min-heap radix heap with native decrease-key
support, matching the API of our other heap implementations.

**Cache Performance:**

Radix heaps have excellent cache locality because:

- Buckets are contiguous vectors
- Most operations touch only 1-2 buckets
- No pointer chasing (unlike Fibonacci/Pairing heaps)

Empirically, radix heaps are ~2x faster than binary heaps for Dijkstra on
road networks with integer edge weights.

**Constraints:**

- **Monotone**: Cannot insert a key smaller than the last extracted minimum
- **Integer keys**: Requires unsigned integer priorities (`u8`, `u16`, `u32`,
  `u64`, `u128`, `usize`)

These constraints are naturally satisfied by Dijkstra's algorithm with
non-negative integer edge weights.

| Operation | Time Complexity |
| --- | --- |
| insert | O(1) |
| find-min | O(1)* |
| delete-min | O(log C) am. |
| decrease-key | O(k)** |
| merge | O(n) |

*O(1) after redistribution; C = max difference between inserted key and
minimum at insertion time. For bounded edge weights, effectively O(1).

**k = bucket size. O(1) expected in typical Dijkstra usage with well-distributed
priorities. Worst case (all elements in one bucket) is O(n).

---

## Skew Binomial Heap (1996)

**Wikipedia:** <https://en.wikipedia.org/wiki/Skew_binomial_heap>

**Brodal, G. S., & Okasaki, C. (1996).** Optimal purely functional priority
queues. *Journal of Functional Programming*, 6(6), 839-857.

- **Cambridge:** <https://doi.org/10.1017/S095679680000201X>

Related work: **Okasaki, C. (1996).** Purely Functional Data Structures. *PhD
Thesis*, Carnegie Mellon University, CMU-CS-96-177.

- **PDF:** <https://www.cs.cmu.edu/~rwh/students/okasaki.pdf>

Skew binomial heaps extend binomial heaps to achieve O(1) worst-case insertion
(vs O(log n) for binomial heaps). The key innovation is the **skew link**: a
special linking operation that can combine three trees at once.

The paper adapts Brodal's imperative data structure to a purely functional
setting, demonstrating that optimal priority queue bounds are achievable
without mutation. This was significant for functional programming languages.

The "skew" in the name refers to the skew binary number system used to
represent tree sizes, which allows constant-time increment operations.

| Operation | Worst-case Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(log n) |
| merge | O(log n) |

---

## 2-3 Heap (1999)

**Takaoka, T. (1999).** Theory of 2-3 heaps. *Discrete Applied Mathematics*,
126, 115-128.

Earlier conference version:
**Takaoka, T. (1999).** Theory of 2-3 heaps. *Computing and Combinatorics
(COCOON)*, LNCS 1627, 41-50.

- **SpringerLink:** <https://link.springer.com/chapter/10.1007/3-540-48686-0_4>

The 2-3 heap is designed as a simpler alternative to Fibonacci heaps while
maintaining the same amortized bounds. The name comes from the constraint that
each internal node has either 2 or 3 children.

The key insight is replacing Fibonacci heaps' cascading cuts with **rank
propagation**: instead of cutting marked nodes, ranks are updated and
propagated upward. This eliminates the complex marking mechanism while
preserving efficiency.

Takaoka emphasizes practical simplicity: "The merit of the 2-3 heap is that it
is conceptually simpler and easier to implement" compared to other
Fibonacci-like structures.

| Operation | Amortized Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(1) |
| merge | O(1) |

---

## Rank-Pairing Heap (2011)

**Wikipedia:** <https://en.wikipedia.org/wiki/Rank-pairing_heap>

**Haeupler, B., Sen, S., & Tarjan, R. E. (2011).** Rank-pairing heaps. *SIAM
Journal on Computing*, 40(6), 1463-1485.

- **SIAM:** <https://epubs.siam.org/doi/10.1137/100785351>
- **PDF (Princeton):**
  <https://www.cs.princeton.edu/courses/archive/spr10/cos423/handouts/rankpairingheaps.pdf>

Rank-pairing heaps combine the asymptotic efficiency of Fibonacci heaps with
much of the simplicity of pairing heaps. The key insight is using explicit rank
values on nodes instead of the marked-node mechanism of Fibonacci heaps.

Unlike other heap implementations matching Fibonacci heap bounds, rank-pairing
heaps need only **one cut per decrease-key** with no other structural changes.
This makes them significantly simpler to implement while maintaining optimal
bounds.

The paper presents two variants:

- **Type 1:** Allows ranks to decrease by at most 1 at each level
- **Type 2:** More permissive rank constraints, slightly simpler

Initial experiments show rank-pairing heaps perform almost as well as pairing
heaps on typical inputs and better on worst-case sequences.

| Operation | Amortized Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(1) |
| merge | O(1) |

---

## Strict Fibonacci Heap (2012)

**Wikipedia:** <https://en.wikipedia.org/wiki/Fibonacci_heap#Strict_Fibonacci_heap>

**Brodal, G. S., Lagogiannis, G., & Tarjan, R. E. (2012).** Strict Fibonacci
heaps. *Proceedings of the 44th Annual ACM Symposium on Theory of Computing
(STOC)*, 1177-1184.

- **PDF (STOC version):** <https://cs.au.dk/~gerth/papers/stoc12.pdf>
- **PDF (Journal version):** <https://cs.au.dk/~gerth/papers/talg25.pdf>
- **ACM Digital Library:** <https://dl.acm.org/doi/10.1145/2213977.2214082>

Strict Fibonacci heaps achieve the same worst-case bounds as Brodal queues but
with a simpler structure that more closely resembles the original Fibonacci
heaps.

Key innovations:

- **Simplified melding:** When merging heaps of different sizes, the smaller
  heap's structure is discarded
- **Pigeonhole-based balancing:** Uses the pigeonhole principle instead of
  redundant counters
- **Active/passive nodes:** Nodes are classified to track structural violations

The paper proves these simplifications still maintain worst-case bounds, making
the structure more accessible than Brodal's original design while remaining
primarily of theoretical interest.

| Operation | Worst-case Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(1) |
| merge | O(1) |

---

## Hollow Heap (2015)

**Hansen, T. D., Kaplan, H., Tarjan, R. E., & Zwick, U. (2015).** Hollow Heaps.
*Proceedings of the 42nd International Colloquium on Automata, Languages, and
Programming (ICALP)*, 689-700.

- **arXiv:** <https://arxiv.org/abs/1510.06535>
- **ACM Digital Library:** <https://dl.acm.org/doi/10.1145/3093240>

Journal version:
**Hansen, T. D., Kaplan, H., Tarjan, R. E., & Zwick, U. (2017).** Hollow Heaps.
*ACM Transactions on Algorithms*, 13(3), Article 42.

Hollow heaps achieve the same amortized bounds as Fibonacci heaps but with a
simpler implementation. The key innovations are:

1. **Lazy deletion for decrease-key**: Instead of restructuring the heap when
   decreasing a key, hollow heaps create a new node with the lower key and mark
   the old node as "hollow" (empty). The hollow node remains in the structure
   until it naturally gets removed during delete-min.

2. **DAG structure**: Hollow nodes can have a "second parent" pointer, creating
   a directed acyclic graph (DAG) instead of a forest of trees. This allows
   efficient handling of the hollow nodes during consolidation.

3. **Simple ranked linking**: During delete-min, the heap performs ranked
   linking similar to Fibonacci heaps, but without the complexity of cascading
   cuts or node marking.

The paper notes that hollow heaps are "the simplest heap structure known that
achieves the optimal amortized bounds" for priority queue operations. The
structure was developed by Robert Tarjan and collaborators as part of ongoing
work to find simpler data structures with optimal complexity.

**Trade-offs vs Fibonacci heaps:**

- Simpler decrease-key (no cascading cuts, no marking)
- Same amortized bounds
- Potentially more hollow nodes in memory (lazy cleanup)
- Requires P: Clone for the priority type

| Operation | Amortized Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(1) |
| merge | O(1) |

---

## Additional Resources

**Brodal, G. S. (2013).** A survey on priority queues.
<https://cs.au.dk/~gerth/papers/ianfest13.pdf>

**Wikipedia comparison:**
<https://en.wikipedia.org/wiki/Fibonacci_heap#Summary_of_running_times>

**Textbooks:**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009).
  *Introduction to Algorithms* (3rd ed.). MIT Press. (Chapters 19-20 cover
  Fibonacci and binomial heaps)
- Okasaki, C. (1998). *Purely Functional Data Structures*. Cambridge University
  Press.

The implementations in this crate prioritize correctness and clarity over raw
performance. For production use with very large datasets, benchmarking against
specific workloads is recommended.

---

## Unimplemented Heaps

The following heaps are documented for reference but are not currently
implemented in this crate.

---

### Brodal Heap (1996)

*Status: Not implemented in this crate.*

The Brodal queue is not implemented due to the complexity of translating its
intricate pointer-based structure into Rust's memory-safe model. The data
structure relies heavily on mutable aliasing patterns that are challenging to
express safely in Rust without significant performance penalties or extensive
use of `unsafe` code.

**Wikipedia:** <https://en.wikipedia.org/wiki/Brodal_queue>

**Brodal, G. S. (1996).** Worst-case efficient priority queues. *Proceedings
of the 7th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, 52-58.

- **PDF (Aarhus):** <http://www.cs.au.dk/~gerth/papers/soda96.pdf>
- **ACM Digital Library:** <https://dl.acm.org/doi/10.5555/313852.313883>

The Brodal queue is a landmark achievement: the **first pointer-based heap with
optimal worst-case time bounds** for all standard operations. Previous heaps
like Fibonacci heaps achieved optimal amortized bounds, but Brodal showed
worst-case optimality is also possible.

The data structure is complex, using a combination of:

- Guide trees to maintain structural invariants
- Violation lists to track heap property violations
- Redundant counter mechanisms for efficient updates

The practical complexity is high - the paper notes significant constant
factors. The structure is primarily of theoretical interest, demonstrating
what is achievable rather than what is practical.

**Why not implemented:**

1. **Complex mutable aliasing**: The structure requires multiple mutable
   references to the same nodes, which conflicts with Rust's borrowing rules
2. **Guide tree navigation**: The guide tree mechanism requires intricate
   pointer manipulation that is difficult to express safely
3. **Marginal practical benefit**: The Strict Fibonacci Heap (which is
   implemented) provides the same worst-case bounds with a simpler structure

For applications requiring worst-case O(1) decrease-key, consider using the
Strict Fibonacci Heap implementation instead.

| Operation | Worst-case Time |
| --- | --- |
| insert | O(1) |
| find-min | O(1) |
| delete-min | O(log n) |
| decrease-key | O(1) |
| merge | O(1) |
