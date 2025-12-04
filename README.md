# Rust Advanced Heaps

A comprehensive collection of advanced heap/priority queue data structures
for Rust with efficient `decrease_key` support.

## Motivation

Rust's standard library provides `BinaryHeap`, but it doesn't support
`decrease_key` operations efficiently. Many third-party libraries use hash
maps paired with binary heaps, which only achieve O(log n) `decrease_key`
operations. This crate implements advanced heap structures from computer
science literature that provide better amortized bounds:

- **Binomial Heap** (1978): O(log n) `decrease_key` - foundational, simple
- **Pairing Heap** (1986): o(log n) amortized `decrease_key` - simple, fast
- **Fibonacci Heap** (1987): O(1) amortized `decrease_key` - optimal amortized
- **Skew Binomial Heap** (1996): O(1) insert, O(log n) `decrease_key`
- **2-3 Heap** (1999): O(1) amortized `decrease_key` - simpler than Fibonacci
- **Rank-Pairing Heap** (2011): O(1) amortized `decrease_key` - simple + optimal
- **Strict Fibonacci Heap** (2012): O(1) worst-case `decrease_key`

## Features

### Implemented Heaps

| Heap Type | Year | Insert | Delete-min | Decrease-key | Merge |
| --------- | ---- | ------ | ---------- | ------------ | ----- |
| **Binomial** | 1978 | O(log n) | O(log n) | O(log n) | O(log n) |
| **Pairing** | 1986 | O(1) am. | O(log n) am. | **o(log n) am.** | O(1) |
| **Fibonacci** | 1987 | O(1) am. | O(log n) am. | **O(1) am.** | O(1) |
| **Skew Binomial** | 1996 | O(1) | O(log n) | O(log n) | O(log n) |
| **2-3 Heap** | 1999 | O(1) am. | O(log n) am. | **O(1) am.** | O(1) am. |
| **Rank-Pairing** | 2011 | O(1) am. | O(log n) am. | **O(1) am.** | O(1) |
| **Strict Fibonacci** | 2012 | O(1) worst | O(log n) worst | **O(1) worst** | O(1) worst |

All times are amortized (am.) where applicable. See the [Wikipedia
comparison table](https://en.wikipedia.org/wiki/Fibonacci_heap#Summary_of_running_times)
for full details.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-advanced-heaps = { path = "../rust-advanced-heaps" }
```

Example:

```rust
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::Heap;

let mut heap = FibonacciHeap::new();

// Insert elements, getting handles for decrease_key
let handle1 = heap.insert(10, "item1");
let handle2 = heap.insert(20, "item2");
let handle3 = heap.insert(30, "item3");

// Find minimum (O(1))
assert_eq!(heap.find_min(), Some((&10, &"item1")));

// Decrease key efficiently (O(1) amortized for Fibonacci heap)
heap.decrease_key(&handle2, 5);
assert_eq!(heap.find_min(), Some((&5, &"item2")));

// Delete minimum (O(log n) amortized)
let min = heap.delete_min();
assert_eq!(min, Some((5, "item2")));
```

## API

All heaps implement the `Heap` trait:

```rust
pub trait Heap<T, P: Ord> {
    type Handle: Handle;

    fn new() -> Self;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn insert(&mut self, priority: P, item: T) -> Self::Handle;
    fn find_min(&self) -> Option<(&P, &T)>;
    fn delete_min(&mut self) -> Option<(P, T)>;
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P);
    fn merge(&mut self, other: Self);
}
```

## Safety Notes

- **Handles are tied to specific heap instances**. Using a handle from one
  heap with another heap, or after the heap is dropped, is undefined
  behavior.
- `decrease_key` expects the new priority to be **less than** the current
  priority. Behavior is undefined if this is not true.

## Implementation Status

- ✅ Binomial Heap (1978)
- ✅ Pairing Heap (1986)
- ✅ Fibonacci Heap (1987)
- ✅ Skew Binomial Heap (1996)
- ✅ 2-3 Heap (1999)
- ✅ Rank-Pairing Heap (2011)
- ✅ Strict Fibonacci Heap (2012)

## Performance Considerations

While these heaps provide excellent theoretical bounds, constant factors
matter in practice:

- **Fibonacci heaps** have significant constant overhead and are often
  slower than binary heaps for small inputs
- **Pairing heaps** are simpler than Fibonacci heaps and often perform
  better in practice
- **Binomial heaps** provide a good balance between simplicity and
  performance

Choose based on your workload:

- Many `decrease_key` operations: Fibonacci or Pairing heap
- Mostly inserts/deletes: Binomial heap may be sufficient
- Small heaps: Binary heap is often faster due to lower constant factors

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution
guidelines.

## License

MIT OR Apache-2.0

## References

For complete academic citations and links to original papers, see
[docs/REFERENCES.md](docs/REFERENCES.md).

Quick links:

- [Fibonacci Heap - Wikipedia](https://en.wikipedia.org/wiki/Fibonacci_heap)
- [Pairing Heap - Wikipedia](https://en.wikipedia.org/wiki/Pairing_heap)
- [Binomial Heap - Wikipedia](https://en.wikipedia.org/wiki/Binomial_heap)
- [Rank-Pairing Heap - Wikipedia](https://en.wikipedia.org/wiki/Rank-pairing_heap)
- [Skew Binomial Heap - Wikipedia](https://en.wikipedia.org/wiki/Skew_binomial_heap)
