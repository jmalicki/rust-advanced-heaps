# Rust Advanced Heaps

A comprehensive collection of advanced heap/priority queue data structures
for Rust with efficient `decrease_key` support.

## Motivation

Rust's standard library provides `BinaryHeap`, but it doesn't support
`decrease_key` operations efficiently. Many third-party libraries use hash
maps paired with binary heaps, which only achieve O(log n) `decrease_key`
operations. This crate implements advanced heap structures from computer
science literature that provide better amortized bounds:

- **Fibonacci Heap**: O(1) amortized `decrease_key`
- **Pairing Heap**: o(log n) amortized `decrease_key`
- **Rank-Pairing Heap**: O(1) amortized `decrease_key`
- **Binomial Heap**: O(log n) `decrease_key` (but simpler implementation)

## Features

### Implemented Heaps

| Heap Type | Insert | Delete-min | Decrease-key | Merge | Notes |
|-----------|--------|------------|--------------|-------|-------|
| **Fibonacci** | O(1) am. | O(log n) am. | **O(1) am.** | O(1) | Complex but optimal |
| **Pairing** | O(1) am. | O(log n) am. | **o(log n) am.** | O(1) | Simpler than Fibonacci |
| **Rank-Pairing** | O(1) am. | O(log n) am. | **O(1) am.** | O(1) | Simpler than Fibonacci, optimal bounds |
| **Binomial** | O(log n) | O(log n) | O(log n) | O(log n) | Simple, well-understood |
| **Brodal** | O(1) worst | O(log n) worst | **O(1) worst** | O(1) worst | Optimal worst-case, high constants |

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

- ✅ Fibonacci Heap
- ✅ Pairing Heap  
- ✅ Rank-Pairing Heap
- ✅ Binomial Heap
- ✅ Brodal Heap
- ⏳ More heap types (Skew heap, Leftist heap, Strict Fibonacci, etc.)

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

Contributions welcome! Areas of focus:

1. More heap implementations (Rank-pairing, Brodal queue, etc.)
2. Performance benchmarks comparing implementations
3. Comprehensive test coverage
4. Documentation improvements

## License

MIT OR Apache-2.0

## References

- [Fibonacci Heap - Wikipedia](https://en.wikipedia.org/wiki/Fibonacci_heap)
- [Pairing Heap - Wikipedia](https://en.wikipedia.org/wiki/Pairing_heap)
- [Binomial Heap - Wikipedia](https://en.wikipedia.org/wiki/Binomial_heap)
- Fredman, M. L., & Tarjan, R. E. (1987). Fibonacci heaps and their uses in
  improved network optimization algorithms. Journal of the ACM, 34(3),
  596-615.
