# Rust Advanced Heaps

A comprehensive collection of advanced heap/priority queue data structures
for Rust with efficient `decrease_key` support.

## Motivation

Rust's standard library provides `BinaryHeap`, but it doesn't support
`decrease_key` operations efficiently. Many third-party libraries use hash
maps paired with binary heaps, which only achieve O(log n) `decrease_key`
operations. This crate implements advanced heap structures from computer
science literature that provide better amortized bounds:

- **Simple Binary Heap**: O(log n) operations - no `decrease_key` support
- **Binomial Heap** (1978): O(log n) `decrease_key` - foundational, simple
- **Pairing Heap** (1986): o(log n) amortized `decrease_key` - simple, fast
- **Fibonacci Heap** (1987): O(1) amortized `decrease_key` - optimal amortized
- **Radix Heap** (1990): O(1) `decrease_key` - monotone, integer keys, cache-friendly
- **Skip List Heap** (1990): O(log n) `decrease_key` - simple wrapper, good cache
- **Skew Binomial Heap** (1996): O(1) insert, O(log n) `decrease_key`
- **2-3 Heap** (1999): O(1) amortized `decrease_key` - simpler than Fibonacci
- **Rank-Pairing Heap** (2011): O(1) amortized `decrease_key` - simple + optimal
- **Strict Fibonacci Heap** (2012): O(1) worst-case `decrease_key`

## Features

### Two-Tier Trait Hierarchy

This crate provides a two-tier trait design:

- **`Heap`**: Base trait for simple heaps without `decrease_key` support
- **`DecreaseKeyHeap`**: Extended trait adding `decrease_key` and handle-based operations

This allows algorithms to be generic over heaps at the appropriate level of abstraction.

### Implemented Heaps

| Heap Type | Year | Push | Pop | Decrease-key | Merge |
| --------- | ---- | ---- | --- | ------------ | ----- |
| **Simple Binary** | - | O(log n) | O(log n) | - | O(n log n) |
| **Binomial** | 1978 | O(log n) | O(log n) | O(log n) | O(log n) |
| **Pairing** | 1986 | O(1) am. | O(log n) am. | **o(log n) am.** | O(1) |
| **Fibonacci** | 1987 | O(1) am. | O(log n) am. | **O(1) am.** | O(1) |
| **Radix**† | 1990 | O(1) | O(log C) am. | **O(1)** | O(n) |
| **Skip List** | 1990 | O(log n) | O(log n) | O(log n + m)* | O(n log n) |
| **Skew Binomial** | 1996 | O(1) | O(log n) | O(log n) | O(log n) |
| **2-3 Heap** | 1999 | O(1) am. | O(log n) am. | **O(1) am.** | O(1) am. |
| **Rank-Pairing** | 2011 | O(1) am. | O(log n) am. | **O(1) am.** | O(1) |
| **Strict Fibonacci** | 2012 | O(1) worst | O(log n) worst | **O(1) worst** | O(1) worst |

*m = duplicate (priority, id) pairs after merge, typically 1.
Requires `T: Default`, `P: Copy`.

†Radix Heap: Monotone priority queue (extracted keys never decrease). C = max key
difference. Requires `P: RadixKey` (unsigned integers). Ideal for Dijkstra.

All times are amortized (am.) where applicable. See the [Wikipedia
comparison table](https://en.wikipedia.org/wiki/Fibonacci_heap#Summary_of_running_times)
for full details.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-advanced-heaps = { path = "../rust-advanced-heaps" }
```

### Basic Example (without decrease_key)

```rust
use rust_advanced_heaps::Heap;
use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;

let mut heap = SimpleBinaryHeap::new();
heap.push(3, "three");
heap.push(1, "one");
heap.push(2, "two");

assert_eq!(heap.peek(), Some((&1, &"one")));
assert_eq!(heap.pop(), Some((1, "one")));
```

### With decrease_key Support

```rust
use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
use rust_advanced_heaps::fibonacci::FibonacciHeap;

let mut heap = FibonacciHeap::new();

// Insert elements, getting handles for decrease_key
let handle1 = heap.push_with_handle(10, "item1");
let handle2 = heap.push_with_handle(20, "item2");
let _handle3 = heap.push_with_handle(30, "item3");

// Peek at minimum (O(1))
assert_eq!(heap.peek(), Some((&10, &"item1")));

// Decrease key efficiently (O(1) amortized for Fibonacci heap)
heap.decrease_key(&handle2, 5).unwrap();
assert_eq!(heap.peek(), Some((&5, &"item2")));

// Pop minimum (O(log n) amortized)
let min = heap.pop();
assert_eq!(min, Some((5, "item2")));
```

## API

### Base Trait: `Heap`

All heaps implement the base `Heap` trait:

```rust
pub trait Heap<T, P: Ord> {
    fn new() -> Self;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn push(&mut self, priority: P, item: T);
    fn peek(&self) -> Option<(&P, &T)>;
    fn pop(&mut self) -> Option<(P, T)>;
    fn merge(&mut self, other: Self);
}
```

### Extended Trait: `DecreaseKeyHeap`

Advanced heaps also implement `DecreaseKeyHeap`:

```rust
pub trait DecreaseKeyHeap<T, P: Ord>: Heap<T, P> {
    type Handle: Handle;

    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle;
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError>;
}
```

## Error Handling

`decrease_key` returns a `Result<(), HeapError>`:

- `HeapError::InvalidHandle`: The handle is no longer valid (element was
  removed or handle is from a different heap)
- `HeapError::PriorityNotDecreased`: The new priority is not less than the
  current priority

## Safety Notes

- **Handles are tied to specific heap instances**. Using a handle from one
  heap with another heap may return an error or produce unexpected results.
- After an element is popped from the heap, its handle becomes invalid and
  `decrease_key` will return `HeapError::InvalidHandle`.

## Implementation Status

- ✅ Simple Binary Heap (base `Heap` only)
- ✅ Binomial Heap (1978)
- ✅ Pairing Heap (1986)
- ✅ Fibonacci Heap (1987)
- ✅ Radix Heap (1990) - monotone, unsigned integer keys only
- ✅ Skip List Heap (1990) - requires `T: Default` for `decrease_key`
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

## Benchmarks

This crate includes benchmarks comparing heap implementations on shortest
path problems using DIMACS road network graphs. See
[docs/benchmarks.md](docs/benchmarks.md) for setup instructions and details.

## Development Setup

This project uses pre-commit hooks for code quality checks. After cloning or
initializing a new worktree, install the hooks:

```bash
# Option 1: Use the setup script
./setup.sh

# Option 2: Manual setup
# Install pre-commit (if not already installed)
pip install pre-commit

# Install git hooks
pre-commit install
```

The hooks will automatically run `cargo fmt`, `cargo clippy`, and markdownlint
on every commit.

**Note for worktrees**: When using git worktrees (including Cursor's "new
worktree init"), you must run `pre-commit install` in each worktree after it's
created. Pre-commit hooks are installed per repository, but each worktree needs
to be set up individually.

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
- [Skip List - Wikipedia](https://en.wikipedia.org/wiki/Skip_list)
- [Rank-Pairing Heap - Wikipedia](https://en.wikipedia.org/wiki/Rank-pairing_heap)
- [Skew Binomial Heap - Wikipedia](https://en.wikipedia.org/wiki/Skew_binomial_heap)
