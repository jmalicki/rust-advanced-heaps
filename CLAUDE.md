# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Build and Test Commands

```bash
# Build
cargo build

# Run all tests
cargo test

# Run a specific test
cargo test test_fibonacci_decrease_key

# Run tests for a specific heap type
cargo test fibonacci
cargo test pairing
cargo test binomial

# Run clippy (with warnings as errors)
cargo clippy --all-targets --all-features -- -D warnings

# Check formatting
cargo fmt --all -- --check

# Build documentation (with warnings as errors)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Run pre-commit hooks manually
pre-commit run --all-files
```

## Architecture

This crate provides advanced heap/priority queue data structures with efficient
`decrease_key` support, which is not available in Rust's standard `BinaryHeap`.

### Core Trait (`src/traits.rs`)

All heaps implement the `Heap<T, P>` trait:

- `T` is the item type
- `P: Ord` is the priority type
- Returns `Handle` on insert for later `decrease_key` operations
- These are **min-heaps** (unlike `BinaryHeap` which is a max-heap)

Key methods: `push`, `pop`, `decrease_key`, `merge`

### Heap Implementations (`src/`)

| Module                 | Heap Type         | decrease_key       | Notes                  |
| ---------------------- | ----------------- | ------------------ | ---------------------- |
| `fibonacci.rs`         | Fibonacci Heap    | O(1) amortized     | Complex, optimal       |
| `pairing.rs`           | Pairing Heap      | o(log n) amortized | Simpler than Fibonacci |
| `rank_pairing.rs`      | Rank-Pairing Heap | O(1) amortized     | Same bounds, simpler   |
| `binomial.rs`          | Binomial Heap     | O(log n)           | Simple, uses bubble-up |
| `strict_fibonacci.rs`  | Strict Fibonacci  | O(1) worst-case    | Worst-case variant     |
| `twothree.rs`          | 2-3 Heap          | O(1) amortized     | Uses Rc/RefCell        |
| `skew_binomial.rs`     | Skew Binomial     | O(log n)           | O(1) insert            |

### Implementation Patterns

- Several heaps use safe `Rc<RefCell<Node>>` with `Weak` references for parent
  pointers (rank-pairing, pairing, 2-3 heap)
- Some heaps use `unsafe` raw pointers (`NonNull<Node<T, P>>`) for node
  management
- Handles are type-erased pointers that identify elements for `decrease_key`
- Handles are **tied to specific heap instances** - using with wrong heap is
  undefined behavior

### Test Structure (`tests/`)

- `generic_heap_tests.rs`: Main test suite with 32 test functions per heap
  type, uses macros to generate tests for all implementations
- `property_tests.rs`: Proptest-based property tests
- `stress_tests.rs`: Large-scale stress tests
- `big_o_proofs.rs`: Complexity verification tests

## Safety Notes

- `decrease_key` requires the new priority to be **less than** current priority
- Handles become invalid after their element is popped
- Documentation lint is enabled: `#![warn(missing_docs)]` in lib.rs
