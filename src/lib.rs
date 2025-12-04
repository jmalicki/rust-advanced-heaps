#![warn(missing_docs)]

//! Advanced Heap Data Structures for Rust
//!
//! This crate provides implementations of various advanced heap/priority queue data structures
//! with efficient `decrease_key` support, as described in computer science literature.
//!
//! # Trait Hierarchy
//!
//! This crate provides a two-tier trait design:
//!
//! - [`Heap`]: Base trait for simple heaps (push, pop, peek, merge)
//! - [`DecreaseKeyHeap`]: Extended trait adding `decrease_key` and handle-based operations
//!
//! This allows algorithms to be generic over heaps at the appropriate level.
//!
//! # Implementations
//!
//! | Heap | push | pop | decrease_key | Trait |
//! |------|------|-----|--------------|-------|
//! | [`simple_binary::SimpleBinaryHeap`] | O(log n) | O(log n) | - | `Heap` only |
//! | [`skiplist::SkipListHeap`] | O(log n) | O(log n) | O(log n + m)* | `DecreaseKeyHeap` |
//! | [`fibonacci::FibonacciHeap`] | O(1) am. | O(log n) am. | O(1) am. | `DecreaseKeyHeap` |
//! | [`pairing::PairingHeap`] | O(1) am. | O(log n) am. | o(log n) am. | `DecreaseKeyHeap` |
//! | [`rank_pairing::RankPairingHeap`] | O(1) am. | O(log n) am. | O(1) am. | `DecreaseKeyHeap` |
//! | [`binomial::BinomialHeap`] | O(log n) | O(log n) | O(log n) | `DecreaseKeyHeap` |
//! | [`strict_fibonacci::StrictFibonacciHeap`] | O(1) worst | O(log n) worst | O(1) worst | `DecreaseKeyHeap` |
//! | [`twothree::TwoThreeHeap`] | O(1) am. | O(log n) am. | O(1) am. | `DecreaseKeyHeap` |
//! | [`skew_binomial::SkewBinomialHeap`] | O(1) | O(log n) | O(log n) | `DecreaseKeyHeap` |
//! | [`radix::RadixHeap`] | O(1) | O(log C) am.† | O(1) | `DecreaseKeyHeap` |
//!
//! *SkipListHeap: m = duplicate (priority,id) pairs (typically 1).
//! Requires `P: Copy`, `T: Default` for `decrease_key`.
//!
//! †RadixHeap: C = max key difference. Monotone heap (requires extracted keys to be non-decreasing).
//! Requires `P: RadixKey` (unsigned integers). Ideal for Dijkstra with integer edge weights.
//!
//! # Basic Example
//!
//! ```rust
//! use rust_advanced_heaps::Heap;
//! use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
//!
//! let mut heap = SimpleBinaryHeap::new();
//! heap.push(3, "three");
//! heap.push(1, "one");
//! assert_eq!(heap.peek(), Some((&1, &"one")));
//! assert_eq!(heap.pop(), Some((1, "one")));
//! ```
//!
//! # Example with decrease_key
//!
//! ```rust
//! use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
//! use rust_advanced_heaps::fibonacci::FibonacciHeap;
//!
//! let mut heap = FibonacciHeap::new();
//! let handle1 = heap.push_with_handle(5, "item1");
//! let _handle2 = heap.push_with_handle(3, "item2");
//! heap.decrease_key(&handle1, 1).unwrap();
//! assert_eq!(heap.peek(), Some((&1, &"item1")));
//! ```

pub mod binomial;
pub mod fibonacci;
pub mod pairing;
pub mod pathfinding;
pub mod radix;
pub mod rank_pairing;
pub mod simple_binary;
pub mod skew_binomial;
pub mod skiplist;
pub mod stdlib_compat;
pub mod strict_fibonacci;
pub mod traits;
pub mod twothree;

// Re-export the main traits for convenience
pub use traits::{DecreaseKeyHeap, Heap};
