//! Advanced Heap Data Structures for Rust
//!
//! This crate provides implementations of various advanced heap/priority queue data structures
//! with efficient `decrease_key` support, as described in computer science literature.
//!
//! # Features
//!
//! - **Fibonacci Heap**: O(1) amortized insert, decrease_key, and merge; O(log n) amortized delete-min
//! - **Pairing Heap**: O(1) amortized insert and merge; O(log n) amortized delete-min; o(log n) amortized decrease_key
//! - **Rank-Pairing Heap**: O(1) amortized insert, decrease_key, and merge; O(log n) amortized delete-min
//! - **Binomial Heap**: O(log n) insert and delete-min; O(log n) decrease_key; O(1) amortized merge
//! - **Strict Fibonacci Heap**: O(1) worst-case insert, decrease_key, and merge; O(log n) worst-case delete-min
//! - **2-3 Heap**: O(1) amortized insert and decrease_key; O(log n) amortized delete-min
//! - **Skew Binomial Heap**: O(1) insert and merge; O(log n) delete-min and decrease_key
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::fibonacci::FibonacciHeap;
//! use rust_advanced_heaps::Heap;
//!
//! let mut heap = FibonacciHeap::new();
//! let handle1 = heap.insert(5, "item1");
//! let handle2 = heap.insert(3, "item2");
//! heap.decrease_key(&handle1, 1);
//! assert_eq!(heap.find_min(), Some((&1, &"item1")));
//! ```

pub mod binomial;
pub mod fibonacci;
pub mod pairing;
pub mod pathfinding;
pub mod rank_pairing;
pub mod skew_binomial;
pub mod stdlib_compat;
pub mod strict_fibonacci;
pub mod traits;
pub mod twothree;

// Re-export the main trait for convenience
pub use traits::Heap;
