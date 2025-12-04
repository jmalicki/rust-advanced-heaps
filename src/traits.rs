//! Common traits for heap data structures
//!
//! This module provides a two-tier trait hierarchy for heap/priority queue data structures:
//!
//! - [`Heap`]: Base trait for simple heaps without `decrease_key` support
//! - [`DecreaseKeyHeap`]: Extended trait adding `decrease_key` and handle-based operations
//!
//! The base [`Heap`] trait is compatible with Rust's standard heap API patterns,
//! while [`DecreaseKeyHeap`] adds the advanced operations needed for algorithms
//! like Dijkstra's shortest path.

use std::fmt;

/// Error type for heap operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapError {
    /// The new priority is not less than the current priority
    PriorityNotDecreased,
    /// The handle is no longer valid (element was removed)
    InvalidHandle,
}

impl fmt::Display for HeapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeapError::PriorityNotDecreased => {
                write!(f, "new priority is not less than current priority")
            }
            HeapError::InvalidHandle => {
                write!(f, "handle is no longer valid (element was removed)")
            }
        }
    }
}

impl std::error::Error for HeapError {}

/// A handle to an element in the heap, used for decrease_key operations
///
/// This is an opaque type that identifies a specific element in the heap.
/// The exact implementation varies by heap type.
///
/// Note: Handles may be `Clone` but not necessarily `Copy`, depending on
/// the underlying implementation (e.g., reference-counted vs raw pointer).
pub trait Handle: Clone + PartialEq + Eq {}

/// Base trait for heap/priority queue data structures
///
/// This trait provides a simple API similar to Rust's `BinaryHeap`:
/// - `push` inserts an element (returns `()`)
/// - `pop` removes and returns the minimum
/// - `peek` returns the minimum without removing it
///
/// Unlike `BinaryHeap` which stores values directly (using `Ord`), these heaps
/// store (priority, item) pairs to separate the ordering key from the data.
///
/// For heaps that support `decrease_key` operations, see [`DecreaseKeyHeap`].
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::Heap;
/// use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
///
/// let mut heap = SimpleBinaryHeap::new();
/// heap.push(3, "three");
/// heap.push(1, "one");
/// heap.push(2, "two");
///
/// assert_eq!(heap.peek(), Some((&1, &"one")));
/// assert_eq!(heap.pop(), Some((1, "one")));
/// ```
pub trait Heap<T, P: Ord> {
    /// Creates a new empty heap
    fn new() -> Self;

    /// Returns true if the heap is empty
    fn is_empty(&self) -> bool;

    /// Returns the number of elements in the heap
    fn len(&self) -> usize;

    /// Inserts an element with the given priority
    ///
    /// # Time Complexity
    /// Typically O(log n) for simple heaps, O(1) amortized for advanced heaps.
    fn push(&mut self, priority: P, item: T);

    /// Returns the minimum priority and associated item without removing it
    ///
    /// Note that `BinaryHeap` is a max-heap, while these heaps are min-heaps.
    ///
    /// # Time Complexity
    /// O(1) for all implementations
    fn peek(&self) -> Option<(&P, &T)>;

    /// Removes and returns the minimum priority and associated item
    ///
    /// Note that `BinaryHeap` is a max-heap, while these heaps are min-heaps.
    ///
    /// # Time Complexity
    /// Typically O(log n) for all implementations.
    fn pop(&mut self) -> Option<(P, T)>;

    /// Merges another heap into this one, consuming the other heap
    ///
    /// # Time Complexity
    /// Varies by implementation: O(1) for Fibonacci/Pairing heaps, O(log n) for Binomial.
    fn merge(&mut self, other: Self);
}

/// Extended heap trait with `decrease_key` support
///
/// This trait extends [`Heap`] with operations that require tracking element handles:
/// - `push_with_handle` returns a handle that can be used with `decrease_key`
/// - `decrease_key` efficiently updates an element's priority
///
/// These operations are essential for algorithms like Dijkstra's shortest path
/// that need to update priorities of elements already in the heap.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
/// use rust_advanced_heaps::fibonacci::FibonacciHeap;
///
/// let mut heap = FibonacciHeap::new();
/// let handle = heap.push_with_handle(10, "item");
/// heap.decrease_key(&handle, 5).unwrap();
/// assert_eq!(heap.peek(), Some((&5, &"item")));
/// ```
pub trait DecreaseKeyHeap<T, P: Ord>: Heap<T, P> {
    /// The handle type for this heap, used to reference elements for decrease_key
    type Handle: Handle;

    /// Inserts an element with the given priority, returning a handle
    ///
    /// The handle can be used later with `decrease_key` to update the priority.
    ///
    /// # Time Complexity
    /// - Fibonacci Heap: O(1) amortized
    /// - Pairing Heap: O(1) amortized
    /// - Rank-Pairing Heap: O(1) amortized
    /// - Binomial Heap: O(log n)
    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle;

    /// Decreases the priority of an element identified by the handle
    ///
    /// This operation is not available in Rust's standard `BinaryHeap`.
    /// It allows efficient priority updates which are essential for algorithms
    /// like Dijkstra's shortest path.
    ///
    /// # Safety
    /// The handle must be valid (from a previous `push_with_handle` and not yet popped).
    /// Returns an error if the handle is invalid.
    ///
    /// # Errors
    /// Returns `HeapError::PriorityNotDecreased` if the new priority is not
    /// less than the current priority.
    ///
    /// # Time Complexity
    /// - Fibonacci Heap: O(1) amortized
    /// - Pairing Heap: o(log n) amortized
    /// - Rank-Pairing Heap: O(1) amortized
    /// - Binomial Heap: O(log n)
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError>;
}
