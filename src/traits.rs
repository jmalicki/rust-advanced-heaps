//! Common traits for heap data structures
//!
//! This module provides traits compatible with Rust's standard heap API while
//! adding support for efficient `decrease_key` operations.

use std::fmt;

/// Error type for heap operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapError {
    /// The new priority is not less than the current priority
    PriorityNotDecreased,
}

impl fmt::Display for HeapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeapError::PriorityNotDecreased => {
                write!(f, "new priority is not less than current priority")
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
/// Note: Handles implement `Clone` to allow duplication. Some implementations
/// may also implement `Copy` for convenience, but it is not required.
pub trait Handle: Clone + PartialEq + Eq {}

/// Common operations for heap/priority queue data structures
///
/// This trait provides an API similar to Rust's `BinaryHeap`, with the addition
/// of `decrease_key` support. All methods follow standard Rust naming conventions:
/// - `push` (standard Rust heap method)
/// - `pop` (standard Rust heap method)
/// - `peek` (standard Rust heap method)
///
/// Unlike `BinaryHeap` which stores values directly (using `Ord`), these heaps
/// store (priority, item) pairs to support efficient `decrease_key` operations.
pub trait Heap<T, P: Ord> {
    /// The handle type for this heap, used to reference elements for decrease_key
    type Handle: Handle;

    /// Creates a new empty heap
    fn new() -> Self;

    /// Returns true if the heap is empty
    fn is_empty(&self) -> bool;

    /// Returns the number of elements in the heap
    fn len(&self) -> usize;

    /// Inserts an element with the given priority, returning a handle
    ///
    /// Equivalent to `BinaryHeap::push`, but returns a handle for `decrease_key`.
    /// The handle can be used later with `decrease_key` to update the priority.
    ///
    /// # Time Complexity
    /// - Fibonacci Heap: O(1) amortized
    /// - Pairing Heap: O(1) amortized
    /// - Rank-Pairing Heap: O(1) amortized
    /// - Binomial Heap: O(log n)
    /// - Brodal Heap: O(1) worst-case
    fn push(&mut self, priority: P, item: T) -> Self::Handle;

    /// Inserts an element with the given priority, returning a handle
    ///
    /// Alias for `push` for consistency with older API.
    /// Prefer using `push` for compatibility with standard Rust heaps.
    #[inline]
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        self.push(priority, item)
    }

    /// Returns the minimum priority and associated item without removing it
    ///
    /// Equivalent to `BinaryHeap::peek`. Note that `BinaryHeap` is a max-heap,
    /// while these heaps are min-heaps.
    ///
    /// # Time Complexity
    /// All implementations: O(1)
    fn peek(&self) -> Option<(&P, &T)>;

    /// Returns the minimum priority and associated item without removing it
    ///
    /// Alias for `peek` for consistency with older API.
    /// Prefer using `peek` for compatibility with standard Rust heaps.
    #[inline]
    fn find_min(&self) -> Option<(&P, &T)> {
        self.peek()
    }

    /// Removes and returns the minimum priority and associated item
    ///
    /// Equivalent to `BinaryHeap::pop`. Note that `BinaryHeap` is a max-heap,
    /// while these heaps are min-heaps.
    ///
    /// # Time Complexity
    /// - Fibonacci Heap: O(log n) amortized
    /// - Pairing Heap: O(log n) amortized
    /// - Rank-Pairing Heap: O(log n) amortized
    /// - Binomial Heap: O(log n)
    /// - Brodal Heap: O(log n) worst-case
    fn pop(&mut self) -> Option<(P, T)>;

    /// Removes and returns the minimum priority and associated item
    ///
    /// Alias for `pop` for consistency with older API.
    /// Prefer using `pop` for compatibility with standard Rust heaps.
    #[inline]
    fn delete_min(&mut self) -> Option<(P, T)> {
        self.pop()
    }

    /// Decreases the priority of an element identified by the handle
    ///
    /// This operation is not available in Rust's standard `BinaryHeap`.
    /// It allows efficient priority updates which are essential for algorithms
    /// like Dijkstra's shortest path.
    ///
    /// # Safety
    /// The handle must be valid (from a previous `push` and not yet popped).
    /// Behavior is undefined if the handle is invalid.
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
    /// - Brodal Heap: O(1) worst-case
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError>;

    /// Merges another heap into this one, consuming the other heap
    ///
    /// Similar to `BinaryHeap::append`, but consumes the other heap.
    ///
    /// # Time Complexity
    /// - Fibonacci Heap: O(1)
    /// - Pairing Heap: O(1)
    /// - Rank-Pairing Heap: O(1)
    /// - Binomial Heap: O(log n)
    /// - Brodal Heap: O(1) worst-case
    fn merge(&mut self, other: Self);
}
