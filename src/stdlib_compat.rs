//! Standard library compatibility layer
//!
//! Provides a drop-in replacement for `std::collections::BinaryHeap`.
//!
//! # Differences from BinaryHeap
//!
//! - **Min-heap vs Max-heap**: This is a min-heap, while `BinaryHeap` is a max-heap.
//!   Use `std::cmp::Reverse<T>` to get max-heap behavior.
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::stdlib_compat::StdHeap;
//! use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
//!
//! // Use like std::collections::BinaryHeap
//! let mut heap: StdHeap<i32, SimpleBinaryHeap<(), i32>> = StdHeap::new();
//! heap.push(5);
//! heap.push(3);
//! heap.push(7);
//! assert_eq!(heap.peek(), Some(&3)); // min-heap, unlike BinaryHeap's max-heap
//! assert_eq!(heap.pop(), Some(3));
//! ```

use crate::traits::Heap;

/// A drop-in replacement for `std::collections::BinaryHeap`
///
/// When the item type `T` implements `Ord`, you can use this wrapper to get
/// a `BinaryHeap`-like API where the item itself serves as the priority.
///
/// # Type Parameters
/// - `T`: The item type, must implement `Ord`
/// - `H`: The underlying heap implementation (e.g., `SimpleBinaryHeap<(), T>`)
pub struct StdHeap<T: Ord, H: Heap<(), T>> {
    heap: H,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Ord, H: Heap<(), T>> StdHeap<T, H> {
    /// Creates a new empty heap
    pub fn new() -> Self {
        Self {
            heap: H::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns true if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the number of elements in the heap
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Pushes an item onto the heap
    ///
    /// The item itself serves as the priority.
    pub fn push(&mut self, item: T) {
        self.heap.push(item, ())
    }

    /// Returns a reference to the smallest item without removing it
    ///
    /// This is equivalent to `BinaryHeap::peek`, but returns the minimum (not maximum).
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|(priority, _)| priority)
    }

    /// Removes and returns the smallest item
    ///
    /// This is equivalent to `BinaryHeap::pop`, but returns the minimum (not maximum).
    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|(priority, _)| priority)
    }
}

impl<T: Ord, H: Heap<(), T>> Default for StdHeap<T, H> {
    fn default() -> Self {
        Self::new()
    }
}
