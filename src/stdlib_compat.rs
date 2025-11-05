//! Standard library compatibility layer
//!
//! Provides a drop-in replacement for `std::collections::BinaryHeap`
//! that supports `decrease_key` operations.
//!
//! # Differences from BinaryHeap
//!
//! - **Min-heap vs Max-heap**: This is a min-heap, while `BinaryHeap` is a max-heap.
//!   Use `std::cmp::Reverse<T>` to get max-heap behavior.
//! - **decrease_key support**: Unlike `BinaryHeap`, this supports efficient `decrease_key`
//!   operations via handles.
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::stdlib_compat::StdHeap;
//! use rust_advanced_heaps::fibonacci::FibonacciHeap;
//!
//! // Use like std::collections::BinaryHeap
//! let mut heap: StdHeap<i32, FibonacciHeap<i32, i32>> = StdHeap::new();
//! heap.push(5);
//! heap.push(3);
//! heap.push(7);
//! assert_eq!(heap.peek(), Some(&3)); // min-heap, unlike BinaryHeap's max-heap
//! assert_eq!(heap.pop(), Some(3));
//!
//! // But with decrease_key support!
//! let handle = heap.push_with_handle(10);
//! heap.decrease_key(&handle.0, 1);
//! assert_eq!(heap.peek(), Some(&1));
//! ```

use crate::Heap;

/// A drop-in replacement for `std::collections::BinaryHeap` with `decrease_key` support
///
/// When the item type `T` implements `Ord`, you can use this wrapper to get
/// a `BinaryHeap`-like API where the item itself serves as the priority.
///
/// # Type Parameters
/// - `T`: The item type, must implement `Ord` and `Clone` (for handles)
/// - `H`: The underlying heap implementation (e.g., `FibonacciHeap<T, T>`)
pub struct StdHeap<T: Ord + Clone, H: Heap<T, T>> {
    heap: H,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Ord + Clone, H: Heap<T, T>> StdHeap<T, H> {
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
    ///
    /// Returns a handle that can be used with `decrease_key`. If you don't need
    /// `decrease_key`, you can ignore the return value.
    pub fn push(&mut self, item: T) -> H::Handle {
        // Since T = P, we use the item as both priority and value
        self.heap.push(item.clone(), item)
    }

    /// Pushes an item and returns both a handle and the item
    ///
    /// Useful when you need to store the handle for later `decrease_key` operations.
    pub fn push_with_handle(&mut self, item: T) -> (H::Handle, T) {
        let handle = self.push(item.clone());
        (handle, item)
    }

    /// Returns a reference to the smallest item without removing it
    ///
    /// This is equivalent to `BinaryHeap::peek`, but returns the minimum (not maximum).
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|(_, item)| item)
    }

    /// Removes and returns the smallest item
    ///
    /// This is equivalent to `BinaryHeap::pop`, but returns the minimum (not maximum).
    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|(_, item)| item)
    }

    /// Decreases the priority of an item identified by the handle
    ///
    /// This operation is not available in `BinaryHeap`.
    ///
    /// # Safety
    /// The handle must be valid (from a previous `push` and not yet popped).
    /// The new priority must be less than the current priority.
    pub fn decrease_key(
        &mut self,
        handle: &H::Handle,
        new_priority: T,
    ) -> Result<(), crate::traits::HeapError> {
        self.heap.decrease_key(handle, new_priority)
    }
}

impl<T: Ord + Clone, H: Heap<T, T>> Default for StdHeap<T, H> {
    fn default() -> Self {
        Self::new()
    }
}
