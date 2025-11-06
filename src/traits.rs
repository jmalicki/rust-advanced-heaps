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
/// # Safety Warning
///
/// **Handles are NOT memory safe by design**. Using a handle after the element
/// has been popped from the heap results in **use-after-free** and undefined behavior.
/// The handle may point to freed memory, and dereferencing it will cause memory
/// safety violations.
///
/// For a memory-safe API, use the safe wrapper types that track handle validity.
pub trait Handle: Clone + Copy + PartialEq + Eq + std::hash::Hash {}

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
    /// # ⚠️ Memory Safety Warning
    ///
    /// **This function is NOT memory safe**. Using a handle after the element has been
    /// popped from the heap results in **use-after-free** and undefined behavior.
    ///
    /// ## The Problem
    ///
    /// Handles are implemented as raw pointers to internal heap nodes. When an element
    /// is popped via `pop()` or `delete_min()`, the node is freed immediately. However,
    /// the handle still points to that freed memory. If you then call `decrease_key()`
    /// with that handle, the function will dereference freed memory, causing:
    ///
    /// - **Use-after-free**: Accessing memory that has been deallocated
    /// - **Undefined behavior**: The program may crash, corrupt memory, or behave unpredictably
    /// - **Security vulnerabilities**: In some cases, this can lead to security issues
    ///
    /// ## Example of Unsafe Usage
    ///
    /// ```rust,no_run
    /// # use rust_advanced_heaps::fibonacci::FibonacciHeap;
    /// # use rust_advanced_heaps::Heap;
    /// let mut heap = FibonacciHeap::new();
    /// let handle = heap.push(5, "item");
    /// heap.pop();  // Element is freed, handle now points to freed memory
    /// heap.decrease_key(&handle, 1);  // ❌ UNSAFE: Use-after-free!
    /// ```
    ///
    /// ## Safe Alternatives
    ///
    /// For memory-safe usage, consider:
    ///
    /// 1. **Track handle validity yourself**: Maintain a set of valid handles and remove
    ///    them when elements are popped. This is error-prone.
    ///
    /// 2. **Use a safe wrapper**: A wrapper type that tracks valid handles automatically
    ///    (not yet implemented in this crate).
    ///
    /// 3. **Avoid using handles after pop**: Only use handles for elements that you know
    ///    are still in the heap. This requires careful program design.
    ///
    /// ## Why This Design?
    ///
    /// The handle-based API is designed for maximum performance and matches the interface
    /// used in academic literature. Implementing memory-safe handles would require:
    ///
    /// - Tracking valid handles (HashSet overhead)
    /// - Reference counting (Rc overhead)
    /// - Or other validation mechanisms
    ///
    /// These overheads would impact the O(1) amortized operations that make these heaps
    /// attractive. The current design prioritizes performance over safety, leaving it to
    /// users to ensure handles are only used while elements are still in the heap.
    ///
    /// # Safety Requirements
    ///
    /// The handle must be valid (from a previous `push` and the element must not have
    /// been popped). Behavior is **undefined** if the handle is invalid.
    ///
    /// # Errors
    ///
    /// Returns `HeapError::PriorityNotDecreased` if the new priority is not
    /// less than the current priority.
    ///
    /// # Time Complexity
    ///
    /// - Fibonacci Heap: O(1) amortized
    /// - Pairing Heap: o(log n) amortized
    /// - Rank-Pairing Heap: O(1) amortized
    /// - Binomial Heap: O(log n)
    /// - Brodal Heap: O(1) worst-case
    ///
    /// # Safety Contract
    ///
    /// This function has an **unsafe contract**: it can cause use-after-free if the handle
    /// is invalid. However, **implementations can provide safe wrappers** by validating
    /// handle validity internally. If an implementation validates handle validity internally,
    /// it should document that it's safe to call.
    ///
    /// The unsafe contract here means that implementations must either:
    /// 1. Ensure the handle is valid (making it safe to call)
    /// 2. Return an error/panic if the handle is invalid (making it safe to call)
    /// 3. Document that callers must ensure handle validity (unsafe to call)
    ///
    /// Current implementations are **NOT memory safe** and will cause use-after-free
    /// if used with invalid handles. Use caution when calling this function.
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

    /// Drains all elements from the heap, returning them in sorted order
    ///
    /// This repeatedly pops all elements from the heap until it is empty,
    /// returning them as a vector. Since the heap is a min-heap, the elements
    /// will be returned in non-decreasing priority order.
    ///
    /// # Time Complexity
    /// O(n log n) where n is the number of elements in the heap
    /// (same as calling pop() n times)
    ///
    /// # Example
    /// ```
    /// use rust_advanced_heaps::fibonacci::FibonacciHeap;
    /// use rust_advanced_heaps::Heap;
    ///
    /// let mut heap = FibonacciHeap::new();
    /// heap.push(3, "c");
    /// heap.push(1, "a");
    /// heap.push(2, "b");
    ///
    /// let drained = heap.drain();
    /// assert_eq!(drained, vec![(1, "a"), (2, "b"), (3, "c")]);
    /// assert!(heap.is_empty());
    /// ```
    #[inline]
    fn drain(&mut self) -> Vec<(P, T)> {
        let mut result = Vec::with_capacity(self.len());
        while let Some(elem) = self.pop() {
            result.push(elem);
        }
        result
    }
}
