//! Simple Binary Heap implementation
//!
//! A straightforward binary min-heap implementation that only implements
//! the base [`Heap`] trait without `decrease_key` support.
//!
//! This is provided as a simpler alternative when you don't need `decrease_key`
//! operations. For algorithms requiring priority updates (like Dijkstra's),
//! use one of the advanced heaps like [`FibonacciHeap`](crate::fibonacci::FibonacciHeap).
//!
//! # Time Complexity
//!
//! | Operation | Complexity |
//! |-----------|------------|
//! | `push`    | O(log n)   |
//! | `pop`     | O(log n)   |
//! | `peek`    | O(1)       |
//! | `merge`   | O(n log n) |
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::Heap;
//! use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
//!
//! let mut heap = SimpleBinaryHeap::new();
//! heap.push(3, "three");
//! heap.push(1, "one");
//! heap.push(2, "two");
//!
//! assert_eq!(heap.peek(), Some((&1, &"one")));
//! assert_eq!(heap.pop(), Some((1, "one")));
//! assert_eq!(heap.pop(), Some((2, "two")));
//! assert_eq!(heap.pop(), Some((3, "three")));
//! assert_eq!(heap.pop(), None);
//! ```

use crate::traits::{Heap, MergeableHeap};

/// A simple binary min-heap
///
/// This heap stores (priority, item) pairs and always returns the element
/// with the minimum priority first.
///
/// Unlike the advanced heaps in this crate, `SimpleBinaryHeap` does not
/// support `decrease_key` operations. If you need to update priorities
/// of elements already in the heap, use [`FibonacciHeap`](crate::fibonacci::FibonacciHeap)
/// or another advanced heap implementation.
#[derive(Debug)]
pub struct SimpleBinaryHeap<T, P: Ord> {
    /// The heap data stored as a vector of (priority, item) pairs
    data: Vec<(P, T)>,
}

impl<T, P: Ord> Heap<T, P> for SimpleBinaryHeap<T, P> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn push(&mut self, priority: P, item: T) {
        self.data.push((priority, item));
        self.sift_up(self.data.len() - 1);
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.data.first().map(|(p, t)| (p, t))
    }

    fn pop(&mut self) -> Option<(P, T)> {
        if self.data.is_empty() {
            return None;
        }

        let last_idx = self.data.len() - 1;
        self.data.swap(0, last_idx);
        let result = self.data.pop();

        if !self.data.is_empty() {
            self.sift_down(0);
        }

        result
    }
}

impl<T, P: Ord> MergeableHeap<T, P> for SimpleBinaryHeap<T, P> {
    fn merge(&mut self, other: Self) {
        // Simple merge: push all elements and let sift_up handle ordering
        // This is O(n log n) but could be optimized to O(n) with heapify
        for (priority, item) in other.data {
            self.push(priority, item);
        }
    }
}

impl<T, P: Ord> SimpleBinaryHeap<T, P> {
    /// Move element at index up to maintain heap property
    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.data[index].0 < self.data[parent].0 {
                self.data.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    /// Move element at index down to maintain heap property
    fn sift_down(&mut self, mut index: usize) {
        let len = self.data.len();
        loop {
            let left = 2 * index + 1;
            let right = 2 * index + 2;
            let mut smallest = index;

            if left < len && self.data[left].0 < self.data[smallest].0 {
                smallest = left;
            }
            if right < len && self.data[right].0 < self.data[smallest].0 {
                smallest = right;
            }

            if smallest != index {
                self.data.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }
}

impl<T, P: Ord> Default for SimpleBinaryHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = SimpleBinaryHeap::new();

        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);

        heap.push(3, "three");
        heap.push(1, "one");
        heap.push(2, "two");

        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some((&1, &"one")));

        assert_eq!(heap.pop(), Some((1, "one")));
        assert_eq!(heap.pop(), Some((2, "two")));
        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_duplicate_priorities() {
        let mut heap = SimpleBinaryHeap::new();

        heap.push(1, "a");
        heap.push(1, "b");
        heap.push(1, "c");

        assert_eq!(heap.len(), 3);

        // All three should pop with priority 1
        let (p1, _) = heap.pop().unwrap();
        let (p2, _) = heap.pop().unwrap();
        let (p3, _) = heap.pop().unwrap();

        assert_eq!(p1, 1);
        assert_eq!(p2, 1);
        assert_eq!(p3, 1);
    }

    #[test]
    fn test_merge() {
        let mut heap1 = SimpleBinaryHeap::new();
        let mut heap2 = SimpleBinaryHeap::new();

        heap1.push(3, "three");
        heap1.push(1, "one");

        heap2.push(4, "four");
        heap2.push(2, "two");

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.pop(), Some((1, "one")));
        assert_eq!(heap1.pop(), Some((2, "two")));
        assert_eq!(heap1.pop(), Some((3, "three")));
        assert_eq!(heap1.pop(), Some((4, "four")));
    }

    #[test]
    fn test_ascending_insertion() {
        let mut heap = SimpleBinaryHeap::new();

        for i in 0..100 {
            heap.push(i, i);
        }

        for i in 0..100 {
            assert_eq!(heap.pop(), Some((i, i)));
        }
    }

    #[test]
    fn test_descending_insertion() {
        let mut heap = SimpleBinaryHeap::new();

        for i in (0..100).rev() {
            heap.push(i, i);
        }

        for i in 0..100 {
            assert_eq!(heap.pop(), Some((i, i)));
        }
    }
}
