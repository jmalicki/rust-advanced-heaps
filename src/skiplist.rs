//! Skip List Heap implementation
//!
//! A heap implementation backed by a skip list, which maintains elements in sorted order.
//!
//! # Time Complexity
//!
//! | Operation      | Complexity       |
//! |----------------|------------------|
//! | `push`         | O(log n)         |
//! | `pop`          | O(log n)         |
//! | `peek`         | O(1)             |
//! | `decrease_key` | O(log n + m)*    |
//! | `merge`        | O(n log n)       |
//!
//! *Where m is the number of duplicate `(priority, id)` pairs. In practice m=1
//! since IDs only collide after merge operations. Requires `T: Default`.
//!
//! # Trade-offs
//!
//! Compared to Fibonacci/Pairing heaps:
//! - Simpler implementation (wraps existing skiplist crate)
//! - Better cache locality
//! - O(log n) `decrease_key` instead of O(1) amortized
//! - O(n log n) merge instead of O(1)
//!
//! Compared to Binary heap:
//! - Supports `decrease_key` operation
//! - Similar complexity for basic operations
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
//! use rust_advanced_heaps::skiplist::SkipListHeap;
//!
//! let mut heap = SkipListHeap::new();
//! let handle = heap.push_with_handle(10, "item");
//! heap.decrease_key(&handle, 5).unwrap();
//! assert_eq!(heap.peek(), Some((&5, &"item")));
//! ```

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError};
use skiplist::OrderedSkipList;
use std::cell::Cell;
use std::cmp::Ordering;
use std::rc::Rc;

/// Internal entry stored in the skip list
///
/// Entries are ordered by (priority, id) to ensure stable ordering
/// even when priorities are equal.
struct Entry<T, P> {
    priority: Rc<Cell<P>>,
    id: u64,
    item: T,
}

impl<T, P: Ord + Copy> PartialEq for Entry<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.priority.get() == other.priority.get() && self.id == other.id
    }
}

impl<T, P: Ord + Copy> Eq for Entry<T, P> {}

impl<T, P: Ord + Copy> PartialOrd for Entry<T, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, P: Ord + Copy> Ord for Entry<T, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.get().cmp(&other.priority.get()) {
            Ordering::Equal => self.id.cmp(&other.id),
            other => other,
        }
    }
}

/// Handle to an element in a SkipListHeap
///
/// The handle stores the unique ID and a shared reference to the current priority,
/// allowing O(log n) lookup for `decrease_key` operations.
pub struct SkipListHandle<P> {
    id: u64,
    priority: Rc<Cell<P>>,
}

impl<P> Clone for SkipListHandle<P> {
    fn clone(&self) -> Self {
        SkipListHandle {
            id: self.id,
            priority: Rc::clone(&self.priority),
        }
    }
}

impl<P> PartialEq for SkipListHandle<P> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<P> Eq for SkipListHandle<P> {}

impl<P> std::fmt::Debug for SkipListHandle<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkipListHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl<P> Handle for SkipListHandle<P> {}

/// A heap backed by a skip list
///
/// This implementation wraps the `skiplist` crate's `OrderedSkipList` to provide
/// heap operations. Elements are stored as `(priority, id, item)` tuples and
/// maintained in sorted order.
///
/// # Type Parameters
///
/// - `T`: The item type stored in the heap
/// - `P`: The priority type, must implement `Ord` and `Copy`
pub struct SkipListHeap<T, P: Ord + Copy> {
    list: OrderedSkipList<Entry<T, P>>,
    next_id: u64,
}

impl<T, P: Ord + Copy> Heap<T, P> for SkipListHeap<T, P> {
    fn new() -> Self {
        Self {
            list: OrderedSkipList::new(),
            next_id: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    fn len(&self) -> usize {
        self.list.len()
    }

    fn push(&mut self, priority: P, item: T) {
        let id = self.next_id;
        self.next_id += 1;

        let entry = Entry {
            priority: Rc::new(Cell::new(priority)),
            id,
            item,
        };

        self.list.insert(entry);
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.list.front().map(|entry| {
            // SAFETY: We need to return a reference to the priority value.
            // The Cell contains a Copy type, so we can't get a reference directly.
            // We use a workaround by getting a pointer to the Rc's inner data.
            // This is safe because:
            // 1. The entry lives as long as &self
            // 2. The Rc keeps the Cell alive
            // 3. We're only reading, not modifying
            // 4. All mutations of priority go through methods requiring &mut self,
            //    so no aliasing &mut P can exist while this &P is held.
            let priority_ptr = entry.priority.as_ptr();
            unsafe { (&*priority_ptr, &entry.item) }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.list
            .pop_front()
            .map(|entry| (entry.priority.get(), entry.item))
    }

    fn merge(&mut self, other: Self) {
        // O(n log n) - insert each element from other into self
        for entry in other.list {
            self.list.insert(entry);
        }

        // Keep IDs monotonic within the merged heap to reduce future ID collisions.
        // This helps keep `m` small in `decrease_key`'s O(log n + m) bound.
        self.next_id = self.next_id.max(other.next_id);
    }
}

impl<T: Default, P: Ord + Copy> DecreaseKeyHeap<T, P> for SkipListHeap<T, P> {
    type Handle = SkipListHandle<P>;

    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        let id = self.next_id;
        self.next_id += 1;

        let priority_cell = Rc::new(Cell::new(priority));
        let handle = SkipListHandle {
            id,
            priority: Rc::clone(&priority_cell),
        };

        let entry = Entry {
            priority: priority_cell,
            id,
            item,
        };

        self.list.insert(entry);
        handle
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let old_priority = handle.priority.get();

        if new_priority >= old_priority {
            return Err(HeapError::PriorityNotDecreased);
        }

        // Create a probe entry to find the position via binary search.
        // Entry comparison only uses (priority, id), so the dummy item value doesn't matter.
        let probe = Entry {
            priority: Rc::new(Cell::new(old_priority)),
            id: handle.id,
            item: T::default(),
        };

        // O(log n) binary search to find the index
        let start_idx = self.list.index_of(&probe).ok_or(HeapError::InvalidHandle)?;

        // Scan from start_idx to find the exact entry matching our Rc pointer.
        // This handles the rare case of duplicate (priority, id) pairs after merge.
        // Time complexity: O(log n + m) where m is typically 1.
        let mut found_idx = None;
        for (offset, entry) in self.list.iter().skip(start_idx).enumerate() {
            // Stop if we've moved past entries with matching (priority, id)
            if entry.priority.get() != old_priority || entry.id != handle.id {
                break;
            }

            if Rc::ptr_eq(&entry.priority, &handle.priority) {
                found_idx = Some(start_idx + offset);
                break;
            }
        }

        let idx = found_idx.ok_or(HeapError::InvalidHandle)?;

        // Remove the entry
        let entry = self.list.remove_index(idx);

        // Update priority in both the entry and the handle's shared cell
        entry.priority.set(new_priority);

        // Reinsert with new priority
        self.list.insert(entry);

        Ok(())
    }
}

impl<T, P: Ord + Copy> Default for SkipListHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap: SkipListHeap<&str, i32> = SkipListHeap::new();

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
    fn test_decrease_key() {
        let mut heap: SkipListHeap<&str, i32> = SkipListHeap::new();

        let handle1 = heap.push_with_handle(10, "a");
        let handle2 = heap.push_with_handle(5, "b");
        let _handle3 = heap.push_with_handle(15, "c");

        assert_eq!(heap.peek(), Some((&5, &"b")));

        // Decrease key of handle1 from 10 to 2
        heap.decrease_key(&handle1, 2).unwrap();
        assert_eq!(heap.peek(), Some((&2, &"a")));

        // Decrease key of handle2 from 5 to 1
        heap.decrease_key(&handle2, 1).unwrap();
        assert_eq!(heap.peek(), Some((&1, &"b")));

        // Pop in order
        assert_eq!(heap.pop(), Some((1, "b")));
        assert_eq!(heap.pop(), Some((2, "a")));
        assert_eq!(heap.pop(), Some((15, "c")));
    }

    #[test]
    fn test_decrease_key_error() {
        let mut heap: SkipListHeap<&str, i32> = SkipListHeap::new();

        let handle = heap.push_with_handle(5, "item");

        // Try to "decrease" to a higher priority - should fail
        let result = heap.decrease_key(&handle, 10);
        assert_eq!(result, Err(HeapError::PriorityNotDecreased));

        // Try to "decrease" to same priority - should fail
        let result = heap.decrease_key(&handle, 5);
        assert_eq!(result, Err(HeapError::PriorityNotDecreased));

        // Original value should still be there
        assert_eq!(heap.peek(), Some((&5, &"item")));
    }

    #[test]
    fn test_merge() {
        let mut heap1: SkipListHeap<i32, i32> = SkipListHeap::new();
        let mut heap2: SkipListHeap<i32, i32> = SkipListHeap::new();

        heap1.push(3, 30);
        heap1.push(1, 10);

        heap2.push(4, 40);
        heap2.push(2, 20);

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.pop(), Some((1, 10)));
        assert_eq!(heap1.pop(), Some((2, 20)));
        assert_eq!(heap1.pop(), Some((3, 30)));
        assert_eq!(heap1.pop(), Some((4, 40)));
    }

    #[test]
    fn test_duplicate_priorities() {
        let mut heap: SkipListHeap<&str, i32> = SkipListHeap::new();

        heap.push(1, "a");
        heap.push(1, "b");
        heap.push(1, "c");

        assert_eq!(heap.len(), 3);

        // All should pop with priority 1
        let (p1, _) = heap.pop().unwrap();
        let (p2, _) = heap.pop().unwrap();
        let (p3, _) = heap.pop().unwrap();

        assert_eq!(p1, 1);
        assert_eq!(p2, 1);
        assert_eq!(p3, 1);
    }
}
