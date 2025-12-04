//! Radix Heap implementation
//!
//! A monotone priority queue optimized for Dijkstra's algorithm with integer edge weights.
//!
//! # Monotone Property
//!
//! A radix heap is a **monotone priority queue**: you cannot insert an element with
//! a priority smaller than the last extracted minimum. This constraint is naturally
//! satisfied by Dijkstra's algorithm with non-negative edge weights, since relaxed
//! distances are always `>= d[u]` where `d[u]` is the distance of the node just extracted.
//!
//! # Why Not Wrap the `radix-heap` Crate?
//!
//! The existing `radix-heap` crate (v0.4.2) is incompatible with our `Heap` trait:
//!
//! 1. **No decrease-key**: The crate provides no mechanism to update priorities.
//!    For Dijkstra, this would require external position tracking and remove+reinsert,
//!    negating the performance benefits.
//!
//! 2. **Max-heap orientation**: The crate implements a max-heap with the constraint
//!    that inserted keys must be `<= top()`. Our heaps are min-heaps.
//!
//! 3. **No merge support**: The crate lacks a `merge` operation required by our trait.
//!
//! 4. **Numeric keys only**: Requires `Radix + Ord + Copy` which limits keys to
//!    primitive integer types. We expose this constraint explicitly via `RadixKey`.
//!
//! This implementation provides a min-heap radix heap with native decrease-key support,
//! matching the API of our other heap implementations.
//!
//! # Time Complexity
//!
//! | Operation      | Complexity          |
//! |----------------|---------------------|
//! | `push`         | O(1)                |
//! | `pop`          | O(log C) amortized* |
//! | `peek`         | O(1) expected**     |
//! | `decrease_key` | O(k)***             |
//! | `merge`        | O(n)                |
//!
//! *Where C is the maximum difference between any key and the minimum key when inserted.
//! For Dijkstra with bounded edge weights, this is effectively O(1).
//!
//! **After `pop` redistributes elements, `peek` is O(1). Before any pop or when bucket 0
//! is empty, `peek` scans all buckets making it O(n) worst-case.
//!
//! ***Where k is the bucket size. In typical Dijkstra usage with well-distributed priorities,
//! k is small and this is effectively O(1). Worst case (all elements in one bucket) is O(n).
//!
//! # Cache Performance
//!
//! Radix heaps have excellent cache locality because:
//! - Buckets are contiguous vectors
//! - Most operations touch only 1-2 buckets
//! - No pointer chasing (unlike Fibonacci/Pairing heaps)
//!
//! Empirically, radix heaps are ~2x faster than binary heaps for Dijkstra on road networks.
//!
//! # References
//!
//! - Ahuja, R. K., Mehlhorn, K., Orlin, J. B., & Tarjan, R. E. (1990).
//!   "Faster algorithms for the shortest path problem."
//!   *Journal of the ACM*, 37(2), 213-223.
//!   [ACM DL](https://dl.acm.org/doi/10.1145/77600.77615)
//!
//! # Example
//!
//! ```rust
//! use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
//! use rust_advanced_heaps::radix::RadixHeap;
//!
//! let mut heap: RadixHeap<&str, u32> = RadixHeap::new();
//! let h1 = heap.push_with_handle(10, "ten");
//! let h2 = heap.push_with_handle(5, "five");
//!
//! assert_eq!(heap.peek(), Some((&5, &"five")));
//!
//! // Decrease key from 10 to 3
//! heap.decrease_key(&h1, 3).unwrap();
//! assert_eq!(heap.peek(), Some((&3, &"ten")));
//!
//! // Pop in order
//! assert_eq!(heap.pop(), Some((3, "ten")));
//! assert_eq!(heap.pop(), Some((5, "five")));
//! ```

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError, MergeableHeap};
use std::cell::Cell;
use std::rc::Rc;

/// Trait for keys that can be used in a radix heap.
///
/// This trait provides the bit-level operations needed for bucket assignment.
/// It is implemented for all unsigned integer types.
///
/// # Safety
///
/// Implementations must satisfy:
/// - `BITS` must equal the number of bits in the type
/// - `leading_zeros()` must return the number of leading zero bits
/// - `as_usize()` must be a lossless conversion for values up to `usize::MAX`
pub trait RadixKey: Ord + Copy + Default {
    /// Number of bits in this key type
    const BITS: u32;

    /// Returns the number of leading zeros in the binary representation
    fn leading_zeros(self) -> u32;

    /// Convert to usize (for indexing). May truncate for u128 on 32-bit platforms.
    fn as_usize(self) -> usize;

    /// Compute XOR of two keys (for finding differing high bit)
    fn bitxor(self, other: Self) -> Self;
}

macro_rules! impl_radix_key {
    ($($t:ty),+) => {
        $(
            impl RadixKey for $t {
                const BITS: u32 = <$t>::BITS;

                #[inline]
                fn leading_zeros(self) -> u32 {
                    <$t>::leading_zeros(self)
                }

                #[inline]
                fn as_usize(self) -> usize {
                    self as usize
                }

                #[inline]
                fn bitxor(self, other: Self) -> Self {
                    self ^ other
                }
            }
        )+
    };
}

impl_radix_key!(u8, u16, u32, u64, usize);

// u128 needs special handling on 32-bit platforms
impl RadixKey for u128 {
    const BITS: u32 = 128;

    #[inline]
    fn leading_zeros(self) -> u32 {
        u128::leading_zeros(self)
    }

    #[inline]
    fn as_usize(self) -> usize {
        // Truncates on 32-bit, but bucket indices are always small
        self as usize
    }

    #[inline]
    fn bitxor(self, other: Self) -> Self {
        self ^ other
    }
}

/// Internal entry stored in a bucket
struct Entry<T, P> {
    item: T,
    /// Shared priority cell - allows O(1) decrease-key by updating in place
    priority: Rc<Cell<P>>,
    /// Unique ID for handle equality
    id: u64,
}

/// Handle to an element in the radix heap
///
/// Used for `decrease_key` operations. The handle remains valid until
/// the element is popped from the heap.
pub struct RadixHandle<P> {
    priority: Rc<Cell<P>>,
    id: u64,
}

impl<P> Clone for RadixHandle<P> {
    fn clone(&self) -> Self {
        RadixHandle {
            priority: Rc::clone(&self.priority),
            id: self.id,
        }
    }
}

impl<P> PartialEq for RadixHandle<P> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<P> Eq for RadixHandle<P> {}

impl<P> std::fmt::Debug for RadixHandle<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RadixHandle").field("id", &self.id).finish()
    }
}

impl<P> Handle for RadixHandle<P> {}

/// A radix heap (monotone priority queue)
///
/// This is a min-heap optimized for Dijkstra's algorithm with integer priorities.
/// It exploits the monotone property: extracted minimums never decrease.
///
/// # Type Parameters
///
/// - `T`: The item type stored in the heap
/// - `P`: The priority type, must implement [`RadixKey`] (unsigned integers)
///
/// # Panics
///
/// - `push` will panic if called with a priority less than the last extracted minimum
///   (violating the monotone property)
pub struct RadixHeap<T, P: RadixKey> {
    /// Buckets indexed by the highest differing bit from `last_min`.
    /// Bucket 0 contains elements equal to `last_min`.
    /// Bucket i (1 <= i <= BITS) contains elements differing at bit (BITS - i).
    buckets: Vec<Vec<Entry<T, P>>>,

    /// The last extracted minimum (or 0 if nothing extracted yet)
    last_min: P,

    /// Total number of elements
    len: usize,

    /// Next unique ID for handles
    next_id: u64,
}

impl<T, P: RadixKey> RadixHeap<T, P> {
    /// Compute the bucket index for a priority value.
    ///
    /// Returns 0 if priority == last_min, otherwise returns (BITS - leading_zeros(priority XOR last_min)).
    #[inline]
    fn bucket_index(&self, priority: P) -> usize {
        let diff = priority.bitxor(self.last_min);
        // Compare against default (zero) directly instead of using as_usize(),
        // which would truncate u128 on 32-bit platforms and cause misclassification.
        if diff == P::default() {
            0
        } else {
            (P::BITS - diff.leading_zeros()) as usize
        }
    }

    /// Redistribute elements from a bucket into finer buckets.
    ///
    /// This is called when we need to pop but bucket 0 is empty.
    /// We find the smallest non-empty bucket, find its minimum,
    /// update `last_min`, and redistribute.
    fn redistribute(&mut self) {
        // Find smallest non-empty bucket > 0
        let src_bucket = match (1..self.buckets.len()).find(|&i| !self.buckets[i].is_empty()) {
            Some(i) => i,
            None => return, // All empty
        };

        // Find the minimum priority in this bucket
        let min_priority = self.buckets[src_bucket]
            .iter()
            .map(|e| e.priority.get())
            .min()
            .unwrap();

        // Update last_min
        self.last_min = min_priority;

        // Take all elements from the source bucket
        let entries: Vec<_> = self.buckets[src_bucket].drain(..).collect();

        // Redistribute to finer buckets
        for entry in entries {
            let new_bucket = self.bucket_index(entry.priority.get());
            self.buckets[new_bucket].push(entry);
        }
    }
}

impl<T, P: RadixKey> Heap<T, P> for RadixHeap<T, P> {
    fn new() -> Self {
        // We need BITS + 1 buckets: bucket 0 for exact matches,
        // buckets 1..=BITS for differences at each bit position
        let num_buckets = (P::BITS + 1) as usize;
        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(Vec::new());
        }

        RadixHeap {
            buckets,
            last_min: P::default(), // 0 for unsigned types
            len: 0,
            next_id: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) {
        // Monotone constraint: priority must be >= last_min
        assert!(
            priority >= self.last_min,
            "RadixHeap: cannot insert priority less than last extracted minimum (monotone violation)"
        );

        let bucket = self.bucket_index(priority);
        let entry = Entry {
            item,
            priority: Rc::new(Cell::new(priority)),
            id: self.next_id,
        };
        self.next_id += 1;

        self.buckets[bucket].push(entry);
        self.len += 1;
    }

    fn peek(&self) -> Option<(&P, &T)> {
        if self.is_empty() {
            return None;
        }

        // Check bucket 0 first (elements equal to last_min)
        if let Some(entry) = self.buckets[0].first() {
            // SAFETY: Same justification as SkipListHeap - we return references
            // tied to &self lifetime, and the Cell contains a Copy type.
            let priority_ptr = entry.priority.as_ptr();
            return Some(unsafe { (&*priority_ptr, &entry.item) });
        }

        // Find minimum across all buckets
        // This is O(n) in worst case but typically O(1) after redistribute
        let mut min_entry: Option<&Entry<T, P>> = None;
        let mut min_priority: Option<P> = None;

        for bucket in &self.buckets {
            for entry in bucket {
                let p = entry.priority.get();
                if min_priority.is_none() || p < min_priority.unwrap() {
                    min_priority = Some(p);
                    min_entry = Some(entry);
                }
            }
        }

        min_entry.map(|entry| {
            let priority_ptr = entry.priority.as_ptr();
            unsafe { (&*priority_ptr, &entry.item) }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        if self.is_empty() {
            return None;
        }

        // If bucket 0 is empty, redistribute
        if self.buckets[0].is_empty() {
            self.redistribute();
        }

        // Now bucket 0 should have the minimum (or we redistribute until it does)
        // After redistribute, all elements with priority == new last_min are in bucket 0

        // Pop any element from bucket 0 (they all have the same priority: last_min)
        if let Some(entry) = self.buckets[0].pop() {
            self.len -= 1;
            let priority = entry.priority.get();
            // last_min stays the same since we popped something == last_min
            return Some((priority, entry.item));
        }

        // This shouldn't happen if len > 0, but handle gracefully
        None
    }
}

impl<T, P: RadixKey> MergeableHeap<T, P> for RadixHeap<T, P> {
    fn merge(&mut self, other: Self) {
        // O(n) merge: iterate through other's buckets and insert
        for bucket in other.buckets {
            for entry in bucket {
                let priority = entry.priority.get();
                // Check monotone constraint
                assert!(
                    priority >= self.last_min,
                    "RadixHeap::merge: other heap contains priority less than last extracted minimum"
                );
                let new_bucket = self.bucket_index(priority);
                self.buckets[new_bucket].push(entry);
                self.len += 1;
            }
        }
        self.next_id = self.next_id.max(other.next_id);
    }
}

impl<T, P: RadixKey> DecreaseKeyHeap<T, P> for RadixHeap<T, P> {
    type Handle = RadixHandle<P>;

    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        // Monotone constraint
        assert!(
            priority >= self.last_min,
            "RadixHeap: cannot insert priority less than last extracted minimum (monotone violation)"
        );

        let priority_cell = Rc::new(Cell::new(priority));
        let handle = RadixHandle {
            priority: Rc::clone(&priority_cell),
            id: self.next_id,
        };

        let bucket = self.bucket_index(priority);
        let entry = Entry {
            item,
            priority: priority_cell,
            id: self.next_id,
        };
        self.next_id += 1;

        self.buckets[bucket].push(entry);
        self.len += 1;

        handle
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let old_priority = handle.priority.get();

        if new_priority >= old_priority {
            return Err(HeapError::PriorityNotDecreased);
        }

        // Monotone constraint: new priority must still be >= last_min
        if new_priority < self.last_min {
            // This would violate monotonicity - in Dijkstra this shouldn't happen
            // with non-negative weights, but we check anyway
            return Err(HeapError::PriorityNotDecreased);
        }

        // Find and move the entry
        let old_bucket = self.bucket_index(old_priority);
        let new_bucket = self.bucket_index(new_priority);

        // Find entry by handle ID in old bucket
        let entry_idx = self.buckets[old_bucket]
            .iter()
            .position(|e| e.id == handle.id)
            .ok_or(HeapError::InvalidHandle)?;

        // If bucket doesn't change, just update priority in place.
        // The entry.priority and handle.priority share the same Rc<Cell<P>>,
        // so updating entry.priority also updates handle.priority.
        if old_bucket == new_bucket {
            self.buckets[old_bucket][entry_idx]
                .priority
                .set(new_priority);
        } else {
            // Remove from old bucket, update priority, add to new bucket
            let entry = self.buckets[old_bucket].swap_remove(entry_idx);
            entry.priority.set(new_priority);
            self.buckets[new_bucket].push(entry);
        }

        Ok(())
    }
}

impl<T, P: RadixKey> Default for RadixHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);

        heap.push(3, "three");
        heap.push(1, "one");
        heap.push(2, "two");

        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 3);

        assert_eq!(heap.pop(), Some((1, "one")));
        assert_eq!(heap.pop(), Some((2, "two")));
        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_decrease_key() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        let h1 = heap.push_with_handle(10, "ten");
        let h2 = heap.push_with_handle(5, "five");
        let _h3 = heap.push_with_handle(15, "fifteen");

        // Decrease key from 10 to 3
        heap.decrease_key(&h1, 3).unwrap();

        // Decrease key from 5 to 2
        heap.decrease_key(&h2, 2).unwrap();

        // Pop in order
        assert_eq!(heap.pop(), Some((2, "five")));
        assert_eq!(heap.pop(), Some((3, "ten")));
        assert_eq!(heap.pop(), Some((15, "fifteen")));
    }

    #[test]
    fn test_decrease_key_error() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        let handle = heap.push_with_handle(5, "item");

        // Try to increase priority - should fail
        let result = heap.decrease_key(&handle, 10);
        assert_eq!(result, Err(HeapError::PriorityNotDecreased));

        // Try same priority - should fail
        let result = heap.decrease_key(&handle, 5);
        assert_eq!(result, Err(HeapError::PriorityNotDecreased));
    }

    #[test]
    fn test_monotone_property() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        heap.push(10, "ten");
        heap.push(5, "five");

        // Pop minimum (5)
        assert_eq!(heap.pop(), Some((5, "five")));

        // Now we can still push values >= 5
        heap.push(7, "seven");
        heap.push(5, "five again");

        assert_eq!(heap.pop(), Some((5, "five again")));
        assert_eq!(heap.pop(), Some((7, "seven")));
        assert_eq!(heap.pop(), Some((10, "ten")));
    }

    #[test]
    #[should_panic(expected = "monotone violation")]
    fn test_monotone_violation_panics() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        heap.push(10, "ten");
        assert_eq!(heap.pop(), Some((10, "ten")));

        // This should panic: trying to insert 5 after extracting 10
        heap.push(5, "five");
    }

    #[test]
    fn test_dijkstra_pattern() {
        // Simulate Dijkstra's algorithm pattern
        let mut heap: RadixHeap<u32, u32> = RadixHeap::new();

        // Insert source with distance 0
        let _h0 = heap.push_with_handle(0, 0);

        // Insert neighbors with initial distances
        let h1 = heap.push_with_handle(10, 1);
        let h2 = heap.push_with_handle(5, 2);
        let _h3 = heap.push_with_handle(u32::MAX, 3);

        // Extract minimum (node 0, distance 0)
        assert_eq!(heap.pop(), Some((0, 0)));

        // Relax edge 0->1: new distance = 0 + 3 = 3 < 10
        heap.decrease_key(&h1, 3).unwrap();

        // Extract minimum (node 1, distance 3)
        assert_eq!(heap.pop(), Some((3, 1)));

        // Relax edge 1->2: new distance = 3 + 1 = 4 < 5
        heap.decrease_key(&h2, 4).unwrap();

        // Extract node 2
        assert_eq!(heap.pop(), Some((4, 2)));

        // The handle h0 is now invalid (already popped), but we shouldn't use it
        // This is the expected Dijkstra usage pattern
    }

    #[test]
    fn test_merge() {
        let mut heap1: RadixHeap<i32, u32> = RadixHeap::new();
        let mut heap2: RadixHeap<i32, u32> = RadixHeap::new();

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
    fn test_different_int_types() {
        // u8
        let mut heap_u8: RadixHeap<&str, u8> = RadixHeap::new();
        heap_u8.push(5, "five");
        heap_u8.push(3, "three");
        assert_eq!(heap_u8.pop(), Some((3, "three")));

        // u64
        let mut heap_u64: RadixHeap<&str, u64> = RadixHeap::new();
        heap_u64.push(1_000_000_000_000, "trillion");
        heap_u64.push(1_000_000, "million");
        assert_eq!(heap_u64.pop(), Some((1_000_000, "million")));

        // usize
        let mut heap_usize: RadixHeap<&str, usize> = RadixHeap::new();
        heap_usize.push(100, "hundred");
        heap_usize.push(10, "ten");
        assert_eq!(heap_usize.pop(), Some((10, "ten")));
    }

    #[test]
    fn test_peek() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        assert_eq!(heap.peek(), None);

        heap.push(5, "five");
        heap.push(3, "three");
        heap.push(7, "seven");

        assert_eq!(heap.peek(), Some((&3, &"three")));

        // Peek doesn't remove
        assert_eq!(heap.peek(), Some((&3, &"three")));
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn test_equal_priorities() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();

        heap.push(5, "a");
        heap.push(5, "b");
        heap.push(5, "c");

        assert_eq!(heap.len(), 3);

        // All should pop with priority 5 (order within same priority is arbitrary)
        let (p1, _) = heap.pop().unwrap();
        let (p2, _) = heap.pop().unwrap();
        let (p3, _) = heap.pop().unwrap();

        assert_eq!(p1, 5);
        assert_eq!(p2, 5);
        assert_eq!(p3, 5);
    }
}
