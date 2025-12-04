//! Strict Fibonacci Heap implementation
//!
//! # Current Implementation Status
//!
//! **IMPORTANT**: This implementation does NOT yet achieve the worst-case bounds
//! described in the Brodal-Lagogiannis-Tarjan paper. It currently provides the
//! same **amortized** bounds as a standard Fibonacci heap:
//!
//! | Operation | Current Bound | Target (Paper) |
//! |-----------|---------------|----------------|
//! | insert | O(1) amortized | O(1) worst-case |
//! | peek | O(1) worst-case | O(1) worst-case |
//! | pop | O(log n) amortized | O(log n) worst-case |
//! | decrease_key | O(1) amortized | O(1) worst-case |
//! | merge | O(1) amortized | O(1) worst-case |
//!
//! This implementation is being incrementally refactored toward the full
//! strict Fibonacci heap algorithm. See GitHub issue #42 for progress.
//!
//! # Implementation Notes
//!
//! This implementation uses:
//! - Raw pointers (`NonNull<Node>`) for node management
//! - Intrusive circular doubly-linked lists for sibling relationships
//! - A `rank` field on each node (preparation for future phases)
//!
//! # References
//!
//! - Brodal, G. S., Lagogiannis, G., & Tarjan, R. E. (2012). "Strict Fibonacci heaps."
//!   *Proceedings of the 44th Annual ACM Symposium on Theory of Computing (STOC)*, 1177-1184.
//!   [ACM DL](https://dl.acm.org/doi/10.1145/2213977.2214082)

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError};
use intrusive_circular_list::{container_of_mut, CircularLink, CircularListOps};
use std::ptr::NonNull;

/// Handle to an element in a Strict Fibonacci heap.
///
/// # Safety
///
/// The handle becomes invalid after the element is removed from the heap via `pop()`.
/// Using an invalid handle with `decrease_key` is undefined behavior (use-after-free).
/// Callers must ensure handles are not used after their element has been popped.
pub struct StrictFibonacciHandle<T, P> {
    node: Option<NonNull<Node<T, P>>>,
}

impl<T, P> Clone for StrictFibonacciHandle<T, P> {
    fn clone(&self) -> Self {
        StrictFibonacciHandle { node: self.node }
    }
}

impl<T, P> PartialEq for StrictFibonacciHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<T, P> Eq for StrictFibonacciHandle<T, P> {}

impl<T, P> std::fmt::Debug for StrictFibonacciHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrictFibonacciHandle")
            .field("valid", &self.node.is_some())
            .finish()
    }
}

impl<T, P> Handle for StrictFibonacciHandle<T, P> {}

/// A node in the Strict Fibonacci heap.
///
/// Uses intrusive circular linked lists for sibling relationships.
struct Node<T, P> {
    /// The item stored in this node.
    item: T,
    /// The priority of this node (smaller = higher priority for min-heap).
    priority: P,
    /// Intrusive link for the sibling circular list.
    sibling_link: CircularLink,
    /// Pointer to the parent node (None if this is a root).
    parent: Option<NonNull<Node<T, P>>>,
    /// Pointer to one child (entry point into children's circular list).
    child: Option<NonNull<Node<T, P>>>,
    /// The rank of this node (number of children).
    /// In strict Fibonacci heaps, this is used for consolidation.
    rank: usize,
    /// Whether this node is "active" in the strict Fibonacci sense.
    /// Active nodes are tracked specially to maintain worst-case bounds.
    active: bool,
}

impl<T, P> Node<T, P> {
    /// Creates a new node with the given priority and item.
    fn new(priority: P, item: T) -> Box<Node<T, P>> {
        Box::new(Node {
            item,
            priority,
            sibling_link: CircularLink::new(),
            parent: None,
            child: None,
            rank: 0,
            active: false,
        })
    }

    /// Gets a pointer to the sibling link.
    fn sibling_link_ptr(&self) -> NonNull<CircularLink> {
        NonNull::from(&self.sibling_link)
    }
}

/// Strict Fibonacci Heap.
///
/// A priority queue with efficient decrease_key operation.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
///
/// let mut heap = StrictFibonacciHeap::new();
/// let handle = heap.push_with_handle(5, "item");
/// heap.decrease_key(&handle, 1).unwrap();
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct StrictFibonacciHeap<T, P: Ord> {
    /// Entry point into the circular list of roots (if any).
    roots: Option<NonNull<Node<T, P>>>,
    /// Pointer to the minimum root.
    min: Option<NonNull<Node<T, P>>>,
    /// Number of elements in the heap.
    len: usize,
}

impl<T, P: Ord> Default for StrictFibonacciHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: StrictFibonacciHeap can be sent between threads if T and P can.
// The raw pointers are owned by the heap.
unsafe impl<T: Send, P: Ord + Send> Send for StrictFibonacciHeap<T, P> {}

impl<T, P: Ord> Drop for StrictFibonacciHeap<T, P> {
    fn drop(&mut self) {
        // We need to deallocate all nodes.
        // Traverse all roots and their descendants.
        if let Some(root_ptr) = self.roots {
            let mut to_free: Vec<NonNull<Node<T, P>>> = Vec::new();
            let ops = CircularListOps::new();

            // Collect all roots
            unsafe {
                let start_link = root_ptr.as_ref().sibling_link_ptr();
                ops.for_each(start_link, |link_ptr| {
                    let node_ptr: *mut Node<T, P> =
                        container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                    to_free.push(NonNull::new_unchecked(node_ptr));
                });
            }

            // Process nodes, collecting children
            let mut i = 0;
            while i < to_free.len() {
                let node_ptr = to_free[i];
                unsafe {
                    let node = node_ptr.as_ref();
                    if let Some(child_ptr) = node.child {
                        let start_link = child_ptr.as_ref().sibling_link_ptr();
                        ops.for_each(start_link, |link_ptr| {
                            let child_node_ptr: *mut Node<T, P> =
                                container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                            to_free.push(NonNull::new_unchecked(child_node_ptr));
                        });
                    }
                }
                i += 1;
            }

            // Free all nodes
            for node_ptr in to_free {
                unsafe {
                    drop(Box::from_raw(node_ptr.as_ptr()));
                }
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for StrictFibonacciHeap<T, P> {
    fn new() -> Self {
        Self {
            roots: None,
            min: None,
            len: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) {
        let _ = self.push_with_handle(priority, item);
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.min.map(|min_ptr| unsafe {
            let node = min_ptr.as_ref();
            (&node.priority, &node.item)
        })
    }

    /// Removes and returns the minimum element.
    ///
    /// **Time Complexity**: O(log n) amortized (target: O(log n) worst-case)
    fn pop(&mut self) -> Option<(P, T)> {
        let min_ptr = self.min?;

        // Remove min from root list
        self.remove_from_roots(min_ptr);

        // Add all children of min to root list
        unsafe {
            let min_node = min_ptr.as_ref();
            if let Some(child_ptr) = min_node.child {
                self.splice_children_to_roots(child_ptr);
            }
        }

        self.len -= 1;

        // Consolidate the heap
        self.consolidate();

        self.debug_assert_invariants();

        // Extract the node's data
        unsafe {
            let node = Box::from_raw(min_ptr.as_ptr());
            Some((node.priority, node.item))
        }
    }

    /// Merges another heap into this heap.
    ///
    /// **Time Complexity**: O(1) amortized (target: O(1) worst-case)
    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        // Splice the root lists together
        if let (Some(self_root), Some(other_root)) = (self.roots, other.roots) {
            let ops = CircularListOps::new();
            unsafe {
                let self_link = self_root.as_ref().sibling_link_ptr();
                let other_link = other_root.as_ref().sibling_link_ptr();
                ops.splice(Some(self_link), Some(other_link));
            }
        }

        // Update minimum
        if let (Some(self_min), Some(other_min)) = (self.min, other.min) {
            unsafe {
                if other_min.as_ref().priority < self_min.as_ref().priority {
                    self.min = Some(other_min);
                }
            }
        }

        self.len += other.len;

        // Prevent other from deallocating nodes
        other.roots = None;
        other.min = None;
        other.len = 0;

        self.debug_assert_invariants();
    }
}

impl<T, P: Ord> DecreaseKeyHeap<T, P> for StrictFibonacciHeap<T, P> {
    type Handle = StrictFibonacciHandle<T, P>;

    /// Inserts a new element into the heap.
    ///
    /// **Time Complexity**: O(1) amortized (target: O(1) worst-case)
    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Node::new(priority, item);
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        let handle = StrictFibonacciHandle {
            node: Some(node_ptr),
        };

        // Add to root list
        self.add_to_roots(node_ptr);

        // Update minimum if necessary
        self.update_min(node_ptr);

        self.len += 1;

        self.debug_assert_invariants();

        handle
    }

    /// Decreases the priority of an element.
    ///
    /// **Time Complexity**: O(1) amortized (target: O(1) worst-case)
    ///
    /// # Safety
    ///
    /// Callers must ensure the handle's node has not been popped.
    /// Using a handle after `pop()` removes its element is undefined behavior (use-after-free).
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node_ptr = handle.node.ok_or(HeapError::InvalidHandle)?;

        // Check and update priority
        unsafe {
            let node = node_ptr.as_ptr();

            if new_priority >= (*node).priority {
                return Err(HeapError::PriorityNotDecreased);
            }

            (*node).priority = new_priority;

            // Check if we need to cut from parent
            if let Some(parent_ptr) = (*node).parent {
                let parent = parent_ptr.as_ref();
                if (*node).priority < parent.priority {
                    self.cut(node_ptr);
                }
            }

            // Update minimum pointer
            self.update_min(node_ptr);
        }

        self.debug_assert_invariants();
        self.debug_assert_heap_property(node_ptr);

        Ok(())
    }
}

impl<T, P: Ord> StrictFibonacciHeap<T, P> {
    /// Debug assertion: checks O(1) structural invariants.
    ///
    /// Invariants checked:
    /// - `len == 0` iff `roots.is_none()` iff `min.is_none()`
    /// - If min exists, it has no parent (is a root)
    /// - If roots exist, the entry point has no parent
    #[inline]
    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        {
            // Emptiness consistency
            let roots_empty = self.roots.is_none();
            let min_empty = self.min.is_none();
            let len_zero = self.len == 0;

            debug_assert_eq!(
                roots_empty, len_zero,
                "Invariant violation: roots.is_none() = {}, len == 0 = {}",
                roots_empty, len_zero
            );
            debug_assert_eq!(
                min_empty, len_zero,
                "Invariant violation: min.is_none() = {}, len == 0 = {}",
                min_empty, len_zero
            );

            // Min is a root (has no parent)
            if let Some(min_ptr) = self.min {
                unsafe {
                    debug_assert!(
                        min_ptr.as_ref().parent.is_none(),
                        "Invariant violation: min node has a parent (not a root)"
                    );
                }
            }

            // Root entry point has no parent
            if let Some(root_ptr) = self.roots {
                unsafe {
                    debug_assert!(
                        root_ptr.as_ref().parent.is_none(),
                        "Invariant violation: root entry point has a parent"
                    );
                }
            }
        }
    }

    /// Debug assertion: checks that a node satisfies heap property with its parent.
    #[inline]
    fn debug_assert_heap_property(&self, node_ptr: NonNull<Node<T, P>>) {
        #[cfg(debug_assertions)]
        unsafe {
            let node = node_ptr.as_ref();
            if let Some(parent_ptr) = node.parent {
                let parent = parent_ptr.as_ref();
                debug_assert!(
                    parent.priority <= node.priority,
                    "Heap property violation: parent priority > child priority"
                );
            }
        }
    }

    /// Adds a node to the root list.
    fn add_to_roots(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            (*node).parent = None;

            let node_link = (*node).sibling_link_ptr();

            match self.roots {
                None => {
                    // First root - make it a circular list of one
                    ops.make_circular(node_link);
                    self.roots = Some(node_ptr);
                }
                Some(root_ptr) => {
                    // Insert after the current root entry point
                    let root_link = root_ptr.as_ref().sibling_link_ptr();
                    ops.insert_after(root_link, node_link);
                }
            }
        }
    }

    /// Removes a node from the root list.
    fn remove_from_roots(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node_link = node_ptr.as_ref().sibling_link_ptr();

            // Check if this is the only root
            let next_link = ops.next(node_link);
            if next_link == Some(node_link) {
                // Only one root, list becomes empty
                ops.unlink(node_link);
                self.roots = None;
            } else {
                // Multiple roots
                // If we're removing the entry point, update it
                if self.roots == Some(node_ptr) {
                    // Get the next node as new entry point
                    let next_node_ptr: *mut Node<T, P> = container_of_mut!(
                        next_link.unwrap().as_ptr(),
                        Node<T, P>,
                        sibling_link
                    );
                    self.roots = Some(NonNull::new_unchecked(next_node_ptr));
                }
                ops.unlink(node_link);
            }
        }
    }

    /// Splices children of a node into the root list.
    fn splice_children_to_roots(&mut self, child_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        // Clear parent pointers for all children
        unsafe {
            let start_link = child_ptr.as_ref().sibling_link_ptr();
            ops.for_each(start_link, |link_ptr| {
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                (*node_ptr).parent = None;
            });

            // Splice with root list
            match self.roots {
                None => {
                    // No roots, children become roots
                    self.roots = Some(child_ptr);
                }
                Some(root_ptr) => {
                    let root_link = root_ptr.as_ref().sibling_link_ptr();
                    let child_link = child_ptr.as_ref().sibling_link_ptr();
                    ops.splice(Some(root_link), Some(child_link));
                }
            }
        }
    }

    /// Updates the minimum pointer if the given node has smaller priority.
    fn update_min(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let should_update = match self.min {
            None => true,
            Some(min_ptr) => unsafe { node_ptr.as_ref().priority < min_ptr.as_ref().priority },
        };

        if should_update {
            self.min = Some(node_ptr);
        }
    }

    /// Finds and sets the new minimum by scanning all roots.
    fn find_new_min(&mut self) {
        self.min = None;

        if let Some(root_ptr) = self.roots {
            let ops = CircularListOps::new();

            unsafe {
                let start_link = root_ptr.as_ref().sibling_link_ptr();
                ops.for_each(start_link, |link_ptr| {
                    let node_ptr: *mut Node<T, P> =
                        container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                    let node_nn = NonNull::new_unchecked(node_ptr);

                    let should_update = match self.min {
                        None => true,
                        Some(min_ptr) => (*node_ptr).priority < min_ptr.as_ref().priority,
                    };

                    if should_update {
                        self.min = Some(node_nn);
                    }
                });
            }
        }
    }

    /// Consolidates the heap by linking trees of the same rank.
    ///
    /// This is called after pop to maintain the heap structure.
    fn consolidate(&mut self) {
        if self.roots.is_none() {
            self.min = None;
            return;
        }

        // Calculate max possible rank (log base phi of n)
        let max_rank = if self.len == 0 {
            1
        } else {
            ((self.len as f64).log2() * 1.5) as usize + 2
        };

        let mut rank_table: Vec<Option<NonNull<Node<T, P>>>> = vec![None; max_rank + 1];
        let ops = CircularListOps::new();

        // Collect all roots into a vector first
        let mut roots_vec: Vec<NonNull<Node<T, P>>> = Vec::new();
        unsafe {
            let root_ptr = self.roots.unwrap();
            let start_link = root_ptr.as_ref().sibling_link_ptr();
            ops.for_each(start_link, |link_ptr| {
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                roots_vec.push(NonNull::new_unchecked(node_ptr));
            });
        }

        // Unlink all roots (they'll be re-linked after consolidation)
        for &node_ptr in &roots_vec {
            unsafe {
                let node_link = node_ptr.as_ref().sibling_link_ptr();
                node_link.as_ref().force_unlink();
            }
        }
        self.roots = None;

        // Process each root
        for root in roots_vec {
            let mut current = root;

            loop {
                let rank = unsafe { current.as_ref().rank };

                if rank >= rank_table.len() {
                    rank_table.resize(rank + 2, None);
                }

                match rank_table[rank].take() {
                    None => {
                        rank_table[rank] = Some(current);
                        break;
                    }
                    Some(other) => {
                        // Link the two trees
                        current = self.link(current, other);
                    }
                }
            }
        }

        // Rebuild root list from rank table
        for node_ptr in rank_table.into_iter().flatten() {
            self.add_to_roots(node_ptr);
        }

        // Find new minimum
        self.find_new_min();
    }

    /// Links two trees, making the one with larger priority a child of the other.
    ///
    /// Returns the root of the combined tree.
    fn link(&mut self, a: NonNull<Node<T, P>>, b: NonNull<Node<T, P>>) -> NonNull<Node<T, P>> {
        let ops = CircularListOps::new();

        // Determine which becomes the parent (smaller priority wins)
        let (parent_ptr, child_ptr) = unsafe {
            if a.as_ref().priority <= b.as_ref().priority {
                (a, b)
            } else {
                (b, a)
            }
        };

        unsafe {
            let parent = parent_ptr.as_ptr();
            let child = child_ptr.as_ptr();

            // Set child's parent
            (*child).parent = Some(parent_ptr);
            (*child).active = false;

            // Add child to parent's children list
            let child_link = (*child).sibling_link_ptr();

            match (*parent).child {
                None => {
                    // First child
                    ops.make_circular(child_link);
                    (*parent).child = Some(child_ptr);
                }
                Some(first_child_ptr) => {
                    // Insert into existing children list
                    let first_child_link = first_child_ptr.as_ref().sibling_link_ptr();
                    ops.insert_after(first_child_link, child_link);
                }
            }

            // Increment parent's rank
            (*parent).rank += 1;
        }

        parent_ptr
    }

    /// Cuts a node from its parent and adds it to the root list.
    fn cut(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            let parent_ptr = match (*node).parent {
                Some(p) => p,
                None => return, // Already a root
            };
            let parent = parent_ptr.as_ptr();

            // Remove from parent's children list
            let node_link = (*node).sibling_link_ptr();
            let next_link = ops.next(node_link);

            if next_link == Some(node_link) {
                // Only child
                (*parent).child = None;
            } else {
                // Update parent's child pointer if needed
                if (*parent).child == Some(node_ptr) {
                    let next_node_ptr: *mut Node<T, P> =
                        container_of_mut!(next_link.unwrap().as_ptr(), Node<T, P>, sibling_link);
                    (*parent).child = Some(NonNull::new_unchecked(next_node_ptr));
                }
            }

            ops.unlink(node_link);

            // Decrement parent's rank
            (*parent).rank -= 1;

            // Clear parent link
            (*node).parent = None;

            // Add to root list
            self.add_to_roots(node_ptr);

            // Note: In a full strict Fibonacci heap, we would handle
            // cascading cuts and active node tracking here.
            // For now, we just do a simple cut (amortized bounds).
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecreaseKeyHeap;

    #[test]
    fn test_basic_operations() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();

        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);

        let h1 = heap.push_with_handle(5, "five");
        let h2 = heap.push_with_handle(3, "three");
        let _h3 = heap.push_with_handle(7, "seven");

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some((&3, &"three")));

        // Test decrease_key
        assert!(heap.decrease_key(&h1, 1).is_ok());
        assert_eq!(heap.peek(), Some((&1, &"five")));

        // Test decrease_key that doesn't change minimum
        assert!(heap.decrease_key(&h2, 2).is_ok());
        assert_eq!(heap.peek(), Some((&1, &"five")));

        // Test pop
        let min = heap.pop();
        assert_eq!(min, Some((1, "five")));
        assert_eq!(heap.len(), 2);

        assert_eq!(heap.peek(), Some((&2, &"three")));
    }

    #[test]
    fn test_decrease_key_errors() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();
        let handle = heap.push_with_handle(5, "item");

        // Try to increase priority
        assert_eq!(
            heap.decrease_key(&handle, 10),
            Err(HeapError::PriorityNotDecreased)
        );

        // Try with same priority
        assert_eq!(
            heap.decrease_key(&handle, 5),
            Err(HeapError::PriorityNotDecreased)
        );
    }

    #[test]
    fn test_merge() {
        let mut heap1: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();
        heap1.push(5, "five");
        heap1.push(3, "three");

        let mut heap2: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();
        heap2.push(1, "one");
        heap2.push(4, "four");

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.peek(), Some((&1, &"one")));
    }

    #[test]
    fn test_empty_operations() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();

        assert!(heap.pop().is_none());
        assert!(heap.peek().is_none());
    }

    #[test]
    fn test_large_sequence() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();
        let mut handles = Vec::new();

        // Insert 1000 elements with priorities i*10
        for i in 0..1000 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        // Decrease keys of every 10th element (starting from i=10 since i=0 can't decrease)
        for i in (10..1000).step_by(10) {
            let result = heap.decrease_key(&handles[i], i as i32);
            assert!(
                result.is_ok(),
                "Failed to decrease key for handle {} (priority {} -> {})",
                i,
                i * 10,
                i
            );
        }

        // Pop first 100
        for _ in 0..100 {
            heap.pop();
        }

        // Insert 200 more
        for i in 1000..1200 {
            heap.push(i * 10, i);
        }

        // Verify heap still works
        assert!(!heap.is_empty());

        // Pop all remaining
        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        // 1000 - 100 + 200 = 1100
        assert_eq!(count, 1100);
    }

    #[test]
    fn test_pop_order() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        heap.push(3, 30);
        heap.push(1, 10);
        heap.push(4, 40);
        heap.push(1, 11); // Duplicate priority
        heap.push(5, 50);
        heap.push(9, 90);
        heap.push(2, 20);

        // Pop should return in priority order
        let mut last_priority = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(
                priority >= last_priority,
                "Out of order: {} after {}",
                priority,
                last_priority
            );
            last_priority = priority;
        }
    }

    #[test]
    fn test_decrease_key_to_new_min() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();

        heap.push(10, "a");
        heap.push(20, "b");
        let h = heap.push_with_handle(30, "c");

        assert_eq!(heap.peek(), Some((&10, &"a")));

        // Decrease c's priority to become new minimum
        assert!(heap.decrease_key(&h, 5).is_ok());
        assert_eq!(heap.peek(), Some((&5, &"c")));
    }

    #[test]
    fn test_decrease_key_with_cut() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Build a tree structure by doing inserts and pops
        for i in 0..10 {
            heap.push(i, i);
        }

        // Pop to trigger consolidation, building trees
        heap.pop();

        // Store handles for remaining elements
        let mut handles = Vec::new();
        for i in 10..20 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        // Decrease a key that should trigger a cut
        if !handles.is_empty() {
            let result = heap.decrease_key(&handles[5], 0);
            assert!(result.is_ok());
            assert_eq!(heap.peek().map(|(p, _)| *p), Some(0));
        }
    }
}
