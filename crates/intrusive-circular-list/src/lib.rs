//! Intrusive circular doubly-linked list.
//!
//! This crate provides a circular doubly-linked list designed to be compatible
//! with the patterns used in `intrusive-collections`. It can potentially be
//! upstreamed to that crate in the future.
//!
//! # Circular vs Linear Lists
//!
//! In a circular list:
//! - A single node points to itself (both `next` and `prev`)
//! - There is no head or tail - any node can be the "entry point"
//! - Splicing two rings together is O(1)
//! - Iteration wraps around (must track starting point)
//!
//! This is useful for data structures like Fibonacci heaps where:
//! - Siblings form a ring around their parent
//! - Nodes may belong to multiple circular lists simultaneously
//! - O(1) merge operations are required
//!
//! # Example
//!
//! ```rust
//! use intrusive_circular_list::{CircularLink, CircularListOps};
//! use std::ptr::NonNull;
//!
//! struct Node {
//!     link: CircularLink,
//!     value: i32,
//! }
//!
//! // Use the low-level operations directly
//! let mut ops = CircularListOps;
//!
//! let mut node1 = Node { link: CircularLink::new(), value: 1 };
//! let mut node2 = Node { link: CircularLink::new(), value: 2 };
//!
//! unsafe {
//!     let ptr1 = NonNull::from(&node1.link);
//!     let ptr2 = NonNull::from(&node2.link);
//!
//!     // Create a single-element ring
//!     ops.make_circular(ptr1);
//!     assert!(node1.link.is_linked());
//!
//!     // Insert node2 after node1
//!     ops.insert_after(ptr1, ptr2);
//!
//!     // Both are now in the same ring
//!     assert_eq!(ops.next(ptr1), Some(ptr2));
//!     assert_eq!(ops.next(ptr2), Some(ptr1));
//! }
//! ```
//!
//! # Compatibility with intrusive-collections
//!
//! This crate follows similar patterns to `intrusive-collections`:
//! - `CircularLink` is analogous to `LinkedListLink`
//! - `CircularListOps` provides the low-level operations
//! - Uses `NonNull` for link pointers
//! - Supports being part of multiple lists (embed multiple links)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

use core::cell::Cell;
use core::fmt;
use core::ptr::NonNull;

// =============================================================================
// CircularLink
// =============================================================================

/// Intrusive link that allows an object to be inserted into a circular list.
///
/// When unlinked, both `next` and `prev` are `None`.
/// When linked (even as a single node), both point to valid nodes.
/// A single node in a circular list points to itself.
#[repr(C)]
pub struct CircularLink {
    next: Cell<Option<NonNull<CircularLink>>>,
    prev: Cell<Option<NonNull<CircularLink>>>,
}

impl CircularLink {
    /// Creates a new unlinked `CircularLink`.
    #[inline]
    pub const fn new() -> CircularLink {
        CircularLink {
            next: Cell::new(None),
            prev: Cell::new(None),
        }
    }

    /// Checks whether the `CircularLink` is linked into a circular list.
    #[inline]
    pub fn is_linked(&self) -> bool {
        self.next.get().is_some()
    }

    /// Forcibly unlinks this node.
    ///
    /// # Safety
    ///
    /// This does not update neighboring nodes. Only use this after the
    /// list has been cleared with `fast_clear`, or when you know the
    /// node is the only element and you're removing it.
    #[inline]
    pub unsafe fn force_unlink(&self) {
        self.next.set(None);
        self.prev.set(None);
    }

    /// Gets the next link pointer.
    #[inline]
    pub fn next(&self) -> Option<NonNull<CircularLink>> {
        self.next.get()
    }

    /// Gets the previous link pointer.
    #[inline]
    pub fn prev(&self) -> Option<NonNull<CircularLink>> {
        self.prev.get()
    }

    /// Sets the next link pointer.
    #[inline]
    pub fn set_next(&self, next: Option<NonNull<CircularLink>>) {
        self.next.set(next);
    }

    /// Sets the previous link pointer.
    #[inline]
    pub fn set_prev(&self, prev: Option<NonNull<CircularLink>>) {
        self.prev.set(prev);
    }
}

impl Default for CircularLink {
    #[inline]
    fn default() -> Self {
        CircularLink::new()
    }
}

impl Clone for CircularLink {
    /// Cloning a link creates a new unlinked link.
    #[inline]
    fn clone(&self) -> Self {
        CircularLink::new()
    }
}

impl fmt::Debug for CircularLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_linked() {
            write!(f, "CircularLink(linked)")
        } else {
            write!(f, "CircularLink(unlinked)")
        }
    }
}

// A CircularLink can be sent to another thread if it is unlinked.
unsafe impl Send for CircularLink {}

// =============================================================================
// CircularListOps
// =============================================================================

/// Operations for manipulating circular doubly-linked lists.
///
/// This struct provides low-level operations on `CircularLink` pointers.
/// All operations are O(1).
///
/// # Safety
///
/// Most methods are unsafe because they operate on raw pointers.
/// The caller must ensure:
/// - Pointers are valid and properly aligned
/// - Nodes are not already in use (for insertion) or are in use (for removal)
/// - The circular list invariants are maintained
#[derive(Clone, Copy, Default)]
pub struct CircularListOps;

impl CircularListOps {
    /// Creates a new `CircularListOps`.
    #[inline]
    pub const fn new() -> Self {
        CircularListOps
    }

    /// Checks if a node is linked.
    #[inline]
    pub fn is_linked(&self, ptr: NonNull<CircularLink>) -> bool {
        unsafe { ptr.as_ref().is_linked() }
    }

    /// Gets the next link in the circular list.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and properly aligned.
    #[inline]
    pub unsafe fn next(&self, ptr: NonNull<CircularLink>) -> Option<NonNull<CircularLink>> {
        ptr.as_ref().next()
    }

    /// Gets the previous link in the circular list.
    ///
    /// # Safety
    ///
    /// The pointer must be valid and properly aligned.
    #[inline]
    pub unsafe fn prev(&self, ptr: NonNull<CircularLink>) -> Option<NonNull<CircularLink>> {
        ptr.as_ref().prev()
    }

    /// Makes a single node into a circular list of one element.
    ///
    /// After this, `ptr.next == ptr` and `ptr.prev == ptr`.
    ///
    /// # Safety
    ///
    /// The node must not already be linked.
    #[inline]
    pub unsafe fn make_circular(&self, ptr: NonNull<CircularLink>) {
        debug_assert!(!ptr.as_ref().is_linked(), "node is already linked");
        ptr.as_ref().set_next(Some(ptr));
        ptr.as_ref().set_prev(Some(ptr));
    }

    /// Unlinks a node and marks it as not linked.
    ///
    /// # Safety
    ///
    /// The node must be linked.
    #[inline]
    pub unsafe fn unlink(&self, ptr: NonNull<CircularLink>) {
        ptr.as_ref().force_unlink();
    }

    /// Inserts `new` after `at` in the circular list.
    ///
    /// # Safety
    ///
    /// - `at` must be linked (part of a circular list)
    /// - `new` must NOT be linked
    #[inline]
    pub unsafe fn insert_after(&self, at: NonNull<CircularLink>, new: NonNull<CircularLink>) {
        debug_assert!(at.as_ref().is_linked(), "at is not linked");
        debug_assert!(!new.as_ref().is_linked(), "new is already linked");

        let next = at.as_ref().next().unwrap();

        at.as_ref().set_next(Some(new));
        new.as_ref().set_prev(Some(at));
        new.as_ref().set_next(Some(next));
        next.as_ref().set_prev(Some(new));
    }

    /// Inserts `new` before `at` in the circular list.
    ///
    /// # Safety
    ///
    /// - `at` must be linked (part of a circular list)
    /// - `new` must NOT be linked
    #[inline]
    pub unsafe fn insert_before(&self, at: NonNull<CircularLink>, new: NonNull<CircularLink>) {
        debug_assert!(at.as_ref().is_linked(), "at is not linked");
        debug_assert!(!new.as_ref().is_linked(), "new is already linked");

        let prev = at.as_ref().prev().unwrap();

        at.as_ref().set_prev(Some(new));
        new.as_ref().set_next(Some(at));
        new.as_ref().set_prev(Some(prev));
        prev.as_ref().set_next(Some(new));
    }

    /// Removes a node from the circular list.
    ///
    /// Returns `true` if the node was the only element (list is now empty),
    /// `false` if other nodes remain.
    ///
    /// # Safety
    ///
    /// The node must be linked (part of a circular list).
    #[inline]
    pub unsafe fn remove(&self, ptr: NonNull<CircularLink>) -> bool {
        debug_assert!(ptr.as_ref().is_linked(), "node is not linked");

        let next = ptr.as_ref().next().unwrap();
        let prev = ptr.as_ref().prev().unwrap();

        if next == ptr {
            // Single node - list becomes empty
            ptr.as_ref().force_unlink();
            true
        } else {
            // Multiple nodes - splice out
            prev.as_ref().set_next(Some(next));
            next.as_ref().set_prev(Some(prev));
            ptr.as_ref().force_unlink();
            false
        }
    }

    /// Splices two circular lists together.
    ///
    /// After this operation, both lists are merged into one circular list.
    /// Returns an entry point into the merged list, or `None` if both were empty.
    ///
    /// If either is `None`, returns the other.
    ///
    /// The merge is O(1) - it just reconnects the endpoints.
    ///
    /// # Safety
    ///
    /// Both pointers (if Some) must be valid and linked.
    #[inline]
    pub unsafe fn splice(
        &self,
        a: Option<NonNull<CircularLink>>,
        b: Option<NonNull<CircularLink>>,
    ) -> Option<NonNull<CircularLink>> {
        match (a, b) {
            (None, None) => None,
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (Some(a), Some(b)) => {
                // Splice the two rings together:
                //
                // Before:
                //   Ring A: ... <-> a_prev <-> a <-> a_next <-> ... <-> a_prev
                //   Ring B: ... <-> b_prev <-> b <-> b_next <-> ... <-> b_prev
                //
                // After:
                //   ... <-> a_prev <-> b <-> ... <-> b_prev <-> a <-> a_next <-> ... <-> a_prev
                //
                // We break the rings at a and b, then connect:
                //   a_prev -> b
                //   b_prev -> a

                let a_prev = a.as_ref().prev().unwrap();
                let b_prev = b.as_ref().prev().unwrap();

                // Connect a_prev <-> b
                a_prev.as_ref().set_next(Some(b));
                b.as_ref().set_prev(Some(a_prev));

                // Connect b_prev <-> a
                b_prev.as_ref().set_next(Some(a));
                a.as_ref().set_prev(Some(b_prev));

                Some(a)
            }
        }
    }

    /// Counts the number of elements in the circular list.
    ///
    /// This is O(n) - use sparingly.
    ///
    /// # Safety
    ///
    /// The pointer must be linked (part of a circular list).
    pub unsafe fn count(&self, start: NonNull<CircularLink>) -> usize {
        debug_assert!(start.as_ref().is_linked(), "node is not linked");

        let mut count = 1;
        let mut current = start.as_ref().next().unwrap();

        while current != start {
            count += 1;
            current = current.as_ref().next().unwrap();
        }

        count
    }

    /// Iterates over all elements in the circular list, calling `f` for each.
    ///
    /// # Safety
    ///
    /// The pointer must be linked (part of a circular list).
    /// The callback must not modify the list structure.
    pub unsafe fn for_each<F>(&self, start: NonNull<CircularLink>, mut f: F)
    where
        F: FnMut(NonNull<CircularLink>),
    {
        debug_assert!(start.as_ref().is_linked(), "node is not linked");

        f(start);
        let mut current = start.as_ref().next().unwrap();

        while current != start {
            f(current);
            current = current.as_ref().next().unwrap();
        }
    }
}

// =============================================================================
// Utility function for calculating container offset
// =============================================================================

/// Calculates the offset of a field within a struct.
///
/// This is useful for converting between link pointers and container pointers.
///
/// # Example
///
/// ```rust
/// use intrusive_circular_list::{CircularLink, container_of};
/// use std::ptr::NonNull;
///
/// struct Node {
///     value: i32,
///     link: CircularLink,
/// }
///
/// let node = Node { value: 42, link: CircularLink::new() };
/// let link_ptr = NonNull::from(&node.link);
///
/// unsafe {
///     let node_ptr: *const Node = container_of!(link_ptr.as_ptr(), Node, link);
///     assert_eq!((*node_ptr).value, 42);
/// }
/// ```
#[macro_export]
macro_rules! container_of {
    ($ptr:expr, $type:ty, $field:ident) => {{
        let ptr = $ptr as *const u8;
        let offset = core::mem::offset_of!($type, $field);
        ptr.sub(offset) as *const $type
    }};
}

/// Mutable version of `container_of`.
#[macro_export]
macro_rules! container_of_mut {
    ($ptr:expr, $type:ty, $field:ident) => {{
        let ptr = $ptr as *mut u8;
        let offset = core::mem::offset_of!($type, $field);
        ptr.sub(offset) as *mut $type
    }};
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct TestNode {
        link: CircularLink,
        value: i32,
    }

    impl TestNode {
        fn new(value: i32) -> Self {
            TestNode {
                link: CircularLink::new(),
                value,
            }
        }
    }

    #[test]
    fn test_new_link_unlinked() {
        let link = CircularLink::new();
        assert!(!link.is_linked());
    }

    #[test]
    fn test_make_circular_single() {
        let node = TestNode::new(1);
        let ops = CircularListOps::new();

        unsafe {
            let ptr = NonNull::from(&node.link);
            ops.make_circular(ptr);

            assert!(node.link.is_linked());
            assert_eq!(ops.next(ptr), Some(ptr));
            assert_eq!(ops.prev(ptr), Some(ptr));
        }
    }

    #[test]
    fn test_insert_after() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);

            ops.make_circular(ptr1);
            ops.insert_after(ptr1, ptr2);

            // Now: 1 <-> 2 <-> 1 (circular)
            assert_eq!(ops.next(ptr1), Some(ptr2));
            assert_eq!(ops.next(ptr2), Some(ptr1));
            assert_eq!(ops.prev(ptr1), Some(ptr2));
            assert_eq!(ops.prev(ptr2), Some(ptr1));
        }
    }

    #[test]
    fn test_insert_before() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);

            ops.make_circular(ptr1);
            ops.insert_before(ptr1, ptr2);

            // Now: 2 <-> 1 <-> 2 (circular)
            // Starting from ptr1: prev is ptr2, next is ptr2
            assert_eq!(ops.prev(ptr1), Some(ptr2));
            assert_eq!(ops.next(ptr1), Some(ptr2));
            assert_eq!(ops.prev(ptr2), Some(ptr1));
            assert_eq!(ops.next(ptr2), Some(ptr1));
        }
    }

    #[test]
    fn test_insert_three_nodes() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let node3 = TestNode::new(3);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);
            let ptr3 = NonNull::from(&node3.link);

            ops.make_circular(ptr1);
            ops.insert_after(ptr1, ptr2);
            ops.insert_after(ptr2, ptr3);

            // Now: 1 <-> 2 <-> 3 <-> 1 (circular)
            assert_eq!(ops.next(ptr1), Some(ptr2));
            assert_eq!(ops.next(ptr2), Some(ptr3));
            assert_eq!(ops.next(ptr3), Some(ptr1));

            assert_eq!(ops.prev(ptr1), Some(ptr3));
            assert_eq!(ops.prev(ptr2), Some(ptr1));
            assert_eq!(ops.prev(ptr3), Some(ptr2));
        }
    }

    #[test]
    fn test_remove_single() {
        let node = TestNode::new(1);
        let ops = CircularListOps::new();

        unsafe {
            let ptr = NonNull::from(&node.link);
            ops.make_circular(ptr);

            let was_last = ops.remove(ptr);
            assert!(was_last);
            assert!(!node.link.is_linked());
        }
    }

    #[test]
    fn test_remove_from_two() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);

            ops.make_circular(ptr1);
            ops.insert_after(ptr1, ptr2);

            let was_last = ops.remove(ptr1);
            assert!(!was_last);
            assert!(!node1.link.is_linked());

            // node2 should now point to itself
            assert_eq!(ops.next(ptr2), Some(ptr2));
            assert_eq!(ops.prev(ptr2), Some(ptr2));
        }
    }

    #[test]
    fn test_remove_middle() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let node3 = TestNode::new(3);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);
            let ptr3 = NonNull::from(&node3.link);

            ops.make_circular(ptr1);
            ops.insert_after(ptr1, ptr2);
            ops.insert_after(ptr2, ptr3);

            // Remove node2
            let was_last = ops.remove(ptr2);
            assert!(!was_last);
            assert!(!node2.link.is_linked());

            // Now: 1 <-> 3 <-> 1
            assert_eq!(ops.next(ptr1), Some(ptr3));
            assert_eq!(ops.next(ptr3), Some(ptr1));
        }
    }

    #[test]
    fn test_splice_empty() {
        let ops = CircularListOps::new();

        unsafe {
            let result = ops.splice(None, None);
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_splice_one_empty() {
        let node = TestNode::new(1);
        let ops = CircularListOps::new();

        unsafe {
            let ptr = NonNull::from(&node.link);
            ops.make_circular(ptr);

            let result = ops.splice(Some(ptr), None);
            assert_eq!(result, Some(ptr));

            let result = ops.splice(None, Some(ptr));
            assert_eq!(result, Some(ptr));
        }
    }

    #[test]
    fn test_splice_two_singles() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);

            ops.make_circular(ptr1);
            ops.make_circular(ptr2);

            let result = ops.splice(Some(ptr1), Some(ptr2));
            assert_eq!(result, Some(ptr1));

            // Should now be: 1 <-> 2 <-> 1
            assert_eq!(ops.next(ptr1), Some(ptr2));
            assert_eq!(ops.next(ptr2), Some(ptr1));
        }
    }

    #[test]
    fn test_splice_two_pairs() {
        let node1 = TestNode::new(1);
        let node2 = TestNode::new(2);
        let node3 = TestNode::new(3);
        let node4 = TestNode::new(4);
        let ops = CircularListOps::new();

        unsafe {
            let ptr1 = NonNull::from(&node1.link);
            let ptr2 = NonNull::from(&node2.link);
            let ptr3 = NonNull::from(&node3.link);
            let ptr4 = NonNull::from(&node4.link);

            // Create ring 1: 1 <-> 2
            ops.make_circular(ptr1);
            ops.insert_after(ptr1, ptr2);

            // Create ring 2: 3 <-> 4
            ops.make_circular(ptr3);
            ops.insert_after(ptr3, ptr4);

            // Splice them
            ops.splice(Some(ptr1), Some(ptr3));

            // Count should be 4
            assert_eq!(ops.count(ptr1), 4);
        }
    }

    #[test]
    fn test_count() {
        let nodes: Vec<_> = (0..5).map(TestNode::new).collect();
        let ops = CircularListOps::new();

        unsafe {
            let ptrs: Vec<_> = nodes.iter().map(|n| NonNull::from(&n.link)).collect();

            ops.make_circular(ptrs[0]);
            for i in 1..5 {
                ops.insert_after(ptrs[i - 1], ptrs[i]);
            }

            assert_eq!(ops.count(ptrs[0]), 5);
            assert_eq!(ops.count(ptrs[2]), 5); // Count from any node
        }
    }

    #[test]
    fn test_for_each() {
        let nodes: Vec<_> = (0..3).map(TestNode::new).collect();
        let ops = CircularListOps::new();

        unsafe {
            let ptrs: Vec<_> = nodes.iter().map(|n| NonNull::from(&n.link)).collect();

            ops.make_circular(ptrs[0]);
            ops.insert_after(ptrs[0], ptrs[1]);
            ops.insert_after(ptrs[1], ptrs[2]);

            let mut visited = Vec::new();
            ops.for_each(ptrs[0], |ptr| {
                let node_ptr: *const TestNode = container_of!(ptr.as_ptr(), TestNode, link);
                visited.push((*node_ptr).value);
            });

            assert_eq!(visited, vec![0, 1, 2]);
        }
    }

    #[test]
    fn test_container_of() {
        let node = TestNode::new(42);
        let link_ptr = NonNull::from(&node.link);

        unsafe {
            let node_ptr: *const TestNode = container_of!(link_ptr.as_ptr(), TestNode, link);
            assert_eq!((*node_ptr).value, 42);
        }
    }
}
