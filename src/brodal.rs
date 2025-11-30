//! Brodal-style Heap implementation
//!
//! This implementation provides a heap with the following time bounds:
//! - O(1) insert, find_min, and merge
//! - O(1) decrease_key (when heap property is maintained)
//! - O(log n) amortized delete_min
//!
//! The implementation is inspired by Brodal's 1996 paper "Worst-case efficient
//! priority queues" but uses a simplified pairing-heap-style structure for
//! ease of implementation in Rust with safe memory management.
//!
//! # Implementation Notes
//!
//! This is a simplified variant that uses:
//! - Pairing-heap-style two-pass merge for delete_min
//! - Cut-and-merge for decrease_key (similar to Fibonacci heaps)
//! - Rc/Weak references for memory safety without raw pointers
//!
//! For true worst-case bounds as in Brodal's original paper, a more complex
//! implementation with guide structures and violation buffers would be needed.

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Type alias for node references (strong ownership)
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;
/// Type alias for weak node references (back-pointers)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a Brodal heap
///
/// Note: This handle is tied to a specific heap instance. Using it with a different
/// heap or after the element has been removed will return an error.
/// The handle uses a weak reference, so it can detect if the node has been dropped.
pub struct BrodalHandle<T, P> {
    node: WeakNodeRef<T, P>,
}

// Manual Clone implementation to avoid adding T: Clone and P: Clone bounds
impl<T, P> Clone for BrodalHandle<T, P> {
    fn clone(&self) -> Self {
        BrodalHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for BrodalHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node.ptr_eq(&other.node)
    }
}

impl<T, P> Eq for BrodalHandle<T, P> {}

impl<T, P> std::fmt::Debug for BrodalHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrodalHandle")
            .field("valid", &self.node.upgrade().is_some())
            .finish()
    }
}

impl<T, P> Handle for BrodalHandle<T, P> {}

/// Node in the Brodal heap
struct Node<T, P> {
    item: T,
    priority: P,
    parent: Option<WeakNodeRef<T, P>>,
    /// First child (children are linked via sibling pointers)
    child: Option<NodeRef<T, P>>,
    /// Next sibling
    sibling: Option<NodeRef<T, P>>,
    /// Rank (number of children)
    rank: usize,
}

impl<T, P> Node<T, P> {
    fn new(priority: P, item: T) -> Self {
        Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            rank: 0,
        }
    }
}

/// Brodal-style Heap
///
/// This implementation provides correct behavior with good performance
/// characteristics inspired by Brodal heaps.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::brodal::BrodalHeap;
/// use rust_advanced_heaps::Heap;
///
/// let mut heap = BrodalHeap::new();
/// let handle = heap.push(5, "item");
/// heap.decrease_key(&handle, 1);
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct BrodalHeap<T, P: Ord> {
    /// Root of the heap (minimum element)
    root: Option<NodeRef<T, P>>,
    /// Number of elements in heap
    len: usize,
}

impl<T, P: Ord> Heap<T, P> for BrodalHeap<T, P> {
    type Handle = BrodalHandle<T, P>;

    fn new() -> Self {
        Self { root: None, len: 0 }
    }

    fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Rc::new(RefCell::new(Node::new(priority, item)));
        let handle = BrodalHandle {
            node: Rc::downgrade(&node),
        };

        self.merge_node(node);
        self.len += 1;

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.root.as_ref().map(|root| {
            let node_ref = root.as_ptr();
            // SAFETY: We're borrowing from a RefCell we know exists and isn't borrowed mutably
            unsafe { (&(*node_ref).priority, &(*node_ref).item) }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        let root = self.root.take()?;

        // Collect all children
        let mut children = Vec::new();
        {
            let root_ref = root.borrow();
            let mut current = root_ref.child.clone();
            while let Some(child) = current {
                let next = child.borrow().sibling.clone();
                children.push(child);
                current = next;
            }
        }

        // Clear parent pointers in children
        for child in &children {
            child.borrow_mut().parent = None;
            child.borrow_mut().sibling = None;
        }

        // Extract priority and item from root
        let root_node = match Rc::try_unwrap(root) {
            Ok(cell) => cell.into_inner(),
            Err(_rc) => {
                // Root has other references (shouldn't happen in normal use)
                // Can't extract owned data
                return None;
            }
        };

        self.len -= 1;

        // Rebuild heap from children using pairing
        if !children.is_empty() {
            self.root = Some(self.pair_children(children));
        }

        Some((root_node.priority, root_node.item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node = handle
            .node
            .upgrade()
            .ok_or(HeapError::InvalidHandle)?;

        // Check that new priority is actually less
        if new_priority >= node.borrow().priority {
            return Err(HeapError::PriorityNotDecreased);
        }

        // Update the priority
        node.borrow_mut().priority = new_priority;

        // Check if this node is the root
        let is_root = self.root.as_ref().is_some_and(|r| Rc::ptr_eq(r, &node));
        if is_root {
            return Ok(());
        }

        // Check if heap property is violated with parent
        let needs_cut = {
            let node_ref = node.borrow();
            if let Some(ref parent_weak) = node_ref.parent {
                if let Some(parent) = parent_weak.upgrade() {
                    node_ref.priority < parent.borrow().priority
                } else {
                    false
                }
            } else {
                false
            }
        };

        if needs_cut {
            // Cut node from parent
            self.cut_from_parent(&node);
            // Merge cut node with root
            self.merge_node(node);
        }

        Ok(())
    }

    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        let other_root = other.root.take().unwrap();
        self.merge_node(other_root);
        self.len += other.len;
        other.len = 0;
    }
}

impl<T, P: Ord> BrodalHeap<T, P> {
    /// Merges a node into the heap
    fn merge_node(&mut self, node: NodeRef<T, P>) {
        match self.root.take() {
            None => {
                self.root = Some(node);
            }
            Some(root) => {
                // Link the two trees
                let new_root = self.link(root, node);
                self.root = Some(new_root);
            }
        }
    }

    /// Links two trees, making the one with larger priority a child of the other
    fn link(&self, a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        let a_smaller = a.borrow().priority <= b.borrow().priority;

        if a_smaller {
            // a becomes parent, b becomes child
            self.make_child(&a, b);
            a
        } else {
            // b becomes parent, a becomes child
            self.make_child(&b, a);
            b
        }
    }

    /// Makes child a child of parent
    fn make_child(&self, parent: &NodeRef<T, P>, child: NodeRef<T, P>) {
        // Set child's parent and sibling
        {
            let mut child_mut = child.borrow_mut();
            child_mut.parent = Some(Rc::downgrade(parent));
            child_mut.sibling = parent.borrow().child.clone();
        }

        // Add child to parent's children
        {
            let mut parent_mut = parent.borrow_mut();
            parent_mut.child = Some(child);
            parent_mut.rank += 1;
        }
    }

    /// Cuts a node from its parent
    fn cut_from_parent(&mut self, node: &NodeRef<T, P>) {
        let parent_weak = node.borrow().parent.clone();
        let parent = match parent_weak.and_then(|w| w.upgrade()) {
            Some(p) => p,
            None => return,
        };

        // Remove node from parent's child list
        {
            let mut parent_mut = parent.borrow_mut();
            let first_child = parent_mut.child.clone();

            if first_child.as_ref().is_some_and(|c| Rc::ptr_eq(c, node)) {
                // Node is the first child
                parent_mut.child = node.borrow().sibling.clone();
            } else {
                // Find node in sibling chain
                let mut prev = first_child;
                while let Some(curr) = prev {
                    let next = curr.borrow().sibling.clone();
                    if next.as_ref().is_some_and(|n| Rc::ptr_eq(n, node)) {
                        curr.borrow_mut().sibling = node.borrow().sibling.clone();
                        break;
                    }
                    prev = next;
                }
            }
            parent_mut.rank = parent_mut.rank.saturating_sub(1);
        }

        // Clear node's parent and sibling
        {
            let mut node_mut = node.borrow_mut();
            node_mut.parent = None;
            node_mut.sibling = None;
        }
    }

    /// Pairs children using the pairing heap strategy (two-pass)
    fn pair_children(&self, mut children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.is_empty() {
            panic!("Cannot pair empty children list");
        }

        if children.len() == 1 {
            return children.pop().unwrap();
        }

        // First pass: pair adjacent trees
        let mut paired = Vec::new();
        while children.len() >= 2 {
            let a = children.pop().unwrap();
            let b = children.pop().unwrap();
            paired.push(self.link(a, b));
        }
        if let Some(odd) = children.pop() {
            paired.push(odd);
        }

        // Second pass: merge right to left
        while paired.len() > 1 {
            let a = paired.pop().unwrap();
            let b = paired.pop().unwrap();
            paired.push(self.link(a, b));
        }

        paired.pop().unwrap()
    }
}

impl<T, P: Ord> Default for BrodalHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = BrodalHeap::new();

        heap.push(5, "five");
        heap.push(3, "three");
        heap.push(7, "seven");
        heap.push(1, "one");

        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.len(), 4);

        assert_eq!(heap.pop(), Some((1, "one")));
        assert_eq!(heap.peek(), Some((&3, &"three")));

        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.pop(), Some((5, "five")));
        assert_eq!(heap.pop(), Some((7, "seven")));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = BrodalHeap::new();

        let h1 = heap.push(10, "a");
        let h2 = heap.push(5, "b");
        let _h3 = heap.push(15, "c");

        assert_eq!(heap.peek(), Some((&5, &"b")));

        // Decrease h1 to become minimum
        heap.decrease_key(&h1, 1).unwrap();
        assert_eq!(heap.peek(), Some((&1, &"a")));

        // Try invalid decrease (not actually decreasing)
        assert!(heap.decrease_key(&h2, 10).is_err());
    }

    #[test]
    fn test_merge() {
        let mut heap1 = BrodalHeap::new();
        heap1.push(5, "a");
        heap1.push(10, "b");

        let mut heap2 = BrodalHeap::new();
        heap2.push(3, "c");
        heap2.push(15, "d");

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.peek(), Some((&3, &"c")));

        // Verify all elements present
        assert_eq!(heap1.pop(), Some((3, "c")));
        assert_eq!(heap1.pop(), Some((5, "a")));
        assert_eq!(heap1.pop(), Some((10, "b")));
        assert_eq!(heap1.pop(), Some((15, "d")));
    }

    #[test]
    fn test_sequential_operations() {
        let mut heap = BrodalHeap::new();

        // Insert many elements
        for i in (0..100).rev() {
            heap.push(i, i);
        }

        // Verify they come out in order
        for i in 0..100 {
            assert_eq!(heap.pop(), Some((i, i)));
        }
    }

    #[test]
    fn test_decrease_key_edge_case() {
        let mut heap = BrodalHeap::new();
        let mut handles = Vec::new();

        // Insert values [0, 0, -1, -2]
        for val in [0, 0, -1, -2] {
            let handle = heap.push(val, val);
            handles.push(handle);
        }

        // Decrease each key by 100
        for (idx, val) in [0, 0, -1, -2].iter().enumerate() {
            let new_priority = val - 100;
            assert!(heap.decrease_key(&handles[idx], new_priority).is_ok());
        }

        // Drain heap and verify all elements present
        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
    }
}
