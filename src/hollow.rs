//! Hollow Heap implementation
//!
//! A hollow heap is a simple data structure with the same amortized efficiency as
//! the classical Fibonacci heap, but with a simpler implementation.
//!
//! # Time Complexity
//!
//! | Operation      | Complexity           |
//! |----------------|----------------------|
//! | `push`         | O(1) worst-case      |
//! | `pop`          | O(log n) amortized   |
//! | `peek`         | O(1) worst-case      |
//! | `decrease_key` | O(1) amortized       |
//! | `merge`        | O(1) worst-case      |
//!
//! # Key Innovation
//!
//! Hollow heaps combine two novel ideas:
//!
//! 1. **Lazy deletion for decrease-key**: Instead of cutting nodes (like Fibonacci heaps),
//!    we create a new node with the lower key and mark the old node as "hollow" (empty).
//!    The item is moved to the new node, and the old node remains in the structure.
//!
//! 2. **DAG structure**: A hollow node can have a "second parent" pointer, creating a
//!    directed acyclic graph (DAG) instead of a tree. This allows multiple paths to
//!    the same node, which is handled during delete-min.
//!
//! # Algorithm Overview
//!
//! - **Insert**: Create a new node and link with root (like Fibonacci heap)
//! - **Decrease-key**: Create new node with lower key, move item there, mark old as hollow
//! - **Delete-min**: Remove root, process children, perform ranked linking (like Fibonacci)
//! - **Merge**: Link roots (like Fibonacci heap)
//!
//! # References
//!
//! - Hansen, T.D., Kaplan, H., Tarjan, R.E., Zwick, U. (2015). "Hollow Heaps."
//!   *ICALP 2015*. [arXiv:1510.06535](https://arxiv.org/abs/1510.06535)
//! - Hansen, T.D., Kaplan, H., Tarjan, R.E., Zwick, U. (2017). "Hollow Heaps."
//!   *ACM Transactions on Algorithms*, 13(3), 42.

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError, MergeableHeap};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Type alias for the mutable weak reference to a node
/// This allows the handle to be updated when decrease_key moves the item to a new node.
type MutableNodeRef<T, P> = Rc<RefCell<Weak<RefCell<Node<T, P>>>>>;

/// Handle to an element in a Hollow heap
///
/// Uses a Weak reference to the node wrapped in RefCell. This allows the
/// handle to be updated when decrease_key moves the item to a new node.
/// The handle always follows the item, not the node structure.
pub struct HollowHandle<T, P> {
    /// Reference to the node containing the item (can be updated on decrease_key)
    node: MutableNodeRef<T, P>,
}

impl<T, P> Clone for HollowHandle<T, P> {
    fn clone(&self) -> Self {
        HollowHandle {
            node: Rc::clone(&self.node),
        }
    }
}

impl<T, P> PartialEq for HollowHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.node, &other.node)
    }
}

impl<T, P> Eq for HollowHandle<T, P> {}

impl<T, P> std::fmt::Debug for HollowHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HollowHandle")
            .field("node", &self.node.borrow().as_ptr())
            .finish()
    }
}

impl<T, P> Handle for HollowHandle<T, P> {}

/// Internal node structure for Hollow heap
///
/// Each node can be either "full" (containing an item) or "hollow" (empty).
/// Hollow nodes are created during decrease_key when we move an item to a
/// new node with a lower priority.
///
/// The structure forms a DAG (directed acyclic graph) rather than a tree,
/// because hollow nodes can have a "second parent" (ep - extra parent).
struct Node<T, P> {
    /// The item stored in this node (None if hollow)
    item: Option<T>,
    /// The priority/key of this node
    priority: P,
    /// First child in the child list
    child: Option<Rc<RefCell<Node<T, P>>>>,
    /// Next sibling in the child list
    next: Option<Rc<RefCell<Node<T, P>>>>,
    /// Extra parent pointer (only used for hollow nodes in DAG)
    /// This creates the DAG structure when decrease_key moves items
    second_parent: Weak<RefCell<Node<T, P>>>,
    /// Rank of this node (similar to degree in Fibonacci heaps)
    rank: usize,
}

/// Type alias for a reference-counted node pointer
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;

impl<T, P> Node<T, P> {
    fn new(item: T, priority: P) -> Self {
        Node {
            item: Some(item),
            priority,
            child: None,
            next: None,
            second_parent: Weak::new(),
            rank: 0,
        }
    }

    /// Returns true if this node is hollow (has no item)
    fn is_hollow(&self) -> bool {
        self.item.is_none()
    }
}

/// Hollow Heap implementation
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::hollow::HollowHeap;
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
///
/// let mut heap = HollowHeap::new();
/// let handle = heap.push_with_handle(5, "item");
/// heap.decrease_key(&handle, 1).unwrap();
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct HollowHeap<T, P: Ord> {
    /// Root of the heap (None if empty)
    root: Option<NodeRef<T, P>>,
    /// Number of non-hollow items in the heap
    len: usize,
}

impl<T, P: Ord + Clone> Default for HollowHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, P: Ord + Clone> Heap<T, P> for HollowHeap<T, P> {
    fn new() -> Self {
        Self { root: None, len: 0 }
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
        self.root.as_ref().and_then(|root_rc| {
            let root = root_rc.borrow();
            if root.is_hollow() {
                // Root should never be hollow in a well-formed heap
                None
            } else {
                // SAFETY: We bypass RefCell's dynamic borrow checking to return
                // references with lifetime tied to `&self`. This is safe because:
                // 1. The Rc in `self.root` keeps the node alive for `&self`'s lifetime
                // 2. Rust's borrow checker prevents `&mut self` calls while these
                //    references exist, so no mutation can occur
                // 3. The root is never hollow in a well-formed heap (checked above)
                unsafe {
                    let ptr = root_rc.as_ptr();
                    let item_ref = (*ptr).item.as_ref().unwrap();
                    Some((&(*ptr).priority, item_ref))
                }
            }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        let root_rc = self.root.take()?;

        // The root must not be hollow
        let (priority, item) = {
            let mut root = root_rc.borrow_mut();
            if root.is_hollow() {
                // This shouldn't happen in a well-formed heap
                return None;
            }
            // Clone the priority since we need to return it but the node
            // may still be referenced during delete_min_rebuild
            (root.priority.clone(), root.item.take().unwrap())
        };

        self.len -= 1;

        // Process children and rebuild heap
        self.delete_min_rebuild(root_rc);

        Some((priority, item))
    }
}

impl<T, P: Ord + Clone> MergeableHeap<T, P> for HollowHeap<T, P> {
    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        // Link the two roots
        let other_len = other.len;
        let other_root = other.root.take().expect("non-empty heap must have root");
        let self_root = self.root.take().expect("non-empty heap must have root");

        self.root = Some(Self::link(self_root, other_root));
        self.len += other_len;
    }
}

impl<T, P: Ord + Clone> DecreaseKeyHeap<T, P> for HollowHeap<T, P> {
    type Handle = HollowHandle<T, P>;

    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Rc::new(RefCell::new(Node::new(item, priority)));
        let handle = HollowHandle {
            node: Rc::new(RefCell::new(Rc::downgrade(&node))),
        };

        // Link with existing root
        match self.root.take() {
            None => {
                self.root = Some(node);
            }
            Some(root) => {
                self.root = Some(Self::link(root, node));
            }
        }

        self.len += 1;
        handle
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        // Upgrade weak reference to get the node
        let node_rc = handle
            .node
            .borrow()
            .upgrade()
            .ok_or(HeapError::InvalidHandle)?;

        // Check that new priority is actually less
        {
            let node = node_rc.borrow();
            if node.is_hollow() {
                return Err(HeapError::InvalidHandle);
            }
            if new_priority >= node.priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Check if this is the root - if so, just update the priority
        if let Some(ref root) = self.root {
            if Rc::ptr_eq(root, &node_rc) {
                node_rc.borrow_mut().priority = new_priority;
                return Ok(());
            }
        }

        // Create a new node with the lower priority
        let new_node = Rc::new(RefCell::new(Node {
            item: node_rc.borrow_mut().item.take(),
            priority: new_priority,
            child: None,
            next: None,
            second_parent: Weak::new(),
            rank: 0,
        }));

        // Update the handle to point to the new node
        *handle.node.borrow_mut() = Rc::downgrade(&new_node);

        // Make the old node hollow and set its second_parent to the new node
        // The new node becomes a "parent" of the hollow node
        {
            let mut old_node = node_rc.borrow_mut();
            // The old node is now hollow (item was moved out above)
            // Set second_parent so we can find this hollow node from the new node
            old_node.second_parent = Rc::downgrade(&new_node);
        }

        // Add the old hollow node as a child of the new node
        {
            let mut new_node_mut = new_node.borrow_mut();
            new_node_mut.child = Some(node_rc.clone());
            new_node_mut.rank = node_rc.borrow().rank.saturating_add(1).max(1);
        }

        // Link new node with root
        let root = self.root.take().expect("heap must have a root");
        self.root = Some(Self::link(root, new_node));

        Ok(())
    }
}

impl<T, P: Ord> HollowHeap<T, P> {
    /// Links two nodes, making the one with higher priority a child of the other.
    /// Returns the winner (node with lower priority).
    fn link(a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        let a_wins = {
            let a_priority = &a.borrow().priority;
            let b_priority = &b.borrow().priority;
            a_priority <= b_priority
        };

        if a_wins {
            // a wins, b becomes child of a
            Self::add_child(&a, b);
            a
        } else {
            // b wins, a becomes child of b
            Self::add_child(&b, a);
            b
        }
    }

    /// Adds child to parent's child list
    fn add_child(parent: &NodeRef<T, P>, child: NodeRef<T, P>) {
        let mut parent_mut = parent.borrow_mut();
        child.borrow_mut().next = parent_mut.child.take();
        parent_mut.child = Some(child);
    }

    /// Rebuilds the heap after removing the minimum.
    ///
    /// This is where the ranked linking happens, similar to Fibonacci heap consolidation.
    /// We process all children (including hollow nodes), collect non-hollow nodes,
    /// and link them by rank.
    fn delete_min_rebuild(&mut self, old_root: NodeRef<T, P>) {
        // Collect all children to process
        let mut to_process: Vec<NodeRef<T, P>> = Vec::new();

        // Add children of old root
        let first_child = old_root.borrow_mut().child.take();
        if let Some(child) = first_child {
            Self::collect_siblings(&mut to_process, child);
        }

        // Process nodes: for hollow nodes, add their children to the list
        // For non-hollow nodes, collect them for ranked linking
        let mut full_nodes: Vec<NodeRef<T, P>> = Vec::new();
        let mut idx = 0;

        while idx < to_process.len() {
            let node_rc = to_process[idx].clone();
            idx += 1;

            let is_hollow = node_rc.borrow().is_hollow();

            if is_hollow {
                // Process hollow node: add its children to the processing list
                let first_child = node_rc.borrow_mut().child.take();
                if let Some(child) = first_child {
                    // Clear the child's next links as we collect siblings
                    Self::collect_siblings(&mut to_process, child);
                }

                // Also check second_parent - if this hollow node has one,
                // we might need to handle it (but the item already moved)
                // The second_parent reference is just for the DAG structure
            } else {
                // Non-hollow node: collect for ranked linking.
                // We keep its child list intact - only the old root's direct
                // children are processed, not grandchildren.
                full_nodes.push(node_rc);
            }
        }

        if full_nodes.is_empty() {
            self.root = None;
            return;
        }

        // Perform ranked linking (like Fibonacci heap consolidation)
        let max_rank = ((self.len + 1) as f64).log2() as usize + 2;
        let mut rank_table: Vec<Option<NodeRef<T, P>>> = vec![None; max_rank + 1];

        for mut node in full_nodes {
            // Clear next pointer since we're reorganizing
            node.borrow_mut().next = None;

            let mut current_rank = node.borrow().rank;

            // Link with existing nodes of the same rank
            while current_rank < rank_table.len() {
                if let Some(other) = rank_table[current_rank].take() {
                    // Link the two nodes
                    let winner = Self::ranked_link(node, other);
                    node = winner;
                    current_rank = node.borrow().rank;
                } else {
                    break;
                }
            }

            // Ensure table is large enough
            if current_rank >= rank_table.len() {
                rank_table.resize(current_rank + 1, None);
            }
            rank_table[current_rank] = Some(node);
        }

        // Find the new minimum among remaining nodes
        let mut new_root: Option<NodeRef<T, P>> = None;
        let mut other_roots: Vec<NodeRef<T, P>> = Vec::new();

        for node_opt in rank_table.into_iter().flatten() {
            match &new_root {
                None => {
                    new_root = Some(node_opt);
                }
                Some(current_min) => {
                    if node_opt.borrow().priority < current_min.borrow().priority {
                        other_roots.push(new_root.take().unwrap());
                        new_root = Some(node_opt);
                    } else {
                        other_roots.push(node_opt);
                    }
                }
            }
        }

        // Link all other roots as children of the new root
        if let Some(ref root) = new_root {
            for other in other_roots {
                Self::add_child(root, other);
            }
        }

        self.root = new_root;
    }

    /// Collects all siblings starting from the given node
    fn collect_siblings(result: &mut Vec<NodeRef<T, P>>, first: NodeRef<T, P>) {
        let mut current = Some(first);
        while let Some(node) = current {
            let next = node.borrow_mut().next.take();
            result.push(node);
            current = next;
        }
    }

    /// Links two nodes during ranked linking, increasing the winner's rank
    fn ranked_link(a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        let a_wins = {
            let a_priority = &a.borrow().priority;
            let b_priority = &b.borrow().priority;
            a_priority <= b_priority
        };

        if a_wins {
            // a wins
            let new_rank = a.borrow().rank + 1;
            Self::add_child(&a, b);
            a.borrow_mut().rank = new_rank;
            a
        } else {
            // b wins
            let new_rank = b.borrow().rank + 1;
            Self::add_child(&b, a);
            b.borrow_mut().rank = new_rank;
            b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap: HollowHeap<&str, i32> = HollowHeap::new();

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
        let mut heap: HollowHeap<&str, i32> = HollowHeap::new();

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
        let mut heap: HollowHeap<&str, i32> = HollowHeap::new();

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
    fn test_decrease_key_root() {
        let mut heap: HollowHeap<&str, i32> = HollowHeap::new();

        let handle = heap.push_with_handle(5, "item");
        assert_eq!(heap.peek(), Some((&5, &"item")));

        // Decrease key of root node
        heap.decrease_key(&handle, 2).unwrap();
        assert_eq!(heap.peek(), Some((&2, &"item")));

        // Pop should work
        assert_eq!(heap.pop(), Some((2, "item")));
    }

    #[test]
    fn test_merge() {
        let mut heap1: HollowHeap<i32, i32> = HollowHeap::new();
        let mut heap2: HollowHeap<i32, i32> = HollowHeap::new();

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
    fn test_multiple_decrease_key() {
        let mut heap: HollowHeap<&str, i32> = HollowHeap::new();

        let handle = heap.push_with_handle(100, "item");
        heap.push(50, "other");

        // Multiple decreases on same handle
        heap.decrease_key(&handle, 80).unwrap();
        heap.decrease_key(&handle, 60).unwrap();
        heap.decrease_key(&handle, 40).unwrap();

        assert_eq!(heap.peek(), Some((&40, &"item")));

        assert_eq!(heap.pop(), Some((40, "item")));
        assert_eq!(heap.pop(), Some((50, "other")));
    }
}
