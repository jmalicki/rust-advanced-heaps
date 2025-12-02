//! 2-3 Heap implementation
//!
//! A 2-3 heap is a balanced tree where each internal node has either 2 or 3 children.
//! It provides:
//! - O(1) amortized insert and decrease_key
//! - O(log n) amortized delete_min
//!
//! The 2-3 structure ensures balance while allowing efficient decrease_key operations.
//!
//! This implementation uses Rc/Weak references for memory safety:
//! - Strong references (Rc) point from parent to children
//! - Weak references point from children back to parents
//! - Handles point directly to nodes; bubble-up swaps node positions, not contents

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::mem;
use std::rc::{Rc, Weak};

/// Type alias for node reference (strong reference for ownership)
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;

/// Type alias for weak node reference (for parent links and handles)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a 2-3 heap
///
/// Uses a Weak reference to the node, which remains valid because bubble-up
/// swaps node positions in the tree rather than swapping contents.
pub struct TwoThreeHandle<T, P> {
    node: WeakNodeRef<T, P>,
}

// Manual Clone implementation to avoid requiring T: Clone, P: Clone
impl<T, P> Clone for TwoThreeHandle<T, P> {
    fn clone(&self) -> Self {
        TwoThreeHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for TwoThreeHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.node, &other.node)
    }
}

impl<T, P> Eq for TwoThreeHandle<T, P> {}

impl<T, P> std::fmt::Debug for TwoThreeHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoThreeHandle")
            .field("node", &Weak::as_ptr(&self.node))
            .finish()
    }
}

impl<T, P> Handle for TwoThreeHandle<T, P> {}

/// Node in the 2-3 tree
///
/// Layout optimized for cache locality: metadata first, then priority, then item.
/// When traversing the heap, we often only need to compare priorities.
struct Node<T, P> {
    // Metadata for tree structure (accessed during traversal)
    parent: WeakNodeRef<T, P>,
    children: Vec<NodeRef<T, P>>,
    // Priority for ordering (accessed during comparisons)
    priority: P,
    // Item data (accessed only when we've found the target node)
    item: T,
}

/// 2-3 Heap
pub struct TwoThreeHeap<T, P: Ord> {
    root: Option<NodeRef<T, P>>,
    len: usize,
}

impl<T: Clone, P: Ord + Clone> Heap<T, P> for TwoThreeHeap<T, P> {
    type Handle = TwoThreeHandle<T, P>;

    fn new() -> Self {
        Self { root: None, len: 0 }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) -> Self::Handle {
        self.insert(priority, item)
    }

    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node with item inline
        let node = Rc::new(RefCell::new(Node {
            parent: Weak::new(),
            children: Vec::new(),
            priority: priority.clone(),
            item,
        }));

        let handle = TwoThreeHandle {
            node: Rc::downgrade(&node),
        };

        if let Some(ref root) = self.root {
            let new_priority = priority;
            let root_priority = root.borrow().priority.clone();

            if new_priority < root_priority {
                // New node becomes root
                let old_root = self.root.take().unwrap();
                old_root.borrow_mut().parent = Rc::downgrade(&node);
                node.borrow_mut().children.push(old_root);
                self.root = Some(Rc::clone(&node));
            } else {
                // Insert as child of root
                self.insert_as_child(Rc::clone(root), Rc::clone(&node));
            }
        } else {
            self.root = Some(Rc::clone(&node));
        }

        self.maintain_structure(Rc::clone(&node));
        self.len += 1;

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        let root = self.root.as_ref()?;
        let node = root.borrow();
        unsafe {
            let priority: &P = &*(&node.priority as *const P);
            let item: &T = &*(&node.item as *const T);
            Some((priority, item))
        }
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    fn delete_min(&mut self) -> Option<(P, T)> {
        let root = self.root.take()?;

        // Get children from root
        let children: Vec<_> = mem::take(&mut root.borrow_mut().children);

        // Extract item and priority from root
        let (priority, item) = match Rc::try_unwrap(root) {
            Ok(cell) => {
                let node = cell.into_inner();
                (node.priority, node.item)
            }
            Err(rc) => {
                // Node still referenced by handle - clone the data
                let node = rc.borrow();
                (node.priority.clone(), node.item.clone())
            }
        };

        self.len -= 1;

        if children.is_empty() {
            self.root = None;
        } else {
            // Clear parent references
            for child in &children {
                child.borrow_mut().parent = Weak::new();
            }
            self.root = Some(self.rebuild_from_children(children));
        }

        Some((priority, item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node = handle
            .node
            .upgrade()
            .ok_or(HeapError::InvalidHandle)?;

        {
            let current_priority = node.borrow().priority.clone();
            if new_priority >= current_priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Update priority
        node.borrow_mut().priority = new_priority;

        // Bubble up by swapping node positions
        self.bubble_up(Rc::clone(&node));

        Ok(())
    }

    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            self.root = other.root.take();
            self.len = other.len;
            other.len = 0;
            return;
        }

        let other_root = other.root.take().unwrap();
        let self_root = self.root.as_ref().unwrap();

        let self_priority = self_root.borrow().priority.clone();
        let other_priority = other_root.borrow().priority.clone();

        if other_priority < self_priority {
            // Other root becomes new root
            let old_self_root = self.root.take().unwrap();
            old_self_root.borrow_mut().parent = Rc::downgrade(&other_root);
            other_root.borrow_mut().children.push(old_self_root);
            self.root = Some(other_root);
        } else {
            // Add other_root as child of self_root
            other_root.borrow_mut().parent = Rc::downgrade(self_root);
            self_root.borrow_mut().children.push(other_root);
        }

        self.len += other.len;
        other.len = 0;

        // Maintain 2-3 structure
        if let Some(ref root) = self.root {
            self.maintain_structure(Rc::clone(root));
        }
    }
}

impl<T: Clone, P: Ord + Clone> Default for TwoThreeHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, P: Ord + Clone> TwoThreeHeap<T, P> {
    /// Counts all nodes in the heap (debug only)
    #[cfg(feature = "expensive_verify")]
    fn count_all_nodes(&self) -> usize {
        fn count_subtree<T, P>(node: &NodeRef<T, P>) -> usize {
            let node_ref = node.borrow();
            let mut count = 1;
            for child in &node_ref.children {
                count += count_subtree(child);
            }
            count
        }

        match &self.root {
            Some(root) => count_subtree(root),
            None => 0,
        }
    }

    /// Insert a node as a child of parent
    fn insert_as_child(&self, parent: NodeRef<T, P>, node: NodeRef<T, P>) {
        node.borrow_mut().parent = Rc::downgrade(&parent);
        parent.borrow_mut().children.push(node);
    }

    /// Maintain 2-3 structure by splitting nodes with too many children
    fn maintain_structure(&mut self, node: NodeRef<T, P>) {
        let num_children = node.borrow().children.len();

        if num_children <= 3 {
            // Check parent
            let parent_weak = node.borrow().parent.clone();
            if let Some(parent) = parent_weak.upgrade() {
                self.maintain_structure(parent);
            }
            return;
        }

        // Node has too many children (4+), need to split
        let mut children: Vec<_> = mem::take(&mut node.borrow_mut().children);

        // Sort children by priority for balanced split
        children.sort_by(|a, b| a.borrow().priority.cmp(&b.borrow().priority));

        // Split: first 2 children stay with node, rest go to new sibling
        let mut second_half: Vec<_> = children.drain(2..).collect();
        node.borrow_mut().children = children;

        // Update parent references for remaining children
        for child in &node.borrow().children {
            child.borrow_mut().parent = Rc::downgrade(&node);
        }

        // The child with minimum priority in second_half becomes the sibling
        // This preserves handle stability - the node identity doesn't change
        let min_child_idx = second_half
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.borrow().priority.cmp(&b.borrow().priority))
            .map(|(i, _)| i)
            .unwrap();

        // Remove min_child from second_half - it becomes the sibling
        let sibling = second_half.remove(min_child_idx);
        sibling.borrow_mut().parent = node.borrow().parent.clone();

        // Add remaining children of second_half as children of sibling
        for child in second_half {
            child.borrow_mut().parent = Rc::downgrade(&sibling);
            sibling.borrow_mut().children.push(child);
        }

        // Add sibling to parent
        let parent_weak = node.borrow().parent.clone();
        if let Some(parent) = parent_weak.upgrade() {
            sibling.borrow_mut().parent = Rc::downgrade(&parent);
            parent.borrow_mut().children.push(sibling);
            self.maintain_structure(parent);
        } else {
            // Node was root - we need to create a new root above both node and sibling
            // The new root's item should be the minimum of node and sibling
            // But since we're just reorganizing, we'll swap if needed

            if sibling.borrow().priority < node.borrow().priority {
                // Sibling has smaller priority, it should be the root
                // Make node a child of sibling
                node.borrow_mut().parent = Rc::downgrade(&sibling);
                sibling.borrow_mut().children.insert(0, Rc::clone(&node));
                sibling.borrow_mut().parent = Weak::new();
                self.root = Some(sibling);
            } else {
                // Node has smaller or equal priority, it stays as root
                // Make sibling a child of node
                sibling.borrow_mut().parent = Rc::downgrade(&node);
                node.borrow_mut().children.push(sibling);
                // node is already the root
                self.root = Some(Rc::clone(&node));
            }
        }
    }

    /// Bubble up a node by swapping its position with ancestors that have larger priorities
    fn bubble_up(&mut self, node: NodeRef<T, P>) {
        let current = node;

        loop {
            let parent_weak = current.borrow().parent.clone();
            let parent = match parent_weak.upgrade() {
                Some(p) => p,
                None => break, // Reached root
            };

            let should_swap = current.borrow().priority < parent.borrow().priority;

            if !should_swap {
                break;
            }

            // Swap node positions: current moves to parent's position
            self.swap_node_with_parent(&current, &parent);

            // current is now in parent's old position, continue to check grandparent
        }
    }

    /// Swap a child node with its parent in the tree structure
    fn swap_node_with_parent(&mut self, child: &NodeRef<T, P>, parent: &NodeRef<T, P>) {
        // Step 1: Remove child from parent's children list
        {
            let mut parent_ref = parent.borrow_mut();
            parent_ref.children.retain(|c| !Rc::ptr_eq(c, child));
        }

        // Step 2: Get parent's old position info
        let grandparent_weak = parent.borrow().parent.clone();
        let parent_siblings: Vec<NodeRef<T, P>>;

        // Step 3: Child takes parent's position
        child.borrow_mut().parent = grandparent_weak.clone();

        // Step 4: Update grandparent's children (or root if parent was root)
        if let Some(gp) = grandparent_weak.upgrade() {
            // Replace parent with child in grandparent's children
            let mut gp_ref = gp.borrow_mut();
            for i in 0..gp_ref.children.len() {
                if Rc::ptr_eq(&gp_ref.children[i], parent) {
                    gp_ref.children[i] = Rc::clone(child);
                    break;
                }
            }
            parent_siblings = Vec::new(); // Not used in this path
        } else {
            // Parent was root - child becomes new root
            self.root = Some(Rc::clone(child));
            parent_siblings = Vec::new();
        }
        let _ = parent_siblings; // Silence unused warning

        // Step 5: Parent becomes a child of child
        // Move child's old children to parent (they become siblings of what was child)
        let child_old_children: Vec<_> = mem::take(&mut child.borrow_mut().children);

        // Parent keeps its remaining children, gets child's old children too
        for old_child in child_old_children {
            old_child.borrow_mut().parent = Rc::downgrade(parent);
            parent.borrow_mut().children.push(old_child);
        }

        // Parent becomes child of the (now promoted) child
        parent.borrow_mut().parent = Rc::downgrade(child);
        child.borrow_mut().children.push(Rc::clone(parent));
    }

    /// Rebuild tree from a list of child nodes after root deletion
    fn rebuild_from_children(&mut self, children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.len() == 1 {
            let root = Rc::clone(&children[0]);
            root.borrow_mut().parent = Weak::new();
            return root;
        }

        // Find child with minimum priority - it becomes the new root
        let min_idx = children
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.borrow().priority.cmp(&b.borrow().priority))
            .map(|(i, _)| i)
            .unwrap();

        let min = Rc::clone(&children[min_idx]);
        min.borrow_mut().parent = Weak::new();

        // Set as temporary root
        self.root = Some(Rc::clone(&min));

        // Add other children as children of the new root
        for (i, child) in children.into_iter().enumerate() {
            if i != min_idx {
                child.borrow_mut().parent = Rc::downgrade(&min);
                min.borrow_mut().children.push(child);
            }
        }

        // Maintain 2-3 structure if needed
        if min.borrow().children.len() > 3 {
            self.maintain_structure(Rc::clone(&min));
        }

        self.root.as_ref().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = TwoThreeHeap::new();

        let h1 = heap.insert(5, "five");
        let h2 = heap.insert(3, "three");
        let h3 = heap.insert(7, "seven");

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some((&3, &"three")));

        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.peek(), Some((&5, &"five")));
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = TwoThreeHeap::new();

        let h1 = heap.insert(10, "ten");
        let h2 = heap.insert(5, "five");
        let h3 = heap.insert(15, "fifteen");

        assert_eq!(heap.peek(), Some((&5, &"five")));

        // Decrease h3's priority to make it the minimum
        heap.decrease_key(&h3, 1).unwrap();
        assert_eq!(heap.peek(), Some((&1, &"fifteen")));
    }

    #[test]
    fn test_merge() {
        let mut heap1 = TwoThreeHeap::new();
        heap1.insert(5, "five");
        heap1.insert(3, "three");

        let mut heap2 = TwoThreeHeap::new();
        heap2.insert(1, "one");
        heap2.insert(7, "seven");

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.peek(), Some((&1, &"one")));
    }
}
