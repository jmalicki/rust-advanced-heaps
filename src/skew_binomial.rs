//! Skew Binomial Heap implementation
//!
//! A skew binomial heap extends binomial heaps with additional flexibility:
//! - O(1) insert and merge
//! - O(log n) delete_min
//! - O(log n) decrease_key
//!
//! Skew binomial heaps allow more flexible tree structures than standard
//! binomial heaps while maintaining efficient operations.
//!
//! This implementation uses Rc/Weak references instead of raw pointers,
//! providing memory safety for tree structure management.

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Type alias for strong node reference
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;
/// Type alias for weak node reference (used for parent backlinks)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;
/// Type alias for optional strong node reference
type OptNodeRef<T, P> = Option<NodeRef<T, P>>;

/// Handle to an element in a Skew binomial heap
#[derive(Debug)]
pub struct SkewBinomialHandle<T, P> {
    node: WeakNodeRef<T, P>,
}

impl<T, P> Clone for SkewBinomialHandle<T, P> {
    fn clone(&self) -> Self {
        SkewBinomialHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for SkewBinomialHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node.ptr_eq(&other.node)
    }
}

impl<T, P> Eq for SkewBinomialHandle<T, P> {}

impl<T, P> Handle for SkewBinomialHandle<T, P> {}

struct Node<T, P> {
    item: Option<T>,
    priority: Option<P>,
    parent: WeakNodeRef<T, P>,
    child: OptNodeRef<T, P>,
    sibling: OptNodeRef<T, P>,
    rank: usize,
    skew: bool,
}

/// Skew Binomial Heap
pub struct SkewBinomialHeap<T, P: Ord> {
    trees: Vec<OptNodeRef<T, P>>,
    min: OptNodeRef<T, P>,
    len: usize,
}

impl<T, P: Ord> Heap<T, P> for SkewBinomialHeap<T, P> {
    type Handle = SkewBinomialHandle<T, P>;

    fn new() -> Self {
        Self {
            trees: Vec::new(),
            min: None,
            len: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.min.is_none()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) -> Self::Handle {
        self.insert(priority, item)
    }

    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Rc::new(RefCell::new(Node {
            item: Some(item),
            priority: Some(priority),
            parent: Weak::new(),
            child: None,
            sibling: None,
            rank: 0,
            skew: true,
        }));

        let handle = SkewBinomialHandle {
            node: Rc::downgrade(&node),
        };

        self.insert_tree(node);
        self.len += 1;

        // Update min pointer AFTER insert_tree, because during insert_tree
        // the node may become a child of another node during linking.
        // find_and_update_min scans roots to find the actual minimum.
        self.find_and_update_min();

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        self.min.as_ref().map(|min_ref| unsafe {
            let ptr = min_ref.as_ptr();
            (
                (*ptr).priority.as_ref().unwrap_unchecked(),
                (*ptr).item.as_ref().unwrap_unchecked(),
            )
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    fn delete_min(&mut self) -> Option<(P, T)> {
        let min_ref = self.min.take()?;

        // Clear from trees
        let rank = min_ref.borrow().rank;
        if rank < self.trees.len() {
            self.trees[rank] = None;
        }

        // Collect all children
        let first_child = min_ref.borrow_mut().child.take();
        let mut children_to_insert: Vec<NodeRef<T, P>> = Vec::new();

        if let Some(first) = first_child {
            let mut current = Some(first);
            while let Some(curr) = current {
                let next = curr.borrow_mut().sibling.take();
                curr.borrow_mut().parent = Weak::new();
                children_to_insert.push(curr);
                current = next;
            }
        }

        let (priority, item) = {
            let mut node = min_ref.borrow_mut();
            (node.priority.take().unwrap(), node.item.take().unwrap())
        };

        drop(min_ref);

        for child in children_to_insert {
            self.insert_tree(child);
        }

        self.find_and_update_min();
        self.len -= 1;

        // Verify invariants after delete_min
        #[cfg(debug_assertions)]
        {
            let count = self.count_all_nodes();
            assert_eq!(
                count, self.len,
                "Length mismatch after delete_min: counted {} nodes but len is {}",
                count, self.len
            );
        }

        Some((priority, item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node_ref = handle
            .node
            .upgrade()
            .ok_or(HeapError::PriorityNotDecreased)?;

        // Check if node has been deleted (priority is None)
        {
            let node = node_ref.borrow();
            if node.priority.is_none() {
                return Err(HeapError::PriorityNotDecreased);
            }
            if new_priority >= *node.priority.as_ref().unwrap() {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Check if node is a root (no parent)
        let has_parent = node_ref.borrow().parent.upgrade().is_some();

        if !has_parent {
            // Node is a root, just update priority and min
            node_ref.borrow_mut().priority = Some(new_priority);
            self.find_and_update_min();
            return Ok(());
        }

        // Cut the node from its parent and reinsert
        self.cut_and_reinsert(node_ref, new_priority);

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

        let mut other_trees: Vec<NodeRef<T, P>> = Vec::new();
        for tree_opt in other.trees.iter_mut() {
            if let Some(tree) = tree_opt.take() {
                other_trees.push(tree);
            }
        }

        for tree in other_trees {
            self.insert_tree(tree);
        }

        self.len += other.len;
        self.find_and_update_min();

        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> SkewBinomialHeap<T, P> {
    /// Counts all nodes reachable from the trees array (debug only)
    #[cfg(debug_assertions)]
    fn count_all_nodes(&self) -> usize {
        let mut count = 0;
        for tree in self.trees.iter().flatten() {
            count += Self::count_subtree(tree);
        }
        count
    }

    #[cfg(debug_assertions)]
    fn count_subtree(node: &NodeRef<T, P>) -> usize {
        let node_ref = node.borrow();
        // Verify node has valid priority
        assert!(
            node_ref.priority.is_some(),
            "Found node with None priority in tree"
        );

        let mut count = 1;

        // Count children
        let mut child_opt = node_ref.child.clone();
        drop(node_ref);

        while let Some(child) = child_opt {
            count += Self::count_subtree(&child);
            child_opt = child.borrow().sibling.clone();
        }

        count
    }

    /// Cuts a node from its parent and reinserts it with a new priority
    fn cut_and_reinsert(&mut self, node: NodeRef<T, P>, new_priority: P) {
        // Get parent before we modify anything
        let parent_weak = node.borrow().parent.clone();
        let parent = parent_weak.upgrade().unwrap();

        // Remove node from parent's child list
        let mut found = false;
        {
            let parent_child = parent.borrow().child.clone();

            if let Some(ref first_child) = parent_child {
                if Rc::ptr_eq(first_child, &node) {
                    // Node is first child - update parent's child pointer
                    let next_sibling = node.borrow().sibling.clone();
                    parent.borrow_mut().child = next_sibling;
                    found = true;
                } else {
                    // Search through sibling list
                    let mut prev = Rc::clone(first_child);
                    loop {
                        let next = prev.borrow().sibling.clone();
                        match next {
                            Some(ref next_node) if Rc::ptr_eq(next_node, &node) => {
                                // Found it - remove from list
                                let skip_to = node.borrow().sibling.clone();
                                prev.borrow_mut().sibling = skip_to;
                                found = true;
                                break;
                            }
                            Some(next_node) => {
                                prev = next_node;
                            }
                            None => break,
                        }
                    }
                }
            }
        }

        if !found {
            // Node might already be detached, just update priority
            node.borrow_mut().priority = Some(new_priority);
            node.borrow_mut().parent = Weak::new();
            self.find_and_update_min();
            return;
        }

        // Get parent's old rank before modifying
        let parent_old_rank = parent.borrow().rank;
        let parent_is_root = parent.borrow().parent.upgrade().is_none();

        // Update parent's rank (one less child)
        {
            parent.borrow_mut().rank = parent_old_rank.saturating_sub(1);
        }

        // If parent is a root, it needs to be repositioned in trees
        if parent_is_root && parent_old_rank < self.trees.len() {
            // Remove parent from old position
            self.trees[parent_old_rank] = None;
            // Reinsert parent at correct rank
            self.insert_tree(Rc::clone(&parent));
        }

        // Clear node's parent and sibling (keep children!)
        node.borrow_mut().parent = Weak::new();
        node.borrow_mut().sibling = None;

        // Update priority
        node.borrow_mut().priority = Some(new_priority);

        // Reinsert the node (with its subtree intact) as a new tree
        self.insert_tree(node);

        // Update minimum
        self.find_and_update_min();
    }

    fn insert_tree(&mut self, mut tree: NodeRef<T, P>) {
        loop {
            let rank = tree.borrow().rank;

            while self.trees.len() <= rank {
                self.trees.push(None);
            }

            if self.trees[rank].is_some() {
                let existing = self.trees[rank].take().unwrap();
                tree = Self::link_trees(existing, tree);
            } else {
                self.trees[rank] = Some(tree);
                break;
            }
        }
    }

    fn link_trees(a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        let a_priority_greater = {
            let a_b = a.borrow();
            let b_b = b.borrow();
            // Debug assertions to catch nodes with invalid priorities
            debug_assert!(
                a_b.priority.is_some(),
                "link_trees: node 'a' has None priority"
            );
            debug_assert!(
                b_b.priority.is_some(),
                "link_trees: node 'b' has None priority"
            );
            a_b.priority.as_ref().unwrap() > b_b.priority.as_ref().unwrap()
        };

        if a_priority_greater {
            return Self::link_trees(b, a);
        }

        {
            let a_child = a.borrow().child.clone();
            let mut b_mut = b.borrow_mut();
            b_mut.parent = Rc::downgrade(&a);
            b_mut.sibling = a_child;
        }

        {
            let mut a_mut = a.borrow_mut();
            a_mut.child = Some(Rc::clone(&b));
            a_mut.rank += 1;

            let b_skew = b.borrow().skew;
            a_mut.skew = b_skew && a_mut.rank > 0;
        }

        a
    }

    fn find_and_update_min(&mut self) {
        self.min = None;
        for root_opt in self.trees.iter().flatten() {
            let should_update = match &self.min {
                Some(min_ref) => {
                    root_opt.borrow().priority.as_ref().unwrap()
                        < min_ref.borrow().priority.as_ref().unwrap()
                }
                None => true,
            };

            if should_update {
                self.min = Some(Rc::clone(root_opt));
            }
        }
    }
}

impl<T, P: Ord> Default for SkewBinomialHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}
