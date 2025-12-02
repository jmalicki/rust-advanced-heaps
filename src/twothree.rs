//! 2-3 Heap implementation (Takaoka)
//!
//! A 2-3 heap is a priority queue consisting of a forest of trees where each tree
//! follows the 2-3 tree structure. It provides:
//! - O(1) amortized insert
//! - O(1) amortized decrease_key (via cut-and-link)
//! - O(log n) amortized delete_min
//! - O(1) merge
//!
//! Based on: Takaoka, T.: "Theory of 2-3 heaps", Discrete Applied Mathematics 126 (2003)
//!
//! The key insight: decrease_key is O(1) because we simply cut the node with its
//! subtree from its parent and add it to the root list. No restructuring is needed
//! during decrease_key - all restructuring is deferred to delete_min.

use crate::traits::{Handle, Heap, HeapError};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::mem;
use std::rc::{Rc, Weak};

/// Type alias for node reference (strong reference for ownership)
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;

/// Type alias for weak node reference (for parent links and handles)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a 2-3 heap
///
/// Uses a Weak reference to the node. Handle stability is preserved because
/// nodes are never moved or reallocated - only tree links are updated.
pub struct TwoThreeHandle<T, P> {
    node: WeakNodeRef<T, P>,
}

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
/// Layout: metadata first (parent, children), then priority, then item.
/// This optimizes for cache locality during priority comparisons.
struct Node<T, P> {
    // Tree structure
    parent: WeakNodeRef<T, P>,
    // SmallVec with inline capacity of 3 (2-3 heap nodes have at most 3 children)
    children: SmallVec<[NodeRef<T, P>; 3]>,
    // Rank (degree) of the tree rooted at this node
    rank: usize,
    // Priority and item (accessed during comparisons and retrieval)
    priority: P,
    item: T,
}

/// 2-3 Heap - a forest of 2-3 trees
///
/// Roots are stored in a Vec for simplicity and correct memory management.
/// A separate min pointer tracks the root with minimum priority.
pub struct TwoThreeHeap<T, P: Ord> {
    /// All tree roots
    roots: Vec<NodeRef<T, P>>,
    /// Pointer to the root with minimum priority (None if empty)
    min: Option<NodeRef<T, P>>,
    /// Total number of elements
    len: usize,
}

impl<T: Clone, P: Ord + Clone> Heap<T, P> for TwoThreeHeap<T, P> {
    type Handle = TwoThreeHandle<T, P>;

    fn new() -> Self {
        Self {
            roots: Vec::with_capacity(16), // Pre-allocate some space
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

    fn push(&mut self, priority: P, item: T) -> Self::Handle {
        self.insert(priority, item)
    }

    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node as a single-node tree (rank 0)
        let node = Rc::new(RefCell::new(Node {
            parent: Weak::new(),
            children: SmallVec::new(), // Inline storage for up to 3 children
            rank: 0,
            priority,
            item,
        }));

        let handle = TwoThreeHandle {
            node: Rc::downgrade(&node),
        };

        // Add to root list
        self.add_root(node);
        self.len += 1;

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        let min = self.min.as_ref()?;
        let node_ref = min.borrow();
        unsafe {
            let priority: &P = &*(&node_ref.priority as *const P);
            let item: &T = &*(&node_ref.item as *const T);
            Some((priority, item))
        }
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    fn delete_min(&mut self) -> Option<(P, T)> {
        let min = self.min.take()?;

        // Find and remove min from roots
        let min_idx = self.roots.iter().position(|r| Rc::ptr_eq(r, &min))?;
        self.roots.swap_remove(min_idx);

        // Extract children from min and add to roots
        let children = mem::take(&mut min.borrow_mut().children);
        for child in children {
            child.borrow_mut().parent = Weak::new();
            self.roots.push(child);
        }

        // Extract item and priority
        let (priority, item) = match Rc::try_unwrap(min) {
            Ok(cell) => {
                let node = cell.into_inner();
                (node.priority, node.item)
            }
            Err(rc) => {
                let node = rc.borrow();
                (node.priority.clone(), node.item.clone())
            }
        };

        self.len -= 1;

        // Consolidate and find new min
        if !self.roots.is_empty() {
            self.consolidate();
        }

        Some((priority, item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node = handle.node.upgrade().ok_or(HeapError::InvalidHandle)?;

        {
            let current_priority = node.borrow().priority.clone();
            if new_priority >= current_priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Update priority
        node.borrow_mut().priority = new_priority.clone();

        // Check if node is a root (no parent)
        let parent_weak = node.borrow().parent.clone();
        if let Some(parent) = parent_weak.upgrade() {
            // Cut this node from parent and add to root list
            self.cut(&node, &parent);
        }

        // Update min pointer if this node now has the smallest priority - O(1)
        if let Some(ref min) = self.min {
            if new_priority < min.borrow().priority {
                self.min = Some(Rc::clone(&node));
            }
        }

        Ok(())
    }

    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            self.roots = mem::take(&mut other.roots);
            self.min = other.min.take();
            self.len = other.len;
            other.len = 0;
            return;
        }

        // Append other's roots to self
        self.roots.append(&mut other.roots);
        self.len += other.len;
        other.len = 0;

        // Update min pointer if other had smaller min - O(1)
        if let Some(other_min) = other.min.take() {
            if let Some(ref self_min) = self.min {
                if other_min.borrow().priority < self_min.borrow().priority {
                    self.min = Some(other_min);
                }
            }
        }
    }
}

impl<T: Clone, P: Ord + Clone> Default for TwoThreeHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone, P: Ord + Clone> TwoThreeHeap<T, P> {
    /// Add a root to the forest and update min if needed - O(1)
    fn add_root(&mut self, node: NodeRef<T, P>) {
        // Update min pointer if this node has smaller priority
        match self.min.as_ref() {
            None => {
                self.min = Some(Rc::clone(&node));
            }
            Some(min) => {
                if node.borrow().priority < min.borrow().priority {
                    self.min = Some(Rc::clone(&node));
                }
            }
        }

        self.roots.push(node);
    }

    /// Cut a node from its parent and add to root list (Takaoka cut)
    fn cut(&mut self, node: &NodeRef<T, P>, parent: &NodeRef<T, P>) {
        // Remove node from parent's children
        {
            let mut parent_ref = parent.borrow_mut();
            parent_ref.children.retain(|c| !Rc::ptr_eq(c, node));
        }

        // Clear node's parent link
        node.borrow_mut().parent = Weak::new();

        // Add node to root list
        self.add_root(Rc::clone(node));
    }

    /// Consolidate trees after delete_min to maintain 2-3 structure
    fn consolidate(&mut self) {
        if self.roots.is_empty() {
            return;
        }

        // Take all roots
        let roots = mem::take(&mut self.roots);
        self.min = None;

        // Use array indexed by rank to merge trees of same rank
        let max_rank = ((self.len + 1) as f64).log2().ceil() as usize + 2;
        let max_rank = max_rank.max(roots.len() + 1);
        let mut rank_array: Vec<Option<NodeRef<T, P>>> = vec![None; max_rank];

        for root in roots {
            let mut current_root = root;

            loop {
                let rank = current_root.borrow().rank;

                // Ensure rank_array is large enough
                while rank >= rank_array.len() {
                    rank_array.push(None);
                }

                match rank_array[rank].take() {
                    None => {
                        rank_array[rank] = Some(current_root);
                        break;
                    }
                    Some(other) => {
                        // Link trees: smaller priority becomes parent
                        current_root = self.link_trees(current_root, other);
                        // Continue to check if there's another tree at the new rank
                    }
                }
            }
        }

        // Rebuild roots from rank_array
        for tree in rank_array.into_iter().flatten() {
            self.add_root(tree);
        }
    }

    /// Link two trees of the same rank into one tree of rank+1
    fn link_trees(&mut self, a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        // Make the one with smaller priority the parent
        let (parent, child) = if a.borrow().priority <= b.borrow().priority {
            (a, b)
        } else {
            (b, a)
        };

        // Add child to parent's children
        child.borrow_mut().parent = Rc::downgrade(&parent);
        parent.borrow_mut().children.push(child);

        // Update parent's rank
        parent.borrow_mut().rank += 1;

        // Maintain 2-3 property: if parent has 4+ children, split
        if parent.borrow().children.len() > 3 {
            self.split_node(&parent);
        }

        parent
    }

    /// Split a node with too many children (>3) into two nodes
    fn split_node(&mut self, node: &NodeRef<T, P>) {
        let num_children = node.borrow().children.len();
        if num_children <= 3 {
            return;
        }

        // Take children and sort by priority
        let mut children: SmallVec<[NodeRef<T, P>; 3]> = mem::take(&mut node.borrow_mut().children);
        children.sort_by(|a, b| a.borrow().priority.cmp(&b.borrow().priority));

        // Keep first 2 children with node
        let second_half: SmallVec<[NodeRef<T, P>; 3]> = children.drain(2..).collect();
        node.borrow_mut().children = children;

        // Update rank for node (based on remaining children)
        let max_child_rank = node
            .borrow()
            .children
            .iter()
            .map(|c| c.borrow().rank)
            .max()
            .unwrap_or(0);
        node.borrow_mut().rank = max_child_rank + 1;

        // Update parent references for remaining children
        for child in &node.borrow().children {
            child.borrow_mut().parent = Rc::downgrade(node);
        }

        // The child with minimum priority in second_half becomes a new sibling tree
        let min_idx = second_half
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.borrow().priority.cmp(&b.borrow().priority))
            .map(|(i, _)| i)
            .unwrap();

        let mut second_half = second_half;
        let sibling = second_half.remove(min_idx);

        // Remaining children become children of sibling
        for child in second_half {
            child.borrow_mut().parent = Rc::downgrade(&sibling);
            sibling.borrow_mut().children.push(child);
        }

        // Update sibling's rank
        let sibling_max_rank = sibling
            .borrow()
            .children
            .iter()
            .map(|c| c.borrow().rank)
            .max()
            .unwrap_or(0);
        sibling.borrow_mut().rank = sibling_max_rank + 1;

        // Handle based on whether node has a parent
        let parent_weak = node.borrow().parent.clone();
        if let Some(parent) = parent_weak.upgrade() {
            // Add sibling as child of parent
            sibling.borrow_mut().parent = Rc::downgrade(&parent);
            parent.borrow_mut().children.push(Rc::clone(&sibling));

            // Recursively check if parent needs splitting
            if parent.borrow().children.len() > 3 {
                self.split_node(&parent);
            }
        } else {
            // Node is a root - sibling becomes a new root
            sibling.borrow_mut().parent = Weak::new();
            self.roots.push(sibling);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = TwoThreeHeap::new();

        let _h1 = heap.insert(5, "five");
        let _h2 = heap.insert(3, "three");
        let _h3 = heap.insert(7, "seven");

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some((&3, &"three")));

        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.peek(), Some((&5, &"five")));
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = TwoThreeHeap::new();

        let _h1 = heap.insert(10, "ten");
        let _h2 = heap.insert(5, "five");
        let h3 = heap.insert(15, "fifteen");

        assert_eq!(heap.peek(), Some((&5, &"five")));

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

    #[test]
    fn test_decrease_key_to_min() {
        let mut heap = TwoThreeHeap::new();

        let h1 = heap.insert(10, "ten");
        let _h2 = heap.insert(5, "five");
        let _h3 = heap.insert(15, "fifteen");

        // Decrease h1 to be the new minimum
        heap.decrease_key(&h1, 1).unwrap();
        assert_eq!(heap.peek(), Some((&1, &"ten")));

        // Pop should return the decreased element
        assert_eq!(heap.pop(), Some((1, "ten")));
    }

    #[test]
    fn test_many_operations() {
        let mut heap = TwoThreeHeap::new();
        let mut handles = Vec::new();

        // Insert many elements
        for i in 0..100 {
            handles.push(heap.insert(i + 100, i));
        }

        assert_eq!(heap.len(), 100);
        assert_eq!(heap.peek(), Some((&100, &0)));

        // Decrease some keys
        heap.decrease_key(&handles[50], 10).unwrap();
        assert_eq!(heap.peek(), Some((&10, &50)));

        heap.decrease_key(&handles[75], 5).unwrap();
        assert_eq!(heap.peek(), Some((&5, &75)));

        // Pop elements and verify order
        let (p, v) = heap.pop().unwrap();
        assert_eq!(p, 5);
        assert_eq!(v, 75);
    }
}
