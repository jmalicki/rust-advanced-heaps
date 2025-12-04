//! 2-3 Heap implementation
//!
//! A 2-3 heap is a priority queue data structure designed by Tadao Takaoka in 1999.
//! It provides:
//! - O(1) amortized insert and decrease_key
//! - O(log n) amortized delete_min
//!
//! The 2-3 heap maintains a forest of heap-ordered trees, similar to binomial or
//! Fibonacci heaps, but uses a "trunk" structure where two trees of the same
//! dimension can be paired together.
//!
//! # References
//!
//! Takaoka, T. (1999). "Theory of 2-3 Heaps". COCOON 1999, LNCS 1627, pp. 41-50.
//! Also published in Discrete Applied Mathematics 126 (2003), pp. 115-128.
//!
//! # Ownership Model
//!
//! Each node has exactly ONE strong Rc reference:
//! - Root nodes: owned by `trees[dim]`
//! - Children: owned by parent's `child` field (first) or prev sibling's `sibling` field
//! - Extra partner: owned by primary partner's `partner` field
//!
//! Weak references are used for back-pointers (parent, prev, partner_back).
//! All internal Weak refs are guaranteed valid while the node is in the heap.
//! Operations move ownership rather than cloning Rc.

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Strong reference to a node
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;
/// Weak reference to a node (for backlinks)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a 2-3 heap
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

/// Node in the 2-3 heap forest
struct Node<T, P> {
    item: T,
    priority: P,
    dim: usize,

    // Strong refs (ownership):
    child: Option<NodeRef<T, P>>,   // First child (owns it)
    sibling: Option<NodeRef<T, P>>, // Next sibling (owns it)
    partner: Option<NodeRef<T, P>>, // Extra partner (primary owns extra)

    // Weak refs (navigation, always valid while in heap):
    parent: WeakNodeRef<T, P>,       // Parent node
    prev: WeakNodeRef<T, P>,         // Previous sibling or parent
    partner_back: WeakNodeRef<T, P>, // Primary partner (extra uses this)

    extra: bool, // Is this the extra partner in a trunk?
}

impl<T, P> Node<T, P> {
    fn new(item: T, priority: P) -> Self {
        Node {
            item,
            priority,
            dim: 0,
            child: None,
            sibling: None,
            partner: None,
            parent: Weak::new(),
            prev: Weak::new(),
            partner_back: Weak::new(),
            extra: false,
        }
    }

    /// Clear all links (called before melding a cut node)
    fn clear_links(&mut self) {
        self.parent = Weak::new();
        self.prev = Weak::new();
        self.sibling = None;
        // Keep child, partner, partner_back - subtree stays intact
    }
}

/// 2-3 Heap
pub struct TwoThreeHeap<T, P: Ord> {
    /// Forest of trees indexed by dimension
    trees: Vec<Option<NodeRef<T, P>>>,
    len: usize,
    /// Bitmask of which dimensions have trees
    tree_mask: usize,
}

impl<T, P: Ord + Clone> Heap<T, P> for TwoThreeHeap<T, P> {
    type Handle = TwoThreeHandle<T, P>;

    fn new() -> Self {
        Self {
            trees: Vec::new(),
            len: 0,
            tree_mask: 0,
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
        let node = Rc::new(RefCell::new(Node::new(item, priority)));
        let handle = TwoThreeHandle {
            node: Rc::downgrade(&node),
        };
        self.meld_node(node);
        self.len += 1;
        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        let min_node = self.find_min_node()?;
        let node = min_node.borrow();
        // SAFETY: The returned references are valid for the lifetime of `&self`:
        // 1. The caller holds `&self`, preventing any `&mut self` methods
        // 2. All mutating operations (push, pop, decrease_key, merge) require `&mut self`
        // 3. Therefore no `borrow_mut()` can occur on any RefCell while these refs exist
        // 4. The data is kept alive by the Rc in `self.trees[]` which outlives `&self`
        // 5. The Ref guard (`node`) is dropped here, but the data remains valid because
        //    no mutation can occur while `&self` is held
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
        if self.is_empty() {
            return None;
        }

        // Find and remove min from trees[]
        let (min_node, dim) = self.extract_min_node()?;

        // Handle partner: it becomes standalone at this dimension
        if let Some(partner) = min_node.borrow_mut().partner.take() {
            partner.borrow_mut().partner_back = Weak::new();
            partner.borrow_mut().extra = false;
            self.trees[dim] = Some(partner);
            self.tree_mask |= 1 << dim;
        }

        self.len -= 1;

        // Meld children back into forest (move through sibling chain)
        let mut child_opt = min_node.borrow_mut().child.take();
        while let Some(child) = child_opt {
            let next = child.borrow_mut().sibling.take();
            child.borrow_mut().clear_links();
            self.meld_node(child);
            child_opt = next;
        }

        // Extract data - into_inner succeeds when refcount is 1, which our
        // single ownership model guarantees (Weak refs don't count)
        let node = Rc::into_inner(min_node)
            .expect("node should have single owner after removal from heap")
            .into_inner();

        Some((node.priority, node.item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node = handle
            .node
            .upgrade()
            .ok_or(HeapError::PriorityNotDecreased)?;

        if new_priority >= node.borrow().priority {
            return Err(HeapError::PriorityNotDecreased);
        }

        node.borrow_mut().priority = new_priority;

        // If node is a root (not a child and not extra), nothing more to do
        let is_root = node.borrow().parent.upgrade().is_none() && !node.borrow().extra;
        if is_root {
            return Ok(());
        }

        // Cut and meld
        self.cut_and_meld(Rc::clone(&node));
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

        // Move all trees from other into self
        for i in 0..other.trees.len() {
            if let Some(tree) = other.trees[i].take() {
                self.meld_node(tree);
            }
        }
        self.len += other.len;
        other.len = 0;
    }
}

impl<T, P: Ord + Clone> Default for TwoThreeHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, P: Ord + Clone> TwoThreeHeap<T, P> {
    /// Find the minimum node (returns a clone since node stays in tree)
    fn find_min_node(&self) -> Option<NodeRef<T, P>> {
        if self.tree_mask == 0 {
            return None;
        }

        let mut min_node: Option<NodeRef<T, P>> = None;
        // Keep the Ref guard alive across iterations so we can safely reference its priority
        let mut min_borrow: Option<std::cell::Ref<'_, Node<T, P>>> = None;

        let mut mask = self.tree_mask;
        while mask != 0 {
            let dim = mask.trailing_zeros() as usize;
            mask &= !(1 << dim);

            if let Some(ref tree) = self.trees[dim] {
                let tree_borrow = tree.borrow();
                let dominated = min_borrow
                    .as_ref()
                    .is_some_and(|mb| tree_borrow.priority >= mb.priority);
                if !dominated {
                    min_borrow = Some(tree_borrow);
                    min_node = Some(Rc::clone(tree));
                }
            }
        }

        min_node
    }

    /// Extract the minimum node from the forest (removes from trees[])
    fn extract_min_node(&mut self) -> Option<(NodeRef<T, P>, usize)> {
        if self.tree_mask == 0 {
            return None;
        }

        let mut min_dim = 0;
        let mut min_priority: Option<P> = None;

        let mut mask = self.tree_mask;
        while mask != 0 {
            let dim = mask.trailing_zeros() as usize;
            mask &= !(1 << dim);

            if let Some(ref tree) = self.trees[dim] {
                let priority = tree.borrow().priority.clone();
                let is_smaller = min_priority.as_ref().is_none_or(|mp| priority < *mp);
                if is_smaller {
                    min_priority = Some(priority);
                    min_dim = dim;
                }
            }
        }

        let min_node = self.trees[min_dim].take()?;
        self.tree_mask &= !(1 << min_dim);
        Some((min_node, min_dim))
    }

    /// Meld a tree into the forest (takes ownership)
    fn meld_node(&mut self, mut tree: NodeRef<T, P>) {
        loop {
            let dim = tree.borrow().dim;

            while self.trees.len() <= dim {
                self.trees.push(None);
            }

            match self.trees[dim].take() {
                None => {
                    // Empty slot - place tree here
                    self.trees[dim] = Some(tree);
                    self.tree_mask |= 1 << dim;
                    return;
                }
                Some(existing) => {
                    self.tree_mask &= !(1 << dim);

                    let existing_has_partner = existing.borrow().partner.is_some();
                    let tree_has_partner = tree.borrow().partner.is_some();

                    if !existing_has_partner && !tree_has_partner {
                        // Form a trunk (pair them)
                        let (primary, extra_node) =
                            if tree.borrow().priority <= existing.borrow().priority {
                                (tree, existing)
                            } else {
                                (existing, tree)
                            };

                        extra_node.borrow_mut().extra = true;
                        extra_node.borrow_mut().partner_back = Rc::downgrade(&primary);
                        primary.borrow_mut().partner = Some(extra_node);

                        self.trees[dim] = Some(primary);
                        self.tree_mask |= 1 << dim;
                        return;
                    } else {
                        // Merge into higher dimension
                        tree = self.link_trees(tree, existing);
                        // Continue loop with merged tree
                    }
                }
            }
        }
    }

    /// Link two trees, making one a child of the other
    /// Takes ownership of both, returns the winner with increased dimension
    fn link_trees(&mut self, a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        // Determine parent (smaller priority) and child
        let (parent, child) = if a.borrow().priority <= b.borrow().priority {
            (a, b)
        } else {
            (b, a)
        };

        // Release partners - they become separate trees
        if let Some(parent_partner) = parent.borrow_mut().partner.take() {
            parent_partner.borrow_mut().partner_back = Weak::new();
            parent_partner.borrow_mut().extra = false;
            self.meld_node(parent_partner);
        }

        if let Some(child_partner) = child.borrow_mut().partner.take() {
            child_partner.borrow_mut().partner_back = Weak::new();
            child_partner.borrow_mut().extra = false;
            self.meld_node(child_partner);
        }

        // Add child to parent
        self.add_child(&parent, child);
        parent.borrow_mut().dim += 1;

        parent
    }

    /// Add a node as a child of parent (moves ownership to parent)
    fn add_child(&self, parent: &NodeRef<T, P>, child: NodeRef<T, P>) {
        child.borrow_mut().parent = Rc::downgrade(parent);
        child.borrow_mut().extra = false;
        child.borrow_mut().prev = Rc::downgrade(parent);

        // Insert at front of child list - move ownership, don't clone
        let first_child = parent.borrow_mut().child.take();
        if let Some(ref first) = first_child {
            first.borrow_mut().prev = Rc::downgrade(&child);
        }
        child.borrow_mut().sibling = first_child; // Move, not clone

        parent.borrow_mut().child = Some(child);
    }

    /// Cut a node from its current position and meld as root
    fn cut_and_meld(&mut self, node: NodeRef<T, P>) {
        // Extract state first to avoid overlapping borrows
        let is_extra = node.borrow().extra;
        let partner_back = node.borrow().partner_back.upgrade();
        let partner = node.borrow_mut().partner.take();
        let parent = node.borrow().parent.upgrade();

        if is_extra {
            // Extra partner - just unlink from primary
            if let Some(primary) = partner_back {
                primary.borrow_mut().partner = None;
            }
            node.borrow_mut().partner_back = Weak::new();
            node.borrow_mut().extra = false;
        } else if let Some(partner_node) = partner {
            // Primary partner - partner takes our place in child list
            partner_node.borrow_mut().partner_back = Weak::new();
            partner_node.borrow_mut().extra = false;

            if let Some(ref parent_node) = parent {
                self.replace_in_child_list(parent_node, &node, partner_node);
            }
        } else if let Some(ref parent_node) = parent {
            // Regular child - remove from parent's child list
            self.remove_from_child_list(parent_node, &node);
        }

        node.borrow_mut().clear_links();
        self.meld_node(node);
    }

    /// Remove a node from parent's child list
    fn remove_from_child_list(&self, parent: &NodeRef<T, P>, node: &NodeRef<T, P>) {
        let next = node.borrow_mut().sibling.take();

        let is_first = parent
            .borrow()
            .child
            .as_ref()
            .is_some_and(|c| Rc::ptr_eq(c, node));

        if is_first {
            // Update next's prev before moving next
            if let Some(ref n) = next {
                n.borrow_mut().prev = Rc::downgrade(parent);
            }
            parent.borrow_mut().child = next; // Move, not clone
        } else if let Some(prev_node) = node.borrow().prev.upgrade() {
            // Update next's prev before moving next
            if let Some(ref n) = next {
                n.borrow_mut().prev = Rc::downgrade(&prev_node);
            }
            prev_node.borrow_mut().sibling = next; // Move, not clone
        }
    }

    /// Replace old_node with new_node in parent's child list (moves new_node)
    fn replace_in_child_list(
        &self,
        parent: &NodeRef<T, P>,
        old_node: &NodeRef<T, P>,
        new_node: NodeRef<T, P>,
    ) {
        let next = old_node.borrow_mut().sibling.take();
        new_node.borrow_mut().parent = Rc::downgrade(parent);

        // Update next's prev to point to new_node before we lose access
        if let Some(ref n) = next {
            n.borrow_mut().prev = Rc::downgrade(&new_node);
        }
        new_node.borrow_mut().sibling = next; // Move, not clone

        let is_first = parent
            .borrow()
            .child
            .as_ref()
            .is_some_and(|c| Rc::ptr_eq(c, old_node));

        if is_first {
            new_node.borrow_mut().prev = Rc::downgrade(parent);
            parent.borrow_mut().child = Some(new_node); // Move, not clone
        } else if let Some(prev_node) = old_node.borrow().prev.upgrade() {
            new_node.borrow_mut().prev = Rc::downgrade(&prev_node);
            prev_node.borrow_mut().sibling = Some(new_node); // Move, not clone
        }
    }
}
