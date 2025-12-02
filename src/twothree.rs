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
//! - Items are stored behind Rc so handles remain valid after bubble-up swaps

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::mem;
use std::rc::{Rc, Weak};

/// Type alias for node reference (strong reference for ownership)
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;

/// Type alias for weak node reference (for parent links)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;

/// Entry containing item and priority, stored behind Rc for stable handles
type Entry<T, P> = Rc<RefCell<EntryData<T, P>>>;
type WeakEntry<T, P> = Weak<RefCell<EntryData<T, P>>>;

struct EntryData<T, P> {
    item: T,
    priority: P,
}

/// Handle to an element in a 2-3 heap
///
/// Uses a Weak reference to the entry data, ensuring the handle remains valid
/// even when nodes swap their entries during bubble-up.
pub struct TwoThreeHandle<T, P> {
    entry: WeakEntry<T, P>,
}

// Manual Clone implementation to avoid requiring T: Clone, P: Clone
impl<T, P> Clone for TwoThreeHandle<T, P> {
    fn clone(&self) -> Self {
        TwoThreeHandle {
            entry: self.entry.clone(),
        }
    }
}

impl<T, P> PartialEq for TwoThreeHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.entry, &other.entry)
    }
}

impl<T, P> Eq for TwoThreeHandle<T, P> {}

impl<T, P> std::fmt::Debug for TwoThreeHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwoThreeHandle")
            .field("entry", &Weak::as_ptr(&self.entry))
            .finish()
    }
}

impl<T, P> Handle for TwoThreeHandle<T, P> {}

/// Node in the 2-3 tree
struct Node<T, P> {
    /// The item/priority this node represents. None for internal structural nodes.
    entry: Option<Entry<T, P>>,
    /// Priority for ordering (always present, even if entry is None for structural nodes)
    priority: P,
    parent: WeakNodeRef<T, P>,
    children: Vec<NodeRef<T, P>>,
}

/// 2-3 Heap
pub struct TwoThreeHeap<T, P: Ord> {
    root: Option<NodeRef<T, P>>,
    len: usize,
}

impl<T: Clone, P: Ord + Clone + std::fmt::Debug> Heap<T, P> for TwoThreeHeap<T, P> {
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
        let entry = Rc::new(RefCell::new(EntryData {
            item,
            priority: priority.clone(),
        }));
        let handle = TwoThreeHandle {
            entry: Rc::downgrade(&entry),
        };

        let node = Rc::new(RefCell::new(Node {
            entry: Some(entry),
            priority,
            parent: Weak::new(),
            children: Vec::new(),
        }));

        if let Some(ref root) = self.root {
            let new_is_smaller = {
                let node_ref = node.borrow();
                let root_ref = root.borrow();
                node_ref.priority < root_ref.priority
            };

            if new_is_smaller {
                let old_root = self.root.take().unwrap();
                old_root.borrow_mut().parent = Rc::downgrade(&node);
                node.borrow_mut().children.push(old_root);
                self.root = Some(Rc::clone(&node));
            } else {
                self.insert_as_child(Rc::clone(root), Rc::clone(&node));
            }
        } else {
            self.root = Some(Rc::clone(&node));
        }

        self.maintain_structure(Rc::clone(&node));

        // Ensure root has an entry after potential restructuring
        if let Some(root) = self.root.clone() {
            ensure_node_has_entry(&root);
        }

        self.len += 1;

        // Verify invariants after insert (expensive O(n) checks, only with feature flag)
        #[cfg(feature = "expensive_verify")]
        {
            let count = self.count_all_nodes();
            assert_eq!(
                count, self.len,
                "Length mismatch after insert: counted {} entries but len is {}",
                count, self.len
            );
            self.verify_min_at_root();
        }

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        let root = self.root.as_ref()?;
        let node = root.borrow();
        // Root should always have an entry (we ensure this)
        node.entry.as_ref().map(|entry| {
            let entry_ref = entry.borrow();
            unsafe {
                let priority: &P = &*(&entry_ref.priority as *const P);
                let item: &T = &*(&entry_ref.item as *const T);
                (priority, item)
            }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    fn delete_min(&mut self) -> Option<(P, T)> {
        let root = self.root.take()?;

        // Root should always have an entry (we ensure this in maintain_structure)
        let entry = root.borrow_mut().entry.take()?;

        // Get children from root
        let children: Vec<_> = mem::take(&mut root.borrow_mut().children);

        // Extract item and priority from entry
        let (priority, item) = match Rc::try_unwrap(entry) {
            Ok(cell) => {
                let data = cell.into_inner();
                (data.priority, data.item)
            }
            Err(rc) => {
                // Entry still referenced by handle - clone the data
                let data = rc.borrow();
                (data.priority.clone(), data.item.clone())
            }
        };

        self.len -= 1;

        if children.is_empty() {
            self.root = None;
        } else {
            for child in &children {
                child.borrow_mut().parent = Weak::new();
            }
            self.root = Some(self.rebuild_from_children(children));
        }

        // Verify invariants after delete_min (expensive O(n) checks, only with feature flag)
        #[cfg(feature = "expensive_verify")]
        {
            let count = self.count_all_nodes();
            assert_eq!(
                count, self.len,
                "Length mismatch after delete_min: counted {} entries but len is {}",
                count, self.len
            );

            // Verify root has minimum priority
            self.verify_min_at_root();
        }

        Some((priority, item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let entry = handle
            .entry
            .upgrade()
            .ok_or(HeapError::PriorityNotDecreased)?;

        if new_priority >= entry.borrow().priority {
            return Err(HeapError::PriorityNotDecreased);
        }

        entry.borrow_mut().priority = new_priority.clone();

        if let Some(root) = self.root.clone() {
            if let Some(node) = find_node_with_entry(Rc::clone(&root), &entry) {
                // Also update the node's priority field
                node.borrow_mut().priority = new_priority;
                self.bubble_up(node);
            }
        }

        // Ensure root has an entry after bubble_up (which may swap entries)
        if let Some(root) = self.root.clone() {
            ensure_node_has_entry(&root);
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

        let self_root = self.root.take().unwrap();
        let other_root = other.root.take().unwrap();

        let other_is_smaller = {
            let s = self_root.borrow();
            let o = other_root.borrow();
            o.priority < s.priority
        };

        if other_is_smaller {
            // Set other_root as temporary root before insert_as_child
            // because maintain_structure may need to find the root
            self.root = Some(Rc::clone(&other_root));
            self.insert_as_child(Rc::clone(&other_root), self_root);
            // Note: maintain_structure might have created a new root,
            // so we don't overwrite self.root here
        } else {
            // Set self_root as temporary root before insert_as_child
            self.root = Some(Rc::clone(&self_root));
            self.insert_as_child(Rc::clone(&self_root), other_root);
            // Note: maintain_structure might have created a new root,
            // so we don't overwrite self.root here
        }

        // Ensure root has an entry after potential restructuring
        if let Some(root) = self.root.clone() {
            ensure_node_has_entry(&root);
        }

        self.len += other.len;
        other.len = 0;

        // Verify invariants after merge (expensive O(n) checks, only with feature flag)
        #[cfg(feature = "expensive_verify")]
        {
            let count = self.count_all_nodes();
            assert_eq!(
                count, self.len,
                "Length mismatch after merge: counted {} entries but len is {}",
                count, self.len
            );
            self.verify_min_at_root();
        }
    }
}

impl<T: Clone, P: Ord + Clone + std::fmt::Debug> Default for TwoThreeHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively searches for a node containing the given entry
fn find_node_with_entry<T, P>(node: NodeRef<T, P>, entry: &Entry<T, P>) -> Option<NodeRef<T, P>> {
    let is_match = {
        let node_ref = node.borrow();
        if let Some(ref node_entry) = node_ref.entry {
            Rc::ptr_eq(node_entry, entry)
        } else {
            false
        }
    };

    if is_match {
        return Some(node);
    }

    let children: Vec<_> = node.borrow().children.to_vec();
    for child in children {
        if let Some(found) = find_node_with_entry(child, entry) {
            return Some(found);
        }
    }

    None
}

impl<T, P: Ord + Clone> TwoThreeHeap<T, P> {
    /// Counts all nodes in the heap (debug only)
    #[cfg(feature = "expensive_verify")]
    fn count_all_nodes(&self) -> usize {
        if let Some(ref root) = self.root {
            Self::count_subtree(root)
        } else {
            0
        }
    }

    #[cfg(feature = "expensive_verify")]
    fn count_subtree(node: &NodeRef<T, P>) -> usize {
        let node_ref = node.borrow();

        // Count this node only if it has an entry (structural nodes don't have items)
        let count = if node_ref.entry.is_some() { 1 } else { 0 };

        // Count children
        let children_count: usize = node_ref
            .children
            .iter()
            .map(|c| Self::count_subtree(c))
            .sum();

        count + children_count
    }

    /// Finds the actual minimum priority in the entire heap (debug only)
    #[cfg(feature = "expensive_verify")]
    fn find_actual_min(&self) -> Option<P> {
        if let Some(ref root) = self.root {
            Self::find_min_in_subtree(root)
        } else {
            None
        }
    }

    #[cfg(feature = "expensive_verify")]
    fn find_min_in_subtree(node: &NodeRef<T, P>) -> Option<P> {
        let node_ref = node.borrow();

        // Get this node's entry priority if it has one
        let mut min = node_ref.entry.as_ref().map(|e| e.borrow().priority.clone());

        // Compare with all children's minimums
        for child in &node_ref.children {
            if let Some(child_min) = Self::find_min_in_subtree(child) {
                match &min {
                    Some(current_min) if child_min < *current_min => {
                        min = Some(child_min);
                    }
                    None => {
                        min = Some(child_min);
                    }
                    _ => {}
                }
            }
        }

        min
    }

    /// Verifies that the root has the minimum priority in the heap
    #[cfg(feature = "expensive_verify")]
    fn verify_min_at_root(&self)
    where
        P: std::fmt::Debug,
    {
        if let Some(ref root) = self.root {
            let root_priority = {
                let root_ref = root.borrow();
                root_ref.entry.as_ref().map(|e| e.borrow().priority.clone())
            };
            if let Some(root_p) = root_priority {
                let actual_min = self.find_actual_min();
                if let Some(min_p) = actual_min {
                    assert!(
                        root_p <= min_p,
                        "Root priority {:?} is greater than actual min {:?}",
                        root_p,
                        min_p
                    );
                }
            }
        }
    }

    fn insert_as_child(&mut self, parent: NodeRef<T, P>, child: NodeRef<T, P>) {
        child.borrow_mut().parent = Rc::downgrade(&parent);
        let child_clone = Rc::clone(&child);
        parent.borrow_mut().children.push(child);

        // Propagate priority up if child has smaller priority
        Self::propagate_priority_up(&child_clone);

        self.maintain_structure(parent);
    }

    /// Propagates priority changes up the tree when a node's subtree minimum changes
    fn propagate_priority_up(node: &NodeRef<T, P>) {
        let node_priority = node.borrow().priority.clone();
        let mut current_weak = node.borrow().parent.clone();

        while let Some(parent) = current_weak.upgrade() {
            let should_update = {
                let parent_ref = parent.borrow();
                node_priority < parent_ref.priority
            };

            if should_update {
                parent.borrow_mut().priority = node_priority.clone();
                current_weak = parent.borrow().parent.clone();
            } else {
                break;
            }
        }
    }

    fn maintain_structure(&mut self, node: NodeRef<T, P>) {
        let num_children = node.borrow().children.len();

        if num_children == 4 {
            let mut children_vec = mem::take(&mut node.borrow_mut().children);
            let new_children = children_vec.split_off(2);
            let parent_weak = node.borrow().parent.clone();

            // Calculate the minimum priority for the new structural node's subtree
            let new_node_priority = new_children
                .iter()
                .map(|c| c.borrow().priority.clone())
                .min()
                .unwrap();

            // Create structural node with min priority of its children
            let new_node = Rc::new(RefCell::new(Node {
                entry: None,
                priority: new_node_priority,
                parent: parent_weak.clone(),
                children: new_children,
            }));

            for child in &new_node.borrow().children {
                child.borrow_mut().parent = Rc::downgrade(&new_node);
            }

            node.borrow_mut().children = children_vec;

            // Update node's priority to be min of remaining children (if it's structural)
            if node.borrow().entry.is_none() {
                let node_min_priority = node
                    .borrow()
                    .children
                    .iter()
                    .map(|c| c.borrow().priority.clone())
                    .min();
                if let Some(p) = node_min_priority {
                    node.borrow_mut().priority = p;
                }
            }

            if let Some(parent) = parent_weak.upgrade() {
                self.insert_as_child(parent, new_node);
            } else {
                // Create new root - find which child has the minimum
                let (min_child, _other_child) =
                    if node.borrow().priority <= new_node.borrow().priority {
                        (Rc::clone(&node), Rc::clone(&new_node))
                    } else {
                        (Rc::clone(&new_node), Rc::clone(&node))
                    };

                let root_priority = min_child.borrow().priority.clone();

                let new_root = Rc::new(RefCell::new(Node {
                    entry: None,
                    priority: root_priority,
                    parent: Weak::new(),
                    children: vec![Rc::clone(&node), Rc::clone(&new_node)],
                }));

                node.borrow_mut().parent = Rc::downgrade(&new_root);
                new_node.borrow_mut().parent = Rc::downgrade(&new_root);

                // Pull up entry from min_child to new_root
                {
                    let mut mc = min_child.borrow_mut();
                    let mut nr = new_root.borrow_mut();
                    mem::swap(&mut mc.entry, &mut nr.entry);
                }

                // Update min_child's priority to reflect its subtree minimum
                // since it no longer has an entry
                {
                    let min_child_new_priority = min_child
                        .borrow()
                        .children
                        .iter()
                        .map(|c| c.borrow().priority.clone())
                        .min();
                    if let Some(p) = min_child_new_priority {
                        min_child.borrow_mut().priority = p;
                    }
                }

                self.root = Some(new_root);
            }
        }
    }

    fn bubble_up(&mut self, node: NodeRef<T, P>) {
        let mut current = node;

        loop {
            let parent_weak = current.borrow().parent.clone();
            let parent = match parent_weak.upgrade() {
                Some(p) => p,
                None => break,
            };

            let should_swap = {
                let c = current.borrow();
                let p = parent.borrow();
                c.priority < p.priority
            };

            if !should_swap {
                break;
            }

            // Swap entries and priorities
            {
                let mut c = current.borrow_mut();
                let mut p = parent.borrow_mut();
                mem::swap(&mut c.entry, &mut p.entry);
                mem::swap(&mut c.priority, &mut p.priority);
            }

            // If current is now structural (no entry), update its priority to actual min of subtree
            if current.borrow().entry.is_none() {
                if let Some(p) = Self::find_min_priority_in_subtree(&current) {
                    current.borrow_mut().priority = p;
                }
            }

            current = parent;
        }
    }

    fn rebuild_from_children(&mut self, children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.len() == 1 {
            let root = Rc::clone(&children[0]);
            root.borrow_mut().parent = Weak::new();
            // Fix priority based on actual minimum in subtree
            Self::update_priority_from_subtree(&root);
            // Ensure root has an entry by pulling up from children if needed
            ensure_node_has_entry(&root);
            return root;
        }

        // First, update all children's priority fields to reflect actual minimums
        for child in &children {
            Self::update_priority_from_subtree(child);
        }

        // Now find min child using correct priorities
        let mut min = Rc::clone(&children[0]);
        for child in children.iter().skip(1) {
            let is_smaller = {
                let c = child.borrow();
                let m = min.borrow();
                c.priority < m.priority
            };
            if is_smaller {
                min = Rc::clone(child);
            }
        }

        // Set min as the temporary root before inserting other children
        // This allows maintain_structure to correctly update self.root if splits occur
        self.root = Some(Rc::clone(&min));
        min.borrow_mut().parent = Weak::new();

        for child in &children {
            if !Rc::ptr_eq(child, &min) {
                child.borrow_mut().parent = Weak::new();
                self.insert_as_child(Rc::clone(&min), Rc::clone(child));
            }
        }

        // After all insertions, self.root may have changed due to splits
        // Ensure the actual root has an entry
        let actual_root = self.root.as_ref().unwrap().clone();
        ensure_node_has_entry(&actual_root);

        actual_root
    }

    /// Update a node's priority field to reflect the actual minimum in its subtree
    fn update_priority_from_subtree(node: &NodeRef<T, P>) {
        let actual_min = Self::find_min_priority_in_subtree(node);
        if let Some(min_p) = actual_min {
            node.borrow_mut().priority = min_p;
        }
    }

    /// Find the minimum entry priority in a subtree (non-debug version)
    fn find_min_priority_in_subtree(node: &NodeRef<T, P>) -> Option<P> {
        let node_ref = node.borrow();

        // Get this node's entry priority if it has one
        let mut min = node_ref.entry.as_ref().map(|e| e.borrow().priority.clone());

        // Compare with all children's minimums
        for child in &node_ref.children {
            if let Some(child_min) = Self::find_min_priority_in_subtree(child) {
                match &min {
                    Some(current_min) if child_min < *current_min => {
                        min = Some(child_min);
                    }
                    None => {
                        min = Some(child_min);
                    }
                    _ => {}
                }
            }
        }

        min
    }
}

/// Find the minimum entry priority in a subtree (standalone function)
fn find_min_priority_in_subtree_standalone<T, P: Ord + Clone>(node: &NodeRef<T, P>) -> Option<P> {
    let node_ref = node.borrow();

    // Get this node's entry priority if it has one
    let mut min = node_ref.entry.as_ref().map(|e| e.borrow().priority.clone());

    // Compare with all children's minimums
    for child in &node_ref.children {
        if let Some(child_min) = find_min_priority_in_subtree_standalone(child) {
            match &min {
                Some(current_min) if child_min < *current_min => {
                    min = Some(child_min);
                }
                None => {
                    min = Some(child_min);
                }
                _ => {}
            }
        }
    }

    min
}

/// Ensures a node has an entry by pulling up from its children if needed
fn ensure_node_has_entry<T, P: Ord + Clone>(node: &NodeRef<T, P>) {
    if node.borrow().entry.is_some() {
        return; // Already has an entry
    }

    // Try each child until we find one with an entry (or can get one)
    loop {
        let children: Vec<_> = node.borrow().children.to_vec();
        if children.is_empty() {
            return; // No children to pull from
        }

        // Find child with minimum ACTUAL priority (scan subtrees for true minimum)
        let mut min_child = Rc::clone(&children[0]);
        let mut min_priority = find_min_priority_in_subtree_standalone(&min_child);

        for child in children.iter().skip(1) {
            let child_min = find_min_priority_in_subtree_standalone(child);
            let is_smaller = match (&child_min, &min_priority) {
                (Some(c), Some(m)) => c < m,
                (Some(_), None) => true,
                _ => false,
            };
            if is_smaller {
                min_child = Rc::clone(child);
                min_priority = child_min;
            }
        }

        // Recursively ensure min_child has an entry
        ensure_node_has_entry(&min_child);

        // Check if min_child now has an entry
        if min_child.borrow().entry.is_some() {
            // Take entry from min_child and put it in this node
            let min_child_entry = min_child.borrow_mut().entry.take();
            if let Some(entry) = min_child_entry {
                let priority = entry.borrow().priority.clone();
                node.borrow_mut().entry = Some(entry);
                node.borrow_mut().priority = priority;
            }

            // If min_child is now a leaf with no entry, remove it from our children
            let min_child_is_empty_leaf = {
                let mc = min_child.borrow();
                mc.entry.is_none() && mc.children.is_empty()
            };

            if min_child_is_empty_leaf {
                node.borrow_mut()
                    .children
                    .retain(|c| !Rc::ptr_eq(c, &min_child));
            } else {
                // Update min_child's priority to reflect its subtree minimum
                if let Some(p) = find_min_priority_in_subtree_standalone(&min_child) {
                    min_child.borrow_mut().priority = p;
                }
            }

            return; // Successfully got an entry
        } else {
            // min_child has no entry and couldn't get one - it's a dead branch
            // Remove it and try again with remaining children
            node.borrow_mut()
                .children
                .retain(|c| !Rc::ptr_eq(c, &min_child));

            // Update node's priority to reflect remaining children
            if let Some(p) = find_min_priority_in_subtree_standalone(node) {
                node.borrow_mut().priority = p;
            }
        }
    }
}
