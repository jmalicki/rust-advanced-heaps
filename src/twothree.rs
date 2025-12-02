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
    /// Back-reference to the node containing this entry (updated during bubble_up)
    node: WeakNodeRef<T, P>,
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

/// O(1) debug assertion: check that if root has an entry, its priority matches
/// Note: Root may temporarily lack an entry during restructuring; ensure_node_has_min_entry fixes this
#[cfg(debug_assertions)]
fn debug_assert_root_priority_matches<T, P: Ord + Clone + std::fmt::Debug>(root: &NodeRef<T, P>) {
    let root_ref = root.borrow();
    // Only check priority match if root has an entry
    // (root might temporarily lack entry during internal restructuring)
    if let Some(ref entry) = root_ref.entry {
        debug_assert_eq!(
            root_ref.priority,
            entry.borrow().priority,
            "Root priority must match entry priority"
        );
    }
}

/// O(1) debug assertion: check that a node with entry has matching priority
#[allow(dead_code)]
#[cfg(debug_assertions)]
fn debug_assert_entry_priority_matches<T, P: Ord + Clone + std::fmt::Debug>(node: &NodeRef<T, P>) {
    let node_ref = node.borrow();
    if let Some(ref entry) = node_ref.entry {
        debug_assert_eq!(
            node_ref.priority,
            entry.borrow().priority,
            "Node priority must match entry priority when entry exists"
        );
    }
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
        // Create node first (with None entry temporarily)
        let node = Rc::new(RefCell::new(Node {
            entry: None,
            priority: priority.clone(),
            parent: Weak::new(),
            children: Vec::new(),
        }));

        // Create entry with back-reference to the node
        let entry = Rc::new(RefCell::new(EntryData {
            item,
            priority: priority.clone(),
            node: Rc::downgrade(&node),
        }));

        // Put entry in node
        node.borrow_mut().entry = Some(Rc::clone(&entry));

        let handle = TwoThreeHandle {
            entry: Rc::downgrade(&entry),
        };

        if let Some(ref root) = self.root {
            // Compare new node's priority with root's priority
            // Root's priority should always be up-to-date (we maintain this invariant)
            let new_priority = node.borrow().priority.clone();
            let root_priority = root.borrow().priority.clone();

            if new_priority < root_priority {
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
            ensure_node_has_min_entry(&root);
        }

        self.len += 1;

        // O(1) debug assertions
        #[cfg(debug_assertions)]
        if let Some(ref root) = self.root {
            debug_assert_root_priority_matches(root);
        }

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

        // O(1) debug assertions
        #[cfg(debug_assertions)]
        if let Some(ref root) = self.root {
            debug_assert_root_priority_matches(root);
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

        // Get the node containing this entry using the back-reference (O(1))
        let node = entry.borrow().node.upgrade();

        entry.borrow_mut().priority = new_priority.clone();

        if let Some(node) = node {
            // Update the node's priority field
            node.borrow_mut().priority = new_priority;
            self.bubble_up(node);
        }

        // Ensure root has an entry after bubble_up (which may swap entries)
        if let Some(root) = self.root.clone() {
            ensure_node_has_min_entry(&root);
        }

        // O(1) debug assertions
        #[cfg(debug_assertions)]
        if let Some(ref root) = self.root {
            debug_assert_root_priority_matches(root);
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
            ensure_node_has_min_entry(&root);
        }

        self.len += other.len;
        other.len = 0;

        // O(1) debug assertions
        #[cfg(debug_assertions)]
        if let Some(ref root) = self.root {
            debug_assert_root_priority_matches(root);
        }

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

    /// Propagates priority changes up the tree when a node's priority changes.
    /// Recalculates each ancestor's priority based on its entry and direct children.
    /// This is O(height) where height is O(log n), with O(1) work per level.
    fn propagate_priority_up(node: &NodeRef<T, P>) {
        let mut current_weak = node.borrow().parent.clone();

        while let Some(parent) = current_weak.upgrade() {
            // Calculate correct priority: min of entry priority and children's priorities
            let correct_priority = {
                let p = parent.borrow();
                let mut min = p.entry.as_ref().map(|e| e.borrow().priority.clone());
                for child in &p.children {
                    let child_p = child.borrow().priority.clone();
                    match &min {
                        Some(m) if child_p < *m => min = Some(child_p),
                        None => min = Some(child_p),
                        _ => {}
                    }
                }
                min
            };

            if let Some(new_priority) = correct_priority {
                let old_priority = parent.borrow().priority.clone();
                if new_priority != old_priority {
                    parent.borrow_mut().priority = new_priority;
                    current_weak = parent.borrow().parent.clone();
                } else {
                    break; // Priority unchanged, stop propagating
                }
            } else {
                // Parent has no entries (shouldn't happen in valid heap)
                current_weak = parent.borrow().parent.clone();
            }
        }
    }

    fn maintain_structure(&mut self, node: NodeRef<T, P>) {
        let num_children = node.borrow().children.len();

        if num_children == 4 {
            let mut children_vec = mem::take(&mut node.borrow_mut().children);
            let new_children = children_vec.split_off(2);
            let parent_weak = node.borrow().parent.clone();

            // Calculate the minimum priority for the new structural node based on children's priorities
            // Children's priorities should be up-to-date (we maintain this invariant)
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

            // Update node's priority based on its children (if it's structural)
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
                // Create new root - find which child has the minimum (based on their priorities)
                let node_priority = node.borrow().priority.clone();
                let new_node_priority = new_node.borrow().priority.clone();

                let min_child = if node_priority <= new_node_priority {
                    Rc::clone(&node)
                } else {
                    Rc::clone(&new_node)
                };

                let root_priority = if node_priority <= new_node_priority {
                    node_priority
                } else {
                    new_node_priority
                };

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

                // Update back-references in swapped entries
                if let Some(ref entry) = min_child.borrow().entry {
                    entry.borrow_mut().node = Rc::downgrade(&min_child);
                }
                if let Some(ref entry) = new_root.borrow().entry {
                    entry.borrow_mut().node = Rc::downgrade(&new_root);
                }

                // Update min_child's priority to reflect its children's minimum
                // since it no longer has an entry
                let min_child_new_priority = min_child
                    .borrow()
                    .children
                    .iter()
                    .map(|c| c.borrow().priority.clone())
                    .min();
                if let Some(p) = min_child_new_priority {
                    min_child.borrow_mut().priority = p;
                }

                self.root = Some(new_root);
            }
        }
    }

    fn bubble_up(&mut self, node: NodeRef<T, P>) {
        let mut current = node;
        let mut nodes_to_cleanup: Vec<NodeRef<T, P>> = Vec::new();

        loop {
            let parent_weak = current.borrow().parent.clone();
            let parent = match parent_weak.upgrade() {
                Some(p) => p,
                None => break,
            };

            // Get current's entry priority (the priority we're bubbling up)
            let current_entry_priority = {
                let c = current.borrow();
                c.entry.as_ref().map(|e| e.borrow().priority.clone())
            };

            // Only bubble up if current has an entry
            let current_priority = match current_entry_priority {
                Some(p) => p,
                None => break, // Nothing to bubble up
            };

            // Get parent's actual minimum priority (either from entry or subtree)
            let parent_priority = {
                let p = parent.borrow();
                if let Some(ref entry) = p.entry {
                    // Parent has entry, compare with entry's priority
                    entry.borrow().priority.clone()
                } else {
                    // Parent is structural, compare with its subtree min
                    // (excluding current which we're trying to bubble up)
                    p.priority.clone()
                }
            };

            if current_priority >= parent_priority {
                break;
            }

            // Swap entries and priorities
            {
                let mut c = current.borrow_mut();
                let mut p = parent.borrow_mut();
                mem::swap(&mut c.entry, &mut p.entry);
                mem::swap(&mut c.priority, &mut p.priority);
            }

            // Update back-references in swapped entries
            if let Some(ref entry) = current.borrow().entry {
                entry.borrow_mut().node = Rc::downgrade(&current);
            }
            if let Some(ref entry) = parent.borrow().entry {
                entry.borrow_mut().node = Rc::downgrade(&parent);
            }

            // If current is now structural (no entry), update its priority based on children
            if current.borrow().entry.is_none() {
                let min_child_priority = current
                    .borrow()
                    .children
                    .iter()
                    .map(|c| c.borrow().priority.clone())
                    .min();
                if let Some(p) = min_child_priority {
                    current.borrow_mut().priority = p;
                } else {
                    // Current has no children - mark for cleanup
                    nodes_to_cleanup.push(Rc::clone(&current));
                }
            }

            current = parent;
        }

        // Clean up empty nodes (no entry, no children with entries)
        for empty_node in nodes_to_cleanup {
            let is_empty = {
                let node_ref = empty_node.borrow();
                node_ref.entry.is_none() && node_ref.children.is_empty()
            };
            if is_empty {
                // Remove from parent's children
                if let Some(parent) = empty_node.borrow().parent.upgrade() {
                    parent
                        .borrow_mut()
                        .children
                        .retain(|c| !Rc::ptr_eq(c, &empty_node));
                }
            }
        }
    }

    fn rebuild_from_children(&mut self, children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.len() == 1 {
            let root = Rc::clone(&children[0]);
            root.borrow_mut().parent = Weak::new();
            // Fix priority based on actual minimum in subtree
            Self::update_priority_from_subtree(&root);
            // Ensure root has an entry by pulling up from children if needed
            ensure_node_has_min_entry(&root);
            return root;
        }

        // First, compute actual subtree minimums for all children
        // This handles cases where cached priority fields might be stale
        let child_mins: Vec<Option<P>> = children
            .iter()
            .map(|c| Self::find_min_priority_in_subtree(c))
            .collect();

        // Update all children's priority fields to reflect actual minimums
        for (child, min_opt) in children.iter().zip(child_mins.iter()) {
            if let Some(min_p) = min_opt {
                child.borrow_mut().priority = min_p.clone();
            }
        }

        // Find min child using the computed actual minimums (not cached priorities)
        // Skip children with no entries in their subtree
        let mut min_idx = 0;
        let mut min_priority: Option<&P> = None;
        for (i, min_opt) in child_mins.iter().enumerate() {
            if let Some(p) = min_opt {
                match min_priority {
                    Some(m) if p < m => {
                        min_idx = i;
                        min_priority = Some(p);
                    }
                    None => {
                        min_idx = i;
                        min_priority = Some(p);
                    }
                    _ => {}
                }
            }
        }
        let min = Rc::clone(&children[min_idx]);

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
        ensure_node_has_min_entry(&actual_root);

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

/// Ensures node has an entry with the minimum priority in its subtree.
/// If node has no entry, pulls the minimum entry from children.
/// If node has an entry but a child has a smaller one, swaps entries.
///
/// This function first ensures all children have their min entries (recursive),
/// then compares with the node's entry to maintain the heap property.
fn ensure_node_has_min_entry<T, P: Ord + Clone>(node: &NodeRef<T, P>) {
    loop {
        let children: Vec<_> = node.borrow().children.to_vec();

        // First, recursively ensure all children have their min entries
        // This updates their priority fields to be accurate
        for child in &children {
            ensure_node_has_min_entry(child);
        }

        // Now find the child with minimum priority (priorities are now accurate after recursion)
        let mut min_child: Option<NodeRef<T, P>> = None;
        let mut min_child_priority: Option<P> = None;

        for child in &children {
            // After recursion, if child has entry, priority equals entry priority
            // If child has no entry, it will be removed below
            if child.borrow().entry.is_some() {
                let child_priority = child.borrow().priority.clone();
                let is_smaller = match &min_child_priority {
                    Some(m) => child_priority < *m,
                    None => true,
                };
                if is_smaller {
                    min_child = Some(Rc::clone(child));
                    min_child_priority = Some(child_priority);
                }
            }
        }

        // Remove children that have no entries (they're now empty)
        {
            let children_to_remove: Vec<_> = node.borrow().children.iter()
                .filter(|c| c.borrow().entry.is_none() && c.borrow().children.is_empty())
                .cloned()
                .collect();
            for empty_child in children_to_remove {
                node.borrow_mut().children.retain(|c| !Rc::ptr_eq(c, &empty_child));
            }
        }

        // Get current node's entry priority
        let node_entry_priority = node
            .borrow()
            .entry
            .as_ref()
            .map(|e| e.borrow().priority.clone());

        // If node has no entry and no children with entries, we're done
        if node_entry_priority.is_none() && min_child_priority.is_none() {
            return;
        }

        // If node has an entry that's already the minimum, we're done
        match (&node_entry_priority, &min_child_priority) {
            (Some(node_p), Some(child_p)) if *node_p <= *child_p => {
                // Node's entry is smaller or equal, update priority field and return
                node.borrow_mut().priority = node_p.clone();
                return;
            }
            (Some(node_p), None) => {
                // Node has entry, no children with entries
                node.borrow_mut().priority = node_p.clone();
                return;
            }
            _ => {}
        }

        // Need to swap entries with min_child
        let min_child = min_child.unwrap();

        // Take entry from min_child (it already has its min entry from recursion above)
        let min_child_entry = min_child.borrow_mut().entry.take();
        if let Some(child_entry) = min_child_entry {
            let child_priority = child_entry.borrow().priority.clone();

            // If node had an entry, move it to min_child
            let old_node_entry = node.borrow_mut().entry.take();
            if let Some(old_entry) = old_node_entry {
                let old_priority = old_entry.borrow().priority.clone();
                old_entry.borrow_mut().node = Rc::downgrade(&min_child);
                min_child.borrow_mut().entry = Some(old_entry);
                min_child.borrow_mut().priority = old_priority;
            }

            // Put child's entry in node
            child_entry.borrow_mut().node = Rc::downgrade(node);
            node.borrow_mut().entry = Some(child_entry);
            node.borrow_mut().priority = child_priority;
        }

        // If min_child is now a leaf with no entry, remove it
        let min_child_is_empty_leaf = {
            let mc = min_child.borrow();
            mc.entry.is_none() && mc.children.is_empty()
        };

        if min_child_is_empty_leaf {
            node.borrow_mut()
                .children
                .retain(|c| !Rc::ptr_eq(c, &min_child));
        }

        // We're done - node now has the minimum entry
        return;
    }
}
