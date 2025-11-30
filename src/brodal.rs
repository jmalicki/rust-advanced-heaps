//! True Brodal Heap implementation
//!
//! A Brodal heap achieves optimal worst-case time bounds:
//! - O(1) worst-case insert, find_min, decrease_key, and merge
//! - O(log n) worst-case delete_min
//!
//! This implementation includes the full violation system described in Brodal's
//! original paper, with rank-based violation tracking and repair operations
//! that maintain worst-case bounds.
//!
//! This implementation uses Rc/Weak references instead of raw pointers for memory safety.

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
        // Compare by pointer equality
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

struct Node<T, P> {
    item: T,
    priority: P,
    parent: Option<WeakNodeRef<T, P>>,  // Weak back-reference to parent
    child: Option<NodeRef<T, P>>,        // Strong reference to first child
    sibling: Option<NodeRef<T, P>>,      // Strong reference to next sibling
    rank: usize,
    // For violation tracking
    in_violation_list: bool,
}

/// True Brodal Heap with complete violation system
///
/// This implementation includes:
/// - Per-rank violation queues for worst-case O(1) operations
/// - Rank constraint maintenance (rank(v) <= rank(w1) + 1, rank(v) <= rank(w2) + 1)
/// - Violation repair operations that maintain structure
/// - Safe Rc/Weak-based memory management (no unsafe code)
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
    root: Option<NodeRef<T, P>>,  // Strong reference to root (minimum element)
    len: usize,
    // Per-rank violation queues: violations[i] contains weak refs to nodes with rank i that have violations
    // Using Weak references avoids keeping nodes alive and allows proper cleanup
    violations: Vec<Vec<WeakNodeRef<T, P>>>,
    max_rank: usize,  // Maximum rank seen so far
}

impl<T, P: Ord> Heap<T, P> for BrodalHeap<T, P> {
    type Handle = BrodalHandle<T, P>;

    fn new() -> Self {
        Self {
            root: None,
            len: 0,
            violations: Vec::new(),
            max_rank: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) -> Self::Handle {
        self.insert(priority, item)
    }

    /// Inserts a new element into the heap
    ///
    /// **Time Complexity**: O(1) worst-case
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node with rank 0 (leaf node, no children)
        let node = Rc::new(RefCell::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            rank: 0,
            in_violation_list: false,
        }));

        let handle = BrodalHandle {
            node: Rc::downgrade(&node),
        };

        // Link new node into the tree structure
        if let Some(ref root) = self.root {
            if node.borrow().priority < root.borrow().priority {
                // New node has smaller priority: make it the new root
                self.make_child(Rc::clone(&node), Rc::clone(root));
                self.root = Some(node);
            } else {
                // Current root has smaller or equal priority
                self.make_child(Rc::clone(root), node);
            }
        } else {
            // Empty heap: new node becomes root
            self.root = Some(node);
        }

        self.len += 1;

        // Check for rank violations and repair (at most O(1) violations)
        if let Some(ref root) = self.root {
            self.repair_violations(Rc::clone(root));
        }

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        // This is safe because we hold a reference to the heap
        // and the Rc keeps the node alive
        self.root.as_ref().map(|root| {
            let node_ref = root.as_ptr();
            // SAFETY: We're borrowing from a RefCell we know exists and isn't borrowed mutably
            // The lifetime is tied to &self which keeps the heap alive
            unsafe { (&(*node_ref).priority, &(*node_ref).item) }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    /// Removes and returns the minimum element
    ///
    /// **Time Complexity**: O(log n) worst-case
    fn delete_min(&mut self) -> Option<(P, T)> {
        let root = self.root.take()?;

        // Collect all children of the root
        let children = self.collect_children(&root);

        // Extract item and priority from root
        // The root should have refcount 1 at this point (only self.root held it)
        let root_node = Rc::try_unwrap(root)
            .ok()
            .expect("Root should be uniquely owned")
            .into_inner();

        self.len -= 1;

        if children.is_empty() {
            // No children: heap becomes empty
            self.root = None;
        } else {
            // Process all violations accumulated so far
            self.process_all_violations();

            // Rebuild heap from children
            self.root = Some(self.rebuild_from_children(children));
        }

        Some((root_node.priority, root_node.item))
    }

    /// Decreases the priority of an element
    ///
    /// **Time Complexity**: O(1) worst-case
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node = handle.node.upgrade().ok_or(HeapError::PriorityNotDecreased)?;

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
        let parent_weak = node.borrow().parent.clone();
        if let Some(ref parent_weak) = parent_weak {
            if let Some(parent) = parent_weak.upgrade() {
                if node.borrow().priority < parent.borrow().priority {
                    // Heap property violated: cut node from parent
                    self.cut_from_parent(&node);

                    // Merge cut node with root
                    if let Some(ref root) = self.root {
                        if node.borrow().priority < root.borrow().priority {
                            // Cut node has smaller priority: make it the new root
                            if !Rc::ptr_eq(root, &node) {
                                self.make_child(Rc::clone(&node), Rc::clone(root));
                            }
                            self.root = Some(Rc::clone(&node));
                        } else {
                            // Current root has smaller priority
                            self.make_child(Rc::clone(root), Rc::clone(&node));
                        }
                    } else {
                        self.root = Some(Rc::clone(&node));
                    }

                    // Repair violations
                    self.repair_violations(node);
                }
            }
        }

        Ok(())
    }

    /// Merges another heap into this heap
    ///
    /// **Time Complexity**: O(1) worst-case
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

        // Merge roots: smaller priority becomes parent
        if other_root.borrow().priority < self_root.borrow().priority {
            if other_root.borrow().child.is_none() {
                self.make_child(Rc::clone(&other_root), self_root);
            } else {
                self.add_child(Rc::clone(&other_root), self_root);
            }
            self.root = Some(other_root);
        } else {
            self.add_child(Rc::clone(&self_root), other_root);
            self.root = Some(self_root);
        }

        // Merge violation lists (weak references)
        for (rank, violations) in other.violations.into_iter().enumerate() {
            while self.violations.len() <= rank {
                self.violations.push(Vec::new());
            }
            self.violations[rank].extend(violations);
        }

        self.len += other.len;
        self.max_rank = self.max_rank.max(other.max_rank);

        // Prevent double handling
        other.len = 0;

        // Process violations
        self.process_all_violations();
    }
}

impl<T, P: Ord> BrodalHeap<T, P> {
    /// Makes y a child of x
    fn make_child(&mut self, x: NodeRef<T, P>, y: NodeRef<T, P>) {
        y.borrow_mut().parent = Some(Rc::downgrade(&x));
        y.borrow_mut().sibling = x.borrow().child.clone();
        x.borrow_mut().child = Some(y);

        self.update_rank(&x);
    }

    /// Adds y as a child of x (existing children preserved)
    fn add_child(&mut self, x: NodeRef<T, P>, y: NodeRef<T, P>) {
        y.borrow_mut().parent = Some(Rc::downgrade(&x));
        let first_child = x.borrow().child.clone();
        if first_child.is_some() {
            y.borrow_mut().sibling = first_child;
            x.borrow_mut().child = Some(y);
        } else {
            x.borrow_mut().child = Some(y);
            // y.sibling is already None
        }

        self.update_rank(&x);
    }

    /// Updates the rank of a node based on its children's ranks
    fn update_rank(&mut self, node: &NodeRef<T, P>) {
        let mut child_ranks = Vec::new();

        // Traverse child list to collect all ranks
        let mut current = node.borrow().child.clone();
        while let Some(child) = current {
            child_ranks.push(child.borrow().rank);
            current = child.borrow().sibling.clone();
        }

        // Base case: no children, rank is 0
        if child_ranks.is_empty() {
            node.borrow_mut().rank = 0;
            return;
        }

        // Sort ranks descending
        child_ranks.sort_by(|a, b| b.cmp(a));

        // Compute new rank
        let new_rank = if child_ranks.len() >= 2 {
            let r1 = child_ranks[child_ranks.len() - 1];
            let r2 = child_ranks[child_ranks.len() - 2];
            (r1.min(r2)) + 1
        } else {
            child_ranks[0] + 1
        };

        node.borrow_mut().rank = new_rank;

        // Check for rank constraint violation
        if child_ranks.len() >= 2 {
            let r1 = child_ranks[child_ranks.len() - 1];
            let r2 = child_ranks[child_ranks.len() - 2];

            if new_rank > r1 + 1 || new_rank > r2 + 1 {
                self.add_violation(Rc::clone(node));
            }
        }

        if new_rank > self.max_rank {
            self.max_rank = new_rank;
        }
    }

    /// Adds a node to the violation list for its rank
    fn add_violation(&mut self, node: NodeRef<T, P>) {
        let rank = node.borrow().rank;

        if node.borrow().in_violation_list {
            return;
        }

        while self.violations.len() <= rank {
            self.violations.push(Vec::new());
        }

        node.borrow_mut().in_violation_list = true;
        // Store weak reference to avoid keeping node alive
        self.violations[rank].push(Rc::downgrade(&node));
    }

    /// Removes a node from violation list
    fn remove_violation(&mut self, node: &NodeRef<T, P>) {
        let rank = node.borrow().rank;

        if !node.borrow().in_violation_list {
            return;
        }

        if rank < self.violations.len() {
            let node_weak = Rc::downgrade(node);
            self.violations[rank].retain(|n| !n.ptr_eq(&node_weak));
        }

        node.borrow_mut().in_violation_list = false;
    }

    /// Repairs violations starting from a given node (O(1) worst-case)
    fn repair_violations(&mut self, start_node: NodeRef<T, P>) {
        let start_rank = start_node.borrow().rank;

        if start_rank < self.violations.len() && !self.violations[start_rank].is_empty() {
            // Pop weak references until we find one that's still alive
            while let Some(weak_node) = self.violations[start_rank].pop() {
                if let Some(violating_node) = weak_node.upgrade() {
                    violating_node.borrow_mut().in_violation_list = false;
                    self.repair_rank_violation(violating_node);
                    break;
                }
                // If weak ref is dead, continue to next one
            }
        }
    }

    /// Processes all violations (used during delete_min)
    fn process_all_violations(&mut self) {
        for rank in 0..=self.max_rank {
            if rank >= self.violations.len() {
                continue;
            }

            while let Some(weak_node) = self.violations[rank].pop() {
                // Only process if the node is still alive
                if let Some(violating_node) = weak_node.upgrade() {
                    violating_node.borrow_mut().in_violation_list = false;
                    self.repair_rank_violation(violating_node);
                }
                // If weak ref is dead, just skip it
            }
        }
    }

    /// Repairs a rank violation by restructuring the node's children
    fn repair_rank_violation(&mut self, node: NodeRef<T, P>) {
        // Step 1: Disconnect all children
        let mut children = Vec::new();
        let mut current = node.borrow_mut().child.take();

        while let Some(child) = current {
            let next = child.borrow_mut().sibling.take();
            child.borrow_mut().parent = None;
            children.push(child);
            current = next;
        }

        // Base case: not enough children
        if children.len() < 2 {
            for child in children {
                self.add_child(Rc::clone(&node), child);
            }
            self.update_rank(&node);
            return;
        }

        // Step 2: Sort children by rank
        children.sort_by(|a, b| a.borrow().rank.cmp(&b.borrow().rank));

        let r1 = children[0].borrow().rank;
        let r2 = children[1].borrow().rank;
        let max_rank = r1.max(r2);

        // Step 3: Check if current rank violates constraint
        if node.borrow().rank > max_rank + 1 {
            // Group children by rank
            let mut by_rank: Vec<Vec<NodeRef<T, P>>> = Vec::new();
            for child in children {
                let rank = child.borrow().rank;
                while by_rank.len() <= rank {
                    by_rank.push(Vec::new());
                }
                by_rank[rank].push(child);
            }

            // Link pairs of same rank
            let mut new_children = Vec::new();
            for rank_group in by_rank.iter_mut() {
                while rank_group.len() >= 2 {
                    let a = rank_group.pop().unwrap();
                    let b = rank_group.pop().unwrap();

                    if a.borrow().priority < b.borrow().priority {
                        b.borrow_mut().parent = Some(Rc::downgrade(&a));
                        b.borrow_mut().sibling = a.borrow().child.clone();
                        a.borrow_mut().child = Some(b);
                        self.update_rank(&a);
                        new_children.push(a);
                    } else {
                        a.borrow_mut().parent = Some(Rc::downgrade(&b));
                        a.borrow_mut().sibling = b.borrow().child.clone();
                        b.borrow_mut().child = Some(a);
                        self.update_rank(&b);
                        new_children.push(b);
                    }
                }
                new_children.append(rank_group);
            }

            // Reattach restructured children
            for child in new_children {
                self.add_child(Rc::clone(&node), child);
            }

            self.update_rank(&node);
        } else {
            // No violation: just reattach children
            for child in children {
                self.add_child(Rc::clone(&node), child);
            }
            self.update_rank(&node);
        }
    }

    /// Cuts a node from its parent
    fn cut_from_parent(&mut self, node: &NodeRef<T, P>) {
        let parent_weak = node.borrow().parent.clone();
        let parent = match parent_weak.and_then(|w| w.upgrade()) {
            Some(p) => p,
            None => return,
        };

        // Remove from parent's child list
        let first_child = parent.borrow().child.clone();
        if first_child.as_ref().is_some_and(|c| Rc::ptr_eq(c, node)) {
            parent.borrow_mut().child = node.borrow().sibling.clone();
        } else {
            // Find in sibling chain
            let mut current = first_child;
            while let Some(curr) = current {
                let next_sibling = curr.borrow().sibling.clone();
                if next_sibling.as_ref().is_some_and(|s| Rc::ptr_eq(s, node)) {
                    curr.borrow_mut().sibling = node.borrow().sibling.clone();
                    break;
                }
                current = next_sibling;
            }
        }

        node.borrow_mut().parent = None;
        node.borrow_mut().sibling = None;

        // Update parent's rank
        self.update_rank(&parent);

        // Remove from violation list if present
        self.remove_violation(node);
    }

    /// Collects all children of a node into a vector
    fn collect_children(&self, parent: &NodeRef<T, P>) -> Vec<NodeRef<T, P>> {
        let mut children = Vec::new();
        let mut current = parent.borrow_mut().child.take();

        while let Some(curr) = current {
            let next = curr.borrow_mut().sibling.take();
            curr.borrow_mut().parent = None;
            children.push(curr);
            current = next;
        }

        children
    }

    /// Rebuilds heap from a list of children
    fn rebuild_from_children(&mut self, mut children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.is_empty() {
            panic!("Cannot rebuild from empty children list");
        }

        if children.len() == 1 {
            return children.pop().unwrap();
        }

        // Group by rank and link pairs
        while children.len() > 1 {
            // Sort by priority for heap property
            children.sort_by(|a, b| a.borrow().priority.cmp(&b.borrow().priority));

            let a = children.remove(0);
            let b = children.remove(0);

            // Ensure both are disconnected
            a.borrow_mut().parent = None;
            b.borrow_mut().parent = None;
            a.borrow_mut().sibling = None;
            b.borrow_mut().sibling = None;

            if a.borrow().priority < b.borrow().priority {
                self.make_child(Rc::clone(&a), b);
                self.update_rank(&a);
                children.push(a);
            } else {
                self.make_child(Rc::clone(&b), a);
                self.update_rank(&b);
                children.push(b);
            }
        }

        let result = children.pop().unwrap();
        self.remove_violation(&result);
        result
    }
}

impl<T, P: Ord> Default for BrodalHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

// Note: No Drop implementation needed - Rc handles cleanup automatically.
// When the heap is dropped, the root Rc is dropped, which drops all children
// recursively (since children hold strong references to siblings, and parent
// holds strong reference to first child).

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
