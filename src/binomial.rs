//! Binomial Heap implementation
//!
//! A binomial heap is a collection of binomial trees with:
//! - O(log n) insert and delete_min
//! - O(log n) decrease_key
//! - O(log n) merge (O(1) amortized if merging many heaps)
//!
//! Binomial heaps are simpler than Fibonacci heaps but have worse
//! amortized bounds for decrease_key.
//!
//! # Algorithm Overview
//!
//! A binomial heap maintains a collection of binomial trees, where:
//! - Each tree satisfies the heap property
//! - At most one tree of each degree (0, 1, 2, ..., log n)
//! - This is analogous to binary representation of n
//!
//! **Binomial Tree Bₖ**: Recursively defined:
//! - B₀ is a single node
//! - Bₖ is formed by linking two B_{k-1} trees
//! - Bₖ has exactly 2ᵏ nodes and height k
//!
//! **Key Operations**:
//! - **Insert**: O(log n) worst - merge single-node tree into heap (like binary addition)
//! - **Delete-min**: O(log n) worst - find min, remove, merge its children
//! - **Decrease-key**: O(log n) worst - bubble up in tree (no cutting)
//! - **Merge**: O(log n) worst - merge trees by degree (carry propagation)
//!
//! **Invariant**: After merge, at most one tree of each degree. This ensures
//! O(log n) trees total, bounding operation costs.

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Type alias for node reference (strong reference)
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;

/// Type alias for optional node reference
type NodePtr<T, P> = Option<NodeRef<T, P>>;

/// Type alias for weak node reference (for parent links and handles)
type WeakNodeRef<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a Binomial heap
///
/// The handle uses a weak reference to the node, allowing detection
/// of whether the node has been removed from the heap.
pub struct BinomialHandle<T, P> {
    node: WeakNodeRef<T, P>,
}

impl<T, P> Clone for BinomialHandle<T, P> {
    fn clone(&self) -> Self {
        BinomialHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for BinomialHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node.ptr_eq(&other.node)
    }
}

impl<T, P> Eq for BinomialHandle<T, P> {}

impl<T, P> std::fmt::Debug for BinomialHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinomialHandle")
            .field("valid", &self.node.strong_count())
            .finish()
    }
}

impl<T, P> Handle for BinomialHandle<T, P> {}

/// Internal node structure for binomial heap
///
/// Each node maintains:
/// - `item` and `priority`: The data stored in the heap
/// - `parent`: Weak reference to parent node (None if root)
/// - `child`: Strong reference to first child (None if leaf)
/// - `sibling`: Strong reference to next sibling in parent's child list (None if last child)
/// - `degree`: Number of children (critical for merge operations)
///
/// **Memory Model**: Strong references flow from roots downward (child, sibling).
/// Weak references flow upward (parent) to avoid reference cycles.
///
/// **Binomial Tree Structure**: Nodes form binomial trees where a node of degree k
/// has exactly k children with degrees 0, 1, 2, ..., k-1. This ensures the tree
/// has exactly 2ᵏ nodes.
struct Node<T, P> {
    item: T,
    priority: P,
    /// Parent node - weak reference to avoid cycles (None if root)
    parent: Option<WeakNodeRef<T, P>>,
    /// First child in child list - strong reference (None if leaf)
    child: NodePtr<T, P>,
    /// Next sibling in parent's child list - strong reference (None if last child)
    sibling: NodePtr<T, P>,
    /// Degree: number of children. A binomial tree Bₖ has root degree k and 2ᵏ nodes
    degree: usize,
}

/// Binomial Heap
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::binomial::BinomialHeap;
/// use rust_advanced_heaps::Heap;
///
/// let mut heap = BinomialHeap::new();
/// let handle = heap.insert(5, "item");
/// heap.decrease_key(&handle, 1);
/// assert_eq!(heap.find_min(), Some((&1, &"item")));
/// ```
pub struct BinomialHeap<T, P: Ord> {
    /// Array of binomial trees indexed by degree. Each slot holds at most one tree.
    trees: Vec<NodePtr<T, P>>,
    /// Weak reference to the minimum node for O(1) access
    min: Option<WeakNodeRef<T, P>>,
    /// Number of elements in the heap
    len: usize,
}

// No manual Drop needed - Rc handles cleanup automatically when strong refs go to 0

impl<T, P: Ord> Heap<T, P> for BinomialHeap<T, P> {
    type Handle = BinomialHandle<T, P>;

    fn new() -> Self {
        Self {
            trees: Vec::new(),
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

    /// Inserts a new element into the heap
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**: This is analogous to binary addition with carry propagation
    /// 1. Create a new single-node tree (degree 0, B₀ tree)
    /// 2. Update minimum pointer if necessary
    /// 3. Merge the single-node tree into the heap:
    ///    - Start at degree 0
    ///    - If slot[degree] is empty, place tree there
    ///    - If slot[degree] has a tree, link them (produces degree+1 tree)
    ///    - Continue with carry propagation (like binary addition)
    ///
    /// **Why O(log n)?**
    /// - At most log₂(n) slots in the trees array (since degrees are 0..log n)
    /// - Each link operation is O(1)
    /// - Carry propagation may occur up to log n times
    /// - Worst-case: all slots have trees, requiring log n links
    ///
    /// **Invariant**: After insert, at most one tree of each degree (maintained by
    /// the carry propagation process, just like binary addition maintains at most
    /// one bit per position).
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new single-node tree (B₀ tree, degree 0)
        let node = Rc::new(RefCell::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            degree: 0,
        }));

        // Create handle (weak reference) before we move node into carry propagation
        let handle = BinomialHandle {
            node: Rc::downgrade(&node),
        };

        // Merge this single-node tree into the heap using carry propagation
        let mut carry: NodePtr<T, P> = Some(node);
        let mut degree = 0;

        while carry.is_some() {
            // Ensure trees array is large enough
            if degree >= self.trees.len() {
                self.trees.push(None);
            }

            if self.trees[degree].is_none() {
                // Slot is empty: place tree here
                self.trees[degree] = carry;
                carry = None;
            } else {
                // Slot is occupied: link the two trees
                let existing = self.trees[degree].take().unwrap();
                let new_tree = carry.unwrap();
                carry = Some(self.link_trees(existing, new_tree));
                degree += 1;
            }
        }

        self.len += 1;

        // Update minimum pointer AFTER carry propagation to ensure min points to a root
        self.find_and_update_min();

        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        // We need to return references that live as long as self
        // Since RefCell borrows are temporary, we search the trees array
        // and use pointer magic to get stable references
        let min_weak = self.min.as_ref()?;
        let min_rc = min_weak.upgrade()?;

        // SAFETY: We return references tied to &self lifetime.
        // The Rc keeps the node alive as long as it's in trees[].
        // This is safe because:
        // 1. The node is owned by trees[] (strong ref)
        // 2. We're borrowing self immutably, so trees[] can't change
        // 3. RefCell contents won't move while we hold &self
        let node_ptr = min_rc.as_ptr();
        unsafe { Some((&(*node_ptr).priority, &(*node_ptr).item)) }
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    /// Removes and returns the minimum element
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**:
    /// 1. Find and remove the minimum root (tracked separately, O(1))
    /// 2. Remove minimum tree from the trees array at its degree
    /// 3. Collect all children of the minimum root
    /// 4. Each child is a binomial tree (B_{k-1} if parent was Bₖ)
    /// 5. Create a temporary heap from children
    /// 6. Merge the temporary heap back into the main heap
    /// 7. Find new minimum by scanning all roots (O(log n))
    ///
    /// **Why O(log n)?**
    /// - Minimum root has at most O(log n) children (degree ≤ log n)
    /// - Collecting children: O(log n)
    /// - Merging heaps: O(log n) - merge trees by degree with carry propagation
    /// - Finding new minimum: O(log n) - scan at most log n roots
    /// - Total: O(log n)
    ///
    /// **Binomial Tree Property**: When the root of a Bₖ tree is removed, its
    /// children are Bₖ₋₁, Bₖ₋₂, ..., B₀ trees. This maintains the binomial
    /// tree structure after deletion.
    fn delete_min(&mut self) -> Option<(P, T)> {
        let min_weak = self.min.take()?;
        let min_rc = min_weak.upgrade()?;

        // Remove minimum tree from trees array
        // We must search all slots because after bubble_up, the node's degree
        // may not match its slot (degree changes during swaps but slot doesn't)
        for i in 0..self.trees.len() {
            if let Some(ref tree) = self.trees[i] {
                if Rc::ptr_eq(tree, &min_rc) {
                    self.trees[i] = None;
                    break;
                }
            }
        }

        // Collect children into a temporary heap
        let mut child_heap = self.collect_children(&min_rc);

        // Merge the child heap back into the main heap
        self.merge_trees(&mut child_heap);

        // Find new minimum
        self.find_and_update_min();

        self.len -= 1;

        // Extract item and priority from the minimum node
        // At this point, min_rc should be the only strong reference
        // (we removed it from trees[] and detached children)
        let node = Rc::try_unwrap(min_rc)
            .ok()
            .expect("min node should have no other strong references")
            .into_inner();
        Some((node.priority, node.item))
    }

    /// Decreases the priority of an element
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**:
    /// 1. Update the priority value
    /// 2. **Bubble up** if heap property is violated:
    ///    - Swap node with parent if parent has larger priority
    ///    - Continue upward until heap property satisfied
    ///
    /// **Why O(log n)?**
    /// - Binomial tree has height O(log n)
    /// - We may need to traverse from leaf to root
    /// - Each swap is O(1), but there may be O(log n) swaps
    ///
    /// **Difference from Fibonacci/Pairing heaps**:
    /// - Binomial heaps use **bubble up** instead of **cutting**
    /// - No cascading cuts or marking needed
    /// - Simpler but slower: O(log n) vs O(1) amortized
    ///
    /// **Trade-off**: Simpler implementation, but worse bound for decrease_key.
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node_rc = handle
            .node
            .upgrade()
            .ok_or(HeapError::PriorityNotDecreased)?;

        // Check: new priority must actually be less
        {
            let node = node_rc.borrow();
            if new_priority >= node.priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Update the priority
        node_rc.borrow_mut().priority = new_priority;

        // Bubble up to restore heap property
        self.bubble_up(node_rc);

        Ok(())
    }

    /// Merges another heap into this heap
    ///
    /// **Time Complexity**: O(log n) worst-case, O(1) amortized for sequential merges
    ///
    /// **Algorithm**:
    /// 1. Merge trees from both heaps by degree
    /// 2. Use carry propagation (like binary addition)
    /// 3. For each degree from 0 to max:
    ///    - Collect trees from both heaps at this degree
    ///    - Link pairs until at most one tree remains
    ///    - Carry the result to next degree if linking occurred
    /// 4. Update minimum pointer
    ///
    /// **Why O(log n)?**
    /// - At most O(log n) distinct degrees in each heap
    /// - Processing each degree: O(1) per tree
    /// - Total trees: O(log n)
    /// - Worst-case: O(log n)
    ///
    /// **Why O(1) amortized for sequential merges?**
    /// - Similar to binary addition: most merges are cheap
    /// - Expensive merges (with many carry propagations) are rare
    /// - Amortized analysis shows average cost is O(1) per merge
    ///
    /// **Invariant**: After merge, at most one tree of each degree (maintained by
    /// carry propagation, exactly like binary addition maintains at most one bit
    /// per position).
    fn merge(&mut self, mut other: Self) {
        // Merge trees from both heaps
        self.merge_trees(&mut other);

        // Update minimum pointer after merge
        self.find_and_update_min();

        // Update length
        self.len += other.len;
        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> BinomialHeap<T, P> {
    /// Collects children of a node into a temporary heap
    ///
    /// This is a helper function for `delete_min` that extracts all children
    /// of a node and creates a new heap from them.
    fn collect_children(&self, node: &NodeRef<T, P>) -> BinomialHeap<T, P> {
        let mut child_heap = BinomialHeap::new();

        // Take the child from the node
        let first_child = node.borrow_mut().child.take();

        if let Some(child) = first_child {
            // Children are linked in a sibling list
            // We need to collect them and clear parent links
            let mut children: Vec<NodeRef<T, P>> = Vec::new();
            let mut current = Some(child);

            // Collect all children, clearing parent links
            while let Some(curr) = current {
                let next = curr.borrow_mut().sibling.take();
                curr.borrow_mut().parent = None; // Clear parent link
                children.push(curr);
                current = next;
            }

            // Add each child tree to the temporary heap at its degree slot
            // IMPORTANT: After bubble_up, degrees might conflict. If a slot is
            // already occupied, we need to merge instead of overwrite.
            for child_node in children {
                let mut node_to_add = child_node;
                loop {
                    let degree = node_to_add.borrow().degree;

                    // Ensure child_heap.trees array is large enough
                    while child_heap.trees.len() <= degree {
                        child_heap.trees.push(None);
                    }

                    // Check if slot is already occupied
                    if let Some(existing) = child_heap.trees[degree].take() {
                        // Conflict: merge the two trees
                        node_to_add = self.link_trees(node_to_add, existing);
                        // Continue loop to place the merged tree
                    } else {
                        // Slot is empty, place the tree
                        child_heap.trees[degree] = Some(node_to_add);
                        break;
                    }
                }
            }
        }

        child_heap
    }

    /// Links two binomial trees of the same degree into one tree of degree+1
    ///
    /// **Time Complexity**: O(1)
    ///
    /// **Algorithm**:
    /// - Compare priorities of the two roots
    /// - Make the tree with larger priority a child of the one with smaller priority
    /// - This maintains heap property: parent <= child
    /// - The resulting tree has degree one higher than the input trees
    ///
    /// **Binomial Tree Property**:
    /// - Linking two Bₖ trees produces a B_{k+1} tree
    /// - If we link Bₖ rooted at a and Bₖ rooted at b, and a.priority < b.priority:
    ///   - a becomes the root of the new B_{k+1} tree
    ///   - b becomes the leftmost child of a
    ///   - a's children (B_{k-1}, B_{k-2}, ..., B₀) become siblings of b
    ///   - The new tree has degree k+1 and 2^{k+1} nodes
    ///
    /// **Invariant**: This operation maintains:
    /// - Heap property: parent priority <= child priority
    /// - Binomial tree structure: exactly 2ᵏ nodes in a Bₖ tree
    fn link_trees(&self, a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        // Determine which should be parent based on priority
        let a_is_parent = a.borrow().priority <= b.borrow().priority;

        let (parent, child) = if a_is_parent { (a, b) } else { (b, a) };

        // Link child as a new first child of parent
        {
            let mut child_ref = child.borrow_mut();
            let mut parent_ref = parent.borrow_mut();

            // child's parent is now parent
            child_ref.parent = Some(Rc::downgrade(&parent));

            // child's sibling is parent's old first child
            child_ref.sibling = parent_ref.child.take();

            // parent's first child is now child
            parent_ref.child = Some(Rc::clone(&child));

            // parent's degree increased by 1
            parent_ref.degree += 1;
        }

        // Drop the child Rc here since we cloned it into parent.child
        drop(child);

        parent
    }

    /// Merges trees from another heap into this one
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**: This is analogous to binary addition with carry propagation
    /// 1. For each degree from 0 to max:
    ///    - Collect trees from both heaps at this degree
    ///    - Add carry from previous degree if present
    ///    - Link pairs of trees until at most one remains
    ///    - If result has degree+1, it becomes the carry for next degree
    ///    - Otherwise, place it at the current degree
    /// 2. Handle final carry if present
    ///
    /// **Why like binary addition?**
    /// - Each heap has at most one tree per degree (like binary digits)
    /// - When two trees of same degree exist, link them (like adding 1+1=2, carry 1)
    /// - The linked tree has degree+1 (like the carry bit)
    /// - Process degrees from 0 to max (like processing bits from right to left)
    ///
    /// **Invariant**: After merge, at most one tree of each degree (just like binary
    /// addition maintains at most one bit per position).
    fn merge_trees(&mut self, other: &mut Self) {
        // Ensure trees array is large enough for both heaps
        let max_degree = self.trees.len().max(other.trees.len());
        while self.trees.len() < max_degree {
            self.trees.push(None);
        }

        // Carry propagation: when we link two trees, we may produce a tree of higher degree
        let mut carry: NodePtr<T, P> = None;

        // Process each degree from 0 to max_degree
        for degree in 0..max_degree {
            let mut trees: Vec<NodeRef<T, P>> = Vec::new();

            // Step 1: Collect trees from both heaps at this degree
            if degree < self.trees.len() {
                if let Some(tree) = self.trees[degree].take() {
                    trees.push(tree);
                }
            }

            if degree < other.trees.len() {
                if let Some(tree) = other.trees[degree].take() {
                    trees.push(tree);
                }
            }

            // Step 2: Add carry from previous degree if present
            if let Some(c) = carry.take() {
                trees.push(c);
            }

            // Step 3: Link pairs of trees until at most one remains
            while trees.len() > 1 {
                let a = trees.pop().unwrap();
                let b = trees.pop().unwrap();
                let linked = self.link_trees(a, b);

                let linked_degree = linked.borrow().degree;
                if linked_degree == degree + 1 {
                    // Linked tree has degree+1: it becomes carry for next degree
                    carry = Some(linked);
                } else {
                    // Linked tree has same degree: continue linking
                    trees.push(linked);
                }
            }

            // Step 4: Place remaining tree (if any) at this degree slot
            if let Some(tree) = trees.pop() {
                let tree_degree = tree.borrow().degree;
                if tree_degree == degree {
                    self.trees[degree] = Some(tree);
                } else {
                    carry = Some(tree);
                }
            }
        }

        // Step 5: Handle final carry
        if let Some(c) = carry {
            let degree = c.borrow().degree;
            while self.trees.len() <= degree {
                self.trees.push(None);
            }
            self.trees[degree] = Some(c);
        }
    }

    /// Bubbles up a node to maintain heap property by swapping node positions.
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**:
    /// 1. While node has a parent and heap property is violated:
    ///    - Swap node positions (child moves to parent's position)
    ///    - The node keeps its data, only its position in tree changes
    /// 2. Update minimum pointer if node became root
    ///
    /// **Handle Semantics**: By swapping node positions instead of values,
    /// handles remain valid - they still point to the same node with the
    /// same data, just at a different tree position.
    ///
    /// **Note**: This relaxes strict binomial tree structure (degrees may not
    /// match slots), but maintains heap property. Structure is restored during
    /// delete_min consolidation.
    fn bubble_up(&mut self, node: NodeRef<T, P>) {
        let current = node;
        let mut did_swap = false;

        loop {
            // Get parent weak ref
            let parent_weak = {
                let node_ref = current.borrow();
                match &node_ref.parent {
                    Some(p) => p.clone(),
                    None => break, // Reached root
                }
            };

            // Upgrade to strong ref
            let parent = match parent_weak.upgrade() {
                Some(p) => p,
                None => break, // Parent gone (shouldn't happen)
            };

            // Check if we should swap
            let should_swap = current.borrow().priority < parent.borrow().priority;

            if !should_swap {
                break; // Heap property satisfied
            }

            // Swap node positions: current moves to parent's position
            self.swap_node_with_parent(&current, &parent);
            did_swap = true;

            // current is now in parent's old position, continue loop
            // to check against grandparent
        }

        // If we did swaps and current is now a root, we need to fix the tree slot
        // because the degree changed but the slot didn't
        if did_swap && current.borrow().parent.is_none() {
            // Current is a root - find it in trees and move to correct slot
            let current_degree = current.borrow().degree;

            // Find and remove current from its old slot
            let mut old_slot = None;
            for i in 0..self.trees.len() {
                if let Some(ref tree) = self.trees[i] {
                    if Rc::ptr_eq(tree, &current) {
                        old_slot = Some(i);
                        break;
                    }
                }
            }

            if let Some(old_i) = old_slot {
                if old_i != current_degree {
                    // Slot mismatch - need to move tree
                    self.trees[old_i] = None;

                    // Place at correct slot, handling conflicts
                    let mut tree_to_place = current.clone();
                    let mut target_degree = current_degree;

                    while self.trees.len() <= target_degree {
                        self.trees.push(None);
                    }

                    loop {
                        if let Some(existing) = self.trees[target_degree].take() {
                            // Conflict - merge and try next slot
                            tree_to_place = self.link_trees(tree_to_place, existing);
                            target_degree = tree_to_place.borrow().degree;
                            while self.trees.len() <= target_degree {
                                self.trees.push(None);
                            }
                        } else {
                            self.trees[target_degree] = Some(tree_to_place);
                            break;
                        }
                    }

                    // After slot fix with potential merges, rescan all roots for min
                    self.find_and_update_min();
                    return;
                }
            }
        }

        // Update minimum pointer if needed
        self.update_min_if_needed(&current);
    }

    /// Swaps a child node with its parent, moving child to parent's position.
    ///
    /// After this operation:
    /// - child is where parent was (same grandparent, same siblings as parent had)
    /// - parent is a child of child
    /// - child's old children become siblings of parent
    /// - parent keeps its other children
    fn swap_node_with_parent(&mut self, child: &NodeRef<T, P>, parent: &NodeRef<T, P>) {
        // Step 1: Remove child from parent's child list
        {
            let child_sibling = child.borrow().sibling.clone();
            let mut parent_ref = parent.borrow_mut();

            if let Some(ref first_child) = parent_ref.child {
                if Rc::ptr_eq(first_child, child) {
                    // Child is first child, replace with its sibling
                    parent_ref.child = child_sibling;
                } else {
                    // Child is not first, find and remove from sibling chain
                    let mut prev = first_child.clone();
                    loop {
                        let next = prev.borrow().sibling.clone();
                        match next {
                            Some(ref n) if Rc::ptr_eq(n, child) => {
                                prev.borrow_mut().sibling = child_sibling;
                                break;
                            }
                            Some(n) => prev = n,
                            None => break, // Not found (shouldn't happen)
                        }
                    }
                }
            }
        }

        // Step 2: Save parent's old position info
        let grandparent_weak = parent.borrow().parent.clone();
        let parent_sibling = parent.borrow().sibling.clone();

        // Step 3: Child takes parent's position
        child.borrow_mut().parent = grandparent_weak.clone();
        child.borrow_mut().sibling = parent_sibling;

        // Step 4: Update grandparent's child pointer (or trees array if parent was root)
        if let Some(ref gp_weak) = grandparent_weak {
            if let Some(gp) = gp_weak.upgrade() {
                let mut gp_ref = gp.borrow_mut();
                if let Some(ref first) = gp_ref.child {
                    if Rc::ptr_eq(first, parent) {
                        gp_ref.child = Some(child.clone());
                    } else {
                        // Find parent in sibling chain and replace
                        let mut prev = first.clone();
                        loop {
                            let next = prev.borrow().sibling.clone();
                            match next {
                                Some(ref n) if Rc::ptr_eq(n, parent) => {
                                    prev.borrow_mut().sibling = Some(child.clone());
                                    break;
                                }
                                Some(n) => prev = n,
                                None => break,
                            }
                        }
                    }
                }
            }
        } else {
            // Parent was a root - update trees array
            // We must search all slots because the parent's degree may have changed
            // through previous bubble_up swaps, so it might not be at its degree slot.
            let mut found = false;
            for i in 0..self.trees.len() {
                if let Some(ref tree) = self.trees[i] {
                    if Rc::ptr_eq(tree, parent) {
                        self.trees[i] = Some(child.clone());
                        found = true;
                        break;
                    }
                }
            }

            debug_assert!(found, "Parent root should be found in trees array");
        }

        // Step 5: Parent becomes a child of child
        // Parent's new siblings = child's old children
        let child_old_children = child.borrow_mut().child.take();
        parent.borrow_mut().sibling = child_old_children;
        parent.borrow_mut().parent = Some(Rc::downgrade(child));
        child.borrow_mut().child = Some(parent.clone());

        // Step 6: Update degrees
        // Child gains one child (parent): degree += 1
        // Parent loses one child (child): degree -= 1
        child.borrow_mut().degree += 1;
        if parent.borrow().degree > 0 {
            parent.borrow_mut().degree -= 1;
        }

        // Invariant: Both nodes must still have strong references (be in the tree)
        debug_assert!(
            Rc::strong_count(child) > 0,
            "Child node lost all strong references during swap"
        );
        debug_assert!(
            Rc::strong_count(parent) > 0,
            "Parent node lost all strong references during swap"
        );
    }

    /// Updates the minimum pointer if the given node has a smaller priority
    fn update_min_if_needed(&mut self, node: &NodeRef<T, P>) {
        let node_priority = &node.borrow().priority;

        let should_update = match &self.min {
            Some(min_weak) => {
                if let Some(min_rc) = min_weak.upgrade() {
                    node_priority < &min_rc.borrow().priority
                } else {
                    true // Min is invalid
                }
            }
            None => true, // No minimum yet
        };

        if should_update {
            self.min = Some(Rc::downgrade(node));
        }
    }

    /// Finds and updates the minimum pointer by scanning all roots
    fn find_and_update_min(&mut self) {
        self.min = None;

        for root in self.trees.iter().flatten() {
            let should_update = match &self.min {
                Some(min_weak) => {
                    if let Some(min_rc) = min_weak.upgrade() {
                        root.borrow().priority < min_rc.borrow().priority
                    } else {
                        true
                    }
                }
                None => true,
            };

            if should_update {
                self.min = Some(Rc::downgrade(root));
            }
        }
    }

    /// Debug helper: count all nodes in the heap recursively
    #[cfg(test)]
    fn count_all_nodes(&self) -> usize {
        fn count_tree<T, P>(node: &NodeRef<T, P>) -> usize {
            let node_ref = node.borrow();
            let mut count = 1; // This node

            // Count children
            if let Some(ref child) = node_ref.child {
                let mut current = Some(child.clone());
                while let Some(curr) = current {
                    count += count_tree(&curr);
                    current = curr.borrow().sibling.clone();
                }
            }

            count
        }

        let mut total = 0;
        for tree in self.trees.iter().flatten() {
            total += count_tree(tree);
        }
        total
    }

    /// Debug helper: find the actual minimum by scanning all nodes
    #[cfg(test)]
    fn find_actual_min(&self) -> Option<P>
    where
        P: Clone,
    {
        fn find_min_in_tree<T, P: Ord + Clone>(
            node: &NodeRef<T, P>,
            current_min: Option<P>,
        ) -> Option<P> {
            let node_ref = node.borrow();
            let node_priority = node_ref.priority.clone();

            let mut min = match current_min {
                Some(m) if m < node_priority => Some(m),
                _ => Some(node_priority),
            };

            // Check children
            if let Some(ref child) = node_ref.child {
                let mut current = Some(child.clone());
                while let Some(curr) = current {
                    min = find_min_in_tree(&curr, min);
                    current = curr.borrow().sibling.clone();
                }
            }

            min
        }

        let mut overall_min: Option<P> = None;
        for tree in self.trees.iter().flatten() {
            overall_min = find_min_in_tree(tree, overall_min);
        }
        overall_min
    }
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decrease_key_min_update() {
        let mut heap: BinomialHeap<i32, i32> = BinomialHeap::new();
        let mut handles = Vec::new();

        // Insert 16 zeros
        for i in 0..16 {
            handles.push(heap.push(0, i));
        }
        println!("After 16 pushes, min: {:?}", heap.peek());
        println!(
            "Trees: {:?}",
            heap.trees.iter().map(|t| t.is_some()).collect::<Vec<_>>()
        );

        // Decrease handle 1 to -1
        heap.decrease_key(&handles[1], -1).unwrap();
        println!("\nAfter decrease_key(1, -1):");
        println!("  min: {:?}", heap.peek());
        println!(
            "  Trees: {:?}",
            heap.trees.iter().map(|t| t.is_some()).collect::<Vec<_>>()
        );
        println!("  handle[1] valid: {}", handles[1].node.strong_count());
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-1));

        // Decrease handle 3 to -98
        println!("\nBefore decrease_key(3, -98):");
        println!("  handle[3] valid: {}", handles[3].node.strong_count());
        heap.decrease_key(&handles[3], -98).unwrap();
        println!("After decrease_key(3, -98):");
        println!(
            "  Trees: {:?}",
            heap.trees.iter().map(|t| t.is_some()).collect::<Vec<_>>()
        );
        println!("  handle[1] valid: {}", handles[1].node.strong_count());
        println!("  handle[3] valid: {}", handles[3].node.strong_count());
        if let Some(min_weak) = &heap.min {
            println!("  min weak strong_count: {}", min_weak.strong_count());
        } else {
            println!("  min is None!");
        }
        println!("  min: {:?}", heap.peek());
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-98), "Min should be -98");
    }

    #[test]
    fn test_complex_operations_failure() {
        // Simplified test case focusing on the issue
        let mut heap: BinomialHeap<i32, i32> = BinomialHeap::new();
        let mut handles = Vec::new();

        // Insert: [54, -34, 48, 55, 19, 8, 23, 87]
        let initial = [54i32, -34, 48, 55, 19, 8, 23, 87];
        for (i, priority) in initial.iter().enumerate() {
            handles.push(heap.push(*priority, i as i32));
        }
        println!("After initial inserts, min: {:?}", heap.peek());
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-34));

        // decrease_key(handle[6], 23 -> -7)
        println!("\nDecreasing handle[6] from 23 to -7");
        heap.decrease_key(&handles[6], -7).unwrap();
        println!("After decrease_key, min: {:?}", heap.peek());
        // Now the minimum should be -34 (not -7, since -34 < -7)
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-34));

        // Push a few more items
        handles.push(heap.push(71, 8)); // idx 8
        handles.push(heap.push(94, 9)); // idx 9
        handles.push(heap.push(-87, 10)); // idx 10
        println!("After pushes, min: {:?}", heap.peek());
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-87)); // -87 is now min

        // Pop -87
        let p = heap.pop();
        println!("Pop: {:?}", p);
        assert_eq!(p.map(|(p, _)| p), Some(-87));
        println!("After pop, min: {:?}", heap.peek());
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-34)); // -34 is next min

        // decrease_key(handle[5], 8 -> 7)
        println!("\nDecreasing handle[5] from 8 to 7");
        heap.decrease_key(&handles[5], 7).unwrap();
        println!("After decrease_key, min: {:?}", heap.peek());
        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-34)); // Still -34

        // Pop -34
        let p = heap.pop();
        println!("Pop: {:?}", p);
        assert_eq!(p.map(|(p, _)| p), Some(-34));

        // Now the minimum should be -7 (handle[6])
        println!("After pop, min: {:?}", heap.peek());
        println!("Expected: -7 (handle 6)");

        // Debug: print all root priorities
        println!("\nRoots in trees:");
        for (i, tree) in heap.trees.iter().enumerate() {
            if let Some(root) = tree {
                println!(
                    "  trees[{}]: priority={}, item={}, degree={}",
                    i,
                    root.borrow().priority,
                    root.borrow().item,
                    root.borrow().degree
                );
            }
        }

        assert_eq!(heap.peek().map(|(p, _)| *p), Some(-7), "Min should be -7");
    }

    #[test]
    fn test_failing_input_from_proptest() {
        use std::collections::HashMap;

        let mut heap: BinomialHeap<i32, i32> = BinomialHeap::new();
        let mut handles = Vec::new();
        let mut priorities: HashMap<usize, i32> = HashMap::new();

        // From failing proptest
        let initial: Vec<i32> = vec![
            54, -72, 30, -53, -78, -71, -30, -7, -41, 93, 14, -5, -84, -60, -9, 5, -93, 34,
        ];

        for (i, priority) in initial.iter().enumerate() {
            handles.push(heap.push(*priority, i as i32));
            priorities.insert(i, *priority);
        }

        // Verify after inserts
        let actual_min = heap.find_actual_min();
        let expected_min = priorities.values().min().copied();
        assert_eq!(actual_min, expected_min, "Initial min mismatch");
        println!(
            "After inserts: peek={:?}, expected_min={:?}",
            heap.peek().map(|(p, _)| *p),
            expected_min
        );

        // Decrease some keys
        let decrease_ops = [(0, -100), (5, -200), (10, -50)];
        for (idx, new_priority) in decrease_ops {
            if priorities.contains_key(&idx) {
                let old_priority = priorities[&idx];
                if new_priority < old_priority {
                    println!(
                        "decrease_key({}, {} -> {})",
                        idx, old_priority, new_priority
                    );
                    heap.decrease_key(&handles[idx], new_priority).unwrap();
                    priorities.insert(idx, new_priority);

                    // Verify heap invariant
                    let peek_val = heap.peek().map(|(p, _)| *p);
                    let actual_min = heap.find_actual_min();
                    let expected_min = priorities.values().min().copied();

                    println!(
                        "  After: peek={:?}, actual_min={:?}, expected={:?}",
                        peek_val, actual_min, expected_min
                    );

                    if peek_val != expected_min || actual_min != expected_min {
                        println!("  MISMATCH!");
                        println!(
                            "  Node count: {}, priorities count: {}",
                            heap.count_all_nodes(),
                            priorities.len()
                        );
                        panic!("Heap invariant violated after decrease_key");
                    }
                }
            }
        }

        // Now do some pops and verify node count
        for i in 0..10 {
            if !priorities.is_empty() {
                let node_count = heap.count_all_nodes();
                let expected_count = priorities.len();

                if node_count != expected_count {
                    println!(
                        "NODE COUNT MISMATCH at step {}: in_heap={}, expected={}",
                        i, node_count, expected_count
                    );
                    panic!("Lost nodes!");
                }

                let peek_before = heap.peek().map(|(p, _)| *p);
                let expected_min = priorities.values().min().copied();

                if peek_before != expected_min {
                    println!(
                        "MISMATCH at pop {}: peek={:?}, expected={:?}",
                        i, peek_before, expected_min
                    );
                    panic!("Heap invariant violated before pop");
                }

                if let Some((_priority, item)) = heap.pop() {
                    println!("pop() -> ({}, {})", _priority, item);
                    priorities.remove(&(item as usize));
                }
            }
        }

        println!("All operations completed successfully");
    }
}
