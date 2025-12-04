//! Pairing Heap implementation
//!
//! A pairing heap is a type of heap-ordered tree with:
//! - O(1) amortized insert and merge
//! - O(log n) amortized delete_min
//! - o(log n) amortized decrease_key (in fact, better than log n)
//!
//! The pairing heap is simpler than Fibonacci heaps while still providing
//! excellent amortized performance for decrease_key operations.
//!
//! # Algorithm Overview
//!
//! The pairing heap maintains a heap-ordered multi-way tree. Unlike binary heaps,
//! nodes can have any number of children. The key operations are:
//!
//! - **Insert**: O(1) amortized - simply compare with root and link
//! - **Delete-min**: O(log n) amortized - uses two-pass pairing to merge children
//! - **Decrease-key**: o(log n) amortized - cut from parent and merge with root
//! - **Merge**: O(1) amortized - compare roots and link
//!
//! The two-pass pairing strategy in delete_min is crucial for achieving the
//! amortized bounds. See the merge_pairs function for details.
//!
//! # Key Invariants
//!
//! 1. **Heap property**: For any node, parent.priority <= child.priority
//! 2. **Tree structure**: Each node has at most one parent
//! 3. **Sibling list**: Children of a node form a linked list via sibling pointers
//! 4. **Root tracking**: The minimum element is always at the root
//!
//! # Why Pairing Heaps?
//!
//! Pairing heaps were designed as a simpler alternative to Fibonacci heaps that
//! would be "competitive in theory and easy to implement and fast in practice."
//! The structure uses a simple multi-way tree where each node stores pointers
//! to its leftmost child and right sibling.
//!
//! The key operation is the two-pass pairing during delete-min: children are
//! paired left-to-right, then the resulting trees are merged right-to-left.
//! This pairing strategy gives the heap its name.
//!
//! Interestingly, the exact complexity of decrease-key remained an open problem
//! for decades. The current best bound is o(log n) amortized (strictly better
//! than log n), proven by Iacono and Özkan in 2014.
//!
//! # References
//!
//! - Fredman, M. L., Sedgewick, R., Sleator, D. D., & Tarjan, R. E. (1986).
//!   "The pairing heap: A new form of self-adjusting heap." *Algorithmica*, 1(1), 111-129.
//!   [Springer](https://link.springer.com/article/10.1007/BF01840439)
//! - [Wikipedia: Pairing heap](https://en.wikipedia.org/wiki/Pairing_heap)

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Type alias for strong reference to a node
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;
/// Type alias for weak reference to a node (used for backlinks)
type NodeWeak<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a Pairing heap
///
/// The handle uses a weak reference to the node. If the node has been removed
/// from the heap (e.g., via delete_min), the handle becomes invalid.
/// Operations on invalid handles will fail gracefully.
pub struct PairingHandle<T, P> {
    node: NodeWeak<T, P>,
}

// Manual Clone implementation to avoid requiring T: Clone and P: Clone
impl<T, P> Clone for PairingHandle<T, P> {
    fn clone(&self) -> Self {
        PairingHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for PairingHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node.ptr_eq(&other.node)
    }
}

impl<T, P> Eq for PairingHandle<T, P> {}

impl<T, P> Handle for PairingHandle<T, P> {}

/// Internal node structure for pairing heap
///
/// Each node maintains:
/// - `item` and `priority`: The data stored in the heap
/// - `child`: Strong reference to the first child in the child list
/// - `sibling`: Strong reference to the next sibling in the parent's child list
/// - `prev`: Weak reference to parent or previous sibling (backlink for efficient cutting)
///
/// The structure forms a multi-way tree where children are linked via sibling pointers.
/// Strong references flow downward (parent to child), weak references point upward (child to parent).
/// This prevents reference cycles while allowing O(1) insertion and merging.
struct Node<T, P> {
    item: T,
    priority: P,
    /// First child in the child list. None if this node is a leaf.
    /// Uses strong reference (Rc) as parent owns children.
    child: Option<NodeRef<T, P>>,
    /// Next sibling in the parent's child list. None if this is the last child.
    /// Uses strong reference (Rc) as earlier siblings own later ones in the list.
    sibling: Option<NodeRef<T, P>>,
    /// Parent node or previous sibling. Uses weak reference to avoid cycles.
    /// Used for:
    /// - Traversing up the tree during decrease_key
    /// - Efficiently removing nodes from child lists
    /// - Maintaining bidirectional links for O(1) updates
    prev: Option<NodeWeak<T, P>>,
}

/// Pairing Heap
///
/// A safe implementation using `Rc` and `Weak` references instead of raw pointers.
/// Strong references (`Rc`) flow from root to children, weak references (`Weak`)
/// point upward for efficient decrease_key operations.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::pairing::PairingHeap;
/// use rust_advanced_heaps::Heap;
///
/// let mut heap = PairingHeap::new();
/// let handle = heap.insert(5, "item");
/// heap.decrease_key(&handle, 1);
/// assert_eq!(heap.find_min(), Some((&1, &"item")));
/// ```
pub struct PairingHeap<T, P: Ord> {
    /// Root of the heap tree. Uses strong reference as the heap owns the root.
    root: Option<NodeRef<T, P>>,
    len: usize,
}

// No Drop implementation needed - Rc handles cleanup automatically.
// When the heap is dropped, the root Rc's refcount goes to zero,
// which recursively drops all children (via their strong references).

impl<T, P: Ord> Heap<T, P> for PairingHeap<T, P> {
    type Handle = PairingHandle<T, P>;

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
        self.insert(priority, item)
    }

    /// Inserts a new element into the heap
    ///
    /// **Time Complexity**: O(1) amortized
    ///
    /// **Algorithm**:
    /// 1. Create a new single-node tree
    /// 2. Compare priority with current root
    /// 3. If new priority is smaller, make new node the root (old root becomes its child)
    /// 4. Otherwise, add new node as a child of the root
    ///
    /// This operation is O(1) because we only compare with the root and perform
    /// a constant amount of reference manipulation. No restructuring needed!
    ///
    /// **Invariant Maintenance**:
    /// - Heap property maintained: smaller priority becomes parent
    /// - Tree structure preserved: parent-child relationships correctly linked
    /// - Root always points to minimum: updated if new element is smaller
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node with no children or siblings yet
        let node = Rc::new(RefCell::new(Node {
            item,
            priority,
            child: None,
            sibling: None,
            prev: None,
        }));

        // Link new node into the tree structure
        if let Some(ref root) = self.root {
            if node.borrow().priority < root.borrow().priority {
                // Case 1: New node has smaller priority
                // Make new node the root, old root becomes its only child
                // This maintains the heap property (parent <= child)
                node.borrow_mut().child = Some(Rc::clone(root));
                root.borrow_mut().prev = Some(Rc::downgrade(&node));
                self.root = Some(Rc::clone(&node));
            } else {
                // Case 2: Current root has smaller or equal priority
                // Add new node as the first child of root
                // This maintains heap property (new node >= root)
                let root_child = root.borrow().child.clone();
                node.borrow_mut().sibling = root_child.clone();
                node.borrow_mut().prev = Some(Rc::downgrade(root));
                if let Some(ref child) = root_child {
                    child.borrow_mut().prev = Some(Rc::downgrade(&node));
                }
                root.borrow_mut().child = Some(Rc::clone(&node));
            }
        } else {
            // Empty heap: new node becomes root
            self.root = Some(Rc::clone(&node));
        }

        self.len += 1;
        PairingHandle {
            node: Rc::downgrade(&node),
        }
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        // Safety: The references we return point to data inside an Rc that the heap owns.
        // As long as the caller has a borrow of &self, the Rc stays alive and the data is valid.
        // This is effectively a lifetime extension that is safe because:
        // 1. The heap owns the Rc (root)
        // 2. The returned references are bounded by the &self lifetime
        // 3. We're only reading, not mutating
        self.root.as_ref().map(|root| {
            let node = root.as_ptr();
            unsafe { (&(*node).priority, &(*node).item) }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    /// Removes and returns the minimum element
    ///
    /// **Time Complexity**: O(log n) amortized
    ///
    /// **Algorithm**:
    /// 1. Remove the root (which contains the minimum)
    /// 2. Collect all children of the root
    /// 3. Use two-pass pairing to merge children into a single tree
    /// 4. The result becomes the new root
    ///
    /// **Two-Pass Pairing Strategy** (critical for amortized bounds):
    /// - **First pass**: Pair adjacent children and merge each pair
    ///   - This reduces n children to approximately n/2 merged trees
    /// - **Second pass**: Merge pairs from right to left
    ///   - This produces a single balanced tree
    ///
    /// The two-pass strategy ensures that the tree doesn't become too unbalanced,
    /// which is essential for the amortized analysis. Without it, we might get
    /// O(n) worst-case delete_min operations.
    ///
    /// **Why Two Passes?**
    /// - Single pass (left-to-right) would create an unbalanced structure
    /// - Two passes ensure logarithmic height in the amortized sense
    /// - Right-to-left merge in second pass balances the tree better
    fn delete_min(&mut self) -> Option<(P, T)> {
        let root = self.root.take()?;

        // Get the first child before we consume the root
        let first_child = root.borrow_mut().child.take();

        // The root has been deleted, so we need to rebuild from its children
        // If there are no children, the heap becomes empty
        // Otherwise, we merge all children using two-pass pairing
        if let Some(child) = first_child {
            // Clear the child's prev pointer (it was pointing to the old root)
            child.borrow_mut().prev = None;
            // Two-pass pairing: merge children efficiently
            // This is the core operation that achieves O(log n) amortized bounds
            self.root = Some(self.merge_pairs(child));
        }
        // If no children, self.root is already None from take()

        self.len -= 1;

        // Verify invariants after restructuring
        #[cfg(feature = "expensive_verify")]
        {
            let count = self.verify_heap_property();
            assert_eq!(
                count, self.len,
                "Length mismatch after delete_min: counted {} nodes but len is {}",
                count, self.len
            );
        }

        // Extract item and priority from the root node
        // At this point, root should be the only strong reference, so we can unwrap
        // If try_unwrap fails, there's a bug in our reference counting
        let cell = Rc::try_unwrap(root).unwrap_or_else(|rc| {
            panic!(
                "BUG: root node has {} strong references during delete_min (expected 1)",
                Rc::strong_count(&rc)
            )
        });
        let node = cell.into_inner();
        Some((node.priority, node.item))
    }

    /// Decreases the priority of an element
    ///
    /// **Time Complexity**: o(log n) amortized (better than O(log n)!)
    ///
    /// **Precondition**: `new_priority < current_priority` (undefined behavior otherwise)
    ///
    /// **Algorithm**:
    /// 1. Update the priority value
    /// 2. If heap property is violated (new priority < parent priority):
    ///    - Cut the node from its parent (remove from child list)
    ///    - Merge the node with the root (or make it the new root if smaller)
    /// 3. The cut operation is O(1), but may cause restructuring
    ///
    /// **Why o(log n) and not O(log n)?**
    /// The notation o(log n) means "strictly better than O(log n)" in the amortized sense.
    /// This means that while individual operations might take O(log n), over a sequence
    /// of operations, the average cost is provably less than any constant times log n.
    ///
    /// This sub-logarithmic bound comes from the amortized analysis showing that
    /// the pairing heap structure allows most decrease_key operations to be cheap
    /// (cutting near the root), while expensive ones (deep cuts) are rare.
    ///
    /// **Cut Operation**:
    /// Cutting a node means removing it from its parent's child list. Since we
    /// maintain prev pointers, this is O(1). We then add it as a child of the
    /// root (or make it root if it's smaller). This maintains heap property.
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        // Upgrade the weak reference to get the node
        // If the upgrade fails, the node was already removed from the heap
        let node = handle
            .node
            .upgrade()
            .ok_or(HeapError::PriorityNotDecreased)?;

        // Safety check: new priority must actually be less
        if new_priority >= node.borrow().priority {
            return Err(HeapError::PriorityNotDecreased);
        }

        // Update the priority value
        node.borrow_mut().priority = new_priority;

        // Check if the node is already the root
        let is_root = self
            .root
            .as_ref()
            .map(|r| Rc::ptr_eq(r, &node))
            .unwrap_or(false);

        if is_root {
            // Node is already the root, no restructuring needed
            return Ok(());
        }

        // The node is not the root, so it has a parent
        // Strategy: Cut the node from its parent and merge it with the root
        // This is the standard decrease_key operation for pairing heaps

        // Step 1: Cut the node from its parent's child list
        self.cut_node(&node);

        // Step 2: Merge the cut node with the root using merge_nodes
        // This properly handles preserving children of both nodes
        if let Some(root) = self.root.take() {
            self.root = Some(self.merge_nodes(node, root));
        } else {
            // Heap is empty (shouldn't happen, but handle gracefully)
            self.root = Some(node);
        }

        // Verify invariants after decrease_key
        #[cfg(feature = "expensive_verify")]
        {
            let count = self.verify_heap_property();
            assert_eq!(
                count, self.len,
                "Length mismatch after decrease_key: counted {} nodes but len is {}",
                count, self.len
            );
        }

        Ok(())
    }

    /// Merges another heap into this heap
    ///
    /// **Time Complexity**: O(1) amortized
    ///
    /// **Algorithm**:
    /// 1. Compare roots of both heaps
    /// 2. Make the larger-priority root a child of the smaller-priority root
    /// 3. Update child/sibling pointers accordingly
    /// 4. The smaller root becomes the new root
    ///
    /// This is trivially O(1): we only need one comparison and a few reference updates.
    /// No restructuring needed because we're just linking two roots!
    ///
    /// **Post-condition**: After merge, `other` is empty (consumed by this heap)
    fn merge(&mut self, mut other: Self) {
        // Empty heaps are easy cases
        if other.is_empty() {
            return; // Nothing to merge
        }

        if self.is_empty() {
            // This heap is empty: just take the other heap
            *self = other;
            return;
        }

        // Both heaps are non-empty: need to link them
        let self_root = self.root.take().unwrap();
        let other_root = other.root.take().unwrap();

        // Compare roots: smaller priority becomes parent (heap property)
        if other_root.borrow().priority < self_root.borrow().priority {
            // Other root has smaller priority: it becomes the new root
            // Self root becomes a child of other root
            let other_child = other_root.borrow().child.clone();
            self_root.borrow_mut().sibling = other_child.clone();
            self_root.borrow_mut().prev = Some(Rc::downgrade(&other_root));
            if let Some(ref child) = other_child {
                child.borrow_mut().prev = Some(Rc::downgrade(&self_root));
            }
            other_root.borrow_mut().child = Some(self_root);
            self.root = Some(other_root);
        } else {
            // Self root has smaller or equal priority: it stays root
            // Other root becomes a child of self root
            let self_child = self_root.borrow().child.clone();
            other_root.borrow_mut().sibling = self_child.clone();
            other_root.borrow_mut().prev = Some(Rc::downgrade(&self_root));
            if let Some(ref child) = self_child {
                child.borrow_mut().prev = Some(Rc::downgrade(&other_root));
            }
            self_root.borrow_mut().child = Some(other_root);
            self.root = Some(self_root);
        }

        // Update length (other is automatically empty after take())
        self.len += other.len;
        other.len = 0;
    }
}

impl<T, P: Ord> PairingHeap<T, P> {
    /// Verifies heap property: all children have priority >= parent
    /// Returns the total count of nodes for length verification
    #[cfg(feature = "expensive_verify")]
    fn verify_heap_property(&self) -> usize {
        if let Some(ref root) = self.root {
            self.count_and_verify_subtree(root, None)
        } else {
            0
        }
    }

    #[cfg(feature = "expensive_verify")]
    #[allow(clippy::only_used_in_recursion)]
    fn count_and_verify_subtree(&self, node: &NodeRef<T, P>, parent_priority: Option<&P>) -> usize {
        // Verify heap property: node priority >= parent priority
        if let Some(parent_p) = parent_priority {
            let node_priority = &node.borrow().priority;
            assert!(
                node_priority >= parent_p,
                "Heap property violated: child priority < parent priority"
            );
        }

        let mut count = 1;

        // Collect children first to avoid borrow conflicts
        let mut children = Vec::new();
        {
            let node_ref = node.borrow();
            let mut child_opt = node_ref.child.clone();
            while let Some(child) = child_opt {
                children.push(child.clone());
                child_opt = child.borrow().sibling.clone();
            }
        }

        // Count all children recursively
        for child in children {
            // Use unsafe to get a stable reference to parent's priority
            let node_priority = unsafe { &*(&node.borrow().priority as *const P) };
            count += self.count_and_verify_subtree(&child, Some(node_priority));
        }

        count
    }

    /// Merges pairs of trees in a two-pass pairing operation
    ///
    /// This is the **critical operation** that achieves O(log n) amortized delete_min.
    /// The two-pass strategy ensures the tree remains balanced in the amortized sense.
    ///
    /// **First Pass (Pairing)**:
    /// - Walk through the child list from left to right
    /// - Pair adjacent children: (child1, child2), (child3, child4), ...
    /// - Merge each pair: smaller-priority becomes parent
    /// - Result: approximately n/2 merged trees
    ///
    /// **Second Pass (Right-to-Left Merge)**:
    /// - Start with the last merged tree
    /// - Repeatedly merge it with the next tree from right to left
    /// - Each merge makes the smaller-priority tree the root
    /// - Result: single merged tree
    ///
    /// **Why Two Passes?**
    /// - Single left-to-right pass would create an unbalanced structure (O(n) height)
    /// - Two passes ensure logarithmic height in the amortized sense
    /// - Right-to-left merge in second pass balances better than left-to-right
    ///
    /// **Time Complexity**: O(log n) amortized
    /// - First pass: O(n) but reduces n children to n/2 trees
    /// - Second pass: O(log n) merges
    /// - Amortized analysis shows total is O(log n)
    fn merge_pairs(&self, first: NodeRef<T, P>) -> NodeRef<T, P> {
        // Base case: only one child, no pairing needed
        if first.borrow().sibling.is_none() {
            return first;
        }

        // Count initial nodes (debug only)
        #[cfg(feature = "expensive_verify")]
        let initial_count = {
            let mut count = 0;
            let mut curr = Some(first.clone());
            while let Some(n) = curr {
                count += Self::count_subtree(&n);
                curr = n.borrow().sibling.clone();
            }
            count
        };

        // First pass: pair adjacent children and merge each pair
        // This reduces the number of trees by approximately half
        let mut pairs: Vec<NodeRef<T, P>> = Vec::new();
        let mut current = Some(first);

        while let Some(node) = current {
            // Get and disconnect sibling before modifying node
            let sibling = node.borrow_mut().sibling.take();
            node.borrow_mut().prev = None;

            if let Some(sib) = sibling {
                // We have a pair: node and its sibling
                // Get the next node before we disconnect
                let next = sib.borrow_mut().sibling.take();
                sib.borrow_mut().prev = None;

                // Merge the pair: smaller priority becomes parent
                // This maintains heap property
                pairs.push(self.merge_nodes(node, sib));
                current = next;
            } else {
                // Odd number of children: last one stands alone
                pairs.push(node);
                current = None;
            }
        }

        // Verify no nodes lost during first pass
        #[cfg(feature = "expensive_verify")]
        {
            let after_first_pass: usize = pairs.iter().map(|p| Self::count_subtree(p)).sum();
            assert_eq!(
                initial_count, after_first_pass,
                "Nodes lost during first pass: initial={} after={}",
                initial_count, after_first_pass
            );
        }

        // Second pass: merge pairs from right to left
        // Starting with the last merged tree, merge each remaining pair into it
        // This right-to-left ordering is crucial for balanced structure
        let mut result = pairs.pop().unwrap();
        while let Some(pair) = pairs.pop() {
            // Merge: smaller priority becomes parent (heap property)
            result = self.merge_nodes(pair, result);
        }

        // Verify no nodes lost during second pass
        #[cfg(feature = "expensive_verify")]
        {
            let final_count = Self::count_subtree(&result);
            assert_eq!(
                initial_count, final_count,
                "Nodes lost during second pass: initial={} final={}",
                initial_count, final_count
            );
        }

        result
    }

    #[cfg(feature = "expensive_verify")]
    fn count_subtree(node: &NodeRef<T, P>) -> usize {
        let node_ref = node.borrow();
        let mut count = 1;

        // Count children
        let mut child_opt = node_ref.child.clone();
        while let Some(child) = child_opt {
            count += Self::count_subtree(&child);
            child_opt = child.borrow().sibling.clone();
        }

        count
    }

    /// Merges two nodes, returning the one with smaller priority
    ///
    /// **Time Complexity**: O(1)
    ///
    /// **Algorithm**:
    /// - Compare priorities of the two nodes
    /// - Make the larger-priority node a child of the smaller-priority node
    /// - This maintains the heap property: parent <= child
    ///
    /// **Invariant**: After merge, the returned node is the root of a heap-ordered tree
    /// containing both original nodes and their descendants.
    fn merge_nodes(&self, a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        if a.borrow().priority < b.borrow().priority {
            // Node a has smaller priority: it becomes parent
            // Node b becomes a child of a
            let a_child = a.borrow().child.clone();
            // Make b the first child of a (b's sibling is a's old first child)
            b.borrow_mut().sibling = a_child.clone();
            b.borrow_mut().prev = Some(Rc::downgrade(&a)); // b's parent is a
            if let Some(ref child) = a_child {
                // Update a's old first child to point back to b
                child.borrow_mut().prev = Some(Rc::downgrade(&b));
            }
            a.borrow_mut().child = Some(b); // a's first child is now b
            a // Return a as the new root
        } else {
            // Node b has smaller or equal priority: it becomes parent
            // Node a becomes a child of b
            let b_child = b.borrow().child.clone();
            // Make a the first child of b (a's sibling is b's old first child)
            a.borrow_mut().sibling = b_child.clone();
            a.borrow_mut().prev = Some(Rc::downgrade(&b)); // a's parent is b
            if let Some(ref child) = b_child {
                // Update b's old first child to point back to a
                child.borrow_mut().prev = Some(Rc::downgrade(&a));
            }
            b.borrow_mut().child = Some(a); // b's first child is now a
            b // Return b as the new root
        }
    }

    /// Cuts a node from its parent, removing it from the child list
    ///
    /// **Time Complexity**: O(1)
    ///
    /// **Algorithm**:
    /// - Remove the node from its parent's child list
    /// - Update sibling pointers to maintain list structure
    /// - Clear the node's parent and sibling pointers
    ///
    /// **Used in**: decrease_key operation to remove a node that violates heap property
    ///
    /// The `prev` pointer can point to either:
    /// - The parent node (if this is the first child)
    /// - The previous sibling (if this is not the first child)
    ///
    /// We check which case by seeing if `prev.child == node`:
    /// - If true, node is first child (prev is parent)
    /// - If false, node is not first child (prev is previous sibling)
    fn cut_node(&mut self, node: &NodeRef<T, P>) {
        // Get the prev reference (parent or previous sibling)
        let prev_weak = match node.borrow().prev.clone() {
            Some(p) => p,
            None => return, // Node has no parent (already root or orphaned)
        };

        let prev = match prev_weak.upgrade() {
            Some(p) => p,
            None => return, // Prev node was already dropped
        };

        // Get node's sibling before we modify things
        let node_sibling = node.borrow().sibling.clone();

        // Determine if prev is the parent or a sibling
        // Check: if prev's first child is this node, then prev is the parent
        let is_first_child = prev
            .borrow()
            .child
            .as_ref()
            .map(|c| Rc::ptr_eq(c, node))
            .unwrap_or(false);

        if is_first_child {
            // Case 1: Node is the first child (prev is the parent)
            // Remove node from parent's child list
            // Parent's first child becomes node's sibling
            prev.borrow_mut().child = node_sibling.clone();
            if let Some(ref sibling) = node_sibling {
                // Update sibling's prev pointer (now points to parent)
                sibling.borrow_mut().prev = Some(Rc::downgrade(&prev));
            }
        } else {
            // Case 2: Node is not the first child (prev is previous sibling)
            // Remove node from sibling list
            // Previous sibling's next sibling becomes node's sibling
            prev.borrow_mut().sibling = node_sibling.clone();
            if let Some(ref sibling) = node_sibling {
                // Update next sibling's prev pointer (now points to previous sibling)
                sibling.borrow_mut().prev = Some(Rc::downgrade(&prev));
            }
        }

        // Clear node's links (it's now disconnected from the tree)
        node.borrow_mut().sibling = None;
        node.borrow_mut().prev = None;
    }

    // Note: No free_node function needed - Rc handles cleanup automatically.
    // When a node's Rc refcount reaches zero, it and all its children are dropped.
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
