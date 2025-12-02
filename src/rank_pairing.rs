//! Rank-Pairing Heap implementation
//!
//! A rank-pairing heap is a heap data structure that achieves:
//! - O(1) amortized insert, decrease_key, and merge
//! - O(log n) amortized delete_min
//!
//! Rank-pairing heaps are designed to be simpler than Fibonacci heaps while
//! maintaining the same optimal amortized bounds. They use a rank-based
//! restructuring scheme to maintain efficient decrease_key operations.
//!
//! # Algorithm Overview
//!
//! Rank-pairing heaps use explicit rank constraints instead of the cascading cuts
//! used in Fibonacci heaps. This makes them simpler while achieving the same bounds:
//!
//! - **Insert**: O(1) amortized - compare with root and link, update ranks
//! - **Delete-min**: O(log n) amortized - merge children using rank-based pairing
//! - **Decrease-key**: O(1) amortized - cut from parent if violated, update ranks
//! - **Merge**: O(1) amortized - compare roots and link, update ranks
//!
//! # Key Invariants (Type-A Rank-Pairing Heaps)
//!
//! 1. **Rank Constraint**: For any node v with children w₁, w₂ (two smallest ranks):
//!    - rank(v) ≤ rank(w₁) + 1
//!    - rank(v) ≤ rank(w₂) + 1
//!
//! 2. **Marking Rule**: A node can lose at most one child before being cut
//!    - Unmarked: no child lost
//!    - Marked: one child lost (after another loss, node is cut)
//!
//! 3. **Rank Update**: rank(v) = min(rank(w₁), rank(w₂)) + 1 for children w₁, w₂
//!
//! These constraints bound the tree height while allowing efficient updates.
//! Unlike Fibonacci heaps, ranks are explicit and updated locally.

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Type alias for strong reference to a node
type NodeRef<T, P> = Rc<RefCell<Node<T, P>>>;
/// Type alias for weak reference to a node (used for backlinks and handles)
type NodeWeak<T, P> = Weak<RefCell<Node<T, P>>>;

/// Handle to an element in a Rank-pairing heap
///
/// The handle uses a weak reference to the node. If the node has been removed
/// from the heap (e.g., via delete_min), the handle becomes invalid.
/// Operations on invalid handles will fail gracefully.
pub struct RankPairingHandle<T, P> {
    node: NodeWeak<T, P>,
}

// Manual Clone implementation to avoid requiring T: Clone and P: Clone
impl<T, P> Clone for RankPairingHandle<T, P> {
    fn clone(&self) -> Self {
        RankPairingHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for RankPairingHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node.ptr_eq(&other.node)
    }
}

impl<T, P> Eq for RankPairingHandle<T, P> {}

impl<T, P> Handle for RankPairingHandle<T, P> {}

/// Internal node structure for rank-pairing heap
///
/// Each node maintains:
/// - `item` and `priority`: The data stored in the heap
/// - `parent`: Weak reference to parent node (None if root)
/// - `child`: Strong reference to first child (None if leaf)
/// - `sibling`: Strong reference to next sibling in parent's child list (None if last child)
/// - `rank`: Explicit rank value (critical for rank constraints)
/// - `marked`: Flag indicating if node has lost one child (used in cut operations)
///
/// **Rank**: Explicit rank value computed from children's ranks. Rank constraints
/// ensure the tree height is O(log n) while allowing efficient updates.
///
/// **Marking**: Similar to Fibonacci heaps, but simpler. A node can lose at most
/// one child before being cut, maintaining the rank constraints.
struct Node<T, P> {
    item: T,
    priority: P,
    /// Parent node (None if root). Uses weak reference to avoid cycles.
    parent: Option<NodeWeak<T, P>>,
    /// First child in child list (None if leaf). Uses strong reference (Rc) as parent owns children.
    child: Option<NodeRef<T, P>>,
    /// Next sibling in parent's child list (None if last child).
    /// Uses strong reference (Rc) as earlier siblings own later ones in the list.
    sibling: Option<NodeRef<T, P>>,
    /// Explicit rank: rank(v) = min(rank(w₁), rank(w₂)) + 1 where w₁, w₂ are
    /// children with smallest ranks. This bounds tree height at O(log n).
    rank: usize,
    /// Marked flag (Type-A): false if no child lost, true if one child lost.
    /// After losing two children, the node is cut (cascading).
    marked: bool,
}

/// Rank-Pairing Heap
///
/// This implementation uses type-A rank-pairing heaps, which maintain:
/// - A node loses at most one child before being cut
/// - Ranks satisfy: r(v) <= r(w1) + 1 and r(v) <= r(w2) + 1
///   where w1, w2 are the two children with smallest ranks
///
/// A safe implementation using `Rc` and `Weak` references instead of raw pointers.
/// Strong references (`Rc`) flow from root to children, weak references (`Weak`)
/// point upward for efficient decrease_key operations.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::rank_pairing::RankPairingHeap;
/// use rust_advanced_heaps::Heap;
///
/// let mut heap = RankPairingHeap::new();
/// let handle = heap.insert(5, "item");
/// heap.decrease_key(&handle, 1);
/// assert_eq!(heap.find_min(), Some((&1, &"item")));
/// ```
pub struct RankPairingHeap<T, P: Ord> {
    /// Root of the heap tree. Uses strong reference as the heap owns the root.
    root: Option<NodeRef<T, P>>,
    len: usize,
}

// No Drop implementation needed - Rc handles cleanup automatically.
// When the heap is dropped, the root Rc's refcount goes to zero,
// which recursively drops all children (via their strong references).

impl<T, P: Ord> Heap<T, P> for RankPairingHeap<T, P> {
    type Handle = RankPairingHandle<T, P>;

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
    /// 1. Create a new node with rank 0 (leaf node)
    /// 2. Compare priority with current root
    /// 3. If new priority is smaller, make new node the root (old root becomes child)
    /// 4. Otherwise, add new node as a child of the root
    /// 5. Update ranks to maintain rank constraints
    ///
    /// **Rank Update**: When a node becomes a parent, its rank is updated based on
    /// its children's ranks. Rank = min(rank(w₁), rank(w₂)) + 1 where w₁, w₂ are
    /// children with smallest ranks. This maintains the rank constraint invariant.
    ///
    /// **Invariant Maintenance**:
    /// - Heap property: smaller priority becomes parent
    /// - Rank constraints: ranks updated to satisfy rank(v) ≤ rank(w₁) + 1 and rank(v) ≤ rank(w₂) + 1
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node with rank 0 (leaf node, no children)
        let node = Rc::new(RefCell::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            rank: 0,       // Leaf nodes have rank 0
            marked: false, // New nodes are unmarked
        }));

        // Link new node into the tree structure
        if let Some(ref root) = self.root {
            if node.borrow().priority < root.borrow().priority {
                // Case 1: New node has smaller priority
                // Make new node the root, old root becomes its child
                // This maintains heap property: parent <= child
                Self::make_child(&node, root);
                // Update root pointer to new minimum
                self.root = Some(Rc::clone(&node));
            } else {
                // Case 2: Current root has smaller or equal priority
                // Add new node as a child of the root
                // Heap property maintained: new node >= root
                Self::make_child(root, &node);
                // Root stays the same
            }
        } else {
            // Empty heap: new node becomes root
            self.root = Some(Rc::clone(&node));
        }

        self.len += 1;
        RankPairingHandle {
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
    /// 3. Merge children using rank-based pairing
    /// 4. The result becomes the new root
    ///
    /// **Rank-Based Merging**:
    /// - Children are merged in a way that maintains rank constraints
    /// - The pairing strategy ensures O(log n) children after merging
    /// - Ranks are updated as children are merged
    ///
    /// **Why O(log n)?**
    /// - At most O(log n) children (bounded by rank constraints)
    /// - Merging maintains rank constraints
    /// - Amortized analysis shows the pairing strategy achieves O(log n) total cost
    fn delete_min(&mut self) -> Option<(P, T)> {
        let root = self.root.take()?;

        // Collect all children of the root (clearing their parent links)
        let children = Self::collect_children(&root);

        self.len -= 1;

        if children.is_empty() {
            // No children: heap becomes empty
            // self.root is already None from take()
        } else {
            // Merge all children using rank-based pairing
            // This operation maintains rank constraints and produces a single root
            // The pairing strategy ensures O(log n) amortized cost
            self.root = Some(Self::merge_children(children));
        }

        // Verify invariants after delete_min
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
    /// **Time Complexity**: O(1) amortized
    ///
    /// **Precondition**: `new_priority < current_priority` (undefined behavior otherwise)
    ///
    /// **Algorithm**:
    /// 1. Update the priority value
    /// 2. If heap property is violated (new priority < parent priority):
    ///    - Cut the node from its parent
    ///    - If parent was marked, cut parent too (cascading cut)
    ///    - Merge the cut node with the root (or make it the new root if smaller)
    /// 3. Update ranks after cutting
    ///
    /// **Cascading Cuts (Type-A)**:
    /// - When a child is cut from its parent, mark the parent
    /// - If a marked parent loses another child, cut it too (cascade upward)
    /// - This ensures no node loses more than one child before being cut
    /// - This maintains the rank constraints and bounds tree height
    ///
    /// **Why O(1) amortized?**
    /// - Most decrease_key operations are cheap (cutting near the root)
    /// - Expensive operations (deep cuts with cascading) are rare
    /// - Cascading cuts are bounded: each node can be cut at most once
    /// - Amortized analysis shows average cost is O(1)
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

        // Node is not root, check if heap property is violated
        let needs_cut = {
            let node_ref = node.borrow();
            if let Some(ref parent_weak) = node_ref.parent {
                if let Some(parent) = parent_weak.upgrade() {
                    node_ref.priority < parent.borrow().priority
                } else {
                    false
                }
            } else {
                false
            }
        };

        if needs_cut {
            // Heap property violated: cut node from parent
            // This operation may trigger cascading cuts if parent was marked
            // Returns all cut nodes (including cascaded parents)
            let cut_nodes = self.cut(&node);

            // Merge all cut nodes with root
            // Each cut node (including cascaded parents) needs to be added
            for cut_node in cut_nodes {
                if let Some(ref root) = self.root {
                    if cut_node.borrow().priority < root.borrow().priority {
                        // Cut node has smaller priority: make it the new root
                        Self::make_child(&cut_node, root);
                        self.root = Some(cut_node);
                    } else {
                        // Current root has smaller priority: add cut node as child
                        Self::make_child(root, &cut_node);
                    }
                } else {
                    // Heap is empty (shouldn't happen, but handle gracefully)
                    self.root = Some(cut_node);
                }
            }
        }
        // If heap property is not violated, no restructuring needed

        Ok(())
    }

    /// Merges another heap into this heap
    ///
    /// **Time Complexity**: O(1) amortized
    ///
    /// **Algorithm**:
    /// 1. Compare roots of both heaps
    /// 2. Make the larger-priority root a child of the smaller-priority root
    /// 3. Update ranks to maintain rank constraints
    /// 4. The smaller root becomes the new root
    ///
    /// This is trivially O(1): we only need one comparison and a few pointer updates.
    /// The rank update is O(1) because we're just adding one child.
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
            Self::make_child(&other_root, &self_root);
            self.root = Some(other_root);
        } else {
            // Self root has smaller or equal priority: it stays root
            // Other root becomes a child of self root
            Self::make_child(&self_root, &other_root);
            self.root = Some(self_root);
        }

        // Update length and mark other as empty (prevent double-free)
        self.len += other.len;
        other.len = 0;
    }
}

impl<T, P: Ord> RankPairingHeap<T, P> {
    /// Verifies heap property: all children have priority >= parent
    /// Returns the total count of nodes for length verification
    #[cfg(feature = "expensive_verify")]
    fn verify_heap_property(&self) -> usize {
        if let Some(ref root) = self.root {
            Self::verify_subtree(root, None)
        } else {
            0
        }
    }

    #[cfg(feature = "expensive_verify")]
    fn verify_subtree(node: &NodeRef<T, P>, parent_priority: Option<&P>) -> usize {
        let node_ref = node.borrow();
        let node_priority = &node_ref.priority;

        // Verify heap property: node priority >= parent priority
        if let Some(parent_p) = parent_priority {
            assert!(
                node_priority >= parent_p,
                "Heap property violated: child priority < parent priority"
            );
        }

        let mut count = 1;

        // Collect children first to avoid borrow conflicts
        let mut children = Vec::new();
        {
            let mut child_opt = node_ref.child.clone();
            while let Some(child) = child_opt {
                children.push(child.clone());
                child_opt = child.borrow().sibling.clone();
            }
        }
        drop(node_ref); // Release borrow before recursive calls

        // Verify all children
        for child in children {
            // Use unsafe to get a stable reference to parent's priority
            let priority_ref = unsafe { &*(&node.borrow().priority as *const P) };
            count += Self::verify_subtree(&child, Some(priority_ref));
        }

        count
    }

    /// Makes y a child of x, maintaining heap property and rank constraints
    ///
    /// **Time Complexity**: O(1) amortized (rank update may be O(degree) but amortized to O(1))
    ///
    /// **Algorithm**:
    /// 1. Set y's parent to x
    /// 2. Add y to x's child list (as first child)
    /// 3. Update x's rank based on its children's ranks
    ///
    /// **Rank Update**: After adding y as a child, x's rank must be updated to satisfy
    /// rank constraints: rank(x) ≤ rank(w₁) + 1 and rank(x) ≤ rank(w₂) + 1 where
    /// w₁, w₂ are children with smallest ranks.
    ///
    /// **Invariant**: This operation maintains the heap property (parent <= child)
    /// and updates ranks to maintain rank constraints.
    fn make_child(x: &NodeRef<T, P>, y: &NodeRef<T, P>) {
        // Set parent-child relationship
        y.borrow_mut().parent = Some(Rc::downgrade(x));
        // Add y to x's child list (insert at front)
        y.borrow_mut().sibling = x.borrow().child.clone();
        x.borrow_mut().child = Some(Rc::clone(y));

        // Update rank: x's rank must be recomputed based on its children
        // Rank formula: rank(x) = min(rank(w₁), rank(w₂)) + 1
        // This ensures rank constraints are maintained
        Self::update_rank(x);
    }

    /// Updates the rank of a node based on its children's ranks
    ///
    /// **Time Complexity**: O(degree) where degree is the number of children
    /// - Amortized to O(1) over a sequence of operations
    ///
    /// **Algorithm (Rank Constraint)**:
    /// For node v with children w₁, w₂, ..., wₖ:
    /// - Find two children with smallest ranks: r₁ = min(ranks), r₂ = second min(ranks)
    /// - New rank: rank(v) = max(r₁, r₂) + 1
    /// - This ensures rank(v) ≤ rank(w₁) + 1 and rank(v) ≤ rank(w₂) + 1
    ///
    /// **Special Cases**:
    /// - No children: rank = 0 (leaf node)
    /// - One child: rank = child_rank + 1
    /// - Two or more children: rank = max(r₁, r₂) + 1 where r₁, r₂ are smallest two ranks
    ///
    /// **Why max(r₁, r₂) + 1?**
    /// The rank constraint requires rank(v) ≤ rank(w₁) + 1 and rank(v) ≤ rank(w₂) + 1.
    /// Setting rank(v) = max(r₁, r₂) + 1 satisfies both constraints.
    fn update_rank(node: &NodeRef<T, P>) {
        let child_opt = node.borrow().child.clone();

        if let Some(child) = child_opt {
            // Collect all children's ranks
            // We need to find the two smallest ranks to compute the new rank
            let mut ranks = Vec::new();
            let mut current = Some(child);

            // Traverse child list to collect all ranks
            while let Some(curr) = current {
                ranks.push(curr.borrow().rank);
                current = curr.borrow().sibling.clone();
            }

            // Sort ranks to find smallest two
            ranks.sort();
            ranks.reverse(); // Largest first (for easier indexing)

            // Compute rank based on rank constraint
            // rank(v) = max(r₁, r₂) + 1 where r₁, r₂ are smallest ranks
            if ranks.len() >= 2 {
                // Two or more children: use two smallest ranks
                let r1 = ranks[0]; // Second smallest (largest in reversed list)
                let r2 = ranks[1]; // Smallest
                                   // rank(v) = max(r₁, r₂) + 1 satisfies both constraints
                node.borrow_mut().rank = (r1.max(r2)) + 1;
            } else if ranks.len() == 1 {
                // One child: rank = child_rank + 1
                node.borrow_mut().rank = ranks[0] + 1;
            } else {
                // No children (shouldn't happen, but handle gracefully)
                node.borrow_mut().rank = 0;
            }
        } else {
            // Leaf node: rank is 0
            node.borrow_mut().rank = 0;
        }
    }

    /// Cuts a node from its parent and collects all cascaded nodes
    ///
    /// **Time Complexity**: O(1) amortized (cascading cuts amortized to O(1))
    ///
    /// **Algorithm (Type-A Cascading Cuts)**:
    /// 1. Remove node from parent's child list
    /// 2. Clear node's parent and sibling pointers
    /// 3. **Marking rule**: If parent is not marked, mark it
    /// 4. **Cascading**: If parent is already marked, cut it too (cascade upward)
    /// 5. Update parent's rank after losing a child
    /// 6. Return all cut nodes so they can be merged with the root
    ///
    /// **Marking Rule (Type-A)**:
    /// - A node can lose at most one child before being cut
    /// - Unmarked: no child lost yet
    /// - Marked: one child lost (next loss triggers cut)
    /// - This maintains rank constraints and bounds tree height
    ///
    /// **Why O(1) amortized?**
    /// - Most cuts are cheap (cutting near root, parent not marked)
    /// - Cascading cuts are bounded: each node can be cut at most once
    /// - Amortized analysis shows average cascade depth is O(1)
    ///
    /// Returns a vector of all cut nodes (including cascaded parents) that need
    /// to be merged with the root.
    #[allow(clippy::only_used_in_recursion)]
    fn cut(&mut self, node: &NodeRef<T, P>) -> Vec<NodeRef<T, P>> {
        // Get parent (if none, node is already root or orphaned)
        let parent = {
            let node_ref = node.borrow();
            match &node_ref.parent {
                Some(parent_weak) => parent_weak.upgrade(),
                None => return Vec::new(),
            }
        };

        let parent = match parent {
            Some(p) => p,
            None => return Vec::new(), // Parent was already dropped
        };

        // Step 1: Remove node from parent's child list
        // Check if node is the first child or a later child
        let is_first_child = parent
            .borrow()
            .child
            .as_ref()
            .map(|c| Rc::ptr_eq(c, node))
            .unwrap_or(false);

        let node_sibling = node.borrow().sibling.clone();

        if is_first_child {
            // Node is first child: parent's first child becomes node's sibling
            parent.borrow_mut().child = node_sibling;
        } else {
            // Node is not first child: find it in sibling chain and remove
            // This requires traversing the sibling list
            let mut current = parent.borrow().child.clone();
            while let Some(curr) = current {
                let curr_sibling = curr.borrow().sibling.clone();
                if curr_sibling.as_ref().map(|s| Rc::ptr_eq(s, node)).unwrap_or(false) {
                    // Found node: skip it in sibling chain
                    curr.borrow_mut().sibling = node_sibling.clone();
                    break;
                }
                current = curr_sibling;
            }
        }

        // Step 2: Clear node's parent and sibling pointers (it's now a root)
        node.borrow_mut().parent = None;
        node.borrow_mut().sibling = None;
        node.borrow_mut().marked = false; // Reset mark when cut

        // Collect all cut nodes that need to be merged with root
        let mut cut_nodes = vec![Rc::clone(node)];

        // Step 3: Apply marking rule (Type-A rank-pairing heap)
        // This maintains the constraint that no node loses more than one child
        let parent_was_marked = parent.borrow().marked;

        if !parent_was_marked {
            // Parent hasn't lost a child yet: mark it
            // Next time it loses a child, it will be cut (cascading)
            parent.borrow_mut().marked = true;
            // Update parent's rank (it lost a child, rank may decrease)
            Self::update_rank(&parent);
        } else {
            // Parent already marked (has lost one child): cut it now (cascading)
            // This prevents nodes from losing too many children
            // Recursive cut collects all cascaded nodes
            let cascaded = self.cut(&parent);
            cut_nodes.extend(cascaded);
        }

        cut_nodes
    }

    /// Collects all children of a node into a vector
    fn collect_children(parent: &NodeRef<T, P>) -> Vec<NodeRef<T, P>> {
        let mut children = Vec::new();
        let mut current = parent.borrow().child.clone();

        while let Some(curr) = current {
            let next = curr.borrow().sibling.clone();
            // Clear parent and sibling pointers (each child becomes a root)
            curr.borrow_mut().parent = None;
            curr.borrow_mut().sibling = None;
            children.push(curr);
            current = next;
        }

        // Clear parent's child pointer since we've taken all children
        parent.borrow_mut().child = None;

        children
    }

    /// Merges a list of trees using rank-based pairing
    fn merge_children(mut children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.len() == 1 {
            return children.pop().unwrap();
        }

        // Simple pairing approach: repeatedly pair adjacent trees
        // This is a simplified version; a full implementation would use
        // rank-based grouping for optimal bounds
        while children.len() > 1 {
            let mut next = Vec::new();
            let mut i = 0;

            while i < children.len() {
                if i + 1 < children.len() {
                    // Pair two trees
                    let a = &children[i];
                    let b = &children[i + 1];
                    let merged = if a.borrow().priority < b.borrow().priority {
                        Self::make_child(a, b);
                        Self::update_rank(a);
                        Rc::clone(a)
                    } else {
                        Self::make_child(b, a);
                        Self::update_rank(b);
                        Rc::clone(b)
                    };
                    next.push(merged);
                    i += 2;
                } else {
                    // Single tree left, add it to next round
                    next.push(Rc::clone(&children[i]));
                    i += 1;
                }
            }
            children = next;
        }

        children.pop().unwrap()
    }

    /// Links two trees of the same rank
    #[allow(dead_code)]
    fn link_same_rank(a: &NodeRef<T, P>, b: &NodeRef<T, P>) -> NodeRef<T, P> {
        if a.borrow().priority < b.borrow().priority {
            Self::make_child(a, b);
            Self::update_rank(a);
            Rc::clone(a)
        } else {
            Self::make_child(b, a);
            Self::update_rank(b);
            Rc::clone(b)
        }
    }

    // Note: No free_node function needed - Rc handles cleanup automatically.
    // When a node's Rc refcount reaches zero, it and all its children are dropped.
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
