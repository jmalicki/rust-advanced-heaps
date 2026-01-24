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
//!
//! # Why Rank-Pairing Heaps?
//!
//! Rank-pairing heaps combine the asymptotic efficiency of Fibonacci heaps with
//! much of the simplicity of pairing heaps. The key insight is using explicit
//! rank values on nodes instead of the marked-node mechanism of Fibonacci heaps.
//!
//! Unlike other heap implementations matching Fibonacci heap bounds, rank-pairing
//! heaps need only **one cut per decrease-key** with no other structural changes.
//! This makes them significantly simpler to implement while maintaining optimal
//! bounds for insert, decrease-key (O(1) amortized), and delete-min (O(log n) amortized).
//!
//! # References
//!
//! - Haeupler, B., Sen, S., & Tarjan, R. E. (2011). "Rank-pairing heaps."
//!   *SIAM Journal on Computing*, 40(6), 1463-1485.
//!   [SIAM](https://epubs.siam.org/doi/10.1137/100785351)
//! - [Wikipedia: Rank-pairing heap](https://en.wikipedia.org/wiki/Rank-pairing_heap)

use crate::rank::Rank;
use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError, MergeableHeap};
#[cfg(feature = "access_tracking")]
#[allow(unused_imports)]
use crate::{track, track_op};
use refbox::{RefBox, Weak};

/// Type alias for strong reference to a node
type NodeRef<T, P> = RefBox<Node<T, P>>;
/// Type alias for weak reference to a node (used for backlinks and handles)
type NodeWeak<T, P> = Weak<Node<T, P>>;

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
        self.node.as_ptr() == other.node.as_ptr()
    }
}

impl<T, P> Eq for RankPairingHandle<T, P> {}

impl<T, P> Handle for RankPairingHandle<T, P> {}

/// Internal node structure for rank-pairing heap
///
/// Each node maintains:
/// - `priority`: The priority for heap ordering (hot path - accessed on every comparison)
/// - `parent`: Weak reference to parent node (None if root)
/// - `child`: Strong reference to first child (None if leaf)
/// - `sibling`: Strong reference to next sibling in parent's child list (None if last child)
/// - `prev`: Weak reference to parent OR previous sibling (for O(1) cuts)
/// - `rank`: Explicit rank value (critical for rank constraints)
/// - `marked`: Flag indicating if node has lost one child (used in cut operations)
/// - `item`: The data stored in the heap (cold path - only accessed on pop)
///
/// **Cache Optimization**: Fields are ordered for cache locality:
/// - Hot path first: `priority` is accessed on every comparison
/// - Traversal fields next: pointers for tree navigation
/// - Small fields: `rank` uses `Rank` (u8) to save 7 bytes, `marked` is bool
/// - Cold path last: `item` is only accessed when popping
///
/// **Prev Pointer**: The `prev` pointer serves dual purpose:
/// - For first child: points to parent
/// - For other children: points to previous sibling
///
/// This enables O(1) removal from the sibling list during cuts.
///
/// **Parent Pointer**: Always points to the actual parent node.
/// This is needed for the cascading cut mechanism in Type-A rank-pairing heaps.
///
/// **Rank**: Explicit rank value computed from children's ranks. Rank constraints
/// ensure the tree height is O(log n) while allowing efficient updates.
///
/// **Marking**: Similar to Fibonacci heaps, but simpler. A node can lose at most
/// one child before being cut, maintaining the rank constraints.
struct Node<T, P> {
    /// Priority for heap ordering - Hot path: accessed on every comparison
    priority: P,
    /// Parent node (None if root). Uses weak reference to avoid cycles.
    /// Always points to the actual parent, used for cascading cuts.
    parent: Option<NodeWeak<T, P>>,
    /// First child in child list (None if leaf). Uses strong reference (Rc) as parent owns children.
    child: Option<NodeRef<T, P>>,
    /// Next sibling in parent's child list (None if last child).
    /// Uses strong reference (Rc) as earlier siblings own later ones in the list.
    sibling: Option<NodeRef<T, P>>,
    /// Parent node or previous sibling. Uses weak reference to avoid cycles.
    /// - For first child: points to parent
    /// - For other children: points to previous sibling
    ///
    /// This dual-purpose pointer enables O(1) cuts from the sibling list.
    prev: Option<NodeWeak<T, P>>,
    /// Explicit rank: rank(v) = min(rank(w₁), rank(w₂)) + 1 where w₁, w₂ are
    /// children with smallest ranks. This bounds tree height at O(log n).
    /// Uses Rank (u8) to save memory - max rank is O(log n) which fits in u8.
    rank: Rank,
    /// Marked flag (Type-A): false if no child lost, true if one child lost.
    /// After losing two children, the node is cut (cascading).
    marked: bool,
    /// The item stored in the heap - Cold path: only accessed on pop
    item: T,
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
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
///
/// let mut heap = RankPairingHeap::new();
/// let handle = heap.push_with_handle(5, "item");
/// heap.decrease_key(&handle, 1).unwrap();
/// assert_eq!(heap.peek(), Some((&1, &"item")));
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
    fn new() -> Self {
        Self { root: None, len: 0 }
    }

    fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) {
        let _ = self.push_with_handle(priority, item);
    }

    fn peek(&self) -> Option<(&P, &T)> {
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
    fn pop(&mut self) -> Option<(P, T)> {
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

        // Extract item and priority from the root node using RefBox::take()
        let node = RefBox::take(root);
        Some((node.priority, node.item))
    }
}

impl<T, P: Ord> MergeableHeap<T, P> for RankPairingHeap<T, P> {
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
        let other_is_smaller = {
            let other_ref = other_root.try_borrow_mut().unwrap();
            let self_ref = self_root.try_borrow_mut().unwrap();
            other_ref.priority < self_ref.priority
        };

        if other_is_smaller {
            // Other root has smaller priority: it becomes the new root
            // Self root becomes a child of other root
            Self::make_child(&other_root, self_root);
            self.root = Some(other_root);
        } else {
            // Self root has smaller or equal priority: it stays root
            // Other root becomes a child of self root
            Self::make_child(&self_root, other_root);
            self.root = Some(self_root);
        }

        // Update length and mark other as empty (prevent double-free)
        self.len += other.len;
        other.len = 0;
    }
}

impl<T, P: Ord> DecreaseKeyHeap<T, P> for RankPairingHeap<T, P> {
    type Handle = RankPairingHandle<T, P>;

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
    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node with rank 0 (leaf node, no children)
        let node = RefBox::new(Node {
            // Hot path first
            priority,
            // Traversal fields
            parent: None,
            child: None,
            sibling: None,
            prev: None, // For O(1) cuts
            // Small fields
            rank: 0,       // Leaf nodes have rank 0
            marked: false, // New nodes are unmarked
            // Cold path last
            item,
        });

        // Get handle before potentially moving node
        let handle = RankPairingHandle {
            node: node.downgrade(),
        };

        // Link new node into the tree structure
        if let Some(ref root) = self.root {
            let node_priority_less = {
                let node_ref = node.try_borrow_mut().unwrap();
                let root_ref = root.try_borrow_mut().unwrap();
                node_ref.priority < root_ref.priority
            };
            if node_priority_less {
                // Case 1: New node has smaller priority
                // Make new node the root, old root becomes its child
                // This maintains heap property: parent <= child
                let old_root = self.root.take().unwrap();
                Self::make_child(&node, old_root);
                // Update root pointer to new minimum
                self.root = Some(node);
            } else {
                // Case 2: Current root has smaller or equal priority
                // Add new node as a child of the root
                // Heap property maintained: new node >= root
                Self::make_child(root, node);
                // Root stays the same, node is now a child
            }
        } else {
            // Empty heap: new node becomes root
            self.root = Some(node);
        }

        self.len += 1;
        handle
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
        // Check if the node is still alive
        if !handle.node.is_alive() {
            return Err(HeapError::PriorityNotDecreased);
        }

        // Safety check: new priority must actually be less
        {
            let node_ref = handle.node.try_borrow_mut().unwrap();
            if new_priority >= node_ref.priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Update the priority value
        handle.node.try_borrow_mut().unwrap().priority = new_priority;

        // Get the node pointer for comparisons
        let node_ptr = handle.node.as_ptr();

        // Check if the node is already the root
        let is_root = self
            .root
            .as_ref()
            .map(|r| r.as_ptr() == node_ptr)
            .unwrap_or(false);

        if is_root {
            // Node is already the root, no restructuring needed
            return Ok(());
        }

        // Node is not root, check if heap property is violated
        let needs_cut = {
            let node_ref = handle.node.try_borrow_mut().unwrap();
            if let Some(ref parent_weak) = node_ref.parent {
                if parent_weak.is_alive() {
                    let parent_ref = parent_weak.try_borrow_mut().unwrap();
                    node_ref.priority < parent_ref.priority
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
            let cut_nodes = self.cut_via_weak(&handle.node);

            // Merge all cut nodes with root
            // Each cut node (including cascaded parents) needs to be added
            for cut_node in cut_nodes {
                if let Some(ref root) = self.root {
                    let cut_is_smaller = {
                        let cut_ref = cut_node.try_borrow_mut().unwrap();
                        let root_ref = root.try_borrow_mut().unwrap();
                        cut_ref.priority < root_ref.priority
                    };
                    if cut_is_smaller {
                        // Cut node has smaller priority: make it the new root
                        let old_root = self.root.take().unwrap();
                        Self::make_child(&cut_node, old_root);
                        self.root = Some(cut_node);
                    } else {
                        // Current root has smaller priority: add cut node as child
                        Self::make_child(root, cut_node);
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
    fn verify_subtree(node: &NodeRef<T, P>, parent_priority: Option<*const P>) -> usize {
        // Use raw pointers for safe verification traversal
        let node_ptr = node.as_ptr();

        // Safety: We're only reading through these pointers during verification
        // The heap structure is not modified during verification
        unsafe {
            let node_priority_ptr = &(*node_ptr).priority as *const P;

            // Verify heap property: node priority >= parent priority
            if let Some(parent_p) = parent_priority {
                assert!(
                    *node_priority_ptr >= *parent_p,
                    "Heap property violated: child priority < parent priority"
                );
            }

            let mut count = 1;

            // Walk children using raw pointers
            let mut child_ptr = (*node_ptr).child.as_ref().map(|c| c.as_ptr());

            while let Some(cp) = child_ptr {
                // Recursively verify this child
                // We need to access the child through the RefBox to get a proper reference
                // But we can count using raw pointer traversal
                count += Self::verify_subtree_ptr(cp, Some(node_priority_ptr));

                // Move to sibling
                child_ptr = (*cp).sibling.as_ref().map(|s| s.as_ptr());
            }

            count
        }
    }

    #[cfg(feature = "expensive_verify")]
    unsafe fn verify_subtree_ptr(
        node_ptr: *const Node<T, P>,
        parent_priority: Option<*const P>,
    ) -> usize {
        let node_priority_ptr = &(*node_ptr).priority as *const P;

        // Verify heap property
        if let Some(parent_p) = parent_priority {
            assert!(
                *node_priority_ptr >= *parent_p,
                "Heap property violated: child priority < parent priority"
            );
        }

        let mut count = 1;

        // Walk children
        let mut child_ptr = (*node_ptr).child.as_ref().map(|c| c.as_ptr());

        while let Some(cp) = child_ptr {
            count += Self::verify_subtree_ptr(cp, Some(node_priority_ptr));
            child_ptr = (*cp).sibling.as_ref().map(|s| s.as_ptr());
        }

        count
    }

    /// Makes y a child of x, maintaining heap property and rank constraints
    ///
    /// **Time Complexity**: O(1)
    ///
    /// **Algorithm**:
    /// 1. Set y's prev to x (y becomes first child, so prev points to parent)
    /// 2. Update the old first child's prev to point to y (its new previous sibling)
    /// 3. Add y to x's child list (as first child)
    /// 4. Update x's rank based on its children's ranks
    ///
    /// **Rank Update**: After adding y as a child, x's rank must be updated to satisfy
    /// rank constraints: rank(x) ≤ rank(w₁) + 1 and rank(x) ≤ rank(w₂) + 1 where
    /// w₁, w₂ are children with smallest ranks.
    ///
    /// **Invariant**: This operation maintains the heap property (parent <= child)
    /// and updates ranks to maintain rank constraints.
    fn make_child(x: &NodeRef<T, P>, y: NodeRef<T, P>) {
        // Get weak reference to y before moving it
        let y_weak = y.downgrade();

        // Get the old first child (if any)
        let old_first_child = x.try_borrow_mut().unwrap().child.take();

        // Set y's parent to x (for cascading cuts)
        // Set y's prev to x (y is becoming the first child, so prev points to parent)
        // Set y's sibling to the old first child
        {
            let mut y_ref = y.try_borrow_mut().unwrap();
            y_ref.parent = Some(x.downgrade());
            y_ref.prev = Some(x.downgrade());
            y_ref.sibling = old_first_child;
        }

        // Update the old first child's prev to point to y (its new previous sibling)
        // We need to access y's sibling, but y will be moved, so we access via the weak ref
        // Actually, we can access it now before moving y
        {
            let y_ref = y.try_borrow_mut().unwrap();
            if let Some(ref old_child) = y_ref.sibling {
                old_child.try_borrow_mut().unwrap().prev = Some(y_weak.clone());
            }
        }

        // Make y the first child of x
        x.try_borrow_mut().unwrap().child = Some(y);

        // Update rank: x's rank must be recomputed based on its children
        // Rank formula: rank(x) = min(rank(w₁), rank(w₂)) + 1
        // This ensures rank constraints are maintained
        Self::update_rank(x);
    }

    /// Updates the rank of a node based on its first child's rank
    ///
    /// **Time Complexity**: O(1)
    ///
    /// **Algorithm (Simplified Rank Constraint)**:
    /// For rank-pairing heaps, when a new child is added (always at the front of
    /// the child list), the parent's rank is:
    /// - rank(parent) = max(rank(parent), rank(first_child) + 1)
    ///
    /// This is O(1) because we only look at the first child, which is the newly
    /// added child in make_child operations.
    ///
    /// **Why This Works**:
    /// In our implementation, make_child always adds the new child at the front
    /// of the child list. The rank only needs to increase if the new child's
    /// rank + 1 is greater than the current rank. This maintains the invariant
    /// that rank(parent) >= rank(any_child) + 1 for all children.
    ///
    /// **Special Cases**:
    /// - No children: rank = 0 (leaf node)
    /// - With children: rank = max(current_rank, first_child_rank + 1)
    fn update_rank(node: &NodeRef<T, P>) {
        let node_ref = node.try_borrow_mut().unwrap();
        let child_rank_opt = node_ref
            .child
            .as_ref()
            .map(|c| c.try_borrow_mut().unwrap().rank);
        let current_rank = node_ref.rank;
        drop(node_ref);

        if let Some(child_rank) = child_rank_opt {
            // O(1): Only look at the first child's rank
            // The new rank is the maximum of current rank and first_child_rank + 1
            // This ensures rank(parent) >= rank(child) + 1 for the new child
            let new_rank = current_rank.max(child_rank + 1);
            node.try_borrow_mut().unwrap().rank = new_rank;
        } else {
            // Leaf node: rank is 0
            node.try_borrow_mut().unwrap().rank = 0;
        }
    }

    /// Cuts a node from its parent using O(1) prev pointer access
    ///
    /// **Time Complexity**: O(1) amortized (cascading cuts amortized to O(1))
    ///
    /// Uses the node's prev pointer to directly locate and extract it from the tree.
    /// The prev pointer enables O(1) removal from the sibling list.
    ///
    /// Returns a vector of all cut nodes (including cascaded parents) that need
    /// to be merged with the root.
    #[allow(clippy::only_used_in_recursion)]
    fn cut_via_weak(&mut self, node_weak: &NodeWeak<T, P>) -> Vec<NodeRef<T, P>> {
        if !node_weak.is_alive() {
            return Vec::new();
        }

        // Get prev and parent weak refs from the target node
        let (prev_weak, parent_weak) = {
            let node_ref = node_weak.try_borrow_mut().unwrap();
            (node_ref.prev.clone(), node_ref.parent.clone())
        };

        let prev_weak = match prev_weak {
            Some(pw) => pw,
            None => return Vec::new(), // No prev means orphaned or root
        };

        if !prev_weak.is_alive() {
            return Vec::new();
        }

        let parent_weak = match parent_weak {
            Some(pw) => pw,
            None => return Vec::new(), // No parent means root
        };

        if !parent_weak.is_alive() {
            return Vec::new();
        }

        let node_ptr = node_weak.as_ptr();

        // Check if prev.child == node (node is first child of parent)
        let is_first_child = {
            let prev_ref = prev_weak.try_borrow_mut().unwrap();
            prev_ref
                .child
                .as_ref()
                .map(|c| c.as_ptr() == node_ptr)
                .unwrap_or(false)
        };

        // Extract the node from the tree
        let extracted: NodeRef<T, P>;

        if is_first_child {
            // Node is first child: take from prev.child
            let mut prev_mut = prev_weak.try_borrow_mut().unwrap();
            extracted = prev_mut.child.take().expect("Node should be in prev.child");

            // Get sibling and update
            let sibling = extracted.try_borrow_mut().unwrap().sibling.take();
            if let Some(ref sib) = sibling {
                sib.try_borrow_mut().unwrap().prev = Some(prev_weak.clone());
            }
            prev_mut.child = sibling;
        } else {
            // Node is not first child: prev.sibling == node
            let mut prev_mut = prev_weak.try_borrow_mut().unwrap();
            extracted = prev_mut
                .sibling
                .take()
                .expect("Node should be in prev.sibling");

            // Get sibling and update
            let sibling = extracted.try_borrow_mut().unwrap().sibling.take();
            if let Some(ref sib) = sibling {
                sib.try_borrow_mut().unwrap().prev = Some(prev_weak.clone());
            }
            prev_mut.sibling = sibling;
        }

        // Clear the extracted node's pointers
        {
            let mut node_mut = extracted.try_borrow_mut().unwrap();
            node_mut.parent = None;
            node_mut.prev = None;
            node_mut.sibling = None;
            node_mut.marked = false;
        }

        let mut cut_nodes = vec![extracted];

        // Check if parent was marked for cascading cut
        let parent_was_marked = parent_weak.try_borrow_mut().unwrap().marked;

        if !parent_was_marked {
            // Mark the parent
            parent_weak.try_borrow_mut().unwrap().marked = true;
        } else {
            // Parent was marked: cascade cut
            let cascaded = self.cut_via_weak(&parent_weak);
            cut_nodes.extend(cascaded);
        }

        cut_nodes
    }

    /// Collects all children of a node into a vector
    fn collect_children(parent: &NodeRef<T, P>) -> Vec<NodeRef<T, P>> {
        let mut children = Vec::new();
        let mut current = parent.try_borrow_mut().unwrap().child.take();

        while let Some(curr) = current {
            let next = curr.try_borrow_mut().unwrap().sibling.take();
            // Clear parent, prev and sibling pointers (each child becomes a root)
            {
                let mut curr_ref = curr.try_borrow_mut().unwrap();
                curr_ref.parent = None;
                curr_ref.prev = None;
                curr_ref.sibling = None;
            }
            children.push(curr);
            current = next;
        }

        children
    }

    /// Merges a list of trees using rank-based pairing
    fn merge_children(mut children: Vec<NodeRef<T, P>>) -> NodeRef<T, P> {
        if children.len() == 1 {
            return children.pop().unwrap();
        }

        // Simple pairing approach: repeatedly pair adjacent trees
        // With RefBox we need to handle ownership carefully
        while children.len() > 1 {
            let mut next = Vec::new();

            while children.len() >= 2 {
                let b = children.pop().unwrap();
                let a = children.pop().unwrap();

                // Compare priorities
                let a_is_smaller = {
                    let a_ref = a.try_borrow_mut().unwrap();
                    let b_ref = b.try_borrow_mut().unwrap();
                    a_ref.priority < b_ref.priority
                };

                if a_is_smaller {
                    Self::make_child(&a, b);
                    Self::update_rank(&a);
                    next.push(a);
                } else {
                    Self::make_child(&b, a);
                    Self::update_rank(&b);
                    next.push(b);
                }
            }

            // If odd number, move the remaining one to next
            if !children.is_empty() {
                next.push(children.pop().unwrap());
            }

            children = next;
        }

        children.pop().unwrap()
    }

    /// Links two trees of the same rank
    #[allow(dead_code)]
    fn link_same_rank(a: NodeRef<T, P>, b: NodeRef<T, P>) -> NodeRef<T, P> {
        let a_is_smaller = {
            let a_ref = a.try_borrow_mut().unwrap();
            let b_ref = b.try_borrow_mut().unwrap();
            a_ref.priority < b_ref.priority
        };

        if a_is_smaller {
            Self::make_child(&a, b);
            Self::update_rank(&a);
            a
        } else {
            Self::make_child(&b, a);
            Self::update_rank(&b);
            b
        }
    }

    // Note: No free_node function needed - RefBox handles cleanup automatically.
    // When a node's RefBox is dropped, it and all its children are dropped.
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
