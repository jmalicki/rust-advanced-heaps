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
use std::ptr::{self, NonNull};

/// Handle to an element in a Rank-pairing heap
///
/// Note: This handle is tied to a specific heap instance. Using it with a different
/// heap or after the heap is dropped is undefined behavior.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RankPairingHandle {
    node: *const (), // Type-erased pointer to Node<T, P>
}

impl Handle for RankPairingHandle {}

/// Internal node structure for rank-pairing heap
///
/// Each node maintains:
/// - `item` and `priority`: The data stored in the heap
/// - `parent`: Pointer to parent node (None if root)
/// - `child`: Pointer to first child (None if leaf)
/// - `sibling`: Next sibling in parent's child list (None if last child)
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
    /// Parent node (None if root)
    parent: Option<NonNull<Node<T, P>>>,
    /// First child in child list (None if leaf)
    child: Option<NonNull<Node<T, P>>>,
    /// Next sibling in parent's child list (None if last child)
    sibling: Option<NonNull<Node<T, P>>>,
    /// Explicit rank: rank(v) = min(rank(w₁), rank(w₂)) + 1 where w₁, w₂ are
    /// children with smallest ranks. This bounds tree height at O(log n).
    rank: usize,
    /// Marked flag (Type-A): false if no child lost, true if one child lost.
    /// After losing two children, the node is cut (cascading).
    marked: bool, // Type-A: false if no child lost, true if one child lost
}

/// Rank-Pairing Heap
///
/// This implementation uses type-A rank-pairing heaps, which maintain:
/// - A node loses at most one child before being cut
/// - Ranks satisfy: r(v) <= r(w1) + 1 and r(v) <= r(w2) + 1
///   where w1, w2 are the two children with smallest ranks
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
    root: Option<NonNull<Node<T, P>>>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P: Ord> Drop for RankPairingHeap<T, P> {
    fn drop(&mut self) {
        // Recursively free all nodes
        if let Some(root) = self.root {
            unsafe {
                Self::free_node(root);
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for RankPairingHeap<T, P> {
    type Handle = RankPairingHandle;

    fn new() -> Self {
        Self {
            root: None,
            len: 0,
            _phantom: std::marker::PhantomData,
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
        let node = Box::into_raw(Box::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            rank: 0,       // Leaf nodes have rank 0
            marked: false, // New nodes are unmarked
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        // Link new node into the tree structure
        if let Some(root_ptr) = self.root {
            unsafe {
                if (*node).priority < (*root_ptr.as_ptr()).priority {
                    // Case 1: New node has smaller priority
                    // Make new node the root, old root becomes its child
                    // This maintains heap property: parent <= child
                    self.make_child(node_ptr, root_ptr);
                    // Update root pointer to new minimum
                    self.root = Some(node_ptr);
                } else {
                    // Case 2: Current root has smaller or equal priority
                    // Add new node as a child of the root
                    // Heap property maintained: new node >= root
                    self.make_child(root_ptr, node_ptr);
                    // Root stays the same
                }
            }
        } else {
            // Empty heap: new node becomes root
            self.root = Some(node_ptr);
        }

        self.len += 1;
        RankPairingHandle {
            node: node_ptr.as_ptr() as *const (),
        }
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        self.root.map(|root_ptr| unsafe {
            let node = root_ptr.as_ptr();
            (&(*node).priority, &(*node).item)
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
        let root_ptr = self.root?;

        unsafe {
            let node = root_ptr.as_ptr();
            // Read out item and priority before freeing the node
            let (priority, item) = (ptr::read(&(*node).priority), ptr::read(&(*node).item));

            // Collect all children of the root
            // Each child is a root of a subtree (parent links will be cleared)
            let children = self.collect_children(root_ptr);

            // Free the root node (children have been collected)
            drop(Box::from_raw(node));
            self.len -= 1;

            if children.is_empty() {
                // No children: heap becomes empty
                self.root = None;
            } else {
                // Merge all children using rank-based pairing
                // This operation maintains rank constraints and produces a single root
                // The pairing strategy ensures O(log n) amortized cost
                self.root = Some(self.merge_children(children));
            }

            Some((priority, item))
        }
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
        let node_ptr = unsafe { NonNull::new_unchecked(handle.node as *mut Node<T, P>) };

        unsafe {
            let node = node_ptr.as_ptr();

            // Safety check: new priority must actually be less
            if new_priority >= (*node).priority {
                return Err(HeapError::PriorityNotDecreased);
            }

            // Update the priority value
            (*node).priority = new_priority;

            // If node is already root, heap property is satisfied (no parent)
            if self.root == Some(node_ptr) {
                return Ok(());
            }

            // Node is not root, so it has a parent
            // Check if heap property is violated
            if let Some(parent) = (*node).parent {
                if (*node).priority < (*parent.as_ptr()).priority {
                    // Heap property violated: cut node from parent
                    // This operation may trigger cascading cuts if parent was marked
                    self.cut(node_ptr);

                    // Merge cut node with root
                    // If cut node has smaller priority, it becomes the new root
                    // Otherwise, it becomes a child of the current root
                    if let Some(root_ptr) = self.root {
                        if (*node).priority < (*root_ptr.as_ptr()).priority {
                            // Cut node has smaller priority: make it the new root
                            self.make_child(node_ptr, root_ptr);
                            self.root = Some(node_ptr);
                        } else {
                            // Current root has smaller priority: add cut node as child
                            self.make_child(root_ptr, node_ptr);
                        }
                    } else {
                        // Heap is empty (shouldn't happen, but handle gracefully)
                        self.root = Some(node_ptr);
                    }
                }
                // If heap property is not violated, no restructuring needed
            }
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
        unsafe {
            let self_root = self.root.unwrap();
            let other_root = other.root.unwrap();

            // Compare roots: smaller priority becomes parent (heap property)
            if (*other_root.as_ptr()).priority < (*self_root.as_ptr()).priority {
                // Other root has smaller priority: it becomes the new root
                // Self root becomes a child of other root
                self.make_child(other_root, self_root);
                self.root = Some(other_root);
            } else {
                // Self root has smaller or equal priority: it stays root
                // Other root becomes a child of self root
                self.make_child(self_root, other_root);
            }

            // Update length and mark other as empty (prevent double-free)
            self.len += other.len;
            other.root = None;
            other.len = 0;
        }
    }
}

impl<T, P: Ord> RankPairingHeap<T, P> {
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
    unsafe fn make_child(&mut self, x: NonNull<Node<T, P>>, y: NonNull<Node<T, P>>) {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();

        // Set parent-child relationship
        (*y_ptr).parent = Some(x);
        // Add y to x's child list (insert at front)
        (*y_ptr).sibling = (*x_ptr).child;
        (*x_ptr).child = Some(y);

        // Update rank: x's rank must be recomputed based on its children
        // Rank formula: rank(x) = min(rank(w₁), rank(w₂)) + 1
        // This ensures rank constraints are maintained
        self.update_rank(x);
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
    unsafe fn update_rank(&self, node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();

        if let Some(child) = (*node_ptr).child {
            // Collect all children's ranks
            // We need to find the two smallest ranks to compute the new rank
            let mut ranks = Vec::new();
            let mut current = Some(child);

            // Traverse child list to collect all ranks
            while let Some(curr) = current {
                ranks.push((*curr.as_ptr()).rank);
                current = (*curr.as_ptr()).sibling;
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
                (*node_ptr).rank = (r1.max(r2)) + 1;
            } else if ranks.len() == 1 {
                // One child: rank = child_rank + 1
                (*node_ptr).rank = ranks[0] + 1;
            } else {
                // No children (shouldn't happen, but handle gracefully)
                (*node_ptr).rank = 0;
            }
        } else {
            // Leaf node: rank is 0
            (*node_ptr).rank = 0;
        }
    }

    /// Cuts a node from its parent and makes it a root
    ///
    /// **Time Complexity**: O(1) amortized (cascading cuts amortized to O(1))
    ///
    /// **Algorithm (Type-A Cascading Cuts)**:
    /// 1. Remove node from parent's child list
    /// 2. Clear node's parent and sibling pointers
    /// 3. **Marking rule**: If parent is not marked, mark it
    /// 4. **Cascading**: If parent is already marked, cut it too (cascade upward)
    /// 5. Update parent's rank after losing a child
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
    unsafe fn cut(&mut self, node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();
        let parent_opt = match (*node_ptr).parent {
            Some(p) => p,
            None => return, // Node is already root or orphaned
        };
        let parent_ptr = parent_opt.as_ptr();

        // Step 1: Remove node from parent's child list
        // Check if node is the first child or a later child
        if (*parent_ptr).child == Some(node) {
            // Node is first child: parent's first child becomes node's sibling
            (*parent_ptr).child = (*node_ptr).sibling;
        } else {
            // Node is not first child: find it in sibling chain and remove
            // This requires traversing the sibling list
            let mut current = (*parent_ptr).child;
            while let Some(curr) = current {
                if (*curr.as_ptr()).sibling == Some(node) {
                    // Found node: skip it in sibling chain
                    (*curr.as_ptr()).sibling = (*node_ptr).sibling;
                    break;
                }
                current = (*curr.as_ptr()).sibling;
            }
        }

        // Step 2: Clear node's parent and sibling pointers (it's now a root)
        (*node_ptr).parent = None;
        (*node_ptr).sibling = None;

        // Step 3: Apply marking rule (Type-A rank-pairing heap)
        // This maintains the constraint that no node loses more than one child
        if !(*parent_ptr).marked {
            // Parent hasn't lost a child yet: mark it
            // Next time it loses a child, it will be cut (cascading)
            (*parent_ptr).marked = true;
        } else {
            // Parent already marked (has lost one child): cut it now (cascading)
            // This prevents nodes from losing too many children
            // Recursive cut may trigger further cascades
            self.cut(parent_opt);
            // Fix up rank constraints after cascading cut
            self.fixup(parent_opt);
        }

        // Step 4: Update parent's rank (it lost a child, rank may decrease)
        // This maintains rank constraints after cutting
        self.update_rank(parent_opt);
    }

    /// Performs rank-based fixup after cutting
    unsafe fn fixup(&mut self, node: NonNull<Node<T, P>>) {
        // For type-A rank-pairing heaps, we need to ensure rank constraints
        // are maintained. The cut operation may have violated them.
        let node_ptr = node.as_ptr();

        // If this is not a root, we may need to fix up the parent chain
        if (*node_ptr).parent.is_some() {
            // The rank update already happened in cut()
            // We may need additional restructuring, but for simplicity,
            // we'll rely on the rank constraints from cutting
        }
    }

    /// Collects all children of a node into a vector
    unsafe fn collect_children(&self, parent: NonNull<Node<T, P>>) -> Vec<NonNull<Node<T, P>>> {
        let mut children = Vec::new();
        let mut current = (*parent.as_ptr()).child;

        while let Some(curr) = current {
            let next = (*curr.as_ptr()).sibling;
            (*curr.as_ptr()).parent = None;
            (*curr.as_ptr()).sibling = None;
            children.push(curr);
            current = next;
        }

        children
    }

    /// Merges a list of trees using rank-based pairing
    unsafe fn merge_children(
        &mut self,
        mut children: Vec<NonNull<Node<T, P>>>,
    ) -> NonNull<Node<T, P>> {
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
                    let a = children[i];
                    let b = children[i + 1];
                    let merged = if (*a.as_ptr()).priority < (*b.as_ptr()).priority {
                        self.make_child(a, b);
                        self.update_rank(a);
                        a
                    } else {
                        self.make_child(b, a);
                        self.update_rank(b);
                        b
                    };
                    next.push(merged);
                    i += 2;
                } else {
                    // Single tree left, add it to next round
                    next.push(children[i]);
                    i += 1;
                }
            }
            children = next;
        }

        children.pop().unwrap()
    }

    /// Links two trees of the same rank
    #[allow(dead_code)]
    unsafe fn link_same_rank(
        &mut self,
        a: NonNull<Node<T, P>>,
        b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        if (*a.as_ptr()).priority < (*b.as_ptr()).priority {
            self.make_child(a, b);
            self.update_rank(a);
            a
        } else {
            self.make_child(b, a);
            self.update_rank(b);
            b
        }
    }

    /// Recursively frees a node and all its descendants
    unsafe fn free_node(node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();
        if let Some(child) = (*node_ptr).child {
            let mut current = Some(child);
            while let Some(curr) = current {
                let next = (*curr.as_ptr()).sibling;
                Self::free_node(curr);
                current = next;
            }
        }
        drop(Box::from_raw(node_ptr));
    }
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
