//! Pairing Heap implementation using RefBox for memory-efficient references
//!
//! A pairing heap is a type of heap-ordered tree with:
//! - O(1) amortized insert and merge
//! - O(log n) amortized delete_min
//! - o(log n) amortized decrease_key (in fact, better than log n)
//!
//! # Memory Model: RefBox with Direct Access
//!
//! This implementation uses RefBox/Weak references with a specific design choice:
//! - Owned references (RefBox) flow from roots to children (ownership)
//! - Weak references for parent/prev backlinks and handles
//! - Single owner with many weak refs (16 bytes less overhead per node vs `Rc<RefCell>`)
//!
//! **Why not use RefBox's try_borrow_mut()?**
//!
//! RefBox provides runtime borrow checking similar to RefCell, but we don't need it:
//! - RefBox is **single-owner**: there's exactly one strong reference to each node
//! - When we have a RefBox (strong ref), we *own* that node exclusively
//! - The borrow will *always* succeed because there can't be conflicting borrows
//!   from a single-owner system without aliasing
//!
//! Instead, we use `as_ptr()` to access node data directly. This is safe because:
//! - When holding &RefBox, we have shared access → safe to read via pointer
//! - When holding &mut RefBox, we have exclusive access → safe to mutate via pointer
//! - No runtime overhead from borrow tracking we don't need
//!
//! **The one exception: decrease_key via weak references**
//!
//! Handles store Weak references to nodes. When `decrease_key` is called, we must
//! access node data through a weak reference (the handle), not through the ownership
//! tree. This is the *only* place where we access data without owning the RefBox.
//!
//! This is still safe because:
//! - We have `&mut self` (exclusive access to the entire heap)
//! - The weak ref points to a node inside our heap (verified by `is_alive()`)
//! - No other code can be accessing that node during this operation
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

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError, MergeableHeap};
#[cfg(feature = "access_tracking")]
use crate::{track, track_op};
use refbox::{RefBox, Weak};

/// Type alias for strong reference to a node (single owner)
type NodeRef<T, P> = RefBox<Node<T, P>>;
/// Type alias for weak reference to a node (used for backlinks and handles)
type NodeWeak<T, P> = Weak<Node<T, P>>;

// =============================================================================
// Direct Access Helpers
// =============================================================================
//
// RefBox is single-owner: when you have a &RefBox or &mut RefBox, you have
// exclusive access to that subtree. Runtime borrow checking (try_borrow_mut)
// is unnecessary overhead because there cannot be conflicting borrows.
//
// These helpers provide direct pointer access, which is safe because:
// - node_ref(&RefBox) → &Node: RefBox lifetime ensures node is valid
// - node_mut(&RefBox) → &mut Node: Single ownership ensures exclusivity

/// Get shared reference to node data from an owned RefBox.
///
/// # Safety Rationale
/// RefBox is single-owner. When we have &RefBox, we have valid access to the node.
/// The returned reference is bounded by the RefBox's lifetime.
#[inline]
fn node_ref<T, P>(refbox: &NodeRef<T, P>) -> &Node<T, P> {
    // SAFETY: RefBox guarantees the pointer is valid while we hold the reference.
    // Single-owner semantics mean no other code has mutable access.
    unsafe { &*refbox.as_ptr() }
}

/// Get mutable reference to node data from an owned RefBox.
///
/// # Safety Rationale
/// RefBox is single-owner. When we hold a RefBox (through ownership or a reference
/// in the heap's ownership tree), we have exclusive access to that node's data.
///
/// This function takes `&RefBox` rather than `&mut RefBox` because:
/// - RefBox's single-owner semantics already guarantee exclusive access
/// - We often need to mutate nodes through Option<RefBox<Node>> fields without
///   taking ownership of the entire Option
/// - The parent-child relationship forms a strict tree, so no aliasing is possible
///
/// Clippy's `mut_from_ref` lint flags this pattern because it appears to violate
/// Rust's borrowing rules. However, it's sound here because RefBox guarantees
/// single ownership - there is exactly one strong reference to each node, so the
/// data cannot be simultaneously accessed elsewhere.
#[allow(clippy::mut_from_ref)]
#[inline]
fn node_mut<T, P>(refbox: &NodeRef<T, P>) -> &mut Node<T, P> {
    // SAFETY: RefBox is single-owner. We have exclusive access to this subtree
    // through the heap's ownership tree. No aliasing is possible.
    unsafe { &mut *(refbox.as_ptr() as *mut Node<T, P>) }
}

/// Handle to an element in a Pairing heap
///
/// The handle uses a weak reference to the node. If the node has been removed
/// from the heap (e.g., via delete_min), the handle becomes invalid.
/// Operations on invalid handles will fail gracefully.
pub struct PairingHandle<T, P> {
    node: NodeWeak<T, P>,
}

// Manual implementations because RefBox's Weak doesn't derive these traits
impl<T, P> Clone for PairingHandle<T, P> {
    fn clone(&self) -> Self {
        PairingHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for PairingHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node.as_ptr() == other.node.as_ptr()
    }
}

impl<T, P> Eq for PairingHandle<T, P> {}

impl<T, P> std::fmt::Debug for PairingHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairingHandle")
            .field("node", &self.node.as_ptr())
            .finish()
    }
}

impl<T, P> Handle for PairingHandle<T, P> {}

/// Internal node structure for pairing heap
///
/// **Cache Optimization**: Fields are ordered for cache locality:
/// - Hot path first: `priority` is accessed on every comparison
/// - Traversal fields next: pointers for tree navigation
/// - Cold path last: `item` is only accessed when popping
///
/// The structure forms a multi-way tree where children are linked via sibling pointers.
/// Owned references (RefBox) flow downward, weak references point upward.
/// This prevents reference cycles while allowing O(1) insertion and merging.
///
/// **Memory Efficiency**: Using RefBox instead of Rc<RefCell> saves ~16 bytes per node.
struct Node<T, P> {
    /// Priority for heap ordering - Hot path: accessed on every comparison
    priority: P,
    /// First child in the child list. None if this node is a leaf.
    /// Uses owned reference (RefBox) as parent owns children.
    child: Option<NodeRef<T, P>>,
    /// Next sibling in the parent's child list. None if this is the last child.
    /// Uses owned reference (RefBox) as earlier siblings own later ones in the list.
    sibling: Option<NodeRef<T, P>>,
    /// Parent node or previous sibling. Uses weak reference to avoid cycles.
    /// Used for O(1) cuts during decrease_key.
    prev: Option<NodeWeak<T, P>>,
    /// The item stored in the heap - Cold path: only accessed on pop
    item: T,
}

/// Pairing Heap using RefBox for memory-efficient references
///
/// A safe implementation using `RefBox` and `Weak` references.
/// Owned references (`RefBox`) flow from root to children, weak references (`Weak`)
/// point upward for efficient decrease_key operations.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::pairing::PairingHeap;
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
///
/// let mut heap = PairingHeap::new();
/// let handle = heap.push_with_handle(5, "item");
/// heap.decrease_key(&handle, 1).unwrap();
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct PairingHeap<T, P: Ord> {
    /// Root of the heap tree. Uses owned reference as the heap owns the root.
    root: Option<NodeRef<T, P>>,
    len: usize,
}

// No Drop implementation needed - RefBox handles cleanup automatically.
// When the heap is dropped, the root RefBox is dropped,
// which recursively drops all children (via their strong references).

impl<T, P: Ord> Heap<T, P> for PairingHeap<T, P> {
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
        // Safety: The references we return point to data inside a RefBox that the heap owns.
        // As long as the caller has a borrow of &self, the RefBox stays alive and the data is valid.
        // This is effectively a lifetime extension that is safe because:
        // 1. The heap owns the RefBox (root)
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
    fn pop(&mut self) -> Option<(P, T)> {
        #[cfg(feature = "access_tracking")]
        let _guard = track_op!("pop");

        let root_refbox = self.root.take()?;

        #[cfg(feature = "access_tracking")]
        track!(node);

        // Consume the RefBox immediately to get owned Node - no borrowing needed
        // RefBox::take() consumes the RefBox and returns the inner Node
        #[cfg(feature = "access_tracking")]
        track!(read ChildPtr);
        let mut root_node = root_refbox.take();
        let first_child = root_node.child.take();

        // The root has been deleted, so we need to rebuild from its children
        // If there are no children, the heap becomes empty
        // Otherwise, we merge all children using two-pass pairing
        if let Some(child) = first_child {
            // Clear the child's prev pointer (it was pointing to the old root)
            #[cfg(feature = "access_tracking")]
            {
                track!(chase);
                track!(node);
                track!(write ParentPtr);
            }
            node_mut(&child).prev = None;
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

        #[cfg(feature = "access_tracking")]
        {
            track!(read Priority);
            track!(read Item);
        }
        Some((root_node.priority, root_node.item))
    }
}

impl<T, P: Ord> MergeableHeap<T, P> for PairingHeap<T, P> {
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
        // Direct access is safe: we own both RefBoxes exclusively
        let other_is_smaller = node_ref(&other_root).priority < node_ref(&self_root).priority;

        if other_is_smaller {
            // Other root has smaller priority: it becomes the new root
            // Self root becomes a child of other root

            // Get weak references first
            let self_root_weak = self_root.downgrade();
            let other_root_weak = other_root.downgrade();

            // Take other_root's existing first child
            let old_first_child = node_mut(&other_root).child.take();

            // Update old_first_child's prev to point to self_root (if it exists)
            if let Some(ref old_child) = old_first_child {
                node_mut(old_child).prev = Some(self_root_weak);
            }

            // Set up self_root as a child
            {
                let self_node = node_mut(&self_root);
                self_node.sibling = old_first_child;
                self_node.prev = Some(other_root_weak);
            }

            // Make self_root the first child of other_root
            node_mut(&other_root).child = Some(self_root);
            self.root = Some(other_root);
        } else {
            // Self root has smaller or equal priority: it stays root
            // Other root becomes a child of self root

            // Get weak references first
            let other_root_weak = other_root.downgrade();
            let self_root_weak = self_root.downgrade();

            // Take self_root's existing first child
            let old_first_child = node_mut(&self_root).child.take();

            // Update old_first_child's prev to point to other_root (if it exists)
            if let Some(ref old_child) = old_first_child {
                node_mut(old_child).prev = Some(other_root_weak);
            }

            // Set up other_root as a child
            {
                let other_node = node_mut(&other_root);
                other_node.sibling = old_first_child;
                other_node.prev = Some(self_root_weak);
            }

            // Make other_root the first child of self_root
            node_mut(&self_root).child = Some(other_root);
            self.root = Some(self_root);
        }

        // Update length (other is automatically empty after take())
        self.len += other.len;
        other.len = 0;
    }
}

impl<T, P: Ord> DecreaseKeyHeap<T, P> for PairingHeap<T, P> {
    type Handle = PairingHandle<T, P>;

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
    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        #[cfg(feature = "access_tracking")]
        let _guard = track_op!("push");

        // Create new node with no children or siblings yet
        let node = RefBox::new(Node {
            // Hot path first
            priority,
            // Traversal fields
            child: None,
            sibling: None,
            prev: None,
            // Cold path last
            item,
        });

        #[cfg(feature = "access_tracking")]
        {
            track!(node);
            track!(write Priority);
            track!(write Item);
        }

        // Create handle before we potentially move the node
        let handle = PairingHandle {
            node: node.downgrade(),
        };

        // Link new node into the tree structure
        if let Some(root) = self.root.take() {
            #[cfg(feature = "access_tracking")]
            {
                track!(read Priority);
                track!(read Priority);
            }
            // Compare priorities - direct access is safe: we own both RefBoxes
            let node_is_smaller = node_ref(&node).priority < node_ref(&root).priority;

            if node_is_smaller {
                // Case 1: New node has smaller priority
                // Make new node the root, old root becomes its only child
                // This maintains the heap property (parent <= child)
                #[cfg(feature = "access_tracking")]
                {
                    track!(write ChildPtr);
                    track!(write ParentPtr);
                }
                let node_weak = node.downgrade();
                node_mut(&root).prev = Some(node_weak);
                node_mut(&node).child = Some(root);
                self.root = Some(node);
            } else {
                // Case 2: Current root has smaller or equal priority
                // Add new node as the first child of root
                // This maintains heap property (new node >= root)
                #[cfg(feature = "access_tracking")]
                {
                    track!(read ChildPtr);
                    track!(write SiblingPtr);
                    track!(write ParentPtr);
                    track!(write ChildPtr);
                }
                // Get weak references first
                let node_weak = node.downgrade();
                let root_weak = root.downgrade();

                // Take root's existing first child
                let old_first_child = node_mut(&root).child.take();

                // Update old_first_child's prev to point to new node (if it exists)
                if let Some(ref old_child) = old_first_child {
                    #[cfg(feature = "access_tracking")]
                    track!(write ParentPtr);
                    node_mut(old_child).prev = Some(node_weak);
                }

                // Set up new node as first child of root
                {
                    let new_node = node_mut(&node);
                    new_node.sibling = old_first_child;
                    new_node.prev = Some(root_weak);
                }

                // Make new node the first child of root
                node_mut(&root).child = Some(node);
                self.root = Some(root);
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
        #[cfg(feature = "access_tracking")]
        let _guard = track_op!("decrease_key");

        // Check if the weak reference is still valid
        // If not, the node was already removed from the heap
        if !handle.node.is_alive() {
            return Err(HeapError::PriorityNotDecreased);
        }

        #[cfg(feature = "access_tracking")]
        track!(node);

        // Get node pointer for comparisons
        let node_ptr = handle.node.as_ptr();

        // Access node through weak reference to check and update priority.
        //
        // SAFETY: This is the ONE place we access a node through a weak reference
        // rather than through the ownership tree. This is safe because:
        // 1. We have &mut self - exclusive access to the entire heap
        // 2. is_alive() confirmed the node exists in our heap
        // 3. No other code can access this node during this operation
        // 4. The pointer remains valid for the duration of &mut self
        #[cfg(feature = "access_tracking")]
        track!(read Priority);
        let node = unsafe { &mut *(node_ptr as *mut Node<T, P>) };
        if new_priority >= node.priority {
            return Err(HeapError::PriorityNotDecreased);
        }
        #[cfg(feature = "access_tracking")]
        track!(write Priority);
        node.priority = new_priority;

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

        // The node is not the root, so it has a parent
        // Strategy: Cut the node from its parent and merge it with the root
        // This is the standard decrease_key operation for pairing heaps

        // Step 1: Cut the node from its parent's child list and get ownership
        let cut_node = match self.cut_node_by_ptr(node_ptr) {
            Some(n) => n,
            None => return Err(HeapError::PriorityNotDecreased), // Node not found
        };

        // Step 2: Merge the cut node with the root using merge_nodes
        // This properly handles preserving children of both nodes
        if let Some(root) = self.root.take() {
            self.root = Some(self.merge_nodes(cut_node, root));
        } else {
            // Heap is empty (shouldn't happen, but handle gracefully)
            self.root = Some(cut_node);
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
        // Verify heap property: node priority >= parent priority using raw pointer
        let node_ptr = node.as_ptr();
        if let Some(parent_p) = parent_priority {
            let node_priority = unsafe { &(*node_ptr).priority };
            assert!(
                node_priority >= parent_p,
                "Heap property violated: child priority < parent priority"
            );
        }

        let mut count = 1;

        // Collect child pointers for traversal (using raw pointers to avoid clone)
        let mut child_ptrs: Vec<*const Node<T, P>> = Vec::new();
        {
            let mut child_opt = unsafe { (*node_ptr).child.as_ref() };
            while let Some(child) = child_opt {
                child_ptrs.push(child.as_ptr());
                child_opt = unsafe { (*child.as_ptr()).sibling.as_ref() };
            }
        }

        // Count all children recursively
        for child_ptr in child_ptrs {
            // Use unsafe to get a stable reference to parent's priority
            let node_priority = unsafe { &(*node_ptr).priority };
            // Get a reference to the child RefBox from its parent
            // Since we collected pointers while holding a borrow, this is safe
            // as long as the tree structure hasn't changed
            let child_ref = unsafe { &*child_ptr };
            count += self.count_and_verify_subtree_inner(child_ref, Some(node_priority));
        }

        count
    }

    #[cfg(feature = "expensive_verify")]
    #[allow(clippy::only_used_in_recursion)]
    fn count_and_verify_subtree_inner(
        &self,
        node: &Node<T, P>,
        parent_priority: Option<&P>,
    ) -> usize {
        // Verify heap property: node priority >= parent priority
        if let Some(parent_p) = parent_priority {
            assert!(
                &node.priority >= parent_p,
                "Heap property violated: child priority < parent priority"
            );
        }

        let mut count = 1;

        // Traverse children using raw pointers
        let mut child_opt = node.child.as_ref();
        while let Some(child) = child_opt {
            let child_ref = unsafe { &*child.as_ptr() };
            count += self.count_and_verify_subtree_inner(child_ref, Some(&node.priority));
            child_opt = child_ref.sibling.as_ref();
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
        #[cfg(feature = "access_tracking")]
        track!(read SiblingPtr);
        let has_sibling = node_ref(&first).sibling.is_some();
        if !has_sibling {
            return first;
        }

        // Count initial nodes (debug only)
        #[cfg(feature = "expensive_verify")]
        let initial_count = {
            let mut count = 0;
            // Use raw pointers for traversal to avoid clone
            let mut curr_ptr: Option<*const Node<T, P>> = Some(first.as_ptr());
            while let Some(ptr) = curr_ptr {
                count += Self::count_subtree_ptr(ptr);
                curr_ptr = unsafe { (*ptr).sibling.as_ref().map(|s| s.as_ptr()) };
            }
            count
        };

        // First pass: pair adjacent children and merge each pair
        // This reduces the number of trees by approximately half
        let mut pairs: Vec<NodeRef<T, P>> = Vec::new();
        let mut current = Some(first);

        while let Some(node) = current {
            #[cfg(feature = "access_tracking")]
            {
                track!(node);
                track!(read SiblingPtr);
                track!(write SiblingPtr);
                track!(write ParentPtr);
            }
            // Get and disconnect sibling before modifying node
            let sibling = node_mut(&node).sibling.take();
            node_mut(&node).prev = None;

            if let Some(sib) = sibling {
                #[cfg(feature = "access_tracking")]
                {
                    track!(chase);
                    track!(node);
                    track!(read SiblingPtr);
                    track!(write SiblingPtr);
                    track!(write ParentPtr);
                }
                // We have a pair: node and its sibling
                // Get the next node before we disconnect
                let next = node_mut(&sib).sibling.take();
                node_mut(&sib).prev = None;

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
            let after_first_pass: usize = pairs
                .iter()
                .map(|p| Self::count_subtree_ptr(p.as_ptr()))
                .sum();
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
            let final_count = Self::count_subtree_ptr(result.as_ptr());
            assert_eq!(
                initial_count, final_count,
                "Nodes lost during second pass: initial={} final={}",
                initial_count, final_count
            );
        }

        result
    }

    #[cfg(feature = "expensive_verify")]
    fn count_subtree_ptr(node_ptr: *const Node<T, P>) -> usize {
        let node = unsafe { &*node_ptr };
        let mut count = 1;

        // Count children using raw pointers
        let mut child_opt = node.child.as_ref().map(|c| c.as_ptr());
        while let Some(child_ptr) = child_opt {
            count += Self::count_subtree_ptr(child_ptr);
            let child = unsafe { &*child_ptr };
            child_opt = child.sibling.as_ref().map(|s| s.as_ptr());
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
        #[cfg(feature = "access_tracking")]
        {
            track!(read Priority);
            track!(read Priority);
        }
        // Compare priorities - direct access is safe: we own both RefBoxes
        let a_is_smaller = node_ref(&a).priority < node_ref(&b).priority;

        if a_is_smaller {
            // Node a has smaller priority: it becomes parent
            // Node b becomes a child of a
            #[cfg(feature = "access_tracking")]
            {
                track!(read ChildPtr);
                track!(write SiblingPtr);
                track!(write ParentPtr);
                track!(write ChildPtr);
            }

            // Get weak references first
            let a_weak = a.downgrade();
            let b_weak = b.downgrade();

            // Take a's old first child
            let old_first_child = node_mut(&a).child.take();

            // Update old_first_child's prev to point to b (if it exists)
            if let Some(ref old_child) = old_first_child {
                #[cfg(feature = "access_tracking")]
                track!(write ParentPtr);
                node_mut(old_child).prev = Some(b_weak);
            }

            // Make b the first child of a
            {
                let b_node = node_mut(&b);
                b_node.sibling = old_first_child;
                b_node.prev = Some(a_weak);
            }
            node_mut(&a).child = Some(b);
            a // Return a as the new root
        } else {
            // Node b has smaller or equal priority: it becomes parent
            // Node a becomes a child of b
            #[cfg(feature = "access_tracking")]
            {
                track!(read ChildPtr);
                track!(write SiblingPtr);
                track!(write ParentPtr);
                track!(write ChildPtr);
            }

            // Get weak references first
            let b_weak = b.downgrade();
            let a_weak = a.downgrade();

            // Take b's old first child
            let old_first_child = node_mut(&b).child.take();

            // Update old_first_child's prev to point to a (if it exists)
            if let Some(ref old_child) = old_first_child {
                #[cfg(feature = "access_tracking")]
                track!(write ParentPtr);
                node_mut(old_child).prev = Some(a_weak);
            }

            // Make a the first child of b
            {
                let a_node = node_mut(&a);
                a_node.sibling = old_first_child;
                a_node.prev = Some(b_weak);
            }
            node_mut(&b).child = Some(a);
            b // Return b as the new root
        }
    }

    /// Cuts a node from its parent by pointer, removing it from the child list
    /// and returning ownership of the node.
    ///
    /// **Time Complexity**: O(1) using prev pointers for direct access
    ///
    /// **Algorithm**:
    /// - Use the node's prev pointer to find the parent/sibling
    /// - Remove it from its parent's child list or sibling chain
    /// - Return ownership of the node
    ///
    /// **Used in**: decrease_key operation to remove a node that violates heap property
    fn cut_node_by_ptr(&mut self, node_ptr: *const Node<T, P>) -> Option<NodeRef<T, P>> {
        // We need to find the node in the tree and get its prev pointer
        // Use the node's prev pointer to find the owner
        //
        // SAFETY for weak ref access: Same rationale as decrease_key - we have &mut self
        // (exclusive heap access) and is_alive() confirms the node exists in our heap.
        let prev_weak = unsafe { (*node_ptr).prev.clone()? };

        if !prev_weak.is_alive() {
            return None;
        }

        #[cfg(feature = "access_tracking")]
        {
            track!(chase);
            track!(node);
        }

        // Access prev node through weak reference
        // SAFETY: We have &mut self, prev is alive, no aliasing possible
        let prev_node = unsafe { &mut *(prev_weak.as_ptr() as *mut Node<T, P>) };

        // Check if prev's child is our node
        let is_first_child = prev_node
            .child
            .as_ref()
            .map(|c| c.as_ptr() == node_ptr)
            .unwrap_or(false);

        if is_first_child {
            // Node is the first child - take it from parent
            let node = prev_node.child.take()?;

            // Get node's sibling for prev
            let node_sibling = node_mut(&node).sibling.take();

            // Update sibling's prev pointer
            let prev_weak_for_sibling = prev_weak.clone();
            if let Some(ref sibling) = node_sibling {
                node_mut(sibling).prev = Some(prev_weak_for_sibling);
            }

            // Set parent's child to node's sibling
            prev_node.child = node_sibling;

            // Clear node's prev
            node_mut(&node).prev = None;

            Some(node)
        } else {
            // Node is not first child - it's in the sibling chain
            // prev is actually previous sibling, need to find and take it
            let node = prev_node.sibling.take()?;

            // Verify we got the right node
            if node.as_ptr() != node_ptr {
                // Put it back and return None - something is wrong
                prev_node.sibling = Some(node);
                return None;
            }

            // Get node's sibling for prev
            let node_sibling = node_mut(&node).sibling.take();

            // Update sibling's prev pointer
            let prev_weak_for_sibling = prev_weak.clone();
            if let Some(ref sibling) = node_sibling {
                node_mut(sibling).prev = Some(prev_weak_for_sibling);
            }

            // Set prev sibling's next sibling to node's sibling
            prev_node.sibling = node_sibling;

            // Clear node's prev
            node_mut(&node).prev = None;

            Some(node)
        }
    }

    // Note: No free_node function needed - RefBox handles cleanup automatically.
    // When a RefBox is dropped, it and all its owned children are dropped.
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
