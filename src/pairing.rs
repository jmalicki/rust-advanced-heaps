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

use crate::traits::{Handle, Heap, HeapError};
use std::ptr::{self, NonNull};

/// Handle to an element in a Pairing heap
///
/// Note: This handle is tied to a specific heap instance. Using it with a different
/// heap or after the heap is dropped is undefined behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PairingHandle {
    node: *const (), // Type-erased pointer to Node<T, P>
}

impl Handle for PairingHandle {}

/// Internal node structure for pairing heap
///
/// Each node maintains:
/// - `item` and `priority`: The data stored in the heap
/// - `child`: Pointer to the first child in the child list
/// - `sibling`: Pointer to the next sibling in the parent's child list
/// - `prev`: Pointer to parent or previous sibling (used for efficient traversal and cutting)
///
/// The structure forms a multi-way tree where children are linked via sibling pointers.
/// This allows O(1) insertion and merging while maintaining heap order.
struct Node<T, P> {
    item: T,
    priority: P,
    /// First child in the child list. None if this node is a leaf.
    child: Option<NonNull<Node<T, P>>>,
    /// Next sibling in the parent's child list. None if this is the last child.
    sibling: Option<NonNull<Node<T, P>>>,
    /// Parent node or previous sibling. Used for:
    /// - Traversing up the tree during decrease_key
    /// - Efficiently removing nodes from child lists
    /// - Maintaining bidirectional links for O(1) updates
    prev: Option<NonNull<Node<T, P>>>, // For decrease_key: parent or previous sibling
}

/// Pairing Heap
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
    root: Option<NonNull<Node<T, P>>>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P: Ord> Drop for PairingHeap<T, P> {
    fn drop(&mut self) {
        // Recursively free all nodes
        if let Some(root) = self.root {
            unsafe {
                Self::free_node(root);
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for PairingHeap<T, P> {
    type Handle = PairingHandle;

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
    /// 1. Create a new single-node tree
    /// 2. Compare priority with current root
    /// 3. If new priority is smaller, make new node the root (old root becomes its child)
    /// 4. Otherwise, add new node as a child of the root
    ///
    /// This operation is O(1) because we only compare with the root and perform
    /// a constant amount of pointer manipulation. No restructuring needed!
    ///
    /// **Invariant Maintenance**:
    /// - Heap property maintained: smaller priority becomes parent
    /// - Tree structure preserved: parent-child relationships correctly linked
    /// - Root always points to minimum: updated if new element is smaller
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new node with no children or siblings yet
        let node = Box::into_raw(Box::new(Node {
            item,
            priority,
            child: None,
            sibling: None,
            prev: None,
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        // Link new node into the tree structure
        if let Some(root_ptr) = self.root {
            unsafe {
                if (*node).priority < (*root_ptr.as_ptr()).priority {
                    // Case 1: New node has smaller priority
                    // Make new node the root, old root becomes its only child
                    // This maintains the heap property (parent <= child)
                    (*node).child = Some(root_ptr);
                    (*root_ptr.as_ptr()).prev = Some(node_ptr);
                    self.root = Some(node_ptr);
                } else {
                    // Case 2: Current root has smaller or equal priority
                    // Add new node as the first child of root
                    // This maintains heap property (new node >= root)
                    let root_child = (*root_ptr.as_ptr()).child;
                    (*node).sibling = root_child; // New node's sibling is old first child
                    (*node).prev = Some(root_ptr); // New node's parent is root
                    if let Some(child) = root_child {
                        (*child.as_ptr()).prev = Some(node_ptr); // Update old first child's prev
                    }
                    (*root_ptr.as_ptr()).child = Some(node_ptr); // Root's first child is now new node
                }
            }
        } else {
            // Empty heap: new node becomes root
            self.root = Some(node_ptr);
        }

        self.len += 1;
        PairingHandle {
            node: node_ptr.as_ptr() as *const (),
        }
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
        let root_ptr = self.root?;

        unsafe {
            let node = root_ptr.as_ptr();
            // Read out the item and priority before freeing the node
            let (priority, item) = (ptr::read(&(*node).priority), ptr::read(&(*node).item));

            // The root has been deleted, so we need to rebuild from its children
            // If there are no children, the heap becomes empty
            // Otherwise, we merge all children using two-pass pairing
            let children = (*node).child;
            if let Some(first_child) = children {
                // Two-pass pairing: merge children efficiently
                // This is the core operation that achieves O(log n) amortized bounds
                self.root = Some(self.merge_pairs(first_child));
            } else {
                // No children: heap is now empty
                self.root = None;
            }

            // Free the root node (children have been moved)
            drop(Box::from_raw(node));
            self.len -= 1;
            Some((priority, item))
        }
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
        let node_ptr = unsafe { NonNull::new_unchecked(handle.node as *mut Node<T, P>) };

        unsafe {
            let node = node_ptr.as_ptr();

            // Safety check: new priority must actually be less
            // If not, the operation is a no-op (or could panic in a checked version)
            if new_priority >= (*node).priority {
                return Err(HeapError::PriorityNotDecreased);
            }

            // Update the priority value
            (*node).priority = new_priority;

            // If the node is already the root, no restructuring needed
            // The heap property is satisfied (root has no parent)
            if self.root == Some(node_ptr) {
                return Ok(());
            }

            // The node is not the root, so it has a parent
            // If the new priority is less than the parent's priority, we violate
            // the heap property and must restructure
            //
            // Strategy: Cut the node from its parent and merge it with the root
            // This is the standard decrease_key operation for pairing heaps

            // Step 1: Cut the node from its parent's child list
            // This removes all parent-child links between parent and this node
            self.cut_node(node_ptr);

            // Step 2: Merge the cut node with the root
            // If the cut node has smaller priority, it becomes the new root
            // Otherwise, it becomes a child of the current root
            // This maintains the heap property: parent <= child
            if let Some(root_ptr) = self.root {
                if (*node).priority < (*root_ptr.as_ptr()).priority {
                    // Cut node has smaller priority: make it the new root
                    // Old root becomes its child (heap property maintained)
                    (*node).child = Some(root_ptr);
                    (*root_ptr.as_ptr()).prev = Some(node_ptr);
                    self.root = Some(node_ptr);
                } else {
                    // Current root has smaller priority: add cut node as child
                    // Heap property maintained: cut node >= root
                    let root_child = (*root_ptr.as_ptr()).child;
                    (*node).sibling = root_child;
                    (*node).prev = Some(root_ptr);
                    if let Some(child) = root_child {
                        (*child.as_ptr()).prev = Some(node_ptr);
                    }
                    (*root_ptr.as_ptr()).child = Some(node_ptr);
                }
            } else {
                // Heap is empty (shouldn't happen, but handle gracefully)
                self.root = Some(node_ptr);
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
    /// 3. Update child/sibling pointers accordingly
    /// 4. The smaller root becomes the new root
    ///
    /// This is trivially O(1): we only need one comparison and a few pointer updates.
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
        unsafe {
            let self_root = self.root.unwrap();
            let other_root = other.root.unwrap();

            // Compare roots: smaller priority becomes parent (heap property)
            if (*other_root.as_ptr()).priority < (*self_root.as_ptr()).priority {
                // Other root has smaller priority: it becomes the new root
                // Self root becomes a child of other root
                let self_child = (*other_root.as_ptr()).child;
                (*self_root.as_ptr()).sibling = self_child; // Self root's sibling is other's old first child
                (*self_root.as_ptr()).prev = Some(other_root); // Self root's parent is other root
                if let Some(child) = self_child {
                    (*child.as_ptr()).prev = Some(self_root); // Update other's old first child
                }
                (*other_root.as_ptr()).child = Some(self_root); // Other root's first child is now self root
                self.root = Some(other_root);
            } else {
                // Self root has smaller or equal priority: it stays root
                // Other root becomes a child of self root
                let self_child = (*self_root.as_ptr()).child;
                (*other_root.as_ptr()).sibling = self_child; // Other root's sibling is self's old first child
                (*other_root.as_ptr()).prev = Some(self_root); // Other root's parent is self root
                if let Some(child) = self_child {
                    (*child.as_ptr()).prev = Some(other_root); // Update self's old first child
                }
                (*self_root.as_ptr()).child = Some(other_root); // Self root's first child is now other root
            }

            // Update length and mark other as empty (prevent double-free)
            self.len += other.len;
            other.root = None;
            other.len = 0;
        }
    }
}

impl<T, P: Ord> PairingHeap<T, P> {
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
    unsafe fn merge_pairs(&self, first: NonNull<Node<T, P>>) -> NonNull<Node<T, P>> {
        // Base case: only one child, no pairing needed
        if (*first.as_ptr()).sibling.is_none() {
            return first;
        }

        // First pass: pair adjacent children and merge each pair
        // This reduces the number of trees by approximately half
        let mut pairs = Vec::new();
        let mut current = Some(first);

        while let Some(node) = current {
            let sibling = (*node.as_ptr()).sibling;

            // Disconnect node from sibling list (we're reorganizing)
            (*node.as_ptr()).sibling = None;
            (*node.as_ptr()).prev = None;

            if let Some(sib) = sibling {
                // We have a pair: node and its sibling
                // Get the next node before we disconnect
                let next = (*sib.as_ptr()).sibling;
                (*sib.as_ptr()).sibling = None;
                (*sib.as_ptr()).prev = None;

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

        // Second pass: merge pairs from right to left
        // Starting with the last merged tree, merge each remaining pair into it
        // This right-to-left ordering is crucial for balanced structure
        let mut result = pairs.pop().unwrap();
        while let Some(pair) = pairs.pop() {
            // Merge: smaller priority becomes parent (heap property)
            result = self.merge_nodes(pair, result);
        }

        result
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
    unsafe fn merge_nodes(
        &self,
        a: NonNull<Node<T, P>>,
        b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        if (*a.as_ptr()).priority < (*b.as_ptr()).priority {
            // Node a has smaller priority: it becomes parent
            // Node b becomes a child of a
            let a_child = (*a.as_ptr()).child;
            // Make b the first child of a (b's sibling is a's old first child)
            (*b.as_ptr()).sibling = a_child;
            (*b.as_ptr()).prev = Some(a); // b's parent is a
            if let Some(child) = a_child {
                // Update a's old first child to point back to b
                (*child.as_ptr()).prev = Some(b);
            }
            (*a.as_ptr()).child = Some(b); // a's first child is now b
            a // Return a as the new root
        } else {
            // Node b has smaller or equal priority: it becomes parent
            // Node a becomes a child of b
            let b_child = (*b.as_ptr()).child;
            // Make a the first child of b (a's sibling is b's old first child)
            (*a.as_ptr()).sibling = b_child;
            (*a.as_ptr()).prev = Some(b); // a's parent is b
            if let Some(child) = b_child {
                // Update b's old first child to point back to a
                (*child.as_ptr()).prev = Some(a);
            }
            (*b.as_ptr()).child = Some(a); // b's first child is now a
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
    unsafe fn cut_node(&mut self, node: NonNull<Node<T, P>>) {
        let prev_opt = match (*node.as_ptr()).prev {
            Some(p) => p,
            None => return, // Node has no parent (already root or orphaned)
        };
        let prev = prev_opt.as_ptr();

        // Determine if prev is the parent or a sibling
        // Check: if prev's first child is this node, then prev is the parent
        if (*prev).child == Some(node) {
            // Case 1: Node is the first child (prev is the parent)
            // Remove node from parent's child list
            // Parent's first child becomes node's sibling
            (*prev).child = (*node.as_ptr()).sibling;
            if let Some(sibling) = (*node.as_ptr()).sibling {
                // Update sibling's prev pointer (now points to parent)
                (*sibling.as_ptr()).prev = Some(prev_opt);
            }
        } else {
            // Case 2: Node is not the first child (prev is previous sibling)
            // Remove node from sibling list
            // Previous sibling's next sibling becomes node's sibling
            (*prev).sibling = (*node.as_ptr()).sibling;
            if let Some(sibling) = (*node.as_ptr()).sibling {
                // Update next sibling's prev pointer (now points to previous sibling)
                (*sibling.as_ptr()).prev = Some(prev_opt);
            }
        }

        // Clear node's links (it's now disconnected from the tree)
        (*node.as_ptr()).sibling = None;
        (*node.as_ptr()).prev = None;
    }

    /// Recursively frees a node and all its descendants
    ///
    /// **Time Complexity**: O(n) where n is the number of nodes in the subtree
    ///
    /// **Algorithm**:
    /// - Recursively free all children (via child pointer)
    /// - Recursively free all siblings (via sibling pointer)
    /// - Free the node itself
    ///
    /// This is used in Drop implementation to clean up all nodes when the heap is dropped.
    /// The recursive structure follows the tree: first free all descendants, then the node.
    unsafe fn free_node(node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();
        // Free all children first (recursive)
        if let Some(child) = (*node_ptr).child {
            Self::free_node(child);
        }
        // Free all siblings (recursive)
        // Note: siblings form a linked list, so this traverses the sibling chain
        if let Some(sibling) = (*node_ptr).sibling {
            Self::free_node(sibling);
        }
        // Finally, free this node itself
        drop(Box::from_raw(node_ptr));
    }
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
