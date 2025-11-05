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

use crate::traits::{Handle, Heap};
use smallvec::SmallVec;
use std::ptr::{self, NonNull};

/// Handle to an element in a Binomial heap
///
/// Note: This handle is tied to a specific heap instance. Using it with a different
/// heap or after the heap is dropped is undefined behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BinomialHandle {
    node: *const (), // Type-erased pointer to Node<T, P>
}

impl Handle for BinomialHandle {}

/// Internal node structure for binomial heap
///
/// Each node maintains:
/// - `item` and `priority`: The data stored in the heap
/// - `parent`: Pointer to parent node (None if root)
/// - `child`: Pointer to first child (None if leaf)
/// - `sibling`: Next sibling in parent's child list (None if last child)
/// - `degree`: Number of children (critical for merge operations)
///
/// **Binomial Tree Structure**: Nodes form binomial trees where a node of degree k
/// has exactly k children with degrees 0, 1, 2, ..., k-1. This ensures the tree
/// has exactly 2ᵏ nodes.
struct Node<T, P> {
    item: T,
    priority: P,
    /// Parent node (None if root)
    parent: Option<NonNull<Node<T, P>>>,
    /// First child in child list (None if leaf)
    child: Option<NonNull<Node<T, P>>>,
    /// Next sibling in parent's child list (None if last child)
    sibling: Option<NonNull<Node<T, P>>>,
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
    #[allow(clippy::type_complexity)]
    trees: SmallVec<[Option<NonNull<Node<T, P>>>; 32]>, // Array indexed by degree, stack-allocated for small heaps
    min: Option<NonNull<Node<T, P>>>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P: Ord> Drop for BinomialHeap<T, P> {
    fn drop(&mut self) {
        // Free all trees
        for root in self.trees.iter().flatten() {
            unsafe {
                Self::free_tree(*root);
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for BinomialHeap<T, P> {
    type Handle = BinomialHandle;

    fn new() -> Self {
        Self {
            trees: SmallVec::new(),
            min: None,
            len: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    fn is_empty(&self) -> bool {
        self.min.is_none()
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
        let node = Box::into_raw(Box::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            degree: 0, // Single node has degree 0
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        // Update minimum pointer (O(1) if we track it separately)
        if let Some(min_ptr) = self.min {
            unsafe {
                if (*node).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node_ptr);
                }
            }
        } else {
            // First node: it's the minimum
            self.min = Some(node_ptr);
        }

        // Merge this single-node tree into the heap
        // This is like binary addition: if slot[degree] is occupied, link and carry
        let mut carry = Some(node_ptr); // The tree we're trying to place
        let mut degree = 0; // Start at degree 0

        // Carry propagation: continue until we place the tree somewhere
        while carry.is_some() {
            // Ensure trees array is large enough
            if degree >= self.trees.len() {
                self.trees.push(None);
            }

            if self.trees[degree].is_none() {
                // Slot is empty: place tree here (like placing a bit)
                self.trees[degree] = carry;
                carry = None; // Done: no more carry
            } else {
                // Slot is occupied: link the two trees (like adding bits with carry)
                unsafe {
                    let existing = self.trees[degree].unwrap();
                    let new_tree = carry.unwrap();
                    // Link two trees of same degree to produce tree of degree+1
                    carry = Some(self.link_trees(existing, new_tree));
                    // Clear slot (we linked the trees)
                    self.trees[degree] = None;
                    // Move to next degree (carry propagation)
                    degree += 1;
                }
            }
        }

        self.len += 1;
        BinomialHandle {
            node: node_ptr.as_ptr() as *const (),
        }
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        self.min.map(|min_ptr| unsafe {
            let node = min_ptr.as_ptr();
            (&(*node).priority, &(*node).item)
        })
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
        let min_ptr = self.min?;

        unsafe {
            let node = min_ptr.as_ptr();
            // Read out item and priority before freeing the node
            let (priority, item) = (ptr::read(&(*node).priority), ptr::read(&(*node).item));

            // Remove minimum tree from trees array
            // The minimum root is at trees[degree]
            let degree = (*node).degree;
            if degree < self.trees.len() {
                self.trees[degree] = None; // Remove tree from this degree slot
            }

            // Collect children of the minimum root
            // Each child is itself a binomial tree (smaller degree)
            let mut child_heap = BinomialHeap::new();
            if let Some(child) = (*node).child {
                // Children are linked in a sibling list
                // We need to reverse the list to get correct order
                let mut current = Some(child);
                let mut prev: Option<NonNull<Node<T, P>>> = None;

                // Reverse the child list (children are linked in decreasing degree order)
                // Reversing ensures we process them in correct order
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    (*curr.as_ptr()).parent = None; // Clear parent link
                    (*curr.as_ptr()).sibling = prev; // Reverse link
                    prev = Some(curr);
                    current = next;
                }

                // Add each child tree to the temporary heap
                // Each child is a root of a binomial tree
                current = prev; // Now reversed list
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    let child_degree = (*curr.as_ptr()).degree;

                    // Reset sibling to break link (each child becomes independent)
                    (*curr.as_ptr()).sibling = None;

                    // Ensure child_heap.trees array is large enough
                    while child_heap.trees.len() <= child_degree {
                        child_heap.trees.push(None);
                    }
                    // Place child tree at its degree slot
                    child_heap.trees[child_degree] = Some(curr);

                    current = next;
                }
            }

            // Free the minimum node (children have been collected)
            drop(Box::from_raw(node));

            // Merge the child heap back into the main heap
            // This uses the same merge algorithm as regular merge
            // Carry propagation ensures at most one tree per degree
            self.merge_trees(&mut child_heap);

            // Find new minimum by scanning all root trees
            // This is O(log n) since there are at most O(log n) trees
            self.find_and_update_min();

            self.len -= 1;
            Some((priority, item))
        }
    }

    /// Decreases the priority of an element
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Precondition**: `new_priority < current_priority` (undefined behavior otherwise)
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
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) {
        let node_ptr = unsafe { NonNull::new_unchecked(handle.node as *mut Node<T, P>) };

        unsafe {
            let node = node_ptr.as_ptr();

            // Safety check: new priority must actually be less
            if new_priority >= (*node).priority {
                return; // No-op if priority didn't decrease
            }

            // Update the priority value
            (*node).priority = new_priority;

            // Bubble up: swap with parent if heap property is violated
            // This maintains heap property by moving smaller priorities upward
            // Unlike Fibonacci/pairing heaps, we don't cut - we swap values
            self.bubble_up(node_ptr);
        }
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
        // This is the core operation: merge by degree with carry propagation
        self.merge_trees(&mut other);

        // Update minimum pointer after merge
        // Need to scan all roots to find new minimum (O(log n))
        self.find_and_update_min();

        // Update length and mark other as empty (prevent double-free)
        self.len += other.len;
        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> BinomialHeap<T, P> {
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
    #[allow(clippy::only_used_in_recursion)]
    unsafe fn link_trees(
        &self,
        mut a: NonNull<Node<T, P>>,
        mut b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        // Ensure smaller priority becomes parent (heap property)
        // If a has larger priority, swap roles
        if (*a.as_ptr()).priority > (*b.as_ptr()).priority {
            std::mem::swap(&mut a, &mut b);
        }

        // Link b as a child of a (a has smaller priority)
        // a becomes the root of the new B_{k+1} tree
        let a_child = (*a.as_ptr()).child;
        (*b.as_ptr()).parent = Some(a); // b's parent is now a
        (*b.as_ptr()).sibling = a_child; // b's sibling is a's old first child
        (*a.as_ptr()).child = Some(b); // a's first child is now b
        (*a.as_ptr()).degree += 1; // a's degree increased by 1

        // Return the root of the new tree (a, since a has smaller priority)
        a
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
        let mut carry: Option<NonNull<Node<T, P>>> = None;

        // Process each degree from 0 to max_degree
        for degree in 0..max_degree {
            let mut trees = Vec::new();

            // Step 1: Collect trees from both heaps at this degree
            if degree < self.trees.len() {
                if let Some(tree) = self.trees[degree] {
                    trees.push(tree);
                    self.trees[degree] = None; // Remove from slot
                }
            }

            if degree < other.trees.len() {
                if let Some(tree) = other.trees[degree] {
                    trees.push(tree);
                    other.trees[degree] = None; // Remove from slot
                }
            }

            // Step 2: Add carry from previous degree if present
            if let Some(c) = carry {
                trees.push(c);
                carry = None; // Consume carry
            }

            // Step 3: Link pairs of trees until at most one remains
            // This is like adding bits: 0+0=0, 0+1=1, 1+1=10 (carry)
            while trees.len() > 1 {
                unsafe {
                    let a = trees.pop().unwrap();
                    let b = trees.pop().unwrap();
                    // Link two trees of same degree to produce tree of degree+1
                    let linked = self.link_trees(a, b);

                    // Check if linked tree has correct degree for this slot
                    if (*linked.as_ptr()).degree == degree + 1 {
                        // Linked tree has degree+1: it becomes carry for next degree
                        carry = Some(linked);
                    } else {
                        // Linked tree has same degree: continue linking
                        trees.push(linked);
                    }
                }
            }

            // Step 4: Place remaining tree (if any) at this degree slot
            if let Some(tree) = trees.pop() {
                unsafe {
                    if (*tree.as_ptr()).degree == degree {
                        // Tree has correct degree: place it here
                        self.trees[degree] = Some(tree);
                    } else {
                        // Tree has higher degree: it becomes carry
                        carry = Some(tree);
                    }
                }
            }
        }

        // Step 5: Handle final carry (like overflow in binary addition)
        // If carry remains after processing all degrees, we need a new slot
        if let Some(c) = carry {
            let degree = unsafe { (*c.as_ptr()).degree };
            // Ensure trees array is large enough
            while self.trees.len() <= degree {
                self.trees.push(None);
            }
            // Place carry tree at its degree slot
            self.trees[degree] = Some(c);
        }
    }

    /// Bubbles up a node to maintain heap property
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**:
    /// 1. While node has a parent and heap property is violated:
    ///    - Swap node's priority and item with parent's
    ///    - Move up to parent
    /// 2. Update minimum pointer if node became root and has smaller priority
    ///
    /// **Why O(log n)?**
    /// - Binomial tree has height O(log n)
    /// - We may need to traverse from leaf to root
    /// - Each swap is O(1), but there may be O(log n) swaps
    ///
    /// **Difference from Fibonacci/Pairing heaps**:
    /// - Binomial heaps **swap values** instead of **cutting** nodes
    /// - Simpler but slower: O(log n) vs O(1) amortized
    /// - No structural changes: tree shape remains the same
    ///
    /// **Note**: We swap priorities and items, not pointers. This maintains the
    /// binomial tree structure while fixing heap property violations.
    unsafe fn bubble_up(&mut self, mut node: NonNull<Node<T, P>>) {
        // Bubble up: swap with parent if heap property is violated
        while let Some(parent) = (*node.as_ptr()).parent {
            // Check if heap property is satisfied
            if (*node.as_ptr()).priority >= (*parent.as_ptr()).priority {
                break; // Heap property satisfied: stop bubbling
            }

            // Heap property violated: swap node with parent
            let node_ptr = node.as_ptr();
            let parent_ptr = parent.as_ptr();

            // Swap priorities and items (not pointers!)
            // This maintains tree structure while fixing heap property
            ptr::swap(&mut (*node_ptr).priority, &mut (*parent_ptr).priority);
            ptr::swap(&mut (*node_ptr).item, &mut (*parent_ptr).item);

            // Move up to parent (continue bubbling)
            node = parent;
        }

        // After bubbling, node may have reached the root
        // Update minimum pointer if node became root and has smaller priority
        unsafe {
            if let Some(min_ptr) = self.min {
                if (*node.as_ptr()).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node);
                }
            } else {
                // No minimum tracked yet: this node is the minimum
                self.min = Some(node);
            }
        }
    }

    /// Finds and updates the minimum pointer
    fn find_and_update_min(&mut self) {
        self.min = None;
        for root in self.trees.iter().flatten() {
            unsafe {
                if self.min.is_none()
                    || (*root.as_ptr()).priority < (*self.min.unwrap().as_ptr()).priority
                {
                    self.min = Some(*root);
                }
            }
        }
    }

    /// Recursively frees a binomial tree
    unsafe fn free_tree(node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();
        if let Some(child) = (*node_ptr).child {
            Self::free_tree(child);
        }
        if let Some(sibling) = (*node_ptr).sibling {
            Self::free_tree(sibling);
        }
        drop(Box::from_raw(node_ptr));
    }
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
