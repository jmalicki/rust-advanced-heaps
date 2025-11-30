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
use std::mem;
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

        // Update minimum pointer
        let should_update_min = match &self.min {
            Some(min_weak) => {
                if let Some(min_rc) = min_weak.upgrade() {
                    node.borrow().priority < min_rc.borrow().priority
                } else {
                    true // Min was invalid, update it
                }
            }
            None => true, // No minimum yet
        };

        if should_update_min {
            self.min = Some(Rc::downgrade(&node));
        }

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

        let degree = min_rc.borrow().degree;

        // Remove minimum tree from trees array
        if degree < self.trees.len() {
            self.trees[degree] = None;
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
            // Children are linked in a sibling list in decreasing degree order
            // We need to reverse the list and clear parent links
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
            for child_node in children {
                let child_degree = child_node.borrow().degree;

                // Ensure child_heap.trees array is large enough
                while child_heap.trees.len() <= child_degree {
                    child_heap.trees.push(None);
                }
                // Place child tree at its degree slot
                child_heap.trees[child_degree] = Some(child_node);
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
    fn bubble_up(&mut self, node: NodeRef<T, P>) {
        let mut current = node;

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

            // Swap priorities and items between current and parent
            {
                let mut current_ref = current.borrow_mut();
                let mut parent_ref = parent.borrow_mut();
                mem::swap(&mut current_ref.priority, &mut parent_ref.priority);
                mem::swap(&mut current_ref.item, &mut parent_ref.item);
            }

            // Move up to parent
            current = parent;
        }

        // Update minimum pointer if needed
        self.update_min_if_needed(&current);
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
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
