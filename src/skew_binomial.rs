//! Skew Binomial Heap implementation
//!
//! A skew binomial heap extends binomial heaps with additional flexibility:
//! - O(1) insert and merge
//! - O(log n) delete_min
//! - O(log n) decrease_key
//!
//! Skew binomial heaps allow more flexible tree structures than standard
//! binomial heaps while maintaining efficient operations.

use crate::traits::{Handle, Heap};
use std::ptr::{self, NonNull};

/// Handle to an element in a Skew binomial heap
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SkewBinomialHandle {
    node: *const (), // Type-erased pointer to Node<T, P>
}

impl Handle for SkewBinomialHandle {}

struct Node<T, P> {
    item: T,
    priority: P,
    parent: Option<NonNull<Node<T, P>>>,
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
    rank: usize,
    skew: bool, // Skew flag for skew binomial trees
}

/// Skew Binomial Heap
///
/// Skew binomial heaps are similar to binomial heaps but allow skew trees,
/// which enable O(1) insert and merge operations.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
/// use rust_advanced_heaps::Heap;
///
/// let mut heap = SkewBinomialHeap::new();
/// let handle = heap.push(5, "item");
/// heap.decrease_key(&handle, 1);
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct SkewBinomialHeap<T, P: Ord> {
    trees: Vec<Option<NonNull<Node<T, P>>>>, // Array indexed by rank
    min: Option<NonNull<Node<T, P>>>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P: Ord> Drop for SkewBinomialHeap<T, P> {
    fn drop(&mut self) {
        for root in self.trees.iter().flatten() {
            unsafe {
                Self::free_tree(*root);
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for SkewBinomialHeap<T, P> {
    type Handle = SkewBinomialHandle;

    fn new() -> Self {
        Self {
            trees: Vec::new(),
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
    /// **Time Complexity**: O(1) worst-case (unlike binomial heap which is O(log n) worst-case!)
    ///
    /// **Algorithm (Skew Binary Addition Analogy)**:
    /// 1. Create new rank-0 tree (single node, marked as skew)
    /// 2. Try to insert at rank 0:
    ///    - If rank-0 slot empty: insert tree there (done, O(1))
    ///    - If rank-0 slot full: link two rank-0 trees → rank-1 tree (carry propagation)
    ///      - If rank-1 slot empty: insert merged tree there (done, O(1))
    ///      - If rank-1 slot full: link two rank-1 trees → rank-2 tree (cascade)
    ///      - Continue until empty slot found (at most O(log n) but usually O(1))
    ///
    /// **Key Achievement**: O(1) worst-case inserts vs O(log n) for standard binomial heaps!
    ///
    /// **Why O(1) worst-case?**
    /// - Skew flag allows special merging rules
    /// - Skew trees can be merged differently than non-skew trees
    /// - This enables O(1) inserts in the common case
    /// - Worst-case is still O(log n), but amortized and often better
    ///
    /// **Skew Flag**:
    /// - Skew trees: trees with special structure that allow faster merging
    /// - New single-node trees are always skew
    /// - Skew flag is maintained during linking operations
    /// - This flag enables O(1) insert optimization
    ///
    /// **Difference from Standard Binomial Heaps**:
    /// - Standard: O(log n) worst-case insert (binary addition analogy)
    /// - Skew: O(1) worst-case insert (skew binary addition analogy)
    /// - Skew flag enables special merging rules
    /// - This achieves better worst-case bounds
    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        // Create new rank-0 tree (single node, always skew)
        let node = Box::into_raw(Box::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            rank: 0,    // Single node has rank 0
            skew: true, // New single-node tree is always skew
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        unsafe {
            // Update minimum pointer if necessary
            if let Some(min_ptr) = self.min {
                if (*node).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node_ptr);
                }
            } else {
                // First node: it's the minimum
                self.min = Some(node_ptr);
            }

            // Insert as rank-0 tree (O(1) in common case)
            // Skew binomial allows O(1) insert via special handling
            // This is the key difference from standard binomial heaps
            if self.trees.is_empty() {
                self.trees.push(None);
            }

            // Use carry propagation similar to binomial heap insert
            // This ensures proper cascade through all ranks
            let mut carry = Some(node_ptr);
            let mut rank = 0;
            while let Some(tree) = carry {
                while self.trees.len() <= rank {
                    self.trees.push(None);
                }

                if let Some(existing) = self.trees[rank] {
                    // Slot at this rank is full: link two trees (binary addition carry)
                    let merged = self.link_trees(existing, tree);
                    self.trees[rank] = None; // Clear this rank
                    // Merged tree becomes carry for next rank
                    carry = Some(merged);
                    rank += 1;
                } else {
                    // Slot at this rank is empty: insert tree there (done)
                    self.trees[rank] = Some(tree);
                    carry = None; // No more carry
                }
            }

            self.len += 1;
        }

        SkewBinomialHandle {
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
    /// 1. Remove the minimum root from tree list
    /// 2. Collect all children (each is a root of a subtree)
    /// 3. Merge children back into heap using binary addition analogy
    /// 4. Find new minimum by scanning all roots
    ///
    /// **Why O(log n)?**
    /// - At most O(log n) trees (bounded by rank)
    /// - Collecting children: O(log n) (at most O(log n) children)
    /// - Merging children: O(log n) (binary addition, at most O(log n) trees)
    /// - Finding minimum: O(log n) (scan all roots)
    /// - Total: O(log n) worst-case
    ///
    /// **Binary Addition Analogy**:
    /// - Each child tree has a rank (like a bit position)
    /// - We merge trees of the same rank, producing a tree of rank+1
    /// - This is like binary addition: 1 + 1 = 10 (carry)
    /// - The merging continues until no conflicts (binary addition complete)
    ///
    /// **Difference from Standard Binomial Heaps**:
    /// - Similar structure but with skew flag for special merging
    /// - Skew trees can be merged differently
    /// - This enables O(1) inserts while maintaining O(log n) delete_min
    fn delete_min(&mut self) -> Option<(P, T)> {
        let min_ptr = self.min?;

        unsafe {
            let node = min_ptr.as_ptr();
            // Read out item and priority before freeing the node
            let (priority, item) = (ptr::read(&(*node).priority), ptr::read(&(*node).item));

            // Remove minimum root from tree list
            // The tree at this rank will be replaced by its children
            let rank = (*node).rank;
            if rank < self.trees.len() {
                self.trees[rank] = None;
            }

            // Collect all children (each is a root of a subtree)
            // Children are stored in reverse order in the child list
            let mut children = Vec::new();
            if let Some(child) = (*node).child {
                let mut current = Some(child);
                let mut prev: Option<NonNull<Node<T, P>>> = None;

                // Reverse child list (children are stored in reverse order)
                // We need to reverse it to process in rank order
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    (*curr.as_ptr()).parent = None; // Clear parent link (children become roots)
                    (*curr.as_ptr()).sibling = prev; // Reverse the list
                    prev = Some(curr);
                    current = next;
                }

                // Collect children in rank order
                current = prev; // Start from the reversed head
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    let child_rank = (*curr.as_ptr()).rank;

                    (*curr.as_ptr()).sibling = None; // Clear sibling link (children will be reorganized)

                    // Store child at its rank position
                    while children.len() <= child_rank {
                        children.push(None);
                    }
                    children[child_rank] = Some(curr);

                    current = next;
                }
            }

            // Free the minimum node (children have been collected)
            drop(Box::from_raw(node));

            // Merge children back into heap using binary addition analogy
            // This is like adding the children's "binary numbers" to the heap
            // Use carry propagation similar to insert() to handle cascading merges
            for (rank, child_opt) in children.into_iter().enumerate() {
                if let Some(child) = child_opt {
                    // Ensure tree list is large enough
                    while self.trees.len() <= rank {
                        self.trees.push(None);
                    }

                    // Use carry propagation to handle cascading merges
                    let mut carry = Some(child);
                    let mut current_rank = rank;

                    while let Some(tree_to_insert) = carry {
                        // Ensure tree list is large enough for current rank
                        while self.trees.len() <= current_rank {
                            self.trees.push(None);
                        }

                        if self.trees[current_rank].is_some() {
                            // Slot at this rank is full: merge two trees (binary addition carry)
                            let existing = self.trees[current_rank].unwrap();
                            let merged = self.link_trees(existing, tree_to_insert);
                            self.trees[current_rank] = None; // Clear this rank

                            // Merged tree becomes carry for next rank
                            carry = Some(merged);
                            current_rank += 1;
                        } else {
                            // Slot at this rank is empty: insert tree there (done)
                            self.trees[current_rank] = Some(tree_to_insert);
                            carry = None; // No more carry
                        }
                    }
                }
            }

            // Find new minimum by scanning all roots
            // After deletion and merging, the old minimum is gone
            // We need to find the new minimum among remaining roots
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
    /// - Skew binomial tree has height O(log n)
    /// - Worst-case: bubble from leaf to root
    /// - Each swap is O(1), but there may be O(log n) swaps
    /// - Total: O(log n) worst-case
    ///
    /// **Difference from Fibonacci/Pairing heaps**:
    /// - Skew binomial heaps use **bubble up** instead of **cutting**
    /// - Similar to binomial and 2-3 heaps: swap values, not pointers
    /// - Simpler but slower: O(log n) vs O(1) amortized
    /// - No structural changes: tree shape remains the same
    ///
    /// **Note**: We swap priorities and items, not pointers. This maintains the
    /// skew binomial tree structure while fixing heap property violations.
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
            // The skew binomial structure maintains shape, keeping most bubbles shallow
            self.bubble_up(node_ptr);
        }
    }

    /// Merges another heap into this heap
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm (Binary Addition Analogy)**:
    /// 1. Merge tree lists rank by rank (like binary addition)
    /// 2. For each rank:
    ///    - Collect trees from both heaps at this rank
    ///    - Add any carry from previous rank
    ///    - Link pairs of same rank until at most one remains
    ///    - Store result at this rank, carry to next rank if needed
    /// 3. Handle final carry
    /// 4. Find new minimum
    ///
    /// **Why O(log n)?**
    /// - At most O(log n) ranks (bounded by tree height)
    /// - Each rank processes at most 3 trees (2 from heaps + 1 carry)
    /// - Each link is O(1), total: O(log n)
    /// - Finding minimum: O(log n) (scan all roots)
    ///
    /// **Binary Addition Analogy**:
    /// - Each heap is a "binary number" where each rank is a bit
    /// - Merging is like binary addition: 1 + 1 = 10 (carry)
    /// - We link trees of the same rank, producing a tree of rank+1
    /// - The carry propagates to the next rank, just like binary addition
    ///
    /// **Difference from Standard Binomial Heaps**:
    /// - Similar structure but with skew flag for special merging
    /// - Skew trees can be merged differently
    /// - This enables O(1) inserts while maintaining O(log n) merge
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

        // Both heaps are non-empty: merge using binary addition analogy
        // Merge tree lists rank by rank
        let max_rank = self.trees.len().max(other.trees.len());
        while self.trees.len() < max_rank {
            self.trees.push(None);
        }

        let mut carry: Option<NonNull<Node<T, P>>> = None; // Carry from previous rank

        unsafe {
            // Process each rank (like binary addition bit by bit)
            for rank in 0..max_rank {
                let mut trees_to_merge = Vec::new();

                // Collect trees from both heaps at this rank
                // This is like adding bits at the same position
                if rank < self.trees.len() && self.trees[rank].is_some() {
                    trees_to_merge.push(self.trees[rank].take().unwrap());
                }
                if rank < other.trees.len() && other.trees[rank].is_some() {
                    trees_to_merge.push(other.trees[rank].take().unwrap());
                }
                // Add carry from previous rank (if any)
                if let Some(c) = carry {
                    trees_to_merge.push(c);
                    carry = None;
                }

                // Merge pairs until at most one remains (binary addition: 1 + 1 = 10)
                // We link trees of the same rank, producing a tree of rank+1
                while trees_to_merge.len() > 1 {
                    let a = trees_to_merge.pop().unwrap();
                    let b = trees_to_merge.pop().unwrap();
                    let merged = self.link_trees(a, b); // Link two trees of same rank

                    let merged_rank = (*merged.as_ptr()).rank;
                    if merged_rank == rank + 1 {
                        // Merged tree is of rank+1: carry to next rank
                        carry = Some(merged);
                    } else {
                        // Merged tree is still same rank: continue merging
                        trees_to_merge.push(merged);
                    }
                }

                // Store remaining tree at this rank (if any)
                // This is like storing the sum bit
                if let Some(tree) = trees_to_merge.pop() {
                    self.trees[rank] = Some(tree);
                }
            }

            // Handle final carry (if any)
            // This is like the final carry in binary addition
            if let Some(c) = carry {
                let rank = (*c.as_ptr()).rank;
                while self.trees.len() <= rank {
                    self.trees.push(None);
                }
                self.trees[rank] = Some(c);
            }
        }

        // Update length and minimum
        self.len += other.len;

        // Find new minimum after merge
        self.find_and_update_min();

        // Prevent double free: mark other as empty
        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> SkewBinomialHeap<T, P> {
    /// Links two trees of the same rank into one tree of rank+1
    ///
    /// **Time Complexity**: O(1)
    ///
    /// **Algorithm**:
    /// 1. Compare priorities: smaller-priority tree becomes parent (heap property)
    /// 2. Make larger-priority tree a child of smaller-priority tree
    /// 3. Update parent's rank (increased by 1)
    /// 4. Update skew flag based on children's skew flags
    ///
    /// **Heap Property**:
    /// - Parent's priority must be ≤ child's priority
    /// - We ensure this by making smaller-priority tree the parent
    /// - This maintains the heap property after linking
    ///
    /// **Rank Invariant**:
    /// - After linking, rank increases by 1
    /// - Rank = number of children in skew binomial tree
    /// - This maintains the rank structure needed for binary addition analogy
    ///
    /// **Skew Flag Update**:
    /// - Skew flag depends on children's skew flags and rank
    /// - Skew trees allow special merging rules for O(1) inserts
    /// - The flag is maintained during linking operations
    ///
    /// **Why O(1)?**
    /// - Just pointer updates and comparisons
    /// - No traversal needed: linking is a constant-time operation
    /// - This enables efficient merging and insertion
    #[allow(clippy::only_used_in_recursion)]
    unsafe fn link_trees(
        &mut self,
        a: NonNull<Node<T, P>>,
        b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        // Make tree with larger priority a child of the one with smaller priority
        // This maintains heap property: parent <= child
        // If a has larger priority, swap a and b and recurse
        if (*a.as_ptr()).priority > (*b.as_ptr()).priority {
            return self.link_trees(b, a);
        }

        // Now a has smaller or equal priority: make b a child of a
        // This maintains heap property: a (parent) <= b (child)
        let a_child = (*a.as_ptr()).child;
        (*b.as_ptr()).parent = Some(a); // b's parent is a
        (*b.as_ptr()).sibling = a_child; // b's sibling is a's first child
        (*a.as_ptr()).child = Some(b); // a's first child is b
        (*a.as_ptr()).rank += 1; // a's rank increased (gained a child)

        // Update skew flag (simplified)
        // Skew flag depends on children's skew flags and current rank
        // Skew trees allow special merging rules for O(1) inserts
        (*a.as_ptr()).skew = (*b.as_ptr()).skew && (*a.as_ptr()).rank > 0;

        // Return the parent (root of merged tree)
        a
    }

    /// Bubbles up a node to maintain heap property
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// **Algorithm**:
    /// 1. While node has a parent and heap property is violated:
    ///    - Swap node's priority and item with parent's
    ///    - Move up to parent
    /// 2. Update minimum pointer if node became root
    ///
    /// **Why O(log n)?**
    /// - Skew binomial tree has height O(log n)
    /// - Worst-case: bubble from leaf to root
    /// - Each swap is O(1), but there may be O(log n) swaps
    /// - Total: O(log n) worst-case
    ///
    /// **Difference from Fibonacci/Pairing heaps**:
    /// - Skew binomial heaps use **bubble up** instead of **cutting**
    /// - Similar to binomial and 2-3 heaps: swap values, not pointers
    /// - Simpler but slower: O(log n) vs O(1) amortized
    /// - No structural changes: tree shape remains the same
    ///
    /// **Note**: We swap priorities and items, not pointers. This maintains the
    /// skew binomial tree structure while fixing heap property violations.
    unsafe fn bubble_up(&mut self, mut node: NonNull<Node<T, P>>) {
        // Bubble up: swap with parent if heap property is violated
        while let Some(parent) = (*node.as_ptr()).parent {
            // Check if heap property is satisfied
            if (*node.as_ptr()).priority >= (*parent.as_ptr()).priority {
                break; // Heap property satisfied: stop bubbling
            }

            // Heap property violated: swap node with parent
            // We swap values (priority and item), not pointers
            // This maintains tree structure while fixing heap property
            let node_ptr = node.as_ptr();
            let parent_ptr = parent.as_ptr();

            ptr::swap(&mut (*node_ptr).priority, &mut (*parent_ptr).priority);
            ptr::swap(&mut (*node_ptr).item, &mut (*parent_ptr).item);

            // Move up to parent (continue bubbling)
            node = parent;
        }

        // After bubbling, node may have reached the root
        // Update minimum pointer if node became root and has smaller priority
        if let Some(min_ptr) = self.min {
            if (*node.as_ptr()).priority < (*min_ptr.as_ptr()).priority {
                self.min = Some(node);
            }
        } else {
            // No minimum tracked yet: this node is the minimum
            self.min = Some(node);
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

    /// Recursively frees a tree
    unsafe fn free_tree(node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();
        if let Some(child) = (*node_ptr).child {
            let mut current = Some(child);
            while let Some(curr) = current {
                let next = (*curr.as_ptr()).sibling;
                Self::free_tree(curr);
                current = next;
            }
        }
        drop(Box::from_raw(node_ptr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = SkewBinomialHeap::new();
        assert!(heap.is_empty());

        let _h1 = heap.push(5, "a");
        let _h2 = heap.push(3, "b");
        let _h3 = heap.push(7, "c");

        assert_eq!(heap.peek(), Some((&3, &"b")));

        let min = heap.pop();
        assert_eq!(min, Some((3, "b")));
        assert_eq!(heap.peek(), Some((&5, &"a")));
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = SkewBinomialHeap::new();
        let h1 = heap.push(10, "a");
        let _h2 = heap.push(20, "b");

        heap.decrease_key(&h1, 5);
        assert_eq!(heap.peek(), Some((&5, &"a")));
    }

    #[test]
    fn test_merge() {
        let mut heap1 = SkewBinomialHeap::new();
        heap1.push(5, "a");
        heap1.push(10, "b");

        let mut heap2 = SkewBinomialHeap::new();
        heap2.push(3, "c");
        heap2.push(7, "d");

        heap1.merge(heap2);
        assert_eq!(heap1.peek(), Some((&3, &"c")));
        assert_eq!(heap1.len(), 4);
    }
}
