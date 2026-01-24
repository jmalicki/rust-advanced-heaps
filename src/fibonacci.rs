//! Fibonacci Heap implementation using RefBox for memory-efficient weak references
//!
//! A Fibonacci heap is a data structure for priority queue operations with:
//! - O(1) amortized insert, decrease_key, and merge
//! - O(log n) amortized delete_min
//!
//! This implementation uses RefBox/Weak references for reduced memory overhead:
//! - Owned references (RefBox) flow from roots to children (ownership)
//! - Weak references for parent backlinks and handles
//! - Single owner with many weak refs (16 bytes less overhead per node vs `Rc<RefCell>`)
//!
//! # Algorithm Overview
//!
//! Fibonacci heaps maintain a collection of heap-ordered trees (not a single tree).
//! This allows O(1) insertion and merging by simply adding trees to the root list.
//! The key operations are:
//!
//! - **Insert**: O(1) - add single-node tree to root list
//! - **Delete-min**: O(log n) amortized - consolidate trees by degree (similar to binary addition)
//! - **Decrease-key**: O(1) amortized - use cascading cuts to maintain structure
//! - **Merge**: O(1) - concatenate root lists
//!
//! # Key Invariants
//!
//! 1. **Heap property**: All trees are heap-ordered (parent <= child)
//! 2. **Degree invariant**: After consolidation, at most one tree of each degree
//! 3. **Marking rule**: A node can lose at most one child before being cut
//! 4. **Fibonacci property**: Tree with root degree k has at least F_{k+2} nodes
//!
//! # Why Fibonacci Heaps?
//!
//! Fibonacci heaps achieve O(1) amortized time for insert, decrease-key, and merge
//! operations, while delete-min remains O(log n) amortized. This makes them
//! theoretically optimal for algorithms like Dijkstra's shortest path (improving
//! from O(E log V) to O(E + V log V)) and minimum spanning tree algorithms.
//!
//! The key innovation is the "lazy" approach: insertions and merges simply add
//! trees to a root list without restructuring. The work is deferred until
//! delete-min, which consolidates trees by degree. Cascading cuts during
//! decrease-key maintain the Fibonacci property that bounds tree sizes.
//!
//! The name "Fibonacci" comes from the Fibonacci numbers appearing in the
//! analysis of the maximum degree of nodes.
//!
//! # References
//!
//! - Fredman, M. L., & Tarjan, R. E. (1987). "Fibonacci heaps and their uses in
//!   improved network optimization algorithms." *Journal of the ACM*, 34(3), 596-615.
//!   [ACM DL](https://dl.acm.org/doi/10.1145/28869.28874)
//! - [Wikipedia: Fibonacci heap](https://en.wikipedia.org/wiki/Fibonacci_heap)

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError, MergeableHeap};
#[cfg(feature = "access_tracking")]
use crate::{track, track_op};
use refbox::{RefBox, Weak};

/// Handle to an element in a Fibonacci heap
///
/// Uses a Weak reference to the node. If the node has been removed from
/// the heap, the Weak reference will fail to borrow, allowing safe
/// detection of invalid handles.
pub struct FibonacciHandle<T, P> {
    node: NodeWeak<T, P>,
}

// Manual implementations because Weak doesn't implement PartialEq/Eq
impl<T, P> Clone for FibonacciHandle<T, P> {
    fn clone(&self) -> Self {
        FibonacciHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P> PartialEq for FibonacciHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        // Compare by pointer address
        self.node.as_ptr() == other.node.as_ptr()
    }
}

impl<T, P> Eq for FibonacciHandle<T, P> {}

impl<T, P> std::fmt::Debug for FibonacciHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FibonacciHandle")
            .field("node", &self.node.as_ptr())
            .finish()
    }
}

impl<T, P> Handle for FibonacciHandle<T, P> {}

/// Internal node structure for Fibonacci heap
///
/// Each node maintains:
/// - `priority`: The priority value (compared frequently during consolidation)
/// - `marked`: Flag used in cascading cuts (see decrease_key)
/// - `parent`: Weak reference to parent node (None if root)
/// - `children`: Vec of owned references to children
/// - `item`: The data stored in the heap (only accessed on pop)
///
/// Fields are ordered by access frequency for cache efficiency:
/// - HOT fields first (priority, marked) - accessed during consolidation & decrease_key
/// - Medium fields next (parent, children) - structure pointers for traversal
/// - COLD field last (item) - only accessed when extracting from heap
///
/// Children are stored in a Vec rather than a circular linked list.
/// This simplifies the implementation while maintaining the same
/// asymptotic complexity (degree is O(log n)).
struct Node<T, P> {
    // HOT fields - accessed during consolidation & decrease_key
    /// Priority value
    priority: P,
    /// Marked flag: true if this node has lost a child (used in cascading cuts)
    marked: bool,

    // Medium fields - structure pointers
    /// Weak reference to parent node (None if this is a root)
    parent: Option<NodeWeak<T, P>>,
    /// Owned references to children (single owner per child)
    children: Vec<NodeRef<T, P>>,

    // COLD field - only accessed on pop
    /// Item stored in heap
    item: T,
}

/// Type alias for an owned node pointer (single owner with weak references)
type NodeRef<T, P> = RefBox<Node<T, P>>;
/// Type alias for a weak reference to a node
type NodeWeak<T, P> = Weak<Node<T, P>>;

impl<T, P> Node<T, P> {
    fn new(item: T, priority: P) -> Self {
        Node {
            // HOT fields
            priority,
            marked: false,
            // Medium fields
            parent: None,
            children: Vec::new(),
            // COLD field
            item,
        }
    }

    fn degree(&self) -> usize {
        self.children.len()
    }
}

/// Fibonacci Heap using RefBox for memory-efficient weak references
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::fibonacci::FibonacciHeap;
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
///
/// let mut heap = FibonacciHeap::new();
/// let handle = heap.push_with_handle(5, "item");
/// heap.decrease_key(&handle, 1).unwrap();
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct FibonacciHeap<T, P: Ord> {
    /// All root trees (owned references)
    roots: Vec<NodeRef<T, P>>,
    /// Weak reference to the minimum node (for O(1) min lookup and update)
    min: Option<NodeWeak<T, P>>,
    /// Total number of elements
    len: usize,
}

impl<T, P: Ord> Default for FibonacciHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, P: Ord> Heap<T, P> for FibonacciHeap<T, P> {
    fn new() -> Self {
        Self {
            roots: Vec::new(),
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

    fn push(&mut self, priority: P, item: T) {
        let _ = self.push_with_handle(priority, item);
    }

    fn peek(&self) -> Option<(&P, &T)> {
        let min_weak = self.min.as_ref()?;
        // SAFETY: We're returning references to data inside RefBox.
        // This is safe because:
        // 1. The RefBox owner keeps the node alive as long as the heap exists
        // 2. We only read the data, never mutate through this path
        // 3. The heap's borrow checker ensures no mutable operations
        //    happen while these references exist
        if min_weak.try_borrow_mut().is_ok() {
            unsafe {
                let ptr = min_weak.as_ptr();
                Some((&(*ptr).priority, &(*ptr).item))
            }
        } else {
            None
        }
    }

    fn pop(&mut self) -> Option<(P, T)> {
        #[cfg(feature = "access_tracking")]
        let _guard = track_op!("pop");

        let min_weak = self.min.as_ref()?;
        let min_ptr = min_weak.as_ptr();

        let min_idx = self
            .roots
            .iter()
            .position(|r| r.as_ptr() == min_ptr)
            .expect("min must be in roots");

        // Remove min from roots - now we have the only owner
        let min_node = self.roots.swap_remove(min_idx);

        #[cfg(feature = "access_tracking")]
        track!(node);

        // Clear min reference before any operations
        self.min = None;

        // Add children to root list
        {
            let mut min_borrowed = min_node.try_borrow_mut().ok()?;
            #[cfg(feature = "access_tracking")]
            track!(read ChildPtr);
            for child in min_borrowed.children.drain(..) {
                #[cfg(feature = "access_tracking")]
                {
                    track!(node);
                    track!(chase);
                    track!(write ParentPtr);
                    track!(write Marked);
                }
                let mut child_borrow = child.try_borrow_mut().ok()?;
                child_borrow.parent = None;
                child_borrow.marked = false;
                drop(child_borrow);
                self.roots.push(child);
            }
        }

        self.len -= 1;

        // Consolidate if there are remaining roots
        if !self.roots.is_empty() {
            self.consolidate();
        }

        // Extract item and priority using take() - moves the node out
        let node = min_node.take();

        #[cfg(feature = "access_tracking")]
        {
            track!(read Priority);
            track!(read Item);
        }

        Some((node.priority, node.item))
    }
}

impl<T, P: Ord> MergeableHeap<T, P> for FibonacciHeap<T, P> {
    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        // Compare mins and update
        let other_is_smaller = match (&self.min, &other.min) {
            (Some(self_min), Some(other_min)) => {
                let self_borrow = self_min.try_borrow_mut().ok();
                let other_borrow = other_min.try_borrow_mut().ok();
                match (self_borrow, other_borrow) {
                    (Some(s), Some(o)) => o.priority < s.priority,
                    _ => false,
                }
            }
            (None, Some(_)) => true,
            _ => false,
        };

        if other_is_smaller {
            self.min = other.min.take();
        }

        // Append other's roots to self
        self.roots.append(&mut other.roots);
        self.len += other.len;

        // Prevent double-cleanup
        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> DecreaseKeyHeap<T, P> for FibonacciHeap<T, P> {
    type Handle = FibonacciHandle<T, P>;

    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        #[cfg(feature = "access_tracking")]
        let _guard = track_op!("push");

        let node = RefBox::new(Node::new(item, priority));
        let handle = FibonacciHandle {
            node: node.downgrade(),
        };

        #[cfg(feature = "access_tracking")]
        {
            track!(node);
            track!(write Priority);
            track!(write Item);
        }

        // Update min if necessary
        let should_update_min = match &self.min {
            Some(min_weak) => {
                if let (Ok(node_ref), Ok(min_ref)) =
                    (node.try_borrow_mut(), min_weak.try_borrow_mut())
                {
                    #[cfg(feature = "access_tracking")]
                    {
                        track!(read Priority);
                        track!(read Priority);
                    }
                    node_ref.priority < min_ref.priority
                } else {
                    false
                }
            }
            None => true, // No current min, so this becomes min
        };

        if should_update_min {
            self.min = Some(node.downgrade());
        }

        self.roots.push(node);
        self.len += 1;
        handle
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        #[cfg(feature = "access_tracking")]
        let _guard = track_op!("decrease_key");

        // Try to borrow through weak reference to check if node is valid
        if !handle.node.is_alive() {
            return Err(HeapError::InvalidHandle);
        }

        #[cfg(feature = "access_tracking")]
        track!(node);

        // Check that new priority is actually less
        {
            let node = handle
                .node
                .try_borrow_mut()
                .map_err(|_| HeapError::InvalidHandle)?;
            #[cfg(feature = "access_tracking")]
            track!(read Priority);
            if new_priority >= node.priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Update priority
        #[cfg(feature = "access_tracking")]
        track!(write Priority);
        {
            let mut node = handle
                .node
                .try_borrow_mut()
                .map_err(|_| HeapError::InvalidHandle)?;
            node.priority = new_priority;
        }

        // Get parent (if any)
        #[cfg(feature = "access_tracking")]
        track!(read ParentPtr);
        let parent_weak = {
            let node = handle
                .node
                .try_borrow_mut()
                .map_err(|_| HeapError::InvalidHandle)?;
            node.parent.clone()
        };

        if let Some(ref parent) = parent_weak {
            if parent.is_alive() {
                #[cfg(feature = "access_tracking")]
                {
                    track!(chase);
                    track!(node);
                }
                // Check if heap property is violated
                let should_cut = {
                    let node = handle
                        .node
                        .try_borrow_mut()
                        .map_err(|_| HeapError::InvalidHandle)?;
                    let parent_borrow = parent
                        .try_borrow_mut()
                        .map_err(|_| HeapError::InvalidHandle)?;
                    #[cfg(feature = "access_tracking")]
                    {
                        track!(read Priority);
                        track!(read Priority);
                    }
                    node.priority < parent_borrow.priority
                };

                if should_cut {
                    // Cut node from parent and add to root list
                    let node_ptr = handle.node.as_ptr();
                    self.cut(node_ptr, parent);
                    self.cascading_cut(parent.clone());
                }
            }
        }

        // Update min if this node now has smaller priority - O(1)
        self.maybe_update_min(&handle.node);

        Ok(())
    }
}

impl<T, P: Ord> FibonacciHeap<T, P> {
    /// Consolidates the heap by linking trees of the same degree
    fn consolidate(&mut self) {
        if self.roots.is_empty() {
            self.min = None;
            return;
        }

        // Max degree is O(log n) due to Fibonacci property
        let max_degree = ((self.len + 1) as f64).log2() as usize + 2;
        let mut degree_table: Vec<Option<NodeRef<T, P>>> = (0..=max_degree).map(|_| None).collect();

        // Process all current roots
        let roots_to_process: Vec<_> = self.roots.drain(..).collect();

        for mut root in roots_to_process {
            #[cfg(feature = "access_tracking")]
            {
                track!(node);
                track!(read Rank); // degree read
            }
            let mut d = root.try_borrow_mut().ok().map(|n| n.degree()).unwrap_or(0);

            // Link with existing trees of the same degree
            while d < degree_table.len() && degree_table[d].is_some() {
                let mut other = degree_table[d].take().unwrap();

                #[cfg(feature = "access_tracking")]
                {
                    track!(node);
                    track!(chase);
                }

                // Ensure root has smaller priority
                let should_swap = {
                    let root_borrow = root.try_borrow_mut().ok();
                    let other_borrow = other.try_borrow_mut().ok();
                    match (root_borrow, other_borrow) {
                        (Some(r), Some(o)) => {
                            #[cfg(feature = "access_tracking")]
                            {
                                track!(read Priority);
                                track!(read Priority);
                            }
                            o.priority < r.priority
                        }
                        _ => false,
                    }
                };

                if should_swap {
                    std::mem::swap(&mut root, &mut other);
                }

                // Link other as child of root (transfers ownership of other)
                self.link(other, &root);

                d += 1;
            }

            while d >= degree_table.len() {
                degree_table.push(None);
            }
            degree_table[d] = Some(root);
        }

        // Rebuild roots and find new min
        self.roots.clear();
        self.min = None;

        for root_opt in degree_table.into_iter().flatten() {
            // Check if this is the new min
            #[cfg(feature = "access_tracking")]
            track!(read Priority);
            let is_new_min = match &self.min {
                Some(current_min) => {
                    let root_borrow = root_opt.try_borrow_mut().ok();
                    let min_borrow = current_min.try_borrow_mut().ok();
                    match (root_borrow, min_borrow) {
                        (Some(r), Some(m)) => {
                            #[cfg(feature = "access_tracking")]
                            track!(read Priority);
                            r.priority < m.priority
                        }
                        _ => false,
                    }
                }
                None => true,
            };

            if is_new_min {
                self.min = Some(root_opt.downgrade());
            }

            self.roots.push(root_opt);
        }
    }

    /// Links node y as a child of node x (transfers ownership of y)
    fn link(&self, y: NodeRef<T, P>, x: &NodeRef<T, P>) {
        #[cfg(feature = "access_tracking")]
        {
            track!(write ParentPtr);
            track!(write Marked);
            track!(write ChildPtr);
        }
        // Make y a child of x
        if let Ok(mut y_borrow) = y.try_borrow_mut() {
            y_borrow.parent = Some(x.downgrade());
            y_borrow.marked = false;
        }
        if let Ok(mut x_borrow) = x.try_borrow_mut() {
            x_borrow.children.push(y);
        }
    }

    /// Cuts node from its parent and adds it to the root list
    /// Uses raw pointer for node identification and transfers ownership
    fn cut(&mut self, node_ptr: *const Node<T, P>, parent_weak: &NodeWeak<T, P>) {
        // Remove from parent's children and get ownership
        let owned_node = {
            let mut parent_borrowed = match parent_weak.try_borrow_mut() {
                Ok(b) => b,
                Err(_) => return,
            };
            #[cfg(feature = "access_tracking")]
            track!(read ChildPtr);
            let pos = parent_borrowed
                .children
                .iter()
                .position(|c| c.as_ptr() == node_ptr);
            match pos {
                Some(p) => {
                    #[cfg(feature = "access_tracking")]
                    track!(write ChildPtr);
                    parent_borrowed.children.swap_remove(p)
                }
                None => return,
            }
        };

        // Clear parent and marked flag
        #[cfg(feature = "access_tracking")]
        {
            track!(write ParentPtr);
            track!(write Marked);
        }
        if let Ok(mut node_borrow) = owned_node.try_borrow_mut() {
            node_borrow.parent = None;
            node_borrow.marked = false;
        }

        // Add to root list
        self.roots.push(owned_node);
    }

    /// Performs cascading cut on parent if it's marked
    /// Takes a weak reference to the node being checked
    fn cascading_cut(&mut self, node_weak: NodeWeak<T, P>) {
        #[cfg(feature = "access_tracking")]
        track!(read ParentPtr);
        let parent_weak = {
            let node_borrow = match node_weak.try_borrow_mut() {
                Ok(b) => b,
                Err(_) => return, // Node was already removed or borrowed
            };
            node_borrow.parent.clone()
        };

        if let Some(ref parent) = parent_weak {
            if parent.is_alive() {
                #[cfg(feature = "access_tracking")]
                {
                    track!(chase);
                    track!(node);
                }
                // Node has a parent (not a root)
                #[cfg(feature = "access_tracking")]
                track!(read Marked);
                let is_marked = node_weak
                    .try_borrow_mut()
                    .ok()
                    .map(|n| n.marked)
                    .unwrap_or(false);

                if !is_marked {
                    // First child lost: mark it
                    #[cfg(feature = "access_tracking")]
                    track!(write Marked);
                    if let Ok(mut node_borrow) = node_weak.try_borrow_mut() {
                        node_borrow.marked = true;
                    }
                } else {
                    // Already marked: cut it and cascade
                    let node_ptr = node_weak.as_ptr();
                    self.cut(node_ptr, parent);
                    self.cascading_cut(parent.clone());
                }
            }
        }
        // If no parent (root), cascading stops
    }

    /// Updates min if the given node has smaller priority than current min
    /// This is O(1) - just compares with current minimum
    fn maybe_update_min(&mut self, node_weak: &NodeWeak<T, P>) {
        let should_update = match &self.min {
            Some(current_min) => {
                let node_borrow = node_weak.try_borrow_mut().ok();
                let min_borrow = current_min.try_borrow_mut().ok();
                match (node_borrow, min_borrow) {
                    (Some(n), Some(m)) => n.priority < m.priority,
                    (Some(_), None) => true,
                    _ => false,
                }
            }
            None => node_weak.is_alive(),
        };

        if should_update {
            self.min = Some(node_weak.clone());
        }
    }
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.

#[cfg(test)]
mod memory_tests {
    use super::*;

    /// Document the memory savings from RefBox vs Rc<RefCell>
    ///
    /// Memory overhead comparison (64-bit):
    /// - `Rc<RefCell<T>>`: 24 bytes heap overhead (strong count + weak count + borrow state)
    /// - `RefBox<T>`: 8 bytes heap overhead (single status/count word)
    ///
    /// For a Fibonacci heap with N nodes, this saves 16*N bytes.
    /// Example: 100k nodes saves ~1.5 MB
    ///
    /// The refbox crate tests verify this overhead internally:
    /// <https://docs.rs/refbox/0.4.1/src/refbox/lib.rs.html#1160-1163>
    #[test]
    fn test_memory_overhead_documented() {
        // The refbox crate guarantees 8 bytes overhead without cyclic_stable feature.
        // This is verified by refbox's own tests (see heap_overhead test).
        //
        // Memory savings per node: 24 - 8 = 16 bytes
        // For 100,000 nodes: 1,600,000 bytes = 1.52 MB saved
        //
        // Additionally, RefBox's Weak provides direct borrow access without
        // the upgrade() step required by std::rc::Weak, improving cache
        // locality by reducing indirection.
        let savings_per_node = 16_usize; // bytes
        let example_nodes = 100_000_usize;
        let total_savings = savings_per_node * example_nodes;

        assert_eq!(
            total_savings, 1_600_000,
            "Expected 1.6 MB savings for 100k nodes"
        );
    }

    /// Verify the Node structure size is reasonable for cache efficiency
    #[test]
    fn test_node_size() {
        // Node<i64, i64> should fit in a reasonable number of cache lines
        // Cache line is typically 64 bytes
        let node_size = std::mem::size_of::<Node<i64, i64>>();

        // Without Option wrappers (using RefBox::take() for extraction):
        // - i64 for priority: 8 bytes
        // - bool for marked: 1 byte (+ padding to 8)
        // - Option<Weak<...>> for parent: 8 bytes
        // - Vec for children: 24 bytes
        // - i64 for item: 8 bytes
        // Total: ~56 bytes (depends on alignment)

        // Verify it fits in one cache line
        assert!(
            node_size <= 64,
            "Node<i64, i64> should fit in 1 cache line, got {} bytes",
            node_size
        );

        // Log actual size for documentation
        println!(
            "Node<i64, i64> size: {} bytes ({:.1} cache lines)",
            node_size,
            node_size as f64 / 64.0
        );
    }

    /// Verify FibonacciHeap struct size
    #[test]
    fn test_heap_struct_size() {
        let heap_size = std::mem::size_of::<FibonacciHeap<i64, i64>>();

        // FibonacciHeap contains:
        // - Vec<NodeRef<T, P>> roots: 24 bytes
        // - Option<NodeWeak<T, P>> min: 8 bytes
        // - usize len: 8 bytes
        // Total: 40 bytes

        println!("FibonacciHeap<i64, i64> struct size: {} bytes", heap_size);
        assert!(
            heap_size <= 64,
            "FibonacciHeap should fit in one cache line"
        );
    }
}
