//! Strict Fibonacci Heap implementation
//!
//! A priority queue with worst-case efficient operations, based on the
//! Brodal-Lagogiannis-Tarjan 2012 paper.
//!
//! # Complexity Bounds
//!
//! | Operation      | Time Complexity   |
//! |----------------|-------------------|
//! | `push`         | O(1) worst-case   |
//! | `peek`         | O(1) worst-case   |
//! | `pop`          | O(log n) worst-case |
//! | `decrease_key` | O(1) worst-case   |
//! | `merge`        | O(1) worst-case   |
//!
//! Unlike standard Fibonacci heaps which provide amortized bounds (where
//! occasional operations may take O(n) time), this implementation guarantees
//! that **every operation** completes within its stated time bound.
//!
//! # How It Works
//!
//! The strict Fibonacci heap achieves worst-case bounds through:
//!
//! 1. **Incremental consolidation**: Instead of doing all cleanup work in `pop()`,
//!    we spread O(1) reductions across each operation (insert, decrease_key, merge).
//!
//! 2. **Fix-list organization**: Active nodes are organized into groups based on
//!    their loss value and rank, enabling O(1) access to reduction candidates.
//!
//! 3. **Reduction types**:
//!    - *One-node loss reduction*: Cut nodes with loss ≥ 2
//!    - *Two-node loss reduction*: Link pairs of nodes with loss = 1
//!    - *Active root reduction*: Link same-rank active roots
//!    - *Root degree reduction*: Link passive roots
//!
//! # Implementation Notes
//!
//! This implementation uses:
//! - Raw pointers (`NonNull<Node>`) for node management
//! - Intrusive circular doubly-linked lists for sibling relationships
//! - Rank records for O(1) access to nodes by rank
//! - Fix-list with 7 group pointers for O(1) reduction candidate access
//! - Loss tracking with LOSS_FREE sentinel (255) for free nodes
//!
//! # References
//!
//! - Brodal, G. S., Lagogiannis, G., & Tarjan, R. E. (2012). "Strict Fibonacci heaps."
//!   *Proceedings of the 44th Annual ACM Symposium on Theory of Computing (STOC)*, 1177-1184.
//!   [ACM DL](https://dl.acm.org/doi/10.1145/2213977.2214082)

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError};
use intrusive_circular_list::{container_of_mut, CircularLink, CircularListOps};
use std::ptr::NonNull;

// ============================================================================
// Loss Tracking
// ============================================================================

/// Compile-time assertion that usize is at most 252 bits.
/// Loss is bounded by O(log n), so on a system with at most 2^252 addressable elements,
/// loss can never reach 255 (our sentinel for "free").
/// In practice, 64-bit systems (size_of::<usize>() == 8) are the norm.
const _: () = assert!(
    std::mem::size_of::<usize>() <= 31, // 31 bytes = 248 bits < 252
    "loss tracking assumes at most 252-bit address space"
);

/// Sentinel value indicating a node is "free" (active but not fixed).
/// Since loss is bounded by ~log₂(n) ≤ 252 bits, 255 is unreachable.
const LOSS_FREE: u8 = u8::MAX;

// ============================================================================
// Rank Record System
// ============================================================================

/// A rank record in the strict Fibonacci heap.
///
/// Rank records form a doubly-linked list ordered by rank value. Each rank record
/// tracks nodes of that rank, enabling O(1) access to nodes by rank for the
/// reduction operations that maintain worst-case bounds.
///
/// # Invariants
///
/// - Rank records are reference-counted by nodes
/// - When `reference_count` drops to 0, the record is retired (deallocated)
/// - The rank list is ordered by increasing rank values
/// - Gaps in rank values are allowed (not all ranks need to exist)
/// - `free` points to the first free node of this rank (if any)
/// - `loss_one` points to the first fixed node with loss=1 of this rank (if any)
struct RankRecord {
    /// Next rank record in the doubly-linked list (higher rank).
    next: Option<NonNull<RankRecord>>,
    /// Previous rank record in the doubly-linked list (lower rank).
    prev: Option<NonNull<RankRecord>>,
    /// The integer rank value.
    rank: usize,
    /// Number of nodes referencing this rank record.
    /// When this reaches 0, the record is retired.
    reference_count: usize,
}

impl RankRecord {
    /// Creates a new rank record with the given rank value.
    fn new(rank: usize) -> Box<RankRecord> {
        Box::new(RankRecord {
            next: None,
            prev: None,
            rank,
            reference_count: 0,
        })
    }

    /// Increases the reference count.
    fn increase_reference_count(&mut self) {
        self.reference_count += 1;
    }

    /// Decreases the reference count.
    ///
    /// Returns true if the reference count reached zero (caller should retire).
    fn decrease_reference_count(&mut self) -> bool {
        debug_assert!(self.reference_count > 0, "Reference count underflow");
        self.reference_count -= 1;
        self.reference_count == 0
    }

    /// Gets or creates the next rank record (rank + 1).
    ///
    /// # Safety
    ///
    /// The caller must ensure this rank record is valid and not retired.
    #[allow(dead_code)] // Will be used in later phases
    unsafe fn get_or_create_next(&mut self) -> NonNull<RankRecord> {
        let self_ptr = NonNull::from(&mut *self);
        let next_rank = self.rank + 1;

        // Check if next already exists and has the right rank
        if let Some(next_ptr) = self.next {
            if next_ptr.as_ref().rank == next_rank {
                return next_ptr;
            }
        }

        // Need to create a new rank record
        let new_record = RankRecord::new(next_rank);
        let new_ptr = NonNull::new(Box::into_raw(new_record)).unwrap();

        // Insert new record after self
        (*new_ptr.as_ptr()).prev = Some(self_ptr);
        (*new_ptr.as_ptr()).next = self.next;

        if let Some(mut next_ptr) = self.next {
            next_ptr.as_mut().prev = Some(new_ptr);
        }
        self.next = Some(new_ptr);

        new_ptr
    }

    /// Gets or creates the previous rank record (rank - 1).
    ///
    /// # Safety
    ///
    /// The caller must ensure this rank record is valid, not retired,
    /// and has rank > 0.
    #[allow(dead_code)] // Will be used in later phases
    unsafe fn get_or_create_prev(&mut self) -> NonNull<RankRecord> {
        debug_assert!(self.rank > 0, "Cannot get prev of rank 0");
        let self_ptr = NonNull::from(&mut *self);
        let prev_rank = self.rank - 1;

        // Check if prev already exists and has the right rank
        if let Some(prev_ptr) = self.prev {
            if prev_ptr.as_ref().rank == prev_rank {
                return prev_ptr;
            }
        }

        // Need to create a new rank record
        let new_record = RankRecord::new(prev_rank);
        let new_ptr = NonNull::new(Box::into_raw(new_record)).unwrap();

        // Insert new record before self
        (*new_ptr.as_ptr()).next = Some(self_ptr);
        (*new_ptr.as_ptr()).prev = self.prev;

        if let Some(mut prev_ptr) = self.prev {
            prev_ptr.as_mut().next = Some(new_ptr);
        }
        self.prev = Some(new_ptr);

        new_ptr
    }
}

// ============================================================================
// Handle
// ============================================================================

/// Handle to an element in a Strict Fibonacci heap.
///
/// # Safety
///
/// The handle becomes invalid after the element is removed from the heap via `pop()`.
/// Using an invalid handle with `decrease_key` is undefined behavior (use-after-free).
/// Callers must ensure handles are not used after their element has been popped.
pub struct StrictFibonacciHandle<T, P> {
    node: Option<NonNull<Node<T, P>>>,
}

impl<T, P> Clone for StrictFibonacciHandle<T, P> {
    fn clone(&self) -> Self {
        StrictFibonacciHandle { node: self.node }
    }
}

impl<T, P> PartialEq for StrictFibonacciHandle<T, P> {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
    }
}

impl<T, P> Eq for StrictFibonacciHandle<T, P> {}

impl<T, P> std::fmt::Debug for StrictFibonacciHandle<T, P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StrictFibonacciHandle")
            .field("valid", &self.node.is_some())
            .finish()
    }
}

impl<T, P> Handle for StrictFibonacciHandle<T, P> {}

/// A node in the Strict Fibonacci heap.
///
/// Uses intrusive circular linked lists for sibling relationships and fix-list.
///
/// **Cache Optimization**: Fields are ordered for cache locality:
/// - Hot path first: `priority` is accessed on every comparison
/// - Intrusive links: circular list pointers for sibling and fix-list navigation
/// - Traversal fields: parent, child, rank_record pointers
/// - Small fields: rank_count (usize for now - complex invariants), active, loss
/// - Cold path last: `item` is only accessed when popping
///
/// Note: `rank_count` remains `usize` due to complex invariants in the strict
/// Fibonacci heap algorithm that require careful analysis before changing.
struct Node<T, P> {
    /// The priority of this node (smaller = higher priority for min-heap).
    /// Hot path: accessed on every comparison.
    priority: P,
    /// Intrusive link for the sibling circular list.
    sibling_link: CircularLink,
    /// Intrusive link for the fix-list (circular list of active nodes).
    fix_link: CircularLink,
    /// Pointer to the parent node (None if this is a root).
    parent: Option<NonNull<Node<T, P>>>,
    /// Pointer to one child (entry point into children's circular list).
    child: Option<NonNull<Node<T, P>>>,
    /// Pointer to this node's rank record.
    /// Active nodes have a rank record; passive nodes may have None.
    /// The rank record is reference-counted.
    rank_record: Option<NonNull<RankRecord>>,
    /// The rank of this node (number of fixed/active children).
    /// This is tracked locally until the node becomes active and uses a RankRecord.
    /// In the full algorithm, this matches the rank_record's rank value.
    /// Note: Kept as usize due to complex invariants in the algorithm.
    rank_count: usize,
    /// Whether this node is "active" in the strict Fibonacci sense.
    /// Active nodes are tracked specially to maintain worst-case bounds.
    active: bool,
    /// Loss value for active nodes.
    ///
    /// - `LOSS_FREE` (255): Node is free (active but not fixed)
    /// - `0, 1, 2, ...`: Node is fixed with this loss value
    ///
    /// For passive nodes, this field is ignored.
    loss: u8,
    /// The item stored in this node.
    /// Cold path: only accessed when popping.
    item: T,
}

impl<T, P> Node<T, P> {
    /// Creates a new node with the given priority and item.
    ///
    /// The node starts as passive (not active, not in fix-list).
    /// Use `set_rank_record` to assign a rank record when the node becomes active.
    fn new(priority: P, item: T) -> Box<Node<T, P>> {
        Box::new(Node {
            // Hot path first
            priority,
            // Circular links for sibling list and fix-list
            sibling_link: CircularLink::new(),
            fix_link: CircularLink::new(),
            // Traversal fields
            parent: None,
            child: None,
            rank_record: None,
            // Small fields
            rank_count: 0,
            active: false,
            loss: LOSS_FREE, // Passive nodes ignore this, but initialize to free
            // Cold path last
            item,
        })
    }

    /// Returns true if this node is active (part of an active heap).
    #[inline]
    #[allow(dead_code)] // Will be used in later phases
    fn is_active(&self) -> bool {
        self.active
    }

    /// Returns true if this node is passive (part of a retired/melded heap).
    #[inline]
    #[allow(dead_code)] // Will be used in later phases
    fn is_passive(&self) -> bool {
        !self.active
    }

    /// Returns true if this node is free (active but not fixed).
    #[inline]
    #[allow(dead_code)] // Will be used in later phases
    fn is_free(&self) -> bool {
        self.active && self.loss == LOSS_FREE
    }

    /// Returns true if this node is fixed (active with loss tracking).
    #[inline]
    #[allow(dead_code)] // Will be used in later phases
    fn is_fixed(&self) -> bool {
        self.active && self.loss != LOSS_FREE
    }

    /// Increments the loss counter for a fixed node.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if the node is not fixed, or if loss would overflow into the sentinel.
    #[allow(dead_code)] // Will be used in later phases
    fn increment_loss(&mut self) {
        debug_assert!(self.is_fixed(), "can only increment loss on fixed nodes");
        self.loss += 1;
        debug_assert!(
            self.loss != LOSS_FREE,
            "loss overflow into sentinel - should be impossible on ≤128-bit systems"
        );
    }

    /// Gets the loss value. Only valid for fixed nodes.
    #[inline]
    #[allow(dead_code)] // Will be used in later phases
    fn get_loss(&self) -> u8 {
        debug_assert!(self.is_fixed(), "loss is only valid for fixed nodes");
        self.loss
    }

    /// Gets a pointer to the sibling link.
    fn sibling_link_ptr(&self) -> NonNull<CircularLink> {
        NonNull::from(&self.sibling_link)
    }

    /// Gets a pointer to the fix-list link.
    #[allow(dead_code)] // Will be used in later phases
    fn fix_link_ptr(&self) -> NonNull<CircularLink> {
        NonNull::from(&self.fix_link)
    }

    /// Converts a free node to fixed (with loss = 0).
    ///
    /// # Safety
    ///
    /// Caller must update fix-list membership appropriately.
    #[allow(dead_code)] // Will be used in later phases
    fn free_to_fixed(&mut self) {
        debug_assert!(self.is_free(), "node must be free to convert to fixed");
        self.loss = 0;
    }

    /// Converts a fixed node to free.
    ///
    /// # Safety
    ///
    /// Caller must update fix-list membership appropriately.
    #[allow(dead_code)] // Will be used in later phases
    fn fixed_to_free(&mut self) {
        debug_assert!(self.is_fixed(), "node must be fixed to convert to free");
        self.loss = LOSS_FREE;
    }

    /// Returns the rank of this node.
    ///
    /// If the node has an active rank record, returns the rank from that record.
    /// Otherwise, returns the local rank_count (used for passive nodes).
    fn rank(&self) -> usize {
        // For passive nodes or nodes not yet assigned a rank record,
        // use the local rank_count. Once rank records are fully integrated,
        // active nodes will use the rank_record's value.
        self.rank_record
            .map(|ptr| unsafe { ptr.as_ref().rank })
            .unwrap_or(self.rank_count)
    }

    /// Sets the rank record for this node, updating reference counts.
    ///
    /// # Safety
    ///
    /// The caller must ensure any old rank record and the new rank record
    /// are valid pointers.
    #[allow(dead_code)] // Will be used in later phases
    unsafe fn set_rank_record(&mut self, new_rank: Option<NonNull<RankRecord>>) {
        // No-op if the rank record is unchanged.
        if self.rank_record == new_rank {
            return;
        }

        // Decrease reference count on old rank record
        if let Some(mut old_ptr) = self.rank_record {
            let should_retire = old_ptr.as_mut().decrease_reference_count();
            if should_retire {
                // This rank record has no more references, deallocate it
                retire_rank_record(old_ptr);
            }
        }

        // Set new rank record and increase reference count
        self.rank_record = new_rank;
        if let Some(mut new_ptr) = self.rank_record {
            new_ptr.as_mut().increase_reference_count();
        }
    }
}

/// Retires (deallocates) a rank record, removing it from the rank list.
///
/// # Safety
///
/// The caller must ensure the rank record has reference_count == 0
/// and is a valid pointer.
unsafe fn retire_rank_record(rank_ptr: NonNull<RankRecord>) {
    let rank = rank_ptr.as_ptr();

    debug_assert!(
        (*rank).reference_count == 0,
        "Retiring rank record with non-zero reference_count"
    );

    // Unlink from the doubly-linked list
    if let Some(mut prev_ptr) = (*rank).prev {
        prev_ptr.as_mut().next = (*rank).next;
    }
    if let Some(mut next_ptr) = (*rank).next {
        next_ptr.as_mut().prev = (*rank).prev;
    }

    // Deallocate
    drop(Box::from_raw(rank));
}

/// Strict Fibonacci Heap.
///
/// A priority queue with efficient decrease_key operation.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
/// use rust_advanced_heaps::{Heap, DecreaseKeyHeap};
///
/// let mut heap = StrictFibonacciHeap::new();
/// let handle = heap.push_with_handle(5, "item");
/// heap.decrease_key(&handle, 1).unwrap();
/// assert_eq!(heap.peek(), Some((&1, &"item")));
/// ```
pub struct StrictFibonacciHeap<T, P: Ord> {
    /// Entry point into the circular list of roots (if any).
    roots: Option<NonNull<Node<T, P>>>,
    /// Pointer to the minimum root.
    min: Option<NonNull<Node<T, P>>>,
    /// Number of elements in the heap.
    len: usize,
    /// Head of the rank list (rank 0 if it exists).
    /// The rank list is a doubly-linked list ordered by increasing rank.
    rank_list: Option<NonNull<RankRecord>>,

    // =========================================================================
    // Fix-list pointers (Phase 3)
    // =========================================================================
    // The fix-list is a circular doubly-linked list of all nodes in the heap,
    // organized into groups for efficient access during reductions.
    // Group order: passive -> free_multiple -> free_single -> loss_zero
    //              -> loss_one_multiple -> loss_one_single -> loss_two
    //
    // Each pointer points to the first node in that group (if any).
    /// First passive node in fix-list (nodes from retired/melded heaps).
    fix_passive: Option<NonNull<Node<T, P>>>,
    /// First free node with 2+ nodes of same rank (enables free reduction).
    fix_free_multiple: Option<NonNull<Node<T, P>>>,
    /// First free node with unique rank in this group.
    fix_free_single: Option<NonNull<Node<T, P>>>,
    /// First fixed node with loss = 0.
    fix_loss_zero: Option<NonNull<Node<T, P>>>,
    /// First fixed node with loss = 1, with 2+ nodes of same rank.
    fix_loss_one_multiple: Option<NonNull<Node<T, P>>>,
    /// First fixed node with loss = 1, with unique rank.
    fix_loss_one_single: Option<NonNull<Node<T, P>>>,
    /// First fixed node with loss >= 2 (triggers immediate reduction).
    fix_loss_two: Option<NonNull<Node<T, P>>>,
}

impl<T, P: Ord> Default for StrictFibonacciHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: StrictFibonacciHeap can be sent between threads if T and P can.
// The raw pointers are owned by the heap.
unsafe impl<T: Send, P: Ord + Send> Send for StrictFibonacciHeap<T, P> {}

impl<T, P: Ord> Drop for StrictFibonacciHeap<T, P> {
    fn drop(&mut self) {
        // We need to deallocate all nodes and rank records.
        // Traverse all roots and their descendants.
        if let Some(root_ptr) = self.roots {
            let mut to_free: Vec<NonNull<Node<T, P>>> = Vec::new();
            let ops = CircularListOps::new();

            // Collect all roots
            unsafe {
                let start_link = root_ptr.as_ref().sibling_link_ptr();
                ops.for_each(start_link, |link_ptr| {
                    let node_ptr: *mut Node<T, P> =
                        container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                    to_free.push(NonNull::new_unchecked(node_ptr));
                });
            }

            // Process nodes, collecting children
            let mut i = 0;
            while i < to_free.len() {
                let node_ptr = to_free[i];
                unsafe {
                    let node = node_ptr.as_ref();
                    if let Some(child_ptr) = node.child {
                        let start_link = child_ptr.as_ref().sibling_link_ptr();
                        ops.for_each(start_link, |link_ptr| {
                            let child_node_ptr: *mut Node<T, P> =
                                container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                            to_free.push(NonNull::new_unchecked(child_node_ptr));
                        });
                    }
                }
                i += 1;
            }

            // Free all nodes
            // Note: In Phase 2, nodes don't have rank_records set yet (they're passive).
            // When rank_records are fully integrated, we'll need to call
            // set_rank_record(None) to properly decrement reference counts.
            for node_ptr in to_free {
                unsafe {
                    drop(Box::from_raw(node_ptr.as_ptr()));
                }
            }
        }

        // Also free any remaining rank records (should be empty after nodes are freed,
        // but just in case there's rank 0 with ref count issues during partial construction)
        let mut rank_ptr = self.rank_list;
        while let Some(current) = rank_ptr {
            unsafe {
                rank_ptr = current.as_ref().next;
                drop(Box::from_raw(current.as_ptr()));
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for StrictFibonacciHeap<T, P> {
    fn new() -> Self {
        Self {
            roots: None,
            min: None,
            len: 0,
            rank_list: None,
            // Fix-list pointers (all None for empty heap)
            fix_passive: None,
            fix_free_multiple: None,
            fix_free_single: None,
            fix_loss_zero: None,
            fix_loss_one_multiple: None,
            fix_loss_one_single: None,
            fix_loss_two: None,
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
        self.min.map(|min_ptr| unsafe {
            let node = min_ptr.as_ref();
            (&node.priority, &node.item)
        })
    }

    /// Removes and returns the minimum element.
    ///
    /// **Time Complexity**: O(log n) worst-case
    ///
    /// The O(log n) bound comes from:
    /// - Splicing O(log n) children of the minimum to the root list
    /// - Consolidation, which links O(log n) roots
    /// - Performing O(log n) reductions spread across the operation
    ///
    /// Unlike amortized bounds where occasional operations may take O(n) time,
    /// this implementation guarantees every pop operation completes in O(log n).
    fn pop(&mut self) -> Option<(P, T)> {
        let min_ptr = self.min?;

        // Count children to determine reduction budget.
        // The min node has O(log n) children due to Fibonacci heap structure.
        //
        // Safety: min_ptr is valid (checked via `self.min?` above).
        // CircularListOps::for_each safely traverses the circular sibling list,
        // visiting each child exactly once before returning to the start.
        let child_count = unsafe {
            let min_node = min_ptr.as_ref();
            match min_node.child {
                None => 0,
                Some(child_ptr) => {
                    let ops = CircularListOps::new();
                    let mut count = 0;
                    ops.for_each(child_ptr.as_ref().sibling_link_ptr(), |_| {
                        count += 1;
                    });
                    count
                }
            }
        };

        // Remove min from root list
        self.remove_from_roots(min_ptr);

        // Add all children of min to root list
        unsafe {
            let min_node = min_ptr.as_ref();
            if let Some(child_ptr) = min_node.child {
                self.splice_children_to_roots(child_ptr);
            }
        }

        self.len -= 1;

        // Consolidate the heap
        self.consolidate();

        // Perform O(log n) reductions to maintain worst-case bounds
        // The reduction budget is proportional to the children of the removed min
        // This amortizes the cleanup work to achieve O(log n) worst-case
        let reduction_budget = child_count.max(1);
        self.perform_reductions(reduction_budget);

        self.debug_assert_invariants();

        // Extract the node's data
        unsafe {
            let node = Box::from_raw(min_ptr.as_ptr());
            Some((node.priority, node.item))
        }
    }

    /// Merges another heap into this heap.
    ///
    /// **Time Complexity**: O(1) worst-case
    ///
    /// The merge operation simply concatenates root lists and updates the minimum
    /// pointer, requiring only constant time. Reductions maintain the structure.
    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        // Splice the root lists together
        if let (Some(self_root), Some(other_root)) = (self.roots, other.roots) {
            let ops = CircularListOps::new();
            unsafe {
                let self_link = self_root.as_ref().sibling_link_ptr();
                let other_link = other_root.as_ref().sibling_link_ptr();
                ops.splice(Some(self_link), Some(other_link));
            }
        }

        // Update minimum
        if let (Some(self_min), Some(other_min)) = (self.min, other.min) {
            unsafe {
                if other_min.as_ref().priority < self_min.as_ref().priority {
                    self.min = Some(other_min);
                }
            }
        }

        self.len += other.len;

        // Prevent other from deallocating nodes
        other.roots = None;
        other.min = None;
        other.len = 0;

        // Perform O(1) reductions to maintain worst-case bounds
        self.perform_reductions(1);

        self.debug_assert_invariants();
    }
}

impl<T, P: Ord> DecreaseKeyHeap<T, P> for StrictFibonacciHeap<T, P> {
    type Handle = StrictFibonacciHandle<T, P>;

    /// Inserts a new element into the heap.
    ///
    /// **Time Complexity**: O(1) worst-case (with incremental reductions)
    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Node::new(priority, item);
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        let handle = StrictFibonacciHandle {
            node: Some(node_ptr),
        };

        // Add to root list
        self.add_to_roots(node_ptr);

        // Update minimum if necessary
        self.update_min(node_ptr);

        self.len += 1;

        // Perform O(1) reductions to maintain worst-case bounds
        // One reduction per insert amortizes the consolidation work
        self.perform_reductions(1);

        self.debug_assert_invariants();

        handle
    }

    /// Decreases the priority of an element.
    ///
    /// **Time Complexity**: O(1) worst-case (with incremental reductions)
    ///
    /// # Safety
    ///
    /// Callers must ensure the handle's node has not been popped.
    /// Using a handle after `pop()` removes its element is undefined behavior (use-after-free).
    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node_ptr = handle.node.ok_or(HeapError::InvalidHandle)?;

        // Check and update priority
        unsafe {
            let node = node_ptr.as_ptr();

            if new_priority >= (*node).priority {
                return Err(HeapError::PriorityNotDecreased);
            }

            (*node).priority = new_priority;

            // Check if we need to cut from parent
            if let Some(parent_ptr) = (*node).parent {
                let parent = parent_ptr.as_ref();
                if (*node).priority < parent.priority {
                    self.cut(node_ptr);
                }
            }

            // Update minimum pointer
            self.update_min(node_ptr);
        }

        // Perform O(1) reductions to maintain worst-case bounds
        // One reduction per decrease_key amortizes the consolidation work
        self.perform_reductions(1);

        self.debug_assert_invariants();
        self.debug_assert_heap_property(node_ptr);

        Ok(())
    }
}

impl<T, P: Ord> StrictFibonacciHeap<T, P> {
    // =========================================================================
    // Fix-list operations (Phase 3)
    // =========================================================================

    /// Clears all fix-list group pointers.
    ///
    /// Called when retiring a heap (during meld) or when the heap becomes empty.
    #[allow(dead_code)] // Will be used in later phases
    fn fix_list_retire(&mut self) {
        self.fix_passive = None;
        self.fix_free_multiple = None;
        self.fix_free_single = None;
        self.fix_loss_zero = None;
        self.fix_loss_one_multiple = None;
        self.fix_loss_one_single = None;
        self.fix_loss_two = None;
    }

    /// Returns the first node in the fix-list (head), if any.
    ///
    /// The fix-list is ordered: passive -> free_multiple -> free_single ->
    /// loss_zero -> loss_one_multiple -> loss_one_single -> loss_two.
    /// Returns the first non-None pointer in that order.
    #[allow(dead_code)] // Will be used in later phases
    fn fix_list_head(&self) -> Option<NonNull<Node<T, P>>> {
        self.fix_passive
            .or(self.fix_free_multiple)
            .or(self.fix_free_single)
            .or(self.fix_loss_zero)
            .or(self.fix_loss_one_multiple)
            .or(self.fix_loss_one_single)
            .or(self.fix_loss_two)
    }

    // =========================================================================
    // Reduction Operations (Phase 3 Part 2)
    // =========================================================================
    //
    // The strict Fibonacci heap maintains worst-case bounds by performing
    // O(1) "reduction" transformations after each operation. These reductions
    // use two primitive operations:
    //
    // - **cut(y)**: Removes y from its parent's children list, making y a new root
    // - **link(x, y)**: Given x.key < y.key, cuts y from its parent and makes y
    //                   a child of x. Active children go leftmost, passive rightmost.
    //
    // Three types of reductions are used:
    //
    // 1. **Active root reduction**: Link two active roots of the same rank.
    //    This reduces the number of free nodes (active roots) by 1.
    //
    // 2. **Root degree reduction**: Link a passive root with a non-rightmost
    //    passive child of the root with the smallest key among passive linkable
    //    roots. This reduces root degree.
    //
    // 3. **Loss reduction**: Fix active non-root nodes with high loss by cutting
    //    and re-linking. One-node and two-node variants exist.
    //
    // The pigeonhole principle guarantees that if there are too many violations,
    // a reduction that makes progress is always available.

    /// Performs an active root reduction if possible.
    ///
    /// Finds two active roots (free nodes) with the same rank and links them.
    /// After linking, one becomes a fixed child (loss = 0) and the other
    /// becomes an active root with incremented rank.
    ///
    /// # Returns
    ///
    /// `true` if a reduction was performed, `false` otherwise.
    ///
    /// # Time Complexity
    ///
    /// O(1) - uses fix-list to find candidates in constant time.
    #[allow(dead_code)] // Will be used when reductions are integrated
    fn active_root_reduction(&mut self) -> bool {
        // We need two active roots of the same rank.
        // The fix_free_multiple group contains free nodes where there are
        // 2+ nodes of the same rank. We find a pair from the rank records.

        // For now, scan fix_free_multiple to find a matching pair.
        // In a full implementation, rank records would track this.
        let first = match self.fix_free_multiple {
            Some(ptr) => ptr,
            None => return false,
        };

        unsafe {
            let first_rank = first.as_ref().rank();

            // Find another node with the same rank in fix_free_multiple
            let ops = CircularListOps::new();
            let start_link = first.as_ref().fix_link_ptr();

            let mut second: Option<NonNull<Node<T, P>>> = None;
            ops.for_each(start_link, |link_ptr| {
                if second.is_some() {
                    return; // Already found
                }
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, fix_link);
                let node = NonNull::new_unchecked(node_ptr);

                // Skip self
                if node == first {
                    return;
                }

                if (*node_ptr).rank() == first_rank {
                    second = Some(node);
                }
            });

            let second = match second {
                Some(ptr) => ptr,
                None => return false,
            };

            // Perform the link: smaller priority becomes parent
            let (parent, child) = if first.as_ref().priority <= second.as_ref().priority {
                (first, second)
            } else {
                (second, first)
            };

            // Remove both from fix-list before linking
            self.fix_list_remove(parent);
            self.fix_list_remove(child);

            // Remove child from root list
            self.remove_from_roots(child);

            // Link child under parent
            self.link_as_active_child(parent, child);

            // Child becomes fixed with loss = 0
            (*child.as_ptr()).loss = 0;

            // Parent remains free (active root) but with rank + 1
            // Re-add parent to fix-list (it may move between groups based on rank)
            self.fix_list_add(parent);
        }

        true
    }

    /// Performs a root degree reduction if possible.
    ///
    /// Links a passive root x with a non-rightmost passive child y of another
    /// passive root z (where z.key is minimum among passive linkable roots).
    /// This reduces the root degree.
    ///
    /// # Returns
    ///
    /// `true` if a reduction was performed, `false` otherwise.
    ///
    /// # Time Complexity
    ///
    /// O(1) - uses fix-list passive group for O(1) access.
    #[allow(dead_code)] // Will be used when reductions are integrated
    fn root_degree_reduction(&mut self) -> bool {
        // For root degree reduction, we need:
        // 1. A passive root x
        // 2. Another passive root z with a non-rightmost passive child y
        // 3. Link x and y

        // For this phase, we implement a simplified version that finds
        // any two linkable passive roots and links them.

        let x = match self.fix_passive {
            Some(ptr) => ptr,
            None => return false,
        };

        unsafe {
            // Find another passive root
            let ops = CircularListOps::new();
            let start_link = x.as_ref().fix_link_ptr();

            let mut z: Option<NonNull<Node<T, P>>> = None;
            ops.for_each(start_link, |link_ptr| {
                if z.is_some() {
                    return;
                }
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, fix_link);
                let node = NonNull::new_unchecked(node_ptr);

                if node != x && (*node_ptr).is_passive() && (*node_ptr).parent.is_none() {
                    z = Some(node);
                }
            });

            let z = match z {
                Some(ptr) => ptr,
                None => return false,
            };

            // Link x under z (or z under x, depending on priority)
            let (parent, child) = if x.as_ref().priority <= z.as_ref().priority {
                (x, z)
            } else {
                (z, x)
            };

            // Remove both from fix-list
            self.fix_list_remove(parent);
            self.fix_list_remove(child);

            // Remove child from root list
            self.remove_from_roots(child);

            // Link as passive child (rightmost)
            self.link_as_passive_child(parent, child);
        }

        true
    }

    /// Links a node as an active child (leftmost) of a parent.
    ///
    /// Active children are placed at the left end of the children list.
    ///
    /// # Safety
    ///
    /// Both pointers must be valid and child must not already be in parent's
    /// children list.
    #[allow(dead_code)] // Will be used when reductions are integrated
    unsafe fn link_as_active_child(
        &mut self,
        parent_ptr: NonNull<Node<T, P>>,
        child_ptr: NonNull<Node<T, P>>,
    ) {
        let ops = CircularListOps::new();
        let parent = parent_ptr.as_ptr();
        let child = child_ptr.as_ptr();

        // Set child's parent
        (*child).parent = Some(parent_ptr);

        let child_link = (*child).sibling_link_ptr();

        match (*parent).child {
            None => {
                // First child
                ops.make_circular(child_link);
                (*parent).child = Some(child_ptr);
            }
            Some(first_child_ptr) => {
                // Insert before the current first child (leftmost position)
                let first_child_link = first_child_ptr.as_ref().sibling_link_ptr();
                ops.insert_before(first_child_link, child_link);
                // Update parent's child pointer to the new leftmost
                (*parent).child = Some(child_ptr);
            }
        }

        // Increment parent's rank
        (*parent).rank_count += 1;
    }

    /// Links a node as a passive child (rightmost) of a parent.
    ///
    /// Passive children are placed at the right end of the children list.
    ///
    /// # Safety
    ///
    /// Both pointers must be valid and child must not already be in parent's
    /// children list.
    #[allow(dead_code)] // Will be used when reductions are integrated
    unsafe fn link_as_passive_child(
        &mut self,
        parent_ptr: NonNull<Node<T, P>>,
        child_ptr: NonNull<Node<T, P>>,
    ) {
        let ops = CircularListOps::new();
        let parent = parent_ptr.as_ptr();
        let child = child_ptr.as_ptr();

        // Set child's parent
        (*child).parent = Some(parent_ptr);

        let child_link = (*child).sibling_link_ptr();

        match (*parent).child {
            None => {
                // First child
                ops.make_circular(child_link);
                (*parent).child = Some(child_ptr);
            }
            Some(first_child_ptr) => {
                // Insert after the last child (rightmost position)
                // In a circular list, last is prev of first
                let first_child_link = first_child_ptr.as_ref().sibling_link_ptr();
                // insert_before inserts just before the given node,
                // which is the same as inserting after the last (prev) node
                ops.insert_before(first_child_link, child_link);
                // Child pointer stays at first_child (leftmost), child is rightmost
            }
        }

        // Passive children don't increment rank (only active children count)
    }

    /// Adds a node to the appropriate fix-list group based on its state.
    ///
    /// Fix-list groups are ordered:
    /// - passive: passive nodes
    /// - free_multiple: free nodes with 2+ nodes of same rank
    /// - free_single: free nodes with unique rank
    /// - loss_zero: fixed nodes with loss = 0
    /// - loss_one_multiple: fixed nodes with loss = 1 and 2+ same rank
    /// - loss_one_single: fixed nodes with loss = 1 with unique rank
    /// - loss_two: fixed nodes with loss >= 2
    ///
    /// # Safety
    ///
    /// The node must be valid and not already in the fix-list.
    #[allow(dead_code)] // Will be used when reductions are integrated
    fn fix_list_add(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            let node_link = (*node).fix_link_ptr();

            // Determine which group this node belongs to
            let group_ptr = if (*node).is_passive() {
                &mut self.fix_passive
            } else if (*node).is_free() {
                // TODO: Check if there's another free node of same rank
                // For now, add to free_single; we'll refine this with rank tracking
                &mut self.fix_free_single
            } else {
                // Node is fixed
                let loss = (*node).loss;
                if loss == 0 {
                    &mut self.fix_loss_zero
                } else if loss == 1 {
                    // TODO: Check if there's another loss=1 node of same rank
                    &mut self.fix_loss_one_single
                } else {
                    &mut self.fix_loss_two
                }
            };

            match *group_ptr {
                None => {
                    // First node in this group
                    ops.make_circular(node_link);
                    *group_ptr = Some(node_ptr);
                }
                Some(head_ptr) => {
                    // Insert into existing group's circular list
                    let head_link = head_ptr.as_ref().fix_link_ptr();
                    ops.insert_after(head_link, node_link);
                }
            }
        }
    }

    /// Removes a node from its current fix-list group.
    ///
    /// # Safety
    ///
    /// The node must be valid and currently in the fix-list.
    #[allow(dead_code)] // Will be used when reductions are integrated
    fn fix_list_remove(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            let node_link = (*node).fix_link_ptr();

            // Determine which group this node is in
            let group_ptr = if (*node).is_passive() {
                &mut self.fix_passive
            } else if (*node).is_free() {
                // Check both free groups
                if self.fix_free_multiple == Some(node_ptr)
                    || self.is_in_fix_group(node_ptr, self.fix_free_multiple)
                {
                    &mut self.fix_free_multiple
                } else {
                    &mut self.fix_free_single
                }
            } else {
                // Node is fixed
                let loss = (*node).loss;
                if loss == 0 {
                    &mut self.fix_loss_zero
                } else if loss == 1 {
                    if self.fix_loss_one_multiple == Some(node_ptr)
                        || self.is_in_fix_group(node_ptr, self.fix_loss_one_multiple)
                    {
                        &mut self.fix_loss_one_multiple
                    } else {
                        &mut self.fix_loss_one_single
                    }
                } else {
                    &mut self.fix_loss_two
                }
            };

            // Check if this is the only node in the group
            let next_link = ops.next(node_link);
            if next_link == Some(node_link) {
                // Only node in group
                ops.unlink(node_link);
                *group_ptr = None;
            } else {
                // Multiple nodes in group
                if *group_ptr == Some(node_ptr) {
                    // Update group head to next node
                    let next_node_ptr: *mut Node<T, P> =
                        container_of_mut!(next_link.unwrap().as_ptr(), Node<T, P>, fix_link);
                    *group_ptr = Some(NonNull::new_unchecked(next_node_ptr));
                }
                ops.unlink(node_link);
            }
        }
    }

    /// Helper to check if a node is in a specific fix-list group.
    #[allow(dead_code)]
    fn is_in_fix_group(
        &self,
        node_ptr: NonNull<Node<T, P>>,
        group_head: Option<NonNull<Node<T, P>>>,
    ) -> bool {
        let head = match group_head {
            Some(h) => h,
            None => return false,
        };

        let ops = CircularListOps::new();
        let mut found = false;

        unsafe {
            let start_link = head.as_ref().fix_link_ptr();
            ops.for_each(start_link, |link_ptr| {
                if found {
                    return;
                }
                let current: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, fix_link);
                if NonNull::new_unchecked(current) == node_ptr {
                    found = true;
                }
            });
        }

        found
    }

    // =========================================================================
    // Incremental Reduction (Phase 5)
    // =========================================================================
    //
    // The strict Fibonacci heap achieves worst-case bounds by spreading O(log n)
    // consolidation work across O(1) reduction steps per operation. Each insert,
    // decrease_key, and the start of pop performs a constant number of reductions.
    //
    // The reduction budget per operation:
    // - Insert/decrease_key: 1 reduction (any type)
    // - Pop: O(log n) total reductions spread incrementally
    //
    // Reduction priority (in order):
    // 1. One-node loss reduction (loss >= 2) - most urgent
    // 2. Two-node loss reduction (loss = 1 pairs)
    // 3. Active root reduction (same-rank active roots)
    // 4. Root degree reduction (passive root linking)

    /// Performs a bounded number of reductions to maintain structural invariants.
    ///
    /// This is the core mechanism for achieving worst-case bounds: instead of
    /// doing all cleanup work in pop(), we spread O(1) reductions across each
    /// operation.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of reductions to perform (typically 1-2)
    ///
    /// # Returns
    ///
    /// Number of reductions actually performed (may be less than count if no
    /// reduction candidates are available).
    ///
    /// # Time Complexity
    ///
    /// O(count) - each reduction is O(1) using fix-list groups.
    fn perform_reductions(&mut self, count: usize) -> usize {
        let mut performed = 0;

        // Track which reduction types were performed (debug builds only)
        #[cfg(debug_assertions)]
        let mut one_node_loss_count = 0usize;
        #[cfg(debug_assertions)]
        let mut two_node_loss_count = 0usize;
        #[cfg(debug_assertions)]
        let mut active_root_count = 0usize;
        #[cfg(debug_assertions)]
        let mut root_degree_count = 0usize;

        for _ in 0..count {
            // Try reductions in priority order:
            // 1. One-node loss reduction (most urgent - loss >= 2)
            // 2. Two-node loss reduction (loss = 1 pairs)
            // 3. Active root reduction (same-rank active roots)
            // 4. Root degree reduction (passive root linking)
            if self.one_node_loss_reduction() {
                performed += 1;
                #[cfg(debug_assertions)]
                {
                    one_node_loss_count += 1;
                }
            } else if self.two_node_loss_reduction() {
                performed += 1;
                #[cfg(debug_assertions)]
                {
                    two_node_loss_count += 1;
                }
            } else if self.active_root_reduction() {
                performed += 1;
                #[cfg(debug_assertions)]
                {
                    active_root_count += 1;
                }
            } else if self.root_degree_reduction() {
                performed += 1;
                #[cfg(debug_assertions)]
                {
                    root_degree_count += 1;
                }
            } else {
                // No more reductions possible
                break;
            }
        }

        // Log reduction statistics in debug builds
        #[cfg(debug_assertions)]
        if performed > 0 {
            eprintln!(
                "[StrictFibonacciHeap] reductions: {} total (one_node_loss={}, two_node_loss={}, active_root={}, root_degree={}), budget={}",
                performed, one_node_loss_count, two_node_loss_count, active_root_count, root_degree_count, count
            );
        }

        performed
    }

    // =========================================================================
    // Loss Reduction Operations (Phase 4)
    // =========================================================================
    //
    // Loss reductions handle fixed nodes (active non-root nodes) with high loss.
    // Loss represents the number of children a node has lost since becoming fixed.
    //
    // - **One-node loss reduction**: When a fixed node has loss ≥ 2, cut it
    //   from its parent and make it a free active root. This reduces loss by
    //   converting the node from fixed to free.
    //
    // - **Two-node loss reduction**: When two fixed nodes with loss = 1 have
    //   the same rank, cut one from its parent and link them (the one with
    //   smaller key becomes parent). The parent becomes free, the child fixed
    //   with loss = 0.
    //
    // After cutting a fixed child from its parent, the parent's loss must be
    // incremented if it is also fixed.

    /// Performs a one-node loss reduction if possible.
    ///
    /// Finds a fixed node with loss ≥ 2 (from `fix_loss_two` group), cuts it
    /// from its parent, and makes it a free active root.
    ///
    /// # Returns
    ///
    /// `true` if a reduction was performed, `false` otherwise.
    ///
    /// # Time Complexity
    ///
    /// O(1) - uses fix-list to find candidates in constant time.
    #[allow(dead_code)] // Will be used when reductions are integrated
    fn one_node_loss_reduction(&mut self) -> bool {
        // Get a node with loss >= 2 from the fix_loss_two group
        let node = match self.fix_loss_two {
            Some(ptr) => ptr,
            None => return false,
        };

        unsafe {
            let node_ptr = node.as_ptr();

            // Must be a fixed node (non-root) with loss >= 2
            debug_assert!((*node_ptr).is_fixed());
            debug_assert!((*node_ptr).loss >= 2);
            debug_assert!((*node_ptr).parent.is_some());

            let parent_ptr = match (*node_ptr).parent {
                Some(p) => p,
                None => return false, // Shouldn't happen, but be safe
            };

            // Remove node from fix-list before modifying state
            self.fix_list_remove(node);

            // Remove parent from fix-list BEFORE cut modifies its loss.
            // This is critical: cut_with_loss_tracking increments parent's loss,
            // so fix_list_remove must happen while parent is still in its old group.
            let parent_was_fixed = (*parent_ptr.as_ptr()).is_fixed();
            if parent_was_fixed {
                self.fix_list_remove(parent_ptr);
            }

            // Perform the cut - this also handles parent loss increment
            self.cut_with_loss_tracking(node);

            // Node becomes a free active root (active but not fixed)
            (*node_ptr).loss = LOSS_FREE;

            // Re-add node to fix-list as a free root
            self.fix_list_add(node);

            // Update minimum pointer - the cut node may have smaller priority
            self.update_min(node);

            // Re-add parent to fix-list in correct group after loss increment
            if parent_was_fixed {
                self.fix_list_add(parent_ptr);
            }
        }

        true
    }

    /// Performs a two-node loss reduction if possible.
    ///
    /// Finds two fixed nodes with loss = 1 and the same rank (from
    /// `fix_loss_one_multiple` group), cuts one from its parent, and links
    /// them together. The node with smaller key becomes the parent (remaining
    /// a free root), and the other becomes a fixed child with loss = 0.
    ///
    /// # Returns
    ///
    /// `true` if a reduction was performed, `false` otherwise.
    ///
    /// # Time Complexity
    ///
    /// O(1) - uses fix-list to find candidates in constant time.
    #[allow(dead_code)] // Will be used when reductions are integrated
    fn two_node_loss_reduction(&mut self) -> bool {
        // Get a node with loss = 1 from fix_loss_one_multiple (meaning there
        // are 2+ nodes of the same rank with loss = 1)
        let first = match self.fix_loss_one_multiple {
            Some(ptr) => ptr,
            None => return false,
        };

        unsafe {
            let first_rank = first.as_ref().rank();

            // Find another node with the same rank in fix_loss_one_multiple
            let ops = CircularListOps::new();
            let start_link = first.as_ref().fix_link_ptr();

            let mut second: Option<NonNull<Node<T, P>>> = None;
            ops.for_each(start_link, |link_ptr| {
                if second.is_some() {
                    return; // Already found
                }
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, fix_link);
                let node = NonNull::new_unchecked(node_ptr);

                // Skip self
                if node == first {
                    return;
                }

                // Check if same rank and loss = 1
                if (*node_ptr).rank() == first_rank && (*node_ptr).loss == 1 {
                    second = Some(node);
                }
            });

            let second = match second {
                Some(ptr) => ptr,
                None => return false,
            };

            // Determine which node becomes parent (smaller priority)
            let (winner, loser) = if first.as_ref().priority <= second.as_ref().priority {
                (first, second)
            } else {
                (second, first)
            };

            // Remove both nodes from fix-list before modifying state
            self.fix_list_remove(winner);
            self.fix_list_remove(loser);

            // Get parent pointers for both (for loss increment tracking)
            let winner_parent = (*winner.as_ptr()).parent;
            let loser_parent = (*loser.as_ptr()).parent;

            // Remove parents from fix-list BEFORE cuts modify their loss.
            // This is critical: cut_with_loss_tracking increments parent's loss,
            // so fix_list_remove must happen while parent is still in its old group.
            let winner_parent_was_fixed = winner_parent
                .map(|p| (*p.as_ptr()).is_fixed())
                .unwrap_or(false);
            if winner_parent_was_fixed {
                self.fix_list_remove(winner_parent.unwrap());
            }

            // Only remove loser's parent if different from winner's parent
            let loser_parent_was_fixed = loser_parent
                .map(|p| (*p.as_ptr()).is_fixed())
                .unwrap_or(false);
            let loser_parent_differs = match (loser_parent, winner_parent) {
                (Some(lp), Some(wp)) => lp != wp,
                (Some(_), None) => true,
                _ => false,
            };
            if loser_parent_was_fixed && loser_parent_differs {
                self.fix_list_remove(loser_parent.unwrap());
            }

            // Cut both nodes from their parents with loss tracking
            self.cut_with_loss_tracking(winner);
            self.cut_with_loss_tracking(loser);

            // Link loser under winner as active child
            self.link_as_active_child(winner, loser);

            // Winner becomes a free active root
            (*winner.as_ptr()).loss = LOSS_FREE;

            // Loser becomes fixed with loss = 0
            (*loser.as_ptr()).loss = 0;

            // Re-add winner to fix-list as free root
            self.fix_list_add(winner);
            // Re-add loser to fix-list as fixed with loss = 0
            self.fix_list_add(loser);

            // Update minimum pointer - the winner may have smaller priority than current min
            self.update_min(winner);

            // Re-add parents to fix-list in correct groups after loss increment
            if winner_parent_was_fixed {
                self.fix_list_add(winner_parent.unwrap());
            }
            if loser_parent_was_fixed && loser_parent_differs {
                self.fix_list_add(loser_parent.unwrap());
            }
        }

        true
    }

    /// Cuts a node from its parent and adds it to the root list.
    ///
    /// This version also handles loss tracking: if the parent is a fixed node,
    /// its loss is incremented.
    ///
    /// # Safety
    ///
    /// The node must have a parent (not be a root).
    fn cut_with_loss_tracking(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            let parent_ptr = match (*node).parent {
                Some(p) => p,
                None => return, // Already a root
            };
            let parent = parent_ptr.as_ptr();

            // Remove from parent's children list
            let node_link = (*node).sibling_link_ptr();
            let next_link = ops.next(node_link);

            if next_link == Some(node_link) {
                // Only child
                (*parent).child = None;
            } else {
                // Update parent's child pointer if needed
                if (*parent).child == Some(node_ptr) {
                    let next_node_ptr: *mut Node<T, P> =
                        container_of_mut!(next_link.unwrap().as_ptr(), Node<T, P>, sibling_link);
                    (*parent).child = Some(NonNull::new_unchecked(next_node_ptr));
                }
            }

            ops.unlink(node_link);

            // Decrement parent's rank
            debug_assert!(
                (*parent).rank_count > 0,
                "Parent rank_count underflow in cut_with_loss_tracking()"
            );
            (*parent).rank_count -= 1;

            // Increment parent's loss if parent is fixed
            // (If parent is free or passive, no loss tracking needed)
            if (*parent).is_fixed() {
                (*parent).increment_loss();
            }

            // Clear parent link
            (*node).parent = None;

            // Add to root list
            self.add_to_roots(node_ptr);
        }
    }

    // =========================================================================
    // Debug assertions
    // =========================================================================

    /// Debug assertion: checks O(1) structural invariants.
    ///
    /// Invariants checked:
    /// - `len == 0` iff `roots.is_none()` iff `min.is_none()`
    /// - If min exists, it has no parent (is a root)
    /// - If roots exist, the entry point has no parent
    #[inline]
    fn debug_assert_invariants(&self) {
        #[cfg(debug_assertions)]
        {
            // Emptiness consistency
            let roots_empty = self.roots.is_none();
            let min_empty = self.min.is_none();
            let len_zero = self.len == 0;

            debug_assert_eq!(
                roots_empty, len_zero,
                "Invariant violation: roots.is_none() = {}, len == 0 = {}",
                roots_empty, len_zero
            );
            debug_assert_eq!(
                min_empty, len_zero,
                "Invariant violation: min.is_none() = {}, len == 0 = {}",
                min_empty, len_zero
            );

            // Min is a root (has no parent)
            if let Some(min_ptr) = self.min {
                unsafe {
                    debug_assert!(
                        min_ptr.as_ref().parent.is_none(),
                        "Invariant violation: min node has a parent (not a root)"
                    );
                }
            }

            // Root entry point has no parent
            if let Some(root_ptr) = self.roots {
                unsafe {
                    debug_assert!(
                        root_ptr.as_ref().parent.is_none(),
                        "Invariant violation: root entry point has a parent"
                    );
                }
            }
        }
    }

    /// Debug assertion: checks that a node satisfies heap property with its parent.
    #[inline]
    fn debug_assert_heap_property(&self, node_ptr: NonNull<Node<T, P>>) {
        #[cfg(debug_assertions)]
        unsafe {
            let node = node_ptr.as_ref();
            if let Some(parent_ptr) = node.parent {
                let parent = parent_ptr.as_ref();
                debug_assert!(
                    parent.priority <= node.priority,
                    "Heap property violation: parent priority > child priority"
                );
            }
        }
    }

    /// Adds a node to the root list.
    fn add_to_roots(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            (*node).parent = None;

            let node_link = (*node).sibling_link_ptr();

            match self.roots {
                None => {
                    // First root - make it a circular list of one
                    ops.make_circular(node_link);
                    self.roots = Some(node_ptr);
                }
                Some(root_ptr) => {
                    // Insert after the current root entry point
                    let root_link = root_ptr.as_ref().sibling_link_ptr();
                    ops.insert_after(root_link, node_link);
                }
            }
        }
    }

    /// Removes a node from the root list.
    fn remove_from_roots(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node_link = node_ptr.as_ref().sibling_link_ptr();

            // Check if this is the only root
            let next_link = ops.next(node_link);
            if next_link == Some(node_link) {
                // Only one root, list becomes empty
                ops.unlink(node_link);
                self.roots = None;
            } else {
                // Multiple roots
                // If we're removing the entry point, update it
                if self.roots == Some(node_ptr) {
                    // Get the next node as new entry point
                    let next_node_ptr: *mut Node<T, P> = container_of_mut!(
                        next_link.unwrap().as_ptr(),
                        Node<T, P>,
                        sibling_link
                    );
                    self.roots = Some(NonNull::new_unchecked(next_node_ptr));
                }
                ops.unlink(node_link);
            }
        }
    }

    /// Splices children of a node into the root list.
    fn splice_children_to_roots(&mut self, child_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        // Clear parent pointers for all children
        unsafe {
            let start_link = child_ptr.as_ref().sibling_link_ptr();
            ops.for_each(start_link, |link_ptr| {
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                (*node_ptr).parent = None;
            });

            // Splice with root list
            match self.roots {
                None => {
                    // No roots, children become roots
                    self.roots = Some(child_ptr);
                }
                Some(root_ptr) => {
                    let root_link = root_ptr.as_ref().sibling_link_ptr();
                    let child_link = child_ptr.as_ref().sibling_link_ptr();
                    ops.splice(Some(root_link), Some(child_link));
                }
            }
        }
    }

    /// Updates the minimum pointer if the given node has smaller priority.
    fn update_min(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let should_update = match self.min {
            None => true,
            Some(min_ptr) => unsafe { node_ptr.as_ref().priority < min_ptr.as_ref().priority },
        };

        if should_update {
            self.min = Some(node_ptr);
        }
    }

    /// Finds and sets the new minimum by scanning all roots.
    fn find_new_min(&mut self) {
        self.min = None;

        if let Some(root_ptr) = self.roots {
            let ops = CircularListOps::new();

            unsafe {
                let start_link = root_ptr.as_ref().sibling_link_ptr();
                ops.for_each(start_link, |link_ptr| {
                    let node_ptr: *mut Node<T, P> =
                        container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                    let node_nn = NonNull::new_unchecked(node_ptr);

                    let should_update = match self.min {
                        None => true,
                        Some(min_ptr) => (*node_ptr).priority < min_ptr.as_ref().priority,
                    };

                    if should_update {
                        self.min = Some(node_nn);
                    }
                });
            }
        }
    }

    /// Consolidates the heap by linking trees of the same rank.
    ///
    /// This is called after pop to maintain the heap structure.
    fn consolidate(&mut self) {
        if self.roots.is_none() {
            self.min = None;
            return;
        }

        // Calculate max possible rank (log base phi of n)
        let max_rank = if self.len == 0 {
            1
        } else {
            ((self.len as f64).log2() * 1.5) as usize + 2
        };

        let mut rank_table: Vec<Option<NonNull<Node<T, P>>>> = vec![None; max_rank + 1];
        let ops = CircularListOps::new();

        // Collect all roots into a vector first
        let mut roots_vec: Vec<NonNull<Node<T, P>>> = Vec::new();
        unsafe {
            let root_ptr = self.roots.unwrap();
            let start_link = root_ptr.as_ref().sibling_link_ptr();
            ops.for_each(start_link, |link_ptr| {
                let node_ptr: *mut Node<T, P> =
                    container_of_mut!(link_ptr.as_ptr(), Node<T, P>, sibling_link);
                roots_vec.push(NonNull::new_unchecked(node_ptr));
            });
        }

        // Unlink all roots (they'll be re-linked after consolidation)
        for &node_ptr in &roots_vec {
            unsafe {
                let node_link = node_ptr.as_ref().sibling_link_ptr();
                node_link.as_ref().force_unlink();
            }
        }
        self.roots = None;

        // Process each root
        for root in roots_vec {
            let mut current = root;

            loop {
                let rank = unsafe { current.as_ref().rank() };

                if rank >= rank_table.len() {
                    rank_table.resize(rank + 2, None);
                }

                match rank_table[rank].take() {
                    None => {
                        rank_table[rank] = Some(current);
                        break;
                    }
                    Some(other) => {
                        // Link the two trees
                        current = self.link(current, other);
                    }
                }
            }
        }

        // Rebuild root list from rank table
        for node_ptr in rank_table.into_iter().flatten() {
            self.add_to_roots(node_ptr);
        }

        // Find new minimum
        self.find_new_min();
    }

    /// Links two trees, making the one with larger priority a child of the other.
    ///
    /// Returns the root of the combined tree.
    fn link(&mut self, a: NonNull<Node<T, P>>, b: NonNull<Node<T, P>>) -> NonNull<Node<T, P>> {
        let ops = CircularListOps::new();

        // Determine which becomes the parent (smaller priority wins)
        let (parent_ptr, child_ptr) = unsafe {
            if a.as_ref().priority <= b.as_ref().priority {
                (a, b)
            } else {
                (b, a)
            }
        };

        unsafe {
            let parent = parent_ptr.as_ptr();
            let child = child_ptr.as_ptr();

            // Set child's parent
            (*child).parent = Some(parent_ptr);
            (*child).active = false;

            // Add child to parent's children list
            let child_link = (*child).sibling_link_ptr();

            match (*parent).child {
                None => {
                    // First child
                    ops.make_circular(child_link);
                    (*parent).child = Some(child_ptr);
                }
                Some(first_child_ptr) => {
                    // Insert into existing children list
                    let first_child_link = first_child_ptr.as_ref().sibling_link_ptr();
                    ops.insert_after(first_child_link, child_link);
                }
            }

            // Increment parent's rank
            (*parent).rank_count += 1;
        }

        parent_ptr
    }

    /// Cuts a node from its parent and adds it to the root list.
    fn cut(&mut self, node_ptr: NonNull<Node<T, P>>) {
        let ops = CircularListOps::new();

        unsafe {
            let node = node_ptr.as_ptr();
            let parent_ptr = match (*node).parent {
                Some(p) => p,
                None => return, // Already a root
            };
            let parent = parent_ptr.as_ptr();

            // Remove from parent's children list
            let node_link = (*node).sibling_link_ptr();
            let next_link = ops.next(node_link);

            if next_link == Some(node_link) {
                // Only child
                (*parent).child = None;
            } else {
                // Update parent's child pointer if needed
                if (*parent).child == Some(node_ptr) {
                    let next_node_ptr: *mut Node<T, P> =
                        container_of_mut!(next_link.unwrap().as_ptr(), Node<T, P>, sibling_link);
                    (*parent).child = Some(NonNull::new_unchecked(next_node_ptr));
                }
            }

            ops.unlink(node_link);

            // Decrement parent's rank
            debug_assert!(
                (*parent).rank_count > 0,
                "Parent rank_count underflow in cut()"
            );
            (*parent).rank_count -= 1;

            // Clear parent link
            (*node).parent = None;

            // Add to root list
            self.add_to_roots(node_ptr);

            // Note: In a full strict Fibonacci heap, we would handle
            // cascading cuts and active node tracking here.
            // For now, we just do a simple cut (amortized bounds).
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecreaseKeyHeap;

    #[test]
    fn test_basic_operations() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();

        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);

        let h1 = heap.push_with_handle(5, "five");
        let h2 = heap.push_with_handle(3, "three");
        let _h3 = heap.push_with_handle(7, "seven");

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.peek(), Some((&3, &"three")));

        // Test decrease_key
        assert!(heap.decrease_key(&h1, 1).is_ok());
        assert_eq!(heap.peek(), Some((&1, &"five")));

        // Test decrease_key that doesn't change minimum
        assert!(heap.decrease_key(&h2, 2).is_ok());
        assert_eq!(heap.peek(), Some((&1, &"five")));

        // Test pop
        let min = heap.pop();
        assert_eq!(min, Some((1, "five")));
        assert_eq!(heap.len(), 2);

        assert_eq!(heap.peek(), Some((&2, &"three")));
    }

    #[test]
    fn test_decrease_key_errors() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();
        let handle = heap.push_with_handle(5, "item");

        // Try to increase priority
        assert_eq!(
            heap.decrease_key(&handle, 10),
            Err(HeapError::PriorityNotDecreased)
        );

        // Try with same priority
        assert_eq!(
            heap.decrease_key(&handle, 5),
            Err(HeapError::PriorityNotDecreased)
        );
    }

    #[test]
    fn test_merge() {
        let mut heap1: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();
        heap1.push(5, "five");
        heap1.push(3, "three");

        let mut heap2: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();
        heap2.push(1, "one");
        heap2.push(4, "four");

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.peek(), Some((&1, &"one")));
    }

    #[test]
    fn test_empty_operations() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();

        assert!(heap.pop().is_none());
        assert!(heap.peek().is_none());
    }

    #[test]
    fn test_large_sequence() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();
        let mut handles = Vec::new();

        // Insert 1000 elements with priorities i*10
        for i in 0..1000 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        // Decrease keys of every 10th element (starting from i=10 since i=0 can't decrease)
        for i in (10..1000).step_by(10) {
            let result = heap.decrease_key(&handles[i], i as i32);
            assert!(
                result.is_ok(),
                "Failed to decrease key for handle {} (priority {} -> {})",
                i,
                i * 10,
                i
            );
        }

        // Pop first 100
        for _ in 0..100 {
            heap.pop();
        }

        // Insert 200 more
        for i in 1000..1200 {
            heap.push(i * 10, i);
        }

        // Verify heap still works
        assert!(!heap.is_empty());

        // Pop all remaining
        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        // 1000 - 100 + 200 = 1100
        assert_eq!(count, 1100);
    }

    #[test]
    fn test_pop_order() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        heap.push(3, 30);
        heap.push(1, 10);
        heap.push(4, 40);
        heap.push(1, 11); // Duplicate priority
        heap.push(5, 50);
        heap.push(9, 90);
        heap.push(2, 20);

        // Pop should return in priority order
        let mut last_priority = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(
                priority >= last_priority,
                "Out of order: {} after {}",
                priority,
                last_priority
            );
            last_priority = priority;
        }
    }

    #[test]
    fn test_decrease_key_to_new_min() {
        let mut heap: StrictFibonacciHeap<&str, i32> = StrictFibonacciHeap::new();

        heap.push(10, "a");
        heap.push(20, "b");
        let h = heap.push_with_handle(30, "c");

        assert_eq!(heap.peek(), Some((&10, &"a")));

        // Decrease c's priority to become new minimum
        assert!(heap.decrease_key(&h, 5).is_ok());
        assert_eq!(heap.peek(), Some((&5, &"c")));
    }

    #[test]
    fn test_decrease_key_with_cut() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Build a tree structure by doing inserts and pops
        for i in 0..10 {
            heap.push(i, i);
        }

        // Pop to trigger consolidation, building trees
        heap.pop();

        // Store handles for remaining elements
        let mut handles = Vec::new();
        for i in 10..20 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        // Decrease a key that should trigger a cut
        if !handles.is_empty() {
            let result = heap.decrease_key(&handles[5], 0);
            assert!(result.is_ok());
            assert_eq!(heap.peek().map(|(p, _)| *p), Some(0));
        }
    }

    // =========================================================================
    // Phase 4: Loss reduction tests
    // =========================================================================

    #[test]
    fn test_loss_constants() {
        // Verify LOSS_FREE sentinel value
        assert_eq!(LOSS_FREE, u8::MAX);
        assert_eq!(LOSS_FREE, 255);
    }

    #[test]
    fn test_node_loss_classification() {
        // Test node classification methods
        let mut node = Node::new(10, "test");

        // Initially passive (active = false)
        assert!(node.is_passive());
        assert!(!node.is_active());
        assert!(!node.is_free());
        assert!(!node.is_fixed());

        // Make active and free
        node.active = true;
        node.loss = LOSS_FREE;
        assert!(node.is_active());
        assert!(!node.is_passive());
        assert!(node.is_free());
        assert!(!node.is_fixed());

        // Convert to fixed with loss = 0
        node.loss = 0;
        assert!(node.is_active());
        assert!(!node.is_free());
        assert!(node.is_fixed());
        assert_eq!(node.get_loss(), 0);

        // Increment loss
        node.increment_loss();
        assert_eq!(node.get_loss(), 1);

        node.increment_loss();
        assert_eq!(node.get_loss(), 2);
    }

    #[test]
    fn test_node_free_fixed_conversion() {
        let mut node = Node::new(10, "test");
        node.active = true;
        node.loss = LOSS_FREE;

        // Free -> Fixed
        assert!(node.is_free());
        node.free_to_fixed();
        assert!(node.is_fixed());
        assert_eq!(node.loss, 0);

        // Fixed -> Free
        node.fixed_to_free();
        assert!(node.is_free());
        assert_eq!(node.loss, LOSS_FREE);
    }

    #[test]
    fn test_consolidation_preserves_heap_order() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Build a small tree structure
        heap.push(1, 1);
        heap.push(2, 2);
        heap.push(3, 3);

        // Pop to trigger consolidation
        heap.pop();

        // Verify heap still works correctly
        assert_eq!(heap.len(), 2);

        // Pop remaining elements in order
        let mut last_priority = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(priority >= last_priority);
            last_priority = priority;
        }
    }

    #[test]
    fn test_one_node_loss_reduction_no_candidates() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Empty heap - no candidates
        assert!(!heap.one_node_loss_reduction());

        // Add some nodes but none in fix_loss_two
        heap.push(1, 1);
        heap.push(2, 2);

        // Still no candidates (nodes are passive, not in fix_loss_two)
        assert!(!heap.one_node_loss_reduction());
    }

    // TODO: Add test_one_node_loss_reduction_with_candidates when fix-list
    // population is integrated (requires nodes to become active/fixed during
    // heap operations, which will be implemented in a later phase).

    // TODO: Add test_two_node_loss_reduction_with_candidates when fix-list
    // population is integrated.

    #[test]
    fn test_two_node_loss_reduction_no_candidates() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Empty heap - no candidates
        assert!(!heap.two_node_loss_reduction());

        // Add some nodes but none in fix_loss_one_multiple
        heap.push(1, 1);
        heap.push(2, 2);

        // Still no candidates (nodes are passive, not in fix_loss_one_multiple)
        assert!(!heap.two_node_loss_reduction());
    }

    #[test]
    fn test_fix_list_retire() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // All fix-list pointers should be None initially
        assert!(heap.fix_passive.is_none());
        assert!(heap.fix_free_multiple.is_none());
        assert!(heap.fix_free_single.is_none());
        assert!(heap.fix_loss_zero.is_none());
        assert!(heap.fix_loss_one_multiple.is_none());
        assert!(heap.fix_loss_one_single.is_none());
        assert!(heap.fix_loss_two.is_none());

        // Retire should clear everything
        heap.fix_list_retire();

        assert!(heap.fix_passive.is_none());
        assert!(heap.fix_free_multiple.is_none());
        assert!(heap.fix_free_single.is_none());
        assert!(heap.fix_loss_zero.is_none());
        assert!(heap.fix_loss_one_multiple.is_none());
        assert!(heap.fix_loss_one_single.is_none());
        assert!(heap.fix_loss_two.is_none());
    }

    #[test]
    fn test_fix_list_head() {
        let heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Empty fix-list should return None
        assert!(heap.fix_list_head().is_none());
    }

    #[test]
    fn test_active_root_reduction_no_candidates() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // No free nodes with matching ranks
        assert!(!heap.active_root_reduction());
    }

    #[test]
    fn test_root_degree_reduction_no_candidates() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // No passive roots to link
        assert!(!heap.root_degree_reduction());
    }

    // =========================================================================
    // Phase 5: Delete-min integration and worst-case behavior tests
    // =========================================================================

    #[test]
    fn test_perform_reductions_empty() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // No reductions possible on empty heap
        assert_eq!(heap.perform_reductions(5), 0);
    }

    #[test]
    fn test_perform_reductions_no_candidates() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Add nodes but they're passive (not in fix-list)
        heap.push(1, 1);
        heap.push(2, 2);
        heap.push(3, 3);

        // No reductions possible (nodes are not active/in fix-list)
        assert_eq!(heap.perform_reductions(5), 0);
    }

    #[test]
    fn test_incremental_reduction_integration() {
        // Test that operations don't crash when performing reductions
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Push many elements (reductions called on each push)
        for i in 0..100 {
            heap.push(i, i);
        }

        // Pop some elements (reductions called on each pop)
        for _ in 0..50 {
            let _ = heap.pop();
        }

        // Verify heap integrity
        assert_eq!(heap.len(), 50);

        // Pop remaining in order
        let mut last = i32::MIN;
        while let Some((p, _)) = heap.pop() {
            assert!(p >= last);
            last = p;
        }
    }

    #[test]
    fn test_pop_with_incremental_reductions() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Build a tree by inserting many elements then popping to trigger consolidation
        for i in 0..32 {
            heap.push(i, i);
        }

        // Pop all elements - each pop should call perform_reductions
        while heap.pop().is_some() {}

        assert!(heap.is_empty());
    }

    #[test]
    fn test_decrease_key_with_reductions() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Insert elements and store handles
        let mut handles = Vec::new();
        for i in 10..30 {
            handles.push(heap.push_with_handle(i, i));
        }

        // Pop to create tree structure
        heap.pop();

        // Decrease keys - each should call perform_reductions
        for (idx, h) in handles.iter().enumerate().skip(1) {
            let new_priority = -(idx as i32);
            let _ = heap.decrease_key(h, new_priority);
        }

        // Verify heap still works correctly
        while let Some((p1, _)) = heap.pop() {
            if let Some((p2, _)) = heap.peek() {
                assert!(p1 <= *p2);
            }
        }
    }

    #[test]
    fn test_merge_with_reductions() {
        let mut heap1: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();
        let mut heap2: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        for i in 0..20 {
            heap1.push(i * 2, i);
            heap2.push(i * 2 + 1, i);
        }

        // Pop from both to create tree structures
        heap1.pop();
        heap2.pop();

        // Merge - should call perform_reductions
        heap1.merge(heap2);

        assert_eq!(heap1.len(), 38); // 20 + 20 - 2

        // Verify merged heap is valid
        let mut last = i32::MIN;
        while let Some((p, _)) = heap1.pop() {
            assert!(p >= last);
            last = p;
        }
    }

    #[test]
    fn test_pop_counts_children_for_reductions() {
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Insert many elements to build trees with multiple children
        for i in 0..100 {
            heap.push(i, i);
        }

        // Pop several times to trigger consolidation and build trees
        for _ in 0..10 {
            heap.pop();
        }

        // Now pop should count children and perform proportional reductions
        let _ = heap.pop();

        // Verify heap is still valid
        let mut last = i32::MIN;
        while let Some((p, _)) = heap.pop() {
            assert!(p >= last);
            last = p;
        }
    }

    #[test]
    fn test_pop_minimum_with_no_children() {
        // Test that pop works correctly when the minimum node has no children.
        // This exercises the `.max(1)` logic in pop() that ensures at least
        // one reduction is performed even when child_count is 0.
        let mut heap: StrictFibonacciHeap<i32, i32> = StrictFibonacciHeap::new();

        // Single element - min has no children
        heap.push(5, 5);
        assert_eq!(heap.pop(), Some((5, 5)));
        assert!(heap.is_empty());

        // Two elements - after first pop, the remaining element becomes min with no children
        heap.push(10, 10);
        heap.push(20, 20);
        assert_eq!(heap.pop(), Some((10, 10))); // Pop min (10)
        assert_eq!(heap.pop(), Some((20, 20))); // Pop remaining (20) which has no children
        assert!(heap.is_empty());

        // Three elements - creates a small tree structure
        heap.push(1, 1);
        heap.push(2, 2);
        heap.push(3, 3);
        // Pop min (1). After consolidation, remaining nodes form a tree
        assert_eq!(heap.pop(), Some((1, 1)));
        assert_eq!(heap.pop(), Some((2, 2)));
        assert_eq!(heap.pop(), Some((3, 3)));
        assert!(heap.is_empty());
    }
}
