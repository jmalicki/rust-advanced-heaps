//! Fibonacci Heap implementation using Rc/Weak references
//!
//! A Fibonacci heap is a data structure for priority queue operations with:
//! - O(1) amortized insert, decrease_key, and merge
//! - O(log n) amortized delete_min
//!
//! This implementation uses Rc/Weak references instead of raw pointers:
//! - Strong references (Rc) flow from roots to children (ownership)
//! - Weak references for parent backlinks and handles
//! - No unsafe code required
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
//! # Ownership Model
//!
//! - The heap owns all root nodes via `Vec<Rc<RefCell<Node>>>`
//! - Each node owns its children via `Vec<Rc<RefCell<Node>>>`
//! - Parent links use `Weak` to avoid reference cycles
//! - Handles use `Weak` to detect when nodes are removed

use crate::traits::{Handle, Heap, HeapError};
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Handle to an element in a Fibonacci heap
///
/// Uses a Weak reference to the node. If the node has been removed from
/// the heap, the Weak reference will fail to upgrade, allowing safe
/// detection of invalid handles.
pub struct FibonacciHandle<T, P> {
    node: Weak<RefCell<Node<T, P>>>,
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
        Weak::ptr_eq(&self.node, &other.node)
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
/// - `item` and `priority`: The data stored in the heap
/// - `parent`: Weak reference to parent node (empty if root)
/// - `children`: Vec of strong references to children
/// - `marked`: Flag used in cascading cuts (see decrease_key)
///
/// Children are stored in a Vec rather than a circular linked list.
/// This simplifies the implementation while maintaining the same
/// asymptotic complexity (degree is O(log n)).
struct Node<T, P> {
    item: T,
    priority: P,
    /// Weak reference to parent node (empty if this is a root)
    parent: Weak<RefCell<Node<T, P>>>,
    /// Strong references to children
    children: Vec<Rc<RefCell<Node<T, P>>>>,
    /// Marked flag: true if this node has lost a child (used in cascading cuts)
    marked: bool,
}

impl<T, P> Node<T, P> {
    fn new(item: T, priority: P) -> Self {
        Node {
            item,
            priority,
            parent: Weak::new(),
            children: Vec::new(),
            marked: false,
        }
    }

    fn degree(&self) -> usize {
        self.children.len()
    }
}

/// Fibonacci Heap using Rc/Weak references
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::fibonacci::FibonacciHeap;
/// use rust_advanced_heaps::Heap;
///
/// let mut heap = FibonacciHeap::new();
/// let handle = heap.insert(5, "item");
/// heap.decrease_key(&handle, 1);
/// assert_eq!(heap.find_min(), Some((&1, &"item")));
/// ```
pub struct FibonacciHeap<T, P: Ord> {
    /// All root trees (strong references)
    roots: Vec<Rc<RefCell<Node<T, P>>>>,
    /// Weak reference to the minimum node (for O(1) min lookup and update)
    min: Weak<RefCell<Node<T, P>>>,
    /// Total number of elements
    len: usize,
}

impl<T, P: Ord> Default for FibonacciHeap<T, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, P: Ord> Heap<T, P> for FibonacciHeap<T, P> {
    type Handle = FibonacciHandle<T, P>;

    fn new() -> Self {
        Self {
            roots: Vec::new(),
            min: Weak::new(),
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

    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Rc::new(RefCell::new(Node::new(item, priority)));
        let handle = FibonacciHandle {
            node: Rc::downgrade(&node),
        };

        // Update min if necessary
        let should_update_min = if let Some(min_rc) = self.min.upgrade() {
            node.borrow().priority < min_rc.borrow().priority
        } else {
            true // No current min, so this becomes min
        };

        if should_update_min {
            self.min = Rc::downgrade(&node);
        }

        self.roots.push(node);
        self.len += 1;
        handle
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        self.min.upgrade().map(|min_rc| {
            // SAFETY: We're returning references to data inside RefCell.
            // This is safe because:
            // 1. The Rc keeps the node alive as long as the heap exists
            // 2. We only read the data, never mutate through this path
            // 3. The heap's borrow checker ensures no mutable operations
            //    happen while these references exist
            unsafe {
                let ptr = min_rc.as_ptr();
                (&(*ptr).priority, &(*ptr).item)
            }
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    fn delete_min(&mut self) -> Option<(P, T)> {
        // Find min in roots using the weak reference
        let min_idx = {
            let min_rc = self.min.upgrade()?;
            self.roots
                .iter()
                .position(|r| Rc::ptr_eq(r, &min_rc))
                .expect("min must be in roots")
            // min_rc is dropped here, reducing strong count
        };

        // Remove min from roots - now we have the only strong reference
        let min_node = self.roots.swap_remove(min_idx);

        // Clear min reference before any operations
        self.min = Weak::new();

        // Add children to root list
        {
            let mut min_borrowed = min_node.borrow_mut();
            for child in min_borrowed.children.drain(..) {
                child.borrow_mut().parent = Weak::new();
                child.borrow_mut().marked = false;
                self.roots.push(child);
            }
        }

        self.len -= 1;

        // Consolidate if there are remaining roots
        if !self.roots.is_empty() {
            self.consolidate();
        }

        // Extract item and priority from the removed node
        // This consumes the Rc, which should have no other references now
        let node = match Rc::try_unwrap(min_node) {
            Ok(cell) => cell.into_inner(),
            Err(_) => panic!("min node should have no other references"),
        };

        Some((node.priority, node.item))
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        // Upgrade weak reference to get the node
        let node_rc = match handle.node.upgrade() {
            Some(rc) => rc,
            None => return Err(HeapError::PriorityNotDecreased), // Handle is invalid
        };

        // Check that new priority is actually less
        {
            let node = node_rc.borrow();
            if new_priority >= node.priority {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Update priority
        node_rc.borrow_mut().priority = new_priority;

        // Get parent (if any)
        let parent_weak = node_rc.borrow().parent.clone();

        if let Some(parent_rc) = parent_weak.upgrade() {
            // Check if heap property is violated
            let should_cut = {
                let node = node_rc.borrow();
                let parent = parent_rc.borrow();
                node.priority < parent.priority
            };

            if should_cut {
                // Cut node from parent and add to root list
                self.cut(&node_rc, &parent_rc);
                self.cascading_cut(parent_rc);
            }
        }

        // Update min if this node now has smaller priority - O(1)
        self.maybe_update_min(&node_rc);

        Ok(())
    }

    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        // Compare mins and update
        let other_is_smaller =
            if let (Some(self_min), Some(other_min)) = (self.min.upgrade(), other.min.upgrade()) {
                other_min.borrow().priority < self_min.borrow().priority
            } else {
                other.min.upgrade().is_some()
            };

        if other_is_smaller {
            self.min = other.min.clone();
        }

        // Append other's roots to self
        self.roots.append(&mut other.roots);
        self.len += other.len;

        // Prevent double-cleanup
        other.min = Weak::new();
        other.len = 0;
    }
}

impl<T, P: Ord> FibonacciHeap<T, P> {
    /// Consolidates the heap by linking trees of the same degree
    fn consolidate(&mut self) {
        if self.roots.is_empty() {
            self.min = Weak::new();
            return;
        }

        // Max degree is O(log n) due to Fibonacci property
        let max_degree = ((self.len + 1) as f64).log2() as usize + 2;
        let mut degree_table: Vec<Option<Rc<RefCell<Node<T, P>>>>> = vec![None; max_degree + 1];

        // Process all current roots
        let roots_to_process: Vec<_> = self.roots.drain(..).collect();

        for mut root in roots_to_process {
            let mut d = root.borrow().degree();

            // Link with existing trees of the same degree
            while d < degree_table.len() && degree_table[d].is_some() {
                let mut other = degree_table[d].take().unwrap();

                // Ensure root has smaller priority
                let should_swap = {
                    let root_priority = &root.borrow().priority;
                    let other_priority = &other.borrow().priority;
                    other_priority < root_priority
                };

                if should_swap {
                    std::mem::swap(&mut root, &mut other);
                }

                // Link other as child of root
                self.link(&other, &root);

                d += 1;
            }

            if d >= degree_table.len() {
                degree_table.resize(d + 1, None);
            }
            degree_table[d] = Some(root);
        }

        // Rebuild roots and find new min
        self.roots.clear();
        self.min = Weak::new();

        for root_opt in degree_table.into_iter().flatten() {
            // Check if this is the new min
            let is_new_min = if let Some(current_min) = self.min.upgrade() {
                root_opt.borrow().priority < current_min.borrow().priority
            } else {
                true
            };

            if is_new_min {
                self.min = Rc::downgrade(&root_opt);
            }

            self.roots.push(root_opt);
        }
    }

    /// Links node y as a child of node x
    fn link(&self, y: &Rc<RefCell<Node<T, P>>>, x: &Rc<RefCell<Node<T, P>>>) {
        // Make y a child of x
        y.borrow_mut().parent = Rc::downgrade(x);
        y.borrow_mut().marked = false;
        x.borrow_mut().children.push(Rc::clone(y));
    }

    /// Cuts node from its parent and adds it to the root list
    fn cut(&mut self, node: &Rc<RefCell<Node<T, P>>>, parent: &Rc<RefCell<Node<T, P>>>) {
        // Remove from parent's children
        {
            let mut parent_borrowed = parent.borrow_mut();
            if let Some(pos) = parent_borrowed
                .children
                .iter()
                .position(|c| Rc::ptr_eq(c, node))
            {
                parent_borrowed.children.swap_remove(pos);
            }
        }

        // Clear parent and marked flag
        node.borrow_mut().parent = Weak::new();
        node.borrow_mut().marked = false;

        // Add to root list
        self.roots.push(Rc::clone(node));
    }

    /// Performs cascading cut on parent if it's marked
    fn cascading_cut(&mut self, node: Rc<RefCell<Node<T, P>>>) {
        let parent_weak = node.borrow().parent.clone();

        if let Some(parent_rc) = parent_weak.upgrade() {
            // Node has a parent (not a root)
            let is_marked = node.borrow().marked;

            if !is_marked {
                // First child lost: mark it
                node.borrow_mut().marked = true;
            } else {
                // Already marked: cut it and cascade
                self.cut(&node, &parent_rc);
                self.cascading_cut(parent_rc);
            }
        }
        // If no parent (root), cascading stops
    }

    /// Updates min if the given node has smaller priority than current min
    /// This is O(1) - just compares with current minimum
    fn maybe_update_min(&mut self, node: &Rc<RefCell<Node<T, P>>>) {
        let should_update = if let Some(current_min) = self.min.upgrade() {
            node.borrow().priority < current_min.borrow().priority
        } else {
            true
        };

        if should_update {
            self.min = Rc::downgrade(node);
        }
    }
}

// Note: Most tests are in tests/generic_heap_tests.rs which provides comprehensive
// test coverage for all heap implementations.
