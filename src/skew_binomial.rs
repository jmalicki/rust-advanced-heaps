//! Skew Binomial Heap implementation
//!
//! A skew binomial heap extends binomial heaps with additional flexibility:
//! - O(1) insert and merge
//! - O(log n) delete_min
//! - O(log n) decrease_key
//!
//! Skew binomial heaps allow more flexible tree structures than standard
//! binomial heaps while maintaining efficient operations.
//!
//! This implementation uses Rc/Weak references instead of raw pointers,
//! providing memory safety for tree structure management.
//!
//! # Storage Strategies
//!
//! This heap supports pluggable storage strategies for priority values via the
//! [`optlike`] crate:
//!
//! - **`Optimized`** (default): Uses sentinel-based storage that halves memory
//!   usage for integer/float priorities. The sentinel value (e.g., `i32::MAX`)
//!   cannot be used as a priority.
//!
//! - **`PlainOption`**: Uses standard `Option<P>` storage with no restrictions
//!   on priority values. Useful for benchmarking or when the full priority
//!   range is needed.
//!
//! # Why Skew Binomial Heaps?
//!
//! Skew binomial heaps extend binomial heaps to achieve O(1) worst-case insertion
//! (vs O(log n) for binomial heaps). The key innovation is the **skew link**: a
//! special linking operation that can combine three trees at once.
//!
//! The "skew" in the name refers to the skew binary number system used to represent
//! tree sizes, which allows constant-time increment operations. Originally designed
//! for purely functional programming languages, this structure demonstrates that
//! optimal priority queue bounds are achievable without mutation.
//!
//! # References
//!
//! - Brodal, G. S., & Okasaki, C. (1996). "Optimal purely functional priority queues."
//!   *Journal of Functional Programming*, 6(6), 839-857.
//!   [Cambridge](https://doi.org/10.1017/S095679680000201X)
//! - [Wikipedia: Skew binomial heap](https://en.wikipedia.org/wiki/Skew_binomial_heap)

use crate::traits::{DecreaseKeyHeap, Handle, Heap, HeapError};
use optlike::{OptLike, Optimized, PlainOption, StorageStrategy};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::{Rc, Weak};

/// Type alias for strong node reference
type NodeRef<T, P, S> = Rc<RefCell<Node<T, P, S>>>;
/// Type alias for weak node reference (used for parent backlinks)
type WeakNodeRef<T, P, S> = Weak<RefCell<Node<T, P, S>>>;
/// Type alias for optional strong node reference
type OptNodeRef<T, P, S> = Option<NodeRef<T, P, S>>;

/// Handle to an element in a Skew binomial heap
#[derive(Debug)]
pub struct SkewBinomialHandle<T, P: OptLike<S>, S: StorageStrategy = Optimized> {
    node: WeakNodeRef<T, P, S>,
}

impl<T, P: OptLike<S>, S: StorageStrategy> Clone for SkewBinomialHandle<T, P, S> {
    fn clone(&self) -> Self {
        SkewBinomialHandle {
            node: self.node.clone(),
        }
    }
}

impl<T, P: OptLike<S>, S: StorageStrategy> PartialEq for SkewBinomialHandle<T, P, S> {
    fn eq(&self, other: &Self) -> bool {
        self.node.ptr_eq(&other.node)
    }
}

impl<T, P: OptLike<S>, S: StorageStrategy> Eq for SkewBinomialHandle<T, P, S> {}

impl<T, P: OptLike<S>, S: StorageStrategy> Handle for SkewBinomialHandle<T, P, S> {}

struct Node<T, P: OptLike<S>, S: StorageStrategy = Optimized> {
    item: Option<T>,
    priority: P::Storage,
    parent: WeakNodeRef<T, P, S>,
    child: OptNodeRef<T, P, S>,
    sibling: OptNodeRef<T, P, S>,
    rank: usize,
    skew: bool,
}

/// Skew Binomial Heap with configurable priority storage strategy.
///
/// The strategy parameter `S` controls how priorities are stored:
/// - [`Optimized`]: Uses sentinel-based storage (half the memory for integers)
/// - [`PlainOption`]: Uses standard `Option<P>` (no value restrictions)
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::Heap;
/// use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
///
/// // Uses optimized storage by default
/// let mut heap: SkewBinomialHeap<&str, i32> = SkewBinomialHeap::new();
/// heap.push(3, "three");
/// heap.push(1, "one");
/// assert_eq!(heap.pop(), Some((1, "one")));
/// ```
pub struct SkewBinomialHeapImpl<T, P: OptLike<S>, S: StorageStrategy = Optimized> {
    trees: Vec<OptNodeRef<T, P, S>>,
    min: OptNodeRef<T, P, S>,
    len: usize,
    _strategy: PhantomData<S>,
}

/// Skew Binomial Heap with plain Option storage (default).
///
/// This is the default type alias for backward compatibility. It uses
/// standard `Option<P>` for priority storage with no restrictions on
/// priority values.
///
/// For optimized storage with integer/float priorities, use
/// [`SkewBinomialHeapOpt`] instead.
pub type SkewBinomialHeap<T, P> = SkewBinomialHeapImpl<T, P, PlainOption>;

/// Skew Binomial Heap with optimized priority storage.
///
/// This variant uses sentinel-based storage for priorities, halving
/// memory usage for integer and float priority types:
/// - Integers use `NonMax*` types (sentinel = MAX value)
/// - Floats use NaN as sentinel
///
/// **Note**: The sentinel value cannot be used as a priority:
/// - For integers: `i32::MAX`, `u64::MAX`, etc.
/// - For floats: `NaN`
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::Heap;
/// use rust_advanced_heaps::skew_binomial::SkewBinomialHeapOpt;
///
/// let mut heap: SkewBinomialHeapOpt<&str, i32> = SkewBinomialHeapOpt::new();
/// heap.push(3, "three");
/// heap.push(1, "one");
/// assert_eq!(heap.pop(), Some((1, "one")));
/// ```
pub type SkewBinomialHeapOpt<T, P> = SkewBinomialHeapImpl<T, P, Optimized>;

impl<T, P: OptLike<S> + Ord + Clone, S: StorageStrategy> Heap<T, P>
    for SkewBinomialHeapImpl<T, P, S>
{
    fn new() -> Self {
        Self {
            trees: Vec::new(),
            min: None,
            len: 0,
            _strategy: PhantomData,
        }
    }

    fn is_empty(&self) -> bool {
        self.min.is_none()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn push(&mut self, priority: P, item: T) {
        let _ = self.push_with_handle(priority, item);
    }

    fn peek(&self) -> Option<(&P, &T)> {
        // For PlainOption strategy, we can return a reference to the priority.
        // For Optimized strategy, get_ref returns None because NonMax types
        // store XOR'd values internally.
        self.min.as_ref().and_then(|min_ref| unsafe {
            let ptr = min_ref.as_ptr();
            let item_ref = (*ptr).item.as_ref().unwrap_unchecked();
            // get_ref works for PlainOption (returns Some(&P))
            // but returns None for Optimized (XOR'd storage)
            P::get_ref(&(*ptr).priority).map(|priority_ref| (priority_ref, item_ref))
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        let min_ref = self.min.take()?;

        // Clear from trees
        let rank = min_ref.borrow().rank;
        if rank < self.trees.len() {
            self.trees[rank] = None;
        }

        // Collect all children
        let first_child = min_ref.borrow_mut().child.take();
        let mut children_to_insert: Vec<NodeRef<T, P, S>> = Vec::new();

        if let Some(first) = first_child {
            let mut current = Some(first);
            while let Some(curr) = current {
                let next = curr.borrow_mut().sibling.take();
                curr.borrow_mut().parent = Weak::new();
                children_to_insert.push(curr);
                current = next;
            }
        }

        let (priority, item) = {
            let mut node = min_ref.borrow_mut();
            (
                P::take(&mut node.priority).unwrap(),
                node.item.take().unwrap(),
            )
        };

        drop(min_ref);

        for child in children_to_insert {
            self.insert_tree(child);
        }

        self.find_and_update_min();
        self.len -= 1;

        // Verify invariants after delete_min
        #[cfg(feature = "expensive_verify")]
        {
            let count = self.count_all_nodes();
            assert_eq!(
                count, self.len,
                "Length mismatch after delete_min: counted {} nodes but len is {}",
                count, self.len
            );
        }

        Some((priority, item))
    }

    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        let mut other_trees: Vec<NodeRef<T, P, S>> = Vec::new();
        for tree_opt in other.trees.iter_mut() {
            if let Some(tree) = tree_opt.take() {
                other_trees.push(tree);
            }
        }

        for tree in other_trees {
            self.insert_tree(tree);
        }

        self.len += other.len;
        self.find_and_update_min();

        other.min = None;
        other.len = 0;
    }
}

impl<T, P: OptLike<S> + Ord + Clone, S: StorageStrategy> DecreaseKeyHeap<T, P>
    for SkewBinomialHeapImpl<T, P, S>
{
    type Handle = SkewBinomialHandle<T, P, S>;

    fn push_with_handle(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Rc::new(RefCell::new(Node {
            item: Some(item),
            priority: P::some(priority),
            parent: Weak::new(),
            child: None,
            sibling: None,
            rank: 0,
            skew: true,
        }));

        let handle = SkewBinomialHandle {
            node: Rc::downgrade(&node),
        };

        self.insert_tree(node);
        self.len += 1;

        // Update min pointer AFTER insert_tree, because during insert_tree
        // the node may become a child of another node during linking.
        // find_and_update_min scans roots to find the actual minimum.
        self.find_and_update_min();

        handle
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) -> Result<(), HeapError> {
        let node_ref = handle.node.upgrade().ok_or(HeapError::InvalidHandle)?;

        // Check if node has been deleted (priority is None)
        {
            let node = node_ref.borrow();
            if P::is_none(&node.priority) {
                return Err(HeapError::InvalidHandle);
            }
            if new_priority >= P::get(&node.priority).unwrap() {
                return Err(HeapError::PriorityNotDecreased);
            }
        }

        // Check if node is a root (no parent)
        let has_parent = node_ref.borrow().parent.upgrade().is_some();

        if !has_parent {
            // Node is a root, just update priority and min
            node_ref.borrow_mut().priority = P::some(new_priority);
            self.find_and_update_min();
            return Ok(());
        }

        // Cut the node from its parent and reinsert
        self.cut_and_reinsert(node_ref, new_priority);

        Ok(())
    }
}

impl<T, P: OptLike<S> + Ord + Clone, S: StorageStrategy> SkewBinomialHeapImpl<T, P, S> {
    /// Counts all nodes reachable from the trees array (debug only)
    #[cfg(feature = "expensive_verify")]
    fn count_all_nodes(&self) -> usize {
        let mut count = 0;
        for tree in self.trees.iter().flatten() {
            count += Self::count_subtree(tree);
        }
        count
    }

    #[cfg(feature = "expensive_verify")]
    fn count_subtree(node: &NodeRef<T, P, S>) -> usize {
        let node_ref = node.borrow();
        // Verify node has valid priority
        assert!(
            P::is_some(&node_ref.priority),
            "Found node with None priority in tree"
        );

        let mut count = 1;

        // Count children
        let mut child_opt = node_ref.child.clone();
        drop(node_ref);

        while let Some(child) = child_opt {
            count += Self::count_subtree(&child);
            child_opt = child.borrow().sibling.clone();
        }

        count
    }

    /// Cuts a node from its parent and reinserts it with a new priority
    fn cut_and_reinsert(&mut self, node: NodeRef<T, P, S>, new_priority: P) {
        // Get parent before we modify anything
        let parent_weak = node.borrow().parent.clone();
        let parent = parent_weak.upgrade().unwrap();

        // Remove node from parent's child list
        let mut found = false;
        {
            let parent_child = parent.borrow().child.clone();

            if let Some(ref first_child) = parent_child {
                if Rc::ptr_eq(first_child, &node) {
                    // Node is first child - update parent's child pointer
                    let next_sibling = node.borrow().sibling.clone();
                    parent.borrow_mut().child = next_sibling;
                    found = true;
                } else {
                    // Search through sibling list
                    let mut prev = Rc::clone(first_child);
                    loop {
                        let next = prev.borrow().sibling.clone();
                        match next {
                            Some(ref next_node) if Rc::ptr_eq(next_node, &node) => {
                                // Found it - remove from list
                                let skip_to = node.borrow().sibling.clone();
                                prev.borrow_mut().sibling = skip_to;
                                found = true;
                                break;
                            }
                            Some(next_node) => {
                                prev = next_node;
                            }
                            None => break,
                        }
                    }
                }
            }
        }

        if !found {
            // Node might already be detached, just update priority
            node.borrow_mut().priority = P::some(new_priority);
            node.borrow_mut().parent = Weak::new();
            self.find_and_update_min();
            return;
        }

        // Get parent's old rank before modifying
        let parent_old_rank = parent.borrow().rank;
        let parent_is_root = parent.borrow().parent.upgrade().is_none();

        // Update parent's rank (one less child)
        {
            parent.borrow_mut().rank = parent_old_rank.saturating_sub(1);
        }

        // If parent is a root, it needs to be repositioned in trees
        if parent_is_root && parent_old_rank < self.trees.len() {
            // Remove parent from old position
            self.trees[parent_old_rank] = None;
            // Reinsert parent at correct rank
            self.insert_tree(Rc::clone(&parent));
        }

        // Clear node's parent and sibling (keep children!)
        node.borrow_mut().parent = Weak::new();
        node.borrow_mut().sibling = None;

        // Update priority
        node.borrow_mut().priority = P::some(new_priority);

        // Reinsert the node (with its subtree intact) as a new tree
        self.insert_tree(node);

        // Update minimum
        self.find_and_update_min();
    }

    fn insert_tree(&mut self, mut tree: NodeRef<T, P, S>) {
        loop {
            let rank = tree.borrow().rank;

            while self.trees.len() <= rank {
                self.trees.push(None);
            }

            if self.trees[rank].is_some() {
                let existing = self.trees[rank].take().unwrap();
                tree = Self::link_trees(existing, tree);
            } else {
                self.trees[rank] = Some(tree);
                break;
            }
        }
    }

    fn link_trees(a: NodeRef<T, P, S>, b: NodeRef<T, P, S>) -> NodeRef<T, P, S> {
        let a_priority_greater = {
            let a_b = a.borrow();
            let b_b = b.borrow();
            // Debug assertions to catch nodes with invalid priorities
            debug_assert!(
                P::is_some(&a_b.priority),
                "link_trees: node 'a' has None priority"
            );
            debug_assert!(
                P::is_some(&b_b.priority),
                "link_trees: node 'b' has None priority"
            );
            P::get(&a_b.priority).unwrap() > P::get(&b_b.priority).unwrap()
        };

        if a_priority_greater {
            return Self::link_trees(b, a);
        }

        {
            let a_child = a.borrow().child.clone();
            let mut b_mut = b.borrow_mut();
            b_mut.parent = Rc::downgrade(&a);
            b_mut.sibling = a_child;
        }

        {
            let mut a_mut = a.borrow_mut();
            a_mut.child = Some(Rc::clone(&b));
            a_mut.rank += 1;

            let b_skew = b.borrow().skew;
            a_mut.skew = b_skew && a_mut.rank > 0;
        }

        a
    }

    fn find_and_update_min(&mut self) {
        self.min = None;
        for root_opt in self.trees.iter().flatten() {
            let should_update = match &self.min {
                Some(min_ref) => {
                    P::get(&root_opt.borrow().priority).unwrap()
                        < P::get(&min_ref.borrow().priority).unwrap()
                }
                None => true,
            };

            if should_update {
                self.min = Some(Rc::clone(root_opt));
            }
        }
    }
}

impl<T, P: OptLike<S> + Ord + Clone, S: StorageStrategy> Default for SkewBinomialHeapImpl<T, P, S> {
    fn default() -> Self {
        Self::new()
    }
}
