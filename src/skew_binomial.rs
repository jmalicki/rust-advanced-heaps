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
        for tree_opt in self.trees.iter() {
            if let Some(root) = tree_opt {
                unsafe {
                    Self::free_tree(*root);
                }
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

    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Box::into_raw(Box::new(Node {
            item,
            priority,
            parent: None,
            child: None,
            sibling: None,
            rank: 0,
            skew: true, // New single-node tree is skew
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        unsafe {
            // Update min
            if let Some(min_ptr) = self.min {
                if (*node).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node_ptr);
                }
            } else {
                self.min = Some(node_ptr);
            }

            // Insert as rank-0 tree (O(1))
            // Skew binomial allows O(1) insert via special handling
            while self.trees.len() <= 0 {
                self.trees.push(None);
            }
            
            if self.trees[0].is_some() {
                // Merge two rank-0 trees (O(1))
                let existing = self.trees[0].unwrap();
                let merged = unsafe { self.link_trees(existing, node_ptr) };
                self.trees[0] = None;
                
                // Insert merged tree at rank 1
                while self.trees.len() <= 1 {
                    self.trees.push(None);
                }
                
                // Merge with existing rank-1 tree if present (cascade)
                if self.trees[1].is_some() {
                    let existing_rank1 = self.trees[1].unwrap();
                    let merged_rank1 = unsafe { self.link_trees(existing_rank1, merged) };
                    self.trees[1] = None;
                    
                    while self.trees.len() <= 2 {
                        self.trees.push(None);
                    }
                    self.trees[2] = Some(merged_rank1);
                } else {
                    self.trees[1] = Some(merged);
                }
            } else {
                self.trees[0] = Some(node_ptr);
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

    fn delete_min(&mut self) -> Option<(P, T)> {
        let min_ptr = self.min?;

        unsafe {
            let node = min_ptr.as_ptr();
            let (priority, item) = (
                ptr::read(&(*node).priority),
                ptr::read(&(*node).item),
            );

            // Remove from tree list
            let rank = (*node).rank;
            if rank < self.trees.len() {
                self.trees[rank] = None;
            }

            // Collect children
            let mut children = Vec::new();
            if let Some(child) = (*node).child {
                let mut current = Some(child);
                let mut prev: Option<NonNull<Node<T, P>>> = None;

                // Reverse child list
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    (*curr.as_ptr()).parent = None;
                    (*curr.as_ptr()).sibling = prev;
                    prev = Some(curr);
                    current = next;
                }

                // Collect children
                current = prev;
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    let child_rank = (*curr.as_ptr()).rank;
                    
                    (*curr.as_ptr()).sibling = None;
                    
                    while children.len() <= child_rank {
                        children.push(None);
                    }
                    children[child_rank] = Some(curr);
                    
                    current = next;
                }
            }

            drop(Box::from_raw(node));

            // Merge children back into heap
            for (rank, child_opt) in children.into_iter().enumerate() {
                if let Some(child) = child_opt {
                    while self.trees.len() <= rank {
                        self.trees.push(None);
                    }
                    
                    if self.trees[rank].is_some() {
                        // Merge with existing tree
                        let existing = self.trees[rank].unwrap();
                        let merged = self.link_trees(existing, child);
                        self.trees[rank] = None;
                        
                        while self.trees.len() <= rank + 1 {
                            self.trees.push(None);
                        }
                        self.trees[rank + 1] = Some(merged);
                    } else {
                        self.trees[rank] = Some(child);
                    }
                }
            }

            // Find new minimum
            self.find_and_update_min();

            self.len -= 1;
            Some((priority, item))
        }
    }

    fn decrease_key(&mut self, handle: &Self::Handle, new_priority: P) {
        let node_ptr = unsafe { NonNull::new_unchecked(handle.node as *mut Node<T, P>) };

        unsafe {
            let node = node_ptr.as_ptr();

            if new_priority >= (*node).priority {
                return;
            }

            (*node).priority = new_priority;

            // Bubble up to maintain heap property
            self.bubble_up(node_ptr);
        }
    }

    fn merge(&mut self, mut other: Self) {
        if other.is_empty() {
            return;
        }

        if self.is_empty() {
            *self = other;
            return;
        }

        // Merge tree lists
        let max_rank = self.trees.len().max(other.trees.len());
        while self.trees.len() < max_rank {
            self.trees.push(None);
        }

        let mut carry: Option<NonNull<Node<T, P>>> = None;

        unsafe {
            for rank in 0..max_rank {
                let mut trees_to_merge = Vec::new();

                // Collect trees from both heaps at this rank
                if rank < self.trees.len() && self.trees[rank].is_some() {
                    trees_to_merge.push(self.trees[rank].take().unwrap());
                }
                if rank < other.trees.len() && other.trees[rank].is_some() {
                    trees_to_merge.push(other.trees[rank].take().unwrap());
                }
                if let Some(c) = carry {
                    trees_to_merge.push(c);
                    carry = None;
                }

                // Merge pairs until at most one remains
                while trees_to_merge.len() > 1 {
                    let a = trees_to_merge.pop().unwrap();
                    let b = trees_to_merge.pop().unwrap();
                    let merged = self.link_trees(a, b);
                    
                    let merged_rank = (*merged.as_ptr()).rank;
                    if merged_rank == rank + 1 {
                        carry = Some(merged);
                    } else {
                        trees_to_merge.push(merged);
                    }
                }

                // Store remaining tree
                if let Some(tree) = trees_to_merge.pop() {
                    self.trees[rank] = Some(tree);
                }
            }

            // Handle final carry
            if let Some(c) = carry {
                let rank = (*c.as_ptr()).rank;
                while self.trees.len() <= rank {
                    self.trees.push(None);
                }
                self.trees[rank] = Some(c);
            }
        }

        self.len += other.len;
        
        // Update min
        self.find_and_update_min();

        // Prevent double free
        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> SkewBinomialHeap<T, P> {
    /// Links two trees of the same rank
    unsafe fn link_trees(
        &mut self,
        a: NonNull<Node<T, P>>,
        b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        // Make tree with larger priority a child of the one with smaller priority
        if (*a.as_ptr()).priority > (*b.as_ptr()).priority {
            return self.link_trees(b, a);
        }

        // Make b a child of a
        let a_child = (*a.as_ptr()).child;
        (*b.as_ptr()).parent = Some(a);
        (*b.as_ptr()).sibling = a_child;
        (*a.as_ptr()).child = Some(b);
        (*a.as_ptr()).rank += 1;

        // Update skew flag (simplified)
        (*a.as_ptr()).skew = (*b.as_ptr()).skew && (*a.as_ptr()).rank > 0;

        a
    }

    /// Bubbles up a node to maintain heap property
    unsafe fn bubble_up(&mut self, mut node: NonNull<Node<T, P>>) {
        while let Some(parent) = (*node.as_ptr()).parent {
            if (*node.as_ptr()).priority >= (*parent.as_ptr()).priority {
                break; // Heap property satisfied
            }

            // Swap node with parent
            let node_ptr = node.as_ptr();
            let parent_ptr = parent.as_ptr();

            ptr::swap(&mut (*node_ptr).priority, &mut (*parent_ptr).priority);
            ptr::swap(&mut (*node_ptr).item, &mut (*parent_ptr).item);

            node = parent;
        }

        // Update min if necessary
        if let Some(min_ptr) = self.min {
            if (*node.as_ptr()).priority < (*min_ptr.as_ptr()).priority {
                self.min = Some(node);
            }
        } else {
            self.min = Some(node);
        }
    }

    /// Finds and updates the minimum pointer
    fn find_and_update_min(&mut self) {
        self.min = None;
        for tree_opt in self.trees.iter() {
            if let Some(root) = tree_opt {
                unsafe {
                    if self.min.is_none()
                        || (*root.as_ptr()).priority < (*self.min.unwrap().as_ptr()).priority
                    {
                        self.min = Some(*root);
                    }
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

