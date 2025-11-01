//! Binomial Heap implementation
//!
//! A binomial heap is a collection of binomial trees with:
//! - O(log n) insert and delete_min
//! - O(log n) decrease_key
//! - O(log n) merge (O(1) amortized if merging many heaps)
//!
//! Binomial heaps are simpler than Fibonacci heaps but have worse
//! amortized bounds for decrease_key.

use crate::traits::{Handle, Heap};
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

struct Node<T, P> {
    item: T,
    priority: P,
    parent: Option<NonNull<Node<T, P>>>,
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
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
    trees: Vec<Option<NonNull<Node<T, P>>>>, // Array indexed by degree
    min: Option<NonNull<Node<T, P>>>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P: Ord> Drop for BinomialHeap<T, P> {
    fn drop(&mut self) {
        // Free all trees
        for tree_opt in self.trees.iter() {
            if let Some(root) = tree_opt {
                unsafe {
                    Self::free_tree(*root);
                }
            }
        }
    }
}

impl<T, P: Ord> Heap<T, P> for BinomialHeap<T, P> {
    type Handle = BinomialHandle;

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
            degree: 0,
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        // Update min
        if let Some(min_ptr) = self.min {
            unsafe {
                if (*node).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node_ptr);
                }
            }
        } else {
            self.min = Some(node_ptr);
        }

        // Merge this single-node tree into the heap
        let mut carry = Some(node_ptr);
        let mut degree = 0;

        while carry.is_some() {
            if degree >= self.trees.len() {
                self.trees.push(None);
            }

            if self.trees[degree].is_none() {
                self.trees[degree] = carry;
                carry = None;
            } else {
                unsafe {
                    let existing = self.trees[degree].unwrap();
                    let new_tree = carry.unwrap();
                    carry = Some(self.link_trees(existing, new_tree));
                    self.trees[degree] = None;
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

    fn delete_min(&mut self) -> Option<(P, T)> {
        let min_ptr = self.min?;

        unsafe {
            let node = min_ptr.as_ptr();
            let (priority, item) = (
                ptr::read(&(*node).priority),
                ptr::read(&(*node).item),
            );

            // Remove min from tree list
            let degree = (*node).degree;
            if degree < self.trees.len() {
                self.trees[degree] = None;
            }

            // Add children as new trees
            let mut child_heap = BinomialHeap::new();
            if let Some(child) = (*node).child {
                let mut current = Some(child);
                let mut prev: Option<NonNull<Node<T, P>>> = None;

                // Reverse the child list
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    (*curr.as_ptr()).parent = None;
                    (*curr.as_ptr()).sibling = prev;
                    prev = Some(curr);
                    current = next;
                }

                // Add each child tree to the child heap
                current = prev;
                while let Some(curr) = current {
                    let next = (*curr.as_ptr()).sibling;
                    let child_degree = (*curr.as_ptr()).degree;
                    
                    // Reset sibling to break the link temporarily
                    (*curr.as_ptr()).sibling = None;
                    
                    while child_heap.trees.len() <= child_degree {
                        child_heap.trees.push(None);
                    }
                    child_heap.trees[child_degree] = Some(curr);
                    
                    current = next;
                }
            }

            drop(Box::from_raw(node));

            // Merge child heap back into main heap
            self.merge_trees(&mut child_heap);

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

            // Verify that new priority is actually less
            if new_priority >= (*node).priority {
                return;
            }

            (*node).priority = new_priority;

            // Bubble up to maintain heap property
            self.bubble_up(node_ptr);
        }
    }

    fn merge(&mut self, mut other: Self) {
        // Merge all trees
        self.merge_trees(&mut other);

        // Update min
        self.find_and_update_min();

        self.len += other.len;
        other.min = None;
        other.len = 0;
    }
}

impl<T, P: Ord> BinomialHeap<T, P> {
    /// Links two binomial trees of the same degree
    unsafe fn link_trees(
        &self,
        a: NonNull<Node<T, P>>,
        b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        // Make the tree with larger priority a child of the one with smaller priority
        if (*a.as_ptr()).priority > (*b.as_ptr()).priority {
            return self.link_trees(b, a);
        }

        // Make b a child of a
        let a_child = (*a.as_ptr()).child;
        (*b.as_ptr()).parent = Some(a);
        (*b.as_ptr()).sibling = a_child;
        (*a.as_ptr()).child = Some(b);
        (*a.as_ptr()).degree += 1;

        a
    }

    /// Merges trees from another heap into this one
    fn merge_trees(&mut self, other: &mut Self) {
        let max_degree = self.trees.len().max(other.trees.len());
        while self.trees.len() < max_degree {
            self.trees.push(None);
        }

        let mut carry: Option<NonNull<Node<T, P>>> = None;

        for degree in 0..max_degree {
            let mut trees = Vec::new();

            // Collect trees from both heaps at this degree
            if degree < self.trees.len() {
                if let Some(tree) = self.trees[degree] {
                    trees.push(tree);
                    self.trees[degree] = None;
                }
            }

            if degree < other.trees.len() {
                if let Some(tree) = other.trees[degree] {
                    trees.push(tree);
                    other.trees[degree] = None;
                }
            }

            // Add carry if present
            if let Some(c) = carry {
                trees.push(c);
                carry = None;
            }

            // Process trees: link pairs until at most one remains
            while trees.len() > 1 {
                unsafe {
                    let a = trees.pop().unwrap();
                    let b = trees.pop().unwrap();
                    let linked = self.link_trees(a, b);
                    
                    if (*linked.as_ptr()).degree == degree + 1 {
                        carry = Some(linked);
                    } else {
                        trees.push(linked);
                    }
                }
            }

            if let Some(tree) = trees.pop() {
                unsafe {
                    if (*tree.as_ptr()).degree == degree {
                        self.trees[degree] = Some(tree);
                    } else {
                        carry = Some(tree);
                    }
                }
            }
        }

        // Handle final carry
        if let Some(c) = carry {
            let degree = unsafe { (*c.as_ptr()).degree };
            while self.trees.len() <= degree {
                self.trees.push(None);
            }
            self.trees[degree] = Some(c);
        }
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

            // Swap priorities and items
            ptr::swap(&mut (*node_ptr).priority, &mut (*parent_ptr).priority);
            ptr::swap(&mut (*node_ptr).item, &mut (*parent_ptr).item);

            node = parent;
        }

        // Update min if necessary
        unsafe {
            if let Some(min_ptr) = self.min {
                if (*node.as_ptr()).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node);
                }
            } else {
                self.min = Some(node);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = BinomialHeap::new();
        assert!(heap.is_empty());

        let _h1 = heap.insert(5, "a");
        let _h2 = heap.insert(3, "b");
        let _h3 = heap.insert(7, "c");

        assert_eq!(heap.find_min(), Some((&3, &"b")));

        let min = heap.delete_min();
        assert_eq!(min, Some((3, "b")));
        assert_eq!(heap.find_min(), Some((&5, &"a")));
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = BinomialHeap::new();
        let h1 = heap.insert(10, "a");
        let _h2 = heap.insert(20, "b");
        let h3 = heap.insert(30, "c");

        assert_eq!(heap.find_min(), Some((&10, &"a")));

        heap.decrease_key(&h1, 5);
        assert_eq!(heap.find_min(), Some((&5, &"a")));

        heap.decrease_key(&h3, 1);
        assert_eq!(heap.find_min(), Some((&1, &"c")));
    }

    #[test]
    fn test_merge() {
        let mut heap1 = BinomialHeap::new();
        heap1.insert(5, "a");
        heap1.insert(10, "b");

        let mut heap2 = BinomialHeap::new();
        heap2.insert(3, "c");
        heap2.insert(7, "d");

        heap1.merge(heap2);
        assert_eq!(heap1.find_min(), Some((&3, &"c")));
    }
}

