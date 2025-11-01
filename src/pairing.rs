//! Pairing Heap implementation
//!
//! A pairing heap is a type of heap-ordered tree with:
//! - O(1) amortized insert and merge
//! - O(log n) amortized delete_min
//! - o(log n) amortized decrease_key (in fact, better than log n)
//!
//! The pairing heap is simpler than Fibonacci heaps while still providing
//! excellent amortized performance for decrease_key operations.

use crate::traits::{Handle, Heap};
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

struct Node<T, P> {
    item: T,
    priority: P,
    child: Option<NonNull<Node<T, P>>>,
    sibling: Option<NonNull<Node<T, P>>>,
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

    fn insert(&mut self, priority: P, item: T) -> Self::Handle {
        let node = Box::into_raw(Box::new(Node {
            item,
            priority,
            child: None,
            sibling: None,
            prev: None,
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };

        if let Some(root_ptr) = self.root {
            unsafe {
                if (*node).priority < (*root_ptr.as_ptr()).priority {
                    // New node becomes root
                    (*node).child = Some(root_ptr);
                    (*root_ptr.as_ptr()).prev = Some(node_ptr);
                    self.root = Some(node_ptr);
                } else {
                    // Root stays, add node as child of root
                    let root_child = (*root_ptr.as_ptr()).child;
                    (*node).sibling = root_child;
                    (*node).prev = Some(root_ptr);
                    if let Some(child) = root_child {
                        (*child.as_ptr()).prev = Some(node_ptr);
                    }
                    (*root_ptr.as_ptr()).child = Some(node_ptr);
                }
            }
        } else {
            self.root = Some(node_ptr);
        }

        self.len += 1;
        PairingHandle {
            node: node_ptr.as_ptr() as *const (),
        }
    }

    fn peek(&self) -> Option<(&P, &T)> {
        self.find_min()
    }

    fn find_min(&self) -> Option<(&P, &T)> {
        self.root.map(|root_ptr| unsafe {
            let node = root_ptr.as_ptr();
            (&(*node).priority, &(*node).item)
        })
    }

    fn pop(&mut self) -> Option<(P, T)> {
        self.delete_min()
    }

    fn delete_min(&mut self) -> Option<(P, T)> {
        let root_ptr = self.root?;

        unsafe {
            let node = root_ptr.as_ptr();
            let (priority, item) = (
                ptr::read(&(*node).priority),
                ptr::read(&(*node).item),
            );

            // Pair up children and merge them
            let children = (*node).child;
            if let Some(first_child) = children {
                self.root = Some(self.merge_pairs(first_child));
            } else {
                self.root = None;
            }

            drop(Box::from_raw(node));
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

            // If node is root, we're done
            if self.root == Some(node_ptr) {
                return;
            }

            // Cut node from its parent and merge with root
            self.cut_node(node_ptr);
            
            // Merge with root if necessary
            if let Some(root_ptr) = self.root {
                if (*node).priority < (*root_ptr.as_ptr()).priority {
                    // New node becomes root
                    (*node).child = Some(root_ptr);
                    (*root_ptr.as_ptr()).prev = Some(node_ptr);
                    self.root = Some(node_ptr);
                } else {
                    // Root stays, add node as child of root
                    let root_child = (*root_ptr.as_ptr()).child;
                    (*node).sibling = root_child;
                    (*node).prev = Some(root_ptr);
                    if let Some(child) = root_child {
                        (*child.as_ptr()).prev = Some(node_ptr);
                    }
                    (*root_ptr.as_ptr()).child = Some(node_ptr);
                }
            } else {
                self.root = Some(node_ptr);
            }
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

        unsafe {
            let self_root = self.root.unwrap();
            let other_root = other.root.unwrap();

            if (*other_root.as_ptr()).priority < (*self_root.as_ptr()).priority {
                // Other becomes root
                let self_child = (*other_root.as_ptr()).child;
                (*self_root.as_ptr()).sibling = self_child;
                (*self_root.as_ptr()).prev = Some(other_root);
                if let Some(child) = self_child {
                    (*child.as_ptr()).prev = Some(self_root);
                }
                (*other_root.as_ptr()).child = Some(self_root);
                self.root = Some(other_root);
            } else {
                // Self stays root
                let self_child = (*self_root.as_ptr()).child;
                (*other_root.as_ptr()).sibling = self_child;
                (*other_root.as_ptr()).prev = Some(self_root);
                if let Some(child) = self_child {
                    (*child.as_ptr()).prev = Some(other_root);
                }
                (*self_root.as_ptr()).child = Some(other_root);
            }

            self.len += other.len;
            other.root = None;
            other.len = 0;
        }
    }
}

impl<T, P: Ord> PairingHeap<T, P> {
    /// Merges pairs of trees in a two-pass pairing operation
    unsafe fn merge_pairs(&self, first: NonNull<Node<T, P>>) -> NonNull<Node<T, P>> {
        if (*first.as_ptr()).sibling.is_none() {
            return first;
        }

        // First pass: pair adjacent trees and merge them
        let mut pairs = Vec::new();
        let mut current = Some(first);

        while let Some(node) = current {
            let sibling = (*node.as_ptr()).sibling;
            
            // Disconnect from sibling list
            (*node.as_ptr()).sibling = None;
            (*node.as_ptr()).prev = None;

            if let Some(sib) = sibling {
                let next = (*sib.as_ptr()).sibling;
                (*sib.as_ptr()).sibling = None;
                (*sib.as_ptr()).prev = None;

                // Merge the pair
                pairs.push(self.merge_nodes(node, sib));
                current = next;
            } else {
                pairs.push(node);
                current = None;
            }
        }

        // Second pass: merge pairs from right to left
        let mut result = pairs.pop().unwrap();
        while let Some(pair) = pairs.pop() {
            result = self.merge_nodes(pair, result);
        }

        result
    }

    /// Merges two nodes, returning the one with smaller priority
    unsafe fn merge_nodes(
        &self,
        a: NonNull<Node<T, P>>,
        b: NonNull<Node<T, P>>,
    ) -> NonNull<Node<T, P>> {
        if (*a.as_ptr()).priority < (*b.as_ptr()).priority {
            let a_child = (*a.as_ptr()).child;
            (*b.as_ptr()).sibling = a_child;
            (*b.as_ptr()).prev = Some(a);
            if let Some(child) = a_child {
                (*child.as_ptr()).prev = Some(b);
            }
            (*a.as_ptr()).child = Some(b);
            a
        } else {
            let b_child = (*b.as_ptr()).child;
            (*a.as_ptr()).sibling = b_child;
            (*a.as_ptr()).prev = Some(b);
            if let Some(child) = b_child {
                (*child.as_ptr()).prev = Some(a);
            }
            (*b.as_ptr()).child = Some(a);
            b
        }
    }

    /// Cuts a node from its parent
    unsafe fn cut_node(&mut self, node: NonNull<Node<T, P>>) {
        let prev_opt = match (*node.as_ptr()).prev {
            Some(p) => p,
            None => return,
        };
        let prev = prev_opt.as_ptr();

        // Check if prev is parent or sibling
        if (*prev).child == Some(node) {
            // Node is first child
            (*prev).child = (*node.as_ptr()).sibling;
            if let Some(sibling) = (*node.as_ptr()).sibling {
                (*sibling.as_ptr()).prev = Some(prev_opt);
            }
        } else {
            // Node is a sibling
            (*prev).sibling = (*node.as_ptr()).sibling;
            if let Some(sibling) = (*node.as_ptr()).sibling {
                (*sibling.as_ptr()).prev = Some(prev_opt);
            }
        }

        (*node.as_ptr()).sibling = None;
        (*node.as_ptr()).prev = None;
    }

    /// Recursively frees a node and all its descendants
    unsafe fn free_node(node: NonNull<Node<T, P>>) {
        let node_ptr = node.as_ptr();
        if let Some(child) = (*node_ptr).child {
            Self::free_node(child);
        }
        if let Some(sibling) = (*node_ptr).sibling {
            Self::free_node(sibling);
        }
        drop(Box::from_raw(node_ptr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = PairingHeap::new();
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
        let mut heap = PairingHeap::new();
        let h1 = heap.insert(10, "a");
        let _h2 = heap.insert(20, "b");
        let h3 = heap.insert(30, "c");

        assert_eq!(heap.find_min(), Some((&10, &"a")));

        heap.decrease_key(&h1, 5);
        assert_eq!(heap.find_min(), Some((&5, &"a")));
    }

    #[test]
    fn test_merge() {
        let mut heap1 = PairingHeap::new();
        heap1.insert(5, "a");
        heap1.insert(10, "b");

        let mut heap2 = PairingHeap::new();
        heap2.insert(3, "c");
        heap2.insert(7, "d");

        heap1.merge(heap2);
        assert_eq!(heap1.find_min(), Some((&3, &"c")));
    }
}

