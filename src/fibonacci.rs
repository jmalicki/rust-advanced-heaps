//! Fibonacci Heap implementation
//!
//! A Fibonacci heap is a data structure for priority queue operations with:
//! - O(1) amortized insert, decrease_key, and merge
//! - O(log n) amortized delete_min
//!
//! The structure consists of a collection of heap-ordered trees. Roots are linked
//! in a circular doubly linked list. The heap maintains the minimum node pointer.

use crate::traits::{Handle, Heap};
use std::ptr::{self, NonNull};

/// Handle to an element in a Fibonacci heap
///
/// Note: This handle is tied to a specific heap instance. Using it with a different
/// heap or after the heap is dropped is undefined behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FibonacciHandle {
    node: *const (), // Type-erased pointer to Node<T, P>
}

impl Handle for FibonacciHandle {}

struct Node<T, P> {
    item: T,
    priority: P,
    parent: Option<NonNull<Node<T, P>>>,
    child: Option<NonNull<Node<T, P>>>,
    left: NonNull<Node<T, P>>,
    right: NonNull<Node<T, P>>,
    degree: usize,
    marked: bool,
}

/// Fibonacci Heap
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
    min: Option<NonNull<Node<T, P>>>,
    len: usize,
    // Phantom data to ensure proper drop semantics
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P: Ord> Drop for FibonacciHeap<T, P> {
    fn drop(&mut self) {
        // Clean up all nodes in the heap
        while self.delete_min().is_some() {}
    }
}

impl<T, P: Ord> Heap<T, P> for FibonacciHeap<T, P> {
    type Handle = FibonacciHandle;

    fn new() -> Self {
        Self {
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
            degree: 0,
            marked: false,
            left: NonNull::dangling(), // Will be set immediately
            right: NonNull::dangling(), // Will be set immediately
        }));

        let node_ptr = unsafe { NonNull::new_unchecked(node) };
        
        // Initialize circular list - node points to itself
        unsafe {
            (*node).left = node_ptr;
            (*node).right = node_ptr;
        }

        if let Some(min_ptr) = self.min {
            // Add to root list
            unsafe {
                let min_left = (*min_ptr.as_ptr()).left;
                (*node).right = min_ptr;
                (*node).left = min_left;
                (*min_left.as_ptr()).right = node_ptr;
                (*min_ptr.as_ptr()).left = node_ptr;

                // Update min if necessary
                if (*node).priority < (*min_ptr.as_ptr()).priority {
                    self.min = Some(node_ptr);
                }
            }
        } else {
            // First node
            self.min = Some(node_ptr);
        }

        self.len += 1;
        FibonacciHandle {
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

            // Add children to root list
            if let Some(child) = (*node).child {
                let mut current = child;
                loop {
                    let next = (*current.as_ptr()).right;
                    (*current.as_ptr()).parent = None;
                    (*current.as_ptr()).marked = false;

                    // Add to root list
                    let min_left = (*min_ptr.as_ptr()).left;
                    (*current.as_ptr()).right = min_ptr;
                    (*current.as_ptr()).left = min_left;
                    (*min_left.as_ptr()).right = current;
                    (*min_ptr.as_ptr()).left = current;

                    if next == child {
                        break;
                    }
                    current = next;
                }
            }

            // Remove min from root list
            let left = (*node).left;
            let right = (*node).right;
            
            if left == min_ptr {
                // Only one root, and it's being deleted
                self.min = None;
            } else {
                (*left.as_ptr()).right = right;
                (*right.as_ptr()).left = left;
                
                // Consolate the heap
                self.consolidate(right);
            }

            // Deallocate the node
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
                return; // Or panic/error
            }

            (*node).priority = new_priority;

            // If node is root or heap property is satisfied, we're done
            if (*node).parent.is_none() {
                // Update min if necessary
                if let Some(min_ptr) = self.min {
                    if (*node).priority < (*min_ptr.as_ptr()).priority {
                        self.min = Some(node_ptr);
                    }
                }
                return;
            }

            // Check if heap property is violated
            if let Some(parent_ptr) = (*node).parent {
                if (*node).priority >= (*parent_ptr.as_ptr()).priority {
                    return; // Heap property still satisfied
                }
            }

            // Cut node from parent and add to root list
            self.cut(node_ptr);
            self.cascading_cut((*node).parent);
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
            let self_min = self.min.unwrap();
            let other_min = other.min.unwrap();
            
            let self_left = (*self_min.as_ptr()).left;
            let other_left = (*other_min.as_ptr()).left;

            // Link the two circular lists
            (*self_left.as_ptr()).right = other_min;
            (*other_min.as_ptr()).left = self_left;
            (*other_left.as_ptr()).right = self_min;
            (*self_min.as_ptr()).left = other_left;

            // Update min
            if (*other_min.as_ptr()).priority < (*self_min.as_ptr()).priority {
                self.min = Some(other_min);
            }

            self.len += other.len;
            
            // Prevent double free
            other.min = None;
            other.len = 0;
        }
    }
}

impl<T, P: Ord> FibonacciHeap<T, P> {
    /// Consolidates the heap by linking trees of the same degree
    fn consolidate(&mut self, start: NonNull<Node<T, P>>) {
        // Array to track trees by degree (log n max)
        let max_degree = (self.len as f64).log2() as usize + 2;
        let mut degree_table: Vec<Option<NonNull<Node<T, P>>>> = vec![None; max_degree + 1];

        unsafe {
            let mut current = start;
            let stop = start;
            let mut nodes_to_process = vec![];

            // Collect all roots
            loop {
                nodes_to_process.push(current);
                current = (*current.as_ptr()).right;
                if current == stop {
                    break;
                }
            }

            // Process each root
            for root in nodes_to_process {
                let mut x = root;
                let mut d = (*x.as_ptr()).degree;

                // Link with existing trees of the same degree
                while degree_table[d].is_some() {
                    let mut y = degree_table[d].unwrap();
                    
                    // Ensure x has smaller priority
                    if (*y.as_ptr()).priority < (*x.as_ptr()).priority {
                        std::mem::swap(&mut x, &mut y);
                    }

                    // Link y as child of x
                    self.link(y, x);

                    degree_table[d] = None;
                    d += 1;
                }

                degree_table[d] = Some(x);
            }

            // Rebuild root list and find new min
            self.min = None;
            for root_opt in degree_table.into_iter().flatten() {
                if self.min.is_none() {
                    // First root
                    self.min = Some(root_opt);
                    (*root_opt.as_ptr()).left = root_opt;
                    (*root_opt.as_ptr()).right = root_opt;
                } else {
                    // Add to root list
                    let min_ptr = self.min.unwrap();
                    let min_left = (*min_ptr.as_ptr()).left;
                    
                    (*root_opt.as_ptr()).right = min_ptr;
                    (*root_opt.as_ptr()).left = min_left;
                    (*min_left.as_ptr()).right = root_opt;
                    (*min_ptr.as_ptr()).left = root_opt;

                    // Update min if necessary
                    if (*root_opt.as_ptr()).priority < (*min_ptr.as_ptr()).priority {
                        self.min = Some(root_opt);
                    }
                }
            }
        }
    }

    /// Links node y as a child of node x
    unsafe fn link(&mut self, y: NonNull<Node<T, P>>, x: NonNull<Node<T, P>>) {
        // Remove y from root list
        let y_left = (*y.as_ptr()).left;
        let y_right = (*y.as_ptr()).right;
        (*y_left.as_ptr()).right = y_right;
        (*y_right.as_ptr()).left = y_left;

        // Make y a child of x
        (*y.as_ptr()).parent = Some(x);
        (*y.as_ptr()).marked = false;

        if let Some(x_child) = (*x.as_ptr()).child {
            // Add y to x's child list
            let x_child_left = (*x_child.as_ptr()).left;
            (*y.as_ptr()).right = x_child;
            (*y.as_ptr()).left = x_child_left;
            (*x_child_left.as_ptr()).right = y;
            (*x_child.as_ptr()).left = y;
        } else {
            // y is x's first child
            (*x.as_ptr()).child = Some(y);
            (*y.as_ptr()).left = y;
            (*y.as_ptr()).right = y;
        }

        (*x.as_ptr()).degree += 1;
    }

    /// Cuts node from its parent and adds it to the root list
    unsafe fn cut(&mut self, node: NonNull<Node<T, P>>) {
        let parent_ptr = match (*node.as_ptr()).parent {
            Some(p) => p,
            None => return,
        };
        
        // Remove from parent's child list
        let node_left = (*node.as_ptr()).left;
        let node_right = (*node.as_ptr()).right;

        if (*parent_ptr.as_ptr()).child == Some(node) {
            if node_left == node {
                // Only child
                (*parent_ptr.as_ptr()).child = None;
            } else {
                (*parent_ptr.as_ptr()).child = Some(node_left);
            }
        }

        (*node_left.as_ptr()).right = node_right;
        (*node_right.as_ptr()).left = node_left;

        (*parent_ptr.as_ptr()).degree -= 1;

        // Add to root list
        if let Some(min_ptr) = self.min {
            let min_left = (*min_ptr.as_ptr()).left;
            (*node.as_ptr()).right = min_ptr;
            (*node.as_ptr()).left = min_left;
            (*min_left.as_ptr()).right = node;
            (*min_ptr.as_ptr()).left = node;
        } else {
            self.min = Some(node);
            (*node.as_ptr()).left = node;
            (*node.as_ptr()).right = node;
        }

        (*node.as_ptr()).parent = None;
        (*node.as_ptr()).marked = false;
    }

    /// Performs cascading cut on parent if it's marked
    unsafe fn cascading_cut(&mut self, parent_opt: Option<NonNull<Node<T, P>>>) {
        if let Some(parent_ptr) = parent_opt {
            let parent = parent_ptr.as_ptr();
            
            if (*parent).parent.is_some() {
                // Parent is not a root
                if !(*parent).marked {
                    (*parent).marked = true;
                } else {
                    self.cut(parent_ptr);
                    self.cascading_cut((*parent).parent);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut heap = FibonacciHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);

        let h1 = heap.insert(5, "a");
        let h2 = heap.insert(3, "b");
        let h3 = heap.insert(7, "c");

        assert_eq!(heap.len(), 3);
        assert_eq!(heap.find_min(), Some((&3, &"b")));

        let min = heap.delete_min();
        assert_eq!(min, Some((3, "b")));
        assert_eq!(heap.find_min(), Some((&5, &"a")));
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = FibonacciHeap::new();
        let h1 = heap.insert(10, "a");
        let h2 = heap.insert(20, "b");
        let h3 = heap.insert(30, "c");

        assert_eq!(heap.find_min(), Some((&10, &"a")));

        heap.decrease_key(&h2, 5);
        assert_eq!(heap.find_min(), Some((&5, &"b")));

        heap.decrease_key(&h3, 1);
        assert_eq!(heap.find_min(), Some((&1, &"c")));
    }

    #[test]
    fn test_merge() {
        let mut heap1 = FibonacciHeap::new();
        heap1.insert(5, "a");
        heap1.insert(10, "b");

        let mut heap2 = FibonacciHeap::new();
        heap2.insert(3, "c");
        heap2.insert(7, "d");

        heap1.merge(heap2);
        assert_eq!(heap1.find_min(), Some((&3, &"c")));
        assert_eq!(heap1.len(), 4);
    }
}

