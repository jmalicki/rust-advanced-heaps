//! Brodal-Okasaki Heap Implementation
//!
//! A purely functional priority queue with optimal worst-case time bounds:
//! - O(1) worst-case: insert, find_min, meld (merge)
//! - O(log n) worst-case: delete_min
//!
//! # Overview
//!
//! This implementation follows Brodal and Okasaki's 1996 paper "Optimal Purely
//! Functional Priority Queues". The key insight is to combine three techniques:
//!
//! 1. **Skew Binomial Trees**: A variant of binomial trees that allows O(1) insertion
//!    by using a "skew link" operation that avoids cascading carries.
//!
//! 2. **Global Root**: A distinguished root holds the minimum element, giving O(1) findMin.
//!
//! 3. **Bootstrapping**: Heaps can contain other heaps as elements, enabling O(1) meld.
//!
//! # Differences from the Imperative Brodal Heap
//!
//! Unlike the imperative Brodal heap (in `brodal.rs`), this implementation:
//! - Is purely functional (persistent) - operations return new heaps
//! - Does NOT support decrease_key (typical for functional priority queues)
//! - Uses immutable data structures with `Rc` for sharing
//! - Achieves true worst-case bounds, not amortized
//!
//! # References
//!
//! - Brodal, G.S. and Okasaki, C. (1996). "Optimal Purely Functional Priority Queues".
//!   Journal of Functional Programming 6(6):839-858.
//! - Okasaki, C. (1998). "Purely Functional Data Structures". Cambridge University Press.

use std::cmp::Ordering;
use std::rc::Rc;

/// A skew binomial tree node
///
/// Each node has:
/// - `rank`: determines the tree's structure
/// - `root`: the element at this node (minimum of subtree)
/// - `children`: list of child trees in decreasing order of rank
#[derive(Clone)]
struct SkewBinomialTree<T> {
    rank: usize,
    root: T,
    children: Rc<Vec<SkewBinomialTree<T>>>,
}

impl<T: Ord + Clone> SkewBinomialTree<T> {
    /// Create a new singleton tree (rank 0, no children)
    fn singleton(x: T) -> Self {
        SkewBinomialTree {
            rank: 0,
            root: x,
            children: Rc::new(Vec::new()),
        }
    }

    /// Simple link: combine two trees of equal rank
    /// The tree with larger root becomes a child of the smaller
    /// Children are maintained in decreasing order of rank
    fn link(self, other: Self) -> Self {
        debug_assert_eq!(self.rank, other.rank);
        if self.root <= other.root {
            // self wins, other becomes the first child (highest rank)
            let mut children = vec![other];
            children.extend((*self.children).iter().cloned());
            SkewBinomialTree {
                rank: self.rank + 1,
                root: self.root,
                children: Rc::new(children),
            }
        } else {
            // other wins, self becomes the first child (highest rank)
            let mut children = vec![self];
            children.extend((*other.children).iter().cloned());
            SkewBinomialTree {
                rank: other.rank + 1,
                root: other.root,
                children: Rc::new(children),
            }
        }
    }

    /// Skew link: combine two trees of equal rank with a new element
    /// This is the key operation that enables O(1) insertion
    ///
    /// Per Okasaki: We link t1 and t2, then add a rank-0 singleton.
    /// If x is smaller, x becomes root and the old root becomes the singleton.
    /// If x is larger, the old root stays and x becomes the singleton.
    fn skew_link(x: T, t1: Self, t2: Self) -> Self {
        debug_assert_eq!(t1.rank, t2.rank);
        let linked = t1.link(t2);
        if x <= linked.root {
            // x becomes new root, linked.root demoted to rank-0 child
            let mut children = (*linked.children).clone();
            children.insert(0, SkewBinomialTree::singleton(linked.root.clone()));
            SkewBinomialTree {
                rank: linked.rank,
                root: x,
                children: Rc::new(children),
            }
        } else {
            // Original root stays, x added as rank-0 child
            let mut children = (*linked.children).clone();
            children.insert(0, SkewBinomialTree::singleton(x));
            SkewBinomialTree {
                rank: linked.rank,
                root: linked.root,
                children: Rc::new(children),
            }
        }
    }
}

/// A skew binomial heap (primitive queue)
///
/// This is the underlying structure before bootstrapping.
/// Maintains a forest of skew binomial trees.
#[derive(Clone)]
struct SkewBinomialHeap<T> {
    /// Trees in increasing order of rank
    /// Invariant: at most one tree of any rank
    trees: Rc<Vec<SkewBinomialTree<T>>>,
}

impl<T: Ord + Clone> SkewBinomialHeap<T> {
    fn empty() -> Self {
        SkewBinomialHeap {
            trees: Rc::new(Vec::new()),
        }
    }

    fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Insert a tree into the heap, maintaining rank order
    fn insert_tree(
        tree: SkewBinomialTree<T>,
        trees: &[SkewBinomialTree<T>],
    ) -> Vec<SkewBinomialTree<T>> {
        if trees.is_empty() {
            vec![tree]
        } else if tree.rank < trees[0].rank {
            let mut result = vec![tree];
            result.extend(trees.iter().cloned());
            result
        } else {
            // tree.rank == trees[0].rank (can't be greater due to invariant)
            let linked = tree.link(trees[0].clone());
            Self::insert_tree(linked, &trees[1..])
        }
    }

    /// Merge two tree lists
    fn merge_trees(
        ts1: &[SkewBinomialTree<T>],
        ts2: &[SkewBinomialTree<T>],
    ) -> Vec<SkewBinomialTree<T>> {
        if ts1.is_empty() {
            ts2.to_vec()
        } else if ts2.is_empty() {
            ts1.to_vec()
        } else {
            match ts1[0].rank.cmp(&ts2[0].rank) {
                Ordering::Less => {
                    let mut result = vec![ts1[0].clone()];
                    result.extend(Self::merge_trees(&ts1[1..], ts2));
                    result
                }
                Ordering::Greater => {
                    let mut result = vec![ts2[0].clone()];
                    result.extend(Self::merge_trees(ts1, &ts2[1..]));
                    result
                }
                Ordering::Equal => {
                    let linked = ts1[0].clone().link(ts2[0].clone());
                    Self::insert_tree(linked, &Self::merge_trees(&ts1[1..], &ts2[1..]))
                }
            }
        }
    }

    /// Normalize: ensure no two trees have the same rank
    fn normalize(trees: &[SkewBinomialTree<T>]) -> Vec<SkewBinomialTree<T>> {
        if trees.is_empty() {
            Vec::new()
        } else {
            Self::insert_tree(trees[0].clone(), &Self::normalize(&trees[1..]))
        }
    }

    /// Insert an element - O(1) worst-case
    fn insert(&self, x: T) -> Self {
        let trees = &*self.trees;
        let new_trees = if trees.len() >= 2 && trees[0].rank == trees[1].rank {
            // Skew link: combine first two trees with new element
            let t1 = trees[0].clone();
            let t2 = trees[1].clone();
            let linked = SkewBinomialTree::skew_link(x, t1, t2);
            let mut result = vec![linked];
            result.extend(trees[2..].iter().cloned());
            result
        } else {
            // Just prepend a singleton
            let mut result = vec![SkewBinomialTree::singleton(x)];
            result.extend(trees.iter().cloned());
            result
        };

        SkewBinomialHeap {
            trees: Rc::new(new_trees),
        }
    }

    /// Merge two heaps - O(log n) worst-case
    fn merge(&self, other: &Self) -> Self {
        let merged = Self::merge_trees(
            &Self::normalize(&self.trees),
            &Self::normalize(&other.trees),
        );
        SkewBinomialHeap {
            trees: Rc::new(merged),
        }
    }

    /// Find the minimum element - O(log n) worst-case for primitive heap
    #[allow(dead_code)]
    fn find_min(&self) -> Option<&T> {
        self.trees.iter().map(|t| &t.root).min()
    }

    /// Remove the minimum tree and return it along with the remaining heap
    fn remove_min_tree(&self) -> Option<(SkewBinomialTree<T>, Self)> {
        if self.trees.is_empty() {
            return None;
        }

        let mut min_idx = 0;
        for (i, tree) in self.trees.iter().enumerate() {
            if tree.root < self.trees[min_idx].root {
                min_idx = i;
            }
        }

        let min_tree = self.trees[min_idx].clone();
        let remaining: Vec<_> = self
            .trees
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != min_idx)
            .map(|(_, t)| t.clone())
            .collect();

        Some((
            min_tree,
            SkewBinomialHeap {
                trees: Rc::new(remaining),
            },
        ))
    }

    /// Delete the minimum element - O(log n) worst-case
    fn delete_min(&self) -> Option<(T, Self)> {
        let (min_tree, rest) = self.remove_min_tree()?;

        // Children are in decreasing order of rank, reverse them
        let mut children_reversed: Vec<_> = (*min_tree.children).to_vec();
        children_reversed.reverse();

        let children_heap = SkewBinomialHeap {
            trees: Rc::new(Self::normalize(&children_reversed)),
        };

        let new_heap = rest.merge(&children_heap);
        Some((min_tree.root, new_heap))
    }
}

/// Bootstrapped Brodal-Okasaki Heap
///
/// This is the main data structure that achieves O(1) meld.
/// It wraps a skew binomial heap with a bootstrapping layer.
#[derive(Clone)]
pub struct BrodalOkasakiHeap<T: Ord + Clone> {
    inner: Option<BootstrappedHeap<T>>,
    len: usize,
}

/// Internal bootstrapped structure
#[derive(Clone)]
struct BootstrappedHeap<T: Ord + Clone> {
    /// The global minimum element
    min: T,
    /// A primitive heap of bootstrapped heaps
    /// This recursive structure enables O(1) meld
    prim_heap: SkewBinomialHeap<Rc<BootstrappedHeap<T>>>,
}

impl<T: Ord + Clone> PartialEq for BootstrappedHeap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min
    }
}

impl<T: Ord + Clone> Eq for BootstrappedHeap<T> {}

impl<T: Ord + Clone> PartialOrd for BootstrappedHeap<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord + Clone> Ord for BootstrappedHeap<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.min.cmp(&other.min)
    }
}

impl<T: Ord + Clone> BrodalOkasakiHeap<T> {
    /// Create a new empty heap
    pub fn new() -> Self {
        BrodalOkasakiHeap {
            inner: None,
            len: 0,
        }
    }

    /// Check if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_none()
    }

    /// Return the number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Insert an element - O(1) worst-case
    ///
    /// Per Brodal-Okasaki: insert(x, H(y, q)) =
    ///   if x <= y: H(x, ins(H(y, q), empty))  -- old heap into EMPTY
    ///   else:      H(y, ins(H(x, empty), q))  -- singleton into old prim_heap
    pub fn insert(&self, x: T) -> Self {
        let new_inner = match &self.inner {
            None => BootstrappedHeap {
                min: x,
                prim_heap: SkewBinomialHeap::empty(),
            },
            Some(h) => {
                if x <= h.min {
                    // New element is the minimum
                    // Insert the ENTIRE old heap into an EMPTY primitive heap
                    BootstrappedHeap {
                        min: x,
                        prim_heap: SkewBinomialHeap::empty().insert(Rc::new(BootstrappedHeap {
                            min: h.min.clone(),
                            prim_heap: h.prim_heap.clone(),
                        })),
                    }
                } else {
                    // Existing minimum stays
                    // Insert a singleton into the existing primitive heap
                    BootstrappedHeap {
                        min: h.min.clone(),
                        prim_heap: h.prim_heap.insert(Rc::new(BootstrappedHeap {
                            min: x,
                            prim_heap: SkewBinomialHeap::empty(),
                        })),
                    }
                }
            }
        };

        BrodalOkasakiHeap {
            inner: Some(new_inner),
            len: self.len + 1,
        }
    }

    /// Find the minimum element - O(1) worst-case
    pub fn find_min(&self) -> Option<&T> {
        self.inner.as_ref().map(|h| &h.min)
    }

    /// Merge two heaps - O(1) worst-case
    pub fn meld(&self, other: &Self) -> Self {
        match (&self.inner, &other.inner) {
            (None, _) => other.clone(),
            (_, None) => self.clone(),
            (Some(h1), Some(h2)) => {
                let (smaller, larger) = if h1.min <= h2.min { (h1, h2) } else { (h2, h1) };

                let new_inner = BootstrappedHeap {
                    min: smaller.min.clone(),
                    prim_heap: smaller.prim_heap.insert(Rc::new(BootstrappedHeap {
                        min: larger.min.clone(),
                        prim_heap: larger.prim_heap.clone(),
                    })),
                };

                BrodalOkasakiHeap {
                    inner: Some(new_inner),
                    len: self.len + other.len,
                }
            }
        }
    }

    /// Remove the minimum element - O(log n) worst-case
    pub fn delete_min(&self) -> Option<(T, Self)> {
        let h = self.inner.as_ref()?;

        if h.prim_heap.is_empty() {
            // Only one element
            return Some((h.min.clone(), BrodalOkasakiHeap::new()));
        }

        // Remove minimum from the primitive heap
        let (min_heap_rc, rest_prim) = h.prim_heap.delete_min()?;
        let min_heap = &*min_heap_rc;

        // Merge the rest with the removed heap's primitive heap
        let merged_prim = rest_prim.merge(&min_heap.prim_heap);

        let new_inner = BootstrappedHeap {
            min: min_heap.min.clone(),
            prim_heap: merged_prim,
        };

        Some((
            h.min.clone(),
            BrodalOkasakiHeap {
                inner: Some(new_inner),
                len: self.len - 1,
            },
        ))
    }

    /// Pop the minimum element (mutable version for convenience)
    pub fn pop(&mut self) -> Option<T> {
        let (min, new_heap) = self.delete_min()?;
        *self = new_heap;
        Some(min)
    }

    /// Push an element (mutable version for convenience)
    pub fn push(&mut self, x: T) {
        *self = self.insert(x);
    }

    /// Peek at the minimum element
    pub fn peek(&self) -> Option<&T> {
        self.find_min()
    }
}

impl<T: Ord + Clone> Default for BrodalOkasakiHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> FromIterator<T> for BrodalOkasakiHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut heap = BrodalOkasakiHeap::new();
        for x in iter {
            heap.push(x);
        }
        heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let heap: BrodalOkasakiHeap<i32> = BrodalOkasakiHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.find_min(), None);
    }

    #[test]
    fn test_insert_and_find_min() {
        let heap = BrodalOkasakiHeap::new();
        let heap = heap.insert(5);
        assert_eq!(heap.find_min(), Some(&5));

        let heap = heap.insert(3);
        assert_eq!(heap.find_min(), Some(&3));

        let heap = heap.insert(7);
        assert_eq!(heap.find_min(), Some(&3));

        let heap = heap.insert(1);
        assert_eq!(heap.find_min(), Some(&1));
    }

    #[test]
    fn test_delete_min() {
        let mut heap = BrodalOkasakiHeap::new();
        heap.push(5);
        heap.push(3);
        heap.push(7);
        heap.push(1);

        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(7));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_meld() {
        let heap1 = BrodalOkasakiHeap::new().insert(5).insert(3).insert(10);

        let heap2 = BrodalOkasakiHeap::new().insert(1).insert(7).insert(4);

        let merged = heap1.meld(&heap2);
        assert_eq!(merged.len(), 6);
        assert_eq!(merged.find_min(), Some(&1));

        let mut merged = merged;
        let mut values = Vec::new();
        while let Some(v) = merged.pop() {
            values.push(v);
        }
        assert_eq!(values, vec![1, 3, 4, 5, 7, 10]);
    }

    #[test]
    fn test_persistence() {
        let heap1 = BrodalOkasakiHeap::new().insert(5).insert(3);
        let heap2 = heap1.insert(1);

        // heap1 should be unchanged
        assert_eq!(heap1.find_min(), Some(&3));
        assert_eq!(heap1.len(), 2);

        // heap2 has the new element
        assert_eq!(heap2.find_min(), Some(&1));
        assert_eq!(heap2.len(), 3);
    }

    #[test]
    fn test_large_sequence() {
        let mut heap = BrodalOkasakiHeap::new();
        for i in (0..1000).rev() {
            heap.push(i);
        }

        for i in 0..1000 {
            assert_eq!(heap.pop(), Some(i));
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn test_duplicates() {
        let mut heap = BrodalOkasakiHeap::new();
        heap.push(5);
        heap.push(5);
        heap.push(3);
        heap.push(3);

        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(5));
        assert_eq!(heap.pop(), Some(5));
    }

    #[test]
    fn test_from_iterator() {
        let heap: BrodalOkasakiHeap<i32> = vec![5, 3, 7, 1, 4].into_iter().collect();
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.find_min(), Some(&1));
    }
}
