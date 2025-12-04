//! Pluggable storage backends for heap node management
//!
//! This module provides a trait abstraction over node storage, allowing heaps to use
//! different memory management strategies:
//!
//! - [`RcStorage`]: Default, stable implementation using `Rc<RefCell<N>>` (current behavior)
//! - [`SlotMapStorage`]: Experimental arena-based storage using `slotmap` for better cache locality
//!
//! # Design
//!
//! The [`NodeStorage`] trait abstracts over how nodes are allocated, accessed, and freed.
//! Each storage backend provides:
//! - A `Key` type for referencing nodes
//! - A `WeakKey` type for non-owning references (used for parent pointers)
//! - Methods for insert, remove, get, get_mut operations
//!
//! # Example
//!
//! ```rust,ignore
//! // Using the default RcStorage (stable)
//! let mut heap: SkewBinomialHeap<String, i32> = SkewBinomialHeap::new();
//!
//! // Using SlotMapStorage (experimental, requires feature flag)
//! #[cfg(feature = "arena-storage")]
//! let mut heap: SkewBinomialHeapArena<String, i32> = SkewBinomialHeapArena::new();
//! ```

use std::cell::RefCell;
use std::fmt;
use std::hash::Hash;
use std::rc::{Rc, Weak};

#[cfg(feature = "arena-storage")]
use slotmap::{new_key_type, SlotMap};

/// Trait for node storage backends
///
/// This trait abstracts over how heap nodes are stored and referenced,
/// enabling different memory management strategies (Rc-based vs arena-based).
///
/// # Key vs WeakKey
///
/// - `Key`: A strong reference that keeps the node alive (for Rc) or valid (for slotmap)
/// - `WeakKey`: A weak reference that does not prevent removal, used for parent pointers
///
/// # Note on get/get_mut
///
/// For `RcStorage`, these methods don't work due to `RefCell` semantics.
/// Use `key.borrow()` and `key.borrow_mut()` directly instead.
/// For `SlotMapStorage`, these methods work as expected.
pub trait NodeStorage<N>: Default {
    /// Strong reference key type
    type Key: Clone + Eq + Hash + fmt::Debug;

    /// Weak reference key type - does not own the node, can detect if node was removed
    type WeakKey: Clone + fmt::Debug;

    /// Insert a node, returning a key to reference it
    fn insert(&mut self, node: N) -> Self::Key;

    /// Remove a node by key, returning the node if it existed
    fn remove(&mut self, key: Self::Key) -> Option<N>;

    /// Get an immutable reference to a node (not supported for RcStorage)
    fn get(&self, key: &Self::Key) -> Option<&N>;

    /// Get a mutable reference to a node (not supported for RcStorage)
    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut N>;

    /// Create a weak reference from a strong key
    fn downgrade(&self, key: &Self::Key) -> Self::WeakKey;

    /// Attempt to upgrade a weak reference to a strong key
    /// Returns None if the node has been removed
    fn upgrade(&self, weak: &Self::WeakKey) -> Option<Self::Key>;

    /// Check if two keys refer to the same node
    fn keys_eq(a: &Self::Key, b: &Self::Key) -> bool;

    /// Create an "empty" weak key (analogous to Weak::new())
    fn empty_weak() -> Self::WeakKey;

    /// Check if a weak key is empty/null (was never set or points to nothing)
    fn is_weak_empty(weak: &Self::WeakKey) -> bool;
}

// ============================================================================
// RcStorage - Default stable implementation
// ============================================================================

/// Rc-based node storage (default, stable)
///
/// This is the traditional implementation using `Rc<RefCell<N>>` for nodes.
/// It provides automatic memory management through reference counting.
///
/// # Characteristics
/// - Automatic cleanup when last reference is dropped
/// - Weak references for parent pointers
/// - Higher per-node overhead (~24 bytes for Rc+RefCell)
/// - Scattered memory allocation (each node is a separate heap allocation)
///
/// # Note
///
/// For `RcStorage`, the `get()` and `get_mut()` methods return `None` because
/// `RefCell` requires runtime borrow checking. Use `key.borrow()` and
/// `key.borrow_mut()` directly on the key instead.
#[derive(Debug)]
pub struct RcStorage<N> {
    _phantom: std::marker::PhantomData<N>,
}

impl<N> Default for RcStorage<N> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<N> Clone for RcStorage<N> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// Key type for RcStorage - wraps `Rc<RefCell<N>>`
///
/// This key type provides direct access to the node through `borrow()` and `borrow_mut()`.
pub struct RcKey<N>(Rc<RefCell<N>>);

impl<N> fmt::Debug for RcKey<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RcKey").field(&Rc::as_ptr(&self.0)).finish()
    }
}

impl<N> Clone for RcKey<N> {
    fn clone(&self) -> Self {
        RcKey(Rc::clone(&self.0))
    }
}

impl<N> PartialEq for RcKey<N> {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl<N> Eq for RcKey<N> {}

impl<N> Hash for RcKey<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

/// Weak key type for RcStorage - wraps `Weak<RefCell<N>>`
pub struct RcWeakKey<N>(Weak<RefCell<N>>);

impl<N> fmt::Debug for RcWeakKey<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RcWeakKey").field(&self.0.as_ptr()).finish()
    }
}

impl<N> Clone for RcWeakKey<N> {
    fn clone(&self) -> Self {
        RcWeakKey(Weak::clone(&self.0))
    }
}

impl<N> RcKey<N> {
    /// Get the underlying Rc for direct RefCell access
    #[inline]
    pub fn rc(&self) -> &Rc<RefCell<N>> {
        &self.0
    }

    /// Borrow the inner value immutably
    #[inline]
    pub fn borrow(&self) -> std::cell::Ref<'_, N> {
        self.0.borrow()
    }

    /// Borrow the inner value mutably
    #[inline]
    pub fn borrow_mut(&self) -> std::cell::RefMut<'_, N> {
        self.0.borrow_mut()
    }

    /// Get raw pointer for unsafe peek operations
    #[inline]
    pub fn as_ptr(&self) -> *mut N {
        self.0.as_ptr()
    }
}

impl<N> RcWeakKey<N> {
    /// Check if two weak keys point to the same node
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl<N> NodeStorage<N> for RcStorage<N> {
    type Key = RcKey<N>;
    type WeakKey = RcWeakKey<N>;

    fn insert(&mut self, node: N) -> Self::Key {
        RcKey(Rc::new(RefCell::new(node)))
    }

    fn remove(&mut self, key: Self::Key) -> Option<N> {
        // For Rc-based storage, we can only extract if this is the last strong reference.
        // In practice, the heap structure should ensure this when a node is being removed.
        // Returns None if there are other strong references (node cannot be extracted).
        Rc::try_unwrap(key.0).ok().map(|cell| cell.into_inner())
    }

    fn get(&self, _key: &Self::Key) -> Option<&N> {
        // Cannot return reference through RefCell - use key.borrow() instead
        None
    }

    fn get_mut(&mut self, _key: &Self::Key) -> Option<&mut N> {
        // Cannot return reference through RefCell - use key.borrow_mut() instead
        None
    }

    fn downgrade(&self, key: &Self::Key) -> Self::WeakKey {
        RcWeakKey(Rc::downgrade(&key.0))
    }

    fn upgrade(&self, weak: &Self::WeakKey) -> Option<Self::Key> {
        weak.0.upgrade().map(RcKey)
    }

    fn keys_eq(a: &Self::Key, b: &Self::Key) -> bool {
        Rc::ptr_eq(&a.0, &b.0)
    }

    fn empty_weak() -> Self::WeakKey {
        RcWeakKey(Weak::new())
    }

    fn is_weak_empty(weak: &Self::WeakKey) -> bool {
        // A weak reference is "empty" if it was never connected to an Rc
        // or if all strong references have been dropped
        weak.0.strong_count() == 0
    }
}

// ============================================================================
// SlotMapStorage - Experimental arena-based implementation
// ============================================================================

#[cfg(feature = "arena-storage")]
new_key_type! {
    /// SlotMap key type for arena storage
    pub struct SlotMapNodeKey;
}

/// SlotMap-based arena storage (experimental)
///
/// This implementation uses `slotmap` for contiguous node storage,
/// providing better cache locality at the cost of manual lifetime management.
///
/// # Characteristics
/// - Contiguous memory allocation (better cache locality)
/// - Generational keys detect stale references
/// - Lower per-node overhead (~8 bytes for key)
/// - Requires explicit removal (no automatic cleanup)
///
/// # Feature Flag
/// Requires the `arena-storage` feature to be enabled.
#[cfg(feature = "arena-storage")]
#[derive(Debug, Clone)]
pub struct SlotMapStorage<N> {
    nodes: SlotMap<SlotMapNodeKey, N>,
}

#[cfg(feature = "arena-storage")]
impl<N> Default for SlotMapStorage<N> {
    fn default() -> Self {
        Self {
            nodes: SlotMap::with_key(),
        }
    }
}

/// Weak key for SlotMapStorage
///
/// Since slotmap keys are generational, they naturally detect stale references.
/// The weak key wraps an `Option<Key>` - None represents an empty/null weak reference.
#[cfg(feature = "arena-storage")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotMapWeakKey(Option<SlotMapNodeKey>);

#[cfg(feature = "arena-storage")]
impl<N> NodeStorage<N> for SlotMapStorage<N> {
    type Key = SlotMapNodeKey;
    type WeakKey = SlotMapWeakKey;

    fn insert(&mut self, node: N) -> Self::Key {
        self.nodes.insert(node)
    }

    fn remove(&mut self, key: Self::Key) -> Option<N> {
        self.nodes.remove(key)
    }

    fn get(&self, key: &Self::Key) -> Option<&N> {
        self.nodes.get(*key)
    }

    fn get_mut(&mut self, key: &Self::Key) -> Option<&mut N> {
        self.nodes.get_mut(*key)
    }

    fn downgrade(&self, key: &Self::Key) -> Self::WeakKey {
        SlotMapWeakKey(Some(*key))
    }

    fn upgrade(&self, weak: &Self::WeakKey) -> Option<Self::Key> {
        weak.0.filter(|k| self.nodes.contains_key(*k))
    }

    fn keys_eq(a: &Self::Key, b: &Self::Key) -> bool {
        a == b
    }

    fn empty_weak() -> Self::WeakKey {
        SlotMapWeakKey(None)
    }

    fn is_weak_empty(weak: &Self::WeakKey) -> bool {
        // Note: This only checks if the weak key was created empty (None).
        // Unlike RcStorage, we cannot check if the node was removed because
        // is_weak_empty is a static method with no access to the storage.
        // SlotMapWeakKey is just an Option<Key> with no back-reference to storage.
        // Use upgrade() to check if a key is still valid.
        weak.0.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc_storage_basic() {
        let mut storage: RcStorage<i32> = RcStorage::default();

        let key = storage.insert(42);
        assert_eq!(*key.borrow(), 42);

        *key.borrow_mut() = 100;
        assert_eq!(*key.borrow(), 100);

        let weak = storage.downgrade(&key);
        let upgraded = storage.upgrade(&weak);
        assert!(upgraded.is_some());
        assert!(RcStorage::<i32>::keys_eq(&key, &upgraded.unwrap()));
    }

    #[test]
    fn test_rc_storage_weak_empty() {
        let storage: RcStorage<i32> = RcStorage::default();
        let empty = RcStorage::<i32>::empty_weak();
        assert!(RcStorage::<i32>::is_weak_empty(&empty));
        assert!(storage.upgrade(&empty).is_none());
    }

    #[test]
    fn test_rc_storage_weak_becomes_invalid() {
        let mut storage: RcStorage<i32> = RcStorage::default();

        let key = storage.insert(42);
        let weak = storage.downgrade(&key);

        // Weak can be upgraded while key exists
        assert!(storage.upgrade(&weak).is_some());

        // After removing, weak becomes invalid
        let _removed = storage.remove(key);
        assert!(storage.upgrade(&weak).is_none());
    }

    #[cfg(feature = "arena-storage")]
    #[test]
    fn test_slotmap_storage_basic() {
        let mut storage: SlotMapStorage<i32> = SlotMapStorage::default();

        let key = storage.insert(42);
        assert_eq!(storage.get(&key), Some(&42));

        *storage.get_mut(&key).unwrap() = 100;
        assert_eq!(storage.get(&key), Some(&100));

        let weak = storage.downgrade(&key);
        let upgraded = storage.upgrade(&weak);
        assert!(upgraded.is_some());
        assert!(SlotMapStorage::<i32>::keys_eq(&key, &upgraded.unwrap()));

        // Remove and check weak is now invalid
        let removed = storage.remove(key);
        assert_eq!(removed, Some(100));
        assert!(storage.upgrade(&weak).is_none());
    }

    #[cfg(feature = "arena-storage")]
    #[test]
    fn test_slotmap_storage_weak_empty() {
        let storage: SlotMapStorage<i32> = SlotMapStorage::default();
        let empty = SlotMapStorage::<i32>::empty_weak();
        assert!(SlotMapStorage::<i32>::is_weak_empty(&empty));
        assert!(storage.upgrade(&empty).is_none());
    }
}
