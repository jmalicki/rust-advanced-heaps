//! # optlike - Option-like storage with pluggable optimization strategies
//!
//! This crate provides a trait-based abstraction over `Option<T>` that allows
//! swapping between different storage representations:
//!
//! - **`Optimized`**: Uses sentinel-based storage (`NonMax*` for integers, NaN for floats)
//!   to achieve the same size as the underlying type.
//! - **`PlainOption`**: Uses standard `Option<T>` storage for comparison/benchmarking.
//!
//! ## Motivation
//!
//! In data structures like heaps, nodes often contain optional values that may be
//! "empty" or "deleted". Using `Option<T>` doubles the storage for small types:
//!
//! - `Option<i32>` is 8 bytes (4 for value + 4 for discriminant alignment)
//! - `Option<i64>` is 16 bytes (8 for value + 8 for discriminant alignment)
//!
//! By using sentinel values (like `i32::MAX` for integers or NaN for floats),
//! we can represent the same information in the original type's size.
//!
//! ## Trade-offs
//!
//! **Optimized storage:**
//! - Pro: Half the memory usage for integer/float priorities
//! - Pro: Better cache utilization (more nodes fit in cache)
//! - Con: Cannot represent the sentinel value (e.g., `i32::MAX` panics)
//! - Con: Small XOR overhead for NonMax types (1 cycle, pipelined)
//!
//! **Plain storage:**
//! - Pro: No value restrictions
//! - Pro: No XOR overhead
//! - Con: 2x memory for small types
//! - Con: Worse cache utilization
//!
//! ## Example
//!
//! ```rust
//! use optlike::{OptLike, Optimized, PlainOption};
//!
//! // Using optimized storage (default)
//! let mut storage = <i32 as OptLike<Optimized>>::some(42);
//! assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(42));
//!
//! // Using plain Option storage
//! let mut plain = <i32 as OptLike<PlainOption>>::some(42);
//! assert_eq!(<i32 as OptLike<PlainOption>>::get(&plain), Some(42));
//!
//! // Size difference
//! use std::mem::size_of;
//! assert_eq!(size_of::<<i32 as OptLike<Optimized>>::Storage>(), 4);
//! assert_eq!(size_of::<<i32 as OptLike<PlainOption>>::Storage>(), 8);
//! ```

#![warn(missing_docs)]

// =============================================================================
// Storage Strategies
// =============================================================================

/// Marker trait for storage strategies.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait StorageStrategy: sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Optimized {}
    impl Sealed for super::PlainOption {}
}

/// Use optimized sentinel-based storage.
///
/// For integer types, uses `NonMax*` from the `nonmax` crate, which stores
/// values XOR'd with MAX so that MAX becomes 0 (the niche value).
///
/// For float types, uses NaN as the sentinel value.
///
/// This achieves `size_of::<Storage>() == size_of::<T>()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Optimized;
impl StorageStrategy for Optimized {}

/// Use plain `Option<T>` storage.
///
/// This is provided for benchmarking comparisons and for cases where
/// the sentinel value must be representable.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlainOption;
impl StorageStrategy for PlainOption {}

// =============================================================================
// OptLike Trait
// =============================================================================

/// Trait for types that can be stored in an Option-like container
/// with potentially optimized representation.
///
/// The strategy parameter `S` controls which storage representation is used.
/// The default is [`Optimized`], which uses sentinel-based storage.
///
/// # Type Parameters
///
/// - `S`: The [`StorageStrategy`] to use. Either [`Optimized`] or [`PlainOption`].
///
/// # Associated Types
///
/// - `Storage`: The actual storage type. For `Optimized`, this might be
///   `Option<NonMaxI32>`. For `PlainOption`, this is `Option<Self>`.
pub trait OptLike<S: StorageStrategy = Optimized>: Sized + Clone {
    /// The storage type used to hold optional values of this type.
    type Storage: Clone;

    /// Create empty/none storage.
    fn none() -> Self::Storage;

    /// Create storage containing a value.
    ///
    /// # Panics
    ///
    /// For `Optimized` strategy, panics if the value equals the sentinel
    /// (e.g., `i32::MAX` for integers, `NaN` for floats).
    fn some(val: Self) -> Self::Storage;

    /// Check if storage is empty.
    fn is_none(storage: &Self::Storage) -> bool;

    /// Check if storage has a value.
    fn is_some(storage: &Self::Storage) -> bool {
        !Self::is_none(storage)
    }

    /// Get a copy of the stored value, or `None` if empty.
    ///
    /// Note: Returns by value because `NonMax*` types use XOR internally,
    /// so we cannot return a reference to the "real" value.
    fn get(storage: &Self::Storage) -> Option<Self>;

    /// Get a reference to the stored value, or `None` if empty.
    ///
    /// This is only available for `PlainOption` strategy. For `Optimized`
    /// strategy, this returns `None` because `NonMax*` types store values
    /// XOR'd with MAX internally, so the stored bytes don't represent the
    /// actual value.
    ///
    /// If you need to access the value by reference, use `PlainOption` strategy.
    fn get_ref(storage: &Self::Storage) -> Option<&Self> {
        // Default implementation returns None (for Optimized)
        let _ = storage;
        None
    }

    /// Take the value out, leaving empty storage.
    fn take(storage: &mut Self::Storage) -> Option<Self>;

    /// Replace the stored value, returning the old one.
    fn replace(storage: &mut Self::Storage, val: Self) -> Option<Self> {
        let old = Self::take(storage);
        *storage = Self::some(val);
        old
    }
}

// =============================================================================
// Blanket Implementation for PlainOption (any Clone type)
// =============================================================================

impl<T: Clone> OptLike<PlainOption> for T {
    type Storage = Option<T>;

    #[inline]
    fn none() -> Self::Storage {
        None
    }

    #[inline]
    fn some(val: Self) -> Self::Storage {
        Some(val)
    }

    #[inline]
    fn is_none(storage: &Self::Storage) -> bool {
        storage.is_none()
    }

    #[inline]
    fn get(storage: &Self::Storage) -> Option<Self> {
        storage.clone()
    }

    #[inline]
    fn get_ref(storage: &Self::Storage) -> Option<&Self> {
        storage.as_ref()
    }

    #[inline]
    fn take(storage: &mut Self::Storage) -> Option<Self> {
        storage.take()
    }
}

// =============================================================================
// Implementations for Integer Types (using NonMax)
// =============================================================================

macro_rules! impl_optlike_integer {
    ($int:ty, $nonmax:ty) => {
        impl OptLike<Optimized> for $int {
            type Storage = Option<$nonmax>;

            #[inline]
            fn none() -> Self::Storage {
                None
            }

            #[inline]
            fn some(val: Self) -> Self::Storage {
                Some(<$nonmax>::new(val).expect(concat!(
                    "priority cannot be ",
                    stringify!($int),
                    "::MAX (reserved as sentinel)"
                )))
            }

            #[inline]
            fn is_none(storage: &Self::Storage) -> bool {
                storage.is_none()
            }

            #[inline]
            fn get(storage: &Self::Storage) -> Option<Self> {
                storage.as_ref().map(|nm| nm.get())
            }

            #[inline]
            fn take(storage: &mut Self::Storage) -> Option<Self> {
                storage.take().map(|nm| nm.get())
            }
        }
    };
}

impl_optlike_integer!(i8, nonmax::NonMaxI8);
impl_optlike_integer!(i16, nonmax::NonMaxI16);
impl_optlike_integer!(i32, nonmax::NonMaxI32);
impl_optlike_integer!(i64, nonmax::NonMaxI64);
impl_optlike_integer!(i128, nonmax::NonMaxI128);
impl_optlike_integer!(isize, nonmax::NonMaxIsize);
impl_optlike_integer!(u8, nonmax::NonMaxU8);
impl_optlike_integer!(u16, nonmax::NonMaxU16);
impl_optlike_integer!(u32, nonmax::NonMaxU32);
impl_optlike_integer!(u64, nonmax::NonMaxU64);
impl_optlike_integer!(u128, nonmax::NonMaxU128);
impl_optlike_integer!(usize, nonmax::NonMaxUsize);

// =============================================================================
// Implementations for Float Types (using NaN sentinel)
// =============================================================================

/// Optimized storage for `f32` using NaN as the sentinel.
///
/// This wrapper is `repr(transparent)` so it has the same size as `f32`.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct OptF32(f32);

impl OptF32 {
    /// The sentinel value representing "none".
    pub const NONE: Self = OptF32(f32::NAN);

    /// Check if this storage is empty (contains NaN).
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0.is_nan()
    }
}

impl OptLike<Optimized> for f32 {
    type Storage = OptF32;

    #[inline]
    fn none() -> Self::Storage {
        OptF32::NONE
    }

    #[inline]
    fn some(val: Self) -> Self::Storage {
        assert!(
            !val.is_nan(),
            "NaN cannot be used as priority (reserved as sentinel)"
        );
        OptF32(val)
    }

    #[inline]
    fn is_none(storage: &Self::Storage) -> bool {
        storage.is_none()
    }

    #[inline]
    fn get(storage: &Self::Storage) -> Option<Self> {
        if storage.is_none() {
            None
        } else {
            Some(storage.0)
        }
    }

    #[inline]
    fn take(storage: &mut Self::Storage) -> Option<Self> {
        if storage.is_none() {
            None
        } else {
            let val = storage.0;
            *storage = OptF32::NONE;
            Some(val)
        }
    }
}

/// Optimized storage for `f64` using NaN as the sentinel.
///
/// This wrapper is `repr(transparent)` so it has the same size as `f64`.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct OptF64(f64);

impl OptF64 {
    /// The sentinel value representing "none".
    pub const NONE: Self = OptF64(f64::NAN);

    /// Check if this storage is empty (contains NaN).
    #[inline]
    pub fn is_none(&self) -> bool {
        self.0.is_nan()
    }
}

impl OptLike<Optimized> for f64 {
    type Storage = OptF64;

    #[inline]
    fn none() -> Self::Storage {
        OptF64::NONE
    }

    #[inline]
    fn some(val: Self) -> Self::Storage {
        assert!(
            !val.is_nan(),
            "NaN cannot be used as priority (reserved as sentinel)"
        );
        OptF64(val)
    }

    #[inline]
    fn is_none(storage: &Self::Storage) -> bool {
        storage.is_none()
    }

    #[inline]
    fn get(storage: &Self::Storage) -> Option<Self> {
        if storage.is_none() {
            None
        } else {
            Some(storage.0)
        }
    }

    #[inline]
    fn take(storage: &mut Self::Storage) -> Option<Self> {
        if storage.is_none() {
            None
        } else {
            let val = storage.0;
            *storage = OptF64::NONE;
            Some(val)
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // =========================================================================
    // Size verification tests
    // =========================================================================

    #[test]
    fn test_optimized_sizes() {
        // All optimized storages should be same size as the base type
        assert_eq!(
            size_of::<<i8 as OptLike<Optimized>>::Storage>(),
            size_of::<i8>()
        );
        assert_eq!(
            size_of::<<i16 as OptLike<Optimized>>::Storage>(),
            size_of::<i16>()
        );
        assert_eq!(
            size_of::<<i32 as OptLike<Optimized>>::Storage>(),
            size_of::<i32>()
        );
        assert_eq!(
            size_of::<<i64 as OptLike<Optimized>>::Storage>(),
            size_of::<i64>()
        );
        assert_eq!(
            size_of::<<u32 as OptLike<Optimized>>::Storage>(),
            size_of::<u32>()
        );
        assert_eq!(
            size_of::<<u64 as OptLike<Optimized>>::Storage>(),
            size_of::<u64>()
        );
        assert_eq!(
            size_of::<<f32 as OptLike<Optimized>>::Storage>(),
            size_of::<f32>()
        );
        assert_eq!(
            size_of::<<f64 as OptLike<Optimized>>::Storage>(),
            size_of::<f64>()
        );
    }

    #[test]
    fn test_plain_sizes() {
        // Plain storages are larger (have discriminant)
        assert!(size_of::<<i32 as OptLike<PlainOption>>::Storage>() > size_of::<i32>());
        assert!(size_of::<<i64 as OptLike<PlainOption>>::Storage>() > size_of::<i64>());
        assert!(size_of::<<f32 as OptLike<PlainOption>>::Storage>() > size_of::<f32>());
        assert!(size_of::<<f64 as OptLike<PlainOption>>::Storage>() > size_of::<f64>());
    }

    #[test]
    fn test_optimized_smaller_than_plain() {
        assert!(
            size_of::<<i32 as OptLike<Optimized>>::Storage>()
                < size_of::<<i32 as OptLike<PlainOption>>::Storage>()
        );
        assert!(
            size_of::<<i64 as OptLike<Optimized>>::Storage>()
                < size_of::<<i64 as OptLike<PlainOption>>::Storage>()
        );
    }

    // =========================================================================
    // Integer functionality tests
    // =========================================================================

    #[test]
    fn test_i32_optimized_basic() {
        let storage = <i32 as OptLike<Optimized>>::some(42);
        assert!(<i32 as OptLike<Optimized>>::is_some(&storage));
        assert!(!<i32 as OptLike<Optimized>>::is_none(&storage));
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(42));
    }

    #[test]
    fn test_i32_optimized_none() {
        let storage = <i32 as OptLike<Optimized>>::none();
        assert!(<i32 as OptLike<Optimized>>::is_none(&storage));
        assert!(!<i32 as OptLike<Optimized>>::is_some(&storage));
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), None);
    }

    #[test]
    fn test_i32_optimized_take() {
        let mut storage = <i32 as OptLike<Optimized>>::some(42);
        let taken = <i32 as OptLike<Optimized>>::take(&mut storage);
        assert_eq!(taken, Some(42));
        assert!(<i32 as OptLike<Optimized>>::is_none(&storage));
    }

    #[test]
    fn test_i32_optimized_negative() {
        let storage = <i32 as OptLike<Optimized>>::some(-100);
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(-100));
    }

    #[test]
    fn test_i32_optimized_zero() {
        let storage = <i32 as OptLike<Optimized>>::some(0);
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(0));
    }

    #[test]
    fn test_i32_optimized_min() {
        // MIN is valid (only MAX is the sentinel)
        let storage = <i32 as OptLike<Optimized>>::some(i32::MIN);
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(i32::MIN));
    }

    #[test]
    #[should_panic(expected = "MAX")]
    fn test_i32_optimized_rejects_max() {
        let _ = <i32 as OptLike<Optimized>>::some(i32::MAX);
    }

    #[test]
    fn test_i32_plain_accepts_max() {
        // PlainOption has no sentinel restriction
        let storage = <i32 as OptLike<PlainOption>>::some(i32::MAX);
        assert_eq!(<i32 as OptLike<PlainOption>>::get(&storage), Some(i32::MAX));
    }

    #[test]
    fn test_u64_optimized() {
        let storage = <u64 as OptLike<Optimized>>::some(12345678901234);
        assert_eq!(
            <u64 as OptLike<Optimized>>::get(&storage),
            Some(12345678901234)
        );
    }

    #[test]
    #[should_panic(expected = "MAX")]
    fn test_u64_optimized_rejects_max() {
        let _ = <u64 as OptLike<Optimized>>::some(u64::MAX);
    }

    // =========================================================================
    // Float functionality tests
    // =========================================================================

    #[test]
    fn test_f32_optimized_basic() {
        let storage = <f32 as OptLike<Optimized>>::some(3.14159);
        assert!(<f32 as OptLike<Optimized>>::is_some(&storage));
        assert_eq!(<f32 as OptLike<Optimized>>::get(&storage), Some(3.14159));
    }

    #[test]
    fn test_f32_optimized_none() {
        let storage = <f32 as OptLike<Optimized>>::none();
        assert!(<f32 as OptLike<Optimized>>::is_none(&storage));
        assert_eq!(<f32 as OptLike<Optimized>>::get(&storage), None);
    }

    #[test]
    fn test_f32_optimized_negative() {
        let storage = <f32 as OptLike<Optimized>>::some(-273.15);
        assert_eq!(<f32 as OptLike<Optimized>>::get(&storage), Some(-273.15));
    }

    #[test]
    fn test_f32_optimized_infinity() {
        // Infinity is valid (only NaN is the sentinel)
        let storage = <f32 as OptLike<Optimized>>::some(f32::INFINITY);
        assert_eq!(
            <f32 as OptLike<Optimized>>::get(&storage),
            Some(f32::INFINITY)
        );

        let storage = <f32 as OptLike<Optimized>>::some(f32::NEG_INFINITY);
        assert_eq!(
            <f32 as OptLike<Optimized>>::get(&storage),
            Some(f32::NEG_INFINITY)
        );
    }

    #[test]
    fn test_f32_optimized_zero() {
        let storage = <f32 as OptLike<Optimized>>::some(0.0);
        assert_eq!(<f32 as OptLike<Optimized>>::get(&storage), Some(0.0));

        let storage = <f32 as OptLike<Optimized>>::some(-0.0);
        assert_eq!(<f32 as OptLike<Optimized>>::get(&storage), Some(-0.0));
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn test_f32_optimized_rejects_nan() {
        let _ = <f32 as OptLike<Optimized>>::some(f32::NAN);
    }

    #[test]
    fn test_f32_plain_accepts_nan() {
        // PlainOption can store NaN
        let storage = <f32 as OptLike<PlainOption>>::some(f32::NAN);
        let val = <f32 as OptLike<PlainOption>>::get(&storage).unwrap();
        assert!(val.is_nan());
    }

    #[test]
    fn test_f64_optimized() {
        let storage = <f64 as OptLike<Optimized>>::some(2.718281828459045);
        assert_eq!(
            <f64 as OptLike<Optimized>>::get(&storage),
            Some(2.718281828459045)
        );
    }

    // =========================================================================
    // Replace functionality tests
    // =========================================================================

    #[test]
    fn test_replace() {
        let mut storage = <i32 as OptLike<Optimized>>::some(10);
        let old = <i32 as OptLike<Optimized>>::replace(&mut storage, 20);
        assert_eq!(old, Some(10));
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(20));
    }

    #[test]
    fn test_replace_none() {
        let mut storage = <i32 as OptLike<Optimized>>::none();
        let old = <i32 as OptLike<Optimized>>::replace(&mut storage, 42);
        assert_eq!(old, None);
        assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(42));
    }
}
