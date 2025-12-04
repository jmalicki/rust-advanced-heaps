//! Rank/degree type and utilities for heap node tree depth tracking.
//!
//! # Why u8?
//!
//! In heap data structures, "rank" or "degree" represents the depth or number
//! of children of a node. The maximum possible rank is bounded by `log₂(n)`
//! where `n` is the number of elements in the heap.
//!
//! For practical purposes:
//! - 2⁶⁴ elements → max rank ~64
//! - 2²⁵⁵ elements → max rank 255
//!
//! Since `u8` can represent values 0-255, it supports heaps with up to 2²⁵⁵
//! elements—far more than could ever fit in memory. Using `u8` instead of
//! `usize` saves 7 bytes per node on 64-bit systems.
//!
//! # Memory Savings
//!
//! For a heap with 1 million nodes:
//! - `usize` rank: 8 MB just for rank fields
//! - `u8` rank: 1 MB just for rank fields
//!
//! Plus, the smaller field often eliminates padding, saving even more.
//!
//! # Runtime Checks
//!
//! The [`checked_increment`] function provides a safe way to increase rank,
//! panicking if the theoretical limit is exceeded (which would indicate a bug,
//! since it's mathematically impossible with valid heap operations).

/// Type alias for node rank/degree.
///
/// Using `u8` saves 7 bytes per node compared to `usize` on 64-bit systems.
/// The maximum value (255) supports heaps with up to 2²⁵⁵ elements.
pub type Rank = u8;

/// Maximum valid rank value.
///
/// This is `u8::MAX` (255), which supports heaps with up to 2²⁵⁵ elements.
/// In practice, rank values will never exceed ~64 even for the largest
/// possible heaps on current hardware.
pub const MAX_RANK: Rank = u8::MAX;

/// Safely increment a rank value, panicking on overflow.
///
/// # Panics
///
/// Panics if `rank == MAX_RANK`. This should never happen in practice since
/// it would require a heap with more than 2²⁵⁵ elements.
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::rank::{checked_increment, Rank};
///
/// let rank: Rank = 5;
/// assert_eq!(checked_increment(rank), 6);
/// ```
#[inline]
pub fn checked_increment(rank: Rank) -> Rank {
    rank.checked_add(1).expect(
        "rank overflow: this should be impossible since max rank is log₂(n) \
         and u8::MAX (255) supports heaps with up to 2²⁵⁵ elements",
    )
}

/// Safely decrement a rank value, returning 0 if already at minimum.
///
/// This uses saturating subtraction since rank can legitimately reach 0
/// (leaf nodes have rank 0).
///
/// # Example
///
/// ```rust
/// use rust_advanced_heaps::rank::{saturating_decrement, Rank};
///
/// let rank: Rank = 5;
/// assert_eq!(saturating_decrement(rank), 4);
///
/// let zero: Rank = 0;
/// assert_eq!(saturating_decrement(zero), 0);
/// ```
#[inline]
pub fn saturating_decrement(rank: Rank) -> Rank {
    rank.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_increment() {
        assert_eq!(checked_increment(0), 1);
        assert_eq!(checked_increment(127), 128);
        assert_eq!(checked_increment(254), 255);
    }

    #[test]
    #[should_panic(expected = "rank overflow")]
    fn test_checked_increment_overflow() {
        checked_increment(255);
    }

    #[test]
    fn test_saturating_decrement() {
        assert_eq!(saturating_decrement(5), 4);
        assert_eq!(saturating_decrement(1), 0);
        assert_eq!(saturating_decrement(0), 0);
    }

    #[test]
    fn test_max_rank_sufficient() {
        // Document that MAX_RANK (255) can represent any practical heap depth.
        // A heap with 2^64 elements (more than addressable memory) has max rank ~64.
        // Since MAX_RANK = 255, we have plenty of headroom.
        assert_eq!(MAX_RANK, 255);

        // Verify that the rank type is what we expect
        assert_eq!(std::mem::size_of::<Rank>(), 1);
    }
}
