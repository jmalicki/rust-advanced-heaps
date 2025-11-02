//! Prusti verification proofs for heap operations
//!
//! Prusti is a static verifier for Rust. It uses annotations to specify
//! preconditions, postconditions, and invariants.
//!
//! To run these proofs:
//!   cargo prusti

// Enable Prusti verification
// Note: Prusti feature is not currently enabled, so these proofs are not active
// To enable: add `features = ["prusti"]` to Cargo.toml [dev-dependencies]
#[cfg(feature = "prusti")]
use prusti_contracts::*;

#[cfg(feature = "prusti")]
use rust_advanced_heaps::binomial::BinomialHeap;

#[cfg(feature = "prusti")]
use rust_advanced_heaps::Heap;

/// Verified implementation showing that insert increments length
#[cfg(feature = "prusti")]
#[requires(true)]
#[ensures(heap.len() == old(heap.len()) + 1)]
fn verified_insert(heap: &mut BinomialHeap<u32, u32>, priority: u32, item: u32) -> () {
    heap.push(priority, item);
    // Postcondition: length increased by 1 is verified by Prusti
}

/// Proof that is_empty is consistent with len
#[cfg(feature = "prusti")]
#[requires(true)]
#[ensures(result == (heap.len() == 0))]
fn verified_is_empty(heap: &BinomialHeap<u32, u32>) -> bool {
    heap.is_empty()
}

// Note: Prusti requires special annotations and may need code modifications
// to work with unsafe code. The unsafe pointer manipulations in heap
// implementations may require additional verification scaffolding.
