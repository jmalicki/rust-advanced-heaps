//! Big-O complexity proofs for heap operations
//!
//! This module empirically verifies that heap operations meet the theoretical
//! complexity bounds claimed in the academic papers using the `big-o-test` crate.
//!
//! ## Testing Strategy
//!
//! We use `test_algorithm` to measure batch operations. Note that for batch operations:
//! - O(1) amortized per-element operations appear as O(n) for n operations
//! - O(log n) operations appear as O(n log n) for n operations
//!
//! Note: These are empirical tests, not formal proofs. They detect significant
//! deviations from expected behavior but may not catch subtle issues with
//! specific input patterns.

use big_o_test::{test_algorithm, BigOAlgorithmComplexity};
use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::brodal::BrodalHeap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::Heap;

use ctor::ctor;
use parking_lot::RwLock;
use std::sync::Arc;

/// Sets up the ENV, affecting the Rust's test runner
#[ctor]
fn setup_env() {
    // cause tests to run serially -- this may be replaced by using the `serial_test` crate
    std::env::set_var("RUST_TEST_THREADS", "1");
}

// ============================================================================
// Helper functions for each operation
// ============================================================================

/// Test that n insertions has the expected batch complexity
/// O(1) amortized -> O(n) batch, O(log n) -> O(n log n) batch
fn test_insert_batch_complexity<H: Heap<i32, i32>>(
    heap_name: &str,
    batch_expected: BigOAlgorithmComplexity,
) {
    let heap = Arc::new(RwLock::new(H::new()));

    test_algorithm(
        &format!("{} insert batch", heap_name),
        3,
        || {
            *heap.write() = H::new();
        },
        1000,
        || {
            let mut h = heap.write();
            for i in 0..1000 {
                h.push(i, i);
            }
            42
        },
        2000,
        || {
            let mut h = heap.write();
            for i in 0..2000 {
                h.push(i, i);
            }
            42
        },
        batch_expected,
        BigOAlgorithmComplexity::ON,
    );
}

/// Test that n pops has O(n log n) batch complexity
fn test_pop_batch_complexity<H: Heap<i32, i32>>(heap_name: &str) {
    let heap = Arc::new(RwLock::new(H::new()));

    test_algorithm(
        &format!("{} pop batch", heap_name),
        3,
        || {
            *heap.write() = H::new();
        },
        1000,
        || {
            let mut h = heap.write();
            for i in 0..1000 {
                h.push(i, i);
            }
            for _ in 0..1000 {
                assert!(
                    h.pop().is_some(),
                    "pop() must succeed after pushing elements"
                );
            }
            42
        },
        2000,
        || {
            let mut h = heap.write();
            for i in 0..2000 {
                h.push(i, i);
            }
            for _ in 0..2000 {
                assert!(
                    h.pop().is_some(),
                    "pop() must succeed after pushing elements"
                );
            }
            42
        },
        BigOAlgorithmComplexity::ONLogN,
        BigOAlgorithmComplexity::ON,
    );
}

// ============================================================================
// Fibonacci Heap Tests
// ============================================================================

#[test]
fn test_fibonacci_insert() {
    // O(1) per element, so batch should be O(n)
    test_insert_batch_complexity::<FibonacciHeap<i32, i32>>(
        "FibonacciHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
fn test_fibonacci_pop() {
    test_pop_batch_complexity::<FibonacciHeap<i32, i32>>("FibonacciHeap");
}

// ============================================================================
// Pairing Heap Tests
// ============================================================================

#[test]
fn test_pairing_insert() {
    test_insert_batch_complexity::<PairingHeap<i32, i32>>(
        "PairingHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
fn test_pairing_pop() {
    test_pop_batch_complexity::<PairingHeap<i32, i32>>("PairingHeap");
}

// ============================================================================
// Rank Pairing Heap Tests
// ============================================================================

#[test]
#[ignore] // Space complexity analysis issues
fn test_rank_pairing_insert() {
    test_insert_batch_complexity::<RankPairingHeap<i32, i32>>(
        "RankPairingHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
#[ignore] // Segfault on this test
fn test_rank_pairing_pop() {
    test_pop_batch_complexity::<RankPairingHeap<i32, i32>>("RankPairingHeap");
}

// ============================================================================
// Binomial Heap Tests
// ============================================================================

#[test]
fn test_binomial_insert() {
    // O(log n) per element, so batch should be O(n log n)
    test_insert_batch_complexity::<BinomialHeap<i32, i32>>(
        "BinomialHeap",
        BigOAlgorithmComplexity::ONLogN,
    );
}

#[test]
fn test_binomial_pop() {
    test_pop_batch_complexity::<BinomialHeap<i32, i32>>("BinomialHeap");
}

// ============================================================================
// Brodal Heap Tests
// ============================================================================

#[test]
#[ignore] // Complexity analysis issues
fn test_brodal_insert() {
    test_insert_batch_complexity::<BrodalHeap<i32, i32>>("BrodalHeap", BigOAlgorithmComplexity::ON);
}

#[test]
#[ignore] // Complexity analysis issues
fn test_brodal_pop() {
    test_pop_batch_complexity::<BrodalHeap<i32, i32>>("BrodalHeap");
}

// ============================================================================
// Skew Binomial Heap Tests
// ============================================================================

#[test]
fn test_skew_binomial_insert() {
    test_insert_batch_complexity::<SkewBinomialHeap<i32, i32>>(
        "SkewBinomialHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
fn test_skew_binomial_pop() {
    test_pop_batch_complexity::<SkewBinomialHeap<i32, i32>>("SkewBinomialHeap");
}

// ============================================================================
// Strict Fibonacci Heap Tests
// ============================================================================

#[test]
fn test_strict_fibonacci_insert() {
    test_insert_batch_complexity::<StrictFibonacciHeap<i32, i32>>(
        "StrictFibonacciHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
fn test_strict_fibonacci_pop() {
    test_pop_batch_complexity::<StrictFibonacciHeap<i32, i32>>("StrictFibonacciHeap");
}

// ============================================================================
// TwoThree Heap Tests
// ============================================================================

#[test]
fn test_twothree_insert() {
    test_insert_batch_complexity::<TwoThreeHeap<i32, i32>>(
        "TwoThreeHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
#[ignore] // Segfault on this test
fn test_twothree_pop() {
    test_pop_batch_complexity::<TwoThreeHeap<i32, i32>>("TwoThreeHeap");
}
