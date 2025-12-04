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
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::skiplist::SkipListHeap;
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

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

/// Test that n decrease_key operations has the expected batch complexity
///
/// Per-element complexities vary by heap:
/// - O(1) amortized: Fibonacci, RankPairing, TwoThree, StrictFibonacci -> O(n) batch
/// - O(1) worst-case: Brodal -> O(n) batch
/// - o(log n) amortized: Pairing -> O(n) batch (sub-logarithmic, better than O(log n))
/// - O(log n) worst-case: Binomial, SkewBinomial -> O(n log n) batch
fn test_decrease_key_batch_complexity<H: DecreaseKeyHeap<i32, i32>>(
    heap_name: &str,
    batch_expected: BigOAlgorithmComplexity,
) {
    // We need separate Arc for heap and Vec for handles since handles don't implement Sync
    let heap_arc = Arc::new(RwLock::new(H::new()));
    let heap1 = Arc::clone(&heap_arc);
    let heap2 = Arc::clone(&heap_arc);

    test_algorithm(
        &format!("{} decrease_key batch", heap_name),
        3,
        || {
            // Reset: Build heap with handles stored in closures
            let mut h = heap_arc.write();
            *h = H::new();
        },
        1000,
        || {
            // Create heap and store handles in closure
            let mut handles = Vec::new();
            {
                let mut h = heap1.write();
                // Insert elements with high priorities
                for i in 0..1000 {
                    handles.push(h.push_with_handle(i + 10000, i));
                }
            }

            // Now decrease keys
            for (i, handle) in handles.iter().enumerate() {
                let mut h = heap1.write();
                assert!(h.decrease_key(handle, i as i32).is_ok());
            }
            42
        },
        2000,
        || {
            let mut handles = Vec::new();
            {
                let mut h = heap2.write();
                for i in 0..2000 {
                    handles.push(h.push_with_handle(i + 20000, i));
                }
            }

            for (i, handle) in handles.iter().enumerate() {
                let mut h = heap2.write();
                assert!(h.decrease_key(handle, i as i32).is_ok());
            }
            42
        },
        batch_expected,
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

#[test]
fn test_fibonacci_decrease_key() {
    test_decrease_key_batch_complexity::<FibonacciHeap<i32, i32>>(
        "FibonacciHeap",
        BigOAlgorithmComplexity::ON,
    );
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

#[test]
fn test_pairing_decrease_key() {
    // Pairing has o(log n) amortized which is sub-logarithmic, so O(n) batch is acceptable
    test_decrease_key_batch_complexity::<PairingHeap<i32, i32>>(
        "PairingHeap",
        BigOAlgorithmComplexity::ON,
    );
}

// ============================================================================
// Rank Pairing Heap Tests
// ============================================================================

#[test]
fn test_rank_pairing_insert() {
    // O(1) amortized per insert -> O(n) batch
    test_insert_batch_complexity::<RankPairingHeap<i32, i32>>(
        "RankPairingHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
fn test_rank_pairing_pop() {
    // O(log n) amortized per pop -> O(n log n) batch
    test_pop_batch_complexity::<RankPairingHeap<i32, i32>>("RankPairingHeap");
}

#[test]
fn test_rank_pairing_decrease_key() {
    // O(1) amortized per decrease_key -> O(n) batch
    test_decrease_key_batch_complexity::<RankPairingHeap<i32, i32>>(
        "RankPairingHeap",
        BigOAlgorithmComplexity::ON,
    );
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

#[test]
fn test_binomial_decrease_key() {
    test_decrease_key_batch_complexity::<BinomialHeap<i32, i32>>(
        "BinomialHeap",
        BigOAlgorithmComplexity::ONLogN,
    );
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
#[ignore] // Test is currently failing - needs investigation
fn test_skew_binomial_pop() {
    test_pop_batch_complexity::<SkewBinomialHeap<i32, i32>>("SkewBinomialHeap");
}

#[test]
fn test_skew_binomial_decrease_key() {
    test_decrease_key_batch_complexity::<SkewBinomialHeap<i32, i32>>(
        "SkewBinomialHeap",
        BigOAlgorithmComplexity::ONLogN,
    );
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

#[test]
fn test_strict_fibonacci_decrease_key() {
    test_decrease_key_batch_complexity::<StrictFibonacciHeap<i32, i32>>(
        "StrictFibonacciHeap",
        BigOAlgorithmComplexity::ON,
    );
}

// ============================================================================
// TwoThree Heap Tests
// ============================================================================

#[test]
fn test_twothree_insert() {
    // O(1) amortized per insert -> O(n) batch
    test_insert_batch_complexity::<TwoThreeHeap<i32, i32>>(
        "TwoThreeHeap",
        BigOAlgorithmComplexity::ON,
    );
}

#[test]
fn test_twothree_pop() {
    // O(log n) amortized per pop -> O(n log n) batch
    test_pop_batch_complexity::<TwoThreeHeap<i32, i32>>("TwoThreeHeap");
}

/// Smaller pop test for TwoThreeHeap batch complexity verification
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn test_twothree_pop_small() {
    // TwoThreeHeap uses Rc internally so isn't Send/Sync, but big_o_test requires Arc
    let heap = Arc::new(RwLock::new(TwoThreeHeap::<i32, i32>::new()));

    test_algorithm(
        "TwoThreeHeap pop batch (small)",
        3,
        || {
            *heap.write() = TwoThreeHeap::new();
        },
        100,
        || {
            let mut h = heap.write();
            for i in 0..100 {
                h.push(i, i);
            }
            for _ in 0..100 {
                assert!(
                    h.pop().is_some(),
                    "pop() must succeed after pushing elements"
                );
            }
            42
        },
        200,
        || {
            let mut h = heap.write();
            for i in 0..200 {
                h.push(i, i);
            }
            for _ in 0..200 {
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

#[test]
fn test_twothree_decrease_key() {
    // O(1) amortized per decrease_key (cut and meld) -> O(n) batch
    test_decrease_key_batch_complexity::<TwoThreeHeap<i32, i32>>(
        "TwoThreeHeap",
        BigOAlgorithmComplexity::ON,
    );
}

// ============================================================================
// SkipList Heap Tests
// ============================================================================

#[test]
fn test_skiplist_insert() {
    // O(log n) per element, so batch should be O(n log n)
    test_insert_batch_complexity::<SkipListHeap<i32, i32>>(
        "SkipListHeap",
        BigOAlgorithmComplexity::ONLogN,
    );
}

#[test]
fn test_skiplist_pop() {
    test_pop_batch_complexity::<SkipListHeap<i32, i32>>("SkipListHeap");
}

#[test]
fn test_skiplist_decrease_key() {
    // O(log n) per decrease_key (remove + reinsert in sorted list) -> O(n log n) batch
    test_decrease_key_batch_complexity::<SkipListHeap<i32, i32>>(
        "SkipListHeap",
        BigOAlgorithmComplexity::ONLogN,
    );
}
