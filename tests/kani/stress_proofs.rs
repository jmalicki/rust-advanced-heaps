//! Stress tests for Kani verification
//!
//! These proofs perform long sequences of operations on multiple heap implementations
//! simultaneously, using the same inputs, and verify that all implementations produce
//! the same results. This is effective for catching implementation bugs - if different
//! heaps are supposed to produce the same results but don't, that's a bug.
//!
//! These tests use much higher unwind bounds to test longer operation sequences.

#[cfg(kani)]
use rust_advanced_heaps::binomial::BinomialHeap;
#[cfg(kani)]
use rust_advanced_heaps::brodal::BrodalHeap;
#[cfg(kani)]
use rust_advanced_heaps::fibonacci::FibonacciHeap;
#[cfg(kani)]
use rust_advanced_heaps::pairing::PairingHeap;
#[cfg(kani)]
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
#[cfg(kani)]
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
#[cfg(kani)]
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
#[cfg(kani)]
use rust_advanced_heaps::twothree::TwoThreeHeap;
#[cfg(kani)]
use rust_advanced_heaps::Heap;

/// Helper function to perform a sequence of operations and verify all heaps produce same results
#[cfg(kani)]
fn verify_heaps_consistent<H1: Heap<u32, u32>, H2: Heap<u32, u32>>(
    mut heap1: H1,
    mut heap2: H2,
    num_operations: usize,
) {
    for _ in 0..num_operations {
        let op = kani::any::<u8>();
        let op_kind = op % 5; // 5 operation types: push, pop, peek, decrease_key, merge

        match op_kind {
            0 => {
                // Push operation
                let priority = kani::any();
                let item = kani::any();
                heap1.push(priority, item);
                heap2.push(priority, item);

                // Verify both heaps have same length
                assert!(
                    heap1.len() == heap2.len(),
                    "Heaps have different lengths after push: {} vs {}",
                    heap1.len(),
                    heap2.len()
                );
                assert!(
                    heap1.is_empty() == heap2.is_empty(),
                    "Heaps have different emptiness after push"
                );
            }
            1 => {
                // Pop operation
                if !heap1.is_empty() {
                    let result1 = heap1.pop();
                    let result2 = heap2.pop();

                    assert!(
                        result1 == result2,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        result1,
                        result2
                    );
                    assert!(
                        heap1.len() == heap2.len(),
                        "Heaps have different lengths after pop: {} vs {}",
                        heap1.len(),
                        heap2.len()
                    );
                }
            }
            2 => {
                // Peek/find_min operation
                let result1 = heap1.peek();
                let result2 = heap2.peek();

                assert!(
                    result1 == result2,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    result1,
                    result2
                );
            }
            3 => {
                // Decrease_key operation (requires handles from previous pushes)
                // This is simplified - in practice we'd track handles
                // For now, we'll skip if heaps are empty
                if heap1.len() > 0 {
                    // We can't easily test decrease_key without tracking handles
                    // So we'll do a peek instead
                    let _ = heap1.peek();
                    let _ = heap2.peek();
                }
            }
            4 => {
                // Merge operation - create two new heaps and merge them
                let mut other1 = H1::new();
                let mut other2 = H2::new();

                let num_items = kani::any::<u8>() % 5;
                for _ in 0..num_items {
                    let priority = kani::any();
                    let item = kani::any();
                    other1.push(priority, item);
                    other2.push(priority, item);
                }

                heap1.merge(other1);
                heap2.merge(other2);

                assert!(
                    heap1.len() == heap2.len(),
                    "Heaps have different lengths after merge: {} vs {}",
                    heap1.len(),
                    heap2.len()
                );
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Binomial vs Fibonacci Stress Tests
// ============================================================================

/// Stress test: Binomial vs Fibonacci with long operation sequence
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)] // Higher unwind bound for longer sequences
fn verify_binomial_fibonacci_stress() {
    let heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let heap2: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    verify_heaps_consistent(heap1, heap2, 20);
}

// ============================================================================
// Pairing vs TwoThree Stress Tests
// ============================================================================

/// Stress test: Pairing vs TwoThree with long operation sequence
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn verify_pairing_twothree_stress() {
    let heap1: PairingHeap<u32, u32> = PairingHeap::new();
    let heap2: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    verify_heaps_consistent(heap1, heap2, 20);
}

// ============================================================================
// SkewBinomial vs Binomial Stress Tests
// ============================================================================

/// Stress test: SkewBinomial vs Binomial with long operation sequence
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn verify_skew_binomial_binomial_stress() {
    let heap1: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();
    let heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    verify_heaps_consistent(heap1, heap2, 20);
}

// ============================================================================
// All Heaps Together Stress Tests
// ============================================================================

/// Stress test: All heaps together with long operation sequence
///
/// Tests all 8 heap implementations simultaneously to ensure they produce
/// identical results for the same operations.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(200)] // Higher unwind for all 8 heaps
fn verify_all_heaps_stress() {
    // Store heaps as concrete types (Heap is not object-safe)
    let mut binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut brodal: BrodalHeap<u32, u32> = BrodalHeap::new();
    let mut fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let mut pairing: PairingHeap<u32, u32> = PairingHeap::new();
    let mut rank_pairing: RankPairingHeap<u32, u32> = RankPairingHeap::new();
    let mut skew: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();
    let mut strict: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();
    let mut twothree: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    let num_operations = 30;

    for _ in 0..num_operations {
        let op = kani::any::<u8>();
        let op_kind = op % 5;

        match op_kind {
            0 => {
                // Push operation
                let priority = kani::any();
                let item = kani::any();
                binomial.push(priority, item);
                brodal.push(priority, item);
                fibonacci.push(priority, item);
                pairing.push(priority, item);
                rank_pairing.push(priority, item);
                skew.push(priority, item);
                strict.push(priority, item);
                twothree.push(priority, item);

                // Verify all heaps have same length
                let len = binomial.len();
                assert!(
                    brodal.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    brodal.len(),
                    len
                );
                assert!(
                    fibonacci.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    fibonacci.len(),
                    len
                );
                assert!(
                    pairing.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    pairing.len(),
                    len
                );
                assert!(
                    rank_pairing.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    rank_pairing.len(),
                    len
                );
                assert!(
                    skew.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    skew.len(),
                    len
                );
                assert!(
                    strict.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    strict.len(),
                    len
                );
                assert!(
                    twothree.len() == len,
                    "Heaps have different lengths after push: {} vs {}",
                    twothree.len(),
                    len
                );
            }
            1 => {
                // Pop operation
                if !binomial.is_empty() {
                    let binomial_result = binomial.pop();
                    let brodal_result = brodal.pop();
                    let fibonacci_result = fibonacci.pop();
                    let pairing_result = pairing.pop();
                    let rank_pairing_result = rank_pairing.pop();
                    let skew_result = skew.pop();
                    let strict_result = strict.pop();
                    let twothree_result = twothree.pop();

                    // Verify all heaps produce same result
                    assert!(
                        brodal_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        brodal_result,
                        binomial_result
                    );
                    assert!(
                        fibonacci_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        fibonacci_result,
                        binomial_result
                    );
                    assert!(
                        pairing_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        pairing_result,
                        binomial_result
                    );
                    assert!(
                        rank_pairing_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        rank_pairing_result,
                        binomial_result
                    );
                    assert!(
                        skew_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        skew_result,
                        binomial_result
                    );
                    assert!(
                        strict_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        strict_result,
                        binomial_result
                    );
                    assert!(
                        twothree_result == binomial_result,
                        "Heaps produced different results on pop: {:?} vs {:?}",
                        twothree_result,
                        binomial_result
                    );

                    // Verify all heaps have same length
                    let len = binomial.len();
                    assert!(
                        brodal.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        brodal.len(),
                        len
                    );
                    assert!(
                        fibonacci.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        fibonacci.len(),
                        len
                    );
                    assert!(
                        pairing.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        pairing.len(),
                        len
                    );
                    assert!(
                        rank_pairing.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        rank_pairing.len(),
                        len
                    );
                    assert!(
                        skew.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        skew.len(),
                        len
                    );
                    assert!(
                        strict.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        strict.len(),
                        len
                    );
                    assert!(
                        twothree.len() == len,
                        "Heaps have different lengths after pop: {} vs {}",
                        twothree.len(),
                        len
                    );
                }
            }
            2 => {
                // Peek operation
                let binomial_result = binomial.peek();
                let brodal_result = brodal.peek();
                let fibonacci_result = fibonacci.peek();
                let pairing_result = pairing.peek();
                let rank_pairing_result = rank_pairing.peek();
                let skew_result = skew.peek();
                let strict_result = strict.peek();
                let twothree_result = twothree.peek();

                // Verify all heaps produce same result
                assert!(
                    brodal_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    brodal_result,
                    binomial_result
                );
                assert!(
                    fibonacci_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    fibonacci_result,
                    binomial_result
                );
                assert!(
                    pairing_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    pairing_result,
                    binomial_result
                );
                assert!(
                    rank_pairing_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    rank_pairing_result,
                    binomial_result
                );
                assert!(
                    skew_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    skew_result,
                    binomial_result
                );
                assert!(
                    strict_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    strict_result,
                    binomial_result
                );
                assert!(
                    twothree_result == binomial_result,
                    "Heaps produced different results on peek: {:?} vs {:?}",
                    twothree_result,
                    binomial_result
                );
            }
            3 => {
                // Merge operation - merge requires Self, so we need to extract concrete types
                // Create new heaps to merge into
                let mut other_binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
                let mut other_brodal: BrodalHeap<u32, u32> = BrodalHeap::new();
                let mut other_fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
                let mut other_pairing: PairingHeap<u32, u32> = PairingHeap::new();
                let mut other_rank_pairing: RankPairingHeap<u32, u32> = RankPairingHeap::new();
                let mut other_skew: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();
                let mut other_strict: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();
                let mut other_twothree: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

                let num_items = kani::any::<u8>() % 5;
                for _ in 0..num_items {
                    let priority = kani::any();
                    let item = kani::any();
                    other_binomial.push(priority, item);
                    other_brodal.push(priority, item);
                    other_fibonacci.push(priority, item);
                    other_pairing.push(priority, item);
                    other_rank_pairing.push(priority, item);
                    other_skew.push(priority, item);
                    other_strict.push(priority, item);
                    other_twothree.push(priority, item);
                }

                // Merge directly on concrete types (no need for downcast)
                binomial.merge(other_binomial);
                brodal.merge(other_brodal);
                fibonacci.merge(other_fibonacci);
                pairing.merge(other_pairing);
                rank_pairing.merge(other_rank_pairing);
                skew.merge(other_skew);
                strict.merge(other_strict);
                twothree.merge(other_twothree);

                // Verify all heaps have same length after merge
                let len = binomial.len();
                assert!(
                    brodal.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    brodal.len(),
                    len
                );
                assert!(
                    fibonacci.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    fibonacci.len(),
                    len
                );
                assert!(
                    pairing.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    pairing.len(),
                    len
                );
                assert!(
                    rank_pairing.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    rank_pairing.len(),
                    len
                );
                assert!(
                    skew.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    skew.len(),
                    len
                );
                assert!(
                    strict.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    strict.len(),
                    len
                );
                assert!(
                    twothree.len() == len,
                    "Heaps have different lengths after merge: {} vs {}",
                    twothree.len(),
                    len
                );
            }
            4 => {
                // Multiple pops in sequence
                let num_pops = kani::any::<u8>() % 3;
                for _ in 0..num_pops {
                    if !binomial.is_empty() {
                        let binomial_result = binomial.pop();
                        let brodal_result = brodal.pop();
                        let fibonacci_result = fibonacci.pop();
                        let pairing_result = pairing.pop();
                        let rank_pairing_result = rank_pairing.pop();
                        let skew_result = skew.pop();
                        let strict_result = strict.pop();
                        let twothree_result = twothree.pop();

                        assert!(
                            brodal_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            brodal_result,
                            binomial_result
                        );
                        assert!(
                            fibonacci_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            fibonacci_result,
                            binomial_result
                        );
                        assert!(
                            pairing_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            pairing_result,
                            binomial_result
                        );
                        assert!(
                            rank_pairing_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            rank_pairing_result,
                            binomial_result
                        );
                        assert!(
                            skew_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            skew_result,
                            binomial_result
                        );
                        assert!(
                            strict_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            strict_result,
                            binomial_result
                        );
                        assert!(
                            twothree_result == binomial_result,
                            "Heaps produced different results on sequential pop: {:?} vs {:?}",
                            twothree_result,
                            binomial_result
                        );
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    // Final verification: all heaps should be empty if we pop everything
    while !binomial.is_empty() {
        let binomial_result = binomial.pop();
        let brodal_result = brodal.pop();
        let fibonacci_result = fibonacci.pop();
        let pairing_result = pairing.pop();
        let rank_pairing_result = rank_pairing.pop();
        let skew_result = skew.pop();
        let strict_result = strict.pop();
        let twothree_result = twothree.pop();

        assert!(
            brodal_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            brodal_result,
            binomial_result
        );
        assert!(
            fibonacci_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            fibonacci_result,
            binomial_result
        );
        assert!(
            pairing_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            pairing_result,
            binomial_result
        );
        assert!(
            rank_pairing_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            rank_pairing_result,
            binomial_result
        );
        assert!(
            skew_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            skew_result,
            binomial_result
        );
        assert!(
            strict_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            strict_result,
            binomial_result
        );
        assert!(
            twothree_result == binomial_result,
            "Heaps produced different results during final pop sequence: {:?} vs {:?}",
            twothree_result,
            binomial_result
        );
    }

    // All heaps should now be empty
    assert!(binomial.is_empty());
    assert!(brodal.is_empty());
    assert!(fibonacci.is_empty());
    assert!(pairing.is_empty());
    assert!(rank_pairing.is_empty());
    assert!(skew.is_empty());
    assert!(strict.is_empty());
    assert!(twothree.is_empty());
}

// ============================================================================
// Very Long Sequence Stress Tests
// ============================================================================

/// Stress test: Very long operation sequence with all heaps
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(250)] // Very high unwind bound for long sequence
fn verify_very_long_sequence_stress() {
    // Store heaps as concrete types (Heap is not object-safe)
    let mut binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut brodal: BrodalHeap<u32, u32> = BrodalHeap::new();
    let mut fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let mut pairing: PairingHeap<u32, u32> = PairingHeap::new();
    let mut rank_pairing: RankPairingHeap<u32, u32> = RankPairingHeap::new();
    let mut skew: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();
    let mut strict: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();
    let mut twothree: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Perform 50 operations
    for _ in 0..50 {
        let op = kani::any::<u8>();
        let op_kind = op % 4; // 4 operation types

        match op_kind {
            0 => {
                // Push
                let priority = kani::any();
                let item = kani::any();
                binomial.push(priority, item);
                brodal.push(priority, item);
                fibonacci.push(priority, item);
                pairing.push(priority, item);
                rank_pairing.push(priority, item);
                skew.push(priority, item);
                strict.push(priority, item);
                twothree.push(priority, item);
            }
            1 => {
                // Pop
                if !binomial.is_empty() {
                    let binomial_result = binomial.pop();
                    let brodal_result = brodal.pop();
                    let fibonacci_result = fibonacci.pop();
                    let pairing_result = pairing.pop();
                    let rank_pairing_result = rank_pairing.pop();
                    let skew_result = skew.pop();
                    let strict_result = strict.pop();
                    let twothree_result = twothree.pop();

                    assert!(
                        brodal_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        brodal_result,
                        binomial_result
                    );
                    assert!(
                        fibonacci_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        fibonacci_result,
                        binomial_result
                    );
                    assert!(
                        pairing_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        pairing_result,
                        binomial_result
                    );
                    assert!(
                        rank_pairing_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        rank_pairing_result,
                        binomial_result
                    );
                    assert!(
                        skew_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        skew_result,
                        binomial_result
                    );
                    assert!(
                        strict_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        strict_result,
                        binomial_result
                    );
                    assert!(
                        twothree_result == binomial_result,
                        "Different results on pop: {:?} vs {:?}",
                        twothree_result,
                        binomial_result
                    );
                }
            }
            2 => {
                // Peek
                let binomial_result = binomial.peek();
                let brodal_result = brodal.peek();
                let fibonacci_result = fibonacci.peek();
                let pairing_result = pairing.peek();
                let rank_pairing_result = rank_pairing.peek();
                let skew_result = skew.peek();
                let strict_result = strict.peek();
                let twothree_result = twothree.peek();

                assert!(
                    brodal_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    brodal_result,
                    binomial_result
                );
                assert!(
                    fibonacci_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    fibonacci_result,
                    binomial_result
                );
                assert!(
                    pairing_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    pairing_result,
                    binomial_result
                );
                assert!(
                    rank_pairing_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    rank_pairing_result,
                    binomial_result
                );
                assert!(
                    skew_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    skew_result,
                    binomial_result
                );
                assert!(
                    strict_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    strict_result,
                    binomial_result
                );
                assert!(
                    twothree_result == binomial_result,
                    "Different results on peek: {:?} vs {:?}",
                    twothree_result,
                    binomial_result
                );
            }
            3 => {
                // Merge - create new heaps and transfer items
                let mut other_binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
                let mut other_brodal: BrodalHeap<u32, u32> = BrodalHeap::new();
                let mut other_fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
                let mut other_pairing: PairingHeap<u32, u32> = PairingHeap::new();
                let mut other_rank_pairing: RankPairingHeap<u32, u32> = RankPairingHeap::new();
                let mut other_skew: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();
                let mut other_strict: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();
                let mut other_twothree: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

                for _ in 0..3 {
                    let priority = kani::any();
                    let item = kani::any();
                    other_binomial.push(priority, item);
                    other_brodal.push(priority, item);
                    other_fibonacci.push(priority, item);
                    other_pairing.push(priority, item);
                    other_rank_pairing.push(priority, item);
                    other_skew.push(priority, item);
                    other_strict.push(priority, item);
                    other_twothree.push(priority, item);
                }

                // Transfer items from other heaps to main heaps
                while !other_binomial.is_empty() {
                    if let Some((priority, item)) = other_binomial.pop() {
                        binomial.push(priority, item);
                    }
                }
                while !other_brodal.is_empty() {
                    if let Some((priority, item)) = other_brodal.pop() {
                        brodal.push(priority, item);
                    }
                }
                while !other_fibonacci.is_empty() {
                    if let Some((priority, item)) = other_fibonacci.pop() {
                        fibonacci.push(priority, item);
                    }
                }
                while !other_pairing.is_empty() {
                    if let Some((priority, item)) = other_pairing.pop() {
                        pairing.push(priority, item);
                    }
                }
                while !other_rank_pairing.is_empty() {
                    if let Some((priority, item)) = other_rank_pairing.pop() {
                        rank_pairing.push(priority, item);
                    }
                }
                while !other_skew.is_empty() {
                    if let Some((priority, item)) = other_skew.pop() {
                        skew.push(priority, item);
                    }
                }
                while !other_strict.is_empty() {
                    if let Some((priority, item)) = other_strict.pop() {
                        strict.push(priority, item);
                    }
                }
                while !other_twothree.is_empty() {
                    if let Some((priority, item)) = other_twothree.pop() {
                        twothree.push(priority, item);
                    }
                }
            }
            _ => unreachable!(),
        }

        // Periodically verify lengths match
        if kani::any::<bool>() {
            let len = binomial.len();
            assert!(
                brodal.len() == len,
                "Lengths don't match: {} vs {}",
                brodal.len(),
                len
            );
            assert!(
                fibonacci.len() == len,
                "Lengths don't match: {} vs {}",
                fibonacci.len(),
                len
            );
            assert!(
                pairing.len() == len,
                "Lengths don't match: {} vs {}",
                pairing.len(),
                len
            );
            assert!(
                rank_pairing.len() == len,
                "Lengths don't match: {} vs {}",
                rank_pairing.len(),
                len
            );
            assert!(
                skew.len() == len,
                "Lengths don't match: {} vs {}",
                skew.len(),
                len
            );
            assert!(
                strict.len() == len,
                "Lengths don't match: {} vs {}",
                strict.len(),
                len
            );
            assert!(
                twothree.len() == len,
                "Lengths don't match: {} vs {}",
                twothree.len(),
                len
            );
        }
    }
}
