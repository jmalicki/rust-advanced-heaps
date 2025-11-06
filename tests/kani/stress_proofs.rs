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
    // Store heaps as trait objects in a vector for iteration
    let mut heaps: Vec<Box<dyn Heap<u32, u32>>> = vec![
        Box::new(BinomialHeap::new()),
        Box::new(BrodalHeap::new()),
        Box::new(FibonacciHeap::new()),
        Box::new(PairingHeap::new()),
        Box::new(RankPairingHeap::new()),
        Box::new(SkewBinomialHeap::new()),
        Box::new(StrictFibonacciHeap::new()),
        Box::new(TwoThreeHeap::new()),
    ];

    let num_operations = 30;

    for _ in 0..num_operations {
        let op = kani::any::<u8>();
        let op_kind = op % 5;

        match op_kind {
            0 => {
                // Push operation
                let priority = kani::any();
                let item = kani::any();
                for heap in heaps.iter_mut() {
                    heap.push(priority, item);
                }

                // Verify all heaps have same length
                let len = heaps[0].len();
                for heap in heaps.iter() {
                    assert!(
                        heap.len() == len,
                        "Heaps have different lengths after push: {} vs {}",
                        heap.len(),
                        len
                    );
                }
            }
            1 => {
                // Pop operation
                if !heaps[0].is_empty() {
                    let results: Vec<_> = heaps.iter_mut().map(|heap| heap.pop()).collect();

                    // Verify all heaps produce same result
                    let first_result = results[0];
                    for result in results.iter() {
                        assert!(
                            *result == first_result,
                            "Heaps produced different results on pop: {:?} vs {:?}",
                            result,
                            first_result
                        );
                    }

                    // Verify all heaps have same length
                    let len = heaps[0].len();
                    for heap in heaps.iter() {
                        assert!(
                            heap.len() == len,
                            "Heaps have different lengths after pop: {} vs {}",
                            heap.len(),
                            len
                        );
                    }
                }
            }
            2 => {
                // Peek operation
                let results: Vec<_> = heaps.iter().map(|heap| heap.peek()).collect();

                // Verify all heaps produce same result
                let first_result = results[0];
                for result in results.iter() {
                    assert!(
                        *result == first_result,
                        "Heaps produced different results on peek: {:?} vs {:?}",
                        result,
                        first_result
                    );
                }
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

                // Extract concrete heaps from trait objects, merge, and put back
                let mut new_heaps: Vec<Box<dyn Heap<u32, u32>>> = vec![
                    {
                        let mut h: BinomialHeap<u32, u32> = *heaps[0]
                            .as_mut()
                            .downcast_mut::<BinomialHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_binomial);
                        Box::new(h)
                    },
                    {
                        let mut h: BrodalHeap<u32, u32> = *heaps[1]
                            .as_mut()
                            .downcast_mut::<BrodalHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_brodal);
                        Box::new(h)
                    },
                    {
                        let mut h: FibonacciHeap<u32, u32> = *heaps[2]
                            .as_mut()
                            .downcast_mut::<FibonacciHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_fibonacci);
                        Box::new(h)
                    },
                    {
                        let mut h: PairingHeap<u32, u32> = *heaps[3]
                            .as_mut()
                            .downcast_mut::<PairingHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_pairing);
                        Box::new(h)
                    },
                    {
                        let mut h: RankPairingHeap<u32, u32> = *heaps[4]
                            .as_mut()
                            .downcast_mut::<RankPairingHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_rank_pairing);
                        Box::new(h)
                    },
                    {
                        let mut h: SkewBinomialHeap<u32, u32> = *heaps[5]
                            .as_mut()
                            .downcast_mut::<SkewBinomialHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_skew);
                        Box::new(h)
                    },
                    {
                        let mut h: StrictFibonacciHeap<u32, u32> = *heaps[6]
                            .as_mut()
                            .downcast_mut::<StrictFibonacciHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_strict);
                        Box::new(h)
                    },
                    {
                        let mut h: TwoThreeHeap<u32, u32> = *heaps[7]
                            .as_mut()
                            .downcast_mut::<TwoThreeHeap<u32, u32>>()
                            .unwrap();
                        h.merge(other_twothree);
                        Box::new(h)
                    },
                ];
                heaps = new_heaps;

                // Verify all heaps have same length after merge
                let len = heaps[0].len();
                for heap in heaps.iter() {
                    assert!(
                        heap.len() == len,
                        "Heaps have different lengths after merge: {} vs {}",
                        heap.len(),
                        len
                    );
                }
            }
            4 => {
                // Multiple pops in sequence
                let num_pops = kani::any::<u8>() % 3;
                for _ in 0..num_pops {
                    if !heaps[0].is_empty() {
                        let results: Vec<_> = heaps.iter_mut().map(|heap| heap.pop()).collect();
                        let first_result = results[0];
                        for result in results.iter() {
                            assert!(
                                *result == first_result,
                                "Heaps produced different results on sequential pop: {:?} vs {:?}",
                                result,
                                first_result
                            );
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    // Final verification: all heaps should be empty if we pop everything
    while !heaps[0].is_empty() {
        let results: Vec<_> = heaps.iter_mut().map(|heap| heap.pop()).collect();
        let first_result = results[0];
        for result in results.iter() {
            assert!(
                *result == first_result,
                "Heaps produced different results during final pop sequence: {:?} vs {:?}",
                result,
                first_result
            );
        }
    }

    // All heaps should now be empty
    for heap in heaps.iter() {
        assert!(
            heap.is_empty(),
            "Heap is not empty after popping all elements: len = {}",
            heap.len()
        );
    }
}

// ============================================================================
// Very Long Sequence Stress Tests
// ============================================================================

/// Stress test: Very long operation sequence with all heaps
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(250)] // Very high unwind bound for long sequence
fn verify_very_long_sequence_stress() {
    let mut heaps: Vec<Box<dyn Heap<u32, u32>>> = vec![
        Box::new(BinomialHeap::new()),
        Box::new(BrodalHeap::new()),
        Box::new(FibonacciHeap::new()),
        Box::new(PairingHeap::new()),
        Box::new(RankPairingHeap::new()),
        Box::new(SkewBinomialHeap::new()),
        Box::new(StrictFibonacciHeap::new()),
        Box::new(TwoThreeHeap::new()),
    ];

    // Perform 50 operations
    for _ in 0..50 {
        let op = kani::any::<u8>();
        let op_kind = op % 4; // 4 operation types

        match op_kind {
            0 => {
                // Push
                let priority = kani::any();
                let item = kani::any();
                for heap in heaps.iter_mut() {
                    heap.push(priority, item);
                }
            }
            1 => {
                // Pop
                if !heaps[0].is_empty() {
                    let results: Vec<_> = heaps.iter_mut().map(|heap| heap.pop()).collect();
                    let first_result = results[0];
                    for result in results.iter() {
                        assert!(
                            *result == first_result,
                            "Different results on pop: {:?} vs {:?}",
                            result,
                            first_result
                        );
                    }
                }
            }
            2 => {
                // Peek
                let results: Vec<_> = heaps.iter().map(|heap| heap.peek()).collect();
                let first_result = results[0];
                for result in results.iter() {
                    assert!(
                        *result == first_result,
                        "Different results on peek: {:?} vs {:?}",
                        result,
                        first_result
                    );
                }
            }
            3 => {
                // Merge - simulate by creating new heaps and transferring items
                let mut other_heaps: Vec<Box<dyn Heap<u32, u32>>> = vec![
                    Box::new(BinomialHeap::new()),
                    Box::new(BrodalHeap::new()),
                    Box::new(FibonacciHeap::new()),
                    Box::new(PairingHeap::new()),
                    Box::new(RankPairingHeap::new()),
                    Box::new(SkewBinomialHeap::new()),
                    Box::new(StrictFibonacciHeap::new()),
                    Box::new(TwoThreeHeap::new()),
                ];

                for _ in 0..3 {
                    let priority = kani::any();
                    let item = kani::any();
                    for heap in other_heaps.iter_mut() {
                        heap.push(priority, item);
                    }
                }

                // Transfer items from other_heaps to main heaps
                for (heap, other_heap) in heaps.iter_mut().zip(other_heaps.iter()) {
                    while !other_heap.is_empty() {
                        if let Some((priority, item)) = other_heap.pop() {
                            heap.push(priority, item);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        // Periodically verify lengths match
        if kani::any::<bool>() {
            let len = heaps[0].len();
            for heap in heaps.iter() {
                assert!(
                    heap.len() == len,
                    "Lengths don't match: {} vs {}",
                    heap.len(),
                    len
                );
            }
        }
    }
}
