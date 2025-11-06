//! Edge case proofs for bug finding
//!
//! These proofs target specific edge cases and potential bug patterns:
//! - Handle validity
//! - Memory safety issues
//! - Empty heap edge cases
//! - Single element edge cases
//! - Repeated operations
//! - Priority edge cases

#[cfg(kani)]
use rust_advanced_heaps::binomial::BinomialHeap;
#[cfg(kani)]
use rust_advanced_heaps::fibonacci::FibonacciHeap;
#[cfg(kani)]
use rust_advanced_heaps::pairing::PairingHeap;
#[cfg(kani)]
use rust_advanced_heaps::Heap;

// ============================================================================
// Handle Validity Proofs
// ============================================================================

/// Proof: Handles remain valid after operations on other elements
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_handle_remains_valid_after_operations_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();
    kani::assume(p1 > p2);
    kani::assume(p1 > p3);

    let handle1 = heap.push(p1, kani::any());
    let _handle2 = heap.push(p2, kani::any());
    let _handle3 = heap.push(p3, kani::any());

    // Operate on other elements (p2 or p3 will be popped, not p1)
    let _ = heap.pop();
    heap.push(kani::any(), kani::any());

    // Original handles should still work
    let new_priority = kani::any();
    kani::assume(new_priority < p1);

    let old_min = heap.find_min().map(|(p, _)| *p);
    if let Some(old_min_val) = old_min {
        if new_priority < old_min_val {
            heap.decrease_key(&handle1, new_priority);
            // After decrease_key, handle1's element should be minimum or less
            let (min_priority, _) = heap
                .find_min()
                .expect("find_min() must return the minimum after pushing elements");
            assert!(*min_priority <= new_priority);
        }
    }
}

/// Proof: Multiple decrease_key operations on same handle
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_multiple_decrease_key_same_handle_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let initial_priority = kani::any();
    let new_priority1 = kani::any();
    let new_priority2 = kani::any();

    kani::assume(new_priority2 < new_priority1);
    kani::assume(new_priority1 < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any()); // Add another element

    // First decrease
    heap.decrease_key(&handle, new_priority1);

    // Second decrease (should still work)
    heap.decrease_key(&handle, new_priority2);

    // Final minimum should be <= new_priority2
    if let Some((&min, _)) = heap.find_min() {
        assert!(min <= new_priority2);
    }
}

// ============================================================================
// Priority Edge Case Proofs
// ============================================================================

/// Proof: Minimum priority works correctly
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_minimum_priority_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Find minimum
    if let Some((&min, _)) = heap.find_min() {
        // Pop should return the same minimum
        if let Some((popped, _)) = heap.pop() {
            assert!(popped == min);
        }
    }
}

/// Proof: Maximum priority works correctly
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_large_priorities_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();

    heap.push(p1, kani::any());
    heap.push(p2, kani::any());
    heap.push(p3, kani::any());

    // Minimum should be min of all priorities
    let expected_min = if p1 < p2 && p1 < p3 {
        p1
    } else if p2 < p3 {
        p2
    } else {
        p3
    };

    if let Some((&min, _)) = heap.find_min() {
        assert!(min == expected_min);
    }
}

// ============================================================================
// Repeated Operations Proofs
// ============================================================================

/// Proof: Repeated push-pop operations maintain invariants
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_repeated_push_pop_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    // Push-pop cycle
    heap.push(kani::any(), kani::any());
    let _ = heap.pop();

    // Another push-pop cycle
    heap.push(kani::any(), kani::any());
    let _ = heap.pop();

    // Heap should be empty
    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
}

/// Proof: Repeated decrease_key operations maintain heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_repeated_decrease_key_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();

    let handle1 = heap.push(p1, kani::any());
    let handle2 = heap.push(p2, kani::any());
    let handle3 = heap.push(p3, kani::any());

    let new_p1 = kani::any();
    let new_p2 = kani::any();
    let new_p3 = kani::any();

    kani::assume(new_p1 < p1);
    kani::assume(new_p2 < p2);
    kani::assume(new_p3 < p3);

    // Decrease all keys
    heap.decrease_key(&handle1, new_p1);
    heap.decrease_key(&handle2, new_p2);
    heap.decrease_key(&handle3, new_p3);

    // Minimum should be min of all new priorities
    let expected_min = if new_p1 < new_p2 && new_p1 < new_p3 {
        new_p1
    } else if new_p2 < new_p3 {
        new_p2
    } else {
        new_p3
    };

    if let Some((&min, _)) = heap.find_min() {
        assert!(min == expected_min);
    }
}

// ============================================================================
// Merge Edge Cases
// ============================================================================

/// Proof: Merging multiple times maintains correctness
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_multiple_merges_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap2: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap3: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap1.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());
    heap3.push(kani::any(), kani::any());

    let len1 = heap1.len();
    let len2 = heap2.len();
    let len3 = heap3.len();

    heap1.merge(heap2);
    assert!(heap1.len() == len1 + len2);

    heap1.merge(heap3);
    assert!(heap1.len() == len1 + len2 + len3);
}

/// Proof: Merging with same minimum maintains correctness
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_same_minimum_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p = kani::any();
    heap1.push(p, kani::any());
    heap2.push(p, kani::any());

    let min_before = heap1.find_min().map(|(p, _)| *p);

    heap1.merge(heap2);

    // After merge, minimum should still be p
    if let Some((&min, _)) = heap1.find_min() {
        if let Some(expected_min) = min_before {
            assert!(min == expected_min);
        }
    }
}

// ============================================================================
// Order Preservation Proofs
// ============================================================================

/// Proof: Elements popped in order (increasing priority)
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_elements_pop_in_order_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();

    heap.push(p1, kani::any());
    heap.push(p2, kani::any());
    heap.push(p3, kani::any());

    // Pop first element
    if let Some((first, _)) = heap.pop() {
        // Pop second element
        if let Some((second, _)) = heap.pop() {
            // Second should be >= first (min-heap property)
            assert!(second >= first);
        }
    }
}

// ============================================================================
// Fibonacci Heap Specific Edge Cases
// ============================================================================

/// Proof: Fibonacci heap consolidation maintains structure
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_fibonacci_consolidation() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    // Insert enough elements to trigger consolidation
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let len_before = heap.len();

    // Pop triggers consolidation
    let _ = heap.pop();

    // Length should decrease by 1
    assert!(heap.len() == len_before - 1);

    // Heap should still be valid
    if !heap.is_empty() {
        assert!(heap.find_min().is_some());
    }
}

// ============================================================================
// Pairing Heap Specific Edge Cases
// ============================================================================

/// Proof: Pairing heap pairing maintains structure
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pairing_heap_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    // Insert elements
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let len_before = heap.len();

    // Pop triggers pairing
    let _ = heap.pop();

    // Length should decrease by 1
    assert!(heap.len() == len_before - 1);

    // Heap should still be valid
    if !heap.is_empty() {
        assert!(heap.find_min().is_some());
    }
}

// ============================================================================
// Length Consistency After Complex Operations
// ============================================================================

/// Proof: Length consistency after complex operation sequence
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_length_after_complex_ops_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    assert!(heap.len() == 0);

    let p_handle1 = kani::any();
    let p_other = kani::any();
    kani::assume(p_other < p_handle1);

    let handle1 = heap.push(p_handle1, kani::any());
    assert!(heap.len() == 1);

    heap.push(p_other, kani::any());
    assert!(heap.len() == 2);

    let _ = heap.pop(); // Will pop p_other, not handle1
    assert!(heap.len() == 1);

    heap.push(kani::any(), kani::any());
    assert!(heap.len() == 2);

    // Decrease key shouldn't change length
    let new_priority = kani::any();
    let current_min = heap.find_min().map(|(p, _)| *p);
    if let Some(min_val) = current_min {
        if new_priority < min_val {
            kani::assume(new_priority < p_handle1);
            heap.decrease_key(&handle1, new_priority);
            assert!(heap.len() == 2); // Length unchanged
        }
    }
}

/// Proof: Empty heap operations are safe
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_empty_heap_operations_all() {
    let mut binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let mut pairing: PairingHeap<u32, u32> = PairingHeap::new();

    // All should be empty
    assert!(binomial.is_empty());
    assert!(fibonacci.is_empty());
    assert!(pairing.is_empty());

    // All should have length 0
    assert!(binomial.len() == 0);
    assert!(fibonacci.len() == 0);
    assert!(pairing.len() == 0);

    // All should return None for find_min
    assert!(binomial.find_min().is_none());
    assert!(fibonacci.find_min().is_none());
    assert!(pairing.find_min().is_none());

    // All should return None for pop
    assert!(binomial.pop().is_none());
    assert!(fibonacci.pop().is_none());
    assert!(pairing.pop().is_none());
}
