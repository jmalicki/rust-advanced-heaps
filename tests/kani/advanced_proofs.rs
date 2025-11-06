//! Advanced proofs for complex scenarios and edge cases
//!
//! These proofs verify complex operation sequences and edge cases that
//! are likely to reveal bugs.

#[cfg(kani)]
use rust_advanced_heaps::binomial::BinomialHeap;
#[cfg(kani)]
use rust_advanced_heaps::fibonacci::FibonacciHeap;
#[cfg(kani)]
use rust_advanced_heaps::pairing::PairingHeap;
#[cfg(kani)]
use rust_advanced_heaps::Heap;

// ============================================================================
// Edge Case Proofs
// ============================================================================

/// Proof: Empty heap operations are safe
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_empty_heap_operations_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    // Empty heap should return None for pop and find_min
    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
    assert!(heap.pop().is_none());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_empty_heap_operations_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
    assert!(heap.pop().is_none());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_empty_heap_operations_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
    assert!(heap.pop().is_none());
}

/// Proof: Single element heap works correctly
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_single_element_heap_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority = kani::any();
    let item = kani::any();
    let handle = heap.push(priority, item);

    assert!(!heap.is_empty());
    assert!(heap.len() == 1);

    // find_min should return the only element
    if let Some((&p, &i)) = heap.find_min() {
        assert!(p == priority);
        assert!(i == item);
    }

    // pop should return the only element
    if let Some((p, i)) = heap.pop() {
        assert!(p == priority);
        assert!(i == item);
        assert!(heap.is_empty());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_single_element_heap_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    let priority = kani::any();
    let item = kani::any();
    let handle = heap.push(priority, item);

    assert!(!heap.is_empty());
    assert!(heap.len() == 1);

    if let Some((&p, &i)) = heap.find_min() {
        assert!(p == priority);
        assert!(i == item);
    }

    if let Some((p, i)) = heap.pop() {
        assert!(p == priority);
        assert!(i == item);
        assert!(heap.is_empty());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn verify_single_element_heap_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    let priority = kani::any();
    let item = kani::any();
    let handle = heap.push(priority, item);

    assert!(!heap.is_empty());
    assert!(heap.len() == 1);

    if let Some((&p, &i)) = heap.find_min() {
        assert!(p == priority);
        assert!(i == item);
    }

    if let Some((p, i)) = heap.pop() {
        assert!(p == priority);
        assert!(i == item);
        assert!(heap.is_empty());
    }
}

// ============================================================================
// Sequence of Operations Proofs
// ============================================================================

/// Proof: Multiple push-pop sequences maintain correctness
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_pop_sequence_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();

    heap.push(p1, kani::any());
    heap.push(p2, kani::any());

    // Pop should return minimum
    let min1 = if p1 < p2 { p1 } else { p2 };
    if let Some((popped, _)) = heap.pop() {
        assert!(popped == min1);
        assert!(heap.len() == 1);
    }

    heap.push(p3, kani::any());

    // After pushing new element, min should be correct
    let min2 = if min1 == p1 {
        if p2 < p3 {
            p2
        } else {
            p3
        }
    } else {
        if p1 < p3 {
            p1
        } else {
            p3
        }
    };
    if let Some((&current_min, _)) = heap.find_min() {
        assert!(current_min == min2);
    }
}

/// Proof: Decrease key then pop returns decreased priority
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_then_pop_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let other_priority = kani::any();

    let handle = heap.push(initial_priority, kani::any());
    heap.push(other_priority, kani::any());

    // Decrease key
    heap.decrease_key(&handle, new_priority);

    // Pop should return the minimum, which might be new_priority
    if let Some((popped_priority, _)) = heap.pop() {
        assert!(popped_priority <= new_priority);
        assert!(popped_priority <= other_priority);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_then_pop_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let other_priority = kani::any();

    let handle = heap.push(initial_priority, kani::any());
    heap.push(other_priority, kani::any());

    heap.decrease_key(&handle, new_priority);

    if let Some((popped_priority, _)) = heap.pop() {
        assert!(popped_priority <= new_priority);
        assert!(popped_priority <= other_priority);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_then_pop_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let other_priority = kani::any();

    let handle = heap.push(initial_priority, kani::any());
    heap.push(other_priority, kani::any());

    heap.decrease_key(&handle, new_priority);

    if let Some((popped_priority, _)) = heap.pop() {
        assert!(popped_priority <= new_priority);
        assert!(popped_priority <= other_priority);
    }
}

// ============================================================================
// Merge Edge Case Proofs
// ============================================================================

/// Proof: Merging with empty heap doesn't break anything
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_with_empty_heap_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap1.push(kani::any(), kani::any());
    let len_before = heap1.len();

    heap1.merge(heap2);

    // Merging with empty heap shouldn't change length
    assert!(heap1.len() == len_before);
    assert!(!heap1.is_empty());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_empty_with_heap_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap2.push(kani::any(), kani::any());
    let len2 = heap2.len();

    heap1.merge(heap2);

    // Merging empty heap with non-empty should transfer all elements
    assert!(heap1.len() == len2);
    assert!(!heap1.is_empty());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_empty_with_empty_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap1.merge(heap2);

    // Merging two empty heaps should result in empty heap
    assert!(heap1.is_empty());
    assert!(heap1.len() == 0);
}

// ============================================================================
// Complex Operation Sequence Proofs
// ============================================================================

/// Proof: Push, decrease_key, pop sequence maintains invariants
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_decrease_pop_sequence_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();
    let new_p1 = kani::any();
    kani::assume(new_p1 < p1);

    let h1 = heap.push(p1, kani::any());
    heap.push(p2, kani::any());
    heap.push(p3, kani::any());

    let initial_len = heap.len();

    // Decrease key
    heap.decrease_key(&h1, new_p1);

    // Length shouldn't change
    assert!(heap.len() == initial_len);

    // Minimum should be <= new_p1
    if let Some((&min, _)) = heap.find_min() {
        assert!(min <= new_p1);
    }

    // Pop should work correctly
    let _ = heap.pop();
    assert!(heap.len() == initial_len - 1);
}

/// Proof: Multiple decrease_key operations maintain heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_multiple_decrease_keys_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();
    let new_p1 = kani::any();
    let new_p2 = kani::any();
    kani::assume(new_p1 < p1);
    kani::assume(new_p2 < p2);

    let h1 = heap.push(p1, kani::any());
    let h2 = heap.push(p2, kani::any());
    heap.push(p3, kani::any());

    heap.decrease_key(&h1, new_p1);
    heap.decrease_key(&h2, new_p2);

    // Minimum should be <= min(new_p1, new_p2)
    let min_new = if new_p1 < new_p2 { new_p1 } else { new_p2 };
    if let Some((&min, _)) = heap.find_min() {
        assert!(min <= min_new);
        assert!(min <= p3);
    }
}

// ============================================================================
// Heap Property Maintenance Proofs
// ============================================================================

/// Proof: After pop, heap property is maintained
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_heap_property_after_pop_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();

    heap.push(p1, kani::any());
    heap.push(p2, kani::any());
    heap.push(p3, kani::any());

    let min_before = if p1 < p2 && p1 < p3 {
        p1
    } else if p2 < p3 {
        p2
    } else {
        p3
    };

    let _ = heap.pop();

    // After pop, find_min should return >= min_before (or None if empty)
    if let Some((&min_after, _)) = heap.find_min() {
        // The new minimum should be >= what we popped
        assert!(min_after >= min_before);
    }
}

/// Proof: Heap maintains minimum correctly after multiple operations
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_minimum_maintenance_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priorities = [kani::any(), kani::any(), kani::any(), kani::any()];

    for &p in &priorities {
        heap.push(p, kani::any());
    }

    // Find the minimum of all priorities
    let expected_min = priorities.iter().min().copied();

    if let Some(min_val) = expected_min {
        if let Some((&actual_min, _)) = heap.find_min() {
            assert!(actual_min == min_val);
        }
    }
}

// ============================================================================
// Length Consistency Proofs
// ============================================================================

/// Proof: Length is consistent across multiple operations
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_length_consistency_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    assert!(heap.len() == 0);
    assert!(heap.is_empty());

    heap.push(kani::any(), kani::any());
    assert!(heap.len() == 1);
    assert!(!heap.is_empty());

    heap.push(kani::any(), kani::any());
    assert!(heap.len() == 2);

    let _ = heap.pop();
    assert!(heap.len() == 1);
    assert!(!heap.is_empty());

    let _ = heap.pop();
    assert!(heap.len() == 0);
    assert!(heap.is_empty());
}

/// Proof: Merge doesn't corrupt length accounting
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_length_accounting_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    let len1 = heap1.len();
    let len2 = heap2.len();
    let expected_total = len1 + len2;

    heap1.merge(heap2);

    assert!(heap1.len() == expected_total);
    assert!(heap1.len() == 3);
}
