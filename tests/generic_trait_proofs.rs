//! Generic trait-level proofs for Heap interface
//!
//! These proofs verify that ALL heap implementations satisfy the Heap trait contract.
//! They use generic functions that work over any Heap<T, P> implementation,
//! eliminating code duplication and ensuring all implementations are covered.

#[cfg(kani)]
use rust_advanced_heaps::Heap;

// ============================================================================
// Generic Proof Helpers (Work for ALL Heap implementations)
// ============================================================================

/// Generic proof: is_empty() is consistent with len() == 0
///
/// For ANY heap implementation, is_empty() should return true iff len() == 0
#[cfg(kani)]
fn verify_is_empty_consistent<H: Heap<u32, u32>>() {
    let heap: H = H::new();
    assert!(heap.is_empty() == (heap.len() == 0));

    let mut heap: H = H::new();
    heap.push(kani::any(), kani::any());
    assert!(!heap.is_empty());
    assert!(heap.len() > 0);
}

/// Generic proof: push always increments length by exactly 1
#[cfg(kani)]
fn verify_push_increments_len<H: Heap<u32, u32>>() {
    let mut heap: H = H::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Generic proof: pop always decrements length by exactly 1 (when not empty)
#[cfg(kani)]
fn verify_pop_decrements_len<H: Heap<u32, u32>>() {
    let mut heap: H = H::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let initial_len = heap.len();

    let popped = heap.pop();
    assert!(
        popped.is_some(),
        "pop() must succeed after pushing elements"
    );
    assert!(heap.len() == initial_len - 1);
}

/// Generic proof: find_min returns the actual minimum priority in the heap
#[cfg(kani)]
fn verify_find_min_correct<H: Heap<u32, u32>>() {
    let mut heap: H = H::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let priority3 = kani::any();

    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());
    heap.push(priority3, kani::any());

    let (min_priority, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    // The minimum must be <= all priorities in the heap
    assert!(*min_priority <= priority1);
    assert!(*min_priority <= priority2);
    assert!(*min_priority <= priority3);

    // The minimum must equal one of the inserted priorities
    assert!(*min_priority == priority1 || *min_priority == priority2 || *min_priority == priority3);
}

/// Generic proof: pop returns the same element that find_min would return
#[cfg(kani)]
fn verify_pop_returns_find_min<H: Heap<u32, u32>>() {
    let mut heap: H = H::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let min_before = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements")
        .0;

    let (popped_priority, _) = heap
        .pop()
        .expect("pop() must succeed after pushing elements");
    assert!(popped_priority == *min_before);
}

/// Generic proof: decrease_key actually decreases the priority
#[cfg(kani)]
fn verify_decrease_key_decreases<H: Heap<u32, u32>>() {
    let mut heap: H = H::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();

    // Precondition: new_priority must be less than initial_priority
    kani::assume(new_priority < initial_priority);

    let item = kani::any();
    let handle = heap.push(initial_priority, item);

    // Add another element to ensure heap is not empty
    heap.push(kani::any(), kani::any());

    heap.decrease_key(&handle, new_priority);

    // After decrease_key, the minimum should be <= new_priority
    let (current_min, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    assert!(*current_min <= new_priority);
}

/// Generic proof: merge combines lengths correctly
#[cfg(kani)]
fn verify_merge_combines_lengths<H: Heap<u32, u32>>() {
    let mut heap1: H = H::new();
    let mut heap2: H = H::new();

    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    let len1 = heap1.len();
    let len2 = heap2.len();

    heap1.merge(heap2);

    assert!(heap1.len() == len1 + len2);
}

/// Generic proof: After popping all elements, heap is empty
#[cfg(kani)]
fn verify_pop_all_makes_empty<H: Heap<u32, u32>>() {
    let mut heap: H = H::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let _ = heap.pop();
    let _ = heap.pop();

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
}

// ============================================================================
// Kani Proof Harnesses (Concrete instances for each heap type)
// ============================================================================

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

// ============================================================================
// BinomialHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_binomial() {
    verify_is_empty_consistent::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_binomial() {
    verify_push_increments_len::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_binomial() {
    verify_pop_decrements_len::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_binomial() {
    verify_find_min_correct::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_binomial() {
    verify_pop_returns_find_min::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_binomial() {
    verify_decrease_key_decreases::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_binomial() {
    verify_merge_combines_lengths::<BinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_binomial() {
    verify_pop_all_makes_empty::<BinomialHeap<u32, u32>>();
}

// ============================================================================
// FibonacciHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_fibonacci() {
    verify_is_empty_consistent::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_fibonacci() {
    verify_push_increments_len::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_fibonacci() {
    verify_pop_decrements_len::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_fibonacci() {
    verify_find_min_correct::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_fibonacci() {
    verify_pop_returns_find_min::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_fibonacci() {
    verify_decrease_key_decreases::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_fibonacci() {
    verify_merge_combines_lengths::<FibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_fibonacci() {
    verify_pop_all_makes_empty::<FibonacciHeap<u32, u32>>();
}

// ============================================================================
// PairingHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_pairing() {
    verify_is_empty_consistent::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_pairing() {
    verify_push_increments_len::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_pairing() {
    verify_pop_decrements_len::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_pairing() {
    verify_find_min_correct::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_pairing() {
    verify_pop_returns_find_min::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_pairing() {
    verify_decrease_key_decreases::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_pairing() {
    verify_merge_combines_lengths::<PairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_pairing() {
    verify_pop_all_makes_empty::<PairingHeap<u32, u32>>();
}

// ============================================================================
// RankPairingHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_rank_pairing() {
    verify_is_empty_consistent::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_rank_pairing() {
    verify_push_increments_len::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_rank_pairing() {
    verify_pop_decrements_len::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_rank_pairing() {
    verify_find_min_correct::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_rank_pairing() {
    verify_pop_returns_find_min::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_rank_pairing() {
    verify_decrease_key_decreases::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_rank_pairing() {
    verify_merge_combines_lengths::<RankPairingHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_rank_pairing() {
    verify_pop_all_makes_empty::<RankPairingHeap<u32, u32>>();
}

// ============================================================================
// SkewBinomialHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_skew_binomial() {
    verify_is_empty_consistent::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_skew_binomial() {
    verify_push_increments_len::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_skew_binomial() {
    verify_pop_decrements_len::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_skew_binomial() {
    verify_find_min_correct::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_skew_binomial() {
    verify_pop_returns_find_min::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_skew_binomial() {
    verify_decrease_key_decreases::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_skew_binomial() {
    verify_merge_combines_lengths::<SkewBinomialHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_skew_binomial() {
    verify_pop_all_makes_empty::<SkewBinomialHeap<u32, u32>>();
}

// ============================================================================
// BrodalHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_brodal() {
    verify_is_empty_consistent::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_brodal() {
    verify_push_increments_len::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_brodal() {
    verify_pop_decrements_len::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_brodal() {
    verify_find_min_correct::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_brodal() {
    verify_pop_returns_find_min::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_brodal() {
    verify_decrease_key_decreases::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_brodal() {
    verify_merge_combines_lengths::<BrodalHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_brodal() {
    verify_pop_all_makes_empty::<BrodalHeap<u32, u32>>();
}

// ============================================================================
// StrictFibonacciHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_strict_fibonacci() {
    verify_is_empty_consistent::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_strict_fibonacci() {
    verify_push_increments_len::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_strict_fibonacci() {
    verify_pop_decrements_len::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_strict_fibonacci() {
    verify_find_min_correct::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_strict_fibonacci() {
    verify_pop_returns_find_min::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_strict_fibonacci() {
    verify_decrease_key_decreases::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_strict_fibonacci() {
    verify_merge_combines_lengths::<StrictFibonacciHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_strict_fibonacci() {
    verify_pop_all_makes_empty::<StrictFibonacciHeap<u32, u32>>();
}

// ============================================================================
// TwoThreeHeap Proofs
// ============================================================================

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_twothree() {
    verify_is_empty_consistent::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_twothree() {
    verify_push_increments_len::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_twothree() {
    verify_pop_decrements_len::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_twothree() {
    verify_find_min_correct::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_twothree() {
    verify_pop_returns_find_min::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_twothree() {
    verify_decrease_key_decreases::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_twothree() {
    verify_merge_combines_lengths::<TwoThreeHeap<u32, u32>>();
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_twothree() {
    verify_pop_all_makes_empty::<TwoThreeHeap<u32, u32>>();
}
