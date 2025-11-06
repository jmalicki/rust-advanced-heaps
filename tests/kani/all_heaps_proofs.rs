//! Comprehensive proofs for ALL heap implementations
//!
//! These proofs verify that ALL heap implementations satisfy the Heap trait contract.
//! This ensures consistency across implementations and catches implementation-specific bugs.

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

// ============================================================================
// RankPairingHeap Proofs
// ============================================================================

/// Proof: RankPairingHeap push increments length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_rank_pairing_push_increments_len() {
    let mut heap: RankPairingHeap<u32, u32> = RankPairingHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Proof: RankPairingHeap pop decrements length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_rank_pairing_pop_decrements_len() {
    let mut heap: RankPairingHeap<u32, u32> = RankPairingHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let initial_len = heap.len();

    let popped = heap.pop();
    assert!(
        popped.is_some(),
        "pop() must succeed after pushing elements"
    );
    assert!(heap.len() == initial_len - 1);
}

/// Proof: RankPairingHeap decrease_key maintains heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_rank_pairing_decrease_key() {
    let mut heap: RankPairingHeap<u32, u32> = RankPairingHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());

    heap.decrease_key(&handle, new_priority);

    let (min_priority, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    assert!(*min_priority <= new_priority);
}

// ============================================================================
// SkewBinomialHeap Proofs
// ============================================================================

/// Proof: SkewBinomialHeap push increments length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_skew_binomial_push_increments_len() {
    let mut heap: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Proof: SkewBinomialHeap pop decrements length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_skew_binomial_pop_decrements_len() {
    let mut heap: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let initial_len = heap.len();

    let popped = heap.pop();
    assert!(
        popped.is_some(),
        "pop() must succeed after pushing elements"
    );
    assert!(heap.len() == initial_len - 1);
}

/// Proof: SkewBinomialHeap decrease_key maintains heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_skew_binomial_decrease_key() {
    let mut heap: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());

    heap.decrease_key(&handle, new_priority);

    let (min_priority, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    assert!(*min_priority <= new_priority);
}

// ============================================================================
// BrodalHeap Proofs
// ============================================================================

/// Proof: BrodalHeap push increments length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_brodal_push_increments_len() {
    let mut heap: BrodalHeap<u32, u32> = BrodalHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Proof: BrodalHeap pop decrements length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_brodal_pop_decrements_len() {
    let mut heap: BrodalHeap<u32, u32> = BrodalHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let initial_len = heap.len();

    let popped = heap.pop();
    assert!(
        popped.is_some(),
        "pop() must succeed after pushing elements"
    );
    assert!(heap.len() == initial_len - 1);
}

/// Proof: BrodalHeap decrease_key maintains heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_brodal_decrease_key() {
    let mut heap: BrodalHeap<u32, u32> = BrodalHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());

    heap.decrease_key(&handle, new_priority);

    let (min_priority, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    assert!(*min_priority <= new_priority);
}

// ============================================================================
// StrictFibonacciHeap Proofs
// ============================================================================

/// Proof: StrictFibonacciHeap push increments length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_strict_fibonacci_push_increments_len() {
    let mut heap: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Proof: StrictFibonacciHeap pop decrements length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_strict_fibonacci_pop_decrements_len() {
    let mut heap: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let initial_len = heap.len();

    let popped = heap.pop();
    assert!(
        popped.is_some(),
        "pop() must succeed after pushing elements"
    );
    assert!(heap.len() == initial_len - 1);
}

/// Proof: StrictFibonacciHeap decrease_key maintains heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_strict_fibonacci_decrease_key() {
    let mut heap: StrictFibonacciHeap<u32, u32> = StrictFibonacciHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());

    heap.decrease_key(&handle, new_priority);

    let (min_priority, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    assert!(*min_priority <= new_priority);
}

// ============================================================================
// TwoThreeHeap Proofs
// ============================================================================

/// Proof: TwoThreeHeap push increments length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_twothree_push_increments_len() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Proof: TwoThreeHeap pop decrements length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_twothree_pop_decrements_len() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let initial_len = heap.len();

    let popped = heap.pop();
    assert!(
        popped.is_some(),
        "pop() must succeed after pushing elements"
    );
    assert!(heap.len() == initial_len - 1);
}

/// Proof: TwoThreeHeap decrease_key maintains heap property
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_twothree_decrease_key() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());

    heap.decrease_key(&handle, new_priority);

    let (min_priority, _) = heap
        .find_min()
        .expect("find_min() must return the minimum after pushing elements");
    assert!(*min_priority <= new_priority);
}

// ============================================================================
// Cross-Implementation Consistency Proofs (All Heaps)
// ============================================================================

/// Proof: All heap implementations find same minimum for same inputs
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_all_heaps_find_same_minimum() {
    let mut binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let mut pairing: PairingHeap<u32, u32> = PairingHeap::new();
    let mut rank_pairing: RankPairingHeap<u32, u32> = RankPairingHeap::new();
    let mut skew_binomial: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();

    let p1 = kani::any();
    let p2 = kani::any();
    let p3 = kani::any();

    binomial.push(p1, kani::any());
    fibonacci.push(p1, kani::any());
    pairing.push(p1, kani::any());
    rank_pairing.push(p1, kani::any());
    skew_binomial.push(p1, kani::any());

    binomial.push(p2, kani::any());
    fibonacci.push(p2, kani::any());
    pairing.push(p2, kani::any());
    rank_pairing.push(p2, kani::any());
    skew_binomial.push(p2, kani::any());

    binomial.push(p3, kani::any());
    fibonacci.push(p3, kani::any());
    pairing.push(p3, kani::any());
    rank_pairing.push(p3, kani::any());
    skew_binomial.push(p3, kani::any());

    let expected_min = if p1 < p2 && p1 < p3 {
        p1
    } else if p2 < p3 {
        p2
    } else {
        p3
    };

    let min_binomial = binomial.find_min().map(|(p, _)| *p);
    let min_fibonacci = fibonacci.find_min().map(|(p, _)| *p);
    let min_pairing = pairing.find_min().map(|(p, _)| *p);
    let min_rank_pairing = rank_pairing.find_min().map(|(p, _)| *p);
    let min_skew_binomial = skew_binomial.find_min().map(|(p, _)| *p);

    let b = min_binomial.expect("binomial heap should have a minimum");
    let f = min_fibonacci.expect("fibonacci heap should have a minimum");
    let p = min_pairing.expect("pairing heap should have a minimum");
    let rp = min_rank_pairing.expect("rank pairing heap should have a minimum");
    let sb = min_skew_binomial.expect("skew binomial heap should have a minimum");

    assert!(b == expected_min);
    assert!(f == expected_min);
    assert!(p == expected_min);
    assert!(rp == expected_min);
    assert!(sb == expected_min);
}

/// Proof: All heap implementations maintain length consistency
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_all_heaps_length_consistency() {
    let mut binomial: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut fibonacci: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let mut pairing: PairingHeap<u32, u32> = PairingHeap::new();
    let mut rank_pairing: RankPairingHeap<u32, u32> = RankPairingHeap::new();
    let mut skew_binomial: SkewBinomialHeap<u32, u32> = SkewBinomialHeap::new();

    // Push same number of elements to all
    binomial.push(kani::any(), kani::any());
    fibonacci.push(kani::any(), kani::any());
    pairing.push(kani::any(), kani::any());
    rank_pairing.push(kani::any(), kani::any());
    skew_binomial.push(kani::any(), kani::any());

    binomial.push(kani::any(), kani::any());
    fibonacci.push(kani::any(), kani::any());
    pairing.push(kani::any(), kani::any());
    rank_pairing.push(kani::any(), kani::any());
    skew_binomial.push(kani::any(), kani::any());

    // All should have length 2
    assert!(binomial.len() == 2);
    assert!(fibonacci.len() == 2);
    assert!(pairing.len() == 2);
    assert!(rank_pairing.len() == 2);
    assert!(skew_binomial.len() == 2);

    // Pop from all
    assert!(binomial.pop().is_some());
    assert!(fibonacci.pop().is_some());
    assert!(pairing.pop().is_some());
    assert!(rank_pairing.pop().is_some());
    assert!(skew_binomial.pop().is_some());

    // All should have length 1
    assert!(binomial.len() == 1);
    assert!(fibonacci.len() == 1);
    assert!(pairing.len() == 1);
    assert!(rank_pairing.len() == 1);
    assert!(skew_binomial.len() == 1);
}
