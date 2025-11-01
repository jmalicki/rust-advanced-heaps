//! Trait-level proofs for Heap interface
//!
//! These proofs verify that ALL heap implementations satisfy the Heap trait contract.
//! They are generic and test the interface, not implementation details.

#[cfg(kani)]
use rust_advanced_heaps::binomial::BinomialHeap;
#[cfg(kani)]
use rust_advanced_heaps::fibonacci::FibonacciHeap;
#[cfg(kani)]
use rust_advanced_heaps::pairing::PairingHeap;
#[cfg(kani)]
use rust_advanced_heaps::Heap;

/// Proof: is_empty() is consistent with len() == 0
///
/// For ALL heap implementations, is_empty() should return true iff len() == 0
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_binomial() {
    let heap: BinomialHeap<u32, u32> = BinomialHeap::new();
    assert!(heap.is_empty() == (heap.len() == 0));

    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();
    heap.push(kani::any(), kani::any());
    assert!(!heap.is_empty());
    assert!(heap.len() > 0);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_fibonacci() {
    let heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    assert!(heap.is_empty() == (heap.len() == 0));

    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    heap.push(kani::any(), kani::any());
    assert!(!heap.is_empty());
    assert!(heap.len() > 0);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_is_empty_consistent_pairing() {
    let heap: PairingHeap<u32, u32> = PairingHeap::new();
    assert!(heap.is_empty() == (heap.len() == 0));

    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();
    heap.push(kani::any(), kani::any());
    assert!(!heap.is_empty());
    assert!(heap.len() > 0);
}

/// Proof: push always increments length by exactly 1
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_push_increments_len_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();
    let initial_len = heap.len();

    heap.push(kani::any(), kani::any());

    assert!(heap.len() == initial_len + 1);
}

/// Proof: pop always decrements length by exactly 1 (when heap is not empty)
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let initial_len = heap.len();

    if let Some(_) = heap.pop() {
        assert!(heap.len() == initial_len - 1);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let initial_len = heap.len();

    if let Some(_) = heap.pop() {
        assert!(heap.len() == initial_len - 1);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let initial_len = heap.len();

    if let Some(_) = heap.pop() {
        assert!(heap.len() == initial_len - 1);
    }
}

/// Proof: find_min returns the actual minimum priority in the heap
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let priority3 = kani::any();

    let handle1 = heap.push(priority1, kani::any());
    let handle2 = heap.push(priority2, kani::any());
    let handle3 = heap.push(priority3, kani::any());

    if let Some((&min_priority, _)) = heap.find_min() {
        // The minimum must be <= all priorities in the heap
        assert!(*heap.find_min().unwrap().0 <= priority1);
        assert!(*heap.find_min().unwrap().0 <= priority2);
        assert!(*heap.find_min().unwrap().0 <= priority3);

        // The minimum must equal one of the inserted priorities
        assert!(
            *heap.find_min().unwrap().0 == priority1
                || *heap.find_min().unwrap().0 == priority2
                || *heap.find_min().unwrap().0 == priority3
        );
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let priority3 = kani::any();

    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());
    heap.push(priority3, kani::any());

    if let Some((&min_priority, _)) = heap.find_min() {
        assert!(*heap.find_min().unwrap().0 <= priority1);
        assert!(*heap.find_min().unwrap().0 <= priority2);
        assert!(*heap.find_min().unwrap().0 <= priority3);

        assert!(
            *heap.find_min().unwrap().0 == priority1
                || *heap.find_min().unwrap().0 == priority2
                || *heap.find_min().unwrap().0 == priority3
        );
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let priority3 = kani::any();

    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());
    heap.push(priority3, kani::any());

    if let Some((&min_priority, _)) = heap.find_min() {
        assert!(*heap.find_min().unwrap().0 <= priority1);
        assert!(*heap.find_min().unwrap().0 <= priority2);
        assert!(*heap.find_min().unwrap().0 <= priority3);

        assert!(
            *heap.find_min().unwrap().0 == priority1
                || *heap.find_min().unwrap().0 == priority2
                || *heap.find_min().unwrap().0 == priority3
        );
    }
}

/// Proof: pop returns the same element that find_min would return
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let min_before = heap.find_min().map(|(p, _)| *p);

    if let Some((popped_priority, _)) = heap.pop() {
        if let Some(expected_min) = min_before {
            assert!(popped_priority == expected_min);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let min_before = heap.find_min().map(|(p, _)| *p);

    if let Some((popped_priority, _)) = heap.pop() {
        if let Some(expected_min) = min_before {
            assert!(popped_priority == expected_min);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_find_min_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    heap.push(priority1, kani::any());
    heap.push(priority2, kani::any());

    let min_before = heap.find_min().map(|(p, _)| *p);

    if let Some((popped_priority, _)) = heap.pop() {
        if let Some(expected_min) = min_before {
            assert!(popped_priority == expected_min);
        }
    }
}

/// Proof: decrease_key actually decreases the priority
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();

    // Precondition: new_priority must be less than initial_priority
    kani::assume(new_priority < initial_priority);

    let item = kani::any();
    let handle = heap.push(initial_priority, item);

    // Store the minimum before decrease_key
    let min_before = heap.find_min().map(|(p, _)| *p);

    heap.decrease_key(&handle, new_priority);

    // After decrease_key, the minimum should be <= new_priority
    if let Some((&current_min, _)) = heap.find_min() {
        assert!(current_min <= new_priority);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();

    kani::assume(new_priority < initial_priority);

    let item = kani::any();
    let handle = heap.push(initial_priority, item);

    heap.decrease_key(&handle, new_priority);

    if let Some((&current_min, _)) = heap.find_min() {
        assert!(current_min <= new_priority);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();

    kani::assume(new_priority < initial_priority);

    let item = kani::any();
    let handle = heap.push(initial_priority, item);

    heap.decrease_key(&handle, new_priority);

    if let Some((&current_min, _)) = heap.find_min() {
        assert!(current_min <= new_priority);
    }
}

/// Proof: merge combines lengths correctly
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_binomial() {
    let mut heap1: BinomialHeap<u32, u32> = BinomialHeap::new();
    let mut heap2: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    let len1 = heap1.len();
    let len2 = heap2.len();

    heap1.merge(heap2);

    assert!(heap1.len() == len1 + len2);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_fibonacci() {
    let mut heap1: FibonacciHeap<u32, u32> = FibonacciHeap::new();
    let mut heap2: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    let len1 = heap1.len();
    let len2 = heap2.len();

    heap1.merge(heap2);

    assert!(heap1.len() == len1 + len2);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_merge_combines_lengths_pairing() {
    let mut heap1: PairingHeap<u32, u32> = PairingHeap::new();
    let mut heap2: PairingHeap<u32, u32> = PairingHeap::new();

    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    let len1 = heap1.len();
    let len2 = heap2.len();

    heap1.merge(heap2);

    assert!(heap1.len() == len1 + len2);
}

/// Proof: After popping all elements, heap is empty
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_binomial() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let _ = heap.pop();
    let _ = heap.pop();

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_fibonacci() {
    let mut heap: FibonacciHeap<u32, u32> = FibonacciHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let _ = heap.pop();
    let _ = heap.pop();

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_all_makes_empty_pairing() {
    let mut heap: PairingHeap<u32, u32> = PairingHeap::new();

    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    let _ = heap.pop();
    let _ = heap.pop();

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(heap.find_min().is_none());
}
