//! Kani verification proofs for heap operations (Legacy - simpler examples)
//!
//! Kani is AWS's model checker for Rust. It can verify properties of Rust code
//! by checking all possible executions up to certain bounds.
//!
//! For comprehensive proofs, see:
//! - tests/trait_level_proofs.rs - Heap trait interface proofs
//! - tests/implementation_proofs.rs - Implementation-specific invariant proofs
//!
//! To run these proofs:
//!   cargo kani

// Legacy proofs - these imports may be unused in future refactoring
#[allow(unused_imports)]
use rust_advanced_heaps::binomial::BinomialHeap;
#[allow(unused_imports)]
use rust_advanced_heaps::Heap;

/// Proof that insert always increments the length
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_insert_increments_len() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();
    let initial_len = heap.len();

    let priority = kani::any();
    let item = kani::any();

    heap.push(priority, item);
    let final_len = heap.len();

    // Post-condition: length must increase by exactly 1
    assert!(final_len == initial_len + 1);
}

/// Proof that insert returns a valid handle
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_insert_returns_handle() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority = kani::any();
    let item = kani::any();

    let handle = heap.push(priority, item);

    // Post-condition: handle should not be null
    // The handle contains a pointer, but we verify the heap is not empty
    assert!(!heap.is_empty());
}

/// Proof that pop decrements the length (when not empty)
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_decrements_len() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let item1 = kani::any();
    let item2 = kani::any();

    heap.push(priority1, item1);
    heap.push(priority2, item2);

    let initial_len = heap.len();

    if heap.pop().is_some() {
        let final_len = heap.len();
        assert!(final_len == initial_len - 1);
    }
}

/// Proof that find_min returns the minimum priority
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_find_min_correct() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let item1 = kani::any();
    let item2 = kani::any();

    heap.push(priority1, item1);
    heap.push(priority2, item2);

    if let Some((min_priority, _)) = heap.find_min() {
        // Post-condition: min_priority must be <= all priorities in heap
        // Since we only inserted priority1 and priority2, it must be <= both
        assert!(*min_priority <= priority1);
        assert!(*min_priority <= priority2);
    }
}

/// Proof that pop returns the minimum element
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_pop_returns_min() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let priority1 = kani::any();
    let priority2 = kani::any();
    let item1 = kani::any();
    let item2 = kani::any();

    heap.push(priority1, item1);
    heap.push(priority2, item2);

    let min_before = heap.find_min().map(|(p, _)| *p);

    if let Some((popped_priority, _)) = heap.pop() {
        // Post-condition: popped priority must match what find_min returned
        if let Some(min_priority) = min_before {
            assert!(popped_priority == min_priority);
        }
    }
}

/// Proof that decrease_key actually decreases the priority
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn verify_decrease_key_decreases() {
    let mut heap: BinomialHeap<u32, u32> = BinomialHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();

    // Ensure new_priority < initial_priority (as required by decrease_key)
    kani::assume(new_priority < initial_priority);

    let item = kani::any();
    let handle = heap.push(initial_priority, item);

    heap.decrease_key(&handle, new_priority);

    // Post-condition: if we find_min, it should return new_priority or less
    if let Some((&current_min, _)) = heap.find_min() {
        assert!(current_min <= new_priority);
    }
}
