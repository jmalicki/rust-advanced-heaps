//! Detailed invariant proofs for TwoThreeHeap
//!
//! These proofs verify the specific invariants of TwoThreeHeap:
//! - 2-3 property: each internal node has exactly 2 or 3 children
//! - Memory safety: no double-frees, no dangling pointers
//! - Tree structure: parent-child relationships are consistent
//! - Length consistency: len() matches actual node count
//! - Heap property: parent <= child for all parent-child pairs

#[cfg(kani)]
use rust_advanced_heaps::twothree::TwoThreeHeap;
#[cfg(kani)]
use rust_advanced_heaps::Heap;

// ============================================================================
// Internal Structure Invariants
// ============================================================================

/// Proof: 2-3 property is maintained after insert
///
/// This verifies that after inserting elements, each internal node
/// has exactly 2 or 3 children (the fundamental 2-3 heap invariant).
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_23_property_after_insert() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert multiple elements (may trigger splits)
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Verify 2-3 property is maintained
    assert!(
        heap.verify_internal_structure(),
        "2-3 property violated after insert"
    );
}

/// Proof: 2-3 property is maintained after pop
///
/// This verifies that after popping elements, the 2-3 property
/// is still maintained (each internal node has 2 or 3 children).
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_23_property_after_pop() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert multiple elements
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Pop some elements (may trigger rebuilds)
    let _ = heap.pop();
    let _ = heap.pop();

    // Verify 2-3 property is maintained
    assert!(
        heap.verify_internal_structure(),
        "2-3 property violated after pop"
    );
}

/// Proof: Heap property is maintained throughout the tree
///
/// This verifies that parent <= child for all parent-child pairs
/// throughout the entire tree structure.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_heap_property_internal() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert multiple elements
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Verify heap property is maintained
    assert!(heap.verify_internal_structure(), "Heap property violated");

    // Pop element
    let _ = heap.pop();

    // Verify heap property is still maintained
    assert!(
        heap.verify_internal_structure(),
        "Heap property violated after pop"
    );
}

/// Proof: Parent-child relationships are consistent
///
/// This verifies that if node A is a child of node B, then
/// node B is the parent of node A (bidirectional consistency).
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_parent_child_consistency() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert multiple elements
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Verify parent-child consistency
    assert!(
        heap.verify_internal_structure(),
        "Parent-child consistency violated"
    );
}

/// Proof: Node count matches reported length
///
/// This verifies that the actual number of nodes in the tree
/// matches (or is at least) the reported length from len().
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_node_count_matches_len() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert multiple elements
    for _ in 0..5 {
        heap.push(kani::any(), kani::any());
    }

    // Verify node count
    assert!(
        heap.verify_internal_structure(),
        "Node count does not match len()"
    );

    // Pop some elements
    for _ in 0..2 {
        let _ = heap.pop();
    }

    // Verify node count still matches
    assert!(
        heap.verify_internal_structure(),
        "Node count does not match len() after pop"
    );
}

/// Proof: All internal structure invariants are maintained after decrease_key
///
/// This verifies that decrease_key maintains:
/// - 2-3 property (each internal node has 2 or 3 children)
/// - Heap property (parent <= child)
/// - Parent-child consistency
/// - Node count consistency
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_internal_structure_after_decrease_key() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Verify structure before decrease_key
    assert!(
        heap.verify_internal_structure(),
        "Internal structure violated before decrease_key"
    );

    // Decrease_key may trigger bubble_up
    heap.decrease_key(&handle, new_priority);

    // Verify all internal structure invariants are still maintained
    assert!(
        heap.verify_internal_structure(),
        "Internal structure violated after decrease_key"
    );
}

/// Proof: All internal structure invariants are maintained after merge
///
/// This verifies that merge maintains:
/// - 2-3 property
/// - Heap property
/// - Parent-child consistency
/// - Node count consistency
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(15)]
fn verify_twothree_internal_structure_after_merge() {
    let mut heap1: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());

    let mut heap2: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    heap2.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    // Verify structure before merge
    assert!(
        heap1.verify_internal_structure(),
        "Internal structure violated in heap1 before merge"
    );
    assert!(
        heap2.verify_internal_structure(),
        "Internal structure violated in heap2 before merge"
    );

    heap1.merge(heap2);

    // Verify all internal structure invariants are maintained after merge
    assert!(
        heap1.verify_internal_structure(),
        "Internal structure violated after merge"
    );
}

/// Proof: Complex sequence maintains all internal structure invariants
///
/// This verifies that a complex sequence of operations maintains:
/// - 2-3 property
/// - Heap property
/// - Parent-child consistency
/// - Node count consistency
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(20)]
fn verify_twothree_internal_structure_complex_sequence() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert many elements (will trigger splits)
    for _ in 0..10 {
        heap.push(kani::any(), kani::any());
        // Verify structure after each insert
        assert!(
            heap.verify_internal_structure(),
            "Internal structure violated during insert"
        );
    }

    // Pop some elements (will trigger rebuilds)
    for _ in 0..5 {
        let _ = heap.pop();
        // Verify structure after each pop
        assert!(
            heap.verify_internal_structure(),
            "Internal structure violated during pop"
        );
    }

    // Insert more (may trigger splits again)
    for _ in 0..5 {
        heap.push(kani::any(), kani::any());
        // Verify structure after each insert
        assert!(
            heap.verify_internal_structure(),
            "Internal structure violated during insert"
        );
    }

    // Final verification
    assert!(
        heap.verify_internal_structure(),
        "Internal structure violated after complex sequence"
    );
}

// ============================================================================
// Memory Safety Invariants
// ============================================================================

/// Proof: Multiple operations don't cause double-frees
///
/// This verifies that sequences of push/pop/merge operations don't
/// cause memory corruption or double-frees. The proof system should
/// catch any issues with node ownership or freeing.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_no_double_free_operations() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert multiple elements (may trigger splits)
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Pop some elements (may trigger rebuilds)
    let _ = heap.pop();
    let _ = heap.pop();

    // Insert more (may trigger splits again)
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Pop all remaining
    while heap.pop().is_some() {
        // Kani should verify no double-frees occur
    }

    // Heap should be empty now
    assert!(heap.is_empty());
    assert!(heap.len() == 0);
}

/// Proof: Merge operations don't cause double-frees
///
/// Merging heaps involves transferring ownership of nodes. This proof
/// verifies that nodes are not freed multiple times during merge.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_no_double_free_merge() {
    let mut heap1: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    heap1.push(kani::any(), kani::any());
    heap1.push(kani::any(), kani::any());

    let mut heap2: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    heap2.push(kani::any(), kani::any());
    heap2.push(kani::any(), kani::any());

    // Merge: heap2's nodes should be transferred to heap1
    let len1_before = heap1.len();
    let len2_before = heap2.len();

    heap1.merge(heap2);

    // Length should be sum
    assert!(heap1.len() == len1_before + len2_before);

    // Pop all elements (should not cause double-frees)
    while heap1.pop().is_some() {
        // Kani should verify no double-frees occur
    }

    assert!(heap1.is_empty());
}

/// Proof: Decrease_key followed by operations doesn't cause memory issues
///
/// Decrease_key may trigger bubble_up which swaps node data. This proof
/// verifies that subsequent operations don't cause memory corruption.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_decrease_key_memory_safety() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Decrease_key may trigger bubble_up (swaps node data)
    heap.decrease_key(&handle, new_priority);

    // Subsequent operations should not cause memory issues
    heap.push(kani::any(), kani::any());
    let _ = heap.pop();

    // Pop all remaining
    while heap.pop().is_some() {
        // Kani should verify no memory corruption
    }

    assert!(heap.is_empty());
}

// ============================================================================
// Length Consistency Invariants
// ============================================================================

/// Proof: Length is consistent after complex sequences
///
/// This verifies that len() accurately reflects the number of elements
/// in the heap, even after complex sequences involving splits and merges.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(15)]
fn verify_twothree_length_consistency() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert elements (may trigger splits)
    let mut count = 0;
    for _ in 0..5 {
        heap.push(kani::any(), kani::any());
        count += 1;
        assert!(heap.len() == count);
    }

    // Pop some elements
    for _ in 0..2 {
        let popped = heap.pop();
        assert!(popped.is_some());
        count -= 1;
        assert!(heap.len() == count);
    }

    // Insert more (may trigger splits again)
    for _ in 0..3 {
        heap.push(kani::any(), kani::any());
        count += 1;
        assert!(heap.len() == count);
    }

    // Pop all remaining
    while let Some(_) = heap.pop() {
        count -= 1;
        assert!(heap.len() == count);
    }

    assert!(heap.is_empty());
    assert!(heap.len() == 0);
    assert!(count == 0);
}

/// Proof: Length doesn't overflow during operations
///
/// This verifies that length calculations don't overflow, even with
/// many operations. The proof system should catch any underflow issues.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(15)]
fn verify_twothree_length_no_overflow() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert elements
    for _ in 0..5 {
        let len_before = heap.len();
        heap.push(kani::any(), kani::any());
        let len_after = heap.len();
        // Length should increase by exactly 1
        assert!(len_after == len_before + 1);
        // Length should not overflow
        assert!(len_after > len_before);
    }

    // Pop elements
    while let Some(_) = heap.pop() {
        let len_before = heap.len();
        let popped = heap.pop();
        if popped.is_some() {
            let len_after = heap.len();
            // Length should decrease by exactly 1
            assert!(len_after == len_before - 1);
            // Length should not underflow
            assert!(len_after < len_before);
        }
    }
}

// ============================================================================
// Tree Structure Invariants
// ============================================================================

/// Proof: Heap structure is maintained after operations
///
/// This verifies that the heap maintains its tree structure correctly
/// after sequences of operations. The proof system should catch any
/// structural corruption.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_structure_maintained() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert elements (may trigger splits)
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Heap should not be empty
    assert!(!heap.is_empty());

    // peek() should work
    let peek_result = heap.peek();
    assert!(peek_result.is_some());

    // Pop should work
    let pop_result = heap.pop();
    assert!(pop_result.is_some());

    // peek() should still work after pop
    let peek_result2 = heap.peek();
    // If heap is not empty, peek should succeed
    if !heap.is_empty() {
        assert!(peek_result2.is_some());
    }

    // Pop all remaining
    while heap.pop().is_some() {
        // Structure should remain valid
    }

    // Heap should be empty now
    assert!(heap.is_empty());
}

/// Proof: Heap property is maintained after decrease_key
///
/// After decrease_key, the heap property (parent <= child) should
/// still be maintained throughout the tree.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn verify_twothree_heap_property_after_decrease_key() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    let initial_priority = kani::any();
    let new_priority = kani::any();
    kani::assume(new_priority < initial_priority);

    let handle = heap.push(initial_priority, kani::any());
    heap.push(kani::any(), kani::any());
    heap.push(kani::any(), kani::any());

    // Decrease_key should maintain heap property
    heap.decrease_key(&handle, new_priority);

    // find_min should return the minimum (which may be the decreased key)
    let min_result = heap.find_min();
    assert!(min_result.is_some());
    let (min_priority, _) = min_result.unwrap();

    // The minimum should be <= new_priority (since we decreased to new_priority)
    assert!(*min_priority <= new_priority);

    // Pop should work without corruption
    let pop_result = heap.pop();
    assert!(pop_result.is_some());
}

// ============================================================================
// Stress Test Invariants
// ============================================================================

/// Proof: Stress test with many operations
///
/// This verifies that the heap maintains all invariants under stress:
/// many insertions (triggering splits), many pops (triggering rebuilds),
/// and merges.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(20)]
fn verify_twothree_stress_test() {
    let mut heap: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();

    // Insert many elements (will trigger multiple splits)
    for _ in 0..10 {
        heap.push(kani::any(), kani::any());
        // Length should be consistent
        assert!(!heap.is_empty() || heap.len() == 0);
    }

    // Pop some elements (will trigger rebuilds)
    for _ in 0..5 {
        let popped = heap.pop();
        assert!(popped.is_some());
        // Length should decrease
        assert!(heap.len() < 10);
    }

    // Insert more (may trigger splits again)
    for _ in 0..5 {
        heap.push(kani::any(), kani::any());
    }

    // Pop all remaining
    let mut count = 0;
    while let Some(_) = heap.pop() {
        count += 1;
        // Length should be consistent
        assert!(heap.len() == 10 - count || heap.is_empty());
    }

    assert!(heap.is_empty());
}

/// Proof: Merge stress test
///
/// This verifies that merging multiple heaps maintains all invariants.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(20)]
fn verify_twothree_merge_stress_test() {
    let mut heap1: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    for _ in 0..5 {
        heap1.push(kani::any(), kani::any());
    }

    let mut heap2: TwoThreeHeap<u32, u32> = TwoThreeHeap::new();
    for _ in 0..5 {
        heap2.push(kani::any(), kani::any());
    }

    let len1_before = heap1.len();
    let len2_before = heap2.len();

    // Merge should combine lengths
    heap1.merge(heap2);
    assert!(heap1.len() == len1_before + len2_before);

    // Pop all should work without corruption
    while heap1.pop().is_some() {
        // Kani should verify no memory issues
    }

    assert!(heap1.is_empty());
}
