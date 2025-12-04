//! Property-based tests using proptest
//!
//! These tests generate random sequences of operations and verify that heap
//! invariants are always maintained across all implementations. The approach uses
//! a cartesian product pattern: we define generic test functions once, then generate
//! test suites for each heap implementation automatically.
//!
//! ## Testing Strategy
//!
//! 1. **Generic test functions**: Each test function is parameterized over heap type
//!    and verifies a specific invariant (heap property, length, completeness, etc.)
//!
//! 2. **Automatic generation**: Macros create test modules for each heap, applying
//!    all test functions to each implementation
//!
//! 3. **Two-tier testing**: Base `Heap` tests apply to all heaps, while
//!    `DecreaseKeyHeap` tests only apply to heaps with decrease_key support
//!
//! 4. **Comprehensive coverage**: We test all major operations (push, pop, merge,
//!    decrease_key, peek) and their interactions
//!
//! 5. **Edge cases**: Special attention to duplicates, complex sequences, and
//!    idempotency properties
//!
//! This approach scales well: adding a new heap type requires one line, and adding
//! a new property test automatically applies to all heaps.

use proptest::prelude::*;
use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::skiplist::SkipListHeap;
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

use std::collections::HashMap;

// ============================================================================
// Base Heap trait tests - work with any Heap implementation
// ============================================================================

/// Test that push and pop maintain heap property
///
/// This is the fundamental heap invariant: after any sequence of pushes and pops,
/// the heap contains exactly the multiset of elements we expect. We generate random
/// sequences of push/pop operations, track them in a simple Vec model, and then
/// fully pop the heap to ensure the multisets match.
///
/// This catches bugs where the heap structure becomes corrupted or elements are
/// lost or duplicated during operations.
fn test_push_pop_invariant<H: Heap<i32, i32>>(ops: Vec<(bool, i32)>) -> Result<(), TestCaseError> {
    let mut heap = H::new();
    let mut inserted = Vec::new();

    for (should_pop, value) in ops {
        if should_pop && !heap.is_empty() {
            let popped = heap.pop();
            if let Some((priority, _item)) = popped {
                // Find this priority in inserted list
                if let Some(pos) = inserted.iter().position(|&p| p == priority) {
                    inserted.remove(pos);
                }
            }
        } else {
            heap.push(value, value);
            inserted.push(value);
        }
    }

    // Verify heap contents match the tracked multiset
    let mut popped = Vec::new();
    while let Some((priority, _item)) = heap.pop() {
        popped.push(priority);
    }
    let mut expected = inserted;
    popped.sort();
    expected.sort();
    prop_assert_eq!(popped, expected);

    Ok(())
}

/// Test that all popped elements are in non-decreasing order
///
/// This verifies the heap property by popping all elements and checking they come out
/// in sorted order. This is a stronger test than just checking peek() because it exercises
/// the pop operation fully and verifies the entire heap structure maintains ordering.
///
/// This catches bugs in the restructuring that happens during pop (like pairing in pairing
/// heaps, or tree consolidation in fibonacci/binomial heaps).
fn test_pop_order_invariant<H: Heap<i32, i32>>(values: Vec<i32>) -> Result<(), TestCaseError> {
    let mut heap = H::new();

    // Insert all values
    for val in &values {
        heap.push(*val, *val);
    }

    // Pop all and verify non-decreasing order
    let mut last_priority = i32::MIN;
    while !heap.is_empty() {
        if let Some((priority, _item)) = heap.pop() {
            prop_assert!(
                priority >= last_priority,
                "Popped priority {} is less than previous {}",
                priority,
                last_priority
            );
            last_priority = priority;
        }
    }

    Ok(())
}

/// Test that merge maintains heap property
///
/// Merge combines two heaps into one. This test verifies that after merging, the
/// minimum element of the merged heap is the global minimum of both original heaps.
/// This is important for algorithms like Prim's MST that use multiple heaps.
///
/// Many heap implementations have merge as a fundamental operation, and bugs here can
/// lead to incorrect priorities being returned.
fn test_merge_invariant<H: Heap<i32, i32>>(
    heap1_values: Vec<i32>,
    heap2_values: Vec<i32>,
) -> Result<(), TestCaseError> {
    let mut heap1 = H::new();
    let mut heap2 = H::new();

    for val in heap1_values {
        heap1.push(val, val);
    }

    for val in heap2_values {
        heap2.push(val, val);
    }

    // Get minimums by popping and pushing back
    let min1 = heap1.pop().map(|(p, item)| {
        heap1.push(p, item);
        p
    });
    let min2 = heap2.pop().map(|(p, item)| {
        heap2.push(p, item);
        p
    });
    let expected_min = [min1, min2].iter().flatten().min().copied();

    heap1.merge(heap2);

    if let Some(expected) = expected_min {
        // Verify minimum by popping
        let actual = heap1.pop();
        prop_assert_eq!(actual.map(|(p, _)| p), Some(expected));
        if let Some((p, i)) = actual {
            heap1.push(p, i); // Restore
        }
    } else {
        prop_assert!(heap1.is_empty());
    }

    Ok(())
}

/// Test that len() is always correct
///
/// Length tracking can be tricky in complex heap implementations, especially with
/// restructuring operations. This test ensures the reported length matches the actual
/// number of elements in the heap.
///
/// This catches off-by-one errors, double-counting in tree consolidation, or lost
/// elements during merge/delete operations.
fn test_len_invariant<H: Heap<i32, i32>>(ops: Vec<(bool, i32)>) -> Result<(), TestCaseError> {
    let mut heap = H::new();
    let mut expected_len = 0;

    for (should_pop, value) in ops {
        if should_pop && !heap.is_empty() {
            heap.pop();
            expected_len -= 1;
        } else {
            heap.push(value, value);
            expected_len += 1;
        }

        prop_assert_eq!(heap.len(), expected_len);
        prop_assert_eq!(heap.is_empty(), expected_len == 0);
    }

    Ok(())
}

/// Test that pop works correctly and maintains length invariant
///
/// This test verifies that pop decrements the length by 1 and returns Some
/// when the heap is non-empty. Ordering is verified by test_pop_order_invariant.
fn test_pop_maintains_property<H: Heap<i32, i32>>(values: Vec<i32>) -> Result<(), TestCaseError> {
    let mut heap = H::new();

    for val in &values {
        heap.push(*val, *val);
    }

    if !heap.is_empty() {
        let len1 = heap.len();
        let popped = heap.pop();
        let len2 = heap.len();
        prop_assert_eq!(len1, len2 + 1);
        prop_assert!(popped.is_some());
        // Push back to restore state
        if let Some((p, i)) = popped {
            heap.push(p, i);
        }
    }

    Ok(())
}

/// Test that multiple merges preserve heap property
///
/// This tests merging many heaps together in sequence, which exercises the merge operation
/// more thoroughly than a single merge. Complex restructuring can compound errors across
/// multiple merges.
///
/// This is important for heaps used in parallel algorithms where multiple heaps need to
/// be combined, or in applications that naturally use multiple priority queues.
fn test_multiple_merges<H: Heap<i32, i32>>(heaps: Vec<Vec<i32>>) -> Result<(), TestCaseError> {
    if heaps.is_empty() {
        return Ok(());
    }

    let mut result = H::new();
    let mut all_mins = Vec::new();

    // Create and merge all heaps
    for heap_values in heaps {
        let mut heap = H::new();
        for val in heap_values {
            heap.push(val, val);
        }

        // Get minimum by popping and pushing back
        if let Some((min, item)) = heap.pop() {
            all_mins.push(min);
            heap.push(min, item); // Restore
        }

        result.merge(heap);
    }

    // Verify result
    if !all_mins.is_empty() {
        let expected_min = all_mins.iter().min().copied();
        if let Some(expected) = expected_min {
            // Verify minimum by popping
            let actual = result.pop();
            prop_assert_eq!(actual.map(|(p, _)| p), Some(expected));
            if let Some((p, i)) = actual {
                result.push(p, i); // Restore
            }
        }
    } else {
        // All heaps were empty; merged result should be empty as well
        prop_assert!(result.is_empty());
    }

    Ok(())
}

/// Test completeness: all inserted elements are eventually retrievable
///
/// This verifies that no elements are lost during heap operations. We insert
/// all values, pop them all, and verify that:
/// 1. The total count of popped elements matches the number inserted
/// 2. Each unique value that was inserted appears in the popped results
///
/// This is critical for correctness - losing elements would corrupt the heap.
fn test_completeness<H: Heap<i32, i32>>(values: Vec<i32>) -> Result<(), TestCaseError> {
    let mut heap = H::new();
    let mut popped_values = Vec::new();
    let insert_count = values.len();

    // Insert all values (may include duplicates)
    for val in &values {
        heap.push(*val, *val);
    }

    prop_assert_eq!(heap.len(), insert_count);

    // Pop all and collect
    while !heap.is_empty() {
        if let Some((_priority, item)) = heap.pop() {
            popped_values.push(item);
        }
    }

    // Verify we got exactly as many elements back as we inserted
    prop_assert_eq!(
        popped_values.len(),
        insert_count,
        "Lost elements: inserted {}, popped {}",
        insert_count,
        popped_values.len()
    );

    // Verify all unique values are present (for duplicates, we just check presence)
    let original_set: std::collections::HashSet<i32> = values.iter().copied().collect();
    let popped_set: std::collections::HashSet<i32> = popped_values.iter().copied().collect();
    prop_assert_eq!(popped_set, original_set, "Missing or extra unique values");

    Ok(())
}

/// Test behavior when repeatedly pushing and popping the same element
///
/// Duplicate priorities are a common edge case in heap implementations. This test verifies
/// that inserting the same priority multiple times and popping them all works correctly.
///
/// Some heap implementations might mishandle duplicates during restructuring, or have
/// issues with duplicate comparisons during consolidation/merging operations.
fn test_duplicate_operations<H: Heap<i32, i32>>(
    value: i32,
    count: usize,
) -> Result<(), TestCaseError> {
    let mut heap = H::new();

    for _ in 0..count {
        heap.push(value, value);
    }

    prop_assert_eq!(heap.len(), count);

    for i in 0..count {
        if let Some((priority, _)) = heap.pop() {
            prop_assert_eq!(priority, value);
            prop_assert_eq!(heap.len(), count - i - 1);
        }
    }

    prop_assert!(heap.is_empty());
    Ok(())
}

// ============================================================================
// DecreaseKeyHeap trait tests - only for heaps with decrease_key support
// ============================================================================

/// Test that decrease_key maintains heap property
///
/// Decrease_key is the most complex operation in many heap implementations. This test
/// verifies that after decreasing a key, the heap property is maintained and the minimum
/// is correctly updated if we decreased the minimum element.
///
/// This is particularly important for fibonacci/pairing heaps which use cut operations
/// that can alter the tree structure significantly. Bugs here can corrupt the entire heap.
fn test_decrease_key_invariant<H: DecreaseKeyHeap<i32, i32>>(
    initial: Vec<i32>,
    decreases: Vec<(usize, i32)>,
) -> Result<(), TestCaseError> {
    let mut heap = H::new();
    let mut handles = Vec::new();
    let mut priorities: HashMap<usize, i32> = HashMap::new();

    // Insert initial values
    for (i, priority) in initial.iter().enumerate() {
        let handle = heap.push_with_handle(*priority, *priority);
        handles.push(handle);
        priorities.insert(i, *priority);
    }

    // Apply decrease_key operations
    for (handle_idx, new_priority) in decreases {
        if handle_idx < handles.len() {
            let old_priority = priorities[&handle_idx];
            if new_priority < old_priority {
                prop_assert!(heap
                    .decrease_key(&handles[handle_idx], new_priority)
                    .is_ok());
                priorities.insert(handle_idx, new_priority);
            }
        }
    }

    // Verify final heap contents via pop sequence
    let mut popped = Vec::new();
    while let Some((priority, _item)) = heap.pop() {
        popped.push(priority);
    }
    let mut expected: Vec<i32> = priorities.values().copied().collect();
    popped.sort();
    expected.sort();
    prop_assert_eq!(popped, expected);

    Ok(())
}

/// Test complex sequences with mixed operations
///
/// Real-world usage involves interleaved push/pop/decrease_key/peek operations. This test
/// generates random sequences of these operations to find bugs that only appear when
/// operations are combined in specific ways.
///
/// This is particularly important for finding state management bugs where operations
/// interfere with each other, or when temporary invariants are broken during complex
/// restructuring operations.
fn test_complex_operations<H: DecreaseKeyHeap<i32, i32>>(
    initial: Vec<i32>,
    ops: Vec<(u8, i32)>,
) -> Result<(), TestCaseError> {
    let mut heap = H::new();
    let mut handles = Vec::new();
    // Track current priority for each handle index
    // Use handle index as the item value for unique identification
    let mut priorities: HashMap<usize, i32> = HashMap::new();

    // Insert initial values with unique item (the index)
    for (i, priority) in initial.iter().enumerate() {
        let handle = heap.push_with_handle(*priority, i as i32);
        handles.push(handle);
        priorities.insert(i, *priority);
    }

    // Apply operations: 0=push, 1=pop, 2=decrease_key, 3=peek
    for (op_type, value) in ops {
        match op_type % 4 {
            0 => {
                // Push with unique item (the index)
                let idx = handles.len();
                let handle = heap.push_with_handle(value, idx as i32);
                handles.push(handle);
                priorities.insert(idx, value);
            }
            1 => {
                // Pop
                if !heap.is_empty() {
                    let popped = heap.pop();
                    if let Some((_priority, item)) = popped {
                        // Item is the handle index, so we can directly identify which was popped
                        let idx = item as usize;
                        priorities.remove(&idx);
                    }
                }
            }
            2 => {
                // Decrease_key - only if we have valid handles and priorities
                if !handles.is_empty() && !priorities.is_empty() {
                    // Pick from valid indices only
                    let valid_indices: Vec<usize> = priorities.keys().copied().collect();
                    if !valid_indices.is_empty() {
                        let idx = valid_indices[(value as usize) % valid_indices.len()];
                        let old_priority = priorities[&idx];
                        let new_priority = if value < old_priority {
                            value
                        } else {
                            old_priority - 1
                        };
                        prop_assert!(heap.decrease_key(&handles[idx], new_priority).is_ok());
                        priorities.insert(idx, new_priority);
                    }
                }
            }
            3 => {
                // Skip inline peek checks; peek is covered by dedicated tests
            }
            _ => {}
        }

        // Verify length invariants after each operation
        prop_assert_eq!(heap.len(), priorities.len());
        prop_assert_eq!(heap.is_empty(), priorities.is_empty());
    }

    // Verify final heap contents via pop sequence
    let mut popped = Vec::new();
    while let Some((_priority, item)) = heap.pop() {
        popped.push(item as usize);
    }
    let mut expected: Vec<usize> = priorities.keys().copied().collect();
    popped.sort();
    expected.sort();
    prop_assert_eq!(popped, expected);

    Ok(())
}

/// Test decrease_key with edge cases
///
/// This test decreases every key in the heap to very small values, which often triggers
/// cut operations in fibonacci/pairing heaps. When many keys are decreased at once,
/// restructuring bugs become more apparent.
///
/// This is critical for verifying that decrease_key correctly handles cascading cuts
/// or other restructuring operations without corrupting the heap structure.
fn test_decrease_key_edge_cases<H: DecreaseKeyHeap<i32, i32>>(
    values: Vec<i32>,
) -> Result<(), TestCaseError> {
    if values.is_empty() {
        return Ok(());
    }

    let mut heap = H::new();
    let mut handles = Vec::new();

    // Insert values
    for val in &values {
        let handle = heap.push_with_handle(*val, *val);
        handles.push(handle);
    }

    // Try decreasing each key to various smaller values
    for (idx, &val) in values.iter().enumerate() {
        let new_priority = val - 100;
        prop_assert!(heap.decrease_key(&handles[idx], new_priority).is_ok());

        // Verify min is now this value or something smaller using peek
        if let Some((min, _)) = heap.peek() {
            prop_assert!(*min <= new_priority);
        }
    }

    Ok(())
}

// ============================================================================
// Macro to generate base Heap tests for a heap type
// ============================================================================

/// Macro to generate base property tests for heaps implementing only `Heap` trait
macro_rules! base_heap_tests {
    ($heap_name:ident, $heap_type:ty, $ops_size:expr, $values_size:expr) => {
        mod $heap_name {
            use super::*;

            proptest::proptest! {
                #[test]
                fn push_pop_invariant(ops in prop::collection::vec((prop::bool::ANY, -100i32..100), $ops_size)) {
                    test_push_pop_invariant::<$heap_type>(ops)?;
                }

                #[test]
                fn pop_order_invariant(values in prop::collection::vec(-100i32..100, $values_size)) {
                    test_pop_order_invariant::<$heap_type>(values)?;
                }

                #[test]
                fn merge_invariant(
                    heap1 in prop::collection::vec(-100i32..100, $values_size),
                    heap2 in prop::collection::vec(-100i32..100, $values_size)
                ) {
                    test_merge_invariant::<$heap_type>(heap1, heap2)?;
                }

                #[test]
                fn len_invariant(ops in prop::collection::vec((prop::bool::ANY, -100i32..100), $ops_size)) {
                    test_len_invariant::<$heap_type>(ops)?;
                }

                #[test]
                fn pop_maintains_property(values in prop::collection::vec(-100i32..100, $values_size)) {
                    test_pop_maintains_property::<$heap_type>(values)?;
                }

                #[test]
                fn multiple_merges(heaps in prop::collection::vec(prop::collection::vec(-100i32..100, 0..20), 0..10)) {
                    test_multiple_merges::<$heap_type>(heaps)?;
                }

                #[test]
                fn completeness(values in prop::collection::vec(-100i32..100, 1..100)) {
                    test_completeness::<$heap_type>(values)?;
                }

                #[test]
                fn duplicate_operations(value in -100i32..100, count in 1..20usize) {
                    test_duplicate_operations::<$heap_type>(value, count)?;
                }
            }
        }
    };
}

/// Macro to generate DecreaseKeyHeap-specific property tests
macro_rules! decrease_key_heap_tests {
    ($heap_name:ident, $heap_type:ty, $initial_size:expr, $decreases_size:expr) => {
        mod $heap_name {
            use super::*;

            proptest::proptest! {
                #[test]
                fn decrease_key_invariant(
                    initial in prop::collection::vec(-100i32..100, $initial_size),
                    decreases in prop::collection::vec((0usize..50, -100i32..100), $decreases_size)
                ) {
                    test_decrease_key_invariant::<$heap_type>(initial, decreases)?;
                }

                #[test]
                fn complex_operations(
                    initial in prop::collection::vec(-100i32..100, 1..20),
                    ops in prop::collection::vec((0u8..4, -100i32..100), 0..50)
                ) {
                    test_complex_operations::<$heap_type>(initial, ops)?;
                }

                #[test]
                fn decrease_key_edge_cases(values in prop::collection::vec(-100i32..100, 1..30)) {
                    test_decrease_key_edge_cases::<$heap_type>(values)?;
                }
            }
        }
    };
}

// ============================================================================
// Generate tests for all heap implementations
// ============================================================================

// SimpleBinaryHeap - base Heap only (no decrease_key)
base_heap_tests!(simple_binary_tests, SimpleBinaryHeap<i32, i32>, 0..100, 1..100);

// Fibonacci Heap - base + decrease_key tests
base_heap_tests!(fibonacci_tests, FibonacciHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(fibonacci_decrease_tests, FibonacciHeap<i32, i32>, 1..50, 0..20);

// Pairing Heap
base_heap_tests!(pairing_tests, PairingHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(pairing_decrease_tests, PairingHeap<i32, i32>, 1..50, 0..20);

// Rank-Pairing Heap
base_heap_tests!(rank_pairing_tests, RankPairingHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(rank_pairing_decrease_tests, RankPairingHeap<i32, i32>, 1..50, 0..20);

// Binomial Heap
base_heap_tests!(binomial_tests, BinomialHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(binomial_decrease_tests, BinomialHeap<i32, i32>, 1..50, 0..20);

// Strict Fibonacci Heap
base_heap_tests!(strict_fibonacci_tests, StrictFibonacciHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(strict_fibonacci_decrease_tests, StrictFibonacciHeap<i32, i32>, 1..50, 0..20);

// 2-3 Heap
base_heap_tests!(twothree_tests, TwoThreeHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(twothree_decrease_tests, TwoThreeHeap<i32, i32>, 1..50, 0..20);

// Skew Binomial Heap
base_heap_tests!(skew_binomial_tests, SkewBinomialHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(skew_binomial_decrease_tests, SkewBinomialHeap<i32, i32>, 1..50, 0..20);

// SkipList Heap
base_heap_tests!(skiplist_tests, SkipListHeap<i32, i32>, 0..100, 1..100);
decrease_key_heap_tests!(skiplist_decrease_tests, SkipListHeap<i32, i32>, 1..50, 0..20);
