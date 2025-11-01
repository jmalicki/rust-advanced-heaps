//! Property-based tests using proptest
//!
//! These tests generate random sequences of operations and verify
//! that the heap invariants are always maintained.

use proptest::prelude::*;
use rust_advanced_heaps::Heap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::brodal::BrodalHeap;

use std::collections::HashMap;

/// Test that push and pop maintain heap property
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
        
        // Verify heap property: min should be in inserted
        if !heap.is_empty() {
            if let Some((min_priority, _)) = heap.peek() {
                let min_in_inserted = inserted.iter().min().copied();
                prop_assert_eq!(*min_priority, min_in_inserted.unwrap());
            }
        }
    }
    
    Ok(())
}

/// Test decrease_key maintains heap property
fn test_decrease_key_invariant<H: Heap<i32, i32>>(
    initial: Vec<i32>,
    decreases: Vec<(usize, i32)>
) -> Result<(), TestCaseError> {
    let mut heap = H::new();
    let mut handles = Vec::new();
    let mut priorities: HashMap<usize, i32> = HashMap::new();
    
    // Insert initial values
    for (i, priority) in initial.iter().enumerate() {
        let handle = heap.push(*priority, *priority);
        handles.push(handle);
        priorities.insert(i, *priority);
    }
    
    // Apply decrease_key operations
    for (handle_idx, new_priority) in decreases {
        if handle_idx < handles.len() {
            let old_priority = priorities[&handle_idx];
            if new_priority < old_priority {
                heap.decrease_key(&handles[handle_idx], new_priority);
                priorities.insert(handle_idx, new_priority);
            }
        }
        
        // Verify heap property maintained
        if !heap.is_empty() {
            let min_in_map = priorities.values().min().copied();
            if let Some(expected_min) = min_in_map {
                if let Some((actual_min, _)) = heap.peek() {
                    prop_assert_eq!(*actual_min, expected_min);
                }
            }
        }
    }
    
    Ok(())
}

/// Test that all popped elements are in non-decreasing order
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
            prop_assert!(priority >= last_priority, 
                "Popped priority {} is less than previous {}", priority, last_priority);
            last_priority = priority;
        }
    }
    
    Ok(())
}

/// Test merge maintains heap property
fn test_merge_invariant<H: Heap<i32, i32>>(
    heap1_values: Vec<i32>,
    heap2_values: Vec<i32>
) -> Result<(), TestCaseError> {
    let mut heap1 = H::new();
    let mut heap2 = H::new();
    
    for val in heap1_values {
        heap1.push(val, val);
    }
    
    for val in heap2_values {
        heap2.push(val, val);
    }
    
    let min1 = heap1.peek().map(|(p, _)| *p);
    let min2 = heap2.peek().map(|(p, _)| *p);
    let expected_min = [min1, min2].iter().flatten().min().copied();
    
    heap1.merge(heap2);
    
    if let Some(expected) = expected_min {
        prop_assert_eq!(heap1.peek().map(|(p, _)| *p), Some(expected));
    } else {
        prop_assert!(heap1.is_empty());
    }
    
    Ok(())
}

/// Test len() is always correct
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

// Generate test cases for each heap implementation

proptest! {
    #[test]
    fn test_fibonacci_push_pop_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_push_pop_invariant::<FibonacciHeap<i32, i32>>(ops)?;
    }
    
    #[test]
    fn test_fibonacci_decrease_key_invariant(
        initial in prop::collection::vec(-100i32..100, 1..50),
        decreases in prop::collection::vec((0usize..50, -100i32..100), 0..20)
    ) {
        test_decrease_key_invariant::<FibonacciHeap<i32, i32>>(initial, decreases)?;
    }
    
    #[test]
    fn test_fibonacci_pop_order_invariant(values in prop::collection::vec(-100i32..100, 1..100)) {
        test_pop_order_invariant::<FibonacciHeap<i32, i32>>(values)?;
    }
    
    #[test]
    fn test_fibonacci_merge_invariant(
        heap1 in prop::collection::vec(-100i32..100, 0..50),
        heap2 in prop::collection::vec(-100i32..100, 0..50)
    ) {
        test_merge_invariant::<FibonacciHeap<i32, i32>>(heap1, heap2)?;
    }
    
    #[test]
    fn test_fibonacci_len_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_len_invariant::<FibonacciHeap<i32, i32>>(ops)?;
    }
    
    // Pairing heap tests
    #[test]
    fn test_pairing_push_pop_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_push_pop_invariant::<PairingHeap<i32, i32>>(ops)?;
    }
    
    #[test]
    fn test_pairing_decrease_key_invariant(
        initial in prop::collection::vec(-100i32..100, 1..50),
        decreases in prop::collection::vec((0usize..50, -100i32..100), 0..20)
    ) {
        test_decrease_key_invariant::<PairingHeap<i32, i32>>(initial, decreases)?;
    }
    
    #[test]
    fn test_pairing_pop_order_invariant(values in prop::collection::vec(-100i32..100, 1..100)) {
        test_pop_order_invariant::<PairingHeap<i32, i32>>(values)?;
    }
    
    #[test]
    fn test_pairing_merge_invariant(
        heap1 in prop::collection::vec(-100i32..100, 0..50),
        heap2 in prop::collection::vec(-100i32..100, 0..50)
    ) {
        test_merge_invariant::<PairingHeap<i32, i32>>(heap1, heap2)?;
    }
    
    #[test]
    fn test_pairing_len_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_len_invariant::<PairingHeap<i32, i32>>(ops)?;
    }
    
    // Rank-pairing heap tests
    #[test]
    fn test_rank_pairing_push_pop_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_push_pop_invariant::<RankPairingHeap<i32, i32>>(ops)?;
    }
    
    #[test]
    fn test_rank_pairing_decrease_key_invariant(
        initial in prop::collection::vec(-100i32..100, 1..50),
        decreases in prop::collection::vec((0usize..50, -100i32..100), 0..20)
    ) {
        test_decrease_key_invariant::<RankPairingHeap<i32, i32>>(initial, decreases)?;
    }
    
    #[test]
    fn test_rank_pairing_pop_order_invariant(values in prop::collection::vec(-100i32..100, 1..100)) {
        test_pop_order_invariant::<RankPairingHeap<i32, i32>>(values)?;
    }
    
    #[test]
    fn test_rank_pairing_merge_invariant(
        heap1 in prop::collection::vec(-100i32..100, 0..50),
        heap2 in prop::collection::vec(-100i32..100, 0..50)
    ) {
        test_merge_invariant::<RankPairingHeap<i32, i32>>(heap1, heap2)?;
    }
    
    #[test]
    fn test_rank_pairing_len_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_len_invariant::<RankPairingHeap<i32, i32>>(ops)?;
    }
    
    // Binomial heap tests
    #[test]
    fn test_binomial_push_pop_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_push_pop_invariant::<BinomialHeap<i32, i32>>(ops)?;
    }
    
    #[test]
    fn test_binomial_decrease_key_invariant(
        initial in prop::collection::vec(-100i32..100, 1..50),
        decreases in prop::collection::vec((0usize..50, -100i32..100), 0..20)
    ) {
        test_decrease_key_invariant::<BinomialHeap<i32, i32>>(initial, decreases)?;
    }
    
    #[test]
    fn test_binomial_pop_order_invariant(values in prop::collection::vec(-100i32..100, 1..100)) {
        test_pop_order_invariant::<BinomialHeap<i32, i32>>(values)?;
    }
    
    #[test]
    fn test_binomial_merge_invariant(
        heap1 in prop::collection::vec(-100i32..100, 0..50),
        heap2 in prop::collection::vec(-100i32..100, 0..50)
    ) {
        test_merge_invariant::<BinomialHeap<i32, i32>>(heap1, heap2)?;
    }
    
    #[test]
    fn test_binomial_len_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..100)) {
        test_len_invariant::<BinomialHeap<i32, i32>>(ops)?;
    }
    
    // Brodal heap tests (fewer due to complexity)
    #[test]
    fn test_brodal_push_pop_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..50)) {
        test_push_pop_invariant::<BrodalHeap<i32, i32>>(ops)?;
    }
    
    #[test]
    fn test_brodal_decrease_key_invariant(
        initial in prop::collection::vec(-100i32..100, 1..30),
        decreases in prop::collection::vec((0usize..30, -100i32..100), 0..10)
    ) {
        test_decrease_key_invariant::<BrodalHeap<i32, i32>>(initial, decreases)?;
    }
    
    #[test]
    fn test_brodal_pop_order_invariant(values in prop::collection::vec(-100i32..100, 1..50)) {
        test_pop_order_invariant::<BrodalHeap<i32, i32>>(values)?;
    }
    
    #[test]
    fn test_brodal_merge_invariant(
        heap1 in prop::collection::vec(-100i32..100, 0..30),
        heap2 in prop::collection::vec(-100i32..100, 0..30)
    ) {
        test_merge_invariant::<BrodalHeap<i32, i32>>(heap1, heap2)?;
    }
    
    #[test]
    fn test_brodal_len_invariant(ops in prop::collection::vec((prop::bool::any(), -100i32..100), 0..50)) {
        test_len_invariant::<BrodalHeap<i32, i32>>(ops)?;
    }
}

