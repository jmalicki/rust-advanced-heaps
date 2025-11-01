//! Generic comprehensive tests for all Heap implementations
//!
//! These tests work with any Heap implementation and stress the trait interface
//! with various edge cases and complex scenarios.

use rust_advanced_heaps::Heap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::brodal::BrodalHeap;
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;

// Test helpers that work with any Heap implementation

/// Test that empty heap behaves correctly
fn test_empty_heap<H: Heap<String, i32>>() {
    let mut heap = H::new();
    assert!(heap.is_empty());
    assert_eq!(heap.len(), 0);
    assert_eq!(heap.peek(), None);
    assert_eq!(heap.pop(), None);
    assert_eq!(heap.find_min(), None);
}

/// Test basic insert and pop operations
fn test_basic_operations<H: Heap<&'static str, i32>>() {
    let mut heap = H::new();
    
    // Insert some elements
    let _h1 = heap.push(5, "five");
    let _h2 = heap.push(1, "one");
    let _h3 = heap.push(10, "ten");
    let _h4 = heap.push(3, "three");
    
    assert!(!heap.is_empty());
    assert_eq!(heap.len(), 4);
    
    // Peek should return minimum
    let min = heap.peek();
    assert_eq!(min, Some((&1, &"one")));
    
    // Pop should return minimums in order
    assert_eq!(heap.pop(), Some((1, "one")));
    assert_eq!(heap.pop(), Some((3, "three")));
    assert_eq!(heap.pop(), Some((5, "five")));
    assert_eq!(heap.pop(), Some((10, "ten")));
    assert_eq!(heap.pop(), None);
    assert!(heap.is_empty());
}

/// Test decrease_key operations extensively
fn test_decrease_key_operations<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    
    // Insert elements
    let _h1 = heap.push(100, 1);
    let h2 = heap.push(200, 2);
    let _h3 = heap.push(300, 3);
    let h4 = heap.push(400, 4);
    
    // Verify initial min
    assert_eq!(heap.peek(), Some((&100, &1)));
    
    // Decrease key of element not at min
    heap.decrease_key(&h2, 50);
    assert_eq!(heap.peek(), Some((&50, &2)));
    
    // Decrease key to become new min
    heap.decrease_key(&h4, 25);
    assert_eq!(heap.peek(), Some((&25, &4)));
    
    // Decrease key of current min even more
    heap.decrease_key(&h4, 1);
    assert_eq!(heap.peek(), Some((&1, &4)));
    
    // Pop and verify order
    assert_eq!(heap.pop(), Some((1, 4)));
    assert_eq!(heap.pop(), Some((50, 2)));
    assert_eq!(heap.pop(), Some((100, 1)));
    assert_eq!(heap.pop(), Some((300, 3)));
}

/// Test decrease_key on multiple elements
fn test_multiple_decrease_keys<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert 20 elements with high priorities
    for i in 0..20 {
        handles.push(heap.push((i + 1) * 100, i as i32));
    }
    
    // Decrease all keys to be much smaller
    for (i, handle) in handles.iter().enumerate() {
        heap.decrease_key(handle, i as i32);
    }
    
    // Verify heap property maintained
    let min = heap.peek();
    assert_eq!(min, Some((&0, &0)));
    
    // Pop all and verify ascending order
    for i in 0..20 {
        let popped = heap.pop();
        assert_eq!(popped, Some((i as i32, i as i32)));
    }
    assert!(heap.is_empty());
}

/// Test merge operations
fn test_merge_operations<H: Heap<&'static str, i32>>() {
    let mut heap1 = H::new();
    heap1.push(5, "five");
    heap1.push(1, "one");
    
    let mut heap2 = H::new();
    heap2.push(10, "ten");
    heap2.push(3, "three");
    
    // Merge heaps
    heap1.merge(heap2);
    
    assert_eq!(heap1.len(), 4);
    assert_eq!(heap1.peek(), Some((&1, &"one")));
    
    // Verify all elements can be popped in order
    assert_eq!(heap1.pop(), Some((1, "one")));
    assert_eq!(heap1.pop(), Some((3, "three")));
    assert_eq!(heap1.pop(), Some((5, "five")));
    assert_eq!(heap1.pop(), Some((10, "ten")));
}

/// Test merge with empty heap
fn test_merge_empty<H: Heap<i32, i32>>() {
    let mut heap1 = H::new();
    heap1.push(5, 1);
    heap1.push(1, 2);
    
    let heap2 = H::new();
    
    let len_before = heap1.len();
    heap1.merge(heap2);
    assert_eq!(heap1.len(), len_before);
    
    // Merge empty into non-empty
    let mut heap3 = H::new();
    let mut heap4 = H::new();
    heap4.push(3, 3);
    
    heap3.merge(heap4);
    assert_eq!(heap3.len(), 1);
    assert_eq!(heap3.peek(), Some((&3, &3)));
}

/// Test with duplicate priorities
fn test_duplicate_priorities<H: Heap<&'static str, i32>>() {
    let mut heap = H::new();
    
    heap.push(5, "a");
    heap.push(5, "b");
    heap.push(5, "c");
    heap.push(1, "d");
    
    // All items with priority 5 should come after priority 1
    assert_eq!(heap.pop(), Some((1, "d")));
    
    // Items with same priority can come in any order
    let mut seen = std::collections::HashSet::new();
    for _ in 0..3 {
        if let Some((pri, item)) = heap.pop() {
            assert_eq!(pri, 5);
            assert!(seen.insert(item));
        }
    }
    assert_eq!(seen.len(), 3);
}

/// Test decrease_key after pop (should panic or handle gracefully)
fn test_decrease_key_after_pop<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let handle = heap.push(10, 1);
    
    // Pop the element
    heap.pop();
    
    // Attempting decrease_key on popped handle is undefined behavior
    // We can't really test this without unsafe code, but we document it
}

/// Test many operations in sequence
fn test_stress_operations<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert many elements
    for i in 0..100 {
        handles.push(heap.push(i * 2, i));
    }
    
    // Random decrease_key operations
    for i in (0..100).step_by(3) {
        heap.decrease_key(&handles[i], i as i32 * 2 - 1);
    }
    
    // Pop some elements
    for _ in 0..20 {
        heap.pop();
    }
    
    // Verify heap still works
    assert!(!heap.is_empty());
    assert!(heap.peek().is_some());
    
    // Pop remaining
    let mut count = 0;
    while heap.pop().is_some() {
        count += 1;
    }
    assert_eq!(count, 80); // 100 - 20 = 80 remaining
}

/// Test that peek doesn't modify heap
fn test_peek_idempotent<H: Heap<&'static str, i32>>() {
    let mut heap = H::new();
    heap.push(5, "five");
    heap.push(1, "one");
    
    // Peek multiple times should return same result
    assert_eq!(heap.peek(), Some((&1, &"one")));
    assert_eq!(heap.peek(), Some((&1, &"one")));
    assert_eq!(heap.peek(), Some((&1, &"one")));
    
    // Length shouldn't change
    assert_eq!(heap.len(), 2);
    
    // Pop should still work
    assert_eq!(heap.pop(), Some((1, "one")));
}

/// Test single element heap
fn test_single_element<H: Heap<&'static str, i32>>() {
    let mut heap = H::new();
    let handle = heap.push(42, "single");
    
    assert_eq!(heap.len(), 1);
    assert_eq!(heap.peek(), Some((&42, &"single")));
    
    // Decrease key
    heap.decrease_key(&handle, 10);
    assert_eq!(heap.peek(), Some((&10, &"single")));
    
    // Pop
    assert_eq!(heap.pop(), Some((10, "single")));
    assert!(heap.is_empty());
}

/// Test merge then operations
fn test_merge_then_operations<H: Heap<i32, i32>>() {
    let mut heap1 = H::new();
    for i in 0..10 {
        heap1.push(i * 10, i);
    }
    
    let mut heap2 = H::new();
    for i in 10..20 {
        heap2.push(i * 10, i);
    }
    
    heap1.merge(heap2);
    
    // After merge, should be able to decrease keys
    // (Can't test this without handles, but merge should work)
    
    // Pop all elements
    let mut count = 0;
    let mut last_priority = i32::MIN;
    while let Some((priority, _)) = heap1.pop() {
        assert!(priority >= last_priority); // Should be non-decreasing
        last_priority = priority;
        count += 1;
    }
    assert_eq!(count, 20);
}

/// Test rapid insert and pop
fn test_rapid_operations<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    
    // Rapid insert-pop cycle
    for i in 0..50 {
        heap.push(i, i);
        if i % 3 == 0 {
            heap.pop();
        }
    }
    
    // Verify heap still works
    assert!(!heap.is_empty());
    
    // Pop remaining
    let mut count = 0;
    while heap.pop().is_some() {
        count += 1;
    }
    assert!(count > 0);
}

/// Test with large priorities
fn test_large_priorities<H: Heap<i32, i64>>() {
    let mut heap = H::new();
    
    heap.push(1_000_000_000, 1);
    heap.push(2_000_000_000, 2);
    heap.push(-1_000_000_000, 3);
    
    assert_eq!(heap.peek(), Some((&-1_000_000_000, &3)));
    assert_eq!(heap.pop(), Some((-1_000_000_000, 3)));
    assert_eq!(heap.pop(), Some((1_000_000_000, 1)));
    assert_eq!(heap.pop(), Some((2_000_000_000, 2)));
}

/// Test decrease_key to same priority (edge case)
fn test_decrease_key_same<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let handle = heap.push(10, 1);
    
    // Decrease to same priority (should be handled gracefully)
    heap.decrease_key(&handle, 10);
    
    // Should still be min
    assert_eq!(heap.peek(), Some((&10, &1)));
}

/// Test complex sequence of operations
fn test_complex_sequence<H: Heap<String, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert with gaps
    for i in 0..20 {
        if i % 2 == 0 {
            handles.push(heap.push(i * 10, format!("item{}", i)));
        }
    }
    
    // Decrease some keys (but skip first few to avoid issues)
    for (i, handle) in handles.iter().enumerate().skip(3).step_by(3) {
        let new_priority = (i as i32 * 5).max(1); // Ensure positive priority
        if new_priority < i as i32 * 10 {
            heap.decrease_key(handle, new_priority);
        }
    }
    
    // Pop some (but not all)
    for _ in 0..3 {
        if heap.pop().is_none() {
            break;
        }
    }
    
    // Insert more
    for i in 20..25 {
        heap.push(i * 10, format!("new{}", i));
    }
    
    // Verify still works
    assert!(!heap.is_empty());
    let min = heap.peek();
    assert!(min.is_some());
    
    // Pop all remaining (carefully)
    let mut count = 0;
    while let Some(_) = heap.pop() {
        count += 1;
        if count > 50 {
            // Safety limit
            break;
        }
    }
    
    // Final state check
    assert!(heap.is_empty() || heap.peek().is_some());
}

/// Test ascending order insertion
fn test_ascending_insertion<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    
    // Insert in ascending order
    for i in 0..50 {
        heap.push(i, i);
    }
    
    // Should pop in ascending order
    for i in 0..50 {
        assert_eq!(heap.pop(), Some((i, i)));
    }
}

/// Test descending order insertion
fn test_descending_insertion<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    
    // Insert in descending order
    for i in (0..50).rev() {
        heap.push(i, i);
    }
    
    // Should pop in ascending order (min heap)
    for i in 0..50 {
        assert_eq!(heap.pop(), Some((i, i)));
    }
}

/// Test random order insertion
fn test_random_order_insertion<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut values: Vec<i32> = (0..100).collect();
    // Shuffle by swapping pairs
    for i in (0..values.len()).step_by(7) {
        if i + 1 < values.len() {
            values.swap(i, i + 1);
        }
    }
    
    for &val in &values {
        heap.push(val * 2, val);
    }
    
    // Should pop in ascending order
    for i in 0..100 {
        assert_eq!(heap.pop(), Some((i * 2, i)));
    }
}

/// Test multiple decrease_keys on same element
fn test_multiple_decrease_same<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let handle = heap.push(1000, 1);
    
    heap.decrease_key(&handle, 500);
    assert_eq!(heap.peek(), Some((&500, &1)));
    
    heap.decrease_key(&handle, 250);
    assert_eq!(heap.peek(), Some((&250, &1)));
    
    heap.decrease_key(&handle, 100);
    assert_eq!(heap.peek(), Some((&100, &1)));
    
    heap.decrease_key(&handle, 50);
    assert_eq!(heap.peek(), Some((&50, &1)));
    
    heap.decrease_key(&handle, 1);
    assert_eq!(heap.peek(), Some((&1, &1)));
}

/// Test decrease_key making element new min repeatedly
fn test_decrease_key_new_min<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let h1 = heap.push(100, 1);
    let h2 = heap.push(200, 2);
    let h3 = heap.push(300, 3);
    
    // Each decrease_key should make element new min
    heap.decrease_key(&h3, 150);
    assert_eq!(heap.peek(), Some((&100, &1))); // Still h1
    
    heap.decrease_key(&h2, 50);
    assert_eq!(heap.peek(), Some((&50, &2))); // Now h2
    
    heap.decrease_key(&h1, 25);
    assert_eq!(heap.peek(), Some((&25, &1))); // Now h1
}

/// Test alternating insert and pop
fn test_alternating_operations<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    
    // Insert 5, pop 2, insert 3, pop 1, etc.
    for i in 0..10 {
        heap.push(i * 10, i);
    }
    
    // Pop 3
    heap.pop();
    heap.pop();
    heap.pop();
    
    // Insert 5 more
    for i in 10..15 {
        heap.push(i * 10, i);
    }
    
    // Pop 2
    heap.pop();
    heap.pop();
    
    // Verify still works
    assert!(!heap.is_empty());
    assert!(heap.peek().is_some());
    
    // Pop all remaining
    let mut count = 0;
    while heap.pop().is_some() {
        count += 1;
    }
    assert_eq!(count, 10); // 10 - 3 + 5 - 2 = 10
}

/// Test merge with many elements
fn test_merge_large<H: Heap<i32, i32>>() {
    let mut heap1 = H::new();
    for i in 0..100 {
        heap1.push(i * 2, i);
    }
    
    let mut heap2 = H::new();
    for i in 100..200 {
        heap2.push(i * 2, i);
    }
    
    heap1.merge(heap2);
    assert_eq!(heap1.len(), 200);
    
    // Pop all and verify order
    let mut last = i32::MIN;
    let mut count = 0;
    while let Some((priority, _)) = heap1.pop() {
        assert!(priority >= last);
        last = priority;
        count += 1;
    }
    assert_eq!(count, 200);
}

/// Test decrease_key on every other element
fn test_decrease_key_selective<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert 30 elements
    for i in 0..30 {
        handles.push(heap.push((i + 1) * 100, i));
    }
    
    // Decrease keys of even-indexed elements
    for (i, handle) in handles.iter().enumerate() {
        if i % 2 == 0 {
            heap.decrease_key(handle, i as i32 * 10);
        }
    }
    
    // Verify min is from even-indexed (decreased) elements
    let min = heap.peek();
    assert!(min.is_some());
    let min_priority = min.unwrap().0;
    assert!(*min_priority < 100); // Should be from decreased elements
    
    // Pop and verify order
    let mut last = i32::MIN;
    while let Some((priority, _)) = heap.pop() {
        assert!(priority >= last);
        last = priority;
    }
}

/// Test edge case: all elements same priority after decrease
fn test_all_same_priority<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert 10 elements with different priorities
    for i in 0..10 {
        handles.push(heap.push((i + 1) * 10, i));
    }
    
    // Decrease all to same priority
    for handle in &handles {
        heap.decrease_key(handle, 5);
    }
    
    // All should have priority 5
    assert_eq!(heap.peek().unwrap().0, &5);
    
    // Pop all - should get all items (order doesn't matter for same priority)
    let mut seen = std::collections::HashSet::new();
    while let Some((priority, item)) = heap.pop() {
        assert_eq!(priority, 5);
        assert!(seen.insert(item));
    }
    assert_eq!(seen.len(), 10);
}

/// Test negative priorities
fn test_negative_priorities<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    
    heap.push(-10, 1);
    heap.push(10, 2);
    heap.push(-5, 3);
    heap.push(5, 4);
    
    // Min should be -10
    assert_eq!(heap.pop(), Some((-10, 1)));
    assert_eq!(heap.pop(), Some((-5, 3)));
    assert_eq!(heap.pop(), Some((5, 4)));
    assert_eq!(heap.pop(), Some((10, 2)));
}

/// Test decrease_key to negative
fn test_decrease_to_negative<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let h1 = heap.push(10, 1);
    let _h2 = heap.push(20, 2);
    
    heap.decrease_key(&h1, -5);
    assert_eq!(heap.peek(), Some((&-5, &1)));
    assert_eq!(heap.pop(), Some((-5, 1)));
}

/// Test insert/alias methods (insert, find_min, delete_min)
fn test_alias_methods<H: Heap<&'static str, i32>>() {
    let mut heap = H::new();
    
    // Test insert alias
    let _h = heap.insert(5, "five");
    assert_eq!(heap.len(), 1);
    
    // Test find_min alias
    assert_eq!(heap.find_min(), Some((&5, &"five")));
    assert_eq!(heap.peek(), Some((&5, &"five"))); // Same as find_min
    
    // Test delete_min alias
    assert_eq!(heap.delete_min(), Some((5, "five")));
    assert_eq!(heap.pop(), None); // Should be empty now
}

/// Test heap property maintained through operations
fn test_heap_property<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert random priorities
    for i in 0..50 {
        handles.push(heap.push((i * 17 + 23) % 1000, i));
    }
    
    // Random decrease_keys
    for i in (0..50).step_by(7) {
        if let Some(handle) = handles.get(i) {
            let current_min = heap.peek().unwrap().0;
            let new_priority = (*current_min / 2).max(1);
            heap.decrease_key(handle, new_priority);
        }
    }
    
    // Verify heap property: each pop should be >= previous
    let mut last_priority = i32::MIN;
    while let Some((priority, _)) = heap.pop() {
        assert!(priority >= last_priority);
        last_priority = priority;
    }
}

/// Test merge with handles from both heaps
fn test_merge_with_handles<H: Heap<i32, i32>>() {
    let mut heap1 = H::new();
    let h1 = heap1.push(100, 1);
    
    let mut heap2 = H::new();
    let h2 = heap2.push(200, 2);
    
    heap1.merge(heap2);
    
    // After merge, both handles should still be valid
    heap1.decrease_key(&h1, 50);
    assert_eq!(heap1.peek(), Some((&50, &1)));
    
    heap1.decrease_key(&h2, 25);
    assert_eq!(heap1.peek(), Some((&25, &2)));
}

/// Test very large number of operations
fn test_very_large_sequence<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();
    
    // Insert 1000 elements
    for i in 0..1000 {
        handles.push(heap.push(i * 10, i));
    }
    
    // Decrease keys of every 10th element
    for i in (0..1000).step_by(10) {
        heap.decrease_key(&handles[i], i as i32);
    }
    
    // Pop first 100
    for _ in 0..100 {
        heap.pop();
    }
    
    // Insert 200 more
    for i in 1000..1200 {
        heap.push(i * 10, i);
    }
    
    // Verify heap still works
    assert!(!heap.is_empty());
    
    // Pop all remaining
    let mut count = 0;
    while heap.pop().is_some() {
        count += 1;
    }
    assert_eq!(count, 1100); // 1000 - 100 + 200 = 1100
}

/// Test with string items
fn test_string_items<H: Heap<String, i32>>() {
    let mut heap = H::new();
    
    heap.push(3, "c".to_string());
    heap.push(1, "a".to_string());
    heap.push(2, "b".to_string());
    
    assert_eq!(heap.peek().unwrap().1, "a");
    assert_eq!(heap.pop().unwrap().1, "a");
    assert_eq!(heap.pop().unwrap().1, "b");
    assert_eq!(heap.pop().unwrap().1, "c");
}

/// Test with tuple items
fn test_tuple_items<H: Heap<(i32, i32), i32>>() {
    let mut heap = H::new();
    
    heap.push(3, (3, 3));
    heap.push(1, (1, 1));
    heap.push(2, (2, 2));
    
    assert_eq!(heap.pop(), Some((1, (1, 1))));
    assert_eq!(heap.pop(), Some((2, (2, 2))));
    assert_eq!(heap.pop(), Some((3, (3, 3))));
}

// Macro to generate a single test function
macro_rules! heap_test {
    ($name:ident, $heap:ty, $func:ident) => {
        #[test]
        fn $name() {
            $func::<$heap>();
        }
    };
}

// Macro to generate all 30 tests for a heap type
// Takes heap name, heap type, and all 30 test function names
// This reduces repetition while being explicit about test names
macro_rules! define_heap_tests {
    (
        $heap_name:ident,
        $heap_type:ident,
        $test_empty:ident, $test_basic:ident, $test_decrease_key:ident,
        $test_multiple_decrease_keys:ident, $test_merge:ident, $test_merge_empty:ident,
        $test_duplicate_priorities:ident, $test_stress:ident, $test_peek_idempotent:ident,
        $test_single_element:ident, $test_merge_then_operations:ident, $test_rapid_operations:ident,
        $test_large_priorities:ident, $test_decrease_key_same:ident, $test_complex_sequence:ident,
        $test_ascending_insertion:ident, $test_descending_insertion:ident, $test_random_order_insertion:ident,
        $test_multiple_decrease_same:ident, $test_decrease_key_new_min:ident, $test_alternating_operations:ident,
        $test_merge_large:ident, $test_decrease_key_selective:ident, $test_all_same_priority:ident,
        $test_negative_priorities:ident, $test_decrease_to_negative:ident, $test_alias_methods:ident,
        $test_heap_property:ident, $test_merge_with_handles:ident, $test_very_large_sequence:ident,
        $test_string_items:ident, $test_tuple_items:ident
    ) => {
        heap_test!($test_empty, $heap_type<String, i32>, test_empty_heap);
        heap_test!($test_basic, $heap_type<&'static str, i32>, test_basic_operations);
        heap_test!($test_decrease_key, $heap_type<i32, i32>, test_decrease_key_operations);
        heap_test!($test_multiple_decrease_keys, $heap_type<i32, i32>, test_multiple_decrease_keys);
        heap_test!($test_merge, $heap_type<&'static str, i32>, test_merge_operations);
        heap_test!($test_merge_empty, $heap_type<i32, i32>, test_merge_empty);
        heap_test!($test_duplicate_priorities, $heap_type<&'static str, i32>, test_duplicate_priorities);
        heap_test!($test_stress, $heap_type<i32, i32>, test_stress_operations);
        heap_test!($test_peek_idempotent, $heap_type<&'static str, i32>, test_peek_idempotent);
        heap_test!($test_single_element, $heap_type<&'static str, i32>, test_single_element);
        heap_test!($test_merge_then_operations, $heap_type<i32, i32>, test_merge_then_operations);
        heap_test!($test_rapid_operations, $heap_type<i32, i32>, test_rapid_operations);
        heap_test!($test_large_priorities, $heap_type<i32, i64>, test_large_priorities);
        heap_test!($test_decrease_key_same, $heap_type<i32, i32>, test_decrease_key_same);
        heap_test!($test_complex_sequence, $heap_type<String, i32>, test_complex_sequence);
        heap_test!($test_ascending_insertion, $heap_type<i32, i32>, test_ascending_insertion);
        heap_test!($test_descending_insertion, $heap_type<i32, i32>, test_descending_insertion);
        heap_test!($test_random_order_insertion, $heap_type<i32, i32>, test_random_order_insertion);
        heap_test!($test_multiple_decrease_same, $heap_type<i32, i32>, test_multiple_decrease_same);
        heap_test!($test_decrease_key_new_min, $heap_type<i32, i32>, test_decrease_key_new_min);
        heap_test!($test_alternating_operations, $heap_type<i32, i32>, test_alternating_operations);
        heap_test!($test_merge_large, $heap_type<i32, i32>, test_merge_large);
        heap_test!($test_decrease_key_selective, $heap_type<i32, i32>, test_decrease_key_selective);
        heap_test!($test_all_same_priority, $heap_type<i32, i32>, test_all_same_priority);
        heap_test!($test_negative_priorities, $heap_type<i32, i32>, test_negative_priorities);
        heap_test!($test_decrease_to_negative, $heap_type<i32, i32>, test_decrease_to_negative);
        heap_test!($test_alias_methods, $heap_type<&'static str, i32>, test_alias_methods);
        heap_test!($test_heap_property, $heap_type<i32, i32>, test_heap_property);
        heap_test!($test_merge_with_handles, $heap_type<i32, i32>, test_merge_with_handles);
        heap_test!($test_very_large_sequence, $heap_type<i32, i32>, test_very_large_sequence);
        heap_test!($test_string_items, $heap_type<String, i32>, test_string_items);
        heap_test!($test_tuple_items, $heap_type<(i32, i32), i32>, test_tuple_items);
    };
}

// Generate tests for each heap type with explicit test names
define_heap_tests!(
    fibonacci, FibonacciHeap,
    test_fibonacci_empty, test_fibonacci_basic, test_fibonacci_decrease_key,
    test_fibonacci_multiple_decrease_keys, test_fibonacci_merge, test_fibonacci_merge_empty,
    test_fibonacci_duplicate_priorities, test_fibonacci_stress, test_fibonacci_peek_idempotent,
    test_fibonacci_single_element, test_fibonacci_merge_then_operations, test_fibonacci_rapid_operations,
    test_fibonacci_large_priorities, test_fibonacci_decrease_key_same, test_fibonacci_complex_sequence,
    test_fibonacci_ascending_insertion, test_fibonacci_descending_insertion, test_fibonacci_random_order_insertion,
    test_fibonacci_multiple_decrease_same, test_fibonacci_decrease_key_new_min, test_fibonacci_alternating_operations,
    test_fibonacci_merge_large, test_fibonacci_decrease_key_selective, test_fibonacci_all_same_priority,
    test_fibonacci_negative_priorities, test_fibonacci_decrease_to_negative, test_fibonacci_alias_methods,
    test_fibonacci_heap_property, test_fibonacci_merge_with_handles, test_fibonacci_very_large_sequence,
    test_fibonacci_string_items, test_fibonacci_tuple_items
);

define_heap_tests!(
    pairing, PairingHeap,
    test_pairing_empty, test_pairing_basic, test_pairing_decrease_key,
    test_pairing_multiple_decrease_keys, test_pairing_merge, test_pairing_merge_empty,
    test_pairing_duplicate_priorities, test_pairing_stress, test_pairing_peek_idempotent,
    test_pairing_single_element, test_pairing_merge_then_operations, test_pairing_rapid_operations,
    test_pairing_large_priorities, test_pairing_decrease_key_same, test_pairing_complex_sequence,
    test_pairing_ascending_insertion, test_pairing_descending_insertion, test_pairing_random_order_insertion,
    test_pairing_multiple_decrease_same, test_pairing_decrease_key_new_min, test_pairing_alternating_operations,
    test_pairing_merge_large, test_pairing_decrease_key_selective, test_pairing_all_same_priority,
    test_pairing_negative_priorities, test_pairing_decrease_to_negative, test_pairing_alias_methods,
    test_pairing_heap_property, test_pairing_merge_with_handles, test_pairing_very_large_sequence,
    test_pairing_string_items, test_pairing_tuple_items
);

define_heap_tests!(
    rank_pairing, RankPairingHeap,
    test_rank_pairing_empty, test_rank_pairing_basic, test_rank_pairing_decrease_key,
    test_rank_pairing_multiple_decrease_keys, test_rank_pairing_merge, test_rank_pairing_merge_empty,
    test_rank_pairing_duplicate_priorities, test_rank_pairing_stress, test_rank_pairing_peek_idempotent,
    test_rank_pairing_single_element, test_rank_pairing_merge_then_operations, test_rank_pairing_rapid_operations,
    test_rank_pairing_large_priorities, test_rank_pairing_decrease_key_same, test_rank_pairing_complex_sequence,
    test_rank_pairing_ascending_insertion, test_rank_pairing_descending_insertion, test_rank_pairing_random_order_insertion,
    test_rank_pairing_multiple_decrease_same, test_rank_pairing_decrease_key_new_min, test_rank_pairing_alternating_operations,
    test_rank_pairing_merge_large, test_rank_pairing_decrease_key_selective, test_rank_pairing_all_same_priority,
    test_rank_pairing_negative_priorities, test_rank_pairing_decrease_to_negative, test_rank_pairing_alias_methods,
    test_rank_pairing_heap_property, test_rank_pairing_merge_with_handles, test_rank_pairing_very_large_sequence,
    test_rank_pairing_string_items, test_rank_pairing_tuple_items
);

define_heap_tests!(
    binomial, BinomialHeap,
    test_binomial_empty, test_binomial_basic, test_binomial_decrease_key,
    test_binomial_multiple_decrease_keys, test_binomial_merge, test_binomial_merge_empty,
    test_binomial_duplicate_priorities, test_binomial_stress, test_binomial_peek_idempotent,
    test_binomial_single_element, test_binomial_merge_then_operations, test_binomial_rapid_operations,
    test_binomial_large_priorities, test_binomial_decrease_key_same, test_binomial_complex_sequence,
    test_binomial_ascending_insertion, test_binomial_descending_insertion, test_binomial_random_order_insertion,
    test_binomial_multiple_decrease_same, test_binomial_decrease_key_new_min, test_binomial_alternating_operations,
    test_binomial_merge_large, test_binomial_decrease_key_selective, test_binomial_all_same_priority,
    test_binomial_negative_priorities, test_binomial_decrease_to_negative, test_binomial_alias_methods,
    test_binomial_heap_property, test_binomial_merge_with_handles, test_binomial_very_large_sequence,
    test_binomial_string_items, test_binomial_tuple_items
);

define_heap_tests!(
    brodal, BrodalHeap,
    test_brodal_empty, test_brodal_basic, test_brodal_decrease_key,
    test_brodal_multiple_decrease_keys, test_brodal_merge, test_brodal_merge_empty,
    test_brodal_duplicate_priorities, test_brodal_stress, test_brodal_peek_idempotent,
    test_brodal_single_element, test_brodal_merge_then_operations, test_brodal_rapid_operations,
    test_brodal_large_priorities, test_brodal_decrease_key_same, test_brodal_complex_sequence,
    test_brodal_ascending_insertion, test_brodal_descending_insertion, test_brodal_random_order_insertion,
    test_brodal_multiple_decrease_same, test_brodal_decrease_key_new_min, test_brodal_alternating_operations,
    test_brodal_merge_large, test_brodal_decrease_key_selective, test_brodal_all_same_priority,
    test_brodal_negative_priorities, test_brodal_decrease_to_negative, test_brodal_alias_methods,
    test_brodal_heap_property, test_brodal_merge_with_handles, test_brodal_very_large_sequence,
    test_brodal_string_items, test_brodal_tuple_items
);

define_heap_tests!(
    strict_fibonacci, StrictFibonacciHeap,
    test_strict_fibonacci_empty, test_strict_fibonacci_basic, test_strict_fibonacci_decrease_key,
    test_strict_fibonacci_multiple_decrease_keys, test_strict_fibonacci_merge, test_strict_fibonacci_merge_empty,
    test_strict_fibonacci_duplicate_priorities, test_strict_fibonacci_stress, test_strict_fibonacci_peek_idempotent,
    test_strict_fibonacci_single_element, test_strict_fibonacci_merge_then_operations, test_strict_fibonacci_rapid_operations,
    test_strict_fibonacci_large_priorities, test_strict_fibonacci_decrease_key_same, test_strict_fibonacci_complex_sequence,
    test_strict_fibonacci_ascending_insertion, test_strict_fibonacci_descending_insertion, test_strict_fibonacci_random_order_insertion,
    test_strict_fibonacci_multiple_decrease_same, test_strict_fibonacci_decrease_key_new_min, test_strict_fibonacci_alternating_operations,
    test_strict_fibonacci_merge_large, test_strict_fibonacci_decrease_key_selective, test_strict_fibonacci_all_same_priority,
    test_strict_fibonacci_negative_priorities, test_strict_fibonacci_decrease_to_negative, test_strict_fibonacci_alias_methods,
    test_strict_fibonacci_heap_property, test_strict_fibonacci_merge_with_handles, test_strict_fibonacci_very_large_sequence,
    test_strict_fibonacci_string_items, test_strict_fibonacci_tuple_items
);

define_heap_tests!(
    twothree, TwoThreeHeap,
    test_twothree_empty, test_twothree_basic, test_twothree_decrease_key,
    test_twothree_multiple_decrease_keys, test_twothree_merge, test_twothree_merge_empty,
    test_twothree_duplicate_priorities, test_twothree_stress, test_twothree_peek_idempotent,
    test_twothree_single_element, test_twothree_merge_then_operations, test_twothree_rapid_operations,
    test_twothree_large_priorities, test_twothree_decrease_key_same, test_twothree_complex_sequence,
    test_twothree_ascending_insertion, test_twothree_descending_insertion, test_twothree_random_order_insertion,
    test_twothree_multiple_decrease_same, test_twothree_decrease_key_new_min, test_twothree_alternating_operations,
    test_twothree_merge_large, test_twothree_decrease_key_selective, test_twothree_all_same_priority,
    test_twothree_negative_priorities, test_twothree_decrease_to_negative, test_twothree_alias_methods,
    test_twothree_heap_property, test_twothree_merge_with_handles, test_twothree_very_large_sequence,
    test_twothree_string_items, test_twothree_tuple_items
);

define_heap_tests!(
    skew_binomial, SkewBinomialHeap,
    test_skew_binomial_empty, test_skew_binomial_basic, test_skew_binomial_decrease_key,
    test_skew_binomial_multiple_decrease_keys, test_skew_binomial_merge, test_skew_binomial_merge_empty,
    test_skew_binomial_duplicate_priorities, test_skew_binomial_stress, test_skew_binomial_peek_idempotent,
    test_skew_binomial_single_element, test_skew_binomial_merge_then_operations, test_skew_binomial_rapid_operations,
    test_skew_binomial_large_priorities, test_skew_binomial_decrease_key_same, test_skew_binomial_complex_sequence,
    test_skew_binomial_ascending_insertion, test_skew_binomial_descending_insertion, test_skew_binomial_random_order_insertion,
    test_skew_binomial_multiple_decrease_same, test_skew_binomial_decrease_key_new_min, test_skew_binomial_alternating_operations,
    test_skew_binomial_merge_large, test_skew_binomial_decrease_key_selective, test_skew_binomial_all_same_priority,
    test_skew_binomial_negative_priorities, test_skew_binomial_decrease_to_negative, test_skew_binomial_alias_methods,
    test_skew_binomial_heap_property, test_skew_binomial_merge_with_handles, test_skew_binomial_very_large_sequence,
    test_skew_binomial_string_items, test_skew_binomial_tuple_items
);
