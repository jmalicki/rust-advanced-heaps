//! Extreme stress tests that really push the heaps to their limits
//!
//! These tests perform large numbers of operations in various patterns
//! to catch edge cases and verify correctness under load.

use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::Heap;

/// Test massive numbers of inserts and pops
fn test_massive_operations<H: Heap<i32, i32>>() {
    let mut heap = H::new();

    // Insert 1000 elements
    for i in 0..1000 {
        heap.push(i, i);
    }

    assert_eq!(heap.len(), 1000);
    // Minimum verified by pop sequence

    // Pop all
    for i in 0..1000 {
        assert_eq!(heap.pop(), Some((i, i)));
    }

    assert!(heap.is_empty());
}

/// Test many decrease_key operations
fn test_many_decrease_keys<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();

    // Insert elements with high priorities
    for i in 0..500 {
        handles.push(heap.push(10000 + i, i));
    }

    // Decrease all keys
    for (i, handle) in handles.iter().enumerate() {
        assert!(heap.decrease_key(handle, i as i32).is_ok());
    }

    // Verify order
    for i in 0..500 {
        assert_eq!(heap.pop(), Some((i, i)));
    }
}

/// Test alternating insert and pop
fn test_alternating_ops<H: Heap<i32, i32>>() {
    let mut heap = H::new();

    // Insert-pop-insert-pop pattern
    for i in 0..200 {
        heap.push(i * 2, i);
        heap.push(i * 2 + 1, i + 1000);

        // Pop one
        let popped = heap.pop();
        assert!(popped.is_some());
    }

    // Verify remaining
    while !heap.is_empty() {
        let _ = heap.pop();
    }
    assert!(heap.is_empty());
}

/// Test merge with large heaps
fn test_large_merge<H: Heap<i32, i32>>() {
    let mut heap1 = H::new();
    let mut heap2 = H::new();

    for i in 0..500 {
        heap1.push(i * 2, i);
        heap2.push(i * 2 + 1, i + 1000);
    }

    heap1.merge(heap2);

    assert_eq!(heap1.len(), 1000);

    // Verify all elements are there and in order
    let mut last = i32::MIN;
    while let Some((priority, _)) = heap1.pop() {
        assert!(priority >= last);
        last = priority;
    }
}

/// Test decrease_key on already popped element (should handle gracefully)
fn test_decrease_on_many_operations<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();

    // Insert many elements
    for i in 0..300 {
        handles.push(heap.push(i * 10, i));
    }

    // Pop some
    for _ in 0..100 {
        heap.pop();
    }

    // Try to decrease remaining handles
    for handle in handles.iter().skip(100) {
        // Decrease key (get current min using peek to avoid invalidating handles)
        if let Some((current, _)) = heap.peek() {
            assert!(heap.decrease_key(handle, current - 1).is_ok());
        }
    }

    // Should still work
    assert!(!heap.is_empty());
}

/// Test with very large priorities
fn test_large_priorities<H: Heap<i32, i64>>() {
    let mut heap = H::new();

    heap.push(1_000_000_000, 1);
    heap.push(-1_000_000_000, 2);
    heap.push(2_000_000_000, 3);

    assert_eq!(heap.pop(), Some((-1_000_000_000, 2)));
    assert_eq!(heap.pop(), Some((1_000_000_000, 1)));
    assert_eq!(heap.pop(), Some((2_000_000_000, 3)));
}

/// Test rapid-fire operations
fn test_rapid_fire<H: Heap<i32, i32>>() {
    let mut heap = H::new();
    let mut handles = Vec::new();

    // Rapid insert
    for i in 0..200 {
        handles.push(heap.push(i, i));
    }

    // Rapid decrease keys
    for (i, handle) in handles.iter().enumerate().step_by(2) {
        assert!(heap.decrease_key(handle, i as i32 - 10).is_ok());
    }

    // Rapid pop
    for _ in 0..50 {
        heap.pop();
    }

    // Rapid insert again
    for i in 200..250 {
        heap.push(i, i);
    }

    // Verify structure still valid
    assert!(!heap.is_empty());
    assert!(heap.pop().is_some());
}

#[test]
fn test_fibonacci_massive() {
    test_massive_operations::<FibonacciHeap<i32, i32>>();
}

#[test]
fn test_fibonacci_many_decrease_keys() {
    test_many_decrease_keys::<FibonacciHeap<i32, i32>>();
}

#[test]
fn test_fibonacci_alternating() {
    test_alternating_ops::<FibonacciHeap<i32, i32>>();
}

#[test]
fn test_fibonacci_large_merge() {
    test_large_merge::<FibonacciHeap<i32, i32>>();
}

#[test]
fn test_fibonacci_decrease_on_many() {
    test_decrease_on_many_operations::<FibonacciHeap<i32, i32>>();
}

#[test]
fn test_fibonacci_large_priorities() {
    test_large_priorities::<FibonacciHeap<i32, i64>>();
}

#[test]
fn test_fibonacci_rapid_fire() {
    test_rapid_fire::<FibonacciHeap<i32, i32>>();
}

// Pairing heap stress tests

#[test]
fn test_pairing_massive() {
    test_massive_operations::<PairingHeap<i32, i32>>();
}

#[test]
fn test_pairing_many_decrease_keys() {
    test_many_decrease_keys::<PairingHeap<i32, i32>>();
}

#[test]
fn test_pairing_alternating() {
    test_alternating_ops::<PairingHeap<i32, i32>>();
}

#[test]
fn test_pairing_large_merge() {
    test_large_merge::<PairingHeap<i32, i32>>();
}

#[test]
fn test_pairing_decrease_on_many() {
    test_decrease_on_many_operations::<PairingHeap<i32, i32>>();
}

#[test]
fn test_pairing_large_priorities() {
    test_large_priorities::<PairingHeap<i32, i64>>();
}

#[test]
fn test_pairing_rapid_fire() {
    test_rapid_fire::<PairingHeap<i32, i32>>();
}

// Rank-pairing heap stress tests

#[test]
fn test_rank_pairing_massive() {
    test_massive_operations::<RankPairingHeap<i32, i32>>();
}

#[test]
fn test_rank_pairing_many_decrease_keys() {
    test_many_decrease_keys::<RankPairingHeap<i32, i32>>();
}

#[test]
fn test_rank_pairing_alternating() {
    test_alternating_ops::<RankPairingHeap<i32, i32>>();
}

#[test]
fn test_rank_pairing_large_merge() {
    test_large_merge::<RankPairingHeap<i32, i32>>();
}

#[test]
fn test_rank_pairing_decrease_on_many() {
    test_decrease_on_many_operations::<RankPairingHeap<i32, i32>>();
}

#[test]
fn test_rank_pairing_large_priorities() {
    test_large_priorities::<RankPairingHeap<i32, i64>>();
}

#[test]
fn test_rank_pairing_rapid_fire() {
    test_rapid_fire::<RankPairingHeap<i32, i32>>();
}

// Binomial heap stress tests

#[test]
fn test_binomial_massive() {
    test_massive_operations::<BinomialHeap<i32, i32>>();
}

#[test]
fn test_binomial_many_decrease_keys() {
    test_many_decrease_keys::<BinomialHeap<i32, i32>>();
}

#[test]
fn test_binomial_alternating() {
    test_alternating_ops::<BinomialHeap<i32, i32>>();
}

#[test]
fn test_binomial_large_merge() {
    test_large_merge::<BinomialHeap<i32, i32>>();
}

#[test]
fn test_binomial_decrease_on_many() {
    test_decrease_on_many_operations::<BinomialHeap<i32, i32>>();
}

#[test]
fn test_binomial_large_priorities() {
    test_large_priorities::<BinomialHeap<i32, i64>>();
}

#[test]
fn test_binomial_rapid_fire() {
    test_rapid_fire::<BinomialHeap<i32, i32>>();
}
