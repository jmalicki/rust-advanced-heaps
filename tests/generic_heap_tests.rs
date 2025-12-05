//! Generic comprehensive tests for all Heap implementations
//!
//! These tests work with any Heap implementation and stress the trait interface
//! with various edge cases and complex scenarios.
//!
//! Test organization:
//! - Base `Heap` trait tests: Applied to ALL heaps including SimpleBinaryHeap
//! - `DecreaseKeyHeap` trait tests: Applied only to heaps with decrease_key support

use rust_advanced_heaps::traits::HeapError;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap, MergeableHeap};

// ============================================================================
// Base Heap trait tests - work with any Heap implementation
// ============================================================================

/// Generate base Heap tests for a heap type using a module
macro_rules! base_heap_tests {
    ($mod_name:ident, $heap_type:ty) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn test_empty_heap() {
                let mut heap = <$heap_type>::new();
                assert!(heap.is_empty());
                assert_eq!(heap.len(), 0);
                assert_eq!(heap.peek(), None);
                assert_eq!(heap.pop(), None);
            }

            #[test]
            fn test_basic_operations() {
                let mut heap = <$heap_type>::new();
                heap.push(5, "five");
                heap.push(1, "one");
                heap.push(10, "ten");
                heap.push(3, "three");

                assert!(!heap.is_empty());
                assert_eq!(heap.len(), 4);
                assert_eq!(heap.peek(), Some((&1, &"one")));

                assert_eq!(heap.pop(), Some((1, "one")));
                assert_eq!(heap.pop(), Some((3, "three")));
                assert_eq!(heap.pop(), Some((5, "five")));
                assert_eq!(heap.pop(), Some((10, "ten")));
                assert_eq!(heap.pop(), None);
                assert!(heap.is_empty());
            }

            #[test]
            fn test_merge_operations() {
                let mut heap1 = <$heap_type>::new();
                heap1.push(5, "five");
                heap1.push(1, "one");

                let mut heap2 = <$heap_type>::new();
                heap2.push(10, "ten");
                heap2.push(3, "three");

                heap1.merge(heap2);

                assert_eq!(heap1.len(), 4);
                assert_eq!(heap1.peek(), Some((&1, &"one")));
                assert_eq!(heap1.pop(), Some((1, "one")));
                assert_eq!(heap1.pop(), Some((3, "three")));
                assert_eq!(heap1.pop(), Some((5, "five")));
                assert_eq!(heap1.pop(), Some((10, "ten")));
            }

            #[test]
            fn test_merge_empty() {
                let mut heap1 = <$heap_type>::new();
                heap1.push(5, "a");
                heap1.push(1, "b");

                let heap2 = <$heap_type>::new();
                let len_before = heap1.len();
                heap1.merge(heap2);
                assert_eq!(heap1.len(), len_before);

                let mut heap3 = <$heap_type>::new();
                let mut heap4 = <$heap_type>::new();
                heap4.push(3, "c");
                heap3.merge(heap4);
                assert_eq!(heap3.len(), 1);
                assert_eq!(heap3.peek(), Some((&3, &"c")));
            }

            #[test]
            fn test_duplicate_priorities() {
                let mut heap = <$heap_type>::new();
                heap.push(5, "a");
                heap.push(5, "b");
                heap.push(5, "c");
                heap.push(1, "d");

                assert_eq!(heap.pop(), Some((1, "d")));

                let mut seen = std::collections::HashSet::new();
                for _ in 0..3 {
                    if let Some((pri, item)) = heap.pop() {
                        assert_eq!(pri, 5);
                        assert!(seen.insert(item));
                    }
                }
                assert_eq!(seen.len(), 3);
            }

            #[test]
            fn test_peek_idempotent() {
                let mut heap = <$heap_type>::new();
                heap.push(5, "five");
                heap.push(1, "one");

                assert_eq!(heap.peek(), Some((&1, &"one")));
                assert_eq!(heap.peek(), Some((&1, &"one")));
                assert_eq!(heap.peek(), Some((&1, &"one")));
                assert_eq!(heap.len(), 2);
                assert_eq!(heap.pop(), Some((1, "one")));
            }

            #[test]
            fn test_merge_then_operations() {
                let mut heap1 = <$heap_type>::new();
                for i in 0..10 {
                    heap1.push(i * 10, &"a");
                }

                let mut heap2 = <$heap_type>::new();
                for i in 10..20 {
                    heap2.push(i * 10, &"b");
                }

                heap1.merge(heap2);

                let mut count = 0;
                let mut last_priority = i32::MIN;
                while let Some((priority, _)) = heap1.pop() {
                    assert!(priority >= last_priority);
                    last_priority = priority;
                    count += 1;
                }
                assert_eq!(count, 20);
            }

            #[test]
            fn test_rapid_operations() {
                let mut heap = <$heap_type>::new();

                for i in 0..50 {
                    heap.push(i, &"x");
                    if i % 3 == 0 {
                        heap.pop();
                    }
                }

                assert!(!heap.is_empty());
                let mut count = 0;
                while heap.pop().is_some() {
                    count += 1;
                }
                assert!(count > 0);
            }

            #[test]
            fn test_large_priorities() {
                let mut heap = <$heap_type>::new();

                heap.push(1_000_000_000, "a");
                heap.push(1_500_000_000, "b");
                heap.push(-1_000_000_000, "c");

                assert_eq!(heap.peek(), Some((&-1_000_000_000, &"c")));
                assert_eq!(heap.pop(), Some((-1_000_000_000, "c")));
                assert_eq!(heap.pop(), Some((1_000_000_000, "a")));
                assert_eq!(heap.pop(), Some((1_500_000_000, "b")));
            }

            #[test]
            fn test_ascending_insertion() {
                let mut heap = <$heap_type>::new();
                for i in 0..50 {
                    heap.push(i, &"x");
                }
                for i in 0..50 {
                    let (pri, _) = heap.pop().unwrap();
                    assert_eq!(pri, i);
                }
            }

            #[test]
            fn test_descending_insertion() {
                let mut heap = <$heap_type>::new();
                for i in (0..50).rev() {
                    heap.push(i, &"x");
                }
                for i in 0..50 {
                    let (pri, _) = heap.pop().unwrap();
                    assert_eq!(pri, i);
                }
            }

            #[test]
            fn test_random_order_insertion() {
                let mut heap = <$heap_type>::new();
                let mut values: Vec<i32> = (0..100).collect();
                for i in (0..values.len()).step_by(7) {
                    if i + 1 < values.len() {
                        values.swap(i, i + 1);
                    }
                }

                for &val in &values {
                    heap.push(val * 2, &"x");
                }

                for i in 0..100 {
                    let (pri, _) = heap.pop().unwrap();
                    assert_eq!(pri, i * 2);
                }
            }

            #[test]
            fn test_alternating_operations() {
                let mut heap = <$heap_type>::new();

                for i in 0..10 {
                    heap.push(i * 10, &"x");
                }

                heap.pop();
                heap.pop();
                heap.pop();

                for i in 10..15 {
                    heap.push(i * 10, &"x");
                }

                heap.pop();
                heap.pop();

                assert!(!heap.is_empty());
                assert!(heap.peek().is_some());

                let mut count = 0;
                while heap.pop().is_some() {
                    count += 1;
                }
                assert_eq!(count, 10);
            }

            #[test]
            fn test_merge_large() {
                let mut heap1 = <$heap_type>::new();
                for i in 0..100 {
                    heap1.push(i * 2, &"a");
                }

                let mut heap2 = <$heap_type>::new();
                for i in 100..200 {
                    heap2.push(i * 2, &"b");
                }

                heap1.merge(heap2);
                assert_eq!(heap1.len(), 200);

                let mut last = i32::MIN;
                let mut count = 0;
                while let Some((priority, _)) = heap1.pop() {
                    assert!(priority >= last);
                    last = priority;
                    count += 1;
                }
                assert_eq!(count, 200);
            }

            #[test]
            fn test_negative_priorities() {
                let mut heap = <$heap_type>::new();

                heap.push(-10, "a");
                heap.push(10, "b");
                heap.push(-5, "c");
                heap.push(5, "d");

                assert_eq!(heap.pop(), Some((-10, "a")));
                assert_eq!(heap.pop(), Some((-5, "c")));
                assert_eq!(heap.pop(), Some((5, "d")));
                assert_eq!(heap.pop(), Some((10, "b")));
            }

            #[test]
            fn test_alias_methods() {
                let mut heap = <$heap_type>::new();
                heap.push(5, "five");
                assert_eq!(heap.len(), 1);
                assert_eq!(heap.peek(), Some((&5, &"five")));
                assert_eq!(heap.pop(), Some((5, "five")));
                assert_eq!(heap.pop(), None);
            }

            #[test]
            fn test_string_items() {
                let mut heap = <$heap_type>::new();

                heap.push(3, "c");
                heap.push(1, "a");
                heap.push(2, "b");

                assert_eq!(heap.peek().unwrap().1, &"a");
                assert_eq!(heap.pop().unwrap().1, "a");
                assert_eq!(heap.pop().unwrap().1, "b");
                assert_eq!(heap.pop().unwrap().1, "c");
            }
        }
    };
}

/// Generate DecreaseKeyHeap tests for a heap type
macro_rules! decrease_key_heap_tests {
    ($mod_name:ident, $heap_type:ty) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn test_decrease_key_operations() {
                let mut heap = <$heap_type>::new();

                let _h1 = heap.push_with_handle(100, 1);
                let h2 = heap.push_with_handle(200, 2);
                let _h3 = heap.push_with_handle(300, 3);
                let h4 = heap.push_with_handle(400, 4);

                assert_eq!(heap.peek(), Some((&100, &1)));

                assert!(heap.decrease_key(&h2, 50).is_ok());
                assert_eq!(heap.peek(), Some((&50, &2)));

                assert!(heap.decrease_key(&h4, 25).is_ok());
                assert_eq!(heap.peek(), Some((&25, &4)));

                assert!(heap.decrease_key(&h4, 1).is_ok());
                assert_eq!(heap.peek(), Some((&1, &4)));

                assert_eq!(heap.pop(), Some((1, 4)));
                assert_eq!(heap.pop(), Some((50, 2)));
                assert_eq!(heap.pop(), Some((100, 1)));
                assert_eq!(heap.pop(), Some((300, 3)));
            }

            #[test]
            fn test_multiple_decrease_keys() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..20 {
                    handles.push(heap.push_with_handle((i + 1) * 100, i));
                }

                for (i, handle) in handles.iter().enumerate() {
                    assert!(heap.decrease_key(handle, i as i32).is_ok());
                }

                assert_eq!(heap.peek(), Some((&0, &0)));

                for i in 0..20 {
                    assert_eq!(heap.pop(), Some((i, i)));
                }
                assert!(heap.is_empty());
            }

            #[test]
            fn test_stress_operations() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..100 {
                    handles.push(heap.push_with_handle(i * 2, i));
                }

                for i in (0..100).step_by(3) {
                    assert!(heap.decrease_key(&handles[i], i as i32 * 2 - 1).is_ok());
                }

                for _ in 0..20 {
                    heap.pop();
                }

                assert!(!heap.is_empty());
                assert!(heap.peek().is_some());

                let mut count = 0;
                while heap.pop().is_some() {
                    count += 1;
                }
                assert_eq!(count, 80);
            }

            #[test]
            fn test_single_element_decrease() {
                let mut heap = <$heap_type>::new();
                let handle = heap.push_with_handle(42, 99);

                assert_eq!(heap.len(), 1);
                assert_eq!(heap.peek(), Some((&42, &99)));

                assert!(heap.decrease_key(&handle, 10).is_ok());
                assert_eq!(heap.peek(), Some((&10, &99)));

                assert_eq!(heap.pop(), Some((10, 99)));
                assert!(heap.is_empty());
            }

            #[test]
            fn test_decrease_key_same() {
                let mut heap = <$heap_type>::new();
                let handle = heap.push_with_handle(10, 1);

                assert_eq!(
                    heap.decrease_key(&handle, 10),
                    Err(HeapError::PriorityNotDecreased)
                );

                assert_eq!(heap.peek(), Some((&10, &1)));
            }

            #[test]
            fn test_complex_sequence() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..20 {
                    if i % 2 == 0 {
                        handles.push(heap.push_with_handle(i * 10, i + 1000));
                    }
                }

                for (i, handle) in handles.iter().enumerate().skip(3).step_by(3) {
                    let new_priority = (i as i32 * 5).max(1);
                    if new_priority < i as i32 * 10 {
                        assert!(heap.decrease_key(handle, new_priority).is_ok());
                    }
                }

                for _ in 0..3 {
                    if heap.pop().is_none() {
                        break;
                    }
                }

                for i in 20..25 {
                    heap.push(i * 10, i + 2000);
                }

                assert!(!heap.is_empty());
                assert!(heap.peek().is_some());

                let mut count = 0;
                while heap.pop().is_some() {
                    count += 1;
                    if count > 50 {
                        break;
                    }
                }

                assert!(heap.is_empty() || heap.peek().is_some());
            }

            #[test]
            fn test_multiple_decrease_same() {
                let mut heap = <$heap_type>::new();
                let handle = heap.push_with_handle(1000, 1);

                assert!(heap.decrease_key(&handle, 500).is_ok());
                assert_eq!(heap.peek(), Some((&500, &1)));

                assert!(heap.decrease_key(&handle, 250).is_ok());
                assert_eq!(heap.peek(), Some((&250, &1)));

                assert!(heap.decrease_key(&handle, 100).is_ok());
                assert_eq!(heap.peek(), Some((&100, &1)));

                assert!(heap.decrease_key(&handle, 50).is_ok());
                assert_eq!(heap.peek(), Some((&50, &1)));

                assert!(heap.decrease_key(&handle, 1).is_ok());
                assert_eq!(heap.peek(), Some((&1, &1)));
            }

            #[test]
            fn test_decrease_key_new_min() {
                let mut heap = <$heap_type>::new();
                let h1 = heap.push_with_handle(100, 1);
                let h2 = heap.push_with_handle(200, 2);
                let h3 = heap.push_with_handle(300, 3);

                assert!(heap.decrease_key(&h3, 150).is_ok());
                assert_eq!(heap.peek(), Some((&100, &1)));

                assert!(heap.decrease_key(&h2, 50).is_ok());
                assert_eq!(heap.peek(), Some((&50, &2)));

                assert!(heap.decrease_key(&h1, 25).is_ok());
                assert_eq!(heap.peek(), Some((&25, &1)));
            }

            #[test]
            fn test_decrease_key_selective() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..30 {
                    handles.push(heap.push_with_handle((i + 1) * 100, i));
                }

                for (i, handle) in handles.iter().enumerate() {
                    if i % 2 == 0 {
                        assert!(heap.decrease_key(handle, i as i32 * 10).is_ok());
                    }
                }

                let min = heap.peek();
                assert!(min.is_some());
                assert!(*min.unwrap().0 < 100);

                let mut last = i32::MIN;
                while let Some((priority, _)) = heap.pop() {
                    assert!(priority >= last);
                    last = priority;
                }
            }

            #[test]
            fn test_all_same_priority() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..10 {
                    handles.push(heap.push_with_handle((i + 1) * 10, i));
                }

                for handle in &handles {
                    assert!(heap.decrease_key(handle, 5).is_ok());
                }

                assert_eq!(heap.peek().unwrap().0, &5);

                let mut seen = std::collections::HashSet::new();
                while let Some((priority, item)) = heap.pop() {
                    assert_eq!(priority, 5);
                    assert!(seen.insert(item));
                }
                assert_eq!(seen.len(), 10);
            }

            #[test]
            fn test_decrease_to_negative() {
                let mut heap = <$heap_type>::new();
                let h1 = heap.push_with_handle(10, 1);
                let _h2 = heap.push_with_handle(20, 2);

                assert!(heap.decrease_key(&h1, -5).is_ok());
                assert_eq!(heap.peek(), Some((&-5, &1)));
                assert_eq!(heap.pop(), Some((-5, 1)));
            }

            #[test]
            fn test_heap_property() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..50 {
                    handles.push(heap.push_with_handle((i * 17 + 23) % 1000, i));
                }

                for i in (0..50).step_by(7) {
                    if let Some(handle) = handles.get(i) {
                        let current_min = heap.peek().unwrap().0;
                        let new_priority = (*current_min / 2).max(1);
                        assert!(heap.decrease_key(handle, new_priority).is_ok());
                    }
                }

                let mut last_priority = i32::MIN;
                while let Some((priority, _)) = heap.pop() {
                    assert!(priority >= last_priority);
                    last_priority = priority;
                }
            }

            #[test]
            fn test_merge_with_handles() {
                let mut heap1 = <$heap_type>::new();
                let h1 = heap1.push_with_handle(100, 1);

                let mut heap2 = <$heap_type>::new();
                let h2 = heap2.push_with_handle(200, 2);

                heap1.merge(heap2);

                assert!(heap1.decrease_key(&h1, 50).is_ok());
                assert_eq!(heap1.peek(), Some((&50, &1)));

                assert!(heap1.decrease_key(&h2, 25).is_ok());
                assert_eq!(heap1.peek(), Some((&25, &2)));
            }

            #[test]
            fn test_very_large_sequence() {
                let mut heap = <$heap_type>::new();
                let mut handles = Vec::new();

                for i in 0..1000 {
                    handles.push(heap.push_with_handle(i * 10, i));
                }

                for i in (0..1000).step_by(10) {
                    assert!(heap.decrease_key(&handles[i], (i as i32) - 1).is_ok());
                }

                for _ in 0..100 {
                    heap.pop();
                }

                for i in 1000..1200 {
                    heap.push(i * 10, i);
                }

                assert!(!heap.is_empty());

                let mut count = 0;
                while heap.pop().is_some() {
                    count += 1;
                }
                assert_eq!(count, 1100);
            }
        }
    };
}

// ============================================================================
// Generate tests for all heap implementations
// ============================================================================

// SimpleBinaryHeap - base Heap only (no decrease_key)
base_heap_tests!(
    simple_binary_base,
    rust_advanced_heaps::simple_binary::SimpleBinaryHeap<&'static str, i32>
);

// Fibonacci Heap
base_heap_tests!(
    fibonacci_base,
    rust_advanced_heaps::fibonacci::FibonacciHeap<&'static str, i32>
);
decrease_key_heap_tests!(fibonacci_decrease, rust_advanced_heaps::fibonacci::FibonacciHeap<i32, i32>);

// Pairing Heap
base_heap_tests!(
    pairing_base,
    rust_advanced_heaps::pairing::PairingHeap<&'static str, i32>
);
decrease_key_heap_tests!(pairing_decrease, rust_advanced_heaps::pairing::PairingHeap<i32, i32>);

// Binomial Heap
base_heap_tests!(
    binomial_base,
    rust_advanced_heaps::binomial::BinomialHeap<&'static str, i32>
);
decrease_key_heap_tests!(binomial_decrease, rust_advanced_heaps::binomial::BinomialHeap<i32, i32>);

// Rank-Pairing Heap
base_heap_tests!(
    rank_pairing_base,
    rust_advanced_heaps::rank_pairing::RankPairingHeap<&'static str, i32>
);
decrease_key_heap_tests!(rank_pairing_decrease, rust_advanced_heaps::rank_pairing::RankPairingHeap<i32, i32>);

// Strict Fibonacci Heap
base_heap_tests!(
    strict_fibonacci_base,
    rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap<&'static str, i32>
);
decrease_key_heap_tests!(strict_fibonacci_decrease, rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap<i32, i32>);

// 2-3 Heap
base_heap_tests!(
    twothree_base,
    rust_advanced_heaps::twothree::TwoThreeHeap<&'static str, i32>
);
decrease_key_heap_tests!(twothree_decrease, rust_advanced_heaps::twothree::TwoThreeHeap<i32, i32>);

// Skew Binomial Heap
base_heap_tests!(
    skew_binomial_base,
    rust_advanced_heaps::skew_binomial::SkewBinomialHeap<&'static str, i32>
);
decrease_key_heap_tests!(skew_binomial_decrease, rust_advanced_heaps::skew_binomial::SkewBinomialHeap<i32, i32>);

// Skew Binomial Heap Arena (experimental - requires arena-storage feature)
// NOTE: Arena storage does NOT implement MergeableHeap because handles from merged
// heaps would become invalid when nodes are moved to new storage. For this reason,
// we use specialized tests that exclude merge operations.
#[cfg(feature = "arena-storage")]
mod skew_binomial_arena_base {
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeapArena;
    use rust_advanced_heaps::Heap;

    #[test]
    fn test_empty_heap() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.peek(), None);
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_basic_operations() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        heap.push(5, "five");
        heap.push(1, "one");
        heap.push(10, "ten");
        heap.push(3, "three");

        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 4);
        assert_eq!(heap.peek(), Some((&1, &"one")));

        assert_eq!(heap.pop(), Some((1, "one")));
        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.pop(), Some((5, "five")));
        assert_eq!(heap.pop(), Some((10, "ten")));
        assert_eq!(heap.pop(), None);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_duplicate_priorities() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        heap.push(5, "a");
        heap.push(5, "b");
        heap.push(5, "c");
        heap.push(1, "d");

        assert_eq!(heap.pop(), Some((1, "d")));

        let mut seen = std::collections::HashSet::new();
        for _ in 0..3 {
            if let Some((pri, item)) = heap.pop() {
                assert_eq!(pri, 5);
                assert!(seen.insert(item));
            }
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_peek_idempotent() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        heap.push(5, "five");
        heap.push(1, "one");

        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.pop(), Some((1, "one")));
    }

    #[test]
    fn test_rapid_operations() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();

        for i in 0..50 {
            heap.push(i, "x");
            if i % 3 == 0 {
                heap.pop();
            }
        }

        assert!(!heap.is_empty());
        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert!(count > 0);
    }

    #[test]
    fn test_large_priorities() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();

        heap.push(1_000_000_000, "a");
        heap.push(1_500_000_000, "b");
        heap.push(-1_000_000_000, "c");

        assert_eq!(heap.peek(), Some((&-1_000_000_000, &"c")));
        assert_eq!(heap.pop(), Some((-1_000_000_000, "c")));
        assert_eq!(heap.pop(), Some((1_000_000_000, "a")));
        assert_eq!(heap.pop(), Some((1_500_000_000, "b")));
    }

    #[test]
    fn test_ascending_insertion() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        for i in 0..50 {
            heap.push(i, "x");
        }
        for i in 0..50 {
            let (pri, _) = heap.pop().unwrap();
            assert_eq!(pri, i);
        }
    }

    #[test]
    fn test_descending_insertion() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        for i in (0..50).rev() {
            heap.push(i, "x");
        }
        for i in 0..50 {
            let (pri, _) = heap.pop().unwrap();
            assert_eq!(pri, i);
        }
    }

    #[test]
    fn test_random_order_insertion() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        let mut values: Vec<i32> = (0..100).collect();
        for i in (0..values.len()).step_by(7) {
            if i + 1 < values.len() {
                values.swap(i, i + 1);
            }
        }

        for &val in &values {
            heap.push(val * 2, "x");
        }

        for i in 0..100 {
            let (pri, _) = heap.pop().unwrap();
            assert_eq!(pri, i * 2);
        }
    }

    #[test]
    fn test_alternating_operations() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();

        for i in 0..10 {
            heap.push(i * 10, "x");
        }

        heap.pop();
        heap.pop();
        heap.pop();

        for i in 10..15 {
            heap.push(i * 10, "x");
        }

        heap.pop();
        heap.pop();

        assert!(!heap.is_empty());
        assert!(heap.peek().is_some());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_negative_priorities() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();

        heap.push(-10, "a");
        heap.push(10, "b");
        heap.push(-5, "c");
        heap.push(5, "d");

        assert_eq!(heap.pop(), Some((-10, "a")));
        assert_eq!(heap.pop(), Some((-5, "c")));
        assert_eq!(heap.pop(), Some((5, "d")));
        assert_eq!(heap.pop(), Some((10, "b")));
    }

    #[test]
    fn test_alias_methods() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();
        heap.push(5, "five");
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.peek(), Some((&5, &"five")));
        assert_eq!(heap.pop(), Some((5, "five")));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_string_items() {
        let mut heap = SkewBinomialHeapArena::<&str, i32>::new();

        heap.push(3, "c");
        heap.push(1, "a");
        heap.push(2, "b");

        assert_eq!(heap.peek().unwrap().1, &"a");
        assert_eq!(heap.pop().unwrap().1, "a");
        assert_eq!(heap.pop().unwrap().1, "b");
        assert_eq!(heap.pop().unwrap().1, "c");
    }
}

// Specialized arena decrease-key tests (excluding test_merge_with_handles due to arena limitation)
#[cfg(feature = "arena-storage")]
mod skew_binomial_arena_decrease {
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeapArena;
    use rust_advanced_heaps::traits::HeapError;
    use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

    #[test]
    fn test_decrease_key_operations() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();

        let _h1 = heap.push_with_handle(100, 1);
        let h2 = heap.push_with_handle(200, 2);
        let _h3 = heap.push_with_handle(300, 3);
        let h4 = heap.push_with_handle(400, 4);

        assert_eq!(heap.peek(), Some((&100, &1)));

        assert!(heap.decrease_key(&h2, 50).is_ok());
        assert_eq!(heap.peek(), Some((&50, &2)));

        assert!(heap.decrease_key(&h4, 25).is_ok());
        assert_eq!(heap.peek(), Some((&25, &4)));

        assert!(heap.decrease_key(&h4, 1).is_ok());
        assert_eq!(heap.peek(), Some((&1, &4)));

        assert_eq!(heap.pop(), Some((1, 4)));
        assert_eq!(heap.pop(), Some((50, 2)));
        assert_eq!(heap.pop(), Some((100, 1)));
        assert_eq!(heap.pop(), Some((300, 3)));
    }

    #[test]
    fn test_multiple_decrease_keys() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..20 {
            handles.push(heap.push_with_handle((i + 1) * 100, i));
        }

        for (i, handle) in handles.iter().enumerate() {
            assert!(heap.decrease_key(handle, i as i32).is_ok());
        }

        assert_eq!(heap.peek(), Some((&0, &0)));

        for i in 0..20 {
            assert_eq!(heap.pop(), Some((i, i)));
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn test_stress_operations() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..100 {
            handles.push(heap.push_with_handle(i * 2, i));
        }

        for i in (0..100).step_by(3) {
            assert!(heap.decrease_key(&handles[i], i as i32 * 2 - 1).is_ok());
        }

        for _ in 0..20 {
            heap.pop();
        }

        assert!(!heap.is_empty());
        assert!(heap.peek().is_some());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 80);
    }

    #[test]
    fn test_single_element_decrease() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let handle = heap.push_with_handle(42, 99);

        assert_eq!(heap.len(), 1);
        assert_eq!(heap.peek(), Some((&42, &99)));

        assert!(heap.decrease_key(&handle, 10).is_ok());
        assert_eq!(heap.peek(), Some((&10, &99)));

        assert_eq!(heap.pop(), Some((10, 99)));
        assert!(heap.is_empty());
    }

    #[test]
    fn test_decrease_key_same() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let handle = heap.push_with_handle(10, 1);

        assert_eq!(
            heap.decrease_key(&handle, 10),
            Err(HeapError::PriorityNotDecreased)
        );

        assert_eq!(heap.peek(), Some((&10, &1)));
    }

    #[test]
    fn test_multiple_decrease_same() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let handle = heap.push_with_handle(1000, 1);

        assert!(heap.decrease_key(&handle, 500).is_ok());
        assert_eq!(heap.peek(), Some((&500, &1)));

        assert!(heap.decrease_key(&handle, 250).is_ok());
        assert_eq!(heap.peek(), Some((&250, &1)));

        assert!(heap.decrease_key(&handle, 100).is_ok());
        assert_eq!(heap.peek(), Some((&100, &1)));

        assert!(heap.decrease_key(&handle, 50).is_ok());
        assert_eq!(heap.peek(), Some((&50, &1)));

        assert!(heap.decrease_key(&handle, 1).is_ok());
        assert_eq!(heap.peek(), Some((&1, &1)));
    }

    #[test]
    fn test_decrease_key_new_min() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let h1 = heap.push_with_handle(100, 1);
        let h2 = heap.push_with_handle(200, 2);
        let h3 = heap.push_with_handle(300, 3);

        assert!(heap.decrease_key(&h3, 150).is_ok());
        assert_eq!(heap.peek(), Some((&100, &1)));

        assert!(heap.decrease_key(&h2, 50).is_ok());
        assert_eq!(heap.peek(), Some((&50, &2)));

        assert!(heap.decrease_key(&h1, 25).is_ok());
        assert_eq!(heap.peek(), Some((&25, &1)));
    }

    #[test]
    fn test_decrease_key_selective() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..30 {
            handles.push(heap.push_with_handle((i + 1) * 100, i));
        }

        for (i, handle) in handles.iter().enumerate() {
            if i % 2 == 0 {
                assert!(heap.decrease_key(handle, i as i32 * 10).is_ok());
            }
        }

        let min = heap.peek();
        assert!(min.is_some());
        assert!(*min.unwrap().0 < 100);

        let mut last = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(priority >= last);
            last = priority;
        }
    }

    #[test]
    fn test_all_same_priority() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..10 {
            handles.push(heap.push_with_handle((i + 1) * 10, i));
        }

        for handle in &handles {
            assert!(heap.decrease_key(handle, 5).is_ok());
        }

        assert_eq!(heap.peek().unwrap().0, &5);

        let mut seen = std::collections::HashSet::new();
        while let Some((priority, item)) = heap.pop() {
            assert_eq!(priority, 5);
            assert!(seen.insert(item));
        }
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn test_decrease_to_negative() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let h1 = heap.push_with_handle(10, 1);
        let _h2 = heap.push_with_handle(20, 2);

        assert!(heap.decrease_key(&h1, -5).is_ok());
        assert_eq!(heap.peek(), Some((&-5, &1)));
        assert_eq!(heap.pop(), Some((-5, 1)));
    }

    #[test]
    fn test_heap_property() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..50 {
            handles.push(heap.push_with_handle((i * 17 + 23) % 1000, i));
        }

        for i in (0..50).step_by(7) {
            if let Some(handle) = handles.get(i) {
                let current_min = heap.peek().unwrap().0;
                let new_priority = (*current_min / 2).max(1);
                assert!(heap.decrease_key(handle, new_priority).is_ok());
            }
        }

        let mut last_priority = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(priority >= last_priority);
            last_priority = priority;
        }
    }

    // NOTE: test_merge_with_handles is excluded for arena storage because
    // handles from merged heaps become invalid when nodes are moved to new storage.
    // This is a known limitation of arena-based storage.

    #[test]
    fn test_very_large_sequence() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..1000 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        for i in (0..1000).step_by(10) {
            assert!(heap.decrease_key(&handles[i], (i as i32) - 1).is_ok());
        }

        for _ in 0..100 {
            heap.pop();
        }

        for i in 1000..1200 {
            heap.push(i * 10, i);
        }

        assert!(!heap.is_empty());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 1100);
    }

    #[test]
    fn test_complex_sequence() {
        let mut heap = SkewBinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..20 {
            if i % 2 == 0 {
                handles.push(heap.push_with_handle(i * 10, i + 1000));
            }
        }

        for (i, handle) in handles.iter().enumerate().skip(3).step_by(3) {
            let new_priority = (i as i32 * 5).max(1);
            if new_priority < i as i32 * 10 {
                assert!(heap.decrease_key(handle, new_priority).is_ok());
            }
        }

        for _ in 0..3 {
            if heap.pop().is_none() {
                break;
            }
        }

        for i in 20..25 {
            heap.push(i * 10, i + 2000);
        }

        assert!(!heap.is_empty());
        assert!(heap.peek().is_some());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
            if count > 50 {
                break;
            }
        }

        assert!(heap.is_empty() || heap.peek().is_some());
    }
}

// Binomial Heap Arena (experimental - requires arena-storage feature)
// NOTE: Arena storage does NOT implement MergeableHeap because handles from merged
// heaps would become invalid when nodes are moved to new storage. For this reason,
// we use specialized tests that exclude merge operations.
#[cfg(feature = "arena-storage")]
mod binomial_arena_base {
    use rust_advanced_heaps::binomial::BinomialHeapArena;
    use rust_advanced_heaps::Heap;

    #[test]
    fn test_empty_heap() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.peek(), None);
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_basic_operations() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        heap.push(5, "five");
        heap.push(1, "one");
        heap.push(10, "ten");
        heap.push(3, "three");

        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 4);
        assert_eq!(heap.peek(), Some((&1, &"one")));

        assert_eq!(heap.pop(), Some((1, "one")));
        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.pop(), Some((5, "five")));
        assert_eq!(heap.pop(), Some((10, "ten")));
        assert_eq!(heap.pop(), None);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_duplicate_priorities() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        heap.push(5, "a");
        heap.push(5, "b");
        heap.push(5, "c");
        heap.push(1, "d");

        assert_eq!(heap.pop(), Some((1, "d")));

        let mut seen = std::collections::HashSet::new();
        for _ in 0..3 {
            if let Some((pri, item)) = heap.pop() {
                assert_eq!(pri, 5);
                assert!(seen.insert(item));
            }
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn test_peek_idempotent() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        heap.push(5, "five");
        heap.push(1, "one");

        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.peek(), Some((&1, &"one")));
        assert_eq!(heap.len(), 2);
        assert_eq!(heap.pop(), Some((1, "one")));
    }

    #[test]
    fn test_rapid_operations() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();

        for i in 0..50 {
            heap.push(i, "x");
            if i % 3 == 0 {
                heap.pop();
            }
        }

        assert!(!heap.is_empty());
        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert!(count > 0);
    }

    #[test]
    fn test_large_priorities() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();

        heap.push(1_000_000_000, "a");
        heap.push(1_500_000_000, "b");
        heap.push(-1_000_000_000, "c");

        assert_eq!(heap.peek(), Some((&-1_000_000_000, &"c")));
        assert_eq!(heap.pop(), Some((-1_000_000_000, "c")));
        assert_eq!(heap.pop(), Some((1_000_000_000, "a")));
        assert_eq!(heap.pop(), Some((1_500_000_000, "b")));
    }

    #[test]
    fn test_ascending_insertion() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        for i in 0..50 {
            heap.push(i, "x");
        }
        for i in 0..50 {
            let (pri, _) = heap.pop().unwrap();
            assert_eq!(pri, i);
        }
    }

    #[test]
    fn test_descending_insertion() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        for i in (0..50).rev() {
            heap.push(i, "x");
        }
        for i in 0..50 {
            let (pri, _) = heap.pop().unwrap();
            assert_eq!(pri, i);
        }
    }

    #[test]
    fn test_random_order_insertion() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        let mut values: Vec<i32> = (0..100).collect();
        for i in (0..values.len()).step_by(7) {
            if i + 1 < values.len() {
                values.swap(i, i + 1);
            }
        }

        for &val in &values {
            heap.push(val * 2, "x");
        }

        for i in 0..100 {
            let (pri, _) = heap.pop().unwrap();
            assert_eq!(pri, i * 2);
        }
    }

    #[test]
    fn test_alternating_operations() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();

        for i in 0..10 {
            heap.push(i * 10, "x");
        }

        heap.pop();
        heap.pop();
        heap.pop();

        for i in 10..15 {
            heap.push(i * 10, "x");
        }

        heap.pop();
        heap.pop();

        assert!(!heap.is_empty());
        assert!(heap.peek().is_some());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn test_negative_priorities() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();

        heap.push(-10, "a");
        heap.push(10, "b");
        heap.push(-5, "c");
        heap.push(5, "d");

        assert_eq!(heap.pop(), Some((-10, "a")));
        assert_eq!(heap.pop(), Some((-5, "c")));
        assert_eq!(heap.pop(), Some((5, "d")));
        assert_eq!(heap.pop(), Some((10, "b")));
    }

    #[test]
    fn test_alias_methods() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();
        heap.push(5, "five");
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.peek(), Some((&5, &"five")));
        assert_eq!(heap.pop(), Some((5, "five")));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_string_items() {
        let mut heap = BinomialHeapArena::<&str, i32>::new();

        heap.push(3, "c");
        heap.push(1, "a");
        heap.push(2, "b");

        assert_eq!(heap.peek().unwrap().1, &"a");
        assert_eq!(heap.pop().unwrap().1, "a");
        assert_eq!(heap.pop().unwrap().1, "b");
        assert_eq!(heap.pop().unwrap().1, "c");
    }
}

// Specialized arena decrease-key tests (excluding test_merge_with_handles due to arena limitation)
#[cfg(feature = "arena-storage")]
mod binomial_arena_decrease {
    use rust_advanced_heaps::binomial::BinomialHeapArena;
    use rust_advanced_heaps::traits::HeapError;
    use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

    #[test]
    fn test_decrease_key_operations() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();

        let _h1 = heap.push_with_handle(100, 1);
        let h2 = heap.push_with_handle(200, 2);
        let _h3 = heap.push_with_handle(300, 3);
        let h4 = heap.push_with_handle(400, 4);

        assert_eq!(heap.peek(), Some((&100, &1)));

        assert!(heap.decrease_key(&h2, 50).is_ok());
        assert_eq!(heap.peek(), Some((&50, &2)));

        assert!(heap.decrease_key(&h4, 25).is_ok());
        assert_eq!(heap.peek(), Some((&25, &4)));

        assert!(heap.decrease_key(&h4, 1).is_ok());
        assert_eq!(heap.peek(), Some((&1, &4)));

        assert_eq!(heap.pop(), Some((1, 4)));
        assert_eq!(heap.pop(), Some((50, 2)));
        assert_eq!(heap.pop(), Some((100, 1)));
        assert_eq!(heap.pop(), Some((300, 3)));
    }

    #[test]
    fn test_multiple_decrease_keys() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..20 {
            handles.push(heap.push_with_handle((i + 1) * 100, i));
        }

        for (i, handle) in handles.iter().enumerate() {
            assert!(heap.decrease_key(handle, i as i32).is_ok());
        }

        assert_eq!(heap.peek(), Some((&0, &0)));

        for i in 0..20 {
            assert_eq!(heap.pop(), Some((i, i)));
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn test_stress_operations() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..100 {
            handles.push(heap.push_with_handle(i * 2, i));
        }

        for i in (0..100).step_by(3) {
            assert!(heap.decrease_key(&handles[i], i as i32 * 2 - 1).is_ok());
        }

        for _ in 0..20 {
            heap.pop();
        }

        assert!(!heap.is_empty());
        assert!(heap.peek().is_some());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 80);
    }

    #[test]
    fn test_single_element_decrease() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let handle = heap.push_with_handle(42, 99);

        assert_eq!(heap.len(), 1);
        assert_eq!(heap.peek(), Some((&42, &99)));

        assert!(heap.decrease_key(&handle, 10).is_ok());
        assert_eq!(heap.peek(), Some((&10, &99)));

        assert_eq!(heap.pop(), Some((10, 99)));
        assert!(heap.is_empty());
    }

    #[test]
    fn test_decrease_key_same() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let handle = heap.push_with_handle(10, 1);

        assert_eq!(
            heap.decrease_key(&handle, 10),
            Err(HeapError::PriorityNotDecreased)
        );

        assert_eq!(heap.peek(), Some((&10, &1)));
    }

    #[test]
    fn test_multiple_decrease_same() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let handle = heap.push_with_handle(1000, 1);

        assert!(heap.decrease_key(&handle, 500).is_ok());
        assert_eq!(heap.peek(), Some((&500, &1)));

        assert!(heap.decrease_key(&handle, 250).is_ok());
        assert_eq!(heap.peek(), Some((&250, &1)));

        assert!(heap.decrease_key(&handle, 100).is_ok());
        assert_eq!(heap.peek(), Some((&100, &1)));

        assert!(heap.decrease_key(&handle, 50).is_ok());
        assert_eq!(heap.peek(), Some((&50, &1)));

        assert!(heap.decrease_key(&handle, 1).is_ok());
        assert_eq!(heap.peek(), Some((&1, &1)));
    }

    #[test]
    fn test_decrease_key_new_min() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let h1 = heap.push_with_handle(100, 1);
        let h2 = heap.push_with_handle(200, 2);
        let h3 = heap.push_with_handle(300, 3);

        assert!(heap.decrease_key(&h3, 150).is_ok());
        assert_eq!(heap.peek(), Some((&100, &1)));

        assert!(heap.decrease_key(&h2, 50).is_ok());
        assert_eq!(heap.peek(), Some((&50, &2)));

        assert!(heap.decrease_key(&h1, 25).is_ok());
        assert_eq!(heap.peek(), Some((&25, &1)));
    }

    #[test]
    fn test_decrease_key_selective() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..30 {
            handles.push(heap.push_with_handle((i + 1) * 100, i));
        }

        for (i, handle) in handles.iter().enumerate() {
            if i % 2 == 0 {
                assert!(heap.decrease_key(handle, i as i32 * 10).is_ok());
            }
        }

        let min = heap.peek();
        assert!(min.is_some());
        assert!(*min.unwrap().0 < 100);

        let mut last = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(priority >= last);
            last = priority;
        }
    }

    #[test]
    fn test_all_same_priority() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..10 {
            handles.push(heap.push_with_handle((i + 1) * 10, i));
        }

        for handle in &handles {
            assert!(heap.decrease_key(handle, 5).is_ok());
        }

        assert_eq!(heap.peek().unwrap().0, &5);

        let mut seen = std::collections::HashSet::new();
        while let Some((priority, item)) = heap.pop() {
            assert_eq!(priority, 5);
            assert!(seen.insert(item));
        }
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn test_decrease_to_negative() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let h1 = heap.push_with_handle(10, 1);
        let _h2 = heap.push_with_handle(20, 2);

        assert!(heap.decrease_key(&h1, -5).is_ok());
        assert_eq!(heap.peek(), Some((&-5, &1)));
        assert_eq!(heap.pop(), Some((-5, 1)));
    }

    #[test]
    fn test_heap_property() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..50 {
            handles.push(heap.push_with_handle((i * 17 + 23) % 1000, i));
        }

        for i in (0..50).step_by(7) {
            if let Some(handle) = handles.get(i) {
                let current_min = heap.peek().unwrap().0;
                let new_priority = (*current_min / 2).max(1);
                assert!(heap.decrease_key(handle, new_priority).is_ok());
            }
        }

        let mut last_priority = i32::MIN;
        while let Some((priority, _)) = heap.pop() {
            assert!(priority >= last_priority);
            last_priority = priority;
        }
    }

    // NOTE: test_merge_with_handles is excluded for arena storage because
    // handles from merged heaps become invalid when nodes are moved to new storage.
    // This is a known limitation of arena-based storage.

    #[test]
    fn test_very_large_sequence() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..1000 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        for i in (0..1000).step_by(10) {
            assert!(heap.decrease_key(&handles[i], (i as i32) - 1).is_ok());
        }

        for _ in 0..100 {
            heap.pop();
        }

        for i in 1000..1200 {
            heap.push(i * 10, i);
        }

        assert!(!heap.is_empty());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        assert_eq!(count, 1100);
    }

    #[test]
    fn test_complex_sequence() {
        let mut heap = BinomialHeapArena::<i32, i32>::new();
        let mut handles = Vec::new();

        for i in 0..20 {
            if i % 2 == 0 {
                handles.push(heap.push_with_handle(i * 10, i + 1000));
            }
        }

        for (i, handle) in handles.iter().enumerate().skip(3).step_by(3) {
            let new_priority = (i as i32 * 5).max(1);
            if new_priority < i as i32 * 10 {
                assert!(heap.decrease_key(handle, new_priority).is_ok());
            }
        }

        for _ in 0..3 {
            if heap.pop().is_none() {
                break;
            }
        }

        for i in 20..25 {
            heap.push(i * 10, i + 2000);
        }

        assert!(!heap.is_empty());
        assert!(heap.peek().is_some());

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
            if count > 50 {
                break;
            }
        }

        assert!(heap.is_empty() || heap.peek().is_some());
    }
}

// Skip List Heap
base_heap_tests!(
    skiplist_base,
    rust_advanced_heaps::skiplist::SkipListHeap<&'static str, i32>
);
decrease_key_heap_tests!(skiplist_decrease, rust_advanced_heaps::skiplist::SkipListHeap<i32, i32>);

// Hollow Heap
base_heap_tests!(
    hollow_base,
    rust_advanced_heaps::hollow::HollowHeap<&'static str, i32>
);
decrease_key_heap_tests!(hollow_decrease, rust_advanced_heaps::hollow::HollowHeap<i32, i32>);

// ============================================================================
// Radix Heap - specialized tests (monotone, unsigned keys only)
// ============================================================================
// Note: RadixHeap cannot use the standard base_heap_tests or decrease_key_heap_tests
// because it:
// 1. Only supports unsigned integer priorities (not i32)
// 2. Is monotone: cannot insert keys < last extracted minimum
// 3. Uses non-negative priorities only
//
// See src/radix.rs for comprehensive unit tests specific to RadixHeap.

mod radix_specialized {
    use rust_advanced_heaps::radix::RadixHeap;
    use rust_advanced_heaps::traits::HeapError;
    use rust_advanced_heaps::{DecreaseKeyHeap, Heap, MergeableHeap};

    #[test]
    fn test_empty_heap() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();
        assert!(heap.is_empty());
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.peek(), None);
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_basic_operations() {
        let mut heap: RadixHeap<&str, u32> = RadixHeap::new();
        heap.push(5, "five");
        heap.push(1, "one");
        heap.push(10, "ten");
        heap.push(3, "three");

        assert!(!heap.is_empty());
        assert_eq!(heap.len(), 4);

        assert_eq!(heap.pop(), Some((1, "one")));
        assert_eq!(heap.pop(), Some((3, "three")));
        assert_eq!(heap.pop(), Some((5, "five")));
        assert_eq!(heap.pop(), Some((10, "ten")));
        assert_eq!(heap.pop(), None);
        assert!(heap.is_empty());
    }

    #[test]
    fn test_merge_operations() {
        let mut heap1: RadixHeap<&str, u32> = RadixHeap::new();
        heap1.push(5, "five");
        heap1.push(1, "one");

        let mut heap2: RadixHeap<&str, u32> = RadixHeap::new();
        heap2.push(10, "ten");
        heap2.push(3, "three");

        heap1.merge(heap2);

        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.pop(), Some((1, "one")));
        assert_eq!(heap1.pop(), Some((3, "three")));
        assert_eq!(heap1.pop(), Some((5, "five")));
        assert_eq!(heap1.pop(), Some((10, "ten")));
    }

    #[test]
    fn test_decrease_key_operations() {
        let mut heap: RadixHeap<u32, u32> = RadixHeap::new();

        let _h1 = heap.push_with_handle(100, 1);
        let h2 = heap.push_with_handle(200, 2);
        let _h3 = heap.push_with_handle(300, 3);
        let h4 = heap.push_with_handle(400, 4);

        assert!(heap.decrease_key(&h2, 50).is_ok());
        assert!(heap.decrease_key(&h4, 25).is_ok());
        assert!(heap.decrease_key(&h4, 1).is_ok());

        assert_eq!(heap.pop(), Some((1, 4)));
        assert_eq!(heap.pop(), Some((50, 2)));
        assert_eq!(heap.pop(), Some((100, 1)));
        assert_eq!(heap.pop(), Some((300, 3)));
    }

    #[test]
    fn test_multiple_decrease_keys() {
        let mut heap: RadixHeap<u32, u32> = RadixHeap::new();
        let mut handles = Vec::new();

        for i in 0u32..20 {
            handles.push(heap.push_with_handle((i + 1) * 100, i));
        }

        // Decrease all to their index value
        for (i, handle) in handles.iter().enumerate() {
            assert!(heap.decrease_key(handle, i as u32).is_ok());
        }

        // Verify extraction order
        for i in 0u32..20 {
            assert_eq!(heap.pop(), Some((i, i)));
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn test_decrease_key_same() {
        let mut heap: RadixHeap<u32, u32> = RadixHeap::new();
        let handle = heap.push_with_handle(10, 1);

        assert_eq!(
            heap.decrease_key(&handle, 10),
            Err(HeapError::PriorityNotDecreased)
        );

        assert_eq!(heap.peek(), Some((&10, &1)));
    }

    #[test]
    fn test_dijkstra_pattern() {
        // Simulate Dijkstra's algorithm pattern
        let mut heap: RadixHeap<u32, u32> = RadixHeap::new();

        // Insert source with distance 0
        let _h0 = heap.push_with_handle(0, 0);

        // Insert neighbors with initial distances
        let h1 = heap.push_with_handle(10, 1);
        let h2 = heap.push_with_handle(5, 2);
        let _h3 = heap.push_with_handle(u32::MAX, 3);

        // Extract minimum (node 0, distance 0)
        assert_eq!(heap.pop(), Some((0, 0)));

        // Relax edge 0->1: new distance = 0 + 3 = 3 < 10
        heap.decrease_key(&h1, 3).unwrap();

        // Extract minimum (node 1, distance 3)
        assert_eq!(heap.pop(), Some((3, 1)));

        // Relax edge 1->2: new distance = 3 + 1 = 4 < 5
        heap.decrease_key(&h2, 4).unwrap();

        // Extract node 2
        assert_eq!(heap.pop(), Some((4, 2)));
    }

    #[test]
    fn test_large_sequence() {
        let mut heap: RadixHeap<u32, u32> = RadixHeap::new();
        let mut handles = Vec::new();

        for i in 0u32..500 {
            handles.push(heap.push_with_handle(i * 10, i));
        }

        // Decrease every 10th element
        for i in (0..500).step_by(10) {
            let new_pri = (i as u32).saturating_sub(1);
            if new_pri < i as u32 * 10 {
                assert!(heap.decrease_key(&handles[i], new_pri).is_ok());
            }
        }

        // Pop some elements
        for _ in 0..50 {
            heap.pop();
        }

        // Add more elements (must respect monotone property)
        // After popping, last_min is at least 0, so we can insert anything >= 0
        for i in 500u32..600 {
            heap.push(i * 10, i);
        }

        assert!(!heap.is_empty());

        // Drain and verify order
        let mut last = 0u32;
        let mut count = 0;
        while let Some((priority, _)) = heap.pop() {
            assert!(priority >= last);
            last = priority;
            count += 1;
        }
        assert_eq!(count, 550);
    }

    #[test]
    fn test_u64_keys() {
        let mut heap: RadixHeap<&str, u64> = RadixHeap::new();
        heap.push(1_000_000_000_000u64, "trillion");
        heap.push(1_000_000u64, "million");
        heap.push(1_000u64, "thousand");

        assert_eq!(heap.pop(), Some((1_000, "thousand")));
        assert_eq!(heap.pop(), Some((1_000_000, "million")));
        assert_eq!(heap.pop(), Some((1_000_000_000_000, "trillion")));
    }
}
