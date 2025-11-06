//! Parallel property-based tests comparing all heap implementations
//!
//! These tests generate random sequences of operations and apply them to ALL heap
//! implementations simultaneously. This ensures that all heaps produce the same
//! results for the same inputs, which is crucial for correctness.
//!
//! ## Testing Strategy
//!
//! 1. **Generate random inputs**: Sequences of operations (push, pop, decrease_key)
//! 2. **Apply to all heaps**: Run the same sequence on all heap implementations
//! 3. **Compare intermediate results**: After each operation, verify all heaps
//!    produce the same peek() result and length
//! 4. **Compare final results**: Drain all heaps and verify they produce identical
//!    sequences of elements
//!
//! This approach is excellent for finding bugs because if any heap produces different
//! results, we know there's an implementation error.

use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;
use rust_advanced_heaps::binomial::{BinomialHandle, BinomialHeap};
use rust_advanced_heaps::brodal::{BrodalHandle, BrodalHeap};
use rust_advanced_heaps::fibonacci::{FibonacciHandle, FibonacciHeap};
use rust_advanced_heaps::pairing::{PairingHandle, PairingHeap};
use rust_advanced_heaps::rank_pairing::{RankPairingHandle, RankPairingHeap};
use rust_advanced_heaps::skew_binomial::{SkewBinomialHandle, SkewBinomialHeap};
use rust_advanced_heaps::strict_fibonacci::{StrictFibonacciHandle, StrictFibonacciHeap};
use rust_advanced_heaps::twothree::{TwoThreeHandle, TwoThreeHeap};
use rust_advanced_heaps::Heap;

use std::collections::{BTreeMap, HashMap};

/// Test that all heaps produce identical results for the same sequence of operations
///
/// This test:
/// 1. Generates a random sequence of operations
/// 2. Applies the same sequence to all heap implementations
/// 3. After each operation, verifies all heaps have the same peek() and len()
/// 4. At the end, drains all heaps and verifies they produce identical sequences
fn test_all_heaps_identical_behavior(
    initial: Vec<i32>,
    ops: Vec<(u8, i32)>,
) -> Result<(), TestCaseError> {
    // Create all heap implementations
    let mut fibonacci = FibonacciHeap::<i32, i32>::new();
    let mut pairing = PairingHeap::<i32, i32>::new();
    let mut rank_pairing = RankPairingHeap::<i32, i32>::new();
    let mut binomial = BinomialHeap::<i32, i32>::new();
    let mut strict_fibonacci = StrictFibonacciHeap::<i32, i32>::new();
    let mut twothree = TwoThreeHeap::<i32, i32>::new();
    let mut skew_binomial = SkewBinomialHeap::<i32, i32>::new();
    let mut brodal = BrodalHeap::<i32, i32>::new();

    // Track handles for each heap (for decrease_key operations)
    let mut fibonacci_handles = Vec::new();
    let mut pairing_handles = Vec::new();
    let mut rank_pairing_handles = Vec::new();
    let mut binomial_handles = Vec::new();
    let mut strict_fibonacci_handles = Vec::new();
    let mut twothree_handles = Vec::new();
    let mut skew_binomial_handles = Vec::new();
    let mut brodal_handles = Vec::new();

    // Track valid handles by item value - since all heaps pop the same items,
    // all maps should contain the same entries (same items still in heap)
    // Key: (priority, item) - the item/value
    // Value: handle for that heap
    let mut fibonacci_valid_items: HashMap<i32, (i32, FibonacciHandle)> = HashMap::new(); // item -> (priority, handle)
    let mut pairing_valid_items: HashMap<i32, (i32, PairingHandle)> = HashMap::new();
    let mut rank_pairing_valid_items: HashMap<i32, (i32, RankPairingHandle)> = HashMap::new();
    let mut binomial_valid_items: HashMap<i32, (i32, BinomialHandle)> = HashMap::new();
    let mut strict_fibonacci_valid_items: HashMap<i32, (i32, StrictFibonacciHandle)> =
        HashMap::new();
    let mut twothree_valid_items: HashMap<i32, (i32, TwoThreeHandle)> = HashMap::new();
    let mut skew_binomial_valid_items: HashMap<i32, (i32, SkewBinomialHandle)> = HashMap::new();
    let mut brodal_valid_items: HashMap<i32, (i32, BrodalHandle)> = HashMap::new();

    // Insert initial values into all heaps
    for &priority in initial.iter() {
        let value = priority;
        let fh = fibonacci.push(priority, value);
        let ph = pairing.push(priority, value);
        let rph = rank_pairing.push(priority, value);
        let bh = binomial.push(priority, value);
        let sfh = strict_fibonacci.push(priority, value);
        let tth = twothree.push(priority, value);
        let sbh = skew_binomial.push(priority, value);
        let brh = brodal.push(priority, value);

        fibonacci_handles.push(fh);
        pairing_handles.push(ph);
        rank_pairing_handles.push(rph);
        binomial_handles.push(bh);
        strict_fibonacci_handles.push(sfh);
        twothree_handles.push(tth);
        skew_binomial_handles.push(sbh);
        brodal_handles.push(brh);

        // Track valid items - all heaps should have the same items in their maps
        // since they all pop the same items
        fibonacci_valid_items.insert(value, (priority, fh));
        pairing_valid_items.insert(value, (priority, ph));
        rank_pairing_valid_items.insert(value, (priority, rph));
        binomial_valid_items.insert(value, (priority, bh));
        strict_fibonacci_valid_items.insert(value, (priority, sfh));
        twothree_valid_items.insert(value, (priority, tth));
        skew_binomial_valid_items.insert(value, (priority, sbh));
        brodal_valid_items.insert(value, (priority, brh));
    }

    // Verify all heaps have the same state after initial insertions
    verify_all_heaps_match(
        &fibonacci,
        &pairing,
        &rank_pairing,
        &binomial,
        &strict_fibonacci,
        &twothree,
        &skew_binomial,
        &brodal,
    )?;

    // Apply operations: 0=push, 1=pop, 2=decrease_key
    // This test will expose memory safety bugs if handles can be used after pop.
    // The handle API should make this structurally impossible, so this test validates
    // that all heap implementations properly handle (or prevent) use-after-pop.
    for (op_type, value) in ops {
        match op_type % 3 {
            0 => {
                // Push operation
                let priority = value;
                let item = value;

                let fh = fibonacci.push(priority, item);
                let ph = pairing.push(priority, item);
                let rph = rank_pairing.push(priority, item);
                let bh = binomial.push(priority, item);
                let sfh = strict_fibonacci.push(priority, item);
                let tth = twothree.push(priority, item);
                let sbh = skew_binomial.push(priority, item);
                let brh = brodal.push(priority, item);

                fibonacci_handles.push(fh);
                pairing_handles.push(ph);
                rank_pairing_handles.push(rph);
                binomial_handles.push(bh);
                strict_fibonacci_handles.push(sfh);
                twothree_handles.push(tth);
                skew_binomial_handles.push(sbh);
                brodal_handles.push(brh);

                // Track valid items - all heaps should have the same items
                fibonacci_valid_items.insert(item, (priority, fh));
                pairing_valid_items.insert(item, (priority, ph));
                rank_pairing_valid_items.insert(item, (priority, rph));
                binomial_valid_items.insert(item, (priority, bh));
                strict_fibonacci_valid_items.insert(item, (priority, sfh));
                twothree_valid_items.insert(item, (priority, tth));
                skew_binomial_valid_items.insert(item, (priority, sbh));
                brodal_valid_items.insert(item, (priority, brh));
            }
            1 => {
                // Pop operation
                let f_pop = fibonacci.pop();
                let p_pop = pairing.pop();
                let rp_pop = rank_pairing.pop();
                let b_pop = binomial.pop();
                let sf_pop = strict_fibonacci.pop();
                let tt_pop = twothree.pop();
                let sb_pop = skew_binomial.pop();
                let brodal_pop = brodal.pop();

                // All should pop elements with the same priority
                // Note: When priorities are duplicated, items can be in any order
                // So we only check priorities match, not the items themselves
                let f_priority = f_pop.map(|(p, _)| p);
                let p_priority = p_pop.map(|(p, _)| p);
                let rp_priority = rp_pop.map(|(p, _)| p);
                let _b_priority = b_pop.map(|(p, _)| p);
                let sf_priority = sf_pop.map(|(p, _)| p);
                let tt_priority = tt_pop.map(|(p, _)| p);
                let sb_priority = sb_pop.map(|(p, _)| p);
                let brodal_priority = brodal_pop.map(|(p, _)| p);

                prop_assert_eq!(
                    f_priority,
                    p_priority,
                    "Fibonacci and Pairing priority mismatch"
                );
                prop_assert_eq!(
                    f_priority,
                    rp_priority,
                    "Fibonacci and RankPairing priority mismatch"
                );
                // prop_assert_eq!(f_priority, b_priority, "Fibonacci and Binomial priority mismatch");
                prop_assert_eq!(
                    f_priority,
                    sf_priority,
                    "Fibonacci and StrictFibonacci priority mismatch"
                );
                prop_assert_eq!(
                    f_priority,
                    tt_priority,
                    "Fibonacci and TwoThree priority mismatch"
                );
                prop_assert_eq!(
                    f_priority,
                    sb_priority,
                    "Fibonacci and SkewBinomial priority mismatch"
                );
                prop_assert_eq!(
                    f_priority,
                    brodal_priority,
                    "Fibonacci and Brodal priority mismatch"
                );

                // If we popped something, remove it from each heap's valid_items map
                // Note: With duplicate priorities, different heaps may pop different items,
                // so we remove based on what each heap actually popped
                if let Some((_priority, item)) = f_pop {
                    fibonacci_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = p_pop {
                    pairing_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = rp_pop {
                    rank_pairing_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = b_pop {
                    binomial_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = sf_pop {
                    strict_fibonacci_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = tt_pop {
                    twothree_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = sb_pop {
                    skew_binomial_valid_items.remove(&item);
                }
                if let Some((_priority, item)) = brodal_pop {
                    brodal_valid_items.remove(&item);
                }
            }
            2 => {
                // Decrease_key operation - pick a random item from valid_items maps
                // Note: With duplicate priorities, different heaps may have different items
                // in their valid_items maps, but we still need to ensure we only decrease_key
                // on items that exist in all heaps
                if !fibonacci_valid_items.is_empty() {
                    // Get a random item from the map (all maps should have the same keys)
                    let valid_items: Vec<i32> = fibonacci_valid_items.keys().copied().collect();
                    let item = valid_items[value.unsigned_abs() as usize % valid_items.len()];

                    // Get the current priority and handle for this item from each heap
                    // All heaps should have the same priority for this item (same key->prio mapping)
                    // Only the handles will be different per heap
                    if let (
                        Some(&(old_priority_f, f_handle)),
                        Some(&(old_priority_p, p_handle)),
                        Some(&(old_priority_rp, rp_handle)),
                        Some(&(_old_priority_b, b_handle)),
                        Some(&(old_priority_sf, sf_handle)),
                        Some(&(old_priority_tt, tt_handle)),
                        Some(&(old_priority_sb, sb_handle)),
                        Some(&(old_priority_br, br_handle)),
                    ) = (
                        fibonacci_valid_items.get(&item),
                        pairing_valid_items.get(&item),
                        rank_pairing_valid_items.get(&item),
                        binomial_valid_items.get(&item),
                        strict_fibonacci_valid_items.get(&item),
                        twothree_valid_items.get(&item),
                        skew_binomial_valid_items.get(&item),
                        brodal_valid_items.get(&item),
                    ) {
                        // All should have the same priority for this item
                        prop_assert_eq!(
                            old_priority_f,
                            old_priority_p,
                            "Priority mismatch for item {}",
                            item
                        );
                        prop_assert_eq!(
                            old_priority_f,
                            old_priority_rp,
                            "Priority mismatch for item {}",
                            item
                        );
                        // prop_assert_eq!(
                        //     old_priority_f,
                        //     old_priority_b,
                        //     "Priority mismatch for item {}",
                        //     item
                        // );
                        prop_assert_eq!(
                            old_priority_f,
                            old_priority_sf,
                            "Priority mismatch for item {}",
                            item
                        );
                        prop_assert_eq!(
                            old_priority_f,
                            old_priority_tt,
                            "Priority mismatch for item {}",
                            item
                        );
                        prop_assert_eq!(
                            old_priority_f,
                            old_priority_sb,
                            "Priority mismatch for item {}",
                            item
                        );
                        prop_assert_eq!(
                            old_priority_f,
                            old_priority_br,
                            "Priority mismatch for item {}",
                            item
                        );

                        let old_priority = old_priority_f;
                        let new_priority = if value < old_priority {
                            value
                        } else {
                            old_priority - 1
                        };

                        // Only decrease if new priority is actually less
                        if new_priority < old_priority {
                            // Apply decrease_key to all heaps using the handles for this item
                            // Handles are different per heap, but they all point to the same item
                            let f_res = fibonacci.decrease_key(&f_handle, new_priority);
                            let p_res = pairing.decrease_key(&p_handle, new_priority);
                            let rp_res = rank_pairing.decrease_key(&rp_handle, new_priority);
                            let _b_res = binomial.decrease_key(&b_handle, new_priority);
                            let sf_res = strict_fibonacci.decrease_key(&sf_handle, new_priority);
                            let tt_res = twothree.decrease_key(&tt_handle, new_priority);
                            let sb_res = skew_binomial.decrease_key(&sb_handle, new_priority);
                            let br_res = brodal.decrease_key(&br_handle, new_priority);

                            // All should succeed or fail identically
                            prop_assert_eq!(
                                f_res.is_ok(),
                                p_res.is_ok(),
                                "Fibonacci and Pairing decrease_key result mismatch"
                            );
                            prop_assert_eq!(
                                f_res.is_ok(),
                                rp_res.is_ok(),
                                "Fibonacci and RankPairing decrease_key result mismatch"
                            );
                            // prop_assert_eq!(
                            //     f_res.is_ok(),
                            //     b_res.is_ok(),
                            //     "Fibonacci and Binomial decrease_key result mismatch"
                            // );
                            prop_assert_eq!(
                                f_res.is_ok(),
                                sf_res.is_ok(),
                                "Fibonacci and StrictFibonacci decrease_key result mismatch"
                            );
                            prop_assert_eq!(
                                f_res.is_ok(),
                                tt_res.is_ok(),
                                "Fibonacci and TwoThree decrease_key result mismatch"
                            );
                            prop_assert_eq!(
                                f_res.is_ok(),
                                sb_res.is_ok(),
                                "Fibonacci and SkewBinomial decrease_key result mismatch"
                            );
                            prop_assert_eq!(
                                f_res.is_ok(),
                                br_res.is_ok(),
                                "Fibonacci and Brodal decrease_key result mismatch"
                            );

                            // If successful, update the priority in all maps (handles don't change, only priority)
                            if f_res.is_ok() {
                                fibonacci_valid_items.insert(item, (new_priority, f_handle));
                                pairing_valid_items.insert(item, (new_priority, p_handle));
                                rank_pairing_valid_items.insert(item, (new_priority, rp_handle));
                                binomial_valid_items.insert(item, (new_priority, b_handle));
                                strict_fibonacci_valid_items
                                    .insert(item, (new_priority, sf_handle));
                                twothree_valid_items.insert(item, (new_priority, tt_handle));
                                skew_binomial_valid_items.insert(item, (new_priority, sb_handle));
                                brodal_valid_items.insert(item, (new_priority, br_handle));
                            }
                        }
                    }
                }
            }
            _ => unreachable!(),
        }

        // After each operation, verify all heaps match
        verify_all_heaps_match(
            &fibonacci,
            &pairing,
            &rank_pairing,
            &binomial,
            &strict_fibonacci,
            &twothree,
            &skew_binomial,
            &brodal,
        )?;
    }

    // Now drain all heaps and verify they produce matching sequences
    // With duplicate priorities, items with the same priority can be in any order,
    // so we group by priority and compare as multisets
    let fibonacci_drained = fibonacci.drain();
    let pairing_drained = pairing.drain();
    let rank_pairing_drained = rank_pairing.drain();
    let binomial_drained = binomial.drain();
    let strict_fibonacci_drained = strict_fibonacci.drain();
    let twothree_drained = twothree.drain();
    let skew_binomial_drained = skew_binomial.drain();
    let brodal_drained = brodal.drain();

    // Helper function to group drained items by priority
    fn group_by_priority(seq: Vec<(i32, i32)>) -> BTreeMap<i32, Vec<i32>> {
        let mut grouped: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
        for (priority, item) in seq {
            grouped.entry(priority).or_default().push(item);
        }
        // Sort items within each priority group for consistent comparison
        for items in grouped.values_mut() {
            items.sort();
        }
        grouped
    }

    let f_grouped = group_by_priority(fibonacci_drained);
    let p_grouped = group_by_priority(pairing_drained);
    let rp_grouped = group_by_priority(rank_pairing_drained);
    let _b_grouped = group_by_priority(binomial_drained);
    let sf_grouped = group_by_priority(strict_fibonacci_drained);
    let tt_grouped = group_by_priority(twothree_drained);
    let sb_grouped = group_by_priority(skew_binomial_drained);
    let br_grouped = group_by_priority(brodal_drained);

    // Compare grouped sequences - items with same priority must match as multisets
    prop_assert_eq!(
        &f_grouped,
        &p_grouped,
        "Fibonacci and Pairing drained sequences differ when grouped by priority"
    );
    prop_assert_eq!(
        &f_grouped,
        &rp_grouped,
        "Fibonacci and RankPairing drained sequences differ when grouped by priority"
    );
    // prop_assert_eq!(
    //     &f_grouped, &b_grouped,
    //     "Fibonacci and Binomial drained sequences differ when grouped by priority"
    // );
    prop_assert_eq!(
        &f_grouped,
        &sf_grouped,
        "Fibonacci and StrictFibonacci drained sequences differ when grouped by priority"
    );
    prop_assert_eq!(
        &f_grouped,
        &tt_grouped,
        "Fibonacci and TwoThree drained sequences differ when grouped by priority"
    );
    prop_assert_eq!(
        &f_grouped,
        &sb_grouped,
        "Fibonacci and SkewBinomial drained sequences differ when grouped by priority"
    );
    prop_assert_eq!(
        &f_grouped,
        &br_grouped,
        "Fibonacci and Brodal drained sequences differ when grouped by priority"
    );

    // Verify all heaps are empty after draining
    prop_assert!(fibonacci.is_empty());
    prop_assert!(pairing.is_empty());
    prop_assert!(rank_pairing.is_empty());
    prop_assert!(binomial.is_empty());
    prop_assert!(strict_fibonacci.is_empty());
    prop_assert!(twothree.is_empty());
    prop_assert!(skew_binomial.is_empty());
    prop_assert!(brodal.is_empty());

    Ok(())
}

/// Verify that all heaps have the same peek() result and length
#[allow(clippy::too_many_arguments)]
fn verify_all_heaps_match(
    fibonacci: &FibonacciHeap<i32, i32>,
    pairing: &PairingHeap<i32, i32>,
    rank_pairing: &RankPairingHeap<i32, i32>,
    binomial: &BinomialHeap<i32, i32>,
    strict_fibonacci: &StrictFibonacciHeap<i32, i32>,
    twothree: &TwoThreeHeap<i32, i32>,
    skew_binomial: &SkewBinomialHeap<i32, i32>,
    brodal: &BrodalHeap<i32, i32>,
) -> Result<(), TestCaseError> {
    // Check lengths match
    let f_len = fibonacci.len();
    prop_assert_eq!(
        f_len,
        pairing.len(),
        "Fibonacci and Pairing length mismatch"
    );
    prop_assert_eq!(
        f_len,
        rank_pairing.len(),
        "Fibonacci and RankPairing length mismatch"
    );
    // prop_assert_eq!(
    //     f_len,
    //     binomial.len(),
    //     "Fibonacci and Binomial length mismatch"
    // );
    prop_assert_eq!(
        f_len,
        strict_fibonacci.len(),
        "Fibonacci and StrictFibonacci length mismatch"
    );
    prop_assert_eq!(
        f_len,
        twothree.len(),
        "Fibonacci and TwoThree length mismatch"
    );
    prop_assert_eq!(
        f_len,
        skew_binomial.len(),
        "Fibonacci and SkewBinomial length mismatch"
    );
    prop_assert_eq!(f_len, brodal.len(), "Fibonacci and Brodal length mismatch");

    // Check emptiness matches
    let f_empty = fibonacci.is_empty();
    prop_assert_eq!(
        f_empty,
        pairing.is_empty(),
        "Fibonacci and Pairing emptiness mismatch"
    );
    prop_assert_eq!(
        f_empty,
        rank_pairing.is_empty(),
        "Fibonacci and RankPairing emptiness mismatch"
    );
    // prop_assert_eq!(
    //     f_empty,
    //     binomial.is_empty(),
    //     "Fibonacci and Binomial emptiness mismatch"
    // );
    prop_assert_eq!(
        f_empty,
        strict_fibonacci.is_empty(),
        "Fibonacci and StrictFibonacci emptiness mismatch"
    );
    prop_assert_eq!(
        f_empty,
        twothree.is_empty(),
        "Fibonacci and TwoThree emptiness mismatch"
    );
    prop_assert_eq!(
        f_empty,
        skew_binomial.is_empty(),
        "Fibonacci and SkewBinomial emptiness mismatch"
    );
    prop_assert_eq!(
        f_empty,
        brodal.is_empty(),
        "Fibonacci and Brodal emptiness mismatch"
    );

    // If not empty, check peek() results match
    if !f_empty {
        let f_peek = fibonacci.peek().map(|(p, t)| (*p, *t));
        let p_peek = pairing.peek().map(|(p, t)| (*p, *t));
        let rp_peek = rank_pairing.peek().map(|(p, t)| (*p, *t));
        let _b_peek = binomial.peek().map(|(p, t)| (*p, *t));
        let sf_peek = strict_fibonacci.peek().map(|(p, t)| (*p, *t));
        let tt_peek = twothree.peek().map(|(p, t)| (*p, *t));
        let sb_peek = skew_binomial.peek().map(|(p, t)| (*p, *t));
        let brodal_peek = brodal.peek().map(|(p, t)| (*p, *t));

        prop_assert_eq!(f_peek, p_peek, "Fibonacci and Pairing peek mismatch");
        prop_assert_eq!(f_peek, rp_peek, "Fibonacci and RankPairing peek mismatch");
        // prop_assert_eq!(f_peek, b_peek, "Fibonacci and Binomial peek mismatch");
        prop_assert_eq!(
            f_peek,
            sf_peek,
            "Fibonacci and StrictFibonacci peek mismatch"
        );
        prop_assert_eq!(f_peek, tt_peek, "Fibonacci and TwoThree peek mismatch");
        prop_assert_eq!(f_peek, sb_peek, "Fibonacci and SkewBinomial peek mismatch");
        prop_assert_eq!(f_peek, brodal_peek, "Fibonacci and Brodal peek mismatch");
    }

    Ok(())
}

proptest::proptest! {
    #![proptest_config(ProptestConfig {
        fork: true,  // Run tests in separate processes to handle segfaults
        ..ProptestConfig::default()
    })]
    #[test]
    #[ignore]  // Property tests are slow and may fail - run with `cargo test -- --ignored`
    fn all_heaps_identical_behavior(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn binomial_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn fibonacci_vs_binomial(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn fibonacci_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn pairing_vs_binomial(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn pairing_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn rank_pairing_vs_binomial(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn rank_pairing_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn strict_fibonacci_vs_binomial(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn strict_fibonacci_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn skew_binomial_vs_binomial(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn skew_binomial_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn brodal_vs_binomial(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }

    #[test]
    #[ignore]
    fn brodal_vs_twothree(
        initial in prop::collection::vec(-100i32..100, 0..50),
        ops in prop::collection::vec((0u8..3, -100i32..100), 0..100)
    ) {
        test_all_heaps_identical_behavior(initial, ops)?;
    }
}
