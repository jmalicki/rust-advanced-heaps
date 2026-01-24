//! Cache benchmark for all heap types
//!
//! Run with:
//!   cargo build --release --bin cache_benchmark_all
//!   valgrind --tool=cachegrind ./target/release/cache_benchmark_all <heap_type> <n>
//!
//! heap_type: fibonacci, skew_binomial, pairing, rank_pairing, binomial, twothree

use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::rank_pairing::RankPairingHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

fn dijkstra_workload<H>(mut heap: H, n: usize) -> (usize, usize)
where
    H: Heap<usize, i64> + DecreaseKeyHeap<usize, i64>,
    <H as DecreaseKeyHeap<usize, i64>>::Handle: Clone,
{
    let mut handles: Vec<Option<H::Handle>> = Vec::with_capacity(n);

    // Phase 1: Insert all elements
    for i in 0..n {
        let priority = (i as i64 * 17) % (n as i64);
        let handle = heap.push_with_handle(priority, i);
        handles.push(Some(handle));
    }

    // Phase 2: Dijkstra-like workload
    let mut popped = 0;
    let mut decreased = 0;

    while let Some((priority, item)) = heap.pop() {
        popped += 1;
        handles[item] = None;

        // Simulate edge relaxation
        for offset in [1, 7, 13, 31, 97] {
            let neighbor = (item + offset) % n;
            if let Some(ref handle) = handles[neighbor] {
                let new_priority = priority + (offset as i64);
                if heap.decrease_key(handle, new_priority).is_ok() {
                    decreased += 1;
                }
            }
        }

        if popped >= n / 2 {
            break;
        }
    }

    // Phase 3: Drain
    while heap.pop().is_some() {
        popped += 1;
    }

    (popped, decreased)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let heap_type = args.get(1).map(|s| s.as_str()).unwrap_or("skew_binomial");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    println!("Running {} benchmark with {} elements", heap_type, n);

    let (popped, decreased) = match heap_type {
        "skew_binomial" => dijkstra_workload(SkewBinomialHeap::<usize, i64>::new(), n),
        "pairing" => dijkstra_workload(PairingHeap::<usize, i64>::new(), n),
        "rank_pairing" => dijkstra_workload(RankPairingHeap::<usize, i64>::new(), n),
        "binomial" => dijkstra_workload(BinomialHeap::<usize, i64>::new(), n),
        "twothree" => dijkstra_workload(TwoThreeHeap::<usize, i64>::new(), n),
        _ => {
            eprintln!(
                "Unknown heap type: {}. Use: skew_binomial, pairing, rank_pairing, binomial, twothree",
                heap_type
            );
            std::process::exit(1);
        }
    };

    println!("Completed: {} pops, {} decrease_keys", popped, decreased);
}
