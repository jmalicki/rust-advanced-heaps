//! Cache benchmark for Fibonacci heap
//!
//! Run with: cargo build --release --bin cache_benchmark
//! Then: valgrind --tool=cachegrind ./target/release/cache_benchmark

use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::{DecreaseKeyHeap, Heap};

fn main() {
    let n = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    println!("Running Fibonacci heap benchmark with {} elements", n);

    // Workload: Dijkstra-like pattern
    // - Insert all nodes
    // - Repeatedly: pop min, decrease some keys
    let mut heap = FibonacciHeap::<usize, i64>::new();
    let mut handles = Vec::with_capacity(n);

    // Phase 1: Insert all elements with initial priorities
    println!("Phase 1: Inserting {} elements...", n);
    for i in 0..n {
        let priority = (i as i64 * 17) % (n as i64); // Pseudo-random priorities
        let handle = heap.push_with_handle(priority, i);
        handles.push(Some(handle));
    }

    // Phase 2: Dijkstra-like workload
    // Pop minimum, then decrease keys of "neighbors"
    println!("Phase 2: Dijkstra-like workload...");
    let mut popped = 0;
    let mut decreased = 0;

    while let Some((priority, item)) = heap.pop() {
        popped += 1;
        handles[item] = None; // Mark as processed

        // Simulate edge relaxation: decrease keys of a few "neighbors"
        // Use a deterministic pattern based on the item
        for offset in [1, 7, 13, 31, 97] {
            let neighbor = (item + offset) % n;
            if let Some(ref handle) = handles[neighbor] {
                let new_priority = priority + (offset as i64);
                if heap.decrease_key(handle, new_priority).is_ok() {
                    decreased += 1;
                }
            }
        }

        // Stop after processing half the elements for reasonable benchmark time
        if popped >= n / 2 {
            break;
        }
    }

    // Phase 3: Drain remaining
    println!("Phase 3: Draining remaining elements...");
    while heap.pop().is_some() {
        popped += 1;
    }

    println!("Completed: {} pops, {} decrease_keys", popped, decreased);
}
