//! Hardware Performance Counter Benchmarks
//!
//! This benchmark uses Linux perf_event to measure hardware counters like
//! cache misses, branch mispredictions, and instructions per cycle.
//!
//! ## Running
//!
//! ```bash
//! # Requires perf_event_paranoid <= 1 or CAP_PERFMON
//! sudo sysctl kernel.perf_event_paranoid=1
//! cargo bench --features perf-counters --bench heap_perf
//! ```
//!
//! ## Metrics
//!
//! - Instructions: Total instructions executed
//! - Cycles: CPU cycles consumed
//! - IPC: Instructions per cycle (higher = more efficient)
//! - Cache misses: LLC (last-level cache) misses
//! - Branch misses: Mispredicted branches

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod perf_benchmarks {
    use perf_event::events::Hardware;
    use perf_event::{Builder, Group};
    use rust_advanced_heaps::binomial::BinomialHeap;
    use rust_advanced_heaps::fibonacci::FibonacciHeap;
    use rust_advanced_heaps::hollow::HollowHeap;
    use rust_advanced_heaps::pairing::PairingHeap;
    // RadixHeap requires special handling (monotone property) - not included in generic benchmarks
    // use rust_advanced_heaps::radix::RadixHeap;
    use rust_advanced_heaps::rank_pairing::RankPairingHeap;
    use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
    use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
    use rust_advanced_heaps::twothree::TwoThreeHeap;
    use rust_advanced_heaps::{DecreaseKeyHeap, Heap};
    use std::hint::black_box;

    /// Metrics collected from hardware performance counters
    #[derive(Debug, Clone)]
    pub struct PerfMetrics {
        pub instructions: u64,
        pub cycles: u64,
        pub cache_references: u64,
        pub cache_misses: u64,
        pub branches: u64,
        pub branch_misses: u64,
    }

    impl PerfMetrics {
        pub fn ipc(&self) -> f64 {
            self.instructions as f64 / self.cycles as f64
        }

        pub fn cache_miss_rate(&self) -> f64 {
            if self.cache_references == 0 {
                0.0
            } else {
                self.cache_misses as f64 / self.cache_references as f64 * 100.0
            }
        }

        pub fn branch_miss_rate(&self) -> f64 {
            if self.branches == 0 {
                0.0
            } else {
                self.branch_misses as f64 / self.branches as f64 * 100.0
            }
        }
    }

    impl std::fmt::Display for PerfMetrics {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Instructions: {:>12}, Cycles: {:>12}, IPC: {:.2}, Cache miss: {:.2}%, Branch miss: {:.2}%",
                self.instructions,
                self.cycles,
                self.ipc(),
                self.cache_miss_rate(),
                self.branch_miss_rate()
            )
        }
    }

    /// Measure a function with hardware performance counters
    pub fn measure_perf<F, R>(f: F) -> (R, PerfMetrics)
    where
        F: FnOnce() -> R,
    {
        let mut group = Group::new().unwrap_or_else(|e| {
            eprintln!("\nError: Failed to create perf group: {}", e);
            eprintln!(
                "\nThis usually means you don't have permission to access hardware counters."
            );
            eprintln!("To fix this, try one of:");
            eprintln!("  1. sudo sysctl kernel.perf_event_paranoid=1");
            eprintln!("  2. Run as root (not recommended)");
            eprintln!("  3. Set CAP_PERFMON capability on the binary");
            eprintln!();
            std::process::exit(1);
        });

        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()
            .expect("Failed to create cycles counter");

        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()
            .expect("Failed to create instructions counter");

        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()
            .expect("Failed to create cache_refs counter");

        let cache_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()
            .expect("Failed to create cache_misses counter");

        let branches = Builder::new()
            .group(&mut group)
            .kind(Hardware::BRANCH_INSTRUCTIONS)
            .build()
            .expect("Failed to create branches counter");

        let branch_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::BRANCH_MISSES)
            .build()
            .expect("Failed to create branch_misses counter");

        group.enable().expect("Failed to enable perf group");
        let result = f();
        group.disable().expect("Failed to disable perf group");

        let counts = group.read().expect("Failed to read perf counters");

        let metrics = PerfMetrics {
            instructions: counts[&instructions],
            cycles: counts[&cycles],
            cache_references: counts[&cache_refs],
            cache_misses: counts[&cache_misses],
            branches: counts[&branches],
            branch_misses: counts[&branch_misses],
        };

        (result, metrics)
    }

    /// Run a benchmark multiple times and return average metrics
    pub fn benchmark_avg<F>(iterations: usize, mut f: F) -> PerfMetrics
    where
        F: FnMut(),
    {
        let mut total = PerfMetrics {
            instructions: 0,
            cycles: 0,
            cache_references: 0,
            cache_misses: 0,
            branches: 0,
            branch_misses: 0,
        };

        for _ in 0..iterations {
            let (_, metrics) = measure_perf(&mut f);
            total.instructions += metrics.instructions;
            total.cycles += metrics.cycles;
            total.cache_references += metrics.cache_references;
            total.cache_misses += metrics.cache_misses;
            total.branches += metrics.branches;
            total.branch_misses += metrics.branch_misses;
        }

        PerfMetrics {
            instructions: total.instructions / iterations as u64,
            cycles: total.cycles / iterations as u64,
            cache_references: total.cache_references / iterations as u64,
            cache_misses: total.cache_misses / iterations as u64,
            branches: total.branches / iterations as u64,
            branch_misses: total.branch_misses / iterations as u64,
        }
    }

    // ========================================================================
    // Benchmark workloads
    // ========================================================================

    /// Push N elements, then pop all - tests basic heap operations
    fn workload_push_pop<H: Heap<i32, u32>>(n: usize) {
        let mut heap = H::new();
        for i in 0..n as u32 {
            heap.push(i, i as i32);
        }
        while heap.pop().is_some() {}
    }

    /// Push N elements with decrease_key on half - tests decrease_key locality
    fn workload_decrease_key<H: DecreaseKeyHeap<i32, u32>>(n: usize) {
        let mut heap = H::new();
        let mut handles = Vec::with_capacity(n);

        // Push all elements
        for i in 0..n as u32 {
            handles.push(heap.push_with_handle(i * 2, i as i32));
        }

        // Decrease key on every other element
        for (i, handle) in handles.iter().enumerate().step_by(2) {
            let new_priority = (i as u32) * 2 - 1;
            let _ = heap.decrease_key(handle, new_priority);
        }

        // Pop all
        while heap.pop().is_some() {}
    }

    /// Dijkstra-like workload: push, decrease_key, pop interleaved
    fn workload_dijkstra_pattern<H: DecreaseKeyHeap<usize, u32>>(n: usize) {
        let mut heap = H::new();
        let mut handles = Vec::with_capacity(n);

        // Initial push
        for i in 0..n {
            handles.push(heap.push_with_handle(u32::MAX, i));
        }

        // Set source distance to 0
        let _ = heap.decrease_key(&handles[0], 0);

        // Simulate relaxation
        let mut settled = 0;
        while let Some((_, node)) = heap.pop() {
            settled += 1;
            // Relax neighbors (simulate 3 neighbors per node)
            for offset in 1..=3 {
                let neighbor = (node + offset) % n;
                if neighbor < handles.len() {
                    let new_dist = settled as u32 + offset as u32;
                    let _ = heap.decrease_key(&handles[neighbor], new_dist);
                }
            }
        }
    }

    // ========================================================================
    // Benchmark runner
    // ========================================================================

    macro_rules! bench_heap {
        ($name:expr, $heap:ty, $workload:ident, $n:expr, $iters:expr) => {{
            let metrics = benchmark_avg($iters, || {
                $workload::<$heap>($n);
            });
            println!("{:25} | {}", $name, metrics);
            metrics
        }};
    }

    pub fn run_benchmarks() {
        const N: usize = 10_000;
        const ITERS: usize = 10;

        println!("\n=== Push/Pop Workload (N={}) ===\n", N);
        println!(
            "{:25} | {:>12} | {:>12} | {:>6} | {:>12} | {:>12}",
            "Heap", "Instructions", "Cycles", "IPC", "Cache Miss%", "Branch Miss%"
        );
        println!("{:-<100}", "");

        bench_heap!(
            "SimpleBinaryHeap",
            SimpleBinaryHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "FibonacciHeap",
            FibonacciHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "PairingHeap",
            PairingHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "HollowHeap",
            HollowHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "BinomialHeap",
            BinomialHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "RankPairingHeap",
            RankPairingHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "TwoThreeHeap",
            TwoThreeHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "SkewBinomialHeap",
            SkewBinomialHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );
        bench_heap!(
            "StrictFibonacciHeap",
            StrictFibonacciHeap<i32, u32>,
            workload_push_pop,
            N,
            ITERS
        );

        println!("\n=== Decrease-Key Workload (N={}) ===\n", N);
        println!(
            "{:25} | {:>12} | {:>12} | {:>6} | {:>12} | {:>12}",
            "Heap", "Instructions", "Cycles", "IPC", "Cache Miss%", "Branch Miss%"
        );
        println!("{:-<100}", "");

        bench_heap!(
            "FibonacciHeap",
            FibonacciHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "PairingHeap",
            PairingHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "HollowHeap",
            HollowHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "BinomialHeap",
            BinomialHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "RankPairingHeap",
            RankPairingHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "TwoThreeHeap",
            TwoThreeHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "SkewBinomialHeap",
            SkewBinomialHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );
        bench_heap!(
            "StrictFibonacciHeap",
            StrictFibonacciHeap<i32, u32>,
            workload_decrease_key,
            N,
            ITERS
        );

        println!("\n=== Dijkstra-like Workload (N={}) ===\n", N);
        println!(
            "{:25} | {:>12} | {:>12} | {:>6} | {:>12} | {:>12}",
            "Heap", "Instructions", "Cycles", "IPC", "Cache Miss%", "Branch Miss%"
        );
        println!("{:-<100}", "");

        bench_heap!(
            "FibonacciHeap",
            FibonacciHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "PairingHeap",
            PairingHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "HollowHeap",
            HollowHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "BinomialHeap",
            BinomialHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "RankPairingHeap",
            RankPairingHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "TwoThreeHeap",
            TwoThreeHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "SkewBinomialHeap",
            SkewBinomialHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );
        bench_heap!(
            "StrictFibonacciHeap",
            StrictFibonacciHeap<usize, u32>,
            workload_dijkstra_pattern,
            N,
            ITERS
        );

        // Suppress unused warning
        let _ = black_box(0);
    }
}

#[cfg(not(all(feature = "perf-counters", target_os = "linux")))]
fn main() {
    eprintln!("Perf benchmarks require:");
    eprintln!("  1. Linux operating system");
    eprintln!("  2. --features perf-counters flag");
    eprintln!();
    eprintln!("Run with: cargo bench --features perf-counters --bench heap_perf");
    std::process::exit(1);
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
    perf_benchmarks::run_benchmarks();
}
