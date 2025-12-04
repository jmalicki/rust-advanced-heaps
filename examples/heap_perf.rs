//! Hardware Performance Counter Benchmarks
//!
//! Measures CPU performance metrics to understand *why* certain heaps are faster:
//! - Instructions executed
//! - CPU cycles consumed
//! - IPC (instructions per cycle) - higher is better
//! - Cache miss rate - lower is better
//!
//! ## Running
//!
//! ```bash
//! # Enable perf access (requires Linux)
//! sudo sysctl kernel.perf_event_paranoid=1
//!
//! # Run the benchmarks
//! cargo run --features perf-counters --release --example heap_perf
//! ```

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
mod perf_benchmarks {
    use perf_event::events::Hardware;
    use perf_event::{Builder, Group};
    use rust_advanced_heaps::binomial::BinomialHeap;
    use rust_advanced_heaps::fibonacci::FibonacciHeap;
    use rust_advanced_heaps::hollow::HollowHeap;
    use rust_advanced_heaps::pairing::PairingHeap;
    // RankPairingHeap excluded due to known performance issue (#54)
    use rust_advanced_heaps::simple_binary::SimpleBinaryHeap;
    use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
    use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
    use rust_advanced_heaps::twothree::TwoThreeHeap;
    use rust_advanced_heaps::{DecreaseKeyHeap, Heap};
    use std::hint::black_box;

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
            if self.cycles == 0 {
                0.0
            } else {
                self.instructions as f64 / self.cycles as f64
            }
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

    fn measure_perf<F, R>(f: F) -> (R, PerfMetrics)
    where
        F: FnOnce() -> R,
    {
        let mut group = Group::new().unwrap_or_else(|e| {
            eprintln!("\nError: Failed to create perf group: {}", e);
            eprintln!(
                "\nThis usually means you don't have permission to access hardware counters."
            );
            eprintln!("To fix this, run: sudo sysctl kernel.perf_event_paranoid=1");
            std::process::exit(1);
        });

        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()
            .expect("cycles");

        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()
            .expect("instructions");

        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()
            .expect("cache_refs");

        let cache_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()
            .expect("cache_misses");

        let branches = Builder::new()
            .group(&mut group)
            .kind(Hardware::BRANCH_INSTRUCTIONS)
            .build()
            .expect("branches");

        let branch_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::BRANCH_MISSES)
            .build()
            .expect("branch_misses");

        group.enable().expect("enable");
        let result = f();
        group.disable().expect("disable");

        let counts = group.read().expect("read");

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

    fn benchmark_avg<F>(iterations: usize, mut f: F) -> PerfMetrics
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
            // Closure is needed to adapt FnMut to FnOnce for measure_perf
            #[allow(clippy::redundant_closure)]
            let (_, metrics) = measure_perf(|| f());
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
    // Workloads
    // ========================================================================

    fn workload_push_pop<H: Heap<i32, u32>>(n: usize) {
        let mut heap = H::new();
        for i in 0..n as u32 {
            heap.push(i, i as i32);
        }
        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        black_box(count);
    }

    fn workload_decrease_key<H: DecreaseKeyHeap<i32, u32>>(n: usize) {
        let mut heap = H::new();
        let mut handles = Vec::with_capacity(n);

        for i in 0..n as u32 {
            handles.push(heap.push_with_handle(i * 2, i as i32));
        }

        for (i, handle) in handles.iter().enumerate().step_by(2) {
            let new_priority = (i as u32).saturating_mul(2).saturating_sub(1);
            let _ = heap.decrease_key(handle, new_priority);
        }

        let mut count = 0;
        while heap.pop().is_some() {
            count += 1;
        }
        black_box(count);
    }

    fn workload_dijkstra<H: DecreaseKeyHeap<usize, u32>>(n: usize) {
        let mut heap = H::new();
        let mut handles = Vec::with_capacity(n);

        for i in 0..n {
            handles.push(heap.push_with_handle(u32::MAX, i));
        }
        let _ = heap.decrease_key(&handles[0], 0);

        let mut settled = 0;
        while let Some((_, node)) = heap.pop() {
            settled += 1;
            for offset in 1..=3 {
                let neighbor = (node + offset) % n;
                if neighbor < handles.len() {
                    let new_dist = settled as u32 + offset as u32;
                    let _ = heap.decrease_key(&handles[neighbor], new_dist);
                }
            }
        }
        black_box(settled);
    }

    // ========================================================================
    // Runner
    // ========================================================================

    macro_rules! bench_heap {
        ($name:expr, $heap:ty, $workload:ident, $n:expr, $iters:expr) => {{
            let metrics = benchmark_avg($iters, || $workload::<$heap>($n));
            println!(
                "{:22} | {:>11} | {:>10} | {:>5.2} | {:>8.2}% | {:>9.2}%",
                $name,
                metrics.instructions,
                metrics.cycles,
                metrics.ipc(),
                metrics.cache_miss_rate(),
                metrics.branch_miss_rate()
            );
        }};
    }

    pub fn run_benchmarks() {
        const N: usize = 10_000;
        const ITERS: usize = 5;

        println!(
            "\n=== Push/Pop Workload (N={}, {} iterations avg) ===\n",
            N, ITERS
        );
        println!(
            "{:22} | {:>11} | {:>10} | {:>5} | {:>9} | {:>10}",
            "Heap", "Instructions", "Cycles", "IPC", "Cache Miss", "Branch Miss"
        );
        println!("{:-<85}", "");

        bench_heap!("SimpleBinaryHeap", SimpleBinaryHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("FibonacciHeap", FibonacciHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("PairingHeap", PairingHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("HollowHeap", HollowHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("BinomialHeap", BinomialHeap<i32, u32>, workload_push_pop, N, ITERS);
        // Skip RankPairingHeap - known performance issue (#54)
        // bench_heap!("RankPairingHeap", RankPairingHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("TwoThreeHeap", TwoThreeHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("SkewBinomialHeap", SkewBinomialHeap<i32, u32>, workload_push_pop, N, ITERS);
        bench_heap!("StrictFibonacciHeap", StrictFibonacciHeap<i32, u32>, workload_push_pop, N, ITERS);

        println!(
            "\n=== Decrease-Key Workload (N={}, {} iterations avg) ===\n",
            N, ITERS
        );
        println!(
            "{:22} | {:>11} | {:>10} | {:>5} | {:>9} | {:>10}",
            "Heap", "Instructions", "Cycles", "IPC", "Cache Miss", "Branch Miss"
        );
        println!("{:-<85}", "");

        bench_heap!("FibonacciHeap", FibonacciHeap<i32, u32>, workload_decrease_key, N, ITERS);
        bench_heap!("PairingHeap", PairingHeap<i32, u32>, workload_decrease_key, N, ITERS);
        bench_heap!("HollowHeap", HollowHeap<i32, u32>, workload_decrease_key, N, ITERS);
        bench_heap!("BinomialHeap", BinomialHeap<i32, u32>, workload_decrease_key, N, ITERS);
        bench_heap!("TwoThreeHeap", TwoThreeHeap<i32, u32>, workload_decrease_key, N, ITERS);
        bench_heap!("SkewBinomialHeap", SkewBinomialHeap<i32, u32>, workload_decrease_key, N, ITERS);
        bench_heap!("StrictFibonacciHeap", StrictFibonacciHeap<i32, u32>, workload_decrease_key, N, ITERS);

        println!(
            "\n=== Dijkstra-like Workload (N={}, {} iterations avg) ===\n",
            N, ITERS
        );
        println!(
            "{:22} | {:>11} | {:>10} | {:>5} | {:>9} | {:>10}",
            "Heap", "Instructions", "Cycles", "IPC", "Cache Miss", "Branch Miss"
        );
        println!("{:-<85}", "");

        bench_heap!("FibonacciHeap", FibonacciHeap<usize, u32>, workload_dijkstra, N, ITERS);
        bench_heap!("PairingHeap", PairingHeap<usize, u32>, workload_dijkstra, N, ITERS);
        bench_heap!("HollowHeap", HollowHeap<usize, u32>, workload_dijkstra, N, ITERS);
        bench_heap!("BinomialHeap", BinomialHeap<usize, u32>, workload_dijkstra, N, ITERS);
        bench_heap!("TwoThreeHeap", TwoThreeHeap<usize, u32>, workload_dijkstra, N, ITERS);
        bench_heap!("SkewBinomialHeap", SkewBinomialHeap<usize, u32>, workload_dijkstra, N, ITERS);
        bench_heap!("StrictFibonacciHeap", StrictFibonacciHeap<usize, u32>, workload_dijkstra, N, ITERS);

        println!();
    }
}

#[cfg(not(all(feature = "perf-counters", target_os = "linux")))]
fn main() {
    eprintln!("Perf benchmarks require:");
    eprintln!("  1. Linux operating system");
    eprintln!("  2. --features perf-counters flag");
    eprintln!();
    eprintln!("Run with: cargo run --features perf-counters --release --example heap_perf");
    std::process::exit(1);
}

#[cfg(all(feature = "perf-counters", target_os = "linux"))]
fn main() {
    perf_benchmarks::run_benchmarks();
}
