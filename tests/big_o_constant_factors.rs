//! Constant Factor Analysis for Heap Operations
//!
//! This module measures and reports the constant factors in Big-O complexity
//! for each heap implementation. While Big-O notation tells us how algorithms
//! scale, the constant factor determines actual performance for practical input sizes.
//!
//! ## Understanding Constant Factors
//!
//! For an algorithm with complexity O(f(n)), the actual time is:
//!     T(n) = c * f(n)
//!
//! where `c` is the constant factor. This module calculates:
//! - For O(n) batch operations: c = total_time / n
//! - For O(n log n) batch operations: c = total_time / (n * log₂(n))
//!
//! A lower constant factor means better real-world performance.

use ctor::ctor;
use rust_advanced_heaps::binomial::BinomialHeap;
use rust_advanced_heaps::fibonacci::FibonacciHeap;
use rust_advanced_heaps::pairing::PairingHeap;
use rust_advanced_heaps::skew_binomial::SkewBinomialHeap;
use rust_advanced_heaps::strict_fibonacci::StrictFibonacciHeap;
use rust_advanced_heaps::twothree::TwoThreeHeap;
use rust_advanced_heaps::Heap;
use std::time::{Duration, Instant};

/// Sets up the ENV for serial test execution
#[ctor]
fn setup_env() {
    std::env::set_var("RUST_TEST_THREADS", "1");
}

// ============================================================================
// Configuration
// ============================================================================

/// Number of warmup iterations before measurement
const WARMUP_ITERATIONS: usize = 2;

/// Number of measurement iterations for statistical reliability
const MEASUREMENT_ITERATIONS: usize = 5;

/// Test sizes to measure at (we use multiple to verify scaling)
const TEST_SIZES: [usize; 3] = [1000, 2000, 4000];

// ============================================================================
// Complexity Types
// ============================================================================

/// Represents the expected batch complexity for an operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchComplexity {
    /// O(n) - linear batch (O(1) amortized per element)
    Linear,
    /// O(n log n) - linearithmic batch (O(log n) per element)
    NLogN,
}

impl BatchComplexity {
    /// Calculate the constant factor given total time and input size
    fn constant_factor(&self, duration: Duration, n: usize) -> f64 {
        let nanos = duration.as_nanos() as f64;
        match self {
            BatchComplexity::Linear => nanos / (n as f64),
            BatchComplexity::NLogN => {
                let log_n = (n as f64).log2();
                nanos / (n as f64 * log_n)
            }
        }
    }

    /// String representation
    fn as_str(&self) -> &'static str {
        match self {
            BatchComplexity::Linear => "O(n)",
            BatchComplexity::NLogN => "O(n log n)",
        }
    }
}

// ============================================================================
// Measurement Infrastructure
// ============================================================================

/// Result of measuring an operation's constant factor
#[derive(Debug, Clone)]
pub struct ConstantFactorResult {
    pub heap_name: String,
    pub operation: String,
    pub complexity: BatchComplexity,
    /// Constant factor in nanoseconds per unit work
    pub constant_factor_ns: f64,
    /// Standard deviation of the constant factor
    pub std_dev_ns: f64,
    /// Minimum observed constant factor
    pub min_factor_ns: f64,
    /// Maximum observed constant factor
    pub max_factor_ns: f64,
    /// Raw timings at each test size (for verification)
    pub timings: Vec<(usize, Duration)>,
}

impl ConstantFactorResult {
    /// Format the constant factor with appropriate units
    pub fn format_factor(&self) -> String {
        if self.constant_factor_ns >= 1000.0 {
            format!("{:.2} µs", self.constant_factor_ns / 1000.0)
        } else {
            format!("{:.2} ns", self.constant_factor_ns)
        }
    }

    /// Format the standard deviation
    pub fn format_std_dev(&self) -> String {
        if self.std_dev_ns >= 1000.0 {
            format!("±{:.2} µs", self.std_dev_ns / 1000.0)
        } else {
            format!("±{:.2} ns", self.std_dev_ns)
        }
    }
}

/// Measure the time for a single operation batch
fn measure_once<F: FnMut()>(mut operation: F) -> Duration {
    let start = Instant::now();
    operation();
    start.elapsed()
}

/// Measure an operation multiple times and return statistics
fn measure_with_stats<F: FnMut()>(
    mut setup: impl FnMut(),
    mut operation: F,
    iterations: usize,
) -> Vec<Duration> {
    let mut timings = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        setup();
        let duration = measure_once(&mut operation);
        timings.push(duration);
    }

    timings
}

/// Calculate constant factors from timings at different sizes
fn calculate_constant_factors(
    timings_by_size: &[(usize, Vec<Duration>)],
    complexity: BatchComplexity,
) -> (f64, f64, f64, f64) {
    let mut all_factors: Vec<f64> = Vec::new();

    for (size, timings) in timings_by_size {
        for timing in timings {
            let factor = complexity.constant_factor(*timing, *size);
            all_factors.push(factor);
        }
    }

    let mean = all_factors.iter().sum::<f64>() / all_factors.len() as f64;
    let variance = all_factors.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / all_factors.len() as f64;
    let std_dev = variance.sqrt();
    let min = all_factors.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = all_factors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    (mean, std_dev, min, max)
}

// ============================================================================
// Operation Benchmarks
// ============================================================================

/// Measure insert batch constant factor for a heap type
fn measure_insert_constant_factor<H: Heap<i32, i32>>(
    heap_name: &str,
    complexity: BatchComplexity,
) -> ConstantFactorResult {
    let mut timings_by_size: Vec<(usize, Vec<Duration>)> = Vec::new();
    let mut all_timings: Vec<(usize, Duration)> = Vec::new();

    for &size in &TEST_SIZES {
        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let mut heap = H::new();
            for i in 0..size {
                heap.push(i as i32, i as i32);
            }
        }

        // Measure
        let timings = measure_with_stats(
            || {},
            || {
                let mut heap = H::new();
                for i in 0..size {
                    heap.push(i as i32, i as i32);
                }
            },
            MEASUREMENT_ITERATIONS,
        );

        for t in &timings {
            all_timings.push((size, *t));
        }
        timings_by_size.push((size, timings));
    }

    let (mean, std_dev, min, max) = calculate_constant_factors(&timings_by_size, complexity);

    ConstantFactorResult {
        heap_name: heap_name.to_string(),
        operation: "insert".to_string(),
        complexity,
        constant_factor_ns: mean,
        std_dev_ns: std_dev,
        min_factor_ns: min,
        max_factor_ns: max,
        timings: all_timings,
    }
}

/// Measure pop batch constant factor for a heap type
fn measure_pop_constant_factor<H: Heap<i32, i32>>(heap_name: &str) -> ConstantFactorResult {
    let complexity = BatchComplexity::NLogN; // All heaps have O(log n) pop
    let mut timings_by_size: Vec<(usize, Vec<Duration>)> = Vec::new();
    let mut all_timings: Vec<(usize, Duration)> = Vec::new();

    for &size in &TEST_SIZES {
        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let mut heap = H::new();
            for i in 0..size {
                heap.push(i as i32, i as i32);
            }
            for _ in 0..size {
                heap.pop();
            }
        }

        // Measure
        let timings = measure_with_stats(
            || {},
            || {
                let mut heap = H::new();
                for i in 0..size {
                    heap.push(i as i32, i as i32);
                }
                for _ in 0..size {
                    heap.pop();
                }
            },
            MEASUREMENT_ITERATIONS,
        );

        for t in &timings {
            all_timings.push((size, *t));
        }
        timings_by_size.push((size, timings));
    }

    let (mean, std_dev, min, max) = calculate_constant_factors(&timings_by_size, complexity);

    ConstantFactorResult {
        heap_name: heap_name.to_string(),
        operation: "pop".to_string(),
        complexity,
        constant_factor_ns: mean,
        std_dev_ns: std_dev,
        min_factor_ns: min,
        max_factor_ns: max,
        timings: all_timings,
    }
}

/// Measure decrease_key batch constant factor for a heap type
fn measure_decrease_key_constant_factor<H: Heap<i32, i32>>(
    heap_name: &str,
    complexity: BatchComplexity,
) -> ConstantFactorResult {
    let mut timings_by_size: Vec<(usize, Vec<Duration>)> = Vec::new();
    let mut all_timings: Vec<(usize, Duration)> = Vec::new();

    for &size in &TEST_SIZES {
        // Warmup
        for _ in 0..WARMUP_ITERATIONS {
            let mut heap = H::new();
            let handles: Vec<_> = (0..size).map(|i| heap.push((i + 10000) as i32, i as i32)).collect();
            for (i, handle) in handles.iter().enumerate() {
                let _ = heap.decrease_key(handle, i as i32);
            }
        }

        // Measure
        let timings = measure_with_stats(
            || {},
            || {
                let mut heap = H::new();
                let handles: Vec<_> = (0..size).map(|i| heap.push((i + 10000) as i32, i as i32)).collect();
                for (i, handle) in handles.iter().enumerate() {
                    let _ = heap.decrease_key(handle, i as i32);
                }
            },
            MEASUREMENT_ITERATIONS,
        );

        for t in &timings {
            all_timings.push((size, *t));
        }
        timings_by_size.push((size, timings));
    }

    let (mean, std_dev, min, max) = calculate_constant_factors(&timings_by_size, complexity);

    ConstantFactorResult {
        heap_name: heap_name.to_string(),
        operation: "decrease_key".to_string(),
        complexity,
        constant_factor_ns: mean,
        std_dev_ns: std_dev,
        min_factor_ns: min,
        max_factor_ns: max,
        timings: all_timings,
    }
}

// ============================================================================
// Reporting
// ============================================================================

/// Print a formatted table of constant factor results
fn print_results_table(title: &str, results: &mut [ConstantFactorResult]) {
    // Sort by constant factor (best first)
    results.sort_by(|a, b| {
        a.constant_factor_ns
            .partial_cmp(&b.constant_factor_ns)
            .unwrap()
    });

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║ {:^76} ║",
        title
    );
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:^20} │ {:^12} │ {:^18} │ {:^16} ║",
        "Heap", "Complexity", "Constant Factor", "Std Dev"
    );
    println!("╟──────────────────────┼──────────────┼────────────────────┼──────────────────╢");

    for result in results.iter() {
        println!(
            "║ {:20} │ {:^12} │ {:>18} │ {:>16} ║",
            result.heap_name,
            result.complexity.as_str(),
            result.format_factor(),
            result.format_std_dev()
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Also print detailed timings for verification
    println!();
    println!("Detailed timings (for scaling verification):");
    for result in results.iter() {
        print!("  {}: ", result.heap_name);
        let mut by_size: std::collections::BTreeMap<usize, Vec<Duration>> =
            std::collections::BTreeMap::new();
        for (size, dur) in &result.timings {
            by_size.entry(*size).or_default().push(*dur);
        }
        for (size, durs) in by_size {
            let avg = durs.iter().map(|d| d.as_micros()).sum::<u128>() / durs.len() as u128;
            print!("n={}: {}µs  ", size, avg);
        }
        println!();
    }
}

/// Print a summary comparison across all operations
fn print_summary_table(
    insert_results: &[ConstantFactorResult],
    pop_results: &[ConstantFactorResult],
    decrease_key_results: &[ConstantFactorResult],
) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                            HEAP PERFORMANCE SUMMARY                                      ║");
    println!("║                    (Constant factors in nanoseconds per unit work)                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ {:^20} │ {:^20} │ {:^20} │ {:^20} ║",
        "Heap", "Insert", "Pop", "Decrease Key"
    );
    println!("╟──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────╢");

    // Collect all heap names
    let mut heap_names: Vec<&str> = insert_results.iter().map(|r| r.heap_name.as_str()).collect();
    heap_names.sort();
    heap_names.dedup();

    for heap_name in heap_names {
        let insert = insert_results
            .iter()
            .find(|r| r.heap_name == heap_name)
            .map(|r| r.format_factor())
            .unwrap_or_else(|| "N/A".to_string());

        let pop = pop_results
            .iter()
            .find(|r| r.heap_name == heap_name)
            .map(|r| r.format_factor())
            .unwrap_or_else(|| "N/A".to_string());

        let decrease_key = decrease_key_results
            .iter()
            .find(|r| r.heap_name == heap_name)
            .map(|r| r.format_factor())
            .unwrap_or_else(|| "N/A".to_string());

        println!(
            "║ {:20} │ {:>20} │ {:>20} │ {:>20} ║",
            heap_name, insert, pop, decrease_key
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════╝");

    println!();
    println!("Notes:");
    println!("  • Lower constant factor = better performance");
    println!("  • Insert/Decrease Key: factor is ns per element for O(n), ns per (n·log n) for O(n log n)");
    println!("  • Pop: factor is ns per (n·log n) since all heaps have O(log n) amortized pop");
    println!(
        "  • Heaps with O(1) amortized decrease_key: Fibonacci, Pairing, StrictFibonacci, TwoThree"
    );
    println!("  • Heaps with O(log n) decrease_key: Binomial, SkewBinomial");
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_constant_factors_insert() {
    println!("\n\n=== MEASURING INSERT CONSTANT FACTORS ===\n");

    let mut results = vec![
        measure_insert_constant_factor::<FibonacciHeap<i32, i32>>(
            "FibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<PairingHeap<i32, i32>>(
            "PairingHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<BinomialHeap<i32, i32>>(
            "BinomialHeap",
            BatchComplexity::NLogN,
        ),
        measure_insert_constant_factor::<SkewBinomialHeap<i32, i32>>(
            "SkewBinomialHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<StrictFibonacciHeap<i32, i32>>(
            "StrictFibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<TwoThreeHeap<i32, i32>>(
            "TwoThreeHeap",
            BatchComplexity::Linear,
        ),
    ];

    print_results_table("INSERT BATCH CONSTANT FACTORS", &mut results);
}

#[test]
fn test_constant_factors_pop() {
    println!("\n\n=== MEASURING POP CONSTANT FACTORS ===\n");

    let mut results = vec![
        measure_pop_constant_factor::<FibonacciHeap<i32, i32>>("FibonacciHeap"),
        measure_pop_constant_factor::<PairingHeap<i32, i32>>("PairingHeap"),
        measure_pop_constant_factor::<BinomialHeap<i32, i32>>("BinomialHeap"),
        measure_pop_constant_factor::<SkewBinomialHeap<i32, i32>>("SkewBinomialHeap"),
        measure_pop_constant_factor::<StrictFibonacciHeap<i32, i32>>("StrictFibonacciHeap"),
        // TwoThreeHeap pop has issues, skip for now
    ];

    print_results_table("POP BATCH CONSTANT FACTORS", &mut results);
}

#[test]
fn test_constant_factors_decrease_key() {
    println!("\n\n=== MEASURING DECREASE_KEY CONSTANT FACTORS ===\n");

    let mut results = vec![
        measure_decrease_key_constant_factor::<FibonacciHeap<i32, i32>>(
            "FibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_decrease_key_constant_factor::<PairingHeap<i32, i32>>(
            "PairingHeap",
            BatchComplexity::Linear,
        ),
        measure_decrease_key_constant_factor::<BinomialHeap<i32, i32>>(
            "BinomialHeap",
            BatchComplexity::NLogN,
        ),
        measure_decrease_key_constant_factor::<SkewBinomialHeap<i32, i32>>(
            "SkewBinomialHeap",
            BatchComplexity::NLogN,
        ),
        measure_decrease_key_constant_factor::<StrictFibonacciHeap<i32, i32>>(
            "StrictFibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_decrease_key_constant_factor::<TwoThreeHeap<i32, i32>>(
            "TwoThreeHeap",
            BatchComplexity::Linear,
        ),
    ];

    print_results_table("DECREASE_KEY BATCH CONSTANT FACTORS", &mut results);
}

/// Comprehensive test that measures and reports all constant factors
#[test]
fn test_all_constant_factors() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              HEAP CONSTANT FACTOR ANALYSIS                                   ║");
    println!("║                                                                              ║");
    println!("║  Measuring the constant factors in Big-O complexity for each heap.          ║");
    println!("║  For T(n) = c·f(n), we report 'c' in nanoseconds.                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Measure insert
    println!("\n[1/3] Measuring insert operations...");
    let mut insert_results = vec![
        measure_insert_constant_factor::<FibonacciHeap<i32, i32>>(
            "FibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<PairingHeap<i32, i32>>(
            "PairingHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<BinomialHeap<i32, i32>>(
            "BinomialHeap",
            BatchComplexity::NLogN,
        ),
        measure_insert_constant_factor::<SkewBinomialHeap<i32, i32>>(
            "SkewBinomialHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<StrictFibonacciHeap<i32, i32>>(
            "StrictFibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_insert_constant_factor::<TwoThreeHeap<i32, i32>>(
            "TwoThreeHeap",
            BatchComplexity::Linear,
        ),
    ];
    print_results_table("INSERT BATCH CONSTANT FACTORS", &mut insert_results);

    // Measure pop
    println!("\n[2/3] Measuring pop operations...");
    let mut pop_results = vec![
        measure_pop_constant_factor::<FibonacciHeap<i32, i32>>("FibonacciHeap"),
        measure_pop_constant_factor::<PairingHeap<i32, i32>>("PairingHeap"),
        measure_pop_constant_factor::<BinomialHeap<i32, i32>>("BinomialHeap"),
        measure_pop_constant_factor::<SkewBinomialHeap<i32, i32>>("SkewBinomialHeap"),
        measure_pop_constant_factor::<StrictFibonacciHeap<i32, i32>>("StrictFibonacciHeap"),
        // TwoThreeHeap pop has known issues
    ];
    print_results_table("POP BATCH CONSTANT FACTORS", &mut pop_results);

    // Measure decrease_key
    println!("\n[3/3] Measuring decrease_key operations...");
    let mut decrease_key_results = vec![
        measure_decrease_key_constant_factor::<FibonacciHeap<i32, i32>>(
            "FibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_decrease_key_constant_factor::<PairingHeap<i32, i32>>(
            "PairingHeap",
            BatchComplexity::Linear,
        ),
        measure_decrease_key_constant_factor::<BinomialHeap<i32, i32>>(
            "BinomialHeap",
            BatchComplexity::NLogN,
        ),
        measure_decrease_key_constant_factor::<SkewBinomialHeap<i32, i32>>(
            "SkewBinomialHeap",
            BatchComplexity::NLogN,
        ),
        measure_decrease_key_constant_factor::<StrictFibonacciHeap<i32, i32>>(
            "StrictFibonacciHeap",
            BatchComplexity::Linear,
        ),
        measure_decrease_key_constant_factor::<TwoThreeHeap<i32, i32>>(
            "TwoThreeHeap",
            BatchComplexity::Linear,
        ),
    ];
    print_results_table(
        "DECREASE_KEY BATCH CONSTANT FACTORS",
        &mut decrease_key_results,
    );

    // Print summary
    print_summary_table(&insert_results, &pop_results, &decrease_key_results);
}
