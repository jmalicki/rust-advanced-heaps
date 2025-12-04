//! Multi-metric Criterion measurement using Linux perf counters.
//!
#![warn(missing_docs)]
//!
//! This crate provides a Criterion `Measurement` implementation that captures
//! wall clock time AND hardware performance counters simultaneously.
//!
//! Criterion uses wall clock time for statistical analysis, but all perf
//! counters are captured atomically and available in the `MultiValue` struct.
//!
//! # Example
//!
//! ```ignore
//! use criterion::{criterion_group, criterion_main, Criterion};
//! use perf_measurement::PerfMultiMeasurement;
//!
//! fn bench(c: &mut Criterion<PerfMultiMeasurement>) {
//!     c.bench_function("my_function", |b| b.iter(|| my_function()));
//! }
//!
//! criterion_group!(
//!     name = benches;
//!     config = Criterion::default().with_measurement(PerfMultiMeasurement::new());
//!     targets = bench
//! );
//! criterion_main!(benches);
//! ```

use criterion::{
    measurement::{Measurement, ValueFormatter},
    Throughput,
};
use perf_event::{events::Hardware, Builder, Counter, Group};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Thread-local accumulators for printing summary after each benchmark
// We accumulate across iterations and print periodically
thread_local! {
    static ITER_COUNT: AtomicU64 = const { AtomicU64::new(0) };
    static TOTAL_INSTRUCTIONS: AtomicU64 = const { AtomicU64::new(0) };
    static TOTAL_CYCLES: AtomicU64 = const { AtomicU64::new(0) };
    static TOTAL_CACHE_REFS: AtomicU64 = const { AtomicU64::new(0) };
    static TOTAL_CACHE_MISSES: AtomicU64 = const { AtomicU64::new(0) };
}

/// Holds multiple perf counter values measured simultaneously.
#[derive(Clone, Debug, Default)]
pub struct PerfCounters {
    /// CPU instructions retired
    pub instructions: u64,
    /// CPU cycles
    pub cycles: u64,
    /// Branch instructions
    pub branches: u64,
    /// Mispredicted branches
    pub branch_misses: u64,
    /// Cache references
    pub cache_refs: u64,
    /// Cache misses
    pub cache_misses: u64,
}

/// Combined measurement value: wall clock time + perf counters.
///
/// Criterion uses the Duration for statistical analysis, but all perf
/// counters are captured and can be accessed via the `counters` field.
#[derive(Clone, Debug)]
pub struct MultiValue {
    /// Wall clock time (used by Criterion for statistics)
    pub duration: Duration,
    /// Hardware performance counters
    pub counters: PerfCounters,
}

impl Default for MultiValue {
    fn default() -> Self {
        Self {
            duration: Duration::ZERO,
            counters: PerfCounters::default(),
        }
    }
}

/// Intermediate state during measurement.
pub struct MultiIntermediate {
    start_time: Instant,
    group: Group,
    instructions: Counter,
    cycles: Counter,
    branches: Counter,
    branch_misses: Counter,
    cache_refs: Counter,
    cache_misses: Counter,
}

/// Multi-metric measurement that captures wall clock time AND hardware counters.
///
/// Returns wall clock time to Criterion for statistical analysis, but also
/// captures instructions, cycles, branches, branch misses, cache refs, and
/// cache misses in a single atomic measurement.
#[derive(Clone, Default)]
pub struct PerfMultiMeasurement {
    formatter: MultiFormatter,
}

impl PerfMultiMeasurement {
    /// Create a new multi-metric measurement.
    #[must_use]
    pub fn new() -> Self {
        Self {
            formatter: MultiFormatter,
        }
    }
}

impl Measurement for PerfMultiMeasurement {
    type Intermediate = MultiIntermediate;
    type Value = MultiValue;

    fn start(&self) -> Self::Intermediate {
        let mut group = Group::new().expect("Failed to create perf group");

        let instructions = Builder::new()
            .group(&mut group)
            .kind(Hardware::INSTRUCTIONS)
            .build()
            .expect("Failed to add instructions counter");
        let cycles = Builder::new()
            .group(&mut group)
            .kind(Hardware::CPU_CYCLES)
            .build()
            .expect("Failed to add cycles counter");
        let branches = Builder::new()
            .group(&mut group)
            .kind(Hardware::BRANCH_INSTRUCTIONS)
            .build()
            .expect("Failed to add branches counter");
        let branch_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::BRANCH_MISSES)
            .build()
            .expect("Failed to add branch_misses counter");
        let cache_refs = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_REFERENCES)
            .build()
            .expect("Failed to add cache_refs counter");
        let cache_misses = Builder::new()
            .group(&mut group)
            .kind(Hardware::CACHE_MISSES)
            .build()
            .expect("Failed to add cache_misses counter");

        group.enable().expect("Failed to enable perf group");
        let start_time = Instant::now();

        MultiIntermediate {
            start_time,
            group,
            instructions,
            cycles,
            branches,
            branch_misses,
            cache_refs,
            cache_misses,
        }
    }

    fn end(&self, mut intermediate: Self::Intermediate) -> Self::Value {
        let duration = intermediate.start_time.elapsed();
        intermediate
            .group
            .disable()
            .expect("Failed to disable perf group");

        let counts = intermediate
            .group
            .read()
            .expect("Failed to read perf group");

        let counters = PerfCounters {
            instructions: counts[&intermediate.instructions],
            cycles: counts[&intermediate.cycles],
            branches: counts[&intermediate.branches],
            branch_misses: counts[&intermediate.branch_misses],
            cache_refs: counts[&intermediate.cache_refs],
            cache_misses: counts[&intermediate.cache_misses],
        };

        // Accumulate stats and print summary periodically
        ITER_COUNT.with(|c| {
            let count = c.fetch_add(1, Ordering::Relaxed) + 1;
            TOTAL_INSTRUCTIONS.with(|i| i.fetch_add(counters.instructions, Ordering::Relaxed));
            TOTAL_CYCLES.with(|cy| cy.fetch_add(counters.cycles, Ordering::Relaxed));
            TOTAL_CACHE_REFS.with(|r| r.fetch_add(counters.cache_refs, Ordering::Relaxed));
            TOTAL_CACHE_MISSES.with(|m| m.fetch_add(counters.cache_misses, Ordering::Relaxed));

            // Print summary every 10 iterations (covers warmup + measurement)
            if count % 10 == 0 {
                let total_instr = TOTAL_INSTRUCTIONS.with(|i| i.swap(0, Ordering::Relaxed));
                let total_cycles = TOTAL_CYCLES.with(|cy| cy.swap(0, Ordering::Relaxed));
                let total_refs = TOTAL_CACHE_REFS.with(|r| r.swap(0, Ordering::Relaxed));
                let total_misses = TOTAL_CACHE_MISSES.with(|m| m.swap(0, Ordering::Relaxed));
                c.store(0, Ordering::Relaxed);

                let ipc = if total_cycles > 0 {
                    total_instr as f64 / total_cycles as f64
                } else {
                    0.0
                };
                let miss_rate = if total_refs > 0 {
                    total_misses as f64 / total_refs as f64 * 100.0
                } else {
                    0.0
                };
                eprintln!(
                    "    [perf avg: IPC={:.2}, cache_miss={:.1}%]",
                    ipc, miss_rate
                );
            }
        });

        MultiValue { duration, counters }
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        MultiValue {
            duration: v1.duration + v2.duration,
            counters: PerfCounters {
                instructions: v1.counters.instructions + v2.counters.instructions,
                cycles: v1.counters.cycles + v2.counters.cycles,
                branches: v1.counters.branches + v2.counters.branches,
                branch_misses: v1.counters.branch_misses + v2.counters.branch_misses,
                cache_refs: v1.counters.cache_refs + v2.counters.cache_refs,
                cache_misses: v1.counters.cache_misses + v2.counters.cache_misses,
            },
        }
    }

    fn zero(&self) -> Self::Value {
        MultiValue::default()
    }

    #[allow(clippy::cast_precision_loss)]
    fn to_f64(&self, val: &Self::Value) -> f64 {
        // Return wall clock time in nanoseconds for Criterion's statistics
        val.duration.as_nanos() as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &self.formatter
    }
}

#[derive(Clone, Default)]
struct MultiFormatter;

impl ValueFormatter for MultiFormatter {
    fn scale_values(&self, typical_value: f64, values: &mut [f64]) -> &'static str {
        // Scale based on typical value (in nanoseconds)
        if typical_value < 1_000.0 {
            "ns"
        } else if typical_value < 1_000_000.0 {
            for val in values.iter_mut() {
                *val /= 1_000.0;
            }
            "µs"
        } else if typical_value < 1_000_000_000.0 {
            for val in values.iter_mut() {
                *val /= 1_000_000.0;
            }
            "ms"
        } else {
            for val in values.iter_mut() {
                *val /= 1_000_000_000.0;
            }
            "s"
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn scale_throughputs(
        &self,
        typical_value: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        // Convert from nanoseconds to seconds for throughput calculation
        for val in values.iter_mut() {
            *val /= 1_000_000_000.0;
        }

        match throughput {
            Throughput::Bytes(n) => {
                let bytes_per_second = *n as f64;
                for val in values.iter_mut() {
                    *val = bytes_per_second / *val;
                }
                // Scale to appropriate unit
                let typical_throughput = bytes_per_second / (typical_value / 1_000_000_000.0);
                if typical_throughput < 1024.0 {
                    "B/s"
                } else if typical_throughput < 1024.0 * 1024.0 {
                    for val in values.iter_mut() {
                        *val /= 1024.0;
                    }
                    "KiB/s"
                } else if typical_throughput < 1024.0 * 1024.0 * 1024.0 {
                    for val in values.iter_mut() {
                        *val /= 1024.0 * 1024.0;
                    }
                    "MiB/s"
                } else {
                    for val in values.iter_mut() {
                        *val /= 1024.0 * 1024.0 * 1024.0;
                    }
                    "GiB/s"
                }
            }
            Throughput::BytesDecimal(n) => {
                let bytes_per_second = *n as f64;
                for val in values.iter_mut() {
                    *val = bytes_per_second / *val;
                }
                let typical_throughput = bytes_per_second / (typical_value / 1_000_000_000.0);
                if typical_throughput < 1000.0 {
                    "B/s"
                } else if typical_throughput < 1_000_000.0 {
                    for val in values.iter_mut() {
                        *val /= 1000.0;
                    }
                    "KB/s"
                } else if typical_throughput < 1_000_000_000.0 {
                    for val in values.iter_mut() {
                        *val /= 1_000_000.0;
                    }
                    "MB/s"
                } else {
                    for val in values.iter_mut() {
                        *val /= 1_000_000_000.0;
                    }
                    "GB/s"
                }
            }
            Throughput::Bits(n) => {
                for val in values.iter_mut() {
                    *val = *n as f64 / *val;
                }
                "bits/s"
            }
            Throughput::Elements(n) => {
                for val in values.iter_mut() {
                    *val = *n as f64 / *val;
                }
                "elem/s"
            }
            Throughput::ElementsAndBytes { elements, .. } => {
                for val in values.iter_mut() {
                    *val = *elements as f64 / *val;
                }
                "elem/s"
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        // Return nanoseconds for machine-readable output
        "ns"
    }
}
