//! Access pattern tracking for cache efficiency analysis
//!
//! This module provides feature-gated instrumentation for tracking field access patterns
//! in heap implementations. Enable with `--features access_tracking`.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::track;
//!
//! // In heap operations, track field accesses:
//! track!(read Priority);     // Priority field read
//! track!(write Priority);    // Priority field write
//! track!(chase);             // Pointer dereference (cache miss proxy)
//! track!(node);              // Node touched
//! ```
//!
//! # Analysis Output
//!
//! The collected statistics help identify:
//! - Which fields are read/written most frequently
//! - Pointer chase depth (correlates with cache misses)
//! - Access patterns for each operation type

use std::cell::RefCell;
use std::collections::HashMap;

/// Field types that can be accessed in heap nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Field {
    /// Priority value (used for comparisons)
    Priority,
    /// Item/data stored in the node
    Item,
    /// Pointer to child node(s)
    ChildPtr,
    /// Pointer to sibling node(s)
    SiblingPtr,
    /// Pointer to parent node
    ParentPtr,
    /// Marked flag (Fibonacci/rank-pairing heaps)
    Marked,
    /// Rank/degree of node
    Rank,
    /// Loss count (strict Fibonacci)
    Loss,
    /// Bucket index (radix heap)
    Bucket,
    /// Dimension (2-3 heap)
    Dimension,
}

/// Type of field access
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessType {
    /// Reading a field value
    Read,
    /// Writing/modifying a field value
    Write,
}

/// Statistics collected during heap operations
#[derive(Debug)]
pub struct AccessStats {
    /// Current operation being profiled
    current_op: &'static str,
    /// Nodes touched in current operation
    nodes_touched: usize,
    /// Field access counts: (field, access_type) -> count
    field_access: HashMap<(Field, AccessType), usize>,
    /// Current pointer chase depth
    chase_depth: usize,
    /// Maximum chase depth seen in current operation
    max_chase_depth: usize,
    /// Per-operation statistics
    op_stats: HashMap<&'static str, OpStats>,
}

/// Statistics for a single operation type
#[derive(Debug, Clone, Default)]
pub struct OpStats {
    /// Number of times this operation was called
    pub call_count: usize,
    /// Total nodes touched across all calls
    pub total_nodes: usize,
    /// Maximum nodes touched in a single call
    pub max_nodes: usize,
    /// Total chase depth across all calls
    pub total_chase_depth: usize,
    /// Maximum chase depth in a single call
    pub max_chase_depth: usize,
    /// Field access counts for this operation
    pub field_access: HashMap<(Field, AccessType), usize>,
}

impl AccessStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self {
            current_op: "",
            nodes_touched: 0,
            field_access: HashMap::new(),
            chase_depth: 0,
            max_chase_depth: 0,
            op_stats: HashMap::new(),
        }
    }

    /// Start tracking a new operation
    pub fn begin_op(&mut self, op_name: &'static str) {
        debug_assert!(
            self.current_op.is_empty(),
            "nested access_tracking ops are not supported"
        );
        self.current_op = op_name;
        self.nodes_touched = 0;
        self.field_access.clear();
        self.chase_depth = 0;
        self.max_chase_depth = 0;
    }

    /// End tracking current operation and record stats
    pub fn end_op(&mut self) {
        if self.current_op.is_empty() {
            return;
        }

        let stats = self.op_stats.entry(self.current_op).or_default();
        stats.call_count += 1;
        stats.total_nodes += self.nodes_touched;
        stats.max_nodes = stats.max_nodes.max(self.nodes_touched);
        stats.total_chase_depth += self.max_chase_depth;
        stats.max_chase_depth = stats.max_chase_depth.max(self.max_chase_depth);

        // Accumulate field access counts
        for ((field, access_type), count) in &self.field_access {
            *stats
                .field_access
                .entry((*field, *access_type))
                .or_default() += *count;
        }

        // Reset per-operation counters
        self.current_op = "";
        self.nodes_touched = 0;
        self.field_access.clear();
        self.chase_depth = 0;
        self.max_chase_depth = 0;
    }

    /// Record a field access
    pub fn record(&mut self, field: Field, access_type: AccessType) {
        if self.current_op.is_empty() {
            return;
        }
        *self.field_access.entry((field, access_type)).or_default() += 1;
    }

    /// Record a node being touched
    pub fn touch_node(&mut self) {
        if self.current_op.is_empty() {
            return;
        }
        self.nodes_touched += 1;
    }

    /// Record a pointer chase (increment depth)
    pub fn inc_chase(&mut self) {
        if self.current_op.is_empty() {
            return;
        }
        self.chase_depth += 1;
        self.max_chase_depth = self.max_chase_depth.max(self.chase_depth);
    }

    /// Record returning from a pointer chase (decrement depth)
    pub fn dec_chase(&mut self) {
        if self.current_op.is_empty() {
            return;
        }
        self.chase_depth = self.chase_depth.saturating_sub(1);
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.current_op = "";
        self.nodes_touched = 0;
        self.field_access.clear();
        self.chase_depth = 0;
        self.max_chase_depth = 0;
        self.op_stats.clear();
    }

    /// Get reference to operation statistics
    pub fn op_stats(&self) -> &HashMap<&'static str, OpStats> {
        &self.op_stats
    }

    /// Print a formatted report of collected statistics
    pub fn print_report(&self, heap_name: &str, workload: &str, n: usize) {
        println!("=== {} Analysis ({}, n={}) ===", heap_name, workload, n);
        println!();

        // Operation statistics
        println!("Operation Statistics:");
        let mut ops: Vec<_> = self.op_stats.iter().collect();
        ops.sort_by_key(|(name, _)| *name);

        for (op_name, stats) in &ops {
            if stats.call_count == 0 {
                continue;
            }
            let avg_nodes = stats.total_nodes as f64 / stats.call_count as f64;
            println!(
                "  {:15} {:8} calls, avg {:.1} nodes touched, max {}",
                op_name, stats.call_count, avg_nodes, stats.max_nodes
            );
        }
        println!();

        // Aggregate field access distribution
        println!("Field Access Distribution:");
        let mut total_accesses = 0usize;
        let mut field_totals: HashMap<Field, (usize, usize)> = HashMap::new(); // (reads, writes)

        for stats in self.op_stats.values() {
            for ((field, access_type), count) in &stats.field_access {
                let entry = field_totals.entry(*field).or_default();
                match access_type {
                    AccessType::Read => entry.0 += count,
                    AccessType::Write => entry.1 += count,
                }
                total_accesses += count;
            }
        }

        let mut fields: Vec<_> = field_totals.iter().collect();
        fields.sort_by_key(|(_, (r, w))| std::cmp::Reverse(r + w));

        for (field, (reads, writes)) in fields {
            let total = reads + writes;
            let pct = if total_accesses > 0 {
                (total as f64 / total_accesses as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "  {:12} read={:8} ({:4.1}%), write={:8} ({:4.1}%)",
                format!("{:?}", field),
                reads,
                if total_accesses > 0 {
                    (*reads as f64 / total_accesses as f64) * 100.0
                } else {
                    0.0
                },
                writes,
                if total_accesses > 0 {
                    (*writes as f64 / total_accesses as f64) * 100.0
                } else {
                    0.0
                },
            );
            let _ = pct; // Suppress unused warning
        }
        println!();

        // Pointer chase depth (cache miss indicator)
        println!("Pointer Chase Depth (cache miss indicator):");
        for (op_name, stats) in &ops {
            if stats.call_count == 0 {
                continue;
            }
            let avg_depth = stats.total_chase_depth as f64 / stats.call_count as f64;
            println!(
                "  {:15} max={:3}, avg={:.1}",
                op_name, stats.max_chase_depth, avg_depth
            );
        }
        println!();
    }
}

impl Default for AccessStats {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local storage for access statistics
thread_local! {
    /// Global access statistics collector
    pub static STATS: RefCell<AccessStats> = RefCell::new(AccessStats::new());
}

/// Begin tracking an operation
#[inline(always)]
pub fn begin_op(op_name: &'static str) {
    STATS.with(|s| s.borrow_mut().begin_op(op_name));
}

/// End tracking current operation
#[inline(always)]
pub fn end_op() {
    STATS.with(|s| s.borrow_mut().end_op());
}

/// Record a field access
#[inline(always)]
pub fn record_access(field: Field, access_type: AccessType) {
    STATS.with(|s| s.borrow_mut().record(field, access_type));
}

/// Record a node touch
#[inline(always)]
pub fn touch_node() {
    STATS.with(|s| s.borrow_mut().touch_node());
}

/// Record a pointer chase
#[inline(always)]
pub fn inc_chase() {
    STATS.with(|s| s.borrow_mut().inc_chase());
}

/// Record returning from a pointer chase
#[inline(always)]
pub fn dec_chase() {
    STATS.with(|s| s.borrow_mut().dec_chase());
}

/// Reset all statistics
pub fn reset_stats() {
    STATS.with(|s| s.borrow_mut().reset());
}

/// Print the access statistics report
pub fn print_report(heap_name: &str, workload: &str, n: usize) {
    STATS.with(|s| s.borrow().print_report(heap_name, workload, n));
}

/// Get a copy of the operation statistics
pub fn get_op_stats() -> HashMap<&'static str, OpStats> {
    STATS.with(|s| s.borrow().op_stats().clone())
}

/// Macro for tracking field accesses with zero overhead when feature is disabled
///
/// # Examples
///
/// ```rust,ignore
/// // Track field reads
/// track!(read Priority);
/// track!(read ChildPtr);
///
/// // Track field writes
/// track!(write Priority);
///
/// // Track pointer chasing
/// track!(chase);      // Going deeper
/// track!(unchase);    // Returning
///
/// // Track node being touched
/// track!(node);
/// ```
#[macro_export]
macro_rules! track {
    (read $field:ident) => {
        #[cfg(feature = "access_tracking")]
        $crate::access_tracking::record_access(
            $crate::access_tracking::Field::$field,
            $crate::access_tracking::AccessType::Read,
        );
    };
    (write $field:ident) => {
        #[cfg(feature = "access_tracking")]
        $crate::access_tracking::record_access(
            $crate::access_tracking::Field::$field,
            $crate::access_tracking::AccessType::Write,
        );
    };
    (chase) => {
        #[cfg(feature = "access_tracking")]
        $crate::access_tracking::inc_chase();
    };
    (unchase) => {
        #[cfg(feature = "access_tracking")]
        $crate::access_tracking::dec_chase();
    };
    (node) => {
        #[cfg(feature = "access_tracking")]
        $crate::access_tracking::touch_node();
    };
}

/// RAII guard for tracking operation scope
pub struct OpGuard {
    #[cfg(feature = "access_tracking")]
    _phantom: (),
}

impl OpGuard {
    /// Create a new operation guard, beginning operation tracking
    #[inline(always)]
    #[allow(unused_variables)]
    pub fn new(op_name: &'static str) -> Self {
        #[cfg(feature = "access_tracking")]
        begin_op(op_name);
        Self {
            #[cfg(feature = "access_tracking")]
            _phantom: (),
        }
    }
}

impl Drop for OpGuard {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(feature = "access_tracking")]
        end_op();
    }
}

/// Macro for creating an operation guard
///
/// # Examples
///
/// ```rust,ignore
/// fn decrease_key(&mut self, ...) {
///     let _guard = track_op!("decrease_key");
///     // ... operation code ...
///     // stats automatically recorded when _guard drops
/// }
/// ```
#[macro_export]
macro_rules! track_op {
    ($op_name:expr) => {
        $crate::access_tracking::OpGuard::new($op_name)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_stats_basic() {
        let mut stats = AccessStats::new();

        stats.begin_op("push");
        stats.record(Field::Priority, AccessType::Write);
        stats.touch_node();
        stats.end_op();

        assert_eq!(stats.op_stats["push"].call_count, 1);
        assert_eq!(stats.op_stats["push"].total_nodes, 1);
    }

    #[test]
    fn test_chase_depth_tracking() {
        let mut stats = AccessStats::new();

        stats.begin_op("cascade");
        stats.inc_chase();
        stats.inc_chase();
        stats.inc_chase();
        assert_eq!(stats.max_chase_depth, 3);
        stats.dec_chase();
        stats.dec_chase();
        assert_eq!(stats.max_chase_depth, 3); // Max preserved
        stats.end_op();

        assert_eq!(stats.op_stats["cascade"].max_chase_depth, 3);
    }
}
