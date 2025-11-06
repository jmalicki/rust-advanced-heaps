//! Kani proof harnesses
//!
//! This module contains all Kani verification proofs for the heap implementations.
//! The files are organized by category:
//!
//! - `trait_level_proofs.rs`: Trait-level proofs for Heap interface
//! - `generic_trait_proofs.rs`: Generic trait-level proofs
//! - `implementation_proofs.rs`: Implementation-specific invariant proofs
//! - `stress_proofs.rs`: Stress tests with multiple heaps
//! - `edge_case_proofs.rs`: Edge case proofs
//! - `advanced_proofs.rs`: Advanced proofs
//! - `all_heaps_proofs.rs`: Proofs for all heap implementations
//! - `kani_proofs.rs`: Legacy simple examples

#[cfg(kani)]
#[path = "advanced_proofs.rs"]
mod advanced_proofs;
#[cfg(kani)]
#[path = "all_heaps_proofs.rs"]
mod all_heaps_proofs;
#[cfg(kani)]
#[path = "edge_case_proofs.rs"]
mod edge_case_proofs;
#[cfg(kani)]
#[path = "generic_trait_proofs.rs"]
mod generic_trait_proofs;
#[cfg(kani)]
#[path = "implementation_proofs.rs"]
mod implementation_proofs;
#[cfg(kani)]
#[path = "kani_proofs.rs"]
mod kani_proofs;
#[cfg(kani)]
#[path = "stress_proofs.rs"]
mod stress_proofs;
#[cfg(kani)]
#[path = "trait_level_proofs.rs"]
mod trait_level_proofs;
