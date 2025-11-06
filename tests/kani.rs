//! Kani proof harnesses
//!
//! This test module contains all Kani verification proofs for the heap implementations.
//! The proofs are organized in the tests/kani/ subdirectory.

#[cfg(kani)]
#[path = "kani/mod.rs"]
mod kani_module;
