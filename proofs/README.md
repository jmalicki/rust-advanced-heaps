# Proof Tests

This directory contains proof tests that use verification tools like Kani,
Creusot, and Verus.

These tests are separate from regular tests because they:

- Require special verification tools to run
- Don't run with `cargo test` (they use `cargo kani`, etc.)
- May have different build requirements and dependencies
- Use `#[cfg(kani)]`, etc. to conditionally compile

## Structure

- `kani/` - Kani verification proofs
- `creusot/` - Creusot verification proofs
- `verus/` - Verus verification proofs

## Running Proofs

Each verification tool has its own command:

- Kani: `cargo kani --tests proofs/kani/<file>`
- Creusot: `cargo creusot proofs/creusot/<file>`
- Verus: `verus proofs/verus/<file>`

## Note

Proof files can be placed directly in the `proofs/` subdirectories. They can
reference the main crate using `use rust_advanced_heaps::...` and will be
compiled separately by the verification tools.
