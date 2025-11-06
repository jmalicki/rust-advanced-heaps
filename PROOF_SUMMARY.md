# Proof Systems Summary

## What Was Set Up

✅ **Two-level proof structure** for finding bugs:

### 1. Trait-Level Proofs (`proofs/kani/trait_level_proofs.rs` and `proofs/kani/generic_trait_proofs.rs`)

Verify that **ALL** heap implementations satisfy the Heap trait contract:

- `is_empty()` ↔ `len() == 0`
- `push()` increments length by exactly 1
- `pop()` decrements length by exactly 1 (when not empty)
- `find_min()` returns the actual minimum priority
- `pop()` returns the same element as `find_min()`
- `decrease_key()` actually decreases priority
- `merge()` combines lengths correctly
- After popping all elements, heap is empty

**Coverage**: Tests BinomialHeap, FibonacciHeap, and PairingHeap for each
property.

### 2. Implementation-Specific Proofs (`proofs/kani/implementation_proofs.rs`)

Verify the specific invariants of each heap implementation:

**Binomial Heap**:

- Degree invariant maintenance
- Heap property after `decrease_key`
- Merge maintains degree invariant

**Fibonacci Heap**:

- Heap property after `decrease_key` (with cascading cuts)
- Structure maintained after consolidation
- Merge maintains heap property

**Pairing Heap**:

- Heap property after `decrease_key`
- Structure maintained after `delete_min`
- Merge maintains heap property

**Cross-Implementation**:

- All heaps produce same results for same operations

## Files Created

- `proofs/kani/trait_level_proofs.rs` - Trait-level proofs for Heap interface
- `proofs/kani/generic_trait_proofs.rs` - Generic trait-level proofs
- `proofs/kani/implementation_proofs.rs` - Implementation-specific invariant proofs
- `proofs/kani/kani_proofs.rs` - Legacy simple examples
- `build.rs` - Allows `cfg(kani)` attributes
- `kani.toml` - Kani configuration
- `PROOF_SETUP.md` - Quick reference guide
- `PROOF_SYSTEMS.md` - Comprehensive documentation (updated)

## Next Steps: Find Bugs

### Install Kani

```bash
cargo install --locked kani-verifier
cargo kani setup
```

### Run Proofs

```bash
# Run trait-level proofs (verify Heap trait contract)
cargo kani proofs/kani/trait_level_proofs.rs

# Run implementation-specific proofs (verify heap invariants)
cargo kani proofs/kani/implementation_proofs.rs

# Run all proofs
cargo kani proofs/kani/*.rs
```

### If Proofs Fail

When Kani finds a bug, it will provide:

- **Counterexample input values** that trigger the bug
- **Execution trace** showing where the assertion fails
- **Suggestions** for fixing the bug

## What These Proofs Will Find

The proofs are designed to find:

- ❌ Length accounting errors (push/pop don't update length correctly)
- ❌ Minimum tracking bugs (find_min returns wrong value)
- ❌ Heap property violations (parent > child in heap-ordered trees)
- ❌ Implementation-specific invariant violations:
  - Binomial: Multiple trees of same degree
  - Fibonacci: Degree invariant violations, marking rule violations
  - Pairing: Tree structure corruption
- ❌ Cross-implementation inconsistencies (same operations give different results)

## Proof Philosophy

1. **Trait-level**: Verify the interface contract that ALL implementations
   must satisfy
2. **Implementation-specific**: Verify each heap's unique invariants that
   guarantee correctness
3. **Comprehensive**: Cover all major operations (push, pop, find_min,
   decrease_key, merge)
4. **Cross-cutting**: Verify consistency across implementations

These proofs will catch bugs that unit tests might miss, especially:

- Edge cases in length accounting
- Heap property violations in complex operations
- Invariant violations after multiple operations
- Memory safety issues in unsafe pointer code
