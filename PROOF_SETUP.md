# Proof System Setup Summary

This document summarizes the proof systems that have been set up for
finding bugs in the heap implementations.

## Structure

### 1. Trait-Level Proofs (`tests/trait_level_proofs.rs`)

These proofs verify that **ALL** heap implementations satisfy the `Heap` trait contract:

- ✅ `is_empty()` is consistent with `len() == 0`
- ✅ `push()` always increments length by exactly 1
- ✅ `pop()` always decrements length by exactly 1 (when not empty)
- ✅ `find_min()` returns the actual minimum priority
- ✅ `pop()` returns the same element that `find_min()` would return
- ✅ `decrease_key()` actually decreases the priority
- ✅ `merge()` combines lengths correctly
- ✅ After popping all elements, heap is empty

**Tested implementations**: BinomialHeap, FibonacciHeap, PairingHeap

### 2. Implementation-Specific Proofs (`tests/implementation_proofs.rs`)

These proofs verify the specific invariants of each heap implementation:

#### Binomial Heap

- Degree invariant: At most one tree of each degree
- Heap property after `decrease_key`
- Merge maintains degree invariant

#### Fibonacci Heap

- Heap property after `decrease_key` (with cascading cuts)
- Structure maintained after consolidation
- Merge maintains heap property

#### Pairing Heap

- Heap property after `decrease_key`
- Structure maintained after `delete_min`
- Merge maintains heap property

#### Cross-Implementation Consistency

- All heaps produce same results for same operations

## Running Proofs

### Install Kani

```bash
cargo install --locked kani-verifier
cargo kani setup
```

### Run All Proofs

```bash
# Trait-level proofs
cargo kani --tests trait_level_proofs

# Implementation-specific proofs
cargo kani --tests implementation_proofs

# All proofs (requires higher unwind limits for complex operations)
cargo kani --tests kani_proofs --tests trait_level_proofs --tests implementation_proofs
```

### Run Specific Proof

```bash
# Example: Verify binomial heap push increments length
cargo kani --tests trait_level_proofs -- verify_push_increments_len_binomial
```

## Configuration

- `kani.toml` - Kani configuration (unwind limits, timeouts)
- `build.rs` - Allows `cfg(kani)` attributes
- `.prusti.toml` - Prusti configuration (for future use)
- `.creusot.toml` - Creusot configuration (for future use)

## Next Steps

1. **Install Kani** and run proofs to find bugs
2. **Increase unwind limits** for complex operations if needed
3. **Add more invariant checks** as bugs are found
4. **Set up Prusti** for more sophisticated static verification
5. **Set up Creusot** for mathematical proofs via Why3

## Finding Bugs

The proofs are designed to find:

- Length accounting errors
- Minimum tracking bugs
- Heap property violations
- Implementation-specific invariant violations
- Cross-implementation inconsistencies

If a proof fails, Kani will provide:

- Counterexample input values
- Execution trace showing where the assertion fails
- Suggestions for fixing the bug
