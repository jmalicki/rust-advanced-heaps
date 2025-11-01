# Proof Systems Setup Guide

This project integrates multiple proof/verification systems to help find bugs in the advanced heap implementations.

## Available Proof Systems

### 1. Kani (Recommended for Getting Started)

**Kani** is AWS's open-source model checker for Rust. It does NOT require an AWS account - it runs entirely locally.

- ✅ **No AWS account needed** - completely free and local
- ✅ **Easy to set up** - just install via cargo
- ✅ **Works with existing Rust code** - minimal code changes needed
- ✅ **Model checking** - explores all possible execution paths up to bounds
- ⚠️ **Bounded verification** - only checks up to a certain unwind limit

#### Installation

```bash
cargo install --locked kani-verifier
cargo kani setup
```

#### Usage

```bash
# Verify a specific proof
cargo kani --tests kani_proofs

# Verify all proofs
cargo kani
```

#### Example Proofs

See `tests/kani_proofs.rs` for examples including:
- `verify_insert_increments_len` - ensures insert always increments length
- `verify_pop_decrements_len` - ensures pop decrements length
- `verify_find_min_correct` - verifies find_min returns minimum
- `verify_decrease_key_decreases` - verifies decrease_key correctness

### 2. Prusti

**Prusti** is a static verifier for Rust, similar to Dafny/Spec#.

- ✅ Free and open source
- ✅ Annotations-based verification
- ⚠️ Requires `#[requires]` and `#[ensures]` annotations
- ⚠️ May need code refactoring for unsafe code

#### Installation

```bash
cargo install prusti
```

#### Usage

```bash
# Enable prusti feature
cargo prusti --features prusti

# Or compile with prusti
cargo prusti
```

### 3. Creusot

**Creusot** uses Why3 for verification of Rust programs.

- ✅ Mathematical proofs via Why3
- ⚠️ Requires significant code annotations
- ⚠️ May need refactoring for unsafe pointer code

#### Installation

See [Creusot documentation](https://github.com/creusot-rs/creusot) for setup.

#### Usage

```bash
cargo creusot
```

### 4. Verus

**Verus** is a verified Rust language variant.

- ✅ Full functional verification
- ⚠️ Requires rewriting code in Verus dialect
- ⚠️ Best for new implementations or verified wrappers

#### Installation

See [Verus documentation](https://github.com/verus-lang/verus) for setup.

## Comparison

| Tool | Setup Difficulty | Code Changes | Strengths |
|------|------------------|--------------|-----------|
| **Kani** | ⭐ Easy | Minimal | Model checking, easy to start |
| **Prusti** | ⭐⭐ Medium | Annotations needed | Static verification, good documentation |
| **Creusot** | ⭐⭐⭐ Hard | Significant changes | Mathematical proofs via Why3 |
| **Verus** | ⭐⭐⭐⭐ Very Hard | Rewrite needed | Full verification, new language |

## Recommended Approach

1. **Start with Kani** - easiest to set up, works with existing code
2. **Add Prusti** - for more sophisticated invariants via annotations
3. **Consider Creusot/Verus** - for deeper verification, but requires more effort

## Running Proofs

### Kani

```bash
# Install
cargo install --locked kani-verifier
cargo kani setup

# Run trait-level proofs (verify Heap trait contract)
cargo kani --tests trait_level_proofs

# Run implementation-specific proofs (verify heap invariants)
cargo kani --tests implementation_proofs

# Run all proofs
cargo kani --tests kani_proofs --tests trait_level_proofs --tests implementation_proofs

# With more unwind iterations (for complex operations)
cargo kani --tests trait_level_proofs -- --unwind 20
```

### Proof Structure

1. **Trait-Level Proofs** (`tests/trait_level_proofs.rs`):
   - Verify that ALL heap implementations satisfy the Heap trait contract
   - Properties like: push increments length, pop decrements length, find_min is correct
   - Tests BinomialHeap, FibonacciHeap, and PairingHeap

2. **Implementation-Specific Proofs** (`tests/implementation_proofs.rs`):
   - Verify specific invariants of each heap implementation
   - Binomial Heap: Degree invariant, heap property
   - Fibonacci Heap: Heap property, cascading cuts, consolidation
   - Pairing Heap: Heap property, structure maintenance
   - Cross-implementation consistency

3. **Legacy Proofs** (`tests/kani_proofs.rs`):
   - Simpler examples for getting started

### Prusti

```bash
# Install
cargo install prusti

# Run verification
cargo prusti --features prusti
```

## CI Integration

You can add proof checking to CI workflows. Example for Kani:

```yaml
# .github/workflows/verify.yml
name: Verify with Kani
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Kani
        run: |
          cargo install --locked kani-verifier
          cargo kani setup
      - name: Run Kani proofs
        run: cargo kani --tests kani_proofs
```

## Configuration

- `kani.toml` - Kani configuration (unwind limits, timeouts)
- `.prusti.toml` - Prusti configuration
- `.creusot.toml` - Creusot configuration

## Notes

- **Unsafe Code**: The heap implementations use unsafe Rust for pointer manipulation. Some proof systems may require additional work to verify unsafe code safely.

- **Unwind Limits**: Model checkers like Kani are bounded - they only check executions up to a certain depth. Increase unwind limits for complex operations.

- **Performance**: Verification can be slow for complex invariants. Start with simple properties and build up.

