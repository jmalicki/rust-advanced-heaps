# Contributing to Rust Advanced Heaps

Contributions are welcome! This document outlines how to contribute to the
project.

## Areas of Focus

1. **Performance benchmarks** - Compare implementations against each other and
   against standard library alternatives
2. **Test coverage** - Expand test coverage, especially edge cases and
   property-based testing
3. **Documentation** - Improve API documentation, examples, and explanations
4. **Additional heap variants** - Skew heap, Leftist heap, etc.

## Development Setup

This project uses pre-commit hooks for code quality checks. After cloning or
initializing a new worktree, install the hooks:

```bash
# Option 1: Use the setup script
./setup.sh

# Option 2: Manual setup
# Install pre-commit (if not already installed)
pip install pre-commit

# Install git hooks
pre-commit install
```

The hooks will automatically run `cargo fmt`, `cargo clippy`, and markdownlint
on every commit.

**Note for worktrees**: When using git worktrees, you must run
`pre-commit install` in each worktree after it's created.

## Code Quality

Before submitting a PR, ensure:

- `cargo fmt` - Code is formatted
- `cargo clippy` - No clippy warnings
- `cargo test` - All tests pass
- `cargo doc` - Documentation builds without warnings

## Adding a New Heap

When adding a new heap implementation:

1. Create a new module in `src/` (e.g., `src/newheap.rs`)
2. Implement the `Heap` trait from `src/traits.rs`
3. Add the module to `src/lib.rs`
4. Add tests in `tests/generic_heap_tests.rs` using the existing test macros
5. Add academic references to `docs/REFERENCES.md`
6. Update the README.md tables

## References

See [docs/REFERENCES.md](docs/REFERENCES.md) for academic papers on each heap
type.
