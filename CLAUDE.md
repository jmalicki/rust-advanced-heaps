# Claude Code Guidelines for rust-advanced-heaps

## Rc/Weak Ownership Model

All heap implementations use `Rc<RefCell<Node>>` for memory safety.
Follow these principles:

### Single Strong Reference Invariant

Each node should have exactly **one strong Rc reference** at any time:

- Root nodes: owned by the heap's root field or forest array (`trees[dim]`)
- Child nodes: owned by parent's `child` field
- Sibling nodes: owned by previous sibling's `sibling`/`next` field
- Partner nodes: primary partner owns extra via `partner` field

The only exception is temporary ownership during operations
(e.g., `find_min_node` returns a clone).

### Weak References for Navigation

Use `Weak<RefCell<Node>>` for all back-pointers and cross-links:

- `parent`: child points back to parent
- `prev`: node points to previous sibling
- `partner_back`: extra partner points to primary

**Invariant**: All internal Weak references (not handles) are guaranteed
valid while the node exists in the heap. They always have a corresponding
strong reference somewhere in the structure.

### Move, Don't Clone

During tree operations, **move** ownership rather than cloning Rc or Weak:

```rust
// GOOD: Move ownership through the chain
fn delete_min(&mut self) -> Option<(P, T)> {
    let mut child_opt = min_node.borrow_mut().child.take();  // Move out
    while let Some(child) = child_opt {
        let next = child.borrow_mut().sibling.take();  // Move next out
        self.meld_node(child);  // Move into forest
        child_opt = next;       // Continue with moved ref
    }
}

// BAD: Cloning creates multiple strong refs and reference counting overhead
let children: Vec<_> = /* collect cloned refs */;
for child in children {
    self.meld_node(Rc::clone(&child));  // Extra refcount
}
```

**Critical rule**: Within a single public API operation
(insert, pop, decrease_key, merge), you should never clone a reference
and then drop it. Every `Rc::clone` or `Weak::clone` should either:

1. Create a new backlink that persists
   (e.g., `child.prev = Rc::downgrade(parent)`)
2. Return a reference to the caller (e.g., handle creation, `find_min_node`)

If you find yourself cloning to work around borrow checker issues,
restructure the code to avoid overlapping borrows instead.

### Handle References

Handles use `Weak<RefCell<Node>>` so they:

1. Don't prevent node cleanup when removed from heap
2. Can detect if node was removed (upgrade returns None)
3. Are the only place where Weak might outlive the strong ref

Handle cloning (for user API) is the one legitimate use of `Weak::clone`.

## Testing

- Run `cargo test` for all tests
- Big-O proof tests in `tests/big_o_proofs.rs` verify algorithmic complexity
- Property tests in `tests/property_tests.rs` use proptest for fuzzing

## References

Each heap implementation should link to its paper in the module docs.
See existing heaps for examples.
