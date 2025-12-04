# intrusive-circular-list

An intrusive circular doubly-linked list implementation for Rust.

## Overview

This crate provides a circular doubly-linked list designed to be compatible
with the patterns used in [`intrusive-collections`](https://crates.io/crates/intrusive-collections).
It can potentially be upstreamed to that crate in the future.

## Circular vs Linear Lists

In a circular list:

- A single node points to itself (both `next` and `prev`)
- There is no head or tail - any node can be the "entry point"
- Splicing two rings together is O(1)
- Iteration wraps around (must track starting point)

This is useful for data structures like Fibonacci heaps where:

- Siblings form a ring around their parent
- Nodes may belong to multiple circular lists simultaneously
- O(1) merge operations are required

## Usage

```rust
use intrusive_circular_list::{CircularLink, CircularListOps};
use std::ptr::NonNull;

struct Node {
    link: CircularLink,
    value: i32,
}

let mut ops = CircularListOps;

let mut node1 = Node { link: CircularLink::new(), value: 1 };
let mut node2 = Node { link: CircularLink::new(), value: 2 };

unsafe {
    let ptr1 = NonNull::from(&node1.link);
    let ptr2 = NonNull::from(&node2.link);

    // Create a single-element ring
    ops.make_circular(ptr1);

    // Insert node2 after node1
    ops.insert_after(ptr1, ptr2);

    // Both are now in the same ring
    assert_eq!(ops.next(ptr1), Some(ptr2));
    assert_eq!(ops.next(ptr2), Some(ptr1));
}
```

## Key Operations

All operations are O(1):

- `make_circular(ptr)` - Create a single-element ring
- `insert_after(at, new)` - Insert a node after another
- `insert_before(at, new)` - Insert a node before another
- `remove(ptr)` - Remove a node from the ring
- `splice(a, b)` - Merge two rings into one

## Features

- `std` (default) - Enable standard library support
- `no_std` compatible when `std` is disabled

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
