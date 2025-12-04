# optlike

Option-like storage with pluggable optimization strategies for Rust.

## Problem Statement

In data structures like heaps, nodes often contain values that may be "empty"
or "deleted" at runtime. The idiomatic Rust approach is to use `Option<T>`:

```rust
struct HeapNode<P> {
    priority: Option<P>,
    // ... other fields
}
```

However, `Option<T>` has a size overhead for types that don't have a niche
(unused bit pattern). For primitive types:

| Type | `size_of::<T>()` | `size_of::<Option<T>>()` | Overhead |
|------|------------------|--------------------------|----------|
| `i32` | 4 bytes | 8 bytes | +100% |
| `i64` | 8 bytes | 16 bytes | +100% |
| `f32` | 4 bytes | 8 bytes | +100% |
| `f64` | 8 bytes | 16 bytes | +100% |

For a heap with millions of nodes, this doubles the memory footprint for
priority storage, which has cascading effects on cache utilization.

## Solution

This crate provides the `OptLike` trait, which abstracts over Option-like
storage and allows plugging in different storage strategies:

- **`Optimized`** (default): Uses sentinel-based storage that achieves
  `size_of::<Storage>() == size_of::<T>()`. For integers, uses `NonMax*`
  types from the `nonmax` crate. For floats, uses NaN as the sentinel.

- **`PlainOption`**: Uses standard `Option<T>` for comparison and benchmarking,
  or when the sentinel value must be representable.

## Usage

```rust
use optlike::{OptLike, Optimized, PlainOption};
use std::mem::size_of;

// Optimized storage: same size as i32
let mut storage = <i32 as OptLike<Optimized>>::some(42);
assert_eq!(size_of_val(&storage), 4);
assert_eq!(<i32 as OptLike<Optimized>>::get(&storage), Some(42));

// Take the value out
let val = <i32 as OptLike<Optimized>>::take(&mut storage);
assert_eq!(val, Some(42));
assert!(<i32 as OptLike<Optimized>>::is_none(&storage));

// Plain storage: for benchmarking or when MAX is a valid value
let storage = <i32 as OptLike<PlainOption>>::some(i32::MAX);
assert_eq!(size_of_val(&storage), 8);  // Larger
assert_eq!(<i32 as OptLike<PlainOption>>::get(&storage), Some(i32::MAX));
```

## How It Works

### Integer Types (NonMax)

For integer types, we use the `nonmax` crate which provides types like
`NonMaxI32` that cannot hold their type's maximum value. Internally, NonMax
stores `value XOR MAX`, so:

- The actual MAX value becomes 0 internally
- Rust's niche optimization kicks in: `Option<NonMaxI32>` uses 0 to represent
  `None`
- Result: `Option<NonMaxI32>` is 4 bytes, same as `i32`

**Trade-off**: `i32::MAX` (or the corresponding MAX for each type) cannot be
stored. For heap priorities, this is usually acceptable since MAX is rarely
a meaningful priority value, and for min-heaps with `decrease_key`, priorities
only decrease (moving away from MAX).

### Float Types (NaN Sentinel)

For floating-point types, we use NaN as the sentinel value. The storage is a
`repr(transparent)` wrapper:

```rust
#[repr(transparent)]
pub struct OptF32(f32);  // Uses NaN to represent "none"
```

**Trade-off**: NaN cannot be stored as a priority. This is usually desirable
anyway since NaN breaks ordering (`NaN < x` is always false).

## Performance Considerations

### Memory and Cache

The primary benefit is **halving memory usage** for priority storage:

```text
1 million nodes with i64 priorities:
- Option<i64>: 16 MB just for priorities
- OptLike<i64>: 8 MB just for priorities
```

More importantly, this improves cache utilization. A 64-byte cache line holds:

- 4 `Option<i64>` values, or
- 8 `OptLike<i64>` values

For heap operations that traverse many nodes (like consolidation after
delete-min), fitting more nodes in cache reduces memory stalls.

### CPU Overhead

**Integer types (NonMax)**: Each `get()` operation performs one XOR instruction
to recover the actual value. This is:

- 1 cycle latency on modern CPUs
- Fully pipelined (doesn't stall)
- Typically hidden by memory latency

The XOR does add a data dependency:

```text
load -> XOR -> use
```

But this is almost always better than the alternative with `Option`:

- `Option` has a branch to check the discriminant
- Branch mispredictions cost ~15-20 cycles
- Even correctly predicted branches have overhead

**Float types**: No XOR needed. Checking for None is just `is_nan()`, which
is a simple bit pattern check.

### Speculative Execution

A reasonable concern: does the XOR transformation interfere with CPU
speculation?

**No.** The CPU can still:

1. Speculatively load the (XOR'd) value from memory
2. Speculatively execute the XOR
3. Speculatively use the result

There's no branch between load and use in the optimized path. The XOR is a
pure arithmetic operation that pipelines normally. Compare this to `Option`,
which has a branch on `is_some()` before the value can be used.

## Benchmarking

The strategy parameter exists specifically to enable A/B benchmarking:

```rust
use optlike::{OptLike, Optimized, PlainOption, StorageStrategy};

struct HeapNode<P: OptLike<S>, S: StorageStrategy = Optimized> {
    priority: P::Storage,
    // ...
}

// Your benchmark can then compare:
// - HeapNode<i64, Optimized>  (sentinel-based)
// - HeapNode<i64, PlainOption> (standard Option)
```

Or define type aliases for your heap:

```rust
pub type FastHeap<T, P> = HeapImpl<T, P, Optimized>;
pub type PlainHeap<T, P> = HeapImpl<T, P, PlainOption>;
```

## Supported Types

| Type | Optimized Storage | Sentinel Value |
|------|-------------------|----------------|
| `i8` | `Option<NonMaxI8>` | `i8::MAX` |
| `i16` | `Option<NonMaxI16>` | `i16::MAX` |
| `i32` | `Option<NonMaxI32>` | `i32::MAX` |
| `i64` | `Option<NonMaxI64>` | `i64::MAX` |
| `i128` | `Option<NonMaxI128>` | `i128::MAX` |
| `isize` | `Option<NonMaxIsize>` | `isize::MAX` |
| `u8` | `Option<NonMaxU8>` | `u8::MAX` |
| `u16` | `Option<NonMaxU16>` | `u16::MAX` |
| `u32` | `Option<NonMaxU32>` | `u32::MAX` |
| `u64` | `Option<NonMaxU64>` | `u64::MAX` |
| `u128` | `Option<NonMaxU128>` | `u128::MAX` |
| `usize` | `Option<NonMaxUsize>` | `usize::MAX` |
| `f32` | `OptF32` | NaN |
| `f64` | `OptF64` | NaN |

## Limitations

1. **Sentinel values cannot be stored**: Attempting to store `i32::MAX` with
   `Optimized` strategy will panic. Use `PlainOption` if you need the full
   range.

2. **Returns by value, not reference**: Because NonMax types use XOR
   internally, `get()` returns `Option<T>` not `Option<&T>`. For small types
   like integers, this is actually more efficient anyway.

3. **No generic fallback**: Custom types don't automatically get an
   implementation. You'd need to implement `OptLike` manually or use
   a newtype wrapper.

## License

MIT OR Apache-2.0
