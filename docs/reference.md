# Reference interpreter

The reference interpreter is the executable semantics for the graph layer that
exists today. It evaluates `Input`, `Constant`, broadcast `Add`, batched
`MatMul`, and `Relu` in topological order. It deliberately owns a distinct
contiguous `f32` payload for every graph node and does not call compiler passes,
including dead-code elimination or structural canonicalization, optimized
kernels, layout code, or the arena runtime.

That separation makes it a correctness oracle, not a performance baseline.

When the interpreter is used as a graph-rewrite oracle, output types, raw bits,
labels, order, and aliases must agree exactly. Tests may also compare every
source value with its provenance-mapped rewritten value. Materialized-byte and
scalar-step totals are expected to differ because eliminated or merged
definitions are no longer evaluated. The complete boundary is specified in
[the compiler-pass contract](compiler.md).

## Public boundary

```cpp
std::array<float, 6> input_data{/* ... */};
std::array<tensorkiln::InputBinding, 1> bindings{{
    {"x", input_data},
}};

auto executed = tensorkiln::ReferenceInterpreter::run(graph, bindings);
if (executed.error_if() != nullptr) {
  report(*executed.error_if());
  return;
}

tensorkiln::ReferenceResult result =
    std::move(*executed.value_if());
const tensorkiln::Tensor* output = result.output("result");
```

An `InputBinding` borrows its name and data only for the duration of `run()`.
Successful execution copies every feed bit-for-bit. The result also copies
tensor types and output names, so it does not retain the graph or binding
lifetimes.

`Tensor` and `ReferenceResult` are move-only to make large ownership transfers
explicit. A pointer returned by `value()` or `output()`, and any span obtained
from that tensor, remains valid until its owning result is moved from, assigned,
or destroyed. As usual for moved-from C++ objects, only assignment and
destruction are supported by this contract.

`value(ValueId)` compares the complete owner-tagged handle, not just its printed
ordinal. A `%0` from another graph therefore returns `nullptr`. Outputs are
looked up by their unique graph label; there is intentionally no
`output(OutputId)` overload because `OutputId` is not owner-tagged. Two output
labels that alias one value return the same `Tensor` address without copying its
payload.

## Validation order

Execution has a stable, phase-based diagnostic order:

1. reject more supplied bindings than the graph has input definitions;
2. reject the first unknown supplied name in binding order;
3. reject the first duplicate known name in binding order;
4. reject the first missing input in graph order;
5. reject the first wrong-sized input in graph order;
6. require round-to-nearest binary32 arithmetic, active binary64 intermediate
   precision, and gradual `f32` underflow without changing those control modes;
7. check aggregate materialized payload in graph order;
8. check aggregate scalar-step work in graph order;
9. allocate node payloads and evaluate topologically.

The binding-count ceiling makes validation memory and work depend on the
already-bounded graph, not on an arbitrarily large caller span. Binding and
environment failures precede resource policy. Materialization failures precede
scalar-step failures when both limits would reject a graph.

## Resource accounting

`ReferenceLimits` has two exact, inclusive ceilings:

| Field | Accounted quantity |
| --- | --- |
| `max_materialized_bytes` | Sum of `numel * 4` for every node, including inputs and constants. Each value is counted once even when several outputs alias it. |
| `max_scalar_steps` | `numel` for each input copy, constant copy, `Add`, or `Relu`; `output_numel * K` for each `MatMul`. |

Both sums use checked `uint64_t` arithmetic and complete before tensor payload
allocation. `ReferenceResult::materialized_bytes()` and `scalar_steps()` expose
the accepted totals.

These are logical payload and deterministic work-unit bounds. They are not
claims about total process memory or wall-clock instructions: container
metadata, allocator overhead, graph-owned constants, coordinate arithmetic,
and binding validation are not included. `std::bad_alloc` remains a C++ runtime
failure rather than a typed TensorKiln diagnostic.

The defaults allow 256 MiB of logical node payload and 2^28 scalar steps. A
caller evaluating untrusted graphs should lower them for its own service
budget.

## Numerical semantics

The Makefile compiles every implementation with `-fno-fast-math` and
`-ffp-contract=off`. The interpreter also fails closed unless all of the
following hold:

- `float` and `double` report IEEE-754 behavior;
- `double` has at least twice the `float` significand precision;
- the active rounding mode is `FE_TONEAREST`, with an arithmetic binary32
  sentinel that also rejects split x87/MXCSR rounding state;
- a runtime binary64 precision sentinel rejects an x87 single-precision
  control word;
- runtime sentinels confirm gradual `f32` underflow, rejecting FTZ/DAZ modes.

On targets whose evaluation method can retain nominal `double` expressions in
a wider format, the interpreter applies an explicit binary64 rounding barrier
after every `MatMul` reduction step. Other targets keep the accumulator in
registers. This pins the reference boundary across SSE and real i386/x87
execution without imposing an unnecessary store on the common path.

The interpreter does not change rounding or subnormal control modes. Like
ordinary floating-point code, graph arithmetic and the underflow sentinels may
set sticky exception status flags such as inexact, underflow, overflow, or
invalid.

| Operation | Reference behavior |
| --- | --- |
| `Input`, `Constant` | Copy each `f32` bit pattern exactly. |
| `Add` | One ordinary `f32` addition per output after trailing-axis broadcast. |
| `MatMul` | Multiply and accumulate in fixed row-major `K` order using `double`, then round once to `f32` per output element. Batch prefixes broadcast from the right. |
| `Relu` | Preserve positive values, positive infinity, positive subnormals, and quiet-NaN payloads; map negative values, negative infinity, and both signed zeros to positive zero. |

`Add` and `MatMul` otherwise follow IEEE arithmetic. Intentional non-finite
inputs can therefore produce non-finite outputs; execution does not convert
those values into diagnostics.

## Independent evidence

`tools/oracle.py` uses only the Python standard library. It rounds inputs and
node boundaries through `struct.pack/unpack("<f")`, emits exact hexadecimal C++
literals, and anchors expected raw bits before rendering. `make oracle` compares
the generated bytes with the committed fixture and fails on missing, stale, or
CRLF-normalized output.

The C++ suite builds an actual `MatMul -> Add -> Relu` MLP and compares all
three intermediate boundaries with that fixture. Separate adversarial cases
cover unequal-rank broadcast padding, batched matrix indexing, double-only
reduction behavior, foreign handles, output aliases, special values, diagnostic
precedence, and exact resource limits.

The arena executor is implemented separately from this interpreter. Its seeded
differential corpus exercises all five current kernel variants, arena reuse,
and optional write auditing while requiring raw-bit output agreement with this
reference path. See [the execution contract](execution.md) for that runtime's
distinct storage, lifetime, memory-integrity, and allocation guarantees.
