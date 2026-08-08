# Numerical policy

Numerical tolerances are part of TensorKiln's contract. They may not be widened
solely to turn a failing test green; every change requires a replayable failing
case and an explanation of the changed error model.

## Comparison rule

For finite reference and actual values, approximate comparisons use:

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Unexpected `NaN` or infinity from finite, bounded test inputs is always a
failure. Tests that intentionally exercise non-finite inputs must state their
own expected behavior.

## Initial operation policy

| Operation | Policy |
| --- | --- |
| Shape-only operations, `Relu` | Bit-exact when they execute the same arithmetic path. |
| Standalone `Add`, `Mul` | Bit-exact on a single compiler/toolchain; cross-toolchain tests use zero relative tolerance and a one-ULP diagnostic allowance only when justified. |
| `Gelu` | `atol = 2e-6`, `rtol = 2e-6`. |
| `Softmax` | `atol = 2e-6`, `rtol = 2e-5`, plus a normalized-slice sum check. |
| `LayerNorm` | `atol = 3e-5`, `rtol = 3e-5`. |
| End-to-end bounded graphs | `atol = 3e-5`, `rtol = 3e-4`. |

Matrix multiplication tests will also use a dot-product-aware forward-error
bound based on the inner dimension and `sum(abs(a[i] * b[i]))`. This avoids a
single arbitrary tolerance for both tiny and cancellation-heavy products.

## Independence rule

The reference interpreter must not call optimized kernels, fusion helpers, or
the arena executor. Its reductions accumulate in `double` and round each node
result to `f32`. The independent Python oracle used for small fixtures may use
only the Python standard library.

This makes agreement meaningful: sharing parsing or shape metadata is allowed,
sharing the numerical implementation under test is not.

The currently executable operation-by-operation behavior, floating-point
environment checks, special-value rules, and exact reference work accounting
are specified in [the reference interpreter contract](reference.md).

The reference Softmax path accepts every graph-valid axis and uses three
ordered passes per slice. It first applies the exact non-finite precedence from
the reference contract. An ordinary slice subtracts its finite maximum,
evaluates `std::exp` in binary64, rounds each exponential to binary32, sums
those rounded values in fixed axis order using binary64, and rounds each
normalized quotient to binary32. The denominator is therefore derived from the
same rounded numerators that are normalized.

`std::exp` is not required to be bit-identical across operating-system math
libraries. Softmax comparisons use the tolerance above and a slice-sum check;
they never inherit the raw-bit claim made for the seven algebraic
kernels. The separate fixture uses 80-digit Python `Decimal` exponentials
rather than extending the existing raw-bit Python corpus. On excess-precision
targets, each denominator addition is explicitly rounded to binary64. Sticky
floating-point exception flags remain observable caller state, as they are for
the existing reference arithmetic.

## Arena executor rule

The verified dense executor is independently implemented but intentionally
matches the reference operation boundaries. Contiguous and broadcasting `Add`
perform one ordinary binary32 addition per output. `Relu` preserves the same
positive values and quiet-NaN payloads and maps negative values and both signed
zeros to positive zero. Differential tests require raw-bit agreement for these
operations; there is no tolerance fallback.

Contiguous and broadcasting `Mul` perform one ordinary binary32 multiplication
per output after applying trailing-axis broadcasting. Mul has no reduction or
higher-precision accumulator: its reference and optimized paths both round at
the ordinary `f32` operation boundary. Differential tests require their raw
output bits to agree, including signed-zero behavior.

Both dense `MatMul` kernels visit `K` in increasing order, multiply binary32
operands after conversion to binary64, accumulate in binary64, and convert once
to binary32 per output. On targets where the floating-point evaluation method
can retain a wider register value, both the executor and reference path apply
an explicit binary64 rounding barrier after every reduction step. Other
targets keep the accumulator in registers without an artificial store.

`softmax_last_axis_f32` independently repeats the reference operation's
three-pass ordering for each contiguous last-axis slice and applies the same
NaN, positive-infinity, and all-negative-infinity precedence. Its ordinary
path stores binary32-rounded exponentials in the output payload, accumulates
those rounded numerators in binary64 with the same excess-precision barrier,
then normalizes in place. Its evidence uses the Softmax tolerance,
normalized-slice, and translation-invariance rules rather than raw-bit
agreement across math libraries.

Before running a kernel, `ExecutionSession` requires `FE_TONEAREST`, confirms
binary32 nearest rounding with an arithmetic sentinel, confirms active
binary64 intermediate precision, and checks both consumption and production of
binary32 subnormals. This rejects split x87/MXCSR rounding, an x87
single-precision control word, and FTZ/DAZ modes. The executor does not change
the environment, and any failed check publishes no result. Builds use
`-fno-fast-math` and `-ffp-contract=off`.

The seeded arena corpus compares complete output bit patterns with the
independent reference interpreter across all seven algebraic
kernel variants.
Optimized last-axis Softmax is intentionally excluded from that algebraic
raw-bit corpus. Its separate seeded corpus uses tolerance, normalized-slice,
and translation-invariance checks because `std::exp` may differ across math
libraries. This evidence is exact for the seven fixed algebraic operation
paths; it is not a license for future fusion, reassociation, contraction, or
transcendental kernels to inherit the same claim without a new policy and
tests. See [execution.md](execution.md).

## Structural pass rule

Dead-code elimination and structural canonicalization have no numerical
tolerance. They neither replace nor reorder retained arithmetic. Exact CSE
reuses one evaluation of an identical operation with identical ordered
operands; it does not canonicalize `Add` or `Mul` by commutativity. Redundant
ReLU removal uses the interpreter's explicitly idempotent special-value
semantics.
Whenever source and rewritten executions with the same bindings both complete
under the caller's `ReferenceLimits`, they must produce bit-identical
provenance-mapped values and outputs, including signed-zero and NaN payload
bits. Reference resource counters and sticky exception flags from eliminated
or merged work are outside this equivalence boundary. See
[the compiler-pass contract](compiler.md).
