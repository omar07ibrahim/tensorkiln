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
