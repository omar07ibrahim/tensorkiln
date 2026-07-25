# Verified dense execution

TensorKiln's executable path is a deliberately narrow, auditable slice. It
compiles one selected `VerifiedGraph` into a verified dense plan and executes
that plan synchronously in one preallocated arena. It is not a claim of a
general tensor runtime or a performance result.

## Scope and pipeline

The caller chooses the graph presented to plan compilation:

```text
VerifiedGraph
  -> optional DeadCodeElimination
  -> optional StructuralCanonicalization
  -> ExecutionPlanCompiler
       -> kernel choices plus arena placements
       -> independent ExecutionPlanVerifier
  -> ExecutionPlan
  -> ExecutionSession
```

Dead-code elimination and structural canonicalization are separate, explicit
calls. `ExecutionPlanCompiler::run()` never invokes them implicitly, so dead
compute remains executable unless the caller selects a rewritten graph.

The compiler candidate contains only a source-node ordinal and kernel choice
for each compute step, plus one offset for each arena allocation. The verifier
independently reconstructs the dense topology, operands, layouts, storage
classes, arena requests and lifetimes, output mappings, scalar work, limits,
and statistics from the source graph. A disagreement is an internal compiler
failure; an unverified candidate never becomes an `ExecutionPlan`.

The returned plan is move-only and owns its copied `VerifiedGraph`, plan-owned
constant payloads, verified arena projection, values, steps, limits, and
statistics. Its deterministic dump contains semantic ordinals and offsets, not
pointers or allocator-dependent state.

## Verified plan contract

All current values use dense row-major `f32` layouts. Inputs are external,
constants remain plan-owned and external to the arena, and every computed
result has arena storage. The current kernel selection is exact:

| Source operation | Verified kernel |
| --- | --- |
| `Add` with two operands matching the output shape | `add_contiguous_f32` |
| broadcasting `Add` | `add_broadcast_f32` |
| `MatMul` with two rank-2 operands and a rank-2 result | `matmul_rank2_f32` |
| every other valid rank-2 through rank-4 `MatMul` | `matmul_batched_f32` |
| `Relu` | `relu_contiguous_f32` |
| `Softmax` whose canonical axis is the final axis | `softmax_last_axis_f32` |

Axis-aware `Softmax` remains valid graph IR for every rank-valid axis and is
executable through the reference interpreter. The optimized compiler and
verifier support only the canonical last axis. They return
`plan_operation_unsupported` before inspecting candidate steps or placements
when any valid non-last Softmax is present. Graph arena storage lowering
remains axis-agnostic because output size and lifetime reconstruction do not
require a kernel.

Plan preflight bounds values, steps, outputs, owned constant bytes, scalar
steps, arena buffers, and workspace bytes before execution state is allocated.
Forward analysis and the independent reverse verifier each derive checked
`3 * numel` work for an accepted last-axis Softmax. Arena offsets remain
64-byte aligned. There are no views, in-place aliases, kernel scratch regions,
prepacked constants, or kernel-side temporary allocations in this slice.

## Session lifecycle

A session has four explicit phases:

```cpp
auto compiled = tensorkiln::ExecutionPlanCompiler::run(graph);
if (compiled.error_if() != nullptr) {
  report(*compiled.error_if());
  return;
}
tensorkiln::ExecutionPlan plan = std::move(*compiled.value_if());

tensorkiln::ExecutionSession session =
    tensorkiln::ExecutionSession::create(
        plan, tensorkiln::ExecutionSessionOptions{true});

std::array<tensorkiln::ExecutionInputBinding, 1> bindings{{
    {"x", input_data},
}};
auto bound = session.bind(bindings);
if (bound.error_if() != nullptr ||
    session.run() != tensorkiln::ExecutionRunStatus::success) {
  return;
}

std::optional<tensorkiln::ExecutionResultView> result = session.result();
if (!result.has_value()) {
  return;
}
std::optional<tensorkiln::TensorView> output = result->output("result");
```

`ExecutionSession::create()` allocates the workspace, value and binding pointer
tables, result-lifetime state, and, when requested, the write-audit shadow. For
a plan containing last-axis Softmax, creation also performs one non-foldable
`std::exp(0)` call so any process-local math-library initialization occurs
before the allocation-free `run()` boundary. The session borrows the immutable
plan; that plan must not be moved or destroyed until the session is destroyed.

A session is deliberately single-threaded. Independent sessions may share one
immutable plan and execute concurrently because each owns separate mutable
workspace and result state. A moved-from session supports destruction only.

## Bindings and borrowed views

`bind()` validates names, uniqueness, completeness, exact element counts, and
address ranges before activating a feed. Input payloads that overlap any byte
of the same session's workspace are rejected, including partial overlap.

A successful binding remains active across repeated `run()` calls until the
next `bind()` attempt or session destruction. Every bound payload must remain
alive and unchanged for that complete interval. This is also required when a
graph output directly names an input. Starting any bind attempt invalidates the
previous result and binding; a failed attempt leaves the session unbound.

Starting `run()`, binding again, or moving or destroying the session makes an
existing `ExecutionResultView` stale. A stale view is safe to query:
`current()` is false and `output()` returns `std::nullopt`, including after its
session has been destroyed.

`TensorView` is intentionally a raw borrowed snapshot for a zero-allocation
lookup. Its `TensorType` reference, data span, and any copied span may be used
only while the originating `ExecutionResultView` is current. They do not carry
the stale-view guard themselves; retaining and dereferencing them after
invalidation violates the public precondition.

## Floating-point contract

`run()` fails closed unless the active environment provides all of the
following:

- round-to-nearest binary32 arithmetic, checked both through `fegetround()` and
  an arithmetic sentinel so split x87/MXCSR modes cannot pass silently;
- active binary64 intermediate precision, including rejection of an x87
  single-precision control word;
- gradual binary32 underflow, with FTZ and DAZ modes rejected by consume and
  produce sentinels.

The executor never changes these modes. `unsupported_rounding_mode`,
`unsupported_binary64_precision`, and `unsupported_subnormal_mode` identify the
failed requirement. A failed run publishes no result.

`Add` and `Relu` follow the same ordinary binary32 paths as the independent
interpreter. Both `MatMul` kernels visit the reduction dimension in increasing
order, multiply and accumulate in binary64, round every reduction step to
binary64 on targets whose evaluation format is wider, and convert once to
binary32 at the output boundary. Fused contraction is disabled by the build.

`softmax_last_axis_f32` visits contiguous slices and coordinates in increasing
row-major order. It applies the documented NaN, positive-infinity, and
all-negative-infinity precedence before the ordinary subtract-maximum path.
The finite path writes each binary32-rounded `std::exp` result into the output
payload, accumulates those rounded numerators in binary64 with the same
excess-precision barrier, and normalizes them in place. The verified arena
lifetimes and external-binding checks ensure that the input and output payloads
do not overlap; the kernel implements no in-place-input mode. The complete
arithmetic policy is in [numerics.md](numerics.md).

## Memory integrity

Every non-empty session workspace has one 64-byte prefix guard and one 64-byte
suffix guard. `run()` checks both before the first kernel and after every
kernel. A guard mismatch returns `memory_corruption` and publishes no result.

`ExecutionSessionOptions{true}` additionally enables a per-kernel write-set
audit. Before each kernel, the session snapshots the complete logical arena;
afterward it requires every byte outside that step's exact output payload to
remain unchanged. This catches writes into another live buffer, a reusable
region, or alignment padding even when the outer guards remain intact. It does
not validate a wrong write that stays inside the declared output payload; the
independent numerical oracle covers that boundary.

`ExecutionPlanStats::workspace_bytes` and
`ExecutionSession::workspace_bytes()` report the logical arena only. Outer
guards, aligned-allocation overhead, pointer tables, result metadata, and the
optional audit shadow are intentionally excluded.

## Allocation evidence

The hot path begins after `create()` and `bind()`. A successful `run()` is
synchronous, `noexcept`, and designed to perform no heap allocation. The
release-profile allocation executable wraps global `new`/`new[]` and the C
`malloc`, `calloc`, `realloc`, `aligned_alloc`, and `posix_memalign` entry
points. With the counter armed, it executes:

- the first and a repeated run after session creation and binding for the five
  algebraic kernel kinds;
- a cold non-foldable `std::exp(0)` initialization during Softmax session
  creation, before the measured boundary;
- first and repeated warm last-axis Softmax runs with the counter armed;
- both regular and per-kernel-audited sessions;
- result lookup and payload observation;
- an audited external-input-only plan with zero workspace and zero kernels.

The probe is part of `make PROFILE=release test`. It is evidence about the
instrumented synchronous run/result path, not a general statement about graph
building, plan compilation, session creation, binding failures, the standard
library, cold transcendental-library initialization, or user callbacks.

## Differential and portability evidence

The deterministic suite includes hand-calculated fixtures, exact diagnostic
boundaries, regular and audited sessions, lifetime invalidation, outer-guard
and in-arena fault injection, and a seeded corpus of 128 DAGs. The corpus uses
audited sessions while exercising the five algebraic kernels, arena reuse, and
raw-bit output agreement with the independently implemented
`ReferenceInterpreter`. A separate seeded last-axis Softmax corpus uses the
documented tolerance, normalized-slice checks, and translation invariance; it
does not extend the five-kernel raw-bit claim across `std::exp`
implementations.

### Runnable Softmax boundary

The self-verifying
[`execute_softmax`](../examples/execute_softmax.cpp) example composes the public
graph, plan, session, result-view, and reference APIs in one inspected path. It
compiles a `f32[5,4]` last-axis graph to one
`softmax_last_axis_f32` step, checks the step's 60-unit `3 * numel` work charge
and 128-byte workspace, then runs it with kernel-write auditing enabled.

Its five slices deliberately have exact expected outputs: equal finite values,
NaN taking precedence over positive infinity, equal mass across positive
infinities, an all-negative-infinity slice, and finite values mixed with
negative infinity. The example prints raw `f32` bits only after the executor,
independent interpreter, and explicit 20-element fixture all agree. This is a
bit-exact claim about that constructed fixture, not arbitrary finite Softmax:
the seeded corpus and general finite comparisons retain the tolerance and
normalization rules in [the numerical policy](numerics.md).

The same executable constructs a graph-valid axis-zero Softmax, executes it
through the reference interpreter, and requires plan compilation to return
`plan_operation_unsupported` with the stable axis and node diagnostic. Its
reference total is 80 scalar steps: 20 for input materialization plus 60 for
Softmax. The optimized plan reports only the 60 kernel steps. The output is
deterministic correctness evidence, not a benchmark.

The full release suite is also executable as a real 32-bit i386/x87 gate on a
multilib host:

```bash
make PROFILE=release CXX_TAG=g++-i386-x87 \
  CXXFLAGS='-m32 -march=i686 -mfpmath=387' LDFLAGS='-m32' test
```

This is a portability gate, not a claim that the default GitHub Actions matrix
contains a 32-bit runner.

## Current exclusions

The executable slice does not implement fusion, strided or transposed views,
in-place aliases, reshape lowering, prepacking, scratch planning, SIMD,
threading, tiling, cache-aware kernels, a broad operator set, or a benchmark.
Those are separate compiler/runtime layers and must land with their own
contracts and evidence rather than being inferred from dense arena execution.
