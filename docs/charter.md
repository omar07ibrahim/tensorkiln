# TensorKiln v0.1.0 target charter

This document is the scope boundary for the non-prerelease v0.1.0 target. A
feature is not part of that target merely because it would be useful; it must
fit the invariants and proof plan below.

## Target product claim

When the v0.1.0 target is complete, TensorKiln will be an educational but
rigorous static tensor-graph compiler and CPU runtime. It will accept a small
programmatic graph, reject invalid programs, derive all result types, compile
valid graphs into explicit execution plans, and run those plans inside a
preallocated workspace. The currently shipped graph front-end, reference
interpreter, graph rewrites, and storage-planning slice are listed in the
README.

It qualifies as a graph compiler by owning and testing semantic verification,
whole-graph rewrites, layout decisions, kernel selection, and storage planning.
It is not a JIT: v0.1 selects precompiled C++ kernels rather than emitting
machine code.

## Source graph invariants

- The public input is a `GraphBuilder`; there is no text parser or model-file
  importer in v0.1.
- Graphs are immutable SSA DAGs in topological order.
- Every value has exactly one definition, and each use follows its definition.
- Inputs, constants, nodes, and outputs have stable numeric identities.
- Result tensor types are inferred by the verifier, never trusted from callers.
- The only element type is IEEE-754 binary32 (`f32`).
- Rank is in `[0, 4]`; rank zero denotes a scalar.
- Extents are strictly positive. Element and byte counts use checked arithmetic
  and explicit resource ceilings.
- The graph printer is deterministic and omits process-local owner tokens.
  Compiler passes emit provenance in separate deterministic reports.

These choices align with the parts of the
[ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html) that define a
topologically sorted, typed computation graph. They do not imply file-format or
operator-level ONNX conformance.

## Operator contract

| Operation | Required v0.1 semantics |
| --- | --- |
| `Input` | Named external tensor with one exact static type. |
| `Constant` | Owned `f32` data whose length equals the inferred element count. |
| `MatMul` | Rank 2-4; matrix dimensions occupy the final two axes; batch prefixes broadcast from the right. |
| `Add`, `Mul` | Rank 0-4 multidirectional trailing-axis broadcasting. |
| `Relu` | Elementwise maximum with zero. |
| `Gelu` | Exact and tanh-approximation modes, selected explicitly. |
| `Softmax` | Any valid axis, with a subtract-maximum implementation. |
| `LayerNorm` | Normalize a suffix; exact suffix-shaped scale and optional bias; finite positive epsilon. |
| `Reshape` | Static target, positive extents, at most one inferred `-1`, unchanged element count. |
| `Transpose` | A required full permutation; no implicit default. |
| `Output` | Named graph result, not a computational node. |

Broadcast rules follow the
[ONNX broadcasting specification](https://onnx.ai/onnx/repo-docs/Broadcasting.html).
Individual semantics are pinned to the official operator specifications when
each operation lands; the repository's own tests remain normative for this
narrow implementation.

## Compilation boundary

Verification produces a `VerifiedGraph`; no executor accepts an unverified
graph. Compilation then performs, in order:

1. dead-code elimination with stable surviving-node order;
2. exact structural CSE and redundant-ReLU canonicalization;
3. idempotent reshape/transpose canonicalization;
4. conservative, single-use `MatMul` epilogue fusion;
5. layout lowering with explicit materialization when a view is unsafe;
6. deterministic kernel selection and constant-RHS prepacking;
7. liveness-based, 64-byte-aligned arena planning;
8. independent verification of the resulting plan.

This sequence is the v0.1 target. Dead-code elimination and exact structural
canonicalization are the graph-to-graph stages available today. A storage-only
proof slice of stage 7 is also available: the standalone interval core accepts
explicit requests, while graph arena lowering derives one request for every
current `Add`, `MatMul`, and `Relu` result and requires independent reverse
agreement before returning it. This projection intentionally precedes the
unfinished intermediate stages and is not a lowered execution plan. Fusion,
layout lowering, kernel selection, view and in-place alias proof, prepacking,
scratch integration, arena allocation, and optimized execution remain under
construction. The shipped storage boundary is specified in
[the arena contract](arena.md).

Dead-code elimination treats every declared output and every `Input`
definition as a root. It preserves the external feed schema, output declaration
order and aliases, and stable relative order of surviving definitions. Its
exact shipped guarantees are specified in
[the compiler-pass contract](compiler.md).

Structural canonicalization performs only exact `Add`, `MatMul`, and `Relu`
common-subexpression elimination plus redundant-ReLU removal. It preserves
output alias classes and does not apply algebraic identities or floating-point
reassociation. Its exact shipped guarantees are specified in the same
compiler-pass contract.

The graph IR describes logical tensors. The plan IR owns strides, aliases,
kernel variants, arena offsets, scratch requirements, and source provenance.
Keeping structured tensor semantics until after graph rewrites follows the
design direction documented by
[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) and delaying storage
decisions mirrors the separation described by
[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/). TensorKiln does
not link either project.

Every graph-to-graph rewrite creates a fresh owner domain. Provenance is an
explicit pass result rather than metadata embedded in the source graph dump.

## Execution boundary

The alpha ships a reference interpreter that executes the verified source graph
and allocates a distinct contiguous result for every node. The items below
describe the v0.1.0 optimized-execution target and are not available in
v0.1.0-alpha.1:

- the optimized executor runs a verified plan in an aligned workspace;
- external inputs and immutable constants do not consume arena storage;
- views alias a root allocation and extend that root's lifetime;
- graph outputs remain live until execution ends;
- a compiled graph is immutable, while an execution session owns mutable
  workspace and is deliberately not thread-safe;
- after session construction, `run()` performs no heap allocation;
- output views remain valid only until the next `run()` on that session.

## Required evidence

- A negative verifier matrix covering SSA, type, axis, permutation, broadcast,
  reshape, constant-size, resource-limit, and arithmetic-overflow failures.
- Hand-calculated golden tests for every operation.
- Seeded randomized differential and metamorphic tests with replayable seeds.
- Pass-idempotence and deterministic-printing tests.
- An independent arena-overlap validator, alignment checks, alias-lifetime
  cases, red-zone canaries, and a demonstrated reduction from naive storage.
- Deterministic MLP and pre-layer-normalized transformer-block fixtures.
- Strict GCC and Clang builds plus AddressSanitizer and
  UndefinedBehaviorSanitizer runs.
- Non-gating benchmark smoke tests in CI; full local reports include CPU,
  compiler, flags, checksums, latency distribution, throughput, and memory.

## Explicit exclusions

- ONNX import/export or full ONNX conformance;
- dynamic, symbolic, or zero-sized dimensions;
- mutation, control flow, training, or autograd;
- data types other than `f32`, quantization, or sparsity;
- convolution or a broad operator zoo;
- LLVM/MLIR linkage, JIT code generation, or architecture-specific assembly;
- Eigen, BLAS, oneDNN, or another numerical backend;
- explicit SIMD intrinsics, an internal thread pool, GPU, or distributed work;
- tokenizer, sampling, RoPE, KV-cache, or model-serving concerns;
- claims of competing with production GEMM libraries.
