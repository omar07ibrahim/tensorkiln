# TensorKiln

[![CI][ci-badge]][ci-workflow]

TensorKiln compiles static `f32` tensor graphs into bounded, cache-aware CPU
execution plans.

The project is a deliberately narrow C++20 compiler/runtime, built to make the
hard parts inspectable: type and shape verification, deterministic graph
rewrites, layout lowering, kernel selection, lifetime-based memory reuse, and
differential validation against a separate reference interpreter.

> **Status:** the bounded type system, typed graph front-end, independent Python
> oracle, bounded reference interpreter, and deterministic dead-code
> elimination with composable provenance are available. Additional
> canonicalization and fusion passes, layout lowering, arena planning, kernels,
> and the optimized executor remain under construction. The v0.1 contract below
> is the target; **Available now** is the shipped subset.

## Why this exists

Tensor runtimes often hide graph semantics, allocation policy, and numerical
trade-offs behind a large dependency stack. TensorKiln keeps one useful slice
small enough to audit end to end:

```text
GraphBuilder
    -> verify and infer
    -> eliminate dead code
    -> canonicalize and fuse safe epilogues
    -> lower layouts and select kernels
    -> plan one bounded arena
    -> execute without per-run heap allocation
```

The goal is evidence, not a production-runtime claim. Every optimized result
must agree with the unoptimized interpreter under a documented numerical
policy, and every memory-plan claim must be derived from a verifiable plan.

## v0.1 contract

- C++20 and the standard library only.
- Immutable, topologically ordered SSA graphs with static shapes.
- `f32`, rank 0 through 4, positive extents, checked element and byte counts.
- Transformer-oriented operations: `MatMul`, `Add`, `Mul`, `Relu`, `Gelu`,
  `Softmax`, `LayerNorm`, `Reshape`, and `Transpose`.
- A logical graph IR separated from the strided, allocation-aware execution
  plan.
- A reference interpreter that does not reuse optimized kernels.
- Deterministic IR and plan dumps suitable for regression tests.
- GCC, Clang, AddressSanitizer, and UndefinedBehaviorSanitizer coverage.

The semantics borrow only the relevant, explicitly documented pieces of the
[ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html) and
[broadcasting](https://onnx.ai/onnx/repo-docs/Broadcasting.html) contracts.
TensorKiln is not an ONNX importer and does not claim ONNX conformance.

## Available now

The current vertical slice is small but executable:

- checked scalar and rank 1-4 tensor types with explicit element/byte ceilings;
- trailing multidirectional broadcasting and rank 2-4 batched `MatMul`
  inference;
- a transactional `GraphBuilder` for `Input`, `Constant`, `Add`, `MatMul`, and
  `Relu`;
- owner-tagged handles that reject accidental cross-graph use;
- immutable verified graphs with deterministic, golden-tested IR dumps;
- graph-wide node, output, name, tensor, and cumulative constant-data limits;
- an isolated contiguous reference interpreter with owner-safe result lookup,
  exact payload/work ceilings, and fail-closed floating-point environment
  checks;
- bit-exact Python-stdlib fixtures consumed at real `MatMul -> Add -> Relu`
  boundaries;
- deterministic dead-code elimination that preserves the complete input
  contract, output declaration order and aliases, exact source construction
  limits, and bitwise constant payloads;
- owner-safe, composable provenance with stable pass statistics and
  deterministic dumps.

Validation failures never consume an ID, reserve a name, or mutate resource
counters. Constants own their exact IEEE-754 payload; the canonical dump uses a
stable bitwise fingerprint and does not depend on locale or pointer values.

```bash
make -j2 test
make -j2 example
make oracle
```

The first command runs the strict dependency-free test suite. The second builds,
prints, and reference-executes a small broadcast-add graph. The third proves
that the committed golden fixture still matches the independent generator. See
[the graph IR contract](docs/ir.md) for construction invariants and
[the reference interpreter contract](docs/reference.md) for execution,
resource, lifetime, and numerical semantics. See
[the compiler-pass contract](docs/compiler.md) for dead-code roots, semantic
equivalence, provenance composition, and determinism.

## Proof obligations

The first release is complete only when the repository demonstrates all of the
following:

1. malformed graphs fail before execution with stable, typed diagnostics;
2. optimized and reference execution agree on golden, randomized, and
   transformer-block workloads;
3. compiler passes preserve provenance and produce deterministic output;
4. the arena verifier rejects overlapping live allocations and invalid aliases;
5. a reusable execution session performs no heap allocation inside `run()`;
6. benchmarks report reproducible measurements, checksums, compiler flags, and
   workspace bytes without hard-coded performance claims.

The exact scope, invariants, and exclusions live in
[the v0.1 charter](docs/charter.md). Numerical comparisons are governed by
[the numerical policy](docs/numerics.md).

## Non-goals

TensorKiln v0.1 will not implement dynamic shapes, zero-sized tensors,
autograd, training, quantization, convolution, a general ONNX frontend, JIT
code generation, GPU execution, distributed execution, or a production BLAS
replacement. It intentionally uses no Eigen, BLAS, oneDNN, LLVM, or MLIR
runtime dependency.

## License

[MIT](LICENSE)

[ci-badge]: https://github.com/omar07ibrahim/tensorkiln/actions/workflows/ci.yml/badge.svg
[ci-workflow]: https://github.com/omar07ibrahim/tensorkiln/actions/workflows/ci.yml
