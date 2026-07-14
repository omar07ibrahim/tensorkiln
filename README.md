# TensorKiln

[![CI][ci-badge]][ci-workflow]

TensorKiln is a dependency-free C++20 project building a bounded static `f32`
tensor compiler/runtime. The shipped alpha verifies and reference-executes
graphs, applies deterministic graph rewrites, and derives reverse-verified
storage plans from graph lifetimes. Layout lowering, kernels, and cache-aware
optimized execution are target layers, not current functionality.

The project keeps one deliberately narrow compiler/runtime architecture small
enough to inspect end to end. Its stable target covers type and shape
verification, deterministic graph rewrites, layout lowering, kernel selection,
lifetime-based memory reuse, and differential validation against a separate
reference interpreter.

> **Status:** the bounded type system, typed graph front-end, independent Python
> oracle, bounded reference interpreter, and deterministic dead-code
> elimination and structural canonicalization with composable provenance are
> available, together with graph-derived compute lifetimes, a deterministic
> 64-byte interval arena planner, and independent reverse placement
> verification. Fusion, layout lowering, alias and scratch lowering, kernels,
> and the optimized executor remain under construction. The stable v0.1.0
> contract below is the target; **Available now** is the shipped subset.

## Why this exists

Tensor runtimes often hide graph semantics, allocation policy, and numerical
trade-offs behind a large dependency stack. TensorKiln keeps one useful slice
small enough to audit end to end:

```text
VerifiedGraph
    -> optional dead-code elimination
    -> optional exact structural canonicalization
    -> derive storage-only compute lifetimes
    -> apply the deterministic arena heuristic
    -> independently reconstruct and verify exact agreement
```

These are explicit API calls: arena lowering operates on whichever verified
graph the caller supplies and does not silently run compiler passes. The goal
is evidence, not a production-runtime claim. Every eventual optimized result
must agree with the unoptimized interpreter under a documented numerical
policy, and every memory-plan claim must be derived from a verifiable plan.

## Target v0.1.0 contract

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

The current vertical slice is small but runnable and inspectable:

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
- deterministic structural canonicalization with exact CSE for `Add`, `MatMul`,
  and `Relu`, plus the semantics-preserving `Relu(Relu(x)) -> Relu(x)` rule;
- an output-alias guard that prevents equivalent source outputs from silently
  collapsing into one result value;
- owner-safe, composable provenance with stable pass statistics and
  deterministic dumps, including many-source-to-one-result lineage;
- a deterministic best-fit arena planner with 64-byte-aligned offsets for
  explicit storage-root sizes and half-open lifetimes, with coalescing and
  boundary reuse;
- an independent placement verifier with checked arithmetic, exact workspace
  accounting, canonical dumps, stable diagnostics, and seeded pairwise-oracle
  coverage;
- a graph-to-arena storage projection that gives every `Add`, `MatMul`, and
  `Relu` result a dense sequential step and buffer ordinal, leaves inputs and
  constants external, retains dead compute, and keeps arena-backed outputs live
  through the final compute step;
- mandatory reverse reconstruction of graph mappings, lifetimes, limits,
  statistics, and allocations before a planned graph projection is returned,
  with seeded DAG, heterogeneous `MatMul`, ownership, fault-injection, and exact
  4096/4097-buffer boundary evidence.

Validation failures never consume an ID, reserve a name, or mutate resource
counters. Constants own their exact IEEE-754 payload; the canonical dump uses a
stable bitwise fingerprint and does not depend on locale or pointer values.

```bash
make -j2 test
make -j2 example
make oracle
```

TensorKiln v0.1.0-alpha.1 is a source-only milestone with an unstable
prerelease API. It is tested on Ubuntu 24.04 with GCC 14 and Clang 18; no
installable package or binary distribution is provided. Version tags are the
authoritative version source. See the
[alpha release notes](docs/releases/v0.1.0-alpha.1.md) and
[changelog](CHANGELOG.md) for the shipped boundary and known limitations.

The first command runs the strict dependency-free test suite and smoke-executes
both checked examples. The second prints the graph-rewrite pipeline, its
non-executable 192-to-128-byte graph storage projection, and a standalone
384-to-192-byte interval-reuse schedule. The third proves that the committed
golden fixture still matches the independent generator. See
[the graph IR contract](docs/ir.md) for construction invariants and
[the reference interpreter contract](docs/reference.md) for execution,
resource, lifetime, and numerical semantics. See
[the compiler-pass contract](docs/compiler.md) for dead-code roots, semantic
equivalence, exact canonicalization rules, output alias classes, provenance
composition, and determinism. The verified storage-planning boundary is
specified in [the arena contract](docs/arena.md).

## Proof obligations

The stable v0.1.0 release is complete only when the repository demonstrates all
of the following:

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
