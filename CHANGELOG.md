# Changelog

This file records user-visible TensorKiln milestones. Git tags are the
authoritative version source. Versions follow Semantic Versioning. Public APIs
may change throughout 0.x, and serialized diagnostic or dump formats may change
between prereleases.

## [Unreleased]

Work toward the v0.1.0 compiler and optimized-execution contract remains
in progress.

### Added

- a move-only dense `ExecutionPlan`, produced from minimal kernel/placement
  decisions only after independent reconstruction of layouts, operands,
  storage, lifetimes, outputs, limits, and work accounting;
- verified contiguous and broadcasting `Add`, rank-2 and batched `MatMul`, and
  contiguous `Relu` kernels;
- an `ExecutionSession` with a 64-byte-aligned workspace, guards around every
  non-empty workspace, persistent validated input bindings, stale-safe result
  lookup, and no published result after failure;
- an optional preallocated per-kernel write-set audit that detects changes
  outside the exact output payload, including arena padding and reusable
  regions;
- a seeded 128-DAG differential corpus covering all five kernels, arena reuse,
  audited execution, and raw-bit agreement with the independent interpreter;
- a release allocation probe covering C and C++ allocation entry points, all
  kernels, regular and audited sessions, zero-work execution, and both the
  first and repeated `run()` paths;
- an audited executable example that prints its verified plan and requires
  exact agreement with independent reference execution.

### Changed

- reference and executor `MatMul` reductions now pin every reduction step to
  binary64 on excess-precision targets, and both execution paths reject split
  binary32 rounding, insufficient binary64 precision, and FTZ/DAZ modes.

## [0.1.0-alpha.1] - 2026-07-14

This is an experimental, source-only snapshot of the verified graph front end,
reference execution path, deterministic rewrite core, and graph-derived storage
planning boundary. It is not an optimized tensor runtime.

### Added

- checked static `f32` tensor types with rank, shape, element, byte, and graph
  resource limits;
- an owner-safe graph builder for `Input`, `Constant`, `Add`, `MatMul`, and
  `Relu`, with immutable verified graphs and stable diagnostics;
- a bounded reference interpreter with an independent Python-stdlib oracle;
- deterministic dead-code elimination and exact structural canonicalization,
  including composable source provenance;
- a 64-byte interval arena planner and an independent placement verifier;
- graph-to-arena storage projection with mandatory reverse reconstruction of
  mappings, lifetimes, limits, statistics, and allocations;
- strict GCC 14 and Clang 18 debug/release builds, Clang AddressSanitizer and
  UndefinedBehaviorSanitizer checks, 170 C++ tests, and the Python oracle.

### Known limitations

- `Mul`, `Gelu`, `Softmax`, `LayerNorm`, `Reshape`, and `Transpose` are target
  operations and are not implemented;
- fusion, layout lowering, alias and scratch lowering, kernel selection,
  prepacking, executable arena allocation, and optimized execution are not
  implemented;
- the graph-derived arena result is a verified storage projection, not an
  executable lowered plan;
- there are no performance benchmarks or cache-performance claims;
- the supported distribution is source only, built from the repository root
  with GNU Make on the tested Ubuntu toolchain matrix;
- public APIs may change throughout 0.x; no compatibility boundary is promised
  before v1.0.0.

[Unreleased]: https://github.com/omar07ibrahim/tensorkiln/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/omar07ibrahim/tensorkiln/releases/tag/v0.1.0-alpha.1
