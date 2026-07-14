# Compiler passes

The graph-to-graph compiler pass available today is deterministic dead-code
elimination. It accepts only a `VerifiedGraph`, produces a new verified graph
in a fresh owner domain, and returns explicit provenance and accounting for the
rewrite.

Canonicalization, fusion, layout lowering, kernel selection, arena planning,
and optimized execution remain later layers.

## Public boundary

```cpp
auto eliminated = tensorkiln::DeadCodeElimination::run(source);
if (eliminated.error_if() != nullptr) {
  report(*eliminated.error_if());
  return;
}

tensorkiln::DeadCodeEliminationResult result =
    std::move(*eliminated.value_if());
const tensorkiln::VerifiedGraph& graph = result.graph();
```

`DeadCodeEliminationResult` is move-only because it owns the rewritten graph.
The source graph is not mutated and need not outlive the result.

## Reachability contract

Every declared graph output is a live root. Every `Input` definition is also a
live root, even when no output depends on it. Retaining otherwise-unused inputs
preserves the complete feed contract: input order, names, exact tensor types,
missing-binding checks, and unknown-binding checks do not change after the
rewrite.

The pass walks live operands backwards through the topologically ordered graph.
It then rebuilds the live definitions as a stable subsequence of source order.
Surviving definitions receive dense IDs in a fresh graph owner domain.

Output declaration order and labels are preserved exactly. Outputs that alias
one source value continue to reference one rewritten value.

## Rebuild and validation

The pass reconstructs the graph through `GraphBuilder` rather than bypassing
verification. It reuses the source graph's exact `GraphLimits`, including
custom shape, tensor, name, node, output, and cumulative constant-data ceilings.

For every retained definition, the operation kind, operand order, inferred
type, definition name, and constant payload are preserved. Constant `f32`
elements are copied without canonicalizing signed zero, NaN payloads,
subnormals, or other IEEE-754 encodings. Their bitwise fingerprint must remain
unchanged.

This pass is structural only. It does not reorder or reassociate arithmetic,
sort operands, fold identities, replace constants, fuse operations, or select a
different numerical implementation.

## Equivalence boundary

For the same input bindings, whenever the source and rewritten graphs both
complete under the caller's reference-execution limits, dead-code elimination
preserves:

- the complete input schema, including unused inputs;
- output labels and declaration order;
- every output tensor type and raw `f32` bit pattern;
- output alias topology;
- the relative order and semantics of every retained definition;
- the source graph's exact construction limits.

The following are intentionally not preserved:

- source `NodeId` and `ValueId` handles or their process-local owner token;
- the ordinals of definitions that followed removed definitions;
- access to eliminated intermediate values through `ReferenceResult::value()`;
- reference-interpreter materialized-byte and scalar-step totals;
- whether a particular `ReferenceLimits` budget accepts both graphs;
- sticky floating-point exception flags caused solely by eliminated work.

Resource counters are outside semantic equivalence because removing a node also
removes its reference payload and scalar work. A rewritten graph may therefore
execute under a budget that rejects the source graph.

## Provenance

`GraphProvenance` is the explicit bridge between graph owner domains. Each
entry identifies one result `NodeId`/`ValueId` pair and a non-empty, canonical
set of source `NodeId`/`ValueId` pairs. Each source definition belongs to at
most one result entry, which makes the singular reverse lookup unambiguous.

Immediate dead-code elimination has exactly one source definition for every
retained result definition. Eliminated definitions have no result entry.
Lookups compare complete owner-tagged handles, so a foreign handle with the same
printed ordinal does not match.

The provenance representation permits multiple source definitions in one
result so later fusion or common-subexpression elimination does not require a
different public model. Source sets are ordered and de-duplicated
deterministically. Passes that clone a definition or combine unrelated source
graphs will require a different reverse-lookup/domain contract and are outside
the current API.

`DeadCodeElimination::run(source, upstream)` composes the immediate rewrite with
existing provenance. The returned lineage points through the source graph to
the earlier definitions represented by `upstream`. The upstream result domain
must match the source graph exactly; an incompatible or incomplete map returns
`provenance_domain_mismatch`.

Provenance dumps print stable ordinals but omit opaque owner tokens, pointer
values, and allocation-dependent data.

## Determinism and accounting

The same verified source produces the same rewritten graph dump, provenance
dump, statistics, and retained-node order. Re-running dead-code elimination on
its result removes zero nodes and produces the same graph dump. Supplying the
first result's provenance during the second run keeps lineage rooted in the
original graph.

`DeadCodeEliminationStats` reports source, retained, and removed definition
counts together with corresponding owned constant-element counts.
`DeadCodeEliminationResult::dump()` combines those statistics with the
deterministic provenance report.

## Diagnostics

A verified source that cannot be replayed with the same limits, or whose
retained operation derives a different type, returns
`compiler_internal_invariant`. This diagnostic indicates a TensorKiln defect,
not malformed caller input.

An incompatible provenance domain returns `provenance_domain_mismatch`.
Allocation failure remains a C++ runtime failure rather than a typed compiler
diagnostic.
