# Compiler passes

The graph-to-graph compiler passes available today are deterministic dead-code
elimination and structural canonicalization. Each accepts only a
`VerifiedGraph`, produces a new verified graph in a fresh owner domain, and
returns explicit provenance and accounting for the rewrite.

Graph arena lowering is also available as a storage-only projection. It derives
lifetimes for the current materializing operations and requires independent
reverse placement verification, but it is not a graph-to-graph rewrite or an
executable plan. A separate `ExecutionPlanCompiler` now builds a narrow dense
row-major plan on top of that verified placement and selects five precompiled
kernel variants. Fusion, views and in-place aliases, scratch, prepacking, and
broader optimized lowering remain later layers. Storage and runtime boundaries
are specified in [the arena contract](arena.md) and
[the execution contract](execution.md).

## Public boundary

```cpp
auto eliminated = tensorkiln::DeadCodeElimination::run(source);
if (eliminated.error_if() != nullptr) {
  report(*eliminated.error_if());
  return;
}

tensorkiln::DeadCodeEliminationResult result =
    std::move(*eliminated.value_if());

auto canonicalized = tensorkiln::StructuralCanonicalization::run(
    result.graph(), result.provenance());
if (canonicalized.error_if() != nullptr) {
  report(*canonicalized.error_if());
  return;
}

tensorkiln::StructuralCanonicalizationResult compiled =
    std::move(*canonicalized.value_if());
const tensorkiln::VerifiedGraph& graph = compiled.graph();
```

Both result types are move-only because they own rewritten graphs. A source
graph is not mutated and need not outlive its result. Passing DCE provenance to
structural canonicalization keeps the final lineage rooted in the graph that
preceded DCE.

## Composition with graph arena lowering

Compiler passes and storage lowering are separate, explicit calls:

```cpp
auto storage = tensorkiln::GraphArenaLowering::run(compiled.graph());
if (storage.error_if() != nullptr) {
  report(*storage.error_if());
  return;
}

tensorkiln::GraphArenaLoweringResult projection =
    std::move(*storage.value_if());
```

Lowering the source graph directly retains all dead compute. Lowering the DCE
result reflects its fresh owner domain and any removed definitions; lowering
the canonicalized result additionally reflects exact CSE and redundant-ReLU
removal. `GraphArenaLowering` does not choose that ordering, mutate the selected
graph, or carry compiler provenance into its result. It assigns dense steps
only to `Add`, `MatMul`, and `Relu`; inputs and constants remain external. A
successful result proves a reverse-verified sequential storage placement, not
numerical equivalence or execution.

## Executable dense plan compilation

Executable compilation is another explicit call on the graph selected by the
caller:

```cpp
auto compiled_plan =
    tensorkiln::ExecutionPlanCompiler::run(compiled.graph());
if (compiled_plan.error_if() != nullptr) {
  report(*compiled_plan.error_if());
  return;
}

tensorkiln::ExecutionPlan plan =
    std::move(*compiled_plan.value_if());
```

`ExecutionPlanCompiler` does not run DCE or structural canonicalization. It
preflights the selected graph, obtains its reverse-verified arena projection,
and proposes only two kinds of decisions: one source-node/kernel pair per
compute step and one byte offset per arena buffer. Kernel selection currently
distinguishes equal-shape and broadcast `Add`, rank-2 and batched `MatMul`, and
contiguous `Relu`.

`ExecutionPlanVerifier` is the only construction path for the returned plan.
It independently reconstructs dense layouts, operand edges, input/constant/
arena storage, output mappings, arena lifetimes, work accounting, and all
limits from the source graph. It then validates the compiler's kernel choices
and placements against those facts. The candidate cannot supply trusted
operands, layouts, lifetimes, statistics, or constant data.

The move-only result owns a copy of the selected graph, its constant payloads,
the verified plan records, and the verified arena projection. Compilation does
not prove numerical equivalence by itself; execution is differentially checked
against the independent interpreter. The full plan, session, lifetime, and
memory-integrity contract is in [execution.md](execution.md).

## Dead-code elimination: reachability

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

## Dead-code elimination: replay and validation

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

## Structural canonicalization

Structural canonicalization visits and maps every source definition; it is not
a reachability pass. Unique dead work therefore remains in the result. Merged
definitions map to their representative rather than being replayed. `Input` and
`Constant` definitions are always replayed individually. Both preserve names
and types; constants additionally preserve fingerprints and raw IEEE-754
payload bits.

The only v0 rewrite rules are:

1. exact CSE for `Add`, `MatMul`, and `Relu` when the explicit operation kind,
   ordered canonical operand IDs, and complete output `TensorType` match;
2. `Relu(Relu(x)) -> Relu(x)`.

Source order decides the representative. `Add(x, y)` and `Add(y, x)` are
different keys. The pass never uses pointers, source-owner tokens, hashes
without equality, or unordered operand sets as identity.

Distinct source values named by graph outputs must remain distinct result
values. Multiple labels of one source value must continue to alias one result
value. The pass enforces this equivalence relation in both directions: an
output definition may merge into a result that does not yet represent another
output definition, but two distinct output definitions never merge. Guarded
equivalents remain as separate canonical nodes.

The pass deliberately does not perform any of the following:

- dead-code elimination or input/constant CSE;
- operand sorting, commutation, reassociation, or distribution;
- `x + 0`, matrix identity, matrix zero, or constant folding;
- signed-zero, subnormal, infinity, or NaN canonicalization;
- `MatMul` reduction changes, FMA contraction, epilogue fusion, or operation
  motion;
- output-alias topology changes.

For the same bindings, whenever source and result executions both complete
under the caller's `ReferenceLimits`, tensors are bit-identical for every
source definition through provenance, not only at declared outputs. Output
labels, order, tensor types, alias classes, and exact `GraphLimits` are
preserved. Internal IDs, owners, intermediate pointer identity, reference
materialization totals, scalar-step totals, and execution-budget acceptance
are not preserved.

The first representative for a structural key is found in logarithmic time.
Output-alias validation is linear in nodes plus output declarations. A second
pass is graph-idempotent. A guarded duplicate can cause
`preserved_output_distinctions` to remain nonzero on that fixed point even when
no definitions merge.

## Dead-code elimination: equivalence boundary

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

`GraphProvenance::create()` is the pass-neutral construction boundary. It takes
the result graph, the immediate source graph, and one source-node set for every
result definition in dense result order. The factory resolves complete
owner-tagged handles through both graphs, rejects empty sets and ambiguous
reverse mappings, and sorts and de-duplicates each set deterministically. The
created provenance owns all of its handles and vectors; neither graph needs to
outlive it.

Immediate dead-code elimination has exactly one source definition for every
retained result definition. Eliminated definitions have no result entry.
Lookups compare complete owner-tagged handles, so a foreign handle with the same
printed ordinal does not match.

Structural canonicalization assigns every immediate source definition to
exactly one result entry. CSE and redundant-ReLU entries can therefore own
multiple source definitions while reverse lookup remains singular. Source sets
are ordered and de-duplicated deterministically. Passes that clone a definition
or combine unrelated source graphs will require a different
reverse-lookup/domain contract and are outside the current API.

`DeadCodeElimination::run(source, upstream)` composes the immediate rewrite with
existing provenance. The returned lineage points through the source graph to
the earlier definitions represented by `upstream`. The upstream result domain
must match the source graph exactly; an incompatible or incomplete map returns
`provenance_domain_mismatch`.

`StructuralCanonicalization::run(source, upstream)` applies the same domain
validation and composition rule. This permits DCE followed by canonicalization
without losing the IDs of live definitions in the original graph.

For `R` expanded root definitions, composition sorts each result's root set
and indexes globally assigned root ordinals in `O(R log R)` time with `O(R)`
additional space. It does not perform a pairwise scan across roots.

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

`StructuralCanonicalizationStats` reports source, result, and merged definition
counts; exact common subexpressions; redundant ReLUs; and source definitions
kept distinct by the output guard. Its accounting invariant is
`merged_nodes == common_subexpressions + redundant_relus`.

## Diagnostics

A verified source that cannot be replayed with the same limits, or whose
retained operation derives a different type, returns
`compiler_internal_invariant`. This diagnostic indicates a TensorKiln defect,
not malformed caller input.

An incompatible provenance domain returns `provenance_domain_mismatch`.
Allocation failure remains a C++ runtime failure rather than a typed compiler
diagnostic.
