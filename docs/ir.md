# Typed graph IR

This document describes the graph layer that exists today. Reference execution
is specified separately in [the interpreter contract](reference.md). The
available graph-to-graph rewrites are specified in
[the compiler-pass contract](compiler.md). Dead-code elimination and structural
canonicalization are available; fusion, lowered layouts, and arena allocation
belong to later layers.

## Construction boundary

`GraphBuilder` is the only public graph-construction path. Callers supply exact
input/constant tensor types and receive `ValueId` handles from successful
definitions. Computational methods derive their result type:

```cpp
ValueId x = unwrap(builder.input("x", f32({2, 3})));
ValueId bias = unwrap(builder.constant("bias", f32({3}), bias_data));
ValueId sum = unwrap(builder.add(x, bias));
ValueId result = unwrap(builder.relu(sum));
unwrap(builder.output("result", result));
VerifiedGraph graph = unwrap(std::move(builder).finish());
```

The builder currently supports `Input`, `Constant`, `Add`, `MatMul`, and
`Relu`. `Add` uses trailing multidirectional broadcasting. `MatMul` accepts
rank 2 through 4, broadcasts batch prefixes, checks `K`, and returns
`broadcast(batch(A), batch(B)) + [M,N]`.

These rules follow the relevant parts of the official
[ONNX broadcasting](https://onnx.ai/onnx/repo-docs/Broadcasting.html) and
[MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html) specifications. The
GraphBuilder is a native C++ API; it is not an ONNX parser or compatibility
claim.

## Transactional failures

Every public mutation is transactional with respect to typed validation
errors. A failed call leaves all of the following unchanged:

- the next node, value, or output ordinal;
- existing definitions and output labels;
- definition/output name availability;
- the cumulative constant-element counter.

The builder remains usable after a failure. A failed `finish()` caused by a
missing output also leaves it usable. A successful `finish()` consumes the
builder; every later operation returns `builder_finished`.

Validation follows a stable high-level order:

1. reject an already consumed builder;
2. validate and de-duplicate names where a call defines a name;
3. resolve graph-owned operand handles from left to right;
4. validate operation semantics and derive the result type;
5. enforce per-tensor and graph-wide resource ceilings;
6. commit one complete definition or output label.

Allocation failure is reported by the C++ runtime rather than converted into a
typed graph diagnostic. The builder reserves container capacity before its
single commit step, so ordinary typed failures never expose a partial node.

## Identity and ownership

Each definition produces one value in the current IR. Nodes, values, and graph
outputs receive dense zero-based ordinals in insertion order.

`ValueId` and `NodeId` also carry an opaque process-local owner token. Their
printed ordinal is deterministic, but passing a handle from another builder or
graph fails lookup even when both handles print as `%0` or `#n0`. The owner
counter terminates before wraparound, so it cannot silently issue a duplicate
token. `OutputId` has no cross-graph lookup API and therefore needs only its
ordinal.

A rewritten graph receives a fresh owner token. Source handles are therefore
not valid in a compiler result even when their printed ordinals are unchanged;
the pass-provided provenance map is the supported bridge between those owner
domains.

Definition names and output labels occupy separate namespaces. This permits a
pass-through graph such as `output @x = %0` where `%0` is `input @x`. Within a
namespace, names are unique and must match this ASCII grammar:

```text
[A-Za-z_][A-Za-z0-9_.-]*
```

The narrow grammar keeps the canonical dump unambiguous without an escaping
scheme.

## Resource ceilings

`GraphLimits` bounds work before the builder retains it:

| Limit | Scope |
| --- | --- |
| `max_nodes` | Inputs, constants, and computational definitions combined. |
| `max_outputs` | Declared graph output labels. |
| `max_name_bytes` | Each definition name or output label. |
| `max_constant_elements` | All owned constant payloads combined. |
| `shape_limits.max_elements` | Every supplied or inferred tensor. |
| `tensor_limits.max_bytes` | Every supplied or inferred tensor. |

Zero is a valid ceiling and rejects the first corresponding resource. Exact
boundaries are accepted. Checked arithmetic and semantic errors take precedence
over a derived result's resource limit.

A `VerifiedGraph` retains the exact `GraphLimits` under which it was built.
Structural compiler passes replay surviving definitions with those same limits
rather than silently substituting defaults.

## Canonical dump

The printer emits definitions in their topological insertion order followed by
declared outputs:

```text
tensorkiln.graph v0 {
  #n0 %0 = input @x : f32[2,3]
  #n1 %1 = constant @bias {elements=3, fnv1a64=0x99e02dab84d74dd8} : f32[3]
  #n2 %2 = add %0, %1 : f32[2,3]
  #n3 %3 = relu %2 : f32[2,3]
  #o0 output @result = %3
}
```

Constant fingerprints are FNV-1a-64 over each `f32` bit pattern serialized
least-significant byte first. Element count is printed separately. The hash is
a deterministic regression fingerprint, not a security primitive or a claim
of collision-free model identity. The graph retains the complete constant data.
