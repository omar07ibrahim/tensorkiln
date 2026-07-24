# Verified arena storage planning

TensorKiln ships two related storage-only API layers:

- `ArenaPlanner::run()` and `ArenaPlacementVerifier::verify()` consume explicit
  buffer sizes, lifetimes, and placements;
- `GraphArenaLowering::run()` and `GraphArenaPlacementVerifier::verify()` derive
  those sizes and lifetimes from a `VerifiedGraph` containing the currently
  supported materializing operations.

The explicit layer returns a read-only `ArenaPlan`. The graph layer returns a
move-only `GraphArenaLoweringResult` that owns the value-to-buffer mapping, the
exact derived requests, and a verified `ArenaPlan`. Neither layer returns an
executable plan: they do not lower layouts, prove view or in-place alias
legality, derive kernel scratch, prepack constants, allocate memory, or execute
kernels.

## Public boundary

```cpp
constexpr std::array<tensorkiln::ArenaBufferRequest, 4> requests{{
    {96, 0, 2},
    {64, 0, 1},
    {32, 1, 3},
    {80, 2, 4},
}};

auto planned = tensorkiln::ArenaPlanner::run(requests);
if (planned.error_if() != nullptr) {
  report(*planned.error_if());
  return;
}
tensorkiln::ArenaPlan plan = std::move(*planned.value_if());
```

The request vector is an ordered identity domain. Position zero is buffer
`#b0`, position one is `#b1`, and so on. Reordering requests changes buffer
identities and may change the deterministic packing. Inputs are borrowed only
during the call. A successful plan owns copied allocation metadata and the
exact accepted `ArenaLimits`.

`ArenaPlacementVerifier::verify(requests, placements, limits)` accepts the same
request domain plus one `ArenaPlacement` per buffer. The placement list itself
may be in any order. A successful result is canonicalized into request-ordinal
order, so equivalent valid placement lists produce the same plan dump.

## Request and lifetime contract

Each request contains a positive payload size and a half-open lifetime
`[live_begin_step, live_end_step_exclusive)`. Buffer `#b` is live at step `s`
exactly when:

```text
live_begin_step <= s && s < live_end_step_exclusive
```

The beginning must be strictly less than the end. `UINT32_MAX` is a valid
exclusive endpoint, so the largest representable live step is
`UINT32_MAX - 1`. If an operation consumes a buffer during step `i`, the caller
must include that step in the interval; when `i + 1` is representable, this
requires `live_end_step_exclusive >= i + 1`.

End events occur before begin events at the same boundary. Two requests with
lifetimes `[0,1)` and `[1,2)` may therefore reuse the same bytes. Two separate
requests whose lifetimes intersect may not overlap in bytes.

Requests describe storage roots, not logical tensor values. Callers of the
explicit API must collapse intentional simultaneous aliases or views into one
root request and extend that root's lifetime across every use. Supplying two
distinct live requests at the same offset is rejected even if the caller
intended them to alias. `ArenaPlacementVerifier` proves safety only for the
sizes and lifetimes it receives; it does not prove that those inputs were
derived correctly.

## Graph-derived storage projection

`GraphArenaLowering::run(source, limits)` borrows one valid, non-moved-from
`VerifiedGraph` for the call and creates a storage projection for exactly that
graph. `GraphArenaPlacementVerifier::verify()` has the same source precondition.
Neither API runs dead-code elimination, canonicalization, or any other compiler
pass. Callers can compose the current stages explicitly:

```text
VerifiedGraph
  -> optional DeadCodeElimination
  -> optional StructuralCanonicalization
  -> GraphArenaLowering::run(selected_graph)
```

The graph projection applies these exact rules:

1. Visit definitions in source-node order. Each `Add`, `MatMul`, and `Relu`
   result receives one dense execution step and one dense buffer ordinal.
   `Input` and `Constant` values remain external and consume neither domain.
2. The request payload is the verified output type's exact byte count. All dead
   compute remains present unless the caller ran DCE first.
3. Let `C` be the compute-step count and let a buffer be produced at step `p`.
   Its lifetime begins at `p`. Its exclusive end is the maximum of `p + 1`,
   every consuming compute step plus one, and `C` when the value is named by at
   least one graph output.
4. Run the deterministic `ArenaPlanner` on those requests.
5. Independently reconstruct the graph-to-buffer mapping and requests in
   `GraphArenaPlacementVerifier`, verify the supplied placements, and require
   exact agreement on source and step counts, owner-tagged values, requests,
   limits, statistics, and every allocation field.

An operation therefore overlaps its result with every arena-backed operand for
the whole consumer step. An arena-backed output remains live through the final
compute step and ends at exclusive terminal boundary `C`; multiple output
labels for one value do not create extra buffers. The successful artifact
proves byte-placement safety for this sequential storage model. It does not
prove layout, strides, views, alias roots, in-place legality, prepacking,
scratch requirements, parallel scheduling, numerical execution, or minimum
possible workspace.

`GraphArenaPlacementVerifier::verify(source, placements, limits)` exposes the
reverse path directly. Placement `#bN` addresses the Nth materializing result
in source-node order, even when its graph value ordinal differs because inputs
and constants are interspersed. This API is useful for checking a separately
created placement without trusting separately supplied lifetimes.

## Alignment and statistics

The alignment is fixed at 64 bytes. Payload size is rounded up with checked
arithmetic, and every placement offset must be a multiple of 64. Offsets are
relative to an arena base: the eventual allocator must also align that base
address to 64 bytes.

`ArenaPlanStats` has exact definitions:

- `buffer_count` is the number of requests;
- `total_payload_bytes` is the checked sum of unrounded payload sizes;
- `total_reserved_bytes` is the checked sum of each 64-byte-rounded size;
- `peak_live_reserved_bytes` is the maximum sum of rounded sizes live at one
  step;
- `workspace_bytes` is zero for an empty plan, otherwise the maximum of
  `offset_bytes + reserved_bytes` across allocations.

Aggregate payload and reservation totals must fit in `uint64_t`. Overflow is a
typed failure even when lifetime reuse might make a smaller physical workspace
possible. An externally supplied, valid placement may contain gaps, so its
`workspace_bytes` can exceed `total_reserved_bytes`.

Workspace accounting in the explicit layer is semantic-agnostic and includes
every supplied request. It excludes only resources omitted from the request
list, along with an aligned allocator's base over-allocation and metadata. The
current graph projection omits external inputs and immutable constants and
includes every `Add`, `MatMul`, and `Relu` result. Prepacked weights,
metadata-only views, aliases, and kernel scratch do not yet exist in that
projection.

## Exact explicit-placement verifier order

`ArenaPlacementVerifier` validation has deterministic precedence:

1. Check request count against both the dense `uint32_t` ordinal domain and
   `limits.max_buffers`.
2. Visit requests in ordinal order. For each request, reject zero payload,
   invalid lifetime, alignment-rounding overflow, aggregate payload overflow,
   and aggregate reservation overflow, in that order.
3. Check that placement count equals request count.
4. Visit placements in supplied order. Reject an unknown ordinal, duplicate
   ordinal, unaligned offset, or `offset + reserved` overflow, in that order.
5. Sweep canonical lifetime events and reject intersecting live byte ranges.
   End events are removed before starts at the same boundary.
6. Reject `workspace_bytes > limits.max_workspace_bytes`. Equality is valid.
7. Reject a workspace that cannot be represented by host `size_t`.

Consequently, a structural overlap is reported before workspace policy, and
workspace policy is reported before host addressability. Placement-domain
diagnostics follow the caller's supplied placement order; valid output order is
always canonical. Internal active-set inconsistencies use
`compiler_internal_invariant` because they indicate a TensorKiln defect rather
than invalid caller input.

The overlap sweep is independent of the planner. It sorts starts and ends,
maintains live allocations by byte offset, and checks neighboring byte ranges
in `O(B log B)` time with `O(B)` auxiliary memory.

## Deterministic planner policy

The planner first applies the same request validation as the verifier. It then:

1. processes requests by `(live_begin_step, buffer_ordinal)`;
2. expires every active block with `end <= begin`;
3. coalesces adjacent expired blocks;
4. selects the smallest free block that fully fits, breaking equal-size ties by
   lower offset;
5. allocates the prefix of that block and returns any tail to the free index;
6. if no block fully fits, consumes a free block ending at the current
   high-water mark and grows only its missing suffix, when such a block exists;
7. otherwise appends the request at the current high-water mark.

Every generated placement is routed through `ArenaPlacementVerifier` before it
can become an `ArenaPlan`. The planner uses `O(B log B)` time and `O(B)`
auxiliary memory.

This is a deterministic heuristic, not a globally optimal packer. A failure at
the workspace limit describes this policy's result; it is not proof that no
different verified placement could satisfy the same limit. Determinism applies
to the same ordered requests and limits, not to an unordered multiset.

## Limits, failures, and ownership

The default policy ceilings are 4096 buffers and 256 MiB of workspace. Callers
may pass narrower or wider `ArenaLimits`; the accepted values are retained in
the plan and its dump. The workspace limit is inclusive.

Typed validation, policy, and host-representability failures return stable
diagnostics such as
`arena_buffer_size_invalid`, `arena_lifetime_invalid`,
`arena_alignment_invalid`, `arena_live_overlap`,
`arena_workspace_limit_exceeded`, and `arena_workspace_unaddressable`.
Checked arithmetic failures use `arena_size_overflow`. Ordinary C++ allocation
failure remains a runtime `std::bad_alloc`, and passing the addressability check
does not guarantee that a physical allocation will succeed.

`ArenaPlan` is copyable and movable. `limits()` and `stats()` return borrowed
references, `allocations()` returns a borrowed span, and `allocation_at()`
returns a borrowed pointer; all refer to plan-owned state. They remain valid
only while that plan is unchanged. Assigning to it, moving from it, or
destroying it invalidates them. After moving from a plan, only assignment and
destruction are supported.

`GraphArenaLoweringResult` is move-only and owns its source counts, dense
owner-tagged value mapping, requests, and verified `ArenaPlan`; the source graph
need not outlive it. Its lookups return no mapping for external, foreign, or
out-of-range values and deliberately do not classify which case occurred.
Moving from, assigning to, or destroying the result invalidates every borrowed
reference, pointer, and span. After moving from a result, the source supports
only assignment or destruction.

The public `GraphArenaPlacementVerifier` propagates placement validation, limit,
and host-addressability diagnostics. `GraphArenaLowering::run()` propagates
buffer-count, checked-size, workspace-policy, and host-addressability failures
from planning. A malformed derived request, an unexpected planner diagnostic,
a failure of mandatory reverse verification, or any forward/reverse
disagreement is `compiler_internal_invariant`, because it indicates a
TensorKiln defect rather than caller input. Ordinary C++ allocation failure
remains `std::bad_alloc`.

## Integration with executable plans

`ExecutionPlanCompiler` consumes the same graph-derived placement boundary when
it builds an executable dense plan. Its independent plan verifier reconstructs
the graph-to-arena requests again and requires the candidate offsets to satisfy
this contract before attaching kernels, layouts, storage classes, and work
accounting.

`ExecutionSession` then allocates one 64-byte-aligned workspace for the plan's
logical `workspace_bytes` and resolves every arena-backed value from its
verified offset. The session's outer guards and optional per-kernel write audit
are runtime protections layered on that allocation; they are not arena-plan
bytes and do not change placement statistics.

This integration does not make `ArenaPlan` or `GraphArenaLoweringResult`
executable on their own. Neither artifact owns kernels, input bindings, result
views, floating-point checks, or mutable workspace. Those contracts belong to
[verified dense execution](execution.md).

## Worked schedule

The runnable [arena example](../examples/plan_arena.cpp) supplies four storage
roots manually:

| Buffer | Payload | Reserved | Lifetime | Offset |
| --- | ---: | ---: | --- | ---: |
| `#b0` | 96 | 128 | `[0,2)` | 0 |
| `#b1` | 64 | 64 | `[0,1)` | 128 |
| `#b2` | 32 | 64 | `[1,3)` | 128 |
| `#b3` | 80 | 128 | `[2,4)` | 0 |

`#b2` reuses `#b1` exactly at step 1, and `#b3` reuses `#b0` exactly at step
2. Allocating a separate aligned region for every buffer would reserve 384
bytes; this schedule uses a 192-byte workspace and has a 192-byte peak. Equality
between peak and workspace is a fact about this fixture, not a general planner
guarantee or an optimality claim.

The example reverses the planner's placement list, submits it to the independent
verifier, and requires the same canonical dump. Focused tests additionally
cover misalignment, predecessor and successor overlap, gaps, exact policy
limits, overflow, split and coalescing behavior, frontier growth, 4096 and 4097
request boundaries, `UINT32_MAX` endpoints, seeded determinism, and comparison
with an independent pairwise overlap oracle.

The runnable [graph example](../examples/inspect_graph.cpp) explicitly applies
DCE and structural canonicalization before graph arena lowering. Its three
`f32[2,3]` compute results carry 24-byte payloads with lifetimes `[0,2)`,
`[1,3)`, and `[2,3)`. Their 64-byte reservations use offsets 0, 64, and 0, so a
valid 128-byte workspace holds 192 bytes of aligned reservations. This is a
verified non-executable storage projection; the example's separate reference
interpretation does not use that workspace.
