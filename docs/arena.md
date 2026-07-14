# Interval arena planning

TensorKiln ships a standalone storage-placement core with two public entry
points: `ArenaPlanner::run()` creates a deterministic placement, and
`ArenaPlacementVerifier::verify()` independently validates a supplied
placement. Both consume explicit buffer sizes and execution lifetimes and
produce a read-only `ArenaPlan` artifact.

This is the storage-placement portion of a future arena stage, not graph-to-plan
integration. It does not inspect a `VerifiedGraph`, derive liveness, prove view
or in-place alias legality, derive requirements for kernel scratch, allocate
memory, or execute kernels.

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

Requests describe storage roots, not logical tensor values. A future lowerer
must collapse intentional simultaneous aliases or views into one root request
and extend that root's lifetime across every use. Supplying two distinct live
requests at the same offset is rejected even if the caller intended them to
alias. The verifier proves safety only for the sizes and lifetimes it receives;
it does not prove that those inputs were derived correctly.

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

Workspace accounting is semantic-agnostic and includes every supplied request.
It excludes only resources omitted from the request list, along with an aligned
allocator's base over-allocation and metadata. A future lowerer is expected to
omit external inputs, immutable constants, prepacked weights, and metadata-only
views, while representing workspace-backed kernel scratch as storage-root
requests.

## Exact verifier order

Validation has deterministic precedence:

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
