# TensorKiln CLI contract

The `tensorkiln` executable is a bounded inspection surface for reproducible
workloads compiled into the program. It exercises the real public graph and
plan APIs without introducing a graph serialization format.

## Build and discover workloads

```bash
make -j2 PROFILE=release cli
build/g++/release/tensorkiln list
build/g++/release/tensorkiln list --format=json
```

The current registry contains one workload:

| ID | Input | Computation | Output |
| --- | --- | --- | --- |
| `dense_relu_v1` | `x: f32[2,3]` | `MatMul(f32[3,2]) -> Add(f32[2]) -> Relu` | `result: f32[2,2]` |

Registry order and identifiers are stable within the
`tensorkiln.cli.workloads.v1` schema. A future workload must receive a new
identifier instead of silently changing this graph.

## Inspect a verified plan

```bash
build/g++/release/tensorkiln inspect \
  --workload dense_relu_v1 \
  --format=json
```

The command constructs the workload with `GraphBuilder`, finalizes a
`VerifiedGraph`, and calls `ExecutionPlanCompiler`. That compiler submits its
kernel and arena-placement decisions to the independent plan verifier before
the CLI can publish a report.

The `tensorkiln.cli.inspect.v1` JSON object contains:

- a compiled-in workload descriptor with exact input and output shapes;
- verified value, input, constant, step, output, constant-byte,
  scalar-work, and logical-workspace counts;
- one record per selected kernel, including step and source-node ordinals; and
- the complete canonical `ExecutionPlan::dump()` string for human review and
  regression comparison.

No timing field is emitted, and the command is not a benchmark. The
`workspace_bytes` field is the logical arena size from the verified plan; it
does not claim to include allocator metadata, outer guards, or any future
audit shadow.

Text output is the default. `--format=json` and
`--format json` are equivalent. Successful output is fully staged under the
classic locale before it is written to the provided stream as one payload,
flushed, and checked; a failed write cannot be reported as success.

## Errors and exit codes

Expected failures write only to stderr. With `--format=json`, stderr contains
one `tensorkiln.cli.error.v1` object:

```json
{"schema":"tensorkiln.cli.error.v1","error":{"code":"unknown_workload","message":"unknown workload 'missing'"}}
```

| Exit | Meaning |
| ---: | --- |
| `0` | Command completed and its output was written completely. |
| `2` | Invalid command, option, format, workload, or argument envelope. |
| `3` | The compiled-in workload failed a typed graph or plan build stage. |
| `70` | Unexpected software failure or incomplete stdout/stderr write. |

The parser accepts at most 32 arguments, at most 8192 bytes per argument, and
at most 16384 bytes in total. Arguments must be printable ASCII. These limits
bound reflected diagnostics and prevent locale- or encoding-dependent JSON.
Duplicate options are rejected rather than resolved by ordering.

## Scope boundary

The CLI does not parse `VerifiedGraph::dump()` or
`ExecutionPlan::dump()`. Those are deterministic inspection formats, not
replay formats: graph dumps intentionally omit constant payloads. The public
v0.1 input remains the native C++ `GraphBuilder`, as specified in the
[target charter](charter.md).

The current command surface inspects a plan; it does not yet accept tensor
inputs or execute arbitrary user workloads. Runtime execution evidence remains
the responsibility of the checked release examples until a separately tested
CLI execution schema is added.

## Verification

`make test` runs both in-process parser/report tests and a black-box subprocess
contract against the actual executable. The checks cover stable JSON replay,
the exact three-kernel plan, typed errors, stdout/stderr separation, argument
limits, locale and stream-flag independence, and failed-output handling. The
same checks run in debug, release, coverage, and sanitizer profiles.
