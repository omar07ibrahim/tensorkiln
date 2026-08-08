# TensorKiln CLI contract

The `tensorkiln` executable is a bounded inspection and audited-execution
surface for reproducible workloads compiled into the program. It exercises
the real public graph, plan, session, and reference APIs without introducing a
graph serialization format.

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

## Execute with exact raw bits

```bash
build/g++/release/tensorkiln execute \
  --workload dense_relu_v1 \
  --input-bits x=0x3f800000,0x40000000,0x40400000,0xbf800000,0x3f000000,0x40800000 \
  --format=json
```

`--input-bits` accepts exactly the binding `x=` followed by six comma-separated
values. Each value is lowercase `0x` plus eight hexadecimal digits; uppercase
hex digits are accepted and normalized on output. Decimal floating-point text,
underscores, whitespace, a second binding, missing values, and extra values are
rejected before a session is created.

Execution is deliberately not a fast path. The command always:

1. constructs and independently verifies the compiled-in plan;
2. creates an `ExecutionSession` with per-kernel write auditing enabled;
3. binds the six decoded `f32` values and executes synchronously;
4. copies the result as raw IEEE-754 binary32 bits;
5. runs the separate `ReferenceInterpreter` on the plan-owned graph; and
6. publishes output only after every result bit agrees.

The `tensorkiln.cli.execute.v1` JSON object contains the same workload
descriptor and plan facts as `inspect`, plus:

- `run_status`, `kernel_write_audit`, and logical workspace bytes;
- the exact canonical input and output bit strings with names, dtypes, and
  shapes;
- a `raw_f32_bits` reference-agreement record; and
- an explicit fixture scope and `benchmark: false`.

For the command above, `result: f32[2,2]` is
`[0x40900000, 0x41300000, 0x00000000, 0x41300000]`, representing
`[4.5, 11, 0, 11]`. The
[captured inspect JSON](visuals/generated/cli-inspect.json),
[captured execute JSON](visuals/generated/cli-execute.json), and
[data-derived workflow](visuals/generated/cli-execution.svg) are generated
from the release binary. Each command is run twice under `LANG=C`,
`LC_ALL=C`, and `TZ=UTC`; publication requires byte-identical stdout, empty
stderr, and exit status zero.

Raw bits avoid decimal parsing and locale ambiguity, but this narrow fixture
does not establish arbitrary-input, arbitrary-platform, or performance
behavior. The process expects the supported floating-point environment
documented in [the numerical policy](numerics.md); it does not change exception
masks to accommodate signaling values.

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
| `3` | A typed TensorKiln graph, plan, binding, or reference diagnostic. |
| `4` | Audited session execution returned a non-success run status. |
| `5` | Executor and independent-reference output metadata or bits differed. |
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

The current command surface can execute only `dense_relu_v1` with one exact
`f32[2,3]` raw-bit binding. It cannot import, deserialize, or execute arbitrary
graphs or models. Adding a second workload requires a new stable identifier,
an explicit input contract, independent-reference comparison, and
process-level replay tests.

## Verification

`make test` runs both in-process parser/report tests and a black-box subprocess
contract against the actual executable. The checks cover stable JSON replay,
the exact three-kernel plan, two distinct executed inputs, raw-bit
normalization, write auditing, reference agreement, a forced run-status
failure, typed errors, stdout/stderr separation, argument limits, locale and
stream-flag independence, and failed-output handling. The same checks run in
debug, release, coverage, and sanitizer profiles.
