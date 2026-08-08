# Coverage evidence contract

TensorKiln publishes one reviewable GCC/LCOV observation of the implemented
runtime. The bundle answers a narrow question: which executable production
lines, functions, and GCC control-flow edges under `src/` were reached by one
clean run of the current deterministic C++ suite and checked examples?

[![TensorKiln GCC and LCOV production-source coverage](generated/summary.svg)](generated/summary.svg)

The visual is rendered from the same validated LCOV records as the text and
machine-readable artifacts. It is not a hand-edited badge. The lower workflow
in the panel is the actual capture path, and the review queue is ordered by the
number of uncovered executable lines so that large remaining sets stay
visible.

## Setup and reproduction

The normal library and test profiles remain dependency-free. Coverage capture
additionally requires GNU `g++`, its matching `gcov`, Python 3.11 or newer, GNU
Make, and LCOV 2.x. On Ubuntu 24.04, LCOV is available from the distribution
package:

```bash
sudo apt-get install lcov
make COVERAGE_JOBS=2 coverage
make COVERAGE_JOBS=2 coverage-check
```

`coverage` removes only `build/<compiler>/coverage`, performs a fresh
instrumented build, runs all four checked examples and the complete C++ test
binary exactly once, then publishes the four payload files and the manifest
last under `generated/`.
`coverage-check` repeats that clean capture and fails unless every committed
byte is identical. `COVERAGE_JOBS` affects compilation concurrency only; the
examples and tests execute in their deterministic Makefile order.

Use explicit matching tools when multiple GCC or LCOV installations coexist:

```bash
make COVERAGE_JOBS=2 CXX=g++-13 GCOV=gcov-13 \
  LCOV=lcov GENINFO=geninfo coverage-check
```

The recorder rejects a compiler/gcov major-minor mismatch, an lcov/geninfo
version mismatch, and LCOV versions outside the supported 2.x trace format.
Byte-identical reproduction requires the versions recorded in
[`manifest.json`](generated/manifest.json); a different compiler may assign
different executable lines or control-flow edges even when runtime behavior is
unchanged.

The capture fixes `PATH` to `/usr/bin:/bin`. If the caller has set
`PERL5LIB`, the recorder passes it through only so a non-system LCOV
installation can locate its Perl modules. The public manifest discloses that
policy without publishing a host path. That module lookup remains an explicitly
unbound tool-installation input: exact reproduction requires an equivalent
effective LCOV module tree, while a normal distribution installation should
leave `PERL5LIB` unset.

## What is measured

| Included in the run | Included in the reported denominator |
| --- | --- |
| All C++ tests | No |
| All four checked examples | No |
| Public headers used by those programs | No |
| Executable records emitted for `src/*.cpp` | Yes |
| Executable records emitted for internal `src/*.hpp` | Yes |
| Standard-library and system headers | No |

In short, tests and examples are excluded from the published denominator while
still being the workloads that exercise production code. A header under
`src/` appears only when GCC emits executable coverage records for it.
Non-executable source lines never enter LCOV's `LF` count.

The current bundle contains:

- [`summary.svg`](generated/summary.svg), a self-contained data-derived visual;
- [`summary.txt`](generated/summary.txt), the exact totals and largest
  uncovered-line sets;
- [`coverage.info`](generated/coverage.info), the normalized LCOV tracefile;
- [`test-run.txt`](generated/test-run.txt), complete stdout from the real
  examples and test binary; and
- [`manifest.json`](generated/manifest.json), hashes, tool identities, exact
  source-input hashes, capture boundaries, and the independently verified
  totals.

The manifest deliberately distinguishes the latest commit touching its direct
input set from the working-tree snapshot. Every regular file under the CLI,
C++ source, include, example, and test roots, plus the Makefile and recorder,
is hashed by bytes and Git blob ID. The only exclusions are inert
`__pycache__`, `.mypy_cache`, `.pytest_cache`, and `.ruff_cache` directories.
Untracked, ignored, and deleted tracked inputs force `commit_bound` to false;
this prevents an untracked header such as `include/vector` from shadowing a
standard header invisibly.
Evidence generated during a staged publication therefore cannot silently
pretend that an older commit contains its inputs. An artifact-only publication
commit does not invalidate this selection because it does not change any
measured or capture input. When the selected paths are clean, the recorder
additionally requires every working blob to match the selected commit before
setting `commit_bound` to true.

## Fail-closed capture path

The standard-library-only recorder performs these checks before publishing:

1. resolves regular executable tool files without a shell and verifies matching
   GCC/gcov and lcov/geninfo versions;
2. deletes only the dedicated coverage build directory, then runs
   `make -s -j2 CXX=g++ PROFILE=coverage test` in a narrow `C`/UTC
   environment;
3. requires empty build/test stderr, all checked example sentinels, every
   individual `[pass]` line, and an exact `N/N tests passed` summary;
4. requires one `.gcda`/`.gcno` pair for every production translation unit;
5. invokes `geninfo` with branch collection, external records disabled,
   checksums disabled, one capture worker, and an empty LCOV configuration;
6. filters to real regular files under `src/`, rejects duplicate/escaping
   sources, and requires all 25 production `.cpp` units;
7. independently parses every function, branch, and line counter, recomputes
   `FNF/FNH`, `BRF/BRH`, and `LF/LH`, then cross-checks those totals against
   LCOV's own summary;
8. rewrites source paths to repository-relative POSIX paths, sorts all records
   deterministically, and rejects host paths, personal identifiers,
   credential-shaped text, active SVG content, or an unexpected artifact set;
9. hashes the four non-manifest payload artifacts plus each direct source
   input, then publishes the manifest last as the bundle integrity marker; and
10. in check mode, recaptures from a newly reset counter directory before
    requiring byte equality.

No coverage percentage is hard-coded into the renderer or tests. The SVG, text
summary, manifest metrics, and displayed review queue are constructed from the
validated trace in memory.

## Interpretation and limitations

This evidence is an observation, not a release gate and not a quality score.
High line coverage does not prove numerical correctness, ownership safety,
resource-bound enforcement, or absence of undefined behavior; those claims
remain the job of specific tests, the independent interpreter/oracles, the
plan and arena verifiers, the allocation probe, and sanitizer runs.

GCC branch coverage counts compiler-generated control-flow edges, including
exception-handling paths. It is useful for locating unexercised decisions but
is not interchangeable with source-level condition coverage, MC/DC, or a
Clang profile. Function and branch denominators can change across compiler
versions. The capture does not measure performance and contains no timing
claim.

The recorder does not enforce network isolation. TensorKiln has no configured
network step in this build or suite, but the manifest states the technical
boundary rather than claiming an unavailable sandbox. Coverage also does not
replace the release-profile no-allocation probe: instrumentation changes the
binary and is therefore kept in its own profile.

When behavior or tests change, regenerate the bundle, inspect the changed
trace and review queue, run `coverage-check`, and update this contract if the
scope itself changed. A percentage increase alone is not sufficient reason to
accept a test; the test must still assert a meaningful contract.
