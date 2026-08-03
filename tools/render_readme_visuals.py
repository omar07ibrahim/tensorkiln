#!/usr/bin/env python3
"""Render README visuals from verified TensorKiln release examples.

The renderer never invents example output.  It executes already-built release
binaries, validates their verification sentinels, and derives every displayed
value from stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence, TypeAlias


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR: Final = Path("build/g++/release")
DEFAULT_OUTPUT_DIR: Final = Path("docs/visuals/generated")
MAX_OUTPUT_BYTES: Final = 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 8 * 1024 * 1024
MAX_BINARY_BYTES: Final = 64 * 1024 * 1024
MAX_SOURCE_BYTES: Final = 16 * 1024 * 1024
EXAMPLE_TIMEOUT_SECONDS: Final = 30
GIT_TIMEOUT_SECONDS: Final = 30
PROCESS_REAP_TIMEOUT_SECONDS: Final = 5
PROCESS_EXIT_POLL_SECONDS: Final = 0.01
GIT_BINARY: Final = Path("/usr/bin/git")
CLI_REPLAYS: Final = 2
Artifact: TypeAlias = str | bytes

PLAN_SENTINELS: Final = (
    "=== verified interval arena plan ===",
)
EXECUTE_SENTINELS: Final = (
    "=== verified dense execution plan ===",
    "result = [4.5, 11, 0, 11]",
    (
        "verified: audited execution matches the independent reference "
        "bit for bit"
    ),
)
SOFTMAX_SENTINELS: Final = (
    "=== verified Softmax execution ===",
    "scope: deterministic correctness example; not a benchmark",
    (
        "agreement {executor_reference_bits=20/20, "
        "executor_fixture_bits=20/20}"
    ),
    "=== optimized axis boundary ===",
    "reference_axis0 {status=accepted, scalar_steps=80}",
    (
        "optimized_axis0 {code=plan_operation_unsupported, "
        "message=plan backend does not support softmax axis 0 at #n1}"
    ),
    (
        "verified: last-axis execution and reference agree; valid axis 0 "
        "remains reference-only"
    ),
)
SOFTMAX_EXPECTED_LINES: Final = (
    "=== verified Softmax execution ===",
    "scope: deterministic correctness example; not a benchmark",
    (
        "plan {shape=f32[5,4], axis=1, kernel=softmax_last_axis_f32, "
        "scalar_steps=60, workspace_bytes=128, audited=true}"
    ),
    (
        "slice finite_equal bits=[0x3e800000, 0x3e800000, "
        "0x3e800000, 0x3e800000]"
    ),
    (
        "slice nan_precedence bits=[0x7fc00000, 0x7fc00000, "
        "0x7fc00000, 0x7fc00000]"
    ),
    (
        "slice positive_infinity_split bits=[0x3f000000, 0x00000000, "
        "0x3f000000, 0x00000000]"
    ),
    (
        "slice all_negative_infinity bits=[0x7fc00000, 0x7fc00000, "
        "0x7fc00000, 0x7fc00000]"
    ),
    (
        "slice mixed_negative_infinity bits=[0x00000000, 0x3f000000, "
        "0x3f000000, 0x00000000]"
    ),
    (
        "agreement {executor_reference_bits=20/20, "
        "executor_fixture_bits=20/20}"
    ),
    "=== optimized axis boundary ===",
    "reference_axis0 {status=accepted, scalar_steps=80}",
    (
        "optimized_axis0 {code=plan_operation_unsupported, "
        "message=plan backend does not support softmax axis 0 at #n1}"
    ),
    (
        "verified: last-axis execution and reference agree; valid axis 0 "
        "remains reference-only"
    ),
)

CLI_INSPECT_ARGUMENTS: Final = (
    "inspect",
    "--workload",
    "dense_relu_v1",
    "--format=json",
)
CLI_EXECUTE_ARGUMENTS: Final = (
    "execute",
    "--workload",
    "dense_relu_v1",
    "--input-bits",
    (
        "x=0x3f800000,0x40000000,0x40400000,"
        "0xbf800000,0x3f000000,0x40800000"
    ),
    "--format=json",
)
CLI_INPUT_BITS: Final = (
    "0x3f800000",
    "0x40000000",
    "0x40400000",
    "0xbf800000",
    "0x3f000000",
    "0x40800000",
)
CLI_OUTPUT_BITS: Final = (
    "0x40900000",
    "0x41300000",
    "0x00000000",
    "0x41300000",
)
CLI_PLAN_STATS: Final = {
    "values": 6,
    "inputs": 1,
    "constants": 2,
    "steps": 3,
    "outputs": 1,
    "constant_bytes": 32,
    "scalar_steps": 20,
    "workspace_bytes": 128,
}
CLI_KERNELS: Final = (
    {
        "step": 0,
        "source_node": 2,
        "kind": "matmul_rank2_f32",
        "scalar_steps": 12,
    },
    {
        "step": 1,
        "source_node": 4,
        "kind": "add_broadcast_f32",
        "scalar_steps": 4,
    },
    {
        "step": 2,
        "source_node": 5,
        "kind": "relu_contiguous_f32",
        "scalar_steps": 4,
    },
)
CLI_WORKLOAD: Final = {
    "id": "dense_relu_v1",
    "kind": "compiled_in",
    "description": (
        "f32[2,3] -> MatMul(f32[3,2]) -> Add(f32[2]) -> Relu"
    ),
    "inputs": [
        {
            "name": "x",
            "dtype": "f32",
            "shape": [2, 3],
            "elements": 6,
        }
    ],
    "outputs": [
        {
            "name": "result",
            "dtype": "f32",
            "shape": [2, 2],
            "elements": 4,
        }
    ],
}
CLI_CANONICAL_DUMP: Final = """\
tensorkiln.execution_plan v0 {
  limits {values=4096, steps=4096, outputs=64, constant_bytes=268435456, scalar_steps=1073741824, arena_buffers=4096, arena_workspace_bytes=268435456}
  stats {values=6, inputs=1, constants=2, steps=3, outputs=1, constant_bytes=32, scalar_steps=20, workspace_bytes=128}
  values {
    %0 f32[2,3] dense strides=[3,1] storage=input #i0 name=x
    %1 f32[3,2] dense strides=[2,1] storage=constant #c0 name=weight fingerprint=6640413917219750661
    %2 f32[2,2] dense strides=[2,1] storage=arena #b0 offset=0
    %3 f32[2] dense strides=[1] storage=constant #c1 name=bias fingerprint=5978795021561992053
    %4 f32[2,2] dense strides=[2,1] storage=arena #b1 offset=64
    %5 f32[2,2] dense strides=[2,1] storage=arena #b2 offset=0
  }
  steps {
    @0 #n2 %2 = matmul_rank2_f32(%0,%1) work=12
    @1 #n4 %4 = add_broadcast_f32(%2,%3) work=4
    @2 #n5 %5 = relu_contiguous_f32(%4) work=4
  }
  outputs {
    #o0 result -> %5
  }
  arena {
    #b0 offset=0 payload=16 reserved=64 live=[0,2)
    #b1 offset=64 payload=16 reserved=64 live=[1,3)
    #b2 offset=0 payload=16 reserved=64 live=[2,3)
  }
}
"""

REGLU_LIST_ARGUMENTS: Final = ("list", "--format=json")
REGLU_INSPECT_ARGUMENTS: Final = (
    "inspect",
    "--workload",
    "reglu_mlp_v1",
    "--format=json",
)
REGLU_EXECUTE_ARGUMENTS: Final = (
    "execute",
    "--workload",
    "reglu_mlp_v1",
    "--input-bits",
    (
        "x=0x3f800000,0x40000000,0x40400000,"
        "0xbf800000,0x3f000000,0x40800000"
    ),
    "--format=json",
)
REGLU_LIST_TEXT_ARGUMENTS: Final = ("list",)
REGLU_INSPECT_TEXT_ARGUMENTS: Final = (
    "inspect",
    "--workload",
    "reglu_mlp_v1",
)
REGLU_EXECUTE_TEXT_ARGUMENTS: Final = REGLU_EXECUTE_ARGUMENTS[:-1]
REGLU_INPUT_BITS: Final = CLI_INPUT_BITS
REGLU_OUTPUT_BITS: Final = (
    "0x00000000",
    "0x40a00000",
    "0x41480000",
    "0x40180000",
    "0x80000000",
    "0xc0f00000",
    "0x42040000",
    "0x00000000",
)
REGLU_PLAN_STATS: Final = {
    "values": 11,
    "inputs": 1,
    "constants": 4,
    "steps": 6,
    "outputs": 1,
    "constant_bytes": 128,
    "scalar_steps": 80,
    "workspace_bytes": 192,
}
REGLU_KERNELS: Final = (
    {
        "step": 0,
        "source_node": 2,
        "kind": "matmul_rank2_f32",
        "scalar_steps": 24,
    },
    {
        "step": 1,
        "source_node": 4,
        "kind": "add_broadcast_f32",
        "scalar_steps": 8,
    },
    {
        "step": 2,
        "source_node": 5,
        "kind": "relu_contiguous_f32",
        "scalar_steps": 8,
    },
    {
        "step": 3,
        "source_node": 7,
        "kind": "matmul_rank2_f32",
        "scalar_steps": 24,
    },
    {
        "step": 4,
        "source_node": 9,
        "kind": "add_broadcast_f32",
        "scalar_steps": 8,
    },
    {
        "step": 5,
        "source_node": 10,
        "kind": "mul_contiguous_f32",
        "scalar_steps": 8,
    },
)
REGLU_WORKLOAD: Final = {
    "id": "reglu_mlp_v1",
    "kind": "compiled_in",
    "description": (
        "f32[2,3] -> dual MatMul+Add branches -> "
        "Relu(gate) * value -> f32[2,4]"
    ),
    "inputs": [
        {
            "name": "x",
            "dtype": "f32",
            "shape": [2, 3],
            "elements": 6,
        }
    ],
    "outputs": [
        {
            "name": "result",
            "dtype": "f32",
            "shape": [2, 4],
            "elements": 8,
        }
    ],
}
REGLU_CANONICAL_DUMP: Final = """\
tensorkiln.execution_plan v0 {
  limits {values=4096, steps=4096, outputs=64, constant_bytes=268435456, scalar_steps=1073741824, arena_buffers=4096, arena_workspace_bytes=268435456}
  stats {values=11, inputs=1, constants=4, steps=6, outputs=1, constant_bytes=128, scalar_steps=80, workspace_bytes=192}
  values {
    %0 f32[2,3] dense strides=[3,1] storage=input #i0 name=x
    %1 f32[3,4] dense strides=[4,1] storage=constant #c0 name=W_gate fingerprint=17942347131671705029
    %2 f32[2,4] dense strides=[4,1] storage=arena #b0 offset=0
    %3 f32[4] dense strides=[1] storage=constant #c1 name=b_gate fingerprint=6861131082573093515
    %4 f32[2,4] dense strides=[4,1] storage=arena #b1 offset=64
    %5 f32[2,4] dense strides=[4,1] storage=arena #b2 offset=0
    %6 f32[3,4] dense strides=[4,1] storage=constant #c2 name=W_value fingerprint=17915088020765019496
    %7 f32[2,4] dense strides=[4,1] storage=arena #b3 offset=64
    %8 f32[4] dense strides=[1] storage=constant #c3 name=b_value fingerprint=2740866829651404690
    %9 f32[2,4] dense strides=[4,1] storage=arena #b4 offset=128
    %10 f32[2,4] dense strides=[4,1] storage=arena #b5 offset=64
  }
  steps {
    @0 #n2 %2 = matmul_rank2_f32(%0,%1) work=24
    @1 #n4 %4 = add_broadcast_f32(%2,%3) work=8
    @2 #n5 %5 = relu_contiguous_f32(%4) work=8
    @3 #n7 %7 = matmul_rank2_f32(%0,%6) work=24
    @4 #n9 %9 = add_broadcast_f32(%7,%8) work=8
    @5 #n10 %10 = mul_contiguous_f32(%5,%9) work=8
  }
  outputs {
    #o0 result -> %10
  }
  arena {
    #b0 offset=0 payload=32 reserved=64 live=[0,2)
    #b1 offset=64 payload=32 reserved=64 live=[1,3)
    #b2 offset=0 payload=32 reserved=64 live=[2,6)
    #b3 offset=64 payload=32 reserved=64 live=[3,5)
    #b4 offset=128 payload=32 reserved=64 live=[4,6)
    #b5 offset=64 payload=32 reserved=64 live=[5,6)
  }
}
"""
RAW_F32_BITS_PATTERN: Final = re.compile(r"^0x[0-9a-f]{8}$")
REGLU_STEP_PATTERN: Final = re.compile(
    r"^    @(?P<step>\d+) #n(?P<node>\d+) %(?P<result>\d+) = "
    r"(?P<kernel>[a-z0-9_]+)\((?P<operands>%\d+(?:,%\d+)*)\) "
    r"work=(?P<work>\d+)$"
)
REGLU_ARENA_PATTERN: Final = re.compile(
    r"^    #b(?P<buffer>\d+) offset=(?P<offset>\d+) "
    r"payload=(?P<payload>\d+) reserved=(?P<reserved>\d+) "
    r"live=\[(?P<start>\d+),(?P<end>\d+)\)$"
)

EVIDENCE_SOURCE_PATHSPECS: Final = (
    "Makefile",
    "examples/execute_graph.cpp",
    "examples/execute_softmax.cpp",
    "examples/plan_arena.cpp",
    "include",
    "src",
)
CLI_EVIDENCE_SOURCE_PATHSPECS: Final = ("cli",)
REQUIRED_EVIDENCE_SOURCES: Final = frozenset(
    {
        "Makefile",
        "examples/execute_graph.cpp",
        "examples/execute_softmax.cpp",
        "examples/plan_arena.cpp",
    }
)
REQUIRED_CLI_EVIDENCE_SOURCES: Final = frozenset(
    {"cli/tensorkiln.cpp"}
)
GENERATOR_PATH: Final = "tools/render_readme_visuals.py"
PUBLISHED_V3_ARTIFACT_SHA256: Final = {
    "arena-plan.txt": (
        "5cda0ab2cc42372c6991c2387aca6635212b35d7dbb2391d0af7895f2af1c0bb"
    ),
    "arena-reuse.svg": (
        "9c0baf3fce1d27bc0fc085e360edac24a5e795d31dff75f0a75a2c547bd36088"
    ),
    "cli-execute.json": (
        "2d74429690aa514770d5a137ebf958f8b475a5b1cb4425a36ae7a4e42566b86b"
    ),
    "cli-execution.svg": (
        "d4f0933e2759a0be0181ec4b131fd2656092496f9dfc48ffd40caa4a07e87839"
    ),
    "cli-inspect.json": (
        "bafba37b0c4ece4545ee011cdb904c070bcb456a5180ef1512b6b579340a8690"
    ),
    "execute-graph.svg": (
        "b8e6741259a7b582da5f14f111a9d821ef0279e5736fb0ad9ee3da6b1b442e8e"
    ),
    "execute-graph.txt": (
        "831194f4d7019e85ba32c7e2355dfc49b4caa3f4b1ae7b9db47b75bdae85a2e4"
    ),
    "execute-softmax.svg": (
        "2499cbb7ddd1877aeb040a0522deeab01ce4aff217c4af602b1289243c22f341"
    ),
    "execute-softmax.txt": (
        "a8fde11bd37ec79801fd9c7d8f370501b2d4197dbd5816ff6f461c7469ae5898"
    ),
}

UNSAFE_PATTERNS: Final = (
    (
        "absolute Unix host path",
        re.compile(
            r"(?<![A-Za-z0-9_.-])/"
            r"(?:home|root|Users|private|tmp|var|etc|opt|srv|mnt|"
            r"workspace|workspaces)(?:/|$)"
        ),
    ),
    (
        "absolute Windows user path",
        re.compile(
            r"\b[A-Za-z]:\\(?:Users|Documents and Settings)\\",
            re.IGNORECASE,
        ),
    ),
    (
        "email address",
        re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "local file URI",
        re.compile(r"\bfile://", re.IGNORECASE),
    ),
    (
        "private-key block",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.IGNORECASE,
        ),
    ),
    (
        "AWS access key",
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (
        "GitHub token",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|"
            r"github_pat_[A-Za-z0-9_]{20,})\b"
        ),
    ),
    (
        "OpenAI-style secret",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "bearer credential",
        re.compile(
            r"\bbearer\s+[A-Za-z0-9._~+/=-]{12,}",
            re.IGNORECASE,
        ),
    ),
    (
        "secret-bearing environment variable",
        re.compile(
            r"\b(?:AWS_SECRET_ACCESS_KEY|OPENAI_API_KEY|"
            r"GITHUB_TOKEN|GH_TOKEN)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential assignment",
        re.compile(
            r"\b(?:api[_-]?key|client[_-]?secret|"
            r"secret(?:[_-]access[_-]?key)?|password|passwd|"
            r"private[_-]?key|authorization|access[_-]?token|token)"
            r"\s*[:=]\s*\S+",
            re.IGNORECASE,
        ),
    ),
)

ALIGNMENT_PATTERN: Final = re.compile(r"^  alignment_bytes=(\d+)$")
STATS_PATTERN: Final = re.compile(
    r"^  stats \{buffers=(\d+), payload_bytes=(\d+), "
    r"reserved_bytes=(\d+), peak_live_reserved_bytes=(\d+), "
    r"workspace_bytes=(\d+)\}$"
)
ALLOCATION_PATTERN: Final = re.compile(
    r"^  #b(?P<ordinal>\d+) "
    r"offset=(?P<offset>\d+) "
    r"payload=(?P<payload>\d+) "
    r"reserved=(?P<reserved>\d+) "
    r"live=\[(?P<start>\d+),(?P<end>\d+)\)$"
)
NAIVE_PATTERN: Final = re.compile(
    r"^naive_separate_reservations_bytes=(\d+)$"
)
REUSED_PATTERN: Final = re.compile(r"^reused_workspace_bytes=(\d+)$")
VERIFIED_REUSE_PATTERN: Final = re.compile(
    r"^verified: "
    r"(?P<count>zero|one|two|three|four|five|six|seven|eight|nine|\d+) "
    r"boundary reuse(?:s)?, "
    r"(?P<workspace>\d+) bytes of workspace for "
    r"(?P<reserved>\d+) bytes of aligned reservations$"
)
COUNT_WORDS: Final = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


class VisualEvidenceError(RuntimeError):
    """Raised when an example cannot be used as visual evidence."""


class _ProcessOutputLimitExceeded(RuntimeError):
    """Raised after a bounded child stream emits one byte past its limit."""

    def __init__(self, stream: str, maximum_bytes: int) -> None:
        super().__init__(f"{stream} exceeds {maximum_bytes} bytes")
        self.stream = stream
        self.maximum_bytes = maximum_bytes


@dataclass(frozen=True)
class ArenaAllocation:
    """One allocation parsed from the verified arena example."""

    ordinal: int
    offset: int
    payload: int
    reserved: int
    live_start: int
    live_end: int


@dataclass(frozen=True)
class ArenaEvidence:
    """Validated arena values derived from ``plan_arena`` stdout."""

    alignment: int
    buffer_count: int
    total_payload: int
    total_reserved: int
    peak_live_reserved: int
    workspace: int
    allocations: tuple[ArenaAllocation, ...]


@dataclass(frozen=True)
class CliEvidence:
    """Cross-checked inspect and execute records from the release CLI."""

    inspect: dict[str, object]
    execute: dict[str, object]
    input_bits: tuple[str, ...]
    output_bits: tuple[str, ...]


@dataclass(frozen=True)
class RegluPlanStep:
    """One dependency-bearing step parsed from the ReGLU canonical plan."""

    step: int
    source_node: int
    result_value: int
    kernel: str
    operands: tuple[int, ...]
    scalar_steps: int


@dataclass(frozen=True)
class RegluArenaBuffer:
    """One source-derived arena interval from the ReGLU canonical plan."""

    buffer: int
    offset: int
    payload: int
    reserved: int
    live_start: int
    live_end: int


@dataclass(frozen=True)
class RegluEvidence:
    """Cross-checked ReGLU JSON, text transcripts, plan, and output bits."""

    workloads: dict[str, object]
    inspect: dict[str, object]
    execute: dict[str, object]
    list_text: str
    inspect_text: str
    execute_text: str
    steps: tuple[RegluPlanStep, ...]
    arena: tuple[RegluArenaBuffer, ...]
    input_bits: tuple[str, ...]
    output_bits: tuple[str, ...]


def reject_unsafe_text(label: str, text: str) -> None:
    """Reject host paths, credential patterns, and unsafe control bytes."""

    for character in text:
        codepoint = ord(character)
        if (codepoint < 32 and character not in "\n\t") or codepoint == 127:
            raise VisualEvidenceError(
                f"{label} contains a disallowed control character"
            )

    for description, pattern in UNSAFE_PATTERNS:
        if pattern.search(text):
            raise VisualEvidenceError(f"{label} contains {description}")


def validate_stdout(
    label: str, stdout: str, expected_sentinels: tuple[str, ...]
) -> str:
    """Validate normalized example stdout before it reaches a renderer."""

    normalized = stdout.replace("\r\n", "\n")
    if "\r" in normalized:
        raise VisualEvidenceError(f"{label} stdout contains a bare carriage return")
    if not normalized or not normalized.endswith("\n"):
        raise VisualEvidenceError(
            f"{label} stdout must be non-empty and newline-terminated"
        )
    try:
        normalized.encode("ascii", errors="strict")
    except UnicodeEncodeError as error:
        raise VisualEvidenceError(
            f"{label} stdout must contain printable ASCII evidence only"
        ) from error
    reject_unsafe_text(f"{label} stdout", normalized)

    output_lines = set(normalized.splitlines())
    missing = [
        sentinel for sentinel in expected_sentinels if sentinel not in output_lines
    ]
    if missing:
        raise VisualEvidenceError(
            f"{label} stdout is missing verification sentinel: {missing[0]!r}"
        )
    return normalized


def validate_softmax_stdout(stdout: str) -> str:
    """Require the complete, exact crafted-fixture Softmax report."""

    normalized = validate_stdout(
        "execute_softmax", stdout, SOFTMAX_SENTINELS
    )
    lines = tuple(normalized.splitlines())
    if lines != SOFTMAX_EXPECTED_LINES:
        mismatch = next(
            (
                index
                for index, pair in enumerate(
                    zip(lines, SOFTMAX_EXPECTED_LINES, strict=False)
                )
                if pair[0] != pair[1]
            ),
            min(len(lines), len(SOFTMAX_EXPECTED_LINES)),
        )
        raise VisualEvidenceError(
            "execute_softmax stdout differs from the complete crafted-fixture "
            f"contract at line {mismatch + 1}"
        )
    return normalized


def _reject_json_constant(value: str) -> object:
    raise VisualEvidenceError(
        f"CLI JSON contains a non-finite constant: {value}"
    )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VisualEvidenceError(
                f"CLI JSON contains a duplicate key: {key}"
            )
        result[key] = value
    return result


def _parse_cli_json(label: str, stdout: str) -> dict[str, object]:
    normalized = validate_stdout(label, stdout, ())
    if normalized.count("\n") != 1:
        raise VisualEvidenceError(
            f"{label} stdout must be exactly one newline-terminated JSON line"
        )
    try:
        value = json.loads(
            normalized,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except VisualEvidenceError:
        raise
    except json.JSONDecodeError as error:
        raise VisualEvidenceError(
            f"{label} stdout is not strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise VisualEvidenceError(f"{label} JSON must be an object")
    return value


def _json_exact(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(
            _json_exact(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(
            _json_exact(actual_item, expected_item)
            for actual_item, expected_item in zip(
                actual, expected, strict=True
            )
        )
    return actual == expected


def _require_exact_keys(
    label: str, value: object, expected: set[str]
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise VisualEvidenceError(f"{label} fields differ from the contract")
    return value


def _validate_cli_plan(
    label: str, plan: object, include_dump: bool
) -> dict[str, object]:
    expected_keys = {"stats", "kernels"}
    if include_dump:
        expected_keys.add("canonical_dump")
    record = _require_exact_keys(label, plan, expected_keys)
    if not _json_exact(record["stats"], CLI_PLAN_STATS):
        raise VisualEvidenceError(f"{label} statistics differ")
    if not _json_exact(record["kernels"], list(CLI_KERNELS)):
        raise VisualEvidenceError(f"{label} kernel sequence differs")
    kernels = record["kernels"]
    assert isinstance(kernels, list)
    if sum(kernel["scalar_steps"] for kernel in kernels) != 20:
        raise VisualEvidenceError(
            f"{label} kernel work does not sum to scalar_steps"
        )

    if include_dump:
        canonical_dump = record["canonical_dump"]
        if not isinstance(canonical_dump, str):
            raise VisualEvidenceError(
                "inspect canonical dump must be a string"
            )
        if canonical_dump != CLI_CANONICAL_DUMP:
            raise VisualEvidenceError(
                "inspect canonical dump differs from the exact workload plan"
            )
        dump_sentinels = (
            "tensorkiln.execution_plan v0 {",
            (
                "  stats {values=6, inputs=1, constants=2, steps=3, "
                "outputs=1, constant_bytes=32, scalar_steps=20, "
                "workspace_bytes=128}"
            ),
            "    @0 #n2 %2 = matmul_rank2_f32(%0,%1) work=12",
            "    @1 #n4 %4 = add_broadcast_f32(%2,%3) work=4",
            "    @2 #n5 %5 = relu_contiguous_f32(%4) work=4",
            "    #b0 offset=0 payload=16 reserved=64 live=[0,2)",
            "    #b1 offset=64 payload=16 reserved=64 live=[1,3)",
            "    #b2 offset=0 payload=16 reserved=64 live=[2,3)",
        )
        validate_stdout(
            "inspect canonical dump", canonical_dump, dump_sentinels
        )
    return record


def validate_cli_evidence(
    inspect_stdout: str, execute_stdout: str
) -> CliEvidence:
    """Cross-check exact release CLI output before rendering or publication."""

    inspect = _parse_cli_json("CLI inspect", inspect_stdout)
    execute = _parse_cli_json("CLI execute", execute_stdout)
    _require_exact_keys(
        "CLI inspect",
        inspect,
        {"schema", "workload", "plan"},
    )
    _require_exact_keys(
        "CLI execute",
        execute,
        {"schema", "workload", "plan", "execution"},
    )
    if inspect["schema"] != "tensorkiln.cli.inspect.v1":
        raise VisualEvidenceError("CLI inspect schema differs")
    if execute["schema"] != "tensorkiln.cli.execute.v1":
        raise VisualEvidenceError("CLI execute schema differs")
    if not _json_exact(inspect["workload"], CLI_WORKLOAD):
        raise VisualEvidenceError("CLI inspect workload differs")
    if not _json_exact(execute["workload"], inspect["workload"]):
        raise VisualEvidenceError(
            "CLI execute workload differs from inspect"
        )

    inspect_plan = _validate_cli_plan(
        "CLI inspect plan", inspect["plan"], include_dump=True
    )
    execute_plan = _validate_cli_plan(
        "CLI execute plan", execute["plan"], include_dump=False
    )
    if not _json_exact(execute_plan["stats"], inspect_plan["stats"]) or not (
        _json_exact(execute_plan["kernels"], inspect_plan["kernels"])
    ):
        raise VisualEvidenceError(
            "CLI execute plan differs from inspect"
        )

    execution = _require_exact_keys(
        "CLI execution",
        execute["execution"],
        {
            "run_status",
            "kernel_write_audit",
            "logical_workspace_bytes",
            "input",
            "outputs",
            "reference_check",
            "verification_scope",
            "benchmark",
        },
    )
    expected_input = {
        "name": "x",
        "dtype": "f32",
        "shape": [2, 3],
        "bits": list(CLI_INPUT_BITS),
    }
    expected_outputs = [
        {
            "name": "result",
            "dtype": "f32",
            "shape": [2, 2],
            "bits": list(CLI_OUTPUT_BITS),
        }
    ]
    expected_reference = {
        "comparison": "raw_f32_bits",
        "matched": 4,
        "total": 4,
        "status": "match",
    }
    if execution["run_status"] != "success":
        raise VisualEvidenceError("CLI execution did not report success")
    if execution["kernel_write_audit"] is not True:
        raise VisualEvidenceError("CLI kernel-write audit is not enabled")
    if (
        type(execution["logical_workspace_bytes"]) is not int
        or execution["logical_workspace_bytes"] != 128
    ):
        raise VisualEvidenceError("CLI execution workspace differs")
    if not _json_exact(execution["input"], expected_input):
        raise VisualEvidenceError("CLI execution input differs")
    if not _json_exact(execution["outputs"], expected_outputs):
        raise VisualEvidenceError("CLI execution output differs")
    if not _json_exact(
        execution["reference_check"], expected_reference
    ):
        raise VisualEvidenceError("CLI reference agreement differs")
    if (
        execution["verification_scope"]
        != "this_workload_and_input_bits"
        or execution["benchmark"] is not False
    ):
        raise VisualEvidenceError("CLI claim boundary differs")

    for label, bits, expected_count in (
        ("input", CLI_INPUT_BITS, 6),
        ("output", CLI_OUTPUT_BITS, 4),
    ):
        if len(bits) != expected_count or any(
            RAW_F32_BITS_PATTERN.fullmatch(item) is None for item in bits
        ):
            raise VisualEvidenceError(
                f"CLI {label} bits are not canonical raw f32 values"
            )
    input_shape = expected_input["shape"]
    output_shape = expected_outputs[0]["shape"]
    if (
        input_shape[0] * input_shape[1] != len(CLI_INPUT_BITS)
        or output_shape[0] * output_shape[1] != len(CLI_OUTPUT_BITS)
    ):
        raise VisualEvidenceError(
            "CLI tensor shapes differ from their raw-bit payloads"
        )

    return CliEvidence(
        inspect=inspect,
        execute=execute,
        input_bits=CLI_INPUT_BITS,
        output_bits=CLI_OUTPUT_BITS,
    )


def _parse_reglu_canonical_plan(
    canonical_dump: str,
) -> tuple[tuple[RegluPlanStep, ...], tuple[RegluArenaBuffer, ...]]:
    """Parse and structurally verify the exact plan used by the diagrams."""

    if canonical_dump != REGLU_CANONICAL_DUMP:
        raise VisualEvidenceError(
            "ReGLU canonical dump differs from the exact workload plan"
        )
    validate_stdout(
        "ReGLU canonical dump",
        canonical_dump,
        (
            "tensorkiln.execution_plan v0 {",
            "    @5 #n10 %10 = mul_contiguous_f32(%5,%9) work=8",
            "    #b5 offset=64 payload=32 reserved=64 live=[5,6)",
        ),
    )
    steps = tuple(
        RegluPlanStep(
            step=int(match.group("step")),
            source_node=int(match.group("node")),
            result_value=int(match.group("result")),
            kernel=match.group("kernel"),
            operands=tuple(
                int(item[1:])
                for item in match.group("operands").split(",")
            ),
            scalar_steps=int(match.group("work")),
        )
        for line in canonical_dump.splitlines()
        if (match := REGLU_STEP_PATTERN.fullmatch(line)) is not None
    )
    arena = tuple(
        RegluArenaBuffer(
            buffer=int(match.group("buffer")),
            offset=int(match.group("offset")),
            payload=int(match.group("payload")),
            reserved=int(match.group("reserved")),
            live_start=int(match.group("start")),
            live_end=int(match.group("end")),
        )
        for line in canonical_dump.splitlines()
        if (match := REGLU_ARENA_PATTERN.fullmatch(line)) is not None
    )
    expected_steps = (
        (0, 2, 2, "matmul_rank2_f32", (0, 1), 24),
        (1, 4, 4, "add_broadcast_f32", (2, 3), 8),
        (2, 5, 5, "relu_contiguous_f32", (4,), 8),
        (3, 7, 7, "matmul_rank2_f32", (0, 6), 24),
        (4, 9, 9, "add_broadcast_f32", (7, 8), 8),
        (5, 10, 10, "mul_contiguous_f32", (5, 9), 8),
    )
    if tuple(
        (
            step.step,
            step.source_node,
            step.result_value,
            step.kernel,
            step.operands,
            step.scalar_steps,
        )
        for step in steps
    ) != expected_steps:
        raise VisualEvidenceError("ReGLU plan dependency graph differs")
    expected_arena = (
        (0, 0, 32, 64, 0, 2),
        (1, 64, 32, 64, 1, 3),
        (2, 0, 32, 64, 2, 6),
        (3, 64, 32, 64, 3, 5),
        (4, 128, 32, 64, 4, 6),
        (5, 64, 32, 64, 5, 6),
    )
    if tuple(
        (
            item.buffer,
            item.offset,
            item.payload,
            item.reserved,
            item.live_start,
            item.live_end,
        )
        for item in arena
    ) != expected_arena:
        raise VisualEvidenceError("ReGLU arena intervals differ")

    producers = {step.result_value: step.step for step in steps}
    for step in steps:
        for operand in step.operands:
            if operand in producers and producers[operand] >= step.step:
                raise VisualEvidenceError(
                    "ReGLU plan contains a non-topological dependency"
                )
    if sum(step.scalar_steps for step in steps) != 80:
        raise VisualEvidenceError("ReGLU plan work does not sum to 80")
    if max(item.offset + item.reserved for item in arena) != 192:
        raise VisualEvidenceError("ReGLU arena does not bind 192 bytes")
    return steps, arena


def _validate_reglu_plan(
    label: str, plan: object, include_dump: bool
) -> tuple[
    dict[str, object],
    tuple[RegluPlanStep, ...] | None,
    tuple[RegluArenaBuffer, ...] | None,
]:
    expected_keys = {"stats", "kernels"}
    if include_dump:
        expected_keys.add("canonical_dump")
    record = _require_exact_keys(label, plan, expected_keys)
    if not _json_exact(record["stats"], REGLU_PLAN_STATS):
        raise VisualEvidenceError(f"{label} statistics differ")
    if not _json_exact(record["kernels"], list(REGLU_KERNELS)):
        raise VisualEvidenceError(f"{label} kernel sequence differs")
    if include_dump:
        canonical_dump = record["canonical_dump"]
        if not isinstance(canonical_dump, str):
            raise VisualEvidenceError(
                "ReGLU inspect canonical dump must be a string"
            )
        steps, arena = _parse_reglu_canonical_plan(canonical_dump)
        return record, steps, arena
    return record, None, None


def _reglu_list_text() -> str:
    return (
        "TensorKiln compiled-in workloads\n"
        f"  {CLI_WORKLOAD['id']}  {CLI_WORKLOAD['description']}\n"
        f"  {REGLU_WORKLOAD['id']}  {REGLU_WORKLOAD['description']}\n"
        "scope: bounded examples; no graph or model-file import\n"
    )


def _reglu_inspect_text() -> str:
    kernels = " -> ".join(
        str(kernel["kind"]) for kernel in REGLU_KERNELS
    )
    return (
        "TensorKiln plan inspection\n"
        "schema: tensorkiln.cli.inspect.v1\n"
        "workload: reglu_mlp_v1\n"
        "scope: compiled-in workload; not a model-file import\n"
        "input: x f32[2,3] elements=6\n"
        "output: result f32[2,4] elements=8\n"
        "plan: values=11 steps=6 scalar_steps=80 workspace_bytes=192\n"
        f"kernels: {kernels}\n\n"
        f"{REGLU_CANONICAL_DUMP}"
    )


def _reglu_execute_text() -> str:
    kernels = " -> ".join(
        str(kernel["kind"]) for kernel in REGLU_KERNELS
    )
    return (
        "TensorKiln audited execution\n"
        "schema: tensorkiln.cli.execute.v1\n"
        "workload: reglu_mlp_v1\n"
        "scope: this compiled-in workload and these raw input bits\n"
        "input: x f32[2,3] bits="
        f"{','.join(REGLU_INPUT_BITS)}\n"
        "output: result f32[2,4] bits="
        f"{','.join(REGLU_OUTPUT_BITS)}\n"
        "plan: values=11 steps=6 scalar_steps=80 workspace_bytes=192\n"
        f"kernels: {kernels}\n"
        "kernel_write_audit: on\n"
        "reference_check: raw_f32_bits match 8/8\n"
        "benchmark: false (no timing measurements)\n"
    )


def validate_reglu_evidence(
    workloads_stdout: str,
    inspect_stdout: str,
    execute_stdout: str,
    list_text_stdout: str,
    inspect_text_stdout: str,
    execute_text_stdout: str,
) -> RegluEvidence:
    """Validate exact replayed ReGLU JSON and terminal-mode transcripts."""

    workloads = _parse_cli_json("ReGLU CLI list", workloads_stdout)
    inspect = _parse_cli_json("ReGLU CLI inspect", inspect_stdout)
    execute = _parse_cli_json("ReGLU CLI execute", execute_stdout)
    _require_exact_keys("ReGLU CLI list", workloads, {"schema", "workloads"})
    if workloads["schema"] != "tensorkiln.cli.workloads.v1" or not (
        _json_exact(workloads["workloads"], [CLI_WORKLOAD, REGLU_WORKLOAD])
    ):
        raise VisualEvidenceError("ReGLU workload registry differs")
    _require_exact_keys(
        "ReGLU CLI inspect", inspect, {"schema", "workload", "plan"}
    )
    _require_exact_keys(
        "ReGLU CLI execute",
        execute,
        {"schema", "workload", "plan", "execution"},
    )
    if inspect["schema"] != "tensorkiln.cli.inspect.v1":
        raise VisualEvidenceError("ReGLU inspect schema differs")
    if execute["schema"] != "tensorkiln.cli.execute.v1":
        raise VisualEvidenceError("ReGLU execute schema differs")
    if not _json_exact(inspect["workload"], REGLU_WORKLOAD):
        raise VisualEvidenceError("ReGLU inspect workload differs")
    if not _json_exact(execute["workload"], inspect["workload"]):
        raise VisualEvidenceError("ReGLU execute workload differs from inspect")

    inspect_plan, steps, arena = _validate_reglu_plan(
        "ReGLU inspect plan", inspect["plan"], include_dump=True
    )
    execute_plan, _, _ = _validate_reglu_plan(
        "ReGLU execute plan", execute["plan"], include_dump=False
    )
    if not _json_exact(execute_plan["stats"], inspect_plan["stats"]) or not (
        _json_exact(execute_plan["kernels"], inspect_plan["kernels"])
    ):
        raise VisualEvidenceError("ReGLU execute plan differs from inspect")
    assert steps is not None
    assert arena is not None

    execution = _require_exact_keys(
        "ReGLU execution",
        execute["execution"],
        {
            "run_status",
            "kernel_write_audit",
            "logical_workspace_bytes",
            "input",
            "outputs",
            "reference_check",
            "verification_scope",
            "benchmark",
        },
    )
    expected_input = {
        "name": "x",
        "dtype": "f32",
        "shape": [2, 3],
        "bits": list(REGLU_INPUT_BITS),
    }
    expected_outputs = [
        {
            "name": "result",
            "dtype": "f32",
            "shape": [2, 4],
            "bits": list(REGLU_OUTPUT_BITS),
        }
    ]
    expected_reference = {
        "comparison": "raw_f32_bits",
        "matched": 8,
        "total": 8,
        "status": "match",
    }
    if execution["run_status"] != "success":
        raise VisualEvidenceError("ReGLU execution did not report success")
    if execution["kernel_write_audit"] is not True:
        raise VisualEvidenceError("ReGLU kernel-write audit is not enabled")
    if (
        type(execution["logical_workspace_bytes"]) is not int
        or execution["logical_workspace_bytes"] != 192
    ):
        raise VisualEvidenceError("ReGLU execution workspace differs")
    if not _json_exact(execution["input"], expected_input):
        raise VisualEvidenceError("ReGLU execution input differs")
    if not _json_exact(execution["outputs"], expected_outputs):
        raise VisualEvidenceError("ReGLU execution output differs")
    if not _json_exact(execution["reference_check"], expected_reference):
        raise VisualEvidenceError("ReGLU reference agreement differs")
    if (
        execution["verification_scope"] != "this_workload_and_input_bits"
        or execution["benchmark"] is not False
    ):
        raise VisualEvidenceError("ReGLU claim boundary differs")
    for label, bits, expected_count in (
        ("input", REGLU_INPUT_BITS, 6),
        ("output", REGLU_OUTPUT_BITS, 8),
    ):
        if len(bits) != expected_count or any(
            RAW_F32_BITS_PATTERN.fullmatch(item) is None for item in bits
        ):
            raise VisualEvidenceError(
                f"ReGLU {label} bits are not canonical raw f32 values"
            )

    expected_text = (
        ("list", list_text_stdout, _reglu_list_text()),
        ("inspect", inspect_text_stdout, _reglu_inspect_text()),
        ("execute", execute_text_stdout, _reglu_execute_text()),
    )
    for label, actual, expected in expected_text:
        normalized = validate_stdout(f"ReGLU CLI {label} text", actual, ())
        if normalized != expected:
            raise VisualEvidenceError(
                f"ReGLU CLI {label} text differs from the JSON-bound contract"
            )

    return RegluEvidence(
        workloads=workloads,
        inspect=inspect,
        execute=execute,
        list_text=list_text_stdout,
        inspect_text=inspect_text_stdout,
        execute_text=execute_text_stdout,
        steps=steps,
        arena=arena,
        input_bits=REGLU_INPUT_BITS,
        output_bits=REGLU_OUTPUT_BITS,
    )


def _one_match(
    label: str, pattern: re.Pattern[str], lines: list[str]
) -> re.Match[str]:
    matches = [match for line in lines if (match := pattern.fullmatch(line))]
    if len(matches) != 1:
        raise VisualEvidenceError(
            f"plan_arena stdout must contain exactly one {label} record"
        )
    return matches[0]


def parse_arena_evidence(stdout: str) -> ArenaEvidence:
    """Parse and cross-check allocation evidence from ``plan_arena``."""

    normalized = validate_stdout("plan_arena", stdout, PLAN_SENTINELS)
    lines = normalized.splitlines()
    alignment = int(_one_match("alignment", ALIGNMENT_PATTERN, lines).group(1))
    stats_match = _one_match("statistics", STATS_PATTERN, lines)
    stats = tuple(int(value) for value in stats_match.groups())
    buffer_count, total_payload, total_reserved, peak_live, workspace = stats
    naive = int(_one_match("naive reservation", NAIVE_PATTERN, lines).group(1))
    reused = int(_one_match("reused workspace", REUSED_PATTERN, lines).group(1))
    summary_match = _one_match(
        "verification summary", VERIFIED_REUSE_PATTERN, lines
    )
    summary_count_text = summary_match.group("count")
    summary_reuses = COUNT_WORDS.get(
        summary_count_text, int(summary_count_text)
        if summary_count_text.isdigit()
        else -1
    )
    summary_workspace = int(summary_match.group("workspace"))
    summary_reserved = int(summary_match.group("reserved"))

    allocations = tuple(
        ArenaAllocation(
            ordinal=int(match.group("ordinal")),
            offset=int(match.group("offset")),
            payload=int(match.group("payload")),
            reserved=int(match.group("reserved")),
            live_start=int(match.group("start")),
            live_end=int(match.group("end")),
        )
        for line in lines
        if (match := ALLOCATION_PATTERN.fullmatch(line))
    )

    if alignment <= 0:
        raise VisualEvidenceError("arena alignment must be positive")
    if buffer_count <= 0 or not allocations:
        raise VisualEvidenceError(
            "verified arena example must contain at least one allocation"
        )
    if len(allocations) != buffer_count:
        raise VisualEvidenceError(
            "parsed allocation count differs from reported buffer count"
        )
    if [item.ordinal for item in allocations] != list(range(buffer_count)):
        raise VisualEvidenceError(
            "arena allocation ordinals must be contiguous and canonical"
        )
    if sum(item.payload for item in allocations) != total_payload:
        raise VisualEvidenceError(
            "parsed payload bytes differ from reported total"
        )
    if sum(item.reserved for item in allocations) != total_reserved:
        raise VisualEvidenceError(
            "parsed reserved bytes differ from reported total"
        )
    if naive != total_reserved or reused != workspace:
        raise VisualEvidenceError(
            "summary byte counts differ from verified arena statistics"
        )
    if workspace <= 0:
        raise VisualEvidenceError("arena workspace must be positive")

    for item in allocations:
        if item.payload <= 0 or item.reserved < item.payload:
            raise VisualEvidenceError(
                f"#b{item.ordinal} has invalid payload/reservation bytes"
            )
        if item.offset % alignment != 0:
            raise VisualEvidenceError(
                f"#b{item.ordinal} has a non-aligned offset"
            )
        expected_reserved = (
            (item.payload + alignment - 1) // alignment
        ) * alignment
        if item.reserved != expected_reserved:
            raise VisualEvidenceError(
                f"#b{item.ordinal} reservation is not the exact "
                "alignment-rounded payload"
            )
        if item.live_start >= item.live_end:
            raise VisualEvidenceError(
                f"#b{item.ordinal} has an empty or reversed lifetime"
            )
        if item.offset + item.reserved > workspace:
            raise VisualEvidenceError(
                f"#b{item.ordinal} exceeds the reported workspace"
            )

    for index, left in enumerate(allocations):
        for right in allocations[index + 1 :]:
            bytes_overlap = (
                left.offset < right.offset + right.reserved
                and right.offset < left.offset + left.reserved
            )
            lifetimes_overlap = (
                left.live_start < right.live_end
                and right.live_start < left.live_end
            )
            if bytes_overlap and lifetimes_overlap:
                raise VisualEvidenceError(
                    f"#b{left.ordinal} and #b{right.ordinal} overlap while live"
                )

    computed_workspace = max(
        item.offset + item.reserved for item in allocations
    )
    if computed_workspace != workspace:
        raise VisualEvidenceError(
            "allocation extents differ from the reported workspace"
        )

    lifetime_events: dict[int, int] = {}
    for item in allocations:
        lifetime_events[item.live_start] = (
            lifetime_events.get(item.live_start, 0) + item.reserved
        )
        lifetime_events[item.live_end] = (
            lifetime_events.get(item.live_end, 0) - item.reserved
        )
    live_reserved = 0
    computed_peak = 0
    for step in sorted(lifetime_events):
        live_reserved += lifetime_events[step]
        computed_peak = max(computed_peak, live_reserved)
    if live_reserved != 0:
        raise VisualEvidenceError("arena lifetime accounting did not close")
    if computed_peak != peak_live:
        raise VisualEvidenceError(
            "allocation lifetimes differ from the reported peak live bytes"
        )

    allocations_by_slot: dict[
        tuple[int, int], list[ArenaAllocation]
    ] = {}
    for item in allocations:
        allocations_by_slot.setdefault(
            (item.offset, item.reserved), []
        ).append(item)
    boundary_reuses = 0
    for slot_allocations in allocations_by_slot.values():
        ordered = sorted(
            slot_allocations,
            key=lambda item: (item.live_start, item.live_end, item.ordinal),
        )
        boundary_reuses += sum(
            left.live_end == right.live_start
            for left, right in zip(ordered, ordered[1:], strict=False)
        )
    if (
        summary_reuses != boundary_reuses
        or summary_workspace != workspace
        or summary_reserved != total_reserved
    ):
        raise VisualEvidenceError(
            "verification summary differs from parsed arena evidence"
        )

    return ArenaEvidence(
        alignment=alignment,
        buffer_count=buffer_count,
        total_payload=total_payload,
        total_reserved=total_reserved,
        peak_live_reserved=peak_live,
        workspace=workspace,
        allocations=allocations,
    )


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _stdout_digest(stdout: str) -> str:
    return hashlib.sha256(stdout.encode("utf-8")).hexdigest()[:16]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _artifact_bytes(artifact: Artifact) -> bytes:
    return artifact.encode("utf-8") if isinstance(artifact, str) else artifact


def _artifact_media_type(filename: str) -> str:
    suffix = Path(filename).suffix
    media_types = {
        ".gif": "image/gif",
        ".json": "application/json",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".txt": "text/plain; charset=utf-8",
    }
    try:
        return media_types[suffix]
    except KeyError as error:
        raise VisualEvidenceError(
            f"visual artifact has unsupported media type: {filename}"
        ) from error


def _artifact_manifest_record(
    filename: str, artifact: Artifact
) -> dict[str, object]:
    payload = _artifact_bytes(artifact)
    return {
        "bytes": len(payload),
        "media_type": _artifact_media_type(filename),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def render_execute_graph_svg(
    stdout: str, command: str = "<release-build>/execute_graph"
) -> str:
    """Render the complete, verified ``execute_graph`` transcript."""

    normalized = validate_stdout(
        "execute_graph", stdout, EXECUTE_SENTINELS
    )
    reject_unsafe_text("execute_graph display command", command)
    transcript_lines = normalized.expandtabs(4).splitlines()
    display_lines = [f"$ {command}", *transcript_lines]
    longest_line = max(len(line) for line in display_lines)
    width = max(960, min(1600, 76 + longest_line * 8))
    line_height = 20
    panel_top = 76
    first_line_y = panel_top + 58
    height = first_line_y + line_height * len(display_lines) + 54
    digest = _stdout_digest(normalized)

    text_nodes: list[str] = []
    for index, line in enumerate(display_lines):
        y = first_line_y + index * line_height
        color = "#c9d1d9"
        weight = "400"
        if index == 0:
            color = "#79c0ff"
            weight = "600"
        elif line.startswith("==="):
            color = "#d2a8ff"
            weight = "600"
        elif line.startswith("result ="):
            color = "#f2cc60"
            weight = "600"
        elif line.startswith("verified:"):
            color = "#7ee787"
            weight = "600"
        text_nodes.append(
            f'    <text x="48" y="{y}" fill="{color}" '
            f'font-weight="{weight}">{_escape(line)}</text>'
        )

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title description">'
        ),
        (
            '  <title id="title">TensorKiln verified execute_graph '
            "example output</title>"
        ),
        (
            '  <desc id="description">Complete stdout captured from the '
            "verified release example; this is an example, not a benchmark."
            "</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="32" y="38" fill="#f0f6fc" font-family="DejaVu Sans, '
            'system-ui, sans-serif" font-size="19" font-weight="700">'
            "Verified dense "
            "execution example</text>"
        ),
        (
            '  <text x="32" y="61" fill="#8b949e" font-family="DejaVu Sans, '
            'system-ui, sans-serif" font-size="12">RELEASE EXAMPLE OUTPUT · '
            "NOT A "
            "BENCHMARK</text>"
        ),
        (
            f'  <rect x="24" y="{panel_top}" width="{width - 48}" '
            f'height="{height - panel_top - 32}" rx="10" fill="#161b22" '
            'stroke="#30363d"/>'
        ),
        (
            '  <g font-family="DejaVu Sans Mono, ui-monospace, '
            'SFMono-Regular, Consolas, monospace" font-size="13" '
            'xml:space="preserve">'
        ),
        *text_nodes,
        "  </g>",
        (
            f'  <text x="{width - 32}" y="{height - 12}" '
            'text-anchor="end" fill="#6e7681" font-family="DejaVu Sans Mono, '
            f'ui-monospace, monospace" font-size="10">stdout sha256:{digest}'
            "</text>"
        ),
        "</svg>",
        "",
    ]
    rendered = "\n".join(svg)
    reject_unsafe_text("execute_graph SVG", rendered)
    return rendered


def render_execute_softmax_svg(
    stdout: str, command: str = "<release-build>/execute_softmax"
) -> str:
    """Render the complete, exact ``execute_softmax`` fixture transcript."""

    normalized = validate_softmax_stdout(stdout)
    reject_unsafe_text("execute_softmax display command", command)
    transcript_lines = normalized.expandtabs(4).splitlines()
    display_lines = [f"$ {command}", *transcript_lines]
    longest_line = max(len(line) for line in display_lines)
    width = max(1080, min(1600, 76 + longest_line * 8))
    line_height = 22
    panel_top = 82
    first_line_y = panel_top + 58
    height = first_line_y + line_height * len(display_lines) + 58
    digest = _stdout_digest(normalized)

    text_nodes: list[str] = []
    for index, line in enumerate(display_lines):
        y = first_line_y + index * line_height
        color = "#c9d1d9"
        weight = "400"
        if index == 0:
            color = "#79c0ff"
            weight = "600"
        elif line.startswith("==="):
            color = "#d2a8ff"
            weight = "600"
        elif line.startswith("scope:"):
            color = "#8b949e"
        elif line.startswith("plan {"):
            color = "#79c0ff"
            weight = "600"
        elif line.startswith("slice "):
            color = "#f2cc60"
        elif line.startswith("agreement {") or line.startswith("verified:"):
            color = "#7ee787"
            weight = "600"
        elif line.startswith("reference_axis0"):
            color = "#a5d6ff"
        elif line.startswith("optimized_axis0"):
            color = "#ff7b72"
        text_nodes.append(
            f'    <text x="48" y="{y}" fill="{color}" '
            f'font-weight="{weight}">{_escape(line)}</text>'
        )

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-labelledby="title description">'
        ),
        (
            '  <title id="title">TensorKiln crafted Softmax correctness '
            "evidence</title>"
        ),
        (
            '  <desc id="description">Complete stdout captured from the '
            "verified release example: five crafted policy slices, 20 of 20 "
            "bit agreements, and the reference-only axis-zero boundary. This "
            "is not a benchmark or an arbitrary-libm bit-exactness claim.</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="32" y="40" fill="#f0f6fc" font-family="DejaVu Sans, '
            'system-ui, sans-serif" font-size="21" font-weight="700">'
            "Crafted Softmax correctness evidence</text>"
        ),
        (
            '  <text x="32" y="66" fill="#8b949e" font-family="DejaVu Sans, '
            'system-ui, sans-serif" font-size="12">COMPLETE RELEASE STDOUT · '
            "FIVE EXACT SLICES · NOT A BENCHMARK</text>"
        ),
        (
            f'  <rect x="24" y="{panel_top}" width="{width - 48}" '
            f'height="{height - panel_top - 34}" rx="10" fill="#161b22" '
            'stroke="#30363d"/>'
        ),
        (
            '  <g font-family="DejaVu Sans Mono, ui-monospace, '
            'SFMono-Regular, Consolas, monospace" font-size="13" '
            'xml:space="preserve">'
        ),
        *text_nodes,
        "  </g>",
        (
            f'  <text x="32" y="{height - 14}" fill="#6e7681" '
            'font-family="DejaVu Sans, system-ui, sans-serif" font-size="10">'
            "60 optimized kernel steps · 80 axis-zero reference total · "
            "fixture-scoped bits</text>"
        ),
        (
            f'  <text x="{width - 32}" y="{height - 14}" '
            'text-anchor="end" fill="#6e7681" font-family="DejaVu Sans Mono, '
            f'ui-monospace, monospace" font-size="10">stdout sha256:{digest}'
            "</text>"
        ),
        "</svg>",
        "",
    ]
    rendered = "\n".join(svg)
    reject_unsafe_text("execute_softmax SVG", rendered)
    return rendered


def render_cli_execution_svg(
    inspect_stdout: str, execute_stdout: str
) -> str:
    """Render the audited CLI workflow from two exact JSON reports."""

    evidence = validate_cli_evidence(inspect_stdout, execute_stdout)
    inspect_plan = evidence.inspect["plan"]
    execution = evidence.execute["execution"]
    assert isinstance(inspect_plan, dict)
    assert isinstance(execution, dict)
    stats = inspect_plan["stats"]
    kernels = inspect_plan["kernels"]
    reference = execution["reference_check"]
    assert isinstance(stats, dict)
    assert isinstance(kernels, list)
    assert isinstance(reference, dict)

    width = 1200
    height = 720
    input_rows = (
        "  ".join(evidence.input_bits[:3]),
        "  ".join(evidence.input_bits[3:]),
    )
    output_bits = "  ".join(evidence.output_bits)
    inspect_digest = _stdout_digest(inspect_stdout)
    execute_digest = _stdout_digest(execute_stdout)
    kernel_labels = [
        (
            str(kernel["kind"]),
            int(kernel["scalar_steps"]),
        )
        for kernel in kernels
    ]

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        (
            '  <title id="title">TensorKiln audited CLI execution '
            "evidence</title>"
        ),
        (
            '  <desc id="description">Release CLI inspect and execute JSON '
            "captured twice, cross-checked against each other, and rendered "
            "as a fixture-scoped workflow. Kernel-write auditing is enabled "
            "and executor output matches the independent reference for four "
            "of four raw f32 values. This is not a benchmark.</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="36" y="44" fill="#f0f6fc" '
            'font-family="DejaVu Sans, system-ui, sans-serif" '
            'font-size="23" font-weight="700">'
            "Audited CLI execution · dense_relu_v1</text>"
        ),
        (
            '  <text x="36" y="70" fill="#8b949e" '
            'font-family="DejaVu Sans, system-ui, sans-serif" '
            'font-size="12">REAL RELEASE JSON · BYTE-IDENTICAL REPLAY ×2 · '
            "FIXTURE-SCOPED · NOT A BENCHMARK</text>"
        ),
        (
            '  <rect x="36" y="94" width="1128" height="66" rx="10" '
            'fill="#161b22" stroke="#30363d"/>'
        ),
        (
            '  <text x="56" y="119" fill="#8b949e" '
            'font-family="DejaVu Sans, system-ui, sans-serif" '
            'font-size="11">SETUP / REPRODUCE</text>'
        ),
        (
            '  <text x="56" y="143" fill="#79c0ff" '
            'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
            'font-size="14">make -j2 PROFILE=release cli</text>'
        ),
        (
            '  <text x="445" y="143" fill="#c9d1d9" '
            'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
            'font-size="13">inspect --format=json  →  execute '
            "--input-bits … --format=json</text>"
        ),
        (
            '  <rect x="36" y="184" width="300" height="172" rx="12" '
            'fill="#161b22" stroke="#30363d"/>'
        ),
        (
            '  <text x="56" y="214" fill="#f0f6fc" '
            'font-family="DejaVu Sans, system-ui, sans-serif" '
            'font-size="16" font-weight="700">1 · Bound input</text>'
        ),
        (
            '  <text x="56" y="240" fill="#79c0ff" '
            'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
            'font-size="13">x : f32[2,3] · 6 values</text>'
        ),
        (
            f'  <text x="56" y="276" fill="#c9d1d9" '
            'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
            f'font-size="11">{_escape(input_rows[0])}</text>'
        ),
        (
            f'  <text x="56" y="299" fill="#c9d1d9" '
            'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
            f'font-size="11">{_escape(input_rows[1])}</text>'
        ),
        (
            '  <text x="56" y="333" fill="#8b949e" '
            'font-family="DejaVu Sans, system-ui, sans-serif" '
            'font-size="11">Canonical lowercase IEEE-754 raw bits</text>'
        ),
        (
            '  <rect x="360" y="184" width="804" height="172" rx="12" '
            'fill="#161b22" stroke="#30363d"/>'
        ),
        (
            '  <text x="380" y="214" fill="#f0f6fc" '
            'font-family="DejaVu Sans, system-ui, sans-serif" '
            'font-size="16" font-weight="700">2 · Independently inspected '
            "execution plan</text>"
        ),
    ]

    colors = ("#58a6ff", "#d2a8ff", "#f2cc60")
    for index, ((kind, scalar_steps), color) in enumerate(
        zip(kernel_labels, colors, strict=True)
    ):
        x = 380 + index * 250
        svg.extend(
            [
                (
                    f'  <rect x="{x}" y="238" width="214" height="82" '
                    f'rx="10" fill="{color}" opacity="0.18" '
                    f'stroke="{color}"/>'
                ),
                (
                    f'  <text x="{x + 16}" y="268" fill="{color}" '
                    'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
                    f'font-size="12" font-weight="700">{_escape(kind)}</text>'
                ),
                (
                    f'  <text x="{x + 16}" y="296" fill="#c9d1d9" '
                    'font-family="DejaVu Sans, system-ui, sans-serif" '
                    f'font-size="12">{scalar_steps} scalar steps</text>'
                ),
            ]
        )
        if index < len(kernel_labels) - 1:
            svg.append(
                (
                    f'  <text x="{x + 228}" y="286" fill="#8b949e" '
                    'font-family="DejaVu Sans, system-ui, sans-serif" '
                    'font-size="22">→</text>'
                )
            )

    svg.extend(
        [
            (
                '  <text x="380" y="340" fill="#8b949e" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                f'font-size="11">{stats["steps"]} kernels · '
                f'{stats["scalar_steps"]} scalar steps · '
                f'{stats["workspace_bytes"]} B verified workspace</text>'
            ),
            (
                '  <rect x="36" y="380" width="548" height="198" rx="12" '
                'fill="#161b22" stroke="#30363d"/>'
            ),
            (
                '  <text x="56" y="412" fill="#f0f6fc" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="16" font-weight="700">3 · Audited executor</text>'
            ),
            (
                '  <circle cx="74" cy="448" r="10" fill="#3fb950"/>'
            ),
            (
                '  <text x="96" y="453" fill="#7ee787" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="14" font-weight="700">run_status = success</text>'
            ),
            (
                '  <circle cx="74" cy="483" r="10" fill="#3fb950"/>'
            ),
            (
                '  <text x="96" y="488" fill="#7ee787" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="14" font-weight="700">kernel_write_audit = '
                "ON</text>"
            ),
            (
                '  <text x="56" y="529" fill="#c9d1d9" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                f'font-size="13">Logical workspace: '
                f'{execution["logical_workspace_bytes"]} B</text>'
            ),
            (
                '  <text x="56" y="554" fill="#8b949e" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="11">Writes outside each kernel result payload '
                "fail closed.</text>"
            ),
            (
                '  <rect x="608" y="380" width="556" height="198" rx="12" '
                'fill="#161b22" stroke="#30363d"/>'
            ),
            (
                '  <text x="628" y="412" fill="#f0f6fc" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="16" font-weight="700">4 · Independent reference '
                "gate</text>"
            ),
            (
                '  <text x="628" y="452" fill="#7ee787" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="26" font-weight="700">4 / 4 RAW BITS MATCH</text>'
            ),
            (
                '  <text x="628" y="482" fill="#8b949e" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                f'font-size="12">comparison = '
                f'{_escape(str(reference["comparison"]))}</text>'
            ),
            (
                '  <text x="628" y="515" fill="#c9d1d9" '
                'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
                f'font-size="12">{_escape(output_bits)}</text>'
            ),
            (
                '  <text x="628" y="546" fill="#f2cc60" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="12">result : f32[2,2] · exact published '
                "output</text>"
            ),
            (
                '  <rect x="36" y="602" width="1128" height="70" rx="10" '
                'fill="#0f1720" stroke="#30363d"/>'
            ),
            (
                '  <text x="56" y="630" fill="#8b949e" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="11">EVIDENCE BOUNDARY</text>'
            ),
            (
                '  <text x="56" y="653" fill="#c9d1d9" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="12">This proves one compiled-in workload and '
                "one input-bit fixture; it is not a model importer or a "
                "performance measurement.</text>"
            ),
            (
                f'  <text x="36" y="702" fill="#6e7681" '
                'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
                f'font-size="10">inspect sha256:{inspect_digest}</text>'
            ),
            (
                f'  <text x="1164" y="702" text-anchor="end" fill="#6e7681" '
                'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
                f'font-size="10">execute sha256:{execute_digest}</text>'
            ),
            "</svg>",
            "",
        ]
    )
    rendered = "\n".join(svg)
    reject_unsafe_text("CLI execution SVG", rendered)
    return rendered


def render_reglu_graph_svg(evidence: RegluEvidence) -> str:
    """Render the two canonical ReGLU branches from parsed plan dependencies."""

    width = 1400
    height = 680
    positions = {
        0: (250, 142),
        1: (500, 142),
        2: (750, 142),
        3: (250, 398),
        4: (500, 398),
        5: (1030, 270),
    }
    producers = {step.result_value: step.step for step in evidence.steps}
    if set(positions) != {step.step for step in evidence.steps}:
        raise VisualEvidenceError("ReGLU graph layout differs from parsed steps")

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '  <title id="title">TensorKiln compiled ReGLU branch graph</title>',
        (
            '  <desc id="description">Dependency graph parsed from the exact '
            "reglu_mlp_v1 canonical execution plan: two MatMul and Add "
            "branches, a ReLU gate, and an elementwise Mul merge.</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="36" y="48" fill="#f0f6fc" '
            'font-family="system-ui, sans-serif" font-size="24" '
            'font-weight="700">Compiled ReGLU-style MLP · reglu_mlp_v1</text>'
        ),
        (
            '  <text x="36" y="76" fill="#8b949e" '
            'font-family="system-ui, sans-serif" font-size="12">'
            "SOURCE-DERIVED CANONICAL PLAN · FIXED F32[2,3] FIXTURE · "
            "NOT A FULL TRANSFORMER</text>"
        ),
        (
            '  <rect x="36" y="258" width="156" height="96" rx="12" '
            'fill="#13233a" stroke="#58a6ff"/>'
        ),
        (
            '  <text x="114" y="294" text-anchor="middle" fill="#79c0ff" '
            'font-family="ui-monospace, monospace" font-size="16" '
            'font-weight="700">%0 · x</text>'
        ),
        (
            '  <text x="114" y="322" text-anchor="middle" fill="#c9d1d9" '
            'font-family="ui-monospace, monospace" font-size="13">f32[2,3]</text>'
        ),
        (
            '  <text x="114" y="342" text-anchor="middle" fill="#8b949e" '
            'font-family="system-ui, sans-serif" font-size="11">one bound input</text>'
        ),
    ]

    node_width = 206
    node_height = 116
    for step in evidence.steps:
        x, y = positions[step.step]
        color = "#d2a8ff" if step.step in {2, 5} else "#58a6ff"
        short_kind = step.kernel.removesuffix("_f32")
        svg.extend(
            [
                (
                    f'  <rect x="{x}" y="{y}" width="{node_width}" '
                    f'height="{node_height}" rx="12" fill="{color}" '
                    f'fill-opacity="0.13" stroke="{color}"/>'
                ),
                (
                    f'  <text x="{x + 16}" y="{y + 30}" fill="{color}" '
                    'font-family="ui-monospace, monospace" font-size="13" '
                    f'font-weight="700">@{step.step} · #n{step.source_node}</text>'
                ),
                (
                    f'  <text x="{x + 16}" y="{y + 58}" fill="#f0f6fc" '
                    'font-family="ui-monospace, monospace" font-size="13" '
                    f'font-weight="700">{_escape(short_kind)}</text>'
                ),
                (
                    f'  <text x="{x + 16}" y="{y + 82}" fill="#c9d1d9" '
                    'font-family="ui-monospace, monospace" font-size="11">'
                    f'{_escape(",".join(f"%{item}" for item in step.operands))} '
                    f'→ %{step.result_value}</text>'
                ),
                (
                    f'  <text x="{x + 16}" y="{y + 103}" fill="#8b949e" '
                    'font-family="system-ui, sans-serif" font-size="11">'
                    f'{step.scalar_steps} scalar steps</text>'
                ),
            ]
        )

    edges: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for step in evidence.steps:
        target_x, target_y = positions[step.step]
        for operand in step.operands:
            if operand == 0:
                edges.append(((192, 306), (target_x, target_y + 58), "%0"))
            elif operand in producers:
                producer = producers[operand]
                source_x, source_y = positions[producer]
                edges.append(
                    (
                        (source_x + node_width, source_y + 58),
                        (target_x, target_y + 58),
                        f"%{operand}",
                    )
                )
    edge_nodes: list[str] = []
    for (x1, y1), (x2, y2), value in edges:
        midpoint = (x1 + x2) // 2
        path = f"M{x1},{y1} C{midpoint},{y1} {midpoint},{y2} {x2},{y2}"
        edge_nodes.extend(
            [
                f'  <path d="{path}" fill="none" stroke="#6e7681" '
                'stroke-width="2"/>',
                (
                    f'  <circle cx="{x2 - 5}" cy="{y2}" r="4" '
                    'fill="#7ee787"/>'
                ),
                (
                    f'  <text x="{midpoint}" y="{(y1 + y2) // 2 - 7}" '
                    'text-anchor="middle" fill="#8b949e" '
                    'font-family="ui-monospace, monospace" font-size="10">'
                    f'{value}</text>'
                ),
            ]
        )
    svg[7:7] = edge_nodes
    svg.extend(
        [
            (
                '  <rect x="1270" y="280" width="94" height="96" rx="12" '
                'fill="#123321" stroke="#3fb950"/>'
            ),
            (
                '  <text x="1317" y="315" text-anchor="middle" fill="#7ee787" '
                'font-family="ui-monospace, monospace" font-size="13" '
                'font-weight="700">result</text>'
            ),
            (
                '  <text x="1317" y="342" text-anchor="middle" fill="#c9d1d9" '
                'font-family="ui-monospace, monospace" font-size="12">f32[2,4]</text>'
            ),
            (
                '  <path d="M1236,328 L1270,328" fill="none" '
                'stroke="#3fb950" stroke-width="2"/>'
            ),
            (
                '  <rect x="36" y="572" width="1328" height="72" rx="10" '
                'fill="#161b22" stroke="#30363d"/>'
            ),
            (
                '  <text x="56" y="602" fill="#c9d1d9" '
                'font-family="system-ui, sans-serif" font-size="13">'
                "6 verified kernels · 80 scalar steps · 4 constants / 128 B · "
                "192 B logical workspace</text>"
            ),
            (
                '  <text x="56" y="626" fill="#8b949e" '
                'font-family="system-ui, sans-serif" font-size="11">'
                "Edges, value IDs, kernel names, and work counts are parsed "
                "from the release inspect canonical dump.</text>"
            ),
            "</svg>",
            "",
        ]
    )
    rendered = "\n".join(svg)
    reject_unsafe_text("ReGLU graph SVG", rendered)
    return rendered


def render_reglu_arena_svg(evidence: RegluEvidence) -> str:
    """Render the exact six-buffer, three-slot ReGLU arena schedule."""

    width = 1320
    height = 600
    axis_x = 230
    step_width = 160
    slot_y = {0: 178, 64: 302, 128: 426}
    colors = ("#58a6ff", "#d2a8ff", "#f2cc60", "#7ee787")
    if {item.offset for item in evidence.arena} != set(slot_y):
        raise VisualEvidenceError("ReGLU arena offsets escaped the three slots")
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '  <title id="title">TensorKiln ReGLU arena lifetime schedule</title>',
        (
            '  <desc id="description">Six arena buffers parsed from the '
            "canonical plan reuse three 64-byte aligned slots across six "
            "kernel boundaries for a 192-byte workspace.</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="36" y="48" fill="#f0f6fc" '
            'font-family="system-ui, sans-serif" font-size="24" '
            'font-weight="700">Arena lifetimes · 192 B verified workspace</text>'
        ),
        (
            '  <text x="36" y="76" fill="#8b949e" '
            'font-family="system-ui, sans-serif" font-size="12">'
            "EXACT OFFSETS + HALF-OPEN LIVE INTERVALS FROM INSPECT JSON</text>"
        ),
    ]
    for boundary in range(7):
        x = axis_x + boundary * step_width
        svg.extend(
            [
                (
                    f'  <line x1="{x}" y1="118" x2="{x}" y2="500" '
                    'stroke="#30363d" stroke-width="1"/>'
                ),
                (
                    f'  <text x="{x}" y="108" text-anchor="middle" '
                    'fill="#8b949e" font-family="ui-monospace, monospace" '
                    f'font-size="11">@{boundary}</text>'
                ),
            ]
        )
    for offset, y in slot_y.items():
        svg.extend(
            [
                (
                    f'  <text x="36" y="{y + 27}" fill="#c9d1d9" '
                    'font-family="ui-monospace, monospace" font-size="13" '
                    f'font-weight="700">offset {offset:3d}</text>'
                ),
                (
                    f'  <rect x="{axis_x}" y="{y}" width="{6 * step_width}" '
                    'height="54" rx="7" fill="#161b22" stroke="#30363d"/>'
                ),
            ]
        )
    for item in evidence.arena:
        y = slot_y[item.offset] + 19
        x = axis_x + item.live_start * step_width + 3
        rect_width = (item.live_end - item.live_start) * step_width - 6
        color = colors[item.buffer % len(colors)]
        svg.extend(
            [
                (
                    f'  <rect x="{x}" y="{y}" width="{rect_width}" '
                    f'height="16" rx="4" fill="{color}" fill-opacity="0.35" '
                    f'stroke="{color}"/>'
                ),
                (
                    f'  <text x="{x + 8}" y="{y + 12}" fill="#f0f6fc" '
                    'font-family="ui-monospace, monospace" font-size="10">'
                    f'#b{item.buffer} · 32/64 B · '
                    f'[{item.live_start},{item.live_end})</text>'
                ),
            ]
        )
    svg.extend(
        [
            (
                '  <text x="36" y="554" fill="#7ee787" '
                'font-family="system-ui, sans-serif" font-size="13" '
                'font-weight="700">Boundary reuse: #b0→#b2, '
                "#b1→#b3→#b5</text>"
            ),
            (
                '  <text x="1284" y="554" text-anchor="end" fill="#8b949e" '
                'font-family="system-ui, sans-serif" font-size="11">'
                "6 × 32 B payloads · 3 × 64 B aligned slots</text>"
            ),
            "</svg>",
            "",
        ]
    )
    rendered = "\n".join(svg)
    reject_unsafe_text("ReGLU arena SVG", rendered)
    return rendered


def _format_raw_f32(bits: str) -> str:
    value = struct.unpack(">f", bytes.fromhex(bits[2:]))[0]
    if bits == "0x00000000":
        return "+0"
    if bits == "0x80000000":
        return "-0"
    return format(value, ".9g")


def render_reglu_output_svg(evidence: RegluEvidence) -> str:
    """Render the complete 2x4 output raw-bit matrix and reference gate."""

    width = 1240
    height = 540
    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '  <title id="title">TensorKiln ReGLU raw f32 output matrix</title>',
        (
            '  <desc id="description">All eight executor output words for '
            "the fixed reglu_mlp_v1 input, including positive and negative "
            "zero, match the independent reference bit for bit.</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="36" y="48" fill="#f0f6fc" '
            'font-family="system-ui, sans-serif" font-size="24" '
            'font-weight="700">Exact ReGLU output · result f32[2,4]</text>'
        ),
        (
            '  <text x="36" y="76" fill="#8b949e" '
            'font-family="system-ui, sans-serif" font-size="12">'
            "ACTUAL RELEASE EXECUTOR BITS · FIXED INPUT · NOT A BENCHMARK</text>"
        ),
    ]
    cell_width = 250
    cell_height = 128
    for index, bits in enumerate(evidence.output_bits):
        row, column = divmod(index, 4)
        x = 72 + column * (cell_width + 20)
        y = 116 + row * (cell_height + 20)
        special_zero = bits in {"0x00000000", "0x80000000"}
        color = "#f2cc60" if special_zero else "#58a6ff"
        svg.extend(
            [
                (
                    f'  <rect x="{x}" y="{y}" width="{cell_width}" '
                    f'height="{cell_height}" rx="12" fill="{color}" '
                    f'fill-opacity="0.12" stroke="{color}"/>'
                ),
                (
                    f'  <text x="{x + 18}" y="{y + 30}" fill="#8b949e" '
                    'font-family="system-ui, sans-serif" font-size="11">'
                    f'[{row},{column}] · raw IEEE-754</text>'
                ),
                (
                    f'  <text x="{x + 18}" y="{y + 67}" fill="{color}" '
                    'font-family="ui-monospace, monospace" font-size="19" '
                    f'font-weight="700">{bits}</text>'
                ),
                (
                    f'  <text x="{x + 18}" y="{y + 101}" fill="#c9d1d9" '
                    'font-family="ui-monospace, monospace" font-size="15">'
                    f'f32 = {_escape(_format_raw_f32(bits))}</text>'
                ),
            ]
        )
    svg.extend(
        [
            (
                '  <rect x="72" y="426" width="1060" height="72" rx="10" '
                'fill="#123321" stroke="#3fb950"/>'
            ),
            (
                '  <text x="96" y="458" fill="#7ee787" '
                'font-family="system-ui, sans-serif" font-size="20" '
                'font-weight="700">8 / 8 RAW F32 WORDS MATCH</text>'
            ),
            (
                '  <text x="96" y="483" fill="#c9d1d9" '
                'font-family="system-ui, sans-serif" font-size="12">'
                "Independent reference comparison · kernel-write audit ON · "
                "signed -0 preserved exactly</text>"
            ),
            "</svg>",
            "",
        ]
    )
    rendered = "\n".join(svg)
    reject_unsafe_text("ReGLU output SVG", rendered)
    return rendered


# Purpose-built 5x7 ASCII glyphs keep terminal rasterization independent of
# host fonts, fontconfig, locale, and optional imaging libraries.  The table is
# source code (not a copied font asset) and covers every character admitted by
# the fixed text-mode evidence contract.
_RASTER_FONT_SPEC: Final = {
    " ": "00000/00000/00000/00000/00000/00000/00000",
    "A": "01110/10001/10001/11111/10001/10001/10001",
    "B": "11110/10001/10001/11110/10001/10001/11110",
    "C": "01111/10000/10000/10000/10000/10000/01111",
    "D": "11110/10001/10001/10001/10001/10001/11110",
    "E": "11111/10000/10000/11110/10000/10000/11111",
    "F": "11111/10000/10000/11110/10000/10000/10000",
    "G": "01111/10000/10000/10111/10001/10001/01111",
    "H": "10001/10001/10001/11111/10001/10001/10001",
    "I": "01110/00100/00100/00100/00100/00100/01110",
    "J": "00111/00010/00010/00010/00010/10010/01100",
    "K": "10001/10010/10100/11000/10100/10010/10001",
    "L": "10000/10000/10000/10000/10000/10000/11111",
    "M": "10001/11011/10101/10101/10001/10001/10001",
    "N": "10001/11001/10101/10011/10001/10001/10001",
    "O": "01110/10001/10001/10001/10001/10001/01110",
    "P": "11110/10001/10001/11110/10000/10000/10000",
    "Q": "01110/10001/10001/10001/10101/10010/01101",
    "R": "11110/10001/10001/11110/10100/10010/10001",
    "S": "01111/10000/10000/01110/00001/00001/11110",
    "T": "11111/00100/00100/00100/00100/00100/00100",
    "U": "10001/10001/10001/10001/10001/10001/01110",
    "V": "10001/10001/10001/10001/10001/01010/00100",
    "W": "10001/10001/10001/10101/10101/10101/01010",
    "X": "10001/10001/01010/00100/01010/10001/10001",
    "Y": "10001/10001/01010/00100/00100/00100/00100",
    "Z": "11111/00001/00010/00100/01000/10000/11111",
    "a": "00000/00000/01110/00001/01111/10001/01111",
    "b": "10000/10000/10110/11001/10001/10001/11110",
    "c": "00000/00000/01110/10000/10000/10001/01110",
    "d": "00001/00001/01101/10011/10001/10001/01111",
    "e": "00000/00000/01110/10001/11111/10000/01110",
    "f": "00110/01001/01000/11100/01000/01000/01000",
    "g": "00000/01111/10001/01111/00001/10001/01110",
    "h": "10000/10000/10110/11001/10001/10001/10001",
    "i": "00100/00000/01100/00100/00100/00100/01110",
    "j": "00010/00000/00110/00010/00010/10010/01100",
    "k": "10000/10000/10010/10100/11000/10100/10010",
    "l": "01100/00100/00100/00100/00100/00100/01110",
    "m": "00000/00000/11010/10101/10101/10101/10101",
    "n": "00000/00000/10110/11001/10001/10001/10001",
    "o": "00000/00000/01110/10001/10001/10001/01110",
    "p": "00000/11110/10001/11110/10000/10000/10000",
    "q": "00000/01111/10001/01111/00001/00001/00001",
    "r": "00000/00000/10110/11001/10000/10000/10000",
    "s": "00000/00000/01111/10000/01110/00001/11110",
    "t": "01000/01000/11100/01000/01000/01001/00110",
    "u": "00000/00000/10001/10001/10001/10011/01101",
    "v": "00000/00000/10001/10001/10001/01010/00100",
    "w": "00000/00000/10001/10001/10101/10101/01010",
    "x": "00000/00000/10001/01010/00100/01010/10001",
    "y": "00000/10001/10001/01111/00001/10001/01110",
    "z": "00000/00000/11111/00010/00100/01000/11111",
    "0": "01110/10001/10011/10101/11001/10001/01110",
    "1": "00100/01100/00100/00100/00100/00100/01110",
    "2": "01110/10001/00001/00010/00100/01000/11111",
    "3": "11110/00001/00001/01110/00001/00001/11110",
    "4": "00010/00110/01010/10010/11111/00010/00010",
    "5": "11111/10000/10000/11110/00001/00001/11110",
    "6": "01110/10000/10000/11110/10001/10001/01110",
    "7": "11111/00001/00010/00100/01000/01000/01000",
    "8": "01110/10001/10001/01110/10001/10001/01110",
    "9": "01110/10001/10001/01111/00001/00001/01110",
    "$": "00100/01111/10100/01110/00101/11110/00100",
    "<": "00010/00100/01000/10000/01000/00100/00010",
    ">": "01000/00100/00010/00001/00010/00100/01000",
    "/": "00001/00010/00100/01000/10000/00000/00000",
    "\\": "10000/01000/00100/00010/00001/00000/00000",
    "[": "01110/01000/01000/01000/01000/01000/01110",
    "]": "01110/00010/00010/00010/00010/00010/01110",
    "(": "00010/00100/01000/01000/01000/00100/00010",
    ")": "01000/00100/00010/00010/00010/00100/01000",
    "{": "00010/00100/00100/01000/00100/00100/00010",
    "}": "01000/00100/00100/00010/00100/00100/01000",
    ",": "00000/00000/00000/00000/00110/00100/01000",
    ".": "00000/00000/00000/00000/00000/01100/01100",
    ":": "00000/01100/01100/00000/01100/01100/00000",
    ";": "00000/01100/01100/00000/01100/00100/01000",
    "_": "00000/00000/00000/00000/00000/00000/11111",
    "-": "00000/00000/00000/11111/00000/00000/00000",
    "=": "00000/00000/11111/00000/11111/00000/00000",
    "+": "00000/00100/00100/11111/00100/00100/00000",
    "#": "01010/01010/11111/01010/11111/01010/01010",
    "%": "11001/11010/00100/01000/10110/00110/00000",
    "*": "00000/10101/01110/11111/01110/10101/00000",
    "|": "00100/00100/00100/00100/00100/00100/00100",
    "?": "01110/10001/00001/00010/00100/00000/00100",
    "!": "00100/00100/00100/00100/00100/00000/00100",
    "@": "01110/10001/10111/10101/10111/10000/01110",
    "'": "00100/00100/00000/00000/00000/00000/00000",
    '"': "01010/01010/00000/00000/00000/00000/00000",
    "&": "01100/10010/10100/01000/10101/10010/01101",
}
_RASTER_FONT: Final = {
    character: tuple(rows.split("/"))
    for character, rows in _RASTER_FONT_SPEC.items()
}
_TERMINAL_PALETTE: Final = (
    (13, 17, 23),
    (22, 27, 34),
    (48, 54, 61),
    (201, 209, 217),
    (121, 192, 255),
    (210, 168, 255),
    (126, 231, 135),
    (242, 204, 96),
)
TERMINAL_WIDTH: Final = 960
TERMINAL_HEIGHT: Final = 540
TERMINAL_GIF_DELAYS_CS: Final = (90, 120, 170)


def _fill_indexed_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    color: int,
) -> None:
    if (
        x < 0
        or y < 0
        or rect_width < 0
        or rect_height < 0
        or x + rect_width > width
        or y + rect_height > height
        or not 0 <= color < len(_TERMINAL_PALETTE)
    ):
        raise VisualEvidenceError("terminal raster rectangle is out of bounds")
    row = bytes([color]) * rect_width
    for target_y in range(y, y + rect_height):
        start = target_y * width + x
        pixels[start : start + rect_width] = row


def _draw_raster_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: int,
    scale: int = 2,
) -> None:
    if scale not in {1, 2, 3}:
        raise VisualEvidenceError("terminal raster uses an unsupported scale")
    cursor_x = x
    for character in text:
        rows = _RASTER_FONT.get(character)
        if rows is None:
            raise VisualEvidenceError(
                f"terminal raster has no source glyph for {character!r}"
            )
        for row_index, row in enumerate(rows):
            for column_index, bit in enumerate(row):
                if bit == "1":
                    _fill_indexed_rect(
                        pixels,
                        width,
                        height,
                        cursor_x + column_index * scale,
                        y + row_index * scale,
                        scale,
                        scale,
                        color,
                    )
        cursor_x += 6 * scale


def _wrap_terminal_lines(
    command: str, stdout: str, columns: int
) -> list[str]:
    reject_unsafe_text("terminal command", command)
    normalized = validate_stdout("terminal capture", stdout, ())
    wrapped: list[str] = []
    for line in [f"$ {command}", *normalized.splitlines()]:
        if not line:
            wrapped.append("")
            continue
        while line:
            wrapped.append(line[:columns])
            line = line[columns:]
    return wrapped


def _render_terminal_frame(
    title: str,
    command: str,
    stdout: str,
    *,
    tail: bool,
) -> bytearray:
    reject_unsafe_text("terminal frame title", title)
    width = TERMINAL_WIDTH
    height = TERMINAL_HEIGHT
    pixels = bytearray([0]) * (width * height)
    _fill_indexed_rect(pixels, width, height, 18, 18, 924, 504, 1)
    _fill_indexed_rect(pixels, width, height, 18, 18, 924, 2, 2)
    _fill_indexed_rect(pixels, width, height, 18, 520, 924, 2, 2)
    _fill_indexed_rect(pixels, width, height, 18, 18, 2, 504, 2)
    _fill_indexed_rect(pixels, width, height, 940, 18, 2, 504, 2)
    _fill_indexed_rect(pixels, width, height, 20, 20, 920, 48, 0)
    for index, color in enumerate((7, 5, 6)):
        _fill_indexed_rect(
            pixels, width, height, 34 + index * 22, 36, 10, 10, color
        )
    _draw_raster_text(pixels, width, height, 112, 33, title, 3, scale=2)

    columns = 73
    lines = _wrap_terminal_lines(command, stdout, columns)
    maximum_rows = 24
    if len(lines) > maximum_rows:
        lines = lines[-maximum_rows:] if tail else lines[:maximum_rows]
    for row, line in enumerate(lines):
        color = 3
        if line.startswith("$"):
            color = 4
        elif line.startswith("TensorKiln"):
            color = 5
        elif line.startswith("reference_check:"):
            color = 6
        elif line.startswith("output:") or "mul_contiguous_f32" in line:
            color = 7
        _draw_raster_text(
            pixels,
            width,
            height,
            36,
            82 + row * 18,
            line,
            color,
            scale=2,
        )
    digest = hashlib.sha256(stdout.encode("ascii")).hexdigest()[:16]
    _draw_raster_text(
        pixels,
        width,
        height,
        716,
        499,
        f"sha256:{digest}",
        2,
        scale=1,
    )
    return pixels


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _adler32(payload: bytes) -> int:
    """Compute the RFC 1950 checksum without a compression-library dependency."""

    first = 1
    second = 0
    modulus = 65521
    for byte in payload:
        first = (first + byte) % modulus
        second = (second + first) % modulus
    return (second << 16) | first


def _stored_zlib(payload: bytes) -> bytes:
    """Return one canonical zlib stream made only of stored DEFLATE blocks."""

    stream = bytearray(b"\x78\x01")
    if not payload:
        stream.extend(b"\x01\x00\x00\xff\xff")
    else:
        for start in range(0, len(payload), 65535):
            block = payload[start : start + 65535]
            final = start + len(block) == len(payload)
            stream.append(1 if final else 0)
            stream.extend(struct.pack("<H", len(block)))
            stream.extend(struct.pack("<H", len(block) ^ 0xFFFF))
            stream.extend(block)
    stream.extend(struct.pack(">I", _adler32(payload)))
    return bytes(stream)


def _indexed_png(
    pixels: bytearray, width: int, height: int
) -> bytes:
    if len(pixels) != width * height:
        raise VisualEvidenceError("terminal PNG pixel count differs")
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for color_index in pixels[y * width : (y + 1) * width]:
            try:
                rows.extend(_TERMINAL_PALETTE[color_index])
            except IndexError as error:
                raise VisualEvidenceError(
                    "terminal PNG contains an invalid palette index"
                ) from error
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"".join(
        (
            signature,
            _png_chunk(b"IHDR", ihdr),
            _png_chunk(b"IDAT", _stored_zlib(bytes(rows))),
            _png_chunk(b"IEND", b""),
        )
    )


def _gif_literal_lzw(pixels: bytearray) -> bytes:
    """Encode literals with bounded clear groups and a fixed four-bit width."""

    clear_code = 8
    end_code = 9
    codes: list[int] = []
    for start in range(0, len(pixels), 6):
        codes.append(clear_code)
        chunk = pixels[start : start + 6]
        if any(value >= 8 for value in chunk):
            raise VisualEvidenceError("terminal GIF palette index differs")
        codes.extend(chunk)
    codes.append(end_code)
    packed = bytearray()
    accumulator = 0
    bit_count = 0
    for code in codes:
        accumulator |= code << bit_count
        bit_count += 4
        while bit_count >= 8:
            packed.append(accumulator & 0xFF)
            accumulator >>= 8
            bit_count -= 8
    if bit_count:
        packed.append(accumulator & 0xFF)
    return bytes(packed)


def _gif_subblocks(payload: bytes) -> bytes:
    blocks = bytearray()
    for start in range(0, len(payload), 255):
        block = payload[start : start + 255]
        blocks.append(len(block))
        blocks.extend(block)
    blocks.append(0)
    return bytes(blocks)


def _indexed_gif(
    frames: Sequence[bytearray], delays_cs: Sequence[int]
) -> bytes:
    if len(frames) != len(delays_cs) or not frames:
        raise VisualEvidenceError("terminal GIF frame contract differs")
    width = TERMINAL_WIDTH
    height = TERMINAL_HEIGHT
    payload = bytearray(b"GIF89a")
    payload.extend(struct.pack("<HH", width, height))
    payload.extend(bytes((0xF2, 0, 0)))
    for red, green, blue in _TERMINAL_PALETTE:
        payload.extend((red, green, blue))
    for frame, delay in zip(frames, delays_cs, strict=True):
        if len(frame) != width * height or not 1 <= delay <= 1000:
            raise VisualEvidenceError("terminal GIF frame bounds differ")
        payload.extend(b"!\xf9\x04")
        payload.extend(bytes((0,)))
        payload.extend(struct.pack("<H", delay))
        payload.extend(bytes((0, 0)))
        payload.extend(b",")
        payload.extend(struct.pack("<HHHH", 0, 0, width, height))
        payload.extend(bytes((0, 3)))
        payload.extend(_gif_subblocks(_gif_literal_lzw(frame)))
    payload.extend(b";")
    return bytes(payload)


def _display_cli_command(arguments: Sequence[str]) -> str:
    command = "<release-build>/tensorkiln"
    if arguments:
        command += " " + " ".join(arguments)
    reject_unsafe_text("display CLI command", command)
    return command


def render_reglu_demo_transcript(evidence: RegluEvidence) -> str:
    """Return complete real list, inspect, and execute terminal captures."""

    captures = (
        (REGLU_LIST_TEXT_ARGUMENTS, evidence.list_text),
        (REGLU_INSPECT_TEXT_ARGUMENTS, evidence.inspect_text),
        (REGLU_EXECUTE_TEXT_ARGUMENTS, evidence.execute_text),
    )
    sections = [
        f"$ {_display_cli_command(arguments)}\n{stdout}"
        for arguments, stdout in captures
    ]
    transcript = "\n".join(sections)
    reject_unsafe_text("ReGLU demo transcript", transcript)
    return transcript


def render_reglu_terminal_png(evidence: RegluEvidence) -> bytes:
    """Rasterize the complete execute view with only source-embedded glyphs."""

    frame = _render_terminal_frame(
        "TensorKiln ReGLU release CLI",
        _display_cli_command(REGLU_EXECUTE_TEXT_ARGUMENTS),
        evidence.execute_text,
        tail=False,
    )
    return _indexed_png(frame, TERMINAL_WIDTH, TERMINAL_HEIGHT)


def render_reglu_demo_gif(evidence: RegluEvidence) -> bytes:
    """Animate three actual CLI text captures with deterministic delays."""

    frame_inputs = (
        (
            "1 / 3  Workload registry",
            REGLU_LIST_TEXT_ARGUMENTS,
            evidence.list_text,
            False,
        ),
        (
            "2 / 3  Canonical plan tail",
            REGLU_INSPECT_TEXT_ARGUMENTS,
            evidence.inspect_text,
            True,
        ),
        (
            "3 / 3  Audited execution",
            REGLU_EXECUTE_TEXT_ARGUMENTS,
            evidence.execute_text,
            False,
        ),
    )
    frames = [
        _render_terminal_frame(
            title,
            _display_cli_command(arguments),
            stdout,
            tail=tail,
        )
        for title, arguments, stdout, tail in frame_inputs
    ]
    return _indexed_gif(frames, TERMINAL_GIF_DELAYS_CS)


def _slot_groups(
    evidence: ArenaEvidence,
) -> tuple[tuple[tuple[int, int], tuple[ArenaAllocation, ...]], ...]:
    grouped: dict[tuple[int, int], list[ArenaAllocation]] = {}
    for allocation in evidence.allocations:
        grouped.setdefault(
            (allocation.offset, allocation.reserved), []
        ).append(allocation)
    return tuple(
        (
            slot,
            tuple(sorted(items, key=lambda item: (item.live_start, item.ordinal))),
        )
        for slot, items in sorted(grouped.items())
    )


def render_arena_reuse_svg(stdout: str) -> str:
    """Render physical-slot reuse directly from ``plan_arena`` stdout."""

    evidence = parse_arena_evidence(stdout)
    normalized = stdout.replace("\r\n", "\n")
    groups = _slot_groups(evidence)
    max_step = max(item.live_end for item in evidence.allocations)
    if max_step <= 0:
        raise VisualEvidenceError("arena lifetime axis is empty")

    width = 1200
    chart_left = 205
    chart_right = 1148
    chart_width = chart_right - chart_left
    chart_top = 190
    row_height = 66
    chart_height = row_height * len(groups)
    height = chart_top + chart_height + 88
    palette = ("#58a6ff", "#d2a8ff", "#f2cc60", "#7ee787", "#ff7b72")

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        (
            '  <title id="title">TensorKiln verified interval arena '
            "reuse example</title>"
        ),
        (
            '  <desc id="description">Physical workspace slots and '
            "half-open lifetimes parsed from verified plan_arena output; "
            "this is an example, not a benchmark.</desc>"
        ),
        '  <rect width="100%" height="100%" rx="18" fill="#0d1117"/>',
        (
            '  <text x="32" y="40" fill="#f0f6fc" font-family="DejaVu Sans, '
            'system-ui, sans-serif" font-size="21" font-weight="700">'
            "Verified interval "
            "arena reuse</text>"
        ),
        (
            '  <text x="32" y="65" fill="#8b949e" font-family="DejaVu Sans, '
            'system-ui, sans-serif" font-size="12">PARSED FROM plan_arena '
            "STDOUT · "
            "VERIFIED EXAMPLE · NOT A BENCHMARK</text>"
        ),
    ]

    metrics = (
        ("Buffers", str(evidence.buffer_count)),
        ("Aligned reservations", f"{evidence.total_reserved} B"),
        ("Reused workspace", f"{evidence.workspace} B"),
        ("Alignment", f"{evidence.alignment} B"),
    )
    for index, (label, value) in enumerate(metrics):
        x = 32 + index * 286
        svg.extend(
            [
                (
                    f'  <rect x="{x}" y="88" width="264" height="70" '
                    'rx="10" fill="#161b22" stroke="#30363d"/>'
                ),
                (
                    f'  <text x="{x + 16}" y="113" fill="#8b949e" '
                    'font-family="DejaVu Sans, system-ui, sans-serif" '
                    'font-size="12">'
                    f"{_escape(label)}</text>"
                ),
                (
                    f'  <text x="{x + 16}" y="143" fill="#f0f6fc" '
                    'font-family="DejaVu Sans, system-ui, sans-serif" '
                    'font-size="22" '
                    f'font-weight="700">{_escape(value)}</text>'
                ),
            ]
        )

    for step in range(max_step + 1):
        x = chart_left + round(chart_width * step / max_step)
        svg.extend(
            [
                (
                    f'  <line x1="{x}" y1="{chart_top - 10}" x2="{x}" '
                    f'y2="{chart_top + chart_height}" stroke="#30363d" '
                    'stroke-width="1"/>'
                ),
                (
                    f'  <text x="{x}" y="{chart_top - 18}" text-anchor="middle" '
                    'fill="#8b949e" font-family="DejaVu Sans Mono, '
                    'ui-monospace, monospace" '
                    f'font-size="12">step {step}</text>'
                ),
            ]
        )

    for row, ((offset, reserved), allocations) in enumerate(groups):
        y = chart_top + row * row_height
        svg.extend(
            [
                (
                    f'  <text x="32" y="{y + 22}" fill="#f0f6fc" '
                    'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
                    'font-size="13" '
                    f'font-weight="600">bytes {offset}–{offset + reserved}'
                    "</text>"
                ),
                (
                    f'  <text x="32" y="{y + 43}" fill="#8b949e" '
                    'font-family="DejaVu Sans, system-ui, sans-serif" '
                    'font-size="11">'
                    f"{reserved} B physical slot</text>"
                ),
                (
                    f'  <line x1="{chart_left}" y1="{y + 28}" '
                    f'x2="{chart_right}" y2="{y + 28}" stroke="#21262d" '
                    'stroke-width="38" stroke-linecap="round"/>'
                ),
            ]
        )
        for allocation in allocations:
            x = chart_left + round(
                chart_width * allocation.live_start / max_step
            )
            end_x = chart_left + round(
                chart_width * allocation.live_end / max_step
            )
            color = palette[allocation.ordinal % len(palette)]
            svg.extend(
                [
                    (
                        f'  <rect x="{x}" y="{y + 10}" '
                        f'width="{end_x - x}" height="36" rx="7" '
                        f'fill="{color}" opacity="0.88"/>'
                    ),
                    (
                        f'  <text x="{x + 10}" y="{y + 33}" fill="#0d1117" '
                        'font-family="DejaVu Sans Mono, ui-monospace, '
                        'monospace" font-size="12" '
                        f'font-weight="700">#b{allocation.ordinal} '
                        f"[{allocation.live_start},{allocation.live_end}) · "
                        f"{allocation.reserved} B</text>"
                    ),
                ]
            )

    digest = _stdout_digest(normalized)
    footer_y = chart_top + chart_height + 52
    svg.extend(
        [
            (
                f'  <text x="32" y="{footer_y}" fill="#8b949e" '
                'font-family="DejaVu Sans, system-ui, sans-serif" '
                'font-size="12">'
                "Adjacent rectangles sharing a physical slot show legal "
                "half-open lifetime boundary reuse.</text>"
            ),
            (
                f'  <text x="{width - 32}" y="{height - 18}" '
                'text-anchor="end" fill="#6e7681" '
                'font-family="DejaVu Sans Mono, ui-monospace, monospace" '
                'font-size="10">'
                f"stdout sha256:{digest}</text>"
            ),
            "</svg>",
            "",
        ]
    )
    rendered = "\n".join(svg)
    reject_unsafe_text("arena reuse SVG", rendered)
    return rendered


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Kill and reap one isolated evidence process and its descendants."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=PROCESS_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as final_error:
            raise VisualEvidenceError(
                "could not reap a bounded evidence process"
            ) from final_error


def _wait_for_process_exit_without_reaping(
    process: subprocess.Popen[bytes],
    *,
    arguments: Sequence[str],
    deadline: float,
    timeout_seconds: float,
    stdout: bytearray,
    stderr: bytearray,
) -> None:
    """Observe leader exit while retaining its PID and process-group identity."""

    wait_options = os.WEXITED | os.WNOHANG | os.WNOWAIT
    while True:
        try:
            status = os.waitid(os.P_PID, process.pid, wait_options)
        except ChildProcessError as error:
            raise VisualEvidenceError(
                "bounded evidence process was reaped unexpectedly"
            ) from error
        if status is not None:
            return

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise subprocess.TimeoutExpired(
                arguments,
                timeout_seconds,
                output=bytes(stdout),
                stderr=bytes(stderr),
            )
        time.sleep(min(PROCESS_EXIT_POLL_SECONDS, remaining_seconds))


def _run_bounded_process(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    timeout_seconds: float,
    maximum_output_bytes: int = MAX_OUTPUT_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    """Capture both child streams concurrently within byte and time bounds."""

    if not command:
        raise VisualEvidenceError("bounded evidence command is empty")
    if maximum_output_bytes < 0 or timeout_seconds <= 0:
        raise VisualEvidenceError("bounded evidence process limits are invalid")

    arguments = list(command)
    process = subprocess.Popen(
        arguments,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        close_fds=True,
        start_new_session=True,
    )
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    stdout = bytearray()
    stderr = bytearray()
    buffers = {"stdout": stdout, "stderr": stderr}
    selector: selectors.BaseSelector | None = None
    deadline = time.monotonic() + timeout_seconds

    try:
        if stdout_stream is None or stderr_stream is None:
            raise VisualEvidenceError(
                "bounded evidence process did not expose both output streams"
            )
        streams = {"stdout": stdout_stream, "stderr": stderr_stream}
        selector = selectors.DefaultSelector()
        for stream_name, stream in streams.items():
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            selector.register(
                descriptor,
                selectors.EVENT_READ,
                data=stream_name,
            )

        while selector.get_map():
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise subprocess.TimeoutExpired(
                    arguments,
                    timeout_seconds,
                    output=bytes(stdout),
                    stderr=bytes(stderr),
                )
            events = selector.select(remaining_seconds)
            if not events:
                raise subprocess.TimeoutExpired(
                    arguments,
                    timeout_seconds,
                    output=bytes(stdout),
                    stderr=bytes(stderr),
                )

            for key, _event_mask in events:
                stream_name = key.data
                buffer = buffers[stream_name]
                remaining_bytes = maximum_output_bytes - len(buffer)
                read_size = min(65536, remaining_bytes + 1)
                try:
                    chunk = os.read(key.fd, read_size)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    streams[stream_name].close()
                    continue
                if len(chunk) > remaining_bytes:
                    raise _ProcessOutputLimitExceeded(
                        stream_name, maximum_output_bytes
                    )
                buffer.extend(chunk)

        _wait_for_process_exit_without_reaping(
            process,
            arguments=arguments,
            deadline=deadline,
            timeout_seconds=timeout_seconds,
            stdout=stdout,
            stderr=stderr,
        )
        _terminate_process_group(process)
        if process.returncode is None:
            raise VisualEvidenceError(
                "bounded evidence process was not reaped"
            )
        return subprocess.CompletedProcess(
            args=arguments,
            returncode=process.returncode,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
    finally:
        try:
            if process.returncode is None:
                _terminate_process_group(process)
        finally:
            try:
                if selector is not None:
                    selector.close()
            finally:
                try:
                    if stdout_stream is not None:
                        stdout_stream.close()
                finally:
                    if stderr_stream is not None:
                        stderr_stream.close()


def _decode_output(label: str, payload: bytes) -> str:
    if len(payload) > MAX_OUTPUT_BYTES:
        raise VisualEvidenceError(
            f"{label} stdout exceeds {MAX_OUTPUT_BYTES} bytes"
        )
    try:
        return payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise VisualEvidenceError(
            f"{label} stdout is not valid UTF-8"
        ) from error


def run_release_example(
    build_dir: Path, binary_name: str, sentinels: tuple[str, ...]
) -> str:
    """Run one prebuilt release example in a minimal deterministic environment."""

    binary = build_dir / binary_name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VisualEvidenceError(
            f"missing executable release example: {binary_name}"
        )

    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
    }
    try:
        completed = _run_bounded_process(
            [str(binary)],
            environment=environment,
            timeout_seconds=EXAMPLE_TIMEOUT_SECONDS,
            maximum_output_bytes=MAX_OUTPUT_BYTES,
        )
    except subprocess.TimeoutExpired as error:
        raise VisualEvidenceError(
            f"{binary_name} exceeded {EXAMPLE_TIMEOUT_SECONDS} seconds"
        ) from error
    except _ProcessOutputLimitExceeded as error:
        if error.stream == "stderr":
            raise VisualEvidenceError(
                f"{binary_name} wrote to stderr"
            ) from error
        raise VisualEvidenceError(
            f"{binary_name} stdout exceeds {MAX_OUTPUT_BYTES} bytes"
        ) from error
    except OSError as error:
        raise VisualEvidenceError(
            f"could not execute release example {binary_name}: {error}"
        ) from error

    if completed.returncode != 0:
        raise VisualEvidenceError(
            f"{binary_name} exited with status {completed.returncode}"
        )
    if completed.stderr:
        raise VisualEvidenceError(f"{binary_name} wrote to stderr")
    stdout = _decode_output(binary_name, completed.stdout)
    return validate_stdout(binary_name, stdout, sentinels)


def run_release_cli(
    build_dir: Path, label: str, arguments: tuple[str, ...]
) -> str:
    """Run one release CLI command twice and require byte-identical stdout."""

    binary = build_dir / "tensorkiln"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VisualEvidenceError(
            "missing executable release CLI: tensorkiln"
        )
    for argument in arguments:
        reject_unsafe_text(f"{label} argument", argument)

    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
    }
    payloads: list[bytes] = []
    for replay in range(CLI_REPLAYS):
        try:
            completed = _run_bounded_process(
                [str(binary), *arguments],
                environment=environment,
                timeout_seconds=EXAMPLE_TIMEOUT_SECONDS,
                maximum_output_bytes=MAX_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired as error:
            raise VisualEvidenceError(
                f"{label} replay {replay + 1} exceeded "
                f"{EXAMPLE_TIMEOUT_SECONDS} seconds"
            ) from error
        except _ProcessOutputLimitExceeded as error:
            if error.stream == "stderr":
                raise VisualEvidenceError(
                    f"{label} replay {replay + 1} wrote to stderr"
                ) from error
            raise VisualEvidenceError(
                f"{label} stdout exceeds {MAX_OUTPUT_BYTES} bytes"
            ) from error
        except OSError as error:
            raise VisualEvidenceError(
                f"could not execute {label} replay {replay + 1}: {error}"
            ) from error
        if completed.returncode != 0:
            raise VisualEvidenceError(
                f"{label} replay {replay + 1} exited with status "
                f"{completed.returncode}"
            )
        if completed.stderr:
            raise VisualEvidenceError(
                f"{label} replay {replay + 1} wrote to stderr"
            )
        if len(completed.stdout) > MAX_OUTPUT_BYTES:
            raise VisualEvidenceError(
                f"{label} stdout exceeds {MAX_OUTPUT_BYTES} bytes"
            )
        payloads.append(completed.stdout)

    if len(payloads) != CLI_REPLAYS or any(
        payload != payloads[0] for payload in payloads[1:]
    ):
        raise VisualEvidenceError(
            f"{label} replays are not byte-identical"
        )
    stdout = _decode_output(label, payloads[0])
    return validate_stdout(label, stdout, ())


def _read_regular_file(path: Path, maximum_bytes: int, label: str) -> bytes:
    """Read one non-symlink regular file while detecting concurrent changes."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VisualEvidenceError(f"cannot open {label}: {error}") from error

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VisualEvidenceError(f"{label} is not a regular file")
        if before.st_size > maximum_bytes:
            raise VisualEvidenceError(
                f"{label} exceeds the {maximum_bytes}-byte evidence limit"
            )

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise VisualEvidenceError(
                    f"{label} exceeds the {maximum_bytes}-byte evidence limit"
                )

        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise VisualEvidenceError(f"{label} changed while it was read")
        if total != after.st_size:
            raise VisualEvidenceError(f"{label} size changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _run_git(arguments: Sequence[str]) -> bytes:
    """Run the fixed Git binary without consulting user or system config."""

    if not GIT_BINARY.is_file() or not os.access(GIT_BINARY, os.X_OK):
        raise VisualEvidenceError(f"required Git binary is unavailable: {GIT_BINARY}")
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
    }
    try:
        completed = _run_bounded_process(
            [
                str(GIT_BINARY),
                "--no-optional-locks",
                "-C",
                str(REPOSITORY_ROOT),
                *arguments,
            ],
            environment=environment,
            timeout_seconds=GIT_TIMEOUT_SECONDS,
            maximum_output_bytes=MAX_OUTPUT_BYTES,
        )
    except subprocess.TimeoutExpired as error:
        raise VisualEvidenceError(
            f"Git provenance command exceeded {GIT_TIMEOUT_SECONDS} seconds"
        ) from error
    except _ProcessOutputLimitExceeded as error:
        if error.stream == "stderr":
            raise VisualEvidenceError(
                "Git provenance command wrote to stderr"
            ) from error
        raise VisualEvidenceError(
            "Git provenance output exceeds the evidence limit"
        ) from error
    except OSError as error:
        raise VisualEvidenceError(
            f"could not inspect Git provenance: {error}"
        ) from error

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise VisualEvidenceError(
            f"Git provenance command exited with status "
            f"{completed.returncode}{suffix}"
        )
    if completed.stderr:
        raise VisualEvidenceError("Git provenance command wrote to stderr")
    if len(completed.stdout) > MAX_OUTPUT_BYTES:
        raise VisualEvidenceError("Git provenance output exceeds the evidence limit")
    return completed.stdout


def _git_text(arguments: Sequence[str], label: str) -> str:
    payload = _run_git(arguments)
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise VisualEvidenceError(f"{label} is not ASCII") from error
    return text.strip()


def _decode_nul_paths(payload: bytes, label: str) -> tuple[str, ...]:
    """Decode a canonical, duplicate-free list of repository-relative paths."""

    if not payload or not payload.endswith(b"\0"):
        raise VisualEvidenceError(f"{label} is empty or not NUL-terminated")
    raw_paths = payload[:-1].split(b"\0")
    try:
        paths = tuple(
            raw_path.decode("utf-8", errors="strict") for raw_path in raw_paths
        )
    except UnicodeDecodeError as error:
        raise VisualEvidenceError(f"{label} contains a non-UTF-8 path") from error
    if len(paths) != len(set(paths)):
        raise VisualEvidenceError(f"{label} contains duplicate paths")
    for path in paths:
        parts = Path(path).parts
        if (
            not path
            or Path(path).is_absolute()
            or ".." in parts
            or "\n" in path
            or "\r" in path
        ):
            raise VisualEvidenceError(f"{label} contains an unsafe path")
    return tuple(sorted(paths))


def _tree_blob_record(
    revision: str, relative_path: str, object_pattern: re.Pattern[str]
) -> tuple[str, str]:
    tree_record = _run_git(
        ("ls-tree", "-z", revision, "--", relative_path)
    )
    if not tree_record.endswith(b"\0") or tree_record.count(b"\0") != 1:
        raise VisualEvidenceError(
            f"source tree has no unique record for {relative_path}"
        )
    try:
        metadata, recorded_path = tree_record[:-1].split(b"\t", maxsplit=1)
        mode, object_type, blob = metadata.decode(
            "ascii", errors="strict"
        ).split(" ")
        decoded_path = recorded_path.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as error:
        raise VisualEvidenceError(
            f"source tree record is malformed for {relative_path}"
        ) from error
    if (
        mode != "100644"
        or object_type != "blob"
        or decoded_path != relative_path
        or not object_pattern.fullmatch(blob)
    ):
        raise VisualEvidenceError(
            f"source tree record is not a non-symlink 100644 blob for "
            f"{relative_path}"
        )
    return mode, blob


def collect_source_provenance(
    include_cli: bool = False,
) -> dict[str, object]:
    """Bind evidence build inputs to one committed source snapshot."""

    pathspecs = EVIDENCE_SOURCE_PATHSPECS
    required_sources = REQUIRED_EVIDENCE_SOURCES
    if include_cli:
        pathspecs = (
            *EVIDENCE_SOURCE_PATHSPECS,
            *CLI_EVIDENCE_SOURCE_PATHSPECS,
        )
        required_sources = (
            REQUIRED_EVIDENCE_SOURCES
            | REQUIRED_CLI_EVIDENCE_SOURCES
        )

    root = _git_text(("rev-parse", "--show-toplevel"), "Git root")
    if Path(root).resolve() != REPOSITORY_ROOT:
        raise VisualEvidenceError("Git root differs from the TensorKiln repository")

    dirty = _run_git(
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *pathspecs,
        )
    )
    if dirty:
        raise VisualEvidenceError(
            "evidence build inputs differ from the committed source snapshot"
        )

    tracked_paths = _decode_nul_paths(
        _run_git(("ls-files", "-z", "--", *pathspecs)),
        "tracked evidence build-input set",
    )
    if not required_sources.issubset(tracked_paths):
        missing = sorted(required_sources.difference(tracked_paths))
        raise VisualEvidenceError(
            f"tracked evidence build-input set is missing {missing[0]}"
        )
    for path in tracked_paths:
        if not (
            path in required_sources
            or path.startswith("include/")
            or path.startswith("src/")
            or (include_cli and path.startswith("cli/"))
        ):
            raise VisualEvidenceError(
                f"tracked evidence build-input set escaped its pathspecs: {path}"
            )

    source_commit = _git_text(
        (
            "log",
            "-1",
            "--format=%H",
            "--",
            *pathspecs,
        ),
        "source commit",
    )
    object_format = _git_text(
        ("rev-parse", "--show-object-format"), "Git object format"
    )
    object_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if object_length is None:
        raise VisualEvidenceError(
            f"unsupported Git object format: {object_format}"
        )
    object_pattern = re.compile(rf"^[0-9a-f]{{{object_length}}}$")
    if not object_pattern.fullmatch(source_commit):
        raise VisualEvidenceError("source commit has a malformed object ID")

    source_tree = _git_text(
        ("show", "-s", "--format=%T", source_commit), "source tree"
    )
    if not object_pattern.fullmatch(source_tree):
        raise VisualEvidenceError("source tree has a malformed object ID")
    _run_git(("merge-base", "--is-ancestor", source_commit, "HEAD"))

    source_files: dict[str, dict[str, object]] = {}
    for relative_path in tracked_paths:
        mode, blob = _tree_blob_record(
            source_commit, relative_path, object_pattern
        )
        path = REPOSITORY_ROOT / relative_path
        payload = _read_regular_file(
            path, MAX_SOURCE_BYTES, f"source file {relative_path}"
        )

        head_blob = _git_text(
            ("rev-parse", f"HEAD:{relative_path}"),
            f"HEAD blob for {relative_path}",
        )
        working_blob = _git_text(
            ("hash-object", "--no-filters", "--", relative_path),
            f"working blob for {relative_path}",
        )
        if head_blob != blob or working_blob != blob:
            raise VisualEvidenceError(
                f"source blob differs from the selected commit for {relative_path}"
            )
        source_files[relative_path] = {
            "bytes": len(payload),
            "git_blob": blob,
            "mode": mode,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    return {
        "commit": source_commit,
        "object_format": object_format,
        "selection": (
            "latest commit touching the complete evidence build-input set"
        ),
        "source_files": source_files,
        "tree": source_tree,
    }


def collect_generator_provenance(
    allow_uncommitted: bool,
) -> dict[str, object]:
    """Bind the renderer to a committed blob, except in explicit preview mode."""

    generator_path = REPOSITORY_ROOT / GENERATOR_PATH
    payload = _read_regular_file(
        generator_path, MAX_SOURCE_BYTES, "visual evidence generator"
    )
    object_format = _git_text(
        ("rev-parse", "--show-object-format"), "Git object format"
    )
    object_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if object_length is None:
        raise VisualEvidenceError(
            f"unsupported Git object format: {object_format}"
        )
    object_pattern = re.compile(rf"^[0-9a-f]{{{object_length}}}$")
    working_blob = _git_text(
        ("hash-object", "--no-filters", "--", GENERATOR_PATH),
        "working generator blob",
    )
    if not object_pattern.fullmatch(working_blob):
        raise VisualEvidenceError("working generator has a malformed blob ID")

    dirty = _run_git(
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            GENERATOR_PATH,
        )
    )
    base_record: dict[str, object] = {
        "bytes": len(payload),
        "path": GENERATOR_PATH,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if dirty:
        if not allow_uncommitted:
            raise VisualEvidenceError(
                "visual evidence generator is not a committed, clean blob"
            )
        return {
            **base_record,
            "commit": None,
            "committed": False,
            "git_blob": None,
            "tree": None,
        }

    generator_commit = _git_text(
        ("log", "-1", "--format=%H", "--", GENERATOR_PATH),
        "generator commit",
    )
    if not object_pattern.fullmatch(generator_commit):
        raise VisualEvidenceError("generator commit has a malformed object ID")
    generator_tree = _git_text(
        ("show", "-s", "--format=%T", generator_commit), "generator tree"
    )
    if not object_pattern.fullmatch(generator_tree):
        raise VisualEvidenceError("generator tree has a malformed object ID")
    _run_git(("merge-base", "--is-ancestor", generator_commit, "HEAD"))
    _mode, committed_blob = _tree_blob_record(
        generator_commit, GENERATOR_PATH, object_pattern
    )
    head_blob = _git_text(
        ("rev-parse", f"HEAD:{GENERATOR_PATH}"), "HEAD generator blob"
    )
    if committed_blob != head_blob or committed_blob != working_blob:
        raise VisualEvidenceError(
            "visual evidence generator differs from its selected commit"
        )
    return {
        **base_record,
        "commit": generator_commit,
        "committed": True,
        "git_blob": committed_blob,
        "tree": generator_tree,
    }


def collect_binary_provenance(
    build_dir: Path, binary_names: Sequence[str]
) -> dict[str, dict[str, object]]:
    """Hash each exact release executable used to capture stdout."""

    records: dict[str, dict[str, object]] = {}
    for binary_name in binary_names:
        binary_path = build_dir / binary_name
        payload = _read_regular_file(
            binary_path, MAX_BINARY_BYTES, f"release binary {binary_name}"
        )
        if not os.access(binary_path, os.X_OK):
            raise VisualEvidenceError(
                f"release binary is not executable: {binary_name}"
            )
        records[binary_name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return records


def resolve_build_dir(argument: Path) -> Path:
    """Resolve an already-built release directory inside the repository."""

    candidate = argument if argument.is_absolute() else REPOSITORY_ROOT / argument
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise VisualEvidenceError(
            "--build-dir must stay inside the TensorKiln repository"
        ) from error
    if resolved.name != "release":
        raise VisualEvidenceError(
            "--build-dir must name an already-built release directory"
        )
    return resolved


def resolve_output_dir(argument: Path) -> Path:
    """Resolve an output directory without allowing repository escape."""

    candidate = argument if argument.is_absolute() else REPOSITORY_ROOT / argument
    resolved = candidate.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise VisualEvidenceError(
            "--output-dir must stay inside the TensorKiln repository"
        ) from error
    return resolved


def render_legacy_visuals(build_dir: Path) -> dict[str, str]:
    """Reproduce the committed v1 bundle until the v2 capture is published."""

    plan_stdout = run_release_example(
        build_dir, "plan_arena", PLAN_SENTINELS
    )
    execute_stdout = run_release_example(
        build_dir, "execute_graph", EXECUTE_SENTINELS
    )
    artifacts = {
        "arena-plan.txt": plan_stdout,
        "execute-graph.txt": execute_stdout,
        "execute-graph.svg": render_execute_graph_svg(
            execute_stdout, "<release-build>/execute_graph"
        ),
        "arena-reuse.svg": render_arena_reuse_svg(plan_stdout),
    }
    manifest = {
        "artifacts": {
            filename: {"sha256": _sha256(text)}
            for filename, text in sorted(artifacts.items())
        },
        "generator": "tools/render_readme_visuals.py",
        "reproduce": [
            "make -j2 PROFILE=release visuals",
            "make PROFILE=release visuals-check",
        ],
        "schema": "tensorkiln.readme-visual-evidence.v1",
        "scope": "verified deterministic examples; not benchmarks",
        "sources": {
            "execute_graph": {
                "binary": "execute_graph",
                "stdout_sha256": _sha256(execute_stdout),
            },
            "plan_arena": {
                "binary": "plan_arena",
                "stdout_sha256": _sha256(plan_stdout),
            },
        },
    }
    artifacts["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifacts


def render_visuals(
    build_dir: Path,
    include_softmax: bool = False,
    include_cli: bool = False,
    include_reglu: bool = False,
    allow_uncommitted_generator: bool = False,
) -> dict[str, Artifact]:
    """Collect verified stdout and return deterministic evidence artifacts."""

    if include_reglu:
        include_cli = True
    if include_cli:
        include_softmax = True
    if not include_softmax:
        if allow_uncommitted_generator:
            raise VisualEvidenceError(
                "uncommitted-generator preview requires a manifest-bound bundle"
            )
        return render_legacy_visuals(build_dir)

    binary_names = ("execute_graph", "execute_softmax", "plan_arena")
    if include_cli:
        binary_names = (*binary_names, "tensorkiln")
    source_before = collect_source_provenance(include_cli=include_cli)
    generator_before = collect_generator_provenance(
        allow_uncommitted_generator
    )
    binaries_before = collect_binary_provenance(build_dir, binary_names)
    plan_stdout = run_release_example(
        build_dir, "plan_arena", PLAN_SENTINELS
    )
    execute_stdout = run_release_example(
        build_dir, "execute_graph", EXECUTE_SENTINELS
    )
    softmax_stdout = run_release_example(
        build_dir, "execute_softmax", SOFTMAX_SENTINELS
    )
    softmax_stdout = validate_softmax_stdout(softmax_stdout)
    inspect_stdout: str | None = None
    cli_execute_stdout: str | None = None
    reglu_workloads_stdout: str | None = None
    reglu_inspect_stdout: str | None = None
    reglu_execute_stdout: str | None = None
    reglu_list_text_stdout: str | None = None
    reglu_inspect_text_stdout: str | None = None
    reglu_execute_text_stdout: str | None = None
    reglu_evidence: RegluEvidence | None = None
    if include_cli:
        inspect_stdout = run_release_cli(
            build_dir, "CLI inspect", CLI_INSPECT_ARGUMENTS
        )
        cli_execute_stdout = run_release_cli(
            build_dir, "CLI execute", CLI_EXECUTE_ARGUMENTS
        )
        validate_cli_evidence(inspect_stdout, cli_execute_stdout)
    if include_reglu:
        reglu_workloads_stdout = run_release_cli(
            build_dir, "ReGLU CLI list JSON", REGLU_LIST_ARGUMENTS
        )
        reglu_inspect_stdout = run_release_cli(
            build_dir, "ReGLU CLI inspect JSON", REGLU_INSPECT_ARGUMENTS
        )
        reglu_execute_stdout = run_release_cli(
            build_dir, "ReGLU CLI execute JSON", REGLU_EXECUTE_ARGUMENTS
        )
        reglu_list_text_stdout = run_release_cli(
            build_dir, "ReGLU CLI list text", REGLU_LIST_TEXT_ARGUMENTS
        )
        reglu_inspect_text_stdout = run_release_cli(
            build_dir,
            "ReGLU CLI inspect text",
            REGLU_INSPECT_TEXT_ARGUMENTS,
        )
        reglu_execute_text_stdout = run_release_cli(
            build_dir,
            "ReGLU CLI execute text",
            REGLU_EXECUTE_TEXT_ARGUMENTS,
        )
        reglu_evidence = validate_reglu_evidence(
            reglu_workloads_stdout,
            reglu_inspect_stdout,
            reglu_execute_stdout,
            reglu_list_text_stdout,
            reglu_inspect_text_stdout,
            reglu_execute_text_stdout,
        )
    source_after = collect_source_provenance(include_cli=include_cli)
    generator_after = collect_generator_provenance(
        allow_uncommitted_generator
    )
    binaries_after = collect_binary_provenance(build_dir, binary_names)
    if source_after != source_before:
        raise VisualEvidenceError(
            "evidence build inputs changed while examples were captured"
        )
    if generator_after != generator_before:
        raise VisualEvidenceError(
            "visual evidence generator changed while examples were captured"
        )
    if binaries_after != binaries_before:
        raise VisualEvidenceError(
            "a release executable changed while examples were captured"
        )

    artifacts = {
        "arena-plan.txt": plan_stdout,
        "execute-graph.txt": execute_stdout,
        "execute-graph.svg": render_execute_graph_svg(
            execute_stdout, "<release-build>/execute_graph"
        ),
        "execute-softmax.txt": softmax_stdout,
        "execute-softmax.svg": render_execute_softmax_svg(
            softmax_stdout, "<release-build>/execute_softmax"
        ),
        "arena-reuse.svg": render_arena_reuse_svg(plan_stdout),
    }
    if include_cli:
        assert inspect_stdout is not None
        assert cli_execute_stdout is not None
        artifacts.update(
            {
                "cli-inspect.json": inspect_stdout,
                "cli-execute.json": cli_execute_stdout,
                "cli-execution.svg": render_cli_execution_svg(
                    inspect_stdout, cli_execute_stdout
                ),
            }
        )
    if include_reglu:
        for filename, expected_sha256 in sorted(
            PUBLISHED_V3_ARTIFACT_SHA256.items()
        ):
            published_artifact = artifacts.get(filename)
            if published_artifact is None or hashlib.sha256(
                _artifact_bytes(published_artifact)
            ).hexdigest() != expected_sha256:
                raise VisualEvidenceError(
                    f"v4 capture would alter published v3 artifact {filename}"
                )
        assert reglu_workloads_stdout is not None
        assert reglu_inspect_stdout is not None
        assert reglu_execute_stdout is not None
        assert reglu_list_text_stdout is not None
        assert reglu_inspect_text_stdout is not None
        assert reglu_execute_text_stdout is not None
        assert reglu_evidence is not None
        artifacts.update(
            {
                "cli-workloads.json": reglu_workloads_stdout,
                "reglu-list.txt": reglu_list_text_stdout,
                "reglu-inspect.json": reglu_inspect_stdout,
                "reglu-inspect.txt": reglu_inspect_text_stdout,
                "reglu-execute.json": reglu_execute_stdout,
                "reglu-execute.txt": reglu_execute_text_stdout,
                "reglu-graph.svg": render_reglu_graph_svg(reglu_evidence),
                "reglu-arena.svg": render_reglu_arena_svg(reglu_evidence),
                "reglu-output.svg": render_reglu_output_svg(reglu_evidence),
                "reglu-terminal.png": render_reglu_terminal_png(
                    reglu_evidence
                ),
                "reglu-demo.gif": render_reglu_demo_gif(reglu_evidence),
                "reglu-demo-transcript.txt": render_reglu_demo_transcript(
                    reglu_evidence
                ),
            }
        )
    claim_boundary = [
        (
            "execute_softmax is crafted five-slice correctness evidence "
            "with 20/20 executor-reference and 20/20 executor-fixture bit "
            "agreements"
        ),
        (
            "the optimized last-axis kernel reports 60 scalar steps; the "
            "axis-zero reference path reports 80 total scalar steps and "
            "remains unsupported by the optimized planner"
        ),
        (
            "the Softmax bit-exact claim is limited to the committed "
            "f32[5,4] fixture, not arbitrary finite inputs or libm "
            "implementations"
        ),
        (
            "these deterministic correctness examples are not benchmarks "
            "or performance measurements"
        ),
        (
            "source blobs and captured executable bytes are hashed; the "
            "compiler, operating system, and binary supply chain are not "
            "attested"
        ),
    ]
    sources = {
        "execute_graph": {
            "binary": "execute_graph",
            "binary_bytes": binaries_before["execute_graph"]["bytes"],
            "binary_sha256": binaries_before["execute_graph"]["sha256"],
            "stdout_sha256": _sha256(execute_stdout),
        },
        "execute_softmax": {
            "binary": "execute_softmax",
            "binary_bytes": binaries_before["execute_softmax"]["bytes"],
            "binary_sha256": binaries_before["execute_softmax"]["sha256"],
            "fixture": "crafted f32[5,4] policy slices",
            "stdout_sha256": _sha256(softmax_stdout),
        },
        "plan_arena": {
            "binary": "plan_arena",
            "binary_bytes": binaries_before["plan_arena"]["bytes"],
            "binary_sha256": binaries_before["plan_arena"]["sha256"],
            "stdout_sha256": _sha256(plan_stdout),
        },
    }
    schema = "tensorkiln.readme-visual-evidence.v2"
    if include_cli:
        assert inspect_stdout is not None
        assert cli_execute_stdout is not None
        schema = "tensorkiln.readme-visual-evidence.v3"
        claim_boundary[0:0] = [
            (
                "CLI execution evidence is limited to dense_relu_v1 and the "
                "six committed input-bit values; the CLI is not a graph or "
                "model-file importer"
            ),
            (
                "the release CLI was replayed twice per command with "
                "byte-identical JSON; the execute report enables kernel-write "
                "auditing and records 4/4 raw-f32-bit reference agreement"
            ),
            (
                "the CLI evidence contains no timing fields and is not a "
                "benchmark or performance claim"
            ),
        ]
        sources["tensorkiln"] = {
            "binary": "tensorkiln",
            "binary_bytes": binaries_before["tensorkiln"]["bytes"],
            "binary_sha256": binaries_before["tensorkiln"]["sha256"],
            "commands": {
                "execute": {
                    "arguments": list(CLI_EXECUTE_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "tensorkiln.cli.execute.v1",
                    "stdout_artifact": "cli-execute.json",
                    "stdout_sha256": _sha256(cli_execute_stdout),
                },
                "inspect": {
                    "arguments": list(CLI_INSPECT_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "tensorkiln.cli.inspect.v1",
                    "stdout_artifact": "cli-inspect.json",
                    "stdout_sha256": _sha256(inspect_stdout),
                },
            },
        }
    if include_reglu:
        assert reglu_workloads_stdout is not None
        assert reglu_inspect_stdout is not None
        assert reglu_execute_stdout is not None
        assert reglu_list_text_stdout is not None
        assert reglu_inspect_text_stdout is not None
        assert reglu_execute_text_stdout is not None
        schema = "tensorkiln.readme-visual-evidence.v4"
        claim_boundary[0:0] = [
            (
                "reglu_mlp_v1 is a fixed ReGLU-style compiled-in MLP "
                "fixture; it is not a full transformer, model importer, "
                "or graph-file importer"
            ),
            (
                "the fixed six-word input produces eight executor words "
                "with 8/8 raw-f32-bit independent-reference agreement, "
                "including exact preservation of negative zero"
            ),
            (
                "registry, inspect, execute, and text terminal evidence was "
                "replayed twice per command with byte-identical stdout"
            ),
            (
                "GIF frame delays are deterministic presentation settings, "
                "not captured timings, latency measurements, or benchmark "
                "results"
            ),
        ]
        commands = sources["tensorkiln"]["commands"]
        assert isinstance(commands, dict)
        commands.update(
            {
                "reglu_execute_json": {
                    "arguments": list(REGLU_EXECUTE_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "tensorkiln.cli.execute.v1",
                    "stdout_artifact": "reglu-execute.json",
                    "stdout_sha256": _sha256(reglu_execute_stdout),
                },
                "reglu_execute_text": {
                    "arguments": list(REGLU_EXECUTE_TEXT_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "text/plain",
                    "stdout_artifact": "reglu-execute.txt",
                    "stdout_sha256": _sha256(reglu_execute_text_stdout),
                },
                "reglu_inspect_json": {
                    "arguments": list(REGLU_INSPECT_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "tensorkiln.cli.inspect.v1",
                    "stdout_artifact": "reglu-inspect.json",
                    "stdout_sha256": _sha256(reglu_inspect_stdout),
                },
                "reglu_inspect_text": {
                    "arguments": list(REGLU_INSPECT_TEXT_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "text/plain",
                    "stdout_artifact": "reglu-inspect.txt",
                    "stdout_sha256": _sha256(reglu_inspect_text_stdout),
                },
                "reglu_list_json": {
                    "arguments": list(REGLU_LIST_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "tensorkiln.cli.workloads.v1",
                    "stdout_artifact": "cli-workloads.json",
                    "stdout_sha256": _sha256(reglu_workloads_stdout),
                },
                "reglu_list_text": {
                    "arguments": list(REGLU_LIST_TEXT_ARGUMENTS),
                    "byte_identical": True,
                    "replays": CLI_REPLAYS,
                    "schema": "text/plain",
                    "stdout_artifact": "reglu-list.txt",
                    "stdout_sha256": _sha256(reglu_list_text_stdout),
                },
            }
        )
        sources["tensorkiln"]["presentation"] = {
            "gif_delay_centiseconds": list(TERMINAL_GIF_DELAYS_CS),
            "timing_semantics": "presentation only; not measurements",
            "transcript_artifact": "reglu-demo-transcript.txt",
        }

    capture_contract = {
        "complete_stdout": True,
        "environment": {
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
        },
        "exit_status": 0,
        "network_isolation": "not claimed",
        "stderr": "empty",
        "stdin": "closed",
        "timeout_seconds": EXAMPLE_TIMEOUT_SECONDS,
    }
    if include_cli:
        capture_contract["cli_replays_per_command"] = CLI_REPLAYS

    artifact_records: dict[str, dict[str, object]]
    if include_reglu:
        artifact_records = {
            filename: _artifact_manifest_record(filename, artifact)
            for filename, artifact in sorted(artifacts.items())
        }
    else:
        artifact_records = {}
        for filename, artifact in sorted(artifacts.items()):
            if not isinstance(artifact, str):
                raise VisualEvidenceError(
                    "pre-v4 visual bundle unexpectedly contains binary media"
                )
            artifact_records[filename] = {"sha256": _sha256(artifact)}

    manifest = {
        "artifacts": artifact_records,
        "capture_contract": capture_contract,
        "claim_boundary": claim_boundary,
        "generator": generator_before,
        "reproduce": [
            "make -j2 PROFILE=release visuals",
            "make PROFILE=release visuals-check",
        ],
        "repository_source": source_before,
        "schema": schema,
        "scope": "verified deterministic examples; not benchmarks",
        "sources": sources,
    }
    artifacts["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifacts


def output_bundle_schema(output_dir: Path) -> str | None:
    """Read one supported committed bundle schema without following symlinks."""

    manifest_path = output_dir / "manifest.json"
    try:
        os.lstat(manifest_path)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise VisualEvidenceError(
            f"cannot inspect evidence manifest: {error}"
        ) from error

    payload = _read_regular_file(
        manifest_path, MAX_OUTPUT_BYTES, "committed evidence manifest"
    )
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VisualEvidenceError(
            "committed evidence manifest is not valid UTF-8 JSON"
        ) from error
    if not isinstance(manifest, dict):
        raise VisualEvidenceError(
            "committed evidence manifest must be a JSON object"
        )
    schema = manifest.get("schema")
    if schema in {
        "tensorkiln.readme-visual-evidence.v1",
        "tensorkiln.readme-visual-evidence.v2",
        "tensorkiln.readme-visual-evidence.v3",
        "tensorkiln.readme-visual-evidence.v4",
    }:
        assert isinstance(schema, str)
        return schema
    raise VisualEvidenceError(
        f"committed evidence manifest has unsupported schema: {schema!r}"
    )


def output_uses_softmax_bundle(output_dir: Path) -> bool:
    """Select the v2-or-newer bundle after its publication."""

    return output_bundle_schema(output_dir) in {
        "tensorkiln.readme-visual-evidence.v2",
        "tensorkiln.readme-visual-evidence.v3",
        "tensorkiln.readme-visual-evidence.v4",
    }


def output_uses_cli_bundle(output_dir: Path) -> bool:
    """Select v3-or-newer CLI evidence after its publication."""

    return output_bundle_schema(output_dir) in {
        "tensorkiln.readme-visual-evidence.v3",
        "tensorkiln.readme-visual-evidence.v4",
    }


def output_uses_reglu_bundle(output_dir: Path) -> bool:
    """Select v4 ReGLU evidence after its publication."""

    return (
        output_bundle_schema(output_dir)
        == "tensorkiln.readme-visual-evidence.v4"
    )


def _validate_recorded_generator(
    generator: object,
) -> dict[str, object]:
    """Verify the exact committed generator that produced a v2 capture."""

    if not isinstance(generator, dict) or set(generator) != {
        "bytes",
        "commit",
        "committed",
        "git_blob",
        "path",
        "sha256",
        "tree",
    }:
        raise VisualEvidenceError(
            "committed evidence has malformed generator provenance"
        )
    if (
        generator.get("committed") is not True
        or generator.get("path") != GENERATOR_PATH
    ):
        raise VisualEvidenceError(
            "committed evidence generator is not a committed renderer"
        )

    byte_length = generator.get("bytes")
    sha256 = generator.get("sha256")
    commit = generator.get("commit")
    tree = generator.get("tree")
    blob = generator.get("git_blob")
    if (
        type(byte_length) is not int
        or byte_length <= 0
        or byte_length > MAX_SOURCE_BYTES
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise VisualEvidenceError(
            "committed evidence generator has malformed byte provenance"
        )

    object_format = _git_text(
        ("rev-parse", "--show-object-format"), "Git object format"
    )
    object_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if object_length is None:
        raise VisualEvidenceError(
            f"unsupported Git object format: {object_format}"
        )
    object_pattern = re.compile(rf"^[0-9a-f]{{{object_length}}}$")
    if not all(
        isinstance(value, str) and object_pattern.fullmatch(value)
        for value in (commit, tree, blob)
    ):
        raise VisualEvidenceError(
            "committed evidence generator has malformed Git provenance"
        )

    assert isinstance(commit, str)
    assert isinstance(tree, str)
    assert isinstance(blob, str)
    shallow = _git_text(
        ("rev-parse", "--is-shallow-repository"),
        "Git shallow-repository state",
    )
    if shallow not in {"false", "true"}:
        raise VisualEvidenceError(
            "Git reported a malformed shallow-repository state"
        )
    if shallow == "true":
        return generator

    _run_git(("merge-base", "--is-ancestor", commit, "HEAD"))
    recorded_tree = _git_text(
        ("show", "-s", "--format=%T", commit),
        "recorded generator tree",
    )
    _mode, recorded_blob = _tree_blob_record(
        commit, GENERATOR_PATH, object_pattern
    )
    payload = _run_git(("show", f"{commit}:{GENERATOR_PATH}"))
    if (
        recorded_tree != tree
        or recorded_blob != blob
        or len(payload) != byte_length
        or hashlib.sha256(payload).hexdigest() != sha256
    ):
        raise VisualEvidenceError(
            "committed evidence generator does not match its Git objects"
        )
    return generator


def _preserve_recorded_repository_source(
    recorded: object, current: object
) -> dict[str, object]:
    """Verify current inputs before retaining their historical capture commit."""

    expected_keys = {
        "commit",
        "object_format",
        "selection",
        "source_files",
        "tree",
    }
    if (
        not isinstance(recorded, dict)
        or not isinstance(current, dict)
        or set(recorded) != expected_keys
        or set(current) != expected_keys
        or recorded.get("object_format") != current.get("object_format")
        or recorded.get("selection") != current.get("selection")
    ):
        raise VisualEvidenceError(
            "committed evidence has malformed repository-source provenance"
        )
    if recorded.get("source_files") != current.get("source_files"):
        raise VisualEvidenceError(
            "current evidence build inputs differ from the recorded capture"
        )

    object_format = recorded["object_format"]
    object_length = {"sha1": 40, "sha256": 64}.get(object_format)
    if object_length is None:
        raise VisualEvidenceError(
            f"unsupported Git object format: {object_format}"
        )
    object_pattern = re.compile(rf"^[0-9a-f]{{{object_length}}}$")
    commit = recorded.get("commit")
    tree = recorded.get("tree")
    if not all(
        isinstance(value, str) and object_pattern.fullmatch(value)
        for value in (commit, tree)
    ):
        raise VisualEvidenceError(
            "committed evidence repository source has malformed Git provenance"
        )

    shallow = _git_text(
        ("rev-parse", "--is-shallow-repository"),
        "Git shallow-repository state",
    )
    if shallow not in {"false", "true"}:
        raise VisualEvidenceError(
            "Git reported a malformed shallow-repository state"
        )
    if shallow == "false":
        assert isinstance(commit, str)
        assert isinstance(tree, str)
        _run_git(("merge-base", "--is-ancestor", commit, "HEAD"))
        recorded_tree = _git_text(
            ("show", "-s", "--format=%T", commit),
            "recorded repository-source tree",
        )
        if recorded_tree != tree:
            raise VisualEvidenceError(
                "committed evidence repository source has the wrong Git tree"
            )

    current["commit"] = commit
    current["tree"] = tree
    return current


def _recorded_binary_fields(
    source: object, binary_name: str
) -> tuple[int, str]:
    """Read bounded historical binary metadata without claiming reproduction."""

    if not isinstance(source, dict) or source.get("binary") != binary_name:
        raise VisualEvidenceError(
            f"committed evidence has malformed {binary_name} source metadata"
        )
    byte_length = source.get("binary_bytes")
    sha256 = source.get("binary_sha256")
    if (
        type(byte_length) is not int
        or byte_length <= 0
        or byte_length > MAX_BINARY_BYTES
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise VisualEvidenceError(
            f"committed evidence has malformed {binary_name} binary provenance"
        )
    return byte_length, sha256


def _validate_preview_generator(generator: object) -> dict[str, object]:
    """Validate an explicitly uncommitted non-production renderer record."""

    if not isinstance(generator, dict) or set(generator) != {
        "bytes",
        "commit",
        "committed",
        "git_blob",
        "path",
        "sha256",
        "tree",
    }:
        raise VisualEvidenceError(
            "preview evidence has malformed generator provenance"
        )
    if (
        generator.get("committed") is not False
        or generator.get("path") != GENERATOR_PATH
        or any(
            generator.get(field) is not None
            for field in ("commit", "git_blob", "tree")
        )
    ):
        raise VisualEvidenceError(
            "preview evidence generator is not explicitly uncommitted"
        )
    byte_length = generator.get("bytes")
    sha256 = generator.get("sha256")
    if (
        type(byte_length) is not int
        or byte_length <= 0
        or byte_length > MAX_SOURCE_BYTES
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise VisualEvidenceError(
            "preview evidence generator has malformed byte provenance"
        )
    return generator


def preserve_recorded_capture_provenance(
    recorded: object,
    current: object,
    allow_uncommitted_generator: bool = False,
) -> dict[str, object]:
    """Keep historical generator/ELF identities during cross-toolchain checks.

    The current verifier still reconstructs every deterministic transcript,
    SVG, source identity, and non-capture manifest field. Compiler-specific
    executable bytes and the generator that performed the original capture
    remain historical facts instead of being falsely required to reproduce
    across GCC/Clang builds.
    """

    recorded_schema = (
        recorded.get("schema") if isinstance(recorded, dict) else None
    )
    current_schema = (
        current.get("schema") if isinstance(current, dict) else None
    )
    if (
        not isinstance(recorded, dict)
        or not isinstance(current, dict)
        or recorded_schema
        not in {
            "tensorkiln.readme-visual-evidence.v2",
            "tensorkiln.readme-visual-evidence.v3",
            "tensorkiln.readme-visual-evidence.v4",
        }
        or current_schema != recorded_schema
    ):
        raise VisualEvidenceError(
            "cannot reconcile incompatible visual evidence provenance"
        )
    recorded_generator = recorded.get("generator")
    current_generator = current.get("generator")
    recorded_is_committed = (
        isinstance(recorded_generator, dict)
        and recorded_generator.get("committed") is True
    )
    current_is_committed = (
        isinstance(current_generator, dict)
        and current_generator.get("committed") is True
    )
    if recorded_is_committed or current_is_committed:
        if not recorded_is_committed or not current_is_committed:
            raise VisualEvidenceError(
                "cannot reconcile committed and preview generators"
            )
        current["generator"] = _validate_recorded_generator(
            recorded_generator
        )
    else:
        if not allow_uncommitted_generator:
            raise VisualEvidenceError(
                "uncommitted generator is allowed only for an explicit preview"
            )
        recorded_preview = _validate_preview_generator(recorded_generator)
        current_preview = _validate_preview_generator(current_generator)
        if recorded_preview == current_preview:
            current["generator"] = recorded_preview
    current["repository_source"] = _preserve_recorded_repository_source(
        recorded.get("repository_source"),
        current.get("repository_source"),
    )

    recorded_sources = recorded.get("sources")
    current_sources = current.get("sources")
    if not isinstance(recorded_sources, dict) or not isinstance(
        current_sources, dict
    ):
        raise VisualEvidenceError(
            "committed evidence has malformed source provenance"
        )
    binary_names = ["execute_graph", "execute_softmax", "plan_arena"]
    if recorded_schema in {
        "tensorkiln.readme-visual-evidence.v3",
        "tensorkiln.readme-visual-evidence.v4",
    }:
        binary_names.append("tensorkiln")
    for binary_name in binary_names:
        recorded_source = recorded_sources.get(binary_name)
        current_source = current_sources.get(binary_name)
        if not isinstance(current_source, dict):
            raise VisualEvidenceError(
                f"current evidence has malformed {binary_name} source metadata"
            )
        byte_length, sha256 = _recorded_binary_fields(
            recorded_source, binary_name
        )
        current_source["binary_bytes"] = byte_length
        current_source["binary_sha256"] = sha256
    return current


def _normalize_manifest_for_check(
    current: bytes,
    expected: bytes,
    allow_uncommitted_generator: bool = False,
) -> bytes:
    """Retain capture-only provenance while checking all reproducible fields."""

    try:
        recorded_manifest = json.loads(current)
        expected_manifest = json.loads(expected)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VisualEvidenceError(
            "visual evidence manifest is not valid UTF-8 JSON"
        ) from error
    if not isinstance(recorded_manifest, dict) or not isinstance(
        expected_manifest, dict
    ):
        raise VisualEvidenceError(
            "visual evidence manifest must be a JSON object"
        )
    if recorded_manifest.get("schema") not in {
        "tensorkiln.readme-visual-evidence.v2",
        "tensorkiln.readme-visual-evidence.v3",
        "tensorkiln.readme-visual-evidence.v4",
    }:
        return expected
    normalized = preserve_recorded_capture_provenance(
        recorded_manifest,
        expected_manifest,
        allow_uncommitted_generator=allow_uncommitted_generator,
    )
    return (
        json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _validate_artifact_filename(filename: str) -> None:
    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise VisualEvidenceError(
            f"visual artifact filename escapes the output directory: {filename!r}"
        )


def _open_output_directory(output_dir: Path) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise VisualEvidenceError(
            "secure visual output requires O_DIRECTORY and O_NOFOLLOW"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(output_dir, flags)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise VisualEvidenceError(
            f"cannot securely open visual output directory: {error}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise VisualEvidenceError(
                "visual output destination is not a directory"
            )
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


def _read_output_artifact(directory_fd: int, filename: str) -> bytes:
    _validate_artifact_filename(filename)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        artifact_fd = os.open(filename, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as error:
        raise VisualEvidenceError(
            f"cannot securely open visual artifact {filename}: {error}"
        ) from error
    try:
        metadata = os.fstat(artifact_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise VisualEvidenceError(
                f"visual artifact is not a regular file: {filename}"
            )
        if metadata.st_size > MAX_ARTIFACT_BYTES:
            raise VisualEvidenceError(
                f"visual artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {filename}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(artifact_fd, min(65536, MAX_ARTIFACT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_ARTIFACT_BYTES:
                raise VisualEvidenceError(
                    f"visual artifact exceeds {MAX_ARTIFACT_BYTES} bytes: "
                    f"{filename}"
                )
        return b"".join(chunks)
    finally:
        os.close(artifact_fd)


def _write_output_artifact(
    directory_fd: int, filename: str, payload: bytes
) -> None:
    _validate_artifact_filename(filename)
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise VisualEvidenceError(
            f"visual artifact exceeds {MAX_ARTIFACT_BYTES} bytes: {filename}"
        )
    try:
        existing = os.stat(
            filename, dir_fd=directory_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise VisualEvidenceError(
            f"cannot inspect visual artifact destination {filename}: {error}"
        ) from error
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise VisualEvidenceError(
            f"visual artifact destination is not a regular file: {filename}"
        )

    temporary_name: str | None = None
    temporary_fd: int | None = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    for _attempt in range(32):
        candidate = f".{filename}.{os.urandom(12).hex()}.tmp"
        try:
            temporary_fd = os.open(
                candidate,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        except OSError as error:
            raise VisualEvidenceError(
                f"cannot create a secure temporary visual for {filename}: {error}"
            ) from error
        temporary_name = candidate
        break
    if temporary_fd is None or temporary_name is None:
        raise VisualEvidenceError(
            f"could not allocate a unique temporary visual for {filename}"
        )

    replaced = False
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(temporary_fd, view[written:])
            if count <= 0:
                raise VisualEvidenceError(
                    f"short write while creating visual artifact {filename}"
                )
            written += count
        os.fchmod(temporary_fd, 0o644)
        os.fsync(temporary_fd)
        descriptor_to_close = temporary_fd
        temporary_fd = None
        os.close(descriptor_to_close)
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        final = os.stat(
            filename, dir_fd=directory_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(final.st_mode):
            raise VisualEvidenceError(
                f"written visual artifact is not regular: {filename}"
            )
    except VisualEvidenceError:
        raise
    except OSError as error:
        raise VisualEvidenceError(
            f"cannot atomically publish visual artifact {filename}: {error}"
        ) from error
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if not replaced:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def check_visuals(
    output_dir: Path,
    generated: dict[str, Artifact],
    allow_uncommitted_generator: bool = False,
) -> int:
    """Return nonzero if committed visuals are absent or stale."""

    stale = False
    try:
        directory_fd = _open_output_directory(output_dir)
    except FileNotFoundError:
        for filename in generated:
            print(
                f"visuals: missing "
                f"{(output_dir / filename).relative_to(REPOSITORY_ROOT)}"
            )
        return 1
    try:
        for filename, artifact in generated.items():
            path = output_dir / filename
            try:
                current = _read_output_artifact(directory_fd, filename)
            except FileNotFoundError:
                print(f"visuals: missing {path.relative_to(REPOSITORY_ROOT)}")
                stale = True
                continue

            expected = _artifact_bytes(artifact)
            if filename == "manifest.json":
                expected = _normalize_manifest_for_check(
                    current,
                    expected,
                    allow_uncommitted_generator=allow_uncommitted_generator,
                )
            if current != expected:
                current_digest = hashlib.sha256(current).hexdigest()[:12]
                expected_digest = hashlib.sha256(expected).hexdigest()[:12]
                print(
                    f"visuals: stale {path.relative_to(REPOSITORY_ROOT)} "
                    f"({current_digest} != {expected_digest})"
                )
                stale = True
        try:
            output_names = os.listdir(directory_fd)
        except OSError as error:
            raise VisualEvidenceError(
                f"cannot enumerate {output_dir}: {error}"
            ) from error
        orphans = sorted(name for name in output_names if name not in generated)
        for orphan in orphans:
            print(
                "visuals: orphan "
                f"{(output_dir / orphan).relative_to(REPOSITORY_ROOT)}"
            )
            stale = True
    finally:
        os.close(directory_fd)
    return 1 if stale else 0


def write_visuals(output_dir: Path, generated: dict[str, Artifact]) -> None:
    """Write generated evidence artifacts atomically."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise VisualEvidenceError(
            f"cannot create visual output directory: {error}"
        ) from error
    directory_fd = _open_output_directory(output_dir)
    try:
        for filename, artifact in generated.items():
            _write_output_artifact(
                directory_fd, filename, _artifact_bytes(artifact)
            )
            path = output_dir / filename
            print(f"visuals: wrote {path.relative_to(REPOSITORY_ROOT)}")
        try:
            os.fsync(directory_fd)
        except OSError as error:
            raise VisualEvidenceError(
                f"cannot synchronize visual output directory: {error}"
            ) from error
    finally:
        os.close(directory_fd)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render reproducible README evidence from already-built, verified "
            "TensorKiln release examples."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=(
            "already-built release directory containing plan_arena and "
            "execute_graph, execute_softmax, and tensorkiln "
            f"(default: {DEFAULT_BUILD_DIR})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"evidence destination (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "verify that committed evidence exactly matches current "
            "example output"
        ),
    )
    parser.add_argument(
        "--capture-softmax-evidence",
        action="store_true",
        help=(
            "explicitly create the v2 manifest-bound Softmax bundle; after "
            "publication, the v2 manifest selects this bundle automatically"
        ),
    )
    parser.add_argument(
        "--capture-cli-evidence",
        action="store_true",
        help=(
            "explicitly create the v3 manifest-bound CLI bundle; after "
            "publication, the v3 manifest selects this bundle automatically"
        ),
    )
    parser.add_argument(
        "--capture-reglu-evidence",
        action="store_true",
        help=(
            "explicitly create the v4 manifest-bound ReGLU media bundle; "
            "after publication, the v4 manifest selects it automatically"
        ),
    )
    parser.add_argument(
        "--preview-uncommitted-generator",
        action="store_true",
        help=(
            "allow a dirty generator only for an explicitly non-production "
            "output directory; the preview manifest records no generator "
            "commit, tree, or blob"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        build_dir = resolve_build_dir(arguments.build_dir)
        output_dir = resolve_output_dir(arguments.output_dir)
        production_output_dir = resolve_output_dir(DEFAULT_OUTPUT_DIR)
        if (
            arguments.preview_uncommitted_generator
            and output_dir == production_output_dir
        ):
            raise VisualEvidenceError(
                "uncommitted-generator preview cannot target committed visuals"
            )
        if (
            arguments.preview_uncommitted_generator
            and not (
                arguments.capture_softmax_evidence
                or arguments.capture_cli_evidence
                or arguments.capture_reglu_evidence
            )
        ):
            raise VisualEvidenceError(
                "uncommitted-generator preview requires an explicit "
                "capture flag"
            )
        published_schema = output_bundle_schema(output_dir)
        include_reglu = (
            arguments.capture_reglu_evidence
            or published_schema
            == "tensorkiln.readme-visual-evidence.v4"
        )
        include_cli = (
            include_reglu
            or arguments.capture_cli_evidence
            or published_schema
            in {
                "tensorkiln.readme-visual-evidence.v3",
                "tensorkiln.readme-visual-evidence.v4",
            }
        )
        include_softmax = (
            include_cli
            or arguments.capture_softmax_evidence
            or published_schema
            in {
                "tensorkiln.readme-visual-evidence.v2",
                "tensorkiln.readme-visual-evidence.v3",
                "tensorkiln.readme-visual-evidence.v4",
            }
        )
        generated = render_visuals(
            build_dir,
            include_softmax=include_softmax,
            include_cli=include_cli,
            include_reglu=include_reglu,
            allow_uncommitted_generator=(
                arguments.preview_uncommitted_generator
            ),
        )
        if arguments.check:
            return check_visuals(
                output_dir,
                generated,
                allow_uncommitted_generator=(
                    arguments.preview_uncommitted_generator
                ),
            )
        write_visuals(output_dir, generated)
        return 0
    except VisualEvidenceError as error:
        print(f"visuals: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
