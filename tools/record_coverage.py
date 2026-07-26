#!/usr/bin/env python3
"""Record and verify TensorKiln's reproducible GCC/LCOV evidence bundle.

The recorder deliberately owns the complete capture sequence: it removes only
the dedicated coverage build directory, performs one silent instrumentation
build and test run, asks ``geninfo`` for an LCOV tracefile, independently
recomputes every summary from that tracefile, and emits deterministic,
path-normalized artifacts.

Only executable records rooted in ``src/`` are published. Public headers,
examples, and tests still drive the run, but their own coverage is excluded
from the reported production-source metric.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Final, Iterable, Sequence


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR: Final = REPOSITORY_ROOT / "docs/coverage/generated"
SOURCE_ROOT: Final = REPOSITORY_ROOT / "src"
ARTIFACT_NAMES: Final = (
    "coverage.info",
    "manifest.json",
    "summary.svg",
    "summary.txt",
    "test-run.txt",
)
EXPECTED_EXAMPLE_SENTINELS: Final = (
    "=== source graph ===",
    "=== verified interval arena plan ===",
    "=== verified dense execution plan ===",
    "=== verified Softmax execution ===",
)
DIRECT_INPUT_ROOTS: Final = ("examples", "include", "src", "tests")
INERT_CACHE_DIRECTORIES: Final = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
MAX_TRACE_BYTES: Final = 4 * 1024 * 1024
MAX_SOURCE_BYTES: Final = 2 * 1024 * 1024
MAX_INPUT_FILES: Final = 4096
MAX_INPUT_BYTES: Final = 64 * 1024 * 1024
MAX_TRANSCRIPT_BYTES: Final = 512 * 1024
COMMAND_TIMEOUT_SECONDS: Final = 300
TRACE_TAGS: Final = frozenset(
    {
        "BRDA",
        "BRF",
        "BRH",
        "DA",
        "FN",
        "FNDA",
        "FNF",
        "FNH",
        "LF",
        "LH",
        "SF",
        "TN",
    }
)
ABSOLUTE_HOST_PATH: Final = re.compile(
    r"(?<![A-Za-z0-9_.-])/"
    r"(?:home|root|Users|private|tmp|var|etc|opt|srv|mnt|workspace|workspaces)"
    r"(?:/|$)"
)
WINDOWS_USER_PATH: Final = re.compile(
    r"\b[A-Za-z]:\\(?:Users|Documents and Settings)\\", re.IGNORECASE
)
EMAIL_ADDRESS: Final = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
SECRET_SHAPES: Final = (
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
)


class CoverageEvidenceError(RuntimeError):
    """Raised when coverage cannot be captured or proved safely."""


@dataclass(frozen=True)
class Metric:
    """One exact covered/total pair."""

    covered: int
    total: int

    @property
    def percentage(self) -> float:
        if self.total == 0:
            return 100.0
        return (100.0 * self.covered) / self.total


@dataclass(frozen=True)
class FileCoverage:
    """Validated coverage totals for one repository-relative source."""

    source: str
    lines: Metric
    functions: Metric
    branches: Metric
    canonical_record: str


@dataclass(frozen=True)
class CoverageTrace:
    """A canonical LCOV tracefile and independently derived totals."""

    files: tuple[FileCoverage, ...]
    lines: Metric
    functions: Metric
    branches: Metric
    canonical_text: str


@dataclass(frozen=True)
class ToolIdentity:
    """A stable, path-free identity for one capture tool."""

    executable: Path
    label: str
    version: str


def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_unsafe_public_text(label: str, text: str) -> None:
    """Reject host identity, credentials, and unsafe control bytes."""

    for character in text:
        codepoint = ord(character)
        if (
            codepoint < 32
            and character not in "\n\t"
            or codepoint == 127
        ):
            raise CoverageEvidenceError(
                f"{label} contains a disallowed control character"
            )
    if ABSOLUTE_HOST_PATH.search(text):
        raise CoverageEvidenceError(f"{label} contains an absolute host path")
    if WINDOWS_USER_PATH.search(text):
        raise CoverageEvidenceError(f"{label} contains a Windows user path")
    if EMAIL_ADDRESS.search(text):
        raise CoverageEvidenceError(f"{label} contains an email address")
    if "file://" in text.lower():
        raise CoverageEvidenceError(f"{label} contains a local file URI")
    for pattern in SECRET_SHAPES:
        if pattern.search(text):
            raise CoverageEvidenceError(
                f"{label} contains a credential-shaped value"
            )


def _run(
    command: Sequence[str],
    *,
    label: str,
    cwd: Path = REPOSITORY_ROOT,
    environment: dict[str, str] | None = None,
    require_empty_stderr: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded command without a shell or inherited build flags."""

    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        raise CoverageEvidenceError(
            f"{label} failed with exit status {result.returncode}: {detail}"
        )
    if require_empty_stderr and result.stderr:
        raise CoverageEvidenceError(f"{label} wrote unexpected stderr")
    return result


def _capture_environment() -> dict[str, str]:
    """Return a narrow deterministic environment for compilation and capture."""

    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
    perl_library = os.environ.get("PERL5LIB")
    if perl_library:
        environment["PERL5LIB"] = perl_library
    return environment


def _resolve_executable(argument: str, label: str) -> Path:
    if not argument or "\0" in argument or "\n" in argument:
        raise CoverageEvidenceError(f"{label} executable is malformed")
    if any(character.isspace() for character in argument):
        raise CoverageEvidenceError(
            f"{label} must be one executable path without embedded flags"
        )
    candidate = (
        Path(argument)
        if "/" in argument
        else Path(shutil.which(argument) or "")
    )
    if not str(candidate):
        raise CoverageEvidenceError(f"{label} executable was not found")
    absolute = candidate.absolute()
    resolved = absolute.resolve()
    try:
        metadata = resolved.stat()
    except OSError as error:
        raise CoverageEvidenceError(
            f"cannot inspect {label} executable"
        ) from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise CoverageEvidenceError(
            f"{label} executable is not a regular executable file"
        )
    return absolute


def _first_version_line(
    executable: Path,
    arguments: Sequence[str],
    label: str,
    environment: dict[str, str],
) -> str:
    result = _run(
        (str(executable), *arguments),
        label=f"{label} version probe",
        environment=environment,
    )
    lines = [
        line.strip()
        for line in (result.stdout + result.stderr).splitlines()
        if line.strip()
    ]
    if not lines:
        raise CoverageEvidenceError(f"{label} returned an empty version")
    version = lines[0]
    reject_unsafe_public_text(f"{label} version", version)
    return version


def identify_tools(
    cxx_argument: str,
    gcov_argument: str,
    lcov_argument: str,
    geninfo_argument: str,
    environment: dict[str, str],
) -> dict[str, ToolIdentity]:
    """Resolve and cross-check the exact GCC and LCOV tool families."""

    cxx = _resolve_executable(cxx_argument, "C++ compiler")
    gcov = _resolve_executable(gcov_argument, "gcov")
    lcov = _resolve_executable(lcov_argument, "lcov")
    geninfo = _resolve_executable(geninfo_argument, "geninfo")

    identities = {
        "cxx": ToolIdentity(
            cxx,
            cxx.name,
            _first_version_line(cxx, ("--version",), "C++ compiler", environment),
        ),
        "gcov": ToolIdentity(
            gcov,
            gcov.name,
            _first_version_line(gcov, ("--version",), "gcov", environment),
        ),
        "lcov": ToolIdentity(
            lcov,
            lcov.name,
            _first_version_line(lcov, ("--version",), "lcov", environment),
        ),
        "geninfo": ToolIdentity(
            geninfo,
            geninfo.name,
            _first_version_line(
                geninfo, ("--version",), "geninfo", environment
            ),
        ),
    }

    gcc_version = re.search(r"\b(\d+)\.(\d+)\.\d+\b", identities["cxx"].version)
    gcov_version = re.search(
        r"\b(\d+)\.(\d+)\.\d+\b", identities["gcov"].version
    )
    if gcc_version is None or gcov_version is None:
        raise CoverageEvidenceError(
            "compiler and gcov must expose semantic GCC version numbers"
        )
    if gcc_version.groups() != gcov_version.groups():
        raise CoverageEvidenceError(
            "compiler and gcov major/minor versions do not match"
        )
    if "g++" not in identities["cxx"].version.lower():
        raise CoverageEvidenceError("coverage capture requires GNU g++")

    lcov_version = re.search(
        r"LCOV version ([0-9][A-Za-z0-9.+~-]*)",
        identities["lcov"].version,
    )
    geninfo_version = re.search(
        r"LCOV version ([0-9][A-Za-z0-9.+~-]*)",
        identities["geninfo"].version,
    )
    if lcov_version is None or geninfo_version is None:
        raise CoverageEvidenceError("LCOV tools returned malformed versions")
    if lcov_version.group(1) != geninfo_version.group(1):
        raise CoverageEvidenceError("lcov and geninfo versions do not match")
    if not lcov_version.group(1).startswith("2."):
        raise CoverageEvidenceError(
            "coverage capture requires the LCOV 2.x trace format"
        )
    return identities


def _parse_nonnegative_integer(value: str, label: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise CoverageEvidenceError(f"{label} is not a canonical integer")
    try:
        return int(value)
    except (ValueError, OverflowError) as error:
        raise CoverageEvidenceError(
            f"{label} exceeds the supported integer range"
        ) from error


def _parse_percentage(value: str, label: str) -> float:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
        raise CoverageEvidenceError(f"{label} is not a canonical percentage")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise CoverageEvidenceError(
            f"{label} exceeds the supported numeric range"
        ) from error
    if not isfinite(result):
        raise CoverageEvidenceError(
            f"{label} exceeds the supported numeric range"
        )
    return result


def _one_summary(
    fields: dict[str, list[str]], tag: str, fallback: int | None = None
) -> int:
    values = fields.get(tag, [])
    if not values and fallback is not None:
        return fallback
    if len(values) != 1:
        raise CoverageEvidenceError(f"LCOV record requires exactly one {tag}")
    return _parse_nonnegative_integer(values[0], f"{tag} summary")


def _source_path(source: str) -> str:
    """Normalize one LCOV source to a safe repository-relative POSIX path."""

    if not source or "\0" in source or "\n" in source:
        raise CoverageEvidenceError("LCOV source path is malformed")
    candidate = Path(source)
    if not candidate.is_absolute():
        pure = PurePosixPath(source)
        if pure.is_absolute() or ".." in pure.parts:
            raise CoverageEvidenceError("LCOV source path escapes the repository")
        candidate = REPOSITORY_ROOT.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except (OSError, ValueError) as error:
        raise CoverageEvidenceError(
            "LCOV source is not a regular repository file"
        ) from error
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise CoverageEvidenceError("cannot inspect LCOV source") from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or candidate.resolve() != candidate.absolute()
    ):
        raise CoverageEvidenceError("LCOV source must not be a symlink")
    normalized = relative.as_posix()
    if not normalized.startswith("src/"):
        raise CoverageEvidenceError("published LCOV source escaped src/")
    return normalized


def _canonical_record(
    fields: dict[str, list[str]], source: str
) -> FileCoverage:
    if set(fields).difference(TRACE_TAGS):
        unexpected = sorted(set(fields).difference(TRACE_TAGS))[0]
        raise CoverageEvidenceError(f"unsupported LCOV tag: {unexpected}")
    if fields.get("TN", [""]) != [""]:
        raise CoverageEvidenceError("LCOV test name must be empty")
    if fields.get("SF") != [source]:
        raise CoverageEvidenceError("LCOV record has a malformed source field")
    source_payload = (REPOSITORY_ROOT / source).read_bytes()
    if not source_payload or len(source_payload) > MAX_SOURCE_BYTES:
        raise CoverageEvidenceError(
            "LCOV source is empty or exceeds its byte limit"
        )
    source_line_count = source_payload.count(b"\n")
    if not source_payload.endswith(b"\n"):
        source_line_count += 1

    function_definitions: dict[str, tuple[int, int]] = {}
    canonical_functions: list[tuple[int, int, str, str]] = []
    for entry in fields.get("FN", []):
        parts = entry.split(",", 2)
        if len(parts) == 2:
            start_text, name = parts
            end_text = start_text
        elif len(parts) == 3:
            start_text, end_text, name = parts
        else:
            raise CoverageEvidenceError("malformed LCOV function definition")
        start = _parse_nonnegative_integer(start_text, "function start line")
        end = _parse_nonnegative_integer(end_text, "function end line")
        if (
            start == 0
            or end < start
            or end > source_line_count
            or not name
            or "\n" in name
        ):
            raise CoverageEvidenceError("malformed LCOV function definition")
        if name in function_definitions:
            raise CoverageEvidenceError("duplicate LCOV function definition")
        function_definitions[name] = (start, end)
        canonical_functions.append((start, end, name, f"FN:{entry}"))

    function_counts: dict[str, int] = {}
    for entry in fields.get("FNDA", []):
        parts = entry.split(",", 1)
        if len(parts) != 2:
            raise CoverageEvidenceError("malformed LCOV function count")
        count = _parse_nonnegative_integer(parts[0], "function count")
        name = parts[1]
        if name not in function_definitions or name in function_counts:
            raise CoverageEvidenceError(
                "LCOV function count does not match one definition"
            )
        function_counts[name] = count
    if set(function_counts) != set(function_definitions):
        raise CoverageEvidenceError("LCOV function counts are incomplete")

    line_counts: dict[int, int] = {}
    canonical_lines: list[tuple[int, str]] = []
    for entry in fields.get("DA", []):
        parts = entry.split(",")
        if len(parts) not in {2, 3}:
            raise CoverageEvidenceError("malformed LCOV line count")
        line = _parse_nonnegative_integer(parts[0], "covered line")
        count = _parse_nonnegative_integer(parts[1], "line execution count")
        if line == 0 or line > source_line_count or line in line_counts:
            raise CoverageEvidenceError("duplicate or zero LCOV line")
        if len(parts) == 3 and re.fullmatch(r"[A-Za-z0-9+/=]+", parts[2]) is None:
            raise CoverageEvidenceError("malformed LCOV line checksum")
        line_counts[line] = count
        canonical_lines.append((line, f"DA:{entry}"))

    branch_counts: dict[tuple[int, str, int], int | None] = {}
    canonical_branches: list[tuple[int, str, int, str]] = []
    for entry in fields.get("BRDA", []):
        parts = entry.split(",")
        if len(parts) != 4:
            raise CoverageEvidenceError("malformed LCOV branch count")
        line = _parse_nonnegative_integer(parts[0], "branch line")
        block = parts[1]
        branch = _parse_nonnegative_integer(parts[2], "branch ordinal")
        if (
            line == 0
            or line > source_line_count
            or re.fullmatch(r"[A-Za-z0-9]+", block) is None
            or (line, block, branch) in branch_counts
        ):
            raise CoverageEvidenceError("duplicate or malformed LCOV branch")
        taken = (
            None
            if parts[3] == "-"
            else _parse_nonnegative_integer(parts[3], "branch count")
        )
        branch_counts[(line, block, branch)] = taken
        canonical_branches.append((line, block, branch, f"BRDA:{entry}"))

    functions = Metric(
        sum(count > 0 for count in function_counts.values()),
        len(function_counts),
    )
    lines = Metric(sum(count > 0 for count in line_counts.values()), len(line_counts))
    branches = Metric(
        sum(count is not None and count > 0 for count in branch_counts.values()),
        len(branch_counts),
    )
    if _one_summary(fields, "FNF") != functions.total:
        raise CoverageEvidenceError("FNF differs from function records")
    if _one_summary(fields, "FNH") != functions.covered:
        raise CoverageEvidenceError("FNH differs from function records")
    if _one_summary(fields, "LF") != lines.total:
        raise CoverageEvidenceError("LF differs from line records")
    if _one_summary(fields, "LH") != lines.covered:
        raise CoverageEvidenceError("LH differs from line records")
    if _one_summary(fields, "BRF", 0) != branches.total:
        raise CoverageEvidenceError("BRF differs from branch records")
    if _one_summary(fields, "BRH", 0) != branches.covered:
        raise CoverageEvidenceError("BRH differs from branch records")

    canonical = ["TN:", f"SF:{source}"]
    canonical.extend(
        value for _start, _end, _name, value in sorted(canonical_functions)
    )
    canonical.extend(
        f"FNDA:{function_counts[name]},{name}"
        for name in sorted(function_counts)
    )
    canonical.extend((f"FNF:{functions.total}", f"FNH:{functions.covered}"))
    canonical.extend(
        value
        for _line, _block, _branch, value in sorted(canonical_branches)
    )
    if branches.total:
        canonical.extend((f"BRF:{branches.total}", f"BRH:{branches.covered}"))
    canonical.extend(value for _line, value in sorted(canonical_lines))
    canonical.extend((f"LF:{lines.total}", f"LH:{lines.covered}", "end_of_record"))
    return FileCoverage(
        source=source,
        lines=lines,
        functions=functions,
        branches=branches,
        canonical_record="\n".join(canonical) + "\n",
    )


def parse_lcov_trace(
    text: str,
    *,
    require_all_production_units: bool = True,
) -> CoverageTrace:
    """Parse, validate, normalize, and deterministically order an LCOV trace."""

    if not text or len(text.encode("utf-8")) > MAX_TRACE_BYTES:
        raise CoverageEvidenceError("LCOV trace is empty or exceeds its byte limit")
    normalized = text.replace("\r\n", "\n")
    if "\r" in normalized or not normalized.endswith("\n"):
        raise CoverageEvidenceError(
            "LCOV trace must be LF-normalized and newline-terminated"
        )

    raw_records = normalized.split("end_of_record\n")
    if raw_records[-1] != "":
        raise CoverageEvidenceError("LCOV trace has trailing record data")
    records: list[FileCoverage] = []
    seen_sources: set[str] = set()
    for raw_record in raw_records[:-1]:
        lines = raw_record.splitlines()
        source_values = [
            line[3:] for line in lines if line.startswith("SF:")
        ]
        if len(source_values) != 1:
            raise CoverageEvidenceError(
                "each LCOV record requires exactly one source"
            )
        candidate = Path(source_values[0])
        if not candidate.is_absolute():
            candidate = REPOSITORY_ROOT / source_values[0]
        try:
            relative = candidate.resolve(strict=True).relative_to(
                REPOSITORY_ROOT
            )
        except (OSError, ValueError) as error:
            raise CoverageEvidenceError(
                "LCOV capture contains a source outside the repository"
            ) from error
        if not relative.as_posix().startswith("src/"):
            continue
        source = _source_path(source_values[0])
        if source in seen_sources:
            raise CoverageEvidenceError("LCOV trace contains duplicate sources")
        seen_sources.add(source)
        fields: dict[str, list[str]] = {}
        for line in lines:
            if ":" not in line:
                raise CoverageEvidenceError("malformed LCOV field")
            tag, value = line.split(":", 1)
            fields.setdefault(tag, []).append(value)
        fields["SF"] = [source]
        record = _canonical_record(fields, source)
        if (
            record.lines.total == 0
            and record.functions.total == 0
            and record.branches.total == 0
        ):
            continue
        records.append(record)

    if not records:
        raise CoverageEvidenceError("LCOV trace has no production-source records")
    records.sort(key=lambda record: record.source)
    if require_all_production_units:
        expected = {
            path.relative_to(REPOSITORY_ROOT).as_posix()
            for path in SOURCE_ROOT.glob("*.cpp")
            if path.is_file() and not path.is_symlink()
        }
        recorded = {
            record.source for record in records if record.source.endswith(".cpp")
        }
        if recorded != expected:
            missing = sorted(expected.difference(recorded))
            extra = sorted(recorded.difference(expected))
            detail = missing[0] if missing else extra[0]
            raise CoverageEvidenceError(
                f"LCOV production translation-unit set differs at {detail}"
            )

    return CoverageTrace(
        files=tuple(records),
        lines=Metric(
            sum(record.lines.covered for record in records),
            sum(record.lines.total for record in records),
        ),
        functions=Metric(
            sum(record.functions.covered for record in records),
            sum(record.functions.total for record in records),
        ),
        branches=Metric(
            sum(record.branches.covered for record in records),
            sum(record.branches.total for record in records),
        ),
        canonical_text="".join(record.canonical_record for record in records),
    )


def _metric_text(metric: Metric) -> str:
    return f"{metric.percentage:.1f}% ({metric.covered}/{metric.total})"


def _validate_lcov_summary(summary: str, trace: CoverageTrace) -> None:
    """Cross-check LCOV's own summary with the independent parser."""

    patterns = {
        "lines": re.compile(
            r"^\s*lines\.*:\s*([0-9.]+)% \((\d+) of (\d+) lines\)\s*$",
            re.MULTILINE,
        ),
        "functions": re.compile(
            r"^\s*functions\.*:\s*([0-9.]+)% "
            r"\((\d+) of (\d+) functions\)\s*$",
            re.MULTILINE,
        ),
        "branches": re.compile(
            r"^\s*branches\.*:\s*([0-9.]+)% "
            r"\((\d+) of (\d+) branches\)\s*$",
            re.MULTILINE,
        ),
    }
    for name, metric in (
        ("lines", trace.lines),
        ("functions", trace.functions),
        ("branches", trace.branches),
    ):
        match = patterns[name].search(summary)
        if match is None:
            raise CoverageEvidenceError(f"LCOV summary omitted {name}")
        reported_percentage = _parse_percentage(
            match.group(1), f"LCOV {name} percentage"
        )
        reported_covered = _parse_nonnegative_integer(
            match.group(2), f"LCOV {name} covered count"
        )
        reported_total = _parse_nonnegative_integer(
            match.group(3), f"LCOV {name} total count"
        )
        if (
            reported_covered != metric.covered
            or reported_total != metric.total
            or abs(reported_percentage - round(metric.percentage, 1)) > 0.001
        ):
            raise CoverageEvidenceError(
                f"LCOV and independent {name} totals disagree"
            )


def validate_test_transcript(
    stdout: str,
    *,
    jobs: int = 2,
    compiler_label: str = "g++",
) -> tuple[str, int]:
    """Require the complete deterministic suite and every checked example."""

    normalized = stdout.replace("\r\n", "\n")
    if "\r" in normalized or not normalized.endswith("\n"):
        raise CoverageEvidenceError(
            "coverage test stdout must be LF-normalized and newline-terminated"
        )
    encoded = normalized.encode("utf-8")
    if not encoded or len(encoded) > MAX_TRANSCRIPT_BYTES:
        raise CoverageEvidenceError(
            "coverage test stdout is empty or exceeds its byte limit"
        )
    reject_unsafe_public_text("coverage test stdout", normalized)
    for sentinel in EXPECTED_EXAMPLE_SENTINELS:
        if normalized.count(sentinel) != 1:
            raise CoverageEvidenceError(
                f"coverage test stdout requires one {sentinel!r} sentinel"
            )
    summary_matches = re.findall(
        r"^(\d+)/(\d+) tests passed$", normalized, re.MULTILINE
    )
    if len(summary_matches) != 1:
        raise CoverageEvidenceError(
            "coverage test stdout requires one test summary"
        )
    passed = _parse_nonnegative_integer(
        summary_matches[0][0],
        "passed-test count",
    )
    total = _parse_nonnegative_integer(
        summary_matches[0][1],
        "total-test count",
    )
    if passed == 0 or passed != total:
        raise CoverageEvidenceError("coverage suite did not pass completely")
    if normalized.count("[pass] ") != passed:
        raise CoverageEvidenceError(
            "coverage test summary differs from individual passes"
        )
    if not 1 <= jobs <= 16 or re.fullmatch(
        r"[A-Za-z0-9_.+-]+", compiler_label
    ) is None:
        raise CoverageEvidenceError("test transcript command is malformed")
    command = (
        f"$ make -s -j{jobs} CXX={compiler_label} "
        "PROFILE=coverage test\n"
    )
    return command + normalized, passed


def render_summary_text(
    trace: CoverageTrace,
    *,
    test_count: int,
    compiler_version: str,
    lcov_version: str,
) -> str:
    """Render an exact, grep-friendly human report from parsed evidence."""

    review_queue = sorted(
        trace.files,
        key=lambda record: (
            -(record.lines.total - record.lines.covered),
            record.source,
        ),
    )
    lines = [
        "TensorKiln production coverage evidence",
        "",
        "scope: executable line, function, and GCC branch-edge records under src/",
        (
            f"run: {test_count}/{test_count} C++ tests and "
            f"{len(EXPECTED_EXAMPLE_SENTINELS)}/"
            f"{len(EXPECTED_EXAMPLE_SENTINELS)} checked examples passed"
        ),
        f"lines: {_metric_text(trace.lines)}",
        f"functions: {_metric_text(trace.functions)}",
        f"branches: {_metric_text(trace.branches)}",
        f"instrumented files: {len(trace.files)}",
        f"compiler: {compiler_version}",
        f"lcov: {lcov_version}",
        "",
        "largest uncovered-line sets:",
    ]
    for record in review_queue[:10]:
        uncovered = record.lines.total - record.lines.covered
        lines.append(
            f"  {record.source}: {uncovered} uncovered, "
            f"{_metric_text(record.lines)}"
        )
    lines.extend(
        (
            "",
            "claim boundary:",
            "  one clean deterministic instrumentation run; not a benchmark",
            "  tests, examples, public headers, and dependencies are excluded",
            "  percentages are observations, not release gates or quality scores",
            (
                "  branch totals are GCC control-flow edges and include "
                "compiler-generated exception paths"
            ),
        )
    )
    text = "\n".join(lines) + "\n"
    reject_unsafe_public_text("coverage summary", text)
    return text


def _svg_text(
    x: int,
    y: int,
    value: str,
    *,
    size: int = 14,
    fill: str = "#cbd5e1",
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="ui-monospace, SFMono-Regular, '
        f'Menlo, Consolas, monospace" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">'
        f"{html.escape(value)}</text>"
    )


def render_summary_svg(
    trace: CoverageTrace,
    *,
    test_count: int,
    compiler_short: str,
    lcov_short: str,
    source_digest: str,
) -> str:
    """Render a self-contained, data-derived coverage and workflow panel."""

    width = 1200
    height = 820
    output = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title description">'
        ),
        "<title id=\"title\">TensorKiln production-source coverage</title>",
        (
            '<desc id="description">Real GCC and LCOV coverage totals plus '
            "the largest uncovered line sets from one clean test run.</desc>"
        ),
        '<rect width="1200" height="820" rx="24" fill="#0b1220"/>',
        '<rect x="24" y="24" width="1152" height="772" rx="18" '
        'fill="#111a2e" stroke="#334155"/>',
        _svg_text(
            58,
            72,
            "GCOV + LCOV / PRODUCTION SOURCE EVIDENCE",
            size=14,
            fill="#67e8f9",
            weight=700,
        ),
        _svg_text(
            58,
            112,
            "What the deterministic suite actually executes",
            size=27,
            fill="#f8fafc",
            weight=700,
        ),
        _svg_text(
            1140,
            70,
            "NOT A QUALITY SCORE",
            size=13,
            fill="#fbbf24",
            weight=700,
            anchor="end",
        ),
    ]

    cards = (
        ("LINES", trace.lines, "#22d3ee"),
        ("FUNCTIONS", trace.functions, "#a78bfa"),
        ("GCC BRANCH EDGES", trace.branches, "#fb7185"),
    )
    for index, (label, metric, color) in enumerate(cards):
        x = 58 + index * 370
        output.extend(
            (
                f'<rect x="{x}" y="145" width="342" height="118" rx="14" '
                'fill="#0f172a" stroke="#334155"/>',
                _svg_text(x + 20, 176, label, size=12, fill="#94a3b8", weight=700),
                _svg_text(
                    x + 20,
                    220,
                    f"{metric.percentage:.1f}%",
                    size=34,
                    fill=color,
                    weight=700,
                ),
                _svg_text(
                    x + 322,
                    218,
                    f"{metric.covered} / {metric.total}",
                    size=13,
                    fill="#cbd5e1",
                    anchor="end",
                ),
                f'<rect x="{x + 20}" y="238" width="302" height="8" rx="4" '
                'fill="#26334a"/>',
                f'<rect x="{x + 20}" y="238" '
                f'width="{302.0 * metric.covered / max(metric.total, 1):.1f}" '
                f'height="8" rx="4" fill="{color}"/>',
            )
        )

    output.extend(
        (
            _svg_text(
                58,
                305,
                "LARGEST UNCOVERED-LINE SETS",
                size=13,
                fill="#f8fafc",
                weight=700,
            ),
            _svg_text(
                1140,
                305,
                "ordered by uncovered count, not by percentage",
                size=12,
                fill="#94a3b8",
                anchor="end",
            ),
        )
    )
    review_queue = sorted(
        trace.files,
        key=lambda record: (
            -(record.lines.total - record.lines.covered),
            record.source,
        ),
    )[:10]
    bar_x = 520
    bar_width = 470
    for index, record in enumerate(review_queue):
        y = 338 + index * 34
        uncovered = record.lines.total - record.lines.covered
        covered_bar_width = (
            bar_width * record.lines.covered / max(record.lines.total, 1)
        )
        output.extend(
            (
                _svg_text(58, y + 12, record.source.removeprefix("src/"), size=12),
                f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="13" '
                'rx="6" fill="#26334a"/>',
                f'<rect x="{bar_x}" y="{y}" '
                f'width="{covered_bar_width:.1f}" '
                'height="13" rx="6" fill="#22d3ee"/>',
                _svg_text(
                    1140,
                    y + 12,
                    f"{record.lines.covered}/{record.lines.total}"
                    f" · {uncovered} open",
                    size=12,
                    fill="#cbd5e1",
                    anchor="end",
                ),
            )
        )

    output.extend(
        (
            '<line x1="58" y1="690" x2="1142" y2="690" stroke="#334155"/>',
            _svg_text(
                58,
                720,
                "CLEAN BUILD",
                size=11,
                fill="#67e8f9",
                weight=700,
            ),
            _svg_text(170, 720, "→", size=16, fill="#64748b"),
            _svg_text(
                205,
                720,
                f"{test_count} TESTS + 4 EXAMPLES",
                size=11,
                fill="#67e8f9",
                weight=700,
            ),
            _svg_text(405, 720, "→", size=16, fill="#64748b"),
            _svg_text(
                440,
                720,
                "GCC COUNTERS",
                size=11,
                fill="#67e8f9",
                weight=700,
            ),
            _svg_text(565, 720, "→", size=16, fill="#64748b"),
            _svg_text(
                600,
                720,
                "LCOV TRACE",
                size=11,
                fill="#67e8f9",
                weight=700,
            ),
            _svg_text(705, 720, "→", size=16, fill="#64748b"),
            _svg_text(
                740,
                720,
                "INDEPENDENT RECOUNT",
                size=11,
                fill="#67e8f9",
                weight=700,
            ),
            _svg_text(915, 720, "→", size=16, fill="#64748b"),
            _svg_text(
                950,
                720,
                "HASHED BUNDLE",
                size=11,
                fill="#67e8f9",
                weight=700,
            ),
            _svg_text(
                58,
                758,
                (
                    f"src/ only · {len(trace.files)} instrumented files · "
                    f"{compiler_short} · {lcov_short}"
                ),
                size=12,
                fill="#94a3b8",
            ),
            _svg_text(
                1140,
                758,
                f"input snapshot sha256:{source_digest[:12]}",
                size=12,
                fill="#94a3b8",
                anchor="end",
            ),
            "</svg>",
        )
    )
    svg = "\n".join(output) + "\n"
    reject_unsafe_public_text("coverage SVG", svg)
    return svg


def _git_output(arguments: Sequence[str]) -> bytes:
    result = subprocess.run(
        ("/usr/bin/git", *arguments),
        cwd=REPOSITORY_ROOT,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise CoverageEvidenceError(
            f"Git provenance command failed: {result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def _is_inert_cache_path(relative_path: str) -> bool:
    return any(
        part in INERT_CACHE_DIRECTORIES
        for part in PurePosixPath(relative_path).parts
    )


def _direct_input_paths() -> set[str]:
    """Inventory every file that can affect this build or evidence recorder."""

    selected = {
        "Makefile",
        "tests/test_coverage_evidence.py",
        "tools/record_coverage.py",
    }
    for root_name in DIRECT_INPUT_ROOTS:
        root = REPOSITORY_ROOT / root_name
        try:
            root_metadata = root.lstat()
        except OSError as error:
            raise CoverageEvidenceError(
                f"direct input root is missing: {root_name}"
            ) from error
        if not stat.S_ISDIR(root_metadata.st_mode) or root.is_symlink():
            raise CoverageEvidenceError(
                f"direct input root must be a real directory: {root_name}"
            )
        for path in root.rglob("*"):
            relative_to_root = path.relative_to(root)
            if _is_inert_cache_path(relative_to_root.as_posix()):
                continue
            try:
                metadata = path.lstat()
            except OSError as error:
                raise CoverageEvidenceError(
                    f"cannot inspect direct input under {root_name}"
                ) from error
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            if path.is_symlink():
                raise CoverageEvidenceError(
                    f"direct input must not be a symlink: {relative_path}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise CoverageEvidenceError(
                    f"direct input must be a regular file: {relative_path}"
                )
            selected.add(relative_path)
            if len(selected) > MAX_INPUT_FILES:
                raise CoverageEvidenceError(
                    f"direct inputs exceed the {MAX_INPUT_FILES}-file limit"
                )
    return selected


def collect_source_snapshot() -> dict[str, object]:
    """Hash every direct build, run, and recorder input in the working tree."""

    selected = _direct_input_paths()
    tracked_payload = _git_output(
        (
            "ls-files",
            "-z",
            "--cached",
            "--",
            "Makefile",
            *DIRECT_INPUT_ROOTS,
            "tools/record_coverage.py",
        )
    )
    tracked = {
        item.decode("utf-8")
        for item in tracked_payload.split(b"\0")
        if item and not _is_inert_cache_path(item.decode("utf-8"))
    }
    records: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for relative_path in sorted(selected):
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise CoverageEvidenceError("source input path is unsafe")
        path = REPOSITORY_ROOT.joinpath(*pure.parts)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise CoverageEvidenceError(
                f"source input is missing: {relative_path}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise CoverageEvidenceError(
                f"source input is not a regular file: {relative_path}"
            )
        payload = path.read_bytes()
        if len(payload) > MAX_SOURCE_BYTES:
            raise CoverageEvidenceError(
                f"source input exceeds its byte limit: {relative_path}"
            )
        total_bytes += len(payload)
        if total_bytes > MAX_INPUT_BYTES:
            raise CoverageEvidenceError(
                f"direct inputs exceed the {MAX_INPUT_BYTES}-byte limit"
            )
        blob = _git_output(
            ("hash-object", "--no-filters", "--", relative_path)
        ).decode("ascii").strip()
        if re.fullmatch(r"[0-9a-f]{40,64}", blob) is None:
            raise CoverageEvidenceError("Git returned a malformed blob ID")
        records[relative_path] = {
            "bytes": len(payload),
            "git_blob": blob,
            "sha256": _sha256(payload),
        }
    serialized = json.dumps(
        records, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    source_commit = _git_output(
        (
            "log",
            "-1",
            "--format=%H",
            "--",
            *sorted(tracked),
        )
    ).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", source_commit) is None:
        raise CoverageEvidenceError("Git returned a malformed source commit")
    source_tree = _git_output(
        ("show", "-s", "--format=%T", source_commit)
    ).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", source_tree) is None:
        raise CoverageEvidenceError("Git returned a malformed source tree")
    status = _git_output(
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *sorted(selected.union(tracked)),
        )
    )
    selected_paths_clean = (
        selected == tracked
        and not bool(status)
    )
    if selected_paths_clean:
        for relative_path, record in records.items():
            committed_blob = _git_output(
                ("rev-parse", f"{source_commit}:{relative_path}")
            ).decode("ascii").strip()
            if committed_blob != record["git_blob"]:
                raise CoverageEvidenceError(
                    "clean source input differs from its selected commit: "
                    f"{relative_path}"
                )
    return {
        "commit_bound": selected_paths_clean,
        "files": records,
        "input_bytes": total_bytes,
        "selection": "latest commit touching the direct evidence-input set",
        "selected_paths_clean": selected_paths_clean,
        "sha256": _sha256(serialized),
        "source_commit": source_commit,
        "source_tree": source_tree,
    }


def _clean_coverage_build(compiler_label: str) -> Path:
    """Remove only build/<compiler>/coverage after strict path checks."""

    if re.fullmatch(r"[A-Za-z0-9_.+-]+", compiler_label) is None:
        raise CoverageEvidenceError("compiler filename is unsafe for a build path")
    build_root = REPOSITORY_ROOT / "build"
    compiler_root = build_root / compiler_label
    build_dir = compiler_root / "coverage"
    expected = PurePosixPath("build", compiler_label, "coverage")
    if build_dir.relative_to(REPOSITORY_ROOT).as_posix() != expected.as_posix():
        raise CoverageEvidenceError("coverage build path is malformed")
    existing: dict[Path, os.stat_result | None] = {}
    for path, label in (
        (build_root, "build root"),
        (compiler_root, "compiler build root"),
        (build_dir, "coverage build path"),
    ):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            metadata = None
        except OSError as error:
            raise CoverageEvidenceError(f"cannot inspect {label}") from error
        if metadata is not None and (
            path.is_symlink() or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise CoverageEvidenceError(
                f"{label} must be a real directory"
            )
        existing[path] = metadata
    if existing[build_dir] is not None:
        shutil.rmtree(build_dir)
    return build_dir


def _counter_inventory(build_dir: Path) -> dict[str, int]:
    gcda = sorted(build_dir.rglob("*.gcda"))
    gcno = sorted(build_dir.rglob("*.gcno"))
    if not gcda or len(gcda) != len(gcno):
        raise CoverageEvidenceError(
            "coverage build has incomplete gcda/gcno counter pairs"
        )
    gcda_relative = {
        path.relative_to(build_dir).with_suffix("").as_posix() for path in gcda
    }
    gcno_relative = {
        path.relative_to(build_dir).with_suffix("").as_posix() for path in gcno
    }
    if gcda_relative != gcno_relative:
        raise CoverageEvidenceError(
            "coverage build has mismatched gcda/gcno counter names"
        )
    expected_production = {
        f"src/{path.stem}"
        for path in SOURCE_ROOT.glob("*.cpp")
        if path.is_file() and not path.is_symlink()
    }
    if not expected_production.issubset(gcda_relative):
        missing = sorted(expected_production.difference(gcda_relative))[0]
        raise CoverageEvidenceError(
            f"coverage counters omit production unit {missing}"
        )
    return {
        "counter_pairs": len(gcda),
        "production_translation_units": len(expected_production),
    }


def _capture_trace(
    build_dir: Path,
    tools: dict[str, ToolIdentity],
    environment: dict[str, str],
) -> tuple[CoverageTrace, str]:
    """Capture with geninfo and independently ask lcov for its summary."""

    build_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="capture-", dir=build_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        raw_trace = temporary_dir / "raw.info"
        empty_config = temporary_dir / "lcovrc"
        empty_config.write_text("", encoding="ascii")
        _run(
            (
                str(tools["geninfo"].executable),
                str(build_dir),
                "--output-filename",
                str(raw_trace),
                "--base-directory",
                str(REPOSITORY_ROOT),
                "--gcov-tool",
                str(tools["gcov"].executable),
                "--config-file",
                str(empty_config),
                "--branch-coverage",
                "--no-checksum",
                "--no-external",
                "--parallel",
                "1",
                "--rc",
                "geninfo_unexecuted_blocks=1",
            ),
            label="LCOV geninfo capture",
            environment=environment,
        )
        payload = raw_trace.read_bytes()
        if len(payload) > MAX_TRACE_BYTES:
            raise CoverageEvidenceError("captured LCOV trace exceeds its byte limit")
        try:
            trace = parse_lcov_trace(payload.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise CoverageEvidenceError("LCOV trace is not valid UTF-8") from error
        canonical_path = temporary_dir / "canonical.info"
        canonical_path.write_text(trace.canonical_text, encoding="utf-8")
        summary = _run(
            (
                str(tools["lcov"].executable),
                "--summary",
                str(canonical_path),
                "--config-file",
                str(empty_config),
                "--branch-coverage",
            ),
            label="LCOV summary",
            environment=environment,
        ).stdout
        _validate_lcov_summary(summary, trace)
        return trace, summary


def _manifest(
    *,
    artifacts: dict[str, bytes],
    trace: CoverageTrace,
    test_count: int,
    tools: dict[str, ToolIdentity],
    source_snapshot: dict[str, object],
    counters: dict[str, int],
    jobs: int,
) -> str:
    expected_payloads = set(ARTIFACT_NAMES).difference({"manifest.json"})
    if set(artifacts) != expected_payloads:
        raise CoverageEvidenceError(
            "manifest payload artifact set is incomplete"
        )
    lcov_match = re.search(
        r"LCOV version ([0-9][A-Za-z0-9.+~-]*)", tools["lcov"].version
    )
    assert lcov_match is not None
    manifest = {
        "artifacts": {
            name: {
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
            for name, payload in sorted(artifacts.items())
        },
        "capture": {
            "command": (
                f"make -s -j{jobs} CXX={tools['cxx'].label} "
                "PROFILE=coverage test"
            ),
            "counter_reset": "dedicated build directory removed before run",
            "environment": {
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PERL5LIB_policy": (
                    "inherited when set for LCOV module lookup; its path "
                    "is not published or provenance-bound"
                ),
                "TZ": "UTC",
            },
            "jobs": jobs,
            "network_isolation": "not enforced",
            "stderr": "empty for the build and test command",
            "test_cases_passed": test_count,
            "checked_examples_passed": len(EXPECTED_EXAMPLE_SENTINELS),
            **counters,
        },
        "claim_boundary": [
            "one deterministic instrumentation run; not a benchmark",
            "only executable records rooted in src/ are reported",
            "tests, examples, public headers, and dependencies are excluded",
            "coverage observations are not release gates or quality scores",
            (
                "branch records are GCC control-flow edges and include "
                "compiler-generated exception paths"
            ),
            (
                "byte-identical reproduction requires matching compiler and "
                "LCOV versions; exact source inputs are hashed"
            ),
            (
                "an inherited PERL5LIB may select LCOV Perl modules and is "
                "an explicitly unbound tool-installation input"
            ),
        ],
        "metrics": {
            "branches": {
                "covered": trace.branches.covered,
                "percentage": round(trace.branches.percentage, 1),
                "total": trace.branches.total,
            },
            "functions": {
                "covered": trace.functions.covered,
                "percentage": round(trace.functions.percentage, 1),
                "total": trace.functions.total,
            },
            "instrumented_files": len(trace.files),
            "lines": {
                "covered": trace.lines.covered,
                "percentage": round(trace.lines.percentage, 1),
                "total": trace.lines.total,
            },
        },
        "reproduce": [
            (
                f"make COVERAGE_JOBS={jobs} CXX={tools['cxx'].label} "
                f"GCOV={tools['gcov'].label} LCOV={tools['lcov'].label} "
                f"GENINFO={tools['geninfo'].label} coverage"
            ),
            (
                f"make COVERAGE_JOBS={jobs} CXX={tools['cxx'].label} "
                f"GCOV={tools['gcov'].label} LCOV={tools['lcov'].label} "
                f"GENINFO={tools['geninfo'].label} coverage-check"
            ),
        ],
        "schema": "tensorkiln.coverage-evidence.v1",
        "source_snapshot": source_snapshot,
        "tools": {
            name: {
                "executable": identity.label,
                "version": identity.version,
            }
            for name, identity in sorted(tools.items())
        },
        "trace": {
            "format": "LCOV tracefile",
            "lcov_version": lcov_match.group(1),
            "normalization": (
                "repository-relative src paths and deterministic record ordering"
            ),
            "scope": "src/",
        },
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    reject_unsafe_public_text("coverage manifest", text)
    return text


def _publish_or_check(
    output_dir: Path,
    generated: dict[str, bytes],
    *,
    check: bool,
) -> None:
    for candidate in (
        REPOSITORY_ROOT / "docs",
        REPOSITORY_ROOT / "docs/coverage",
        output_dir,
    ):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CoverageEvidenceError(
                "cannot inspect coverage output path"
            ) from error
        if (
            candidate.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise CoverageEvidenceError(
                "coverage output path must contain only real directories"
            )
    try:
        relative = output_dir.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise CoverageEvidenceError(
            "coverage output directory must stay inside the repository"
        ) from error
    if relative.as_posix() != "docs/coverage/generated":
        raise CoverageEvidenceError(
            "coverage output directory must be docs/coverage/generated"
        )
    if set(generated) != set(ARTIFACT_NAMES):
        raise CoverageEvidenceError("coverage artifact set is incomplete")
    if output_dir.exists():
        existing_names = {path.name for path in output_dir.iterdir()}
        unexpected = sorted(existing_names.difference(ARTIFACT_NAMES))
        if unexpected:
            raise CoverageEvidenceError(
                "coverage output contains an unexpected entry: "
                f"{unexpected[0]}"
            )

    if check:
        stale: list[str] = []
        for name in ARTIFACT_NAMES:
            path = output_dir / name
            try:
                metadata = path.lstat()
                current = path.read_bytes()
            except OSError:
                stale.append(name)
                continue
            if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
                stale.append(name)
                continue
            if current != generated[name]:
                stale.append(name)
        if stale:
            raise CoverageEvidenceError(
                "coverage evidence is stale: " + ", ".join(stale)
            )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    publication_order = tuple(
        name for name in ARTIFACT_NAMES if name != "manifest.json"
    ) + ("manifest.json",)
    for name in publication_order:
        path = output_dir / name
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            metadata = None
        except OSError as error:
            raise CoverageEvidenceError(
                f"cannot inspect coverage artifact: {name}"
            ) from error
        if metadata is not None and (
            path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
        ):
            raise CoverageEvidenceError(
                f"coverage artifact path is unsafe: {name}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=output_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(generated[name])
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def record_coverage(arguments: argparse.Namespace) -> CoverageTrace:
    environment = _capture_environment()
    tools = identify_tools(
        arguments.cxx,
        arguments.gcov,
        arguments.lcov,
        arguments.geninfo,
        environment,
    )
    compiler_label = Path(arguments.cxx).name
    if arguments.cxx == tools["cxx"].label:
        compiler_label = tools["cxx"].label
    build_dir = _clean_coverage_build(compiler_label)
    build = _run(
        (
            "/usr/bin/make",
            "-s",
            f"-j{arguments.jobs}",
            f"CXX={tools['cxx'].executable}",
            "PROFILE=coverage",
            "test",
        ),
        label="coverage build and test",
        environment=environment,
        require_empty_stderr=True,
    )
    transcript, test_count = validate_test_transcript(
        build.stdout,
        jobs=arguments.jobs,
        compiler_label=tools["cxx"].label,
    )
    counters = _counter_inventory(build_dir)
    trace, _lcov_summary = _capture_trace(build_dir, tools, environment)
    source_snapshot = collect_source_snapshot()
    compiler_short_match = re.search(
        r"\b(\d+\.\d+\.\d+)\b", tools["cxx"].version
    )
    lcov_short_match = re.search(
        r"LCOV version ([0-9][A-Za-z0-9.+~-]*)", tools["lcov"].version
    )
    assert compiler_short_match is not None
    assert lcov_short_match is not None

    artifacts: dict[str, bytes] = {
        "coverage.info": trace.canonical_text.encode("utf-8"),
        "summary.svg": render_summary_svg(
            trace,
            test_count=test_count,
            compiler_short=f"GCC {compiler_short_match.group(1)}",
            lcov_short=f"LCOV {lcov_short_match.group(1)}",
            source_digest=str(source_snapshot["sha256"]),
        ).encode("utf-8"),
        "summary.txt": render_summary_text(
            trace,
            test_count=test_count,
            compiler_version=tools["cxx"].version,
            lcov_version=f"LCOV {lcov_short_match.group(1)}",
        ).encode("utf-8"),
        "test-run.txt": transcript.encode("utf-8"),
    }
    artifacts["manifest.json"] = _manifest(
        artifacts=artifacts,
        trace=trace,
        test_count=test_count,
        tools=tools,
        source_snapshot=source_snapshot,
        counters=counters,
        jobs=arguments.jobs,
    ).encode("utf-8")
    for name, payload in artifacts.items():
        reject_unsafe_public_text(name, payload.decode("utf-8"))
    _publish_or_check(
        DEFAULT_OUTPUT_DIR,
        artifacts,
        check=arguments.check,
    )
    return trace


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "run one clean GCC coverage suite and generate or byte-check the "
            "committed LCOV evidence bundle"
        )
    )
    parser.add_argument("--cxx", default="g++", help="GNU C++ compiler")
    parser.add_argument("--gcov", default="gcov", help="matching gcov binary")
    parser.add_argument("--lcov", default="lcov", help="LCOV 2.x binary")
    parser.add_argument(
        "--geninfo", default="geninfo", help="matching LCOV geninfo binary"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="parallel compilation jobs (1-16; default: 2)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="recapture and require byte-identical committed artifacts",
    )
    parsed = parser.parse_args(argv)
    if parsed.jobs < 1 or parsed.jobs > 16:
        parser.error("--jobs must be between 1 and 16")
    return parsed


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        trace = record_coverage(arguments)
    except (
        CoverageEvidenceError,
        OSError,
        subprocess.SubprocessError,
        UnicodeError,
    ) as error:
        print(f"coverage evidence error: {error}", file=sys.stderr)
        return 1
    action = "verified" if arguments.check else "recorded"
    print(
        f"{action} coverage evidence: "
        f"lines {_metric_text(trace.lines)}, "
        f"functions {_metric_text(trace.functions)}, "
        f"branches {_metric_text(trace.branches)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
