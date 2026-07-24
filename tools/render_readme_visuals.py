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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_DIR: Final = Path("build/g++/release")
DEFAULT_OUTPUT_DIR: Final = Path("docs/visuals/generated")
MAX_OUTPUT_BYTES: Final = 1024 * 1024
EXAMPLE_TIMEOUT_SECONDS: Final = 30

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
        completed = subprocess.run(
            [str(binary)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise VisualEvidenceError(
            f"{binary_name} exceeded {EXAMPLE_TIMEOUT_SECONDS} seconds"
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


def render_visuals(build_dir: Path) -> dict[str, str]:
    """Collect verified stdout and return deterministic evidence artifacts."""

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


def check_visuals(output_dir: Path, generated: dict[str, str]) -> int:
    """Return nonzero if committed visuals are absent or stale."""

    stale = False
    for filename, text in generated.items():
        path = output_dir / filename
        try:
            current = path.read_bytes()
        except FileNotFoundError:
            print(f"visuals: missing {path.relative_to(REPOSITORY_ROOT)}")
            stale = True
            continue
        except OSError as error:
            raise VisualEvidenceError(f"cannot read {path}: {error}") from error

        expected = text.encode("utf-8")
        if current != expected:
            current_digest = hashlib.sha256(current).hexdigest()[:12]
            expected_digest = hashlib.sha256(expected).hexdigest()[:12]
            print(
                f"visuals: stale {path.relative_to(REPOSITORY_ROOT)} "
                f"({current_digest} != {expected_digest})"
            )
            stale = True
    return 1 if stale else 0


def write_visuals(output_dir: Path, generated: dict[str, str]) -> None:
    """Write generated evidence artifacts atomically."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in generated.items():
        path = output_dir / filename
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8", newline="\n")
            temporary.chmod(0o644)
            os.replace(temporary, path)
        except OSError as error:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise VisualEvidenceError(f"cannot write {path}: {error}") from error
        print(f"visuals: wrote {path.relative_to(REPOSITORY_ROOT)}")


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
            f"execute_graph (default: {DEFAULT_BUILD_DIR})"
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    try:
        build_dir = resolve_build_dir(arguments.build_dir)
        output_dir = resolve_output_dir(arguments.output_dir)
        generated = render_visuals(build_dir)
        if arguments.check:
            return check_visuals(output_dir, generated)
        write_visuals(output_dir, generated)
        return 0
    except VisualEvidenceError as error:
        print(f"visuals: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
