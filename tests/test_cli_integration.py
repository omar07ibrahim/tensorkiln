#!/usr/bin/env python3
"""Black-box contract checks for the built TensorKiln CLI."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Final, Sequence


TIMEOUT_SECONDS: Final = 10
MAX_OUTPUT_BYTES: Final = 1024 * 1024


class CliContractError(RuntimeError):
    """Raised when the process-level CLI contract is violated."""


def run(binary: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
    completed = subprocess.run(
        [str(binary), *arguments],
        cwd=binary.parent,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if (
        len(completed.stdout) > MAX_OUTPUT_BYTES
        or len(completed.stderr) > MAX_OUTPUT_BYTES
    ):
        raise CliContractError("CLI output exceeded the integration limit")
    return completed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CliContractError(message)


def decode_ascii(payload: bytes, label: str) -> str:
    try:
        return payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise CliContractError(f"{label} is not ASCII") from error


def parse_json(payload: bytes, label: str) -> object:
    text = decode_ascii(payload, label)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise CliContractError(f"{label} is not valid JSON") from error


def verify(binary: Path) -> int:
    checks = 0

    help_result = run(binary, ["--help"])
    require(help_result.returncode == 0, "--help did not exit successfully")
    require(not help_result.stderr, "--help wrote to stderr")
    require(
        b"not a graph dump parser" in help_result.stdout,
        "--help omitted the parser boundary",
    )
    checks += 1

    listed = run(binary, ["list", "--format=json"])
    require(listed.returncode == 0, "list did not exit successfully")
    require(not listed.stderr, "list wrote to stderr")
    listing = parse_json(listed.stdout, "list stdout")
    require(
        isinstance(listing, dict)
        and listing.get("schema") == "tensorkiln.cli.workloads.v1",
        "list schema differs",
    )
    workloads = listing.get("workloads")
    require(
        isinstance(workloads, list)
        and len(workloads) == 1
        and workloads[0].get("id") == "dense_relu_v1",
        "workload registry differs",
    )
    checks += 1

    command = [
        "inspect",
        "--workload",
        "dense_relu_v1",
        "--format=json",
    ]
    inspected = run(binary, command)
    replayed = run(binary, command)
    require(inspected.returncode == 0, "inspect did not exit successfully")
    require(not inspected.stderr, "inspect wrote to stderr")
    require(
        inspected.stdout == replayed.stdout
        and inspected.stderr == replayed.stderr
        and replayed.returncode == inspected.returncode,
        "inspect process result is not byte-replayable",
    )
    report = parse_json(inspected.stdout, "inspect stdout")
    require(
        isinstance(report, dict)
        and set(report) == {"schema", "workload", "plan"}
        and report.get("schema") == "tensorkiln.cli.inspect.v1",
        "inspect envelope differs",
    )
    checks += 1

    plan = report["plan"]
    require(
        plan["stats"]
        == {
            "values": 6,
            "inputs": 1,
            "constants": 2,
            "steps": 3,
            "outputs": 1,
            "constant_bytes": 32,
            "scalar_steps": 20,
            "workspace_bytes": 128,
        },
        "compiled plan statistics differ",
    )
    require(
        [kernel["kind"] for kernel in plan["kernels"]]
        == [
            "matmul_rank2_f32",
            "add_broadcast_f32",
            "relu_contiguous_f32",
        ],
        "compiled kernel sequence differs",
    )
    require(
        plan["canonical_dump"].startswith("tensorkiln.execution_plan v0 {\n")
        and "#b2 offset=0 payload=16 reserved=64 live=[2,3)" in
        plan["canonical_dump"],
        "canonical plan dump differs",
    )
    checks += 1

    unknown = run(
        binary,
        [
            "inspect",
            "--workload=missing",
            "--format",
            "json",
        ],
    )
    require(unknown.returncode == 2, "unknown workload exit status differs")
    require(not unknown.stdout, "unknown workload wrote to stdout")
    error = parse_json(unknown.stderr, "error stderr")
    require(
        error
        == {
            "schema": "tensorkiln.cli.error.v1",
            "error": {
                "code": "unknown_workload",
                "message": "unknown workload 'missing'",
            },
        },
        "structured error differs",
    )
    checks += 1

    print(f"CLI integration: {checks}/{checks} checks passed")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    arguments = parser.parse_args()
    supplied_binary = arguments.binary
    try:
        metadata = supplied_binary.lstat()
    except OSError:
        parser.error("--binary must be a real executable file")
    if (
        supplied_binary.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(supplied_binary, os.X_OK)
    ):
        parser.error("--binary must be a real executable file")
    arguments.binary = supplied_binary.resolve()
    return arguments


def main() -> int:
    try:
        arguments = parse_arguments()
        return verify(arguments.binary)
    except (CliContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CLI integration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
