#!/usr/bin/env python3
"""Black-box contract checks for the built TensorKiln CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Final, Sequence


TIMEOUT_SECONDS: Final = 10
MAX_OUTPUT_BYTES: Final = 1024 * 1024
DOCUMENTED_INPUT_BITS: Final = (
    "x=0x3f800000,0x40000000,0x40400000,"
    "0xbf800000,0x3f000000,0x40800000"
)
DOCUMENTED_OUTPUT_BITS: Final = [
    "0x40900000",
    "0x41300000",
    "0x00000000",
    "0x41300000",
]
DENSE_STDOUT_RECEIPTS: Final = {
    "inspect_json": (
        1834,
        "bafba37b0c4ece4545ee011cdb904c070bcb456a5180ef1512b6b579340a8690",
    ),
    "inspect_text": (
        1459,
        "378aabfe5c1f8d999a92d537bae890a9479a214d835281918ff6f88918e9b2ec",
    ),
    "execute_json": (
        1167,
        "2d74429690aa514770d5a137ebf958f8b475a5b1cb4425a36ae7a4e42566b86b",
    ),
    "execute_text": (
        541,
        "4aa7da6ab1f27683b56e92496736c50a52c21a8838db0fc42ec518be9dbd2379",
    ),
}
REGLU_DOCUMENTED_OUTPUT_BITS: Final = [
    "0x00000000",
    "0x40a00000",
    "0x41480000",
    "0x40180000",
    "0x80000000",
    "0xc0f00000",
    "0x42040000",
    "0x00000000",
]
REGLU_SECOND_INPUT_BITS: Final = (
    "x=0x3e800000,0xc0000000,0x3fc00000,"
    "0x40400000,0xbf000000,0xbf800000"
)
REGLU_SECOND_OUTPUT_BITS: Final = [
    "0x00000000",
    "0x80000000",
    "0x405e0000",
    "0x00000000",
    "0x41b00000",
    "0x00000000",
    "0xc1480000",
    "0x41500000",
]


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


def require_receipt(payload: bytes, receipt: str) -> None:
    expected_bytes, expected_sha256 = DENSE_STDOUT_RECEIPTS[receipt]
    require(len(payload) == expected_bytes, f"{receipt} byte count differs")
    require(
        hashlib.sha256(payload).hexdigest() == expected_sha256,
        f"{receipt} SHA-256 differs",
    )


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
    listed_again = run(binary, ["list", "--format=json"])
    require(listed.returncode == 0, "list did not exit successfully")
    require(not listed.stderr, "list wrote to stderr")
    require(
        listed.stdout == listed_again.stdout
        and listed.stderr == listed_again.stderr
        and listed.returncode == listed_again.returncode,
        "workload list is not byte-replayable",
    )
    listing = parse_json(listed.stdout, "list stdout")
    require(
        isinstance(listing, dict)
        and listing.get("schema") == "tensorkiln.cli.workloads.v1",
        "list schema differs",
    )
    workloads = listing.get("workloads")
    require(
        isinstance(workloads, list)
        and [workload.get("id") for workload in workloads]
        == ["dense_relu_v1", "reglu_mlp_v1"],
        "workload registry differs",
    )
    require(
        workloads[1]
        == {
            "id": "reglu_mlp_v1",
            "kind": "compiled_in",
            "description": (
                "f32[2,3] -> dual MatMul+Add branches -> "
                "Relu(gate) * value -> f32[2,4]"
            ),
            "inputs": [
                {"name": "x", "dtype": "f32", "shape": [2, 3], "elements": 6}
            ],
            "outputs": [
                {
                    "name": "result",
                    "dtype": "f32",
                    "shape": [2, 4],
                    "elements": 8,
                }
            ],
        },
        "ReGLU-like workload descriptor differs",
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
    require_receipt(inspected.stdout, "inspect_json")
    inspect_text = run(
        binary,
        ["inspect", "--workload", "dense_relu_v1", "--format=text"],
    )
    require(inspect_text.returncode == 0, "text inspect failed")
    require(not inspect_text.stderr, "text inspect wrote to stderr")
    require_receipt(inspect_text.stdout, "inspect_text")
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

    execute_command = [
        "execute",
        "--workload",
        "dense_relu_v1",
        "--input-bits",
        DOCUMENTED_INPUT_BITS,
        "--format=json",
    ]
    executed = run(binary, execute_command)
    executed_again = run(binary, execute_command)
    require(executed.returncode == 0, "execute did not exit successfully")
    require(not executed.stderr, "execute wrote to stderr")
    require(
        executed.stdout == executed_again.stdout
        and executed.stderr == executed_again.stderr
        and executed.returncode == executed_again.returncode,
        "execute process result is not byte-replayable",
    )
    require_receipt(executed.stdout, "execute_json")
    execute_text = run(
        binary,
        [
            "execute",
            "--workload",
            "dense_relu_v1",
            "--input-bits",
            DOCUMENTED_INPUT_BITS,
            "--format=text",
        ],
    )
    require(execute_text.returncode == 0, "text execute failed")
    require(not execute_text.stderr, "text execute wrote to stderr")
    require_receipt(execute_text.stdout, "execute_text")
    execution_report = parse_json(executed.stdout, "execute stdout")
    require(
        isinstance(execution_report, dict)
        and set(execution_report) == {"schema", "workload", "plan", "execution"}
        and execution_report.get("schema") == "tensorkiln.cli.execute.v1",
        "execute envelope differs",
    )
    require(
        execution_report["workload"] == workloads[0],
        "execute workload differs from the registry",
    )
    execute_plan = execution_report["plan"]
    require(
        execute_plan
        == {
            "stats": plan["stats"],
            "kernels": plan["kernels"],
        },
        "execute plan differs from inspect",
    )
    execution = execution_report["execution"]
    require(
        isinstance(execution, dict)
        and set(execution)
        == {
            "run_status",
            "kernel_write_audit",
            "logical_workspace_bytes",
            "input",
            "outputs",
            "reference_check",
            "verification_scope",
            "benchmark",
        },
        "execution record fields differ",
    )
    require(
        execution["run_status"] == "success"
        and execution["kernel_write_audit"] is True
        and execution["logical_workspace_bytes"] == 128
        and execution["verification_scope"] == "this_workload_and_input_bits"
        and execution["benchmark"] is False,
        "execution guarantees differ",
    )
    require(
        execution["input"]
        == {
            "name": "x",
            "dtype": "f32",
            "shape": [2, 3],
            "bits": DOCUMENTED_INPUT_BITS.removeprefix("x=").split(","),
        },
        "execution input record differs",
    )
    require(
        execution["outputs"]
        == [
            {
                "name": "result",
                "dtype": "f32",
                "shape": [2, 2],
                "bits": DOCUMENTED_OUTPUT_BITS,
            }
        ],
        "execution output record differs",
    )
    require(
        execution["reference_check"]
        == {
            "comparison": "raw_f32_bits",
            "matched": 4,
            "total": 4,
            "status": "match",
        },
        "reference check differs",
    )
    checks += 1

    zero_bits = "x=" + ",".join(["0x00000000"] * 6)
    zero_execution = run(
        binary,
        [
            "execute",
            f"--input-bits={zero_bits}",
            "--format=json",
            "--workload=dense_relu_v1",
        ],
    )
    require(zero_execution.returncode == 0, "zero-input execute failed")
    require(not zero_execution.stderr, "zero-input execute wrote to stderr")
    zero_report = parse_json(zero_execution.stdout, "zero execute stdout")
    require(
        zero_report["execution"]["outputs"][0]["bits"]
        == [
            "0x3f000000",
            "0x00000000",
            "0x3f000000",
            "0x00000000",
        ]
        and zero_report["execution"]["outputs"][0]["bits"]
        != DOCUMENTED_OUTPUT_BITS,
        "zero-input execute output differs",
    )
    checks += 1

    reglu_inspect_command = [
        "inspect",
        "--workload=reglu_mlp_v1",
        "--format=json",
    ]
    reglu_inspected = run(binary, reglu_inspect_command)
    reglu_inspected_again = run(binary, reglu_inspect_command)
    require(reglu_inspected.returncode == 0, "ReGLU-like inspect failed")
    require(not reglu_inspected.stderr, "ReGLU-like inspect wrote to stderr")
    require(
        reglu_inspected.stdout == reglu_inspected_again.stdout
        and reglu_inspected.stderr == reglu_inspected_again.stderr
        and reglu_inspected.returncode == reglu_inspected_again.returncode,
        "ReGLU-like inspect is not byte-replayable",
    )
    reglu_inspect = parse_json(reglu_inspected.stdout, "ReGLU-like inspect")
    require(
        reglu_inspect["workload"] == workloads[1],
        "ReGLU-like inspect descriptor differs from list",
    )
    reglu_plan = reglu_inspect["plan"]
    require(
        reglu_plan["stats"]
        == {
            "values": 11,
            "inputs": 1,
            "constants": 4,
            "steps": 6,
            "outputs": 1,
            "constant_bytes": 128,
            "scalar_steps": 80,
            "workspace_bytes": 192,
        },
        "ReGLU-like compiled plan statistics differ",
    )
    require(
        [kernel["kind"] for kernel in reglu_plan["kernels"]]
        == [
            "matmul_rank2_f32",
            "add_broadcast_f32",
            "relu_contiguous_f32",
            "matmul_rank2_f32",
            "add_broadcast_f32",
            "mul_contiguous_f32",
        ],
        "ReGLU-like kernel sequence differs",
    )
    require(
        [kernel["scalar_steps"] for kernel in reglu_plan["kernels"]]
        == [24, 8, 8, 24, 8, 8]
        and "#o0 result -> %10" in reglu_plan["canonical_dump"],
        "ReGLU-like work accounting or output binding differs",
    )
    checks += 1

    reglu_execute_command = [
        "execute",
        "--workload",
        "reglu_mlp_v1",
        "--input-bits",
        DOCUMENTED_INPUT_BITS,
        "--format=json",
    ]
    reglu_executed = run(binary, reglu_execute_command)
    reglu_executed_again = run(binary, reglu_execute_command)
    require(reglu_executed.returncode == 0, "ReGLU-like execute failed")
    require(not reglu_executed.stderr, "ReGLU-like execute wrote to stderr")
    require(
        reglu_executed.stdout == reglu_executed_again.stdout
        and reglu_executed.stderr == reglu_executed_again.stderr
        and reglu_executed.returncode == reglu_executed_again.returncode,
        "ReGLU-like execute is not byte-replayable",
    )
    reglu_execution_report = parse_json(
        reglu_executed.stdout, "ReGLU-like execute"
    )
    require(
        reglu_execution_report["workload"] == workloads[1]
        and reglu_execution_report["plan"]
        == {
            "stats": reglu_plan["stats"],
            "kernels": reglu_plan["kernels"],
        },
        "ReGLU-like execute differs from list or inspect",
    )
    reglu_execution = reglu_execution_report["execution"]
    require(
        reglu_execution["outputs"]
        == [
            {
                "name": "result",
                "dtype": "f32",
                "shape": [2, 4],
                "bits": REGLU_DOCUMENTED_OUTPUT_BITS,
            }
        ],
        "ReGLU-like documented output bits differ",
    )
    require(
        reglu_execution["reference_check"]
        == {
            "comparison": "raw_f32_bits",
            "matched": 8,
            "total": 8,
            "status": "match",
        }
        and reglu_execution["logical_workspace_bytes"] == 192
        and reglu_execution["kernel_write_audit"] is True
        and reglu_execution["benchmark"] is False,
        "ReGLU-like execution guarantees differ",
    )
    checks += 1

    reglu_second = run(
        binary,
        [
            "execute",
            f"--input-bits={REGLU_SECOND_INPUT_BITS}",
            "--format=json",
            "--workload=reglu_mlp_v1",
        ],
    )
    require(reglu_second.returncode == 0, "second ReGLU-like execute failed")
    require(not reglu_second.stderr, "second ReGLU-like execute wrote to stderr")
    reglu_second_report = parse_json(
        reglu_second.stdout, "second ReGLU-like execute"
    )
    reglu_second_execution = reglu_second_report["execution"]
    require(
        reglu_second_execution["outputs"][0]["bits"] == REGLU_SECOND_OUTPUT_BITS
        and reglu_second_execution["outputs"][0]["bits"]
        != REGLU_DOCUMENTED_OUTPUT_BITS
        and reglu_second_execution["reference_check"]["matched"] == 8,
        "second ReGLU-like output or reference agreement differs",
    )
    checks += 1

    reglu_short = run(
        binary,
        [
            "execute",
            "--workload=reglu_mlp_v1",
            "--input-bits=x=0x00000000,0x00000000,0x00000000,"
            "0x00000000,0x00000000",
            "--format=json",
        ],
    )
    reglu_foreign = run(
        binary,
        [
            "execute",
            "--workload=reglu_mlp_v1",
            "--input-bits=y=0x00000000,0x00000000,0x00000000,"
            "0x00000000,0x00000000,0x00000000",
            "--format=json",
        ],
    )
    reglu_malformed = run(
        binary,
        [
            "execute",
            "--workload=reglu_mlp_v1",
            "--input-bits=x=0x00000000,0x00000000,0x0000000g,"
            "0x00000000,0x00000000,0x00000000",
            "--format=json",
        ],
    )
    require(
        reglu_short.returncode == 2
        and not reglu_short.stdout
        and parse_json(reglu_short.stderr, "short ReGLU-like input")["error"][
            "code"
        ]
        == "input_element_count_mismatch",
        "short ReGLU-like input boundary differs",
    )
    require(
        reglu_foreign.returncode == 2
        and not reglu_foreign.stdout
        and parse_json(reglu_foreign.stderr, "foreign ReGLU-like input")["error"][
            "code"
        ]
        == "input_binding_unknown",
        "foreign ReGLU-like binding boundary differs",
    )
    require(
        reglu_malformed.returncode == 2
        and not reglu_malformed.stdout
        and parse_json(reglu_malformed.stderr, "malformed ReGLU-like input")[
            "error"
        ]["code"]
        == "invalid_input_bits",
        "malformed ReGLU-like input boundary differs",
    )
    checks += 1

    malformed = run(
        binary,
        [
            "execute",
            "--workload=dense_relu_v1",
            "--input-bits=x=0x00000000,0x00000000,0x0000000g,"
            "0x00000000,0x00000000,0x00000000",
            "--format=json",
        ],
    )
    require(malformed.returncode == 2, "malformed input exit status differs")
    require(not malformed.stdout, "malformed input wrote to stdout")
    malformed_error = parse_json(malformed.stderr, "malformed input stderr")
    require(
        malformed_error["error"]["code"] == "invalid_input_bits",
        "malformed input error code differs",
    )
    checks += 1

    missing_input = run(
        binary,
        [
            "execute",
            "--workload=dense_relu_v1",
            "--format=json",
        ],
    )
    require(missing_input.returncode == 2, "missing input exit status differs")
    require(not missing_input.stdout, "missing input wrote to stdout")
    missing_error = parse_json(missing_input.stderr, "missing input stderr")
    require(
        missing_error["error"]["code"] == "missing_input_bits",
        "missing input error code differs",
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

    missing_workload = run(binary, ["inspect", "--format=json"])
    require(
        missing_workload.returncode == 2,
        "missing workload exit status differs",
    )
    require(not missing_workload.stdout, "missing workload wrote to stdout")
    missing_workload_error = parse_json(
        missing_workload.stderr, "missing workload stderr"
    )
    require(
        missing_workload_error
        == {
            "schema": "tensorkiln.cli.error.v1",
            "error": {
                "code": "missing_workload",
                "message": (
                    "inspect requires --workload ID (available: "
                    "dense_relu_v1, reglu_mlp_v1)"
                ),
            },
        },
        "missing workload registry guidance differs",
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
