"""Pure unit tests for the README visual evidence renderer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import render_readme_visuals as visuals  # noqa: E402


PLAN_STDOUT = """\
=== verified interval arena plan ===
tensorkiln.arena_plan v0 {
  alignment_bytes=64
  limits {buffers=4096, workspace_bytes=268435456}
  stats {buffers=4, payload_bytes=272, reserved_bytes=384, peak_live_reserved_bytes=192, workspace_bytes=192}
  #b0 offset=0 payload=96 reserved=128 live=[0,2)
  #b1 offset=128 payload=64 reserved=64 live=[0,1)
  #b2 offset=128 payload=32 reserved=64 live=[1,3)
  #b3 offset=0 payload=80 reserved=128 live=[2,4)
}
naive_separate_reservations_bytes=384
reused_workspace_bytes=192
verified: two boundary reuses, 192 bytes of workspace for 384 bytes of aligned reservations
"""

EXECUTE_STDOUT = """\
=== verified dense execution plan ===
tensorkiln.execution_plan v0 {
  stats {values=6, steps=3, workspace_bytes=128}
}
result = [4.5, 11, 0, 11]
verified: audited execution matches the independent reference bit for bit
"""

SOFTMAX_STDOUT = """\
=== verified Softmax execution ===
scope: deterministic correctness example; not a benchmark
plan {shape=f32[5,4], axis=1, kernel=softmax_last_axis_f32, scalar_steps=60, workspace_bytes=128, audited=true}
slice finite_equal bits=[0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000]
slice nan_precedence bits=[0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]
slice positive_infinity_split bits=[0x3f000000, 0x00000000, 0x3f000000, 0x00000000]
slice all_negative_infinity bits=[0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]
slice mixed_negative_infinity bits=[0x00000000, 0x3f000000, 0x3f000000, 0x00000000]
agreement {executor_reference_bits=20/20, executor_fixture_bits=20/20}
=== optimized axis boundary ===
reference_axis0 {status=accepted, scalar_steps=80}
optimized_axis0 {code=plan_operation_unsupported, message=plan backend does not support softmax axis 0 at #n1}
verified: last-axis execution and reference agree; valid axis 0 remains reference-only
"""

CLI_INSPECT_STDOUT = (
    json.dumps(
        {
            "schema": "tensorkiln.cli.inspect.v1",
            "workload": visuals.CLI_WORKLOAD,
            "plan": {
                "stats": visuals.CLI_PLAN_STATS,
                "kernels": list(visuals.CLI_KERNELS),
                "canonical_dump": visuals.CLI_CANONICAL_DUMP,
            },
        },
        separators=(",", ":"),
    )
    + "\n"
)
CLI_EXECUTE_STDOUT = (
    json.dumps(
        {
            "schema": "tensorkiln.cli.execute.v1",
            "workload": visuals.CLI_WORKLOAD,
            "plan": {
                "stats": visuals.CLI_PLAN_STATS,
                "kernels": list(visuals.CLI_KERNELS),
            },
            "execution": {
                "run_status": "success",
                "kernel_write_audit": True,
                "logical_workspace_bytes": 128,
                "input": {
                    "name": "x",
                    "dtype": "f32",
                    "shape": [2, 3],
                    "bits": list(visuals.CLI_INPUT_BITS),
                },
                "outputs": [
                    {
                        "name": "result",
                        "dtype": "f32",
                        "shape": [2, 2],
                        "bits": list(visuals.CLI_OUTPUT_BITS),
                    }
                ],
                "reference_check": {
                    "comparison": "raw_f32_bits",
                    "matched": 4,
                    "total": 4,
                    "status": "match",
                },
                "verification_scope": "this_workload_and_input_bits",
                "benchmark": False,
            },
        },
        separators=(",", ":"),
    )
    + "\n"
)

REGLU_WORKLOADS_STDOUT = (
    json.dumps(
        {
            "schema": "tensorkiln.cli.workloads.v1",
            "workloads": [visuals.CLI_WORKLOAD, visuals.REGLU_WORKLOAD],
        },
        separators=(",", ":"),
    )
    + "\n"
)
REGLU_INSPECT_STDOUT = (
    json.dumps(
        {
            "schema": "tensorkiln.cli.inspect.v1",
            "workload": visuals.REGLU_WORKLOAD,
            "plan": {
                "stats": visuals.REGLU_PLAN_STATS,
                "kernels": list(visuals.REGLU_KERNELS),
                "canonical_dump": visuals.REGLU_CANONICAL_DUMP,
            },
        },
        separators=(",", ":"),
    )
    + "\n"
)
REGLU_EXECUTE_STDOUT = (
    json.dumps(
        {
            "schema": "tensorkiln.cli.execute.v1",
            "workload": visuals.REGLU_WORKLOAD,
            "plan": {
                "stats": visuals.REGLU_PLAN_STATS,
                "kernels": list(visuals.REGLU_KERNELS),
            },
            "execution": {
                "run_status": "success",
                "kernel_write_audit": True,
                "logical_workspace_bytes": 192,
                "input": {
                    "name": "x",
                    "dtype": "f32",
                    "shape": [2, 3],
                    "bits": list(visuals.REGLU_INPUT_BITS),
                },
                "outputs": [
                    {
                        "name": "result",
                        "dtype": "f32",
                        "shape": [2, 4],
                        "bits": list(visuals.REGLU_OUTPUT_BITS),
                    }
                ],
                "reference_check": {
                    "comparison": "raw_f32_bits",
                    "matched": 8,
                    "total": 8,
                    "status": "match",
                },
                "verification_scope": "this_workload_and_input_bits",
                "benchmark": False,
            },
        },
        separators=(",", ":"),
    )
    + "\n"
)
REGLU_LIST_TEXT_STDOUT = visuals._reglu_list_text()
REGLU_INSPECT_TEXT_STDOUT = visuals._reglu_inspect_text()
REGLU_EXECUTE_TEXT_STDOUT = visuals._reglu_execute_text()
PUBLISHED_V3_SHA256 = {
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


def validated_reglu_evidence() -> visuals.RegluEvidence:
    return visuals.validate_reglu_evidence(
        REGLU_WORKLOADS_STDOUT,
        REGLU_INSPECT_STDOUT,
        REGLU_EXECUTE_STDOUT,
        REGLU_LIST_TEXT_STDOUT,
        REGLU_INSPECT_TEXT_STDOUT,
        REGLU_EXECUTE_TEXT_STDOUT,
    )


class ArenaEvidenceTests(unittest.TestCase):
    def test_parser_cross_checks_source_values(self) -> None:
        evidence = visuals.parse_arena_evidence(PLAN_STDOUT)

        self.assertEqual(evidence.workspace, 192)
        self.assertEqual(evidence.total_reserved, 384)
        self.assertEqual(
            [
                (
                    item.ordinal,
                    item.offset,
                    item.reserved,
                    item.live_start,
                    item.live_end,
                )
                for item in evidence.allocations
            ],
            [
                (0, 0, 128, 0, 2),
                (1, 128, 64, 0, 1),
                (2, 128, 64, 1, 3),
                (3, 0, 128, 2, 4),
            ],
        )

    def test_arena_svg_is_deterministic_and_data_derived(self) -> None:
        first = visuals.render_arena_reuse_svg(PLAN_STDOUT)
        second = visuals.render_arena_reuse_svg(PLAN_STDOUT)

        self.assertEqual(first, second)
        self.assertIn("Aligned reservations", first)
        self.assertIn("384 B", first)
        self.assertIn("Reused workspace", first)
        self.assertIn("bytes 0–128", first)
        self.assertIn("#b3 [2,4) · 128 B", first)
        self.assertIn("NOT A BENCHMARK", first)
        ElementTree.fromstring(first)

    def test_parser_rejects_overlapping_live_allocations(self) -> None:
        unsafe = PLAN_STDOUT.replace(
            "#b3 offset=0 payload=80 reserved=128 live=[2,4)",
            "#b3 offset=0 payload=80 reserved=128 live=[1,4)",
        )

        with self.assertRaisesRegex(
            visuals.VisualEvidenceError, "overlap while live"
        ):
            visuals.parse_arena_evidence(unsafe)

    def test_parser_rejects_noncanonical_reservation_rounding(self) -> None:
        unsafe = PLAN_STDOUT.replace(
            "#b0 offset=0 payload=96 reserved=128 live=[0,2)",
            "#b0 offset=0 payload=96 reserved=126 live=[0,2)",
        ).replace(
            "#b3 offset=0 payload=80 reserved=128 live=[2,4)",
            "#b3 offset=0 payload=80 reserved=130 live=[2,4)",
        )

        with self.assertRaisesRegex(
            visuals.VisualEvidenceError, "alignment-rounded payload"
        ):
            visuals.parse_arena_evidence(unsafe)

    def test_parser_recomputes_workspace_extent_and_live_peak(self) -> None:
        wrong_workspace = PLAN_STDOUT.replace(
            "peak_live_reserved_bytes=192, workspace_bytes=192",
            "peak_live_reserved_bytes=256, workspace_bytes=256",
        ).replace(
            "reused_workspace_bytes=192",
            "reused_workspace_bytes=256",
        )
        with self.assertRaisesRegex(
            visuals.VisualEvidenceError, "reported workspace"
        ):
            visuals.parse_arena_evidence(wrong_workspace)

        wrong_peak = PLAN_STDOUT.replace(
            "peak_live_reserved_bytes=192",
            "peak_live_reserved_bytes=191",
        )
        with self.assertRaisesRegex(
            visuals.VisualEvidenceError, "reported peak live bytes"
        ):
            visuals.parse_arena_evidence(wrong_peak)

    def test_parser_rejects_an_empty_sentinel_bearing_plan(self) -> None:
        unsafe_lines = [
            line
            for line in PLAN_STDOUT.splitlines()
            if not line.startswith("  #b")
        ]
        unsafe = (
            "\n".join(unsafe_lines)
            .replace(
                (
                    "stats {buffers=4, payload_bytes=272, "
                    "reserved_bytes=384, peak_live_reserved_bytes=192, "
                    "workspace_bytes=192}"
                ),
                (
                    "stats {buffers=0, payload_bytes=0, "
                    "reserved_bytes=0, peak_live_reserved_bytes=0, "
                    "workspace_bytes=1}"
                ),
            )
            .replace(
                "naive_separate_reservations_bytes=384",
                "naive_separate_reservations_bytes=0",
            )
            .replace(
                "reused_workspace_bytes=192",
                "reused_workspace_bytes=1",
            )
            + "\n"
        )

        with self.assertRaisesRegex(
            visuals.VisualEvidenceError, "at least one allocation"
        ):
            visuals.parse_arena_evidence(unsafe)

    def test_parser_binds_verification_summary_to_evidence(self) -> None:
        wrong_bytes = PLAN_STDOUT.replace(
            "192 bytes of workspace for 384 bytes",
            "191 bytes of workspace for 383 bytes",
        )
        with self.assertRaisesRegex(
            visuals.VisualEvidenceError,
            "summary differs from parsed arena evidence",
        ):
            visuals.parse_arena_evidence(wrong_bytes)

        wrong_count = PLAN_STDOUT.replace(
            "two boundary reuses",
            "one boundary reuse",
        )
        with self.assertRaisesRegex(
            visuals.VisualEvidenceError,
            "summary differs from parsed arena evidence",
        ):
            visuals.parse_arena_evidence(wrong_count)


class TranscriptTests(unittest.TestCase):
    def test_terminal_svg_is_deterministic_and_preserves_stdout(self) -> None:
        first = visuals.render_execute_graph_svg(EXECUTE_STDOUT)
        second = visuals.render_execute_graph_svg(EXECUTE_STDOUT)

        self.assertEqual(first, second)
        self.assertIn("$ &lt;release-build&gt;/execute_graph", first)
        self.assertIn("result = [4.5, 11, 0, 11]", first)
        self.assertIn("independent reference bit for bit", first)
        self.assertIn("NOT A BENCHMARK", first)
        ElementTree.fromstring(first)

    def test_validation_rejects_home_paths_and_secret_markers(self) -> None:
        for unsafe_line in (
            "artifact=/home/someone/private.txt",
            "artifact=/root/.ssh/id_rsa",
            "artifact=/Users/someone/private.txt",
            r"artifact=C:\Users\someone\private.txt",
            "owner=someone@example.com",
            "artifact=file:///workspace/private.txt",
            "AWS_SECRET_ACCESS_KEY=not-a-real-value",
            "Authorization: Bearer abcdefghijklmnop",
            "token=not-a-real-token",
            "-----BEGIN PRIVATE KEY-----",
        ):
            with self.subTest(unsafe_line=unsafe_line):
                stdout = EXECUTE_STDOUT.replace(
                    "result = [4.5, 11, 0, 11]",
                    f"result = [4.5, 11, 0, 11]\n{unsafe_line}",
                )
                with self.assertRaises(visuals.VisualEvidenceError):
                    visuals.render_execute_graph_svg(stdout)

        for unsafe_character in ("\ufffe", "\u202e"):
            with self.subTest(unsafe_character=repr(unsafe_character)):
                stdout = EXECUTE_STDOUT.replace(
                    "result = [4.5, 11, 0, 11]",
                    f"result = [4.5, 11, 0, 11]{unsafe_character}",
                )
                with self.assertRaisesRegex(
                    visuals.VisualEvidenceError, "ASCII evidence"
                ):
                    visuals.render_execute_graph_svg(stdout)

    def test_xml_metacharacters_are_escaped(self) -> None:
        stdout = EXECUTE_STDOUT.replace(
            "tensorkiln.execution_plan v0 {",
            "tensorkiln.execution_plan v0 { <verified & escaped>",
        )
        rendered = visuals.render_execute_graph_svg(stdout)

        self.assertIn("&lt;verified &amp; escaped&gt;", rendered)
        self.assertNotIn("<verified & escaped>", rendered)

    def test_softmax_terminal_svg_preserves_the_complete_fixture_report(
        self,
    ) -> None:
        first = visuals.render_execute_softmax_svg(SOFTMAX_STDOUT)
        second = visuals.render_execute_softmax_svg(SOFTMAX_STDOUT)

        self.assertEqual(first, second)
        self.assertIn("$ &lt;release-build&gt;/execute_softmax", first)
        self.assertIn("FIVE EXACT SLICES", first)
        self.assertIn("executor_reference_bits=20/20", first)
        self.assertIn("scalar_steps=60", first)
        self.assertIn("scalar_steps=80", first)
        self.assertIn("remains reference-only", first)
        self.assertIn("NOT A BENCHMARK", first)
        self.assertIn("fixture-scoped bits", first)
        ElementTree.fromstring(first)

    def test_softmax_capture_rejects_mutation_or_additional_output(self) -> None:
        cases = (
            SOFTMAX_STDOUT.replace("0x3e800000", "0x3e800001", 1),
            SOFTMAX_STDOUT.replace(
                "executor_fixture_bits=20/20",
                "executor_fixture_bits=19/20",
            ),
            SOFTMAX_STDOUT.replace(
                "scalar_steps=80", "scalar_steps=79"
            ),
            SOFTMAX_STDOUT
            + "claim: arbitrary libm output is bit-identical\n",
        )
        for unsafe in cases:
            with self.subTest(
                unsafe_sha256=hashlib.sha256(unsafe.encode()).hexdigest()
            ):
                with self.assertRaises(visuals.VisualEvidenceError):
                    visuals.render_execute_softmax_svg(unsafe)

    def test_legacy_bundle_remains_available_during_v2_publication(
        self,
    ) -> None:
        def fake_example(
            _build_dir: Path,
            binary_name: str,
            _sentinels: tuple[str, ...],
        ) -> str:
            if binary_name == "plan_arena":
                return PLAN_STDOUT
            if binary_name == "execute_graph":
                return EXECUTE_STDOUT
            raise AssertionError(f"unexpected binary: {binary_name}")

        with mock.patch.object(
            visuals, "run_release_example", side_effect=fake_example
        ):
            artifacts = visuals.render_legacy_visuals(
                Path("unused-release-dir")
            )

        self.assertEqual(
            set(artifacts),
            {
                "arena-plan.txt",
                "arena-reuse.svg",
                "execute-graph.svg",
                "execute-graph.txt",
                "manifest.json",
            },
        )
        manifest = json.loads(artifacts["manifest.json"])
        self.assertEqual(
            manifest["schema"], "tensorkiln.readme-visual-evidence.v1"
        )
        self.assertEqual(
            manifest["generator"], "tools/render_readme_visuals.py"
        )

    def test_evidence_bundle_binds_transcripts_and_visuals(self) -> None:
        def fake_example(
            _build_dir: Path,
            binary_name: str,
            _sentinels: tuple[str, ...],
        ) -> str:
            if binary_name == "plan_arena":
                return PLAN_STDOUT
            if binary_name == "execute_graph":
                return EXECUTE_STDOUT
            if binary_name == "execute_softmax":
                return SOFTMAX_STDOUT
            raise AssertionError(f"unexpected binary: {binary_name}")

        source_provenance = {
            "commit": "1" * 40,
            "object_format": "sha1",
            "selection": "test source selection",
            "source_files": {
                "examples/execute_softmax.cpp": {
                    "bytes": 1,
                    "git_blob": "2" * 40,
                    "mode": "100644",
                    "sha256": "3" * 64,
                }
            },
            "tree": "4" * 40,
        }
        binary_provenance = {
            "execute_graph": {"bytes": 101, "sha256": "5" * 64},
            "execute_softmax": {"bytes": 102, "sha256": "6" * 64},
            "plan_arena": {"bytes": 103, "sha256": "7" * 64},
        }
        generator_provenance = {
            "bytes": 104,
            "commit": "8" * 40,
            "committed": True,
            "git_blob": "9" * 40,
            "path": "tools/render_readme_visuals.py",
            "sha256": "a" * 64,
            "tree": "b" * 40,
        }
        with mock.patch.object(
            visuals, "run_release_example", side_effect=fake_example
        ), mock.patch.object(
            visuals,
            "collect_source_provenance",
            return_value=source_provenance,
        ), mock.patch.object(
            visuals,
            "collect_binary_provenance",
            return_value=binary_provenance,
        ), mock.patch.object(
            visuals,
            "collect_generator_provenance",
            return_value=generator_provenance,
        ):
            artifacts = visuals.render_visuals(
                Path("unused-release-dir"), include_softmax=True
            )

        self.assertEqual(artifacts["arena-plan.txt"], PLAN_STDOUT)
        self.assertEqual(artifacts["execute-graph.txt"], EXECUTE_STDOUT)
        self.assertEqual(artifacts["execute-softmax.txt"], SOFTMAX_STDOUT)
        manifest = json.loads(artifacts["manifest.json"])
        self.assertEqual(
            manifest["schema"], "tensorkiln.readme-visual-evidence.v2"
        )
        self.assertEqual(
            manifest["repository_source"], source_provenance
        )
        self.assertEqual(manifest["generator"], generator_provenance)
        self.assertEqual(
            manifest["sources"]["execute_graph"]["stdout_sha256"],
            hashlib.sha256(EXECUTE_STDOUT.encode()).hexdigest(),
        )
        self.assertEqual(
            manifest["sources"]["execute_softmax"]["stdout_sha256"],
            hashlib.sha256(SOFTMAX_STDOUT.encode()).hexdigest(),
        )
        self.assertEqual(
            manifest["sources"]["execute_softmax"]["binary_sha256"],
            "6" * 64,
        )
        self.assertEqual(
            manifest["capture_contract"]["network_isolation"],
            "not claimed",
        )
        self.assertTrue(
            any(
                "not arbitrary finite inputs or libm" in claim
                for claim in manifest["claim_boundary"]
            )
        )
        for filename, record in manifest["artifacts"].items():
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(artifacts[filename].encode()).hexdigest(),
            )
        self.assertNotIn("/home/", artifacts["manifest.json"])


class BoundedProcessCaptureTests(unittest.TestCase):
    ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}

    def capture(
        self,
        script: str,
        *,
        maximum_output_bytes: int,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[bytes]:
        return visuals._run_bounded_process(
            [sys.executable, "-I", "-c", script],
            environment=self.ENVIRONMENT,
            timeout_seconds=timeout_seconds,
            maximum_output_bytes=maximum_output_bytes,
        )

    def parse_process_identity(self, payload: bytes) -> tuple[int, int]:
        fields = payload.strip().split()
        self.assertEqual(len(fields), 2)
        process_id, starttime = (int(field) for field in fields)
        self.assertGreater(process_id, 0)
        self.assertGreater(starttime, 0)
        return process_id, starttime

    def read_process_identity(
        self, process_id: int
    ) -> tuple[str, int] | None:
        try:
            stat = Path(f"/proc/{process_id}/stat").read_text(
                encoding="ascii"
            )
        except (FileNotFoundError, ProcessLookupError):
            return None
        stat_fields = stat.rsplit(") ", maxsplit=1)[1].split()
        return stat_fields[0], int(stat_fields[19])

    def assert_process_terminated(self, identity: tuple[int, int]) -> None:
        process_id, expected_starttime = identity
        reap_deadline = time.monotonic() + 1.0
        while time.monotonic() < reap_deadline:
            current = self.read_process_identity(process_id)
            if current is None or current[1] != expected_starttime:
                return
            state, _starttime = current
            if state in {"X", "Z"}:
                return
            time.sleep(0.01)

        try:
            process_fd = os.pidfd_open(process_id)
        except ProcessLookupError:
            return
        try:
            current = self.read_process_identity(process_id)
            if current is None or current[1] != expected_starttime:
                return
            try:
                signal.pidfd_send_signal(process_fd, signal.SIGKILL)
            except ProcessLookupError:
                return
        finally:
            os.close(process_fd)
        self.fail(
            "bounded helper left its descendant running "
            f"(pid={process_id}, starttime={expected_starttime})"
        )

    def test_exact_limit_is_accepted_and_limit_plus_one_is_killed(self) -> None:
        exact_script = """\
import os
payload = b"x" * 4096
while payload:
    payload = payload[os.write(1, payload):]
"""
        completed = self.capture(
            exact_script,
            maximum_output_bytes=4096,
            timeout_seconds=2.0,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"x" * 4096)
        self.assertEqual(completed.stderr, b"")

        overflow_script = """\
import os
import time
payload = b"x" * 4097
while payload:
    payload = payload[os.write(1, payload):]
time.sleep(10)
"""
        started = time.monotonic()
        with self.assertRaises(
            visuals._ProcessOutputLimitExceeded
        ) as raised:
            self.capture(
                overflow_script,
                maximum_output_bytes=4096,
                timeout_seconds=2.0,
            )
        self.assertEqual(raised.exception.stream, "stdout")
        self.assertEqual(raised.exception.maximum_bytes, 4096)
        self.assertLess(time.monotonic() - started, 1.5)

    def test_stdout_and_stderr_are_drained_concurrently(self) -> None:
        script = """\
import fcntl
import os

count = fcntl.fcntl(1, fcntl.F_GETPIPE_SZ) + 4096
for descriptor, byte in ((1, b"o"), (2, b"e")):
    payload = byte * count
    while payload:
        payload = payload[os.write(descriptor, payload):]
"""
        completed = self.capture(
            script,
            maximum_output_bytes=2 * 1024 * 1024,
            timeout_seconds=2.0,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(len(completed.stdout), len(completed.stderr))
        self.assertGreater(len(completed.stdout), 4096)
        self.assertEqual(set(completed.stdout), {ord("o")})
        self.assertEqual(set(completed.stderr), {ord("e")})

    def test_timeout_covers_closed_pipes_and_grandchild_inherited_pipes(
        self,
    ) -> None:
        closed_pipes = """\
import os
import time
os.close(1)
os.close(2)
time.sleep(10)
"""
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            self.capture(
                closed_pipes,
                maximum_output_bytes=4096,
                timeout_seconds=0.5,
            )
        self.assertLess(time.monotonic() - started, 1.5)

        inherited_pipes = """\
import subprocess
import sys

child = subprocess.Popen(
    [sys.executable, "-I", "-c", "import time; time.sleep(10)"]
)
stat = open(f"/proc/{child.pid}/stat", encoding="ascii").read()
starttime = stat.rsplit(") ", 1)[1].split()[19]
print(child.pid, starttime, flush=True)
"""
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            self.capture(
                inherited_pipes,
                maximum_output_bytes=4096,
                timeout_seconds=0.5,
            )
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIsInstance(raised.exception.output, bytes)
        grandchild_identity = self.parse_process_identity(
            raised.exception.output
        )
        self.assert_process_terminated(grandchild_identity)

    def test_successful_parent_cannot_leave_devnull_child_running(self) -> None:
        detached_output_child = """\
import subprocess
import sys

child = subprocess.Popen(
    [sys.executable, "-I", "-c", "import time; time.sleep(10)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
stat = open(f"/proc/{child.pid}/stat", encoding="ascii").read()
starttime = stat.rsplit(") ", 1)[1].split()[19]
print(child.pid, starttime, flush=True)
"""
        completed = self.capture(
            detached_output_child,
            maximum_output_bytes=4096,
            timeout_seconds=2.0,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stderr, b"")
        child_identity = self.parse_process_identity(completed.stdout)
        self.assert_process_terminated(child_identity)

    def test_completed_parent_preserves_exit_7_and_kills_devnull_child(
        self,
    ) -> None:
        detached_output_child = """\
import subprocess
import sys

child = subprocess.Popen(
    [sys.executable, "-I", "-c", "import time; time.sleep(10)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
stat = open(f"/proc/{child.pid}/stat", encoding="ascii").read()
starttime = stat.rsplit(") ", 1)[1].split()[19]
print(child.pid, starttime, flush=True)
raise SystemExit(7)
"""
        completed = self.capture(
            detached_output_child,
            maximum_output_bytes=4096,
            timeout_seconds=2.0,
        )
        self.assertEqual(completed.returncode, 7)
        self.assertEqual(completed.stderr, b"")
        child_identity = self.parse_process_identity(completed.stdout)
        self.assert_process_terminated(child_identity)

    def test_selector_construction_failure_kills_and_closes_process(self) -> None:
        original_terminate = visuals._terminate_process_group
        terminated: list[subprocess.Popen[bytes]] = []

        def terminate(process: subprocess.Popen[bytes]) -> None:
            terminated.append(process)
            original_terminate(process)

        with mock.patch.object(
            visuals.selectors,
            "DefaultSelector",
            side_effect=RuntimeError("selector setup failed"),
        ), mock.patch.object(
            visuals,
            "_terminate_process_group",
            side_effect=terminate,
        ) as terminate_process:
            with self.assertRaisesRegex(RuntimeError, "selector setup failed"):
                self.capture(
                    "import time; time.sleep(10)",
                    maximum_output_bytes=4096,
                    timeout_seconds=2.0,
                )

        terminate_process.assert_called_once()
        self.assertEqual(len(terminated), 1)
        process = terminated[0]
        self.assertIsNotNone(process.returncode)
        self.assertIsNotNone(process.stdout)
        self.assertIsNotNone(process.stderr)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_termination_failure_cannot_skip_selector_or_pipe_closure(
        self,
    ) -> None:
        original_selector = visuals.selectors.DefaultSelector()
        selector = mock.Mock(wraps=original_selector)
        original_terminate = visuals._terminate_process_group
        terminated: list[subprocess.Popen[bytes]] = []

        def terminate_then_fail(process: subprocess.Popen[bytes]) -> None:
            terminated.append(process)
            original_terminate(process)
            raise RuntimeError("termination cleanup failed")

        closed_pipes = """\
import os
import time
os.close(1)
os.close(2)
time.sleep(10)
"""
        with mock.patch.object(
            visuals.selectors,
            "DefaultSelector",
            return_value=selector,
        ), mock.patch.object(
            visuals,
            "_terminate_process_group",
            side_effect=terminate_then_fail,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "termination cleanup failed"
            ):
                self.capture(
                    closed_pipes,
                    maximum_output_bytes=4096,
                    timeout_seconds=0.2,
                )

        selector.close.assert_called_once()
        self.assertEqual(len(terminated), 1)
        process = terminated[0]
        self.assertIsNotNone(process.returncode)
        self.assertIsNotNone(process.stdout)
        self.assertIsNotNone(process.stderr)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)


class CliEvidenceTests(unittest.TestCase):
    def test_cli_runner_requires_two_identical_bounded_replays(self) -> None:
        payload = CLI_INSPECT_STDOUT.encode("ascii")

        def completed(
            stdout: bytes = payload,
            returncode: int = 0,
            stderr: bytes = b"",
        ) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(
                args=[],
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        with mock.patch.object(
            Path, "is_file", return_value=True
        ), mock.patch.object(
            visuals.os, "access", return_value=True
        ), mock.patch.object(
            visuals,
            "_run_bounded_process",
            side_effect=[completed(), completed()],
        ) as run:
            stdout = visuals.run_release_cli(
                Path("unused-release"),
                "CLI inspect",
                visuals.CLI_INSPECT_ARGUMENTS,
            )

        self.assertEqual(stdout, CLI_INSPECT_STDOUT)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(
                command[1:], list(visuals.CLI_INSPECT_ARGUMENTS)
            )
            self.assertEqual(
                call.kwargs["environment"],
                {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(
                call.kwargs["timeout_seconds"],
                visuals.EXAMPLE_TIMEOUT_SECONDS,
            )
            self.assertEqual(
                call.kwargs["maximum_output_bytes"],
                visuals.MAX_OUTPUT_BYTES,
            )

        failure_cases = (
            (
                "divergent",
                [
                    completed(),
                    completed(
                        CLI_INSPECT_STDOUT.replace(
                            "dense_relu_v1", "dense_relu_v2", 1
                        ).encode("ascii")
                    ),
                ],
            ),
            ("nonzero", [completed(returncode=2)]),
            ("stderr", [completed(stderr=b"unexpected\n")]),
        )
        for label, results in failure_cases:
            with self.subTest(label=label), mock.patch.object(
                Path, "is_file", return_value=True
            ), mock.patch.object(
                visuals.os, "access", return_value=True
            ), mock.patch.object(
                visuals, "_run_bounded_process", side_effect=results
            ), self.assertRaises(visuals.VisualEvidenceError):
                visuals.run_release_cli(
                    Path("unused-release"),
                    "CLI inspect",
                    visuals.CLI_INSPECT_ARGUMENTS,
                )

    def test_cli_validator_and_svg_are_exact_and_data_derived(self) -> None:
        evidence = visuals.validate_cli_evidence(
            CLI_INSPECT_STDOUT, CLI_EXECUTE_STDOUT
        )
        first = visuals.render_cli_execution_svg(
            CLI_INSPECT_STDOUT, CLI_EXECUTE_STDOUT
        )
        second = visuals.render_cli_execution_svg(
            CLI_INSPECT_STDOUT, CLI_EXECUTE_STDOUT
        )

        self.assertEqual(evidence.input_bits, visuals.CLI_INPUT_BITS)
        self.assertEqual(evidence.output_bits, visuals.CLI_OUTPUT_BITS)
        self.assertEqual(first, second)
        for expected in (
            "make -j2 PROFILE=release cli",
            "matmul_rank2_f32",
            "add_broadcast_f32",
            "relu_contiguous_f32",
            "128 B verified workspace",
            "kernel_write_audit = ON",
            "4 / 4 RAW BITS MATCH",
            "0x40900000",
            "FIXTURE-SCOPED",
            "NOT A BENCHMARK",
        ):
            self.assertIn(expected, first)
        self.assertNotIn("<script", first.lower())
        self.assertNotIn("<image", first.lower())
        self.assertNotIn(" href=", first.lower())
        ElementTree.fromstring(first)

    def test_cli_validator_rejects_contract_mutations(self) -> None:
        cases: list[tuple[str, object]] = []

        def mutated_execute() -> dict[str, object]:
            return json.loads(CLI_EXECUTE_STDOUT)

        audit = mutated_execute()
        audit["execution"]["kernel_write_audit"] = False
        cases.append(("audit", audit))

        output = mutated_execute()
        output["execution"]["outputs"][0]["bits"][0] = "0x40900001"
        cases.append(("output", output))

        reference = mutated_execute()
        reference["execution"]["reference_check"]["matched"] = 3
        cases.append(("reference", reference))

        workspace = mutated_execute()
        workspace["plan"]["stats"]["workspace_bytes"] = 127
        cases.append(("workspace", workspace))

        kernel = mutated_execute()
        kernel["plan"]["kernels"][1]["kind"] = "wrong_kernel"
        cases.append(("kernel", kernel))

        workload = mutated_execute()
        workload["workload"]["id"] = "other_workload"
        cases.append(("workload", workload))

        extra = mutated_execute()
        extra["execution"]["unverified"] = True
        cases.append(("extra field", extra))

        for label, record in cases:
            with self.subTest(label=label), self.assertRaises(
                visuals.VisualEvidenceError
            ):
                visuals.validate_cli_evidence(
                    CLI_INSPECT_STDOUT,
                    json.dumps(record, separators=(",", ":")) + "\n",
                )

        inspect = json.loads(CLI_INSPECT_STDOUT)
        inspect["plan"]["canonical_dump"] += "unverified claim\n"
        with self.assertRaisesRegex(
            visuals.VisualEvidenceError, "canonical dump differs"
        ):
            visuals.validate_cli_evidence(
                json.dumps(inspect, separators=(",", ":")) + "\n",
                CLI_EXECUTE_STDOUT,
            )

    def test_cli_validator_rejects_duplicate_keys_and_nonfinite_json(
        self,
    ) -> None:
        duplicate = CLI_EXECUTE_STDOUT.replace(
            '{"schema":',
            '{"schema":"duplicate","schema":',
            1,
        )
        nonfinite = CLI_EXECUTE_STDOUT.replace(
            '"logical_workspace_bytes":128',
            '"logical_workspace_bytes":NaN',
            1,
        )
        for label, stdout in (
            ("duplicate", duplicate),
            ("nonfinite", nonfinite),
        ):
            with self.subTest(label=label), self.assertRaises(
                visuals.VisualEvidenceError
            ):
                visuals.validate_cli_evidence(
                    CLI_INSPECT_STDOUT, stdout
                )

    def test_v3_bundle_binds_replayed_cli_json_and_binary(self) -> None:
        def fake_example(
            _build_dir: Path,
            binary_name: str,
            _sentinels: tuple[str, ...],
        ) -> str:
            return {
                "plan_arena": PLAN_STDOUT,
                "execute_graph": EXECUTE_STDOUT,
                "execute_softmax": SOFTMAX_STDOUT,
            }[binary_name]

        def fake_cli(
            _build_dir: Path,
            label: str,
            _arguments: tuple[str, ...],
        ) -> str:
            if label == "CLI inspect":
                return CLI_INSPECT_STDOUT
            if label == "CLI execute":
                return CLI_EXECUTE_STDOUT
            raise AssertionError(f"unexpected CLI label: {label}")

        source_provenance = {
            "commit": "1" * 40,
            "object_format": "sha1",
            "selection": "test source selection",
            "source_files": {
                "cli/tensorkiln.cpp": {
                    "bytes": 1,
                    "git_blob": "2" * 40,
                    "mode": "100644",
                    "sha256": "3" * 64,
                }
            },
            "tree": "4" * 40,
        }
        binary_provenance = {
            "execute_graph": {"bytes": 101, "sha256": "5" * 64},
            "execute_softmax": {"bytes": 102, "sha256": "6" * 64},
            "plan_arena": {"bytes": 103, "sha256": "7" * 64},
            "tensorkiln": {"bytes": 104, "sha256": "8" * 64},
        }
        generator_provenance = {
            "bytes": 105,
            "commit": "9" * 40,
            "committed": True,
            "git_blob": "a" * 40,
            "path": "tools/render_readme_visuals.py",
            "sha256": "b" * 64,
            "tree": "c" * 40,
        }
        with mock.patch.object(
            visuals, "run_release_example", side_effect=fake_example
        ), mock.patch.object(
            visuals, "run_release_cli", side_effect=fake_cli
        ), mock.patch.object(
            visuals,
            "collect_source_provenance",
            return_value=source_provenance,
        ), mock.patch.object(
            visuals,
            "collect_binary_provenance",
            return_value=binary_provenance,
        ), mock.patch.object(
            visuals,
            "collect_generator_provenance",
            return_value=generator_provenance,
        ):
            artifacts = visuals.render_visuals(
                Path("unused-release-dir"),
                include_cli=True,
            )

        self.assertEqual(artifacts["cli-inspect.json"], CLI_INSPECT_STDOUT)
        self.assertEqual(artifacts["cli-execute.json"], CLI_EXECUTE_STDOUT)
        ElementTree.fromstring(artifacts["cli-execution.svg"])
        manifest = json.loads(artifacts["manifest.json"])
        self.assertEqual(
            manifest["schema"], "tensorkiln.readme-visual-evidence.v3"
        )
        self.assertEqual(
            manifest["capture_contract"]["cli_replays_per_command"], 2
        )
        cli_source = manifest["sources"]["tensorkiln"]
        self.assertEqual(cli_source["binary_sha256"], "8" * 64)
        self.assertEqual(
            cli_source["commands"]["inspect"]["arguments"],
            list(visuals.CLI_INSPECT_ARGUMENTS),
        )
        self.assertEqual(
            cli_source["commands"]["execute"]["stdout_sha256"],
            hashlib.sha256(CLI_EXECUTE_STDOUT.encode()).hexdigest(),
        )
        self.assertTrue(
            cli_source["commands"]["execute"]["byte_identical"]
        )
        self.assertEqual(cli_source["commands"]["execute"]["replays"], 2)
        self.assertIn(
            "cli/tensorkiln.cpp",
            manifest["repository_source"]["source_files"],
        )
        for filename, record in manifest["artifacts"].items():
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(artifacts[filename].encode()).hexdigest(),
            )

    def test_v3_cross_toolchain_normalization_preserves_cli_elf(
        self,
    ) -> None:
        sources = {
            name: {
                "binary": name,
                "binary_bytes": index + 100,
                "binary_sha256": str(index) * 64,
            }
            for index, name in enumerate(
                (
                    "execute_graph",
                    "execute_softmax",
                    "plan_arena",
                    "tensorkiln",
                ),
                start=1,
            )
        }
        sources["tensorkiln"]["commands"] = {"execute": {"replays": 2}}
        recorded = {
            "schema": "tensorkiln.readme-visual-evidence.v3",
            "generator": {"committed": True, "recorded": True},
            "repository_source": {"recorded": True},
            "sources": sources,
        }
        current = json.loads(json.dumps(recorded))
        for source in current["sources"].values():
            source["binary_bytes"] += 1000
            source["binary_sha256"] = "f" * 64

        with mock.patch.object(
            visuals,
            "_validate_recorded_generator",
            return_value=recorded["generator"],
        ), mock.patch.object(
            visuals,
            "_preserve_recorded_repository_source",
            return_value=recorded["repository_source"],
        ):
            normalized = visuals.preserve_recorded_capture_provenance(
                recorded, current
            )

        for binary_name, source in sources.items():
            self.assertEqual(
                normalized["sources"][binary_name]["binary_bytes"],
                source["binary_bytes"],
            )
            self.assertEqual(
                normalized["sources"][binary_name]["binary_sha256"],
                source["binary_sha256"],
            )
        self.assertEqual(
            normalized["sources"]["tensorkiln"]["commands"],
            {"execute": {"replays": 2}},
        )


class RegluVisualEvidenceTests(unittest.TestCase):
    def test_terminal_tail_starts_at_a_complete_canonical_line(self) -> None:
        lines = visuals._terminal_frame_lines(
            visuals._display_cli_command(visuals.REGLU_INSPECT_TEXT_ARGUMENTS),
            REGLU_INSPECT_TEXT_STDOUT,
            73,
            24,
            tail=True,
        )

        self.assertEqual(len(lines), 23)
        self.assertEqual(
            lines[0],
            "    %9 f32[2,4] dense strides=[4,1] storage=arena "
            "#b4 offset=128",
        )
        self.assertEqual(lines[-1], "}")

    def test_exact_contract_drives_graph_arena_output_and_terminal_media(
        self,
    ) -> None:
        evidence = validated_reglu_evidence()

        self.assertEqual(
            hashlib.sha256(REGLU_WORKLOADS_STDOUT.encode()).hexdigest(),
            "8ed8ce35c8925bce34f97677696a04e90f7ef24d27d4f447d5dbae76d6a230bc",
        )
        self.assertEqual(
            hashlib.sha256(REGLU_INSPECT_STDOUT.encode()).hexdigest(),
            "87c703c0afded52b27b471d79db6279b43fe46edd6d8ff17b53d153c62305fb9",
        )
        self.assertEqual(
            hashlib.sha256(REGLU_EXECUTE_STDOUT.encode()).hexdigest(),
            "05fa2c9eb78236547e49120fa216dd47ed3c0cfe385cc4ec164741af74009381",
        )
        self.assertEqual(
            hashlib.sha256(REGLU_EXECUTE_TEXT_STDOUT.encode()).hexdigest(),
            "e8ad4c040dc16b11b821eee8b6e744cd678a14bed35bbab2f950fe5c95aef896",
        )

        self.assertEqual(len(evidence.steps), 6)
        self.assertEqual(
            evidence.steps[-1].operands,
            (5, 9),
        )
        self.assertEqual(
            [(item.offset, item.live_start, item.live_end) for item in evidence.arena],
            [
                (0, 0, 2),
                (64, 1, 3),
                (0, 2, 6),
                (64, 3, 5),
                (128, 4, 6),
                (64, 5, 6),
            ],
        )
        self.assertEqual(evidence.output_bits[4], "0x80000000")

        graph = visuals.render_reglu_graph_svg(evidence)
        arena = visuals.render_reglu_arena_svg(evidence)
        output = visuals.render_reglu_output_svg(evidence)
        for svg in (graph, arena, output):
            ElementTree.fromstring(svg)
            self.assertNotIn("<script", svg.lower())
            self.assertNotIn("<image", svg.lower())
            self.assertNotIn(" href=", svg.lower())
            visuals.reject_unsafe_text("ReGLU SVG", svg)
        for expected in (
            "matmul_rank2",
            "relu_contiguous",
            "mul_contiguous",
            "%5,%9",
            "80 scalar steps",
            "Constants %1/%3/%6/%8 are omitted",
            "NOT A FULL TRANSFORMER",
        ):
            self.assertIn(expected, graph)
        for expected in (
            "#b0→#b2",
            "#b1→#b3→#b5",
            "192 B verified workspace",
            "[5,6)",
        ):
            self.assertIn(expected, arena)
        arena_root = ElementTree.fromstring(arena)
        arena_rectangles = [
            node
            for node in arena_root.findall(
                "{http://www.w3.org/2000/svg}rect"
            )
            if node.attrib.get("height") == "16"
        ]
        arena_y = [int(node.attrib["y"]) for node in arena_rectangles]
        self.assertEqual(arena_y, [197, 321, 197, 321, 445, 321])
        self.assertEqual(arena_y[1], arena_y[3])
        self.assertEqual(arena_y[3], arena_y[5])
        for y, slot_top in zip(
            arena_y,
            (178, 302, 178, 302, 426, 302),
            strict=True,
        ):
            self.assertGreaterEqual(y, slot_top)
            self.assertLessEqual(y + 16, slot_top + 54)
        for expected in (
            "8 / 8 RAW F32 WORDS MATCH",
            "0x80000000",
            "f32 = -0",
            "NOT A BENCHMARK",
        ):
            self.assertIn(expected, output)

        png = visuals.render_reglu_terminal_png(evidence)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(
            int.from_bytes(png[16:20], "big"), visuals.TERMINAL_WIDTH
        )
        self.assertEqual(
            int.from_bytes(png[20:24], "big"), visuals.TERMINAL_HEIGHT
        )
        chunks: list[tuple[bytes, bytes]] = []
        cursor = 8
        while cursor < len(png):
            length = int.from_bytes(png[cursor : cursor + 4], "big")
            kind = png[cursor + 4 : cursor + 8]
            payload = png[cursor + 8 : cursor + 8 + length]
            checksum = int.from_bytes(
                png[cursor + 8 + length : cursor + 12 + length], "big"
            )
            self.assertEqual(checksum, zlib.crc32(kind + payload) & 0xFFFFFFFF)
            chunks.append((kind, payload))
            cursor += 12 + length
        self.assertEqual([kind for kind, _ in chunks], [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(cursor, len(png))
        self.assertEqual(
            chunks[0][1],
            (
                visuals.TERMINAL_WIDTH.to_bytes(4, "big")
                + visuals.TERMINAL_HEIGHT.to_bytes(4, "big")
                + bytes((8, 2, 0, 0, 0))
            ),
        )

        stored_stream = chunks[1][1]
        self.assertEqual(stored_stream[:2], b"\x78\x01")
        self.assertEqual(int.from_bytes(stored_stream[:2], "big") % 31, 0)
        stored_cursor = 2
        decoded = bytearray()
        block_count = 0
        while True:
            header = stored_stream[stored_cursor]
            stored_cursor += 1
            self.assertEqual(header & 0xFE, 0)
            final = bool(header & 1)
            length = int.from_bytes(
                stored_stream[stored_cursor : stored_cursor + 2], "little"
            )
            complement = int.from_bytes(
                stored_stream[stored_cursor + 2 : stored_cursor + 4], "little"
            )
            self.assertEqual(length ^ complement, 0xFFFF)
            stored_cursor += 4
            decoded.extend(stored_stream[stored_cursor : stored_cursor + length])
            stored_cursor += length
            block_count += 1
            if final:
                break
        self.assertEqual(block_count, 24)
        self.assertEqual(stored_cursor + 4, len(stored_stream))
        self.assertEqual(
            int.from_bytes(stored_stream[stored_cursor:], "big"),
            visuals._adler32(bytes(decoded)),
        )

        source_frame = visuals._render_terminal_frame(
            "TensorKiln ReGLU release CLI",
            visuals._display_cli_command(visuals.REGLU_EXECUTE_TEXT_ARGUMENTS),
            evidence.execute_text,
            tail=False,
        )
        expected_rows = bytearray()
        for y in range(visuals.TERMINAL_HEIGHT):
            expected_rows.append(0)
            for color in source_frame[
                y * visuals.TERMINAL_WIDTH : (y + 1) * visuals.TERMINAL_WIDTH
            ]:
                expected_rows.extend(visuals._TERMINAL_PALETTE[color])
        self.assertEqual(decoded, expected_rows)
        inflater = zlib.decompressobj()
        self.assertEqual(
            inflater.decompress(stored_stream) + inflater.flush(),
            expected_rows,
        )
        self.assertTrue(inflater.eof)
        self.assertEqual(inflater.unconsumed_tail, b"")
        self.assertEqual(inflater.unused_data, b"")
        self.assertEqual(visuals.render_reglu_terminal_png(evidence), png)

        gif = visuals.render_reglu_demo_gif(evidence)
        self.assertEqual(gif[:6], b"GIF89a")
        self.assertEqual(
            int.from_bytes(gif[6:8], "little"), visuals.TERMINAL_WIDTH
        )
        self.assertEqual(
            int.from_bytes(gif[8:10], "little"), visuals.TERMINAL_HEIGHT
        )
        self.assertEqual(gif.count(b"!\xf9\x04"), 3)
        self.assertNotIn(b"NETSCAPE", gif)
        self.assertNotIn(b"Comment", gif)
        self.assertEqual(
            visuals.render_reglu_demo_gif(evidence),
            gif,
        )

        transcript = visuals.render_reglu_demo_transcript(evidence)
        self.assertIn("tensorkiln list", transcript)
        self.assertIn("tensorkiln inspect --workload reglu_mlp_v1", transcript)
        self.assertIn("tensorkiln execute --workload reglu_mlp_v1", transcript)
        self.assertIn(visuals.REGLU_CANONICAL_DUMP, transcript)
        self.assertIn(REGLU_EXECUTE_TEXT_STDOUT, transcript)
        self.assertNotIn("/home/", transcript)

    def test_gif_literal_streams_are_bounded_complete_frames(self) -> None:
        evidence = validated_reglu_evidence()
        gif = visuals.render_reglu_demo_gif(evidence)
        expected_frames = [
            visuals._render_terminal_frame(
                title,
                visuals._display_cli_command(arguments),
                stdout,
                tail=tail,
            )
            for title, arguments, stdout, tail in (
                (
                    "1 / 3  Workload registry",
                    visuals.REGLU_LIST_TEXT_ARGUMENTS,
                    evidence.list_text,
                    False,
                ),
                (
                    "2 / 3  Canonical plan tail",
                    visuals.REGLU_INSPECT_TEXT_ARGUMENTS,
                    evidence.inspect_text,
                    True,
                ),
                (
                    "3 / 3  Audited execution",
                    visuals.REGLU_EXECUTE_TEXT_ARGUMENTS,
                    evidence.execute_text,
                    False,
                ),
            )
        ]
        cursor = 13 + 3 * 8
        decoded_frames = 0
        delays: list[int] = []
        while gif[cursor] != 0x3B:
            self.assertEqual(gif[cursor : cursor + 3], b"!\xf9\x04")
            delays.append(int.from_bytes(gif[cursor + 4 : cursor + 6], "little"))
            cursor += 8
            self.assertEqual(gif[cursor], 0x2C)
            cursor += 10
            self.assertEqual(gif[cursor], 3)
            cursor += 1
            compressed = bytearray()
            while gif[cursor] != 0:
                block_length = gif[cursor]
                cursor += 1
                compressed.extend(gif[cursor : cursor + block_length])
                cursor += block_length
            cursor += 1

            codes: list[int] = []
            for byte in compressed:
                codes.extend((byte & 0x0F, byte >> 4))
            pixel_count = 0
            decoded = bytearray()
            literals_since_clear = 0
            saw_end = False
            for code in codes:
                if code == 8:
                    literals_since_clear = 0
                elif code == 9:
                    saw_end = True
                    break
                else:
                    self.assertLess(code, 8)
                    literals_since_clear += 1
                    self.assertLessEqual(literals_since_clear, 6)
                    pixel_count += 1
                    decoded.append(code)
            self.assertTrue(saw_end)
            self.assertEqual(
                pixel_count,
                visuals.TERMINAL_WIDTH * visuals.TERMINAL_HEIGHT,
            )
            self.assertEqual(decoded, expected_frames[decoded_frames])
            decoded_frames += 1
        self.assertEqual(cursor, len(gif) - 1)
        self.assertEqual(decoded_frames, 3)
        self.assertEqual(delays, list(visuals.TERMINAL_GIF_DELAYS_CS))

    def test_reglu_validator_fails_closed_on_mutation_and_ansi(self) -> None:
        output = json.loads(REGLU_EXECUTE_STDOUT)
        output["execution"]["outputs"][0]["bits"][4] = "0x00000000"
        mutated_output = json.dumps(output, separators=(",", ":")) + "\n"

        inspect = json.loads(REGLU_INSPECT_STDOUT)
        inspect["plan"]["canonical_dump"] = inspect["plan"][
            "canonical_dump"
        ].replace("mul_contiguous_f32(%5,%9)", "mul_contiguous_f32(%9,%5)")
        mutated_inspect = json.dumps(inspect, separators=(",", ":")) + "\n"

        workloads = json.loads(REGLU_WORKLOADS_STDOUT)
        workloads["workloads"].reverse()
        mutated_workloads = json.dumps(workloads, separators=(",", ":")) + "\n"

        cases = (
            (mutated_workloads, REGLU_INSPECT_STDOUT, REGLU_EXECUTE_STDOUT,
             REGLU_LIST_TEXT_STDOUT),
            (REGLU_WORKLOADS_STDOUT, mutated_inspect, REGLU_EXECUTE_STDOUT,
             REGLU_LIST_TEXT_STDOUT),
            (REGLU_WORKLOADS_STDOUT, REGLU_INSPECT_STDOUT, mutated_output,
             REGLU_LIST_TEXT_STDOUT),
            (REGLU_WORKLOADS_STDOUT, REGLU_INSPECT_STDOUT, REGLU_EXECUTE_STDOUT,
             REGLU_LIST_TEXT_STDOUT + "\x1b[31m"),
        )
        for list_json, inspect_json, execute_json, list_text in cases:
            with self.subTest(
                digest=hashlib.sha256(
                    (list_json + inspect_json + execute_json + list_text).encode()
                ).hexdigest()
            ), self.assertRaises(visuals.VisualEvidenceError):
                visuals.validate_reglu_evidence(
                    list_json,
                    inspect_json,
                    execute_json,
                    list_text,
                    REGLU_INSPECT_TEXT_STDOUT,
                    REGLU_EXECUTE_TEXT_STDOUT,
                )

    def test_v4_bundle_preserves_v3_artifacts_and_binds_binary_media(
        self,
    ) -> None:
        published_execute_stdout = (
            REPOSITORY_ROOT
            / "docs"
            / "visuals"
            / "generated"
            / "execute-graph.txt"
        ).read_text(encoding="ascii")

        def fake_example(
            _build_dir: Path,
            binary_name: str,
            _sentinels: tuple[str, ...],
        ) -> str:
            return {
                "plan_arena": PLAN_STDOUT,
                "execute_graph": published_execute_stdout,
                "execute_softmax": SOFTMAX_STDOUT,
            }[binary_name]

        cli_outputs = {
            "CLI inspect": CLI_INSPECT_STDOUT,
            "CLI execute": CLI_EXECUTE_STDOUT,
            "ReGLU CLI list JSON": REGLU_WORKLOADS_STDOUT,
            "ReGLU CLI inspect JSON": REGLU_INSPECT_STDOUT,
            "ReGLU CLI execute JSON": REGLU_EXECUTE_STDOUT,
            "ReGLU CLI list text": REGLU_LIST_TEXT_STDOUT,
            "ReGLU CLI inspect text": REGLU_INSPECT_TEXT_STDOUT,
            "ReGLU CLI execute text": REGLU_EXECUTE_TEXT_STDOUT,
        }

        def fake_cli(
            _build_dir: Path,
            label: str,
            _arguments: tuple[str, ...],
        ) -> str:
            return cli_outputs[label]

        source_provenance = {
            "commit": "1" * 40,
            "object_format": "sha1",
            "selection": "test source selection",
            "source_files": {},
            "tree": "2" * 40,
        }
        binary_provenance = {
            "execute_graph": {"bytes": 101, "sha256": "3" * 64},
            "execute_softmax": {"bytes": 102, "sha256": "4" * 64},
            "plan_arena": {"bytes": 103, "sha256": "5" * 64},
            "tensorkiln": {"bytes": 104, "sha256": "6" * 64},
        }
        generator_provenance = {
            "bytes": 105,
            "commit": "7" * 40,
            "committed": True,
            "git_blob": "8" * 40,
            "path": "tools/render_readme_visuals.py",
            "sha256": "9" * 64,
            "tree": "a" * 40,
        }

        def render(*, include_reglu: bool) -> dict[str, visuals.Artifact]:
            with mock.patch.object(
                visuals, "run_release_example", side_effect=fake_example
            ), mock.patch.object(
                visuals, "run_release_cli", side_effect=fake_cli
            ), mock.patch.object(
                visuals,
                "collect_source_provenance",
                return_value=source_provenance,
            ), mock.patch.object(
                visuals,
                "collect_binary_provenance",
                return_value=binary_provenance,
            ), mock.patch.object(
                visuals,
                "collect_generator_provenance",
                return_value=generator_provenance,
            ):
                return visuals.render_visuals(
                    Path("unused-release-dir"),
                    include_cli=True,
                    include_reglu=include_reglu,
                )

        v3 = render(include_reglu=False)
        v4 = render(include_reglu=True)
        existing_artifacts = {
            "arena-plan.txt",
            "arena-reuse.svg",
            "cli-execute.json",
            "cli-execution.svg",
            "cli-inspect.json",
            "execute-graph.svg",
            "execute-graph.txt",
            "execute-softmax.svg",
            "execute-softmax.txt",
        }
        self.assertEqual(
            visuals.PUBLISHED_V3_ARTIFACT_SHA256, PUBLISHED_V3_SHA256
        )
        self.assertEqual(set(PUBLISHED_V3_SHA256), existing_artifacts)
        for filename in existing_artifacts:
            self.assertEqual(v4[filename], v3[filename])
            self.assertEqual(
                hashlib.sha256(
                    visuals._artifact_bytes(v4[filename])
                ).hexdigest(),
                PUBLISHED_V3_SHA256[filename],
            )

        manifest = json.loads(v4["manifest.json"])
        self.assertEqual(
            manifest["schema"], "tensorkiln.readme-visual-evidence.v4"
        )
        for filename, artifact in v4.items():
            if filename == "manifest.json":
                continue
            record = manifest["artifacts"][filename]
            payload = visuals._artifact_bytes(artifact)
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(
                record["sha256"], hashlib.sha256(payload).hexdigest()
            )
            self.assertEqual(
                record["media_type"], visuals._artifact_media_type(filename)
            )
        self.assertIsInstance(v4["reglu-terminal.png"], bytes)
        self.assertIsInstance(v4["reglu-demo.gif"], bytes)
        commands = manifest["sources"]["tensorkiln"]["commands"]
        expected_commands = (
            (
                "execute",
                visuals.CLI_EXECUTE_ARGUMENTS,
                "tensorkiln.cli.execute.v1",
                "cli-execute.json",
                CLI_EXECUTE_STDOUT,
            ),
            (
                "inspect",
                visuals.CLI_INSPECT_ARGUMENTS,
                "tensorkiln.cli.inspect.v1",
                "cli-inspect.json",
                CLI_INSPECT_STDOUT,
            ),
            (
                "reglu_execute_json",
                visuals.REGLU_EXECUTE_ARGUMENTS,
                "tensorkiln.cli.execute.v1",
                "reglu-execute.json",
                REGLU_EXECUTE_STDOUT,
            ),
            (
                "reglu_execute_text",
                visuals.REGLU_EXECUTE_TEXT_ARGUMENTS,
                "text/plain",
                "reglu-execute.txt",
                REGLU_EXECUTE_TEXT_STDOUT,
            ),
            (
                "reglu_inspect_json",
                visuals.REGLU_INSPECT_ARGUMENTS,
                "tensorkiln.cli.inspect.v1",
                "reglu-inspect.json",
                REGLU_INSPECT_STDOUT,
            ),
            (
                "reglu_inspect_text",
                visuals.REGLU_INSPECT_TEXT_ARGUMENTS,
                "text/plain",
                "reglu-inspect.txt",
                REGLU_INSPECT_TEXT_STDOUT,
            ),
            (
                "reglu_list_json",
                visuals.REGLU_LIST_ARGUMENTS,
                "tensorkiln.cli.workloads.v1",
                "cli-workloads.json",
                REGLU_WORKLOADS_STDOUT,
            ),
            (
                "reglu_list_text",
                visuals.REGLU_LIST_TEXT_ARGUMENTS,
                "text/plain",
                "reglu-list.txt",
                REGLU_LIST_TEXT_STDOUT,
            ),
        )
        self.assertEqual(
            set(commands), {item[0] for item in expected_commands}
        )
        for name, arguments, schema, artifact, stdout in expected_commands:
            self.assertEqual(
                commands[name],
                {
                    "arguments": list(arguments),
                    "byte_identical": True,
                    "replays": 2,
                    "schema": schema,
                    "stdout_artifact": artifact,
                    "stdout_sha256": hashlib.sha256(
                        stdout.encode()
                    ).hexdigest(),
                },
            )
        self.assertEqual(
            manifest["sources"]["tensorkiln"]["presentation"]
            ["gif_delay_centiseconds"],
            list(visuals.TERMINAL_GIF_DELAYS_CS),
        )
        self.assertTrue(
            any(
                "not a full transformer" in claim
                for claim in manifest["claim_boundary"]
            )
        )
        self.assertTrue(
            any(
                "preserved v3 dense CLI evidence" in claim
                for claim in manifest["claim_boundary"]
            )
        )
        self.assertTrue(
            any(
                "shallow checks bind current generator" in claim
                for claim in manifest["claim_boundary"]
            )
        )

    def test_output_io_rejects_symlinks_and_reports_orphans(self) -> None:
        build_dir = REPOSITORY_ROOT / "build"
        build_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="visual-io-test-", dir=build_dir
        ) as temporary:
            root = Path(temporary)
            output = root / "generated"
            generated: dict[str, visuals.Artifact] = {
                "sample.txt": "exact text\n",
                "sample.png": b"\x89PNG\r\n\x1a\n",
            }
            with mock.patch("builtins.print"):
                visuals.write_visuals(output, generated)
                self.assertEqual(visuals.check_visuals(output, generated), 0)
            self.assertEqual(
                {path.name for path in output.iterdir()}, set(generated)
            )
            self.assertEqual(
                (output / "sample.txt").stat().st_mode & 0o777, 0o644
            )

            orphan = output / "orphan.txt"
            orphan.write_text("orphan\n", encoding="ascii")
            with mock.patch("builtins.print") as report:
                self.assertEqual(visuals.check_visuals(output, generated), 1)
            self.assertTrue(
                any("orphan" in str(call) for call in report.call_args_list)
            )
            orphan.unlink()

            outside = root / "outside.txt"
            outside.write_text("must remain unchanged\n", encoding="ascii")
            unsafe_output = root / "unsafe"
            unsafe_output.mkdir()
            (unsafe_output / "sample.txt").symlink_to(outside)
            with self.assertRaisesRegex(
                visuals.VisualEvidenceError, "securely open visual artifact"
            ):
                visuals.check_visuals(
                    unsafe_output, {"sample.txt": "replacement\n"}
                )
            with self.assertRaisesRegex(
                visuals.VisualEvidenceError, "not a regular file"
            ):
                visuals.write_visuals(
                    unsafe_output, {"sample.txt": "replacement\n"}
                )
            self.assertEqual(
                outside.read_text(encoding="ascii"),
                "must remain unchanged\n",
            )

            linked_directory = root / "linked-output"
            linked_directory.symlink_to(output, target_is_directory=True)
            with self.assertRaisesRegex(
                visuals.VisualEvidenceError,
                "securely open visual output directory",
            ):
                visuals.write_visuals(
                    linked_directory, {"sample.txt": "replacement\n"}
                )

    def test_uncommitted_v4_preview_is_checkable_only_when_explicit(self) -> None:
        preview_generator = {
            "bytes": 123,
            "commit": None,
            "committed": False,
            "git_blob": None,
            "path": "tools/render_readme_visuals.py",
            "sha256": "a" * 64,
            "tree": None,
        }
        sources = {
            name: {
                "binary": name,
                "binary_bytes": index + 100,
                "binary_sha256": str(index) * 64,
            }
            for index, name in enumerate(
                (
                    "execute_graph",
                    "execute_softmax",
                    "plan_arena",
                    "tensorkiln",
                ),
                start=1,
            )
        }
        preview_manifest = {
            "generator": preview_generator,
            "repository_source": {"preview": True},
            "schema": "tensorkiln.readme-visual-evidence.v4",
            "sources": sources,
        }
        generated: dict[str, visuals.Artifact] = {
            "manifest.json": (
                json.dumps(preview_manifest, indent=2, sort_keys=True) + "\n"
            )
        }
        build_dir = REPOSITORY_ROOT / "build"
        with tempfile.TemporaryDirectory(
            prefix="visual-preview-test-", dir=build_dir
        ) as temporary, mock.patch.object(
            visuals,
            "_preserve_recorded_repository_source",
            side_effect=lambda _recorded, current: current,
        ), mock.patch("builtins.print"):
            output = Path(temporary)
            visuals.write_visuals(output, generated)
            self.assertEqual(
                visuals.check_visuals(
                    output,
                    generated,
                    allow_uncommitted_generator=True,
                ),
                0,
            )
            with self.assertRaisesRegex(
                visuals.VisualEvidenceError,
                "only for an explicit preview",
            ):
                visuals.check_visuals(output, generated)


class SourceProvenanceTests(unittest.TestCase):
    def test_shallow_generator_provenance_binds_current_blob_content(
        self,
    ) -> None:
        payload = b"committed renderer fixture\n"
        git_blob = hashlib.sha1(
            f"blob {len(payload)}\0".encode("ascii") + payload
        ).hexdigest()
        generator = {
            "bytes": len(payload),
            "commit": "1" * 40,
            "committed": True,
            "git_blob": git_blob,
            "path": visuals.GENERATOR_PATH,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "tree": "2" * 40,
        }

        with mock.patch.object(
            visuals,
            "_git_text",
            side_effect=("sha1", "true"),
        ), mock.patch.object(
            visuals,
            "_read_regular_file",
            return_value=payload,
        ):
            self.assertEqual(
                visuals._validate_recorded_generator(generator),
                generator,
            )

        with mock.patch.object(
            visuals,
            "_git_text",
            side_effect=("sha1", "true"),
        ), mock.patch.object(
            visuals,
            "_read_regular_file",
            return_value=payload + b"drift",
        ), self.assertRaisesRegex(
            visuals.VisualEvidenceError,
            "shallow checkout generator differs",
        ):
            visuals._validate_recorded_generator(generator)

    def test_dirty_pathspec_scope_fails_before_source_discovery(self) -> None:
        with mock.patch.object(
            visuals,
            "_git_text",
            return_value=str(REPOSITORY_ROOT),
        ), mock.patch.object(
            visuals,
            "_run_git",
            return_value=b" D include/tensorkiln/execution.hpp\0",
        ):
            with self.assertRaisesRegex(
                visuals.VisualEvidenceError,
                "build inputs differ",
            ):
                visuals.collect_source_provenance()

    def test_tracked_path_decoder_rejects_duplicates_and_escape(self) -> None:
        for payload in (
            b"src/a.cpp\0src/a.cpp\0",
            b"src/../outside.cpp\0",
            b"/absolute.cpp\0",
            b"src/line\nbreak.cpp\0",
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(visuals.VisualEvidenceError):
                    visuals._decode_nul_paths(payload, "test paths")

    def test_tree_record_rejects_symlinks_and_submodules(self) -> None:
        object_pattern = re.compile(r"^[0-9a-f]{40}$")
        for mode, object_type in (
            ("120000", "blob"),
            ("160000", "commit"),
        ):
            record = (
                f"{mode} {object_type} {'c' * 40}\tsrc/input.cpp\0"
            ).encode()
            with self.subTest(mode=mode), mock.patch.object(
                visuals, "_run_git", return_value=record
            ):
                with self.assertRaisesRegex(
                    visuals.VisualEvidenceError,
                    "non-symlink 100644 blob",
                ):
                    visuals._tree_blob_record(
                        "d" * 40, "src/input.cpp", object_pattern
                    )

    def test_current_evidence_build_inputs_are_commit_bound(self) -> None:
        provenance = visuals.collect_source_provenance()
        source_files = provenance["source_files"]

        self.assertRegex(provenance["commit"], r"^[0-9a-f]{40,64}$")
        self.assertRegex(provenance["tree"], r"^[0-9a-f]{40,64}$")
        self.assertIn("Makefile", source_files)
        self.assertIn("examples/execute_softmax.cpp", source_files)
        self.assertIn(
            "include/tensorkiln/execution.hpp", source_files
        )
        self.assertIn("src/execution_kernels.cpp", source_files)
        for record in source_files.values():
            self.assertEqual(record["mode"], "100644")
            self.assertRegex(record["git_blob"], r"^[0-9a-f]{40,64}$")
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")

    def test_cli_evidence_source_scope_includes_the_real_entrypoint(
        self,
    ) -> None:
        provenance = visuals.collect_source_provenance(include_cli=True)
        self.assertIn(
            "cli/tensorkiln.cpp", provenance["source_files"]
        )


class CommittedVisualEvidenceTests(unittest.TestCase):
    def test_v4_bundle_is_complete_safe_and_manifest_bound(self) -> None:
        evidence_dir = REPOSITORY_ROOT / "docs" / "visuals" / "generated"
        manifest_path = evidence_dir / "manifest.json"

        def reject_duplicate_keys(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise AssertionError(
                        f"duplicate manifest key: {key}"
                    )
                result[key] = value
            return result

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
        self.assertEqual(
            manifest["schema"], "tensorkiln.readme-visual-evidence.v4"
        )
        expected_artifacts = {
            "arena-plan.txt",
            "arena-reuse.svg",
            "cli-execute.json",
            "cli-execution.svg",
            "cli-inspect.json",
            "cli-workloads.json",
            "execute-graph.svg",
            "execute-graph.txt",
            "execute-softmax.svg",
            "execute-softmax.txt",
            "reglu-arena.svg",
            "reglu-demo-transcript.txt",
            "reglu-demo.gif",
            "reglu-execute.json",
            "reglu-execute.txt",
            "reglu-graph.svg",
            "reglu-inspect.json",
            "reglu-inspect.txt",
            "reglu-list.txt",
            "reglu-output.svg",
            "reglu-terminal.png",
        }
        self.assertEqual(set(manifest["artifacts"]), expected_artifacts)
        self.assertEqual(
            {path.name for path in evidence_dir.iterdir() if path.is_file()},
            expected_artifacts | {"manifest.json"},
        )
        self.assertTrue(manifest["generator"]["committed"])
        self.assertRegex(
            manifest["generator"]["commit"], r"^[0-9a-f]{40,64}$"
        )
        self.assertRegex(
            manifest["generator"]["tree"], r"^[0-9a-f]{40,64}$"
        )
        self.assertRegex(
            manifest["generator"]["git_blob"], r"^[0-9a-f]{40,64}$"
        )
        self.assertEqual(
            manifest["capture_contract"]["network_isolation"],
            "not claimed",
        )
        self.assertEqual(
            manifest["capture_contract"]["cli_replays_per_command"], 2
        )
        self.assertEqual(
            manifest["repository_source"]["selection"],
            "latest commit touching the complete evidence build-input set",
        )
        self.assertIn(
            "examples/execute_softmax.cpp",
            manifest["repository_source"]["source_files"],
        )
        self.assertIn(
            "cli/tensorkiln.cpp",
            manifest["repository_source"]["source_files"],
        )
        visuals._validate_recorded_generator(manifest["generator"])

        current = json.loads(json.dumps(manifest))
        current["generator"] = {
            **current["generator"],
            "commit": "0" * len(current["generator"]["commit"]),
        }
        current["repository_source"] = {
            **current["repository_source"],
            "commit": "0" * len(
                current["repository_source"]["commit"]
            ),
            "tree": "0" * len(current["repository_source"]["tree"]),
        }
        for source in current["sources"].values():
            source["binary_bytes"] += 1
            source["binary_sha256"] = "0" * 64
        normalized = visuals.preserve_recorded_capture_provenance(
            manifest, current
        )
        self.assertEqual(normalized["generator"], manifest["generator"])
        self.assertEqual(
            normalized["repository_source"],
            manifest["repository_source"],
        )
        for binary_name in (
            "execute_graph",
            "execute_softmax",
            "plan_arena",
            "tensorkiln",
        ):
            self.assertEqual(
                normalized["sources"][binary_name]["binary_bytes"],
                manifest["sources"][binary_name]["binary_bytes"],
            )
            self.assertEqual(
                normalized["sources"][binary_name]["binary_sha256"],
                manifest["sources"][binary_name]["binary_sha256"],
            )

        for filename, record in manifest["artifacts"].items():
            payload = (evidence_dir / filename).read_bytes()
            self.assertEqual(
                set(record), {"bytes", "media_type", "sha256"}
            )
            self.assertEqual(record["bytes"], len(payload))
            self.assertEqual(
                record["media_type"],
                visuals._artifact_media_type(filename),
            )
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), record["sha256"]
            )
        for filename, expected_sha256 in PUBLISHED_V3_SHA256.items():
            self.assertEqual(
                manifest["artifacts"][filename]["sha256"],
                expected_sha256,
            )

        transcript_path = evidence_dir / "execute-softmax.txt"
        transcript = transcript_path.read_text(encoding="ascii")
        self.assertEqual(transcript, SOFTMAX_STDOUT)
        self.assertEqual(
            manifest["sources"]["execute_softmax"]["stdout_sha256"],
            hashlib.sha256(transcript.encode()).hexdigest(),
        )

        svg_path = evidence_dir / "execute-softmax.svg"
        svg = svg_path.read_text(encoding="utf-8")
        visuals.reject_unsafe_text("committed Softmax SVG", svg)
        self.assertNotIn("<script", svg.lower())
        self.assertNotIn("<image", svg.lower())
        self.assertNotIn(" href=", svg.lower())
        root = ElementTree.fromstring(svg)
        width = int(root.attrib["width"])
        height = int(root.attrib["height"])
        self.assertEqual(root.attrib["viewBox"], f"0 0 {width} {height}")

        namespace = {"svg": "http://www.w3.org/2000/svg"}
        terminal_group = root.find("svg:g", namespace)
        self.assertIsNotNone(terminal_group)
        terminal_lines = [
            "".join(node.itertext())
            for node in terminal_group.findall("svg:text", namespace)
        ]
        self.assertEqual(
            terminal_lines,
            [
                "$ <release-build>/execute_softmax",
                *SOFTMAX_STDOUT.splitlines(),
            ],
        )
        for node, line in zip(
            terminal_group.findall("svg:text", namespace),
            terminal_lines,
            strict=True,
        ):
            x = int(node.attrib["x"])
            y = int(node.attrib["y"])
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + len(line) * 8, width - 24)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, height)

        inspect_path = evidence_dir / "cli-inspect.json"
        execute_path = evidence_dir / "cli-execute.json"
        inspect = inspect_path.read_text(encoding="ascii")
        execute = execute_path.read_text(encoding="ascii")
        visuals.validate_cli_evidence(inspect, execute)
        visuals.reject_unsafe_text("committed CLI inspect JSON", inspect)
        visuals.reject_unsafe_text("committed CLI execute JSON", execute)

        reglu_outputs = {
            "reglu_list_json": (
                visuals.REGLU_LIST_ARGUMENTS,
                "tensorkiln.cli.workloads.v1",
                "cli-workloads.json",
            ),
            "reglu_list_text": (
                visuals.REGLU_LIST_TEXT_ARGUMENTS,
                "text/plain",
                "reglu-list.txt",
            ),
            "reglu_inspect_json": (
                visuals.REGLU_INSPECT_ARGUMENTS,
                "tensorkiln.cli.inspect.v1",
                "reglu-inspect.json",
            ),
            "reglu_inspect_text": (
                visuals.REGLU_INSPECT_TEXT_ARGUMENTS,
                "text/plain",
                "reglu-inspect.txt",
            ),
            "reglu_execute_json": (
                visuals.REGLU_EXECUTE_ARGUMENTS,
                "tensorkiln.cli.execute.v1",
                "reglu-execute.json",
            ),
            "reglu_execute_text": (
                visuals.REGLU_EXECUTE_TEXT_ARGUMENTS,
                "text/plain",
                "reglu-execute.txt",
            ),
        }
        reglu_stdout = {
            name: (evidence_dir / artifact).read_text(encoding="ascii")
            for name, (_arguments, _schema, artifact) in reglu_outputs.items()
        }
        visuals.validate_reglu_evidence(
            reglu_stdout["reglu_list_json"],
            reglu_stdout["reglu_inspect_json"],
            reglu_stdout["reglu_execute_json"],
            reglu_stdout["reglu_list_text"],
            reglu_stdout["reglu_inspect_text"],
            reglu_stdout["reglu_execute_text"],
        )
        for label, stdout in reglu_stdout.items():
            visuals.reject_unsafe_text(f"committed {label}", stdout)

        for filename in (
            "reglu-graph.svg",
            "reglu-arena.svg",
            "reglu-output.svg",
        ):
            rendered = (evidence_dir / filename).read_text(encoding="utf-8")
            ElementTree.fromstring(rendered)
            visuals.reject_unsafe_text(f"committed {filename}", rendered)
            self.assertNotIn("<script", rendered.lower())
            self.assertNotIn("<image", rendered.lower())
            self.assertNotIn(" href=", rendered.lower())
        self.assertTrue(
            (evidence_dir / "reglu-terminal.png")
            .read_bytes()
            .startswith(b"\x89PNG\r\n\x1a\n")
        )
        self.assertTrue(
            (evidence_dir / "reglu-demo.gif").read_bytes().startswith(
                b"GIF89a"
            )
        )

        cli_source = manifest["sources"]["tensorkiln"]
        self.assertEqual(cli_source["binary"], "tensorkiln")
        self.assertRegex(cli_source["binary_sha256"], r"^[0-9a-f]{64}$")
        expected_commands = {
            "inspect": (
                visuals.CLI_INSPECT_ARGUMENTS,
                "tensorkiln.cli.inspect.v1",
                "cli-inspect.json",
                inspect,
            ),
            "execute": (
                visuals.CLI_EXECUTE_ARGUMENTS,
                "tensorkiln.cli.execute.v1",
                "cli-execute.json",
                execute,
            ),
            **{
                name: (arguments, schema, artifact, reglu_stdout[name])
                for name, (arguments, schema, artifact) in reglu_outputs.items()
            },
        }
        self.assertEqual(set(cli_source["commands"]), set(expected_commands))
        for command_name, (
            arguments,
            schema,
            artifact,
            stdout,
        ) in expected_commands.items():
            command = cli_source["commands"][command_name]
            self.assertEqual(command["arguments"], list(arguments))
            self.assertEqual(command["schema"], schema)
            self.assertEqual(command["stdout_artifact"], artifact)
            self.assertEqual(command["replays"], 2)
            self.assertTrue(command["byte_identical"])
            self.assertEqual(
                command["stdout_sha256"],
                hashlib.sha256(stdout.encode()).hexdigest(),
            )

        cli_svg_path = evidence_dir / "cli-execution.svg"
        cli_svg = cli_svg_path.read_text(encoding="utf-8")
        visuals.reject_unsafe_text("committed CLI SVG", cli_svg)
        self.assertNotIn("<script", cli_svg.lower())
        self.assertNotIn("<image", cli_svg.lower())
        self.assertNotIn(" href=", cli_svg.lower())
        cli_root = ElementTree.fromstring(cli_svg)
        self.assertEqual(cli_root.attrib["width"], "1200")
        self.assertEqual(cli_root.attrib["height"], "720")
        for label in (
            "Audited CLI execution",
            "BYTE-IDENTICAL REPLAY ×2",
            "kernel_write_audit = ON",
            "4 / 4 RAW BITS MATCH",
            "NOT A BENCHMARK",
        ):
            self.assertIn(label, cli_svg)


class DocumentationAssetTests(unittest.TestCase):
    def test_architecture_is_self_contained_and_linked(self) -> None:
        architecture_path = (
            REPOSITORY_ROOT / "docs" / "visuals" / "architecture.svg"
        )
        architecture = architecture_path.read_text(encoding="utf-8")
        ElementTree.fromstring(architecture)
        visuals.reject_unsafe_text("architecture SVG", architecture)

        for required_label in (
            "GraphBuilder",
            "ExecutionPlanVerifier",
            "ExecutionSession",
            "ReferenceInterpreter",
        ):
            self.assertIn(required_label, architecture)
        self.assertNotIn("<script", architecture.lower())
        self.assertNotIn("<image", architecture.lower())
        self.assertNotIn(" href=", architecture.lower())

        reproduction_path = (
            REPOSITORY_ROOT / "docs" / "visuals" / "reproduce.svg"
        )
        reproduction = reproduction_path.read_text(encoding="utf-8")
        ElementTree.fromstring(reproduction)
        visuals.reject_unsafe_text("reproduction SVG", reproduction)
        for command in (
            "make -j2 PROFILE=release test",
            "make -j2 visuals",
            "make sanitize",
            "make oracle",
        ):
            self.assertIn(command, reproduction)
        self.assertNotIn("<script", reproduction.lower())
        self.assertNotIn("<image", reproduction.lower())
        self.assertNotIn(" href=", reproduction.lower())

        makefile = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
        for target in ("test:", "visuals:", "sanitize:", "oracle:"):
            self.assertIn(target, makefile)
        self.assertIn(
            "visuals-generate: $(EXAMPLE_BINARIES) $(CLI_BINARY)",
            makefile,
        )
        self.assertIn(
            "visuals-verify: $(EXAMPLE_BINARIES) $(CLI_BINARY)",
            makefile,
        )

        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        for relative_path in (
            "docs/visuals/architecture.svg",
            "docs/visuals/reproduce.svg",
            "docs/visuals/generated/arena-reuse.svg",
            "docs/visuals/generated/execute-graph.svg",
            "docs/visuals/generated/execute-softmax.svg",
            "docs/visuals/generated/execute-softmax.txt",
            "docs/visuals/generated/cli-inspect.json",
            "docs/visuals/generated/cli-execute.json",
            "docs/visuals/generated/cli-execution.svg",
            "docs/visuals/generated/cli-workloads.json",
            "docs/visuals/generated/reglu-list.txt",
            "docs/visuals/generated/reglu-inspect.json",
            "docs/visuals/generated/reglu-inspect.txt",
            "docs/visuals/generated/reglu-execute.json",
            "docs/visuals/generated/reglu-execute.txt",
            "docs/visuals/generated/reglu-graph.svg",
            "docs/visuals/generated/reglu-arena.svg",
            "docs/visuals/generated/reglu-output.svg",
            "docs/visuals/generated/reglu-terminal.png",
            "docs/visuals/generated/reglu-demo.gif",
            "docs/visuals/generated/reglu-demo-transcript.txt",
            "docs/visuals/generated/manifest.json",
        ):
            self.assertIn(relative_path, readme)
            self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())


if __name__ == "__main__":
    unittest.main()
