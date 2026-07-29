"""Pure unit tests for the README visual evidence renderer."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
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
            visuals.subprocess,
            "run",
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
                call.kwargs["env"],
                {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(call.kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(call.kwargs["stdout"], subprocess.PIPE)
            self.assertEqual(call.kwargs["stderr"], subprocess.PIPE)
            self.assertFalse(call.kwargs["check"])
            self.assertEqual(
                call.kwargs["timeout"],
                visuals.EXAMPLE_TIMEOUT_SECONDS,
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
                visuals.subprocess, "run", side_effect=results
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
            "generator": {"recorded": True},
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


class SourceProvenanceTests(unittest.TestCase):
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


class CommittedSoftmaxEvidenceTests(unittest.TestCase):
    def test_softmax_bundle_is_complete_safe_and_manifest_bound(self) -> None:
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
            manifest["schema"], "tensorkiln.readme-visual-evidence.v2"
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
            manifest["repository_source"]["selection"],
            "latest commit touching the complete evidence build-input set",
        )
        self.assertIn(
            "examples/execute_softmax.cpp",
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
                hashlib.sha256(payload).hexdigest(), record["sha256"]
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
            "docs/visuals/generated/manifest.json",
        ):
            self.assertIn(relative_path, readme)
            self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())


if __name__ == "__main__":
    unittest.main()
