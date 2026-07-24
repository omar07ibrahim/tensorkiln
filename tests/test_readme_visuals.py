"""Pure unit tests for the README visual evidence renderer."""

from __future__ import annotations

import hashlib
import json
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
            raise AssertionError(f"unexpected binary: {binary_name}")

        with mock.patch.object(
            visuals, "run_release_example", side_effect=fake_example
        ):
            artifacts = visuals.render_visuals(Path("unused-release-dir"))

        self.assertEqual(artifacts["arena-plan.txt"], PLAN_STDOUT)
        self.assertEqual(artifacts["execute-graph.txt"], EXECUTE_STDOUT)
        manifest = json.loads(artifacts["manifest.json"])
        self.assertEqual(
            manifest["sources"]["execute_graph"]["stdout_sha256"],
            hashlib.sha256(EXECUTE_STDOUT.encode()).hexdigest(),
        )
        for filename, record in manifest["artifacts"].items():
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(artifacts[filename].encode()).hexdigest(),
            )
        self.assertNotIn("/home/", artifacts["manifest.json"])


if __name__ == "__main__":
    unittest.main()
