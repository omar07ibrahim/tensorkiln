"""Pure tests for the GCC/LCOV evidence recorder and committed bundle."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import record_coverage as coverage  # noqa: E402


SAMPLE_TRACE = """\
TN:
SF:src/arena.cpp
FN:1,2,_Z_first
FNDA:3,_Z_first
FNF:1
FNH:1
BRDA:1,0,0,3
BRDA:1,e0,1,-
BRF:2
BRH:1
DA:1,3
DA:2,0
LF:2
LH:1
end_of_record
TN:
SF:src/shape.cpp
FN:3,4,_Z_second
FNDA:0,_Z_second
FNF:1
FNH:0
DA:3,2
LF:1
LH:1
end_of_record
"""


def sample_transcript() -> str:
    return """\
=== source graph ===
=== verified interval arena plan ===
=== verified dense execution plan ===
=== verified Softmax execution ===
[pass] first
[pass] second

2/2 tests passed
"""


class TraceParserTests(unittest.TestCase):
    def test_parser_recomputes_all_metrics_and_normalizes_order(self) -> None:
        parsed = coverage.parse_lcov_trace(
            SAMPLE_TRACE, require_all_production_units=False
        )

        self.assertEqual(parsed.lines, coverage.Metric(2, 3))
        self.assertEqual(parsed.functions, coverage.Metric(1, 2))
        self.assertEqual(parsed.branches, coverage.Metric(1, 2))
        self.assertEqual(
            [record.source for record in parsed.files],
            ["src/arena.cpp", "src/shape.cpp"],
        )
        self.assertNotIn(str(REPOSITORY_ROOT), parsed.canonical_text)
        self.assertEqual(
            coverage.parse_lcov_trace(
                parsed.canonical_text, require_all_production_units=False
            ).canonical_text,
            parsed.canonical_text,
        )

    def test_parser_normalizes_a_repository_absolute_source(self) -> None:
        absolute = SAMPLE_TRACE.replace(
            "SF:src/arena.cpp",
            f"SF:{REPOSITORY_ROOT / 'src/arena.cpp'}",
        )
        parsed = coverage.parse_lcov_trace(
            absolute, require_all_production_units=False
        )

        self.assertIn("SF:src/arena.cpp", parsed.canonical_text)
        self.assertNotIn(str(REPOSITORY_ROOT), parsed.canonical_text)

    def test_parser_rejects_summary_mismatch_and_duplicate_counts(self) -> None:
        mismatches = (
            SAMPLE_TRACE.replace("LH:1", "LH:2", 1),
            SAMPLE_TRACE.replace("FNH:1", "FNH:0", 1),
            SAMPLE_TRACE.replace("BRH:1", "BRH:2", 1),
            SAMPLE_TRACE.replace("DA:2,0", "DA:1,0", 1),
            SAMPLE_TRACE.replace("FNDA:3,_Z_first", "FNDA:3,_Z_other", 1),
        )
        for malformed in mismatches:
            with self.subTest(trace_sha256=coverage._sha256(malformed)):
                with self.assertRaises(coverage.CoverageEvidenceError):
                    coverage.parse_lcov_trace(
                        malformed, require_all_production_units=False
                    )

    def test_parser_rejects_repository_escape_and_duplicate_sources(self) -> None:
        outside = SAMPLE_TRACE.replace(
            "SF:src/arena.cpp", "SF:../outside.cpp", 1
        )
        duplicate = SAMPLE_TRACE.replace(
            "SF:src/shape.cpp", "SF:src/arena.cpp", 1
        )
        for malformed in (outside, duplicate):
            with self.subTest(trace_sha256=coverage._sha256(malformed)):
                with self.assertRaises(coverage.CoverageEvidenceError):
                    coverage.parse_lcov_trace(
                        malformed, require_all_production_units=False
                    )

    def test_unexecuted_branch_is_found_but_not_covered(self) -> None:
        parsed = coverage.parse_lcov_trace(
            SAMPLE_TRACE, require_all_production_units=False
        )

        self.assertEqual(parsed.files[0].branches, coverage.Metric(1, 2))
        self.assertIn("BRDA:1,e0,1,-", parsed.canonical_text)

    def test_lcov_summary_is_independently_cross_checked(self) -> None:
        parsed = coverage.parse_lcov_trace(
            SAMPLE_TRACE, require_all_production_units=False
        )
        valid = """\
Summary coverage rate:
  lines......: 66.7% (2 of 3 lines)
  functions..: 50.0% (1 of 2 functions)
  branches...: 50.0% (1 of 2 branches)
"""
        coverage._validate_lcov_summary(valid, parsed)

        with self.assertRaisesRegex(
            coverage.CoverageEvidenceError, "independent lines"
        ):
            coverage._validate_lcov_summary(
                valid.replace("(2 of 3 lines)", "(3 of 3 lines)"), parsed
            )

    def test_numeric_parsers_fail_closed_on_hostile_magnitudes(self) -> None:
        huge = "9" * 5000
        with self.assertRaises(coverage.CoverageEvidenceError):
            coverage._parse_nonnegative_integer(huge, "test count")
        for malformed in ("1..2", huge):
            with self.subTest(malformed_sha256=coverage._sha256(malformed)):
                with self.assertRaises(coverage.CoverageEvidenceError):
                    coverage._parse_percentage(
                        malformed,
                        "test percentage",
                    )
        parsed = coverage.parse_lcov_trace(
            SAMPLE_TRACE,
            require_all_production_units=False,
        )
        hostile_summary = """\
Summary coverage rate:
  lines......: 66.7% (2 of 3 lines)
  functions..: 50.0% (1 of 2 functions)
  branches...: 50.0% (1 of 2 branches)
""".replace("(2 of 3 lines)", f"({huge} of 3 lines)")
        with self.assertRaises(coverage.CoverageEvidenceError):
            coverage._validate_lcov_summary(hostile_summary, parsed)


class TranscriptTests(unittest.TestCase):
    def test_transcript_requires_examples_and_exact_test_count(self) -> None:
        transcript, count = coverage.validate_test_transcript(
            sample_transcript()
        )

        self.assertEqual(count, 2)
        self.assertTrue(
            transcript.startswith(
                "$ make -s -j2 CXX=g++ PROFILE=coverage test\n"
            )
        )
        self.assertTrue(transcript.endswith("2/2 tests passed\n"))

    def test_transcript_records_nondefault_compiler_and_jobs(self) -> None:
        transcript, count = coverage.validate_test_transcript(
            sample_transcript(),
            jobs=3,
            compiler_label="g++-13",
        )

        self.assertEqual(count, 2)
        self.assertTrue(
            transcript.startswith(
                "$ make -s -j3 CXX=g++-13 PROFILE=coverage test\n"
            )
        )

    def test_transcript_rejects_missing_or_inconsistent_evidence(self) -> None:
        malformed = (
            sample_transcript().replace("=== source graph ===\n", ""),
            sample_transcript().replace("2/2 tests passed", "1/2 tests passed"),
            sample_transcript().replace("[pass] second\n", ""),
            sample_transcript().replace(
                "[pass] first", "owner=someone" + "@" + "example.com"
            ),
        )
        for text in malformed:
            with self.subTest(transcript_sha256=coverage._sha256(text)):
                with self.assertRaises(coverage.CoverageEvidenceError):
                    coverage.validate_test_transcript(text)

        huge = "9" * 5000
        with self.assertRaises(coverage.CoverageEvidenceError):
            coverage.validate_test_transcript(
                sample_transcript().replace(
                    "2/2 tests passed",
                    f"{huge}/{huge} tests passed",
                )
            )


class RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = coverage.parse_lcov_trace(
            SAMPLE_TRACE, require_all_production_units=False
        )

    def test_summary_svg_is_deterministic_safe_and_data_derived(self) -> None:
        arguments = {
            "test_count": 2,
            "compiler_short": "GCC 13.3.0",
            "lcov_short": "LCOV 2.0-1",
            "source_digest": "a" * 64,
        }
        first = coverage.render_summary_svg(self.trace, **arguments)
        second = coverage.render_summary_svg(self.trace, **arguments)

        self.assertEqual(first, second)
        self.assertIn("66.7%", first)
        self.assertIn("1 / 2", first)
        self.assertIn("arena.cpp", first)
        self.assertIn("INDEPENDENT RECOUNT", first)
        self.assertIn("NOT A QUALITY SCORE", first)
        coverage.reject_unsafe_public_text("test SVG", first)
        root = ElementTree.fromstring(first)
        self.assertEqual(root.attrib["viewBox"], "0 0 1200 820")
        self.assertNotIn("<script", first.lower())
        self.assertNotIn("<image", first.lower())
        self.assertNotIn(" href=", first.lower())

    def test_summary_text_states_scope_and_claim_boundary(self) -> None:
        text = coverage.render_summary_text(
            self.trace,
            test_count=2,
            compiler_version="g++ 13.3.0",
            lcov_version="lcov: LCOV version 2.0-1",
        )

        self.assertIn("lines: 66.7% (2/3)", text)
        self.assertIn("src/arena.cpp: 1 uncovered", text)
        self.assertIn("not a benchmark", text)
        self.assertIn("compiler-generated exception paths", text)

    def test_public_text_guard_rejects_identity_and_secret_shapes(self) -> None:
        unsafe_values = (
            "path=/home/person/project",
            r"path=C:\Users\person\project",
            "owner=someone" + "@" + "example.com",
            "key=" + "AK" + "IA" + "A" * 16,
            "key=" + "gh" + "p_" + "a" * 26,
            "-----BEGIN " + "PRIVATE KEY-----",
        )
        for value in unsafe_values:
            with self.subTest(value_sha256=coverage._sha256(value)):
                with self.assertRaises(coverage.CoverageEvidenceError):
                    coverage.reject_unsafe_public_text("test", value)


class SourceSnapshotTests(unittest.TestCase):
    def _isolated_snapshot(
        self,
        *,
        extra_files: dict[str, str],
        tracked_extra: tuple[str, ...] = (),
    ) -> dict[str, object]:
        build = REPOSITORY_ROOT / "build"
        build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="coverage-snapshot-test-",
            dir=build,
        ) as raw:
            root = Path(raw) / "repository"
            for directory in (
                "examples",
                "include",
                "src",
                "tests",
                "tools",
            ):
                (root / directory).mkdir(parents=True, exist_ok=True)
            baseline = (
                "Makefile",
                "tests/test_coverage_evidence.py",
                "tools/record_coverage.py",
            )
            for relative_path in baseline:
                (root / relative_path).write_text(
                    f"{relative_path}\n",
                    encoding="utf-8",
                )
            for relative_path, payload in extra_files.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload, encoding="utf-8")
            tracked_paths = baseline + tracked_extra

            def fake_git(arguments: tuple[str, ...]) -> bytes:
                command = arguments[0]
                if command == "ls-files":
                    return (
                        "\0".join(tracked_paths) + "\0"
                    ).encode("utf-8")
                if command == "hash-object":
                    return b"a" * 40 + b"\n"
                if command == "log":
                    return b"b" * 40 + b"\n"
                if command == "show":
                    return b"c" * 40 + b"\n"
                if command == "status":
                    return b""
                raise AssertionError(f"unexpected Git command: {command}")

            with (
                mock.patch.object(coverage, "REPOSITORY_ROOT", root),
                mock.patch.object(
                    coverage,
                    "_git_output",
                    side_effect=fake_git,
                ),
            ):
                return coverage.collect_source_snapshot()

    def test_snapshot_hashes_every_direct_evidence_input(self) -> None:
        snapshot = coverage.collect_source_snapshot()
        files = snapshot["files"]

        self.assertIn("Makefile", files)
        self.assertIn("src/execution.cpp", files)
        self.assertIn("include/tensorkiln/execution.hpp", files)
        self.assertIn("tests/test_execution.cpp", files)
        self.assertIn("tests/test_coverage_evidence.py", files)
        self.assertIn("tests/test_readme_visuals.py", files)
        self.assertIn("tools/record_coverage.py", files)
        self.assertRegex(snapshot["sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(snapshot["source_commit"], r"^[0-9a-f]{40,64}$")
        self.assertRegex(snapshot["source_tree"], r"^[0-9a-f]{40,64}$")
        self.assertEqual(
            snapshot["selection"],
            "latest commit touching the direct evidence-input set",
        )
        self.assertIs(type(snapshot["selected_paths_clean"]), bool)
        self.assertEqual(
            snapshot["commit_bound"], snapshot["selected_paths_clean"]
        )
        for record in files.values():
            self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(record["git_blob"], r"^[0-9a-f]{40,64}$")

    def test_untracked_and_ignored_input_files_force_an_unbound_snapshot(
        self,
    ) -> None:
        snapshot = self._isolated_snapshot(
            extra_files={
                "include/vector": "shadow candidate\n",
                "include/coverage-probe.gcda": "ignored candidate\n",
            }
        )

        self.assertIn("include/vector", snapshot["files"])
        self.assertIn("include/coverage-probe.gcda", snapshot["files"])
        self.assertFalse(snapshot["selected_paths_clean"])
        self.assertFalse(snapshot["commit_bound"])

    def test_deleted_tracked_input_forces_an_unbound_snapshot(self) -> None:
        snapshot = self._isolated_snapshot(
            extra_files={},
            tracked_extra=("include/deleted.hpp",),
        )

        self.assertNotIn("include/deleted.hpp", snapshot["files"])
        self.assertFalse(snapshot["selected_paths_clean"])
        self.assertFalse(snapshot["commit_bound"])


class FilesystemSafetyTests(unittest.TestCase):
    def _workspace(self) -> tempfile.TemporaryDirectory[str]:
        build = REPOSITORY_ROOT / "build"
        build.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(
            prefix="coverage-recorder-test-",
            dir=build,
        )

    def test_clean_rejects_symlinked_build_parents_without_deleting_target(
        self,
    ) -> None:
        for symlink_level in ("build", "compiler"):
            with self.subTest(symlink_level=symlink_level), self._workspace() as raw:
                root = Path(raw) / "repository"
                outside = Path(raw) / "outside"
                root.mkdir()
                outside.mkdir()
                sentinel = outside / "sentinel.txt"
                sentinel.write_text("keep\n", encoding="utf-8")
                if symlink_level == "build":
                    (root / "build").symlink_to(
                        outside,
                        target_is_directory=True,
                    )
                else:
                    (root / "build").mkdir()
                    (root / "build/g++").symlink_to(
                        outside,
                        target_is_directory=True,
                    )

                with mock.patch.object(
                    coverage,
                    "REPOSITORY_ROOT",
                    root,
                ):
                    with self.assertRaises(coverage.CoverageEvidenceError):
                        coverage._clean_coverage_build("g++")

                self.assertEqual(
                    sentinel.read_text(encoding="utf-8"),
                    "keep\n",
                )

    def test_publication_rejects_a_predictable_temporary_symlink(
        self,
    ) -> None:
        with self._workspace() as raw:
            root = Path(raw) / "repository"
            output = root / "docs/coverage/generated"
            outside = Path(raw) / "outside.txt"
            output.mkdir(parents=True)
            outside.write_text("keep\n", encoding="utf-8")
            generated = {
                name: f"{name}\n".encode()
                for name in coverage.ARTIFACT_NAMES
            }
            for name, payload in generated.items():
                (output / name).write_bytes(payload)
            (output / ".coverage.info.tmp").symlink_to(outside)

            with mock.patch.object(
                coverage,
                "REPOSITORY_ROOT",
                root,
            ):
                with self.assertRaises(coverage.CoverageEvidenceError):
                    coverage._publish_or_check(
                        output,
                        generated,
                        check=False,
                    )

            self.assertEqual(outside.read_text(encoding="utf-8"), "keep\n")

    def test_publication_uses_manifest_as_the_final_integrity_marker(
        self,
    ) -> None:
        with self._workspace() as raw:
            root = Path(raw) / "repository"
            output = root / "docs/coverage/generated"
            output.parent.mkdir(parents=True)
            generated = {
                name: f"{name}\n".encode()
                for name in coverage.ARTIFACT_NAMES
            }

            with mock.patch.object(
                coverage,
                "REPOSITORY_ROOT",
                root,
            ):
                coverage._publish_or_check(
                    output,
                    generated,
                    check=False,
                )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                set(coverage.ARTIFACT_NAMES),
            )
            for name, payload in generated.items():
                self.assertEqual((output / name).read_bytes(), payload)


class CommittedEvidenceTests(unittest.TestCase):
    def test_bundle_is_complete_safe_and_internally_consistent(self) -> None:
        output_dir = REPOSITORY_ROOT / "docs/coverage/generated"
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["schema"], "tensorkiln.coverage-evidence.v1"
        )
        self.assertEqual(manifest["trace"]["scope"], "src/")
        self.assertEqual(
            manifest["source_snapshot"], coverage.collect_source_snapshot()
        )
        self.assertEqual(
            set(path.name for path in output_dir.iterdir()),
            set(coverage.ARTIFACT_NAMES),
        )
        self.assertEqual(
            set(manifest["artifacts"]),
            set(coverage.ARTIFACT_NAMES).difference({"manifest.json"}),
        )
        for name, record in manifest["artifacts"].items():
            payload = (output_dir / name).read_bytes()
            self.assertEqual(len(payload), record["bytes"])
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(), record["sha256"]
            )

        trace_path = output_dir / "coverage.info"
        trace_text = trace_path.read_text(encoding="utf-8")
        trace = coverage.parse_lcov_trace(trace_text)
        self.assertEqual(trace.canonical_text, trace_text)
        self.assertEqual(
            manifest["metrics"]["lines"],
            {
                "covered": trace.lines.covered,
                "percentage": round(trace.lines.percentage, 1),
                "total": trace.lines.total,
            },
        )
        self.assertEqual(
            manifest["metrics"]["functions"]["covered"],
            trace.functions.covered,
        )
        self.assertEqual(
            manifest["metrics"]["branches"]["total"],
            trace.branches.total,
        )
        for record in trace.files:
            self.assertTrue(record.source.startswith("src/"))

        for name in coverage.ARTIFACT_NAMES:
            text = (output_dir / name).read_text(encoding="utf-8")
            coverage.reject_unsafe_public_text(name, text)

        transcript_text = (output_dir / "test-run.txt").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            transcript_text.splitlines()[0],
            f"$ {manifest['capture']['command']}",
        )
        transcript, tests = coverage.validate_test_transcript(
            transcript_text.split("\n", 1)[1],
            jobs=manifest["capture"]["jobs"],
            compiler_label=manifest["tools"]["cxx"]["executable"],
        )
        self.assertEqual(
            transcript,
            transcript_text,
        )
        self.assertEqual(tests, manifest["capture"]["test_cases_passed"])

        svg = (output_dir / "summary.svg").read_text(encoding="utf-8")
        ElementTree.fromstring(svg)
        self.assertIn(f"{trace.lines.percentage:.1f}%", svg)
        self.assertIn(f"{trace.functions.percentage:.1f}%", svg)
        self.assertIn(f"{trace.branches.percentage:.1f}%", svg)
        self.assertNotIn("<script", svg.lower())
        self.assertNotIn("<image", svg.lower())
        self.assertNotIn(" href=", svg.lower())

    def test_documentation_links_every_public_coverage_artifact(self) -> None:
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        methodology = (
            REPOSITORY_ROOT / "docs/coverage/README.md"
        ).read_text(encoding="utf-8")

        for relative_path in (
            "docs/coverage/generated/summary.svg",
            "docs/coverage/generated/summary.txt",
            "docs/coverage/generated/coverage.info",
            "docs/coverage/generated/test-run.txt",
            "docs/coverage/generated/manifest.json",
        ):
            self.assertIn(relative_path, readme + methodology)
            self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())
        for phrase in (
            "not a release gate",
            "compiler-generated",
            "tests and examples are excluded",
            "make COVERAGE_JOBS=2 coverage",
        ):
            self.assertIn(phrase, methodology)

        trace = coverage.parse_lcov_trace(
            (
                REPOSITORY_ROOT
                / "docs/coverage/generated/coverage.info"
            ).read_text(encoding="utf-8")
        )
        for expected in (
            (
                f"{trace.lines.covered}/{trace.lines.total} lines "
                f"({trace.lines.percentage:.1f}%)"
            ),
            (
                f"{trace.functions.covered}/{trace.functions.total} "
                f"functions ({trace.functions.percentage:.1f}%)"
            ),
            (
                f"{trace.branches.covered}/{trace.branches.total} GCC "
                f"branch edges ({trace.branches.percentage:.1f}%)"
            ),
        ):
            self.assertIn(expected, readme)


if __name__ == "__main__":
    unittest.main()
