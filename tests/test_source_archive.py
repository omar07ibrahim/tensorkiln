#!/usr/bin/env python3
"""Pure safety tests for the committed-source archive gate."""

from __future__ import annotations

import hashlib
import io
import os
import signal
import stat
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import verify_source_archive as archive_gate  # noqa: E402


def process_state(process_id: int) -> str | None:
    try:
        payload = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    return payload.rsplit(") ", maxsplit=1)[1].split(maxsplit=1)[0]


def inventory_for(
    files: dict[str, tuple[bytes, str]],
) -> tuple[archive_gate.TrackedFile, ...]:
    return tuple(
        archive_gate.TrackedFile(
            path=path,
            mode=mode,
            object_id=archive_gate.git_blob_object_id(payload, "sha1"),
        )
        for path, (payload, mode) in sorted(files.items())
    )


def _tar_info(name: str, *, mode: int, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode = mode
    info.size = size
    info.mtime = 1
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def render_archive(
    files: dict[str, tuple[bytes, str]],
    *,
    extra_members: tuple[tuple[tarfile.TarInfo, bytes | None], ...] = (),
    duplicate_file: str | None = None,
    mode_overrides: dict[str, int] | None = None,
) -> bytes:
    mode_overrides = mode_overrides or {}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as output:
        directories = archive_gate._expected_archive_directories(files)
        for directory in sorted(
            directories, key=lambda value: (value.count("/"), value)
        ):
            directory_info = _tar_info(
                directory + "/",
                mode=archive_gate.ARCHIVE_DIRECTORY_MODE,
            )
            directory_info.type = tarfile.DIRTYPE
            output.addfile(directory_info)
        for path, (payload, git_mode) in sorted(files.items()):
            mode = mode_overrides.get(
                path,
                archive_gate.ARCHIVE_EXECUTABLE_MODE
                if git_mode == "100755"
                else archive_gate.ARCHIVE_FILE_MODE,
            )
            info = _tar_info(
                archive_gate.ARCHIVE_PREFIX + path,
                mode=mode,
                size=len(payload),
            )
            output.addfile(info, io.BytesIO(payload))
            if duplicate_file == path:
                output.addfile(info, io.BytesIO(payload))
        for info, payload in extra_members:
            output.addfile(info, None if payload is None else io.BytesIO(payload))
    return stream.getvalue()


class ProcessBoundaryTests(unittest.TestCase):
    def assert_process_stopped(self, process_id: int) -> None:
        deadline = time.monotonic() + 1.0
        state = process_state(process_id)
        while state not in {None, "Z"} and time.monotonic() < deadline:
            time.sleep(0.01)
            state = process_state(process_id)
        self.assertIn(state, {None, "Z"})

    @staticmethod
    def stop_process_if_running(process_id: int | None) -> None:
        if process_id is None or process_state(process_id) in {None, "Z"}:
            return
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def test_selector_setup_failure_still_kills_the_process(self) -> None:
        archive_gate.BUILD_ROOT.mkdir(exist_ok=True)
        process_id: int | None = None
        with tempfile.TemporaryDirectory(
            prefix="source-archive-process-", dir=archive_gate.BUILD_ROOT
        ) as temporary:
            marker = Path(temporary) / "leader.pid"

            class FailingSelector:
                def __init__(self) -> None:
                    self.closed = False

                def register(self, _descriptor: int, _events: int) -> None:
                    deadline = time.monotonic() + 1.0
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    raise OSError("forced selector failure")

                def close(self) -> None:
                    self.closed = True

            selector = FailingSelector()
            program = "\n".join(
                (
                    "import os",
                    "import pathlib",
                    "import time",
                    f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))",
                    "time.sleep(60)",
                )
            )
            try:
                with (
                    mock.patch.object(
                        archive_gate.selectors,
                        "DefaultSelector",
                        return_value=selector,
                    ),
                    self.assertRaisesRegex(
                        archive_gate.SourceArchiveError, "could not monitor"
                    ),
                ):
                    archive_gate.run_bounded(
                        (sys.executable, "-I", "-c", program),
                        cwd=Path(temporary),
                        environment=archive_gate._minimal_environment(),
                        timeout_seconds=2,
                        max_stdout_bytes=1024,
                        max_stderr_bytes=1024,
                        label="selector fixture",
                    )
                self.assertTrue(selector.closed)
                process_id = int(marker.read_text(encoding="ascii"))
                self.assert_process_stopped(process_id)
            finally:
                self.stop_process_if_running(process_id)

    def test_timeout_kills_a_descendant_after_its_leader_exits(self) -> None:
        archive_gate.BUILD_ROOT.mkdir(exist_ok=True)
        child_id: int | None = None
        with tempfile.TemporaryDirectory(
            prefix="source-archive-process-", dir=archive_gate.BUILD_ROOT
        ) as temporary:
            marker = Path(temporary) / "child.pid"
            program = "\n".join(
                (
                    "import os",
                    "import pathlib",
                    "import time",
                    "child = os.fork()",
                    "if child == 0:",
                    "    time.sleep(60)",
                    "else:",
                    f"    pathlib.Path({str(marker)!r}).write_text(str(child))",
                    "    os._exit(0)",
                )
            )
            try:
                with self.assertRaisesRegex(
                    archive_gate.SourceArchiveError, "exceeded"
                ):
                    archive_gate.run_bounded(
                        (sys.executable, "-I", "-c", program),
                        cwd=Path(temporary),
                        environment=archive_gate._minimal_environment(),
                        timeout_seconds=1,
                        max_stdout_bytes=1024,
                        max_stderr_bytes=1024,
                        label="forked fixture",
                    )
                child_id = int(marker.read_text(encoding="ascii"))
                self.assert_process_stopped(child_id)
            finally:
                self.stop_process_if_running(child_id)

    def test_success_kills_a_descendant_that_closes_both_streams(self) -> None:
        archive_gate.BUILD_ROOT.mkdir(exist_ok=True)
        child_id: int | None = None
        with tempfile.TemporaryDirectory(
            prefix="source-archive-process-", dir=archive_gate.BUILD_ROOT
        ) as temporary:
            marker = Path(temporary) / "child.pid"
            program = "\n".join(
                (
                    "import os",
                    "import pathlib",
                    "import time",
                    "child = os.fork()",
                    "if child == 0:",
                    "    os.close(1)",
                    "    os.close(2)",
                    "    time.sleep(60)",
                    "else:",
                    f"    pathlib.Path({str(marker)!r}).write_text(str(child))",
                    "    os._exit(0)",
                )
            )
            try:
                result = archive_gate.run_bounded(
                    (sys.executable, "-I", "-c", program),
                    cwd=Path(temporary),
                    environment=archive_gate._minimal_environment(),
                    timeout_seconds=2,
                    max_stdout_bytes=1024,
                    max_stderr_bytes=1024,
                    label="detached-stream fixture",
                )
                self.assertEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertEqual(result.stderr, b"")
                child_id = int(marker.read_text(encoding="ascii"))
                self.assert_process_stopped(child_id)
            finally:
                self.stop_process_if_running(child_id)


class InventoryTests(unittest.TestCase):
    def test_parses_regular_and_executable_blobs(self) -> None:
        first = b"alpha\n"
        second = b"#!/bin/sh\nexit 0\n"
        payload = (
            f"100644 blob {archive_gate.git_blob_object_id(first, 'sha1')}"
            "\tdocs/a.txt\0"
            f"100755 blob {archive_gate.git_blob_object_id(second, 'sha1')}"
            "\ttools/run.sh\0"
        ).encode("ascii")

        parsed = archive_gate.parse_tracked_inventory(payload, "sha1")

        self.assertEqual(
            [entry.path for entry in parsed], ["docs/a.txt", "tools/run.sh"]
        )
        self.assertEqual([entry.mode for entry in parsed], ["100644", "100755"])

    def test_rejects_links_submodules_duplicates_and_unsafe_paths(self) -> None:
        object_id = "a" * 40
        cases = (
            f"120000 blob {object_id}\tlink\0",
            f"160000 commit {object_id}\tdependency\0",
            (
                f"100644 blob {object_id}\tdocs/a\0"
                f"100644 blob {object_id}\tdocs/a\0"
            ),
            f"100644 blob {object_id}\t../escape\0",
            f"100644 blob {object_id}\tbuild/output.o\0",
            f"100644 blob {object_id}\tdocs/.git/config\0",
        )
        for payload in cases:
            digest = hashlib.sha256(payload.encode()).hexdigest()
            with self.subTest(payload_sha256=digest):
                with self.assertRaises(archive_gate.SourceArchiveError):
                    archive_gate.parse_tracked_inventory(
                        payload.encode("ascii"), "sha1"
                    )


class ArchiveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.files = {
            "Makefile": (b"cli:\n\t@true\n", "100644"),
            "cli/main.cpp": (b"int main() { return 0; }\n", "100644"),
            "tools/check": (b"#!/bin/sh\nexit 0\n", "100755"),
        }
        self.inventory = inventory_for(self.files)

    def test_accepts_exact_inventory_modes_and_git_blob_payloads(self) -> None:
        payload = render_archive(self.files)

        validated = archive_gate.validate_archive(
            payload, self.inventory, "sha1"
        )

        self.assertEqual(
            [entry.path for entry in validated.files], sorted(self.files)
        )
        self.assertEqual(
            validated.regular_bytes,
            sum(len(value[0]) for value in self.files.values()),
        )
        self.assertEqual(validated.sha256, hashlib.sha256(payload).hexdigest())
        self.assertGreater(validated.member_count, len(validated.files))

    def test_rejects_traversal_and_absolute_members(self) -> None:
        for name in (
            archive_gate.ARCHIVE_PREFIX + "../escape",
            "/absolute/file",
            archive_gate.ARCHIVE_PREFIX + "docs/../../escape",
        ):
            info = _tar_info(name, mode=archive_gate.ARCHIVE_FILE_MODE, size=1)
            payload = render_archive(
                self.files, extra_members=((info, b"x"),)
            )
            with self.subTest(name=name), self.assertRaises(
                archive_gate.SourceArchiveError
            ):
                archive_gate.validate_archive(payload, self.inventory, "sha1")

    def test_rejects_symbolic_hard_device_and_fifo_members(self) -> None:
        members: list[tarfile.TarInfo] = []
        symbolic = _tar_info(
            archive_gate.ARCHIVE_PREFIX + "extra-symlink",
            mode=archive_gate.ARCHIVE_FILE_MODE,
        )
        symbolic.type = tarfile.SYMTYPE
        symbolic.linkname = "Makefile"
        members.append(symbolic)
        hard = _tar_info(
            archive_gate.ARCHIVE_PREFIX + "extra-hardlink",
            mode=archive_gate.ARCHIVE_FILE_MODE,
        )
        hard.type = tarfile.LNKTYPE
        hard.linkname = archive_gate.ARCHIVE_PREFIX + "Makefile"
        members.append(hard)
        for member_type, name in (
            (tarfile.CHRTYPE, "device"),
            (tarfile.FIFOTYPE, "fifo"),
        ):
            special = _tar_info(
                archive_gate.ARCHIVE_PREFIX + name,
                mode=archive_gate.ARCHIVE_FILE_MODE,
            )
            special.type = member_type
            members.append(special)

        for member in members:
            payload = render_archive(
                self.files, extra_members=((member, None),)
            )
            with self.subTest(member=member.name), self.assertRaises(
                archive_gate.SourceArchiveError
            ):
                archive_gate.validate_archive(payload, self.inventory, "sha1")

    def test_rejects_duplicate_member_names(self) -> None:
        payload = render_archive(self.files, duplicate_file="Makefile")

        with self.assertRaisesRegex(
            archive_gate.SourceArchiveError, "duplicate"
        ):
            archive_gate.validate_archive(payload, self.inventory, "sha1")

    def test_rejects_mode_blob_and_inventory_mismatches(self) -> None:
        wrong_mode = render_archive(
            self.files,
            mode_overrides={"Makefile": archive_gate.ARCHIVE_EXECUTABLE_MODE},
        )
        with self.assertRaisesRegex(archive_gate.SourceArchiveError, "mode"):
            archive_gate.validate_archive(wrong_mode, self.inventory, "sha1")

        tampered_files = dict(self.files)
        tampered_files["Makefile"] = (b"tampered\n", "100644")
        tampered = render_archive(tampered_files)
        with self.assertRaisesRegex(archive_gate.SourceArchiveError, "blob"):
            archive_gate.validate_archive(tampered, self.inventory, "sha1")

        omitted_files = dict(self.files)
        del omitted_files["Makefile"]
        omitted = render_archive(omitted_files)
        with self.assertRaisesRegex(
            archive_gate.SourceArchiveError, "omits"
        ):
            archive_gate.validate_archive(omitted, self.inventory, "sha1")

    def test_rejects_archive_member_and_total_byte_limits(self) -> None:
        payload = render_archive(self.files)
        with (
            mock.patch.object(archive_gate, "MAX_ARCHIVE_BYTES", len(payload) - 1),
            self.assertRaisesRegex(archive_gate.SourceArchiveError, "archive.*limit"),
        ):
            archive_gate.validate_archive(payload, self.inventory, "sha1")

        largest = max(len(value[0]) for value in self.files.values())
        with (
            mock.patch.object(archive_gate, "MAX_MEMBER_BYTES", largest - 1),
            self.assertRaisesRegex(archive_gate.SourceArchiveError, "member.*limit"),
        ):
            archive_gate.validate_archive(payload, self.inventory, "sha1")

        total = sum(len(value[0]) for value in self.files.values())
        with (
            mock.patch.object(archive_gate, "MAX_TRACKED_BYTES", total - 1),
            self.assertRaisesRegex(
                archive_gate.SourceArchiveError, "regular files.*limit"
            ),
        ):
            archive_gate.validate_archive(payload, self.inventory, "sha1")

        directory_count = len(archive_gate._expected_archive_directories(self.files))
        with (
            mock.patch.object(
                archive_gate,
                "MAX_ARCHIVE_MEMBERS",
                directory_count + len(self.files) - 1,
            ),
            self.assertRaisesRegex(archive_gate.SourceArchiveError, "member limit"),
        ):
            archive_gate.validate_archive(payload, self.inventory, "sha1")


class ExtractionTests(unittest.TestCase):
    def test_extracts_only_regular_payloads_with_normalized_modes(self) -> None:
        files = {
            "docs/readme.txt": (b"safe fixture\n", "100644"),
            "tools/check": (b"#!/bin/sh\nexit 0\n", "100755"),
        }
        validated = archive_gate.validate_archive(
            render_archive(files), inventory_for(files), "sha1"
        )
        archive_gate.BUILD_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="source-archive-unit-", dir=archive_gate.BUILD_ROOT
        ) as temporary:
            root = Path(temporary)
            extracted = archive_gate.extract_validated_archive(validated, root)

            for path, (expected, mode) in files.items():
                target = extracted.joinpath(*PurePosixPath(path).parts)
                self.assertEqual(target.read_bytes(), expected)
                self.assertFalse(target.is_symlink())
                expected_mode = 0o755 if mode == "100755" else 0o644
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), expected_mode)

    def test_refuses_to_clobber_an_existing_target(self) -> None:
        files = {"Makefile": (b"safe fixture\n", "100644")}
        validated = archive_gate.validate_archive(
            render_archive(files), inventory_for(files), "sha1"
        )
        archive_gate.BUILD_ROOT.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="source-archive-unit-", dir=archive_gate.BUILD_ROOT
        ) as temporary:
            root = Path(temporary)
            archive_gate.extract_validated_archive(validated, root)

            with self.assertRaisesRegex(
                archive_gate.SourceArchiveError, "clobber"
            ):
                archive_gate.extract_validated_archive(validated, root)


class ContractTests(unittest.TestCase):
    def test_execution_contract_requires_write_audit_and_reference_match(self) -> None:
        valid = {
            "schema": "tensorkiln.cli.execute.v1",
            "workload": {"id": "reglu_mlp_v1"},
            "execution": {
                "run_status": "success",
                "kernel_write_audit": True,
                "benchmark": False,
                "reference_check": {"status": "match"},
            },
        }
        archive_gate._verify_execution_report(valid, "reglu_mlp_v1", "fixture")

        for field, value in (
            ("kernel_write_audit", False),
            ("benchmark", True),
            ("run_status", "failed"),
        ):
            malformed = {
                **valid,
                "execution": {**valid["execution"], field: value},
            }
            with self.subTest(field=field), self.assertRaises(
                archive_gate.SourceArchiveError
            ):
                archive_gate._verify_execution_report(
                    malformed, "reglu_mlp_v1", "fixture"
                )

        wrong_reference = {
            **valid,
            "execution": {
                **valid["execution"],
                "reference_check": {"status": "mismatch"},
            },
        }
        with self.assertRaises(archive_gate.SourceArchiveError):
            archive_gate._verify_execution_report(
                wrong_reference, "reglu_mlp_v1", "fixture"
            )

    def test_reglu_contract_checks_stats_output_bits_and_eight_matches(self) -> None:
        inspect = {
            "schema": "tensorkiln.cli.inspect.v1",
            "workload": {"id": "reglu_mlp_v1"},
            "plan": {"stats": archive_gate.REGLU_PLAN_STATS},
        }
        execute = {
            "schema": "tensorkiln.cli.execute.v1",
            "workload": {"id": "reglu_mlp_v1"},
            "plan": {"stats": archive_gate.REGLU_PLAN_STATS},
            "execution": {
                "outputs": [
                    {
                        "name": "result",
                        "dtype": "f32",
                        "shape": [2, 4],
                        "bits": list(archive_gate.REGLU_OUTPUT_BITS),
                    }
                ],
                "reference_check": {
                    "comparison": "raw_f32_bits",
                    "matched": 8,
                    "total": 8,
                    "status": "match",
                },
            },
        }

        archive_gate._verify_reglu_semantics(inspect, execute)

        wrong = {
            **execute,
            "execution": {
                **execute["execution"],
                "reference_check": {
                    "comparison": "raw_f32_bits",
                    "matched": 7,
                    "total": 8,
                    "status": "match",
                },
            },
        }
        with self.assertRaisesRegex(archive_gate.SourceArchiveError, "8/8"):
            archive_gate._verify_reglu_semantics(inspect, wrong)

    def test_json_parser_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(
            archive_gate.SourceArchiveError, "duplicate JSON key"
        ):
            archive_gate._parse_cli_json(b'{"schema":1,"schema":2}\n', "fixture")


if __name__ == "__main__":
    unittest.main()
