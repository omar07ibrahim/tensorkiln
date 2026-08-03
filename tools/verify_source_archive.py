#!/usr/bin/env python3
"""Build and verify TensorKiln from an exact, bounded ``git archive``.

The gate deliberately ignores working-tree file bytes.  It resolves one
committed HEAD, validates every archived regular file against that commit's
tree and blob IDs, extracts without following links, and builds the release
CLI in a private directory below ``build/``.  Its stdout is a path-free,
deterministic receipt; compiler output is captured only for pass/fail.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Iterable, Mapping, Sequence


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
BUILD_ROOT: Final = REPOSITORY_ROOT / "build"
ARCHIVE_PREFIX: Final = "tensorkiln-source/"

GIT_BINARY: Final = Path("/usr/bin/git")
MAKE_BINARY: Final = Path("/usr/bin/make")
CXX_BINARY: Final = Path("/usr/bin/g++")
AR_BINARY: Final = Path("/usr/bin/ar")

MAX_ARCHIVE_BYTES: Final = 16 * 1024 * 1024
MAX_ARCHIVE_MEMBERS: Final = 4096
MAX_TRACKED_FILES: Final = 4096
MAX_MEMBER_BYTES: Final = 4 * 1024 * 1024
MAX_TRACKED_BYTES: Final = 16 * 1024 * 1024
MAX_GIT_OUTPUT_BYTES: Final = 8 * 1024 * 1024
MAX_PROCESS_STDERR_BYTES: Final = 1024 * 1024
MAX_BUILD_OUTPUT_BYTES: Final = 4 * 1024 * 1024
MAX_CLI_OUTPUT_BYTES: Final = 1024 * 1024

GIT_TIMEOUT_SECONDS: Final = 30
BUILD_TIMEOUT_SECONDS: Final = 300
CLI_TIMEOUT_SECONDS: Final = 15

ARCHIVE_FILE_MODE: Final = 0o664
ARCHIVE_EXECUTABLE_MODE: Final = 0o775
ARCHIVE_DIRECTORY_MODE: Final = 0o775

DENSE_INPUT_BITS: Final = (
    "x=0x3f800000,0x40000000,0x40400000,"
    "0xbf800000,0x3f000000,0x40800000"
)
DENSE_INSPECT_SHA256: Final = (
    "bafba37b0c4ece4545ee011cdb904c070bcb456a5180ef1512b6b579340a8690"
)
DENSE_EXECUTE_SHA256: Final = (
    "2d74429690aa514770d5a137ebf958f8b475a5b1cb4425a36ae7a4e42566b86b"
)
REGULAR_INPUT_PALETTE: Final = (
    "0x3f800000",
    "0x40000000",
    "0x40400000",
    "0xbf800000",
    "0x3f000000",
    "0x40800000",
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

GATE_INPUT_PATHS: Final = (
    "Makefile",
    "cli",
    "include",
    "src",
    "tools/verify_source_archive.py",
    "tests/test_source_archive.py",
)
FORBIDDEN_COMPONENTS: Final = frozenset(
    {
        ".git",
        "build",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
FORBIDDEN_SUFFIXES: Final = (
    ".a",
    ".d",
    ".gcda",
    ".gcno",
    ".o",
    ".pyc",
    ".so",
)
SAFE_OBJECT_IDS: Final = {
    "sha1": re.compile(r"^[0-9a-f]{40}$"),
    "sha256": re.compile(r"^[0-9a-f]{64}$"),
}


class SourceArchiveError(RuntimeError):
    """Raised when committed-source archive evidence cannot be trusted."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class TrackedFile:
    path: str
    mode: str
    object_id: str


@dataclass(frozen=True)
class ArchivedFile:
    path: str
    mode: str
    payload: bytes


@dataclass(frozen=True)
class ValidatedArchive:
    files: tuple[ArchivedFile, ...]
    archive_bytes: int
    member_count: int
    regular_bytes: int
    sha256: str


@dataclass(frozen=True)
class CliReceipt:
    help_sha256: str
    list_sha256: str
    workloads: tuple[str, ...]
    dense_inspect_sha256: str
    dense_execute_sha256: str
    reglu_status: str
    reglu_inspect_sha256: str
    reglu_execute_sha256: str


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_bounded(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    label: str,
) -> ProcessResult:
    """Run without a shell while bounding time and both captured streams."""

    if not command or any("\0" in argument for argument in command):
        raise SourceArchiveError(f"{label} command is malformed")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise SourceArchiveError(f"could not start {label}") from error

    selector: selectors.BaseSelector | None = None
    try:
        assert process.stdout is not None
        assert process.stderr is not None
        streams = {
            process.stdout.fileno(): ("stdout", max_stdout_bytes),
            process.stderr.fileno(): ("stderr", max_stderr_bytes),
        }
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            selector = selectors.DefaultSelector()
            for descriptor in streams:
                selector.register(descriptor, selectors.EVENT_READ)
        except OSError as error:
            raise SourceArchiveError(
                f"could not monitor {label} output"
            ) from error
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SourceArchiveError(
                    f"{label} exceeded its {timeout_seconds}-second limit"
                )
            events = selector.select(min(remaining, 0.25))
            for key, _mask in events:
                descriptor = int(key.fd)
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except OSError as error:
                    raise SourceArchiveError(
                        f"could not read {label} output"
                    ) from error
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                stream_name, limit = streams[descriptor]
                target = buffers[stream_name]
                target.extend(chunk)
                if len(target) > limit:
                    raise SourceArchiveError(
                        f"{label} {stream_name} exceeded its byte limit"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SourceArchiveError(
                f"{label} exceeded its {timeout_seconds}-second limit"
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise SourceArchiveError(
                f"{label} exceeded its {timeout_seconds}-second limit"
            ) from error
    finally:
        try:
            if selector is not None:
                selector.close()
        finally:
            try:
                _kill_process_group(process)
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
    return ProcessResult(
        returncode=returncode,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
    )


def _minimal_environment(**extra: str) -> dict[str, str]:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "TZ": "UTC",
    }
    environment.update(extra)
    return environment


def _git(
    arguments: Sequence[str],
    *,
    max_stdout_bytes: int = MAX_GIT_OUTPUT_BYTES,
    label: str,
) -> bytes:
    result = run_bounded(
        (
            str(GIT_BINARY),
            "--no-optional-locks",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "tar.umask=0002",
            *arguments,
        ),
        cwd=REPOSITORY_ROOT,
        environment=_minimal_environment(),
        timeout_seconds=GIT_TIMEOUT_SECONDS,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=MAX_PROCESS_STDERR_BYTES,
        label=label,
    )
    if result.returncode != 0:
        raise SourceArchiveError(
            f"{label} failed with exit status {result.returncode}"
        )
    if result.stderr:
        raise SourceArchiveError(f"{label} wrote unexpected stderr")
    return result.stdout


def _ascii_line(payload: bytes, label: str) -> str:
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise SourceArchiveError(f"{label} is not ASCII") from error
    if not text.endswith("\n") or "\n" in text[:-1] or "\r" in text:
        raise SourceArchiveError(f"{label} is not one canonical line")
    return text[:-1]


def _safe_relative_path(path: str, label: str) -> PurePosixPath:
    if (
        not path
        or "\0" in path
        or "\\" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        raise SourceArchiveError(f"{label} contains an unsafe path")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise SourceArchiveError(f"{label} contains an unsafe path")
    if FORBIDDEN_COMPONENTS.intersection(pure.parts):
        raise SourceArchiveError(f"{label} contains Git metadata or build output")
    if path.endswith(FORBIDDEN_SUFFIXES):
        raise SourceArchiveError(f"{label} contains a compiled build artifact")
    return pure


def git_blob_object_id(payload: bytes, object_format: str) -> str:
    """Return the Git object ID for one exact regular-file payload."""

    if object_format not in SAFE_OBJECT_IDS:
        raise SourceArchiveError(f"unsupported Git object format: {object_format}")
    digest = hashlib.new(object_format)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def parse_tracked_inventory(
    payload: bytes, object_format: str
) -> tuple[TrackedFile, ...]:
    """Parse and validate one NUL-delimited ``git ls-tree -r`` result."""

    object_pattern = SAFE_OBJECT_IDS.get(object_format)
    if object_pattern is None:
        raise SourceArchiveError(f"unsupported Git object format: {object_format}")
    if not payload or not payload.endswith(b"\0"):
        raise SourceArchiveError("tracked inventory is empty or not NUL-terminated")
    records = payload[:-1].split(b"\0")
    if len(records) > MAX_TRACKED_FILES:
        raise SourceArchiveError("tracked inventory exceeds its file limit")
    entries: list[TrackedFile] = []
    seen: set[str] = set()
    for raw_record in records:
        try:
            metadata, raw_path = raw_record.split(b"\t", maxsplit=1)
            mode, object_type, raw_object_id = metadata.split(b" ")
            path = raw_path.decode("utf-8", errors="strict")
            decoded_mode = mode.decode("ascii", errors="strict")
            decoded_type = object_type.decode("ascii", errors="strict")
            object_id = raw_object_id.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise SourceArchiveError("tracked inventory record is malformed") from error
        _safe_relative_path(path, "tracked inventory")
        if path in seen:
            raise SourceArchiveError("tracked inventory contains a duplicate path")
        if decoded_type != "blob" or decoded_mode not in {"100644", "100755"}:
            raise SourceArchiveError(
                "tracked inventory contains a link, submodule, or special entry"
            )
        if object_pattern.fullmatch(object_id) is None:
            raise SourceArchiveError("tracked inventory has a malformed object ID")
        seen.add(path)
        entries.append(TrackedFile(path, decoded_mode, object_id))
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _expected_archive_directories(paths: Iterable[str]) -> set[str]:
    root = ARCHIVE_PREFIX.rstrip("/")
    directories = {root}
    for path in paths:
        parts = PurePosixPath(path).parts[:-1]
        current = root
        for part in parts:
            current = f"{current}/{part}"
            directories.add(current)
    return directories


def validate_archive(
    archive: bytes,
    tracked: Sequence[TrackedFile],
    object_format: str,
) -> ValidatedArchive:
    """Validate the tar envelope, exact inventory, modes, and Git blobs."""

    if not archive or len(archive) > MAX_ARCHIVE_BYTES:
        raise SourceArchiveError("source archive is empty or exceeds its byte limit")
    expected = {entry.path: entry for entry in tracked}
    if len(expected) != len(tracked) or not expected:
        raise SourceArchiveError("tracked inventory is empty or contains duplicates")
    expected_directories = _expected_archive_directories(expected)
    directories: set[str] = set()
    files: dict[str, ArchivedFile] = {}
    seen_members: set[str] = set()
    regular_bytes = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            members = source.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise SourceArchiveError("source archive exceeds its member limit")
            for member in members:
                normalized_name = (
                    member.name.rstrip("/") if member.isdir() else member.name
                )
                if normalized_name in seen_members:
                    raise SourceArchiveError(
                        "source archive contains duplicate members"
                    )
                seen_members.add(normalized_name)
                if (
                    normalized_name == ARCHIVE_PREFIX.rstrip("/")
                    and member.isdir()
                ):
                    relative = ""
                elif normalized_name.startswith(ARCHIVE_PREFIX):
                    relative = normalized_name[len(ARCHIVE_PREFIX) :]
                else:
                    raise SourceArchiveError(
                        "source archive escaped its one required prefix"
                    )
                if member.isdir():
                    if member.mode != ARCHIVE_DIRECTORY_MODE or member.size != 0:
                        raise SourceArchiveError(
                            "source archive directory mode or size differs"
                        )
                    if relative:
                        _safe_relative_path(relative, "source archive")
                    directories.add(normalized_name)
                    continue
                if not member.isreg():
                    raise SourceArchiveError(
                        "source archive contains a link, device, or special member"
                    )
                _safe_relative_path(relative, "source archive")
                if relative in files:
                    raise SourceArchiveError("source archive contains duplicate files")
                tracked_entry = expected.get(relative)
                if tracked_entry is None:
                    raise SourceArchiveError(
                        "source archive has a file outside the tracked inventory"
                    )
                expected_mode = (
                    ARCHIVE_EXECUTABLE_MODE
                    if tracked_entry.mode == "100755"
                    else ARCHIVE_FILE_MODE
                )
                if member.mode != expected_mode:
                    raise SourceArchiveError(
                        "source archive file mode differs from Git"
                    )
                if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                    raise SourceArchiveError(
                        "source archive member exceeds its byte limit"
                    )
                regular_bytes += member.size
                if regular_bytes > MAX_TRACKED_BYTES:
                    raise SourceArchiveError(
                        "source archive regular files exceed their byte limit"
                    )
                stream = source.extractfile(member)
                if stream is None:
                    raise SourceArchiveError(
                        "source archive regular file is unreadable"
                    )
                payload = stream.read(MAX_MEMBER_BYTES + 1)
                if len(payload) != member.size:
                    raise SourceArchiveError(
                        "source archive member size changed while read"
                    )
                if (
                    git_blob_object_id(payload, object_format)
                    != tracked_entry.object_id
                ):
                    raise SourceArchiveError(
                        "source archive payload differs from its committed Git blob"
                    )
                files[relative] = ArchivedFile(
                    relative, tracked_entry.mode, payload
                )
    except (tarfile.TarError, EOFError) as error:
        raise SourceArchiveError(
            "source archive is not a valid plain tar file"
        ) from error
    if set(files) != set(expected):
        raise SourceArchiveError("source archive omits a tracked regular file")
    if directories != expected_directories:
        raise SourceArchiveError("source archive directory inventory differs")
    return ValidatedArchive(
        files=tuple(files[path] for path in sorted(files)),
        archive_bytes=len(archive),
        member_count=len(seen_members),
        regular_bytes=regular_bytes,
        sha256=hashlib.sha256(archive).hexdigest(),
    )


def _open_directory(parent_fd: int, component: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(component, flags, dir_fd=parent_fd)
    except OSError as error:
        raise SourceArchiveError(
            "could not open a private extraction directory"
        ) from error


def _ensure_directory_path(root_fd: int, parts: Sequence[str]) -> int:
    current_fd = os.dup(root_fd)
    try:
        for component in parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = _open_directory(current_fd, component)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise SourceArchiveError("could not write an extracted regular file")
        written += count


def extract_validated_archive(
    validated: ValidatedArchive, private_root: Path
) -> Path:
    """Extract regular files with dirfd traversal and exclusive no-follow opens."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise SourceArchiveError("platform lacks no-follow directory extraction")
    metadata = private_root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or private_root.is_symlink():
        raise SourceArchiveError("private extraction root is not a real directory")
    root_fd = os.open(private_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        prefix_parts = PurePosixPath(ARCHIVE_PREFIX.rstrip("/")).parts
        for archived in validated.files:
            relative_parts = PurePosixPath(archived.path).parts
            parent_fd = _ensure_directory_path(
                root_fd, (*prefix_parts, *relative_parts[:-1])
            )
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                try:
                    descriptor = os.open(
                        relative_parts[-1], flags, 0o600, dir_fd=parent_fd
                    )
                except OSError as error:
                    raise SourceArchiveError(
                        "refused to clobber or follow an extraction target"
                    ) from error
                try:
                    _write_all(descriptor, archived.payload)
                    os.fchmod(
                        descriptor,
                        0o755 if archived.mode == "100755" else 0o644,
                    )
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent_fd)
    finally:
        os.close(root_fd)
    return private_root / ARCHIVE_PREFIX.rstrip("/")


def _validate_cli_text(payload: bytes, label: str) -> str:
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise SourceArchiveError(f"{label} output is not ASCII") from error
    if not text or not text.endswith("\n") or "\r" in text or "\0" in text:
        raise SourceArchiveError(f"{label} output is not canonical text")
    return text


def _run_cli(binary: Path, arguments: Sequence[str], label: str) -> bytes:
    result = run_bounded(
        (str(binary), *arguments),
        cwd=binary.parent,
        environment=_minimal_environment(),
        timeout_seconds=CLI_TIMEOUT_SECONDS,
        max_stdout_bytes=MAX_CLI_OUTPUT_BYTES,
        max_stderr_bytes=MAX_CLI_OUTPUT_BYTES,
        label=label,
    )
    if result.returncode != 0:
        raise SourceArchiveError(
            f"{label} failed with exit status {result.returncode}"
        )
    if result.stderr:
        raise SourceArchiveError(f"{label} wrote unexpected stderr")
    _validate_cli_text(result.stdout, label)
    return result.stdout


def _replay_cli(binary: Path, arguments: Sequence[str], label: str) -> bytes:
    first = _run_cli(binary, arguments, label)
    second = _run_cli(binary, arguments, f"{label} replay")
    if first != second:
        raise SourceArchiveError(f"{label} is not byte-replayable")
    return first


def _parse_cli_json(payload: bytes, label: str) -> dict[str, object]:
    text = _validate_cli_text(payload, label)

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SourceArchiveError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as error:
        raise SourceArchiveError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SourceArchiveError(f"{label} is not a JSON object")
    return value


def _verify_execution_report(
    report: dict[str, object], workload_id: str, label: str
) -> None:
    workload = report.get("workload")
    execution = report.get("execution")
    if (
        report.get("schema") != "tensorkiln.cli.execute.v1"
        or not isinstance(workload, dict)
        or workload.get("id") != workload_id
        or not isinstance(execution, dict)
        or execution.get("run_status") != "success"
        or execution.get("kernel_write_audit") is not True
        or execution.get("benchmark") is not False
    ):
        raise SourceArchiveError(f"{label} omitted audited execution guarantees")
    reference = execution.get("reference_check")
    if not isinstance(reference, dict) or reference.get("status") != "match":
        raise SourceArchiveError(f"{label} did not match the independent reference")


def _verify_reglu_semantics(
    inspect_report: dict[str, object], execute_report: dict[str, object]
) -> None:
    """Check ReGLU fields semantically without pinning its whole JSON bytes."""

    inspect_plan = inspect_report.get("plan")
    execute_plan = execute_report.get("plan")
    execution = execute_report.get("execution")
    if (
        not isinstance(inspect_plan, dict)
        or inspect_plan.get("stats") != REGLU_PLAN_STATS
        or not isinstance(execute_plan, dict)
        or execute_plan.get("stats") != REGLU_PLAN_STATS
        or not isinstance(execution, dict)
    ):
        raise SourceArchiveError("ReGLU plan statistics differ from the contract")
    outputs = execution.get("outputs")
    expected_outputs = [
        {
            "name": "result",
            "dtype": "f32",
            "shape": [2, 4],
            "bits": list(REGLU_OUTPUT_BITS),
        }
    ]
    reference = execution.get("reference_check")
    if outputs != expected_outputs or reference != {
        "comparison": "raw_f32_bits",
        "matched": 8,
        "total": 8,
        "status": "match",
    }:
        raise SourceArchiveError("ReGLU fixture output or 8/8 receipt differs")


def verify_cli(binary: Path) -> CliReceipt:
    """Run the stable dense contract and any archived ReGLU workload."""

    help_payload = _replay_cli(binary, ("--help",), "CLI help")
    help_text = _validate_cli_text(help_payload, "CLI help")
    if (
        "TensorKiln bounded workload CLI" not in help_text
        or "not a graph dump parser" not in help_text
    ):
        raise SourceArchiveError("CLI help omitted its documented boundary")

    list_payload = _replay_cli(
        binary, ("list", "--format=json"), "CLI workload list"
    )
    listing = _parse_cli_json(list_payload, "CLI workload list")
    workloads_value = listing.get("workloads")
    if (
        listing.get("schema") != "tensorkiln.cli.workloads.v1"
        or not isinstance(workloads_value, list)
        or not workloads_value
    ):
        raise SourceArchiveError("CLI workload registry has the wrong schema")
    workload_records: dict[str, dict[str, object]] = {}
    workload_ids: list[str] = []
    for value in workloads_value:
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise SourceArchiveError("CLI workload registry contains a malformed entry")
        workload_id = str(value["id"])
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", workload_id) is None:
            raise SourceArchiveError("CLI workload registry contains an unsafe ID")
        if workload_id in workload_records:
            raise SourceArchiveError("CLI workload registry contains duplicate IDs")
        workload_records[workload_id] = value
        workload_ids.append(workload_id)
    if tuple(workload_ids) not in {
        ("dense_relu_v1",),
        ("dense_relu_v1", "reglu_mlp_v1"),
    }:
        raise SourceArchiveError("CLI workload registry order or IDs differ")

    dense_inspect = _replay_cli(
        binary,
        ("inspect", "--workload", "dense_relu_v1", "--format=json"),
        "dense inspect",
    )
    dense_inspect_sha = hashlib.sha256(dense_inspect).hexdigest()
    if dense_inspect_sha != DENSE_INSPECT_SHA256:
        raise SourceArchiveError(
            "dense inspect receipt differs from the stable contract"
        )
    dense_execute = _replay_cli(
        binary,
        (
            "execute",
            "--workload",
            "dense_relu_v1",
            "--input-bits",
            DENSE_INPUT_BITS,
            "--format=json",
        ),
        "dense execute",
    )
    dense_execute_sha = hashlib.sha256(dense_execute).hexdigest()
    if dense_execute_sha != DENSE_EXECUTE_SHA256:
        raise SourceArchiveError(
            "dense execute receipt differs from the stable contract"
        )
    _verify_execution_report(
        _parse_cli_json(dense_execute, "dense execute"),
        "dense_relu_v1",
        "dense execute",
    )

    reglu_status = "absent_in_archived_commit"
    reglu_inspect_sha = "not_applicable"
    reglu_execute_sha = "not_applicable"
    reglu = workload_records.get("reglu_mlp_v1")
    if reglu is not None:
        inputs = reglu.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise SourceArchiveError("ReGLU workload must expose one bounded input")
        input_record = inputs[0]
        if not isinstance(input_record, dict):
            raise SourceArchiveError("ReGLU input descriptor is malformed")
        input_name = input_record.get("name")
        input_elements = input_record.get("elements")
        if (
            not isinstance(input_name, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", input_name) is None
            or type(input_elements) is not int
            or not 1 <= input_elements <= 4096
        ):
            raise SourceArchiveError("ReGLU input descriptor exceeds its bounds")
        bits = ",".join(
            REGULAR_INPUT_PALETTE[index % len(REGULAR_INPUT_PALETTE)]
            for index in range(input_elements)
        )
        reglu_inspect = _replay_cli(
            binary,
            ("inspect", "--workload", "reglu_mlp_v1", "--format=json"),
            "ReGLU inspect",
        )
        inspect_report = _parse_cli_json(reglu_inspect, "ReGLU inspect")
        inspect_workload = inspect_report.get("workload")
        if (
            inspect_report.get("schema") != "tensorkiln.cli.inspect.v1"
            or not isinstance(inspect_workload, dict)
            or inspect_workload.get("id") != "reglu_mlp_v1"
        ):
            raise SourceArchiveError("ReGLU inspect report has the wrong workload")
        reglu_execute = _replay_cli(
            binary,
            (
                "execute",
                "--workload",
                "reglu_mlp_v1",
                "--input-bits",
                f"{input_name}={bits}",
                "--format=json",
            ),
            "ReGLU execute",
        )
        execute_report = _parse_cli_json(reglu_execute, "ReGLU execute")
        _verify_execution_report(
            execute_report,
            "reglu_mlp_v1",
            "ReGLU execute",
        )
        _verify_reglu_semantics(inspect_report, execute_report)
        reglu_status = "verified_from_archived_registry"
        reglu_inspect_sha = hashlib.sha256(reglu_inspect).hexdigest()
        reglu_execute_sha = hashlib.sha256(reglu_execute).hexdigest()

    return CliReceipt(
        help_sha256=hashlib.sha256(help_payload).hexdigest(),
        list_sha256=hashlib.sha256(list_payload).hexdigest(),
        workloads=tuple(sorted(workload_records)),
        dense_inspect_sha256=dense_inspect_sha,
        dense_execute_sha256=dense_execute_sha,
        reglu_status=reglu_status,
        reglu_inspect_sha256=reglu_inspect_sha,
        reglu_execute_sha256=reglu_execute_sha,
    )


def _require_regular_executable(path: Path, label: str) -> None:
    try:
        metadata = path.stat()
    except OSError as error:
        raise SourceArchiveError(f"required {label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
        raise SourceArchiveError(f"required {label} is not executable")


def _ensure_build_root() -> None:
    try:
        metadata = BUILD_ROOT.lstat()
    except FileNotFoundError:
        BUILD_ROOT.mkdir(mode=0o755)
        metadata = BUILD_ROOT.lstat()
    except OSError as error:
        raise SourceArchiveError(
            "cannot inspect the repository build directory"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or BUILD_ROOT.is_symlink():
        raise SourceArchiveError("repository build path is not a real directory")


def _resolve_committed_head() -> tuple[str, str, str, int]:
    repository = _ascii_line(
        _git(("rev-parse", "--show-toplevel"), label="Git root query"),
        "Git root",
    )
    if Path(repository).resolve() != REPOSITORY_ROOT:
        raise SourceArchiveError("Git root differs from the TensorKiln repository")
    object_format = _ascii_line(
        _git(("rev-parse", "--show-object-format"), label="object-format query"),
        "Git object format",
    )
    pattern = SAFE_OBJECT_IDS.get(object_format)
    if pattern is None:
        raise SourceArchiveError(f"unsupported Git object format: {object_format}")
    commit = _ascii_line(
        _git(
            ("rev-parse", "--verify", "HEAD^{commit}"),
            label="HEAD commit query",
        ),
        "HEAD commit",
    )
    tree = _ascii_line(
        _git(("show", "-s", "--format=%T", commit), label="HEAD tree query"),
        "HEAD tree",
    )
    timestamp_text = _ascii_line(
        _git(
            ("show", "-s", "--format=%ct", commit),
            label="HEAD timestamp query",
        ),
        "HEAD timestamp",
    )
    if pattern.fullmatch(commit) is None or pattern.fullmatch(tree) is None:
        raise SourceArchiveError("Git returned a malformed commit or tree ID")
    if re.fullmatch(r"[1-9][0-9]{0,11}", timestamp_text) is None:
        raise SourceArchiveError("Git returned a malformed commit timestamp")
    return commit, tree, object_format, int(timestamp_text)


def _require_gate_inputs_clean() -> None:
    status = _git(
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *GATE_INPUT_PATHS,
        ),
        max_stdout_bytes=1024 * 1024,
        label="gate-input status query",
    )
    if status:
        raise SourceArchiveError(
            "source-archive build or verifier inputs differ from committed HEAD"
        )


def _build_archived_cli(
    extracted_root: Path, private_root: Path, commit_timestamp: int
) -> Path:
    temporary_dir = private_root / "tmp"
    temporary_dir.mkdir(mode=0o700)
    environment = _minimal_environment(
        SOURCE_DATE_EPOCH=str(commit_timestamp),
        TMPDIR=str(temporary_dir),
    )
    result = run_bounded(
        (
            str(MAKE_BINARY),
            "-j2",
            f"CXX={CXX_BINARY}",
            f"AR={AR_BINARY}",
            "PROFILE=release",
            "cli",
        ),
        cwd=extracted_root,
        environment=environment,
        timeout_seconds=BUILD_TIMEOUT_SECONDS,
        max_stdout_bytes=MAX_BUILD_OUTPUT_BYTES,
        max_stderr_bytes=MAX_BUILD_OUTPUT_BYTES,
        label="archived release CLI build",
    )
    if result.returncode != 0:
        raise SourceArchiveError(
            "archived release CLI build failed with exit status "
            f"{result.returncode}"
        )
    binary = extracted_root / "build/g++/release/tensorkiln"
    try:
        metadata = binary.lstat()
    except OSError as error:
        raise SourceArchiveError("archived release build omitted the CLI") from error
    if (
        binary.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(binary, os.X_OK)
        or metadata.st_size <= 0
        or metadata.st_size > 64 * 1024 * 1024
    ):
        raise SourceArchiveError("archived release CLI is not a bounded executable")
    return binary


def render_receipt(
    *,
    commit: str,
    tree: str,
    tracked_count: int,
    validated: ValidatedArchive,
    cli: CliReceipt,
) -> str:
    """Render the only successful stdout: stable IDs, counts, and receipts."""

    workloads = ",".join(cli.workloads)
    receipt = "\n".join(
        (
            "tensorkiln.source_archive_receipt v1",
            f"commit={commit}",
            f"tree={tree}",
            f"archive_sha256={validated.sha256}",
            f"archive_prefix={ARCHIVE_PREFIX}",
            f"archive_bytes={validated.archive_bytes}",
            f"archive_members={validated.member_count}",
            f"tracked_files={tracked_count}",
            f"tracked_regular_bytes={validated.regular_bytes}",
            "build_profile=release",
            f"help_sha256={cli.help_sha256}",
            f"list_sha256={cli.list_sha256}",
            f"workloads={workloads}",
            f"dense_inspect_sha256={cli.dense_inspect_sha256}",
            f"dense_execute_sha256={cli.dense_execute_sha256}",
            f"reglu_mlp_v1={cli.reglu_status}",
            f"reglu_inspect_sha256={cli.reglu_inspect_sha256}",
            f"reglu_execute_sha256={cli.reglu_execute_sha256}",
            "kernel_write_audit=required",
            "independent_reference_match=required",
            "claim=committed-source correctness gate; not a benchmark",
        )
    ) + "\n"
    if re.search(r"(?:/home/|/root/|/Users/|/tmp/|file://)", receipt):
        raise SourceArchiveError("source-archive receipt contains a host path")
    return receipt


def verify_source_archive() -> str:
    for path, label in (
        (GIT_BINARY, "Git"),
        (MAKE_BINARY, "make"),
        (CXX_BINARY, "GNU C++ compiler"),
        (AR_BINARY, "archive tool"),
    ):
        _require_regular_executable(path, label)
    _require_gate_inputs_clean()
    commit, tree, object_format, timestamp = _resolve_committed_head()
    inventory = parse_tracked_inventory(
        _git(
            ("ls-tree", "-r", "-z", "--full-tree", commit),
            label="committed tree inventory",
        ),
        object_format,
    )
    archive = _git(
        (
            "archive",
            "--format=tar",
            f"--prefix={ARCHIVE_PREFIX}",
            commit,
        ),
        max_stdout_bytes=MAX_ARCHIVE_BYTES,
        label="committed source archive",
    )
    validated = validate_archive(archive, inventory, object_format)
    _ensure_build_root()
    with tempfile.TemporaryDirectory(
        prefix="source-archive-", dir=BUILD_ROOT
    ) as temporary:
        private_root = Path(temporary)
        private_root.chmod(0o700)
        extracted_root = extract_validated_archive(validated, private_root)
        binary = _build_archived_cli(extracted_root, private_root, timestamp)
        cli_receipt = verify_cli(binary)
    current_head, current_tree, _format, _timestamp = _resolve_committed_head()
    if current_head != commit or current_tree != tree:
        raise SourceArchiveError("HEAD changed during source-archive verification")
    _require_gate_inputs_clean()
    return render_receipt(
        commit=commit,
        tree=tree,
        tracked_count=len(inventory),
        validated=validated,
        cli=cli_receipt,
    )


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "build the release CLI from a bounded committed-HEAD git archive "
            "and emit a canonical receipt"
        )
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_arguments(argv)
    try:
        receipt = verify_source_archive()
    except (
        SourceArchiveError,
        OSError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"source archive verification error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
