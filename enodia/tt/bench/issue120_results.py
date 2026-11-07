"""Attempt-bound native results and byte-preserving process diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("/out")
RESULT_SCHEMA = "issue120-native-result-v1"
EXECUTION_SCHEMA = "issue120-native-execution-v1"
MAX_RESULT_BYTES = 64 * 1024 * 1024


class NativeResultError(ValueError):
    """A native attempt cannot be accepted as a result."""


@dataclass(frozen=True)
class NativeAttempt:
    identity: str
    directory: Path

    @property
    def result_directory(self) -> Path:
        return self.directory / "result"

    @property
    def result_file(self) -> Path:
        return self.result_directory / "result.json"

    @property
    def stdout_path(self) -> Path:
        return self.directory / "stdout.bin"

    @property
    def stderr_path(self) -> Path:
        return self.directory / "stderr.bin"

    @property
    def execution_path(self) -> Path:
        return self.directory / "execution.json"


def create_attempt(identity: str | None) -> NativeAttempt:
    if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity):
        raise NativeResultError("a valid wrapper attempt identity is required")
    try:
        if not OUTPUT_ROOT.is_dir() or OUTPUT_ROOT.resolve() != OUTPUT_ROOT.absolute():
            raise NativeResultError("native output root is missing or redirected")
        attempt = NativeAttempt(identity, OUTPUT_ROOT / f"native-{identity}")
        attempt.directory.mkdir()  # Exclusive claim; never resume or erase old data.
        attempt.result_directory.mkdir()
    except OSError as exc:
        raise NativeResultError("native attempt directory must be new and writable") from exc
    return attempt


def _artifact(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"file": path.name, "byte_count": path.stat().st_size, "sha256": digest.hexdigest()}


def _atomic_metadata(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    if path.exists() or path.is_symlink():
        raise NativeResultError("native execution metadata already exists")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.rename(path)


def _wait_native(command: list[str], stdout, stderr, timeout: int, environment: Callable[[], dict[str, str]], cwd: Path, state: dict[str, Any]) -> None:
    process = None
    try:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment(), cwd=cwd)
        state["native_started"] = True
        try:
            state["native_returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            state.update(native_returncode=process.wait(), timed_out=True)
    finally:
        failure = sys.exc_info()[1]
        if failure is not None:
            state.update(failure_type=type(failure).__name__, failure_message=str(failure))
        if process is not None and state["native_returncode"] is None:
            process.kill()
            state["native_returncode"] = process.wait()


def execute(command: list[str], attempt: NativeAttempt, timeout: int, environment: Callable[[], dict[str, str]], cwd: Path) -> subprocess.CompletedProcess:
    """Redirect at spawn: output survives exceptions and never needs decoding."""
    state = {"native_started": False, "native_returncode": None, "timed_out": False,
             "failure_type": None, "failure_message": None}
    error = None
    try:
        with attempt.stdout_path.open("xb") as stdout, attempt.stderr_path.open("xb") as stderr:
            try:
                _wait_native(command, stdout, stderr, timeout, environment, cwd, state)
            finally:
                stdout.flush()
                stderr.flush()
                os.fsync(stdout.fileno())
                os.fsync(stderr.fileno())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = exc
    finally:
        if attempt.stdout_path.is_file() and attempt.stderr_path.is_file():
            _atomic_metadata(attempt.execution_path, {
                "schema": EXECUTION_SCHEMA, "attempt_identity": attempt.identity, **state,
                "stdout": _artifact(attempt.stdout_path), "stderr": _artifact(attempt.stderr_path),
            })
    if error is not None:
        raise NativeResultError(f"native launch/preflight failed: {error}; diagnostics retained") from error
    if state["failure_type"]:
        raise NativeResultError(f"native launch/preflight failed: {state['failure_message']}; diagnostics retained")
    if state["timed_out"]:
        raise NativeResultError("native exceeded its bounded cap; timeout diagnostics retained")
    return subprocess.CompletedProcess(command, state["native_returncode"])


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise NativeResultError("duplicate native JSON key")
        value[key] = item
    return value


def _read_json(path: Path) -> Any:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
            raise NativeResultError("native JSON must be a bounded regular file")
        return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeResultError("native JSON file is missing or malformed") from exc


def read_result(attempt: NativeAttempt) -> dict[str, Any]:
    """Accept exactly one fresh file, never stdout or a filename-selected fallback."""
    execution = _read_json(attempt.execution_path)
    if not isinstance(execution, dict):
        raise NativeResultError("native execution metadata must be an object")
    expected = {"schema": EXECUTION_SCHEMA, "attempt_identity": attempt.identity, "native_started": True,
                "native_returncode": 0, "timed_out": False, "failure_type": None}
    if any(execution.get(key) != value for key, value in expected.items()):
        raise NativeResultError("native exit/identity does not permit success")
    if type(execution.get("native_returncode")) is not int or any(
        type(execution.get(key)) is not bool for key in ("native_started", "timed_out")
    ):
        raise NativeResultError("native exit metadata types are invalid")
    try:
        directory = attempt.result_directory
        if directory.resolve() != directory.absolute():
            raise NativeResultError("native result directory is redirected")
        candidates = list(directory.iterdir())
    except OSError as exc:
        raise NativeResultError("native result directory is unavailable") from exc
    if len(candidates) != 1 or candidates[0] != attempt.result_file:
        raise NativeResultError("exactly one native result file is required; missing or multiple candidates")
    envelope = _read_json(attempt.result_file)
    if not isinstance(envelope, dict) or set(envelope) != {"schema", "attempt_identity", "result"}:
        raise NativeResultError("native result envelope is invalid")
    if envelope["schema"] != RESULT_SCHEMA or envelope["attempt_identity"] != attempt.identity:
        raise NativeResultError("native result schema/attempt identity mismatch")
    if not isinstance(envelope["result"], dict):
        raise NativeResultError("native result payload must be an object")
    return envelope["result"]
