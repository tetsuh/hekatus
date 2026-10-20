"""Issue #146 grouped Newton-Schulz runner and canonical record adapter.

The board-free planning and validation seams live here; device access is
performed only by :mod:`tools.issue146_device` after its complete preflight.
The adapter deliberately reuses the Issue #137 core-plan, identity, cooldown,
and wrapper lifecycle helpers without changing their defaults.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench import power_trace_issue146, telemetry
from enodia.tt.bench.duration_load import DEFAULT_OVERRUN_LIMIT_SECONDS
from enodia.tt.bench.load_harness import (
    NewtonSchulzCorePlan,
    build_newton_schulz_core_plan,
)
from enodia.tt.bench.resident_harness import validate_harness_preflight
from enodia.tt.bench.run_binding import validate_run_id
from tools import issue137, issue146_contract

DEFAULT_PLAN_PATH = issue146_contract.DEFAULT_PLAN_PATH
ISSUE146_RECORD_SCHEMA = "issue146-measurement-v1"
ISSUE146_DEVICE_RESULT_SCHEMA = "issue146-device-result-v1"
ISSUE146_SNAPSHOT_SCHEMA = "issue146-tt-smi-snapshot-v1"
ISSUE146_GROUPS_SCHEMA = "issue146-group-timings-v1"
ISSUE146_GROUP_TIMINGS_SCHEMA = ISSUE146_GROUPS_SCHEMA
ISSUE146_LOAD_MODE = "host-duration-bounded"
CURRENT_PLAN_REVISION = issue146_contract.CURRENT_PLAN_REVISION
ISSUE146_DEVICE_RUNNER = "tools/issue146_device.py"
WRAPPER_SCRIPT = "enodia/tt/bench/run_in_container.sh"
DEVICE_NODE = "/dev/tenstorrent/1"
HOST_DEVICE_ID = 1
CONTAINER_VISIBLE_DEVICE_ID = 0
DEVICE_ID = HOST_DEVICE_ID
BOARD_SERIAL = "0000040331914124"
BOARD_ID = BOARD_SERIAL
BUS_ID = "0000:09:00.0"
COOLDOWN_CAP_SECONDS = 600.0
COOLDOWN_POWER_TOLERANCE_W = 5.0
COOLDOWN_TEMPERATURE_TOLERANCE_C = 3.0
TRACE_INTERVAL_SECONDS = 2.0
_ENV_BOARD_SERIAL_FIELD = "board.serial"
_ENV_BOARD_ID_FIELD = "board.identity.board_id"
_ENV_BUS_ID_FIELD = "board.identity.bus_id"
_ENV_FIRMWARE_BUNDLE_FIELD = "firmware.bundle"
ISSUE146_POWER_TRACE_SCHEMA = power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA
ISSUE146_POWER_TRACE_COLUMNS = power_trace_issue146.ISSUE146_POWER_TRACE_COLUMNS
SMOKE_IDLE_WINDOW = (10.0, 40.0)
LONG_IDLE_WINDOW = (60.0, 180.0)
DIAGNOSTIC_TAIL_LINES = 20
DIAGNOSTIC_TAIL_CHARS = 2000
AUTHORITATIVE_PLAN_RAW_SHA256 = issue146_contract.AUTHORITATIVE_PLAN_RAW_SHA256
AUTHORITATIVE_PLAN_CANONICAL_SHA256 = issue146_contract.AUTHORITATIVE_PLAN_CANONICAL_SHA256
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIAGNOSTIC_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
_DIAGNOSTIC_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?<!/)/(?!/)(?!(?:work|out)(?:/|$)|dev/tenstorrent(?:/|$))"
    r"(?=[A-Za-z0-9_.~+-])[^\s\"'<>()[\]{};,]*"
)
_DIAGNOSTIC_SAFE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:work|out)(?=/|$)"
    r"(?:/[^\s\"'<>()[\]{};,]*)?"
    r"|(?<![A-Za-z0-9_.-])/dev/tenstorrent(?=/|$)"
    r"(?:/[^\s\"'<>()[\]{};,]*)?"
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_RESULT_MARKER_RE = re.compile(
    r"(?:results?|issue146 result)\s*[-=]*>\s*(\S+)", re.IGNORECASE
)
_FORBIDDEN_KEYS = frozenset(
    {
        "hostname",
        "host_name",
        "username",
        "user_name",
        "worktree_path",
        "repository_path",
        "working_directory",
        "cwd",
    }
)


class Issue146Error(ValueError):
    """Raised when the Issue #146 plan, identity, or record is unsafe."""


@dataclass(frozen=True)
class Issue146Configuration:
    """One owner-plan configuration with the inherited #137 core plan."""

    identifier: str
    worker_core_count: int
    role: str
    verdict_bearing: bool
    core_plan: NewtonSchulzCorePlan

    @property
    def total_blocks(self) -> int:
        return self.core_plan.total_blocks

    def as_record(self) -> dict[str, Any]:
        core = self.core_plan.as_record()
        return {
            "id": self.identifier,
            "worker_core_count": self.worker_core_count,
            "role": self.role,
            "verdict_bearing": self.verdict_bearing,
            "total_blocks": self.total_blocks,
            "batch": self.core_plan.batch,
            "matrix_block": self.core_plan.matrix_block,
            "worker_core_coordinates": [list(core) for core in self.core_plan.worker_cores],
            "worker_pool": [list(core) for core in self.core_plan.worker_pool],
            "block_distribution": [dict(row) for row in self.core_plan.block_distribution],
            "work_distribution": [dict(row) for row in self.core_plan.work_distribution],
            "l1_ledger": dict(self.core_plan.l1_ledger),
            "selection_rule": core["selection_rule"],
            "core_plan": core,
        }


@dataclass(frozen=True)
class Issue146RunOptions:
    """Injected execution seams and one approved run identity."""

    run_number: int | None = None
    approve_long_run: bool = False
    command_runner: Callable[..., Any] | None = None
    environment: Mapping[str, Any] | None = None
    preceding_pre_snapshot: Mapping[str, Any] | None = None
    cooldown_readings: Sequence[Mapping[str, Any]] = ()
    cooldown_elapsed_seconds: float = 0.0
    board_id: str | None = None
    bus_id: str | None = None
    trace_path: Path | None = None
    groups_path: Path | None = None
    trace_interval_seconds: float = TRACE_INTERVAL_SECONDS
    capture_fn: Callable[..., dict[str, Any]] | None = None


@dataclass(frozen=True)
class _PreparedRun:
    safe_run_id: str
    timing: dict[str, Any]
    board_id: str
    bus_id: str
    output_dir: Path
    trace_path: Path | None
    groups_path: Path
    wrapper_stdout_path: Path
    wrapper_stderr_path: Path
    command: tuple[str, ...]
    record: dict[str, Any]


@dataclass
class _ExecutionState:
    captures: list[dict[str, Any]]
    publication_state: Any
    trace_stop: threading.Event
    trace_thread: threading.Thread | None = None
    completed: Any = None
    result: dict[str, Any] | None = None
    result_name: str | None = None
    returncode: int | None = None
    timed_out: bool = False
    cleanup_failure: str | None = None
    result_error: str | None = None
    command_error: str | None = None
    command_exception_type: str | None = None
    stdout: Any = None
    stderr: Any = None
    wrapper_diagnostics: dict[str, Any] | None = None
    diagnostics_error: str | None = None
    device_groups: dict[str, Any] | None = None
    companion_power_trace: dict[str, Any] | None = None


def _canonical_plan_hash(plan: Mapping[str, Any]) -> str:
    try:
        raw = strict_json_dumps(plan, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Issue146Error("Issue #146 plan cannot be hashed") from exc
    return hashlib.sha256(raw).hexdigest()


def _plan_hash(plan: Mapping[str, Any]) -> str:
    digest = _canonical_plan_hash(plan)
    if digest != AUTHORITATIVE_PLAN_CANONICAL_SHA256:
        raise Issue146Error("Issue #146 plan is not the immutable owner revision 4")
    return digest


def _trusted_path(path: Path, *, field: str) -> Path:
    try:
        return issue137._trusted_path(Path(path), field=field)
    except issue137.Issue137Error as exc:
        raise Issue146Error(str(exc)) from exc


def _safe_basename(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise Issue146Error(f"{field} must be a basename")
    return value


def _safe_output_dir(path: Path) -> Path:
    try:
        return issue137._safe_output_dir(Path(path))
    except issue137.Issue137Error as exc:
        raise Issue146Error(str(exc)) from exc


def resolve_result_path(
    *, cli_path: Path | None = None, environment_path: Path | None = None
) -> Path:
    """Apply the wrapper-owned result-path equality contract board-free."""
    if cli_path is None and environment_path is None:
        raise Issue146Error("Issue #146 result path is required")
    if cli_path is not None and environment_path is not None:
        first = _trusted_path(cli_path, field="--out destination")
        second = _trusted_path(environment_path, field="HEKATUS_TT_RESULT_PATH destination")
        if first != second:
            raise Issue146Error("--out and HEKATUS_TT_RESULT_PATH must identify one result path")
        return first
    return _trusted_path(cli_path or environment_path, field="Issue #146 result destination")


def _safe_artifact_path(output_dir: Path, name: str, *, field: str) -> Path:
    try:
        return issue137._safe_artifact_path(output_dir, _safe_basename(name, field=field), field=field)
    except issue137.Issue137Error as exc:
        raise Issue146Error(str(exc)) from exc


def _safe_json_value(value: Any) -> Any:
    return issue137._safe_json_value(value)


def _write_artifact(path: Path, payload: Any) -> bytes:
    destination = _trusted_path(path, field="Issue #146 artifact")
    if destination.exists() or destination.is_symlink():
        raise Issue146Error(f"Issue #146 artifact already exists: {destination.name}")
    try:
        encoded = (strict_json_dumps(_safe_json_value(payload), indent=2) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(destination, flags, 0o666)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except (OSError, TypeError, ValueError) as exc:
        raise Issue146Error(f"Issue #146 artifact could not be written: {destination.name}") from exc
    return encoded


def _write_bytes_artifact(path: Path, payload: bytes, *, field: str) -> bytes:
    destination = _trusted_path(path, field=field)
    if destination.exists() or destination.is_symlink():
        raise Issue146Error(f"{field} collides with an existing artifact: {destination.name}")
    descriptor = -1
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(destination, flags, 0o666)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
    except (OSError, TypeError, ValueError) as exc:
        raise Issue146Error(f"{field} could not be written: {destination.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return payload


def _diagnostic_path_prefixes(output_dir: Path) -> tuple[str, ...]:
    candidates: list[Path | str] = list(issue137._private_path_prefixes())
    candidates.extend((_REPOSITORY_ROOT, output_dir, Path.cwd()))
    prefixes: set[str] = set()
    for candidate in candidates:
        try:
            path = Path(candidate).resolve()
        except OSError:
            path = Path(candidate)
        normalized = path.as_posix().rstrip("/")
        if normalized:
            prefixes.add(normalized)
    return tuple(sorted(prefixes, key=len, reverse=True))


def _diagnostic_host_tokens() -> tuple[str, ...]:
    values: set[str] = set()
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            value = getter()
        except OSError:
            continue
        if not isinstance(value, str):
            continue
        token = value.strip().rstrip(".")
        if len(token) >= 2:
            values.add(token)
            short = token.split(".", 1)[0]
            if len(short) >= 2:
                values.add(short)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_diagnostic_hostnames(
    text: str, *, host_tokens: tuple[str, ...] | None = None
) -> str:
    tokens = _diagnostic_host_tokens() if host_tokens is None else host_tokens
    if not tokens:
        return text
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.-])(?:"
        + "|".join(re.escape(token) for token in tokens)
        + r")(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    )
    return pattern.sub("<redacted>", text)


def _redact_diagnostic_text(
    value: Any,
    *,
    output_dir: Path,
    host_tokens: tuple[str, ...] | None = None,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)
    if host_tokens is None:
        host_tokens = _diagnostic_host_tokens()
    protected_urls: list[str] = []
    protected_safe_paths: list[str] = []

    def protect_url(match: re.Match[str]) -> str:
        protected_urls.append(
            _redact_diagnostic_hostnames(match.group(0), host_tokens=host_tokens)
        )
        return f"\\x00issue146-url-{len(protected_urls) - 1}\\x00"

    def protect_safe_path(match: re.Match[str]) -> str:
        protected_safe_paths.append(match.group(0))
        return f"\\x00issue146-safe-path-{len(protected_safe_paths) - 1}\\x00"

    text = _DIAGNOSTIC_URL_RE.sub(protect_url, text)
    text = _DIAGNOSTIC_SAFE_PATH_RE.sub(protect_safe_path, text)
    for prefix in _diagnostic_path_prefixes(output_dir):
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.-])(?<!/)"
            + re.escape(prefix)
            + r"(?:/[^\s\"'<>()[\]{};,]*)?"
        )
        text = pattern.sub("<redacted-path>", text)
    text = _DIAGNOSTIC_ABSOLUTE_PATH_RE.sub("<redacted-path>", text)
    text = _redact_diagnostic_hostnames(text, host_tokens=host_tokens)
    for index, url in enumerate(protected_urls):
        text = text.replace(f"\\x00issue146-url-{index}\\x00", url)
    for index, path in enumerate(protected_safe_paths):
        text = text.replace(
            f"\\x00issue146-safe-path-{index}\\x00",
            _redact_diagnostic_hostnames(path, host_tokens=host_tokens),
        )
    return text


def _redact_diagnostic_value(
    value: Any,
    *,
    output_dir: Path,
    host_tokens: tuple[str, ...] | None = None,
) -> Any:
    if host_tokens is None:
        host_tokens = _diagnostic_host_tokens()
    safe = _safe_json_value(value)
    if isinstance(safe, Mapping):
        return {
            str(key): _redact_diagnostic_value(
                child, output_dir=output_dir, host_tokens=host_tokens
            )
            for key, child in safe.items()
        }
    if isinstance(safe, list):
        return [
            _redact_diagnostic_value(
                child, output_dir=output_dir, host_tokens=host_tokens
            )
            for child in safe
        ]
    if isinstance(safe, str):
        return _redact_diagnostic_text(
            safe, output_dir=output_dir, host_tokens=host_tokens
        )
    return safe


def _diagnostic_tail(text: str) -> list[str]:
    lines = text.splitlines()
    bounded = lines[-DIAGNOSTIC_TAIL_LINES:]
    return [
        line if len(line) <= DIAGNOSTIC_TAIL_CHARS
        else line[:DIAGNOSTIC_TAIL_CHARS] + "..."
        for line in bounded
    ]


def _persist_wrapper_diagnostics(
    prepared: _PreparedRun, state: _ExecutionState
) -> dict[str, Any]:
    host_tokens = _diagnostic_host_tokens()
    stdout = _redact_diagnostic_text(
        state.stdout, output_dir=prepared.output_dir, host_tokens=host_tokens
    )
    stderr = _redact_diagnostic_text(
        state.stderr, output_dir=prepared.output_dir, host_tokens=host_tokens
    )
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    _write_bytes_artifact(
        prepared.wrapper_stdout_path,
        stdout_bytes,
        field="Issue #146 wrapper stdout",
    )
    _write_bytes_artifact(
        prepared.wrapper_stderr_path,
        stderr_bytes,
        field="Issue #146 wrapper stderr",
    )
    return {
        "stdout_path": prepared.wrapper_stdout_path.name,
        "stderr_path": prepared.wrapper_stderr_path.name,
        "stdout_byte_count": len(stdout_bytes),
        "stderr_byte_count": len(stderr_bytes),
        "stdout_tail": _diagnostic_tail(stdout),
        "stderr_tail": _diagnostic_tail(stderr),
    }


def _load_plan_path(path: Path) -> Path:
    resolved = _trusted_path(path, field="Issue #146 plan")
    if not resolved.is_file():
        raise Issue146Error("Issue #146 plan path is not a file")
    return resolved


def load_plan(path: Path = DEFAULT_PLAN_PATH) -> dict[str, Any]:
    """Load the exact immutable owner revision-4 plan without changing bytes."""
    source = _load_plan_path(Path(path))
    try:
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != AUTHORITATIVE_PLAN_RAW_SHA256:
            raise Issue146Error("Issue #146 plan bytes are not the immutable owner revision")
        payload = json.loads(raw.decode("utf-8"))
    except Issue146Error:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue146Error("cannot read Issue #146 plan") from exc
    if not isinstance(payload, dict):
        raise Issue146Error("Issue #146 plan must be an object")
    if payload.get("record_schema") != "issue146-premeasurement-plan-v1":
        raise Issue146Error("unsupported Issue #146 plan schema")
    if payload.get("status") != "planned" or payload.get("board_run") is not False:
        raise Issue146Error("Issue #146 plan is not the immutable board-free plan")
    history = payload.get("revision_history")
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(history[-1], Mapping)
        or history[-1].get("revision") != CURRENT_PLAN_REVISION
    ):
        raise Issue146Error(f"Issue #146 plan revision {CURRENT_PLAN_REVISION} is required")
    execution = payload.get("load_execution")
    if not isinstance(execution, Mapping) or execution.get("mode") != ISSUE146_LOAD_MODE + ", grouped launches":
        raise Issue146Error("Issue #146 grouped execution mode is not selected")
    _plan_hash(payload)
    grouped_load_settings(payload)
    return payload


def plan_configurations(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = plan.get("configurations")
    if not isinstance(values, list) or not values:
        raise Issue146Error("Issue #146 configurations are missing")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise Issue146Error("Issue #146 configuration must be an object")
        identifier = item.get("id")
        count = item.get("worker_core_count")
        if (
            not isinstance(identifier, str)
            or identifier in seen
            or identifier not in {"C018", "C081", "C108"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("verdict_bearing"), bool)
        ):
            raise Issue146Error("Issue #146 configuration fields are invalid")
        seen.add(identifier)
        result.append(dict(item))
    if set(seen) != {"C018", "C081", "C108"}:
        raise Issue146Error("Issue #146 configurations must be C018, C081, and C108")
    return result


def configuration(plan: Mapping[str, Any], identifier: str) -> Issue146Configuration:
    matches = [item for item in plan_configurations(plan) if item["id"] == identifier]
    if len(matches) != 1:
        raise Issue146Error(f"unknown Issue #146 configuration {identifier!r}")
    item = matches[0]
    try:
        core_plan = build_newton_schulz_core_plan(
            core_count=item["worker_core_count"], matrix_block=8
        )
    except (TypeError, ValueError) as exc:
        raise Issue146Error(str(exc)) from exc
    return Issue146Configuration(
        identifier=identifier,
        worker_core_count=item["worker_core_count"],
        role=item["role"],
        verdict_bearing=item["verdict_bearing"],
        core_plan=core_plan,
    )


def configuration_for_core_count(plan: Mapping[str, Any], core_count: int) -> Issue146Configuration:
    matches = [item for item in plan_configurations(plan) if item["worker_core_count"] == core_count]
    if len(matches) != 1:
        raise Issue146Error("Issue #146 core count is not a unique planned configuration")
    return configuration(plan, str(matches[0]["id"]))


def long_run_order(plan: Mapping[str, Any]) -> list[str]:
    _plan_hash(plan)
    scope = plan.get("measurement_scope")
    order = scope.get("order") if isinstance(scope, Mapping) else None
    if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
        raise Issue146Error("Issue #146 long-run order is missing")
    expected = ["C018", "C108", "C081", "C081", "C108", "C018"]
    if order != expected:
        raise Issue146Error("Issue #146 long-run order is not the owner revision-4 order")
    return list(order)


def grouped_load_settings(plan: Mapping[str, Any]) -> tuple[int, int]:
    """Read positive grouped counts from the already pinned owner plan."""
    _plan_hash(plan)
    execution = plan.get("load_execution")
    if not isinstance(execution, Mapping):
        raise Issue146Error("Issue #146 load execution is missing")
    try:
        outer = execution["outer_repetitions"]
        launches = execution["launches_per_sync"]
    except KeyError as exc:
        raise Issue146Error("Issue #146 grouped load settings are missing") from exc
    if (
        isinstance(outer, bool) or not isinstance(outer, int) or outer <= 0
        or isinstance(launches, bool) or not isinstance(launches, int) or launches <= 0
    ):
        raise Issue146Error("Issue #146 grouped load settings must be positive integers")
    return outer, launches


def run_timing(plan: Mapping[str, Any], mode: str) -> dict[str, Any]:
    scope = plan.get("measurement_scope")
    if not isinstance(scope, Mapping):
        raise Issue146Error("Issue #146 measurement scope is missing")
    if mode == "long":
        values = scope
        target_key, guard_key = "run_duration_seconds", "run_guard_seconds"
        diagnostic = False
    elif mode == "smoke":
        values = scope.get("smoke_runs")
        target_key, guard_key = "workload_seconds", "guard_seconds"
        diagnostic = True
    else:
        raise Issue146Error("Issue #146 mode must be smoke or long")
    if not isinstance(values, Mapping):
        raise Issue146Error(f"Issue #146 {mode} timing is missing")
    try:
        target = float(values[target_key])
        guard = float(values[guard_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise Issue146Error(f"Issue #146 {mode} timing is invalid") from exc
    expected = (40.0, 90.0) if mode == "smoke" else (180.0, 230.0)
    if (target, guard) != expected:
        raise Issue146Error(f"Issue #146 {mode} timing does not match revision 4")
    if guard < target + 40.0 + DEFAULT_OVERRUN_LIMIT_SECONDS:
        raise Issue146Error("Issue #146 guard does not cover runner margin and overrun")
    return {"load_seconds": target, "guard_seconds": guard, "diagnostic_only": diagnostic}


def _selected_ids(plan: Mapping[str, Any], options: Issue146RunOptions) -> tuple[str, str]:
    scope = plan.get("measurement_scope")
    expected_board = scope.get("board_serial") if isinstance(scope, Mapping) else BOARD_SERIAL
    expected_bus = scope.get("pci_bus_id") if isinstance(scope, Mapping) else BUS_ID
    board_id = options.board_id or expected_board
    bus_id = options.bus_id or expected_bus
    if board_id != BOARD_SERIAL or bus_id != BUS_ID:
        raise Issue146Error("Issue #146 selected board identity does not match the owner plan")
    return str(board_id), str(bus_id)


def validate_selected_identity(
    *,
    device_node: Any,
    host_device_id: Any,
    container_device_id: Any,
    serial: Any,
    board_id: Any,
    bus_id: Any,
) -> None:
    if device_node != DEVICE_NODE:
        raise Issue146Error(f"Issue #146 requires device_node {DEVICE_NODE}")
    if isinstance(host_device_id, bool) or host_device_id != HOST_DEVICE_ID:
        raise Issue146Error("Issue #146 host device_id must be 1")
    if isinstance(container_device_id, bool) or container_device_id != CONTAINER_VISIBLE_DEVICE_ID:
        raise Issue146Error("Issue #146 container device_id must be 0")
    if serial != BOARD_SERIAL or board_id != BOARD_SERIAL:
        raise Issue146Error("Issue #146 board serial and board_id must match the selected board")
    if bus_id != BUS_ID:
        raise Issue146Error("Issue #146 PCI bus identity does not match the selected board")


def _environment_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    board = environment.get("board") or environment.get("board_info")
    if not isinstance(board, Mapping):
        raise Issue146Error("environment board identity is missing")
    identity = board.get("identity") if isinstance(board.get("identity"), Mapping) else {}
    serial = board.get("serial") or board.get("board_id") or identity.get("board_id")
    board_id = board.get("board_id") or identity.get("board_id") or serial
    bus_id = board.get("bus_id") or board.get("pci_bus_id") or identity.get("bus_id")
    return {"serial": serial, "board_id": board_id, "bus_id": bus_id}


def _environment_digest(environment: Mapping[str, Any]) -> str | None:
    value = environment.get("image_digest")
    if isinstance(value, str):
        return value
    image = environment.get("image")
    if isinstance(image, str) and "@sha256:" in image:
        return image.rsplit("@", 1)[1]
    return None


def _required_environment_fields(plan: Mapping[str, Any]) -> tuple[str, ...]:
    provenance = plan.get("provenance")
    fields = provenance.get("required_environment_fields") if isinstance(provenance, Mapping) else None
    if not isinstance(fields, list) or not fields or not all(isinstance(item, str) for item in fields):
        raise Issue146Error("Issue #146 required environment fields are missing")
    return tuple(fields)


def _environment_provenance_values(
    environment: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    firmware = environment.get("firmware") or environment.get("firmwares")
    if not isinstance(firmware, Mapping):
        firmware = {}
    return {
        _ENV_BOARD_SERIAL_FIELD: identity.get("serial"),
        _ENV_BOARD_ID_FIELD: identity.get("board_id"),
        _ENV_BUS_ID_FIELD: identity.get("bus_id"),
        _ENV_FIRMWARE_BUNDLE_FIELD: firmware.get("bundle") or firmware.get("fw_bundle_version"),
        "kernel": environment.get("kernel"),
        "kernel_driver_version": environment.get("kernel_driver_version") or environment.get("kmd_version"),
        "image_digest": _environment_digest(environment),
        "harness_commit": environment.get("harness_commit"),
        "device_node": environment.get("device_node"),
        "worker_core_count": environment.get("worker_core_count"),
        "worker_core_coordinates": environment.get("worker_core_coordinates"),
        "total_blocks": environment.get("total_blocks"),
        "outer_repetitions": environment.get("outer_repetitions"),
        "launches_per_sync": environment.get("launches_per_sync"),
    }


def _validate_environment_identity(environment: Mapping[str, Any]) -> dict[str, Any]:
    if environment.get("device_node") != DEVICE_NODE:
        raise Issue146Error(f"Issue #146 requires device_node {DEVICE_NODE}")
    identity = _environment_identity(environment)
    validate_selected_identity(
        device_node=environment.get("device_node"),
        host_device_id=environment.get("device_id", environment.get("host_device_id", HOST_DEVICE_ID)),
        container_device_id=environment.get(
            "container_device_id", environment.get("container_visible_device_id", CONTAINER_VISIBLE_DEVICE_ID)
        ),
        serial=identity["serial"],
        board_id=identity["board_id"],
        bus_id=identity["bus_id"],
    )
    return identity


def _validate_environment_configuration(
    environment: Mapping[str, Any], configuration: Issue146Configuration | None,
) -> None:
    if configuration is None:
        return
    if environment.get("worker_core_count") != configuration.worker_core_count:
        raise Issue146Error("environment worker_core_count does not match the plan")
    if environment.get("total_blocks") != configuration.total_blocks:
        raise Issue146Error("environment total_blocks does not match the plan")
    expected_coordinates = [list(core) for core in configuration.core_plan.worker_cores]
    if environment.get("worker_core_coordinates") != expected_coordinates:
        raise Issue146Error("environment worker_core_coordinates do not match the plan")


def _validate_environment_fields(
    environment: Mapping[str, Any], *, plan: Mapping[str, Any], identity: Mapping[str, Any],
    outer_repetitions: int, launches_per_sync: int,
) -> None:
    values = _environment_provenance_values(environment, identity)
    missing = [
        field for field in _required_environment_fields(plan)
        if values.get(field) is None or values.get(field) == "" or values.get(field) == []
    ]
    if missing:
        raise Issue146Error("environment provenance is missing: " + ", ".join(missing))
    if environment.get("outer_repetitions") != outer_repetitions:
        raise Issue146Error("environment outer_repetitions does not match the owner plan")
    if environment.get("launches_per_sync") != launches_per_sync:
        raise Issue146Error("environment launches_per_sync does not match the owner plan")


def _validate_environment_versions(environment: Mapping[str, Any]) -> None:
    digest = _environment_digest(environment)
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise Issue146Error("environment image_digest is not a pinned SHA-256")
    commit = environment.get("harness_commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise Issue146Error("environment harness_commit is invalid")
    host_versions = environment.get("host_sw_vers")
    if not isinstance(host_versions, Mapping):
        raise Issue146Error("environment host reader provenance is missing")
    missing_readers = [
        name for name in ("tt_smi", "tt_umd")
        if name not in host_versions or host_versions.get(name) in (None, "")
    ]
    if missing_readers:
        raise Issue146Error("environment host reader versions are missing: " + ", ".join(missing_readers))
    sampler = environment.get("telemetry_sampler")
    if sampler is not None and not isinstance(sampler, Mapping):
        raise Issue146Error("telemetry_sampler provenance must be an object")


def validate_environment_for_run(
    environment: Mapping[str, Any], *, configuration: Issue146Configuration | None = None,
    plan: Mapping[str, Any] | None = None,
) -> None:
    """Validate every required provenance field before importing or opening TTNN."""
    if not isinstance(environment, Mapping):
        raise Issue146Error("environment must be an object")
    effective_plan = load_plan() if plan is None else plan
    _plan_hash(effective_plan)
    outer_repetitions, launches_per_sync = grouped_load_settings(effective_plan)
    identity = _validate_environment_identity(environment)
    try:
        validate_harness_preflight(environment, pushed=environment.get("harness_pushed"))
    except Exception as exc:
        raise Issue146Error(str(exc)) from exc
    _validate_environment_configuration(environment, configuration)
    _validate_environment_fields(
        environment, plan=effective_plan, identity=identity,
        outer_repetitions=outer_repetitions, launches_per_sync=launches_per_sync,
    )
    _validate_environment_versions(environment)


def build_environment_record(
    environment: Mapping[str, Any], configuration: Issue146Configuration,
    *, plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist required environment provenance without host identity or paths."""
    effective_plan = load_plan() if plan is None else plan
    outer_repetitions, launches_per_sync = grouped_load_settings(effective_plan)
    validate_environment_for_run(environment, configuration=configuration, plan=effective_plan)
    identity = _environment_identity(environment)
    firmware = environment.get("firmware") or environment.get("firmwares")
    if not isinstance(firmware, Mapping):
        raise Issue146Error("environment firmware provenance is missing")
    kernel_driver = environment.get("kernel_driver_version") or environment.get("kmd_version")
    facts = {
        _ENV_BOARD_SERIAL_FIELD: identity["serial"],
        _ENV_BOARD_ID_FIELD: identity["board_id"],
        _ENV_BUS_ID_FIELD: identity["bus_id"],
        _ENV_FIRMWARE_BUNDLE_FIELD: firmware.get("bundle") or firmware.get("fw_bundle_version"),
        "kernel": environment.get("kernel"),
        "kernel_driver_version": kernel_driver,
        "image_digest": _environment_digest(environment),
        "harness_commit": environment.get("harness_commit"),
        "device_node": environment.get("device_node"),
    }
    missing = [name for name, value in facts.items() if not isinstance(value, str) or not value]
    if missing:
        raise Issue146Error("environment provenance is missing: " + ", ".join(missing))
    safe = _safe_json_value(environment)
    record: dict[str, Any] = {
        "board": {
            "serial": facts[_ENV_BOARD_SERIAL_FIELD],
            "identity": {
                "board_id": facts[_ENV_BOARD_ID_FIELD],
                "bus_id": facts[_ENV_BUS_ID_FIELD],
            },
        },
        "firmware": {"bundle": facts[_ENV_FIRMWARE_BUNDLE_FIELD]},
        "kernel": facts["kernel"],
        "kernel_driver_version": facts["kernel_driver_version"],
        "image_digest": facts["image_digest"],
        "harness_commit": facts["harness_commit"],
        "harness_dirty": environment.get("harness_dirty"),
        "harness_pushed": environment.get("harness_pushed"),
        "device_node": facts["device_node"],
        "device_id": HOST_DEVICE_ID,
        "container_device_id": CONTAINER_VISIBLE_DEVICE_ID,
        "worker_core_count": configuration.worker_core_count,
        "worker_core_coordinates": [list(core) for core in configuration.core_plan.worker_cores],
        "total_blocks": configuration.total_blocks,
        "outer_repetitions": outer_repetitions,
        "launches_per_sync": launches_per_sync,
        "core_block_distribution": [dict(row) for row in configuration.core_plan.block_distribution],
        "safe_environment_digest": hashlib.sha256(
            strict_json_dumps(safe, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    host_versions = environment.get("host_sw_vers")
    if isinstance(host_versions, Mapping):
        record["host_sw_vers"] = _safe_json_value(host_versions)
    sampler = environment.get("telemetry_sampler")
    if isinstance(sampler, Mapping):
        record["telemetry_sampler"] = _safe_json_value(sampler)
    return record


def _snapshot_device(capture: Mapping[str, Any], *, board_id: str, bus_id: str) -> Mapping[str, Any]:
    snapshot = capture.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise Issue146Error("snapshot payload is missing")
    try:
        from enodia.tt.bench.power_trace_extended import select_extended_board

        device = select_extended_board(snapshot, board_id=board_id, bus_id=bus_id)
    except (TypeError, ValueError, KeyError) as exc:
        raise Issue146Error("snapshot selected-device identity is invalid") from exc
    board_info = device.get("board_info")
    if not isinstance(board_info, Mapping):
        raise Issue146Error("snapshot selected-device board_info is missing")
    serial = board_info.get("serial")
    if serial is not None and str(serial) != str(board_id):
        raise Issue146Error("snapshot board_info.serial does not match the selected board")
    return device


def validate_capture(
    capture: Mapping[str, Any], *, stage: str, board_id: str, bus_id: str
) -> None:
    if capture.get("schema") != ISSUE146_SNAPSHOT_SCHEMA:
        raise Issue146Error(f"{stage} snapshot schema is unsupported")
    if capture.get("stage") not in {None, stage}:
        raise Issue146Error(f"{stage} snapshot stage is invalid")
    if capture.get("status") != "ok" or capture.get("returncode") != 0:
        raise Issue146Error(f"{stage} snapshot capture did not complete successfully")
    timestamp = capture.get("timestamp_monotonic_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise Issue146Error(f"{stage} snapshot monotonic timestamp is invalid")
    _snapshot_device(capture, board_id=board_id, bus_id=bus_id)


def _raw_word(value: Any, *, field: str) -> int:
    if not isinstance(value, str) or not value.startswith(("0x", "0X")):
        raise Issue146Error(f"{field} must be a raw hexadecimal word")
    try:
        number = int(value, 16)
    except (TypeError, ValueError) as exc:
        raise Issue146Error(f"{field} must be a raw hexadecimal word") from exc
    if not 0 <= number <= 0xFFFFFFFF:
        raise Issue146Error(f"{field} must be a 32-bit raw hexadecimal word")
    return number


def _limit_number(value: Any, *, field: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, bool) or value is None:
        raise Issue146Error(f"{field} is missing")
    try:
        number = float(str(value).strip().removesuffix("W").removesuffix("MHz").removesuffix("degC").strip())
    except (TypeError, ValueError) as exc:
        raise Issue146Error(f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise Issue146Error(f"{field} is not finite")
    return number


def validate_pre_snapshot_configuration(
    capture: Mapping[str, Any], *, plan: Mapping[str, Any], board_id: str, bus_id: str
) -> dict[str, Any]:
    """Check plan words and numeric limits before any workload starts."""
    device = _snapshot_device(capture, board_id=board_id, bus_id=bus_id)
    smbus = device.get("smbus_telem")
    limits = device.get("limits")
    if not isinstance(smbus, Mapping):
        raise Issue146Error("pre-run snapshot smbus_telem is missing")
    if not isinstance(limits, Mapping):
        raise Issue146Error("pre-run snapshot limits are missing")
    board_limit_word = _raw_word(smbus.get("BOARD_POWER_LIMIT"), field="BOARD_POWER_LIMIT")
    enabled_max_word = _raw_word(smbus.get("ENABLED_MAX_ARB"), field="ENABLED_MAX_ARB")
    if board_limit_word != 0x12C:
        raise Issue146Error("pre-run BOARD_POWER_LIMIT does not match 0x12c")
    if enabled_max_word != 0x1D1:
        raise Issue146Error("pre-run ENABLED_MAX_ARB does not match 0x1d1")
    configured = {
        "board_power_limit_w": _limit_number(limits.get("board_power_limit"), field="limits.board_power_limit"),
        "asic_fmax_mhz": _limit_number(limits.get("asic_fmax"), field="limits.asic_fmax"),
        "therm_trip_l1_limit_c": _limit_number(
            limits.get("therm_trip_l1_limit"), field="limits.therm_trip_l1_limit"
        ),
        "gddr_thm_limit_c": float(plan["limits"]["gddr_thm_limit_c"]),
        "gddr_thm_limit_source": "firmware_config_table.pinned_19.6.0.0",
        "board_power_limit_word": board_limit_word,
        "enabled_max_arb_word": enabled_max_word,
    }
    expected = plan["limits"]
    if configured["board_power_limit_w"] != float(expected["board_power_limit_w"]):
        raise Issue146Error("limits.board_power_limit does not match the plan")
    if configured["asic_fmax_mhz"] != float(expected["asic_fmax_mhz"]):
        raise Issue146Error("limits.asic_fmax does not match the plan")
    if configured["therm_trip_l1_limit_c"] != float(expected["thm_limit_c"]):
        raise Issue146Error("limits.therm_trip_l1_limit does not match the plan")
    return configured


def capture_readonly_snapshot(*, stage: str, **kwargs: Any) -> dict[str, Any]:
    """Capture through the established bounded Q2 read-only command seam."""
    capture = dict(issue137.capture_readonly_snapshot(stage=stage, **kwargs))
    capture["schema"] = ISSUE146_SNAPSHOT_SCHEMA
    return capture


def _load_preceding_pre_snapshot(path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    source = _trusted_path(path, field="Issue #146 preceding pre-run snapshot")
    if not source.is_file():
        raise Issue146Error("Issue #146 preceding pre-run snapshot is not a file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue146Error("Issue #146 preceding pre-run snapshot is malformed") from exc
    if not isinstance(payload, Mapping):
        raise Issue146Error("Issue #146 preceding pre-run snapshot must be an object")
    if payload.get("stage") != "pre_run":
        raise Issue146Error("Issue #146 preceding pre-run snapshot stage must be pre_run")
    board_id, bus_id = _selected_ids(plan, Issue146RunOptions())
    validate_capture(payload, stage="pre_run", board_id=board_id, bus_id=bus_id)
    return dict(payload)


def _snapshot_baseline(capture: Mapping[str, Any], *, board_id: str, bus_id: str) -> dict[str, float]:
    device = _snapshot_device(capture, board_id=board_id, bus_id=bus_id)
    telemetry = device.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise Issue146Error("cooldown snapshot telemetry is unavailable")
    try:
        return {
            "power_w": float(telemetry["power"]),
            "asic_temp_c": float(telemetry["asic_temperature"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise Issue146Error("cooldown snapshot is missing power or ASIC temperature") from exc


def _cooldown_settings(plan: Mapping[str, Any]) -> tuple[float, float, float]:
    cooldown = plan.get("cooldown")
    if not isinstance(cooldown, Mapping):
        raise Issue146Error("Issue #146 cooldown plan is missing")
    try:
        cap = float(cooldown["cap_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Issue146Error("Issue #146 cooldown cap is invalid") from exc
    criterion = str(cooldown.get("criterion", ""))
    power_match = re.search(
        r"power_w\s*<=\s*baseline\s*\+\s*(\d+(?:\.\d+)?)",
        criterion, flags=re.ASCII,
    )
    temperature_match = re.search(
        r"asic_temp_c\s*<=\s*baseline\s*\+\s*(\d+(?:\.\d+)?)",
        criterion, flags=re.ASCII,
    )
    if not power_match or not temperature_match:
        raise Issue146Error("Issue #146 cooldown tolerances are missing")
    power_tolerance = float(power_match.group(1))
    temperature_tolerance = float(temperature_match.group(1))
    if (cap, power_tolerance, temperature_tolerance) != (
        COOLDOWN_CAP_SECONDS, COOLDOWN_POWER_TOLERANCE_W, COOLDOWN_TEMPERATURE_TOLERANCE_C
    ):
        raise Issue146Error("Issue #146 cooldown settings do not match revision 4")
    return cap, power_tolerance, temperature_tolerance


def evaluate_cooldown(
    baseline: Mapping[str, Any], observed: Mapping[str, Any], *,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    try:
        baseline_power = float(baseline["power_w"])
        baseline_temp = float(baseline["asic_temp_c"])
        observed_power = float(observed["power_w"])
        observed_temp = float(observed["asic_temp_c"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Issue146Error("cooldown reading is missing required telemetry") from exc
    power_pass = observed_power <= baseline_power + power_tolerance_w
    temp_pass = observed_temp <= baseline_temp + temperature_tolerance_c
    return {
        "baseline_source": "preceding selected-device pre-run snapshot",
        "baseline_power_w": baseline_power,
        "baseline_temperature_c": baseline_temp,
        "power_tolerance_w": power_tolerance_w,
        "temperature_tolerance_c": temperature_tolerance_c,
        "power_limit_w": baseline_power + power_tolerance_w,
        "temperature_limit_c": baseline_temp + temperature_tolerance_c,
        "observed_power_w": observed_power,
        "observed_temperature_c": observed_temp,
        "power_gate": power_pass,
        "temperature_gate": temp_pass,
        "aiclk_gate": False,
        "verified": power_pass and temp_pass,
        "gate_result": "pass" if power_pass and temp_pass else "fail",
    }


def wait_for_cooldown(
    preceding_pre_snapshot: Mapping[str, Any], *,
    capture_fn: Callable[..., dict[str, Any]],
    board_id: str,
    bus_id: str,
    cap_seconds: float,
    power_tolerance_w: float,
    temperature_tolerance_c: float,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Poll the selected device until the plan-relative cooldown gate passes."""
    baseline = _snapshot_baseline(
        preceding_pre_snapshot, board_id=board_id, bus_id=bus_id
    )
    started = monotonic_fn()
    readings: list[Mapping[str, Any]] = []
    while True:
        elapsed = float(monotonic_fn() - started)
        if elapsed >= cap_seconds:
            return cooldown_gate(
                baseline, readings=readings, elapsed_seconds=cap_seconds,
                cap_seconds=cap_seconds, power_tolerance_w=power_tolerance_w,
                temperature_tolerance_c=temperature_tolerance_c,
            )
        capture = capture_fn(stage="cooldown")
        validate_capture(capture, stage="cooldown", board_id=board_id, bus_id=bus_id)
        reading = _snapshot_baseline(capture, board_id=board_id, bus_id=bus_id)
        readings.append(reading)
        evaluation = evaluate_cooldown(
            baseline, reading, power_tolerance_w=power_tolerance_w,
            temperature_tolerance_c=temperature_tolerance_c,
        )
        after_capture = float(monotonic_fn() - started)
        if evaluation["verified"]:
            evaluation.update({"poll_count": len(readings), "cooldown_wait_seconds": after_capture, "wait_cap_seconds": cap_seconds})
            return evaluation
        delay = min(1.0, max(0.0, cap_seconds - after_capture))
        if delay <= 0.0:
            return cooldown_gate(
                baseline, readings=readings, elapsed_seconds=cap_seconds,
                cap_seconds=cap_seconds, power_tolerance_w=power_tolerance_w,
                temperature_tolerance_c=temperature_tolerance_c,
            )
        sleep_fn(delay)


def cooldown_gate(
    baseline: Mapping[str, Any],
    *,
    readings: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
    cap_seconds: float = COOLDOWN_CAP_SECONDS,
    power_tolerance_w: float = COOLDOWN_POWER_TOLERANCE_W,
    temperature_tolerance_c: float = COOLDOWN_TEMPERATURE_TOLERANCE_C,
) -> dict[str, Any]:
    elapsed = float(elapsed_seconds)
    for index, reading in enumerate(readings):
        evaluation = evaluate_cooldown(
            baseline, reading, power_tolerance_w=power_tolerance_w,
            temperature_tolerance_c=temperature_tolerance_c,
        )
        evaluation.update({"poll_count": index + 1, "cooldown_wait_seconds": elapsed, "wait_cap_seconds": cap_seconds})
        if evaluation["verified"] and elapsed < cap_seconds:
            return evaluation
    return {
        "baseline_source": "preceding selected-device pre-run snapshot",
        "poll_count": len(readings),
        "cooldown_wait_seconds": elapsed,
        "wait_cap_seconds": cap_seconds,
        "power_tolerance_w": power_tolerance_w,
        "temperature_tolerance_c": temperature_tolerance_c,
        "verified": False,
        "gate_result": "fail",
        "aiclk_gate": False,
        "failure_reason": "cooldown wait cap expired" if elapsed >= COOLDOWN_CAP_SECONDS else "cooldown gate failed",
    }


def _safe_trace_or_groups_path(path: Path | None, *, output_dir: Path, field: str) -> Path | None:
    if path is None:
        return None
    resolved = _trusted_path(path, field=field)
    if resolved.parent != output_dir:
        raise Issue146Error(f"{field} must be beneath the run output directory")
    return resolved


def _reject_output_collisions(
    output_dir: Path,
    *,
    run_id: str,
    trace_path: Path | None,
    groups_path: Path,
    wrapper_stdout_path: Path,
    wrapper_stderr_path: Path,
) -> None:
    names = [
        f"run-{run_id}.json",
        f"pre-run-snapshot-{run_id}.json",
        f"post-run-snapshot-{run_id}.json",
        groups_path.name,
        f"{groups_path.stem}-envelope.json",
        wrapper_stdout_path.name,
        wrapper_stderr_path.name,
    ]
    if trace_path is not None:
        names.append(trace_path.name)
    if len(names) != len(set(names)):
        raise Issue146Error("Issue #146 output collision: artifact names overlap")
    for name in names:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            raise Issue146Error(f"Issue #146 output collision: {name}")


def build_run_command(
    *,
    plan: Mapping[str, Any],
    configuration: Issue146Configuration,
    mode: str,
    output_dir: Path,
    run_id: str,
    run_number: int | None,
    approve_long_run: bool = False,
) -> tuple[str, ...]:
    timing = run_timing(plan, mode)
    outer_repetitions, launches_per_sync = grouped_load_settings(plan)
    if mode == "long":
        if run_number is None or not 1 <= run_number <= 6:
            raise Issue146Error("long runs require run_number from 1 through 6")
        if long_run_order(plan)[run_number - 1] != configuration.identifier:
            raise Issue146Error("run_number does not match the preregistered Issue #146 order")
        if not approve_long_run:
            raise Issue146Error("long Issue #146 runs require per-instance owner approval")
    elif configuration.identifier not in set(
        plan["measurement_scope"]["smoke_runs"]["configurations"]
    ):
        raise Issue146Error("configuration is not in the preregistered Issue #146 smoke set")
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise Issue146Error(str(exc)) from exc
    output = _safe_output_dir(output_dir)
    return (
        WRAPPER_SCRIPT,
        str(output),
        "--",
        "--device-id",
        str(CONTAINER_VISIBLE_DEVICE_ID),
        "--condition",
        "newton_schulz",
        "--configuration",
        configuration.identifier,
        "--requested-run-id",
        run_id,
        "--load-core-count",
        str(configuration.worker_core_count),
        "--load-total-blocks",
        str(configuration.total_blocks),
        "--load-matrix-block",
        str(configuration.core_plan.matrix_block),
        "--outer-repetitions",
        str(outer_repetitions),
        "--launches-per-sync",
        str(launches_per_sync),
        "--target-seconds",
        str(int(timing["load_seconds"])),
        "--overrun-limit-seconds",
        str(int(DEFAULT_OVERRUN_LIMIT_SECONDS)),
        "--plan",
        str(DEFAULT_PLAN_PATH),
        "--plan-hash",
        _plan_hash(plan),
        "--mode",
        mode,
        "--outer-timeout-seconds",
        str(int(timing["guard_seconds"])),
    )


def _base_record(
    *, plan: Mapping[str, Any], configuration: Issue146Configuration, mode: str,
    run_id: str, run_number: int | None, command: Sequence[str], output_dir: Path,
) -> dict[str, Any]:
    timing = run_timing(plan, mode)
    command_record = ["<run-output-dir>" if item == str(output_dir) else item for item in command]
    return {
        "record_schema": ISSUE146_RECORD_SCHEMA,
        "record_revision": 1,
        "issue": "#146",
        "status": "planned",
        "plan_sha256": _plan_hash(plan),
        "run": {
            "run_id": run_id,
            "run_number": run_number,
            "mode": mode,
            "load_mode": ISSUE146_LOAD_MODE,
            "target_seconds": timing["load_seconds"],
            "guard_seconds": timing["guard_seconds"],
            "first_launch_start_monotonic_ns": None,
            "workload_end_monotonic_ns": None,
            "last_group_end_monotonic_ns": None,
            "observed_duration_seconds": None,
            "group_count": None,
            "launch_count": None,
            "overrun_seconds": None,
        },
        "configuration": configuration.as_record(),
        "command": command_record,
        "diagnostic_only": mode == "smoke",
        "verdict_evidence": False,
        "workload_started": False,
        "environment": None,
        "pre_run_snapshot": None,
        "post_run_snapshot": None,
        "groups": None,
        "trace": None,
        "wrapper_diagnostics": None,
        "provenance": {
            "wrapper_entrypoint": WRAPPER_SCRIPT,
            "runner_entrypoint": ISSUE146_DEVICE_RUNNER,
            "requested_run_id": run_id,
            "wrapper_run_id": None,
            "plan_sha256": _plan_hash(plan),
            "unit_provenance": None,
            "unit_provenance_status": "unknown",
            "host_identity_persisted": False,
        },
    }


def _failure_record(record: Mapping[str, Any], *, kind: str, message: str, returncode: int | None = None) -> dict[str, Any]:
    result = dict(record)
    result["status"] = "failed"
    result["verdict_evidence"] = False
    failure: dict[str, Any] = {"kind": kind, "message": message}
    if returncode is not None:
        failure["returncode"] = returncode
    result["failure"] = failure
    return result


def _persist_record(output_dir: Path, run_id: str, record: Mapping[str, Any]) -> Path:
    path = _safe_artifact_path(output_dir, f"run-{run_id}.json", field="record")
    _write_artifact(path, _redact_diagnostic_value(record, output_dir=output_dir))
    return path


def _store_capture(output_dir: Path, *, run_id: str, stage: str, capture: Mapping[str, Any]) -> dict[str, Any]:
    name = f"{stage.replace('_', '-')}-snapshot-{run_id}.json"
    path = _safe_artifact_path(output_dir, name, field=f"{stage} snapshot")
    raw = _write_artifact(path, capture)
    return {
        "path": name,
        "schema": ISSUE146_SNAPSHOT_SCHEMA,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "selected_device": {"board_id": BOARD_ID, "bus_id": BUS_ID},
    }


def _snapshot_config_record(configured: Mapping[str, Any]) -> dict[str, Any]:
    return {"limits": dict(configured), "configuration_words": {"BOARD_POWER_LIMIT": "0x12c", "ENABLED_MAX_ARB": "0x1d1"}}


def _prepare_run(
    *, plan: Mapping[str, Any], configuration: Issue146Configuration, mode: str,
    output_dir: Path, run_id: str, options: Issue146RunOptions,
) -> _PreparedRun:
    if mode not in {"smoke", "long"}:
        raise Issue146Error("Issue #146 mode must be smoke or long")
    timing = run_timing(plan, mode)
    try:
        safe_id = validate_run_id(run_id)
    except ValueError as exc:
        raise Issue146Error(str(exc)) from exc
    board_id, bus_id = _selected_ids(plan, options)
    if mode == "long":
        if options.run_number is None or not 1 <= options.run_number <= 6:
            raise Issue146Error("long runs require run_number from 1 through 6")
        if long_run_order(plan)[options.run_number - 1] != configuration.identifier:
            raise Issue146Error("run_number does not match the preregistered Issue #146 order")
        if not options.approve_long_run:
            raise Issue146Error("long Issue #146 runs require per-instance owner approval")
    elif configuration.identifier not in set(plan["measurement_scope"]["smoke_runs"]["configurations"]):
        raise Issue146Error("configuration is not in the preregistered Issue #146 smoke set")
    output = _safe_output_dir(output_dir)
    trace = _safe_trace_or_groups_path(options.trace_path, output_dir=output, field="Issue #146 trace")
    groups = _safe_trace_or_groups_path(options.groups_path, output_dir=output, field="Issue #146 groups")
    if groups is None:
        groups = output / f"group-timings-{safe_id}.json"
    wrapper_stdout = _safe_artifact_path(
        output, f"wrapper-stdout-{safe_id}.txt", field="Issue #146 wrapper stdout"
    )
    wrapper_stderr = _safe_artifact_path(
        output, f"wrapper-stderr-{safe_id}.txt", field="Issue #146 wrapper stderr"
    )
    _reject_output_collisions(
        output,
        run_id=safe_id,
        trace_path=trace,
        groups_path=groups,
        wrapper_stdout_path=wrapper_stdout,
        wrapper_stderr_path=wrapper_stderr,
    )
    command = build_run_command(
        plan=plan, configuration=configuration, mode=mode, output_dir=output,
        run_id=safe_id, run_number=options.run_number, approve_long_run=options.approve_long_run,
    )
    record = _base_record(
        plan=plan, configuration=configuration, mode=mode, run_id=safe_id,
        run_number=options.run_number, command=command, output_dir=output,
    )
    if options.environment is not None:
        record["environment"] = build_environment_record(
            options.environment, configuration, plan=plan
        )
    return _PreparedRun(
        safe_id,
        timing,
        board_id,
        bus_id,
        output,
        trace,
        groups,
        wrapper_stdout,
        wrapper_stderr,
        command,
        record,
    )


def _apply_cooldown_gate(plan: Mapping[str, Any], prepared: _PreparedRun, options: Issue146RunOptions) -> dict[str, Any] | None:
    if prepared.record["run"]["mode"] != "long" or (options.run_number or 0) <= 1:
        prepared.record["cooldown_gate"] = {"required": False, "verified": True, "gate_result": "not_applicable", "aiclk_gate": False}
        return None
    if options.preceding_pre_snapshot is None:
        raise Issue146Error("preceding selected-device pre-run baseline is required")
    baseline = _snapshot_baseline(options.preceding_pre_snapshot, board_id=prepared.board_id, bus_id=prepared.bus_id)
    cap, power_tolerance, temperature_tolerance = _cooldown_settings(plan)
    if not options.cooldown_readings and options.capture_fn is not None:
        gate = wait_for_cooldown(
            options.preceding_pre_snapshot,
            capture_fn=options.capture_fn,
            board_id=prepared.board_id,
            bus_id=prepared.bus_id,
            cap_seconds=cap,
            power_tolerance_w=power_tolerance,
            temperature_tolerance_c=temperature_tolerance,
        )
    else:
        gate = cooldown_gate(
            baseline,
            readings=options.cooldown_readings,
            elapsed_seconds=options.cooldown_elapsed_seconds,
            cap_seconds=cap,
            power_tolerance_w=power_tolerance,
            temperature_tolerance_c=temperature_tolerance,
        )
    prepared.record["cooldown_gate"] = {"required": True, **gate}
    if gate["verified"]:
        return None
    failed = _failure_record(prepared.record, kind="cooldown_gate_failure", message="cooldown gate failed closed; workload was not started")
    _persist_record(prepared.output_dir, prepared.safe_run_id, failed)
    return failed


def _pre_snapshot_configuration_mismatch(error: BaseException) -> bool:
    message = str(error)
    return (
        "BOARD_POWER_LIMIT does not match" in message
        or "ENABLED_MAX_ARB does not match" in message
        or "limits.board_power_limit does not match" in message
        or "limits.asic_fmax does not match" in message
        or "limits.therm_trip_l1_limit does not match" in message
    )


def _capture_pre_run(prepared: _PreparedRun, state: _ExecutionState, capture_fn: Callable[..., dict[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        capture = capture_fn(stage="pre_run")
        validate_capture(capture, stage="pre_run", board_id=prepared.board_id, bus_id=prepared.bus_id)
        # Identity-valid bytes are evidence even when the configuration gate rejects them.
        descriptor = _store_capture(prepared.output_dir, run_id=prepared.safe_run_id, stage="pre_run", capture=capture)
        prepared.record["pre_run_snapshot"] = descriptor
        state.captures.append(dict(capture))
        try:
            configured = validate_pre_snapshot_configuration(
                capture, plan=plan, board_id=prepared.board_id, bus_id=prepared.bus_id
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            if _pre_snapshot_configuration_mismatch(exc):
                descriptor["configuration"] = {
                    "status": "not_evaluable",
                    "classification": "not_evaluable",
                    "reason": "configuration",
                    "diagnostic": str(exc),
                }
                failed = _failure_record(
                    prepared.record,
                    kind="configuration_mismatch",
                    message="pre-run configuration mismatch; workload was not started",
                )
                failed["status"] = "not_evaluable"
                failed["classification"] = "not_evaluable"
                failed["reason"] = "configuration"
                _persist_record(prepared.output_dir, prepared.safe_run_id, failed)
                return failed
            descriptor["configuration"] = {
                "status": "failed",
                "classification": "failed",
                "reason": "telemetry",
                "diagnostic": str(exc),
            }
            failed = _failure_record(
                prepared.record,
                kind="pre_run_snapshot_failure",
                message=str(exc),
            )
            failed["reason"] = "telemetry"
            _persist_record(prepared.output_dir, prepared.safe_run_id, failed)
            return failed
        descriptor["configuration"] = _snapshot_config_record(configured)
        plan_limits = dict(plan.get("limits", {}))
        plan_limits.update(configured)
        prepared.record["limits"] = plan_limits
    except (OSError, TypeError, ValueError, KeyError) as exc:
        failed = _failure_record(prepared.record, kind="pre_run_snapshot_failure", message=str(exc))
        _persist_record(prepared.output_dir, prepared.safe_run_id, failed)
        return failed
    return None


def _start_trace_sampler(state: _ExecutionState, options: Issue146RunOptions, capture_fn: Callable[..., dict[str, Any]]) -> float:
    interval = float(options.trace_interval_seconds)
    if not math.isfinite(interval) or interval <= 0.0:
        raise Issue146Error("trace interval must be positive")
    state.trace_thread = threading.Thread(
        target=issue137.sample_issue137_trace,
        args=(state.trace_stop, state.captures),
        kwargs={"capture_fn": capture_fn, "interval_seconds": interval, "publication_state": state.publication_state},
        name="issue146-power-trace", daemon=True,
    )
    state.trace_thread.start()
    return interval


def _run_command(
    prepared: _PreparedRun,
    state: _ExecutionState,
    options: Issue146RunOptions,
    interval: float,
) -> set[str]:
    if options.command_runner is None:
        raise Issue146Error("Issue #146 command runner is required for execution")
    before = issue137._artifact_names(prepared.output_dir)
    try:
        state.completed = options.command_runner(
            prepared.command,
            capture_output=True,
            text=True,
            timeout=prepared.timing["guard_seconds"],
            check=False,
            env=_execution_environment(
                options.environment,
                guard_seconds=prepared.timing["guard_seconds"],
                requested_run_id=prepared.safe_run_id,
            ),
        )
        state.stdout = getattr(state.completed, "stdout", None)
        state.stderr = getattr(state.completed, "stderr", None)
        code = getattr(state.completed, "returncode", None)
        if isinstance(code, int) and not isinstance(code, bool):
            state.returncode = code
            if code != 0:
                state.command_error = f"wrapper exited with return code {code}"
        else:
            state.command_error = "wrapper return code is missing"
    except subprocess.TimeoutExpired as exc:
        state.completed = SimpleNamespace(
            returncode=None,
            stdout=getattr(exc, "stdout", None) or getattr(exc, "output", None),
            stderr=getattr(exc, "stderr", None),
        )
        state.stdout = state.completed.stdout
        state.stderr = state.completed.stderr
        state.timed_out = True
        state.command_exception_type = type(exc).__name__
        state.command_error = _redact_diagnostic_text(
            f"{type(exc).__name__}: {exc}", output_dir=prepared.output_dir
        )
    except (OSError, subprocess.SubprocessError) as exc:
        state.completed = SimpleNamespace(
            returncode=None,
            stdout=getattr(exc, "stdout", None) or getattr(exc, "output", None),
            stderr=getattr(exc, "stderr", None),
        )
        state.stdout = state.completed.stdout
        state.stderr = state.completed.stderr
        state.command_exception_type = type(exc).__name__
        state.command_error = _redact_diagnostic_text(
            f"{type(exc).__name__}: {exc}", output_dir=prepared.output_dir
        )
    finally:
        state.publication_state.revoke()
        state.trace_stop.set()
        if state.trace_thread is not None:
            state.trace_thread.join(timeout=max(interval + 1.0, 6.0))
            state.captures.extend(state.publication_state.drain_late_captures())
            if state.trace_thread.is_alive():
                state.cleanup_failure = "power sampler thread did not stop within the bounded cleanup interval"
        try:
            state.wrapper_diagnostics = _persist_wrapper_diagnostics(prepared, state)
        except (OSError, TypeError, ValueError) as exc:
            state.diagnostics_error = _redact_diagnostic_text(
                f"{type(exc).__name__}: {exc}", output_dir=prepared.output_dir
            )
    return before


def _execution_environment(
    environment: Mapping[str, Any] | None, *, guard_seconds: float, requested_run_id: str
) -> dict[str, str]:
    values = dict(os.environ)
    values["HEKATUS_TT_RUNNER"] = ISSUE146_DEVICE_RUNNER
    values["HEKATUS_TT_DEVICE_NODE"] = DEVICE_NODE
    values["HEKATUS_TT_REQUESTED_RUN_ID"] = requested_run_id
    values["HEKATUS_TT_CONTAINER_DEVICE_ID"] = str(CONTAINER_VISIBLE_DEVICE_ID)
    values["HEKATUS_TT_CONTAINER_TIMEOUT_S"] = str(int(guard_seconds))
    if isinstance(environment, Mapping):
        sampler = environment.get("telemetry_sampler")
        if isinstance(sampler, Mapping):
            mode = sampler.get("mode")
            if isinstance(mode, str):
                values["HEKATUS_TT_TELEMETRY_MODE"] = mode
            if sampler.get("interval_seconds") is not None and mode == "explicit":
                values["HEKATUS_TT_TELEMETRY_INTERVAL_S"] = str(sampler["interval_seconds"])
        digest = _environment_digest(environment)
        if digest:
            values["HEKATUS_TT_IMAGE"] = "ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-24.04-release-amd64@" + digest
    return values


def _capture_post_run(prepared: _PreparedRun, state: _ExecutionState, capture_fn: Callable[..., dict[str, Any]]) -> None:
    try:
        capture = capture_fn(stage="post_run")
        validate_capture(capture, stage="post_run", board_id=prepared.board_id, bus_id=prepared.bus_id)
        state.captures.append(dict(capture))
        prepared.record["post_run_snapshot"] = _store_capture(prepared.output_dir, run_id=prepared.safe_run_id, stage="post_run", capture=capture)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        prepared.record["post_run_snapshot"] = {"status": "unavailable", "reason": str(exc)}
        state.cleanup_failure = state.cleanup_failure or f"post-run snapshot failure: {type(exc).__name__}"


def _extract_result_name(stdout: Any) -> str | None:
    if isinstance(stdout, bytes):
        text = stdout.decode("utf-8", "replace")
    elif isinstance(stdout, str):
        text = stdout
    else:
        text = ""
    matches = _RESULT_MARKER_RE.findall(text)
    if not matches:
        return None
    try:
        return _safe_basename(Path(matches[-1]).name, field="runner result name")
    except Issue146Error:
        return None


def _resolve_result(output_dir: Path, *, before: set[str], completed: Any) -> tuple[Path, str]:
    marker = _extract_result_name(getattr(completed, "stdout", None))
    after = issue137._artifact_names(output_dir)
    new = sorted(name for name in after - before if name.startswith("runner-result") and name.endswith(".json"))
    candidates = ([marker] if marker else []) + [name for name in new if name != marker]
    if len(candidates) != 1 or candidates[0] not in new:
        raise Issue146Error("runner result artifact identity is missing or ambiguous")
    path = _safe_artifact_path(output_dir, candidates[0], field="runner result")
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise Issue146Error("runner result artifact identity is unsafe")
    return path, candidates[0]


def _read_json(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue146Error(f"{field} is malformed") from exc
    if not isinstance(payload, dict):
        raise Issue146Error(f"{field} must be an object")
    return payload, raw


def _validate_runner_identity(
    result: Mapping[str, Any], *, mode: str, requested_run_id: str | None,
) -> None:
    if result.get("record_schema") != ISSUE146_DEVICE_RESULT_SCHEMA:
        raise Issue146Error("runner result schema is unsupported")
    if result.get("status") != "ok":
        raise Issue146Error("runner result did not report successful execution")
    wrapper_run_id = result.get("wrapper_run_id", result.get("run_id"))
    if not isinstance(wrapper_run_id, str):
        raise Issue146Error("runner result wrapper run identity is missing")
    try:
        validate_run_id(wrapper_run_id)
    except ValueError as exc:
        raise Issue146Error("runner result wrapper run identity is unsafe") from exc
    if result.get("run_id") != wrapper_run_id:
        raise Issue146Error("runner result run_id does not match wrapper_run_id")
    semantic_run_id = result.get("requested_run_id")
    if not isinstance(semantic_run_id, str):
        raise Issue146Error("runner result requested run identity is missing")
    try:
        validate_run_id(semantic_run_id)
    except ValueError as exc:
        raise Issue146Error("runner result requested run identity is unsafe") from exc
    if requested_run_id is not None and semantic_run_id != requested_run_id:
        raise Issue146Error("runner result requested run identity does not match the requested run")
    if result.get("mode") != mode or result.get("load_mode") != ISSUE146_LOAD_MODE:
        raise Issue146Error("runner result mode binding is invalid")


def _validate_runner_metadata(
    result: Mapping[str, Any], *, plan: Mapping[str, Any], configuration: Issue146Configuration,
    mode: str,
) -> dict[str, Any]:
    timing = run_timing(plan, mode)
    outer_repetitions, launches_per_sync = grouped_load_settings(plan)
    expected_values = (
        ("target_seconds", timing["load_seconds"]),
        ("guard_seconds", timing["guard_seconds"]),
        ("overrun_limit_seconds", DEFAULT_OVERRUN_LIMIT_SECONDS),
        ("outer_repetitions", outer_repetitions),
        ("launches_per_sync", launches_per_sync),
    )
    for field, expected in expected_values:
        if float(result.get(field)) != expected:
            raise Issue146Error(f"runner result {field} does not match the plan")
    if result.get("plan_sha256") != _plan_hash(plan):
        raise Issue146Error("runner result plan hash does not match the owner plan")
    if result.get("core_plan") != configuration.core_plan.as_record():
        raise Issue146Error("runner result core plan does not match the selected configuration")
    environment = result.get("environment")
    if not isinstance(environment, Mapping):
        raise Issue146Error("runner result environment provenance is missing")
    validate_environment_for_run(environment, configuration=configuration, plan=plan)
    return timing


def _parse_runner_groups(result: Mapping[str, Any]) -> list[dict[str, int]]:
    raw_groups = result.get("group_timings")
    if not isinstance(raw_groups, list):
        raise Issue146Error("runner result group_timings are missing")
    groups: list[dict[str, int]] = []
    for index, group in enumerate(raw_groups):
        if not isinstance(group, Mapping):
            raise Issue146Error(f"runner result group {index} is invalid")
        start, end = group.get("start_monotonic_ns"), group.get("end_monotonic_ns")
        if isinstance(start, bool) or not isinstance(start, int) or isinstance(end, bool) or not isinstance(end, int):
            raise Issue146Error(f"runner result group {index} bounds are invalid")
        groups.append({"start_monotonic_ns": start, "end_monotonic_ns": end})
    return groups


def _runner_group_facts(
    result: Mapping[str, Any], groups: Sequence[Mapping[str, int]], timing: Mapping[str, Any],
) -> dict[str, Any]:
    first = result.get("first_launch_start_monotonic_ns")
    if isinstance(first, bool) or not isinstance(first, int):
        raise Issue146Error("runner result first launch clock is invalid")
    from tools.issue146_analysis import group_timing_facts

    facts = group_timing_facts(
        groups, first_launch_start_monotonic_ns=first,
        plateau_start_seconds=60.0, plateau_end_seconds=180.0,
        workload_duration_seconds=timing["load_seconds"],
        idle_fraction_limit=0.02, overrun_limit_seconds=10.0,
    )
    if not facts["coverage_pass"] or not facts["overrun_pass"]:
        raise Issue146Error("runner result group coverage or overrun failed")
    observed = (facts["last_group_end_monotonic_ns"] - first) / 1_000_000_000.0
    supplied = result.get("observed_duration_seconds")
    if supplied is None or not math.isclose(float(supplied), observed, rel_tol=0.0, abs_tol=1e-9):
        raise Issue146Error("runner result observed duration does not match group bounds")
    launch_count = result.get("launch_count")
    if isinstance(launch_count, bool) or not isinstance(launch_count, int) or launch_count < 1:
        raise Issue146Error("runner result launch_count is invalid")
    return {
        "first_launch_start_monotonic_ns": first,
        "workload_end_monotonic_ns": facts["workload_end_monotonic_ns"],
        "last_group_end_monotonic_ns": facts["last_group_end_monotonic_ns"],
        "observed_duration_seconds": observed,
        "group_count": len(groups),
        "launch_count": launch_count,
        "overrun_seconds": facts["overrun_seconds"],
        "idle_fraction": facts["idle_fraction"],
    }


def _validate_runner_result(
    result: Mapping[str, Any], *, plan: Mapping[str, Any], configuration: Issue146Configuration,
    mode: str, requested_run_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    _validate_runner_identity(result, mode=mode, requested_run_id=requested_run_id)
    timing = _validate_runner_metadata(result, plan=plan, configuration=configuration, mode=mode)
    groups = _parse_runner_groups(result)
    return _runner_group_facts(result, groups, timing), groups


def validate_runner_result(
    result: Mapping[str, Any], *, plan: Mapping[str, Any],
    configuration: Issue146Configuration, mode: str, requested_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate one returned device result and recompute its group facts."""
    facts, _groups = _validate_runner_result(
        result, plan=plan, configuration=configuration, mode=mode,
        requested_run_id=requested_run_id,
    )
    return facts


def _result_artifact_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise Issue146Error(f"{field} is missing")
    value = value.removeprefix("/out/")
    return _safe_basename(value, field=field)


def _validate_device_group_artifact(
    output_dir: Path, *, before: set[str], result: Mapping[str, Any],
    groups: Sequence[Mapping[str, int]], wrapper_run_id: str,
) -> dict[str, Any]:
    name = _result_artifact_name(result.get("groups_path"), field="runner group artifact")
    expected_name = f"device-group-timings-{wrapper_run_id}.json"
    if name != expected_name:
        raise Issue146Error("runner group artifact is not bound to the wrapper invocation")
    if name in before:
        raise Issue146Error("runner group artifact is stale")
    path = _safe_artifact_path(output_dir, name, field="runner group artifact")
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise Issue146Error("runner group artifact is missing or unsafe")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue146Error("runner group artifact is malformed") from exc
    if payload != [dict(group) for group in groups]:
        raise Issue146Error("runner group artifact does not equal runner timings")
    digest = result.get("groups_sha256")
    byte_count = result.get("groups_byte_count")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise Issue146Error("runner group artifact hash is missing")
    if digest != hashlib.sha256(raw).hexdigest():
        raise Issue146Error("runner group artifact hash does not match its bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count != len(raw):
        raise Issue146Error("runner group artifact byte count does not match its bytes")
    return {
        "path": name,
        "schema": "issue146-device-group-array-v1",
        "sha256": digest,
        "byte_count": byte_count,
        "group_count": len(payload),
        "source": "device-produced raw group array",
    }


def _validate_companion_power_trace(
    output_dir: Path, *, before: set[str], result: Mapping[str, Any],
    wrapper_run_id: str, mode: str, environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sampler = environment.get("telemetry_sampler") if isinstance(environment, Mapping) else None
    if mode == "smoke" and isinstance(sampler, Mapping) and sampler.get("mode") == "off":
        return {"status": "not_applicable", "reason": "diagnostic sampler is off"}
    name = _result_artifact_name(
        result.get("power_trace_path", f"power-{wrapper_run_id}.csv"),
        field="legacy companion power trace",
    )
    expected_name = f"power-{wrapper_run_id}.csv"
    if name != expected_name:
        raise Issue146Error("legacy companion power trace is not bound to the wrapper invocation")
    if name in before:
        raise Issue146Error("legacy companion power trace is stale")
    path = _safe_artifact_path(output_dir, name, field="legacy companion power trace")
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise Issue146Error("legacy companion power trace is missing or unsafe")
    try:
        raw = path.read_bytes()
        parsed = telemetry.parse_power_trace(path)
    except (OSError, TypeError, ValueError) as exc:
        raise Issue146Error("legacy companion power trace is not valid") from exc
    if (
        parsed.get("readable") is not True
        or parsed.get("nonempty") is not True
        or parsed.get("csv_row_count", 0) < 1
        or parsed.get("valid_row_count") != parsed.get("csv_row_count")
        or parsed.get("columns") != list(telemetry.POWER_TRACE_COLUMNS)
    ):
        raise Issue146Error("legacy companion power trace does not satisfy its schema")
    return {
        "status": "ok",
        "path": name,
        "schema": "legacy-power-trace-v1",
        "columns": list(telemetry.POWER_TRACE_COLUMNS),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "csv_row_count": parsed.get("csv_row_count"),
        "valid_row_count": parsed.get("valid_row_count"),
        "coverage": parsed.get("coverage"),
        "source": "wrapper legacy power sampler companion evidence",
    }


def _write_groups(
    prepared: _PreparedRun, facts: Mapping[str, Any], groups: Sequence[Mapping[str, int]],
    device_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    envelope_name = f"{prepared.groups_path.stem}-envelope.json"
    raw_payload = [dict(group) for group in groups]
    envelope = {
        "schema": ISSUE146_GROUPS_SCHEMA,
        "schema_version": 1,
        "first_launch_start_monotonic_ns": facts["first_launch_start_monotonic_ns"],
        "groups": raw_payload,
    }
    envelope_path = _safe_artifact_path(prepared.output_dir, envelope_name, field="groups envelope")
    envelope_bytes = _write_artifact(envelope_path, envelope)
    return {
        "path": envelope_name,
        "schema": ISSUE146_GROUPS_SCHEMA,
        "schema_version": 1,
        "first_launch_start_monotonic_ns": facts["first_launch_start_monotonic_ns"],
        "sha256": hashlib.sha256(envelope_bytes).hexdigest(),
        "byte_count": len(envelope_bytes),
        "raw_path": device_artifact["path"],
        "raw_sha256": device_artifact["sha256"],
        "raw_byte_count": device_artifact["byte_count"],
        "device_artifact": dict(device_artifact),
        "group_count": len(raw_payload),
        "last_group_end_monotonic_ns": facts["last_group_end_monotonic_ns"],
        "overrun_seconds": facts["overrun_seconds"],
        "host_visible_idle_fraction": facts["idle_fraction"],
    }


def _trace_rows_from_captures(captures: Sequence[Mapping[str, Any]], *, board_id: str, bus_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    saw_in_run = False
    for index, capture in enumerate(captures):
        if not isinstance(capture, Mapping):
            raise Issue146Error(f"capture {index} is not an object")
        stage = capture.get("stage")
        if stage not in {"pre_run", "in_run", "post_run"}:
            continue
        validate_capture(capture, stage=str(stage), board_id=board_id, bus_id=bus_id)
        if stage == "in_run":
            saw_in_run = True
        try:
            rows.append(
                power_trace_issue146.parse_issue146_snapshot(
                    capture["snapshot"], board_id=board_id, bus_id=bus_id,
                    timestamp_utc=capture.get("captured_at_utc"),
                    timestamp_monotonic_ns=capture.get("timestamp_monotonic_ns"),
                    snapshot_elapsed_ns=capture.get("snapshot_elapsed_ns"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Issue146Error(f"capture {index} could not produce a complete raw trace row") from exc
    if not saw_in_run or not rows:
        raise Issue146Error("Issue #146 trace has no in-run telemetry capture")
    timestamps = [row["timestamp_monotonic_ns"] for row in rows]
    if any(right < left for left, right in itertools.pairwise(timestamps)):
        raise Issue146Error("Issue #146 trace capture timestamps are not monotonic")
    return rows


def trace_rows_from_captures(
    captures: Sequence[Mapping[str, Any]], *, board_id: str = BOARD_ID, bus_id: str = BUS_ID
) -> list[dict[str, Any]]:
    """Parse every retained capture without dropping malformed telemetry."""
    return _trace_rows_from_captures(captures, board_id=board_id, bus_id=bus_id)


def write_trace_from_captures(
    path: Path, captures: Sequence[Mapping[str, Any]], *,
    board_id: str = BOARD_ID, bus_id: str = BUS_ID,
) -> Path:
    """Persist the exact raw Issue #146 rows from every retained capture."""
    destination = _trusted_path(path, field="Issue #146 trace destination")
    if destination.exists() or destination.is_symlink():
        raise Issue146Error("Issue #146 trace destination collides with an existing artifact")
    rows = trace_rows_from_captures(captures, board_id=board_id, bus_id=bus_id)
    return power_trace_issue146.write_issue146_power_trace(destination, rows)


def sample_issue146_trace(*args: Any, **kwargs: Any) -> None:
    """Run the existing bounded capture loop for the Issue #146 trace seam."""
    issue137.sample_issue137_trace(*args, **kwargs)


def _source_versions_verified(environment: Mapping[str, Any] | None, captures: Sequence[Mapping[str, Any]]) -> bool:
    candidates: list[Any] = []
    if isinstance(environment, Mapping):
        candidates.append(environment.get("host_sw_vers"))
    for capture in captures:
        snapshot = capture.get("snapshot") if isinstance(capture, Mapping) else None
        if isinstance(snapshot, Mapping):
            candidates.append(snapshot.get("host_sw_vers"))
    observed = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    return bool(observed) and all(power_trace_issue146.source_versions_verified(item) for item in observed)


def _write_trace(prepared: _PreparedRun, state: _ExecutionState, *, facts: Mapping[str, Any], source_verified: bool) -> dict[str, Any]:
    if prepared.trace_path is None:
        raise Issue146Error("Issue #146 trace destination is required for execution")
    rows = _trace_rows_from_captures(state.captures, board_id=prepared.board_id, bus_id=prepared.bus_id)
    target_end = facts["workload_end_monotonic_ns"]
    timestamps = [int(row["timestamp_monotonic_ns"]) for row in rows]
    if min(timestamps) > facts["first_launch_start_monotonic_ns"] or max(timestamps) < target_end:
        raise Issue146Error("Issue #146 trace lacks terminal workload coverage")
    power_trace_issue146.write_issue146_power_trace(prepared.trace_path, rows)
    raw = prepared.trace_path.read_bytes()
    parsed = power_trace_issue146.read_issue146_power_trace(
        prepared.trace_path, run_start_monotonic_ns=facts["first_launch_start_monotonic_ns"],
        run_end_monotonic_ns=target_end, source_established=source_verified,
    )
    if parsed.get("valid") is not True:
        raise Issue146Error("Issue #146 persisted trace is not valid")
    return {
        "path": prepared.trace_path.name,
        "schema": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA,
        "schema_version": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(power_trace_issue146.ISSUE146_POWER_TRACE_COLUMNS),
        "sample_count": len(rows),
        "run_start_monotonic_ns": facts["first_launch_start_monotonic_ns"],
        "run_end_monotonic_ns": target_end,
        "actual_end_monotonic_ns": facts["last_group_end_monotonic_ns"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "unit_provenance": power_trace_issue146.ISSUE146_UNIT_PROVENANCE if source_verified else None,
    }


def _ingest_result(prepared: _PreparedRun, state: _ExecutionState, *, before: set[str], plan: Mapping[str, Any], configuration: Issue146Configuration) -> tuple[dict[str, Any], list[dict[str, int]], dict[str, Any]]:
    path, name = _resolve_result(prepared.output_dir, before=before, completed=state.completed)
    result, raw = _read_json(path, field="Issue #146 runner result")
    state.result = result
    prepared.record["runner_result"] = _redact_diagnostic_value(
        result, output_dir=prepared.output_dir
    )
    descriptor = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw), "schema": result.get("record_schema")}
    prepared.record["provenance"]["device_result"] = descriptor
    facts, groups = _validate_runner_result(
        result, plan=plan, configuration=configuration,
        mode=prepared.record["run"]["mode"], requested_run_id=prepared.safe_run_id,
    )
    wrapper_run_id = result["wrapper_run_id"]
    expected_result_name = f"runner-result-{wrapper_run_id}.json"
    if name != expected_result_name:
        raise Issue146Error("runner result artifact is not bound to the wrapper invocation")
    device_artifact = _validate_device_group_artifact(
        prepared.output_dir, before=before, result=result, groups=groups,
        wrapper_run_id=wrapper_run_id,
    )
    companion = _validate_companion_power_trace(
        prepared.output_dir, before=before, result=result, wrapper_run_id=wrapper_run_id,
        mode=prepared.record["run"]["mode"], environment=result.get("environment"),
    )
    state.device_groups = device_artifact
    state.companion_power_trace = companion
    prepared.record["provenance"]["wrapper_run_id"] = wrapper_run_id
    prepared.record["provenance"]["requested_run_id"] = result["requested_run_id"]
    prepared.record["provenance"]["legacy_power_trace"] = companion
    prepared.record["environment"] = build_environment_record(result["environment"], configuration, plan=plan)
    prepared.record["run"].update(facts)
    prepared.record["run"]["first_launch_start_monotonic_ns"] = facts["first_launch_start_monotonic_ns"]
    prepared.record["run"]["workload_started"] = True
    prepared.record["workload_started"] = True
    return facts, groups, descriptor


def _finish(
    prepared: _PreparedRun,
    state: _ExecutionState,
    *,
    failure: str | None = None,
) -> dict[str, Any]:
    if state.wrapper_diagnostics is not None:
        prepared.record["wrapper_diagnostics"] = state.wrapper_diagnostics
    if failure or state.cleanup_failure or state.command_error or state.diagnostics_error or state.returncode not in {None, 0} or state.timed_out:
        message = (
            failure
            or state.command_error
            or state.cleanup_failure
            or state.diagnostics_error
            or ("device runner reached the fixed guard" if state.timed_out else "device runner did not return status zero")
        )
        if failure and state.command_error and state.command_error != failure:
            message = f"{failure}; {state.command_error}"
        message = _redact_diagnostic_text(message, output_dir=prepared.output_dir)
        record = _failure_record(
            prepared.record,
            kind="runner_execution_failure",
            message=message,
            returncode=state.returncode,
        )
        details = record["failure"]
        if state.command_exception_type is not None:
            details["exception_type"] = state.command_exception_type
        if state.command_error is not None and state.command_error != message:
            details["wrapper_error"] = state.command_error
        if state.wrapper_diagnostics is not None:
            details["wrapper_diagnostics"] = state.wrapper_diagnostics
        if state.diagnostics_error is not None:
            details["diagnostics_error"] = state.diagnostics_error
    else:
        record = prepared.record
        record["status"] = "ok"
        record["verdict_evidence"] = record["run"]["mode"] == "long"
    _persist_record(prepared.output_dir, prepared.safe_run_id, record)
    return record


def _execute_run(prepared: _PreparedRun, *, plan: Mapping[str, Any], configuration: Issue146Configuration, options: Issue146RunOptions) -> dict[str, Any]:
    capture_fn = options.capture_fn or capture_readonly_snapshot
    state = _ExecutionState(captures=[], publication_state=issue137._CapturePublicationState(), trace_stop=threading.Event())
    failed = _capture_pre_run(prepared, state, capture_fn, plan)
    if failed is not None:
        return failed
    interval = (
        _start_trace_sampler(state, options, capture_fn)
        if prepared.trace_path is not None
        else float(options.trace_interval_seconds)
    )
    before = _run_command(prepared, state, options, interval)
    _capture_post_run(prepared, state, capture_fn)
    if state.result is None:
        try:
            facts, groups, _device_descriptor = _ingest_result(prepared, state, before=before, plan=plan, configuration=configuration)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            return _finish(prepared, state, failure=str(exc))
    else:
        return _finish(prepared, state, failure=state.result_error or "runner result was already consumed")
    try:
        if state.device_groups is None:
            raise Issue146Error("device group artifact provenance is missing")
        prepared.record["groups"] = _write_groups(prepared, facts, groups, state.device_groups)
        verified = _source_versions_verified(prepared.record.get("environment"), state.captures)
        prepared.record["provenance"]["unit_provenance"] = power_trace_issue146.ISSUE146_UNIT_PROVENANCE if verified else None
        prepared.record["provenance"]["unit_provenance_status"] = "established" if verified else "unknown"
        prepared.record["trace"] = _write_trace(prepared, state, facts=facts, source_verified=verified)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        return _finish(prepared, state, failure=str(exc))
    return _finish(prepared, state)


def run_one(
    *, plan: Mapping[str, Any], configuration: Issue146Configuration, mode: str,
    output_dir: Path, run_id: str, options: Issue146RunOptions | None = None,
) -> dict[str, Any]:
    """Prepare or execute exactly one smoke/long Issue #146 instance."""
    selected = options or Issue146RunOptions()
    prepared = _prepare_run(
        plan=plan, configuration=configuration, mode=mode, output_dir=output_dir,
        run_id=run_id, options=selected,
    )
    failed = _apply_cooldown_gate(plan, prepared, selected)
    if failed is not None:
        return failed
    if selected.command_runner is None:
        return prepared.record
    return _execute_run(prepared, plan=plan, configuration=configuration, options=selected)


def build_planning_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a board-free plan record; no device or command is opened."""
    return {
        "record_schema": "issue146-planning-v1",
        "record_revision": 1,
        "issue": "#146",
        "board_run": False,
        "plan_revision": CURRENT_PLAN_REVISION,
        "plan_sha256": _plan_hash(plan),
        "long_run_order": long_run_order(plan),
        "timing": {"smoke": run_timing(plan, "smoke"), "long": run_timing(plan, "long")},
        "load_execution": _safe_json_value(plan["load_execution"]),
        "configurations": [configuration(plan, item["id"]).as_record() for item in plan_configurations(plan)],
        "trace": {
            "schema": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA,
            "schema_version": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA_VERSION,
            "columns": list(power_trace_issue146.ISSUE146_POWER_TRACE_COLUMNS),
            "unit_provenance": power_trace_issue146.ISSUE146_UNIT_PROVENANCE,
        },
    }


def _parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--mode", choices=("plan", "smoke", "long"), default="plan")
    parser.add_argument("--configuration")
    parser.add_argument("--run-id")
    parser.add_argument("--run-number", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("out/issue146"))
    parser.add_argument("--preceding-pre-snapshot", type=Path, default=None)
    parser.add_argument("--approve-long-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.mode == "plan":
            print(strict_json_dumps(build_planning_record(plan), indent=2))
            return 0
        if args.configuration is None or args.run_id is None:
            raise Issue146Error("measurement modes require --configuration and --run-id")
        selected = configuration(plan, args.configuration)
        preceding = None
        if args.preceding_pre_snapshot is not None:
            preceding = _load_preceding_pre_snapshot(args.preceding_pre_snapshot, plan)
        elif args.mode == "long" and (args.run_number or 0) > 1:
            raise Issue146Error("long runs after the first require --preceding-pre-snapshot")
        output = _safe_output_dir(args.out_dir)
        record = run_one(
            plan=plan, configuration=selected, mode=args.mode, output_dir=output,
            run_id=args.run_id,
            options=Issue146RunOptions(
                run_number=args.run_number, approve_long_run=args.approve_long_run,
                command_runner=issue137._run_issue137_workload if args.execute else None,
                capture_fn=capture_readonly_snapshot if args.execute else None,
                preceding_pre_snapshot=preceding,
                trace_path=output / f"issue146-power-trace-{args.run_id}.csv" if args.execute else None,
            ),
        )
        print(strict_json_dumps(_safe_json_value(record), indent=2))
        return 0 if record.get("status") in {"planned", "ok"} else 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"issue146: {type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 2


__all__ = [
    "BOARD_ID",
    "BOARD_SERIAL",
    "BUS_ID",
    "CONTAINER_VISIBLE_DEVICE_ID",
    "COOLDOWN_CAP_SECONDS",
    "CURRENT_PLAN_REVISION",
    "DEVICE_ID",
    "DEVICE_NODE",
    "HOST_DEVICE_ID",
    "ISSUE146_DEVICE_RESULT_SCHEMA",
    "ISSUE146_GROUPS_SCHEMA",
    "ISSUE146_GROUP_TIMINGS_SCHEMA",
    "ISSUE146_LOAD_MODE",
    "ISSUE146_POWER_TRACE_COLUMNS",
    "ISSUE146_POWER_TRACE_SCHEMA",
    "ISSUE146_RECORD_SCHEMA",
    "ISSUE146_SNAPSHOT_SCHEMA",
    "Issue146Configuration",
    "Issue146Error",
    "Issue146RunOptions",
    "build_environment_record",
    "build_planning_record",
    "build_run_command",
    "capture_readonly_snapshot",
    "configuration",
    "configuration_for_core_count",
    "cooldown_gate",
    "evaluate_cooldown",
    "grouped_load_settings",
    "load_plan",
    "long_run_order",
    "main",
    "plan_configurations",
    "resolve_result_path",
    "run_one",
    "run_timing",
    "sample_issue146_trace",
    "trace_rows_from_captures",
    "validate_capture",
    "validate_environment_for_run",
    "validate_pre_snapshot_configuration",
    "validate_runner_result",
    "validate_selected_identity",
    "wait_for_cooldown",
    "write_trace_from_captures",
]


if __name__ == "__main__":
    raise SystemExit(main())
