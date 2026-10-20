"""Board-free analysis of the preregistered Issue #146 conformance seam.

The analyzer consumes only persisted records.  It never opens a board, starts a
container, invokes a telemetry utility, or turns a smoke diagnostic into long-
run evidence.  The runner-facing record contract is intentionally small:
``environment``, ``configuration``, ``plan_sha256``, ``run`` (including the
first-launch clock and observed duration), ``groups`` JSON, pre/post snapshots,
and an Issue #146 trace with immutable byte provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench import power_trace_issue146
from tools import issue137 as issue137_runner
from tools.issue146_analysis import (
    ARBITER_FIELDS,
    EXPECTED_BUSY_MIN,
    EXPECTED_CONTROL_MAX,
    EXPECTED_ENABLED_MAX,
    EXPECTED_ENABLED_MIN,
    Issue146AnalysisError,
    decode_arbiter_row,
    decode_gddr_row,
    finite,
    gddr_zero_word,
    group_timing_facts,
    integer,
    left_held_mean,
    point_statistics,
    pooled_statistics,
    select_plateau_rows,
)

ANALYSIS_SCHEMA = "issue146-analysis-v1"
PLAN_SCHEMA = "issue146-premeasurement-plan-v1"
RECORD_SCHEMA = "issue146-measurement-v1"
GROUPS_SCHEMA = "issue146-group-timings-v1"
SNAPSHOT_SCHEMA = "issue146-tt-smi-snapshot-v1"
LOAD_MODE = "host-duration-bounded"
DEFAULT_PLAN_PATH = Path("docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json")
AUTHORITATIVE_PLAN_RAW_SHA256 = "84ea24fc4533237f82a60cf7936c7f960f6cbc6449dcf427530eab34bad840b8"
AUTHORITATIVE_PLAN_CANONICAL_SHA256 = "b48e1a8dfdeb8ab3bde3208ae68f85371e52108413d6d14f3c8629ad58509d4e"
ISSUE146_RECORD_SCHEMA = RECORD_SCHEMA
ISSUE146_SNAPSHOT_SCHEMA = SNAPSHOT_SCHEMA
PLATEAU_START_SECONDS = 60.0
PLATEAU_END_SECONDS = 180.0
SMOKE_START_SECONDS = 10.0
SMOKE_END_SECONDS = 40.0
MINIMUM_SAMPLES = 50
IDLE_FRACTION_LIMIT = 0.02
OVERRUN_LIMIT_SECONDS = 10.0
PRECISION_LIMIT_W = 10.0
TRACE_COLUMNS = power_trace_issue146.ISSUE146_POWER_TRACE_COLUMNS
TRACE_SCHEMA = power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA
PLAN_PROVENANCE = power_trace_issue146.ISSUE146_UNIT_PROVENANCE

_MISSING = object()
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PATH_FIELDS = frozenset(
    {"hostname", "host_name", "username", "user_name", "cwd", "working_directory"}
)


class AnalysisError(ValueError):
    """Raised when the Issue #146 analysis contract cannot be represented."""


@dataclass(frozen=True)
class _PlanSpec:
    plan: Mapping[str, Any]
    plan_sha256: str
    order: tuple[str, ...]
    configurations: dict[str, Mapping[str, Any]]
    expected_configurations: dict[str, Mapping[str, Any]]
    required_environment_fields: tuple[str, ...]
    board_serial: str
    board_id: str
    bus_id: str
    device_node: str
    target_seconds: float
    guard_seconds: float
    smoke_target_seconds: float
    smoke_guard_seconds: float
    overrun_limit_seconds: float
    plateau_start_seconds: float
    plateau_end_seconds: float
    minimum_samples: int
    idle_fraction_limit: float
    board_power_limit: float
    asic_fmax: float
    asic_temperature_limit: float
    gddr_temperature_limit: float
    regulated_interval: tuple[float, float]
    outer_repetitions: int
    launches_per_sync: int


@dataclass(frozen=True)
class _RecordInput:
    record: Mapping[str, Any]
    base_dir: Path | None
    name: str


@dataclass
class _LoadedTrace:
    rows: list[dict[str, Any]]
    source_name: str | None
    source_sha256: str | None
    source_byte_count: int | None
    errors: list[str]
    missing: set[str]
    gddr_provenance: bool
    payload: Mapping[str, Any] | None


@dataclass
class _LoadedGroups:
    groups: list[dict[str, Any]]
    source_name: str | None
    source_sha256: str | None
    source_byte_count: int | None
    errors: list[str]
    missing: set[str]
    payload: Mapping[str, Any] | None


@dataclass
class _LoadedSnapshot:
    snapshot: Mapping[str, Any] | None
    device: Mapping[str, Any] | None
    source_name: str | None
    source_sha256: str | None
    source_byte_count: int | None
    capture_timestamp: int | None
    errors: list[str]


def _safe_public(value: Any) -> Any:
    """Copy JSON values while removing host identity and private paths."""
    if isinstance(value, Mapping):
        return {
            str(key): _safe_public(child)
            for key, child in value.items()
            if str(key).casefold() not in _PATH_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_public(child) for child in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        if normalized.startswith(("/home/", "/Users/", "/root/", "/tmp/")):
            return "<redacted-path>"
        return value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _trusted_roots() -> tuple[Path, ...]:
    return tuple(
        root.resolve(strict=False) for root in (_REPOSITORY_ROOT, Path("/tmp"), Path("/out"))
    )


def _safe_path(value: str | Path, *, purpose: str, output: bool = False) -> Path:
    try:
        candidate = Path(value)
        if output and candidate.is_symlink():
            raise AnalysisError(f"{purpose} must not be a symbolic link")
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, TypeError) as exc:
        raise AnalysisError(f"{purpose} path cannot be resolved safely") from exc
    if not any(resolved.is_relative_to(root) for root in _trusted_roots()):
        raise AnalysisError(f"{purpose} path must be beneath a trusted project directory")
    return resolved


def _safe_artifact_path(value: Any, *, base_dir: Path | None, purpose: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise AnalysisError(f"{purpose} path must be a file name")
    if base_dir is None:
        raise AnalysisError(f"{purpose} has no trusted record directory")
    root = _safe_path(base_dir, purpose=f"{purpose} directory")
    path = root / value
    checked = _safe_path(path, purpose=purpose)
    if checked.parent != root:
        raise AnalysisError(f"{purpose} escapes its trusted record directory")
    return checked


def _read_json(path: Path, *, purpose: str) -> tuple[Mapping[str, Any], bytes]:
    checked = _safe_path(path, purpose=purpose)
    try:
        raw = checked.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{purpose} cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise AnalysisError(f"{purpose} must be an object")
    return payload, raw


def _write_text(path: Path, text: str, *, purpose: str) -> None:
    checked = _safe_path(path, purpose=purpose, output=True)
    checked.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(checked, flags, 0o666)
    except OSError as exc:
        raise AnalysisError(f"{purpose} cannot be written") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(text)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def plan_sha256(plan: Mapping[str, Any]) -> str:
    """Return the immutable SHA-256 used by the runner/analyzer binding."""
    try:
        encoded = strict_json_dumps(plan, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Issue #146 plan cannot be hashed") from exc
    return hashlib.sha256(encoded).hexdigest()


def _verify_authoritative_plan_file() -> Mapping[str, Any]:
    """Verify the checked-in owner plan before accepting any caller mapping."""
    path = _REPOSITORY_ROOT / DEFAULT_PLAN_PATH
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("authoritative Issue #146 plan cannot be read") from exc
    if hashlib.sha256(raw).hexdigest() != AUTHORITATIVE_PLAN_RAW_SHA256:
        raise AnalysisError("authoritative Issue #146 plan bytes have drifted")
    if not isinstance(payload, Mapping) or plan_sha256(payload) != AUTHORITATIVE_PLAN_CANONICAL_SHA256:
        raise AnalysisError("authoritative Issue #146 plan canonical digest has drifted")
    return payload


def _canonical_source_bytes(payload: Mapping[str, Any], *, hash_fields: set[str]) -> bytes:
    normalized = {key: value for key, value in payload.items() if key not in hash_fields}
    return strict_json_dumps(_safe_public(normalized), sort_keys=True).encode("utf-8")


def canonical_trace_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical bytes for inline trace fixtures and record-side provenance."""
    return _canonical_source_bytes(
        payload, hash_fields={"sha256", "byte_count", "trace_sha256", "trace_byte_count", "path"}
    )


def canonical_groups_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonical bytes for inline group-timing fixtures."""
    return _canonical_source_bytes(
        payload, hash_fields={"sha256", "byte_count", "groups_sha256", "groups_byte_count", "path"}
    )


def _plan_spec(plan: Mapping[str, Any]) -> _PlanSpec:
    _verify_authoritative_plan_file()
    if plan_sha256(plan) != AUTHORITATIVE_PLAN_CANONICAL_SHA256:
        raise AnalysisError("supplied Issue #146 plan is not the immutable revision-3 plan")
    if plan.get("record_schema") != PLAN_SCHEMA:
        raise AnalysisError("unsupported Issue #146 plan schema")
    if plan.get("status") != "planned" or plan.get("board_run") is not False:
        raise AnalysisError("Issue #146 plan is not the immutable board-free plan")
    history = plan.get("revision_history")
    if (
        not isinstance(history, list)
        or not history
        or not isinstance(history[-1], Mapping)
        or history[-1].get("revision") != 3
    ):
        raise AnalysisError("Issue #146 plan revision 3 is required")
    scope = plan.get("measurement_scope")
    if not isinstance(scope, Mapping):
        raise AnalysisError("Issue #146 measurement_scope is missing")
    raw_order = scope.get("order")
    if not isinstance(raw_order, list) or not all(isinstance(item, str) for item in raw_order):
        raise AnalysisError("Issue #146 run order is missing")
    configurations_raw = plan.get("configurations")
    if not isinstance(configurations_raw, list) or not configurations_raw:
        raise AnalysisError("Issue #146 configurations are missing")
    configurations: dict[str, Mapping[str, Any]] = {}
    for item in configurations_raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise AnalysisError("Issue #146 configuration is invalid")
        identifier = item["id"]
        if identifier in configurations:
            raise AnalysisError("Issue #146 configuration IDs are not unique")
        configurations[identifier] = dict(item)
    if len(raw_order) != 6 or any(
        raw_order.count(identifier) != 2 for identifier in configurations
    ):
        raise AnalysisError("Issue #146 run order must contain two instances per configuration")
    if set(raw_order) != set(configurations) or set(configurations) != {"C018", "C081", "C108"}:
        raise AnalysisError("Issue #146 run order/configurations do not match the plan")
    provenance = plan.get("provenance")
    required_fields = (
        provenance.get("required_environment_fields") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(required_fields, list) or not all(
        isinstance(item, str) for item in required_fields
    ):
        raise AnalysisError("Issue #146 environment provenance fields are missing")
    board_serial = scope.get("board_serial")
    bus_id = scope.get("pci_bus_id")
    device_node = scope.get("device_node")
    if not all(isinstance(value, str) and value for value in (board_serial, bus_id, device_node)):
        raise AnalysisError("Issue #146 board identity is incomplete")
    plateau = plan.get("plateau")
    if not isinstance(plateau, Mapping):
        raise AnalysisError("Issue #146 plateau is missing")
    start = finite(plateau.get("start_seconds"), field="plateau start")
    end = finite(plateau.get("end_seconds"), field="plateau end")
    minimum = integer(plateau.get("minimum_samples"), field="minimum samples")
    if (start, end, minimum) != (60.0, 180.0, 50):
        raise AnalysisError("Issue #146 plateau contract is not revision 3")
    scope_smoke = scope.get("smoke_runs")
    if not isinstance(scope_smoke, Mapping):
        raise AnalysisError("Issue #146 smoke timing is missing")
    target = finite(scope.get("run_duration_seconds"), field="long target")
    guard = finite(scope.get("run_guard_seconds"), field="long guard")
    smoke_target = finite(scope_smoke.get("workload_seconds"), field="smoke target")
    smoke_guard = finite(scope_smoke.get("guard_seconds"), field="smoke guard")
    execution = plan.get("load_execution")
    if (
        not isinstance(execution, Mapping)
        or not isinstance(execution.get("mode"), str)
        or not execution["mode"].startswith(LOAD_MODE)
    ):
        raise AnalysisError("Issue #146 execution mode is unsupported")
    overrun = finite(execution.get("overrun_limit_seconds"), field="overrun limit")
    outer_repetitions = integer(execution.get("outer_repetitions"), field="outer repetitions")
    launches_per_sync = integer(execution.get("launches_per_sync"), field="launches per sync")
    if outer_repetitions <= 0 or launches_per_sync <= 0:
        raise AnalysisError("Issue #146 grouped load settings must be positive")
    if (target, guard, smoke_target, smoke_guard, overrun) != (180.0, 230.0, 40.0, 90.0, 10.0):
        raise AnalysisError("Issue #146 timing contract is not revision 3")
    limits = plan.get("limits")
    if not isinstance(limits, Mapping):
        raise AnalysisError("Issue #146 limits are missing")
    board_limit = finite(limits.get("board_power_limit_w"), field="board power limit")
    fmax = finite(limits.get("asic_fmax_mhz"), field="ASIC fmax")
    thm_limit = finite(limits.get("thm_limit_c"), field="ASIC thermal limit")
    gddr_limit = finite(limits.get("gddr_thm_limit_c"), field="GDDR thermal limit")
    criteria = plan.get("criteria")
    if not isinstance(criteria, Mapping):
        raise AnalysisError("Issue #146 criteria are missing")
    interval = criteria.get("predicted_regulated_interval_w")
    if not isinstance(interval, list) or len(interval) != 2:
        raise AnalysisError("Issue #146 regulated interval is missing")
    regulated_interval = (
        finite(interval[0], field="regulated interval low"),
        finite(interval[1], field="regulated interval high"),
    )
    try:
        expected_configurations = {
            identifier: issue137_runner.configuration(plan, identifier).as_record()
            for identifier in configurations
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("Issue #146 core-plan configuration cannot be derived") from exc
    return _PlanSpec(
        plan=plan,
        plan_sha256=plan_sha256(plan),
        order=tuple(raw_order),
        configurations=configurations,
        expected_configurations=expected_configurations,
        required_environment_fields=tuple(required_fields),
        board_serial=board_serial,
        board_id=board_serial,
        bus_id=bus_id,
        device_node=device_node,
        target_seconds=target,
        guard_seconds=guard,
        smoke_target_seconds=smoke_target,
        smoke_guard_seconds=smoke_guard,
        overrun_limit_seconds=overrun,
        plateau_start_seconds=start,
        plateau_end_seconds=end,
        minimum_samples=minimum,
        idle_fraction_limit=IDLE_FRACTION_LIMIT,
        board_power_limit=board_limit,
        asic_fmax=fmax,
        asic_temperature_limit=thm_limit,
        gddr_temperature_limit=gddr_limit,
        regulated_interval=regulated_interval,
        outer_repetitions=outer_repetitions,
        launches_per_sync=launches_per_sync,
    )


def load_plan(path: Path = DEFAULT_PLAN_PATH) -> Mapping[str, Any]:
    """Load the immutable revision-3 plan without changing its bytes."""
    payload, raw = _read_json(Path(path), purpose="Issue #146 plan")
    if hashlib.sha256(raw).hexdigest() != AUTHORITATIVE_PLAN_RAW_SHA256:
        raise AnalysisError("Issue #146 plan bytes are not the immutable owner revision")
    _plan_spec(payload)
    return payload


def _record_inputs(records: Sequence[Mapping[str, Any] | str | Path]) -> list[_RecordInput]:
    result: list[_RecordInput] = []
    for index, item in enumerate(records):
        if isinstance(item, (str, Path)):
            path = _safe_path(item, purpose="Issue #146 record")
            payload, _raw = _read_json(path, purpose="Issue #146 record")
            nested = payload.get("runs")
            if isinstance(nested, list):
                result.extend(
                    _RecordInput(run, path.parent, path.name)
                    for run in nested
                    if isinstance(run, Mapping)
                )
            else:
                result.append(_RecordInput(payload, path.parent, path.name))
        elif isinstance(item, Mapping):
            nested = item.get("runs")
            if isinstance(nested, list):
                result.extend(
                    _RecordInput(run, None, f"record-{index + 1}")
                    for run in nested
                    if isinstance(run, Mapping)
                )
            else:
                result.append(_RecordInput(item, None, f"record-{index + 1}"))
        else:
            raise AnalysisError("Issue #146 records must be objects or JSON paths")
    return result


def _record_value(record: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = record
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                value = _MISSING
                break
            value = value[key]
        if value is not _MISSING:
            return value
    return None


def _name(record_input: _RecordInput, index: int) -> str:
    run_id = _record_value(record_input.record, ("run", "run_id"), ("run_id",))
    if isinstance(run_id, str) and run_id:
        return Path(run_id).name
    return record_input.name or f"record-{index + 1}"


def _identity_from_environment(environment: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    board = environment.get("board")
    if not isinstance(board, Mapping):
        board = environment.get("board_info")
    identity = board.get("identity") if isinstance(board, Mapping) else None
    if not isinstance(identity, Mapping):
        identity = {}
    serial = board.get("serial") if isinstance(board, Mapping) else None
    board_id = (
        identity.get("board_id")
        or (board.get("board_id") if isinstance(board, Mapping) else None)
        or serial
    )
    bus_id = (
        identity.get("bus_id")
        or (board.get("bus_id") if isinstance(board, Mapping) else None)
        or (board.get("pci_bus_id") if isinstance(board, Mapping) else None)
    )
    return serial, board_id, bus_id


def _environment_errors(
    environment: Any, configuration: Mapping[str, Any] | None, spec: _PlanSpec
) -> list[str]:
    if not isinstance(environment, Mapping):
        return ["environment"]
    board = environment.get("board")
    identity = board.get("identity") if isinstance(board, Mapping) else None
    if not isinstance(identity, Mapping):
        identity = {}
    firmware = environment.get("firmware")
    values = {
        "board.serial": board.get("serial") if isinstance(board, Mapping) else None,
        "board.identity.board_id": identity.get("board_id"),
        "board.identity.bus_id": identity.get("bus_id"),
        "firmware.bundle": firmware.get("bundle") if isinstance(firmware, Mapping) else None,
        "kernel": environment.get("kernel"),
        "kernel_driver_version": environment.get("kernel_driver_version")
        or environment.get("kmd_version"),
        "image_digest": environment.get("image_digest") or environment.get("image"),
        "harness_commit": environment.get("harness_commit"),
        "device_node": environment.get("device_node"),
        "worker_core_count": environment.get("worker_core_count"),
        "worker_core_coordinates": environment.get("worker_core_coordinates"),
        "total_blocks": environment.get("total_blocks"),
        "outer_repetitions": environment.get("outer_repetitions"),
        "launches_per_sync": environment.get("launches_per_sync"),
    }
    errors = [field for field in spec.required_environment_fields if not values.get(field)]
    if values["board.serial"] != spec.board_serial:
        errors.append("board.serial.binding")
    if values["board.identity.board_id"] != spec.board_id:
        errors.append("board.identity.board_id.binding")
    if values["board.identity.bus_id"] != spec.bus_id:
        errors.append("board.identity.bus_id.binding")
    if values["device_node"] != spec.device_node:
        errors.append("device_node.binding")
    if configuration is not None:
        expected = {
            "worker_core_count": configuration.get("worker_core_count"),
            "worker_core_coordinates": configuration.get("worker_core_coordinates"),
            "total_blocks": configuration.get("total_blocks"),
        }
        for field, value in expected.items():
            if values[field] != value:
                errors.append(f"environment.{field}.binding")
    if values["outer_repetitions"] != spec.outer_repetitions:
        errors.append("environment.outer_repetitions.binding")
    if values["launches_per_sync"] != spec.launches_per_sync:
        errors.append("environment.launches_per_sync.binding")
    return list(dict.fromkeys(errors))


def _configuration_errors(
    record: Mapping[str, Any], configuration_id: str | None, spec: _PlanSpec
) -> list[str]:
    configuration = record.get("configuration")
    errors: list[str] = []
    if not isinstance(configuration, Mapping):
        return ["configuration"]
    expected = spec.expected_configurations.get(configuration_id or "")
    if expected is None or dict(configuration) != dict(expected):
        errors.append("configuration.binding")
    if isinstance(record.get("core_plan"), Mapping) and record["core_plan"] != configuration.get(
        "core_plan"
    ):
        errors.append("core_plan.binding")
    result = record.get("runner_result")
    if isinstance(result, Mapping):
        if result.get("core_plan") != configuration.get("core_plan"):
            errors.append("runner_result.core_plan.binding")
        if result.get("plan_sha256") not in {None, spec.plan_sha256}:
            errors.append("runner_result.plan_sha256")
    return errors


def _unwrap_snapshot(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("snapshot")
    return nested if isinstance(nested, Mapping) else payload


def _snapshot_device(
    snapshot: Mapping[str, Any], spec: _PlanSpec
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """Select one board by independent board and bus identities."""
    devices = snapshot.get("device_info")
    if not isinstance(devices, list):
        return None, ["snapshot.device_info"]
    board_matches: list[Mapping[str, Any]] = []
    bus_matches: list[Mapping[str, Any]] = []
    pair_matches: list[Mapping[str, Any]] = []
    serial_errors: list[str] = []
    for device in devices:
        if not isinstance(device, Mapping):
            continue
        board = device.get("board_info")
        if not isinstance(board, Mapping):
            continue
        explicit_serial = board.get("serial") if "serial" in board else _MISSING
        board_id = board.get("board_id")
        if board_id is None:
            # tt-smi does not promise a raw serial field; retain the established
            # board_id-as-serial alias when the planned identity is present.
            board_id = explicit_serial if explicit_serial is not _MISSING else None
        bus_id = board.get("bus_id") or board.get("pci_bus_id")
        board_matches_identity = str(board_id) == spec.board_id
        bus_matches_identity = str(bus_id) == spec.bus_id
        if board_matches_identity:
            board_matches.append(device)
        if bus_matches_identity:
            bus_matches.append(device)
        if board_matches_identity and bus_matches_identity:
            pair_matches.append(device)
            if explicit_serial is not _MISSING and str(explicit_serial) != spec.board_serial:
                serial_errors.append("snapshot.serial")
    errors = list(serial_errors)
    if len(board_matches) != 1 or len(bus_matches) != 1 or len(pair_matches) != 1:
        errors.append("snapshot.identity")
    if errors:
        return None, list(dict.fromkeys(errors))
    return pair_matches[0], []


def _load_snapshot(
    record: Mapping[str, Any], key: str, *, base_dir: Path | None, spec: _PlanSpec
) -> _LoadedSnapshot:
    value = record.get(key)
    if not isinstance(value, Mapping):
        return _LoadedSnapshot(None, None, None, None, None, None, [key])
    source_name: str | None = None
    source_sha256: str | None = None
    source_byte_count: int | None = None
    payload: Mapping[str, Any] = value
    errors: list[str] = []
    sidecar = (
        isinstance(value.get("path"), str)
        and "device_info" not in value
        and "snapshot" not in value
    )
    raw: bytes | None = None
    if sidecar:
        try:
            path = _safe_artifact_path(value["path"], base_dir=base_dir, purpose=key)
            raw = path.read_bytes()
            source_name = path.name
            source_sha256, source_byte_count, hash_errors = _source_metadata(
                value, raw, purpose="snapshot"
            )
            errors.extend(f"{key}.{error}" for error in hash_errors)
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"{key}.read: {type(exc).__name__}")
                return _LoadedSnapshot(
                    None,
                    None,
                    source_name,
                    source_sha256,
                    source_byte_count,
                    None,
                    list(dict.fromkeys(errors)),
                )
            if not isinstance(decoded, Mapping):
                errors.append(f"{key}.object")
                return _LoadedSnapshot(
                    None,
                    None,
                    source_name,
                    source_sha256,
                    source_byte_count,
                    None,
                    list(dict.fromkeys(errors)),
                )
            payload = decoded
        except (AnalysisError, OSError) as exc:
            return _LoadedSnapshot(
                None,
                None,
                Path(str(value.get("path"))).name,
                None,
                None,
                None,
                [str(exc)],
            )
        if value.get("schema") != SNAPSHOT_SCHEMA:
            errors.append(f"{key}.descriptor_schema")
    elif value.get("schema") != SNAPSHOT_SCHEMA:
        errors.append(f"{key}.schema")
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        errors.append(f"{key}.schema")
    capture_timestamp = payload.get("timestamp_monotonic_ns")
    if not isinstance(capture_timestamp, int):
        capture_timestamp = None
    if payload.get("status") not in {None, "ok"} or payload.get("returncode") not in {None, 0}:
        errors.append(f"{key}.capture")
    snapshot = _unwrap_snapshot(payload)
    device, identity_errors = _snapshot_device(snapshot, spec)
    errors.extend(f"{key}.{error}" for error in identity_errors)
    unique_errors = list(dict.fromkeys(errors))
    if unique_errors:
        device = None
    return _LoadedSnapshot(
        snapshot,
        device,
        source_name,
        source_sha256,
        source_byte_count,
        capture_timestamp,
        unique_errors,
    )


def _raw_word(value: Any, *, field: str) -> int | None:
    if isinstance(value, str) and value.startswith(("0x", "0X")):
        try:
            number = int(value, 16)
        except (TypeError, ValueError):
            return None
        return number if 0 <= number <= 0xFFFFFFFF else None
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF:
        return value
    return None


def _limit_number(value: Any, *, unit: str) -> float | None:
    if isinstance(value, Mapping):
        source_unit = value.get("unit", value.get("units"))
        value = value.get("value", _MISSING)
        if value is _MISSING:
            return None
        if (
            source_unit is not None
            and str(source_unit).replace("°", "deg").casefold()
            != unit.replace("°", "deg").casefold()
        ):
            return None
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if text.lower().startswith(("0x", "+0x", "-0x")):
        return None
    for suffix in ("degC", "°C", "MHz", "W"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _pre_configuration(
    device: Mapping[str, Any] | None, spec: _PlanSpec
) -> tuple[dict[str, Any], list[str]]:
    if device is None:
        return {}, ["configuration.snapshot"]
    smbus = device.get("smbus_telem")
    limits = device.get("limits")
    errors: list[str] = []
    if not isinstance(smbus, Mapping):
        smbus = {}
        errors.append("configuration.smbus_telem")
    if not isinstance(limits, Mapping):
        limits = {}
        errors.append("limits")
    board_word = _raw_word(smbus.get("BOARD_POWER_LIMIT"), field="BOARD_POWER_LIMIT")
    max_word = _raw_word(smbus.get("ENABLED_MAX_ARB"), field="ENABLED_MAX_ARB")
    if board_word != 0x12C:
        errors.append("configuration.BOARD_POWER_LIMIT")
    if max_word != EXPECTED_ENABLED_MAX:
        errors.append("configuration.ENABLED_MAX_ARB")
    values = {
        "board_power_limit_w": _limit_number(limits.get("board_power_limit"), unit="W"),
        "asic_fmax_mhz": _limit_number(limits.get("asic_fmax"), unit="MHz"),
        "therm_trip_l1_limit_c": _limit_number(limits.get("therm_trip_l1_limit"), unit="degC"),
        "board_power_limit_word": board_word,
        "enabled_max_arb_word": max_word,
    }
    for field, expected in (
        ("board_power_limit_w", spec.board_power_limit),
        ("asic_fmax_mhz", spec.asic_fmax),
        ("therm_trip_l1_limit_c", spec.asic_temperature_limit),
    ):
        value = values[field]
        if value is None:
            errors.append(f"limits.{field}")
        elif not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            errors.append(f"limits.{field}.binding")
            if field == "board_power_limit_w":
                errors.append("configuration.BOARD_POWER_LIMIT")
    return values, list(dict.fromkeys(errors))


def _source_metadata(
    descriptor: Mapping[str, Any], raw: bytes, *, purpose: str
) -> tuple[str, int, list[str]]:
    expected_hash = descriptor.get("sha256", descriptor.get(f"{purpose}_sha256"))
    expected_bytes = descriptor.get("byte_count", descriptor.get(f"{purpose}_byte_count"))
    errors: list[str] = []
    actual_hash = hashlib.sha256(raw).hexdigest()
    if not isinstance(expected_hash, str) or expected_hash != actual_hash:
        errors.append(f"{purpose}_sha256")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes != len(raw)
    ):
        errors.append(f"{purpose}_byte_count")
    return actual_hash, len(raw), errors


def _trace_provenance(record: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    candidates: list[Any] = [payload.get("unit_provenance"), record.get("unit_provenance")]
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        candidates.append(provenance.get("unit_provenance"))
    trace = record.get("trace")
    if isinstance(trace, Mapping):
        candidates.append(trace.get("unit_provenance"))
    return any(candidate == PLAN_PROVENANCE for candidate in candidates)


def _csv_payload(raw: bytes) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(text.splitlines())
        if tuple(reader.fieldnames or ()) != TRACE_COLUMNS:
            raise AnalysisError("Issue #146 trace CSV columns are invalid")
        rows = [dict(row) for row in reader]
    except (UnicodeError, csv.Error) as exc:
        raise AnalysisError("Issue #146 trace CSV cannot be read") from exc
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(TRACE_COLUMNS),
        "samples": rows,
    }


def _load_trace(
    record: Mapping[str, Any], *, base_dir: Path | None, first: int | None, target_end: int | None
) -> _LoadedTrace:
    descriptor = record.get("trace")
    if not isinstance(descriptor, Mapping):
        return _LoadedTrace([], None, None, None, ["trace"], {"trace"}, False, None)
    payload: Mapping[str, Any] | None = None
    raw: bytes | None = None
    source_name: str | None = None
    errors: list[str] = []
    try:
        if isinstance(descriptor.get("path"), str) and "samples" not in descriptor:
            path = _safe_artifact_path(descriptor["path"], base_dir=base_dir, purpose="trace")
            raw = path.read_bytes()
            source_name = path.name
            if path.suffix.casefold() == ".json":
                payload = json.loads(raw.decode("utf-8"))
            else:
                payload = _csv_payload(raw)
            if isinstance(payload, Mapping):
                payload = dict(payload)
                for field in (
                    "run_start_monotonic_ns",
                    "run_end_monotonic_ns",
                    "actual_end_monotonic_ns",
                ):
                    if field in descriptor:
                        payload[field] = descriptor[field]
                if "unit_provenance" in descriptor:
                    payload["unit_provenance"] = descriptor["unit_provenance"]
            if not isinstance(payload, Mapping):
                raise AnalysisError("trace source must be an object")
        else:
            payload = descriptor
            raw = canonical_trace_bytes(payload)
            source_name = Path(str(descriptor.get("path", "inline-trace"))).name
    except (AnalysisError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _LoadedTrace([], source_name, None, None, [str(exc)], {"trace"}, False, payload)
    source_hash, source_bytes, hash_errors = _source_metadata(descriptor, raw, purpose="trace")
    errors.extend(hash_errors)
    if payload.get("schema") != TRACE_SCHEMA:
        errors.append("trace.schema")
    if payload.get("schema_version") != power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA_VERSION:
        errors.append("trace.schema_version")
    if tuple(payload.get("columns", ())) != TRACE_COLUMNS:
        errors.append("trace.columns")
    raw_rows = payload.get("samples")
    missing: set[str] = set()
    if not isinstance(raw_rows, list):
        return _LoadedTrace(
            [],
            source_name,
            source_hash,
            source_bytes,
            errors + ["trace.samples"],
            {"trace.samples"},
            _trace_provenance(record, payload),
            payload,
        )
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            missing.add(f"telemetry.row_{index}")
            continue
        for field in TRACE_COLUMNS:
            nullable = field in {"throttle_status", "throttle_reason"}
            if field not in row or (not nullable and (row[field] is None or row[field] == "")):
                missing.add(f"telemetry.row_{index}.{field}")
        for field in ARBITER_FIELDS:
            if field not in row:
                missing.add(f"telemetry.row_{index}.{field}")
    source_established = _trace_provenance(record, payload)
    try:
        reader = power_trace_issue146.read_issue146_power_trace_mapping(
            payload,
            run_start_monotonic_ns=first,
            run_end_monotonic_ns=target_end,
            source_established=source_established,
        )
        errors.extend(str(error) for error in reader.get("errors", ()))
        rows = [dict(row) for row in reader.get("samples", ()) if isinstance(row, Mapping)]
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"trace reader: {type(exc).__name__}")
        rows = []
    timestamps = [row.get("timestamp_monotonic_ns") for row in rows]
    if any(not isinstance(timestamp, int) for timestamp in timestamps):
        missing.add("telemetry.timestamp_monotonic_ns")
    if any(left > right for left, right in pairwise(timestamps)):
        missing.add("telemetry.timestamp_order")
    if first is not None:
        trace_start = payload.get("run_start_monotonic_ns")
        if trace_start is None:
            errors.append("trace.run_start_monotonic_ns")
        elif trace_start != first:
            errors.append("trace.run_start_monotonic_ns.binding")
    if target_end is not None:
        trace_end = payload.get("run_end_monotonic_ns")
        if trace_end is None:
            errors.append("trace.run_end_monotonic_ns")
        elif trace_end != target_end:
            errors.append("trace.run_end_monotonic_ns.binding")
    return _LoadedTrace(
        rows,
        source_name,
        source_hash,
        source_bytes,
        list(dict.fromkeys(errors)),
        missing,
        source_established,
        payload,
    )


def _load_groups(record: Mapping[str, Any], *, base_dir: Path | None) -> _LoadedGroups:
    descriptor = record.get("groups", record.get("group_timings"))
    if not isinstance(descriptor, Mapping):
        if isinstance(descriptor, list):
            descriptor = {
                "schema": GROUPS_SCHEMA,
                "schema_version": 1,
                "groups": descriptor,
            }
        else:
            return _LoadedGroups([], None, None, None, ["groups"], {"groups"}, None)
    payload: Mapping[str, Any] | None = None
    raw: bytes | None = None
    source_name: str | None = None
    errors: list[str] = []
    try:
        if isinstance(descriptor.get("path"), str) and "groups" not in descriptor:
            path = _safe_artifact_path(descriptor["path"], base_dir=base_dir, purpose="groups")
            raw = path.read_bytes()
            source_name = path.name
            payload = json.loads(raw.decode("utf-8"))
        else:
            payload = descriptor
            raw = canonical_groups_bytes(payload)
            source_name = Path(str(descriptor.get("path", "inline-groups"))).name
        if not isinstance(payload, Mapping):
            raise AnalysisError("groups source must be an object")
    except (AnalysisError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _LoadedGroups([], source_name, None, None, [str(exc)], {"groups"}, payload)
    source_hash, source_bytes, hash_errors = _source_metadata(descriptor, raw, purpose="groups")
    errors.extend(hash_errors)
    if payload.get("schema") != GROUPS_SCHEMA:
        errors.append("groups.schema")
    if payload.get("schema_version") != 1:
        errors.append("groups.schema_version")
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list):
        return _LoadedGroups(
            [],
            source_name,
            source_hash,
            source_bytes,
            errors + ["groups.groups"],
            {"groups.groups"},
            payload,
        )
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(raw_groups):
        if not isinstance(group, Mapping):
            errors.append(f"groups.row_{index}")
            continue
        groups.append(dict(group))
    return _LoadedGroups(
        groups,
        source_name,
        source_hash,
        source_bytes,
        list(dict.fromkeys(errors)),
        set(),
        payload,
    )


def _run_identity(record: Mapping[str, Any]) -> tuple[str | None, int | None, str]:
    configuration = record.get("configuration")
    configuration_id = configuration.get("id") if isinstance(configuration, Mapping) else None
    if not isinstance(configuration_id, str):
        value = record.get("configuration_id")
        configuration_id = value if isinstance(value, str) else None
    run = record.get("run")
    run_number_value = (
        run.get("run_number") if isinstance(run, Mapping) else record.get("run_number")
    )
    try:
        run_number = (
            None if run_number_value is None else integer(run_number_value, field="run_number")
        )
    except Issue146AnalysisError:
        run_number = None
    mode_value = run.get("mode") if isinstance(run, Mapping) else record.get("mode")
    return configuration_id, run_number, mode_value if isinstance(mode_value, str) else "unknown"


def _duration_binding(
    record: Mapping[str, Any],
    groups: _LoadedGroups,
    *,
    mode: str,
    first: int | None,
    spec: _PlanSpec,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    run = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    result = (
        record.get("runner_result") if isinstance(record.get("runner_result"), Mapping) else None
    )
    target = spec.target_seconds if mode == "long" else spec.smoke_target_seconds
    guard = spec.guard_seconds if mode == "long" else spec.smoke_guard_seconds
    errors: list[str] = []
    if record.get("record_schema") != RECORD_SCHEMA:
        errors.append("record_schema")
    if record.get("record_revision") != 1:
        errors.append("record_revision")
    if record.get("issue") not in {None, "#146"}:
        errors.append("issue")
    if record.get("status") != "ok":
        errors.append("run_status")
    if record.get("plan_sha256") != spec.plan_sha256:
        errors.append("plan_sha256")
    if mode not in {"long", "smoke"}:
        errors.append("run.mode")
    if run.get("load_mode", record.get("load_mode")) != LOAD_MODE:
        errors.append("load_mode")
    if not groups.groups or groups.errors or groups.missing:
        errors.extend(groups.errors or ["groups"])
    facts: dict[str, Any] = {
        "target_seconds": target,
        "guard_seconds": guard,
        "first_launch_start_monotonic_ns": first,
        "workload_end_monotonic_ns": None,
        "last_group_end_monotonic_ns": None,
        "observed_duration_seconds": None,
    }
    if first is None:
        errors.append("first_launch_start_monotonic_ns")
    else:
        try:
            group_facts = group_timing_facts(
                groups.groups,
                first_launch_start_monotonic_ns=first,
                plateau_start_seconds=spec.plateau_start_seconds,
                plateau_end_seconds=spec.plateau_end_seconds,
                workload_duration_seconds=target,
                idle_fraction_limit=spec.idle_fraction_limit,
                overrun_limit_seconds=spec.overrun_limit_seconds,
            )
            facts.update(group_facts)
        except Issue146AnalysisError as exc:
            errors.append(f"groups.binding: {exc}")
    for field, expected in (("target_seconds", target), ("guard_seconds", guard)):
        supplied = run.get(field)
        if supplied is not None:
            try:
                if not math.isclose(
                    finite(supplied, field=field), expected, abs_tol=1e-12, rel_tol=0.0
                ):
                    errors.append(f"run.{field}.binding")
            except Issue146AnalysisError:
                errors.append(f"run.{field}")
    expected_duration = facts.get("last_group_end_monotonic_ns")
    if isinstance(expected_duration, int) and first is not None:
        expected_seconds = (expected_duration - first) / 1_000_000_000.0
        facts["observed_duration_seconds"] = expected_seconds
        supplied = _record_value(
            record,
            ("run", "observed_duration_seconds"),
            ("observed_duration_seconds",),
            ("run", "duration_seconds"),
        )
        if supplied is None and isinstance(result, Mapping):
            supplied = result.get("observed_duration_seconds", result.get("duration_seconds"))
        if supplied is None:
            errors.append("observed_duration_seconds")
        else:
            try:
                if not math.isclose(
                    finite(supplied, field="observed duration"),
                    expected_seconds,
                    abs_tol=1e-9,
                    rel_tol=0.0,
                ):
                    errors.append("observed_duration_seconds.binding")
            except Issue146AnalysisError:
                errors.append("observed_duration_seconds")
    if facts.get("overrun_seconds") is not None:
        if facts.get("coverage_pass") is False:
            errors.append("groups.coverage")
        if facts["overrun_seconds"] > spec.overrun_limit_seconds:
            errors.append("overrun_seconds.limit")
        if facts["last_group_end_monotonic_ns"] > first + round(guard * 1_000_000_000.0):
            errors.append("guard_seconds")
        supplied_end = run.get("workload_end_monotonic_ns")
        if supplied_end is not None and supplied_end != facts["workload_end_monotonic_ns"]:
            errors.append("run.workload_end_monotonic_ns.binding")
        supplied_last = run.get("last_group_end_monotonic_ns")
        if supplied_last is not None and supplied_last != facts["last_group_end_monotonic_ns"]:
            errors.append("run.last_group_end_monotonic_ns.binding")
    run_first = _record_value(
        record,
        ("run", "first_launch_start_monotonic_ns"),
        ("first_launch_start_monotonic_ns",),
    )
    if run_first is not None and first is not None and run_first != first:
        errors.append("first_launch_start_monotonic_ns.binding")
    return facts, list(dict.fromkeys(errors)), result


def _all_gddr_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return decode_gddr_row(row)


def _row_nominal(
    row: Mapping[str, Any],
    decoded: Mapping[str, Any],
    *,
    mode: str,
    spec: _PlanSpec,
    omit_gddr: bool = False,
) -> bool:
    try:
        arb = decode_arbiter_row(row)
        board_power = finite(row.get("board_power_w"), field="board power")
        asic = finite(row.get("asic_temp_c"), field="ASIC temperature")
        aiclk = finite(row.get("aiclk_mhz"), field="AICLK")
    except Issue146AnalysisError:
        return False
    board_values = decoded.get("board_values")
    if not isinstance(board_values, Sequence):
        return False
    median = float(decoded["board_median"])
    sample_sd = float(decoded["board_sample_sd"])
    if not (median - 3.0 * sample_sd <= board_power <= median + 3.0 * sample_sd):
        return False
    if arb["aiclk_arb_min_raw_int"] != EXPECTED_BUSY_MIN:
        return False
    if abs(aiclk - arb["aiclk_arb_max_frequency_mhz"]) > 2.0:
        return False
    if asic >= spec.asic_temperature_limit:
        return False
    if mode == "long-regulated" or mode == "smoke-regulated":
        if aiclk >= spec.asic_fmax or arb["aiclk_arb_max_frequency_mhz"] >= spec.asic_fmax:
            return False
        if arb["aiclk_arb_max_index"] not in {7, 8}:
            return False
    else:
        if aiclk != spec.asic_fmax or arb["aiclk_arb_max_raw_int"] != EXPECTED_CONTROL_MAX:
            return False
    if not omit_gddr:
        gddr = decoded.get("gddr")
        if not isinstance(gddr, Mapping) or gddr.get("max_c") is None:
            return False
        if float(gddr["max_c"]) >= spec.gddr_temperature_limit:
            return False
    return True


def _flag_glitches(
    rows: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    spec: _PlanSpec,
    original_indices: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], set[int], set[int], list[str]]:
    board_values = [finite(row.get("board_power_w"), field="board_power_w") for row in rows]
    median = statistics.median(board_values) if board_values else 0.0
    sample_sd = statistics.stdev(board_values) if len(board_values) >= 2 else 0.0
    decoded_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            arb = decode_arbiter_row(row)
            gddr = _all_gddr_values(row)
            decoded_rows.append(
                {
                    "row": row,
                    "row_index": index,
                    "arb": arb,
                    "gddr": gddr,
                    "board_values": board_values,
                    "board_median": median,
                    "board_sample_sd": sample_sd,
                }
            )
        except Issue146AnalysisError as exc:
            errors.append(f"telemetry.row_{index}: {exc}")
    whole: set[int] = set()
    r6_only: set[int] = set()
    glitches: list[dict[str, Any]] = []
    for decoded in decoded_rows:
        row = decoded["row"]
        position = decoded["row_index"]
        index = original_indices[position] if original_indices is not None else position
        arb = decoded["arb"]
        reasons: list[str] = []
        whole_sample = False
        try:
            if all(
                arb[field] == 0
                for field in (
                    "aiclk_arb_min_raw_int",
                    "aiclk_arb_max_raw_int",
                    "enabled_min_arb",
                    "enabled_max_arb",
                )
            ):
                reasons.append("all_arbiter_words_zero")
                whole_sample = True
            if finite(row.get("board_power_w"), field="board_power_w") == 0.0:
                reasons.append("board_power_zero")
                whole_sample = True
            nominal = _row_nominal(row, decoded, mode=mode, spec=spec)
            nominal_without_gddr = _row_nominal(row, decoded, mode=mode, spec=spec, omit_gddr=True)
            if (
                arb["enabled_max_arb"] != EXPECTED_ENABLED_MAX
                and arb["enabled_min_arb"] == EXPECTED_ENABLED_MIN
                and nominal
            ):
                reasons.append("enabled_max_arb_glitch")
                whole_sample = True
            if gddr_zero_word(row) and nominal_without_gddr:
                reasons.append("gddr_zero_word")
                r6_only.add(position)
            if whole_sample:
                whole.add(position)
                r6_only.discard(position)
            if reasons:
                glitches.append(
                    {
                        "row_index": index,
                        "plateau_position": position,
                        "reasons": reasons,
                        "excludes": "whole_sample" if whole_sample else "R6",
                    }
                )
        except (Issue146AnalysisError, KeyError, TypeError) as exc:
            errors.append(f"telemetry.row_{index}: {exc}")
    return glitches, whole, r6_only, errors


def _criterion(
    name: str, passed: bool | None, *, reason: str | None = None, **extra: Any
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "pass": passed,
        "status": "pass" if passed is True else "fail" if passed is False else "not_evaluable",
    }
    if reason:
        result["reason"] = reason
    result.update(extra)
    return result


def _all_rows_pass(rows: Sequence[Mapping[str, Any]], predicate: Any) -> bool:
    return all(bool(predicate(row)) for row in rows)


def _criteria(
    retained: Sequence[Mapping[str, Any]],
    retained_r6: Sequence[Mapping[str, Any]],
    decoded_by_id: Mapping[int, Mapping[str, Any]],
    *,
    configuration_id: str,
    spec: _PlanSpec,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], set[str]]:
    regulated = configuration_id in {"C081", "C108"}
    criteria: dict[str, dict[str, Any]] = {}
    statistics_report: dict[str, Any] = {}
    missing: set[str] = set()
    values_by_field: dict[str, list[Any]] = {}
    for field in ("board_power_w", "aiclk_mhz", "current_a", "power_w", "asic_temp_c", "fan_rpm"):
        values = [row.get(field) for row in retained if field in row]
        if values:
            try:
                values_by_field[field] = [finite(value, field=field) for value in values]
                statistics_report[field] = point_statistics(values_by_field[field])
            except Issue146AnalysisError:
                missing.add(f"telemetry.{field}")
        else:
            missing.add(f"telemetry.{field}")
    if len(retained) < spec.minimum_samples:
        missing.add("sample_count")
    if len(retained_r6) < spec.minimum_samples:
        missing.add("sample_count.R6")
    board_stats = statistics_report.get("board_power_w")
    aiclk_stats = statistics_report.get("aiclk_mhz")
    if regulated:
        if board_stats is None:
            criteria["R1-power-regulated"] = _criterion("R1-power-regulated", None)
        elif not board_stats["precision_pass"]:
            missing.add("precision.board_power_w")
            criteria["R1-power-regulated"] = _criterion(
                "R1-power-regulated", None, reason="99 percent half-width exceeds 10 W"
            )
        else:
            low, high = spec.regulated_interval
            passed = board_stats["ci_low"] <= high and board_stats["ci_high"] >= low
            criteria["R1-power-regulated"] = _criterion(
                "R1-power-regulated",
                passed,
                ci_low=board_stats["ci_low"],
                ci_high=board_stats["ci_high"],
                target_interval=[low, high],
            )
        if aiclk_stats is None or len(retained) < spec.minimum_samples:
            criteria["R2-arbiter-unclamped"] = _criterion("R2-arbiter-unclamped", None)
        else:
            passed = (
                _all_rows_pass(
                    retained,
                    lambda row: (
                        finite(row["aiclk_mhz"], field="aiclk_mhz") < spec.asic_fmax
                        and decoded_by_id[id(row)]["arb"]["aiclk_arb_max_frequency_mhz"]
                        < spec.asic_fmax
                    ),
                )
                and aiclk_stats["mean"] < 0.95 * spec.asic_fmax
            )
            criteria["R2-arbiter-unclamped"] = _criterion("R2-arbiter-unclamped", passed)
        criteria["R3-mask-unchanged"] = _criterion(
            "R3-mask-unchanged",
            _all_rows_pass(
                retained,
                lambda row: (
                    decoded_by_id[id(row)]["arb"]["enabled_max_arb"] == EXPECTED_ENABLED_MAX
                    and decoded_by_id[id(row)]["arb"]["enabled_min_arb"] == EXPECTED_ENABLED_MIN
                ),
            ),
        )
        criteria["R4-busy"] = _criterion(
            "R4-busy",
            _all_rows_pass(
                retained,
                lambda row: (
                    decoded_by_id[id(row)]["arb"]["aiclk_arb_min_raw_int"] == EXPECTED_BUSY_MIN
                ),
            ),
        )
        criteria["R5-aiclk-tracks-arbiter"] = _criterion(
            "R5-aiclk-tracks-arbiter",
            _all_rows_pass(
                retained,
                lambda row: (
                    abs(
                        finite(row["aiclk_mhz"], field="aiclk_mhz")
                        - decoded_by_id[id(row)]["arb"]["aiclk_arb_max_frequency_mhz"]
                    )
                    <= 2.0
                ),
            ),
        )
        criteria["R6-thermal-not-binding"] = _criterion(
            "R6-thermal-not-binding",
            _all_rows_pass(
                retained_r6,
                lambda row: (
                    finite(row["asic_temp_c"], field="asic_temp_c") < spec.asic_temperature_limit
                    and float(decoded_by_id[id(row)]["gddr"]["max_c"]) < spec.gddr_temperature_limit
                ),
            ),
        )
        r7_values = [decoded_by_id[id(row)]["arb"]["aiclk_arb_max_index"] for row in retained]
        r7_pass = all(value in {7, 8} for value in r7_values)
        r7_reason = None
        if not r7_pass:
            labels = {
                9: "doppler_critical",
                4: "ambiguous arbiter alternatives",
                6: "ambiguous arbiter alternatives",
                0: "ambiguous arbiter alternatives",
                1: "impossible mask index",
                2: "impossible mask index",
                3: "impossible mask index",
                5: "impossible mask index",
            }
            r7_reason = ", ".join(
                sorted(
                    {
                        labels.get(value, "unsupported index")
                        for value in r7_values
                        if value not in {7, 8}
                    }
                )
            )
        criteria["R7-reported-index"] = _criterion("R7-reported-index", r7_pass, reason=r7_reason)
    else:
        criteria["K1-clamped"] = _criterion(
            "K1-clamped",
            _all_rows_pass(
                retained,
                lambda row: (
                    finite(row["aiclk_mhz"], field="aiclk_mhz") == spec.asic_fmax
                    and decoded_by_id[id(row)]["arb"]["aiclk_arb_max_raw_int"]
                    == EXPECTED_CONTROL_MAX
                ),
            ),
        )
        if board_stats is None:
            criteria["K2-unregulated"] = _criterion("K2-unregulated", None)
        elif not board_stats["precision_pass"]:
            missing.add("precision.board_power_w")
            criteria["K2-unregulated"] = _criterion(
                "K2-unregulated", None, reason="99 percent half-width exceeds 10 W"
            )
        else:
            criteria["K2-unregulated"] = _criterion(
                "K2-unregulated",
                board_stats["ci_high"] < spec.board_power_limit,
                ci_low=board_stats["ci_low"],
                ci_high=board_stats["ci_high"],
            )
        criteria["K3-mask-and-busy"] = _criterion(
            "K3-mask-and-busy",
            criteria["K1-clamped"]["pass"] is True
            and _all_rows_pass(
                retained,
                lambda row: (
                    decoded_by_id[id(row)]["arb"]["enabled_max_arb"] == EXPECTED_ENABLED_MAX
                    and decoded_by_id[id(row)]["arb"]["enabled_min_arb"] == EXPECTED_ENABLED_MIN
                    and decoded_by_id[id(row)]["arb"]["aiclk_arb_min_raw_int"] == EXPECTED_BUSY_MIN
                ),
            ),
        )
        criteria["K4-thermal"] = _criterion(
            "K4-thermal",
            _all_rows_pass(
                retained_r6,
                lambda row: (
                    finite(row["asic_temp_c"], field="asic_temp_c") < spec.asic_temperature_limit
                    and float(decoded_by_id[id(row)]["gddr"]["max_c"]) < spec.gddr_temperature_limit
                ),
            ),
        )
    return criteria, statistics_report, missing


def _exploratory(
    rows: Sequence[Mapping[str, Any]],
    retained: Sequence[Mapping[str, Any]],
    decoded_by_id: Mapping[int, Mapping[str, Any]],
    *,
    first: int,
    post_device: Mapping[str, Any] | None,
    last_group_end: int | None,
    spec: _PlanSpec,
    duration_weighted: Mapping[str, Any],
    post_capture_timestamp: int | None,
) -> dict[str, Any]:
    residuals = [
        abs(
            finite(row["aiclk_mhz"], field="aiclk_mhz")
            - decoded_by_id[id(row)]["arb"]["aiclk_arb_max_frequency_mhz"]
        )
        for row in retained
    ]
    indices: dict[str, int] = {}
    for row in retained:
        index = str(decoded_by_id[id(row)]["arb"]["aiclk_arb_max_index"])
        indices[index] = indices.get(index, 0) + 1
    onset_clock = next(
        (
            (int(row["timestamp_monotonic_ns"]) - first) / 1_000_000_000.0
            for row in rows
            if decoded_by_id.get(id(row), {})
            .get("arb", {})
            .get("aiclk_arb_max_frequency_mhz", spec.asic_fmax)
            < spec.asic_fmax
        ),
        None,
    )
    onset_power = next(
        (
            (int(row["timestamp_monotonic_ns"]) - first) / 1_000_000_000.0
            for row in rows
            if abs(finite(row["board_power_w"], field="board_power_w") - spec.board_power_limit)
            <= 5.0
        ),
        None,
    )
    drift_rows: list[dict[str, Any]] = []
    for row in retained:
        decoded = decoded_by_id[id(row)]
        drift_rows.append(
            {
                "timestamp_seconds": (int(row["timestamp_monotonic_ns"]) - first) / 1_000_000_000.0,
                "aiclk_mhz": row.get("aiclk_mhz"),
                "asic_temp_c": row.get("asic_temp_c"),
                "fan_rpm": row.get("fan_rpm"),
                "max_gddr_temp_c": decoded["gddr"].get("max_c"),
            }
        )
    current_values = [
        finite(row["current_a"], field="current_a") for row in retained if "current_a" in row
    ]
    release: dict[str, Any] = {"status": "not_evaluable"}
    if isinstance(post_device, Mapping):
        smbus = post_device.get("smbus_telem")
        if isinstance(smbus, Mapping):
            raw = _raw_word(smbus.get("AICLK_ARB_MAX"), field="post AICLK_ARB_MAX")
            if raw is not None:
                delay = None
                if post_capture_timestamp is not None and last_group_end is not None:
                    delay = (post_capture_timestamp - last_group_end) / 1_000_000_000.0
                release = {
                    "status": "ok",
                    "aiclk_arb_max_raw": smbus.get("AICLK_ARB_MAX"),
                    "aiclk_arb_max_frequency_mhz": raw & 0xFFFF,
                    "aiclk_arb_max_index": raw >> 16,
                    "capture_delay_seconds": delay,
                    "plateau_mean_aiclk_mhz": point_statistics(
                        [finite(row["aiclk_mhz"], field="aiclk_mhz") for row in retained]
                    )["mean"]
                    if retained
                    else None,
                }
    return {
        "r5_residual_max_mhz": max(residuals) if residuals else None,
        "r5_residual_values_mhz": residuals,
        "reported_index_distribution": indices,
        "lag1": {
            "r1": point_statistics(
                [finite(row["aiclk_mhz"], field="aiclk_mhz") for row in retained]
            )["r1"]
            if retained
            else None,
            "n_eff": point_statistics(
                [finite(row["aiclk_mhz"], field="aiclk_mhz") for row in retained]
            )["n_eff"]
            if retained
            else None,
        },
        "drift": {"samples": drift_rows, "status": "reported" if drift_rows else "not_evaluable"},
        "onset": {
            "first_aiclk_arb_max_below_fmax_seconds": onset_clock,
            "first_board_power_within_5_w_of_limit_seconds": onset_power,
        },
        "busy_current_against_disabled_tdc": {
            "disabled_tdc_limit_a": 200.0,
            "sample_count": len(current_values),
            "min_a": min(current_values) if current_values else None,
            "max_a": max(current_values) if current_values else None,
            "mean_a": math.fsum(current_values) / len(current_values) if current_values else None,
        },
        "duration_weighted_left_held_board_power": duration_weighted,
        "post_load_arbiter_release": release,
    }


def _empty_exploratory(duration_weighted: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "r5_residual_max_mhz": None,
        "r5_residual_values_mhz": [],
        "reported_index_distribution": {},
        "lag1": {"r1": None, "n_eff": None},
        "drift": {"samples": [], "status": "not_evaluable"},
        "onset": {
            "first_aiclk_arb_max_below_fmax_seconds": None,
            "first_board_power_within_5_w_of_limit_seconds": None,
        },
        "busy_current_against_disabled_tdc": {
            "disabled_tdc_limit_a": 200.0,
            "sample_count": 0,
            "min_a": None,
            "max_a": None,
            "mean_a": None,
        },
        "duration_weighted_left_held_board_power": duration_weighted,
        "post_load_arbiter_release": {"status": "not_evaluable"},
        "status": "not_evaluable",
    }


def _analyze_one(record_input: _RecordInput, index: int, spec: _PlanSpec) -> dict[str, Any]:
    record = record_input.record
    name = _name(record_input, index)
    configuration_id, run_number, mode = _run_identity(record)
    expected_position = run_number in range(1, len(spec.order) + 1) and (
        configuration_id == spec.order[run_number - 1] if run_number is not None else False
    )
    expected_candidate = mode == "long" and expected_position
    configuration = (
        record.get("configuration") if isinstance(record.get("configuration"), Mapping) else None
    )
    base_errors = _configuration_errors(record, configuration_id, spec)
    environment_errors = _environment_errors(record.get("environment"), configuration, spec)
    pre_loaded = _load_snapshot(
        record, "pre_run_snapshot", base_dir=record_input.base_dir, spec=spec
    )
    pre_snapshot = pre_loaded.snapshot
    pre_device = pre_loaded.device
    limits, pre_configuration_errors = _pre_configuration(pre_device, spec)
    first_value = _record_value(
        record,
        ("run", "first_launch_start_monotonic_ns"),
        ("first_launch_start_monotonic_ns",),
        ("runner_result", "first_launch_start_monotonic_ns"),
    )
    try:
        first = None if first_value is None else integer(first_value, field="first launch")
    except Issue146AnalysisError:
        first = None
    configuration_only = bool(pre_configuration_errors)
    if configuration_only:
        # Revision 3 makes the pre-run configuration gate authoritative.  Do
        # not open any trace or group sidecar after it fails: malformed
        # telemetry must not contaminate the configuration-only result.
        post_loaded = _LoadedSnapshot(None, None, None, None, None, None, [])
        groups = _LoadedGroups([], None, None, None, [], set(), None)
        target = spec.target_seconds if mode == "long" else spec.smoke_target_seconds
        duration = {
            "target_seconds": target,
            "guard_seconds": spec.guard_seconds if mode == "long" else spec.smoke_guard_seconds,
            "first_launch_start_monotonic_ns": first,
            "workload_end_monotonic_ns": None,
            "last_group_end_monotonic_ns": None,
            "observed_duration_seconds": None,
            "groups": [],
        }
        duration_errors: list[str] = []
        trace = _LoadedTrace([], None, None, None, [], set(), False, None)
    else:
        post_loaded = _load_snapshot(
            record, "post_run_snapshot", base_dir=record_input.base_dir, spec=spec
        )
        groups = _load_groups(record, base_dir=record_input.base_dir)
        duration, duration_errors, _result = _duration_binding(
            record, groups, mode=mode, first=first, spec=spec
        )
        target_end = duration.get("workload_end_monotonic_ns")
        trace = _load_trace(
            record, base_dir=record_input.base_dir, first=first, target_end=target_end
        )
    post_snapshot = post_loaded.snapshot
    post_device = post_loaded.device
    post_capture_timestamp = post_loaded.capture_timestamp
    rows = trace.rows
    plateau_rows = (
        select_plateau_rows(
            rows,
            first_launch_start_monotonic_ns=first,
            start_seconds=spec.plateau_start_seconds,
            end_seconds=spec.plateau_end_seconds,
        )
        if first is not None
        else []
    )
    missing: set[str] = set(
        base_errors
        + environment_errors
        + pre_loaded.errors
        + post_loaded.errors
        + pre_configuration_errors
        + duration_errors
    )
    if not configuration_only:
        missing.update(trace.missing)
        missing.update(f"telemetry.{error}" for error in trace.errors)
        if (
            duration.get("idle_fraction") is not None
            and duration["idle_fraction"] > spec.idle_fraction_limit
        ):
            missing.add("load_contract.idle_fraction")
        if not trace.gddr_provenance:
            missing.add("gddr_provenance")
    if not expected_position and mode == "long":
        missing.add("ordered_evidence")
    if mode == "long" and run_number is None:
        missing.add("run_number")
    if mode == "long" and configuration_id not in spec.configurations:
        missing.add("configuration")
    glitches: list[dict[str, Any]] = []
    whole_indices: set[int] = set()
    r6_indices: set[int] = set()
    decoded_by_id: dict[int, dict[str, Any]] = {}
    criteria: dict[str, dict[str, Any]] = {}
    statistics_report: dict[str, Any] = {}
    duration_weighted: dict[str, Any]
    if first is not None:
        duration_weighted = left_held_mean(
            rows,
            field="board_power_w",
            first_launch_start_monotonic_ns=first,
            start_seconds=spec.plateau_start_seconds,
            end_seconds=spec.plateau_end_seconds,
        )
    else:
        duration_weighted = {"status": "not_evaluable", "reason": "missing first launch clock"}
    glitch_errors: list[str] = []
    if not pre_configuration_errors and plateau_rows and trace.gddr_provenance:
        mode_for_predicate = (
            "long-regulated" if configuration_id in {"C081", "C108"} else "long-control"
        )
        row_positions = {id(row): position for position, row in enumerate(rows)}
        plateau_original_indices = [row_positions[id(row)] for row in plateau_rows]
        glitches, whole_indices, r6_indices, glitch_errors = _flag_glitches(
            plateau_rows,
            mode=mode_for_predicate,
            spec=spec,
            original_indices=plateau_original_indices,
        )
        if len(glitches) > 2:
            missing.add("telemetry.glitch_cap")
        unflagged_max_mismatch = any(
            index not in whole_indices
            and decode_arbiter_row(row)["enabled_max_arb"] != EXPECTED_ENABLED_MAX
            for index, row in enumerate(plateau_rows)
        )
        if unflagged_max_mismatch:
            missing.add("configuration.ENABLED_MAX_ARB.plateau")
        retained = [row for index, row in enumerate(plateau_rows) if index not in whole_indices]
        retained_r6 = [
            row
            for index, row in enumerate(plateau_rows)
            if index not in whole_indices and index not in r6_indices
        ]
        for row in plateau_rows:
            try:
                decoded = decode_arbiter_row(row)
                decoded["gddr"] = decode_gddr_row(row)
                decoded_by_id[id(row)] = {"arb": decoded, "gddr": decoded["gddr"]}
            except Issue146AnalysisError as exc:
                glitch_errors.append(str(exc))
        if (
            not trace.errors
            and not trace.missing
            and not glitch_errors
            and not unflagged_max_mismatch
            and retained
            and retained_r6
            and configuration_id in spec.configurations
        ):
            criteria, statistics_report, criterion_missing = _criteria(
                retained,
                retained_r6,
                decoded_by_id,
                configuration_id=configuration_id or "",
                spec=spec,
            )
            missing.update(criterion_missing)
        if len(plateau_rows) < spec.minimum_samples:
            missing.add("sample_count.raw")
        if retained and statistics_report:
            post_device_for_exploration = post_device
            exploratory = _exploratory(
                rows,
                retained,
                decoded_by_id,
                first=first,
                post_device=post_device_for_exploration,
                last_group_end=duration.get("last_group_end_monotonic_ns"),
                spec=spec,
                duration_weighted=duration_weighted,
                post_capture_timestamp=post_capture_timestamp,
            )
        else:
            exploratory = _empty_exploratory(duration_weighted)
    else:
        missing.update(glitch_errors)
        exploratory = _empty_exploratory(duration_weighted)
    if glitch_errors and not configuration_only:
        missing.update(f"telemetry.{error}" for error in glitch_errors)
    if not plateau_rows and not configuration_only:
        missing.add("plateau.samples")
    if not configuration_only and mode == "long" and not isinstance(post_snapshot, Mapping):
        missing.add("post_run_snapshot")
    if not configuration_only and mode == "long" and not isinstance(pre_snapshot, Mapping):
        missing.add("pre_run_snapshot")
    failed = [
        f"run_{run_number}.{configuration_id}.{name}"
        for name, item in criteria.items()
        if item.get("pass") is False
    ]
    eligible = expected_candidate and not missing and bool(criteria)
    return {
        "run_id": _record_value(record, ("run", "run_id"), ("run_id",)) or name,
        "run_number": run_number,
        "configuration_id": configuration_id,
        "mode": mode,
        "status": record.get("status"),
        "expected_order_configuration": spec.order[run_number - 1]
        if run_number in range(1, 7)
        else None,
        "eligible_for_verdict": eligible,
        "source": {
            "record": name,
            "trace": trace.source_name,
            "trace_sha256": trace.source_sha256,
            "trace_byte_count": trace.source_byte_count,
            "groups": groups.source_name,
            "groups_sha256": groups.source_sha256,
            "groups_byte_count": groups.source_byte_count,
            "pre_run_snapshot": pre_loaded.source_name,
            "pre_run_snapshot_sha256": pre_loaded.source_sha256,
            "pre_run_snapshot_byte_count": pre_loaded.source_byte_count,
            "post_run_snapshot": post_loaded.source_name,
            "post_run_snapshot_sha256": post_loaded.source_sha256,
            "post_run_snapshot_byte_count": post_loaded.source_byte_count,
        },
        "duration": duration,
        "group_timings": duration.get("groups", []),
        "limits": limits,
        "plateau": {
            "selection": "[60,180) seconds from first_launch_start_monotonic_ns",
            "raw_count": len(plateau_rows),
            "retained_count": len(plateau_rows) - len(whole_indices),
            "r6_retained_count": len(plateau_rows) - len(whole_indices) - len(r6_indices),
            "no_held_point": True,
            "no_interpolation": True,
        },
        "glitches": glitches,
        "glitch_count": len(glitches),
        "criteria": criteria,
        "statistics": statistics_report,
        "exploratory": exploratory,
        "trace_rows": rows,
        "missing_fields": sorted(missing),
        "failed_criteria": failed,
    }


def _precondition(
    name: str,
    reports: Sequence[dict[str, Any]],
    criterion_name: str,
    *,
    expected_count: int,
) -> dict[str, Any]:
    values = [report.get("criteria", {}).get(criterion_name, {}).get("pass") for report in reports]
    passed: bool | None
    if len(reports) != expected_count or any(value is None for value in values):
        passed = None
    else:
        passed = all(value is True for value in values)
    return {
        "name": name,
        "pass": passed,
        "status": "pass" if passed is True else "fail" if passed is False else "not_evaluable",
        "runs": [
            {
                "run_number": report.get("run_number"),
                "configuration_id": report.get("configuration_id"),
                "criterion": criterion_name,
                "pass": report.get("criteria", {}).get(criterion_name, {}).get("pass"),
            }
            for report in reports
        ],
    }


def _preconditions(
    reports_by_position: Mapping[int, dict[str, Any]], spec: _PlanSpec
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for half, positions in (("half_1", (1, 2, 3)), ("half_2", (4, 5, 6))):
        reports = [
            reports_by_position[position]
            for position in positions
            if position in reports_by_position
        ]
        c018 = [report for report in reports if report.get("configuration_id") == "C018"]
        regulated = [
            report for report in reports if report.get("configuration_id") in {"C081", "C108"}
        ]
        p1 = _precondition("P1-control-unthrottled", c018, "K1-clamped", expected_count=1)
        p2 = _precondition(
            "P2-regulated-depressed", regulated, "R2-arbiter-unclamped", expected_count=2
        )
        c081 = next(
            (report for report in regulated if report.get("configuration_id") == "C081"), None
        )
        c108 = next(
            (report for report in regulated if report.get("configuration_id") == "C108"), None
        )
        c081_mean = c081.get("statistics", {}).get("aiclk_mhz", {}).get("mean") if c081 else None
        c108_mean = c108.get("statistics", {}).get("aiclk_mhz", {}).get("mean") if c108 else None
        p3_pass = None if c081_mean is None or c108_mean is None else c081_mean > c108_mean
        p3 = {
            "name": "P3-monotonic",
            "pass": p3_pass,
            "status": "pass"
            if p3_pass is True
            else "fail"
            if p3_pass is False
            else "not_evaluable",
            "C081_plateau_aiclk_mhz": c081_mean,
            "C108_plateau_aiclk_mhz": c108_mean,
            "operator": ">",
        }
        result[half] = {"P1": p1, "P2": p2, "P3": p3}
    return result


def _pooled(reports: Sequence[dict[str, Any]], spec: _PlanSpec) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for configuration_id in ("C018", "C081", "C108"):
        selected = [
            report
            for report in reports
            if report.get("configuration_id") == configuration_id and report.get("mode") == "long"
        ]
        runs_values: list[list[float]] = []
        for report in sorted(selected, key=lambda item: item.get("run_number") or 0):
            rows = report.get("trace_rows", [])
            first = report.get("duration", {}).get("first_launch_start_monotonic_ns")
            whole_indices = {
                item.get("plateau_position", item.get("row_index"))
                for item in report.get("glitches", [])
                if item.get("excludes") == "whole_sample"
            }
            if isinstance(rows, list) and isinstance(first, int):
                plateau = select_plateau_rows(rows, first_launch_start_monotonic_ns=first)
                runs_values.append(
                    [
                        finite(row["board_power_w"], field="board_power_w")
                        for row_index, row in enumerate(plateau)
                        if row_index not in whole_indices
                    ]
                )
        if len(runs_values) == 2 and all(runs_values):
            result[configuration_id] = pooled_statistics(runs_values)
        else:
            result[configuration_id] = {"status": "not_evaluable", "run_count": len(runs_values)}
    return result


def _analysis_plan(spec: _PlanSpec) -> dict[str, Any]:
    return {
        "record_schema": spec.plan.get("record_schema"),
        "revision": 3,
        "sha256": spec.plan_sha256,
        "source": DEFAULT_PLAN_PATH.name,
        "run_order": list(spec.order),
        "load_mode": LOAD_MODE,
        "plateau": {
            "start_seconds": spec.plateau_start_seconds,
            "end_seconds": spec.plateau_end_seconds,
            "minimum_samples_after_exclusions": spec.minimum_samples,
            "selection": "strict points relative to first_launch_start_monotonic_ns; no held point or interpolation",
        },
        "timing": {
            "long": {"target_seconds": spec.target_seconds, "guard_seconds": spec.guard_seconds},
            "smoke": {
                "target_seconds": spec.smoke_target_seconds,
                "guard_seconds": spec.smoke_guard_seconds,
            },
            "overrun_limit_seconds": spec.overrun_limit_seconds,
            "idle_fraction_limit": spec.idle_fraction_limit,
        },
        "statistics": {
            "per_run_confidence": 0.99,
            "pooled_confidence": 0.95,
            "precision_half_width_w": PRECISION_LIMIT_W,
            "r1": "adjacent centered products divided by centered sum of squares",
        },
    }


def analyze_runs(
    records: Sequence[Mapping[str, Any] | str | Path], *, plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Analyze long records board-free; smoke records remain diagnostics."""
    effective_plan = dict(plan) if isinstance(plan, Mapping) else dict(load_plan())
    spec = _plan_spec(effective_plan)
    inputs = _record_inputs(records)
    reports = [_analyze_one(item, index, spec) for index, item in enumerate(inputs)]
    by_position: dict[int, dict[str, Any]] = {}
    missing: set[str] = set()
    for report in reports:
        if report.get("mode") != "long":
            continue
        position = report.get("run_number")
        if not isinstance(position, int) or position not in range(1, 7):
            missing.add(f"ordered_evidence.{report.get('run_id')}")
            continue
        expected = spec.order[position - 1]
        if report.get("configuration_id") != expected:
            missing.add(f"run_{position}.{expected}.ordered_evidence")
            continue
        if position in by_position:
            missing.add(f"run_{position}.{expected}.duplicate_evidence")
            continue
        by_position[position] = report
        missing.update(
            f"run_{position}.{expected}.{field}" for field in report.get("missing_fields", [])
        )
    for position, configuration_id in enumerate(spec.order, 1):
        if position not in by_position:
            missing.add(f"run_{position}.{configuration_id}.ordered_evidence")
    preconditions = _preconditions(by_position, spec)
    failed_preconditions = [
        item.get("name", name)
        for values in preconditions.values()
        for name, item in values.items()
        if item.get("pass") is False
    ]
    failed_criteria = [
        criterion
        for report in reports
        if report.get("mode") == "long"
        for criterion in report.get("failed_criteria", [])
    ]
    missing = set(missing)
    if missing:
        classification = "not_evaluable"
        reason = "required evidence is missing: " + ", ".join(sorted(missing))
    elif failed_preconditions or failed_criteria:
        classification = "H-OTHER"
        reason = "failed preregistered criteria: " + ", ".join(
            sorted(set(failed_preconditions + failed_criteria))
        )
    else:
        classification = "H-DS"
        reason = "H-DS: all P1-P3 and R1-R7/K1-K4 criteria passed"
    hds_supported = classification == "H-DS"
    long_reports = [report for report in reports if report.get("mode") == "long"]
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "issue": "#146",
        "status": "not_evaluable" if missing else "ok",
        "classification": classification,
        "classification_reason": reason,
        "supported_hypotheses": ["H-DS"] if hds_supported else [],
        "failed_preconditions": sorted(set(failed_preconditions)),
        "failed_criteria": sorted(set(failed_criteria)),
        "missing_fields": sorted(missing),
        "verdict_evidence_complete": not missing,
        "plan": _analysis_plan(spec),
        "preconditions": preconditions,
        "hypotheses": {
            "H-DS": {
                "classification": "supported"
                if hds_supported
                else "not_supported"
                if not missing
                else "not_evaluable",
                "supported": hds_supported,
                "criteria": "P1-P3 and R1-R7/K1-K4",
            },
            "H-OTHER": {
                "classification": "selected" if classification == "H-OTHER" else "not_selected",
                "supported": classification == "H-OTHER",
                "failed_criteria": sorted(set(failed_criteria + failed_preconditions)),
            },
        },
        "runs": reports,
        "verdict_runs": [by_position[position] for position in sorted(by_position)],
        "pooled_secondary": _pooled(long_reports, spec),
        "exploratory": {
            "smoke_affects_long_verdict": False,
            "duration_weighted_crosscheck_affects_verdict": False,
            "non_long_runs": [
                {
                    "run_id": report.get("run_id"),
                    "run_number": report.get("run_number"),
                    "configuration_id": report.get("configuration_id"),
                    "mode": report.get("mode"),
                }
                for report in reports
                if report.get("mode") != "long"
            ],
        },
    }


def _smoke_rows(report: dict[str, Any], *, first: int) -> list[Mapping[str, Any]]:
    return select_plateau_rows(
        report.get("trace_rows", []),
        first_launch_start_monotonic_ns=first,
        start_seconds=SMOKE_START_SECONDS,
        end_seconds=SMOKE_END_SECONDS,
    )


def _smoke_rows_at_or_after(report: dict[str, Any], *, first: int) -> list[Mapping[str, Any]]:
    threshold = first + round(SMOKE_START_SECONDS * 1_000_000_000.0)
    rows = report.get("trace_rows", [])
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("timestamp_monotonic_ns"), int)
        and not isinstance(row.get("timestamp_monotonic_ns"), bool)
        and row["timestamp_monotonic_ns"] >= threshold
    ]


def _numeric_smoke_values(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[list[float] | None, str | None]:
    values: list[float] = []
    for index, row in enumerate(rows):
        if field not in row:
            return None, f"{field} is missing at row {index}"
        try:
            values.append(finite(row[field], field=field))
        except Issue146AnalysisError as exc:
            return None, str(exc)
    if not values:
        return None, f"{field} has no samples"
    return values, None


def _smoke_acceptance(report: dict[str, Any], spec: _PlanSpec) -> list[dict[str, Any]]:
    first = report.get("duration", {}).get("first_launch_start_monotonic_ns")
    rows_10_40 = _smoke_rows(report, first=first) if isinstance(first, int) else []
    rows_at_or_after_10 = (
        _smoke_rows_at_or_after(report, first=first) if isinstance(first, int) else []
    )
    groups = report.get("group_timings", [])
    try:
        if not isinstance(first, int):
            raise Issue146AnalysisError("first launch clock is unavailable")
        group_facts = group_timing_facts(
            groups,
            first_launch_start_monotonic_ns=first,
            plateau_start_seconds=SMOKE_START_SECONDS,
            plateau_end_seconds=SMOKE_END_SECONDS,
            workload_duration_seconds=spec.smoke_target_seconds,
            idle_fraction_limit=spec.idle_fraction_limit,
            overrun_limit_seconds=spec.overrun_limit_seconds,
        )
        idle_pass: bool | None = group_facts["idle_fraction"] <= spec.idle_fraction_limit
        idle_detail: Any = group_facts
    except (Issue146AnalysisError, KeyError, TypeError, ValueError, OverflowError) as exc:
        idle_pass = None
        idle_detail = {"reason": f"group timings unavailable: {exc}"}
    acceptance: list[dict[str, Any]] = [
        _criterion(
            "idle_fraction_[10,40)",
            idle_pass,
            window="[10,40)",
            idle_fraction=idle_detail.get("idle_fraction")
            if isinstance(idle_detail, Mapping)
            else None,
            reason=idle_detail.get("reason") if isinstance(idle_detail, Mapping) else None,
        )
    ]
    configuration_id = report.get("configuration_id")
    if configuration_id not in spec.configurations:
        reason = "unknown smoke configuration"
        acceptance[0] = _criterion(
            "idle_fraction_[10,40)", None, window="[10,40)", reason=reason
        )
        acceptance.extend(
            [
                _criterion(
                    "regulated_current_and_power_[10,40)",
                    None,
                    window="[10,∞)",
                    current_window="[10,∞)",
                    power_sd_window="[10,40)",
                    reason=reason,
                ),
                _criterion("arbiter_words_and_enabled_max", None, reason=reason),
                _criterion("R5_residual_[10,40)", None, window="[10,∞)", reason=reason),
                _criterion("R7_index_distribution_[10,40)", None, window="[10,∞)", reason=reason),
            ]
        )
        acceptance.append(
            _criterion(
                "smoke_failure_requires_plan_revision",
                False,
                reason="smoke configuration is not in the immutable plan",
            )
        )
        return acceptance
    if configuration_id in {"C081", "C108"}:
        currents, current_error = _numeric_smoke_values(rows_at_or_after_10, "current_a")
        powers, power_error = _numeric_smoke_values(rows_10_40, "board_power_w")
        if current_error or power_error:
            acceptance.append(
                _criterion(
                    "regulated_current_and_power_[10,40)",
                    None,
                    window="[10,∞)",
                    current_window="[10,∞)",
                    power_sd_window="[10,40)",
                    reason=current_error or power_error,
                )
            )
        else:
            assert currents is not None and powers is not None
            power_sd = statistics.stdev(powers) if len(powers) >= 2 else 0.0
            current_pass = min(currents) >= 150.0 and power_sd < 20.0
            acceptance.append(
                _criterion(
                    "regulated_current_and_power_[10,40)",
                    current_pass,
                    window="[10,∞)",
                    current_window="[10,∞)",
                    power_sd_window="[10,40)",
                    minimum_current_a=min(currents),
                    current_sample_count=len(currents),
                    power_sample_count=len(powers),
                    board_power_sample_sd_w=power_sd,
                )
            )
    else:
        acceptance.append(
            _criterion(
                "regulated_current_and_power_[10,40)",
                True,
                status="not_applicable",
                window="[10,∞)",
                current_window="[10,∞)",
                power_sd_window="[10,40)",
            )
        )
    trace_rows = report.get("trace_rows", [])
    if not isinstance(trace_rows, list) or not trace_rows:
        acceptance.append(
            _criterion(
                "arbiter_words_and_enabled_max",
                None,
                reason="trace rows are unavailable",
            )
        )
    else:
        decoded_trace: list[dict[str, Any]] = []
        word_error: str | None = None
        for index, row in enumerate(trace_rows):
            if not isinstance(row, Mapping) or not all(field in row for field in ARBITER_FIELDS):
                word_error = f"arbiter words are missing at row {index}"
                break
            try:
                decoded_trace.append(decode_arbiter_row(row))
            except (Issue146AnalysisError, KeyError, TypeError, ValueError) as exc:
                word_error = f"arbiter words are malformed at row {index}: {exc}"
                break
        acceptance.append(
            _criterion(
                "arbiter_words_and_enabled_max",
                None if word_error else all(
                    decoded["enabled_max_arb"] == EXPECTED_ENABLED_MAX
                    for decoded in decoded_trace
                ),
                reason=word_error,
            )
        )
    residuals: list[float] = []
    residual_error: str | None = None
    for index, row in enumerate(rows_at_or_after_10):
        if not isinstance(row, Mapping) or not all(field in row for field in ARBITER_FIELDS):
            residual_error = f"arbiter words are missing at row {index}"
            break
        try:
            decoded = decode_arbiter_row(row)
            residuals.append(
                abs(
                    finite(row["aiclk_mhz"], field="aiclk_mhz")
                    - decoded["aiclk_arb_max_frequency_mhz"]
                )
            )
        except (Issue146AnalysisError, KeyError, TypeError, ValueError) as exc:
            residual_error = f"R5 inputs are malformed at row {index}: {exc}"
            break
    acceptance.append(
        _criterion(
            "R5_residual_[10,40)",
            None if residual_error or not residuals else max(residuals) <= 2.0,
            window="[10,∞)",
            max_residual_mhz=max(residuals) if residuals else None,
            reason=residual_error or ("no samples at or after 10 s" if not residuals else None),
        )
    )
    indices: list[int] = []
    index_error: str | None = None
    decoded_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows_at_or_after_10):
        if not isinstance(row, Mapping) or not all(field in row for field in ARBITER_FIELDS):
            index_error = f"arbiter words are missing at row {index}"
            break
        try:
            decoded_rows.append(decode_arbiter_row(row))
            indices.append(decoded_rows[-1]["aiclk_arb_max_index"])
        except (Issue146AnalysisError, KeyError, TypeError, ValueError) as exc:
            index_error = f"R7 inputs are malformed at row {index}: {exc}"
            break
    if index_error or not indices:
        index_pass: bool | None = None
    elif configuration_id == "C018":
        index_pass = all(decoded["aiclk_arb_max_raw_int"] == EXPECTED_CONTROL_MAX for decoded in decoded_rows)
    else:
        index_pass = all(index in {7, 8} for index in indices)
    acceptance.append(
        _criterion(
            "R7_index_distribution_[10,40)",
            index_pass,
            window="[10,∞)",
            indices=indices,
            reason=index_error or ("no samples at or after 10 s" if not indices else None),
        )
    )
    acceptance.append(
        _criterion(
            "smoke_failure_requires_plan_revision",
            all(
                item.get("pass") is True or item.get("status") == "not_applicable"
                for item in acceptance
            ),
        )
    )
    return acceptance


def analyze_smoke(
    records: Sequence[Mapping[str, Any] | str | Path], *, plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return diagnostic smoke acceptance items; never return a long verdict."""
    effective_plan = dict(plan) if isinstance(plan, Mapping) else dict(load_plan())
    spec = _plan_spec(effective_plan)
    inputs = _record_inputs(records)
    reports = [_analyze_one(item, index, spec) for index, item in enumerate(inputs)]
    diagnostics = []
    for report in reports:
        if report.get("mode") != "smoke":
            continue
        diagnostics.append(
            {
                "run_id": report.get("run_id"),
                "configuration_id": report.get("configuration_id"),
                "acceptance": _smoke_acceptance(report, spec),
                "run": report,
            }
        )
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "issue": "#146",
        "mode": "smoke-diagnostic",
        "affects_long_verdict": False,
        "acceptance": [item for diagnostic in diagnostics for item in diagnostic["acceptance"]],
        "runs": diagnostics,
        "plan": _analysis_plan(spec),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--record", action="append", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--smoke", action="store_true", help="analyze records as smoke diagnostics")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = load_plan(args.plan)
        result = (
            analyze_smoke(args.record, plan=plan)
            if args.smoke
            else analyze_runs(args.record, plan=plan)
        )
        serialized = strict_json_dumps(_safe_public(result), indent=2) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            _write_text(args.output, serialized, purpose="analysis output")
    except (AnalysisError, Issue146AnalysisError, OSError, TypeError, ValueError) as exc:
        print(f"Issue #146 analysis rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARBITER_FIELDS",
    "DEFAULT_PLAN_PATH",
    "GROUPS_SCHEMA",
    "ISSUE146_RECORD_SCHEMA",
    "ISSUE146_SNAPSHOT_SCHEMA",
    "AnalysisError",
    "analyze_record_set",
    "analyze_runs",
    "analyze_smoke",
    "canonical_groups_bytes",
    "canonical_trace_bytes",
    "group_timing_facts",
    "load_plan",
    "main",
    "plan_sha256",
    "point_statistics",
    "select_plateau_rows",
]


def analyze_record_set(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Discoverable alias for callers that pass a persisted record set."""
    return analyze_runs(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
