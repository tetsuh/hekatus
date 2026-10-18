"""Analyze canonical, board-free Issue #137 measurement records.

Only persisted revision-2 runner records can provide verdict evidence.  The
analyzer never opens a device, reads a live telemetry source, or substitutes a
plan/post-run value for a missing pre-run fact.  Smoke and C054 records are
retained as diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench import power_trace_issue137

ANALYSIS_SCHEMA = "issue137-analysis-v1"
ISSUE137_SNAPSHOT_SCHEMA = "issue137-tt-smi-snapshot-v1"
ISSUE137_RECORD_SCHEMA = "issue137-measurement-v1"
ISSUE137_DEVICE_RESULT_SCHEMA = "issue137-device-result-v1"
ISSUE137_LOAD_MODE = "host-duration-bounded"
DEFAULT_PLAN_PATH = Path(
    "docs/measurements/2026-10-18-host-newton-schulz-issue137-plan.json"
)
PLATEAU_START_SECONDS = 60.0
PLATEAU_END_SECONDS = 120.0
LIMIT_TOLERANCE_FRACTION = 0.02
AICLK_DEPRESSION_FRACTION = 0.95
THERMAL_PIN_FRACTION = 0.98
DEFAULT_OVERRUN_LIMIT_SECONDS = 10.0

SIGNAL_UNITS = {
    "aiclk_mhz": "MHz",
    "board_power_w": "W",
    "power_w": "W",
    "current_a": "A",
    "voltage_v": "V",
    "asic_temp_c": "degC",
    "fan_rpm": "rpm",
}
SCALAR_FIELDS = tuple(SIGNAL_UNITS)
VERDICT_SIGNAL_FIELDS = (
    "aiclk_mhz",
    "board_power_w",
    "power_w",
    "current_a",
    "asic_temp_c",
)
GDDR_FIELDS = power_trace_issue137.ISSUE137_GDDR_COLUMNS
HYPOTHESIS_SIGNALS = {
    "H-BPL": ("board_power_w", "board_power_limit_w"),
    "H-TDC": ("current_a", "tdc_limit"),
    "H-TDP": ("power_w", "tdp_limit_w"),
    "H-THM": ("asic_temp_c", "therm_trip_l1_limit_c"),
}
LIMIT_ALIASES = {
    "board_power_limit_w": (
        "board_power_limit_w",
        "board_power_limit",
        "BOARD_POWER_LIMIT",
    ),
    "tdc_limit": ("tdc_limit", "tdc_limit_a", "TDC_LIMIT_MAX"),
    "tdp_limit_w": ("tdp_limit_w", "tdp_limit", "TDP_LIMIT_MAX"),
    "therm_trip_l1_limit_c": (
        "therm_trip_l1_limit_c",
        "therm_trip_l1_limit",
        "THERM_TRIP_L1_LIMIT",
    ),
    "asic_fmax_mhz": ("asic_fmax_mhz", "asic_fmax", "ASIC_FMAX"),
}
LIMIT_UNITS = {
    "board_power_limit_w": "W",
    "tdc_limit": "A",
    "tdp_limit_w": "W",
    "therm_trip_l1_limit_c": "degC",
    "asic_fmax_mhz": "MHz",
}
TRACE_REQUIRED_FIELDS = ("timestamp_monotonic_ns", *SCALAR_FIELDS)
_MISSING = object()
_PATH_RE = re.compile(
    r"(?:^|[\s'\"(])(?:/home/|/Users/|/root/|/tmp/|~[/\\]|[A-Za-z]:[\\/])"
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AnalysisError(ValueError):
    """Raised when the analysis contract or input shape cannot be represented."""


@dataclass(frozen=True)
class _PlanSpec:
    plan: Mapping[str, Any]
    order: tuple[str, ...]
    verdict_configurations: tuple[str, ...]
    configuration_roles: dict[str, dict[str, Any]]
    expected_configurations: dict[str, Mapping[str, Any]]
    plateau_start_seconds: float
    plateau_end_seconds: float
    expected_limits: dict[str, float]
    tolerance_fraction: float
    aiclk_fraction: float
    thermal_pin_fraction: float
    classification_policy: dict[str, str]
    long_target_seconds: float
    long_guard_seconds: float
    smoke_target_seconds: float
    smoke_guard_seconds: float
    overrun_limit_seconds: float
    board_serial: str
    pci_bus_id: str
    device_node: str
    required_environment_fields: tuple[str, ...]
    plan_sha256: str


@dataclass(frozen=True)
class _RecordInput:
    record: Mapping[str, Any]
    base_dir: Path | None = None
    record_name: str | None = None


@dataclass
class _TraceLoad:
    rows: list[dict[str, Any]]
    source_name: str | None
    source_sha256: str | None
    source_byte_count: int | None
    source_established: bool
    bounds_start_ns: int | None
    bounds_end_ns: int | None
    errors: list[str]
    missing_fields: set[str]
    reader_facts: dict[str, Any]


@dataclass
class _RunAnalysis:
    report: dict[str, Any]
    expected_candidate: bool
    eligible: bool
    verdict_position: int | None


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise AnalysisError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise AnalysisError(f"{field} must be finite")
    return number


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise AnalysisError(f"{field} must be a non-negative integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        try:
            number = int(value.strip(), 10)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError(f"{field} must be a non-negative integer") from exc
    else:
        raise AnalysisError(f"{field} must be a non-negative integer")
    if number < 0:
        raise AnalysisError(f"{field} must be a non-negative integer")
    return number


def _safe_public(value: Any) -> Any:
    """Copy facts while omitting host identity and user-specific paths."""
    if isinstance(value, Mapping):
        forbidden = {
            "hostname",
            "host_name",
            "username",
            "user_name",
            "worktree_path",
            "repository_path",
            "working_directory",
            "cwd",
        }
        return {
            str(key): _safe_public(child)
            for key, child in value.items()
            if str(key).casefold() not in forbidden
        }
    if isinstance(value, (list, tuple)):
        return [_safe_public(child) for child in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str) and (
        _PATH_RE.search(value)
        or value.startswith("/")
        or any(part == ".." for part in re.split(r"[/\\\\]", value))
    ):
        return "<redacted-path>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _trusted_roots() -> tuple[Path, ...]:
    return tuple(
        root.resolve(strict=False)
        for root in (_REPOSITORY_ROOT, Path(tempfile.gettempdir()), Path("/out"))
    )


def _safe_cli_path(value: str | Path, *, purpose: str, output: bool = False) -> Path:
    try:
        candidate = Path(value)
        resolved = candidate.resolve(strict=False)
        if output and candidate.is_symlink():
            raise AnalysisError(f"{purpose} path must not be a symbolic link")
    except (OSError, RuntimeError, TypeError) as exc:
        raise AnalysisError(f"{purpose} path cannot be resolved safely") from exc
    roots = _trusted_roots()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise AnalysisError(f"{purpose} path must be beneath a trusted project directory")
    return resolved


def _read_bytes(path: Path, *, purpose: str) -> bytes:
    checked = _safe_cli_path(path, purpose=purpose)
    try:
        return checked.read_bytes()
    except OSError as exc:
        raise AnalysisError(f"{purpose} cannot be read") from exc


def _load_json(path: Path, *, purpose: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(_read_bytes(path, purpose=purpose).decode("utf-8"))
    except (AnalysisError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{purpose} cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise AnalysisError(f"{purpose} must be an object")
    return payload


def _write_text(path: Path, text: str, *, purpose: str) -> None:
    checked = _safe_cli_path(path, purpose=purpose, output=True)
    try:
        checked.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise AnalysisError(f"{purpose} cannot be written") from exc


def _plan_hash(plan: Mapping[str, Any]) -> str:
    try:
        encoded = strict_json_dumps(plan, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalysisError("Issue #137 plan cannot be hashed") from exc
    return hashlib.sha256(encoded).hexdigest()


def _extract_number_from_text(text: str, patterns: Sequence[str], *, field: str) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _finite(match.group(1), field=field)
    raise AnalysisError(f"immutable plan does not define {field}")


def _expected_configurations(
    configurations: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Derive the runner's board-free configuration records, never a device."""
    try:
        from tools import issue137 as runner

        return {
            identifier: runner.configuration(plan, identifier).as_record()
            for identifier in configurations
        }
    except (ImportError, TypeError, ValueError, KeyError):
        return {}


def _plan_roles(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configurations = plan.get("configurations")
    if not isinstance(configurations, list) or not configurations:
        raise AnalysisError("Issue #137 plan configurations are missing")
    roles: dict[str, dict[str, Any]] = {}
    for item in configurations:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise AnalysisError("Issue #137 plan configuration is invalid")
        identifier = item["id"]
        if identifier in roles:
            raise AnalysisError("Issue #137 plan has duplicate configuration IDs")
        roles[identifier] = dict(item)
    return roles


def _plan_order(
    scope: Mapping[str, Any], roles: Mapping[str, Mapping[str, Any]]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_order = scope.get("order")
    if not isinstance(raw_order, list) or not all(isinstance(item, str) for item in raw_order):
        raise AnalysisError("Issue #137 plan order is missing")
    order = tuple(raw_order)
    if len(order) != 8 or set(order) != set(roles):
        raise AnalysisError("Issue #137 plan order is invalid")
    if any(order.count(identifier) != 2 for identifier in roles):
        raise AnalysisError("Issue #137 plan order must contain two runs per configuration")
    verdict = tuple(
        dict.fromkeys(
            identifier for identifier in order if roles[identifier].get("verdict_bearing") is True
        )
    )
    return order, verdict


def _plan_criteria(
    plan: Mapping[str, Any],
) -> tuple[float, float, float]:
    criteria = plan.get("criteria")
    if not isinstance(criteria, Mapping):
        raise AnalysisError("Issue #137 plan criteria are missing")
    tolerance_text = criteria.get("tolerance_rule")
    aiclk_text = criteria.get("aiclk_depressed")
    if not isinstance(tolerance_text, str) or not isinstance(aiclk_text, str):
        raise AnalysisError("Issue #137 criteria are incomplete")
    tolerance = _extract_number_from_text(
        tolerance_text,
        (r"within\s+([0-9]+(?:\.[0-9]+)?)\s*percent",),
        field="limit tolerance",
    ) / 100.0
    aiclk_fraction = _extract_number_from_text(
        aiclk_text,
        (r"<\s*([0-9]+(?:\.[0-9]+)?)\s*\*",),
        field="AICLK depression fraction",
    )
    pinned = criteria.get("pinned")
    if not isinstance(pinned, Mapping) or not isinstance(pinned.get("H-THM"), str):
        raise AnalysisError("Issue #137 pinned criteria are missing")
    thermal_fraction = _extract_number_from_text(
        pinned["H-THM"],
        (r"\(([0-9]+(?:\.[0-9]+)?)\s*percent",),
        field="thermal pin fraction",
    ) / 100.0
    return tolerance, aiclk_fraction, thermal_fraction


def _plan_limits(plan: Mapping[str, Any]) -> dict[str, float]:
    limits = plan.get("limits")
    if not isinstance(limits, Mapping):
        raise AnalysisError("Issue #137 expected limits are missing")
    expected_limits: dict[str, float] = {}
    for name, plan_key in (
        ("board_power_limit_w", "board_power_limit_w"),
        ("tdc_limit", "tdc_limit_a"),
        ("tdp_limit_w", "tdp_limit_w"),
        ("therm_trip_l1_limit_c", "therm_trip_l1_limit_c"),
        ("asic_fmax_mhz", "asic_fmax_mhz"),
    ):
        if plan_key not in limits:
            raise AnalysisError(f"Issue #137 plan is missing expected limit {plan_key}")
        expected_limits[name] = _finite(limits[plan_key], field=f"plan limits.{plan_key}")
    return expected_limits


def _plan_policy(plan: Mapping[str, Any]) -> dict[str, str]:
    policy = plan.get("classification_policy")
    if not isinstance(policy, Mapping):
        raise AnalysisError("Issue #137 classification_policy is missing")
    keys = ("exactly_one_supported", "more_than_one_supported", "none_supported", "not_evaluable")
    result: dict[str, str] = {}
    for key in keys:
        value = policy.get(key)
        if not isinstance(value, str) or not value:
            raise AnalysisError(f"Issue #137 classification policy is missing {key}")
        result[key] = value
    return result


def _plan_plateau(plan: Mapping[str, Any]) -> tuple[float, float]:
    plateau = plan.get("plateau")
    if not isinstance(plateau, Mapping):
        raise AnalysisError("Issue #137 plateau is missing")
    start = _finite(plateau.get("start_seconds"), field="plan plateau start_seconds")
    end = _finite(plateau.get("end_seconds"), field="plan plateau end_seconds")
    if start < 0.0 or end <= start:
        raise AnalysisError("Issue #137 plateau interval is invalid")
    return start, end


def _plan_durations(
    plan: Mapping[str, Any], scope: Mapping[str, Any]
) -> tuple[float, float, float, float, float]:
    execution = plan.get("load_execution")
    if not isinstance(execution, Mapping) or execution.get("mode") != ISSUE137_LOAD_MODE:
        raise AnalysisError("Issue #137 plan does not select host-duration-bounded execution")
    try:
        overrun = _finite(
            execution.get("overrun_limit_seconds", DEFAULT_OVERRUN_LIMIT_SECONDS),
            field="plan overrun_limit_seconds",
        )
        long_target = _finite(scope["run_duration_seconds"], field="plan long target")
        long_guard = _finite(scope["run_guard_seconds"], field="plan long guard")
        smoke_scope = scope["smoke_runs"]
        smoke_target = _finite(smoke_scope["workload_seconds"], field="plan smoke target")
        smoke_guard = _finite(smoke_scope["guard_seconds"], field="plan smoke guard")
    except (KeyError, TypeError) as exc:
        raise AnalysisError("Issue #137 plan duration bounds are incomplete") from exc
    if overrun < 0.0 or long_target <= 0.0 or long_guard < long_target:
        raise AnalysisError("Issue #137 long duration bounds are invalid")
    if smoke_target <= 0.0 or smoke_guard < smoke_target:
        raise AnalysisError("Issue #137 smoke duration bounds are invalid")
    return overrun, long_target, long_guard, smoke_target, smoke_guard


def _plan_identity(
    plan: Mapping[str, Any], scope: Mapping[str, Any]
) -> tuple[tuple[str, ...], str, str, str]:
    provenance = plan.get("provenance", {})
    required = provenance.get("required_environment_fields") if isinstance(provenance, Mapping) else None
    if not isinstance(required, list) or not all(
        isinstance(item, str) and item for item in required
    ):
        raise AnalysisError("Issue #137 required environment provenance is missing")
    board_serial = scope.get("board_serial")
    pci_bus_id = scope.get("pci_bus_id")
    device_node = scope.get("device_node")
    if not all(isinstance(value, str) and value for value in (board_serial, pci_bus_id, device_node)):
        raise AnalysisError("Issue #137 selected board identity is incomplete")
    return tuple(required), board_serial, pci_bus_id, device_node


def _plan_spec(plan: Mapping[str, Any]) -> _PlanSpec:
    if plan.get("record_schema") != "issue137-premeasurement-plan-v1":
        raise AnalysisError("unsupported Issue #137 plan schema")
    roles = _plan_roles(plan)
    scope = plan.get("measurement_scope")
    if not isinstance(scope, Mapping):
        raise AnalysisError("Issue #137 plan measurement_scope is missing")
    order, verdict = _plan_order(scope, roles)
    tolerance, aiclk_fraction, thermal_fraction = _plan_criteria(plan)
    expected_limits = _plan_limits(plan)
    policy = _plan_policy(plan)
    plateau_start, plateau_end = _plan_plateau(plan)
    overrun, long_target, long_guard, smoke_target, smoke_guard = _plan_durations(plan, scope)
    required, board_serial, pci_bus_id, device_node = _plan_identity(plan, scope)
    return _PlanSpec(
        plan=plan,
        order=order,
        verdict_configurations=verdict,
        configuration_roles=roles,
        expected_configurations=_expected_configurations(roles, plan),
        plateau_start_seconds=plateau_start,
        plateau_end_seconds=plateau_end,
        expected_limits=expected_limits,
        tolerance_fraction=tolerance,
        aiclk_fraction=aiclk_fraction,
        thermal_pin_fraction=thermal_fraction,
        classification_policy=policy,
        long_target_seconds=long_target,
        long_guard_seconds=long_guard,
        smoke_target_seconds=smoke_target,
        smoke_guard_seconds=smoke_guard,
        overrun_limit_seconds=overrun,
        board_serial=board_serial,
        pci_bus_id=pci_bus_id,
        device_node=device_node,
        required_environment_fields=required,
        plan_sha256=_plan_hash(plan),
    )


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


def _safe_name(value: Any) -> str:
    text = str(value)
    if _PATH_RE.search(text):
        return Path(text).name or "<redacted-path>"
    return Path(text).name if text not in {"", ".", ".."} else "<unnamed>"


def _record_name(record_input: _RecordInput, index: int) -> str:
    if record_input.record_name:
        return _safe_name(record_input.record_name)
    run_id = _record_value(record_input.record, ("run", "run_id"), ("run_id",))
    if isinstance(run_id, str) and run_id:
        return _safe_name(run_id)
    return f"record-{index + 1}"


def _source_path(value: Any, *, base_dir: Path | None, purpose: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise AnalysisError(f"{purpose} path is missing")
    if base_dir is None:
        raise AnalysisError(f"{purpose} path has no trusted record directory")
    reference = str(value)
    candidate_name = Path(reference).name
    if (
        Path(reference).is_absolute()
        or candidate_name != reference
        or reference in {".", ".."}
        or any(separator in reference for separator in ("/", "\\"))
    ):
        raise AnalysisError(f"{purpose} path must be a file name")
    trusted_base = _safe_cli_path(base_dir, purpose=f"{purpose} directory")
    resolved = _safe_cli_path(trusted_base / candidate_name, purpose=purpose)
    if not resolved.is_relative_to(trusted_base):
        raise AnalysisError(f"{purpose} path escapes its trusted record directory")
    return resolved


_TRACE_HASH_FIELDS = frozenset({"sha256", "byte_count", "trace_sha256", "trace_byte_count"})


def _canonical_trace_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = {
        key: value for key, value in payload.items() if key not in _TRACE_HASH_FIELDS
    }
    return strict_json_dumps(_safe_public(normalized), sort_keys=True).encode("utf-8")


def _trace_mapping(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trace = record.get("trace")
    if not isinstance(trace, Mapping):
        return None
    for key in ("source", "payload", "trace"):
        nested = trace.get(key)
        if isinstance(nested, Mapping) and "samples" in nested:
            return nested
    return trace if "samples" in trace else None


def _provenance_is_bound(value: Any) -> bool:
    return isinstance(value, Mapping) and value == power_trace_issue137.ISSUE137_UNIT_PROVENANCE


def _bound_gddr_provenance(record: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    explicit = payload.get("gddr_source_established")
    record_provenance = record.get("provenance")
    trace_metadata = record.get("trace")
    if not isinstance(record_provenance, Mapping) or not _provenance_is_bound(
        record_provenance.get("unit_provenance")
    ):
        return False
    if not isinstance(trace_metadata, Mapping) or not _provenance_is_bound(
        trace_metadata.get("unit_provenance")
    ):
        return False
    if explicit is False or payload.get("gddr_source_status") in {
        "unknown",
        "not_evaluable",
        "unestablished",
    }:
        return False
    candidates = [
        payload.get("unit_provenance"),
        record.get("unit_provenance"),
        trace_metadata.get("unit_provenance") if isinstance(trace_metadata, Mapping) else None,
    ]
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        candidates.append(provenance.get("unit_provenance"))
    result = record.get("runner_result")
    if isinstance(result, Mapping):
        result_provenance = result.get("provenance")
        if isinstance(result_provenance, Mapping):
            candidates.append(result_provenance.get("unit_provenance"))
        candidates.append(result.get("unit_provenance"))
    return any(_provenance_is_bound(candidate) for candidate in candidates)


def _copy_trace_row_metadata(raw_row: Mapping[str, Any], row: dict[str, Any]) -> None:
    metadata_fields = (
        "timestamp_utc",
        "snapshot_elapsed_ns",
        "heartbeat",
        "therm_trip_count",
        "throttle_status",
        "throttle_reason",
        *GDDR_FIELDS,
    )
    for field in metadata_fields:
        if field in raw_row:
            row[field] = raw_row[field]
    for field in ("MAX_GDDR_TEMP", "max_gddr_temp", "max_gddr_temp_raw"):
        if field in raw_row:
            row["MAX_GDDR_TEMP"] = raw_row[field]


def _parse_trace_row(raw_row: Mapping[str, Any], index: int, load: _TraceLoad) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in TRACE_REQUIRED_FIELDS:
        if field not in raw_row or raw_row[field] is None or raw_row[field] == "":
            load.missing_fields.add(field)
            continue
        try:
            if field == "timestamp_monotonic_ns":
                row[field] = _integer(raw_row[field], field=f"trace row {index} {field}")
            else:
                row[field] = _finite(raw_row[field], field=f"trace row {index} {field}")
        except AnalysisError as exc:
            load.errors.append(str(exc))
            load.missing_fields.add(field)
    _copy_trace_row_metadata(raw_row, row)
    return row


def _trace_has_missing_gddr(raw_rows: Sequence[Any]) -> bool:
    return any(
        field not in raw_row
        for raw_row in raw_rows
        if isinstance(raw_row, Mapping)
        for field in GDDR_FIELDS
    )


def _raw_trace_rows(payload: Mapping[str, Any], load: _TraceLoad) -> None:
    raw_rows = payload.get("samples")
    if not isinstance(raw_rows, list):
        load.errors.append("trace samples are missing")
        load.missing_fields.add("trace.samples")
        return
    for index, raw_row in enumerate(raw_rows):
        if isinstance(raw_row, Mapping):
            load.rows.append(_parse_trace_row(raw_row, index, load))
        else:
            load.errors.append(f"trace row {index} is not an object")
    if _trace_has_missing_gddr(raw_rows):
        load.errors.append("GDDR temperature fields are missing from one or more rows")


def _trace_payload_from_csv(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = _read_bytes(path, purpose="trace source")
        text = raw.decode("utf-8")
    except (AnalysisError, UnicodeError) as exc:
        raise AnalysisError("trace source cannot be read") from exc
    try:
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            raise AnalysisError("trace CSV has no header")
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise AnalysisError("trace CSV is malformed") from exc
    return {
        "schema": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA,
        "schema_version": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(power_trace_issue137.ISSUE137_POWER_TRACE_COLUMNS),
        "samples": rows,
    }, raw


def _extract_bounds(
    record: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    start_paths = (
        ("run_start_monotonic_ns",),
        ("workload_start_monotonic_ns",),
        ("run", "workload_start_monotonic_ns"),
        ("run", "start_monotonic_ns"),
        ("trace", "run_start_monotonic_ns"),
    )
    end_paths = (
        ("run_end_monotonic_ns",),
        ("workload_end_monotonic_ns",),
        ("run", "workload_end_monotonic_ns"),
        ("run", "end_monotonic_ns"),
        ("trace", "run_end_monotonic_ns"),
    )

    def choose(paths: Sequence[tuple[str, ...]], fallback: Any) -> int | None:
        value = _record_value(record, *paths)
        if value is None:
            value = fallback
        if value is None:
            return None
        try:
            return _integer(value, field="run monotonic bound")
        except AnalysisError:
            return None

    return choose(start_paths, payload.get("run_start_monotonic_ns")), choose(
        end_paths, payload.get("run_end_monotonic_ns")
    )


def _trace_file_payload(
    record: Mapping[str, Any], *, base_dir: Path | None
) -> tuple[Mapping[str, Any] | None, str | None, bytes | None, str | None]:
    trace = record.get("trace")
    path_value = trace.get("path") if isinstance(trace, Mapping) else None
    if path_value is None:
        return None, None, None, "trace source is missing"
    try:
        path = _source_path(path_value, base_dir=base_dir, purpose="trace source")
        if path.suffix.lower() == ".json":
            return _load_json(path, purpose="trace source"), path.name, _read_bytes(
                path, purpose="trace source"
            ), None
        payload, source_bytes = _trace_payload_from_csv(path)
        return payload, path.name, source_bytes, None
    except (OSError, TypeError, ValueError) as exc:
        return None, Path(str(path_value)).name, None, str(exc)


def _trace_load_error(source_name: str | None, error: str) -> _TraceLoad:
    return _TraceLoad(
        rows=[],
        source_name=source_name,
        source_sha256=None,
        source_byte_count=None,
        source_established=False,
        bounds_start_ns=None,
        bounds_end_ns=None,
        errors=[error],
        missing_fields={"trace"},
        reader_facts={},
    )


def _trace_hash_error(
    source_bytes: bytes, expected_hash: Any, expected_bytes: Any
) -> str | None:
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        return "trace SHA-256 provenance is missing or invalid"
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 1:
        return "trace byte-count provenance is missing or invalid"
    if hashlib.sha256(source_bytes).hexdigest() != expected_hash or len(source_bytes) != expected_bytes:
        return "trace source bytes do not match persisted hash/byte-count provenance"
    return None


def _trace_reader_facts(load: _TraceLoad, payload: Mapping[str, Any]) -> None:
    try:
        reader_payload = dict(payload)
        reader_payload["samples"] = [dict(row) for row in payload.get("samples", [])]
        if load.bounds_start_ns is not None:
            reader_payload["run_start_monotonic_ns"] = load.bounds_start_ns
        if load.bounds_end_ns is not None:
            reader_payload["run_end_monotonic_ns"] = load.bounds_end_ns
        facts = power_trace_issue137.read_issue137_power_trace_mapping(
            reader_payload,
            run_start_monotonic_ns=load.bounds_start_ns,
            run_end_monotonic_ns=load.bounds_end_ns,
            source_established=load.source_established,
        )
        load.reader_facts = _safe_public(facts)
        if facts.get("valid") is not True:
            load.errors.extend(str(error) for error in facts.get("errors", ()))
    except (TypeError, ValueError, KeyError) as exc:
        load.errors.append(f"trace reader rejected source: {type(exc).__name__}")


def _trace_metadata(record: Mapping[str, Any]) -> tuple[Any, Any]:
    trace_metadata = record.get("trace")
    if not isinstance(trace_metadata, Mapping):
        return None, None
    return (
        trace_metadata.get("sha256", trace_metadata.get("trace_sha256")),
        trace_metadata.get("byte_count", trace_metadata.get("trace_byte_count")),
    )


def _trace_schema_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema") != power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA:
        errors.append("Issue #137 trace schema is unsupported")
    if tuple(payload.get("columns", ())) != power_trace_issue137.ISSUE137_POWER_TRACE_COLUMNS:
        errors.append("Issue #137 trace columns are invalid")
    return errors


def _trace_bound_errors(record: Mapping[str, Any], payload: Mapping[str, Any], load: _TraceLoad) -> None:
    load.bounds_start_ns, load.bounds_end_ns = _extract_bounds(record, payload)
    if load.bounds_start_ns is None:
        load.missing_fields.add("run_start_monotonic_ns")
    if load.bounds_end_ns is None:
        load.missing_fields.add("run_end_monotonic_ns")
    if (
        load.bounds_start_ns is not None
        and load.bounds_end_ns is not None
        and load.bounds_end_ns <= load.bounds_start_ns
    ):
        load.errors.append("run monotonic bounds are not a positive interval")


def _load_trace_source(
    record: Mapping[str, Any],
    *,
    base_dir: Path | None,
    record_name: str,
    source_verified: bool = False,
) -> _TraceLoad:
    payload = _trace_mapping(record)
    source_name: str | None = None
    source_bytes: bytes | None = None
    if payload is None:
        payload, source_name, source_bytes, load_error = _trace_file_payload(
            record, base_dir=base_dir
        )
        if load_error is not None or payload is None or source_bytes is None:
            return _trace_load_error(source_name, load_error or "trace source is missing")
    if source_name is None:
        source_name = str(record.get("trace_name") or f"{record_name}.trace")
    if source_bytes is None:
        source_bytes = _canonical_trace_bytes(payload)
    established = source_verified and _bound_gddr_provenance(record, payload)
    expected_hash, expected_bytes = _trace_metadata(record)
    load_hash_error = _trace_hash_error(source_bytes, expected_hash, expected_bytes)
    load = _TraceLoad(
        rows=[],
        source_name=Path(source_name).name,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_byte_count=len(source_bytes),
        source_established=established,
        bounds_start_ns=None,
        bounds_end_ns=None,
        errors=[],
        missing_fields=set(),
        reader_facts={},
    )
    if load_hash_error is not None:
        load.errors.append(load_hash_error)
        load.missing_fields.update({"trace_sha256", "trace_byte_count"})
    load.errors.extend(_trace_schema_errors(payload))
    _raw_trace_rows(payload, load)
    _trace_bound_errors(record, payload, load)
    _trace_reader_facts(load, payload)
    return load


def _snapshot_capture_errors(snapshot: Mapping[str, Any]) -> list[str]:
    if snapshot.get("status") not in {None, "ok"} or snapshot.get("returncode") not in {None, 0}:
        return ["pre-run snapshot capture did not complete successfully"]
    return []


def _snapshot_selected_device_errors(
    pre: Mapping[str, Any], spec: _PlanSpec
) -> list[str]:
    selected = pre.get("selected_device")
    if not isinstance(selected, Mapping):
        return []
    errors: list[str] = []
    if selected.get("board_id") != spec.board_serial or selected.get("bus_id") != spec.pci_bus_id:
        errors.append("pre-run snapshot selected-device identity does not match the owner plan")
    if selected.get("device_node") not in {None, spec.device_node}:
        errors.append("pre-run snapshot selected-device node does not match the owner plan")
    return errors


def _load_snapshot_payload(
    pre: Mapping[str, Any], *, base_dir: Path | None
) -> tuple[Mapping[str, Any], str | None, list[str]]:
    if isinstance(pre.get("path"), (str, Path)) and "snapshot" not in pre:
        try:
            path = _source_path(pre["path"], base_dir=base_dir, purpose="pre-run snapshot")
            return _load_json(path, purpose="pre-run snapshot"), path.name, []
        except AnalysisError as exc:
            return {}, Path(str(pre["path"])).name, [str(exc)]
    path_name = Path(str(pre.get("path"))).name if pre.get("path") else None
    return pre, path_name, []


def _pre_snapshot_mapping(
    record: Mapping[str, Any], *, base_dir: Path | None, spec: _PlanSpec
) -> tuple[Mapping[str, Any] | None, str | None, list[str]]:
    pre = record.get("pre_run_snapshot", record.get("pre_snapshot"))
    if not isinstance(pre, Mapping):
        return None, None, ["pre_run_snapshot"]
    errors = _snapshot_capture_errors(pre) + _snapshot_selected_device_errors(pre, spec)
    payload, path_name, path_errors = _load_snapshot_payload(pre, base_dir=base_dir)
    if path_errors:
        return None, path_name, path_errors
    errors.extend(_snapshot_capture_errors(payload))
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), Mapping) else payload
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("device_info"), list):
        errors.append("pre-run snapshot device identity is missing")
        return None, path_name, list(dict.fromkeys(errors))
    try:
        _selected_snapshot_device(snapshot, board_id=spec.board_serial, bus_id=spec.pci_bus_id)
    except AnalysisError as exc:
        errors.append(str(exc))
    return snapshot, path_name, list(dict.fromkeys(errors))


def _snapshot_device_matches(
    devices: Sequence[Any], *, board_id: str, bus_id: str
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    pair_matches: list[Mapping[str, Any]] = []
    board_matches: list[Mapping[str, Any]] = []
    bus_matches: list[Mapping[str, Any]] = []
    for device in devices:
        if not isinstance(device, Mapping):
            continue
        info = device.get("board_info")
        if not isinstance(info, Mapping):
            continue
        candidate_board = info.get("board_id", info.get("serial"))
        candidate_bus = info.get("bus_id", info.get("pci_bus_id"))
        if str(candidate_board) == str(board_id):
            board_matches.append(device)
        if str(candidate_bus) == str(bus_id):
            bus_matches.append(device)
        if str(candidate_board) == str(board_id) and str(candidate_bus) == str(bus_id):
            pair_matches.append(device)
    return pair_matches, board_matches, bus_matches


def _selected_snapshot_device(
    snapshot: Mapping[str, Any], *, board_id: str, bus_id: str
) -> Mapping[str, Any]:
    devices = snapshot.get("device_info")
    if not isinstance(devices, list) or not devices:
        raise AnalysisError("pre-run snapshot device_info is missing")
    pair_matches, board_matches, bus_matches = _snapshot_device_matches(
        devices, board_id=board_id, bus_id=bus_id
    )
    if len(board_matches) != 1 or len(bus_matches) != 1 or len(pair_matches) != 1:
        raise AnalysisError("pre-run snapshot selected-device identity is not unique")
    if pair_matches[0] is not board_matches[0] or pair_matches[0] is not bus_matches[0]:
        raise AnalysisError("pre-run snapshot board and BDF identify different devices")
    return pair_matches[0]


def _mapping_limit_value(
    value: Mapping[str, Any],
) -> tuple[Any, str | None, str | None, bool]:
    explicit_unit = value.get("unit", value.get("units", value.get("source_unit")))
    raw_value = value.get("value", _MISSING)
    conversion_unit: str | None = None
    conversion_proven = False
    if raw_value is _MISSING and value.get("converted_value") is not None:
        raw_value = value["converted_value"]
        conversion_unit = value.get("converted_unit")
        explicit_unit = conversion_unit or explicit_unit
        conversion_proven = isinstance(value.get("conversion_provenance"), Mapping)
    return raw_value, explicit_unit, conversion_unit, conversion_proven


def _unit_literal(text: str) -> bool:
    return bool(text) and all(
        ("A" <= character <= "Z")
        or ("a" <= character <= "z")
        or character == "°"
        for character in text
    )


def _split_numeric_unit(text: str) -> tuple[str, str | None] | None:
    text = text.strip()
    if not text:
        return None
    index = 1 if text[0] in "+-" else 0
    mantissa_start = index
    while index < len(text) and text[index].isdecimal():
        index += 1
    has_whole_digits = index > mantissa_start
    has_fraction_digits = False
    if index < len(text) and text[index] == ".":
        index += 1
        fraction_start = index
        while index < len(text) and text[index].isdecimal():
            index += 1
        has_fraction_digits = index > fraction_start
    if not has_whole_digits and not has_fraction_digits:
        return None
    if index < len(text) and text[index] in "eE":
        exponent_at = index
        exponent_index = index + 1
        if exponent_index < len(text) and text[exponent_index] in "+-":
            exponent_index += 1
        exponent_start = exponent_index
        while exponent_index < len(text) and text[exponent_index].isdecimal():
            exponent_index += 1
        if exponent_index == exponent_start:
            unit_text = text[exponent_at:].strip()
            if not _unit_literal(unit_text):
                return None
            return text[:exponent_at].strip(), unit_text
        index = exponent_index
    unit_text = text[index:].strip()
    if unit_text and not _unit_literal(unit_text):
        return None
    return text[:index].strip(), unit_text or None


def _parse_unit_value(value: Any, *, expected_unit: str) -> tuple[float | None, str, str | None]:
    """Parse a limit without interpreting raw/hex values as a measured unit."""
    explicit_unit: str | None = None
    conversion_unit: str | None = None
    conversion_proven = False
    if isinstance(value, Mapping):
        value, explicit_unit, conversion_unit, conversion_proven = _mapping_limit_value(value)
        if value is _MISSING:
            return None, "missing", "limit value is missing"
    if isinstance(value, bool) or value is None:
        return None, "missing", "limit value is not numeric"
    text = str(value).strip()
    if text.lower().startswith(("0x", "+0x", "-0x")):
        return None, "unknown", "raw hexadecimal limit packing has no bound unit provenance"
    split = _split_numeric_unit(text)
    if split is None:
        return None, "unknown", "limit value is not a decimal number with a known unit"
    number = _finite(split[0], field="limit value")
    embedded_unit = split[1]
    unit = explicit_unit or embedded_unit
    if unit is None:
        return number, "source_field", None
    normalized = str(unit).replace("°", "deg").casefold()
    expected = expected_unit.replace("°", "deg").casefold()
    if normalized == expected:
        return number, "explicit", None
    if conversion_proven and str(conversion_unit or "").casefold() == expected:
        return number, "converted", None
    return None, "incompatible", f"limit unit {unit!r} is not comparable with {expected_unit}"


def _limit_value(
    source: Mapping[str, Any], aliases: Sequence[str], *, expected_unit: str
) -> tuple[float | None, str | None, str, str | None]:
    for alias in aliases:
        if alias not in source:
            continue
        try:
            value, unit_status, error = _parse_unit_value(
                source[alias], expected_unit=expected_unit
            )
        except AnalysisError as exc:
            return None, alias, "unknown", str(exc)
        return value, alias, unit_status, error
    return None, None, "missing", None


def _extract_limits(
    snapshot: Mapping[str, Any] | None,
    spec: _PlanSpec,
) -> tuple[dict[str, Any], set[str], list[str]]:
    device: Mapping[str, Any] | None = None
    errors: list[str] = []
    if snapshot is not None:
        try:
            device = _selected_snapshot_device(
                snapshot, board_id=spec.board_serial, bus_id=spec.pci_bus_id
            )
        except AnalysisError as exc:
            errors.append(str(exc))
    source = device.get("limits") if isinstance(device, Mapping) else None
    if not isinstance(source, Mapping):
        errors.append("pre-run snapshot limits are unavailable")
        source = {}
    result: dict[str, Any] = {}
    missing: set[str] = set()
    for name, aliases in LIMIT_ALIASES.items():
        expected_unit = LIMIT_UNITS[name]
        value, alias, unit_status, error = _limit_value(
            source, aliases, expected_unit=expected_unit
        )
        if value is None:
            missing.add(name)
        if error:
            errors.append(f"{name}: {error}")
        result[name] = {
            "value": value,
            "unit": expected_unit,
            "source": f"pre_snapshot.limits.{alias}" if alias else None,
            "source_unit_status": unit_status,
            "expected_plan_value": spec.expected_limits[name],
            "status": "available" if value is not None else "not_evaluable",
            "differs_from_plan": (
                value is not None
                and not math.isclose(value, spec.expected_limits[name], rel_tol=0.0, abs_tol=1e-12)
            ),
        }
    return result, missing, list(dict.fromkeys(errors))


def _left_held_segments(
    rows: Sequence[Mapping[str, Any]], *, start_ns: int, end_ns: int
) -> list[tuple[Mapping[str, Any], int, int]]:
    segments: list[tuple[Mapping[str, Any], int, int]] = []
    if start_ns >= end_ns:
        return segments
    for index, row in enumerate(rows):
        timestamp = int(row["timestamp_monotonic_ns"])
        next_timestamp = (
            int(rows[index + 1]["timestamp_monotonic_ns"])
            if index + 1 < len(rows)
            else end_ns
        )
        segment_start = max(timestamp, start_ns)
        segment_end = min(next_timestamp, end_ns)
        if segment_end > segment_start:
            segments.append((row, segment_start, segment_end))
    return segments


def _aggregate_segments(
    segments: Sequence[tuple[Mapping[str, Any], int, int]], field: str
) -> dict[str, Any] | None:
    values = [float(row[field]) for row, _start, _end in segments if field in row]
    if len(values) != len(segments) or not values:
        return None
    duration_ns = sum(end - start for _row, start, end in segments)
    if duration_ns <= 0:
        return None
    weighted = math.fsum(
        float(row[field]) * ((end - start) / duration_ns)
        for row, start, end in segments
    )
    return {
        "unit": SIGNAL_UNITS[field],
        "min": min(values),
        "max": max(values),
        "time_weighted_mean": weighted,
        "duration_weighted": weighted,
        "duration_ns": duration_ns,
        "segment_count": len(segments),
    }


def _gddr_diagnostic(
    rows: Sequence[Mapping[str, Any]], source_established: bool
) -> dict[str, Any]:
    raw_fields: dict[str, list[Any]] = {field: [] for field in GDDR_FIELDS}
    max_raw: list[Any] = []
    for row in rows:
        for field in GDDR_FIELDS:
            if field in row:
                raw_fields[field].append(row[field])
        if "MAX_GDDR_TEMP" in row:
            max_raw.append(row["MAX_GDDR_TEMP"])
    return {
        "status": "decoded" if source_established else "not_evaluable",
        "raw_fields": raw_fields,
        "max_gddr_temp_raw": max_raw,
    }


def _decode_gddr_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]] | None, str | None]:
    try:
        decoded = {
            id(row): power_trace_issue137.decode_issue137_gddr_row(
                row, source_established=True
            )
            for row in rows
        }
    except (TypeError, ValueError, KeyError) as exc:
        return None, f"GDDR die decode failed: {type(exc).__name__}"
    return decoded, None


def _gddr_channel_values(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    decoded_by_row: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    channels: dict[str, list[dict[str, Any]]] = {}
    for row, start, end in segments:
        decoded = decoded_by_row.get(id(row), {})
        for item in decoded.values():
            channel_values = item.get("channels_c") if isinstance(item, Mapping) else None
            if not isinstance(channel_values, Mapping):
                continue
            for channel, die_values in channel_values.items():
                if isinstance(die_values, Mapping):
                    channels.setdefault(channel, []).append(
                        {
                            "bottom_c": die_values.get("bottom_c"),
                            "top_c": die_values.get("top_c"),
                            "duration_ns": end - start,
                        }
                    )
    return channels


def _gddr_channel_aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = [int(item["duration_ns"]) for item in values]
    total = sum(durations)
    bottom = [float(item["bottom_c"]) for item in values]
    top = [float(item["top_c"]) for item in values]
    return {
        "unit": "degC",
        "bottom_c": {
            "min": min(bottom),
            "max": max(bottom),
            "time_weighted_mean": math.fsum(
                value * duration / total
                for value, duration in zip(bottom, durations, strict=True)
            ),
        },
        "top_c": {
            "min": min(top),
            "max": max(top),
            "time_weighted_mean": math.fsum(
                value * duration / total
                for value, duration in zip(top, durations, strict=True)
            ),
        },
        "duration_ns": total,
        "die_pair_diagnostic": {
            "sample_count": len(values),
            "bottom_top_values_retained": True,
            "not_a_packed_scalar": True,
        },
    }


def _aggregate_gddr(
    rows: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    source_established: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic = _gddr_diagnostic(rows, source_established)
    if not source_established:
        diagnostic["reason"] = "GDDR source unit/packing provenance is not established"
        return {}, diagnostic
    decoded_by_row, error = _decode_gddr_rows(rows)
    if error is not None or decoded_by_row is None:
        diagnostic.update(status="not_evaluable", reason=error)
        return {}, diagnostic
    channels = _gddr_channel_values(segments, decoded_by_row)
    aggregates = {
        channel: _gddr_channel_aggregate(values) for channel, values in channels.items()
    }
    diagnostic["channel_count"] = len(aggregates)
    return aggregates, diagnostic


def _trace_expected_bounds(
    load: _TraceLoad,
    spec: _PlanSpec,
    expected_start_ns: int | None,
    expected_end_ns: int | None,
) -> tuple[int | None, int | None, int | None, int | None, set[str]]:
    missing = set(load.missing_fields)
    start_ns, end_ns = load.bounds_start_ns, load.bounds_end_ns
    if expected_start_ns is not None and start_ns != expected_start_ns:
        load.errors.append("trace start bound does not match first launch start")
        missing.add("trace_start_binding")
    if expected_end_ns is not None and end_ns != expected_end_ns:
        load.errors.append("trace end bound does not match target workload end")
        missing.add("trace_end_binding")
    if start_ns is None or end_ns is None:
        return start_ns, end_ns, None, None, missing
    plateau_start_ns = start_ns + round(spec.plateau_start_seconds * 1_000_000_000.0)
    plateau_end_ns = start_ns + round(spec.plateau_end_seconds * 1_000_000_000.0)
    return start_ns, end_ns, plateau_start_ns, plateau_end_ns, missing


def _interval_coverage(
    timed_rows: Sequence[Mapping[str, Any]],
    plateau_start_ns: int,
    plateau_end_ns: int,
    missing: set[str],
) -> tuple[list[tuple[Mapping[str, Any], int, int]], str]:
    first_timestamp = timed_rows[0].get("timestamp_monotonic_ns") if timed_rows else None
    last_timestamp = timed_rows[-1].get("timestamp_monotonic_ns") if timed_rows else None
    if not isinstance(first_timestamp, int) or first_timestamp > plateau_start_ns:
        missing.add("plateau_start_coverage")
    if not isinstance(last_timestamp, int) or last_timestamp < plateau_end_ns:
        missing.add("plateau_terminal_coverage")
    if not isinstance(first_timestamp, int) or not isinstance(last_timestamp, int):
        return [], "not_evaluable"
    segments = _left_held_segments(timed_rows, start_ns=plateau_start_ns, end_ns=plateau_end_ns)
    if sum(end - start for _row, start, end in segments) != plateau_end_ns - plateau_start_ns:
        missing.add("plateau_coverage")
        return segments, "not_evaluable"
    if missing.intersection({"plateau_start_coverage", "plateau_terminal_coverage"}):
        return segments, "not_evaluable"
    return segments, "complete"


def _row_coverage(
    rows: Sequence[Mapping[str, Any]],
    plateau_start_ns: int | None,
    plateau_end_ns: int | None,
    missing: set[str],
) -> tuple[list[tuple[Mapping[str, Any], int, int]], str]:
    timed_rows = [row for row in rows if isinstance(row.get("timestamp_monotonic_ns"), int)]
    if not rows:
        missing.add("trace_samples")
        return [], "not_evaluable"
    if any(
        int(left["timestamp_monotonic_ns"]) > int(right["timestamp_monotonic_ns"])
        for left, right in itertools.pairwise(timed_rows)
    ):
        missing.add("trace_timestamp_order")
        return [], "not_evaluable"
    if plateau_start_ns is None or plateau_end_ns is None:
        missing.update({"run_start_monotonic_ns", "run_end_monotonic_ns"})
        return [], "not_evaluable"
    return _interval_coverage(timed_rows, plateau_start_ns, plateau_end_ns, missing)


def _trace_coverage(
    load: _TraceLoad,
    spec: _PlanSpec,
    *,
    expected_start_ns: int | None,
    expected_end_ns: int | None,
) -> tuple[
    int | None,
    int | None,
    int | None,
    int | None,
    list[tuple[Mapping[str, Any], int, int]],
    str,
    set[str],
]:
    start_ns, end_ns, plateau_start_ns, plateau_end_ns, missing = _trace_expected_bounds(
        load, spec, expected_start_ns, expected_end_ns
    )
    segments, coverage_status = _row_coverage(
        load.rows, plateau_start_ns, plateau_end_ns, missing
    )
    return start_ns, end_ns, plateau_start_ns, plateau_end_ns, segments, coverage_status, missing


def _trace_statistics(
    rows: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    load: _TraceLoad,
) -> tuple[dict[str, Any], set[str]]:
    missing: set[str] = set()
    statistics: dict[str, Any] = {}
    for field in SCALAR_FIELDS:
        aggregate = _aggregate_segments(segments, field)
        if aggregate is None:
            if rows and any(field not in row for row in rows):
                missing.add(field)
        else:
            statistics[field] = aggregate
    gddr_statistics, gddr_exploratory = _aggregate_gddr(
        rows, segments, source_established=load.source_established
    )
    statistics["gddr_channels"] = gddr_statistics
    statistics["gddr_exploratory"] = gddr_exploratory
    return statistics, missing


def _trace_analysis(
    load: _TraceLoad,
    spec: _PlanSpec,
    *,
    expected_start_ns: int | None,
    expected_end_ns: int | None,
) -> tuple[dict[str, Any], set[str]]:
    (
        start_ns,
        end_ns,
        plateau_start_ns,
        plateau_end_ns,
        segments,
        coverage_status,
        missing,
    ) = _trace_coverage(
        load, spec, expected_start_ns=expected_start_ns, expected_end_ns=expected_end_ns
    )
    statistics, statistic_missing = _trace_statistics(load.rows, segments, load)
    missing.update(statistic_missing)
    return {
        "coverage": {
            "status": coverage_status,
            "run_start_monotonic_ns": start_ns,
            "run_end_monotonic_ns": end_ns,
            "plateau_start_monotonic_ns": plateau_start_ns,
            "plateau_end_monotonic_ns": plateau_end_ns,
            "terminal_coverage_required": True,
            "no_extrapolation": True,
            "segment_count": len(segments),
        },
        "statistics": statistics,
        "trace_rows": [dict(row) for row in load.rows],
        "reader_facts": load.reader_facts,
        "exploratory": {"gddr": statistics["gddr_exploratory"]},
    }, missing


def _observed_source_versions_verified(
    record: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    snapshot: Mapping[str, Any] | None,
) -> bool:
    candidates: list[Any] = []
    environment = record.get("environment")
    if isinstance(environment, Mapping):
        candidates.append(environment.get("host_sw_vers"))
    if isinstance(result, Mapping) and isinstance(result.get("environment"), Mapping):
        candidates.append(result["environment"].get("host_sw_vers"))
    if isinstance(snapshot, Mapping):
        candidates.append(snapshot.get("host_sw_vers"))
    observed = [candidate for candidate in candidates if isinstance(candidate, Mapping)]
    return bool(observed) and all(
        power_trace_issue137.source_versions_verified(candidate) for candidate in observed
    )


def _identity_from_environment(environment: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    board = environment.get("board")
    if not isinstance(board, Mapping):
        board = environment.get("board_info")
    if not isinstance(board, Mapping):
        return None, None, None
    identity = board.get("identity") if isinstance(board.get("identity"), Mapping) else {}
    serial = board.get("serial") or board.get("board_id") or identity.get("board_id")
    board_id = board.get("board_id") or identity.get("board_id") or serial
    bus_id = board.get("bus_id") or board.get("pci_bus_id") or identity.get("bus_id")
    return (
        str(serial) if isinstance(serial, str) and serial else None,
        str(board_id) if isinstance(board_id, str) and board_id else None,
        str(bus_id) if isinstance(bus_id, str) and bus_id else None,
    )


def _environment_required(environment: Mapping[str, Any]) -> dict[str, Any]:
    board = environment.get("board")
    identity = board.get("identity") if isinstance(board, Mapping) else None
    firmware = environment.get("firmware")
    return {
        "board.serial": board.get("serial") if isinstance(board, Mapping) else None,
        "board.identity.board_id": identity.get("board_id") if isinstance(identity, Mapping) else None,
        "board.identity.bus_id": identity.get("bus_id") if isinstance(identity, Mapping) else None,
        "firmware.bundle": firmware.get("bundle") if isinstance(firmware, Mapping) else None,
        "kernel": environment.get("kernel"),
        "kernel_driver_version": environment.get("kernel_driver_version"),
        "image_digest": environment.get("image_digest"),
        "harness_commit": environment.get("harness_commit"),
        "device_node": environment.get("device_node"),
        "worker_core_count": environment.get("worker_core_count"),
        "worker_core_coordinates": environment.get("worker_core_coordinates"),
        "total_blocks": environment.get("total_blocks"),
    }


def _environment_binding_errors(
    environment: Mapping[str, Any],
    configuration: Mapping[str, Any] | None,
    spec: _PlanSpec,
) -> list[str]:
    required = _environment_required(environment)
    errors = [
        field
        for field in spec.required_environment_fields
        if field not in required or required[field] is None or required[field] == ""
    ]
    if required["board.serial"] != spec.board_serial:
        errors.append("board.serial.binding")
    if required["board.identity.board_id"] != spec.board_serial:
        errors.append("board.identity.board_id.binding")
    if required["board.identity.bus_id"] != spec.pci_bus_id:
        errors.append("board.identity.bus_id.binding")
    if required["device_node"] != spec.device_node:
        errors.append("device_node.binding")
    if configuration is not None:
        expected_environment = {
            "worker_core_count": configuration.get("worker_core_count"),
            "worker_core_coordinates": configuration.get("worker_core_coordinates"),
            "total_blocks": configuration.get("total_blocks"),
            "core_block_distribution": configuration.get("block_distribution"),
        }
        errors.extend(
            f"environment.{field}.binding"
            for field, expected in expected_environment.items()
            if environment.get(field) != expected
        )
    return errors


def _raw_environment_errors(
    raw_environment: Mapping[str, Any], spec: _PlanSpec
) -> list[str]:
    errors: list[str] = []
    if _identity_from_environment(raw_environment) != (
        spec.board_serial,
        spec.board_serial,
        spec.pci_bus_id,
    ):
        errors.append("runner_result.environment.identity.binding")
    raw_firmware = raw_environment.get("firmware") or raw_environment.get("firmwares")
    raw_required = {
        "firmware": raw_firmware.get("bundle") or raw_firmware.get("fw_bundle_version")
        if isinstance(raw_firmware, Mapping)
        else None,
        "kernel": raw_environment.get("kernel"),
        "kernel_driver_version": raw_environment.get("kernel_driver_version")
        or raw_environment.get("kmd_version"),
        "image_digest": raw_environment.get("image_digest") or raw_environment.get("image"),
        "harness_commit": raw_environment.get("harness_commit"),
        "device_node": raw_environment.get("device_node"),
    }
    errors.extend(
        f"runner_result.environment.{field}"
        for field, value in raw_required.items()
        if value is None or value == ""
    )
    if raw_environment.get("device_node") != spec.device_node:
        errors.append("runner_result.environment.device_node.binding")
    return errors


def _canonical_environment_errors(
    environment: Mapping[str, Any],
    raw_environment: Mapping[str, Any],
    configuration: Mapping[str, Any] | None,
    spec: _PlanSpec,
) -> list[str]:
    try:
        from tools import issue137 as runner

        raw_configuration_id = configuration.get("id") if configuration is not None else None
        canonical_configuration = runner.configuration(spec.plan, raw_configuration_id)
        canonical_environment = runner.build_environment_record(raw_environment, canonical_configuration)
        persisted_fields = (
            "board",
            "firmware",
            "kernel",
            "kernel_driver_version",
            "image_digest",
            "harness_commit",
            "device_node",
            "worker_core_count",
            "worker_core_coordinates",
            "total_blocks",
            "core_block_distribution",
        )
        return [
            f"runner_result.environment.{field}.binding"
            for field in persisted_fields
            if environment.get(field) != canonical_environment.get(field)
        ]
    except (ImportError, KeyError, TypeError, ValueError):
        return ["runner_result.environment.canonical"]


def _runner_environment_binding_errors(
    environment: Mapping[str, Any],
    result: Mapping[str, Any],
    configuration: Mapping[str, Any] | None,
    spec: _PlanSpec,
) -> list[str]:
    raw_environment = result.get("environment")
    if not isinstance(raw_environment, Mapping):
        return ["runner_result.environment"]
    errors = _raw_environment_errors(raw_environment, spec)
    errors.extend(_canonical_environment_errors(environment, raw_environment, configuration, spec))
    return errors


def _environment_errors(
    record: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    configuration: Mapping[str, Any] | None,
    spec: _PlanSpec,
) -> list[str]:
    environment = record.get("environment")
    if not isinstance(environment, Mapping):
        return ["environment"]
    errors = _environment_binding_errors(environment, configuration, spec)
    if isinstance(result, Mapping):
        errors.extend(
            _runner_environment_binding_errors(environment, result, configuration, spec)
        )
    return list(dict.fromkeys(errors))


def _duration_target_guard(spec: _PlanSpec, mode: str) -> tuple[float, float]:
    if mode == "long":
        return spec.long_target_seconds, spec.long_guard_seconds
    return spec.smoke_target_seconds, spec.smoke_guard_seconds


def _record_duration_errors(record: Mapping[str, Any], spec: _PlanSpec) -> list[str]:
    checks = (
        (record.get("record_schema") != ISSUE137_RECORD_SCHEMA, "record_schema"),
        (record.get("record_revision") != 2, "record_revision"),
        (record.get("plan_sha256") != spec.plan_sha256, "plan_sha256"),
        (record.get("status") != "ok", "run_status"),
        (record.get("workload_started") is not True, "workload_started"),
    )
    return [name for failed, name in checks if failed]


def _result_shape_errors(
    run: Mapping[str, Any], result: Mapping[str, Any], mode: str, target: float, guard: float
) -> list[str]:
    errors: list[str] = []
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        errors.append("run.run_id")
    for field, expected in (
        ("record_schema", ISSUE137_DEVICE_RESULT_SCHEMA),
        ("status", "ok"),
        ("mode", mode),
        ("load_mode", ISSUE137_LOAD_MODE),
    ):
        if result.get(field) != expected:
            errors.append(f"runner_result.{field}" if field != "load_mode" else "load_mode")
    timing = run.get("timing")
    if not isinstance(timing, Mapping):
        errors.append("run.timing")
    elif timing.get("load_seconds") != target or timing.get("guard_seconds") != guard:
        errors.append("run.timing.binding")
    return errors


def _result_provenance_errors(
    record: Mapping[str, Any], result: Mapping[str, Any], spec: _PlanSpec
) -> list[str]:
    errors: list[str] = []
    if result.get("plan_sha256") != spec.plan_sha256:
        errors.append("runner_result.plan_sha256")
    if not isinstance(result.get("run_id"), str) or not result["run_id"]:
        errors.append("runner_result.run_id")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance")
        return errors
    if provenance.get("wrapper_run_id") != result.get("run_id"):
        errors.append("provenance.wrapper_run_id")
    path = provenance.get("runner_result_path")
    if not isinstance(path, str) or Path(path).name != path:
        errors.append("provenance.runner_result_path")
    return errors


def _result_header_errors(
    record: Mapping[str, Any],
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    spec: _PlanSpec,
    mode: str,
    target: float,
    guard: float,
) -> list[str]:
    errors = _result_shape_errors(run, result, mode, target, guard)
    errors.extend(_result_provenance_errors(record, result, spec))
    return errors


def _result_scalar_errors(
    result: Mapping[str, Any], spec: _PlanSpec, target: float, guard: float
) -> list[str]:
    errors: list[str] = []
    for field, expected in (
        ("target_seconds", target),
        ("guard_seconds", guard),
        ("overrun_limit_seconds", spec.overrun_limit_seconds),
    ):
        try:
            if not math.isclose(_finite(result.get(field), field=field), expected, rel_tol=0.0, abs_tol=1e-12):
                errors.append(f"runner_result.{field}")
        except AnalysisError:
            errors.append(f"runner_result.{field}")
    return errors


def _result_launch_count(result: Mapping[str, Any]) -> tuple[int | None, list[str]]:
    try:
        count = _integer(result.get("launch_count"), field="launch_count")
        return count, ["launch_count"] if count <= 0 else []
    except AnalysisError:
        return None, ["launch_count"]


def _result_monotonic_bounds(
    result: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, list[str]]:
    try:
        first = _integer(result.get("first_launch_start_monotonic_ns"), field="first launch")
        workload_end = _integer(result.get("workload_end_monotonic_ns"), field="workload end")
        last = _integer(result.get("last_launch_end_monotonic_ns"), field="last launch")
        return first, workload_end, last, []
    except AnalysisError:
        return None, None, None, ["monotonic_bounds"]


def _result_overrun_errors(
    result: Mapping[str, Any],
    spec: _PlanSpec,
    first: int,
    workload_end: int,
    last: int,
    target: float,
    guard: float,
) -> list[str]:
    errors: list[str] = []
    expected_end = first + round(target * 1_000_000_000.0)
    if workload_end != expected_end:
        errors.append("workload_end_monotonic_ns")
    if last < workload_end:
        errors.append("last_launch_end_monotonic_ns")
    if last > first + round(guard * 1_000_000_000.0):
        errors.append("guard_seconds")
    try:
        overrun = _finite(result.get("overrun_seconds"), field="overrun_seconds")
        expected_overrun = (last - workload_end) / 1_000_000_000.0
        if not math.isclose(overrun, expected_overrun, rel_tol=0.0, abs_tol=1e-9):
            errors.append("overrun_seconds.binding")
        if overrun > spec.overrun_limit_seconds:
            errors.append("overrun_seconds.limit")
    except AnalysisError:
        errors.append("overrun_seconds")
    return errors


def _result_duration_values(
    result: Mapping[str, Any], spec: _PlanSpec, target: float, guard: float
) -> tuple[int | None, int | None, int | None, int | None, list[str]]:
    count, count_errors = _result_launch_count(result)
    first, workload_end, last, bound_errors = _result_monotonic_bounds(result)
    errors = count_errors + bound_errors
    if first is None or workload_end is None or last is None:
        return None, None, None, None, errors
    errors.extend(_result_overrun_errors(result, spec, first, workload_end, last, target, guard))
    return count, first, workload_end, last, errors


def _configuration_duration_errors(
    result: Mapping[str, Any] | None,
    configuration: Mapping[str, Any] | None,
    configuration_id: str | None,
    spec: _PlanSpec,
) -> list[str]:
    errors: list[str] = []
    if configuration is None:
        errors.append("configuration")
    elif configuration != spec.expected_configurations.get(configuration_id or ""):
        errors.append("configuration.binding")
    if result is not None and configuration is not None and result.get("core_plan") != configuration.get("core_plan"):
        errors.append("runner_result.core_plan.binding")
    return errors


def _run_scalar_binding_errors(
    run: Mapping[str, Any], spec: _PlanSpec, target: float, guard: float
) -> list[str]:
    errors: list[str] = []
    for field, expected in (
        ("load_mode", ISSUE137_LOAD_MODE),
        ("target_seconds", target),
        ("guard_seconds", guard),
        ("overrun_limit_seconds", spec.overrun_limit_seconds),
    ):
        actual = run.get(field)
        try:
            matches = actual == expected or math.isclose(
                _finite(actual, field=field), expected, rel_tol=0.0, abs_tol=1e-12
            )
        except AnalysisError:
            matches = False
        if not matches:
            errors.append(f"run.{field}")
    return errors


def _run_result_binding_errors(
    record: Mapping[str, Any],
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    first: int | None,
    workload_end: int | None,
    last: int | None,
) -> list[str]:
    errors: list[str] = []
    for run_field, result_field in (
        ("launch_count", "launch_count"),
        ("workload_start_monotonic_ns", "first_launch_start_monotonic_ns"),
        ("workload_end_monotonic_ns", "workload_end_monotonic_ns"),
        ("last_launch_end_monotonic_ns", "last_launch_end_monotonic_ns"),
        ("overrun_seconds", "overrun_seconds"),
    ):
        if run.get(run_field) != result.get(result_field):
            errors.append(f"run.{run_field}.binding")
    for field, expected in (
        ("workload_start_monotonic_ns", first),
        ("workload_end_monotonic_ns", workload_end),
        ("workload_actual_end_monotonic_ns", last),
        ("run_end_monotonic_ns", workload_end),
    ):
        if field in record and record.get(field) != expected:
            errors.append(f"{field}.binding")
    return errors


def _forbidden_duration_inputs(
    record: Mapping[str, Any], run: Mapping[str, Any], result: Mapping[str, Any] | None
) -> list[str]:
    if any(
        key in container
        for container in (record, run, result or {})
        for key in ("calibration", "calibration_name", "repeat_count", "repeat_counts")
    ):
        return ["calibration_or_repeat_count"]
    return []


def _trace_binding_errors(
    trace: Any, first: int | None, workload_end: int | None, last: int | None
) -> list[str]:
    if not isinstance(trace, Mapping):
        return ["trace"]
    errors: list[str] = []
    checks = (
        (trace.get("schema") not in {None, power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA}, "trace.schema"),
        (trace.get("run_start_monotonic_ns") not in {None, first}, "trace.run_start_monotonic_ns"),
        (trace.get("run_end_monotonic_ns") not in {None, workload_end}, "trace.run_end_monotonic_ns"),
        (trace.get("actual_end_monotonic_ns") not in {None, last}, "trace.actual_end_monotonic_ns"),
        (not isinstance(trace.get("path"), str) or Path(trace["path"]).name != trace["path"], "trace.path"),
    )
    errors.extend(name for failed, name in checks if failed)
    return errors


def _run_duration_errors(
    record: Mapping[str, Any],
    run: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    spec: _PlanSpec,
    configuration: Mapping[str, Any] | None,
    configuration_id: str | None,
    target: float,
    guard: float,
    first: int | None,
    workload_end: int | None,
    last: int | None,
) -> list[str]:
    errors = _configuration_duration_errors(result, configuration, configuration_id, spec)
    errors.extend(_run_scalar_binding_errors(run, spec, target, guard))
    if result is not None:
        errors.extend(_run_result_binding_errors(record, run, result, first, workload_end, last))
    errors.extend(_forbidden_duration_inputs(record, run, result))
    errors.extend(_trace_binding_errors(record.get("trace"), first, workload_end, last))
    return errors


def _duration_binding(
    record: Mapping[str, Any],
    spec: _PlanSpec,
    configuration: Mapping[str, Any] | None,
    configuration_id: str | None,
    mode: str,
) -> tuple[dict[str, Any], list[str], Mapping[str, Any] | None]:
    run = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    result_value = record.get("runner_result")
    result = result_value if isinstance(result_value, Mapping) else None
    target, guard = _duration_target_guard(spec, mode)
    errors = _record_duration_errors(record, spec)
    count = first = workload_end = last = None
    if result is None:
        errors.append("runner_result")
    else:
        errors.extend(_result_header_errors(record, run, result, spec, mode, target, guard))
        errors.extend(_result_scalar_errors(result, spec, target, guard))
        count, first, workload_end, last, value_errors = _result_duration_values(
            result, spec, target, guard
        )
        errors.extend(value_errors)
    errors.extend(
        _run_duration_errors(
            record,
            run,
            result,
            spec,
            configuration,
            configuration_id,
            target,
            guard,
            first,
            workload_end,
            last,
        )
    )
    return {
        "target_seconds": target,
        "guard_seconds": guard,
        "launch_count": count,
        "workload_start_monotonic_ns": first,
        "workload_end_monotonic_ns": workload_end,
        "last_launch_end_monotonic_ns": last,
    }, list(dict.fromkeys(errors)), result


def _run_identity(record: Mapping[str, Any]) -> tuple[str | None, int | None, str]:
    run = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    configuration = record.get("configuration") if isinstance(record.get("configuration"), Mapping) else {}
    configuration_id = configuration.get("id", record.get("configuration_id"))
    configuration_id = configuration_id if isinstance(configuration_id, str) else None
    run_number_raw = run.get("run_number", record.get("run_number"))
    try:
        run_number = None if run_number_raw is None else _integer(run_number_raw, field="run_number")
    except AnalysisError:
        run_number = None
    mode = run.get("mode", record.get("mode"))
    return configuration_id, run_number, mode if isinstance(mode, str) else "unknown"


def _verdict_trace_missing(field: str) -> bool:
    return field in VERDICT_SIGNAL_FIELDS or field.startswith(
        ("run_", "plateau_", "trace", "timestamp")
    )


def _aiclk_fact(
    statistics: Mapping[str, Any], limits: Mapping[str, Any], spec: _PlanSpec
) -> tuple[bool | None, bool | None]:
    aggregate = statistics.get("aiclk_mhz")
    limit_info = limits.get("asic_fmax_mhz")
    if not isinstance(aggregate, Mapping) or not isinstance(limit_info, Mapping):
        return None, None
    if limit_info.get("value") is None or limit_info.get("source_unit_status") not in {"source_field", "explicit", "converted"}:
        return None, None
    mean = aggregate.get("time_weighted_mean")
    if mean is None:
        return None, None
    depressed = float(mean) < spec.aiclk_fraction * float(limit_info["value"])
    return depressed, not depressed


def _pinned_fact(
    hypothesis: str,
    statistics: Mapping[str, Any],
    limits: Mapping[str, Any],
    spec: _PlanSpec,
) -> bool | None:
    signal, limit_name = HYPOTHESIS_SIGNALS[hypothesis]
    aggregate = statistics.get(signal)
    limit_info = limits.get(limit_name)
    if not isinstance(aggregate, Mapping) or not isinstance(limit_info, Mapping):
        return None
    limit = limit_info.get("value")
    mean = aggregate.get("time_weighted_mean")
    if limit is None or mean is None or limit_info.get("source_unit_status") not in {"source_field", "explicit", "converted"}:
        return None
    if hypothesis == "H-THM":
        return float(mean) >= spec.thermal_pin_fraction * float(limit)
    lower = (1.0 - spec.tolerance_fraction) * float(limit)
    upper = (1.0 + spec.tolerance_fraction) * float(limit)
    return lower <= float(mean) <= upper


def _expected_verdict_candidate(
    mode: str,
    role: Mapping[str, Any],
    run_number: int | None,
    spec: _PlanSpec,
) -> bool:
    return (
        mode == "long"
        and role.get("verdict_bearing") is True
        and run_number in range(1, len(spec.order) + 1)
    )


def _append_snapshot_gddr_raw(
    trace_report: dict[str, Any], snapshot: Mapping[str, Any] | None, spec: _PlanSpec
) -> None:
    gddr_exploratory = trace_report["exploratory"].get("gddr")
    if not isinstance(gddr_exploratory, dict) or snapshot is None:
        return
    try:
        selected = _selected_snapshot_device(
            snapshot, board_id=spec.board_serial, bus_id=spec.pci_bus_id
        )
    except AnalysisError:
        selected = None
    smbus = selected.get("smbus_telem") if isinstance(selected, Mapping) else None
    if isinstance(smbus, Mapping) and "MAX_GDDR_TEMP" in smbus:
        gddr_exploratory.setdefault("max_gddr_temp_raw", []).append(smbus["MAX_GDDR_TEMP"])


def _base_verdict_missing(
    binding_errors: Sequence[str],
    environment_errors: Sequence[str],
    snapshot_errors: Sequence[str],
    missing_limits: set[str],
    missing_trace: set[str],
    load: _TraceLoad,
) -> set[str]:
    missing = set(binding_errors) | set(environment_errors) | set(snapshot_errors) | set(missing_limits)
    missing.update(field for field in missing_trace if _verdict_trace_missing(field))
    trace_requires_verdict = not load.missing_fields or any(
        _verdict_trace_missing(field) for field in load.missing_fields
    )
    if load.errors and trace_requires_verdict:
        missing.add("trace")
    return missing


def _signal_verdict_missing(
    statistics: Mapping[str, Any],
    limits: Mapping[str, Any],
    pinned: Mapping[str, bool | None],
    aiclk_depressed: bool | None,
) -> set[str]:
    missing: set[str] = set()
    if aiclk_depressed is None:
        missing.add("aiclk_mhz")
        if limits.get("asic_fmax_mhz", {}).get("value") is None:
            missing.add("asic_fmax_mhz")
    for hypothesis, (signal, limit_name) in HYPOTHESIS_SIGNALS.items():
        if pinned[hypothesis] is None:
            if statistics.get(signal, {}).get("time_weighted_mean") is None:
                missing.add(signal)
            if limits.get(limit_name, {}).get("value") is None:
                missing.add(limit_name)
    return missing


def _verdict_missing_fields(
    expected_candidate: bool,
    binding_errors: Sequence[str],
    environment_errors: Sequence[str],
    snapshot_errors: Sequence[str],
    missing_limits: set[str],
    missing_trace: set[str],
    load: _TraceLoad,
    statistics: Mapping[str, Any],
    limits: Mapping[str, Any],
    pinned: Mapping[str, bool | None],
    aiclk_depressed: bool | None,
) -> set[str]:
    if not expected_candidate:
        return set()
    missing = _base_verdict_missing(
        binding_errors,
        environment_errors,
        snapshot_errors,
        missing_limits,
        missing_trace,
        load,
    )
    missing.update(_signal_verdict_missing(statistics, limits, pinned, aiclk_depressed))
    return missing


def _run_report(
    record: Mapping[str, Any],
    name: str,
    run_number: int | None,
    configuration_id: str | None,
    mode: str,
    expected_candidate: bool,
    verdict_missing: set[str],
    spec: _PlanSpec,
    load: _TraceLoad,
    snapshot_name: str | None,
    limits: Mapping[str, Any],
    statistics: Mapping[str, Any],
    aiclk_depressed: bool | None,
    aiclk_not_depressed: bool | None,
    pinned: Mapping[str, bool | None],
    trace_report: Mapping[str, Any],
    binding_errors: Sequence[str],
    environment_errors: Sequence[str],
    limit_errors: Sequence[str],
    snapshot_errors: Sequence[str],
) -> dict[str, Any]:
    exploratory_missing = (
        set(verdict_missing) | set(load.missing_fields) | set(load.errors) | set(limit_errors)
        if expected_candidate
        else set(load.missing_fields) | set(load.errors) | set(limit_errors)
    )
    return {
        "run_id": _safe_name(_record_value(record, ("run", "run_id")) or name),
        "run_number": run_number,
        "configuration_id": configuration_id,
        "mode": mode,
        "status": record.get("status"),
        "verdict_evidence": record.get("verdict_evidence") is True,
        "diagnostic_only": not expected_candidate,
        "eligible_for_verdict": expected_candidate and not verdict_missing,
        "expected_order_configuration": (
            spec.order[run_number - 1] if run_number in range(1, len(spec.order) + 1) else None
        ),
        "source": {
            "record": name,
            "trace": load.source_name,
            "trace_sha256": load.source_sha256,
            "trace_byte_count": load.source_byte_count,
            "pre_run_snapshot": snapshot_name,
        },
        "limits": limits,
        "statistics": statistics,
        "facts": {
            "aiclk_depressed": aiclk_depressed,
            "aiclk_not_depressed": aiclk_not_depressed,
            "pinned": pinned,
        },
        "aiclk_depressed": aiclk_depressed,
        "aiclk_not_depressed": aiclk_not_depressed,
        "pinned": pinned,
        "coverage": trace_report["coverage"],
        "trace_rows": trace_report["trace_rows"],
        "reader_facts": trace_report["reader_facts"],
        "exploratory": {
            **trace_report["exploratory"],
            "load_errors": list(dict.fromkeys(load.errors + list(limit_errors) + list(snapshot_errors))),
            "binding_errors": list(dict.fromkeys(binding_errors + environment_errors)),
            "missing_fields": sorted(exploratory_missing),
        },
        "missing_fields": sorted(verdict_missing),
    }


def _run_analysis(record_input: _RecordInput, index: int, spec: _PlanSpec) -> _RunAnalysis:
    record = record_input.record
    name = _record_name(record_input, index)
    configuration_id, run_number, mode = _run_identity(record)
    role = spec.configuration_roles.get(configuration_id or "", {})
    expected_candidate = _expected_verdict_candidate(mode, role, run_number, spec)
    configuration = record.get("configuration") if isinstance(record.get("configuration"), Mapping) else None
    binding, binding_errors, result = _duration_binding(
        record, spec, configuration, configuration_id, mode
    )
    environment_errors = _environment_errors(record, result, configuration, spec)
    snapshot, snapshot_name, snapshot_errors = _pre_snapshot_mapping(
        record, base_dir=record_input.base_dir, spec=spec
    )
    limits, missing_limits, limit_errors = _extract_limits(snapshot, spec)
    source_verified = _observed_source_versions_verified(record, result, snapshot)
    load = _load_trace_source(
        record,
        base_dir=record_input.base_dir,
        record_name=name,
        source_verified=source_verified,
    )
    trace_report, missing_trace = _trace_analysis(
        load,
        spec,
        expected_start_ns=binding["workload_start_monotonic_ns"],
        expected_end_ns=binding["workload_end_monotonic_ns"],
    )
    _append_snapshot_gddr_raw(trace_report, snapshot, spec)
    statistics = trace_report["statistics"]
    aiclk_depressed, aiclk_not_depressed = _aiclk_fact(statistics, limits, spec)
    pinned = {
        hypothesis: _pinned_fact(hypothesis, statistics, limits, spec)
        for hypothesis in HYPOTHESIS_SIGNALS
    }
    verdict_missing = _verdict_missing_fields(
        expected_candidate,
        binding_errors,
        environment_errors,
        snapshot_errors,
        missing_limits,
        missing_trace,
        load,
        statistics,
        limits,
        pinned,
        aiclk_depressed,
    )
    report = _run_report(
        record,
        name,
        run_number,
        configuration_id,
        mode,
        expected_candidate,
        verdict_missing,
        spec,
        load,
        snapshot_name,
        limits,
        statistics,
        aiclk_depressed,
        aiclk_not_depressed,
        pinned,
        trace_report,
        binding_errors,
        environment_errors,
        limit_errors,
        snapshot_errors,
    )
    return _RunAnalysis(
        report=report,
        expected_candidate=expected_candidate,
        eligible=expected_candidate and not verdict_missing,
        verdict_position=run_number if expected_candidate else None,
    )


def _expected_verdict_positions(spec: _PlanSpec) -> set[int]:
    return {
        index
        for index, configuration_id in enumerate(spec.order, start=1)
        if configuration_id in spec.verdict_configurations
    }


def _position_run(
    item: _RunAnalysis,
    expected_positions: set[int],
    spec: _PlanSpec,
    by_position: dict[int, _RunAnalysis],
    missing: list[str],
    duplicates: list[str],
    extras: list[str],
) -> None:
    position = item.verdict_position
    actual_config = item.report.get("configuration_id")
    if position not in expected_positions:
        if item.expected_candidate:
            extras.append(
                f"{actual_config or '<missing-config>'}.run_{item.report.get('run_number') or '<missing-number>'}.ordered_evidence"
            )
        return
    expected_config = spec.order[position - 1]
    if actual_config != expected_config:
        missing.append(f"run_{position}.{expected_config}.ordered_evidence")
        extras.append(f"run_{position}.{actual_config or '<missing-config>'}.ordered_evidence")
        return
    if not item.eligible:
        missing.append(f"run_{position}.{expected_config}.ordered_evidence")
        missing.extend(
            f"run_{position}.{expected_config}.{field}"
            for field in item.report.get("missing_fields", [])
        )
        return
    if position in by_position:
        duplicates.append(f"run_{position}.{expected_config}.duplicate_evidence")
    else:
        by_position[position] = item


def _position_runs(
    run_analyses: Sequence[_RunAnalysis], spec: _PlanSpec
) -> tuple[dict[int, _RunAnalysis], list[str], list[str], list[str]]:
    expected_positions = _expected_verdict_positions(spec)
    by_position: dict[int, _RunAnalysis] = {}
    missing: list[str] = []
    duplicates: list[str] = []
    extras: list[str] = []
    for item in run_analyses:
        _position_run(item, expected_positions, spec, by_position, missing, duplicates, extras)
    for position in sorted(expected_positions):
        if position not in by_position:
            missing.append(f"run_{position}.{spec.order[position - 1]}.ordered_evidence")
    return by_position, missing, duplicates, extras


def _precondition_result(
    name: str,
    run_items: Sequence[_RunAnalysis],
    *,
    expected: str,
    field: str,
    expected_count: int,
) -> dict[str, Any]:
    values = [item.report["facts"].get(field) for item in run_items]
    if len(run_items) != expected_count or any(value is None for value in values):
        passed: bool | None = None
        status = "not_evaluable"
    else:
        passed = all(value is True for value in values)
        status = "pass" if passed else "fail"
    return {
        "name": name,
        "status": status,
        "pass": passed,
        "expected": expected,
        "runs": [
            {
                "run_number": item.report.get("run_number"),
                "configuration_id": item.report.get("configuration_id"),
                "value": item.report["facts"].get(field),
            }
            for item in run_items
        ],
    }


def _half_preconditions(
    by_position: Mapping[int, _RunAnalysis], spec: _PlanSpec, half: int
) -> dict[str, Any]:
    positions = tuple(range(1, 5) if half == 1 else range(5, 9))
    positions = tuple(position for position in positions if position in by_position)
    c018 = tuple(by_position[position] for position in positions if spec.order[position - 1] == "C018")
    high = tuple(
        by_position[position]
        for position in positions
        if spec.order[position - 1] in {"C081", "C108"}
    )
    p1 = _precondition_result(
        "P1-below-limit-unthrottled",
        c018,
        expected="C018 is aiclk_not_depressed in this order half",
        field="aiclk_not_depressed",
        expected_count=1,
    )
    p2 = _precondition_result(
        "P2-above-limit-throttled",
        high,
        expected="C081 and C108 are aiclk_depressed in this order half",
        field="aiclk_depressed",
        expected_count=2,
    )
    c081 = next((item for item in high if item.report.get("configuration_id") == "C081"), None)
    c108 = next((item for item in high if item.report.get("configuration_id") == "C108"), None)
    c081_mean = c081.report["statistics"].get("aiclk_mhz", {}).get("time_weighted_mean") if c081 else None
    c108_mean = c108.report["statistics"].get("aiclk_mhz", {}).get("time_weighted_mean") if c108 else None
    p3_pass = None if c081_mean is None or c108_mean is None else float(c081_mean) > float(c108_mean)
    if p3_pass is None:
        p3_status = "not_evaluable"
    elif p3_pass:
        p3_status = "pass"
    else:
        p3_status = "fail"
    p3 = {
        "name": "P3-monotonic",
        "status": p3_status,
        "pass": p3_pass,
        "expected": "C081 plateau mean AICLK is higher than C108 in this order half",
        "C081_plateau_aiclk_mhz": c081_mean,
        "C108_plateau_aiclk_mhz": c108_mean,
        "operator": ">",
    }
    return {"P1": p1, "P2": p2, "P3": p3}


def _hypothesis_results(
    by_position: Mapping[int, _RunAnalysis],
    preconditions: Mapping[str, Mapping[str, Any]],
    spec: _PlanSpec,
    missing_fields: Sequence[str],
) -> tuple[dict[str, Any], list[str]]:
    all_preconditions = [
        preconditions[half][name]
        for half in ("half_1", "half_2")
        for name in ("P1", "P2", "P3")
    ]
    preconditions_pass = not missing_fields and all(item["pass"] is True for item in all_preconditions)
    high = [
        item
        for position, item in by_position.items()
        if spec.order[position - 1] in {"C081", "C108"}
    ]
    low = [
        item
        for position, item in by_position.items()
        if spec.order[position - 1] == "C018"
    ]
    results: dict[str, Any] = {}
    supported: list[str] = []
    for hypothesis, (signal, limit_name) in HYPOTHESIS_SIGNALS.items():
        high_values = [item.report["facts"]["pinned"].get(hypothesis) for item in high]
        low_values = [item.report["facts"]["pinned"].get(hypothesis) for item in low]
        if missing_fields or any(value is None for value in (*high_values, *low_values)):
            classification = "not_evaluable"
        elif not preconditions_pass:
            classification = "not_supported"
        elif all(high_values) and all(value is False for value in low_values):
            classification = "supported"
            supported.append(hypothesis)
        else:
            classification = "not_supported"
        results[hypothesis] = {
            "classification": classification,
            "supported": classification == "supported",
            "signal": signal,
            "limit": limit_name,
            "preconditions_pass": preconditions_pass,
            "high_configuration_pinned": high_values,
            "low_configuration_not_pinned": [
                value is False if value is not None else None for value in low_values
            ],
            "run_pinned": {
                str(item.report.get("run_number")): item.report["facts"]["pinned"].get(hypothesis)
                for item in (*high, *low)
            },
        }
    return results, supported


def _run_limit_name(item: _RunAnalysis) -> str:
    return str(item.report.get("run_number") or item.report.get("run_id"))


def _cross_run_limit_values(
    run_analyses: Sequence[_RunAnalysis],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {name: {} for name in LIMIT_ALIASES}
    for item in run_analyses:
        run_name = _run_limit_name(item)
        for name, info in item.report.get("limits", {}).items():
            if isinstance(info, Mapping) and isinstance(info.get("value"), (int, float)):
                values[name][run_name] = float(info["value"])
    return values


def _other_limit_values(
    values: Mapping[str, Mapping[str, float]], name: str, run_name: str
) -> dict[str, float]:
    return {
        run: value for run, value in values.get(name, {}).items() if run != run_name
    }


def _annotate_cross_run_limits(run_analyses: Sequence[_RunAnalysis]) -> None:
    values = _cross_run_limit_values(run_analyses)
    for item in run_analyses:
        run_name = _run_limit_name(item)
        for name, info in item.report.get("limits", {}).items():
            if not isinstance(info, dict):
                continue
            current = info.get("value")
            others = _other_limit_values(values, name, run_name)
            differing = bool(
                current is not None
                and any(
                    not math.isclose(float(current), value, rel_tol=0.0, abs_tol=1e-12)
                    for value in others.values()
                )
            )
            info["differs_from_other_runs"] = differing
            info["other_run_values"] = others


def _path_record_inputs(item: str | Path) -> list[_RecordInput]:
    path = _safe_cli_path(item, purpose="runner record")
    payload = _load_json(path, purpose="runner record")
    nested = payload.get("runs")
    if not isinstance(nested, list):
        return [_RecordInput(payload, base_dir=path.parent, record_name=path.name)]
    return [
        _RecordInput(run, base_dir=path.parent, record_name=path.name)
        for run in nested
        if isinstance(run, Mapping)
    ]


def _mapping_record_inputs(item: Mapping[str, Any], index: int) -> list[_RecordInput]:
    nested = item.get("runs")
    if not isinstance(nested, list):
        return [_RecordInput(item)]
    return [
        _RecordInput(run, record_name=f"record-{index + 1}")
        for run in nested
        if isinstance(run, Mapping)
    ]


def _record_inputs(
    records: Sequence[Mapping[str, Any] | str | Path],
) -> list[_RecordInput]:
    inputs: list[_RecordInput] = []
    for index, item in enumerate(records):
        if isinstance(item, (str, Path)):
            inputs.extend(_path_record_inputs(item))
        elif isinstance(item, Mapping):
            inputs.extend(_mapping_record_inputs(item, index))
        else:
            raise AnalysisError("runner records must be objects or JSON paths")
    return inputs


def _classification_result(
    missing_fields: set[str],
    failed_preconditions: Sequence[str],
    supported: Sequence[str],
) -> tuple[str, str]:
    if missing_fields:
        return (
            "not_evaluable",
            "required verdict evidence is missing: " + ", ".join(sorted(missing_fields)),
        )
    if failed_preconditions:
        return (
            "H-OTHER",
            "failed preregistered preconditions: "
            + ", ".join(sorted(dict.fromkeys(failed_preconditions))),
        )
    if len(supported) == 1:
        return supported[0], f"exactly one supported hypothesis: {supported[0]}"
    if len(supported) > 1:
        return "multiple", "multiple supported hypotheses: " + ", ".join(supported)
    return "H-OTHER", "no supported hypothesis satisfies P1-P3 and its pinned-signal rule"


def _analysis_plan(spec: _PlanSpec, effective_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_schema": effective_plan.get("record_schema"),
        "source": DEFAULT_PLAN_PATH.name,
        "sha256": spec.plan_sha256,
        "load_mode": ISSUE137_LOAD_MODE,
        "long_run_order": list(spec.order),
        "verdict_configurations": list(spec.verdict_configurations),
        "plateau": {
            "start_seconds": spec.plateau_start_seconds,
            "end_seconds": spec.plateau_end_seconds,
            "interval_definition": "left-held positive-duration segments; terminal coverage required; no extrapolation",
        },
        "timing": {
            "long": {"target_seconds": spec.long_target_seconds, "guard_seconds": spec.long_guard_seconds},
            "smoke": {"target_seconds": spec.smoke_target_seconds, "guard_seconds": spec.smoke_guard_seconds},
            "overrun_limit_seconds": spec.overrun_limit_seconds,
        },
        "criteria": {
            "limit_tolerance_fraction": spec.tolerance_fraction,
            "aiclk_depression_fraction": spec.aiclk_fraction,
            "thermal_pin_fraction": spec.thermal_pin_fraction,
            "hypothesis_signals": HYPOTHESIS_SIGNALS,
        },
        "classification_policy": dict(spec.classification_policy),
    }


def analyze_runs(
    records: Sequence[Mapping[str, Any] | str | Path],
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze persisted runner records without calibration or device access."""
    effective_plan = dict(plan) if isinstance(plan, Mapping) else _load_default_plan()
    spec = _plan_spec(effective_plan)
    inputs = _record_inputs(records)
    run_analyses = [_run_analysis(item, index, spec) for index, item in enumerate(inputs)]
    _annotate_cross_run_limits(run_analyses)
    by_position, absent, duplicates, extras = _position_runs(run_analyses, spec)
    missing_fields: set[str] = set(absent) | set(duplicates) | set(extras)
    preconditions = {
        "half_1": _half_preconditions(by_position, spec, 1),
        "half_2": _half_preconditions(by_position, spec, 2),
    }
    failed_preconditions = [
        criterion["name"]
        for half in preconditions.values()
        for criterion in half.values()
        if criterion.get("pass") is False
    ]
    hypotheses, supported = _hypothesis_results(
        by_position, preconditions, spec, sorted(missing_fields)
    )
    classification, classification_reason = _classification_result(
        missing_fields, failed_preconditions, supported
    )
    hypotheses["H-OTHER"] = {
        "classification": "selected" if classification == "H-OTHER" else "not_selected",
        "supported": classification == "H-OTHER",
        "failed_preconditions": sorted(dict.fromkeys(failed_preconditions)),
        "missing_fields": sorted(missing_fields),
        "reason": classification_reason,
    }
    all_runs = [item.report for item in run_analyses]
    verdict_runs = [by_position[position].report for position in sorted(by_position)]
    exploratory_runs = [report for report in all_runs if not report.get("eligible_for_verdict")]
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "status": "not_evaluable" if missing_fields else "ok",
        "issue": "#137",
        "classification": classification,
        "classification_reason": classification_reason,
        "supported_hypotheses": supported,
        "failed_preconditions": sorted(dict.fromkeys(failed_preconditions)),
        "missing_fields": sorted(missing_fields),
        "verdict_evidence_complete": not missing_fields,
        "plan": _analysis_plan(spec, effective_plan),
        "preconditions": preconditions,
        "hypotheses": hypotheses,
        "runs": all_runs,
        "verdict_runs": verdict_runs,
        "exploratory": {
            "ignored_non_verdict_count": len(exploratory_runs),
            "ignored_non_verdict_runs": [
                {
                    "run_id": run.get("run_id"),
                    "run_number": run.get("run_number"),
                    "configuration_id": run.get("configuration_id"),
                    "mode": run.get("mode"),
                    "diagnostic_only": run.get("diagnostic_only"),
                    "reason": "not verdict-bearing long-run evidence",
                }
                for run in exploratory_runs
            ],
            "C054_affects_verdict": False,
            "smoke_and_calibration_affect_verdict": False,
        },
    }


def load_plan(path: Path = DEFAULT_PLAN_PATH) -> Mapping[str, Any]:
    """Load the immutable Issue #137 plan through the board-free runner seam."""
    try:
        safe_path = _safe_cli_path(path, purpose="immutable plan")
        from tools.issue137 import load_plan as load_runner_plan

        return load_runner_plan(safe_path)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise AnalysisError("immutable Issue #137 plan cannot be loaded") from exc


def _load_default_plan() -> Mapping[str, Any]:
    return load_plan(DEFAULT_PLAN_PATH)


def analyze_record_set(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Discoverable alias for callers that describe a record set explicitly."""
    return analyze_runs(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--record", action="append", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan_path = _safe_cli_path(args.plan, purpose="immutable plan")
        plan = _load_json(plan_path, purpose="immutable plan")
        result = analyze_runs(args.record, plan=plan)
        serialized = strict_json_dumps(_safe_public(result), indent=2) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            _write_text(args.output, serialized, purpose="analysis output")
    except (OSError, TypeError, ValueError) as exc:
        print(f"Issue #137 analysis rejected: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


__all__ = [
    "ANALYSIS_SCHEMA",
    "DEFAULT_PLAN_PATH",
    "ISSUE137_SNAPSHOT_SCHEMA",
    "AnalysisError",
    "analyze_record_set",
    "analyze_runs",
    "load_plan",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
