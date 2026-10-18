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
_UNIT_VALUE_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z°]+)?\s*$"
)


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
    if isinstance(value, str) and _PATH_RE.search(value):
        return "<redacted-path>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _trusted_roots() -> tuple[Path, ...]:
    return tuple(
        root.resolve(strict=False)
        for root in (
            Path.cwd(),
            Path(tempfile.gettempdir()),
            Path("/out"),
            Path(__file__).resolve().parents[1],
        )
    )


def _safe_cli_path(value: str | Path, *, purpose: str, output: bool = False) -> Path:
    try:
        candidate = Path(value)
        resolved = candidate.resolve(strict=False)
        if output and candidate.is_symlink():
            raise AnalysisError(f"{purpose} path must not be a symbolic link")
    except (OSError, RuntimeError, TypeError) as exc:
        raise AnalysisError(f"{purpose} path cannot be resolved safely") from exc
    if not any(resolved.is_relative_to(root) for root in _trusted_roots()):
        raise AnalysisError(f"{purpose} path must be beneath a trusted project directory")
    return resolved


def _load_json(path: Path, *, purpose: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"{purpose} cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise AnalysisError(f"{purpose} must be an object")
    return payload


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


def _plan_spec(plan: Mapping[str, Any]) -> _PlanSpec:
    if plan.get("record_schema") != "issue137-premeasurement-plan-v1":
        raise AnalysisError("unsupported Issue #137 plan schema")
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
    scope = plan.get("measurement_scope")
    if not isinstance(scope, Mapping):
        raise AnalysisError("Issue #137 plan measurement_scope is missing")
    raw_order = scope.get("order")
    if not isinstance(raw_order, list) or not all(isinstance(item, str) for item in raw_order):
        raise AnalysisError("Issue #137 plan order is missing")
    order = tuple(raw_order)
    if len(order) != 8 or set(order) != set(roles):
        raise AnalysisError("Issue #137 plan order is invalid")
    if any(order.count(identifier) != 2 for identifier in roles):
        raise AnalysisError("Issue #137 plan order must contain two runs per configuration")
    verdict_configurations = tuple(
        dict.fromkeys(
            identifier
            for identifier in order
            if roles[identifier].get("verdict_bearing") is True
        )
    )
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
    policy = plan.get("classification_policy")
    if not isinstance(policy, Mapping):
        raise AnalysisError("Issue #137 classification_policy is missing")
    policy_keys = (
        "exactly_one_supported",
        "more_than_one_supported",
        "none_supported",
        "not_evaluable",
    )
    classification_policy: dict[str, str] = {}
    for key in policy_keys:
        value = policy.get(key)
        if not isinstance(value, str) or not value:
            raise AnalysisError(f"Issue #137 classification policy is missing {key}")
        classification_policy[key] = value
    plateau = plan.get("plateau")
    if not isinstance(plateau, Mapping):
        raise AnalysisError("Issue #137 plateau is missing")
    plateau_start = _finite(plateau.get("start_seconds"), field="plan plateau start_seconds")
    plateau_end = _finite(plateau.get("end_seconds"), field="plan plateau end_seconds")
    if plateau_start < 0.0 or plateau_end <= plateau_start:
        raise AnalysisError("Issue #137 plateau interval is invalid")
    execution = plan.get("load_execution")
    if not isinstance(execution, Mapping) or execution.get("mode") != ISSUE137_LOAD_MODE:
        raise AnalysisError("Issue #137 plan does not select host-duration-bounded execution")
    try:
        overrun_limit = _finite(
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
    if overrun_limit < 0.0 or long_target <= 0.0 or long_guard < long_target:
        raise AnalysisError("Issue #137 long duration bounds are invalid")
    if smoke_target <= 0.0 or smoke_guard < smoke_target:
        raise AnalysisError("Issue #137 smoke duration bounds are invalid")
    required_environment = plan.get("provenance", {}).get("required_environment_fields")
    if not isinstance(required_environment, list) or not all(
        isinstance(item, str) and item for item in required_environment
    ):
        raise AnalysisError("Issue #137 required environment provenance is missing")
    board_serial = scope.get("board_serial")
    pci_bus_id = scope.get("pci_bus_id")
    device_node = scope.get("device_node")
    if not all(isinstance(value, str) and value for value in (board_serial, pci_bus_id, device_node)):
        raise AnalysisError("Issue #137 selected board identity is incomplete")
    return _PlanSpec(
        plan=plan,
        order=order,
        verdict_configurations=verdict_configurations,
        configuration_roles=roles,
        expected_configurations=_expected_configurations(roles, plan),
        plateau_start_seconds=plateau_start,
        plateau_end_seconds=plateau_end,
        expected_limits=expected_limits,
        tolerance_fraction=tolerance,
        aiclk_fraction=aiclk_fraction,
        thermal_pin_fraction=thermal_fraction,
        classification_policy=classification_policy,
        long_target_seconds=long_target,
        long_guard_seconds=long_guard,
        smoke_target_seconds=smoke_target,
        smoke_guard_seconds=smoke_guard,
        overrun_limit_seconds=overrun_limit,
        board_serial=board_serial,
        pci_bus_id=pci_bus_id,
        device_node=device_node,
        required_environment_fields=tuple(required_environment),
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
    candidate = Path(value)
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return _safe_cli_path(candidate, purpose=purpose)


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


def _raw_trace_rows(payload: Mapping[str, Any], load: _TraceLoad) -> None:
    raw_rows = payload.get("samples")
    if not isinstance(raw_rows, list):
        load.errors.append("trace samples are missing")
        load.missing_fields.add("trace.samples")
        return
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            load.errors.append(f"trace row {index} is not an object")
            continue
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
        for field in (
            "timestamp_utc",
            "snapshot_elapsed_ns",
            "heartbeat",
            "therm_trip_count",
            "throttle_status",
            "throttle_reason",
            *GDDR_FIELDS,
        ):
            if field in raw_row:
                row[field] = raw_row[field]
        for field in ("MAX_GDDR_TEMP", "max_gddr_temp", "max_gddr_temp_raw"):
            if field in raw_row:
                row["MAX_GDDR_TEMP"] = raw_row[field]
        load.rows.append(row)
    if any(
        field not in raw_row
        for raw_row in raw_rows
        if isinstance(raw_row, Mapping)
        for field in GDDR_FIELDS
    ):
        load.errors.append("GDDR temperature fields are missing from one or more rows")


def _trace_payload_from_csv(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
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
        trace = record.get("trace")
        path_value = trace.get("path") if isinstance(trace, Mapping) else None
        if path_value is None:
            return _TraceLoad(
                rows=[],
                source_name=None,
                source_sha256=None,
                source_byte_count=None,
                source_established=False,
                bounds_start_ns=None,
                bounds_end_ns=None,
                errors=["trace source is missing"],
                missing_fields={"trace"},
                reader_facts={},
            )
        try:
            path = _source_path(path_value, base_dir=base_dir, purpose="trace source")
            source_name = path.name
            if path.suffix.lower() == ".json":
                payload = _load_json(path, purpose="trace source")
                source_bytes = path.read_bytes()
            else:
                payload, source_bytes = _trace_payload_from_csv(path)
        except (AnalysisError, OSError, TypeError, ValueError) as exc:
            return _TraceLoad(
                rows=[],
                source_name=Path(str(path_value)).name,
                source_sha256=None,
                source_byte_count=None,
                source_established=False,
                bounds_start_ns=None,
                bounds_end_ns=None,
                errors=[str(exc)],
                missing_fields={"trace"},
                reader_facts={},
            )
    if source_name is None:
        source_name = str(record.get("trace_name") or f"{record_name}.trace")
    if source_bytes is None:
        source_bytes = _canonical_trace_bytes(payload)
    established = source_verified and _bound_gddr_provenance(record, payload)
    trace_metadata = record.get("trace")
    expected_hash = (
        trace_metadata.get("sha256", trace_metadata.get("trace_sha256"))
        if isinstance(trace_metadata, Mapping)
        else None
    )
    expected_bytes = (
        trace_metadata.get("byte_count", trace_metadata.get("trace_byte_count"))
        if isinstance(trace_metadata, Mapping)
        else None
    )
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        load_hash_error = "trace SHA-256 provenance is missing or invalid"
    elif not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 1:
        load_hash_error = "trace byte-count provenance is missing or invalid"
    elif hashlib.sha256(source_bytes).hexdigest() != expected_hash or len(source_bytes) != expected_bytes:
        load_hash_error = "trace source bytes do not match persisted hash/byte-count provenance"
    else:
        load_hash_error = None
    if load_hash_error is not None:
        load_hash_missing = {"trace_sha256", "trace_byte_count"}
    else:
        load_hash_missing = set()
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
        load.missing_fields.update(load_hash_missing)
    if payload.get("schema") != power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA:
        load.errors.append("Issue #137 trace schema is unsupported")
    if tuple(payload.get("columns", ())) != power_trace_issue137.ISSUE137_POWER_TRACE_COLUMNS:
        load.errors.append("Issue #137 trace columns are invalid")
    _raw_trace_rows(payload, load)
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
            source_established=established,
        )
        load.reader_facts = _safe_public(facts)
        if facts.get("valid") is not True:
            load.errors.extend(str(error) for error in facts.get("errors", ()))
    except (TypeError, ValueError, KeyError) as exc:
        load.errors.append(f"trace reader rejected source: {type(exc).__name__}")
    return load


def _pre_snapshot_mapping(
    record: Mapping[str, Any], *, base_dir: Path | None, spec: _PlanSpec
) -> tuple[Mapping[str, Any] | None, str | None, list[str]]:
    pre = record.get("pre_run_snapshot", record.get("pre_snapshot"))
    if not isinstance(pre, Mapping):
        return None, None, ["pre_run_snapshot"]
    path_name: str | None = None
    errors: list[str] = []
    if pre.get("status") not in {None, "ok"} or pre.get("returncode") not in {None, 0}:
        errors.append("pre-run snapshot capture did not complete successfully")
    if isinstance(pre.get("selected_device"), Mapping):
        selected = pre["selected_device"]
        if selected.get("board_id") != spec.board_serial or selected.get("bus_id") != spec.pci_bus_id:
            errors.append("pre-run snapshot selected-device identity does not match the owner plan")
        if selected.get("device_node") not in {None, spec.device_node}:
            errors.append("pre-run snapshot selected-device node does not match the owner plan")
    if isinstance(pre.get("path"), (str, Path)) and "snapshot" not in pre:
        try:
            path = _source_path(pre["path"], base_dir=base_dir, purpose="pre-run snapshot")
            payload = _load_json(path, purpose="pre-run snapshot")
            path_name = path.name
        except AnalysisError as exc:
            return None, Path(str(pre["path"])).name, [str(exc)]
    else:
        payload = pre
        path_name = Path(str(pre.get("path"))).name if pre.get("path") else None
    if payload.get("status") not in {None, "ok"} or payload.get("returncode") not in {None, 0}:
        errors.append("pre-run snapshot capture did not complete successfully")
    snapshot = payload.get("snapshot") if isinstance(payload.get("snapshot"), Mapping) else payload
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("device_info"), list):
        errors.append("pre-run snapshot device identity is missing")
        return None, path_name, list(dict.fromkeys(errors))
    try:
        _selected_snapshot_device(snapshot, board_id=spec.board_serial, bus_id=spec.pci_bus_id)
    except AnalysisError as exc:
        errors.append(str(exc))
    return snapshot, path_name, list(dict.fromkeys(errors))


def _selected_snapshot_device(
    snapshot: Mapping[str, Any], *, board_id: str, bus_id: str
) -> Mapping[str, Any]:
    devices = snapshot.get("device_info")
    if not isinstance(devices, list) or not devices:
        raise AnalysisError("pre-run snapshot device_info is missing")
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
    if len(board_matches) != 1 or len(bus_matches) != 1 or len(pair_matches) != 1:
        raise AnalysisError("pre-run snapshot selected-device identity is not unique")
    if pair_matches[0] is not board_matches[0] or pair_matches[0] is not bus_matches[0]:
        raise AnalysisError("pre-run snapshot board and BDF identify different devices")
    return pair_matches[0]


def _parse_unit_value(value: Any, *, expected_unit: str) -> tuple[float | None, str, str | None]:
    """Parse a limit without interpreting raw/hex values as a measured unit."""
    explicit_unit: str | None = None
    conversion_unit: str | None = None
    conversion_proven = False
    if isinstance(value, Mapping):
        mapping_value = value
        explicit_unit = mapping_value.get(
            "unit", mapping_value.get("units", mapping_value.get("source_unit"))
        )
        raw_value = mapping_value.get("value", _MISSING)
        if raw_value is _MISSING and mapping_value.get("converted_value") is not None:
            raw_value = mapping_value.get("converted_value")
            conversion_unit = mapping_value.get("converted_unit")
            explicit_unit = conversion_unit or explicit_unit
            conversion_proven = isinstance(mapping_value.get("conversion_provenance"), Mapping)
        if raw_value is _MISSING:
            return None, "missing", "limit value is missing"
        value = raw_value
    if isinstance(value, bool) or value is None:
        return None, "missing", "limit value is not numeric"
    text = str(value).strip()
    if text.lower().startswith(("0x", "+0x", "-0x")):
        return None, "unknown", "raw hexadecimal limit packing has no bound unit provenance"
    match = _UNIT_VALUE_RE.fullmatch(text)
    if match is None:
        return None, "unknown", "limit value is not a decimal number with a known unit"
    number = _finite(match.group(1), field="limit value")
    embedded_unit = match.group(2)
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


def _aggregate_gddr(
    rows: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    source_established: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_fields: dict[str, list[Any]] = {field: [] for field in GDDR_FIELDS}
    max_raw: list[Any] = []
    for row in rows:
        for field in GDDR_FIELDS:
            if field in row:
                raw_fields[field].append(row[field])
        if "MAX_GDDR_TEMP" in row:
            max_raw.append(row["MAX_GDDR_TEMP"])
    diagnostic: dict[str, Any] = {
        "status": "decoded" if source_established else "not_evaluable",
        "raw_fields": raw_fields,
        "max_gddr_temp_raw": max_raw,
    }
    if not source_established:
        diagnostic["reason"] = "GDDR source unit/packing provenance is not established"
        return {}, diagnostic
    decoded_by_row: dict[int, dict[str, Any]] = {}
    try:
        for row in rows:
            decoded_by_row[id(row)] = power_trace_issue137.decode_issue137_gddr_row(
                row, source_established=True
            )
    except (TypeError, ValueError, KeyError) as exc:
        diagnostic.update(
            status="not_evaluable",
            reason=f"GDDR die decode failed: {type(exc).__name__}",
        )
        return {}, diagnostic
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
    aggregates: dict[str, Any] = {}
    for channel, values in channels.items():
        durations = [int(item["duration_ns"]) for item in values]
        total = sum(durations)
        bottom = [float(item["bottom_c"]) for item in values]
        top = [float(item["top_c"]) for item in values]
        aggregates[channel] = {
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
    diagnostic["channel_count"] = len(aggregates)
    return aggregates, diagnostic


def _trace_analysis(
    load: _TraceLoad,
    spec: _PlanSpec,
    *,
    expected_start_ns: int | None,
    expected_end_ns: int | None,
) -> tuple[dict[str, Any], set[str]]:
    missing = set(load.missing_fields)
    rows = load.rows
    timed_rows = [row for row in rows if isinstance(row.get("timestamp_monotonic_ns"), int)]
    start_ns, end_ns = load.bounds_start_ns, load.bounds_end_ns
    if expected_start_ns is not None and start_ns != expected_start_ns:
        load.errors.append("trace start bound does not match first launch start")
        missing.add("trace_start_binding")
    if expected_end_ns is not None and end_ns != expected_end_ns:
        load.errors.append("trace end bound does not match target workload end")
        missing.add("trace_end_binding")
    plateau_start_ns = None
    plateau_end_ns = None
    segments: list[tuple[Mapping[str, Any], int, int]] = []
    coverage_status = "not_evaluable"
    if start_ns is not None and end_ns is not None:
        plateau_start_ns = start_ns + round(spec.plateau_start_seconds * 1_000_000_000.0)
        plateau_end_ns = start_ns + round(spec.plateau_end_seconds * 1_000_000_000.0)
    if not rows:
        missing.add("trace_samples")
    elif any(
        int(left["timestamp_monotonic_ns"]) > int(right["timestamp_monotonic_ns"])
        for left, right in itertools.pairwise(timed_rows)
    ):
        missing.add("trace_timestamp_order")
    elif plateau_start_ns is None or plateau_end_ns is None:
        missing.update({"run_start_monotonic_ns", "run_end_monotonic_ns"})
    else:
        first_timestamp = timed_rows[0].get("timestamp_monotonic_ns") if timed_rows else None
        last_timestamp = timed_rows[-1].get("timestamp_monotonic_ns") if timed_rows else None
        if not isinstance(first_timestamp, int) or first_timestamp > plateau_start_ns:
            missing.add("plateau_start_coverage")
        if not isinstance(last_timestamp, int) or last_timestamp < plateau_end_ns:
            missing.add("plateau_terminal_coverage")
        if isinstance(first_timestamp, int) and isinstance(last_timestamp, int):
            segments = _left_held_segments(
                timed_rows, start_ns=plateau_start_ns, end_ns=plateau_end_ns
            )
            if sum(end - start for _row, start, end in segments) != plateau_end_ns - plateau_start_ns:
                missing.add("plateau_coverage")
            elif not missing.intersection({"plateau_start_coverage", "plateau_terminal_coverage"}):
                coverage_status = "complete"
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
        "trace_rows": [dict(row) for row in rows],
        "reader_facts": load.reader_facts,
        "exploratory": {"gddr": gddr_exploratory},
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


def _environment_errors(
    record: Mapping[str, Any],
    result: Mapping[str, Any] | None,
    configuration: Mapping[str, Any] | None,
    spec: _PlanSpec,
) -> list[str]:
    errors: list[str] = []
    environment = record.get("environment")
    if not isinstance(environment, Mapping):
        return ["environment"]
    board = environment.get("board")
    identity = board.get("identity") if isinstance(board, Mapping) else None
    firmware = environment.get("firmware")
    required = {
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
    for field in spec.required_environment_fields:
        if field not in required or required[field] is None or required[field] == "":
            errors.append(field)
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
        for field, expected in expected_environment.items():
            if environment.get(field) != expected:
                errors.append(f"environment.{field}.binding")
    if isinstance(result, Mapping):
        raw_environment = result.get("environment")
        if not isinstance(raw_environment, Mapping):
            errors.append("runner_result.environment")
        else:
            raw_serial, raw_board, raw_bus = _identity_from_environment(raw_environment)
            if (raw_serial, raw_board, raw_bus) != (
                spec.board_serial,
                spec.board_serial,
                spec.pci_bus_id,
            ):
                errors.append("runner_result.environment.identity.binding")
            raw_firmware = raw_environment.get("firmware") or raw_environment.get("firmwares")
            raw_image = raw_environment.get("image_digest") or raw_environment.get("image")
            raw_driver = raw_environment.get("kernel_driver_version") or raw_environment.get("kmd_version")
            raw_required = {
                "firmware": raw_firmware.get("bundle") or raw_firmware.get("fw_bundle_version")
                if isinstance(raw_firmware, Mapping)
                else None,
                "kernel": raw_environment.get("kernel"),
                "kernel_driver_version": raw_driver,
                "image_digest": raw_image,
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
            try:
                from tools import issue137 as runner

                raw_configuration_id = (
                    configuration.get("id") if isinstance(configuration, Mapping) else None
                )
                canonical_configuration = runner.configuration(spec.plan, raw_configuration_id)
                canonical_environment = runner.build_environment_record(
                    raw_environment, canonical_configuration
                )
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
                for field in persisted_fields:
                    if environment.get(field) != canonical_environment.get(field):
                        errors.append(f"runner_result.environment.{field}.binding")
            except (ImportError, KeyError, TypeError, ValueError):
                errors.append("runner_result.environment.canonical")
    return list(dict.fromkeys(errors))


def _duration_binding(
    record: Mapping[str, Any],
    spec: _PlanSpec,
    configuration: Mapping[str, Any] | None,
    configuration_id: str | None,
    mode: str,
) -> tuple[dict[str, Any], list[str], Mapping[str, Any] | None]:
    errors: list[str] = []
    run = record.get("run") if isinstance(record.get("run"), Mapping) else {}
    result = record.get("runner_result")
    result = result if isinstance(result, Mapping) else None
    target, guard = (
        (spec.long_target_seconds, spec.long_guard_seconds)
        if mode == "long"
        else (spec.smoke_target_seconds, spec.smoke_guard_seconds)
    )
    if record.get("record_schema") != ISSUE137_RECORD_SCHEMA:
        errors.append("record_schema")
    if record.get("record_revision") != 2:
        errors.append("record_revision")
    if record.get("plan_sha256") != spec.plan_sha256:
        errors.append("plan_sha256")
    if record.get("status") != "ok":
        errors.append("run_status")
    if record.get("workload_started") is not True:
        errors.append("workload_started")
    count = first = workload_end = last = None
    if result is None:
        errors.append("runner_result")
    else:
        if not isinstance(run.get("run_id"), str) or not run["run_id"]:
            errors.append("run.run_id")
        if result.get("record_schema") != ISSUE137_DEVICE_RESULT_SCHEMA:
            errors.append("runner_result.record_schema")
        if result.get("status") != "ok":
            errors.append("runner_result.status")
        if result.get("mode") != mode:
            errors.append("runner_result.mode")
        if result.get("load_mode") != ISSUE137_LOAD_MODE:
            errors.append("load_mode")
        timing = run.get("timing")
        if not isinstance(timing, Mapping):
            errors.append("run.timing")
        else:
            if timing.get("load_seconds") != target or timing.get("guard_seconds") != guard:
                errors.append("run.timing.binding")
        if result.get("plan_sha256") != spec.plan_sha256:
            errors.append("runner_result.plan_sha256")
        if not isinstance(result.get("run_id"), str) or not result["run_id"]:
            errors.append("runner_result.run_id")
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            errors.append("provenance")
        else:
            if provenance.get("wrapper_run_id") != result.get("run_id"):
                errors.append("provenance.wrapper_run_id")
            path = provenance.get("runner_result_path")
            if not isinstance(path, str) or Path(path).name != path:
                errors.append("provenance.runner_result_path")
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
        try:
            count = _integer(result.get("launch_count"), field="launch_count")
            if count <= 0:
                errors.append("launch_count")
        except AnalysisError:
            count = None
            errors.append("launch_count")
        try:
            first = _integer(result.get("first_launch_start_monotonic_ns"), field="first launch")
            workload_end = _integer(result.get("workload_end_monotonic_ns"), field="workload end")
            last = _integer(result.get("last_launch_end_monotonic_ns"), field="last launch")
        except AnalysisError:
            first = workload_end = last = None
            errors.append("monotonic_bounds")
        if first is not None and workload_end is not None and last is not None:
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
        else:
            count = None
            first = workload_end = last = None
    if configuration is None:
        errors.append("configuration")
    elif configuration != spec.expected_configurations.get(configuration_id or ""):
        errors.append("configuration.binding")
    if result is not None and configuration is not None and result.get("core_plan") != configuration.get("core_plan"):
        errors.append("runner_result.core_plan.binding")
    for field, expected in (
        ("load_mode", ISSUE137_LOAD_MODE),
        ("target_seconds", target),
        ("guard_seconds", guard),
        ("overrun_limit_seconds", spec.overrun_limit_seconds),
    ):
        actual = run.get(field)
        try:
            matches = actual == expected or math.isclose(_finite(actual, field=field), expected, rel_tol=0.0, abs_tol=1e-12)
        except AnalysisError:
            matches = False
        if not matches:
            errors.append(f"run.{field}")
    if result is not None:
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
    for container in (record, run, result or {}):
        if any(key in container for key in ("calibration", "calibration_name", "repeat_count", "repeat_counts")):
            errors.append("calibration_or_repeat_count")
    trace = record.get("trace")
    if not isinstance(trace, Mapping):
        errors.append("trace")
    else:
        if trace.get("schema") not in {None, power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA}:
            errors.append("trace.schema")
        if trace.get("run_start_monotonic_ns") not in {None, first}:
            errors.append("trace.run_start_monotonic_ns")
        if trace.get("run_end_monotonic_ns") not in {None, workload_end}:
            errors.append("trace.run_end_monotonic_ns")
        if trace.get("actual_end_monotonic_ns") not in {None, last}:
            errors.append("trace.actual_end_monotonic_ns")
        path = trace.get("path")
        if not isinstance(path, str) or Path(path).name != path:
            errors.append("trace.path")
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


def _run_analysis(record_input: _RecordInput, index: int, spec: _PlanSpec) -> _RunAnalysis:
    record = record_input.record
    name = _record_name(record_input, index)
    configuration_id, run_number, mode = _run_identity(record)
    role = spec.configuration_roles.get(configuration_id or "", {})
    expected_candidate = (
        mode == "long"
        and role.get("verdict_bearing") is True
        and run_number in range(1, len(spec.order) + 1)
    )
    configuration = record.get("configuration") if isinstance(record.get("configuration"), Mapping) else None
    binding, binding_errors, result = _duration_binding(
        record, spec, configuration, configuration_id, mode
    )
    environment_errors = _environment_errors(record, result, configuration, spec)
    snapshot, snapshot_name, snapshot_errors = _pre_snapshot_mapping(
        record,
        base_dir=record_input.base_dir,
        spec=spec,
    )
    limits, missing_limits, limit_errors = _extract_limits(snapshot, spec)
    source_versions_verified = _observed_source_versions_verified(
        record, result, snapshot
    )
    load = _load_trace_source(
        record,
        base_dir=record_input.base_dir,
        record_name=name,
        source_verified=source_versions_verified,
    )
    trace_report, missing_trace = _trace_analysis(
        load,
        spec,
        expected_start_ns=binding["workload_start_monotonic_ns"],
        expected_end_ns=binding["workload_end_monotonic_ns"],
    )
    gddr_exploratory = trace_report["exploratory"].get("gddr")
    if isinstance(gddr_exploratory, dict) and snapshot is not None:
        try:
            selected = _selected_snapshot_device(
                snapshot, board_id=spec.board_serial, bus_id=spec.pci_bus_id
            )
        except AnalysisError:
            selected = None
        smbus = selected.get("smbus_telem") if isinstance(selected, Mapping) else None
        if isinstance(smbus, Mapping) and "MAX_GDDR_TEMP" in smbus:
            gddr_exploratory.setdefault("max_gddr_temp_raw", []).append(
                smbus["MAX_GDDR_TEMP"]
            )
    verdict_missing: set[str] = set()
    if expected_candidate:
        verdict_missing.update(binding_errors)
        verdict_missing.update(environment_errors)
        verdict_missing.update(snapshot_errors)
        verdict_missing.update(missing_limits)
        verdict_missing.update(field for field in missing_trace if _verdict_trace_missing(field))
        trace_errors_require_verdict = not load.missing_fields or any(
            _verdict_trace_missing(field) for field in load.missing_fields
        )
        if load.errors and trace_errors_require_verdict:
            verdict_missing.add("trace")
    statistics = trace_report["statistics"]
    aiclk_depressed, aiclk_not_depressed = _aiclk_fact(statistics, limits, spec)
    pinned = {
        hypothesis: _pinned_fact(hypothesis, statistics, limits, spec)
        for hypothesis in HYPOTHESIS_SIGNALS
    }
    if expected_candidate:
        if aiclk_depressed is None:
            verdict_missing.add("aiclk_mhz")
            if limits.get("asic_fmax_mhz", {}).get("value") is None:
                verdict_missing.add("asic_fmax_mhz")
        for hypothesis, (signal, limit_name) in HYPOTHESIS_SIGNALS.items():
            if pinned[hypothesis] is None:
                if statistics.get(signal, {}).get("time_weighted_mean") is None:
                    verdict_missing.add(signal)
                if limits.get(limit_name, {}).get("value") is None:
                    verdict_missing.add(limit_name)
    position = run_number if expected_candidate else None
    report = {
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
            "load_errors": list(dict.fromkeys(load.errors + limit_errors + snapshot_errors)),
            "binding_errors": list(dict.fromkeys(binding_errors + environment_errors)),
            "missing_fields": sorted(
                set(verdict_missing)
                | set(load.missing_fields)
                | set(load.errors)
                | set(limit_errors)
                if expected_candidate
                else set(load.missing_fields) | set(load.errors) | set(limit_errors)
            ),
        },
        "missing_fields": sorted(verdict_missing),
    }
    return _RunAnalysis(
        report=report,
        expected_candidate=expected_candidate,
        eligible=expected_candidate and not verdict_missing,
        verdict_position=position,
    )


def _position_runs(
    run_analyses: Sequence[_RunAnalysis], spec: _PlanSpec
) -> tuple[dict[int, _RunAnalysis], list[str], list[str], list[str]]:
    expected_positions = {
        index
        for index, configuration_id in enumerate(spec.order, start=1)
        if configuration_id in spec.verdict_configurations
    }
    by_position: dict[int, _RunAnalysis] = {}
    missing: list[str] = []
    duplicates: list[str] = []
    extras: list[str] = []
    for item in run_analyses:
        position = item.verdict_position
        actual_config = item.report.get("configuration_id")
        if position not in expected_positions:
            if item.expected_candidate:
                extras.append(
                    f"{actual_config or '<missing-config>'}.run_{item.report.get('run_number') or '<missing-number>'}.ordered_evidence"
                )
            continue
        expected_config = spec.order[position - 1]
        if actual_config != expected_config:
            missing.append(f"run_{position}.{expected_config}.ordered_evidence")
            extras.append(f"run_{position}.{actual_config or '<missing-config>'}.ordered_evidence")
            continue
        if not item.eligible:
            missing.append(f"run_{position}.{expected_config}.ordered_evidence")
            missing.extend(
                f"run_{position}.{expected_config}.{field}"
                for field in item.report.get("missing_fields", [])
            )
            continue
        if position in by_position:
            duplicates.append(f"run_{position}.{expected_config}.duplicate_evidence")
        else:
            by_position[position] = item
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
    p3 = {
        "name": "P3-monotonic",
        "status": "not_evaluable" if p3_pass is None else ("pass" if p3_pass else "fail"),
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


def _annotate_cross_run_limits(run_analyses: Sequence[_RunAnalysis]) -> None:
    values: dict[str, dict[str, float]] = {name: {} for name in LIMIT_ALIASES}
    for item in run_analyses:
        run_name = str(item.report.get("run_number") or item.report.get("run_id"))
        for name, info in item.report.get("limits", {}).items():
            if isinstance(info, Mapping) and isinstance(info.get("value"), (int, float)):
                values[name][run_name] = float(info["value"])
    for item in run_analyses:
        for name, info in item.report.get("limits", {}).items():
            if not isinstance(info, dict):
                continue
            current = info.get("value")
            others = {
                run: value
                for run, value in values.get(name, {}).items()
                if run != str(item.report.get("run_number") or item.report.get("run_id"))
            }
            differing = bool(
                current is not None
                and any(not math.isclose(float(current), value, rel_tol=0.0, abs_tol=1e-12) for value in others.values())
            )
            info["differs_from_other_runs"] = differing
            info["other_run_values"] = others


def analyze_runs(
    records: Sequence[Mapping[str, Any] | str | Path],
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze persisted runner records without calibration or device access."""
    effective_plan = dict(plan) if isinstance(plan, Mapping) else _load_default_plan()
    spec = _plan_spec(effective_plan)
    inputs: list[_RecordInput] = []
    for item in records:
        if isinstance(item, (str, Path)):
            path = _safe_cli_path(item, purpose="runner record")
            payload = _load_json(path, purpose="runner record")
            nested = payload.get("runs")
            if isinstance(nested, list):
                inputs.extend(
                    _RecordInput(run, base_dir=path.parent, record_name=path.name)
                    for run in nested
                    if isinstance(run, Mapping)
                )
            else:
                inputs.append(_RecordInput(payload, base_dir=path.parent, record_name=path.name))
        elif isinstance(item, Mapping):
            nested = item.get("runs")
            if isinstance(nested, list):
                inputs.extend(
                    _RecordInput(run, record_name=f"record-{index + 1}")
                    for index, run in enumerate(nested)
                    if isinstance(run, Mapping)
                )
            else:
                inputs.append(_RecordInput(item))
        else:
            raise AnalysisError("runner records must be objects or JSON paths")
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
    if missing_fields:
        classification = "not_evaluable"
        classification_reason = "required verdict evidence is missing: " + ", ".join(sorted(missing_fields))
    elif failed_preconditions:
        classification = "H-OTHER"
        classification_reason = "failed preregistered preconditions: " + ", ".join(
            sorted(dict.fromkeys(failed_preconditions))
        )
    elif len(supported) == 1:
        classification = supported[0]
        classification_reason = f"exactly one supported hypothesis: {supported[0]}"
    elif len(supported) > 1:
        classification = "multiple"
        classification_reason = "multiple supported hypotheses: " + ", ".join(supported)
    else:
        classification = "H-OTHER"
        classification_reason = "no supported hypothesis satisfies P1-P3 and its pinned-signal rule"
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
        "plan": {
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
        },
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
        from tools.issue137 import load_plan as load_runner_plan

        return load_runner_plan(Path(path))
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
            output_path = _safe_cli_path(args.output, purpose="analysis output", output=True)
            output_path.write_text(serialized, encoding="utf-8")
    except (AnalysisError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
