"""Analyze Issue #121 Q2 extended-v1 traces against the frozen plan.

The analyzer is board-free.  It consumes the versioned extended trace reader,
uses the monotonic timestamp axis for every interval fact, and emits only
structured JSON.  A hypothesis label means that the fixed criteria match; it
does not claim a physical causal mechanism.
"""

from __future__ import annotations

import argparse
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
from enodia.tt.bench.power_trace_extended import (
    EXTENDED_POWER_TRACE_COLUMNS,
    EXTENDED_POWER_TRACE_SCHEMA,
    EXTENDED_POWER_TRACE_SCHEMA_VERSION,
    read_extended_power_trace_snapshot,
)

ANALYSIS_SCHEMA = "issue121-q2-analysis-v1"
DEFAULT_PLATEAU_START_SECONDS = 150.0
DEFAULT_PLATEAU_END_SECONDS = 285.0
AICLK_DROP_FRACTION = 0.05
H_TDC_LIMIT = 200.0
H_TDC_PROXIMITY_FRACTION = 0.10
H_TDC_ONSET_THRESHOLD = 180.0
H_THM_LIMIT_C = 90.0
H_THM_SHUTDOWN_C = 110.0
H_THM_MAX_DIFFERENCE_C = 5.0
H_TDP_LIMIT_W = 150.0
BOARD_POWER_LIMIT_W = 300.0
H_TDP_PROXIMITY_FRACTION = 0.05
EARLY_RAW_WINDOW_SECONDS = 10.0
SAMPLING_CADENCE_SECONDS = 2.0
AICLK_BASELINE_SOURCE = "limits metadata asic_fmax_mhz"

SIGNAL_UNITS = {
    "power_w": "W",
    "board_power_w": "W",
    "aiclk_mhz": "MHz",
    "current_a": "A as reported by snapshot telemetry",
    "voltage_v": "V as reported by snapshot telemetry",
    "asic_temp_c": "degC",
}
_SIGNAL_FIELDS = tuple(SIGNAL_UNITS)
_RAW_WINDOW_FIELDS = (
    "timestamp_monotonic_ns",
    "seconds_from_run_start",
    "aiclk_mhz",
    "power_w",
    "current_a",
    "asic_temp_c",
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
_FORBIDDEN_PATH_TEXT = re.compile(
    r"(?:^|[\s'\"(])(?:/home/|/Users/|/root/|/tmp/|~[/\\]|[A-Za-z]:[\\/])"
)


class AnalysisError(ValueError):
    """Raised when an extended trace cannot be analyzed safely."""


def _trusted_cli_roots() -> tuple[Path, ...]:
    roots = (
        Path.cwd(),
        Path(tempfile.gettempdir()),
        Path("/out"),
        Path(__file__).resolve().parents[1],
    )
    return tuple(root.resolve(strict=False) for root in roots)


def _safe_cli_path(value: str | Path, *, purpose: str, output: bool = False) -> Path:
    """Resolve CLI paths only when they stay in a trusted project or test root."""
    try:
        candidate = Path(value)
        resolved = candidate.resolve(strict=False)
        is_symlink = candidate.is_symlink()
        trusted_roots = _trusted_cli_roots()
    except (OSError, RuntimeError, TypeError) as exc:
        raise AnalysisError(f"{purpose} path cannot be resolved safely") from exc
    if output and is_symlink:
        raise AnalysisError(f"{purpose} path must not be a symbolic link")
    if not any(resolved.is_relative_to(root) for root in trusted_roots):
        raise AnalysisError(f"{purpose} path must be beneath a trusted project directory")
    return resolved


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
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{field} must be a non-negative integer") from exc
    if number < 0 or (isinstance(value, float) and number != value):
        raise AnalysisError(f"{field} must be a non-negative integer")
    return number


def _safe_provenance(value: Any) -> Any:
    """Retain source facts while excluding host identity and absolute paths."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if key_text.casefold() in _FORBIDDEN_KEYS:
                continue
            result[key_text] = _safe_provenance(child)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_provenance(child) for child in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        if _FORBIDDEN_PATH_TEXT.search(value):
            return "<redacted-path>"
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _row_copy(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    missing = [field for field in EXTENDED_POWER_TRACE_COLUMNS if field not in row]
    extra = [field for field in row if field not in EXTENDED_POWER_TRACE_COLUMNS]
    if missing or extra:
        raise AnalysisError(
            f"extended trace row {index} columns are invalid (missing={missing!r}, extra={extra!r})"
        )
    copied: dict[str, Any] = dict(row)
    copied["timestamp_monotonic_ns"] = _integer(
        row["timestamp_monotonic_ns"], field=f"row {index} timestamp_monotonic_ns"
    )
    for field in _SIGNAL_FIELDS:
        copied[field] = _finite(row[field], field=f"row {index} {field}")
    return copied


def _load_mapping_trace(
    payload: Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    schema = payload.get("schema", EXTENDED_POWER_TRACE_SCHEMA)
    version = payload.get("schema_version", EXTENDED_POWER_TRACE_SCHEMA_VERSION)
    columns = payload.get("columns", list(EXTENDED_POWER_TRACE_COLUMNS))
    if schema != EXTENDED_POWER_TRACE_SCHEMA or version != EXTENDED_POWER_TRACE_SCHEMA_VERSION:
        raise AnalysisError("extended trace schema is unsupported")
    if tuple(columns) != EXTENDED_POWER_TRACE_COLUMNS:
        raise AnalysisError("extended trace columns are unsupported")
    raw_rows = payload.get("samples")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise AnalysisError("extended trace samples must be a non-empty array")
    rows = [
        _row_copy(row, index)
        for index, row in enumerate(raw_rows)
        if isinstance(row, Mapping)
    ]
    if len(rows) != len(raw_rows):
        raise AnalysisError("extended trace samples must contain objects")
    start = (
        run_start_monotonic_ns
        if run_start_monotonic_ns is not None
        else payload.get("run_start_monotonic_ns", rows[0]["timestamp_monotonic_ns"])
    )
    end = (
        run_end_monotonic_ns
        if run_end_monotonic_ns is not None
        else payload.get("run_end_monotonic_ns", rows[-1]["timestamp_monotonic_ns"])
    )
    source_metadata = payload.get("provenance", payload.get("source", {}))
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}
    source_metadata = dict(source_metadata)
    for identity_field in ("board_id", "bus_id"):
        if identity_field in payload and identity_field not in source_metadata:
            source_metadata[identity_field] = payload[identity_field]
    reader_facts = {
        "schema": schema,
        "schema_version": version,
        "columns": list(columns),
        "run_start_monotonic_ns": _integer(start, field="run_start_monotonic_ns"),
        "run_end_monotonic_ns": _integer(end, field="run_end_monotonic_ns"),
        "valid": True,
    }
    return rows, reader_facts, dict(source_metadata)


def _load_trace_file(
    path: Path,
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    safe_path = _safe_cli_path(path, purpose="trace source")
    trace = read_extended_power_trace_snapshot(
        safe_path,
        run_start_monotonic_ns=run_start_monotonic_ns,
        run_end_monotonic_ns=run_end_monotonic_ns,
    )
    if trace.get("valid") is not True:
        raise AnalysisError(
            "extended trace reader rejected the source: "
            + "; ".join(str(error) for error in trace.get("errors", ()))
        )
    rows = [_row_copy(row, index) for index, row in enumerate(trace.get("samples", []))]
    try:
        raw = safe_path.read_bytes()
    except OSError as exc:
        raise AnalysisError("extended trace source cannot be hashed") from exc
    source_metadata = {
        "file": safe_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    return rows, trace, source_metadata


def _load_trace_source(
    source: str | Path | Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load one trace and return rows, reader facts, and source provenance."""
    if isinstance(source, Mapping):
        return _load_mapping_trace(
            source,
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
        )
    path = _safe_cli_path(source, purpose="trace source")
    return _load_trace_file(
        path,
        run_start_monotonic_ns=run_start_monotonic_ns,
        run_end_monotonic_ns=run_end_monotonic_ns,
    )


def _left_held_segments(
    rows: Sequence[Mapping[str, Any]], *, start_ns: int, end_ns: int
) -> list[tuple[Mapping[str, Any], int, int]]:
    if start_ns >= end_ns:
        return []
    segments: list[tuple[Mapping[str, Any], int, int]] = []
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


def _duration_weighted_mean(
    segments: Sequence[tuple[Mapping[str, Any], int, int]], field: str
) -> float | None:
    duration = sum(end - start for _row, start, end in segments)
    if duration <= 0:
        return None
    weighted = math.fsum(
        float(row[field]) * ((end - start) / duration)
        for row, start, end in segments
    )
    return weighted if math.isfinite(weighted) else None


def _duration_weighted_percentile(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    field: str,
    percentile: float,
) -> float | None:
    if not 0.0 <= percentile <= 100.0:
        raise AnalysisError("percentile must be between 0 and 100")
    weighted_values = sorted(
        (float(row[field]), end - start) for row, start, end in segments if end > start
    )
    total_duration = sum(duration for _value, duration in weighted_values)
    if total_duration <= 0:
        return None
    if percentile <= 0.0:
        return weighted_values[0][0]
    target = total_duration * percentile / 100.0
    cumulative = 0
    for value, duration in weighted_values:
        cumulative += duration
        if cumulative >= target:
            return value
    return weighted_values[-1][0]


def _aggregate_signal(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    field: str,
    limit: float | None = None,
    limit_key: str | None = None,
    limit_unit: str | None = None,
) -> dict[str, Any]:
    values = [float(row[field]) for row, _start, _end in segments]
    minimum = min(values) if values else None
    maximum = max(values) if values else None
    weighted = _duration_weighted_mean(segments, field)
    proximity: dict[str, Any] | None = None
    if limit is not None and weighted is not None:
        if not math.isfinite(limit) or not limit:
            proximity = {"status": "not_evaluable", "reason": "limit is not finite or zero"}
        else:
            difference = abs(weighted - limit)
            proximity = {
                "status": "available",
                "limit_key": limit_key,
                "limit_value": limit,
                "limit_unit": limit_unit,
                "difference": difference,
                "fraction": difference / abs(limit),
            }
    elif limit_key is not None:
        proximity = {
            "status": "not_evaluable",
            "limit_key": limit_key,
            "limit_value": limit,
            "limit_unit": limit_unit,
            "reason": "signal or source-proven limit is unavailable",
        }
    return {
        "unit": SIGNAL_UNITS[field],
        "min": minimum,
        "max": maximum,
        "time_weighted_mean": weighted,
        "time_weighted": weighted,
        "limit_proximity": proximity,
    }


def _find_onset(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_ns: int,
    end_ns: int,
    field: str,
    predicate: Any,
) -> dict[str, Any]:
    for row, segment_start, segment_end in _left_held_segments(
        rows, start_ns=start_ns, end_ns=end_ns
    ):
        if predicate(float(row[field])):
            return {
                "timestamp_monotonic_ns": segment_start,
                "seconds_from_run_start": None,
                "field": field,
                "value": float(row[field]),
                "segment_end_monotonic_ns": segment_end,
            }
    return {
        "timestamp_monotonic_ns": None,
        "seconds_from_run_start": None,
        "field": field,
        "value": None,
        "segment_end_monotonic_ns": None,
    }


def _add_relative_onset(onset: dict[str, Any], run_start_ns: int) -> dict[str, Any]:
    timestamp = onset.get("timestamp_monotonic_ns")
    if timestamp is not None:
        onset["seconds_from_run_start"] = (timestamp - run_start_ns) / 1_000_000_000.0
    return onset


def _unavailable_onset(
    *, field: str, reason: str, threshold: float | None = None
) -> dict[str, Any]:
    """Represent a source-dependent onset that cannot be evaluated."""
    return {
        "status": "not_evaluable",
        "timestamp_monotonic_ns": None,
        "seconds_from_run_start": None,
        "field": field,
        "value": None,
        "threshold": threshold,
        "segment_end_monotonic_ns": None,
        "reason": reason,
    }


def _lower_proximity_bound(limit: float | None, fraction: float) -> float | None:
    if limit is None or not math.isfinite(limit) or limit <= 0.0:
        return None
    return limit * (1.0 - fraction)


def _raw_window(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_start_ns: int,
    duration_seconds: float = EARLY_RAW_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Return the raw signal rows in the first fixed seconds of the run."""
    end_ns = run_start_ns + round(duration_seconds * 1_000_000_000.0)
    raw_rows: list[dict[str, Any]] = []
    for row in rows:
        timestamp = int(row["timestamp_monotonic_ns"])
        if timestamp < run_start_ns or timestamp > end_ns:
            continue
        raw_rows.append(
            {
                "timestamp_monotonic_ns": timestamp,
                "seconds_from_run_start": (timestamp - run_start_ns) / 1_000_000_000.0,
                "aiclk_mhz": float(row["aiclk_mhz"]),
                "power_w": float(row["power_w"]),
                "current_a": float(row["current_a"]),
                "asic_temp_c": float(row["asic_temp_c"]),
            }
        )
    return {
        "start_seconds": 0.0,
        "end_seconds": duration_seconds,
        "columns": list(_RAW_WINDOW_FIELDS),
        "rows": raw_rows,
    }


_LIMIT_ALIASES = {
    "tdc_limit": ("tdc_limit", "tdc_limit_value", "TDC_LIMIT_MAX"),
    "therm_trip_l1_limit_c": (
        "therm_trip_l1_limit_c",
        "therm_trip_l1_limit",
        "THERM_TRIP_L1_LIMIT",
    ),
    "thm_shutdown_limit_c": (
        "thm_shutdown_limit_c",
        "thm_shutdown_limit",
        "thm_limit",
        "THM_LIMIT_SHUTDOWN",
    ),
    "tdp_limit_w": ("tdp_limit_w", "tdp_limit", "TDP_LIMIT_MAX"),
    "board_power_limit_w": (
        "board_power_limit_w",
        "board_power_limit",
        "BOARD_POWER_LIMIT",
    ),
    "thm_limit_throttle": ("thm_limit_throttle", "THM_LIMIT_THROTTLE"),
    "asic_fmax_mhz": ("asic_fmax_mhz", "asic_fmax", "ASIC_FMAX"),
}


def _snapshot_limit_mappings(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return persisted pre/post limit maps without inventing absent values."""
    snapshot_limits = metadata.get("snapshot_limits")
    if not isinstance(snapshot_limits, Mapping):
        return []
    mappings: list[Mapping[str, Any]] = []
    for stage in ("pre", "post"):
        stage_limits = snapshot_limits.get(stage)
        if not isinstance(stage_limits, Mapping):
            continue
        recognized = stage_limits.get("recognized")
        if isinstance(recognized, Mapping):
            mappings.append(recognized)
        for raw_name in ("raw", "raw_smbus", "raw_smbus_fields"):
            raw = stage_limits.get(raw_name)
            if isinstance(raw, Mapping):
                mappings.append(raw)
        mappings.append(stage_limits)
    return mappings


def _coerce_limit_value(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        return _finite(value, field=field)
    except AnalysisError:
        try:
            return _finite(int(str(value).strip(), 0), field=field)
        except (TypeError, ValueError):
            return None


def _apply_limit_value(
    name: str,
    keys: Sequence[str],
    source: Mapping[str, Any],
    values: dict[str, float | None],
    provided: set[str],
) -> None:
    for key in keys:
        if key not in source:
            continue
        value = source[key]
        if isinstance(value, Mapping):
            value = value.get("value")
        values[name] = _coerce_limit_value(value, field=name)
        if values[name] is not None:
            provided.add(name)
        break


def _apply_limit_source(
    source: Mapping[str, Any],
    values: dict[str, float | None],
    provided: set[str],
    *,
    only_unset: bool,
) -> None:
    for name, keys in _LIMIT_ALIASES.items():
        if only_unset and name in provided:
            continue
        _apply_limit_value(name, keys, source, values, provided)


def _direct_limit_sources(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    limits = metadata.get("limits")
    if isinstance(limits, Mapping):
        return [limits]
    direct_sources: list[Mapping[str, Any]] = []
    source_snapshot = metadata.get("source_snapshot")
    if isinstance(source_snapshot, Mapping):
        reported = source_snapshot.get("reported_limit_key_values")
        if isinstance(reported, Mapping):
            direct_sources.append(reported)
    direct_sources.append(metadata)
    return direct_sources


def _limits_from_metadata(metadata: Mapping[str, Any] | None) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "tdc_limit": H_TDC_LIMIT,
        "therm_trip_l1_limit_c": H_THM_LIMIT_C,
        "thm_shutdown_limit_c": H_THM_SHUTDOWN_C,
        "tdp_limit_w": H_TDP_LIMIT_W,
        "board_power_limit_w": BOARD_POWER_LIMIT_W,
        "thm_limit_throttle": None,
        # AICLK's baseline is source metadata, never a guessed default.
        "asic_fmax_mhz": None,
    }
    if not isinstance(metadata, Mapping):
        return values

    # A direct limits mapping wins over the persisted pre/post fallback.  This
    # preserves explicit record metadata while still making --record useful
    # for records that only carry snapshot_limits.
    provided: set[str] = set()
    for source in _direct_limit_sources(metadata):
        _apply_limit_source(source, values, provided, only_unset=False)
    for source in _snapshot_limit_mappings(metadata):
        _apply_limit_source(source, values, provided, only_unset=True)
    return values


def _hypothesis_result(
    *,
    plateau_pass: bool | None,
    onset_pass: bool | None,
    details: Mapping[str, Any],
    optional_criteria: Mapping[str, bool | None] | None = None,
    shutdown_pass: bool | None = None,
) -> dict[str, Any]:
    criteria: dict[str, bool | None] = {
        "plateau_proximity": plateau_pass,
        "onset_order": onset_pass,
    }
    if shutdown_pass is not None:
        criteria["shutdown_bound"] = shutdown_pass
    if optional_criteria:
        criteria.update(optional_criteria)
    required = tuple(criteria[key] for key in ("plateau_proximity", "onset_order"))
    if shutdown_pass is not None:
        required += (shutdown_pass,)
    if any(value is False for value in required):
        classification = "not_supported"
    elif any(value is None for value in required):
        classification = "inconclusive"
    else:
        classification = "supported"
    return {
        "classification": classification,
        "criteria": criteria,
        "details": dict(details),
        "causality": "correlation_only",
    }


@dataclass(frozen=True)
class _TraceWindow:
    rows: list[dict[str, Any]]
    facts: dict[str, Any]
    source_metadata: dict[str, Any]
    run_start: int
    run_end: int
    plateau_start: float
    plateau_end: float
    plateau_start_ns: int
    plateau_end_ns: int
    plateau_segments: list[tuple[Mapping[str, Any], int, int]]
    plateau_duration_ns: int
    expected_duration_ns: int


@dataclass(frozen=True)
class _HypothesisAnalysis:
    hypotheses: dict[str, Any]
    classification: str
    classification_reason: str
    onset_order: dict[str, Any]


def _prepare_trace_window(
    source: str | Path | Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
    plateau_start_seconds: float,
    plateau_end_seconds: float,
) -> _TraceWindow:
    plateau_start = _finite(plateau_start_seconds, field="plateau_start_seconds")
    plateau_end = _finite(plateau_end_seconds, field="plateau_end_seconds")
    if plateau_start < 0.0 or plateau_end <= plateau_start:
        raise AnalysisError("plateau interval must be positive and ordered")
    rows, facts, source_metadata = _load_trace_source(
        source,
        run_start_monotonic_ns=run_start_monotonic_ns,
        run_end_monotonic_ns=run_end_monotonic_ns,
    )
    if any(
        int(left["timestamp_monotonic_ns"]) > int(right["timestamp_monotonic_ns"])
        for left, right in itertools.pairwise(rows)
    ):
        raise AnalysisError("trace monotonic timestamps are not ordered")
    run_start = int(facts["run_start_monotonic_ns"])
    run_end = int(facts["run_end_monotonic_ns"])
    plateau_start_ns = run_start + round(plateau_start * 1_000_000_000.0)
    plateau_end_ns = run_start + round(plateau_end * 1_000_000_000.0)
    if run_end < plateau_end_ns:
        raise AnalysisError("trace does not cover the fixed plateau end")
    plateau_segments = _left_held_segments(
        rows,
        start_ns=plateau_start_ns,
        end_ns=plateau_end_ns,
    )
    plateau_duration_ns = sum(end - start for _row, start, end in plateau_segments)
    expected_duration_ns = plateau_end_ns - plateau_start_ns
    if plateau_duration_ns != expected_duration_ns:
        raise AnalysisError("trace has a gap in the fixed plateau interval")
    return _TraceWindow(
        rows=rows,
        facts=facts,
        source_metadata=source_metadata,
        run_start=run_start,
        run_end=run_end,
        plateau_start=plateau_start,
        plateau_end=plateau_end,
        plateau_start_ns=plateau_start_ns,
        plateau_end_ns=plateau_end_ns,
        plateau_segments=plateau_segments,
        plateau_duration_ns=plateau_duration_ns,
        expected_duration_ns=expected_duration_ns,
    )


def _aggregate_signals(
    plateau_segments: Sequence[tuple[Mapping[str, Any], int, int]],
    source_limits: Mapping[str, float | None],
) -> dict[str, Any]:
    signal_limits: dict[str, tuple[float | None, str | None, str | None]] = {
        "power_w": (source_limits["tdp_limit_w"], "TDP_LIMIT_MAX", "W"),
        "board_power_w": (source_limits["board_power_limit_w"], "BOARD_POWER_LIMIT", "W"),
        "aiclk_mhz": (None, None, None),
        "current_a": (source_limits["tdc_limit"], "TDC_LIMIT_MAX", "reported snapshot unit"),
        "voltage_v": (None, None, None),
        "asic_temp_c": (
            source_limits["therm_trip_l1_limit_c"],
            "therm_trip_l1_limit",
            "degC",
        ),
    }
    signals: dict[str, Any] = {}
    for field in _SIGNAL_FIELDS:
        limit, limit_key, limit_unit = signal_limits[field]
        signals[field] = _aggregate_signal(
            plateau_segments,
            field=field,
            limit=limit,
            limit_key=limit_key,
            limit_unit=limit_unit,
        )
    return signals


def _aiclk_onset(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_ns: int,
    end_ns: int,
    asic_fmax: float | None,
) -> tuple[dict[str, Any], float | None]:
    threshold = _lower_proximity_bound(asic_fmax, AICLK_DROP_FRACTION)
    if threshold is None:
        onset = _unavailable_onset(
            field="aiclk_mhz",
            threshold=None,
            reason=(
                "limits metadata did not establish asic_fmax_mhz; "
                "AICLK drop onset is unknown"
            ),
        )
        onset.update(
            {
                "baseline_source": AICLK_BASELINE_SOURCE,
                "baseline_asic_fmax_mhz": None,
            }
        )
        return onset, None
    # Do not let a pre-start sample supply the first in-run AICLK segment.
    aiclk_rows = tuple(row for row in rows if int(row["timestamp_monotonic_ns"]) >= start_ns)
    onset = _find_onset(
        aiclk_rows,
        start_ns=start_ns,
        end_ns=end_ns,
        field="aiclk_mhz",
        predicate=lambda value: value <= threshold,
    )
    onset.update(
        {
            "status": (
                "observed"
                if onset["timestamp_monotonic_ns"] is not None
                else "not_observed"
            ),
            "baseline_source": AICLK_BASELINE_SOURCE,
            "baseline_asic_fmax_mhz": asic_fmax,
            "threshold_mhz": threshold,
        }
    )
    return onset, threshold


def _threshold_onset(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_ns: int,
    end_ns: int,
    field: str,
    threshold: float | None,
    unavailable_reason: str,
    predicate: Any,
) -> dict[str, Any]:
    if threshold is None:
        return _unavailable_onset(
            field=field,
            threshold=None,
            reason=unavailable_reason,
        )
    return _find_onset(
        rows,
        start_ns=start_ns,
        end_ns=end_ns,
        field=field,
        predicate=predicate,
    )


def _build_onset_data(
    window: _TraceWindow, source_limits: Mapping[str, float | None]
) -> dict[str, Any]:
    aiclk_drop, aiclk_threshold = _aiclk_onset(
        window.rows,
        start_ns=window.run_start,
        end_ns=window.run_end,
        asic_fmax=source_limits["asic_fmax_mhz"],
    )
    current_onset = _find_onset(
        window.rows,
        start_ns=window.run_start,
        end_ns=window.run_end,
        field="current_a",
        predicate=lambda value: value >= H_TDC_ONSET_THRESHOLD,
    )
    therm_trip_limit = source_limits["therm_trip_l1_limit_c"]
    thm_onset_threshold = (
        None if therm_trip_limit is None else therm_trip_limit - H_THM_MAX_DIFFERENCE_C
    )
    temperature_onset = _threshold_onset(
        window.rows,
        start_ns=window.run_start,
        end_ns=window.run_end,
        field="asic_temp_c",
        threshold=thm_onset_threshold,
        unavailable_reason="therm_trip_l1_limit_c is unavailable",
        predicate=lambda value: value >= thm_onset_threshold,
    )
    tdp_onset_threshold = _lower_proximity_bound(
        source_limits["tdp_limit_w"], H_TDP_PROXIMITY_FRACTION
    )
    power_onset = _threshold_onset(
        window.rows,
        start_ns=window.run_start,
        end_ns=window.run_end,
        field="power_w",
        threshold=tdp_onset_threshold,
        unavailable_reason="tdp_limit_w is unavailable",
        predicate=lambda value: value >= tdp_onset_threshold,
    )
    onsets = (aiclk_drop, current_onset, temperature_onset, power_onset)
    for onset in onsets:
        _add_relative_onset(onset, window.run_start)
    if tdp_onset_threshold is not None:
        power_onset["threshold"] = tdp_onset_threshold
    if thm_onset_threshold is not None:
        temperature_onset["threshold"] = thm_onset_threshold
    return {
        "aiclk_drop": aiclk_drop,
        "current_onset": current_onset,
        "temperature_onset": temperature_onset,
        "power_onset": power_onset,
        "aiclk_threshold": aiclk_threshold,
        "thm_onset_threshold": thm_onset_threshold,
        "tdp_onset_threshold": tdp_onset_threshold,
        "onsets": {
            "aiclk_drop": aiclk_drop,
            "current_tdc_threshold": current_onset,
            "temperature_thm_threshold": temperature_onset,
            "power_tdp_threshold": power_onset,
        },
    }


def _onset_order_pass(
    signal_onset: Mapping[str, Any], aiclk_drop: Mapping[str, Any]
) -> bool | None:
    signal_timestamp = signal_onset.get("timestamp_monotonic_ns")
    clock_timestamp = aiclk_drop.get("timestamp_monotonic_ns")
    if clock_timestamp is None:
        return None
    # A known AICLK drop plus no signal onset is a failed onset criterion,
    # not missing evidence for the hypothesis.
    if signal_timestamp is None:
        return False
    return int(signal_timestamp) <= int(clock_timestamp)


def _throttle_comparison(
    throttle_limit: float | None, thm_maximum: float | None
) -> dict[str, Any]:
    if throttle_limit is None:
        return {
            "status": "not_evaluable",
            "knowledge_status": "unknown",
            "limit_key": "THM_LIMIT_THROTTLE",
            "limit_value": None,
            "limit_unit": "degC",
            "reason": (
                "source snapshot did not establish THM_LIMIT_THROTTLE; "
                "no value was inferred from therm_trip_l1"
            ),
        }
    if thm_maximum is None:
        return {
            "status": "not_evaluable",
            "limit_key": "THM_LIMIT_THROTTLE",
            "limit_value": throttle_limit,
            "limit_unit": "degC",
            "reason": "ASIC temperature plateau maximum is unavailable",
        }
    throttle_pass = thm_maximum < throttle_limit
    return {
        "status": "pass" if throttle_pass else "fail",
        "limit_key": "THM_LIMIT_THROTTLE",
        "limit_value": throttle_limit,
        "limit_unit": "degC",
        "plateau_asic_temp_max_c": thm_maximum,
        "criterion": (
            "run plateau maximum ASIC temperature is strictly below "
            "THM_LIMIT_THROTTLE"
        ),
    }


def _hypothesis_metrics(
    signals: Mapping[str, Any],
    onset_data: Mapping[str, Any],
    source_limits: Mapping[str, float | None],
) -> dict[str, Any]:
    aiclk_drop = onset_data["aiclk_drop"]
    current_onset = onset_data["current_onset"]
    temperature_onset = onset_data["temperature_onset"]
    power_onset = onset_data["power_onset"]
    tdc_proximity = signals["current_a"]["limit_proximity"]
    tdc_plateau_pass = (
        tdc_proximity is not None
        and tdc_proximity.get("status") == "available"
        and tdc_proximity["fraction"] <= H_TDC_PROXIMITY_FRACTION
    )
    thm_weighted = signals["asic_temp_c"]["time_weighted_mean"]
    thm_maximum = signals["asic_temp_c"]["max"]
    therm_trip_limit = source_limits["therm_trip_l1_limit_c"]
    thm_plateau_pass = (
        thm_weighted is not None
        and therm_trip_limit is not None
        and abs(thm_weighted - therm_trip_limit) <= H_THM_MAX_DIFFERENCE_C
    )
    thm_shutdown_limit = source_limits["thm_shutdown_limit_c"]
    thm_shutdown_pass = (
        thm_maximum is not None
        and thm_shutdown_limit is not None
        and thm_maximum < thm_shutdown_limit
    )
    thm_throttle_comparison = _throttle_comparison(
        source_limits["thm_limit_throttle"], thm_maximum
    )
    tdp_signal = signals["power_w"]
    tdp_proximity = tdp_signal["limit_proximity"]
    tdp_lower_bound = _lower_proximity_bound(
        source_limits["tdp_limit_w"], H_TDP_PROXIMITY_FRACTION
    )
    tdp_plateau_pass = (
        tdp_signal["time_weighted_mean"] is not None
        and tdp_lower_bound is not None
        and tdp_signal["time_weighted_mean"] >= tdp_lower_bound
        and tdp_proximity is not None
        and tdp_proximity.get("status") == "available"
        and tdp_proximity["fraction"] <= H_TDP_PROXIMITY_FRACTION
    )
    board_power_proximity = signals["board_power_w"]["limit_proximity"]
    known_signal_onsets = (current_onset, temperature_onset, power_onset)
    aiclk_timestamp = aiclk_drop.get("timestamp_monotonic_ns")
    h_other_order_pass = (
        None
        if aiclk_timestamp is None
        else all(
            onset.get("timestamp_monotonic_ns") is None
            or int(onset["timestamp_monotonic_ns"]) > int(aiclk_timestamp)
            for onset in known_signal_onsets
        )
    )
    tdc_order_pass = _onset_order_pass(current_onset, aiclk_drop)
    thm_order_pass = _onset_order_pass(temperature_onset, aiclk_drop)
    tdp_order_pass = _onset_order_pass(power_onset, aiclk_drop)
    throttle_criterion_pass = {"pass": True, "fail": False}.get(
        thm_throttle_comparison["status"]
    )
    return {
        "aiclk_drop": aiclk_drop,
        "current_onset": current_onset,
        "temperature_onset": temperature_onset,
        "power_onset": power_onset,
        "tdc_proximity": tdc_proximity,
        "tdc_plateau_pass": tdc_plateau_pass,
        "thm_weighted": thm_weighted,
        "thm_maximum": thm_maximum,
        "therm_trip_limit": therm_trip_limit,
        "thm_plateau_pass": thm_plateau_pass,
        "thm_shutdown_limit": thm_shutdown_limit,
        "thm_shutdown_pass": thm_shutdown_pass,
        "thm_throttle_comparison": thm_throttle_comparison,
        "tdp_signal": tdp_signal,
        "tdp_proximity": tdp_proximity,
        "tdp_lower_bound": tdp_lower_bound,
        "tdp_plateau_pass": tdp_plateau_pass,
        "board_power_proximity": board_power_proximity,
        "known_signal_onsets": known_signal_onsets,
        "h_other_order_pass": h_other_order_pass,
        "tdc_order_pass": tdc_order_pass,
        "thm_order_pass": thm_order_pass,
        "tdp_order_pass": tdp_order_pass,
        "throttle_criterion_pass": throttle_criterion_pass,
    }


def _build_hypothesis_analysis(
    signals: Mapping[str, Any],
    onset_data: Mapping[str, Any],
    source_limits: Mapping[str, float | None],
) -> _HypothesisAnalysis:
    metrics = _hypothesis_metrics(signals, onset_data, source_limits)
    aiclk_drop = metrics["aiclk_drop"]
    current_onset = metrics["current_onset"]
    temperature_onset = metrics["temperature_onset"]
    power_onset = metrics["power_onset"]
    tdc_plateau_pass = metrics["tdc_plateau_pass"]
    thm_weighted = metrics["thm_weighted"]
    therm_trip_limit = metrics["therm_trip_limit"]
    thm_plateau_pass = metrics["thm_plateau_pass"]
    thm_shutdown_limit = metrics["thm_shutdown_limit"]
    thm_shutdown_pass = metrics["thm_shutdown_pass"]
    thm_throttle_comparison = metrics["thm_throttle_comparison"]
    tdp_signal = metrics["tdp_signal"]
    tdp_proximity = metrics["tdp_proximity"]
    tdp_lower_bound = metrics["tdp_lower_bound"]
    tdp_plateau_pass = metrics["tdp_plateau_pass"]
    board_power_proximity = metrics["board_power_proximity"]
    known_signal_onsets = metrics["known_signal_onsets"]
    h_other_order_pass = metrics["h_other_order_pass"]
    tdc_order_pass = metrics["tdc_order_pass"]
    thm_order_pass = metrics["thm_order_pass"]
    tdp_order_pass = metrics["tdp_order_pass"]
    throttle_criterion_pass = metrics["throttle_criterion_pass"]
    hypotheses = {
        "H-TDC": _hypothesis_result(
            plateau_pass=tdc_plateau_pass,
            onset_pass=tdc_order_pass,
            details={
                "signal": "current_a",
                "threshold": H_TDC_ONSET_THRESHOLD,
                "limit": source_limits["tdc_limit"],
                "signal_onset": current_onset,
                "aiclk_drop_onset": aiclk_drop,
                "onset_order_pass": tdc_order_pass,
            },
        ),
        "H-THM": _hypothesis_result(
            plateau_pass=thm_plateau_pass,
            onset_pass=thm_order_pass,
            shutdown_pass=thm_shutdown_pass,
            optional_criteria={"thm_limit_throttle": throttle_criterion_pass},
            details={
                "signal": "asic_temp_c",
                "threshold": onset_data["thm_onset_threshold"],
                "therm_trip_l1_limit_c": therm_trip_limit,
                "therm_trip_proximity": signals["asic_temp_c"]["limit_proximity"],
                "plateau_time_weighted_mean_c": thm_weighted,
                "thm_shutdown_limit_c": thm_shutdown_limit,
                "signal_onset": temperature_onset,
                "aiclk_drop_onset": aiclk_drop,
                "onset_order_pass": thm_order_pass,
                "shutdown_bound_pass": thm_shutdown_pass,
                "thm_limit_throttle": thm_throttle_comparison,
                "criteria_note": (
                    "THM_LIMIT_THROTTLE is reported as an optional snapshot-limit "
                    "comparison and does not replace therm_trip_l1/shutdown criteria"
                ),
            },
        ),
        "H-TDP": _hypothesis_result(
            plateau_pass=tdp_plateau_pass,
            onset_pass=tdp_order_pass,
            details={
                "signal": "power_w",
                "limit_w": source_limits["tdp_limit_w"],
                "tdp_limit_proximity": tdp_proximity,
                "plateau_time_weighted_mean_w": tdp_signal["time_weighted_mean"],
                "proximity_fraction": H_TDP_PROXIMITY_FRACTION,
                "lower_bound_w": tdp_lower_bound,
                "criterion": (
                    "power_w plateau time-weighted mean is at least "
                    "TDP_LIMIT_MAX * (1 - 0.05) and within 5 percent of "
                    "TDP_LIMIT_MAX; its onset is no later than AICLK drop onset"
                ),
                "reference": "Issue #118 plateau 132-135 W versus TDP 150 W (prediction/reference only)",
                "board_power_signal": "board_power_w",
                "board_power_limit_w": source_limits["board_power_limit_w"],
                "board_power_proximity_fraction": H_TDP_PROXIMITY_FRACTION,
                "board_power_proximity": board_power_proximity,
                "board_power_plateau_time_weighted_mean_w": signals["board_power_w"][
                    "time_weighted_mean"
                ],
                "board_power_criterion": (
                    "report board_power_w plateau proximity to BOARD_POWER_LIMIT "
                    "using the same 5 percent limit-near form; this is not the "
                    "H-TDP signal criterion"
                ),
                "signal_onset": power_onset,
                "aiclk_drop_onset": aiclk_drop,
                "onset_order_pass": tdp_order_pass,
            },
        ),
        "H-OTHER": _hypothesis_result(
            plateau_pass=None,
            onset_pass=h_other_order_pass,
            details={
                "fallback_only_when_no_known_hypothesis_is_supported": True,
                "signal_onsets": list(known_signal_onsets),
                "aiclk_drop_onset": aiclk_drop,
                "onset_order_pass": h_other_order_pass,
            },
        ),
    }
    hypotheses["H-OTHER"]["details"]["known_hypothesis_classifications"] = {
        name: hypotheses[name]["classification"]
        for name in ("H-TDC", "H-THM", "H-TDP")
    }
    supported = [
        name
        for name, result in hypotheses.items()
        if name != "H-OTHER" and result["classification"] == "supported"
    ]
    if len(supported) == 1:
        classification = supported[0]
        classification_reason = "exactly one known-limit hypothesis satisfies all fixed criteria"
    elif len(supported) > 1:
        classification = "multiple"
        classification_reason = "more than one known-limit hypothesis satisfies the fixed criteria"
    else:
        classification = "H-OTHER"
        classification_reason = "no known-limit hypothesis satisfies every fixed criterion"
    return _HypothesisAnalysis(
        hypotheses=hypotheses,
        classification=classification,
        classification_reason=classification_reason,
        onset_order={
            "H-TDC": {
                "signal": current_onset,
                "aiclk_drop": aiclk_drop,
                "pass": tdc_order_pass,
            },
            "H-THM": {
                "signal": temperature_onset,
                "aiclk_drop": aiclk_drop,
                "pass": thm_order_pass,
            },
            "H-TDP": {
                "signal": power_onset,
                "aiclk_drop": aiclk_drop,
                "pass": tdp_order_pass,
            },
        },
    )


def _build_analysis_report(
    window: _TraceWindow,
    source_limits: Mapping[str, float | None],
    signals: Mapping[str, Any],
    onset_data: Mapping[str, Any],
    hypothesis_analysis: _HypothesisAnalysis,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_metadata = window.source_metadata
    source_report = {
        "schema": window.facts.get("schema", EXTENDED_POWER_TRACE_SCHEMA),
        "schema_version": window.facts.get(
            "schema_version", EXTENDED_POWER_TRACE_SCHEMA_VERSION
        ),
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "file": source_metadata.get("file") if source_metadata else None,
        "sha256": source_metadata.get("sha256") if source_metadata else None,
        "source_metadata": _safe_provenance(source_metadata),
        "provenance": _safe_provenance(provenance or {}),
    }
    raw_window = _raw_window(window.rows, run_start_ns=window.run_start)
    limitations = [
        (
            "The extended trace samples at a 2 second cadence; onset ordering "
            "has approximately 2 second temporal resolution."
        ),
        "Onset timestamps are starts of left-held sample segments, not interpolated events.",
    ]
    if source_limits["asic_fmax_mhz"] is None:
        limitations.append(
            "AICLK drop onset is not evaluable because limits metadata omitted asic_fmax_mhz."
        )
    return {
        "analysis_schema": ANALYSIS_SCHEMA,
        "status": "ok",
        "classification": hypothesis_analysis.classification,
        "classification_reason": hypothesis_analysis.classification_reason,
        "causality": "correlation_is_not_causation",
        "source": source_report,
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "units": dict(SIGNAL_UNITS),
        "time_axis": {
            "column": "timestamp_monotonic_ns",
            "semantics": "left_held_positive_duration_segments",
            "run_start_monotonic_ns": window.run_start,
            "run_end_monotonic_ns": window.run_end,
            "sampling_cadence_seconds": SAMPLING_CADENCE_SECONDS,
        },
        "aiclk_baseline": {
            "source": AICLK_BASELINE_SOURCE,
            "asic_fmax_mhz": source_limits["asic_fmax_mhz"],
            "drop_fraction": AICLK_DROP_FRACTION,
            "drop_threshold_mhz": onset_data["aiclk_threshold"],
            "missing_source_status": (
                "not_evaluable" if source_limits["asic_fmax_mhz"] is None else None
            ),
        },
        "plateau": {
            "start_seconds": window.plateau_start,
            "end_seconds": window.plateau_end,
            "start_monotonic_ns": window.plateau_start_ns,
            "end_monotonic_ns": window.plateau_end_ns,
            "duration_seconds": window.expected_duration_ns / 1_000_000_000.0,
            "segment_count": len(window.plateau_segments),
            "coverage_complete": window.plateau_duration_ns == window.expected_duration_ns,
        },
        "statistics": {
            "aiclk_mhz": {
                "min": signals["aiclk_mhz"]["min"],
                "max": signals["aiclk_mhz"]["max"],
                "time_weighted_mean": signals["aiclk_mhz"]["time_weighted_mean"],
                "percentiles": {
                    f"p{percentile:02d}": _duration_weighted_percentile(
                        window.plateau_segments, "aiclk_mhz", percentile
                    )
                    for percentile in (1, 5, 50, 95, 99)
                },
                "percentile_definition": "duration-weighted nearest-rank over left-held segments",
            },
            "signals": signals,
        },
        "onsets": onset_data["onsets"],
        "onset_order": hypothesis_analysis.onset_order,
        "hypotheses": hypothesis_analysis.hypotheses,
        "limits": _safe_provenance(source_limits),
        "raw_window": raw_window,
        "limitations": limitations,
        "reader_facts": _safe_provenance(window.facts),
    }


def analyze_trace(
    source: str | Path | Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    plateau_start_seconds: float = DEFAULT_PLATEAU_START_SECONDS,
    plateau_end_seconds: float = DEFAULT_PLATEAU_END_SECONDS,
    provenance: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply every fixed Q2 plateau and onset criterion to one trace."""
    window = _prepare_trace_window(
        source,
        run_start_monotonic_ns=run_start_monotonic_ns,
        run_end_monotonic_ns=run_end_monotonic_ns,
        plateau_start_seconds=plateau_start_seconds,
        plateau_end_seconds=plateau_end_seconds,
    )
    source_limits = _limits_from_metadata(limits)
    signals = _aggregate_signals(window.plateau_segments, source_limits)
    onset_data = _build_onset_data(window, source_limits)
    hypothesis_analysis = _build_hypothesis_analysis(
        signals, onset_data, source_limits
    )
    return _build_analysis_report(
        window,
        source_limits,
        signals,
        onset_data,
        hypothesis_analysis,
        provenance,
    )


def analyze_extended_trace(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Discoverable alias for callers that name the extended trace explicitly."""
    return analyze_trace(*args, **kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-start-monotonic-ns", type=int, default=None)
    parser.add_argument("--run-end-monotonic-ns", type=int, default=None)
    parser.add_argument("--plateau-start-seconds", type=float, default=DEFAULT_PLATEAU_START_SECONDS)
    parser.add_argument("--plateau-end-seconds", type=float, default=DEFAULT_PLATEAU_END_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        provenance = None
        limits = None
        record_run_start = None
        record_run_end = None
        if args.record is not None:
            record_path = _safe_cli_path(args.record, purpose="record")
            record_payload = json.loads(record_path.read_text(encoding="utf-8"))
            if isinstance(record_payload, Mapping):
                provenance = {
                    "record": record_payload.get("provenance", {}),
                    "trace": record_payload.get("trace", {}),
                    "environment": record_payload.get("environment", {}),
                    "snapshot_limits": record_payload.get("snapshot_limits", {}),
                }
                environment = record_payload.get("environment")
                environment_limits = (
                    environment.get("limits") if isinstance(environment, Mapping) else None
                )
                snapshot_limits = record_payload.get("snapshot_limits")
                limits = {
                    "limits": environment_limits,
                    "snapshot_limits": snapshot_limits,
                }
                timing = record_payload.get("run_timing")
                if isinstance(timing, Mapping):
                    record_run_start = timing.get("workload_start_monotonic_ns")
                    record_run_end = timing.get("workload_end_monotonic_ns")
        result = analyze_trace(
            args.trace,
            run_start_monotonic_ns=(
                args.run_start_monotonic_ns
                if args.run_start_monotonic_ns is not None
                else record_run_start
            ),
            run_end_monotonic_ns=(
                args.run_end_monotonic_ns
                if args.run_end_monotonic_ns is not None
                else record_run_end
            ),
            plateau_start_seconds=args.plateau_start_seconds,
            plateau_end_seconds=args.plateau_end_seconds,
            provenance=provenance,
            limits=limits,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"Issue #121 Q2 analysis rejected: {exc}", file=__import__("sys").stderr)
        return 2
    serialized = strict_json_dumps(result, indent=2) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        output_path = _safe_cli_path(args.output, purpose="analysis output", output=True)
        output_path.write_text(serialized, encoding="utf-8")
    return 0


__all__ = [
    "ANALYSIS_SCHEMA",
    "DEFAULT_PLATEAU_END_SECONDS",
    "DEFAULT_PLATEAU_START_SECONDS",
    "AnalysisError",
    "analyze_extended_trace",
    "analyze_trace",
]


if __name__ == "__main__":
    raise SystemExit(main())
