"""Versioned board-free extended power-trace support for Issue #121 Q2.

The established four-column trace remains owned by :mod:`telemetry` and is
not changed here.  This module provides an opt-in sidecar whose first four
columns are that trace's columns, followed by the ten Issue #121 fields.
Snapshot values are read from the snapshot's normalized ``telemetry`` object;
raw SMBus values are used only for raw fields such as ``THERM_TRIP_COUNT``.
"""

from __future__ import annotations

import csv
import datetime
import io
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from enodia.tt.bench.board_identity import (
    normalize_board_id,
    normalize_pci_bus_id,
    resolve_board_identity,
)

# Keep this tuple independent of telemetry.py so the extended module can be
# imported by telemetry.py without a circular import.  Its order is the
# established legacy order and is deliberately the prefix of the sidecar.
_LEGACY_POWER_TRACE_COLUMNS = (
    "timestamp_utc",
    "power_w",
    "aiclk_mhz",
    "asic_temp_c",
)
_EXTENDED_SIDECAR_COLUMNS = (
    "timestamp_monotonic_ns",
    "snapshot_elapsed_ns",
    "current_a",
    "voltage_v",
    "board_power_w",
    "fan_rpm",
    "heartbeat",
    "therm_trip_count",
    "throttle_status",
    "throttle_reason",
)

# The schema name is separate from the legacy POWER_TRACE_COLUMNS contract.
EXTENDED_POWER_TRACE_SCHEMA = "issue-121-q2-power-trace-v1"
EXTENDED_POWER_TRACE_SCHEMA_VERSION = 1
EXTENDED_POWER_TRACE_COLUMNS = (
    *_LEGACY_POWER_TRACE_COLUMNS,
    *_EXTENDED_SIDECAR_COLUMNS,
)
EXTENDED_POWER_TRACE_HEADER = ",".join(EXTENDED_POWER_TRACE_COLUMNS)

# Descriptive aliases make the versioned seam discoverable without changing
# the legacy POWER_TRACE_COLUMNS name.
POWER_TRACE_EXTENDED_SCHEMA = EXTENDED_POWER_TRACE_SCHEMA
POWER_TRACE_EXTENDED_SCHEMA_VERSION = EXTENDED_POWER_TRACE_SCHEMA_VERSION
POWER_TRACE_EXTENDED_COLUMNS = EXTENDED_POWER_TRACE_COLUMNS

# Numeric values are the fields for which min/max/time-weighted aggregation is
# meaningful.  Both timestamp columns are intentionally absent.  The
# snapshot duration is a measured numeric value, not a timestamp column.
_EXTENDED_MEASUREMENT_COLUMNS = (
    "power_w",
    "aiclk_mhz",
    "asic_temp_c",
    "snapshot_elapsed_ns",
    "current_a",
    "voltage_v",
    "board_power_w",
    "fan_rpm",
    "heartbeat",
    "therm_trip_count",
)
_EXTENDED_INTEGER_COLUMNS = (
    "timestamp_monotonic_ns",
    "snapshot_elapsed_ns",
    "fan_rpm",
    "heartbeat",
    "therm_trip_count",
)
_EXTENDED_NULLABLE_COLUMNS = ("throttle_status", "throttle_reason")

# These are the normalized fields emitted under device_info[].telemetry.  In
# particular, board_power_w has no power fallback: a board-power measurement
# must be present as ``telemetry.board_power``.
_NORMALIZED_TELEMETRY_KEYS = {
    "power_w": "power",
    "aiclk_mhz": "aiclk",
    "asic_temp_c": "asic_temperature",
    "current_a": "current",
    "voltage_v": "voltage",
    "board_power_w": "board_power",
    "fan_rpm": "fan_speed",
    "heartbeat": "heartbeat",
}

# Raw SMBus names are retained as source names.  Hexadecimal integer strings
# are converted to numbers, but their units and meaning are not inferred.
_RAW_THERM_TRIP_KEYS = (
    "THERM_TRIP_COUNT",
    "therm_trip_count",
    "thermal_trip_count",
    "thermal_trips",
)
_RAW_THROTTLE_STATUS_KEYS = (
    "THROTTLE_STATUS",
    "throttle_status",
    "throttling",
    "throttle",
)
_RAW_THROTTLE_REASON_KEYS = ("THROTTLE_REASON", "throttle_reason")
_MAX_NONNEGATIVE_INTEGER = (1 << 63) - 1
_MAX_EXACT_FLOAT_INTEGER = (1 << 53) - 1
_MISSING = object()
_TRACE_BOUNDS_ORDER_ERROR = (
    "run_start_monotonic_ns must be at or before run_end_monotonic_ns"
)


class ExtendedPowerTraceError(ValueError):
    """Raised by strict helpers when an extended trace is not usable."""


def _as_snapshot_object(snapshot: str | bytes | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(snapshot, Mapping):
        return snapshot
    if isinstance(snapshot, bytes):
        try:
            snapshot = snapshot.decode("utf-8")
        except UnicodeError as exc:
            raise ExtendedPowerTraceError("snapshot is not valid JSON") from exc
    if not isinstance(snapshot, str):
        raise ExtendedPowerTraceError("snapshot must be a JSON object")
    try:
        value = json.loads(snapshot)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ExtendedPowerTraceError("snapshot is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ExtendedPowerTraceError("snapshot must be a JSON object")
    return value


def _identity_text(value: Any, *, field: str) -> str:
    text = normalize_board_id(value)
    if text is None:
        raise ExtendedPowerTraceError(f"{field} must be a non-empty string")
    return text


def _normalize_pci_bus_id(value: object) -> str | None:
    return normalize_pci_bus_id(value)


def _required_pci_bus_id(value: Any) -> str:
    normalized = _normalize_pci_bus_id(value)
    if normalized is None:
        raise ExtendedPowerTraceError("bus_id must be a valid PCI BDF")
    return normalized


def _board_entries(root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    devices = root.get("device_info")
    if not isinstance(devices, list) or not devices:
        raise ExtendedPowerTraceError("snapshot device_info must be a non-empty array")
    entries: list[Mapping[str, Any]] = []
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping):
            raise ExtendedPowerTraceError(f"device_info[{index}] must be an object")
        entries.append(device)
    return entries


def is_extended_snapshot_well_formed(
    snapshot: str | bytes | Mapping[str, Any],
) -> bool:
    """Return whether a snapshot has a usable object-shaped device list."""
    try:
        root = _as_snapshot_object(snapshot)
        _board_entries(root)
    except ExtendedPowerTraceError:
        return False
    return True


def _requested_extended_identity(
    board_id: str | None, bus_id: str | None
) -> tuple[str | None, str | None]:
    try:
        return resolve_board_identity(board_id, bus_id)
    except ValueError as exc:
        raise ExtendedPowerTraceError(str(exc)) from exc


def _extended_board_matches(
    devices: Sequence[Mapping[str, Any]],
    *,
    requested_board_id: str,
    requested_bus_id: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Collect independent and paired matches for explicit board identities."""
    board_matches: list[Mapping[str, Any]] = []
    bus_matches: list[Mapping[str, Any]] = []
    pair_matches: list[Mapping[str, Any]] = []
    for device in devices:
        board_info = device.get("board_info")
        if not isinstance(board_info, Mapping):
            continue
        candidate_board_id = normalize_board_id(board_info.get("board_id"))
        candidate_bus_id = _normalize_pci_bus_id(board_info.get("bus_id"))
        board_matches_identity = candidate_board_id == requested_board_id
        bus_matches_identity = candidate_bus_id == requested_bus_id
        if board_matches_identity:
            board_matches.append(device)
        if bus_matches_identity:
            bus_matches.append(device)
        if board_matches_identity and bus_matches_identity:
            pair_matches.append(device)
    return board_matches, bus_matches, pair_matches


def select_extended_board(
    snapshot: str | bytes | Mapping[str, Any],
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> Mapping[str, Any]:
    """Select exactly one board by its explicit board and PCI identities.

    Both identities are required.  A duplicate board ID, duplicate bus ID, or
    duplicate pair is rejected even when one candidate would otherwise match.
    No array position or ``device_id`` value participates.
    """
    root = _as_snapshot_object(snapshot)
    board_id, bus_id = _requested_extended_identity(board_id, bus_id)
    requested_board_id = _identity_text(board_id, field="board_id")
    requested_bus_id = _required_pci_bus_id(_identity_text(bus_id, field="bus_id"))
    devices = _board_entries(root)
    board_matches, bus_matches, pair_matches = _extended_board_matches(
        devices,
        requested_board_id=requested_board_id,
        requested_bus_id=requested_bus_id,
    )

    if len(board_matches) != 1:
        raise ExtendedPowerTraceError(
            "board_id must identify exactly one device_info entry"
        )
    if len(bus_matches) != 1:
        raise ExtendedPowerTraceError("bus_id must identify exactly one device_info entry")
    if len(pair_matches) != 1:
        raise ExtendedPowerTraceError(
            "board_id and bus_id must identify exactly one device_info entry"
        )
    if pair_matches[0] is not board_matches[0] or pair_matches[0] is not bus_matches[0]:
        raise ExtendedPowerTraceError("board_id and bus_id identify different boards")
    return pair_matches[0]


def _normalized_telemetry(device: Mapping[str, Any]) -> Mapping[str, Any]:
    telemetry = device.get("telemetry")
    if not isinstance(telemetry, Mapping):
        raise ExtendedPowerTraceError("selected device telemetry must be an object")
    return telemetry


def _raw_smbus_telemetry(device: Mapping[str, Any]) -> Mapping[str, Any]:
    smbus_telem = device.get("smbus_telem")
    if smbus_telem is None:
        return {}
    if not isinstance(smbus_telem, Mapping):
        raise ExtendedPowerTraceError("selected device smbus_telem must be an object")
    return smbus_telem


def _mapping_value(
    source: Mapping[str, Any], key: str, *, field: str
) -> Any:
    value = source.get(key, _MISSING)
    if value is _MISSING or value is None:
        raise ExtendedPowerTraceError(f"required snapshot value {field!r} is missing")
    return value


def _first_mapping_value(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return _MISSING


def _parse_numeric(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ExtendedPowerTraceError(f"{field} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        if isinstance(value, str):
            text = value.strip()
            try:
                parsed = float(int(text, 0))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ExtendedPowerTraceError(f"{field} must be numeric") from exc
        else:
            raise ExtendedPowerTraceError(f"{field} must be numeric")
    if not math.isfinite(parsed) or (positive and parsed <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise ExtendedPowerTraceError(f"{field} must be {requirement}")
    return parsed


def _invalid_nonnegative_integer(field: str) -> ExtendedPowerTraceError:
    return ExtendedPowerTraceError(f"{field} must be a non-negative integer")


def _integer_from_float(value: float, *, field: str) -> int:
    if (
        not math.isfinite(value)
        or not value.is_integer()
        or abs(value) > _MAX_EXACT_FLOAT_INTEGER
    ):
        raise _invalid_nonnegative_integer(field)
    return int(value)


def _integer_from_text(value: str, *, field: str) -> int:
    if not isinstance(value, str):
        raise _invalid_nonnegative_integer(field)
    text = value.strip()
    unsigned_text = text[1:] if text[:1] in "+-" else text
    if unsigned_text.lower().startswith("0x"):
        digits = unsigned_text[2:]
        if digits and all(digit in "0123456789abcdefABCDEF" for digit in digits):
            base = 16
        else:
            base = None
    elif unsigned_text and all("0" <= digit <= "9" for digit in unsigned_text):
        base = 10
    else:
        base = None
    if base is None:
        raise _invalid_nonnegative_integer(field)
    try:
        # Base 16 preserves the hexadecimal integer spelling emitted by SMBus.
        return int(text, base)
    except (OverflowError, ValueError) as exc:
        raise _invalid_nonnegative_integer(field) from exc


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise _invalid_nonnegative_integer(field)
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = _integer_from_float(value, field=field)
    elif isinstance(value, str):
        parsed = _integer_from_text(value, field=field)
    else:
        raise _invalid_nonnegative_integer(field)
    if not 0 <= parsed <= _MAX_NONNEGATIVE_INTEGER:
        raise _invalid_nonnegative_integer(field)
    return parsed


def _normalise_source_number(value: Any, *, field: str) -> float | int:
    if field in _EXTENDED_INTEGER_COLUMNS:
        return _nonnegative_integer(value, field=field)
    return _parse_numeric(value, field=field, positive=field == "aiclk_mhz")


def _source_number_text(value: Any, *, field: str) -> str:
    """Validate one source value while retaining ordinary decimal spelling."""
    normalized = _normalise_source_number(value, field=field)
    if isinstance(value, str):
        text = value.strip()
        if text.lower().startswith(("0x", "+0x", "-0x")):
            return str(normalized)
        return text
    return str(normalized)


def _timestamp_value(value: Any, *, field: str) -> tuple[str, datetime.datetime]:
    if isinstance(value, datetime.datetime):
        parsed = value
        text = value.isoformat()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ExtendedPowerTraceError(f"{field} must be a non-empty ISO-8601 timestamp")
        parse_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.datetime.fromisoformat(parse_text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExtendedPowerTraceError(f"{field} is not a valid ISO-8601 timestamp") from exc
    else:
        raise ExtendedPowerTraceError(f"{field} must be a non-empty ISO-8601 timestamp")
    try:
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ExtendedPowerTraceError(f"{field} must include a timezone")
        normalized = parsed.astimezone(datetime.UTC)
    except (OverflowError, ValueError) as exc:
        raise ExtendedPowerTraceError(f"{field} is not a valid ISO-8601 timestamp") from exc
    return text, normalized


def _nullable_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value.strip() or None
    # Status values are labels, not measurements.  Do not interpret booleans
    # or derive a label from AICLK or any thermal value.
    return str(value)


def _snapshot_timestamp_text(
    root: Mapping[str, Any], timestamp_utc: datetime.datetime | str | None
) -> str:
    if timestamp_utc is None:
        timestamp_utc = root.get("timestamp_utc")
    if timestamp_utc is None:
        raise ExtendedPowerTraceError("required snapshot value 'timestamp_utc' is missing")
    return _timestamp_value(timestamp_utc, field="timestamp_utc")[0]


def _snapshot_normalized_values(
    telemetry: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in fields:
        source_key = _NORMALIZED_TELEMETRY_KEYS[field]
        values[field] = _source_number_text(
            _mapping_value(telemetry, source_key, field=f"telemetry.{source_key}"),
            field=field,
        )
    return values


def _snapshot_timing_values(
    root: Mapping[str, Any],
    *,
    timestamp_monotonic_ns: int | None,
    snapshot_elapsed_ns: int | None,
) -> dict[str, int]:
    supplied_values = {
        "timestamp_monotonic_ns": timestamp_monotonic_ns,
        "snapshot_elapsed_ns": snapshot_elapsed_ns,
    }
    values: dict[str, int] = {}
    for field, supplied in supplied_values.items():
        value = supplied if supplied is not None else root.get(field, _MISSING)
        if value is _MISSING or value is None:
            raise ExtendedPowerTraceError(f"required snapshot value {field!r} is missing")
        values[field] = _nonnegative_integer(value, field=field)
    return values


def _snapshot_therm_trip_count(smbus_telem: Mapping[str, Any]) -> str:
    therm_trip_count = _first_mapping_value(smbus_telem, _RAW_THERM_TRIP_KEYS)
    if therm_trip_count is _MISSING or therm_trip_count is None:
        raise ExtendedPowerTraceError(
            "required snapshot value 'smbus_telem.THERM_TRIP_COUNT' is missing"
        )
    return _source_number_text(therm_trip_count, field="therm_trip_count")


def _snapshot_throttle_values(
    telemetry: Mapping[str, Any], smbus_telem: Mapping[str, Any]
) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for field, keys in (
        ("throttle_status", _RAW_THROTTLE_STATUS_KEYS),
        ("throttle_reason", _RAW_THROTTLE_REASON_KEYS),
    ):
        value = _first_mapping_value(smbus_telem, keys)
        if value is _MISSING:
            # Some tool versions expose these optional labels in normalized
            # telemetry.  They are read as labels only when present there.
            value = telemetry.get(field, _MISSING)
        values[field] = None if value is _MISSING else _nullable_text(value)
    return values


def _extended_snapshot_row(
    root: Mapping[str, Any],
    device: Mapping[str, Any],
    *,
    timestamp_utc: datetime.datetime | str | None,
    timestamp_monotonic_ns: int | None,
    snapshot_elapsed_ns: int | None,
) -> dict[str, Any]:
    telemetry = _normalized_telemetry(device)
    smbus_telem = _raw_smbus_telemetry(device)
    row: dict[str, Any] = {
        "timestamp_utc": _snapshot_timestamp_text(root, timestamp_utc),
    }
    row.update(_snapshot_normalized_values(telemetry, ("power_w", "aiclk_mhz", "asic_temp_c")))
    row.update(
        _snapshot_timing_values(
            root,
            timestamp_monotonic_ns=timestamp_monotonic_ns,
            snapshot_elapsed_ns=snapshot_elapsed_ns,
        )
    )
    row.update(
        _snapshot_normalized_values(
            telemetry,
            ("current_a", "voltage_v", "board_power_w", "fan_rpm", "heartbeat"),
        )
    )
    row["therm_trip_count"] = _snapshot_therm_trip_count(smbus_telem)
    row.update(_snapshot_throttle_values(telemetry, smbus_telem))
    return row


def parse_extended_snapshot(
    snapshot: str | bytes | Mapping[str, Any],
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
    timestamp_utc: datetime.datetime | str | None = None,
    timestamp_monotonic_ns: int | None = None,
    snapshot_elapsed_ns: int | None = None,
) -> dict[str, Any]:
    """Parse one selected snapshot into a complete fourteen-column row."""
    root = _as_snapshot_object(snapshot)
    device = select_extended_board(root, board_id=board_id, bus_id=bus_id)
    return _extended_snapshot_row(
        root,
        device,
        timestamp_utc=timestamp_utc,
        timestamp_monotonic_ns=timestamp_monotonic_ns,
        snapshot_elapsed_ns=snapshot_elapsed_ns,
    )


def _validate_extended_row_columns(row: Mapping[str, Any]) -> None:
    if set(row) != set(EXTENDED_POWER_TRACE_COLUMNS):
        missing = sorted(set(EXTENDED_POWER_TRACE_COLUMNS) - set(row))
        extra = sorted(set(row) - set(EXTENDED_POWER_TRACE_COLUMNS))
        raise ExtendedPowerTraceError(
            f"extended trace row has unexpected columns (missing={missing!r}, extra={extra!r})"
        )


def _format_extended_power_trace_value(field: str, value: Any) -> str:
    if field == "timestamp_utc":
        if value is None:
            raise ExtendedPowerTraceError("required trace column 'timestamp_utc' is missing")
        return _timestamp_value(value, field=field)[0]
    if field in _EXTENDED_NULLABLE_COLUMNS:
        return "" if value is None else _nullable_text(value) or ""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ExtendedPowerTraceError(f"required trace column {field!r} is missing")
    return _source_number_text(value, field=field)


def format_extended_power_trace_row(row: Mapping[str, Any]) -> str:
    """Validate and serialize one complete extended row in column order."""
    _validate_extended_row_columns(row)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="")
    writer.writerow(
        _format_extended_power_trace_value(field, row.get(field))
        for field in EXTENDED_POWER_TRACE_COLUMNS
    )
    return buffer.getvalue()


def extended_telemetry_csv_row(
    snapshot: str | bytes | Mapping[str, Any],
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
    timestamp_utc: datetime.datetime | str | None = None,
    timestamp_monotonic_ns: int | None = None,
    snapshot_elapsed_ns: int | None = None,
) -> str | None:
    """Return one complete versioned extended row, or ``None`` if unusable."""
    try:
        row = parse_extended_snapshot(
            snapshot,
            board_id=board_id,
            bus_id=bus_id,
            timestamp_utc=timestamp_utc,
            timestamp_monotonic_ns=timestamp_monotonic_ns,
            snapshot_elapsed_ns=snapshot_elapsed_ns,
        )
        return format_extended_power_trace_row(row)
    except ExtendedPowerTraceError:
        return None


# A shorter name is useful at the sampler seam and does not affect the legacy
# telemetry_csv_row function.
extended_csv_row = extended_telemetry_csv_row


def extended_power_trace_header() -> str:
    """Return the exact header for the versioned extended CSV sidecar."""
    return EXTENDED_POWER_TRACE_HEADER


def _coerce_trace_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if set(row) != set(EXTENDED_POWER_TRACE_COLUMNS):
        missing = sorted(set(EXTENDED_POWER_TRACE_COLUMNS) - set(row))
        extra = sorted(set(row) - set(EXTENDED_POWER_TRACE_COLUMNS))
        raise ExtendedPowerTraceError(
            f"extended trace row {index} has unexpected columns "
            f"(missing={missing!r}, extra={extra!r})"
        )
    converted: dict[str, Any] = {}
    timestamp = row.get("timestamp_utc")
    if timestamp is None or (isinstance(timestamp, str) and not timestamp.strip()):
        raise ExtendedPowerTraceError(
            f"extended trace row {index} column 'timestamp_utc' is null"
        )
    converted["timestamp_utc"] = _timestamp_value(
        timestamp, field=f"extended trace row {index} timestamp_utc"
    )[0]
    for field in EXTENDED_POWER_TRACE_COLUMNS[1:]:
        value = row.get(field)
        if field in _EXTENDED_NULLABLE_COLUMNS:
            converted[field] = _nullable_text(value)
            continue
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ExtendedPowerTraceError(
                f"extended trace row {index} column {field!r} is null"
            )
        converted[field] = _normalise_source_number(value, field=field)
    return converted


def _as_monotonic_bound(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, field=field)


def _left_held_segments(
    rows: Sequence[Mapping[str, Any]], *, start_ns: int, end_ns: int
) -> list[tuple[Mapping[str, Any], int, int]]:
    """Return positive-duration ``[start, end)`` segments for left-held rows."""
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


def _checked_aggregate_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ExtendedPowerTraceError(f"aggregate {field} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtendedPowerTraceError(f"aggregate {field} must be finite") from exc
    if not math.isfinite(parsed):
        raise ExtendedPowerTraceError(f"aggregate {field} must be finite")
    return parsed


def _clamp_aggregate_weighted(
    weighted: float,
    minimum: float,
    maximum: float,
    *,
    field: str,
) -> float:
    """Clamp one-ULP accumulation noise, but reject a real bounds violation."""
    if minimum <= weighted <= maximum:
        return weighted
    bound = minimum if weighted < minimum else maximum
    if abs(weighted - bound) <= math.ulp(bound):
        return bound
    raise ExtendedPowerTraceError(
        f"aggregate {field}.time_weighted is outside its bounds"
    )


def _validate_aggregate_duration(result: Mapping[str, Any]) -> None:
    duration_ns = result.get("duration_ns")
    if (
        isinstance(duration_ns, bool)
        or not isinstance(duration_ns, int)
        or not 0 <= duration_ns <= _MAX_NONNEGATIVE_INTEGER
    ):
        raise ExtendedPowerTraceError("aggregate duration_ns is outside signed 64-bit range")
    duration_seconds = _checked_aggregate_float(
        result.get("duration_seconds"), field="duration_seconds"
    )
    if duration_seconds < 0.0:
        raise ExtendedPowerTraceError("aggregate duration_seconds must be non-negative")


def _validate_aggregate_index_fields(result: Mapping[str, Any]) -> None:
    for field in (
        "interval_start_monotonic_ns",
        "interval_end_monotonic_ns",
        "segment_count",
    ):
        value = result.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= _MAX_NONNEGATIVE_INTEGER
        ):
            raise ExtendedPowerTraceError(
                f"aggregate {field} is outside signed 64-bit range"
            )


def _validate_aggregate_container(
    container_name: str, container: Any
) -> None:
    if not isinstance(container, Mapping):
        raise ExtendedPowerTraceError(f"aggregate {container_name} values must be an object")
    for field, value in container.items():
        if value is None:
            continue
        parsed = _checked_aggregate_float(value, field=f"{container_name}.{field}")
        if container_name == "time_weighted_seconds" and parsed < 0.0:
            raise ExtendedPowerTraceError(
                f"aggregate {container_name}.{field} must be non-negative"
            )


def _validate_aggregate_containers(result: Mapping[str, Any]) -> None:
    for container_name in ("min", "max", "time_weighted", "time_weighted_seconds"):
        _validate_aggregate_container(container_name, result.get(container_name))


def _validate_flat_aggregate_values(result: Mapping[str, Any]) -> None:
    for field, value in result.items():
        if not field.endswith(
            ("_min", "_max", "_time_weighted", "_time_weighted_seconds")
        ):
            continue
        if value is None:
            continue
        parsed = _checked_aggregate_float(value, field=field)
        if field.endswith("_time_weighted_seconds") and parsed < 0.0:
            raise ExtendedPowerTraceError(f"aggregate {field} must be non-negative")


def _validate_one_aggregate_field(field: str, aggregate: Any) -> None:
    if not isinstance(aggregate, Mapping):
        raise ExtendedPowerTraceError(f"aggregate {field} must be an object")
    minimum = _checked_aggregate_float(aggregate.get("min"), field=f"{field}.min")
    maximum = _checked_aggregate_float(aggregate.get("max"), field=f"{field}.max")
    if minimum > maximum:
        raise ExtendedPowerTraceError(f"aggregate {field} bounds are inverted")
    weighted = aggregate.get("time_weighted")
    if weighted is not None:
        weighted_float = _checked_aggregate_float(
            weighted, field=f"{field}.time_weighted"
        )
        _clamp_aggregate_weighted(
            weighted_float,
            minimum,
            maximum,
            field=field,
        )
    weighted_seconds = _checked_aggregate_float(
        aggregate.get("time_weighted_seconds"),
        field=f"{field}.time_weighted_seconds",
    )
    if weighted_seconds < 0.0:
        raise ExtendedPowerTraceError(
            f"aggregate {field}.time_weighted_seconds must be non-negative"
        )
    weighted_ns = aggregate.get("time_weighted_ns")
    if (
        isinstance(weighted_ns, bool)
        or not isinstance(weighted_ns, int)
        or not 0 <= weighted_ns <= _MAX_NONNEGATIVE_INTEGER
    ):
        raise ExtendedPowerTraceError(
            f"aggregate {field}.time_weighted_ns is outside signed 64-bit range"
        )


def _validate_aggregate_fields(result: Mapping[str, Any]) -> None:
    aggregates = result.get("aggregates")
    if not isinstance(aggregates, Mapping):
        raise ExtendedPowerTraceError("aggregate values must be an object")
    for field, aggregate in aggregates.items():
        _validate_one_aggregate_field(field, aggregate)


def _validate_aggregate_result(result: Mapping[str, Any]) -> None:
    _validate_aggregate_duration(result)
    _validate_aggregate_index_fields(result)
    _validate_aggregate_containers(result)
    _validate_flat_aggregate_values(result)
    _validate_aggregate_fields(result)


def _time_weighted_measurement(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    field: str,
    duration_ns: int,
    minimum: Any,
    maximum: Any,
) -> float:
    """Compute a duration-weighted value without multiplying by huge durations."""
    try:
        weighted = math.fsum(
            float(row[field]) * ((end - start) / duration_ns)
            for row, start, end in segments
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExtendedPowerTraceError(
            f"aggregate {field}.time_weighted is not finite"
        ) from exc
    if not math.isfinite(weighted):
        raise ExtendedPowerTraceError(f"aggregate {field}.time_weighted is not finite")
    minimum_float = _checked_aggregate_float(minimum, field=f"{field}.min")
    maximum_float = _checked_aggregate_float(maximum, field=f"{field}.max")
    return _clamp_aggregate_weighted(
        weighted,
        minimum_float,
        maximum_float,
        field=field,
    )


def _aggregate_result_failures(result: Mapping[str, Any]) -> list[str]:
    try:
        _validate_aggregate_result(result)
    except ExtendedPowerTraceError as exc:
        return [str(exc)]
    return []


def _aggregate_measurement_field(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    field: str,
    duration_ns: int,
) -> tuple[float | int, float | int, float | None] | None:
    values = [row[field] for row, _start, _end in segments]
    if not values:
        return None
    minimum = min(values)
    maximum = max(values)
    weighted = (
        _time_weighted_measurement(
            segments,
            field=field,
            duration_ns=duration_ns,
            minimum=minimum,
            maximum=maximum,
        )
        if duration_ns
        else None
    )
    return minimum, maximum, weighted


def _aggregate_measurement_fields(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
    *,
    duration_ns: int,
    duration_seconds: float,
) -> tuple[
    dict[str, Any],
    dict[str, float | int],
    dict[str, float | int],
    dict[str, float],
    dict[str, float],
]:
    aggregates: dict[str, Any] = {}
    minima: dict[str, float | int] = {}
    maxima: dict[str, float | int] = {}
    weighted_values: dict[str, float] = {}
    weighted_seconds: dict[str, float] = {}

    for field in _EXTENDED_MEASUREMENT_COLUMNS:
        measurement = _aggregate_measurement_field(
            segments,
            field=field,
            duration_ns=duration_ns,
        )
        if measurement is None:
            continue
        minimum, maximum, weighted = measurement
        minima[field] = minimum
        maxima[field] = maximum
        if weighted is not None:
            weighted_values[field] = weighted
            weighted_seconds[field] = duration_seconds
        aggregates[field] = {
            "min": minimum,
            "max": maximum,
            "time_weighted": weighted,
            "time_weighted_seconds": duration_seconds,
            "time_weighted_ns": duration_ns,
        }
    return aggregates, minima, maxima, weighted_values, weighted_seconds


def _aggregate_categorical_values(
    segments: Sequence[tuple[Mapping[str, Any], int, int]],
) -> dict[str, list[str]]:
    categorical: dict[str, list[str]] = {}
    for field in _EXTENDED_NULLABLE_COLUMNS:
        values = [row[field] for row, _start, _end in segments if row[field] is not None]
        categorical[field] = list(dict.fromkeys(values))
    return categorical


def _add_flat_aggregate_values(
    result: dict[str, Any],
    aggregates: Mapping[str, Any],
    minima: Mapping[str, float | int],
    maxima: Mapping[str, float | int],
    weighted_values: Mapping[str, float],
    weighted_seconds: Mapping[str, float],
) -> None:
    for field in aggregates:
        result[f"{field}_min"] = minima[field]
        result[f"{field}_max"] = maxima[field]
        result[f"{field}_time_weighted"] = weighted_values.get(field)
        result[f"{field}_time_weighted_seconds"] = weighted_seconds.get(field)


def _aggregate_left_held(
    rows: Sequence[Mapping[str, Any]], *, start_ns: int, end_ns: int
) -> dict[str, Any]:
    segments = _left_held_segments(rows, start_ns=start_ns, end_ns=end_ns)
    duration_ns = sum(end - start for _row, start, end in segments)
    if duration_ns > _MAX_NONNEGATIVE_INTEGER:
        raise ExtendedPowerTraceError(
            "aggregate duration_ns is outside signed 64-bit range"
        )
    try:
        duration_seconds = duration_ns / 1_000_000_000.0
    except OverflowError as exc:
        raise ExtendedPowerTraceError("aggregate duration_seconds is not finite") from exc
    (
        aggregates,
        minima,
        maxima,
        weighted_values,
        weighted_seconds,
    ) = _aggregate_measurement_fields(
        segments,
        duration_ns=duration_ns,
        duration_seconds=duration_seconds,
    )
    result = {
        "interval_start_monotonic_ns": start_ns,
        "interval_end_monotonic_ns": end_ns,
        "duration_ns": duration_ns,
        "duration_seconds": duration_seconds,
        "aggregates": aggregates,
        "min": minima,
        "max": maxima,
        "time_weighted": weighted_values,
        "time_weighted_seconds": weighted_seconds,
        "categorical_values": _aggregate_categorical_values(segments),
        "segment_count": len(segments),
    }
    _add_flat_aggregate_values(
        result,
        aggregates,
        minima,
        maxima,
        weighted_values,
        weighted_seconds,
    )
    aggregate_failures = _aggregate_result_failures(result)
    if aggregate_failures:
        raise ExtendedPowerTraceError(aggregate_failures[0])
    return result


def _trace_result_base(*, error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": EXTENDED_POWER_TRACE_SCHEMA,
        "schema_version": EXTENDED_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "samples": [],
        "readable": False,
        "nonempty": False,
        "csv_row_count": 0,
        "sample_count": 0,
        "valid_row_count": 0,
        "invalid_row_count": 0,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "timestamp_utc_ordered": False,
        "timestamp_monotonic_ordered": False,
        "coverage_complete": False,
        "errors": [],
        "aggregates": {},
        "valid": False,
    }
    if error is not None:
        result["error"] = error
    return result


def _decode_extended_payload(
    path: Path, *, read_text_fn: Callable[[Path], str] | None
) -> tuple[bytes, str]:
    raw = (
        path.read_bytes()
        if read_text_fn is None
        else read_text_fn(path).encode("utf-8")
    )
    return raw, raw.decode("utf-8")


def _extended_trace_bounds(
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
    run_start: int | None,
    run_end: int | None,
) -> tuple[int | None, int | None]:
    if run_start_monotonic_ns is not None and run_start is not None:
        raise ExtendedPowerTraceError("run_start and run_start_monotonic_ns both supplied")
    if run_end_monotonic_ns is not None and run_end is not None:
        raise ExtendedPowerTraceError("run_end and run_end_monotonic_ns both supplied")
    start_ns = _as_monotonic_bound(
        run_start_monotonic_ns if run_start_monotonic_ns is not None else run_start,
        field="run_start_monotonic_ns",
    )
    end_ns = _as_monotonic_bound(
        run_end_monotonic_ns if run_end_monotonic_ns is not None else run_end,
        field="run_end_monotonic_ns",
    )
    if start_ns is not None and end_ns is not None and start_ns > end_ns:
        raise ExtendedPowerTraceError(_TRACE_BOUNDS_ORDER_ERROR)
    return start_ns, end_ns


def _read_extended_csv_rows(
    path: Path, *, read_text_fn: Callable[[Path], str] | None
) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        raw, text = _decode_extended_payload(path, read_text_fn=read_text_fn)
        missing_value = object()
        extra_key = object()
        reader = csv.DictReader(
            io.StringIO(text, newline=""),
            restval=missing_value,
            restkey=extra_key,
            strict=True,
        )
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != EXTENDED_POWER_TRACE_COLUMNS:
            raise ExtendedPowerTraceError(
                f"extended power trace has unexpected columns {fieldnames!r}"
            )
        rows: list[dict[str, Any]] = []
        expected_field_count = len(EXTENDED_POWER_TRACE_COLUMNS)
        for index, row in enumerate(reader):
            if extra_key in row:
                field_count = expected_field_count + len(row[extra_key])
                raise ExtendedPowerTraceError(
                    f"extended power trace row {index} has invalid physical field count "
                    f"(expected {expected_field_count}, got {field_count})"
                )
            missing_fields = [
                field
                for field in EXTENDED_POWER_TRACE_COLUMNS
                if row[field] is missing_value
            ]
            if missing_fields:
                field_count = expected_field_count - len(missing_fields)
                raise ExtendedPowerTraceError(
                    f"extended power trace row {index} has invalid physical field count "
                    f"(expected {expected_field_count}, got {field_count})"
                )
            rows.append(dict(row))
        return raw, rows
    except ExtendedPowerTraceError:
        raise
    except (csv.Error, ValueError) as exc:
        raise ExtendedPowerTraceError(str(exc)) from exc


def _coerce_extended_trace_rows(
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, raw_row in enumerate(raw_rows):
        try:
            rows.append(_coerce_trace_row(raw_row, index=index))
        except ExtendedPowerTraceError as exc:
            errors.append(str(exc))
    return rows, errors


def _set_extended_timestamp_facts(
    result: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[datetime.datetime], list[int], bool]:
    utc_timestamps = [
        _timestamp_value(row["timestamp_utc"], field="timestamp_utc")[1]
        for row in rows
    ]
    monotonic_timestamps = [int(row["timestamp_monotonic_ns"]) for row in rows]
    result["first_timestamp"] = utc_timestamps[0].isoformat()
    result["last_timestamp"] = utc_timestamps[-1].isoformat()
    result["first_timestamp_monotonic_ns"] = monotonic_timestamps[0]
    result["last_timestamp_monotonic_ns"] = monotonic_timestamps[-1]
    result["timestamps_parse"] = True
    result["timestamp_utc_ordered"] = all(
        left <= right for left, right in itertools.pairwise(utc_timestamps)
    )
    result["timestamp_monotonic_ordered"] = all(
        left <= right for left, right in itertools.pairwise(monotonic_timestamps)
    )
    ordered = result["timestamp_utc_ordered"] and result["timestamp_monotonic_ordered"]
    result["timestamps_ordered"] = ordered
    if not ordered:
        result["errors"].append("extended trace timestamps are not ordered")
        result["error"] = result["errors"][-1]
    return utc_timestamps, monotonic_timestamps, ordered


def _set_extended_trace_coverage(
    result: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    monotonic_timestamps: Sequence[int],
    *,
    start_ns: int | None,
    end_ns: int | None,
    errors: Sequence[str],
) -> None:
    if start_ns is None:
        start_ns = monotonic_timestamps[0]
    if end_ns is None:
        end_ns = monotonic_timestamps[-1]
    result["run_start_monotonic_ns"] = start_ns
    result["run_end_monotonic_ns"] = end_ns
    covers_start = monotonic_timestamps[0] <= start_ns
    covers_end = monotonic_timestamps[-1] >= end_ns
    result["covers_run_start"] = covers_start
    result["covers_run_end"] = covers_end
    if start_ns < end_ns and covers_start and covers_end:
        try:
            aggregate = _aggregate_left_held(rows, start_ns=start_ns, end_ns=end_ns)
        except (OverflowError, TypeError, ValueError) as exc:
            result["errors"].append(str(exc))
            result["error"] = str(exc)
            return
        result.update(aggregate)
        result["coverage_complete"] = True
        result["valid"] = not errors
        return
    result["errors"].append(
        "extended trace interval is not bounded by the usable trace rows"
    )
    result["error"] = result["errors"][-1]


def read_extended_power_trace_csv(
    path: Path,
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    run_start: int | None = None,
    run_end: int | None = None,
    schema: str = EXTENDED_POWER_TRACE_SCHEMA,
    read_text_fn: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Read and aggregate one complete versioned extended CSV sidecar."""
    result = _trace_result_base()
    if schema != EXTENDED_POWER_TRACE_SCHEMA:
        result["errors"] = [f"unsupported extended power trace schema: {schema!r}"]
        result["error"] = result["errors"][0]
        return result
    try:
        start_ns, end_ns = _extended_trace_bounds(
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
            run_start=run_start,
            run_end=run_end,
        )
        path = Path(path)
        raw, raw_rows = _read_extended_csv_rows(path, read_text_fn=read_text_fn)
        result["byte_count"] = len(raw)
    except (OSError, UnicodeError, ExtendedPowerTraceError) as exc:
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        return result

    result["readable"] = True
    result["csv_row_count"] = len(raw_rows)
    result["nonempty"] = bool(raw_rows)
    rows, errors = _coerce_extended_trace_rows(raw_rows)
    result["samples"] = rows
    result["sample_count"] = len(rows)
    result["valid_row_count"] = len(rows)
    result["invalid_row_count"] = len(raw_rows) - len(rows)
    result["errors"] = errors
    if not rows:
        result["error"] = errors[0] if errors else "extended power trace has no rows"
        return result

    try:
        _utc_timestamps, monotonic_timestamps, ordered = _set_extended_timestamp_facts(
            result, rows
        )
    except (OverflowError, TypeError, ValueError) as exc:
        result["errors"].append(str(exc))
        result["error"] = str(exc)
        return result
    if not ordered:
        return result
    _set_extended_trace_coverage(
        result,
        rows,
        monotonic_timestamps,
        start_ns=start_ns,
        end_ns=end_ns,
        errors=errors,
    )
    return result


# Common naming variants at the parser seam.
read_extended_power_trace = read_extended_power_trace_csv
parse_extended_power_trace = read_extended_power_trace_csv
read_power_trace_extended_csv = read_extended_power_trace_csv
parse_power_trace_extended = read_extended_power_trace_csv
parse_extended_telemetry = parse_extended_snapshot


def read_extended_power_trace_snapshot(
    path: Path,
    *,
    schema: str = EXTENDED_POWER_TRACE_SCHEMA,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    run_start: int | None = None,
    run_end: int | None = None,
) -> dict[str, Any]:
    """Read either an extended CSV or a JSON sidecar carrying extended rows."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        return read_extended_power_trace_csv(
            path,
            schema=schema,
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
            run_start=run_start,
            run_end=run_end,
        )
    result = _trace_result_base()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ExtendedPowerTraceError("extended trace sidecar must be an object")
        declared_schema = payload.get("schema")
        declared_version = payload.get("schema_version", EXTENDED_POWER_TRACE_SCHEMA_VERSION)
        if (
            declared_schema != schema
            or declared_schema != EXTENDED_POWER_TRACE_SCHEMA
            or not isinstance(declared_version, int)
            or isinstance(declared_version, bool)
            or declared_version != EXTENDED_POWER_TRACE_SCHEMA_VERSION
        ):
            raise ExtendedPowerTraceError("extended trace sidecar schema is not supported")
        columns = tuple(payload.get("columns", ()))
        if columns != EXTENDED_POWER_TRACE_COLUMNS:
            raise ExtendedPowerTraceError("extended trace sidecar columns are invalid")
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ExtendedPowerTraceError("extended trace sidecar samples must be an array")
        rows = []
        for index, row in enumerate(samples):
            if not isinstance(row, Mapping):
                raise ExtendedPowerTraceError(
                    f"extended trace sidecar samples[{index}] must be an object"
                )
            rows.append(_coerce_trace_row(row, index=index))
    except (OSError, RecursionError, ValueError, TypeError) as exc:
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        return result
    # Reuse CSV validation and aggregation without writing a file.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EXTENDED_POWER_TRACE_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return read_extended_power_trace_csv(
        path,
        schema=schema,
        run_start_monotonic_ns=run_start_monotonic_ns,
        run_end_monotonic_ns=run_end_monotonic_ns,
        run_start=run_start,
        run_end=run_end,
        read_text_fn=lambda _path: buffer.getvalue(),
    )


_MAPPING_RECHECK_IGNORED_FIELDS = frozenset({
    "byte_count",
    "error",
    "schema",
    "schema_version",
    "columns",
})


def _trace_validation_failures(trace: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if trace.get("schema") != EXTENDED_POWER_TRACE_SCHEMA:
        failures.append("extended trace schema is missing or unsupported")
    schema_version = trace.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != EXTENDED_POWER_TRACE_SCHEMA_VERSION
    ):
        failures.append("extended trace schema_version is missing or unsupported")
    if trace.get("columns") != list(EXTENDED_POWER_TRACE_COLUMNS):
        failures.append("extended trace columns do not match the versioned schema")
    for field in (
        "readable",
        "nonempty",
        "timestamps_parse",
        "timestamps_ordered",
        "coverage_complete",
    ):
        if trace.get(field) is not True:
            failures.append(f"extended trace {field} check failed")
    if trace.get("invalid_row_count", 0) != 0:
        failures.append("extended trace contains invalid rows")
    errors = trace.get("errors", ())
    if isinstance(errors, (str, bytes)) or not isinstance(errors, Sequence):
        failures.append("extended trace errors must be an array")
    else:
        failures.extend(str(error) for error in errors)
    failures.extend(_aggregate_result_failures(trace))
    return failures


def _summary_mapping_matches(
    actual: Any,
    expected: Mapping[Any, Any],
    field_path: tuple[str, ...],
) -> bool:
    if not isinstance(actual, Mapping) or set(actual) != set(expected):
        return False
    return all(
        _summary_values_match(
            actual[key],
            expected[key],
            field_path=(*field_path, str(key)),
        )
        for key in expected
    )


def _summary_list_matches(
    actual: Any,
    expected: list[Any],
    field_path: tuple[str, ...],
) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(
            _summary_values_match(
                actual_value,
                expected_value,
                field_path=(*field_path, str(index)),
            )
            for index, (actual_value, expected_value) in enumerate(
                zip(actual, expected)
            )
        )
    )


def _summary_tuple_matches(
    actual: Any,
    expected: tuple[Any, ...],
    field_path: tuple[str, ...],
) -> bool:
    return (
        isinstance(actual, (tuple, list))
        and len(actual) == len(expected)
        and all(
            _summary_values_match(
                actual_value,
                expected_value,
                field_path=(*field_path, str(index)),
            )
            for index, (actual_value, expected_value) in enumerate(
                zip(actual, expected)
            )
        )
    )


def _summary_float_matches(
    actual: Any,
    expected: float,
    field_path: tuple[str, ...],
) -> bool:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    try:
        actual_float = float(actual)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(actual_float) or not math.isfinite(expected):
        return actual_float == expected
    if field_path and (
        field_path[-1] == "time_weighted"
        or field_path[-1].endswith("_time_weighted")
    ):
        return abs(actual_float - expected) <= math.ulp(expected)
    return actual_float == expected


def _summary_values_match(
    actual: Any,
    expected: Any,
    *,
    field_path: tuple[str, ...] = (),
) -> bool:
    if isinstance(expected, Mapping):
        return _summary_mapping_matches(actual, expected, field_path)
    if isinstance(expected, list):
        return _summary_list_matches(actual, expected, field_path)
    if isinstance(expected, tuple):
        return _summary_tuple_matches(actual, expected, field_path)
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return _summary_float_matches(actual, expected, field_path)
    return type(actual) is type(expected) and actual == expected


def _mapping_summary_failures(
    declared: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    normalized_samples: Sequence[Mapping[str, Any]],
) -> list[str]:
    expected_summary = dict(recomputed)
    expected_summary["samples"] = list(normalized_samples)
    failures: list[str] = []
    for field, expected in expected_summary.items():
        if field in _MAPPING_RECHECK_IGNORED_FIELDS:
            continue
        if field not in declared:
            failures.append(f"extended trace summary {field} is missing")
        elif not _summary_values_match(declared[field], expected):
            failures.append(f"extended trace summary {field} does not match the reader")
    return failures


def _normalise_mapping_sample(
    sample: Any, *, index: int
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(sample, Mapping):
        return None, f"extended trace samples[{index}] must be an object"
    try:
        return _coerce_trace_row(sample, index=index), None
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _normalise_mapping_sample_list(
    samples: list[Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    normalized_samples: list[dict[str, Any]] = []
    if not samples:
        failures.append("extended trace samples must be non-empty")
    for index, sample in enumerate(samples):
        normalized, failure = _normalise_mapping_sample(sample, index=index)
        if failure is not None:
            failures.append(failure)
        elif normalized is not None:
            normalized_samples.append(normalized)
    return failures, normalized_samples


def _normalise_mapping_samples(
    trace: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    samples = trace.get("samples", _MISSING)
    if samples is _MISSING:
        return ["extended trace samples are missing"], []
    if not isinstance(samples, list):
        return ["extended trace samples must be an array"], []
    return _normalise_mapping_sample_list(samples)


def _mapping_trace_bounds(
    trace: Mapping[str, Any],
) -> tuple[dict[str, int | None], list[str]]:
    failures: list[str] = []
    bounds: dict[str, int | None] = {}
    for field in ("run_start_monotonic_ns", "run_end_monotonic_ns"):
        value = trace.get(field, _MISSING)
        if value is _MISSING:
            failures.append(f"extended trace {field} is missing")
            bounds[field] = None
            continue
        try:
            bound = _as_monotonic_bound(value, field=field)
        except (TypeError, ValueError) as exc:
            failures.append(str(exc))
            bound = None
        if bound is None:
            failures.append(f"extended trace {field} is missing")
        bounds[field] = bound
    start_ns = bounds["run_start_monotonic_ns"]
    end_ns = bounds["run_end_monotonic_ns"]
    if start_ns is not None and end_ns is not None and start_ns > end_ns:
        failures.append(_TRACE_BOUNDS_ORDER_ERROR)
    return bounds, failures


def _reread_mapping_trace(
    normalized_samples: Sequence[Mapping[str, Any]],
    *,
    start_ns: int | None,
    end_ns: int | None,
) -> tuple[dict[str, Any], str | None]:
    buffer = io.StringIO(newline="")
    try:
        writer = csv.DictWriter(buffer, fieldnames=EXTENDED_POWER_TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_samples)
        recomputed = read_extended_power_trace_csv(
            Path("<mapping>"),
            run_start_monotonic_ns=start_ns,
            run_end_monotonic_ns=end_ns,
            read_text_fn=lambda _path: buffer.getvalue(),
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OSError,
        csv.Error,
        RuntimeError,
    ) as exc:
        message = f"extended trace mapping could not be re-read: {exc}"
        recomputed = _trace_result_base(error=message)
        recomputed["errors"] = [message]
        return recomputed, message
    return recomputed, None


def _recompute_mapping_trace(
    trace: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    sample_failures, normalized_samples = _normalise_mapping_samples(trace)
    bounds, bound_failures = _mapping_trace_bounds(trace)
    failures = [*sample_failures, *bound_failures]
    recomputed, reread_failure = _reread_mapping_trace(
        normalized_samples,
        start_ns=bounds["run_start_monotonic_ns"],
        end_ns=bounds["run_end_monotonic_ns"],
    )
    if reread_failure is not None:
        failures.append(reread_failure)
    return recomputed, failures, normalized_samples


def _mapping_optional_summary_failures(
    declared: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> list[str]:
    """Check only summaries supplied by a raw mapping source.

    A mapping source is allowed to contain just schema, bounds, and samples,
    like the JSON reader's source sidecar.  When it does provide reader flags
    or aggregates, those declarations are checked rather than trusted.
    """
    failures: list[str] = []
    for field, actual in declared.items():
        if field in _MAPPING_RECHECK_IGNORED_FIELDS or field == "samples":
            continue
        if field not in recomputed:
            continue
        if not _summary_values_match(
            actual,
            recomputed[field],
            field_path=(str(field),),
        ):
            failures.append(f"extended trace summary {field} does not match the reader")
    return failures


def _mapping_schema_failures(payload: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if payload.get("schema", EXTENDED_POWER_TRACE_SCHEMA) != EXTENDED_POWER_TRACE_SCHEMA:
        failures.append("extended trace schema is missing or unsupported")
    version = payload.get("schema_version", EXTENDED_POWER_TRACE_SCHEMA_VERSION)
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != EXTENDED_POWER_TRACE_SCHEMA_VERSION
    ):
        failures.append("extended trace schema_version is missing or unsupported")
    columns = payload.get("columns", list(EXTENDED_POWER_TRACE_COLUMNS))
    try:
        columns_match = tuple(columns) == EXTENDED_POWER_TRACE_COLUMNS
    except (TypeError, ValueError):
        columns_match = False
    if not columns_match:
        failures.append("extended trace columns do not match the versioned schema")
    return failures


def _mapping_samples(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    raw_samples = payload.get("samples", _MISSING)
    if raw_samples is _MISSING:
        return ["extended trace samples are missing"], []
    if not isinstance(raw_samples, list):
        return ["extended trace samples must be an array"], []
    return _normalise_mapping_sample_list(raw_samples)


def _mapping_bound(
    payload: Mapping[str, Any],
    normalized_samples: Sequence[Mapping[str, Any]],
    *,
    field: str,
    supplied: int | None,
) -> tuple[int | None, list[str]]:
    value = supplied if supplied is not None else payload.get(field, _MISSING)
    if value is _MISSING:
        if normalized_samples:
            value = (
                normalized_samples[0]["timestamp_monotonic_ns"]
                if field == "run_start_monotonic_ns"
                else normalized_samples[-1]["timestamp_monotonic_ns"]
            )
        else:
            return None, [f"extended trace {field} is missing"]
    if value is None:
        return None, [f"extended trace {field} is missing"]
    try:
        return _as_monotonic_bound(value, field=field), []
    except (TypeError, ValueError) as exc:
        return None, [str(exc)]


def _mapping_bounds(
    payload: Mapping[str, Any],
    normalized_samples: Sequence[Mapping[str, Any]],
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
) -> tuple[dict[str, int | None], list[str]]:
    bounds: dict[str, int | None] = {}
    failures: list[str] = []
    for field, supplied in (
        ("run_start_monotonic_ns", run_start_monotonic_ns),
        ("run_end_monotonic_ns", run_end_monotonic_ns),
    ):
        bound, bound_failures = _mapping_bound(
            payload,
            normalized_samples,
            field=field,
            supplied=supplied,
        )
        bounds[field] = bound
        failures.extend(bound_failures)
    start_ns = bounds["run_start_monotonic_ns"]
    end_ns = bounds["run_end_monotonic_ns"]
    if start_ns is not None and end_ns is not None and start_ns > end_ns:
        failures.append(_TRACE_BOUNDS_ORDER_ERROR)
    return bounds, failures


def _invalid_mapping_result(
    trace: dict[str, Any], failures: Sequence[str]
) -> dict[str, Any]:
    unique_failures = list(dict.fromkeys(failures))
    trace["schema"] = EXTENDED_POWER_TRACE_SCHEMA
    trace["schema_version"] = EXTENDED_POWER_TRACE_SCHEMA_VERSION
    trace["columns"] = list(EXTENDED_POWER_TRACE_COLUMNS)
    if unique_failures:
        trace["valid"] = False
        trace["errors"] = list(
            dict.fromkeys([*trace.get("errors", []), *unique_failures])
        )
        trace["error"] = trace["errors"][0]
    return trace


def read_extended_power_trace_mapping(
    payload: Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Read a raw extended-trace mapping through the versioned reader.

    The samples are normalized, serialized through the same CSV reader used by
    file and JSON sources, and then re-aggregated.  Missing bounds retain the
    source reader's default of the first and last normalized sample; explicit
    bounds still have to be covered by those samples.  Any precomputed reader
    fields supplied by the mapping are checked, while the returned samples and
    coverage always come from the re-read result.
    """
    if not isinstance(payload, Mapping):
        return _invalid_mapping_result(
            _trace_result_base(), ["extended trace mapping must be an object"]
        )
    failures = _mapping_schema_failures(payload)
    sample_failures, normalized_samples = _mapping_samples(payload)
    failures.extend(sample_failures)
    bounds, bound_failures = _mapping_bounds(
        payload,
        normalized_samples,
        run_start_monotonic_ns=run_start_monotonic_ns,
        run_end_monotonic_ns=run_end_monotonic_ns,
    )
    failures.extend(bound_failures)
    recomputed, reread_failure = _reread_mapping_trace(
        normalized_samples,
        start_ns=bounds["run_start_monotonic_ns"],
        end_ns=bounds["run_end_monotonic_ns"],
    )
    if reread_failure is not None:
        failures.append(reread_failure)
    failures.extend(_trace_validation_failures(recomputed))
    failures.extend(_mapping_optional_summary_failures(payload, recomputed))
    return _invalid_mapping_result(recomputed, failures)


def _copy_mapping_trace(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    try:
        trace = dict(payload)
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OSError,
        RuntimeError,
    ) as exc:
        message = f"extended trace mapping could not be copied: {exc}"
        trace = _trace_result_base(error=message)
        trace["errors"] = [message]
        return trace, [message]
    return trace, []


def _validate_mapping_trace(trace: Mapping[str, Any]) -> list[str]:
    try:
        failures = _trace_validation_failures(trace)
        recomputed, recheck_failures, normalized_samples = _recompute_mapping_trace(
            trace
        )
        failures.extend(recheck_failures)
        failures.extend(_trace_validation_failures(recomputed))
        failures.extend(
            _mapping_summary_failures(
                trace,
                recomputed,
                normalized_samples=normalized_samples,
            )
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        OSError,
        csv.Error,
        RuntimeError,
    ) as exc:
        return [f"extended trace mapping validation failed: {exc}"]
    return failures


def _mapping_validation_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    trace, failures = _copy_mapping_trace(payload)
    if not failures:
        failures = _validate_mapping_trace(trace)
    failures = list(dict.fromkeys(failures))
    return {
        "valid": not failures,
        "schema": EXTENDED_POWER_TRACE_SCHEMA,
        "schema_version": EXTENDED_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "failures": failures,
        "trace": trace,
        "aggregates": trace.get("aggregates", {}),
    }


def validate_extended_power_trace(
    trace_or_path: Mapping[str, Any] | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a structured validation report for an extended trace."""
    if isinstance(trace_or_path, Mapping):
        return _mapping_validation_report(trace_or_path)
    trace = read_extended_power_trace_snapshot(trace_or_path, **kwargs)
    failures = _trace_validation_failures(trace)
    failures = list(dict.fromkeys(failures))
    return {
        "valid": not failures,
        "schema": EXTENDED_POWER_TRACE_SCHEMA,
        "schema_version": EXTENDED_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "failures": failures,
        "trace": trace,
        "aggregates": trace.get("aggregates", {}),
    }


# A result-level name is useful to callers that already use ``validate_*``
# terminology for the legacy reader.
validate_power_trace_extended = validate_extended_power_trace


__all__ = [
    "EXTENDED_POWER_TRACE_COLUMNS",
    "EXTENDED_POWER_TRACE_HEADER",
    "EXTENDED_POWER_TRACE_SCHEMA",
    "EXTENDED_POWER_TRACE_SCHEMA_VERSION",
    "POWER_TRACE_EXTENDED_COLUMNS",
    "POWER_TRACE_EXTENDED_SCHEMA",
    "POWER_TRACE_EXTENDED_SCHEMA_VERSION",
    "ExtendedPowerTraceError",
    "extended_csv_row",
    "extended_power_trace_header",
    "extended_telemetry_csv_row",
    "format_extended_power_trace_row",
    "parse_extended_power_trace",
    "parse_extended_snapshot",
    "parse_extended_telemetry",
    "parse_power_trace_extended",
    "read_extended_power_trace",
    "read_extended_power_trace_csv",
    "read_extended_power_trace_mapping",
    "read_extended_power_trace_snapshot",
    "read_power_trace_extended_csv",
    "select_extended_board",
    "validate_extended_power_trace",
    "validate_power_trace_extended",
]
