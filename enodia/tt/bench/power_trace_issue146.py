"""Strict board-free Issue #146 power traces with raw arbiter telemetry.

Issue #146 extends the immutable Issue #137 trace contract with the four raw
arbiter words emitted by ``smbus_telem``.  The words stay hexadecimal strings,
including their spelling and case; frequency, index, and mask interpretation
belongs to the analyzer and is deliberately absent here.  The Issue #137
GDDR fields and their source-unit provenance are retained unchanged.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from enodia.tt.bench import power_trace_extended as _q2
from enodia.tt.bench import power_trace_issue137 as _issue137
from enodia.tt.bench import run_resident

ISSUE146_POWER_TRACE_SCHEMA = "issue-146-power-trace-v1"
ISSUE146_POWER_TRACE_SCHEMA_VERSION = 1

ISSUE137_POWER_TRACE_COLUMNS = _issue137.ISSUE137_POWER_TRACE_COLUMNS
ISSUE137_POWER_TRACE_HEADER = _issue137.ISSUE137_POWER_TRACE_HEADER
ISSUE146_ARB_COLUMNS = (
    "aiclk_arb_min_raw",
    "aiclk_arb_max_raw",
    "enabled_min_arb_raw",
    "enabled_max_arb_raw",
)
ISSUE146_ARB_SOURCE_FIELDS = {
    "aiclk_arb_min_raw": "AICLK_ARB_MIN",
    "aiclk_arb_max_raw": "AICLK_ARB_MAX",
    "enabled_min_arb_raw": "ENABLED_MIN_ARB",
    "enabled_max_arb_raw": "ENABLED_MAX_ARB",
}
# A source-field name that describes the same mapping is useful to callers
# that do not need to distinguish the arbiter subset from other raw fields.
ISSUE146_RAW_SOURCE_FIELDS = ISSUE146_ARB_SOURCE_FIELDS
ISSUE146_POWER_TRACE_COLUMNS = (*ISSUE137_POWER_TRACE_COLUMNS, *ISSUE146_ARB_COLUMNS)
ISSUE146_POWER_TRACE_HEADER = ",".join(ISSUE146_POWER_TRACE_COLUMNS)

ISSUE137_GDDR_COLUMNS = _issue137.ISSUE137_GDDR_COLUMNS
ISSUE137_GDDR_SOURCE_FIELDS = _issue137.ISSUE137_GDDR_SOURCE_FIELDS
ISSUE137_SOURCE_VERSIONS = _issue137.ISSUE137_SOURCE_VERSIONS
ISSUE137_UNIT_PROVENANCE = _issue137.ISSUE137_UNIT_PROVENANCE
ISSUE146_UNIT_PROVENANCE = ISSUE137_UNIT_PROVENANCE

_RAW_HEX_WORD_RE = re.compile(r"^0x[0-9a-fA-F]{1,8}$")
_MISSING = object()


class Issue146PowerTraceError(ValueError):
    """Raised when an Issue #146 trace or raw word is not representable."""


def source_versions_verified(host_sw_vers: Any) -> bool:
    """Return whether the fixed Issue #137 GDDR source audit is present."""
    return _issue137.source_versions_verified(host_sw_vers)


def _raw_hex_word(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise Issue146PowerTraceError(f"{field} must retain a raw hexadecimal word")
    raw = value.strip()
    if _RAW_HEX_WORD_RE.fullmatch(raw) is None:
        raise Issue146PowerTraceError(f"{field} must be a raw hexadecimal word (32-bit, 0x...)")
    return raw


def parse_issue146_snapshot(
    snapshot: Mapping[str, Any] | str | bytes,
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
    timestamp_utc: Any = None,
    timestamp_monotonic_ns: int | None = None,
    snapshot_elapsed_ns: int | None = None,
) -> dict[str, Any]:
    """Convert one explicitly selected snapshot into a complete Issue #146 row."""
    try:
        row = _issue137.parse_issue137_snapshot(
            snapshot,
            board_id=board_id,
            bus_id=bus_id,
            timestamp_utc=timestamp_utc,
            timestamp_monotonic_ns=timestamp_monotonic_ns,
            snapshot_elapsed_ns=snapshot_elapsed_ns,
        )
        device = _q2.select_extended_board(
            snapshot,
            board_id=board_id,
            bus_id=bus_id,
        )
        smbus = device.get("smbus_telem")
        if not isinstance(smbus, Mapping):
            raise Issue146PowerTraceError("selected device smbus_telem must be an object")
        for column, source in ISSUE146_ARB_SOURCE_FIELDS.items():
            value = smbus.get(source, _MISSING)
            if value is _MISSING or value is None:
                raise Issue146PowerTraceError(
                    f"required snapshot value smbus_telem.{source} is missing"
                )
            row[column] = _raw_hex_word(value, field=f"smbus_telem.{source}")
        return row
    except Issue146PowerTraceError:
        raise
    except (_issue137.Issue137PowerTraceError, _q2.ExtendedPowerTraceError) as exc:
        raise Issue146PowerTraceError(str(exc)) from exc


def issue146_power_trace_header() -> str:
    """Return the exact Issue #146 CSV header."""
    return ISSUE146_POWER_TRACE_HEADER


def _validate_issue146_row(row: Mapping[str, Any]) -> dict[str, str]:
    expected = set(ISSUE146_POWER_TRACE_COLUMNS)
    try:
        actual = set(row)
    except (TypeError, ValueError) as exc:
        raise Issue146PowerTraceError("Issue #146 trace row must be an object") from exc
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Issue146PowerTraceError(
            f"Issue #146 trace row columns are invalid (missing={missing!r}, extra={extra!r})"
        )
    return {column: _raw_hex_word(row[column], field=column) for column in ISSUE146_ARB_COLUMNS}


def format_issue146_power_trace_row(row: Mapping[str, Any]) -> str:
    """Strictly format one Issue #146 row in schema column order."""
    raw_words = _validate_issue146_row(row)
    base = {column: row[column] for column in ISSUE137_POWER_TRACE_COLUMNS}
    try:
        base_fields = next(
            csv.reader(io.StringIO(_issue137.format_issue137_power_trace_row(base), newline=""))
        )
    except (_issue137.Issue137PowerTraceError, _q2.ExtendedPowerTraceError) as exc:
        raise Issue146PowerTraceError(str(exc)) from exc
    values = [*base_fields, *[raw_words[column] for column in ISSUE146_ARB_COLUMNS]]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="").writerow(values)
    return output.getvalue()


def issue146_trace_row(
    snapshot: Mapping[str, Any] | str | bytes,
    **kwargs: Any,
) -> str | None:
    """Return one formatted row, or ``None`` when the snapshot is unusable."""
    try:
        return format_issue146_power_trace_row(parse_issue146_snapshot(snapshot, **kwargs))
    except Issue146PowerTraceError:
        return None


def _trace_result_base() -> dict[str, Any]:
    return {
        "schema": ISSUE146_POWER_TRACE_SCHEMA,
        "schema_version": ISSUE146_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(ISSUE146_POWER_TRACE_COLUMNS),
        "samples": [],
        "readable": False,
        "nonempty": False,
        "csv_row_count": 0,
        "sample_count": 0,
        "valid_row_count": 0,
        "invalid_row_count": 0,
        "errors": [],
        "valid": False,
        "unit_provenance": None,
        "gddr_source_status": "unknown",
        "gddr_decoded": [],
    }


def _coerce_issue146_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise Issue146PowerTraceError(f"Issue #146 trace row {index} must be an object")
    expected = set(ISSUE146_POWER_TRACE_COLUMNS)
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Issue146PowerTraceError(
            f"Issue #146 trace row {index} columns are invalid "
            f"(missing={missing!r}, extra={extra!r})"
        )
    base = {column: row[column] for column in ISSUE137_POWER_TRACE_COLUMNS}
    try:
        normalized = _issue137._coerce_issue137_row(base, index=index)
    except (_issue137.Issue137PowerTraceError, _q2.ExtendedPowerTraceError) as exc:
        raise Issue146PowerTraceError(str(exc)) from exc
    for column in ISSUE146_ARB_COLUMNS:
        normalized[column] = _raw_hex_word(
            row[column], field=f"Issue #146 trace row {index} {column}"
        )
    return normalized


def _validated_issue146_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise Issue146PowerTraceError("Issue #146 trace must be an object")
    if payload.get("schema") != ISSUE146_POWER_TRACE_SCHEMA:
        raise Issue146PowerTraceError("Issue #146 trace schema is unsupported")
    version = payload.get("schema_version")
    if version != ISSUE146_POWER_TRACE_SCHEMA_VERSION or isinstance(version, bool):
        raise Issue146PowerTraceError("Issue #146 trace schema_version is unsupported")
    if tuple(payload.get("columns", ())) != ISSUE146_POWER_TRACE_COLUMNS:
        raise Issue146PowerTraceError("Issue #146 trace columns are invalid")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise Issue146PowerTraceError("Issue #146 trace samples must be an array")
    return [_coerce_issue146_row(row, index) for index, row in enumerate(raw_samples)]


def _base_payload(
    samples: Sequence[Mapping[str, Any]],
    *,
    run_start_monotonic_ns: int | None,
    run_end_monotonic_ns: int | None,
) -> dict[str, Any]:
    if not samples:
        start = run_start_monotonic_ns
        end = run_end_monotonic_ns
    else:
        start = (
            samples[0]["timestamp_monotonic_ns"]
            if run_start_monotonic_ns is None
            else run_start_monotonic_ns
        )
        end = (
            samples[-1]["timestamp_monotonic_ns"]
            if run_end_monotonic_ns is None
            else run_end_monotonic_ns
        )
    return {
        "schema": _q2.EXTENDED_POWER_TRACE_SCHEMA,
        "schema_version": _q2.EXTENDED_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(_q2.EXTENDED_POWER_TRACE_COLUMNS),
        "samples": [
            {column: row[column] for column in _q2.EXTENDED_POWER_TRACE_COLUMNS} for row in samples
        ],
        "run_start_monotonic_ns": start,
        "run_end_monotonic_ns": end,
    }


def _read_rows_from_text(text: str) -> list[dict[str, str]]:
    missing = object()
    extra = object()
    reader = csv.DictReader(
        io.StringIO(text, newline=""), restval=missing, restkey=extra, strict=True
    )
    fieldnames = tuple(reader.fieldnames or ())
    if fieldnames != ISSUE146_POWER_TRACE_COLUMNS:
        expected = set(ISSUE146_POWER_TRACE_COLUMNS)
        actual = set(fieldnames)
        missing_fields = sorted(expected - actual)
        extra_fields = sorted(actual - expected)
        raise Issue146PowerTraceError(
            "Issue #146 power trace header is invalid "
            f"(missing={missing_fields!r}, extra={extra_fields!r})"
        )
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if extra in row:
            raise Issue146PowerTraceError(f"Issue #146 trace row {index} has extra fields")
        missing_fields = [
            column for column in ISSUE146_POWER_TRACE_COLUMNS if row[column] is missing
        ]
        if missing_fields:
            raise Issue146PowerTraceError(
                f"Issue #146 trace row {index} has missing fields {missing_fields!r}"
            )
        rows.append({column: row[column] for column in ISSUE146_POWER_TRACE_COLUMNS})
    return rows


def _read_issue146_mapping(
    payload: Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    source_established: bool = False,
) -> dict[str, Any]:
    result = _trace_result_base()
    raw_samples = payload.get("samples") if isinstance(payload, Mapping) else None
    raw_count = len(raw_samples) if isinstance(raw_samples, list) else 0
    try:
        rows = _validated_issue146_rows(payload)
    except (ValueError, TypeError) as exc:
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        result["csv_row_count"] = raw_count
        result["invalid_row_count"] = raw_count
        return result
    base = _q2.read_extended_power_trace_mapping(
        _base_payload(
            rows,
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
        )
    )
    result.update(base)
    result["schema"] = ISSUE146_POWER_TRACE_SCHEMA
    result["schema_version"] = ISSUE146_POWER_TRACE_SCHEMA_VERSION
    result["columns"] = list(ISSUE146_POWER_TRACE_COLUMNS)
    result["samples"] = rows
    result["unit_provenance"] = ISSUE137_UNIT_PROVENANCE if source_established else None
    result["gddr_source_status"] = "established" if source_established else "unknown"
    result["gddr_decoded"] = [
        _issue137.decode_issue137_gddr_row(row, source_established=source_established)
        for row in rows
    ]
    failures = list(base.get("errors", ()))
    result["valid_row_count"] = len(rows) if not failures else max(0, len(rows) - len(failures))
    result["invalid_row_count"] = len(failures)
    if failures:
        result["valid"] = False
        result["errors"] = list(dict.fromkeys(str(item) for item in failures))
        result["error"] = result["errors"][0]
    else:
        result["errors"] = []
        result["valid"] = bool(rows) and base.get("valid") is True
        if not result["valid"]:
            result["error"] = "Issue #146 trace has no complete bounded interval"
            result["errors"] = [result["error"]]
    return result


def _trace_path(path: Path, *, operation: str) -> Path:
    """Resolve trace IO beneath the shared trusted output roots."""
    candidate = Path(path)
    try:
        resolved = run_resident._safe_output_path(
            candidate, description=f"Issue #146 trace {operation} path"
        )
    except Exception as exc:
        raise Issue146PowerTraceError(
            f"Issue #146 trace {operation} path must remain beneath a trusted output root"
        ) from exc
    current = candidate
    while True:
        if current.is_symlink():
            raise Issue146PowerTraceError(
                f"Issue #146 trace {operation} path must not traverse a symlink"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    return resolved


def read_issue146_power_trace(
    path: Path,
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    source_established: bool = False,
) -> dict[str, Any]:
    """Read one strict Issue #146 CSV trace and recompute Q2 aggregates."""
    result = _trace_result_base()
    raw = ""
    try:
        source = _trace_path(path, operation="read")
        raw = source.read_text(encoding="utf-8")
        rows = _read_rows_from_text(raw)
        payload = {
            "schema": ISSUE146_POWER_TRACE_SCHEMA,
            "schema_version": ISSUE146_POWER_TRACE_SCHEMA_VERSION,
            "columns": list(ISSUE146_POWER_TRACE_COLUMNS),
            "samples": rows,
        }
        result = _read_issue146_mapping(
            payload,
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
            source_established=source_established,
        )
        result["byte_count"] = len(raw.encode("utf-8"))
        result["readable"] = True
        result["csv_row_count"] = len(rows)
        result["nonempty"] = bool(rows)
    except (OSError, UnicodeError, csv.Error, Issue146PowerTraceError) as exc:
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        if raw:
            result["csv_row_count"] = max(0, len(raw.splitlines()) - 1)
            result["invalid_row_count"] = result["csv_row_count"]
    return result


def read_issue146_power_trace_mapping(payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Read an in-memory Issue #146 mapping with strict schema checks."""
    return _read_issue146_mapping(payload, **kwargs)


def read_issue146_power_trace_snapshot(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Read CSV or a JSON sidecar carrying complete Issue #146 rows."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        return read_issue146_power_trace(path, **kwargs)
    try:
        source = _trace_path(path, operation="read")
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, Issue146PowerTraceError) as exc:
        result = _trace_result_base()
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        return result
    if not isinstance(payload, Mapping):
        result = _trace_result_base()
        result["errors"] = ["Issue #146 trace sidecar must be an object"]
        result["error"] = result["errors"][0]
        return result
    return _read_issue146_mapping(payload, **kwargs)


def write_issue146_power_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a complete Issue #146 CSV trace without changing input rows."""
    serialized = [format_issue146_power_trace_row(row) for row in rows]
    destination = _trace_path(path, operation="write")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        ISSUE146_POWER_TRACE_HEADER + "\n" + "\n".join(serialized) + "\n",
        encoding="utf-8",
        newline="",
    )
    return destination


def validate_issue146_power_trace(
    trace_or_path: Mapping[str, Any] | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a fail-closed validation report for an Issue #146 trace."""
    trace = (
        read_issue146_power_trace_mapping(trace_or_path, **kwargs)
        if isinstance(trace_or_path, Mapping)
        else read_issue146_power_trace_snapshot(trace_or_path, **kwargs)
    )
    failures = list(trace.get("errors", ()))
    if trace.get("schema") != ISSUE146_POWER_TRACE_SCHEMA:
        failures.append("Issue #146 trace schema is unsupported")
    if trace.get("columns") != list(ISSUE146_POWER_TRACE_COLUMNS):
        failures.append("Issue #146 trace columns are invalid")
    return {
        "valid": bool(trace.get("valid") is True and not failures),
        "schema": ISSUE146_POWER_TRACE_SCHEMA,
        "schema_version": ISSUE146_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(ISSUE146_POWER_TRACE_COLUMNS),
        "failures": list(dict.fromkeys(str(item) for item in failures)),
        "trace": trace,
    }


# Discoverable aliases at the CSV and parser seams.
format_power_trace_row = format_issue146_power_trace_row
parse_issue146_power_trace = read_issue146_power_trace
read_issue146_power_trace_csv = read_issue146_power_trace
read_power_trace_issue146 = read_issue146_power_trace
write_issue146_power_trace_csv = write_issue146_power_trace
write_power_trace_issue146 = write_issue146_power_trace
validate_power_trace_issue146 = validate_issue146_power_trace
decode_gddr_word = _issue137.decode_gddr_word
decode_issue146_gddr_row = _issue137.decode_issue137_gddr_row
decode_gddr_temperature_word = decode_gddr_word


__all__ = [
    "ISSUE137_GDDR_COLUMNS",
    "ISSUE137_GDDR_SOURCE_FIELDS",
    "ISSUE137_POWER_TRACE_COLUMNS",
    "ISSUE137_POWER_TRACE_HEADER",
    "ISSUE137_SOURCE_VERSIONS",
    "ISSUE137_UNIT_PROVENANCE",
    "ISSUE146_ARB_COLUMNS",
    "ISSUE146_ARB_SOURCE_FIELDS",
    "ISSUE146_POWER_TRACE_COLUMNS",
    "ISSUE146_POWER_TRACE_HEADER",
    "ISSUE146_POWER_TRACE_SCHEMA",
    "ISSUE146_POWER_TRACE_SCHEMA_VERSION",
    "ISSUE146_RAW_SOURCE_FIELDS",
    "ISSUE146_UNIT_PROVENANCE",
    "Issue146PowerTraceError",
    "decode_gddr_temperature_word",
    "decode_gddr_word",
    "decode_issue146_gddr_row",
    "format_issue146_power_trace_row",
    "format_power_trace_row",
    "issue146_power_trace_header",
    "issue146_trace_row",
    "parse_issue146_power_trace",
    "parse_issue146_snapshot",
    "read_issue146_power_trace",
    "read_issue146_power_trace_csv",
    "read_issue146_power_trace_mapping",
    "read_issue146_power_trace_snapshot",
    "read_power_trace_issue146",
    "source_versions_verified",
    "validate_issue146_power_trace",
    "validate_power_trace_issue146",
    "write_issue146_power_trace",
    "write_issue146_power_trace_csv",
    "write_power_trace_issue146",
]
