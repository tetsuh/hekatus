"""Strict Issue #137 power-trace records with raw packed GDDR telemetry.

The Issue #121 extended-v1 reader is intentionally left untouched.  This
module owns the Issue #137 schema, which appends the four raw GDDR pair-word
fields to that schema.  A pair word is retained exactly as emitted by tt-smi;
when the pinned source provenance is available, a separate helper decodes all
four byte fields (two dies, bottom and top) without reducing them to a scalar.
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

ISSUE137_POWER_TRACE_SCHEMA = "issue-137-power-trace-v1"
ISSUE137_POWER_TRACE_SCHEMA_VERSION = 1
ISSUE137_GDDR_COLUMNS = (
    "gddr_0_1_temp_c",
    "gddr_2_3_temp_c",
    "gddr_4_5_temp_c",
    "gddr_6_7_temp_c",
)
ISSUE137_GDDR_SOURCE_FIELDS = {
    "gddr_0_1_temp_c": "GDDR_0_1_TEMP",
    "gddr_2_3_temp_c": "GDDR_2_3_TEMP",
    "gddr_4_5_temp_c": "GDDR_4_5_TEMP",
    "gddr_6_7_temp_c": "GDDR_6_7_TEMP",
}
ISSUE137_POWER_TRACE_COLUMNS = (
    *_q2.EXTENDED_POWER_TRACE_COLUMNS,
    *ISSUE137_GDDR_COLUMNS,
)
ISSUE137_POWER_TRACE_HEADER = ",".join(ISSUE137_POWER_TRACE_COLUMNS)

# The values are release-tag commits, not a claim about an installed package.
# They bind the byte positions and the raw serializer used by this reader.
ISSUE137_SOURCE_VERSIONS = {
    "tt_smi": "6.2.1",
    "tt_umd": "0.9.8",
}
ISSUE137_UNIT_PROVENANCE: dict[str, Any] = {
    "schema": "issue137-power-trace-unit-provenance-v1",
    "status": "established",
    "packages": [
        {
            "name": "tt-smi",
            "version": "6.2.1",
            "repository": "tenstorrent/tt-smi",
            "source_commit": "07208d7e783b8bb585440460db53f515b5828f16",
            "source_hash": "07208d7e783b8bb585440460db53f515b5828f16",
            "source_files": ["tt_smi/backend.py"],
        },
        {
            "name": "tt-umd",
            "version": "0.9.8",
            "repository": "tenstorrent/tt-umd",
            "source_commit": "12e28af324fe65223b8f10ff70fc060f4f184214",
            "source_hash": "12e28af324fe65223b8f10ff70fc060f4f184214",
            "source_files": [
                "device/firmware/firmware_info_provider.cpp",
                "device/api/umd/device/types/gddr_telemetry.hpp",
            ],
        },
    ],
    "fields": {
        column: {
            "source_field": source,
            "unit": "degC per die byte",
            "raw_encoding": "four unsigned 8-bit die temperatures in one 32-bit word",
            "packing": (
                "even module bottom bits [7:0], top bits [15:8]; odd module "
                "bottom bits [23:16], top bits [31:24]"
            ),
            "source_files": [
                "tt_smi/backend.py",
                "device/firmware/firmware_info_provider.cpp",
                "device/api/umd/device/types/gddr_telemetry.hpp",
            ],
        }
        for column, source in ISSUE137_GDDR_SOURCE_FIELDS.items()
    },
    "max_gddr_temp": {
        "source_field": "MAX_GDDR_TEMP",
        "unit": "unknown",
        "status": "raw_only",
        "policy": "retain raw; never decode or aggregate",
    },
}

_RAW_GDDR_WORD_RE = re.compile(r"^0x[0-9a-fA-F]{1,8}$")
_MISSING = object()


def source_versions_verified(host_sw_vers: Any) -> bool:
    """Return whether observed host tool versions match the fixed source audit."""
    if not isinstance(host_sw_vers, Mapping):
        return False

    def version_for(*keys: str) -> str | None:
        for key in keys:
            value = host_sw_vers.get(key)
            if isinstance(value, Mapping):
                value = value.get("version")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    return (
        version_for("tt_smi", "tt-smi", "tt_smi_version", "tt-smi-version")
        == ISSUE137_SOURCE_VERSIONS["tt_smi"]
        and version_for("tt_umd", "tt-umd", "tt_umd_version", "tt-umd-version")
        == ISSUE137_SOURCE_VERSIONS["tt_umd"]
    )


class Issue137PowerTraceError(ValueError):
    """Raised when an Issue #137 trace or packed word is not representable."""


def _raw_gddr_word(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise Issue137PowerTraceError(f"{field} must retain a raw hexadecimal word")
    raw = value.strip()
    if _RAW_GDDR_WORD_RE.fullmatch(raw) is None:
        raise Issue137PowerTraceError(
            f"{field} must be a raw 32-bit hexadecimal word (0x...), not a decoded scalar"
        )
    return raw


def _module_names(column: str) -> tuple[str, str]:
    source = ISSUE137_GDDR_SOURCE_FIELDS[column]
    pair = source.removeprefix("GDDR_").removesuffix("_TEMP")
    first, second = pair.split("_")
    return f"GDDR_{first}", f"GDDR_{second}"


def decode_gddr_word(
    raw: Any,
    *,
    module_pair: tuple[int, int] = (0, 1),
    source_established: bool = False,
) -> dict[str, Any]:
    """Decode one packed pair word, retaining all four die temperatures.

    Unknown source provenance deliberately returns no numeric values.  The raw
    word remains available for a later source audit.
    """
    if (
        not isinstance(module_pair, tuple)
        or len(module_pair) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in module_pair)
    ):
        raise Issue137PowerTraceError("module_pair must contain two non-negative integers")
    value = _raw_gddr_word(raw, field="GDDR word")
    if not source_established:
        return {
            "raw": value,
            "unit": "unknown",
            "status": "not_evaluable",
            "channels_c": None,
        }
    packed = int(value, 16)
    first, second = (f"GDDR_{item}" for item in module_pair)
    return {
        "raw": value,
        "unit": "degC",
        "status": "decoded",
        "channels_c": {
            first: {
                "bottom_c": packed & 0xFF,
                "top_c": (packed >> 8) & 0xFF,
            },
            second: {
                "bottom_c": (packed >> 16) & 0xFF,
                "top_c": (packed >> 24) & 0xFF,
            },
        },
    }


def _decode_named_gddr_word(
    column: str,
    raw: Any,
    *,
    source_established: bool = False,
) -> dict[str, Any]:
    names = _module_names(column)
    pair = (int(names[0].split("_")[1]), int(names[1].split("_")[1]))
    return decode_gddr_word(
        raw,
        module_pair=pair,
        source_established=source_established,
    )


def decode_issue137_gddr_row(
    row: Mapping[str, Any], *, source_established: bool = False
) -> dict[str, dict[str, Any]]:
    """Return one structured value for each raw GDDR column in a trace row."""
    result: dict[str, dict[str, Any]] = {}
    for column in ISSUE137_GDDR_COLUMNS:
        value = row.get(column, _MISSING)
        if value is _MISSING:
            raise Issue137PowerTraceError(f"trace row is missing {column}")
        result[column] = _decode_named_gddr_word(
            column, value, source_established=source_established
        )
    return result


def _selected_device_snapshot(
    snapshot: Mapping[str, Any] | str | bytes,
    *,
    board_id: str | None,
    bus_id: str | None,
) -> Mapping[str, Any]:
    return _q2.select_extended_board(snapshot, board_id=board_id, bus_id=bus_id)


def parse_issue137_snapshot(
    snapshot: Mapping[str, Any] | str | bytes,
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
    timestamp_utc: Any = None,
    timestamp_monotonic_ns: int | None = None,
    snapshot_elapsed_ns: int | None = None,
) -> dict[str, Any]:
    """Convert one selected tt-smi snapshot into a complete trace row."""
    row = _q2.parse_extended_snapshot(
        snapshot,
        board_id=board_id,
        bus_id=bus_id,
        timestamp_utc=timestamp_utc,
        timestamp_monotonic_ns=timestamp_monotonic_ns,
        snapshot_elapsed_ns=snapshot_elapsed_ns,
    )
    device = _selected_device_snapshot(snapshot, board_id=board_id, bus_id=bus_id)
    smbus = device.get("smbus_telem")
    if not isinstance(smbus, Mapping):
        raise Issue137PowerTraceError("selected device smbus_telem must be an object")
    for column, source in ISSUE137_GDDR_SOURCE_FIELDS.items():
        raw = smbus.get(source, _MISSING)
        if raw is _MISSING or raw is None:
            raise Issue137PowerTraceError(
                f"required snapshot value smbus_telem.{source!s} is missing"
            )
        row[column] = _raw_gddr_word(raw, field=f"smbus_telem.{source}")
    return row


def issue137_power_trace_header() -> str:
    """Return the exact Issue #137 CSV header."""
    return ISSUE137_POWER_TRACE_HEADER


def _validate_issue137_row(row: Mapping[str, Any]) -> None:
    expected = set(ISSUE137_POWER_TRACE_COLUMNS)
    actual = set(row)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Issue137PowerTraceError(
            f"Issue #137 trace row columns are invalid (missing={missing!r}, extra={extra!r})"
        )
    for column in ISSUE137_GDDR_COLUMNS:
        _raw_gddr_word(row[column], field=column)


def format_issue137_power_trace_row(row: Mapping[str, Any]) -> str:
    """Strictly format one Issue #137 row in schema column order."""
    _validate_issue137_row(row)
    base_columns = tuple(_q2.EXTENDED_POWER_TRACE_COLUMNS)
    base = {column: row[column] for column in base_columns}
    base_fields = next(
        csv.reader(
            io.StringIO(_q2.format_extended_power_trace_row(base), newline="")
        )
    )
    values = [*base_fields, *[row[column] for column in ISSUE137_GDDR_COLUMNS]]
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="").writerow(values)
    return output.getvalue()


def issue137_trace_row(
    snapshot: Mapping[str, Any] | str | bytes,
    **kwargs: Any,
) -> str | None:
    """Return one formatted row, or ``None`` for an unusable snapshot."""
    try:
        return format_issue137_power_trace_row(parse_issue137_snapshot(snapshot, **kwargs))
    except (Issue137PowerTraceError, _q2.ExtendedPowerTraceError):
        return None


def _trace_result_base() -> dict[str, Any]:
    return {
        "schema": ISSUE137_POWER_TRACE_SCHEMA,
        "schema_version": ISSUE137_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(ISSUE137_POWER_TRACE_COLUMNS),
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


def _coerce_issue137_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    if set(row) != set(ISSUE137_POWER_TRACE_COLUMNS):
        missing = sorted(set(ISSUE137_POWER_TRACE_COLUMNS) - set(row))
        extra = sorted(set(row) - set(ISSUE137_POWER_TRACE_COLUMNS))
        raise Issue137PowerTraceError(
            f"Issue #137 trace row {index} columns are invalid "
            f"(missing={missing!r}, extra={extra!r})"
        )
    base = {column: row[column] for column in _q2.EXTENDED_POWER_TRACE_COLUMNS}
    try:
        normalized = _q2._coerce_trace_row(base, index=index)
    except (AttributeError, TypeError, ValueError, _q2.ExtendedPowerTraceError) as exc:
        raise Issue137PowerTraceError(str(exc)) from exc
    for column in ISSUE137_GDDR_COLUMNS:
        normalized[column] = _raw_gddr_word(row[column], field=f"row {index} {column}")
    return normalized


def _base_payload(
    samples: list[dict[str, Any]],
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
            {column: row[column] for column in _q2.EXTENDED_POWER_TRACE_COLUMNS}
            for row in samples
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
    if tuple(reader.fieldnames or ()) != ISSUE137_POWER_TRACE_COLUMNS:
        raise Issue137PowerTraceError(
            f"Issue #137 power trace has unexpected columns {reader.fieldnames!r}"
        )
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if extra in row:
            raise Issue137PowerTraceError(f"Issue #137 trace row {index} has extra fields")
        if any(row[column] is missing for column in ISSUE137_POWER_TRACE_COLUMNS):
            raise Issue137PowerTraceError(f"Issue #137 trace row {index} has missing fields")
        rows.append({column: row[column] for column in ISSUE137_POWER_TRACE_COLUMNS})
    return rows


def _read_issue137_mapping(
    payload: Mapping[str, Any],
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    source_established: bool = False,
) -> dict[str, Any]:
    result = _trace_result_base()
    try:
        if not isinstance(payload, Mapping):
            raise Issue137PowerTraceError("Issue #137 trace must be an object")
        if payload.get("schema") != ISSUE137_POWER_TRACE_SCHEMA:
            raise Issue137PowerTraceError("Issue #137 trace schema is unsupported")
        version = payload.get("schema_version")
        if version != ISSUE137_POWER_TRACE_SCHEMA_VERSION or isinstance(version, bool):
            raise Issue137PowerTraceError("Issue #137 trace schema_version is unsupported")
        if tuple(payload.get("columns", ())) != ISSUE137_POWER_TRACE_COLUMNS:
            raise Issue137PowerTraceError("Issue #137 trace columns are invalid")
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list):
            raise Issue137PowerTraceError("Issue #137 trace samples must be an array")
        rows = [_coerce_issue137_row(row, index) for index, row in enumerate(raw_samples)]
    except (Issue137PowerTraceError, TypeError, ValueError) as exc:
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        return result
    base = _q2.read_extended_power_trace_mapping(
        _base_payload(
            rows,
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
        )
    )
    result.update(base)
    result["schema"] = ISSUE137_POWER_TRACE_SCHEMA
    result["schema_version"] = ISSUE137_POWER_TRACE_SCHEMA_VERSION
    result["columns"] = list(ISSUE137_POWER_TRACE_COLUMNS)
    result["samples"] = rows
    result["unit_provenance"] = ISSUE137_UNIT_PROVENANCE if source_established else None
    result["gddr_source_status"] = "established" if source_established else "unknown"
    result["gddr_decoded"] = [
        decode_issue137_gddr_row(row, source_established=source_established)
        for row in rows
    ]
    failures = list(base.get("errors", ()))
    result["valid_row_count"] = len(rows) if not failures else max(
        0, len(rows) - len(failures)
    )
    result["invalid_row_count"] = len(failures)
    if not source_established:
        result["gddr_source_status"] = "unknown"
    if failures:
        result["valid"] = False
        result["errors"] = list(dict.fromkeys(failures))
        result["error"] = result["errors"][0]
    else:
        result["errors"] = []
        result["valid"] = bool(rows) and base.get("valid") is True
        if not result["valid"]:
            result["error"] = "Issue #137 trace has no complete bounded interval"
            result["errors"] = [result["error"]]
    return result


def read_issue137_power_trace(
    path: Path,
    *,
    run_start_monotonic_ns: int | None = None,
    run_end_monotonic_ns: int | None = None,
    source_established: bool = False,
) -> dict[str, Any]:
    """Read one strict Issue #137 CSV trace and recompute Q2 aggregates."""
    result = _trace_result_base()
    try:
        raw = Path(path).read_text(encoding="utf-8")
        rows = _read_rows_from_text(raw)
        payload = {
            "schema": ISSUE137_POWER_TRACE_SCHEMA,
            "schema_version": ISSUE137_POWER_TRACE_SCHEMA_VERSION,
            "columns": list(ISSUE137_POWER_TRACE_COLUMNS),
            "samples": rows,
        }
        result = _read_issue137_mapping(
            payload,
            run_start_monotonic_ns=run_start_monotonic_ns,
            run_end_monotonic_ns=run_end_monotonic_ns,
            source_established=source_established,
        )
        result["byte_count"] = len(raw.encode("utf-8"))
        result["readable"] = True
        result["csv_row_count"] = len(rows)
        result["nonempty"] = bool(rows)
    except (OSError, UnicodeError, csv.Error, Issue137PowerTraceError) as exc:
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
    return result


def read_issue137_power_trace_mapping(
    payload: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Read an in-memory trace with the same strict schema checks."""
    return _read_issue137_mapping(payload, **kwargs)


def read_issue137_power_trace_snapshot(path: Path, **kwargs: Any) -> dict[str, Any]:
    """Read CSV or a JSON sidecar carrying complete Issue #137 rows."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        return read_issue137_power_trace(path, **kwargs)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result = _trace_result_base()
        result["errors"] = [str(exc)]
        result["error"] = str(exc)
        return result
    if not isinstance(payload, Mapping):
        result = _trace_result_base()
        result["errors"] = ["Issue #137 trace sidecar must be an object"]
        result["error"] = result["errors"][0]
        return result
    return _read_issue137_mapping(payload, **kwargs)


def write_issue137_power_trace(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a complete Issue #137 CSV trace without changing input rows."""
    serialized = [format_issue137_power_trace_row(row) for row in rows]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        ISSUE137_POWER_TRACE_HEADER + "\n" + "\n".join(serialized) + "\n",
        encoding="utf-8",
        newline="",
    )
    return destination


def validate_issue137_power_trace(
    trace_or_path: Mapping[str, Any] | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a fail-closed validation report for a new trace."""
    trace = (
        read_issue137_power_trace_mapping(trace_or_path, **kwargs)
        if isinstance(trace_or_path, Mapping)
        else read_issue137_power_trace_snapshot(trace_or_path, **kwargs)
    )
    failures = list(trace.get("errors", ()))
    if trace.get("schema") != ISSUE137_POWER_TRACE_SCHEMA:
        failures.append("Issue #137 trace schema is unsupported")
    if trace.get("columns") != list(ISSUE137_POWER_TRACE_COLUMNS):
        failures.append("Issue #137 trace columns are invalid")
    return {
        "valid": bool(trace.get("valid") is True and not failures),
        "schema": ISSUE137_POWER_TRACE_SCHEMA,
        "schema_version": ISSUE137_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(ISSUE137_POWER_TRACE_COLUMNS),
        "failures": list(dict.fromkeys(str(item) for item in failures)),
        "trace": trace,
    }


# Discoverable aliases for callers that use CSV/parse terminology.
format_power_trace_row = format_issue137_power_trace_row
parse_issue137_power_trace = read_issue137_power_trace
read_issue137_power_trace_csv = read_issue137_power_trace
read_power_trace_issue137 = read_issue137_power_trace
write_issue137_power_trace_csv = write_issue137_power_trace
write_power_trace_issue137 = write_issue137_power_trace
validate_power_trace_issue137 = validate_issue137_power_trace
decode_gddr_temperature_word = decode_gddr_word

__all__ = [
    "ISSUE137_GDDR_COLUMNS",
    "ISSUE137_GDDR_SOURCE_FIELDS",
    "ISSUE137_POWER_TRACE_COLUMNS",
    "ISSUE137_POWER_TRACE_HEADER",
    "ISSUE137_POWER_TRACE_SCHEMA",
    "ISSUE137_POWER_TRACE_SCHEMA_VERSION",
    "ISSUE137_SOURCE_VERSIONS",
    "ISSUE137_UNIT_PROVENANCE",
    "Issue137PowerTraceError",
    "decode_gddr_temperature_word",
    "decode_gddr_word",
    "decode_issue137_gddr_row",
    "format_issue137_power_trace_row",
    "format_power_trace_row",
    "issue137_power_trace_header",
    "issue137_trace_row",
    "parse_issue137_power_trace",
    "parse_issue137_snapshot",
    "read_issue137_power_trace",
    "read_issue137_power_trace_csv",
    "read_issue137_power_trace_mapping",
    "read_issue137_power_trace_snapshot",
    "read_power_trace_issue137",
    "source_versions_verified",
    "validate_issue137_power_trace",
    "validate_power_trace_issue137",
    "write_issue137_power_trace",
    "write_issue137_power_trace_csv",
    "write_power_trace_issue137",
]
