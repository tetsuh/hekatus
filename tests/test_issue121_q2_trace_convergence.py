"""Regression tests for Issue #134 extended trace convergence findings."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
from pathlib import Path

import pytest

from enodia.tt.bench.power_trace_extended import (
    EXTENDED_POWER_TRACE_COLUMNS,
    EXTENDED_POWER_TRACE_HEADER,
    EXTENDED_POWER_TRACE_SCHEMA,
    EXTENDED_POWER_TRACE_SCHEMA_VERSION,
    format_extended_power_trace_row,
    read_extended_power_trace_csv,
    read_extended_power_trace_snapshot,
    validate_extended_power_trace,
)


def _trace_row(
    timestamp_monotonic_ns: int,
    *,
    board_power_w: float,
    power_w: float | None = None,
    throttle_reason: str | None = None,
) -> dict[str, object]:
    if power_w is None:
        power_w = board_power_w
    seconds = timestamp_monotonic_ns // 1_000_000_000
    return {
        "timestamp_utc": f"2026-01-01T00:00:{seconds:02d}+00:00",
        "power_w": power_w,
        "aiclk_mhz": 1000,
        "asic_temp_c": 40,
        "timestamp_monotonic_ns": timestamp_monotonic_ns,
        "snapshot_elapsed_ns": 10,
        "current_a": 2.0,
        "voltage_v": 12.0,
        "board_power_w": board_power_w,
        "fan_rpm": 1000,
        "heartbeat": 1,
        "therm_trip_count": 0,
        "throttle_status": "active" if throttle_reason is not None else None,
        "throttle_reason": throttle_reason,
    }


def _write_sidecar(path: Path, samples: list[dict[str, object]], **overrides: object) -> None:
    payload: dict[str, object] = {
        "schema": EXTENDED_POWER_TRACE_SCHEMA,
        "schema_version": EXTENDED_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "samples": samples,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def test_extended_row_csv_escaping_round_trips_throttle_labels(tmp_path: Path) -> None:
    status = "active, 温度\n状態"
    reason = "熱制限, 二次\n制限"
    first = _trace_row(0, board_power_w=10.0, throttle_reason=reason)
    first["throttle_status"] = status
    second = _trace_row(2_000_000_000, board_power_w=20.0)

    serialized = format_extended_power_trace_row(first)
    parsed_fields = next(csv.reader(io.StringIO(serialized, newline="")))
    assert len(parsed_fields) == len(EXTENDED_POWER_TRACE_COLUMNS) == 14
    assert parsed_fields[EXTENDED_POWER_TRACE_COLUMNS.index("throttle_status")] == status
    assert parsed_fields[EXTENDED_POWER_TRACE_COLUMNS.index("throttle_reason")] == reason

    path = tmp_path / "extended.csv"
    path.write_text(
        EXTENDED_POWER_TRACE_HEADER
        + "\n"
        + serialized
        + "\n"
        + format_extended_power_trace_row(second)
        + "\n",
        encoding="utf-8",
    )
    trace = read_extended_power_trace_csv(path)

    assert trace["valid"] is True
    assert trace["samples"][0]["throttle_status"] == status
    assert trace["samples"][0]["throttle_reason"] == reason


@pytest.mark.parametrize("field_count", (12, 13, 15))
def test_extended_csv_reader_rejects_non_complete_physical_records(
    tmp_path: Path, field_count: int
) -> None:
    serialized = format_extended_power_trace_row(_trace_row(0, board_power_w=10.0))
    fields = next(csv.reader(io.StringIO(serialized)))
    if field_count < len(EXTENDED_POWER_TRACE_COLUMNS):
        fields = fields[:field_count]
    else:
        fields.append("unexpected-extra-field")
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(fields)

    path = tmp_path / f"width-{field_count}.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + buffer.getvalue() + "\n")

    trace = read_extended_power_trace_csv(path)

    assert trace["readable"] is False
    assert trace["valid"] is False
    assert trace["errors"] == [trace["error"]]
    assert "physical field count" in trace["error"]


def test_extended_json_sidecar_reader_aggregates_left_held_samples(tmp_path: Path) -> None:
    path = tmp_path / "extended.json"
    _write_sidecar(
        path,
        [
            _trace_row(0, board_power_w=10.0),
            _trace_row(2_000_000_000, board_power_w=36.0),
            _trace_row(5_000_000_000, board_power_w=22.0),
        ],
    )

    trace = read_extended_power_trace_snapshot(
        path,
        run_start_monotonic_ns=1_000_000_000,
        run_end_monotonic_ns=5_000_000_000,
    )

    assert trace["valid"] is True
    assert trace["sample_count"] == 3
    assert trace["duration_ns"] == 4_000_000_000
    assert trace["segment_count"] == 2
    board_power = trace["aggregates"]["board_power_w"]
    assert board_power["min"] == 10.0
    assert board_power["max"] == 36.0
    assert board_power["time_weighted"] == pytest.approx(29.5)


def _valid_reader_trace(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "mapping.csv"
    path.write_text(
        EXTENDED_POWER_TRACE_HEADER
        + "\n"
        + format_extended_power_trace_row(_trace_row(0, board_power_w=10.0))
        + "\n"
        + format_extended_power_trace_row(
            _trace_row(2_000_000_000, board_power_w=20.0)
        )
        + "\n"
    )
    return read_extended_power_trace_csv(path)


def test_extended_validator_accepts_a_valid_reader_mapping(tmp_path: Path) -> None:
    trace = _valid_reader_trace(tmp_path)

    report = validate_extended_power_trace(trace)

    assert report["valid"] is True
    assert report["failures"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "sample_required_field",
        "sample_nan",
        "samples_empty",
        "sample_count",
        "timestamp_reverse",
        "run_end_beyond_last",
        "aggregates_missing",
        "aggregate_value",
        "duration_ns",
    ],
    ids=[
        "sample-required-field",
        "sample-nan",
        "samples-empty",
        "sample-count",
        "timestamp-reverse",
        "run-end-beyond-last",
        "aggregates-missing",
        "aggregate-value",
        "duration-ns",
    ],
)
def test_extended_validator_rechecks_mapping_samples_and_summary(
    tmp_path: Path, mutation: str
) -> None:
    trace = copy.deepcopy(_valid_reader_trace(tmp_path))
    if mutation == "sample_required_field":
        trace["samples"][0].pop("board_power_w")
    elif mutation == "sample_nan":
        trace["samples"][0]["board_power_w"] = math.nan
    elif mutation == "samples_empty":
        trace["samples"] = []
    elif mutation == "sample_count":
        trace["sample_count"] += 1
    elif mutation == "timestamp_reverse":
        trace["samples"].reverse()
    elif mutation == "run_end_beyond_last":
        trace["run_end_monotonic_ns"] = 3_000_000_000
    elif mutation == "aggregates_missing":
        trace["aggregates"].pop("board_power_w")
    elif mutation == "aggregate_value":
        trace["aggregates"]["board_power_w"]["min"] = 99.0
    elif mutation == "duration_ns":
        trace["duration_ns"] += 1
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    report = validate_extended_power_trace(trace)

    assert report["valid"] is False
    assert report["failures"]


def test_extended_reader_rejects_out_of_range_monotonic_timestamp_without_escaping(
    tmp_path: Path,
) -> None:
    row = _trace_row(0, board_power_w=1e308)
    fields = next(csv.reader(io.StringIO(format_extended_power_trace_row(row))))
    fields[EXTENDED_POWER_TRACE_COLUMNS.index("timestamp_monotonic_ns")] = str(1 << 63)
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(fields)
    path = tmp_path / "out-of-range.csv"
    path.write_text(EXTENDED_POWER_TRACE_HEADER + "\n" + buffer.getvalue() + "\n")

    trace = read_extended_power_trace_csv(path)

    assert trace["valid"] is False
    assert trace["invalid_row_count"] == 1
    assert "non-negative integer" in trace["error"]
    assert validate_extended_power_trace(trace)["valid"] is False


def test_extended_reader_keeps_extreme_finite_power_time_weighted(tmp_path: Path) -> None:
    path = tmp_path / "extreme-power.csv"
    path.write_text(
        EXTENDED_POWER_TRACE_HEADER
        + "\n"
        + format_extended_power_trace_row(_trace_row(0, board_power_w=1e308))
        + "\n"
        + format_extended_power_trace_row(_trace_row(2_000_000_000, board_power_w=1e308))
        + "\n"
    )

    trace = read_extended_power_trace_csv(
        path, run_start_monotonic_ns=0, run_end_monotonic_ns=2_000_000_000
    )

    assert trace["valid"] is True
    weighted = trace["aggregates"]["board_power_w"]["time_weighted"]
    assert math.isfinite(weighted)
    assert weighted == pytest.approx(1e308)


def test_extended_validator_rejects_a_nonfinite_aggregate(tmp_path: Path) -> None:
    path = tmp_path / "finite.csv"
    path.write_text(
        EXTENDED_POWER_TRACE_HEADER
        + "\n"
        + format_extended_power_trace_row(_trace_row(0, board_power_w=10.0))
        + "\n"
        + format_extended_power_trace_row(_trace_row(2_000_000_000, board_power_w=20.0))
        + "\n"
    )
    trace = read_extended_power_trace_csv(path)
    trace["aggregates"]["board_power_w"]["time_weighted"] = math.inf

    report = validate_extended_power_trace(trace)

    assert report["valid"] is False
    assert any("finite" in failure for failure in report["failures"])


@pytest.mark.parametrize("failure", ("schema", "malformed_sample"))
def test_extended_json_sidecar_reader_rejects_invalid_payload(
    tmp_path: Path, failure: str
) -> None:
    path = tmp_path / f"invalid-{failure}.json"
    samples = [_trace_row(0, board_power_w=10.0)]
    if failure == "schema":
        _write_sidecar(path, samples, schema="unsupported-schema")
    else:
        samples[0].pop("board_power_w")
        _write_sidecar(path, samples)

    trace = read_extended_power_trace_snapshot(path)

    assert trace["readable"] is False
    assert trace["valid"] is False
    if failure == "schema":
        assert "schema is not supported" in trace["error"]
    else:
        assert "board_power_w" in trace["error"]


def test_extended_json_sidecar_rejects_boolean_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "boolean-version.json"
    _write_sidecar(
        path,
        [_trace_row(0, board_power_w=10.0)],
        schema_version=True,
    )

    trace = read_extended_power_trace_snapshot(path)
    report = validate_extended_power_trace(path)

    assert trace["readable"] is False
    assert trace["valid"] is False
    assert report["valid"] is False
    assert report["failures"]


def test_extended_json_sidecar_reader_rejects_over_limit_integer_token(
    tmp_path: Path,
) -> None:
    path = tmp_path / "over-limit-integer.json"
    payload = {
        "schema": EXTENDED_POWER_TRACE_SCHEMA,
        "schema_version": EXTENDED_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(EXTENDED_POWER_TRACE_COLUMNS),
        "samples": [_trace_row(0, board_power_w=10.0)],
    }
    serialized = json.dumps(payload)
    oversized_integer = "1" * 4301
    needle = '"timestamp_monotonic_ns": 0'
    assert needle in serialized
    path.write_text(
        serialized.replace(
            needle,
            f'"timestamp_monotonic_ns": {oversized_integer}',
            1,
        )
    )

    trace = read_extended_power_trace_snapshot(path)

    assert trace["readable"] is False
    assert trace["valid"] is False
    assert trace["errors"] == [trace["error"]]

    report = validate_extended_power_trace(path)
    assert report["valid"] is False
