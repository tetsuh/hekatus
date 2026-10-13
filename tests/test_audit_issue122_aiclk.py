"""Board-free tests for the Issue #122 AICLK audit helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import pytest

from tools.audit_issue122_aiclk import (
    IntervalBounds,
    TraceSample,
    _coverage_rows,
    _record_trace_facts,
    classify_aiclk_values,
    classify_intermediate_position,
    left_held_time_weighted_aiclk,
    read_trace,
    resolve_trace,
)

UTC = dt.UTC


def _timestamp(seconds: float) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def test_aiclk_classification_recognizes_endpoint_only_trace():
    classification = classify_aiclk_values([800.0, 1350.0, 800.0])

    assert classification.label == "two-valued {800,1350}"
    assert classification.minimum_intermediate_mhz is None
    assert classification.intermediate_position is None


def test_intermediate_position_uses_csv_thirds():
    assert classify_intermediate_position(0, 9) == "beginning"
    assert classify_intermediate_position(4, 9) == "middle"
    assert classify_intermediate_position(8, 9) == "end"
    assert classify_aiclk_values([1018.0, 800.0, 1350.0]).intermediate_position == "beginning"
    assert classify_aiclk_values([800.0, 1018.0, 1350.0]).intermediate_position == "middle"
    assert classify_aiclk_values([800.0, 1350.0, 1018.0]).intermediate_position == "end"


def test_left_held_weighted_aiclk_uses_each_sample_until_next_sample():
    rows = (
        TraceSample(_timestamp(0), 800.0, 2),
        TraceSample(_timestamp(2), 1200.0, 3),
        TraceSample(_timestamp(5), 1350.0, 4),
    )

    weighted = left_held_time_weighted_aiclk(rows, _timestamp(1), _timestamp(5))

    assert weighted == pytest.approx(1100.0)


def test_left_held_weighted_aiclk_refuses_uncovered_interval():
    rows = (TraceSample(_timestamp(1), 800.0, 2), TraceSample(_timestamp(2), 1350.0, 3))

    assert left_held_time_weighted_aiclk(rows, _timestamp(0), _timestamp(2)) is None
    assert left_held_time_weighted_aiclk(rows, _timestamp(1), _timestamp(3)) is None


def test_read_trace_classifies_endpoint_csv_and_weights_explicit_interval(tmp_path: Path):
    csv_path = tmp_path / "synthetic-power.csv"
    csv_path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,40,800,50\n"
        "2026-01-01T00:00:02+00:00,70,1350,51\n"
        "2026-01-01T00:00:05+00:00,40,800,50\n",
        encoding="utf-8",
    )

    trace = read_trace(csv_path)
    classification = classify_aiclk_values([sample.aiclk_mhz for sample in trace.samples])

    assert trace.csv_row_count == 3
    assert classification.label == "two-valued {800,1350}"
    assert classification.unique_mhz == (800.0, 1350.0)
    assert left_held_time_weighted_aiclk(
        trace.samples,
        _timestamp(1),
        _timestamp(5),
    ) == pytest.approx(1212.5)


def test_trace_pairing_prefers_sha256_provenance_over_stem(tmp_path: Path):
    csv_path = tmp_path / "committed-power.csv"
    csv_path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,40,800,50\n"
        "2026-01-01T00:00:01+00:00,70,1350,51\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    record = {
        "power_trace": "external/power-snapshot.csv",
        "power_trace_sha256": digest,
    }

    resolution = resolve_trace(tmp_path, tmp_path / "record.json", record, {digest: (csv_path,)})

    assert resolution.path == csv_path
    assert resolution.method == "SHA-256 provenance"
    assert read_trace(csv_path).csv_row_count == 2


def test_explicit_interval_extrema_use_positive_left_held_segments(tmp_path: Path):
    csv_path = tmp_path / "boundary-power.csv"
    csv_path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,40,800,50\n"
        "2026-01-01T00:00:02+00:00,70,1200,51\n"
        "2026-01-01T00:00:04+00:00,40,1350,50\n",
        encoding="utf-8",
    )
    trace = read_trace(csv_path)
    interval = IntervalBounds(_timestamp(1), _timestamp(4), "run", "test")

    inside, complete = _coverage_rows(trace, interval)
    evidence, minimum, maximum, weighted, weighted_is_exact = _record_trace_facts(
        trace, interval
    )

    assert complete is True
    assert [sample.aiclk_mhz for sample in inside] == [800.0, 1200.0]
    assert evidence == "exact run interval"
    assert minimum == 800.0
    assert maximum == 1200.0
    assert weighted == pytest.approx(1066.6666667)
    assert weighted_is_exact is True
