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
    _left_held_positive_segments,
    _record_trace_facts,
    _report_input_data,
    classify_aiclk_values,
    classify_intermediate_position,
    left_held_time_weighted_aiclk,
    read_trace,
    render_report,
    resolve_trace,
)

UTC = dt.UTC


def _timestamp(seconds: float) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def _write_trace(path: Path, values: tuple[int, ...] = (800, 1350)) -> str:
    rows = ["timestamp_utc,power_w,aiclk_mhz,asic_temp_c"]
    rows.extend(
        f"2026-01-01T00:00:0{index}+00:00,40,{value},50"
        for index, value in enumerate(values)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_left_held_interval_allows_duplicate_timestamp_with_decreasing_aiclk():
    rows = (
        TraceSample(_timestamp(0), 800.0, 2),
        TraceSample(_timestamp(1), 1350.0, 3),
        TraceSample(_timestamp(1), 1200.0, 4),
        TraceSample(_timestamp(3), 900.0, 5),
    )

    segments = _left_held_positive_segments(rows, _timestamp(0), _timestamp(3))

    assert [(segment.row_index, segment.aiclk_mhz, segment.seconds) for segment in segments] == [
        (0, 800.0, 1.0),
        (2, 1200.0, 2.0),
    ]
    assert left_held_time_weighted_aiclk(
        rows, _timestamp(0), _timestamp(3)
    ) == pytest.approx((800.0 + 2.0 * 1200.0) / 3.0)


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
    digest = _write_trace(csv_path)
    record = {
        "power_trace": "external/power-snapshot.csv",
        "power_trace_sha256": digest,
    }

    resolution = resolve_trace(tmp_path, tmp_path / "record.json", record, {digest: (csv_path,)})

    assert resolution.path == csv_path
    assert resolution.method == "SHA-256 provenance"
    assert read_trace(csv_path).csv_row_count == 2


def test_trace_stem_fallback_rejects_hash_mismatch(tmp_path: Path):
    candidate = tmp_path / "record-summary-power.csv"
    _write_trace(candidate)
    expected_hash = hashlib.sha256(b"different committed trace").hexdigest()

    resolution = resolve_trace(
        tmp_path,
        tmp_path / "record-summary.json",
        {
            "power_trace": "external/record-power.csv",
            "power_trace_sha256": expected_hash,
        },
        {},
    )

    assert resolution.path is None
    assert resolution.method == "unresolved"
    assert resolution.sha256 == expected_hash


def test_trace_stem_fallback_checks_each_candidate_hash(tmp_path: Path):
    _write_trace(tmp_path / "record-summary-power.csv", (800,))
    expected_hash = _write_trace(tmp_path / "record-power.csv")

    resolution = resolve_trace(
        tmp_path,
        tmp_path / "record-summary.json",
        {"power_trace_sha256": expected_hash},
        {},
    )

    assert resolution.path == tmp_path / "record-power.csv"
    assert resolution.method == "record-stem fallback"
    assert resolution.sha256 == expected_hash


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


def test_report_lists_per_record_conversion_impact_for_every_varying_trace():
    root = Path("docs/measurements")
    report = render_report(root)
    section_start = report.index("## Per-record conversion impact for varying AICLK traces")
    section_end = report.index("## JSON record inventory", section_start)
    section = report[section_start:section_end]
    _, _, _, analyses, _ = _report_input_data(root)
    varying = [
        analysis
        for analysis in analyses
        if analysis.classification is not None
        and len(analysis.classification.unique_mhz) > 1
    ]

    assert f"every one of the {len(varying)} committed JSON records" in section
    assert sum(line.startswith("| 20") for line in section.splitlines()) == len(varying)
    assert all(f"| {analysis.path.name} |" in section for analysis in varying)
    assert "persisted semantic conversion; configured-budget policy" in section
    assert (
        "persisted semantic conversion; fixed 1350-MHz normalization; "
        "configured-budget policy"
    ) in section
    assert "Fixed-reference derived metrics retained" in section
    assert "no AICLK conversion (host perf_counter)" in section
    assert "legacy/unverifiable" in section
    assert "not verifiable from retained interval evidence" in section
    assert "Selected 1050.914 MHz" in section
    assert "No; the existing #88" in section
