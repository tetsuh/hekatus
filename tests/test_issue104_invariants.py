"""Board-free tests for the Issue #104 resident and wrap invariants."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from enodia.tt.bench.resident_harness import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    ResidentConfig,
    build_measurement_record,
    timestamp_digest,
    validate_outlier_analysis,
)
from enodia.tt.bench.telemetry import parse_power_trace
from tools.analyze_issue104_wrap import validate_record_invariants

RUN_START = "2026-01-01T00:00:00.500000+00:00"
RUN_END = "2026-01-01T00:00:01.500000+00:00"


def _environment(*, mode: str = "explicit", image: str = AUDITED_CLOCK_SOURCE_IMAGE) -> dict:
    if mode == "off":
        sampler = {
            "mode": "off",
            "interval_seconds": None,
            "power_trace": "absent_by_design",
            "timing_evidence": "diagnostic_only",
        }
    else:
        sampler = {
            "mode": mode,
            "interval_seconds": 1.0 if mode == "explicit" else 2.0,
            "power_trace": "required",
            "timing_evidence": "available",
        }
    return {
        "board": {"serial": "board", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": image,
        "image_pinned": True,
        "harness_commit": "a" * 40,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": sampler,
        "aiclk_mhz_observed": [800],
    }


def _config() -> ResidentConfig:
    return ResidentConfig(
        frame_count=3,
        frame_interval_ticks=1_000_000,
        producer_core=(0, 0),
        consumer_core=(1, 0),
        ring_pages=4,
        work_per_frame=64,
        designated_timestamp_core=(1, 0),
        cycle_budget=10_000,
        outer_timeout_seconds=60,
        fixed_work_ticks_per_frame=100,
        budget_aiclk_mhz=1_350,
    )


def _write_trace(path: Path, rows: str) -> None:
    path.write_text("timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n" + rows)


def _complete_trace(path: Path) -> None:
    _write_trace(
        path,
        "2026-01-01T00:00:00+00:00,75,800,60\n"
        "2026-01-01T00:00:01+00:00,76,1350,61\n"
        "2026-01-01T00:00:02+00:00,75,1350,61\n",
    )


def _record_with_trace(tmp_path: Path, timestamps: list[int] | None = None) -> tuple[dict, Path]:
    trace = tmp_path / "power-run.csv"
    _complete_trace(trace)
    values = [1_000, 2_000, 3_000] if timestamps is None else timestamps
    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=values,
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_environment(),
        power_trace=trace.name,
        power_trace_path=trace,
        run_start=RUN_START,
        run_end=RUN_END,
        timing_evidence=True,
    )
    return record, trace


def test_power_trace_parser_accepts_complete_coverage_and_hashes_actual_bytes(tmp_path):
    trace = tmp_path / "power.csv"
    _complete_trace(trace)

    parsed = parse_power_trace(trace, run_start=RUN_START, run_end=RUN_END)

    assert parsed["readable"] is True
    assert parsed["sample_count"] == parsed["csv_row_count"] == parsed["valid_row_count"] == 3
    assert parsed["coverage_complete"] is True
    assert parsed["aiclk_source"] == "run_trace_samples"
    assert parsed["in_run_valid_row_count"] == 1
    assert parsed["aiclk_mhz"] == 1350
    assert parsed["sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "not-a-timestamp,75,1350,60\n",
    ],
    ids=["header-only", "invalid-row"],
)
def test_power_trace_header_only_or_invalid_row_is_not_usable(tmp_path, contents):
    trace = tmp_path / "power.csv"
    if contents:
        _write_trace(trace, contents)
    else:
        trace.write_text("timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n")

    parsed = parse_power_trace(trace, run_start=RUN_START, run_end=RUN_END)

    assert parsed["coverage_complete"] is False
    assert parsed["valid_row_count"] < parsed["sample_count"] or parsed["sample_count"] == 0
    assert parsed["aiclk_source"] == "no_valid_in_run_samples"


def test_power_trace_incomplete_first_or_last_coverage_is_rejected(tmp_path):
    trace = tmp_path / "power.csv"
    _write_trace(
        trace,
        "2026-01-01T00:00:01+00:00,75,1350,60\n"
        "2026-01-01T00:00:02+00:00,75,1350,60\n",
    )

    parsed = parse_power_trace(trace, run_start=RUN_START, run_end=RUN_END)

    assert parsed["covers_run_start"] is False
    assert parsed["coverage_complete"] is False


def test_builder_uses_only_in_run_aiclk_and_publishes_power_provenance(tmp_path):
    record, trace = _record_with_trace(tmp_path)

    assert record["timing_evidence"] is True
    assert record["clock"]["aiclk_source"] == "run_trace_samples"
    assert record["power_trace_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert record["power_trace_sample_count"] == 3
    assert record["power_clock_provenance"]["coverage_complete"] is True
    assert record["power_clock_provenance"]["in_run_valid_row_count"] == 1


def test_builder_defaults_timing_evidence_false_even_with_complete_trace(tmp_path):
    trace = tmp_path / "power-default.csv"
    _complete_trace(trace)
    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_environment(),
        power_trace=trace.name,
        power_trace_path=trace,
        run_start=RUN_START,
        run_end=RUN_END,
    )

    assert record["timing_evidence"] is False
    assert record["timing_evidence_reason"] == "timing_evidence_not_requested"


def test_builder_uses_configured_aiclk_when_sampled_trace_has_no_in_run_rows(tmp_path):
    trace = tmp_path / "power.csv"
    _write_trace(
        trace,
        "2026-01-01T00:00:00+00:00,75,1350,60\n"
        "2026-01-01T00:00:10+00:00,75,1350,60\n",
    )
    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=800,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_environment(),
        power_trace=trace.name,
        power_trace_path=trace,
        run_start="2026-01-01T00:00:01+00:00",
        run_end="2026-01-01T00:00:09+00:00",
        timing_evidence=True,
    )

    assert record["timing_evidence"] is False
    assert record["timing_evidence_reason"] == "power_trace_no_valid_in_run_rows"
    assert record["clock"]["aiclk_mhz"] == 1_350
    assert record["clock"]["aiclk_source"] == "configured"
    assert record["outlier_analysis"]["aiclk_mhz_for_elapsed_seconds"] == 1_350
    assert record["outlier_analysis"]["aiclk_source"] == "configured"
    assert record["outlier_analysis"]["run_elapsed_seconds"] == pytest.approx(
        2_000 / (1_350 * 1_000_000)
    )
    assert record["outlier_analysis"]["run_elapsed_seconds"] != pytest.approx(
        2_000 / (800 * 1_000_000)
    )
    assert record["power_clock_provenance"]["aiclk_source"] == "no_valid_in_run_samples"


def test_analyzer_reports_sampler_trace_mismatch_and_power_hash_count_mismatches(tmp_path):
    record, trace = _record_with_trace(tmp_path)
    record["environment"]["telemetry_sampler"]["power_trace"] = "absent_by_design"
    record["power_trace_sha256"] = "0" * 64
    record["raw_timestamps"]["count"] = 99
    record["histogram"]["N"] = 1
    timestamps = [1_000, 2_000, 3_000]
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    metadata = {"count": len(timestamps), "sha256": hashlib.sha256(raw).hexdigest()}

    report = validate_record_invariants(
        record,
        raw_metadata=metadata,
        timestamps=timestamps,
        power_trace_path=trace,
    )

    assert report["valid"] is False
    assert any("sampled sampler mode requires" in failure for failure in report["failures"])
    assert any("SHA-256" in failure for failure in report["failures"])
    assert any("raw_timestamps.count" in failure for failure in report["failures"])
    assert any("histogram N" in failure for failure in report["failures"])


def test_analyzer_reports_actual_raw_hash_mismatch_and_keeps_count_strict(tmp_path):
    record, trace = _record_with_trace(tmp_path)
    timestamps = [1_000, 2_000, 3_000]
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    metadata = {"count": len(timestamps), "sha256": hashlib.sha256(raw).hexdigest()}
    record["raw_timestamps"]["sha256"] = "f" * 64

    report = validate_record_invariants(
        record,
        raw_metadata=metadata,
        timestamps=timestamps,
        power_trace_path=trace,
    )

    assert report["valid"] is False
    assert any("raw_timestamps.sha256" in failure for failure in report["failures"])


def test_builder_and_analyzer_reject_pair_count_start_length_mismatch(tmp_path):
    outlier = {
        "pair_count": 2,
        "frame_start_indices": [1],
        "pairs": [],
    }
    with pytest.raises(ValueError, match="pair_count must equal frame_start_indices"):
        validate_outlier_analysis(outlier)

    record, trace = _record_with_trace(tmp_path)
    record["outlier_analysis"] = outlier
    report = validate_record_invariants(record, power_trace_path=trace)

    assert report["valid"] is False
    assert any("frame_start_indices length" in failure for failure in report["failures"])


def test_analyzer_accepts_valid_record_power_counts_and_n_plus_one(tmp_path):
    record, trace = _record_with_trace(tmp_path)
    timestamps = [1_000, 2_000, 3_000]
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    metadata = {"count": 3, "sha256": hashlib.sha256(raw).hexdigest()}

    report = validate_record_invariants(
        record,
        raw_metadata=metadata,
        timestamps=timestamps,
        power_trace_path=trace,
    )

    assert report["valid"] is True
    assert report["checks"]["histogram_n_plus_one"]["ok"] is True
    assert report["checks"]["power_trace"]["ok"] is True
    assert record["raw_timestamps"] == timestamp_digest(timestamps)
    json.dumps(report, allow_nan=False)
