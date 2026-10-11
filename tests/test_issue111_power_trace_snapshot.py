"""Board-free coverage for Issue #111 power-trace snapshots."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from enodia.tt.bench import telemetry
from enodia.tt.bench.power_trace_snapshot import (
    EXPLICIT_FINAL_SAMPLE,
    ONE_INTERVAL_BOUND,
    atomic_power_trace_snapshot,
    read_power_trace_snapshot,
)
from enodia.tt.bench.resident_harness import (
    ResidentConfig,
    build_measurement_record,
    validate_resident_record,
)
from enodia.tt.bench.telemetry import parse_power_trace

RUN_START = "2026-01-01T00:00:00+00:00"
RUN_END = "2026-01-01T00:00:01.500000+00:00"
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"


def _config() -> ResidentConfig:
    return ResidentConfig(
        frame_count=3,
        frame_interval_ticks=1_000,
        producer_core=(0, 0),
        consumer_core=(1, 0),
        ring_pages=4,
        work_per_frame=64,
        designated_timestamp_core=(1, 0),
        cycle_budget=10_000,
        outer_timeout_seconds=60,
        fixed_work_ticks_per_frame=100,
    )


def _environment() -> dict[str, object]:
    return {
        "board": {"serial": "board-free-fixture", "board_type": "synthetic"},
        "firmware": {"fw_bundle_version": "fixture"},
        "kmd_version": "fixture",
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True,
        "harness_commit": "0123456789abcdef",
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {
            "mode": "default",
            "interval_seconds": 2.0,
            "power_trace": "required",
            "timing_evidence": "available",
        },
    }


def _write_trace(path: Path, *timestamps: str) -> None:
    path.write_text(
        CSV_HEADER
        + "".join(f"{timestamp},75,1350,60\n" for timestamp in timestamps)
    )


def test_one_interval_bound_accepts_inclusive_pre_run_end_boundary(tmp_path: Path):
    trace = tmp_path / "power-run-111.csv"
    _write_trace(trace, RUN_START)

    accepted = parse_power_trace(
        trace,
        run_start=RUN_START,
        run_end="2026-01-01T00:00:02+00:00",
        coverage_definition=ONE_INTERVAL_BOUND,
        sampler_interval_seconds=2.0,
    )
    rejected = parse_power_trace(
        trace,
        run_start=RUN_START,
        run_end="2026-01-01T00:00:02.000001+00:00",
        coverage_definition=ONE_INTERVAL_BOUND,
        sampler_interval_seconds=2.0,
    )

    assert accepted["coverage_definition"] == ONE_INTERVAL_BOUND
    assert accepted["coverage"]["last_within_interval_bound"] is True
    assert accepted["coverage_complete"] is True
    assert accepted["covers_run_end"] is False
    assert rejected["coverage_complete"] is False
    assert rejected["coverage"]["last_within_interval_bound"] is False


def test_resident_record_keeps_snapshot_when_live_csv_grows_during_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_id = "run-111"
    live = tmp_path / f"power-{run_id}.csv"
    _write_trace(live, RUN_START)
    snapshot = atomic_power_trace_snapshot(
        live,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="resident-power-snapshot",
    )
    snapshot_bytes = snapshot.read_bytes()
    snapshot_hash = hashlib.sha256(snapshot_bytes).hexdigest()
    trace = read_power_trace_snapshot(
        snapshot,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        snapshot_prefix="resident-power-snapshot",
        source_path=live,
        sampler_interval_seconds=2.0,
        coverage_definition=ONE_INTERVAL_BOUND,
    )

    # Make the sampler win a write while the builder reparses the selected
    # snapshot. It must not alter the bytes already selected for judging.
    original_parse_power_trace = telemetry.parse_power_trace
    appended_during_construction = False

    def parse_power_trace_during_construction(*args, **kwargs):
        nonlocal appended_during_construction
        if not appended_during_construction:
            _write_trace(live, RUN_START, "2026-01-01T00:00:03+00:00")
            appended_during_construction = True
        return original_parse_power_trace(*args, **kwargs)

    monkeypatch.setattr(
        telemetry, "parse_power_trace", parse_power_trace_during_construction
    )
    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="0123456789abcdef",
        environment=_environment(),
        power_trace=snapshot.name,
        power_trace_path=snapshot,
        power_trace_metadata=trace,
        run_start=RUN_START,
        run_end=RUN_END,
        timing_evidence=False,
    )
    record_after_construction = dict(record)
    assert appended_during_construction is True
    _write_trace(
        live,
        RUN_START,
        "2026-01-01T00:00:03+00:00",
        "2026-01-01T00:00:04+00:00",
    )

    assert record["power_trace"] == snapshot.name
    assert record["power_trace_sha256"] == snapshot_hash
    assert record["power_trace_sample_count"] == 1
    assert record["power_trace_csv_row_count"] == 1
    assert record["power_trace_valid_row_count"] == 1
    assert record["power_trace_coverage"]["coverage_definition"] == ONE_INTERVAL_BOUND
    assert record["power_clock_provenance"]["coverage_complete"] is True
    assert record["power_clock_provenance"]["coverage_definition"] == ONE_INTERVAL_BOUND
    assert record["clock"]["aiclk_mhz"] == 1_350
    assert record["clock"]["aiclk_source"] == "run_trace_samples"
    assert record["power_clock_provenance"]["immutable_snapshot"] is True
    assert record["power_clock_provenance"]["source_file"] == live.name
    report = validate_resident_record(
        record,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=snapshot,
    )
    assert report["valid"] is True
    assert "power_trace_coverage.coverage_definition" not in report["warning_fields"]
    assert "power_clock_provenance.coverage_definition" not in report["warning_fields"]
    invalid_definition = copy.deepcopy(record)
    invalid_definition["power_trace_coverage"]["coverage_definition"] = "invalid"
    invalid_definition["power_clock_provenance"]["coverage_definition"] = "invalid"
    assert validate_resident_record(
        invalid_definition,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=snapshot,
    )["valid"] is False
    assert snapshot.read_bytes() == snapshot_bytes
    assert record_after_construction == record


def test_schema_marker_one_without_coverage_definition_uses_explicit_final_sample(
    tmp_path: Path,
):
    run_id = "run-111-legacy"
    live = tmp_path / f"power-{run_id}.csv"
    _write_trace(live, RUN_START)
    snapshot = atomic_power_trace_snapshot(
        live,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="legacy-snapshot",
    )
    trace = read_power_trace_snapshot(
        snapshot,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        snapshot_prefix="legacy-snapshot",
        source_path=live,
        sampler_interval_seconds=2.0,
        coverage_definition=ONE_INTERVAL_BOUND,
    )
    assert trace["coverage_complete"] is True
    assert trace["aiclk_source"] == "run_trace_samples"
    assert trace["aiclk_mhz"] == 1_350
    assert trace["in_run_valid_row_count"] == 1
    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="0123456789abcdef",
        environment=_environment(),
        power_trace=snapshot.name,
        power_trace_path=snapshot,
        power_trace_metadata=trace,
        run_start=RUN_START,
        run_end=RUN_END,
    )
    assert validate_resident_record(
        record,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=snapshot,
    )["valid"] is True

    legacy_record = copy.deepcopy(record)
    legacy_record["power_trace_coverage"].pop("coverage_definition")
    legacy_record["power_clock_provenance"].pop("coverage_definition")
    assert validate_resident_record(
        legacy_record,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=snapshot,
    )["valid"] is False


@pytest.mark.parametrize("definition", (EXPLICIT_FINAL_SAMPLE, ONE_INTERVAL_BOUND))
def test_snapshot_reader_declares_named_coverage_definition(tmp_path: Path, definition: str):
    run_id = "run-111-named"
    live = tmp_path / f"power-{run_id}.csv"
    _write_trace(live, RUN_START, "2026-01-01T00:00:02+00:00")
    snapshot = atomic_power_trace_snapshot(
        live,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="named-snapshot",
    )
    trace = read_power_trace_snapshot(
        snapshot,
        run_id=run_id,
        run_start=RUN_START,
        run_end=RUN_END,
        snapshot_prefix="named-snapshot",
        source_path=live,
        sampler_interval_seconds=2.0,
        coverage_definition=definition,
    )

    assert trace["coverage_definition"] == definition
    assert trace["coverage"]["definition"] == definition
    assert trace["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
