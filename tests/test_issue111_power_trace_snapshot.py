"""Board-free coverage for Issue #111 power-trace snapshots."""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench import run_resident, telemetry
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


def test_snapshot_reader_counts_all_rows_only_for_strict_resident_mode(
    tmp_path: Path,
):
    run_id = "run-111-row-count"
    live = tmp_path / f"power-{run_id}.csv"
    live.write_text(
        CSV_HEADER
        + f"{RUN_START},75,1350,60\n"
        + "2026-01-01T00:00:00.500000+00:00,NaN,1350,60\n"
        + f"{RUN_END},75,1350,60\n"
    )
    snapshot = atomic_power_trace_snapshot(
        live,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="row-count-snapshot",
    )
    reader_args = {
        "run_id": run_id,
        "run_start": RUN_START,
        "run_end": RUN_END,
        "snapshot_prefix": "row-count-snapshot",
        "coverage_definition": EXPLICIT_FINAL_SAMPLE,
    }

    strict = read_power_trace_snapshot(
        snapshot, require_all_rows_valid=True, **reader_args
    )
    compatible = read_power_trace_snapshot(snapshot, **reader_args)

    assert strict["sample_count"] == 3
    assert strict["csv_row_count"] == 3
    assert strict["valid_row_count"] == 2
    assert strict["invalid_row_count"] == 1
    assert strict["coverage_complete"] is False
    assert compatible["sample_count"] == 2
    assert compatible["csv_row_count"] == 3
    assert compatible["valid_row_count"] == 2
    assert compatible["invalid_row_count"] == 1
    assert compatible["coverage_complete"] is True


def test_competing_snapshot_writers_keep_the_first_installed_bytes(tmp_path: Path):
    run_id = "run-111-race"
    live = tmp_path / f"power-{run_id}.csv"
    live.write_text("source bytes\n")
    payloads = ("first writer bytes\n", "second writer bytes\n")
    ready = threading.Barrier(2)

    def install(payload: str):
        def read_payload(_path: Path) -> str:
            ready.wait(timeout=5)
            return payload

        return atomic_power_trace_snapshot(
            live,
            tmp_path,
            run_id=run_id,
            snapshot_number=1,
            snapshot_prefix="race-snapshot",
            read_text_fn=read_payload,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(install, payload) for payload in payloads]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=5))
            except Exception as exc:  # noqa: BLE001 - assert the race outcome
                outcomes.append(exc)

    successes = [outcome for outcome in outcomes if isinstance(outcome, Path)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    winner_payload = payloads[0] if isinstance(outcomes[0], Path) else payloads[1]
    assert successes[0].read_text() == winner_payload


def test_one_interval_bound_rejects_late_samples_in_shared_reader_and_telemetry(
    tmp_path: Path,
):
    run_id = "run-111-bound"
    live = tmp_path / f"power-{run_id}.csv"
    late_timestamp = "2026-01-01T00:00:03+00:00"
    end = "2026-01-01T00:00:02+00:00"
    _write_trace(live, RUN_START, late_timestamp)
    snapshot = atomic_power_trace_snapshot(
        live,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="bound-snapshot",
    )

    telemetry_trace = parse_power_trace(
        live,
        run_start=RUN_START,
        run_end=end,
        coverage_definition=ONE_INTERVAL_BOUND,
        sampler_interval_seconds=2.0,
    )
    shared_trace = read_power_trace_snapshot(
        snapshot,
        run_id=run_id,
        run_start=RUN_START,
        run_end=end,
        snapshot_prefix="bound-snapshot",
        source_path=live,
        sampler_interval_seconds=2.0,
        coverage_definition=ONE_INTERVAL_BOUND,
    )

    for trace in (telemetry_trace, shared_trace):
        assert trace["last_timestamp"] == late_timestamp
        assert trace["covers_run_end"] is True
        assert trace["coverage"]["last_within_interval_bound"] is False
        assert trace["coverage_complete"] is False

    explicit = parse_power_trace(
        live,
        run_start=RUN_START,
        run_end=end,
        coverage_definition=EXPLICIT_FINAL_SAMPLE,
    )
    assert explicit["coverage_complete"] is True

    boundary_timestamp = "2026-01-01T00:00:00+00:00"
    _write_trace(live, boundary_timestamp)
    boundary_snapshot = atomic_power_trace_snapshot(
        live,
        tmp_path,
        run_id=run_id,
        snapshot_number=2,
        snapshot_prefix="bound-snapshot",
    )
    boundary_telemetry = parse_power_trace(
        live,
        run_start=RUN_START,
        run_end=end,
        coverage_definition=ONE_INTERVAL_BOUND,
        sampler_interval_seconds=2.0,
    )
    boundary_shared = read_power_trace_snapshot(
        boundary_snapshot,
        run_id=run_id,
        run_start=RUN_START,
        run_end=end,
        snapshot_prefix="bound-snapshot",
        source_path=live,
        sampler_interval_seconds=2.0,
        coverage_definition=ONE_INTERVAL_BOUND,
    )
    assert boundary_telemetry["coverage_complete"] is True
    assert boundary_shared["coverage_complete"] is True
    assert boundary_telemetry["coverage"]["last_within_interval_bound"] is True
    assert boundary_shared["coverage"]["last_within_interval_bound"] is True


def _fake_resident_result(config: ResidentConfig) -> dict[str, object]:
    count = config.frame_count
    return {
        "timestamps": [1_000 + config.frame_interval_ticks * index for index in range(count)],
        "producer_full_count": 0,
        "consumer_empty_count": 0,
        "kernel_error_flag": 0,
        "frames_attempted": count,
        "frames_produced": count,
        "frames_dropped": 0,
        "frames_aborted": 0,
        "frames_consumed": count,
        "startup_ticks": 0,
        "startup_ticks_valid": False,
        "work_min_ticks": None,
        "work_max_ticks": None,
        "work_ticks_valid": False,
        "failure_check": {"code": 0, "name": "none", "source": "none"},
    }


def _patch_fake_resident_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_resident,
        "_run_device",
        lambda _ttnn, _device, config, *, watcher: _fake_resident_result(config),
    )
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(
            open_device=lambda **_: object(), close_device=lambda _device: None
        ),
    )


def test_resident_binding_failure_rejects_wrong_run_without_reading_live_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "runner-output"
    output_dir.mkdir()
    environment_path = output_dir / "env-current.json"
    environment_path.write_text(json.dumps(_environment()))
    wrong_run_trace = output_dir / "power-old-run.csv"
    _write_trace(wrong_run_trace, RUN_START, RUN_END)
    output = tmp_path / "resident-rejected.json"
    monkeypatch.setattr(run_resident, "_runner_output_dir", lambda: output_dir)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "current-run")
    _patch_fake_resident_device(monkeypatch)

    def fail_if_trace_is_parsed(*_args, **_kwargs):
        raise AssertionError("binding failure must not parse the live trace")

    monkeypatch.setattr(telemetry, "parse_power_trace", fail_if_trace_is_parsed)
    assert run_resident.main(
        [
            "--out",
            str(output),
            "--env-json",
            str(environment_path),
            "--power-trace",
            wrong_run_trace.name,
            "--frame-count",
            "3",
        ]
    ) == 2
    rejection = json.loads(output.read_text())
    assert rejection["status"] == "rejected"
    assert rejection["timing_evidence"] is False
    assert rejection["clock"]["aiclk_mhz"] is None
    assert rejection["power_trace"] is None
    assert "binding rejected" in rejection["rejection_reason"]
    assert wrong_run_trace.read_text().endswith(f"{RUN_END},75,1350,60\n")


def test_resident_main_records_run_bound_snapshot_and_ignores_later_live_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "runner-output"
    output_dir.mkdir()
    environment_path = output_dir / "env-current.json"
    environment_path.write_text(json.dumps(_environment()))
    live = output_dir / "power-current-run.csv"
    _write_trace(live, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00")
    output = tmp_path / "resident-result.json"
    monkeypatch.setattr(run_resident, "_runner_output_dir", lambda: output_dir)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "current-run")
    _patch_fake_resident_device(monkeypatch)

    real_datetime = datetime.datetime

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = next(run_times)
            return value if tz is None else value.astimezone(tz)

    run_times = iter(
        (
            FixedDatetime(2026, 1, 1, tzinfo=datetime.UTC),
            FixedDatetime(2026, 1, 1, 0, 0, 1, tzinfo=datetime.UTC),
        )
    )
    monkeypatch.setattr(run_resident.datetime, "datetime", FixedDatetime)
    assert run_resident.main(
        [
            "--out",
            str(output),
            "--env-json",
            str(environment_path),
            "--power-trace",
            live.name,
            "--frame-count",
            "3",
        ]
    ) == 0

    record = json.loads(output.read_text())
    snapshot = output_dir / record["power_trace"]
    snapshot_bytes = snapshot.read_bytes()
    record_before_growth = copy.deepcopy(record)
    assert snapshot.name == "resident-power-snapshot-current-run-1.csv"
    assert record["power_trace_sha256"] == hashlib.sha256(snapshot_bytes).hexdigest()
    assert record["power_clock_provenance"]["immutable_snapshot"] is True
    assert record["power_clock_provenance"]["source_file"] == live.name
    assert record["power_trace_sample_count"] == 2
    assert record["power_trace_csv_row_count"] == 2

    _write_trace(
        live,
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T00:00:01+00:00",
        "2026-01-01T00:00:02+00:00",
    )
    assert snapshot.read_bytes() == snapshot_bytes
    assert json.loads(output.read_text()) == record_before_growth


def test_resident_main_rejects_invalid_snapshot_rows_as_timing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output_dir = tmp_path / "runner-output"
    output_dir.mkdir()
    environment_path = output_dir / "env-current.json"
    environment_path.write_text(json.dumps(_environment()))
    live = output_dir / "power-current-run.csv"
    live.write_text(
        CSV_HEADER
        + "2026-01-01T00:00:00+00:00,75,1350,60\n"
        "2026-01-01T00:00:00.500000+00:00,NaN,1350,60\n"
        "2026-01-01T00:00:01+00:00,75,1350,60\n"
    )
    output = tmp_path / "resident-invalid.json"
    monkeypatch.setattr(run_resident, "_runner_output_dir", lambda: output_dir)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "current-run")
    _patch_fake_resident_device(monkeypatch)

    real_datetime = datetime.datetime

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = next(run_times)
            return value if tz is None else value.astimezone(tz)

    run_times = iter(
        (
            FixedDatetime(2026, 1, 1, tzinfo=datetime.UTC),
            FixedDatetime(2026, 1, 1, 0, 0, 1, tzinfo=datetime.UTC),
        )
    )
    monkeypatch.setattr(run_resident.datetime, "datetime", FixedDatetime)

    assert run_resident.main(
        [
            "--out",
            str(output),
            "--env-json",
            str(environment_path),
            "--power-trace",
            live.name,
            "--frame-count",
            "3",
        ]
    ) == 0

    record = json.loads(output.read_text())
    snapshot = output_dir / record["power_trace"]
    telemetry_trace = parse_power_trace(
        snapshot,
        run_start="2026-01-01T00:00:00+00:00",
        run_end="2026-01-01T00:00:01+00:00",
        coverage_definition=ONE_INTERVAL_BOUND,
        sampler_interval_seconds=2.0,
    )
    provenance = record["power_clock_provenance"]
    assert telemetry_trace["coverage_complete"] is False
    assert record["power_trace_coverage"]["coverage_complete"] is False
    assert provenance["coverage_complete"] is False
    assert record["timing_evidence"] is False
    assert provenance["timing_evidence"] is False
    assert record["timing_evidence_reason"] == "power_trace_no_valid_in_run_rows"
    assert provenance["timing_evidence_reason"] == "power_trace_no_valid_in_run_rows"
    assert record["clock"]["aiclk_source"] == "configured"
    for field in (
        "sample_count",
        "csv_row_count",
        "valid_row_count",
        "in_run_valid_row_count",
        "timestamps_parse",
        "timestamps_ordered",
        "first_timestamp",
        "last_timestamp",
        "covers_run_start",
        "covers_run_end",
        "coverage_complete",
        "aiclk_source",
        "aiclk_mhz",
    ):
        assert provenance[field] == telemetry_trace[field]

    report = validate_resident_record(
        record,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=snapshot,
    )
    assert report["valid"] is False
    assert report["checks"]["power_trace.metadata_duplicates"]["ok"] is True
    assert report["checks"]["clock.timing_evidence"]["ok"] is True
