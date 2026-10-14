"""Board-free acceptance coverage for Issue #110 validator hardening."""

from __future__ import annotations

import copy
import datetime
import itertools
import json
from pathlib import Path

import pytest

from enodia.tt.bench.load_harness import (
    CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    CalibrationMetadata,
    LoadConfigurationError,
)
from enodia.tt.bench.power_trace_snapshot import (
    PowerTraceSnapshotOptions,
    atomic_power_trace_snapshot,
    read_power_trace_snapshot,
)
from enodia.tt.bench.resident_harness import (
    ResidentConfig,
    build_measurement_record,
    build_rejection_record,
)
from enodia.tt.bench.resident_record import (
    INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD,
    PROGRAM_INTERVAL_UTC_MONOTONIC_TOLERANCE_NS,
    validate_resident_record,
)
from enodia.tt.bench.run_resident import _read_load_calibration
from enodia.tt.bench.sampler_contract import SAMPLER_MODES

CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
OUTCOMES = ("ok", "ok_with_drops", "error", "rejected")
PRODUCT = tuple(itertools.product(SAMPLER_MODES, OUTCOMES))


def _config(*, frame_count: int) -> ResidentConfig:
    return ResidentConfig(
        frame_count=frame_count,
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


def _environment(mode: str) -> dict[str, object]:
    interval = None if mode == "off" else (2.0 if mode == "default" else 5.0)
    return {
        "board": {"serial": "issue110-fixture", "board_type": "synthetic"},
        "firmware": {"fw_bundle_version": "fixture"},
        "kmd_version": "fixture",
        "image": "registry.example/tt@sha256:" + "a" * 64,
        "image_pinned": True,
        "harness_commit": "0123456789abcdef",
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {
            "mode": mode,
            "interval_seconds": interval,
            "power_trace": "absent_by_design" if mode == "off" else "required",
            "timing_evidence": "diagnostic_only" if mode == "off" else "available",
        },
    }


def _trace(path: Path) -> None:
    path.write_text(
        CSV_HEADER
        + "2026-01-01T00:00:00+00:00,75,1350,60\n"
        + "2026-01-01T00:00:01+00:00,75,1350,60\n"
        + "2026-01-01T00:00:02+00:00,75,1350,60\n"
    )


def _success_record(
    *,
    mode: str,
    outcome: str,
    trace_path: Path | None,
    trace_metadata: dict[str, object] | None = None,
    interval_extrema_policy: str | None = None,
) -> dict[str, object]:
    frame_count = 4 if outcome == "ok_with_drops" else 3
    timestamps = [1_000, 2_000, 3_000]
    dropped = 1 if outcome == "ok_with_drops" else 0
    record = build_measurement_record(
        config=_config(frame_count=frame_count),
        aiclk_mhz=1_350,
        timestamps=timestamps,
        producer_full_count=dropped,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="0123456789abcdef",
        environment=_environment(mode),
        power_trace=trace_path.name if trace_path is not None else None,
        power_trace_path=trace_path,
        power_trace_metadata=trace_metadata,
        interval_extrema_policy=interval_extrema_policy,
        run_start="2026-01-01T00:00:00+00:00" if trace_path is not None else None,
        run_end="2026-01-01T00:00:02+00:00" if trace_path is not None else None,
        attempted_frame_count=frame_count,
        produced_frame_count=len(timestamps),
        dropped_frame_count=dropped,
        timing_evidence=False,
    )
    assert record["status"] == outcome
    return record


def _error_record(
    *, mode: str, trace_path: Path | None
) -> dict[str, object]:
    record = _success_record(mode=mode, outcome="ok", trace_path=trace_path)
    record["status"] = "error"
    record["termination_reason"] = "run_budget"
    record["failure_check"] = {
        "code": 1,
        "name": "run_wide_budget",
        "source": "producer",
        "elapsed_ticks": 2,
        "limit_ticks": 1,
        "unit": "device_clock_ticks",
        "valid": True,
    }
    record["cycle_budget"]["error_flag"] = 1
    record["timing_evidence_reason"] = "run_error_not_timing_evidence"
    record["power_clock_provenance"]["timing_evidence"] = False
    record["power_clock_provenance"]["timing_evidence_reason"] = (
        "run_error_not_timing_evidence"
    )
    return record


@pytest.mark.parametrize("mode,outcome", PRODUCT, ids=lambda value: str(value))
def test_record_kind_matrix_covers_sampler_outcome_product(
    tmp_path: Path, mode: str, outcome: str
):
    """Every runner sampler mode and terminal outcome remains a valid record."""
    trace_path = None
    if mode != "off":
        trace_path = tmp_path / f"power-{mode}.csv"
        _trace(trace_path)
    if outcome in {"ok", "ok_with_drops"}:
        record = _success_record(mode=mode, outcome=outcome, trace_path=trace_path)
    elif outcome == "error":
        record = _error_record(mode=mode, trace_path=trace_path)
    else:
        record = build_rejection_record(
            config=_config(frame_count=3),
            reason="board-free Issue #110 rejection",
            environment={"telemetry_sampler": _environment(mode)["telemetry_sampler"]},
        )
    report = validate_resident_record(
        copy.deepcopy(record),
        timestamps=None if outcome == "rejected" else [1_000, 2_000, 3_000],
        power_trace_path=trace_path,
        strict_trace=trace_path is not None,
    )
    assert report["valid"] is True, report["mismatches"]


TRACE_ROOT = Path(__file__).parents[1] / "docs/measurements"
TRACE_RECORD = TRACE_ROOT / "2026-10-11-p150a-issue118-newton-schulz-500000-adr0005.json"
TRACE_CSV = TRACE_ROOT / "2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv"


@pytest.fixture
def trace_record_with_nested_coverage_aliases() -> dict[str, object]:
    """Add every resident-record coverage alias from recomputed CSV facts."""
    record = json.loads(TRACE_RECORD.read_text())
    recomputed = validate_resident_record(
        record,
        power_trace_path=TRACE_CSV,
        strict_trace=True,
    )
    assert recomputed["valid"] is True, recomputed["mismatches"]
    expected = {
        name: recomputed["power_trace"][name]
        for name in ("covers_run_start", "covers_run_end", "coverage_complete")
    }
    for container_name in ("power_clock_provenance", "power_trace_coverage"):
        record[container_name]["coverage"].update(expected)
    record["power_trace_coverage"]["complete"] = expected["coverage_complete"]
    return record


@pytest.fixture
def policy_trace_record(tmp_path: Path) -> dict[str, object]:
    """Build a board-free resident snapshot with interval-extrema provenance."""
    run_id = "issue110-snapshot"
    source_path = tmp_path / f"power-{run_id}.csv"
    _trace(source_path)
    trace_path = atomic_power_trace_snapshot(
        source_path,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="resident-power-snapshot",
    )
    trace_metadata = read_power_trace_snapshot(
        trace_path,
        options=PowerTraceSnapshotOptions(
            run_id=run_id,
            run_start="2026-01-01T00:00:00+00:00",
            run_end="2026-01-01T00:00:02+00:00",
            snapshot_prefix="resident-power-snapshot",
            source_path=source_path,
            require_all_rows_valid=True,
        ),
    )
    return _success_record(
        mode="default",
        outcome="ok",
        trace_path=trace_path,
        trace_metadata=trace_metadata,
        interval_extrema_policy=INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD,
    )


def test_interval_policy_on_ordinary_csv_is_not_snapshot_evidence(tmp_path: Path):
    """A policy alone cannot promote an ordinary CSV to an immutable snapshot."""
    trace_path = tmp_path / "power-policy.csv"
    _trace(trace_path)
    record = _success_record(
        mode="default",
        outcome="ok",
        trace_path=trace_path,
    )
    record["power_trace_interval_extrema_policy"] = (
        INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD
    )
    provenance = record["power_clock_provenance"]
    provenance["interval_extrema_policy"] = (
        INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD
    )
    provenance["source_file"] = trace_path.name
    provenance["immutable_snapshot"] = True

    report = validate_resident_record(
        record,
        power_trace_path=trace_path,
        strict_trace=True,
    )

    assert report["power_trace"]["immutable_snapshot"] is False
    assert report["valid"] is False
    assert any(
        mismatch["invariant"] == "power_trace.facts"
        for mismatch in report["mismatches"]
    )


TRACE_FLAG_PATHS = (
    ("power_clock_provenance", "readable"),
    ("power_clock_provenance", "nonempty"),
    ("power_clock_provenance", "immutable_snapshot"),
    ("power_clock_provenance", "timestamps_parse"),
    ("power_clock_provenance", "timestamps_ordered"),
    ("power_clock_provenance", "covers_run_start"),
    ("power_clock_provenance", "covers_run_end"),
    ("power_clock_provenance", "coverage_complete"),
    ("power_clock_provenance", "coverage", "readable"),
    ("power_clock_provenance", "coverage", "nonempty"),
    ("power_clock_provenance", "coverage", "timestamps_parse"),
    ("power_clock_provenance", "coverage", "timestamps_ordered"),
    ("power_clock_provenance", "coverage", "first_at_or_before_run_start"),
    ("power_clock_provenance", "coverage", "last_at_or_after_run_end"),
    ("power_clock_provenance", "coverage", "last_within_interval_bound"),
    ("power_clock_provenance", "coverage", "complete"),
    ("power_clock_provenance", "coverage", "covers_run_start"),
    ("power_clock_provenance", "coverage", "covers_run_end"),
    ("power_clock_provenance", "coverage", "coverage_complete"),
    ("power_trace_coverage", "coverage_complete"),
    ("power_trace_coverage", "complete"),
    ("power_trace_coverage", "coverage", "readable"),
    ("power_trace_coverage", "coverage", "nonempty"),
    ("power_trace_coverage", "coverage", "timestamps_parse"),
    ("power_trace_coverage", "coverage", "timestamps_ordered"),
    ("power_trace_coverage", "coverage", "first_at_or_before_run_start"),
    ("power_trace_coverage", "coverage", "last_at_or_after_run_end"),
    ("power_trace_coverage", "coverage", "last_within_interval_bound"),
    ("power_trace_coverage", "coverage", "complete"),
    ("power_trace_coverage", "coverage", "covers_run_start"),
    ("power_trace_coverage", "coverage", "covers_run_end"),
    ("power_trace_coverage", "coverage", "coverage_complete"),
)


@pytest.mark.parametrize("path", TRACE_FLAG_PATHS, ids=lambda value: ".".join(value))
def test_every_declared_trace_coverage_flag_is_bound_to_recomputed_csv_facts(
    trace_record_with_nested_coverage_aliases: dict[str, object],
    policy_trace_record: dict[str, object],
    path: tuple[str, ...],
    tmp_path: Path,
):
    """Flipping any persisted coverage/provenance boolean invalidates the record."""
    source = (
        policy_trace_record
        if path == ("power_clock_provenance", "immutable_snapshot")
        else trace_record_with_nested_coverage_aliases
    )
    candidate = copy.deepcopy(source)
    current = candidate
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = not current[path[-1]]
    trace_path = (
        tmp_path / str(candidate["power_trace"])
        if path == ("power_clock_provenance", "immutable_snapshot")
        else TRACE_CSV
    )
    report = validate_resident_record(
        candidate,
        power_trace_path=trace_path,
        strict_trace=True,
    )
    assert report["valid"] is False, path
    assert any(
        mismatch["invariant"] == "power_trace.coverage_flags"
        for mismatch in report["mismatches"]
    )


@pytest.mark.parametrize(
    "path",
    (
        ("power_clock_provenance", "readable"),
        ("power_clock_provenance", "coverage", "readable"),
    ),
    ids=lambda value: ".".join(value),
)
@pytest.mark.parametrize("numeric_value", (0, 1))
def test_numeric_trace_coverage_flags_are_not_accepted_as_booleans(
    path: tuple[str, ...], numeric_value: int
):
    """Numeric JSON flags must not compare equal to recomputed booleans."""
    record = json.loads(TRACE_RECORD.read_text())
    candidate = copy.deepcopy(record)
    current = candidate
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = numeric_value
    report = validate_resident_record(
        candidate,
        power_trace_path=TRACE_CSV,
        strict_trace=True,
    )
    assert report["valid"] is False, (path, numeric_value, report["mismatches"])
    assert any(
        mismatch["invariant"] == "power_trace.coverage_flags"
        and mismatch["fields"] == [".".join(path)]
        for mismatch in report["mismatches"]
    )


def _utc_elapsed_ns(record: dict[str, object]) -> int:
    provenance = record["power_clock_provenance"]
    start = datetime.datetime.fromisoformat(provenance["load_interval_start"])
    end = datetime.datetime.fromisoformat(provenance["load_interval_end"])
    delta = end - start
    return (
        delta.days * 86_400 * 1_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


def _set_program_elapsed(record: dict[str, object], elapsed_ns: int) -> None:
    interval = record["program_host_interval"]
    start = interval["start_ns"]
    end = start + elapsed_ns
    for name in ("program_host_interval", "host_program_interval"):
        record[name].update(
            end_ns=end,
            completion_ns=end,
            elapsed_ns=elapsed_ns,
        )
    record["program_completion_ns"] = end
    record["program_completion_monotonic_ns"] = end


def _loaded_calibration_payload(condition: str) -> dict[str, object]:
    calibration = CalibrationMetadata(
        condition=condition,
        duration_seconds=5,
        achieved_throughput=2,
        throughput_unit="repeats_per_card_second",
        repeat_count=1,
        target_seconds=500,
        load_card_ticks=321,
        load_measurement_repeat_count=1,
        load_interval_aiclk_mhz=1_350,
        load_interval_aiclk_source=CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    )
    return {"calibration": calibration.as_record()}


def _read_loaded_calibration(
    tmp_path: Path, payload: dict[str, object]
) -> CalibrationMetadata:
    path = tmp_path / "resident-calibration.json"
    path.write_text(json.dumps(payload))
    return _read_load_calibration(path)


@pytest.mark.parametrize("condition", ("newton_schulz", "dram"))
def test_loaded_calibration_preserves_tick_preference_and_fallbacks(
    tmp_path: Path, condition: str
):
    payload = _loaded_calibration_payload(condition)
    payload["load"] = {
        "card_load_interval": {"start_ticks": 100, "end_ticks": 225},
        "throughput_interval": {"elapsed_ticks": 0},
    }
    aggregate_fallback = _read_loaded_calibration(tmp_path, payload)
    assert aggregate_fallback.load_card_ticks == 125

    payload["load"]["throughput_interval"]["elapsed_ticks"] = 333
    preferred = _read_loaded_calibration(tmp_path, payload)
    assert preferred.load_card_ticks == 333

    payload["load"]["throughput_interval"] = {}
    absent_with_fallback = _read_loaded_calibration(tmp_path, payload)
    assert absent_with_fallback.load_card_ticks == 125

    payload["load"]["card_load_interval"] = {}
    absent_without_fallback = _read_loaded_calibration(tmp_path, payload)
    assert absent_without_fallback.load_card_ticks == 321


@pytest.mark.parametrize("condition", ("newton_schulz", "dram"))
@pytest.mark.parametrize(
    "elapsed_ticks",
    (-1, 0, False, 1.5, "invalid", [1], {"value": 1}),
    ids=("negative", "zero", "false", "float", "string", "list", "object"),
)
def test_loaded_calibration_rejects_invalid_ticks_without_aggregate_fallback(
    tmp_path: Path, condition: str, elapsed_ticks: object
):
    payload = _loaded_calibration_payload(condition)
    payload["load"] = {"throughput_interval": {"elapsed_ticks": elapsed_ticks}}

    with pytest.raises(
        LoadConfigurationError, match="calibration load-time metadata is invalid"
    ):
        _read_loaded_calibration(tmp_path, payload)


@pytest.mark.parametrize("condition", ("newton-schulz", "dram"))
def test_committed_loaded_program_intervals_pass_with_substantial_margin(condition):
    record_path = TRACE_ROOT / f"2026-10-11-p150a-issue118-{condition}-500000-adr0005.json"
    trace_path = TRACE_ROOT / f"2026-10-11-p150a-issue118-{condition}-500000-adr0005-power.csv"
    record = json.loads(record_path.read_text())
    report = validate_resident_record(record, power_trace_path=trace_path, strict_trace=True)
    assert report["valid"] is True, report["mismatches"]
    difference = abs(
        _utc_elapsed_ns(record) - record["program_host_interval"]["elapsed_ns"]
    )
    assert difference * 100 < PROGRAM_INTERVAL_UTC_MONOTONIC_TOLERANCE_NS


@pytest.mark.parametrize(
    "interval_name", ("program_host_interval", "host_program_interval")
)
def test_loaded_program_interval_alias_mutation_is_rejected(interval_name: str):
    """A mutation of only one persisted interval alias cannot pass validation."""
    record = json.loads(
        (TRACE_ROOT / "2026-10-11-p150a-issue118-newton-schulz-500000-adr0005.json").read_text()
    )
    candidate = copy.deepcopy(record)
    candidate[interval_name]["elapsed_ns"] += 50_000_000
    report = validate_resident_record(
        candidate,
        power_trace_path=TRACE_ROOT
        / "2026-10-11-p150a-issue118-newton-schulz-500000-adr0005-power.csv",
        strict_trace=True,
    )
    assert report["valid"] is False, report["mismatches"]
    assert any(
        mismatch["invariant"] == "program_interval.utc_monotonic_binding"
        and "host_program_interval.elapsed_ns" in mismatch["fields"]
        for mismatch in report["mismatches"]
    )


@pytest.mark.parametrize("condition", ("newton-schulz", "dram"))
@pytest.mark.parametrize("extra_ns", (0, 1))
def test_loaded_program_interval_boundary_is_inclusive_and_outside_fails(
    condition: str, extra_ns: int
):
    record_path = TRACE_ROOT / f"2026-10-11-p150a-issue118-{condition}-500000-adr0005.json"
    trace_path = TRACE_ROOT / f"2026-10-11-p150a-issue118-{condition}-500000-adr0005-power.csv"
    record = json.loads(record_path.read_text())
    candidate = copy.deepcopy(record)
    _set_program_elapsed(
        candidate,
        _utc_elapsed_ns(candidate)
        + PROGRAM_INTERVAL_UTC_MONOTONIC_TOLERANCE_NS
        + extra_ns,
    )
    report = validate_resident_record(
        candidate,
        power_trace_path=trace_path,
        strict_trace=True,
    )
    expected_valid = extra_ns == 0
    assert report["valid"] is expected_valid, report["mismatches"]
    if not expected_valid:
        assert any(
            mismatch["invariant"] == "program_interval.utc_monotonic_binding"
            for mismatch in report["mismatches"]
        )


def test_unloaded_run_bounds_interval_is_not_compared_to_monotonic_interval():
    record = json.loads(
        (TRACE_ROOT / "2026-10-11-p150a-issue118-control-500000-adr0005.json").read_text()
    )
    candidate = copy.deepcopy(record)
    _set_program_elapsed(candidate, candidate["program_host_interval"]["elapsed_ns"] * 2)
    report = validate_resident_record(
        candidate,
        power_trace_path=TRACE_ROOT / "2026-10-11-p150a-issue118-control-500000-adr0005-power.csv",
        strict_trace=True,
    )
    assert report["valid"] is True, report["mismatches"]
