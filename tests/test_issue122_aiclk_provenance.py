"""Board-free Issue #122 conversion provenance regressions."""

from __future__ import annotations

import hashlib
import struct
from copy import deepcopy
from pathlib import Path

import pytest

from enodia.tt.bench.load_harness import (
    CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    CalibrationMetadata,
    predicted_load_time_seconds,
    predicted_load_time_selection,
)
from enodia.tt.bench.power_trace_snapshot import (
    ISSUE118_LOAD_INTERVAL_SOURCE,
    PowerTraceCsvOptions,
    PowerTraceSnapshotOptions,
    atomic_power_trace_snapshot,
    read_power_trace_csv,
    read_power_trace_snapshot,
)
from enodia.tt.bench.resident_clock import (
    AICLK_SELECTION_IN_RUN_MINIMUM,
    AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL,
    AICLK_SOURCE_CONFIGURED,
    AICLK_SOURCE_LOAD_INTERVAL_TRACE,
    AICLK_SOURCE_RUN_TRACE_SAMPLES,
    aiclk_selection_for_source,
    select_elapsed_aiclk,
    select_load_interval_aiclk,
)
from enodia.tt.bench.resident_harness import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    ResidentConfig,
    build_measurement_record,
)
from enodia.tt.bench.resident_record import (
    INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD,
    build_outlier_analysis,
    validate_resident_record,
)
from enodia.tt.bench.run_resident import (
    _calibration_actual_duration,
    _load_achieved_throughput,
)
from tools.analyze_issue104_wrap import analyze_run
from tools.audit_issue122_aiclk import _conversion_inventory, render_report


def test_varying_trace_keeps_maximum_separate_from_weighted_elapsed_selection():
    trace_selection = select_elapsed_aiclk(
        budget_aiclk_mhz=1_350,
        sampler_mode="default",
        trace_aiclk_mhz=1_350,
        trace_aiclk_source=AICLK_SOURCE_RUN_TRACE_SAMPLES,
    )
    weighted_selection = select_load_interval_aiclk(
        weighted_aiclk_mhz=1_150.0,
        provenance=AICLK_SOURCE_LOAD_INTERVAL_TRACE,
    )

    assert trace_selection["aiclk_mhz"] == 1_350
    assert trace_selection["aiclk_source"] == AICLK_SOURCE_RUN_TRACE_SAMPLES
    assert aiclk_selection_for_source(trace_selection["aiclk_source"]) == "in_run_maximum"
    configured = select_elapsed_aiclk(
        budget_aiclk_mhz=1_200,
        sampler_mode="off",
    )
    assert configured["aiclk_source"] == AICLK_SOURCE_CONFIGURED
    assert aiclk_selection_for_source(configured["aiclk_source"]) == "configured_budget"
    assert weighted_selection["aiclk_mhz_exact"] == pytest.approx(1_150.0)
    assert weighted_selection["aiclk_source"] == AICLK_SOURCE_LOAD_INTERVAL_TRACE

    outlier = build_outlier_analysis(
        [0, 1_000_000, 2_700_000],
        aiclk_mhz=trace_selection["aiclk_mhz"],
        aiclk_source=trace_selection["aiclk_source"],
        elapsed_aiclk_mhz=weighted_selection["aiclk_mhz_exact"],
        elapsed_aiclk_source=weighted_selection["aiclk_source"],
    )
    assert outlier["aiclk_mhz_for_elapsed_seconds"] == pytest.approx(1_150.0)
    assert outlier["aiclk_source_for_elapsed_seconds"] == AICLK_SOURCE_LOAD_INTERVAL_TRACE
    assert outlier["aiclk_selection_for_elapsed_seconds"] == AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL
    assert outlier["run_elapsed_seconds"] == pytest.approx(2_700_000 / (1_150.0 * 1_000_000))


def test_prediction_persists_distinct_load_interval_and_prediction_selections():
    calibration = CalibrationMetadata(
        condition="newton_schulz",
        duration_seconds=60.0,
        achieved_throughput=2.0,
        throughput_unit="repeats_per_card_second",
        repeat_count=1100,
        load_card_ticks=1_000_000,
        load_measurement_repeat_count=2,
        load_interval_aiclk_mhz=1_200.0,
        load_interval_aiclk_min_mhz=800.0,
        load_interval_aiclk_source=CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    )

    selection = predicted_load_time_selection(calibration)
    assert selection == {
        "aiclk_mhz": 800.0,
        "aiclk_source": AICLK_SELECTION_IN_RUN_MINIMUM,
        "aiclk_selection": AICLK_SELECTION_IN_RUN_MINIMUM,
    }
    persisted = calibration.as_record()
    assert persisted["load_interval_aiclk_mhz"] == 1_200.0
    assert persisted["load_interval_aiclk_source"] == AICLK_SOURCE_LOAD_INTERVAL_TRACE
    assert persisted["load_interval_aiclk_selection"] == AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL
    assert persisted["predicted_load_time_aiclk_mhz"] == 800.0
    assert persisted["predicted_load_time_aiclk_source"] == AICLK_SELECTION_IN_RUN_MINIMUM
    assert persisted["predicted_load_time_aiclk_selection"] == AICLK_SELECTION_IN_RUN_MINIMUM
    assert calibration.load_interval_aiclk_selection == AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL
    assert calibration.predicted_load_time_aiclk_mhz == 800.0
    assert calibration.predicted_load_time_aiclk_source == AICLK_SELECTION_IN_RUN_MINIMUM
    assert calibration.predicted_load_time_aiclk_selection == AICLK_SELECTION_IN_RUN_MINIMUM
    assert predicted_load_time_seconds(calibration) == pytest.approx(0.6875)


def test_calibration_validation_distinguishes_interval_and_prediction_metadata():
    common = {
        "condition": "newton_schulz",
        "duration_seconds": 60.0,
        "achieved_throughput": 2.0,
        "throughput_unit": "repeats_per_card_second",
        "repeat_count": 1100,
        "load_card_ticks": 1_000_000,
        "load_measurement_repeat_count": 2,
        "load_interval_aiclk_mhz": 1_200.0,
        "load_interval_aiclk_min_mhz": 800.0,
        "load_interval_aiclk_source": CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE,
    }
    with pytest.raises(ValueError, match="load interval AICLK selection"):
        CalibrationMetadata(
            **common,
            load_interval_aiclk_selection=AICLK_SELECTION_IN_RUN_MINIMUM,
        )
    with pytest.raises(ValueError, match="prediction AICLK selection"):
        CalibrationMetadata(
            **common,
            predicted_load_time_aiclk_selection=AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL,
        )


def test_fixed_reference_throughput_and_actual_duration_keep_distinct_sources():
    from enodia.tt.bench.resident_clock import fixed_reference_aiclk_selection, ticks_to_seconds
    from enodia.tt.bench.resident_record import DEFAULT_AICLK_MHZ
    from enodia.tt.bench.run_resident import CARD_THROUGHPUT_REFERENCE_AICLK_MHZ

    fixed = fixed_reference_aiclk_selection()
    assert fixed["aiclk_mhz"] == CARD_THROUGHPUT_REFERENCE_AICLK_MHZ == DEFAULT_AICLK_MHZ
    assert ticks_to_seconds(ticks=1_350_000, selection=fixed) == pytest.approx(0.001)

    actual = _calibration_actual_duration(
        frame_count=2,
        frame_interval_ticks=1_000_000,
        aiclk_mhz=1_200.0,
        aiclk_source=AICLK_SOURCE_LOAD_INTERVAL_TRACE,
    )
    assert actual["actual_duration_seconds"] == pytest.approx(2 / 1_200)
    assert actual["actual_duration_aiclk_mhz"] == 1_200.0
    assert actual["actual_duration_aiclk_source"] == AICLK_SOURCE_LOAD_INTERVAL_TRACE
    assert actual["actual_duration_aiclk_selection"] == AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL


def test_load_calibration_throughput_records_fixed_reference_source():
    from enodia.tt.bench.load_harness import CardLoadInterval
    from enodia.tt.bench.resident_record import DEFAULT_AICLK_MHZ
    from enodia.tt.bench.run_resident import CARD_THROUGHPUT_REFERENCE_AICLK_MHZ

    result = {
        "card_load_interval": CardLoadInterval(0, 1000),
        "card_load_worker_intervals": [
            {"start_ticks": 100, "end_ticks": 700},
            {"start_ticks": 100, "end_ticks": 800},
        ],
    }
    _load_achieved_throughput(result, condition="dram", repeat_count=2, aiclk_mhz=1_100.0)

    interval = result["throughput_interval"]
    assert interval["reference_aiclk_mhz"] == CARD_THROUGHPUT_REFERENCE_AICLK_MHZ == DEFAULT_AICLK_MHZ
    assert interval["reference_aiclk_source"] == "fixed_reference_1350_mhz"
    assert interval["load_interval_aiclk_mhz"] == 1_100.0
    assert interval["load_interval_aiclk_source"] == AICLK_SOURCE_LOAD_INTERVAL_TRACE


def test_load_calibration_throughput_records_trace_source_for_aggregate_interval():
    from enodia.tt.bench.load_harness import CardLoadInterval

    result = {"card_load_interval": CardLoadInterval(100, 800)}
    _load_achieved_throughput(
        result,
        condition="dram",
        repeat_count=2,
        aiclk_mhz=1_100.0,
    )

    assert result["throughput_interval"]["load_interval_aiclk_source"] == (
        AICLK_SOURCE_LOAD_INTERVAL_TRACE
    )


def test_power_snapshot_names_in_run_maximum_for_varying_trace(tmp_path: Path):
    path = tmp_path / "trace.csv"
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,40,800,50\n"
        "2026-01-01T00:00:01+00:00,70,1350,51\n"
        "2026-01-01T00:00:02+00:00,60,1200,51\n",
        encoding="utf-8",
    )
    trace = read_power_trace_csv(
        path,
        options=PowerTraceCsvOptions(
            run_start="2026-01-01T00:00:00+00:00",
            run_end="2026-01-01T00:00:02+00:00",
            load_interval_start="2026-01-01T00:00:00+00:00",
            load_interval_end="2026-01-01T00:00:02+00:00",
            load_interval_source=ISSUE118_LOAD_INTERVAL_SOURCE,
            require_explicit_load_interval=True,
            require_all_rows_valid=True,
        ),
    )

    assert trace["aiclk_mhz"] == 1_350.0
    assert trace["aiclk_source"] == AICLK_SOURCE_RUN_TRACE_SAMPLES
    assert trace["aiclk_mhz_in_run_min"] == 800.0
    assert trace["aiclk_mhz_in_run_max"] == 1_350.0
    assert trace["aiclk_mhz_load_interval_weighted"] == pytest.approx(1_075.0)
    assert trace["aiclk_mhz_load_interval_weighted_source"] == AICLK_SOURCE_LOAD_INTERVAL_TRACE


def test_historical_analyzer_labels_legacy_elapsed_conversion():
    timestamps = [0, 1_000_000, 2_700_000]
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    record = {
        "clock": {"aiclk_mhz": 1_350},
        "parameters": {"budget_aiclk_mhz": 1_350},
        "telemetry_sampler": {"mode": "default", "interval_seconds": 2.0},
        "raw_timestamps": {
            "count": len(timestamps),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }

    result = analyze_run(
        timestamps,
        raw_metadata={"count": len(timestamps), "sha256": hashlib.sha256(raw).hexdigest()},
        record=record,
    )

    assert result["aiclk_mhz"] == 1_350
    assert result["aiclk_source"] == "legacy_unverified"
    assert result["aiclk_source_for_elapsed_seconds"] == "legacy_unverified"
    assert result["aiclk_selection_for_elapsed_seconds"] == "legacy_unverified"
    assert result["outlier_analysis"]["aiclk_source_for_elapsed_seconds"] == "legacy_unverified"


def test_conversion_inventory_lines_track_current_source():
    inventory = _conversion_inventory()
    source_lines = Path("enodia/tt/bench/run_resident.py").read_text().splitlines()
    actual_line = next(
        index for index, line in enumerate(source_lines, start=1) if "def _calibration_actual_duration(" in line
    )
    assert any(
        path == "enodia/tt/bench/run_resident.py"
        and line > actual_line
        and "calibration actual duration" in conversion
        for path, line, conversion, *_rest in inventory
    )
    assert "## Post-#124 cycle-ticks-to-seconds conversion inventory" in render_report(
        Path("docs/measurements")
    )


def _interval_extrema_config() -> ResidentConfig:
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


def _interval_extrema_environment() -> dict:
    return {
        "board": {"serial": "board", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": AUDITED_CLOCK_SOURCE_IMAGE,
        "image_pinned": True,
        "harness_commit": "a" * 40,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {
            "mode": "explicit",
            "interval_seconds": 1.0,
            "power_trace": "required",
            "timing_evidence": "available",
        },
        "aiclk_mhz_observed": [800],
    }


_INTERVAL_EXTREMA_CASES = (
    {
        "name": "pre-start-held",
        "run_start": "2026-01-01T00:00:01+00:00",
        "run_end": "2026-01-01T00:00:03+00:00",
        "aiclk_mhz": 1_350,
        "minimum": 800.0,
        "maximum": 1_350.0,
        "rows": (
            "2026-01-01T00:00:00+00:00,40,800,50\n"
            "2026-01-01T00:00:02+00:00,70,1350,51\n"
            "2026-01-01T00:00:03+00:00,70,1350,51\n"
            "2026-01-01T00:00:04+00:00,60,1200,50\n"
        ),
    },
    {
        "name": "endpoint",
        "run_start": "2026-01-01T00:00:00+00:00",
        "run_end": "2026-01-01T00:00:02+00:00",
        "aiclk_mhz": 1_350,
        "minimum": 800.0,
        "maximum": 1_350.0,
        "rows": (
            "2026-01-01T00:00:00+00:00,40,800,50\n"
            "2026-01-01T00:00:01+00:00,70,1350,51\n"
            "2026-01-01T00:00:02+00:00,40,700,50\n"
            "2026-01-01T00:00:03+00:00,60,1200,50\n"
        ),
    },
    {
        "name": "duplicate-start",
        "run_start": "2026-01-01T00:00:00+00:00",
        "run_end": "2026-01-01T00:00:03+00:00",
        "aiclk_mhz": 1_200,
        "minimum": 800.0,
        "maximum": 1_200.0,
        "rows": (
            "2026-01-01T00:00:00+00:00,40,800,50\n"
            "2026-01-01T00:00:01+00:00,70,1350,51\n"
            "2026-01-01T00:00:01+00:00,60,1200,50\n"
            "2026-01-01T00:00:03+00:00,40,900,50\n"
        ),
    },
    {
        "name": "no-interior-sample",
        "run_start": "2026-01-01T00:00:01+00:00",
        "run_end": "2026-01-01T00:00:03+00:00",
        "aiclk_mhz": 800,
        "minimum": 800.0,
        "maximum": 800.0,
        "rows": (
            "2026-01-01T00:00:00+00:00,40,800,50\n"
            "2026-01-01T00:00:03+00:00,70,1350,51\n"
            "2026-01-01T00:00:04+00:00,60,1200,50\n"
        ),
    },
)


def _build_interval_extrema_record(tmp_path: Path, case: dict) -> tuple[dict, Path]:
    run_id = f"run-122-{case['name']}"
    source = tmp_path / f"power-{run_id}.csv"
    source.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n" + "".join(case["rows"]),
        encoding="utf-8",
    )
    trace = atomic_power_trace_snapshot(
        source,
        tmp_path,
        run_id=run_id,
        snapshot_number=1,
        snapshot_prefix="resident-power-snapshot",
    )
    trace_metadata = read_power_trace_snapshot(
        trace,
        options=PowerTraceSnapshotOptions(
            run_id=run_id,
            run_start=case["run_start"],
            run_end=case["run_end"],
            snapshot_prefix="resident-power-snapshot",
            source_path=source,
            require_all_rows_valid=True,
            include_issue118_metadata=True,
        ),
    )
    record = build_measurement_record(
        config=_interval_extrema_config(),
        aiclk_mhz=case["aiclk_mhz"],
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_interval_extrema_environment(),
        power_trace=trace.name,
        power_trace_path=trace,
        power_trace_metadata=trace_metadata,
        interval_extrema_policy=INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD,
        run_start=case["run_start"],
        run_end=case["run_end"],
        timing_evidence=True,
    )
    return record, trace


@pytest.mark.parametrize(
    "case",
    _INTERVAL_EXTREMA_CASES,
    ids=[case["name"] for case in _INTERVAL_EXTREMA_CASES],
)
def test_new_resident_records_validate_positive_duration_interval_extrema(
    tmp_path: Path, case: dict
):
    record, trace = _build_interval_extrema_record(tmp_path, case)

    assert record["power_trace_interval_extrema_policy"] == (
        INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD
    )
    assert record["power_clock_provenance"]["interval_extrema_policy"] == (
        INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD
    )
    assert record["aiclk_mhz_in_run_min"] == case["minimum"]
    assert record["aiclk_mhz_in_run_max"] == case["maximum"]
    assert record["power_clock_provenance"]["aiclk_mhz_in_run_min"] == case[
        "minimum"
    ]
    assert record["power_clock_provenance"]["aiclk_mhz_in_run_max"] == case[
        "maximum"
    ]

    report = validate_resident_record(
        record,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=trace,
    )
    assert report["valid"] is True
    assert report["power_trace"]["immutable_snapshot"] is True
    assert report["power_trace"]["aiclk_mhz_in_run_min"] == case["minimum"]
    assert report["power_trace"]["aiclk_mhz_in_run_max"] == case["maximum"]


def test_historical_immutable_snapshot_keeps_timestamp_extrema_compatibility(
    tmp_path: Path,
):
    record, trace = _build_interval_extrema_record(tmp_path, _INTERVAL_EXTREMA_CASES[0])
    historical = deepcopy(record)
    historical.pop("power_trace_interval_extrema_policy")
    historical["power_clock_provenance"].pop("interval_extrema_policy")
    historical["aiclk_mhz_in_run_min"] = 1_350.0
    historical["power_clock_provenance"]["aiclk_mhz_in_run_min"] = 1_350.0

    report = validate_resident_record(
        historical,
        timestamps=[1_000, 2_000, 3_000],
        power_trace_path=trace,
    )
    assert report["valid"] is True
    assert historical["power_clock_provenance"]["immutable_snapshot"] is True
    assert report["power_trace"]["immutable_snapshot"] is False
    assert report["power_trace"]["aiclk_mhz_in_run_min"] == 1_350.0
    assert report["power_trace"]["aiclk_mhz_in_run_max"] == 1_350.0
