"""Twelfth-round regressions for resident telemetry, timing, and numeric safety."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from enodia.tt.bench import run_resident, telemetry
from enodia.tt.bench.resident_harness import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    ResidentConfig,
    build_measurement_record,
    build_rejection_record,
)
from enodia.tt.bench.resident_record import (
    TIMING_EVIDENCE_ERROR_RECORD_REASON,
    TIMING_EVIDENCE_REJECTED_RECORD_REASON,
    validate_resident_record,
)

SNAPSHOT = json.dumps(
    {
        "device_info": [
            {
                "board_info": {
                    "bus_id": "0000:06:00.0",
                    "board_type": "p150a",
                    "board_id": "board-123",
                    "coords": "N/A",
                    "dram_status": True,
                    "dram_speed": "16G",
                    "pcie_speed": 4,
                    "pcie_width": "16",
                },
                "firmwares": {
                    "fw_bundle_version": "19.6.0.0",
                    "tt_flash_version": "N/A",
                    "cm_fw": "0.28.0.0",
                    "cm_fw_date": "2020-00-28",
                    "eth_fw": "0.0.0",
                    "dm_bl_fw": "0.0.0.0",
                    "dm_app_fw": "0.22.0.0",
                    "gddr_fw": "2.12",
                },
                "limits": {
                    "vdd_min": "0.70",
                    "vdd_max": "0.90",
                    "tdp_limit": "150",
                    "tdc_limit": "200",
                    "asic_fmax": "1350",
                    "therm_trip_l1_limit": "90",
                    "thm_limit": "110",
                    "bus_peak_limit": 0,
                    "fan_rpm_limit": 0,
                    "board_power_limit": "300",
                },
                "telemetry": {
                    "power": "75.0",
                    "aiclk": "1350",
                    "asic_temperature": "60.0",
                },
            }
        ]
    }
)


def _config(frame_count: int = 3) -> ResidentConfig:
    return ResidentConfig(
        frame_count=frame_count,
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


def _environment(mode: str = "off") -> dict:
    return {
        "board": {"serial": "board", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": AUDITED_CLOCK_SOURCE_IMAGE,
        "image_pinned": True,
        "harness_commit": "a" * 40,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": {
            "mode": mode,
            "interval_seconds": None if mode == "off" else 1.0,
            "power_trace": "absent_by_design" if mode == "off" else "required",
            "timing_evidence": "diagnostic_only" if mode == "off" else "available",
        },
        "aiclk_mhz_observed": [1_350],
    }


def _error_record() -> dict:
    return build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=[],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=1,
        harness_commit="a" * 40,
        environment=_environment(),
        power_trace=None,
        attempted_frame_count=1,
        produced_frame_count=0,
        dropped_frame_count=0,
        aborted_attempts=1,
        failure_check={
            "code": 2,
            "name": "producer_pacing_wait",
            "source": "producer",
            "elapsed_ticks": 1,
            "limit_ticks": 2,
            "unit": "device_clock_ticks",
            "valid": True,
        },
        timing_evidence=True,
    )


def test_capture_environment_shape_is_retained_with_warning_paths(monkeypatch):
    monkeypatch.setattr(telemetry, "_run", lambda _command: SNAPSHOT)
    monkeypatch.setattr(
        telemetry,
        "harness_identity",
        lambda: {"harness_commit": "a" * 40, "harness_dirty": False},
    )
    environment = telemetry.capture_environment(
        AUDITED_CLOCK_SOURCE_IMAGE,
        True,
        sampler_mode="off",
    )

    record = build_measurement_record(
        config=_config(),
        aiclk_mhz=1_350,
        timestamps=[1_000, 2_000, 3_000],
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit=environment["harness_commit"],
        environment=environment,
        power_trace=None,
    )
    record["frames_produced"] = 3
    record["frames_consumed"] = 3
    report = validate_resident_record(
        record,
        timestamps=[1_000, 2_000, 3_000],
        builder=True,
        raise_on_error=False,
    )

    assert report["valid"] is True
    assert record["environment"]["board_info"]["board_id"] == "board-123"
    expected_paths = {
        "environment.captured_at",
        "environment.kernel",
        "environment.harness_dirty",
        "environment.board_info",
        "environment.board_info.board_id",
        "environment.firmwares",
        "environment.firmwares.gddr_fw",
        "environment.limits",
        "environment.limits.board_power_limit",
        "environment.frames_produced",
    }
    assert expected_paths - {"environment.frames_produced"} <= set(report["warning_fields"])
    assert {"frames_produced", "frames_consumed"} <= set(record["validation_warnings"])


@pytest.mark.parametrize("field", ("timing_evidence", "power_clock_provenance.timing_evidence"))
def test_error_timing_mutations_are_rejected_by_analyzer_and_normalized_by_builder(field):
    candidate = _error_record()
    target = candidate
    components = field.split(".")
    for component in components[:-1]:
        target = target[component]
    target[components[-1]] = True

    report = validate_resident_record(candidate)
    assert report["valid"] is False
    assert any(
        mismatch["invariant"] == "clock.timing_evidence"
        for mismatch in report["mismatches"]
    )

    normalized = copy.deepcopy(candidate)
    validate_resident_record(normalized, builder=True, raise_on_error=False)
    assert normalized["timing_evidence"] is False
    assert normalized["power_clock_provenance"]["timing_evidence"] is False
    assert normalized["timing_evidence_reason"] == TIMING_EVIDENCE_ERROR_RECORD_REASON
    assert (
        normalized["power_clock_provenance"]["timing_evidence_reason"]
        == TIMING_EVIDENCE_ERROR_RECORD_REASON
    )


@pytest.mark.parametrize("mutation", ("timing_evidence", "power_clock_provenance", "both"))
def test_rejected_timing_mutations_cannot_use_the_rejected_skip_path(mutation):
    candidate = build_rejection_record(
        config=_config(), reason="preflight rejected", environment=_environment()
    )
    if mutation in {"timing_evidence", "both"}:
        candidate["timing_evidence"] = True
    if mutation in {"power_clock_provenance", "both"}:
        candidate["power_clock_provenance"]["timing_evidence"] = True

    report = validate_resident_record(candidate)
    assert report["valid"] is False

    normalized = copy.deepcopy(candidate)
    validate_resident_record(normalized, builder=True, raise_on_error=False)
    assert normalized["timing_evidence"] is False
    assert normalized["power_clock_provenance"]["timing_evidence"] is False
    assert normalized["timing_evidence_reason"] == TIMING_EVIDENCE_REJECTED_RECORD_REASON


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e9999", "not-a-number", "0", "-1"])
def test_nonfinite_and_out_of_range_telemetry_values_are_skipped(value):
    snapshot = json.dumps(
        {
            "device_info": [
                {
                    "telemetry": {
                        "power": value,
                        "aiclk": value,
                        "asic_temperature": value,
                    }
                }
            ]
        }
    )
    assert telemetry.parse_telemetry(snapshot) is None
    assert telemetry.telemetry_csv_row(snapshot, timestamp="2026-01-01T00:00:00+00:00") is None
    assert "aiclk_mhz_observed" not in telemetry.parse_environment(snapshot)


@pytest.mark.parametrize("field", ("power_w", "aiclk_mhz", "asic_temp_c"))
def test_power_trace_nonfinite_and_overflow_rows_are_unusable(tmp_path: Path, field: str):
    values = {"power_w": "75", "aiclk_mhz": "1350", "asic_temp_c": "60"}
    values[field] = "1e9999"
    trace = tmp_path / f"{field}.csv"
    trace.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        f"2026-01-01T00:00:00+00:00,{values['power_w']},{values['aiclk_mhz']},{values['asic_temp_c']}\n"
    )
    parsed = telemetry.parse_power_trace(
        trace,
        run_start="2026-01-01T00:00:00+00:00",
        run_end="2026-01-01T00:00:00+00:00",
    )
    assert parsed["valid_row_count"] == 0
    assert parsed["coverage_complete"] is False
    assert parsed["errors"]


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "1e9999", "bad", 0, -1])
def test_runner_aiclk_conversions_classify_unusable_values(value):
    assert run_resident._safe_aiclk_integer(value) is None
    assert run_resident._environment_aiclk_values({"aiclk_mhz_observed": [value]}) == []
    assert run_resident._power_aiclk(
        None, {"aiclk_mhz_observed": [value]}, allow_environment_snapshot=True
    ) is None


def test_runner_trace_aiclk_conversion_skips_bad_rows_and_preserves_valid_rows(
    monkeypatch, tmp_path: Path
):
    trace = tmp_path / "power.csv"
    trace.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "t,75,inf,60\n"
        "t,75,1350,60\n"
        "t,75,not-a-number,60\n"
    )

    class _OutRoot:
        def __truediv__(self, _name):
            return trace

    monkeypatch.setattr(run_resident, "Path", lambda _value: _OutRoot())
    assert run_resident._power_trace_aiclk_values("power.csv") == [1_350]


def test_runner_device_word_conversion_classifies_nonfinite_values():
    for value in (float("inf"), float("-inf"), float("nan"), "bad", 2**32):
        with pytest.raises(run_resident.ResidentResultUnavailable):
            run_resident._device_word(value, "test word")
