"""Board-free acceptance coverage for Issue #110 validator hardening."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pytest

from enodia.tt.bench.resident_harness import (
    ResidentConfig,
    build_measurement_record,
    build_rejection_record,
)
from enodia.tt.bench.resident_record import validate_resident_record
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
    *, mode: str, outcome: str, trace_path: Path | None
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
