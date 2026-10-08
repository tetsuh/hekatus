"""Issue #104 regression tests for the canonical outlier-analysis catalog."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from enodia.tt.bench import run_resident
from enodia.tt.bench.resident_harness import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    ResidentConfig,
    build_measurement_record,
)
from enodia.tt.bench.resident_record import build_outlier_analysis
from tools.analyze_issue104_wrap import RUN_SPECS, analyze_file, analyze_run

AICLK_MHZ = 1_350
RECORDS_DIR = Path("docs/measurements")


def _environment(mode: str, interval_seconds: float | None) -> dict:
    sampler = {
        "mode": mode,
        "interval_seconds": interval_seconds,
        "power_trace": "absent_by_design" if mode == "off" else "required",
        "timing_evidence": "diagnostic_only" if mode == "off" else "available",
    }
    return {
        "board": {"serial": "board", "board_type": "p150a"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kmd_version": "2.11.0",
        "image": AUDITED_CLOCK_SOURCE_IMAGE,
        "image_pinned": True,
        "harness_commit": "a" * 40,
        "tt_env_active_release": "0.75.0",
        "telemetry_sampler": sampler,
        "aiclk_mhz_observed": [AICLK_MHZ],
    }


def _config(frame_count: int) -> ResidentConfig:
    return ResidentConfig(
        frame_count=frame_count,
        frame_interval_ticks=1_350_000,
        producer_core=(0, 0),
        consumer_core=(1, 0),
        ring_pages=4,
        work_per_frame=64,
        designated_timestamp_core=(1, 0),
        cycle_budget=10_000,
        outer_timeout_seconds=60,
        fixed_work_ticks_per_frame=100,
        budget_aiclk_mhz=AICLK_MHZ,
    )


def _raw_metadata(timestamps: list[int]) -> dict[str, object]:
    raw = b"".join(struct.pack("<Q", value) for value in timestamps)
    return {"count": len(timestamps), "sha256": hashlib.sha256(raw).hexdigest()}


def test_runner_imports_the_canonical_outlier_builder():
    assert run_resident.build_outlier_analysis is build_outlier_analysis


def _runner_record(timestamps: list[int]) -> dict:
    return build_measurement_record(
        config=_config(len(timestamps)),
        aiclk_mhz=AICLK_MHZ,
        timestamps=timestamps,
        producer_full_count=0,
        consumer_empty_count=0,
        kernel_error_flag=0,
        harness_commit="a" * 40,
        environment=_environment("off", None),
        power_trace=None,
    )


@pytest.mark.parametrize(
    ("timestamps", "expected_pair_count"),
    [
        ([0, 1_350_000, 2_700_000, 4_050_000], 0),
        ([0, 1_000_000, 2_700_000, 4_050_000], 1),
        (
            [
                0,
                1_000_000,
                2_700_000,
                4_050_000,
                5_050_000,
                6_750_000,
                8_100_000,
            ],
            2,
        ),
    ],
    ids=["zero-pair", "one-pair", "multi-pair"],
)
def test_runner_and_board_free_analyzer_outlier_catalogs_are_identical(
    timestamps: list[int], expected_pair_count: int
):
    record = _runner_record(timestamps)
    analyzer = analyze_run(
        timestamps,
        raw_metadata=_raw_metadata(timestamps),
        record=record,
    )

    runner_outlier = record["outlier_analysis"]
    analyzer_outlier = analyzer["outlier_analysis"]
    assert runner_outlier == analyzer_outlier
    assert runner_outlier["pair_count"] == expected_pair_count
    assert set(analyzer_outlier) - set(runner_outlier) == set()


def test_sampler_interval_comparison_reports_adjacent_gap_residuals():
    intervals = [1_000_000, 1_700_000] + [1_350_000] * 2_698 + [1_000_000, 1_700_000]
    timestamps = [0]
    for interval in intervals:
        timestamps.append(timestamps[-1] + interval)

    for mode, interval_seconds in (("default", 2.0), ("explicit", 5.0)):
        outlier = build_outlier_analysis(
            timestamps,
            aiclk_mhz=AICLK_MHZ,
            sampler_metadata={"mode": mode, "interval_seconds": interval_seconds},
        )
        comparison = outlier["sampler_interval_comparison"]
        assert comparison["status"] == "no_strict_alignment_observed"
        assert comparison["adjacent_gaps"]
        gap = comparison["adjacent_gaps"][0]
        elapsed = gap["gap_elapsed_seconds"]
        sampler_intervals = elapsed / interval_seconds
        nearest = round(sampler_intervals)
        assert gap["sampler_intervals"] == sampler_intervals
        assert gap["nearest_integer_sampler_intervals"] == nearest
        assert gap["residual_to_nearest_sampler_multiple_seconds"] == (
            elapsed - nearest * interval_seconds
        )

    off = build_outlier_analysis(
        timestamps,
        aiclk_mhz=AICLK_MHZ,
        sampler_metadata={"mode": "off", "interval_seconds": None},
    )
    assert off["sampler_interval_comparison"]["status"] == "not_applicable"
    assert "adjacent_gaps" not in off["sampler_interval_comparison"]


def test_committed_record_keys_are_preserved_by_regenerated_runner_catalog():
    timestamps = [0, 1_000_000, 2_700_000, 4_050_000]
    for record_path in sorted(RECORDS_DIR.glob("2026-10-06-p150a-issue104-sampler-*.json")):
        record = json.loads(record_path.read_text())
        regenerated = run_resident.build_outlier_analysis(
            timestamps,
            aiclk_mhz=AICLK_MHZ,
            sampler_metadata=record["telemetry_sampler"],
        )
        committed_keys = set(record["outlier_analysis"])
        assert committed_keys <= set(regenerated), record_path.name


def _optional_raw_path(relative_raw: str) -> Path | None:
    candidates = (
        Path("tests/fixtures/issue104") / relative_raw,
        Path("tests/data/issue104") / relative_raw,
        RECORDS_DIR / relative_raw,
    )
    return next((path for path in candidates if path.is_file()), None)


@pytest.mark.parametrize("run_name", tuple(RUN_SPECS))
def test_committed_sampler_shapes_regenerate_when_raw_artifact_is_available(run_name: str):
    relative_raw, record_name = RUN_SPECS[run_name]
    raw_path = _optional_raw_path(relative_raw)
    if raw_path is None:
        pytest.skip("the retained raw timestamp artifact is external to this checkout")

    record_path = RECORDS_DIR / record_name
    analysis = analyze_file(raw_path, record_path)
    regenerated = analysis["outlier_analysis"]
    committed = json.loads(record_path.read_text())["outlier_analysis"]
    assert set(committed) <= set(regenerated)
    assert analysis["pair_count"] == regenerated["pair_count"]
    comparison = regenerated["sampler_interval_comparison"]
    if run_name == "sampler-off":
        assert comparison["status"] == "not_applicable"
        assert "adjacent_gaps" not in comparison
    else:
        assert comparison["adjacent_gaps"] == committed["sampler_interval_comparison"]["adjacent_gaps"]
