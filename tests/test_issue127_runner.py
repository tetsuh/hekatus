"""Board-free contract tests for the Issue #127 ABBA/BADC runner."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from enodia.tt.bench.power_trace_snapshot import EXPLICIT_FINAL_SAMPLE
from tools import newton_schulz_issue88 as issue88
from tools import newton_schulz_issue127 as runner

ROOT = Path(__file__).parents[1]
MEASUREMENTS = ROOT / "docs/measurements"
PLAN = MEASUREMENTS / "2026-10-12-host-newton-schulz-issue127-abba-plan.json"
SUMMARY = MEASUREMENTS / "2026-10-15-p150a-newton-schulz-issue127-abba-summary.json"
RUN_RECORDS = {
    "confirmation": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-confirmation-run1.json",
    "measurement-1": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-measurement-1.json",
    "measurement-2": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-measurement-2-failed-cooldown.json",
    "measurement-2-rerun": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-measurement-2-rerun.json",
}
POWER_TRACES = {
    "confirmation": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-confirmation-run1-power.csv",
    "measurement-1": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-measurement-1-power.csv",
    "measurement-2": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-measurement-2-failed-cooldown-power.csv",
    "measurement-2-rerun": MEASUREMENTS
    / "2026-10-15-p150a-newton-schulz-issue127-measurement-2-rerun-power.csv",
}
ACCEPTED_SCOPES = ("measurement-1", "measurement-2-rerun")
CSV_COLUMNS = ("timestamp_utc", "power_w", "aiclk_mhz", "asic_temp_c")
CSV_HEADER = ",".join(CSV_COLUMNS) + "\n"


def _environment(*, observed: list[int] | None = None) -> dict:
    return {
        "run_id": "run127-test",
        "image": "ghcr.io/tenstorrent/tt-metal@sha256:" + "a" * 64,
        "image_pinned": True,
        "image_digest": "sha256:" + "a" * 64,
        "kernel": "Linux 6.8.0-1-generic",
        "kmd_version": "2.11.0",
        "tt_env_active_release": "0.75.0",
        "harness_commit": "b" * 40,
        "harness_dirty": False,
        "board_info": {
            "board_type": "p150a",
            "board_id": "0000040331914124",
            "bus_id": "0000:09:00.0",
        },
        "firmwares": {"fw_bundle_version": "19.6.0.0"},
        "limits": {"asic_fmax": "1350"},
        "device_node": "/dev/tenstorrent/0",
        "telemetry_sampler": {
            "mode": "default",
            "interval_seconds": 2.0,
            "power_trace": "required",
            "timing_evidence": "available",
        },
        "aiclk_mhz_observed": [1350] if observed is None else observed,
        "hostname": "must-not-be-recorded",
        "worktree_path": "/home/user/private-worktree",
    }


def test_issue127_canonical_controls_and_placements_are_exactly_four_rows():
    rows = runner.ISSUE127_COMPARISON_ROWS
    runner.validate_comparison_rows()

    assert [row["name"] for row in rows] == [
        "bf16-r-L16",
        "fp32-r-L16",
        "bf16-r-L32-r-dram",
        "fp32-r-L32-r-dram",
    ]
    assert [(row["variant"], row["size"], row["r_memory"]) for row in rows] == [
        ("bf16", 16, "l1"),
        ("fp32-r", 16, "l1"),
        ("bf16", 32, "dram"),
        ("fp32-r", 32, "dram"),
    ]
    assert {
        key: rows[0][key]
        for key in (
            "batch",
            "iterations",
            "initial_value",
            "state_format",
            "destination_format",
            "fp32_dest_acc_en",
            "math_fidelity",
            "fuse_s",
            "input_memory",
            "matrix_block",
            "double_buffer",
            "dst_full_sync_en",
            "output_memory",
            "x0_memory",
        )
    } == {
        "batch": 8192,
        "iterations": 12,
        "initial_value": "I/||R||inf",
        "state_format": "BF16",
        "destination_format": "FP32",
        "fp32_dest_acc_en": True,
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "input_memory": "l1",
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "output_memory": "dram",
        "x0_memory": "l1",
    }


def test_issue127_orders_and_occurrence_balance_are_fixed():
    run_1 = runner.build_occurrence_plan(1)
    run_2 = runner.build_occurrence_plan(2)

    assert [row["row"] for row in run_1] == list(runner.RUN_1_ORDER)
    assert [row["row"] for row in run_2] == list(runner.RUN_2_ORDER)
    for plan in (run_1, run_2):
        assert [row["order_index"] for row in plan] == list(range(8))
        assert {row["row"] for row in plan} == set(runner.ISSUE127_ROW_NAMES)
        assert {name: sum(row["row"] == name for row in plan) for name in runner.ISSUE127_ROW_NAMES} == {
            name: 2 for name in runner.ISSUE127_ROW_NAMES
        }
        assert all(row["occurrence_id"] == f"{row['row']}#{row['occurrence']}" for row in plan)


def test_issue127_launch_counts_and_duration_guard_are_explicit():
    assert runner.launches_for_row("bf16-r-L16") == 11000
    assert runner.launches_for_row("fp32-r-L16") == 11000
    assert runner.launches_for_row("bf16-r-L32-r-dram") == 5800
    assert runner.launches_for_row("fp32-r-L32-r-dram") == 5800

    passing = runner.row_duration_guard(2.7, "bf16-r-L16")
    assert passing["passes"] is True
    assert passing["within_target"] is True
    assert passing["target_seconds_max"] is None
    assert "hard_max_pass" not in passing
    assert runner.row_duration_guard(3.123, "bf16-r-L32-r-dram")["passes"] is True
    assert runner.row_duration_guard(2.3, "bf16-r-L16")["passes"] is True
    assert runner.row_duration_guard(2.24, "bf16-r-L16")["passes"] is False


def test_issue127_run_budget_projection_stops_before_projected_overrun():
    within = runner.project_run_budget(
        12.0,
        [{"timed_duration_seconds": 4.2}],
        remaining_occurrences=7,
    )
    assert within["within_budget"] is True
    assert within["estimated_seconds_per_remaining_occurrence"] == pytest.approx(6.0)

    over = runner.project_run_budget(
        25.0,
        [{"timed_duration_seconds": 7.0}],
        remaining_occurrences=6,
    )
    assert over["within_budget"] is False
    assert over["projected_total_seconds"] > runner.RUN_TIMEOUT_SECONDS


def test_issue127_run_comparison_stops_before_projected_next_occurrence(monkeypatch):
    clock_values = iter((0.0, 0.0, 10.0, 10.0))
    occurrence_calls = []

    monkeypatch.setattr(
        runner,
        "_prepare_inputs",
        lambda *args, **kwargs: (object(), {}),
    )
    monkeypatch.setattr(
        runner,
        "_run_sustained_warmup",
        lambda *args, **kwargs: {"end_utc": "2026-10-12T00:00:00+00:00"},
    )

    def fake_occurrence(*args, **kwargs):
        occurrence = args[4]
        occurrence_calls.append(occurrence["order_index"])
        return {
            **occurrence,
            "status": "ok",
            "timed_duration_seconds": 7.0,
        }

    monkeypatch.setattr(runner, "_run_occurrence", fake_occurrence)
    result = runner.run_comparison(
        object(),
        object(),
        run_number=1,
        run_id="run127-test",
        time_fn=lambda: next(clock_values),
        kernel_class=object,
    )

    assert occurrence_calls == [0]
    assert result["status"] == "failed"
    assert result["occurrences_completed"] == 1
    assert result["run_budget"]["stopped_before_occurrence"] == 1
    assert result["failure"]["order_index"] == 1


def test_issue127_timeout_policy_keeps_default_and_closes_approved_exception():
    default_timeout, default_policy = runner.resolve_run_timeout(1)
    assert default_timeout == 60.0
    assert default_policy["approved_override"] is False

    for scope, run_number in runner.APPROVED_RUN_TARGETS.items():
        timeout, policy = runner.resolve_run_timeout(
            run_number,
            runner.APPROVED_RUN_TIMEOUT_SECONDS,
            scope,
        )
        assert timeout == 90.0
        assert policy["approved_override"] is True
        assert policy["approval_date"] == "2026-10-16"
        assert policy["scope"] == scope
        assert policy["run_number"] == run_number

    with pytest.raises(ValueError, match="requires --run-scope"):
        runner.resolve_run_timeout(1, runner.APPROVED_RUN_TIMEOUT_SECONDS)
    with pytest.raises(ValueError, match="approved target"):
        runner.resolve_run_timeout(1, runner.APPROVED_RUN_TIMEOUT_SECONDS, "other")
    with pytest.raises(ValueError, match="requires run_number=2"):
        runner.resolve_run_timeout(1, runner.APPROVED_RUN_TIMEOUT_SECONDS, "measurement-2")
    with pytest.raises(ValueError, match="only the default 60-second"):
        runner.resolve_run_timeout(1, 75.0, "confirmation")

    projected = runner.project_run_budget(
        25.0,
        [{"timed_duration_seconds": 7.0}],
        remaining_occurrences=6,
        run_timeout_seconds=90.0,
    )
    assert projected["run_timeout_seconds"] == 90.0
    assert projected["within_budget"] is True


def test_issue127_timed_bounds_and_adjusted_p50_formula():
    row = {
        "timed_start_utc": "2026-10-12T00:00:00+00:00",
        "timed_end_utc": "2026-10-12T00:00:02.700000+00:00",
        "timed_start_monotonic": 10.0,
        "timed_end_monotonic": 12.7,
        "timed_duration_seconds": 2.7,
        "size": 16,
    }
    runner.validate_row_bounds(row)
    assert runner.adjusted_p50_tflops(16.0, 1200.0) == pytest.approx(18.0)
    assert runner.adjusted_p50_tflops(16.0, None) is None

    with pytest.raises(ValueError, match="strictly increasing"):
        runner.validate_row_bounds({**row, "timed_end_utc": row["timed_start_utc"]})


def test_issue127_shared_left_held_row_facts_include_start_and_reject_unbounded_end():
    samples = [
        {"timestamp_utc": "2026-10-12T00:00:00+00:00", "power_w": "40", "aiclk_mhz": "800", "asic_temp_c": "50"},
        {"timestamp_utc": "2026-10-12T00:00:02+00:00", "power_w": "70", "aiclk_mhz": "1350", "asic_temp_c": "51"},
        {"timestamp_utc": "2026-10-12T00:00:03+00:00", "power_w": "40", "aiclk_mhz": "700", "asic_temp_c": "50"},
        {"timestamp_utc": "2026-10-12T00:00:04+00:00", "power_w": "60", "aiclk_mhz": "1200", "asic_temp_c": "50"},
    ]
    facts = runner.left_held_aiclk_interval_facts(
        samples,
        start="2026-10-12T00:00:01+00:00",
        end="2026-10-12T00:00:04+00:00",
    )
    assert facts["available"] is True
    assert facts["first_held_aiclk_mhz"] == 800.0
    assert facts["aiclk_mhz_min"] == 700.0
    assert facts["aiclk_mhz_max"] == 1350.0
    assert facts["time_weighted_aiclk_mhz"] == pytest.approx((800 + 1350 + 700) / 3)
    assert facts["semantics"] == "left_held_positive_segments"

    unavailable = runner.left_held_aiclk_interval_facts(
        samples,
        start="2026-10-12T00:00:03+00:00",
        end="2026-10-12T00:00:05+00:00",
    )
    assert unavailable["available"] is False
    assert unavailable["time_weighted_aiclk_mhz"] is None
    assert unavailable["aiclk_mhz_min"] is None
    assert unavailable["aiclk_mhz_max"] is None
    assert unavailable["first_held_aiclk_mhz"] is None
    assert unavailable["last_held_aiclk_mhz"] is None
    assert unavailable["segments"] == []


@pytest.mark.parametrize(
    ("start", "end", "invalid_bound"),
    (
        (
            "not-a-timestamp",
            "2026-10-12T00:00:04+00:00",
            "start",
        ),
        (
            "2026-10-12T00:00:00+00:00",
            "not-a-timestamp",
            "end",
        ),
    ),
)
def test_issue127_invalid_interval_timestamp_keeps_common_unavailable_schema(
    start, end, invalid_bound
):
    facts = runner.left_held_aiclk_interval_facts(
        [
            {
                "timestamp_utc": "2026-10-12T00:00:00+00:00",
                "power_w": "40",
                "aiclk_mhz": "800",
                "asic_temp_c": "50",
            }
        ],
        start=start,
        end=end,
    )

    assert {
        "available",
        "coverage_complete",
        "reason",
        "error",
        "aiclk_mhz_min",
        "aiclk_mhz_max",
        "time_weighted_aiclk_mhz",
        "coverage_seconds",
        "requested_start_utc",
        "requested_end_utc",
        "first_held_aiclk_mhz",
        "first_held_sample_timestamp_utc",
        "last_held_aiclk_mhz",
        "segments",
        "usable_sample_count",
        "invalid_sample_count",
        "errors",
        "semantics",
    } <= facts.keys()
    assert facts["available"] is False
    assert facts["coverage_complete"] is False
    assert facts["reason"] == "interval_timestamp_invalid"
    assert facts["error"] == (
        f"power interval {invalid_bound} is not a valid ISO-8601 timestamp"
    )
    assert facts["aiclk_mhz_min"] is None
    assert facts["aiclk_mhz_max"] is None
    assert facts["time_weighted_aiclk_mhz"] is None
    assert facts["coverage_seconds"] == 0.0
    assert facts["requested_start_utc"] is None
    assert facts["requested_end_utc"] is None
    assert facts["first_held_aiclk_mhz"] is None
    assert facts["first_held_sample_timestamp_utc"] is None
    assert facts["last_held_aiclk_mhz"] is None
    assert facts["segments"] == []
    assert facts["usable_sample_count"] == 0
    assert facts["invalid_sample_count"] == 0
    assert facts["errors"] == []
    assert facts["semantics"] == "left_held_positive_segments"


def test_issue127_final_trace_facts_record_source_and_started_sample_target():
    row = {
        **runner.build_occurrence_plan(1)[0],
        "timed_start_utc": "2026-10-12T00:00:01+00:00",
        "timed_end_utc": "2026-10-12T00:00:03+00:00",
        "timed_start_monotonic": 1.0,
        "timed_end_monotonic": 3.0,
        "timed_duration_seconds": 2.0,
        "raw_p50_tflops": 16.0,
    }
    two_started_row = {
        **row,
        "timed_end_utc": "2026-10-12T00:00:05+00:00",
        "timed_end_monotonic": 5.0,
        "timed_duration_seconds": 4.0,
    }
    trace = {
        "file": "issue127-abba-power-snapshot-run127-test-1.csv",
        "source_file": "power-run127-test.csv",
        "sha256": "c" * 64,
        "sha256_definition": "SHA-256 of the complete immutable per-run power CSV snapshot bytes",
        "coverage_definition": EXPLICIT_FINAL_SAMPLE,
        "coverage_complete": True,
        "immutable_snapshot": True,
        "samples": [
            {"timestamp_utc": "2026-10-12T00:00:00+00:00", "power_w": "40", "aiclk_mhz": "1350", "asic_temp_c": "50"},
            {"timestamp_utc": "2026-10-12T00:00:02+00:00", "power_w": "70", "aiclk_mhz": "1200", "asic_temp_c": "51"},
            {"timestamp_utc": "2026-10-12T00:00:04+00:00", "power_w": "60", "aiclk_mhz": "1200", "asic_temp_c": "50"},
            {"timestamp_utc": "2026-10-12T00:00:06+00:00", "power_w": "60", "aiclk_mhz": "1200", "asic_temp_c": "50"},
        ],
    }
    runner.apply_final_trace_facts([row], power_trace=trace)

    assert row["aiclk_interval"]["trace_sha256"] == "c" * 64
    assert row["row_time_weighted_aiclk_mhz"] == pytest.approx(1275.0)
    assert row["adjusted_p50_tflops"] == pytest.approx(16.0 * 1350.0 / 1275.0)
    assert row["aiclk_interval"]["requested_sampler_interval_seconds"] == 2.0
    assert row["aiclk_interval"]["power_sample_count_in_interval"] == 2
    assert row["aiclk_interval"]["power_sample_count_definition"] == (
        "samples contributing to the timed interval [start, end), including the "
        "sample held across the interval start"
    )
    assert row["aiclk_interval"]["power_samples_started_in_interval"] == 1
    assert row["aiclk_interval"]["power_samples_started_in_interval_definition"] == (
        "samples whose timestamp starts in the timed interval [start, end); the "
        "left-held sample whose timestamp precedes the interval start is excluded"
    )
    assert row["aiclk_interval"]["power_sample_target_met"] is False

    runner.apply_final_trace_facts([two_started_row], power_trace=trace)

    assert two_started_row["aiclk_interval"]["power_sample_count_in_interval"] == 3
    assert two_started_row["aiclk_interval"]["power_samples_started_in_interval"] == 2
    assert two_started_row["aiclk_interval"]["power_sample_target_met"] is True


def test_issue127_started_sample_count_excludes_left_held_and_end_boundary_samples():
    samples = [
        {"timestamp_utc": "2026-10-12T00:00:00+00:00"},
        {"timestamp_utc": "2026-10-12T00:00:02+00:00"},
        {"timestamp_utc": "2026-10-12T00:00:04+00:00"},
    ]

    assert runner._power_samples_started_in_interval(
        samples,
        start="2026-10-12T00:00:01+00:00",
        end="2026-10-12T00:00:04+00:00",
    ) == 1
    assert runner._power_samples_started_in_interval(
        samples,
        start="2026-10-12T00:00:02+00:00",
        end="2026-10-12T00:00:04+00:00",
    ) == 1


def test_issue127_final_trace_waits_for_sample_after_run_end(tmp_path: Path):
    run_id = "run127-rollover"
    source = tmp_path / f"power-{run_id}.csv"
    header = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
    before_end = (
        header
        + "2026-10-12T00:00:00+00:00,40,1200,50\n"
        + "2026-10-12T00:00:03+00:00,50,1200,51\n"
    )
    after_end = before_end + "2026-10-12T00:00:05+00:00,60,1350,52\n"
    source.write_text(before_end, encoding="utf-8")
    def update_sampler_source(_interval: float) -> None:
        source.write_text(after_end, encoding="utf-8")

    trace = runner._capture_final_power_trace(
        tmp_path,
        run_id=run_id,
        run_start="2026-10-12T00:00:00+00:00",
        run_end="2026-10-12T00:00:04+00:00",
        power_path=source,
        monotonic_fn=lambda: 0.0,
        sleep_fn=update_sampler_source,
    )

    assert trace["coverage_definition"] == EXPLICIT_FINAL_SAMPLE
    assert trace["coverage_complete"] is True
    assert trace["last_timestamp"] == "2026-10-12T00:00:05+00:00"
    assert trace["file"].endswith("-2.csv")


def test_issue127_warmup_800mhz_hold_gate_is_verified_from_row_facts():
    warmup = {"end_utc": "2026-10-12T00:00:01+00:00"}
    first_row = {
        "timed_start_utc": "2026-10-12T00:00:02+00:00",
        "aiclk_interval": {
            "first_held_aiclk_mhz": 1350.0,
            "coverage_complete": True,
            "trace_coverage_complete": True,
        },
    }
    gate = runner.verify_warmup_hold_gate(
        warmup,
        first_row,
        power_trace={"file": "snapshot.csv", "sha256": "d" * 64},
    )
    assert gate["verified"] is True
    assert warmup["hold_gate_verified"] is True


def test_issue127_cooldown_gate_uses_power_and_temperature_tolerances_only():
    passing = runner.evaluate_cooldown_gate(
        52.0,
        61.5,
        observed_idle_aiclk_mhz=800.0,
    )
    assert passing["passes"] is True
    assert passing["result"] == "pass"
    assert passing["power_limit_w"] == 52.0
    assert passing["temperature_limit_c"] == 61.5
    assert passing["aiclk_gate"] is False

    failing = runner.evaluate_cooldown_gate(
        52.01,
        61.5,
        observed_idle_aiclk_mhz=800.0,
    )
    assert failing["passes"] is False
    assert failing["failure_type"] == "cooldown_gate_failure"
    assert runner.evaluate_cooldown_gate(47.0, 58.5)["passes"] is True


def test_issue127_cooldown_metadata_requires_wait_and_keeps_pre_cooldown_aiclk_provenance():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    runner.validate_harness_environment(environment, run_id="run127-test")
    metadata = runner.build_cooldown_metadata(
        runner.CooldownMetadataParameters(
            run_number=2,
            wait_seconds=30.0,
            expected_aiclk_mhz=1350.0,
            environment=environment,
            start_utc="2026-10-12T00:00:00+00:00",
            end_utc="2026-10-12T00:00:30+00:00",
            start_monotonic=10.0,
            end_monotonic=40.0,
            wait_performed=True,
        )
    )
    assert metadata["environment_verified"] is True
    assert metadata["environment_observed_aiclk_mhz"] == [800]
    assert metadata["observed_idle_aiclk_mhz"] == 800.0
    assert metadata["baseline_source"] == "measurement-1 pre-run power trace"
    assert metadata["baseline_power_w"] == 47.0
    assert metadata["baseline_temperature_c"] == 58.5
    assert metadata["power_tolerance_w"] == 5.0
    assert metadata["temperature_tolerance_c"] == 3.0
    assert metadata["power_limit_w"] == 52.0
    assert metadata["temperature_limit_c"] == 61.5
    assert metadata["wait_cap_seconds"] == 600.0
    assert metadata["aiclk_gate"] is False
    assert metadata["verified"] is False
    with pytest.raises(ValueError, match="gate proof"):
        runner.validate_cooldown_metadata(metadata, environment=environment, require_power_trace=True)
    with pytest.raises(ValueError, match="wait bounds"):
        runner.validate_cooldown_metadata(
            {**metadata, "cooldown_wait_seconds": None}, environment=environment
        )
    with pytest.raises(ValueError, match="600-second cap"):
        runner.validate_cooldown_metadata(
            {**metadata, "cooldown_wait_seconds": 600.1}, environment=environment
        )
    # AICLK remains valid provenance even when it is not the nominal clock.
    runner.validate_cooldown_metadata(
        {**metadata, "expected_aiclk_mhz": 1200}, environment=environment
    )


def _cooldown_trace(*, terminal_timestamp: str = "2026-10-12T00:00:04+00:00") -> dict:
    return {
        "file": "issue127-abba-power-snapshot-run127-test-1.csv",
        "source_file": "power-run127-test.csv",
        "sha256": "e" * 64,
        "sha256_definition": "SHA-256 of the complete immutable per-run power CSV snapshot bytes",
        "coverage_definition": EXPLICIT_FINAL_SAMPLE,
        "coverage_complete": True,
        "immutable_snapshot": True,
        "samples": [
            {"timestamp_utc": "2026-10-12T00:00:00+00:00", "power_w": "40", "aiclk_mhz": "800", "asic_temp_c": "50"},
            {"timestamp_utc": "2026-10-12T00:00:01+00:00", "power_w": "50", "aiclk_mhz": "1000", "asic_temp_c": "51"},
            {"timestamp_utc": "2026-10-12T00:00:02+00:00", "power_w": "50", "aiclk_mhz": "800", "asic_temp_c": "51"},
            {"timestamp_utc": terminal_timestamp, "power_w": "50", "aiclk_mhz": "800", "asic_temp_c": "51"},
        ],
    }


def test_issue127_cooldown_proof_uses_power_temperature_gate_and_keeps_idle_clock_provenance():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    runner.validate_harness_environment(environment, run_id="run127-test")
    metadata = runner.build_cooldown_metadata(
        runner.CooldownMetadataParameters(
            run_number=2,
            wait_seconds=3.0,
            expected_aiclk_mhz=1350.0,
            environment=environment,
            start_utc="2026-10-12T00:00:00.500000+00:00",
            end_utc="2026-10-12T00:00:03.500000+00:00",
            wait_performed=True,
            observed_power_w=50.0,
            observed_temperature_c=51.0,
            gate_result="pass",
            gate_verified=True,
        )
    )
    verified = runner.verify_cooldown_power_trace(
        metadata, power_trace=_cooldown_trace()
    )
    assert verified["power_trace_verified"] is True
    assert verified["verified"] is True
    assert verified["power_trace_facts"]["first_held_aiclk_mhz"] == 800.0
    assert verified["power_trace_facts"]["terminal_held_aiclk_mhz"] == 800.0
    assert verified["power_trace_facts"]["gate"]["result"] == "pass"
    assert verified["power_trace_facts"]["gate"]["aiclk_gate"] is False

    unbounded = runner.verify_cooldown_power_trace(
        metadata,
        power_trace=_cooldown_trace(terminal_timestamp="2026-10-12T00:00:03+00:00"),
    )
    # Missing terminal AICLK coverage remains visible provenance, but it does
    # not turn an otherwise passing power/temperature gate into a failure.
    assert unbounded["power_trace_verified"] is True
    assert unbounded["power_trace_facts"]["coverage_complete"] is False
    assert unbounded["verified"] is True


def test_issue127_run2_args_require_explicit_passing_verified_gate():
    parser = runner._build_parser()
    base_args = [
        "--run-number",
        "2",
        "--cooldown-start-utc",
        "2026-10-12T00:00:00+00:00",
        "--cooldown-end-utc",
        "2026-10-12T00:00:03+00:00",
    ]
    for extra_args in (
        [],
        ["--cooldown-gate-result", "pass"],
        [
            "--cooldown-gate-result",
            "pass",
            "--no-cooldown-gate-verified",
        ],
        ["--cooldown-gate-result", "fail", "--cooldown-gate-verified"],
    ):
        args = parser.parse_args(base_args + extra_args)
        with pytest.raises(SystemExit):
            runner._validate_args(parser, args)

    runner._validate_args(
        parser,
        parser.parse_args(
            base_args
            + ["--cooldown-gate-result", "pass", "--cooldown-gate-verified"]
        ),
    )


def test_issue127_cooldown_proof_requires_finite_wrapper_observations():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    metadata = runner.build_cooldown_metadata(
        runner.CooldownMetadataParameters(
            run_number=2,
            wait_seconds=3.0,
            expected_aiclk_mhz=1350.0,
            environment=environment,
            start_utc="2026-10-12T00:00:00.500000+00:00",
            end_utc="2026-10-12T00:00:03.500000+00:00",
            wait_performed=True,
            observed_power_w=50.0,
            observed_temperature_c=51.0,
            gate_result="pass",
            gate_verified=True,
        )
    )
    metadata.pop("observed_power_w")
    metadata["observed_temperature_c"] = float("nan")

    verified = runner.verify_cooldown_power_trace(
        metadata, power_trace=_cooldown_trace()
    )

    assert verified["power_trace_facts"]["gate"]["wrapper_gate_proof"] is True
    assert verified["power_trace_facts"]["metadata_observation_match"] is False
    assert set(verified["power_trace_facts"]["metadata_observation_mismatch"]) == {
        "observed_power_w",
        "observed_temperature_c",
    }
    assert verified["gate_verified"] is False
    assert verified["power_trace_verified"] is False
    assert verified["verified"] is False


@pytest.mark.parametrize(
    ("gate_result", "gate_verified"),
    (
        ("pending", True),
        ("pass", False),
        (None, True),
        ("pass", None),
    ),
)
def test_issue127_cooldown_proof_requires_passing_wrapper_gate_proof(
    gate_result: str | None, gate_verified: bool | None
):
    environment = runner.sanitize_environment(_environment(observed=[800]))
    metadata = runner.build_cooldown_metadata(
        runner.CooldownMetadataParameters(
            run_number=2,
            wait_seconds=3.0,
            expected_aiclk_mhz=1350.0,
            environment=environment,
            start_utc="2026-10-12T00:00:00.500000+00:00",
            end_utc="2026-10-12T00:00:03.500000+00:00",
            wait_performed=True,
            observed_power_w=50.0,
            observed_temperature_c=51.0,
            gate_result="pass",
            gate_verified=True,
        )
    )
    if gate_result is None:
        metadata.pop("gate_result")
    else:
        metadata["gate_result"] = gate_result
    if gate_verified is None:
        metadata.pop("gate_verified")
    else:
        metadata["gate_verified"] = gate_verified

    verified = runner.verify_cooldown_power_trace(
        metadata, power_trace=_cooldown_trace()
    )

    assert verified["power_trace_facts"]["metadata_observation_match"] is True
    assert verified["power_trace_facts"]["gate"]["wrapper_gate_proof"] is False
    assert verified["gate_verified"] is False
    assert verified["power_trace_verified"] is False
    assert verified["verified"] is False


def test_issue127_cooldown_proof_rejects_metadata_that_overrides_a_failing_trace_sample():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    runner.validate_harness_environment(environment, run_id="run127-test")
    metadata = runner.build_cooldown_metadata(
        runner.CooldownMetadataParameters(
            run_number=2,
            wait_seconds=3.0,
            expected_aiclk_mhz=1350.0,
            environment=environment,
            start_utc="2026-10-12T00:00:00.500000+00:00",
            end_utc="2026-10-12T00:00:03.500000+00:00",
            wait_performed=True,
            observed_power_w=47.0,
            observed_temperature_c=58.5,
        )
    )
    trace = _cooldown_trace()
    trace["samples"][2] = {
        "timestamp_utc": "2026-10-12T00:00:02+00:00",
        "power_w": "70",
        "aiclk_mhz": "800",
        "asic_temp_c": "75",
    }

    verified = runner.verify_cooldown_power_trace(metadata, power_trace=trace)

    assert verified["observed_power_w"] == 70.0
    assert verified["observed_temperature_c"] == 75.0
    assert verified["power_trace_facts"]["gate_sample"]["power_w"] == "70"
    assert verified["power_trace_facts"]["gate"]["passes"] is False
    assert verified["gate_verified"] is False
    assert verified["power_trace_verified"] is False
    assert verified["verified"] is False
    assert verified["power_trace_facts"]["gate_observation_source"] == "immutable_trace"
    assert set(verified["power_trace_facts"]["metadata_observation_mismatch"]) == {
        "observed_power_w",
        "observed_temperature_c",
    }


def test_issue127_cooldown_proof_rejects_passing_metadata_without_an_in_window_trace_sample():
    environment = runner.sanitize_environment(_environment(observed=[800]))
    runner.validate_harness_environment(environment, run_id="run127-test")
    metadata = runner.build_cooldown_metadata(
        runner.CooldownMetadataParameters(
            run_number=2,
            wait_seconds=3.0,
            expected_aiclk_mhz=1350.0,
            environment=environment,
            start_utc="2026-10-12T00:00:00.500000+00:00",
            end_utc="2026-10-12T00:00:03.500000+00:00",
            wait_performed=True,
            observed_power_w=47.0,
            observed_temperature_c=58.5,
        )
    )
    trace = _cooldown_trace()
    trace["samples"] = [trace["samples"][0], trace["samples"][3]]

    verified = runner.verify_cooldown_power_trace(metadata, power_trace=trace)

    assert verified["power_trace_facts"]["gate_sample"] is None
    assert verified["power_trace_facts"]["gate"]["passes"] is False
    assert verified["power_trace_facts"]["gate"]["result"] == "fail"
    assert verified["gate_verified"] is False
    assert verified["power_trace_verified"] is False
    assert verified["verified"] is False
    assert (
        verified["power_trace_facts"]["gate_observation"]["reason"]
        == "no_usable_trace_sample_within_cooldown_bounds"
    )


@pytest.mark.parametrize("close_failure", ("device", "occurrence"))
def test_issue127_close_failure_cannot_produce_pass_record(close_failure):
    environment = runner.sanitize_environment(_environment())
    rows = []
    cleanup_result = {"status": "ok"}
    if close_failure == "occurrence":
        class RaisingKernel:
            def close(self):
                raise RuntimeError("occurrence close failed")

        runner._close_occurrence_kernel(cleanup_result, RaisingKernel())
        assert cleanup_result == {
            "status": "failed",
            "cleanup_failures": ["RuntimeError"],
        }
    for row in runner.build_occurrence_plan(1):
        measured = dict(row)
        measured.update(
            {
                "status": cleanup_result["status"],
                "raw_p50_tflops": 16.0,
                "adjusted_p50_tflops": 16.0,
                "row_time_weighted_aiclk_mhz": 1350.0,
                "timed_start_utc": "2026-10-12T00:00:02+00:00",
                "aiclk_interval": {
                    "coverage_complete": True,
                    "trace_coverage_complete": True,
                    "first_held_aiclk_mhz": 1350.0,
                },
            }
        )
        if close_failure == "occurrence":
            measured["cleanup_failures"] = cleanup_result["cleanup_failures"]
        rows.append(measured)
    record = runner.build_record(
        run={
            "status": "pass",
            "comparison_rows": rows,
            "warmup": {"end_utc": "2026-10-12T00:00:01+00:00"},
            "run_timeout_seconds": 90.0,
            "run_under_timeout": True,
            "run_timeout_policy": runner.resolve_run_timeout(
                1, 90.0, "confirmation"
            )[1],
            "run_duration_seconds": 20.0,
            "run_start_monotonic": 0.0,
            "run_end_monotonic": 20.0,
            "input_generation": {},
            "device_close_failed": close_failure == "device",
        },
        run_number=1,
        run_id="run127-test",
        environment=environment,
        cooldown=runner.build_cooldown_metadata(
            runner.CooldownMetadataParameters(
                run_number=1,
                wait_seconds=None,
                expected_aiclk_mhz=None,
                environment=environment,
            )
        ),
        run_start_utc="2026-10-12T00:00:00+00:00",
        run_end_utc="2026-10-12T00:01:00+00:00",
        power_trace={
            "file": "snapshot.csv",
            "source_file": "power-run127-test.csv",
            "sha256": "f" * 64,
            "sha256_definition": "SHA-256 of the complete immutable per-run power CSV snapshot bytes",
            "coverage_definition": EXPLICIT_FINAL_SAMPLE,
            "coverage_complete": True,
            "immutable_snapshot": True,
            "timestamp_gap_facts": {"observed_timestamp_gap_seconds": [2.24]},
        },
    )
    assert record["status"] == "failed"
    assert record["device_close_failed"] is (close_failure == "device")
    if close_failure == "occurrence":
        assert record["comparison_rows"][0]["cleanup_failures"] == ["RuntimeError"]
    assert record["run_timing"]["run_timeout_seconds"] == 90.0
    assert record["run_timing"]["under_timeout"] is True
    assert record["run_timeout_policy"]["approval_date"] == "2026-10-16"
    assert record["run_timeout_policy"]["scope"] == "confirmation"


def test_issue127_serialization_replaces_known_path_fields_and_audits_fields():
    serialized = runner._serialize_record(
        {
            "failure": {
                "message": "the expression /abs(response_at_look) is explanatory prose",
            },
            "power_trace": {
                "source_file": "/tmp/private-trace.csv",
            },
            "device_node": "/dev/tenstorrent/1",
        }
    )
    decoded = json.loads(serialized)
    assert decoded["failure"]["message"] == (
        "the expression /abs(response_at_look) is explanatory prose"
    )
    assert decoded["power_trace"]["source_file"] == "private-trace.csv"
    assert decoded["record_safety"]["path_replacements"] == [
        {
            "field": "power_trace.source_file",
            "replacement": "private-trace.csv",
            "reason": "absolute path replaced by harness-relative basename",
        }
    ]
    assert "/tmp/private-trace.csv" not in serialized
    assert "/dev/tenstorrent/1" in serialized


def test_issue127_serialization_rejects_unknown_absolute_path_with_field():
    with pytest.raises(ValueError, match=r"field failure\.location"):
        runner._serialize_record({"failure": {"location": "/tmp/private.json"}})
    with pytest.raises(ValueError, match=r"field metadata\.description"):
        runner._serialize_record(
            {"metadata": {"description": "/home/user/private report.csv"}}
        )


def test_issue127_serialization_rejects_embedded_host_path_with_field():
    with pytest.raises(ValueError, match=r"field failure\.message"):
        runner._serialize_record(
            {
                "failure": {
                    "message": "open failed for '/home/user/private report.csv'",
                }
            }
        )
    with pytest.raises(ValueError, match=r"field failure\.message"):
        runner._serialize_record(
            {"failure": {"message": r"open failed for C:\\Users\\user\\private report.csv"}}
        )


@pytest.mark.parametrize(
    "value",
    (
        "error:/home/user/private.txt",
        r"error:C:\\Users\\alice\\private.txt",
        "/ home/user/private.txt",
        "/||private/user/file.txt",
        "/'private/user/file.txt",
        "~alice/private/file.txt",
    ),
)
def test_issue127_path_classifier_covers_prefixed_and_legal_first_components(value: str):
    environment = runner.sanitize_environment(
        {"metadata": {"diagnostic": value, value: "private"}}
    )

    assert environment["metadata"]["diagnostic"] == runner.REDACTED_PATH
    assert list(environment["metadata"]) == ["diagnostic"]
    assert value not in json.dumps(environment, ensure_ascii=False)

    with pytest.raises(ValueError, match=r"field payload\.nested"):
        runner._serialize_record({"payload": {"nested": [value]}})
    with pytest.raises(ValueError, match=r"field metadata\.<path-key>") as exc_info:
        runner._serialize_record({"metadata": {value: "failure"}})
    assert value not in str(exc_info.value)


@pytest.mark.parametrize(
    ("value", "expected_basename"),
    (
        ("error:/home/user/private.txt", "private.txt"),
        (r"error:C:\\Users\\alice\\private.txt", "private.txt"),
        ("/ home/user/private.txt", "private.txt"),
        ("/||private/user/file.txt", "file.txt"),
        ("/'private/user/file.txt", "file.txt"),
        ("~alice/private/file.txt", "file.txt"),
    ),
)
def test_issue127_known_path_fields_sanitize_new_path_forms(
    value: str, expected_basename: str
):
    serialized = runner._serialize_record({"nested": {"source_file": value}})
    decoded = json.loads(serialized)

    assert value not in serialized
    assert decoded["nested"]["source_file"] == expected_basename
    assert decoded["record_safety"]["path_replacements"][0]["field"] == (
        "nested.source_file"
    )


@pytest.mark.parametrize(
    "definition",
    (
        (
            "||response_candidate - response_reference||_F / "
            "||response_reference||_F; complex phase and response scale are retained"
        ),
        "FP32-R raw P50 / BF16-R raw P50 - 1",
    ),
)
def test_issue127_serialization_preserves_ordinary_quotient_definitions(definition: str):
    serialized = runner._serialize_record({"correctness": {"definition": definition}})
    decoded = json.loads(serialized)

    assert decoded["correctness"]["definition"] == definition
    environment = runner.sanitize_environment({"definition": definition})
    assert environment["definition"] == definition


def test_issue127_full_timing_payload_serializes_raw_p50_definition_byte_identically():
    timing = runner._timing_summary([0.001, 0.002], 1234)

    serialized = runner._serialize_record(timing)
    decoded = json.loads(serialized)

    assert decoded == timing
    assert decoded["raw_p50_definition"].encode("utf-8") == timing[
        "raw_p50_definition"
    ].encode("utf-8")


def test_issue127_issue88_phase_aligned_definition_serializes_byte_identically():
    definition = issue88.BEAM_RESPONSE_METRIC_DEFINITIONS[
        "phase_aligned_complex_response_relative_frobenius_error"
    ]

    serialized = runner._serialize_record(
        {"beam_pattern": {"metric_definitions": dict(issue88.BEAM_RESPONSE_METRIC_DEFINITIONS)}}
    )
    decoded = json.loads(serialized)
    retained = decoded["beam_pattern"]["metric_definitions"][
        "phase_aligned_complex_response_relative_frobenius_error"
    ]

    assert retained.encode("utf-8") == definition.encode("utf-8")


@pytest.mark.parametrize(
    "value",
    (
        "/srv/private report/秘密.csv",
        "open failed for '/opt/équipe data/private report.csv'",
        "\\\\server\\share\\private file",
        "open failed for \\\\server\\share\\private file",
    ),
)
def test_issue127_environment_sanitizes_arbitrary_absolute_and_unc_paths(value: str):
    environment = runner.sanitize_environment({"diagnostic": value})

    assert environment.get("diagnostic") in (None, "<redacted-path>")
    assert value not in json.dumps(environment, ensure_ascii=False)


@pytest.mark.parametrize(
    "value",
    (
        "/srv/private report/秘密.csv",
        "open failed for '/opt/équipe data/private report.csv'",
        "\\\\server\\share\\private file",
        "open failed for \\\\server\\share\\private file",
    ),
)
def test_issue127_serialization_rejects_arbitrary_absolute_and_unc_paths(value: str):
    with pytest.raises(ValueError, match=r"field payload\.value"):
        runner._serialize_record({"payload": {"value": value}})


def test_issue127_known_path_field_keeps_basename_audit_for_unc_path():
    serialized = runner._serialize_record(
        {"power_trace": {"source_file": "\\\\server\\share\\private-file.csv"}}
    )
    decoded = json.loads(serialized)

    assert decoded["power_trace"]["source_file"] == "private-file.csv"
    assert decoded["record_safety"]["path_replacements"] == [
        {
            "field": "power_trace.source_file",
            "replacement": "private-file.csv",
            "reason": "absolute path replaced by harness-relative basename",
        }
    ]


@pytest.mark.parametrize(
    "value",
    (
        "/home/user/private(1).csv",
        "/home/user/a[b].txt",
        "/home/user/a{b}.txt",
        r"C:\Users\user\private(1).csv",
        r"C:/Users/user/private(1).csv",
        r"\\server\share\private(1).csv",
        "open failed for '/home/user/private(1).csv'",
        "open failed for /home/user/private(1).csv",
        "~/private(1).txt",
        "file:///home/user/private.txt",
        "file:/home/user/private.txt",
        "path:/home/user/private.txt",
        "/home/|user/private.txt",
    ),
)
def test_issue127_adversarial_paths_are_redacted_from_environment(value: str):
    environment = runner.sanitize_environment({"diagnostic": value})

    assert environment["diagnostic"] == runner.REDACTED_PATH
    assert value not in json.dumps(environment, ensure_ascii=False)


@pytest.mark.parametrize(
    "value",
    (
        "/home/user/private(1).csv",
        "/home/user/a[b].txt",
        "/home/user/a{b}.txt",
        r"C:\Users\user\private(1).csv",
        r"C:/Users/user/private(1).csv",
        r"\\server\share\private(1).csv",
        "open failed for '/home/user/private(1).csv'",
        "open failed for /home/user/private(1).csv",
        "~/private(1).txt",
        "file:///home/user/private.txt",
        "file:/home/user/private.txt",
        "path:/home/user/private.txt",
        "/home/|user/private.txt",
    ),
)
def test_issue127_adversarial_paths_are_rejected_from_unknown_record_fields(value: str):
    with pytest.raises(ValueError, match=r"field payload\.value"):
        runner._serialize_record({"payload": {"value": value}})


@pytest.mark.parametrize(
    "value",
    (
        "/home/user/private(1).csv",
        r"C:\Users\user\private(1).csv",
        r"\\server\share\private(1).csv",
        "file:///home/user/private.txt",
        "file:/home/user/private.txt",
        "path:/home/user/private.txt",
        "/home/|user/private.txt",
    ),
)
def test_issue127_adversarial_paths_in_known_fields_keep_only_safe_audit_values(value: str):
    serialized = runner._serialize_record({"artifact_path": value})

    assert value not in serialized
    decoded = json.loads(serialized)
    assert decoded["artifact_path"] in ("private.txt", runner.REDACTED_PATH)
    assert decoded["record_safety"]["path_replacements"][0]["field"] == "artifact_path"


@pytest.mark.parametrize(
    "private_path",
    (
        "/home/user/private.txt",
        r"C:\Users\user\private(1).csv",
        r"\\server\share\private(1).csv",
        "file:/home/user/private.txt",
        "/home/|user/private.txt",
    ),
)
def test_issue127_environment_omits_path_shaped_mapping_keys(private_path: str):
    environment = runner.sanitize_environment(
        {"metadata": {private_path: "failure", "status": "ok"}}
    )

    assert environment == {"metadata": {"status": "ok"}}
    assert private_path not in json.dumps(environment, ensure_ascii=False)


@pytest.mark.parametrize(
    "private_path",
    (
        "/home/user/private.txt",
        r"C:\Users\user\private(1).csv",
        r"\\server\share\private(1).csv",
        "file:/home/user/private.txt",
        "/home/|user/private.txt",
    ),
)
def test_issue127_record_rejects_path_shaped_mapping_keys_without_echoing_them(
    private_path: str,
):
    payload = {"metadata": {private_path: "failure"}}

    with pytest.raises(ValueError, match=r"field metadata\.<path-key>") as exc_info:
        runner._serialize_record(payload)
    assert private_path not in str(exc_info.value)
    with pytest.raises(ValueError, match=r"field metadata\.<path-key>"):
        runner._validate_record_safety(payload)


@pytest.mark.parametrize(
    "formula",
    (
        "I/||R||inf",
        "... / ||response_reference||_F",
        "FP32-R raw P50 / BF16-R raw P50 - 1",
        "/abs(response_at_look)",
    ),
)
def test_issue127_exact_formula_controls_remain_serializable(formula: str):
    for field in ("formula", "path"):
        serialized = runner._serialize_record({field: formula})
        environment = runner.sanitize_environment({field: formula})

        assert json.loads(serialized)[field] == formula
        assert environment[field] == formula


def test_issue127_formula_exception_does_not_hide_a_later_host_path():
    value = "/abs(response_at_look) is explanatory prose /home/user/private.txt"

    environment = runner.sanitize_environment({"diagnostic": value})
    assert environment["diagnostic"] == runner.REDACTED_PATH
    with pytest.raises(ValueError, match=r"field diagnostic"):
        runner._serialize_record({"diagnostic": value})


@pytest.mark.parametrize(
    "device_node",
    ("/dev/tenstorrent/0", "/dev/tenstorrent/by-id/p150a-0"),
)
def test_issue127_allowed_device_nodes_remain_serializable(device_node: str):
    serialized = runner._serialize_record({"device_node": device_node})
    environment = runner.sanitize_environment({"device_node": device_node})

    assert json.loads(serialized)["device_node"] == device_node
    assert environment["device_node"] == device_node


@pytest.mark.parametrize(
    ("board_id", "device_node"),
    (
        ("１２３４５６７８", "/dev/tenstorrent/１２３"),
        ("board-p150a-é", "/dev/tenstorrent/by-id/p150a-é"),
    ),
)
def test_issue127_board_and_device_identifiers_remain_ascii_only(
    board_id: str, device_node: str
):
    assert runner._BOARD_ID_RE.fullmatch(board_id) is None
    assert runner._DEVICE_NODE_RE.fullmatch(device_node) is None


def test_issue127_non_ascii_formula_like_token_remains_a_path_candidate():
    value = "/response_é(value)"

    assert runner._FORMULA_SLASH_RE.fullmatch(value) is None
    assert runner._classify_path(value) == runner._PATH_KIND_WHOLE


def test_issue127_environment_path_lists_are_placeholdered_without_formula_false_positive():
    environment = runner.sanitize_environment(
        {
            "pythonpath": "/home/user/private path:/work/tree",
            "path": "/usr/bin:/usr/local/bin",
            "definition": "/abs(response_at_look)",
        }
    )
    assert environment["pythonpath"] == "<redacted-path>"
    assert environment["path"] == "<redacted-path>"
    assert environment["definition"] == "/abs(response_at_look)"
    assert "/home/user" not in json.dumps(environment)


def test_issue127_formula_is_byte_identical_to_issue88_and_slash_is_not_a_path():
    issue88_record = json.loads(
        (ROOT / "docs/measurements/2026-10-10-p150a-newton-schulz-issue88-fp32-r-final-runner-retake.json").read_text(
            encoding="utf-8"
        )
    )
    expected = (
        "20*log10(max(abs(response)/abs(response_at_look), 10**(floor_db/20)))"
    )
    definitions = []

    def collect(value):
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and expected in value:
            definitions.append(value)

    collect(issue88_record)
    assert definitions
    reference_definition = definitions[0]
    assert all(definition == reference_definition for definition in definitions)
    assert reference_definition.split(";", 1)[0] == expected
    serialized = runner._serialize_record(
        {"correctness": {"definition": reference_definition, "text": "/abs("}}
    )
    decoded = json.loads(serialized)
    assert decoded["correctness"]["definition"].encode("utf-8") == reference_definition.encode(
        "utf-8"
    )
    assert decoded["correctness"]["text"] == "/abs("


def test_issue127_environment_is_complete_but_does_not_retain_host_identity_or_worktree():
    environment = runner.sanitize_environment(_environment())
    runner.validate_harness_environment(environment, run_id="run127-test")

    assert environment["board"]["serial"] == "0000040331914124"
    assert environment["firmware"]["fw_bundle_version"] == "19.6.0.0"
    assert environment["kernel_driver_version"] == "2.11.0"
    assert environment["image_digest"].startswith("sha256:")
    assert "hostname" not in environment
    assert "worktree_path" not in environment
    assert environment["device_node"] == "/dev/tenstorrent/0"
    with pytest.raises(ValueError, match="path"):
        runner._serialize_record({"bad": "/home/user/private-worktree"})


def test_issue127_sampler_decision_is_default_and_not_shortened():
    runner.validate_sampler_decision(runner.SAMPLER_DECISION, trace_timestamp_gaps=[2.24])
    assert runner.SAMPLER_DECISION["requested_interval_seconds"] == 2.0
    assert runner.SAMPLER_DECISION["shortened_interval"] is False
    with pytest.raises(ValueError, match="2.0"):
        runner.validate_sampler_decision(
            {**runner.SAMPLER_DECISION, "requested_interval_seconds": 0.5},
            trace_timestamp_gaps=[0.5],
        )


def test_issue127_prediction_plan_pins_orders_sampler_and_thresholds():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["runner"] == runner.ISSUE127_RUNNER
    assert tuple(plan["run_orders"]["run_1"]) == runner.RUN_1_ORDER
    assert tuple(plan["run_orders"]["run_2"]) == runner.RUN_2_ORDER
    assert plan["sampler_decision"]["requested_interval_seconds"] == 2.0
    assert plan["sampler_decision"]["shortened_interval"] is False
    assert {
        row["name"]: row["launches_per_occurrence"]
        for row in plan["comparison_configurations"]
    } == {
        "bf16-r-L16": 11000,
        "fp32-r-L16": 11000,
        "bf16-r-L32-r-dram": 5800,
        "fp32-r-L32-r-dram": 5800,
    }
    assert plan["timing_protocol"]["startup_warmup"]["launches"] == 4500
    assert plan["timing_protocol"]["row_duration_target_seconds"] == [2.24, None]
    assert plan["timing_protocol"]["power_samples_per_row_target"] == 2
    assert plan["timing_protocol"]["run_timeout_seconds"] == 60.0
    assert plan["timing_protocol"]["approved_run_timeout"] == {
        "run_timeout_seconds": 90.0,
        "docker_timeout_seconds": 90.0,
        "approval_date": "2026-10-16",
        "approval": "owner-approved exception",
        "targets": [
            {"scope": "confirmation", "run_number": 1},
            {"scope": "measurement-1", "run_number": 1},
            {"scope": "measurement-2", "run_number": 2},
            {"scope": "measurement-2-rerun", "run_number": 2},
        ],
        "required_cli": "--run-scope <approved target> --run-timeout-seconds 90",
        "default_unchanged_seconds": 60.0,
    }
    assert plan["revision_history"][-1]["selected_launches_per_occurrence"] == {
        "L16": 11000,
        "L32": 5800,
    }
    assert plan["revision_history"][-1]["cooldown_correction"] == {
        "prior_run_failure": "The prior measurement-2 cooldown trace remained at idle 800 MHz, so the 1350-MHz success condition could not pass.",
        "baseline_source": "measurement-1 pre-run power trace",
        "baseline_power_w": 47.0,
        "baseline_temperature_c": 58.5,
        "power_tolerance_w": 5.0,
        "temperature_tolerance_c": 3.0,
        "power_limit_w": 52.0,
        "temperature_limit_c": 61.5,
        "wait_cap_seconds": 600.0,
        "observed_idle_aiclk_mhz": 800.0,
        "aiclk_gate": False,
        "prediction_and_decision_criteria_changed": False,
    }
    assert plan["predicted_aiclk_ranges_mhz"]["basis"].endswith("predictions, not measured results.")
    for run_number, key in ((1, "run_1"), (2, "run_2")):
        planned = runner.build_occurrence_plan(run_number)
        expected = runner.PREDICTED_AICLK_RANGES_MHZ[run_number]
        assert [item["range_mhz"] for item in plan["predicted_aiclk_ranges_mhz"][key]] == [
            list(value) for value in expected
        ]
        assert [row["predicted_aiclk_range_mhz"] for row in planned] == [
            list(value) for value in expected
        ]
    assert plan["decision_criteria"]["r_format_hypothesis"]["thresholds_percent"] == {
        "L16_mean_at_most": -0.847,
        "L32_DRAM_mean_at_most": -2.341,
    }
    assert plan["decision_criteria"]["adjusted_p50_is_primary"] is False


def _read_issue127_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_issue127_power_samples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(CSV_COLUMNS)
        return list(reader)


def _load_issue127_summary_inputs():
    plan = _read_issue127_json(PLAN)
    records = {
        scope: _read_issue127_json(path) for scope, path in RUN_RECORDS.items()
    }
    summary = _read_issue127_json(SUMMARY)
    power_samples = {
        scope: _read_issue127_power_samples(path)
        for scope, path in POWER_TRACES.items()
    }
    trace_hashes = {scope: _sha256(path) for scope, path in POWER_TRACES.items()}
    return plan, records, summary, power_samples, trace_hashes


def _plan_pair_names(plan: dict, *, size: int, r_memory: str) -> tuple[str, str]:
    configurations = [
        configuration
        for configuration in plan["comparison_configurations"]
        if configuration["matrix_size"] == size
        and configuration["r_memory"] == r_memory
    ]
    return (
        next(configuration["name"] for configuration in configurations if configuration["variant"] == "bf16"),
        next(configuration["name"] for configuration in configurations if configuration["variant"] == "fp32-r"),
    )


def _derived_issue127_row(row: dict, power_samples: list[dict[str, str]]) -> dict:
    facts = runner.left_held_aiclk_interval_facts(
        power_samples,
        start=row["timed_start_utc"],
        end=row["timed_end_utc"],
        row_label=row["occurrence_id"],
    )
    assert facts["available"] is True
    assert facts["coverage_complete"] is True
    time_weighted_aiclk = facts["time_weighted_aiclk_mhz"]
    adjusted_p50 = runner.adjusted_p50_tflops(
        row["raw_p50_tflops"], time_weighted_aiclk
    )
    assert adjusted_p50 is not None
    started_samples = runner._power_samples_started_in_interval(
        power_samples,
        start=row["timed_start_utc"],
        end=row["timed_end_utc"],
    )
    return {
        "source": row,
        "facts": facts,
        "raw_p50": float(row["raw_p50_tflops"]),
        "adjusted_p50": float(adjusted_p50),
        "time_weighted_aiclk": float(time_weighted_aiclk),
        "aiclk_min": float(facts["aiclk_mhz_min"]),
        "aiclk_max": float(facts["aiclk_mhz_max"]),
        "power_sample_count": len(facts["segments"]),
        "started_samples": started_samples,
    }


def _derived_issue127_rows(
    record: dict, power_samples: list[dict[str, str]]
) -> list[dict]:
    return [
        _derived_issue127_row(row, power_samples)
        for row in record["comparison_rows"]
    ]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _raw_delta(left: float, right: float) -> float:
    return 100.0 * (right / left - 1.0)


def _assert_approx_fields(actual: dict, expected: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        assert actual[field] == pytest.approx(expected[field])


def _assert_approx_list(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert actual_value == pytest.approx(expected_value)


def test_issue127_abba_summary_recomputes_decision_and_sensitivity():
    plan, records, summary, power_samples, trace_hashes = _load_issue127_summary_inputs()
    plan_criteria = plan["decision_criteria"]

    assert summary["status"] == "pass"
    assert summary["plan"]["file"] == PLAN.name
    assert summary["plan"]["sha256"] == _sha256(PLAN)
    assert summary["plan"]["raw_p50_definition"] == plan_criteria["delta_definition"]
    assert summary["plan"]["adjusted_p50_is_primary"] is plan_criteria[
        "adjusted_p50_is_primary"
    ]
    assert summary["plan"]["prediction_and_decision_criteria_changed"] is plan[
        "revision_history"
    ][-1]["cooldown_correction"]["prediction_and_decision_criteria_changed"]

    expected_source_refs = []
    summary_runs = {run["scope"]: run for run in summary["runs"]}
    assert tuple(summary_runs) == ACCEPTED_SCOPES
    derived_rows = {}
    for scope in ACCEPTED_SCOPES:
        record_path = RUN_RECORDS[scope]
        record = records[scope]
        trace_hash = trace_hashes[scope]
        expected_source_refs.append(
            {
                "file": record_path.name,
                "sha256": _sha256(record_path),
                "run_number": record["run_number"],
                "scope": scope,
            }
        )
        assert record["status"] == "pass"
        assert record["power_trace"]["sha256"] == trace_hash
        assert record["power_trace_provenance"]["sha256"] == trace_hash
        assert record["power_trace"]["immutable_snapshot"] is True

        summary_run = summary_runs[scope]
        assert summary_run["run_number"] == record["run_number"]
        assert summary_run["status"] == record["status"]
        assert summary_run["record"] == {
            "file": record_path.name,
            "sha256": _sha256(record_path),
        }
        assert summary_run["order"] == record["order"]["names"]

        derived = _derived_issue127_rows(record, power_samples[scope])
        derived_rows[scope] = derived
        assert len(summary_run["rows"]) == len(derived) == 8
        for source_row, expected, summary_row in zip(
            record["comparison_rows"], derived, summary_run["rows"], strict=True
        ):
            for field in (
                "order_index",
                "row",
                "occurrence",
                "occurrence_id",
                "timed_start_utc",
                "timed_end_utc",
                "timed_duration_seconds",
                "launches_requested",
                "launches_measured",
            ):
                assert summary_row[field] == source_row[field]
            assert source_row["aiclk_interval"]["trace_sha256"] == trace_hash
            assert source_row["row_time_weighted_aiclk_mhz"] == pytest.approx(
                expected["time_weighted_aiclk"]
            )
            assert source_row["adjusted_p50_tflops"] == pytest.approx(
                expected["adjusted_p50"]
            )
            assert source_row["aiclk_interval"]["power_sample_count_in_interval"] == (
                expected["power_sample_count"]
            )
            assert summary_row["raw_p50_tflops"] == pytest.approx(expected["raw_p50"])
            assert summary_row["adjusted_p50_tflops"] == pytest.approx(
                expected["adjusted_p50"]
            )
            assert summary_row["aiclk_mhz_min"] == pytest.approx(expected["aiclk_min"])
            assert summary_row["aiclk_mhz_max"] == pytest.approx(expected["aiclk_max"])
            assert summary_row["row_time_weighted_aiclk_mhz"] == pytest.approx(
                expected["time_weighted_aiclk"]
            )
            assert summary_row["power_sample_count_in_interval"] == expected[
                "power_sample_count"
            ]
            assert summary_row["power_samples_started_in_interval"] == expected[
                "started_samples"
            ]

        for name in (
            configuration["name"]
            for configuration in plan["comparison_configurations"]
        ):
            source_summary = next(
                item for item in record["comparison_summary"] if item["row"] == name
            )
            summary_item = next(
                item for item in summary_run["comparison_summary"] if item["row"] == name
            )
            rows_for_name = [item for item in derived if item["source"]["row"] == name]
            assert summary_item["occurrence_indices"] == [
                item["source"]["order_index"] for item in rows_for_name
            ]
            assert summary_item["occurrence_count"] == len(rows_for_name)
            assert summary_item["aggregation"] == source_summary["aggregation"]
            assert summary_item["raw_p50_tflops"] == pytest.approx(
                _mean([item["raw_p50"] for item in rows_for_name])
            )
            assert summary_item["adjusted_p50_tflops"] == pytest.approx(
                _mean([item["adjusted_p50"] for item in rows_for_name])
            )
            assert summary_item["row_time_weighted_aiclk_mhz"] == pytest.approx(
                _mean([item["time_weighted_aiclk"] for item in rows_for_name])
            )

    assert summary["source_records"] == expected_source_refs

    pair_specs = (
        ("L16_L1", 16, "l1", "L16_mean_at_most", "L16"),
        ("L32_DRAM", 32, "dram", "L32_DRAM_mean_at_most", "L32_DRAM"),
    )
    derived_by_scope = {
        scope: {
            (item["source"]["row"], item["source"]["occurrence"]): item
            for item in rows
        }
        for scope, rows in derived_rows.items()
    }
    expected_r_format_passes = []
    expected_clock_passes = []
    for pair_key, size, r_memory, threshold_key, clock_key in pair_specs:
        left_name, right_name = _plan_pair_names(
            plan, size=size, r_memory=r_memory
        )
        comparison = summary["comparisons"][pair_key]
        assert comparison["pair"] == [left_name, right_name]
        expected_per_run = []
        for scope in ACCEPTED_SCOPES:
            run_number = records[scope]["run_number"]
            occurrence_deltas = []
            left_rows = []
            right_rows = []
            for occurrence in (1, 2):
                left = derived_by_scope[scope][(left_name, occurrence)]
                right = derived_by_scope[scope][(right_name, occurrence)]
                raw_delta = _raw_delta(left["raw_p50"], right["raw_p50"])
                adjusted_delta = _raw_delta(
                    left["adjusted_p50"], right["adjusted_p50"]
                )
                left_rows.append(left)
                right_rows.append(right)
                occurrence_deltas.append(
                    {
                        "occurrence": occurrence,
                        "raw_p50_delta_percent": raw_delta,
                        "adjusted_p50_delta_percent": adjusted_delta,
                        "bf16_r_raw_p50_tflops": left["raw_p50"],
                        "fp32_r_raw_p50_tflops": right["raw_p50"],
                        "bf16_r_row_time_weighted_aiclk_mhz": left[
                            "time_weighted_aiclk"
                        ],
                        "fp32_r_row_time_weighted_aiclk_mhz": right[
                            "time_weighted_aiclk"
                        ],
                    }
                )
            left_raw_mean = _mean([row["raw_p50"] for row in left_rows])
            right_raw_mean = _mean([row["raw_p50"] for row in right_rows])
            left_adjusted_mean = _mean([row["adjusted_p50"] for row in left_rows])
            right_adjusted_mean = _mean(
                [row["adjusted_p50"] for row in right_rows]
            )
            left_aiclk_mean = _mean(
                [row["time_weighted_aiclk"] for row in left_rows]
            )
            right_aiclk_mean = _mean(
                [row["time_weighted_aiclk"] for row in right_rows]
            )
            expected_per_run.append(
                {
                    "run_number": run_number,
                    "scope": scope,
                    "bf16_r_raw_p50_tflops": left_raw_mean,
                    "fp32_r_raw_p50_tflops": right_raw_mean,
                    "raw_p50_delta_percent": _raw_delta(left_raw_mean, right_raw_mean),
                    "bf16_r_adjusted_p50_tflops": left_adjusted_mean,
                    "fp32_r_adjusted_p50_tflops": right_adjusted_mean,
                    "adjusted_p50_delta_percent": _raw_delta(
                        left_adjusted_mean, right_adjusted_mean
                    ),
                    "bf16_r_mean_row_time_weighted_aiclk_mhz": left_aiclk_mean,
                    "fp32_r_mean_row_time_weighted_aiclk_mhz": right_aiclk_mean,
                    "occurrence_deltas": occurrence_deltas,
                }
            )

        assert len(comparison["per_run"]) == len(expected_per_run) == 2
        for actual, expected in zip(
            comparison["per_run"], expected_per_run, strict=True
        ):
            assert actual["run_number"] == expected["run_number"]
            assert actual["scope"] == expected["scope"]
            _assert_approx_fields(
                actual,
                expected,
                (
                    "bf16_r_raw_p50_tflops",
                    "fp32_r_raw_p50_tflops",
                    "raw_p50_delta_percent",
                    "bf16_r_adjusted_p50_tflops",
                    "fp32_r_adjusted_p50_tflops",
                    "adjusted_p50_delta_percent",
                    "bf16_r_mean_row_time_weighted_aiclk_mhz",
                    "fp32_r_mean_row_time_weighted_aiclk_mhz",
                ),
            )
            assert len(actual["occurrence_deltas"]) == 2
            for actual_occurrence, expected_occurrence in zip(
                actual["occurrence_deltas"],
                expected["occurrence_deltas"],
                strict=True,
            ):
                assert actual_occurrence["occurrence"] == expected_occurrence[
                    "occurrence"
                ]
                _assert_approx_fields(
                    actual_occurrence,
                    expected_occurrence,
                    (
                        "raw_p50_delta_percent",
                        "adjusted_p50_delta_percent",
                        "bf16_r_raw_p50_tflops",
                        "fp32_r_raw_p50_tflops",
                        "bf16_r_row_time_weighted_aiclk_mhz",
                        "fp32_r_row_time_weighted_aiclk_mhz",
                    ),
                )

        raw_deltas = [item["raw_p50_delta_percent"] for item in expected_per_run]
        adjusted_deltas = [
            item["adjusted_p50_delta_percent"] for item in expected_per_run
        ]
        raw_summary = comparison["raw_p50_delta_percent"]
        adjusted_summary = comparison["adjusted_p50_delta_percent_supplementary"]
        assert raw_summary["run_1"] == pytest.approx(raw_deltas[0])
        assert raw_summary["run_2"] == pytest.approx(raw_deltas[1])
        assert raw_summary["mean_of_two_runs"] == pytest.approx(_mean(raw_deltas))
        assert adjusted_summary["run_1"] == pytest.approx(adjusted_deltas[0])
        assert adjusted_summary["run_2"] == pytest.approx(adjusted_deltas[1])
        assert adjusted_summary["mean_of_two_runs"] == pytest.approx(
            _mean(adjusted_deltas)
        )

        threshold = plan_criteria["r_format_hypothesis"]["thresholds_percent"][
            threshold_key
        ]
        criterion = comparison["r_format_criterion"]
        both_negative = all(delta < 0.0 for delta in raw_deltas)
        mean_at_most = _mean(raw_deltas) <= threshold
        assert criterion["threshold_mean_at_most_percent"] == pytest.approx(threshold)
        assert criterion["both_run_deltas_negative"] is both_negative
        assert criterion["mean_delta_at_most_threshold"] is mean_at_most
        assert criterion["passes"] is (both_negative and mean_at_most)
        expected_r_format_passes.append(criterion["passes"])

        half_effect = plan_criteria["clock_hypothesis"][
            "half_effect_thresholds_percent"
        ][clock_key]
        clock_criterion = comparison["clock_criterion"]
        sign_reversal = min(raw_deltas) < 0.0 < max(raw_deltas)
        mean_absolute = abs(_mean(raw_deltas))
        mean_below_half = mean_absolute < half_effect
        assert clock_criterion["half_effect_threshold_percent"] == pytest.approx(
            half_effect
        )
        assert clock_criterion["sign_reversal"] is sign_reversal
        assert clock_criterion["mean_absolute_delta_percent"] == pytest.approx(
            mean_absolute
        )
        assert clock_criterion["mean_absolute_below_half_effect"] is mean_below_half
        if not (sign_reversal or mean_below_half):
            assert clock_criterion["passes"] is False
            assert clock_criterion["position_effect_evaluation"] == (
                "not required because the sign-reversal/low-delta gate is false"
            )
        expected_clock_passes.append(clock_criterion["passes"])

    assert summary["decision"]["r_format_criterion_passes"] is all(
        expected_r_format_passes
    )
    assert summary["decision"]["clock_criterion_passes"] is all(expected_clock_passes)
    assert summary["decision"]["conclusion"] == (
        "r_format" if all(expected_r_format_passes) else "inconclusive"
    )

    expected_sensitivity = {pair_key: [] for pair_key, *_ in pair_specs}
    for pair_key, size, r_memory, _threshold_key, _clock_key in pair_specs:
        left_name, right_name = _plan_pair_names(
            plan, size=size, r_memory=r_memory
        )
        for scope in ACCEPTED_SCOPES:
            run_number = records[scope]["run_number"]
            for occurrence in (1, 2):
                left = derived_by_scope[scope][(left_name, occurrence)]
                right = derived_by_scope[scope][(right_name, occurrence)]
                left_range = [left["aiclk_min"], left["aiclk_max"]]
                right_range = [right["aiclk_min"], right["aiclk_max"]]
                overlap = [
                    max(left_range[0], right_range[0]),
                    min(left_range[1], right_range[1]),
                ]
                if overlap[0] > overlap[1]:
                    overlap = None
                raw_delta = _raw_delta(left["raw_p50"], right["raw_p50"])
                adjusted_delta = _raw_delta(
                    left["adjusted_p50"], right["adjusted_p50"]
                )
                interval = [min(raw_delta, adjusted_delta), max(raw_delta, adjusted_delta)]
                expected_sensitivity[pair_key].append(
                    {
                        "run_number": run_number,
                        "occurrence": occurrence,
                        "left_row": left_name,
                        "right_row": right_name,
                        "left_range_mhz": left_range,
                        "right_range_mhz": right_range,
                        "same_range": left_range == right_range,
                        "descriptive_range_overlap_mhz": overlap,
                        "range_overlap_is_effect_bound": False,
                        "left_raw_p50_tflops": left["raw_p50"],
                        "right_raw_p50_tflops": right["raw_p50"],
                        "left_adjusted_p50_tflops": left["adjusted_p50"],
                        "right_adjusted_p50_tflops": right["adjusted_p50"],
                        "raw_p50_delta_percent": raw_delta,
                        "adjusted_p50_delta_percent": adjusted_delta,
                        "left_time_weighted_aiclk_mhz": left["time_weighted_aiclk"],
                        "right_time_weighted_aiclk_mhz": right["time_weighted_aiclk"],
                        "sensitivity_interval_percent": interval,
                        "crosses_zero": interval[0] <= 0.0 <= interval[1],
                    }
                )

    all_sensitivity = []
    for pair_key, expected_items in expected_sensitivity.items():
        assert len(expected_items) == 4
        alignment = summary["aiclk_alignment"][pair_key]
        actual_items = alignment["matched_occurrences"]
        assert len(actual_items) == len(expected_items)
        for actual, expected in zip(actual_items, expected_items, strict=True):
            for field in (
                "run_number",
                "occurrence",
                "left_row",
                "right_row",
                "same_range",
                "range_overlap_is_effect_bound",
                "crosses_zero",
            ):
                assert actual[field] == expected[field]
            for field in (
                "left_range_mhz",
                "right_range_mhz",
                "sensitivity_interval_percent",
            ):
                _assert_approx_list(actual[field], expected[field])
            if expected["descriptive_range_overlap_mhz"] is None:
                assert actual["descriptive_range_overlap_mhz"] is None
            else:
                _assert_approx_list(
                    actual["descriptive_range_overlap_mhz"],
                    expected["descriptive_range_overlap_mhz"],
                )
            _assert_approx_fields(
                actual,
                expected,
                (
                    "left_raw_p50_tflops",
                    "right_raw_p50_tflops",
                    "left_adjusted_p50_tflops",
                    "right_adjusted_p50_tflops",
                    "raw_p50_delta_percent",
                    "adjusted_p50_delta_percent",
                    "left_time_weighted_aiclk_mhz",
                    "right_time_weighted_aiclk_mhz",
                ),
            )
        all_sensitivity.extend(expected_items)
        all_negative = all(
            item["sensitivity_interval_percent"][1] < 0.0
            for item in expected_items
        )
        any_crosses_zero = any(item["crosses_zero"] for item in expected_items)
        assert alignment["same_range_for_all_matched_occurrences"] is all(
            item["same_range"] for item in expected_items
        )
        assert alignment["all_sensitivity_intervals_negative"] is all_negative
        assert alignment["range_overlap_is_effect_bound"] is False
        assert alignment["any_crosses_zero"] is any_crosses_zero

    assert len(all_sensitivity) == 8
    assert summary["aiclk_alignment"]["all_eight_sensitivity_intervals_negative"] is all(
        item["sensitivity_interval_percent"][1] < 0.0 for item in all_sensitivity
    )
    assert summary["aiclk_alignment"]["any_crosses_zero"] is any(
        item["crosses_zero"] for item in all_sensitivity
    )
    sensitivity_decision = summary["decision"]["sensitivity_intervals_under_assumptions"]
    assert sensitivity_decision["matched_occurrence_count"] == len(all_sensitivity)
    assert sensitivity_decision["all_negative"] is all(
        item["sensitivity_interval_percent"][1] < 0.0 for item in all_sensitivity
    )
    assert sensitivity_decision["any_crosses_zero"] is any(
        item["crosses_zero"] for item in all_sensitivity
    )


def test_issue127_abba_summary_recomputes_all_32_sampling_audit_rows():
    plan, records, summary, power_samples, trace_hashes = _load_issue127_summary_inputs()
    audit = summary["sampling_audit"]
    audit_rows = audit["comparison_rows"]
    expected_count = sum(
        len(plan["run_orders"]["run_1" if records[scope]["run_number"] == 1 else "run_2"])
        for scope in RUN_RECORDS
    )
    assert expected_count == 32
    assert audit["comparison_row_count"] == len(audit_rows) == expected_count
    assert summary["excluded_records"] == [
        RUN_RECORDS["confirmation"].name,
        RUN_RECORDS["measurement-2"].name,
    ]

    trace_names = {path.name for path in POWER_TRACES.values()}
    assert {row["power_trace"] for row in audit_rows} == trace_names
    rows_by_scope = {
        scope: [row for row in audit_rows if row["scope"] == scope]
        for scope in RUN_RECORDS
    }
    started_counts = []
    for scope, record_path in RUN_RECORDS.items():
        record = records[scope]
        trace_hash = trace_hashes[scope]
        trace_name = POWER_TRACES[scope].name
        scope_rows = rows_by_scope[scope]
        run_order_key = "run_1" if record["run_number"] == 1 else "run_2"
        assert len(scope_rows) == len(plan["run_orders"][run_order_key]) == 8
        assert record["power_trace"]["sha256"] == trace_hash
        assert record["power_trace_provenance"]["sha256"] == trace_hash
        for source_row, audit_row in zip(
            record["comparison_rows"], scope_rows, strict=True
        ):
            assert audit_row["record"] == record_path.name
            assert audit_row["power_trace"] == trace_name
            assert audit_row["power_trace_sha256"] == trace_hash
            assert audit_row["run_number"] == record["run_number"]
            assert audit_row["scope"] == scope
            assert audit_row["status"] == record["status"]
            for field in (
                "order_index",
                "row",
                "occurrence",
                "occurrence_id",
                "timed_start_utc",
                "timed_end_utc",
            ):
                assert audit_row[field] == source_row[field]
            expected_started = runner._power_samples_started_in_interval(
                power_samples[scope],
                start=source_row["timed_start_utc"],
                end=source_row["timed_end_utc"],
            )
            assert audit_row["power_samples_started_in_interval"] == expected_started
            started_counts.append(expected_started)

    target = plan["timing_protocol"]["power_samples_per_row_target"]
    assert target == runner.POWER_SAMPLES_PER_ROW_TARGET == 2
    assert audit["power_sample_count_in_interval"]["field"] == (
        "power_sample_count_in_interval"
    )
    assert audit["power_sample_count_in_interval"]["definition"] == (
        runner.POWER_SAMPLE_COUNT_IN_INTERVAL_DEFINITION
    )
    assert audit["power_samples_started_in_interval"]["field"] == (
        "power_samples_started_in_interval"
    )
    assert audit["power_samples_started_in_interval"]["definition"] == (
        runner.POWER_SAMPLES_STARTED_IN_INTERVAL_DEFINITION
    )

    target_summary = audit["sampling_target"]
    rows_meeting_target = sum(count >= target for count in started_counts)
    rows_below_target = sum(count < target for count in started_counts)
    assert target_summary["target_samples_per_row"] == target
    assert target_summary["field"] == "power_samples_started_in_interval"
    assert target_summary["rows_meeting_target"] == rows_meeting_target
    assert target_summary["rows_below_target"] == rows_below_target
    assert target_summary["minimum_started_samples"] == min(started_counts)
    assert target_summary["maximum_started_samples"] == max(started_counts)
    assert target_summary["result"] == (
        "pass" if rows_below_target == 0 else "fail"
    )


def _fake_wrapper_tools(tmp_path: Path) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(
        """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\n' '{"fake": true}' > "$out"
    ;;
  sample)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    trap 'exit 0' TERM INT
    while :; do
      timestamp="$(date -u '+%Y-%m-%dT%H:%M:%S.%N%:z')"
      {
        printf '%s\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c'
        printf '%s\n' "${timestamp},47,800,58.5"
      } > "${out}.tmp"
      mv "${out}.tmp" "$out"
      /bin/sleep 0.05
    done
    ;;
esac
""",
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "${1:-}" >> "${DOCKER_CALLS:-${DOCKER_ARGS}.calls}"
printf '%s\n' "$@" > "${DOCKER_ARGS:?}"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
    (bindir / "tt-smi").write_text(
        "#!/bin/sh\nset -eu\nexit 0\n", encoding="utf-8"
    )
    (bindir / "tt-smi").chmod(stat.S_IRWXU)
    (bindir / "timeout").write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "$@" > "${TIMEOUT_LOG:?}"
shift 3
exec "$@"
""",
        encoding="utf-8",
    )
    (bindir / "timeout").chmod(stat.S_IRWXU)
    return bindir


@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_fails_closed_when_cooldown_start_timestamp_cannot_be_parsed(
    tmp_path: Path,
):
    bindir = _fake_wrapper_tools(tmp_path)
    (bindir / "date").write_text(
        """#!/bin/sh
set -eu
if [ "${2:-}" = "--iso-8601=ns" ]; then
  printf '%s\n' 'not-a-timestamp'
else
  printf '%s\n' '2026-10-12T00:00:00.000000000+00:00'
fi
""",
        encoding="utf-8",
    )
    (bindir / "date").chmod(stat.S_IRWXU)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    docker_args = tmp_path / "docker-args"
    docker_history = tmp_path / "docker-calls"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "unparsed-start"),
            "--",
            "--run-number",
            "2",
            "--cooldown-wait-seconds",
            "0.1",
        ],
        cwd=tmp_path / "repo",
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "DOCKER_CALLS": str(docker_history),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "cooldown start timestamp could not be parsed" in completed.stderr
    assert docker_history.read_text(encoding="utf-8").splitlines() == ["kill"]


@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_runs_cooldown_before_docker_timeout_window(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    events = tmp_path / "events"
    (bindir / "python3").write_text(
        """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' '{"fake": true}' > "$out"
    ;;
  sample)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    trap 'exit 0' TERM INT
    while :; do
      timestamp="$(date -u '+%Y-%m-%dT%H:%M:%S.%N%:z')"
      {
        printf '%s\\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c'
        printf '%s\\n' "${timestamp},40,1350,50"
      } > "${out}.tmp"
      mv "${out}.tmp" "$out"
      /bin/sleep 0.05
    done
    ;;
esac
""",
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "sleep").write_text(
        f"""#!/bin/sh
printf 'sleep:%s\\n' \"$*\" >> \"{events}\"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "sleep").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        f"""#!/bin/sh
set -eu
if [ "${{1:-}}" = kill ]; then exit 0; fi
printf 'docker\\n' >> "{events}"
printf '%s\\n' "$@" > "${{DOCKER_ARGS:?}}"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
    (bindir / "tt-smi").write_text(
        "#!/bin/sh\nset -eu\nexit 0\n", encoding="utf-8"
    )
    (bindir / "tt-smi").chmod(stat.S_IRWXU)
    (bindir / "timeout").write_text(
        """#!/bin/sh
set -eu
printf '%s\\n' "$@" > "${TIMEOUT_LOG:?}"
shift 3
exec "$@"
""",
        encoding="utf-8",
    )
    (bindir / "timeout").chmod(stat.S_IRWXU)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    docker_args = tmp_path / "docker-args"
    timeout_log = tmp_path / "timeout-args"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output"),
            "--",
            "--run-number",
            "2",
            "--cooldown-wait-seconds",
            "0.1",
            "--expected-aiclk-mhz",
            "1350",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    events_seen = events.read_text(encoding="utf-8").splitlines()
    assert "docker" in events_seen
    assert all(event != "sleep:-- 0.1" for event in events_seen)
    docker = docker_args.read_text(encoding="utf-8").splitlines()
    assert docker[docker.index("--cooldown-wait-seconds") + 1] == "600"
    assert "--cooldown-start-utc" in docker
    assert "--cooldown-end-utc" in docker
    assert docker[docker.index("--cooldown-baseline-power-w") + 1] == "47.0"
    assert docker[docker.index("--cooldown-power-tolerance-w") + 1] == "5.0"
    assert docker[docker.index("--cooldown-gate-result") + 1] == "pass"
    assert "--cooldown-gate-verified" in docker
    assert timeout_log.read_text(encoding="utf-8").splitlines()[2] == "60s"
    assert "sleep(args.cooldown_wait_seconds)" not in Path(runner.__file__).read_text(
        encoding="utf-8"
    )

    rejected = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output-large"),
            "--",
            "--run-number",
            "2",
            "--cooldown-wait-seconds",
            "1000.0",
            "--expected-aiclk-mhz",
            "1350",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert rejected.returncode == 2
    assert "at most 600" in rejected.stderr
    assert events.read_text(encoding="utf-8").splitlines() == events_seen


@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_rejects_stale_usable_sample_before_docker(tmp_path: Path):
    bindir = _fake_wrapper_tools(tmp_path)
    (bindir / "python3").write_text(
        """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' '{"fake": true}' > "$out"
    ;;
  sample)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c' > "$out"
    printf '%s\\n' '1970-01-01T00:00:00+00:00,40,1350,50' >> "$out"
    trap 'exit 0' TERM INT
    while :; do sleep 1; done
    ;;
esac
""",
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    docker_args = tmp_path / "docker-args"
    docker_history = tmp_path / "docker-calls"
    output_dir = tmp_path / "stale-sample"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(output_dir),
            "--",
            "--run-number",
            "2",
            "--cooldown-wait-seconds",
            "0.1",
        ],
        cwd=tmp_path / "repo",
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "DOCKER_CALLS": str(docker_history),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
            "HEKATUS_TT_ISSUE127_COOLDOWN_CAP_SECONDS": "0.2",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 1
    assert "cooldown gate failure" in completed.stderr
    docker_args_tokens = docker_args.read_text(encoding="utf-8").splitlines()
    assert docker_args_tokens[0] == "kill"
    docker_call_history = docker_history.read_text(encoding="utf-8").splitlines()
    assert docker_call_history
    assert all(call == "kill" for call in docker_call_history)
    assert "run" not in docker_call_history
    failures = list(output_dir.glob("cooldown-gate-failure-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["measurement_started"] is False
    assert failure["observed_sample_timestamp_utc"] == "1970-01-01T00:00:00+00:00"
    assert failure["observed_power_w"] == 40.0
    assert failure["observed_temperature_c"] == 50.0
    assert failure["observed_idle_aiclk_mhz"] == 1350.0


def _run_cooldown_cap_wrapper(
    tmp_path: Path,
    *,
    sampler_script: str,
    run_scope: str | None,
    clock_shim: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    """Run the two cap scenarios with inert tools and one shared wrapper setup."""
    bindir = _fake_wrapper_tools(tmp_path)
    (bindir / "python3").write_text(sampler_script, encoding="utf-8")
    (bindir / "python3").chmod(stat.S_IRWXU)
    if clock_shim is not None:
        (bindir / "date").write_text(clock_shim, encoding="utf-8")
        (bindir / "date").chmod(stat.S_IRWXU)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    docker_args = tmp_path / "docker-args"
    docker_history = tmp_path / "docker-calls"
    output_dir = tmp_path / "cooldown-cap"
    command = [str(copied_wrapper), str(output_dir), "--", "--run-number", "2"]
    if run_scope is not None:
        command.extend(["--run-scope", run_scope, "--run-timeout-seconds", "90"])
    completed = subprocess.run(
        command,
        cwd=tmp_path / "repo",
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "DOCKER_CALLS": str(docker_history),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
            "HEKATUS_TT_ISSUE127_COOLDOWN_CAP_SECONDS": "0.2",
            "CLOCK_STATE": str(tmp_path / "clock-state"),
            "READY": str(tmp_path / "sample-ready"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return completed, docker_args, docker_history, output_dir


@pytest.mark.parametrize(
    ("run_scope", "expected_run_scope"),
    (
        (None, None),
        ("measurement-2-rerun", "measurement-2-rerun"),
    ),
)
@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_refuses_measurement_when_cooldown_gate_reaches_cap(
    tmp_path: Path, run_scope: str | None, expected_run_scope: str | None
):
    """Freeze at wrapper cooldown start; latch one read after the first CSV mv.

    The sampler timestamps after the clock freezes, so the gate rejects it as
    "after the current wait time". Observations are assigned before that check:
    the failure record must still retain 800 MHz and 80 W on the next cap check.
    """
    clock_shim = """#!/bin/sh
set -eu
if [ "$#" -eq 2 ] && [ "$1" = '-u' ] && [ "$2" = '--iso-8601=ns' ]; then
  format=iso
elif [ "$#" -eq 1 ] && [ "$1" = '+%s%N' ]; then
  format=ns
else
  exec /bin/date "$@"
fi
if [ "$format" = iso ] && [ ! -f "${CLOCK_STATE:?}" ]; then
  /bin/date +%s%N > "${CLOCK_STATE}.tmp"
  mv "${CLOCK_STATE}.tmp" "$CLOCK_STATE"
fi
read -r now < "$CLOCK_STATE"
if [ -f "${CLOCK_STATE}.latched" ]; then
  # Advance past the test-only 0.2-second cap by one second.
  now=$((now + 200000000 + 1000000000))
elif [ "$format" = ns ] && [ -f "${READY:?}" ]; then
  : > "${CLOCK_STATE}.latched"
fi
if [ "$format" = ns ]; then
  printf '%s\\n' "$now"
else
  epoch="$(printf '%s.%09d' "$((now / 1000000000))" "$((now % 1000000000))")"
  exec /bin/date -u -d "@${epoch}" --iso-8601=ns
fi
"""
    sampler_script = """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' '{"fake": true}' > "$out"
    ;;
  sample)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    trap 'exit 0' TERM INT
    while [ ! -f "${CLOCK_STATE:?}" ]; do /bin/sleep 0.01; done
    timestamp="$(date -u '+%Y-%m-%dT%H:%M:%S.%N%:z')"
    {
      printf '%s\\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c'
      printf '%s\\n' "${timestamp},80,800,70"
    } > "${out}.tmp"
    mv "${out}.tmp" "$out"
    : > "${READY:?}"
    while :; do /bin/sleep 0.05; done
    ;;
esac
"""
    completed, docker_args, docker_history, output_dir = _run_cooldown_cap_wrapper(
        tmp_path, sampler_script=sampler_script, run_scope=run_scope, clock_shim=clock_shim
    )
    assert completed.returncode == 1
    assert "cooldown gate failure" in completed.stderr
    assert "measurement was not started" in completed.stderr
    assert docker_args.exists()
    assert docker_args.read_text(encoding="utf-8").splitlines()[0] == "kill"
    docker_call_history = docker_history.read_text(encoding="utf-8").splitlines()
    assert docker_call_history
    assert all(call == "kill" for call in docker_call_history)
    assert "run" not in docker_call_history
    failures = list(output_dir.glob("cooldown-gate-failure-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["failure_type"] == "cooldown_gate_failure"
    assert failure["measurement_started"] is False
    assert failure["run_scope"] == expected_run_scope
    assert failure["wait_cap_seconds"] == 600.0
    assert failure["observed_idle_aiclk_mhz"] == 800.0
    assert failure["observed_power_w"] == 80.0
    assert "after the current wait time" in completed.stderr
    assert failure["actual_wait_seconds"] > 0
    assert failure["actual_wait_seconds"] == pytest.approx(1.2, abs=1e-6)
    frozen = int((tmp_path / "clock-state").read_text(encoding="utf-8"))
    for field, offset_ns in (("wait_start_utc", 0), ("wait_end_utc", 1_200_000_000)):
        seconds, nanoseconds = divmod(frozen + offset_ns, 1_000_000_000)
        expected = subprocess.check_output(
            ["/bin/date", "-u", "-d", f"@{seconds}.{nanoseconds:09d}", "--iso-8601=ns"],
            text=True,
        ).strip()
        assert failure[field] == expected


@pytest.mark.parametrize(
    ("run_scope", "expected_run_scope"),
    (
        (None, None),
        ("measurement-2-rerun", "measurement-2-rerun"),
    ),
)
@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_refuses_measurement_without_complete_sample_at_startup(
    tmp_path: Path, run_scope: str | None, expected_run_scope: str | None
):
    """Real-time cap expiry records null observations while a silent sampler lives."""
    sampler_script = """#!/bin/sh
set -eu
case "${2:-}" in
  capture-env)
    out=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = --out ]; then out="$2"; shift 2; continue; fi
      shift
    done
    printf '%s\\n' '{"fake": true}' > "$out"
    ;;
  sample)
    trap 'exit 0' TERM INT
    while :; do /bin/sleep 0.05; done
    ;;
esac
"""
    completed, docker_args, docker_history, output_dir = _run_cooldown_cap_wrapper(
        tmp_path, sampler_script=sampler_script, run_scope=run_scope
    )
    assert completed.returncode == 1
    assert "cooldown gate failure" in completed.stderr
    assert "measurement was not started" in completed.stderr
    assert "did not pass before the wait cap" in completed.stderr
    assert docker_args.read_text(encoding="utf-8").splitlines()[0] == "kill"
    docker_call_history = docker_history.read_text(encoding="utf-8").splitlines()
    assert docker_call_history
    assert all(call == "kill" for call in docker_call_history)
    failures = list(output_dir.glob("cooldown-gate-failure-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["failure_type"] == "cooldown_gate_failure"
    assert failure["measurement_started"] is False
    assert failure["run_scope"] == expected_run_scope
    assert failure["wait_cap_seconds"] == 600.0
    for field in (
        "observed_power_w",
        "observed_temperature_c",
        "observed_idle_aiclk_mhz",
        "observed_sample_timestamp_utc",
    ):
        assert failure[field] is None
    assert not list(output_dir.glob("power-*.csv"))


@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_requires_explicit_approved_90_second_scope(tmp_path: Path):
    bindir = _fake_wrapper_tools(tmp_path)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    docker_args = tmp_path / "docker-args"
    timeout_log = tmp_path / "timeout-args"
    base_env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "DOCKER_ARGS": str(docker_args),
        "TIMEOUT_LOG": str(timeout_log),
        "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
    }

    approved = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "approved"),
            "--",
            "--run-number",
            "1",
            "--run-scope",
            "confirmation",
            "--run-timeout-seconds",
            "90",
        ],
        cwd=copied_root,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert approved.returncode == 0, approved.stderr
    assert timeout_log.read_text(encoding="utf-8").splitlines()[2] == "90s"
    docker = docker_args.read_text(encoding="utf-8").splitlines()
    assert "--run-scope" in docker
    assert docker[docker.index("--run-scope") + 1] == "confirmation"
    assert "--run-timeout-seconds" in docker
    assert docker[docker.index("--run-timeout-seconds") + 1] == "90"

    rerun = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "measurement-2-rerun"),
            "--",
            "--run-number",
            "2",
            "--run-scope",
            "measurement-2-rerun",
            "--run-timeout-seconds",
            "90",
        ],
        cwd=copied_root,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert rerun.returncode == 0, rerun.stderr
    rerun_docker = docker_args.read_text(encoding="utf-8").splitlines()
    assert rerun_docker[rerun_docker.index("--run-scope") + 1] == "measurement-2-rerun"
    assert timeout_log.read_text(encoding="utf-8").splitlines()[2] == "90s"

    rejected = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "rejected"),
            "--",
            "--run-number",
            "1",
            "--run-timeout-seconds",
            "90",
        ],
        cwd=copied_root,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert rejected.returncode == 2
    assert "approved target" in rejected.stderr

    env_only = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "env-only"),
            "--",
            "--run-number",
            "1",
        ],
        cwd=copied_root,
        env={**base_env, "HEKATUS_TT_CONTAINER_TIMEOUT_S": "90"},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert env_only.returncode == 2
    assert "must match" in env_only.stderr

    mismatched = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "mismatched"),
            "--",
            "--run-number",
            "2",
            "--run-scope",
            "confirmation",
            "--run-timeout-seconds",
            "90",
            "--cooldown-wait-seconds",
            "0.1",
            "--expected-aiclk-mhz",
            "1350",
        ],
        cwd=copied_root,
        env=base_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert mismatched.returncode == 2
    assert "requires --run-number 1" in mismatched.stderr


@pytest.mark.usefixtures("wrapper_execution")
def test_issue127_wrapper_selects_generic_custom_runner_with_bound_artifacts(tmp_path: Path):
    bindir = _fake_wrapper_tools(tmp_path)
    copied_wrapper = tmp_path / "repo/enodia/tt/bench/run_in_container.sh"
    copied_wrapper.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "enodia/tt/bench/run_in_container.sh", copied_wrapper)
    shutil.copy2(ROOT / "enodia/tt/bench/telemetry.py", copied_wrapper.parent / "telemetry.py")
    copied_root = copied_wrapper.parents[3]
    docker_args = tmp_path / "docker-args"
    timeout_log = tmp_path / "timeout-args"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output"),
            "--",
            "--run-number",
            "1",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    docker = docker_args.read_text(encoding="utf-8").splitlines()
    assert "tools/newton_schulz_issue127.py" in docker
    assert docker[docker.index("--entrypoint") + 1] == "python3"
    assert "--env-json" in docker
    assert "--power-trace" in docker
    assert timeout_log.read_text(encoding="utf-8").splitlines()[2] == "60s"

    docker_before_sampler_off = docker_args.read_text(encoding="utf-8")
    sampler_off = subprocess.run(
        [
            str(copied_wrapper),
            str(tmp_path / "output-sampler-off"),
            "--",
            "--run-number",
            "1",
        ],
        cwd=copied_root,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "TIMEOUT_LOG": str(timeout_log),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
            "HEKATUS_TT_TELEMETRY_MODE": "off",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert sampler_off.returncode == 2
    assert "requires telemetry sampler" in sampler_off.stderr
    assert docker_args.read_text(encoding="utf-8") == docker_before_sampler_off
