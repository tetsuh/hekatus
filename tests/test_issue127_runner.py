"""Board-free contract tests for the Issue #127 ABBA/BADC runner."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from enodia.tt.bench.power_trace_snapshot import EXPLICIT_FINAL_SAMPLE
from tools import newton_schulz_issue127 as runner

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/measurements/2026-10-12-host-newton-schulz-issue127-abba-plan.json"
CSV_HEADER = "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"


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


def test_issue127_final_trace_facts_record_source_and_adjusted_p50():
    row = {
        **runner.build_occurrence_plan(1)[0],
        "timed_start_utc": "2026-10-12T00:00:01+00:00",
        "timed_end_utc": "2026-10-12T00:00:03+00:00",
        "timed_start_monotonic": 1.0,
        "timed_end_monotonic": 3.0,
        "timed_duration_seconds": 2.0,
        "raw_p50_tflops": 16.0,
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
        ],
    }
    runner.apply_final_trace_facts([row], power_trace=trace)

    assert row["aiclk_interval"]["trace_sha256"] == "c" * 64
    assert row["row_time_weighted_aiclk_mhz"] == pytest.approx(1275.0)
    assert row["adjusted_p50_tflops"] == pytest.approx(16.0 * 1350.0 / 1275.0)
    assert row["aiclk_interval"]["requested_sampler_interval_seconds"] == 2.0
    assert row["aiclk_interval"]["power_sample_count_in_interval"] == 2
    assert row["aiclk_interval"]["power_sample_target_met"] is True


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
        run_number=2,
        wait_seconds=3.0,
        expected_aiclk_mhz=1350.0,
        environment=environment,
        start_utc="2026-10-12T00:00:00.500000+00:00",
        end_utc="2026-10-12T00:00:03.500000+00:00",
        wait_performed=True,
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


def test_issue127_close_failure_cannot_produce_pass_record():
    environment = runner.sanitize_environment(_environment())
    rows = []
    for row in runner.build_occurrence_plan(1):
        measured = dict(row)
        measured.update(
            {
                "status": "ok",
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
            "device_close_failed": True,
        },
        run_number=1,
        run_id="run127-test",
        environment=environment,
        cooldown=runner.build_cooldown_metadata(
            run_number=1,
            wait_seconds=None,
            expected_aiclk_mhz=None,
            environment=environment,
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
    assert record["device_close_failed"] is True
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
    printf '%s\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c' > "$out"
    printf '%s\n' '2026-10-12T00:00:00+00:00,47,800,58.5' >> "$out"
    trap 'exit 0' TERM INT
    while :; do sleep 1; done
    ;;
esac
""",
        encoding="utf-8",
    )
    (bindir / "python3").chmod(stat.S_IRWXU)
    (bindir / "docker").write_text(
        """#!/bin/sh
set -eu
printf '%s\n' "$@" > "${DOCKER_ARGS:?}"
exit 0
""",
        encoding="utf-8",
    )
    (bindir / "docker").chmod(stat.S_IRWXU)
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
    printf '%s\\n' 'timestamp_utc,power_w,aiclk_mhz,asic_temp_c' > "$out"
    printf '%s\\n' '2026-10-12T00:00:00+00:00,40,1350,50' >> "$out"
    trap 'exit 0' TERM INT
    while :; do :; done
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


def test_issue127_wrapper_refuses_measurement_when_cooldown_gate_reaches_cap(tmp_path: Path):
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
    printf '%s\\n' '2026-10-12T00:00:00+00:00,80,800,70' >> "$out"
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
    output_dir = tmp_path / "rejected-at-cap"
    completed = subprocess.run(
        [
            str(copied_wrapper),
            str(output_dir),
            "--",
            "--run-number",
            "2",
            "--run-scope",
            "measurement-2-rerun",
            "--run-timeout-seconds",
            "90",
        ],
        cwd=tmp_path / "repo",
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "DOCKER_ARGS": str(docker_args),
            "HEKATUS_TT_RUNNER": "tools/newton_schulz_issue127.py",
            "HEKATUS_TT_ISSUE127_COOLDOWN_CAP_SECONDS": "0.01",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 1
    assert "cooldown gate failure" in completed.stderr
    assert "measurement was not started" in completed.stderr
    assert docker_args.exists()
    assert docker_args.read_text(encoding="utf-8").splitlines()[0] == "kill"
    failures = list(output_dir.glob("cooldown-gate-failure-*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["failure_type"] == "cooldown_gate_failure"
    assert failure["measurement_started"] is False
    assert failure["wait_cap_seconds"] == 600.0
    assert failure["observed_idle_aiclk_mhz"] == 800.0


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
