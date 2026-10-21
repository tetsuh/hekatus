"""Focused Issue #146 revision-5 board-free contract tests."""

from __future__ import annotations

from tests.test_issue146_analyzer import _plan, _record
from tests.test_issue146_runner import _environment
from tools import analyze_issue146 as analyzer
from tools import issue146
from tools.issue146_analysis import current_dip_facts


def _r5_row(aiclk: float, arbiter: int) -> dict[str, object]:
    return {
        "aiclk_mhz": aiclk,
        "aiclk_arb_min_raw": "0x10546",
        "aiclk_arb_max_raw": hex((7 << 16) | arbiter),
        "enabled_min_arb_raw": "0x3",
        "enabled_max_arb_raw": "0x1d1",
    }


def test_issue146_revision5_signed_r5_bounds() -> None:
    assert analyzer._r5_signed_pass(0.0) is True
    assert analyzer._r5_signed_pass(7.0) is True
    assert analyzer._r5_signed_pass(-1.0) is False
    assert analyzer._r5_signed_pass(8.0) is False


def test_issue146_revision5_missing_trace_still_counts_arbiter_gate() -> None:
    criterion = analyzer._smoke_arbiter_criterion([])
    assert criterion["gate"] is True


def test_issue146_revision5_missing_trace_policy_has_four_gates() -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["run"]["observed_duration_seconds"] = 40.0
    record["trace"]["samples"] = []
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert result["policy"]["gate_count"] == 4
    assert len(result["gates"]) == 4


def test_issue146_revision5_smoke_signed_r5_rejects_negative_delta() -> None:
    result = analyzer._smoke_residual_criterion([_r5_row(1001.0, 1000)])
    assert result["pass"] is False
    assert result["signed_delta_min_mhz"] == -1.0
    assert result["signed_delta_max_mhz"] == -1.0


def test_issue146_revision5_regulated_signed_r5_bounds() -> None:
    rows = [_r5_row(aiclk, 1000) for aiclk in (1000.0, 993.0, 1001.0)]
    decoded = {
        id(row): {"arb": analyzer.decode_arbiter_row(row)} for row in rows
    }
    result = analyzer._regulated_tracking(decoded, rows)
    assert result["pass"] is False
    assert result["signed_delta_min_mhz"] == -1.0
    assert result["signed_delta_max_mhz"] == 7.0


def test_issue146_revision5_smoke_diagnostics_are_not_a_gate() -> None:
    result = analyzer._smoke_reported_diagnostic(
        [{"current_a": 100.0, "board_power_w": 250.0},
         {"current_a": 140.0, "board_power_w": 350.0}],
        "C081",
    )
    assert result["gate"] is False
    assert result["pass"] is None
    assert result["minimum_current_a"] == 100.0
    assert result["below_150_a_count"] == 2
    assert result["power_mean_w"] == 300.0
    assert result["power_sample_sd_w"] > 0.0
    assert result["power_r1"] < 0.0


def test_issue146_revision5_plan_derives_cadence_and_explicit_override(tmp_path) -> None:
    plan = issue146.load_plan()
    assert issue146.plan_trace_interval_seconds(plan) == 1.0
    assert issue146.trace_interval_seconds(plan) == (1.0, "measurement_scope.trace_interval_seconds")
    assert issue146.trace_interval_seconds(plan, 0.001) == (0.001, "explicit_test_override")
    config = issue146.configuration(plan, "C018")
    record = issue146.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue146-cadence-override",
        options=issue146.Issue146RunOptions(trace_interval_seconds=0.001),
    )
    assert record["trace_cadence"]["host_sampler_interval_seconds"] == 0.001
    assert record["trace_cadence"]["test_override"] is True
    assert record["provenance"]["trace_interval_source"] == "explicit_test_override"


def test_issue146_revision5_cadence_provenance_rejects_drift() -> None:
    plan = issue146.load_plan()
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    environment["telemetry_sampler"] = {"mode": "explicit", "interval_seconds": 2.0}
    try:
        issue146.validate_environment_for_run(environment, configuration=config, plan=plan)
    except issue146.Issue146Error as error:
        assert "sampler interval" in str(error)
    else:
        raise AssertionError("cadence drift must fail closed")


def test_issue146_revision5_smoke_idle_window_is_not_long_plateau() -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["run"]["observed_duration_seconds"] = 40.0
    report = analyzer.analyze_smoke([record], plan=_plan())["runs"][0]["run"]
    assert report["duration"]["idle_window_seconds"] == {
        "start_seconds": 10.0,
        "end_seconds": 40.0,
    }


def test_issue146_revision5_current_dip_equality_and_group_boundary() -> None:
    rows = [
        {"timestamp_monotonic_ns": offset * 1_000_000_000,
         "current_a": current, "board_power_w": 300.0, "power_w": 140.0}
        for offset, current in ((10, 100.0), (11, 100.0), (12, 79.0), (13, 80.0), (14, 100.0))
    ]
    report = current_dip_facts(
        rows,
        first_launch_start_monotonic_ns=0,
        window_start_seconds=10.0,
        window_end_seconds=40.0,
        group_timings=[
            {"start_monotonic_ns": 0, "end_monotonic_ns": 12_000_000_000},
            {"start_monotonic_ns": 13_000_000_000, "end_monotonic_ns": 40_000_000_000},
        ],
        sample_interval_seconds=1.0,
    )
    assert report["median_current_a"] == 100.0
    assert report["threshold_current_a"] == 80.0
    assert report["dip_count"] == 1
    assert report["dip_fraction"] == 0.2
    dip = report["dips"][0]
    assert dip["offset_seconds"] == 12.0
    assert dip["group_correspondence"]["containing_gap_index"] == 0
    assert dip["group_correspondence"]["containing_group_index"] is None
