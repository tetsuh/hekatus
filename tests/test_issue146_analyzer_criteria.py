"""Synthetic criterion and fail-closed coverage for Issue #146."""

from __future__ import annotations

import pytest

from tests.test_issue146_analyzer import _plan, _plateau_rows, _record, _records, _rehash
from tools import analyze_issue146 as analyzer
from tools.issue146_analysis import pooled_statistics


@pytest.mark.parametrize(
    ("criterion", "configuration_id", "changes"),
    [
        ("R1-power-regulated", "C108", {"board_power_w": 250.0}),
        ("R2-arbiter-unclamped", "C108", {"aiclk_mhz": 1350.0, "aiclk_arb_max_raw": "0x80546"}),
        ("R3-mask-unchanged", "C108", {"enabled_min_arb_raw": "0x0"}),
        ("R4-busy", "C108", {"aiclk_arb_min_raw": "0x10320"}),
        ("R5-aiclk-tracks-arbiter", "C108", {"aiclk_mhz": 1005.0}),
        ("R6-thermal-not-binding", "C108", {"asic_temp_c": 90.0}),
        ("R7-reported-index", "C108", {"aiclk_arb_max_raw": "0x90000"}),
        ("K1-clamped", "C018", {"aiclk_mhz": 1349.0}),
        ("K2-unregulated", "C018", {"board_power_w": 310.0}),
        ("K3-mask-and-busy", "C018", {"enabled_min_arb_raw": "0x0"}),
        ("K4-thermal", "C018", {"asic_temp_c": 90.0}),
    ],
)
def test_each_registered_run_criterion_failure_is_h_other(
    criterion: str, configuration_id: str, changes: dict[str, object]
) -> None:
    records = _records()
    run_number = next(
        index
        for index, record in enumerate(records, 1)
        if record["configuration"]["id"] == configuration_id
    )
    record = records[run_number - 1]
    for row in _plateau_rows(record):
        row.update(changes)
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "H-OTHER"
    assert any(criterion in item for item in result["failed_criteria"])


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("p1", "P1-control-unthrottled"),
        ("p2", "P2-regulated-depressed"),
        ("p3", "P3-monotonic"),
    ],
)
def test_each_registered_precondition_failure_is_h_other(mutation: str, expected: str) -> None:
    records = _records()
    if mutation == "p1":
        target = records[0]
        for row in _plateau_rows(target):
            row.update(aiclk_mhz=1200.0, aiclk_arb_max_raw="0x704b0")
    elif mutation == "p2":
        target = records[1]
        for row in _plateau_rows(target):
            row.update(aiclk_mhz=1350.0, aiclk_arb_max_raw="0x80546")
    else:
        for target in (records[1], records[2]):
            for row in _plateau_rows(target):
                if target["configuration"]["id"] == "C108":
                    row.update(aiclk_mhz=1200.0, aiclk_arb_max_raw="0x704b0")
                else:
                    row.update(aiclk_mhz=1000.0, aiclk_arb_max_raw="0x703e8")
    for record in records:
        _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "H-OTHER"
    assert any(expected in item for item in result["failed_preconditions"])


def test_issue146_r7_decode_reports_special_alternatives() -> None:
    for raw, phrase in (
        ("0x90000", "doppler_critical"),
        ("0x40000", "ambiguous arbiter alternatives"),
        ("0x10000", "impossible mask index"),
    ):
        records = _records()
        for row in _plateau_rows(records[1]):
            row["aiclk_arb_max_raw"] = raw
        _rehash(records[1])
        result = analyzer.analyze_runs(records, plan=_plan())
        assert result["classification"] == "H-OTHER"
        report = result["runs"][1]
        assert phrase in report["criteria"]["R7-reported-index"]["reason"]


def test_issue146_precision_and_exact_count_boundaries() -> None:
    good = analyzer.point_statistics([300.0] * 50)
    assert good["n"] == 50
    assert good["half_width"] <= 10.0
    records = _records()
    record = records[1]
    rows = record["trace"]["samples"]
    assert isinstance(rows, list)
    plateau = _plateau_rows(record)
    for index, row in enumerate(plateau):
        row["board_power_w"] = 200.0 if index % 2 else 400.0
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "not_evaluable"
    assert any("precision" in field for field in result["runs"][1]["missing_fields"])
    for index in range(10):
        rows.remove(plateau[-(index + 1)])
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["runs"][1]["plateau"]["raw_count"] == 50


def test_issue146_glitch_nominal_boundary_gddr_scope_and_original_row_index() -> None:
    records = _records()
    record = records[1]
    plateau = _plateau_rows(record)
    plateau[0]["enabled_max_arb_raw"] = "0x0"
    plateau[1]["gddr_0_1_temp_c"] = "0x0"
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    report = result["runs"][1]
    assert report["glitch_count"] == 2
    assert report["glitches"][0]["row_index"] > 0
    assert report["glitches"][1]["excludes"] == "R6"
    assert report["plateau"]["r6_retained_count"] == 58


def test_issue146_unflagged_max_mask_is_configuration_before_r3() -> None:
    records = _records()
    record = records[1]
    row = _plateau_rows(record)[0]
    row["enabled_max_arb_raw"] = "0x0"
    row["aiclk_mhz"] = 1350.0
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "not_evaluable"
    assert any(
        "configuration.ENABLED_MAX_ARB" in field for field in result["runs"][1]["missing_fields"]
    )


def test_issue146_pooled_r1_omits_cross_run_pair() -> None:
    pooled = pooled_statistics([[1.0, 2.0], [10.0, 11.0]])
    assert pooled["pair_count"] == 2


def test_issue146_group_gap_and_overrun_are_not_trusted_from_record_fields() -> None:
    records = _records()
    record = records[0]
    record["groups"]["groups"] = [
        {"start_monotonic_ns": 0, "end_monotonic_ns": 65_000_000_000},
        {"start_monotonic_ns": 70_000_000_000, "end_monotonic_ns": 191_000_000_000},
    ]
    record["run"]["observed_duration_seconds"] = 191.0
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "not_evaluable"
    fields = result["runs"][0]["missing_fields"]
    assert "load_contract.idle_fraction" in fields
    assert "overrun_seconds.limit" in fields


def test_issue146_smoke_each_acceptance_item_is_individual() -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["run"]["observed_duration_seconds"] = 40.0
    result = analyzer.analyze_smoke([record], plan=_plan())
    acceptance = result["acceptance"]
    assert [item["name"] for item in acceptance] == [
        "idle_fraction_[10,40)",
        "regulated_current_and_power_[10,40)",
        "arbiter_words_and_enabled_max",
        "R5_residual_[10,40)",
        "R7_index_distribution_[10,40)",
        "smoke_failure_requires_plan_revision",
    ]
    assert result["affects_long_verdict"] is False


@pytest.mark.parametrize("failure", ["idle", "current", "power", "words", "residual", "index"])
def test_issue146_smoke_acceptance_each_failure_is_reported(failure: str) -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["run"]["observed_duration_seconds"] = 40.0
    rows = record["trace"]["samples"]
    assert isinstance(rows, list)
    smoke_rows = [row for row in rows if 10 <= row["timestamp_monotonic_ns"] // 1_000_000_000 < 40]
    if failure == "idle":
        record["groups"]["groups"] = [
            {"start_monotonic_ns": 0, "end_monotonic_ns": 20_000_000_000},
            {"start_monotonic_ns": 25_000_000_000, "end_monotonic_ns": 40_000_000_000},
        ]
    elif failure == "current":
        for row in smoke_rows:
            row["current_a"] = 100.0
    elif failure == "power":
        for index, row in enumerate(smoke_rows):
            row["board_power_w"] = 250.0 if index % 2 else 350.0
    elif failure == "words":
        smoke_rows[0].pop("enabled_max_arb_raw")
    elif failure == "residual":
        for row in smoke_rows:
            row["aiclk_mhz"] = 1205.0
    elif failure == "index":
        for row in smoke_rows:
            row["aiclk_arb_max_raw"] = "0x904b0"
    _rehash(record)
    result = analyzer.analyze_smoke([record], plan=_plan())
    acceptance = {item["name"]: item for item in result["acceptance"]}
    expected = {
        "idle": "idle_fraction_[10,40)",
        "current": "regulated_current_and_power_[10,40)",
        "power": "regulated_current_and_power_[10,40)",
        "words": "arbiter_words_and_enabled_max",
        "residual": "R5_residual_[10,40)",
        "index": "R7_index_distribution_[10,40)",
    }[failure]
    if failure == "words":
        assert acceptance[expected]["status"] == "not_evaluable"
    else:
        assert acceptance[expected]["pass"] is False


def test_issue146_smoke_current_residual_and_index_include_post_40_rows() -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["run"]["observed_duration_seconds"] = 40.0
    rows = record["trace"]["samples"]
    assert isinstance(rows, list)
    post_40 = next(row for row in rows if row["timestamp_monotonic_ns"] == 60_000_000_000)
    post_40["current_a"] = 100.0
    post_40["aiclk_mhz"] = 1205.0
    post_40["aiclk_arb_max_raw"] = "0x904b0"
    _rehash(record)
    result = analyzer.analyze_smoke([record], plan=_plan())
    acceptance = {item["name"]: item for item in result["acceptance"]}
    assert acceptance["regulated_current_and_power_[10,40)"]["pass"] is False
    assert acceptance["R5_residual_[10,40)"]["pass"] is False
    assert acceptance["R7_index_distribution_[10,40)"]["pass"] is False
    assert acceptance["regulated_current_and_power_[10,40)"]["power_sd_window"] == "[10,40)"
    assert acceptance["R5_residual_[10,40)"]["window"] == "[10,∞)"


def test_issue146_smoke_unknown_configuration_is_not_evaluable() -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["configuration"]["id"] = "C999"
    _rehash(record)
    result = analyzer.analyze_smoke([record], plan=_plan())
    acceptance = result["acceptance"]
    assert all(item["pass"] is not True for item in acceptance)
    assert all(item["status"] in {"not_evaluable", "fail"} for item in acceptance)
