"""Revision-6 any-sample/snapshot policy; all probes are board-free (I1-I8)."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from enodia.strict_json import dumps
from tests.test_issue146_analyzer import _plan, _record, _records, _rehash, _trace_row
from tools import analyze_issue146 as analyzer

MISSING = object()
BAD_WORDS = ["0x1d0", "0x0", None, "", "invalid", "0x100000000", True, 465, MISSING]
# Snapshots retain their existing integer-word support; raw traces require hex text.
BAD_POST_WORDS = ["0x1d0", "0x0", None, "", "invalid", "0x100000000", True, 464, MISSING]
# Include both exact boundaries, loaded-but-not-plateau, and pre/post-load points.
OUTSIDE_TIMES = [-1, 0, 59, 180, 181, 190]
ARCHIVE = Path(__file__).parents[1] / "docs/measurements/issue146-doppler-conformance/2026-10-21"
ORDER = ["c018", "c108", "c081", "c081", "c108", "c018"]


def _assign(mapping, key, value):
    if value is MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = value


def _configuration_failure(result, position):
    report = result["runs"][position]
    assert result["classification"] == "not_evaluable"
    assert any(field.startswith("configuration.") for field in report["missing_fields"])
    assert report["eligible_for_verdict"] is False
    assert report["failed_criteria"] == []
    assert report["criteria"] == {}
    return report


@pytest.mark.parametrize("position", range(6))
def test_i7_nominal_masks_all_positions(position):
    result = analyzer.analyze_runs(_records(), plan=_plan())
    assert result["classification"] == "H-DS"
    report = result["runs"][position]
    assert report["status"] == "ok"
    assert report["eligible_for_verdict"] is True
    assert report["missing_fields"] == report["failed_criteria"] == []


@pytest.mark.parametrize("position", range(6))
@pytest.mark.parametrize("word", BAD_POST_WORDS)
def test_i2_post_mask_fail_closed_all_positions(position, word):
    records = _records()
    device = records[position]["post_run_snapshot"]["snapshot"]["device_info"][0]
    _assign(device["smbus_telem"], "ENABLED_MAX_ARB", word)
    _configuration_failure(analyzer.analyze_runs(records, plan=_plan()), position)


@pytest.mark.parametrize("position", range(6))
@pytest.mark.parametrize("fault", ["missing", "device", "duplicate", "serial", "smbus"])
def test_i2_missing_or_unresolved_post_device(position, fault):
    records = _records()
    record = records[position]
    devices = record["post_run_snapshot"]["snapshot"]["device_info"]
    if fault == "missing":
        record.pop("post_run_snapshot")
    elif fault == "device":
        devices[0]["board_info"]["bus_id"] = "0000:00:00.0"
    elif fault == "duplicate":
        devices.append(copy.deepcopy(devices[0]))
    elif fault == "serial":
        devices[0]["board_info"]["serial"] = "other-board"
    else:
        devices[0].pop("smbus_telem")
    _configuration_failure(analyzer.analyze_runs(records, plan=_plan()), position)


@pytest.mark.parametrize("position", range(6))
def test_i2_selected_device_only_and_snapshot_numeric_word(position):
    records = _records()
    devices = records[position]["post_run_snapshot"]["snapshot"]["device_info"]
    other = copy.deepcopy(devices[0])
    other["board_info"] = {"board_id": "other-board", "bus_id": "0000:00:00.0"}
    other["smbus_telem"]["ENABLED_MAX_ARB"] = "0x0"
    devices.insert(0, other)
    devices[1]["smbus_telem"]["ENABLED_MAX_ARB"] = 0x1D1
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "H-DS"
    # A nominal unrelated device cannot rescue the selected device's mask.
    devices[0]["smbus_telem"]["ENABLED_MAX_ARB"] = "0x1d1"
    devices[1]["smbus_telem"]["ENABLED_MAX_ARB"] = "0x1d0"
    _configuration_failure(analyzer.analyze_runs(records, plan=_plan()), position)


@pytest.mark.parametrize("position", range(6))
@pytest.mark.parametrize("offset", OUTSIDE_TIMES)
@pytest.mark.parametrize("word", BAD_WORDS)
def test_i3_raw_mask_outside_plateau_fail_closed(position, offset, word):
    records = _records()
    record = records[position]
    row = _trace_row(offset, configuration_id=record["configuration"]["id"])
    _assign(row, "enabled_max_arb_raw", word)
    record["trace"]["samples"].append(row)
    record["trace"]["samples"].sort(key=lambda item: item["timestamp_monotonic_ns"])
    _rehash(record)
    report = _configuration_failure(analyzer.analyze_runs(records, plan=_plan()), position)
    assert report["glitch_count"] == 0


@pytest.mark.parametrize("position", range(6))
@pytest.mark.parametrize("offset", [60, 178])
@pytest.mark.parametrize("kind", ["all_zero", "mask_only", "nonnominal", "missing", "malformed"])
def test_i3_i4_plateau_glitch_precedes_configuration_and_criteria(position, offset, kind):
    records = _records()
    record = records[position]
    row = next(
        row
        for row in record["trace"]["samples"]
        if row["timestamp_monotonic_ns"] == offset * 1_000_000_000
    )
    if kind == "all_zero":
        for field in analyzer.ARBITER_FIELDS:
            row[field] = "0x0"
    elif kind == "mask_only":
        row["enabled_max_arb_raw"] = "0x1d0"
    elif kind == "nonnominal":
        row.update(enabled_max_arb_raw="0x1d0", enabled_min_arb_raw="0x0")
    else:
        _assign(row, "enabled_max_arb_raw", MISSING if kind == "missing" else "invalid")
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    if kind in {"all_zero", "mask_only"}:
        assert result["classification"] == "H-DS"
        report = result["runs"][position]
        assert report["glitch_count"] == 1
        assert report["plateau"]["retained_count"] == 59
        assert report["missing_fields"] == report["failed_criteria"] == []
    else:
        _configuration_failure(result, position)


@pytest.mark.parametrize("position", range(6))
@pytest.mark.parametrize("offset", [-1, 59, 180, 190])
def test_i3_no_outside_plateau_all_zero_glitch_exemption(position, offset):
    records = _records()
    record = records[position]
    row = _trace_row(offset, configuration_id=record["configuration"]["id"])
    for field in analyzer.ARBITER_FIELDS:
        row[field] = "0x0"
    record["trace"]["samples"].append(row)
    record["trace"]["samples"].sort(key=lambda item: item["timestamp_monotonic_ns"])
    _rehash(record)
    report = _configuration_failure(analyzer.analyze_runs(records, plan=_plan()), position)
    assert report["glitch_count"] == 0


@pytest.mark.parametrize("position", range(6))
@pytest.mark.parametrize("offset", [0, 60, 180])
def test_i6_minimum_mask_keeps_existing_criterion_scope(position, offset):
    records = _records()
    record = records[position]
    row = (
        next(
            row
            for row in record["trace"]["samples"]
            if row["timestamp_monotonic_ns"] == offset * 1_000_000_000
        )
        if offset != 180
        else _trace_row(180, configuration_id=record["configuration"]["id"])
    )
    if offset == 180:
        record["trace"]["samples"].insert(-1, row)
    row["enabled_min_arb_raw"] = "0x0"
    _rehash(record)
    result = analyzer.analyze_runs(records, plan=_plan())
    report = result["runs"][position]
    assert report["missing_fields"] == []
    assert result["classification"] == ("H-OTHER" if offset == 60 else "H-DS")
    assert bool(report["failed_criteria"]) is (offset == 60)


@pytest.mark.parametrize("position", range(6))
def test_i1_pre_failure_never_loads_groups_trace_or_post(position, monkeypatch):
    records = _records()
    record = records[position]
    record["pre_run_snapshot"]["snapshot"]["device_info"][0]["limits"]["board_power_limit"] = 299
    for name in ("_load_trace", "_load_groups"):
        original = getattr(analyzer, name)

        def checked(candidate, _original=original, **kwargs):
            assert candidate is not record, "sidecar opened after authoritative pre-check failed"
            return _original(candidate, **kwargs)

        monkeypatch.setattr(analyzer, name, checked)
    record.pop("post_run_snapshot")
    report = _configuration_failure(analyzer.analyze_runs(records, plan=_plan()), position)
    assert report["trace_rows"] == []
    assert not any(
        "post_run" in field or "telemetry" in field for field in report["missing_fields"]
    )


@pytest.mark.parametrize("configuration", ["C018", "C081", "C108"])
@pytest.mark.parametrize("fault", ["nominal", "post", "pre_window", "post_window", "malformed"])
def test_i5_shared_smoke_policy_keeps_four_gates(configuration, fault):
    record = _record(configuration, 1)
    record["run"].update(
        mode="smoke", observed_duration_seconds=40.0, workload_end_monotonic_ns=40_000_000_000
    )
    record["groups"]["groups"] = [{"start_monotonic_ns": 0, "end_monotonic_ns": 40_000_000_000}]
    record["trace"]["run_end_monotonic_ns"] = 40_000_000_000
    _rehash(record)
    if fault == "post":
        record["post_run_snapshot"]["snapshot"]["device_info"][0]["smbus_telem"][
            "ENABLED_MAX_ARB"
        ] = "0x1d0"
    elif fault != "nominal":
        offset = 0 if fault == "pre_window" else 40
        record["trace"]["samples"].append(
            _trace_row(
                offset,
                configuration_id=configuration,
                enabled_max_arb_raw=("invalid" if fault == "malformed" else "0x1d0"),
            )
        )
        record["trace"]["samples"].sort(key=lambda item: item["timestamp_monotonic_ns"])
        _rehash(record)
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert len(result["gates"]) == 4
    assert result["affects_long_verdict"] is False
    report = result["runs"][0]["run"]
    if fault == "nominal":
        assert all(item["pass"] is True for item in result["gates"])
    else:
        assert any(field.startswith("configuration.") for field in report["missing_fields"])
        assert result["policy"]["plan_revision_required"] is True


def test_i7_i8_archived_long_outcome_and_canonical_bytes_are_unchanged():
    records = [
        ARCHIVE / "runs" / f"run{number}-{config}" / f"run-issue146-long-run{number}-{config}.json"
        for number, config in enumerate(ORDER, 1)
    ]
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "H-DS"
    for report in result["runs"]:
        assert report["status"] == "ok" and report["eligible_for_verdict"] is True
        assert report["missing_fields"] == report["failed_criteria"] == []
    raw = (dumps(analyzer._safe_public(result), indent=2) + "\n").encode()
    original = (ARCHIVE / "analysis.json").read_bytes()
    assert hashlib.sha256(original).hexdigest() == (
        "2d1537fea1d40d8964b0e0fa145808274b733c89c6cd382f5964103314656445"
    )
    assert raw == original


def test_i7_archived_revision5_smoke_replay_keeps_twelve_passes():
    records = [
        ARCHIVE / "smoke/revision5" / config / f"run-issue146-smoke-rev5-{config}.json"
        for config in ("c018", "c081", "c108")
    ]
    result = analyzer.analyze_smoke(records, plan=_plan())
    assert len(result["gates"]) == 12
    assert all(item["pass"] is True for item in result["gates"])
    assert result["affects_long_verdict"] is False
