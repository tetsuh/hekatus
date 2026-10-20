"""Board-free Issue #146 analyzer contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from enodia.tt.bench import power_trace_issue146
from tools import analyze_issue146 as analyzer
from tools import issue137 as issue137_runner

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json"
GDDR_WORDS = {
    "gddr_0_1_temp_c": "0x36363234",
    "gddr_2_3_temp_c": "0x32343432",
    "gddr_4_5_temp_c": "0x30323638",
    "gddr_6_7_temp_c": "0x34342e30",
}


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _trace_row(offset: int, *, configuration_id: str, **changes: object) -> dict[str, object]:
    values: dict[str, object]
    if configuration_id == "C018":
        values = {
            "aiclk_mhz": 1350.0,
            "board_power_w": 250.0,
            "power_w": 100.0,
            "current_a": 100.0,
        }
        arb_max = "0x80546"
    elif configuration_id == "C081":
        values = {
            "aiclk_mhz": 1200.0,
            "board_power_w": 300.0,
            "power_w": 140.0,
            "current_a": 180.0,
        }
        arb_max = "0x704b0"
    else:
        values = {
            "aiclk_mhz": 1000.0,
            "board_power_w": 300.0,
            "power_w": 140.0,
            "current_a": 200.0,
        }
        arb_max = "0x703e8"
    values.update(
        {
            "timestamp_utc": "2026-10-20T00:00:00+00:00",
            "asic_temp_c": 70.0,
            "timestamp_monotonic_ns": offset * 1_000_000_000,
            "snapshot_elapsed_ns": 10,
            "voltage_v": 0.8,
            "fan_rpm": 1000.0,
            "heartbeat": 1,
            "therm_trip_count": 0,
            "throttle_status": None,
            "throttle_reason": None,
            **GDDR_WORDS,
            "aiclk_arb_min_raw": "0x10546",
            "aiclk_arb_max_raw": arb_max,
            "enabled_min_arb_raw": "0x3",
            "enabled_max_arb_raw": "0x1d1",
        }
    )
    values.update(changes)
    return values


def _trace(configuration_id: str, **changes: object) -> dict[str, object]:
    rows = [_trace_row(0, configuration_id=configuration_id)]
    rows.extend(
        _trace_row(offset, configuration_id=configuration_id, **changes)
        for offset in range(10, 40, 2)
    )
    rows.extend(
        _trace_row(offset, configuration_id=configuration_id, **changes)
        for offset in range(60, 180, 2)
    )
    rows.append(_trace_row(190, configuration_id=configuration_id, **changes))
    return {
        "schema": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA,
        "schema_version": power_trace_issue146.ISSUE146_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(power_trace_issue146.ISSUE146_POWER_TRACE_COLUMNS),
        "samples": rows,
        "run_start_monotonic_ns": 0,
        "run_end_monotonic_ns": 180_000_000_000,
        "actual_end_monotonic_ns": 190_000_000_000,
        "unit_provenance": power_trace_issue146.ISSUE146_UNIT_PROVENANCE,
    }


def _snapshot(configuration_id: str, *, board_power_limit: object = 300) -> dict[str, object]:
    if configuration_id == "C018":
        arb_max = "0x80546"
    elif configuration_id == "C081":
        arb_max = "0x704b0"
    else:
        arb_max = "0x703e8"
    return {
        "schema": "issue146-tt-smi-snapshot-v1",
        "stage": "pre_run",
        "status": "ok",
        "returncode": 0,
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "snapshot": {
            "device_info": [
                {
                    "board_info": {
                        "board_id": "0000040331914124",
                        "serial": "0000040331914124",
                        "bus_id": "0000:09:00.0",
                    },
                    "limits": {
                        "board_power_limit": board_power_limit,
                        "asic_fmax": 1350,
                        "therm_trip_l1_limit": 90,
                    },
                    "smbus_telem": {
                        "BOARD_POWER_LIMIT": "0x12c",
                        "ENABLED_MAX_ARB": "0x1d1",
                        "AICLK_ARB_MIN": "0x10546",
                        "AICLK_ARB_MAX": arb_max,
                        **{field.upper(): value for field, value in GDDR_WORDS.items()},
                    },
                }
            ]
        },
    }


def _environment(plan: dict[str, object], configuration_id: str) -> dict[str, object]:
    configuration = issue137_runner.configuration(plan, configuration_id).as_record()
    return {
        "board": {
            "serial": "0000040331914124",
            "identity": {"board_id": "0000040331914124", "bus_id": "0000:09:00.0"},
        },
        "firmware": {"bundle": "19.6.0.0"},
        "kernel": "fixture-kernel",
        "kernel_driver_version": "fixture-driver",
        "image_digest": "sha256:" + "a" * 64,
        "harness_commit": "b" * 40,
        "device_node": "/dev/tenstorrent/1",
        "worker_core_count": configuration["worker_core_count"],
        "worker_core_coordinates": configuration["worker_core_coordinates"],
        "total_blocks": configuration["total_blocks"],
        "outer_repetitions": 16,
        "launches_per_sync": 8,
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
    }


def _record(configuration_id: str, run_number: int, **changes: object) -> dict[str, object]:
    plan = _plan()
    configuration = issue137_runner.configuration(plan, configuration_id).as_record()
    trace = _trace(configuration_id)
    trace.update(changes.pop("trace", {}))  # type: ignore[arg-type]
    trace["sha256"] = hashlib.sha256(analyzer.canonical_trace_bytes(trace)).hexdigest()
    trace["byte_count"] = len(analyzer.canonical_trace_bytes(trace))
    groups = {
        "schema": analyzer.GROUPS_SCHEMA,
        "schema_version": 1,
        "first_launch_start_monotonic_ns": 0,
        "groups": [{"start_monotonic_ns": 0, "end_monotonic_ns": 190_000_000_000}],
    }
    groups["sha256"] = hashlib.sha256(analyzer.canonical_groups_bytes(groups)).hexdigest()
    groups["byte_count"] = len(analyzer.canonical_groups_bytes(groups))
    record: dict[str, object] = {
        "record_schema": analyzer.ISSUE146_RECORD_SCHEMA,
        "record_revision": 1,
        "issue": "#146",
        "status": "ok",
        "plan_sha256": analyzer.plan_sha256(plan),
        "run": {
            "run_id": f"fixture-{configuration_id.lower()}-{run_number}",
            "run_number": run_number,
            "mode": "long",
            "load_mode": "host-duration-bounded",
            "first_launch_start_monotonic_ns": 0,
            "workload_end_monotonic_ns": 180_000_000_000,
            "observed_duration_seconds": 190.0,
        },
        "configuration": configuration,
        "environment": _environment(plan, configuration_id),
        "pre_run_snapshot": _snapshot(configuration_id),
        "post_run_snapshot": _snapshot(configuration_id),
        "trace": trace,
        "groups": groups,
        "provenance": {"unit_provenance": power_trace_issue146.ISSUE146_UNIT_PROVENANCE},
    }
    record.update(changes)
    return record


def _records() -> list[dict[str, object]]:
    order = ["C018", "C108", "C081", "C081", "C108", "C018"]
    return [_record(configuration_id, number) for number, configuration_id in enumerate(order, 1)]


def _rehash(record: dict[str, object]) -> None:
    trace = record["trace"]
    groups = record["groups"]
    assert isinstance(trace, dict)
    assert isinstance(groups, dict)
    trace["sha256"] = hashlib.sha256(analyzer.canonical_trace_bytes(trace)).hexdigest()
    trace["byte_count"] = len(analyzer.canonical_trace_bytes(trace))
    groups["sha256"] = hashlib.sha256(analyzer.canonical_groups_bytes(groups)).hexdigest()
    groups["byte_count"] = len(analyzer.canonical_groups_bytes(groups))


def _plateau_rows(record: dict[str, object]) -> list[dict[str, object]]:
    trace = record["trace"]
    assert isinstance(trace, dict)
    rows = trace["samples"]
    assert isinstance(rows, list)
    return [row for row in rows if 60 <= row["timestamp_monotonic_ns"] // 1_000_000_000 < 180]


def test_issue146_analyzer_reports_h_ds_for_synthetic_preregistered_set() -> None:
    result = analyzer.analyze_runs(_records(), plan=_plan())
    assert result["classification"] == "H-DS"
    assert result["verdict_evidence_complete"] is True
    assert all(result["hypotheses"][name]["supported"] for name in ("H-DS",))


def test_issue146_statistics_use_sample_sd_and_nonpositive_r1() -> None:
    stats = analyzer.point_statistics([1.0, 3.0, 1.0, 3.0])
    assert stats["sample_sd"] == pytest.approx(1.154700538)
    assert stats["r1"] < 0
    assert stats["n_eff"] == 4
    assert stats["df"] == 3


def test_issue146_positive_r1_uses_effective_n_and_floor_df() -> None:
    stats = analyzer.point_statistics([0.0, 1.0, 2.0, 3.0, 4.0])
    assert stats["r1"] > 0
    assert stats["n_eff"] < stats["n"]
    assert stats["df"] == max(2, int(stats["n_eff"] // 1) - 1)


def test_issue146_precision_boundary_is_inclusive_and_pooled_is_95_percent() -> None:
    base = analyzer.point_statistics([(-1.0 if index % 2 else 1.0) for index in range(50)])
    scale = 10.0 / base["half_width"]
    boundary = analyzer.point_statistics(
        [scale * (-1.0 if index % 2 else 1.0) for index in range(50)]
    )
    assert boundary["half_width"] == pytest.approx(10.0)
    assert boundary["precision_pass"] is True
    pooled = analyzer.pooled_statistics([[1.0, 2.0], [10.0, 11.0]])
    assert pooled["confidence"] == 0.95
    assert pooled["t_quantile"] == 0.975


def test_issue146_strict_plateau_selection_has_no_held_point_or_interpolation() -> None:
    rows = [_trace_row(59, configuration_id="C018"), _trace_row(61, configuration_id="C018")]
    selected = analyzer.select_plateau_rows(rows, first_launch_start_monotonic_ns=0)
    assert [row["timestamp_monotonic_ns"] for row in selected] == [61_000_000_000]


def test_issue146_group_idle_fraction_clips_consecutive_gaps_and_recomputes_overrun() -> None:
    groups = [
        {"start_monotonic_ns": 0, "end_monotonic_ns": 65_000_000_000},
        {"start_monotonic_ns": 70_000_000_000, "end_monotonic_ns": 190_000_000_000},
    ]
    result = analyzer.group_timing_facts(groups, first_launch_start_monotonic_ns=0)
    assert result["idle_fraction"] == pytest.approx(5 / 120)
    assert result["overrun_seconds"] == pytest.approx(10)


def test_issue146_missing_arbiter_word_is_not_evaluable_telemetry() -> None:
    records = _records()
    rows = records[0]["trace"]["samples"]
    assert isinstance(rows, list)
    rows[10].pop("aiclk_arb_max_raw")
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "not_evaluable"
    assert any("telemetry" in field for field in result["missing_fields"])


def test_issue146_pre_snapshot_configuration_is_checked_before_sample_criteria() -> None:
    records = _records()
    records[0]["pre_run_snapshot"] = _snapshot("C018", board_power_limit=299)
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "not_evaluable"
    run = result["runs"][0]
    assert any("configuration" in field for field in run["missing_fields"])


def test_issue146_glitch_cap_and_exclusions_are_reported() -> None:
    records = _records()
    rows = records[0]["trace"]["samples"]
    assert isinstance(rows, list)
    plateau = _plateau_rows(records[0])
    for row in plateau[:2]:
        row["aiclk_arb_min_raw"] = "0x0"
        row["aiclk_arb_max_raw"] = "0x0"
        row["enabled_min_arb_raw"] = "0x0"
        row["enabled_max_arb_raw"] = "0x0"
    _rehash(records[0])
    result = analyzer.analyze_runs(records, plan=_plan())
    report = result["runs"][0]
    assert len(report["glitches"]) == 2
    assert report["glitch_count"] == 2


def test_issue146_three_glitches_make_run_not_evaluable() -> None:
    records = _records()
    rows = records[0]["trace"]["samples"]
    assert isinstance(rows, list)
    plateau = _plateau_rows(records[0])
    for row in plateau[:3]:
        for field in analyzer.ARBITER_FIELDS:
            row[field] = "0x0"
    _rehash(records[0])
    result = analyzer.analyze_runs(records, plan=_plan())
    assert result["classification"] == "not_evaluable"
    assert any("telemetry" in field for field in result["runs"][0]["missing_fields"])


def test_issue146_smoke_acceptance_reports_each_item_without_verdict() -> None:
    record = _record("C081", 1)
    record["run"]["mode"] = "smoke"
    record["run"]["observed_duration_seconds"] = 40.0
    result = analyzer.analyze_smoke([record], plan=_plan())
    assert result["mode"] == "smoke-diagnostic"
    assert len(result["acceptance"]) == 6
    assert result["affects_long_verdict"] is False


def test_issue146_plan_bytes_are_unchanged() -> None:
    digest = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    assert digest == "84ea24fc4533237f82a60cf7936c7f960f6cbc6449dcf427530eab34bad840b8"


def test_issue146_plan_digest_rejects_mutation_with_rebound_record_hash() -> None:
    records = _records()
    plan = _plan()
    configurations = plan["configurations"]
    assert isinstance(configurations, list)
    configurations[0]["role"] = "mutated"
    records[0]["plan_sha256"] = analyzer.plan_sha256(plan)
    with pytest.raises(analyzer.AnalysisError, match="immutable|canonical|plan"):
        analyzer.analyze_runs(records, plan=plan)


def test_issue146_pre_configuration_mismatch_does_not_load_trace(monkeypatch) -> None:
    records = _records()
    records[0]["pre_run_snapshot"] = _snapshot("C018", board_power_limit=299)
    records[0]["trace"] = {"path": "missing-malformed-trace.json"}

    original_loader = analyzer._load_trace

    def fail_if_loaded(record, **kwargs):
        if record.get("trace", {}).get("path") == "missing-malformed-trace.json":
            raise AssertionError("trace loader must not run after pre-configuration mismatch")
        return original_loader(record, **kwargs)

    monkeypatch.setattr(analyzer, "_load_trace", fail_if_loaded)
    result = analyzer.analyze_runs(records, plan=_plan())
    report = result["runs"][0]
    assert result["classification"] == "not_evaluable"
    assert any("configuration" in field for field in report["missing_fields"])
    assert not any("telemetry" in field for field in report["missing_fields"])
    assert report["trace_rows"] == []


def test_issue146_snapshot_alias_and_conflicting_serial_are_checked() -> None:
    record = _records()[0]
    snapshot = record["pre_run_snapshot"]
    assert isinstance(snapshot, dict)
    device_info = snapshot["snapshot"]["device_info"]
    assert isinstance(device_info, list)
    board_info = device_info[0]["board_info"]
    assert isinstance(board_info, dict)
    del board_info["serial"]
    result = analyzer.analyze_runs([record], plan=_plan())
    report = result["runs"][0]
    assert not any("configuration.snapshot" in field for field in report["missing_fields"])
    board_info["serial"] = "other-board"
    result = analyzer.analyze_runs([record], plan=_plan())
    assert any("pre_run_snapshot" in field and "serial" in field for field in result["runs"][0]["missing_fields"])


def test_issue146_snapshot_schema_and_duplicate_identity_fail_closed() -> None:
    record = _records()[0]
    record["pre_run_snapshot"]["schema"] = "wrong-snapshot-schema"
    result = analyzer.analyze_runs([record], plan=_plan())
    assert any("pre_run_snapshot.schema" in field for field in result["runs"][0]["missing_fields"])

    record = _records()[0]
    snapshot = record["pre_run_snapshot"]["snapshot"]
    device_info = snapshot["device_info"]
    device_info.append(dict(device_info[0]))
    result = analyzer.analyze_runs([record], plan=_plan())
    assert any("pre_run_snapshot.snapshot.identity" in field for field in result["runs"][0]["missing_fields"])


def test_issue146_snapshot_sidecar_hash_and_byte_count_are_exposed_and_checked(
    tmp_path: Path,
) -> None:
    record = _records()[0]
    snapshot = record["pre_run_snapshot"]
    assert isinstance(snapshot, dict)
    raw = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
    (tmp_path / "pre.json").write_bytes(raw)
    record["pre_run_snapshot"] = {
        "path": "pre.json",
        "schema": analyzer.ISSUE146_SNAPSHOT_SCHEMA,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    result = analyzer.analyze_runs([record_path], plan=_plan())
    report = result["runs"][0]
    assert report["source"]["pre_run_snapshot_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["source"]["pre_run_snapshot_byte_count"] == len(raw)
    (tmp_path / "pre.json").write_bytes(raw + b"tampered")
    result = analyzer.analyze_runs([record_path], plan=_plan())
    assert any("snapshot_sha256" in field for field in result["runs"][0]["missing_fields"])

    descriptor = record["pre_run_snapshot"]
    assert isinstance(descriptor, dict)
    descriptor.pop("sha256")
    descriptor.pop("byte_count")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (tmp_path / "pre.json").write_bytes(raw)
    result = analyzer.analyze_runs([record_path], plan=_plan())
    missing = result["runs"][0]["missing_fields"]
    assert "pre_run_snapshot.snapshot_sha256" in missing
    assert "pre_run_snapshot.snapshot_byte_count" in missing
