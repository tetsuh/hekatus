"""Board-free multi-run Issue #137 analyzer tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from enodia.tt.bench import power_trace_issue137
from tools import analyze_issue137 as analysis
from tools import issue137 as runner

ROOT = Path(__file__).parents[1]
PLAN_PATH = ROOT / "docs/measurements/2026-10-18-host-newton-schulz-issue137-plan.json"


def _limits(**overrides: float) -> dict[str, float]:
    values = {
        "board_power_limit_w": 300.0,
        "tdc_limit": 200.0,
        "tdp_limit_w": 150.0,
        "therm_trip_l1_limit_c": 90.0,
        "asic_fmax_mhz": 1350.0,
    }
    values.update(overrides)
    return values


def _owner_identity() -> tuple[str, str]:
    scope = runner.load_plan(PLAN_PATH)["measurement_scope"]
    assert isinstance(scope, dict)
    return scope["board_serial"], scope["pci_bus_id"]


def _pre_snapshot(
    limits: dict[str, float] | None = None,
    *,
    board_id: str | None = None,
    bus_id: str | None = None,
) -> dict[str, object]:
    owner_board, owner_bus = _owner_identity()
    board_id = owner_board if board_id is None else board_id
    bus_id = owner_bus if bus_id is None else bus_id
    return {
        "schema": analysis.ISSUE137_SNAPSHOT_SCHEMA,
        "stage": "pre_run",
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "status": "ok",
        "returncode": 0,
        "snapshot": {
            "device_info": [
                {
                    "board_info": {
                        "board_id": board_id,
                        "serial": board_id,
                        "bus_id": bus_id,
                    },
                    "limits": dict(limits or _limits()),
                    "smbus_telem": {},
                    "telemetry": {"power": 50, "asic_temperature": 50},
                }
            ]
        },
    }


def _row(
    timestamp: int,
    *,
    aiclk_mhz: float = 1350.0,
    board_power_w: float = 250.0,
    power_w: float = 100.0,
    current_a: float = 100.0,
    asic_temp_c: float = 70.0,
    voltage_v: float = 0.8,
    fan_rpm: float = 1000.0,
) -> dict[str, object]:
    return {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "power_w": power_w,
        "aiclk_mhz": aiclk_mhz,
        "asic_temp_c": asic_temp_c,
        "timestamp_monotonic_ns": timestamp,
        "snapshot_elapsed_ns": 10,
        "current_a": current_a,
        "voltage_v": voltage_v,
        "board_power_w": board_power_w,
        "fan_rpm": fan_rpm,
        "heartbeat": 1,
        "therm_trip_count": 0,
        "throttle_status": None,
        "throttle_reason": None,
        "gddr_0_1_temp_c": "0x36363234",
        "gddr_2_3_temp_c": "0x32343432",
        "gddr_4_5_temp_c": "0x30323638",
        "gddr_6_7_temp_c": "0x34342e30",
    }


def _trace(
    *,
    aiclk_mhz: float = 1350.0,
    board_power_w: float = 250.0,
    power_w: float = 100.0,
    current_a: float = 100.0,
    asic_temp_c: float = 70.0,
    fan_rpm: float = 1000.0,
    weighted_second_values: tuple[float, float] | None = None,
    terminal: bool = True,
) -> dict[str, object]:
    values = {
        "aiclk_mhz": aiclk_mhz,
        "board_power_w": board_power_w,
        "power_w": power_w,
        "current_a": current_a,
        "asic_temp_c": asic_temp_c,
        "fan_rpm": fan_rpm,
    }
    first = _row(0, **values)
    plateau_a = dict(values)
    plateau_b = dict(values)
    if weighted_second_values is not None:
        plateau_a["board_power_w"] = weighted_second_values[0]
        plateau_b["board_power_w"] = weighted_second_values[1]
    rows = [
        first,
        _row(60_000_000_000, **plateau_a),
        _row(90_000_000_000, **plateau_b),
    ]
    if terminal:
        rows.append(_row(120_000_000_000, **plateau_b))
    payload = {
        "schema": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA,
        "schema_version": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA_VERSION,
        "columns": list(power_trace_issue137.ISSUE137_POWER_TRACE_COLUMNS),
        "samples": rows,
        "run_start_monotonic_ns": 0,
        "run_end_monotonic_ns": 120_000_000_000,
        "path": "inline-trace.csv",
        "unit_provenance": power_trace_issue137.ISSUE137_UNIT_PROVENANCE,
    }
    payload["sha256"] = hashlib.sha256(analysis._canonical_trace_bytes(payload)).hexdigest()
    payload["byte_count"] = len(analysis._canonical_trace_bytes(payload))
    return payload


def _canonical_environment(configuration_id: str) -> dict[str, object]:
    plan = runner.load_plan(PLAN_PATH)
    scope = plan["measurement_scope"]
    assert isinstance(scope, dict)
    configuration = runner.configuration(plan, configuration_id)
    return runner.build_environment_record(
        {
            "board": {
                "serial": scope["board_serial"],
                "board_id": scope["board_serial"],
                "bus_id": scope["pci_bus_id"],
            },
            "firmware": {"bundle": "19.6.0.0"},
            "kernel": "fixture-kernel",
            "kmd_version": "fixture-driver",
            "image": "fixture-image@sha256:" + "a" * 64,
            "harness_commit": "b" * 40,
            "device_node": scope["device_node"],
            "harness_dirty": False,
            "harness_pushed": True,
            "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        },
        configuration,
    )


def _refresh_trace_hash(record: dict[str, object]) -> None:
    trace = record["trace"]
    assert isinstance(trace, dict)
    trace["sha256"] = hashlib.sha256(analysis._canonical_trace_bytes(trace)).hexdigest()
    trace["byte_count"] = len(analysis._canonical_trace_bytes(trace))


def _record(
    configuration_id: str,
    run_number: int,
    *,
    trace: dict[str, object] | None = None,
    limits: dict[str, float] | None = None,
    mode: str = "long",
    verdict_evidence: bool = True,
    diagnostic_only: bool = False,
    status: str = "ok",
) -> dict[str, object]:
    plan = runner.load_plan(PLAN_PATH)
    configuration = runner.configuration(plan, configuration_id)
    target = 120.0 if mode == "long" else 20.0
    guard = 170.0 if mode == "long" else 70.0
    first = 0
    workload_end = round(target * 1_000_000_000.0)
    trace_payload = dict(trace if trace is not None else _trace())
    trace_payload["path"] = f"power-fixture-{configuration_id}-{run_number}.csv"
    if trace_payload.get("schema") == power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA:
        trace_payload.setdefault("unit_provenance", power_trace_issue137.ISSUE137_UNIT_PROVENANCE)
    trace_payload["sha256"] = hashlib.sha256(analysis._canonical_trace_bytes(trace_payload)).hexdigest()
    trace_payload["byte_count"] = len(analysis._canonical_trace_bytes(trace_payload))
    environment = _canonical_environment(configuration_id)
    result = {
        "record_schema": runner.ISSUE137_DEVICE_RESULT_SCHEMA,
        "status": "ok",
        "run_id": f"wrapper-fixture-{configuration_id.lower()}-{run_number}",
        "mode": mode,
        "load_mode": runner.ISSUE137_LOAD_MODE,
        "target_seconds": target,
        "guard_seconds": guard,
        "overrun_limit_seconds": 10.0,
        "launch_count": 1,
        "first_launch_start_monotonic_ns": first,
        "workload_end_monotonic_ns": workload_end,
        "last_launch_end_monotonic_ns": workload_end,
        "overrun_seconds": 0.0,
        "core_plan": configuration.core_plan.as_record(),
        "plan_sha256": runner._plan_hash(plan),
        "environment": environment,
    }
    return {
        "record_schema": "issue137-measurement-v1",
        "record_revision": 2,
        "issue": "#137",
        "status": status,
        "plan_sha256": runner._plan_hash(plan),
        "run": {
            "run_id": f"fixture-{configuration_id.lower()}-{run_number}",
            "run_number": run_number,
            "mode": mode,
            "load_mode": runner.ISSUE137_LOAD_MODE,
            "timing": {"load_seconds": target, "guard_seconds": guard},
            "target_seconds": target,
            "guard_seconds": guard,
            "overrun_limit_seconds": 10.0,
            "launch_count": 1,
            "workload_start_monotonic_ns": first,
            "workload_end_monotonic_ns": workload_end,
            "last_launch_end_monotonic_ns": workload_end,
            "overrun_seconds": 0.0,
        },
        "configuration": configuration.as_record(),
        "verdict_evidence": verdict_evidence,
        "diagnostic_only": diagnostic_only,
        "workload_started": mode == "long",
        "environment": environment,
        "runner_result": result,
        "pre_run_snapshot": _pre_snapshot(limits),
        "trace": trace_payload,
        "provenance": {
            "runner_result_path": f"runner-result-{configuration_id}-{run_number}.json",
            "wrapper_run_id": result["run_id"],
            "unit_provenance": power_trace_issue137.ISSUE137_UNIT_PROVENANCE,
        },
        "workload_start_monotonic_ns": first,
        "workload_end_monotonic_ns": workload_end,
        "run_end_monotonic_ns": workload_end,
        "workload_actual_end_monotonic_ns": workload_end,
    }


def _trace_set(
    *,
    hypothesis: str = "H-BPL",
    p1_aiclk: float = 1350.0,
    p2_aiclk: tuple[float, float] = (1100.0, 900.0),
    p3_reverse: bool = False,
    limits_by_config: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, object]]:
    values = {
        "C018": {"board_power_w": 250.0, "power_w": 100.0, "current_a": 100.0, "asic_temp_c": 70.0},
        "C081": {"board_power_w": 250.0, "power_w": 100.0, "current_a": 100.0, "asic_temp_c": 70.0},
        "C108": {"board_power_w": 250.0, "power_w": 100.0, "current_a": 100.0, "asic_temp_c": 70.0},
    }
    signal = {"H-BPL": "board_power_w", "H-TDC": "current_a", "H-TDP": "power_w", "H-THM": "asic_temp_c"}[hypothesis]
    high = dict(values["C081"])
    high[signal] = {"board_power_w": 300.0, "current_a": 200.0, "power_w": 150.0, "asic_temp_c": 90.0}[signal]
    values["C081"] = dict(high)
    values["C108"] = dict(high)
    ordered = ["C018", "C108", "C054", "C081", "C081", "C054", "C108", "C018"]
    result: list[dict[str, object]] = []
    for index, config in enumerate(ordered, start=1):
        if config == "C054":
            result.append(
                _record(
                    config,
                    index,
                    trace=_trace(aiclk_mhz=1100.0),
                    mode="long",
                    verdict_evidence=False,
                    diagnostic_only=True,
                )
            )
            continue
        if config == "C018":
            clock = p1_aiclk
        else:
            clock = p2_aiclk[0 if index <= 4 else 1] + (50.0 if config == "C081" else 0.0)
        if p3_reverse and config in {"C081", "C108"} and index <= 4:
            clock = 900.0 if config == "C081" else 1100.0
        result.append(
            _record(
                config,
                index,
                trace=_trace(aiclk_mhz=clock, **values[config]),
                limits=(limits_by_config or {}).get(config),
            )
        )
    return result


def _analyze(records: list[dict[str, object]]) -> dict[str, object]:
    return analysis.analyze_runs(records, plan=analysis.load_plan(PLAN_PATH))


def test_analyzer_contract_is_importable() -> None:
    assert analysis.ANALYSIS_SCHEMA == "issue137-analysis-v1"


def test_source_path_rejects_traversal_outside_record_directory(tmp_path: Path) -> None:
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    with pytest.raises(analysis.AnalysisError, match="file name"):
        analysis._source_path(
            "../outside.json", base_dir=record_dir, purpose="pre-run snapshot"
        )


def test_source_path_rejects_symlink_escape(tmp_path: Path) -> None:
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (record_dir / "snapshot.json").symlink_to(outside)
    with pytest.raises(analysis.AnalysisError, match="escapes"):
        analysis._source_path(
            "snapshot.json", base_dir=record_dir, purpose="pre-run snapshot"
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+1.25e+2", 125.0),
        ("-.5E-2", -0.005),
        ("1.", 1.0),
        (".75", 0.75),
        ("  -0  ", -0.0),
    ],
)
def test_limit_parser_numeric_edge_cases(value: str, expected: float) -> None:
    parsed, status, error = analysis._parse_unit_value(value, expected_unit="W")
    assert parsed == pytest.approx(expected)
    assert status == "source_field"
    assert error is None


@pytest.mark.parametrize("value", ["nan", "inf", "1_000", "1e", "0x10"])
def test_limit_parser_rejects_noncanonical_numeric_values(value: str) -> None:
    parsed, _status, error = analysis._parse_unit_value(value, expected_unit="W")
    assert parsed is None
    assert error


def test_limit_parser_rejects_nonfinite_numeric_values() -> None:
    with pytest.raises(analysis.AnalysisError, match="finite"):
        analysis._parse_unit_value("1e309", expected_unit="W")


def test_supports_each_preregistered_hypothesis() -> None:
    for hypothesis in ("H-BPL", "H-TDC", "H-TDP", "H-THM"):
        result = _analyze(_trace_set(hypothesis=hypothesis))
        assert result["classification"] == hypothesis
        assert result["hypotheses"][hypothesis]["classification"] == "supported"
        assert result["preconditions"]["half_1"]["P3"]["pass"] is True


def test_multiple_support_is_exactly_one_final_label() -> None:
    records = _trace_set(hypothesis="H-BPL")
    for record in records:
        if record["configuration"]["id"] in {"C081", "C108"}:
            for row in record["trace"]["samples"][1:]:
                row["current_a"] = 200.0
            _refresh_trace_hash(record)
    result = _analyze(records)
    assert result["classification"] == "multiple"
    assert result["supported_hypotheses"] == ["H-BPL", "H-TDC"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [("p1", "P1-below-limit-unthrottled"), ("p2", "P2-above-limit-throttled"), ("p3", "P3-monotonic")],
)
def test_failed_precondition_is_named_in_h_other(mutation: str, expected: str) -> None:
    kwargs: dict[str, object] = {}
    if mutation == "p1":
        kwargs["p1_aiclk"] = 1200.0
    elif mutation == "p2":
        kwargs["p2_aiclk"] = (1350.0, 900.0)
    else:
        kwargs["p3_reverse"] = True
    result = _analyze(_trace_set(**kwargs))
    assert result["classification"] == "H-OTHER"
    assert expected in result["failed_preconditions"]
    assert expected in result["classification_reason"]


def test_h_other_reports_no_support_without_failed_precondition() -> None:
    records = _trace_set(hypothesis="H-BPL")
    for record in records:
        if record["configuration"]["id"] in {"C081", "C108"}:
            for row in record["trace"]["samples"]:
                row["board_power_w"] = 250.0
            _refresh_trace_hash(record)
    result = _analyze(records)
    assert result["classification"] == "H-OTHER"
    assert result["failed_preconditions"] == []
    assert "no supported hypothesis" in result["classification_reason"]


def test_missing_verdict_signal_or_limit_is_not_evaluable_and_named() -> None:
    records = _trace_set(hypothesis="H-BPL")
    next(record for record in records if record["configuration"]["id"] == "C081")["trace"]["samples"][1].pop("board_power_w")
    next(record for record in records if record["configuration"]["id"] == "C108")["pre_run_snapshot"] = _pre_snapshot(
        {key: value for key, value in _limits().items() if key != "board_power_limit_w"}
    )
    result = _analyze(records)
    assert result["classification"] == "not_evaluable"
    assert any("board_power_w" in field for field in result["missing_fields"])
    assert any("board_power_limit_w" in field for field in result["missing_fields"])


def test_c054_and_smoke_calibration_are_exploratory_only() -> None:
    records = _trace_set(hypothesis="H-BPL")
    records.append(_record("C018", 1, mode="smoke", verdict_evidence=False, diagnostic_only=True))
    records.append(
        _record(
            "C054",
            3,
            trace={"schema": "broken", "samples": []},
            verdict_evidence=False,
            diagnostic_only=True,
        )
    )
    result = _analyze(records)
    assert result["classification"] == "H-BPL"
    assert result["exploratory"]["ignored_non_verdict_count"] >= 1
    assert all(run["configuration_id"] != "C054" for run in result["verdict_runs"])


def test_weighted_stats_and_gddr_channels_do_not_aggregate_packed_words() -> None:
    records = _trace_set(hypothesis="H-BPL")
    c018 = next(record for record in records if record["configuration"]["id"] == "C018")
    c018["trace"] = _trace(weighted_second_values=(280.0, 300.0))
    result = _analyze(records)
    run = next(run for run in result["runs"] if run["run_number"] == 1)
    assert run["statistics"]["board_power_w"]["time_weighted_mean"] == pytest.approx(290.0)
    assert run["statistics"]["fan_rpm"]["time_weighted_mean"] == pytest.approx(1000.0)
    assert run["statistics"]["gddr_channels"]["GDDR_0"]["bottom_c"]["time_weighted_mean"] == pytest.approx(52.0)
    assert "gddr_0_1_temp_c" not in run["statistics"]


def test_terminal_coverage_is_required_without_extrapolation() -> None:
    records = _trace_set(hypothesis="H-BPL")
    c018 = next(record for record in records if record["configuration"]["id"] == "C018")
    c018["trace"] = _trace(terminal=False)
    result = _analyze(records)
    assert result["classification"] == "not_evaluable"
    assert any("terminal" in field or "coverage" in field for field in result["missing_fields"])


def test_per_run_snapshot_limits_are_used_and_differences_reported() -> None:
    records = _trace_set(hypothesis="H-BPL", limits_by_config={"C108": _limits(board_power_limit_w=310.0)})
    for record in records:
        if record["configuration"]["id"] == "C108":
            for row in record["trace"]["samples"]:
                row["board_power_w"] = 310.0
            _refresh_trace_hash(record)
    result = _analyze(records)
    assert result["classification"] == "H-BPL"
    c108 = [run for run in result["runs"] if run["configuration_id"] == "C108"]
    assert all(run["limits"]["board_power_limit_w"]["value"] == 310.0 for run in c108)
    assert all(run["limits"]["board_power_limit_w"]["differs_from_plan"] for run in c108)


def test_unknown_gddr_source_is_diagnostic_not_a_scalar() -> None:
    records = _trace_set(hypothesis="H-BPL")
    records[0]["trace"]["gddr_source_established"] = False
    result = _analyze(records)
    run = result["runs"][0]
    assert result["classification"] == "not_evaluable"
    assert run["statistics"]["gddr_channels"] == {}
    assert run["exploratory"]["gddr"]["status"] == "not_evaluable"
    assert "time_weighted_mean" not in json.dumps(run["exploratory"]["gddr"])


def test_absent_or_duplicate_ordered_verdict_evidence_fails_closed() -> None:
    records = [record for record in _trace_set(hypothesis="H-BPL") if record["run"]["run_number"] != 8]
    result = _analyze(records)
    assert result["classification"] == "not_evaluable"
    assert any("run_8.C018.ordered_evidence" in field for field in result["missing_fields"])

    records = _trace_set(hypothesis="H-BPL")
    records[-1]["run"]["run_number"] = 1
    result = _analyze(records)
    assert result["classification"] == "not_evaluable"
    assert any("duplicate_evidence" in field for field in result["missing_fields"])


def test_missing_non_verdict_telemetry_is_retained_as_diagnostic() -> None:
    records = _trace_set(hypothesis="H-BPL")
    c018 = next(record for record in records if record["configuration"]["id"] == "C018")
    for row in c018["trace"]["samples"]:
        row.pop("voltage_v")
        row.pop("fan_rpm")
    _refresh_trace_hash(c018)
    result = _analyze(records)
    assert result["classification"] == "H-BPL"
    first_run = next(run for run in result["runs"] if run["run_number"] == 1)
    assert first_run["statistics"].get("voltage_v") is None
    assert "voltage_v" in first_run["exploratory"]["missing_fields"]
    assert "fan_rpm" in first_run["exploratory"]["missing_fields"]


def test_cli_integration_consumes_runner_shaped_fixture_files(tmp_path: Path, capsys) -> None:
    records = _trace_set(hypothesis="H-BPL")
    record_paths: list[Path] = []
    for record in records:
        number = record["run"]["run_number"]
        trace_path = tmp_path / f"trace-{number}.csv"
        power_trace_issue137.write_issue137_power_trace(trace_path, record["trace"]["samples"])
        snapshot_path = tmp_path / f"pre-{number}.json"
        snapshot_path.write_text(json.dumps(record["pre_run_snapshot"]), encoding="utf-8")
        fixture = deepcopy(record)
        fixture["run_start_monotonic_ns"] = 0
        fixture["run_end_monotonic_ns"] = 120_000_000_000
        trace_bytes = trace_path.read_bytes()
        fixture["trace"] = {
            "path": trace_path.name,
            "schema": power_trace_issue137.ISSUE137_POWER_TRACE_SCHEMA,
            "unit_provenance": power_trace_issue137.ISSUE137_UNIT_PROVENANCE,
            "sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "byte_count": len(trace_bytes),
        }
        fixture["pre_run_snapshot"] = {"path": snapshot_path.name}
        record_path = tmp_path / f"run-{number}.json"
        record_path.write_text(json.dumps(fixture), encoding="utf-8")
        record_paths.append(record_path)
    output = tmp_path / "analysis.json"
    argv = ["--output", str(output)]
    for path in record_paths:
        argv.extend(("--record", str(path)))
    assert analysis.main(argv) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["classification"] == "H-BPL"
    assert all(Path(run["source"]["trace"]).name == run["source"]["trace"] for run in result["runs"])
    assert capsys.readouterr().out == ""
