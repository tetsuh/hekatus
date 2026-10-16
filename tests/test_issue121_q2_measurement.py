"""Board-free tests for the Issue #121 Q2 plan, harness, and analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_issue121_q2 as analysis
from tools import issue121_q2 as runner

ROOT = Path(__file__).parents[1]
PLAN = ROOT / runner.ISSUE121_Q2_PLAN


def _snapshot() -> str:
    return json.dumps(
        {
            "host_info": {"hostname": "not-retained", "user": "not-retained"},
            "processes": [{"pid": 1, "command": "not-retained"}],
            "host_sw_vers": {"tt_smi": "fixture"},
            "device_info": [
                {
                    "board_info": {
                        "board_id": "fixture-board",
                        "serial": "fixture-board",
                        "board_type": "p150a",
                        "bus_id": "0000:01:00.0",
                    },
                    "firmwares": {"fw_bundle_version": "19.6.0.0"},
                    "limits": {
                        "tdp_limit": "150",
                        "board_power_limit": "300",
                        "tdc_limit": "200",
                        "therm_trip_l1_limit": "90",
                        "thm_limit": "110",
                        "asic_fmax": "1350",
                    },
                    "smbus_telem": {
                        "BOARD_POWER_LIMIT": "0x12c",
                        "TDC_LIMIT_MAX": "0xc8",
                        "THERM_TRIP_COUNT": "0x0",
                    },
                    "telemetry": {
                        "power": "50",
                        "aiclk": "1350",
                        "asic_temperature": "60",
                        "current": "4",
                        "voltage": "12",
                        "board_power": "50",
                        "fan_speed": "1000",
                        "heartbeat": "1",
                    },
                }
            ],
        }
    )


def _row(
    seconds: int,
    *,
    aiclk: float = 1350,
    power: float = 50,
    board_power: float = 50,
    current: float = 50,
    temperature: float = 60,
) -> dict[str, object]:
    return {
        "timestamp_utc": f"2026-01-01T00:00:{seconds:02d}+00:00",
        "power_w": power,
        "aiclk_mhz": aiclk,
        "asic_temp_c": temperature,
        "timestamp_monotonic_ns": seconds * 1_000_000_000,
        "snapshot_elapsed_ns": 1,
        "current_a": current,
        "voltage_v": 12,
        "board_power_w": board_power,
        "fan_rpm": 1000,
        "heartbeat": seconds,
        "therm_trip_count": 0,
        "throttle_status": None,
        "throttle_reason": None,
    }


def _trace(
    *,
    aiclk: float = 1350,
    power: float = 50,
    board_power: float = 50,
    current: float = 50,
    temperature: float = 60,
) -> dict[str, object]:
    return {
        "schema": runner.ISSUE121_Q2_TRACE_SCHEMA,
        "schema_version": 1,
        "columns": list(runner.EXTENDED_TRACE_COLUMNS),
        "samples": [
            _row(0, aiclk=1350, power=50, board_power=50, current=50, temperature=60),
            _row(
                100,
                aiclk=1350,
                power=power,
                board_power=board_power,
                current=current,
                temperature=temperature,
            ),
            _row(
                200,
                aiclk=1200,
                power=power,
                board_power=board_power,
                current=current,
                temperature=temperature,
            ),
            _row(
                300,
                aiclk=1200,
                power=power,
                board_power=board_power,
                current=current,
                temperature=temperature,
            ),
        ],
    }


def test_q2_plan_is_frozen_and_names_the_extended_contract() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["status"] == "planned"
    assert plan["board_run"] is False
    assert plan["measurement_scope"]["device_id"] == 1
    assert plan["measurement_scope"]["runs"] == 3
    assert plan["measurement_scope"]["run_duration_seconds"] == 300
    assert plan["measurement_scope"]["run_guard_seconds"] == 330
    assert plan["trace"]["mode"] == "extended-v1"
    assert plan["trace"]["legacy_columns"] == list(runner.LEGACY_POWER_TRACE_COLUMNS)
    assert plan["trace"]["columns"] == list(runner.EXTENDED_TRACE_COLUMNS)
    assert plan["source_snapshot"]["source_type"] == "prior_host_snapshot"
    assert plan["source_snapshot"]["raw_smbus_fields"]["THM_LIMIT_THROTTLE"]["value"] is None
    assert plan["revision_history"][0]["revision"] == 1
    assert plan["revision_history"][0]["date"] == "2026-10-16"
    assert plan["hypotheses"]["H-TDP"]["reference_only"].startswith("Issue #118 plateau 132-135 W")
    assert plan["hypotheses"]["H-TDP"]["plateau_proximity"]["lower_bound_at_150_w"] == 142.5
    assert "numeric_nonproximity_threshold_fraction" not in plan["hypotheses"]["H-OTHER"]
    assert plan["classification_policy"]["multiple_supported_known_hypotheses"] == "multiple"
    assert plan["prediction"]["hypotheses"]["H-TDP"].startswith("unsupported")
    assert plan["plateau"] == {
        "start_seconds": 150,
        "end_seconds": 300,
        "interval_definition": plan["plateau"]["interval_definition"],
        "requires_positive_duration_segments": True,
        "terminal_coverage_required": True,
    }


def test_snapshot_command_is_bounded_and_host_fields_are_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=_snapshot(), returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    capture = runner.capture_readonly_snapshot(stage="pre_run")

    assert calls[0][0] == ("tt-smi", "-s", "--snapshot_no_tty")
    assert calls[0][1]["timeout"] == runner.SNAPSHOT_TIMEOUT_SECONDS
    assert capture["status"] == "ok"
    assert "host_info" not in json.dumps(capture)
    assert "processes" not in json.dumps(capture)
    assert capture["snapshot"]["device_info"][0]["smbus_telem"]["TDC_LIMIT_MAX"] == "0xc8"


def test_workload_command_pins_device_one_and_guard() -> None:
    command = runner.build_workload_command(run_id="q2-run-1")

    assert "--device-id" in command
    assert command[command.index("--device-id") + 1] == "1"
    assert command[command.index("--outer-timeout-seconds") + 1] == "330"
    assert "/out/" in " ".join(command)
    assert "/home/" not in " ".join(command)


def test_run_one_persists_pre_post_snapshots_and_provenance(tmp_path: Path) -> None:
    counter = iter((0, 1_000_000, 2_000_000, 3_000_000))
    def fake_capture(*, stage: str, **_kwargs: object) -> dict[str, object]:
        timestamp = next(counter)
        return {
            "schema": runner.ISSUE121_Q2_SNAPSHOT_SCHEMA,
            "stage": stage,
            "command": list(runner.SNAPSHOT_COMMAND_TUPLE),
            "timeout_seconds": 5.0,
            "captured_at_utc": "2026-01-01T00:00:00+00:00",
            "timestamp_monotonic_start_ns": timestamp,
            "timestamp_monotonic_ns": timestamp + 1,
            "snapshot_elapsed_ns": 1,
            "status": "ok",
            "returncode": 0,
            "error_type": None,
            "stdout_sha256": "a" * 64,
            "snapshot": json.loads(_snapshot()),
        }

    workload_calls: list[tuple[tuple[str, ...], dict[str, str], float]] = []

    def fake_workload(
        command: tuple[str, ...], environment: dict[str, str], timeout: float
    ) -> dict[str, object]:
        workload_calls.append((command, environment, timeout))
        return {"returncode": 0, "status": "completed", "timed_out": False}

    record = runner.run_one(
        tmp_path,
        run_id="q2-run-1",
        board_id="fixture-board",
        bus_id="0000:01:00.0",
        image_digest="sha256:" + "a" * 64,
        harness_commit="b" * 40,
        kernel="Linux fixture",
        kernel_driver_version="2.11.0",
        workload_runner=fake_workload,
        capture_fn=fake_capture,
    )

    assert record["status"] == "ok"
    assert record["workload"]["device_id"] == 1
    assert record["environment"]["board"]["serial"] == "fixture-board"
    assert record["environment"]["firmware"]["bundle"] == "19.6.0.0"
    assert record["environment"]["kernel_driver_version"] == "2.11.0"
    assert record["environment"]["image_digest"] == "sha256:" + "a" * 64
    assert record["environment"]["harness_commit"] == "b" * 40
    assert record["trace"]["board_id"] == "fixture-board"
    assert record["trace"]["bus_id"] == "0000:01:00.0"
    assert record["snapshot_limits"]["pre"]["recognized"]["asic_fmax_mhz"]["value"] == "1350"
    assert record["snapshot_limits"]["pre"]["THM_LIMIT_THROTTLE"] is None
    assert record["raw_window"]["columns"] == list(runner.RAW_WINDOW_COLUMNS)
    assert record["snapshots"]["pre"]["path"].startswith("pre-snapshot-")
    assert record["snapshots"]["post"]["path"].startswith("post-snapshot-")
    assert len(workload_calls) == 1
    assert workload_calls[0][2] == runner.RUN_GUARD_SECONDS
    persisted = json.dumps(record)
    assert "hostname" not in persisted
    assert "username" not in persisted
    assert "/home/" not in persisted
    assert (tmp_path / record["output_paths"]["record"]).exists()


def test_snapshot_limits_preserve_raw_throttle_and_recognized_values() -> None:
    snapshot = json.loads(_snapshot())
    snapshot["device_info"][0]["smbus_telem"]["THM_LIMIT_THROTTLE"] = "0x5a"
    captured = {"stage": "post_run", "snapshot": snapshot}

    limits = runner._extract_snapshot_limits(
        captured, board_id="fixture-board", bus_id="0000:01:00.0"
    )

    assert limits["THM_LIMIT_THROTTLE"] == "0x5a"
    assert limits["raw"]["THM_LIMIT_THROTTLE"] == "0x5a"
    assert limits["recognized"]["thm_limit_throttle"]["value"] == "0x5a"
    assert limits["recognized"]["thm_limit_throttle"]["source_key"] == (
        "smbus_telem.THM_LIMIT_THROTTLE"
    )


def test_cooldown_gate_matches_issue127_power_temperature_rule() -> None:
    passing = runner.evaluate_cooldown(52.0, 61.5, observed_idle_aiclk_mhz=800.0)
    failing = runner.evaluate_cooldown(52.01, 61.5, observed_idle_aiclk_mhz=800.0)

    assert passing["verified"] is True
    assert passing["aiclk_gate"] is False
    assert passing["power_limit_w"] == 52.0
    assert passing["temperature_limit_c"] == 61.5
    assert failing["verified"] is False


@pytest.mark.parametrize(
    ("trace", "expected"),
    [
        (_trace(current=190), "H-TDC"),
        (_trace(temperature=88), "H-THM"),
        (_trace(power=145), "H-TDP"),
        (_trace(), "H-OTHER"),
    ],
    ids=["tdc", "thm", "tdp", "other"],
)
def test_analysis_classifies_each_synthetic_hypothesis(
    trace: dict[str, object], expected: str
) -> None:
    limits = {
        "tdc_limit": 200,
        "therm_trip_l1_limit_c": 90,
        "thm_shutdown_limit_c": 110,
        "tdp_limit_w": 150,
        "board_power_limit_w": 300,
        "asic_fmax_mhz": 1350,
    }
    result = analysis.analyze_trace(
        trace,
        plateau_start_seconds=150,
        plateau_end_seconds=300,
        limits=limits,
    )

    assert result["classification"] == expected
    assert result["causality"] == "correlation_is_not_causation"
    assert result["time_axis"]["column"] == "timestamp_monotonic_ns"
    assert result["plateau"]["coverage_complete"] is True


def test_analysis_reports_time_weighted_statistics_and_percentiles() -> None:
    result = analysis.analyze_trace(
        _trace(), plateau_start_seconds=150, plateau_end_seconds=300
    )
    aiclk = result["statistics"]["aiclk_mhz"]

    assert aiclk["min"] == 1200.0
    assert aiclk["max"] == 1350.0
    assert aiclk["time_weighted_mean"] == pytest.approx(1250.0)
    assert aiclk["percentiles"]["p50"] == 1200.0
    assert aiclk["percentiles"]["p95"] == 1350.0


def _issue118_trace() -> dict[str, object]:
    samples = [
        _row(
            seconds,
            aiclk=1350 if seconds == 0 else 1200,
            power=133,
            board_power=285,
            current=50,
            temperature=81,
        )
        for seconds in range(0, 301, 2)
    ]
    return {
        "schema": runner.ISSUE121_Q2_TRACE_SCHEMA,
        "schema_version": 1,
        "columns": list(runner.EXTENDED_TRACE_COLUMNS),
        "samples": samples,
    }


def _source_limits() -> dict[str, float]:
    return {
        "tdc_limit": 200,
        "therm_trip_l1_limit_c": 90,
        "thm_shutdown_limit_c": 110,
        "tdp_limit_w": 150,
        "board_power_limit_w": 300,
        "asic_fmax_mhz": 1350,
    }


def test_issue118_prediction_is_unsupported_and_keeps_two_second_onset_resolution() -> None:
    result = analysis.analyze_trace(
        _issue118_trace(),
        plateau_start_seconds=150,
        plateau_end_seconds=300,
        limits=_source_limits(),
    )

    assert result["classification"] == "H-OTHER"
    assert result["hypotheses"]["H-TDP"]["classification"] == "not_supported"
    assert result["hypotheses"]["H-THM"]["classification"] == "not_supported"
    assert result["onsets"]["aiclk_drop"]["timestamp_monotonic_ns"] == 2_000_000_000
    assert result["raw_window"]["rows"][0]["seconds_from_run_start"] == 0.0
    assert len(result["raw_window"]["rows"]) == 6
    assert any("approximately 2 second" in item for item in result["limitations"])
    board = result["statistics"]["signals"]["board_power_w"]["limit_proximity"]
    assert board["limit_key"] == "BOARD_POWER_LIMIT"
    assert board["fraction"] == pytest.approx(0.05)


def test_multiple_classification_is_explicit_for_multiple_supported_known_hypotheses() -> None:
    result = analysis.analyze_trace(
        _trace(current=190, temperature=88, power=145),
        plateau_start_seconds=150,
        plateau_end_seconds=300,
        limits=_source_limits(),
    )

    assert result["classification"] == "multiple"
    assert result["hypotheses"]["H-TDC"]["classification"] == "supported"
    assert result["hypotheses"]["H-THM"]["classification"] == "supported"
    assert result["hypotheses"]["H-TDP"]["classification"] == "supported"


def test_analysis_does_not_guess_aiclk_baseline_when_fmax_is_missing() -> None:
    result = analysis.analyze_trace(
        _issue118_trace(),
        plateau_start_seconds=150,
        plateau_end_seconds=300,
        limits={"tdp_limit_w": 150, "therm_trip_l1_limit_c": 90},
    )

    assert result["aiclk_baseline"]["asic_fmax_mhz"] is None
    assert result["onsets"]["aiclk_drop"]["status"] == "not_evaluable"
    assert result["onsets"]["aiclk_drop"]["timestamp_monotonic_ns"] is None


def test_analysis_does_not_invent_missing_throttle_limit() -> None:
    result = analysis.analyze_trace(
        _trace(temperature=88),
        plateau_start_seconds=150,
        plateau_end_seconds=300,
        limits={"asic_fmax_mhz": 1350},
    )

    comparison = result["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"]
    assert comparison["status"] == "not_evaluable"
    assert comparison["knowledge_status"] == "unknown"
    assert comparison["limit_value"] is None
    assert result["hypotheses"]["H-THM"]["classification"] == "supported"

    known = analysis.analyze_trace(
        _trace(temperature=88),
        plateau_start_seconds=150,
        plateau_end_seconds=300,
        limits={"asic_fmax_mhz": 1350, "thm_limit_throttle": 100},
    )
    assert known["hypotheses"]["H-THM"]["details"]["thm_limit_throttle"]["status"] == "pass"
