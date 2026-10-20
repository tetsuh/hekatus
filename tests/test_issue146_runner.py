"""Board-free Issue #146 runner integration contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench import power_trace_issue146
from tools import analyze_issue146, issue146

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/measurements/2026-10-20-host-newton-schulz-issue146-plan.json"
SERIAL = "0000040331914124"
BDF = "0000:09:00.0"


def test_issue146_plan_order_timing_and_command_are_bound() -> None:
    plan = issue146.load_plan(PLAN)
    assert issue146.long_run_order(plan) == ["C018", "C108", "C081", "C081", "C108", "C018"]
    assert issue146.run_timing(plan, "long") == {
        "load_seconds": 180.0,
        "guard_seconds": 230.0,
        "diagnostic_only": False,
    }
    assert issue146.run_timing(plan, "smoke") == {
        "load_seconds": 40.0,
        "guard_seconds": 90.0,
        "diagnostic_only": True,
    }
    config = issue146.configuration(plan, "C081")
    command = issue146.build_run_command(
        plan=plan,
        configuration=config,
        mode="long",
        output_dir=Path("out/issue146"),
        run_id="issue146-long-c081-3",
        run_number=3,
        approve_long_run=True,
    )
    assert "--load-core-count" in command
    assert command[command.index("--load-core-count") + 1] == "81"
    assert command[command.index("--outer-timeout-seconds") + 1] == "230"
    assert "--out" not in command


def test_issue146_rejects_identity_mismatch_before_execution() -> None:
    with pytest.raises(issue146.Issue146Error, match="device_node"):
        issue146.validate_environment_for_run({"device_node": "/dev/tenstorrent/0"})
    with pytest.raises(issue146.Issue146Error, match="serial"):
        issue146.validate_selected_identity(
            device_node="/dev/tenstorrent/1", host_device_id=1,
            container_device_id=0, serial="wrong", board_id=SERIAL, bus_id=BDF,
        )
    with pytest.raises(issue146.Issue146Error, match="container"):
        issue146.validate_selected_identity(
            device_node="/dev/tenstorrent/1", host_device_id=1,
            container_device_id=1, serial=SERIAL, board_id=SERIAL, bus_id=BDF,
        )


def _environment(config: issue146.Issue146Configuration) -> dict[str, object]:
    return {
        "board": {"serial": SERIAL, "board_id": SERIAL, "bus_id": BDF},
        "firmware": {"bundle": "19.6.0.0"},
        "kernel": "fixture-kernel",
        "kernel_driver_version": "fixture-driver",
        "image_digest": "sha256:" + "a" * 64,
        "harness_commit": "b" * 40,
        "device_node": "/dev/tenstorrent/1",
        "device_id": 1,
        "container_device_id": 0,
        "worker_core_count": config.worker_core_count,
        "worker_core_coordinates": [list(core) for core in config.core_plan.worker_cores],
        "total_blocks": config.total_blocks,
        "outer_repetitions": 16,
        "launches_per_sync": 8,
        "harness_dirty": False,
        "harness_pushed": True,
        "telemetry_sampler": {"mode": "default", "interval_seconds": 2},
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "hostname": "must-not-persist",
    }


def _snapshot(*, timestamp: int, config_id: str) -> dict[str, object]:
    if config_id == "C018":
        aiclk, board_power, current, arb_max = 1350, 250, 100, "0x80546"
    elif config_id == "C081":
        aiclk, board_power, current, arb_max = 1200, 300, 180, "0x704b0"
    else:
        aiclk, board_power, current, arb_max = 1000, 300, 200, "0x703e8"
    return {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "device_info": [{
            "board_info": {"board_id": SERIAL, "serial": SERIAL, "bus_id": BDF},
            "firmwares": {"fw_bundle_version": "19.6.0.0"},
            "limits": {"board_power_limit": "300", "asic_fmax": "1350", "therm_trip_l1_limit": "90"},
            "smbus_telem": {
                "BOARD_POWER_LIMIT": "0x12c", "ENABLED_MAX_ARB": "0x1d1",
                "AICLK_ARB_MIN": "0x10546", "AICLK_ARB_MAX": arb_max,
                "ENABLED_MIN_ARB": "0x3",
                "GDDR_0_1_TEMP": "0x36363234", "GDDR_2_3_TEMP": "0x32343432",
                "GDDR_4_5_TEMP": "0x30323638", "GDDR_6_7_TEMP": "0x34342e30",
                "THERM_TRIP_COUNT": "0x0",
            },
            "telemetry": {
                "power": str(board_power - 20), "aiclk": str(aiclk), "asic_temperature": "70",
                "current": str(current), "voltage": "0.8", "board_power": str(board_power),
                "fan_speed": "1000", "heartbeat": "1",
            },
        }],
    }


def _capture_factory(config_id: str, mode: str):
    target = 40 if mode == "smoke" else 180
    timestamps = (
        [10, 20, 30, 35, 39]
        if mode == "smoke"
        else [10, 20, 30, 60, 80, 100, 120, 140, 160, 179]
    )
    calls = {"in_run": 0}

    def capture(*, stage: str) -> dict[str, object]:
        if stage == "pre_run":
            timestamp = 0
        elif stage == "post_run":
            timestamp = target
        else:
            timestamp = timestamps[min(calls["in_run"], len(timestamps) - 1)]
            calls["in_run"] += 1
        return {
            "schema": issue146.ISSUE146_SNAPSHOT_SCHEMA, "stage": stage,
            "status": "ok", "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": timestamp * 1_000_000_000,
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(timestamp=timestamp, config_id=config_id),
        }

    return capture


def _device_result(
    plan, config, environment, mode: str, run_id: str, requested_run_id: str
) -> dict[str, object]:
    target = 40 if mode == "smoke" else 180
    groups = [{"start_monotonic_ns": 0, "end_monotonic_ns": target * 1_000_000_000}]
    group_bytes = (strict_json_dumps(groups, indent=2) + "\n").encode("utf-8")
    return {
        "record_schema": issue146.ISSUE146_DEVICE_RESULT_SCHEMA, "status": "ok",
        "run_id": run_id, "wrapper_run_id": run_id, "requested_run_id": requested_run_id,
        "mode": mode, "load_mode": issue146.ISSUE146_LOAD_MODE,
        "target_seconds": float(target), "guard_seconds": float(90 if mode == "smoke" else 230),
        "overrun_limit_seconds": 10.0, "outer_repetitions": 16, "launches_per_sync": 8,
        "first_launch_start_monotonic_ns": 0,
        "workload_end_monotonic_ns": target * 1_000_000_000,
        "observed_duration_seconds": float(target),
        "group_timings": groups,
        "groups_path": f"device-group-timings-{run_id}.json",
        "groups_sha256": __import__("hashlib").sha256(group_bytes).hexdigest(),
        "groups_byte_count": len(group_bytes),
        "power_trace_path": f"power-{run_id}.csv",
        "launch_count": 8, "core_plan": config.core_plan.as_record(),
        "plan_sha256": issue146._plan_hash(plan), "environment": environment,
    }


def test_issue146_run_one_persists_group_and_trace_bindings(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    config = issue146.configuration(plan, "C018")
    environment = _environment(config)
    result = _device_result(plan, config, environment, "smoke", "wrapper-run", "issue146-smoke-c018-1")

    def command_runner(*_args: object, **_kwargs: object):
        import time

        time.sleep(0.005)
        (tmp_path / "runner-result-wrapper-run.json").write_text(json.dumps(result), encoding="utf-8")
        (tmp_path / result["groups_path"]).write_bytes(
            (strict_json_dumps(result["group_timings"], indent=2) + "\n").encode("utf-8")
        )
        (tmp_path / result["power_trace_path"]).write_text(
            "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
            "2026-10-20T00:00:00+00:00,250,1350,70\n",
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stdout": "issue146 result -> /out/runner-result-wrapper-run.json\n"})()

    record = issue146.run_one(
        plan=plan, configuration=config, mode="smoke", output_dir=tmp_path,
        run_id="issue146-smoke-c018-1",
        options=issue146.Issue146RunOptions(
            command_runner=command_runner, environment=environment,
            capture_fn=_capture_factory("C018", "smoke"), trace_path=tmp_path / "trace.csv",
            trace_interval_seconds=0.001,
        ),
    )
    assert record["status"] == "ok"
    assert record["diagnostic_only"] is True
    assert record["groups"]["path"].endswith("-envelope.json")
    assert record["trace"]["path"] == "trace.csv"
    assert record["pre_run_snapshot"]["schema"] == issue146.ISSUE146_SNAPSHOT_SCHEMA
    assert record["post_run_snapshot"]["schema"] == issue146.ISSUE146_SNAPSHOT_SCHEMA
    assert record["pre_run_snapshot"]["byte_count"] == (tmp_path / record["pre_run_snapshot"]["path"]).stat().st_size
    assert record["post_run_snapshot"]["byte_count"] == (tmp_path / record["post_run_snapshot"]["path"]).stat().st_size
    parsed_trace = power_trace_issue146.read_issue146_power_trace(tmp_path / "trace.csv")
    assert any(row["timestamp_monotonic_ns"] == 40_000_000_000 for row in parsed_trace["samples"])
    assert (tmp_path / record["groups"]["path"]).read_text(encoding="utf-8")
    assert "hostname" not in json.dumps(record)
    assert str(tmp_path) not in json.dumps(record)


def test_issue146_six_long_records_are_accepted_by_canonical_analyzer(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    records: list[Path] = []
    previous = None
    counter = 0
    for run_number, config_id in enumerate(issue146.long_run_order(plan), 1):
        config = issue146.configuration(plan, config_id)
        environment = _environment(config)
        result = _device_result(
            plan, config, environment, "long", f"wrapper-{run_number}",
            f"issue146-long-{run_number}",
        )

        def command_runner(*_args: object, result=result, run_number=run_number, **_kwargs: object):
            import time

            nonlocal counter
            time.sleep(0.06)
            counter += 1
            name = f"runner-result-wrapper-{run_number}.json"
            (tmp_path / name).write_text(json.dumps(result), encoding="utf-8")
            (tmp_path / result["groups_path"]).write_bytes(
                (strict_json_dumps(result["group_timings"], indent=2) + "\n").encode("utf-8")
            )
            (tmp_path / result["power_trace_path"]).write_text(
                "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
                "2026-10-20T00:00:00+00:00,250,1350,70\n",
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": f"issue146 result -> /out/{name}\n"})()

        record = issue146.run_one(
            plan=plan, configuration=config, mode="long", output_dir=tmp_path,
            run_id=f"issue146-long-{run_number}",
            options=issue146.Issue146RunOptions(
                run_number=run_number, approve_long_run=True, command_runner=command_runner,
                environment=environment, capture_fn=_capture_factory(config_id, "long"),
                trace_path=tmp_path / f"trace-{run_number}.csv", trace_interval_seconds=0.001,
                preceding_pre_snapshot=previous,
                cooldown_readings=({"power_w": 50, "asic_temp_c": 60},) if run_number > 1 else (),
                cooldown_elapsed_seconds=1 if run_number > 1 else 0,
            ),
        )
        assert record["status"] == "ok"
        records.append(tmp_path / f"run-issue146-long-{run_number}.json")
        previous = _capture_factory(config_id, "long")(stage="pre_run")
    result = analyze_issue146.analyze_runs(records, plan=plan)
    assert result["classification"] == "H-DS"
    assert result["verdict_evidence_complete"] is True


def test_issue146_smoke_set_is_analysis_only(tmp_path: Path) -> None:
    plan = issue146.load_plan(PLAN)
    records: list[Path] = []
    counter = 0
    for config_id in ("C018", "C081", "C108"):
        config = issue146.configuration(plan, config_id)
        environment = _environment(config)
        run_id = f"issue146-smoke-{config_id.lower()}"
        result = _device_result(plan, config, environment, "smoke", f"wrapper-{config_id}", run_id)

        def command_runner(*_args: object, result=result, config_id=config_id, **_kwargs: object):
            import time

            nonlocal counter
            time.sleep(0.005)
            counter += 1
            name = f"runner-result-wrapper-{config_id}.json"
            (tmp_path / name).write_text(json.dumps(result), encoding="utf-8")
            (tmp_path / result["groups_path"]).write_bytes(
                (strict_json_dumps(result["group_timings"], indent=2) + "\n").encode("utf-8")
            )
            (tmp_path / result["power_trace_path"]).write_text(
                "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
                "2026-10-20T00:00:00+00:00,250,1350,70\n",
                encoding="utf-8",
            )
            return type("Completed", (), {"returncode": 0, "stdout": f"issue146 result -> /out/{name}\n"})()

        record = issue146.run_one(
            plan=plan, configuration=config, mode="smoke", output_dir=tmp_path,
            run_id=f"issue146-smoke-{config_id.lower()}",
            options=issue146.Issue146RunOptions(
                command_runner=command_runner, environment=environment,
                capture_fn=_capture_factory(config_id, "smoke"),
                trace_path=tmp_path / f"smoke-{config_id}.csv", trace_interval_seconds=0.001,
            ),
        )
        assert record["status"] == "ok"
        assert record["diagnostic_only"] is True
        records.append(tmp_path / f"run-issue146-smoke-{config_id.lower()}.json")
    result = analyze_issue146.analyze_smoke(records, plan=plan)
    assert result["mode"] == "smoke-diagnostic"
    assert result["affects_long_verdict"] is False
