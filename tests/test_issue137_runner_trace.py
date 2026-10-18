"""Board-free Issue #137 runner and GDDR trace contracts."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import threading
from pathlib import Path

import pytest

from enodia.tt.bench import power_trace_issue137 as trace
from enodia.tt.bench import run_resident
from tools import issue137 as runner

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "docs/measurements/2026-10-18-host-newton-schulz-issue137-plan.json"


def _snapshot(*, gddr: dict[str, str] | None = None) -> dict[str, object]:
    words = {
        "GDDR_0_1_TEMP": "0x36363234",
        "GDDR_2_3_TEMP": "0x32343432",
        "GDDR_4_5_TEMP": "0x30323638",
        "GDDR_6_7_TEMP": "0x34342e30",
    }
    if gddr is not None:
        words.update(gddr)
    return {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
        "device_info": [
            {
                "board_info": {
                    "board_id": "fixture-board",
                    "serial": "fixture-board",
                    "bus_id": "0000:09:00.0",
                },
                "firmwares": {"fw_bundle_version": "19.6.0.0"},
                "limits": {"asic_fmax": "1350"},
                "smbus_telem": {
                    "THERM_TRIP_COUNT": "0x0",
                    **words,
                    "MAX_GDDR_TEMP": "0x48",
                },
                "telemetry": {
                    "power": "50",
                    "aiclk": "1350",
                    "asic_temperature": "60",
                    "current": "50",
                    "voltage": "0.8",
                    "board_power": "250",
                    "fan_speed": "1000",
                    "heartbeat": "1",
                },
            }
        ],
    }


def _row() -> dict[str, object]:
    base = {
        "timestamp_utc": "2026-10-20T00:00:00+00:00",
        "power_w": 50,
        "aiclk_mhz": 1350,
        "asic_temp_c": 60,
        "timestamp_monotonic_ns": 0,
        "snapshot_elapsed_ns": 10,
        "current_a": 50,
        "voltage_v": 0.8,
        "board_power_w": 250,
        "fan_rpm": 1000,
        "heartbeat": 1,
        "therm_trip_count": 0,
        "throttle_status": None,
        "throttle_reason": None,
    }
    return {
        **base,
        "gddr_0_1_temp_c": "0x36363234",
        "gddr_2_3_temp_c": "0x32343432",
        "gddr_4_5_temp_c": "0x30323638",
        "gddr_6_7_temp_c": "0x34342e30",
    }


def test_issue137_plan_is_consumed_without_changing_its_preregistered_values() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    loaded = runner.load_plan(PLAN)

    assert loaded["record_schema"] == "issue137-premeasurement-plan-v1"
    assert [item["id"] for item in runner.plan_configurations(loaded)] == [
        "C018",
        "C054",
        "C081",
        "C108",
    ]
    assert runner.long_run_order(loaded) == [
        "C018",
        "C108",
        "C054",
        "C081",
        "C081",
        "C054",
        "C108",
        "C018",
    ]
    assert runner.run_timing(loaded, "smoke") == {
        "load_seconds": 20.0,
        "guard_seconds": 70.0,
        "diagnostic_only": True,
    }
    assert runner.run_timing(loaded, "long") == {
        "load_seconds": 120.0,
        "guard_seconds": 170.0,
        "diagnostic_only": False,
    }
    assert loaded == plan


def test_issue137_reader_defaults_to_raw_without_verified_source_provenance(tmp_path: Path) -> None:
    decoded = trace.decode_gddr_word("0x36363234")
    assert decoded["status"] == "not_evaluable"
    assert decoded["channels_c"] is None
    assert trace.source_versions_verified({"tt_smi": "6.2.1", "tt_umd": "0.9.8"}) is True
    assert trace.source_versions_verified({"tt_smi": "6.2.0", "tt_umd": "0.9.8"}) is False
    assert trace.source_versions_verified({"tt_smi": "6.2.1"}) is False
    path = tmp_path / "unverified.csv"
    trace.write_issue137_power_trace(path, [_row(), {**_row(), "timestamp_monotonic_ns": 10}])
    parsed = trace.read_issue137_power_trace(path)
    assert parsed["gddr_decoded"][0]["gddr_0_1_temp_c"]["status"] == "not_evaluable"
    assert parsed["gddr_decoded"][0]["gddr_0_1_temp_c"]["channels_c"] is None


def test_issue137_gddr_word_round_trip_preserves_all_four_die_values() -> None:
    raw = "0x36363234"
    decoded = trace.decode_gddr_word(raw, source_established=True)

    assert decoded == {
        "raw": raw,
        "unit": "degC",
        "status": "decoded",
        "channels_c": {
            "GDDR_0": {"bottom_c": 52, "top_c": 50},
            "GDDR_1": {"bottom_c": 54, "top_c": 54},
        },
    }
    row = trace.parse_issue137_snapshot(
        _snapshot(), board_id="fixture-board", bus_id="0000:09:00.0",
        timestamp_monotonic_ns=0, snapshot_elapsed_ns=10,
    )
    assert row["gddr_0_1_temp_c"] == raw
    assert trace.decode_issue137_gddr_row(
        row, source_established=True
    )["gddr_0_1_temp_c"] == decoded


def test_issue137_trace_round_trip_keeps_raw_gddr_words(tmp_path: Path) -> None:
    path = tmp_path / "power.csv"
    trace.write_issue137_power_trace(path, [_row(), {**_row(), "timestamp_monotonic_ns": 10}])
    parsed = trace.read_issue137_power_trace(path, source_established=True)

    assert parsed["valid"] is True
    assert parsed["schema"] == "issue-137-power-trace-v1"
    assert parsed["columns"] == list(trace.ISSUE137_POWER_TRACE_COLUMNS)
    assert parsed["samples"][0]["gddr_6_7_temp_c"] == "0x34342e30"
    assert parsed["gddr_decoded"][0]["gddr_6_7_temp_c"]["status"] == "decoded"
    assert trace.decode_issue137_gddr_row(
        parsed["samples"][0], source_established=True
    )["gddr_6_7_temp_c"]["channels_c"]["GDDR_7"] == {
        "bottom_c": 52,
        "top_c": 52,
    }


def test_issue137_reader_unknown_source_keeps_raw_and_marks_decode_not_evaluable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown-source.csv"
    trace.write_issue137_power_trace(path, [_row(), {**_row(), "timestamp_monotonic_ns": 10}])
    parsed = trace.read_issue137_power_trace(path, source_established=False)

    assert parsed["valid"] is True
    assert parsed["samples"][0]["gddr_0_1_temp_c"] == "0x36363234"
    assert parsed["gddr_decoded"][0]["gddr_0_1_temp_c"] == {
        "raw": "0x36363234",
        "unit": "unknown",
        "status": "not_evaluable",
        "channels_c": None,
    }
    assert not any(
        isinstance(value, (int, float))
        for value in parsed["gddr_decoded"][0]["gddr_0_1_temp_c"].values()
    )


def test_issue137_unknown_source_retains_raw_and_is_not_evaluable() -> None:
    result = trace.decode_gddr_word("0x36363234", source_established=False)

    assert result == {
        "raw": "0x36363234",
        "unit": "unknown",
        "status": "not_evaluable",
        "channels_c": None,
    }
    assert not any(isinstance(value, (int, float)) for value in result.values())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("gddr_2_3_temp_c"),
        lambda row: row.__setitem__("gddr_2_3_temp_c", "not-a-word"),
        lambda row: row.__setitem__("gddr_2_3_temp_c", "0x100000000"),
    ],
)
def test_issue137_trace_rejects_missing_or_invalid_gddr_data(tmp_path: Path, mutation) -> None:
    row = _row()
    mutation(row)
    path = tmp_path / "invalid.csv"
    if "gddr_2_3_temp_c" not in row:
        fields = list(trace.ISSUE137_POWER_TRACE_COLUMNS)
        fields.remove("gddr_2_3_temp_c")
        path.write_text(",".join(fields) + "\n", encoding="utf-8")
    else:
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="").writerow(
            [row[column] for column in trace.ISSUE137_POWER_TRACE_COLUMNS]
        )
        path.write_text(
            trace.ISSUE137_POWER_TRACE_HEADER + "\n" + buffer.getvalue() + "\n",
            encoding="utf-8",
        )
    parsed = trace.read_issue137_power_trace(path)
    assert parsed["valid"] is False
    assert parsed["errors"]


def test_issue137_run_options_reject_unknown_keys() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        runner.Issue137RunOptions(unapproved_option=True)


def test_issue137_run_one_rejects_unknown_keyword() -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    output_dir = Path("out/issue137")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        runner.run_one(
            plan=plan,
            configuration=config,
            mode="smoke",
            output_dir=output_dir,
            run_id="issue137-unknown-option",
            unapproved_option=True,
        )


def test_issue137_duration_mode_has_no_calibration_api() -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")

    with pytest.raises(runner.Issue137Error, match="calibration"):
        runner.run_timing(plan, "calibration")
    with pytest.raises(runner.Issue137Error, match="calibration"):
        runner.build_run_command(
            plan=plan,
            configuration=config,
            mode="long",
            output_dir=Path("out/issue137"),
            run_id="issue137-long-calibration",
            calibration_name="issue137-calibration.json",
            approve_long_run=True,
        )


def test_issue137_resident_legacy_path_rejects_malformed_top_level_calibration_without_attribute_error(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "malformed-calibration.json"
    payload.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration"):
        run_resident._read_load_calibration(payload)


def test_issue137_smoke_set_and_long_order_are_consumed_without_series_execution():
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C108")
    output_dir = Path("out/issue137")
    with pytest.raises(runner.Issue137Error, match="smoke set"):
        runner.run_one(
            plan=plan,
            configuration=config,
            mode="smoke",
            output_dir=output_dir,
            run_id="issue137-smoke-c108",
        )
    options = runner.Issue137RunOptions(
        run_number=1,
        approve_long_run=True,
    )
    with pytest.raises(runner.Issue137Error, match="preregistered Issue #137 order"):
        runner.run_one(
            plan=plan,
            configuration=config,
            mode="long",
            output_dir=output_dir,
            run_id="issue137-long-order",
            options=options,
        )


def test_issue137_long_command_requires_one_instance_approval(tmp_path: Path) -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")

    with pytest.raises(runner.Issue137Error, match="approval"):
        runner.build_run_command(
            plan=plan, configuration=config, mode="long", output_dir=tmp_path,
            run_id="issue137-long-1", calibration_name="calibration.json",
        )
    command = runner.build_run_command(
        plan=plan, configuration=config, mode="long", output_dir=tmp_path,
        run_id="issue137-long-1", approve_long_run=True,
    )
    assert "--load-core-count" in command
    assert command[command.index("--load-core-count") + 1] == "18"
    assert command[command.index("--load-total-blocks") + 1] == "171"
    assert command[command.index("--outer-timeout-seconds") + 1] == "170"
    assert "--calibration-json" not in command
    assert "--repeat-count" not in command
    assert "--out" not in command
    assert "--approve-long-run" not in command


def test_issue137_cooldown_does_not_use_aiclk_and_fails_closed_at_cap() -> None:
    baseline = {"power_w": 100.0, "asic_temp_c": 50.0, "aiclk_mhz": 1350.0}
    assert runner.evaluate_cooldown(baseline, {"power_w": 105.0, "asic_temp_c": 53.0, "aiclk_mhz": 800.0})["verified"] is True
    assert runner.evaluate_cooldown(baseline, {"power_w": 105.1, "asic_temp_c": 53.0, "aiclk_mhz": 1350.0})["verified"] is False
    result = runner.cooldown_gate(
        baseline,
        readings=[],
        elapsed_seconds=runner.COOLDOWN_CAP_SECONDS,
    )
    assert result["verified"] is False
    assert result["failure_reason"] == "cooldown wait cap expired"


def test_issue137_one_run_captures_pre_run_and_issue137_trace_board_free(
    tmp_path: Path,
) -> None:
    import time
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    capture_counter = 0

    def capture(*, stage: str) -> dict[str, object]:
        nonlocal capture_counter
        capture_counter += 1
        timestamp = {
            "pre_run": 0,
            "in_run": 10,
            "post_run": 20_000_000_100,
        }.get(stage, capture_counter * 10)
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": timestamp,
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        result_path = tmp_path / "runner-result.json"
        environment = _issue137_environment()
        result_path.write_text(
            json.dumps(_device_result(config, environment)), encoding="utf-8"
        )
        time.sleep(0.01)
        return SimpleNamespace(
            returncode=0,
            stdout="results     -> /out/runner-result.json\\n",
        )

    trace_path = tmp_path / "issue137-trace.csv"
    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-smoke-capture",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=_issue137_environment(),
            capture_fn=capture,
            trace_path=trace_path,
            trace_interval_seconds=0.001,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )

    assert record["status"] == "ok"
    assert record["pre_run_snapshot"]["path"] == "pre-run-snapshot-issue137-smoke-capture.json"
    parsed = trace.read_issue137_power_trace(trace_path)
    assert parsed["valid"] is True
    assert parsed["samples"][0]["gddr_0_1_temp_c"] == "0x36363234"
    assert str(tmp_path) not in json.dumps(record)


def test_issue137_record_environment_has_required_fields_and_no_host_path() -> None:
    config = runner.configuration(runner.load_plan(PLAN), "C018")
    environment = {
        "board": {"serial": "fixture-board", "board_id": "fixture-board", "bus_id": "0000:09:00.0"},
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kernel": "Linux fixture",
        "kmd_version": "2.11.0",
        "image": "image@sha256:" + "a" * 64,
        "harness_commit": "b" * 40,
        "device_node": "/dev/tenstorrent/1",
        "hostname": "must-not-persist",
        "worktree_path": "/home/private/tree",
    }
    record = runner.build_environment_record(environment, config)

    assert record["board"]["serial"] == "fixture-board"
    assert record["board"]["identity"]["bus_id"] == "0000:09:00.0"
    assert record["worker_core_count"] == 18
    assert record["total_blocks"] == 171
    assert "hostname" not in json.dumps(record)
    assert "/home/" not in json.dumps(record)


def test_issue137_run_record_redacts_host_output_path(tmp_path: Path) -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-smoke-redacted",
    )

    encoded = json.dumps(record)
    assert str(tmp_path) not in encoded
    assert "<run-output-dir>" in encoded


def _issue137_environment() -> dict[str, object]:
    return {
        "board": {
            "serial": "fixture-board",
            "board_id": "fixture-board",
            "bus_id": "0000:09:00.0",
        },
        "firmware": {"fw_bundle_version": "19.6.0.0"},
        "kernel": "Linux fixture",
        "kmd_version": "2.11.0",
        "image": "image@sha256:" + "a" * 64,
        "harness_commit": "b" * 40,
        "device_node": "/dev/tenstorrent/1",
        "harness_dirty": False,
        "harness_pushed": True,
        "telemetry_sampler": {"mode": "default", "interval_seconds": 2},
        "host_sw_vers": {"tt_smi": "6.2.1", "tt_umd": "0.9.8"},
    }


def _device_result(
    config: runner.Issue137Configuration,
    environment: dict[str, object],
    *,
    launch_count: int = 2,
) -> dict[str, object]:
    first = 100
    target_end = first + 20_000_000_000
    return {
        "record_schema": "issue137-device-result-v1",
        "status": "ok",
        "run_id": "wrapper-fresh-run-id",
        "mode": "smoke",
        "load_mode": "host-duration-bounded",
        "target_seconds": 20.0,
        "guard_seconds": 70.0,
        "overrun_limit_seconds": 10.0,
        "launch_count": launch_count,
        "first_launch_start_monotonic_ns": first,
        "workload_end_monotonic_ns": target_end,
        "last_launch_end_monotonic_ns": target_end,
        "overrun_seconds": 0.0,
        "core_plan": config.core_plan.as_record(),
        "plan_sha256": runner._plan_hash(runner.load_plan(PLAN)),
        "environment": environment,
    }


def test_issue137_execute_ingests_actual_result_and_persists_canonical_record(
    tmp_path: Path,
) -> None:
    import time
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    result_path = tmp_path / "runner-result.json"
    captures = {"pre_run": 0, "in_run": 0, "post_run": 20_000_000_100}

    def capture(*, stage: str) -> dict[str, object]:
        timestamp = captures.get(stage, 1)
        if stage == "in_run":
            captures[stage] = timestamp + 1
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": timestamp,
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        result_path.write_text(
            json.dumps(_device_result(config, environment)), encoding="utf-8"
        )
        time.sleep(0.01)
        return SimpleNamespace(returncode=0, stdout="results     -> /out/runner-result.json\\n")

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-smoke-ingest",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            trace_path=tmp_path / "trace.csv",
            trace_interval_seconds=0.001,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )

    assert record["status"] == "ok"
    assert record["run"]["launch_count"] == 2
    assert record["run"]["load_mode"] == "host-duration-bounded"
    assert record["run"]["workload_start_monotonic_ns"] == 100
    assert record["run"]["workload_end_monotonic_ns"] == 20_000_000_100
    assert record["provenance"]["runner_result_path"] == "runner-result.json"
    persisted = tmp_path / "run-issue137-smoke-ingest.json"
    assert persisted.exists()
    assert json.loads(persisted.read_text(encoding="utf-8"))["status"] == "ok"
    encoded = json.dumps(record)
    assert str(tmp_path) not in encoded


def test_issue137_device_entrypoint_runs_mocked_duration_callback_and_persists_result(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    from tools import issue137_device

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    scope = plan["measurement_scope"]
    environment["board"] = {
        "serial": scope["board_serial"],
        "board_id": scope["board_serial"],
        "bus_id": scope["pci_bus_id"],
    }
    environment_path = tmp_path / "env.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    result_path = tmp_path / "runner-result.json"
    launch_count = 0
    sync_count = 0

    class Prepared:
        def launch(self):
            nonlocal launch_count
            launch_count += 1

        def close(self):
            return None

    class FakeTtnn:
        def open_device(self, *, device_id, num_command_queues):
            assert (device_id, num_command_queues) == (0, 2)
            return object()

        def synchronize_device(self, _device):
            nonlocal sync_count
            sync_count += 1

        def close_device(self, _device):
            return None

    monkeypatch.setitem(sys.modules, "ttnn", FakeTtnn())
    monkeypatch.setattr(issue137_device, "prepare_newton_schulz_duration_load", lambda *args, **kwargs: Prepared())
    clock = iter((0, 20_000_000_000))
    monkeypatch.setattr(issue137_device.time, "monotonic_ns", lambda: next(clock))
    args = issue137_device._parser().parse_args(
        [
            "--env-json",
            str(environment_path),
            "--mode",
            "smoke",
            "--load-core-count",
            "18",
            "--load-total-blocks",
            "171",
            "--load-matrix-block",
            "8",
            "--target-seconds",
            "20",
            "--outer-timeout-seconds",
            "70",
            "--plan-hash",
            runner._plan_hash(plan),
        ]
    )
    monkeypatch.setenv("HEKATUS_TT_RESULT_PATH", str(result_path))
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "wrapper-fresh-run-id")

    assert issue137_device.run(args) == 0
    assert launch_count == 1
    assert sync_count == 1
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["launch_count"] == 1
    assert result["load_mode"] == "host-duration-bounded"
    assert result["core_plan"] == config.core_plan.as_record()


def test_issue137_record_does_not_claim_gddr_provenance_without_observed_versions(
    tmp_path: Path,
) -> None:
    import time
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    environment.pop("host_sw_vers")
    result_path = tmp_path / "runner-result.json"

    def capture(*, stage: str) -> dict[str, object]:
        snapshot = _snapshot()
        snapshot.pop("host_sw_vers")
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": {"pre_run": 0, "in_run": 1, "post_run": 20_000_000_100}[stage],
            "snapshot_elapsed_ns": 1,
            "snapshot": snapshot,
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        result = _device_result(config, environment)
        result_path.write_text(json.dumps(result), encoding="utf-8")
        time.sleep(0.01)
        return SimpleNamespace(returncode=0, stdout="results -> /out/runner-result.json\n")

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-unknown-gddr-source",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            trace_path=tmp_path / "trace.csv",
            trace_interval_seconds=0.001,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )
    assert record["provenance"]["unit_provenance"] is None
    assert record["trace"]["unit_provenance"] is None


def test_issue137_exit_zero_with_missing_or_stale_result_fails_closed(tmp_path: Path) -> None:
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()

    def capture(*, stage: str) -> dict[str, object]:
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": {"pre_run": 0, "post_run": 20_000_000_100}[stage],
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="results     -> /out/runner-result.json\\n")

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-smoke-missing-result",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )

    assert record["status"] == "failed"
    assert record["verdict_evidence"] is False
    assert record["failure"]["kind"] == "runner_result_unavailable"
    assert (tmp_path / "run-issue137-smoke-missing-result.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("dirty", "clean"), ("unpushed", "pushed"), ("missing", "clean")),
)
def test_issue137_environment_preflight_rejects_dirty_unpushed_or_missing_before_command(
    tmp_path: Path, mutation: str, message: str
) -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    if mutation == "dirty":
        environment["harness_dirty"] = True
    elif mutation == "unpushed":
        environment["harness_pushed"] = False
    else:
        environment = {}
    called = False

    def command_runner(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    options = runner.Issue137RunOptions(
        command_runner=command_runner,
        environment=environment,
    )
    with pytest.raises(runner.Issue137Error, match=message):
        runner.run_one(
            plan=plan,
            configuration=config,
            mode="smoke",
            output_dir=tmp_path,
            run_id=f"issue137-env-{mutation}",
            options=options,
        )
    assert called is False


def test_issue137_workload_timeout_cleans_process_group_and_preserves_trap_grace(monkeypatch) -> None:
    class FakeProcess:
        pid = 1234

        def __init__(self) -> None:
            self.communicate_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(("wrapper",), timeout, output="partial", stderr="failure")
            return "cleaned stdout", "cleaned stderr"

        def send_signal(self, _signum):
            return None

    process = FakeProcess()
    popen_calls: list[dict[str, object]] = []
    cleanup_calls: list[object] = []

    def fake_popen(*_args: object, **kwargs: object):
        popen_calls.append(kwargs)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        runner.issue121_q2,
        "_cleanup_workload_process_group",
        lambda workload, **kwargs: cleanup_calls.append(workload) or ("cleaned stdout", "cleaned stderr"),
    )
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        runner._run_issue137_workload(("wrapper",), timeout=70.0, env={})
    assert raised.value.stdout == "cleaned stdout"
    assert popen_calls[0]["start_new_session"] is True
    assert cleanup_calls == [process]


def test_issue137_workload_interrupt_cleans_process_group(monkeypatch) -> None:
    class FakeProcess:
        pid = 4321

        def communicate(self, timeout=None):
            raise KeyboardInterrupt

        def send_signal(self, _signum):
            return None

    process = FakeProcess()
    cleaned: list[object] = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner.issue121_q2,
        "_cleanup_workload_process_group",
        lambda workload, **kwargs: cleaned.append(workload) or ("", ""),
    )
    with pytest.raises(KeyboardInterrupt):
        runner._run_issue137_workload(("wrapper",), timeout=70.0, env={})
    assert cleaned == [process]


def test_issue137_sampler_join_timeout_covers_snapshot_timeout() -> None:
    assert runner._sampler_join_timeout_seconds(0.001) >= 6.0


def test_issue137_sampler_retains_capture_failure_after_revocation() -> None:
    stop = threading.Event()
    state = runner._CapturePublicationState()
    state.revoke()

    def capture(*, stage: str) -> dict[str, object]:
        stop.set()
        return {"stage": stage, "status": "capture_failed", "returncode": None}

    captures: list[dict[str, object]] = []
    runner.sample_issue137_trace(
        stop,
        captures,
        capture_fn=capture,
        interval_seconds=0.001,
        publication_state=state,
    )
    late = state.drain_late_captures()
    assert captures == []
    assert late[0]["status"] == "capture_failed"


def test_issue137_nonzero_or_unknown_returncode_is_not_evidence(tmp_path: Path) -> None:
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    result_path = tmp_path / "runner-result.json"
    command_kwargs: dict[str, object] = {}

    def capture(*, stage: str) -> dict[str, object]:
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": {"pre_run": 0, "in_run": 1, "post_run": 20_000_000_100}[stage],
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        command_kwargs.update(_kwargs)
        result_path.write_text(
            json.dumps(_device_result(config, environment)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=None, stdout="results -> /out/runner-result.json\n")

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-unknown-returncode",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            trace_path=tmp_path / "trace.csv",
            trace_interval_seconds=0.001,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )
    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "runner_execution_failure"
    assert command_kwargs["timeout"] == 70.0


def test_issue137_output_collision_is_rejected_before_command(tmp_path: Path) -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    (tmp_path / "run-issue137-collision.json").write_text("old", encoding="utf-8")
    called = False

    def command_runner(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    options = runner.Issue137RunOptions(
        command_runner=command_runner,
        environment=_issue137_environment(),
    )
    with pytest.raises(runner.Issue137Error, match="collision"):
        runner.run_one(
            plan=plan,
            configuration=config,
            mode="smoke",
            output_dir=tmp_path,
            run_id="issue137-collision",
            options=options,
        )
    assert called is False


def test_issue137_artifact_write_rejects_a_symlink_escape_before_io(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    destination.symlink_to(Path.cwd().parent)

    with pytest.raises(runner.Issue137Error, match="trusted|symlink"):
        runner._write_artifact(destination, {"status": "ok"})


def test_issue137_device_environment_read_rejects_a_symlink_escape_before_io(
    tmp_path: Path,
) -> None:
    from tools import issue137_device

    source = tmp_path / "environment.json"
    source.symlink_to(Path.cwd().parent)

    with pytest.raises(issue137_device.Issue137DeviceError, match="trusted|symlink"):
        issue137_device._load_environment(source)


def test_issue137_device_result_write_rejects_a_symlink_escape_before_io(
    tmp_path: Path,
) -> None:
    from tools import issue137_device

    destination = tmp_path / "runner-result.json"
    destination.symlink_to(Path.cwd().parent)

    with pytest.raises(issue137_device.Issue137DeviceError, match="trusted|symlink"):
        issue137_device._write_result(destination, {"status": "ok"})


def test_issue137_device_result_write_race_does_not_overwrite_fixture_files(
    tmp_path: Path, monkeypatch
) -> None:
    from tools import issue137_device

    existing = tmp_path / "existing-result.json"
    existing.write_text("existing-result", encoding="utf-8")
    with pytest.raises(issue137_device.Issue137DeviceError):
        issue137_device._write_result(existing, {"status": "replacement"})
    assert existing.read_text(encoding="utf-8") == "existing-result"

    destination = tmp_path / "runner-result.json"
    outside = tmp_path / "outside-result.json"
    outside.write_text("outside-result", encoding="utf-8")
    original_open_descriptor = issue137_device._open_result_descriptor

    def substitute_final_component(path: Path) -> int:
        path.symlink_to(outside)
        return original_open_descriptor(path)

    monkeypatch.setattr(
        issue137_device, "_open_result_descriptor", substitute_final_component
    )
    with pytest.raises(issue137_device.Issue137DeviceError):
        issue137_device._write_result(destination, {"status": "replacement"})
    assert outside.read_text(encoding="utf-8") == "outside-result"


def test_issue137_trace_write_rejects_a_symlink_escape_before_io(tmp_path: Path) -> None:
    destination = tmp_path / "power.csv"
    destination.symlink_to(Path.cwd().parent)

    rows = [_row()]
    with pytest.raises(trace.Issue137PowerTraceError, match="trusted|symlink"):
        trace.write_issue137_power_trace(destination, rows)


def test_issue137_plain_valueerror_pre_capture_is_persisted_as_failure(
    tmp_path: Path,
) -> None:
    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    called = False

    def capture(*, stage: str) -> dict[str, object]:
        raise ValueError(f"plain capture failure: {stage}")

    def command_runner(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-plain-valueerror-pre",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            capture_fn=capture,
        ),
    )

    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "pre_run_snapshot_failure"
    assert (tmp_path / "run-issue137-plain-valueerror-pre.json").exists()
    assert called is False


def test_issue137_plain_valueerror_post_capture_is_retained_as_cleanup_failure(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    result_path = tmp_path / "runner-result.json"

    def capture(*, stage: str) -> dict[str, object]:
        if stage == "post_run":
            raise ValueError("plain post-capture failure")
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": 0,
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        result_path.write_text(
            json.dumps(_device_result(config, environment)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="results -> /out/runner-result.json\n")

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-plain-valueerror-post",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )

    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "runner_execution_failure"
    assert record["failure"]["message"] == "post-run snapshot failure: ValueError"
    assert record["post_run_snapshot"] == {
        "status": "unavailable",
        "reason": "ValueError",
    }


def test_issue137_trace_plain_valueerror_normalization_is_not_evaluable(
    monkeypatch,
) -> None:
    def raise_valueerror(_payload: object) -> list[dict[str, object]]:
        raise ValueError("plain normalization failure")

    monkeypatch.setattr(trace, "_validated_issue137_rows", raise_valueerror)
    result = trace.read_issue137_power_trace_mapping({})

    assert result["valid"] is False
    assert result["errors"] == ["plain normalization failure"]


@pytest.mark.parametrize("failure_stage", ["plan", "environment"])
def test_issue137_device_cli_plain_valueerror_returns_structured_failure(
    tmp_path: Path, monkeypatch, failure_stage: str
) -> None:
    from tools import issue137_device

    plan = runner.load_plan(PLAN)

    def raise_valueerror(*_args: object, **_kwargs: object) -> object:
        raise ValueError(f"plain {failure_stage} failure")

    if failure_stage == "plan":
        monkeypatch.setattr(issue137_device.issue137, "load_plan", raise_valueerror)
    else:
        monkeypatch.setattr(issue137_device.issue137, "load_plan", lambda *_args: plan)
        monkeypatch.setattr(issue137_device, "_load_environment", raise_valueerror)

    result = issue137_device.main(
        [
            "--env-json",
            str(tmp_path / "environment.json"),
            "--mode",
            "smoke",
            "--load-core-count",
            "18",
            "--load-total-blocks",
            "171",
            "--load-matrix-block",
            "8",
            "--target-seconds",
            "20",
            "--outer-timeout-seconds",
            "70",
            "--plan-hash",
            "fixture-plan-hash",
        ]
    )

    assert result == 2


def test_issue137_plain_valueerror_trace_finalization_is_recorded(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    result_path = tmp_path / "runner-result.json"

    def capture(*, stage: str) -> dict[str, object]:
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": 0,
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        result_path.write_text(
            json.dumps(_device_result(config, environment)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="results -> /out/runner-result.json\n")

    def raise_valueerror(*_args: object, **_kwargs: object) -> None:
        raise ValueError("plain trace finalization failure")

    monkeypatch.setattr(runner, "write_trace_from_captures", raise_valueerror)
    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-plain-valueerror-trace",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
            trace_path=tmp_path / "trace.csv",
            trace_interval_seconds=60.0,
        ),
    )

    assert record["status"] == "failed"
    assert record["failure"]["kind"] == "trace_capture_failure"
    assert record["trace_error"] == "ValueError"


def test_issue137_failed_runner_result_retains_sanitized_raw_partial_duration(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    plan = runner.load_plan(PLAN)
    config = runner.configuration(plan, "C018")
    environment = _issue137_environment()
    result_path = tmp_path / "runner-result.json"
    raw_result = _device_result(config, environment)
    raw_result.update(
        {
            "status": "failed",
            "launch_count": 7,
            "first_launch_start_monotonic_ns": 100,
            "workload_end_monotonic_ns": 20_000_000_100,
            "last_launch_end_monotonic_ns": 23_500_000_100,
            "overrun_seconds": 3.5,
            "failure": {"kind": "duration_failure", "message": "partial fixture"},
            "environment": {"board": {"serial": "untrusted-raw-result"}},
        }
    )

    def capture(*, stage: str) -> dict[str, object]:
        return {
            "schema": runner.ISSUE137_SNAPSHOT_SCHEMA,
            "stage": stage,
            "status": "ok",
            "returncode": 0,
            "captured_at_utc": "2026-10-20T00:00:00+00:00",
            "timestamp_monotonic_ns": 0,
            "snapshot_elapsed_ns": 1,
            "snapshot": _snapshot(),
        }

    def command_runner(*_args: object, **_kwargs: object) -> SimpleNamespace:
        result_path.write_text(json.dumps(raw_result), encoding="utf-8")
        return SimpleNamespace(returncode=2, stdout="results -> /out/runner-result.json\n")

    record = runner.run_one(
        plan=plan,
        configuration=config,
        mode="smoke",
        output_dir=tmp_path,
        run_id="issue137-failed-result-raw",
        options=runner.Issue137RunOptions(
            command_runner=command_runner,
            environment=environment,
            capture_fn=capture,
            board_id="fixture-board",
            bus_id="0000:09:00.0",
        ),
    )

    persisted = json.loads(
        (tmp_path / "run-issue137-failed-result-raw.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["verdict_evidence"] is False
    assert persisted["runner_result_raw"]["status"] == "failed"
    assert persisted["runner_result_raw"]["launch_count"] == 7
    assert persisted["runner_result_raw"]["overrun_seconds"] == 3.5
    assert persisted["runner_result_raw"]["failure"]["message"] == "partial fixture"
    assert persisted["runner_result_raw"]["environment"]["board"]["serial"] == (
        "untrusted-raw-result"
    )
    assert persisted["provenance"]["runner_result_path"] == "runner-result.json"
    assert "runner_result" not in persisted
    assert persisted["run"]["launch_count"] is None
    assert persisted["environment"]["board"]["serial"] == "fixture-board"
