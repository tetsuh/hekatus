"""Board-free coverage for Issue #88 convergence and publication gates."""

import datetime
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import newton_schulz_issue88 as runner

RUN_START = datetime.datetime(2026, 1, 1, 0, 0, 1, tzinfo=datetime.UTC)
RUN_END = datetime.datetime(2026, 1, 1, 0, 0, 1, 500000, tzinfo=datetime.UTC)


def _environment() -> dict:
    return runner._normalize_environment(
        {
            "image": "ghcr.io/example/image@sha256:" + "a" * 64,
            "image_pinned": True,
            "kernel": "Linux 6.8.0-test",
            "kmd_version": "2.11.0",
            "tt_env_active_release": "0.75.0",
            "harness_commit": "b" * 40,
            "harness_dirty": False,
            "board_info": {
                "board_type": "p150a",
                "board_id": "board-serial",
                "bus_id": "0000:09:00.0",
            },
            "firmwares": {"fw_bundle_version": "19.6.0.0"},
        },
        "run-1",
    )


def _power_trace(tmp_path: Path) -> dict:
    path = tmp_path / "power-run-1.csv"
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,1350,60\n"
        "2026-01-01T00:00:02+00:00,76,1350,61\n"
    )
    trace = runner._read_power_trace(
        path,
        run_id="run-1",
        run_start=RUN_START.isoformat(),
        run_end=RUN_END.isoformat(),
    )
    trace["poll_complete"] = True
    return trace


def _passing_run() -> dict:
    rows = [
        {
            "row": config["name"],
            "status": "preflight_rejected"
            if config.get("expected_preflight_rejection")
            else "ok",
        }
        for config in runner.ISSUE88_COMPARISON_ROWS
    ]
    return {
        "status": "pass",
        "comparison_rows": rows,
        "rows_completed": len(rows),
        "rows_requested": len(rows),
        "stopped_on_failure": False,
    }


def _passing_parts(tmp_path: Path) -> tuple[dict, dict, dict]:
    run = _passing_run()
    telemetry = {
        "status": "complete",
        "normalized_environment": _environment(),
        "power_trace": _power_trace(tmp_path),
        "failures": [],
    }
    cleanup = {
        "device_opened": True,
        "close_attempted": True,
        "close_succeeded": True,
    }
    return run, telemetry, cleanup


def test_issue88_catalogue_has_six_rows_and_host_only_rejection():
    runner.validate_comparison_rows()
    assert len(runner.ISSUE88_COMPARISON_ROWS) == 6
    rejected = next(
        row for row in runner.ISSUE88_COMPARISON_ROWS if row["name"] == "fp32-r-L32"
    )
    assert rejected["name"] == "fp32-r-L32"
    assert rejected["expected_preflight_rejection"] is True
    assert rejected["preflight_status"] == "rejected_before_allocation"
    assert rejected["preflight_bytes"] == 1_884_928
    assert rejected["preflight_over_budget_bytes"] == 312_064

    result = runner.run_comparison(
        SimpleNamespace(),
        object(),
        rows=(rejected,),
    )

    assert result["status"] == "pass"
    assert result["rows_completed"] == 1
    assert result["comparison_rows"] == [runner._preflight_rejected_row(rejected)]
    assert result["comparison_rows"][0]["preflight"]["allocation_attempted"] is False
    assert result["comparison_rows"][0]["preflight"]["launches"] == 0


def test_issue88_status_component_table_is_the_complete_publication_gate(tmp_path):
    run, telemetry, cleanup = _passing_parts(tmp_path)
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)

    assert tuple(statuses) == tuple(name for name, _description in runner.ISSUE88_STATUS_COMPONENTS)
    assert all(set(component) == {"ok", "reason"} for component in statuses.values())
    assert runner._status_components_pass(statuses)

    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
    )
    assert record["status"] == "pass"
    assert record["status_components"] == statuses


@pytest.mark.parametrize(
    ("component", "mutate"),
    [
        (
            "rows",
            lambda run, telemetry: run["comparison_rows"][0].update(
                status="failed", error="row injected"
            ),
        ),
        ("board_selection", lambda run, telemetry: telemetry["normalized_environment"]["board"].pop("board_type")),
        ("image_toolchain", lambda run, telemetry: telemetry["normalized_environment"].update(image_pinned=False)),
        ("harness", lambda run, telemetry: telemetry["normalized_environment"].update(harness_dirty=True)),
        (
            "power_trace",
            lambda run, telemetry: telemetry["power_trace"].update(
                coverage_complete=False, covers_run_end=False
            ),
        ),
    ],
)
def test_issue88_each_provenance_failure_forces_failed_record(
    tmp_path, component, mutate
):
    run, telemetry, cleanup = _passing_parts(tmp_path)
    mutate(run, telemetry)

    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)
    assert statuses[component]["ok"] is False
    assert statuses[component]["reason"]
    assert not runner._status_components_pass(statuses)

    record = runner._record_payload(
        run,
        telemetry=telemetry,
        run_id="run-1",
        raw_path=tmp_path / "raw-run-1.json",
        cleanup=cleanup,
        status_components=statuses,
    )
    assert record["status"] == "failed"
    assert component in record["failure"]["component"] or component == "rows"
    assert statuses[component]["reason"] in json.dumps(record)


def test_issue88_close_failure_is_a_status_component_failure(tmp_path):
    run, telemetry, cleanup = _passing_parts(tmp_path)
    cleanup["close_succeeded"] = False
    statuses = runner._status_components(run, telemetry=telemetry, cleanup=cleanup)

    assert statuses["device_close"] == {
        "ok": False,
        "reason": "device close was not completed successfully",
    }
    assert not runner._status_components_pass(statuses)


def _write_main_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    environment_path = tmp_path / "env-run-1.json"
    power_path = tmp_path / "power-run-1.csv"
    result_path = tmp_path / "result.json"
    environment_path.write_text(
        json.dumps(
            {
                "image": "ghcr.io/example/image@sha256:" + "a" * 64,
                "image_pinned": True,
                "kernel": "Linux 6.8.0-test",
                "kmd_version": "2.11.0",
                "tt_env_active_release": "0.75.0",
                "harness_commit": "b" * 40,
                "harness_dirty": False,
                "board_info": {
                    "board_type": "p150a",
                    "board_id": "board-serial",
                    "bus_id": "0000:09:00.0",
                },
                "firmwares": {"fw_bundle_version": "19.6.0.0"},
            }
        )
    )
    power_path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,1350,60\n"
        "2026-01-01T00:00:02+00:00,76,1350,61\n"
    )
    return environment_path, power_path, result_path


def _run_main_with_close(tmp_path, monkeypatch, close_device):
    environment_path, power_path, result_path = _write_main_artifacts(tmp_path)
    monkeypatch.setenv("HEKATUS_TT_RUN_ID", "run-1")
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(
            open_device=lambda **kwargs: object(),
            close_device=close_device,
        ),
    )
    monkeypatch.setattr(runner, "run_comparison", lambda *args, **kwargs: _passing_run())
    return runner.main(
        [
            "--out",
            str(result_path),
            "--env-json",
            str(environment_path),
            "--power-trace",
            str(power_path),
        ],
        now_fn=iter((RUN_START, RUN_END)).__next__,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    ), result_path


def test_issue88_close_success_is_recorded_after_normal_return(tmp_path, monkeypatch):
    result, result_path = _run_main_with_close(tmp_path, monkeypatch, lambda device: None)

    assert result == 0
    record = json.loads(result_path.read_text())
    assert record["run_protocol"]["cleanup"]["close_succeeded"] is True
    assert record["status_components"]["device_close"]["ok"] is True


def test_issue88_close_exception_keeps_false_cleanup_metadata(tmp_path, monkeypatch):
    def fail_close(device):
        raise RuntimeError("close injected")

    result, result_path = _run_main_with_close(tmp_path, monkeypatch, fail_close)

    assert result == 1
    record = json.loads(result_path.read_text())
    assert record["run_protocol"]["cleanup"]["close_succeeded"] is False
    assert record["status_components"]["device_close"]["ok"] is False
    assert "close injected" in json.dumps(record)


def _power_csv(timestamps: list[str]) -> str:
    return "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n" + "".join(
        f"{timestamp},75,1350,60\n" for timestamp in timestamps
    )


def test_issue88_power_poll_waits_for_sample_covering_run_end(tmp_path):
    path = tmp_path / "power-run-1.csv"
    path.write_text(_power_csv(["2026-01-01T00:00:01+00:00"]))
    monotonic = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        monotonic[0] += seconds
        path.write_text(
            _power_csv(
                [
                    "2026-01-01T00:00:01+00:00",
                    "2026-01-01T00:00:02+00:00",
                ]
            )
        )

    trace = runner._wait_for_power_trace(
        tmp_path,
        run_id="run-1",
        explicit=path,
        run_start=RUN_START,
        run_end=RUN_END,
        timeout_s=2.0,
        interval_s=1.0,
        monotonic_fn=lambda: monotonic[0],
        sleep_fn=sleep,
    )

    assert sleeps == [1.0]
    assert trace["poll_complete"] is True
    assert trace["coverage_complete"] is True
    assert trace["sample_count"] == 2
    assert trace["first_timestamp"] <= trace["run_start"]
    assert trace["run_end"] <= trace["last_timestamp"]


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2026-01-01T00:00:01+00:00"],
        ["not-a-timestamp"],
    ],
)
def test_issue88_power_poll_records_incomplete_or_invalid_timestamp_failure(
    tmp_path, timestamps
):
    path = tmp_path / "power-run-1.csv"
    path.write_text(_power_csv(timestamps))
    monotonic = [0.0]

    def sleep(seconds):
        monotonic[0] += seconds

    trace = runner._wait_for_power_trace(
        tmp_path,
        run_id="run-1",
        explicit=path,
        run_start=RUN_START,
        run_end=RUN_END,
        timeout_s=1.0,
        interval_s=0.5,
        monotonic_fn=lambda: monotonic[0],
        sleep_fn=sleep,
    )

    assert trace["poll_complete"] is False
    assert trace["coverage_complete"] is False
    assert trace["poll_error"]
    if timestamps[0] == "not-a-timestamp":
        assert trace["timestamps_parse"] is False
    else:
        assert trace["covers_run_end"] is False


def test_issue88_power_poll_configuration_stays_within_row_cap():
    assert runner.POWER_TRACE_POLL_TIMEOUT_S <= runner.ROW_TIMEOUT_S
