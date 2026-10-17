import json
import sys
from types import SimpleNamespace

import pytest

from tools import newton_schulz_issue101_combined as issue101


def _raw_environment() -> dict:
    return {
        "image": "ghcr.io/example/image@sha256:" + "a" * 64,
        "image_pinned": True,
        "kernel": "Linux 6.8.0-test",
        "kmd_version": "2.11.0",
        "tt_env_active_release": "0.75.0",
        "harness_commit": "a" * 40,
        "harness_dirty": False,
        "board_info": {"board_type": "p150a", "board_id": "board-only"},
        "firmwares": {"fw_bundle_version": "19.6.0.0"},
        "hostname": "must-not-be-recorded",
        "username": "must-not-be-recorded",
        "cwd": "/home/private/source",
    }


def _write_telemetry(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "env-1.json").write_text(json.dumps(_raw_environment()))
    (output_dir / "power-1.csv").write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,1350,60.0\n"
    )


def _correctness_rows():
    return [
        {
            "case": f"batch{batch}-L{size}",
            "batch": batch,
            "size": size,
            "status": "pass",
        }
        for batch, size in issue101.ISSUE101_CORRECTNESS_CASES
    ]


def _passing_run(repeats=3):
    performance = []
    for shape in issue101.ISSUE100_SHAPES:
        for config in issue101.ISSUE100_COMPARISON_CONFIGS:
            samples = [0.001 + index * 0.000001 for index in range(repeats)]
            performance.append(
                {
                    "status": "ok",
                    "shape_name": shape,
                    "comparison_config": dict(config),
                    "seconds_per_launch_samples": samples,
                    "seconds_per_launch_p50": samples[-1],
                    "seconds_per_launch_p99": samples[-1],
                    "seconds_per_launch_p99_9": samples[-1],
                    "flops_per_iteration": 1.0,
                    "tflops_p50_derived": 0.001,
                    "tflops_fastest_launch_derived": 0.001,
                }
            )
    return {
        "status": "pass",
        "correctness_cases": _correctness_rows(),
        "performance_rows": performance,
    }


def _fake_ttnn(open_device, close_device):
    return SimpleNamespace(open_device=open_device, close_device=close_device)


def _run_main(tmp_path, monkeypatch, *, open_device, close_device, result_path=None):
    output_dir = tmp_path / "device-output"
    _write_telemetry(output_dir)
    monkeypatch.setenv("HEKATUS_TT_OUTPUT_DIR", str(output_dir))
    if result_path is not None:
        monkeypatch.setenv("HEKATUS_TT_RESULT_PATH", str(result_path))
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        _fake_ttnn(open_device, close_device),
    )
    return output_dir, issue101.main([])


def _raw(output_dir):
    raw_path = output_dir / issue101.ISSUE101_RAW_OUTPUT_NAME
    assert raw_path.exists()
    raw = json.loads(raw_path.read_text())
    assert raw["raw_schema"] == issue101.ISSUE101_RAW_SCHEMA
    assert raw["artifact_file"] == raw_path.name
    assert isinstance(raw["run_id"], str)
    assert raw["run"]["run_id"] == raw["run_id"]
    assert raw["telemetry"]["run_id"] == raw["run_id"]
    assert raw["telemetry"]["power_trace"] is not None
    assert raw["telemetry"]["power_trace"]["run_id"] == raw["run_id"]
    serialized = json.dumps(raw)
    assert "hostname" not in serialized
    assert "username" not in serialized
    assert "/home/private/source" not in serialized
    return raw


def test_issue101_result_env_path_is_the_primary_output(monkeypatch, tmp_path):
    result_path = tmp_path / "runner-result.json"
    monkeypatch.setattr(
        issue101, "run_issue101_combined", lambda *args, **kwargs: _passing_run(1000)
    )

    output_dir, result = _run_main(
        tmp_path,
        monkeypatch,
        open_device=lambda **kwargs: object(),
        close_device=lambda device: None,
        result_path=result_path,
    )

    assert result == 0
    assert json.loads(result_path.read_text())["status"] == "pass"
    assert not (output_dir / issue101.ISSUE101_OUTPUT_NAME).exists()


def test_issue101_open_exception_publishes_raw_failure_without_cleanup(monkeypatch, tmp_path):
    close_calls = []

    def fail_open(**kwargs):
        raise RuntimeError("open injected")

    output_dir, result = _run_main(
        tmp_path,
        monkeypatch,
        open_device=fail_open,
        close_device=lambda device: close_calls.append(device),
    )

    assert result == 1
    raw = _raw(output_dir)
    assert raw["failure"]["stage"] == "open"
    assert "open injected" in raw["failure"]["error"]
    assert raw["correctness_results"] == []
    assert raw["performance_results"] == []
    assert raw["run"]["cleanup"] == {
        "device_opened": False,
        "close_attempted": False,
        "close_succeeded": False,
    }
    assert close_calls == []
    assert "hostname" not in json.dumps(raw)
    assert "/home/private/source" not in json.dumps(raw)


def test_issue101_correctness_exception_keeps_completed_rows(monkeypatch, tmp_path):
    def failing_correctness(ttnn, device, *, rows):
        rows.extend(_correctness_rows()[:2])
        raise RuntimeError("correctness injected")

    monkeypatch.setattr(issue101, "run_issue101_correctness", failing_correctness)
    output_dir, result = _run_main(
        tmp_path,
        monkeypatch,
        open_device=lambda **kwargs: object(),
        close_device=lambda device: None,
    )

    assert result == 1
    raw = _raw(output_dir)
    assert raw["failure"]["stage"] == "correctness"
    assert "correctness injected" in raw["failure"]["error"]
    assert len(raw["correctness_results"]) == 2
    assert raw["performance_results"] == []
    assert raw["telemetry"]["normalized_environment"]["board"]["serial"] == "board-only"


def test_issue101_escaped_performance_exception_keeps_correctness_and_perf_rows(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        issue101,
        "run_issue101_correctness",
        lambda ttnn, device, *, rows: rows.extend(_correctness_rows()) or rows,
    )

    def failing_performance(ttnn, device, *, repeats, stop_on_failure, results_sink):
        results_sink.append({"status": "ok", "row": "completed"})
        raise RuntimeError("performance injected")

    monkeypatch.setattr(issue101, "run_issue100_comparison", failing_performance)
    output_dir, result = _run_main(
        tmp_path,
        monkeypatch,
        open_device=lambda **kwargs: object(),
        close_device=lambda device: None,
    )

    assert result == 1
    raw = _raw(output_dir)
    assert raw["failure"]["stage"] == "performance"
    assert "performance injected" in raw["failure"]["error"]
    assert len(raw["correctness_results"]) == len(issue101.ISSUE101_CORRECTNESS_CASES)
    assert raw["performance_results"] == [{"status": "ok", "row": "completed"}]


def test_issue101_telemetry_exception_publishes_failure_without_fabrication(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        issue101,
        "run_issue101_combined",
        lambda *args, **kwargs: _passing_run(1000),
    )
    monkeypatch.setattr(
        issue101,
        "_environment_artifact",
        lambda output_dir: (_ for _ in ()).throw(RuntimeError("telemetry injected")),
    )

    output_dir, result = _run_main(
        tmp_path,
        monkeypatch,
        open_device=lambda **kwargs: object(),
        close_device=lambda device: None,
    )

    assert result == 1
    raw = _raw(output_dir)
    assert raw["failure"]["stage"] == "telemetry"
    assert "telemetry injected" in raw["failure"]["error"]
    assert raw["telemetry"]["environment"]["board_info"]["board_id"] == "board-only"
    assert raw["telemetry"]["normalized_environment"] is None
    assert raw["telemetry"]["power_trace"] is not None
    assert len(raw["correctness_results"]) == 9
    assert len(raw["performance_results"]) == 4


def test_issue101_close_exception_persists_rows_before_reraise(monkeypatch, tmp_path):
    monkeypatch.setattr(issue101, "run_issue101_combined", lambda *args, **kwargs: _passing_run(1000))

    def fail_close(device):
        raise RuntimeError("close injected")

    output_dir = tmp_path / "device-output"
    with pytest.raises(RuntimeError, match="close injected"):
        _run_main(
            tmp_path,
            monkeypatch,
            open_device=lambda **kwargs: object(),
            close_device=fail_close,
        )

    raw = _raw(output_dir)
    assert raw["failure"]["stage"] == "close"
    assert "close injected" in raw["failure"]["error"]
    assert len(raw["correctness_results"]) == 9
    assert len(raw["performance_results"]) == 4
    assert raw["run"]["cleanup"]["close_attempted"] is True
    assert raw["run"]["cleanup"]["close_succeeded"] is False


def test_issue101_record_construction_exception_preserves_raw_failure_metadata(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "device-output"
    _write_telemetry(output_dir)
    monkeypatch.setattr(
        issue101,
        "build_combined_record",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("record construction injected")
        ),
    )

    run = _passing_run()
    with pytest.raises(RuntimeError, match="record construction injected"):
        issue101.persist_raw_and_build(run, output_dir=output_dir, repeats=3)

    raw = _raw(output_dir)
    assert raw["run_id"] == raw["run"]["run_id"]
    assert raw["telemetry"]["run_id"] == raw["run_id"]
    assert raw["telemetry"]["normalized_environment"]["run_id"] == raw["run_id"]
    assert raw["telemetry"]["power_trace"]["run_id"] == raw["run_id"]
    assert raw["artifact_status"] == "failed"
    assert raw["failure"]["stage"] == "record_construction"
    assert raw["run"]["failure_stage"] == "record_construction"
    assert len(raw["correctness_results"]) == 9
    assert len(raw["performance_results"]) == 4
    assert raw["run"]["correctness_cases"] == raw["correctness_results"]
    assert raw["run"]["performance_rows"] == raw["performance_results"]
    assert raw["recovery_run"]["status"] == "pass"
    raw_path = output_dir / issue101.ISSUE101_RAW_OUTPUT_NAME
    recovered = issue101.recover_combined_record(raw_path)
    assert recovered["status"] == "pass"


def _write_rebuild_raw(
    output_dir,
    *,
    run_id="captured-run",
    board_id="captured-board",
    captured_at="2020-02-03T04:05:06+00:00",
    python="3.9.7",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _raw_environment()
    environment.update(
        {
            "captured_at": captured_at,
            "python": python,
            "run_id": run_id,
        }
    )
    environment["board_info"] = {"board_type": "p150a", "board_id": board_id}
    (output_dir / f"env-{run_id}.json").write_text(json.dumps(environment))
    (output_dir / f"power-{run_id}.csv").write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        f"{captured_at},75,1350,60.0\n"
    )
    telemetry = issue101._capture_telemetry(output_dir, run_id=run_id)
    raw_path = issue101.write_raw_artifact(
        _passing_run(3),
        output_dir=output_dir,
        repeats=3,
        telemetry=telemetry,
        run_id=run_id,
    )
    return raw_path, telemetry


def test_issue102_raw_writer_rejects_dangling_destination_symlink(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, telemetry = _write_rebuild_raw(output_dir)
    raw_path.unlink()
    outside_path = tmp_path / "outside" / raw_path.name
    raw_path.symlink_to(outside_path)

    with pytest.raises(FileExistsError, match="symbolic link"):
        issue101.write_raw_artifact(
            _passing_run(3),
            output_dir=output_dir,
            repeats=3,
            telemetry=telemetry,
            run_id="captured-run",
        )

    assert raw_path.is_symlink()
    assert not outside_path.exists()


def test_issue102_live_capture_drops_only_terminal_incomplete_power_row(tmp_path):
    output_dir = tmp_path / "device-output"
    run_id = "live-run"
    output_dir.mkdir()
    environment = _raw_environment()
    environment.update(
        {
            "captured_at": "2020-02-03T04:05:06+00:00",
            "run_id": run_id,
            "board_info": {"board_type": "p150a", "board_id": "live-board"},
        }
    )
    (output_dir / f"env-{run_id}.json").write_text(json.dumps(environment))
    (output_dir / f"power-{run_id}.csv").write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2020-02-03T04:05:06+00:00,75,1350,60.0\n"
        "2020-02-03T04:05:08+00:00,75,1350"
    )

    telemetry = issue101._capture_telemetry(output_dir, run_id=run_id)

    assert telemetry["status"] == "complete"
    assert len(telemetry["power_trace"]["samples"]) == 1
    raw_path = issue101.write_raw_artifact(
        _passing_run(3),
        output_dir=output_dir,
        repeats=3,
        telemetry=telemetry,
        run_id=run_id,
    )
    raw = json.loads(raw_path.read_text())
    assert len(raw["telemetry"]["power_trace"]["samples"]) == 1
    assert issue101.recover_combined_record(raw_path)["run_id"] == run_id


@pytest.mark.parametrize(
    ("terminal_row", "message"),
    [
        ("2020-02-03T04:05:08+00:00,75,1350,60.0,extra", "unsupported shape"),
        ("2020-02-03T04:05:08+00:00,not-a-number,1350,60.0", "not numeric"),
    ],
)
def test_issue102_live_capture_keeps_rejectable_terminal_power_rows(
    tmp_path, terminal_row, message
):
    run_id = "live-run"
    path = tmp_path / f"power-{run_id}.csv"
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        f"{terminal_row}\n"
    )

    trace = issue101._power_trace_artifact(path, run_id=run_id)

    with pytest.raises(ValueError, match=message):
        issue101._validate_captured_power_trace(trace, run_id=run_id)


def test_issue102_recovery_rejects_missing_power_sample_field(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    del payload["telemetry"]["power_trace"]["samples"][0]["asic_temp_c"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported shape"):
        issue101.recover_combined_record(raw_path)


def test_issue102_rebuilds_clean_record_from_raw_artifact_only(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, telemetry = _write_rebuild_raw(output_dir)
    expected_environment = json.loads(
        json.dumps(telemetry["normalized_environment"])
    )

    record = issue101.recover_combined_record(raw_path)

    assert record["run_id"] == "captured-run"
    assert record["captured_at"] == "2020-02-03T04:05:06+00:00"
    assert record["harness_commit"] == "a" * 40
    assert record["environment"] == expected_environment
    assert record["environment"]["python"] == "3.9.7"
    assert record["environment"]["board"]["serial"] == "captured-board"
    assert record["measurement"]["power_clock_provenance"]["samples"] == 1
    assert record["raw_artifact"]["file"] == raw_path.name


def test_issue102_rebuild_selects_run_id_in_reused_directory(tmp_path):
    output_dir = tmp_path / "device-output"
    _write_rebuild_raw(output_dir, run_id="run-a", board_id="board-a")
    raw_b, _ = _write_rebuild_raw(output_dir, run_id="run-b", board_id="board-b")

    selected = issue101.recover_combined_record(output_dir, run_id="run-b")

    assert selected["run_id"] == "run-b"
    assert selected["environment"]["board"]["serial"] == "board-b"
    assert selected["raw_artifact"]["file"] == raw_b.name
    with pytest.raises(RuntimeError, match="explicit run_id"):
        issue101.recover_combined_record(output_dir)
    with pytest.raises(RuntimeError, match="run_id"):
        issue101.recover_combined_record(output_dir, run_id="unknown")

    duplicate = output_dir / "issue101-combined-raw.json"
    duplicate.write_bytes(raw_b.read_bytes())
    with pytest.raises(RuntimeError, match="found 2"):
        issue101.recover_combined_record(output_dir, run_id="run-b")

    missing = output_dir / "issue101-combined-raw-missing.json"
    missing_payload = json.loads(raw_b.read_text())
    del missing_payload["run_id"]
    missing.write_text(json.dumps(missing_payload))
    with pytest.raises(ValueError, match="run_id"):
        issue101.recover_combined_record(output_dir, run_id="run-b")


def test_issue102_rebuild_does_not_read_mutable_capture_files_or_host_state(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "device-output"
    raw_path, telemetry = _write_rebuild_raw(output_dir)
    original_raw = raw_path.read_bytes()
    (output_dir / "env-captured-run.json").unlink()
    (output_dir / "power-captured-run.csv").write_text("changed\n")

    def fail_normalization(_raw):
        raise AssertionError("rebuild must not normalize captured environment")

    def fail_python_version():
        raise AssertionError("rebuild must not inspect current Python")

    class NoCurrentTime(issue101.datetime.datetime):
        @classmethod
        def now(cls, *args, **kwargs):
            raise AssertionError("rebuild must not call datetime.now")

    monkeypatch.setattr(issue101, "normalize_environment", fail_normalization)
    monkeypatch.setattr(issue101.platform, "python_version", fail_python_version)
    monkeypatch.setattr(issue101.datetime, "datetime", NoCurrentTime)
    record = issue101.recover_combined_record(raw_path)

    assert record["environment"] == telemetry["normalized_environment"]
    assert record["captured_at"] == "2020-02-03T04:05:06+00:00"
    assert raw_path.read_bytes() == original_raw


def test_issue102_rebuild_rejects_missing_captured_field(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    del payload["telemetry"]["normalized_environment"]["harness_commit"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="harness_commit"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("schema", issue101.SUPPORTED_RAW_SCHEMAS)
def test_issue102_rebuild_supports_each_declared_raw_schema(tmp_path, schema):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    payload["raw_schema"] = schema
    raw_path.write_text(json.dumps(payload))

    assert issue101.recover_combined_record(raw_path)["run_id"] == "captured-run"


def test_issue102_rebuild_rejects_unknown_schema_by_name(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    payload["raw_schema"] = "adr-0005-issue101-combined-raw-v999"
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="v999"):
        issue101.recover_combined_record(raw_path)


def test_issue102_rebuild_rejects_conflicting_or_missing_identity(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    conflicting = output_dir / "issue101-combined-raw-other.json"
    conflicting.write_bytes(raw_path.read_bytes())
    with pytest.raises(ValueError, match="conflicting run_id"):
        issue101.recover_combined_record(conflicting)

    missing = output_dir / "missing-identity.json"
    payload = json.loads(raw_path.read_text())
    del payload["run_id"]
    missing.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="run_id"):
        issue101.recover_combined_record(missing)


def test_issue102_rebuild_cli_is_board_free_and_writes_record(tmp_path, monkeypatch):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    output_path = tmp_path / "rebuilt.json"
    monkeypatch.setitem(sys.modules, "ttnn", None)

    assert (
        issue101.main(
            [
                "--recover-raw",
                str(raw_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text())["run_id"] == "captured-run"


def test_issue102_rebuild_cli_canonicalizes_absolute_cli_paths(tmp_path, monkeypatch):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    output_dir_alias = output_dir / "." / ".." / output_dir.name
    output_path_alias = output_dir / ".." / "rebuilt.json"
    monkeypatch.setitem(sys.modules, "ttnn", None)

    assert (
        issue101.main(
            [
                "--recover-raw",
                str(output_dir_alias / raw_path.name),
                "--output",
                str(output_path_alias),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "rebuilt.json").read_text())["run_id"] == "captured-run"


def test_issue102_rebuild_cli_does_not_overwrite_raw_input(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    original_raw = raw_path.read_bytes()

    with pytest.raises(SystemExit):
        issue101.main(
            [
                "--recover-raw",
                str(raw_path),
                "--output",
                str(raw_path),
            ]
        )

    assert raw_path.read_bytes() == original_raw
