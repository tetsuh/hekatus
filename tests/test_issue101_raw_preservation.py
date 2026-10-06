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
    return json.loads((output_dir / issue101.ISSUE101_RAW_OUTPUT_NAME).read_text())


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


def test_issue101_record_construction_exception_marks_raw_and_keeps_recovery(
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

    with pytest.raises(RuntimeError, match="record construction injected"):
        issue101.persist_raw_and_build(_passing_run(), output_dir=output_dir, repeats=3)

    raw = _raw(output_dir)
    assert raw["artifact_status"] == "failed"
    assert raw["failure"]["stage"] == "record_construction"
    assert raw["run"]["failure_stage"] == "record_construction"
    assert raw["recovery_run"]["status"] == "pass"
    assert len(raw["correctness_results"]) == 9
    assert len(raw["performance_results"]) == 4
    recovered = issue101.recover_combined_record(
        output_dir / issue101.ISSUE101_RAW_OUTPUT_NAME
    )
    assert recovered["status"] == "pass"
    assert recovered["raw_artifact"]["file"] == issue101.ISSUE101_RAW_OUTPUT_NAME
