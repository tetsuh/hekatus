import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench.sampler_contract import power_trace_completion_marker_path
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
            "reference": issue101.ISSUE101_CORRECTNESS_REFERENCE,
            "threshold": issue101.ISSUE101_CORRECTNESS_THRESHOLD,
            "relative_error": 0.001,
            "status": "pass",
        }
        for batch, size in issue101.ISSUE101_CORRECTNESS_CASES
    ]


def _producer_percentile(samples, quantile):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _passing_run(repeats=3):
    performance = []
    for shape in issue101.ISSUE100_SHAPES:
        for config in issue101.ISSUE100_COMPARISON_CONFIGS:
            samples = [0.001 + index * 0.000001 for index in range(repeats)]
            p50 = _producer_percentile(samples, 0.50)
            p99 = _producer_percentile(samples, 0.99)
            p99_9 = _producer_percentile(samples, 0.999)
            performance.append(
                {
                    "status": "ok",
                    "shape_name": shape,
                    "comparison_config": dict(config),
                    "launches_measured": repeats,
                    "seconds_per_launch_samples": samples,
                    "seconds_per_launch_p50": p50,
                    "seconds_per_launch_p99": p99,
                    "seconds_per_launch_p99_9": p99_9,
                    "flops_per_iteration": 1.0,
                    "tflops_p50_derived": 1.0 / p50 / 1e12,
                    "tflops_fastest_launch_derived": 1.0 / min(samples) / 1e12,
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


def test_issue102_keyboard_interrupt_closes_open_device_once(monkeypatch):
    device = object()
    close_calls = []

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(issue101, "run_issue101_combined", interrupt)
    ttnn = SimpleNamespace(close_device=lambda closed: close_calls.append(closed))

    with pytest.raises(KeyboardInterrupt):
        issue101._run_open_device_session(ttnn, device)

    assert close_calls == [device]


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


@pytest.mark.parametrize("failure_stage", ["close", "telemetry"])
def test_issue102_recovery_rejects_non_builder_enclosing_failure(
    monkeypatch, tmp_path, failure_stage
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

    raw_path = output_dir / issue101.ISSUE101_RAW_OUTPUT_NAME
    payload = json.loads(raw_path.read_text())
    assert payload["recovery_run"]["status"] == "pass"
    failure = {"stage": failure_stage, "error": f"{failure_stage} injected"}
    payload["run"].update(
        {"failure_stage": failure_stage, "error": failure["error"], "failure": failure}
    )
    payload["failure"] = failure
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="builder-only"):
        issue101.recover_combined_record(raw_path)


def _write_legacy_builder_failure_raw(output_dir):
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    failure = {
        "stage": "record_construction",
        "error": "RuntimeError: legacy builder failure",
    }
    payload["artifact_status"] = "failed"
    payload["failure"] = failure
    payload["run"]["status"] = "failed"
    payload["run"]["failure_stage"] = "record_construction"
    payload["run"]["error"] = failure["error"]
    payload["run"]["failure"] = failure
    raw_path.write_text(json.dumps(payload))
    return raw_path


def test_issue102_legacy_builder_failure_payload_recovers_board_free():
    raw_path = Path(__file__).parent / "fixtures/issue101/legacy-builder-failure.json"

    record = issue101.recover_combined_record(raw_path)

    assert record["status"] == "pass"
    assert len(record["measurement"]["correctness_cases"]) == 9
    assert len(record["measurement"]["performance_rows"]) == 4
    assert json.loads(raw_path.read_text())["run"]["status"] == "failed"


def test_issue102_recovery_rejects_failed_artifact_with_passing_run(tmp_path):
    fixture = Path(__file__).parent / "fixtures/issue101/legacy-builder-failure.json"
    payload = json.loads(fixture.read_text())
    payload["artifact_status"] = "failed"
    payload["run"]["status"] = "pass"
    raw_path = tmp_path / fixture.name
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="artifact_status"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    "failure_stage",
    ["open", "device_session", "correctness", "performance", "telemetry", "close"],
)
def test_issue102_legacy_recovery_rejects_non_builder_failure(tmp_path, failure_stage):
    raw_path = _write_legacy_builder_failure_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    failure = {"stage": failure_stage, "error": f"{failure_stage} failure"}
    payload["run"].update(
        {"failure_stage": failure_stage, "error": failure["error"], "failure": failure}
    )
    payload["failure"] = failure
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=failure_stage):
        issue101.recover_combined_record(raw_path)


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


@pytest.mark.parametrize(
    ("first_value", "second_value"),
    [("first-run", "second-run"), ("second-run", "first-run")],
    ids=["first-then-second", "second-then-first"],
)
def test_issue102_recovery_rejects_duplicate_root_run_id_in_either_order(
    tmp_path, first_value, second_value
):
    raw_path = tmp_path / "duplicate-root.json"
    raw_path.write_text(
        "{\n"
        f'  "raw_schema": "{issue101.ISSUE101_RAW_SCHEMA}",\n'
        f'  "run_id": "{first_value}",\n'
        f'  "run_id": "{second_value}"\n'
        "}\n"
    )

    with pytest.raises(ValueError, match="duplicate JSON object member 'run_id'"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    ("container", "member"),
    [("run", "run_id"), ("board_serial_identity", "serial")],
    ids=["run-id", "board-identity"],
)
def test_issue102_recovery_rejects_duplicate_nested_members(
    tmp_path, container, member
):
    raw_path = tmp_path / "duplicate-nested.json"
    raw_path.write_text(
        "{\n"
        f'  "{container}": {{\n'
        f'    "{member}": "first",\n'
        f'    "{member}": "second"\n'
        "  }\n"
        "}\n"
    )

    with pytest.raises(ValueError, match=f"duplicate JSON object member {member!r}"):
        issue101.recover_combined_record(raw_path)


def test_issue102_recovery_rejects_escaped_equivalent_member_names(tmp_path):
    raw_path = tmp_path / "duplicate-escaped.json"
    raw_path.write_text(
        "{\n"
        f'  "raw_schema": "{issue101.ISSUE101_RAW_SCHEMA}",\n'
        '  "run_id": "plain",\n'
        '  "run\\u005fid": "escaped"\n'
        "}\n"
    )

    with pytest.raises(ValueError, match="duplicate JSON object member 'run_id'"):
        issue101.recover_combined_record(raw_path)


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


@pytest.mark.parametrize(
    "terminal_row",
    [
        "2020-02-03T04:05:08+00:00,75,1350",
        "2020-02-03T04:05:08+00:00,75,1350,",
        "2020-02-03T04:05:08+00:00,75,1350,6",
        "2020-02-03T04:05:08+00:00,75,1350,60.0",
    ],
    ids=[
        "missing-final-field",
        "trailing-delimiter",
        "partial-numeric-fragment",
        "valid-looking-unterminated-row",
    ],
)
def test_issue102_live_capture_drops_unmarked_terminal_incomplete_power_row(
    tmp_path, terminal_row
):
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
        f"{terminal_row}"
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


def test_issue102_live_capture_keeps_line_terminated_short_power_row(tmp_path):
    run_id = "live-run"
    path = tmp_path / f"power-{run_id}.csv"
    path.write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2020-02-03T04:05:08+00:00,75,1350\n"
    )

    trace = issue101._power_trace_artifact(path, run_id=run_id)

    with pytest.raises(ValueError, match="not numeric"):
        issue101._validate_captured_power_trace(trace, run_id=run_id)


def test_issue102_live_capture_retains_marked_complete_unterminated_power_row(tmp_path):
    run_id = "live-run"
    path = tmp_path / f"power-{run_id}.csv"
    path.write_bytes(
        b"timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        b"2020-02-03T04:05:06+00:00,75,1350,60.0\n"
        b"2020-02-03T04:05:08+00:00,75,1350,60.0"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    power_trace_completion_marker_path(path).write_text(f"sha256:{digest}\n")

    trace = issue101._power_trace_artifact(path, run_id=run_id)

    assert len(trace["samples"]) == 2
    issue101._validate_captured_power_trace(trace, run_id=run_id)


@pytest.mark.parametrize(
    ("raw_board", "source"),
    [
        ({"board_type": "p150a", "board_id": "board-only"}, "board_id_alias"),
        ({"board_type": "p150a", "board_serial": "legacy-only"}, "board_serial_alias"),
        ({"board_type": "p150a", "serial": "explicit"}, "explicit_serial"),
    ],
    ids=["board-id-alias", "board-serial-alias", "explicit-serial"],
)
def test_issue102_identity_accepts_normalized_board_sources(raw_board, source):
    environment = issue101.normalize_environment({"board_info": raw_board})
    environment["run_id"] = "saved-run"

    assert environment["board"]["serial_identity_source"] == source
    assert (
        issue101._validate_environment_identity(environment, run_id="saved-run")
        == "saved-run"
    )


def test_issue102_identity_accepts_alias_source_saved_only_in_identity_metadata():
    environment = issue101.normalize_environment(
        {"board_info": {"board_type": "p150a", "board_id": "board-only"}}
    )
    environment["run_id"] = "saved-run"
    del environment["board"]["serial_identity_source"]

    assert issue101._validate_environment_identity(environment, run_id="saved-run") == "saved-run"


def test_issue102_recovery_rejects_board_serial_rule_mismatch(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    environment = payload["telemetry"]["normalized_environment"]
    environment["board"].update({"serial": "serial-b", "board_id": "board-a"})
    environment["board_serial_identity"].update(
        {"serial": "serial-b", "board_id": "board-a"}
    )
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="serial and board_id"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("identity", "source", "explicit_serial"),
        ("identity", "alias_applied", False),
        ("identity", "rule", "not-the-captured-rule"),
        ("board", "serial_identity_source", "explicit_serial"),
    ],
)
def test_issue102_recovery_rejects_mismatched_saved_board_identity_metadata(
    tmp_path, location, field, value
):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    environment = payload["telemetry"]["normalized_environment"]
    environment["board" if location == "board" else "board_serial_identity"][field] = value
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="identity|serial_identity_source"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("missing_field", ["reference", "threshold", "relative_error"])
def test_issue102_recovery_rejects_missing_correctness_fields(tmp_path, missing_field):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    row = payload["correctness_results"][0]
    del row[missing_field]
    payload["run"]["correctness_cases"] = payload["correctness_results"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="reference|threshold|shape"):
        issue101.recover_combined_record(raw_path)


def test_issue102_recovery_rejects_pass_with_error_above_threshold(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    row = payload["correctness_results"][0]
    row["relative_error"] = issue101.ISSUE101_CORRECTNESS_THRESHOLD * 2
    payload["run"]["correctness_cases"] = payload["correctness_results"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="threshold"):
        issue101.recover_combined_record(raw_path)


def test_issue102_recovery_rejects_missing_power_sample_field(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    del payload["telemetry"]["power_trace"]["samples"][0]["asic_temp_c"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported shape"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("field", ["power_w", "aiclk_mhz", "asic_temp_c"])
def test_issue102_recovery_rejects_boolean_power_sample_values(tmp_path, field):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["power_trace"]["samples"][0][field] = True
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=rf"field {field!r} is not numeric"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("aiclk_mhz", [0, -1], ids=["zero", "negative"])
def test_issue102_recovery_rejects_non_positive_aiclk(tmp_path, aiclk_mhz):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["power_trace"]["samples"][0]["aiclk_mhz"] = aiclk_mhz
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="field 'aiclk_mhz' must be positive"):
        issue101.recover_combined_record(raw_path)


def test_issue102_recovery_rejects_oversized_json_integer(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["power_trace"]["samples"][0]["power_w"] = 10**309
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="field 'power_w' is not numeric"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("seconds_per_launch_samples", "not-a-number"),
        ("seconds_per_launch_samples", True),
        ("seconds_per_launch_samples", -0.1),
        ("seconds_per_launch_samples", 0.0),
        ("seconds_per_launch_p50", None),
        ("seconds_per_launch_p50", -0.1),
        ("seconds_per_launch_p50", 0.0),
        ("seconds_per_launch_p99", float("nan")),
        ("seconds_per_launch_p99", -0.1),
        ("seconds_per_launch_p99", 0.0),
        ("seconds_per_launch_p99_9", float("inf")),
        ("seconds_per_launch_p99_9", -0.1),
        ("seconds_per_launch_p99_9", 0.0),
        ("flops_per_iteration", None),
        ("flops_per_iteration", -0.1),
        ("flops_per_iteration", 0.0),
        ("tflops_p50_derived", float("nan")),
        ("tflops_p50_derived", -0.1),
        ("tflops_p50_derived", 0.0),
        ("tflops_fastest_launch_derived", float("inf")),
        ("tflops_fastest_launch_derived", -0.1),
        ("tflops_fastest_launch_derived", 0.0),
    ],
    ids=[
        "sample-string",
        "sample-bool",
        "sample-negative",
        "sample-zero",
        "p50-null",
        "p50-negative",
        "p50-zero",
        "p99-nan",
        "p99-negative",
        "p99-zero",
        "p99_9-infinity",
        "p99_9-negative",
        "p99_9-zero",
        "flops-null",
        "flops-negative",
        "flops-zero",
        "tflops-p50-nan",
        "tflops-p50-negative",
        "tflops-p50-zero",
        "tflops-fastest-infinity",
        "tflops-fastest-negative",
        "tflops-fastest-zero",
    ],
)
def test_issue102_rebuild_rejects_malformed_performance_numbers(
    tmp_path, field, bad_value
):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    row = payload["performance_results"][0]
    if field == "seconds_per_launch_samples":
        row[field][0] = bad_value
    else:
        row[field] = bad_value
    payload["run"]["performance_rows"] = payload["performance_results"]

    with pytest.raises(ValueError, match="finite numeric|non-negative|positive"):
        issue101._record_from_raw_payload(raw_path, payload)


def test_issue102_recovery_rejects_oversized_flops_integer(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    payload["performance_results"][0]["flops_per_iteration"] = 10**400
    payload["run"]["performance_rows"] = payload["performance_results"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="finite numeric"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    "field",
    [
        "seconds_per_launch_p50",
        "seconds_per_launch_p99",
        "seconds_per_launch_p99_9",
        "tflops_p50_derived",
        "tflops_fastest_launch_derived",
    ],
    ids=["p50", "p99", "p99_9", "tflops-p50", "tflops-fastest"],
)
def test_issue102_rebuild_rejects_inconsistent_performance_summaries(tmp_path, field):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    row = payload["performance_results"][0]
    row[field] *= 2
    payload["run"]["performance_rows"] = payload["performance_results"]

    with pytest.raises(ValueError, match=field):
        issue101._record_from_raw_payload(raw_path, payload)


@pytest.mark.parametrize("mutation", ["missing", "mismatched"], ids=["missing", "mismatched"])
def test_issue102_recovery_rejects_missing_or_mismatched_launches_measured(
    tmp_path, mutation
):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    row = payload["performance_results"][0]
    if mutation == "missing":
        del row["launches_measured"]
    else:
        row["launches_measured"] = 2
    payload["run"]["performance_rows"] = payload["performance_results"]
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="launches_measured"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    ("field", "alias_field"),
    [
        ("kernel", "host_kernel"),
        ("kmd_version", "kernel_driver_version"),
    ],
    ids=["kernel-host-kernel", "kmd-kernel-driver-version"],
)
def test_issue102_recovery_rejects_conflicting_environment_aliases(
    tmp_path, field, alias_field
):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    environment = payload["telemetry"]["normalized_environment"]
    environment[alias_field] = f"conflicting-{field}"
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"{field}|{alias_field}"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "registry.example/other@sha256:" + "b" * 64),
        ("image_pinned", False),
        ("kernel", "Linux conflicting"),
        ("kmd_version", "conflicting-kmd"),
        ("tt_env_active_release", "conflicting-release"),
        ("harness_commit", "b" * 40),
        ("harness_dirty", True),
        ("captured_at", "2020-02-03T04:05:07+00:00"),
        ("python", "3.10.0"),
        ("run_id", "other-run"),
    ],
    ids=[
        "image",
        "image-pinned",
        "kernel",
        "kmd-version",
        "toolchain-release",
        "harness-commit",
        "harness-dirty",
        "captured-at",
        "python",
        "run-id",
    ],
)
def test_issue102_recovery_rejects_raw_normalized_scalar_conflicts(
    tmp_path, field, value
):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["environment"][field] = value
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("field", ["board_type", "board_id", "serial"])
def test_issue102_recovery_rejects_raw_normalized_board_conflicts(tmp_path, field):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["environment"]["board_info"][field] = "conflicting-board"
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        issue101.recover_combined_record(raw_path)


def test_issue102_recovery_rejects_raw_normalized_firmware_conflict(tmp_path):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["environment"]["firmwares"]["fw_bundle_version"] = (
        "conflicting-firmware"
    )
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="firmware"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("raw", "image_digest", "sha256:" + "b" * 64),
        ("raw", "host_kernel", "Linux conflicting"),
        ("raw", "kernel_driver_version", "conflicting-driver"),
        ("raw", "toolchain_release", "conflicting-toolchain"),
        ("normalized", "image_digest", "sha256:" + "b" * 64),
        ("normalized", "host_kernel", "Linux conflicting"),
        ("normalized", "kernel_driver_version", "conflicting-driver"),
        ("normalized", "toolchain_release", "conflicting-toolchain"),
    ],
    ids=[
        "raw-image-digest",
        "raw-host-kernel",
        "raw-kernel-driver-version",
        "raw-toolchain-release",
        "normalized-image-digest",
        "normalized-host-kernel",
        "normalized-kernel-driver-version",
        "normalized-toolchain-release",
    ],
)
def test_issue102_recovery_rejects_raw_normalized_scalar_alias_conflicts(
    tmp_path, location, field, value
):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    raw_environment = payload["telemetry"]["environment"]
    normalized_environment = payload["telemetry"]["normalized_environment"]
    if location == "normalized":
        raw_environment[field] = normalized_environment[field]
        normalized_environment[field] = value
    else:
        raw_environment[field] = value
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize(
    "field", ["board_type", "board_id", "serial", "board_serial"]
)
def test_issue102_recovery_rejects_normalized_board_info_conflicts(tmp_path, field):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["normalized_environment"]["board_info"][field] = (
        "conflicting-board"
    )
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=field):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("location", ["board_info", "board"])
def test_issue102_recovery_rejects_raw_board_serial_alias_conflicts(tmp_path, location):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    raw_environment = payload["telemetry"]["environment"]
    if location == "board_info":
        raw_environment[location]["board_serial"] = "conflicting-board"
    else:
        raw_environment[location] = {"board_serial": "conflicting-board"}
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="board_serial"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("field", ["fw_bundle_version"])
def test_issue102_recovery_rejects_normalized_firmwares_conflicts(tmp_path, field):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    payload["telemetry"]["normalized_environment"]["firmwares"][field] = (
        "conflicting-firmware"
    )
    raw_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="firmware"):
        issue101.recover_combined_record(raw_path)


@pytest.mark.parametrize("field", ["captured_at", "python"])
def test_issue102_recovery_allows_omitted_optional_raw_environment_fields(tmp_path, field):
    raw_path, _ = _write_rebuild_raw(tmp_path / "device-output")
    payload = json.loads(raw_path.read_text())
    del payload["telemetry"]["environment"][field]
    raw_path.write_text(json.dumps(payload))

    assert issue101.recover_combined_record(raw_path)["run_id"] == "captured-run"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "out-of-order"])
def test_issue102_rebuild_rejects_invalid_performance_row_identity(tmp_path, mutation):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    payload = json.loads(raw_path.read_text())
    rows = payload["performance_results"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = dict(rows[0])
    elif mutation == "unexpected":
        rows[0]["shape_name"] = "unexpected-shape"
    else:
        rows[0], rows[1] = rows[1], rows[0]
    payload["run"]["performance_rows"] = rows

    with pytest.raises(ValueError, match="performance row|all four"):
        issue101._record_from_raw_payload(raw_path, payload)


def test_issue102_rebuilds_clean_record_from_raw_artifact_only(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, telemetry = _write_rebuild_raw(output_dir)
    expected_environment = json.loads(
        json.dumps(telemetry["normalized_environment"])
    )

    record = issue101.recover_combined_record(raw_path)
    raw_payload = json.loads(raw_path.read_text())

    assert record["run_id"] == "captured-run"
    assert record["captured_at"] == "2020-02-03T04:05:06+00:00"
    assert record["harness_commit"] == "a" * 40
    assert record["environment"] == expected_environment
    assert record["environment"]["python"] == "3.9.7"
    assert record["environment"]["board"]["serial"] == "captured-board"
    assert record["measurement"]["power_clock_provenance"]["samples"] == 1
    assert record["measurement"]["performance_rows"] == raw_payload["run"]["performance_rows"]
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
    record = json.loads(output_path.read_text())
    assert record["run_id"] == "captured-run"
    assert (
        "The raw artifact is retained for audit; the record can be rebuilt board-free "
        "with --recover-raw."
    ) in record["notes"]


def test_issue102_atomic_json_no_clobber_preserves_first_bytes(tmp_path):
    output_path = tmp_path / "record.json"

    issue101._atomic_json_write(
        output_path, {"winner": "first"}, no_clobber=True
    )
    first_bytes = output_path.read_bytes()

    with pytest.raises(FileExistsError):
        issue101._atomic_json_write(
            output_path, {"winner": "second"}, no_clobber=True
        )

    assert output_path.read_bytes() == first_bytes
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))

    issue101._atomic_json_write(output_path, {"winner": "replacement"})
    assert json.loads(output_path.read_text()) == {"winner": "replacement"}


def test_issue102_rebuild_cli_translates_publication_race_error(
    tmp_path, monkeypatch, capsys
):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    output_path = tmp_path / "rebuilt.json"
    first_bytes = b"published by the competing rebuild\n"

    def publish_race(_temporary_path, destination):
        Path(destination).write_bytes(first_bytes)
        raise FileExistsError("publication lost race")

    monkeypatch.setattr(issue101.os, "link", publish_race)
    with pytest.raises(SystemExit) as excinfo:
        issue101.main(
            [
                "--recover-raw",
                str(raw_path),
                "--output",
                str(output_path),
            ]
        )

    assert excinfo.value.code == 2
    assert output_path.read_bytes() == first_bytes
    assert (
        f"rebuild output already exists: {output_path.resolve()}; refusing to overwrite"
        in capsys.readouterr().err
    )


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


@pytest.mark.parametrize(
    "alias_kind",
    ["missing-parent", "nested-missing", "regular-file-parent"],
    ids=["missing-parent", "nested-missing", "regular-file-parent"],
)
def test_issue102_rebuild_cli_rejects_existing_destination_aliases(
    tmp_path, alias_kind
):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    existing = tmp_path / "existing.json"
    original = b"do not overwrite\n"
    existing.write_bytes(original)
    if alias_kind == "missing-parent":
        output = tmp_path / "missing" / ".." / existing.name
    elif alias_kind == "nested-missing":
        output = tmp_path / "missing" / "nested" / ".." / ".." / existing.name
    else:
        regular_file = tmp_path / "regular-file"
        regular_file.write_text("not a directory")
        output = regular_file / ".." / existing.name

    with pytest.raises(SystemExit) as excinfo:
        issue101.main(
            [
                "--recover-raw",
                str(raw_path),
                "--output",
                str(output),
            ]
        )

    assert excinfo.value.code == 2
    assert existing.read_bytes() == original


def test_issue102_rebuild_cli_rejects_dangling_output_symlink(tmp_path):
    output_dir = tmp_path / "device-output"
    raw_path, _ = _write_rebuild_raw(output_dir)
    output = tmp_path / "output.json"
    target = tmp_path / "missing-target" / "rebuilt.json"
    output.symlink_to(target)

    with pytest.raises(SystemExit) as excinfo:
        issue101.main(
            [
                "--recover-raw",
                str(raw_path),
                "--output",
                str(output),
            ]
        )

    assert excinfo.value.code == 2
    assert output.is_symlink()
    assert not target.exists()


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
