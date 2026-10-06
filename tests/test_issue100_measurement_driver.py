import json
import sys
from types import SimpleNamespace

import pytest

from enodia.tt.bench.newton_schulz_reference import (
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
)
from tools import newton_schulz_issue100_same_run as issue100
from tools import newton_schulz_issue101_combined as issue101
from tools.newton_schulz_issue100_same_run import (
    ISSUE100_COMPARISON_CONFIGS,
    ISSUE100_SHAPES,
)
from tools.newton_schulz_issue101_combined import (
    ISSUE101_CORRECTNESS_CASES,
    ISSUE101_NEW_DEFAULT,
    normalize_environment,
    run_issue101_combined,
)


def test_issue100_same_run_driver_pins_both_default_configurations():
    assert ISSUE100_SHAPES == (
        "newton_schulz_L32_b8192",
        "newton_schulz_L16_b8192",
    )
    assert ISSUE100_COMPARISON_CONFIGS == (
        {
            "name": "new_default",
            "variant": "bf16",
            "matrix_block": 8,
            "double_buffer": True,
            "fuse_s": True,
            "math_fidelity": "HiFi3",
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": True,
        },
        {
            "name": "previous_default",
            "variant": "bf16-fp32state",
            "matrix_block": 4,
            "double_buffer": True,
            "fuse_s": True,
            "math_fidelity": "HiFi3",
            "fp32_dest_acc_en": True,
            "dst_full_sync_en": True,
        },
    )


def test_issue100_result_env_path_is_the_primary_output(monkeypatch, tmp_path):
    result_path = tmp_path / "runner-result.json"
    monkeypatch.setenv("HEKATUS_TT_RESULT_PATH", str(result_path))
    monkeypatch.setitem(
        sys.modules,
        "ttnn",
        SimpleNamespace(
            open_device=lambda device_id: object(),
            close_device=lambda device: None,
        ),
    )
    monkeypatch.setattr(
        issue100,
        "run_issue100_comparison",
        lambda *args, **kwargs: [{"status": "ok"}],
    )

    assert issue100.main() == 0
    assert json.loads(result_path.read_text())["results"] == [{"status": "ok"}]


def test_issue101_combined_driver_pins_all_requested_cases_and_defaults():
    assert ISSUE101_CORRECTNESS_CASES == (
        (4, 16),
        (8192, 16),
        (4, 32),
        (8192, 32),
        (1, 16),
        (3, 16),
        (5, 32),
        (31, 32),
        (63, 32),
    )
    assert ISSUE101_NEW_DEFAULT == {
        "variant": "bf16",
        "state_format": "BF16",
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "fp32_dest_acc_en": True,
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "input_memory": "l1",
        "r_memory": "l1",
        "x0_memory": "l1",
        "output_memory": "dram",
        "initial_value": "I/||R||inf",
        "iterations": 12,
    }


def test_issue101_combined_driver_uses_one_session_and_stops_on_correctness_failure(
    monkeypatch,
):
    calls = []

    def fake_kernel(ttnn, device, matrices, **kwargs):
        calls.append((ttnn, device, matrices.shape, kwargs))
        result = newton_schulz_reference(
            bf16_round_complex(matrices), x0=initial_value(matrices)
        )
        if matrices.shape == (4, 32, 32):
            result = result + 1.0
        return result

    monkeypatch.setattr(
        "tools.newton_schulz_issue101_combined.run_newton_schulz_kernel", fake_kernel
    )
    monkeypatch.setattr(
        "tools.newton_schulz_issue101_combined.run_issue100_comparison",
        lambda *args, **kwargs: pytest.fail("performance must not run after a failed case"),
    )

    result = run_issue101_combined(object(), object(), repeats=1)

    assert result["status"] == "failed"
    assert result["failure_stage"] == "correctness"
    assert len(result["correctness_cases"]) == 3
    assert len(calls) == 3


def test_issue101_combined_driver_runs_performance_after_all_correctness_on_same_objects(
    monkeypatch,
):
    session = object()
    device = object()
    calls = []

    def fake_kernel(ttnn, selected_device, matrices, **kwargs):
        calls.append((ttnn, selected_device))
        return newton_schulz_reference(
            bf16_round_complex(matrices), x0=initial_value(matrices)
        )

    performance_calls = []

    def fake_performance(ttnn, selected_device, *, repeats, stop_on_failure):
        performance_calls.append((ttnn, selected_device, repeats, stop_on_failure))
        return [{"status": "ok"}] * 4

    monkeypatch.setattr(
        "tools.newton_schulz_issue101_combined.run_newton_schulz_kernel", fake_kernel
    )
    monkeypatch.setattr(
        "tools.newton_schulz_issue101_combined.run_issue100_comparison",
        fake_performance,
    )

    result = run_issue101_combined(session, device, repeats=7)

    assert result["status"] == "pass"
    assert len(calls) == len(ISSUE101_CORRECTNESS_CASES)
    assert performance_calls == [(session, device, 7, True)]


def test_issue101_combined_environment_uses_measurement_names():
    raw = {
        "image": "ghcr.io/example/image@sha256:" + "a" * 64,
        "image_pinned": True,
        "kernel": "Linux 6.8.0-test",
        "kmd_version": "2.11.0",
        "tt_env_active_release": "0.75.0",
        "harness_commit": "a" * 40,
        "harness_dirty": False,
        "board_info": {
            "board_type": "p150a",
            "serial": "serial",
        },
        "firmwares": {"fw_bundle_version": "19.6.0.0"},
    }

    environment = normalize_environment(raw)

    assert environment["image_digest"] == "sha256:" + "a" * 64
    assert environment["host_kernel"] == "Linux 6.8.0-test"
    assert environment["kernel_driver_version"] == "2.11.0"
    assert environment["board"]["board_id"] == "serial"
    assert environment["board"]["device_id"] == 0
    assert environment["firmware"]["fw_bundle_version"] == "19.6.0.0"


def _raw_environment(board: dict) -> dict:
    return {
        "image": "ghcr.io/example/image@sha256:" + "a" * 64,
        "image_pinned": True,
        "kernel": "Linux 6.8.0-test",
        "kmd_version": "2.11.0",
        "tt_env_active_release": "0.75.0",
        "harness_commit": "a" * 40,
        "harness_dirty": False,
        "board_info": board,
        "firmwares": {"fw_bundle_version": "19.6.0.0"},
    }


def test_issue101_board_id_only_is_a_deterministic_serial_alias():
    environment = normalize_environment(
        _raw_environment({"board_type": "p150a", "board_id": "board-only"})
    )

    assert environment["board"]["serial"] == "board-only"
    assert environment["board"]["board_id"] == "board-only"
    assert environment["board"]["serial_identity_source"] == "board_id_alias"
    assert environment["board_serial_identity"] == {
        "serial": "board-only",
        "board_id": "board-only",
        "source": "board_id_alias",
        "alias_applied": True,
        "rule": (
            "board_id is accepted as serial identity only when telemetry has no "
            "explicit serial; an explicit serial takes precedence and must match board_id."
        ),
    }


def test_issue101_explicit_serial_wins_and_mismatch_fails_fast():
    environment = normalize_environment(
        _raw_environment(
            {"board_type": "p150a", "serial": "explicit", "board_id": "explicit"}
        )
    )
    assert environment["board"]["serial"] == "explicit"
    assert environment["board_serial_identity"]["source"] == "explicit_serial"

    invalid_environment = _raw_environment(
        {"board_type": "p150a", "serial": "explicit", "board_id": "different"}
    )
    with pytest.raises(ValueError, match="serial and board_id"):
        normalize_environment(invalid_environment)


def _dummy_combined_run(repeats: int = 3) -> dict:
    correctness = [
        {
            "case": f"batch{batch}-L{size}",
            "batch": batch,
            "size": size,
            "reference": "BF16-rounded-R fixed-N=12 reference",
            "threshold": 0.01,
            "relative_error": 0.001,
            "status": "pass",
        }
        for batch, size in ISSUE101_CORRECTNESS_CASES
    ]
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
                    "seconds_per_launch_p50": samples[1],
                    "seconds_per_launch_p99": samples[-1],
                    "seconds_per_launch_p99_9": samples[-1],
                    "flops_per_iteration": 1.0,
                    "tflops_p50_derived": 0.001,
                    "tflops_fastest_launch_derived": 0.001,
                }
            )
    return {
        "status": "pass",
        "correctness_cases": correctness,
        "performance_rows": performance,
    }


def test_issue101_dummy_record_projection_is_stable_without_device():
    run_id = "fixture-run"
    run = _dummy_combined_run(repeats=3)
    run["run_id"] = run_id
    environment = {
        "captured_at": "2026-10-01T00:00:00+00:00",
        "image": "ghcr.io/example/image@sha256:" + "b" * 64,
        "image_digest": "sha256:" + "b" * 64,
        "image_pinned": True,
        "kernel": "Linux 6.8.0-fixture",
        "host_kernel": "Linux 6.8.0-fixture",
        "kmd_version": "2.11.0",
        "kernel_driver_version": "2.11.0",
        "tt_env_active_release": "0.75.0",
        "toolchain_release": "0.75.0",
        "python": "3.11.0",
        "harness_commit": "c" * 40,
        "harness_dirty": False,
        "run_id": run_id,
        "board": {
            "board_type": "p150a",
            "board_id": "fixture-board",
            "serial": "fixture-board",
        },
        "board_serial_identity": {
            "serial": "fixture-board",
            "board_id": "fixture-board",
        },
        "firmware": {"fw_bundle_version": "19.6.0.0"},
    }
    power_trace = {
        "file": "power-fixture.csv",
        "run_id": run_id,
        "samples": [{"timestamp_utc": "2026-10-01T00:00:00+00:00", "power_w": "75"}],
    }

    record = issue101._record_from_parts(
        run,
        environment=environment,
        power_trace=power_trace,
        repeats=3,
        raw_artifact_name="issue101-combined-raw-fixture.json",
        run_id=run_id,
    )
    expected_projection = {
        "record_schema": "adr-0005-issue101-combined-catalog-1000-v1",
        "status": "pass",
        "issue": "#100",
        "adr": "ADR-0005",
        "captured_at": "2026-10-01T00:00:00+00:00",
        "run_id": run_id,
        "harness_commit": "c" * 40,
        "correctness_reference": "BF16-rounded-R fixed-N=12 reference",
        "correctness_cases": run["correctness_cases"],
        "performance_rows": run["performance_rows"],
        "power_trace": "power-fixture.csv",
        "raw_artifact": {
            "schema": "adr-0005-issue101-combined-raw-v1",
            "file": "issue101-combined-raw-fixture.json",
            "run_id": run_id,
            "external_temporary": True,
            "not_committed": True,
        },
    }
    actual_projection = {
        "record_schema": record["record_schema"],
        "status": record["status"],
        "issue": record["issue"],
        "adr": record["adr"],
        "captured_at": record["captured_at"],
        "run_id": record["run_id"],
        "harness_commit": record["harness_commit"],
        "correctness_reference": record["measurement"]["correctness_reference"],
        "correctness_cases": record["measurement"]["correctness_cases"],
        "performance_rows": record["measurement"]["performance_rows"],
        "power_trace": record["power_trace"],
        "raw_artifact": record["raw_artifact"],
    }

    assert actual_projection == expected_projection
    assert json.dumps(actual_projection, sort_keys=True, separators=(",", ":")) == json.dumps(
        expected_projection, sort_keys=True, separators=(",", ":")
    )


def test_issue101_raw_artifact_survives_builder_failure(tmp_path, monkeypatch):
    output_dir = tmp_path / "device-output"
    output_dir.mkdir()
    (output_dir / "env-1.json").write_text(
        json.dumps(_raw_environment({"board_type": "p150a", "board_id": "board-only"}))
    )
    (output_dir / "power-1.csv").write_text(
        "timestamp_utc,power_w,aiclk_mhz,asic_temp_c\n"
        "2026-01-01T00:00:00+00:00,75,1350,60.0\n"
    )
    run = _dummy_combined_run()

    def fail_builder(*args, **kwargs):
        raise ValueError("synthetic builder failure")

    monkeypatch.setattr(issue101, "build_combined_record", fail_builder)
    with pytest.raises(RuntimeError, match="issue101-combined-raw.json"):
        issue101.persist_raw_and_build(run, output_dir=output_dir, repeats=3)

    raw_path = output_dir / issue101.ISSUE101_RAW_OUTPUT_NAME
    assert raw_path.exists()
    raw = json.loads(raw_path.read_text())
    assert raw["raw_schema"] == issue101.ISSUE101_RAW_SCHEMA
    assert raw["artifact_file"] == raw_path.name
    assert isinstance(raw["run_id"], str)
    assert raw["run"]["run_id"] == raw["run_id"]
    assert len(raw["correctness_results"]) == 9
    assert len(raw["performance_results"]) == 4
    assert len(raw["performance_results"][0]["seconds_per_launch_samples"]) == 3
    assert raw["telemetry"]["environment"]["board_info"]["board_id"] == "board-only"
    assert raw["telemetry"]["run_id"] == raw["run_id"]
    assert raw["telemetry"]["normalized_environment"] is not None
    assert raw["telemetry"]["power_trace"]["run_id"] == raw["run_id"]
    assert len(raw["telemetry"]["power_trace"]["samples"]) == 1
    assert raw["artifact_status"] == "failed"
    assert raw["failure"]["stage"] == "record_construction"
    assert raw["run"]["failure_stage"] == "record_construction"
    assert "/home/private/source" not in json.dumps(raw)
    assert "hostname" not in json.dumps(raw)
    assert "username" not in json.dumps(raw)
