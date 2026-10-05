import pytest

from enodia.tt.bench.newton_schulz_reference import (
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
)
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
