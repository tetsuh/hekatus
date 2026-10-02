"""Strict JSON coverage for measurement and diagnostic record writers."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.spec.beamform import decimation_sweep
from enodia.spec.probe import linear_5mhz
from enodia.tt.bench import run_matmul, telemetry
from tools import newton_schulz_bringup as bringup


def _strict_load(text: str):
    def reject_constant(token: str):
        raise ValueError(f"non-standard JSON constant: {token}")

    return json.loads(text, parse_constant=reject_constant)


def test_shared_normalizer_recursively_maps_non_finite_and_numpy_scalars():
    from enodia.strict_json import normalize_json

    value = {
        np.int64(1): {
            "nan": np.float32(np.nan),
            "positive_infinity": float("inf"),
            "negative_infinity": -float("inf"),
            "sequence": (np.float64(2.5), np.bool_(True)),
        }
    }

    assert normalize_json(value) == {
        "1": {
            "nan": None,
            "positive_infinity": None,
            "negative_infinity": None,
            "sequence": [2.5, True],
        }
    }


def test_shared_normalizer_preserves_one_element_numpy_array_shape():
    from enodia.strict_json import dumps

    value = {"finite": np.array([1.0]), "non_finite": np.array([np.nan])}

    assert _strict_load(dumps(value)) == {"finite": [1.0], "non_finite": [None]}


def test_shared_normalizer_preserves_nested_numpy_array_shape():
    from enodia.strict_json import dumps

    value = np.array([[1.0, np.nan], [np.inf, -np.inf]])

    assert _strict_load(dumps({"value": value})) == {
        "value": [[1.0, None], [None, None]],
    }


def test_shared_normalizer_reduces_zero_dimensional_numpy_array_to_a_scalar():
    from enodia.strict_json import dumps

    assert _strict_load(dumps({"value": np.array(2.5)})) == {"value": 2.5}


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(np.float64).max,
    reason="NumPy longdouble has no wider range than float64 on this platform",
)
def test_shared_normalizer_preserves_finite_numpy_extended_precision_as_decimal_strings():
    from enodia.strict_json import dumps

    value = {
        "in_range": np.longdouble("1.234567890123456789"),
        "out_of_range": np.longdouble("1e400"),
    }

    assert _strict_load(dumps(value)) == {
        "in_range": "1.234567890123456789",
        "out_of_range": "1e+400",
    }


def test_shared_normalizer_maps_non_finite_numpy_extended_precision_to_null():
    from enodia.strict_json import dumps

    value = {
        "nan": np.longdouble("nan"),
        "positive_infinity": np.longdouble("inf"),
        "negative_infinity": np.longdouble("-inf"),
    }

    assert _strict_load(dumps(value)) == {
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
    }


def test_shared_normalizer_keeps_unknown_values_for_json_encoder_rejection():
    from enodia.strict_json import dumps

    with pytest.raises(TypeError):
        dumps({"unsupported": object()})


def test_shared_normalizer_keeps_unsupported_numpy_values_for_json_encoder_rejection():
    from enodia.strict_json import dumps

    with pytest.raises(TypeError):
        dumps({"unsupported": np.clongdouble(1 + 2j)})


def test_bringup_diagnostic_emitter_outputs_strict_json_for_nested_non_finite_values():
    stream = io.StringIO()

    bringup._emit_json(
        {
            "diagnostics": {
                "nan": np.float32(np.nan),
                "infinity": float("inf"),
                "values": [np.int64(7), -float("inf")],
            }
        },
        stream=stream,
    )

    assert _strict_load(stream.getvalue()) == {
        "diagnostics": {"infinity": None, "nan": None, "values": [7, None]}
    }


def test_environment_writer_outputs_strict_json_for_nested_non_finite_values(monkeypatch, tmp_path):
    monkeypatch.setattr(
        telemetry,
        "capture_environment",
        lambda image, image_pinned: {
            "image": image,
            "image_pinned": image_pinned,
            "nested": {"nan": float("nan"), "values": [float("inf"), -float("inf")]},
        },
    )
    output = tmp_path / "environment.json"

    telemetry.main(
        [
            "capture-env",
            "--out",
            str(output),
            "--image",
            "test-image",
        ]
    )

    assert _strict_load(output.read_text()) == {
        "image": "test-image",
        "image_pinned": False,
        "nested": {"nan": None, "values": [None, None]},
    }


def test_benchmark_writer_outputs_strict_json_for_nested_non_finite_values(monkeypatch, tmp_path):
    ttnn = SimpleNamespace(
        bfloat16="bf16",
        DRAM_MEMORY_CONFIG="dram",
        L1_MEMORY_CONFIG="l1",
        open_device=lambda device_id: object(),
        close_device=lambda device: None,
    )
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(
        run_matmul,
        "run_shape",
        lambda *args, **kwargs: {
            "status": "failed",
            "error": "host-only stub",
            "diagnostics": {"nan": float("nan"), "values": [float("inf")]},
        },
    )
    output = tmp_path / "results.json"

    assert (
        run_matmul.main(
            [
                "--only",
                "frontend_fir_taps64_w2",
                "--dtype",
                "bfloat16",
                "--memory",
                "dram",
                "--iters",
                "1",
                "--repeats",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    payload = _strict_load(output.read_text())
    assert payload["results"][0]["diagnostics"] == {"nan": None, "values": [None]}


def test_acceptance_catalogue_writer_outputs_strict_json(monkeypatch, tmp_path):
    ttnn = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        DRAM_MEMORY_CONFIG="dram",
        L1_MEMORY_CONFIG="l1",
        open_device=lambda device_id: object(),
        close_device=lambda device: None,
    )
    monkeypatch.setitem(sys.modules, "ttnn", ttnn)
    monkeypatch.setattr(
        run_matmul,
        "_run_acceptance_catalogue",
        lambda *args, **kwargs: [{"nan": float("nan"), "nested": [float("inf")]}],
    )
    output = tmp_path / "acceptance.json"

    assert run_matmul.main(["--acceptance-catalogue", "--out", str(output)]) == 0

    payload = _strict_load(output.read_text())
    assert payload["results"] == [{"nan": None, "nested": [None]}]


def test_measurement_writer_output_is_strict_json(monkeypatch, tmp_path):
    profile = linear_5mhz()
    result = decimation_sweep.SweepResult(
        profile=profile.name,
        bandwidth_status=profile.bandwidth_status,
        golden=(),
        iq={},
        reports={},
        seconds={"unavailable": float("nan")},
    )
    monkeypatch.setattr(decimation_sweep, "sweep", lambda *args: result)
    monkeypatch.setattr(
        "enodia.spec.sim.simulate_frame",
        lambda *args: [],
    )
    output = tmp_path / "measurement.json"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["decimation_sweep", "--record", str(output)])
    decimation_sweep.main()

    parsed = _strict_load(output.read_text())
    assert parsed["seconds"] == {"unavailable": None}
    assert parsed["profile"]["name"] == profile.name
