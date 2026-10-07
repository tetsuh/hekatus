"""Board-free coverage for the explicit Issue #88 FP32-R variant."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel, run_matmul
from enodia.tt.bench.newton_schulz_reference import (
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)
from enodia.tt.bench.shapes import MatmulShape
from tools import newton_schulz_issue88 as issue88_runner

TTNN = SimpleNamespace(bfloat16="bf16", float32="fp32", uint32="u32")


def _shape(size: int = 32) -> MatmulShape:
    return MatmulShape(
        name=f"newton_schulz_L{size}_b8192",
        batch=8192,
        m=size,
        k=size,
        n=size,
        real_matmuls=4,
        family="newton_schulz",
        note="",
    )


def test_fp32_r_is_explicit_and_keeps_the_default_state_and_x0_formats():
    assert newton_schulz_kernel.DEFAULT_VARIANT == "bf16"
    assert newton_schulz_kernel.BENCHMARK_INPUT_SEED == 6300
    assert newton_schulz_kernel._state_dtype(TTNN, "fp32-r") == "bf16"
    assert newton_schulz_kernel._r_dtype(TTNN, "fp32-r") == "fp32"
    assert newton_schulz_kernel._r_format("fp32-r") == "FP32"
    assert newton_schulz_kernel._reader_input_dtypes(
        TTNN, "bf16", fuse_s=True, r_dtype="fp32"
    ) == ["fp32", "fp32", "fp32", "bf16", "bf16", "bf16", "fp32"]

    definitions = newton_schulz_kernel._cb_definitions(
        TTNN,
        "bf16",
        fuse_s=True,
        matrix_block=8,
        double_buffer=True,
        r_dtype="fp32",
    )
    for index in (
        newton_schulz_kernel.CB_R_NEG_IMAG,
        newton_schulz_kernel.CB_R_IMAG,
        newton_schulz_kernel.CB_R_NEG_REAL,
    ):
        assert definitions[index][0] == "fp32"
    assert definitions[newton_schulz_kernel.CB_X0_REAL][0] == "bf16"
    assert definitions[newton_schulz_kernel.CB_STATE_REAL][0] == "bf16"


def test_fp32_r_default_l32_l1_preflight_fails_before_runner_prepare():
    called = False

    def fail_prepare(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("FP32-R L1 rejection must precede kernel prepare")

    original_prepare = newton_schulz_kernel.NewtonSchulzKernel.prepare
    newton_schulz_kernel.NewtonSchulzKernel.prepare = fail_prepare
    try:
        record = run_matmul.run_custom_newton_schulz(
            TTNN,
            device=object(),
            shape=_shape(),
            dtype_name="bfloat16",
            memory_name="l1",
            variant="fp32-r",
            iters=1,
            repeats=1,
        )
    finally:
        newton_schulz_kernel.NewtonSchulzKernel.prepare = original_prepare

    assert record["status"] == "failed"
    assert record["variant"] == "fp32-r"
    assert record["r_format"] == "FP32"
    assert "matrix_block=8 L1 preflight failed" in record["error"]
    assert "total=1884928 bytes" in record["error"]
    assert called is False


def test_fp32_r_l1_ledger_exposes_the_nearest_fitting_alternatives():
    common = {
        "core_count": 110,
        "state_dtype": "bf16",
        "variant": "fp32-r",
        "fuse_s": True,
        "input_memory": "l1",
        "x0_memory": "l1",
        "output_memory": "dram",
        "double_buffer": True,
    }

    # Keeping the selected block and moving only R to DRAM is an explicit fit.
    assert newton_schulz_kernel._validate_l1_preflight(
        TTNN, batch=8192, matrix_block=8, r_memory="dram", **common
    ) == 901888

    # Block 4 is the immediate smaller block but still fails; block 2 is the
    # nearest smaller fitting choice and remains an explicit caller decision.
    with pytest.raises(ValueError, match="total=1581824 bytes"):
        newton_schulz_kernel._validate_l1_preflight(
            TTNN, batch=8192, matrix_block=4, r_memory="l1", **common
        )
    assert newton_schulz_kernel._validate_l1_preflight(
        TTNN, batch=8192, matrix_block=2, r_memory="l1", **common
    ) == 1479424

    # The required L=16 diagonal-pair case fits with the default placement.
    l16_tiles = newton_schulz_kernel._padded_tile_count(
        newton_schulz_kernel._physical_tile_count(8192, 16), 8
    )
    assert newton_schulz_kernel._validate_l1_preflight(
        TTNN, batch=l16_tiles, matrix_block=8, r_memory="l1", **common
    ) == 1229568


def test_fp32_r_tensor_and_cb_accounting_double_only_r_bytes():
    common = {
        "batch": 8192,
        "core_count": 110,
        "state_dtype": "bf16",
        "fuse_s": True,
        "input_memory": "l1",
        "r_memory": "l1",
        "x0_memory": "l1",
        "output_memory": "dram",
        "matrix_block": 8,
        "double_buffer": True,
    }
    bf16 = newton_schulz_kernel._validate_l1_preflight(
        TTNN, variant="bf16", **common
    )
    tensor_common = {key: value for key, value in common.items() if key != "double_buffer"}
    fp32_r_tensor = newton_schulz_kernel._tensor_l1_bytes(
        TTNN, r_dtype="fp32", **tensor_common
    )
    fp32_r_definitions = newton_schulz_kernel._cb_definitions(
        TTNN,
        "bf16",
        fuse_s=True,
        matrix_block=8,
        double_buffer=True,
        r_dtype="fp32",
    )
    fp32_r_cb = newton_schulz_kernel._cb_l1_bytes(TTNN, fp32_r_definitions)

    assert bf16 == 1295104
    assert fp32_r_tensor == 1316864
    assert fp32_r_cb == 456704
    with pytest.raises(ValueError, match="total=1884928 bytes"):
        newton_schulz_kernel._validate_l1_preflight(
            TTNN, variant="fp32-r", **common
        )


def test_fp32_r_parser_and_independent_reference_are_explicit():
    args = run_matmul._build_parser().parse_args(["--custom-variant", "fp32-r"])
    assert args.custom_variant == "fp32-r"

    matrices = random_hpd_batch(2, 16, seed=8800)
    expected = newton_schulz_reference(matrices, x0=initial_value(matrices))
    assert expected.dtype == np.dtype(np.complex64)
    assert np.all(np.isfinite(expected))


def test_issue88_plan_matches_runner_seed_and_has_executable_placements():
    plan_path = (
        Path(__file__).parents[1]
        / "docs/measurements/2026-10-07-host-newton-schulz-issue88-fp32-r-plan.json"
    )
    plan = json.loads(plan_path.read_text())

    assert plan["comparison_plan"]["seed"] == newton_schulz_kernel.BENCHMARK_INPUT_SEED
    first_launch_parser = run_matmul._build_parser()
    first_launch_args = first_launch_parser.parse_args(
        plan["first_launch_plan"]["runner_args"]
    )
    run_matmul._validate(first_launch_parser, first_launch_args)

    comparison_parser = issue88_runner._build_parser()
    for section in plan["comparison_plan"]["rows"]:
        args = comparison_parser.parse_args(section["runner_args"])
        issue88_runner._validate_args(comparison_parser, args)
        assert args.custom_variant == section["variant"]
        assert args.r_memory == section["r_memory"]
        assert args.x0_memory == section["x0_memory"]
        assert args.matrix_block == int(plan["algorithm"]["matrix_block"])

    l32_fp32 = plan["comparison_plan"]["rows"][-1]
    assert l32_fp32["variant"] == "fp32-r"
    assert l32_fp32["r_memory"] == "dram"
