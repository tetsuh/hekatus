"""Board-free tests for the Issue #88 comparison runner."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from enodia.tt.bench.newton_schulz_reference import (
    bf16_round_complex,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)
from tools import newton_schulz_issue88 as runner

PLAN = Path(__file__).parents[1] / "docs/measurements/2026-10-07-host-newton-schulz-issue88-fp32-r-plan.json"


def test_issue88_catalogue_pins_every_execution_control_and_placement():
    runner.validate_comparison_rows()

    rows = runner.ISSUE88_COMPARISON_ROWS
    assert [(row["variant"], row["size"], row["r_memory"]) for row in rows] == [
        ("bf16", 16, "l1"),
        ("fp32-r", 16, "l1"),
        ("bf16", 32, "l1"),
        ("fp32-r", 32, "dram"),
    ]
    assert {
        key: rows[0][key]
        for key in (
            "batch",
            "iterations",
            "initial_value",
            "state_format",
            "destination_format",
            "math_fidelity",
            "fuse_s",
            "matrix_block",
            "double_buffer",
            "dst_full_sync_en",
            "x0_memory",
            "output_memory",
        )
    } == {
        "batch": 8192,
        "iterations": 12,
        "initial_value": "I/||R||inf",
        "state_format": "BF16",
        "destination_format": "FP32",
        "math_fidelity": "HiFi3",
        "fuse_s": True,
        "matrix_block": 8,
        "double_buffer": True,
        "dst_full_sync_en": True,
        "x0_memory": "l1",
        "output_memory": "dram",
    }


def test_issue88_timing_summary_retains_samples_and_all_derivations():
    summary = runner.timing_summary([1.0, 2.0, 4.0], flops_per_launch=8)

    assert summary["seconds_per_launch_samples"] == [1.0, 2.0, 4.0]
    assert summary["seconds_per_launch_min"] == 1.0
    assert summary["seconds_per_launch_p50"] == 2.0
    assert summary["seconds_per_launch_p99"] == pytest.approx(3.96)
    assert summary["seconds_per_launch_p99_9"] == pytest.approx(3.996)
    assert summary["tflops_min_derived"] == pytest.approx(8e-12)
    assert summary["tflops_p50_derived"] == pytest.approx(4e-12)


def test_issue88_reference_context_uses_original_x0_for_both_r_formats():
    matrices = random_hpd_batch(2, 4, condition_number=100.0, seed=runner.INPUT_SEED)
    x0 = initial_value(matrices)
    bf16_context = runner.reference_context(matrices, "bf16")
    fp32_context = runner.reference_context(matrices, "fp32-r")

    np.testing.assert_array_equal(bf16_context["x0"], x0)
    np.testing.assert_array_equal(fp32_context["x0"], x0)
    np.testing.assert_array_equal(bf16_context["reference_r"], bf16_round_complex(matrices))
    np.testing.assert_array_equal(fp32_context["reference_r"], matrices)
    np.testing.assert_allclose(
        bf16_context["fixed_reference"],
        newton_schulz_reference(bf16_round_complex(matrices), x0=x0),
    )
    np.testing.assert_allclose(
        fp32_context["fixed_reference"],
        newton_schulz_reference(matrices, x0=x0),
    )


def test_issue88_same_array_metrics_are_zero_for_identical_inverses():
    inverse = np.broadcast_to(np.eye(4, dtype=np.complex128), (2, 4, 4)).copy()
    metrics = runner.same_array_metrics(inverse, inverse)

    assert metrics["mv_weight_direction"]["max_cosine_deficit"] == pytest.approx(0.0)
    assert metrics["beam_pattern"]["relative_frobenius_error"] == pytest.approx(0.0)
    assert metrics["beam_pattern"]["max_absolute_error"] == pytest.approx(0.0)


def test_issue88_runner_has_no_spec_reference_and_matches_plan():
    source = Path(runner.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert all(not module.startswith("enodia.spec") for module in imported_modules)

    plan = json.loads(PLAN.read_text())
    assert plan["comparison_plan"]["runner"] == runner.ISSUE88_RUNNER
    assert plan["comparison_plan"]["input_generator"].endswith("random_hpd_batch")
    assert plan["comparison_plan"]["condition_number"] == runner.CONDITION_NUMBER
    assert plan["comparison_plan"]["seed"] == runner.INPUT_SEED
    parser = runner._build_parser()
    for row in plan["comparison_plan"]["rows"]:
        args = parser.parse_args(row["runner_args"])
        runner._validate_args(parser, args)


def test_issue88_runner_parser_rejects_the_forbidden_fp32_l32_l1_placement():
    parser = runner._build_parser()
    args = parser.parse_args(
        [
            "--only",
            "newton_schulz_L32_b8192",
            "--custom-variant",
            "fp32-r",
            "--r-memory",
            "l1",
        ]
    )
    with pytest.raises(SystemExit):
        runner._validate_args(parser, args)
