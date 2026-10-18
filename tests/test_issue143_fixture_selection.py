"""Structural guard for Issue #143's converted fixture dependencies."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

HEAVY_FIXTURES = {"frame", "golden", "sweep", "iq_path"}

CONVERTED_CASES = (
    (
        "contribution-mla",
        "tests/test_contribution_map.py",
        "test_the_beamformer_is_not_told_which_mla_it_is_running",
    ),
    (
        "contribution-iq-shaped",
        "tests/test_contribution_map.py",
        "test_the_iq_consumer_refuses_a_map_shaped_object_that_is_not_a_map",
    ),
    (
        "contribution-iq-double",
        "tests/test_contribution_map.py",
        "test_the_iq_consumer_refuses_a_double_that_only_claims_the_type",
    ),
    (
        "contribution-iq-map",
        "tests/test_contribution_map.py",
        "test_the_iq_path_reads_the_same_map_structure",
    ),
    (
        "golden-floor",
        "tests/test_golden_compare.py",
        "test_the_yardstick_floor_is_quoted_and_a_difference_below_it_is_flagged",
    ),
    (
        "golden-silent",
        "tests/test_golden_compare.py",
        "test_a_silent_reference_reports_nan_rather_than_dropping_the_event",
    ),
    (
        "golden-report",
        "tests/test_golden_compare.py",
        "test_the_report_says_what_the_sweep_measured_and_on_what",
    ),
    (
        "iq-decimation",
        "tests/test_iq_das.py",
        "test_the_beamformer_refuses_records_decimated_at_another_ratio",
    ),
    (
        "iq-integer",
        "tests/test_iq_das.py",
        "test_the_beamformer_rejects_an_integer_dtype",
    ),
    (
        "iq-dtype-float32-complex64",
        "tests/test_iq_das.py",
        "test_the_intermediate_is_complex_fp32_or_wider",
    ),
    (
        "iq-dtype-float64-complex128",
        "tests/test_iq_das.py",
        "test_the_intermediate_is_complex_fp32_or_wider",
    ),
)


def _source_root() -> Path:
    configured = os.environ.get("ISSUE143_SOURCE_ROOT")
    return Path(configured) if configured else Path(__file__).resolve().parents[1]


def _fixture_parameters(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        return {argument.arg for argument in arguments}
    raise AssertionError(f"could not find {function_name} in {path}")


@pytest.mark.parametrize("case,path,function_name", CONVERTED_CASES, ids=[case[0] for case in CONVERTED_CASES])
def test_converted_cases_request_no_heavy_fixture(case, path, function_name):
    """Converted cases must remain structurally independent of heavy fixtures."""
    requested = _fixture_parameters(_source_root() / path, function_name)

    assert not requested & HEAVY_FIXTURES, f"{case} requests {requested & HEAVY_FIXTURES}"
    assert requested & {"cheap_frame", "cheap_sweep", "cheap_silent_frame"}, (
        f"{case} no longer requests a shared cheap fixture"
    )
