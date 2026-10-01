"""Intentional RED diagnostics for the block-8 BF16-X0 investigation.

These tests are board-free source/ledger checks.  They are intentionally red on
commit 0f43008: the production code is not changed here.  Keep them separate
from the passing contract tests until the owner selects and fixes a cause.
"""

from types import SimpleNamespace

from enodia.tt.bench import newton_schulz_kernel


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


def test_RED_diagnostic_block8_x0_bf16_l1_uses_largest_aligned_core_range():
    """RED: L1 input accounting must cover the largest aligned block-8 core."""
    ttnn = _ttnn()
    batch = 8192
    core_count = 110
    matrix_block = 8

    work_ranges = newton_schulz_kernel._balanced_ranges(batch, core_count, matrix_block)
    assert max(count for _, count in work_ranges) == 80

    actual_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=batch,
        core_count=core_count,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="l1",
        x0_dtype="bf16",
    )
    page_bf16 = 32 * 32 * 2
    expected_tensor_bytes = 80 * (3 * page_bf16 + 2 * page_bf16) + page_bf16 + 2 * page_bf16

    # The current implementation uses ceil(batch / core_count) == 75 instead
    # of the largest aligned range (80), so this is an intentional RED check.
    assert actual_tensor_bytes == expected_tensor_bytes


def test_RED_diagnostic_block8_x0_bf16_preflight_includes_largest_core_tensor_footprint():
    """RED: the advertised block-8 preflight must use the same max-core ledger."""
    ttnn = _ttnn()
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn,
        "fp32",
        fuse_s=True,
        matrix_block=8,
        x0_dtype="bf16",
    )

    actual_total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="l1",
        matrix_block=8,
        variant="bf16-fp32state",
        x0_dtype="bf16",
    )
    expected_total = (
        newton_schulz_kernel._L1_STATIC_BASE_BYTES
        + newton_schulz_kernel._cb_l1_bytes(ttnn, definitions)
        + 825_344
    )

    # Expected max-core total: 111,360 + 391,168 + 825,344 = 1,327,872.
    assert actual_total == expected_total
