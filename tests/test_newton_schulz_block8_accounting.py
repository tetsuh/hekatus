"""Board-free regression tests for block-8 L1 accounting and placement."""

from types import SimpleNamespace

from enodia.tt.bench import newton_schulz_kernel


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


def test_block8_accounting_uses_largest_aligned_core_range():
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
        matrix_block=matrix_block,
    )
    page_bf16 = 32 * 32 * 2
    page_fp32 = 32 * 32 * 4
    expected_tensor_bytes = 80 * (3 * page_bf16 + 2 * page_fp32) + page_bf16 + page_fp32

    assert actual_tensor_bytes == expected_tensor_bytes == 1_153_024


def test_block8_preflight_passes_with_r_l1_and_x0_dram():
    ttnn = _ttnn()
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn,
        "fp32",
        fuse_s=True,
        matrix_block=8,
    )

    actual_total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        r_memory="l1",
        x0_memory="dram",
        matrix_block=8,
        double_buffer=False,
        variant="bf16-fp32state",
    )
    expected_total = (
        newton_schulz_kernel._L1_STATIC_BASE_BYTES
        + newton_schulz_kernel._cb_l1_bytes(ttnn, definitions)
        + 497_664
    )

    assert actual_total == expected_total == 1_032_960
