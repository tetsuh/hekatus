from pathlib import Path
from types import SimpleNamespace

import pytest

from enodia.tt.bench import newton_schulz_kernel, run_matmul

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


@pytest.mark.parametrize("matrix_block", [1, 2, 4])
def test_supported_matrix_blocks_validate_and_scale_matrix_queues(matrix_block):
    newton_schulz_kernel._validate_matrix_block(matrix_block)
    definitions = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=matrix_block
    )

    expected_queue_pages = 2 if matrix_block == 1 else matrix_block
    assert definitions[newton_schulz_kernel.CB_R_REAL][1] == expected_queue_pages
    assert definitions[newton_schulz_kernel.CB_OUTPUT_REAL][1] == expected_queue_pages
    # Fused S never routes products; keep their descriptors to one page for
    # compile-time CB identity without reserving unused block pages.
    assert definitions[newton_schulz_kernel.CB_PRODUCT_REAL][1] == 1
    # Constants remain resident singletons rather than consuming block slots.
    assert definitions[newton_schulz_kernel.CB_IDENTITY][1] == 1
    assert definitions[newton_schulz_kernel.CB_ZERO][1] == 1


def test_matrix_block_default_is_baseline_and_invalid_values_fail_host_side():
    assert newton_schulz_kernel.MATRIX_BLOCK_CHOICES == (1, 2, 4)
    baseline = newton_schulz_kernel._cb_definitions(_ttnn(), "fp32", fuse_s=True)
    explicit_baseline = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=1
    )
    assert baseline == explicit_baseline

    for invalid in (0, 3, 5, 8):
        with pytest.raises(ValueError, match="matrix_block"):
            newton_schulz_kernel._validate_matrix_block(invalid)


def test_dest_limit_uses_fp32_and_sync_mode_not_a_soft_block_cap():
    newton_schulz_kernel._validate_matrix_block(
        4, fp32_dest_acc_en=True, dst_full_sync_en=True, variant="bf16-fp32state"
    )
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            4, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
    with pytest.raises(ValueError, match="requires fp32_dest_acc_en"):
        newton_schulz_kernel._validate_matrix_block(
            2, fp32_dest_acc_en=False, dst_full_sync_en=True, variant="bf16-fp32state"
        )


def test_cb_l1_accounting_includes_batch_tensors_for_the_block4_target():
    ttnn = _ttnn()
    usages = []
    for matrix_block in (1, 2, 4):
        definitions = newton_schulz_kernel._cb_definitions(
            ttnn, "fp32", fuse_s=True, matrix_block=matrix_block
        )
        usages.append(newton_schulz_kernel._cb_l1_bytes(ttnn, definitions))
        tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
            ttnn,
            batch=8192,
            core_count=110,
            state_dtype="fp32",
            fuse_s=True,
            output_memory="dram",
        )
        total = newton_schulz_kernel._validate_l1_budget(
            ttnn, definitions, tensor_bytes=tensor_bytes
        )
        assert total == (
            newton_schulz_kernel._L1_STATIC_BASE_BYTES + usages[-1] + tensor_bytes
        )
    assert usages[0] < usages[1] < usages[2]
    assert usages[-1] <= newton_schulz_kernel._L1_CB_BUDGET_BYTES
    assert total <= newton_schulz_kernel._L1_TOTAL_BUDGET_BYTES


def test_matrix_block_ranges_keep_a_final_partial_group_and_align_core_ranges():
    assert newton_schulz_kernel._matrix_block_ranges(4, 6, 4) == [(4, 4), (8, 2)]
    ranges = newton_schulz_kernel._balanced_ranges(5, 2)
    assert ranges == [(0, 3), (3, 2)]
    assert newton_schulz_kernel._balanced_ranges(5, 2, 4) == [(0, 4), (4, 1)]

    aligned = newton_schulz_kernel._balanced_ranges(8192, 110, 4)
    assert {count for _, count in aligned} == {72, 76}
    assert all(start % 4 == 0 for start, _ in aligned)
    assert all(count % 4 == 0 for _, count in aligned)
    assert sum(count for _, count in aligned) == 8192


def test_reader_writer_stream_groups_and_pop_bulk():
    reader = (KERNEL_DIR / "newton_schulz_reader_optimized.cpp").read_text()
    writer = (KERNEL_DIR / "newton_schulz_writer.cpp").read_text()
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()

    assert "read_matrix_block" in reader
    assert '#include "api/compute/pack.h"' in compute
    assert "cb_reserve_back(cb_r_real, block_count)" in reader
    assert "cb_push_back(cb_r_real, block_count)" in reader
    assert "offset += matrix_block" in reader
    assert "cb_wait_front(cb_output_real, block_count)" in writer
    assert "cb_pop_front(cb_output_real, block_count)" in writer
    assert "pack_tile_block(0, output_real, block_count)" in compute
    assert "cb_push_back(output_real, block_count)" in compute
    assert "cb_pop_front(left_real, block_count)" in compute
    assert "fused_s_matmul_block" in compute


def test_cli_exposes_matrix_block_with_baseline_default():
    parser = run_matmul._build_parser()
    assert parser.parse_args([]).matrix_block == 1
    assert parser.parse_args(["--matrix-block", "2"]).matrix_block == 2
    assert parser.parse_args(["--matrix-block", "4"]).matrix_block == 4
    with pytest.raises(SystemExit):
        parser.parse_args(["--matrix-block", "3"])
