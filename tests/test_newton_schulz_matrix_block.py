import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel, run_matmul

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


@pytest.mark.parametrize("matrix_block", [1, 2, 4, 8])
def test_supported_matrix_blocks_validate_and_scale_matrix_queues(matrix_block):
    newton_schulz_kernel._validate_matrix_block(matrix_block)
    definitions = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=matrix_block
    )

    expected_queue_pages = 2 if matrix_block == 1 else matrix_block
    matrix_queue_indices = (
        newton_schulz_kernel.CB_R_REAL,
        newton_schulz_kernel.CB_R_NEG_IMAG,
        newton_schulz_kernel.CB_R_IMAG,
        newton_schulz_kernel.CB_X0_REAL,
        newton_schulz_kernel.CB_X0_IMAG,
        newton_schulz_kernel.CB_S_REAL,
        newton_schulz_kernel.CB_S_IMAG,
        newton_schulz_kernel.CB_NEG_X_IMAG,
        newton_schulz_kernel.CB_R_NEG_REAL,
        newton_schulz_kernel.CB_OUTPUT_REAL,
        newton_schulz_kernel.CB_OUTPUT_IMAG,
    )
    if matrix_block == 1:
        assert definitions[newton_schulz_kernel.CB_R_REAL][1] == expected_queue_pages
        assert definitions[newton_schulz_kernel.CB_OUTPUT_REAL][1] == expected_queue_pages
    else:
        assert all(definitions[index][1] == expected_queue_pages for index in matrix_queue_indices)
    state_queue_pages = 2 * matrix_block
    assert definitions[newton_schulz_kernel.CB_STATE_REAL][1] == state_queue_pages
    assert definitions[newton_schulz_kernel.CB_STATE_IMAG][1] == state_queue_pages
    # Fused S never routes products; keep their descriptors to one page for
    # compile-time CB identity without reserving unused block pages.
    assert definitions[newton_schulz_kernel.CB_PRODUCT_REAL][1] == 1
    assert definitions[newton_schulz_kernel.CB_PRODUCT_IMAG][1] == 1
    # Constants remain resident singletons rather than consuming block slots.
    assert definitions[newton_schulz_kernel.CB_IDENTITY][1] == 1
    assert definitions[newton_schulz_kernel.CB_ZERO][1] == 1


def test_matrix_block_default_is_baseline_and_invalid_values_fail_host_side():
    assert newton_schulz_kernel.MATRIX_BLOCK_CHOICES == (1, 2, 4, 8)
    baseline = newton_schulz_kernel._cb_definitions(_ttnn(), "fp32", fuse_s=True)
    explicit_baseline = newton_schulz_kernel._cb_definitions(
        _ttnn(), "fp32", fuse_s=True, matrix_block=1
    )
    assert baseline == explicit_baseline

    for invalid in (0, 3, 5):
        with pytest.raises(ValueError, match="matrix_block"):
            newton_schulz_kernel._validate_matrix_block(invalid)


def test_dest_limit_uses_fp32_and_sync_mode_not_a_soft_block_cap():
    for matrix_block in (1, 2, 4):
        newton_schulz_kernel._validate_matrix_block(
            matrix_block,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
            variant="bf16-fp32state",
        )
    # Block 8 uses one DEST half at a time and reaches the eight-slot FP32
    # limit, while the existing blocks retain two DEST slots per matrix.
    newton_schulz_kernel._validate_matrix_block(
        8, fp32_dest_acc_en=True, dst_full_sync_en=True, variant="bf16-fp32state"
    )
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            4, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            8, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
    with pytest.raises(ValueError, match="requires fp32_dest_acc_en"):
        newton_schulz_kernel._validate_matrix_block(
            2, fp32_dest_acc_en=False, dst_full_sync_en=True, variant="bf16-fp32state"
        )


@pytest.mark.parametrize("matrix_block", [1, 2, 4, 8])
def test_cb_l1_accounting_uses_two_state_windows_and_dram_inputs_fit(matrix_block):
    ttnn = _ttnn()
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=matrix_block
    )
    expected_cb_bytes = {1: 92160, 2: 120832, 4: 227328, 8: 440320}
    expected_total_bytes = {1: 203520, 2: 232192, 4: 338688, 8: 551680}

    assert definitions[newton_schulz_kernel.CB_STATE_REAL][1] == 2 * matrix_block
    assert definitions[newton_schulz_kernel.CB_STATE_IMAG][1] == 2 * matrix_block
    assert (
        newton_schulz_kernel._cb_l1_bytes(ttnn, definitions)
        == expected_cb_bytes[matrix_block]
    )

    total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="dram",
        matrix_block=matrix_block,
        variant="bf16-fp32state",
    )
    assert total == expected_total_bytes[matrix_block]
    assert total == (
        newton_schulz_kernel._L1_STATIC_BASE_BYTES
        + expected_cb_bytes[matrix_block]
    )


def test_l1_preflight_rejects_l1_input_when_state_windows_exceed_budget():
    ttnn = _ttnn()
    expected_total_bytes = {1: 1438464, 2: 1467136}
    for matrix_block in (1, 2):
        total = newton_schulz_kernel._validate_l1_preflight(
            ttnn,
            batch=8192,
            core_count=110,
            state_dtype="fp32",
            fuse_s=True,
            output_memory="dram",
            matrix_block=matrix_block,
            variant="bf16-fp32state",
        )
        assert total == expected_total_bytes[matrix_block]

    for matrix_block, overage in ((4, 768), (8, 213760)):
        with pytest.raises(
            ValueError, match=f"matrix_block={matrix_block} L1 preflight failed"
        ) as excinfo:
            newton_schulz_kernel._validate_l1_preflight(
                ttnn,
                batch=8192,
                core_count=110,
                state_dtype="fp32",
                fuse_s=True,
                output_memory="dram",
                matrix_block=matrix_block,
                variant="bf16-fp32state",
            )
        assert f"L1 budget over by {overage} bytes" in str(excinfo.value)


def test_block8_l1_preflight_rejects_with_full_accounting_and_cb_breakdown():
    ttnn = _ttnn()
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=8
    )
    tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
    )

    with pytest.raises(ValueError) as excinfo:
        newton_schulz_kernel._validate_l1_budget(
            ttnn, definitions, tensor_bytes=tensor_bytes, matrix_block=8
        )

    message = str(excinfo.value)
    assert "matrix_block=8 L1 preflight failed" in message
    assert "total CB bytes=440320" in message
    assert "static prefix=111360 bytes" in message
    assert "tensor bytes=1234944" in message
    assert "total=1786624 bytes" in message
    assert "budget=1572864 bytes" in message
    assert "L1 budget over by 213760 bytes" in message
    assert "largest CBs:" in message
    assert "CB_STATE_REAL=65536 bytes (cb_state_real)" in message
    assert "CBs in over-budget total:" in message
    assert "CB_X0_REAL=32768 bytes (cb_x0_real)" in message
    assert "CB_PRODUCT_REAL=4096 bytes (cb_product_real)" in message


def test_matrix_block_ranges_keep_a_final_partial_group_and_align_core_ranges():
    assert newton_schulz_kernel._matrix_block_ranges(4, 6, 4) == [(4, 4), (8, 2)]
    assert newton_schulz_kernel._balanced_ranges(9, 2, 8) == [(0, 8), (8, 1)]
    ranges = newton_schulz_kernel._balanced_ranges(5, 2)
    assert ranges == [(0, 3), (3, 2)]
    assert newton_schulz_kernel._balanced_ranges(5, 2, 4) == [(0, 4), (4, 1)]

    aligned = newton_schulz_kernel._balanced_ranges(8192, 110, 4)
    assert {count for _, count in aligned} == {72, 76}
    assert all(start % 4 == 0 for start, _ in aligned)
    assert all(count % 4 == 0 for _, count in aligned)
    assert sum(count for _, count in aligned) == 8192


def test_input_memory_uses_one_interleaved_placement_for_every_device_input():
    class _Tensor:
        pass

    class _Ttnn:
        TILE_LAYOUT = "tile"
        L1_MEMORY_CONFIG = "l1"
        DRAM_MEMORY_CONFIG = "dram"

        def Tensor(self, values, dtype):
            tensor = _Tensor()
            tensor.values = values
            tensor.dtype = dtype
            return tensor

        def to_layout(self, tensor, layout):
            tensor.layout = layout
            return tensor

        def to_device(self, tensor, device, *, memory_config):
            tensor.device = device
            tensor.memory_config = memory_config
            return tensor

    ttnn = _Ttnn()
    values = np.zeros((1, 1, 32, 32), dtype=np.float32)
    assert newton_schulz_kernel._device_tensor(
        ttnn, values, object(), dtype="bf16"
    ).memory_config == "l1"
    assert newton_schulz_kernel._device_tensor(
        ttnn, values, object(), dtype="bf16", input_memory="dram"
    ).memory_config == "dram"
    with pytest.raises(ValueError, match="input_memory"):
        newton_schulz_kernel._device_tensor(
            ttnn, values, object(), dtype="bf16", input_memory="sram"
        )


def test_dram_inputs_remove_tensor_l1_bytes_but_keep_static_cb_accounting():
    ttnn = _ttnn()
    l1_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="l1",
    )
    dram_tensor_bytes = newton_schulz_kernel._tensor_l1_bytes(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="dram",
    )
    assert l1_tensor_bytes == 1_234_944
    assert dram_tensor_bytes == 0
    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", fuse_s=True, matrix_block=8
    )
    total = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        fuse_s=True,
        output_memory="dram",
        input_memory="dram",
        matrix_block=8,
        variant="bf16-fp32state",
    )
    assert total == (
        newton_schulz_kernel._L1_STATIC_BASE_BYTES
        + newton_schulz_kernel._cb_l1_bytes(ttnn, definitions)
    )
    assert total == 551_680
    with pytest.raises(ValueError, match="input_memory"):
        newton_schulz_kernel.NewtonSchulzKernel.prepare(
            None,
            None,
            np.zeros((1, 32, 32), dtype=np.complex64),
            input_memory="sram",
        )


def _function_source(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated C++ function: {signature}")


_CB_CALL = re.compile(
    r"\b(cb_(?:reserve_back|push_back|pop_front))\(\s*([A-Za-z_][A-Za-z0-9_]*)"
)


def _cb_ledger(source: str) -> Counter:
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    return Counter(_CB_CALL.findall(source))


def _complex_block_branch_source(source: str, *, one_dest: bool) -> str:
    function = _function_source(source, "void complex_matmul_block(")
    branch_start = function.index("if constexpr (one_dest_half) {")
    else_start = function.index("    } else {", branch_start)
    post_start = function.index("\n    if constexpr (one_dest_half) {", else_start)
    right_start = function.index("\n    if (consume_right)", post_start)
    if one_dest:
        return function[branch_start:else_start] + function[post_start:]
    return function[else_start:post_start] + function[right_start:]


@pytest.mark.parametrize(
    ("matrix_block", "one_dest"),
    ((1, False), (2, False), (4, False), (8, True)),
)
def test_matrix_block_branch_ledgers_publish_and_consume_every_page(
    matrix_block, one_dest
):
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    expected = Counter(
        {
            ("cb_reserve_back", "output_real"): 1,
            ("cb_reserve_back", "output_imag"): 1,
            ("cb_push_back", "output_real"): 1,
            ("cb_push_back", "output_imag"): 1,
            ("cb_pop_front", "left_real"): 1,
            ("cb_pop_front", "left_imag_for_real"): 1,
            ("cb_pop_front", "left_imag_for_imag"): 1,
            ("cb_pop_front", "right_real"): 1,
            ("cb_pop_front", "right_imag"): 1,
        }
    )

    if matrix_block == 1:
        real = _function_source(compute, "void complex_real_impl(")
        imag = _function_source(compute, "void complex_imag_impl(")
        pack = _function_source(compute, "void pack_one(")
        single = _function_source(compute, "void complex_matmul(")
        assert _cb_ledger(real + imag) == Counter(
            {
                ("cb_reserve_back", "output"): 2,
            }
        )
        assert _cb_ledger(pack) == Counter({("cb_push_back", "output"): 1})
        assert real.count("pack_one(output)") == 1
        assert imag.count("pack_one(output)") == 1
        assert _cb_ledger(single) == Counter(
            {
                ("cb_pop_front", "left_real"): 1,
                ("cb_pop_front", "left_imag_for_real"): 1,
                ("cb_pop_front", "left_imag_for_imag"): 1,
                ("cb_pop_front", "right_real"): 1,
                ("cb_pop_front", "right_imag"): 1,
            }
        )
        return

    branch = _complex_block_branch_source(compute, one_dest=one_dest)
    assert _cb_ledger(branch) == expected
    pack_offsets = {
        "output_real": branch.index("pack_tile_block(0, output_real, block_count)"),
        "output_imag": branch.index(
            "pack_tile_block(0, output_imag, block_count)"
            if one_dest
            else "pack_tile_block(block_count, output_imag, block_count)"
        ),
    }
    for output, pack_offset in pack_offsets.items():
        push_offset = branch.index(f"cb_push_back({output}, block_count)")
        assert pack_offset < push_offset


def test_block8_compute_uses_one_dest_half_for_products_s_and_output():
    compute = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    complex_start = compute.index("template <bool one_dest_half>\nvoid complex_matmul_block")
    fused_start = compute.index("template <bool one_dest_half>\nvoid fused_s_matmul_block")
    complex_source = compute[complex_start:fused_start]
    assert "if constexpr (one_dest_half)" in complex_source
    assert complex_source.count("tile_regs_acquire();") == 3
    assert complex_source.count("tile_regs_commit();") == 3
    assert "pack_tile_block(0, output_real, block_count);" in complex_source
    assert "pack_tile_block(0, output_imag, block_count);" in complex_source
    two_dest_source = complex_source[complex_source.index("    } else {") :]
    assert two_dest_source.index("cb_pop_front(left_real, block_count)") < two_dest_source.index(
        "cb_reserve_back(output_real, block_count)"
    )
    assert "complex_matmul_block<one_dest_half>" in compute
    assert "fused_s_matmul_block<one_dest_half>" in compute
    assert "process_matrix_block<iterations, state_fp32, fuse_s, (matrix_block == 8)>" in compute


def test_invalid_input_memory_is_rejected_before_custom_prepare():
    shape = SimpleNamespace(family="newton_schulz", m=32, k=32, n=32, batch=8192)
    from enodia.tt.bench import run_matmul

    record = run_matmul.run_custom_newton_schulz(
        object(),
        object(),
        shape,
        dtype_name="bfloat16",
        memory_name="l1",
        variant="bf16-fp32state",
        input_memory="sram",
        iters=1,
        repeats=1,
    )
    assert record["status"] == "failed"
    assert "input_memory" in record["error"]


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
    assert parser.parse_args([]).input_memory == "l1"
    assert parser.parse_args(["--input-memory", "dram"]).input_memory == "dram"
    assert parser.parse_args(["--matrix-block", "2"]).matrix_block == 2
    assert parser.parse_args(["--matrix-block", "4"]).matrix_block == 4
    assert parser.parse_args(["--matrix-block", "8"]).matrix_block == 8
    with pytest.raises(SystemExit):
        parser.parse_args(["--matrix-block", "3"])
