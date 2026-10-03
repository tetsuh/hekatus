"""Host contracts for the opt-in two-tile complex-product path."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel, run_matmul

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


def test_two_tile_r_host_layout_preserves_r_x_order_and_signs():
    matrices = np.zeros((1, 32, 32), dtype=np.complex64)
    matrices[0].real.fill(3.0)
    matrices[0].imag.fill(5.0)
    x0 = np.zeros_like(matrices)

    values = newton_schulz_kernel._reader_input_values(
        matrices,
        x0,
        fuse_s=False,
        tile_count=1,
        two_tile_complex=True,
    )

    assert len(values) == 3
    assert values[0].shape == (1, 4, 32, 32)
    np.testing.assert_array_equal(values[0][0, 0], -3.0)
    np.testing.assert_array_equal(values[0][0, 1], -5.0)
    np.testing.assert_array_equal(values[0][0, 2], 5.0)
    np.testing.assert_array_equal(values[0][0, 3], -3.0)
    np.testing.assert_array_equal(values[1], np.zeros((1, 1, 32, 32), dtype=np.float32))
    np.testing.assert_array_equal(values[2], np.zeros((1, 1, 32, 32), dtype=np.float32))


def test_two_tile_reader_descriptor_uses_state_inputs_and_bfloat16_constants():
    ttnn = _ttnn()

    assert newton_schulz_kernel._reader_input_dtypes(
        ttnn, "fp32", fuse_s=False, two_tile_complex=True
    ) == ["bf16", "fp32", "fp32", "bf16", "bf16"]
    assert newton_schulz_kernel._reader_input_memories(
        two_tile_complex=True, input_memory="dram", r_memory="l1", x0_memory="dram"
    ) == ["l1", "dram", "dram", "dram", "dram"]

    definitions = newton_schulz_kernel._cb_definitions(
        ttnn, "fp32", two_tile_complex=True, matrix_block=2
    )
    assert newton_schulz_kernel.CB_R_REAL not in definitions
    assert newton_schulz_kernel.CB_R_NEG_IMAG not in definitions
    assert newton_schulz_kernel.CB_PRODUCT_REAL not in definitions
    assert newton_schulz_kernel.CB_PRODUCT_IMAG not in definitions
    assert definitions[newton_schulz_kernel.CB_TWO_TILE_R] == ("bf16", 8)
    assert definitions[newton_schulz_kernel.CB_TWO_TILE_X] == ("fp32", 8)
    assert definitions[newton_schulz_kernel.CB_TWO_TILE_S] == ("fp32", 4)
    assert definitions[newton_schulz_kernel.CB_IDENTITY] == ("bf16", 1)
    assert definitions[newton_schulz_kernel.CB_ZERO] == ("bf16", 1)


def test_bfloat16_initial_constants_are_bit_exact_and_zero_is_explicit():
    identity, zero = newton_schulz_kernel._two_tile_initial_values()

    identity_bits = newton_schulz_kernel._bfloat16_bits(identity.ravel())
    zero_bits = newton_schulz_kernel._bfloat16_bits(zero.ravel())
    assert set(identity_bits.tolist()) == {0, 0x4000}
    assert np.count_nonzero(identity_bits == 0x4000) == 32
    assert np.all(zero_bits == 0)
    assert np.count_nonzero(identity) == 32
    assert np.count_nonzero(zero) == 0


def _llk_rt2_kt1_accumulate(
    in0_pages: list[np.ndarray],
    in1_pages: list[np.ndarray],
    *,
    in0_start: int,
    in1_start: int,
    destination: np.ndarray,
    destination_start: int,
) -> None:
    """Model the pinned Blackhole ``ct=1, rt=2, kt=1`` traversal.

    ``matmul_block`` maps in0 to SrcB and in1 to SrcA, but the mathematical
    operands remain ``A=in0`` and ``B=in1``.  With one output column the
    unpack MOP visits two consecutive in0 pages (the two output rows) and
    reuses the one in1 page.  The math MOP writes destination slots in row
    order and accumulates rather than replacing them.
    """
    for row in range(2):
        destination[destination_start + row] += (
            in0_pages[in0_start + row] @ in1_pages[in1_start]
        )


def test_two_tile_llk_traversal_matches_complex_r_x_and_x_s_products():
    """Verify both physical page layouts against the pinned LLK traversal."""
    r_real = np.array([[2.0, 3.0], [5.0, 7.0]])
    r_imag = np.array([[11.0, 13.0], [17.0, 19.0]])
    x_real = np.array([[23.0, 29.0], [31.0, 37.0]])
    x_imag = np.array([[41.0, 43.0], [47.0, 53.0]])
    # The two K=1 products consume each operand block by columns: [A00, A10]
    # then [A01, A11].  This is why each four-page CB block is column-major,
    # even though each tile itself is row-major.
    r_pages = [-r_real, -r_imag, r_imag, -r_real]
    x_pages = [x_real, x_imag, -x_imag, x_real]
    x_column = [x_real, x_imag]
    s_dest = np.stack((2.0 * np.eye(2), np.zeros((2, 2))))
    _llk_rt2_kt1_accumulate(
        r_pages,
        x_column,
        in0_start=0,
        in1_start=0,
        destination=s_dest,
        destination_start=0,
    )
    _llk_rt2_kt1_accumulate(
        r_pages,
        x_column,
        in0_start=2,
        in1_start=1,
        destination=s_dest,
        destination_start=0,
    )
    expected_s = 2.0 * np.eye(2, dtype=np.float64) - (
        r_real + 1j * r_imag
    ) @ (x_real + 1j * x_imag)
    np.testing.assert_allclose(s_dest[0] + 1j * s_dest[1], expected_s)

    s_column = [expected_s.real, expected_s.imag]
    x_dest = np.zeros((2, 2, 2))
    _llk_rt2_kt1_accumulate(
        x_pages,
        s_column,
        in0_start=0,
        in1_start=0,
        destination=x_dest,
        destination_start=0,
    )
    _llk_rt2_kt1_accumulate(
        x_pages,
        s_column,
        in0_start=2,
        in1_start=1,
        destination=x_dest,
        destination_start=0,
    )
    expected_x = (x_real + 1j * x_imag) @ expected_s
    np.testing.assert_allclose(x_dest[0] + 1j * x_dest[1], expected_x)


def test_two_tile_dest_preflight_matches_full_and_half_sync_limits():
    ttnn = _ttnn()
    for matrix_block in (1, 2, 4):
        newton_schulz_kernel._validate_matrix_block(
            matrix_block,
            fp32_dest_acc_en=True,
            dst_full_sync_en=True,
            variant="bf16-fp32state",
        )
    newton_schulz_kernel._validate_matrix_block(
        2,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
        variant="bf16-fp32state",
    )
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_matrix_block(
            4,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
            variant="bf16-fp32state",
        )

    full = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        two_tile_complex=True,
        output_memory="dram",
        input_memory="dram",
        matrix_block=4,
        variant="bf16-fp32state",
        dst_full_sync_en=True,
    )
    half = newton_schulz_kernel._validate_l1_preflight(
        ttnn,
        batch=8192,
        core_count=110,
        state_dtype="fp32",
        two_tile_complex=True,
        output_memory="dram",
        input_memory="dram",
        matrix_block=2,
        variant="bf16-fp32state",
        dst_full_sync_en=False,
    )
    assert full > 0
    assert half > 0
    with pytest.raises(ValueError, match="DEST slots"):
        newton_schulz_kernel._validate_l1_preflight(
            ttnn,
            batch=8192,
            core_count=110,
            state_dtype="fp32",
            two_tile_complex=True,
            output_memory="dram",
            input_memory="dram",
            matrix_block=4,
            variant="bf16-fp32state",
            dst_full_sync_en=False,
        )


def test_two_tile_source_pairs_init_and_execute_dimensions_and_two_output_pack():
    source = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    split_source = (KERNEL_DIR / "newton_schulz_fidelity_split_compute.cpp").read_text()
    reader = (KERNEL_DIR / "newton_schulz_reader_two_tile.cpp").read_text()

    assert source.count("matmul_block_init(cb_two_tile_r, cb_two_tile_x, false, 1, 2, 1);") == 1
    assert source.count("matmul_block_init(cb_two_tile_r, cb_two_tile_s, false, 1, 2, 1);") == 1
    assert source.count("matmul_block_init(cb_two_tile_x, cb_two_tile_s, false, 1, 2, 1);") == 1
    two_tile_start = source.index("void build_two_tile_x_block")
    two_tile_end = source.index("// The block path is selected", two_tile_start)
    two_tile_source = source[two_tile_start:two_tile_end]
    assert "copy_tile(x_imag, index, 1);" in two_tile_source
    assert "copy_tile(negative_x_imag, index, 2);" in two_tile_source
    assert "void build_two_tile_x_column" in two_tile_source
    assert two_tile_source.count("matmul_block(\n            cb_two_tile_r,") == 2
    assert two_tile_source.count("matmul_block(\n            cb_two_tile_x,") == 2
    assert "matmul_block_init(cb_two_tile_r, cb_two_tile_s, false, 1, 2, 1);" in two_tile_source
    assert "4 * index,\n            2 * index,\n            2 * index,\n            false,\n            1,\n            2,\n            1);" in two_tile_source
    assert "cb_wait_front(cb_zero, 1);" in two_tile_source
    assert "copy_tile(cb_zero, 0, 2 * index + 1);" in two_tile_source
    assert "pack_reconfig_data_format(cb_s_imag, cb_s_real);" in two_tile_source
    assert "pack_reconfig_data_format(cb_s_real, cb_s_imag);" in two_tile_source
    assert "pack_reconfig_data_format(output_real, output_imag);" in two_tile_source
    assert "pack_tile<true>" in two_tile_source
    assert "constexpr bool two_tile_complex = get_compile_time_arg_val(5) != 0;" in source
    assert "get_compile_time_arg_val(two_tile_complex ? 6 : 5)" in split_source
    assert "cb_two_tile_r" in reader


def test_cli_and_dispatch_metadata_keep_two_tile_and_sync_defaults_explicit():
    parser = run_matmul._build_parser()
    defaults = parser.parse_args([])
    assert defaults.two_tile_complex is False
    assert defaults.complex_product_batch == 8192
    assert defaults.complex_product_correctness_only is False
    assert defaults.fp32_dest_acc_en is True
    assert defaults.dst_full_sync_en is True
    enabled = parser.parse_args(
        ["--two-tile-complex", "--no-dst-full-sync-en", "--matrix-block", "2"]
    )
    assert enabled.two_tile_complex is True
    assert enabled.dst_full_sync_en is False
    assert enabled.matrix_block == 2

    probe = parser.parse_args(
        [
            "--complex-product-catalogue",
            "--complex-product-batch",
            "4",
            "--complex-product-correctness-only",
        ]
    )
    assert probe.complex_product_batch == 4
    assert probe.complex_product_correctness_only is True
