"""Board-free fidelity mapping audit for the probe and production path."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from enodia.tt.bench import newton_schulz_kernel
from enodia.tt.bench.two_tile_probe import (
    probe_input_pages,
    two_tile_fidelity_audit,
)

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"


def _ttnn():
    return SimpleNamespace(bfloat16="bf16", float32="fp32")


def test_fidelity_audit_contract_covers_the_six_requested_mappings():
    audit = two_tile_fidelity_audit()

    assert audit["contract"] == "issue-63-two-tile-fidelity-split-audit"
    assert audit["scope"] == {
        "batch": 4,
        "size": 32,
        "iterations": 1,
        "variant": "bf16-fp32state",
        "input_memory": "dram",
        "output_memory": "dram",
        "board_execution": "not run",
    }
    assert set(audit["source_files"]) == {"probe", "production"}
    assert {
        "input_tensors",
        "cb_definitions",
        "reader",
        "compute",
        "writer",
        "host_readback",
        "findings",
    }.issubset(audit)

    assert audit["input_tensors"]["probe"]["runtime_tensor_order"] == [
        "R",
        "Xr",
        "Xi",
        "-Xi",
        "I",
        "0",
    ]
    assert audit["input_tensors"]["production"]["runtime_tensor_order"] == [
        "R",
        "X0r",
        "X0i",
        "I",
        "0",
    ]
    assert audit["reader"]["production"]["runtime_args"] == [
        "R",
        "X0r",
        "X0i",
        "I",
        "0",
        "start_tile",
        "tile_count",
    ]
    assert audit["writer"]["mapping"].startswith("one FP32 page")
    assert audit["host_readback"]["comparison"].startswith("identical")


def test_probe_and_production_host_pages_have_the_same_physical_r_mapping():
    matrices = np.zeros((2, 32, 32), dtype=np.complex64)
    matrices.real[:, 0, 0] = (3.25, -4.5)
    matrices.imag[:, 0, 0] = (5.75, 6.5)
    x0 = np.ones_like(matrices)
    probe = probe_input_pages(matrices, x0)["in0_r"]
    production = newton_schulz_kernel._reader_input_values(
        matrices,
        x0,
        fuse_s=False,
        tile_count=2,
        two_tile_complex=True,
    )[0]

    # The probe pre-rounds its R host pages; production supplies FP32 pages to
    # a BF16 tensor. Their device-visible words must therefore be identical.
    probe_words = probe.view(np.uint32) >> 16
    production_words = production.astype(np.float32).view(np.uint32) >> 16
    np.testing.assert_array_equal(probe_words, production_words)
    np.testing.assert_array_equal(production[:, 0, 0, 0], [-3.25, 4.5])
    np.testing.assert_array_equal(production[:, 1, 0, 0], [-5.75, -6.5])
    np.testing.assert_array_equal(production[:, 2, 0, 0], [5.75, 6.5])
    np.testing.assert_array_equal(production[:, 4, 0, 0], 2.0)


def test_two_tile_descriptor_capacity_is_exact_without_changing_default_ledger():
    ttnn = _ttnn()
    for matrix_block in (1, 2, 4, 8):
        definitions = newton_schulz_kernel._cb_definitions(
            ttnn,
            "fp32",
            two_tile_complex=True,
            matrix_block=matrix_block,
        )
        assert definitions[newton_schulz_kernel.CB_TWO_TILE_R] == (
            "bf16",
            6 * matrix_block,
        )
        assert definitions[newton_schulz_kernel.CB_TWO_TILE_X] == (
            "fp32",
            4 * matrix_block,
        )
        assert definitions[newton_schulz_kernel.CB_TWO_TILE_S] == (
            "fp32",
            2 * matrix_block,
        )
        for index in (
            newton_schulz_kernel.CB_X0_REAL,
            newton_schulz_kernel.CB_X0_IMAG,
            newton_schulz_kernel.CB_STATE_REAL,
            newton_schulz_kernel.CB_STATE_IMAG,
            newton_schulz_kernel.CB_NEG_X_IMAG,
            newton_schulz_kernel.CB_OUTPUT_REAL,
            newton_schulz_kernel.CB_OUTPUT_IMAG,
        ):
            assert definitions[index][1] == matrix_block

    # The default one-tile/fidelity ledger remains unchanged.
    baseline = newton_schulz_kernel._cb_definitions(ttnn, "fp32")
    assert baseline[newton_schulz_kernel.CB_OUTPUT_REAL] == ("fp32", 2)
    assert baseline[newton_schulz_kernel.CB_OUTPUT_IMAG] == ("fp32", 2)


def test_production_two_tile_format_transition_matches_probe_boundaries():
    source = (KERNEL_DIR / "newton_schulz_compute.cpp").read_text()
    reader = (KERNEL_DIR / "newton_schulz_reader_two_tile.cpp").read_text()
    writer = (KERNEL_DIR / "newton_schulz_writer.cpp").read_text()
    audit = two_tile_fidelity_audit()

    negate_start = source.index("void negate_two_tile_state_imag_block(")
    negate_end = source.index("// Build S directly", negate_start)
    negate = source[negate_start:negate_end]
    assert "reconfig_data_format_srca(cb_zero);" in negate
    assert "reconfig_data_format_srcb(x_imag);" in negate
    assert "init_common" not in negate

    x_block_start = source.index("void build_two_tile_x_block(")
    x_block_end = source.index("// Build the state-format", x_block_start)
    x_block = source[x_block_start:x_block_end]
    assert "reconfig_data_format_srca(cb_zero, x_real);" in x_block
    assert "reconfig_data_format_srca(cb_two_tile_s, x_real);" not in x_block

    process_start = source.index("void process_two_tile_matrix_block(")
    process_end = source.index("// The block path is selected", process_start)
    process = source[process_start:process_end]
    assert "negate_two_tile_state_imag_block(x_imag, block_count);" in process
    assert "negate_state_imag_block(x_imag, cb_zero, x_imag, block_count);" not in process

    r_start = source.index("void two_tile_s_matmul_block(")
    x_start = source.index("void two_tile_x_matmul_block(")
    r_source = source[r_start:x_start]
    x_source = source[x_start:process_start]
    assert "reconfig_data_format_srca(cb_two_tile_s);" in r_source
    assert "reconfig_data_format_srcb(cb_two_tile_r);" in r_source
    assert "reconfig_data_format_srca(cb_identity);" in r_source
    assert "reconfig_data_format_srcb(cb_two_tile_r);" in r_source
    assert "reconfig_data_format_srca(cb_two_tile_s);" in x_source
    assert "reconfig_data_format_srcb(cb_two_tile_x);" in x_source
    assert "pack_tile<true>(2 * index, output_real, index);" in x_source
    assert "pack_tile<true>(2 * index + 1, output_imag, index);" in x_source
    assert "init_common" not in r_source + x_source

    assert "const std::uint32_t r_block_address = get_arg_val<std::uint32_t>(0);" in reader
    assert "const std::uint32_t x0_real_address = get_arg_val<std::uint32_t>(1);" in reader
    assert "const std::uint32_t x0_imag_address = get_arg_val<std::uint32_t>(2);" in reader
    assert "const std::uint32_t identity_address = get_arg_val<std::uint32_t>(3);" in reader
    assert "const std::uint32_t zero_address = get_arg_val<std::uint32_t>(4);" in reader
    assert "constexpr auto r_block_args = TensorAccessorArgs<3>();" in reader
    assert "tile * two_tile_r_pages + face" in reader
    assert "tile,\n                x0_real" in reader
    assert "tile,\n                x0_imag" in reader
    assert "get_arg_val<std::uint32_t>(0)" in writer
    assert "get_arg_val<std::uint32_t>(1)" in writer
    assert "_unpack_matrices" in Path(
        __file__
    ).parents[1].joinpath("enodia/tt/bench/newton_schulz_kernel.py").read_text()

    assert audit["findings"][0]["status"] == "fixed"
    assert audit["compute"]["state_output_alias"] is False
