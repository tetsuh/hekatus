"""Board-free contracts for the minimal two-tile a/b/c probe."""

from pathlib import Path

import numpy as np

from enodia.tt.bench.two_tile_probe import (
    ALL_PROBE_STAGES,
    DECOMPOSITION_STAGES,
    PROBE_STAGES,
    expected_probe_outputs,
    expected_probe_partials,
    probe_input_pages,
    probe_stage_contract,
)

KERNEL_DIR = Path(__file__).parents[1] / "enodia/tt/bench/kernels"


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    r_real = np.array(
        [[2.0, 3.0], [5.0, 7.0]], dtype=np.float32
    )
    r_imag = np.array(
        [[11.0, 13.0], [17.0, 19.0]], dtype=np.float32
    )
    x_real = np.array(
        [[23.0, 29.0], [31.0, 37.0]], dtype=np.float32
    )
    x_imag = np.array(
        [[41.0, 43.0], [47.0, 53.0]], dtype=np.float32
    )
    return r_real + 1j * r_imag, x_real + 1j * x_imag


def test_probe_has_ordered_minimal_stages_and_expected_products():
    r, x = _matrices()
    pages = probe_input_pages(r[None], x[None])

    assert PROBE_STAGES == ("a", "b", "c")
    assert pages["in0_r"].shape == (1, 4, 2, 2)
    assert pages["in1_x_column"].shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(pages["in0_r"][0, :, 0, 0], [-2.0, -11.0, 11.0, -2.0])
    np.testing.assert_array_equal(pages["in1_x_column"][0, :, 0, 0], [23.0, 41.0])

    expected = expected_probe_outputs(r[None], x[None])
    np.testing.assert_allclose(expected["a"][0], -(r @ x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(expected["b"][0], 2.0 * np.eye(2) - r @ x, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(expected["c"][0], x @ expected["b"][0], rtol=1e-6, atol=1e-6)


def test_probe_decomposition_covers_partial_products_and_offsets():
    r, x = _matrices()
    expected = expected_probe_partials(r[None], x[None])
    np.testing.assert_allclose(expected["a3"], expected["a1"] + expected["a2"])
    assert DECOMPOSITION_STAGES == ("a1", "a2", "a3")
    assert set(DECOMPOSITION_STAGES).issubset(ALL_PROBE_STAGES)
    assert expected["a1"].shape == expected["a2"].shape == (1, 2, 2)
    assert probe_stage_contract("a1")["matmul_calls"] == [
        {"in0_offset": 0, "in1_offset": 0, "dest_real": 0, "dest_imag": 1}
    ]
    assert probe_stage_contract("a2")["matmul_calls"] == [
        {"in0_offset": 2, "in1_offset": 1, "dest_real": 0, "dest_imag": 1}
    ]
    assert len(probe_stage_contract("a3")["matmul_calls"]) == 2


def test_probe_contract_records_cb_order_dest_slots_and_sync_sequence():
    for stage in ALL_PROBE_STAGES:
        contract = probe_stage_contract(stage)
        assert contract["matmul_dimensions"] == {"rt": 2, "ct": 1, "kt": 1}
        assert contract["in0_register"] == "SrcB"
        assert contract["in1_register"] == "SrcA"
        assert contract["dest_slots"] == [0, 1]
        assert contract["tile_regs_sequence"] == ["acquire", "commit", "wait", "release"]
        assert contract["cb_counts"]["wait"] > 0
        assert contract["cb_counts"]["reserve"] > 0
        assert contract["cb_counts"]["push"] > 0
        assert contract["cb_counts"]["pop"] > 0


def test_probe_compute_source_keeps_two_k_terms_and_stage_boundaries():
    source = (KERNEL_DIR / "two_tile_probe_compute.cpp").read_text()
    assert "matmul_block_init(cb_two_tile_r, cb_two_tile_s, false, 1, 2, 1);" in source
    assert "matmul_block_init(cb_two_tile_x, cb_two_tile_s, false, 1, 2, 1);" in source
    assert "copy_tile(cb_x_real, 0, 0);" in source
    assert "copy_tile(cb_x_imag, 0, 1);" in source
    assert "copy_tile(cb_negative_x_imag, 0, 2);" in source
    assert "copy_tile(cb_x_real, 0, 3);" in source
    assert "tile_regs_acquire();" in source
    assert "tile_regs_commit();" in source
    assert "tile_regs_wait();" in source
    assert "tile_regs_release();" in source
    assert "run_k0" in source
    assert "run_k1" in source
    assert "r_times_x(false, false, true, false)" in source
    assert "r_times_x(false, false, false, true)" in source
    assert "r_times_x(false, false, true, true)" in source
    for offset in ("0, 0, 0", "2, 1, 0"):
        assert offset in source


def test_probe_reader_and_writer_expose_explicit_cb_pack_path():
    reader = (KERNEL_DIR / "two_tile_probe_reader.cpp").read_text()
    writer = (KERNEL_DIR / "two_tile_probe_writer.cpp").read_text()
    assert "cb_two_tile_r" in reader
    assert "cb_x_real" in reader
    assert "cb_x_imag" in reader
    assert "cb_push_back(cb_two_tile_r" in reader
    assert "cb_wait_front(cb_output_real" in writer
    assert "cb_wait_front(cb_output_imag" in writer
    assert "noc_async_write_page" in writer
