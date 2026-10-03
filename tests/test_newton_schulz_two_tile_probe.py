"""Board-free contracts for the minimal two-tile a/b/c probe."""

from pathlib import Path

import numpy as np

from enodia.tt.bench.two_tile_probe import (
    ALL_PROBE_STAGES,
    DECOMPOSITION_STAGES,
    DEST_PROBE_BATCH,
    DEST_PROBE_STAGE,
    PROBE_STAGES,
    classify_dest_probe_outputs,
    dest_probe_contract,
    dest_probe_input_pages,
    dest_probe_source_audit,
    expected_dest_probe_outputs,
    expected_probe_outputs,
    expected_probe_partials,
    known_dest_probe_complex_x,
    known_dest_probe_xr,
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


def test_dest_probe_uses_known_real_xr_and_distinct_stride_sentinels():
    complex_x = known_dest_probe_complex_x()
    xr = known_dest_probe_xr()
    pages = dest_probe_input_pages(xr)

    assert complex_x.shape == (DEST_PROBE_BATCH, 32, 32)
    assert np.iscomplexobj(complex_x)
    assert np.any(complex_x.imag != 0)
    assert not np.array_equal(complex_x, complex_x.transpose(0, 2, 1))
    assert pages["in0"].shape == (DEST_PROBE_BATCH, 4, 32, 32)
    assert pages["in1"].shape == (DEST_PROBE_BATCH, 1, 32, 32)
    np.testing.assert_array_equal(pages["in0"][:, 0], pages["identity"])
    np.testing.assert_array_equal(pages["in0"][:, 1], pages["twice_identity"])
    np.testing.assert_array_equal(pages["in1"][:, 0], xr)
    assert not np.array_equal(pages["in0"][:, 2], pages["in0"][:, 3])

    expected = expected_dest_probe_outputs(xr)
    np.testing.assert_array_equal(expected["slot0"], xr)
    np.testing.assert_array_equal(expected["slot1"], 2.0 * xr)


def test_dest_probe_contract_pins_one_call_dest_slots_and_cb_lifecycle():
    contract = dest_probe_contract()

    assert DEST_PROBE_STAGE == "dest"
    assert contract["matmul_dimensions"] == {"rt": 2, "ct": 1, "kt": 1}
    assert contract["matmul_block_init"]["count"] == 1
    assert contract["matmul_calls"][0]["count"] == 1
    assert contract["matmul_calls"][0]["in0_offset"] == 0
    assert contract["matmul_calls"][0]["in1_offset"] == 0
    assert contract["in0_register"] == "SrcB"
    assert contract["in1_register"] == "SrcA"
    assert contract["in0_pages_consumed"] == [0, 1]
    assert contract["in0_pages_sentinels"] == [2, 3]
    assert contract["expected_dest_tiles"] == {"0": "Xr", "1": "2Xr"}
    assert contract["pack_operations"] == [
        {"dst_index": 0, "output_cb": 15, "pack_api": "pack_tile", "count": 1},
        {"dst_index": 1, "output_cb": 16, "pack_api": "pack_tile", "count": 1},
    ]
    assert contract["tile_regs_sequence"] == [
        "acquire",
        "matmul",
        "commit",
        "wait",
        "pack",
        "release",
    ]
    assert contract["cb_lifecycle_per_matrix"] == {
        "wait_front": 2,
        "reserve_back": 4,
        "push_back": 4,
        "pop_front": 2,
    }


def test_dest_probe_classification_is_machine_readable_for_all_matrices_and_slots():
    xr = known_dest_probe_xr()
    expected = expected_dest_probe_outputs(xr)
    rows = classify_dest_probe_outputs(expected["slot0"], expected["slot1"], xr)

    assert len(rows) == DEST_PROBE_BATCH * 2
    assert {(row["matrix"], row["dest_slot"]) for row in rows} == {
        (matrix, slot) for matrix in range(DEST_PROBE_BATCH) for slot in (0, 1)
    }
    assert all(row["pass"] for row in rows)
    assert all(
        row["classification"] == row["expected_label"]
        for row in rows
    )

    pages = dest_probe_input_pages(xr)
    wrong = classify_dest_probe_outputs(
        pages["sentinel0"] @ xr,
        pages["sentinel1"] @ xr,
        xr,
    )
    assert all(row["classification"] == "other_tile" for row in wrong)
    assert not any(row["pass"] for row in wrong)


def test_dest_probe_compute_has_one_matmul_and_two_explicit_pack_paths():
    source = (KERNEL_DIR / "two_tile_dest_probe_compute.cpp").read_text()
    assert source.count("matmul_block_init(") == 1
    assert source.count("matmul_block(") == 1
    assert "matmul_block_init(cb_in0, cb_in1, false, 1, 2, 1);" in source
    assert "matmul_block(cb_in0, cb_in1, 0, 0, 0, false, 1, 2, 1);" in source
    assert source.count("pack_tile(") == 2
    assert "pack_tile(0, cb_output_slot0);" in source
    assert "pack_tile(1, cb_output_slot1);" in source
    assert "pack_tile_block" not in source
    assert "DEST_PROBE_DPRINT(0, cb_output_slot0);" in source
    assert "DEST_PROBE_DPRINT(1, cb_output_slot1);" in source
    order = [
        source.index("tile_regs_acquire();"),
        source.index("matmul_block(cb_in0"),
        source.index("tile_regs_commit();"),
        source.index("tile_regs_wait();"),
        source.index("pack_tile(0"),
        source.index("pack_tile(1"),
        source.index("tile_regs_release();"),
    ]
    assert order == sorted(order)


def test_dest_probe_reader_and_writer_pin_page_offsets_and_cb_lifecycle():
    reader = (KERNEL_DIR / "two_tile_dest_probe_reader.cpp").read_text()
    writer = (KERNEL_DIR / "two_tile_dest_probe_writer.cpp").read_text()
    assert "cb_reserve_back(cb_in0, in0_pages_per_matrix);" in reader
    assert "tile * in0_pages_per_matrix + page" in reader
    assert "get_write_ptr(cb_in0) + page * get_tile_size(cb_in0)" in reader
    assert "cb_push_back(cb_in0, in0_pages_per_matrix);" in reader
    assert "read_page(cb_in1, tile, in1);" in reader
    assert "cb_wait_front(cb_output_slot0, 1);" in writer
    assert "cb_wait_front(cb_output_slot1, 1);" in writer
    assert "noc_async_write_page(tile, slot0" in writer
    assert "noc_async_write_page(tile, slot1" in writer


def test_dest_probe_source_audit_has_pinned_line_references_and_mapping():
    audit = dest_probe_source_audit()

    assert audit["revision"] == "901dd9ce93816ffd1fd185b801fc727065e9ae07"
    assert {source["path"] for source in audit["sources"]} == {
        "tt_metal/hw/inc/api/compute/matmul.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_common_api.h",
    }
    assert all(source["lines"] for source in audit["sources"])
    assert audit["application"] == {
        "in0_pages_consumed": [0, 1],
        "in0_pages_sentinels": [2, 3],
        "in0_row_stride_pages": 1,
        "in0_k_stride_pages": 1,
        "in1_page_offset": 0,
        "dest_slots": [0, 1],
        "expected": ["Xr", "2Xr"],
    }
