"""Board-free L=64 capacity arithmetic for Issue #63 Scope 5."""

from tools.newton_schulz_l64_capacity import capacity_rows

EXPECTED = {
    (1, "all-L1"): (75, 300, 309_248, 4_306_944, 4_727_552),
    (1, "R-L1/X0-DRAM"): (75, 300, 309_248, 1_849_344, 2_269_952),
    (1, "R-DRAM/X0-L1"): (75, 300, 309_248, 2_463_744, 2_884_352),
    (1, "all-DRAM"): (75, 300, 309_248, 0, 420_608),
    (4, "all-L1"): (76, 304, 702_464, 4_364_288, 5_178_112),
    (4, "R-L1/X0-DRAM"): (76, 304, 702_464, 1_873_920, 2_687_744),
    (4, "R-DRAM/X0-L1"): (76, 304, 702_464, 2_496_512, 3_310_336),
    (4, "all-DRAM"): (76, 304, 702_464, 0, 813_824),
}


def test_l64_capacity_uses_aligned_logical_assignment_and_four_tiles():
    rows = {(row.matrix_block, row.placement): row for row in capacity_rows()}

    assert set(rows) == set(EXPECTED)
    for key, expected in EXPECTED.items():
        row = rows[key]
        max_logical, max_physical, cb_bytes, tensor_bytes, total_bytes = expected
        assert row.max_logical_matrices_per_core == max_logical
        assert row.max_physical_tiles_per_core == max_physical
        assert row.cb_bytes == cb_bytes
        assert row.tensor_bytes == tensor_bytes
        assert row.total_bytes == total_bytes
        assert row.budget_bytes == 1_572_864
        assert row.headroom_bytes == row.budget_bytes - row.total_bytes


def test_l64_all_dram_is_the_only_listed_fit_boundary():
    rows = capacity_rows()

    for row in rows:
        if row.placement == "all-DRAM":
            assert row.fits
        else:
            assert not row.fits
