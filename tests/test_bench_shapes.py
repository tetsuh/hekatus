"""The shape catalogue is the argument; the runner only executes it.

These run without an accelerator, which is the point: the FLOP accounting
that turns a duration into an efficiency figure has to be reviewable on any
machine, not only on the one holding the card.
"""

import pytest

from enodia.tt.bench.shapes import MatmulShape, default_catalogue, total_flops


def test_flops_count_every_real_matmul_the_logical_operation_costs():
    real = MatmulShape(
        name="r", batch=2, m=4, k=8, n=16, real_matmuls=1, family="reference", note=""
    )
    complex_ = MatmulShape(
        name="c", batch=2, m=4, k=8, n=16, real_matmuls=4, family="reference", note=""
    )

    assert total_flops(real) == 2 * 2 * 4 * 8 * 16
    assert total_flops(complex_) == 4 * total_flops(real)


def test_catalogue_names_are_unique_and_dimensions_are_positive():
    catalogue = default_catalogue()

    assert len(catalogue) > 1
    assert len({s.name for s in catalogue}) == len(catalogue)
    for s in catalogue:
        assert min(s.batch, s.m, s.k, s.n, s.real_matmuls) >= 1, s.name


def test_catalogue_pins_the_exact_workload_inventory():
    catalogue = default_catalogue()

    actual = {
        (s.name, s.batch, s.m, s.k, s.n, s.real_matmuls, s.family) for s in catalogue
    }
    expected = {
        *{
            (f"newton_schulz_L{size}_b{batch}", batch, size, size, size, 4, "newton_schulz")
            for size in (16, 32, 64)
            for batch in (1024, 8192, 65536)
        },
        *{
            (f"beamspace_B16_ch{channels}_p{pixels}", 1, 16, channels, pixels, 4, "beamspace")
            for channels in (128, 256)
            for pixels in (4096, 65536)
        },
        *{
            (f"frontend_fir_taps64_w{width}", 1, 262144, 64, width, 2, "frontend_fir")
            for width in (2, 8, 32)
        },
        ("reference_square_4096", 1, 4096, 4096, 4096, 1, "reference"),
    }

    assert actual == expected


def test_catalogue_pins_representative_flop_accounting():
    by_name = {s.name: s for s in default_catalogue()}

    assert total_flops(by_name["newton_schulz_L16_b1024"]) == 1024 * 4 * 2 * 16**3
    assert total_flops(by_name["beamspace_B16_ch256_p4096"]) == 4 * 2 * 16 * 256 * 4096
    assert total_flops(by_name["frontend_fir_taps64_w2"]) == 2 * 2 * 262144 * 64 * 2
    assert total_flops(by_name["reference_square_4096"]) == 2 * 4096**3


def test_catalogue_covers_every_representative_family():
    families = {s.family for s in default_catalogue()}

    assert {"newton_schulz", "beamspace", "frontend_fir"} <= families


def test_the_large_square_shape_is_marked_as_a_reference_not_a_result():
    """A flattering shape included for contrast must not be quotable as the
    workload's efficiency, so it is labelled in the data itself."""
    reference = [s for s in default_catalogue() if s.family == "reference"]

    assert reference, "the contrast shape is missing"
    for s in reference:
        assert not s.representative


def test_representative_and_reference_are_mutually_exclusive():
    for s in default_catalogue():
        assert s.representative == (s.family != "reference"), s.name


def test_a_shape_rejects_a_batch_that_would_not_run():
    with pytest.raises(ValueError):
        MatmulShape(
            name="bad", batch=0, m=4, k=4, n=4, real_matmuls=1, family="reference", note=""
        )
