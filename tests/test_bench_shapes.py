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
