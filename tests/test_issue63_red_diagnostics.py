"""Intentional, board-free RED diagnostics for Issue #63.

These tests model two review failures rather than production behavior. They are
skipped unless HEKATUS_RUN_INTENTIONAL_RED=1 so the normal suite remains green.
The expected failures and command are retained here and in the pull-request evidence.

Captured RED command:
HEKATUS_RUN_INTENTIONAL_RED=1 .venv/bin/python -m pytest -q -m red_diagnostic tests/test_issue63_red_diagnostics.py

Expected result:
FAILED tests/test_issue63_red_diagnostics.py::test_red_wrong_diagonal_tile_placement_fails_l0_equivalence
FAILED tests/test_issue63_red_diagnostics.py::test_red_padded_l16_kernel_flops_diverge_from_logical_shape_flops
2 failed

Normal host validation excludes these diagnostics:
.venv/bin/python -m pytest -q -m 'not red_diagnostic'
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from enodia.tt.bench.newton_schulz_kernel import NEWTON_SCHULZ_ITERATIONS, _physical_tile_count
from enodia.tt.bench.newton_schulz_reference import (
    initial_value,
    inverse_flops,
    newton_schulz_reference,
    random_hpd_batch,
)
from enodia.tt.bench.shapes import MatmulShape, total_flops

RUN_INTENTIONAL_RED = os.environ.get("HEKATUS_RUN_INTENTIONAL_RED") == "1"
RED_SKIP_REASON = "intentional RED diagnostics require HEKATUS_RUN_INTENTIONAL_RED=1"


def _wrong_upper_right_pair_pack(matrices: np.ndarray) -> np.ndarray:
    """Place the second L=16 matrix in the wrong, off-diagonal quadrant."""
    packed = np.zeros((1, 32, 32), dtype=np.complex64)
    packed[0, :16, :16] = matrices[0]
    packed[0, :16, 16:] = matrices[1]
    return packed


@pytest.mark.red_diagnostic
@pytest.mark.skipif(not RUN_INTENTIONAL_RED, reason=RED_SKIP_REASON)
def test_red_wrong_diagonal_tile_placement_fails_l0_equivalence():
    matrices = random_hpd_batch(2, 16, seed=6300)
    expected = newton_schulz_reference(matrices)
    packed_r = _wrong_upper_right_pair_pack(matrices)
    # Use the independent initial-value construction, not accelerator code,
    # in the deliberately wrong variant.
    packed_x = _wrong_upper_right_pair_pack(initial_value(matrices))
    identity = 2.0 * np.eye(32, dtype=np.complex64)
    for _ in range(NEWTON_SCHULZ_ITERATIONS):
        packed_x = packed_x @ (identity - packed_r @ packed_x)

    # The production unpacker expects the second matrix at the lower-right
    # diagonal. This intentionally wrong placement therefore cannot satisfy L0.
    actual = np.stack((packed_x[0, :16, :16], packed_x[0, 16:, 16:]))
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.red_diagnostic
@pytest.mark.skipif(not RUN_INTENTIONAL_RED, reason=RED_SKIP_REASON)
def test_red_padded_l16_kernel_flops_diverge_from_logical_shape_flops():
    shape = MatmulShape(
        name="newton_schulz_L16_b8192",
        batch=8192,
        m=16,
        k=16,
        n=16,
        real_matmuls=4,
        family="newton_schulz",
        note="intentional padded-work accounting diagnostic",
    )
    physical_tiles = _physical_tile_count(shape.batch, shape.m)
    claimed_kernel_flops = (
        physical_tiles
        * shape.real_matmuls
        * 2
        * 32**3
        * NEWTON_SCHULZ_ITERATIONS
    )

    # This intentionally claims every padded 32x32 product as useful work.
    # L=16 accounting must remain the logical shapes.total_flops denominator.
    assert claimed_kernel_flops == NEWTON_SCHULZ_ITERATIONS * total_flops(shape)
    assert claimed_kernel_flops == inverse_flops(shape)
