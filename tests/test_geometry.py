import numpy as np
import pytest

from enodia.spec.beamform.interp_sweep import no_interpolation_worst_phase_deg
from enodia.spec.probe import linear_5mhz


def test_element_positions_are_symmetric_and_float64():
    """Geometry stays float64 (design.md §14) and the aperture is centred."""
    p = linear_5mhz()
    x = p.element_x()
    assert x.dtype == np.float64
    assert np.isclose(x[0], -x[-1])
    assert np.allclose(np.diff(x), p.pitch_m)


def test_linear_5mhz_bandwidth_contract_is_explicit():
    """design.md §4 / #46: `bandwidth_frac` is the full fractional width of the
    effective two-way pulse between its half-amplitude (−6.0206 dB) points,
    the one-sided analysis edge is half of it, and a value with no stated
    provenance is provisional rather than silently physical. `linear-5mhz`
    keeps 0.7 and names no source."""
    p = linear_5mhz()

    assert p.bandwidth_frac == 0.7
    assert p.bandwidth_source is None
    assert p.bandwidth_hz == pytest.approx(3.5e6)
    assert p.bandwidth_edge_hz == pytest.approx(1.75e6)
    # Phase rotation alone at the edge, worst fraction, at the two decimation
    # ratios §5 names for this probe.
    for decimation, degrees in ((8, 63.0), (4, 31.5)):
        edge = p.bandwidth_edge_hz / (p.fs_hz / decimation)
        assert no_interpolation_worst_phase_deg(edge) == pytest.approx(degrees, abs=1e-9)
