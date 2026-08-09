import numpy as np

from enodia.spec.probe import linear_5mhz


def test_element_positions_are_symmetric_and_float64():
    """Geometry stays float64 (design.md §14) and the aperture is centred."""
    p = linear_5mhz()
    x = p.element_x()
    assert x.dtype == np.float64
    assert np.isclose(x[0], -x[-1])
    assert np.allclose(np.diff(x), p.pitch_m)
