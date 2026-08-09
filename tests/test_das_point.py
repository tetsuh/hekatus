"""Acceptance test for MVP-1: point scatterers image at their true positions."""

import numpy as np

from enodia.demo import DEFAULT_SCATTERERS, run_pipeline
from enodia.spec.probe import linear_5mhz


def test_point_scatterers_image_at_their_true_positions():
    profile = linear_5mhz()
    db, z, line_x, _ = run_pipeline(profile, DEFAULT_SCATTERERS)

    for s in DEFAULT_SCATTERERS:
        near_z = np.abs(z - s.z_m) < 2e-3
        near_x = np.abs(line_x - s.x_m) < 2e-3
        window = db[np.ix_(near_z, near_x)]
        iz, ix = np.unravel_index(np.argmax(window), window.shape)

        assert abs(z[near_z][iz] - s.z_m) < 0.5e-3, f"axial peak off at {s}"
        assert abs(line_x[near_x][ix] - s.x_m) < 1.0e-3, f"lateral peak off at {s}"
        # Within 6 dB of the frame maximum, so the peak found in the window
        # is the scatterer rather than a sidelobe or a noise ridge.
        assert window.max() > -6.0
