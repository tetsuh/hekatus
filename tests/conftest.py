"""Shared fixtures: one simulated frame and one golden image per session.

The RF golden costs about ten seconds per frame; the IQ-path tests all
compare against the same frame, so it is computed once.
"""

import numpy as np
import pytest

from enodia.demo import DEFAULT_SCATTERERS
from enodia.spec.beamform import das_rf_golden
from enodia.spec.probe import linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import simulate_frame


@pytest.fixture(scope="session")
def frame():
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events)
    records = simulate_frame(profile, config, DEFAULT_SCATTERERS)
    return profile, events, records, list(DEFAULT_SCATTERERS)


@pytest.fixture(scope="session")
def golden(frame):
    profile, events, records, _ = frame
    return das_rf_golden(profile, events, records, dtype=np.float32)
