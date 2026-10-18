"""Shared heavy and cheap fixtures for the reference-path tests.

The RF golden costs about ten seconds per full frame; the IQ-path tests all
compare against the same frame, so it is computed once. Contract tests use a
four-event view of the same canonical profile instead, keeping their signals
nonzero without paying for the full frame.
"""

from dataclasses import replace

import numpy as np
import pytest

from enodia.demo import DEFAULT_SCATTERERS
from enodia.spec.beamform import das_rf_golden
from enodia.spec.probe import linear_5mhz
from enodia.spec.records import RFEventRecord
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import PointScatterer, simulate_frame

pytest_plugins = ("pytester",)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    invalid_nodes = {
        node.nodeid
        for item in items
        for node in item.listchain()
        if isinstance(node, (pytest.Module, pytest.Class))
        and any(marker.name == "slow" for marker in node.own_markers)
    }
    if invalid_nodes:
        raise pytest.UsageError(
            "slow marker is only allowed on individual tests; found on: "
            + ", ".join(sorted(invalid_nodes))
        )


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


@pytest.fixture(scope="session")
def cheap_profile():
    """The canonical profile for cheap contract fixtures; cost is reduced by event count."""
    return linear_5mhz()


@pytest.fixture(scope="session")
def cheap_config(cheap_profile):
    """A four-event view of a canonical accepted config for nonzero ingress checks."""
    accepted = make_bmode_config(cheap_profile)
    return replace(accepted, events=accepted.events[:4])


@pytest.fixture(scope="session")
def cheap_scatterers(cheap_config):
    """Two point scatterers on representative cheap-fixture lines for signal-bearing checks."""
    return [
        PointScatterer(cheap_config.events[0].line_x_m, 15e-3),
        PointScatterer(cheap_config.events[2].line_x_m, 25e-3),
    ]


@pytest.fixture(scope="session")
def cheap_frame(cheap_profile, cheap_config, cheap_scatterers):
    """A nonzero four-event frame for contract, routing, shape, and type checks."""
    records = simulate_frame(cheap_profile, cheap_config, cheap_scatterers)
    return cheap_profile, list(cheap_config.events), records, list(cheap_scatterers)


@pytest.fixture(scope="session")
def cheap_silent_frame(cheap_frame):
    """A cheap frame with one genuinely silent event for NaN-report boundary checks."""
    profile, events, records, scatterers = cheap_frame
    silent = RFEventRecord(header=records[0].header, data=np.zeros_like(records[0].data))
    return profile, events, [silent, *records[1:]], scatterers


@pytest.fixture(scope="session")
def cheap_silent_golden(cheap_silent_frame):
    """The golden image paired with the cheap frame whose checkpoint event is silent."""
    profile, events, records, _ = cheap_silent_frame
    return das_rf_golden(profile, events, records, dtype=np.float32)


@pytest.fixture(scope="module")
def cheap_sweep(cheap_frame):
    """A real lightweight sweep/report integration fixture, never prefilled with expected output."""
    from enodia.spec.beamform import decimation_sweep

    profile, events, records, scatterers = cheap_frame
    return decimation_sweep.sweep(profile, events, records, scatterers)
