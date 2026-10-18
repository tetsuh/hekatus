"""Shared fixtures: one simulated frame and one golden image per session.

The RF golden costs about ten seconds per frame; the IQ-path tests all
compare against the same frame, so it is computed once.
"""

import numpy as np
import pytest
from hardware_isolation import (
    deny_access,
    install_audit_hook,
    permit_hardware,
    permit_wrapper,
    restore_access,
)

from enodia.demo import DEFAULT_SCATTERERS
from enodia.spec.beamform import das_rf_golden
from enodia.spec.probe import linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import simulate_frame


def pytest_addoption(parser):
    parser.addoption(
        "--allow-hardware-tests",
        action="store_true",
        default=False,
        help="explicitly permit tests that access accelerator hardware",
    )


def pytest_configure(config):
    install_audit_hook()


@pytest.fixture(autouse=True)
def _hardware_isolation():
    token = deny_access()
    try:
        yield
    finally:
        restore_access(token)


@pytest.fixture
def hardware_access(request, _hardware_isolation):
    if not request.config.getoption("--allow-hardware-tests"):
        pytest.skip("hardware access is disabled; pass --allow-hardware-tests explicitly")
    token = permit_hardware()
    try:
        yield
    finally:
        restore_access(token)


@pytest.fixture
def wrapper_execution(tmp_path_factory, _hardware_isolation):
    token = permit_wrapper(tmp_path_factory.getbasetemp())
    try:
        yield
    finally:
        restore_access(token)


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
