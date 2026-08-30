"""A transmit configuration names the one probe profile it runs on (#46).

design.md §19: setup maps each configuration to exactly one
`probe_profile_id`; the runtime config ID a record carries selects the
profile transitively; the bandwidth lives in the profile and is duplicated
into neither the transmit description nor the frame header
(docs/dataplane.md). Pinned here so the contract documents and the code
cannot part.
"""

from dataclasses import fields, replace

import pytest

from enodia.spec.probe import linear_5mhz
from enodia.spec.records import EventHeader
from enodia.spec.sequence import TransmitConfig, TxEvent, make_bmode_config, make_bmode_sequence
from enodia.spec.sim import PointScatterer, simulate_frame


def test_each_configuration_references_exactly_one_probe_profile_by_id():
    p = linear_5mhz()
    config = make_bmode_config(p)

    assert isinstance(config, TransmitConfig)
    assert config.probe_profile_id == p.name == "linear-5mhz"
    assert config.config_id == "bmode-focused"
    assert list(config.events) == make_bmode_sequence(p)


def test_the_runtime_config_id_selects_the_profile_transitively_and_carries_no_bandwidth():
    """A record carries the config ID and nothing about the pulse; the
    profile — and so the bandwidth — is reached through the configuration."""
    p = linear_5mhz()
    config = make_bmode_config(p)
    scatterers = [PointScatterer(0.0, 20e-3)]
    records = simulate_frame(
        p,
        replace(config, events=config.events[:2]),
        scatterers,
    )
    assert all(r.header.config_id == config.config_id for r in records)
    with pytest.raises(TypeError):
        simulate_frame(
            p,
            config,
            scatterers,
            config_id="independent-config",
        )

    setup = {config.config_id: config}
    assert setup[records[0].header.config_id].probe_profile_id == p.name


def test_no_bandwidth_field_exists_outside_the_profile():
    """The narrow additive mapping is all #46 adds: the frame identity keeps
    its shape, and neither the configuration nor an event carries bandwidth."""
    header_fields = {f.name for f in fields(EventHeader)}
    assert header_fields == {
        "seq",
        "config_id",
        "param_generation",
        "tx_event_index",
        "tx_type",
        "timestamp_ns",
    }
    assert {f.name for f in fields(TransmitConfig)} == {
        "config_id",
        "probe_profile_id",
        "element_x_m",
        "events",
    }
    for cls in (TransmitConfig, TxEvent, EventHeader):
        assert not any("bandwidth" in f.name for f in fields(cls)), cls.__name__
