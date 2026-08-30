"""The stage-boundary contract is enforced, not merely documented.

A reference implementation that is also the L0 oracle cannot afford to
misalign data silently: a wrong-but-plausible image is worse than an error
(design.md §19).
"""

from dataclasses import replace

import numpy as np
import pytest

from enodia.spec.beamform import das_rf_golden, envelope, log_compress
from enodia.spec.probe import linear_5mhz
from enodia.spec.records import EventHeader, RFEventRecord
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import PointScatterer, simulate_frame


def _config_for_events(profile, count):
    config = make_bmode_config(profile)
    return replace(config, events=config.events[:count])


def _header(**kw):
    base = {
        "seq": 0,
        "config_id": "c",
        "param_generation": 0,
        "tx_event_index": 0,
        "tx_type": "bmode_focused",
        "timestamp_ns": 0,
    }
    base.update(kw)
    return EventHeader(**base)


def test_record_rejects_wrong_dtype_and_shape():
    header = _header()
    wrong_dtype = np.zeros((4, 8), dtype=np.float32)
    wrong_rank = np.zeros(8, dtype=np.int16)

    with pytest.raises(ValueError):
        RFEventRecord(header=header, data=wrong_dtype)
    with pytest.raises(ValueError):
        RFEventRecord(header=header, data=wrong_rank)


def test_record_payload_is_read_only():
    rec = RFEventRecord(header=_header(), data=np.zeros((4, 8), dtype=np.int16))
    with pytest.raises(ValueError):
        rec.data[0, 0] = 1


def test_beamform_rejects_a_record_set_that_does_not_match_the_events():
    profile = linear_5mhz()
    config = _config_for_events(profile, 4)
    events = list(config.events)
    records = simulate_frame(profile, config, [PointScatterer(0.0, 20e-3)])

    short = records[:-1]

    with pytest.raises(ValueError):
        das_rf_golden(profile, events, short)


def test_beamform_rejects_records_from_mixed_generations():
    profile = linear_5mhz()
    config = _config_for_events(profile, 4)
    events = list(config.events)
    records = simulate_frame(profile, config, [PointScatterer(0.0, 20e-3)])
    stale = RFEventRecord(
        header=_header(
            seq=records[-1].header.seq,
            config_id=records[-1].header.config_id,
            param_generation=records[-1].header.param_generation + 1,
            tx_event_index=records[-1].header.tx_event_index,
        ),
        data=np.array(records[-1].data),
    )

    mixed = [*records[:-1], stale]

    with pytest.raises(ValueError):
        das_rf_golden(profile, events, mixed)


def test_beamform_matches_records_by_event_index_not_by_position():
    """The header names the event; arrival order carries no meaning."""
    profile = linear_5mhz()
    config = _config_for_events(profile, 6)
    events = list(config.events)
    records = simulate_frame(profile, config, [PointScatterer(0.0, 20e-3)])

    in_order, _, _ = das_rf_golden(profile, events, records)
    shuffled, _, _ = das_rf_golden(profile, events, list(reversed(records)))

    assert np.array_equal(in_order, shuffled)


def test_log_compression_of_a_silent_frame_is_the_floor_not_nan():
    env = envelope(np.zeros((32, 4), dtype=np.float32))
    db = log_compress(env, dynamic_range_db=50.0)

    assert np.all(np.isfinite(db))
    assert np.allclose(db, -50.0)


def test_record_rejects_a_payload_too_short_to_interpolate():
    """Fractional delays need two neighbouring samples; one cannot be interpolated."""
    header = _header()
    one_sample = np.zeros((4, 1), dtype=np.int16)
    no_samples = np.zeros((4, 0), dtype=np.int16)

    with pytest.raises(ValueError):
        RFEventRecord(header=header, data=one_sample)
    with pytest.raises(ValueError):
        RFEventRecord(header=header, data=no_samples)


def test_beamform_rejects_an_integer_dtype():
    """The dtype parameter sweeps precision; an integer one silently destroys
    the fractional delays and the apodization weights alike."""
    profile = linear_5mhz()
    config = _config_for_events(profile, 2)
    events = list(config.events)
    records = simulate_frame(profile, config, [PointScatterer(0.0, 20e-3)])

    with pytest.raises(ValueError):
        das_rf_golden(profile, events, records, dtype=np.int16)


def test_beamform_rejects_a_payload_whose_channel_count_is_not_the_aperture():
    """A record must carry exactly the profile's channels: too few fails on an
    incidental index, too many are summed as if the aperture were smaller."""
    profile = linear_5mhz()
    config = _config_for_events(profile, 2)
    events = list(config.events)
    records = simulate_frame(profile, config, [PointScatterer(0.0, 20e-3)])

    for n_ch in (profile.n_elements - 1, profile.n_elements + 1):
        resized = [
            RFEventRecord(
                header=r.header,
                data=np.zeros((n_ch, r.data.shape[1]), dtype=np.int16),
            )
            for r in records
        ]
        with pytest.raises(ValueError):
            das_rf_golden(profile, events, resized)
