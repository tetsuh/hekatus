from itertools import pairwise

import numpy as np

from enodia.spec.probe import linear_5mhz
from enodia.spec.sequence import make_bmode_sequence
from enodia.spec.sim import PointScatterer, n_rf_samples, simulate_bmode_frame


def test_rf_records_carry_the_specified_format_and_metadata():
    """Format accuracy is the simulator's first requirement (design.md §15),
    and every record names its generation (docs/dataplane.md)."""
    p = linear_5mhz()
    events = make_bmode_sequence(p)[:4]
    records = simulate_bmode_frame(p, events, [PointScatterer(0.0, 20e-3)])

    assert len(records) == 4
    for k, r in enumerate(records):
        assert r.data.dtype == np.int16
        assert r.data.shape == (p.n_elements, n_rf_samples(p))
        assert r.header.seq == k
        assert r.header.config_id == "default"
        assert r.header.tx_type == "bmode_focused"

    timestamps = [r.header.timestamp_ns for r in records]
    assert all(later > earlier for earlier, later in pairwise(timestamps))
