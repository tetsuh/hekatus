from itertools import pairwise

import numpy as np
import pytest

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


def test_the_simulator_pulse_is_half_amplitude_at_the_profile_band_edges():
    """The profile's bandwidth definition (design.md §4, ADR-0008) is what the
    simulator's pulse implements: the amplitude spectrum of
    `gaussian_pulse(t, f0, bandwidth_frac)` is half its peak at
    f0 ± bandwidth_edge_hz. Checked on the spectrum, not on the σ formula,
    so the two cannot agree by construction."""
    from enodia.spec.probe import BANDWIDTH_LEVEL_DB
    from enodia.spec.sim import gaussian_pulse

    p = linear_5mhz()
    fs = 400e6  # fine grid: the check is on the continuous-time pulse
    n = 2**16
    t = (np.arange(n) - n // 2) / fs
    spectrum = np.abs(np.fft.rfft(gaussian_pulse(t, p.f0_hz, p.bandwidth_frac)))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    peak = spectrum.max()
    for edge in (p.f0_hz - p.bandwidth_edge_hz, p.f0_hz + p.bandwidth_edge_hz):
        at_edge = np.interp(edge, f, spectrum)
        assert 20.0 * np.log10(at_edge / peak) == pytest.approx(-BANDWIDTH_LEVEL_DB, abs=0.01)
    assert BANDWIDTH_LEVEL_DB == pytest.approx(6.020599913, abs=1e-9)


def test_bandwidth_provenance_is_none_or_a_named_source():
    """An empty source string would pass a `is None` check while naming
    nothing; provisional is spelled None and only None."""
    from dataclasses import replace

    p = linear_5mhz()
    assert p.bandwidth_status == "provisional"
    assert replace(p, bandwidth_source="manufacturer datasheet X").bandwidth_status == "sourced"
    with pytest.raises(ValueError, match="bandwidth_source"):
        replace(p, bandwidth_source="   ")


@pytest.mark.parametrize("bad", [0.0, -0.7, float("nan"), float("inf")])
def test_bandwidth_frac_must_be_finite_and_positive(bad):
    """NaN passes `x <= 0` because every comparison with NaN is false, and
    +inf passes it too; either would reach the sweep as a band edge."""
    from dataclasses import replace

    p = linear_5mhz()
    with pytest.raises(ValueError, match="bandwidth_frac"):
        replace(p, bandwidth_frac=bad)
