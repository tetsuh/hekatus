"""The front end: complex band-pass FIR fused with the mixer, then decimation (design.md §5, #6).

What is pinned here is the contract a port must meet, not the numbers a
sweep happens to produce: the demodulation sign convention, the alignment
of an IQ sample with the RF time it stands for, the int16 complex record
format (docs/dataplane.md T1), and the decimation ratio travelling with the
record so a consumer cannot silently read D=4 data as D=8.
"""

import numpy as np
import pytest

from enodia.spec.probe import linear_5mhz


def test_demodulation_sign_convention_rotates_a_tone_above_f0_positively():
    """design.md §5 fixes z(t) = s(t)·e^(−j2πf0·t): a real tone at f0 + Δ
    becomes a baseband phasor turning *forward* at +Δ. Flipping the
    convention turns it backward — and still produces a plausible image
    downstream, which is why the sign is asserted here and not in a comment."""
    from enodia.spec.frontend import complex_bpf_decimate

    p = linear_5mhz()
    decimation = 8
    delta = 0.2e6
    n = 4096
    t = np.arange(n) / p.fs_hz
    tone = np.cos(2.0 * np.pi * (p.f0_hz + delta) * t)[None, :]

    z = complex_bpf_decimate(tone, p, decimation=decimation)
    # Steady state, away from the filter's edges.
    mid = z[0, 64:-64]
    phase_step = np.angle(mid[1:] * np.conj(mid[:-1]))
    expected = 2.0 * np.pi * delta * decimation / p.fs_hz

    assert np.allclose(phase_step, expected, atol=1e-3)
    assert expected > 0.0


def test_the_fused_filter_has_64_complex_taps_and_analytic_gain():
    """design.md §5: 64 complex taps per probe. The prototype's DC gain is 2,
    so a tone A·cos(2πf0·t) becomes a phasor of magnitude A — the complex
    envelope at the signal's own amplitude, not half of it."""
    from enodia.spec.frontend import ANALYTIC_GAIN, N_TAPS, bpf_taps, complex_bpf_decimate

    p = linear_5mhz()
    h = bpf_taps(p, 8)
    assert h.shape == (64,)
    assert N_TAPS == 64
    assert np.iscomplexobj(h)
    assert ANALYTIC_GAIN == 2.0
    t = np.arange(4096) / p.fs_hz
    amplitude = 1000.0
    z = complex_bpf_decimate(amplitude * np.cos(2.0 * np.pi * p.f0_hz * t)[None, :], p, decimation=8)
    assert np.allclose(np.abs(z[0, 32:-32]), amplitude, rtol=1e-3)


def test_iq_sample_m_stands_for_rf_position_m_times_d_plus_half_a_sample():
    """With an even tap count the FIR's group delay is a half sample, and
    `rf_offset` says so: an RF impulse at sample k peaks in |IQ| at the
    sample nearest (k − 0.5)/D, and the phase of that IQ sample is the
    carrier's at the RF time it stands for. A port that references the
    carrier to tap 0 instead is off by 22.5° at every sample."""
    from enodia.spec.frontend import complex_bpf_decimate, rf_offset

    assert rf_offset(64) == 0.5
    assert rf_offset(65) == 0.0
    p = linear_5mhz()
    n = 2048
    for decimation in (8, 4):
        for k in (1000, 1003, 1005):
            rf = np.zeros((1, n))
            rf[0, k] = 1.0
            z = complex_bpf_decimate(rf, p, decimation=decimation)
            peak = int(np.argmax(np.abs(z[0])))
            assert peak == round((k - 0.5) / decimation)
    # Phase: a tone at f0 with phase φ at t=0 gives IQ phase φ everywhere, to
    # within the window's tiny ripple — regardless of D and of the half sample.
    t = np.arange(n) / p.fs_hz
    phi = 0.7
    for decimation in (8, 4):
        z = complex_bpf_decimate(np.cos(2.0 * np.pi * p.f0_hz * t + phi)[None, :], p, decimation=decimation)
        assert np.allclose(np.angle(z[0, 32:-32]), phi, atol=1e-3)


def test_demodulate_returns_an_int16_two_plane_record_that_names_its_ratio_and_offset():
    from enodia.spec.frontend import demodulate
    from enodia.spec.records import IQEventRecord
    from enodia.spec.sequence import make_bmode_sequence
    from enodia.spec.sim import PointScatterer, simulate_bmode_frame

    p = linear_5mhz()
    rec = simulate_bmode_frame(p, make_bmode_sequence(p)[:1], [PointScatterer(0.0, 20e-3)])[0]
    iq = demodulate(rec, p, decimation=8)

    assert isinstance(iq, IQEventRecord)
    assert iq.data.dtype == np.int16
    assert iq.data.shape == (p.n_elements, rec.data.shape[1] // 8, 2)
    assert iq.decimation == 8
    assert iq.rf_offset == 0.5
    assert iq.header == rec.header
    assert not iq.data.flags.writeable
    z = iq.complex(np.complex64)
    assert z.dtype == np.complex64
    np.testing.assert_array_equal(z.real, iq.data[..., 0])
    np.testing.assert_array_equal(z.imag, iq.data[..., 1])


def test_demodulate_refuses_to_clip_rather_than_clipping_silently():
    from enodia.spec.frontend import demodulate
    from enodia.spec.sequence import make_bmode_sequence
    from enodia.spec.sim import PointScatterer, simulate_bmode_frame

    p = linear_5mhz()
    rec = simulate_bmode_frame(p, make_bmode_sequence(p)[:1], [PointScatterer(0.0, 20e-3)])[0]
    with pytest.raises(ValueError, match="int16"):
        demodulate(rec, p, decimation=8, iq_scale=4.0)


def test_the_iq_record_contract_is_checked():
    from enodia.spec.records import EventHeader, IQEventRecord

    h = EventHeader(0, "c", 0, 0, "bmode_focused", 0)
    good = np.zeros((2, 8, 2), dtype=np.int16)
    IQEventRecord(h, good, 8, 0.5)
    as_float = good.astype(np.float32)
    with pytest.raises(ValueError, match="int16"):
        IQEventRecord(h, as_float, 8, 0.5)
    one_plane = np.zeros((2, 8), dtype=np.int16)
    with pytest.raises(ValueError, match="I, Q"):
        IQEventRecord(h, one_plane, 8, 0.5)
    with pytest.raises(ValueError, match="decimation"):
        IQEventRecord(h, good, 0, 0.5)


def test_invalid_decimation_is_rejected():
    from enodia.spec.frontend import complex_bpf_decimate

    p = linear_5mhz()
    rf = np.zeros((1, 64))
    with pytest.raises(ValueError, match="decimation"):
        complex_bpf_decimate(rf, p, decimation=0)
