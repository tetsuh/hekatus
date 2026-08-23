"""The IQ-domain DAS against the RF golden (design.md §5, §15, #6).

Acceptance: the point scatterers land where `tests/test_das_point.py`
requires of the golden, within a stated tolerance of the golden's own
peaks; the phase-sign convention is asserted at checkpoint 2, where a
flipped sign is ~97° and the correct one under a degree — §5 warns the image
cannot tell them apart; and nothing in `enodia/spec/` names an accelerator.
"""

import pathlib
import re

import numpy as np
import pytest

from enodia.spec.beamform import envelope, log_compress
from enodia.spec.beamform.golden_compare import golden_channel_vectors
from enodia.spec.beamform.iq_das import das_iq, delayed_channel_vectors
from enodia.spec.frontend import demodulate_frame

# Tolerances the acceptance is stated with.
AXIAL_TOLERANCE_M = 25e-6  # one RF sample is 19.25 µm; D=8 moves one peak by that
LATERAL_TOLERANCE_M = 0.1e-3  # the same scanline
LEVEL_TOLERANCE_DB = 1.0  # D=8 costs up to 0.37 dB on a peak


@pytest.fixture(scope="module", params=[8, 4], ids=["D=8", "D=4"])
def iq_path(request, frame):
    profile, events, records, _ = frame
    d = request.param
    iq_records = demodulate_frame(records, profile, decimation=d)
    image, z, line_x = das_iq(profile, events, iq_records, decimation=d)
    return d, iq_records, image, z, line_x


def _peak(db, z, line_x, s):
    near_z = np.abs(z - s.z_m) < 2e-3
    near_x = np.abs(line_x - s.x_m) < 2e-3
    window = db[np.ix_(near_z, near_x)]
    iz, ix = np.unravel_index(np.argmax(window), window.shape)
    return z[near_z][iz], line_x[near_x][ix], window.max()


def test_point_scatterers_land_where_the_golden_puts_them(frame, golden, iq_path):
    """The criteria of tests/test_das_point.py, and additionally within the
    stated tolerance of the golden's own peaks."""
    _, _, _, scatterers = frame
    d, _, image, z, line_x = iq_path
    db_iq = log_compress(np.abs(image))
    db_g = log_compress(envelope(golden[0]))
    for s in scatterers:
        zi, xi, li = _peak(db_iq, z, line_x, s)
        zg, xg, lg = _peak(db_g, z, line_x, s)
        assert abs(zi - s.z_m) < 0.5e-3, f"D={d}: axial peak off at {s}"
        assert abs(xi - s.x_m) < 1.0e-3, f"D={d}: lateral peak off at {s}"
        assert li > -6.0
        assert abs(zi - zg) <= AXIAL_TOLERANCE_M, f"D={d}: axial peak differs from golden at {s}"
        assert abs(xi - xg) <= LATERAL_TOLERANCE_M
        assert abs(li - lg) <= LEVEL_TOLERANCE_DB


def test_the_phase_sign_convention_is_asserted_at_checkpoint_2_not_in_a_comment(frame, iq_path):
    """With e^(−j2πf0·τ) the post-delay channel vectors agree in phase with
    the golden's analytic channel samples to well under a degree, energy-
    weighted; with the sign flipped they are off by ~97°. An image made
    with the flipped sign still has its peaks in plausible places — which
    is why this is asserted here and not by looking at one."""
    profile, events, records, scatterers = frame
    d, iq_records, _, z, _ = iq_path
    ev = min(events, key=lambda e: abs(e.line_x_m - scatterers[0].x_m))
    rec = next(r for r in records if r.header.tx_event_index == ev.event_index)
    iq = next(r for r in iq_records if r.header.tx_event_index == ev.event_index)
    x = delayed_channel_vectors(profile, ev, iq, z, decimation=d, dtype=np.float64)
    ref, w = golden_channel_vectors(profile, ev, rec, z)
    inside = w > 0

    def phase_rms(a):
        energy = np.abs(ref[inside]) ** 2 * w[inside]
        ph = np.angle(a[inside] * np.conj(ref[inside]))
        return np.degrees(np.sqrt(np.sum(energy * ph**2) / energy.sum()))

    dx = profile.element_x()[:, None] - ev.line_x_m
    tau_i = (z[None, :] + np.hypot(dx, z[None, :])) / profile.c_m_s
    tau = 2.0 * z[None, :] / profile.c_m_s - tau_i
    flipped = x * np.exp(+2j * 2.0 * np.pi * profile.f0_hz * tau)  # e^(+j2πf0τ) instead

    assert phase_rms(x) < 1.5
    assert phase_rms(flipped) > 60.0


def test_the_demo_iq_path_images_the_scatterers_on_the_shared_grid():
    """`run_pipeline(path="iq")` is the one-command path of #6: a finite
    log-compressed image on the golden's grid, scatterers at their true
    positions, the same criteria tests/test_das_point.py applies to the golden."""
    from enodia.demo import DEFAULT_SCATTERERS, run_pipeline
    from enodia.spec.beamform import depth_grid
    from enodia.spec.probe import linear_5mhz

    profile = linear_5mhz()
    db, z, line_x, records = run_pipeline(profile, DEFAULT_SCATTERERS, path="iq", decimation=8)
    assert np.all(np.isfinite(db))
    assert db.shape == (depth_grid(profile).size, profile.n_elements)
    np.testing.assert_array_equal(z, depth_grid(profile))
    assert line_x.shape == (profile.n_elements,)
    assert len(records) == profile.n_elements
    for s in DEFAULT_SCATTERERS:
        zi, xi, li = _peak(db, z, line_x, s)
        assert abs(zi - s.z_m) < 0.5e-3
        assert abs(xi - s.x_m) < 1.0e-3
        assert li > -6.0
    with pytest.raises(ValueError, match="path"):
        run_pipeline(profile, DEFAULT_SCATTERERS, path="rf")


def test_the_beamformer_refuses_records_decimated_at_another_ratio(frame):
    profile, events, records, _ = frame
    iq4 = demodulate_frame(records[:2], profile, decimation=4)
    with pytest.raises(ValueError, match="decimated by 4"):
        das_iq(profile, events[:2], iq4, decimation=8)


def test_the_beamformer_rejects_an_integer_dtype(frame):
    profile, events, records, _ = frame
    iq = demodulate_frame(records[:1], profile, decimation=8)
    with pytest.raises(ValueError, match="floating"):
        das_iq(profile, events[:1], iq, decimation=8, dtype=np.int16)


@pytest.mark.parametrize(
    ("dtype", "expected"), [(np.float32, np.complex64), (np.float64, np.complex128)]
)
def test_the_intermediate_is_complex_fp32_or_wider(frame, dtype, expected):
    profile, events, records, _ = frame
    iq = demodulate_frame(records[:1], profile, decimation=8)
    image, _, _ = das_iq(profile, events[:1], iq, decimation=8, dtype=dtype)
    assert image.dtype == expected


def test_nothing_in_the_spec_imports_an_accelerator_backend():
    """CLAUDE.md: the reference implementation is hardware-neutral, and #6
    asks that nothing in enodia/spec acquire an accelerator-specific concept.
    What a test can check is the executable part of that: no module under
    enodia/spec imports an accelerator backend. (The generic word
    "accelerator" appears in docstrings, in its generic sense; that is not
    what this guards against.)"""
    forbidden = re.compile(
        r"^\s*(import|from)\s+(ttnn|tt_lib|tt_metal|ttml|holoscan|cupy|torch|cuda|pycuda)\b",
        re.MULTILINE,
    )
    spec = pathlib.Path(__file__).resolve().parents[1] / "enodia" / "spec"
    for path in spec.rglob("*.py"):
        assert not forbidden.search(path.read_text(encoding="utf-8")), path
