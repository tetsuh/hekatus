"""The per-stage comparison and the decimation sweep, pinned (design.md §15, §17, #6).

Every figure design.md quotes from `golden_compare` and `decimation_sweep`
is held here, on the profile's provisional bandwidth (§4) — rerun when that
changes. The yardstick's own residual is quoted beside the figures and a
difference below it is flagged, not silently reported.
"""

import math

import pytest

from enodia.spec.beamform import decimation_sweep, golden_compare
from enodia.spec.beamform.golden_compare import StageError, compare


@pytest.fixture(scope="module")
def sweep(frame):
    profile, events, records, scatterers = frame
    return decimation_sweep.sweep(profile, events, records, scatterers)


@pytest.mark.parametrize(
    ("decimation", "cp1_pct", "cp1_phase", "cp2_pct", "cp2_phase", "rms_db", "max_db"),
    [
        (8, 7.946, 8.23, (21.09, 21.07, 19.93), (1.04, 1.20, 0.82), 5.102, 20.818),
        (4, 0.204, 0.19, (2.433, 2.432, 2.433), (0.20, 0.22, 0.20), 0.850, 5.569),
    ],
)
def test_the_per_stage_figures_quoted_in_the_design_are_pinned(
    sweep, decimation, cp1_pct, cp1_phase, cp2_pct, cp2_phase, rms_db, max_db
):
    r = sweep.reports[decimation]
    assert r.profile == "linear-5mhz"
    assert r.bandwidth_status == "provisional"
    assert r.checkpoint1[1].relative_error_pct == pytest.approx(cp1_pct, abs=0.005)
    assert r.checkpoint1[1].phase_error_deg == pytest.approx(cp1_phase, abs=0.05)
    # The int16 record adds almost nothing to the FIR's own error.
    assert abs(r.checkpoint1[1].relative_error_pct - r.checkpoint1[0].relative_error_pct) < 0.002
    assert r.checkpoint2_events == (47, 63, 87)
    for e, pct, ph in zip(r.checkpoint2, cp2_pct, cp2_phase, strict=True):
        assert e.relative_error_pct == pytest.approx(pct, abs=0.02)
        assert e.phase_error_deg == pytest.approx(ph, abs=0.05)
    assert r.image.rms_db == pytest.approx(rms_db, abs=0.01)
    assert r.image.max_db == pytest.approx(max_db, abs=0.05)


def test_the_yardstick_floor_is_quoted_and_a_difference_below_it_is_flagged(sweep):
    r = sweep.reports[8]
    assert r.floor_pct == pytest.approx(0.0003, abs=0.00005)
    text = "\n".join(r.lines())
    assert "yardstick floor" in text
    assert "0.0003 %" in text
    assert "not attributable" in text
    tiny = StageError("hypothetical", 0.0001, 0.0)
    assert "not attributable" in tiny.line(r.floor_pct)
    assert "not attributable" not in r.checkpoint1[1].line(r.floor_pct)


def test_a_silent_reference_reports_nan_rather_than_dropping_the_event(frame, golden):
    profile, events, records, scatterers = frame
    # Event 0's line is 19 mm from every scatterer: its record is quantized silence.
    r = compare(
        profile,
        events,
        records,
        scatterers,
        decimation=8,
        golden=golden,
        checkpoint2_events=events[:1],
    )
    assert r.checkpoint2_events == (0,)
    assert math.isnan(r.checkpoint2[0].relative_error_pct)
    assert "nan" in r.checkpoint2[0].line(r.floor_pct)


@pytest.mark.parametrize(
    ("decimation", "w6", "w20", "peaks"),
    [
        (8, (0.276, 0.270, 0.257), (0.420, 0.423, 0.426), (-0.761, -0.283, 0.000)),
        (4, (0.202, 0.201, 0.201), (0.361, 0.360, 0.360), (-0.380, 0.000, -0.008)),
    ],
)
def test_the_axial_psf_at_each_decimation_ratio_is_pinned(sweep, decimation, w6, w20, peaks):
    """design.md §5 / §17: how the point-scatterer axial PSF changes between
    D=8 and D=4, on the provisional 5 MHz profile. Golden: 0.194 mm at −6 dB,
    0.355–0.356 mm at −20 dB."""
    for g in sweep.golden:
        assert g.widths_mm[-6.0] == pytest.approx(0.194, abs=0.002)
        assert g.widths_mm[-20.0] == pytest.approx(0.355, abs=0.002)
    for psf, a, b, pk in zip(sweep.iq[decimation], w6, w20, peaks, strict=True):
        assert psf.widths_mm[-6.0] == pytest.approx(a, abs=0.002)
        assert psf.widths_mm[-20.0] == pytest.approx(b, abs=0.002)
        assert psf.peak_db == pytest.approx(pk, abs=0.01)


def test_the_report_says_what_the_sweep_measured_and_on_what(sweep):
    text = "\n".join(sweep.lines())
    for needle in ("linear-5mhz", "provisional", "IQ D=8", "IQ D=4", "-6 / -20 dB", "floor"):
        assert needle in text, needle
    assert golden_compare.__doc__
    assert "not attributable" in golden_compare.__doc__
