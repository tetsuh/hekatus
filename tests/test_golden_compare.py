"""The per-stage comparison and the decimation sweep, pinned (design.md §15, §17, #6).

Every figure design.md quotes from `golden_compare` and `decimation_sweep`
is held here, on the profile's provisional bandwidth (§4) — rerun when that
changes. The yardstick's own residual is quoted beside the figures and a
difference below it is flagged, not silently reported.
"""

import math

import numpy as np
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
    ("decimation", "w6", "w20", "w40", "peaks"),
    [
        (
            8,
            (0.276, 0.270, 0.257),
            (0.420, 0.423, 0.426),
            (0.614, 0.626, 0.804),
            (-0.761, -0.283, 0.000),
        ),
        (
            4,
            (0.202, 0.201, 0.201),
            (0.361, 0.360, 0.360),
            (0.492, 0.490, 0.490),
            (-0.380, 0.000, -0.008),
        ),
    ],
)
def test_the_axial_psf_at_each_decimation_ratio_is_pinned(sweep, decimation, w6, w20, w40, peaks):
    """design.md §5 / §17: how the point-scatterer axial PSF changes between
    D=8 and D=4, on the provisional 5 MHz profile. Golden: 0.194 mm at −6 dB,
    0.355–0.356 mm at −20 dB, 0.500–0.501 mm at −40 dB — the −40 dB width
    because §15 says never to argue resolution from the −6 dB width alone."""
    assert decimation_sweep.WIDTH_LEVELS_DB == (-6.0, -20.0, -40.0)
    for g in sweep.golden:
        assert g.widths_mm[-6.0] == pytest.approx(0.194, abs=0.002)
        assert g.widths_mm[-20.0] == pytest.approx(0.355, abs=0.002)
        assert g.widths_mm[-40.0] == pytest.approx(0.500, abs=0.002)
    for psf, a, b, c, pk in zip(sweep.iq[decimation], w6, w20, w40, peaks, strict=True):
        assert psf.widths_mm[-6.0] == pytest.approx(a, abs=0.002)
        assert psf.widths_mm[-20.0] == pytest.approx(b, abs=0.002)
        assert psf.widths_mm[-40.0] == pytest.approx(c, abs=0.002)
        assert psf.peak_db == pytest.approx(pk, abs=0.01)


def test_a_silent_golden_frame_compares_to_nan_rather_than_raising():
    from enodia.spec.beamform.golden_compare import image_comparison
    from enodia.spec.sim import PointScatterer

    silent = np.full((16, 4), -50.0)
    z = np.linspace(0.01, 0.02, 16)
    line_x = np.linspace(-1e-3, 1e-3, 4)
    r = image_comparison(silent, silent, z, line_x, [PointScatterer(0.0, 15e-3)])
    assert math.isnan(r.rms_db)
    assert math.isnan(r.max_db)


def _fake_sweep_git(status_output):
    def fake_git(*args):
        return "abc1234def" if args[0] == "rev-parse" else status_output

    return fake_git


def test_a_clean_sweep_harness_is_recorded_as_clean(monkeypatch):
    monkeypatch.setattr(decimation_sweep, "_git", _fake_sweep_git(""))

    identity = decimation_sweep.environment()

    assert identity["harness_commit"] == "abc1234def"
    assert identity["harness_dirty"] is False


def test_a_modified_sweep_harness_is_recorded_as_dirty(monkeypatch):
    monkeypatch.setattr(
        decimation_sweep, "_git", _fake_sweep_git(" M enodia/spec/frontend/__init__.py")
    )

    assert decimation_sweep.environment()["harness_dirty"] is True


def test_an_unavailable_sweep_harness_status_remains_unknown(monkeypatch):
    monkeypatch.setattr(decimation_sweep, "_git", _fake_sweep_git("unknown"))

    assert decimation_sweep.environment()["harness_dirty"] is None


def test_the_measurement_record_is_strict_json_with_nulls_for_non_finite(sweep, frame):
    """ADR-0005 records are data other tools read; `json.dumps` would happily
    emit `NaN`, which strict parsers reject. Non-finite floats become null and
    the record is dumped with allow_nan=False."""
    import json

    from enodia.spec.beamform.decimation_sweep import json_safe, measurement_record

    profile = frame[0]
    record = measurement_record(sweep, profile)
    text = json.dumps(record, allow_nan=False)
    strict = json.loads(text, parse_constant=lambda tok: (_ for _ in ()).throw(ValueError(tok)))
    assert strict["environment"]["harness_commit"]
    assert strict["profile"]["bandwidth_status"] == "provisional"
    assert strict["iq_psf"]["8"][0]["full_width_mm"]["-40"] == pytest.approx(0.614, abs=0.002)
    # The sanitizer itself, on the shapes the report produces.
    nasty = {"a": float("nan"), "b": (1.0, float("inf")), "c": np.float32(2.5), "d": np.int64(3)}
    assert json_safe(nasty) == {"a": None, "b": [1.0, None], "c": 2.5, "d": 3}
    json.dumps(json_safe(nasty), allow_nan=False)


def test_the_report_says_what_the_sweep_measured_and_on_what(sweep):
    text = "\n".join(sweep.lines())
    for needle in ("linear-5mhz", "provisional", "IQ D=8", "IQ D=4", "-6 / -20 / -40 dB", "floor"):
        assert needle in text, needle
    assert golden_compare.__doc__
    assert "not attributable" in golden_compare.__doc__
