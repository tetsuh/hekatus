"""A profile's own result, beside the frozen RF oracle and never in its place (#46).

`rf-oracle-frozen-0p7` is a historical synthetic record; what a named
profile implies is reported separately, with every provenance field a
quoted figure needs, and is rerun when the profile's value or provenance
changes. These tests pin the 5 MHz reconciliation and the separation.
"""

import numpy as np
import pytest

from enodia.spec.beamform import profile_reconciliation as pr
from enodia.spec.beamform import rf_delay_sweep as rf
from enodia.spec.probe import BANDWIDTH_LEVEL_DB, linear_5mhz


@pytest.fixture(scope="module")
def five_mhz():
    profile = linear_5mhz()
    return {d: pr.reconcile(profile, d, revision="test") for d in (8, 4)}


def test_the_output_carries_every_provenance_field(five_mhz):
    r = five_mhz[8]
    assert r.profile == "linear-5mhz"
    assert r.status == "provisional"
    assert r.source is None
    assert r.f0_hz == 5e6
    assert r.bandwidth_frac == 0.7
    assert r.bandwidth_edge_hz == pytest.approx(1.75e6)
    assert r.level_db == BANDWIDTH_LEVEL_DB
    assert "half amplitude" in r.convention
    assert "one-sided" in r.convention
    assert r.decimation == 8
    assert r.producing_revision == "test"
    assert r.iq_kernel == "lagrange4"
    assert r.frozen_oracle == rf.BENCHMARK_NAME == "rf-oracle-frozen-0p7"
    text = "\n".join(r.lines())
    for needle in (
        "linear-5mhz",
        "provisional",
        "source: none",
        "1.75 MHz",
        "6.0206 dB",
        "decimation: 8",
        "revision: test",
        "rf-oracle-frozen-0p7",
    ):
        assert needle in text, needle


@pytest.mark.parametrize(
    ("decimation", "phase", "mag", "weighted"),
    [(8, 8.25, 0.3658, 14.00), (4, 0.29, 0.0310, 2.39)],
)
def test_the_5mhz_profile_specific_iq_result_is_pinned(five_mhz, decimation, phase, mag, weighted):
    """The acceptance targets #46 froze, within 0.01° and 0.01 point, and
    exactly the figures the §5 sweep reports for the profile's cases."""
    r = five_mhz[decimation]
    assert r.iq_worst_phase_deg == pytest.approx(phase, abs=0.01)
    assert r.iq_worst_magnitude_error == pytest.approx(mag, abs=0.0001)
    assert r.iq_weighted_rms_pct == pytest.approx(weighted, abs=0.01)


def test_the_5mhz_profile_record_is_numerically_the_frozen_one_and_its_residual_the_floor(five_mhz):
    """For `linear-5mhz` the RF input is the frozen 5 MHz record to the bit —
    same pulse, same 0.7 — so the golden residual is the frozen 0.0003 %. The
    IQ-side figure above is what changed, not this."""
    r = five_mhz[8]
    assert r.rf_record_is_frozen_oracle is True
    np.testing.assert_array_equal(pr.profile_record(linear_5mhz()), rf.benchmark_record(5e6))
    assert r.rf_golden_residual_pct == pytest.approx(0.0003, abs=0.00005)


def test_a_profile_that_is_not_the_frozen_record_says_so():
    from dataclasses import replace

    other = replace(linear_5mhz(), bandwidth_frac=0.6, bandwidth_source="hypothetical")
    r = pr.reconcile(other, 8, revision="test")
    assert r.rf_record_is_frozen_oracle is False
    assert r.status == "sourced"
    # The frozen record itself did not move with it.
    np.testing.assert_array_equal(rf.benchmark_record(5e6), pr.profile_record(linear_5mhz()))


def test_the_frozen_oracle_does_not_drift_with_the_profile():
    """The record is 0.7 by its own constant, the limits are the frozen ones,
    and the production residuals are what §15 quotes — all unchanged by #46."""
    assert rf.BENCHMARK_BANDWIDTH_FRAC == 0.7
    assert rf.BENCHMARK_N == 256
    assert rf.BENCHMARK_FS_HZ == 40e6
    assert rf.BENCHMARK_FRACTIONS == 201
    assert rf.RESIDUAL_LIMIT_PCT == {"5MHz": 1.082, "13MHz": 0.791}
    for carrier, residual in (("5MHz", 0.0003), ("13MHz", 0.0992)):
        got = rf.residual_pct(rf.production, rf.benchmark_record(rf.BENCHMARK_CARRIERS_HZ[carrier]))
        assert got == pytest.approx(residual, abs=0.00005)


def test_issue_10_is_named_as_the_trigger_for_the_13mhz_rerun():
    assert "#10" in pr.__doc__
    assert "13 MHz" in pr.__doc__


def test_the_producing_revision_is_a_string():
    assert isinstance(pr.producing_revision(), str)
    assert pr.producing_revision() != ""
