"""The fractional-delay interpolation kernel, pinned so two ports cannot differ.

Acceptance of any port is numerical equivalence with the reference (L0), and
two implementations that pick different 4-tap kernels both look right in
isolation while differing in exactly the way L0 exists to catch (#22). So
the kernel is defined here to the coefficient, its coordinate convention and
boundary rule are tested, and the sweep behind the choice is reproduced on
every run rather than quoted from a session.
"""

import numpy as np
import pytest

from enodia.spec.beamform import interp
from enodia.spec.beamform.interp import (
    KERNELS,
    fractional_delay,
    fractional_delay_taps,
)
from enodia.spec.beamform.interp_sweep import (
    BAND_EDGES,
    CANDIDATES,
    worst_band_edge_error,
)


def test_the_kernel_set_is_closed_and_has_one_member():
    """One value defined; the argument is a seam for L0, not a menu. A second
    kernel is a contract change (ADR-0007), not a new string."""
    assert KERNELS == ("lagrange4",)
    with pytest.raises(ValueError, match="kernel"):
        fractional_delay_taps(0.5, kernel="cubic_4tap")


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.25, 0.5, 0.75, 0.999])
def test_taps_have_unit_dc_gain(mu):
    assert fractional_delay_taps(mu).sum() == pytest.approx(1.0, abs=1e-12)


def test_the_closed_form_is_the_lagrange_cubic_basis():
    """The coefficients are the Lagrange basis on nodes {-1, 0, 1, 2}, in that
    tap order — written out so a port can check its own against a formula
    rather than against this module."""
    mu = 0.3
    expected = np.array(
        [
            -mu * (mu - 1) * (mu - 2) / 6,
            (mu + 1) * (mu - 1) * (mu - 2) / 2,
            -(mu + 1) * mu * (mu - 2) / 2,
            (mu + 1) * mu * (mu - 1) / 6,
        ]
    )
    np.testing.assert_allclose(fractional_delay_taps(mu), expected, rtol=0, atol=1e-15)


def test_at_zero_fraction_the_kernel_reads_the_sample_itself():
    np.testing.assert_allclose(fractional_delay_taps(0.0), [0.0, 1.0, 0.0, 0.0], atol=1e-15)


def test_taps_are_computed_in_batch_over_an_array_of_fractions():
    mus = np.array([[0.0, 0.5], [0.25, 0.75]])
    taps = fractional_delay_taps(mus)
    assert taps.shape == (2, 2, 4)
    np.testing.assert_allclose(taps[0, 0], fractional_delay_taps(0.0))
    np.testing.assert_allclose(taps[1, 1], fractional_delay_taps(0.75))


def test_a_cubic_polynomial_is_reproduced_exactly_between_samples():
    """Lagrange cubic interpolation is exact for polynomials up to degree 3;
    that is what makes it maximally flat at DC and the reason it is chosen."""
    n = np.arange(32, dtype=np.float64)
    z = 0.5 * n**3 - 2.0 * n**2 + 3.0 * n - 1.0
    t = np.array([3.25, 10.5, 17.9, 20.0])
    expected = 0.5 * t**3 - 2.0 * t**2 + 3.0 * t - 1.0
    np.testing.assert_allclose(fractional_delay(z, t), expected, rtol=1e-12)


def test_the_coordinate_convention_is_position_in_samples_from_the_record_start():
    """`t = n - d` in the design's notation: sample coordinate, zero at the
    first sample. Integer t reads that sample; the fraction runs forward
    from floor(t) to the next one."""
    z = np.arange(10, dtype=np.float64) * 10.0  # z[k] = 10 k
    np.testing.assert_allclose(fractional_delay(z, np.array([4.0])), [40.0])
    np.testing.assert_allclose(fractional_delay(z, np.array([4.5])), [45.0])
    np.testing.assert_allclose(fractional_delay(z, np.array([4.999])), [49.99], rtol=1e-12)


def test_the_record_is_zero_outside_its_ends():
    """Before the first sample is pre-transmit and past the last is no data;
    both are read as zero. Stated so a port does not clamp, mirror, or
    extrapolate — each is defensible, and each disagrees with the others."""
    z = np.ones(8)
    # Well inside: exact.
    np.testing.assert_allclose(fractional_delay(z, np.array([3.5])), [1.0])
    # A tap reaching past the end (t=6.5 uses samples 5..8; 8 is outside):
    # the outside tap contributes zero, so the value is the partial sum.
    taps = fractional_delay_taps(0.5)
    np.testing.assert_allclose(fractional_delay(z, np.array([6.5])), [taps[:3].sum()])
    # Entirely outside: zero, not an index error and not an edge value.
    np.testing.assert_allclose(fractional_delay(z, np.array([-3.0, 12.0])), [0.0, 0.0])


def test_complex_records_are_interpolated_componentwise():
    """The consumer is IQ (design.md §5): the same real taps on I and Q."""
    n = np.arange(16, dtype=np.float64)
    z = np.exp(1j * 0.3 * n)
    t = np.array([5.25])
    got = fractional_delay(z, t)
    expected = fractional_delay(z.real, t) + 1j * fractional_delay(z.imag, t)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert np.iscomplexobj(got)


def test_interpolation_runs_along_the_last_axis_of_a_channel_stack():
    """Channels × samples in, channels × positions out."""
    z = np.stack([np.arange(8.0), 2.0 * np.arange(8.0)])
    t = np.array([[1.5, 2.5], [1.5, 2.5]])
    got = fractional_delay(z, t)
    np.testing.assert_allclose(got, [[1.5, 2.5], [3.0, 5.0]])


# --- the sweep behind the choice --------------------------------------------


def test_the_sweep_covers_the_decimation_cases_the_design_names():
    """design.md §5 names three: 5 MHz at D=8 and D=4, 13 MHz at D=2."""
    assert set(BAND_EDGES) == {"5MHz_D8", "13MHz_D2", "5MHz_D4"}
    assert BAND_EDGES["5MHz_D8"] == pytest.approx(0.30)
    assert BAND_EDGES["13MHz_D2"] == pytest.approx(0.26)
    assert BAND_EDGES["5MHz_D4"] == pytest.approx(0.15)


def test_the_chosen_kernel_matches_the_figures_quoted_in_the_design():
    """The numbers in design.md §5 are these, to the digit shown there."""
    phase, mag = worst_band_edge_error(CANDIDATES["lagrange4"], BAND_EDGES["5MHz_D8"])
    assert phase == pytest.approx(3.87, abs=0.01)
    assert mag == pytest.approx(0.220, abs=0.001)
    phase, mag = worst_band_edge_error(CANDIDATES["lagrange4"], BAND_EDGES["13MHz_D2"])
    assert phase == pytest.approx(1.95, abs=0.01)
    assert mag == pytest.approx(0.134, abs=0.001)
    phase, mag = worst_band_edge_error(CANDIDATES["lagrange4"], BAND_EDGES["5MHz_D4"])
    assert phase == pytest.approx(0.14, abs=0.01)
    assert mag == pytest.approx(0.017, abs=0.001)


def test_the_spec_kernel_and_the_sweep_candidate_are_the_same_function():
    """The sweep must measure the kernel that ships, not a re-typed copy."""
    for mu in (0.0, 0.3, 0.5, 0.9):
        np.testing.assert_allclose(CANDIDATES["lagrange4"](mu), fractional_delay_taps(mu))


def test_lagrange_is_the_best_closed_form_four_tap_on_phase_and_ties_on_magnitude():
    """Why Lagrange: half of linear's phase error, and no other closed-form
    4-tap in the sweep beats it on either axis. Keys (Catmull-Rom) buys the
    same magnitude flatness but its phase error equals linear's."""
    edge = BAND_EDGES["5MHz_D8"]
    lag_ph, lag_mag = worst_band_edge_error(CANDIDATES["lagrange4"], edge)
    for name in ("linear2", "keys4", "hann_sinc4"):
        ph, mag = worst_band_edge_error(CANDIDATES[name], edge)
        assert lag_ph <= ph + 1e-9, name
        assert lag_mag <= mag + 1e-9, name
    lin_ph, _ = worst_band_edge_error(CANDIDATES["linear2"], edge)
    assert lag_ph < 0.55 * lin_ph


def test_phase_rotation_alone_is_the_error_the_design_states():
    """The 47° and 54° in design.md §5 are the no-interpolation case at the
    worst fraction; the sweep's metric reduces to them, which pins the metric
    to the design's framing."""
    assert interp_sweep_no_interp_worst(BAND_EDGES["13MHz_D2"]) == pytest.approx(46.8, abs=0.1)
    assert interp_sweep_no_interp_worst(BAND_EDGES["5MHz_D8"]) == pytest.approx(54.0, abs=0.1)


def interp_sweep_no_interp_worst(edge: float) -> float:
    from enodia.spec.beamform.interp_sweep import no_interpolation_worst_phase_deg

    return no_interpolation_worst_phase_deg(edge)


def test_the_module_docstring_names_the_issue_that_pinned_it():
    assert "#22" in interp.__doc__
