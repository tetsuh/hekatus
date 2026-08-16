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
    PULSE_ROLLOFFS_DB,
    least_squares4,
    pulse_weighted_rms_error,
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
    a tap landing there contributes zero. Stated so a port does not clamp,
    mirror, or extrapolate — each is defensible, and each disagrees."""
    z = np.ones(8)
    # Well inside: exact.
    np.testing.assert_allclose(fractional_delay(z, np.array([3.5])), [1.0])
    # A tap reaching past the end (t=6.5 uses samples 5..8; 8 is outside):
    # the outside tap contributes zero, so the value is the partial sum.
    taps = fractional_delay_taps(0.5)
    np.testing.assert_allclose(fractional_delay(z, np.array([6.5])), [taps[:3].sum()])
    # All four taps outside: zero, not an index error and not an edge value.
    np.testing.assert_allclose(fractional_delay(z, np.array([-3.0, 12.0])), [0.0, 0.0])


def test_a_position_just_outside_reads_the_taps_that_are_still_inside():
    """The rule is about taps, not positions. t = -0.5 has two taps inside the
    record, so it is a partial sum and not zero; only t < -2 or t >= N+1 puts
    all four outside. Getting this backwards would silently zero the first and
    last samples of every channel."""
    z = np.ones(8)
    taps = fractional_delay_taps(0.5)

    np.testing.assert_allclose(fractional_delay(z, np.array([-0.5])), [taps[2:].sum()])
    np.testing.assert_allclose(fractional_delay(z, np.array([7.5])), [taps[:2].sum()])
    # One step further out: a single tap remains.
    np.testing.assert_allclose(fractional_delay(z, np.array([-1.5])), [taps[3]])
    np.testing.assert_allclose(fractional_delay(z, np.array([8.5])), [taps[0]])
    # The first position with nothing inside, at each end.
    np.testing.assert_allclose(fractional_delay(z, np.array([-2.0, 9.0])), [0.0, 0.0])


@pytest.mark.parametrize(
    ("record_dtype", "expected"),
    [
        (np.int16, np.float32),
        (np.int32, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
    ],
)
def test_an_integer_record_is_promoted_the_way_the_dataflow_says(record_dtype, expected):
    """design.md §14: the IQ record is int16 complex and interpolation runs at
    FP32 intermediate. Interpolating in the record's own integer type
    truncates every tap to an integer — at t = 2.5 the four Lagrange weights
    become 0, 0, 0, 0 and the output is a black image, not a coarse one."""
    z = np.arange(8).astype(record_dtype)
    got = fractional_delay(z, np.array([2.5]))

    assert got.dtype == expected
    np.testing.assert_allclose(got, [2.5], rtol=1e-6)


def test_complex_integer_style_records_keep_both_components():
    """int16 complex is the L1-resident format; NumPy carries it as two int16
    planes, so both must survive the promotion."""
    z = (np.arange(8) + 1j * np.arange(8, 0, -1)).astype(np.complex64)

    got = fractional_delay(z, np.array([2.5]))

    assert got.dtype == np.complex64
    np.testing.assert_allclose(got, [2.5 + 5.5j], rtol=1e-6)


@pytest.mark.parametrize("t_shape", [(2,), (1, 2), (2, 2)])
def test_positions_broadcast_against_the_leading_axes_of_the_record(t_shape):
    """One set of positions applied to every channel is the ordinary call: the
    delay is per (channel, depth), but a caller sweeping one channel's record
    against shared positions must not have to tile them by hand."""
    z = np.arange(16.0).reshape(2, 8)
    t = np.broadcast_to(np.array([1.5, 2.5]), t_shape)

    got = fractional_delay(z, t)

    assert got.shape == (2, 2)
    np.testing.assert_allclose(got, [[1.5, 2.5], [9.5, 10.5]])


def test_a_single_record_broadcasts_against_many_position_rows():
    z = np.arange(8.0)
    t = np.array([[1.5], [2.5], [3.5]])

    got = fractional_delay(z, t)

    assert got.shape == (3, 1)
    np.testing.assert_allclose(got, [[1.5], [2.5], [3.5]])


def test_leading_axes_that_cannot_broadcast_are_an_error():
    with pytest.raises(ValueError):
        fractional_delay(np.zeros((3, 8)), np.zeros((2, 4)))


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


# Every figure design.md §5 quotes, pinned so the record cannot drift from
# what the code computes. Sol found the first table quoted values no test
# held (#45), which is how a table stops matching its own sweep.
_EDGE_TABLE = {
    "5MHz_D8": {
        "linear2": (7.54, 0.412),
        "lagrange4": (3.87, 0.220),
        "keys4_a050": (7.54, 0.220),
        "keys4_a075": (3.71, 0.124),
        "keys4_a100": (0.27, 0.028),
        "hann_sinc4": (10.06, 0.319),
    },
    "13MHz_D2": {
        "linear2": (4.64, 0.315),
        "lagrange4": (1.95, 0.134),
        "keys4_a050": (4.64, 0.134),
        "keys4_a075": (1.09, 0.043),
        "keys4_a100": (2.15, 0.048),
        "hann_sinc4": (6.45, 0.227),
    },
    "5MHz_D4": {
        "linear2": (0.81, 0.109),
        "lagrange4": (0.14, 0.017),
        "keys4_a050": (0.81, 0.017),
        "keys4_a075": (1.57, 0.029),
        "keys4_a100": (3.84, 0.075),
        "hann_sinc4": (1.23, 0.064),
    },
}


@pytest.mark.parametrize("case", sorted(_EDGE_TABLE))
def test_every_band_edge_figure_quoted_in_the_design_is_pinned(case):
    for name, (phase, mag) in _EDGE_TABLE[case].items():
        got_phase, got_mag = worst_band_edge_error(CANDIDATES[name], BAND_EDGES[case])
        assert got_phase == pytest.approx(phase, abs=0.01), f"{case}/{name} phase"
        assert got_mag == pytest.approx(mag, abs=0.001), f"{case}/{name} magnitude"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("5MHz_D8", (1.63, 0.090)),
        ("13MHz_D2", (0.73, 0.047)),
        ("5MHz_D4", (0.04, 0.004)),
    ],
)
def test_the_least_squares_bound_is_pinned_too(case, expected):
    """Quoted in §5 as the bound on what four taps can do, so it is evidence
    and is held to the same standard as the candidates."""
    edge = BAND_EDGES[case]
    phase, mag = worst_band_edge_error(lambda mu: least_squares4(mu, edge), edge)
    assert phase == pytest.approx(expected[0], abs=0.01)
    assert mag == pytest.approx(expected[1], abs=0.001)


_WEIGHTED_TABLE = {
    "5MHz_D8": {
        "linear2": (17.71, 5.83, 2.98),
        "lagrange4": (12.67, 1.93, 0.55),
        "keys4_a050": (12.68, 2.04, 0.62),
        "keys4_a075": (11.29, 1.86, 1.40),
        "keys4_a100": (11.33, 4.08, 3.16),
    },
    "13MHz_D2": {
        "lagrange4": (8.51, 1.16, 0.32),
        "keys4_a050": (8.57, 1.26, 0.38),
        "keys4_a075": (7.13, 1.60, 1.26),
        "keys4_a100": (7.60, 3.71, 2.77),
    },
    "5MHz_D4": {
        "lagrange4": (1.40, 0.15, 0.04),
        "keys4_a050": (1.50, 0.19, 0.06),
        "keys4_a075": (1.67, 1.07, 0.77),
        "keys4_a100": (3.85, 2.26, 1.58),
    },
}


@pytest.mark.parametrize("case", sorted(_WEIGHTED_TABLE))
def test_every_pulse_weighted_figure_quoted_in_the_design_is_pinned(case):
    for name, expected in _WEIGHTED_TABLE[case].items():
        for db, want in zip(PULSE_ROLLOFFS_DB, expected, strict=True):
            got = 100.0 * pulse_weighted_rms_error(
                CANDIDATES[name], BAND_EDGES[case], rolloff_db=db
            )
            assert got == pytest.approx(want, abs=0.01), f"{case}/{name}/-{db:g}dB"


def test_the_spec_kernel_and_the_sweep_candidate_are_the_same_function():
    """The sweep must measure the kernel that ships, not a re-typed copy."""
    for mu in (0.0, 0.3, 0.5, 0.9):
        np.testing.assert_allclose(CANDIDATES["lagrange4"](mu), fractional_delay_taps(mu))


def test_the_band_edge_metric_alone_does_not_choose_lagrange():
    """Recorded because the first version of §5 claimed it did. Keys at
    a = -1 beats Lagrange at the 5 MHz D=8 edge on both axes, by
    pre-emphasizing high frequencies — which one frequency cannot see the
    cost of. The claim was wrong; keeping the counterexample under test stops
    it being made again."""
    edge = BAND_EDGES["5MHz_D8"]
    lag_ph, lag_mag = worst_band_edge_error(CANDIDATES["lagrange4"], edge)
    keys_ph, keys_mag = worst_band_edge_error(CANDIDATES["keys4_a100"], edge)

    assert keys_ph < lag_ph
    assert keys_mag < lag_mag


def test_the_ranking_flips_with_the_assumed_pulse_bandwidth():
    """The reason §5 chooses on robustness rather than on a winner, and the
    reason #46 exists: the design states a band edge without a level, and the
    order changes across the plausible range."""
    edge = BAND_EDGES["5MHz_D8"]
    lag = CANDIDATES["lagrange4"]
    keys = CANDIDATES["keys4_a100"]

    assert pulse_weighted_rms_error(keys, edge, rolloff_db=6.0) < pulse_weighted_rms_error(
        lag, edge, rolloff_db=6.0
    )
    assert pulse_weighted_rms_error(lag, edge, rolloff_db=40.0) < pulse_weighted_rms_error(
        keys, edge, rolloff_db=40.0
    )


@pytest.mark.parametrize("case", sorted(BAND_EDGES))
def test_only_the_third_order_accurate_kernels_vanish_as_the_pulse_narrows(case):
    """The property Lagrange is chosen for. Kernels that are not third-order
    accurate carry an irreducible near-DC penalty, so their error stalls
    while Lagrange's keeps falling — which is what makes it safe against a
    pulse bandwidth nobody has written down."""
    edge = BAND_EDGES[case]
    narrow = {
        name: pulse_weighted_rms_error(CANDIDATES[name], edge, rolloff_db=40.0)
        for name in CANDIDATES
    }

    assert narrow["lagrange4"] == min(narrow.values())
    # A factor of two separates the two third-order accurate kernels from
    # everything else, at every decimation case: Catmull-Rom lands at
    # 1.13x, 1.18x, 1.53x, and the nearest kernel that is not third-order
    # accurate at 2.54x, 3.63x, 9.30x.
    assert narrow["keys4_a050"] < 2.0 * narrow["lagrange4"]
    for name in ("linear2", "keys4_a075", "keys4_a100", "hann_sinc4"):
        assert narrow[name] > 2.0 * narrow["lagrange4"], name


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
