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
    CASES,
    CASES_BY_NAME,
    PULSE_ROLLOFFS_DB,
    SYNTHETIC_13MHZ_ENVELOPE,
    least_squares4,
    profile_case,
    pulse_weighted_rms_error,
    worst_band_edge_error,
)
from enodia.spec.probe import BANDWIDTH_LEVEL_DB, linear_5mhz


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
    # Zero at both, for different reasons. At t = 9 with N = 8 every tap is
    # outside. At t = -2 sample 0 is still in range, but mu = 0 puts all the
    # weight on the tap for z[m], which is outside — so the rule bites
    # through the tap weight rather than through the index range.
    np.testing.assert_allclose(fractional_delay(z, np.array([-2.0, 9.0])), [0.0, 0.0])
    # Just past it, the last position that still reads something.
    assert fractional_delay(z, np.array([-1.999]))[0] != 0.0


@pytest.mark.parametrize(
    ("record_dtype", "expected"),
    [
        (np.int16, np.float32),
        (np.int32, np.float64),
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.complex64, np.complex64),
        (np.complex128, np.complex128),
    ],
)
def test_a_record_uses_the_contract_minimum_interpolation_dtype(record_dtype, expected):
    """design.md §14: the IQ record is int16 complex and interpolation runs at
    an FP32-or-wider intermediate. Integer arithmetic truncates every tap at
    t = 2.5 to zero, while float16 is narrower than the contract minimum."""
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
    """design.md §5 names three: 5 MHz at D=8 and D=4, 13 MHz at D=2. The
    5 MHz edge is the profile's (0.7 × 5 MHz / 2 = 1.75 MHz); the 13 MHz one
    is §4's 80% envelope (5.2 MHz)."""
    assert set(BAND_EDGES) == {"5MHz_D8", "13MHz_D2", "5MHz_D4"}
    assert BAND_EDGES["5MHz_D8"] == pytest.approx(0.35)
    assert BAND_EDGES["13MHz_D2"] == pytest.approx(0.26)
    assert BAND_EDGES["5MHz_D4"] == pytest.approx(0.175)


def test_every_case_names_the_pulse_it_assumes():
    """#46: a figure cannot be quoted without its identity, status, source,
    spectral level and width convention. The 5 MHz cases are the
    `linear-5mhz` profile, provisional with no source; the 13 MHz case is a
    synthetic design envelope and claims no profile or physical authority —
    and is not the frozen RF oracle of §15, which is a different 13 MHz
    record."""
    profile = linear_5mhz()
    for name, decimation in (("5MHz_D8", 8), ("5MHz_D4", 4)):
        case = CASES_BY_NAME[name]
        assert case == profile_case(profile, decimation)
        assert case.identity == "linear-5mhz"
        assert case.status == "provisional"
        assert case.source is None
        assert case.bandwidth_frac == 0.7
        assert case.edge_hz == pytest.approx(1.75e6)
        assert case.band_edge == pytest.approx(profile.bandwidth_edge_hz / (40e6 / decimation))
    envelope = CASES_BY_NAME["13MHz_D2"]
    assert envelope.identity == SYNTHETIC_13MHZ_ENVELOPE == "synthetic-80pct-design-envelope"
    assert envelope.status == "synthetic"
    assert envelope.source is None
    assert envelope.bandwidth_frac == 0.8
    assert envelope.edge_hz == pytest.approx(5.2e6)
    assert envelope.identity != "rf-oracle-frozen-0p7"
    for case in CASES:
        assert case.level_db == BANDWIDTH_LEVEL_DB
        line = case.describe()
        for field in (
            case.name,
            case.identity,
            case.status,
            "source: none",
            "6.0206 dB",
            "one-sided",
        ):
            assert field in line
    # The first weighted-table column is the case's own model: half amplitude
    # at the edge, by definition.
    assert PULSE_ROLLOFFS_DB[0] == BANDWIDTH_LEVEL_DB


# Every figure design.md §5 quotes, pinned so the record cannot drift from
# what the code computes. Sol found the first table quoted values no test
# held (#45), which is how a table stops matching its own sweep.
_EDGE_TABLE = {
    "5MHz_D8": {
        "linear2": (13.10, 0.546),
        "lagrange4": (8.25, 0.366),
        "keys4_a050": (13.10, 0.366),
        "keys4_a075": (9.04, 0.276),
        "keys4_a100": (5.40, 0.186),
        "hann_sinc4": (16.55, 0.459),
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
        "linear2": (1.30, 0.147),
        "lagrange4": (0.29, 0.031),
        "keys4_a050": (1.30, 0.031),
        "keys4_a075": (1.39, 0.027),
        "keys4_a100": (3.93, 0.085),
        "hann_sinc4": (1.95, 0.091),
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
        ("5MHz_D8", (4.09, 0.184)),
        ("13MHz_D2", (0.73, 0.047)),
        ("5MHz_D4", (0.09, 0.009)),
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
        "linear2": (19.98, 7.82, 4.02),
        "lagrange4": (14.00, 3.27, 0.97),
        "keys4_a050": (14.03, 3.38, 1.06),
        "keys4_a075": (12.02, 2.59, 1.54),
        "keys4_a100": (11.44, 4.55, 3.58),
        "hann_sinc4": (16.36, 5.20, 2.29),
        "ls4_bound": (10.87, 3.15, 2.80),
    },
    "13MHz_D2": {
        "linear2": (13.38, 4.42, 2.25),
        "lagrange4": (7.88, 1.16, 0.32),
        "keys4_a050": (7.97, 1.26, 0.38),
        "keys4_a075": (6.35, 1.60, 1.26),
        "keys4_a100": (6.85, 3.71, 2.77),
        "hann_sinc4": (10.13, 2.57, 1.16),
        "ls4_bound": (6.22, 0.92, 0.83),
    },
    "5MHz_D4": {
        "linear2": (6.55, 2.04, 1.03),
        "lagrange4": (2.39, 0.27, 0.07),
        "keys4_a050": (2.50, 0.32, 0.10),
        "keys4_a075": (2.06, 1.21, 0.89),
        "keys4_a100": (4.25, 2.64, 1.85),
        "hann_sinc4": (4.17, 1.04, 0.49),
        "ls4_bound": (1.81, 0.18, 0.17),
    },
}


def _candidate(name, edge):
    if name == "ls4_bound":
        return lambda mu: least_squares4(mu, edge)
    return CANDIDATES[name]


@pytest.mark.parametrize("case", sorted(_WEIGHTED_TABLE))
def test_every_pulse_weighted_figure_quoted_in_the_design_is_pinned(case):
    """The first column is the case's own pulse model — half amplitude at the
    edge, exactly −6.0206 dB, not a 6 dB rounding (#46); the other two are
    narrower pulses, kept as the sensitivity sweep."""
    edge = BAND_EDGES[case]
    for name, expected in _WEIGHTED_TABLE[case].items():
        for db, want in zip(PULSE_ROLLOFFS_DB, expected, strict=True):
            got = 100.0 * pulse_weighted_rms_error(_candidate(name, edge), edge, rolloff_db=db)
            assert got == pytest.approx(want, abs=0.01), f"{case}/{name}/-{db:g}dB"


def test_the_pulse_metric_stops_at_the_decimated_nyquist_frequency():
    """The record cannot represent anything above 0.5 cycles/sample, so
    scoring the kernel's response there measures a signal that is not in it.
    The profile's own pulse reaches past it at D=8 — its 3σ is 0.89 at the
    0.35 fs' edge — and integrating that far changed the figures, though not
    the order. A wider integration cannot lower the error, so clipping must
    not raise it."""
    edge = BAND_EDGES["5MHz_D8"]
    sigma = edge / np.sqrt(2.0 * np.log(10.0 ** (BANDWIDTH_LEVEL_DB / 20.0)))
    assert 3.0 * sigma == pytest.approx(0.89, abs=0.005)
    assert 3.0 * sigma > 0.5

    clipped = pulse_weighted_rms_error(CANDIDATES["lagrange4"], edge, rolloff_db=BANDWIDTH_LEVEL_DB)
    assert clipped == pytest.approx(0.1400, abs=0.0001)


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
    """The reason §5 chooses on robustness rather than on a winner. #46 fixed
    the level the edge sits at, but the 5 MHz width itself is provisional, so
    the order across narrower pulses still matters: Keys a=−1 wins on the
    profile's own pulse and loses by a factor of several on a narrow one."""
    edge = BAND_EDGES["5MHz_D8"]
    lag = CANDIDATES["lagrange4"]
    keys = CANDIDATES["keys4_a100"]
    model = BANDWIDTH_LEVEL_DB

    assert pulse_weighted_rms_error(keys, edge, rolloff_db=model) < pulse_weighted_rms_error(
        lag, edge, rolloff_db=model
    )
    assert pulse_weighted_rms_error(lag, edge, rolloff_db=40.0) < pulse_weighted_rms_error(
        keys, edge, rolloff_db=40.0
    )


def test_lagrange_leads_five_of_the_nine_cells_and_never_falls_below_third():
    """The claim §5 makes, pinned. At the 1.5 MHz edge it read "six of nine";
    at the profile-derived 1.75 MHz edge (#46) Lagrange is first in five,
    second in two and third in two — the widest-pulse cells, where the Keys
    kernels lead by 14–22%. An earlier version said "never worse than second",
    which was false then too; the trade is stated, not a ranking."""
    ranks = []
    for edge in BAND_EDGES.values():
        for db in PULSE_ROLLOFFS_DB:
            scores = {
                name: pulse_weighted_rms_error(CANDIDATES[name], edge, rolloff_db=db)
                for name in CANDIDATES
            }
            ranks.append(sorted(scores, key=scores.get).index("lagrange4") + 1)

    assert ranks.count(1) == 5
    assert ranks.count(2) == 2
    assert max(ranks) == 3


@pytest.mark.parametrize(
    ("rolloff_db", "factor"), [(BANDWIDTH_LEVEL_DB, 5.86), (20.0, 12.32), (40.0, 13.97)]
)
def test_the_decimation_gain_quoted_in_the_design_is_pinned(rolloff_db, factor):
    """§5 quotes what D=4 buys over D=8 for the specified kernel, and says
    every figure in the section is held by this file. These arrived with a
    review fix and were the one set that was not — the same gap, one round
    later, as the one that produced them."""
    lag = CANDIDATES["lagrange4"]
    heavy = pulse_weighted_rms_error(lag, BAND_EDGES["5MHz_D8"], rolloff_db=rolloff_db)
    light = pulse_weighted_rms_error(lag, BAND_EDGES["5MHz_D4"], rolloff_db=rolloff_db)

    assert heavy / light == pytest.approx(factor, abs=0.01)


@pytest.mark.parametrize("case", sorted(BAND_EDGES))
def test_lagrange_decays_faster_than_every_other_candidate(case):
    """The claim §5 rests on, now that "stalls" has been withdrawn: every
    normalized kernel converges — unit DC gain guarantees it — so what
    separates them is the rate. Lagrange is exact to degree 3, so its error
    falls with a higher power of the bandwidth than a second-order accurate
    kernel's."""
    edge = BAND_EDGES[case]
    rates = {
        name: (
            pulse_weighted_rms_error(CANDIDATES[name], edge, rolloff_db=40.0)
            / pulse_weighted_rms_error(CANDIDATES[name], edge, rolloff_db=80.0)
        )
        for name in CANDIDATES
    }

    assert rates["lagrange4"] == max(rates.values())
    assert rates["lagrange4"] > 3.0
    assert rates["keys4_a100"] < 1.5


@pytest.mark.parametrize("case", sorted(BAND_EDGES))
def test_candidates_decay_toward_dc_and_lagrange_has_lowest_narrow_pulse_error(case):
    """All normalized candidates improve toward DC, while Lagrange decays
    faster and is lowest in the measured narrow-pulse cases."""
    edge = BAND_EDGES[case]
    broad = {
        name: pulse_weighted_rms_error(CANDIDATES[name], edge, rolloff_db=40.0)
        for name in CANDIDATES
    }
    narrower = {
        name: pulse_weighted_rms_error(CANDIDATES[name], edge, rolloff_db=80.0)
        for name in CANDIDATES
    }

    for name in CANDIDATES:
        assert narrower[name] < broad[name], name
    assert narrower["lagrange4"] == min(narrower.values())
    # At -40 dB the other third-order accurate kernel (Keys a=-1/2) stays
    # within 1.5x of Lagrange in every case, and every other candidate is
    # more than 1.5x away. It used to read "a factor of two"; at the
    # profile-derived 5 MHz D=8 edge Keys a=-3/4 is 1.59x, so two was no
    # longer true (#46).
    assert broad["keys4_a050"] < 1.5 * broad["lagrange4"]
    for name in ("linear2", "keys4_a075", "keys4_a100", "hann_sinc4"):
        assert broad[name] > 1.5 * broad["lagrange4"], name


@pytest.mark.parametrize(
    ("case", "degrees"), [("5MHz_D8", 63.0), ("13MHz_D2", 46.8), ("5MHz_D4", 31.5)]
)
def test_phase_rotation_alone_is_the_error_the_design_states(case, degrees):
    """The 47° and 63° in design.md §5 are the no-interpolation case at the
    worst fraction; the sweep's metric reduces to them, which pins the metric
    to the design's framing. The 31.5° at D=4 is quoted in the same table.
    (Before #46 the 5 MHz figures were 54° and 27°, from a 1.5 MHz edge the
    profile does not have.)"""
    assert interp_sweep_no_interp_worst(BAND_EDGES[case]) == pytest.approx(degrees, abs=0.1)


def interp_sweep_no_interp_worst(edge: float) -> float:
    from enodia.spec.beamform.interp_sweep import no_interpolation_worst_phase_deg

    return no_interpolation_worst_phase_deg(edge)


def test_the_module_docstring_names_the_issue_that_pinned_it():
    assert "#22" in interp.__doc__


def test_profile_derived_5mhz_sweep_is_pinned():
    """#46: the 5 MHz cases take their band edge from the `linear-5mhz`
    profile — 0.7 × 5 MHz / 2 = 1.75 MHz, so 0.35 fs' at D=8 and 0.175 fs'
    at D=4 — and the pulse model is that profile's: a Gaussian exactly half
    amplitude (−6.0206 dB) at the edge. The Lagrange figures the issue froze
    as acceptance targets are reproduced here to 0.01° and 0.01 point."""
    profile = linear_5mhz()
    level_db = 20.0 * np.log10(2.0)
    assert BANDWIDTH_LEVEL_DB == pytest.approx(level_db, abs=1e-12)
    expected = {
        "5MHz_D8": (8, 8.25, 0.3658, 14.00),
        "5MHz_D4": (4, 0.29, 0.0310, 2.39),
    }
    for name, (decimation, phase, mag, weighted) in expected.items():
        edge = BAND_EDGES[name]
        assert edge == pytest.approx(
            profile.bandwidth_frac * profile.f0_hz / 2 / (40e6 / decimation)
        )
        assert CASES_BY_NAME[name].identity == profile.name
        assert CASES_BY_NAME[name].status == "provisional"
        got_phase, got_mag = worst_band_edge_error(CANDIDATES["lagrange4"], edge)
        assert got_phase == pytest.approx(phase, abs=0.01), name
        assert got_mag == pytest.approx(mag, abs=0.0001), name
        got = 100.0 * pulse_weighted_rms_error(CANDIDATES["lagrange4"], edge, rolloff_db=level_db)
        assert got == pytest.approx(weighted, abs=0.01), name
