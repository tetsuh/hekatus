"""The golden path's own fractional delay, measured against a frozen oracle.

The RF-domain ideal-delay DAS is the yardstick that quantifies the error of
the IQ + 4-tap approximation, so its own interpolation error has to sit well
below what it measures (#25). These tests pin the oracle, the acceptance
limit, every residual quoted in design.md §15, and the fact that the
operator the golden actually runs is under that limit at both carriers.

Two words are kept apart throughout: the *acceptance limit* is the upper
bound the yardstick's residual must stay under; the *floor* is the residual
it actually reaches — below which a comparison still observes differences
but cannot attribute them, since they are indistinguishable from the
yardstick's own error. The name of the first test below is the one #25
mandated and uses "floor" in the first sense; its docstring says which.
"""

import math

import numpy as np
import pytest

from enodia.spec.beamform import rf_delay_sweep as sweep
from enodia.spec.beamform.interp import interpolation_dtype
from enodia.spec.beamform.rf_delay import UPSAMPLE_FACTOR, ZERO_PAD, delay_rf, upsample_rf
from enodia.spec.beamform.rf_delay_sweep import (
    BENCHMARK_CARRIERS_HZ,
    BENCHMARK_N,
    BEST_SEARCHED_4TAP_SUPPORT,
    CANDIDATES,
    CONTIGUOUS_4TAP_SUPPORT,
    RESIDUAL_LIMIT_PCT,
    SEARCHED_4TAP_OFFSETS,
    benchmark_positions,
    benchmark_record,
    least_squares_4tap_bound,
    oracle,
    residual_pct,
)

# --- the acceptance limit, and the operator the golden runs ------------------


@pytest.mark.parametrize("carrier", sorted(BENCHMARK_CARRIERS_HZ))
def test_golden_residual_is_below_declared_floor_at_both_carriers(carrier):
    """The operator `das_rf_golden` runs, scored on the frozen benchmark, sits
    under the acceptance limit — one tenth of the IQ-side error it is used to
    measure. (#25 named this test with "floor" for that limit; the name is
    kept as mandated.) RED against the linear golden read 6.216 % and
    38.707 %."""
    record = benchmark_record(BENCHMARK_CARRIERS_HZ[carrier])

    got = residual_pct(sweep.production, record)

    assert got <= RESIDUAL_LIMIT_PCT[carrier], (
        f"{carrier}: golden residual {got:.3f}% exceeds the acceptance limit "
        f"{RESIDUAL_LIMIT_PCT[carrier]}%"
    )


def test_the_acceptance_limit_is_one_tenth_of_the_iq_error_the_golden_measures():
    """One tenth of §5's pulse-weighted figures as they stood when #25 froze
    the benchmark: 10.82 % at 5 MHz D=8 and 7.91 % at 13 MHz D=2. Frozen with
    the record, and not re-derived when #46 moved §5's figures to 14.00 % and
    7.88 %: the 5 MHz limit is now stricter than a tenth of what is measured
    and the 13 MHz one 0.003 points looser, both far above the production
    residuals. A yardstick contributes negligibly at a tenth."""
    assert RESIDUAL_LIMIT_PCT == {"5MHz": pytest.approx(1.082), "13MHz": pytest.approx(0.791)}
    assert 1.082 < 14.00 / 10
    assert 0.791 == pytest.approx(7.88 / 10, abs=0.0031)


@pytest.mark.parametrize(
    ("carrier", "table", "prose"),
    [("5MHz", 0.000, 0.0003), ("13MHz", 0.099, 0.0992)],
)
def test_the_production_residual_quoted_in_the_design_is_pinned(carrier, table, prose):
    """At both precisions the design uses: the table's three decimals and the
    prose's four, each to half a unit in its last place. This is the floor of
    every comparison made with the golden."""
    got = residual_pct(sweep.production, benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))
    assert got == pytest.approx(table, abs=0.0005)
    assert got == pytest.approx(prose, abs=0.00005)


# --- the frozen preliminary figures ------------------------------------------


@pytest.mark.parametrize(
    ("name", "carrier", "residual"),
    [
        ("linear2", "5MHz", 6.216),
        ("linear2", "13MHz", 38.707),
        ("lagrange4", "5MHz", 0.961),
        ("lagrange4", "13MHz", 28.560),
    ],
)
def test_the_frozen_preliminary_residuals_are_reproduced(name, carrier, residual):
    """The four figures #25 froze before any code existed. #25 allowed 0.01
    point; they reproduce to the third decimal, so that is what is held."""
    got = residual_pct(CANDIDATES[name], benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))
    assert got == pytest.approx(residual, abs=0.0005)


# Every row of the residual table, both carriers, so "every figure quoted is
# pinned" is a statement about this dict and not about memory.
_RESIDUAL_TABLE = {
    "lagrange8": (0.047, 20.586),
    "lagrange16": (0.000, 15.094),
    "kaiser8_sinc16": (0.004, 11.463),
    "kaiser8_sinc32": (0.003, 7.421),
    "rect_sinc256": (0.200, 0.242),
    "poly_up4_kaiser4_hl320_lagrange4": (0.004, 0.361),
}


@pytest.mark.parametrize("name", sorted(_RESIDUAL_TABLE))
def test_every_alternative_in_the_residual_table_is_pinned_at_both_carriers(name):
    for carrier, want in zip(("5MHz", "13MHz"), _RESIDUAL_TABLE[name], strict=True):
        got = residual_pct(CANDIDATES[name], benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))
        assert got == pytest.approx(want, abs=0.0005), f"{name}/{carrier}"


def test_a_finite_kernel_does_reach_the_13mhz_limit_and_its_length_is_the_point():
    """The rectangular 256-tap sinc is finite and its measured residual,
    0.242 %, is under the 0.791 % acceptance limit. So the design's claim is
    not that finite support fails at 13 MHz — it is that no evaluated kernel
    of 32 taps or fewer reaches the limit, and the one that does is 64 times
    the taps of a cubic per sample. Tap count is the stable fact; runtime is
    a measurement of one host and is not asserted. Kept as a test so the
    claim cannot widen again."""
    record = benchmark_record(13e6)

    assert residual_pct(CANDIDATES["rect_sinc256"], record) < RESIDUAL_LIMIT_PCT["13MHz"]
    for short in ("lagrange8", "lagrange16", "kaiser8_sinc16", "kaiser8_sinc32"):
        assert residual_pct(CANDIDATES[short], record) > RESIDUAL_LIMIT_PCT["13MHz"], short


@pytest.mark.parametrize(
    ("support", "carrier", "bound"),
    [
        (CONTIGUOUS_4TAP_SUPPORT, "5MHz", 0.186),
        (CONTIGUOUS_4TAP_SUPPORT, "13MHz", 16.472),
        (BEST_SEARCHED_4TAP_SUPPORT, "5MHz", 0.641),
        (BEST_SEARCHED_4TAP_SUPPORT, "13MHz", 13.041),
    ],
)
def test_the_four_tap_least_squares_bounds_are_pinned_per_support(support, carrier, bound):
    """A least-squares fit of four taps to the oracle bounds every kernel on
    *that support* — and no other. Review found {-2, 0, 1, 3} beats the
    contiguous support at 13 MHz, and a search over all 3060 four-tap
    supports drawn from offsets -8 .. +9 found nothing better than it. Both
    are quoted in the design, so both are pinned."""
    got = residual_pct(
        lambda r, t: least_squares_4tap_bound(r, t, support=support),
        benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]),
    )
    assert got == pytest.approx(bound, abs=0.0005)


def test_no_searched_four_tap_support_comes_within_an_order_of_magnitude_at_13mhz():
    """The claim the design makes, scoped to what was searched: on the
    contiguous support and on the best of the 3060 supports drawn from
    offsets -8 .. +9, four taps miss the 13 MHz limit by more than sixteen
    times. Says nothing about supports outside that range — and the range
    is asserted here so the prose cannot drift from the code again."""
    record = benchmark_record(13e6)
    for support in (CONTIGUOUS_4TAP_SUPPORT, BEST_SEARCHED_4TAP_SUPPORT):
        got = residual_pct(
            lambda r, t, s=support: least_squares_4tap_bound(r, t, support=s), record
        )
        assert got > 16 * RESIDUAL_LIMIT_PCT["13MHz"], support
    assert set(BEST_SEARCHED_4TAP_SUPPORT) <= set(SEARCHED_4TAP_OFFSETS)
    assert list(SEARCHED_4TAP_OFFSETS) == list(range(-8, 10))
    assert len(SEARCHED_4TAP_OFFSETS) == 18
    assert math.comb(len(SEARCHED_4TAP_OFFSETS), 4) == 3060


# --- the oracle itself --------------------------------------------------------


def test_the_oracle_reproduces_the_record_at_integer_positions():
    """sinc(integer) is the Kronecker delta, so an ideal delay by a whole
    number of samples is a shift; this pins the oracle to that meaning."""
    record = benchmark_record(5e6)
    t = np.arange(BENCHMARK_N, dtype=np.float64)

    np.testing.assert_allclose(oracle(record, t), record, atol=1e-12)


def test_the_benchmark_grid_is_the_frozen_one():
    t = benchmark_positions()
    assert t.shape == (201, BENCHMARK_N)
    assert t[0, 5] == 5.0
    assert t[-1, 5] == 4.0
    assert BENCHMARK_N == 256


def test_the_benchmark_record_is_the_simulators_pulse():
    """Same function as the simulator's, at 0.7 — which is the number the
    5 MHz profile happens to carry, but the record is frozen at 0.7 by name
    (`rf-oracle-frozen-0p7`), not derived from any profile (#46). A profile's
    own result is a separate reconciliation output."""
    from enodia.spec.sim import gaussian_pulse

    assert sweep.BENCHMARK_NAME == "rf-oracle-frozen-0p7"
    assert sweep.BENCHMARK_BANDWIDTH_FRAC == 0.7
    record = benchmark_record(13e6)
    n = np.arange(256)
    np.testing.assert_array_equal(record, gaussian_pulse((n - 128) / 40e6, 13e6, 0.7))


# --- the operator's contract --------------------------------------------------


def test_upsampling_reproduces_every_original_sample_on_the_fine_grid():
    rng = np.random.default_rng(1)
    record = rng.standard_normal((3, 64))

    fine = upsample_rf(record)

    assert fine.shape == (3, (64 + 2 * ZERO_PAD) * UPSAMPLE_FACTOR)
    np.testing.assert_allclose(
        fine[:, ZERO_PAD * UPSAMPLE_FACTOR :: UPSAMPLE_FACTOR][:, :64], record, atol=1e-10
    )


def test_upsampling_is_band_limited_to_the_original_nyquist():
    """Nothing appears above fs/2 of the coarse grid — the fine grid carries
    the same signal, not a new one."""
    rng = np.random.default_rng(2)
    record = rng.standard_normal((1, 128))

    fine = upsample_rf(record)
    spectrum = np.abs(np.fft.rfft(fine[0]))
    coarse_nyquist_bin = fine.shape[-1] // (2 * UPSAMPLE_FACTOR)

    assert spectrum[coarse_nyquist_bin + 1 :].max() < 1e-9 * spectrum.max()


def test_delay_rf_reads_the_sample_itself_at_integer_positions():
    record = np.arange(40, dtype=np.float64)[None, :] * 3.0
    positions = np.array([[0.0, 7.0, 39.0]])

    np.testing.assert_allclose(delay_rf(record, positions), [[0.0, 21.0, 117.0]], atol=1e-9)


def test_delay_rf_indexes_row_i_of_positions_into_row_i_of_the_record():
    """Row i of the positions reads row i of the record — and what it reads
    is the ideal delay of that row's finite record, not of some smooth
    function it happens to sample. A truncated ramp rings; the oracle says
    exactly how much, and the operator agrees with it."""
    record = np.stack([np.arange(64.0), 10.0 * np.arange(64.0)])
    positions = np.array([[2.5, 30.25], [2.5, 30.25]])

    got = delay_rf(record, positions)

    assert got.shape == (2, 2)
    # A truncated ramp is as far from band-limited as a record gets — a
    # full-scale step at each end — so the periodic images the padding holds
    # off are at their strongest here. Half a percent is that stress case,
    # not the operating regime; the frozen benchmark pins the real figure.
    for i in range(2):
        np.testing.assert_allclose(got[i], oracle(record[i], positions[i]), rtol=5e-3)
    np.testing.assert_allclose(got[1], 10.0 * got[0], rtol=1e-9)


def test_delay_rf_near_the_ends_rings_like_the_ideal_delay_rather_than_clamping():
    """The record is zero outside [0, N); its band-limited reconstruction is
    not, just past the ends. The old linear golden clamped and extrapolated
    there. The new one reads what the oracle reads."""
    record = benchmark_record(13e6)
    t = np.array([[-0.5, -1.5, 256.5]])

    got = delay_rf(record[None, :], t)[0]
    ideal = oracle(record, t[0])

    np.testing.assert_allclose(got, ideal, atol=2e-3 * np.abs(record).max())
    # And well beyond the padding there is nothing to read.
    far = delay_rf(record[None, :], np.array([[-ZERO_PAD - 10.0, 256.0 + ZERO_PAD + 10.0]]))
    np.testing.assert_array_equal(far, [[0.0, 0.0]])


@pytest.mark.parametrize(
    ("record_dtype", "expected"),
    [(np.int16, np.float32), (np.float32, np.float32), (np.float64, np.float64)],
)
def test_delay_rf_returns_the_dataflows_intermediate_dtype(record_dtype, expected):
    """design.md §14: int16 in, FP32 intermediate; a float64 golden stays
    float64. Same rule `interp.py` follows for the IQ side."""
    record = (np.arange(64) * 7).astype(record_dtype)[None, :]
    positions = np.array([[30.25]])

    got = delay_rf(record, positions)

    assert got.dtype == expected == interpolation_dtype(record_dtype)
    np.testing.assert_allclose(
        got[0], oracle(record[0].astype(np.float64), positions[0]), rtol=5e-3
    )


def test_upsampling_by_one_returns_the_record_intact():
    """Review found the Nyquist-bin split applied at factor 1 too, where there
    is nothing to split; that would have halved the record's Nyquist
    component. Factor 1 is now the identity, padded."""
    rng = np.random.default_rng(3)
    record = rng.standard_normal((2, 30))  # even length: the Nyquist bin exists

    fine = upsample_rf(record, factor=1)

    np.testing.assert_allclose(fine[:, ZERO_PAD : ZERO_PAD + 30], record, atol=1e-12)
    with pytest.raises(ValueError):
        upsample_rf(record, factor=0)


def test_the_upsampling_parameters_are_the_ones_the_design_names():
    assert UPSAMPLE_FACTOR == 8
    assert ZERO_PAD == 256


def test_padding_is_what_holds_the_13mhz_residual_under_the_limit():
    """Periodic-sinc interpolation has images one padded length away; with no
    padding they reach the record and the residual is 0.97 %, over the
    acceptance limit. Pinned so the constant is understood as load-bearing,
    not cosmetic."""
    from enodia.spec.beamform.interp import fractional_delay

    record = benchmark_record(13e6)

    def unpadded(rec, t):
        rows = np.broadcast_to(rec, (t.shape[0], rec.size))
        return fractional_delay(upsample_rf(rows, pad=0), t * UPSAMPLE_FACTOR)

    got = residual_pct(unpadded, record)
    assert got == pytest.approx(0.972, abs=0.0005)
    assert got > RESIDUAL_LIMIT_PCT["13MHz"]
    # The same figure sits under the separate 5 MHz limit, which is why the
    # design names the carrier when it quotes it.
    assert got < RESIDUAL_LIMIT_PCT["5MHz"]
