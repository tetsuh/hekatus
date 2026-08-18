"""The golden path's own fractional delay, measured against a frozen oracle.

The RF-domain ideal-delay DAS is the yardstick that quantifies the error of
the IQ + 4-tap approximation, so its own interpolation error has to sit well
below what it measures (#25). These tests pin the oracle, the floor, every
figure quoted in design.md §15, and the fact that the operator the golden
actually runs is under that floor at both carriers.
"""

import numpy as np
import pytest

from enodia.spec.beamform import rf_delay_sweep as sweep
from enodia.spec.beamform.interp import interpolation_dtype
from enodia.spec.beamform.rf_delay import UPSAMPLE_FACTOR, ZERO_PAD, delay_rf, upsample_rf
from enodia.spec.beamform.rf_delay_sweep import (
    BENCHMARK_CARRIERS_HZ,
    BENCHMARK_N,
    CANDIDATES,
    RESIDUAL_LIMIT_PCT,
    benchmark_positions,
    benchmark_record,
    least_squares_4tap_bound,
    oracle,
    residual_pct,
)

# --- the floor, and the operator the golden runs ------------------------------


@pytest.mark.parametrize("carrier", sorted(BENCHMARK_CARRIERS_HZ))
def test_golden_residual_is_below_declared_floor_at_both_carriers(carrier):
    """The operator `das_rf_golden` runs, scored on the frozen benchmark, sits
    under one tenth of the IQ-side error it is used to measure. RED against
    the linear golden read 6.216 % and 38.707 %."""
    record = benchmark_record(BENCHMARK_CARRIERS_HZ[carrier])

    got = residual_pct(sweep.production, record)

    assert got <= RESIDUAL_LIMIT_PCT[carrier], (
        f"{carrier}: golden residual {got:.3f}% exceeds floor {RESIDUAL_LIMIT_PCT[carrier]}%"
    )


def test_the_floor_is_one_tenth_of_the_iq_error_the_golden_measures():
    """design.md §5's −6 dB pulse-weighted figures: 10.82 % at 5 MHz D=8 and
    7.91 % at 13 MHz D=2. A yardstick contributes negligibly at a tenth."""
    assert RESIDUAL_LIMIT_PCT == {"5MHz": pytest.approx(1.082), "13MHz": pytest.approx(0.791)}


@pytest.mark.parametrize(("carrier", "residual"), [("5MHz", 0.000), ("13MHz", 0.099)])
def test_the_production_residual_quoted_in_the_design_is_pinned(carrier, residual):
    got = residual_pct(sweep.production, benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))
    assert got == pytest.approx(residual, abs=0.01)


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
    """The four figures #25 froze before any code existed, to 0.01 point."""
    got = residual_pct(CANDIDATES[name], benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))
    assert got == pytest.approx(residual, abs=0.01)


@pytest.mark.parametrize(
    ("name", "carrier", "residual"),
    [
        ("lagrange16", "13MHz", 15.094),
        ("kaiser8_sinc32", "13MHz", 7.421),
        ("rect_sinc256", "5MHz", 0.200),
        ("rect_sinc256", "13MHz", 0.242),
        ("poly_up4_kaiser4_hl320_lagrange4", "5MHz", 0.004),
        ("poly_up4_kaiser4_hl320_lagrange4", "13MHz", 0.361),
    ],
)
def test_the_alternatives_quoted_in_the_design_are_pinned(name, carrier, residual):
    got = residual_pct(CANDIDATES[name], benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))
    assert got == pytest.approx(residual, abs=0.01)


@pytest.mark.parametrize(("carrier", "bound"), [("5MHz", 0.186), ("13MHz", 16.472)])
def test_no_four_tap_rf_kernel_can_meet_the_13mhz_floor(carrier, bound):
    """The least-squares fit of four taps to the oracle on the record itself
    bounds every four-tap zero-extended kernel. At 13 MHz it misses the floor
    by a factor of twenty, so the design's move to upsampling is forced by
    the family, not by the choice of member."""
    got = residual_pct(least_squares_4tap_bound, benchmark_record(BENCHMARK_CARRIERS_HZ[carrier]))

    assert got == pytest.approx(bound, abs=0.01)
    if carrier == "13MHz":
        assert got > 20 * RESIDUAL_LIMIT_PCT[carrier]


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
    assert t[0, 5] == 5.0 and t[-1, 5] == 4.0
    assert BENCHMARK_N == 256


def test_the_benchmark_record_is_the_simulators_pulse():
    """Same function, same bandwidth the profile carries: what #6 consumes."""
    from enodia.spec.sim import gaussian_pulse

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


def test_the_upsampling_parameters_are_the_ones_the_design_names():
    assert UPSAMPLE_FACTOR == 8
    assert ZERO_PAD == 256


def test_padding_is_what_holds_the_13mhz_residual_under_the_floor():
    """Periodic-sinc interpolation has images one padded length away; with no
    padding they reach the record and the residual is 0.97 %, over the floor.
    Pinned so the constant is understood as load-bearing, not cosmetic."""
    from enodia.spec.beamform.interp import fractional_delay

    record = benchmark_record(13e6)

    def unpadded(rec, t):
        rows = np.broadcast_to(rec, (t.shape[0], rec.size))
        return fractional_delay(upsample_rf(rows, pad=0), t * UPSAMPLE_FACTOR)

    assert residual_pct(unpadded, record) == pytest.approx(0.972, abs=0.01)
    assert residual_pct(unpadded, record) > RESIDUAL_LIMIT_PCT["13MHz"]
