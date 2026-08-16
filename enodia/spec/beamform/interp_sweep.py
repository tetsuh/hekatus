"""The sweep behind the fractional-delay kernel choice (design.md §5, #22).

design.md §5 frames the requirement as band-edge phase error against
decimation ratio: with decimation to Nyquist, phase rotation alone leaves
~50° at the band edge, and 4-tap interpolation is required. This module
computes that figure for each candidate 4-tap kernel at each of the
decimation cases the design names.

**The band-edge metric alone is not enough to choose by**, and review of
#22 is what showed it: a kernel can buy band-edge accuracy by
pre-emphasizing high frequencies, paying for it across the rest of the band
where the signal actually lives. Keys' family does exactly that as its
parameter goes below −1/2, and beats Lagrange at the edge while being worse
almost everywhere else. So a second metric is computed here — the error the
beamformer actually sees on a pulse — and, because the answer depends on how
wide that pulse is, at several assumed bandwidths. What the two metrics
together say is recorded in design.md §5.

The candidates other than the chosen one live here and nowhere else: they
are evidence for a decision, not kernels a port may run.

Run as a script for the tables; `tests/test_interp_kernel.py` pins every
figure quoted in design.md so the record cannot drift from what the code
computes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from enodia.spec.beamform.interp import fractional_delay_taps

# Band edge as a fraction of the decimated sampling rate fs', from design.md
# §5: 5 MHz probe, band edge 1.5 MHz, fs' = 40/D MHz; 13 MHz probe, band edge
# 5.2 MHz, fs' = 20 MHz at D=2.
BAND_EDGES: dict[str, float] = {
    "5MHz_D8": 1.5 / 5.0,
    "13MHz_D2": 5.2 / 20.0,
    "5MHz_D4": 1.5 / 10.0,
}

# How far down the pulse's amplitude spectrum is at that band edge. design.md
# does not say, and the kernel ranking depends on it, so the sweep reports a
# range instead of picking one (#46).
PULSE_ROLLOFFS_DB: tuple[float, ...] = (6.0, 20.0, 40.0)

_OFFSETS = np.array([-1, 0, 1, 2])


def _linear2(mu: float) -> np.ndarray:
    """2-tap linear, padded to the 4-tap layout for a like-for-like metric."""
    return np.array([0.0, 1.0 - mu, mu, 0.0])


def _keys4(mu: float, a: float = -0.5) -> np.ndarray:
    """Keys cubic convolution. a = −1/2 is Catmull-Rom, the only third-order
    accurate member; a < −1/2 pre-emphasizes high frequencies."""

    def k(x: float) -> float:
        x = abs(x)
        if x < 1.0:
            return (a + 2.0) * x**3 - (a + 3.0) * x**2 + 1.0
        if x < 2.0:
            return a * x**3 - 5.0 * a * x**2 + 8.0 * a * x - 4.0 * a
        return 0.0

    return np.array([k(1.0 + mu), k(mu), k(1.0 - mu), k(2.0 - mu)])


def _hann_sinc4(mu: float) -> np.ndarray:
    """sinc truncated to four taps under a Hann window, renormalized to unit DC."""
    d = _OFFSETS - mu
    h = np.sinc(d) * 0.5 * (1.0 + np.cos(np.pi * d / 2.0))
    return h / h.sum()


def least_squares4(mu: float, band_edge: float) -> np.ndarray:
    """Real 4-tap FIR minimizing squared complex-response error over |f| ≤ edge.

    Not a candidate for the specification — a table per pass-band is not
    something two implementations can check against each other from four
    lines of algebra — but it bounds what four taps can do, which is worth
    knowing.
    """
    w = np.linspace(-2.0 * np.pi * band_edge, 2.0 * np.pi * band_edge, 401)
    a = np.exp(-1j * np.outer(w, _OFFSETS))
    b = np.exp(-1j * w * mu)
    system = np.vstack([a.real, a.imag])
    rhs = np.concatenate([b.real, b.imag])
    h, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return h


CANDIDATES: dict[str, Callable[[float], np.ndarray]] = {
    "linear2": _linear2,
    "lagrange4": lambda mu: fractional_delay_taps(mu),
    "keys4_a050": lambda mu: _keys4(mu, -0.50),
    "keys4_a075": lambda mu: _keys4(mu, -0.75),
    "keys4_a100": lambda mu: _keys4(mu, -1.00),
    "hann_sinc4": _hann_sinc4,
}


def _response(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.exp(-1j * np.outer(w, _OFFSETS)) @ np.asarray(h, dtype=np.float64)


def worst_band_edge_error(
    taps: Callable[[float], np.ndarray], band_edge: float, *, fractions: int = 1001
) -> tuple[float, float]:
    """(worst phase error in degrees, worst |magnitude − 1|) at the band edge.

    Worst over the fraction μ ∈ [0, 1]: the interpolator's response at the
    edge frequency, H(ω) = Σ h_k(μ) e^(−jωk), against the ideal delay
    e^(−jωμ). This is the quantity design.md §5 states for the
    no-interpolation case (47° and 54°), extended to a kernel.

    **One frequency only.** A kernel that trades in-band accuracy for edge
    accuracy scores well here and badly in use; that is what
    `pulse_weighted_rms_error` is for.
    """
    w = np.array([2.0 * np.pi * band_edge])
    worst_phase = 0.0
    worst_mag = 0.0
    for mu in np.linspace(0.0, 1.0, fractions):
        response = _response(taps(mu), w)[0]
        ideal = np.exp(-1j * w[0] * mu)
        worst_phase = max(worst_phase, abs(np.degrees(np.angle(response * np.conj(ideal)))))
        worst_mag = max(worst_mag, abs(abs(response) - 1.0))
    return float(worst_phase), float(worst_mag)


def pulse_weighted_rms_error(
    taps: Callable[[float], np.ndarray],
    band_edge: float,
    *,
    rolloff_db: float = 6.0,
    fractions: int = 201,
    n_freq: int = 1201,
) -> float:
    """RMS error over the pulse spectrum, averaged over the fraction.

    The beamformer sees the whole pulse, not its edge. Modelling the
    baseband pulse as a Gaussian whose amplitude is ``rolloff_db`` down at
    ``band_edge``, this returns

        sqrt( E_μ[ ∫S(f)·|H(f) − e^(−j2πfμ)|² df / ∫S(f) df ] )

    which is the normalized error energy the delayed channel signal carries,
    integrated over the band the decimated record can represent.
    ``rolloff_db`` is an assumption, not a measurement — design.md gives a
    band edge without saying at what level (#46) — so a caller sweeps it.
    """
    sigma = band_edge / np.sqrt(2.0 * np.log(10.0 ** (rolloff_db / 20.0)))
    # Stop at the decimated Nyquist frequency. The record cannot represent
    # anything above 0.5 cycles/sample, so scoring the kernel's response
    # there measures a signal that does not exist in it. Worth noticing that
    # the widest assumption reaches past it — 3σ is 0.77 at the 5 MHz D=8
    # edge — which says that assumption and that decimation ratio are in
    # tension, and is one more thing #46 has to settle.
    f_max = min(3.0 * sigma, 0.5)
    f = np.linspace(-f_max, f_max, n_freq)
    spectrum = np.exp(-(f**2) / (2.0 * sigma**2)) ** 2
    w = 2.0 * np.pi * f
    mus = np.linspace(0.0, 1.0, fractions)
    # The basis is the same for every fraction; building it inside the loop
    # made this the slowest thing in the test suite by two orders of magnitude.
    basis = np.exp(-1j * np.outer(w, _OFFSETS))
    weights = np.stack([np.asarray(taps(mu), dtype=np.float64) for mu in mus])
    response = weights @ basis.T  # (mu, f)
    error = np.abs(response - np.exp(-1j * np.outer(mus, w))) ** 2
    per_mu = np.trapezoid(spectrum * error, f, axis=-1) / np.trapezoid(spectrum, f)
    return float(np.sqrt(per_mu.mean()))


def no_interpolation_worst_phase_deg(band_edge: float) -> float:
    """Phase rotation alone: the fraction is ignored, worst at μ = 1/2."""
    return float(np.degrees(2.0 * np.pi * band_edge * 0.5))


def _all_candidates(band_edge: float) -> dict[str, Callable[[float], np.ndarray]]:
    return {**CANDIDATES, "ls4_bound": lambda mu: least_squares4(mu, band_edge)}


def edge_table() -> str:
    lines = [f"{'kernel':12s}" + "".join(f"{name:>22s}" for name in BAND_EDGES)]
    for name in _all_candidates(0.0):
        row = f"{name:12s}"
        for edge in BAND_EDGES.values():
            phase, mag = worst_band_edge_error(_all_candidates(edge)[name], edge)
            row += f"{phase:10.2f}° {mag * 100:8.1f}% "
        lines.append(row)
    row = f"{'none':12s}"
    for edge in BAND_EDGES.values():
        row += f"{no_interpolation_worst_phase_deg(edge):10.1f}° {'—':>9s} "
    lines.append(row)
    return "\n".join(lines)


def weighted_table() -> str:
    lines = []
    for label, edge in BAND_EDGES.items():
        lines.append(f"-- {label} (edge {edge:.2f} fs') --")
        lines.append(
            f"   {'kernel':12s}" + "".join(f"{f'-{db:g}dB':>10s}" for db in PULSE_ROLLOFFS_DB)
        )
        for name, fn in _all_candidates(edge).items():
            row = f"   {name:12s}"
            for db in PULSE_ROLLOFFS_DB:
                row += f"{pulse_weighted_rms_error(fn, edge, rolloff_db=db) * 100:9.2f}%"
            lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Worst-case error at the band edge (phase / magnitude):")
    print(edge_table())
    print()
    print("RMS error over a pulse whose amplitude is N dB down at that edge:")
    print(weighted_table())
