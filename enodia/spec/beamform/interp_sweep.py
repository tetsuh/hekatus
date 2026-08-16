"""The sweep behind the fractional-delay kernel choice (design.md §5, #22).

design.md §5 frames the requirement as band-edge phase error against
decimation ratio: with decimation to Nyquist, phase rotation alone leaves
~50° at the band edge, and 4-tap interpolation is required. This module
computes that figure — and the magnitude error beside it, which turns out
to be the larger number — for each candidate 4-tap kernel at each of the
decimation cases the design names, so the choice in `interp.py` is a
number rather than an argument.

The candidates other than the chosen one live here and nowhere else: they
are evidence for a decision, not kernels a port may run.

Run as a script for the table; `tests/test_interp_kernel.py` pins the
figures quoted in design.md so they cannot drift from what the code says.
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

_OFFSETS = np.array([-1, 0, 1, 2])


def _linear2(mu: float) -> np.ndarray:
    """2-tap linear, padded to the 4-tap layout for a like-for-like metric."""
    return np.array([0.0, 1.0 - mu, mu, 0.0])


def _keys4(mu: float, a: float = -0.5) -> np.ndarray:
    """Keys cubic convolution, a = −1/2 (Catmull-Rom)."""

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

    Not a candidate for the specification — it is a table per pass-band
    rather than a closed form, and it trades in-band flatness for edge
    accuracy — but it bounds what four taps can do, which is worth knowing.
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
    "keys4": _keys4,
    "hann_sinc4": _hann_sinc4,
}


def worst_band_edge_error(
    taps: Callable[[float], np.ndarray], band_edge: float, *, fractions: int = 1001
) -> tuple[float, float]:
    """(worst phase error in degrees, worst |magnitude − 1|) at the band edge.

    Worst over the fraction μ ∈ [0, 1]: the interpolator's response at the
    edge frequency, H(ω) = Σ h_k(μ) e^(−jωk), against the ideal delay
    e^(−jωμ). This is the quantity design.md §5 states for the
    no-interpolation case (47° and 54°), extended to a kernel.
    """
    w = 2.0 * np.pi * band_edge
    worst_phase = 0.0
    worst_mag = 0.0
    for mu in np.linspace(0.0, 1.0, fractions):
        h = np.asarray(taps(mu), dtype=np.float64)
        response = np.sum(h * np.exp(-1j * w * _OFFSETS))
        ideal = np.exp(-1j * w * mu)
        worst_phase = max(worst_phase, abs(np.degrees(np.angle(response * np.conj(ideal)))))
        worst_mag = max(worst_mag, abs(abs(response) - 1.0))
    return float(worst_phase), float(worst_mag)


def no_interpolation_worst_phase_deg(band_edge: float) -> float:
    """Phase rotation alone: the fraction is ignored, worst at μ = 1/2."""
    return float(np.degrees(2.0 * np.pi * band_edge * 0.5))


def table() -> str:
    lines = [f"{'kernel':14s}" + "".join(f"{name:>22s}" for name in BAND_EDGES)]
    for name, fn in CANDIDATES.items():
        row = f"{name:14s}"
        for edge in BAND_EDGES.values():
            phase, mag = worst_band_edge_error(fn, edge)
            row += f"{phase:10.2f}° {mag * 100:8.1f}% "
        lines.append(row)
    row = f"{'ls4 (bound)':14s}"
    for edge in BAND_EDGES.values():
        phase, mag = worst_band_edge_error(lambda mu, e=edge: least_squares4(mu, e), edge)
        row += f"{phase:10.2f}° {mag * 100:8.1f}% "
    lines.append(row)
    row = f"{'none':14s}"
    for edge in BAND_EDGES.values():
        row += f"{no_interpolation_worst_phase_deg(edge):10.1f}° {'—':>9s} "
    lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
