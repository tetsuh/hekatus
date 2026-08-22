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

**Every case names the pulse it assumes** (#46). The 5 MHz cases take their
band edge from the `linear-5mhz` profile — `bandwidth_frac · f0 / 2`, the
one-sided edge of the full half-amplitude width design.md §4 defines — and
carry that profile's provenance status, which is *provisional* until a
sourced value replaces it. The 13 MHz case is not a profile: no 13 MHz
`ProbeProfile` exists (#10 creates it), so it is the §4 design envelope of
80% fractional bandwidth under the same convention, labelled
`synthetic-80pct-design-envelope` and claiming no physical authority. Each
emitted figure carries its case's identity, spectral level, width
convention, status and source, so a number cannot be quoted without what it
assumed.

The candidates other than the chosen one live here and nowhere else: they
are evidence for a decision, not kernels a port may run.

Run as a script for the tables; `tests/test_interp_kernel.py` pins every
figure quoted in design.md so the record cannot drift from what the code
computes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from enodia.spec.beamform.interp import fractional_delay_taps
from enodia.spec.probe import BANDWIDTH_LEVEL_DB, ProbeProfile, linear_5mhz

# The label every 13 MHz figure of this sweep carries until #10 supplies a
# real profile. It is not `rf-oracle-frozen-0p7` (design.md §15), which is a
# different, separately frozen 13 MHz record.
SYNTHETIC_13MHZ_ENVELOPE = "synthetic-80pct-design-envelope"


@dataclass(frozen=True)
class SweepCase:
    """One decimation case, with the provenance of the pulse it assumes.

    `identity` is a profile name or a synthetic-envelope label; `status` is
    the profile's bandwidth status (`provisional` / `sourced`) or
    `synthetic`; `source` is the profile's `bandwidth_source`, None when
    there is none to name. `bandwidth_frac` is the full fractional width at
    `BANDWIDTH_LEVEL_DB` (half amplitude), and `band_edge` is the one-sided
    edge in cycles per *decimated* sample — the quantity the metrics below
    take.
    """

    name: str
    identity: str
    status: str
    source: str | None
    f0_hz: float
    bandwidth_frac: float
    decimation: int
    fs_hz: float = 40e6

    @property
    def level_db(self) -> float:
        return BANDWIDTH_LEVEL_DB

    @property
    def edge_hz(self) -> float:
        """One-sided analysis edge [Hz]: half the full half-amplitude width."""
        return self.bandwidth_frac * self.f0_hz / 2.0

    @property
    def fs_decimated_hz(self) -> float:
        return self.fs_hz / self.decimation

    @property
    def band_edge(self) -> float:
        """The edge as a fraction of the decimated sampling rate fs'."""
        return self.edge_hz / self.fs_decimated_hz

    def describe(self) -> str:
        """One line carrying every provenance field a quoted figure needs."""
        return (
            f"{self.name}: {self.identity} [{self.status}; source: {self.source or 'none'}] "
            f"f0 {self.f0_hz / 1e6:g} MHz, full fraction {self.bandwidth_frac:g} "
            f"at -{self.level_db:.4f} dB amplitude, one-sided edge "
            f"{self.edge_hz / 1e6:g} MHz = {self.band_edge:.3f} fs' at D={self.decimation}"
        )


def profile_case(profile: ProbeProfile, decimation: int) -> SweepCase:
    """The case a profile defines at one decimation ratio, provenance included."""
    return SweepCase(
        name=f"{profile.f0_hz / 1e6:g}MHz_D{decimation}",
        identity=profile.name,
        status=profile.bandwidth_status,
        source=profile.bandwidth_source,
        f0_hz=profile.f0_hz,
        bandwidth_frac=profile.bandwidth_frac,
        decimation=decimation,
        fs_hz=profile.fs_hz,
    )


# design.md §4's 13 MHz envelope: 80% fractional bandwidth, 7.8–18.2 MHz, a
# one-sided edge of 5.2 MHz at D=2 (fs' = 20 MHz). Synthetic, not a profile.
SYNTHETIC_13MHZ_D2 = SweepCase(
    name="13MHz_D2",
    identity=SYNTHETIC_13MHZ_ENVELOPE,
    status="synthetic",
    source=None,
    f0_hz=13e6,
    bandwidth_frac=0.8,
    decimation=2,
)

# The three cases design.md §5 names: the 5 MHz profile at D=8 and D=4, and
# the 13 MHz envelope at D=2.
CASES: tuple[SweepCase, ...] = (
    profile_case(linear_5mhz(), 8),
    SYNTHETIC_13MHZ_D2,
    profile_case(linear_5mhz(), 4),
)
CASES_BY_NAME: dict[str, SweepCase] = {case.name: case for case in CASES}

# Band edge as a fraction of the decimated sampling rate fs', per case.
BAND_EDGES: dict[str, float] = {case.name: case.band_edge for case in CASES}

# How far down the pulse's amplitude spectrum is at the band edge. The first
# entry is the case's own model — the edge is by definition the half-amplitude
# point (#46). The two narrower pulses are kept as a sensitivity sweep: the
# 5 MHz value is provisional and the 13 MHz one synthetic, and the kernel
# ranking depends on which applies.
PULSE_ROLLOFFS_DB: tuple[float, ...] = (BANDWIDTH_LEVEL_DB, 20.0, 40.0)

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
    no-interpolation case (47° and 63°), extended to a kernel.

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
    At ``rolloff_db = BANDWIDTH_LEVEL_DB`` the Gaussian *is* the case's pulse
    model — the edge is its half-amplitude point by definition (#46); a
    larger value is a narrower pulse, swept because the 5 MHz width is
    provisional and the 13 MHz one synthetic.
    """
    sigma = band_edge / np.sqrt(2.0 * np.log(10.0 ** (rolloff_db / 20.0)))
    # Stop at the decimated Nyquist frequency. The record cannot represent
    # anything above 0.5 cycles/sample, so scoring the kernel's response
    # there measures a signal that does not exist in it. Worth noticing that
    # the profile's own pulse reaches past it at D=8 — 3σ is 0.89 at the
    # 5 MHz edge of 0.35 fs' — which says that width and that decimation
    # ratio are in tension: what D=8 costs is the axial-PSF question §17
    # keeps open.
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


def provenance_block() -> str:
    """What every figure below assumed, one line per case."""
    return "\n".join(case.describe() for case in CASES)


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
    for case in CASES:
        edge = case.band_edge
        lines.append(f"-- {case.name} ({case.identity}, edge {edge:.3f} fs') --")
        lines.append(
            f"   {'kernel':12s}" + "".join(f"{f'-{db:.2f}dB':>11s}" for db in PULSE_ROLLOFFS_DB)
        )
        for name, fn in _all_candidates(edge).items():
            row = f"   {name:12s}"
            for db in PULSE_ROLLOFFS_DB:
                row += f"{pulse_weighted_rms_error(fn, edge, rolloff_db=db) * 100:10.2f}%"
            lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Cases (identity [status; source], width convention, level, edge):")
    print(provenance_block())
    print()
    print("Worst-case error at the band edge (phase / magnitude):")
    print(edge_table())
    print()
    print(
        "RMS error over a Gaussian pulse whose amplitude is N dB down at that edge\n"
        f"(the first column, -{BANDWIDTH_LEVEL_DB:.2f} dB, is the case's own pulse model):"
    )
    print(weighted_table())
    print()
    # The profile-specific reconciliation beside the frozen RF oracle
    # (design.md §15). Imported here so the sweep's own import stays light.
    from enodia.spec.beamform.profile_reconciliation import report

    print(report(linear_5mhz()))
