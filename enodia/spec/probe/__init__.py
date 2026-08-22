"""Probe profiles and geometry (design.md §4, §6).

Every parameter is sweepable, and geometry is computed in float64 — it runs
once, so the precision is free and removes the reference implementation from
the list of suspects when a phase error shows up (design.md §14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The spectral level at which `ProbeProfile.bandwidth_frac` sits: half
# amplitude, 20·log10(2) ≈ 6.0206 dB below the peak of the pulse's amplitude
# spectrum (equivalently −6.0206 dB in power, since P ∝ |A|²). Stated as a
# number because "−6 dB" was read three ways before #46 fixed it (design.md
# §4).
BANDWIDTH_LEVEL_DB: float = float(20.0 * np.log10(2.0))


@dataclass(frozen=True)
class ProbeProfile:
    """One probe as a single settings bundle.

    Switching probes is a table swap; no kernel or code path changes with it.

    **Bandwidth (normative, design.md §4, ADR-0008).** `bandwidth_frac` is the
    full fractional bandwidth of the *effective two-way pulse* the processing
    profile assumes, relative to `f0_hz`: its endpoints are the two points
    `BANDWIDTH_LEVEL_DB` (half amplitude, ≈ −6.0206 dB) below the spectral
    peak, so the full width is `bandwidth_frac · f0_hz` and the one-sided
    analysis edge of the symmetric baseband model is half of that
    (`bandwidth_edge_hz`). It describes neither a transmit-only spectrum nor
    an AFE anti-alias guarantee. The simulator's pulse (`enodia.spec.sim`)
    is built to this definition, and §5's band-edge figures derive from it.

    `bandwidth_source` is provenance: a non-empty string naming the
    manufacturer data or measurement the value comes from, or `None`, which
    means **provisional** — a working value with no physical backing. `None`
    never means measured or validated. The bandwidth lives here and nowhere
    else: the transmit description (§19) references a profile by id and
    carries no bandwidth of its own.
    """

    name: str
    n_elements: int
    pitch_m: float
    f0_hz: float
    bandwidth_frac: float  # full fractional width at BANDWIDTH_LEVEL_DB, relative to f0
    fs_hz: float  # ADC sampling rate
    c_m_s: float  # reference sound speed
    depth_m: float  # imaging depth
    tx_focus_m: float  # transmit focal depth
    f_number: float  # F-number of the transmit and receive apertures
    bandwidth_source: str | None = None  # provenance; None = provisional

    def __post_init__(self) -> None:
        if self.bandwidth_source is not None and not self.bandwidth_source.strip():
            # An empty string would read as "sourced" to anything that only
            # checks for None, while naming no source. Provisional is spelled
            # None, and only None.
            raise ValueError("bandwidth_source must name a source or be None (provisional)")
        if not self.bandwidth_frac > 0.0:
            raise ValueError(f"bandwidth_frac must be positive, got {self.bandwidth_frac}")

    @property
    def wavelength_m(self) -> float:
        return self.c_m_s / self.f0_hz

    @property
    def bandwidth_hz(self) -> float:
        """Full width of the effective two-way pulse between its half-amplitude points [Hz]."""
        return self.bandwidth_frac * self.f0_hz

    @property
    def bandwidth_edge_hz(self) -> float:
        """One-sided analysis edge of the symmetric baseband model [Hz]: half the full width."""
        return self.bandwidth_frac * self.f0_hz / 2.0

    @property
    def bandwidth_status(self) -> str:
        """`"provisional"` when no source is named, else `"sourced"`."""
        return "provisional" if self.bandwidth_source is None else "sourced"

    def element_x(self) -> np.ndarray:
        """Element centre x coordinates [m], float64.

        The aperture is centred on the origin and the element face lies at
        z = 0.
        """
        n = self.n_elements
        return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * self.pitch_m


def linear_5mhz() -> ProbeProfile:
    """5 MHz linear array.

    Placeholder geometry per design.md §4: the kind and frequency band are
    what matter, and exact element counts and pitches swap in later as
    profile data. The bandwidth is **provisional** (`bandwidth_source=None`):
    0.7 is the working value the simulator and the sweeps have used since
    MVP-1, and no manufacturer or measured pulse response stands behind it.
    #46 made that status explicit rather than inventing one; a sourced value
    replaces it through a reviewed profile update and reruns what consumed it.
    """
    return ProbeProfile(
        name="linear-5mhz",
        n_elements=128,
        pitch_m=0.3e-3,
        f0_hz=5.0e6,
        bandwidth_frac=0.7,
        bandwidth_source=None,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=60e-3,
        tx_focus_m=20e-3,
        f_number=2.0,
    )
