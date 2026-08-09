"""Probe profiles and geometry (design.md §4, §6).

Every parameter is sweepable, and geometry is computed in float64 — it runs
once, so the precision is free and removes the reference implementation from
the list of suspects when a phase error shows up (design.md §14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeProfile:
    """One probe as a single settings bundle.

    Switching probes is a table swap; no kernel or code path changes with it.
    """

    name: str
    n_elements: int
    pitch_m: float
    f0_hz: float
    bandwidth_frac: float  # -6 dB fractional bandwidth, relative to f0
    fs_hz: float  # ADC sampling rate
    c_m_s: float  # reference sound speed
    depth_m: float  # imaging depth
    tx_focus_m: float  # transmit focal depth
    f_number: float  # F-number of the transmit and receive apertures

    @property
    def wavelength_m(self) -> float:
        return self.c_m_s / self.f0_hz

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
    profile data.
    """
    return ProbeProfile(
        name="linear-5mhz",
        n_elements=128,
        pitch_m=0.3e-3,
        f0_hz=5.0e6,
        bandwidth_frac=0.7,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=60e-3,
        tx_focus_m=20e-3,
        f_number=2.0,
    )
