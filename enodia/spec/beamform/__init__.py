"""RF-domain ideal-delay DAS — the golden path — plus envelope and log compression.

The host reference implementation must always keep an RF-domain ideal-delay
DAS (CLAUDE.md, absolute rules). It is not how the accelerator will do the
work — there, delays are applied after IQ demodulation, because raw RF does
not fit in L1 — but it is the only yardstick that quantifies the error of
that approximation. The IQ path and the comparison against this golden are
#6.

Receive uses a fixed-F-number dynamic aperture with Hann apodization. The
dtype is a parameter, as every design parameter is.

Known approximation in this golden path: fractional delays are taken by
linear interpolation on the 40 MHz RF. At 5 MHz that is 8x oversampled, so
the error is small, but it is not zero, and a yardstick with its own
interpolation error contaminates the very comparison it exists for. Raised
as #25.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from enodia.spec.probe import ProbeProfile
from enodia.spec.records import RFEventRecord
from enodia.spec.sequence import TxEvent


def depth_grid(profile: ProbeProfile, *, z_min_m: float = 2e-3) -> np.ndarray:
    """Image depth axis [m], spaced one RF sample apart (c / 2fs). float64."""
    dz = profile.c_m_s / (2.0 * profile.fs_hz)
    n = int((profile.depth_m - z_min_m) / dz)
    return z_min_m + np.arange(n, dtype=np.float64) * dz


def das_rf_golden(
    profile: ProbeProfile,
    events: list[TxEvent],
    records: list[RFEventRecord],
    *,
    dtype=np.float32,
    z_min_m: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal-delay delay-and-sum in the RF domain.

    Returns (RF image [depth x scanline], depth axis, scanline x axis).

    The delay is the transmit leg z/c along the beam axis plus the receive
    leg |r_p - r_j| / c. The contribution map is the identity (event k forms
    line k); MLA and transmit compounding generalize it in #7.
    """
    el_x = profile.element_x()
    z = depth_grid(profile, z_min_m=z_min_m)
    c = profile.c_m_s
    line_x = np.array([ev.line_x_m for ev in events], dtype=np.float64)

    image = np.zeros((z.size, len(events)), dtype=dtype)
    rows = np.arange(profile.n_elements)[:, None]
    for ev, rec in zip(events, records):
        data = rec.data.astype(dtype)
        n_t = data.shape[1]
        dx = el_x[:, None] - ev.line_x_m  # (n_ch, 1)
        tau = (z[None, :] + np.hypot(dx, z[None, :])) / c  # (n_ch, n_depth)
        pos = tau * profile.fs_hz
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, n_t - 2)
        frac = (pos - i0).astype(dtype)
        sampled = (1.0 - frac) * data[rows, i0] + frac * data[rows, i0 + 1]

        # Fixed-F-number dynamic aperture with Hann apodization, normalized
        # by the weight sum at each depth so the aperture growth does not
        # imprint a depth-dependent gain.
        u = dx / (z[None, :] / (2.0 * profile.f_number))
        w = np.where(np.abs(u) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
        w = (w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)).astype(dtype)

        image[:, ev.line_index] += (w * sampled).sum(axis=0)

    return image, z, line_x


def envelope(rf_image: np.ndarray) -> np.ndarray:
    """Envelope by the Hilbert transform along depth."""
    return np.abs(hilbert(rf_image, axis=0))


def log_compress(env: np.ndarray, *, dynamic_range_db: float = 50.0) -> np.ndarray:
    """Log compression [dB], relative to the frame maximum, floored at the range."""
    db = 20.0 * np.log10(np.maximum(env, 1e-30) / env.max())
    return np.clip(db, -dynamic_range_db, 0.0)
