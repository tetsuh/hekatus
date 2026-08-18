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

from enodia.spec.beamform.rf_delay import delay_rf
from enodia.spec.probe import ProbeProfile
from enodia.spec.records import RFEventRecord
from enodia.spec.sequence import TxEvent


def depth_grid(profile: ProbeProfile, *, z_min_m: float = 2e-3) -> np.ndarray:
    """Image depth axis [m], spaced one RF sample apart (c / 2fs). float64."""
    dz = profile.c_m_s / (2.0 * profile.fs_hz)
    n = int((profile.depth_m - z_min_m) / dz)
    return z_min_m + np.arange(n, dtype=np.float64) * dz


def _records_by_event(
    events: list[TxEvent], records: list[RFEventRecord]
) -> dict[int, RFEventRecord]:
    """Index records by the event they name, rejecting anything inconsistent.

    Records are matched through `tx_event_index` rather than by position:
    the header is what names the data, so arrival order carries no meaning
    (docs/dataplane.md). Positional pairing would silently place RF data on
    the wrong scanline when a record is missing, extra, or reordered.

    Generations must also agree across the frame. Compounding several
    transmits acquired under different tables is the accident that
    snapshot-based switching exists to prevent (design.md §4, §19), and a
    frame assembled from mixed generations is exactly that accident.
    """
    by_event: dict[int, RFEventRecord] = {}
    for rec in records:
        idx = rec.header.tx_event_index
        if idx in by_event:
            raise ValueError(f"duplicate record for transmit event {idx}")
        by_event[idx] = rec

    wanted = {ev.event_index for ev in events}
    missing = sorted(wanted - by_event.keys())
    extra = sorted(by_event.keys() - wanted)
    if missing or extra:
        raise ValueError(
            f"record set does not match the events: missing {missing}, unexpected {extra}"
        )

    generations = {(r.header.config_id, r.header.param_generation) for r in records}
    if len(generations) > 1:
        raise ValueError(f"records span several parameter generations: {sorted(generations)}")

    return by_event


def _check_channel_count(profile: ProbeProfile, rec: RFEventRecord) -> None:
    """A payload must carry exactly the aperture's channels.

    Too few and the delay indexing fails on an incidental IndexError; too
    many and the extra channels are silently dropped, beamforming a smaller
    aperture than the profile describes while reporting nothing. Channel
    dropout is expressed as an apodization mask (design.md §6), never by
    handing over a shorter array.
    """
    n_ch = rec.data.shape[0]
    if n_ch != profile.n_elements:
        raise ValueError(
            f"transmit event {rec.header.tx_event_index}: payload has {n_ch} channels, "
            f"profile {profile.name} has {profile.n_elements}"
        )


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
    if not np.issubdtype(np.dtype(dtype), np.floating):
        # The dtype parameter exists to sweep precision (float64 / float32 /
        # bfloat16). An integer one truncates every fractional delay to zero
        # and every normalized apodization weight with it, which is a
        # silently wrong image rather than a lower-precision one.
        raise ValueError(f"dtype must be floating-point for beamforming, got {np.dtype(dtype)}")

    el_x = profile.element_x()
    z = depth_grid(profile, z_min_m=z_min_m)
    c = profile.c_m_s
    line_x = np.array([ev.line_x_m for ev in events], dtype=np.float64)

    by_event = _records_by_event(events, records)
    image = np.zeros((z.size, len(events)), dtype=dtype)
    for ev in events:
        rec = by_event[ev.event_index]
        _check_channel_count(profile, rec)
        data = rec.data.astype(dtype)
        dx = el_x[:, None] - ev.line_x_m  # (n_ch, 1)
        tau = (z[None, :] + np.hypot(dx, z[None, :])) / c  # (n_ch, n_depth)
        pos = tau * profile.fs_hz
        sampled = delay_rf(data, pos)

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
    """Log compression [dB], relative to the frame maximum, floored at the range.

    A silent frame — no scatterers, or an all-zero acquisition — has no
    maximum to normalize against; it compresses to the floor rather than to
    NaN, so a diagnostic image stays displayable instead of turning into
    something no downstream stage can interpret.
    """
    peak = float(env.max())
    if peak <= 0.0:
        return np.full(env.shape, -dynamic_range_db, dtype=env.dtype)
    db = 20.0 * np.log10(np.maximum(env, 1e-30) / peak)
    return np.clip(db, -dynamic_range_db, 0.0)
