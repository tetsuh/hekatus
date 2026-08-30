"""RF-domain ideal-delay DAS — the golden path — plus envelope and log compression.

The host reference implementation must always keep an RF-domain ideal-delay
DAS (CLAUDE.md, absolute rules). It is not how the accelerator will do the
work — there, delays are applied after IQ demodulation, because raw RF does
not fit in L1 — but it is the only yardstick that quantifies the error of
that approximation. The IQ path and the comparison against this golden are
#6.

Receive uses a fixed-F-number dynamic aperture with Hann apodization. The
dtype is a parameter, as every design parameter is.

The fractional delays are the band-limited ideal delay of `rf_delay.py` —
upsampling by 8 through the FFT, then the Lagrange cubic — measured against
a frozen oracle at 0.000 % residual at 5 MHz and 0.099 % at 13 MHz, both
under a tenth of the IQ-side error this yardstick measures (#25). MVP-1's
linear interpolation stood at 6.2 % and 38.7 %.
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


def aperture_weights(dx: np.ndarray, z: np.ndarray, f_number: float, *, dtype=np.float32):
    """Fixed-F-number dynamic receive aperture with Hann apodization.

    ``dx`` is ``(n_ch, 1)`` lateral offsets of the elements from the line
    and ``z`` the depth axis; returns ``(n_ch, n_depth)`` weights normalized
    by their sum at each depth, so the aperture growth does not imprint a
    depth-dependent gain. Shared by the RF golden and the IQ path, so a
    comparison of their delayed channel vectors sees the front end and the
    delay stage and nothing else.
    """
    u = dx / (z[None, :] / (2.0 * f_number))
    w = np.where(np.abs(u) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
    return (w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)).astype(dtype)


def _slots_by_event(contribution) -> dict[int, list[tuple[int, float]]]:
    """Group a map's live slots by the event they read: {event: [(line, w)]}.

    The reference implementation is a plain loop (§7 scheme (c)), but per
    event rather than per slot, so the delayed record of one transmit is
    computed once however many lines read it. Inert slots (weight zero)
    are skipped here — on the host their contribution is exactly zero and
    the fixed-work property is the map's shape, asserted by test, not a
    property this loop needs to reproduce.
    """
    by_event: dict[int, list[tuple[int, float]]] = {}
    indices = np.asarray(contribution.event_indices)
    weights = np.asarray(contribution.weights)
    for line in range(contribution.n_lines):
        for slot in range(contribution.cap):
            w = float(weights[line, slot])
            if w > 0.0:
                by_event.setdefault(int(indices[line, slot]), []).append((line, w))
    return by_event


def das_rf_golden(
    profile: ProbeProfile,
    events: list[TxEvent],
    records: list[RFEventRecord],
    *,
    contribution=None,
    dtype=np.float32,
    z_min_m: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal-delay delay-and-sum in the RF domain.

    Returns (RF image [depth x line], depth axis, line x axis).

    The delay is the transmit leg z/c along the beam axis plus the receive
    leg |r_p - r_j| / c. ``contribution`` is the map of §7 — which events
    form which lines, with weights (`enodia.spec.sequence.contribution`,
    #53); by default the identity, event k forming line k, which is a case
    of the general structure and reproduces the pre-#53 image to the bit.
    The beamformer is never told an MLA count: the map alone carries it.
    """
    if not np.issubdtype(np.dtype(dtype), np.floating):
        # The dtype parameter exists to sweep precision (float64 / float32 /
        # bfloat16). An integer one truncates every fractional delay to zero
        # and every normalized apodization weight with it, which is a
        # silently wrong image rather than a lower-precision one.
        raise ValueError(f"dtype must be floating-point for beamforming, got {np.dtype(dtype)}")

    if contribution is None:
        contribution = _identity_contribution(events)
    el_x = profile.element_x()
    z = depth_grid(profile, z_min_m=z_min_m)
    c = profile.c_m_s
    line_x = np.array(contribution.line_x_m, dtype=np.float64)

    by_event = _records_by_event(events, records)
    event_by_index = {ev.event_index: ev for ev in events}
    image = np.zeros((z.size, contribution.n_lines), dtype=dtype)
    for event_index, slots in _slots_by_event(contribution).items():
        ev = event_by_index[event_index]
        rec = by_event[ev.event_index]
        _check_channel_count(profile, rec)
        data = rec.data.astype(dtype)
        for line, weight in slots:
            dx = el_x[:, None] - line_x[line]  # (n_ch, 1)
            tau = (z[None, :] + np.hypot(dx, z[None, :])) / c  # (n_ch, n_depth)
            pos = tau * profile.fs_hz
            sampled = delay_rf(data, pos)
            w = aperture_weights(dx, z, profile.f_number, dtype=dtype)
            image[:, line] += dtype(weight) * (w * sampled).sum(axis=0)

    return image, z, line_x


def _identity_contribution(events: list[TxEvent]):
    from enodia.spec.sequence.contribution import ContributionMap

    n = len(events)
    return ContributionMap(
        line_x_m=tuple(ev.line_x_m for ev in events),
        event_indices=np.array([[ev.event_index] for ev in events], dtype=np.intp),
        weights=np.ones((n, 1), dtype=np.float64),
    )


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
