"""IQ-domain delay-and-sum: integer shift + 4-tap interpolation + phase rotation (design.md §5, #6).

This is the delay stage the accelerator runs, written as the reference it
is compared with at L0 checkpoint 2 (design.md §15). The RF-domain golden
(`das_rf_golden`) is the yardstick that measures what this path costs; the
two share the geometry, the depth grid and the aperture weights, so at
checkpoint 2 a difference between them is the front end and the delay
stage and nothing else. (At image level the envelope estimators differ
too: Hilbert transform along depth for the golden, |complex sum| here.)

**The delay, in design.md §5's notation.** For a pixel at depth z on the
line of transmit event k, with round-trip time t_p = 2z/c along the beam
axis, channel i's echo arrives at τ_i(z) = (z + |r_p − r_i|)/c. The delay
to apply to channel i's IQ is τ = t_p − τ_i (non-positive: off-axis
channels are advanced), d = τ·fs' in decimated samples, and

    x_i[n] = interp4(z_i, n − d) · e^(−j2πf0·τ)

read at the pixel's own sample n = t_p·fs'. The read position n − d is
τ_i·fs' — the same position the golden reads the RF at, at the decimated
rate — shifted by the front end's half-sample residue: the IQ record is
read at (τ_i·fs − rf_offset)/D (`enodia.spec.frontend`). `interp4` is the
Lagrange cubic `interp.py` fixes (ADR-0007), taken by name. The phase
factor is **e^(−j2πf0·τ), normative**; its counterpart is the front end's
e^(−j2πf0·t) demodulation. With both, Σ_i w_i·x_i is the complex envelope
of the golden's RF sum times e^(−j2πf0·t_p), and the envelope |Σ| is the
image; with either flipped, the sum is incoherent in a way that still
images plausibly, which is why `tests/test_iq_das.py` asserts the phase of
x_i against the golden's analytic channel vectors rather than looking at
an image.

**Precision.** The IQ record is int16 complex; the delay stage runs at the
FP32-or-wider intermediate design.md §14 fixes: ``dtype`` float32 → complex64,
float64 → complex128, with the phase factor formed in float64 and cast.
"""

from __future__ import annotations

import math

import numpy as np

from enodia.spec.beamform import _records_by_event, aperture_weights, depth_grid
from enodia.spec.beamform.interp import fractional_delay
from enodia.spec.probe import ProbeProfile
from enodia.spec.records import IQEventRecord
from enodia.spec.sequence import TxEvent


def _complex_dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype must be floating-point for beamforming, got {dtype}")
    return np.result_type(dtype, np.complex64)


def _check_record(profile: ProbeProfile, rec: IQEventRecord, decimation: int) -> None:
    if rec.n_channels != profile.n_elements:
        raise ValueError(
            f"transmit event {rec.header.tx_event_index}: payload has {rec.n_channels} channels, "
            f"profile {profile.name} has {profile.n_elements}"
        )
    if rec.decimation != decimation:
        # Reading D=4 data as D=8 places every echo at half its depth. The
        # record names its ratio so this is a refusal, not an image.
        raise ValueError(
            f"transmit event {rec.header.tx_event_index}: record is decimated by "
            f"{rec.decimation}, beamformer configured for {decimation}"
        )
    if not math.isfinite(rec.rf_offset):
        # The record refuses this at construction; checked again here because
        # a NaN read position would come out of `fractional_delay` as zeros.
        raise ValueError(
            f"transmit event {rec.header.tx_event_index}: rf_offset is {rec.rf_offset}"
        )


def delayed_channel_vectors(
    profile: ProbeProfile,
    line_x_m: float,
    rec: IQEventRecord,
    z: np.ndarray,
    *,
    decimation: int,
    dtype=np.float32,
    kernel: str = "lagrange4",
) -> np.ndarray:
    """x_i[n] for every channel and depth of one event — L0 checkpoint 2.

    Returns ``(n_ch, n_depth)`` complex in the intermediate dtype.

    Takes the line abscissa rather than the transmit event it used to be
    read from. After #53 a line belongs to the contribution map, not to an
    event: under MLA one event feeds several lines, and passing the event
    meant rebuilding a throwaway copy of it per line to override the one
    field this function reads.
    """
    _check_record(profile, rec, decimation)
    cdtype = _complex_dtype(dtype)
    c = profile.c_m_s
    fs_dec = profile.fs_hz / decimation
    dx = profile.element_x()[:, None] - line_x_m  # (n_ch, 1)
    tau_i = (z[None, :] + np.hypot(dx, z[None, :])) / c  # (n_ch, n_depth) arrival time
    t_p = 2.0 * z[None, :] / c  # (1, n_depth) pixel round-trip time
    tau = t_p - tau_i  # the delay of design.md §5, ≤ 0
    d = tau * fs_dec
    n = t_p * fs_dec
    read_pos = n - d - rec.rf_offset / decimation  # = (τ_i·fs − rf_offset)/D
    z_iq = rec.complex(cdtype)
    sampled = fractional_delay(z_iq, read_pos, kernel=kernel)
    rotation = np.exp(-2j * np.pi * profile.f0_hz * tau).astype(cdtype)
    return sampled * rotation


def das_iq(
    profile: ProbeProfile,
    events: list[TxEvent],
    records: list[IQEventRecord],
    *,
    decimation: int,
    contribution=None,
    dtype=np.float32,
    z_min_m: float = 2e-3,
    kernel: str = "lagrange4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IQ-domain delay-and-sum.

    Returns (complex image [depth × line], depth axis, line x axis) —
    the same grid as `das_rf_golden`, so the two compare pixel for pixel.
    The image is the complex envelope along each line; |image| is the
    B-mode envelope (no Hilbert transform needed).

    ``contribution`` is §7's map, exactly as in `das_rf_golden` (#53): the
    identity by default, and there is one summation structure across both
    paths, not two.
    """
    from enodia.spec.beamform import (
        _check_event_sequence,
        _check_frame_provenance,
        _check_transmit_types,
        _identity_contribution,
        _slots_by_event,
    )

    cdtype = _complex_dtype(dtype)
    _check_event_sequence(events)
    if contribution is None:
        contribution = _identity_contribution(events)
    contribution.check_profile(profile.name)
    _check_frame_provenance(contribution, events, records)
    el_x = profile.element_x()
    z = depth_grid(profile, z_min_m=z_min_m)
    line_x = np.array(contribution.line_x_m, dtype=np.float64)
    by_event = _records_by_event(events, records)
    event_by_index = {ev.event_index: ev for ev in events}
    _check_transmit_types(contribution, event_by_index, by_event)
    image = np.zeros((z.size, contribution.n_lines), dtype=cdtype)
    for event_index, slots in _slots_by_event(contribution).items():
        ev = event_by_index[event_index]
        rec = by_event[ev.event_index]
        for line, weight in slots:
            x = delayed_channel_vectors(
                profile,
                float(line_x[line]),
                rec,
                z,
                decimation=decimation,
                dtype=dtype,
                kernel=kernel,
            )
            dx = el_x[:, None] - line_x[line]
            w = aperture_weights(dx, z, profile.f_number, dtype=dtype)
            image[:, line] += weight * (w * x).sum(axis=0)
    return image, z, line_x
