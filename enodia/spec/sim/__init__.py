"""Analytic simulator — smallest form.

What the simulator owes is formal accuracy, not acoustic fidelity
(design.md §15): the data format, volume, and geometry must be right, while
finite-element-size effects matter least. That is why this is a few hundred
lines of NumPy rather than a full-wave or spatial-impulse-response package.

Fidelity that is deliberately absent here and arrives with #8: TGC, 12-bit
quantization, element directivity, attenuation, aberration injection with a
correlation length, and element dropout.

The transmit beam model lives in `enodia.spec.sim.transmit` (#9): the
virtual-source approximation with a Gaussian amplitude profile, blended
across the focal singularity, and aperture superposition as the switchable
higher-fidelity alternative. This module owns the receive path and the
record format; it asks that module what the transmit pulse looks like at a
scatterer and does not model the beam itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from enodia.spec.probe import ProbeProfile
from enodia.spec.records import EventHeader, RFEventRecord
from enodia.spec.sequence import TransmitConfig, TxEvent
from enodia.spec.sim.transmit import transmit_contributions

DEFAULT_PULSE_SUPPORT_SIGMAS = 6.0

__all__ = [
    "DEFAULT_PULSE_SUPPORT_SIGMAS",
    "PointScatterer",
    "gaussian_pulse",
    "n_rf_samples",
    "simulate_frame",
]


@dataclass(frozen=True)
class PointScatterer:
    x_m: float
    z_m: float
    amplitude: float = 1.0


def _gaussian_envelope_sigma(f0_hz: float, bandwidth_frac: float) -> float:
    """Return the envelope σ implied by the profile's half-amplitude bandwidth."""
    return float(np.sqrt(2.0 * np.log(2.0)) / (np.pi * bandwidth_frac * f0_hz))


def _validate_pulse_support_sigmas(pulse_support_sigmas: float | None) -> float | None:
    if pulse_support_sigmas is None:
        return None
    if isinstance(pulse_support_sigmas, (bool, np.bool_)) or not isinstance(
        pulse_support_sigmas, (int, float, np.integer, np.floating)
    ):
        raise TypeError("pulse_support_sigmas must be a real number or None")
    value = float(pulse_support_sigmas)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("pulse_support_sigmas must be finite and non-negative")
    return value


def gaussian_pulse(t: np.ndarray, f0_hz: float, bandwidth_frac: float) -> np.ndarray:
    """Two-way pulse: a Gaussian envelope on the carrier.

    `bandwidth_frac` is the profile's definition (design.md §4, ADR-0008): the
    full fractional width between the two points where the amplitude
    spectrum is half its peak — `BANDWIDTH_LEVEL_DB` ≈ 6.0206 dB down. The
    envelope's σ follows from that: the amplitude spectrum of
    exp(−t²/2σ²) is exp(−2π²f²σ²), which equals 1/2 at
    f = bandwidth_frac·f0/2 when σ = √(2 ln 2) / (π·bandwidth_frac·f0).
    `tests/test_sim_format.py` checks the spectrum rather than the formula.
    """
    sigma = _gaussian_envelope_sigma(f0_hz, bandwidth_frac)
    return np.exp(-0.5 * (t / sigma) ** 2) * np.cos(2.0 * np.pi * f0_hz * t)


def n_rf_samples(profile: ProbeProfile) -> int:
    """Samples per channel: the round trip to the imaging depth, plus pulse tail."""
    t_max = 2.0 * profile.depth_m / profile.c_m_s
    return int(np.ceil(t_max * profile.fs_hz)) + 256


def _accumulate_supported_pulse(
    event_frame: np.ndarray,
    t_axis: np.ndarray,
    arrival: np.ndarray,
    amplitude: float,
    profile: ProbeProfile,
    sigma: float,
    support_sigmas: float,
) -> None:
    """Add one contribution, evaluating pulse values only inside its clipped window."""
    radius_s = support_sigmas * sigma
    starts = np.searchsorted(t_axis, arrival - radius_s, side="left")
    stops = np.searchsorted(t_axis, arrival + radius_s, side="right")
    counts = stops - starts

    # This bound gives every contribution the same work shape; fractional
    # arrival positions may leave the final slot unused for a channel.
    width_in_samples = 2.0 * radius_s * profile.fs_hz
    if not math.isfinite(width_in_samples) or width_in_samples >= t_axis.size:
        width = t_axis.size
    else:
        width = math.ceil(width_in_samples) + 1
    offsets = np.arange(width)
    sample_indices = starts[:, None] + offsets[None, :]
    valid = (offsets[None, :] < counts[:, None]) & (sample_indices < t_axis.size)
    safe_indices = np.minimum(sample_indices, t_axis.size - 1)
    actual_times = t_axis[safe_indices] - arrival[:, None]
    valid &= np.abs(actual_times) <= radius_s
    # Padding evaluates only the pulse centre (inside every nonnegative
    # support) and is excluded from accumulation, keeping work data-independent.
    pulse_times = np.where(valid, actual_times, 0.0)
    pulse_values = gaussian_pulse(pulse_times, profile.f0_hz, profile.bandwidth_frac)
    if np.any(valid):
        channel_indices = np.broadcast_to(np.arange(arrival.size)[:, None], sample_indices.shape)[valid]
        event_frame[channel_indices, sample_indices[valid]] += amplitude * pulse_values[valid]


def _simulate_bmode_float64_frame(
    profile: ProbeProfile,
    events: list[TxEvent],
    scatterers: list[PointScatterer],
    *,
    transmit_model: str = "virtual-source",
    blend_half_width: float | None = None,
    pulse_support_sigmas: float | None = DEFAULT_PULSE_SUPPORT_SIGMAS,
) -> np.ndarray:
    """Build the unscaled float64 frame; the full-record path is the reference."""
    support_sigmas = _validate_pulse_support_sigmas(pulse_support_sigmas)
    el_x = profile.element_x()
    c = profile.c_m_s
    n_t = n_rf_samples(profile)
    t_axis = np.arange(n_t, dtype=np.float64) / profile.fs_hz
    sigma = _gaussian_envelope_sigma(profile.f0_hz, profile.bandwidth_frac)

    frame = np.zeros((len(events), profile.n_elements, n_t), dtype=np.float64)
    for event_index, event in enumerate(events):
        for scatterer in scatterers:
            taus, weights = transmit_contributions(
                profile,
                event,
                scatterer.x_m,
                scatterer.z_m,
                model=transmit_model,
                blend_half_width=blend_half_width,
            )
            t_rx = np.hypot(scatterer.x_m - el_x, scatterer.z_m) / c
            for tau, weight in zip(taus, weights, strict=True):
                arrival = float(tau) + t_rx
                amplitude = scatterer.amplitude * float(weight)
                if support_sigmas is None:
                    frame[event_index] += amplitude * gaussian_pulse(
                        t_axis[None, :] - arrival[:, None],
                        profile.f0_hz,
                        profile.bandwidth_frac,
                    )
                else:
                    _accumulate_supported_pulse(
                        frame[event_index],
                        t_axis,
                        arrival,
                        amplitude,
                        profile,
                        sigma,
                        support_sigmas,
                    )
    return frame


def _simulate_bmode_frame(
    profile: ProbeProfile,
    events: list[TxEvent],
    scatterers: list[PointScatterer],
    *,
    config_id: str = "default",
    param_generation: int = 0,
    int16_fullscale_frac: float = 0.5,
    transmit_model: str = "virtual-source",
    blend_half_width: float | None = None,
    pulse_support_sigmas: float | None = DEFAULT_PULSE_SUPPORT_SIGMAS,
) -> list[RFEventRecord]:
    """Generate one frame of raw RF.

    The signal received on element j for transmit event k is

        sum over scatterers s, over transmit contributions (tau, w) of
            A_s * w * pulse(t - tau - |r_s - r_j| / c)

    where `enodia.spec.sim.transmit` supplies the (arrival time, amplitude)
    contributions. The virtual-source model supplies one of them per
    scatterer and aperture superposition one per element; the summation here
    is the same either way, so the two models differ in the beam and in
    nothing else.

    **Every contribution is summed, silent elements included.** Skipping the
    zero-weight ones would make the work depend on the aperture, and this
    implementation is the specification a port is written against.

    `pulse_support_sigmas` bounds pulse evaluation to samples within that many
    envelope standard deviations of each arrival. Its default is
    `DEFAULT_PULSE_SUPPORT_SIGMAS`; `None` evaluates the full record and is the
    numerical reference path.
    """
    support_sigmas = _validate_pulse_support_sigmas(pulse_support_sigmas)
    n_t = n_rf_samples(profile)
    pri_ns = round(n_t / profile.fs_hz * 1e9)  # pulse repetition interval, for timestamps
    frame = _simulate_bmode_float64_frame(
        profile,
        events,
        scatterers,
        transmit_model=transmit_model,
        blend_half_width=blend_half_width,
        pulse_support_sigmas=support_sigmas,
    )

    # One scale for the whole frame: per-event scaling would destroy the
    # relative amplitudes that transmit compounding depends on.
    peak = np.abs(frame).max()
    scale = int16_fullscale_frac * np.iinfo(np.int16).max / peak if peak > 0 else 1.0

    records = []
    for k, ev in enumerate(events):
        header = EventHeader(
            seq=k,
            config_id=config_id,
            param_generation=param_generation,
            tx_event_index=ev.event_index,
            tx_type=ev.tx_type,
            timestamp_ns=k * pri_ns,
        )
        records.append(
            RFEventRecord(header=header, data=np.round(frame[k] * scale).astype(np.int16))
        )
    return records


def simulate_frame(
    profile: ProbeProfile,
    config: TransmitConfig,
    scatterers: list[PointScatterer],
    *,
    int16_fullscale_frac: float = 0.5,
    transmit_model: str = "virtual-source",
    blend_half_width: float | None = None,
    pulse_support_sigmas: float | None = DEFAULT_PULSE_SUPPORT_SIGMAS,
) -> list[RFEventRecord]:
    """Generate one frame from an accepted transmit configuration (#52).

    The simulator is the transmit schema's first consumer, which is how the
    schema gets exercised rather than merely defined (design.md §18 item 2,
    §19). Taking the configuration rather than a bare event list is the whole
    of it: the config ID travels with the events it describes, so a frame
    cannot be stamped with the identity of a configuration it did not come
    from.

    What the default model reads from each event is the virtual source, the
    beam axis and the transmit-type tag. `transmit_model=
    "aperture-superposition"` reads the per-element firing delays and
    apodization instead — the schema fields #52 carries and validates, and
    the only consumer that synthesizes a field from them. It is the yardstick
    the blend width is swept against (#9), not the default: design.md §18
    adopts the virtual-source approximation and keeps this switchable.

    The frame is stamped with the configuration's own **parameter
    generation**, not a separately supplied one. It used to default to 0
    regardless, so a configuration accepted at any other generation produced
    records the beamformer then refused — the official path failing closed
    unless the caller passed the generation a second time (§19: the tag has
    one source, and it is the configuration).

    `pulse_support_sigmas` bounds pulse evaluation to samples within that many
    envelope standard deviations of each arrival. Its default is
    `DEFAULT_PULSE_SUPPORT_SIGMAS`; pass `None` to evaluate the full record as
    a reference.
    """
    if config.probe_profile_id != profile.name:
        raise ValueError(
            f"configuration {config.config_id!r} runs on probe profile"
            f" {config.probe_profile_id!r}, simulated on {profile.name!r}"
        )
    return _simulate_bmode_frame(
        profile,
        list(config.events),
        scatterers,
        config_id=config.config_id,
        param_generation=config.param_generation,
        int16_fullscale_frac=int16_fullscale_frac,
        transmit_model=transmit_model,
        blend_half_width=blend_half_width,
        pulse_support_sigmas=pulse_support_sigmas,
    )
