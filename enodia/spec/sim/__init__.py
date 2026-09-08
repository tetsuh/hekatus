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

from dataclasses import dataclass

import numpy as np

from enodia.spec.probe import ProbeProfile
from enodia.spec.records import EventHeader, RFEventRecord
from enodia.spec.sequence import TransmitConfig, TxEvent
from enodia.spec.sim.transmit import transmit_contributions

__all__ = ["PointScatterer", "gaussian_pulse", "n_rf_samples", "simulate_frame"]


@dataclass(frozen=True)
class PointScatterer:
    x_m: float
    z_m: float
    amplitude: float = 1.0


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
    sigma = np.sqrt(2.0 * np.log(2.0)) / (np.pi * bandwidth_frac * f0_hz)
    return np.exp(-0.5 * (t / sigma) ** 2) * np.cos(2.0 * np.pi * f0_hz * t)


def n_rf_samples(profile: ProbeProfile) -> int:
    """Samples per channel: the round trip to the imaging depth, plus pulse tail."""
    t_max = 2.0 * profile.depth_m / profile.c_m_s
    return int(np.ceil(t_max * profile.fs_hz)) + 256


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
    """
    el_x = profile.element_x()
    c = profile.c_m_s
    n_t = n_rf_samples(profile)
    t_axis = np.arange(n_t, dtype=np.float64) / profile.fs_hz
    pri_ns = round(n_t / profile.fs_hz * 1e9)  # pulse repetition interval, for timestamps

    frame = np.zeros((len(events), profile.n_elements, n_t), dtype=np.float64)
    for k, ev in enumerate(events):
        for s in scatterers:
            taus, w_tx = transmit_contributions(
                profile,
                ev,
                s.x_m,
                s.z_m,
                model=transmit_model,
                blend_half_width=blend_half_width,
            )
            t_rx = np.hypot(s.x_m - el_x, s.z_m) / c  # (n_ch,)
            for tau, w in zip(taus, w_tx, strict=True):
                arrival = float(tau) + t_rx
                frame[k] += (
                    s.amplitude
                    * float(w)
                    * gaussian_pulse(
                        t_axis[None, :] - arrival[:, None], profile.f0_hz, profile.bandwidth_frac
                    )
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
    )
