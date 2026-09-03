"""Analytic simulator — smallest form.

What the simulator owes is formal accuracy, not acoustic fidelity
(design.md §15): the data format, volume, and geometry must be right, while
finite-element-size effects matter least. That is why this is a few hundred
lines of NumPy rather than a full-wave or spatial-impulse-response package.

Fidelity that is deliberately absent here and arrives with #8: TGC, 12-bit
quantization, element directivity, attenuation, aberration injection with a
correlation length, and element dropout.

The transmit beam uses the virtual-source approximation with a Gaussian
amplitude profile. The switchable aperture-superposition model and the blend
across the focal singularity — where the transmit delay flips sign and
produces hourglass artifacts, a classic retrospective-transmit-focusing
pitfall — arrive with #9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enodia.spec.probe import ProbeProfile
from enodia.spec.records import EventHeader, RFEventRecord
from enodia.spec.sequence import TransmitConfig, TxEvent

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
) -> list[RFEventRecord]:
    """Generate one frame of raw RF.

    The signal received on element j for transmit event k is

        sum over scatterers s of
            A_s * w_tx(s, k) * pulse(t - t_tx(s, k) - |r_s - r_j| / c)

    where the transmit time of flight uses the virtual-source approximation,

        t_tx = (z_f + sign(z_s - z_f) * |r_s - v_k|) / c,

    and w_tx is a Gaussian across the transmit beam whose width grows with
    distance from the focus at the rate the F-number implies.
    """
    el_x = profile.element_x()
    c = profile.c_m_s
    n_t = n_rf_samples(profile)
    t_axis = np.arange(n_t, dtype=np.float64) / profile.fs_hz
    pri_ns = round(n_t / profile.fs_hz * 1e9)  # pulse repetition interval, for timestamps

    frame = np.zeros((len(events), profile.n_elements, n_t), dtype=np.float64)
    for k, ev in enumerate(events):
        vx, vz = ev.virtual_source_m
        for s in scatterers:
            r_sv = float(np.hypot(s.x_m - vx, s.z_m - vz))
            t_tx = (vz + np.sign(s.z_m - vz) * r_sv) / c
            beam_w = max(
                profile.wavelength_m * profile.f_number,
                abs(s.z_m - vz) / (2.0 * profile.f_number),
            )
            w_tx = np.exp(-0.5 * ((s.x_m - ev.line_x_m) / beam_w) ** 2)
            t_rx = np.hypot(s.x_m - el_x, s.z_m) / c  # (n_ch,)
            arrival = t_tx + t_rx
            frame[k] += (
                s.amplitude
                * w_tx
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
) -> list[RFEventRecord]:
    """Generate one frame from an accepted transmit configuration (#52).

    The simulator is the transmit schema's first consumer, which is how the
    schema gets exercised rather than merely defined (design.md §18 item 2,
    §19). Taking the configuration rather than a bare event list is the whole
    of it: the config ID travels with the events it describes, so a frame
    cannot be stamped with the identity of a configuration it did not come
    from.

    What the simulator reads from each event is the virtual source, the beam
    axis and the transmit-type tag. The per-element firing delays and
    apodization the schema carries are checked at ingress
    (`enodia.spec.sequence.accept`) and not synthesized into a field here:
    the transmit beam model is #9.

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
    )
