"""The IQ path against the RF-domain golden, stage by stage (design.md §15, #6).

This is the prototype of the L0 checkpoints: intermediate quantities, not
images, each reported as a relative error and — where the quantity is
complex — a phase error, so a later mismatch is traceable to a stage.
Checkpoints 1 (front-end output) and 2 (post-delay channel vectors) are
what the IQ path reaches today; the image comparison at the end is the
§15 "after envelope / log compression — dB difference" row, kept because
it is what a reader asks for first, and placed last because it is the
least diagnostic.

**The yardstick's own floor is quoted beside every figure.** The golden's
fractional delay carries a residual of its own — 0.0003 % at 5 MHz under
the frozen oracle, and the profile reconciliation of §15 states it for the
profile in use — and a reported difference smaller than that residual is
observed but not attributable: it cannot be told apart from the yardstick's
own error (design.md §15, #25). The report says so next to any such figure
rather than leaving the reader to notice.

**References.**

- Checkpoint 1: the ideal baseband of the RF record — its analytic signal
  (Hilbert) times e^(−j2πf0·t), band-limited by a brick-wall low-pass at the
  decimated Nyquist frequency, read at the RF positions m·D + δ the front
  end's samples stand for (δ = `IQEventRecord.rf_offset`). Reported twice:
  for the unquantized filter output, which isolates the FIR's passband and
  aliasing error, and for the int16 record, which adds the quantization.
- Checkpoint 2: the golden's own delayed channel samples, made complex —
  the analytic signal of each channel delayed by the golden's band-limited
  ideal delay at τ_i·fs, times e^(−j2πf0·t_p) — against the IQ path's
  x_i[n]. Restricted to channels inside the receive aperture, weighted by
  the same apodization both paths apply, on the transmit events whose
  lines pass through the scatterers (a line far from every scatterer holds
  nothing but quantization noise, and a relative error on noise measures
  the noise). The phase error is the energy-weighted RMS of
  arg(x·conj(ref)), in degrees; it is what catches a flipped sign
  convention (≈ 97° here) that the image does not.
- Image: the two log-compressed envelopes on the shared grid — RMS and
  maximum dB difference over the pixels the golden puts above a floor, and
  per scatterer the axial and lateral peak positions and peak level of each.

Relative error is ‖a − b‖ / ‖b‖ over the compared set, in percent; phase
error is sqrt(Σ |ref|²·arg² / Σ |ref|²) over it, in degrees. A silent
reference (‖b‖ = 0) yields NaN and is reported as such, not dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

from enodia.spec.beamform import aperture_weights, das_rf_golden, envelope, log_compress
from enodia.spec.beamform.iq_das import das_iq, delayed_channel_vectors
from enodia.spec.beamform.profile_reconciliation import reconcile
from enodia.spec.beamform.rf_delay import delay_rf
from enodia.spec.frontend import complex_bpf_decimate, demodulate_frame
from enodia.spec.probe import ProbeProfile
from enodia.spec.records import IQEventRecord, RFEventRecord
from enodia.spec.sequence import TxEvent
from enodia.spec.sim import PointScatterer


@dataclass(frozen=True)
class StageError:
    stage: str
    relative_error_pct: float
    phase_error_deg: float | None = None

    def line(self, floor_pct: float) -> str:
        s = f"  {self.stage:44s} rel {self.relative_error_pct:9.4f} %"
        if self.phase_error_deg is not None:
            s += f"   phase RMS {self.phase_error_deg:7.2f} deg"
        if self.relative_error_pct < floor_pct:
            s += f"   [below the yardstick's own residual {floor_pct:.4f} %: not attributable]"
        return s


@dataclass(frozen=True)
class PeakComparison:
    scatterer: tuple[float, float]
    golden: tuple[float, float, float]  # axial [m], lateral [m], dB
    iq: tuple[float, float, float]


@dataclass(frozen=True)
class ImageComparison:
    rms_db: float
    max_db: float
    region_floor_db: float
    peaks: tuple[PeakComparison, ...]


@dataclass(frozen=True)
class ComparisonReport:
    profile: str
    bandwidth_status: str
    decimation: int
    floor_pct: float
    checkpoint1: tuple[StageError, StageError]
    checkpoint2: tuple[StageError, ...]
    checkpoint2_events: tuple[int, ...]
    image: ImageComparison
    seconds: dict[str, float] = field(default_factory=dict)

    def lines(self) -> list[str]:
        floor = (
            f"  yardstick floor (golden's own residual on this profile's record): {self.floor_pct:.4f} %"
            " — a difference below it is observed, not attributable"
        )
        out = [
            f"golden comparison: {self.profile} (bandwidth {self.bandwidth_status}), D={self.decimation}",
            floor,
            "checkpoint 1 — front-end output against the ideal band-limited baseband, all channels",
        ]
        out += [e.line(self.floor_pct) for e in self.checkpoint1]
        out.append(
            "checkpoint 2 — post-delay channel vectors against the golden's analytic channel"
            f" samples, lines through the scatterers (events {list(self.checkpoint2_events)}),"
            " inside the aperture"
        )
        out += [e.line(self.floor_pct) for e in self.checkpoint2]
        im = self.image
        out.append(
            f"image — log envelope, pixels the golden puts above {im.region_floor_db:g} dB:"
            f" RMS {im.rms_db:.3f} dB, max {im.max_db:.3f} dB"
        )
        for pk in im.peaks:
            gx, ix = pk.golden, pk.iq
            out.append(
                f"  scatterer ({pk.scatterer[0] * 1e3:+.0f}, {pk.scatterer[1] * 1e3:.0f}) mm:"
                f" golden z {gx[0] * 1e3:.4f} x {gx[1] * 1e3:+.3f} {gx[2]:+.3f} dB |"
                f" iq z {ix[0] * 1e3:.4f} x {ix[1] * 1e3:+.3f} {ix[2]:+.3f} dB |"
                f" Δz {(ix[0] - gx[0]) * 1e6:+.1f} µm Δx {(ix[1] - gx[1]) * 1e6:+.1f} µm"
                f" ΔdB {ix[2] - gx[2]:+.3f}"
            )
        if self.seconds:
            out.append("runtime: " + ", ".join(f"{k} {v:.1f} s" for k, v in self.seconds.items()))
        return out


def _relative_error_pct(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan")
    return 100.0 * float(np.linalg.norm(a - b) / denom)


def _energy_weighted_phase_rms_deg(a: np.ndarray, ref: np.ndarray, weight=None) -> float:
    energy = np.abs(ref) ** 2 if weight is None else np.abs(ref) ** 2 * weight
    total = float(np.sum(energy))
    if total <= 0.0:
        return float("nan")
    ph = np.angle(a * np.conj(ref))
    return float(np.degrees(np.sqrt(np.sum(energy * ph**2) / total)))


def ideal_iq_reference(
    rf: np.ndarray, profile: ProbeProfile, *, decimation: int, rf_offset: float, n_iq: int
) -> np.ndarray:
    """Analytic signal → baseband → brick-wall low-pass at fs/(2D) → read at m·D + rf_offset."""
    rf = np.asarray(rf, dtype=np.float64)
    n_t = rf.shape[-1]
    n = np.arange(n_t)
    baseband = hilbert(rf, axis=-1) * np.exp(-2j * np.pi * profile.f0_hz * n / profile.fs_hz)
    spectrum = np.fft.fft(baseband, axis=-1)
    f = np.fft.fftfreq(n_t, 1.0 / profile.fs_hz)
    keep = np.abs(f) <= profile.fs_hz / (2.0 * decimation)
    # x(t + δ) ↔ X(f)·e^(+j2πfδ): the value at RF position n + δ.
    shift = np.exp(2j * np.pi * f * rf_offset / profile.fs_hz)
    shifted = np.fft.ifft(spectrum * (keep * shift)[None, :], axis=-1)
    return shifted[:, ::decimation][:, :n_iq]


def checkpoint1(
    record: RFEventRecord,
    iq: IQEventRecord,
    profile: ProbeProfile,
    *,
    decimation: int,
    **frontend_kwargs,
) -> tuple[StageError, StageError]:
    unquantized = complex_bpf_decimate(
        record.data, profile, decimation=decimation, **frontend_kwargs
    )
    ref = ideal_iq_reference(
        record.data, profile, decimation=decimation, rf_offset=iq.rf_offset, n_iq=iq.n_samples
    )
    quantized = iq.complex(np.complex128)
    return (
        StageError(
            "front-end output, unquantized FIR",
            _relative_error_pct(unquantized, ref),
            _energy_weighted_phase_rms_deg(unquantized, ref),
        ),
        StageError(
            "front-end output, int16 record",
            _relative_error_pct(quantized, ref),
            _energy_weighted_phase_rms_deg(quantized, ref),
        ),
    )


def golden_channel_vectors(
    profile: ProbeProfile, ev: TxEvent, record: RFEventRecord, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The golden's delayed channel samples made complex, and the aperture
    weights: (ref (n_ch, n_depth) complex128, w (n_ch, n_depth))."""
    s = record.data.astype(np.float64)
    a = hilbert(s, axis=-1)
    dx = profile.element_x()[:, None] - ev.line_x_m
    tau_i = (z[None, :] + np.hypot(dx, z[None, :])) / profile.c_m_s
    pos = tau_i * profile.fs_hz
    t_p = 2.0 * z[None, :] / profile.c_m_s
    ref = (delay_rf(a.real, pos) + 1j * delay_rf(a.imag, pos)) * np.exp(
        -2j * np.pi * profile.f0_hz * t_p
    )
    w = aperture_weights(dx, z, profile.f_number, dtype=np.float64)
    return ref, w


def checkpoint2(
    profile: ProbeProfile,
    ev: TxEvent,
    record: RFEventRecord,
    iq: IQEventRecord,
    z: np.ndarray,
    *,
    decimation: int,
    dtype=np.float32,
) -> StageError:
    x = delayed_channel_vectors(profile, ev, iq, z, decimation=decimation, dtype=dtype)
    ref, w = golden_channel_vectors(profile, ev, record, z)
    inside = w > 0.0
    return StageError(
        f"post-delay channel vectors, event {ev.event_index} (x {ev.line_x_m * 1e3:+.2f} mm)",
        _relative_error_pct(x[inside], ref[inside]),
        _energy_weighted_phase_rms_deg(x[inside], ref[inside], w[inside]),
    )


def events_through(events: list[TxEvent], scatterers: list[PointScatterer]) -> list[TxEvent]:
    """The transmit event whose line is nearest each scatterer, in event order."""
    chosen: dict[int, TxEvent] = {}
    for s in scatterers:
        ev = min(events, key=lambda e, x=s.x_m: abs(e.line_x_m - x))
        chosen[ev.event_index] = ev
    return [chosen[k] for k in sorted(chosen)]


def _peak(db: np.ndarray, z: np.ndarray, line_x: np.ndarray, s: PointScatterer):
    near_z = np.abs(z - s.z_m) < 2e-3
    near_x = np.abs(line_x - s.x_m) < 2e-3
    window = db[np.ix_(near_z, near_x)]
    if window.size == 0:
        # A scatterer outside the imaged region is a fact to report, not a crash.
        return float("nan"), float("nan"), float("nan")
    iz, ix = np.unravel_index(np.argmax(window), window.shape)
    return float(z[near_z][iz]), float(line_x[near_x][ix]), float(window.max())


def image_comparison(
    db_iq: np.ndarray,
    db_golden: np.ndarray,
    z: np.ndarray,
    line_x: np.ndarray,
    scatterers: list[PointScatterer],
    *,
    region_floor_db: float = -40.0,
) -> ImageComparison:
    region = db_golden > region_floor_db
    diff = db_iq - db_golden
    if not region.any():
        # A silent golden frame has no pixel above the floor: nothing to
        # compare over, reported as NaN rather than raised from an empty max.
        rms_db, max_db = float("nan"), float("nan")
    else:
        rms_db = float(np.sqrt(np.mean(diff[region] ** 2)))
        max_db = float(np.abs(diff[region]).max())
    return ImageComparison(
        rms_db=rms_db,
        max_db=max_db,
        region_floor_db=region_floor_db,
        peaks=tuple(
            PeakComparison(
                (s.x_m, s.z_m), _peak(db_golden, z, line_x, s), _peak(db_iq, z, line_x, s)
            )
            for s in scatterers
        ),
    )


def compare(
    profile: ProbeProfile,
    events: list[TxEvent],
    records: list[RFEventRecord],
    scatterers: list[PointScatterer],
    *,
    decimation: int,
    golden: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    checkpoint2_events: list[TxEvent] | None = None,
    dtype=np.float32,
    dynamic_range_db: float = 50.0,
    **frontend_kwargs,
) -> ComparisonReport:
    """Run the IQ path and compare it with the golden (computed here unless given)."""
    seconds: dict[str, float] = {}
    if golden is None:
        t0 = time.perf_counter()
        golden = das_rf_golden(profile, events, records, dtype=dtype)
        seconds["golden"] = time.perf_counter() - t0
    rf_image, z, line_x = golden

    t0 = time.perf_counter()
    iq_records = demodulate_frame(records, profile, decimation=decimation, **frontend_kwargs)
    seconds["front end"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    iq_image, _, _ = das_iq(profile, events, iq_records, decimation=decimation, dtype=dtype)
    seconds["iq das"] = time.perf_counter() - t0

    by_event = {r.header.tx_event_index: r for r in records}
    by_event_iq = {r.header.tx_event_index: r for r in iq_records}
    centre = events[len(events) // 2]
    cp1 = checkpoint1(
        by_event[centre.event_index],
        by_event_iq[centre.event_index],
        profile,
        decimation=decimation,
        **frontend_kwargs,
    )
    if checkpoint2_events is None:
        checkpoint2_events = events_through(events, scatterers)
    chosen = tuple(ev.event_index for ev in checkpoint2_events)
    t0 = time.perf_counter()
    cp2 = tuple(
        checkpoint2(
            profile,
            ev,
            by_event[ev.event_index],
            by_event_iq[ev.event_index],
            z,
            decimation=decimation,
            dtype=dtype,
        )
        for ev in checkpoint2_events
    )
    seconds["checkpoint 2"] = time.perf_counter() - t0

    db_golden = log_compress(envelope(rf_image), dynamic_range_db=dynamic_range_db)
    db_iq = log_compress(np.abs(iq_image), dynamic_range_db=dynamic_range_db)
    image = image_comparison(db_iq, db_golden, z, line_x, scatterers)
    floor = reconcile(profile, decimation, revision="n/a").rf_golden_residual_pct
    return ComparisonReport(
        profile=profile.name,
        bandwidth_status=profile.bandwidth_status,
        decimation=decimation,
        floor_pct=floor,
        checkpoint1=cp1,
        checkpoint2=cp2,
        checkpoint2_events=chosen,
        image=image,
        seconds=seconds,
    )


def main() -> None:
    import argparse

    from enodia.demo import DEFAULT_SCATTERERS

    parser = argparse.ArgumentParser(description="IQ path against the RF golden, per stage")
    parser.add_argument("--decimation", type=int, nargs="+", default=[8])
    args = parser.parse_args()
    from enodia.spec.probe import linear_5mhz
    from enodia.spec.sequence import make_bmode_config
    from enodia.spec.sim import simulate_frame

    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events)
    records = simulate_frame(profile, config, DEFAULT_SCATTERERS)
    golden = das_rf_golden(profile, events, records)
    for d in args.decimation:
        report = compare(profile, events, records, DEFAULT_SCATTERERS, decimation=d, golden=golden)
        print("\n".join(report.lines()))


if __name__ == "__main__":
    main()
