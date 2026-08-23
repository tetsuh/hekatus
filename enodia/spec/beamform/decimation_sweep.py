"""Axial point-spread function of the IQ path at D=8 against D=4 (design.md §5, §17, #6).

design.md §5 asks, in those words, to "keep the ability to measure how the
point-scatterer axial PSF changes between decimation ratios 8 and 4", and
§17 keeps "decimation ratio and interpolation tap count" as a parameter
decided by measurement. This is that measurement, on the point-scatterer
phantom the demo uses, for the `linear-5mhz` profile — whose bandwidth is
provisional (§4), so the figures are labelled with that status and are
rerun if the profile's value or provenance changes.

For each scatterer the axial profile is the log envelope along the
scanline through its lateral peak; the widths are the full widths at −6 dB
and −20 dB below the profile's peak, found by linear interpolation of the
crossings either side, in millimetres. The golden's own widths are the
reference; the IQ path's are given at each decimation ratio, with the
peak level relative to the golden's and the checkpoint errors of
`golden_compare` beside them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from enodia.spec.beamform import das_rf_golden, envelope, log_compress
from enodia.spec.beamform.golden_compare import ComparisonReport, compare
from enodia.spec.beamform.iq_das import das_iq
from enodia.spec.frontend import demodulate_frame
from enodia.spec.probe import ProbeProfile
from enodia.spec.records import RFEventRecord
from enodia.spec.sequence import TxEvent
from enodia.spec.sim import PointScatterer

DECIMATIONS: tuple[int, ...] = (8, 4)
WIDTH_LEVELS_DB: tuple[float, ...] = (-6.0, -20.0)


@dataclass(frozen=True)
class AxialPSF:
    scatterer: tuple[float, float]
    peak_z_m: float
    peak_db: float  # relative to the frame maximum
    widths_mm: dict[float, float]  # level [dB] → full width [mm]


def _crossing(zs: np.ndarray, vals: np.ndarray, level: float, i_peak: int, step: int) -> float:
    """Depth where the profile crosses ``level`` walking from the peak in ``step``."""
    i = i_peak
    while 0 <= i + step < vals.size and vals[i + step] > level:
        i += step
    j = i + step
    if not 0 <= j < vals.size:
        return float("nan")
    # linear interpolation between i (above) and j (below)
    frac = (vals[i] - level) / (vals[i] - vals[j])
    return float(zs[i] + frac * (zs[j] - zs[i]))


def axial_psf(db: np.ndarray, z: np.ndarray, line_x: np.ndarray, s: PointScatterer) -> AxialPSF:
    near_z = np.abs(z - s.z_m) < 2e-3
    near_x = np.abs(line_x - s.x_m) < 2e-3
    window = db[np.ix_(near_z, near_x)]
    iz, ix = np.unravel_index(np.argmax(window), window.shape)
    col = np.flatnonzero(near_x)[ix]
    profile = db[:, col]
    i_peak = int(np.flatnonzero(near_z)[iz])
    peak = float(profile[i_peak])
    widths = {}
    for level in WIDTH_LEVELS_DB:
        lo = _crossing(z, profile, peak + level, i_peak, -1)
        hi = _crossing(z, profile, peak + level, i_peak, +1)
        widths[level] = (hi - lo) * 1e3
    return AxialPSF((s.x_m, s.z_m), float(z[i_peak]), peak, widths)


@dataclass(frozen=True)
class SweepResult:
    profile: str
    bandwidth_status: str
    golden: tuple[AxialPSF, ...]
    iq: dict[int, tuple[AxialPSF, ...]]
    reports: dict[int, ComparisonReport]
    seconds: dict[str, float]

    def lines(self) -> list[str]:
        title = (
            f"axial PSF, {self.profile} (bandwidth {self.bandwidth_status}), point scatterers;"
            " full widths at -6 / -20 dB [mm], peak relative to frame max [dB]"
        )
        out = [title]
        out.append(
            f"  {'scatterer (x, z) mm':22s} {'golden':>26s}"
            + "".join(f"{f'IQ D={d}':>26s}" for d in self.iq)
        )
        for i, g in enumerate(self.golden):
            row = f"  ({g.scatterer[0] * 1e3:+.0f}, {g.scatterer[1] * 1e3:.0f}){'':14s}"
            row += f"{g.widths_mm[-6.0]:7.3f} / {g.widths_mm[-20.0]:6.3f} {g.peak_db:+7.3f}"
            for d, psfs in self.iq.items():
                q = psfs[i]
                row += f"   {q.widths_mm[-6.0]:7.3f} / {q.widths_mm[-20.0]:6.3f} {q.peak_db:+7.3f}"
            out.append(row)
        for d, r in self.reports.items():
            cp2 = [e for e in r.checkpoint2]
            rel = np.nanmean([e.relative_error_pct for e in cp2])
            ph = np.nanmean([e.phase_error_deg for e in cp2])
            out.append(
                f"  D={d}: checkpoint 1 {r.checkpoint1[1].relative_error_pct:.3f} %,"
                f" checkpoint 2 mean over scatterer lines {rel:.3f} % / {ph:.2f} deg,"
                f" image RMS {r.image.rms_db:.3f} dB max {r.image.max_db:.3f} dB"
                f" (floor {r.floor_pct:.4f} %)"
            )
        out.append("runtime: " + ", ".join(f"{k} {v:.1f} s" for k, v in self.seconds.items()))
        return out


def sweep(
    profile: ProbeProfile,
    events: list[TxEvent],
    records: list[RFEventRecord],
    scatterers: list[PointScatterer],
    *,
    decimations: tuple[int, ...] = DECIMATIONS,
    dtype=np.float32,
) -> SweepResult:
    seconds: dict[str, float] = {}
    t0 = time.perf_counter()
    golden = das_rf_golden(profile, events, records, dtype=dtype)
    seconds["golden"] = time.perf_counter() - t0
    rf_image, z, line_x = golden
    db_g = log_compress(envelope(rf_image))
    golden_psf = tuple(axial_psf(db_g, z, line_x, s) for s in scatterers)
    iq_psf: dict[int, tuple[AxialPSF, ...]] = {}
    reports: dict[int, ComparisonReport] = {}
    for d in decimations:
        t0 = time.perf_counter()
        iq_records = demodulate_frame(records, profile, decimation=d)
        image, _, _ = das_iq(profile, events, iq_records, decimation=d, dtype=dtype)
        seconds[f"iq path D={d}"] = time.perf_counter() - t0
        db_i = log_compress(np.abs(image))
        iq_psf[d] = tuple(axial_psf(db_i, z, line_x, s) for s in scatterers)
        reports[d] = compare(
            profile, events, records, scatterers, decimation=d, golden=golden, dtype=dtype
        )
    return SweepResult(profile.name, profile.bandwidth_status, golden_psf, iq_psf, reports, seconds)


def main() -> None:
    from enodia.demo import DEFAULT_SCATTERERS
    from enodia.spec.probe import linear_5mhz
    from enodia.spec.sequence import make_bmode_config
    from enodia.spec.sim import simulate_bmode_frame

    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events)
    records = simulate_bmode_frame(profile, events, DEFAULT_SCATTERERS, config_id=config.config_id)
    print("\n".join(sweep(profile, events, records, DEFAULT_SCATTERERS).lines()))


if __name__ == "__main__":
    main()
