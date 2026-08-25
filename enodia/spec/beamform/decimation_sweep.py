"""Axial point-spread function of the IQ path at D=8 against D=4 (design.md §5, §17, #6).

design.md §5 asks, in those words, to "keep the ability to measure how the
point-scatterer axial PSF changes between decimation ratios 8 and 4", and
§17 keeps "decimation ratio and interpolation tap count" as a parameter
decided by measurement. This is that measurement, on the point-scatterer
phantom the demo uses, for the `linear-5mhz` profile — whose bandwidth is
provisional (§4), so the figures are labelled with that status and are
rerun if the profile's value or provenance changes.

For each scatterer the axial profile is the log envelope along the
scanline through its lateral peak; the widths are the full widths at −6,
−20 and −40 dB below the profile's peak — design.md §15 says never to argue
resolution from the −6 dB width alone and to check −40 dB too — found by
linear interpolation of the crossings either side, in millimetres. The golden's own widths are the
reference; the IQ path's are given at each decimation ratio, with the
peak level relative to the golden's and the checkpoint errors of
`golden_compare` beside them.

**The result is a measurement record** (ADR-0005): `--record PATH` writes
the figures as data under `docs/measurements/`, with the environment that
produced them — host, CPU, kernel, Python, NumPy, SciPy, the harness
revision and whether the tree was dirty — and the provenance of the profile
they were taken on. A figure quoted in design.md names that record.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
WIDTH_LEVELS_DB: tuple[float, ...] = (-6.0, -20.0, -40.0)


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
            " full widths at -6 / -20 / -40 dB [mm], peak relative to frame max [dB]"
        )
        out = [title]
        out.append(
            f"  {'scatterer (x, z) mm':22s} {'golden':>34s}"
            + "".join(f"{f'IQ D={d}':>34s}" for d in self.iq)
        )

        def cell(p: AxialPSF) -> str:
            return (
                f"{p.widths_mm[-6.0]:7.3f} / {p.widths_mm[-20.0]:6.3f} / {p.widths_mm[-40.0]:6.3f}"
                f" {p.peak_db:+7.3f}"
            )

        for i, g in enumerate(self.golden):
            row = f"  ({g.scatterer[0] * 1e3:+.0f}, {g.scatterer[1] * 1e3:.0f}){'':14s}{cell(g)}"
            for psfs in self.iq.values():
                row += f"   {cell(psfs[i])}"
            out.append(row)
        for d, r in self.reports.items():
            cp2 = list(r.checkpoint2)
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


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict:
    """The environment block ADR-0005 asks for, for a host-side measurement:
    no board, so the host, its kernel and the numeric stack stand in."""
    import numpy
    import scipy

    harness_commit = _git("rev-parse", "HEAD")
    harness_status = _git("status", "--porcelain")
    return {
        "captured_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "harness_commit": None if harness_commit == "unknown" else harness_commit,
        "harness_dirty": None if harness_status == "unknown" else bool(harness_status),
        "board": None,
        "note": "host-side reference implementation; no accelerator involved",
    }


def json_safe(obj):
    """Recursively turn a result into strict JSON: non-finite floats become
    null (a silent reference, an empty region or a crossing that never
    happens is NaN in the report), NumPy scalars become Python scalars,
    tuples become lists. `json.dumps(..., allow_nan=False)` then cannot emit
    the non-standard `NaN` / `Infinity` tokens."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, np.bool_)):
        return obj.item()
    return obj


def measurement_record(result: SweepResult, profile: ProbeProfile) -> dict:
    """The sweep and the per-stage comparison as data (ADR-0005), strict-JSON
    ready: pass it to `json.dumps(..., allow_nan=False)`."""

    def psf(p: AxialPSF) -> dict:
        return {
            "scatterer_x_m": p.scatterer[0],
            "scatterer_z_m": p.scatterer[1],
            "peak_z_m": p.peak_z_m,
            "peak_db": p.peak_db,
            "full_width_mm": {f"{lvl:g}": w for lvl, w in p.widths_mm.items()},
        }

    def report(r: ComparisonReport) -> dict:
        return {
            "decimation": r.decimation,
            "yardstick_floor_pct": r.floor_pct,
            "checkpoint1": [asdict(e) for e in r.checkpoint1],
            "checkpoint2": {
                "events": list(r.checkpoint2_events),
                "stages": [asdict(e) for e in r.checkpoint2],
            },
            "image": {
                "rms_db": r.image.rms_db,
                "max_db": r.image.max_db,
                "region_floor_db": r.image.region_floor_db,
                "peaks": [
                    {
                        "scatterer": list(pk.scatterer),
                        "golden_z_m_x_m_db": list(pk.golden),
                        "iq_z_m_x_m_db": list(pk.iq),
                    }
                    for pk in r.image.peaks
                ],
            },
            "seconds": r.seconds,
        }

    return json_safe(
        {
            "environment": environment(),
            "what": (
                "IQ path (front end + IQ DAS) against the RF golden: per-stage errors and"
                " axial PSF at D=8 and D=4 (#6)"
            ),
            "profile": {
                "name": profile.name,
                "f0_hz": profile.f0_hz,
                "fs_hz": profile.fs_hz,
                "bandwidth_frac": profile.bandwidth_frac,
                "bandwidth_edge_hz": profile.bandwidth_edge_hz,
                "bandwidth_status": profile.bandwidth_status,
                "bandwidth_source": profile.bandwidth_source,
            },
            "frozen_oracle": "rf-oracle-frozen-0p7",
            "front_end": {"n_taps": 64, "window": "hann", "cutoff_frac": 1.0, "iq_scale": 1.0},
            "kernel": "lagrange4",
            "width_levels_db": list(WIDTH_LEVELS_DB),
            "golden_psf": [psf(p) for p in result.golden],
            "iq_psf": {str(d): [psf(p) for p in ps] for d, ps in result.iq.items()},
            "comparison": {str(d): report(r) for d, r in result.reports.items()},
            "seconds": result.seconds,
        }
    )


def main() -> None:
    import argparse

    from enodia.demo import DEFAULT_SCATTERERS
    from enodia.spec.probe import linear_5mhz
    from enodia.spec.sequence import make_bmode_config
    from enodia.spec.sim import simulate_bmode_frame

    parser = argparse.ArgumentParser(description="axial PSF and per-stage errors, D=8 vs D=4")
    parser.add_argument(
        "--record", type=Path, default=None, help="write the measurement record here"
    )
    args = parser.parse_args()

    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events)
    records = simulate_bmode_frame(profile, events, DEFAULT_SCATTERERS, config_id=config.config_id)
    result = sweep(profile, events, records, DEFAULT_SCATTERERS)
    print("\n".join(result.lines()))
    if args.record is not None:
        # The record lands beneath the current directory — docs/measurements/
        # when run from the repository root — never outside it, whatever the
        # argument says.
        root = Path.cwd().resolve()
        out = args.record.resolve()
        if not out.is_relative_to(root):
            raise SystemExit(f"--record must point beneath the current directory {root}, got {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        record = measurement_record(result, profile)
        out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(f"record: {out.relative_to(root)}")


if __name__ == "__main__":
    main()
