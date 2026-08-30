"""Demo: point scatterers -> RF golden DAS, or front end -> IQ DAS -> B-mode image.

One command, one image, one acceptance test behind it. The point of an MVP
here is that the whole path runs end to end, not that any stage is finished.
`--path golden` is MVP-1's RF-domain ideal-delay yardstick; `--path iq` is
the pipeline design.md §5 describes — complex band-pass FIR, decimation by
`--decimation`, integer shift + 4-tap interpolation + phase rotation (#6).
`python -m enodia.spec.beamform.golden_compare` is the second command that
compares the two, stage by stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from enodia.spec.beamform import das_rf_golden, envelope, log_compress
from enodia.spec.beamform.iq_das import das_iq
from enodia.spec.frontend import demodulate_frame
from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sequence.contribution import mla_map
from enodia.spec.sim import PointScatterer, simulate_frame

DEFAULT_SCATTERERS = [
    PointScatterer(0.0, 15e-3),
    PointScatterer(-5e-3, 25e-3),
    PointScatterer(7e-3, 40e-3),
]


PATHS = ("golden", "iq")
DEFAULT_DECIMATION = 8


def run_pipeline(
    profile: ProbeProfile,
    scatterers: list[PointScatterer],
    *,
    dynamic_range_db: float = 50.0,
    path: str = "golden",
    decimation: int = DEFAULT_DECIMATION,
    mla: int = 1,
):
    """Simulate, beamform, and compress. The acceptance tests call this too.

    ``path`` is "golden" (RF-domain ideal delay) or "iq" (front end at
    ``decimation``, then the IQ-domain DAS). Both return the log-compressed
    envelope on the same grid. ``mla`` forms that many receive lines per
    transmit, through the contribution map alone (#53): the beamformer call
    does not change, only the map handed to it. 1 is the conventional case
    and stays the default.
    """
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    config = make_bmode_config(profile)
    events = list(config.events)
    records = simulate_frame(profile, config, scatterers)
    contribution = mla_map(config, profile, mla=mla) if mla != 1 else None
    if path == "golden":
        rf_image, z, line_x = das_rf_golden(profile, events, records, contribution=contribution)
        env = envelope(rf_image)
    else:
        iq_records = demodulate_frame(records, profile, decimation=decimation)
        iq_image, z, line_x = das_iq(
            profile, events, iq_records, decimation=decimation, contribution=contribution
        )
        env = np.abs(iq_image)
    db = log_compress(env, dynamic_range_db=dynamic_range_db)
    return db, z, line_x, records


def save_png(
    db: np.ndarray,
    z: np.ndarray,
    line_x: np.ndarray,
    scatterers: list[PointScatterer],
    out_path: Path,
    *,
    dynamic_range_db: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(
        db,
        cmap="gray",
        aspect="equal",
        extent=(line_x[0] * 1e3, line_x[-1] * 1e3, z[-1] * 1e3, z[0] * 1e3),
        vmin=-dynamic_range_db,
        vmax=0.0,
    )
    ax.scatter(
        [s.x_m * 1e3 for s in scatterers],
        [s.z_m * 1e3 for s in scatterers],
        s=120,
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=0.8,
        label="true position",
    )
    ax.set_xlabel("lateral [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="enodia demo: RF golden or IQ path")
    parser.add_argument("--out", type=Path, default=Path("out/bmode.png"))
    parser.add_argument("--dynamic-range-db", type=float, default=50.0)
    parser.add_argument("--path", choices=PATHS, default="golden")
    parser.add_argument("--decimation", type=int, default=DEFAULT_DECIMATION)
    parser.add_argument("--mla", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()

    profile = linear_5mhz()
    scatterers = DEFAULT_SCATTERERS
    db, z, line_x, records = run_pipeline(
        profile,
        scatterers,
        dynamic_range_db=args.dynamic_range_db,
        path=args.path,
        decimation=args.decimation,
        mla=args.mla,
    )
    stage = (
        "RF golden DAS"
        if args.path == "golden"
        else f"front end D={args.decimation} + IQ DAS (Lagrange cubic)"
    )
    if args.mla != 1:
        stage += f", MLA {args.mla}"
    save_png(
        db,
        z,
        line_x,
        scatterers,
        args.out,
        dynamic_range_db=args.dynamic_range_db,
        title=f"{profile.name} — {stage}",
    )

    n_ch, n_t = records[0].data.shape
    print(f"sim: {len(scatterers)} point scatterers, {profile.name}")
    print(
        f"probe: {profile.name}, f0 {profile.f0_hz / 1e6:g} MHz, bandwidth_frac"
        f" {profile.bandwidth_frac:g} (full width at half amplitude; one-sided edge"
        f" {profile.bandwidth_edge_hz / 1e6:g} MHz) — {profile.bandwidth_status}"
        f", source: {profile.bandwidth_source or 'none'}"
    )
    print(
        f"cfg: config ID {records[0].header.config_id!r} -> probe profile"
        f" {profile.name!r} (design.md §19; bandwidth is the profile's only)"
    )
    print(
        f"rf:  {len(records)} transmit events x {n_ch} ch x {n_t} samples"
        f" (int16, {profile.fs_hz / 1e6:.0f} MHz)"
    )
    if args.path == "golden":
        print(f"das: RF golden (ideal delay, float32), dynamic aperture F={profile.f_number}")
    else:
        n_iq = n_t // args.decimation
        print(
            f"fe:  complex BPF (64 taps, Hann, cutoff fs/2D) + decimation by {args.decimation}"
            f" -> {n_ch} ch x {n_iq} IQ samples (int16 complex, {profile.fs_hz / 1e6 / args.decimation:g} MHz)"
        )
        print(
            "das: IQ (integer shift + Lagrange cubic + phase rotation, float32), dynamic"
            f" aperture F={profile.f_number}"
        )
    print(f"out: {args.out}")


if __name__ == "__main__":
    main()
