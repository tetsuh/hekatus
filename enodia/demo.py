"""MVP-1 demo: point scatterers -> golden DAS -> B-mode image.

One command, one image, one acceptance test behind it. The point of an MVP
here is that the whole path runs end to end, not that any stage is finished.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from enodia.spec.beamform import das_rf_golden, envelope, log_compress
from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import make_bmode_sequence
from enodia.spec.sim import PointScatterer, simulate_bmode_frame

DEFAULT_SCATTERERS = [
    PointScatterer(0.0, 15e-3),
    PointScatterer(-5e-3, 25e-3),
    PointScatterer(7e-3, 40e-3),
]


def run_pipeline(
    profile: ProbeProfile,
    scatterers: list[PointScatterer],
    *,
    dynamic_range_db: float = 50.0,
):
    """Simulate, beamform, and compress. The acceptance test calls this too."""
    events = make_bmode_sequence(profile)
    records = simulate_bmode_frame(profile, events, scatterers, config_id=profile.name)
    rf_image, z, line_x = das_rf_golden(profile, events, records)
    db = log_compress(envelope(rf_image), dynamic_range_db=dynamic_range_db)
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
    parser = argparse.ArgumentParser(description="enodia MVP-1 demo")
    parser.add_argument("--out", type=Path, default=Path("out/bmode.png"))
    parser.add_argument("--dynamic-range-db", type=float, default=50.0)
    args = parser.parse_args()

    profile = linear_5mhz()
    scatterers = DEFAULT_SCATTERERS
    db, z, line_x, records = run_pipeline(
        profile, scatterers, dynamic_range_db=args.dynamic_range_db
    )
    save_png(
        db,
        z,
        line_x,
        scatterers,
        args.out,
        dynamic_range_db=args.dynamic_range_db,
        title=f"{profile.name} — RF golden DAS",
    )

    n_ch, n_t = records[0].data.shape
    print(f"sim: {len(scatterers)} point scatterers, {profile.name}")
    print(
        f"rf:  {len(records)} transmit events x {n_ch} ch x {n_t} samples"
        f" (int16, {profile.fs_hz / 1e6:.0f} MHz)"
    )
    print(f"das: RF golden (ideal delay, float32), dynamic aperture F={profile.f_number}")
    print(f"out: {args.out}")


if __name__ == "__main__":
    main()
