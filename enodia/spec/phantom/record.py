"""End-to-end CPU phantom metrics record for Issue #14.

Run ``uv run python -m enodia.spec.phantom.record --output-dir
 docs/measurements/issue14-phantom-metrics/YYYY-MM-DD`` on a clean, committed
checkout. Run with ``--estimate-only`` first to time a small full pipeline and
estimate the production workload; the harness refuses a production run whose
estimated total exceeds two hours. Output is write-once and confined to the
Issue #14 measurement tree.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from enodia.spec.beamform import das_rf_golden, envelope
from enodia.spec.eval.metrics import gcnr, speckle_snr, width_at
from enodia.spec.phantom import (
    DEFAULT_Z_MIN_M,
    anechoic_cysts,
    high_echo_structures,
    imaging_region,
    point_scatterer_array,
    resolution_cell,
    uniform_speckle,
)
from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import DEFAULT_PULSE_SUPPORT_SIGMAS, PointScatterer, simulate_frame

OUTPUT_JSON = "phantom-metrics.json"
OUTPUT_README = "README.md"
MEASUREMENT_ROOT = (
    Path(__file__).resolve().parents[3] / "docs/measurements/issue14-phantom-metrics"
)
MAX_ESTIMATED_SECONDS = 2 * 60 * 60
ESTIMATE_SCATTERER_COUNT = 64
FWHM_LEVEL_DB = float(20.0 * np.log10(0.5))
WIDTH_LEVELS_DB = (FWHM_LEVEL_DB, -40.0)


# The paired diameter/x coordinates make three non-overlapping cyst sizes at
# each of two depths: 2, 4 and 8 mm. All numerical settings are explicit in
# each case record.
def _default_cases(seed: int) -> dict[str, dict[str, Any]]:
    return {
        "point_scatterer_array": {
            "generator": point_scatterer_array,
            "parameters": {
                "seed": seed,
                "lateral_positions_m": (-8e-3, 0.0, 8e-3),
                "z_start_m": 15e-3,
                "z_stop_m": 45e-3,
                "z_spacing_m": 5e-3,
                "amplitude_ratio": 1.0,
                "z_min_m": DEFAULT_Z_MIN_M,
            },
        },
        "anechoic_cysts": {
            "generator": anechoic_cysts,
            "parameters": {
                "seed": seed + 1,
                "density_per_cell": 10.0,
                "cyst_diameters_m": (2e-3, 4e-3, 8e-3),
                "cyst_centers_x_m": (-10e-3, 0.0, 10e-3),
                "cyst_centers_z_m": (20e-3, 40e-3),
                "background_amplitude_std": 1.0,
                "z_min_m": DEFAULT_Z_MIN_M,
            },
        },
        "uniform_speckle": {
            "generator": uniform_speckle,
            "parameters": {
                "seed": seed + 2,
                "density_per_cell": 10.0,
                "amplitude_std": 1.0,
                "z_min_m": DEFAULT_Z_MIN_M,
            },
        },
        "high_echo_structures": {
            "generator": high_echo_structures,
            "parameters": {
                "seed": seed + 3,
                "density_per_cell": 10.0,
                "background_amplitude_std": 1.0,
                "needle_center_x_m": 0.0,
                "needle_center_z_m": 32e-3,
                "needle_length_m": 10e-3,
                "needle_angle_rad": math.radians(30.0),
                "needle_spacing_m": 0.5e-3,
                "needle_amplitude_ratio": 10.0,
                "calcification_positions_m": ((-12e-3, 32e-3), (12e-3, 38e-3)),
                "calcification_amplitude_ratio": 20.0,
                "z_min_m": DEFAULT_Z_MIN_M,
            },
        },
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _require_clean_checkout() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("phantom measurements require a clean committed checkout")


def _cpu_model() -> str:
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _uptime_load_average() -> dict[str, str]:
    raw = subprocess.run(["uptime"], check=True, capture_output=True, text=True).stdout.strip()
    lowered = raw.lower()
    for label in ("load average:", "load averages:"):
        prefix, separator, _ = lowered.partition(label)
        value = raw[len(prefix) + len(label) :].strip()
        if separator and value:
            # Deliberately retain only the load values and timestamp: uptime's
            # other fields may reveal user counts or host-specific details.
            return {"captured_at_utc": datetime.now(UTC).isoformat(), "load_average": value}
    raise RuntimeError("uptime output did not include a load average")


def _finite_json(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _profile_record(profile: ProbeProfile) -> dict[str, Any]:
    return asdict(profile)


def _roi_record() -> dict[str, Any]:
    return {
        "point_targets": {
            "peak_search_half_width_x_m": 3e-3,
            "peak_search_half_width_z_m": 2e-3,
            "profile_crop_half_width_x_m": 3e-3,
            "profile_crop_half_width_z_m": 2e-3,
            "levels_db": [FWHM_LEVEL_DB, -40.0],
        },
        "cysts": {"inside_radius_fraction": 0.75, "outside_radius_fractions": [1.25, 1.75], "gcnr_bins": 256},
        "uniform_speckle": {"x_bounds_m": [-10e-3, 10e-3], "z_bounds_m": [15e-3, 45e-3]},
        "high_echo": {
            "needle_half_width_m": 0.6e-3,
            "needle_outside_distance_m": [1.25e-3, 1.75e-3],
            "calcification_inside_radius_m": 0.6e-3,
            "calcification_outside_radius_m": [1.0e-3, 1.5e-3],
            "gcnr_bins": 256,
        },
    }


def _mask_region(
    image: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = image[mask]
    if values.size < 2:
        raise ValueError("metric region contains fewer than two envelope pixels")
    return values


def _gcnr_from_masks(image: np.ndarray, inside: np.ndarray, outside: np.ndarray) -> float | None:
    if np.count_nonzero(inside) < 2 or np.count_nonzero(outside) < 2:
        return None
    try:
        return _finite_json(gcnr(image[inside], image[outside], bins=256))
    except ValueError:
        return None


def _point_metrics(
    image: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    scatterers: list[PointScatterer],
) -> dict[str, Any]:
    targets = []
    for index, scatterer in enumerate(scatterers):
        x_mask = np.abs(x_axis - scatterer.x_m) <= 3e-3
        z_mask = np.abs(z_axis - scatterer.z_m) <= 2e-3
        if not np.any(x_mask) or not np.any(z_mask):
            peak = None
            lateral = {"fwhm_m": None, "width_40db_m": None}
            axial = {"fwhm_m": None, "width_40db_m": None}
        else:
            patch = image[np.ix_(z_mask, x_mask)]
            local_z, local_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
            z_indices = np.flatnonzero(z_mask)
            x_indices = np.flatnonzero(x_mask)
            peak_z = int(z_indices[local_z])
            peak_x = int(x_indices[local_x])
            peak = {"x_m": float(x_axis[peak_x]), "z_m": float(z_axis[peak_z]), "envelope": float(image[peak_z, peak_x])}
            lateral_profile = image[peak_z, x_mask]
            lateral_coords = x_axis[x_mask]
            axial_profile = image[z_mask, peak_x]
            axial_coords = z_axis[z_mask]
            widths = []
            for values, coords in ((lateral_profile, lateral_coords), (axial_profile, axial_coords)):
                row = []
                for level in WIDTH_LEVELS_DB:
                    try:
                        row.append(_finite_json(width_at(values, coords, level)))
                    except ValueError:
                        row.append(None)
                widths.append(row)
            lateral = {"fwhm_m": widths[0][0], "width_40db_m": widths[0][1]}
            axial = {"fwhm_m": widths[1][0], "width_40db_m": widths[1][1]}
        targets.append(
            {
                "target_index": index,
                "scatterer_x_m": scatterer.x_m,
                "scatterer_z_m": scatterer.z_m,
                "peak": peak,
                "lateral_fwhm_m": lateral["fwhm_m"],
                "lateral_width_40db_m": lateral["width_40db_m"],
                "axial_fwhm_m": axial["fwhm_m"],
                "axial_width_40db_m": axial["width_40db_m"],
            }
        )
    return {"targets": targets}


def _cyst_metrics(
    image: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    diameters = parameters["cyst_diameters_m"]
    centers_x = parameters["cyst_centers_x_m"]
    centers_z = parameters["cyst_centers_z_m"]
    cysts = []
    xx, zz = np.meshgrid(x_axis, z_axis)
    for x0, diameter in zip(centers_x, diameters, strict=True):
        radius = float(diameter) / 2.0
        for z0 in centers_z:
            distance = np.hypot(xx - float(x0), zz - float(z0))
            inside = distance <= 0.75 * radius
            outside = (distance >= 1.25 * radius) & (distance <= 1.75 * radius)
            cysts.append(
                {
                    "center_x_m": float(x0),
                    "center_z_m": float(z0),
                    "diameter_m": float(diameter),
                    "inside_radius_m": 0.75 * radius,
                    "outside_radius_m": [1.25 * radius, 1.75 * radius],
                    "gcnr": _gcnr_from_masks(image, inside, outside),
                    "inside_pixel_count": int(np.count_nonzero(inside)),
                    "outside_pixel_count": int(np.count_nonzero(outside)),
                }
            )
    return {"cysts": cysts}


def _uniform_metrics(image: np.ndarray, x_axis: np.ndarray, z_axis: np.ndarray) -> dict[str, Any]:
    x_bounds = (-10e-3, 10e-3)
    z_bounds = (15e-3, 45e-3)
    mask = (
        (z_axis[:, None] >= z_bounds[0])
        & (z_axis[:, None] <= z_bounds[1])
        & (x_axis[None, :] >= x_bounds[0])
        & (x_axis[None, :] <= x_bounds[1])
    )
    values = image[mask]
    try:
        snr = _finite_json(speckle_snr(values))
    except ValueError:
        snr = None
    return {
        "speckle_snr": snr,
        "roi": {"x_bounds_m": list(x_bounds), "z_bounds_m": list(z_bounds), "pixel_count": int(values.size)},
    }


def _high_echo_metrics(
    image: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    xx, zz = np.meshgrid(x_axis, z_axis)
    x0 = float(parameters["needle_center_x_m"])
    z0 = float(parameters["needle_center_z_m"])
    length = float(parameters["needle_length_m"])
    angle = float(parameters["needle_angle_rad"])
    along = (xx - x0) * math.cos(angle) + (zz - z0) * math.sin(angle)
    perpendicular = np.abs(-(xx - x0) * math.sin(angle) + (zz - z0) * math.cos(angle))
    needle_inside = (np.abs(along) <= length / 2.0) & (perpendicular <= 0.6e-3)
    needle_outside = (
        (np.abs(along) <= length / 2.0)
        & (perpendicular >= 1.25e-3)
        & (perpendicular <= 1.75e-3)
    )
    needle = {
        "center_x_m": x0,
        "center_z_m": z0,
        "length_m": length,
        "angle_rad": angle,
        "inside_half_width_m": 0.6e-3,
        "outside_distance_m": [1.25e-3, 1.75e-3],
        "gcnr": _gcnr_from_masks(image, needle_inside, needle_outside),
        "inside_pixel_count": int(np.count_nonzero(needle_inside)),
        "outside_pixel_count": int(np.count_nonzero(needle_outside)),
    }
    calcifications = []
    for cx, cz in parameters["calcification_positions_m"]:
        distance = np.hypot(xx - float(cx), zz - float(cz))
        inside = distance <= 0.6e-3
        outside = (distance >= 1.0e-3) & (distance <= 1.5e-3)
        calcifications.append(
            {
                "center_x_m": float(cx),
                "center_z_m": float(cz),
                "inside_radius_m": 0.6e-3,
                "outside_radius_m": [1.0e-3, 1.5e-3],
                "gcnr": _gcnr_from_masks(image, inside, outside),
                "inside_pixel_count": int(np.count_nonzero(inside)),
                "outside_pixel_count": int(np.count_nonzero(outside)),
            }
        )
    return {"needle": needle, "calcifications": calcifications}


def _case_metrics(
    name: str,
    image: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    scatterers: list[PointScatterer],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if name == "point_scatterer_array":
        return _point_metrics(image, x_axis, z_axis, scatterers)
    if name == "anechoic_cysts":
        return _cyst_metrics(image, x_axis, z_axis, parameters)
    if name == "uniform_speckle":
        return _uniform_metrics(image, x_axis, z_axis)
    if name == "high_echo_structures":
        return _high_echo_metrics(image, x_axis, z_axis, parameters)
    # Custom small cases still record an applicable texture metric. This path
    # exists for fast structural tests and is not used by the production CLI.
    return _uniform_metrics(image, x_axis, z_axis)


def _run_pipeline(
    profile: ProbeProfile, scatterers: list[PointScatterer], *, z_min_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = make_bmode_config(profile)
    records = simulate_frame(
        profile,
        config,
        scatterers,
        pulse_support_sigmas=DEFAULT_PULSE_SUPPORT_SIGMAS,
    )
    rf_image, z_axis, x_axis = das_rf_golden(
        profile, list(config.events), records, z_min_m=z_min_m
    )
    return envelope(rf_image), z_axis, x_axis


def build_record(
    *,
    profile: ProbeProfile | None = None,
    seed: int = 14,
    cases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure supplied/default phantoms through simulator, golden DAS, envelope."""
    profile = linear_5mhz() if profile is None else profile
    case_specs = _default_cases(seed) if cases is None else cases
    xmin, xmax, zmin, zmax = imaging_region(profile)
    cell_width, cell_height = resolution_cell(profile)
    results = []
    for name, case in case_specs.items():
        generator = case["generator"]
        parameters = dict(case["parameters"])
        started_load = _uptime_load_average()
        started = time.perf_counter()
        scatterers = generator(profile, **parameters)
        image, z_axis, x_axis = _run_pipeline(profile, scatterers, z_min_m=zmin)
        metrics = _case_metrics(name, image, x_axis, z_axis, scatterers, parameters)
        elapsed = time.perf_counter() - started
        ended_load = _uptime_load_average()
        results.append(
            {
                "name": name,
                "generator": generator.__name__,
                "generator_arguments": _json_safe(parameters),
                "scatterer_count": len(scatterers),
                "elapsed_seconds": float(elapsed),
                "uptime_load_average": {"before": started_load, "after": ended_load},
                "metrics": _json_safe(metrics),
            }
        )
        del scatterers, image
    sanity = evaluate_sanity(results)
    return {
        "issue": 14,
        "measurement_date_utc": datetime.now(UTC).date().isoformat(),
        "measurement_type": "board-free CPU phantom metrics",
        "environment": {
            "commit": _git("rev-parse", "HEAD"),
            "cpu_model": _cpu_model(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "profile": _profile_record(profile),
        "processing": {
            "pipeline": ["simulate_frame", "das_rf_golden", "envelope"],
            "pulse_support_sigmas": DEFAULT_PULSE_SUPPORT_SIGMAS,
            "envelope_domain": "linear envelope before log compression",
            "metric_domain": "linear envelope before log compression",
            "z_min_m": zmin,
            "imaging_region_m": {"x": [xmin, xmax], "z": [zmin, zmax]},
        },
        "definitions": {
            "resolution_cell": {
                "lateral_width_m": cell_width,
                "axial_height_m": cell_height,
                "lateral_formula": "lambda * profile.f_number; lambda = c / f0",
                "axial_formula": "c / (2 * profile.bandwidth_frac * f0)",
            },
            "density_count_formula": "ceil(density_per_cell * imaging_region_area / (resolution_cell_lateral_width * resolution_cell_axial_height))",
            "cyst_exclusion": "discard random candidate points whose center distance is strictly less than cyst radius; do not replenish",
            "background_amplitude": "zero-mean Gaussian; amplitude_std is configurable",
        },
        "metric_roi_arguments": _roi_record(),
        "cases": results,
        "sanity": sanity,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_json(float(value))
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def evaluate_sanity(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the three Issue #14 record bounds without suppressing a result."""
    checks = []
    uniform = next((case for case in cases if case["name"] == "uniform_speckle"), None)
    snr = None if uniform is None else uniform["metrics"].get("speckle_snr")
    checks.append(
        {
            "name": "uniform_speckle_snr_inclusive_1.5_to_2.3",
            "status": "pass" if snr is not None and 1.5 <= snr <= 2.3 else "fail",
            "observed": snr,
            "bounds": [1.5, 2.3],
        }
    )
    points = next((case for case in cases if case["name"] == "point_scatterer_array"), None)
    targets = [] if points is None else points["metrics"].get("targets", [])
    fwhms = [target.get("lateral_fwhm_m") for target in targets]
    checks.append(
        {
            "name": "all_point_targets_have_finite_lateral_fwhm",
            "status": "pass" if fwhms and all(value is not None and math.isfinite(value) for value in fwhms) else "fail",
            "observed": fwhms,
        }
    )
    cyst_case = next((case for case in cases if case["name"] == "anechoic_cysts"), None)
    large_cysts = [] if cyst_case is None else [
        cyst for cyst in cyst_case["metrics"].get("cysts", []) if math.isclose(cyst.get("diameter_m", 0.0), 8e-3)
    ]
    checks.append(
        {
            "name": "all_8mm_cysts_gcnr_above_0.5",
            "status": "pass" if large_cysts and all(cyst.get("gcnr") is not None and cyst["gcnr"] > 0.5 for cyst in large_cysts) else "fail",
            "observed": [cyst.get("gcnr") for cyst in large_cysts],
            "lower_exclusive_bound": 0.5,
        }
    )
    return {"status": "pass" if all(check["status"] == "pass" for check in checks) else "fail", "checks": checks}


def _render_readme(record: dict[str, Any]) -> str:
    env = record["environment"]
    profile = record["profile"]
    lines = [
        "# Issue #14 phantom metrics",
        "",
        "Board-free CPU measurement record generated by `enodia.spec.phantom.record`; this README is rendered from the JSON companion.",
        "",
        "## What was measured",
        "",
        "Each phantom was passed through `simulate_frame` → `das_rf_golden` → `envelope` using the `linear_5mhz` profile. Metrics use the linear envelope before log compression; no dB image or CNR is used.",
        "",
        f"- Harness commit: `{env['commit']}`",
        f"- CPU model: `{env['cpu_model']}`",
        f"- Python / NumPy: `{env['python_version']}` / `{env['numpy_version']}`",
        f"- Profile: `{profile['name']}` (bandwidth fraction {profile['bandwidth_frac']:g}; source {profile['bandwidth_source'] or 'provisional'})",
        f"- Imaging region: x {record['processing']['imaging_region_m']['x']} m; z {record['processing']['imaging_region_m']['z']} m",
        f"- Resolution cell: lateral {record['definitions']['resolution_cell']['lateral_width_m']:.9g} m; axial {record['definitions']['resolution_cell']['axial_height_m']:.9g} m",
        f"- Resolution-cell density count: `{record['definitions']['density_count_formula']}`",
        f"- Cyst exclusions: {record['definitions']['cyst_exclusion']}",
        f"- Speckle amplitude model: {record['definitions']['background_amplitude']}",
        "",
        "## Results",
        "",
        "| Phantom | Scatterers | Elapsed (s) | Metrics recorded |",
        "| --- | ---: | ---: | --- |",
    ]
    for case in record["cases"]:
        if "speckle_snr" in case["metrics"]:
            metric_summary = f"speckle SNR {_display_number(case['metrics']['speckle_snr'])}"
        elif "targets" in case["metrics"]:
            targets = case["metrics"]["targets"]
            widths = [target["lateral_fwhm_m"] for target in targets if target["lateral_fwhm_m"] is not None]
            width_summary = ", ".join(_display_number(value) for value in widths)
            metric_summary = f"lateral FWHM (m): [{width_summary}]; −40 dB widths also in JSON"
        elif "cysts" in case["metrics"]:
            values = [
                f"{cyst['diameter_m'] * 1e3:g} mm: {_display_number(cyst['gcnr'])}"
                for cyst in case["metrics"]["cysts"]
            ]
            metric_summary = "gCNR " + "; ".join(values)
        else:
            needle = _display_number(case["metrics"].get("needle", {}).get("gcnr"))
            calcifications = ", ".join(
                _display_number(item.get("gcnr")) for item in case["metrics"].get("calcifications", [])
            )
            metric_summary = f"needle gCNR {needle}; calcification gCNR [{calcifications}]"
        lines.append(f"| {case['name']} | {case['scatterer_count']} | {case['elapsed_seconds']:.6f} | {metric_summary} |")
    lines.extend(
        [
            "",
            f"Sanity-bound evaluation: **{record['sanity']['status']}**. Individual observations and the complete generator/ROI arguments are in the JSON record.",
            "",
            "The record reports the measurements performed; it makes no performance claim.",
            "",
            "Load averages were captured with `uptime` immediately before and after each phantom pipeline run. No host name, user name, or absolute path is included.",
            "",
            "The README and JSON are committed unchanged from the measurement output.",
            "",
        ]
    )
    return "\n".join(lines)


def _display_number(value: float | None) -> str:
    return "null" if value is None else f"{value:.6g}"


def render_readme(record: dict[str, Any]) -> str:
    """Public wrapper for deterministic README rendering from a JSON object."""
    return _render_readme(record)


def _validate_output_dir(output_dir: Path) -> tuple[Path, Path]:
    root = MEASUREMENT_ROOT.resolve()
    requested = output_dir if output_dir.is_absolute() else Path.cwd() / output_dir
    resolved = requested.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("output must be a date subdirectory of the Issue #14 measurement directory")
    if resolved.parent != root or len(resolved.name) != 10 or resolved.name[4:5] != "-" or resolved.name[7:8] != "-":
        raise ValueError("output must be a date subdirectory of the Issue #14 measurement directory")
    try:
        date.fromisoformat(resolved.name)
    except ValueError as error:
        raise ValueError("output must be a date subdirectory of the Issue #14 measurement directory") from error
    json_path = resolved / OUTPUT_JSON
    readme_path = resolved / OUTPUT_README
    if resolved.exists() or resolved.is_symlink() or any(
        item.exists() or item.is_symlink() for item in (json_path, readme_path)
    ):
        raise FileExistsError(f"refusing to overwrite measurement output in {resolved}")
    return resolved, root


def write_record(output_dir: Path, record: dict[str, Any]) -> tuple[Path, Path]:
    """Atomically create a dated write-once JSON/README record."""
    destination, root = _validate_output_dir(Path(output_dir))
    root.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite measurement output in {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    try:
        json_path = temporary / OUTPUT_JSON
        readme_path = temporary / OUTPUT_README
        json_record = _json_safe(record)
        json_path.write_text(json.dumps(json_record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        readme_path.write_text(_render_readme(json_record), encoding="utf-8")
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return destination / OUTPUT_JSON, destination / OUTPUT_README


def estimate_project(*, profile: ProbeProfile | None = None, seed: int = 14) -> dict[str, Any]:
    """Time a 64-point full pipeline and extrapolate the four case workloads."""
    profile = linear_5mhz() if profile is None else profile
    cases = _default_cases(seed)
    counts = {}
    for name, case in cases.items():
        points = case["generator"](profile, **case["parameters"])
        counts[name] = len(points)
        del points
    xmin, xmax, zmin, zmax = imaging_region(profile)
    region_area = (xmax - xmin) * (zmax - zmin)
    cell_width, cell_height = resolution_cell(profile)
    sample_density = 63.5 * cell_width * cell_height / region_area
    sample = uniform_speckle(profile, seed=seed, density_per_cell=sample_density)
    sample_count = len(sample)
    started = time.perf_counter()
    image, _, _ = _run_pipeline(profile, sample, z_min_m=zmin)
    elapsed = time.perf_counter() - started
    del sample, image
    per_scatterer = elapsed / sample_count
    estimates = {
        name: {
            "scatterer_count": count,
            "estimated_seconds": elapsed * count / sample_count,
        }
        for name, count in counts.items()
    }
    total = sum(row["estimated_seconds"] for row in estimates.values())
    return {
        "profile": profile.name,
        "sample_scatterer_count": sample_count,
        "sample_full_pipeline_seconds": elapsed,
        "measured_seconds_per_scatterer": per_scatterer,
        "estimated_cases": estimates,
        "estimated_total_seconds": total,
        "estimated_total_hours": total / 3600.0,
        "maximum_allowed_seconds": MAX_ESTIMATED_SECONDS,
        "production_allowed": total <= MAX_ESTIMATED_SECONDS,
    }


def run_record(
    output_dir: Path, *, seed: int = 14, estimate_only: bool = False
) -> tuple[Path, Path] | dict[str, Any]:
    """Preflight then, only within the two-hour limit, take and write a record."""
    output_dir = Path(output_dir)
    _validate_output_dir(output_dir)
    if estimate_only:
        return estimate_project(seed=seed)
    _require_clean_checkout()
    estimate = estimate_project(seed=seed)
    if not estimate["production_allowed"]:
        raise RuntimeError(
            "estimated phantom measurement exceeds two hours; production was not run: "
            + json.dumps(estimate, sort_keys=True)
        )
    result = build_record(seed=seed)
    result["preflight_estimate"] = estimate
    written = write_record(output_dir, result)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.estimate_only:
            print(json.dumps(estimate_project(seed=args.seed), indent=2, allow_nan=False))
            return 0
        result = run_record(args.output_dir, seed=args.seed)
        assert isinstance(result, tuple)
        print(result[0].as_posix())
        print(result[1].as_posix())
        # The record is persisted even when a sanity bound fails.
        record = json.loads(result[0].read_text(encoding="utf-8"))
        return 0 if record["sanity"]["status"] == "pass" else 1
    except (RuntimeError, ValueError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
