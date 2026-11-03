"""Seeded phantom generators for the analytic simulator.

A resolution cell has lateral width ``lambda * profile.f_number`` and axial
height ``c / (2 * profile.bandwidth_frac * profile.f0_hz)``. At density
``d`` scatterers per cell, the candidate count is
``ceil(d * imaging_region_area / resolution_cell_area)``. The imaging region
is the golden DAS scan-line extent by ``[z_min_m, profile.depth_m]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from enodia.spec.probe import ProbeProfile
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import PointScatterer

DEFAULT_Z_MIN_M = 2e-3
DEFAULT_SEED = 14


def imaging_region(
    profile: ProbeProfile, *, z_min_m: float = DEFAULT_Z_MIN_M
) -> tuple[float, float, float, float]:
    """Return ``(xmin, xmax, zmin, zmax)`` for the profile's golden DAS image.

    The lateral limits are the minimum and maximum ``line_x`` returned by
    ``das_rf_golden`` for the accepted standard B-mode configuration. Its
    identity contribution has exactly one line per event, so the accepted
    event coordinates define that same axis without synthesizing a frame.
    """
    if not math.isfinite(z_min_m) or not 0.0 <= z_min_m < profile.depth_m:
        raise ValueError("z_min_m must be finite and within the profile imaging depth")
    config = make_bmode_config(profile)
    line_x = np.asarray([event.line_x_m for event in config.events], dtype=np.float64)
    if line_x.size == 0 or not np.all(np.isfinite(line_x)):
        raise ValueError("the standard B-mode configuration has no finite imaging lines")
    return float(line_x.min()), float(line_x.max()), float(z_min_m), float(profile.depth_m)


def resolution_cell(profile: ProbeProfile) -> tuple[float, float]:
    """Return lateral/axial resolution-cell dimensions in metres."""
    return (
        profile.wavelength_m * profile.f_number,
        profile.c_m_s / (2.0 * profile.bandwidth_frac * profile.f0_hz),
    )


def _validate_density(density_per_cell: float) -> float:
    if isinstance(density_per_cell, (bool, np.bool_)) or not isinstance(
        density_per_cell, (int, float, np.integer, np.floating)
    ):
        raise TypeError("density_per_cell must be a real number")
    density = float(density_per_cell)
    if not math.isfinite(density) or density < 0.0:
        raise ValueError("density_per_cell must be finite and non-negative")
    return density


def _candidate_count(profile: ProbeProfile, density_per_cell: float, *, z_min_m: float) -> int:
    density = _validate_density(density_per_cell)
    xmin, xmax, zmin, zmax = imaging_region(profile, z_min_m=z_min_m)
    cell_width, cell_height = resolution_cell(profile)
    area = (xmax - xmin) * (zmax - zmin)
    return math.ceil(density * area / (cell_width * cell_height))


def _uniform_points(
    profile: ProbeProfile,
    rng: np.random.Generator,
    count: int,
    *,
    z_min_m: float,
    amplitude_std: float,
    excluded_circles: Sequence[tuple[float, float, float]] = (),
) -> list[PointScatterer]:
    xmin, xmax, zmin, zmax = imaging_region(profile, z_min_m=z_min_m)
    x = rng.uniform(xmin, xmax, count)
    z = rng.uniform(zmin, zmax, count)
    amplitude = rng.normal(0.0, amplitude_std, count)
    points: list[PointScatterer] = []
    for x_m, z_m, amp in zip(x, z, amplitude, strict=True):
        if any(math.hypot(float(x_m) - cx, float(z_m) - cz) < radius for cx, cz, radius in excluded_circles):
            continue
        points.append(PointScatterer(float(x_m), float(z_m), float(amp)))
    return points


def _validate_amplitude(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _validate_positions(name: str, values: Sequence[float], *, nonempty: bool = True) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if (nonempty and not result) or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite coordinates")
    return result


def _check_coordinates(
    profile: ProbeProfile, points: Sequence[tuple[float, float]], *, z_min_m: float
) -> None:
    xmin, xmax, zmin, zmax = imaging_region(profile, z_min_m=z_min_m)
    if any(not xmin <= x <= xmax or not zmin <= z <= zmax for x, z in points):
        raise ValueError("phantom geometry falls outside the profile imaging region")


def point_scatterer_array(
    profile: ProbeProfile,
    *,
    seed: int = DEFAULT_SEED,
    lateral_positions_m: Sequence[float] = (-8e-3, 0.0, 8e-3),
    z_start_m: float = 15e-3,
    z_stop_m: float = 45e-3,
    z_spacing_m: float = 5e-3,
    amplitude_ratio: float = 1.0,
    z_min_m: float = DEFAULT_Z_MIN_M,
) -> list[PointScatterer]:
    """Make point targets at lateral positions and regularly spaced depths.

    Target magnitudes are ``amplitude_ratio`` times the unit background
    reference; independent seeded random polarities exercise signed RF
    scatterer amplitudes while preserving the target geometry.
    """
    lateral = _validate_positions("lateral_positions_m", lateral_positions_m)
    ratio = _validate_amplitude(amplitude_ratio, "amplitude_ratio")
    if not all(math.isfinite(v) for v in (z_start_m, z_stop_m, z_spacing_m)):
        raise ValueError("point-array depth settings must be finite")
    if z_spacing_m <= 0.0 or z_stop_m < z_start_m:
        raise ValueError("z_spacing_m must be positive and z_stop_m >= z_start_m")
    depth_count = math.floor((z_stop_m - z_start_m) / z_spacing_m + 1e-12) + 1
    depths = tuple(z_start_m + i * z_spacing_m for i in range(depth_count))
    coordinates = [(x, z) for x in lateral for z in depths]
    _check_coordinates(profile, coordinates, z_min_m=z_min_m)
    rng = np.random.default_rng(seed)
    polarities = rng.choice(np.array([-1.0, 1.0]), size=len(coordinates))
    return [
        PointScatterer(x, z, float(sign * ratio))
        for (x, z), sign in zip(coordinates, polarities, strict=True)
    ]


def anechoic_cysts(
    profile: ProbeProfile,
    *,
    seed: int = DEFAULT_SEED,
    density_per_cell: float = 10.0,
    cyst_diameters_m: Sequence[float] = (2e-3, 4e-3, 8e-3),
    cyst_centers_x_m: Sequence[float] = (-10e-3, 0.0, 10e-3),
    cyst_centers_z_m: Sequence[float] = (20e-3, 40e-3),
    background_amplitude_std: float = 1.0,
    z_min_m: float = DEFAULT_Z_MIN_M,
) -> list[PointScatterer]:
    """Make seeded Gaussian speckle with anechoic cysts excluded.

    Each diameter is paired with the corresponding x coordinate, and each
    paired cyst is repeated at every configured depth. Random candidates are
    counted over the full imaging region using the resolution-cell formula;
    candidates strictly inside any cyst are removed without replenishment.
    """
    density = _validate_density(density_per_cell)
    diameters = _validate_positions("cyst_diameters_m", cyst_diameters_m)
    centers_x = _validate_positions("cyst_centers_x_m", cyst_centers_x_m)
    centers_z = _validate_positions("cyst_centers_z_m", cyst_centers_z_m)
    if len(diameters) != len(centers_x):
        raise ValueError("cyst_diameters_m and cyst_centers_x_m must have equal lengths")
    if any(diameter <= 0.0 for diameter in diameters):
        raise ValueError("cyst diameters must be positive")
    std = _validate_amplitude(background_amplitude_std, "background_amplitude_std")
    circles = [
        (x, z, diameter / 2.0)
        for x, diameter in zip(centers_x, diameters, strict=True)
        for z in centers_z
    ]
    _check_coordinates(
        profile,
        [(x + radius, z) for x, z, radius in circles]
        + [(x - radius, z) for x, z, radius in circles]
        + [(x, z + radius) for x, z, radius in circles]
        + [(x, z - radius) for x, z, radius in circles],
        z_min_m=z_min_m,
    )
    rng = np.random.default_rng(seed)
    count = _candidate_count(profile, density, z_min_m=z_min_m)
    return _uniform_points(
        profile,
        rng,
        count,
        z_min_m=z_min_m,
        amplitude_std=std,
        excluded_circles=circles,
    )


def uniform_speckle(
    profile: ProbeProfile,
    *,
    seed: int = DEFAULT_SEED,
    density_per_cell: float = 10.0,
    amplitude_std: float = 1.0,
    z_min_m: float = DEFAULT_Z_MIN_M,
) -> list[PointScatterer]:
    """Make a uniform random field with zero-mean Gaussian RF amplitudes.

    The number of points is ``ceil(density_per_cell * region_area /
    (cell_width * cell_height))``. Resolution-cell width is ``lambda * F#``;
    height is ``c / (2 * bandwidth_frac * f0)``. ``amplitude_std`` is the
    standard deviation of the zero-mean normal amplitude distribution.
    """
    density = _validate_density(density_per_cell)
    std = _validate_amplitude(amplitude_std, "amplitude_std", allow_zero=True)
    rng = np.random.default_rng(seed)
    return _uniform_points(
        profile,
        rng,
        _candidate_count(profile, density, z_min_m=z_min_m),
        z_min_m=z_min_m,
        amplitude_std=std,
    )


def high_echo_structures(
    profile: ProbeProfile,
    *,
    seed: int = DEFAULT_SEED,
    density_per_cell: float = 10.0,
    background_amplitude_std: float = 1.0,
    needle_center_x_m: float = 0.0,
    needle_center_z_m: float = 32e-3,
    needle_length_m: float = 10e-3,
    needle_angle_rad: float = math.radians(30.0),
    needle_spacing_m: float = 0.5e-3,
    needle_amplitude_ratio: float = 10.0,
    calcification_positions_m: Sequence[tuple[float, float]] = ((-12e-3, 32e-3), (12e-3, 38e-3)),
    calcification_amplitude_ratio: float = 20.0,
    z_min_m: float = DEFAULT_Z_MIN_M,
) -> list[PointScatterer]:
    """Make Gaussian speckle plus an angled needle and point calcifications.

    The needle is a line with regular spacing and random seeded polarity;
    calcifications use independent random seeded polarities. Structure
    amplitudes are ratios to the Gaussian background's standard-deviation
    reference. Background candidate count follows the resolution-cell formula.
    """
    density = _validate_density(density_per_cell)
    std = _validate_amplitude(background_amplitude_std, "background_amplitude_std")
    needle_ratio = _validate_amplitude(needle_amplitude_ratio, "needle_amplitude_ratio")
    calc_ratio = _validate_amplitude(calcification_amplitude_ratio, "calcification_amplitude_ratio")
    if not all(
        math.isfinite(v)
        for v in (needle_center_x_m, needle_center_z_m, needle_length_m, needle_angle_rad, needle_spacing_m)
    ):
        raise ValueError("needle geometry must be finite")
    if needle_length_m <= 0.0 or needle_spacing_m <= 0.0:
        raise ValueError("needle_length_m and needle_spacing_m must be positive")
    calcifications = tuple((float(x), float(z)) for x, z in calcification_positions_m)
    if not all(math.isfinite(v) for point in calcifications for v in point):
        raise ValueError("calcification positions must be finite")
    direction_x, direction_z = math.cos(needle_angle_rad), math.sin(needle_angle_rad)
    intervals = math.floor(needle_length_m / needle_spacing_m + 1e-12) + 1
    offsets = [(i - (intervals - 1) / 2.0) * needle_spacing_m for i in range(intervals)]
    needle = [
        (needle_center_x_m + offset * direction_x, needle_center_z_m + offset * direction_z)
        for offset in offsets
    ]
    _check_coordinates(
        profile,
        needle + list(calcifications),
        z_min_m=z_min_m,
    )
    rng = np.random.default_rng(seed)
    points = _uniform_points(
        profile,
        rng,
        _candidate_count(profile, density, z_min_m=z_min_m),
        z_min_m=z_min_m,
        amplitude_std=std,
    )
    polarities = rng.choice(np.array([-1.0, 1.0]), size=len(needle) + len(calcifications))
    amplitudes = [std * needle_ratio] * len(needle) + [std * calc_ratio] * len(calcifications)
    points.extend(
        PointScatterer(float(x), float(z), float(sign * amp))
        for (x, z), sign, amp in zip(needle + list(calcifications), polarities, amplitudes, strict=True)
    )
    return points


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_Z_MIN_M",
    "anechoic_cysts",
    "high_echo_structures",
    "imaging_region",
    "point_scatterer_array",
    "resolution_cell",
    "uniform_speckle",
]
