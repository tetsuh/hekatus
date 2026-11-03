"""Issue #14 phantom generators and end-to-end measurement records."""

from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import enodia.spec.phantom as phantom_generators
from enodia.spec.beamform import das_rf_golden
from enodia.spec.phantom import (
    anechoic_cysts,
    high_echo_structures,
    imaging_region,
    point_scatterer_array,
    record,
    resolution_cell,
    uniform_speckle,
)
from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim import PointScatterer, simulate_frame


def _scatterer_tuples(scatterers):
    return [(point.x_m, point.z_m, point.amplitude) for point in scatterers]


def _small_profile():
    return ProbeProfile(
        name="test-linear",
        n_elements=8,
        pitch_m=0.3e-3,
        f0_hz=5.0e6,
        bandwidth_frac=0.7,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=18e-3,
        tx_focus_m=12e-3,
        f_number=2.0,
    )


def _golden_line_region(profile):
    config = make_bmode_config(profile)
    records = simulate_frame(profile, config, [PointScatterer(0.0, 10e-3, 0.0)])
    _, _, line_x = das_rf_golden(profile, list(config.events), records, z_min_m=2e-3)
    return line_x


def test_all_generators_are_seeded_and_keep_points_inside_golden_imaging_region():
    profile = _small_profile()
    line_x = _golden_line_region(profile)
    xmin, xmax = float(line_x.min()), float(line_x.max())
    zmin = 2e-3
    generators = [
        lambda seed: point_scatterer_array(
            profile, seed=seed, lateral_positions_m=(-0.5e-3, 0.5e-3), z_start_m=5e-3, z_stop_m=10e-3
        ),
        lambda seed: anechoic_cysts(
            profile, seed=seed, density_per_cell=0.2, cyst_diameters_m=(0.2e-3, 0.4e-3, 0.8e-3),
            cyst_centers_x_m=(-0.5e-3, 0.0, 0.5e-3), cyst_centers_z_m=(5e-3, 12e-3),
            background_x_bounds_m=(-1e-3, 1e-3), background_z_bounds_m=(3e-3, 17e-3)
        ),
        lambda seed: uniform_speckle(
            profile, seed=seed, density_per_cell=0.2,
            background_x_bounds_m=(-1e-3, 1e-3), background_z_bounds_m=(3e-3, 17e-3)
        ),
        lambda seed: high_echo_structures(
            profile, seed=seed, density_per_cell=0.2, needle_center_m=(0.0, 9e-3), needle_length_m=1e-3,
            calcification_positions_m=((-0.7e-3, 6e-3), (0.7e-3, 12e-3)),
            background_x_bounds_m=(-1e-3, 1e-3), background_z_bounds_m=(3e-3, 17e-3)
        ),
    ]
    for generate in generators:
        first = generate(14)
        repeated = generate(14)
        different = generate(15)
        assert first
        assert all(isinstance(point, PointScatterer) for point in first)
        assert _scatterer_tuples(first) == _scatterer_tuples(repeated)
        assert _scatterer_tuples(first) != _scatterer_tuples(different)
        assert all(xmin <= point.x_m <= xmax for point in first)
        assert all(zmin <= point.z_m <= profile.depth_m for point in first)
    region = imaging_region(profile)
    assert (region[0], region[1]) == pytest.approx((xmin, xmax))
    assert region[2:] == (zmin, profile.depth_m)


def test_cyst_generator_excludes_strict_interior_and_uses_configured_radii():
    profile = linear_5mhz()
    centers_x = (-0.010, -0.003, 0.006)
    centers_z = (0.022, 0.040)
    diameters = (0.002, 0.004, 0.008)
    points = anechoic_cysts(
        profile,
        seed=1,
        density_per_cell=2.0,
        cyst_diameters_m=diameters,
        cyst_centers_x_m=centers_x,
        cyst_centers_z_m=centers_z,
    )
    for x0, diameter in zip(centers_x, diameters, strict=True):
        radius = diameter / 2
        for z0 in centers_z:
            for point in points:
                assert math.hypot(point.x_m - x0, point.z_m - z0) >= radius

    class BoundaryRng:
        def __init__(self):
            self.uniform_calls = 0

        def uniform(self, low, high, count):
            self.uniform_calls += 1
            return np.array([0.001 if self.uniform_calls == 1 else 0.022])

        def normal(self, mean, std, count):
            return np.array([1.0])

    boundary = phantom_generators._uniform_points(
        profile,
        BoundaryRng(),
        1,
        background_x_bounds_m=(-15e-3, 15e-3),
        background_z_bounds_m=(10e-3, 50e-3),
        z_min_m=2e-3,
        amplitude_std=1.0,
        excluded_circles=[(0.0, 0.022, 0.001)],
    )
    assert len(boundary) == 1
    assert math.hypot(boundary[0].x_m, boundary[0].z_m - 0.022) == 0.001


def test_speckle_count_uses_the_documented_resolution_cell_density_formula():
    profile = linear_5mhz()
    density = 0.25
    bounds_x = (-15e-3, 15e-3)
    bounds_z = (10e-3, 50e-3)
    cell_width, cell_height = resolution_cell(profile)
    expected = math.ceil(density * (bounds_x[1] - bounds_x[0]) * (bounds_z[1] - bounds_z[0]) / (cell_width * cell_height))
    assert len(uniform_speckle(profile, seed=5, density_per_cell=density)) == expected


def test_generator_parameters_change_geometry_and_amplitudes():
    profile = linear_5mhz()
    default = point_scatterer_array(profile, seed=2)
    shifted = point_scatterer_array(
        profile, seed=2, lateral_positions_m=(-0.006, 0.0, 0.006), amplitude_ratio=4.0
    )
    assert len(default) == len(shifted)
    assert [point.z_m for point in default] == [point.z_m for point in shifted]
    assert [point.x_m for point in default] != [point.x_m for point in shifted]
    assert all(abs(b.amplitude) == 4.0 * abs(a.amplitude) for a, b in zip(default, shifted, strict=True))

    low = uniform_speckle(profile, seed=3, density_per_cell=0.1)
    high = uniform_speckle(profile, seed=3, density_per_cell=0.2)
    assert len(high) in {2 * len(low) - 1, 2 * len(low)}
    assert np.mean([point.amplitude for point in high]) == pytest.approx(0.0, abs=0.15)
    assert high_echo_structures(
        profile, seed=4, density_per_cell=0.05, structure_amplitude_ratios=(12.0, 20.0)
    ) != high_echo_structures(
        profile, seed=4, density_per_cell=0.05, structure_amplitude_ratios=(24.0, 20.0)
    )


def test_high_echo_ratios_scale_with_background_standard_deviation():
    profile = linear_5mhz()
    arguments = {"seed": 14, "density_per_cell": 0.01}
    unit = high_echo_structures(profile, **arguments, background_amplitude_std=1.0)
    scaled = high_echo_structures(profile, **arguments, background_amplitude_std=3.0)
    assert [(p.x_m, p.z_m) for p in unit] == [(p.x_m, p.z_m) for p in scaled]
    np.testing.assert_allclose(
        [p.amplitude for p in scaled], 3.0 * np.array([p.amplitude for p in unit])
    )


def test_high_echo_grouped_settings_preserve_defaults_and_remain_adjustable():
    profile = linear_5mhz()
    assert len(inspect.signature(high_echo_structures).parameters) <= 13
    default = high_echo_structures(profile, seed=27, density_per_cell=0.01)
    explicit_defaults = high_echo_structures(
        profile,
        seed=27,
        density_per_cell=0.01,
        needle_center_m=(0.0, 32e-3),
        structure_amplitude_ratios=(10.0, 20.0),
    )
    assert _scatterer_tuples(default) == _scatterer_tuples(explicit_defaults)

    moved = high_echo_structures(
        profile,
        seed=27,
        density_per_cell=0.01,
        needle_center_m=(1e-3, 34e-3),
        structure_amplitude_ratios=(12.0, 24.0),
    )
    background_count = len(default) - 23
    assert _scatterer_tuples(default[:background_count]) == _scatterer_tuples(moved[:background_count])
    default_needle = default[background_count : background_count + 21]
    moved_needle = moved[background_count : background_count + 21]
    for before, after in zip(default_needle, moved_needle, strict=True):
        assert after.x_m - before.x_m == pytest.approx(1e-3)
        assert after.z_m - before.z_m == pytest.approx(2e-3)
        assert abs(after.amplitude) == 12.0
    assert all(abs(point.amplitude) == 24.0 for point in moved[-2:])


def test_zero_deviation_speckle_is_recorded_as_an_undefined_metric():
    metrics = record._uniform_metrics(
        np.zeros((3, 3)), np.array([-0.001, 0.0, 0.001]), np.array([0.02, 0.03, 0.04])
    )
    assert metrics["speckle_snr"] is None


def test_production_estimate_guard_does_not_start_or_write_a_record(tmp_path, monkeypatch):
    root = tmp_path / "records"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "_require_clean_checkout", lambda: None)
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: {"production_allowed": False})
    monkeypatch.setattr(record, "build_record", lambda **kwargs: pytest.fail("production started"))
    result = record.run_record(root / "2026-11-03")
    assert result["status"] == "blocked"
    assert result["preflight"] == {"production_allowed": False}
    assert not root.exists()


def test_failed_sanity_record_is_preserved(tmp_path, monkeypatch):
    root = tmp_path / "records"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "_require_clean_checkout", lambda: None)
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: {"production_allowed": True})
    result = record.build_record(
        profile=_small_profile(),
        cases={"uniform_speckle": {"generator": uniform_speckle,
                                  "parameters": {
                                      "seed": 8,
                                      "density_per_cell": 0.01,
                                      "background_x_bounds_m": (-1e-3, 1e-3),
                                      "background_z_bounds_m": (3e-3, 17e-3),
                                  }}},
    )
    assert result["sanity"]["status"] == "fail"
    monkeypatch.setattr(record, "build_record", lambda **kwargs: result)
    json_path, readme_path = record.run_record(root / "2026-11-03")
    assert json.loads(json_path.read_text())["sanity"]["status"] == "fail"
    assert readme_path.read_text() == record.render_readme(json.loads(json_path.read_text()))


def test_build_record_rejects_unknown_phantom_name_before_generator_execution():
    def unexpected_generator(profile, **parameters):
        pytest.fail("unknown phantom generator ran before name validation")

    profile = _small_profile()
    with pytest.raises(ValueError, match="unknown phantom"):
        record.build_record(
            profile=profile,
            cases={"not_a_phantom": {"generator": unexpected_generator, "parameters": {}}},
        )


def test_case_metrics_rejects_unknown_phantom_name():
    image = np.zeros((2, 2))
    x_axis = np.array([0.0, 1.0])
    z_axis = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="unknown phantom"):
        record._case_metrics("not_a_phantom", image, x_axis, z_axis, [], {})


def test_committed_measurement_readme_renders_unchanged_with_historical_generator_arguments():
    record_dir = (
        Path(__file__).resolve().parents[1]
        / "docs/measurements/issue14-phantom-metrics/2026-11-03"
    )
    saved_record = json.loads((record_dir / record.OUTPUT_JSON).read_text(encoding="utf-8"))
    high_echo_case = next(case for case in saved_record["cases"] if case["name"] == "high_echo_structures")
    arguments = high_echo_case["generator_arguments"]
    assert "needle_center_x_m" in arguments
    assert "needle_center_z_m" in arguments
    assert "needle_amplitude_ratio" in arguments
    assert "calcification_amplitude_ratio" in arguments
    assert record.render_readme(saved_record) == (record_dir / record.OUTPUT_README).read_text(encoding="utf-8")


def test_record_rendering_and_small_end_to_end_json(tmp_path, monkeypatch):
    profile = _small_profile()
    output_root = tmp_path / "measurements"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", output_root)
    monkeypatch.setattr(record, "_git", lambda *args: "a" * 40)
    # A deliberately small case set exercises real simulate_frame -> golden DAS -> envelope
    # without invoking the production-scale speckle field.
    tiny_cases = {
        "uniform_speckle": {
            "generator": uniform_speckle,
            "parameters": {
                "seed": 8,
                "density_per_cell": 0.01,
                "background_x_bounds_m": (-1e-3, 1e-3),
                "background_z_bounds_m": (3e-3, 17e-3),
            },
        }
    }
    result = record.build_record(profile=profile, cases=tiny_cases)
    rendered = record.render_readme(result)
    assert "What was measured" in rendered
    assert result["environment"]["python_version"]
    assert result["environment"]["numpy_version"]
    assert result["environment"]["cpu_model"]
    case = result["cases"][0]
    assert case["scatterer_count"] > 0
    assert case["generator_arguments"]["seed"] == 8
    assert case["generator_arguments"]["density_per_cell"] == 0.01
    assert case["generator_arguments"]["background_x_bounds_m"] == [-1e-3, 1e-3]
    assert case["generator_arguments"]["background_z_bounds_m"] == [3e-3, 17e-3]
    assert case["elapsed_seconds"] >= 0.0
    assert "before" in case["uptime_load_average"]
    assert "after" in case["uptime_load_average"]
    assert "speckle_snr" in case["metrics"]
    assert result["metric_roi_arguments"]["uniform_speckle"]
    assert result["definitions"]["resolution_cell"]

    path = output_root / "2026-11-03"
    json_path, readme_path = record.write_record(path, result)
    json_text = json_path.read_text(encoding="utf-8")
    decoded = json.loads(json_text)
    assert str(tmp_path) not in json_text
    assert json.dumps(decoded, allow_nan=False)
    assert record.render_readme(decoded) == readme_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        record.write_record(path, result)


def test_measurement_sanity_passes_and_reports_failure_without_raising():
    passing = record.evaluate_sanity(
        [
            {"name": "uniform_speckle", "metrics": {"speckle_snr": 2.0}},
            {"name": "point_scatterer_array", "metrics": {"targets": [{"lateral_fwhm_m": 1e-3}]}},
            {"name": "anechoic_cysts", "metrics": {"cysts": [{"diameter_m": 0.008, "gcnr": 0.7}]}},
        ]
    )
    assert passing["status"] == "pass"
    assert all(check["status"] == "pass" for check in passing["checks"])

    checks = record.evaluate_sanity(
        [
            {
                "name": "uniform_speckle",
                "metrics": {"speckle_snr": 2.5},
            },
            {
                "name": "point_scatterer_array",
                "metrics": {"targets": [{"lateral_fwhm_m": None}]},
            },
            {
                "name": "anechoic_cysts",
                "metrics": {"cysts": [{"diameter_m": 0.008, "gcnr": 0.4}]},
            },
        ]
    )
    assert checks["status"] == "fail"
    assert all(check["status"] == "fail" for check in checks["checks"])


def test_production_record_refuses_a_dirty_checkout_before_measuring(tmp_path, monkeypatch):
    root = tmp_path / "records"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "_git", lambda *args: " M dirty.py")
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: pytest.fail("must not estimate"))
    with pytest.raises(RuntimeError, match="clean committed checkout"):
        record.run_record(root / "2026-11-03")
    assert not root.exists()


def test_default_background_bounds_set_candidate_count_and_contain_scatterers(monkeypatch):
    profile = linear_5mhz()
    x_bounds = (-15e-3, 15e-3)
    z_bounds = (10e-3, 50e-3)
    density = 0.01
    cell_width, cell_height = resolution_cell(profile)
    candidate_count = math.ceil(
        density * (x_bounds[1] - x_bounds[0]) * (z_bounds[1] - z_bounds[0])
        / (cell_width * cell_height)
    )
    generators = [
        lambda: uniform_speckle(profile, seed=11, density_per_cell=density),
        lambda: anechoic_cysts(profile, seed=11, density_per_cell=density),
        lambda: high_echo_structures(profile, seed=11, density_per_cell=density),
    ]
    for generate in generators:
        points = generate()
        assert points
        assert all(x_bounds[0] <= point.x_m <= x_bounds[1] for point in points)
        assert all(z_bounds[0] <= point.z_m <= z_bounds[1] for point in points)
        xmin, xmax, zmin, zmax = imaging_region(profile)
        assert all(xmin <= point.x_m <= xmax for point in points)
        assert all(zmin <= point.z_m <= zmax for point in points)
    assert len(generators[0]()) == candidate_count
    assert len(generators[2]()) == candidate_count + 23

    observed_counts = []
    monkeypatch.setattr(
        phantom_generators,
        "_uniform_points",
        lambda profile, rng, count, **kwargs: observed_counts.append(count) or [],
    )
    anechoic_cysts(profile, seed=11, density_per_cell=density)
    assert observed_counts == [candidate_count]


def test_background_bounds_must_be_finite_ordered_and_inside_imaging_region():
    profile = linear_5mhz()
    generators = (uniform_speckle, anechoic_cysts, high_echo_structures)
    invalid_bounds = [
        {"background_x_bounds_m": (-20e-3, 20e-3), "background_z_bounds_m": (10e-3, 50e-3)},
        {"background_x_bounds_m": (1e-3, -1e-3), "background_z_bounds_m": (10e-3, 50e-3)},
        {"background_x_bounds_m": (-1e-3, 1e-3), "background_z_bounds_m": (float("nan"), 50e-3)},
    ]
    for generator in generators:
        for bounds in invalid_bounds:
            with pytest.raises(ValueError, match="background bounds"):
                generator(profile, density_per_cell=0.0, **bounds)


def test_default_structure_and_metric_roi_extents_fit_background_bounds():
    cases = record._default_cases(14)
    x_bounds = (-15e-3, 15e-3)
    z_bounds = (10e-3, 50e-3)
    for name in ("anechoic_cysts", "uniform_speckle", "high_echo_structures"):
        params = cases[name]["parameters"]
        assert params["background_x_bounds_m"] == x_bounds
        assert params["background_z_bounds_m"] == z_bounds

    points = cases["point_scatterer_array"]["parameters"]
    point_targets = point_scatterer_array(linear_5mhz(), **points)
    assert all(x_bounds[0] <= target.x_m <= x_bounds[1] for target in point_targets)
    assert all(z_bounds[0] <= target.z_m <= z_bounds[1] for target in point_targets)
    assert min(points["lateral_positions_m"]) - 3e-3 >= x_bounds[0]
    assert max(points["lateral_positions_m"]) + 3e-3 <= x_bounds[1]
    assert points["z_start_m"] - 2e-3 >= z_bounds[0]
    assert points["z_stop_m"] + 2e-3 <= z_bounds[1]

    cyst = cases["anechoic_cysts"]["parameters"]
    for center_x, diameter in zip(cyst["cyst_centers_x_m"], cyst["cyst_diameters_m"], strict=True):
        outer_radius = 1.75 * diameter / 2
        assert x_bounds[0] <= center_x - outer_radius
        assert center_x + outer_radius <= x_bounds[1]
    assert tuple(cyst["cyst_centers_x_m"]) == (-10e-3, -3e-3, 6e-3)
    assert all(z_bounds[0] <= z - 1.75 * 8e-3 / 2 for z in cyst["cyst_centers_z_m"])
    assert all(z + 1.75 * 8e-3 / 2 <= z_bounds[1] for z in cyst["cyst_centers_z_m"])

    rois = record._roi_record()
    speckle_roi = rois["uniform_speckle"]
    assert x_bounds[0] <= speckle_roi["x_bounds_m"][0]
    assert speckle_roi["x_bounds_m"][1] <= x_bounds[1]
    assert z_bounds[0] <= speckle_roi["z_bounds_m"][0]
    assert speckle_roi["z_bounds_m"][1] <= z_bounds[1]
    high = cases["high_echo_structures"]["parameters"]
    needle_center_x_m, needle_center_z_m = high["needle_center_m"]
    needle_margin = high["needle_length_m"] / 2 + rois["high_echo"]["needle_outside_distance_m"][1]
    assert x_bounds[0] <= needle_center_x_m - needle_margin
    assert needle_center_x_m + needle_margin <= x_bounds[1]
    assert z_bounds[0] <= needle_center_z_m - needle_margin
    assert needle_center_z_m + needle_margin <= z_bounds[1]
    calc_margin = rois["high_echo"]["calcification_outside_radius_m"][1]
    for x, z in high["calcification_positions_m"]:
        assert x_bounds[0] <= x - calc_margin
        assert x + calc_margin <= x_bounds[1]
        assert z_bounds[0] <= z - calc_margin
        assert z + calc_margin <= z_bounds[1]


def test_two_point_timing_fit_and_invalid_slope_fail_closed():
    model = record.fit_pilot_timing_model(
        [
            {"scatterer_count": 500, "elapsed_seconds": 20.0},
            {"scatterer_count": 2000, "elapsed_seconds": 50.0},
        ]
    )
    assert model["valid"]
    assert model["raw_seconds_per_scatterer"] == pytest.approx(0.02)
    assert model["raw_intercept_seconds"] == pytest.approx(10.0)
    assert model["applied_seconds_per_scatterer"] == pytest.approx(0.02)
    assert model["applied_intercept_seconds"] == pytest.approx(10.0)

    clamped = record.fit_pilot_timing_model(
        [
            {"scatterer_count": 500, "elapsed_seconds": 10.0},
            {"scatterer_count": 2000, "elapsed_seconds": 100.0},
        ]
    )
    assert clamped["valid"]
    assert clamped["raw_intercept_seconds"] == pytest.approx(-20.0)
    assert clamped["applied_intercept_seconds"] == 0.0
    assert clamped["clamped"]["intercept"]

    invalid = record.fit_pilot_timing_model(
        [
            {"scatterer_count": 500, "elapsed_seconds": 20.0},
            {"scatterer_count": 2000, "elapsed_seconds": 19.0},
        ]
    )
    assert not invalid["valid"]
    assert invalid["failure_reason"]


def test_estimator_runs_exact_500_then_2000_point_pilots_and_fits_recorded_json(monkeypatch):
    profile = _small_profile()
    calls = []

    def fixed_case(profile, *, seed, gain):
        return [PointScatterer(0.0, 10e-3, float(seed) * gain) for _ in range(100)]

    cases = {"pilot_test": {"generator": fixed_case, "parameters": {"seed": 37, "gain": 2.0}}}

    def timed_pilot(profile, *, scatterer_count, seed):
        calls.append((scatterer_count, seed))
        elapsed = 20.0 if scatterer_count == 500 else 50.0
        return {
            "scatterer_count": scatterer_count,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "uptime_load_average": {"before": {"load_average": "0"}, "after": {"load_average": "0"}},
        }

    monkeypatch.setattr(record, "_time_pilot", timed_pilot)
    monkeypatch.setattr(
        record, "_git", lambda *args: "" if args[0] == "status" else "a" * 40
    )
    monkeypatch.setattr(record, "_cpu_model", lambda: "test-cpu")
    estimate = record.estimate_project(profile=profile, seed=36, cases=cases)
    assert [count for count, _ in calls] == [500, 2000]
    assert estimate["seed"] == 36
    assert [pilot["scatterer_count"] for pilot in estimate["pilots"]] == [500, 2000]
    assert [pilot["seed"] for pilot in estimate["pilots"]] == [36, 37]
    for pilot in estimate["pilots"]:
        assert "before" in pilot["uptime_load_average"]
        assert "after" in pilot["uptime_load_average"]
    assert estimate["timing_model"]["applied_intercept_seconds"] == pytest.approx(10.0)
    assert estimate["timing_model"]["applied_seconds_per_scatterer"] == pytest.approx(0.02)
    assert estimate["estimated_cases"][0]["estimated_seconds"] == pytest.approx(12.0)
    assert estimate["estimated_cases"][0]["generator_arguments"] == {"seed": 37, "gain": 2.0}
    assert estimate["environment"]["commit"] == "a" * 40
    assert estimate["environment"]["cpu_model"] == "test-cpu"
    assert estimate["environment"]["python_version"]
    assert estimate["environment"]["numpy_version"]
    assert estimate["formula"]
    assert estimate["production_allowed"]
    assert record.MAX_ESTIMATED_SECONDS == 3 * 60 * 60


def test_estimator_blocks_when_fitted_total_exceeds_three_hours(monkeypatch):
    profile = _small_profile()

    def large_case(profile, *, seed):
        return [PointScatterer(0.0, 10e-3, float(seed)) for _ in range(3000)]

    cases = {"large": {"generator": large_case, "parameters": {"seed": 4}}}
    pilot_times = iter((2000.0, 8000.0))

    def timed_pilot(profile, *, scatterer_count, seed):
        return {
            "scatterer_count": scatterer_count,
            "seed": seed,
            "elapsed_seconds": next(pilot_times),
            "uptime_load_average": {"before": {}, "after": {}},
        }

    monkeypatch.setattr(record, "_time_pilot", timed_pilot)
    monkeypatch.setattr(
        record, "_git", lambda *args: "" if args[0] == "status" else "a" * 40
    )
    monkeypatch.setattr(record, "_cpu_model", lambda: "test-cpu")
    estimate = record.estimate_project(profile=profile, seed=4, cases=cases)
    assert estimate["estimated_total_seconds"] == pytest.approx(12_000.0)
    assert estimate["maximum_allowed_seconds"] == 10_800
    assert not estimate["production_allowed"]
    assert estimate["failure_reason"] == "estimated total exceeds the three-hour limit"


def test_pilot_scatterer_helper_returns_exact_requested_counts():
    profile = linear_5mhz()
    for count in (500, 2000):
        points = record._pilot_scatterers(profile, scatterer_count=count, seed=14)
        assert len(points) == count
        assert all(-15e-3 <= point.x_m <= 15e-3 for point in points)
        assert all(10e-3 <= point.z_m <= 50e-3 for point in points)


def test_estimate_only_returns_preflight_without_production(tmp_path, monkeypatch):
    root = tmp_path / "records"
    preflight = {"production_allowed": False, "pilots": [{"scatterer_count": 500}, {"scatterer_count": 2000}]}
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: preflight)
    monkeypatch.setattr(record, "build_record", lambda **kwargs: pytest.fail("production started"))
    result = record.run_record(root / "2026-11-03", estimate_only=True)
    assert result == preflight
    assert not root.exists()


def test_invalid_pilot_fit_is_returned_as_blocked_preflight(monkeypatch):
    profile = _small_profile()

    def fixed_case(profile, *, seed):
        return [PointScatterer(0.0, 10e-3, float(seed)) for _ in range(5)]

    cases = {"pilot_test": {"generator": fixed_case, "parameters": {"seed": 9}}}
    durations = iter((20.0, 19.0))

    def noisy_pilot(profile, *, scatterer_count, seed):
        return {
            "scatterer_count": scatterer_count,
            "seed": seed,
            "elapsed_seconds": next(durations),
            "uptime_load_average": {"before": {"load_average": "0"}, "after": {"load_average": "0"}},
        }

    monkeypatch.setattr(record, "_time_pilot", noisy_pilot)
    monkeypatch.setattr(
        record, "_git", lambda *args: "" if args[0] == "status" else "a" * 40
    )
    monkeypatch.setattr(record, "_cpu_model", lambda: "test-cpu")
    result = record.estimate_project(profile=profile, seed=3, cases=cases)
    assert not result["production_allowed"]
    assert result["estimated_total_seconds"] is None
    assert [pilot["elapsed_seconds"] for pilot in result["pilots"]] == [20.0, 19.0]
    assert result["timing_model"]["raw_seconds_per_scatterer"] < 0.0
    assert result["failure_reason"]


def test_blocked_preflight_preserves_pilots_and_never_starts_production(tmp_path, monkeypatch):
    root = tmp_path / "records"
    preflight = {
        "production_allowed": False,
        "estimated_total_seconds": 10_801.0,
        "pilots": [{"scatterer_count": 500, "elapsed_seconds": 20.0}, {"scatterer_count": 2000, "elapsed_seconds": 50.0}],
    }
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "_require_clean_checkout", lambda: None)
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: preflight)
    monkeypatch.setattr(record, "build_record", lambda **kwargs: pytest.fail("production started"))
    result = record.run_record(root / "2026-11-03")
    assert result == {"status": "blocked", "preflight": preflight}
    assert not root.exists()


def _minimal_write_record():
    return {
        "environment": {"commit": "x", "cpu_model": "test", "python_version": "3", "numpy_version": "2"},
        "profile": {"name": "linear-5mhz", "bandwidth_frac": 0.7, "bandwidth_source": None},
        "processing": {"imaging_region_m": {"x": [-1.0, 1.0], "z": [0.0, 1.0]}},
        "definitions": {
            "resolution_cell": {"lateral_width_m": 1.0, "axial_height_m": 1.0},
            "density_count_formula": "ceil(...) ",
            "cyst_exclusion": "strict interior excluded",
            "background_amplitude": "normal",
        },
        "cases": [],
        "sanity": {"status": "pass"},
    }


def test_fresh_nested_dated_record_is_write_once(tmp_path, monkeypatch):
    root = tmp_path / "records"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    destination = root / "2026-11-03" / "rerun-1"
    json_path, readme_path = record.write_record(destination, _minimal_write_record())
    assert json_path.exists()
    assert readme_path.exists()
    existing_record = _minimal_write_record()
    with pytest.raises(FileExistsError):
        record.write_record(destination, existing_record)


@pytest.mark.parametrize("mode", ["production", "estimate"])
@pytest.mark.parametrize(
    "path_kind",
    ["traversal", "absolute_outside", "directory_symlink", "file_symlink", "dangling_symlink", "root"],
)
def test_cli_rejects_unsafe_output_before_estimate_or_record(
    tmp_path, monkeypatch, capsys, mode, path_kind
):
    root = tmp_path / "measurements"
    dated = root / "2026-11-03"
    dated.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    if path_kind == "traversal":
        destination = dated / ".." / ".." / "escaped"
    elif path_kind == "absolute_outside":
        destination = outside / "2026-11-03"
    elif path_kind == "directory_symlink":
        destination = dated / "directory-link" / "run-1"
        (dated / "directory-link").symlink_to(outside, target_is_directory=True)
    elif path_kind == "file_symlink":
        target = dated / "file-target"
        target.write_text("not a directory", encoding="utf-8")
        destination = dated / "file-link" / "run-1"
        (dated / "file-link").symlink_to(target)
    elif path_kind == "dangling_symlink":
        destination = dated / "dangling-link" / "run-1"
        (dated / "dangling-link").symlink_to(outside / "missing", target_is_directory=True)
    else:
        destination = root
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: pytest.fail("estimate ran before path validation"))
    monkeypatch.setattr(record, "run_record", lambda *args, **kwargs: pytest.fail("record runner ran before path validation"))
    argv = ["record", "--output-dir", str(destination)]
    if mode == "estimate":
        argv.append("--estimate-only")
    monkeypatch.setattr(sys, "argv", argv)
    assert record.main() == 2
    assert "output" in capsys.readouterr().err


def test_measurement_root_rejects_escape_paths_and_existing_outputs(tmp_path, monkeypatch):
    root = tmp_path / "records"
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    root.mkdir()
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    minimal = {"environment": {"commit": "x"}, "cases": []}
    with pytest.raises(ValueError, match="measurement directory"):
        record.write_record(outside / "run", minimal)
    with pytest.raises(ValueError, match="measurement directory"):
        record.write_record(link / "run", minimal)
    destination = root / "2026-11-03"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        record.write_record(destination, minimal)
