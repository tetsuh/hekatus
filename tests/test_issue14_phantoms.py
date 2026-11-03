"""Issue #14 phantom generators and end-to-end measurement records."""

from __future__ import annotations

import json
import math

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
            cyst_centers_x_m=(-0.5e-3, 0.0, 0.5e-3), cyst_centers_z_m=(5e-3, 12e-3)
        ),
        lambda seed: uniform_speckle(profile, seed=seed, density_per_cell=0.2),
        lambda seed: high_echo_structures(
            profile, seed=seed, density_per_cell=0.2, needle_center_z_m=9e-3, needle_length_m=1e-3,
            calcification_positions_m=((-0.7e-3, 6e-3), (0.7e-3, 12e-3))
        ),
    ]
    for generate in generators:
        first = generate(14)
        repeated = generate(14)
        different = generate(15)
        assert first and all(isinstance(point, PointScatterer) for point in first)
        assert _scatterer_tuples(first) == _scatterer_tuples(repeated)
        assert _scatterer_tuples(first) != _scatterer_tuples(different)
        assert all(xmin <= point.x_m <= xmax for point in first)
        assert all(zmin <= point.z_m <= profile.depth_m for point in first)
    region = imaging_region(profile)
    assert (region[0], region[1]) == pytest.approx((xmin, xmax))
    assert region[2:] == (zmin, profile.depth_m)


def test_cyst_generator_excludes_strict_interior_and_uses_configured_radii():
    profile = linear_5mhz()
    centers_x = (-0.010, 0.0, 0.010)
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
        z_min_m=2e-3,
        amplitude_std=1.0,
        excluded_circles=[(0.0, 0.022, 0.001)],
    )
    assert len(boundary) == 1
    assert math.hypot(boundary[0].x_m, boundary[0].z_m - 0.022) == 0.001


def test_speckle_count_uses_the_documented_resolution_cell_density_formula():
    profile = linear_5mhz()
    density = 0.25
    xmin, xmax, zmin, zmax = imaging_region(profile)
    cell_width, cell_height = resolution_cell(profile)
    expected = math.ceil(density * (xmax - xmin) * (zmax - zmin) / (cell_width * cell_height))
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
    assert len(high) == 2 * len(low) or len(high) == 2 * len(low) - 1
    assert np.mean([point.amplitude for point in high]) == pytest.approx(0.0, abs=0.15)
    assert high_echo_structures(profile, seed=4, density_per_cell=0.05, needle_amplitude_ratio=12.0) != high_echo_structures(
        profile, seed=4, density_per_cell=0.05, needle_amplitude_ratio=24.0
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
    with pytest.raises(RuntimeError, match="exceeds two hours"):
        record.run_record(root / "2026-11-03")
    assert not root.exists()


def test_failed_sanity_record_is_preserved(tmp_path, monkeypatch):
    root = tmp_path / "records"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", root)
    monkeypatch.setattr(record, "_require_clean_checkout", lambda: None)
    monkeypatch.setattr(record, "estimate_project", lambda **kwargs: {"production_allowed": True})
    result = record.build_record(
        profile=_small_profile(),
        cases={"uniform_speckle": {"generator": uniform_speckle,
                                  "parameters": {"seed": 8, "density_per_cell": 0.01}}},
    )
    assert result["sanity"]["status"] == "fail"
    monkeypatch.setattr(record, "build_record", lambda **kwargs: result)
    json_path, readme_path = record.run_record(root / "2026-11-03")
    assert json.loads(json_path.read_text())["sanity"]["status"] == "fail"
    assert readme_path.read_text() == record.render_readme(json.loads(json_path.read_text()))


def test_record_rendering_and_small_end_to_end_json(tmp_path, monkeypatch):
    profile = ProbeProfile(
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
    output_root = tmp_path / "measurements"
    monkeypatch.setattr(record, "MEASUREMENT_ROOT", output_root)
    monkeypatch.setattr(record, "_git", lambda *args: "a" * 40)
    # A deliberately small case set exercises real simulate_frame -> golden DAS -> envelope
    # without invoking the production-scale speckle field.
    tiny_cases = {
        "uniform_speckle": {
            "generator": uniform_speckle,
            "parameters": {"seed": 8, "density_per_cell": 0.01},
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
    assert case["elapsed_seconds"] >= 0.0
    assert "before" in case["uptime_load_average"] and "after" in case["uptime_load_average"]
    assert "speckle_snr" in case["metrics"]
    assert result["definitions"]["resolution_cell"]

    path = output_root / "2026-11-03"
    json_path, readme_path = record.write_record(path, result)
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
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
