"""Analytic and boundary checks for Issue #14 PR A invariants."""

import ast
import subprocess
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pytest

from enodia.spec.eval.metrics import gcnr, speckle_snr, width_at


# Invariant 1: pure array metrics, no CNR function anywhere in repository Python.
def test_no_cnr_function_in_repository():
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=root,
        text=True,
    ).splitlines()
    for path in paths:
        tree = ast.parse((root / path).read_text(), filename=path)
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lower() == "cnr"
            for node in ast.walk(tree)
        ), path


def test_metrics_do_not_mutate_inputs():
    inside = np.array([0.1, 0.2, 0.3])
    outside = np.array([0.2, 0.3, 0.4])
    profile = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
    coords = np.arange(5.0)
    arrays = [inside, outside, profile, coords]
    copies = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    gcnr(inside, outside)
    width_at(profile, coords, -6.0)
    speckle_snr(inside)
    for array, copy in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(array, copy)


# Invariants 2 and 3: common equal-width bins, normalized counts, analytic overlap.
def test_gcnr_common_edges_and_separate_normalization():
    inside = np.array([0.0, 0.0, 1.0, 2.0])
    outside = np.array([1.0, 3.0])
    assert gcnr(inside, outside, bins=3) == pytest.approx(0.5)
    assert gcnr(inside, outside, bins=1) == 0.0
    assert gcnr(outside, inside, bins=3) == gcnr(inside, outside, bins=3)


def test_gcnr_disjoint_and_identical():
    samples = np.linspace(0.0, 1.0, 1000)
    assert gcnr(samples, samples + 2.0) == 1.0
    assert gcnr(samples, samples) == 0.0


@pytest.mark.parametrize("value", [0.0, 2.5])
def test_gcnr_all_values_equal(value):
    assert gcnr(np.full(3, value), np.full(7, value)) == 0.0


@pytest.mark.parametrize("inside,outside", [([], [1.0]), ([1.0], []), ([], [])])
def test_gcnr_empty_input(inside, outside):
    with pytest.raises(ValueError):
        gcnr(np.asarray(inside), np.asarray(outside))


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0])
def test_gcnr_seeded_normal_overlap(d):
    rng = np.random.default_rng(14)
    inside = rng.normal(0.0, 1.0, 100_000)
    outside = rng.normal(d, 1.0, 100_000)
    expected = erf(d / (2.0 * sqrt(2.0)))
    assert gcnr(inside, outside) == pytest.approx(expected, abs=0.01)


# Invariants 4 through 6: first crossings, interpolation in dB, physical coordinates.
@pytest.mark.parametrize("level_db", [20.0 * np.log10(0.5), -40.0])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_width_gaussian_closed_form(level_db, nonuniform):
    s = 1.7
    grid = np.linspace(-1.0, 1.0, 2001)
    coords = 8.0 * s * (np.sign(grid) * np.abs(grid) ** 1.3 if nonuniform else grid)
    profile = 3.0 * np.exp(-0.5 * (coords / s) ** 2)
    expected = 2.0 * s * np.sqrt(-2.0 * np.log(10.0 ** (level_db / 20.0)))
    assert width_at(profile, coords, level_db) == pytest.approx(expected, rel=0.01)


def test_width_first_crossings_and_db_interpolation():
    coords = np.array([-9.0, -5.0, -2.0, 0.0, 1.0, 5.0, 8.0])
    db = np.array([-30.0, -2.0, -10.0, 0.0, -10.0, -2.0, -30.0])
    profile = 7.0 * 10.0 ** (db / 20.0)
    assert width_at(profile, coords, -5.0) == pytest.approx(1.5)


def test_width_crossings_exactly_at_array_endpoints():
    assert width_at(
        np.array([0.5, 1.0, 0.5]), np.array([-2.0, 0.0, 3.0]), 20.0 * np.log10(0.5)
    ) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "profile", [[0.9, 1.0, 0.9], [0.9, 1.0, 0.1], [0.1, 1.0, 0.9], [1.0, 0.5, 0.1]]
)
def test_width_missing_either_crossing_is_nan(profile):
    assert np.isnan(width_at(np.asarray(profile), np.arange(3.0), -20.0))


@pytest.mark.parametrize("bad_sample", [np.nan, np.inf, -np.inf, -0.1])
def test_width_rejects_nonfinite_or_negative_samples(bad_sample):
    with pytest.raises(ValueError):
        width_at(np.array([bad_sample, 1.0, 0.1]), np.arange(3.0), -6.0)


def test_width_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        width_at(np.ones(3), np.arange(4.0), -6.0)


@pytest.mark.parametrize("coords", [[0.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
def test_width_requires_strictly_increasing_coords(coords):
    with pytest.raises(ValueError):
        width_at(np.array([0.1, 1.0, 0.1]), np.asarray(coords), -6.0)


# Invariant 7: population deviation, stated invalid cases, seeded Rayleigh reference.
def test_speckle_snr_population_deviation():
    region = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert speckle_snr(region) == pytest.approx(2.5 / np.sqrt(1.25))


@pytest.mark.parametrize("region", [[], [1.0], [0.0, 0.0], [2.0, 2.0, 2.0]])
def test_speckle_snr_rejects_small_or_constant_region(region):
    with pytest.raises(ValueError):
        speckle_snr(np.asarray(region))


def test_speckle_snr_seeded_rayleigh():
    region = np.random.default_rng(14).rayleigh(size=100_000)
    assert speckle_snr(region) == pytest.approx(1.913, abs=0.02)
