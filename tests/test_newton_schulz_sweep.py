"""Tests for the deterministic offline Newton-Schulz sweep."""

import json

import numpy as np

from enodia.spec.beamform import newton_schulz_sweep as sweep_module


def test_deterministic_hpd_is_repeatable_and_has_requested_condition_number():
    first = sweep_module.deterministic_hpd(8, 30.0, seed=17)
    second = sweep_module.deterministic_hpd(8, 30.0, seed=17)

    assert np.array_equal(first, second)
    np.testing.assert_allclose(first, first.conj().T, rtol=0.0, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(first) > 0.0)
    np.testing.assert_allclose(np.linalg.cond(first), 30.0, rtol=1e-5)


def test_direction_cosine_deficit_rejects_zero_and_non_finite_norms():
    zero = np.zeros(2, dtype=np.complex128)
    non_finite = np.array([np.nan + 0j])

    assert np.isnan(sweep_module._direction_cosine_deficit(zero, zero))
    assert np.isnan(sweep_module._direction_cosine_deficit(non_finite, non_finite))


def test_small_sweep_reports_both_x0_choices_and_metrics():
    results, elapsed = sweep_module.sweep(
        condition_numbers=(10.0,),
        iterations=(8, 9),
        aperture_sizes=(4,),
        dtypes=("float32", "float64"),
    )

    assert elapsed >= 0.0
    assert len(results) == 2 * 2 * 2
    assert {result.x0 for result in results} == set(sweep_module.X0_CHOICES)
    assert {result.dtype for result in results} == {"float32", "float64"}
    assert all(np.isfinite(result.inverse_relative_frobenius_error) for result in results)
    assert all(np.isfinite(result.beam_pattern_relative_error) for result in results)
    assert all(
        len(result.mv_direction_error) == len(sweep_module.LOOK_DIRECTIONS_DEG)
        for result in results
    )


def test_measurement_record_is_strict_json_and_has_host_provenance():
    results, elapsed = sweep_module.sweep(
        condition_numbers=(10.0,), iterations=(8,), aperture_sizes=(4,), dtypes=("float32",)
    )
    record = sweep_module.measurement_record(
        results,
        elapsed,
        condition_numbers=(10.0,),
        iterations=(8,),
        aperture_sizes=(4,),
        dtypes=("float32",),
    )
    parsed = json.loads(sweep_module.strict_json_dumps(record, allow_nan=False))

    assert parsed["environment"]["board"] is None
    assert "hostname" not in parsed["environment"]
    assert "machine" not in parsed["environment"]
    assert parsed["algorithm"]["early_exit"] is False
    assert len(parsed["results"]) == 2
    assert "provisional_recommendation" in parsed
