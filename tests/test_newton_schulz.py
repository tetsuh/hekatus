"""Tests for the fixed-work NumPy Newton-Schulz reference."""

import numpy as np
import pytest

from enodia.spec.beamform.newton_schulz import (
    NEWTON_SCHULZ_ITERATIONS,
    X0_IDENTITY_NORM_INF,
    X0_R_H_NORMS,
    newton_schulz_inverse,
)
from enodia.spec.beamform.newton_schulz_sweep import deterministic_hpd

X0_CHOICES = (X0_R_H_NORMS, X0_IDENTITY_NORM_INF)


@pytest.fixture
def hand_checkable_matrix() -> np.ndarray:
    """A small Hermitian positive-definite matrix with an easy inverse."""
    return np.array(
        [
            [2.0 + 0.0j, 1.0 - 1.0j],
            [1.0 + 1.0j, 3.0 + 0.0j],
        ],
        dtype=np.complex128,
    )


def test_hand_checkable_matrix_matches_true_inverse(hand_checkable_matrix):
    # The determinant is 4, so this inverse is easy to check by hand.
    expected = np.array(
        [[0.75 + 0.0j, -0.25 + 0.25j], [-0.25 - 0.25j, 0.5 + 0.0j]],
        dtype=np.complex128,
    )

    got = newton_schulz_inverse(
        hand_checkable_matrix, iterations=NEWTON_SCHULZ_ITERATIONS
    )

    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("x0", "expected"),
    [
        (
            "r_h_norms",
            np.diag([2.0 / 16.0, 4.0 / 16.0]).astype(np.complex128),
        ),
        (
            "identity_norminf",
            np.diag([1.0 / 4.0, 1.0 / 4.0]).astype(np.complex128),
        ),
    ],
)
def test_both_initial_values_are_available_without_an_iteration(x0, expected):
    matrix = np.diag([2.0, 4.0]).astype(np.complex128)

    got = newton_schulz_inverse(matrix, iterations=0, x0=x0)

    assert np.array_equal(got, expected)


def test_stage2_default_uses_identity_norminf_and_fixed_iteration_constant():
    matrix = np.diag([2.0, 4.0]).astype(np.complex128)

    assert NEWTON_SCHULZ_ITERATIONS == 12
    np.testing.assert_array_equal(
        newton_schulz_inverse(matrix, iterations=0),
        np.diag([0.25, 0.25]).astype(np.complex128),
    )
    np.testing.assert_array_equal(
        newton_schulz_inverse(matrix, iterations=0, x0=X0_R_H_NORMS),
        np.diag([2.0 / 16.0, 4.0 / 16.0]).astype(np.complex128),
    )


def test_stage2_default_kappa300_l32_float32_error_matches_selected_bound():
    matrix = deterministic_hpd(32, 300.0, seed=85, dtype="float32")
    approximate = newton_schulz_inverse(matrix)
    truth = np.linalg.inv(matrix.astype(np.complex128))
    relative_error = np.linalg.norm(
        approximate.astype(np.complex128) - truth, ord="fro"
    ) / np.linalg.norm(truth, ord="fro")

    assert relative_error <= 5e-4


def test_iteration_count_is_fixed_by_the_caller():
    matrix = np.diag([2.0, 4.0]).astype(np.complex128)
    initial = np.diag([0.25, 0.25]).astype(np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    expected_one = initial @ (2.0 * identity - matrix @ initial)
    expected_two = expected_one @ (2.0 * identity - matrix @ expected_one)

    one = newton_schulz_inverse(matrix, iterations=1, x0=X0_IDENTITY_NORM_INF)
    two = newton_schulz_inverse(matrix, iterations=2, x0=X0_IDENTITY_NORM_INF)

    np.testing.assert_allclose(one, expected_one, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(two, expected_two, rtol=0.0, atol=1e-15)
    assert not np.array_equal(one, two)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_precision_is_preserved(dtype, hand_checkable_matrix):
    matrix = hand_checkable_matrix.astype(dtype)

    got = newton_schulz_inverse(
        matrix, iterations=NEWTON_SCHULZ_ITERATIONS, x0=X0_R_H_NORMS
    )

    assert got.dtype == np.dtype(dtype)
    np.testing.assert_allclose(got, np.linalg.inv(matrix), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("complex_input", [False, True])
def test_normal_range_power_of_two_scaling_is_bitwise_stable(real_dtype, complex_input):
    work_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    matrix = np.array([[1.25, 0.25], [0.25, 2.5]], dtype=real_dtype)
    if complex_input:
        matrix = matrix.astype(work_dtype)
        matrix[0, 1] = 0.25 + 0.125j
        matrix[1, 0] = 0.25 - 0.125j

    scale_exponent = 8
    scaled_matrix = matrix * np.array(2**scale_exponent, dtype=real_dtype)
    baseline = newton_schulz_inverse(
        matrix, iterations=4, x0=X0_IDENTITY_NORM_INF
    )
    scaled = newton_schulz_inverse(
        scaled_matrix, iterations=4, x0=X0_IDENTITY_NORM_INF
    )
    expected = np.empty_like(scaled)
    expected.real[...] = np.ldexp(baseline.real, -scale_exponent)
    expected.imag[...] = np.ldexp(baseline.imag, -scale_exponent)

    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize(
    ("real_dtype", "scale"),
    [
        (np.float32, 1e20),
        (np.float32, 1e-20),
        (np.float64, 1e160),
        (np.float64, 1e-160),
    ],
)
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("x0", X0_CHOICES)
def test_extreme_finite_scales_produce_finite_inverse(real_dtype, scale, complex_input, x0):
    work_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    matrix = np.eye(2, dtype=real_dtype) * np.array(scale, dtype=real_dtype)
    if complex_input:
        matrix = matrix.astype(work_dtype)

    got = newton_schulz_inverse(
        matrix, iterations=NEWTON_SCHULZ_ITERATIONS, x0=x0
    )
    expected_scale = np.array(1.0 / float(matrix[0, 0].real), dtype=real_dtype)
    expected = np.eye(2, dtype=work_dtype) * expected_scale

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=5e-5, atol=0.0)


@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("complex_input", [False, True])
@pytest.mark.parametrize("x0", X0_CHOICES)
def test_finite_max_components_do_not_overflow_scaling(real_dtype, complex_input, x0):
    work_dtype = np.complex64 if real_dtype == np.float32 else np.complex128
    limit = np.finfo(real_dtype).max
    matrix = np.zeros((2, 2), dtype=work_dtype if complex_input else real_dtype)
    if complex_input:
        matrix[0, 0] = complex(limit, limit)
        matrix[1, 1] = complex(limit, -limit)
    else:
        matrix[...] = np.eye(2, dtype=real_dtype) * limit

    got = newton_schulz_inverse(
        matrix, iterations=NEWTON_SCHULZ_ITERATIONS, x0=x0
    )
    expected = np.zeros((2, 2), dtype=work_dtype)
    _, scale_exponent = np.frexp(np.array(limit, dtype=real_dtype))
    scale_exponent = int(scale_exponent)
    for index in range(2):
        scaled_real = np.ldexp(matrix[index, index].real, -scale_exponent)
        scaled_imag = np.ldexp(matrix[index, index].imag, -scale_exponent)
        scaled_value = complex(float(scaled_real), float(scaled_imag))
        expected_real = np.ldexp((1.0 / scaled_value).real, -scale_exponent)
        expected_imag = np.ldexp((1.0 / scaled_value).imag, -scale_exponent)
        expected[index, index] = complex(expected_real, expected_imag)

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=5e-5, atol=0.0)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_rejected(nonfinite):
    matrix = np.array([[nonfinite, 0.0], [0.0, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        newton_schulz_inverse(matrix, iterations=0)


def test_casting_to_requested_precision_rejects_nonfinite_result():
    matrix = np.array([[np.finfo(np.float64).max]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        newton_schulz_inverse(matrix, iterations=0, dtype=np.float32)


def test_unknown_initial_value_is_rejected(hand_checkable_matrix):
    with pytest.raises(ValueError, match="x0"):
        newton_schulz_inverse(
            hand_checkable_matrix,
            iterations=NEWTON_SCHULZ_ITERATIONS,
            x0="other",
        )
