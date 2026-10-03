"""Tests for the fixed-work NumPy Newton-Schulz reference."""

import numpy as np
import pytest

from enodia.spec.beamform.newton_schulz import newton_schulz_inverse


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

    got = newton_schulz_inverse(hand_checkable_matrix, iterations=12)

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

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-15)


def test_iteration_count_is_fixed_by_the_caller():
    matrix = np.diag([2.0, 4.0]).astype(np.complex128)
    initial = np.diag([0.25, 0.25]).astype(np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    expected_one = initial @ (2.0 * identity - matrix @ initial)
    expected_two = expected_one @ (2.0 * identity - matrix @ expected_one)

    one = newton_schulz_inverse(matrix, iterations=1, x0="identity_norminf")
    two = newton_schulz_inverse(matrix, iterations=2, x0="identity_norminf")

    np.testing.assert_allclose(one, expected_one, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(two, expected_two, rtol=0.0, atol=1e-15)
    assert not np.array_equal(one, two)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_precision_is_preserved(dtype, hand_checkable_matrix):
    matrix = hand_checkable_matrix.astype(dtype)

    got = newton_schulz_inverse(matrix, iterations=12, x0="r_h_norms")

    assert got.dtype == np.dtype(dtype)
    np.testing.assert_allclose(got, np.linalg.inv(matrix), rtol=3e-5, atol=3e-6)


def test_unknown_initial_value_is_rejected(hand_checkable_matrix):
    with pytest.raises(ValueError, match="x0"):
        newton_schulz_inverse(hand_checkable_matrix, iterations=8, x0="other")
