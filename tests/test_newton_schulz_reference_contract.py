"""Contract tests for the independent Newton-Schulz reference inputs."""

import warnings

import numpy as np
import pytest

from enodia.tt.bench import newton_schulz_kernel
from enodia.tt.bench.newton_schulz_reference import (
    NEWTON_SCHULZ_ITERATIONS,
    bf16_round_complex,
    bf16_round_to_float32,
    initial_value,
    newton_schulz_reference,
    random_hpd_batch,
)


def _matrices() -> np.ndarray:
    return random_hpd_batch(2, 4, condition_number=10.0, seed=18)


def test_reference_rejects_unbatched_matrices_with_or_without_explicit_x0():
    matrices = np.eye(4, dtype=np.complex64)
    x0 = np.eye(4, dtype=np.complex64)

    for explicit in (None, x0):
        with pytest.raises(ValueError, match="matrices must have shape"):
            newton_schulz_reference(matrices, x0=explicit)


def test_reference_rejects_non_square_and_empty_matrix_batches():
    for matrices in (
        np.zeros((2, 3, 4), dtype=np.complex64),
        np.zeros((0, 4, 4), dtype=np.complex64),
        np.zeros((2, 4), dtype=np.complex64),
    ):
        with pytest.raises(ValueError, match="matrices must have shape"):
            newton_schulz_reference(matrices)


def test_reference_rejects_explicit_x0_shape_and_dtype_mismatches():
    matrices = _matrices()
    for x0 in (
        np.zeros((1, 4, 4), dtype=np.complex64),
        np.zeros((2, 3, 4), dtype=np.complex64),
        np.zeros((2, 4), dtype=np.complex64),
        np.zeros((2, 4, 4), dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="x0"):
            newton_schulz_reference(matrices, x0=x0)


def test_reference_accepts_complex128_x0_after_canonicalization():
    matrices = _matrices()
    x0 = initial_value(matrices).astype(np.complex128)

    got = newton_schulz_reference(matrices, iterations=0, x0=x0)

    assert got.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(got, initial_value(matrices))


def test_narrowing_conversions_reject_overflow_without_runtime_warnings():
    matrices = np.eye(2, dtype=np.complex128)[None, :, :] * 1e300
    x0 = np.eye(2, dtype=np.complex128)[None, :, :] * 1e300
    real_values = np.array([1e300], dtype=np.float64)
    complex_values = np.array([1e300 + 1j * 1e300], dtype=np.complex128)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="finite"):
            newton_schulz_reference(matrices, x0=x0)
        with pytest.raises(ValueError, match="finite"):
            bf16_round_to_float32(real_values)
        with pytest.raises(ValueError, match="finite"):
            bf16_round_complex(complex_values)
    assert caught == []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="finite"):
            newton_schulz_kernel._initial_value(matrices)
    assert caught == []


@pytest.mark.parametrize("iterations", [-1, 1.0, True, "12"])
def test_reference_rejects_ambiguous_iteration_counts(iterations):
    expected_exception = ValueError if iterations == -1 else TypeError
    with pytest.raises(expected_exception, match="iterations"):
        newton_schulz_reference(_matrices(), iterations=iterations)


def test_reference_preserves_valid_zero_and_fixed_count_behavior():
    matrices = _matrices()

    zero = newton_schulz_reference(matrices, iterations=0)
    twelve = newton_schulz_reference(matrices, iterations=NEWTON_SCHULZ_ITERATIONS)

    np.testing.assert_array_equal(zero, initial_value(matrices))
    assert twelve.shape == matrices.shape
    assert np.all(np.isfinite(twelve))


def test_initial_value_rejects_zero_and_nonfinite_matrix_batches():
    for matrices in (
        np.zeros((2, 4, 4), dtype=np.complex64),
        np.array([[[np.inf + 0j]]], dtype=np.complex64),
        np.eye(4, dtype=np.complex64),
    ):
        with pytest.raises(ValueError):
            initial_value(matrices)


def test_kernel_initial_value_rejects_the_same_invalid_shapes_and_norms():
    for matrices in (
        np.eye(4, dtype=np.complex64),
        np.zeros((2, 4, 4), dtype=np.complex64),
        np.array([[[np.nan + 0j]]], dtype=np.complex64),
    ):
        with pytest.raises(ValueError):
            newton_schulz_kernel._initial_value(matrices)


def test_bf16_helpers_preserve_shape_and_reject_invalid_planes():
    values = np.ones((2, 4, 4), dtype=np.complex64)
    rounded = bf16_round_complex(values)

    assert rounded.shape == values.shape
    assert rounded.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(rounded, values)

    with pytest.raises(ValueError, match="float32 or float64"):
        bf16_round_to_float32(np.ones(4, dtype=np.int32))
    with pytest.raises(ValueError, match="finite"):
        bf16_round_to_float32(np.array([np.inf], dtype=np.float32))
    with pytest.raises(ValueError, match="complex64 or complex128"):
        bf16_round_complex(np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        bf16_round_complex(np.array([np.nan + 0j], dtype=np.complex64))


def test_kernel_compile_helpers_reject_noninteger_iteration_values():
    for iterations in (-1, 12.0, True):
        expected_exception = ValueError if iterations == -1 else TypeError
        with pytest.raises(expected_exception, match="iterations"):
            newton_schulz_kernel._reader_compile_args(
                iterations=iterations,
                profile=False,
                fuse_s=False,
                batch_reads=False,
                matrix_block=1,
                reload_r=False,
            )
        with pytest.raises(expected_exception, match="iterations"):
            newton_schulz_kernel._compute_compile_args(
                iterations=iterations,
                state_fp32=True,
                profile=False,
                fuse_s=False,
                matrix_block=1,
                reload_r=False,
            )
