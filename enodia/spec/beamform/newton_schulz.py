"""NumPy Newton-Schulz inverse reference (design.md §9, issue #85).

The reference deliberately has a fixed-work loop.  ``iterations`` is supplied
by the caller and every requested update is executed; convergence and residual
checks do not belong in this implementation.  The two initial values are the
ones considered by the stage-1 sweep:

* ``r_h_norms``: ``Rᴴ / (||R||₁ ||R||∞)``
* ``identity_norminf``: ``I / ||R||∞``

The canonical default follows the owner-approved Stage 2 choice in ADR-0012:
``identity_norminf`` with twelve updates. Callers may still select the
historical ``r_h_norms`` initial value and any non-negative iteration count for
Stage-1 sweeps and comparison experiments. Complex64 and complex128 inputs use
the same computation with their matching real precision. The optional
``dtype`` argument selects that precision when a caller wants to pass a real
matrix or explicitly cast an input.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

X0_R_H_NORMS = "r_h_norms"
X0_IDENTITY_NORM_INF = "identity_norminf"
X0_CHOICES = (X0_R_H_NORMS, X0_IDENTITY_NORM_INF)

# ADR-0012 records the owner-approved Stage 2 choice: the canonical spec
# default uses identity_norminf and twelve fixed updates. Callers may override
# this count for Stage-1 sweeps and historical experiments.
NEWTON_SCHULZ_ITERATIONS = 12


def _complex_dtype(array: np.ndarray, dtype: np.dtype | type | None) -> np.dtype:
    """Return the complex dtype corresponding to the requested real precision."""
    source = np.dtype(array.dtype if dtype is None else dtype)
    if source.kind == "f" and source.itemsize in (4, 8):
        return np.dtype(np.complex64 if source.itemsize == 4 else np.complex128)
    if source.kind == "c" and source.itemsize in (8, 16):
        return source
    raise ValueError("R and dtype must use float32/float64 or complex64/complex128 precision")


def _validate_iterations(iterations: int) -> int:
    """Validate the fixed iteration count without changing its requested value."""
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, Integral):
        raise TypeError(f"iterations must be an integer, got {type(iterations).__name__}")
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    return iterations


def _ldexp_complex(array: np.ndarray, exponent: int) -> np.ndarray:
    """Scale a complex array by an exact power of two without complex overflow."""
    scaled = np.empty_like(array)
    scaled.real[...] = np.ldexp(array.real, exponent)
    scaled.imag[...] = np.ldexp(array.imag, exponent)
    return scaled


def newton_schulz_inverse(
    R: np.ndarray,
    iterations: int = NEWTON_SCHULZ_ITERATIONS,
    *,
    x0: str = X0_IDENTITY_NORM_INF,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Approximate ``R⁻¹`` with a fixed number of Newton-Schulz updates.

    Parameters
    ----------
    R:
        A non-empty square matrix.  Hermitian positive-definiteness is expected
        by the MV caller but is not required to perform the algebra here.
    iterations:
        Exact number of updates to execute. Defaults to the twelve updates
        selected by ADR-0012; callers may override it for sweeps or historical
        experiments. Zero returns the selected initial value, and no residual
        or convergence condition can shorten the loop.
    x0:
        ``"identity_norminf"`` (the ADR-0012 default) for ``I / ||R||∞`` or
        ``"r_h_norms"`` for the historical Stage-1 comparison
        ``Rᴴ / (||R||₁ ||R||∞)``.
    dtype:
        Optional float or complex precision.  Float32 and complex64 select the
        complex64 path; float64 and complex128 select complex128.  If omitted,
        the precision is inferred from ``R``.

    Returns
    -------
    numpy.ndarray
        The approximate inverse in complex64 or complex128, matching the
        selected real precision.
    """
    iterations = _validate_iterations(iterations)
    if x0 not in X0_CHOICES:
        choices = ", ".join(repr(choice) for choice in X0_CHOICES)
        raise ValueError(f"x0 must be one of {choices}, got {x0!r}")

    matrix = np.asarray(R)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"R must be a non-empty square matrix, got shape {matrix.shape}")
    work_dtype = _complex_dtype(matrix, dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.asarray(matrix, dtype=work_dtype)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("R must contain only finite values")

    real_dtype = np.empty((), dtype=work_dtype).real.dtype
    component_max = np.maximum(
        np.max(np.abs(matrix.real)),
        np.max(np.abs(matrix.imag)),
    )
    if component_max == 0:
        raise ValueError("R must not be the zero matrix")

    _, scaling_exponent = np.frexp(component_max)
    scaled_matrix = _ldexp_complex(matrix, -int(scaling_exponent))
    abs_matrix = np.abs(scaled_matrix)
    norm_one = np.max(np.sum(abs_matrix, axis=0, dtype=real_dtype))
    norm_inf = np.max(np.sum(abs_matrix, axis=1, dtype=real_dtype))
    if norm_one == 0 or norm_inf == 0:
        raise ValueError("R must not be the zero matrix")

    if x0 == X0_R_H_NORMS:
        initial = scaled_matrix.conj().T / norm_one / norm_inf
    else:
        initial = np.eye(matrix.shape[0], dtype=work_dtype) / norm_inf

    identity = np.eye(matrix.shape[0], dtype=work_dtype)
    inverse = np.asarray(initial, dtype=work_dtype)
    for _ in range(iterations):
        inverse = inverse @ (2.0 * identity - scaled_matrix @ inverse)

    inverse = _ldexp_complex(inverse, -int(scaling_exponent))
    if not np.all(np.isfinite(inverse)):
        raise ValueError("R inverse must be finite in the selected precision")
    return inverse


__all__ = [
    "NEWTON_SCHULZ_ITERATIONS",
    "X0_CHOICES",
    "X0_IDENTITY_NORM_INF",
    "X0_R_H_NORMS",
    "newton_schulz_inverse",
]
