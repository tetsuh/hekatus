"""NumPy reference and accounting for the fixed Newton-Schulz inverse.

This module deliberately has no dependency on the accelerator toolchain.  It
is the L0 oracle for the hand-written kernel and can be reviewed and tested on
any host.
"""

from __future__ import annotations

import numpy as np

from enodia.tt.bench.shapes import MatmulShape, total_flops

NEWTON_SCHULZ_ITERATIONS = 12
COMPLEX_MATMULS_PER_INVERSE = 2 * NEWTON_SCHULZ_ITERATIONS


def bf16_round_to_float32(values: np.ndarray) -> np.ndarray:
    """Round finite real floating values to BF16 with RNE."""
    values = np.asarray(values)
    if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
        raise ValueError("values must have a float32 or float64 dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    values = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("values must remain finite after float32 conversion")
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def bf16_round_complex(values: np.ndarray) -> np.ndarray:
    """Round complex floating planes independently to BF16."""
    values = np.asarray(values)
    if values.dtype.kind != "c" or values.dtype.itemsize not in (8, 16):
        raise ValueError("values must have a complex64 or complex128 dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    values = values.astype(np.complex64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("values must remain finite after complex64 conversion")
    return (
        bf16_round_to_float32(values.real) + 1j * bf16_round_to_float32(values.imag)
    ).astype(np.complex64)


def random_hpd_batch(
    batch: int,
    size: int,
    *,
    condition_number: float = 100.0,
    seed: int = 0,
) -> np.ndarray:
    """Return random complex Hermitian positive-definite matrices.

    A fixed orthogonal basis and random diagonal phases make the construction
    inexpensive enough for the required batch of 8192 while preserving the
    requested eigenvalues exactly under unitary similarity.
    """
    if batch < 1 or size < 1:
        raise ValueError("batch and size must be positive")
    if not np.isfinite(condition_number) or condition_number < 1.0:
        raise ValueError("condition_number must be finite and at least one")

    indices = np.arange(size, dtype=np.float64)
    basis = np.cos(np.pi * (indices[:, None] + 0.5) * indices[None, :] / size)
    basis[:, 0] *= np.sqrt(1.0 / size)
    if size > 1:
        basis[:, 1:] *= np.sqrt(2.0 / size)
    eigenvalues = np.geomspace(1.0, condition_number, size)
    base = (basis * eigenvalues) @ basis.T

    rng = np.random.default_rng(seed)
    phases = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(batch, size)))
    matrices = base[None, :, :] * phases[:, :, None] * phases[:, None, :].conj()
    return matrices.astype(np.complex64)


def _canonicalize_matrices(matrices: np.ndarray) -> np.ndarray:
    """Canonicalize and validate batched square matrices."""
    values = np.asarray(matrices)
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1] < 1
        or values.shape[1] != values.shape[2]
    ):
        raise ValueError("matrices must have shape (batch, size, size)")
    try:
        canonical = values.astype(np.complex64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("matrices must be numeric") from exc
    if not np.all(np.isfinite(canonical)):
        raise ValueError("matrices must be finite")
    norm_inf = np.linalg.norm(canonical, ord=np.inf, axis=(-2, -1))
    if not np.all(np.isfinite(norm_inf)) or np.any(norm_inf <= 0.0):
        raise ValueError("matrices must have finite, non-zero infinity norms")
    return canonical


def _validate_iterations(iterations: int) -> int:
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, (int, np.integer)
    ):
        raise TypeError("iterations must be a non-negative integer")
    if iterations < 0:
        raise ValueError("iterations must not be negative")
    return int(iterations)


def _canonicalize_x0(x0: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(x0)
    if values.shape != matrices.shape:
        raise ValueError(f"x0 must have shape {matrices.shape}, got {values.shape}")
    if values.dtype.kind != "c" or values.dtype.itemsize not in (8, 16):
        raise ValueError("x0 must have a complex64 or complex128 dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("x0 must be finite")
    return values.astype(np.complex64, copy=False)


def initial_value(matrices: np.ndarray) -> np.ndarray:
    """Return X0 = I / ||R||_inf."""
    matrices = _canonicalize_matrices(matrices)
    norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
    identity = np.eye(matrices.shape[-1], dtype=np.complex64)
    return identity[None, :, :] / norm_inf[:, None, None]


def newton_schulz_reference(
    matrices: np.ndarray,
    *,
    iterations: int = NEWTON_SCHULZ_ITERATIONS,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Run a fixed count, optionally with an explicitly supplied X0."""
    iterations = _validate_iterations(iterations)
    matrices = _canonicalize_matrices(matrices)
    x = initial_value(matrices) if x0 is None else _canonicalize_x0(x0, matrices)
    identity = np.eye(matrices.shape[-1], dtype=np.complex64)
    for _ in range(iterations):
        x = x @ (2.0 * identity - matrices @ x)
    return x


def inverse_flops(shape: MatmulShape) -> int:
    """Useful FLOPs in one fixed-count inverse for a catalogue shape."""
    if shape.family != "newton_schulz":
        raise ValueError(f"expected a newton_schulz shape, got {shape.family!r}")
    return COMPLEX_MATMULS_PER_INVERSE * total_flops(shape)
