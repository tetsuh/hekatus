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
    """Round finite float32 values to BF16 with round-to-nearest-even."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def bf16_round_complex(values: np.ndarray) -> np.ndarray:
    """Round complex64 real and imaginary planes independently to BF16."""
    values = np.asarray(values, dtype=np.complex64)
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


def initial_value(matrices: np.ndarray) -> np.ndarray:
    """Return X0 = I / ||R||_inf."""
    matrices = np.asarray(matrices, dtype=np.complex64)
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("matrices must have shape (batch, size, size)")
    norm_inf = np.linalg.norm(matrices, ord=np.inf, axis=(-2, -1))
    identity = np.eye(matrices.shape[-1], dtype=np.complex64)
    return identity[None, :, :] / norm_inf[:, None, None]


def newton_schulz_reference(
    matrices: np.ndarray,
    *,
    iterations: int = NEWTON_SCHULZ_ITERATIONS,
) -> np.ndarray:
    """Run the specified fixed count; there is no convergence-gated exit."""
    if iterations < 0:
        raise ValueError("iterations must not be negative")
    matrices = np.asarray(matrices, dtype=np.complex64)
    x = initial_value(matrices).astype(np.complex64, copy=False)
    identity = np.eye(matrices.shape[-1], dtype=np.complex64)
    for _ in range(iterations):
        x = x @ (2.0 * identity - matrices @ x)
    return x


def inverse_flops(shape: MatmulShape) -> int:
    """Useful FLOPs in one fixed-count inverse for a catalogue shape."""
    if shape.family != "newton_schulz":
        raise ValueError(f"expected a newton_schulz shape, got {shape.family!r}")
    return COMPLEX_MATMULS_PER_INVERSE * total_flops(shape)
