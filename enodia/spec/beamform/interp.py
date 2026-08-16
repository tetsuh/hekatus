"""Fractional-delay interpolation on the decimated IQ record.

design.md §5 writes the delayed channel sample as

    x_i[n] ≈ interp4(z_dec, n − d) · e^(−j2πf0·τ),   d = τ·fs'

and this module is `interp4`. It is defined to the coefficient because L0
acceptance is numerical equivalence with this reference, and two ports that
choose different 4-tap kernels both look right alone and differ exactly where
L0 looks (#22). A port therefore runs the kernel named here, and L0 compares
like with like; a difference between kernels is never absorbed by a wider
tolerance (ADR-0007).

**Kernel: Lagrange cubic** — the four-point Lagrange basis on the nodes
{−1, 0, +1, +2} around the target. It is the maximally-flat fractional-delay
FIR: exact for polynomials to degree 3, so its error vanishes fastest at DC,
where the IQ spectrum has its energy. The sweep that chose it over Keys
(Catmull-Rom), a windowed sinc, and a least-squares design is
`interp_sweep.py`, and its figures are quoted in design.md §5.

**Coordinate convention.** `t = n − d` is a position in samples from the
first sample of the record, which sits at t = 0. Let m = ⌊t⌋ and μ = t − m,
so μ ∈ [0, 1). The four taps multiply z[m−1], z[m], z[m+1], z[m+2], in that
order. μ = 0 reads z[m] exactly.

**Boundary rule.** The record is zero outside [0, N): before the first
sample is pre-transmit, past the last there is no data. A tap that falls
outside contributes zero, so a target near an end is a partial sum, and a
target wholly outside is zero. Not clamped, not mirrored, not extrapolated —
each of those is defensible and each disagrees with the others.

**Precision.** The taps are formed in the record's floating dtype; the L0
threshold of design.md §15 covers what remains.
"""

from __future__ import annotations

import numpy as np

# Closed: one value defined. The argument exists so an L0 comparison names
# the kernel it runs rather than assuming it; widening this set is a change
# to the verification contract (ADR-0007), not a new string.
KERNELS: tuple[str, ...] = ("lagrange4",)

_TAP_OFFSETS = np.array([-1, 0, 1, 2])


def _check_kernel(kernel: str) -> None:
    if kernel not in KERNELS:
        raise ValueError(f"unknown interpolation kernel {kernel!r}; defined: {KERNELS}")


def fractional_delay_taps(mu, *, kernel: str = "lagrange4") -> np.ndarray:
    """The four tap weights for fraction ``mu`` ∈ [0, 1), last axis of length 4.

    Order matches the samples z[m−1], z[m], z[m+1], z[m+2]. Accepts a scalar
    or an array of fractions and returns shape ``mu.shape + (4,)``.
    """
    _check_kernel(kernel)
    mu = np.asarray(mu, dtype=np.result_type(mu, np.float64))
    return np.stack(
        [
            -mu * (mu - 1.0) * (mu - 2.0) / 6.0,
            (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0,
            -(mu + 1.0) * mu * (mu - 2.0) / 2.0,
            (mu + 1.0) * mu * (mu - 1.0) / 6.0,
        ],
        axis=-1,
    )


def fractional_delay(z: np.ndarray, t: np.ndarray, *, kernel: str = "lagrange4") -> np.ndarray:
    """Read record ``z`` at fractional sample positions ``t``.

    ``z`` is ``(..., N)`` — any leading axes (channels, events) — and ``t``
    is ``(..., P)`` with leading axes broadcastable against ``z``'s. Returns
    ``(..., P)`` in ``z``'s dtype (complex stays complex; the same real taps
    apply to I and Q). Positions outside [0, N) read the record as zero.
    """
    _check_kernel(kernel)
    z = np.asarray(z)
    t = np.asarray(t, dtype=np.float64)
    n = z.shape[-1]

    m = np.floor(t)
    mu = t - m
    idx = m.astype(np.int64)[..., None] + _TAP_OFFSETS  # (..., P, 4)
    inside = (idx >= 0) & (idx < n)
    taps = fractional_delay_taps(mu).astype(z.real.dtype, copy=False)
    taps = np.where(inside, taps, 0.0)

    gathered = np.take_along_axis(
        np.broadcast_to(z, np.broadcast_shapes(z.shape[:-1], t.shape[:-1]) + (n,)),
        np.clip(idx, 0, n - 1).reshape(*idx.shape[:-2], -1),
        axis=-1,
    ).reshape(idx.shape)
    return (gathered * taps).sum(axis=-1)
