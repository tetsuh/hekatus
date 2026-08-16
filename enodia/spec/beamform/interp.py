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
FIR: exact for polynomials to degree 3, so its error vanishes as the pulse
narrows, where kernels tuned to the band edge stall. It is not best at every
operating point — `interp_sweep.py` shows where Keys with a < −1/2 beats it —
and it is chosen for being robust across the decimation ratio (§17) and the
pulse bandwidth (#46), both of which are open. design.md §5 carries the
evidence, and the choice is provisional on the axial-PSF measurement §5
asks for. What is *not* provisional is that the reference and every port run
the same kernel, whichever it is (ADR-0007).

**Coordinate convention.** `t = n − d` is a position in samples from the
first sample of the record, which sits at t = 0. Let m = ⌊t⌋ and μ = t − m,
so μ ∈ [0, 1). The four taps multiply z[m−1], z[m], z[m+1], z[m+2], in that
order. μ = 0 reads z[m] exactly.

**Boundary rule: the record is zero outside [0, N).** Before the first
sample is pre-transmit; past the last there is no data. The rule is about
the *taps*, not about the position: a tap landing outside contributes
nothing, so a position within two samples of an end reads a partial sum of
the taps that are still inside, and only a position whose four taps all fall
outside — t < −2 or t ≥ N+1 — reads zero. Not clamped, not mirrored, not
extrapolated; each of those is defensible and each disagrees with the others.

**Precision.** design.md §14 fixes the dataflow: the IQ record is int16
complex and interpolation runs at FP32 intermediate. An integer record is
therefore promoted rather than interpolated in its own type — integer taps
would truncate every fractional weight to zero, which is a silently black
image and not a lower-precision one. A floating record keeps its own
precision, since dtype is a swept parameter. The taps themselves are formed
in float64 and cast down on application, for the reason §14 gives for
geometry and delay tables: they are cheap, and computing them exactly keeps
"the reference's own rounding" off the list of suspects when a port
disagrees.
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
    or an array of fractions and returns shape ``mu.shape + (4,)``, always in
    float64: the weights are a delay table, and design.md §14 keeps those
    exact so the reference's own rounding is never the explanation for a
    disagreement.
    """
    _check_kernel(kernel)
    mu = np.asarray(mu, dtype=np.float64)
    return np.stack(
        [
            -mu * (mu - 1.0) * (mu - 2.0) / 6.0,
            (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0,
            -(mu + 1.0) * mu * (mu - 2.0) / 2.0,
            (mu + 1.0) * mu * (mu - 1.0) / 6.0,
        ],
        axis=-1,
    )


def interpolation_dtype(record_dtype) -> np.dtype:
    """The dtype interpolation runs in for a record of ``record_dtype``.

    design.md §14: int16 complex IQ, FP32 intermediate. An integer record is
    promoted to floating point; a floating record keeps its own precision.
    """
    return np.result_type(record_dtype, np.float32)


def fractional_delay(z: np.ndarray, t: np.ndarray, *, kernel: str = "lagrange4") -> np.ndarray:
    """Read record ``z`` at fractional sample positions ``t``.

    ``z`` is ``(..., N)`` — any leading axes, typically channels — and ``t``
    is ``(..., P)``. The leading axes are broadcast against each other, so a
    single set of positions may be applied to every channel. Returns
    ``(broadcast leading axes) + (P,)`` in ``interpolation_dtype(z.dtype)``;
    complex stays complex, with the same real taps on I and Q. Taps landing
    outside ``[0, N)`` contribute zero.
    """
    _check_kernel(kernel)
    z = np.asarray(z)
    t = np.asarray(t, dtype=np.float64)
    if z.ndim == 0 or t.ndim == 0:
        raise ValueError("z and t must each have at least one axis (record and positions)")
    n = z.shape[-1]
    out_dtype = interpolation_dtype(z.dtype)

    lead = np.broadcast_shapes(z.shape[:-1], t.shape[:-1])
    z = np.broadcast_to(z, (*lead, n))
    t = np.broadcast_to(t, (*lead, t.shape[-1]))
    n_pos = t.shape[-1]

    m = np.floor(t)
    idx = m.astype(np.int64)[..., None] + _TAP_OFFSETS  # (*lead, P, 4)
    taps = np.where(
        (idx >= 0) & (idx < n),
        fractional_delay_taps(t - m, kernel=kernel),
        0.0,
    ).astype(out_dtype.type(0).real.dtype, copy=False)

    gathered = np.take_along_axis(
        z, np.clip(idx, 0, n - 1).reshape(*lead, n_pos * 4), axis=-1
    ).reshape(*lead, n_pos, 4)
    return (gathered.astype(out_dtype, copy=False) * taps).sum(axis=-1)
