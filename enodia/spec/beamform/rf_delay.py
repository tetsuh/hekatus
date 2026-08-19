"""Fractional delay on the raw RF record — the golden path's own interpolation.

The RF-domain ideal-delay DAS is the yardstick that quantifies the error of
the IQ + 4-tap approximation (CLAUDE.md, absolute rules). Its own fractional
delays therefore have to sit well below the thing being measured, or the
yardstick contaminates the comparison it exists for (#25). MVP-1 took them by
2-tap linear interpolation on the 40 MHz RF, which under the frozen benchmark
of `rf_delay_sweep.py` carries 6.2 % RMS error at 5 MHz and 38.7 % at
13 MHz — against 10.8 % and 7.9 % for the IQ path it measures.

**Method: band-limited upsampling by 8, then Lagrange cubic.** Stated so that
it can be reproduced to the sample:

1. The record is zero-padded by `ZERO_PAD` = 256 samples at each end, in
   float64. Padding is what keeps the periodic images of step 2 far enough
   from the record that they do not reach it: with none, the residual at
   13 MHz is 0.972 %, over that carrier's 0.791 % acceptance limit; with
   256, 0.099 %.
2. It is upsampled by `UPSAMPLE_FACTOR` = 8 through the real FFT: the
   spectrum of the padded record (length M) is zero-stuffed to length M·8,
   the Nyquist bin halved when M is even, inverse-transformed, and scaled by
   8. This is periodic-sinc (Dirichlet) interpolation of the padded record,
   which is why the padding matters — and it interpolates: every original
   sample is reproduced exactly on the fine grid.
3. The delayed value is read from the fine grid at position 8·t by the
   Lagrange cubic of `interp.py` — the same kernel design.md §5 fixes for
   the IQ side, with the same coordinate convention and zero-extension —
   and the padded region is discarded from the result's frame of reference.
4. The result is cast to `interpolation_dtype(record.dtype)`: float32 for a
   float32 or integer record, float64 for a float64 one (design.md §14).

Under the frozen benchmark this leaves **0.000 % at 5 MHz and 0.099 % at
13 MHz**, both under the acceptance limit of one tenth of the IQ-side
error, and it costs on the order of ten seconds per 128-event frame on the
demo workload against under a second for linear — the fastest by measured
frame time of the costed candidates that reach the limit, by an order of
magnitude, though not the lightest in memory (`rf_delay_sweep.py`). Why no
evaluated kernel of 32 taps or fewer reaches the limit at 13 MHz is recorded
there: the record has −14 dB of energy at Nyquist, a kernel of modest
support has a transition band that energy occupies, and the least-squares
bound on four taps is 16.5 % on the contiguous support and 13.0 % on the
best support drawn from offsets −8 … +9. A 256-tap rectangular sinc does
reach it,
at 64 times the taps of a cubic per sample.

**Boundary.** The record is zero outside [0, N). Because step 2 reconstructs
band-limited, a position within a few samples of either end reads the
band-limited ringing of that zero-extended record into the padding, exactly
as the ideal delay does; a position beyond the padding reads zero.
"""

from __future__ import annotations

import numpy as np

from enodia.spec.beamform.interp import fractional_delay, interpolation_dtype

UPSAMPLE_FACTOR = 8
ZERO_PAD = 256


def upsample_rf(record: np.ndarray, *, factor: int = UPSAMPLE_FACTOR, pad: int = ZERO_PAD):
    """Band-limited upsample of ``(..., n_t)`` by ``factor``; returns the fine
    grid over the padded extent, ``(..., (n_t + 2·pad)·factor)``, in float64.

    Fine sample ``(pad + k)·factor`` is ``record[..., k]`` exactly, for any
    ``factor`` ≥ 1; at 1 the record is returned padded and otherwise intact.
    """
    if factor < 1:
        raise ValueError(f"upsampling factor must be a positive integer, got {factor}")
    z = np.pad(np.asarray(record, dtype=np.float64), [(0, 0)] * (record.ndim - 1) + [(pad, pad)])
    m = z.shape[-1]
    spectrum = np.fft.rfft(z, axis=-1)
    stuffed = np.zeros(z.shape[:-1] + (m * factor // 2 + 1,), dtype=np.complex128)
    stuffed[..., : spectrum.shape[-1]] = spectrum
    if m % 2 == 0 and factor > 1:
        # The Nyquist bin of the coarse grid is shared by ±fs/2; splitting it
        # keeps the fine-grid signal real and the original samples exact. At
        # factor 1 there is nothing to split — the bin stays where it was.
        stuffed[..., spectrum.shape[-1] - 1] *= 0.5
    return np.fft.irfft(stuffed, n=m * factor, axis=-1) * factor


def delay_rf(record: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Read ``record`` at fractional sample ``positions``, one row per channel.

    ``record`` is ``(n_ch, n_t)`` and ``positions`` is ``(n_ch, n_pos)`` in
    samples from the record start; row ``i`` of the positions indexes row
    ``i`` of the record. Returns ``(n_ch, n_pos)`` in
    ``interpolation_dtype(record.dtype)``.
    """
    record = np.asarray(record)
    positions = np.asarray(positions, dtype=np.float64)
    fine = upsample_rf(record)
    fine_positions = (positions + ZERO_PAD) * UPSAMPLE_FACTOR
    return fractional_delay(fine, fine_positions).astype(interpolation_dtype(record.dtype))
