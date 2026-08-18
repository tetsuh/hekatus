"""Fractional delay on the raw RF record — the golden path's own interpolation.

The RF-domain ideal-delay DAS is the yardstick that quantifies the error of
the IQ + 4-tap approximation (CLAUDE.md, absolute rules). Its own fractional
delays therefore have to be much better than the thing being measured, or the
yardstick contaminates the comparison it exists for (#25).

This module is that operator, lifted out of `das_rf_golden` so that it can be
measured against a fixed oracle on its own. The method it currently uses is
**2-tap linear interpolation on the sampled RF** — the MVP-1 approximation
this issue exists to replace.
"""

from __future__ import annotations

import numpy as np


def delay_rf(record: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Read ``record`` at fractional sample ``positions``, one row per channel.

    ``record`` is ``(n_ch, n_t)`` and ``positions`` is ``(n_ch, n_pos)`` in
    samples from the record start; row ``i`` of the positions indexes row
    ``i`` of the record. Returns ``(n_ch, n_pos)`` in the record's dtype.

    Linear interpolation between the two neighbouring samples. Positions
    past the last pair are clamped to it, so a position beyond the record
    extrapolates from the last two samples rather than reading zero.
    """
    record = np.asarray(record)
    n_t = record.shape[-1]
    rows = np.arange(record.shape[0])[:, None]
    i0 = np.clip(np.floor(positions).astype(np.int64), 0, n_t - 2)
    frac = (positions - i0).astype(record.dtype)
    return (1.0 - frac) * record[rows, i0] + frac * record[rows, i0 + 1]
