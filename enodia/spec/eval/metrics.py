"""Pure NumPy quality metrics on linear envelope amplitudes, before log compression.

All input arrays, including coordinates, are converted to float64 on entry;
arithmetic uses float64. Equal represented values give equal results across
floating dtypes. Integers beyond 2**53 are unsupported due to conversion loss.
Overflowing intermediate sample ranges or coordinate distances raise ValueError.
"""

import numpy as np


def gcnr(inside: np.ndarray, outside: np.ndarray, *, bins: int = 256) -> float:
    """Return 1 minus histogram overlap on common, equal-width min-to-max edges.

    Each histogram is normalized by its own sample count. Empty inputs or
    non-finite samples raise ValueError; if every value in both inputs is
    equal, return 0.0. A non-finite sample range also raises ValueError.
    """
    inside = np.asarray(inside, dtype=np.float64).ravel()
    outside = np.asarray(outside, dtype=np.float64).ravel()
    if inside.size == 0 or outside.size == 0:
        raise ValueError("inside and outside must be nonempty")
    if not np.all(np.isfinite(inside)) or not np.all(np.isfinite(outside)):
        raise ValueError("inside and outside samples must be finite")
    low = min(inside.min(), outside.min())
    high = max(inside.max(), outside.max())
    if low == high:
        return 0.0
    with np.errstate(over="ignore"):
        span = high - low
    if not np.isfinite(span):
        raise ValueError("sample range must be finite in float64")
    edges = np.linspace(low, high, bins + 1)
    p_in = np.histogram(inside, bins=edges)[0] / inside.size
    p_out = np.histogram(outside, bins=edges)[0] / outside.size
    if np.array_equal(p_in, p_out):
        return 0.0
    # Bound roundoff in the normalized histogram sum, not the input amplitudes.
    return float(np.clip(1.0 - np.minimum(p_in, p_out).sum(), 0.0, 1.0))


def _crossing(db: np.ndarray, coords: np.ndarray, peak: int, step: int, level: float) -> float:
    """Find the first level crossing walking away from the peak in one direction."""
    stop = db.size if step == 1 else -1
    for index in range(peak + step, stop, step):
        if db[index] <= level:
            if np.isneginf(db[index]):
                raise ValueError("outer crossing sample must be nonzero")
            inner = index - step
            crossing = float(
                np.interp(level, [db[index], db[inner]], [coords[index], coords[inner]])
            )
            if not np.isfinite(crossing):
                raise ValueError("crossing coordinate must be finite in float64")
            return crossing
    return float("nan")


def width_at(profile: np.ndarray, coords: np.ndarray, level_db: float) -> float:
    """Return main-lobe width at a level below the peak, in coordinate units.

    Normalize the 1-D envelope to its maximum and convert with 20*log10.
    Interpolate the first crossing on each side linearly in dB, including on
    nonuniform, strictly increasing coordinates. A missing crossing returns
    NaN. Non-finite or negative samples and mismatched lengths raise ValueError.
    Non-finite coordinate distances or result widths also raise ValueError.
    A zero outer bracketing sample raises ValueError; zeros beyond the first crossing are ignored.
    FWHM uses level_db=20*log10(0.5); the -40 dB width uses level_db=-40.0.
    """
    profile = np.asarray(profile, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if profile.ndim != 1 or coords.ndim != 1 or profile.size != coords.size:
        raise ValueError("profile and coords must be 1-D arrays of matching lengths")
    if profile.size == 0:
        raise ValueError("profile must be nonempty")
    if not np.all(np.isfinite(profile)) or np.any(profile < 0):
        raise ValueError("profile samples must be finite and nonnegative")
    if not np.all(np.isfinite(coords)) or np.any(coords[1:] <= coords[:-1]):
        raise ValueError("coords must be finite and strictly increasing")
    with np.errstate(over="ignore"):
        distances = np.diff(coords)
    if not np.all(np.isfinite(distances)):
        raise ValueError("coordinate distances must be finite in float64")
    peak = int(np.argmax(profile))
    if profile[peak] == 0:
        return float("nan")
    with np.errstate(divide="ignore"):
        db = 20.0 * (np.log10(profile) - np.log10(profile[peak]))
    left = _crossing(db, coords, peak, -1, level_db)
    right = _crossing(db, coords, peak, 1, level_db)
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    width = right - left
    if not np.isfinite(width):
        raise ValueError("result width must be finite in float64")
    return width


def speckle_snr(region: np.ndarray) -> float:
    """Return envelope mean / population standard deviation (ddof=0).

    Fewer than two samples, non-finite samples, or zero deviation raises
    ValueError. The Rayleigh reference is sqrt(pi / (4 - pi)), approximately 1.913.
    """
    region = np.asarray(region, dtype=np.float64)
    if region.size < 2:
        raise ValueError("region must contain at least two samples")
    if not np.all(np.isfinite(region)):
        raise ValueError("region samples must be finite")
    if np.all(region == region.flat[0]):
        raise ValueError("region must have nonzero standard deviation")
    # The common scale cancels in mean/std and keeps the variance representable.
    scaled = region / np.max(np.abs(region))
    deviation = np.std(scaled, ddof=0)
    if deviation == 0:
        raise ValueError("region must have nonzero standard deviation")
    return float(np.mean(scaled) / deviation)
