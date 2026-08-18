"""The frozen benchmark behind the golden path's fractional delay (#25).

The RF golden is the yardstick for the IQ + 4-tap approximation, so its own
interpolation error has to sit well below what it measures. This module is
the oracle that error is measured against, the candidates that were weighed,
and the cost of each on the golden's own workload — so that the method
`rf_delay.py` uses is a number rather than an argument.

**The oracle is frozen and discrete-time.** For a carrier f0 in {5, 13} MHz,
the record is 256 samples at 40 MHz of the simulator's own pulse,
`gaussian_pulse((n − 128)/fs, f0, 0.7)`, in float64. The ideal delay of that
finite sampled record is its zero-extended sinc reconstruction,

    ideal(t) = Σ_{k=0}^{255} record[k] · sinc(t − k),

evaluated at t = n − μ for every output sample n and μ ∈ linspace(0, 1, 201).
There is no periodic wrap, clamping, mirroring, or comparison against the
continuous pulse: this benchmark measures interpolation of the same sampled
record #6 consumes, not pre-ADC fidelity or AFE aliasing. A candidate is
scored on the same grid with the same zero-extended boundary rule, as

    residual = sqrt( Σ (candidate − ideal)² / Σ ideal² ).

The acceptance limit a candidate has to stay under is one tenth of the
IQ-side error it will be used to measure: **1.082 % at 5 MHz and 0.791 % at
13 MHz** (design.md §5, the −6 dB pulse-weighted error at D=8 and D=2). Both
are pinned by `tests/test_rf_golden_interp.py`. The residual a candidate
actually reaches is the *floor* of any comparison made with it — the
smallest difference that comparison can see.

The candidates other than the production method live here and nowhere else.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable

import numpy as np
from scipy.signal import firwin, resample_poly

from enodia.spec.beamform.interp import fractional_delay
from enodia.spec.beamform.rf_delay import delay_rf
from enodia.spec.sim import gaussian_pulse

BENCHMARK_N = 256
BENCHMARK_FS_HZ = 40e6
BENCHMARK_BANDWIDTH_FRAC = 0.7
BENCHMARK_CARRIERS_HZ: dict[str, float] = {"5MHz": 5e6, "13MHz": 13e6}
BENCHMARK_FRACTIONS = 201

# One tenth of the IQ + 4-tap error the golden is used to measure: the −6 dB
# pulse-weighted figures of design.md §5 at 5 MHz D=8 (10.82 %) and 13 MHz
# D=2 (7.91 %).
RESIDUAL_LIMIT_PCT: dict[str, float] = {"5MHz": 1.082, "13MHz": 0.791}


def benchmark_record(f0_hz: float) -> np.ndarray:
    """The frozen record: the simulator's pulse, centred, 256 samples at 40 MHz."""
    n = np.arange(BENCHMARK_N)
    return gaussian_pulse((n - BENCHMARK_N // 2) / BENCHMARK_FS_HZ, f0_hz, BENCHMARK_BANDWIDTH_FRAC)


def benchmark_positions() -> np.ndarray:
    """t = n − μ for every output sample n and μ on the frozen grid; (μ, n)."""
    mus = np.linspace(0.0, 1.0, BENCHMARK_FRACTIONS)
    return np.arange(BENCHMARK_N)[None, :] - mus[:, None]


def oracle(record: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Zero-extended sinc reconstruction of the finite record at positions t."""
    k = np.arange(record.shape[-1])
    return np.sinc(t[..., None] - k) @ record


def residual_pct(candidate: Callable[[np.ndarray, np.ndarray], np.ndarray], record) -> float:
    """The frozen metric, in percent, for a candidate on a benchmark record.

    ``candidate(record, t)`` takes a 1-D record and positions of any shape.
    """
    t = benchmark_positions()
    ideal = oracle(record, t)
    got = np.asarray(candidate(record, t), dtype=np.float64)
    return 100.0 * float(np.sqrt(np.sum((got - ideal) ** 2) / np.sum(ideal**2)))


# --- candidates --------------------------------------------------------------


def _gather0(record: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = record.shape[-1]
    inside = (idx >= 0) & (idx < n)
    return np.where(inside, record[np.clip(idx, 0, n - 1)], 0.0)


def _lagrange_taps(mu: np.ndarray, points: int) -> tuple[np.ndarray, np.ndarray]:
    """P-point Lagrange basis on nodes −(P/2 − 1) … P/2 around the target."""
    nodes = np.arange(-(points // 2 - 1), points // 2 + 1)
    taps = np.ones(mu.shape + (points,))
    for j, nj in enumerate(nodes):
        for ni in nodes:
            if ni != nj:
                taps[..., j] *= (mu - ni) / (nj - ni)
    return taps, nodes


def lagrange(points: int) -> Callable:
    def f(record, t):
        m = np.floor(t)
        taps, nodes = _lagrange_taps(t - m, points)
        return (_gather0(record, m.astype(np.int64)[..., None] + nodes) * taps).sum(-1)

    return f


def windowed_sinc(points: int, window: Callable[[np.ndarray], np.ndarray]) -> Callable:
    """P-tap sinc under ``window(x)``, x ∈ (−1, 1) over the support, unit DC gain."""
    nodes = np.arange(-(points // 2 - 1), points // 2 + 1)

    def f(record, t):
        m = np.floor(t)
        d = nodes - (t - m)[..., None]
        h = np.sinc(d) * window(d / (points / 2))
        h /= h.sum(-1, keepdims=True)
        return (_gather0(record, m.astype(np.int64)[..., None] + nodes) * h).sum(-1)

    return f


def kaiser(beta: float) -> Callable[[np.ndarray], np.ndarray]:
    def w(x):
        inside = np.abs(x) < 1.0
        return np.where(
            inside, np.i0(beta * np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))) / np.i0(beta), 0.0
        )

    return w


def rectangular(x: np.ndarray) -> np.ndarray:
    return (np.abs(x) < 1.0).astype(np.float64)


def polyphase_upsample(factor: int, half_len: int, beta: float, then: Callable) -> Callable:
    """Kaiser-windowed-sinc polyphase upsampling by ``factor``, then ``then`` on
    the fine grid. ``half_len`` is the filter half-length in input samples."""
    taps = firwin(2 * half_len * factor + 1, 1.0 / factor, window=("kaiser", beta))
    pad = half_len + 8

    def f(record, t):
        z = np.concatenate([np.zeros(pad), record, np.zeros(pad)])
        up = resample_poly(z, factor, 1, window=taps)[pad * factor : (pad + record.size) * factor]
        return then(up, t * factor)

    return f


def least_squares_4tap_bound(record: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The best any real 4-tap zero-extended kernel can do on this record.

    For each fraction, the four taps are fitted by least squares to the
    oracle over every output position — a bound on the whole family, not a
    kernel a port could run, since it is fitted to the record it is scored
    on. If this misses the acceptance limit, no four-tap RF kernel meets it.
    """
    m = np.floor(t)
    idx = m.astype(np.int64)[..., None] + np.array([-1, 0, 1, 2])
    gathered = _gather0(record, idx)  # (mu, n, 4)
    ideal = oracle(record, t)  # (mu, n)
    out = np.empty_like(ideal)
    for i in range(t.shape[0]):
        h, *_ = np.linalg.lstsq(gathered[i], ideal[i], rcond=None)
        out[i] = gathered[i] @ h
    return out


def production(record: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The operator the golden path actually runs, scored on one record.

    `delay_rf` takes a channel stack with one row of positions per channel;
    the benchmark's (μ, n) grid is fed to it as μ rows of the same record.
    """
    rows = np.broadcast_to(record, (t.shape[0], record.size))
    return delay_rf(rows, t)


CANDIDATES: dict[str, Callable] = {
    "linear2": lagrange(2),
    "lagrange4": lambda record, t: fractional_delay(record, t),
    "lagrange8": lagrange(8),
    "lagrange16": lagrange(16),
    "kaiser8_sinc16": windowed_sinc(16, kaiser(8.0)),
    "kaiser8_sinc32": windowed_sinc(32, kaiser(8.0)),
    "rect_sinc256": windowed_sinc(256, rectangular),
    "poly_up4_kaiser4_hl320_lagrange4": polyphase_upsample(
        4, 320, 4.0, lambda up, tt: fractional_delay(up, tt)
    ),
    "production": production,
}


def residual_table() -> str:
    lines = [f"{'method':36s}" + "".join(f"{c:>10s}" for c in BENCHMARK_CARRIERS_HZ)]
    for name, fn in {**CANDIDATES, "ls4_bound": least_squares_4tap_bound}.items():
        row = f"{name:36s}"
        for f0 in BENCHMARK_CARRIERS_HZ.values():
            row += f"{residual_pct(fn, benchmark_record(f0)):9.3f}%"
        lines.append(row)
    lines.append(
        f"{'acceptance limit':36s}" + "".join(f"{v:9.3f}%" for v in RESIDUAL_LIMIT_PCT.values())
    )
    return "\n".join(lines)


# --- cost on the golden's workload ------------------------------------------


def golden_workload(n_ch: int = 128, n_t: int = 3373, n_pos: int = 3012, seed: int = 0):
    """One transmit event of the 5 MHz demo: channels × RF samples in, channels
    × depth positions out, positions monotone in depth as the golden's are."""
    rng = np.random.default_rng(seed)
    record = rng.standard_normal((n_ch, n_t)).astype(np.float32)
    positions = np.linspace(104.0, 3196.0, n_pos)[None, :] + rng.uniform(0, 1, (n_ch, 1))
    return record, positions


def _rowwise(fn1d: Callable) -> Callable:
    """Lift a single-record candidate to a channel stack, one row at a time."""

    def f(record, positions):
        return np.stack([fn1d(record[i], positions[i]) for i in range(record.shape[0])])

    return f


def _linear_channel_stack(record, positions):
    """MVP-1's operator, as `das_rf_golden` ran it: clamped, in the record dtype."""
    n_t = record.shape[-1]
    rows = np.arange(record.shape[0])[:, None]
    i0 = np.clip(np.floor(positions).astype(np.int64), 0, n_t - 2)
    frac = (positions - i0).astype(record.dtype)
    return (1.0 - frac) * record[rows, i0] + frac * record[rows, i0 + 1]


def _polyphase_channel_stack(factor: int, half_len: int, beta: float) -> Callable:
    taps = firwin(2 * half_len * factor + 1, 1.0 / factor, window=("kaiser", beta))
    pad = half_len + 8

    def f(record, positions):
        z = np.pad(record.astype(np.float64), ((0, 0), (pad, pad)))
        up = resample_poly(z, factor, 1, window=taps, axis=-1)
        up = up[..., pad * factor : (pad + record.shape[-1]) * factor]
        return fractional_delay(up, positions * factor).astype(record.dtype)

    return f


COSTED: dict[str, Callable] = {
    "linear (MVP-1)": _linear_channel_stack,
    "lagrange4 direct": lambda record, positions: fractional_delay(record, positions),
    "lagrange8 direct": _rowwise(CANDIDATES["lagrange8"]),
    "lagrange16 direct": _rowwise(CANDIDATES["lagrange16"]),
    "kaiser8_sinc16 direct": _rowwise(CANDIDATES["kaiser8_sinc16"]),
    "kaiser8_sinc32 direct": _rowwise(CANDIDATES["kaiser8_sinc32"]),
    "rect_sinc256 direct": _rowwise(CANDIDATES["rect_sinc256"]),
    "poly_up4_kaiser4_hl320 + lagrange4": _polyphase_channel_stack(4, 320, 4.0),
    "production (fft up8 pad256 + lagrange4)": delay_rf,
}


def cost(
    operator: Callable[[np.ndarray, np.ndarray], np.ndarray], *, repeats: int = 3
) -> tuple[float, float]:
    """(seconds per event, peak MiB) for a channel-stack operator on the workload."""
    record, positions = golden_workload()
    operator(record, positions)  # warm
    tracemalloc.start()
    best = np.inf
    for _ in range(repeats):
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        operator(record, positions)
        best = min(best, time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return best, peak / 2**20


def cost_table(*, events_per_frame: int = 128) -> str:
    """Runtime and peak memory of each costed operator on one event of the
    demo workload, and the frame it implies. Machine-dependent by nature, so
    reported and not pinned."""
    lines = [f"{'operator':42s} {'s/event':>8s} {'s/frame':>8s} {'peak MiB':>9s}"]
    for name, op in COSTED.items():
        seconds, mib = cost(op)
        lines.append(f"{name:42s} {seconds:8.3f} {seconds * events_per_frame:8.1f} {mib:9.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Residual under the frozen oracle:")
    print(residual_table())
    print()
    print("Cost on one 5 MHz demo event (128 ch x 3373 samples -> 128 x 3012):")
    print(cost_table())
