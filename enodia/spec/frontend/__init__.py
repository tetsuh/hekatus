"""Front end: complex band-pass FIR fused with the mixer, then decimation (design.md §5, #6).

Raw RF does not fit in the accelerator's L1, so every delay downstream is
applied to IQ. The front end is what makes that possible and it is defined
here as the reference every port is compared with at L0 checkpoint 1
(design.md §15).

**Processing.** For a channel record s[n] at fs:

    y[n]   = Σ_k h_bp[k] · s[n − k]          complex band-pass FIR, L taps
    z[m]   = y[m·D + L//2] · e^(−j2πf0·(m·D + δ)/fs)    decimate by D, rotate to baseband

where `h_bp[k] = h_lp[k] · e^(+j2πf0·(k − (L−1)/2)/fs)` is a real low-pass
prototype translated to +f0 with the modulation centred on the filter —
the mixer `s(t)·e^(−j2πf0·t)` of design.md §5 and the low-pass *fuse* into
this one complex filter, and what remains per output sample is one complex
rotation at the RF time the sample stands for. (Centring the modulation is
what makes that so: referenced to tap 0 instead, every IQ sample carries a
constant phase error of 2πf0·(L−1)/2/fs — 22.5° for 64 taps at fs = 8f0 —
that no image would show. Found by the checkpoint-2 comparison, #6.) At
fs = 8·f0 and D = 8 the rotation per sample is e^(−jπ/4)·(1)^m up to the
half-sample term below; at D = 4, (−1)^m times the same constant.

**Gain.** The prototype has DC gain 2, so z is the complex envelope of s at
s's own amplitude: a tone A·cos(2πf0·t) becomes the phasor A. (Unit gain
would halve it — the band-pass keeps only the positive-frequency half of a
real signal — and the checkpoint-2 comparison would read that as a 50 %
error against the analytic golden, which is how it was found.)

**Alignment.** A symmetric FIR of L taps delays by (L − 1)/2 samples. Taking
y at m·D + L//2 puts IQ sample m at RF-sample position m·D + δ with
δ = L//2 − (L − 1)/2 — one half sample for an even L such as the 64 of
design.md §5, zero for an odd one. δ travels with the record as
`IQEventRecord.rf_offset`, and the delay stage reads the IQ record at
(p − δ)/D for an RF position p. Stated to the half sample because a delay
error of half an RF sample is 12.5 ns — 23° at 5 MHz — and would otherwise
be invisible in an image that merely looks plausible.

**The low-pass prototype** is a windowed sinc with cutoff `cutoff_frac`
times the decimated Nyquist frequency fs/(2D), DC gain 2 as above, Hann
window by default; L = 64. These are design parameters (CLAUDE.md: every parameter is
sweepable) and the golden comparison (`enodia.spec.beamform.golden_compare`)
is what says what they cost; nothing here is tuned by argument.

**Sign convention (normative).** z is the baseband of the positive-frequency
half of s: a tone at f0 + Δ becomes a phasor turning forward at +Δ. The
delay stage's rotation e^(−j2πf0·τ) is its counterpart (design.md §5) and
`tests/test_frontend.py` asserts the sign here, `tests/test_iq_das.py` at
checkpoint 2 — flipping either still produces a plausible image, which is
the warning §5 gives.

**Output format.** `demodulate` returns an `IQEventRecord`: int16 complex as
two int16 planes (docs/dataplane.md T1, design.md §14). The IQ is scaled by
`iq_scale` before rounding, and a value that would not fit int16 raises
rather than clipping silently. The unquantized complex output is available
from `complex_bpf_decimate` for sweeps and for the comparison that
separates the filter's error from the quantization's.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import get_window

from enodia.spec.probe import ProbeProfile
from enodia.spec.records import IQEventRecord, RFEventRecord

N_TAPS = 64
CUTOFF_FRAC = 1.0  # of the decimated Nyquist frequency fs / (2D)
WINDOW = "hann"
IQ_SCALE = 1.0


ANALYTIC_GAIN = 2.0  # the complex envelope carries the real signal's amplitude


def lowpass_taps(n_taps: int, cutoff_cycles: float, *, window: str = WINDOW) -> np.ndarray:
    """Windowed-sinc low-pass prototype, ``n_taps`` real taps, cutoff in
    cycles per sample, normalized to DC gain ``ANALYTIC_GAIN`` = 2. float64."""
    if n_taps < 2:
        raise ValueError(f"n_taps must be at least 2, got {n_taps}")
    if not 0.0 < cutoff_cycles <= 0.5:
        raise ValueError(f"cutoff must be in (0, 0.5] cycles/sample, got {cutoff_cycles}")
    k = np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0
    h = (
        2.0
        * cutoff_cycles
        * np.sinc(2.0 * cutoff_cycles * k)
        * get_window(window, n_taps, fftbins=False)
    )
    return ANALYTIC_GAIN * h / h.sum()


def bpf_taps(
    profile: ProbeProfile,
    decimation: int,
    *,
    n_taps: int = N_TAPS,
    cutoff_frac: float = CUTOFF_FRAC,
    window: str = WINDOW,
) -> np.ndarray:
    """The fused complex band-pass taps: the low-pass prototype translated to +f0.

    complex128, ``(n_taps,)``; tap k multiplies s[n − k]. Held per probe and
    per decimation ratio (design.md §4, §5).
    """
    cutoff_cycles = cutoff_frac * 0.5 / decimation
    h_lp = lowpass_taps(n_taps, cutoff_cycles, window=window)
    k = np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0  # modulation centred
    return h_lp * np.exp(2j * np.pi * profile.f0_hz * k / profile.fs_hz)


def rf_offset(n_taps: int) -> float:
    """RF-sample position of IQ sample 0: the FIR's half-sample residue, δ."""
    return n_taps // 2 - (n_taps - 1) / 2.0


def n_iq_samples(n_rf: int, decimation: int) -> int:
    return n_rf // decimation


def complex_bpf_decimate(
    rf: np.ndarray,
    profile: ProbeProfile,
    *,
    decimation: int,
    n_taps: int = N_TAPS,
    cutoff_frac: float = CUTOFF_FRAC,
    window: str = WINDOW,
) -> np.ndarray:
    """RF ``(n_ch, n_t)`` → baseband IQ ``(n_ch, n_t // D)`` in complex128.

    IQ sample m stands for RF-sample position m·D + rf_offset(n_taps).
    """
    if int(decimation) != decimation or decimation < 1:
        raise ValueError(f"decimation must be a positive integer, got {decimation}")
    rf = np.asarray(rf, dtype=np.float64)
    if rf.ndim != 2:
        raise ValueError(f"rf must be (channel, sample), got shape {rf.shape}")
    n_t = rf.shape[-1]
    h = bpf_taps(profile, decimation, n_taps=n_taps, cutoff_frac=cutoff_frac, window=window)
    # Full linear convolution: y_full[i] is the filter centred on RF time
    # i − (L−1)/2, the record being zero outside [0, n_t).
    y_full = fftconvolve(rf.astype(np.complex128), h[None, :], mode="full", axes=-1)
    m = np.arange(n_iq_samples(n_t, decimation))
    y = y_full[:, m * decimation + n_taps // 2]
    t_rf = (m * decimation + rf_offset(n_taps)) / profile.fs_hz
    return y * np.exp(-2j * np.pi * profile.f0_hz * t_rf)[None, :]


def demodulate(
    record: RFEventRecord,
    profile: ProbeProfile,
    *,
    decimation: int,
    n_taps: int = N_TAPS,
    cutoff_frac: float = CUTOFF_FRAC,
    window: str = WINDOW,
    iq_scale: float = IQ_SCALE,
) -> IQEventRecord:
    """One RF record → one int16 complex IQ record, header carried through."""
    z = complex_bpf_decimate(
        record.data,
        profile,
        decimation=decimation,
        n_taps=n_taps,
        cutoff_frac=cutoff_frac,
        window=window,
    )
    scaled = z * iq_scale
    limit = np.iinfo(np.int16).max
    peak = max(float(np.abs(scaled.real).max()), float(np.abs(scaled.imag).max()))
    if peak > limit:
        # Clipping would be a silent nonlinearity in the one place the
        # dataflow is supposed to be linear (design.md §14); say so instead.
        raise ValueError(
            f"IQ peak {peak:.0f} exceeds int16 at iq_scale={iq_scale}; lower the scale"
        )
    planes = np.stack([np.round(scaled.real), np.round(scaled.imag)], axis=-1).astype(np.int16)
    return IQEventRecord(
        header=record.header, data=planes, decimation=decimation, rf_offset=rf_offset(n_taps)
    )


def demodulate_frame(
    records: list[RFEventRecord], profile: ProbeProfile, *, decimation: int, **kwargs
) -> list[IQEventRecord]:
    return [demodulate(r, profile, decimation=decimation, **kwargs) for r in records]
