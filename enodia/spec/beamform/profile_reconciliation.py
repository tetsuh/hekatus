"""Profile-specific reconciliation beside the frozen RF oracle (design.md §15, #46).

`rf_delay_sweep.py` freezes one benchmark — `rf-oracle-frozen-0p7`: 256
samples at 40 MHz of the simulator's pulse at 0.7 fractional bandwidth, for
5 and 13 MHz — so that the golden path's own residual is a number that
cannot drift with the session. That record is historical synthetic evidence.
It is not a probe profile, and it stays what it is even when a profile
happens to carry the same 0.7.

What a *profile* implies is a separate output, produced here: for a named
`ProbeProfile` and a decimation ratio, the profile's own provenance
(name, status, source, carrier, bandwidth fraction and edge, spectral level,
width convention), the revision that produced the figures, the
profile-specific IQ-side result — the Lagrange cubic's band-edge and
pulse-weighted error from `interp_sweep.py` at the profile's edge — and
the golden operator's residual on a record built from the profile,
together with whether that record is numerically the frozen one. For
`linear-5mhz` at 5 MHz it is: same pulse, same 0.7, so the RF residual is
the frozen 0.0003 % — while the IQ-side figure differs from what §5 said
before #46, because the former 1.5 MHz edge was inconsistent with that
pulse.

**#10 is the trigger for the 13 MHz reconciliation**: when it supplies a
13 MHz profile under §4's bandwidth definition, this module is run on it,
and the result is reported beside — never in place of — the frozen 13 MHz
oracle figure. A change to a profile's value or provenance reruns its
reconciliation and whatever consumed it (#6).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np

from enodia.spec.beamform import rf_delay_sweep as rf
from enodia.spec.beamform.interp_sweep import (
    CANDIDATES,
    profile_case,
    pulse_weighted_rms_error,
    worst_band_edge_error,
)
from enodia.spec.probe import BANDWIDTH_LEVEL_DB, ProbeProfile, linear_5mhz

WIDTH_CONVENTION = "full width at half amplitude (-6.0206 dB); one-sided edge = half"


@dataclass(frozen=True)
class ProfileReconciliation:
    """One profile's result at one decimation ratio, with its provenance.

    Every field is part of the record: a figure from here is not quotable
    without the profile, its status and source, the level and convention
    the bandwidth is stated at, and the revision that produced it.
    """

    profile: str
    status: str
    source: str | None
    f0_hz: float
    bandwidth_frac: float
    bandwidth_edge_hz: float
    level_db: float
    convention: str
    decimation: int
    producing_revision: str
    iq_kernel: str
    iq_worst_phase_deg: float
    iq_worst_magnitude_error: float
    iq_weighted_rms_pct: float
    rf_golden_residual_pct: float
    rf_record_is_frozen_oracle: bool
    frozen_oracle: str = rf.BENCHMARK_NAME

    def lines(self) -> list[str]:
        source = self.source or "none"
        iq = (
            f"  IQ ({self.iq_kernel}): worst phase {self.iq_worst_phase_deg:.2f} deg, "
            f"worst |H|-1 {self.iq_worst_magnitude_error * 100:.2f} %, "
            f"pulse-weighted RMS {self.iq_weighted_rms_pct:.4f} %"
        )
        golden = (
            f"  RF golden residual on the profile's record: {self.rf_golden_residual_pct:.4f} % "
            f"(record is the frozen oracle's: {self.rf_record_is_frozen_oracle})"
        )
        band = (
            f"  f0: {self.f0_hz / 1e6:g} MHz  bandwidth_frac: {self.bandwidth_frac:g}  "
            f"edge: {self.bandwidth_edge_hz / 1e6:g} MHz  level: -{self.level_db:.4f} dB"
        )
        return [
            f"profile-reconciliation ({self.frozen_oracle} is a separate, frozen record)",
            f"  profile: {self.profile}  status: {self.status}  source: {source}",
            band,
            f"  convention: {self.convention}",
            f"  decimation: {self.decimation}  producing revision: {self.producing_revision}",
            iq,
            golden,
        ]


def producing_revision() -> str:
    """The git revision of the tree that produced a result, or ``unknown``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out or "unknown"


def profile_record(profile: ProbeProfile) -> np.ndarray:
    """The benchmark-shaped record the profile implies: the simulator's pulse
    at the profile's carrier and bandwidth, centred, `BENCHMARK_N` samples at
    the profile's sampling rate."""
    from enodia.spec.sim import gaussian_pulse

    n = np.arange(rf.BENCHMARK_N)
    return gaussian_pulse(
        (n - rf.BENCHMARK_N // 2) / profile.fs_hz, profile.f0_hz, profile.bandwidth_frac
    )


def _is_frozen_record(profile: ProbeProfile, record: np.ndarray) -> bool:
    if profile.fs_hz != rf.BENCHMARK_FS_HZ:
        return False
    if profile.f0_hz not in rf.BENCHMARK_CARRIERS_HZ.values():
        return False
    return bool(np.array_equal(record, rf.benchmark_record(profile.f0_hz)))


def reconcile(
    profile: ProbeProfile, decimation: int, *, revision: str | None = None
) -> ProfileReconciliation:
    """The profile-specific result at one decimation ratio."""
    case = profile_case(profile, decimation)
    lagrange = CANDIDATES["lagrange4"]
    phase, mag = worst_band_edge_error(lagrange, case.band_edge)
    weighted = pulse_weighted_rms_error(lagrange, case.band_edge, rolloff_db=BANDWIDTH_LEVEL_DB)
    record = profile_record(profile)
    return ProfileReconciliation(
        profile=profile.name,
        status=profile.bandwidth_status,
        source=profile.bandwidth_source,
        f0_hz=profile.f0_hz,
        bandwidth_frac=profile.bandwidth_frac,
        bandwidth_edge_hz=profile.bandwidth_edge_hz,
        level_db=BANDWIDTH_LEVEL_DB,
        convention=WIDTH_CONVENTION,
        decimation=decimation,
        producing_revision=revision if revision is not None else producing_revision(),
        iq_kernel="lagrange4",
        iq_worst_phase_deg=phase,
        iq_worst_magnitude_error=mag,
        iq_weighted_rms_pct=100.0 * weighted,
        rf_golden_residual_pct=rf.residual_pct(rf.production, record),
        rf_record_is_frozen_oracle=_is_frozen_record(profile, record),
    )


def report(profile: ProbeProfile, decimations: tuple[int, ...] = (8, 4)) -> str:
    return "\n".join(line for d in decimations for line in reconcile(profile, d).lines())


if __name__ == "__main__":
    print(report(linear_5mhz()))
