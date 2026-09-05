"""The sweep behind the focal-blend width (design.md §18, #9, ADR-0011).

The blend across the virtual-source focal singularity has one free
parameter, its half-width, and the implementation style says a design
parameter is decided by measurement rather than chosen by eye. This module
is that measurement.

**The two models do not share a time origin, and are not asked to.** The
virtual-source arrival is measured from the array face along the axis; the
superposition arrival includes the firing delays, which
`enodia.spec.sequence.focusing_delays_ns` references so that the earliest
firing element sits at zero — "a common offset is the pulse repetition
instant, which the frame header already names". The difference between the
two origins is a per-event constant of several periods, and comparing raw
arrival times would measure that convention rather than the blend. Both
fields are therefore referred to their own value on the beam axis at the
focal depth, and what is compared is the **shape** of the arrival-time field
around the focus.

**What is measured.** The virtual-source model claims the transmit pulse
reaches a scatterer at one instant. Aperture superposition makes no such
claim: it says the pulse arrives as one copy per element, at times
`firing_delay_e + |r_s − r_e| / c`, weighted by the apodization. Two numbers
follow from that set, and both matter:

- its apodization-weighted **centroid**, which is the single arrival time a
  one-delay model should be reproducing, and
- its apodization-weighted **spread**, reported beside the error as a
  diagnostic of how far the field is from being a single arrival at all.
  It is **not a bound on the centroid error**: a one-delay model whose
  arrival equals the centroid has zero centroid error at any spread
  (`tests/test_transmit_model.py` exhibits that). What a large spread does
  say is that the field is one the metric cannot fully describe, whichever
  single arrival is chosen.

The blend width is chosen to minimise the worst centroid error over the
region where the blend is active, reported in periods of `f0` so the figure
travels between profiles.

**Why the error does not simply fall to zero as the width shrinks.** A
narrow blend leaves the arrival time close to the unblended one, which is
discontinuous, and the centroid is not. A wide blend flattens the arrival
time across a region where the true centroid still varies. The minimum
between those is the quantity this sweep locates.

Run as a script for the table; `tests/test_transmit_model.py` pins the
selected factor so the constant in `enodia.spec.sim.transmit` cannot drift
from what this computes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import TxEvent, make_bmode_config
from enodia.spec.sim.transmit import aperture_superposition, blend_half_width_m, virtual_source

# The factors the sweep reports. Spanning two decades either side of unity
# because the scale argument (lambda F#^2) fixes the units, not the constant.
FACTORS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)

# How far either side of the focus the comparison runs, in units of the
# depth-of-field scale lambda*F#^2. **Fixed, not scaled by the candidate**:
# a window that grew with the factor would give each candidate its own exam.
# Three of them covers the focal region and leaves the widest candidates
# extending past it, which is a cost the figure should show rather than hide.
AXIAL_REACH_DOF: float = 3.0

# Lateral offsets sampled, in units of the focal beam width lambda*F#. Zero
# is on the beam axis, where the unblended model has no discontinuity at all
# — the artifact is an off-axis one, so the sweep must look off axis to see
# it — and beyond about two beam widths the transmit amplitude is negligible
# and an arrival-time error there reaches no image.
LATERAL_OFFSETS_BEAMWIDTHS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class BlendCase:
    """One (profile, event) the sweep measures on, with its scales."""

    profile: ProbeProfile
    event: TxEvent

    @property
    def period_s(self) -> float:
        return 1.0 / self.profile.f0_hz


def centre_case(profile: ProbeProfile | None = None) -> BlendCase:
    """The centre scanline of the conventional B-mode configuration."""
    profile = linear_5mhz() if profile is None else profile
    config = make_bmode_config(profile)
    return BlendCase(profile=profile, event=config.events[len(config.events) // 2])


def superposition_centroid_and_spread(case: BlendCase, x_m: float, z_m: float) -> tuple[float, float]:
    """Apodization-weighted mean and standard deviation of the arrival times [s]."""
    taus, weights = aperture_superposition(case.profile, case.event, x_m, z_m)
    mean = float(np.sum(weights * taus))
    var = float(np.sum(weights * (taus - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


def _sample_points(case: BlendCase, n_depths: int) -> tuple[np.ndarray, tuple[float, ...]]:
    vx, vz = case.event.virtual_source_m
    dof = case.profile.wavelength_m * case.profile.f_number**2
    reach = AXIAL_REACH_DOF * dof
    z = np.linspace(vz - reach, vz + reach, n_depths)
    beam_w = case.profile.wavelength_m * case.profile.f_number
    offsets = tuple(vx + n * beam_w for n in LATERAL_OFFSETS_BEAMWIDTHS)
    return z[z > 0.0], offsets


def _referenced(model_at, case: BlendCase, x: float, z: float, ref: float) -> float:
    """A model's arrival at (x, z), referred to its own on-axis focal value."""
    return model_at(x, z) - ref


def _worst_shape_error_periods(case: BlendCase, arrival, *, n_depths: int) -> float:
    """Worst disagreement in arrival-time *shape*, in periods of f0.

    Both fields are referred to their own value on the beam axis at the focal
    depth, so the per-event origin difference cancels and what is left is
    the disagreement the blend can actually do something about.
    """
    vx, vz = case.event.virtual_source_m
    model_ref = arrival(vx, vz)
    centroid_ref, _ = superposition_centroid_and_spread(case, vx, vz)
    z_axis, offsets = _sample_points(case, n_depths)
    worst = 0.0
    for x in offsets:
        for z in z_axis:
            model = arrival(x, float(z)) - model_ref
            centroid, _ = superposition_centroid_and_spread(case, x, float(z))
            worst = max(worst, abs(model - (centroid - centroid_ref)) / case.period_s)
    return worst


def worst_centroid_error_periods(case: BlendCase, factor: float, *, n_depths: int = 241) -> float:
    """Worst arrival-shape error of the blended model, in periods."""
    half = blend_half_width_m(case.profile, factor=factor)

    def arrival(x: float, z: float) -> float:
        taus, _ = virtual_source(case.profile, case.event, x, z, blend_half_width=half)
        return float(taus[0])

    return _worst_shape_error_periods(case, arrival, n_depths=n_depths)


def worst_spread_periods(case: BlendCase, *, n_depths: int = 241) -> float:
    """Worst intrinsic arrival spread over the window, in periods.

    A diagnostic, not a bound: it says how far the superposition field is
    from a single arrival, and nothing about how well a given single arrival
    tracks the centroid.
    """
    z_axis, offsets = _sample_points(case, n_depths)
    worst = 0.0
    for x in offsets:
        for z in z_axis:
            _, spread = superposition_centroid_and_spread(case, x, float(z))
            worst = max(worst, spread / case.period_s)
    return worst


def unblended_worst_centroid_error_periods(case: BlendCase, *, n_depths: int = 241) -> float:
    """The same figure for the model with no blend — the number to beat."""
    from enodia.spec.sim.transmit import virtual_source_unblended

    def arrival(x: float, z: float) -> float:
        taus, _ = virtual_source_unblended(case.profile, case.event, x, z)
        return float(taus[0])

    return _worst_shape_error_periods(case, arrival, n_depths=n_depths)


def selected_factor(case: BlendCase | None = None, *, n_depths: int = 241) -> float:
    """The factor this sweep selects: the one minimising the worst shape error.

    The minimum is interior to `FACTORS`, which is what makes it a selection
    rather than a boundary: too narrow a blend leaves the arrival time close
    to the discontinuous one, too wide a blend flattens it across depths
    where the true field still varies.
    """
    case = centre_case() if case is None else case
    return min(FACTORS, key=lambda f: worst_centroid_error_periods(case, f, n_depths=n_depths))


def table(case: BlendCase | None = None) -> str:
    case = centre_case() if case is None else case
    lines = [
        (
            f"profile {case.profile.name}, f0 {case.profile.f0_hz / 1e6:.1f} MHz,"
            f" F# {case.profile.f_number:g},"
            f" focus {case.event.virtual_source_m[1] * 1e3:.1f} mm"
        ),
        f"lambda*F#^2 = {case.profile.wavelength_m * case.profile.f_number**2 * 1e3:.4f} mm",
        "",
        f"{'factor':>8s} {'half-width [mm]':>16s} {'worst centroid err [periods]':>30s}",
    ]
    for f in FACTORS:
        half_mm = blend_half_width_m(case.profile, factor=f) * 1e3
        lines.append(f"{f:8.2f} {half_mm:16.4f} {worst_centroid_error_periods(case, f):30.4f}")
    lines.append("")
    lines.append(
        f"{'unblended':>8s} {'—':>16s} {unblended_worst_centroid_error_periods(case):30.4f}"
    )
    lines.append(
        f"{'spread':>8s} {'—':>16s} {worst_spread_periods(case):30.4f}"
        + "   (intrinsic arrival spread; a diagnostic, not a bound)"
    )
    return "\n".join(lines)


# Where the away-from-focus comparison samples, as multiples of the focal
# depth. The near field is where the virtual-source picture is weakest — the
# wavefront is still converging — so it is sampled more finely than the far
# field, whose figures settle.
AWAY_FROM_FOCUS_DEPTH_MULTIPLES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.5, 2.0, 2.5, 3.0)


def away_from_focus_report(
    case: BlendCase | None = None,
) -> list[tuple[float, float, float, float]]:
    """Shape error and intrinsic spread away from the focus (#9 criterion 3).

    Rows of `(z / z_f, lateral offset in beam widths, error in periods,
    spread in periods)`, over the depths in `AWAY_FROM_FOCUS_DEPTH_MULTIPLES`
    that lie within the profile's imaging depth and the lateral offsets in
    `LATERAL_OFFSETS_BEAMWIDTHS`. Both fields are referred to their own
    on-axis focal value, as in the sweep.
    """
    case = centre_case() if case is None else case
    vx, vz = case.event.virtual_source_m
    beam_w = case.profile.wavelength_m * case.profile.f_number

    def arrival(x: float, z: float) -> float:
        taus, _ = virtual_source(case.profile, case.event, x, z)
        return float(taus[0])

    model_ref = arrival(vx, vz)
    centroid_ref, _ = superposition_centroid_and_spread(case, vx, vz)
    rows = []
    for mult in AWAY_FROM_FOCUS_DEPTH_MULTIPLES:
        z = vz * mult
        if z > case.profile.depth_m:
            continue
        for n in LATERAL_OFFSETS_BEAMWIDTHS:
            x = vx + n * beam_w
            centroid, spread = superposition_centroid_and_spread(case, x, z)
            err = ((arrival(x, z) - model_ref) - (centroid - centroid_ref)) / case.period_s
            rows.append((mult, n, err, spread / case.period_s))
    return rows


def away_from_focus_table(case: BlendCase | None = None) -> str:
    lines = [f"{'z / z_f':>8s} {'offset [bw]':>12s} {'shape err [periods]':>20s} {'spread':>8s}"]
    for mult, n, err, spread in away_from_focus_report(case):
        lines.append(f"{mult:8.2f} {n:12.1f} {err:+20.3f} {spread:8.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
    print()
    print(f"selected factor: {selected_factor():g}")
    print()
    print("Away from the focus, under the selected factor:")
    print(away_from_focus_table())
