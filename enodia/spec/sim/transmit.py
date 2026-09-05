"""The transmit beam model (#9, design.md §18).

The simulator needs, for one transmit event and one scatterer, the transmit
pulse arriving at that scatterer. Both models here express it the same way —
**as a set of (arrival time, amplitude) pairs**, each naming one delayed copy
of the two-way pulse — so the simulator's summation is identical under either
and neither model introduces an interpolation the other does not. The
virtual-source model returns one pair; aperture superposition returns one per
transmitting element.

**The virtual-source approximation is discontinuous at the focus.** Writing
the transmit time of flight as `(z_v + sign(z_s − z_v)·|r_s − r_v|)/c`
gives, on either side of the focal depth, `(z_v ∓ |x_s − x_v|)/c` — a jump of
`2|x_s − x_v|/c` for every scatterer off the beam axis. design.md §18 names
the consequence (hourglass artifacts around the focus, a classic
retrospective-transmit-focusing pitfall) and says to build the blend from the
start. `blended_sign` is that blend.

**Where the approximation actually fails** sets the blend's width. A virtual
source is a spherical wave from a point; the real converging wavefront stops
resembling one when its curvature across the aperture falls below a
wavelength, which happens over an axial extent of order `λ·F#²` around the
focus — the depth-of-field scale. The blend half-width is therefore
`BLEND_HALF_WIDTH_FACTOR · λ · F#²`, with the factor a swept quantity rather
than a constant chosen by eye (`enodia.spec.sim.blend_sweep`).

**Aperture superposition is the yardstick, not the default.** design.md §18
adopts the virtual-source approximation and keeps superposition switchable;
this module follows that. Superposition exists because an approximation with
a free parameter needs a reference that does not share its assumption — the
same reason the RF-domain ideal-delay DAS is kept as the golden path for the
IQ approximation (absolute rules). It is also the only consumer of the
per-element firing delays and apodization that `enodia.spec.sequence` carries
and validates (#52).

The two models describe the same beam by construction, not by coincidence:
`accept` refuses a configuration whose per-element delays disagree with its
declared virtual source by more than `DELAY_TOLERANCE_NS`.
"""

from __future__ import annotations

import numpy as np

from enodia.spec.probe import ProbeProfile
from enodia.spec.sequence import TxEvent

__all__ = [
    "BLEND_HALF_WIDTH_FACTOR",
    "TRANSMIT_MODELS",
    "aperture_superposition",
    "blend_half_width_m",
    "blended_sign",
    "transmit_contributions",
    "virtual_source",
    "virtual_source_unblended",
]

# Blend half-width in units of λ·F#², the axial scale over which a converging
# wavefront stops resembling a spherical wave from a point. **Selected by
# `enodia.spec.sim.blend_sweep`**, which minimises the worst arrival-time
# shape error against the aperture-superposition model over the focal region:
# 1.33 periods here against 4.02 unblended, with the minimum interior to the
# swept range rather than at an end. Not chosen by eye; see ADR-0011.
BLEND_HALF_WIDTH_FACTOR: float = 2.0


def blend_half_width_m(profile: ProbeProfile, *, factor: float = BLEND_HALF_WIDTH_FACTOR) -> float:
    """Half-width of the transition zone around the focal depth [m]."""
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"blend half-width factor must be finite and positive, got {factor!r}")
    return float(factor) * profile.wavelength_m * profile.f_number**2


def blended_sign(u: np.ndarray | float, half_width: float) -> np.ndarray:
    """`sign(u)`, made continuous and once differentiable across zero.

    Equal to `sign(u)` for `|u| ≥ half_width`, so the blend is **local**: the
    delay field outside the transition zone is the unblended one, bit for
    bit, and this is not a global reshaping of the beam.

    Inside, `sin(πu / 2h)` carries the sign smoothly through zero. It meets
    ±1 at ±h with zero slope, which is `sign`'s own slope there, so the
    result is C¹ across both seams as well as across the focus. A linear ramp
    would be continuous but would leave a corner at ±h, and a global `tanh`
    would perturb the delay at every depth to remove a defect that exists at
    one.
    """
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError(f"blend half-width must be finite and positive, got {half_width!r}")
    u = np.asarray(u, dtype=np.float64)
    inside = np.abs(u) < half_width
    return np.where(inside, np.sin(np.pi * u / (2.0 * half_width)), np.sign(u))


def virtual_source(
    profile: ProbeProfile,
    event: TxEvent,
    x_m: float,
    z_m: float,
    *,
    blend_half_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One (arrival time, amplitude) pair: the virtual-source approximation.

    `blend_half_width=None` takes the profile's swept default. Passing `0.0`
    is refused rather than silently meaning "unblended": the unblended model
    is a named function (`virtual_source_unblended`) so that using it is a
    choice a reader can see, not an argument value.
    """
    vx, vz = event.virtual_source_m
    half_width = blend_half_width_m(profile) if blend_half_width is None else blend_half_width
    r_sv = float(np.hypot(x_m - vx, z_m - vz))
    t_tx = (vz + float(blended_sign(z_m - vz, half_width)) * r_sv) / profile.c_m_s
    return np.array([t_tx]), np.array([_beam_amplitude(profile, event, x_m, z_m)])


def virtual_source_unblended(
    profile: ProbeProfile, event: TxEvent, x_m: float, z_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-#9 model, kept so the defect it has can be exhibited.

    Not a transmit model a caller should select for imaging: it is the
    discontinuity `blended_sign` exists to remove, and it is here because a
    fix whose defect cannot be shown is not evidence of anything.
    """
    vx, vz = event.virtual_source_m
    r_sv = float(np.hypot(x_m - vx, z_m - vz))
    t_tx = (vz + float(np.sign(z_m - vz)) * r_sv) / profile.c_m_s
    return np.array([t_tx]), np.array([_beam_amplitude(profile, event, x_m, z_m)])


def _beam_amplitude(profile: ProbeProfile, event: TxEvent, x_m: float, z_m: float) -> float:
    """Gaussian across the beam, widening away from the focus at the F-number's rate.

    Floored at `λ·F#`: the amplitude side of the focus is already regular, and
    this floor is what keeps it so. The defect #9 addresses is in the delay.
    """
    _, vz = event.virtual_source_m
    beam_w = max(profile.wavelength_m * profile.f_number, abs(z_m - vz) / (2.0 * profile.f_number))
    return float(np.exp(-0.5 * ((x_m - event.line_x_m) / beam_w) ** 2))


def aperture_superposition(
    profile: ProbeProfile, event: TxEvent, x_m: float, z_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """One pair per transmitting element: the higher-fidelity alternative.

    The scatterer receives the apodization-weighted sum of the pulse from
    every element, each delayed by its own firing delay plus its own travel
    time. There is no virtual source in this, and so no `sign()` and nothing
    to blend — which is what makes it the yardstick the blend's width is
    swept against.

    Amplitudes are normalized by the apodization sum, so that a scatterer at
    the focus, where the element contributions arrive together, sees unit
    amplitude — the same normalization the Gaussian beam profile carries.
    The lateral beam shape is then not assumed but emerges from the elements
    falling out of phase off axis, which is the whole point of the model.

    Silent elements (`apodization == 0`) are still returned rather than
    dropped: their pulse copies are multiplied by zero, and a variable-length
    return would make the work depend on the aperture, which is the shape the
    absolute rules forbid on the real-time path and which this reference
    implementation must not sanction by example.
    """
    el_x = profile.element_x()
    apod = np.asarray(event.apodization, dtype=np.float64)
    delays = np.asarray(event.firing_delays_s, dtype=np.float64)
    total = float(apod.sum())
    if total <= 0.0:
        raise ValueError(f"transmit event {event.event_index} has no firing elements")
    taus = delays + np.hypot(x_m - el_x, z_m) / profile.c_m_s
    return taus, apod / total


TRANSMIT_MODELS = {
    "virtual-source": virtual_source,
    "virtual-source-unblended": virtual_source_unblended,
    "aperture-superposition": aperture_superposition,
}


def transmit_contributions(
    profile: ProbeProfile,
    event: TxEvent,
    x_m: float,
    z_m: float,
    *,
    model: str = "virtual-source",
    blend_half_width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Arrival times [s] and amplitudes of the transmit pulse at one scatterer.

    The seam design.md §18 calls switchable. `model` names one of
    `TRANSMIT_MODELS`; the default is the virtual-source approximation §18
    adopts, blended.
    """
    if model not in TRANSMIT_MODELS:
        raise ValueError(
            f"unknown transmit model {model!r}; expected one of {sorted(TRANSMIT_MODELS)}"
        )
    if model == "virtual-source":
        return virtual_source(profile, event, x_m, z_m, blend_half_width=blend_half_width)
    if blend_half_width is not None:
        raise ValueError(f"transmit model {model!r} takes no blend half-width")
    return TRANSMIT_MODELS[model](profile, event, x_m, z_m)
