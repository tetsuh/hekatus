"""The transmit description enodia receives, and what it accepts from it.

design.md §19 makes enodia a **subordinate system**: control software owns
the transmit master data, and enodia receives a description of it through an
API and configures receive-side computation from that. This module is that
boundary.

**Two layers, and the difference matters.** `TransmitDescription` is the
external form — the vocabulary of physical fact §19 fixes, in the units §19
names: element coordinates and virtual-source positions in millimetres,
firing delays in nanoseconds, dimensionless apodization weights, and a
transmit-type tag that is an open set of strings. Nothing in it is an FPGA
representation; §19 rejects interpreting those by name, because it would add
a reverse converter, break on FPGA revisions, and give L0 a second
specification to disagree with. `TransmitConfig` is what `accept()` returns:
the same description in SI units, checked, with the derived quantities
enodia computes for itself attached.

**What ingress does with geometry.** The description transports element
coordinates; the probe profile already holds them (§4). They are compared,
and then the transported numbers are *dropped* — everything downstream reads
`ProbeProfile.element_x()`. Two reasons. A port is accepted by numerical
equivalence with this reference implementation (L0, ADR-0007's principle),
which requires both sides to compute from one geometry rather than from
whatever each was handed. And the comparison cannot be equality: converting
this profile's coordinates to millimetres and back moves 6 of 128 of them,
by at most 8.7e-19 m, so a tolerance is unavoidable. It is stated in units
in the last place of the aperture scale, which puts it thirteen orders below
one element pitch — tight enough that a real geometry mismatch cannot hide
under it, loose enough that binary64 unit conversion never trips it.

**What ingress does with delays.** §19 sets out three lines of defence
against a transmit description that disagrees with the machine, and the
first is that the physical schema is the specification. So the description
has to be internally consistent before anything is derived from it: for
every element that actually fires, the firing delay plus the geometric time
of flight to the declared virtual source must be the same instant, within
`DELAY_TOLERANCE_NS` — §19's "ns-class delay tolerance". A configuration
that fails is refused rather than beamformed, because processing with a
wrong description is worse than dropping frames (absolute rules).

**What this module deliberately does not do.** The per-element firing delays
and apodization weights are carried and validated here; no transmit field is
synthesized from them. The transmit beam model — the virtual-source focal
blend and the switch to aperture superposition — is #9, and §18 records it
as a detail settled at implementation time. And the contribution map stays
the identity: event k forms line k. §19 calls maps derivatives, which enodia
derives rather than receives, and the external description accordingly says
nothing about which line an event forms. Generalizing that map to MLA and
transmit compounding is #53.

What is also already true here (#46, ADR-0008): a transmit configuration
names the **one probe profile it runs on**, by id, and carries no bandwidth
of its own. The runtime config ID a record carries selects the configuration
and therefore, transitively, the profile; the profile is part of the table
set that config ID names (docs/dataplane.md). Nothing about the pulse is
duplicated into the transmit description or the frame header.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enodia.spec.probe import ProbeProfile

# Ingress tolerance for transported element coordinates, in units in the last
# place of the aperture half-width. Four rather than one: a description may
# have passed through more than one binary64 unit conversion on its way here,
# and each can move the value by half an ulp. Sweepable, like every design
# parameter — `coordinate_tolerance_m()` is what callers use.
COORDINATE_TOLERANCE_ULP: int = 4

# Ingress tolerance for the firing-delay consistency check [ns]. §19 asks for
# a "ns-class delay tolerance" on the round trip that guards the same
# quantity from the other side. One nanosecond is 1.5 µm of path at the
# reference sound speed — two hundred times finer than a wavelength at
# 5 MHz, and four hundred times coarser than the residue of converting a
# delay to nanoseconds and back.
DELAY_TOLERANCE_NS: float = 1.0


def coordinate_tolerance_m(profile: ProbeProfile) -> float:
    """How far a transported element coordinate may sit from the profile's [m].

    Stated at the aperture scale rather than per coordinate: the spacing of
    binary64 changes with magnitude, and an element near the array centre
    would otherwise get a tolerance hundreds of times tighter than one at the
    edge for no physical reason.
    """
    aperture_half_width = float(np.abs(profile.element_x()).max())
    return COORDINATE_TOLERANCE_ULP * float(np.spacing(aperture_half_width))


@dataclass(frozen=True)
class TxEventDescription:
    """One transmit event as control software describes it (design.md §19).

    Millimetres and nanoseconds, because those are the units §19 names. The
    arrays run over every element of the profile: `apodization` is what says
    which of them fire, so a sparse or walking aperture needs no separate
    element list.
    """

    event_index: int
    tx_type: str  # open set of strings, never an enum (absolute rules)
    line_x_mm: float  # lateral position of the scanline (the beam axis)
    virtual_source_mm: tuple[float, float]  # virtual source (x, z)
    firing_delays_ns: tuple[float, ...]  # per element, relative to the event
    apodization: tuple[float, ...]  # per element, dimensionless, zero = silent


@dataclass(frozen=True)
class TransmitDescription:
    """One entry of the setup-time configuration set, as received (§19).

    `config_id` is what the runtime announces and what every record carries;
    `probe_profile_id` is the one `ProbeProfile.name` this configuration runs
    on. No bandwidth field: that is the profile's, and duplicating it is what
    ADR-0008 forbids.
    """

    config_id: str
    probe_profile_id: str
    element_x_mm: tuple[float, ...]
    events: tuple[TxEventDescription, ...]


@dataclass(frozen=True)
class TxEvent:
    """One accepted transmit event, in SI units.

    `line_index` is the first derivative: which output line this event forms.
    It is not received — §19 keeps maps on enodia's side of the boundary —
    and while the contribution map is the identity it is simply the event
    index. #53 generalizes it to MLA and transmit compounding, at which point
    one event reaches several lines with weights and this field is replaced
    by the map.
    """

    event_index: int
    line_index: int
    tx_type: str
    line_x_m: float
    virtual_source_m: tuple[float, float]
    firing_delays_s: tuple[float, ...]
    apodization: tuple[float, ...]


@dataclass(frozen=True)
class TransmitConfig:
    """An accepted configuration: SI units, checked, canonical geometry.

    `element_x_m` is the **profile's** geometry, not the transported numbers
    — see the module docstring. It is carried here so that a consumer holding
    only a configuration still reads the same coordinates the delay tables
    were derived from.
    """

    config_id: str
    probe_profile_id: str
    element_x_m: tuple[float, ...]
    events: tuple[TxEvent, ...]


def _finite(values, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        # Written as "not all finite" rather than "any non-finite <= tol":
        # every comparison against NaN is false, so a bare tolerance check
        # accepts NaN and puts it in a delay table.
        raise ValueError(f"{what} must be finite")
    return array


def _check_geometry(description: TransmitDescription, profile: ProbeProfile) -> np.ndarray:
    if description.probe_profile_id != profile.name:
        raise ValueError(
            f"configuration names probe profile {description.probe_profile_id!r},"
            f" accepted against {profile.name!r}"
        )
    canonical = profile.element_x()
    if len(description.element_x_mm) != canonical.size:
        raise ValueError(
            f"description carries {len(description.element_x_mm)} element coordinates,"
            f" profile {profile.name!r} has {canonical.size}"
        )
    transported = _finite(description.element_x_mm, "element coordinates") * 1e-3
    tolerance = coordinate_tolerance_m(profile)
    worst = int(np.argmax(np.abs(transported - canonical)))
    error = abs(float(transported[worst] - canonical[worst]))
    if error > tolerance:
        raise ValueError(
            f"element coordinate {worst} is {error:.3e} m from the profile's,"
            f" past the {tolerance:.3e} m ingress tolerance"
        )
    return canonical


def _check_event(
    ev: TxEventDescription, canonical: np.ndarray, profile: ProbeProfile
) -> TxEvent:
    if not ev.tx_type.strip():
        raise ValueError(f"transmit event {ev.event_index} has an empty tx_type")

    weights = _finite(ev.apodization, f"apodization of event {ev.event_index}")
    if weights.size != canonical.size:
        raise ValueError(
            f"event {ev.event_index} carries {weights.size} apodization weights,"
            f" profile {profile.name!r} has {canonical.size} elements"
        )
    if np.any(weights < 0.0):
        raise ValueError(f"apodization of event {ev.event_index} has a negative weight")
    active = weights > 0.0
    if not np.any(active):
        raise ValueError(f"event {ev.event_index} describes a silent aperture")

    delays_s = _finite(ev.firing_delays_ns, f"firing delays of event {ev.event_index}") * 1e-9
    if delays_s.size != canonical.size:
        raise ValueError(
            f"event {ev.event_index} carries {delays_s.size} firing delays,"
            f" profile {profile.name!r} has {canonical.size} elements"
        )

    source = _finite(ev.virtual_source_mm, f"virtual source of event {ev.event_index}") * 1e-3
    if source.size != 2:
        raise ValueError(f"event {ev.event_index} virtual source must be (x, z)")
    vx, vz = float(source[0]), float(source[1])

    # The wavefront each firing element launches has to reach the virtual
    # source at one instant; that is what "focused at the virtual source"
    # means as a physical claim, and it is checkable without a beam model.
    time_of_flight = np.hypot(canonical - vx, vz) / profile.c_m_s
    arrival = (delays_s + time_of_flight)[active]
    spread_ns = float(arrival.max() - arrival.min()) * 1e9
    if spread_ns > DELAY_TOLERANCE_NS:
        raise ValueError(
            f"firing delays of event {ev.event_index} disagree with its virtual source"
            f" by {spread_ns:.3f} ns, past the {DELAY_TOLERANCE_NS:g} ns ingress tolerance"
        )

    line_x_m = float(_finite((ev.line_x_mm,), f"scanline of event {ev.event_index}")[0]) * 1e-3
    return TxEvent(
        event_index=ev.event_index,
        line_index=ev.event_index,  # identity contribution map; #53 generalizes it
        tx_type=ev.tx_type,
        line_x_m=line_x_m,
        virtual_source_m=(vx, vz),
        firing_delays_s=tuple(float(d) for d in delays_s),
        apodization=tuple(float(w) for w in weights),
    )


def accept(description: TransmitDescription, profile: ProbeProfile) -> TransmitConfig:
    """Check a received transmit description and convert it to SI (§19).

    Refuses rather than repairs. A description that disagrees with the probe
    profile it names, or with itself, describes a machine other than the one
    the data is coming from, and an image formed from it looks plausible and
    is wrong — the failure mode §19's three lines of defence exist for.
    """
    canonical = _check_geometry(description, profile)
    if not description.events:
        raise ValueError(f"configuration {description.config_id!r} describes no transmit events")
    seen: set[int] = set()
    events = []
    for ev in description.events:
        if ev.event_index in seen:
            raise ValueError(f"duplicate transmit event index {ev.event_index}")
        seen.add(ev.event_index)
        events.append(_check_event(ev, canonical, profile))
    return TransmitConfig(
        config_id=description.config_id,
        probe_profile_id=description.probe_profile_id,
        element_x_m=tuple(float(x) for x in canonical),
        events=tuple(events),
    )


def focused_aperture(profile: ProbeProfile, vx: float, vz: float) -> np.ndarray:
    """Hann apodization over the F-number aperture about `vx` (dimensionless).

    The aperture is `vz / f_number` wide, which is what the F-number means,
    and elements outside it are silent. Hann rather than uniform because that
    is what the receive aperture uses (`enodia.spec.beamform`), and a
    transmit apodization that disagreed with it would be a second convention
    to keep track of for no reason.
    """
    half_width = vz / (2.0 * profile.f_number)
    offset = profile.element_x() - vx
    inside = np.abs(offset) <= half_width
    weights = np.zeros_like(offset)
    weights[inside] = 0.5 * (1.0 + np.cos(np.pi * offset[inside] / half_width))
    if not np.any(weights > 0.0):
        # A focus so shallow that no element is inside the aperture. Silent
        # here means silent everywhere downstream, so say it now.
        raise ValueError(f"no element lies inside the F={profile.f_number:g} aperture at z={vz} m")
    return weights


def focusing_delays_ns(
    profile: ProbeProfile, vx: float, vz: float, apodization: np.ndarray
) -> np.ndarray:
    """Firing delays [ns] that focus the firing elements at (vx, vz).

    Referenced so the earliest firing element sits at zero: the description
    carries delays within an event, and a common offset is the pulse
    repetition instant, which the frame header already names.
    """
    time_of_flight = np.hypot(profile.element_x() - vx, vz) / profile.c_m_s
    active = apodization > 0.0
    delays = np.zeros_like(time_of_flight)
    delays[active] = time_of_flight[active].max() - time_of_flight[active]
    return delays * 1e9


def describe_bmode(
    profile: ProbeProfile, *, config_id: str = "bmode-focused"
) -> TransmitDescription:
    """The conventional B-mode configuration, as control software would send it.

    One scanline above each element, focused at the profile's transmit focus.
    This is the reference implementation standing in for the host: it is how
    the schema gets exercised rather than merely defined (§18 item 2, §19).
    """
    element_x = profile.element_x()
    events = []
    for k, x in enumerate(element_x):
        vx, vz = float(x), profile.tx_focus_m
        weights = focused_aperture(profile, vx, vz)
        delays = focusing_delays_ns(profile, vx, vz, weights)
        events.append(
            TxEventDescription(
                event_index=k,
                tx_type="bmode_focused",
                line_x_mm=vx * 1e3,
                virtual_source_mm=(vx * 1e3, vz * 1e3),
                firing_delays_ns=tuple(float(d) for d in delays),
                apodization=tuple(float(w) for w in weights),
            )
        )
    return TransmitDescription(
        config_id=config_id,
        probe_profile_id=profile.name,
        element_x_mm=tuple(float(x) * 1e3 for x in element_x),
        events=tuple(events),
    )


def make_bmode_config(profile: ProbeProfile, *, config_id: str = "bmode-focused") -> TransmitConfig:
    """The conventional B-mode configuration on one profile, described and accepted."""
    return accept(describe_bmode(profile, config_id=config_id), profile)


def make_bmode_sequence(profile: ProbeProfile) -> list[TxEvent]:
    """Conventional focused-beam B-mode: one scanline above each element."""
    return list(make_bmode_config(profile).events)


__all__ = [
    "COORDINATE_TOLERANCE_ULP",
    "DELAY_TOLERANCE_NS",
    "TransmitConfig",
    "TransmitDescription",
    "TxEvent",
    "TxEventDescription",
    "accept",
    "coordinate_tolerance_m",
    "describe_bmode",
    "focused_aperture",
    "focusing_delays_ns",
    "make_bmode_config",
    "make_bmode_sequence",
]