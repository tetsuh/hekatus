"""The external transmit description and what enodia accepts from it (#52).

design.md §19 defines the boundary of the whole system: enodia does not
decide sequences, it receives a description of them in the vocabulary of
physical fact — element coordinates, virtual-source positions [mm], firing
delays [ns], apodization values, transmit-type tags — and configures
receive-side computation from it. These tests pin the two halves that prose
cannot: what the vocabulary is, and what happens at ingress when the
description and the named probe profile disagree.

The ingress rule is the part that needs a test rather than a comment. Bit
equality is not available — converting the current profile's element
coordinates to millimetres and back moves some of them — so a tolerance
exists, and a tolerance that is never exercised is a tolerance nobody knows
the sign of.
"""

from dataclasses import fields

import numpy as np
import pytest

from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.records import EventHeader
from enodia.spec.sequence import (
    COORDINATE_TOLERANCE_ULP,
    DELAY_TOLERANCE_NS,
    TransmitConfig,
    TransmitDescription,
    TxEvent,
    TxEventDescription,
    accept,
    coordinate_tolerance_m,
    describe_bmode,
    make_bmode_config,
)

# Field names that carry no unit because they are not physical quantities:
# identifiers, indices, tags, and the dimensionless apodization weights.
DIMENSIONLESS_FIELDS = {
    "config_id",
    "probe_profile_id",
    "events",
    "event_index",
    "line_index",
    "tx_type",
    "apodization",
}

# Words that would mean the description had drifted into the FPGA's
# vocabulary — the thing §19 rejects by name.
MACHINE_WORDS = ("clock", "cycle", "register", "tick", "count", "sample_index", "addr")


def _description(profile: ProbeProfile) -> TransmitDescription:
    return describe_bmode(profile)


def _replace_event(
    description: TransmitDescription, index: int, **changes
) -> TransmitDescription:
    from dataclasses import replace

    events = list(description.events)
    events[index] = replace(events[index], **changes)
    return replace(description, events=tuple(events))


# --- the vocabulary -------------------------------------------------------


def test_every_described_quantity_names_its_unit_or_is_dimensionless():
    """§19: physical fact only. A field whose name carries no unit and is not
    on the dimensionless list is a field whose units live in a comment."""
    for cls in (TransmitDescription, TxEventDescription):
        for f in fields(cls):
            if f.name in DIMENSIONLESS_FIELDS:
                continue
            assert f.name.endswith(("_mm", "_ns")), f"{cls.__name__}.{f.name}"


def test_the_description_carries_no_machine_facing_vocabulary():
    """"No FPGA-internal representations (clock counts, register values)" is
    §19's wording, and this is what it means field by field."""
    for cls in (TransmitDescription, TxEventDescription):
        for f in fields(cls):
            for word in MACHINE_WORDS:
                assert word not in f.name.lower(), f"{cls.__name__}.{f.name}"


def test_the_description_carries_the_whole_section_19_vocabulary():
    assert {f.name for f in fields(TransmitDescription)} == {
        "config_id",
        "probe_profile_id",
        "element_x_mm",
        "events",
    }
    assert {f.name for f in fields(TxEventDescription)} == {
        "event_index",
        "tx_type",
        "line_x_mm",
        "virtual_source_mm",
        "firing_delays_ns",
        "apodization",
    }


def test_no_bandwidth_travels_in_the_transmit_description():
    """ADR-0008, #46: the bandwidth is the profile's and is duplicated
    nowhere. The new external layer is the obvious place for it to leak."""
    for cls in (TransmitDescription, TxEventDescription, TransmitConfig, TxEvent, EventHeader):
        assert not any("bandwidth" in f.name for f in fields(cls)), cls.__name__


def test_the_description_does_not_say_which_line_an_event_forms():
    """§19: contribution maps are derivatives, which enodia computes for
    itself and never receives. The generalization is #53; what this pins is
    that the external description has no say in it either way."""
    described = {f.name for f in fields(TxEventDescription)}
    assert "line_index" not in described


def test_the_accepted_configuration_carries_the_derived_line_index():
    config = make_bmode_config(linear_5mhz())

    assert [ev.line_index for ev in config.events] == list(range(len(config.events)))


# --- the transmit-type tag is an open set ---------------------------------


def test_an_unseen_transmit_type_tag_is_accepted_and_carried():
    """The absolute rule: transmit-type tags are an open set, because
    shear-wave push and tracking transmits join it later (§11.5). An enum
    here would have to be edited then; a string does not."""
    profile = linear_5mhz()
    description = _replace_event(_description(profile), 0, tx_type="shear_wave_push")

    config = accept(description, profile)

    assert config.events[0].tx_type == "shear_wave_push"


def test_an_empty_transmit_type_tag_is_refused():
    """Open set, not absent set: an unnamed transmit type would reach the
    frame header and label data nothing at all."""
    profile = linear_5mhz()
    description = _replace_event(_description(profile), 0, tx_type="")

    with pytest.raises(ValueError, match="tx_type"):
        accept(description, profile)


# --- the ingress rule -----------------------------------------------------


def test_the_millimetre_round_trip_is_not_bit_equal_on_the_current_profile():
    """The measured fact the tolerance exists for. If this ever passes with
    zero differing coordinates, the tolerance is still right but its
    justification has changed, and the ADR needs rereading."""
    canonical = linear_5mhz().element_x()

    round_tripped = (canonical * 1e3) * 1e-3

    differing = int(np.count_nonzero(round_tripped != canonical))
    assert differing > 0
    assert np.abs(round_tripped - canonical).max() < coordinate_tolerance_m(linear_5mhz())


def test_ingress_accepts_the_millimetre_round_trip_of_its_own_profile():
    profile = linear_5mhz()

    config = accept(_description(profile), profile)

    assert isinstance(config, TransmitConfig)
    assert config.probe_profile_id == profile.name


def test_what_enodia_computes_from_is_the_canonical_profile_geometry():
    """The transported numbers are checked and then dropped. Everything
    downstream reads one geometry, so a port comparing against the reference
    implementation compares like with like (L0, ADR-0007's principle)."""
    profile = linear_5mhz()

    config = accept(_description(profile), profile)

    assert np.array_equal(np.asarray(config.element_x_m), profile.element_x())


def test_a_geometry_mismatch_beyond_the_tolerance_is_refused():
    """One element displaced by a pitch: an image would still form, and
    would be subtly wrong everywhere. Processing with a wrong description is
    worse than refusing it (absolute rules)."""
    profile = linear_5mhz()
    description = _description(profile)
    moved = list(description.element_x_mm)
    moved[3] += profile.pitch_m * 1e3

    from dataclasses import replace

    with pytest.raises(ValueError, match="element coordinate"):
        accept(replace(description, element_x_mm=tuple(moved)), profile)


def test_a_displacement_just_past_the_tolerance_is_refused():
    """The tolerance is a boundary, and a boundary nobody tests is a
    boundary that drifts."""
    profile = linear_5mhz()
    description = _description(profile)
    moved = list(description.element_x_mm)
    moved[0] += coordinate_tolerance_m(profile) * 1e3 * 4.0

    from dataclasses import replace

    with pytest.raises(ValueError, match="element coordinate"):
        accept(replace(description, element_x_mm=tuple(moved)), profile)


def test_a_configuration_naming_another_profile_is_refused():
    profile = linear_5mhz()
    description = _description(profile)

    from dataclasses import replace

    with pytest.raises(ValueError, match="probe profile"):
        accept(replace(description, probe_profile_id="linear-13mhz"), profile)


def test_a_wrong_element_count_is_refused():
    profile = linear_5mhz()
    description = _description(profile)

    from dataclasses import replace

    with pytest.raises(ValueError, match="element"):
        accept(replace(description, element_x_mm=description.element_x_mm[:-1]), profile)


def test_a_non_finite_coordinate_is_refused():
    """NaN compares false against every tolerance, so a bare `<= tol` check
    would let it through and put NaN in a delay table."""
    profile = linear_5mhz()
    description = _description(profile)
    poisoned = list(description.element_x_mm)
    poisoned[10] = float("nan")

    from dataclasses import replace

    with pytest.raises(ValueError, match="finite"):
        accept(replace(description, element_x_mm=tuple(poisoned)), profile)


# --- delays against the declared virtual source ---------------------------


def test_firing_delays_agree_with_the_declared_virtual_source():
    """§19's first line of defence: the physical schema is the specification,
    so the description has to be internally consistent before anything is
    derived from it."""
    profile = linear_5mhz()

    config = accept(_description(profile), profile)

    ev = config.events[0]
    assert len(ev.firing_delays_s) == profile.n_elements
    assert all(np.isfinite(ev.firing_delays_s))


def test_delays_inconsistent_with_the_virtual_source_are_refused():
    """A converter bug that shifts one element's delay produces an image that
    looks fine and is wrong. §19 puts three lines of defence against exactly
    this; the schema being self-checking is the first."""
    profile = linear_5mhz()
    description = _description(profile)
    broken = list(description.events[0].firing_delays_ns)
    broken[64] += 50.0 * DELAY_TOLERANCE_NS

    with pytest.raises(ValueError, match="firing delay"):
        accept(_replace_event(description, 0, firing_delays_ns=tuple(broken)), profile)


def test_a_delay_error_inside_the_tolerance_is_accepted():
    profile = linear_5mhz()
    description = _description(profile)
    nudged = list(description.events[0].firing_delays_ns)
    nudged[64] += 0.5 * DELAY_TOLERANCE_NS

    config = accept(_replace_event(description, 0, firing_delays_ns=tuple(nudged)), profile)

    assert config.events[0].tx_type == description.events[0].tx_type


def test_only_the_active_aperture_is_checked_for_delay_consistency():
    """Apodization defines which elements fire. A zero-weighted element's
    delay is not a physical claim about the wavefront, and refusing a
    configuration for it would refuse every sparse aperture."""
    profile = linear_5mhz()
    description = _description(profile)
    ev = description.events[0]
    silent = [i for i, w in enumerate(ev.apodization) if w == 0.0]
    assert silent, "the focused aperture should leave some elements silent"
    delays = list(ev.firing_delays_ns)
    delays[silent[0]] += 1e6

    config = accept(_replace_event(description, 0, firing_delays_ns=tuple(delays)), profile)

    assert config.events[0].apodization[silent[0]] == 0.0


def test_a_negative_apodization_weight_is_refused():
    profile = linear_5mhz()
    description = _description(profile)
    weights = list(description.events[0].apodization)
    weights[20] = -0.5

    with pytest.raises(ValueError, match="apodization"):
        accept(_replace_event(description, 0, apodization=tuple(weights)), profile)


def test_a_silent_aperture_is_refused():
    """Every weight zero describes a transmit that never happened, and the
    delay check would pass vacuously."""
    profile = linear_5mhz()
    description = _description(profile)
    weights = (0.0,) * profile.n_elements

    with pytest.raises(ValueError, match="aperture"):
        accept(_replace_event(description, 0, apodization=weights), profile)


# --- the schema changes no image ------------------------------------------


def test_the_accepted_events_match_the_geometry_the_pipeline_used_before():
    """#52 re-describes MVP-1; it does not move a scanline. The pre-#52 rule
    was: one scanline above each element, virtual source at the transmit
    focus on that abscissa."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    canonical = profile.element_x()

    tolerance = coordinate_tolerance_m(profile)
    for k, ev in enumerate(config.events):
        assert abs(ev.line_x_m - canonical[k]) <= tolerance
        assert abs(ev.virtual_source_m[0] - canonical[k]) <= tolerance
        assert ev.virtual_source_m[1] == profile.tx_focus_m


def test_the_simulated_frame_is_unchanged_by_the_schema():
    """The acceptance criterion, at frame scale: the raw RF the simulator
    produces through the schema is the RF it produced from events built
    directly in SI."""
    from enodia.spec.sim import PointScatterer, simulate_bmode_frame

    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events[60:64])
    scatterers = [PointScatterer(0.0, 20e-3)]

    canonical = profile.element_x()
    direct = [
        TxEvent(
            event_index=ev.event_index,
            line_index=ev.line_index,
            tx_type=ev.tx_type,
            line_x_m=float(canonical[ev.event_index]),
            virtual_source_m=(float(canonical[ev.event_index]), profile.tx_focus_m),
            firing_delays_s=ev.firing_delays_s,
            apodization=ev.apodization,
        )
        for ev in events
    ]

    through_schema = simulate_bmode_frame(profile, events, scatterers)
    built_directly = simulate_bmode_frame(profile, direct, scatterers)

    for a, b in zip(through_schema, built_directly, strict=True):
        assert np.array_equal(a.data, b.data)


def test_the_tolerance_is_far_below_anything_physical_and_far_above_the_noise():
    """The number has to sit in a gap, and the gap is thirteen orders wide:
    the round-trip residue is ~1e-18 m, one element pitch is 3e-4 m."""
    profile = linear_5mhz()

    tolerance = coordinate_tolerance_m(profile)

    assert tolerance < profile.pitch_m / 1e10
    assert tolerance > np.abs((profile.element_x() * 1e3) * 1e-3 - profile.element_x()).max()
    assert COORDINATE_TOLERANCE_ULP >= 1
