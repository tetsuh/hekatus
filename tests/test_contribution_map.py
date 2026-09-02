"""The contribution map: which transmit events form which lines (#53, §7).

design.md §7 names the structure — a weighted sparse mapping from transmit
event to output line — and what it unifies: MLA (one transmit, several
receive lines) and transmit compounding (several transmits, one pixel) are
two uses of the same fact, the finite transmit-beam width. §19 fixes where
it comes from: maps are derivatives, computed by enodia from the transmit
description and never received.

What these tests hold to the absolute rules: **work per line is constant.**
The contributing-transmit count is a fixed cap and out-of-range weights are
zero, because the natural implementation — iterate only the transmits that
actually contribute — passes every image check while breaking worst-case
latency. And frame-edge lines are renormalized, because the artifact of not
doing so is visible but easy to accept by eye, so the eye is not the check.
"""

from dataclasses import FrozenInstanceError, replace
from functools import partial

import numpy as np
import pytest

from enodia.spec.beamform import das_rf_golden
from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.records import EventHeader, RFEventRecord
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sequence.contribution import (
    MLA_COUNTS,
    WEIGHT_SUM_FLOOR,
    ContributionMap,
    element_pitch_m,
    identity_map,
    mla_map,
    synthetic_uniform_map,
    transmit_line_pitch_m,
)


def small_profile() -> ProbeProfile:
    """A shrunken profile, so map-level integration tests stay fast.

    Every parameter is sweepable and geometry is data, so a small aperture
    is a configuration, not a special case.
    """
    return ProbeProfile(
        name="linear-5mhz-small",
        n_elements=16,
        pitch_m=0.3e-3,
        f0_hz=5.0e6,
        bandwidth_frac=0.7,
        bandwidth_source=None,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=10e-3,
        tx_focus_m=5e-3,
        f_number=2.0,
    )


def constant_records(
    profile: ProbeProfile,
    n_events: int,
    n_t: int = 512,
    *,
    config_id: str = "bmode-focused",
    tx_types: tuple[str, ...] | None = None,
):
    """One frame in which every event carries the same constant data.

    On such a frame, every output line must come out identical: any line
    that differs is showing the summation structure, not the field. The
    header names the configuration the map is derived from, because a
    beamformer refuses a map derived for another one.
    """
    data = np.full((profile.n_elements, n_t), 100, dtype=np.int16)
    return [
        RFEventRecord(
            header=EventHeader(
                seq=k,
                config_id=config_id,
                param_generation=0,
                tx_event_index=k,
                tx_type=tx_types[k] if tx_types else "bmode_focused",
                timestamp_ns=k,
            ),
            data=data.copy(),
        )
        for k in range(n_events)
    ]


# --- the map is data, with constant work --------------------------------


def test_the_identity_map_is_derived_from_the_configuration():
    config = make_bmode_config(linear_5mhz())

    cmap = identity_map(config)

    assert isinstance(cmap, ContributionMap)
    assert len(cmap.line_x_m) == len(config.events)
    assert cmap.event_indices.shape == (len(config.events), 1)
    assert np.array_equal(cmap.event_indices[:, 0], np.arange(len(config.events)))
    assert np.all(cmap.weights == 1.0)
    assert np.array_equal(
        np.asarray(cmap.line_x_m), np.asarray([ev.line_x_m for ev in config.events])
    )


def test_every_line_carries_the_same_number_of_slots():
    """The fixed cap, as a property of the data. Edge lines do not get a
    shorter row; they get inert slots (weight zero) in the same row shape."""
    config = make_bmode_config(linear_5mhz())

    for cmap in (identity_map(config), synthetic_uniform_map(config, cap=3)):
        n_lines = len(cmap.line_x_m)
        assert cmap.event_indices.shape == (n_lines, cmap.event_indices.shape[1])
        assert cmap.weights.shape == cmap.event_indices.shape
        # Every slot names a real event, inert ones included: the compute
        # runs the same reads and multiply-adds on every line.
        assert cmap.event_indices.min() >= 0
        assert cmap.event_indices.max() < len(config.events)


def test_the_map_is_immutable_once_derived():
    cmap = identity_map(make_bmode_config(linear_5mhz()))

    with pytest.raises(ValueError):
        cmap.weights[0, 0] = 2.0


def test_every_line_of_a_derived_map_has_unit_weight_sum():
    """Renormalization is a property of derivation, not of the beamformer:
    each line's weights sum to one, frame edges included, so a plain
    weighted sum cannot make edge lines systematically darker."""
    config = make_bmode_config(linear_5mhz())

    cmap = synthetic_uniform_map(config, cap=3)

    sums = np.asarray(cmap.weights).sum(axis=1)
    assert np.allclose(sums, 1.0, rtol=0.0, atol=1e-12)


def test_finite_weights_with_an_overflowing_sum_are_refused():
    """Review-driven regression: finite weights can overflow their aggregate.

    The resulting infinity must be refused before normalization could turn it
    into an accepted all-zero row.
    """
    with pytest.raises(ValueError, match="non-finite"):
        ContributionMap(
            line_x_m=(0.0,),
            event_indices=np.array([[0, 1]], dtype=np.intp),
            weights=np.array([[1e308, 1e308]], dtype=np.float64),
            config_id="overflow",
            probe_profile_id="linear-5mhz",
            n_events=2,
            param_generation=0,
            line_tx_type=("bmode_focused",),
        )


def test_a_near_zero_weight_sum_is_refused_not_amplified():
    """Dividing by a vanishing sum multiplies noise into a plausible line.
    The floor turns that into a refusal (absolute rules: wrong output is
    worse than no output)."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)

    tiny = np.asarray(cmap.weights) * (WEIGHT_SUM_FLOOR / 10.0)

    with pytest.raises(ValueError, match="weight sum"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=np.asarray(cmap.event_indices),
            weights=tiny,
            config_id=cmap.config_id,
            probe_profile_id=cmap.probe_profile_id,
            n_events=cmap.n_events,
            param_generation=cmap.param_generation,
            line_tx_type=cmap.line_tx_type,
        )


def test_a_map_without_provenance_cannot_be_built():
    """`SOL-57-3`: while the fields could be empty, `check_frame` had nothing
    to compare and an explicitly supplied unbound map was consumed
    unchecked — the same hole as the default path, through the other door.
    Provenance is required at construction, so no such map exists to pass."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)

    with pytest.raises(ValueError, match="must name the configuration"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=np.asarray(cmap.event_indices),
            weights=np.asarray(cmap.weights),
            config_id="",
            probe_profile_id=cmap.probe_profile_id,
            n_events=len(config.events),
            param_generation=0,
            line_tx_type=cmap.line_tx_type,
        )


def test_a_map_claiming_no_events_cannot_be_built():
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)

    with pytest.raises(ValueError, match="must name the configuration"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=np.asarray(cmap.event_indices),
            weights=np.asarray(cmap.weights),
            config_id=config.config_id,
            probe_profile_id=config.probe_profile_id,
            n_events=0,
            param_generation=0,
            line_tx_type=cmap.line_tx_type,
        )


def test_every_derived_map_takes_its_generation_from_the_configuration():
    """`SOL-57-2`: the helpers each defaulted to generation 0 while the
    events carried the accepted one, so a configuration accepted at
    generation 7 produced maps claiming 0 — a derivative disagreeing with
    what it was derived from, which no check could see because both sides
    were self-consistent. The configuration is now the single source."""
    profile = linear_5mhz()
    config = make_bmode_config(profile, param_generation=7)

    assert config.param_generation == 7
    assert {ev.param_generation for ev in config.events} == {7}
    for cmap in (
        identity_map(config),
        mla_map(config, mla=2),
        synthetic_uniform_map(config, cap=3),
    ):
        assert cmap.param_generation == 7


def test_a_frame_at_another_generation_is_refused_through_the_default_path():
    """The consequence at consumption: a configuration accepted at
    generation 7 cannot form a frame whose records carry 0."""
    profile = small_profile()
    config = make_bmode_config(profile, param_generation=7)
    records = constant_records(profile, len(config.events))  # generation 0

    with pytest.raises(ValueError, match="at generation 7, frame carries"):
        das_rf_golden(profile, list(config.events), records)


def _hand_built(config, *, indices=None, weights=None, **overrides):
    """A map built from caller-owned, writable inputs.

    Any field can be overridden, so a sweep of one field against one
    invariant is one call.
    """
    cmap = identity_map(config)
    fields = {
        "line_x_m": cmap.line_x_m,
        "event_indices": np.array(cmap.event_indices, dtype=np.intp)
        if indices is None
        else indices,
        "weights": np.array(cmap.weights, dtype=np.float64) if weights is None else weights,
        "config_id": cmap.config_id,
        "probe_profile_id": cmap.probe_profile_id,
        "n_events": cmap.n_events,
        "param_generation": cmap.param_generation,
        "line_tx_type": cmap.line_tx_type,
    }
    fields.update(overrides)
    return ContributionMap(**fields)


def test_the_map_does_not_alias_a_caller_owned_array():
    """`TERRA-57-001`: `np.ascontiguousarray` returns the caller's own array
    when it is already contiguous, so clearing the writeable flag froze the
    caller's array rather than owning a copy — and the caller could set it
    back. The map copies into immutable bytes at construction, as
    `IQEventRecord` does at its publication boundary (#6)."""
    config = make_bmode_config(linear_5mhz())
    indices = np.array(identity_map(config).event_indices, dtype=np.intp)
    weights = np.array(identity_map(config).weights, dtype=np.float64)

    cmap = _hand_built(config, indices=indices, weights=weights)
    before = int(cmap.event_indices[0, 0]), float(cmap.weights[0, 0])
    indices[0, 0] = 99
    weights[0, 0] = 42.0

    assert cmap.event_indices is not indices
    assert (int(cmap.event_indices[0, 0]), float(cmap.weights[0, 0])) == before
    assert indices.flags.writeable
    assert weights.flags.writeable


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("param_generation", -1, ValueError),
        ("param_generation", 1.5, TypeError),
        ("param_generation", float("nan"), TypeError),
        ("n_events", 1e9, TypeError),
    ],
    ids=["negative-generation", "fractional-generation", "nan-generation", "float-event-count"],
)
def test_the_provenance_counters_must_be_whole_and_non_negative(field, value, error):
    """Found by sweeping every field against every invariant class rather
    than waiting for the next review to name one. The generation is a
    counter (§19); a fractional or negative one compares unequal to every
    real frame, so it fails closed — but it fails at consumption, far from
    the construction that made it wrong."""
    config = make_bmode_config(linear_5mhz())

    with pytest.raises(error):
        _hand_built(config, **{field: value})


def test_a_non_string_transmit_type_condition_is_refused():
    """`str()` would turn 123 into "123": a tag that looks valid, matches no
    record, and surfaces at consumption instead of here."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)

    with pytest.raises(TypeError, match="must be strings"):
        _hand_built(config, line_tx_type=(123,) + tuple(cmap.line_tx_type[1:]))


def test_a_map_that_forms_no_line_is_refused():
    """An empty map produces an empty image rather than an error."""
    with pytest.raises(ValueError, match="at least one line"):
        ContributionMap(
            line_x_m=(),
            event_indices=np.zeros((0, 1), dtype=np.intp),
            weights=np.zeros((0, 1), dtype=np.float64),
            config_id="empty",
            probe_profile_id="linear-5mhz",
            n_events=1,
            param_generation=0,
            line_tx_type=(),
        )


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "+inf", "-inf"]
)
def test_a_non_finite_line_coordinate_is_refused(bad):
    """`TERRA-57-002`: weights and routes were checked for finiteness and the
    abscissae were not. It does not surface as a non-finite image — measured
    before fixing, the read position casts to a garbage integer index and the
    line comes back **all zero**, a silently black scanline rather than an
    error, which is the failure the absolute rules call worse than no
    output."""
    config = make_bmode_config(linear_5mhz())
    coordinates = list(identity_map(config).line_x_m)
    coordinates[3] = bad

    with pytest.raises(ValueError, match="non-finite abscissa"):
        _hand_built(config, line_x_m=coordinates)


@pytest.mark.parametrize("attribute", ["event_indices", "weights"])
def test_writeability_cannot_be_re_enabled(attribute):
    """NumPy lets an array that owns its memory be unfrozen, so a cleared
    flag is not ownership. The exposed arrays are views over `bytes`, whose
    base chain no flag can make writable."""
    cmap = identity_map(make_bmode_config(linear_5mhz()))

    with pytest.raises(ValueError, match="WRITEABLE"):
        getattr(cmap, attribute).flags.writeable = True


def test_a_mutable_line_coordinate_input_is_coerced():
    """A list passed as `line_x_m` used to be retained, so the validated
    line geometry could be rewritten after every check had passed."""
    config = make_bmode_config(linear_5mhz())
    coordinates = list(identity_map(config).line_x_m)

    cmap = _hand_built(config, line_x_m=coordinates)
    coordinates[0] = -1.0

    assert isinstance(cmap.line_x_m, tuple)
    assert cmap.line_x_m[0] != -1.0


def test_the_map_fields_cannot_be_reassigned():
    """`frozen=True` covers rebinding; the tests above cover the buffers it
    does not reach."""
    cmap = identity_map(make_bmode_config(linear_5mhz()))

    with pytest.raises(FrozenInstanceError):
        cmap.line_x_m = ()


# --- MLA as a property of the map ---------------------------------------


@pytest.mark.parametrize("mla", [2, 4])
def test_mla_produces_its_line_count_from_the_same_sequence(mla):
    """MLA 2 and MLA 4, same transmit sequence, different map — nothing
    else varies (§7: MLA reduces transmit count, not formed-line count;
    here the transmit sequence is held fixed so the line count multiplies)."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    cmap = mla_map(config, mla=mla)

    assert len(cmap.line_x_m) == mla * len(config.events)
    assert cmap.event_indices.shape[1] == 1  # pure MLA: one transmit per line


def test_mla_1_degenerates_to_the_identity_geometry_exactly():
    """The decision #53 records: receive lines subdivide the transmit line
    pitch evenly, symmetric about the transmit axis — chosen so MLA 1 is
    the current geometry with no epsilon."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    cmap = mla_map(config, mla=1)
    ident = identity_map(config)

    assert np.array_equal(np.asarray(cmap.line_x_m), np.asarray(ident.line_x_m))
    assert np.array_equal(cmap.event_indices, ident.event_indices)


@pytest.mark.parametrize("mla", [2, 4])
def test_mla_lines_subdivide_the_transmit_pitch_symmetrically(mla):
    """The placement decision #53 records, checked as geometry rather than
    taken on trust: symmetry about the transmit axis is what makes MLA 1
    degenerate exactly, and even spacing is what keeps the group
    translating with the transmit."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    cmap = mla_map(config, mla=mla)

    line_x = np.asarray(cmap.line_x_m)
    for k, ev in enumerate(config.events):
        group = line_x[k * mla : (k + 1) * mla]
        offsets = group - ev.line_x_m
        # symmetric about the transmit axis...
        assert np.allclose(offsets, -offsets[::-1], atol=0.0)
        # ...and evenly spaced at pitch / mla
        assert np.allclose(np.diff(group), profile.pitch_m / mla, atol=1e-15)


def test_an_unsupported_mla_count_is_refused():
    """{2, 4} fixed, 8 the experiment slot (§7). Anything else is a typo,
    not a sweep."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    assert MLA_COUNTS == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="MLA"):
        mla_map(config, mla=3)


def test_a_non_integer_event_index_is_refused():
    """A float index truncates at the read: line k would draw event
    floor(k) and nothing would ever raise."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)

    with pytest.raises(ValueError, match="integer"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=np.asarray(cmap.event_indices, dtype=np.float64) + 0.5,
            weights=np.asarray(cmap.weights),
            config_id=cmap.config_id,
            probe_profile_id=cmap.probe_profile_id,
            n_events=cmap.n_events,
            param_generation=cmap.param_generation,
            line_tx_type=cmap.line_tx_type,
        )


def test_a_negative_event_index_is_refused():
    """Negative indices resolve backwards through the event list rather
    than failing, so the frame forms on the wrong transmits."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)
    indices = np.asarray(cmap.event_indices).copy()
    indices[3, 0] = -1

    with pytest.raises(ValueError, match="negative"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=indices,
            weights=np.asarray(cmap.weights),
            config_id=cmap.config_id,
            probe_profile_id=cmap.probe_profile_id,
            n_events=cmap.n_events,
            param_generation=cmap.param_generation,
            line_tx_type=cmap.line_tx_type,
        )


def test_an_event_index_past_the_configuration_is_refused_at_derivation():
    """Caught where the map is built, not later during image formation."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)
    indices = np.asarray(cmap.event_indices).copy()
    indices[0, 0] = len(config.events)

    with pytest.raises(ValueError, match="past the"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=indices,
            weights=np.asarray(cmap.weights),
            config_id=config.config_id,
            probe_profile_id=config.probe_profile_id,
            n_events=len(config.events),
            param_generation=config.param_generation,
            line_tx_type=cmap.line_tx_type,
        )


def test_the_mla_pitch_comes_from_the_configuration_not_a_profile():
    """The map is derived from the accepted configuration's own geometry,
    so a caller cannot pair it with a different profile's pitch: there is
    no profile argument to get wrong."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    assert element_pitch_m(config) == profile.pitch_m

    cmap = mla_map(config, mla=2)
    offsets = np.asarray(cmap.line_x_m)[:2] - config.events[0].line_x_m
    assert np.allclose(offsets, [-profile.pitch_m / 4, profile.pitch_m / 4], atol=1e-18)


def test_an_even_synthetic_cap_is_refused():
    """A line-centred map has no centre slot at an even cap: the
    contributions straddle the line and shift the fixture laterally by half
    a line, biasing exactly what it is built to isolate."""
    config = make_bmode_config(linear_5mhz())

    with pytest.raises(ValueError, match="odd"):
        synthetic_uniform_map(config, cap=2)


def _sparse_line_config(profile: ProbeProfile, stride: int):
    """A configuration whose transmits sit every `stride` elements.

    The conventional sequence puts one transmit above each element, which
    makes the element pitch and the transmit-line pitch the same number and
    hides which one MLA is subdividing.
    """
    config = make_bmode_config(profile)
    return replace(config, events=tuple(config.events[::stride]))


def test_the_transmit_line_pitch_is_not_the_element_pitch():
    """They coincide only for one-transmit-per-element, which is the whole
    reason a map built on the wrong one goes unnoticed."""
    profile = linear_5mhz()
    sparse = _sparse_line_config(profile, 2)

    assert element_pitch_m(sparse) == profile.pitch_m
    # One ulp of tolerance on the derived value: the span of a decimated
    # line grid does not land on 2·pitch bit-exactly the way the full
    # element grid lands on pitch. The point is the factor, not the bits.
    assert transmit_line_pitch_m(sparse) == pytest.approx(2 * profile.pitch_m, rel=1e-15)


def test_mla_subdivides_the_transmit_line_pitch_not_the_element_pitch():
    """§7 defines MLA against the gap to the next *transmit*: the receive
    lines of one transmit fill it. Deriving the pitch from the elements made
    the groups too narrow whenever a sequence fires fewer events than there
    are elements, and MLA 1 could not show it, its offset being zero either
    way."""
    profile = linear_5mhz()
    sparse = _sparse_line_config(profile, 2)
    line_pitch = transmit_line_pitch_m(sparse)

    cmap = mla_map(sparse, mla=2)

    line_x = np.asarray(cmap.line_x_m)
    offsets = line_x[:2] - sparse.events[0].line_x_m
    assert np.allclose(offsets, [-line_pitch / 4, line_pitch / 4], atol=1e-18)
    # Consecutive groups tile the line grid without gap or overlap.
    assert np.allclose(np.diff(line_x), line_pitch / 2, atol=1e-15)


def test_a_non_uniform_transmit_line_grid_is_refused():
    """A sequence whose beam axes are unevenly spaced has no single pitch to
    subdivide, and what MLA should do there is not defined."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    uneven = replace(config, events=(config.events[0], config.events[1], config.events[3]))

    with pytest.raises(ValueError, match="non-uniform transmit line"):
        mla_map(uneven, mla=2)


def _interleaved_config(profile: ProbeProfile):
    """A sequence alternating two transmit kinds, as a B-mode/colour
    interleave does. Built through the schema, so the tags travel the same
    open-set path a real interleave would (§11.5)."""
    from enodia.spec.sequence import accept, describe_bmode

    description = describe_bmode(profile)
    events = tuple(
        replace(ev, tx_type="bmode_focused" if k % 2 == 0 else "color_flow")
        for k, ev in enumerate(description.events)
    )
    return accept(replace(description, events=events), profile)


def test_each_line_records_the_one_transmit_kind_it_is_formed_from():
    """§7: the map carries transmit-type matching conditions."""
    config = make_bmode_config(linear_5mhz())

    cmap = identity_map(config)

    assert len(cmap.line_tx_type) == cmap.n_lines
    assert set(cmap.line_tx_type) == {"bmode_focused"}


def test_mla_lines_inherit_the_kind_of_their_transmit():
    profile = linear_5mhz()
    interleaved = _interleaved_config(profile)

    cmap = mla_map(interleaved, mla=2)

    # Pure MLA never mixes: each line reads exactly one transmit.
    assert cmap.line_tx_type[:4] == (
        "bmode_focused",
        "bmode_focused",
        "color_flow",
        "color_flow",
    )


def test_a_line_that_would_sum_two_transmit_kinds_is_refused():
    """The condition §7 states in those words: B-mode and colour-Doppler
    interleaves never mix transmit kinds. A multi-contribution map over an
    interleaved sequence would sum a B-mode and a colour transmit into one
    pixel — different pulses, different slow-time meaning, and a
    plausible-looking number with no physical reading."""
    profile = linear_5mhz()
    interleaved = _interleaved_config(profile)

    with pytest.raises(ValueError, match="would sum transmit kinds"):
        synthetic_uniform_map(interleaved, cap=3)


def test_an_interleaved_sequence_still_forms_single_kind_lines():
    """The matching condition refuses mixed rows, not interleaves: the same
    sequence maps fine while each line reads one kind."""
    profile = linear_5mhz()
    interleaved = _interleaved_config(profile)

    cmap = identity_map(interleaved)

    assert set(cmap.line_tx_type) == {"bmode_focused", "color_flow"}
    assert cmap.cap == 1


def test_the_simulator_stamps_the_configurations_own_generation():
    """`SOL-57-003`: the frame used to default to generation 0 whatever the
    configuration was accepted at, so the official default path failed
    closed at any other generation unless the caller passed it twice."""
    from enodia.spec.sim import PointScatterer, simulate_frame

    profile = small_profile()
    config = make_bmode_config(profile, param_generation=7)

    records = simulate_frame(profile, config, [PointScatterer(0.0, 5e-3)])

    assert {r.header.param_generation for r in records} == {7}
    image, _, _ = das_rf_golden(profile, list(config.events), records)
    assert image.shape[1] == len(config.events)


# --- the beamformer reads the map ---------------------------------------


def test_the_identity_map_is_the_general_path_not_a_second_one(frame, golden):
    """Acceptance: with one event per line and unit weights, the pipeline
    reproduces the golden **numerically**. `golden` is the session fixture
    computed with no map argument; passing the explicit identity map must
    give the same array to the last bit, or the identity is a preserved
    special case rather than a case of the general structure."""
    profile, events, records, _ = frame
    config = make_bmode_config(profile)

    image, z, line_x = das_rf_golden(
        profile, events, records, contribution=identity_map(config), dtype=np.float32
    )

    ref_image, ref_z, ref_line_x = golden
    assert np.array_equal(image, ref_image)
    assert np.array_equal(z, ref_z)
    assert np.array_equal(line_x, ref_line_x)


def test_the_beamformer_is_not_told_which_mla_it_is_running(frame):
    """The MLA count appears nowhere in the beamformer's arguments: the map
    alone carries it. Passing an MLA-4 map of the same frame must produce
    four times the lines with no other change of call."""
    profile, events, records, _ = frame
    config = make_bmode_config(profile)
    cmap = mla_map(config, mla=4)

    image, _, line_x = das_rf_golden(profile, events, records, contribution=cmap, dtype=np.float32)

    assert image.shape[1] == 4 * len(events)
    assert np.array_equal(line_x, np.asarray(cmap.line_x_m))


def test_frame_edge_lines_are_not_systematically_darker():
    """The renormalization acceptance, on the field it is stated for: every
    event carries the same constant data, so every output line must come
    out identical. Interior lines draw the full cap of contributions and
    edge lines fewer; only the renormalization makes them equal."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    cmap = synthetic_uniform_map(config, cap=3)

    image, _, _ = das_rf_golden(profile, list(config.events), records, contribution=cmap)

    # Interior depths only: the depth extremes see the record's edges.
    interior = image[image.shape[0] // 4 : -image.shape[0] // 4, :]
    per_line = np.abs(interior).mean(axis=0)
    assert per_line.min() > 0.0
    np.testing.assert_allclose(per_line, per_line[len(per_line) // 2], rtol=1e-5)


class _UnnormalizedMap:
    """A stand-in that reads like a map but skips the normalization.

    It lives here rather than as a flag on `ContributionMap` because
    ADR-0010 decision 3 puts the normalization at derivation so that **every**
    consumer of a map sees the same one; a production escape hatch would be a
    second normalization the decision says does not exist (`SOL-57-001`).
    A negative control needs the behaviour, not the type, so it builds the
    behaviour.
    """

    def __init__(self, derived):
        weights = np.asarray(derived.weights).copy()
        live = weights > 0.0
        # Every live slot at the uniform weight the interior lines get, with
        # nothing dividing the shorter edge rows back up.
        weights[live] = 1.0 / weights.shape[1]
        self.weights = weights
        self.line_x_m = derived.line_x_m
        self.event_indices = np.asarray(derived.event_indices)
        self.line_tx_type = derived.line_tx_type
        self.config_id = derived.config_id
        self.probe_profile_id = derived.probe_profile_id
        self.n_events = derived.n_events
        self.param_generation = derived.param_generation
        self.n_lines = derived.n_lines
        self.cap = derived.cap

    def check_profile(self, profile_name):
        pass

    def check_frame(self, config_id, n_events, param_generation):
        pass


def test_every_constructible_map_has_unit_row_sums():
    """ADR-0010 decision 3, as an invariant of the type rather than a habit
    of its constructors: there is no argument that produces a map whose rows
    sum to anything else."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)
    lopsided = np.asarray(cmap.weights).copy()
    lopsided[:, 0] = 7.0

    for candidate in (
        identity_map(config),
        mla_map(config, mla=2),
        synthetic_uniform_map(config, cap=3),
        _hand_built(config, weights=lopsided),
    ):
        np.testing.assert_allclose(
            np.asarray(candidate.weights).sum(axis=1), 1.0, rtol=0.0, atol=1e-12
        )


def test_the_same_field_shows_the_artifact_when_renormalization_is_removed():
    """The check on the check: un-normalized weights on the same constant
    field leave the outermost lines visibly darker. If this stops failing
    the way it should, the previous test is not testing anything.

    The stand-in is a test-local object, not a `ContributionMap`: the
    accepted type cannot be built this way at all."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    cmap = synthetic_uniform_map(config, cap=3)

    raw = _UnnormalizedMap(cmap)

    image, _, _ = das_rf_golden(profile, list(config.events), records, contribution=raw)

    interior = image[image.shape[0] // 4 : -image.shape[0] // 4, :]
    per_line = np.abs(interior).mean(axis=0)
    centre = per_line[len(per_line) // 2]
    assert per_line[0] < 0.75 * centre
    assert per_line[-1] < 0.75 * centre


def test_the_iq_path_reads_the_same_map_structure(frame):
    """One summation structure, not two: the identity map through `das_iq`
    equals the default call bit for bit, and an MLA map multiplies its
    lines with no other change of call."""
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile, events, records, _ = frame
    config = make_bmode_config(profile)
    iq_records = demodulate_frame(records, profile, decimation=8)

    default_image, _, _ = das_iq(profile, events, iq_records, decimation=8)
    ident_image, _, _ = das_iq(
        profile, events, iq_records, decimation=8, contribution=identity_map(config)
    )

    assert np.array_equal(ident_image, default_image)


def test_a_map_from_another_configuration_is_refused():
    """The Major case: event indices are small integers every configuration
    has, so a stale map resolves cleanly and forms the frame on another
    configuration's scanlines with nothing raised. The records name their
    configuration in the header — the single source of truth for exactly
    this (§19) — so the map names its own and they are compared."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    other = identity_map(make_bmode_config(profile, config_id="another-config"))

    with pytest.raises(ValueError, match="contribution map was derived for"):
        das_rf_golden(profile, list(config.events), records, contribution=other)


def test_a_frame_mixing_configurations_is_refused():
    """Compounding several transmits acquired under different tables is the
    accident snapshot switching exists to prevent (§4, §19)."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    mixed = list(records)
    from dataclasses import replace as dc_replace

    mixed[0] = RFEventRecord(
        header=dc_replace(records[0].header, config_id="other"), data=records[0].data.copy()
    )

    cmap = identity_map(config)

    with pytest.raises(ValueError, match="mixes transmit configurations"):
        das_rf_golden(profile, list(config.events), mixed, contribution=cmap)


def test_every_line_gets_the_same_number_of_delay_evaluations():
    """Fixed work, asserted on the executable path rather than only on the
    map's shape. An earlier version of `_slots_by_event` skipped inert
    slots, so a cap=3 edge line ran two delay-and-aperture evaluations
    where an interior line ran three — the variable-work shape the absolute
    rules forbid, in the very implementation a port is written against."""
    from enodia.spec.beamform import _slots_by_event

    config = make_bmode_config(small_profile())
    cmap = synthetic_uniform_map(config, cap=3)

    per_line: dict[int, int] = {}
    for slots in _slots_by_event(cmap).values():
        for line, _ in slots:
            per_line[line] = per_line.get(line, 0) + 1

    assert set(per_line) == set(range(cmap.n_lines))
    assert set(per_line.values()) == {cmap.cap}


def test_a_map_from_an_earlier_parameter_generation_is_refused():
    """A depth or focus change is an in-config change (§19): the id holds
    still while every derivative behind it is invalidated. A map checked on
    the id alone would survive exactly the change that invalidates it."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))  # generation 0
    stale = identity_map(replace(config, param_generation=1))

    with pytest.raises(ValueError, match="events were accepted under"):
        das_rf_golden(profile, list(config.events), records, contribution=stale)


def test_a_frame_mixing_parameter_generations_is_refused():
    """The generation half of the same rule: a frame assembled across a
    depth or focus change is compounded from transmits acquired under
    different tables (§4, §19)."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    from dataclasses import replace as dc_replace

    mixed = list(records)
    mixed[0] = RFEventRecord(
        header=dc_replace(records[0].header, param_generation=7), data=records[0].data.copy()
    )

    cmap = identity_map(config)

    with pytest.raises(ValueError, match="mixes parameter generations"):
        das_rf_golden(profile, list(config.events), mixed, contribution=cmap)


def test_the_default_identity_path_refuses_another_configurations_records():
    """`SOL-57-001`: the default `contribution=None` path used to carry no
    provenance, so a caller could pair configuration A's events with
    configuration B's records and render them on A's line geometry with
    nothing raised. Event indices are the same small integers in both, so
    `_records_by_event` cannot see it. Accepted events now carry their
    generation tag, which makes the default path checkable like any other."""
    profile = small_profile()
    config_a = make_bmode_config(profile)
    config_b = make_bmode_config(profile, config_id="another-config")
    records_b = constant_records(profile, len(config_b.events), config_id=config_b.config_id)

    with pytest.raises(ValueError, match="accepted under configuration"):
        das_rf_golden(profile, list(config_a.events), records_b)


def test_the_default_iq_path_refuses_another_configurations_records():
    """The same hole through the IQ path, which reads the map identically."""
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile = small_profile()
    config_a = make_bmode_config(profile)
    config_b = make_bmode_config(profile, config_id="another-config")
    records_b = constant_records(profile, len(config_b.events), config_id=config_b.config_id)
    iq_b = demodulate_frame(records_b, profile, decimation=8)

    with pytest.raises(ValueError, match="accepted under configuration"):
        das_iq(profile, list(config_a.events), iq_b, decimation=8)


def test_the_default_path_refuses_events_that_name_no_configuration():
    """Provenance is required, not merely compared: an event list assembled
    by hand has nothing a frame can be checked against."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    unbound = [replace(ev, config_id="") for ev in config.events]

    with pytest.raises(ValueError, match="name no configuration"):
        das_rf_golden(profile, unbound, records)


def test_the_default_path_refuses_events_from_two_configurations():
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    mixed = list(config.events)
    mixed[0] = replace(mixed[0], config_id="other")

    with pytest.raises(ValueError, match="span transmit configurations"):
        das_rf_golden(profile, mixed, records)


def test_a_map_with_no_transmit_type_condition_cannot_be_built():
    """`SOL-57-004`: while the condition was optional, a hand-built map could
    omit it and a consumer that skipped absent conditions was back where the
    field started. It is required, and no entry may be empty — an empty
    condition matches everything, which is the same as having none."""
    config = make_bmode_config(linear_5mhz())
    cmap = identity_map(config)

    with pytest.raises(TypeError):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=np.asarray(cmap.event_indices),
            weights=np.asarray(cmap.weights),
            config_id=cmap.config_id,
            probe_profile_id=cmap.probe_profile_id,
            n_events=cmap.n_events,
            param_generation=cmap.param_generation,
        )

    with pytest.raises(ValueError, match="must name the transmit type"):
        ContributionMap(
            line_x_m=cmap.line_x_m,
            event_indices=np.asarray(cmap.event_indices),
            weights=np.asarray(cmap.weights),
            config_id=cmap.config_id,
            probe_profile_id=cmap.probe_profile_id,
            n_events=cmap.n_events,
            param_generation=cmap.param_generation,
            line_tx_type=("",) * cmap.n_lines,
        )


def test_the_default_identity_map_names_each_lines_transmit_type():
    """The default path populates the condition like any derived map."""
    from enodia.spec.beamform import _identity_contribution

    config = _interleaved_config(linear_5mhz())

    cmap = _identity_contribution(list(config.events))

    assert cmap.line_tx_type == tuple(ev.tx_type for ev in config.events)


def _mixed_kind_map(config):
    """A map that violates the condition, built by hand rather than derived.

    The derivation helpers refuse this shape, which is exactly why the
    consumers need their own check: a map can reach them without passing
    through a helper.
    """
    cmap = identity_map(config)
    indices = np.asarray(cmap.event_indices).copy()
    indices[0, 0] = 1  # line 0 says bmode_focused, now reads a color_flow event
    return ContributionMap(
        line_x_m=cmap.line_x_m,
        event_indices=indices,
        weights=np.asarray(cmap.weights),
        config_id=cmap.config_id,
        probe_profile_id=cmap.probe_profile_id,
        n_events=cmap.n_events,
        param_generation=cmap.param_generation,
        line_tx_type=cmap.line_tx_type,
    )


def test_the_rf_consumer_refuses_a_slot_of_the_wrong_transmit_type():
    profile = small_profile()
    config = _interleaved_config(profile)
    records = constant_records(
        profile, len(config.events), tx_types=tuple(ev.tx_type for ev in config.events)
    )
    mixed = _mixed_kind_map(config)

    with pytest.raises(ValueError, match="matches transmit type"):
        das_rf_golden(profile, list(config.events), records, contribution=mixed)


def test_the_iq_consumer_refuses_a_slot_of_the_wrong_transmit_type():
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile = small_profile()
    config = _interleaved_config(profile)
    records = constant_records(
        profile, len(config.events), tx_types=tuple(ev.tx_type for ev in config.events)
    )
    iq = demodulate_frame(records, profile, decimation=8)
    mixed = _mixed_kind_map(config)

    with pytest.raises(ValueError, match="matches transmit type"):
        das_iq(profile, list(config.events), iq, decimation=8, contribution=mixed)


def test_a_record_whose_header_disagrees_with_its_event_is_refused():
    """The header is what the data calls itself, so both sides are checked:
    a record tagged `color_flow` cannot be summed onto a `bmode_focused`
    line even though the event it names is B-mode."""
    from dataclasses import replace as dc_replace

    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    records[0] = RFEventRecord(
        header=dc_replace(records[0].header, tx_type="color_flow"),
        data=records[0].data.copy(),
    )

    with pytest.raises(ValueError, match="matches transmit type"):
        das_rf_golden(profile, list(config.events), records)


def _duplicated_events(config, n=4):
    """A provenance-valid event list whose position 1 repeats event 0.

    Every other check agrees: same configuration, same generation, same
    transmit type, unit weights. Only the identity of the acquisition is
    wrong, and it carries a line coordinate of its own.
    """
    events = list(config.events[:n])
    events[1] = replace(events[0], line_x_m=events[1].line_x_m)
    return events


def test_the_rf_consumer_refuses_a_duplicated_transmit_event():
    """`TERRA-57-003`: `_records_by_event` derives the wanted indices as a
    **set**, so a duplicate collapses there and its missing-record check
    passes. Measured before fixing: four events naming three transmits
    formed a finite four-line image from three acquisitions, the duplicated
    line carrying the wrong one at a coordinate of its own — plausible and
    wrong, which is what §19's fail-stop rule exists to stop."""
    profile = small_profile()
    config = make_bmode_config(profile)
    events = _duplicated_events(config)
    records = [
        r
        for r in constant_records(profile, len(config.events))
        if r.header.tx_event_index in {ev.event_index for ev in events}
    ]

    with pytest.raises(ValueError, match="appear more than once"):
        das_rf_golden(profile, events, records)


def test_the_iq_consumer_refuses_a_duplicated_transmit_event():
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile = small_profile()
    config = make_bmode_config(profile)
    events = _duplicated_events(config)
    records = [
        r
        for r in constant_records(profile, len(config.events))
        if r.header.tx_event_index in {ev.event_index for ev in events}
    ]

    iq = demodulate_frame(records, profile, decimation=8)

    with pytest.raises(ValueError, match="appear more than once"):
        das_iq(profile, events, iq, decimation=8)


def test_a_reordered_or_sparse_event_list_is_refused():
    """The other half of the same rule `accept` applies to a configuration:
    indices run 0..n-1 in sequence order. Reordering does not lose an
    acquisition, it puts each transmit's data on the other's line."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    swapped = list(config.events)
    swapped[0], swapped[1] = swapped[1], swapped[0]

    with pytest.raises(ValueError, match="in sequence order"):
        das_rf_golden(profile, swapped, records)


def _other_profile() -> ProbeProfile:
    """A different probe of the same channel count — the case a shape check
    cannot catch."""
    return ProbeProfile(
        name="linear-13mhz",
        n_elements=128,
        pitch_m=0.1e-3,
        f0_hz=13.0e6,
        bandwidth_frac=0.7,
        bandwidth_source=None,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=60e-3,
        tx_focus_m=20e-3,
        f_number=2.0,
    )


def test_the_rf_consumer_refuses_another_probe_profile():
    """A configuration names exactly one probe profile (§19, #46), and the
    producing side already refuses a mismatch (`simulate_frame`). The
    consumers took the profile as a separate argument and read the whole
    geometry from it — element positions, sound speed, sampling rate,
    F-number — so a mismatched profile beamformed a real acquisition on
    another probe's delays and returned a plausible image. Found by sweeping
    the consumers' parameters, not by a review."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))

    other = _other_profile()
    events = list(config.events)

    with pytest.raises(ValueError, match="derived for probe profile"):
        das_rf_golden(other, events, records)


def test_the_iq_consumer_refuses_another_probe_profile():
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile = linear_5mhz()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    iq = demodulate_frame(records, profile, decimation=8)

    other = _other_profile()
    events = list(config.events)

    with pytest.raises(ValueError, match="derived for probe profile"):
        das_iq(other, events, iq, decimation=8)


def test_the_default_path_refuses_events_from_two_probe_profiles():
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    mixed = list(config.events)
    mixed[0] = replace(mixed[0], probe_profile_id="linear-13mhz")

    with pytest.raises(ValueError, match="span transmit configurations"):
        das_rf_golden(profile, mixed, records)


def _four_line_map(config):
    """A valid explicit map over the first four events of a configuration."""
    cmap = identity_map(config)
    return ContributionMap(
        line_x_m=cmap.line_x_m[:4],
        event_indices=np.asarray(cmap.event_indices)[:4],
        weights=np.asarray(cmap.weights)[:4],
        config_id=cmap.config_id,
        probe_profile_id=cmap.probe_profile_id,
        n_events=4,
        param_generation=cmap.param_generation,
        line_tx_type=cmap.line_tx_type[:4],
    )


@pytest.mark.parametrize("path", ["rf", "iq"])
def test_the_explicit_map_path_refuses_events_from_another_configuration(path):
    """`SOL-57-001`: the event-provenance check lived inside
    `_identity_contribution`, which only runs when a caller passes **no**
    map — so the explicit-map path never reached it. Measured before fixing:
    config-B / generation-2 events with a config-A / generation-1 map and
    config-A / generation-1 records formed an image, with the events'
    provenance never once compared."""
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile = small_profile()
    config_a = make_bmode_config(profile, config_id="config-A", param_generation=1)
    config_b = make_bmode_config(profile, config_id="config-B", param_generation=2)
    records = constant_records(profile, 4, config_id="config-A")
    cmap = _four_line_map(config_a)
    events_b = list(config_b.events[:4])

    if path == "rf":
        beamform = partial(das_rf_golden, profile, events_b, records, contribution=cmap)
    else:
        iq = demodulate_frame(records, profile, decimation=8)
        beamform = partial(das_iq, profile, events_b, iq, decimation=8, contribution=cmap)

    with pytest.raises(ValueError, match="accepted under configuration"):
        beamform()


@pytest.mark.parametrize("path", ["rf", "iq"])
def test_the_explicit_map_path_refuses_unbound_events(path):
    """The same path, the other shape: events that name no configuration at
    all were previously only refused when they had to build the map."""
    from enodia.spec.beamform.iq_das import das_iq
    from enodia.spec.frontend import demodulate_frame

    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, 4)
    cmap = _four_line_map(config)
    unbound = [replace(ev, config_id="") for ev in config.events[:4]]

    if path == "rf":
        beamform = partial(das_rf_golden, profile, unbound, records, contribution=cmap)
    else:
        iq = demodulate_frame(records, profile, decimation=8)
        beamform = partial(das_iq, profile, unbound, iq, decimation=8, contribution=cmap)

    with pytest.raises(ValueError, match="name no configuration"):
        beamform()


# --- the demo runs it ----------------------------------------------------


def test_the_demo_runs_the_mla_path_end_to_end():
    from enodia.demo import run_pipeline

    profile = small_profile()
    db, _, line_x, _ = run_pipeline(profile, [], path="golden", mla=4, dynamic_range_db=50.0)

    assert db.shape[1] == 4 * profile.n_elements
    assert line_x.size == 4 * profile.n_elements
