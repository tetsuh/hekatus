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
    identity_map,
    mla_map,
    synthetic_uniform_map,
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


def constant_records(profile: ProbeProfile, n_events: int, n_t: int = 512):
    """One frame in which every event carries the same constant data.

    On such a frame, every output line must come out identical: any line
    that differs is showing the summation structure, not the field.
    """
    data = np.full((profile.n_elements, n_t), 100, dtype=np.int16)
    return [
        RFEventRecord(
            header=EventHeader(
                seq=k,
                config_id="uniform",
                param_generation=0,
                tx_event_index=k,
                tx_type="bmode_focused",
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
        np.asarray(cmap.line_x_m), np.array([ev.line_x_m for ev in config.events])
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
        )


# --- MLA as a property of the map ---------------------------------------


@pytest.mark.parametrize("mla", [2, 4])
def test_mla_produces_its_line_count_from_the_same_sequence(mla):
    """MLA 2 and MLA 4, same transmit sequence, different map — nothing
    else varies (§7: MLA reduces transmit count, not formed-line count;
    here the transmit sequence is held fixed so the line count multiplies)."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    cmap = mla_map(config, profile, mla=mla)

    assert len(cmap.line_x_m) == mla * len(config.events)
    assert cmap.event_indices.shape[1] == 1  # pure MLA: one transmit per line


def test_mla_1_degenerates_to_the_identity_geometry_exactly():
    """The decision #53 records: receive lines subdivide the transmit line
    pitch evenly, symmetric about the transmit axis — chosen so MLA 1 is
    the current geometry with no epsilon."""
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    cmap = mla_map(config, profile, mla=1)
    ident = identity_map(config)

    assert np.array_equal(np.asarray(cmap.line_x_m), np.asarray(ident.line_x_m))
    assert np.array_equal(cmap.event_indices, ident.event_indices)


@pytest.mark.parametrize("mla", [2, 4])
def test_mla_lines_subdivide_the_transmit_pitch_symmetrically(mla):
    profile = linear_5mhz()
    config = make_bmode_config(profile)

    cmap = mla_map(config, profile, mla=mla)

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
        mla_map(config, profile, mla=3)


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
    cmap = mla_map(config, profile, mla=4)

    image, _, line_x = das_rf_golden(
        profile, events, records, contribution=cmap, dtype=np.float32
    )

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

    image, z, _ = das_rf_golden(profile, list(config.events), records, contribution=cmap)

    # Interior depths only: the depth extremes see the record's edges.
    interior = image[image.shape[0] // 4 : -image.shape[0] // 4, :]
    per_line = np.abs(interior).mean(axis=0)
    assert per_line.min() > 0.0
    np.testing.assert_allclose(per_line, per_line[len(per_line) // 2], rtol=1e-5)


def test_the_same_field_shows_the_artifact_when_renormalization_is_removed():
    """The check on the check: un-normalized weights on the same constant
    field leave the outermost lines visibly darker. If this stops failing
    the way it should, the previous test is not testing anything."""
    profile = small_profile()
    config = make_bmode_config(profile)
    records = constant_records(profile, len(config.events))
    cmap = synthetic_uniform_map(config, cap=3)

    # The raw structure with renormalization undone: every live slot at the
    # uniform weight the interior lines get.
    weights = np.asarray(cmap.weights).copy()
    live = weights > 0.0
    weights[live] = 1.0 / cmap.weights.shape[1]
    raw = ContributionMap(
        line_x_m=cmap.line_x_m,
        event_indices=np.asarray(cmap.event_indices),
        weights=weights,
        normalized=False,
    )

    image, _, _ = das_rf_golden(profile, list(config.events), records, contribution=raw)

    interior = image[image.shape[0] // 4 : -image.shape[0] // 4, :]
    per_line = np.abs(interior).mean(axis=0)
    centre = per_line[len(per_line) // 2]
    assert per_line[0] < 0.75 * centre
    assert per_line[-1] < 0.75 * centre


# --- the demo runs it ----------------------------------------------------


def test_the_demo_runs_the_mla_path_end_to_end():
    from enodia.demo import run_pipeline

    profile = small_profile()
    db, z, line_x, _ = run_pipeline(
        profile, [], path="golden", mla=4, dynamic_range_db=50.0
    )

    assert db.shape[1] == 4 * profile.n_elements
    assert line_x.size == 4 * profile.n_elements
