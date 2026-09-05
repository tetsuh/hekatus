"""The transmit beam model: the focal singularity and its yardstick (#9, §18).

design.md §18 records the blend across the virtual-source focal singularity
as mandatory and says to build it from the start. It was not built, so these
tests start by exhibiting the defect: a fix whose defect cannot be shown is
not evidence of anything (the lesson `SOL-54-005` left on tautological
tests).

The second thing these tests hold is that the blend is **local**. A blend
that quietly reshaped the delay field at every depth would remove the
artifact and silently move every echo; outside the transition zone the
blended model must reproduce the unblended one exactly, not approximately.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import make_bmode_config
from enodia.spec.sim.transmit import (
    BLEND_HALF_WIDTH_FACTOR,
    aperture_superposition,
    blend_half_width_m,
    blended_sign,
    transmit_contributions,
    virtual_source,
    virtual_source_unblended,
)


def small_profile() -> ProbeProfile:
    """A shrunken aperture, so the superposition model stays fast in tests."""
    return ProbeProfile(
        name="linear-5mhz-small",
        n_elements=16,
        pitch_m=0.3e-3,
        f0_hz=5.0e6,
        bandwidth_frac=0.7,
        bandwidth_source=None,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=20e-3,
        tx_focus_m=8e-3,
        f_number=2.0,
    )


def centre_event(profile: ProbeProfile):
    config = make_bmode_config(profile)
    return config.events[len(config.events) // 2]


def _arrival(model, profile, event, x_m, z_m) -> float:
    taus, _ = model(profile, event, x_m, z_m)
    return float(taus[0])


def test_the_unblended_delay_jumps_across_the_focus():
    """The defect, in the model that has it.

    Off the beam axis the two one-sided limits differ by 2|x − x_v| / c: the
    scatterer is behind the virtual source on one side and in front of it on
    the other, and nothing carries the arrival time between the two.
    """
    profile = small_profile()
    event = centre_event(profile)
    vx, vz = event.virtual_source_m
    offset = 4.0 * profile.wavelength_m
    eps = 1e-9

    below = _arrival(virtual_source_unblended, profile, event, vx + offset, vz - eps)
    above = _arrival(virtual_source_unblended, profile, event, vx + offset, vz + eps)

    expected_jump = 2.0 * offset / profile.c_m_s
    assert abs(above - below) == pytest.approx(expected_jump, rel=1e-6)
    assert expected_jump > 1.0 / profile.f0_hz  # a jump of more than one period


def test_the_blend_removes_the_jump():
    """The same two points under the default model, which is the blended one."""
    profile = small_profile()
    event = centre_event(profile)
    vx, vz = event.virtual_source_m
    offset = 4.0 * profile.wavelength_m
    eps = 1e-9

    below = _arrival(virtual_source, profile, event, vx + offset, vz - eps)
    above = _arrival(virtual_source, profile, event, vx + offset, vz + eps)

    assert abs(above - below) < 1e-4 / profile.f0_hz


def test_the_blended_arrival_is_continuous_through_the_whole_transition():
    """Continuity at one pair of points is not continuity.

    Sampled across the transition zone and out both sides, consecutive
    arrival times must never step by more than the smooth field's own local
    slope allows. The unblended model fails this at exactly one sample gap,
    which is what makes the check meaningful rather than vacuous.
    """
    profile = small_profile()
    event = centre_event(profile)
    vx, vz = event.virtual_source_m
    half = blend_half_width_m(profile)
    offset = 4.0 * profile.wavelength_m
    z = np.linspace(vz - 3.0 * half, vz + 3.0 * half, 601)

    blended = np.array([_arrival(virtual_source, profile, event, vx + offset, zi) for zi in z])
    unblended = np.array(
        [_arrival(virtual_source_unblended, profile, event, vx + offset, zi) for zi in z]
    )

    budget = 4.0 * float(np.median(np.abs(np.diff(blended))))
    assert np.abs(np.diff(blended)).max() < budget
    assert np.abs(np.diff(unblended)).max() > 10.0 * budget


def test_the_blend_is_local():
    """Outside the transition zone the blended model is the unblended one, exactly."""
    profile = small_profile()
    event = centre_event(profile)
    vx, vz = event.virtual_source_m
    half = blend_half_width_m(profile)

    for dz in (half, 1.5 * half, 5.0 * half, 20.0 * half):
        for sign in (-1.0, 1.0):
            z = vz + sign * dz
            if z <= 0.0:
                continue
            for x in (vx, vx + 2.0 * profile.wavelength_m):
                assert _arrival(virtual_source, profile, event, x, z) == _arrival(
                    virtual_source_unblended, profile, event, x, z
                )


def test_blended_sign_is_the_sign_function_outside_and_odd_inside():
    half = 1e-3
    outside = np.array([-5e-3, -1e-3, 1e-3, 5e-3])
    np.testing.assert_array_equal(blended_sign(outside, half), np.sign(outside))

    inside = np.linspace(-0.99e-3, 0.99e-3, 51)
    np.testing.assert_allclose(blended_sign(inside, half), -blended_sign(-inside, half), atol=0.0)
    assert float(blended_sign(0.0, half)) == 0.0


def test_blended_sign_meets_the_sign_function_with_matching_slope():
    """C¹ at the seams, not merely continuous.

    A linear ramp through the zone would meet ±1 at ±h and leave a corner
    there. The test is scale-free: the slope where the blend meets `sign`
    must be negligible against the slope it reaches in the middle of the
    zone, since `sign`'s own slope outside is zero.
    """
    half = 1e-3
    d = half * 1e-6

    def slope(u: float) -> float:
        return (float(blended_sign(u + d, half)) - float(blended_sign(u - d, half))) / (2.0 * d)

    assert abs(float(blended_sign(half - d, half)) - 1.0) < 1e-9
    assert abs(slope(half - 2.0 * d)) / abs(slope(0.0)) < 1e-5
    assert abs(slope(-half + 2.0 * d)) / abs(slope(0.0)) < 1e-5


def test_a_zero_or_negative_blend_width_is_refused():
    profile = small_profile()
    event = centre_event(profile)
    with pytest.raises(ValueError, match="half-width must be finite and positive"):
        virtual_source(profile, event, 0.0, 5e-3, blend_half_width=0.0)
    with pytest.raises(ValueError, match="factor must be finite and positive"):
        blend_half_width_m(profile, factor=0.0)


def test_the_superposition_model_returns_one_contribution_per_element():
    """Including the silent ones: dropping zero-weight elements would make the
    work depend on the aperture, and this implementation is the specification
    a port is written against."""
    profile = small_profile()
    event = centre_event(profile)
    taus, weights = aperture_superposition(profile, event, 0.0, 6e-3)

    assert taus.shape == (profile.n_elements,)
    assert weights.shape == (profile.n_elements,)
    assert float(weights.sum()) == pytest.approx(1.0)
    assert (np.asarray(event.apodization) == 0.0).any()  # the check above is not vacuous


def test_the_superposition_model_focuses_where_the_virtual_source_says_it_does():
    """At the focus the element contributions arrive together; a wavelength off
    axis they do not. That is the beam, emerging rather than assumed."""
    profile = small_profile()
    event = centre_event(profile)
    vx, vz = event.virtual_source_m

    on_axis, _ = aperture_superposition(profile, event, vx, vz)
    spread_on_axis = float(np.ptp(on_axis[np.asarray(event.apodization) > 0.0]))

    off_axis, _ = aperture_superposition(profile, event, vx + 6.0 * profile.wavelength_m, vz)
    spread_off_axis = float(np.ptp(off_axis[np.asarray(event.apodization) > 0.0]))

    assert spread_on_axis < 1.0 / (10.0 * profile.f0_hz)
    assert spread_off_axis > spread_on_axis


def test_the_model_seam_names_its_models_and_refuses_others():
    profile = small_profile()
    event = centre_event(profile)

    default_taus, _ = transmit_contributions(profile, event, 0.0, 6e-3)
    named_taus, _ = transmit_contributions(profile, event, 0.0, 6e-3, model="virtual-source")
    np.testing.assert_array_equal(default_taus, named_taus)

    sup, _ = transmit_contributions(profile, event, 0.0, 6e-3, model="aperture-superposition")
    assert sup.shape == (profile.n_elements,)

    with pytest.raises(ValueError, match="unknown transmit model"):
        transmit_contributions(profile, event, 0.0, 6e-3, model="plane-wave")
    with pytest.raises(ValueError, match="takes no blend half-width"):
        transmit_contributions(
            profile, event, 0.0, 6e-3, model="aperture-superposition", blend_half_width=1e-4
        )


def test_the_default_blend_width_is_the_depth_of_field_scale():
    """λ·F#² is where a converging wavefront stops resembling a point source's,
    so that is the scale the swept factor multiplies — not a length in
    millimetres that would not travel between profiles."""
    profile = linear_5mhz()
    dof = profile.wavelength_m * profile.f_number**2
    assert blend_half_width_m(profile) == pytest.approx(BLEND_HALF_WIDTH_FACTOR * dof)
    assert blend_half_width_m(profile, factor=3.0) == pytest.approx(3.0 * dof)


def test_the_default_factor_is_the_one_the_sweep_selects():
    """The constant in the module is the sweep's answer, not a number beside it.

    Pinned so the two cannot drift apart: if the sweep's metric, region or
    candidate list changes, this fails rather than leaving a stale constant
    in the beam model (the shape design.md §5's kernel figures are pinned in).
    """
    from enodia.spec.sim.blend_sweep import centre_case, selected_factor

    assert selected_factor(centre_case(), n_depths=41) == BLEND_HALF_WIDTH_FACTOR


def test_the_blend_beats_the_unblended_model_against_the_yardstick():
    """The blend is an improvement measured against a model that does not share
    its assumption, not merely a smoother version of the same claim."""
    from enodia.spec.sim.blend_sweep import (
        centre_case,
        unblended_worst_centroid_error_periods,
        worst_centroid_error_periods,
    )

    case = centre_case()
    blended = worst_centroid_error_periods(case, BLEND_HALF_WIDTH_FACTOR, n_depths=41)
    unblended = unblended_worst_centroid_error_periods(case, n_depths=41)

    assert blended < 0.5 * unblended


def test_the_spread_is_a_diagnostic_and_not_a_bound_on_the_error():
    """A one-delay model whose arrival *is* the centroid scores zero under the
    sweep's metric at every point, however wide the arrivals are spread. So
    the spread the sweep reports beside the error is a statement about the
    field, not a floor under any model (`ADV-62-002` corrected an earlier
    "cannot beat" claim)."""
    from enodia.spec.sim.blend_sweep import (
        _worst_shape_error_periods,
        centre_case,
        superposition_centroid_and_spread,
        worst_spread_periods,
    )

    case = centre_case()

    def centroid_model(x: float, z: float) -> float:
        centroid, _ = superposition_centroid_and_spread(case, x, z)
        return centroid

    assert worst_spread_periods(case, n_depths=21) > 0.1
    assert _worst_shape_error_periods(case, centroid_model, n_depths=21) == pytest.approx(0.0, abs=1e-9)


def test_the_models_agree_away_from_the_focus_to_a_stated_tolerance():
    """#9 acceptance criterion 3, with the numbers in it.

    Beyond the focus the virtual-source picture is at its best and the two
    models agree to within 0.3 periods out to two beam widths off axis;
    before it, where the wavefront is still converging and least resembles a
    point source's, they agree to within 1.4 periods from a quarter of the
    focal depth. The figures are pinned here and reported by
    `blend_sweep.away_from_focus_table`, so the prose that quotes them
    cannot drift from what the code computes.
    """
    from enodia.spec.sim.blend_sweep import away_from_focus_report, centre_case

    rows = away_from_focus_report(centre_case())
    beyond = [abs(err) for mult, _, err, _ in rows if mult > 1.0]
    before = [abs(err) for mult, _, err, _ in rows if mult < 1.0]

    assert beyond and before  # both sides sampled, or the tolerances mean nothing
    assert max(beyond) < 0.3
    assert max(before) < 1.4


def test_the_virtual_source_model_refuses_an_aperture_it_does_not_describe():
    """The ingress binds the two models to the same focus, not the same
    aperture (`ADV-62-001`). A single-element event passes `accept` and
    focuses exactly where it says; the Gaussian set by the profile's F-number
    describes nothing about how it is lit. Refused, naming the model that
    reads the apodization, rather than approximated."""
    from dataclasses import replace as dc_replace

    profile = small_profile()
    event = centre_event(profile)
    single = dc_replace(
        event,
        apodization=tuple(1.0 if i == profile.n_elements // 2 else 0.0 for i in range(profile.n_elements)),
    )

    with pytest.raises(ValueError, match="aperture-superposition"):
        virtual_source(profile, single, 0.0, 6e-3)
    with pytest.raises(ValueError, match="aperture-superposition"):
        virtual_source_unblended(profile, single, 0.0, 6e-3)

    taus, weights = aperture_superposition(profile, single, 0.0, 6e-3)
    assert taus.shape == (profile.n_elements,)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_the_virtual_source_model_refuses_a_beam_axis_off_the_virtual_source():
    from dataclasses import replace as dc_replace

    profile = small_profile()
    event = centre_event(profile)
    skewed = dc_replace(event, line_x_m=event.line_x_m + 2.0 * profile.pitch_m)

    with pytest.raises(ValueError, match="beam axis passes through the virtual source"):
        virtual_source(profile, skewed, 0.0, 6e-3)


def test_every_event_of_the_profile_configuration_is_inside_the_domain():
    """Edge events have truncated, asymmetric apertures — and they are still
    the profile's focused aperture, because that is what `focused_aperture`
    returns at the edge. The domain is what the configuration produces, not
    an idealisation the edges fall outside of."""
    from enodia.spec.sim.transmit import check_virtual_source_domain

    profile = linear_5mhz()
    for event in make_bmode_config(profile).events:
        check_virtual_source_domain(profile, event)


def _peak_sample(record, channel: int) -> int:
    return int(np.argmax(np.abs(np.asarray(record.data)[channel])))


def test_the_simulator_carries_the_blend_into_the_rf():
    """The model seam is not decoration: the frame changes across the focus.

    Two scatterers a tenth of a blend width either side of the focal depth,
    off the beam axis. Under the unblended model the received echo jumps by
    the round-trip of `2|x − x_v|`; under the default model it does not.

    **The existing suite could not have caught this.** Every other simulator
    test places its scatterer at its profile's focal depth — 20 mm on
    `linear-5mhz`, 5 mm on the shrunken profile `test_contribution_map` uses
    — and that is the one depth where the unblended model is well behaved,
    because `sign(0)` is zero and the two models agree there exactly. The
    artifact lives just off the focus, not on it.
    """
    from enodia.spec.sim import PointScatterer, simulate_frame

    profile = small_profile()
    config = make_bmode_config(profile)
    event = centre_event(profile)
    vx, vz = event.virtual_source_m
    one_event = replace(config, events=(event,))
    offset = 4.0 * profile.wavelength_m
    dz = 0.1 * blend_half_width_m(profile)
    channel = profile.n_elements // 2

    peaks = {}
    for model in ("virtual-source", "virtual-source-unblended"):
        below = simulate_frame(
            profile, one_event, [PointScatterer(vx + offset, vz - dz)], transmit_model=model
        )
        above = simulate_frame(
            profile, one_event, [PointScatterer(vx + offset, vz + dz)], transmit_model=model
        )
        peaks[model] = (_peak_sample(below[0], channel), _peak_sample(above[0], channel))

    # What the echo should move by, from geometry rather than from the
    # implementation: the transmit time flips by 2r/c across the focus, and
    # the two scatterers are also at different depths, so the receive leg
    # changes too. Both terms, or the number means nothing.
    r_sv = float(np.hypot(offset, dz))
    el_x = profile.element_x()[channel]
    t_rx_below = float(np.hypot(vx + offset - el_x, vz - dz)) / profile.c_m_s
    t_rx_above = float(np.hypot(vx + offset - el_x, vz + dz)) / profile.c_m_s
    jump_samples = (2.0 * r_sv / profile.c_m_s + (t_rx_above - t_rx_below)) * profile.fs_hz

    blended_step = abs(peaks["virtual-source"][1] - peaks["virtual-source"][0])
    unblended_step = abs(peaks["virtual-source-unblended"][1] - peaks["virtual-source-unblended"][0])

    assert unblended_step == pytest.approx(jump_samples, abs=1.0)

    # The receive leg genuinely differs between the two depths and no blend
    # should touch it. Subtracting it isolates the transmit discontinuity,
    # which is the whole of what is being repaired; comparing the raw steps
    # would credit the blend for a term it does not act on.
    rx_only = (t_rx_above - t_rx_below) * profile.fs_hz
    assert abs(blended_step - rx_only) < 0.25 * abs(unblended_step - rx_only)

    # And the residual is the size the sweep says to expect: about one period
    # of f0, not zero. The blend is a one-delay stand-in for a field the
    # sweep shows is not a single arrival near the focus; zero would be
    # surprising, not reassuring.
    samples_per_period = profile.fs_hz / profile.f0_hz
    assert abs(blended_step - rx_only) < 2.0 * samples_per_period


def test_the_simulator_leaves_the_frame_alone_away_from_the_focus():
    """Where the blend is inactive the frame is bit-for-bit the pre-#9 one.

    This is what makes the change a local repair rather than a new simulator:
    every existing golden and format test sits outside the transition zone or
    exactly on the focal depth, and none of them moved.
    """
    from enodia.spec.sim import PointScatterer, simulate_frame

    profile = small_profile()
    config = make_bmode_config(profile)
    _, vz = profile.tx_focus_m, profile.tx_focus_m
    far = vz + 4.0 * blend_half_width_m(profile)

    blended = simulate_frame(profile, config, [PointScatterer(0.0, far)])
    unblended = simulate_frame(
        profile, config, [PointScatterer(0.0, far)], transmit_model="virtual-source-unblended"
    )
    for a, b in zip(blended, unblended, strict=True):
        np.testing.assert_array_equal(np.asarray(a.data), np.asarray(b.data))


def test_the_superposition_model_produces_a_frame_of_the_same_shape():
    """The switch changes the beam, not the record contract."""
    from enodia.spec.sim import PointScatterer, simulate_frame

    profile = small_profile()
    config = make_bmode_config(profile)
    scatterers = [PointScatterer(0.0, 6e-3)]

    default = simulate_frame(profile, config, scatterers)
    superposed = simulate_frame(profile, config, scatterers, transmit_model="aperture-superposition")

    assert len(superposed) == len(default)
    for a, b in zip(default, superposed, strict=True):
        assert np.asarray(a.data).shape == np.asarray(b.data).shape
        assert a.header.config_id == b.header.config_id
        assert a.header.param_generation == b.header.param_generation
    assert np.abs(np.asarray(superposed[0].data)).max() > 0
