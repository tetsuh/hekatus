"""The simulator evaluates pulse samples only inside a fixed Gaussian support (#165)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from enodia.demo import DEFAULT_SCATTERERS
from enodia.spec import sim
from enodia.spec.probe import ProbeProfile, linear_5mhz
from enodia.spec.sequence import make_bmode_config


def _old_full_frame(profile, events, scatterers, *, transmit_model="virtual-source"):
    """Independent copy of the pre-#165 full-record float64 summation."""
    el_x = profile.element_x()
    n_t = sim.n_rf_samples(profile)
    t_axis = np.arange(n_t, dtype=np.float64) / profile.fs_hz
    frame = np.zeros((len(events), profile.n_elements, n_t), dtype=np.float64)
    for event_index, event in enumerate(events):
        for scatterer in scatterers:
            taus, weights = sim.transmit_contributions(
                profile, event, scatterer.x_m, scatterer.z_m, model=transmit_model
            )
            t_rx = np.hypot(scatterer.x_m - el_x, scatterer.z_m) / profile.c_m_s
            for tau, weight in zip(taus, weights, strict=True):
                arrival = float(tau) + t_rx
                frame[event_index] += (
                    scatterer.amplitude
                    * float(weight)
                    * sim.gaussian_pulse(
                        t_axis[None, :] - arrival[:, None],
                        profile.f0_hz,
                        profile.bandwidth_frac,
                    )
                )
    return frame


def _events(profile, count=1):
    return list(make_bmode_config(profile).events[:count])


def test_both_simulator_entrypoints_expose_the_defaulted_keyword():
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    events = list(config.events[:1])
    scatterers = [sim.PointScatterer(0.0, 20e-3)]
    default_private = sim._simulate_bmode_frame(profile, events, scatterers)
    explicit_private = sim._simulate_bmode_frame(
        profile, events, scatterers, pulse_support_sigmas=sim.DEFAULT_PULSE_SUPPORT_SIGMAS
    )
    default_public = sim.simulate_frame(profile, replace(config, events=tuple(events)), scatterers)
    explicit_public = sim.simulate_frame(
        profile,
        replace(config, events=tuple(events)),
        scatterers,
        pulse_support_sigmas=sim.DEFAULT_PULSE_SUPPORT_SIGMAS,
    )
    np.testing.assert_array_equal(default_private[0].data, explicit_private[0].data)
    np.testing.assert_array_equal(default_public[0].data, explicit_public[0].data)
    assert sim.DEFAULT_PULSE_SUPPORT_SIGMAS > 0


def test_none_is_the_unchanged_full_record_reference_and_scaling_is_frame_wide():
    profile = linear_5mhz()
    events = _events(profile, 2)
    scatterers = [sim.PointScatterer(-1e-3, 15e-3), sim.PointScatterer(3e-3, 35e-3, 0.3)]
    expected_float = _old_full_frame(profile, events, scatterers)

    got_float = sim._simulate_bmode_float64_frame(
        profile, events, scatterers, pulse_support_sigmas=None
    )
    np.testing.assert_array_equal(got_float, expected_float)

    records = sim._simulate_bmode_frame(
        profile, events, scatterers, pulse_support_sigmas=None, int16_fullscale_frac=0.4
    )
    frame_scale = 0.4 * np.iinfo(np.int16).max / np.abs(expected_float).max()
    assert np.max(np.abs(expected_float[0])) != pytest.approx(np.max(np.abs(expected_float[1])))
    for event_index, record in enumerate(records):
        np.testing.assert_array_equal(
            record.data, np.round(expected_float[event_index] * frame_scale).astype(np.int16)
        )


def test_support_evaluates_exactly_the_samples_inside_the_sigma_window(monkeypatch):
    profile = linear_5mhz()
    events = _events(profile)
    scatterers = [sim.PointScatterer(1.7e-3, 23e-3, 0.75)]
    k = 5.5
    sigma = sim._gaussian_envelope_sigma(profile.f0_hz, profile.bandwidth_frac)
    expected_full = _old_full_frame(profile, events, scatterers)
    original_pulse = sim.gaussian_pulse
    evaluated_times = []

    def pulse_spy(t, f0_hz, bandwidth_frac):
        evaluated_times.append(np.asarray(t).copy())
        return original_pulse(t, f0_hz, bandwidth_frac)

    monkeypatch.setattr(sim, "gaussian_pulse", pulse_spy)
    got = sim._simulate_bmode_float64_frame(
        profile, events, scatterers, pulse_support_sigmas=k
    )
    assert len(evaluated_times) == 1
    assert np.all(np.abs(evaluated_times[0]) <= k * sigma)
    expected_width = int(np.ceil(2.0 * k * sigma * profile.fs_hz)) + 1
    assert evaluated_times[0].size == profile.n_elements * expected_width

    t_axis = np.arange(sim.n_rf_samples(profile), dtype=np.float64) / profile.fs_hz
    taus, _ = sim.transmit_contributions(
        profile, events[0], scatterers[0].x_m, scatterers[0].z_m
    )
    t_rx = np.hypot(scatterers[0].x_m - profile.element_x(), scatterers[0].z_m) / profile.c_m_s
    for tau in taus:
        arrival = float(tau) + t_rx
        inside = np.abs(t_axis[None, :] - arrival[:, None]) <= k * sigma
        np.testing.assert_array_equal(got[0][inside], expected_full[0][inside])
        assert np.all(got[0][~inside] == 0.0)


@pytest.mark.parametrize("transmit_model", ["virtual-source", "aperture-superposition"])
def test_windows_clip_at_both_record_edges_without_wrapping(monkeypatch, transmit_model):
    profile = ProbeProfile(
        name="linear-5mhz-two-elements",
        n_elements=2,
        pitch_m=1e-9,
        f0_hz=5e6,
        bandwidth_frac=0.7,
        fs_hz=40e6,
        c_m_s=1540.0,
        depth_m=10e-3,
        tx_focus_m=5e-3,
        f_number=2.0,
    )
    events = _events(profile)
    scatterer = sim.PointScatterer(0.0, 5e-3)
    monkeypatch.setattr(sim, "n_rf_samples", lambda _: 64)
    sigma = sim._gaussian_envelope_sigma(profile.f0_hz, profile.bandwidth_frac)
    sample_period = 1.0 / profile.fs_hz

    for target_arrival in (0.25 * sample_period, 63.75 * sample_period):
        def near_edge_transmit(p, event, x_m, z_m, target=target_arrival, **kwargs):
            t_rx = np.hypot(x_m - p.element_x(), z_m) / p.c_m_s
            tau = target - float(t_rx.mean())
            return np.array([tau]), np.array([1.0])

        monkeypatch.setattr(sim, "transmit_contributions", near_edge_transmit)
        reference = _old_full_frame(
            profile, events, [scatterer], transmit_model=transmit_model
        )
        clipped = sim._simulate_bmode_float64_frame(
            profile,
            events,
            [scatterer],
            transmit_model=transmit_model,
            pulse_support_sigmas=6.0,
        )
        actual_arrival = (
            target_arrival
            - float(
                (
                    np.hypot(scatterer.x_m - profile.element_x(), scatterer.z_m)
                    / profile.c_m_s
                ).mean()
            )
            + np.hypot(scatterer.x_m - profile.element_x(), scatterer.z_m) / profile.c_m_s
        )
        t_axis = np.arange(64, dtype=np.float64) / profile.fs_hz
        inside = np.abs(t_axis[None, :] - actual_arrival[:, None]) <= 6.0 * sigma
        np.testing.assert_array_equal(clipped[0][inside], reference[0][inside])
        assert np.all(clipped[0][~inside] == 0.0)
        if target_arrival < sample_period:
            assert np.all(clipped[..., -1] == 0.0)
        else:
            assert np.all(clipped[..., 0] == 0.0)


def test_accumulator_includes_exact_support_boundaries_and_handles_zero_k_and_no_overlap(
    monkeypatch,
):
    profile = ProbeProfile(
        name="support-grid",
        n_elements=2,
        pitch_m=1e-3,
        f0_hz=5e6,
        bandwidth_frac=0.7,
        fs_hz=4.0,
        c_m_s=1540.0,
        depth_m=1.0,
        tx_focus_m=0.5,
        f_number=2.0,
    )
    t_axis = np.arange(5, dtype=np.float64) / profile.fs_hz
    evaluated = []

    def pulse_spy(t, f0_hz, bandwidth_frac):
        evaluated.append(np.asarray(t).copy())
        return np.ones_like(t)

    monkeypatch.setattr(sim, "gaussian_pulse", pulse_spy)
    frame = np.zeros((1, t_axis.size), dtype=np.float64)
    sim._accumulate_supported_pulse(
        frame, t_axis, np.array([0.5]), 1.0, profile, 0.25, 1.0
    )
    np.testing.assert_array_equal(frame[0], np.array([0.0, 1.0, 1.0, 1.0, 0.0]))
    assert np.all(np.abs(evaluated[-1]) <= 0.25)

    zero_k_frame = np.zeros_like(frame)
    sim._accumulate_supported_pulse(
        zero_k_frame, t_axis, np.array([0.5]), 1.0, profile, 0.25, 0.0
    )
    np.testing.assert_array_equal(zero_k_frame[0], np.array([0.0, 0.0, 1.0, 0.0, 0.0]))

    non_grid_arrival_frame = np.zeros_like(frame)
    sim._accumulate_supported_pulse(
        non_grid_arrival_frame, t_axis, np.array([0.55]), 1.0, profile, 0.25, 0.0
    )
    assert np.all(non_grid_arrival_frame == 0.0)

    outside_frame = np.zeros((2, t_axis.size), dtype=np.float64)
    sim._accumulate_supported_pulse(
        outside_frame, t_axis, np.array([-3.0, 4.0]), 1.0, profile, 0.25, 1.0
    )
    assert np.all(outside_frame == 0.0)


def test_rounding_at_a_searchsorted_endpoint_never_evaluates_outside_support(monkeypatch):
    profile = ProbeProfile(
        name="floating-support-grid",
        n_elements=1,
        pitch_m=1e-3,
        f0_hz=5e6,
        bandwidth_frac=0.7,
        fs_hz=1e15,
        c_m_s=1540.0,
        depth_m=1.0,
        tx_focus_m=0.5,
        f_number=2.0,
    )
    t_axis = np.arange(2000, dtype=np.float64) / profile.fs_hz
    arrival = np.array([1e-12])
    sigma = 1e-15
    radius = sigma
    boundary_index = 1001
    assert t_axis[boundary_index] == arrival[0] + radius
    assert t_axis[boundary_index] - arrival[0] > radius
    evaluated = []

    def pulse_spy(t, f0_hz, bandwidth_frac):
        evaluated.append(np.asarray(t).copy())
        return np.ones_like(t)

    monkeypatch.setattr(sim, "gaussian_pulse", pulse_spy)
    frame = np.zeros((1, t_axis.size), dtype=np.float64)
    sim._accumulate_supported_pulse(frame, t_axis, arrival, 1.0, profile, sigma, 1.0)
    assert frame[0, boundary_index] == 0.0
    assert np.all(np.abs(evaluated[0]) <= radius)


def test_zero_weight_contributions_are_still_evaluated(monkeypatch):
    profile = ProbeProfile(
        name="linear-5mhz-two-elements",
        n_elements=2,
        pitch_m=0.3e-3,
        f0_hz=5e6,
        bandwidth_frac=0.7,
        fs_hz=40e6,
        c_m_s=1540.0,
        depth_m=10e-3,
        tx_focus_m=5e-3,
        f_number=2.0,
    )
    events = _events(profile)
    scatterers = [sim.PointScatterer(0.0, 5e-3)]
    calls = []
    original_pulse = sim.gaussian_pulse

    def two_contributions(p, event, x_m, z_m, **kwargs):
        return np.array([0.0, 0.25 / p.f0_hz]), np.array([0.0, 1.0])

    def pulse_spy(t, f0_hz, bandwidth_frac):
        calls.append(np.asarray(t).size)
        return original_pulse(t, f0_hz, bandwidth_frac)

    monkeypatch.setattr(sim, "transmit_contributions", two_contributions)
    monkeypatch.setattr(sim, "gaussian_pulse", pulse_spy)
    sim._simulate_bmode_float64_frame(
        profile, events, scatterers, pulse_support_sigmas=6.0
    )
    assert len(calls) == 2
    assert len(set(calls)) == 1
    assert all(size > 0 for size in calls)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_support_sigmas_must_be_finite_and_nonnegative(bad):
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    with pytest.raises(ValueError, match="pulse_support_sigmas"):
        sim.simulate_frame(
            profile,
            replace(config, events=config.events[:1]),
            [],
            pulse_support_sigmas=bad,
        )
    with pytest.raises(ValueError, match="pulse_support_sigmas"):
        sim._simulate_bmode_frame(profile, list(config.events[:1]), [], pulse_support_sigmas=bad)


@pytest.mark.parametrize("bad", [True, "6.0"])
def test_support_sigmas_reject_non_numeric_values(bad):
    profile = linear_5mhz()
    config = make_bmode_config(profile)
    with pytest.raises(TypeError, match="pulse_support_sigmas"):
        sim.simulate_frame(
            profile,
            replace(config, events=config.events[:1]),
            [],
            pulse_support_sigmas=bad,
        )


def test_a_finite_support_wider_than_the_record_evaluates_the_full_record():
    profile = linear_5mhz()
    events = _events(profile)
    scatterers = [sim.PointScatterer(0.0, 20e-3)]
    reference = _old_full_frame(profile, events, scatterers)
    got = sim._simulate_bmode_float64_frame(
        profile, events, scatterers, pulse_support_sigmas=1e300
    )
    np.testing.assert_array_equal(got, reference)


def test_gaussian_pulse_and_truncation_use_the_shared_envelope_sigma(monkeypatch):
    profile = linear_5mhz()
    expected_sigma = np.sqrt(2.0 * np.log(2.0)) / (
        np.pi * profile.bandwidth_frac * profile.f0_hz
    )
    assert sim._gaussian_envelope_sigma(profile.f0_hz, profile.bandwidth_frac) == expected_sigma
    t = np.array([0.0, expected_sigma])
    expected = np.exp(-0.5 * (t / expected_sigma) ** 2) * np.cos(
        2.0 * np.pi * profile.f0_hz * t
    )
    np.testing.assert_array_equal(
        sim.gaussian_pulse(t, profile.f0_hz, profile.bandwidth_frac), expected
    )


def test_default_support_meets_the_relative_error_bound_on_existing_and_seeded_fields():
    profile = linear_5mhz()
    events = _events(profile)
    scatterer_sets = [
        [sim.PointScatterer(0.0, 20e-3)],
        DEFAULT_SCATTERERS,
        [sim.PointScatterer(-5e-3, 15e-3), sim.PointScatterer(7e-3, 25e-3)],
    ]
    rng = np.random.default_rng(165)
    scatterer_sets.append(
        [
            sim.PointScatterer(
                float(x), float(z), float(amplitude)
            )
            for x, z, amplitude in zip(
                rng.uniform(-10e-3, 10e-3, 200),
                rng.uniform(20e-3, 40e-3, 200),
                rng.uniform(0.25, 1.5, 200),
                strict=True,
            )
        ]
    )

    for scatterers in scatterer_sets:
        reference = _old_full_frame(profile, events, scatterers)
        got = sim._simulate_bmode_float64_frame(
            profile,
            events,
            scatterers,
            pulse_support_sigmas=sim.DEFAULT_PULSE_SUPPORT_SIGMAS,
        )
        relative_error = np.max(np.abs(got - reference)) / np.max(np.abs(reference))
        assert relative_error <= 1e-6, f"{len(scatterers)} scatterers: {relative_error:.3g}"


@pytest.mark.slow
def test_default_support_meets_error_bound_for_full_profile_all_events():
    profile = linear_5mhz()
    events = list(make_bmode_config(profile).events)
    rng = np.random.default_rng(165)
    random_field = [
        sim.PointScatterer(float(x), float(z), float(amplitude))
        for x, z, amplitude in zip(
            rng.uniform(-10e-3, 10e-3, 200),
            rng.uniform(20e-3, 40e-3, 200),
            rng.uniform(0.25, 1.5, 200),
            strict=True,
        )
    ]
    scatterer_sets = [
        [sim.PointScatterer(0.0, 20e-3)],
        DEFAULT_SCATTERERS,
        random_field,
    ]

    for scatterers in scatterer_sets:
        reference = _old_full_frame(profile, events, scatterers)
        got = sim._simulate_bmode_float64_frame(
            profile,
            events,
            scatterers,
            pulse_support_sigmas=sim.DEFAULT_PULSE_SUPPORT_SIGMAS,
        )
        relative_error = np.max(np.abs(got - reference)) / np.max(np.abs(reference))
        assert relative_error <= 1e-6, f"{len(scatterers)} scatterers: {relative_error:.3g}"


@pytest.mark.slow
@pytest.mark.parametrize("transmit_model", ["virtual-source", "aperture-superposition"])
def test_default_support_matches_existing_cheap_fixture_for_both_models(
    cheap_profile, cheap_config, cheap_scatterers, transmit_model
):
    events = list(cheap_config.events)
    reference = _old_full_frame(
        cheap_profile, events, cheap_scatterers, transmit_model=transmit_model
    )
    got = sim._simulate_bmode_float64_frame(
        cheap_profile,
        events,
        cheap_scatterers,
        transmit_model=transmit_model,
        pulse_support_sigmas=sim.DEFAULT_PULSE_SUPPORT_SIGMAS,
    )
    relative_error = np.max(np.abs(got - reference)) / np.max(np.abs(reference))
    assert relative_error <= 1e-6


def test_default_support_matches_existing_small_aperture_cases_for_both_models():
    profile = ProbeProfile(
        name="linear-5mhz-small",
        n_elements=16,
        pitch_m=0.3e-3,
        f0_hz=5.0e6,
        bandwidth_frac=0.7,
        fs_hz=40.0e6,
        c_m_s=1540.0,
        depth_m=20e-3,
        tx_focus_m=8e-3,
        f_number=2.0,
    )
    config = make_bmode_config(profile)
    events = [config.events[len(config.events) // 2]]
    scatterers = [sim.PointScatterer(0.0, 6e-3)]

    for transmit_model in ("virtual-source", "aperture-superposition"):
        reference = _old_full_frame(
            profile, events, scatterers, transmit_model=transmit_model
        )
        got = sim._simulate_bmode_float64_frame(
            profile,
            events,
            scatterers,
            transmit_model=transmit_model,
            pulse_support_sigmas=sim.DEFAULT_PULSE_SUPPORT_SIGMAS,
        )
        relative_error = np.max(np.abs(got - reference)) / np.max(np.abs(reference))
        assert relative_error <= 1e-6, transmit_model


def test_sweep_run_writes_complete_records_and_refuses_overwrite(tmp_path, monkeypatch):
    import json

    from enodia.spec.sim import pulse_support_sweep as sweep

    commit = "165c0ffee123456789abcdef0123456789abcdef"
    monkeypatch.setattr(
        sweep,
        "_git",
        lambda *args: commit if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(sweep, "_cpu_model", lambda: "Test CPU model")
    calls = []
    error_by_k = {4.0: 2e-6, 5.0: 8e-7, 5.5: 7e-7, 6.0: 1e-7, 7.0: 2e-7, 8.0: 3e-7}
    seconds_by_k = {4.0: 0.9, 5.0: 0.7, 5.5: 0.6, 6.0: 0.2, 7.0: 0.3, 8.0: 0.4}

    def fake_timed_frame(profile, events, scatterers, support_sigmas):
        calls.append((len(scatterers), support_sigmas))
        frame = np.array([1.0 if support_sigmas is None else 1.0 + error_by_k[support_sigmas]])
        seconds = 0.01 * len(scatterers) if support_sigmas is None else seconds_by_k[support_sigmas]
        load = {"load_average": "0.10, 0.20, 0.30", "uptime_output": "uptime load average: 0.10, 0.20, 0.30"}
        return frame, seconds, {"before": load, "after": load}

    monkeypatch.setattr(sweep, "_timed_frame", fake_timed_frame)
    output_dir = tmp_path / "issue165" / "2026-11-02"
    json_path, readme_path = sweep.run_sweep(output_dir, seed=27)

    assert json_path.exists()
    assert readme_path.exists()
    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["environment"]["commit"] == commit
    assert record["environment"]["cpu_model"] == "Test CPU model"
    assert record["scatterers"]["seed"] == 27
    assert record["scatterers"]["counts"] == [1, 3, 200]
    assert record["recommended_default_sigmas"] == 6.0
    assert record["sweep_summary"][0]["k"] is None
    assert record["sweep_summary"][4]["seconds"] < record["sweep_summary"][2]["seconds"]
    random_case = record["cases"][-1]
    assert random_case["scatterer_count"] == 200
    assert random_case["rows"][0]["seconds_per_scatterer"] == pytest.approx(0.01)
    assert random_case["rows"][0]["uptime_load_average"]["before"]["load_average"].startswith(
        "0.10"
    )
    assert random_case["rows"][1]["uptime_load_average"][0]["after"]["load_average"].startswith(
        "0.10"
    )
    readme = readme_path.read_text(encoding="utf-8")
    assert "Selected default: **6.0σ**" in readme
    assert "0.010000 s/scatterer" in readme
    call_count = len(calls)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sweep.run_sweep(output_dir, seed=27)
    assert len(calls) == call_count


def test_sweep_default_selection_picks_the_fastest_passing_k():
    from enodia.spec.sim.pulse_support_sweep import select_default_sigmas

    candidates = [
        {"k": 4.0, "max_relative_error": 2e-5, "seconds": 1.0},
        {"k": 5.0, "max_relative_error": 8e-7, "seconds": 1.2},
        {"k": 6.0, "max_relative_error": 1e-8, "seconds": 1.4},
    ]
    assert select_default_sigmas(candidates, max_relative_error=1e-6) == 5.0
    faster_wide_candidate = [
        {"k": 5.0, "max_relative_error": 8e-7, "seconds": 1.2},
        {"k": 6.0, "max_relative_error": 1e-8, "seconds": 0.9},
    ]
    assert select_default_sigmas(faster_wide_candidate, max_relative_error=1e-6) == 6.0
    with pytest.raises(ValueError, match="no pulse support candidate"):
        select_default_sigmas(candidates[:1], max_relative_error=1e-6)
