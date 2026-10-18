"""Board-free revision2 host-duration load execution tests."""

from __future__ import annotations

import pytest

from enodia.tt.bench import load_generators, newton_schulz_kernel
from enodia.tt.bench.duration_load import (
    DEFAULT_OVERRUN_LIMIT_SECONDS,
    LONG_GUARD_SECONDS,
    LONG_TARGET_SECONDS,
    SMOKE_GUARD_SECONDS,
    SMOKE_TARGET_SECONDS,
    DurationLoadError,
    run_duration_bounded_load,
    validate_duration_bounds,
)
from enodia.tt.bench.load_harness import build_newton_schulz_core_plan


class FakeClock:
    def __init__(self, *values: int):
        self.values = list(values)
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        if not self.values:
            raise AssertionError("clock was called more often than expected")
        return self.values.pop(0)


@pytest.mark.parametrize(
    ("target", "guard"),
    ((SMOKE_TARGET_SECONDS, SMOKE_GUARD_SECONDS), (LONG_TARGET_SECONDS, LONG_GUARD_SECONDS)),
)
def test_issue137_duration_bounds_are_the_preregistered_smoke_and_long_values(
    target: float, guard: float
):
    bounds = validate_duration_bounds(target_seconds=target, guard_seconds=guard)

    assert bounds.target_seconds == target
    assert bounds.guard_seconds == guard
    assert bounds.overrun_limit_seconds == DEFAULT_OVERRUN_LIMIT_SECONDS


def test_issue137_exact_boundary_stops_without_an_additional_launch():
    clock = FakeClock(0, 5_000_000_000, 5_000_000_000, 10_000_000_000)
    launches: list[int] = []

    result = run_duration_bounded_load(
        target_seconds=10,
        launch_callback=lambda: launches.append(len(launches)),
        monotonic_ns=clock,
    )

    assert result.status == "passed"
    assert result.launch_count == 2
    assert launches == [0, 1]
    assert clock.calls == 4
    assert result.first_launch_start_monotonic_ns == 0
    assert result.last_launch_end_monotonic_ns == 10_000_000_000
    assert result.workload_end_monotonic_ns == 10_000_000_000
    assert result.overrun_seconds == 0


def test_issue137_duration_loop_counts_successful_multi_launches_only():
    clock = FakeClock(
        100,
        2_000_000_100,
        2_000_000_100,
        4_000_000_100,
        4_000_000_100,
        7_000_000_100,
    )
    launches: list[int] = []

    result = run_duration_bounded_load(
        target_seconds=5,
        launch_callback=lambda: launches.append(len(launches)),
        monotonic_ns=clock,
    )

    assert result.launch_count == 3
    assert len(launches) == result.launch_count
    assert result.first_launch_start_monotonic_ns == 100
    assert result.last_launch_end_monotonic_ns == 7_000_000_100
    assert result.workload_end_monotonic_ns == 5_000_000_100
    assert result.overrun_seconds == pytest.approx(2.0)
    assert result.as_record()["workload_interval"] == {
        "clock": "host_monotonic_ns",
        "start_ns": 100,
        "target_end_ns": 5_000_000_100,
        "actual_end_ns": 7_000_000_100,
    }


def test_issue137_overrun_at_limit_passes_and_above_limit_fails_without_extension():
    passing_clock = FakeClock(0, 20_000_000_000)
    passing = run_duration_bounded_load(
        target_seconds=10,
        launch_callback=lambda: None,
        monotonic_ns=passing_clock,
    )
    assert passing.status == "passed"
    assert passing.overrun_seconds == pytest.approx(10.0)
    assert passing.launch_count == 1

    failing_clock = FakeClock(0, 20_000_000_001)
    launches: list[int] = []
    failing = run_duration_bounded_load(
        target_seconds=10,
        launch_callback=lambda: launches.append(1),
        monotonic_ns=failing_clock,
    )
    assert failing.status == "failed"
    assert failing.failure_reason == "launch overrun exceeds the configured limit"
    assert failing.overrun_seconds > DEFAULT_OVERRUN_LIMIT_SECONDS
    assert failing.launch_count == 1
    assert len(launches) == 1
    assert failing_clock.calls == 2


def test_issue137_callback_error_is_not_retried_and_success_count_excludes_failure():
    clock = FakeClock(0, 1_000_000_000, 1_000_000_000)
    launches = 0

    def launch() -> None:
        nonlocal launches
        launches += 1
        if launches == 2:
            raise RuntimeError("mock launch failed")

    with pytest.raises(DurationLoadError, match="mock launch failed") as raised:
        run_duration_bounded_load(
            target_seconds=5,
            launch_callback=launch,
            monotonic_ns=clock,
        )

    assert launches == 2
    assert raised.value.result is not None
    assert raised.value.result.launch_count == 1
    assert raised.value.result.status == "failed"


def test_issue137_incomplete_callback_is_rejected_without_counting_a_launch():
    clock = FakeClock(0)

    with pytest.raises(DurationLoadError, match="completion") as raised:
        run_duration_bounded_load(
            target_seconds=1,
            launch_callback=lambda: False,
            monotonic_ns=clock,
        )

    assert raised.value.result is not None
    assert raised.value.result.launch_count == 0
    assert raised.value.result.last_launch_end_monotonic_ns is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_seconds": 0}, "target_seconds"),
        ({"target_seconds": float("nan")}, "target_seconds"),
        ({"target_seconds": float("inf")}, "target_seconds"),
        ({"target_seconds": 1, "overrun_limit_seconds": -1}, "overrun"),
    ),
)
def test_issue137_duration_loop_rejects_invalid_numeric_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_duration_bounded_load(
            launch_callback=lambda: None,
            monotonic_ns=lambda: 0,
            **kwargs,
        )


def test_issue137_duration_loop_fails_uncovered_interval_without_starting_an_extra_launch():
    clock = FakeClock(0, 4_000_000_000, 10_000_000_000)
    launches: list[int] = []

    result = run_duration_bounded_load(
        target_seconds=10,
        launch_callback=lambda: launches.append(len(launches)),
        monotonic_ns=clock,
    )

    assert result.status == "failed"
    assert result.failure_reason == "workload interval was uncovered before the next launch boundary"
    assert result.launch_count == 1
    assert launches == [0]
    assert result.last_launch_end_monotonic_ns == 4_000_000_000
    assert result.workload_end_monotonic_ns == 10_000_000_000
    assert result.overrun_seconds == pytest.approx(-6.0)


def test_issue137_duration_loop_rejects_clock_regression():
    clock = FakeClock(10, 9)

    with pytest.raises(DurationLoadError, match="monotonic"):
        run_duration_bounded_load(
            target_seconds=1,
            launch_callback=lambda: None,
            monotonic_ns=clock,
        )


def test_issue137_duration_helper_rejects_non_issue137_matrix_block():
    plan = build_newton_schulz_core_plan(core_count=18, matrix_block=4)

    with pytest.raises(ValueError, match="matrix_block=8"):
        load_generators.prepare_newton_schulz_duration_load(
            object(), object(), core_plan=plan, profile=False
        )


def test_issue137_duration_helper_uses_one_outer_repetition_without_changing_q2_default(
    monkeypatch,
):
    plan = build_newton_schulz_core_plan(core_count=18)
    captured: list[dict[str, object]] = []

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, matrices, **kwargs):
            captured.append({"batch": matrices.shape[0], **kwargs})
            return object()

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", FakeKernel)
    load_generators.prepare_newton_schulz_load(
        object(), object(), core_plan=plan, profile=False
    )
    load_generators.prepare_newton_schulz_duration_load(
        object(), object(), core_plan=plan, profile=False
    )

    assert captured[0]["outer_repetitions"] == 3
    assert captured[1]["outer_repetitions"] == 1
    assert captured[1]["iterations"] == 12
    assert captured[1]["matrix_block"] == 8
    assert captured[1]["core_coordinates"] == plan.worker_cores


def test_issue137_preparation_happens_before_the_first_timed_clock_read(monkeypatch):
    events: list[str] = []

    class FakeKernel:
        @classmethod
        def prepare(cls, ttnn, device, matrices, **kwargs):
            events.append("prepare")
            return object()

    monkeypatch.setattr(newton_schulz_kernel, "NewtonSchulzKernel", FakeKernel)
    plan = build_newton_schulz_core_plan(core_count=18)
    prepared = load_generators.prepare_newton_schulz_duration_load(
        object(), object(), core_plan=plan, profile=False
    )
    assert prepared is not None

    clock = FakeClock(0, 1_000_000_000)

    def timed_clock() -> int:
        assert events == ["prepare"] or events == ["prepare", "clock", "launch"]
        events.append("clock")
        return clock()

    def launch_once() -> None:
        assert events[-1] == "clock"
        events.append("launch")

    result = run_duration_bounded_load(
        target_seconds=1,
        launch_callback=launch_once,
        monotonic_ns=timed_clock,
    )

    assert result.status == "passed"
    assert events == ["prepare", "clock", "launch", "clock"]
