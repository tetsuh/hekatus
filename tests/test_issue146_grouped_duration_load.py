"""Board-free grouped duration-load contract tests for Issue #146."""

from __future__ import annotations

import json

import pytest

from enodia.tt.bench.duration_load import (
    GroupedDurationLoadResult,
    run_duration_bounded_load,
    run_grouped_duration_bounded_load,
)


class FakeClock:
    def __init__(self, *values: int):
        self.values = list(values)
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        if not self.values:
            raise AssertionError("clock was called more often than expected")
        return self.values.pop(0)


def test_issue146_grouped_loop_stops_at_the_next_complete_group_boundary():
    clock = FakeClock(0, 3_000_000_000, 3_000_000_000, 6_000_000_000)
    events: list[str] = []

    result = run_grouped_duration_bounded_load(
        lambda: events.append("launch"),
        lambda: events.append("sync"),
        target_seconds=5,
        launches_per_sync=2,
        monotonic_ns=clock,
        idle_window_seconds=(0, 5),
    )

    assert isinstance(result, GroupedDurationLoadResult)
    assert result.status == "passed"
    assert result.group_count == 2
    assert result.launch_count == 4
    assert events == [
        "launch",
        "launch",
        "sync",
        "launch",
        "launch",
        "sync",
    ]
    assert result.first_launch_start_monotonic_ns == 0
    assert result.last_group_end_monotonic_ns == 6_000_000_000
    assert result.workload_end_monotonic_ns == 5_000_000_000
    assert result.overrun_seconds == pytest.approx(1.0)
    assert clock.calls == 4


def test_issue146_grouped_loop_records_group_bounds_and_syncs_once_per_group():
    clock = FakeClock(
        100,
        2_000_000_100,
        2_000_000_500,
        5_000_000_500,
    )
    events: list[str] = []

    result = run_grouped_duration_bounded_load(
        lambda: events.append("launch"),
        lambda: events.append("sync"),
        target_seconds=5,
        launches_per_sync=3,
        monotonic_ns=clock,
        idle_window_seconds=(0, 5),
    )

    assert result.group_count == 2
    assert result.launch_count == 6
    assert result.group_timings == (
        {
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 2_000_000_100,
        },
        {
            "start_monotonic_ns": 2_000_000_500,
            "end_monotonic_ns": 5_000_000_500,
        },
    )
    assert events == [
        "launch",
        "launch",
        "launch",
        "sync",
        "launch",
        "launch",
        "launch",
        "sync",
    ]
    record = result.as_record()
    assert record["group_timings"] == [
        {
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 2_000_000_100,
        },
        {
            "start_monotonic_ns": 2_000_000_500,
            "end_monotonic_ns": 5_000_000_500,
        },
    ]
    json.dumps(record)


def test_issue146_idle_fraction_is_clipped_to_the_requested_window():
    clock = FakeClock(
        0,
        2_000_000_000,
        5_000_000_000,
        9_000_000_000,
        12_000_000_000,
        13_000_000_000,
    )

    result = run_grouped_duration_bounded_load(
        lambda: None,
        lambda: None,
        target_seconds=10,
        launches_per_sync=1,
        monotonic_ns=clock,
        idle_window_seconds=(4, 10),
    )

    assert result.group_count == 3
    assert result.host_visible_idle_fraction == pytest.approx(2 / 6)
    assert result.as_record()["idle_window_seconds"] == {
        "start_seconds": 4.0,
        "end_seconds": 10.0,
    }


def test_issue146_accepts_the_long_window_without_issue137_target_cap():
    result = run_grouped_duration_bounded_load(
        lambda: None,
        lambda: None,
        target_seconds=180,
        launches_per_sync=8,
        monotonic_ns=FakeClock(0, 180_000_000_000),
        idle_window_seconds=(60, 180),
    )

    assert result.status == "passed"
    assert result.group_count == 1
    assert result.group_timings == (
        {
            "start_monotonic_ns": 0,
            "end_monotonic_ns": 180_000_000_000,
        },
    )


def test_issue146_overrun_uses_the_caller_limit():
    passing = run_grouped_duration_bounded_load(
        lambda: None,
        lambda: None,
        target_seconds=5,
        launches_per_sync=1,
        monotonic_ns=FakeClock(0, 6_000_000_000),
        overrun_limit_seconds=1,
        idle_window_seconds=(0, 5),
    )
    assert passing.status == "passed"
    assert passing.overrun_seconds == pytest.approx(1)

    failing = run_grouped_duration_bounded_load(
        lambda: None,
        lambda: None,
        target_seconds=5,
        launches_per_sync=1,
        monotonic_ns=FakeClock(0, 6_000_000_001),
        overrun_limit_seconds=1,
        idle_window_seconds=(0, 5),
    )
    assert failing.status == "failed"
    assert failing.failure_reason == "group overrun exceeds the configured limit"
    assert failing.overrun_seconds > 1


@pytest.mark.parametrize(
    "launches_per_sync",
    (0, -1, True, 1.5),
)
def test_issue146_grouped_loop_rejects_invalid_group_size(launches_per_sync):
    with pytest.raises((TypeError, ValueError), match="launches_per_sync"):
        run_grouped_duration_bounded_load(
            lambda: None,
            lambda: None,
            target_seconds=1,
            launches_per_sync=launches_per_sync,
            monotonic_ns=lambda: 0,
            idle_window_seconds=(0, 1),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_seconds": 0}, "target_seconds"),
        ({"target_seconds": float("nan")}, "target_seconds"),
        ({"overrun_limit_seconds": -1}, "overrun"),
        ({"idle_window_seconds": (-1, 1)}, "idle_window"),
        ({"idle_window_seconds": (1, 1)}, "idle_window"),
        ({"idle_window_seconds": (0, 2)}, "idle_window"),
    ),
)
def test_issue146_grouped_loop_rejects_invalid_parameters(kwargs, message):
    arguments = {
        "target_seconds": 1,
        "launches_per_sync": 1,
        "monotonic_ns": lambda: 0,
        "idle_window_seconds": (0, 1),
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        run_grouped_duration_bounded_load(
            lambda: None,
            lambda: None,
            **arguments,
        )


def test_issue146_grouped_api_does_not_change_issue137_blocking_callback_contract():
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
    assert result.first_launch_start_monotonic_ns == 0
    assert result.last_launch_end_monotonic_ns == 10_000_000_000
    assert result.workload_end_monotonic_ns == 10_000_000_000
