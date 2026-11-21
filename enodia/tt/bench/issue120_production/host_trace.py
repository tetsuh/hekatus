"""Versioned host submissions: diagnostic observations, never card coverage."""
from __future__ import annotations

from .records import integer, mapping

NATIVE_SCHEMA = "issue120-production-native-v2"
TRACE_SCHEMA = "issue120-host-submission-trace-v1"
PERIOD = 65536
CAPACITY = 65536


def _layout(trace: dict, loaded: bool) -> None:
    expected = {"schema": TRACE_SCHEMA, "clock": "CLOCK_MONOTONIC", "period_writes": PERIOD,
                "capacity": CAPACITY, "sample_storage_bytes": CAPACITY * 16}
    if any(trace.get(key) != value or type(trace[key]) is not type(value) for key, value in expected.items()):
        raise ValueError("host trace contract differs")
    if type(trace.get("applicable")) is not bool or trace["applicable"] != loaded:
        raise ValueError("host trace condition differs")


def _samples(trace: dict, count: int, start: int, end: int) -> None:
    rows = trace.get("samples")
    expected = [1, *range(PERIOD, count + 1, PERIOD)] if count else []
    if not isinstance(rows, list) or len(rows) != len(expected) or len(rows) > CAPACITY:
        raise ValueError("host trace sample count differs")
    previous = start
    for row, number in zip(rows, expected, strict=True):
        row = mapping(row, "host sample")
        if integer(row.get("committed_count"), "sample count", 1, 2**32 - 1) != number:
            raise ValueError("host trace cadence differs")
        ns = integer(row.get("monotonic_ns"), "sample timestamp")
        if not previous <= ns <= end:
            raise ValueError("host trace sample clock differs")
        previous = ns
    first = rows[0]["monotonic_ns"] if rows else 0
    if integer(trace.get("first_committed_enqueue_observed_ns"), "first host observation") != first:
        raise ValueError("first host observation differs")


def _snapshot(trace: dict, scheduling: dict, count: int, joined: int) -> None:
    value = mapping(trace.get("main_enqueue_return_observation"), "main host observation")
    before = integer(value.get("before_ns"), "main before")
    after = integer(value.get("after_ns"), "main after")
    integer(value.get("committed_count"), "main count", 0, count)
    if before != scheduling.get("enqueue_return_ns") or not before <= after <= joined:
        raise ValueError("main count observation is outside its bracket")


def validate_native(native: dict, *, require_new: bool = False) -> None:
    if native.get("schema") != NATIVE_SCHEMA:
        if require_new or "host_phase_trace" in native:
            raise ValueError("new host trace requires its native version")
        return  # Historical records retain their existing validator/interpretation.
    trace = mapping(native.get("host_phase_trace"), "host trace")
    condition = native.get("condition")
    if condition not in ("control", "loaded"):
        raise ValueError("host trace has unknown condition")
    loaded = condition == "loaded"
    _layout(trace, loaded)
    cpu = mapping(native.get("host_cpu_work"), "host CPU work")
    scheduling = mapping(native.get("host_scheduling"), "host scheduling")
    count = integer(native.get("repeat_count"), "host repeats", 1, 2**32 - 1) if loaded else 0
    start = integer(trace.get("worker_start_ns"), "worker start")
    end = integer(trace.get("worker_end_ns"), "worker end")
    joined = integer(trace.get("join_ns"), "worker join")
    if not start <= end <= joined or joined != scheduling.get("load_join_completed_ns"):
        raise ValueError("host trace lifecycle differs")
    expected_bounds = (cpu.get("start_ns"), cpu.get("end_ns")) if loaded else (0, 0)
    if (start, end) != expected_bounds or integer(trace.get("final_count"), "final host count", 0, 2**32 - 1) != count:
        raise ValueError("host trace work differs")
    if cpu.get("transfers") != count:
        raise ValueError("host trace final prefix differs from successful submissions")
    _snapshot(trace, scheduling, count, joined)
    _samples(trace, count, start, end)
