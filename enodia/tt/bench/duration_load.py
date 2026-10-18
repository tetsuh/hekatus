"""Board-free host-duration execution for Issue #137 revision 2.

The caller prepares a fixed Newton-Schulz launch before entering this
module. This generic helper does not perform JIT compilation. The callback
supplied to :func:`run_duration_bounded_load` must block until that one launch
has completed; the loop only observes host
``CLOCK_MONOTONIC`` and never supplies a repeat count.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

LONG_TARGET_SECONDS = 120.0
LONG_GUARD_SECONDS = 170.0
SMOKE_TARGET_SECONDS = 20.0
SMOKE_GUARD_SECONDS = 70.0
DEFAULT_OVERRUN_LIMIT_SECONDS = 10.0
RUNNER_MARGIN_SECONDS = 40.0
MAX_TARGET_SECONDS = LONG_TARGET_SECONDS
_NANOSECONDS_PER_SECOND = 1_000_000_000
_CALLBACK_EXCEPTION_TYPE = Exception


class DurationLoadError(RuntimeError):
    """A duration load could not complete one of its required boundaries."""

    def __init__(self, message: str, *, result: DurationLoadResult | None = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class DurationLoadBounds:
    """Validated target, guard, and launch-overrun bounds."""

    target_seconds: float
    guard_seconds: float
    overrun_limit_seconds: float = DEFAULT_OVERRUN_LIMIT_SECONDS

    def as_record(self) -> dict[str, float]:
        return {
            "target_seconds": self.target_seconds,
            "guard_seconds": self.guard_seconds,
            "overrun_limit_seconds": self.overrun_limit_seconds,
        }


@dataclass(frozen=True)
class DurationLoadResult:
    """Observed host interval and launch-boundary facts for one duration run."""

    status: str
    target_seconds: float
    overrun_limit_seconds: float
    launch_count: int
    first_launch_start_monotonic_ns: int | None
    workload_end_monotonic_ns: int | None
    last_launch_end_monotonic_ns: int | None
    overrun_seconds: float | None
    failure_reason: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether the launch boundary and overrun checks passed."""
        return self.status == "passed"

    @property
    def successful_launches(self) -> int:
        """Return the observed count of completed callback invocations."""
        return self.launch_count

    def as_record(self) -> dict[str, Any]:
        interval = None
        if self.first_launch_start_monotonic_ns is not None:
            interval = {
                "clock": "host_monotonic_ns",
                "start_ns": self.first_launch_start_monotonic_ns,
                "target_end_ns": self.workload_end_monotonic_ns,
                "actual_end_ns": self.last_launch_end_monotonic_ns,
            }
        return {
            "status": self.status,
            "target_seconds": self.target_seconds,
            "overrun_limit_seconds": self.overrun_limit_seconds,
            "launch_count": self.launch_count,
            "first_launch_start_monotonic_ns": self.first_launch_start_monotonic_ns,
            "workload_end_monotonic_ns": self.workload_end_monotonic_ns,
            "last_launch_end_monotonic_ns": self.last_launch_end_monotonic_ns,
            "overrun_seconds": self.overrun_seconds,
            "failure_reason": self.failure_reason,
            "workload_interval": interval,
        }


def _finite_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (converted < 0.0 if allow_zero else converted <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return converted


def _validated_target_seconds(value: Any) -> float:
    target = _finite_number(value, "target_seconds")
    if target > MAX_TARGET_SECONDS:
        raise ValueError(f"target_seconds must be <= {MAX_TARGET_SECONDS:g} seconds")
    return target


def _validated_overrun_seconds(value: Any) -> float:
    overrun = _finite_number(value, "overrun_limit_seconds", allow_zero=True)
    if overrun > DEFAULT_OVERRUN_LIMIT_SECONDS:
        raise ValueError(
            "overrun_limit_seconds must be <= "
            f"{DEFAULT_OVERRUN_LIMIT_SECONDS:g} seconds"
        )
    return overrun


def validate_duration_bounds(
    *,
    target_seconds: Any,
    guard_seconds: Any,
    overrun_limit_seconds: Any = DEFAULT_OVERRUN_LIMIT_SECONDS,
) -> DurationLoadBounds:
    """Validate a plan target and its fixed runner guard without opening a device."""
    target = _validated_target_seconds(target_seconds)
    guard = _finite_number(guard_seconds, "guard_seconds")
    overrun = _validated_overrun_seconds(overrun_limit_seconds)
    required_guard = target + RUNNER_MARGIN_SECONDS + overrun
    if guard < required_guard:
        raise ValueError(
            "guard_seconds must cover target, runner margin, and launch overrun: "
            f"at least {required_guard:g} seconds"
        )
    return DurationLoadBounds(
        target_seconds=target,
        guard_seconds=guard,
        overrun_limit_seconds=overrun,
    )


def _clock_value(clock: Callable[[], int], *, previous: int | None, name: str) -> int:
    try:
        value = clock()
    except Exception as exc:
        raise DurationLoadError(f"{name} monotonic clock failed: {exc}") from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DurationLoadError(f"{name} monotonic clock must return a non-negative integer ns")
    if previous is not None and value < previous:
        raise DurationLoadError(f"{name} monotonic clock moved backwards")
    return value


def _partial_result(
    *,
    target_seconds: float,
    overrun_limit_seconds: float,
    first_start_ns: int | None,
    launch_count: int,
    last_end_ns: int | None,
    failure_reason: str,
) -> DurationLoadResult:
    target_end_ns = (
        first_start_ns + round(target_seconds * _NANOSECONDS_PER_SECOND)
        if first_start_ns is not None
        else None
    )
    return DurationLoadResult(
        status="failed",
        target_seconds=target_seconds,
        overrun_limit_seconds=overrun_limit_seconds,
        launch_count=launch_count,
        first_launch_start_monotonic_ns=first_start_ns,
        workload_end_monotonic_ns=target_end_ns,
        last_launch_end_monotonic_ns=last_end_ns,
        overrun_seconds=(
            (last_end_ns - target_end_ns) / _NANOSECONDS_PER_SECOND
            if last_end_ns is not None and target_end_ns is not None
            else None
        ),
        failure_reason=failure_reason,
    )


def _raise_duration_error(
    *,
    target_seconds: float,
    overrun_limit_seconds: float,
    first_start_ns: int | None,
    launch_count: int,
    last_end_ns: int | None,
    reason: str,
    cause: BaseException | None = None,
) -> NoReturn:
    result = _partial_result(
        target_seconds=target_seconds,
        overrun_limit_seconds=overrun_limit_seconds,
        first_start_ns=first_start_ns,
        launch_count=launch_count,
        last_end_ns=last_end_ns,
        failure_reason=reason,
    )
    if cause is None:
        raise DurationLoadError(reason, result=result)
    raise DurationLoadError(reason, result=result) from cause


def _initial_start_or_raise(
    monotonic_ns: Callable[[], int], *, target_seconds: float, overrun_limit_seconds: float
) -> int:
    try:
        return _clock_value(monotonic_ns, previous=None, name="first launch start")
    except DurationLoadError as exc:
        _raise_duration_error(
            target_seconds=target_seconds,
            overrun_limit_seconds=overrun_limit_seconds,
            first_start_ns=None,
            launch_count=0,
            last_end_ns=None,
            reason=str(exc),
            cause=exc,
        )


def _launch_once_or_raise(
    launch_callback: Callable[[], Any],
    *,
    target_seconds: float,
    overrun_limit_seconds: float,
    first_start_ns: int,
    launch_count: int,
    last_end_ns: int | None,
) -> None:
    try:
        completed = launch_callback()
    except _CALLBACK_EXCEPTION_TYPE as exc:
        _raise_duration_error(
            target_seconds=target_seconds,
            overrun_limit_seconds=overrun_limit_seconds,
            first_start_ns=first_start_ns,
            launch_count=launch_count,
            last_end_ns=last_end_ns,
            reason=f"launch callback failed: {exc}",
            cause=exc,
        )
    if completed is not None and completed is not True:
        _raise_duration_error(
            target_seconds=target_seconds,
            overrun_limit_seconds=overrun_limit_seconds,
            first_start_ns=first_start_ns,
            launch_count=launch_count,
            last_end_ns=last_end_ns,
            reason="launch callback did not report completion",
        )


def _clock_or_raise(
    monotonic_ns: Callable[[], int],
    *,
    previous_clock_ns: int,
    name: str,
    target_seconds: float,
    overrun_limit_seconds: float,
    first_start_ns: int,
    launch_count: int,
    last_end_ns: int | None,
) -> int:
    try:
        return _clock_value(monotonic_ns, previous=previous_clock_ns, name=name)
    except DurationLoadError as exc:
        _raise_duration_error(
            target_seconds=target_seconds,
            overrun_limit_seconds=overrun_limit_seconds,
            first_start_ns=first_start_ns,
            launch_count=launch_count,
            last_end_ns=last_end_ns,
            reason=str(exc),
            cause=exc,
        )


def _completed_result(
    *,
    target_seconds: float,
    overrun_limit_seconds: float,
    first_start_ns: int,
    workload_end_ns: int,
    launch_end_ns: int,
    launch_count: int,
) -> DurationLoadResult:
    overrun_seconds = (launch_end_ns - workload_end_ns) / _NANOSECONDS_PER_SECOND
    passed = overrun_seconds <= overrun_limit_seconds
    return DurationLoadResult(
        status="passed" if passed else "failed",
        target_seconds=target_seconds,
        overrun_limit_seconds=overrun_limit_seconds,
        launch_count=launch_count,
        first_launch_start_monotonic_ns=first_start_ns,
        workload_end_monotonic_ns=workload_end_ns,
        last_launch_end_monotonic_ns=launch_end_ns,
        overrun_seconds=overrun_seconds,
        failure_reason=(None if passed else "launch overrun exceeds the configured limit"),
    )


def run_duration_bounded_load(
    launch_callback: Callable[[], Any] | None = None,
    *,
    target_seconds: Any,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    overrun_limit_seconds: Any = DEFAULT_OVERRUN_LIMIT_SECONDS,
) -> DurationLoadResult:
    """Run blocking one-launch callbacks until the target boundary is covered.

    ``launch_callback`` is invoked back-to-back and must not merely enqueue
    asynchronous work.  Returning ``None`` or ``True`` means the callback's
    blocking contract completed; any other return is an incomplete callback.
    The first clock read is immediately before the first callback, so launch
    preparation performed by the caller is outside the workload interval.
    """
    target = _validated_target_seconds(target_seconds)
    overrun_limit = _validated_overrun_seconds(overrun_limit_seconds)
    if not callable(launch_callback):
        raise TypeError("launch_callback must be callable")
    if not callable(monotonic_ns):
        raise TypeError("monotonic_ns must be callable")
    target_ns = round(target * _NANOSECONDS_PER_SECOND)
    if target_ns < 1:
        raise ValueError("target_seconds is below one nanosecond")

    first_start_ns = _initial_start_or_raise(
        monotonic_ns,
        target_seconds=target,
        overrun_limit_seconds=overrun_limit,
    )
    previous_clock_ns = first_start_ns
    last_end_ns: int | None = None
    launch_count = 0
    workload_end_ns = first_start_ns + target_ns
    while True:
        _launch_once_or_raise(
            launch_callback,
            target_seconds=target,
            overrun_limit_seconds=overrun_limit,
            first_start_ns=first_start_ns,
            launch_count=launch_count,
            last_end_ns=last_end_ns,
        )
        launch_end_ns = _clock_or_raise(
            monotonic_ns,
            previous_clock_ns=previous_clock_ns,
            name="launch completion",
            target_seconds=target,
            overrun_limit_seconds=overrun_limit,
            first_start_ns=first_start_ns,
            launch_count=launch_count,
            last_end_ns=last_end_ns,
        )
        previous_clock_ns = launch_end_ns
        last_end_ns = launch_end_ns
        launch_count += 1
        if launch_end_ns >= workload_end_ns:
            result = _completed_result(
                target_seconds=target,
                overrun_limit_seconds=overrun_limit,
                first_start_ns=first_start_ns,
                workload_end_ns=workload_end_ns,
                launch_end_ns=launch_end_ns,
                launch_count=launch_count,
            )
            return result

        next_launch_check_ns = _clock_or_raise(
            monotonic_ns,
            previous_clock_ns=previous_clock_ns,
            name="next launch check",
            target_seconds=target,
            overrun_limit_seconds=overrun_limit,
            first_start_ns=first_start_ns,
            launch_count=launch_count,
            last_end_ns=last_end_ns,
        )
        previous_clock_ns = next_launch_check_ns
        if next_launch_check_ns >= workload_end_ns:
            return _partial_result(
                target_seconds=target,
                overrun_limit_seconds=overrun_limit,
                first_start_ns=first_start_ns,
                launch_count=launch_count,
                last_end_ns=last_end_ns,
                failure_reason=(
                    "workload interval was uncovered before the next launch boundary"
                ),
            )


# The explicit name is useful to callers that already use "host duration" in
# their driver vocabulary while preserving one implementation and one contract.
run_host_duration_load = run_duration_bounded_load


__all__ = [
    "DEFAULT_OVERRUN_LIMIT_SECONDS",
    "LONG_GUARD_SECONDS",
    "LONG_TARGET_SECONDS",
    "RUNNER_MARGIN_SECONDS",
    "SMOKE_GUARD_SECONDS",
    "SMOKE_TARGET_SECONDS",
    "DurationLoadBounds",
    "DurationLoadError",
    "DurationLoadResult",
    "run_duration_bounded_load",
    "run_host_duration_load",
    "validate_duration_bounds",
]
