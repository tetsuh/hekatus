"""Host-side accounting and record helpers for Issue #12 Stage 1.

The device runner is intentionally kept separate from these helpers.  This
module has no TTNN import, so ring accounting, timestamp arithmetic, and the
measurement contract remain testable on a development machine.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import PurePath
from typing import Any

PAGE_WORDS = 32 * 32
PAGE_BYTES = PAGE_WORDS * 4
MAX_OUTER_TIMEOUT_SECONDS = 600
MIN_RING_PAGES = 2
MAX_RING_L1_BYTES = 900 * 1024
PERCENTILES = {
    "p50": 0.50,
    "p99": 0.99,
    "p99_9": 0.999,
    "p99_99": 0.9999,
}


@dataclass(frozen=True)
class ResidentConfig:
    """Explicit configuration shared by the host record and device kernels."""

    frame_count: int
    frame_interval_ticks: int
    producer_core: tuple[int, int]
    consumer_core: tuple[int, int]
    ring_pages: int
    work_per_frame: int
    designated_timestamp_core: tuple[int, int]
    cycle_budget: int
    outer_timeout_seconds: int
    histogram_bin_ticks: int = 1

    def as_record(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("producer_core", "consumer_core", "designated_timestamp_core"):
            values[name] = list(values[name])
        return values


class RingAccounting:
    """Reference model for one producer and one consumer ring.

    ``reserve_producer`` models the producer's full-ring wait and ``publish``
    models the pointer/semaphore update after the payload is visible.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < MIN_RING_PAGES:
            raise ValueError(f"capacity must be an integer >= {MIN_RING_PAGES}")
        self.capacity = capacity
        self.produced = 0
        self.consumed = 0
        self._reserved = 0
        self.producer_full_count = 0
        self.consumer_empty_count = 0

    def reserve_producer(self) -> int | None:
        """Reserve the next slot, or count and report a full-ring event."""
        if self.produced + self._reserved - self.consumed >= self.capacity:
            self.producer_full_count += 1
            return None
        slot = (self.produced + self._reserved) % self.capacity
        self._reserved += 1
        return slot

    def publish(self) -> None:
        if self._reserved < 1:
            raise RuntimeError("publish without a producer reservation")
        self._reserved -= 1
        self.produced += 1

    def consume(self) -> int | None:
        """Consume the next published slot, or count an empty-ring event."""
        if self.consumed >= self.produced:
            self.consumer_empty_count += 1
            return None
        slot = self.consumed % self.capacity
        self.consumed += 1
        return slot


class ResidentPreflightError(ValueError):
    """A configuration was rejected before device allocation or execution."""


def _core(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 0 for item in value)
    ):
        raise ResidentPreflightError(f"{name} must be a non-negative (x, y) pair")
    return int(value[0]), int(value[1])


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResidentPreflightError(f"{name} must be a positive integer")
    return int(value)


def validate_configuration(config: ResidentConfig) -> ResidentConfig:
    """Reject unsafe or unbounded configurations before device work."""
    if not isinstance(config, ResidentConfig):
        raise TypeError("config must be ResidentConfig")
    frame_count = _positive_int(config.frame_count, "frame_count")
    interval = _positive_int(config.frame_interval_ticks, "frame_interval_ticks")
    ring_pages = _positive_int(config.ring_pages, "ring_pages")
    work = _positive_int(config.work_per_frame, "work_per_frame")
    budget = _positive_int(config.cycle_budget, "cycle_budget")
    timeout = _positive_int(config.outer_timeout_seconds, "outer_timeout_seconds")
    bin_width = _positive_int(config.histogram_bin_ticks, "histogram_bin_ticks")
    if ring_pages < MIN_RING_PAGES:
        raise ResidentPreflightError(f"ring_pages must be >= {MIN_RING_PAGES}")
    if timeout > MAX_OUTER_TIMEOUT_SECONDS:
        raise ResidentPreflightError(
            f"outer_timeout_seconds={timeout} exceeds the {MAX_OUTER_TIMEOUT_SECONDS}s Stage 1 cap"
        )
    producer = _core(config.producer_core, "producer_core")
    consumer = _core(config.consumer_core, "consumer_core")
    designated = _core(config.designated_timestamp_core, "designated_timestamp_core")
    if producer == consumer:
        raise ResidentPreflightError("producer and consumer must run on different cores")
    if designated != consumer:
        raise ResidentPreflightError(
            "designated_timestamp_core must equal consumer_core; all intervals use one clock"
        )
    if ring_pages * PAGE_BYTES + 2 * PAGE_BYTES > MAX_RING_L1_BYTES:
        raise ResidentPreflightError(
            f"ring_pages={ring_pages} exceeds the L1 preflight budget "
            f"({MAX_RING_L1_BYTES} bytes including scratch pages)"
        )
    return ResidentConfig(
        frame_count=frame_count,
        frame_interval_ticks=interval,
        producer_core=producer,
        consumer_core=consumer,
        ring_pages=ring_pages,
        work_per_frame=work,
        designated_timestamp_core=designated,
        cycle_budget=budget,
        outer_timeout_seconds=timeout,
        histogram_bin_ticks=bin_width,
    )


def wrap_delta(end: int, start: int, *, bits: int = 64) -> int:
    """Return an unsigned modular counter delta for a 32- or 64-bit clock."""
    if bits not in (32, 64):
        raise ValueError("bits must be 32 or 64")
    limit = 1 << bits
    if (
        isinstance(end, bool)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or not isinstance(start, int)
        or not 0 <= end < limit
        or not 0 <= start < limit
    ):
        raise ValueError(f"timestamps must be unsigned {bits}-bit integers")
    return (end - start) & (limit - 1)


def cycle_budget_exceeded(*, elapsed_ticks: int, cycle_budget: int) -> bool:
    """The in-kernel budget is inclusive: reaching it is an error."""
    if elapsed_ticks < 0 or cycle_budget <= 0:
        raise ValueError("elapsed_ticks must be non-negative and cycle_budget positive")
    return elapsed_ticks >= cycle_budget


def termination_reason(
    *, frame_count_reached: bool, cycle_budget_hit: bool, outer_timeout: bool
) -> str:
    """Classify the first terminal condition, with host timeout taking priority."""
    if outer_timeout:
        return "outer_timeout"
    if cycle_budget_hit:
        return "cycle_budget"
    if frame_count_reached:
        return "frame_count"
    return "running"


def required_samples_for_percentile(percentile: float) -> int:
    """Return the exact Issue #12 sufficiency threshold ``ceil(20/(1-p))``."""
    try:
        fraction = Fraction(str(percentile))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("percentile must be a finite decimal between 0 and 1") from exc
    if not 0 < fraction < 1:
        raise ValueError("percentile must be strictly between 0 and 1")
    threshold = Fraction(20, 1) / (1 - fraction)
    return (threshold.numerator + threshold.denominator - 1) // threshold.denominator


def _linear_percentile(values: list[int], percentile: float) -> float:
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _fine_histogram(values: list[int], bin_width_ticks: int) -> dict[str, Any]:
    if not values:
        return {"bin_width_ticks": bin_width_ticks, "bins": []}
    counts: dict[int, int] = {}
    for value in values:
        start = (value // bin_width_ticks) * bin_width_ticks
        counts[start] = counts.get(start, 0) + 1
    return {
        "bin_width_ticks": bin_width_ticks,
        "bins": [
            {
                "start_ticks": start,
                "end_ticks": start + bin_width_ticks,
                "count": counts[start],
            }
            for start in sorted(counts)
        ],
    }


def frame_interval_statistics(
    intervals_ticks: Iterable[int], *, bin_width_ticks: int = 1
) -> dict[str, Any]:
    """Compute fine-bin statistics and N-dependent percentiles."""
    bin_width_ticks = _positive_int(bin_width_ticks, "bin_width_ticks")
    values = list(intervals_ticks)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("intervals_ticks must contain non-negative integers")
    ordered = sorted(values)
    percentiles: dict[str, dict[str, Any]] = {}
    for name, percentile in PERCENTILES.items():
        threshold = required_samples_for_percentile(percentile)
        if len(ordered) < threshold:
            percentiles[name] = {
                "status": "insufficient",
                "minimum_samples": threshold,
            }
        else:
            percentiles[name] = {
                "status": "ok",
                "minimum_samples": threshold,
                "value_ticks": _linear_percentile(ordered, percentile),
            }
    return {
        "N": len(ordered),
        "min_ticks": ordered[0] if ordered else None,
        "max_ticks": ordered[-1] if ordered else None,
        "percentiles": percentiles,
        "histogram": _fine_histogram(ordered, bin_width_ticks),
    }


def timestamp_digest(timestamps: Iterable[int]) -> dict[str, Any]:
    """Return count and SHA-256 for raw little-endian 64-bit timestamps only."""
    digest = hashlib.sha256()
    count = 0
    for timestamp in timestamps:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or not 0 <= timestamp < 1 << 64:
            raise ValueError("timestamps must be unsigned 64-bit integers")
        digest.update(struct.pack("<Q", timestamp))
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _require_environment(environment: Mapping[str, Any]) -> None:
    required = ("board", "firmware", "kmd_version", "image")
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ValueError("environment is missing required fields: " + ", ".join(missing))


def _safe_trace_name(power_trace: str) -> str:
    path = PurePath(power_trace)
    if path.is_absolute() or len(path.parts) != 1 or path.name != power_trace:
        raise ValueError("power_trace must be a repository-relative filename")
    return power_trace


def _source_evidence() -> dict[str, Any]:
    return {
        "toolchain": "tt-metal v0.75.0",
        "tt_metal_revision": "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
        "clock_api": "tt_metal/hw/inc/internal/tt-1xx/risc_common.h:245-255",
        "blackhole_read_api": "tt_metal/hw/inc/internal/tt-1xx/blackhole/c_tensix_core.h:503-510",
        "dataflow_ring_api": "tt_metal/hw/inc/api/dataflow/dataflow_api.h:404-485",
        "semaphore_api": "tt_metal/hw/inc/api/dataflow/dataflow_api.h:1514-1525,1934-1992",
        "frequency_api": "tt_metal/api/tt-metalium/device.hpp:86-89",
        "profiler_conversion": "tt_metal/impl/profiler/profiler_analysis.cpp:300-303",
        "cross_core_note": "intervals use only the designated consumer core; cross-core correlation is out of scope",
    }


def build_measurement_record(
    *,
    config: ResidentConfig,
    aiclk_mhz: int,
    timestamps: Iterable[int],
    producer_full_count: int,
    consumer_empty_count: int,
    cycle_budget_hit: bool,
    kernel_error_flag: int,
    harness_commit: str,
    environment: Mapping[str, Any],
    power_trace: str,
    watcher: bool = False,
    timing_evidence: bool = True,
) -> dict[str, Any]:
    """Build the committed-schema record without retaining raw timestamps."""
    config = validate_configuration(config)
    if isinstance(aiclk_mhz, bool) or not isinstance(aiclk_mhz, int) or aiclk_mhz <= 0:
        raise ValueError("aiclk_mhz must be a positive integer")
    if not isinstance(harness_commit, str) or not harness_commit.strip():
        raise ValueError("harness_commit is required")
    _require_environment(environment)
    if isinstance(producer_full_count, bool) or producer_full_count < 0:
        raise ValueError("producer_full_count must be non-negative")
    if isinstance(consumer_empty_count, bool) or consumer_empty_count < 0:
        raise ValueError("consumer_empty_count must be non-negative")
    if kernel_error_flag not in (0, 1, False, True):
        raise ValueError("kernel_error_flag must be a boolean or 0/1")
    timestamp_values = list(timestamps)
    digest = timestamp_digest(timestamp_values)
    intervals = [
        wrap_delta(timestamp_values[index], timestamp_values[index - 1])
        for index in range(1, len(timestamp_values))
    ]
    frame_count_reached = len(timestamp_values) == config.frame_count
    reason = termination_reason(
        frame_count_reached=frame_count_reached,
        cycle_budget_hit=bool(cycle_budget_hit),
        outer_timeout=False,
    )
    if reason == "running":
        reason = "incomplete"
    stats = frame_interval_statistics(intervals, bin_width_ticks=config.histogram_bin_ticks)
    return {
        "schema": "issue-12-stage-1-resident-v1",
        "issue": 12,
        "stage": 1,
        "status": "ok" if not cycle_budget_hit and not kernel_error_flag and frame_count_reached else "error",
        "termination_reason": reason,
        "parameters": {
            **config.as_record(),
            "frame_interval_is_not_acquisition_rate_claim": True,
            "frame_interval_note": (
                "The device-clock frame interval is a harness parameter; Stage 1 does not claim the real acquisition rate."
            ),
        },
        "clock": {
            "name": "RISCV_DEBUG_REG_WALL_CLOCK",
            "timestamp_api": "get_timestamp()",
            "width_bits": 64,
            "interval_unit": "device_clock_ticks",
            "frequency_source": "AICLK",
            "aiclk_mhz": aiclk_mhz,
            "designated_core": list(config.designated_timestamp_core),
            "cross_core_correlation": "out_of_scope",
        },
        "clock_source_evidence": _source_evidence(),
        "histogram": stats,
        "ring": {
            "producer_full_count": int(producer_full_count),
            "consumer_empty_count": int(consumer_empty_count),
            "synchronization": "ring pointers and L1 semaphores only",
        },
        "cycle_budget": {
            "budget_ticks": config.cycle_budget,
            "exceeded": bool(cycle_budget_hit),
            "error_flag": int(bool(kernel_error_flag)),
        },
        "raw_timestamps": digest,
        "power_trace": _safe_trace_name(power_trace),
        "environment": dict(environment),
        "harness_commit": harness_commit,
        "watcher": bool(watcher),
        "timing_evidence": bool(timing_evidence),
    }


def build_rejection_record(
    *, config: ResidentConfig, reason: str, environment: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Record a rejected configuration without running it on a device."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("rejection reason is required")
    result: dict[str, Any] = {
        "schema": "issue-12-stage-1-resident-v1",
        "issue": 12,
        "stage": 1,
        "status": "rejected",
        "rejection_reason": reason,
        "parameters": config.as_record(),
        "frame_interval_is_not_acquisition_rate_claim": True,
        "clock_source_evidence": _source_evidence(),
    }
    if environment:
        result["environment"] = dict(environment)
    return result


__all__ = [
    "PAGE_BYTES",
    "PAGE_WORDS",
    "ResidentConfig",
    "ResidentPreflightError",
    "RingAccounting",
    "build_measurement_record",
    "build_rejection_record",
    "cycle_budget_exceeded",
    "frame_interval_statistics",
    "required_samples_for_percentile",
    "termination_reason",
    "timestamp_digest",
    "validate_configuration",
    "wrap_delta",
]
