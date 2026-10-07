"""Host-side accounting and record helpers for Issue #12 Stage 1.

The device runner is intentionally kept separate from these helpers.  This
module has no TTNN import, so ring accounting, timestamp arithmetic, and the
measurement contract remain testable on a development machine.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import PurePath
from typing import Any

PAGE_WORDS = 32 * 32
PAGE_BYTES = PAGE_WORDS * 4
MAX_TIMING_TIMEOUT_SECONDS = 600
MAX_WATCHER_TIMEOUT_SECONDS = 60
# Retain the old symbol for callers that imported it; timing runs now use the
# approved 600-second ceiling.
MAX_OUTER_TIMEOUT_SECONDS = MAX_TIMING_TIMEOUT_SECONDS
MIN_RING_PAGES = 2
MAX_RING_L1_BYTES = 900 * 1024
RESIDENT_SEMAPHORE_COUNT = 4
SEMAPHORE_BYTES = 4
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
TIMESTAMP_GAP_LIMIT_TICKS = 1 << 31
CURRENT_WRAP_WORK_PER_FRAME = 64
CURRENT_WRAP_OBSERVED_WORK_MAX_TICKS = 1_087
WORK_TICKS_MARGIN_PERCENT = 10
WORK_TICKS_PER_UNIT_UPPER_BOUND = (
    (CURRENT_WRAP_OBSERVED_WORK_MAX_TICKS + CURRENT_WRAP_WORK_PER_FRAME - 1)
    // CURRENT_WRAP_WORK_PER_FRAME
    * (100 + WORK_TICKS_MARGIN_PERCENT)
    + 99
) // 100
RUN_BUDGET_SAFETY_MARGIN_PERCENT = 10
WATCHER_OVERHEAD_MARGIN_PERCENT = 100
STARTUP_ALLOWANCE_MICROSECONDS = 100_000
FAILURE_CODES = {
    0: "none",
    1: "run_wide_budget",
    2: "producer_pacing_wait",
    3: "consumer_empty_wait",
    4: "consumer_fixed_work_budget",
    5: "other_check",
}
FAILURE_PRIORITY = {
    "consumer_fixed_work_budget": 0,
    "producer_pacing_wait": 1,
    "consumer_empty_wait": 2,
    "run_wide_budget": 3,
    "other_check": 4,
    "none": 5,
}
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
    fixed_work_ticks_per_frame: int = 100_000
    budget_aiclk_mhz: int = 1_350

    def as_record(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("producer_core", "consumer_core", "designated_timestamp_core"):
            values[name] = list(values[name])
        return values


class RingAccounting:
    """Reference model for one producer and one consumer ring.

    ``reserve_producer`` is non-blocking: a full ring drops the new attempt
    and increments the overflow count. ``publish`` models the pointer update
    after the payload is visible.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < MIN_RING_PAGES:
            raise ValueError(f"capacity must be an integer >= {MIN_RING_PAGES}")
        self.capacity = capacity
        self.attempted = 0
        self.produced = 0
        self.consumed = 0
        self.dropped = 0
        self._reserved = 0
        self.producer_full_count = 0
        self.consumer_empty_count = 0

    def reserve_producer(self) -> int | None:
        """Attempt one frame without waiting; full rings drop the new frame."""
        self.attempted += 1
        if self.produced + self._reserved - self.consumed >= self.capacity:
            self.producer_full_count += 1
            self.dropped += 1
            return None
        slot = (self.produced + self._reserved) % self.capacity
        self._reserved += 1
        return slot

    def publish(self) -> None:
        if self._reserved < 1:
            raise RuntimeError("publish without a producer reservation")
        self._reserved -= 1
        self.produced += 1

    def drain_complete(self, producer_done: bool) -> bool:
        """Return true only after producer termination and ring drain."""
        return bool(producer_done and self.consumed == self.produced)

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


def _uint32_positive_int(value: Any, name: str) -> int:
    value = _positive_int(value, name)
    if value > UINT32_MAX:
        raise ResidentPreflightError(f"{name} must fit uint32 runtime argument")
    return value


def resident_l1_allocation_table(*, ring_pages: int, watcher: bool = False) -> list[dict[str, Any]]:
    """Return the authoritative resident L1 allocation ledger.

    Per-core rows: producer has one producer-anchor page, one producer CB page,
    and two semaphore words; consumer has the shared ring storage in its L1,
    one control page, one consumer CB page, and two semaphore words. Shared
    timestamp and producer/consumer stats tensors are DRAM allocations (zero
    L1 bytes). Watcher adds no resident tensor/CB/semaphore allocation.
    Total bytes are the row sum: ``(ring_pages + 4) * PAGE_BYTES + 4 * 4``.
    """
    rows = [
        {
            "scope": "consumer_core",
            "component": "ring_storage",
            "pages": ring_pages,
            "bytes": ring_pages * PAGE_BYTES,
            "placement": "consumer L1",
        },
        {
            "scope": "consumer_core",
            "component": "control_page",
            "pages": 1,
            "bytes": PAGE_BYTES,
            "placement": "consumer L1",
        },
        {
            "scope": "consumer_core",
            "component": "consumer_cb_page",
            "pages": 1,
            "bytes": PAGE_BYTES,
            "placement": "consumer L1",
        },
        {
            "scope": "consumer_core",
            "component": "ready_done_semaphores",
            "pages": 0,
            "bytes": 2 * SEMAPHORE_BYTES,
            "placement": "consumer L1",
        },
        {
            "scope": "producer_core",
            "component": "producer_anchor_page",
            "pages": 1,
            "bytes": PAGE_BYTES,
            "placement": "producer L1",
        },
        {
            "scope": "producer_core",
            "component": "producer_cb_page",
            "pages": 1,
            "bytes": PAGE_BYTES,
            "placement": "producer L1",
        },
        {
            "scope": "producer_core",
            "component": "free_error_semaphores",
            "pages": 0,
            "bytes": 2 * SEMAPHORE_BYTES,
            "placement": "producer L1",
        },
        {
            "scope": "shared",
            "component": "timestamps_DRAM",
            "pages": 0,
            "bytes": 0,
            "placement": "DRAM",
        },
        {
            "scope": "shared",
            "component": "producer_consumer_stats_DRAM",
            "pages": 0,
            "bytes": 0,
            "placement": "DRAM",
        },
        {
            "scope": "shared",
            "component": "watcher_extra_resident_allocation",
            "pages": 0,
            "bytes": 0,
            "placement": "none",
        },
    ]
    if watcher:
        rows[-1]["note"] = "Watcher instrumentation adds no host-accounted resident L1 allocation."
    return rows


def resident_l1_allocation_bytes(*, ring_pages: int, watcher: bool = False) -> int:
    """Return the sum of :func:`resident_l1_allocation_table` bytes."""
    return sum(row["bytes"] for row in resident_l1_allocation_table(ring_pages=ring_pages, watcher=watcher))


def validate_configuration(
    config: ResidentConfig, *, watcher: bool = False
) -> ResidentConfig:
    """Reject unsafe or unbounded configurations before device work.

    The wrap-tracked low-word clock requires every single read gap to remain
    below ``TIMESTAMP_GAP_LIMIT_TICKS`` (2**31 ticks). Producer pacing and
    consumer ready-wait loops poll at each configured frame interval; the
    per-frame budget is also a single-interval bound. The fixed-work loop is
    bounded using the current-wrap record
    ``2026-10-06-p150a-issue12-stage1-current-wrap-500000-adr0005.json``:
    observed work max is 1,087 ticks for 64 units, so
    ``ceil(1087 / 64) * 1.10`` is conservatively rounded to
    ``WORK_TICKS_PER_UNIT_UPPER_BOUND = 19`` ticks/unit. Finalization adds no
    timestamp-read gap after the last bounded work/read checkpoint.
    """
    if not isinstance(config, ResidentConfig):
        raise TypeError("config must be ResidentConfig")
    frame_count = _uint32_positive_int(config.frame_count, "frame_count")
    interval = _uint32_positive_int(config.frame_interval_ticks, "frame_interval_ticks")
    ring_pages = _uint32_positive_int(config.ring_pages, "ring_pages")
    work = _uint32_positive_int(config.work_per_frame, "work_per_frame")
    budget = _uint32_positive_int(config.cycle_budget, "cycle_budget")
    if interval >= TIMESTAMP_GAP_LIMIT_TICKS:
        raise ResidentPreflightError(
            f"frame_interval_ticks must be < {TIMESTAMP_GAP_LIMIT_TICKS} for wrap-tracked timestamp gaps"
        )
    if budget >= TIMESTAMP_GAP_LIMIT_TICKS:
        raise ResidentPreflightError(
            f"cycle_budget must be < {TIMESTAMP_GAP_LIMIT_TICKS} for wrap-tracked timestamp gaps"
        )
    if work * WORK_TICKS_PER_UNIT_UPPER_BOUND >= TIMESTAMP_GAP_LIMIT_TICKS:
        raise ResidentPreflightError(
            "work_per_frame upper bound would reach the 2**31 wrap-tracked timestamp gap"
        )
    fixed_work_ticks = _positive_int(
        config.fixed_work_ticks_per_frame, "fixed_work_ticks_per_frame"
    )
    budget_aiclk_mhz = _positive_int(config.budget_aiclk_mhz, "budget_aiclk_mhz")
    timeout = _positive_int(config.outer_timeout_seconds, "outer_timeout_seconds")
    bin_width = _positive_int(config.histogram_bin_ticks, "histogram_bin_ticks")
    if ring_pages < MIN_RING_PAGES:
        raise ResidentPreflightError(f"ring_pages must be >= {MIN_RING_PAGES}")
    timeout_cap = MAX_WATCHER_TIMEOUT_SECONDS if watcher else MAX_TIMING_TIMEOUT_SECONDS
    if timeout > timeout_cap:
        mode = "Watcher" if watcher else "timing"
        raise ResidentPreflightError(
            f"outer_timeout_seconds={timeout} exceeds the {timeout_cap}s Stage 1 {mode} cap"
        )
    producer = _core(config.producer_core, "producer_core")
    consumer = _core(config.consumer_core, "consumer_core")
    designated = _core(config.designated_timestamp_core, "designated_timestamp_core")
    if fixed_work_ticks * 2 > interval:
        raise ResidentPreflightError(
            "fixed_work_ticks_per_frame must be <= half the frame interval in device ticks"
        )
    if producer == consumer:
        raise ResidentPreflightError("producer and consumer must run on different cores")
    if designated != consumer:
        raise ResidentPreflightError(
            "designated_timestamp_core must equal consumer_core; all intervals use one clock"
        )
    l1_bytes = resident_l1_allocation_bytes(ring_pages=ring_pages, watcher=watcher)
    if l1_bytes > MAX_RING_L1_BYTES:
        raise ResidentPreflightError(
            f"ring_pages={ring_pages} exceeds the L1 preflight budget "
            f"({MAX_RING_L1_BYTES} bytes including scratch pages and semaphores)"
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
        fixed_work_ticks_per_frame=fixed_work_ticks,
        budget_aiclk_mhz=budget_aiclk_mhz,
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
    """The per-frame fixed-work budget is inclusive: reaching it is an error."""
    if elapsed_ticks < 0 or cycle_budget <= 0:
        raise ValueError("elapsed_ticks must be non-negative and cycle_budget positive")
    return elapsed_ticks >= cycle_budget


def run_budget_exceeded(*, elapsed_ticks: int, run_budget_ticks: int) -> bool:
    """Return whether the run-wide budget has been reached."""
    if elapsed_ticks < 0 or run_budget_ticks <= 0:
        raise ValueError("elapsed_ticks must be non-negative and run_budget_ticks positive")
    return elapsed_ticks >= run_budget_ticks


def ticks_to_seconds(*, ticks: int, aiclk_mhz: int) -> float:
    """Convert device-clock ticks to seconds using the observed AICLK."""
    if (
        isinstance(ticks, bool)
        or not isinstance(ticks, int)
        or ticks < 0
        or isinstance(aiclk_mhz, bool)
        or not isinstance(aiclk_mhz, int)
        or aiclk_mhz <= 0
    ):
        raise ValueError("ticks must be non-negative and aiclk_mhz must be positive integers")
    return ticks / (aiclk_mhz * 1_000_000)


def periodic_gap_decomposition(
    gaps: Iterable[int], *, period_frames: int
) -> dict[str, Any]:
    """Return sanitized quotient/remainder entries and aggregate counts."""
    if isinstance(period_frames, bool) or not isinstance(period_frames, int) or period_frames <= 0:
        raise ValueError("period_frames must be a positive integer")
    entries = []
    aggregate: dict[str, int] = {}
    for gap in gaps:
        if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
            raise ValueError("gaps must contain non-negative integers")
        quotient, remainder = divmod(gap, period_frames)
        entries.append(
            {
                "gap_frames": gap,
                "quotient": quotient,
                "remainder": remainder,
            }
        )
        key = f"{quotient},{remainder}"
        aggregate[key] = aggregate.get(key, 0) + 1
    return {
        "period_frames": period_frames,
        "entries": entries,
        "aggregate_by_quotient_remainder": aggregate,
    }


def interval_ticks_for_microseconds(*, microseconds: int, aiclk_mhz: int) -> int:
    """Convert an integer wall-time interval to AICLK ticks.

    AICLK in MHz is cycles per microsecond, so this conversion is exact for
    integer microseconds and avoids mixing nanoseconds with device ticks.
    """
    if (
        isinstance(microseconds, bool)
        or not isinstance(microseconds, int)
        or microseconds <= 0
        or isinstance(aiclk_mhz, bool)
        or not isinstance(aiclk_mhz, int)
        or aiclk_mhz <= 0
    ):
        raise ValueError("microseconds and aiclk_mhz must be positive integers")
    ticks = microseconds * aiclk_mhz
    if ticks > UINT64_MAX:
        raise ResidentPreflightError("interval conversion exceeds the 64-bit tick range")
    return ticks


def startup_allowance_ticks(*, aiclk_mhz: int) -> int:
    """Return the explicit 100 ms startup allowance in device ticks."""
    return interval_ticks_for_microseconds(
        microseconds=STARTUP_ALLOWANCE_MICROSECONDS,
        aiclk_mhz=aiclk_mhz,
    )


def run_budget_breakdown(config: ResidentConfig, *, watcher: bool = False) -> dict[str, int]:
    """Compute the conservative run-wide budget in AICLK ticks.

    The endpoint convention deliberately charges ``N`` frame intervals, not
    ``N-1``: this covers the configured pacing interval plus the terminal
    frame's pacing/teardown boundary. ``fixed_work_ticks_per_frame`` is the
    conservative work estimate; ``cycle_budget`` remains the per-frame safety
    cap and is not itself the run-wide limit. A fixed 10% margin is
    explicit and bounded, rather than hidden in the device kernel.
    """
    config = validate_configuration(config, watcher=watcher)
    pacing_ticks = config.frame_count * config.frame_interval_ticks
    fixed_work_ticks = config.frame_count * config.fixed_work_ticks_per_frame
    startup_ticks = startup_allowance_ticks(aiclk_mhz=config.budget_aiclk_mhz)
    critical_path_ticks = max(pacing_ticks, fixed_work_ticks)
    base_ticks = critical_path_ticks + startup_ticks
    watcher_overhead_percent = WATCHER_OVERHEAD_MARGIN_PERCENT if watcher else 0
    margin_percent = RUN_BUDGET_SAFETY_MARGIN_PERCENT + watcher_overhead_percent
    margin_ticks = (base_ticks * margin_percent + 99) // 100
    run_budget_ticks = base_ticks + margin_ticks
    if run_budget_ticks > UINT64_MAX:
        raise ResidentPreflightError("run-wide cycle budget exceeds the 64-bit tick range")
    return {
        "frame_count": config.frame_count,
        "frame_interval_ticks": config.frame_interval_ticks,
        "pacing_ticks": pacing_ticks,
        "per_frame_work_budget_ticks": config.fixed_work_ticks_per_frame,
        "cycle_budget_ticks_per_frame": config.cycle_budget,
        "fixed_work_ticks": fixed_work_ticks,
        "critical_path_ticks": critical_path_ticks,
        "schedule_ticks": critical_path_ticks + startup_ticks,
        "overlap_model": "producer pacing and consumer fixed work overlap; critical path is max",
        "startup_allowance_ms": STARTUP_ALLOWANCE_MICROSECONDS // 1_000,
        "budget_aiclk_mhz": config.budget_aiclk_mhz,
        "startup_allowance_ticks": startup_ticks,
        "safety_margin_percent": RUN_BUDGET_SAFETY_MARGIN_PERCENT,
        "watcher_overhead_margin_percent": watcher_overhead_percent,
        "total_margin_percent": margin_percent,
        "safety_margin_ticks": margin_ticks,
        "run_budget_ticks": run_budget_ticks,
    }


def validate_run_budget_fits_outer_cap(
    config: ResidentConfig,
    *,
    watcher: bool = False,
) -> dict[str, int]:
    """Reject a margin-inclusive schedule that exceeds its approved cap."""
    config = validate_configuration(config, watcher=watcher)
    breakdown = run_budget_breakdown(config, watcher=watcher)
    cap_ticks = config.outer_timeout_seconds * config.budget_aiclk_mhz * 1_000_000
    if breakdown["run_budget_ticks"] > cap_ticks:
        raise ResidentPreflightError(
            "run budget exceeds outer cap: "
            f"{breakdown['run_budget_ticks']} ticks > {cap_ticks} ticks "
            f"at {config.budget_aiclk_mhz} MHz"
        )
    return breakdown


def failure_name(code: int) -> str:
    if isinstance(code, bool) or not isinstance(code, int) or code not in FAILURE_CODES:
        raise ValueError("unknown resident failure code")
    return FAILURE_CODES[code]


def select_failure_check(*failures: dict[str, Any] | None) -> dict[str, Any]:
    """Select one deterministic failure when producer/consumer report together."""
    candidates = [failure for failure in failures if failure and failure["name"] != "none"]
    if not candidates:
        return {"code": 0, "name": "none", "source": "none"}
    return min(candidates, key=lambda failure: FAILURE_PRIORITY[failure["name"]])


def split_u64(value: int) -> tuple[int, int]:
    """Split a non-negative 64-bit tick value into runtime-argument words."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ValueError("value must be an unsigned 64-bit integer")
    return value & 0xFFFFFFFF, value >> 32


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


def validate_pinned_environment(environment: Mapping[str, Any]) -> None:
    """Require the exact immutable image identity before resident execution."""
    image = environment.get("image")
    if (
        environment.get("image_pinned") is not True
        or not isinstance(image, str)
        or re.search(r"@sha256:[0-9a-f]{64}$", image) is None
    ):
        raise ResidentPreflightError(
            "resident execution requires image_pinned=true and image @sha256:<64 lowercase hex>"
        )


def _require_environment(environment: Mapping[str, Any]) -> None:
    required = ("board", "firmware", "kmd_version", "image")
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ValueError("environment is missing required fields: " + ", ".join(missing))
    validate_pinned_environment(environment)


def _safe_trace_name(power_trace: str) -> str:
    path = PurePath(power_trace)
    if path.is_absolute() or len(path.parts) != 1 or path.name != power_trace:
        raise ValueError("power_trace must be a repository-relative filename")
    return power_trace


def _source_evidence() -> dict[str, Any]:
    return {
        "toolchain": "tt-metal v0.75.0",
        "tt_metal_revision": "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd",
        "clock_api": "tt_metal/hw/inc/internal/tt-1xx/risc_common.h:254-255",
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
    attempted_frame_count: int | None = None,
    produced_frame_count: int | None = None,
    dropped_frame_count: int | None = None,
    aborted_attempts: int = 0,
    failure_check: Mapping[str, Any] | None = None,
    startup_ticks: int | None = None,
    startup_ticks_valid: bool = False,
    work_min_ticks: int | None = None,
    work_max_ticks: int | None = None,
    watcher: bool = False,
    timing_evidence: bool = True,
) -> dict[str, Any]:
    """Build the committed-schema record without retaining raw timestamps.

    Counter protocol (the current producer/consumer semaphore contract):
    ``ready_count`` is the cumulative producer-ready value and equals
    ``produced_frame_count``; the consumer increments the cumulative free value
    once per consumed timestamp, so ``consumed_frame_count`` is its value at
    final download. A normal run satisfies
    ``attempted = produced + dropped``, ``aborted_attempts = 0``, and
    ``consumed = produced``. An error run may stop during one started cadence
    attempt and satisfies ``attempted = produced + dropped + aborted_attempts``
    with ``aborted_attempts`` constrained to 0 or 1; it still requires
    ``consumed <= produced`` and final ring occupancy ``produced - consumed``
    no greater than ``ring_pages``. Producer pacing-budget, consumer fixed-work
    budget, consumer ready-wait, producer cancellation/other, and run-budget
    failures all use this same relation; drop-new attempts contribute only to
    ``dropped`` and never overwrite ring data.
    """
    config = validate_configuration(config, watcher=watcher)
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
    attempted = config.frame_count if attempted_frame_count is None else attempted_frame_count
    produced = len(timestamp_values) if produced_frame_count is None else produced_frame_count
    dropped = attempted - produced if dropped_frame_count is None else dropped_frame_count
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (attempted, produced, dropped))
        or produced < len(timestamp_values)
        or produced > attempted
        or dropped < 0
    ):
        raise ValueError("attempted, produced, and dropped frame counts are inconsistent")
    if isinstance(aborted_attempts, bool) or not isinstance(aborted_attempts, int) or aborted_attempts not in (0, 1):
        raise ValueError("aborted_attempts must be an integer 0 or 1")
    if producer_full_count != dropped:
        raise ValueError("producer_full_count must equal dropped_frame_count")
    intervals = [
        wrap_delta(timestamp_values[index], timestamp_values[index - 1])
        for index in range(1, len(timestamp_values))
    ]
    frame_count_reached = attempted == config.frame_count and produced == attempted - dropped
    reason = termination_reason(
        frame_count_reached=frame_count_reached and produced == len(timestamp_values),
        cycle_budget_hit=bool(cycle_budget_hit),
        outer_timeout=False,
    )
    if reason == "running":
        reason = "incomplete"
    stats = frame_interval_statistics(intervals, bin_width_ticks=config.histogram_bin_ticks)
    failure_hint = bool(
        cycle_budget_hit
        or kernel_error_flag
        or (failure_check is not None and failure_check.get("name") not in (None, "none"))
    )
    if failure_hint:
        if attempted != produced + dropped + aborted_attempts:
            raise ValueError("error counter relation requires attempted=produced+dropped+aborted_attempts")
    elif aborted_attempts != 0 or attempted != produced + dropped:
        raise ValueError("normal counter relation requires attempted=produced+dropped and aborted_attempts=0")
    consumed = len(timestamp_values)
    if consumed > produced or produced - consumed > config.ring_pages:
        raise ValueError("produced, consumed, and ring occupancy counters are inconsistent")
    completed = (
        not failure_hint
        and not cycle_budget_hit
        and not kernel_error_flag
        and frame_count_reached
        and aborted_attempts == 0
        and produced == len(timestamp_values)
    )
    if completed and dropped:
        status = "ok_with_drops"
    elif completed:
        status = "ok"
    else:
        status = "error"
    stats["sample_definition"] = (
        "Intervals between consumer completion timestamps; the first frame is excluded and dropped producer attempts are excluded."
    )
    selected_failure = dict(failure_check) if failure_check is not None else select_failure_check()
    if "name" not in selected_failure or "code" not in selected_failure:
        raise ValueError("failure_check must contain code and name")
    if failure_name(selected_failure["code"]) != selected_failure["name"]:
        raise ValueError("failure_check code/name mismatch")
    if selected_failure["name"] != "none" and any(
        field not in selected_failure for field in ("elapsed_ticks", "limit_ticks", "unit")
    ):
        raise ValueError("failure_check must serialize elapsed and limit units")
    if startup_ticks is not None and (
        isinstance(startup_ticks, bool) or not isinstance(startup_ticks, int) or startup_ticks < 0
    ):
        raise ValueError("startup_ticks must be a non-negative integer or None")
    if (work_min_ticks is None) != (work_max_ticks is None):
        raise ValueError("work minimum and maximum must be provided together")
    if work_min_ticks is not None and (
        isinstance(work_min_ticks, bool)
        or isinstance(work_max_ticks, bool)
        or not isinstance(work_min_ticks, int)
        or not isinstance(work_max_ticks, int)
        or work_min_ticks < 0
        or work_max_ticks < work_min_ticks
    ):
        raise ValueError("work minimum/maximum ticks are invalid")
    return {
        "schema": "issue-12-stage-1-resident-v1",
        "issue": 12,
        "stage": 1,
        "status": status,
        "termination_reason": reason,
        "parameters": {
            **config.as_record(),
            "run_budget": run_budget_breakdown(config, watcher=watcher),
            "cycle_budget_scope": "per_frame_fixed_work; run_budget.run_budget_ticks is run-wide",
            "full_ring_policy": "drop_new_frame_without_waiting_or_overwriting",
            "attempted_frame_count": attempted,
            "produced_frame_count": produced,
            "dropped_frame_count": dropped,
            "aborted_attempts": aborted_attempts,
            "frame_interval_is_not_acquisition_rate_claim": True,
            "frame_interval_note": (
                "The device-clock frame interval is a harness parameter; Stage 1 does not claim the real acquisition rate."
            ),
        },
        "clock": {
            "name": "RISCV_DEBUG_REG_WALL_CLOCK_L",
            "timestamp_api": "get_timestamp_32b",
            "timestamp_semantics": "32-bit low word, software-extended (wrap-tracked)",
            "width_bits": 64,
            "interval_unit": "device_clock_ticks",
            "frequency_source": "AICLK",
            "aiclk_mhz": aiclk_mhz,
            "designated_core": list(config.designated_timestamp_core),
            "cross_core_correlation": "out_of_scope",
        },
        "clock_source_evidence": _source_evidence(),
        "work_ticks": {
            "minimum": work_min_ticks,
            "maximum": work_max_ticks,
            "valid": work_min_ticks is not None,
            "unit": "device_clock_ticks",
        },
        "startup": {
            "configured_allowance_ms": STARTUP_ALLOWANCE_MICROSECONDS // 1_000,
            "configured_allowance_ticks": run_budget_breakdown(config, watcher=watcher)[
                "startup_allowance_ticks"
            ],
            "budget_aiclk_mhz": config.budget_aiclk_mhz,
            "observed_ticks": startup_ticks if startup_ticks_valid else None,
            "observed_valid": bool(startup_ticks_valid and startup_ticks is not None),
            "clock": "designated consumer RISCV_DEBUG_REG_WALL_CLOCK_L; 32-bit low word, software-extended (wrap-tracked)",
        },
        "histogram": stats,
        "ring": {
            "producer_full_count": int(producer_full_count),
            "consumer_empty_count": int(consumer_empty_count),
            "attempted_frame_count": attempted,
            "produced_frame_count": produced,
            "consumed_frame_count": len(timestamp_values),
            "dropped_frame_count": dropped,
            "aborted_attempts": aborted_attempts,
            "overflow_count": int(producer_full_count),
            "synchronization": "ring pointers and control metadata only",
            "full_ring_policy": "drop_new_frame_without_waiting_or_overwriting",
        },
        "failure_check": selected_failure,
        "cycle_budget": {
            "budget_ticks": config.cycle_budget,
            "unit": "device_clock_ticks",
            "scope": "per_frame_fixed_work",
            "run_budget_ticks": run_budget_breakdown(config, watcher=watcher)["run_budget_ticks"],
            "exceeded": bool(cycle_budget_hit),
            "error_flag": int(bool(kernel_error_flag)),
        },
        "raw_timestamps": digest,
        "power_trace": _safe_trace_name(power_trace),
        "environment": dict(environment),
        "harness_commit": harness_commit,
        "watcher": bool(watcher),
        "timing_evidence": bool(not watcher and timing_evidence and completed and dropped == 0),
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
    "CURRENT_WRAP_OBSERVED_WORK_MAX_TICKS",
    "CURRENT_WRAP_WORK_PER_FRAME",
    "MAX_OUTER_TIMEOUT_SECONDS",
    "MAX_TIMING_TIMEOUT_SECONDS",
    "MAX_WATCHER_TIMEOUT_SECONDS",
    "PAGE_BYTES",
    "PAGE_WORDS",
    "RESIDENT_SEMAPHORE_COUNT",
    "SEMAPHORE_BYTES",
    "TIMESTAMP_GAP_LIMIT_TICKS",
    "UINT32_MAX",
    "WORK_TICKS_MARGIN_PERCENT",
    "WORK_TICKS_PER_UNIT_UPPER_BOUND",
    "ResidentConfig",
    "ResidentPreflightError",
    "RingAccounting",
    "build_measurement_record",
    "build_rejection_record",
    "cycle_budget_exceeded",
    "failure_name",
    "frame_interval_statistics",
    "interval_ticks_for_microseconds",
    "periodic_gap_decomposition",
    "required_samples_for_percentile",
    "resident_l1_allocation_bytes",
    "resident_l1_allocation_table",
    "run_budget_breakdown",
    "run_budget_exceeded",
    "select_failure_check",
    "split_u64",
    "startup_allowance_ticks",
    "termination_reason",
    "ticks_to_seconds",
    "timestamp_digest",
    "validate_configuration",
    "validate_pinned_environment",
    "validate_run_budget_fits_outer_cap",
    "wrap_delta",
]
