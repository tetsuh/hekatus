"""Host-side accounting and record helpers for Issue #12 Stage 1.

The Issue #104 record relationships and paired-outlier algorithm live in the
single executable catalog in :mod:`enodia.tt.bench.resident_record`.  This
module supplies the builder-side configuration, accounting, and schema
assembly; it imports and calls that shared validator rather than maintaining a
second invariant implementation.  It has no TTNN import, so configuration,
ring accounting, and record construction remain board-free and testable.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path, PurePath
from typing import Any

from enodia.tt.bench.clock_source_audit import (
    AUDITED_CLOCK_SOURCE_IMAGE,
    CLOCK_SOURCE_AUDIT_TABLE,
    CLOCK_SOURCE_UNAUDITED_DIAGNOSTIC,
    clock_source_audit_for_image,
    source_evidence_for_image,
)
from enodia.tt.bench.resident_clock import (
    AICLK_SOURCE_CONFIGURED,
    AICLK_SOURCE_LEGACY_UNVERIFIED,
    AICLK_SOURCE_RUN_TRACE_SAMPLES,
    safe_aiclk_integer,
    select_elapsed_aiclk,
    ticks_to_seconds,
)
from enodia.tt.bench.resident_record import (
    FAILURE_CODE_TABLE,
    FAILURE_CODES,
    RESIDENT_INVARIANT_CATALOG,
    RESIDENT_RECORD_SCHEMA,
    TIMING_EVIDENCE_ERROR_RECORD_REASON,
    TIMING_EVIDENCE_NOT_REQUESTED,
    TIMING_EVIDENCE_REJECTED_RECORD_REASON,
    TIMING_EVIDENCE_RUN_TRACE_REASON,
    ResidentFailureClassification,
    build_outlier_analysis,
    validate_pair_analysis,
    validate_resident_record,
)
from enodia.tt.bench.sampler_contract import (
    MAX_EXPLICIT_SAMPLER_INTERVAL_SECONDS,
    MIN_EXPLICIT_SAMPLER_INTERVAL_SECONDS,
    SAMPLER_CONTRACT,
    normalize_sampler_metadata,
)

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
PERCENTILES = {
    "p50": 0.50,
    "p99": 0.99,
    "p99_9": 0.999,
    "p99_99": 0.9999,
}
_safe_aiclk_integer = safe_aiclk_integer


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


def validate_failure_check(failure: Mapping[str, Any]) -> ResidentFailureClassification:
    """Validate a serialized failure and return its authoritative classification."""
    if not isinstance(failure, Mapping):
        raise TypeError("failure_check must be an object")
    classification = ResidentFailureClassification.for_code(failure.get("code"))
    if failure.get("name") != classification.name:
        raise ValueError("failure_check code/name mismatch")
    source = failure.get("source")
    if source is not None and source not in classification.sources:
        raise ValueError(
            f"failure_check source {source!r} is invalid for code {classification.code}"
        )
    return classification


def failure_name(code: int) -> str:
    return ResidentFailureClassification.for_code(code).name


def select_failure_check(*failures: dict[str, Any] | None) -> dict[str, Any]:
    """Select one deterministic failure when producer/consumer report together."""
    candidates: list[tuple[ResidentFailureClassification, dict[str, Any]]] = []
    for failure in failures:
        if failure is None:
            continue
        classification = validate_failure_check(failure)
        if classification.code != 0:
            candidates.append((classification, failure))
    if not candidates:
        return {"code": 0, "name": "none", "source": "none"}
    return min(candidates, key=lambda item: item[0].priority)[1]


def split_u64(value: int) -> tuple[int, int]:
    """Split a non-negative 64-bit tick value into runtime-argument words."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ValueError("value must be an unsigned 64-bit integer")
    return value & 0xFFFFFFFF, value >> 32


def termination_reason(
    *,
    frame_count_reached: bool,
    failure_check: Mapping[str, Any] | None = None,
    outer_timeout: bool,
) -> str:
    """Classify the terminal condition, with host timeout taking priority."""
    if outer_timeout:
        return "outer_timeout"
    failure = (
        {"code": 0, "name": "none", "source": "none"}
        if failure_check is None
        else failure_check
    )
    classification = validate_failure_check(failure)
    if classification.termination_reason is not None:
        return classification.termination_reason
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


@dataclass(frozen=True)
class RequiredProvenanceField:
    """One field in the resident record provenance contract."""

    path: str
    phase: str
    expected_type: str
    validator: Callable[[Any], bool]


PROVENANCE_PHASE_PREFLIGHT = "preflight"
PROVENANCE_PHASE_POST_RUN = "post_run"


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_pinned_image(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and re.search(r"@sha256:[0-9a-f]{64}$", value) is not None
    )


def _is_true_boolean(value: Any) -> bool:
    return value is True


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


# This table is the single source of truth for the fields required to identify
# a resident run.  Its paths are the paths in the accepted measurement record;
# the top-level harness_commit entry is also the argument supplied to the run.
REQUIRED_PROVENANCE_FIELDS: tuple[RequiredProvenanceField, ...] = (
    RequiredProvenanceField(
        path="environment.board.serial",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="environment.board.board_type",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="environment.firmware.fw_bundle_version",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="environment.kmd_version",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="environment.image",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type=(
            "a digest-pinned image string with image_pinned=true and lowercase "
            "@sha256:<64 hex>"
        ),
        validator=_is_pinned_image,
    ),
    RequiredProvenanceField(
        path="environment.image_pinned",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="the boolean true",
        validator=_is_true_boolean,
    ),
    RequiredProvenanceField(
        path="environment.harness_commit",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="environment.tt_env_active_release",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="harness_commit",
        phase=PROVENANCE_PHASE_PREFLIGHT,
        expected_type="a non-empty string",
        validator=_is_nonempty_string,
    ),
    RequiredProvenanceField(
        path="clock.aiclk_mhz",
        phase=PROVENANCE_PHASE_POST_RUN,
        expected_type="a positive integer",
        validator=_is_positive_integer,
    ),
)

_MISSING_PROVENANCE_VALUE = object()


def _provenance_value(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING_PROVENANCE_VALUE
        value = value[component]
    return value


def normalize_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical environment copy without mutating the input.

    Telemetry names the board serial as ``board_id``.  A present serial is
    authoritative only when it agrees with a present board ID; a board ID is
    copied to the canonical ``serial`` field only when that field is absent.
    Non-mapping board values are left for the required-field table to reject.
    A sampler object is canonicalized when present, while a missing sampler is
    retained for legacy callers that do not need a runner-side default.
    """
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be an object")
    normalized = dict(environment)
    board = normalized.get("board")
    if isinstance(board, Mapping):
        normalized_board = dict(board)
        has_serial = "serial" in normalized_board
        has_board_id = "board_id" in normalized_board
        if has_serial and has_board_id and normalized_board["serial"] != normalized_board["board_id"]:
            raise ResidentPreflightError(
                "environment.board.serial and environment.board.board_id must match"
            )
        if not has_serial and has_board_id:
            normalized_board["serial"] = normalized_board["board_id"]
        normalized["board"] = normalized_board
    if "telemetry_sampler" in normalized:
        normalized["telemetry_sampler"] = normalize_sampler_metadata(
            normalized["telemetry_sampler"]
        )
    return normalized


def _validate_provenance_fields(
    record: Mapping[str, Any],
    *,
    phase: str,
    fields: Iterable[RequiredProvenanceField] | None = None,
) -> None:
    """Validate one phase by iterating the authoritative provenance table."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be an object")
    selected = REQUIRED_PROVENANCE_FIELDS if fields is None else fields
    for field in selected:
        if field.phase != phase:
            continue
        value = _provenance_value(record, field.path)
        if value is _MISSING_PROVENANCE_VALUE or not field.validator(value):
            raise ResidentPreflightError(
                f"{field.path} is required and must be {field.expected_type}"
            )


def validate_preflight_provenance(
    *, harness_commit: Any, environment: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize and validate all provenance available before opening a device."""
    normalized_environment = normalize_environment(environment)
    _validate_provenance_fields(
        {"harness_commit": harness_commit, "environment": normalized_environment},
        phase=PROVENANCE_PHASE_PREFLIGHT,
    )
    return normalized_environment


def validate_post_run_provenance(record: Mapping[str, Any]) -> None:
    """Validate provenance that is available only after a run completes."""
    _validate_provenance_fields(record, phase=PROVENANCE_PHASE_POST_RUN)


def validate_post_run_aiclk(aiclk_mhz: Any) -> None:
    """Validate the observed AICLK before constructing a final record."""
    validate_post_run_provenance({"clock": {"aiclk_mhz": aiclk_mhz}})


def validate_pinned_environment(environment: Mapping[str, Any]) -> None:
    """Require the immutable image identity using the table's image validators."""
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be an object")
    _validate_provenance_fields(
        {"environment": environment},
        phase=PROVENANCE_PHASE_PREFLIGHT,
        fields=(
            field
            for field in REQUIRED_PROVENANCE_FIELDS
            if field.path.startswith("environment.image")
        ),
    )


def _safe_trace_name(power_trace: str | None) -> str | None:
    if power_trace is None:
        return None
    path = PurePath(power_trace)
    if path.is_absolute() or len(path.parts) != 1 or path.name != power_trace:
        raise ValueError("power_trace must be a repository-relative filename")
    return power_trace


def _telemetry_sampler(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical sampler metadata, defaulting legacy records to 2 seconds."""
    return normalize_sampler_metadata(environment.get("telemetry_sampler"))


def _validate_sampler_trace(environment: Mapping[str, Any], power_trace: str | None) -> dict[str, Any]:
    sampler = _telemetry_sampler(environment)
    contract = SAMPLER_CONTRACT[sampler["mode"]]
    sampler_off = sampler["mode"] == "off"
    if sampler.get("power_trace") != contract["power_trace"]:
        raise ValueError("telemetry sampler power_trace metadata does not match sampler mode")
    if sampler.get("timing_evidence") != contract["timing_evidence"]:
        raise ValueError("telemetry sampler timing_evidence metadata does not match sampler mode")
    if sampler_off != (power_trace is None):
        raise ValueError(
            "sampler-off records must omit the power trace, and sampled records must name it"
        )
    return sampler


def validate_record_inputs(
    *, harness_commit: Any, environment: Mapping[str, Any], power_trace: str | None
) -> dict[str, Any]:
    """Validate and return the runner's canonical preflight environment."""
    normalized_environment = validate_preflight_provenance(
        harness_commit=harness_commit, environment=environment
    )
    sampler = _validate_sampler_trace(normalized_environment, power_trace)
    normalized_environment["telemetry_sampler"] = sampler
    _safe_trace_name(power_trace)
    return normalized_environment


def _trace_failure_reason(trace: Mapping[str, Any] | None) -> str:
    """Return a stable reason code for a trace that cannot support timing."""
    if not isinstance(trace, Mapping):
        return "power_trace_missing"
    if trace.get("readable") is not True:
        return "power_trace_unreadable"
    if not trace.get("nonempty"):
        return "power_trace_empty"
    if not isinstance(trace.get("sample_count"), int) or not isinstance(trace.get("csv_row_count"), int):
        return "power_trace_sample_count_unavailable"
    if trace.get("sample_count") != trace.get("csv_row_count"):
        return "power_trace_sample_count_mismatch"
    if not isinstance(trace.get("valid_row_count"), int):
        return "power_trace_valid_row_count_unavailable"
    if trace.get("valid_row_count") != trace.get("csv_row_count"):
        return "power_trace_invalid_rows"
    if trace.get("timestamps_parse") is not True:
        return "power_trace_timestamp_parse_failed"
    if trace.get("timestamps_ordered") is not True:
        return "power_trace_timestamp_order_failed"
    if trace.get("coverage_complete") is not True:
        return "power_trace_coverage_incomplete"
    if trace.get("in_run_valid_row_count", 0) < 1:
        return "power_trace_no_valid_in_run_rows"
    return "power_trace_coverage_complete"


def _public_power_trace_metadata(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Keep file facts and coverage while excluding parser-only row arrays."""
    public_keys = (
        "file",
        "columns",
        "sha256",
        "readable",
        "nonempty",
        "csv_row_count",
        "sample_count",
        "valid_row_count",
        "in_run_valid_row_count",
        "timestamps_parse",
        "timestamps_ordered",
        "first_timestamp",
        "last_timestamp",
        "run_start",
        "run_end",
        "covers_run_start",
        "covers_run_end",
        "coverage_complete",
        "coverage",
        "aiclk_source",
        "aiclk_mhz",
        "errors",
    )
    return {key: trace.get(key) for key in public_keys}


def _power_trace_metadata(
    *,
    power_trace: str,
    power_trace_path: Path | None,
    power_trace_metadata: Mapping[str, Any] | None,
    run_start: Any,
    run_end: Any,
) -> dict[str, Any]:
    """Read one run-bound trace or normalize supplied board-free metadata."""
    if power_trace_path is not None:
        from enodia.tt.bench.telemetry import parse_power_trace

        parsed = parse_power_trace(power_trace_path, run_start=run_start, run_end=run_end)
        if parsed.get("file") != power_trace:
            raise ValueError("power_trace_path basename must match power_trace")
        return _public_power_trace_metadata(parsed)
    if power_trace_metadata is None:
        return {
            "file": power_trace,
            "columns": [],
            "sha256": None,
            "readable": False,
            "nonempty": False,
            "csv_row_count": 0,
            "sample_count": None,
            "valid_row_count": 0,
            "in_run_valid_row_count": 0,
            "timestamps_parse": False,
            "timestamps_ordered": False,
            "first_timestamp": None,
            "last_timestamp": None,
            "run_start": run_start,
            "run_end": run_end,
            "covers_run_start": False,
            "covers_run_end": False,
            "coverage_complete": False,
            "coverage": {
                "nonempty": False,
                "timestamps_parse": False,
                "timestamps_ordered": False,
                "first_at_or_before_run_start": False,
                "last_at_or_after_run_end": False,
                "readable": False,
                "complete": False,
            },
            "aiclk_source": "no_valid_in_run_samples",
            "aiclk_mhz": None,
            "errors": ["power trace metadata was not supplied"],
        }
    metadata = dict(power_trace_metadata)
    metadata.setdefault("file", power_trace)
    if metadata["file"] != power_trace:
        raise ValueError("power trace metadata file must match power_trace")
    if run_start is not None:
        metadata["run_start"] = run_start
    if run_end is not None:
        metadata["run_end"] = run_end
    return {key: metadata.get(key) for key in (
        "file", "columns", "sha256", "readable", "nonempty", "csv_row_count",
        "sample_count", "valid_row_count", "in_run_valid_row_count", "timestamps_parse",
        "timestamps_ordered", "first_timestamp", "last_timestamp", "run_start", "run_end",
        "covers_run_start", "covers_run_end", "coverage_complete", "coverage", "aiclk_source",
        "aiclk_mhz", "errors"
    )}


def _diagnostic_trace_metadata(
    *, run_start: Any = None, run_end: Any = None
) -> dict[str, Any]:
    """Return explicit trace facts for runs that have no usable trace bytes."""
    coverage = {
        "nonempty": False,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "first_at_or_before_run_start": False,
        "last_at_or_after_run_end": False,
        "readable": False,
        "complete": False,
    }
    return {
        "file": None,
        "columns": [],
        "sha256": None,
        "readable": False,
        "nonempty": False,
        "csv_row_count": 0,
        "sample_count": 0,
        "valid_row_count": 0,
        "in_run_valid_row_count": 0,
        "timestamps_parse": False,
        "timestamps_ordered": False,
        "first_timestamp": None,
        "last_timestamp": None,
        "run_start": run_start,
        "run_end": run_end,
        "covers_run_start": False,
        "covers_run_end": False,
        "coverage_complete": False,
        "coverage": coverage,
        "aiclk_source": "sampler_off_diagnostic",
        "aiclk_mhz": None,
        "errors": [],
    }


def validate_outlier_analysis(
    outlier_analysis: Mapping[str, Any], *, timestamps: Iterable[int] | None = None
) -> None:
    """Compatibility wrapper over the shared pair invariant checker."""
    mismatches = validate_pair_analysis(outlier_analysis, timestamps=timestamps)
    if mismatches:
        first = mismatches[0]
        if first["message"].startswith("outlier_analysis must be"):
            raise TypeError(first["message"])
        raise ValueError(first["message"])


def _source_evidence(image: Any) -> dict[str, Any]:
    return source_evidence_for_image(image)


def build_measurement_record(
    *,
    config: ResidentConfig,
    aiclk_mhz: int,
    aiclk_source: str | None = None,
    timestamps: Iterable[int],
    producer_full_count: int,
    consumer_empty_count: int,
    kernel_error_flag: int,
    harness_commit: str,
    environment: Mapping[str, Any],
    power_trace: str | None,
    power_trace_path: Path | None = None,
    power_trace_metadata: Mapping[str, Any] | None = None,
    run_start: Any = None,
    run_end: Any = None,
    outlier_analysis: Mapping[str, Any] | None = None,
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
    timing_evidence: bool = False,
    timing_evidence_reason: str | None = None,
) -> dict[str, Any]:
    """Build and validate the committed-schema record without raw timestamps.

    When ``power_trace_path`` or explicit trace metadata is supplied, the
    power-provenance field table is strict: the actual CSV bytes, rows, timestamps,
    run coverage, and in-run AICLK are checked here.  Calls that only provide
    the historical basename and omit sampler metadata remain readable for
    older board-free callers, but are marked ``legacy_unverified`` in power
    provenance and can never claim timing evidence; the resident runner
    always supplies the strict path and run bounds.  ``timing_evidence``
    defaults to false.  A true request is retained only after the shared
    validator proves the strict trace, coverage, AICLK, sampler, and run
    invariants.

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
    normalized_environment = validate_preflight_provenance(
        harness_commit=harness_commit, environment=environment
    )
    sampler_metadata_supplied = "telemetry_sampler" in normalized_environment
    sampler = _validate_sampler_trace(normalized_environment, power_trace)
    normalized_environment["telemetry_sampler"] = dict(sampler)
    sampler_off = sampler["mode"] == "off"
    if sampler_off and (power_trace_path is not None or power_trace_metadata is not None):
        raise ValueError("sampler-off records must not carry power trace metadata")
    strict_trace = bool(
        not sampler_off
        and (
            power_trace_path is not None
            or power_trace_metadata is not None
            or run_start is not None
            or run_end is not None
            or sampler_metadata_supplied
        )
    )
    trace_metadata: dict[str, Any] | None = None
    trace_reason: str
    if sampler_off:
        trace_metadata = _diagnostic_trace_metadata(run_start=run_start, run_end=run_end)
        trace_reason = "sampler_off_by_design"
    elif power_trace is None:
        trace_reason = "power_trace_missing"
    elif strict_trace:
        trace_metadata = _power_trace_metadata(
            power_trace=power_trace,
            power_trace_path=power_trace_path,
            power_trace_metadata=power_trace_metadata,
            run_start=run_start,
            run_end=run_end,
        )
        trace_reason = _trace_failure_reason(trace_metadata)
        trace_aiclk = _safe_aiclk_integer(trace_metadata.get("aiclk_mhz"))
        if trace_reason == "power_trace_coverage_complete":
            if trace_aiclk is None:
                trace_reason = "power_trace_no_valid_in_run_rows"
            elif trace_aiclk != aiclk_mhz:
                raise ValueError("aiclk_mhz must equal the maximum valid in-run power trace AICLK")
    else:
        # Historical direct callers supplied only a basename.  Keep that API
        # readable, but make the unverifiable status explicit; the device
        # runner never uses this compatibility branch.
        trace_metadata = _power_trace_metadata(
            power_trace=power_trace,
            power_trace_path=None,
            power_trace_metadata=None,
            run_start=None,
            run_end=None,
        )
        trace_metadata["aiclk_source"] = AICLK_SOURCE_LEGACY_UNVERIFIED
        trace_reason = "legacy_power_trace_unverified"
    selected_source = aiclk_source or (
        trace_metadata.get("aiclk_source")
        if not sampler_off and trace_metadata is not None
        else (
            AICLK_SOURCE_CONFIGURED
            if sampler_off
            else AICLK_SOURCE_RUN_TRACE_SAMPLES
        )
    )
    selection = select_elapsed_aiclk(
        budget_aiclk_mhz=config.budget_aiclk_mhz,
        sampler_mode=sampler["mode"],
        trace_aiclk_mhz=(
            trace_metadata.get("aiclk_mhz")
            if strict_trace and trace_metadata is not None
            else None
        ),
        trace_aiclk_source=(
            trace_metadata.get("aiclk_source")
            if strict_trace and trace_metadata is not None
            else None
        ),
        legacy_aiclk_mhz=(
            aiclk_mhz
            if not sampler_off
            and selected_source not in {AICLK_SOURCE_CONFIGURED, AICLK_SOURCE_RUN_TRACE_SAMPLES}
            and (
                not strict_trace
                or trace_metadata is None
                or trace_metadata.get("aiclk_source") != AICLK_SOURCE_RUN_TRACE_SAMPLES
            )
            else None
        ),
        legacy_aiclk_source=(
            trace_metadata.get("aiclk_source", AICLK_SOURCE_LEGACY_UNVERIFIED)
            if strict_trace and trace_metadata is not None
            else AICLK_SOURCE_LEGACY_UNVERIFIED
        ),
        allow_configured_fallback=(
            not sampler_off
            and selected_source == AICLK_SOURCE_CONFIGURED
        ),
    )
    aiclk_mhz = selection["aiclk_mhz"]
    aiclk_source = selection["aiclk_source"]
    selected_image = normalized_environment["image"]
    source_evidence = _source_evidence(selected_image)
    validate_post_run_aiclk(aiclk_mhz)
    if isinstance(producer_full_count, bool) or producer_full_count < 0:
        raise ValueError("producer_full_count must be non-negative")
    if isinstance(consumer_empty_count, bool) or consumer_empty_count < 0:
        raise ValueError("consumer_empty_count must be non-negative")
    if kernel_error_flag not in (0, 1, False, True):
        raise ValueError("kernel_error_flag must be a boolean or 0/1")
    selected_failure = (
        dict(failure_check)
        if failure_check is not None
        else select_failure_check()
    )
    classification = validate_failure_check(selected_failure)
    if classification.code != 0 and any(
        field not in selected_failure for field in ("elapsed_ticks", "limit_ticks", "unit")
    ):
        raise ValueError("failure_check must serialize elapsed and limit units")
    if bool(kernel_error_flag) != classification.error_flag:
        raise ValueError("kernel_error_flag must agree with failure_check")
    timestamp_values = list(timestamps)
    digest = timestamp_digest(timestamp_values)
    if outlier_analysis is None:
        outlier_analysis = build_outlier_analysis(
            timestamp_values,
            aiclk_mhz=aiclk_mhz,
            aiclk_source=aiclk_source,
            sampler_metadata=sampler,
        )
    else:
        outlier_analysis = dict(outlier_analysis)
    outlier_analysis.setdefault("aiclk_source", aiclk_source)
    attempted = config.frame_count if attempted_frame_count is None else attempted_frame_count
    produced = len(timestamp_values) if produced_frame_count is None else produced_frame_count
    dropped = attempted - produced if dropped_frame_count is None else dropped_frame_count
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (attempted, produced, dropped)
    ):
        raise ValueError("attempted, produced, and dropped frame counts are inconsistent")
    if (
        isinstance(aborted_attempts, bool)
        or not isinstance(aborted_attempts, int)
        or aborted_attempts not in (0, 1)
    ):
        raise ValueError("aborted_attempts must be an integer 0 or 1")
    intervals = [
        wrap_delta(timestamp_values[index], timestamp_values[index - 1])
        for index in range(1, len(timestamp_values))
    ]
    frame_count_reached = attempted == config.frame_count and produced == attempted - dropped
    reason = termination_reason(
        frame_count_reached=frame_count_reached and produced == len(timestamp_values),
        failure_check=selected_failure,
        outer_timeout=False,
    )
    if reason == "running":
        reason = "incomplete"
    stats = frame_interval_statistics(intervals, bin_width_ticks=config.histogram_bin_ticks)
    failure_hint = classification.error_flag
    completed = (
        not failure_hint
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
    error_record = classification.error_flag or status == "error"
    if error_record:
        final_timing_reason = TIMING_EVIDENCE_ERROR_RECORD_REASON
    elif timing_evidence_reason is not None:
        final_timing_reason = timing_evidence_reason
    elif sampler_off:
        final_timing_reason = "sampler_off_by_design"
    elif watcher:
        final_timing_reason = "watcher_diagnostic_only"
    elif not completed or dropped:
        final_timing_reason = "run_incomplete_or_dropped_frames"
    elif strict_trace and trace_reason != "power_trace_coverage_complete":
        final_timing_reason = trace_reason
    elif strict_trace and timing_evidence is True:
        final_timing_reason = TIMING_EVIDENCE_RUN_TRACE_REASON
    elif strict_trace:
        final_timing_reason = TIMING_EVIDENCE_NOT_REQUESTED
    else:
        final_timing_reason = trace_reason
    # The shared validator is the sole timing gate.  This value is only the
    # caller's request; validate_resident_record below proves or rejects it
    # against the actual trace, coverage, provenance, and sampler invariants.
    timing_ok = timing_evidence is True and not error_record
    record = {
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
            "aiclk_source": aiclk_source,
            "aiclk_observation": (
                "configured run AICLK; sampler off provides no loaded-clock trace"
                if aiclk_source == AICLK_SOURCE_CONFIGURED
                else (
                    "valid in-run power trace samples"
                    if aiclk_source == AICLK_SOURCE_RUN_TRACE_SAMPLES
                    else "unverified legacy caller value"
                )
            ),
            "designated_core": list(config.designated_timestamp_core),
            "cross_core_correlation": "out_of_scope",
        },
        "timestamp_attribution": {
            "status": "open",
            "consumer_completion": {
                "clock": "designated consumer RISCV_DEBUG_REG_WALL_CLOCK",
                "core": list(config.designated_timestamp_core),
            },
            "producer_write": {
                "available": False,
                "clock": "producer-local RISCV_DEBUG_REG_WALL_CLOCK",
                "core": list(config.producer_core),
                "reason": (
                    "The producer and consumer call get_timestamp_32b() on different cores. "
                    "The source has no same-designated-core producer stamp without changing "
                    "the pacing kernel, so producer-versus-consumer attribution remains open."
                ),
            },
        },
        "clock_source_evidence": source_evidence,
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
            "exceeded": classification.cycle_budget_exceeded,
            "error_flag": int(classification.error_flag),
        },
        "raw_timestamps": digest,
        "power_trace": _safe_trace_name(power_trace),
        "power_trace_sample_count": (
            trace_metadata.get("sample_count") if trace_metadata is not None else None
        ),
        "power_trace_csv_row_count": (
            trace_metadata.get("csv_row_count") if trace_metadata is not None else None
        ),
        "power_trace_valid_row_count": (
            trace_metadata.get("valid_row_count") if trace_metadata is not None else None
        ),
        "power_trace_in_run_valid_row_count": (
            trace_metadata.get("in_run_valid_row_count") if trace_metadata is not None else None
        ),
        "power_trace_sha256": (
            trace_metadata.get("sha256") if trace_metadata is not None else None
        ),
        "power_trace_absent_reason": "sampler_off_by_design" if sampler_off else None,
        "power_trace_coverage": (
            {
                "coverage": trace_metadata.get("coverage"),
                "coverage_complete": trace_metadata.get("coverage_complete"),
                "first_timestamp": trace_metadata.get("first_timestamp"),
                "last_timestamp": trace_metadata.get("last_timestamp"),
                "run_start": trace_metadata.get("run_start"),
                "run_end": trace_metadata.get("run_end"),
                "valid_row_count": trace_metadata.get("valid_row_count"),
                "in_run_valid_row_count": trace_metadata.get("in_run_valid_row_count"),
            }
            if trace_metadata is not None
            else None
        ),
        "power_clock_provenance": (
            {
                "trace": trace_metadata.get("file"),
                **trace_metadata,
                "aiclk_source": trace_metadata.get("aiclk_source"),
                "timing_evidence": bool(timing_ok),
                "timing_evidence_reason": final_timing_reason,
            }
            if trace_metadata is not None
            else {
                "trace": None,
                "aiclk_source": "sampler_off_diagnostic",
                "timing_evidence": False,
                "timing_evidence_reason": final_timing_reason,
                "run_start": run_start,
                "run_end": run_end,
            }
        ),
        "run_start": trace_metadata.get("run_start") if trace_metadata is not None else run_start,
        "run_end": trace_metadata.get("run_end") if trace_metadata is not None else run_end,
        "telemetry_sampler": sampler,
        "environment": normalized_environment,
        "harness_commit": harness_commit,
        "watcher": bool(watcher),
        "timing_evidence": timing_ok,
        "timing_evidence_reason": final_timing_reason,
        "outlier_analysis": dict(outlier_analysis),
    }
    validate_resident_record(
        record,
        timestamps=timestamp_values,
        power_trace_path=power_trace_path,
        power_trace_metadata=trace_metadata,
        builder=True,
        strict_trace=strict_trace,
        raise_on_error=True,
    )
    return record


def build_rejection_record(
    *, config: ResidentConfig, reason: str, environment: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Record a rejected configuration with explicit diagnostic provenance."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("rejection reason is required")
    supplied_environment = dict(environment) if isinstance(environment, Mapping) else {}
    image = supplied_environment.get("image")
    try:
        sampler = normalize_sampler_metadata(supplied_environment.get("telemetry_sampler"))
    except (TypeError, ValueError):
        sampler = {
            "mode": "off",
            "interval_seconds": None,
            "power_trace": "absent_by_design",
            "timing_evidence": "diagnostic_only",
        }
    supplied_environment["image"] = image
    supplied_environment.setdefault("image_pinned", False)
    supplied_environment["telemetry_sampler"] = sampler
    supplied_environment.setdefault("harness_commit", None)
    trace_metadata = _diagnostic_trace_metadata()
    outlier = build_outlier_analysis([], sampler_metadata=sampler)
    histogram = frame_interval_statistics([], bin_width_ticks=1)
    parameters = {
        **config.as_record(),
        "attempted_frame_count": 0,
        "produced_frame_count": 0,
        "dropped_frame_count": 0,
        "aborted_attempts": 0,
    }
    result: dict[str, Any] = {
        "schema": RESIDENT_RECORD_SCHEMA,
        "issue": 12,
        "stage": 1,
        "status": "rejected",
        "termination_reason": "rejected",
        "rejection_reason": reason,
        "parameters": parameters,
        "frame_interval_is_not_acquisition_rate_claim": True,
        "clock": {
            "aiclk_mhz": None,
            "aiclk_source": "rejected_preflight",
        },
        "clock_source_evidence": _source_evidence(image),
        "histogram": {
            **histogram,
            "sample_definition": (
                "No device timestamps were collected because preflight rejected the run."
            ),
        },
        "ring": {
            "producer_full_count": 0,
            "consumer_empty_count": 0,
            "attempted_frame_count": 0,
            "produced_frame_count": 0,
            "consumed_frame_count": 0,
            "dropped_frame_count": 0,
            "aborted_attempts": 0,
            "overflow_count": 0,
        },
        "failure_check": {"code": 0, "name": "none", "source": "none"},
        "cycle_budget": {
            "budget_ticks": config.cycle_budget,
            "unit": "device_clock_ticks",
            "scope": "preflight_not_run",
            "run_budget_ticks": None,
            "exceeded": False,
            "error_flag": 0,
        },
        "raw_timestamps": timestamp_digest([]),
        "power_trace": None,
        "power_trace_sample_count": trace_metadata["sample_count"],
        "power_trace_csv_row_count": trace_metadata["csv_row_count"],
        "power_trace_valid_row_count": trace_metadata["valid_row_count"],
        "power_trace_in_run_valid_row_count": trace_metadata["in_run_valid_row_count"],
        "power_trace_sha256": trace_metadata["sha256"],
        "power_trace_absent_reason": "rejected_preflight",
        "power_trace_coverage": {
            "coverage": trace_metadata["coverage"],
            "coverage_complete": trace_metadata["coverage_complete"],
            "first_timestamp": trace_metadata["first_timestamp"],
            "last_timestamp": trace_metadata["last_timestamp"],
            "run_start": trace_metadata["run_start"],
            "run_end": trace_metadata["run_end"],
            "valid_row_count": trace_metadata["valid_row_count"],
            "in_run_valid_row_count": trace_metadata["in_run_valid_row_count"],
        },
        "power_clock_provenance": {
            "trace": None,
            **trace_metadata,
            "timing_evidence": False,
            "timing_evidence_reason": TIMING_EVIDENCE_REJECTED_RECORD_REASON,
        },
        "run_start": None,
        "run_end": None,
        "telemetry_sampler": sampler,
        "environment": supplied_environment,
        "harness_commit": supplied_environment.get("harness_commit"),
        "watcher": False,
        "timing_evidence": False,
        "timing_evidence_reason": TIMING_EVIDENCE_REJECTED_RECORD_REASON,
        "outlier_analysis": outlier,
    }
    validate_resident_record(result, builder=True, raise_on_error=False)
    return result


__all__ = [
    "AICLK_SOURCE_CONFIGURED",
    "AICLK_SOURCE_LEGACY_UNVERIFIED",
    "AICLK_SOURCE_RUN_TRACE_SAMPLES",
    "AUDITED_CLOCK_SOURCE_IMAGE",
    "CLOCK_SOURCE_AUDIT_TABLE",
    "CLOCK_SOURCE_UNAUDITED_DIAGNOSTIC",
    "CURRENT_WRAP_OBSERVED_WORK_MAX_TICKS",
    "CURRENT_WRAP_WORK_PER_FRAME",
    "FAILURE_CODES",
    "FAILURE_CODE_TABLE",
    "MAX_EXPLICIT_SAMPLER_INTERVAL_SECONDS",
    "MAX_OUTER_TIMEOUT_SECONDS",
    "MAX_TIMING_TIMEOUT_SECONDS",
    "MAX_WATCHER_TIMEOUT_SECONDS",
    "MIN_EXPLICIT_SAMPLER_INTERVAL_SECONDS",
    "PAGE_BYTES",
    "PAGE_WORDS",
    "PROVENANCE_PHASE_POST_RUN",
    "PROVENANCE_PHASE_PREFLIGHT",
    "REQUIRED_PROVENANCE_FIELDS",
    "RESIDENT_INVARIANT_CATALOG",
    "RESIDENT_SEMAPHORE_COUNT",
    "SAMPLER_CONTRACT",
    "SEMAPHORE_BYTES",
    "TIMESTAMP_GAP_LIMIT_TICKS",
    "UINT32_MAX",
    "WORK_TICKS_MARGIN_PERCENT",
    "WORK_TICKS_PER_UNIT_UPPER_BOUND",
    "RequiredProvenanceField",
    "ResidentConfig",
    "ResidentFailureClassification",
    "ResidentPreflightError",
    "RingAccounting",
    "build_measurement_record",
    "build_outlier_analysis",
    "build_rejection_record",
    "clock_source_audit_for_image",
    "cycle_budget_exceeded",
    "failure_name",
    "frame_interval_statistics",
    "interval_ticks_for_microseconds",
    "normalize_environment",
    "periodic_gap_decomposition",
    "required_samples_for_percentile",
    "resident_l1_allocation_bytes",
    "resident_l1_allocation_table",
    "run_budget_breakdown",
    "run_budget_exceeded",
    "select_elapsed_aiclk",
    "select_failure_check",
    "split_u64",
    "startup_allowance_ticks",
    "termination_reason",
    "ticks_to_seconds",
    "timestamp_digest",
    "validate_configuration",
    "validate_failure_check",
    "validate_outlier_analysis",
    "validate_pinned_environment",
    "validate_post_run_aiclk",
    "validate_post_run_provenance",
    "validate_preflight_provenance",
    "validate_record_inputs",
    "validate_resident_record",
    "validate_run_budget_fits_outer_cap",
    "wrap_delta",
]
