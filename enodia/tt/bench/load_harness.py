"""Board-free contracts and accounting for Issue #118 load measurements.

The module intentionally contains no TTNN import and no device side effect.  It
is the host-side authority for the four measurement conditions, the fixed load
work, the p150 worker partition, calibration metadata, and the interval
coverage checks that decide whether a run can be used as evidence.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

# The reproducible p150 worker model is deliberately a closed set.  A device
# reporting another grid is rejected rather than silently filling in cores.
P150_WORKER_GRID = (11, 10)
P150_WORKER_COUNT = P150_WORKER_GRID[0] * P150_WORKER_GRID[1]
RESIDENT_PRODUCER_CORE = (0, 0)
RESIDENT_CONSUMER_CORE = (1, 0)
RESIDENT_CORES = (RESIDENT_PRODUCER_CORE, RESIDENT_CONSUMER_CORE)
P150_ALL_CORE_COORDINATES = tuple(
    (x, y)
    for y in range(P150_WORKER_GRID[1])
    for x in range(P150_WORKER_GRID[0])
)
P150_RESIDUAL_WORKER_COORDINATES = tuple(
    coordinate for coordinate in P150_ALL_CORE_COORDINATES if coordinate not in RESIDENT_CORES
)
P150_RESIDUAL_WORKER_COUNT = len(P150_RESIDUAL_WORKER_COORDINATES)

# All eventual modes open the device this way.  CQ0 is the resident program;
# CQ1 is reserved for condition (c)'s host transfers and is otherwise idle.
NUM_COMMAND_QUEUES = 2
RESIDENT_COMMAND_QUEUE = 0
TRANSFER_COMMAND_QUEUE = 1
DEVICE_OPEN_SETTINGS = {
    "num_command_queues": NUM_COMMAND_QUEUES,
}

# ADR-0013 defaults and the existing catalogue point used by the load.
NS_LOAD_SHAPE = "newton_schulz_L32_b8192"
NS_LOAD_L = 32
NS_LOAD_BATCH = 8192
NS_LOAD_ITERATIONS = 12
NS_LOAD_OUTER_REPETITIONS = 3
NS_LOAD_VARIANT = "bf16"
NS_LOAD_MATH_FIDELITY = "HiFi3"
NS_LOAD_FUSE_S = True
NS_LOAD_FP32_DEST_ACC_EN = True
NS_LOAD_MATRIX_BLOCK = 8
NS_LOAD_DOUBLE_BUFFER = True
NS_LOAD_DST_FULL_SYNC_EN = True
NS_LOAD_R_MEMORY = "l1"
NS_LOAD_X0_MEMORY = "l1"
NS_LOAD_OUTPUT_MEMORY = "dram"
NS_LOAD_MAX_TILES_PER_CORE = 80
NS_LOAD_L1_STATIC_BYTES = 111_360
NS_LOAD_L1_CB_BYTES = 358_400
NS_LOAD_L1_TENSOR_BYTES = 825_344
NS_LOAD_L1_TOTAL_BYTES = 1_295_104
L1_TOTAL_BUDGET_BYTES = 1_572_864

# DRAM stream constants are fixed work, not a throughput-dependent loop.
DRAM_BANK_COUNT = 8
DRAM_PAGE_BYTES = 4_096
DRAM_PAGES_PER_BANK = 256
DRAM_TOTAL_PAGES = DRAM_BANK_COUNT * DRAM_PAGES_PER_BANK
DRAM_STREAM_BYTES_PER_DIRECTION = DRAM_TOTAL_PAGES * DRAM_PAGE_BYTES
DRAM_STREAM_TOTAL_BYTES = 2 * DRAM_STREAM_BYTES_PER_DIRECTION

# Condition (c) uses 4,096 fixed arithmetic rounds of 4,096 words and 256
# fixed 64 KiB CQ1 transfer pairs. The runner reuses this small ring so
# calibrated host repeats do not allocate buffers proportional to repeat_count.
HOST_ARITHMETIC_ROUNDS = 4_096
HOST_ARITHMETIC_WORDS = 4_096
HOST_TRANSFER_BYTES = 64 * 1024
HOST_TRANSFER_COUNT = 256
HOST_TRANSFER_BUFFER_POOL_SIZE = 4
MAX_CALIBRATION_SECONDS = 60.0
MAX_MEASUREMENT_TARGET_SECONDS = 500.0

LOAD_CONDITIONS = (
    "unloaded",
    "newton_schulz",
    "dram",
    "host_cpu_transfer",
)
_CONDITION_ALIASES = {
    "control": "unloaded",
    "unloaded_control": "unloaded",
    "newton-schulz": "newton_schulz",
    "newton_schulz_load": "newton_schulz",
    "ns": "newton_schulz",
    "dram_stream": "dram",
    "dram_load": "dram",
    "host": "host_cpu_transfer",
    "host_cpu": "host_cpu_transfer",
    "condition_c": "host_cpu_transfer",
    "c": "host_cpu_transfer",
}
_DEVICE_LOAD_CONDITIONS = frozenset({"newton_schulz", "dram"})
_REPEAT_THROUGHPUT_UNITS = {
    "newton_schulz": "repeats_per_card_second",
    "dram": "repeats_per_card_second",
    "host_cpu_transfer": "repeats_per_host_second",
}


class LoadConfigurationError(ValueError):
    """A load configuration is unsafe or cannot produce evidence."""


class LoadEvidenceError(ValueError):
    """A record does not contain valid interval coverage evidence."""


def repeat_throughput_unit(condition: str) -> str:
    """Return the only valid calibration unit for one condition."""
    canonical = normalize_condition(condition)
    if canonical == "unloaded":
        raise LoadConfigurationError("unloaded control has no repeat throughput unit")
    return _REPEAT_THROUGHPUT_UNITS[canonical]


def normalize_condition(condition: str) -> str:
    """Return one canonical condition name or reject an unknown mode."""
    if not isinstance(condition, str):
        raise LoadConfigurationError("condition must be a string")
    canonical = _CONDITION_ALIASES.get(condition, condition)
    if canonical not in LOAD_CONDITIONS:
        raise LoadConfigurationError(
            f"condition must be one of {LOAD_CONDITIONS}, got {condition!r}"
        )
    return canonical


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LoadConfigurationError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LoadConfigurationError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoadConfigurationError(f"{name} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise LoadConfigurationError(f"{name} must be a finite positive number")
    return converted


def _uint64(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise LoadEvidenceError(f"{name} must be an unsigned 64-bit integer")
    return int(value)


def _uint32(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 32:
        raise LoadEvidenceError(f"{name} must be an unsigned 32-bit integer")
    return int(value)


def validate_worker_model(
    *,
    grid: tuple[int, int] | list[int] = P150_WORKER_GRID,
    coordinates: tuple[tuple[int, int], ...] | list[tuple[int, int]] = P150_RESIDUAL_WORKER_COORDINATES,
) -> tuple[tuple[int, int], ...]:
    """Validate the exact 11x10 p150 model and return its 108 residual cores."""
    if tuple(grid) != P150_WORKER_GRID:
        raise LoadConfigurationError(
            f"unsupported worker grid {grid!r}; expected the pinned p150 model {P150_WORKER_GRID}"
        )
    normalized = tuple(tuple(core) for core in coordinates)
    if normalized != P150_RESIDUAL_WORKER_COORDINATES:
        raise LoadConfigurationError(
            "worker coordinates do not match the pinned residual set after reserving "
            f"{RESIDENT_CORES}"
        )
    if len(normalized) != P150_RESIDUAL_WORKER_COUNT:
        raise LoadConfigurationError("the pinned p150 residual worker set must contain 108 cores")
    if set(normalized) & set(RESIDENT_CORES):
        raise LoadConfigurationError("resident and load core ranges overlap")
    return normalized


@dataclass(frozen=True)
class CorePartition:
    """The single resident/load partition used by every device condition."""

    grid: tuple[int, int]
    resident_cores: tuple[tuple[int, int], tuple[int, int]]
    worker_cores: tuple[tuple[int, int], ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "grid": list(self.grid),
            "resident_cores": [list(core) for core in self.resident_cores],
            "worker_cores": [list(core) for core in self.worker_cores],
            "worker_count": len(self.worker_cores),
            "non_overlap": True,
        }


def p150_core_partition(
    *,
    grid: tuple[int, int] | list[int] = P150_WORKER_GRID,
) -> CorePartition:
    """Build the closed partition; never infer a different physical layout."""
    workers = validate_worker_model(grid=grid)
    return CorePartition(
        grid=P150_WORKER_GRID,
        resident_cores=RESIDENT_CORES,
        worker_cores=workers,
    )


def balanced_work_ranges(
    work_items: int = NS_LOAD_BATCH,
    workers: int = P150_RESIDUAL_WORKER_COUNT,
    *,
    alignment: int = NS_LOAD_MATRIX_BLOCK,
) -> tuple[tuple[int, int], ...]:
    """Split fixed work into contiguous alignment-sized ranges."""
    work_items = _positive_int(work_items, "work_items")
    workers = _positive_int(workers, "workers")
    alignment = _positive_int(alignment, "alignment")
    full_groups, partial = divmod(work_items, alignment)
    group_count = full_groups + int(partial != 0)
    active_workers = min(group_count, workers)
    base, remainder = divmod(group_count, active_workers)
    ranges: list[tuple[int, int]] = []
    group_start = 0
    for index in range(active_workers):
        groups = base + int(index < remainder)
        start = group_start * alignment
        end = min((group_start + groups) * alignment, work_items)
        ranges.append((start, end - start))
        group_start += groups
    return tuple(ranges)


def ns_work_distribution() -> tuple[dict[str, Any], ...]:
    """Return the deterministic L=32/batch=8192 assignment over 108 workers."""
    partition = p150_core_partition()
    ranges = balanced_work_ranges()
    if len(ranges) != len(partition.worker_cores):
        raise LoadConfigurationError("Newton-Schulz work did not use all residual workers")
    rows = tuple(
        {
            "core": list(core),
            "start_tile": start,
            "tile_count": count,
        }
        for core, (start, count) in zip(partition.worker_cores, ranges, strict=True)
    )
    if max(row["tile_count"] for row in rows) != NS_LOAD_MAX_TILES_PER_CORE:
        raise LoadConfigurationError("Newton-Schulz max tiles/core is not the pinned value 80")
    if sum(row["tile_count"] for row in rows) != NS_LOAD_BATCH:
        raise LoadConfigurationError("Newton-Schulz work distribution does not cover batch 8192")
    return rows


def newton_schulz_l1_ledger() -> dict[str, Any]:
    """Return the fixed board-free L1 ledger for the load configuration."""
    total = NS_LOAD_L1_STATIC_BYTES + NS_LOAD_L1_CB_BYTES + NS_LOAD_L1_TENSOR_BYTES
    if total != NS_LOAD_L1_TOTAL_BYTES:
        raise LoadConfigurationError("Newton-Schulz L1 ledger constants are inconsistent")
    if total >= L1_TOTAL_BUDGET_BYTES:
        raise LoadConfigurationError("Newton-Schulz L1 ledger exceeds the board budget")
    return {
        "workers": P150_RESIDUAL_WORKER_COUNT,
        "max_tiles_per_core": NS_LOAD_MAX_TILES_PER_CORE,
        "static_prefix_bytes": NS_LOAD_L1_STATIC_BYTES,
        "circular_buffer_bytes": NS_LOAD_L1_CB_BYTES,
        "tensor_bytes": NS_LOAD_L1_TENSOR_BYTES,
        "total_bytes": total,
        "profile_overhead_bytes": 12_288,
        "profile_total_bytes": total + 12_288,
        "budget_bytes": L1_TOTAL_BUDGET_BYTES,
        "within_budget": True,
        "profile_within_budget": total + 12_288 < L1_TOTAL_BUDGET_BYTES,
    }


def dram_stream_work() -> dict[str, Any]:
    """Return the fixed eight-bank, read/write DRAM stream description."""
    rows = [
        {
            "bank": bank,
            "read_pages": DRAM_PAGES_PER_BANK,
            "write_pages": DRAM_PAGES_PER_BANK,
            "page_bytes": DRAM_PAGE_BYTES,
        }
        for bank in range(DRAM_BANK_COUNT)
    ]
    return {
        "banks": rows,
        "bank_count": DRAM_BANK_COUNT,
        "pages_per_bank": DRAM_PAGES_PER_BANK,
        "total_pages": DRAM_TOTAL_PAGES,
        "page_bytes": DRAM_PAGE_BYTES,
        "read_bytes": DRAM_STREAM_BYTES_PER_DIRECTION,
        "write_bytes": DRAM_STREAM_BYTES_PER_DIRECTION,
        "total_bytes": DRAM_STREAM_TOTAL_BYTES,
        "placement": "interleaved_dram",
        "bank_mapping": "eight_banks",
        "control_flow": "fixed_page_and_counter_loops",
    }


def dram_achieved_gbps(*, elapsed_seconds: float) -> float:
    """Compute read-plus-write stream throughput from a fixed interval."""
    elapsed = _finite_positive(elapsed_seconds, "elapsed_seconds")
    return DRAM_STREAM_TOTAL_BYTES / elapsed / 1_000_000_000.0


def host_transfer_work() -> dict[str, Any]:
    """Return fixed host arithmetic and CQ1 transfer work metadata."""
    return {
        "arithmetic_rounds": HOST_ARITHMETIC_ROUNDS,
        "arithmetic_words": HOST_ARITHMETIC_WORDS,
        "transfer_bytes": HOST_TRANSFER_BYTES,
        "transfer_count": HOST_TRANSFER_COUNT,
        "command_queue": TRANSFER_COMMAND_QUEUE,
        "control_flow": "fixed_arithmetic_and_fixed_size_transfers",
    }


@dataclass(frozen=True)
class CalibrationMetadata:
    """Watcher-off throughput evidence used to choose a fixed run repeat count."""

    condition: str
    duration_seconds: float
    achieved_throughput: float
    throughput_unit: str
    repeat_count: int
    watcher_enabled: bool = False
    source: str = "watcher_off_calibration"
    target_seconds: float | None = None
    safety_factor: float = 1.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition", normalize_condition(self.condition))
        if self.watcher_enabled:
            raise LoadConfigurationError("load calibration must run with Watcher disabled")
        duration = _finite_positive(self.duration_seconds, "duration_seconds")
        if duration > MAX_CALIBRATION_SECONDS:
            raise LoadConfigurationError("calibration duration must be <= 60 seconds")
        _finite_positive(self.achieved_throughput, "achieved_throughput")
        if self.throughput_unit != repeat_throughput_unit(self.condition):
            raise LoadConfigurationError(
                "throughput_unit does not match the calibration condition"
            )
        _positive_int(self.repeat_count, "repeat_count")
        if not isinstance(self.source, str) or not self.source.strip():
            raise LoadConfigurationError("calibration source must be a non-empty string")
        if self.target_seconds is not None:
            target = _finite_positive(self.target_seconds, "target_seconds")
            if target > MAX_MEASUREMENT_TARGET_SECONDS:
                raise LoadConfigurationError(
                    "target_seconds must be <= 500 seconds for the 600-second run cap"
                )
        _finite_positive(self.safety_factor, "safety_factor")

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def choose_repeat_count(
    *,
    condition: str,
    calibration_throughput: float,
    target_seconds: float,
    safety_factor: float = 1.10,
) -> int:
    """Derive a fixed repeat count from one Watcher-off throughput result."""
    canonical = normalize_condition(condition)
    if canonical == "unloaded":
        raise LoadConfigurationError("unloaded control has no load repeat count")
    throughput = _finite_positive(calibration_throughput, "calibration_throughput")
    target = _finite_positive(target_seconds, "target_seconds")
    if target > MAX_MEASUREMENT_TARGET_SECONDS:
        raise LoadConfigurationError(
            "target_seconds must be <= 500 seconds for the 600-second run cap"
        )
    factor = _finite_positive(safety_factor, "safety_factor")
    # ``target_seconds`` may describe the later long measurement.  Only the
    # calibration observation itself is capped at 60 seconds.
    repeats = max(1, math.ceil(throughput * target * factor))
    return repeats


def calibration_metadata(
    *,
    condition: str,
    duration_seconds: float,
    achieved_throughput: float,
    throughput_unit: str,
    target_seconds: float,
    safety_factor: float = 1.10,
    source: str = "watcher_off_calibration",
) -> CalibrationMetadata:
    """Construct the persisted calibration choice and repeat count."""
    canonical = normalize_condition(condition)
    repeats = choose_repeat_count(
        condition=canonical,
        calibration_throughput=achieved_throughput,
        target_seconds=target_seconds,
        safety_factor=safety_factor,
    )
    return CalibrationMetadata(
        condition=canonical,
        duration_seconds=duration_seconds,
        achieved_throughput=achieved_throughput,
        throughput_unit=throughput_unit,
        repeat_count=repeats,
        target_seconds=target_seconds,
        safety_factor=safety_factor,
        source=source,
    )


@dataclass(frozen=True)
class CardLoadInterval:
    """A wrap-tracked card-clock interval with 64-bit extended endpoints."""

    start_ticks: int
    end_ticks: int
    clock: str = "card_riscv_wall_clock"
    width_bits: int = 64
    wrap_tracked: bool = True

    def __post_init__(self) -> None:
        _uint64(self.start_ticks, "card load start_ticks")
        _uint64(self.end_ticks, "card load end_ticks")
        if self.end_ticks < self.start_ticks:
            raise LoadEvidenceError("card load interval end precedes start")
        if self.width_bits != 64:
            raise LoadEvidenceError("card load interval must use a 64-bit extended clock")
        if self.clock != "card_riscv_wall_clock":
            raise LoadEvidenceError("card load interval must use the card clock")
        if self.wrap_tracked is not True:
            raise LoadEvidenceError("card load interval must be wrap-tracked")

    @property
    def elapsed_ticks(self) -> int:
        return (self.end_ticks - self.start_ticks) & ((1 << self.width_bits) - 1)

    def as_record(self) -> dict[str, Any]:
        return {
            "clock": self.clock,
            "width_bits": self.width_bits,
            "timestamp_semantics": "wrap-tracked_extended",
            "start_ticks": self.start_ticks,
            "end_ticks": self.end_ticks,
            "elapsed_ticks": self.elapsed_ticks,
            "wrap_tracked": self.wrap_tracked,
        }


@dataclass(frozen=True)
class HostLoadInterval:
    """A host monotonic interval used only for condition (c)."""

    start_ns: int
    end_ns: int
    clock: str = "host_monotonic_ns"

    def __post_init__(self) -> None:
        _nonnegative_int(self.start_ns, "host load start_ns")
        _nonnegative_int(self.end_ns, "host load end_ns")
        if self.end_ns < self.start_ns:
            raise LoadEvidenceError("host load interval end precedes start")
        if self.clock != "host_monotonic_ns":
            raise LoadEvidenceError("host load interval must use host_monotonic_ns")

    @property
    def elapsed_ns(self) -> int:
        return self.end_ns - self.start_ns

    def as_record(self) -> dict[str, Any]:
        return {
            "clock": self.clock,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "elapsed_ns": self.elapsed_ns,
        }


def extend_card_clock_timestamps(
    low_words: list[int] | tuple[int, ...],
    *,
    initial_extended: int | None = None,
) -> tuple[int, ...]:
    """Extend a monotonic sequence of 32-bit card-clock reads across wraps."""
    if not low_words:
        raise LoadEvidenceError("card-clock timestamp sequence cannot be empty")
    lows = tuple(_uint32(value, "card clock low word") for value in low_words)
    previous = lows[0] if initial_extended is None else _uint64(initial_extended, "initial card timestamp")
    if initial_extended is None:
        extended = [previous]
    else:
        extended = [previous + ((lows[0] - (previous & 0xFFFFFFFF)) & 0xFFFFFFFF)]
    for low in lows[1:]:
        previous = extended[-1]
        extended.append(previous + ((low - (previous & 0xFFFFFFFF)) & 0xFFFFFFFF))
    return tuple(extended)


def _card_coverage(
    interval: CardLoadInterval | None,
    resident_timestamps: list[int] | tuple[int, ...] | None,
) -> dict[str, Any]:
    if interval is None:
        return {
            "clock": "card_riscv_wall_clock",
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "missing_card_load_interval",
        }
    if not resident_timestamps:
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "missing_resident_frame_timestamps",
        }
    try:
        resident = tuple(_uint64(value, "resident timestamp") for value in resident_timestamps)
    except LoadEvidenceError as exc:
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": str(exc),
        }
    if any(end < start for start, end in itertools.pairwise(resident)):
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "resident_card_timestamps_not_monotonic",
        }
    first_frame = resident[0]
    last_frame = resident[-1]
    covered = interval.start_ticks <= first_frame and interval.end_ticks >= last_frame
    return {
        "clock": interval.clock,
        "valid": True,
        "covered": covered,
        "verdict": "covered" if covered else "uncovered",
        "reason": None if covered else "card_load_does_not_cover_resident_interval",
        "first_resident_frame_ticks": first_frame,
        "last_resident_frame_ticks": last_frame,
        "load_start_ticks": interval.start_ticks,
        "load_end_ticks": interval.end_ticks,
    }


def _host_coverage(
    interval: HostLoadInterval | None,
    measurement_start_ns: int | None,
    measurement_end_ns: int | None,
) -> dict[str, Any]:
    if interval is None:
        return {
            "clock": "host_monotonic_ns",
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "missing_host_load_interval",
        }
    try:
        start = _nonnegative_int(measurement_start_ns, "measurement_start_ns")
        end = _nonnegative_int(measurement_end_ns, "measurement_end_ns")
    except LoadConfigurationError as exc:
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": str(exc),
        }
    if end < start:
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "measurement_interval_end_precedes_start",
        }
    covered = interval.start_ns <= start and interval.end_ns >= end
    return {
        "clock": interval.clock,
        "valid": True,
        "covered": covered,
        "verdict": "covered" if covered else "uncovered",
        "reason": None if covered else "host_load_does_not_cover_measurement_interval",
        "measurement_start_ns": start,
        "measurement_end_ns": end,
        "load_start_ns": interval.start_ns,
        "load_end_ns": interval.end_ns,
    }


def validate_device_open_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact two-CQ setting before any mode can be evidence."""
    if not isinstance(settings, Mapping):
        raise LoadConfigurationError("device open settings must be an object")
    value = settings.get("num_command_queues")
    if isinstance(value, bool) or value != NUM_COMMAND_QUEUES:
        raise LoadConfigurationError(
            f"all Issue #118 modes require num_command_queues={NUM_COMMAND_QUEUES}"
        )
    return {
        "num_command_queues": NUM_COMMAND_QUEUES,
        "resident_command_queue": RESIDENT_COMMAND_QUEUE,
        "transfer_command_queue": TRANSFER_COMMAND_QUEUE,
    }


def build_load_record(
    *,
    condition: str,
    repeat_count: int,
    resident_timestamps: list[int] | tuple[int, ...] | None,
    device_open_settings: Mapping[str, Any] | None = None,
    card_interval: CardLoadInterval | None = None,
    host_interval: HostLoadInterval | None = None,
    measurement_start_ns: int | None = None,
    measurement_end_ns: int | None = None,
    calibration: CalibrationMetadata | None = None,
    achieved_throughput: float | None = None,
    throughput_unit: str | None = None,
) -> dict[str, Any]:
    """Assemble one condition record and fail closed on interval omissions."""
    canonical = normalize_condition(condition)
    open_settings = validate_device_open_settings(
        DEVICE_OPEN_SETTINGS if device_open_settings is None else device_open_settings
    )
    if canonical == "unloaded":
        if repeat_count != 0:
            raise LoadConfigurationError("unloaded control must record repeat_count=0")
        if card_interval is not None or host_interval is not None:
            raise LoadConfigurationError("unloaded control cannot claim a load interval")
        coverage = {"verdict": "not_applicable", "valid": True, "covered": True}
    elif canonical in _DEVICE_LOAD_CONDITIONS:
        _positive_int(repeat_count, "repeat_count")
        if host_interval is not None:
            raise LoadEvidenceError("device-load coverage cannot use a host interval")
        coverage = _card_coverage(card_interval, resident_timestamps)
        if not coverage["valid"] or not coverage["covered"]:
            raise LoadEvidenceError(coverage["reason"])
    else:
        _positive_int(repeat_count, "repeat_count")
        if card_interval is not None:
            raise LoadEvidenceError("host condition cannot claim a card-clock load interval")
        coverage = _host_coverage(host_interval, measurement_start_ns, measurement_end_ns)
        if not coverage["valid"] or not coverage["covered"]:
            raise LoadEvidenceError(coverage["reason"])

    if calibration is not None:
        if calibration.condition != canonical:
            raise LoadConfigurationError("calibration condition does not match the load condition")
        if canonical != "unloaded" and repeat_count != calibration.repeat_count:
            raise LoadConfigurationError("repeat_count must match calibration-selected repeat_count")
    if achieved_throughput is not None:
        _finite_positive(achieved_throughput, "achieved_throughput")
        if not isinstance(throughput_unit, str) or not throughput_unit.strip():
            raise LoadConfigurationError("throughput_unit is required with achieved_throughput")

    record: dict[str, Any] = {
        "schema": "issue-118-load-v1",
        "issue": 118,
        "condition": canonical,
        "status": "valid",
        "valid_evidence": True,
        "device_open": open_settings,
        "num_command_queues": NUM_COMMAND_QUEUES,
        "command_queue_roles": {
            "cq0": "resident_program",
            "cq1": "transfer_only_for_host_condition",
        },
        "core_partition": p150_core_partition().as_record(),
        "repeat_count": repeat_count,
        "selected_repeat_count": repeat_count,
        "load": {
            "condition": canonical,
            "repeat_count": repeat_count,
            "outer_repetitions": NS_LOAD_OUTER_REPETITIONS if canonical == "newton_schulz" else None,
            "fixed_work": canonical != "unloaded",
            "data_dependent_exit": False,
        },
        "coverage": coverage,
        "card_load_interval": card_interval.as_record() if card_interval is not None else None,
        "host_load_interval": host_interval.as_record() if host_interval is not None else None,
        "resident_frame_interval": (
            {
                "first_ticks": resident_timestamps[0],
                "last_ticks": resident_timestamps[-1],
                "clock": "card_riscv_wall_clock",
            }
            if resident_timestamps
            else None
        ),
    }
    if canonical == "newton_schulz":
        record["newton_schulz"] = {
            "shape": NS_LOAD_SHAPE,
            "L": NS_LOAD_L,
            "batch": NS_LOAD_BATCH,
            "iterations": NS_LOAD_ITERATIONS,
            "outer_repetitions": NS_LOAD_OUTER_REPETITIONS,
            "effective_outer_repetitions": NS_LOAD_OUTER_REPETITIONS * repeat_count,
            "variant": NS_LOAD_VARIANT,
            "math_fidelity": NS_LOAD_MATH_FIDELITY,
            "fuse_s": NS_LOAD_FUSE_S,
            "fp32_dest_acc_en": NS_LOAD_FP32_DEST_ACC_EN,
            "matrix_block": NS_LOAD_MATRIX_BLOCK,
            "double_buffer": NS_LOAD_DOUBLE_BUFFER,
            "dst_full_sync_en": NS_LOAD_DST_FULL_SYNC_EN,
            "r_memory": NS_LOAD_R_MEMORY,
            "x0_memory": NS_LOAD_X0_MEMORY,
            "output_memory": NS_LOAD_OUTPUT_MEMORY,
            "l1_ledger": newton_schulz_l1_ledger(),
            "work_distribution": list(ns_work_distribution()),
        }
    elif canonical == "dram":
        record["dram_stream"] = dram_stream_work()
    elif canonical == "host_cpu_transfer":
        record["host_work"] = {
            **host_transfer_work(),
            "outer_repetitions": repeat_count,
            "effective_transfer_count": HOST_TRANSFER_COUNT * repeat_count,
        }
    if calibration is not None:
        record["calibration"] = calibration.as_record()
    if achieved_throughput is not None:
        record["achieved_throughput"] = achieved_throughput
        record["throughput_unit"] = throughput_unit
    validate_load_record(record)
    return record


def build_calibration_load_record(
    *,
    condition: str,
    repeat_count: int,
    resident_timestamps: list[int] | tuple[int, ...] | None,
    calibration: CalibrationMetadata,
    card_interval: CardLoadInterval | None = None,
    host_interval: HostLoadInterval | None = None,
    measurement_start_ns: int | None = None,
    measurement_end_ns: int | None = None,
    achieved_throughput: float | None = None,
    throughput_unit: str | None = None,
) -> dict[str, Any]:
    """Build a calibration result without promoting coverage to timing evidence."""
    canonical = normalize_condition(condition)
    if canonical == "unloaded":
        raise LoadConfigurationError("unloaded control does not need calibration")
    if calibration.condition != canonical:
        raise LoadConfigurationError("calibration condition does not match the load condition")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadConfigurationError("calibration measurement repeat_count must be positive")
    coverage = (
        _card_coverage(card_interval, resident_timestamps)
        if canonical in _DEVICE_LOAD_CONDITIONS
        else _host_coverage(host_interval, measurement_start_ns, measurement_end_ns)
    )
    record: dict[str, Any] = {
        "schema": "issue-118-load-calibration-v1",
        "issue": 118,
        "status": "calibration",
        "valid_evidence": False,
        "condition": canonical,
        "device_open": validate_device_open_settings(DEVICE_OPEN_SETTINGS),
        "num_command_queues": NUM_COMMAND_QUEUES,
        "repeat_count": repeat_count,
        "selected_repeat_count": calibration.repeat_count,
        "calibration": calibration.as_record(),
        "coverage": coverage,
        "card_load_interval": card_interval.as_record() if card_interval is not None else None,
        "host_load_interval": host_interval.as_record() if host_interval is not None else None,
        "resident_frame_interval": (
            {
                "first_ticks": resident_timestamps[0],
                "last_ticks": resident_timestamps[-1],
                "clock": "card_riscv_wall_clock",
            }
            if resident_timestamps
            else None
        ),
        "load": {
            "condition": canonical,
            "repeat_count": repeat_count,
            "fixed_work": True,
            "data_dependent_exit": False,
        },
    }
    if achieved_throughput is not None:
        _finite_positive(achieved_throughput, "achieved_throughput")
        if not isinstance(throughput_unit, str) or not throughput_unit.strip():
            raise LoadConfigurationError("throughput_unit is required with achieved_throughput")
        record["achieved_throughput"] = achieved_throughput
        record["throughput_unit"] = throughput_unit
    return record


def validate_load_record(record: Mapping[str, Any]) -> None:
    """Reject malformed, uncovered, or one-CQ records as condition evidence."""
    if not isinstance(record, Mapping):
        raise LoadEvidenceError("load record must be an object")
    if record.get("schema") != "issue-118-load-v1" or record.get("issue") != 118:
        raise LoadEvidenceError("record is not an Issue #118 load record")
    settings = record.get("device_open")
    validate_device_open_settings(settings if isinstance(settings, Mapping) else {})
    if record.get("num_command_queues") != NUM_COMMAND_QUEUES:
        raise LoadEvidenceError("record does not retain the required two-CQ setting")
    canonical = normalize_condition(record.get("condition"))
    load = record.get("load")
    if not isinstance(load, Mapping) or load.get("condition") != canonical:
        raise LoadEvidenceError("load metadata is missing or inconsistent")
    repeat_count = load.get("repeat_count")
    if record.get("repeat_count") != repeat_count or record.get("selected_repeat_count") != repeat_count:
        raise LoadEvidenceError("record repeat metadata is inconsistent")
    if canonical == "unloaded":
        if repeat_count != 0:
            raise LoadEvidenceError("unloaded control must record repeat_count=0")
    elif isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadEvidenceError("loaded conditions must retain a positive fixed repeat_count")
    coverage = record.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("valid") is not True:
        raise LoadEvidenceError("load coverage metadata is missing or invalid")
    if canonical != "unloaded":
        if coverage.get("covered") is not True or coverage.get("verdict") != "covered":
            raise LoadEvidenceError("uncovered load interval is invalid evidence")
        expected_clock = (
            "card_riscv_wall_clock"
            if canonical in _DEVICE_LOAD_CONDITIONS
            else "host_monotonic_ns"
        )
        if coverage.get("clock") != expected_clock:
            raise LoadEvidenceError("load coverage clock does not match the condition")
    if canonical in _DEVICE_LOAD_CONDITIONS:
        interval = record.get("card_load_interval")
        if not isinstance(interval, Mapping) or interval.get("clock") != "card_riscv_wall_clock":
            raise LoadEvidenceError("device load evidence requires card-clock interval metadata")
        start = _uint64(interval.get("start_ticks"), "card load start_ticks")
        end = _uint64(interval.get("end_ticks"), "card load end_ticks")
        if (
            interval.get("width_bits") != 64
            or interval.get("timestamp_semantics") != "wrap-tracked_extended"
            or interval.get("wrap_tracked") is not True
            or interval.get("elapsed_ticks") != end - start
            or end < start
        ):
            raise LoadEvidenceError("card load interval metadata is not wrap-tracked and extended")
        if record.get("host_load_interval") is not None:
            raise LoadEvidenceError("device-load coverage cannot substitute host interval metadata")
    elif canonical == "host_cpu_transfer":
        interval = record.get("host_load_interval")
        if not isinstance(interval, Mapping) or interval.get("clock") != "host_monotonic_ns":
            raise LoadEvidenceError("host condition requires host interval metadata")
        try:
            start = _nonnegative_int(interval.get("start_ns"), "host load start_ns")
            end = _nonnegative_int(interval.get("end_ns"), "host load end_ns")
        except LoadConfigurationError as exc:
            raise LoadEvidenceError(str(exc)) from exc
        if end < start or interval.get("elapsed_ns") != end - start:
            raise LoadEvidenceError("host load interval elapsed_ns is inconsistent")
        if record.get("card_load_interval") is not None:
            raise LoadEvidenceError("host condition must not claim a device load interval")
    if record.get("valid_evidence") is not True or record.get("status") != "valid":
        raise LoadEvidenceError("record is not marked as valid evidence")


def load_evidence_valid(record: Mapping[str, Any]) -> bool:
    """Return a fail-closed boolean wrapper around :func:`validate_load_record`."""
    try:
        validate_load_record(record)
    except (LoadConfigurationError, LoadEvidenceError, TypeError, KeyError):
        return False
    return True


def validate_mode_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate all four mode records and require identical two-CQ settings."""
    missing = set(LOAD_CONDITIONS) - set(records)
    if missing:
        raise LoadEvidenceError(f"mode records missing conditions: {sorted(missing)}")
    for condition in LOAD_CONDITIONS:
        record = records[condition]
        validate_load_record(record)
        if normalize_condition(record.get("condition")) != condition:
            raise LoadEvidenceError(f"mode record key {condition} does not match its condition")
        if record.get("num_command_queues") != NUM_COMMAND_QUEUES:
            raise LoadEvidenceError(f"condition {condition} does not use two command queues")


__all__ = [
    "DEVICE_OPEN_SETTINGS",
    "DRAM_BANK_COUNT",
    "DRAM_PAGES_PER_BANK",
    "DRAM_PAGE_BYTES",
    "DRAM_STREAM_BYTES_PER_DIRECTION",
    "DRAM_STREAM_TOTAL_BYTES",
    "L1_TOTAL_BUDGET_BYTES",
    "LOAD_CONDITIONS",
    "MAX_CALIBRATION_SECONDS",
    "MAX_MEASUREMENT_TARGET_SECONDS",
    "NS_LOAD_BATCH",
    "NS_LOAD_DOUBLE_BUFFER",
    "NS_LOAD_DST_FULL_SYNC_EN",
    "NS_LOAD_FP32_DEST_ACC_EN",
    "NS_LOAD_FUSE_S",
    "NS_LOAD_ITERATIONS",
    "NS_LOAD_L",
    "NS_LOAD_L1_TOTAL_BYTES",
    "NS_LOAD_MATH_FIDELITY",
    "NS_LOAD_MATRIX_BLOCK",
    "NS_LOAD_MAX_TILES_PER_CORE",
    "NS_LOAD_OUTER_REPETITIONS",
    "NS_LOAD_OUTPUT_MEMORY",
    "NS_LOAD_R_MEMORY",
    "NS_LOAD_SHAPE",
    "NS_LOAD_VARIANT",
    "NS_LOAD_X0_MEMORY",
    "NUM_COMMAND_QUEUES",
    "P150_ALL_CORE_COORDINATES",
    "P150_RESIDUAL_WORKER_COORDINATES",
    "P150_RESIDUAL_WORKER_COUNT",
    "P150_WORKER_GRID",
    "RESIDENT_COMMAND_QUEUE",
    "RESIDENT_CONSUMER_CORE",
    "RESIDENT_CORES",
    "RESIDENT_PRODUCER_CORE",
    "TRANSFER_COMMAND_QUEUE",
    "CalibrationMetadata",
    "CardLoadInterval",
    "CorePartition",
    "HostLoadInterval",
    "LoadConfigurationError",
    "LoadEvidenceError",
    "balanced_work_ranges",
    "build_calibration_load_record",
    "build_load_record",
    "calibration_metadata",
    "choose_repeat_count",
    "dram_achieved_gbps",
    "dram_stream_work",
    "extend_card_clock_timestamps",
    "host_transfer_work",
    "load_evidence_valid",
    "newton_schulz_l1_ledger",
    "normalize_condition",
    "ns_work_distribution",
    "p150_core_partition",
    "repeat_throughput_unit",
    "validate_device_open_settings",
    "validate_load_record",
    "validate_mode_records",
    "validate_worker_model",
]
