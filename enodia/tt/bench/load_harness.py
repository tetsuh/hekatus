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

from enodia.tt.bench.resident_clock import (
    AICLK_SELECTION_FIXED_REFERENCE_1350,
    AICLK_SELECTION_IN_RUN_MINIMUM,
    AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL,
    AICLK_SOURCE_LOAD_INTERVAL_TRACE,
    aiclk_selection_for_source,
    ticks_to_seconds,
)

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
# v0.75.0 Fast Dispatch synchronizes the configured sub-device stall group;
# reserve one residual core so CQ1 can complete without waiting for resident CQ0.
HOST_RESIDENT_SUB_DEVICE_ID = 0
HOST_TRANSFER_SUB_DEVICE_ID = 1
HOST_TRANSFER_CORE = P150_RESIDUAL_WORKER_COORDINATES[0]
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
CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS = 40.0
CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE = AICLK_SOURCE_LOAD_INTERVAL_TRACE
_CARD_LOAD_START_FIELD = "card load start_ticks"
_CARD_LOAD_END_FIELD = "card load end_ticks"
_HOST_LOAD_START_FIELD = "host load start_ns"
_HOST_LOAD_END_FIELD = "host load end_ns"
CARD_THROUGHPUT_REFERENCE_AICLK_MHZ = 1_350
CARD_THROUGHPUT_UNIT_DEFINITION = (
    "load repeats per 1.35e9 card-clock ticks (historical unit name "
    "repeats_per_card_second); convert those ticks to real seconds with the "
    "observed load-interval AICLK"
)
CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE = AICLK_SELECTION_FIXED_REFERENCE_1350

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
_MISSING = object()
_REPEAT_THROUGHPUT_UNITS = {
    # Retained for schema-v1 compatibility. New records carry the explicit
    # CARD_THROUGHPUT_UNIT_DEFINITION alongside this historical unit.
    "newton_schulz": "repeats_per_card_second",
    "dram": "repeats_per_card_second",
    "host_cpu_transfer": "repeats_per_host_second",
}
_DRAM_LOAD_EVIDENCE_INTERVALS_ERROR = (
    "DRAM load evidence must retain all eight worker intervals"
)
_CALIBRATION_CONDITION_MISMATCH_ERROR = (
    "calibration condition does not match the load condition"
)
_CALIBRATION_METADATA_INVALID_ERROR = "calibration metadata is invalid"


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
        "resident_sub_device_id": HOST_RESIDENT_SUB_DEVICE_ID,
        "transfer_sub_device_id": HOST_TRANSFER_SUB_DEVICE_ID,
        "transfer_core": list(HOST_TRANSFER_CORE),
        "completion_scope": "transfer_sub_device_only",
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
    throughput_reference_aiclk_mhz: int = CARD_THROUGHPUT_REFERENCE_AICLK_MHZ
    throughput_reference_aiclk_source: str = CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE
    throughput_unit_definition: str = CARD_THROUGHPUT_UNIT_DEFINITION
    load_card_ticks: int | None = None
    load_measurement_repeat_count: int | None = None
    load_interval_aiclk_mhz: float | None = None
    load_interval_aiclk_min_mhz: float | None = None
    load_interval_aiclk_source: str | None = None
    load_interval_aiclk_selection: str | None = None
    predicted_load_time_aiclk_mhz: float | None = None
    predicted_load_time_aiclk_source: str | None = None
    actual_duration_aiclk_mhz: float | None = None
    actual_duration_aiclk_source: str | None = None
    actual_duration_aiclk_selection: str | None = None

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
        if not isinstance(self.throughput_reference_aiclk_source, str) or not self.throughput_reference_aiclk_source.strip():
            raise LoadConfigurationError(
                "throughput reference AICLK source must be a non-empty string"
            )
        timing_fields = (
            self.load_card_ticks,
            self.load_measurement_repeat_count,
            self.load_interval_aiclk_mhz,
            self.load_interval_aiclk_source,
        )
        if any(value is not None for value in timing_fields):
            if any(value is None for value in timing_fields):
                raise LoadConfigurationError(
                    "calibration load-time metadata must include ticks, repeat count, "
                    "AICLK, and provenance"
                )
            _positive_int(self.load_card_ticks, "load_card_ticks")
            _positive_int(
                self.load_measurement_repeat_count,
                "load_measurement_repeat_count",
            )
            _finite_positive(self.load_interval_aiclk_mhz, "load_interval_aiclk_mhz")
            if self.load_interval_aiclk_min_mhz is not None:
                _finite_positive(
                    self.load_interval_aiclk_min_mhz,
                    "load_interval_aiclk_min_mhz",
                )
            if self.load_interval_aiclk_source != CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE:
                raise LoadConfigurationError(
                    "calibration load-time AICLK requires sampler-on load-interval provenance"
                )
            expected_prediction_aiclk, expected_prediction_source, expected_selection = (
                _prediction_aiclk_selection(
                    self.load_interval_aiclk_mhz,
                    self.load_interval_aiclk_min_mhz,
                )
            )
            if self.load_interval_aiclk_selection not in {None, expected_selection}:
                raise LoadConfigurationError(
                    "load interval AICLK selection does not match weighted/minimum values"
                )
            if self.predicted_load_time_aiclk_mhz is None:
                object.__setattr__(
                    self, "predicted_load_time_aiclk_mhz", expected_prediction_aiclk
                )
            elif not math.isclose(
                float(self.predicted_load_time_aiclk_mhz),
                expected_prediction_aiclk,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise LoadConfigurationError(
                    "predicted load-time AICLK does not match weighted/minimum values"
                )
            if self.predicted_load_time_aiclk_source is None:
                object.__setattr__(
                    self, "predicted_load_time_aiclk_source", expected_prediction_source
                )
            elif self.predicted_load_time_aiclk_source != expected_prediction_source:
                raise LoadConfigurationError(
                    "predicted load-time AICLK source does not match weighted/minimum values"
                )
            if self.load_interval_aiclk_selection is None:
                object.__setattr__(
                    self, "load_interval_aiclk_selection", expected_selection
                )
        if (self.actual_duration_aiclk_mhz is None) != (
            self.actual_duration_aiclk_source is None
        ):
            raise LoadConfigurationError(
                "actual-duration AICLK metadata must include value and provenance"
            )
        if self.actual_duration_aiclk_mhz is not None:
            _finite_positive(
                self.actual_duration_aiclk_mhz, "actual_duration_aiclk_mhz"
            )
            if not isinstance(self.actual_duration_aiclk_source, str) or not self.actual_duration_aiclk_source.strip():
                raise LoadConfigurationError(
                    "actual-duration AICLK source must be a non-empty string"
                )
            expected_selection = aiclk_selection_for_source(
                self.actual_duration_aiclk_source
            )
            if self.actual_duration_aiclk_selection not in {
                None,
                expected_selection,
            }:
                raise LoadConfigurationError(
                    "actual-duration AICLK selection does not match its source"
                )
            if self.actual_duration_aiclk_selection is None:
                object.__setattr__(
                    self, "actual_duration_aiclk_selection", expected_selection
                )
        if (
            self.actual_duration_aiclk_mhz is not None
            and self.load_interval_aiclk_mhz is not None
            and (
                not math.isclose(
                    float(self.actual_duration_aiclk_mhz),
                    float(self.load_interval_aiclk_mhz),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or self.actual_duration_aiclk_source
                != self.load_interval_aiclk_source
            )
        ):
            raise LoadConfigurationError(
                "loaded actual-duration AICLK must use the weighted load interval"
            )
        if self.condition in _DEVICE_LOAD_CONDITIONS:
            if (
                isinstance(self.throughput_reference_aiclk_mhz, bool)
                or not isinstance(self.throughput_reference_aiclk_mhz, int)
                or self.throughput_reference_aiclk_mhz
                != CARD_THROUGHPUT_REFERENCE_AICLK_MHZ
            ):
                raise LoadConfigurationError(
                    "device throughput reference AICLK must be 1350 MHz"
                )
            if self.throughput_unit_definition != CARD_THROUGHPUT_UNIT_DEFINITION:
                raise LoadConfigurationError(
                    "device throughput unit definition is missing or inconsistent"
                )
            if self.throughput_reference_aiclk_source != CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE:
                raise LoadConfigurationError(
                    "device throughput reference AICLK source is inconsistent"
                )

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _prediction_aiclk_selection(
    weighted_aiclk_mhz: float, minimum_aiclk_mhz: float | None
) -> tuple[float, str, str]:
    """Select the conservative prediction clock and persist its provenance."""
    weighted = _finite_positive(weighted_aiclk_mhz, "load_interval_aiclk_mhz")
    if minimum_aiclk_mhz is not None:
        minimum = _finite_positive(minimum_aiclk_mhz, "load_interval_aiclk_min_mhz")
        if minimum < weighted:
            return minimum, AICLK_SELECTION_IN_RUN_MINIMUM, AICLK_SELECTION_IN_RUN_MINIMUM
    return (
        weighted,
        AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL,
        AICLK_SELECTION_TIME_WEIGHTED_LOAD_INTERVAL,
    )


def predicted_load_time_selection(calibration: CalibrationMetadata) -> dict[str, Any]:
    """Return the selected AICLK and source for a calibrated load prediction."""
    if not isinstance(calibration, CalibrationMetadata):
        raise LoadConfigurationError("calibration metadata is required for load-time prediction")
    if calibration.load_interval_aiclk_mhz is None:
        raise LoadConfigurationError(
            "calibration metadata lacks sampler-on load-interval AICLK"
        )
    value, source, selection = _prediction_aiclk_selection(
        calibration.load_interval_aiclk_mhz,
        calibration.load_interval_aiclk_min_mhz,
    )
    return {
        "aiclk_mhz": value,
        "aiclk_source": source,
        "aiclk_selection": selection,
    }


def predicted_load_time_seconds(calibration: CalibrationMetadata) -> float:
    """Convert calibrated card ticks for the selected workload to real seconds."""
    if not isinstance(calibration, CalibrationMetadata):
        raise LoadConfigurationError("calibration metadata is required for load-time prediction")
    if calibration.load_card_ticks is None:
        raise LoadConfigurationError("calibration metadata lacks calibrated load card ticks")
    if calibration.load_measurement_repeat_count is None:
        raise LoadConfigurationError(
            "calibration metadata lacks the measured load repeat count"
        )
    if calibration.load_interval_aiclk_mhz is None:
        raise LoadConfigurationError(
            "calibration metadata lacks sampler-on load-interval AICLK"
        )
    scaled_ticks = calibration.load_card_ticks * (
        calibration.repeat_count / calibration.load_measurement_repeat_count
    )
    prediction = predicted_load_time_selection(calibration)
    return ticks_to_seconds(
        ticks=scaled_ticks,
        selection={
            "aiclk_mhz": prediction["aiclk_mhz"],
            "aiclk_mhz_exact": prediction["aiclk_mhz"],
            "aiclk_source": prediction["aiclk_source"],
        },
    )


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
    load_card_ticks: int | None = None,
    load_measurement_repeat_count: int | None = None,
    load_interval_aiclk_mhz: float | None = None,
    load_interval_aiclk_min_mhz: float | None = None,
    load_interval_aiclk_source: str | None = None,
    actual_duration_aiclk_mhz: float | None = None,
    actual_duration_aiclk_source: str | None = None,
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
        load_card_ticks=load_card_ticks,
        load_measurement_repeat_count=load_measurement_repeat_count,
        load_interval_aiclk_mhz=load_interval_aiclk_mhz,
        load_interval_aiclk_min_mhz=load_interval_aiclk_min_mhz,
        load_interval_aiclk_source=load_interval_aiclk_source,
        actual_duration_aiclk_mhz=actual_duration_aiclk_mhz,
        actual_duration_aiclk_source=actual_duration_aiclk_source,
        actual_duration_aiclk_selection=(
            aiclk_selection_for_source(actual_duration_aiclk_source)
            if actual_duration_aiclk_source is not None
            else None
        ),
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
        _uint64(self.start_ticks, _CARD_LOAD_START_FIELD)
        _uint64(self.end_ticks, _CARD_LOAD_END_FIELD)
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
class CardWorkerInterval:
    """One worker's own wrap-tracked card interval and optional statistics."""

    worker_index: int
    start_ticks: int
    end_ticks: int
    core: tuple[int, int] | None = None
    label: str | None = None
    statistics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.worker_index, "worker_index")
        interval = CardLoadInterval(self.start_ticks, self.end_ticks)
        if self.core is not None and (
            not isinstance(self.core, (tuple, list))
            or len(self.core) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.core
            )
        ):
            raise LoadEvidenceError("worker core must be a non-negative coordinate pair")
        if self.label is not None and (
            not isinstance(self.label, str) or not self.label.strip()
        ):
            raise LoadEvidenceError("worker interval label must be non-empty")
        if self.statistics is not None and not isinstance(self.statistics, Mapping):
            raise LoadEvidenceError("worker interval statistics must be an object")
        object.__setattr__(self, "start_ticks", interval.start_ticks)
        object.__setattr__(self, "end_ticks", interval.end_ticks)

    @property
    def elapsed_ticks(self) -> int:
        return self.end_ticks - self.start_ticks

    def as_record(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "worker_index": self.worker_index,
            "clock": "card_riscv_wall_clock",
            "width_bits": 64,
            "timestamp_semantics": "wrap-tracked_extended",
            "start_ticks": self.start_ticks,
            "end_ticks": self.end_ticks,
            "elapsed_ticks": self.elapsed_ticks,
            "wrap_tracked": True,
        }
        if self.core is not None:
            result["core"] = list(self.core)
        if self.label is not None:
            result["label"] = self.label
        if self.statistics is not None:
            result["statistics"] = dict(self.statistics)
        return result


@dataclass(frozen=True)
class HostLoadInterval:
    """A host monotonic interval used only for condition (c)."""

    start_ns: int
    end_ns: int
    clock: str = "host_monotonic_ns"

    def __post_init__(self) -> None:
        _nonnegative_int(self.start_ns, _HOST_LOAD_START_FIELD)
        _nonnegative_int(self.end_ns, _HOST_LOAD_END_FIELD)
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


def _persisted_card_coverage(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute card coverage from the persisted resident/load endpoints."""
    raw_interval = record.get("card_load_interval")
    if raw_interval is None:
        return _card_coverage(None, None)
    if not isinstance(raw_interval, Mapping):
        return {
            "clock": "card_riscv_wall_clock",
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "card_load_interval_must_be_an_object",
        }
    try:
        interval = CardLoadInterval(
            _uint64(raw_interval.get("start_ticks"), _CARD_LOAD_START_FIELD),
            _uint64(raw_interval.get("end_ticks"), _CARD_LOAD_END_FIELD),
        )
    except (LoadEvidenceError, TypeError) as exc:
        return {
            "clock": "card_riscv_wall_clock",
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": str(exc),
        }
    raw_resident = record.get("resident_frame_interval")
    if not isinstance(raw_resident, Mapping):
        return _card_coverage(interval, None)
    if raw_resident.get("clock") != "card_riscv_wall_clock":
        return {
            "clock": "card_riscv_wall_clock",
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "resident_frame_interval_must_use_card_clock",
        }
    if "first_ticks" not in raw_resident or "last_ticks" not in raw_resident:
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": "missing_resident_frame_interval_endpoint",
        }
    try:
        first = _uint64(raw_resident.get("first_ticks"), "resident first timestamp")
        last = _uint64(raw_resident.get("last_ticks"), "resident last timestamp")
    except LoadEvidenceError as exc:
        return {
            "clock": interval.clock,
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": str(exc),
        }
    return _card_coverage(interval, (first, last))


def _persisted_host_coverage(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute host coverage from the persisted interval and coverage bounds."""
    raw_interval = record.get("host_load_interval")
    declared = record.get("coverage")
    if not isinstance(declared, Mapping):
        return _host_coverage(None, None, None)
    measurement_start = declared.get("measurement_start_ns")
    measurement_end = declared.get("measurement_end_ns")
    if not isinstance(raw_interval, Mapping):
        return _host_coverage(None, measurement_start, measurement_end)
    try:
        interval = HostLoadInterval(
            _nonnegative_int(raw_interval.get("start_ns"), _HOST_LOAD_START_FIELD),
            _nonnegative_int(raw_interval.get("end_ns"), _HOST_LOAD_END_FIELD),
        )
    except (LoadEvidenceError, LoadConfigurationError, TypeError) as exc:
        return {
            "clock": "host_monotonic_ns",
            "valid": False,
            "covered": False,
            "verdict": "invalid",
            "reason": str(exc),
        }
    return _host_coverage(interval, measurement_start, measurement_end)


def _require_declared_coverage_matches(
    declared: Any, recomputed: Mapping[str, Any]
) -> None:
    """Reject persisted coverage flags or facts that disagree with endpoints."""
    if not isinstance(declared, Mapping):
        raise LoadEvidenceError("load coverage metadata is missing or invalid")
    for field, expected in recomputed.items():
        if declared.get(field, _MISSING) != expected:
            raise LoadEvidenceError(
                f"declared load coverage field {field!r} disagrees with persisted endpoints"
            )


def _worker_interval_row(item: Any) -> dict[str, Any]:
    if isinstance(item, CardWorkerInterval):
        return item.as_record()
    if isinstance(item, Mapping):
        return dict(item)
    raise LoadEvidenceError("worker interval rows must be objects")


def _normalize_worker_interval(item: Any) -> dict[str, Any]:
    row = _worker_interval_row(item)
    index = _nonnegative_int(row.get("worker_index"), "worker_index")
    try:
        start_ticks = _uint64(row.get("start_ticks"), _CARD_LOAD_START_FIELD)
        end_ticks = _uint64(row.get("end_ticks"), _CARD_LOAD_END_FIELD)
        interval = CardLoadInterval(start_ticks, end_ticks)
    except (LoadEvidenceError, TypeError) as exc:
        raise LoadEvidenceError(str(exc)) from exc
    if "elapsed_ticks" in row and row.get("elapsed_ticks") != interval.elapsed_ticks:
        raise LoadEvidenceError("worker interval elapsed_ticks is inconsistent")
    for field, expected in (
        ("clock", interval.clock),
        ("width_bits", interval.width_bits),
        ("timestamp_semantics", "wrap-tracked_extended"),
        ("wrap_tracked", True),
    ):
        if field in row and row.get(field) != expected:
            raise LoadEvidenceError(f"worker interval {field} is inconsistent")
    canonical = interval.as_record()
    canonical["worker_index"] = index
    for field in ("core", "label", "statistics", "risc", "profile_worker"):
        if field in row:
            canonical[field] = row[field]
    if "core" in canonical:
        core = canonical["core"]
        valid_core = (
            isinstance(core, (list, tuple))
            and len(core) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in core
            )
        )
        if not valid_core:
            raise LoadEvidenceError("worker core must be a non-negative coordinate pair")
        canonical["core"] = list(core)
    return canonical


def _normalize_worker_intervals(
    worker_intervals: Any,
) -> list[dict[str, Any]]:
    """Normalize per-worker facts without collapsing their independent spans."""
    if worker_intervals is None:
        return []
    if not isinstance(worker_intervals, (list, tuple)):
        raise LoadEvidenceError("worker_intervals must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in worker_intervals:
        canonical = _normalize_worker_interval(item)
        index = canonical["worker_index"]
        if index in seen:
            raise LoadEvidenceError("worker interval indexes must be unique")
        seen.add(index)
        normalized.append(canonical)
    return sorted(normalized, key=lambda row: row["worker_index"])


def _worker_interval_statistics(
    intervals: list[dict[str, Any]], aggregate: CardLoadInterval
) -> dict[str, Any] | None:
    if not intervals:
        return None
    slowest = max(intervals, key=lambda row: row["elapsed_ticks"])
    return {
        "worker_count": len(intervals),
        "aggregation": "slowest_worker_own_elapsed_interval",
        "slowest_worker_index": slowest["worker_index"],
        "slowest_worker_elapsed_ticks": slowest["elapsed_ticks"],
        "aggregate_start_ticks": aggregate.start_ticks,
        "aggregate_end_ticks": aggregate.end_ticks,
        "aggregate_elapsed_ticks": aggregate.elapsed_ticks,
        "aggregate_span_role": "coverage_only",
    }


def _load_record_coverage(
    canonical: str,
    repeat_count: int,
    resident_timestamps: list[int] | tuple[int, ...] | None,
    card_interval: CardLoadInterval | None,
    host_interval: HostLoadInterval | None,
    normalized_workers: list[dict[str, Any]],
    measurement_start_ns: int | None,
    measurement_end_ns: int | None,
) -> dict[str, Any]:
    if canonical == "unloaded":
        if normalized_workers:
            raise LoadConfigurationError("unloaded control cannot claim worker intervals")
        if repeat_count != 0:
            raise LoadConfigurationError("unloaded control must record repeat_count=0")
        if card_interval is not None or host_interval is not None:
            raise LoadConfigurationError("unloaded control cannot claim a load interval")
        return {"verdict": "not_applicable", "valid": True, "covered": True}
    if canonical in _DEVICE_LOAD_CONDITIONS:
        _positive_int(repeat_count, "repeat_count")
        if host_interval is not None:
            raise LoadEvidenceError("device-load coverage cannot use a host interval")
        coverage = _card_coverage(card_interval, resident_timestamps)
    else:
        _positive_int(repeat_count, "repeat_count")
        if card_interval is not None:
            raise LoadEvidenceError("host condition cannot claim a card-clock load interval")
        coverage = _host_coverage(host_interval, measurement_start_ns, measurement_end_ns)
    if not coverage["valid"] or not coverage["covered"]:
        raise LoadEvidenceError(coverage["reason"])
    if canonical in _DEVICE_LOAD_CONDITIONS and normalized_workers:
        if card_interval is None:
            raise LoadEvidenceError("worker intervals require an aggregate card interval")
        if canonical == "dram" and len(normalized_workers) != DRAM_BANK_COUNT:
            raise LoadEvidenceError(_DRAM_LOAD_EVIDENCE_INTERVALS_ERROR)
    return coverage


def _validate_load_calibration(
    canonical: str, repeat_count: int, calibration: CalibrationMetadata | None
) -> None:
    if calibration is None:
        return
    if calibration.condition != canonical:
        raise LoadConfigurationError(_CALIBRATION_CONDITION_MISMATCH_ERROR)
    if canonical != "unloaded" and repeat_count != calibration.repeat_count:
        raise LoadConfigurationError("repeat_count must match calibration-selected repeat_count")


def _validate_achieved_throughput(
    achieved_throughput: float | None, throughput_unit: str | None
) -> None:
    if achieved_throughput is None:
        return
    _finite_positive(achieved_throughput, "achieved_throughput")
    if not isinstance(throughput_unit, str) or not throughput_unit.strip():
        raise LoadConfigurationError("throughput_unit is required with achieved_throughput")


def _add_load_condition_metadata(
    record: dict[str, Any], canonical: str, repeat_count: int
) -> None:
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


def _add_load_throughput_metadata(
    record: dict[str, Any],
    canonical: str,
    achieved_throughput: float | None,
    throughput_unit: str | None,
) -> None:
    if achieved_throughput is None:
        return
    record["achieved_throughput"] = achieved_throughput
    record["throughput_unit"] = throughput_unit
    record["throughput_reference_aiclk_mhz"] = (
        CARD_THROUGHPUT_REFERENCE_AICLK_MHZ
        if canonical in _DEVICE_LOAD_CONDITIONS
        else None
    )
    record["throughput_reference_aiclk_source"] = (
        CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE
        if canonical in _DEVICE_LOAD_CONDITIONS
        else None
    )
    record["throughput_unit_definition"] = (
        CARD_THROUGHPUT_UNIT_DEFINITION
        if canonical in _DEVICE_LOAD_CONDITIONS
        else "host monotonic interval repeats per second"
    )


def _load_record_payload(
    *,
    canonical: str,
    repeat_count: int,
    open_settings: Mapping[str, Any],
    resident_timestamps: list[int] | tuple[int, ...] | None,
    card_interval: CardLoadInterval | None,
    host_interval: HostLoadInterval | None,
    normalized_workers: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    return {
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
        "card_load_worker_intervals": normalized_workers or None,
        "card_load_interval_statistics": (
            _worker_interval_statistics(normalized_workers, card_interval)
            if normalized_workers and card_interval is not None
            else None
        ),
        "throughput_interval": (
            {
                "source": "slowest_worker_own_elapsed_interval",
                "elapsed_ticks": max(row["elapsed_ticks"] for row in normalized_workers),
                "elapsed_seconds_at_reference_aiclk": ticks_to_seconds(
                    ticks=max(row["elapsed_ticks"] for row in normalized_workers),
                    selection={
                        "aiclk_mhz": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
                        "aiclk_mhz_exact": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
                        "aiclk_source": CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
                    },
                ),
                "reference_aiclk_mhz": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
                "reference_aiclk_source": CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
                "aggregate_span_elapsed_ticks": card_interval.elapsed_ticks,
            }
            if normalized_workers and card_interval is not None
            else None
        ),
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


def build_load_record(
    *,
    condition: str,
    repeat_count: int,
    resident_timestamps: list[int] | tuple[int, ...] | None,
    device_open_settings: Mapping[str, Any] | None = None,
    card_interval: CardLoadInterval | None = None,
    host_interval: HostLoadInterval | None = None,
    worker_intervals: list[Mapping[str, Any] | CardWorkerInterval] | None = None,
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
    normalized_workers = _normalize_worker_intervals(worker_intervals)
    coverage = _load_record_coverage(
        canonical,
        repeat_count,
        resident_timestamps,
        card_interval,
        host_interval,
        normalized_workers,
        measurement_start_ns,
        measurement_end_ns,
    )
    _validate_load_calibration(canonical, repeat_count, calibration)
    _validate_achieved_throughput(achieved_throughput, throughput_unit)
    record = _load_record_payload(
        canonical=canonical,
        repeat_count=repeat_count,
        open_settings=open_settings,
        resident_timestamps=resident_timestamps,
        card_interval=card_interval,
        host_interval=host_interval,
        normalized_workers=normalized_workers,
        coverage=coverage,
    )
    _add_load_condition_metadata(record, canonical, repeat_count)
    if calibration is not None:
        record["calibration"] = calibration.as_record()
    _add_load_throughput_metadata(record, canonical, achieved_throughput, throughput_unit)
    validate_load_record(record)
    return record


def _validate_calibration_load_inputs(
    canonical: str,
    repeat_count: int,
    calibration: CalibrationMetadata,
    normalized_workers: list[dict[str, Any]],
) -> None:
    if normalized_workers and canonical not in _DEVICE_LOAD_CONDITIONS:
        raise LoadEvidenceError("host calibration cannot claim card worker intervals")
    if canonical == "dram" and len(normalized_workers) != DRAM_BANK_COUNT:
        raise LoadEvidenceError("DRAM calibration must retain all eight worker intervals")
    if canonical == "unloaded":
        raise LoadConfigurationError("unloaded control does not need calibration")
    if calibration.condition != canonical:
        raise LoadConfigurationError(_CALIBRATION_CONDITION_MISMATCH_ERROR)
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadConfigurationError("calibration measurement repeat_count must be positive")


def _calibration_load_payload(
    *,
    canonical: str,
    repeat_count: int,
    calibration: CalibrationMetadata,
    coverage: dict[str, Any],
    resident_timestamps: list[int] | tuple[int, ...] | None,
    card_interval: CardLoadInterval | None,
    host_interval: HostLoadInterval | None,
    normalized_workers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
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
        "card_load_worker_intervals": normalized_workers or None,
        "card_load_interval_statistics": (
            _worker_interval_statistics(normalized_workers, card_interval)
            if normalized_workers and card_interval is not None
            else None
        ),
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
        "throughput_interval": (
            {
                "source": "slowest_worker_own_elapsed_interval",
                "elapsed_ticks": max(row["elapsed_ticks"] for row in normalized_workers),
                "elapsed_seconds_at_reference_aiclk": ticks_to_seconds(
                    ticks=max(row["elapsed_ticks"] for row in normalized_workers),
                    selection={
                        "aiclk_mhz": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
                        "aiclk_mhz_exact": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
                        "aiclk_source": CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
                    },
                ),
                "reference_aiclk_mhz": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
                "reference_aiclk_source": CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
                "aggregate_span_elapsed_ticks": card_interval.elapsed_ticks,
            }
            if normalized_workers and card_interval is not None
            else None
        ),
    }


def build_calibration_load_record(
    *,
    condition: str,
    repeat_count: int,
    resident_timestamps: list[int] | tuple[int, ...] | None,
    calibration: CalibrationMetadata,
    card_interval: CardLoadInterval | None = None,
    host_interval: HostLoadInterval | None = None,
    worker_intervals: list[Mapping[str, Any] | CardWorkerInterval] | None = None,
    measurement_start_ns: int | None = None,
    measurement_end_ns: int | None = None,
    achieved_throughput: float | None = None,
    throughput_unit: str | None = None,
) -> dict[str, Any]:
    """Build a calibration result without promoting coverage to timing evidence."""
    canonical = normalize_condition(condition)
    normalized_workers = _normalize_worker_intervals(worker_intervals)
    _validate_calibration_load_inputs(
        canonical, repeat_count, calibration, normalized_workers
    )
    coverage = (
        _card_coverage(card_interval, resident_timestamps)
        if canonical in _DEVICE_LOAD_CONDITIONS
        else _host_coverage(host_interval, measurement_start_ns, measurement_end_ns)
    )
    record = _calibration_load_payload(
        canonical=canonical,
        repeat_count=repeat_count,
        calibration=calibration,
        coverage=coverage,
        resident_timestamps=resident_timestamps,
        card_interval=card_interval,
        host_interval=host_interval,
        normalized_workers=normalized_workers,
    )
    _validate_achieved_throughput(achieved_throughput, throughput_unit)
    _add_load_throughput_metadata(record, canonical, achieved_throughput, throughput_unit)
    validate_calibration_load_record(record)
    return record


def _validate_worker_interval_records(
    record: Mapping[str, Any], *, condition: str
) -> None:
    rows = record.get("card_load_worker_intervals")
    if rows is None:
        if condition == "dram":
            raise LoadEvidenceError(_DRAM_LOAD_EVIDENCE_INTERVALS_ERROR)
        return
    normalized = _normalize_worker_intervals(rows)
    if condition == "dram":
        if len(normalized) != DRAM_BANK_COUNT:
            raise LoadEvidenceError(_DRAM_LOAD_EVIDENCE_INTERVALS_ERROR)
        if {row["worker_index"] for row in normalized} != set(range(DRAM_BANK_COUNT)):
            raise LoadEvidenceError("DRAM worker indexes must cover all eight banks")
        load_metadata = record.get("load")
        if not isinstance(load_metadata, Mapping):
            raise LoadEvidenceError("DRAM worker rows require load metadata")
        expected_repeat_count = load_metadata.get("repeat_count")
        if (
            isinstance(expected_repeat_count, bool)
            or not isinstance(expected_repeat_count, int)
            or expected_repeat_count <= 0
        ):
            raise LoadEvidenceError("DRAM worker rows require a positive load repeat_count")
        expected_pages = DRAM_PAGES_PER_BANK * expected_repeat_count
        for row in normalized:
            statistics = row.get("statistics")
            if not isinstance(statistics, Mapping):
                raise LoadEvidenceError("DRAM worker rows require read/write page counters")
            for field, expected in (
                ("repeat_count", expected_repeat_count),
                ("read_pages", expected_pages),
                ("write_pages", expected_pages),
            ):
                actual = statistics.get(field, _MISSING)
                if (
                    isinstance(actual, bool)
                    or not isinstance(actual, int)
                    or actual != expected
                ):
                    raise LoadEvidenceError(
                        f"DRAM worker statistics {field} must equal the fixed load count"
                    )
    aggregate = record.get("card_load_interval")
    if not isinstance(aggregate, Mapping):
        raise LoadEvidenceError("worker intervals require aggregate card interval metadata")
    try:
        aggregate_interval = CardLoadInterval(
            _uint64(aggregate.get("start_ticks"), _CARD_LOAD_START_FIELD),
            _uint64(aggregate.get("end_ticks"), _CARD_LOAD_END_FIELD),
        )
    except (LoadEvidenceError, TypeError) as exc:
        raise LoadEvidenceError(str(exc)) from exc
    starts = [row["start_ticks"] for row in normalized]
    ends = [row["end_ticks"] for row in normalized]
    if aggregate_interval.start_ticks != min(starts) or aggregate_interval.end_ticks != max(ends):
        raise LoadEvidenceError("aggregate card interval must remain separate and cover workers")
    if any(
        row["start_ticks"] < aggregate_interval.start_ticks
        or row["end_ticks"] > aggregate_interval.end_ticks
        for row in normalized
    ):
        raise LoadEvidenceError("worker interval endpoints do not belong to the aggregate load run")
    if condition == "dram":
        expected_cores = P150_RESIDUAL_WORKER_COORDINATES[:DRAM_BANK_COUNT]
        for row in normalized:
            core = row.get("core")
            if core is not None and core != list(expected_cores[row["worker_index"]]):
                raise LoadEvidenceError("DRAM worker core does not match its bank worker")
            statistics = row.get("statistics")
            load_metadata = record.get("load")
            repeat_count = (
                load_metadata.get("repeat_count")
                if isinstance(load_metadata, Mapping)
                else None
            )
            if (
                isinstance(statistics, Mapping)
                and "repeat_count" in statistics
                and statistics.get("repeat_count") != repeat_count
            ):
                raise LoadEvidenceError("DRAM worker statistics belong to another load repeat")
    statistics = record.get("card_load_interval_statistics")
    if not isinstance(statistics, Mapping):
        raise LoadEvidenceError("worker interval statistics are required with worker rows")
    slowest = max(row["elapsed_ticks"] for row in normalized)
    if statistics.get("worker_count") != len(normalized):
        raise LoadEvidenceError("worker statistics count does not match worker rows")
    if statistics.get("slowest_worker_elapsed_ticks") != slowest:
        raise LoadEvidenceError("worker statistics must use the slowest worker own interval")
    if (
        statistics.get("aggregate_start_ticks") != aggregate_interval.start_ticks
        or statistics.get("aggregate_end_ticks") != aggregate_interval.end_ticks
        or statistics.get("aggregate_elapsed_ticks") != aggregate_interval.elapsed_ticks
    ):
        raise LoadEvidenceError("worker statistics aggregate span is inconsistent")
    if statistics.get("aggregate_span_role") != "coverage_only":
        raise LoadEvidenceError("aggregate worker span is coverage-only")
    throughput = record.get("throughput_interval")
    if isinstance(throughput, Mapping):
        if throughput.get("source") != "slowest_worker_own_elapsed_interval":
            raise LoadEvidenceError("throughput interval must use the slowest worker own interval")
        if throughput.get("elapsed_ticks") != slowest:
            raise LoadEvidenceError("throughput interval does not match the slowest worker")
        if "reference_aiclk_mhz" in throughput and throughput.get(
            "reference_aiclk_mhz"
        ) != CARD_THROUGHPUT_REFERENCE_AICLK_MHZ:
            raise LoadEvidenceError("throughput interval reference AICLK must be 1350 MHz")
        if "reference_aiclk_source" in throughput and throughput.get(
            "reference_aiclk_source"
        ) != CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE:
            raise LoadEvidenceError("throughput interval reference AICLK source is invalid")
        if throughput.get("aggregate_span_elapsed_ticks") != aggregate_interval.elapsed_ticks:
            raise LoadEvidenceError("throughput interval aggregate span is inconsistent")


def _validate_calibration_repeat_binding(
    record: Mapping[str, Any],
    *,
    condition: str,
    measurement_repeat_count: int,
    require_calibration: bool,
) -> None:
    """Bind a selected run repeat to its calibration without conflating counts."""
    calibration = record.get("calibration")
    if calibration is None:
        if require_calibration:
            raise LoadEvidenceError("calibration metadata is missing")
        return
    if not isinstance(calibration, Mapping):
        raise LoadEvidenceError(_CALIBRATION_METADATA_INVALID_ERROR)
    try:
        metadata = CalibrationMetadata(**dict(calibration))
    except (TypeError, ValueError) as exc:
        raise LoadEvidenceError(_CALIBRATION_METADATA_INVALID_ERROR) from exc
    if metadata.condition != condition:
        raise LoadEvidenceError(_CALIBRATION_CONDITION_MISMATCH_ERROR)
    selected_repeat_count = record.get("selected_repeat_count")
    if selected_repeat_count != metadata.repeat_count:
        raise LoadEvidenceError(
            "selected_repeat_count must match calibration-selected repeat_count"
        )
    if require_calibration and (
        isinstance(selected_repeat_count, bool)
        or not isinstance(selected_repeat_count, int)
        or selected_repeat_count <= 0
    ):
        raise LoadEvidenceError("calibration selected_repeat_count must be positive")
    # This field describes the calibration run itself.  It is distinct from
    # metadata.repeat_count, which is the selected repeat count for a future
    # loaded run.  Only a calibration record binds it to its own measurement.
    calibration_measurement_repeat_count = metadata.load_measurement_repeat_count
    if require_calibration and (
        calibration_measurement_repeat_count is not None
        and calibration_measurement_repeat_count != measurement_repeat_count
    ):
        raise LoadEvidenceError(
            "calibration measurement repeat_count does not match the persisted measurement"
        )


def _validate_device_fixed_work(load: Mapping[str, Any], condition: str) -> None:
    """Require the fixed-work contract for persisted card-loaded evidence."""
    if condition not in _DEVICE_LOAD_CONDITIONS:
        return
    if load.get("fixed_work") is not True:
        raise LoadEvidenceError("loaded device evidence requires fixed_work=true")
    if load.get("data_dependent_exit") is not False:
        raise LoadEvidenceError(
            "loaded device evidence requires data_dependent_exit=false"
        )


def _validate_unloaded_intervals(record: Mapping[str, Any]) -> None:
    """Keep an unloaded control from carrying any load interval evidence."""
    for field in (
        "card_load_interval",
        "card_load_worker_intervals",
        "card_load_interval_statistics",
        "throughput_interval",
        "host_load_interval",
    ):
        if record.get(field) is not None:
            raise LoadEvidenceError(
                f"unloaded control cannot claim {field.replace('_', ' ')}"
            )


def _calibration_record_header(record: Mapping[str, Any]) -> tuple[str, int]:
    if record.get("schema") != "issue-118-load-calibration-v1" or record.get("issue") != 118:
        raise LoadEvidenceError("record is not an Issue #118 calibration record")
    validate_device_open_settings(
        record.get("device_open") if isinstance(record.get("device_open"), Mapping) else {}
    )
    if record.get("num_command_queues") != NUM_COMMAND_QUEUES:
        raise LoadEvidenceError("calibration record does not retain the required two-CQ setting")
    condition = normalize_condition(record.get("condition"))
    load = record.get("load")
    if not isinstance(load, Mapping) or load.get("condition") != condition:
        raise LoadEvidenceError("calibration load metadata is missing or inconsistent")
    repeat_count = record.get("repeat_count")
    if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadEvidenceError("calibration repeat_count must be positive")
    if load.get("repeat_count") != repeat_count:
        raise LoadEvidenceError("calibration load repeat metadata is inconsistent")
    _validate_device_fixed_work(load, condition)
    _validate_calibration_repeat_binding(
        record,
        condition=condition,
        measurement_repeat_count=repeat_count,
        require_calibration=True,
    )
    return condition, repeat_count


def _validate_calibration_metadata(record: Mapping[str, Any]) -> None:
    calibration = record.get("calibration")
    if not isinstance(calibration, Mapping):
        raise LoadEvidenceError("calibration metadata is missing")
    try:
        CalibrationMetadata(**dict(calibration))
    except (TypeError, ValueError) as exc:
        raise LoadEvidenceError(_CALIBRATION_METADATA_INVALID_ERROR) from exc
    if not isinstance(record.get("coverage"), Mapping):
        raise LoadEvidenceError("calibration coverage metadata is missing or invalid")


def _validate_calibration_intervals(record: Mapping[str, Any], condition: str) -> None:
    if condition in _DEVICE_LOAD_CONDITIONS:
        interval = record.get("card_load_interval")
        if interval is not None:
            CardLoadInterval(
                _uint64(interval.get("start_ticks"), _CARD_LOAD_START_FIELD),
                _uint64(interval.get("end_ticks"), _CARD_LOAD_END_FIELD),
            )
        if record.get("host_load_interval") is not None:
            raise LoadEvidenceError("device calibration cannot claim host interval metadata")
        _validate_worker_interval_records(record, condition=condition)
        return
    interval = record.get("host_load_interval")
    if interval is not None:
        HostLoadInterval(
            _nonnegative_int(interval.get("start_ns"), _HOST_LOAD_START_FIELD),
            _nonnegative_int(interval.get("end_ns"), _HOST_LOAD_END_FIELD),
        )
    if record.get("card_load_interval") is not None:
        raise LoadEvidenceError("host calibration must not claim card interval metadata")


def validate_calibration_load_record(record: Mapping[str, Any]) -> None:
    """Validate a calibration record without requiring full resident coverage."""
    if not isinstance(record, Mapping):
        raise LoadEvidenceError("calibration load record must be an object")
    condition, _repeat_count = _calibration_record_header(record)
    _validate_calibration_metadata(record)
    _validate_calibration_intervals(record, condition)
    recomputed = (
        _persisted_card_coverage(record)
        if condition in _DEVICE_LOAD_CONDITIONS
        else _persisted_host_coverage(record)
    )
    _require_declared_coverage_matches(record.get("coverage"), recomputed)
    if record.get("valid_evidence") is not False or record.get("status") != "calibration":
        raise LoadEvidenceError("calibration record must remain non-evidence")


def _load_record_header(record: Mapping[str, Any]) -> tuple[str, Mapping[str, Any], Any]:
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
        _validate_unloaded_intervals(record)
    elif isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadEvidenceError("loaded conditions must retain a positive fixed repeat_count")
    _validate_device_fixed_work(load, canonical)
    _validate_calibration_repeat_binding(
        record,
        condition=canonical,
        measurement_repeat_count=repeat_count,
        require_calibration=False,
    )
    return canonical, load, repeat_count


def _validate_load_coverage(record: Mapping[str, Any], canonical: str) -> None:
    coverage = record.get("coverage")
    if canonical == "unloaded":
        _require_declared_coverage_matches(
            coverage,
            {"verdict": "not_applicable", "valid": True, "covered": True},
        )
        return
    recomputed = (
        _persisted_card_coverage(record)
        if canonical in _DEVICE_LOAD_CONDITIONS
        else _persisted_host_coverage(record)
    )
    _require_declared_coverage_matches(coverage, recomputed)
    if recomputed.get("valid") is not True:
        raise LoadEvidenceError(recomputed.get("reason", "load coverage is invalid"))
    if recomputed.get("covered") is not True or recomputed.get("verdict") != "covered":
        raise LoadEvidenceError("uncovered load interval is invalid evidence")
    expected_clock = (
        "card_riscv_wall_clock"
        if canonical in _DEVICE_LOAD_CONDITIONS
        else "host_monotonic_ns"
    )
    if recomputed.get("clock") != expected_clock:
        raise LoadEvidenceError("load coverage clock does not match the condition")


def _validate_newton_schulz_work(
    record: Mapping[str, Any], load: Mapping[str, Any], repeat_count: int
) -> None:
    """Validate persisted NS loop counts against the fixed profile and repeats."""
    work = record.get("newton_schulz")
    if not isinstance(work, Mapping):
        raise LoadEvidenceError("Newton-Schulz evidence requires load profile metadata")
    for field, expected in (
        ("iterations", NS_LOAD_ITERATIONS),
        ("outer_repetitions", NS_LOAD_OUTER_REPETITIONS),
    ):
        if work.get(field) != expected:
            raise LoadEvidenceError(
                f"Newton-Schulz {field} must match the fixed load profile"
            )
    expected_effective = NS_LOAD_OUTER_REPETITIONS * repeat_count
    if work.get("effective_outer_repetitions") != expected_effective:
        raise LoadEvidenceError(
            "Newton-Schulz effective_outer_repetitions must match repeat_count"
        )
    if load.get("outer_repetitions") != NS_LOAD_OUTER_REPETITIONS:
        raise LoadEvidenceError(
            "Newton-Schulz load outer_repetitions must match the fixed load profile"
        )

    worker_rows = record.get("card_load_worker_intervals")
    if not isinstance(worker_rows, list):
        return
    for row in worker_rows:
        statistics = row.get("statistics") if isinstance(row, Mapping) else None
        if not isinstance(statistics, Mapping) or "profile_records" not in statistics:
            continue
        profile_records = statistics.get("profile_records")
        if not isinstance(profile_records, list):
            raise LoadEvidenceError("Newton-Schulz profile_records must be a list")
        for profile in profile_records:
            if not isinstance(profile, Mapping):
                raise LoadEvidenceError("Newton-Schulz profile records must be objects")
            if profile.get("risc") in {"TRISC0", "TRISC1", "TRISC2"} and profile.get(
                "warmup_event_count"
            ) != expected_effective:
                raise LoadEvidenceError(
                    "Newton-Schulz profile warmup_event_count must match effective_outer_repetitions"
                )


def _validate_device_load_record(record: Mapping[str, Any], canonical: str) -> None:
    interval = record.get("card_load_interval")
    if not isinstance(interval, Mapping) or interval.get("clock") != "card_riscv_wall_clock":
        raise LoadEvidenceError("device load evidence requires card-clock interval metadata")
    start = _uint64(interval.get("start_ticks"), _CARD_LOAD_START_FIELD)
    end = _uint64(interval.get("end_ticks"), _CARD_LOAD_END_FIELD)
    valid_interval = (
        interval.get("width_bits") == 64
        and interval.get("timestamp_semantics") == "wrap-tracked_extended"
        and interval.get("wrap_tracked") is True
        and interval.get("elapsed_ticks") == end - start
        and end >= start
    )
    if not valid_interval:
        raise LoadEvidenceError("card load interval metadata is not wrap-tracked and extended")
    if record.get("host_load_interval") is not None:
        raise LoadEvidenceError("device-load coverage cannot substitute host interval metadata")
    _validate_worker_interval_records(record, condition=canonical)
    if canonical == "newton_schulz":
        _validate_newton_schulz_work(
            record,
            record["load"],
            record["load"]["repeat_count"],
        )
    if "throughput_reference_aiclk_mhz" in record and record.get(
        "throughput_reference_aiclk_mhz"
    ) != CARD_THROUGHPUT_REFERENCE_AICLK_MHZ:
        raise LoadEvidenceError("device throughput reference AICLK must be 1350 MHz")
    if "throughput_reference_aiclk_source" in record and record.get(
        "throughput_reference_aiclk_source"
    ) != CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE:
        raise LoadEvidenceError(
            "device throughput reference AICLK source must be fixed reference"
        )
    if "throughput_unit_definition" in record and record.get(
        "throughput_unit_definition"
    ) != CARD_THROUGHPUT_UNIT_DEFINITION:
        raise LoadEvidenceError("device throughput unit definition is inconsistent")


def _validate_host_load_record(record: Mapping[str, Any]) -> None:
    interval = record.get("host_load_interval")
    if not isinstance(interval, Mapping) or interval.get("clock") != "host_monotonic_ns":
        raise LoadEvidenceError("host condition requires host interval metadata")
    try:
        start = _nonnegative_int(interval.get("start_ns"), _HOST_LOAD_START_FIELD)
        end = _nonnegative_int(interval.get("end_ns"), _HOST_LOAD_END_FIELD)
    except LoadConfigurationError as exc:
        raise LoadEvidenceError(str(exc)) from exc
    if end < start or interval.get("elapsed_ns") != end - start:
        raise LoadEvidenceError("host load interval elapsed_ns is inconsistent")
    if record.get("card_load_interval") is not None:
        raise LoadEvidenceError("host condition must not claim a device load interval")


def validate_load_record(record: Mapping[str, Any]) -> None:
    """Reject malformed, uncovered, or one-CQ records as condition evidence."""
    if isinstance(record, Mapping) and record.get("schema") == "issue-118-load-calibration-v1":
        validate_calibration_load_record(record)
        return
    if not isinstance(record, Mapping):
        raise LoadEvidenceError("load record must be an object")
    canonical, _load, _repeat_count = _load_record_header(record)
    _validate_load_coverage(record, canonical)
    if canonical in _DEVICE_LOAD_CONDITIONS:
        _validate_device_load_record(record, canonical)
    elif canonical == "host_cpu_transfer":
        _validate_host_load_record(record)
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
    "CALIBRATION_LOAD_INTERVAL_AICLK_SOURCE",
    "CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS",
    "CARD_THROUGHPUT_REFERENCE_AICLK_MHZ",
    "CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE",
    "CARD_THROUGHPUT_UNIT_DEFINITION",
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
    "CardWorkerInterval",
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
    "predicted_load_time_seconds",
    "predicted_load_time_selection",
    "repeat_throughput_unit",
    "validate_calibration_load_record",
    "validate_device_open_settings",
    "validate_load_record",
    "validate_mode_records",
    "validate_worker_model",
]
