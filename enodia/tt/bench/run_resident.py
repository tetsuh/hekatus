"""Run the Issue #12 Stage 1 resident producer/consumer program.

This module is executed inside the pinned toolchain container by
``run_in_container.sh``.  It allocates the ring and output buffers, launches
both kernels once, synchronizes, and only then downloads timestamps.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import stat
import struct
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.load_generators import (
    card_interval_from_profile_records,
    card_interval_from_words,
    merge_program_parts,
    prepare_dram_load,
    prepare_newton_schulz_load,
)
from enodia.tt.bench.load_harness import (
    CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS,
    CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
    CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
    DRAM_STREAM_TOTAL_BYTES,
    HOST_ARITHMETIC_ROUNDS,
    HOST_ARITHMETIC_WORDS,
    HOST_RESIDENT_SUB_DEVICE_ID,
    HOST_TRANSFER_BUFFER_POOL_SIZE,
    HOST_TRANSFER_BYTES,
    HOST_TRANSFER_CORE,
    HOST_TRANSFER_COUNT,
    HOST_TRANSFER_SUB_DEVICE_ID,
    LOAD_CONDITIONS,
    NS_LOAD_MATRIX_BLOCK,
    NS_LOAD_OUTER_REPETITIONS,
    P150_RESIDUAL_WORKER_COORDINATES,
    RESIDENT_COMMAND_QUEUE,
    RESIDENT_CORES,
    TRANSFER_COMMAND_QUEUE,
    CalibrationMetadata,
    CalibrationMetadataOptions,
    CardLoadInterval,
    HostLoadInterval,
    LoadConfigurationError,
    NewtonSchulzCorePlan,
    build_calibration_load_record,
    build_load_record,
    build_newton_schulz_core_plan,
    calibration_metadata,
    predicted_load_time_seconds,
    validate_device_load_repeat_count,
    validate_measurement_target_seconds,
)
from enodia.tt.bench.power_trace_snapshot import (
    ONE_INTERVAL_BOUND,
    PowerTraceSnapshotOptions,
    atomic_power_trace_snapshot,
    empty_power_trace_metadata,
    public_power_trace_metadata,
    read_power_trace_snapshot,
)
from enodia.tt.bench.resident_clock import (
    aiclk_selection_for_source,
    fixed_reference_aiclk_selection,
    safe_aiclk_integer,
    select_elapsed_aiclk,
    select_load_interval_aiclk,
    ticks_to_seconds,
)
from enodia.tt.bench.resident_harness import (
    INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD,
    PAGE_BYTES,
    PAGE_WORDS,
    ResidentConfig,
    ResidentFailureClassification,
    ResidentPreflightError,
    build_measurement_record,
    build_outlier_analysis,
    build_rejection_record,
    run_budget_breakdown,
    select_failure_check,
    split_u64,
    validate_configuration,
    validate_failure_check,
    validate_harness_preflight,
    validate_post_run_aiclk,
    validate_record_inputs,
    validate_resident_allocation_bounds,
    validate_resident_record,
    validate_run_budget_fits_outer_cap,
)
from enodia.tt.bench.resident_tables import (
    build_table_evidence,
    metadata_words,
    table_word,
)
from enodia.tt.bench.resident_tables import (
    enabled as tables_enabled,
)
from enodia.tt.bench.run_binding import (
    RunBindingError,
    current_run_id,
    run_artifact_path,
)

_KERNEL_DIR = Path(__file__).with_name("kernels")
RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX = "resident-power-snapshot"
RAW_TIMESTAMP_OUTPUT_LABEL = "raw timestamp output"


class ResidentResultUnavailable(ValueError):
    """A device result contained a numeric value that cannot be classified."""


class HostCleanupUnsafe(RuntimeError):
    """The CQ1 host thread could not be joined before device cleanup."""


class ResidentResultPathMismatch(ResidentPreflightError):
    """Both result-path inputs were supplied but identify different files."""


@dataclass
class _LoadedDeviceResources:
    """Mutable state owned by one merged resident/load execution."""

    condition: str
    resident_tensors: dict[str, Any] = field(default_factory=dict)
    load_kernel: Any | None = None
    newton_schulz_core_plan: NewtonSchulzCorePlan | None = None
    load_tensors: list[Any] = field(default_factory=list)
    program: Any | None = None
    host_sub_device_manager: Any | None = None
    host_transfer_sub_device_ids: list[Any] | None = None
    resident_completed: bool = False
    host_work: dict[str, Any] | None = None
    host_thread: threading.Thread | None = None
    host_started: threading.Event = field(default_factory=threading.Event)
    resident_launched: threading.Event = field(default_factory=threading.Event)
    host_stop: threading.Event = field(default_factory=threading.Event)
    host_result: dict[str, Any] = field(default_factory=dict)
    host_error: list[Exception] = field(default_factory=list)
    host_start_holder: dict[str, int] = field(default_factory=dict)
    host_device_tensors: list[Any] = field(default_factory=list)
    host_result_tensors: list[Any] = field(default_factory=list)


_safe_aiclk_integer = safe_aiclk_integer


def _device_word(value: Any, name: str) -> int:
    """Read one finite uint32 device word or classify the result unavailable."""
    if isinstance(value, (bool, str, bytes)):
        raise ResidentResultUnavailable(f"{name} is not a device word")
    try:
        numeric = float(value)
        converted = int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ResidentResultUnavailable(f"{name} is not a finite device word") from exc
    if (
        not math.isfinite(numeric)
        or numeric != converted
        or not 0 <= converted <= 0xFFFFFFFF
    ):
        raise ResidentResultUnavailable(f"{name} is not a finite uint32 device word")
    return converted


def _core(value: str) -> tuple[int, int]:
    try:
        x, y = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("core must be written as X,Y") from exc
    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError("core coordinates must be non-negative")
    return x, y


def _core_range(ttnn: Any, core: tuple[int, int]):
    coord = ttnn.CoreCoord(core[0], core[1])
    return ttnn.CoreRangeSet([ttnn.CoreRange(coord, coord)])


def _core_ranges(ttnn: Any, cores: tuple[tuple[int, int], ...]):
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in cores
        ]
    )


def _configure_host_transfer_subdevices(ttnn: Any, device: Any):
    """Partition resident and transfer dispatch so CQ1 does not stall on CQ0."""
    # Pinned v0.75.0 source maps an omitted sub-device list to the configured
    # stall group, whose default is every sub-device.  Queue IDs alone therefore
    # do not make CQ1 completion independent of a resident CQ0 program.  A
    # one-core transfer sub-device is enough because host copies do not launch a
    # worker program; the resident descriptor remains wholly on sub-device 0.
    resident_sub_device = ttnn.SubDevice([_core_ranges(ttnn, RESIDENT_CORES)])
    transfer_sub_device = ttnn.SubDevice([_core_range(ttnn, HOST_TRANSFER_CORE)])
    manager_id = device.create_sub_device_manager(
        [resident_sub_device, transfer_sub_device], 0
    )
    device.load_sub_device_manager(manager_id)
    transfer_sub_device_id = ttnn.SubDeviceId(HOST_TRANSFER_SUB_DEVICE_ID)
    device.set_sub_device_stall_group([transfer_sub_device_id])
    return manager_id, [transfer_sub_device_id]


def _allocate(ttnn: Any, shape, dtype, layout, device, memory_config):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dtype, layout, device, memory_config
    )


def _sharded_pages_config(
    ttnn: Any, config: ResidentConfig, pages: int, *, core: tuple[int, int] | None = None
):
    shape = (1, pages, PAGE_WORDS)
    core = config.consumer_core if core is None else core
    return ttnn.create_sharded_memory_config(
        shape,
        _core_range(ttnn, core),
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# Resident CB indices stay above the 0--19 Newton-Schulz ABI so a merged
# ProgramDescriptor cannot alias a resident queue with a load queue.
RESIDENT_PRODUCER_CB_INDEX = 20
RESIDENT_CONSUMER_CB_INDEX = 21


def _cb(ttnn: Any, *, index: int, core_ranges, total_size: int = PAGE_BYTES):
    format_descriptor = ttnn.CBFormatDescriptor(
        buffer_index=index,
        data_format=ttnn.uint32,
        page_size=PAGE_BYTES,
        tile=ttnn.TileDescriptor(32, 32, False),
    )
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[format_descriptor],
    )


def _runtime_args(ttnn: Any, core: tuple[int, int], values: list[int]):
    args = ttnn.RuntimeArgs()
    args[core[0]][core[1]] = values
    return args


def _runtime_u32(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ResidentPreflightError(f"{name} must fit uint32 runtime argument")
    return int(value)


def _resolve_watcher_mode(watcher_flag: bool, env_value: str | None) -> bool:
    """Require the CLI flag and TT_METAL_WATCHER to select one mode."""
    if env_value is None:
        env_watcher = False
    elif env_value == "1":
        env_watcher = True
    else:
        raise ResidentPreflightError("TT_METAL_WATCHER must be unset or exactly 1")
    if bool(watcher_flag) != env_watcher:
        raise ResidentPreflightError("--watcher and TT_METAL_WATCHER must select the same mode")
    return env_watcher


def _program(
    ttnn: Any,
    config: ResidentConfig,
    tensors: dict[str, Any],
    *,
    watcher: bool,
    return_parts: bool = False,
):
    producer_ranges = _core_range(ttnn, config.producer_core)
    consumer_ranges = _core_range(ttnn, config.consumer_core)
    ring = tensors["ring"]
    control = tensors["control"]
    producer_anchor = tensors["producer_anchor"]
    producer_stats = tensors["producer_stats"]
    consumer_stats = tensors["consumer_stats"]
    timestamps = tensors["timestamps"]
    ring_address = _runtime_u32(ring.buffer_address(), "ring buffer address")
    control_address = _runtime_u32(control.buffer_address(), "control buffer address")
    producer_anchor_address = _runtime_u32(
        producer_anchor.buffer_address(), "producer anchor buffer address"
    )
    producer_stats_address = _runtime_u32(
        producer_stats.buffer_address(), "producer stats buffer address"
    )
    consumer_stats_address = _runtime_u32(
        consumer_stats.buffer_address(), "consumer stats buffer address"
    )
    timestamp_address = _runtime_u32(timestamps.buffer_address(), "timestamp buffer address")

    ring_compile = ttnn.TensorAccessorArgs(ring).get_compile_time_args()
    control_compile = ttnn.TensorAccessorArgs(control).get_compile_time_args()
    producer_stats_compile = ttnn.TensorAccessorArgs(producer_stats).get_compile_time_args()
    producer_anchor_compile = ttnn.TensorAccessorArgs(producer_anchor).get_compile_time_args()
    timestamp_compile = ttnn.TensorAccessorArgs(timestamps).get_compile_time_args()
    consumer_stats_compile = ttnn.TensorAccessorArgs(consumer_stats).get_compile_time_args()

    active = tables_enabled(config)
    metadata = tensors["table_metadata"] if active else producer_stats
    tables = tensors["profile_tables"] if active else consumer_stats
    metadata_compile = ttnn.TensorAccessorArgs(metadata).get_compile_time_args()
    tables_compile = ttnn.TensorAccessorArgs(tables).get_compile_time_args()
    producer_compile = [*ring_compile, *control_compile, *producer_stats_compile, *metadata_compile]
    consumer_compile = [
        *ring_compile,
        *producer_anchor_compile,
        *timestamp_compile,
        *consumer_stats_compile,
        *metadata_compile,
        *tables_compile,
    ]
    run_budget_low, run_budget_high = split_u64(
        run_budget_breakdown(config, watcher=watcher)["run_budget_ticks"]
    )
    producer_args = _runtime_args(
        ttnn,
        config.producer_core,
        [
            ring_address,
            control_address,
            producer_stats_address,
            config.frame_count,
            config.frame_interval_ticks,
            config.ring_pages,
            run_budget_low,
            run_budget_high,
            int(active),
            _runtime_u32(metadata.buffer_address(), "table metadata address"),
        ],
    )
    consumer_args = _runtime_args(
        ttnn,
        config.consumer_core,
        [
            ring_address,
            producer_anchor_address,
            timestamp_address,
            consumer_stats_address,
            config.frame_count,
            config.ring_pages,
            config.work_per_frame,
            config.cycle_budget,
            run_budget_low,
            run_budget_high,
            int(active),
            _runtime_u32(metadata.buffer_address(), "table metadata address"),
            _runtime_u32(tables.buffer_address(), "profile tables address"),
            config.table_bytes // PAGE_BYTES if active else 0,
            int(config.swap_strategy == "prefetch"),
        ],
    )
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str((_KERNEL_DIR / "resident_producer.cpp").resolve()),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=producer_ranges,
            compile_time_args=producer_compile,
            runtime_args=producer_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str((_KERNEL_DIR / "resident_consumer.cpp").resolve()),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=consumer_ranges,
            compile_time_args=consumer_compile,
            runtime_args=consumer_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, consumer_ranges, 0),
        ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, producer_ranges, 0),
        ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, consumer_ranges, 0),
        ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, producer_ranges, 0),
    ]
    cbs = [
        _cb(ttnn, index=RESIDENT_PRODUCER_CB_INDEX, core_ranges=producer_ranges),
        _cb(ttnn, index=RESIDENT_CONSUMER_CB_INDEX, core_ranges=consumer_ranges),
    ]
    if active:
        cbs.extend([
            _cb(ttnn, index=22, core_ranges=producer_ranges),
            _cb(ttnn, index=23, core_ranges=consumer_ranges),
            _cb(ttnn, index=24, core_ranges=consumer_ranges,
                total_size=config.table_bytes * (2 if config.swap_strategy == "prefetch" else 1)),
            _cb(ttnn, index=25, core_ranges=consumer_ranges),
        ])
    if return_parts:
        return kernels, semaphores, cbs
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def _decode_failure(values, *, base: int, source: str) -> dict[str, Any]:
    try:
        code = _device_word(values[base], f"{source} failure code")
        classification = ResidentFailureClassification.for_code(code)
        elapsed_low = _device_word(values[base + 1], f"{source} elapsed low word")
        elapsed_high = _device_word(values[base + 2], f"{source} elapsed high word")
        limit_low = _device_word(values[base + 3], f"{source} limit low word")
        limit_high = _device_word(values[base + 4], f"{source} limit high word")
        valid = _device_word(values[base + 5], f"{source} failure validity")
    except (IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, ResidentResultUnavailable):
            raise
        raise ResidentResultUnavailable(f"{source} failure result is unavailable") from exc
    failure = {
        "code": code,
        "name": classification.name,
        "source": source,
        "elapsed_ticks": elapsed_low | (elapsed_high << 32),
        "limit_ticks": limit_low | (limit_high << 32),
        "unit": "device_clock_ticks",
        "valid": bool(valid),
    }
    try:
        validate_failure_check(failure)
    except (TypeError, ValueError) as exc:
        raise ResidentResultUnavailable(f"{source} failure result is unavailable") from exc
    return failure


def _download(ttnn: Any, tensor):
    host = ttnn.from_device(tensor)
    row_major = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    return row_major.to_numpy()


def _allocate_resident_tensors(ttnn: Any, device: Any, config: ResidentConfig) -> dict[str, Any]:
    """Allocate resident tensors for either the control or merged load run."""
    ring_shape = (1, config.ring_pages, PAGE_WORDS)
    control_shape = (1, 1, PAGE_WORDS)
    timestamp_shape = (config.frame_count, 1, 1, PAGE_WORDS)
    stats_shape = (1, 1, 1, PAGE_WORDS)
    ring = _allocate(
        ttnn,
        ring_shape,
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        _sharded_pages_config(ttnn, config, config.ring_pages),
    )
    control = ttnn.zeros(
        ttnn.Shape(control_shape),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_sharded_pages_config(ttnn, config, 1),
    )
    producer_anchor = ttnn.zeros(
        ttnn.Shape(control_shape),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_sharded_pages_config(ttnn, config, 1, core=config.producer_core),
    )
    timestamps = _allocate(
        ttnn,
        timestamp_shape,
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    producer_stats = _allocate(
        ttnn,
        stats_shape,
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    consumer_stats = _allocate(
        ttnn,
        stats_shape,
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    tensors = {
        "ring": ring,
        "control": control,
        "producer_anchor": producer_anchor,
        "timestamps": timestamps,
        "producer_stats": producer_stats,
        "consumer_stats": consumer_stats,
    }
    if tables_enabled(config):
        import numpy as np

        def preload(values):
            host = ttnn.Tensor(np.asarray(values, dtype=np.uint32).reshape(1, -1, PAGE_WORDS),
                               ttnn.uint32)
            return ttnn.to_device(host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            tensors["table_metadata"] = preload(metadata_words(config))
            # Vectorize host generation; do not build a Python-int list per table.
            # Keep a real accessor even for an all-unknown-identity positive control.
            count = len(config.profile_tables)
            words_per_table = config.table_bytes // 4
            words = np.zeros((max(count, 1), words_per_table), dtype=np.uint32)
            indices = np.arange(words_per_table, dtype=np.uint32)
            for index, (config_id, generation, _) in enumerate(config.profile_tables):
                words[index] = indices ^ table_word((config_id, generation), 0)
            tensors["profile_tables"] = preload(words)
            ttnn.synchronize_device(device, cq_id=0)
        except Exception:
            _deallocate_resident_tensors(ttnn, tensors)
            raise
    return tensors


def _decode_resident_result(ttnn: Any, config: ResidentConfig, tensors: dict[str, Any]) -> dict[str, Any]:
    """Download and classify resident outputs after any merged program run."""
    timestamp_values = _download(ttnn, tensors["timestamps"])
    producer_values = _download(ttnn, tensors["producer_stats"]).reshape(-1)
    consumer_values = _download(ttnn, tensors["consumer_stats"]).reshape(-1)
    producer_failure = _decode_failure(producer_values, base=6, source="producer")
    consumer_failure = _decode_failure(consumer_values, base=7, source="consumer")
    producer_full_count = _device_word(producer_values[0], "producer full count")
    consumer_empty_count = _device_word(consumer_values[0], "consumer empty count")
    frames_attempted = _device_word(producer_values[3], "frames attempted")
    frames_produced = _device_word(producer_values[1], "frames produced")
    frames_dropped = _device_word(producer_values[4], "frames dropped")
    reported_consumed = _device_word(consumer_values[1], "frames consumed")
    if tables_enabled(config) and reported_consumed > config.frame_count:
        raise ResidentResultUnavailable("consumed frame count exceeds configured frame_count")
    frames_consumed = min(reported_consumed, config.frame_count)
    raw_timestamps = [
        _device_word(timestamp_values[index, 0, 0, 0], f"timestamp {index} low word")
        | (_device_word(timestamp_values[index, 0, 0, 1], f"timestamp {index} high word") << 32)
        for index in range(frames_consumed)
    ]
    failure_check = select_failure_check(producer_failure, consumer_failure)
    try:
        failure_classification = validate_failure_check(failure_check)
        producer_error = _device_word(producer_values[2], "producer error flag")
        consumer_error = _device_word(consumer_values[2], "consumer error flag")
        startup_low = _device_word(consumer_values[4], "startup low word")
        startup_high = _device_word(consumer_values[5], "startup high word")
        startup_valid = _device_word(consumer_values[6], "startup validity")
        work_min_low = _device_word(consumer_values[13], "work minimum low word")
        work_min_high = _device_word(consumer_values[14], "work minimum high word")
        work_max_low = _device_word(consumer_values[15], "work maximum low word")
        work_max_high = _device_word(consumer_values[16], "work maximum high word")
        work_valid = _device_word(consumer_values[17], "work validity")
    except (IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, ResidentResultUnavailable):
            raise
        raise ResidentResultUnavailable("resident device result is unavailable") from exc
    kernel_error_flag = int(bool(producer_error or consumer_error))
    if kernel_error_flag != int(failure_classification.error_flag):
        raise ResidentResultUnavailable("kernel summary error flags are unavailable")
    table_evidence = None
    if tables_enabled(config):
        rows = [
            [_device_word(timestamp_values[index, 0, 0, word], f"frame {index} word {word}")
             for word in range(2, 8)]
            for index in range(frames_consumed)
        ]
        summary = [_device_word(value, "table summary word") for value in consumer_values]
        try:
            table_evidence = build_table_evidence(config, rows, raw_timestamps, summary)
        except ValueError as exc:
            raise ResidentResultUnavailable(str(exc)) from exc
    result = {
        "timestamps": raw_timestamps,
        "producer_full_count": producer_full_count,
        "consumer_empty_count": consumer_empty_count,
        "kernel_error_flag": kernel_error_flag,
        "frames_attempted": frames_attempted,
        "frames_produced": frames_produced,
        "frames_dropped": frames_dropped,
        "frames_aborted": frames_attempted - frames_produced - frames_dropped,
        "frames_consumed": frames_consumed,
        "startup_ticks": startup_low | (startup_high << 32),
        "startup_ticks_valid": bool(startup_valid),
        "work_min_ticks": work_min_low | (work_min_high << 32),
        "work_max_ticks": work_max_low | (work_max_high << 32),
        "work_ticks_valid": bool(work_valid),
        "producer_failure": producer_failure,
        "consumer_failure": consumer_failure,
        "failure_check": failure_check,
    }
    if table_evidence is not None:
        result["table_swap"] = table_evidence
    return result


def _deallocate_resident_tensors(ttnn: Any, tensors: dict[str, Any]) -> None:
    for tensor in tensors.values():
        try:
            ttnn.deallocate(tensor)
        except Exception:  # noqa: BLE001, S110 - cleanup must not mask device errors
            pass


def _run_device(
    ttnn: Any, device, config: ResidentConfig, *, watcher: bool
) -> dict[str, Any]:
    tensors = _allocate_resident_tensors(ttnn, device, config)
    try:
        program = _program(ttnn, config, tensors, watcher=watcher)
        program_launch_ns = time.monotonic_ns()
        program_launch_utc = datetime.datetime.now(datetime.UTC).isoformat()
        ttnn.generic_op(list(tensors.values()), program)
        ttnn.synchronize_device(device, cq_id=0)
        program_completion_ns = time.monotonic_ns()
        program_completion_utc = datetime.datetime.now(datetime.UTC).isoformat()
        result = _decode_resident_result(ttnn, config, tensors)
        result.update(
            {
                "program_launch_ns": program_launch_ns,
                "program_completion_ns": program_completion_ns,
                "program_launch_utc": program_launch_utc,
                "program_completion_utc": program_completion_utc,
            }
        )
        return result
    finally:
        _deallocate_resident_tensors(ttnn, tensors)


def _host_load_clock(
    *,
    stop_event: threading.Event | None,
    start_holder: dict[str, int],
    monotonic_ns: Any = time.monotonic_ns,
) -> Any:
    """Return a real host interval clock with a fresh end edge."""
    calls = 0

    def read() -> int:
        nonlocal calls
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("host load stopped before transfer")
        value = monotonic_ns()
        if calls == 0:
            start_holder["start_ns"] = value
        calls += 1
        return value

    return read


def _allocate_host_transfer_buffers(
    ttnn: Any,
    device: Any,
    *,
    device_tensors: list[Any],
    host_tensors: list[Any],
) -> None:
    """Allocate one fixed reusable host/device ring for CQ1 transfers."""
    import numpy as np

    shape = (1, 1, 1, HOST_TRANSFER_BYTES // 4)
    for _ in range(HOST_TRANSFER_BUFFER_POOL_SIZE):
        host_tensor = ttnn.Tensor(np.zeros(shape, dtype=np.uint32), ttnn.uint32)
        host_tensor = ttnn.to_layout(host_tensor, ttnn.ROW_MAJOR_LAYOUT)
        host_tensors.append(host_tensor)
        device_tensors.append(
            _allocate(
                ttnn,
                shape,
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        )


def _queue_host_transfer_batch(
    ttnn: Any,
    device: Any,
    *,
    device_tensors: list[Any],
    host_tensors: list[Any],
    stop_event: threading.Event | None,
) -> None:
    """Queue one fixed CQ1 transfer batch on the reusable ring.

    The pinned v0.75.0 copy APIs enqueue nonblocking work on the selected CQ.
    CQ1 FIFO ordering keeps each slot's D2H complete before its next H2D reuse,
    so an intermediate host synchronization is neither required nor allowed.
    The caller synchronizes once after the complete batch at the repeat boundary.
    """
    if len(device_tensors) != HOST_TRANSFER_BUFFER_POOL_SIZE:
        raise RuntimeError("CQ1 device transfer pool has an unexpected size")
    if len(host_tensors) != HOST_TRANSFER_BUFFER_POOL_SIZE:
        raise RuntimeError("CQ1 host transfer pool has an unexpected size")
    for transfer_index in range(HOST_TRANSFER_COUNT):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("host load stopped while queuing CQ1 transfers")
        slot = transfer_index % HOST_TRANSFER_BUFFER_POOL_SIZE
        ttnn.copy_host_to_device_tensor(
            host_tensors[slot],
            device_tensors[slot],
            cq_id=TRANSFER_COMMAND_QUEUE,
        )
        ttnn.copy_device_to_host_tensor(
            device_tensors[slot],
            host_tensors[slot],
            blocking=False,
            cq_id=TRANSFER_COMMAND_QUEUE,
        )


def _run_host_arithmetic_repeat(
    accumulator: int, stop_event: threading.Event | None
) -> int:
    for _ in range(HOST_ARITHMETIC_ROUNDS):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("host load stopped during CPU work")
        for word in range(HOST_ARITHMETIC_WORDS):
            accumulator = ((accumulator * 33) ^ word) & 0xFFFFFFFF
    return accumulator


def _wait_for_resident_launch(
    resident_launched: threading.Event, stop_event: threading.Event | None
) -> None:
    while not resident_launched.wait(timeout=0.1):
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("host load stopped before queuing CQ1 transfers")


def _synchronize_transfer_queue(
    ttnn: Any, device: Any, transfer_sub_device_ids: list[Any] | None
) -> None:
    if transfer_sub_device_ids is None:
        ttnn.synchronize_device(device, cq_id=TRANSFER_COMMAND_QUEUE)
    else:
        ttnn.synchronize_device(
            device,
            cq_id=TRANSFER_COMMAND_QUEUE,
            sub_device_ids=transfer_sub_device_ids,
        )


def _run_host_transfer_repetitions(
    ttnn: Any,
    device: Any,
    *,
    repetitions: int,
    device_tensors: list[Any],
    host_tensors: list[Any],
    resident_launched: threading.Event | None,
    stop_event: threading.Event | None,
    transfer_sub_device_ids: list[Any] | None,
) -> int:
    accumulator = 0
    for repetition in range(repetitions):
        accumulator = _run_host_arithmetic_repeat(accumulator, stop_event)
        if repetition == 0 and resident_launched is not None:
            # Arithmetic starts before GenericOp, but keep the first CQ1 batch
            # behind its submission so the transfer path cannot serialize CQ0.
            _wait_for_resident_launch(resident_launched, stop_event)
        _queue_host_transfer_batch(
            ttnn,
            device,
            device_tensors=device_tensors,
            host_tensors=host_tensors,
            stop_event=stop_event,
        )
        # Same-CQ FIFO protects the four reusable slots. This is the only
        # synchronization in a repeat and establishes its completion boundary.
        _synchronize_transfer_queue(ttnn, device, transfer_sub_device_ids)
    return accumulator


def _run_host_transfers(
    ttnn: Any,
    device: Any,
    *,
    repetitions: int = 1,
    started_event: threading.Event | None = None,
    resident_launched: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    start_holder: dict[str, int] | None = None,
    device_tensors_holder: list[Any] | None = None,
    host_tensors_holder: list[Any] | None = None,
    transfer_sub_device_ids: list[Any] | None = None,
) -> dict[str, Any]:
    """Run fixed arithmetic, one CQ1 batch, and one CQ1 completion per repeat."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise LoadConfigurationError("host repetitions must be a positive integer")
    device_tensors = device_tensors_holder if device_tensors_holder is not None else []
    host_tensors = host_tensors_holder if host_tensors_holder is not None else []
    if start_holder is None:
        start_holder = {}
    _allocate_host_transfer_buffers(
        ttnn,
        device,
        device_tensors=device_tensors,
        host_tensors=host_tensors,
    )
    start_holder["start_ns"] = time.monotonic_ns()
    if started_event is not None:
        # The main thread submits CQ0 after this edge. The first fixed CPU
        # arithmetic section therefore overlaps resident GenericOp launch;
        # CQ1 is not populated before the launch boundary.
        started_event.set()
    accumulator = _run_host_transfer_repetitions(
        ttnn,
        device,
        repetitions=repetitions,
        device_tensors=device_tensors,
        host_tensors=host_tensors,
        resident_launched=resident_launched,
        stop_event=stop_event,
        transfer_sub_device_ids=transfer_sub_device_ids,
    )
    end_ns = time.monotonic_ns()
    return {
        "arithmetic_rounds": HOST_ARITHMETIC_ROUNDS,
        "arithmetic_words": HOST_ARITHMETIC_WORDS,
        "transfer_bytes": HOST_TRANSFER_BYTES,
        "transfer_count": HOST_TRANSFER_COUNT * repetitions,
        "command_queue": TRANSFER_COMMAND_QUEUE,
        "outer_repetitions": repetitions,
        "start_ns": start_holder["start_ns"],
        "end_ns": end_ns,
        "elapsed_ns": end_ns - start_holder["start_ns"],
        "arithmetic_checksum": accumulator,
        "transfer_completion": "per_repeat_cq1",
        "buffer_pool_size": HOST_TRANSFER_BUFFER_POOL_SIZE,
    }


def _host_join_timeout_seconds(*, repeat_count: int, outer_timeout_seconds: int) -> float:
    """Bound host-load cleanup by the configured resident run cap."""
    if repeat_count <= 0 or outer_timeout_seconds <= 0:
        raise LoadConfigurationError("host join timeout inputs must be positive")
    # The configured outer cap is the only validated upper bound available
    # without relying on an untrusted throughput estimate.
    return float(outer_timeout_seconds)


def _merged_program_io_tensors(resident_tensors: dict[str, Any]) -> list[Any]:
    """Supply a homogeneous GenericOp IO pair for custom merged programs.

    GenericOp uses only the final IO tensor for its output specification; all
    custom kernel addresses are explicit RuntimeArgs. The pair is therefore a
    deliberate uint32 resident tensor duplicate, avoiding v0.75.0 mixed-dtype
    GenericOp validation while retaining every allocated load/resident tensor.
    """
    consumer_stats = resident_tensors["consumer_stats"]
    return [consumer_stats, consumer_stats]


def _complete_host_transfers(
    ttnn: Any,
    device: Any,
    device_tensors: list[Any],
    host_tensors: list[Any],
    *,
    raise_sync_error: bool = True,
    transfer_sub_device_ids: list[Any] | None = None,
) -> None:
    """Complete CQ1 work and release every fixed pool buffer."""
    sync_error: Exception | None = None
    if device_tensors:
        try:
            if transfer_sub_device_ids is None:
                ttnn.synchronize_device(device, cq_id=TRANSFER_COMMAND_QUEUE)
            else:
                ttnn.synchronize_device(
                    device,
                    cq_id=TRANSFER_COMMAND_QUEUE,
                    sub_device_ids=transfer_sub_device_ids,
                )
        except Exception as exc:  # noqa: BLE001 - cleanup still releases both pools
            sync_error = exc
    for tensor in [*device_tensors, *host_tensors]:
        try:
            ttnn.deallocate(tensor)
        except Exception:  # noqa: BLE001, S110 - cleanup must not mask device errors
            pass
    device_tensors.clear()
    host_tensors.clear()
    if sync_error is not None and raise_sync_error:
        raise sync_error


def _prepare_loaded_device(
    ttnn: Any,
    device: Any,
    config: ResidentConfig,
    resources: _LoadedDeviceResources,
    *,
    repeat_count: int,
    watcher: bool,
    newton_schulz_core_plan: NewtonSchulzCorePlan | None = None,
) -> None:
    if resources.condition == "host_cpu_transfer":
        (
            resources.host_sub_device_manager,
            resources.host_transfer_sub_device_ids,
        ) = _configure_host_transfer_subdevices(ttnn, device)
    resources.resident_tensors = _allocate_resident_tensors(ttnn, device, config)
    resident_parts = _program(
        ttnn, config, resources.resident_tensors, watcher=watcher, return_parts=True
    )
    load_parts: tuple[list[Any], list[Any], list[Any]] = ([], [], [])
    if resources.condition == "newton_schulz":
        if newton_schulz_core_plan is None:
            newton_schulz_core_plan = build_newton_schulz_core_plan(
                core_count=len(P150_RESIDUAL_WORKER_COORDINATES)
            )
        load_kernel = prepare_newton_schulz_load(
            ttnn,
            device,
            repeat_count=repeat_count,
            profile=True,
            core_plan=newton_schulz_core_plan,
        )
        resources.newton_schulz_core_plan = newton_schulz_core_plan
        resources.load_kernel = load_kernel
        load_parts = load_kernel.program_parts()
        load_tensors = resources.load_tensors = [*load_kernel.inputs]
        if load_kernel.profile_output is not None:
            load_tensors.append(load_kernel.profile_output)
            load_tensors.extend(load_kernel.outputs[:2])
        else:
            load_tensors.extend(load_kernel.outputs)
    elif resources.condition == "dram":
        load_kernel = prepare_dram_load(ttnn, device, repeat_count=repeat_count)
        resources.load_kernel = load_kernel
        load_parts = load_kernel.program_parts()
        resources.load_tensors = [
            load_kernel.read_buffer,
            load_kernel.write_buffer,
            load_kernel.stats_buffer,
        ]
    resources.program = merge_program_parts(ttnn, resident_parts, load_parts)


def _run_loaded_host_thread(
    ttnn: Any, device: Any, resources: _LoadedDeviceResources, repeat_count: int
) -> None:
    try:
        resources.host_result.update(
            _run_host_transfers(
                ttnn,
                device,
                started_event=resources.host_started,
                resident_launched=resources.resident_launched,
                stop_event=resources.host_stop,
                start_holder=resources.host_start_holder,
                device_tensors_holder=resources.host_device_tensors,
                host_tensors_holder=resources.host_result_tensors,
                repetitions=repeat_count,
                transfer_sub_device_ids=resources.host_transfer_sub_device_ids,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        resources.host_error.append(exc)
    finally:
        resources.host_started.set()


def _start_loaded_host_thread(
    ttnn: Any, device: Any, resources: _LoadedDeviceResources, repeat_count: int
) -> None:
    resources.host_thread = threading.Thread(
        target=_run_loaded_host_thread,
        args=(ttnn, device, resources, repeat_count),
        name="issue118-cq1-host-load",
    )
    resources.host_thread.start()
    if not resources.host_started.wait(timeout=60.0):
        raise RuntimeError("host transfers did not queue")
    if resources.host_error:
        raise RuntimeError("host transfer queueing failed") from resources.host_error[0]


def _synchronize_loaded_resident(
    ttnn: Any, device: Any, resources: _LoadedDeviceResources
) -> None:
    if resources.host_transfer_sub_device_ids is None:
        ttnn.synchronize_device(device, cq_id=RESIDENT_COMMAND_QUEUE)
    else:
        ttnn.synchronize_device(
            device,
            cq_id=RESIDENT_COMMAND_QUEUE,
            sub_device_ids=[ttnn.SubDeviceId(HOST_RESIDENT_SUB_DEVICE_ID)],
        )


def _execute_loaded_program(
    ttnn: Any,
    device: Any,
    resources: _LoadedDeviceResources,
) -> dict[str, Any]:
    measurement_start_ns = time.monotonic_ns()
    measurement_start_utc = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        resident_tensors = resources.resident_tensors
        ttnn.generic_op(_merged_program_io_tensors(resident_tensors), resources.program)
    except BaseException:
        resources.host_stop.set()
        resources.resident_launched.set()
        raise
    finally:
        resources.resident_launched.set()
    _synchronize_loaded_resident(ttnn, device, resources)
    resources.resident_completed = True
    # Capture the CQ0 resident endpoint before waiting for any host-load tail.
    measurement_end_ns = time.monotonic_ns()
    measurement_end_utc = datetime.datetime.now(datetime.UTC).isoformat()
    return {
        "measurement_start_ns": measurement_start_ns,
        "measurement_end_ns": measurement_end_ns,
        "program_launch_ns": measurement_start_ns,
        "program_completion_ns": measurement_end_ns,
        "program_launch_utc": measurement_start_utc,
        "program_completion_utc": measurement_end_utc,
    }


def _finish_loaded_host(
    ttnn: Any,
    device: Any,
    config: ResidentConfig,
    resources: _LoadedDeviceResources,
    repeat_count: int,
) -> None:
    if resources.host_thread is None:
        return
    join_timeout = _host_join_timeout_seconds(
        repeat_count=repeat_count,
        outer_timeout_seconds=config.outer_timeout_seconds,
    )
    resources.host_thread.join(timeout=join_timeout)
    if resources.host_thread.is_alive():
        raise RuntimeError("host load transfer thread did not finish")
    if resources.host_error:
        raise RuntimeError("host load transfer failed") from resources.host_error[0]
    _complete_host_transfers(
        ttnn,
        device,
        resources.host_device_tensors,
        resources.host_result_tensors,
        transfer_sub_device_ids=resources.host_transfer_sub_device_ids,
    )
    resources.host_work = resources.host_result


def _record_newton_schulz_load(
    result: dict[str, Any], resources: _LoadedDeviceResources
) -> None:
    profile_records = resources.load_kernel.profile_records()
    result["card_load_interval"] = card_interval_from_profile_records(profile_records)
    profile_worker = next(
        (
            row
            for row in profile_records
            if row.get("risc") == "TRISC0"
            and row.get("load_end_ticks", 0) != row.get("load_start_ticks", 0)
        ),
        None,
    )
    if profile_worker is None:
        raise RuntimeError("NS profile has no profile worker interval")
    result["card_load_worker_intervals"] = [
        {
            "worker_index": profile_worker.get(
                "core_index", profile_worker.get("measurement_core", 0)
            ),
            "core_index": profile_worker.get("core_index"),
            "risc": "TRISC0",
            "profile_worker": True,
            "start_ticks": profile_worker["load_start_ticks"],
            "end_ticks": profile_worker["load_end_ticks"],
            "statistics": {"profile_records": profile_records},
        }
    ]


def _record_dram_load(
    result: dict[str, Any],
    resources: _LoadedDeviceResources,
    repeat_count: int,
) -> None:
    words = resources.load_kernel.interval_words()
    rows = [
        words[index * PAGE_WORDS : (index + 1) * PAGE_WORDS]
        for index in range(len(resources.load_kernel.worker_coordinates))
    ]
    if any(
        len(row) < 8
        or row[4] != 256 * repeat_count
        or row[5] != 256 * repeat_count
        or row[6] != repeat_count
        for row in rows
    ):
        raise RuntimeError("DRAM load stats did not cover fixed repetitions")
    intervals = [card_interval_from_words(row[:4]) for row in rows]
    result["card_load_interval"] = CardLoadInterval(
        min(interval.start_ticks for interval in intervals),
        max(interval.end_ticks for interval in intervals),
    )
    result["card_load_worker_intervals"] = [
        {
            "worker_index": index,
            "core": list(resources.load_kernel.worker_coordinates[index]),
            "start_ticks": interval.start_ticks,
            "end_ticks": interval.end_ticks,
            "statistics": {
                "read_pages": rows[index][4],
                "write_pages": rows[index][5],
                "repeat_count": rows[index][6],
                "checksum": rows[index][7],
            },
        }
        for index, interval in enumerate(intervals)
    ]
    result["dram_stats"] = rows


def _record_loaded_result(
    result: dict[str, Any],
    resources: _LoadedDeviceResources,
    *,
    repeat_count: int,
) -> None:
    if resources.condition == "newton_schulz":
        _record_newton_schulz_load(result, resources)
    elif resources.condition == "dram":
        _record_dram_load(result, resources, repeat_count)
    else:
        result["host_load_interval"] = HostLoadInterval(
            resources.host_work["start_ns"], resources.host_work["end_ns"]
        )


def _cleanup_loaded_host(
    ttnn: Any,
    device: Any,
    config: ResidentConfig,
    resources: _LoadedDeviceResources,
    repeat_count: int,
) -> None:
    if resources.host_thread is None:
        return
    resources.host_stop.set()
    resources.resident_launched.set()
    cleanup_timeout = _host_join_timeout_seconds(
        repeat_count=repeat_count,
        outer_timeout_seconds=config.outer_timeout_seconds,
    )
    resources.host_thread.join(timeout=cleanup_timeout)
    if resources.host_thread.is_alive():
        raise HostCleanupUnsafe(
            "CQ1 host thread remained active; device cleanup was not attempted"
        )
    _complete_host_transfers(
        ttnn,
        device,
        resources.host_device_tensors,
        resources.host_result_tensors,
        raise_sync_error=False,
        transfer_sub_device_ids=resources.host_transfer_sub_device_ids,
    )


def _cleanup_loaded_device(
    ttnn: Any, device: Any, resources: _LoadedDeviceResources
) -> None:
    if resources.load_kernel is not None:
        resources.load_kernel.close()
    if resources.host_sub_device_manager is None:
        _deallocate_resident_tensors(ttnn, resources.resident_tensors)
        return
    try:
        if not resources.resident_completed:
            ttnn.synchronize_device(
                device,
                cq_id=RESIDENT_COMMAND_QUEUE,
                sub_device_ids=[ttnn.SubDeviceId(HOST_RESIDENT_SUB_DEVICE_ID)],
            )
    finally:
        _deallocate_resident_tensors(ttnn, resources.resident_tensors)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(resources.host_sub_device_manager)


def _run_loaded_device(
    ttnn: Any,
    device: Any,
    config: ResidentConfig,
    *,
    condition: str,
    repeat_count: int,
    watcher: bool,
    newton_schulz_core_plan: NewtonSchulzCorePlan | None = None,
) -> dict[str, Any]:
    """Launch resident and one selected load group in one ProgramDescriptor."""
    if condition not in {"newton_schulz", "dram", "host_cpu_transfer"}:
        raise LoadConfigurationError(f"unsupported loaded condition: {condition}")
    resources = _LoadedDeviceResources(condition)
    try:
        _prepare_loaded_device(
            ttnn,
            device,
            config,
            resources,
            repeat_count=repeat_count,
            watcher=watcher,
            newton_schulz_core_plan=newton_schulz_core_plan,
        )
        if condition == "host_cpu_transfer":
            _start_loaded_host_thread(ttnn, device, resources, repeat_count)
        timing = _execute_loaded_program(ttnn, device, resources)
        _finish_loaded_host(ttnn, device, config, resources, repeat_count)
        result = {
            **_decode_resident_result(ttnn, config, resources.resident_tensors),
            **timing,
            "host_work": resources.host_work,
        }
        _record_loaded_result(result, resources, repeat_count=repeat_count)
        return result
    finally:
        _cleanup_loaded_host(ttnn, device, config, resources, repeat_count)
        _cleanup_loaded_device(ttnn, device, resources)


def _runner_output_dir() -> Path:
    """Return the wrapper output directory; tests may redirect this seam."""
    return Path("/out")


_RESULT_PATH_ENV = "HEKATUS_TT_RESULT_PATH"
_OUTPUT_PATH_RESOLUTION_ERROR = "path cannot be resolved"
_OUTPUT_PATH_TRUST_ERROR = "must be beneath the current, temporary, or /out directory"


def _trusted_output_roots() -> tuple[Path, ...]:
    return (
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/out").resolve(),
    )


def _safe_output_path(path: Path, *, description: str) -> Path:
    """Return a resolved output path only beneath a trusted output root."""
    try:
        candidate = Path(path)
        resolved_path = candidate.resolve(strict=False)
        trusted_roots = _trusted_output_roots()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ResidentPreflightError(
            f"{description} {_OUTPUT_PATH_RESOLUTION_ERROR}"
        ) from exc
    if not any(resolved_path.is_relative_to(root) for root in trusted_roots):
        raise ResidentPreflightError(f"{description} {_OUTPUT_PATH_TRUST_ERROR}")
    return resolved_path


def _output_write_path(path: Path, *, description: str) -> Path:
    """Validate a destination while retaining its final component for no-follow open."""
    candidate = Path(path)
    _safe_output_path(candidate, description=description)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _open_output_descriptor(path: Path, *, description: str) -> int:
    """Open a trusted output without following a substituted final-component symlink."""
    write_path = _output_write_path(path, description=description)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(
        write_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o666,
    )


def _write_output_bytes(path: Path, payload: bytes, *, description: str) -> None:
    """Write one output through a descriptor that rejects a final symlink."""
    descriptor = _open_output_descriptor(path, description=description)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _output_destination_writable(path: Path) -> bool:
    """Check a destination with a temporary sibling write, without replacing it."""
    try:
        resolved_path = path.resolve(strict=False)
        trusted_roots = _trusted_output_roots()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if not any(resolved_path.is_relative_to(root) for root in trusted_roots):
        return False
    try:
        parent = resolved_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if resolved_path.exists():
            return resolved_path.is_file() and os.access(resolved_path, os.W_OK)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{resolved_path.name}.", dir=parent
        )
        os.close(descriptor)
        Path(temporary_name).unlink()
    except (OSError, ValueError):
        return False
    return True


def _destination_inode(path: Path, *, name: str) -> tuple[int, int] | None:
    """Return an existing destination inode without following its final component."""
    try:
        destination_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResidentPreflightError(
            f"{name} destination cannot be inspected"
        ) from exc
    if stat.S_ISLNK(destination_stat.st_mode):
        raise ResidentPreflightError(f"{name} destination must not be a symlink")
    return destination_stat.st_dev, destination_stat.st_ino


def _validate_existing_destination_inodes(
    destinations: list[tuple[str, Path]],
) -> None:
    """Reject existing destinations that are hard-link aliases."""
    seen: dict[tuple[int, int], str] = {}
    for name, path in destinations:
        inode = _destination_inode(path, name=name)
        if inode is None:
            continue
        previous_name = seen.get(inode)
        if previous_name is not None:
            raise ResidentPreflightError(
                f"{name} collides with {previous_name}: existing destinations "
                "share an inode (hard-link alias)"
            )
        seen[inode] = name


def _validate_destination_collisions(
    destinations: list[tuple[str, Path]],
) -> None:
    """Reject runner artifacts that resolve to equal, nested, or aliased paths."""
    seen: dict[Path, str] = {}
    for name, path in destinations:
        previous_name = seen.get(path)
        if previous_name is not None:
            raise ResidentPreflightError(
                f"{name} collides with {previous_name} at resolved path {path}"
            )
        for previous_path, previous_name in seen.items():
            if path in previous_path.parents or previous_path in path.parents:
                raise ResidentPreflightError(
                    f"{name} collides with {previous_name}: paths are an "
                    f"ancestor/descendant pair ({path} and {previous_path})"
                )
        seen[path] = name
    _validate_existing_destination_inodes(destinations)


def _environment_result_path() -> Path | None:
    """Return the environment result path when the contract supplies one."""
    value = os.environ.get(_RESULT_PATH_ENV)
    if value is None or value == "":
        return None
    return Path(value)


def _resolve_result_path(args: argparse.Namespace) -> Path:
    """Resolve CLI and environment result paths before any device work."""
    cli_path = args.out
    environment_path = _environment_result_path()
    if cli_path is None and environment_path is None:
        raise ResidentPreflightError(
            f"result path is required: provide --out or {_RESULT_PATH_ENV}"
        )
    if cli_path is not None and environment_path is not None:
        safe_cli_path = _safe_output_path(cli_path, description="--out destination")
        safe_environment_path = _safe_output_path(
            environment_path, description=f"{_RESULT_PATH_ENV} destination"
        )
        if safe_cli_path != safe_environment_path:
            raise ResidentResultPathMismatch(
                f"--out and {_RESULT_PATH_ENV} must identify the same result path"
            )
        return safe_cli_path
    selected_path = cli_path if cli_path is not None else environment_path
    assert selected_path is not None
    description = "--out destination" if cli_path is not None else f"{_RESULT_PATH_ENV} destination"
    return _safe_output_path(selected_path, description=description)


def _validate_output_destinations(
    args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    """Resolve output paths and check collisions without mutating their parents."""
    destinations = [("output", args.out)]
    if args.raw_timestamps_out is not None:
        destinations.append((RAW_TIMESTAMP_OUTPUT_LABEL, args.raw_timestamps_out))
    validated_destinations = [
        (name, _safe_output_path(path, description=f"{name} destination"))
        for name, path in destinations
    ]
    _validate_destination_collisions(validated_destinations)
    return (
        validated_destinations[0][1],
        validated_destinations[1][1] if len(validated_destinations) > 1 else None,
    )


def _validate_output_writability(
    output_destinations: tuple[Path, Path | None],
) -> None:
    """Probe output destinations only after all protected paths are collision-checked."""
    output_path, raw_timestamps_path = output_destinations
    destinations = [("output", output_path)]
    if raw_timestamps_path is not None:
        destinations.append((RAW_TIMESTAMP_OUTPUT_LABEL, raw_timestamps_path))
    for name, safe_path in destinations:
        if not _output_destination_writable(safe_path):
            raise ResidentPreflightError(f"{name} destination is not writable")


def _validate_live_power_trace_source(path: Path) -> None:
    """Validate an existing live source without parsing or blocking on it."""
    try:
        source_stat = path.lstat()
    except FileNotFoundError:
        # The sampler creates the source after the runner starts.  Keep the
        # historical post-run missing-source diagnostic for that case.
        return
    except OSError as exc:
        raise RunBindingError("power trace source cannot be inspected") from exc
    if stat.S_ISLNK(source_stat.st_mode):
        raise RunBindingError("power trace source must not be a symlink")
    if not stat.S_ISREG(source_stat.st_mode):
        raise RunBindingError("power trace source must be a regular file")
    try:
        # Opening a regular file does not consume or parse sampler bytes.  The
        # type check above is deliberately before this open so FIFOs cannot
        # block the preflight.
        with path.open("rb") as handle:
            handle.read(0)
    except FileNotFoundError:
        # A sampler can remove or rotate the source between lstat and open;
        # retain the missing-source compatibility path in that race.
        return
    except OSError as exc:
        raise RunBindingError("power trace source is not readable") from exc


def _validate_card_sampler_preflight(
    condition: str, sampler: Mapping[str, Any]
) -> None:
    """Reject card-load records that cannot carry load-interval telemetry."""
    if condition in {"dram", "newton_schulz"} and sampler.get("mode") == "off":
        raise LoadConfigurationError(
            f"{condition} load runs require telemetry sampling; sampler mode off "
            "cannot produce loaded load-interval evidence"
        )


def _validate_power_trace_binding(
    power_trace: str,
    *,
    environment: Mapping[str, Any],
) -> str | None:
    """Validate wrapper identity and return the bound run ID when present."""
    # A caller that invokes the runner directly has no wrapper identity and
    # retains the historical unbound-trace compatibility path.  Once either
    # the process or captured environment carries a wrapper identity, both
    # identities must agree and all run-owned paths are checked before TTNN.
    wrapper_run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    captured_run_id = environment.get("run_id")
    looks_wrapper_bound = (
        isinstance(power_trace, str)
        and power_trace.startswith("power-")
        and power_trace.endswith(".csv")
    )
    if wrapper_run_id is None:
        if captured_run_id is not None or looks_wrapper_bound:
            raise RunBindingError(
                "power trace binding rejected: HEKATUS_TT_RUN_ID is missing"
            )
        return None
    try:
        run_id = current_run_id()
    except ValueError as exc:
        raise RunBindingError(
            "power trace binding rejected: HEKATUS_TT_RUN_ID is invalid"
        ) from exc
    if captured_run_id is not None and captured_run_id != run_id:
        raise RunBindingError(
            "power trace binding rejected: captured run identity does not match"
        )
    return run_id


def _power_trace_artifact_paths(run_id: str) -> tuple[Path, Path]:
    """Return the wrapper-bound live and snapshot artifact paths for one run."""
    runner_output_dir = _runner_output_dir()
    expected_power_trace = run_artifact_path(
        runner_output_dir, prefix="power-", suffix=".csv", run_id=run_id
    )
    snapshot_path = runner_output_dir / (
        f"{RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX}-{run_id}-1.csv"
    )
    return expected_power_trace, snapshot_path


def _protected_power_trace_destinations(
    power_trace: str | None,
    environment: Mapping[str, Any],
) -> tuple[Path | None, Path | None]:
    """Return safe paths that must not receive a rejection record."""
    if power_trace is None:
        return (None, None)
    wrapper_run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    captured_run_id = (
        environment.get("run_id") if isinstance(environment, Mapping) else None
    )
    if wrapper_run_id is None and captured_run_id is None:
        return (None, None)
    run_id = current_run_id() if wrapper_run_id is not None else captured_run_id
    live_path, snapshot_path = _power_trace_artifact_paths(run_id)
    return (
        _safe_output_path(live_path, description="power trace source"),
        _safe_output_path(snapshot_path, description="power trace snapshot destination"),
    )


def _validate_power_trace_preflight(
    power_trace: str | None,
    *,
    environment: Mapping[str, Any],
    output_destinations: tuple[Path, Path | None] | None = None,
) -> None:
    """Validate wrapper-owned trace and snapshot destinations without reading data."""
    if power_trace is None:
        return
    run_id = _validate_power_trace_binding(power_trace, environment=environment)
    if run_id is None:
        return

    expected_power_trace, snapshot_path = _power_trace_artifact_paths(run_id)
    if power_trace != expected_power_trace.name:
        raise RunBindingError(
            "power trace binding rejected: name is not the current run artifact"
        )
    safe_power_trace_path = _safe_output_path(
        expected_power_trace, description="power trace source"
    )
    safe_snapshot_path = _safe_output_path(
        snapshot_path, description="power trace snapshot destination"
    )
    collision_destinations: list[tuple[str, Path]] = []
    if output_destinations is not None:
        output_path, raw_timestamps_path = output_destinations
        collision_destinations.append(("output", output_path))
        if raw_timestamps_path is not None:
            collision_destinations.append((RAW_TIMESTAMP_OUTPUT_LABEL, raw_timestamps_path))
    collision_destinations.extend(
        [
            ("wrapper-bound live power trace source", safe_power_trace_path),
            ("generated power trace snapshot destination", safe_snapshot_path),
        ]
    )
    _validate_destination_collisions(collision_destinations)
    _validate_live_power_trace_source(expected_power_trace)
    if (
        snapshot_path.exists()
        or snapshot_path.is_symlink()
        or safe_snapshot_path.exists()
        or safe_snapshot_path.is_symlink()
    ):
        raise FileExistsError(
            "power trace snapshot installation rejected: destination already exists: "
            f"{safe_snapshot_path.name}"
        )
    if not _output_destination_writable(safe_snapshot_path):
        raise ResidentPreflightError(
            "power trace snapshot destination is not writable"
        )


def _write_rejection_if_possible(
    output_path: Path | None,
    *,
    config: ResidentConfig,
    environment: Mapping[str, Any],
    reason: str,
    power_trace: str | None = None,
    raw_timestamps_path: Path | None = None,
) -> None:
    """Write a rejection only when no protected artifact would be truncated."""
    if output_path is None:
        print(
            f"resident preflight rejected: {reason}; rejection record unavailable",
            file=sys.stderr,
        )
        return
    try:
        safe_output_path = _safe_output_path(
            output_path, description="output destination"
        )
        protected_destinations = [
            ("protected power trace artifact", path)
            for path in _protected_power_trace_destinations(power_trace, environment)
            if path is not None
        ]
        _validate_destination_collisions(
            [("output", safe_output_path), *protected_destinations]
        )
        if raw_timestamps_path is not None:
            try:
                safe_raw_path = _safe_output_path(
                    raw_timestamps_path, description="raw timestamp output destination"
                )
            except ResidentPreflightError:
                safe_raw_path = None
            if safe_raw_path is not None:
                _validate_existing_destination_inodes(
                    [
                        ("output", safe_output_path),
                        (RAW_TIMESTAMP_OUTPUT_LABEL, safe_raw_path),
                    ]
                )
        if _output_destination_writable(safe_output_path):
            _write(
                safe_output_path,
                build_rejection_record(
                    config=config, reason=reason, environment=environment
                ),
            )
        else:
            raise OSError("output destination is not writable")
    except (OSError, RuntimeError, TypeError, ValueError):
        print(
            f"resident preflight rejected: {reason}; rejection record unavailable",
            file=sys.stderr,
        )
    else:
        print(f"resident preflight rejected: {reason}", file=sys.stderr)


def _missing_power_trace_metadata(
    path: Path, *, sampler_interval_seconds: float | None, error: Exception
) -> dict[str, Any]:
    """Describe an absent trace without reading a mutable artifact."""
    trace = empty_power_trace_metadata(
        path,
        coverage_definition=ONE_INTERVAL_BOUND,
        sampler_interval_seconds=sampler_interval_seconds,
        aiclk_source="no_valid_in_run_samples",
        errors=[f"power trace snapshot unavailable: {error}"],
    )
    trace["sampler_interval_seconds"] = sampler_interval_seconds
    trace["sha256_definition"] = None
    return public_power_trace_metadata(trace)


def _environment_output_path() -> Path:
    """Return the wrapper's exact environment artifact for this run."""
    return run_artifact_path(
        _runner_output_dir(), prefix="env-", suffix=".json", run_id=current_run_id()
    )


def _reject_power_trace_run(
    output_path: Path,
    *,
    config: ResidentConfig,
    environment: dict[str, Any],
    reason: str,
) -> int:
    """Persist one diagnostic rejection for an immutable trace failure."""
    _write(
        output_path,
        build_rejection_record(config=config, reason=reason, environment=environment),
    )
    print(f"resident run rejected: {reason}", file=sys.stderr)
    return 2


def _power_trace_aiclk_values(power_trace: str | None) -> list[int]:
    if power_trace is None:
        return []
    path = _runner_output_dir() / power_trace
    values: list[int] = []
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not isinstance(row, dict):
                    continue
                value = _safe_aiclk_integer(row.get("aiclk_mhz"))
                if value is not None:
                    values.append(value)
    except (OSError, csv.Error):
        pass
    return values


def _environment_aiclk_values(environment: dict[str, Any]) -> list[int]:
    observed = environment.get("aiclk_mhz_observed")
    if isinstance(observed, list):
        values = observed
    else:
        values = [observed, environment.get("aiclk_mhz")]
    result: list[int] = []
    for value in values:
        parsed = _safe_aiclk_integer(value)
        if parsed is not None:
            result.append(parsed)
    return result


def _trace_has_sample_at_edge(path: Path, edge: datetime.datetime) -> bool:
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                timestamp_text = row.get("timestamp_utc")
                if not timestamp_text:
                    continue
                timestamp = datetime.datetime.fromisoformat(timestamp_text)
                if timestamp >= edge:
                    return True
    except (OSError, csv.Error, TypeError, ValueError):
        return False
    return False


def _sampler_wait_interval(value: float | None) -> float:
    try:
        interval = float(value or 2.0)
    except (TypeError, ValueError):
        return 2.0
    return interval if math.isfinite(interval) and interval > 0.0 else 2.0


def _wait_for_power_trace_sample(
    path: Path,
    edge_iso: str,
    *,
    sampler_interval_seconds: float | None,
) -> None:
    """Wait for a live sampler row at or after a host UTC edge."""
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RunBindingError("power trace path cannot be resolved") from exc
    trusted_roots = (
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/out").resolve(),
    )
    if not any(resolved_path.is_relative_to(root) for root in trusted_roots):
        raise RunBindingError(
            "power trace must be beneath the current, temporary, or /out directory"
        )
    try:
        edge = datetime.datetime.fromisoformat(edge_iso)
    except ValueError as exc:
        raise RunBindingError("program completion UTC timestamp is invalid") from exc
    interval = _sampler_wait_interval(sampler_interval_seconds)
    deadline = time.monotonic() + interval * 1.5 + 1.0
    while time.monotonic() < deadline:
        if _trace_has_sample_at_edge(resolved_path, edge):
            return
        time.sleep(0.1)
    raise RunBindingError(
        "power trace has no usable sample at or after program completion UTC"
    )


def _power_aiclk(
    power_trace: str | None,
    environment: dict[str, Any],
    *,
    allow_environment_snapshot: bool = False,
) -> int | None:
    """Return a trace-only AICLK for legacy callers; never use snapshots."""
    del environment, allow_environment_snapshot
    values = _power_trace_aiclk_values(power_trace)
    if not values:
        return None
    selection = select_elapsed_aiclk(
        budget_aiclk_mhz=values[0],
        sampler_mode="default",
        trace_aiclk_mhz=max(values),
        trace_aiclk_source="run_trace_samples",
    )
    return selection["aiclk_mhz"]


def _environment(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read environment JSON: {path.name}: {exc}") from exc


def _write_raw_timestamps(path: Path, timestamps: list[int]) -> None:
    _write_output_bytes(
        path,
        b"".join(struct.pack("<Q", value) for value in timestamps),
        description="raw timestamp output destination",
    )


def _write(path: Path, payload: dict[str, Any]) -> None:
    _write_output_bytes(
        path,
        (strict_json_dumps(payload, indent=2) + "\n").encode("utf-8"),
        description="output destination",
    )


def _newton_schulz_core_plan_from_args(
    args: argparse.Namespace,
) -> NewtonSchulzCorePlan | None:
    """Build an explicit NS core plan before any TTNN import or device open."""
    core_count = getattr(args, "load_core_count", None)
    total_blocks = getattr(args, "load_total_blocks", None)
    matrix_block = getattr(args, "load_matrix_block", None)
    if args.condition != "newton_schulz":
        if any(value is not None for value in (core_count, total_blocks, matrix_block)):
            raise LoadConfigurationError(
                "--load-core-count, --load-total-blocks, and --load-matrix-block "
                "require --condition newton_schulz"
            )
        return None
    if all(value is None for value in (core_count, total_blocks, matrix_block)):
        # Preserve the historical Q2 record shape.  The loaded execution path
        # constructs its unchanged 108-core default when no Issue #137
        # selection was supplied.
        return None
    return build_newton_schulz_core_plan(
        core_count=(
            len(P150_RESIDUAL_WORKER_COORDINATES)
            if core_count is None
            else core_count
        ),
        total_blocks=total_blocks,
        matrix_block=(NS_LOAD_MATRIX_BLOCK if matrix_block is None else matrix_block),
    )


def _config_from_args(args: argparse.Namespace) -> ResidentConfig:
    """Build the resident configuration without importing or opening TTNN."""
    timestamp_core = args.timestamp_core or args.consumer_core
    return ResidentConfig(
        frame_count=args.frame_count,
        frame_interval_ticks=args.frame_interval_ticks,
        producer_core=args.producer_core,
        consumer_core=args.consumer_core,
        ring_pages=args.ring_pages,
        work_per_frame=args.work_per_frame,
        designated_timestamp_core=timestamp_core,
        cycle_budget=args.cycle_budget,
        outer_timeout_seconds=args.outer_timeout_seconds,
        fixed_work_ticks_per_frame=args.fixed_work_ticks_per_frame,
        budget_aiclk_mhz=args.budget_aiclk_mhz,
        histogram_bin_ticks=args.histogram_bin_ticks,
        profile_tables=tuple(args.profile_table),
        switch_schedule=tuple(args.switch),
        initial_identity=args.initial_identity,
        table_bytes=args.table_bytes,
        swap_strategy=args.swap_strategy,
        outer_timeout_cap_seconds=(
            800
            if args.condition == "newton_schulz" and args.outer_timeout_seconds == 800
            else 600
        ),
        outer_timeout_cap_source=(
            "issue118_compute_load_800s"
            if args.condition == "newton_schulz" and args.outer_timeout_seconds == 800
            else "issue12_timing_600s"
        ),
    )


def _identity_words(value: str, count: int) -> tuple[int, ...]:
    try:
        words = tuple(int(word) for word in value.split(":"))
        if len(words) != count:
            raise ValueError
        return words
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected {count} colon-separated integers") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env-json", type=Path, default=None)
    parser.add_argument("--power-trace", default=None)
    parser.add_argument("--raw-timestamps-out", type=Path, default=None)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--frame-interval-ticks", type=int, default=1_350_000)
    parser.add_argument("--producer-core", type=_core, default=(0, 0))
    parser.add_argument("--consumer-core", type=_core, default=(1, 0))
    parser.add_argument("--ring-pages", type=int, default=4)
    parser.add_argument("--work-per-frame", type=int, default=64)
    parser.add_argument("--timestamp-core", type=_core, default=None)
    parser.add_argument("--cycle-budget", type=int, default=10_000_000)
    parser.add_argument("--fixed-work-ticks-per-frame", type=int, default=100_000)
    parser.add_argument("--budget-aiclk-mhz", type=int, default=1_350)
    parser.add_argument("--outer-timeout-seconds", type=int, default=60)
    parser.add_argument("--histogram-bin-ticks", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--condition", choices=LOAD_CONDITIONS, default="unloaded")
    parser.add_argument("--load-core-count", type=int, default=None)
    parser.add_argument("--load-total-blocks", type=int, default=None)
    parser.add_argument("--load-matrix-block", type=int, default=None)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-duration-seconds", type=float, default=60.0)
    parser.add_argument("--target-seconds", type=float, default=500.0)
    parser.add_argument("--profile-table", type=lambda value: _identity_words(value, 3),
                        action="append", default=[], metavar="CONFIG:GENERATION:READY_FRAME")
    parser.add_argument("--switch", type=lambda value: _identity_words(value, 3),
                        action="append", default=[], metavar="FRAME:CONFIG:GENERATION")
    parser.add_argument("--initial-identity", type=lambda value: _identity_words(value, 2),
                        default=(0, 0), metavar="CONFIG:GENERATION")
    parser.add_argument("--table-bytes", type=int, default=0)
    parser.add_argument("--swap-strategy", choices=("copy_on_demand", "prefetch"),
                        default="copy_on_demand")
    parser.add_argument("--watcher", action="store_true")
    return parser

def _replace_resident_config(config: ResidentConfig, **changes: Any) -> ResidentConfig:
    return cast(ResidentConfig, replace(config, **changes))


def _bound_calibration_config(config: ResidentConfig, duration_seconds: float) -> ResidentConfig:
    """Bound calibration frames to a conservative sub-60-second run budget."""
    if duration_seconds <= 0.0 or duration_seconds > 60.0:
        raise LoadConfigurationError("calibration-duration-seconds must be in (0, 60]")
    # Reserve 10% for scheduling variance and keep the effective schedule at
    # or below 54 seconds even when the caller requests the full 60-second cap.
    target_seconds = min(duration_seconds * 0.90, 54.0)
    target_ticks = int(target_seconds * config.budget_aiclk_mhz * 1_000_000)
    startup_ticks = 100_000 * config.budget_aiclk_mhz
    usable_ticks = (target_ticks * 100) // 110
    candidate = max(
        1,
        (usable_ticks - startup_ticks) // config.frame_interval_ticks,
    )
    bounded = _replace_resident_config(config, frame_count=candidate, outer_timeout_seconds=60)
    while candidate > 1:
        budget = run_budget_breakdown(bounded, watcher=False)["run_budget_ticks"]
        if budget <= target_ticks:
            break
        candidate -= 1
        bounded = _replace_resident_config(bounded, frame_count=candidate)
    if run_budget_breakdown(bounded, watcher=False)["run_budget_ticks"] > target_ticks:
        raise LoadConfigurationError("calibration duration is shorter than one safe resident frame")
    return bounded


def _load_achieved_throughput(
    result: dict[str, Any], *, condition: str, repeat_count: int, aiclk_mhz: float
) -> tuple[float, str]:
    """Convert load work to the historical 1.35e9-tick reference unit."""
    if condition == "host_cpu_transfer":
        interval = result.get("host_load_interval")
        elapsed_seconds = (interval.end_ns - interval.start_ns) / 1_000_000_000.0
        transfer_count = result.get("host_work", {}).get("transfer_count", HOST_TRANSFER_COUNT)
        work_rate = transfer_count / max(elapsed_seconds, 1e-12)
        result["load_work_rate"] = {
            "value": work_rate,
            "unit": "transfers_per_host_second",
        }
        return transfer_count / HOST_TRANSFER_COUNT / max(elapsed_seconds, 1e-12), "repeats_per_host_second"
    interval = result.get("card_load_interval")
    worker_rows = result.get("card_load_worker_intervals")
    worker_elapsed_ticks = []
    if isinstance(worker_rows, list):
        for row in worker_rows:
            if isinstance(row, Mapping):
                start = row.get("start_ticks")
                end = row.get("end_ticks")
                if isinstance(start, int) and isinstance(end, int) and end >= start:
                    worker_elapsed_ticks.append(end - start)
    if worker_elapsed_ticks:
        elapsed_ticks = max(worker_elapsed_ticks)
        result["throughput_interval"] = {
            "source": "slowest_worker_own_elapsed_interval",
            "elapsed_ticks": elapsed_ticks,
            "elapsed_seconds_at_reference_aiclk": ticks_to_seconds(
                ticks=elapsed_ticks, selection=fixed_reference_aiclk_selection()
            ),
            "reference_aiclk_mhz": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
            "reference_aiclk_source": CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
            "load_interval_aiclk_mhz": aiclk_mhz,
            "load_interval_aiclk_source": "load_interval_trace_samples",
            "aggregate_span_elapsed_ticks": interval.elapsed_ticks,
        }
    else:
        elapsed_ticks = interval.end_ticks - interval.start_ticks
        result["throughput_interval"] = {
            "source": "aggregate_card_interval_legacy_compatibility",
            "elapsed_ticks": elapsed_ticks,
            "elapsed_seconds_at_reference_aiclk": ticks_to_seconds(
                ticks=elapsed_ticks, selection=fixed_reference_aiclk_selection()
            ),
            "reference_aiclk_mhz": CARD_THROUGHPUT_REFERENCE_AICLK_MHZ,
            "reference_aiclk_source": CARD_THROUGHPUT_REFERENCE_AICLK_SOURCE,
            "load_interval_aiclk_mhz": aiclk_mhz,
            "load_interval_aiclk_source": "load_interval_trace_samples",
        }
    elapsed_seconds = ticks_to_seconds(
        ticks=elapsed_ticks, selection=fixed_reference_aiclk_selection()
    )
    if condition == "dram":
        work_rate = (DRAM_STREAM_TOTAL_BYTES * repeat_count) / max(elapsed_seconds, 1e-12)
        result["load_work_rate"] = {
            "value": work_rate,
            "unit": "bytes_per_card_second",
        }
        return repeat_count / max(elapsed_seconds, 1e-12), "repeats_per_card_second"
    work_rate = (NS_LOAD_OUTER_REPETITIONS * repeat_count) / max(elapsed_seconds, 1e-12)
    result["load_work_rate"] = {
        "value": work_rate,
        "unit": "outer_repetitions_per_card_second",
    }
    return repeat_count / max(elapsed_seconds, 1e-12), "repeats_per_card_second"


def _calibration_actual_duration(
    *,
    frame_count: int,
    frame_interval_ticks: int,
    aiclk_mhz: float,
    aiclk_source: str,
) -> dict[str, Any]:
    """Convert the configured calibration schedule with its selected clock."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
        raise ValueError("frame_count must be a non-negative integer")
    if isinstance(frame_interval_ticks, bool) or not isinstance(frame_interval_ticks, int) or frame_interval_ticks < 0:
        raise ValueError("frame_interval_ticks must be a non-negative integer")
    if not isinstance(aiclk_source, str) or not aiclk_source.strip():
        raise ValueError("aiclk_source must be a non-empty string")
    selection = {
        "aiclk_mhz": aiclk_mhz,
        "aiclk_mhz_exact": aiclk_mhz,
        "aiclk_source": aiclk_source,
    }
    return {
        "actual_duration_seconds": ticks_to_seconds(
            ticks=frame_count * frame_interval_ticks,
            selection=selection,
        ),
        "actual_duration_aiclk_mhz": aiclk_mhz,
        "actual_duration_aiclk_source": aiclk_source,
        "actual_duration_aiclk_selection": aiclk_selection_for_source(aiclk_source),
    }


def _calibration_load_mappings(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return the nested load and its preferred interval metadata."""
    nested_load = payload.get("load")
    load = nested_load if isinstance(nested_load, Mapping) else {}
    interval = load.get("card_load_interval")
    if not isinstance(interval, Mapping):
        interval = payload.get("card_load_interval")
    throughput_interval = load.get("throughput_interval")
    if not isinstance(throughput_interval, Mapping):
        throughput_interval = payload.get("throughput_interval")
    return load, {
        "interval": interval if isinstance(interval, Mapping) else {},
        "throughput": (
            throughput_interval if isinstance(throughput_interval, Mapping) else {}
        ),
    }


def _calibration_card_ticks(
    metadata: Mapping[str, Any],
) -> Any:
    """Prefer worker ticks, fall back to aggregate endpoints, and retain invalid input."""
    throughput = metadata["throughput"]
    card_ticks = throughput.get("elapsed_ticks")
    if isinstance(card_ticks, int) and card_ticks > 0:
        return card_ticks
    interval = metadata["interval"]
    start = interval.get("start_ticks")
    end = interval.get("end_ticks")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return end - start
    return card_ticks


def _calibration_measurement_repeat_count(
    payload: Mapping[str, Any],
    load: Mapping[str, Any],
    calibration: CalibrationMetadata,
) -> Any:
    """Recover the repeat count used by the calibration measurement."""
    repeat_count = calibration.load_measurement_repeat_count
    if repeat_count is not None:
        return repeat_count
    repeat_count = payload.get("load_repeat_count")
    if not isinstance(repeat_count, int) or repeat_count <= 0:
        repeat_count = load.get("repeat_count")
    if not isinstance(repeat_count, int) or repeat_count <= 0:
        raw_calibration = payload.get("calibration")
        if isinstance(raw_calibration, Mapping):
            repeat_count = raw_calibration.get("measurement_repeat_count")
    return repeat_count


def _calibration_interval_aiclk(
    payload: Mapping[str, Any], calibration: CalibrationMetadata
) -> tuple[Any, Any]:
    """Recover weighted load-interval AICLK and its provenance."""
    aiclk = calibration.load_interval_aiclk_mhz
    source = calibration.load_interval_aiclk_source
    if aiclk is None:
        aiclk = payload.get("aiclk_mhz_load_interval_weighted")
    if source is None:
        source = payload.get("aiclk_mhz_load_interval_weighted_source")
    provenance = payload.get("power_clock_provenance")
    if isinstance(provenance, Mapping):
        if aiclk is None:
            aiclk = provenance.get("aiclk_mhz_load_interval_weighted")
        if source is None:
            source = provenance.get("aiclk_mhz_load_interval_weighted_source")
    return aiclk, source


def _calibration_timing_metadata(
    payload: Mapping[str, Any], calibration: CalibrationMetadata
) -> CalibrationMetadata:
    """Recover calibrated card ticks and trace AICLK from a resident record."""
    if calibration.condition not in {"newton_schulz", "dram"}:
        return calibration
    load, metadata = _calibration_load_mappings(payload)
    card_ticks = _calibration_card_ticks(metadata)
    measurement_repeat_count = _calibration_measurement_repeat_count(
        payload, load, calibration
    )
    aiclk, source = _calibration_interval_aiclk(payload, calibration)
    if card_ticks is None or measurement_repeat_count is None or aiclk is None or source is None:
        return calibration
    try:
        return cast(
            CalibrationMetadata,
            replace(
                calibration,
                load_card_ticks=card_ticks,
                load_measurement_repeat_count=measurement_repeat_count,
                load_interval_aiclk_mhz=aiclk,
                load_interval_aiclk_source=source,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise LoadConfigurationError(
            "calibration load-time metadata is invalid"
        ) from exc


def _read_load_calibration(path: Path) -> CalibrationMetadata:
    try:
        resolved_path = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise LoadConfigurationError("load calibration metadata path cannot be resolved") from exc
    trusted_roots = (
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/out").resolve(),
    )
    if not any(resolved_path.is_relative_to(root) for root in trusted_roots):
        raise LoadConfigurationError(
            "load calibration metadata must be beneath the current, temporary, or /out directory"
        )
    path = resolved_path
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadConfigurationError(f"cannot read load calibration metadata: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise LoadConfigurationError("load calibration metadata must be a JSON object")
    if payload.get("record_schema") == "issue137-calibration-v1":
        raise LoadConfigurationError(
            "Issue #137 calibration metadata is unsupported in the resident runner"
        )
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise LoadConfigurationError("load calibration metadata has no calibration object")
    if calibration.get("source") != "watcher_off_calibration":
        raise LoadConfigurationError(
            "board runs require watcher_off_calibration source metadata"
        )
    try:
        fields = {
            "condition",
            "duration_seconds",
            "achieved_throughput",
            "throughput_unit",
            "repeat_count",
            "watcher_enabled",
            "source",
            "target_seconds",
            "safety_factor",
            "throughput_reference_aiclk_mhz",
            "throughput_reference_aiclk_source",
            "throughput_unit_definition",
            "load_card_ticks",
            "load_measurement_repeat_count",
            "load_interval_aiclk_mhz",
            "load_interval_aiclk_min_mhz",
            "load_interval_aiclk_source",
            "load_interval_aiclk_selection",
            "predicted_load_time_aiclk_mhz",
            "predicted_load_time_aiclk_source",
            "predicted_load_time_aiclk_selection",
            "actual_duration_aiclk_mhz",
            "actual_duration_aiclk_source",
            "actual_duration_aiclk_selection",
        }
        parsed = CalibrationMetadata(
            **{key: calibration[key] for key in fields if key in calibration}
        )
        return _calibration_timing_metadata(payload, parsed)
    except (TypeError, ValueError) as exc:
        raise LoadConfigurationError(
            f"load calibration metadata is invalid: {exc}"
        ) from exc


def _select_unloaded_execution(args: argparse.Namespace) -> tuple[str, int, None]:
    if args.calibrate or args.calibration_json is not None or args.repeat_count is not None:
        raise LoadConfigurationError("unloaded control does not accept load calibration options")
    return "unloaded", 0, None


def _select_calibration_execution(
    args: argparse.Namespace, condition: str
) -> tuple[str, int, None]:
    if args.calibration_json is not None:
        raise LoadConfigurationError("calibration run cannot consume calibration-json")
    repeat_count = 1 if args.repeat_count is None else args.repeat_count
    if condition in {"newton_schulz", "dram"}:
        repeat_count = validate_device_load_repeat_count(condition, repeat_count)
    elif isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
        raise LoadConfigurationError("calibration repeat-count must be a positive integer")
    if args.calibration_duration_seconds <= 0 or args.calibration_duration_seconds > 60:
        raise LoadConfigurationError("calibration-duration-seconds must be in (0, 60]")
    validate_measurement_target_seconds(getattr(args, "target_seconds", 500.0))
    return condition, repeat_count, None


def _select_loaded_execution(
    args: argparse.Namespace, condition: str
) -> tuple[str, int, CalibrationMetadata]:
    if condition == "host_cpu_transfer":
        raise LoadConfigurationError(
            "Issue #120 host_cpu_transfer measurements are retained for the #120 "
            "starting point and cannot produce Issue #118 evidence"
        )
    if args.calibration_json is None:
        raise LoadConfigurationError("loaded runs require calibration-json")
    calibration = _read_load_calibration(args.calibration_json)
    if calibration.condition != condition:
        raise LoadConfigurationError("calibration condition does not match --condition")
    if calibration.target_seconds is None:
        raise LoadConfigurationError("calibration metadata lacks a bounded target_seconds")
    try:
        predicted_load_seconds = predicted_load_time_seconds(calibration)
    except LoadConfigurationError as exc:
        raise LoadConfigurationError(
            "loaded device runs require sampler-on calibration load card ticks and "
            "load-interval AICLK provenance"
        ) from exc
    required_timeout = predicted_load_seconds + CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS
    if args.outer_timeout_seconds < required_timeout:
        raise LoadConfigurationError(
            "outer-timeout-seconds is shorter than the calibrated real load time "
            f"({predicted_load_seconds:.3f}s + "
            f"{CALIBRATION_LOAD_TIMEOUT_MARGIN_SECONDS:.0f}s required margin; "
            f"AICLK source {calibration.load_interval_aiclk_source})"
        )
    repeat_count = calibration.repeat_count
    if condition in {"newton_schulz", "dram"}:
        repeat_count = validate_device_load_repeat_count(condition, repeat_count)
        if args.repeat_count is not None:
            requested_repeat_count = validate_device_load_repeat_count(
                condition, args.repeat_count
            )
            if requested_repeat_count != repeat_count:
                raise LoadConfigurationError("repeat-count does not match calibration metadata")
    elif args.repeat_count is not None and args.repeat_count != repeat_count:
        raise LoadConfigurationError("repeat-count does not match calibration metadata")
    return condition, repeat_count, calibration


def _load_execution_selection(
    args: argparse.Namespace,
) -> tuple[str, int, CalibrationMetadata | None]:
    condition = args.condition
    if condition == "unloaded":
        return _select_unloaded_execution(args)
    if args.calibrate:
        return _select_calibration_execution(args, condition)
    return _select_loaded_execution(args, condition)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The wrapper owns per-run trace selection. A missing option means that
    # sampler-off mode has no trace; it does not infer one from output files.
    power_trace = args.power_trace
    config = _config_from_args(args)
    environment: Any = {}
    condition = "unloaded"
    repeat_count = 0
    calibration: CalibrationMetadata | None = None
    newton_schulz_core_plan: NewtonSchulzCorePlan | None = None
    environment_result_path = _environment_result_path()
    if args.out is not None and environment_result_path is not None:
        output_path = None
    elif args.out is not None:
        output_path = args.out
    else:
        output_path = environment_result_path
    raw_timestamps_path = args.raw_timestamps_out
    try:
        args.out = _resolve_result_path(args)
        output_path = args.out
        output_path, raw_timestamps_path = _validate_output_destinations(args)
        environment_path = args.env_json or _environment_output_path()
        environment = _environment(environment_path)
        environment = validate_record_inputs(
            harness_commit=(
                environment.get("harness_commit")
                if isinstance(environment, Mapping)
                else None
            ),
            environment=environment,
            power_trace=power_trace,
        )
        validate_harness_preflight(
            environment,
            pushed=environment.get("harness_pushed"),
        )
        sampler = environment["telemetry_sampler"]
        newton_schulz_core_plan = _newton_schulz_core_plan_from_args(args)
        _validate_card_sampler_preflight(args.condition, sampler)
        condition, repeat_count, calibration = _load_execution_selection(args)
        _validate_power_trace_preflight(
            power_trace,
            environment=environment,
            output_destinations=(output_path, raw_timestamps_path),
        )
        _validate_output_writability((output_path, raw_timestamps_path))
        watcher = _resolve_watcher_mode(args.watcher, os.environ.get("TT_METAL_WATCHER"))
        if args.calibrate and watcher:
            raise LoadConfigurationError("load calibration must run with Watcher disabled")
        if args.calibrate:
            # Validate caller operands before bounding performs division by the
            # frame interval, then validate the bounded schedule again below.
            if condition in {"newton_schulz", "dram"}:
                config = validate_configuration(
                    config, watcher=watcher, load_condition=condition
                )
            else:
                config = validate_configuration(config, watcher=watcher)
            config = _bound_calibration_config(config, args.calibration_duration_seconds)
        if condition in {"newton_schulz", "dram"}:
            config = validate_configuration(
                config, watcher=watcher, load_condition=condition
            )
        else:
            config = validate_configuration(config, watcher=watcher)
        validate_resident_allocation_bounds(
            config, raw_timestamps_out=args.raw_timestamps_out is not None
        )
        validate_run_budget_fits_outer_cap(config, watcher=watcher)
    except (OSError, TypeError, ValueError) as exc:
        rejection_output_path = (
            None if isinstance(exc, ResidentResultPathMismatch) else output_path
        )
        _write_rejection_if_possible(
            rejection_output_path,
            config=config,
            environment=environment,
            reason=str(exc),
            power_trace=power_trace,
            raw_timestamps_path=raw_timestamps_path,
        )
        return 2

    try:
        import ttnn
    except ImportError as exc:  # pragma: no cover - exercised only outside the image
        raise RuntimeError("run_resident.py requires the pinned TTNN image") from exc

    run_start = datetime.datetime.now(datetime.UTC)
    # Issue #118 keeps the unloaded control on the same two-CQ device-open
    # contract as every eventual loaded mode; CQ0 remains the resident queue.
    device = ttnn.open_device(device_id=args.device_id, num_command_queues=2)
    device_cleanup_safe = True
    try:
        try:
            result = (
                _run_device(ttnn, device, config, watcher=watcher)
                if condition == "unloaded"
                else _run_loaded_device(
                    ttnn,
                    device,
                    config,
                    condition=condition,
                    repeat_count=repeat_count,
                    watcher=watcher,
                    newton_schulz_core_plan=newton_schulz_core_plan,
                )
            )
        except HostCleanupUnsafe as exc:
            device_cleanup_safe = False
            _write(
                output_path,
                build_rejection_record(
                    config=config,
                    reason=f"host cleanup unsafe: {exc}",
                    environment=environment,
                ),
            )
            print(f"host cleanup unsafe: {exc}", file=sys.stderr)
            return 2
        except (IndexError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
            _write(
                output_path,
                build_rejection_record(
                    config=config,
                    reason=f"resident result unavailable: {exc}",
                    environment=environment,
                ),
            )
            print(f"resident result unavailable: {exc}", file=sys.stderr)
            return 2
    finally:
        if device_cleanup_safe:
            ttnn.close_device(device)
    run_end = datetime.datetime.now(datetime.UTC)

    if raw_timestamps_path is not None:
        _write_raw_timestamps(raw_timestamps_path, result["timestamps"])

    runner_output_dir = _runner_output_dir()
    live_power_trace_path: Path | None = None
    run_id: str | None = None
    if power_trace is not None:
        # Freeze the live sampler bytes before constructing the record.  The
        # wrapper may append another row while or after this construction, but
        # every declared fact below is read from this immutable snapshot.
        sampler_interval = sampler.get("interval_seconds")
        try:
            run_id = current_run_id()
        except ValueError:
            run_id = None
        if run_id is None:
            # Do not inspect a caller-supplied path without the wrapper identity.
            live_power_trace_path = runner_output_dir / "unbound-power-trace.csv"
        else:
            expected_power_trace = run_artifact_path(
                runner_output_dir, prefix="power-", suffix=".csv", run_id=run_id
            )
            if power_trace != expected_power_trace.name:
                return _reject_power_trace_run(
                    output_path,
                    config=config,
                    environment=environment,
                    reason="power trace binding rejected: name is not the current run artifact",
                )
            live_power_trace_path = expected_power_trace
    if live_power_trace_path is not None:
        try:
            if run_id is None:
                if live_power_trace_path.exists():
                    raise RunBindingError(
                        "power trace artifact exists without a wrapper run identity"
                    )
                raise FileNotFoundError("wrapper run identity is absent")
            if condition in {"newton_schulz", "dram"}:
                _wait_for_power_trace_sample(
                    live_power_trace_path,
                    result["program_completion_utc"],
                    sampler_interval_seconds=sampler_interval,
                )
                # Include the observer row in the run-bound coverage interval.
                run_end = datetime.datetime.now(datetime.UTC)
            snapshot_path = atomic_power_trace_snapshot(
                live_power_trace_path,
                runner_output_dir,
                run_id=run_id,
                snapshot_number=1,
                snapshot_prefix=RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX,
            )
            trace_metadata = read_power_trace_snapshot(
                snapshot_path,
                options=PowerTraceSnapshotOptions(
                    run_id=run_id,
                    run_start=run_start,
                    run_end=run_end,
                    snapshot_prefix=RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX,
                    source_path=live_power_trace_path,
                    sampler_interval_seconds=sampler_interval,
                    coverage_definition=ONE_INTERVAL_BOUND,
                    load_interval_start=(
                        result.get("program_launch_utc")
                        if condition != "unloaded"
                        else None
                    ),
                    load_interval_end=(
                        result.get("program_completion_utc")
                        if condition != "unloaded"
                        else None
                    ),
                    load_interval_source=(
                        "issue118_program_host_interval"
                        if condition != "unloaded"
                        else None
                    ),
                    include_issue118_metadata=True,
                    require_all_rows_valid=True,
                    error_factory=lambda kind, error: kind,
                ),
            )
            power_trace_path = snapshot_path
            power_trace = snapshot_path.name
        except RunBindingError as exc:
            reason = f"power trace binding rejected: {exc}"
            return _reject_power_trace_run(
                output_path, config=config, environment=environment, reason=reason
            )
        except FileExistsError as exc:
            reason = f"power trace snapshot installation rejected: {exc}"
            return _reject_power_trace_run(
                output_path, config=config, environment=environment, reason=reason
            )
        except FileNotFoundError as exc:
            # A missing artifact is retained only as a diagnostic compatibility
            # path.  It carries no bytes, no AICLK, and cannot claim timing.
            trace_metadata = _missing_power_trace_metadata(
                live_power_trace_path,
                sampler_interval_seconds=sampler_interval,
                error=exc,
            )
            if run_id is None and power_trace is not None:
                trace_metadata["file"] = power_trace
            power_trace_path = None
    else:
        power_trace_path = None
        trace_metadata = None
    try:
        aiclk_selection = select_elapsed_aiclk(
            budget_aiclk_mhz=config.budget_aiclk_mhz,
            sampler_mode=sampler["mode"],
            trace_aiclk_mhz=(
                trace_metadata.get("aiclk_mhz") if trace_metadata is not None else None
            ),
            trace_aiclk_source=(
                trace_metadata.get("aiclk_source") if trace_metadata is not None else None
            ),
            allow_configured_fallback=True,
        )
    except ValueError as exc:
        reason = str(exc)
        _write(output_path, build_rejection_record(config=config, reason=reason, environment=environment))
        print(f"resident record rejected: {reason}", file=sys.stderr)
        return 2
    aiclk_mhz = aiclk_selection["aiclk_mhz"]
    aiclk_source = aiclk_selection["aiclk_source"]
    validate_post_run_aiclk(aiclk_mhz)
    trace_timing_ok = (
        trace_metadata is not None
        and trace_metadata.get("coverage_complete") is True
        and trace_metadata.get("aiclk_source") == "run_trace_samples"
        and trace_metadata.get("in_run_valid_row_count", 0) > 0
    )
    trace_timing_reason = (
        "power_trace_run_samples"
        if trace_timing_ok
        else (
            "sampler_off_by_design"
            if power_trace is None
            else (
                "power_trace_no_valid_in_run_rows"
                if trace_metadata is not None
                and trace_metadata.get("in_run_valid_row_count", 0) < 1
                else "power_trace_coverage_incomplete"
            )
        )
    )
    outlier_analysis = build_outlier_analysis(
        result["timestamps"],
        aiclk_mhz=aiclk_mhz,
        aiclk_source=aiclk_source,
        sampler_metadata=sampler,
    )
    calibration_output: CalibrationMetadata | None = None
    calibration_achieved: float | None = None
    calibration_unit: str | None = None
    resident_load_record: dict[str, Any] | None = None
    load_record: dict[str, Any] | None = None
    throughput_aiclk_mhz = aiclk_mhz
    throughput_aiclk_source: str | None = aiclk_source
    if condition in {"newton_schulz", "dram"}:
        try:
            weighted_selection = select_load_interval_aiclk(
                weighted_aiclk_mhz=trace_metadata.get(
                    "aiclk_mhz_load_interval_weighted"
                )
                if trace_metadata is not None
                else None,
                provenance=trace_metadata.get(
                    "aiclk_mhz_load_interval_weighted_source"
                )
                if trace_metadata is not None
                else None,
            )
        except ValueError as exc:
            reason = f"load interval AICLK provenance rejected: {exc}"
            _write(
                output_path,
                build_rejection_record(
                    config=config, reason=reason, environment=environment
                ),
            )
            print(f"resident record rejected: {reason}", file=sys.stderr)
            return 2
        throughput_aiclk_mhz = weighted_selection["aiclk_mhz_exact"]
        throughput_aiclk_source = weighted_selection["aiclk_source"]
    try:
        if args.calibrate:
            calibration_achieved, calibration_unit = _load_achieved_throughput(
                result,
                condition=condition,
                repeat_count=repeat_count,
                aiclk_mhz=throughput_aiclk_mhz,
            )
            throughput_interval = result.get("throughput_interval", {})
            calibration_output = calibration_metadata(
                options=CalibrationMetadataOptions(
                    condition=condition,
                    duration_seconds=args.calibration_duration_seconds,
                    achieved_throughput=calibration_achieved,
                    throughput_unit=calibration_unit,
                    target_seconds=args.target_seconds,
                    load_card_ticks=(
                        throughput_interval.get("elapsed_ticks")
                        if condition in {"newton_schulz", "dram"}
                        else None
                    ),
                    load_measurement_repeat_count=(
                        repeat_count if condition in {"newton_schulz", "dram"} else None
                    ),
                    load_interval_aiclk_mhz=(
                        throughput_aiclk_mhz
                        if condition in {"newton_schulz", "dram"}
                        else None
                    ),
                    load_interval_aiclk_min_mhz=(
                        trace_metadata.get("aiclk_mhz_in_run_min")
                        if trace_metadata is not None
                        and condition in {"newton_schulz", "dram"}
                        else None
                    ),
                    load_interval_aiclk_source=(
                        throughput_aiclk_source
                        if condition in {"newton_schulz", "dram"}
                        else None
                    ),
                    actual_duration_aiclk_mhz=throughput_aiclk_mhz,
                    actual_duration_aiclk_source=throughput_aiclk_source,
                )
            )
            if condition == "host_cpu_transfer":
                load_record = build_calibration_load_record(
                    condition=condition,
                    repeat_count=repeat_count,
                    resident_timestamps=None,
                    host_interval=result["host_load_interval"],
                    measurement_start_ns=result["measurement_start_ns"],
                    measurement_end_ns=result["measurement_end_ns"],
                    calibration=calibration_output,
                    achieved_throughput=calibration_achieved,
                    throughput_unit=calibration_unit,
                )
            else:
                load_record = build_calibration_load_record(
                    condition=condition,
                    repeat_count=repeat_count,
                    resident_timestamps=result["timestamps"],
                    card_interval=result["card_load_interval"],
                    worker_intervals=result.get("card_load_worker_intervals"),
                    calibration=calibration_output,
                    achieved_throughput=calibration_achieved,
                    throughput_unit=calibration_unit,
                )
        elif condition == "unloaded":
            resident_load_record = build_load_record(
                condition="unloaded",
                repeat_count=0,
                resident_timestamps=result["timestamps"],
            )
        elif condition == "host_cpu_transfer":
            resident_load_record = build_load_record(
                condition=condition,
                repeat_count=repeat_count,
                resident_timestamps=None,
                host_interval=result["host_load_interval"],
                measurement_start_ns=result["measurement_start_ns"],
                measurement_end_ns=result["measurement_end_ns"],
                calibration=calibration,
            )
        else:
            resident_load_record = build_load_record(
                condition=condition,
                repeat_count=repeat_count,
                resident_timestamps=result["timestamps"],
                card_interval=result["card_load_interval"],
                worker_intervals=result.get("card_load_worker_intervals"),
                calibration=calibration,
            )
    except (ValueError, TypeError) as exc:
        _write(
            output_path,
            build_rejection_record(
                config=config,
                reason=f"load interval evidence rejected: {exc}",
                environment=environment,
            ),
        )
        print(f"load interval evidence rejected: {exc}", file=sys.stderr)
        return 2
    if load_record is not None:
        resident_load_record = load_record
    record = build_measurement_record(
        config=config,
        aiclk_mhz=aiclk_mhz,
        aiclk_source=aiclk_source,
        timestamps=result["timestamps"],
        producer_full_count=result["producer_full_count"],
        consumer_empty_count=result["consumer_empty_count"],
        attempted_frame_count=result["frames_attempted"],
        produced_frame_count=result["frames_produced"],
        dropped_frame_count=result["frames_dropped"],
        aborted_attempts=result["frames_aborted"],
        startup_ticks=result["startup_ticks"],
        startup_ticks_valid=result["startup_ticks_valid"],
        work_min_ticks=result["work_min_ticks"] if result["work_ticks_valid"] else None,
        work_max_ticks=result["work_max_ticks"] if result["work_ticks_valid"] else None,
        failure_check=result["failure_check"],
        kernel_error_flag=result["kernel_error_flag"],
        harness_commit=str(environment.get("harness_commit") or ""),
        environment=environment,
        power_trace=power_trace,
        power_trace_path=power_trace_path,
        power_trace_metadata=trace_metadata,
        interval_extrema_policy=(
            INTERVAL_EXTREMA_POLICY_POSITIVE_DURATION_LEFT_HELD
            if power_trace_path is not None
            else None
        ),
        outlier_analysis=outlier_analysis,
        run_start=run_start.isoformat(),
        run_end=run_end.isoformat(),
        load_interval_start=(
            result.get("program_launch_utc") if condition != "unloaded" else None
        ),
        load_interval_end=(
            result.get("program_completion_utc") if condition != "unloaded" else None
        ),
        load_interval_source=(
            "issue118_program_host_interval" if condition != "unloaded" else None
        ),
        program_launch_ns=result.get("program_launch_ns"),
        program_completion_ns=result.get("program_completion_ns"),
        watcher=watcher,
        timing_evidence=(
            trace_timing_ok
            and not watcher
            and result["frames_attempted"] == config.frame_count
            and result["frames_consumed"] == result["frames_produced"]
            and result["frames_dropped"] == 0
            and not validate_failure_check(result["failure_check"]).error_flag
        ),
        timing_evidence_reason=trace_timing_reason,
        load_record=resident_load_record,
        table_swap=result.get("table_swap"),
    )
    record["frames_produced"] = result["frames_produced"]
    record["frames_consumed"] = result["frames_consumed"]
    if newton_schulz_core_plan is not None:
        record["newton_schulz_core_plan"] = newton_schulz_core_plan.as_record()
    if calibration_output is not None:
        calibration_record = calibration_output.as_record()
        record["status"] = "calibration"
        record["timing_evidence"] = False
        record["timing_evidence_reason"] = "calibration_only"
        if isinstance(record.get("power_clock_provenance"), dict):
            record["power_clock_provenance"]["timing_evidence"] = False
            record["power_clock_provenance"]["timing_evidence_reason"] = "calibration_only"
        record["load"] = load_record
        record["load_coverage"] = load_record["coverage"]
        record["card_load_interval"] = load_record["card_load_interval"]
        record["host_load_interval"] = load_record["host_load_interval"]
        record["valid_evidence"] = False
        calibration_record.update(
            {
                "requested_duration_seconds": args.calibration_duration_seconds,
                "actual_frame_count": config.frame_count,
                **_calibration_actual_duration(
                    frame_count=config.frame_count,
                    frame_interval_ticks=config.frame_interval_ticks,
                    aiclk_mhz=throughput_aiclk_mhz,
                    aiclk_source=throughput_aiclk_source,
                ),
                "measurement_repeat_count": repeat_count,
            }
        )
        record["calibration_schema"] = "issue-118-load-calibration-v1"
        record["calibration_measurement_mode"] = "device_merged_resident_load"
        record["calibration"] = calibration_record
        record["selected_repeat_count"] = calibration_output.repeat_count
        record["calibration_measurement_repeat_count"] = repeat_count
    if calibration_output is None:
        validate_resident_record(
            record,
            timestamps=result["timestamps"],
            power_trace_path=power_trace_path,
            power_trace_metadata=trace_metadata,
            builder=True,
            strict_trace=power_trace is not None,
            raise_on_error=False,
        )
    _write(output_path, record)
    print(f"resident record -> {output_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
