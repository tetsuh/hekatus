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
import struct
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.load_generators import (
    card_interval_from_profile_records,
    card_interval_from_words,
    merge_program_parts,
    prepare_dram_load,
    prepare_newton_schulz_load,
    run_host_cpu_transfer_work,
)
from enodia.tt.bench.load_harness import (
    DRAM_STREAM_TOTAL_BYTES,
    HOST_TRANSFER_BYTES,
    HOST_TRANSFER_COUNT,
    LOAD_CONDITIONS,
    NS_LOAD_OUTER_REPETITIONS,
    CalibrationMetadata,
    CardLoadInterval,
    HostLoadInterval,
    LoadConfigurationError,
    build_calibration_load_record,
    build_load_record,
    calibration_metadata,
)
from enodia.tt.bench.power_trace_snapshot import (
    ONE_INTERVAL_BOUND,
    atomic_power_trace_snapshot,
    empty_power_trace_metadata,
    public_power_trace_metadata,
    read_power_trace_snapshot,
)
from enodia.tt.bench.resident_clock import (
    safe_aiclk_integer,
    select_elapsed_aiclk,
)
from enodia.tt.bench.resident_harness import (
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
    validate_post_run_aiclk,
    validate_record_inputs,
    validate_resident_record,
    validate_run_budget_fits_outer_cap,
)
from enodia.tt.bench.run_binding import (
    RunBindingError,
    current_run_id,
    run_artifact_path,
)

_KERNEL_DIR = Path(__file__).with_name("kernels")
RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX = "resident-power-snapshot"


class ResidentResultUnavailable(ValueError):
    """A device result contained a numeric value that cannot be classified."""


class HostCleanupUnsafe(RuntimeError):
    """The CQ1 host thread could not be joined before device cleanup."""


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


def _cb(ttnn: Any, *, index: int, core_ranges):
    format_descriptor = ttnn.CBFormatDescriptor(
        buffer_index=index,
        data_format=ttnn.uint32,
        page_size=PAGE_BYTES,
        tile=ttnn.TileDescriptor(32, 32, False),
    )
    return ttnn.CBDescriptor(
        total_size=PAGE_BYTES,
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

    producer_compile = [*ring_compile, *control_compile, *producer_stats_compile]
    consumer_compile = [
        *ring_compile,
        *producer_anchor_compile,
        *timestamp_compile,
        *consumer_stats_compile,
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
    return {
        "ring": ring,
        "control": control,
        "producer_anchor": producer_anchor,
        "timestamps": timestamps,
        "producer_stats": producer_stats,
        "consumer_stats": consumer_stats,
    }


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
    frames_consumed = min(
        _device_word(consumer_values[1], "frames consumed"), config.frame_count
    )
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
    return {
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
        ttnn.generic_op(list(tensors.values()), program)
        ttnn.synchronize_device(device, cq_id=0)
        return _decode_resident_result(ttnn, config, tensors)
    finally:
        _deallocate_resident_tensors(ttnn, tensors)


def _host_load_clock(
    *,
    resident_launched: threading.Event | None,
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


def _run_host_transfers(
    ttnn: Any,
    device: Any,
    *,
    repetitions: int = 1,
    started_event: threading.Event | None = None,
    resident_launched: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    start_holder: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Perform fixed arithmetic and explicit CQ1 host/device transfers."""
    import numpy as np

    host = ttnn.Tensor(
        np.zeros((1, 1, 1, HOST_TRANSFER_BYTES // 4), dtype=np.uint32),
        ttnn.uint32,
    )
    host = ttnn.to_layout(host, ttnn.ROW_MAJOR_LAYOUT)
    device_tensors: list[Any] = []

    # Signal readiness before the first timing clock. The host work starts
    # before GenericOp so fixed CQ1 transfers can overlap the resident launch.
    if started_event is not None:
        started_event.set()

    clock_ns = _host_load_clock(
        resident_launched=resident_launched,
        stop_event=stop_event,
        start_holder=start_holder if start_holder is not None else {},
    )

    def transfer(_payload: bytes, queue_id: int) -> None:
        if queue_id != 1:
            raise RuntimeError("host condition transfer was not assigned to CQ1")
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("host load stopped before CQ1 transfer")
        device_tensor = ttnn.to_device(
            host,
            device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            queue_id=queue_id,
        )
        device_tensors.append(device_tensor)
        ttnn.from_device(device_tensor, blocking=True, queue_id=queue_id)
        ttnn.deallocate(device_tensor)

    work = run_host_cpu_transfer_work(
        repetitions=repetitions,
        clock_ns=clock_ns,
        transfer=transfer,
    )
    return work


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


def _run_loaded_device(
    ttnn: Any,
    device: Any,
    config: ResidentConfig,
    *,
    condition: str,
    repeat_count: int,
    watcher: bool,
) -> dict[str, Any]:
    """Launch resident and one selected load group in one ProgramDescriptor."""
    if condition not in {"newton_schulz", "dram", "host_cpu_transfer"}:
        raise LoadConfigurationError(f"unsupported loaded condition: {condition}")
    resident_tensors = _allocate_resident_tensors(ttnn, device, config)
    load_kernel: Any | None = None
    load_tensors: list[Any] = []
    host_work: dict[str, Any] | None = None
    host_thread: threading.Thread | None = None
    host_started = threading.Event()
    resident_launched = threading.Event()
    host_stop = threading.Event()
    host_result: dict[str, Any] = {}
    host_error: list[BaseException] = []
    host_start_holder: dict[str, int] = {}
    try:
        resident_parts = _program(
            ttnn, config, resident_tensors, watcher=watcher, return_parts=True
        )
        if condition == "newton_schulz":
            load_kernel = prepare_newton_schulz_load(
                ttnn, device, repeat_count=repeat_count, profile=True
            )
            load_parts = load_kernel.program_parts()
            load_tensors = [*load_kernel.inputs]
            if load_kernel.profile_output is not None:
                load_tensors.append(load_kernel.profile_output)
                load_tensors.extend(load_kernel.outputs[:2])
            else:
                load_tensors.extend(load_kernel.outputs)
        elif condition == "dram":
            load_kernel = prepare_dram_load(ttnn, device, repeat_count=repeat_count)
            load_parts = load_kernel.program_parts()
            load_tensors = [
                load_kernel.read_buffer,
                load_kernel.write_buffer,
                load_kernel.stats_buffer,
            ]
        else:
            load_parts = ([], [], [])
        program = merge_program_parts(ttnn, resident_parts, load_parts)
        if condition == "host_cpu_transfer":
            def run_host() -> None:
                try:
                    host_result.update(
                        _run_host_transfers(
                            ttnn,
                            device,
                            started_event=host_started,
                            resident_launched=resident_launched,
                            stop_event=host_stop,
                            start_holder=host_start_holder,
                            repetitions=repeat_count,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001 - transfer failure crosses thread boundary
                    host_error.append(exc)

            host_thread = threading.Thread(target=run_host, name="issue118-cq1-host-load")
            host_thread.start()
            if not host_started.wait(timeout=60.0):
                raise RuntimeError("host load did not start")
        measurement_start_ns = time.monotonic_ns()
        try:
            ttnn.generic_op(_merged_program_io_tensors(resident_tensors), program)
        except BaseException:
            host_stop.set()
            resident_launched.set()
            raise
        finally:
            resident_launched.set()
        ttnn.synchronize_device(device, cq_id=0)
        # Capture the CQ0 resident endpoint before waiting for any host-load
        # tail. This lets coverage accept a host load that extends beyond the
        # resident interval without making the endpoint host-load-dependent.
        measurement_end_ns = time.monotonic_ns()
        if host_thread is not None:
            join_timeout = _host_join_timeout_seconds(
                repeat_count=repeat_count,
                outer_timeout_seconds=config.outer_timeout_seconds,
            )
            host_thread.join(timeout=join_timeout)
            if host_thread.is_alive():
                raise RuntimeError("host load transfer thread did not finish")
            if host_error:
                raise RuntimeError("host load transfer failed") from host_error[0]
            host_work = host_result
        result = _decode_resident_result(ttnn, config, resident_tensors)
        result["measurement_start_ns"] = measurement_start_ns
        result["measurement_end_ns"] = measurement_end_ns
        result["host_work"] = host_work
        if condition == "newton_schulz":
            result["card_load_interval"] = card_interval_from_profile_records(
                load_kernel.profile_records()
            )
        elif condition == "dram":
            words = load_kernel.interval_words()
            rows = [
                words[index * PAGE_WORDS : (index + 1) * PAGE_WORDS]
                for index in range(len(load_kernel.worker_coordinates))
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
            result["dram_stats"] = rows
        else:
            result["host_load_interval"] = HostLoadInterval(
                host_work["start_ns"], host_work["end_ns"]
            )
        return result
    finally:
        if host_thread is not None:
            host_stop.set()
            resident_launched.set()
            cleanup_timeout = _host_join_timeout_seconds(
                repeat_count=repeat_count,
                outer_timeout_seconds=config.outer_timeout_seconds,
            )
            host_thread.join(timeout=cleanup_timeout)
            if host_thread.is_alive():
                raise HostCleanupUnsafe(
                    "CQ1 host thread remained active; device cleanup was not attempted"
                )
        if load_kernel is not None:
            load_kernel.close()
        _deallocate_resident_tensors(ttnn, resident_tensors)
def _runner_output_dir() -> Path:
    """Return the wrapper output directory; tests may redirect this seam."""
    return Path("/out")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(struct.pack("<Q", value) for value in timestamps))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json_dumps(payload, indent=2) + "\n")


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
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
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
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-duration-seconds", type=float, default=60.0)
    parser.add_argument("--target-seconds", type=float, default=500.0)
    parser.add_argument("--watcher", action="store_true")
    return parser


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
    bounded = replace(config, frame_count=candidate, outer_timeout_seconds=60)
    while candidate > 1:
        budget = run_budget_breakdown(bounded, watcher=False)["run_budget_ticks"]
        if budget <= target_ticks:
            break
        candidate -= 1
        bounded = replace(bounded, frame_count=candidate)
    if run_budget_breakdown(bounded, watcher=False)["run_budget_ticks"] > target_ticks:
        raise LoadConfigurationError("calibration duration is shorter than one safe resident frame")
    return bounded


def _load_achieved_throughput(
    result: dict[str, Any], *, condition: str, repeat_count: int, aiclk_mhz: int
) -> tuple[float, str]:
    """Convert actual load interval work into calibration throughput."""
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
    elapsed_seconds = (interval.end_ticks - interval.start_ticks) / (aiclk_mhz * 1_000_000.0)
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


def _read_load_calibration(path: Path) -> CalibrationMetadata:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadConfigurationError(f"cannot read load calibration metadata: {path.name}") from exc
    calibration = payload.get("calibration") if isinstance(payload, dict) else None
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
        }
        return CalibrationMetadata(**{key: calibration[key] for key in fields if key in calibration})
    except (TypeError, ValueError) as exc:
        raise LoadConfigurationError("load calibration metadata is invalid") from exc


def _load_execution_selection(args: argparse.Namespace) -> tuple[str, int, CalibrationMetadata | None]:
    condition = args.condition
    if condition == "unloaded":
        if args.calibrate or args.calibration_json is not None or args.repeat_count is not None:
            raise LoadConfigurationError("unloaded control does not accept load calibration options")
        return condition, 0, None
    if args.calibrate:
        if args.calibration_json is not None:
            raise LoadConfigurationError("calibration run cannot consume calibration-json")
        repeat_count = 1 if args.repeat_count is None else args.repeat_count
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
            raise LoadConfigurationError("calibration repeat-count must be a positive integer")
        if args.calibration_duration_seconds <= 0 or args.calibration_duration_seconds > 60:
            raise LoadConfigurationError("calibration-duration-seconds must be in (0, 60]")
        return condition, repeat_count, None
    if args.calibration_json is None:
        raise LoadConfigurationError("loaded runs require calibration-json")
    calibration = _read_load_calibration(args.calibration_json)
    if calibration.condition != condition:
        raise LoadConfigurationError("calibration condition does not match --condition")
    if calibration.target_seconds is None:
        raise LoadConfigurationError("calibration metadata lacks a bounded target_seconds")
    required_timeout = calibration.target_seconds * calibration.safety_factor
    if args.outer_timeout_seconds < required_timeout:
        raise LoadConfigurationError(
            "outer-timeout-seconds is shorter than the calibrated load target"
        )
    repeat_count = calibration.repeat_count
    if args.repeat_count is not None and args.repeat_count != repeat_count:
        raise LoadConfigurationError("repeat-count does not match calibration metadata")
    return condition, repeat_count, calibration


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    environment_path = args.env_json or _environment_output_path()
    environment = _environment(environment_path)
    # The wrapper owns per-run trace selection. A missing option means that
    # sampler-off mode has no trace; it does not infer one from output files.
    power_trace = args.power_trace
    config = _config_from_args(args)
    condition = "unloaded"
    repeat_count = 0
    calibration: CalibrationMetadata | None = None
    try:
        condition, repeat_count, calibration = _load_execution_selection(args)
        environment = validate_record_inputs(
            harness_commit=environment.get("harness_commit"),
            environment=environment,
            power_trace=power_trace,
        )
        sampler = environment["telemetry_sampler"]
        watcher = _resolve_watcher_mode(args.watcher, os.environ.get("TT_METAL_WATCHER"))
        if args.calibrate and watcher:
            raise LoadConfigurationError("load calibration must run with Watcher disabled")
        if args.calibrate:
            config = _bound_calibration_config(config, args.calibration_duration_seconds)
        config = validate_configuration(config, watcher=watcher)
        validate_run_budget_fits_outer_cap(config, watcher=watcher)
    except (TypeError, ValueError) as exc:
        _write(args.out, build_rejection_record(config=config, reason=str(exc), environment=environment))
        print(f"resident configuration rejected: {exc}", file=sys.stderr)
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
                )
            )
        except HostCleanupUnsafe as exc:
            device_cleanup_safe = False
            _write(
                args.out,
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
                args.out,
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

    if args.raw_timestamps_out is not None:
        _write_raw_timestamps(args.raw_timestamps_out, result["timestamps"])

    runner_output_dir = _runner_output_dir()
    live_power_trace_path = (
        runner_output_dir / power_trace if power_trace is not None else None
    )
    if live_power_trace_path is not None:
        # Freeze the live sampler bytes before constructing the record.  The
        # wrapper may append another row while or after this construction, but
        # every declared fact below is read from this immutable snapshot.
        sampler_interval = sampler.get("interval_seconds")
        try:
            run_id = current_run_id()
        except ValueError:
            run_id = None
        try:
            if run_id is None:
                if live_power_trace_path.exists():
                    raise RunBindingError(
                        "power trace artifact exists without a wrapper run identity"
                    )
                raise FileNotFoundError("wrapper run identity is absent")
            snapshot_path = atomic_power_trace_snapshot(
                live_power_trace_path,
                runner_output_dir,
                run_id=run_id,
                snapshot_number=1,
                snapshot_prefix=RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX,
            )
            trace_metadata = read_power_trace_snapshot(
                snapshot_path,
                run_id=run_id,
                run_start=run_start,
                run_end=run_end,
                snapshot_prefix=RESIDENT_POWER_TRACE_SNAPSHOT_PREFIX,
                source_path=live_power_trace_path,
                sampler_interval_seconds=sampler_interval,
                coverage_definition=ONE_INTERVAL_BOUND,
                require_all_rows_valid=True,
                error_factory=lambda kind, error: kind,
            )
            power_trace_path = snapshot_path
            power_trace = snapshot_path.name
        except RunBindingError as exc:
            reason = f"power trace binding rejected: {exc}"
            return _reject_power_trace_run(
                args.out, config=config, environment=environment, reason=reason
            )
        except FileExistsError as exc:
            reason = f"power trace snapshot installation rejected: {exc}"
            return _reject_power_trace_run(
                args.out, config=config, environment=environment, reason=reason
            )
        except FileNotFoundError as exc:
            # A missing artifact is retained only as a diagnostic compatibility
            # path.  It carries no bytes, no AICLK, and cannot claim timing.
            trace_metadata = _missing_power_trace_metadata(
                live_power_trace_path,
                sampler_interval_seconds=sampler_interval,
                error=exc,
            )
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
        _write(args.out, build_rejection_record(config=config, reason=reason, environment=environment))
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
    try:
        if args.calibrate:
            calibration_achieved, calibration_unit = _load_achieved_throughput(
                result,
                condition=condition,
                repeat_count=repeat_count,
                aiclk_mhz=aiclk_mhz,
            )
            calibration_output = calibration_metadata(
                condition=condition,
                duration_seconds=args.calibration_duration_seconds,
                achieved_throughput=calibration_achieved,
                throughput_unit=calibration_unit,
                target_seconds=args.target_seconds,
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
                calibration=calibration,
            )
    except (LoadConfigurationError, ValueError, TypeError) as exc:
        _write(
            args.out,
            build_rejection_record(
                config=config,
                reason=f"load interval evidence rejected: {exc}",
                environment=environment,
            ),
        )
        print(f"load interval evidence rejected: {exc}", file=sys.stderr)
        return 2
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
        outlier_analysis=outlier_analysis,
        run_start=run_start.isoformat(),
        run_end=run_end.isoformat(),
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
    )
    record["frames_produced"] = result["frames_produced"]
    record["frames_consumed"] = result["frames_consumed"]
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
                "actual_duration_seconds": (
                    config.frame_count * config.frame_interval_ticks
                    / (aiclk_mhz * 1_000_000.0)
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
    _write(args.out, record)
    print(f"resident record -> {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
