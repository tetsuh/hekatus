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
import re
import struct
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from enodia.strict_json import dumps as strict_json_dumps
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
from enodia.tt.bench.telemetry import parse_power_trace

_KERNEL_DIR = Path(__file__).with_name("kernels")
_AICLK_MAX_MHZ = (1 << 63) - 1


class ResidentResultUnavailable(ValueError):
    """A device result contained a numeric value that cannot be classified."""


def _safe_aiclk_integer(value: Any) -> int | None:
    """Convert one AICLK reading, treating non-finite/out-of-range values as absent."""
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > _AICLK_MAX_MHZ:
        return None
    try:
        converted = int(parsed)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if converted > 0 else None


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
    ttnn: Any, config: ResidentConfig, tensors: dict[str, Any], *, watcher: bool
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
    return ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[
            ttnn.SemaphoreDescriptor(0, ttnn.CoreType.WORKER, consumer_ranges, 0),
            ttnn.SemaphoreDescriptor(1, ttnn.CoreType.WORKER, producer_ranges, 0),
            ttnn.SemaphoreDescriptor(2, ttnn.CoreType.WORKER, consumer_ranges, 0),
            ttnn.SemaphoreDescriptor(3, ttnn.CoreType.WORKER, producer_ranges, 0),
        ],
        cbs=[
            _cb(ttnn, index=0, core_ranges=producer_ranges),
            _cb(ttnn, index=1, core_ranges=consumer_ranges),
        ],
    )


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


def _run_device(
    ttnn: Any, device, config: ResidentConfig, *, watcher: bool
) -> dict[str, Any]:
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
    try:
        program = _program(ttnn, config, tensors, watcher=watcher)
        ttnn.generic_op(list(tensors.values()), program)
        ttnn.synchronize_device(device)
        timestamp_values = _download(ttnn, timestamps)
        producer_values = _download(ttnn, producer_stats).reshape(-1)
        consumer_values = _download(ttnn, consumer_stats).reshape(-1)
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
            | (
                _device_word(timestamp_values[index, 0, 0, 1], f"timestamp {index} high word")
                << 32
            )
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
    finally:
        for tensor in tensors.values():
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask device errors
                pass


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _environment_output_path() -> Path:
    run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    if run_id is None or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError(
            "HEKATUS_TT_RUN_ID must be a nonempty safe filename component "
            "when --env-json is omitted"
        )
    return Path("/out") / f"env-{run_id}.json"


def _power_trace_aiclk_values(power_trace: str | None) -> list[int]:
    if power_trace is None:
        return []
    path = Path("/out") / power_trace
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
    """Select a trace AICLK, with snapshot fallback explicitly diagnostic-only."""
    values = _power_trace_aiclk_values(power_trace)
    if values:
        return max(values)
    if allow_environment_snapshot:
        fallback = _environment_aiclk_values(environment)
        if fallback:
            return max(fallback)
    return None


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
    parser.add_argument("--watcher", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    environment_path = args.env_json or _environment_output_path()
    environment = _environment(environment_path)
    # The wrapper owns per-run trace selection. A missing option means that
    # sampler-off mode has no trace; it does not infer one from output files.
    power_trace = args.power_trace
    trace_aiclk = _power_trace_aiclk_values(power_trace)
    if trace_aiclk:
        environment["aiclk_mhz_observed"] = sorted(set(trace_aiclk))
    config = _config_from_args(args)
    try:
        environment = validate_record_inputs(
            harness_commit=environment.get("harness_commit"),
            environment=environment,
            power_trace=power_trace,
        )
        sampler = environment["telemetry_sampler"]
        watcher = _resolve_watcher_mode(args.watcher, os.environ.get("TT_METAL_WATCHER"))
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
    device = ttnn.open_device(device_id=args.device_id)
    try:
        try:
            result = _run_device(ttnn, device, config, watcher=watcher)
        except (IndexError, OverflowError, TypeError, ValueError) as exc:
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
        ttnn.close_device(device)
    run_end = datetime.datetime.now(datetime.UTC)

    if args.raw_timestamps_out is not None:
        _write_raw_timestamps(args.raw_timestamps_out, result["timestamps"])

    power_trace_path = Path("/out") / power_trace if power_trace is not None else None
    trace_metadata = (
        parse_power_trace(power_trace_path, run_start=run_start, run_end=run_end)
        if power_trace_path is not None
        else None
    )
    trace_aiclk = trace_metadata.get("aiclk_mhz") if trace_metadata is not None else None
    aiclk_mhz = _safe_aiclk_integer(trace_aiclk)
    if aiclk_mhz is None:
        aiclk_mhz = _power_aiclk(
            power_trace,
            environment,
            allow_environment_snapshot=True,
        )
    if aiclk_mhz is None:
        reason = "no valid AICLK sample was available"
        _write(args.out, build_rejection_record(config=config, reason=reason, environment=environment))
        print(f"resident record rejected: {reason}", file=sys.stderr)
        return 2
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
        sampler_metadata=sampler,
    )
    record = build_measurement_record(
        config=config,
        aiclk_mhz=aiclk_mhz,
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
    )
    record["frames_produced"] = result["frames_produced"]
    record["frames_consumed"] = result["frames_consumed"]
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
