"""Run the Issue #12 Stage 1 resident producer/consumer program.

This module is executed inside the pinned toolchain container by
``run_in_container.sh``.  It allocates the ring and output buffers, launches
both kernels once, synchronizes, and only then downloads timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    ResidentPreflightError,
    build_measurement_record,
    build_rejection_record,
    failure_name,
    run_budget_breakdown,
    select_failure_check,
    split_u64,
    validate_configuration,
    validate_record_inputs,
    validate_run_budget_fits_outer_cap,
)

_KERNEL_DIR = Path(__file__).with_name("kernels")


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
    code = int(values[base])
    return {
        "code": code,
        "name": failure_name(code),
        "source": source,
        "elapsed_ticks": int(values[base + 1]) | (int(values[base + 2]) << 32),
        "limit_ticks": int(values[base + 3]) | (int(values[base + 4]) << 32),
        "unit": "device_clock_ticks",
        "valid": bool(values[base + 5]),
    }


def _is_consumer_fixed_work_budget(failure: dict[str, Any]) -> bool:
    return failure.get("code") == 4 and failure.get("name") == "consumer_fixed_work_budget"


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
        frames_consumed = min(int(consumer_values[1]), config.frame_count)
        raw_timestamps = [
            int(timestamp_values[index, 0, 0, 0])
            | (int(timestamp_values[index, 0, 0, 1]) << 32)
            for index in range(frames_consumed)
        ]
        return {
            "timestamps": raw_timestamps,
            "producer_full_count": int(producer_values[0]),
            "consumer_empty_count": int(consumer_values[0]),
            "cycle_budget_hit": _is_consumer_fixed_work_budget(consumer_failure),
            "kernel_error_flag": int(bool(producer_values[2] or consumer_values[2])),
            "frames_attempted": int(producer_values[3]),
            "frames_produced": int(producer_values[1]),
            "frames_dropped": int(producer_values[4]),
            "frames_aborted": int(producer_values[3]) - int(producer_values[1]) - int(producer_values[4]),
            "frames_consumed": int(consumer_values[1]),
            "startup_ticks": int(consumer_values[4])
            | (int(consumer_values[5]) << 32),
            "startup_ticks_valid": bool(consumer_values[6]),
            "work_min_ticks": int(consumer_values[13]) | (int(consumer_values[14]) << 32),
            "work_max_ticks": int(consumer_values[15]) | (int(consumer_values[16]) << 32),
            "work_ticks_valid": bool(consumer_values[17]),
            "producer_failure": producer_failure,
            "consumer_failure": consumer_failure,
            "failure_check": select_failure_check(producer_failure, consumer_failure),
        }
    finally:
        for tensor in tensors.values():
            try:
                ttnn.deallocate(tensor)
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask device errors
                pass


def _latest_output(prefix: str) -> Path:
    candidates = sorted(Path("/out").glob(f"{prefix}*.json" if prefix == "env-" else f"{prefix}*.csv"))
    if not candidates:
        raise ValueError(f"wrapper output {prefix!r} was not found")
    return candidates[-1]


def _power_aiclk(power_trace: str, environment: dict[str, Any]) -> int:
    path = Path("/out") / power_trace
    values: list[int] = []
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    values.append(int(float(row["aiclk_mhz"])))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    if values:
        return max(values)
    fallback = environment.get("aiclk_mhz")
    try:
        value = int(float(fallback))
    except (TypeError, ValueError) as exc:
        raise ValueError("the power trace did not contain an AICLK sample") from exc
    if value <= 0:
        raise ValueError("AICLK must be positive")
    return value


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
    environment_path = args.env_json or _latest_output("env-")
    power_trace = args.power_trace or _latest_output("power-").name
    environment = _environment(environment_path)
    timestamp_core = args.timestamp_core or args.consumer_core
    config = ResidentConfig(
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
    try:
        validate_record_inputs(
            harness_commit=environment.get("harness_commit"),
            environment=environment,
            power_trace=power_trace,
        )
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

    device = ttnn.open_device(device_id=args.device_id)
    try:
        result = _run_device(ttnn, device, config, watcher=watcher)
    finally:
        ttnn.close_device(device)

    if args.raw_timestamps_out is not None:
        _write_raw_timestamps(args.raw_timestamps_out, result["timestamps"])

    record = build_measurement_record(
        config=config,
        aiclk_mhz=_power_aiclk(power_trace, environment),
        timestamps=result["timestamps"],
        producer_full_count=result["producer_full_count"],
        consumer_empty_count=result["consumer_empty_count"],
        cycle_budget_hit=result["cycle_budget_hit"],
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
        watcher=watcher,
        timing_evidence=(
            not watcher
            and result["frames_attempted"] == config.frame_count
            and result["frames_consumed"] == result["frames_produced"]
            and result["frames_dropped"] == 0
            and not result["cycle_budget_hit"]
        ),
    )
    record["frames_produced"] = result["frames_produced"]
    record["frames_consumed"] = result["frames_consumed"]
    _write(args.out, record)
    print(f"resident record -> {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
