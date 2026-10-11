"""Board-free calibration and run-configuration CLI for Issue #118.

The default CLI only builds deterministic configuration/calibration metadata;
it never imports TTNN, opens a device, invokes Docker, or touches a device
node.  The eventual board runner consumes the emitted repeat count and must
supply card-clock interval metadata before a record can be evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.load_generators import (
    build_load_plan,
    dram_load_work,
    host_cpu_transfer_work,
    newton_schulz_load_work,
    open_device_settings,
)
from enodia.tt.bench.load_harness import (
    LOAD_CONDITIONS,
    MAX_CALIBRATION_SECONDS,
    CalibrationMetadata,
    CardLoadInterval,
    HostLoadInterval,
    LoadConfigurationError,
    calibration_metadata,
    normalize_condition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--condition",
        "--load-condition",
        dest="condition",
        choices=LOAD_CONDITIONS,
        required=True,
    )
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--run-config", action="store_true")
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument(
        "--duration-seconds",
        "--calibration-duration-seconds",
        dest="duration_seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--target-seconds",
        "--target-duration-seconds",
        dest="target_seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument("--achieved-throughput", type=float, default=None)
    parser.add_argument("--throughput-unit", default="fixed_work_units_per_second")
    parser.add_argument("--safety-factor", type=float, default=1.10)
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _validate_duration(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoadConfigurationError("duration_seconds must be a finite positive number")
    duration = float(value)
    if duration <= 0.0 or not math.isfinite(duration):
        raise LoadConfigurationError("duration_seconds must be a finite positive number")
    if duration > MAX_CALIBRATION_SECONDS:
        raise LoadConfigurationError("calibration duration must be <= 60 seconds")
    return duration


def _board_free_throughput(condition: str, duration_seconds: float) -> float:
    """Measure a deterministic fixed-work model without claiming board speed."""
    duration = _validate_duration(duration_seconds)
    # The model intentionally executes a fixed number of host operations.  A
    # caller that has a Watcher-off board result supplies --achieved-throughput;
    # this fallback exists only so the configuration path is runnable anywhere.
    if condition == "newton_schulz":
        work = 12 * 3 * 8192
    elif condition == "dram":
        work = 8 * 256 * 2
    else:
        work = 1_024
    started = time.monotonic_ns()
    accumulator = 0
    for index in range(min(work, 100_000)):
        accumulator = ((accumulator * 33) ^ index) & 0xFFFFFFFF
    elapsed = max((time.monotonic_ns() - started) / 1_000_000_000.0, 1e-9)
    # Keep the nominal duration in the metadata while using a measured fixed
    # operation count.  The accumulator prevents the loop from being removed.
    del accumulator, duration
    return float(work) / elapsed


def _read_calibration(path: Path) -> CalibrationMetadata:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadConfigurationError(f"cannot read calibration metadata: {path}") from exc
    calibration = payload.get("calibration") if isinstance(payload, dict) else None
    if not isinstance(calibration, dict):
        raise LoadConfigurationError("calibration JSON has no calibration object")
    try:
        return CalibrationMetadata(**calibration)
    except (TypeError, ValueError) as exc:
        raise LoadConfigurationError("calibration metadata is invalid") from exc


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json_dumps(payload, indent=2) + "\n")


def build_calibration(
    *,
    condition: str,
    duration_seconds: float,
    achieved_throughput: float | None = None,
    throughput_unit: str = "fixed_work_units_per_second",
    target_seconds: float = 1.0,
    safety_factor: float = 1.10,
    watcher_enabled: bool = False,
) -> dict[str, Any]:
    """Build one watcher-off calibration result without importing TTNN."""
    canonical = normalize_condition(condition)
    if watcher_enabled:
        raise LoadConfigurationError("Issue #118 calibration must run with Watcher disabled")
    if canonical == "unloaded":
        raise LoadConfigurationError("unloaded control does not need calibration")
    duration = _validate_duration(duration_seconds)
    throughput = (
        _board_free_throughput(canonical, duration)
        if achieved_throughput is None
        else achieved_throughput
    )
    calibration = calibration_metadata(
        condition=canonical,
        duration_seconds=duration,
        achieved_throughput=throughput,
        throughput_unit=throughput_unit,
        target_seconds=target_seconds,
        safety_factor=safety_factor,
    )
    return {
        "schema": "issue-118-load-calibration-v1",
        "issue": 118,
        "status": "calibration",
        "measurement_mode": "board_free_configuration",
        "watcher_enabled": False,
        "device_open": open_device_settings(),
        "num_command_queues": 2,
        "calibration": calibration.as_record(),
        "selected_repeat_count": calibration.repeat_count,
        "condition_plan": build_load_plan(canonical).as_record(),
        "work": {
            "newton_schulz": newton_schulz_load_work(),
            "dram": dram_load_work(),
            "host_cpu_transfer": host_cpu_transfer_work(),
        }[canonical],
    }


def build_condition_record(
    *,
    condition: str,
    repeat_count: int,
    resident_timestamps: list[int] | tuple[int, ...] | None,
    card_start_ticks: int | None = None,
    card_end_ticks: int | None = None,
    host_start_ns: int | None = None,
    host_end_ns: int | None = None,
    measurement_start_ns: int | None = None,
    measurement_end_ns: int | None = None,
    calibration: CalibrationMetadata | None = None,
    achieved_throughput: float | None = None,
    throughput_unit: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed condition record from downloaded interval facts."""
    from enodia.tt.bench.load_harness import build_load_record

    canonical = normalize_condition(condition)
    card_interval = None
    if card_start_ticks is not None or card_end_ticks is not None:
        if card_start_ticks is None or card_end_ticks is None:
            raise LoadConfigurationError("card interval requires both endpoints")
        card_interval = CardLoadInterval(card_start_ticks, card_end_ticks)
    host_interval = None
    if host_start_ns is not None or host_end_ns is not None:
        if host_start_ns is None or host_end_ns is None:
            raise LoadConfigurationError("host interval requires both endpoints")
        host_interval = HostLoadInterval(host_start_ns, host_end_ns)
    return build_load_record(
        condition=canonical,
        repeat_count=repeat_count,
        resident_timestamps=resident_timestamps,
        card_interval=card_interval,
        host_interval=host_interval,
        measurement_start_ns=measurement_start_ns,
        measurement_end_ns=measurement_end_ns,
        calibration=calibration,
        achieved_throughput=achieved_throughput,
        throughput_unit=throughput_unit,
    )


def build_run_configuration(
    *,
    condition: str,
    calibration: CalibrationMetadata | None = None,
    repeat_count: int | None = None,
) -> dict[str, Any]:
    """Build a run config; interval-free configs are never evidence records."""
    canonical = normalize_condition(condition)
    if canonical == "unloaded":
        selected = 0
    elif calibration is None:
        raise LoadConfigurationError(
            "a loaded run requires Watcher-off calibration metadata before repeat_count selection"
        )
    elif calibration.condition != canonical:
        raise LoadConfigurationError("calibration condition does not match run condition")
    elif repeat_count is not None:
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count <= 0:
            raise LoadConfigurationError("repeat_count must be a positive integer")
        if repeat_count != calibration.repeat_count:
            raise LoadConfigurationError("repeat_count must match calibration-selected repeat_count")
        selected = repeat_count
    else:
        selected = calibration.repeat_count
    return {
        "schema": "issue-118-load-run-config-v1",
        "issue": 118,
        "status": "configuration_only",
        "condition": canonical,
        "valid_evidence": False,
        "device_open": open_device_settings(),
        "num_command_queues": 2,
        "repeat_count": selected,
        "calibration": calibration.as_record() if calibration is not None else None,
        "plan": build_load_plan(canonical).as_record(),
        "coverage_required": canonical != "unloaded",
        "interval_source": "card_clock_required_for_device_load"
        if canonical in {"newton_schulz", "dram"}
        else "host_monotonic_required_for_host_condition",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.calibrate == args.run_config:
            raise LoadConfigurationError("select exactly one of --calibrate or --run-config")
        if args.calibrate:
            payload = build_calibration(
                condition=args.condition,
                duration_seconds=args.duration_seconds,
                achieved_throughput=args.achieved_throughput,
                throughput_unit=args.throughput_unit,
                target_seconds=args.target_seconds,
                safety_factor=args.safety_factor,
                watcher_enabled=os.environ.get("TT_METAL_WATCHER") == "1",
            )
        else:
            calibration = _read_calibration(args.calibration_json) if args.calibration_json else None
            payload = build_run_configuration(
                condition=args.condition,
                calibration=calibration,
                repeat_count=args.repeat_count,
            )
    except LoadConfigurationError as exc:
        print(f"load configuration rejected: {exc}")
        return 2
    _write(args.out, payload)
    print(strict_json_dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
