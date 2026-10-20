"""Device-side grouped Issue #146 Newton-Schulz runner.

All plan, identity, environment, timing, and L1 checks run before importing
TTNN or opening a device.  The device path performs exactly eight asynchronous
launches per group and one synchronization after each group; the host group
clock is the only duration clock.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enodia.strict_json import dumps as strict_json_dumps
from enodia.tt.bench.duration_load import (
    DEFAULT_OVERRUN_LIMIT_SECONDS,
    DurationLoadError,
    run_grouped_duration_bounded_load,
)
from enodia.tt.bench.load_generators import prepare_newton_schulz_grouped_load
from tools import issue146


class Issue146DeviceError(ValueError):
    """Raised for a device-runner preflight or persistence error."""


class _StopRequested(RuntimeError):
    """Raised when the wrapper asks this runner to stop."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env-json", type=Path, required=True)
    parser.add_argument("--power-trace", type=Path, default=None)
    parser.add_argument("--groups", type=Path, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--condition", choices=("newton_schulz",), default="newton_schulz")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--mode", choices=("smoke", "long"), required=True)
    parser.add_argument("--load-core-count", type=int, required=True)
    parser.add_argument("--load-total-blocks", type=int, required=True)
    parser.add_argument("--load-matrix-block", type=int, required=True)
    parser.add_argument("--outer-repetitions", type=int, required=True)
    parser.add_argument("--launches-per-sync", type=int, required=True)
    parser.add_argument("--target-seconds", type=float, required=True)
    parser.add_argument("--overrun-limit-seconds", type=float, default=DEFAULT_OVERRUN_LIMIT_SECONDS)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    parser.add_argument("--plan", type=Path, default=issue146.DEFAULT_PLAN_PATH)
    parser.add_argument("--plan-hash", required=True)
    return parser


def _load_environment(path: Path) -> dict[str, Any]:
    source = issue146._trusted_path(path, field="Issue #146 environment artifact")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue146DeviceError("cannot read Issue #146 environment artifact") from exc
    if not isinstance(payload, dict):
        raise Issue146DeviceError("Issue #146 environment artifact must be an object")
    return payload


def _result_path(args: argparse.Namespace) -> Path:
    environment_path = os.environ.get("HEKATUS_TT_RESULT_PATH")
    if args.out is not None and environment_path is not None:
        first = issue146._trusted_path(args.out, field="--out destination")
        second = issue146._trusted_path(Path(environment_path), field="HEKATUS_TT_RESULT_PATH destination")
        if first != second:
            raise Issue146DeviceError("--out and HEKATUS_TT_RESULT_PATH must identify one result path")
        return first
    if args.out is not None:
        return issue146._trusted_path(args.out, field="Issue #146 result destination")
    if environment_path is None:
        raise Issue146DeviceError("HEKATUS_TT_RESULT_PATH is required")
    return issue146._trusted_path(Path(environment_path), field="Issue #146 result destination")


def _validate_destination(path: Path, *, field: str) -> Path:
    destination = issue146._trusted_path(path, field=field)
    if destination.exists() or destination.is_symlink():
        raise Issue146DeviceError(f"{field} collides with an existing artifact")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Issue146DeviceError(f"{field} parent cannot be created") from exc
    return destination


def _bind_environment(
    environment: Mapping[str, Any], config: issue146.Issue146Configuration
) -> dict[str, Any]:
    """Bind wrapper-provided plan facts without inventing board telemetry."""
    bound = dict(environment)
    expected: dict[str, Any] = {
        "device_id": issue146.HOST_DEVICE_ID,
        "container_device_id": issue146.CONTAINER_VISIBLE_DEVICE_ID,
        "worker_core_count": config.worker_core_count,
        "worker_core_coordinates": [list(core) for core in config.core_plan.worker_cores],
        "total_blocks": config.total_blocks,
        "outer_repetitions": issue146.OUTER_REPETITIONS,
        "launches_per_sync": issue146.LAUNCHES_PER_SYNC,
    }
    for field, value in expected.items():
        if field in bound and bound[field] != value:
            raise Issue146DeviceError(f"environment {field} does not match the plan")
        bound[field] = value
    return bound


def _validate_args(
    args: argparse.Namespace,
    *,
    plan: Mapping[str, Any],
    config: issue146.Issue146Configuration,
    environment: Mapping[str, Any],
) -> None:
    if args.condition != "newton_schulz":
        raise Issue146DeviceError("Issue #146 device runner requires Newton-Schulz")
    if args.plan_hash != issue146._plan_hash(plan):
        raise Issue146DeviceError("Issue #146 plan hash does not match the owner plan")
    if args.device_id != issue146.CONTAINER_VISIBLE_DEVICE_ID:
        raise Issue146DeviceError("Issue #146 container-visible device_id must be 0")
    if args.configuration != config.identifier:
        raise Issue146DeviceError("Issue #146 configuration does not match the selected plan")
    if args.load_core_count != config.worker_core_count:
        raise Issue146DeviceError("Issue #146 core count does not match the selected plan")
    if args.load_total_blocks != config.total_blocks:
        raise Issue146DeviceError("Issue #146 total blocks do not match the selected plan")
    if args.load_matrix_block != config.core_plan.matrix_block or args.load_matrix_block != 8:
        raise Issue146DeviceError("Issue #146 matrix_block must be 8 and match the plan")
    expected_outer_repetitions, expected_launches_per_sync = issue146.grouped_load_settings(plan)
    if args.outer_repetitions != expected_outer_repetitions:
        raise Issue146DeviceError("Issue #146 outer_repetitions does not match the plan")
    if args.launches_per_sync != expected_launches_per_sync:
        raise Issue146DeviceError("Issue #146 launches_per_sync does not match the plan")
    timing = issue146.run_timing(plan, args.mode)
    if args.target_seconds != timing["load_seconds"]:
        raise Issue146DeviceError("Issue #146 target duration does not match the plan")
    if args.outer_timeout_seconds != timing["guard_seconds"]:
        raise Issue146DeviceError("Issue #146 guard does not match the plan")
    if args.overrun_limit_seconds != DEFAULT_OVERRUN_LIMIT_SECONDS:
        raise Issue146DeviceError("Issue #146 overrun limit does not match the plan")
    issue146.validate_environment_for_run(environment, configuration=config)


def _write_json(path: Path, payload: Any, *, field: str) -> bytes:
    destination = _validate_destination(path, field=field)
    try:
        raw = (strict_json_dumps(issue146._safe_json_value(payload), indent=2) + "\n").encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(destination, flags, 0o666)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(raw)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except (OSError, TypeError, ValueError) as exc:
        raise Issue146DeviceError(f"{field} could not be persisted") from exc
    return raw


def _safe_environment(environment: Mapping[str, Any], config: issue146.Issue146Configuration) -> dict[str, Any]:
    return issue146.build_environment_record(environment, config)


def _failure_result(
    *, args: argparse.Namespace, config: issue146.Issue146Configuration,
    environment: Mapping[str, Any], reason: str, run_id: str,
    duration: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "record_schema": issue146.ISSUE146_DEVICE_RESULT_SCHEMA,
        "status": "failed",
        "run_id": run_id,
        "mode": args.mode,
        "load_mode": issue146.ISSUE146_LOAD_MODE,
        "target_seconds": args.target_seconds,
        "guard_seconds": args.outer_timeout_seconds,
        "overrun_limit_seconds": args.overrun_limit_seconds,
        "outer_repetitions": args.outer_repetitions,
        "launches_per_sync": args.launches_per_sync,
        "failure": {"kind": "device_execution_failure", "message": reason},
        "core_plan": config.core_plan.as_record(),
        "plan_sha256": args.plan_hash,
        "environment": _safe_environment(environment, config),
    }
    if duration is not None:
        result.update(duration.as_record())
    return result


def run(args: argparse.Namespace) -> int:
    # No TTNN import or device open occurs before this complete section.
    plan = issue146.load_plan(args.plan)
    config = issue146.configuration(plan, args.configuration)
    environment = _bind_environment(_load_environment(args.env_json), config)
    result_path = _result_path(args)
    _validate_args(args, plan=plan, config=config, environment=environment)
    safe_result = _validate_destination(result_path, field="Issue #146 result destination")
    group_path = args.groups
    if group_path is not None:
        group_path = _validate_destination(group_path, field="Issue #146 group array")
    wrapper_run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    if not wrapper_run_id:
        raise Issue146DeviceError("HEKATUS_TT_RUN_ID is required")

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_term = signal.signal(signal.SIGTERM, request_stop)
    old_int = signal.signal(signal.SIGINT, request_stop)
    prepared: Any = None
    device: Any = None
    try:
        # This is the first TTNN import in the process and follows all checks.
        import ttnn

        device = ttnn.open_device(device_id=args.device_id, num_command_queues=2)
        prepared = prepare_newton_schulz_grouped_load(
            ttnn,
            device,
            core_plan=config.core_plan,
            profile=True,
            outer_repetitions=args.outer_repetitions,
        )

        def enqueue_one() -> None:
            if stop_requested:
                raise _StopRequested("Issue #146 stop requested")
            prepared.launch()

        def synchronize() -> None:
            if stop_requested:
                raise _StopRequested("Issue #146 stop requested")
            ttnn.synchronize_device(device)

        idle_window = issue146.SMOKE_IDLE_WINDOW if args.mode == "smoke" else issue146.LONG_IDLE_WINDOW
        try:
            duration = run_grouped_duration_bounded_load(
                enqueue_one,
                synchronize,
                target_seconds=args.target_seconds,
                launches_per_sync=args.launches_per_sync,
                idle_window_seconds=idle_window,
                monotonic_ns=time.monotonic_ns,
                overrun_limit_seconds=args.overrun_limit_seconds,
            )
            result = {
                "record_schema": issue146.ISSUE146_DEVICE_RESULT_SCHEMA,
                **duration.as_record(),
                "status": "ok" if duration.passed else "failed",
                "run_id": wrapper_run_id,
                "mode": args.mode,
                "load_mode": issue146.ISSUE146_LOAD_MODE,
                "target_seconds": args.target_seconds,
                "guard_seconds": args.outer_timeout_seconds,
                "overrun_limit_seconds": args.overrun_limit_seconds,
                "outer_repetitions": args.outer_repetitions,
                "launches_per_sync": args.launches_per_sync,
                "core_plan": config.core_plan.as_record(),
                "plan_sha256": args.plan_hash,
                "environment": _safe_environment(environment, config),
                "execution_provenance": {
                    "clock": "host.CLOCK_MONOTONIC",
                    "fixed_iterations": 12,
                    "matrix_block": 8,
                    "outer_repetitions_per_launch": args.outer_repetitions,
                    "launches_per_sync": args.launches_per_sync,
                    "preparation_before_first_timed_clock_read": True,
                    "synchronization": "one ttnn.synchronize_device after each group",
                },
            }
        except (DurationLoadError, _StopRequested) as exc:
            duration = exc.result if isinstance(exc, DurationLoadError) else None
            result = _failure_result(
                args=args,
                config=config,
                environment=environment,
                reason=str(exc),
                run_id=wrapper_run_id,
                duration=duration,
            )
        if group_path is not None and isinstance(result.get("group_timings"), list):
            group_raw = _write_json(group_path, result["group_timings"], field="Issue #146 group array")
            result["groups_path"] = group_path.name
            result["groups_sha256"] = __import__("hashlib").sha256(group_raw).hexdigest()
            result["groups_byte_count"] = len(group_raw)
        _write_json(safe_result, result, field="Issue #146 result")
    finally:
        if prepared is not None:
            try:
                prepared.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the execution result
                print(
                    f"Issue #146 prepared cleanup failed: {type(cleanup_error).__name__}",
                    file=sys.stderr,
                )
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve the execution result
                print(
                    f"Issue #146 device cleanup failed: {type(cleanup_error).__name__}",
                    file=sys.stderr,
                )
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    print(f"issue146 result -> {safe_result.name}")
    return 0 if result.get("status") == "ok" else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except (Issue146DeviceError, issue146.Issue146Error, OSError, TypeError, ValueError) as exc:
        print(f"issue146-device: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


__all__ = ["Issue146DeviceError", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
