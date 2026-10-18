"""Device-side Issue #137 duration-bounded Newton-Schulz entrypoint.

The wrapper invokes this opt-in runner inside the pinned toolchain image.  All
plan, environment, result-path, and duration checks happen before importing
TTNN or opening a device.  The timed callback launches one prepared fixed
Newton-Schulz program and synchronizes its completion before the next launch.
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
    run_duration_bounded_load,
    validate_duration_bounds,
)
from enodia.tt.bench.load_generators import prepare_newton_schulz_duration_load
from tools import issue137


class Issue137DeviceError(ValueError):
    """Raised for a device-runner preflight or result error."""


class _StopRequested(RuntimeError):
    """Raised when the wrapper asks the device runner to stop."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env-json", type=Path, required=True)
    parser.add_argument("--power-trace", default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--condition", choices=("newton_schulz",), default="newton_schulz")
    parser.add_argument("--mode", choices=("smoke", "long"), required=True)
    parser.add_argument("--load-core-count", type=int, required=True)
    parser.add_argument("--load-total-blocks", type=int, required=True)
    parser.add_argument("--load-matrix-block", type=int, required=True)
    parser.add_argument("--target-seconds", type=float, required=True)
    parser.add_argument("--overrun-limit-seconds", type=float, default=DEFAULT_OVERRUN_LIMIT_SECONDS)
    parser.add_argument("--outer-timeout-seconds", type=float, required=True)
    parser.add_argument("--plan", type=Path, default=issue137.DEFAULT_PLAN_PATH)
    parser.add_argument("--plan-hash", required=True)
    return parser


def _load_environment(path: Path) -> dict[str, Any]:
    try:
        source = issue137._trusted_path(path, field="Issue #137 environment artifact")
        payload = json.loads(source.read_text(encoding="utf-8"))
    except issue137.Issue137Error as exc:
        raise Issue137DeviceError(str(exc)) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Issue137DeviceError(f"cannot read environment artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise Issue137DeviceError("environment artifact must be an object")
    issue137.validate_environment_for_run(payload)
    return payload


def _result_path(args: argparse.Namespace) -> Path:
    environment_path = os.environ.get("HEKATUS_TT_RESULT_PATH")
    if args.out is not None and environment_path is not None:
        return issue137.resolve_result_path(
            cli_path=args.out, environment_path=Path(environment_path)
        )
    if args.out is not None:
        return issue137.resolve_result_path(cli_path=args.out)
    if environment_path is None:
        raise Issue137DeviceError("HEKATUS_TT_RESULT_PATH is required")
    return issue137.resolve_result_path(environment_path=Path(environment_path))


def _validate_load_arguments(
    args: argparse.Namespace,
    *,
    plan: Mapping[str, Any],
    config: issue137.Issue137Configuration,
) -> None:
    if args.condition != "newton_schulz":
        raise Issue137DeviceError("Issue #137 device runner requires Newton-Schulz")
    if args.plan_hash != issue137._plan_hash(plan):
        raise Issue137DeviceError("Issue #137 plan hash does not match the owner plan")
    if args.device_id != issue137.CONTAINER_VISIBLE_DEVICE_ID:
        raise Issue137DeviceError("Issue #137 container-visible device_id must be 0")
    if args.load_matrix_block != 8:
        raise Issue137DeviceError("Issue #137 duration mode requires matrix_block=8")
    expected = config.core_plan
    if args.load_core_count != expected.core_count:
        raise Issue137DeviceError("Issue #137 core count does not match the selected plan")
    if args.load_total_blocks != expected.total_blocks:
        raise Issue137DeviceError("Issue #137 total blocks do not match the selected plan")
    try:
        bounds = validate_duration_bounds(
            target_seconds=args.target_seconds,
            guard_seconds=args.outer_timeout_seconds,
            overrun_limit_seconds=args.overrun_limit_seconds,
        )
    except (TypeError, ValueError) as exc:
        raise Issue137DeviceError(str(exc)) from exc
    expected_timing = issue137.run_timing(plan, args.mode)
    if bounds.target_seconds != expected_timing["load_seconds"]:
        raise Issue137DeviceError("Issue #137 target duration does not match the plan")
    if bounds.guard_seconds != expected_timing["guard_seconds"]:
        raise Issue137DeviceError("Issue #137 guard does not match the plan")
    if args.overrun_limit_seconds != DEFAULT_OVERRUN_LIMIT_SECONDS:
        raise Issue137DeviceError("Issue #137 overrun limit does not match the plan")


def _validate_result_destination(path: Path) -> Path:
    try:
        destination = issue137._trusted_write_path(
            path, field="Issue #137 result destination"
        )
    except issue137.Issue137Error as exc:
        raise Issue137DeviceError(str(exc)) from exc
    if destination.exists() or destination.is_symlink():
        raise Issue137DeviceError("Issue #137 result destination collides with an existing artifact")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise Issue137DeviceError("Issue #137 result destination parent cannot be created") from exc
    if not destination.parent.is_dir():
        raise Issue137DeviceError("Issue #137 result destination parent is not a directory")
    return destination


def _validate_environment_binding(
    plan: Mapping[str, Any],
    config: issue137.Issue137Configuration,
    environment: Mapping[str, Any],
) -> None:
    environment_record = issue137.build_environment_record(environment, config)
    scope = plan.get("measurement_scope")
    board_id = scope.get("board_serial") if isinstance(scope, Mapping) else None
    bus_id = scope.get("pci_bus_id") if isinstance(scope, Mapping) else None
    if not issue137._environment_identity_matches(
        environment_record, board_id=board_id, bus_id=bus_id
    ):
        raise Issue137DeviceError("Issue #137 environment identity does not match the plan")


def _validate_preflight(
    args: argparse.Namespace,
    *,
    plan: Mapping[str, Any],
    config: issue137.Issue137Configuration,
    environment: Mapping[str, Any],
    result_path: Path,
) -> Path:
    _validate_load_arguments(args, plan=plan, config=config)
    safe_result_path = _validate_result_destination(result_path)
    _validate_environment_binding(plan, config, environment)
    return safe_result_path


def _safe_result_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    return issue137._safe_json_value(environment)


def _open_result_descriptor(path: Path) -> int:
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    result_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    parent_fd = os.open(path.parent, parent_flags)
    try:
        return os.open(path.name, result_flags, 0o666, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    try:
        destination = issue137._trusted_write_path(
            path, field="Issue #137 result destination"
        )
        if destination.exists() or destination.is_symlink():
            raise Issue137DeviceError("Issue #137 result destination already exists")
        serialized = strict_json_dumps(issue137._safe_json_value(result), indent=2) + "\n"
        descriptor: int | None = None
        try:
            descriptor = _open_result_descriptor(destination)
            try:
                stream = os.fdopen(descriptor, "w", encoding="utf-8")
            except (OSError, ValueError):
                os.close(descriptor)
                descriptor = None
                raise
            descriptor = None
            with stream:
                stream.write(serialized)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise Issue137DeviceError("Issue #137 result could not be persisted") from exc
    except Issue137DeviceError:
        raise
    except issue137.Issue137Error as exc:
        raise Issue137DeviceError(str(exc)) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise Issue137DeviceError("Issue #137 result could not be persisted") from exc


def _failure_result(
    *,
    args: argparse.Namespace,
    config: issue137.Issue137Configuration,
    environment: Mapping[str, Any],
    reason: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "record_schema": issue137.ISSUE137_DEVICE_RESULT_SCHEMA,
        "status": "failed",
        "run_id": run_id,
        "mode": args.mode,
        "load_mode": issue137.ISSUE137_LOAD_MODE,
        "target_seconds": args.target_seconds,
        "guard_seconds": args.outer_timeout_seconds,
        "overrun_limit_seconds": args.overrun_limit_seconds,
        "launch_count": 0,
        "first_launch_start_monotonic_ns": None,
        "workload_end_monotonic_ns": None,
        "last_launch_end_monotonic_ns": None,
        "overrun_seconds": None,
        "failure": {"kind": "device_execution_failure", "message": reason},
        "core_plan": config.core_plan.as_record(),
        "plan_sha256": args.plan_hash,
        "environment": _safe_result_environment(environment),
    }


def run(args: argparse.Namespace) -> int:
    plan = issue137.load_plan(args.plan)
    matches = [
        item for item in issue137.plan_configurations(plan)
        if item["worker_core_count"] == args.load_core_count
    ]
    # The host command binds the selected identifier through the plan hash and
    # exact core plan.  There is only one planned configuration per count.
    if len(matches) != 1:
        raise Issue137DeviceError("Issue #137 core count is not a unique planned configuration")
    config = issue137.configuration(plan, matches[0]["id"])
    environment = _load_environment(args.env_json)
    result_path = _result_path(args)
    result_path = _validate_preflight(
        args,
        plan=plan,
        config=config,
        environment=environment,
        result_path=result_path,
    )
    wrapper_run_id = os.environ.get("HEKATUS_TT_RUN_ID")
    if not wrapper_run_id:
        raise Issue137DeviceError("HEKATUS_TT_RUN_ID is required")
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_term = signal.signal(signal.SIGTERM, request_stop)
    old_int = signal.signal(signal.SIGINT, request_stop)
    prepared: Any = None
    try:
        # TTNN is imported only after every board-free check above.
        import ttnn

        device = ttnn.open_device(device_id=args.device_id, num_command_queues=2)
        try:
            prepared = prepare_newton_schulz_duration_load(
                ttnn,
                device,
                core_plan=config.core_plan,
                profile=True,
            )

            def launch_one() -> None:
                if stop_requested:
                    raise _StopRequested("Issue #137 stop requested")
                prepared.launch()
                # GenericOp is asynchronous; this synchronization is the
                # callback completion boundary required by the duration loop.
                ttnn.synchronize_device(device)

            try:
                duration = run_duration_bounded_load(
                    launch_one,
                    target_seconds=args.target_seconds,
                    monotonic_ns=time.monotonic_ns,
                    overrun_limit_seconds=args.overrun_limit_seconds,
                )
                result = {
                    "record_schema": issue137.ISSUE137_DEVICE_RESULT_SCHEMA,
                    **duration.as_record(),
                    "status": "ok" if duration.passed else "failed",
                    "run_id": wrapper_run_id,
                    "mode": args.mode,
                    "load_mode": issue137.ISSUE137_LOAD_MODE,
                    "target_seconds": args.target_seconds,
                    "guard_seconds": args.outer_timeout_seconds,
                    "overrun_limit_seconds": args.overrun_limit_seconds,
                    "core_plan": config.core_plan.as_record(),
                    "plan_sha256": args.plan_hash,
                    "environment": _safe_result_environment(environment),
                    "execution_provenance": {
                        "clock": "host.CLOCK_MONOTONIC",
                        "fixed_iterations": 12,
                        "matrix_block": 8,
                        "outer_repetitions_per_launch": 1,
                        "completion": "ttnn.synchronize_device after prepared.launch",
                        "preparation_before_first_timed_clock_read": True,
                        "jit_semantics": (
                            "preparation is outside the host interval; TTNN compilation "
                            "may occur on the first launch and is not separately timed"
                        ),
                    },
                }
            except (DurationLoadError, _StopRequested) as exc:
                partial = exc.result.as_record() if isinstance(exc, DurationLoadError) and exc.result else {}
                result = _failure_result(
                    args=args,
                    config=config,
                    environment=environment,
                    reason=str(exc),
                    run_id=wrapper_run_id,
                )
                result.update(partial)
        finally:
            if prepared is not None:
                prepared.close()
            ttnn.close_device(device)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    _write_result(result_path, result)
    print(f"issue137 result -> {result_path.name}")
    return 0 if result.get("status") == "ok" else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, TypeError) as exc:
        # A result is not fabricated when preflight fails: no device work was
        # started and the wrapper's exit status is the structured diagnostic.
        print(f"issue137-device: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
