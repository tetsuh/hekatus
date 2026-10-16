"""Construct the resident wrapper's Docker argv without touching a device.

This is a board-free static model of the resident branch in
``enodia/tt/bench/run_in_container.sh``.  It deliberately omits the wrapper's
``--device`` pair from the returned argv: the result is for inspection only and
must never be passed to Docker.  The omitted pair is reported separately by the
CLI so a manual probe can compare the hardware binding without starting one.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

RESIDENT_RUNNER = "enodia/tt/bench/run_resident.py"
DEFAULT_HUGEPAGES_SOURCE = "/dev/hugepages-1G"
DEFAULT_DEVICE_NODE = "/dev/tenstorrent/1"


def _rewrite_resident_argument(
    args: Sequence[str],
    index: int,
    *,
    telemetry_mode: str,
    power_trace_name: str,
) -> tuple[tuple[str, ...], int, bool]:
    argument = args[index]
    if argument == "--env-json":
        if index + 1 >= len(args):
            raise ValueError("--env-json requires a value")
        return (), index + 2, False
    if argument.startswith("--env-json="):
        return (), index + 1, False
    if argument == "--power-trace":
        if index + 1 >= len(args):
            raise ValueError("--power-trace requires a value")
        replacement = ("--power-trace", power_trace_name) if telemetry_mode != "off" else ()
        return replacement, index + 2, True
    if argument.startswith("--power-trace="):
        replacement = (f"--power-trace={power_trace_name}",) if telemetry_mode != "off" else ()
        return replacement, index + 1, True
    return (argument,), index + 1, False


def _resident_runner_args(
    runner_args: Sequence[str],
    *,
    telemetry_mode: str,
    environment_name: str,
    power_trace_name: str,
    runner_result_path: str,
) -> tuple[str, ...]:
    """Apply only the resident-branch argument rewrites from the shell wrapper."""
    args = list(runner_args)
    if "--out" not in args:
        args = ["--out", runner_result_path, *args]

    resident_args: list[str] = []
    has_power_trace = False
    index = 0
    while index < len(args):
        replacement, index, has_trace = _rewrite_resident_argument(
            args,
            index,
            telemetry_mode=telemetry_mode,
            power_trace_name=power_trace_name,
        )
        resident_args.extend(replacement)
        has_power_trace = has_power_trace or has_trace

    resident_args.extend(("--env-json", environment_name))
    if telemetry_mode != "off" and not has_power_trace:
        resident_args.extend(("--power-trace", power_trace_name))
    return tuple(resident_args)


def _validate_resident_options(
    run_id: str,
    container_name: str,
    telemetry_mode: str,
    device_node: str,
    container_timeout_seconds: int,
    telemetry_interval_seconds: str,
) -> None:
    if not run_id or any(character in run_id for character in "\r\n"):
        raise ValueError("run_id must be non-empty and contain no newline")
    if not container_name or any(character in container_name for character in "\r\n"):
        raise ValueError("container_name must be non-empty and contain no newline")
    if telemetry_mode not in {"off", "default", "explicit"}:
        raise ValueError("telemetry_mode must be off, default, or explicit")
    if not device_node or any(character in device_node for character in "\r\n"):
        raise ValueError("device_node must be non-empty and contain no newline")
    if container_timeout_seconds <= 0:
        raise ValueError("container_timeout_seconds must be positive")
    if telemetry_mode == "explicit" and not telemetry_interval_seconds:
        raise ValueError("explicit telemetry mode requires an interval")


def _resident_docker_prefix(
    *,
    container_name: str,
    hugepages_source: str,
    resolved_repo: Path,
    resolved_output: Path,
    run_id: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-v",
        f"{hugepages_source}:/dev/hugepages-1G",
        "-v",
        f"{resolved_repo}:/work",
        "-v",
        f"{resolved_output}:/out",
        "-w",
        "/work",
        "-e",
        "PYTHONPATH=/work",
        "-e",
        f"HEKATUS_TT_RUN_ID={run_id}",
    ]


def _append_resident_environment(
    argv: list[str],
    *,
    telemetry_mode: str,
    telemetry_interval_seconds: str,
    runner_result_path: str,
) -> None:
    if telemetry_mode != "off":
        argv.extend(("-e", f"HEKATUS_TT_TELEMETRY_MODE={telemetry_mode}"))
        argv.extend(("-e", f"HEKATUS_TT_TELEMETRY_INTERVAL_S={telemetry_interval_seconds}"))
    argv.extend(("-e", f"HEKATUS_TT_RESULT_PATH={runner_result_path}"))


def build_resident_docker_argv(
    *,
    repo_root: Path,
    output_dir: Path,
    image: str,
    run_id: str,
    container_name: str,
    container_timeout_seconds: int,
    runner_args: Sequence[str] = (),
    telemetry_mode: str = "default",
    telemetry_interval_seconds: str = "2",
    device_node: str = DEFAULT_DEVICE_NODE,
    hugepages_source: str = DEFAULT_HUGEPAGES_SOURCE,
    runner_result_path: str = "/out/runner-result.json",
) -> tuple[str, ...]:
    """Return the resident ``docker run`` argv, excluding ``--device``.

    ``run_id`` is the wrapper-generated id, not the caller's workload id.  The
    wrapper uses it for the sibling environment and power-trace basenames.
    ``output_dir`` is resolved because the shell wrapper normalizes the exact
    host bind source before invoking Docker.  ``device_node`` is accepted only
    so callers can report the omitted hardware argument alongside this vector.
    """
    _validate_resident_options(
        run_id,
        container_name,
        telemetry_mode,
        device_node,
        container_timeout_seconds,
        telemetry_interval_seconds,
    )
    resolved_repo = repo_root.resolve()
    resolved_output = output_dir.resolve()
    resident_args = _resident_runner_args(
        runner_args,
        telemetry_mode=telemetry_mode,
        environment_name=f"/out/env-{run_id}.json",
        power_trace_name=f"power-{run_id}.csv",
        runner_result_path=runner_result_path,
    )
    argv = _resident_docker_prefix(
        container_name=container_name,
        hugepages_source=hugepages_source,
        resolved_repo=resolved_repo,
        resolved_output=resolved_output,
        run_id=run_id,
    )
    _append_resident_environment(
        argv,
        telemetry_mode=telemetry_mode,
        telemetry_interval_seconds=telemetry_interval_seconds,
        runner_result_path=runner_result_path,
    )
    argv.extend(("--entrypoint", "python3", image, RESIDENT_RUNNER, *resident_args))
    return tuple(argv)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--container-name", default="hekatus-bench-static-probe")
    parser.add_argument("--container-timeout-seconds", type=int, default=330)
    parser.add_argument("--telemetry-mode", choices=("off", "default", "explicit"), default="default")
    parser.add_argument("--telemetry-interval-seconds", default="2")
    parser.add_argument("--device-node", default=DEFAULT_DEVICE_NODE)
    parser.add_argument("--runner-arg", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    vector = build_resident_docker_argv(
        repo_root=args.repo_root,
        output_dir=args.out_dir,
        image=args.image,
        run_id=args.run_id,
        container_name=args.container_name,
        container_timeout_seconds=args.container_timeout_seconds,
        runner_args=args.runner_arg,
        telemetry_mode=args.telemetry_mode,
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        device_node=args.device_node,
    )
    print(
        json.dumps(
            {
                "docker_argv_without_device": list(vector),
                "omitted_hardware_argv": ["--device", args.device_node],
                "started_container": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
